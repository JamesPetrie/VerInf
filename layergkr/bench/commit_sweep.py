"""Where does the 78x go? Hold C and N fixed, vary only the number of CUDA calls.

The factorial experiment (§4.9.5) fitted an end-to-end `C*N` coefficient of
0.469 ns/slot-position against 0.006 ns measured for `gl_matmul` in isolation. It
did NOT establish why. A regression gives one number; it does not decompose into
kernel launches, H2D/D2H, synchronisation, occupancy, matrix construction,
allocation, hashing or intermediate materialisation.

So this measures each stage separately, with CUDA events, while holding the total
work `C*N` CONSTANT and changing only how it is split:

    rows_total * ELL * N     fixed
    rows_per_call            swept: many small calls  ->  few large calls

against

    t = a + L*K_launch + b_transfer*B + c_kernel*C*N + t_hash

If `c_kernel` walks down toward 0.006 ns as the batch grows, the batching
hypothesis holds. If it settles somewhere else -- 0.05, 0.2 -- that value is the
real in-situ rate and it is what a projection must use. Either outcome is an
answer; the point is to stop asserting one.

Reporting is median and p95 after warm-up, not best-of-five: what is wanted is an
upper bound on time, and a best-of statistic is the wrong tail.

  .venv/bin/python layergkr/bench/commit_sweep.py
"""
import argparse
import json
import pathlib
import statistics
import sys
import time
from typing import Dict, List

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import torch

from layergkr import gpu, rs
from prover import cuda_primitives as cp
from prover.protocol import P

OUT = pathlib.Path(__file__).parent / "commit_sweep.jsonl"


class Stage:
    """CUDA-event timer for one stage. Events are recorded on the stream, so this
    measures device time without forcing a sync per stage."""

    def __init__(self):
        self.a = torch.cuda.Event(enable_timing=True)
        self.b = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        self.a.record()
        return self

    def __exit__(self, *exc):
        self.b.record()

    def ms(self) -> float:
        torch.cuda.synchronize()
        return self.a.elapsed_time(self.b)


def one_pass(cfg: rs.Config, rows_total: int, rows_per_call: int,
             msg_host: List[List[int]]) -> Dict[str, float]:
    """Encode + commit `rows_total` rows in calls of `rows_per_call`, timing each
    stage. Host-side prep is wall-clocked (it is not on the device at all)."""
    L = gpu.lagrange_gpu(cfg)
    n_calls = (rows_total + rows_per_call - 1) // rows_per_call
    t = {"prep_ms": 0.0, "h2d_ms": 0.0, "matmul_ms": 0.0, "hash_ms": 0.0,
         "d2h_ms": 0.0, "calls": n_calls}

    for k in range(n_calls):
        chunk = msg_host[k * rows_per_call:(k + 1) * rows_per_call]
        if not chunk:
            break

        t0 = time.perf_counter()                       # host-side prep
        flat = chunk
        t["prep_ms"] += (time.perf_counter() - t0) * 1e3

        s_h2d = Stage()
        with s_h2d:
            A = torch.tensor(flat, dtype=torch.uint64, device="cuda")
        s_mm = Stage()
        with s_mm:
            C = cp.gl_matmul(A, L)
        s_hash = Stage()
        with s_hash:
            dig = cp.hash_columns_streamed(C.contiguous())
            root_t, _ = cp.merkle_build_blake3(dig)
        s_d2h = Stage()
        with s_d2h:
            _ = root_t.cpu()

        t["h2d_ms"] += s_h2d.ms()
        t["matmul_ms"] += s_mm.ms()
        t["hash_ms"] += s_hash.ms()
        t["d2h_ms"] += s_d2h.ms()
    return t


def measure(cfg: rs.Config, rows_total: int, rows_per_call: int,
            reps: int = 5, warmup: int = 1) -> Dict[str, float]:
    import random
    r = random.Random(3)
    msg = [[r.randrange(P) for _ in range(cfg.ELL)] for _ in range(rows_total)]

    for _ in range(warmup):
        one_pass(cfg, rows_total, rows_per_call, msg)

    runs = []
    for _ in range(reps):
        t0 = time.perf_counter()
        t = one_pass(cfg, rows_total, rows_per_call, msg)
        torch.cuda.synchronize()
        t["wall_ms"] = (time.perf_counter() - t0) * 1e3
        runs.append(t)

    keys = [k for k in runs[0] if k != "calls"]
    out = {"calls": runs[0]["calls"]}
    for k in keys:
        vals = sorted(x[k] for x in runs)
        out[k + "_med"] = statistics.median(vals)
        out[k + "_p95"] = vals[min(len(vals) - 1, int(0.95 * len(vals)))]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ell", type=int, default=1024)
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--rows", type=int, default=2048)
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()

    if not gpu.available():
        print("no CUDA — nothing to measure")
        return 1

    cfg = rs.Config(ELL=args.ell, K_DEG=2 * args.ell, N_LIG=args.n, T_QUERIES=16)
    t0 = time.perf_counter()
    rs.lagrange_matrix(cfg)
    gpu.lagrange_gpu(cfg)
    print(f"setup {time.perf_counter() - t0:.1f}s   ELL={cfg.ELL} N={cfg.N_LIG} "
          f"rows={args.rows}  (total C*N held fixed at "
          f"{args.rows * cfg.ELL * cfg.N_LIG:,} slot-positions)\n")

    work = args.rows * cfg.ELL * cfg.N_LIG
    print(f"{'rows/call':>9} {'calls':>6} {'wall ms':>9} {'p95':>8} "
          f"{'prep':>7} {'h2d':>7} {'matmul':>8} {'hash':>8} {'d2h':>7} "
          f"{'ns/slot-pos':>12} {'kernel only':>12}")
    rows = []
    for rpc in (8, 16, 32, 64, 128, 256, 512, 1024, 2048):
        if rpc > args.rows:
            break
        m = measure(cfg, args.rows, rpc, reps=args.reps)
        ns_total = m["wall_ms_med"] * 1e6 / work
        ns_kernel = m["matmul_ms_med"] * 1e6 / work
        rows.append({"rows_per_call": rpc, "ell": cfg.ELL, "n": cfg.N_LIG,
                     "rows_total": args.rows, "work": work,
                     "ns_per_slotpos": ns_total, "ns_kernel": ns_kernel, **m})
        print(f"{rpc:>9} {m['calls']:>6} {m['wall_ms_med']:>9.1f} "
              f"{m['wall_ms_p95']:>8.1f} {m['prep_ms_med']:>7.1f} "
              f"{m['h2d_ms_med']:>7.1f} {m['matmul_ms_med']:>8.1f} "
              f"{m['hash_ms_med']:>8.1f} {m['d2h_ms_med']:>7.1f} "
              f"{ns_total:>12.4f} {ns_kernel:>12.4f}")

    print("\n'ns/slot-pos' is the end-to-end cost the factorial fit measured "
          "(0.469 there).")
    print("'kernel only' is gl_matmul's device time alone. If the first walks down")
    print("toward the second as rows/call grows, batching is the answer. Where it")
    print("STOPS is the real in-situ rate, and that is the number to project with.")
    with open(OUT, "a") as fh:
        for r in rows:
            fh.write(json.dumps({**r, "ts": int(time.time())}) + "\n")
    print(f"\nappended to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
