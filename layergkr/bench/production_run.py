"""ONE run at PRODUCTION Ligero geometry. Not a sweep.

ELL=8192, K_DEG=16384, N_LIG=65536, q=54 — the parameters the theorem is written
for, and the axis every previous measurement held constant. Per row of message
capacity this is ~512x everything measured so far, and it is the term the cost
model has no data on.

The model width is a separate matter and is NOT production here. The forward pass
in `semantics.py` is plain Python: at d=5120, S=1000, E=128 it is ~8e10
multiply-accumulates for the dense part plus ~1.3e11 for the MoE projections, i.e.
hours before any proving starts. That is a property of this prototype's semantics
layer, not of the protocol or the prover. So width is set to whatever completes,
and the run reports it plainly rather than implying a full Maverick layer.

What comes out: the staged breakdown for the prover AND the verifier at production
geometry, with the `unattributed` line, the op counters, and the peak device
memory. Warm-up is impossible at this size (one build of the Lagrange matrix is
~14 minutes and ~28 GB of host RAM), so first-call cost is INSIDE these numbers
and is called out rather than hidden.

  .venv/bin/python layergkr/bench/production_run.py --d 512 --s 32 --e 4
"""
import argparse
import json
import pathlib
import random
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import torch

from layergkr import full_layer as fl, profile, rs, semantics as sem
from layergkr.counters import Counter

OUT = pathlib.Path(__file__).parent / "production_run.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ell", type=int, default=8192)
    ap.add_argument("--n", type=int, default=65536)
    ap.add_argument("--q", type=int, default=54)
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--s", type=int, default=32)
    ap.add_argument("--e", type=int, default=4)
    args = ap.parse_args()

    cfg = rs.Config(ELL=args.ell, K_DEG=2 * args.ell, N_LIG=args.n,
                    T_QUERIES=args.q)
    print(f"PRODUCTION GEOMETRY  ELL={cfg.ELL} K={cfg.K_DEG} N={cfg.N_LIG} "
          f"q={cfg.T_QUERIES}")
    print(f"model width          d={args.d} d_ff={2*args.d} S={args.s} E={args.e}"
          f"   (NOT production width — the Python forward pass is the limit)\n")

    # No Lagrange matrix. The NTT encoder made it unnecessary (§4.9.8): it was
    # 537M cells, ~13 min and ~30 GB of host RAM for a result that is reproduced
    # bit-for-bit by iNTT_K -> coset scale -> NTT_N.
    t_lag = 0.0

    toy = sem.ToyConfig(S=args.s, d=args.d, d_ff=2 * args.d, E=args.e,
                        table_bits=6, scale_bits=6)
    t0 = time.perf_counter()
    dev_trace = sem.forward_tensor(toy, None,
                                   weights=sem.LayerWeights.draw_tensor(toy, 7))
    torch.cuda.synchronize()
    t_fwd = time.perf_counter() - t0
    print(f"[{t_fwd:6.2f} s ] forward pass on device, "
          f"range headroom {dev_trace.headroom_bits:.0f} bits", flush=True)

    # The prover still consumes Python objects, so the trace is materialised here
    # and the cost of doing so is reported rather than hidden inside "prove".
    t0 = time.perf_counter()
    trace = sem.to_python(dev_trace)
    t_mat = time.perf_counter() - t0
    del dev_trace
    ok, why = sem.check_trace(trace)
    if not ok:
        raise AssertionError(f"trace inconsistent: {why}")
    c = trace.counts()
    print(f"[{t_mat:6.2f} s ] materialised to Python objects", flush=True)
    print(f"           trace: {c['matmuls']} matmuls, {c['moe_nodes']} MoE nodes, "
          f"{c['matmul_cells']:,} + {c['moe_cells']:,} cells, "
          f"{c['gates']} gates, {c['lookup_queries']:,} lookups", flush=True)
    # LogUp commits one RS row per lookup QUERY, so this is the row count that
    # decides whether the run fits in memory at all: rows * N_LIG * 8 bytes.
    print(f"           LogUp rows {c['lookup_queries'] + c['lookup_table_rows']:,}"
          f" -> {(c['lookup_queries'] + c['lookup_table_rows']) * cfg.N_LIG * 8 / 1e9:.1f}"
          f" GB of codeword at N={cfg.N_LIG}", flush=True)

    torch.cuda.reset_peak_memory_stats()
    enrol = fl.Enrollment(cfg)
    t0 = time.perf_counter()
    with profile.no_gc():
        with profile.timeline("prove") as tlp, Counter("prove") as cp:
            proof = fl.prove_full_layer(trace, cfg, enrol, cfg.T_QUERIES,
                                        random.Random(7))
    t_prove = time.perf_counter() - t0
    peak_prove = torch.cuda.max_memory_allocated() / 1e9
    print(f"[{t_prove/60:5.1f} min] PROVE done, peak GPU {peak_prove:.2f} GB\n",
          flush=True)
    print(tlp.report(min_share=0.5))

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    with profile.no_gc():
        with profile.timeline("verify") as tlv, Counter("verify") as cv:
            vok, vwhy = fl.verify_full_layer(cfg, proof, trace.gates)
    t_verify = time.perf_counter() - t0
    peak_verify = torch.cuda.max_memory_allocated() / 1e9
    print(f"\n[{t_verify/60:5.1f} min] VERIFY -> {vok} ({vwhy}), "
          f"peak GPU {peak_verify:.2f} GB\n", flush=True)
    print(tlv.report(min_share=0.5))

    # The question this run exists to answer: of the prove wall time, how much is
    # device work and how much is the Python side? Stage device_ms comes from CUDA
    # events, so this is measured, not inferred.
    dev_ms = sum(s.device_ms for s in tlp.spans if s.depth == 0)
    print(f"\nPROVE SPLIT   wall {t_prove:.1f} s, of which device {dev_ms/1000:.1f} s "
          f"({100*dev_ms/1000/max(t_prove,1e-9):.1f}%) -- the rest is the Python side")

    row = {"ell": cfg.ELL, "k": cfg.K_DEG, "n": cfg.N_LIG, "q": cfg.T_QUERIES,
           "d": args.d, "S": args.s, "E": args.e,
           "t_lagrange_s": t_lag, "t_materialise_s": t_mat,
           "prove_device_ms": dev_ms, "t_forward_s": t_fwd,
           "t_prove_s": t_prove, "t_verify_s": t_verify, "verified": vok,
           "peak_gpu_prove_gb": peak_prove, "peak_gpu_verify_gb": peak_verify,
           "trace_counts": c, "prove_counts": cp.report(),
           "verify_counts": cv.report(),
           "prove_stages": {s.name: {"depth": s.depth, "wall": s.wall_s,
                                     "dev_ms": s.device_ms, "n": s.n}
                            for s in tlp.spans},
           "verify_stages": {s.name: {"depth": s.depth, "wall": s.wall_s,
                                      "dev_ms": s.device_ms, "n": s.n}
                             for s in tlv.spans},
           "ts": int(time.time())}
    with open(OUT, "a") as fh:
        fh.write(json.dumps(row) + "\n")

    print(f"\nSUMMARY  prove {t_prove/60:.1f} min   verify {t_verify/60:.1f} min   "
          f"verified={vok}")
    print(f"         forward {t_fwd:.2f} s + materialise {t_mat:.2f} s "
          f"(no Lagrange build: the NTT encoder removed it)")
    print(f"         appended to {OUT}")
    return 0 if vok else 1


if __name__ == "__main__":
    sys.exit(main())
