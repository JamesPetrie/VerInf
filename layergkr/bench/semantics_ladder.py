"""Where the layer's time goes, now that the forward pass is on the device.

Three questions this answers, and it exists because each of them had been asked
more than once:

  1. How much faster is the tensor forward than the Python one, at matched sizes
     and matched weights?
  2. How wide can the forward pass go, and WHAT stops it -- time, memory, or the
     2^63 range wall the int64 representation needs?
  3. Given a fast forward pass, what is now the wall for a whole proof?

(3) is the one that matters. A ladder that only reported the forward pass would
invite the reading that a production-width proof is close, and it is not.

Every row reports its RANGE HEADROOM: log2(2^63 / largest accumulator bound seen).
A guard that never fires proves nothing, so the number is printed whether or not
it is comfortable.

Weights are drawn on the DEVICE. Drawing them in Python became the largest single
cost in the run once the forward pass moved (19.6 s of `random.randrange` against
4.5 s of everything else at d=1024), so a ladder that kept `LayerWeights.draw`
would mostly have been measuring the standard library.

  .venv/bin/python layergkr/bench/semantics_ladder.py            # forward only
  .venv/bin/python layergkr/bench/semantics_ladder.py --pipeline # + prove/verify
"""
import argparse
import json
import pathlib
import random
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import torch

from layergkr import full_layer as fl, rs, semantics as sem

OUT = pathlib.Path(__file__).parent / "semantics_ladder.jsonl"

# (d, S, E) -- d_ff is 2d throughout, as everywhere else in this prototype.
# The Python reference is only run where it finishes in reasonable time; past
# that the column is left empty rather than extrapolated.
FORWARD = [(128, 8, 4), (256, 16, 4), (384, 16, 4), (512, 32, 4),
           (1024, 32, 8), (2048, 64, 8), (4096, 64, 8)]
PY_LIMIT = 512

PIPELINE = [(128, 8, 4), (256, 8, 4), (512, 8, 4)]


def _warm() -> None:
    """One tiny layer, discarded. First-call cost (CUDA context, kernel JIT) is
    what three wrong explanations of the 10.9% residual turned out to be."""
    c = sem.ToyConfig(S=4, d=8, d_ff=16, E=2)
    sem.forward_tensor(c, None, weights=sem.LayerWeights.draw_tensor(c, 1))


def forward_ladder(seed: int) -> list:
    rows = []
    print(f"{'d':>6}{'d_ff':>7}{'S':>5}{'E':>4}{'python':>10}{'tensor':>9}"
          f"{'speedup':>9}{'peakGB':>8}{'headroom':>10}  note")
    for d, S, E in FORWARD:
        cfg = sem.ToyConfig(S=S, d=d, d_ff=2 * d, E=E, table_bits=6, scale_bits=6)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        note, t_py, t_t, head = "", float("nan"), float("nan"), float("nan")
        try:
            w = sem.LayerWeights.draw_tensor(cfg, seed)
            if d <= PY_LIMIT:
                lists = w.to_lists()
                t0 = time.perf_counter()
                sem.forward(cfg, None, weights=lists)
                t_py = time.perf_counter() - t0
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            trace = sem.forward_tensor(cfg, None, weights=w)
            torch.cuda.synchronize()
            t_t = time.perf_counter() - t0
            head = trace.headroom_bits
            counts = trace.counts()
            del trace, w
        except sem.RangeOverflow as e:
            note, counts = "RANGE WALL: " + str(e).split(",")[0][:44], {}
        except torch.cuda.OutOfMemoryError:
            note, counts = "CUDA OOM", {}
        peak = torch.cuda.max_memory_allocated() / 1e9
        sp = t_py / t_t if t_py == t_py and t_t == t_t else float("nan")
        print(f"{d:>6}{2*d:>7}{S:>5}{E:>4}{t_py:>10.2f}{t_t:>9.2f}{sp:>8.1f}x"
              f"{peak:>8.2f}{head:>10.0f}  {note}", flush=True)
        rows.append({"kind": "forward", "d": d, "d_ff": 2 * d, "S": S, "E": E,
                     "t_python_s": t_py, "t_tensor_s": t_t, "peak_gpu_gb": peak,
                     "headroom_bits": head, "note": note, "counts": counts,
                     "ts": int(time.time())})
    return rows


def pipeline_ladder(seed: int, ell: int, n: int, q: int) -> list:
    """The honest framing: the forward pass is no longer the cost. Everything
    downstream still consumes Python objects, so `to_python` and the prover are
    what a production-width run now runs into."""
    cfg_rs = rs.Config(ELL=ell, K_DEG=2 * ell, N_LIG=n, T_QUERIES=q)
    rs.lagrange_matrix(cfg_rs)
    rows = []
    print(f"\nwhole pipeline at ELL={ell} N={n} q={q}")
    print(f"{'d':>6}{'S':>5}{'E':>4}{'forward':>9}{'to_python':>11}{'prove':>9}"
          f"{'verify':>9}{'fwd share':>11}  ok")
    for d, S, E in PIPELINE:
        cfg = sem.ToyConfig(S=S, d=d, d_ff=2 * d, E=E, table_bits=6, scale_bits=6)
        w = sem.LayerWeights.draw_tensor(cfg, seed)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        dev = sem.forward_tensor(cfg, None, weights=w)
        torch.cuda.synchronize()
        t_f = time.perf_counter() - t0
        t0 = time.perf_counter()
        trace = sem.to_python(dev)
        t_c = time.perf_counter() - t0
        t0 = time.perf_counter()
        proof = fl.prove_full_layer(trace, cfg_rs, fl.Enrollment(cfg_rs), q,
                                    random.Random(seed))
        t_p = time.perf_counter() - t0
        t0 = time.perf_counter()
        ok, why = fl.verify_full_layer(cfg_rs, proof, trace.gates)
        t_v = time.perf_counter() - t0
        total = t_f + t_c + t_p + t_v
        print(f"{d:>6}{S:>5}{E:>4}{t_f:>9.2f}{t_c:>11.2f}{t_p:>9.2f}{t_v:>9.2f}"
              f"{100*t_f/total:>10.2f}%  {ok}{'' if ok else ' ' + why}", flush=True)
        rows.append({"kind": "pipeline", "d": d, "S": S, "E": E, "ell": ell,
                     "n": n, "q": q, "t_forward_s": t_f, "t_to_python_s": t_c,
                     "t_prove_s": t_p, "t_verify_s": t_v, "verified": ok,
                     "ts": int(time.time())})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--pipeline", action="store_true",
                    help="also time prove/verify, which is where the wall now is")
    ap.add_argument("--ell", type=int, default=1024)
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--q", type=int, default=16)
    args = ap.parse_args()

    _warm()
    rows = forward_ladder(args.seed)
    if args.pipeline:
        rows += pipeline_ladder(args.seed, args.ell, args.n, args.q)
    with open(OUT, "a") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"\n{len(rows)} rows appended to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
