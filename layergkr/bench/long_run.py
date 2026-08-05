"""The long run: prover AND verifier, fully instrumented, both axes.

Everything learned the hard way is baked into how this measures:

  * a warm-up proof is run and DISCARDED at every configuration. The 10.9%
    residual that took three wrong explanations to chase down was first-call cost
    (CUDA context, kernel JIT, the initial Lagrange upload) landing between stages.
  * the cycle collector is off during the timed proof (`profile.no_gc()`), worth
    ~9% by order-controlled measurement.
  * every point reports the per-stage breakdown WITH its `unattributed` line, for
    the prover and the verifier separately. A point whose residual exceeds 10% is
    flagged in the output and must not be reasoned from.
  * median and p95 over repeats, not best-of: the question is an upper bound.
  * counters (field ops, hashed bytes, opened values) are recorded alongside the
    times, so the count model can be re-validated from the same rows.

Two axes, swept separately, because conflating them is what produced the earlier
false "linear in d" reading:

  width     d at fixed Ligero geometry
  geometry  (ELL, N) at fixed d, and independently of each other

  .venv/bin/python layergkr/bench/long_run.py --hours 4
"""
import argparse
import json
import pathlib
import random
import statistics
import sys
import time
from typing import Dict, List

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from layergkr import full_layer as fl, profile, rs, semantics as sem
from layergkr.counters import Counter

OUT = pathlib.Path(__file__).parent / "long_run.jsonl"


def _spec(name, S, d, d_ff, E, ELL, N, q=16):
    return dict(name=name, S=S, d=d, d_ff=d_ff, E=E, ELL=ELL, K=2 * ELL, N=N, q=q)


PLAN: List[dict] = [
    # width at fixed geometry (ELL=1024, N=4096)
    _spec("w-d128", 8, 128, 256, 4, 1024, 4096),
    _spec("w-d256", 8, 256, 512, 4, 1024, 4096),
    _spec("w-d512", 8, 512, 1024, 4, 1024, 4096),
    _spec("w-d768", 8, 768, 1536, 4, 1024, 4096),
    # ELL at fixed N=4096, fixed d=128
    _spec("g-E512", 8, 128, 256, 4, 512, 4096),
    _spec("g-E2048", 8, 128, 256, 4, 2048, 4096),
    _spec("g-E4096", 8, 128, 256, 4, 4096, 4096),
    # N at fixed ELL=1024, fixed d=128
    _spec("g-N2048", 8, 128, 256, 4, 1024, 2048),
    _spec("g-N8192", 8, 128, 256, 4, 1024, 8192),
    _spec("g-N16384", 8, 128, 256, 4, 1024, 16384),
    # joint, toward production geometry
    _spec("j-E2048N8192", 8, 128, 256, 4, 2048, 8192),
    _spec("j-E4096N16384", 8, 128, 256, 4, 4096, 16384),
    # sequence length, the other quadratic (dense attention is Theta(S^2 d))
    _spec("s-S16", 16, 128, 256, 4, 1024, 4096),
    _spec("s-S32", 32, 128, 256, 4, 1024, 4096),
]


def capacity(trace, ell: int) -> int:
    total = 0
    for m in trace.matmuls:
        total += len(m.W) * ell * -(-len(m.W[0]) // ell)
        total += len(m.X) * ell * -(-len(m.X[0]) // ell)
    for m in trace.moe:
        total += len(m.W) * len(m.W[0]) * ell * -(-len(m.W[0][0]) // ell)
    return total


def one(spec: dict, reps: int, seed: int = 7) -> dict:
    cfg = rs.Config(ELL=spec["ELL"], K_DEG=spec["K"], N_LIG=spec["N"],
                    T_QUERIES=spec["q"])
    t0 = time.perf_counter()
    rs.lagrange_matrix(cfg)
    t_setup = time.perf_counter() - t0

    toy = sem.ToyConfig(S=spec["S"], d=spec["d"], d_ff=spec["d_ff"], E=spec["E"],
                        table_bits=6, scale_bits=6)
    trace = sem.forward(toy, random.Random(seed))
    ok, why = sem.check_trace(trace)
    if not ok:
        raise AssertionError(f"{spec['name']}: trace inconsistent: {why}")

    # WARM-UP, discarded: first-call cost is what the residual chase turned out
    # to be, and it does not belong in a steady-state measurement.
    warm_en = fl.Enrollment(cfg)
    warm = fl.prove_full_layer(trace, cfg, warm_en, spec["q"], random.Random(seed))
    fl.verify_full_layer(cfg, warm, trace.gates)

    p_times, v_times, p_rep, v_rep, counts = [], [], None, None, None
    for _ in range(reps):
        en = fl.Enrollment(cfg)
        rng = random.Random(seed)
        with profile.no_gc():
            with profile.timeline("prove") as tlp, Counter("prove") as c:
                proof = fl.prove_full_layer(trace, cfg, en, spec["q"], rng)
            with profile.timeline("verify") as tlv:
                vok, vwhy = fl.verify_full_layer(cfg, proof, trace.gates)
        if not vok:
            raise AssertionError(f"{spec['name']}: did not verify: {vwhy}")
        p_times.append(tlp.wall_s)
        v_times.append(tlv.wall_s)
        p_rep, v_rep, counts = tlp, tlv, c.report()

    def stat(xs):
        xs = sorted(xs)
        return (statistics.median(xs), xs[min(len(xs) - 1, int(0.95 * len(xs)))])

    p_med, p_p95 = stat(p_times)
    v_med, v_p95 = stat(v_times)
    resid_p = 1 - sum(s.wall_s for s in p_rep.spans if s.depth == 0) / p_rep.wall_s
    resid_v = 1 - sum(s.wall_s for s in v_rep.spans if s.depth == 0) / v_rep.wall_s
    return {
        "name": spec["name"], "spec": spec, "capacity": capacity(trace, cfg.ELL),
        "t_setup_s": t_setup, "prove_med": p_med, "prove_p95": p_p95,
        "verify_med": v_med, "verify_p95": v_p95,
        "resid_prove": resid_p, "resid_verify": resid_v,
        "prove_stages": {s.name: {"depth": s.depth, "wall": s.wall_s,
                                  "dev_ms": s.device_ms, "n": s.n}
                         for s in p_rep.spans},
        "verify_stages": {s.name: {"depth": s.depth, "wall": s.wall_s,
                                   "dev_ms": s.device_ms, "n": s.n}
                          for s in v_rep.spans},
        "counts": counts, "reps": reps, "ts": int(time.time()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hours", type=float, default=4.0)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    deadline = time.time() + args.hours * 3600
    rows: List[dict] = []
    print(f"budget {args.hours}h, {len(PLAN)} configurations, {args.reps} reps each, "
          f"warm-up discarded, GC off during timing\n")
    for spec in PLAN:
        if time.time() > deadline:
            print(f"  [budget spent, stopping before {spec['name']}]")
            break
        t0 = time.time()
        try:
            r = one(spec, args.reps)
        except Exception as e:
            print(f"  {spec['name']:<15} FAILED {type(e).__name__}: {str(e)[:90]}",
                  flush=True)
            continue
        rows.append(r)
        with open(OUT, "a") as fh:                 # append as we go, not at the end
            fh.write(json.dumps(r) + "\n")
        flag = ""
        if max(r["resid_prove"], r["resid_verify"]) > 0.10:
            flag = "  !! residual >10%, do not reason from this point"
        print(f"  {spec['name']:<15} setup {r['t_setup_s']:6.1f}s  "
              f"prove {r['prove_med']:7.1f}s (p95 {r['prove_p95']:7.1f})  "
              f"verify {r['verify_med']:6.1f}s (p95 {r['verify_p95']:6.1f})  "
              f"resid p/v {100*r['resid_prove']:4.1f}/{100*r['resid_verify']:4.1f}%"
              f"{flag}  [{time.time()-t0:.0f}s]", flush=True)

    print(f"\n{'run':<15}{'d':>5}{'S':>4}{'ELL':>6}{'N':>7}{'capacity':>13}"
          f"{'prove':>9}{'verify':>8}{'v/p':>7}")
    for r in rows:
        sp = r["spec"]
        print(f"{r['name']:<15}{sp['d']:>5}{sp['S']:>4}{sp['ELL']:>6}{sp['N']:>7}"
              f"{r['capacity']:>13,}{r['prove_med']:>9.1f}{r['verify_med']:>8.1f}"
              f"{r['verify_med']/r['prove_med']:>7.2f}")
    print(f"\n{len(rows)} points appended to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
