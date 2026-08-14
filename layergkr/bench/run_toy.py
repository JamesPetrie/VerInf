"""Prove real toy/small layers, measure everything, and CHECK THE MODEL.

This is the run the previous version of this work could not do, because there was
nothing to run. Each configuration is proved and verified end to end; the
counters record what actually happened; `count_model` predicts what should have
happened; the two are compared per phase.

  .venv/bin/python layergkr/bench/run_toy.py                # toy + small (~minutes)
  .venv/bin/python layergkr/bench/run_toy.py --set toy      # seconds
  .venv/bin/python layergkr/bench/run_toy.py --set large    # ~10 min

Output: a table of predicted-vs-measured counts, and one JSONL row per run in
`layergkr/bench/runs.jsonl` (same convention as analysis/bench/prove_runs.jsonl).

The pass criterion is deliberately strict: the model predicts field-op counts to
within 1%, because counts are arithmetic, not an estimate. Anything worse is a
missing term, and the run prints which phase it is in.
"""
import argparse
import json
import pathlib
import random
import sys
import time
from typing import Dict, List

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from layergkr import count_model as cm, full_layer as fl, rs, semantics as sem
from layergkr.counters import Counter, Rates

RUNS = pathlib.Path(__file__).parent / "runs.jsonl"
RATES_FILE = pathlib.Path(__file__).parent / "rates.json"

SETS: Dict[str, List[dict]] = {
    "toy": [
        dict(name="toy-xs", S=2, d=4, d_ff=8, E=1, ELL=8, K=16, N=32, q=4),
        dict(name="toy-s", S=4, d=8, d_ff=16, E=1, ELL=16, K=32, N=64, q=4),
        dict(name="toy-moe", S=4, d=8, d_ff=16, E=2, ELL=16, K=32, N=64, q=4),
    ],
    "small": [
        dict(name="small-1", S=6, d=16, d_ff=32, E=2, ELL=32, K=64, N=128, q=8),
        dict(name="small-2", S=8, d=16, d_ff=32, E=4, ELL=32, K=64, N=128, q=8),
    ],
    "large": [
        dict(name="large-1", S=8, d=32, d_ff=64, E=4, ELL=64, K=128, N=256, q=12),
        dict(name="large-2", S=12, d=32, d_ff=64, E=8, ELL=64, K=128, N=256, q=12),
    ],
    # Widths and Ligero geometries approaching a real layer. These need the GPU
    # backend (gpu.py) to finish in minutes rather than hours; on CPU only they
    # still run, just slowly. d=128 is one Maverick attention head group; ELL=256
    # with N=1024 keeps the rate K/N = 1/4 of the production geometry.
    "real": [
        dict(name="real-64", S=8, d=64, d_ff=128, E=4, ELL=128, K=256, N=512, q=12),
        dict(name="real-128", S=16, d=128, d_ff=256, E=8, ELL=256, K=512, N=1024, q=16),
    ],
    # d=256 with ELL=512/N=2048 runs, but takes ~45 min on this box -- kept out of
    # the default tier so a validation pass stays a coffee break.
    "real-xl": [
        dict(name="real-256", S=16, d=256, d_ff=512, E=16, ELL=512, K=1024, N=2048, q=20),
    ],
}


def load_rates() -> Rates:
    if RATES_FILE.exists():
        cards = json.loads(RATES_FILE.read_text())
        c = cards[0]
        return Rates(c["name"], c["mul_ns"], c["mul_defer_ns"], c["add_ns"],
                     c["hash_GBps"])
    return Rates("unmeasured", 100.0, 60.0, 60.0, 5.0)


def rel_err(pred: int, meas: int) -> float:
    if meas == 0:
        return 0.0 if pred == 0 else float("inf")
    return (pred - meas) / meas


def run_one(spec: dict, rates: Rates, seed: int = 7) -> dict:
    cfg = rs.Config(ELL=spec["ELL"], K_DEG=spec["K"], N_LIG=spec["N"],
                    T_QUERIES=spec["q"])
    toy = sem.ToyConfig(S=spec["S"], d=spec["d"], d_ff=spec["d_ff"], E=spec["E"],
                        table_bits=6, scale_bits=6)
    rng = random.Random(seed)

    t0 = time.perf_counter()
    trace = sem.forward(toy, rng)
    t_fwd = time.perf_counter() - t0
    ok, why = sem.check_trace(trace)
    if not ok:
        raise AssertionError(f"{spec['name']}: trace is not self-consistent: {why}")

    enrol = fl.Enrollment(cfg)
    t0 = time.perf_counter()
    with Counter("prove") as cp:
        proof = fl.prove_full_layer(trace, cfg, enrol, spec["q"], rng, use_masks=True)
    t_prove = time.perf_counter() - t0

    t0 = time.perf_counter()
    with Counter("verify") as cv:
        vok, vwhy = fl.verify_full_layer(cfg, proof, trace.gates)
    t_verify = time.perf_counter() - t0
    if not vok:
        raise AssertionError(f"{spec['name']}: proof did not verify: {vwhy}")

    pred = cm.predict_layer(trace, cfg, spec["q"], len(enrol.weights))
    predv = cm.predict_verify(trace, cfg, spec["q"])
    meas = {k: v for k, v in cp.flat().items()}
    measv = {k: v for k, v in cv.flat().items()}

    shape = cm.predict_trace_shape(toy)
    actual_shape = trace.counts()

    return {
        "name": spec["name"], "spec": spec,
        "t_forward_s": t_fwd, "t_prove_s": t_prove, "t_verify_s": t_verify,
        "prove_counts": cp.report(), "verify_counts": cv.report(),
        "prove_phases": {k: v for k, v in meas.items() if k != "prove"},
        "verify_phases": {k: v for k, v in measv.items() if k != "verify"},
        "pred_prove": pred["TOTAL"], "pred_verify": predv["TOTAL"],
        "pred_phases": {k: v for k, v in pred.items() if k != "TOTAL"},
        "pred_verify_phases": {k: v for k, v in predv.items() if k != "TOTAL"},
        "trace_shape_pred": shape.__dict__, "trace_shape_actual": actual_shape,
        "modelled_prove_s": rates.seconds(pred["TOTAL"]),
        "modelled_verify_s": rates.seconds(predv["TOTAL"]),
        "rates": rates.name,
        "n_enrolled": len(enrol.weights),
        "n_matmuls": len(trace.matmuls), "n_gates": len(trace.gates),
        "proof_bytes": cp.report()["proof_bytes"],
    }


def report(rows: List[dict]) -> int:
    print(f"\n{'run':<10} {'prove s':>8} {'verify s':>9} {'mul (meas)':>13} "
          f"{'mul (pred)':>13} {'err':>8} {'hash MB':>8}")
    worst = 0.0
    for r in rows:
        m, p = r["prove_counts"]["mul"], r["pred_prove"]["mul"]
        e = rel_err(p, m)
        worst = max(worst, abs(e))
        print(f"{r['name']:<10} {r['t_prove_s']:8.2f} {r['t_verify_s']:9.2f} "
              f"{m:13,} {p:13,} {100*e:+7.2f}% "
              f"{r['prove_counts']['hash_bytes']/1e6:8.2f}")

    print(f"\nTIME MODEL QUALITY (counts are exact, so this isolates the RATE card)")
    print(f"{'run':<10} {'meas s':>9} {'modelled s':>11} {'ratio':>7}")
    ratios = []
    for r in rows:
        ratio = r["modelled_prove_s"] / max(r["t_prove_s"], 1e-9)
        ratios.append(ratio)
        print(f"{r['name']:<10} {r['t_prove_s']:9.2f} {r['modelled_prove_s']:11.2f} "
              f"{ratio:7.2f}x")
    if ratios:
        lo, hi = min(ratios), max(ratios)
        print(f"  spread {lo:.2f}x .. {hi:.2f}x. The counts are exact, so this is the "
              f"RATE CARD's fidelity")
        print(f"  alone. The first version of this table read 1.08x..4.14x; splitting "
              f"reduced from")
        print(f"  deferred multiplies (the hot loops accumulate a*b and reduce once) "
              f"halved it. The")
        print(f"  residual drifts with size because CPython's per-op cost grows with "
              f"operand width --")
        print(f"  a property of THIS prototype, not of the protocol, and it does not "
              f"transfer to a")
        print(f"  GPU projection. Note what did NOT happen: no factor was introduced "
              f"to close the gap.")

    print(f"\nVERIFIER (the doc models none of this)")
    print(f"{'run':<10} {'mul (meas)':>13} {'mul (pred)':>13} {'err':>8} "
          f"{'v/p mul':>9} {'v/p time':>9}")
    for r in rows:
        m, p = r["verify_counts"]["mul"], r["pred_verify"]["mul"]
        ratio_c = m / max(r["prove_counts"]["mul"], 1)
        ratio_t = r["t_verify_s"] / max(r["t_prove_s"], 1e-9)
        print(f"{r['name']:<10} {m:13,} {p:13,} {100*rel_err(p, m):+7.2f}% "
              f"{ratio_c:9.3f} {ratio_t:9.3f}")

    print(f"\nPER-PHASE prediction error (prover, field muls)")
    phases = sorted({k for r in rows for k in r["pred_phases"]})
    print(f"{'phase':<18} " + " ".join(f"{r['name']:>10}" for r in rows))
    for ph in phases:
        cells = []
        for r in rows:
            p = r["pred_phases"].get(ph, {}).get("mul", 0)
            m = r["prove_phases"].get(f"prove.{ph}", {}).get("mul", 0)
            cells.append("       n/a" if m == 0 and p == 0
                         else f"{100*rel_err(p, m):+9.1f}%")
        print(f"{ph:<18} " + " ".join(cells))

    print(f"\nTRACE SHAPE (geometry -> relation counts, model level 1)")
    for r in rows:
        s, a = r["trace_shape_pred"], r["trace_shape_actual"]
        print(f"  {r['name']:<10} matmuls {s['matmuls']:>4}/{a['matmuls']:<4} "
              f"cells {s['matmul_cells']:>8,}/{a['matmul_cells']:<8,} "
              f"gates {s['gates']:>4}/{a['gates']:<4}")
    return 0 if worst < 0.01 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", default="toy,small",
                    help="comma list: toy, small, large, real")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rates = load_rates()
    print(f"rate card: {rates.name}  mul {rates.mul_ns:.2f} ns  "
          f"mul-defer {rates.mul_defer_ns:.2f} ns  add {rates.add_ns:.2f} ns  "
          f"blake3 {rates.hash_GBps:.2f} GB/s")

    rows = []
    for key in args.set.split(","):
        for spec in SETS[key.strip()]:
            print(f"  running {spec['name']} ...", flush=True)
            rows.append(run_one(spec, rates, args.seed))

    with open(RUNS, "a") as fh:
        for r in rows:
            fh.write(json.dumps({**r, "ts": int(time.time())}) + "\n")
    rc = report(rows)
    print(f"\n{len(rows)} runs appended to {RUNS}")
    print("PASS: model predicts field-op counts to <1%" if rc == 0 else
          "FAIL: a phase is mis-modelled -- see the per-phase table above")
    return rc


if __name__ == "__main__":
    sys.exit(main())
