"""Localise the machine-model error: which PRIMITIVE is priced wrong, and why.

The count model predicts operation counts exactly. The time model does not match
the stopwatch. The wrong answer to that is a correction factor; the right answer
is to find out which primitive's cost is mis-measured and fix how the machine is
modelled. This script does the finding.

Method:

  1. Prove several configurations, recording per-phase COUNTS and per-phase
     SECONDS.
  2. Solve, by least squares over all phases and configurations, for the
     per-primitive costs that best explain the measured seconds:

         seconds_phase  ~=  a*mul + b*mul_defer + c*add + d*hash_bytes

     These are the costs the machine ACTUALLY charges, inferred from the real
     workload.
  3. Compare them against the microbenchmark rates in rates.json.

Where the two disagree, the microbenchmark is measuring something other than what
the code does, and the difference says what. That is a machine-modelling bug with
a location, not a fudge factor.

  .venv/bin/python layergkr/bench/diagnose.py
"""
import json
import pathlib
import sys
from typing import Dict, List

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from layergkr.bench.run_toy import SETS, load_rates, run_one

PRIMS = ["mul", "mul_defer", "add", "hash_bytes"]


def collect(sets: List[str]) -> List[dict]:
    rates = load_rates()
    rows = []
    for key in sets:
        for spec in SETS[key]:
            print(f"  {spec['name']} ...", flush=True)
            rows.append(run_one(spec, rates))
    return rows


def phase_table(rows: List[dict]) -> List[dict]:
    """One record per (run, phase) with its counts and its measured seconds."""
    out = []
    for r in rows:
        for name, c in r["prove_phases"].items():
            if "." in name[len("prove."):]:
                continue                      # keep only top-level phases
            if c.get("seconds", 0) <= 0:
                continue
            out.append({"run": r["name"], "phase": name,
                        "seconds": c["seconds"],
                        **{p: c.get(p, 0) for p in PRIMS}})
    return out


def solve(recs: List[dict]) -> Dict[str, float]:
    """Non-negative least squares would be ideal; plain least squares is enough
    to see the sign and size of the disagreement."""
    A = np.array([[r[p] for p in PRIMS] for r in recs], dtype=float)
    y = np.array([r["seconds"] for r in recs], dtype=float)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
    return {"coef": dict(zip(PRIMS, coef)), "r2": r2, "n": len(recs)}


def main() -> int:
    print("proving instances (counts + per-phase wall time)")
    rows = collect(["toy", "small", "large"])
    recs = phase_table(rows)

    print(f"\nPER-PHASE measurements ({len(recs)} phase-observations)")
    print(f"{'run':<10} {'phase':<16} {'seconds':>9} {'mul':>10} {'mul_defer':>11} "
          f"{'add':>11} {'hashMB':>8}")
    for r in recs:
        print(f"{r['run']:<10} {r['phase'][6:]:<16} {r['seconds']:9.4f} "
              f"{r['mul']:10,} {r['mul_defer']:11,} {r['add']:11,} "
              f"{r['hash_bytes']/1e6:8.2f}")

    fit = solve(recs)
    card = load_rates()
    micro = {"mul": card.mul_ns, "mul_defer": card.mul_defer_ns,
             "add": card.add_ns, "hash_bytes": 1e9 / (card.hash_GBps * 1e9) * 1e9}

    print(f"\nIMPLIED cost per primitive (least squares over the real workload)"
          f"   R^2 = {fit['r2']:.4f}")
    print(f"{'primitive':<12} {'implied':>12} {'microbench':>12} {'ratio':>8}")
    for p in PRIMS:
        imp = fit["coef"][p] * 1e9
        mic = micro[p]
        unit = "ns/byte" if p == "hash_bytes" else "ns/op"
        print(f"{p:<12} {imp:9.3f} {unit:<3} {mic:9.3f} {unit:<3} "
              f"{imp / mic if mic else float('nan'):7.2f}x")

    print("\nREAD THIS AS: where ratio != 1, the microbenchmark is not measuring "
          "what the code does.")
    print("A ratio > 1 means the real workload pays MORE per op than the isolated "
          "loop suggests;")
    print("< 1 means the isolated loop is pessimistic. Either way the fix is to "
          "measure the")
    print("primitive the way the code uses it -- not to scale the total.")

    out = pathlib.Path(__file__).parent / "diagnose.json"
    out.write_text(json.dumps({"fit": fit, "micro": micro, "phases": recs}, indent=2))
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
