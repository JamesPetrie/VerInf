"""Price one ROW OF CAPACITY, and find out whether that price is stable.

Motivated by a correction. An earlier reading of the width ladder concluded "time
grows linearly in d" — false. The model's own projection cost

    C_proj = n_out * ELL * ceil(n_in / ELL)

is Θ(d²) for a transformer matrix at fixed ELL; the multi-row layout splits the d²
weights into blocks, it does not remove them. The ladder looked linear because it
was small, because most of each row was padding while n_in ≤ ELL, and above all
because the projection term was not yet the dominant cost: from d=384 to d=768 the
capacity processed grew 3.67x while the stopwatch grew 2.04x.

So the useful measurement is not "seconds vs d" but

    seconds / total row capacity

and whether it CONVERGES. If it does, that is an honest price per row and
extrapolation to production geometry is defensible. If it drifts, the run is still
dominated by something else and no projection may be made from it.

This script sweeps two axes separately, which the width ladder conflated:

  * width      d at fixed (ELL, N)  -- crosses block boundaries
  * geometry   (ELL, N) at fixed d  -- the axis production actually differs on
                                       (ELL=8192, N=65536 is ~512x the per-row
                                        cost of the ladder we have)

  .venv/bin/python layergkr/bench/row_capacity.py --set width
  .venv/bin/python layergkr/bench/row_capacity.py --set geometry
"""
import argparse
import json
import pathlib
import random
import sys
import time
from typing import Dict, List

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from layergkr import full_layer as fl, rs, semantics as sem
from layergkr.counters import Counter

OUT = pathlib.Path(__file__).parent / "row_capacity.jsonl"

SETS: Dict[str, List[dict]] = {
    # width at fixed geometry: what the earlier ladder did
    "width": [
        dict(name="w-128", S=8, d=128, d_ff=256, E=4, ELL=512, K=1024, N=2048, q=16),
        dict(name="w-256", S=8, d=256, d_ff=512, E=4, ELL=512, K=1024, N=2048, q=16),
        dict(name="w-512", S=8, d=512, d_ff=1024, E=4, ELL=512, K=1024, N=2048, q=16),
    ],
    # FACTORIAL: ELL and N varied INDEPENDENTLY at fixed model width, which is
    # what the earlier geometry ladder failed to do (it moved both together, so a
    # flat step attributed nothing). Row 1 holds N and sweeps ELL; row 2 holds ELL
    # and sweeps N; the last two are joint far points toward production geometry.
    "factorial": [
        dict(name="E512-N4096", S=8, d=128, d_ff=256, E=4, ELL=512, K=1024, N=4096, q=16),
        dict(name="E1024-N4096", S=8, d=128, d_ff=256, E=4, ELL=1024, K=2048, N=4096, q=16),
        dict(name="E2048-N4096", S=8, d=128, d_ff=256, E=4, ELL=2048, K=4096, N=4096, q=16),
        dict(name="E1024-N2048", S=8, d=128, d_ff=256, E=4, ELL=1024, K=2048, N=2048, q=16),
        dict(name="E1024-N8192", S=8, d=128, d_ff=256, E=4, ELL=1024, K=2048, N=8192, q=16),
        dict(name="E2048-N8192", S=8, d=128, d_ff=256, E=4, ELL=2048, K=4096, N=8192, q=16),
        dict(name="E4096-N16384", S=8, d=128, d_ff=256, E=4, ELL=4096, K=8192,
             N=16384, q=16),
    ],
    # geometry at fixed width: the axis production differs on
    "geometry": [
        dict(name="g-256", S=8, d=128, d_ff=256, E=4, ELL=256, K=512, N=1024, q=16),
        dict(name="g-512", S=8, d=128, d_ff=256, E=4, ELL=512, K=1024, N=2048, q=16),
        dict(name="g-1024", S=8, d=128, d_ff=256, E=4, ELL=1024, K=2048, N=4096, q=16),
    ],
}


def capacity(trace, ell: int) -> int:
    """Total RS row capacity the projections touch: sum over every projected
    tensor of n_out * ELL * ceil(n_in / ELL). This is the quantity the cost model
    says the work is proportional to -- capacity, not model entries."""
    total = 0
    for m in trace.matmuls:
        n_out, n_in = len(m.W), len(m.W[0])
        total += n_out * ell * -(-n_in // ell)
        rows, x_in = len(m.X), len(m.X[0])
        total += rows * ell * -(-x_in // ell)          # the activation seam
    for m in trace.moe:
        E, d_out, n_in = len(m.W), len(m.W[0]), len(m.W[0][0])
        total += E * d_out * ell * -(-n_in // ell)     # every expert is projected
    return total


def run(spec: dict, seed: int = 7) -> dict:
    cfg = rs.Config(ELL=spec["ELL"], K_DEG=spec["K"], N_LIG=spec["N"],
                    T_QUERIES=spec["q"])
    t0 = time.perf_counter()
    rs.lagrange_matrix(cfg)                 # one-time setup, kept out of the clock
    t_setup = time.perf_counter() - t0

    toy = sem.ToyConfig(S=spec["S"], d=spec["d"], d_ff=spec["d_ff"], E=spec["E"],
                        table_bits=6, scale_bits=6)
    rng = random.Random(seed)
    trace = sem.forward(toy, rng)
    enrol = fl.Enrollment(cfg)

    t0 = time.perf_counter()
    with Counter("prove") as c:
        proof = fl.prove_full_layer(trace, cfg, enrol, spec["q"], rng, use_masks=True)
    t_prove = time.perf_counter() - t0

    t0 = time.perf_counter()
    ok, why = fl.verify_full_layer(cfg, proof, trace.gates)
    t_verify = time.perf_counter() - t0
    if not ok:
        raise AssertionError(f"{spec['name']} did not verify: {why}")

    cap = capacity(trace, cfg.ELL)
    return {"name": spec["name"], "spec": spec, "capacity": cap,
            "t_setup_s": t_setup, "t_prove_s": t_prove, "t_verify_s": t_verify,
            "ns_per_capacity": t_prove / cap * 1e9,
            "enc_gpu": c.report()["enc_gpu"], "counts": c.report(), "ts": int(time.time())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default="width,geometry")
    args = ap.parse_args()

    rows = []
    for key in args.set.split(","):
        print(f"--- {key.strip()}")
        for spec in SETS[key.strip()]:
            print(f"  {spec['name']} ...", flush=True)
            rows.append(run(spec))

    print(f"\n{'run':<14} {'d':>4} {'ELL':>5} {'N':>6} {'capacity':>13} "
          f"{'setup s':>8} {'prove s':>8} {'ns/cap':>9} {'ns/cap/N':>9}")
    for r in rows:
        sp = r["spec"]
        print(f"{r['name']:<14} {sp['d']:>4} {sp['ELL']:>5} {sp['N']:>6} "
              f"{r['capacity']:>13,} {r['t_setup_s']:>8.1f} {r['t_prove_s']:>8.1f} "
              f"{r['ns_per_capacity']:>9.1f} "
              f"{r['ns_per_capacity'] / sp['N'] * 1000:>9.2f}")
    print("\nns/cap/N is per row-slot per codeword position, scaled x1000. If the")
    print("cost really is encode-bound it should be the FLAT column, not ns/cap --")
    print("capacity counts message slots and says nothing about N, while the encode")
    print("work is capacity*N. Whichever column is flat names the dominant term.")

    print("\nns/cap is the price of one row of capacity. If it CONVERGES across a")
    print("sweep, that is an honest per-row price and extrapolating is defensible.")
    print("If it keeps falling, the projection term is still not dominant and the")
    print("run is measuring fixed costs -- which is exactly the mistake the width")
    print("ladder led to. Neither outcome is a 400B number on its own.")

    with open(OUT, "a") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"\nappended to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
