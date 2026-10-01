"""A/B of the routed-output cache (LIGERO_ROUTED_Y_CACHE) on the toy routed
tape, read through the per-sweep table (LIGERO_SWEEP_TIMING).

Toy scale says nothing about speed — the shards are a few hundred bytes and
recompute is free — so the numbers to read are the COUNTS: shard loads per
sweep, cache reads and writes, projections computed. Uncached, every sweep
resolves the active shards for Y and the R2 sweep every shard for P; cached,
R1 stores Y, R2 still walks the shards for P, and R3, the fold and the
opening resolve none. The fold and opening rows also carry the enrolled
block's encode-path reads, identical on both arms. The proof bytes are the
same on both arms (gated in prover/tests/test_shard_streaming.py); this
script is the instrument's smoke test and the shape of the session-2 A/B on
the instrumented Maverick prove (profiler/instrumented_prove.py
--sweep-timing [--routed-cache]).

    LIGERO_SWEEP_TIMING=1 python3 analysis/bench/ab_routed_cache.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("LIGERO_SWEEP_TIMING", "1")
R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R / "prover"))
sys.path.insert(0, str(R / "prover/tests"))
import _uint64_compat  # noqa
import core
import routed_projected
import test_shard_streaming as tss


def arm(cache_on: bool):
    watch = tss.ShardWatch()
    tape, _ = tss._build(watch, persistent=True, tokens=tss.E)
    wc = core.WeightCommitment.from_tape(tape, tss.CFG)
    loads0 = watch.loads
    print(f"\n=== routed-output cache {'ON' if cache_on else 'OFF'} ===", flush=True)
    prev = core._ROUTED_Y_CACHE_ON
    try:
        core._ROUTED_Y_CACHE_ON = cache_on
        tape.prove(weight_commitment=wc)
    finally:
        core._ROUTED_Y_CACHE_ON = prev
    recs = [(r['label'], r['n'].get('loads', 0), r['n'].get('routed_rd', 0),
             r['n'].get('routed_wr', 0), r['n'].get('proj', 0)) for r in core._SWEEP_RECS]
    print(f"    shard loads in prove: {watch.loads - loads0}; "
          f"projections: {routed_projected.P_CACHE_STATS['misses']}; "
          f"peak resident shards: {watch.peak_live}", flush=True)
    return watch.loads - loads0, recs


off, recs_off = arm(False)
on, recs_on = arm(True)
E = tss.E
print(f"\nshard loads per proof: uncached {off} -> cached {on} "
      f"(saved {off - on}; three passes of {E} = {3 * E})")
print("per sweep (label, loads, routed_rd, routed_wr, proj):")
for a, b in zip(recs_off, recs_on):
    print(f"    {a[0]:5s} off {a[1:]}  on {b[1:]}")
assert off - on == 3 * E, f"expected exactly three passes saved, got {off - on}"
print("ab_routed_cache: counts as expected")
