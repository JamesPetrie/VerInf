"""S4 structural gate: what the prover actually touches, and how often.

Four things have to be true before a 400B run is worth starting, and none of
them is visible in a verdict — a proof that OOMs and a proof that reads the
enrolled weights three times too often both ACCEPT when they finish at toy
scale. So they are measured directly:

  1. exactly ONE expert shard is resident at a time (128 lazy loaders, a
     finalizer counting live resolutions);
  2. the semantic witness pass runs exactly five times — the number the
     admission model charges, no hidden sixth pass;
  3. P = W*rho is computed exactly once and is FUSED into the sweep that
     already had the shard resident, so no epoch reads the weights twice;
  4. a proof that REFERENCES an enrolled weight commitment performs zero
     RS encodes of the weight block (root and paths come from the
     commitment; encoding it anyway is a full pass over 402.7G slots at
     Maverick scale).
"""
import gc
import pathlib
import sys
import tempfile
import weakref

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch
import core
import claims as _C       # noqa: F401
import packets as _PK     # noqa: F401
import routed_projected
from tape import Tape
from routed_projected import routed_projected_matmul, RoutedProjectedMatmulClaim
from _rust_verify import rust_verify_tape

CFG = core.LigeroConfig(ELL=8, K_DEG=8, N_LIG=32, T_QUERIES=4)
T, K, J, E = 2, 4, 4, 128


class ShardWatch:
    """Counts resolutions and how many resolved shards are alive at once."""

    def __init__(self):
        self.live = 0
        self.peak_live = 0
        self.loads = 0

    def loader(self, values):
        def load():
            self.loads += 1
            t = torch.tensor(values, dtype=torch.int64,
                             device="cuda").to(torch.uint64)
            self.live += 1
            self.peak_live = max(self.peak_live, self.live)

            def release(_ref=None):
                self.live -= 1
            weakref.finalize(t, release)
            return t
        return load


def _build(watch, persistent=False, tokens=T):
    """`tokens=E` routes every expert, which is the production case: at S=1000
    every Maverick expert receives tokens, so the semantic pass reads all 128
    shards and the projection can ride along with them."""
    core._COSET_POWERS_K_CACHE.clear()
    tape = Tape(CFG, lazy=True)
    u64 = lambda xs: torch.tensor(xs, dtype=torch.int64,
                                  device="cuda").to(torch.uint64)
    T_ = tokens
    X = [1 + i for i in range(T_ * K)]
    Mm = [0] * (T_ * E)
    for t in range(T_):
        Mm[t * E + (t * 7) % E] = 1
    x = tape.commit("X", u64(X), (T_, K))
    m = tape.commit("M", u64(Mm), (T_, E))
    w = [tape.commit_lazy(f"W{e}", watch.loader([(e + 1 + i) % 97
                                                 for i in range(K * J)]),
                          (K, J), K * J, persistent=persistent)
         for e in range(E)]
    y = routed_projected_matmul(tape, x, m, w, T=T_, K=K, J=J, E=E)
    return tape, y


def test_one_expert_shard_resident_at_a_time():
    watch = ShardWatch()
    tape, _ = _build(watch)
    tape.prove()
    gc.collect()
    assert watch.peak_live <= 1, (
        f"{watch.peak_live} expert shards were resident at once (want 1); at "
        f"Maverick shapes each is ~336 MB and there are 128 of them")
    print(f"    {E} lazy shards, {watch.loads} resolutions, peak resident "
          f"= {watch.peak_live}")


def test_five_semantic_sweeps_and_one_projection():
    watch = ShardWatch()
    tape, _ = _build(watch)
    import compute_fns as cf
    real = cf.COMPUTE_FNS[RoutedProjectedMatmulClaim]
    calls = {"n": 0}

    def counted(claim, live, rho=None):
        calls["n"] += 1
        return real(claim, live, rho)

    cf.COMPUTE_FNS[RoutedProjectedMatmulClaim] = counted
    try:
        proof = tape.prove()
    finally:
        cf.COMPUTE_FNS[RoutedProjectedMatmulClaim] = real
    stats = routed_projected.P_CACHE_STATS
    assert calls["n"] == 5, f"semantic witness pass ran {calls['n']} times, want 5"
    assert stats["misses"] == 1, f"projection computed {stats['misses']} times, want 1"
    acc, msg = rust_verify_tape(tape, proof, seed=None)
    assert acc, f"streamed proof: expected ACCEPT ({msg})"
    print(f"    sweeps={calls['n']} projections={stats['misses']} "
          f"cache_hits={stats['hits']} -> ACCEPT")


def test_projection_adds_no_second_read_of_the_weights():
    """The projection is fused into the sweep that already had the shard
    resident, so switching it on must not add a single extra shard load."""
    # Every expert routed — the production case; with only a couple of routed
    # experts the semantic pass would not have had the other shards resident
    # anyway and fusing could not save their reads.
    all_routed = dict(persistent=True, tokens=E)
    tape_off, _ = _build(ShardWatch(), **all_routed)
    wc = core.WeightCommitment.from_tape(tape_off, CFG)
    # A run in which the projection is computed separately, the way an
    # unfused implementation would do it.
    watch_split = ShardWatch()
    tape_split, _ = _build(watch_split, **all_routed)
    import compute_fns as cf
    real = cf.COMPUTE_FNS[RoutedProjectedMatmulClaim]
    cf.COMPUTE_FNS[RoutedProjectedMatmulClaim] = (
        lambda claim, live, rho=None: real(claim, live, None))   # never fuse
    try:
        tape_split.prove(weight_commitment=wc)
    finally:
        cf.COMPUTE_FNS[RoutedProjectedMatmulClaim] = real
    split_loads = watch_split.loads

    watch_fused = ShardWatch()
    tape_fused, _ = _build(watch_fused, **all_routed)
    tape_fused.prove(weight_commitment=wc)
    saved = split_loads - watch_fused.loads
    assert saved == E, (
        f"fusing the projection saved {saved} shard loads, want {E} (the "
        f"projection riding along with a pass the sweep already made)")
    print(f"    shard loads: unfused {split_loads} -> fused "
          f"{watch_fused.loads} (one full weight pass saved)")


def _proof_bytes(tape, proof):
    """The production wire, for byte comparison (test_weight_split's pattern)."""
    import os
    import tempfile
    import protocol as pr
    from proof_dump import dump_proof
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        dump_proof(path, pr.claims_to_json(tape.claims, CFG), None, proof, None, None)
        with open(path, "rb") as f:
            return f.read()
    finally:
        for p in (path, path + ".part"):
            if os.path.exists(p):
                os.unlink(p)


ZK_SEED = b"\x33" * 32


def test_routed_output_cache_is_byte_identical_and_accepts():
    """LIGERO_ROUTED_Y_CACHE reuses the routed claims' first-sweep Y. Y is a
    deterministic function of committed inputs, so the proof may not change by
    a byte, and the Rust verifier must accept the cached-Y proof."""
    tape_off, _ = _build(ShardWatch(), persistent=True, tokens=E)
    wc = core.WeightCommitment.from_tape(tape_off, CFG)
    prev = core._ROUTED_Y_CACHE_ON
    try:
        core._ROUTED_Y_CACHE_ON = False
        base = _proof_bytes(tape_off, tape_off.prove(zk_seed=ZK_SEED, weight_commitment=wc))
        core._ROUTED_Y_CACHE_ON = True
        tape_on, _ = _build(ShardWatch(), persistent=True, tokens=E)
        proof_on = tape_on.prove(zk_seed=ZK_SEED, weight_commitment=wc)
    finally:
        core._ROUTED_Y_CACHE_ON = prev
    got = _proof_bytes(tape_on, proof_on)
    assert got == base, "routed-output cache: proof bytes differ from the uncached proof"
    assert len(got) > 1000
    acc, msg = rust_verify_tape(tape_on, proof_on, seed=None)
    assert acc, f"cached-Y routed proof: expected ACCEPT ({msg})"
    print(f"    cache off/on: {len(base)} proof bytes identical; Rust ACCEPT")


def test_routed_output_cache_skips_three_of_five_shard_passes():
    """Every expert routed, block enrolled: without the cache the shards are
    read for Y in all five sweeps and for P once, fused into R2. With it, Y
    is computed in R1 and stored, R2 still walks the shards for P, and R3,
    the fold and the opening read no shard — three full passes fewer. The
    encode-path reads of the fold and opening sweeps are the same on both
    sides, so the difference is exactly the three passes."""
    def loads_in_prove(cache_on):
        watch = ShardWatch()
        tape, _ = _build(watch, persistent=True, tokens=E)
        wc = core.WeightCommitment.from_tape(tape, CFG)
        before = watch.loads
        prev = core._ROUTED_Y_CACHE_ON
        try:
            core._ROUTED_Y_CACHE_ON = cache_on
            tape.prove(weight_commitment=wc)
        finally:
            core._ROUTED_Y_CACHE_ON = prev
        return watch.loads - before, dict(routed_projected.P_CACHE_STATS)

    off, st_off = loads_in_prove(False)
    on, st_on = loads_in_prove(True)
    assert off - on == 3 * E, (
        f"routed-output cache saved {off - on} shard loads, want {3 * E} "
        f"(uncached {off}, cached {on})")
    assert st_off["misses"] == 1 and st_on["misses"] == 1, (st_off, st_on)
    print(f"    shard loads per proof: uncached {off} -> cached {on} "
          f"(three passes of {E} saved; projection still computed once)")


def test_routed_output_cache_frees_y_after_its_consumer():
    """The copy of Y the cache hands the sweep must die when the sweep frees
    Y after its last consumer, not linger in the loop's temporaries until the
    next streaming claim: at production that is a 65 MB tensor per MoE layer
    held across the whole attention block that follows. Y feeds a plain
    hadamard by integer ones, which feeds a second; when the second one
    computes, every copy of Y the cache loaded in this sweep must already
    be gone. Plain products, no scales: Y reaches 173,214 on this fixture,
    past any 16-bit output window, and a rescaled hadamard would range-fail."""
    import compute_fns as cf
    from claims import HadamardClaim
    watch = ShardWatch()
    tape, y = _build(watch, persistent=True, tokens=E)
    u64 = lambda xs: torch.tensor(xs, dtype=torch.int64,
                                  device="cuda").to(torch.uint64)
    ones = tape.commit("ones", u64([1] * (E * J)), (E, J))
    h1 = tape.hadamard(y, ones)
    tape.hadamard(h1, ones)
    h2 = tape.claims[-1]
    assert isinstance(h2, HadamardClaim)
    wc = core.WeightCommitment.from_tape(tape, CFG)

    flags = []                                   # one per copy the cache loaded
    real_load = core._routed_load

    def tracked_load(entry):
        t = real_load(entry)
        flag = {"alive": True}
        weakref.finalize(t, flag.__setitem__, "alive", False)
        flags.append(flag)
        return t

    seen = []                                    # alive? at the second hadamard
    real_h = cf.COMPUTE_FNS[HadamardClaim]

    def observing(claim, inputs):
        if claim is h2 and flags:
            seen.append(flags[-1]["alive"])
        return real_h(claim, inputs)

    prev = core._ROUTED_Y_CACHE_ON
    core._routed_load, cf.COMPUTE_FNS[HadamardClaim] = tracked_load, observing
    try:
        core._ROUTED_Y_CACHE_ON = True
        proof = tape.prove(weight_commitment=wc)
    finally:
        core._ROUTED_Y_CACHE_ON = prev
        core._routed_load, cf.COMPUTE_FNS[HadamardClaim] = real_load, real_h
    assert len(flags) == 4, f"expected a cache load in each of R2, R3, fold, open; got {len(flags)}"
    assert len(seen) == 4, f"the second hadamard was observed in {len(seen)} sweeps, want 4"
    assert not any(seen), (
        f"a cached Y was still alive at the claim after its consumer in "
        f"{sum(seen)} of {len(seen)} sweeps")
    acc, msg = rust_verify_tape(tape, proof, seed=None)
    assert acc, f"retention fixture: expected ACCEPT ({msg})"
    print(f"    cached Y freed after its consumer in all {len(seen)} sweeps that loaded it; Rust ACCEPT")


def _rows_encoded(prove_kwargs):
    """Total RS-encoded rows in one proof, counted at the encoder itself."""
    watch = ShardWatch()
    tape, _ = _build(watch, persistent=True)
    seen = {"rows": 0}
    real_enc = core.encode_messages

    def counting(messages, cfg, **kw):
        seen["rows"] += messages.size(0)
        return real_enc(messages, cfg, **kw)

    core.encode_messages = counting
    try:
        proof = tape.prove(**prove_kwargs)
    finally:
        core.encode_messages = real_enc
    return seen["rows"], tape, proof


def test_referenced_commitment_saves_the_r1_weight_encode():
    """A referenced enrolled commitment supplies root and paths, so R1 has no
    sink for the weight block. Encoding it anyway is a full RS pass over the
    enrolled model — 402.7G slots at Maverick scale, ~3625 s by the admission
    model's own rate. The saving must be exactly the weight block."""
    tape0, _ = _build(ShardWatch(), persistent=True)
    wc = core.WeightCommitment.from_tape(tape0, CFG)
    rows_rebuild, _t1, _p1 = _rows_encoded({})
    rows_ref, tape2, proof = _rows_encoded({"weight_commitment": wc})
    saved = rows_rebuild - rows_ref
    assert saved == wc.m_w, (
        f"referencing saved {saved} encoded rows, want exactly the weight "
        f"block ({wc.m_w}); R1 is still re-encoding the enrolled model")
    assert proof.root_w == wc.root
    acc, msg = rust_verify_tape(tape2, proof, seed=None)
    assert acc, f"referenced-weight routed proof: expected ACCEPT ({msg})"
    print(f"    encoded rows: rebuild {rows_rebuild} -> referenced {rows_ref} "
          f"(saved {saved} = the whole weight block), ACCEPT")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    fails = 0
    for t in tests:
        try:
            t(); print(f"[OK ] {t.__name__}")
        except Exception as e:
            fails += 1; print(f"[XX ] {t.__name__}: {type(e).__name__}: {e}")
    print(f"=== shard-streaming: {len(tests)-fails}/{len(tests)} "
          f"{'PASS' if not fails else 'FAIL'} ===")
    return fails


if __name__ == "__main__":
    raise SystemExit(main())
