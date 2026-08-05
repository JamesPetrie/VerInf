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
