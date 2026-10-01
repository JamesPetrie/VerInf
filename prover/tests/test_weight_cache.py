"""Gates for the decoded-weight cache (LIGERO_WEIGHT_CACHE).

The prover resolves every dense weight's loader once per sweep for the
compute, once more for the aux, and once in each encode pass — on Maverick
that is 4,344 decodes of 362 weights per proof, and on the H200 session-2
profile those decodes were the whole loader column. The cache keeps each
persistent non-shard weight in pinned host memory from its first resolution
and serves a fresh copy afterwards. A decoded weight is a deterministic
function of the GGUF, so the proof may not change by a byte; the Rust
verifier must accept it; every dense weight is decoded ONCE per proof; the
routed shards are never cached (their own cache is LIGERO_ROUTED_Y_CACHE,
and one shard resident at a time is a separate gate); and when the host
budget refuses a weight the loader is simply called again, proof unchanged.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import torch
import core
import claims as _C       # noqa: F401
import packets as _PK     # noqa: F401
from tape import Tape
from routed_projected import routed_projected_matmul
from _rust_verify import rust_verify_tape
from test_shard_streaming import CFG, ShardWatch, _proof_bytes, ZK_SEED, T, K, J, E


class DenseWatch:
    """Counts resolutions of one lazily decoded dense weight."""

    def __init__(self):
        self.loads = 0

    def loader(self, values):
        def load():
            self.loads += 1
            return torch.tensor(values, dtype=torch.int64,
                                device="cuda").to(torch.uint64)
        return load


def _build(dense, shards, tokens=E):
    """X · D (one persistent dense weight, lazily decoded) feeds a routed
    claim over E lazily decoded shards, every expert routed — the two loader
    kinds side by side, as in the Maverick tape."""
    core._COSET_POWERS_K_CACHE.clear()
    tape = Tape(CFG, lazy=True)
    u64 = lambda xs: torch.tensor(xs, dtype=torch.int64,
                                  device="cuda").to(torch.uint64)
    T_ = tokens
    x = tape.commit("X", u64([1 + i for i in range(T_ * K)]), (T_, K))
    d = tape.commit_lazy("D", dense.loader([(3 + i) % 11 for i in range(K * K)]),
                         (K, K), K * K, persistent=True)
    xd = tape.matmul(x, d)
    Mm = [0] * (T_ * E)
    for t in range(T_):
        Mm[t * E + (t * 7) % E] = 1
    m = tape.commit("M", u64(Mm), (T_, E))
    w = [tape.commit_lazy(f"W{e}", shards.loader([(e + 1 + i) % 97
                                                  for i in range(K * J)]),
                          (K, J), K * J, persistent=True)
         for e in range(E)]
    routed_projected_matmul(tape, xd, m, w, T=T_, K=K, J=J, E=E)
    return tape


def _prove(cache_on, wc=None, *, host_fraction=None):
    """One proof; returns (tape, proof, wc, dense loads, shard loads) with
    the loads counted INSIDE the prove (enrollment's own reads excluded)."""
    dense, shards = DenseWatch(), ShardWatch()
    tape = _build(dense, shards)
    if wc is None:
        wc = core.WeightCommitment.from_tape(tape, CFG)
    d0, s0 = dense.loads, shards.loads
    prev = (core._WEIGHT_CACHE_ON, core._WEIGHT_CACHE_HOST_FRACTION, core._WEIGHT_CACHE_GPU_FRACTION)
    try:
        core._WEIGHT_CACHE_ON = cache_on
        if host_fraction is not None:            # both tiers' budgets
            core._WEIGHT_CACHE_HOST_FRACTION = core._WEIGHT_CACHE_GPU_FRACTION = host_fraction
        proof = tape.prove(zk_seed=ZK_SEED, weight_commitment=wc)
    finally:
        core._WEIGHT_CACHE_ON, core._WEIGHT_CACHE_HOST_FRACTION, core._WEIGHT_CACHE_GPU_FRACTION = prev
    assert core._WEIGHT_CACHE is None, "the proof's weight cache must be released after the prove"
    return tape, proof, wc, dense.loads - d0, shards.loads - s0


def test_weight_cache_is_byte_identical_and_accepts():
    """Cache off and on, same enrolled commitment, pinned zk seed: the same
    bytes on the wire, and the cached proof accepted by the Rust verifier."""
    tape_off, proof_off, wc, _, _ = _prove(False)
    base = _proof_bytes(tape_off, proof_off)
    tape_on, proof_on, _, _, _ = _prove(True, wc)
    got = _proof_bytes(tape_on, proof_on)
    assert got == base, "decoded-weight cache: proof bytes differ from the uncached proof"
    assert len(got) > 1000
    acc, msg = rust_verify_tape(tape_on, proof_on, seed=None)
    assert acc, f"cached-weight proof: expected ACCEPT ({msg})"
    print(f"    cache off/on: {len(base)} proof bytes identical; Rust ACCEPT")


def test_dense_weight_decoded_once_and_shards_never_cached():
    """Off, the dense weight is decoded in every sweep and encode pass; on,
    once per proof. The shards are excluded from the cache by construction
    (they carry their own cache and residency rule), so their loads do not
    move."""
    _, _, _, d_off, s_off = _prove(False)
    _, _, _, d_on, s_on = _prove(True)
    assert d_on == 1, f"dense weight decoded {d_on} times with the cache on, want 1"
    assert d_off > d_on, f"uncached prove decoded the dense weight only {d_off} times"
    assert s_on == s_off, f"shard loads moved with the weight cache: {s_off} -> {s_on}"
    print(f"    dense weight decodes per proof: uncached {d_off} -> cached 1; "
          f"shard loads {s_off} on both arms")


def test_zero_budget_falls_back_to_the_loader():
    """Budgets of zero on both tiers refuse every store: the loader is called
    as often as with the cache off, and the proof is unchanged."""
    tape_off, proof_off, wc, d_off, _ = _prove(False)
    tape_z, proof_z, _, d_z, _ = _prove(True, wc, host_fraction=0.0)
    assert d_z == d_off, f"zero budget: dense decodes {d_z}, want the uncached {d_off}"
    assert _proof_bytes(tape_z, proof_z) == _proof_bytes(tape_off, proof_off), \
        "zero-budget prove changed the proof bytes"
    print(f"    zero budget: {d_z} decodes, bytes identical")


def _u64(vals):
    """Field elements from Python ints (including those at or above 2^63)."""
    return torch.tensor([v - (1 << 64) if v >= (1 << 63) else v for v in vals],
                        dtype=torch.int64, device="cuda").view(torch.uint64)


def test_pack_roundtrip_at_the_boundaries_and_the_charged_size():
    """The centered-int32 packing is exact over its whole domain — 0, small
    w, P - w, and both ends 2^31 - 1 and P - (2^31 - 1) — and everything
    outside it (2^31, P - 2^31, P itself, a non-canonical P + 5, 2^40, a
    non-field tensor) takes the int64 entry with the same bits. The budget
    is charged the pinned size, the allocator's power-of-two ceiling."""
    P = core.P
    small = _u64([0, 7, P - 7, (1 << 31) - 1, P - ((1 << 31) - 1)])
    e = core._weight_pack(small, 10 ** 9)
    assert e[0] == 'i32' and e[2] == 32, e[:3]           # 5 x 4 B, pinned as 32
    back = core._weight_unpack(e)
    assert back.dtype == torch.uint64 and back.device == small.device
    assert torch.equal(back.view(torch.int64), small.view(torch.int64))
    for vals in ([1 << 31], [P - (1 << 31)], [P], [P + 5], [1 << 40]):
        t = _u64(vals)
        e = core._weight_pack(t, 10 ** 9)
        assert e[0] == 'i64', (vals, e[0])
        assert torch.equal(core._weight_unpack(e).view(torch.int64), t.view(torch.int64))
    e = core._weight_pack(torch.zeros(3, dtype=torch.int64, device="cuda"), 10 ** 9)
    assert e[0] == 'i64' and torch.equal(core._weight_unpack(e), torch.zeros(3, dtype=torch.int64, device="cuda"))
    assert core._weight_pack(_u64([1] * 9), 10 ** 9)[2] == 64      # 36 B pinned as 64
    assert core._weight_pack(small, 31) is None and core._weight_pack(_u64([1 << 40]), 7) is None
    assert core._pinned_bytes(105 * 2 ** 20) == 128 * 2 ** 20 and core._pinned_bytes(0) == 0
    # the GPU tier: the same packing kept on the device, charged its packed size
    e = core._weight_pack(small, 0, 10 ** 9)
    assert e[0] == 'g32' and e[1].is_cuda and e[1].dtype == torch.int32 and e[2] == 20, e[:3]
    assert torch.equal(core._weight_unpack(e).view(torch.int64), small.view(torch.int64))
    assert core._weight_pack(small, 10 ** 9, 19)[0] == 'i32'          # over the GPU budget -> host tier
    assert core._weight_pack(_u64([1 << 40]), 10 ** 9, 10 ** 9)[0] == 'i64'   # int64 fallback stays on the host
    print("    packing: exact at the boundaries on both tiers; int64 fallbacks exact; sizes charged")


def test_weight_cache_released_when_the_prove_raises():
    """The dense weight is stored in R1's first resolution; the routed claim's
    first shard loader then raises. The cache global must be None afterwards
    (prove_streaming's finally), not held until the next prove."""
    class Raising(ShardWatch):
        def loader(self, values):
            def load():
                self.loads += 1
                raise RuntimeError("shard loader failed on purpose")
            return load

    dense, shards = DenseWatch(), Raising()
    tape = _build(dense, shards)
    d0 = dense.loads
    prev = core._WEIGHT_CACHE_ON
    core._WEIGHT_CACHE_ON = True
    try:
        try:
            tape.prove(zk_seed=ZK_SEED)
        except RuntimeError as err:
            assert "on purpose" in str(err)
        else:
            raise AssertionError("the prove did not raise")
    finally:
        core._WEIGHT_CACHE_ON = prev
    assert dense.loads - d0 == 1 and shards.loads == 1, (dense.loads - d0, shards.loads)
    assert core._WEIGHT_CACHE is None, "weight cache still held after a prove that raised"
    print("    raise after the first store: cache released")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    fails = 0
    for t in tests:
        try:
            t(); print(f"[OK ] {t.__name__}")
        except Exception as e:
            fails += 1; print(f"[XX ] {t.__name__}: {type(e).__name__}: {e}")
    print(f"=== weight-cache: {len(tests)-fails}/{len(tests)} "
          f"{'PASS' if not fails else 'FAIL'} ===")
    return fails


if __name__ == "__main__":
    raise SystemExit(main())
