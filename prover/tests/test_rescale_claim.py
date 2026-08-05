"""Standalone RescaleClaim (S2b) — the signed-floor rescale that must follow
every routed raw accumulator.

The routed-projected matmul commits Y raw (the projection yr = Y*rho is taken
over it), so the rescale the old in-matmul path performed internally becomes
its own claim. These tests check it proves the same thing:

  * an honest rescale of a routed output Rust-ACCEPTs;
  * the rescaled value equals the signed floor reference;
  * a prover that rounds the wrong way REJECTs;
  * a value outside the output range REJECTs (the loose LogUp).
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch
import core
import claims as _C       # noqa: F401
import packets as _PK     # noqa: F401
from tape import Tape, _signed_floor_decomp
from rescale_claim import rescale, RescaleClaim
from routed_projected import routed_projected_matmul
from _rust_verify import rust_verify_tape

CFG = core.LigeroConfig(ELL=8, K_DEG=8, N_LIG=32, T_QUERIES=4)
T, K, J, E = 3, 2, 2, 2
S_IN, S_OUT, WIDTH = 1 << 8, 1 << 4, 12


def _u64(t):
    return t.to(torch.int64).to(torch.uint64).cuda()


def _build():
    """A routed matmul followed by its standalone rescale."""
    core._COSET_POWERS_K_CACHE.clear()
    tape = Tape(CFG, lazy=True)
    X = torch.arange(1, T * K + 1).reshape(T, K)
    W = torch.arange(1, E * K * J + 1).reshape(E, K * J)
    M = torch.zeros(T, E, dtype=torch.int64)
    for t, e in enumerate((0, 1, 0)):
        M[t, e] = 1
    x = tape.commit("X", _u64(X.reshape(-1)), (T, K))
    m = tape.commit("M", _u64(M.reshape(-1)), (T, E))
    w = tape.commit("W", _u64(W.reshape(-1)), (E, K * J))
    y_raw = routed_projected_matmul(tape, x, m, w, T=T, K=K, J=J, E=E)
    y = rescale(tape, y_raw, s_in=S_IN, s_out=S_OUT, output_width=WIDTH)
    return tape, y_raw, y


def test_routed_then_rescale_verifies():
    tape, y_raw, y = _build()
    assert any(isinstance(c, RescaleClaim) for c in tape.claims)
    acc, msg = rust_verify_tape(tape, tape.prove(), seed=None)
    assert acc, f"routed + rescale: expected ACCEPT ({msg})"
    print("    routed raw output + standalone rescale: ACCEPT")


def test_matches_signed_floor_reference():
    tape, y_raw, y = _build()
    live = tape.run_engine_pass()
    raw = live[y_raw.var]
    want, _low, _sh = _signed_floor_decomp(raw.contiguous().view(-1),
                                           S_IN // S_OUT, WIDTH)
    got = live[y.var]
    assert torch.equal(got.view(torch.int64), want.view(torch.int64)), \
        f"{got.view(torch.int64).tolist()} != {want.view(torch.int64).tolist()}"
    print(f"    signed-floor reference matches: {got.view(torch.int64).tolist()}")


def _prove_with_compute(patched):
    import compute_fns as cf
    tape, y_raw, y = _build()
    real = cf.COMPUTE_FNS[RescaleClaim]
    cf.COMPUTE_FNS[RescaleClaim] = patched(real)
    try:
        return tape, tape.prove()
    finally:
        cf.COMPUTE_FNS[RescaleClaim] = real


def test_wrong_rounding_rejects():
    """Round up instead of down while keeping x_low consistent with nothing —
    the first linear (x_full = 2^r*x + x_low) is what catches it."""
    def off_by_one(real):
        def f(claim, live):
            out = real(claim, live)
            out[claim.x] = (out[claim.x].view(torch.int64) + 1).view(torch.uint64)
            return out
        return f
    tape, proof = _prove_with_compute(off_by_one)
    acc, msg = rust_verify_tape(tape, proof, seed=None)
    assert not acc, "wrong rounding: expected REJECT"
    print("    wrong rounding (x_full = 2^r*x + x_low breaks): REJECT ok")


def test_out_of_range_low_word_rejects():
    """x_low >= 2^r keeps the linear satisfiable but leaves the tight range
    table — only the LogUp can see it."""
    def widen_low(real):
        def f(claim, live):
            out = real(claim, live)
            k = 1 << claim.rescale_bits
            low = out[claim.x_low].view(torch.int64)
            high = out[claim.x].view(torch.int64)
            out[claim.x_low] = (low + k).view(torch.uint64)      # >= 2^r
            out[claim.x] = (high - 1).view(torch.uint64)         # keep the sum
            return out
        return f
    tape, proof = _prove_with_compute(widen_low)
    acc, msg = rust_verify_tape(tape, proof, seed=None)
    assert not acc, "out-of-range low word: expected REJECT"
    print("    x_low outside [0, 2^r) (tight LogUp breaks): REJECT ok")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    fails = 0
    for t in tests:
        try:
            t(); print(f"[OK ] {t.__name__}")
        except Exception as e:
            fails += 1; print(f"[XX ] {t.__name__}: {type(e).__name__}: {e}")
    print(f"=== rescale-claim: {len(tests)-fails}/{len(tests)} "
          f"{'PASS' if not fails else 'FAIL'} ===")
    return fails


if __name__ == "__main__":
    raise SystemExit(main())
