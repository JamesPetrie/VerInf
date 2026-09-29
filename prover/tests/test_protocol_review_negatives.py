"""Protocol review (2026-09-25) regression cases.

Each case is the FULL alternative witness the pre-fix constraint system
accepted — every other constraint of the claim still holds — and each must
now REJECT in the Rust verifier, while the honest witness of the same tape
ACCEPTs:

  F01  causal Softmax: a masked cell's free z reaches the nonzero half of
       the doubled table (row 0 attends to its future token instead of itself)
  F03  SiLU: the unranged branch word a_1 selects the other table half
       (x = 65536 -> -1) or a negative word with the other sign (x = 4 -> 65534)
  F04  SiLU: sign = 1 at x = 0 reads T_neg[0] = -1 instead of T_pos[0] = 1
  F05  surprisal: a non-canonical remainder gives a field quotient that
       changes the reported sum

Needs a card.  Run:  python prover/tests/test_protocol_review_negatives.py
"""
import os
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import core
import claims as _C        # noqa: F401
import packets as _PK      # noqa: F401
import max_claim as _MX    # noqa: F401
import ui_claim as _UI     # noqa: F401
from claims import SiluConfig, silu_tpos_tneg
from tape import Tape
from compute_fns import WITNESS_TAMPER
from ui_claim import InfoFinalizeClaim
from unexplained_info import prove_unexplained_info
from _rust_verify import rust_verify_tape

P = 2**64 - 2**32 + 1
CFG = core.LigeroConfig(ELL=8, K_DEG=8, N_LIG=32, T_QUERIES=4)
SEED = b"protocol-review-negatives"


def _u64(vals):
    """Signed or field ints -> a cuda uint64 tensor (values may exceed int64)."""
    return torch.from_numpy(np.array([v % P for v in vals], dtype=np.uint64)).to("cuda")


def _override(cells):
    """Tamper fn: replace the given flat indices with the given field values."""
    def fn(t):
        t = t.clone().contiguous()
        signed = t.view(torch.int64)
        for i, v in cells.items():
            v %= P
            signed[i] = v - (1 << 64) if v >= (1 << 63) else v
        return t
    return fn


def _tamper(claim_name, fields):
    WITNESS_TAMPER.clear()
    for field, cells in fields.items():
        WITNESS_TAMPER[(claim_name, field)] = _override(cells)


def _verdict(tape):
    try:
        proof = tape.prove(seed=SEED)
        return rust_verify_tape(tape, proof, seed=SEED)
    finally:
        WITNESS_TAMPER.clear()


# --------------------------------------------------------------------- SiLU
SC = SiluConfig(b=4, T_LEN=1 << 14, b_2=1 << 16, b_3=1 << 32, b_4=1 << 48,
                width_2=16, width_3=16, width_4=14, r=12)          # production
T_POS, T_NEG = silu_tpos_tneg(SC)
T_Y = T_POS + T_NEG
X_SILU = [65536, 4, 0, 100, -3, 12345, -65536, 7]


def silu_cell(x, sign, a0, a1, a2, a3, a4):
    """Every phase-1 value of one SiLU cell from its declarations (sign and
    the words), exactly as the constraints tie them; the honest generator's
    choice is one such assignment."""
    x %= P
    a1 %= P
    C = sign * x % P
    mag = (a0 + SC.b * a1 + SC.b_2 * a2 + SC.b_3 * a3 + SC.b_4 * a4) % P
    g = (SC.b_2 * a2 + SC.b_3 * a3 + SC.b_4 * a4) % P
    key = (SC.T_LEN * sign + a1) % P
    assert 0 <= key < 2 * SC.T_LEN, "the paired lookup itself would reject"
    y = T_Y[key]
    is_high = 1 if g else 0
    inv_g = pow(g, P - 2, P) if g else 0
    out_sat = (x - C) % P
    mux_a = is_high * y % P
    mux_b = is_high * out_sat % P
    output = (y - mux_a + mux_b) % P
    inv_x = pow(x, P - 2, P) if x else 0
    return dict(sign=sign, magnitude=mag, C=C, a_0=a0, a_1=a1, a_2=a2, a_3=a3,
                a_4=a4, g=g, inv_g=inv_g, is_high=is_high, inv_x=inv_x, key=key,
                output_sat=out_sat, mux_a=mux_a, mux_b=mux_b, y=y, output=output)


def _silu_tape(cell=None, values=None):
    core._COSET_POWERS_K_CACHE.clear()
    tape = Tape(CFG, silu_config=SC, lazy=True)
    x = tape.commit("x", _u64(X_SILU), (len(X_SILU),))
    tape.silu(x)
    if cell is not None:
        _tamper("SiluClaim", {f: {cell: v} for f, v in values.items()})
    return tape


def test_silu_honest_accepts():
    acc, msg = _verdict(_silu_tape())
    assert acc, f"honest SiLU: expected ACCEPT ({msg})"
    print("    honest SiLU: ACCEPT")


def test_f03_branch_word_selects_the_other_half():
    # x = 65536: honest is (a_1, a_2) = (0, 1), output 65536 (saturated);
    # the alternative (16384, 0) reads T_neg[0] and outputs -1.
    alt = silu_cell(65536, 0, 0, 16384, 0, 0, 0)
    assert alt["output"] == P - 1
    acc, msg = _verdict(_silu_tape(0, alt))
    assert not acc, "F03: a_1 = T_LEN accepted"
    print(f"    F03 a_1 = 16384 at x = 65536: REJECT ({msg.splitlines()[-1]})")


def test_f03_negative_word_with_the_other_sign():
    # x = 4: honest is sign 0, a_1 = 1, output 3; sign 1 with a_1 = -1 has
    # magnitude -4, key 16383, output 65534.  The sign pin passes (C = 4,
    # inv_x = 1/4), so only the a_1 range catches it.
    alt = silu_cell(4, 1, 0, -1, 0, 0, 0)
    assert alt["output"] == 65534
    acc, msg = _verdict(_silu_tape(1, alt))
    assert not acc, "F03: a_1 = -1 accepted"
    print(f"    F03 a_1 = -1, sign 1 at x = 4: REJECT ({msg.splitlines()[-1]})")


def test_f04_sign_at_zero():
    # x = 0: both signs give zero magnitude; sign 1 reads T_neg[0] = -1.
    alt = silu_cell(0, 1, 0, 0, 0, 0, 0)
    assert alt["output"] == P - 1 and alt["inv_x"] == 0
    acc, msg = _verdict(_silu_tape(2, alt))
    assert not acc, "F04: sign = 1 at x = 0 accepted"
    print(f"    F04 sign = 1 at x = 0: REJECT ({msg.splitlines()[-1]})")


# ------------------------------------------------------------------ Softmax
Z_MAX, S = 40000, 4096           # demo_maverick_block's causal softmax


def _softmax_tape(tamper=False, saturate=True, scores=(0, 0, 0, 0)):
    core._COSET_POWERS_K_CACHE.clear()
    tape = Tape(CFG, lazy=True)
    # two query positions, one head: row 0 has one permitted cell (i = 0) and
    # one masked cell (i = 1); row 1 is fully permitted
    x = tape.commit("sc", _u64(list(scores)), (4,))
    tape.softmax(x, M=2, s_x=S, s_c=S, s_y=S, Z_max=Z_MAX, saturate=saturate,
                 Z_high_width=16, aux_chunk_width=24, causal=True, heads=1)
    if tamper:
        # Row 0 under shift c = Z_max: the permitted cell decomposes as
        # (z, z_high) = (0, 1) and is muxed to 0; the masked cell declares
        # z = -Z_max, so its key z + Z_max = 0 reads T_A[0] = s_y.  The row
        # sums and the bracket are unchanged; the weight moved to the future
        # token.  Row 1 is the honest row.
        _tamper("SoftmaxClaim", {
            "c2": {0: Z_MAX}, "c2_shifted": {0: Z_MAX + (1 << 23)},
            "z": {0: 0, 1: -Z_MAX}, "z_high": {0: 1, 1: 0},
            "inv_z_high": {0: 1}, "is_high": {0: 1},
            "y_A_raw": {0: S, 1: S}, "y_B_raw": {0: S + 1, 1: S + 1},
            "mux_y_A": {0: S, 1: 0}, "mux_y_B": {0: S + 1, 1: 0},
            "y_A": {0: 0, 1: S}, "y_B": {0: 0, 1: S + 1},
            "s1": {0: S}, "s2": {0: S + 1}, "r_lo": {0: 0}, "r_hi": {0: 0},
        })
    return tape


def test_softmax_honest_accepts():
    acc, msg = _verdict(_softmax_tape())
    assert acc, f"honest causal softmax: expected ACCEPT ({msg})"
    print("    honest causal softmax: ACCEPT")


def test_softmax_honest_accepts_without_saturation():
    # the masked pin applies to every causal claim: row [10, 0] has shift 10,
    # and its masked cell must commit z = 0, not the shift — from BOTH witness
    # generators (the torch one is the default; LIGERO_GPU_SOFTMAX=0 is numpy)
    import compute_fns as _cf
    for gpu in (True, False):
        _cf._GPU_SOFTMAX_ON = gpu
        try:
            acc, msg = _verdict(_softmax_tape(saturate=False, scores=(10, 0, 0, 0)))
        finally:
            _cf._GPU_SOFTMAX_ON = os.environ.get("LIGERO_GPU_SOFTMAX", "1") != "0"
        assert acc, f"honest causal softmax, no saturation, gpu={gpu}: expected ACCEPT ({msg})"
        print(f"    honest causal softmax without saturation (gpu generator={gpu}): ACCEPT")


def test_f01_masked_cell_takes_the_weight():
    acc, msg = _verdict(_softmax_tape(tamper=True))
    assert not acc, "F01: attention on a masked (future) cell accepted"
    print(f"    F01 masked z = -Z_max: REJECT ({msg.splitlines()[-1]})")


# ---------------------------------------------------------------- Surprisal
T_UI, V_UI = 2, 8
UI = dict(T=T_UI, V=V_UI, s_c=256, s_y=1 << 12, s_b=16, gap_max=128)   # k = 16
ROWS = [[40, 8, 32, 0, 24, 16, 36, 4], [4, 44, 12, 36, 20, 8, 40, 28]]
TOKENS = [1, 5]


def _info_tape():
    core._COSET_POWERS_K_CACHE.clear()
    tape = Tape(CFG, lazy=True)
    flat = [v for row in ROWS for v in row]
    logits = tape.commit("logits", _u64(flat), (T_UI, V_UI))
    prove_unexplained_info(tape, logits, TOKENS, **UI)
    return tape


def _live_info(tape):
    live = tape.run_engine_pass()
    c = next(cl for cl in tape.claims if isinstance(cl, InfoFinalizeClaim))
    get = lambda v: [int(u) for u in live[v].cpu().numpy().astype(np.uint64)]
    return c, get(c.gap_o2), get(c.rem), get(c.b), get(c.z_o)


def test_info_honest_accepts():
    acc, msg = _verdict(_info_tape())
    assert acc, f"honest surprisal: expected ACCEPT ({msg})"
    print("    honest surprisal: ACCEPT")


def test_f05_field_quotient_moves_the_sum():
    c, g2, rem, b, z_o = _live_info(_info_tape())
    k = c.k
    # position 0 takes the next remainder; the field quotient that satisfies
    # k·z ≡ g² + rem is then a P-sized value, and the reported sum moves.
    rem1 = [(rem[0] + 1) % k, rem[1]]
    kinv = pow(k, P - 2, P)
    z1 = [(g2[t] + rem1[t]) * kinv % P for t in range(T_UI)]
    assert z1[0] != z_o[0] and z1[0] >= (1 << 24), "the alternative quotient is not P-sized"
    sur1 = [(z1[t] + b[t]) % P for t in range(T_UI)]
    words = {j: {t: (z1[t] >> (c.wb * j)) & ((1 << c.wb) - 1) for t in range(T_UI)}
             for j in range(len(c.zw))}
    tape = _info_tape()
    _tamper("InfoFinalizeClaim", {
        "rem": dict(enumerate(rem1)), "z_o": dict(enumerate(z1)),
        "surprisal": dict(enumerate(sur1)),
        **{f"zw[{j}]": words[j] for j in range(len(c.zw))},
    })
    acc, msg = _verdict(tape)
    assert not acc, "F05: a non-canonical remainder with its field quotient accepted"
    print(f"    F05 remainder + field quotient: REJECT ({msg.splitlines()[-1]})")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    fails = 0
    for t in tests:
        try:
            t(); print(f"[OK ] {t.__name__}")
        except Exception as e:
            fails += 1; print(f"[XX ] {t.__name__}: {type(e).__name__}: {e}")
    print(f"=== protocol-review-negatives: {len(tests)-fails}/{len(tests)} "
          f"{'PASS' if not fails else 'FAIL'} ===")
    return fails


if __name__ == "__main__":
    raise SystemExit(main())
