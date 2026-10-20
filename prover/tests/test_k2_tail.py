"""The unexplained-information tail at the a14w14 and a16w16 profiles'
constants (GPU): logits read natively at S = 2^14 or 2^16; the kernel and
the gap domain as at 2^12 in real logit units; the surprisal at s_b = 2^12
(k2_loader.tail_params(S): s_c = 16·S², s_y = 2^28, s_b = 2^12,
gap_max = 256·S, k = s_c/s_b; at 2^14: 2^32, 2^22, k = 2^20; at 2^16:
2^36, 2^24, k = 2^24).

- The largest gap in the domain, gap_max − 1 (just under 256 real logit
  units), as the output token's own gap, puts the ceiling quotient at its
  maximum: 16,777,209 at 2^14, 16,777,215 = 2^24 − 1 at 2^16, each below
  2^24. The proof is accepted, and U, at s_b·ln 2, bounds the float
  reference from above, tightly.
- A gap of exactly gap_max, one past the domain, is rejected by the gap
  table's range.

Plain test functions, for tests/run_tests.py."""
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch                    # noqa: E402
import core                     # noqa: E402
import claims as _C             # noqa: E402,F401
import packets as _PK           # noqa: E402,F401
import max_claim as _MX         # noqa: E402,F401
import ui_claim as _UI          # noqa: E402,F401
import k2_loader as kl          # noqa: E402
from tape import Tape           # noqa: E402
from _rust_verify import rust_verify_tape   # noqa: E402
from unexplained_info import bound_bits, prove_unexplained_info   # noqa: E402

CFG = core.LigeroConfig(ELL=64, K_DEG=64, N_LIG=256, T_QUERIES=4)   # 2^22 (2^24)-entry tables in 2^16 (2^18) rows
SEED = b"k2-tail-a14w14"
T, V = 2, 8


def _rows(S, top_gap):
    """Row 0: gaps 0 to top_gap, the last token's the largest; row 1: gaps
    0 to 49 real units. Nonnegative logits (committed as field elements)."""
    hi = 10 + top_gap
    r0 = [hi, hi - S, hi - 3 * S, hi - 40 * S, hi - 200 * S, hi - 255 * S, hi - top_gap + 1, hi - top_gap]
    r1 = [hi - 7 * i * S for i in range(V)]
    return [r0, r1]


def _u_ref(tail, rows, tokens):
    """The float bound in bits for the kernel exp(−gap²/s_c), in log space."""
    U = 0.0
    for row, o in zip(rows, tokens):
        vs = max(row)
        e = [-((vs - x) ** 2) / tail["s_c"] for x in row]
        m = max(e)
        U += (m + math.log(sum(math.exp(v - m) for v in e)) - e[o]) / math.log(2.0)
    return U


def _logits(tape, rows):
    flat = torch.tensor([v for r in rows for v in r], dtype=torch.int64, device="cuda")
    return tape.commit("logits", flat.to(torch.uint64), (T, V))


def _run(tail, rows, tokens):
    """(U from the engine pass, the Rust verdict on a proof of a fresh tape)."""
    core._COSET_POWERS_K_CACHE.clear()
    tape = Tape(CFG, lazy=True)
    Sz, _ = prove_unexplained_info(tape, _logits(tape, rows), tokens, T=T, V=V, **tail)
    U = bound_bits(int(tape.run_engine_pass()[Sz.var].cpu().item()), s_b=tail["s_b"])
    core._COSET_POWERS_K_CACHE.clear()
    tape2 = Tape(CFG, lazy=True)
    prove_unexplained_info(tape2, _logits(tape2, rows), tokens, T=T, V=V, **tail)
    acc, msg = rust_verify_tape(tape2, tape2.prove(seed=SEED), seed=SEED)
    return U, acc, msg


def _constants(S, want, quotient):
    tail = kl.tail_params(S)
    assert tail == want
    k = tail["s_c"] // tail["s_b"]
    assert ((tail["gap_max"] - 1) ** 2 + k - 1) // k == quotient < 1 << _UI.Z_O_BITS


def _largest_gap_accepts(S):
    tail = kl.tail_params(S)
    rows, tokens = _rows(S, tail["gap_max"] - 1), [7, 3]   # token 7's gap is the largest: the quotient at its maximum
    U, acc, msg = _run(tail, rows, tokens)
    assert acc, msg
    U_ref = _u_ref(tail, rows, tokens)
    assert U_ref - 0.05 <= U < U_ref + 0.3, (U, U_ref)


def _one_past_rejects(S):
    tail = kl.tail_params(S)
    rows, tokens = _rows(S, tail["gap_max"]), [0, 3]       # a gap of exactly gap_max in row 0, not the output's
    _, acc, msg = _run(tail, rows, tokens)
    assert not acc, msg


def test_the_constants_are_the_a14w14_profiles():
    _constants(1 << 14, dict(s_c=1 << 32, s_y=1 << 28, s_b=1 << 12, gap_max=1 << 22), 16_777_209)


def test_the_largest_gap_in_the_domain_accepts_with_a_sound_tight_bound():
    _largest_gap_accepts(1 << 14)


def test_one_past_the_gap_domain_rejects():
    _one_past_rejects(1 << 14)


def test_the_constants_are_the_a16w16_profiles():
    _constants(1 << 16, dict(s_c=1 << 36, s_y=1 << 28, s_b=1 << 12, gap_max=1 << 24), (1 << 24) - 1)


def test_at_a16w16_the_largest_gap_in_the_domain_accepts_with_a_sound_tight_bound():
    _largest_gap_accepts(1 << 16)


def test_at_a16w16_one_past_the_gap_domain_rejects():
    _one_past_rejects(1 << 16)
