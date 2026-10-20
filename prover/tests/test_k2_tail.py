"""The unexplained-information tail at the a14w14 profile's constants (GPU):
logits read natively at 2^14; the kernel and the gap domain as at 2^12 in
real logit units; the surprisal at s_b = 2^12 (k2_loader.tail_params(2^14):
s_c = 2^32, s_y = 2^28, s_b = 2^12, gap_max = 2^22, k = s_c/s_b = 2^20).

- The largest gap in the domain, gap_max − 1 (just under 256 real logit
  units), as the output token's own gap, puts the ceiling quotient at its
  maximum, 16,777,209 < 2^24. The proof is accepted, and U, at s_b·ln 2,
  bounds the float reference from above, tightly.
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

CFG = core.LigeroConfig(ELL=64, K_DEG=64, N_LIG=256, T_QUERIES=4)   # the 2^22-entry tables in 2^16 rows
SEED = b"k2-tail-a14w14"
S = 1 << 14
TAIL = kl.tail_params(S)
G = TAIL["gap_max"]
T, V = 2, 8


def _rows(top_gap):
    """Row 0: gaps 0 to top_gap, the last token's the largest; row 1: gaps
    0 to 49 real units. Nonnegative logits (committed as field elements)."""
    hi = 10 + top_gap
    r0 = [hi, hi - S, hi - 3 * S, hi - 40 * S, hi - 200 * S, hi - 255 * S, hi - top_gap + 1, hi - top_gap]
    r1 = [hi - 7 * i * S for i in range(V)]
    return [r0, r1]


def _u_ref(rows, tokens):
    """The float bound in bits for the kernel exp(−gap²/s_c), in log space."""
    U = 0.0
    for row, o in zip(rows, tokens):
        vs = max(row)
        e = [-((vs - x) ** 2) / TAIL["s_c"] for x in row]
        m = max(e)
        U += (m + math.log(sum(math.exp(v - m) for v in e)) - e[o]) / math.log(2.0)
    return U


def _logits(tape, rows):
    flat = torch.tensor([v for r in rows for v in r], dtype=torch.int64, device="cuda")
    return tape.commit("logits", flat.to(torch.uint64), (T, V))


def _run(rows, tokens):
    """(U from the engine pass, the Rust verdict on a proof of a fresh tape)."""
    core._COSET_POWERS_K_CACHE.clear()
    tape = Tape(CFG, lazy=True)
    Sz, _ = prove_unexplained_info(tape, _logits(tape, rows), tokens, T=T, V=V, **TAIL)
    U = bound_bits(int(tape.run_engine_pass()[Sz.var].cpu().item()), s_b=TAIL["s_b"])
    core._COSET_POWERS_K_CACHE.clear()
    tape2 = Tape(CFG, lazy=True)
    prove_unexplained_info(tape2, _logits(tape2, rows), tokens, T=T, V=V, **TAIL)
    acc, msg = rust_verify_tape(tape2, tape2.prove(seed=SEED), seed=SEED)
    return U, acc, msg


def test_the_constants_are_the_a14w14_profiles():
    assert TAIL == dict(s_c=1 << 32, s_y=1 << 28, s_b=1 << 12, gap_max=1 << 22)
    k = TAIL["s_c"] // TAIL["s_b"]
    assert ((G - 1) ** 2 + k - 1) // k == 16_777_209 < 1 << _UI.Z_O_BITS


def test_the_largest_gap_in_the_domain_accepts_with_a_sound_tight_bound():
    rows, tokens = _rows(G - 1), [7, 3]          # token 7's gap is G − 1: the quotient at its maximum
    U, acc, msg = _run(rows, tokens)
    assert acc, msg
    U_ref = _u_ref(rows, tokens)
    assert U_ref - 0.05 <= U < U_ref + 0.3, (U, U_ref)


def test_one_past_the_gap_domain_rejects():
    rows, tokens = _rows(G), [0, 3]              # a gap of exactly gap_max in row 0, not the output's
    _, acc, msg = _run(rows, tokens)
    assert not acc, msg
