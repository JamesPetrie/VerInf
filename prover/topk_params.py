"""Parameters and soundness guards of the top-k routing claims
(analysis/topk-routing-design.md §3.1–3.2). Pure Python, no torch, so the
CPU gate tests them; prover/topk_routing.py and the Rust compile enforce the
same inequalities."""
import math

P = 0xFFFF_FFFF_0000_0001


def l_bits(E: int) -> int:
    """Tiebreak width: the bonus (E − 1 − e) fits L bits."""
    return max(1, math.ceil(math.log2(E)))


def top1_guard_ok(width: int, range_bits: int) -> bool:
    """route_top1's guard: one gap with both ends pinned, |gap| < 2^width."""
    return (1 << range_bits) <= P - (1 << width)


def threshold_guard_ok(width: int, range_bits: int) -> bool:
    """The threshold form's guard (design §3.1). τ is free, so the ranges bound
    q̃_s − τ and τ − q̃_u only as residues; their sum q̃_s − q̃_u is then some
    a + c < 2^{R+1} − 1, and a wrong order (|q̃_s − q̃_u| < 2^width, negative)
    has residue above P − 2^width. Sound iff the sum cannot reach it."""
    return (1 << (range_bits + 1)) - 1 <= P - (1 << width)


def threshold_words(width: int, word_bits: int, n_words: int = 0):
    """(n_words, R) for the dominance range on v, or ValueError when the words
    do not cover the width or break the threshold guard."""
    n = n_words or max(1, math.ceil(width / word_bits))
    R = n * word_bits
    if R < width:
        raise ValueError(f"{n}x{word_bits}-bit words cover {R} bits, below the "
                         f"q̃ difference width {width}")
    if not threshold_guard_ok(width, R):
        raise ValueError(f"unsound top-k parameterization: {n}x{word_bits}-bit words "
                         f"(R = {R}) let a wrapped threshold split between a selected "
                         f"and an unselected relation (width {width}); the threshold "
                         f"guard needs 2^(R+1) - 1 <= P - 2^width")
    return n, R


def bracket_guard_ok(z_bits: int, rem_bits: int, w_bits: int, cs_bits: int) -> bool:
    """The gate bracket's integer identity w·Z + rem = C·s (design §3.2): with
    Z < 2^z_bits, rem and Z − 1 − rem ranged to [0, 2^rem_bits), w to
    [0, 2^w_bits) and C·s < 2^cs_bits, both sides stay below P (so the field
    equation is the integer one), and a negative Z − 1 − rem cannot sit in its
    window."""
    lhs = (1 << (w_bits + z_bits)) + (1 << rem_bits)
    return (lhs < P and (1 << cs_bits) < P
            and (1 << rem_bits) <= P - (1 << max(rem_bits, z_bits)) - 1)
