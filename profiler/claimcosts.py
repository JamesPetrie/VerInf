"""Per-claim cost accounting: (W, cids, Q) from claim type + shape params.

W    = committed witness slots (phase-1 + phase-2 + phase-3 late aux)
cids = distinct linear constraint ids
Q    = quadratic products (slot-products)

Formulas are the active-in-production forms from analysis/maverick-cost-model.md
(grounded in prover/claims.py, prover/routing_claim.py, CLAIM_SPECS.md; rmsnorm
follows the paper's A.1 wrap-free bracket row): output-rescale ON for
matmul/hadamard/rope/rmsnorm, softmax saturate+causal, silu without rescale,
rmsnorm/softmax without INPUT rescale (s_in=0). matmul/hadamard price both
rescale settings; for the rest, a claim carrying a non-production mode flag
(an extracted rmsnorm/softmax/silu with input rescale, rmsnorm without
output rescale, rope without rescale, softmax without saturation) is
REJECTED with ValueError rather than silently priced with the wrong
formula — extend the formula table when such a mode ships. A claim record
carries `type` (canonical name or the prover dataclass name) and `params`;
cost() returns the triple.

Pure Python, no dependencies. The tape extractor records exact W from Variable
lengths and uses these formulas for cids/Q; the synthetic builders use them for
all three. Unknown claim types fall back to (W_hint, W_hint, 0) with a warning
so totals stay order-of-magnitude honest rather than silently dropping work.

Params arrive either canonical (synth: L, B, M, ...) or as the prover's own
field names flattened by the extractor (length, SEQ/d_h/heads, n_tokens); the
accessors below accept both spellings.
"""
from __future__ import annotations

import sys
from typing import Dict, Tuple

Triple = Tuple[float, float, float]  # (W, cids, Q)


def _length(p) -> float:
    """Element count: canonical `L`, or the prover claims' `length` field."""
    return p["L"] if "L" in p else p["length"]


def _reject_mode(name: str, mode: str):
    raise ValueError(
        f"claimcosts: {name} {mode} has no formula — rejecting rather than "
        f"pricing the production mode's form (see README caveats)")


def _matmul(p) -> Triple:
    m, k, n = p["m"], p["k"], p["n"]
    H = p.get("heads", 1)
    mHn = m * H * n
    if p.get("rescale", True):
        return (6 * mHn + 3 * k, 2 * k + H + 2 * mHn, k + 2 * mHn)
    return (mHn + 3 * k, 2 * k + H, k)


def _rmsnorm(p) -> Triple:
    # Wrap-free bracket constants (paper A.1/B.4; seventeen base linear
    # families in rmsnorm_compile plus the 2Bd output-rescale families).
    # The per-cell 7Bd/2Bd/3Bd terms are unchanged from the pre-fix row.
    # Production mode: INPUT rescale OFF (s_in=0 — the 7 Bd-sized vars are
    # all output-side), OUTPUT rescale ON.
    if p.get("rescale", False):
        _reject_mode("rmsnorm", "with input rescale")
    if p.get("output_rescale_bits", 1) == 0:
        _reject_mode("rmsnorm", "without output rescale")
    B, d = p["B"], p["d"]
    return (7 * B * d + 82 * B, 17 * B + 2 * B * d, 3 * B * d + 42 * B)


def _softmax(p) -> Triple:
    if p.get("rescale", False):      # s_in word-decomp adds unmodeled cids/Q
        _reject_mode("softmax", "with input rescale")
    if not p.get("saturate", True):
        _reject_mode("softmax", "without saturation")
    B, M = p["B"], p["M"]
    BM = B * M
    causal_f0 = B * (M + 1) * 0.5 if p.get("causal", True) else BM
    return (15 * BM + 9 * B, causal_f0 + 4 * BM + 5 * B, 8 * BM + 3 * B)


def _silu(p) -> Triple:
    if p.get("rescale", False):
        _reject_mode("silu", "with rescale")
    L = _length(p)
    return (23 * L, 7 * L, 12 * L)


def _hadamard(p) -> Triple:
    L = _length(p)
    if p.get("rescale", True):
        return (6 * L, 2 * L, 3 * L)
    # No rescale -> hadamard_compile emits NO linear packets (cur never
    # advances past base), just the one quad family of L products.
    return (L, 0, L)


def _rope(p) -> Triple:
    if not p.get("rescale", True):
        _reject_mode("rope", "without output rescale")
    # Canonical L, or RoPEConfig's SEQ x heads x d_h flattened by extract.
    L = p["L"] if "L" in p else p["SEQ"] * p["heads"] * p["d_h"]
    return (6 * L, 3 * L, 2 * L)


def _add(p) -> Triple:
    L = _length(p)
    return (L, L, 0)


def _embed_lookup(p) -> Triple:
    # Canonical L, or EmbeddingLookupClaim's len(token_ids) x d.
    L = p["L"] if "L" in p else p["n_tokens"] * p["d"]
    return (L, L, 0)


def _ptlookup(p) -> Triple:
    L = _length(p)
    return (3 * L, L, L)


def _routing(p) -> Triple:
    # Synth-only BUNDLED form: routing_core + the word-extraction/range-word
    # aux that route_top1 emits as separate claims on a real tape (the
    # 2*nw*TE W, TE cids, nw*TE Q). Extracted tapes never hit this row:
    # RoutingClaim aliases to routing_core and the word/range claims carry
    # their own formulas, so the pieces sum to exactly this.
    # n_words=3 matches the MoE router (width 26+7 over 11-bit words). The
    # hidden-prompt one-hot (E=V, B_logit=1) is n_words=2 -- overcharged by
    # 1*TE until B_logit/word_bits are plumbed through.
    T, E = p["T"], p["E"]
    nw = p.get("n_words", 3)
    TE = T * E
    return ((4 + 2 * nw) * TE + 2 * T, 3 * TE + 3 * T, (2 + nw) * TE)


def _routing_core(p) -> Triple:
    # RoutingClaim alone: m/rt/mrt/gap (TE) + rstar/r_chosen (T); five linear
    # families F1-F5 (L + T + T + L + T) and two quad families (m*m, m*rt).
    T, E = p["T"], p["E"]
    TE = T * E
    return (4 * TE + 2 * T, 2 * TE + 3 * T, 2 * TE)


def _word_extract(p) -> Triple:
    # N committed word variables; one linear recomposition family (L cids),
    # no quads. Range checks live on the companion RangeWordClaims.
    L = _length(p)
    return (p.get("n_words", 1) * L, L, 0)


def _range_word(p) -> Triple:
    # Quad-only: (alpha - x)*z = 1 per slot; z is the committed aux.
    L = _length(p)
    return (L, 0, L)


def _table_settle(p) -> Triple:
    # TableSettlement: produces the phase-2 w vector (length T_LEN, its W);
    # T_LEN per-row product constraints + 1 sum identity. The shared mult
    # commitment is counted as a run-input variable, not claim W.
    return (p["T_LEN"], p["T_LEN"] + 1, 0.0)


def _freivalds_combine(p) -> Triple:
    T, E, F = p["T"], p["E"], p["F"]
    ET = E * T
    return (T * F + 4 * ET + T, 3 * ET + 2 * T, ET)


# Canonical name -> formula. Prover dataclass names are aliased below so the
# tape extractor and the synthetic builders hit the same rows.
def _routed_projected(p):
    """RoutedProjectedMatmulClaim (prover/routed_projected.py): one routed
    expert matmul under the projected protocol. Own witness: Y (T*J raw
    routed output; the following rescale_claim binds the scaled output),
    P (E*K projected weights), Q and H (T*K each), yr (T), f_y/f_u/f_p
    (E each). Constraints per routed_compile: E*K (P = W rho) + T
    (yr = Y rho) + T (sum_k H = yr) + E (f_y) + E (f_u) + 1 (final
    scalar). Quads: H = X*Q (T*K) + f_p = f_u*f_y (E). Verified against
    analysis/routed_projected_4h_model.py's L_ROUTE/Q_ROUTE ledger."""
    T, K, J, E = p["T"], p["K"], p["J"], p["E"]
    W = T * J + E * K + 2 * T * K + T + 3 * E
    cids = E * K + 2 * T + 2 * E + 1
    Q = T * K + E
    return (float(W), float(cids), float(Q))


def _rescale_claim(p):
    """RescaleClaim (prover/rescale_claim.py): the standalone signed-floor
    rescale that follows every routed raw output. Own witness: x, x_low,
    x_shifted, z_low, z_shifted (5L); two linears (2L cids); two per-slot
    LogUp-inverse quad families (2L)."""
    L = _length(p)
    return (5.0 * L, 2.0 * L, 2.0 * L)


def _lincomb(p):
    """LinCombClaim (prover/claims.py lincomb_compile): a public linear
    combination over existing variables — no own witness, one
    L2_IdentityScalar cid per slot ([base, base+L)), no quads. Emitted by
    the token-binding path only (prover/token_binding.py), never by the
    Maverick/Llama demos."""
    L = _length(p)
    return (0.0, float(L), 0.0)


_FORMULAS = {
    "lincomb": _lincomb,
    "matmul": _matmul,
    "routed_projected": _routed_projected,
    "rescale_claim": _rescale_claim,
    "rmsnorm": _rmsnorm,
    "softmax": _softmax,
    "silu": _silu,
    "hadamard": _hadamard,
    "rope": _rope,
    "add": _add,
    "embed_lookup": _embed_lookup,
    "ptlookup": _ptlookup,
    "routing": _routing,
    "routing_core": _routing_core,
    "word_extract": _word_extract,
    "range_word": _range_word,
    "table_settle": _table_settle,
    "freivalds_combine": _freivalds_combine,
}

_ALIASES = {
    "MatmulClaim": "matmul",
    "RmsNormClaim": "rmsnorm",
    "SoftmaxClaim": "softmax",
    "SiluClaim": "silu",
    "HadamardClaim": "hadamard",
    "RoPEClaim": "rope",
    "AddClaim": "add",
    "EmbeddingLookupClaim": "embed_lookup",
    "PairedTlookupClaim": "ptlookup",
    "RoutingClaim": "routing_core",   # word/range aux arrive as own claims
    "WordExtractionClaim": "word_extract",
    "RangeWordClaim": "range_word",
    "TableSettlement": "table_settle",
    "FreivaldsCombineClaim": "freivalds_combine",
    "RoutedProjectedMatmulClaim": "routed_projected",
    "RescaleClaim": "rescale_claim",
    "LinCombClaim": "lincomb",
}

_warned: set = set()


def canonical(claim_type: str) -> str:
    return _ALIASES.get(claim_type, claim_type)


def cost(claim_type: str, params: Dict, w_hint: float = 0.0) -> Triple:
    """(W, cids, Q) for one claim. `w_hint` (e.g. exact slots from the tape)
    backs the fallback for types without a formula."""
    name = canonical(claim_type)
    fn = _FORMULAS.get(name)
    if fn is None:
        if name not in _warned:
            _warned.add(name)
            print(f"[claimcosts] no formula for claim type '{claim_type}'; "
                  f"using fallback (W_hint, W_hint, 0)", file=sys.stderr)
        return (w_hint, w_hint, 0.0)
    return fn(params)
