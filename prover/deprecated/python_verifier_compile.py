"""DEPRECATED — the constraint-compile half of the old standalone Python verifier.

This is the Python `REAL_COMPILE` / `compile_claims` that the Rust verifier
(verifier-rs) was bit-exact ported from. Nothing in the live prover or verifier
uses it; it survives only as the retired compile-parity difftest oracle
(test_compile_parity.py / dump_compile_parity.py, here in deprecated/). The Rust
verifier is the single source of truth for the constraint compile; end-to-end
ACCEPT is the bit-exactness check the system relies on.

Extracted from protocol.py (P5c); imports the trusted primitives it needs.
"""
from dataclasses import dataclass
from typing import Callable, List, Tuple

from protocol import (
    P, GLOBAL_G, NUM_BLINDING_ROWS, Config,
    add, sub, mul, inv, poly_eval, lagrange, eval_zeta_form,
    _seed_bytes, challenge, random_columns, op_vec, round_seeds,
    _is_var, _nrows, _obj_vars, _distinct_tables,
)


@dataclass
class Quadratic:
    """For c in [0, n):  w[x_row][c] · w[y_row][c] + a · w[z_row][c] == b."""
    x_row: int
    y_row: int
    z_row: int
    a: int
    b: int
    n: int


@dataclass
class Constraints:
    # Linear constraints are never materialized. rows[i] holds this row's
    # contributions as (expand_fn, params) pairs — the pure row expanders below.
    rows: List[List[Tuple[Callable, tuple]]]   # rows[i] -> [(expand_fn, params), ...]
    rhs:  List[Tuple[int, int]]                # SPARSE RHS: (cid, b_cid) for b != 0
    quadratic: List[Quadratic]
    m_total: int                               # total committed rows (incl. blinding)


def expand_identity(params, cfg):
    """params = (cid_base, n, coef). Slot s feeds constraint cid_base+s with a
    scalar coef. Covers copies, residual adds, and the output side of a claim."""
    cid_base, n, coef = params
    for s in range(n):
        yield s, cid_base + s, coef


def expand_weighted(params, cfg):
    """params = (cid_base, coefs). Slot s feeds cid_base+s with per-slot public
    coef coefs[s]. Covers RMSNorm γ and any diagonal / per-slot-weighted row."""
    cid_base, coefs = params
    for s, c in enumerate(coefs):
        yield s, cid_base + s, c


def expand_rope(params, cfg):
    """params = (cid_base, a, b, ca, cb, row_lo, row_hi). Output o (constraint
    cid_base+o) reads global x-slots a[o], b[o] with computed coefs ca[o], cb[o]
    (the cos / ±sin pair). row_lo/row_hi are THIS row's global slot window — the
    offset that handles multi-row x: emit only the terms whose source falls in
    this row (in-row col = global − row_lo). A pair straddling a row boundary is
    emitted from two rows' expanders sharing the same cid; r^T A stitches them."""
    cid_base, a, b, ca, cb, row_lo, row_hi = params
    for o in range(len(a)):
        if row_lo <= a[o] < row_hi:
            yield a[o] - row_lo, cid_base + o, ca[o]
        if row_lo <= b[o] < row_hi:
            yield b[o] - row_lo, cid_base + o, cb[o]


def expand_rowsum(params, cfg):
    """params = (cid_base, stride, coef_vec, flat_lo, n_slots). Slot s (global
    flat = flat_lo+s) feeds constraint cid_base + flat//stride with per-slot
    public coef coef_vec[flat % stride]. Covers any strided aggregation: RMSNorm
    row sums (stride=d), softmax s1/s2 (stride=M), fingerprint (stride=L → all
    slots collapse onto cid_base). coef_vec has length `stride`."""
    cid_base, stride, coef_vec, flat_lo, n_slots = params
    for s in range(n_slots):
        flat = flat_lo + s
        yield s, cid_base + flat // stride, coef_vec[flat % stride]


def expand_stride_o2m(params, cfg):
    """params = (cid_base, stride, coef, flat_lo, n_slots). The fan-OUT dual of
    expand_rowsum: source slot s (global flat = flat_lo+s) feeds `stride`
    consecutive constraints [cid_base + flat·stride, … + (flat+1)·stride), all
    reading THIS slot (col = s) with the same scalar coef. Used by softmax
    z = c2 − x: c2[b] appears in all M of row b's z-constraints (coef −1)."""
    cid_base, stride, coef, flat_lo, n_slots = params
    for s in range(n_slots):
        flat = flat_lo + s
        for t in range(stride):
            yield s, cid_base + flat * stride + t, coef


def expand_transpose_o2m(params, cfg):
    """params = (cid_base, rows, cols, fan, coef, flat_lo, n_slots). Transposed
    fan-out: the source is a row-major (rows, cols) matrix; slot at global flat
    f = t·cols + e feeds `fan` consecutive constraints in TRANSPOSED (col-major)
    order, scalar coef:
        cid = cid_base + (f % cols)·rows·fan + (f // cols)·fan + k,  k ∈ [0, fan)
    Matches packets.py L2_TransposeO2MScalar — MaskedCombineClaim's replicated-
    mask pin m_rep[e][t, j] = m[t, e] (rows = T, cols = E, fan = F)."""
    cid_base, rows_n, cols, fan, coef, flat_lo, n_slots = params
    for s in range(n_slots):
        flat = flat_lo + s
        cid_lo = cid_base + (flat % cols) * (rows_n * fan) + (flat // cols) * fan
        for k in range(fan):
            yield s, cid_lo + k, coef


def expand_causal_id(params, cfg):
    """Causal-filtered identity (causal softmax z/x/z_high). params = (cid_base,
    M, H, coef, flat_lo, n_slots). Cell at flat = b·M+j with b = i_qry·H+h is
    UNMASKED iff j ≤ i_qry; an unmasked cell feeds cid_base + rank with scalar
    coef, where rank = H·i_qry·(i_qry+1)/2 + h·(i_qry+1) + j (closed form; the
    rank enumerates the lower-triangular cells)."""
    cid_base, M, H, coef, flat_lo, n_slots = params
    for s in range(n_slots):
        flat = flat_lo + s
        b, j = flat // M, flat % M
        i_qry, h = b // H, b % H
        if j <= i_qry:
            rank = H * i_qry * (i_qry + 1) // 2 + h * (i_qry + 1) + j
            yield s, cid_base + rank, coef


def expand_causal_c2(params, cfg):
    """Causal-filtered c2 fan-out (causal softmax z = c2 − x). params = (cid_base,
    H, coef, flat_lo, n_slots). c2[b] (b = i_qry·H+h) fans out to (i_qry+1)
    unmasked constraints [rank_start, rank_start + i_qry+1) with scalar coef,
    rank_start = H·i_qry·(i_qry+1)/2 + h·(i_qry+1)."""
    cid_base, H, coef, flat_lo, n_slots = params
    for s in range(n_slots):
        b = flat_lo + s
        i_qry, h = b // H, b % H
        rank_start = H * i_qry * (i_qry + 1) // 2 + h * (i_qry + 1)
        for j_off in range(i_qry + 1):
            yield s, cid_base + rank_start + j_off, coef


def expand_embed(params, cfg):
    """params = (cid_base, d, token_ids, vocab_lo, rows_per_w). E-side of an
    embedding lookup for ONE row of E covering vocab rows [vocab_lo, vocab_lo+
    rows_per_w). For each prompt position i whose token lands in this row, slot
    (token_ids[i]−vocab_lo)·d + j feeds constraint cid_base + i·d + j with coef
    −1 (binding x[i,j] = E[token_ids[i], j]). Pure: token_ids is public."""
    cid_base, d, token_ids, vocab_lo, rows_per_w = params
    vocab_hi = vocab_lo + rows_per_w
    for i, tid in enumerate(token_ids):
        if vocab_lo <= tid < vocab_hi:
            rel = tid - vocab_lo
            for j in range(d):
                yield rel * d + j, cid_base + i * d + j, P - 1


def expand_freivalds_b(params, cfg):
    """LF1 B-side: y[i_k] − Σ_j ρ_head[j]·B[…,j] = 0, cid = base + i_k, coef =
    −ρ[head·n + j]. params = (base, k, n, H, K, transpose_b, neg_rho, flat_lo,
    n_slots); neg_rho is the public (H·n) list of −ρ. Decode of B's flat slot:
      transpose_b:  j = f//k,  i_k = f%k
      else:         j = f%n, rest = f//n, h = rest%H, r = rest//H, i_k = h·K+r"""
    base, k, n, H, K, transpose_b, neg_rho, flat_lo, n_slots = params
    for s in range(n_slots):
        f = flat_lo + s
        if transpose_b:
            j, i_k = f // k, f % k
        else:
            j, rest = f % n, f // n
            i_k = (rest % H) * K + rest // H
        head = i_k // K
        yield s, base + i_k, neg_rho[head * n + j]


def expand_freivalds_a(params, cfg):
    """LF2 A-side: u[i_k] − Σ_i λ_head[i]·A[i,h,r] = 0, cid = base + i_k, coef =
    −λ[head·m + i]. A is row-major A[i,h,r] at f = i·k + i_k. params =
    (base, k, m, H, K, neg_lam, flat_lo, n_slots)."""
    base, k, m, H, K, neg_lam, flat_lo, n_slots = params
    for s in range(n_slots):
        f = flat_lo + s
        i_k, i_outer = f % k, f // k
        head = i_k // K
        yield s, base + i_k, neg_lam[head * m + i_outer]


def expand_freivalds_c(params, cfg):
    """LF3 C-side (one constraint per head): Σ_r p[h·K+r] − Σ_{i,j} λ[h·m+i]·
    ρ[h·n+j]·C[i,h,j] = 0, cid = base + h, coef = −(λ[h·m+i]·ρ[h·n+j]) computed
    per slot. C is row-major C[i,h,j] at f = i·H·n + h·n + j. params =
    (base, m, n, H, lam, rho, flat_lo, n_slots); lam/rho are public lists."""
    base, m, n, H, lam, rho, flat_lo, n_slots = params
    for s in range(n_slots):
        f = flat_lo + s
        j, rest = f % n, f // n
        h, i_outer = rest % H, rest // H
        yield s, base + h, (P - mul(lam[h * m + i_outer], rho[h * n + j])) % P


def _walk_vars(cl):
    """Yield every Variable reachable from a claim: direct fields, List[Variable]
    fields (e.g. WordExtraction.words), and Table fields' mult_var/w_var/z_vars.
    Mirrors core.py:_layout's traversal so row accounting matches the prover."""
    for v in _obj_vars(cl).values():
        if _is_var(v):
            yield v
        elif isinstance(v, (list, tuple)):
            for it in v:
                if _is_var(it):
                    yield it
        elif hasattr(v, "mult_var") and hasattr(v, "w_var"):   # Table
            yield v.mult_var
            yield v.w_var
            for z in v.z_vars:
                yield z


def _m_total_real(claim_list, cfg: Config) -> int:
    top = NUM_BLINDING_ROWS - 1
    for cl in claim_list:
        for it in _walk_vars(cl):
            top = max(top, it.row_start + _nrows(it.length, cfg.ELL) - 1)
    return top + 1


def _emit_id(rows, var, cid_base, coef, cfg):
    """Identity family on `var`: global slot g feeds cid_base+g with scalar coef,
    sliced across the rows var occupies (matches claims.py L2_IdentityScalar)."""
    ell = cfg.ELL
    for ro in range(_nrows(var.length, ell)):
        n_slots = min(ell, var.length - ro * ell)
        rows[var.row_start + ro].append(
            (expand_identity, (cid_base + ro * ell, n_slots, coef)))


def _emit_quad(quad, x, y, z, a_s, b_s, L, cfg):
    """x[i]·y[i] + a_s·z[i] = b_s for i in [0,L), one Quadratic per row chunk."""
    ell = cfg.ELL
    for t in range(_nrows(L, ell)):
        quad.append(Quadratic(x.row_start + t, y.row_start + t, z.row_start + t,
                              a=a_s, b=b_s, n=min(ell, L - t * ell)))


def _emit_rowsum(rows, var, cid_base, stride, coef_vec, cfg):
    """Strided aggregation on `var` (length L): slot at flat f feeds constraint
    cid_base + f//stride with coef coef_vec[f % stride]. coef_vec has length
    `stride`. One expand_rowsum per occupied row. Matches claims.py
    L2_RowSumPerSlotVector — used for rmsnorm S/u/p sums (stride=d) and any
    row-reduction with a per-slot public coefficient."""
    ell = cfg.ELL
    for ro in range(_nrows(var.length, ell)):
        flat_lo = ro * ell
        n_slots = min(ell, var.length - flat_lo)
        rows[var.row_start + ro].append(
            (expand_rowsum, (cid_base, stride, coef_vec, flat_lo, n_slots)))


def _emit_stride_o2m(rows, var, cid_base, stride, coef, cfg):
    """Fan-out: each slot of `var` (length L) feeds `stride` consecutive cids
    with scalar coef. One expand_stride_o2m per occupied row. Matches claims.py
    L2_StrideOneToManyScalar — softmax z = c2 − x (c2[b] → M z-constraints)."""
    ell = cfg.ELL
    for ro in range(_nrows(var.length, ell)):
        flat_lo = ro * ell
        n_slots = min(ell, var.length - flat_lo)
        rows[var.row_start + ro].append(
            (expand_stride_o2m, (cid_base, stride, coef, flat_lo, n_slots)))


def _emit_transpose_o2m(rows, var, cid_base, rows_n, cols, fan, coef, cfg):
    """Transposed fan-out on `var` (a row-major (rows_n, cols) matrix), one
    expand_transpose_o2m per occupied row. Matches claims.py
    L2_TransposeO2MScalar — MaskedCombineClaim's replicated-mask pin."""
    ell = cfg.ELL
    for ro in range(_nrows(var.length, ell)):
        flat_lo = ro * ell
        n_slots = min(ell, var.length - flat_lo)
        rows[var.row_start + ro].append(
            (expand_transpose_o2m, (cid_base, rows_n, cols, fan, coef, flat_lo, n_slots)))


def _emit_causal_id(rows, var, cid_base, M, H, coef, cfg):
    """Causal-filtered identity on `var` (length B·M), one expand_causal_id per
    occupied row. Causal softmax z/x/z_high side."""
    ell = cfg.ELL
    for ro in range(_nrows(var.length, ell)):
        flat_lo = ro * ell
        n_slots = min(ell, var.length - flat_lo)
        rows[var.row_start + ro].append(
            (expand_causal_id, (cid_base, M, H, coef, flat_lo, n_slots)))


def _emit_causal_c2(rows, var, cid_base, H, coef, cfg):
    """Causal-filtered c2 fan-out on `var` (length B), one expand_causal_c2 per
    occupied row. Causal softmax z = c2 − x."""
    ell = cfg.ELL
    for ro in range(_nrows(var.length, ell)):
        flat_lo = ro * ell
        n_slots = min(ell, var.length - flat_lo)
        rows[var.row_start + ro].append(
            (expand_causal_c2, (cid_base, H, coef, flat_lo, n_slots)))


def _emit_lin_combo(rows, target, words, coeffs, L, base, cfg):
    """Linear family 1·target[i] + Σ_n (P−coeffs[n])·words[n][i] = const, cids
    [base, base+L). Mirrors claims.py _emit_lin_csr_idscalar; the RHS const (if
    any) is added separately by the caller."""
    _emit_id(rows, target, base, 1, cfg)
    for w, co in zip(words, coeffs):
        _emit_id(rows, w, base, (P - co % P) % P, cfg)


def _emit_rescale(rows, rhs, quad, cur, cfg, *, high, low, full, shifted,
                  rescale_bits, shift_width, z_low, z_shifted,
                  tight, loose, L):
    """THE rescale contract — the one scale-reducing primitive, shared by every
    op that produces a wide intermediate (matmul, hadamard, rope, silu, rmsnorm,
    softmax). Scale algebra: an op's product is at scale s_lo·2^rescale_bits; this
    requantizes it back to s_lo by extracting the high word.

    The gadget is division-with-remainder, range-checked:

        full   = 2^rescale_bits · high + low      (cids [cur,   cur+L))   high = ⌊full/2^r⌋
        shifted = high + 2^(shift_width−1)         (cids [cur+L, cur+2L))   low  = full mod 2^r
        range(low):     low     ∈ [0, 2^rescale_bits)  — checked vs `tight`  table
        range(shifted): shifted ∈ [0, 2^shift_width)   — checked vs `loose`  table
                        (the offset trick range-checks the SIGNED high word)

    UNIFORM CONTRACT (this is the (A) standardization — see PORT_PLAN/CLEANUP):
      - `high`   = the op's requantized output variable (its `*_full` if it has
                   one, else the visible output).
      - `low`    = the dropped low word; `shifted` = high + half-range offset.
      - `z_low`/`z_shifted` = the LogUp inverse witnesses for the two range checks.
      - `tight`  = the 2^rescale_bits range table  → α for the low check.
      - `loose`  = the 2^shift_width  range table  → α for the shifted check.
        Ops name these tables differently (matmul/hadamard/rope: range_rescale +
        range_output; silu: range_rescale + range_x≡range_w2). Pass whichever two
        tables the op carries — the ROLE (tight-for-low, loose-for-shifted) is the
        invariant, not the field name. Every op uses exactly this pairing.
      - `shift_width` = the loose table's bit width (output_width for most ops,
        width_2 for silu). It sets BOTH the offset 2^(shift_width−1) AND must equal
        the loose table's range.

    Two linear families FIRST (so the quads' cids follow — matches the prover's
    n_added order), then the two range quads. `cur` = next free cid; returns
    cur+2L. Parity-verified via test_rescale.py (hadamard rescale: the prover's
    own prove+verify ACCEPTs AND _emit_rescale byte-matches the prover's compile;
    this also confirmed the F1 retraction)."""
    offset = 1 << (shift_width - 1)
    _emit_lin_combo(rows, full, [high, low], [1 << rescale_bits, 1], L, cur, cfg)
    _emit_lin_combo(rows, shifted, [high], [1], L, cur + L, cfg)
    _add_rhs(rhs, cur, [(L, L, offset)])                       # shifted's +offset
    _emit_quad(quad, low,     z_low,     z_low,     (P - tight.alpha) % P, P - 1, L, cfg)
    _emit_quad(quad, shifted, z_shifted, z_shifted, (P - loose.alpha) % P, P - 1, L, cfg)
    return cur + 2 * L


def _add_rhs(rhs, base, nz):
    """nz = [(offset_within_claim, length, value)] → sparse (cid, b) entries."""
    for off, length, val in nz:
        if val % P:
            for k in range(length):
                rhs.append((base + off + k, val % P))


def _c_add(cl, ch, cfg, rows, rhs, quad, nxt):
    base = nxt[0]; nxt[0] += cl.length
    pr_ = getattr(cl, "public_rhs", None)
    if pr_ is not None:
        # REVEAL pin: 1*a = public_rhs (public constant in the claim).
        _emit_id(rows, cl.a, base, 1, cfg)
        _add_rhs(rhs, base, [(0, cl.length, int(pr_) % P)])
        return
    for var, coef in [(cl.a, 1), (cl.b, 1), (cl.c, P - 1)]:
        _emit_id(rows, var, base, coef, cfg)


def _c_hadamard(cl, ch, cfg, rows, rhs, quad, nxt):
    L = cl.length
    if cl.rescale_bits > 0:
        # Linear rescale families occupy cids [base, base+2L); the product quad
        # targets c_full. QUAD ORDER IS LOAD-BEARING: the combiner s_t is indexed
        # by quad position, so we must match the prover (claims.py emits the
        # product quad FIRST, then the rescale range quads) — not just the set.
        base = nxt[0]; nxt[0] += 2 * L
        _emit_quad(quad, cl.a, cl.b, cl.c_full, P - 1, 0, L, cfg)
        _emit_rescale(rows, rhs, quad, base, cfg,
                      high=cl.c, low=cl.c_low, full=cl.c_full, shifted=cl.c_shifted,
                      z_low=cl.z_c_low, z_shifted=cl.z_c_shifted,
                      rescale_bits=cl.rescale_bits, shift_width=cl.output_width,
                      tight=cl.range_rescale, loose=cl.range_output, L=L)
    else:
        _emit_quad(quad, cl.a, cl.b, cl.c, P - 1, 0, L, cfg)


def _c_embedding(cl, ch, cfg, rows, rhs, quad, nxt):
    ell = cfg.ELL
    SEQ, d = len(cl.token_ids), cl.d
    L = SEQ * d
    assert ell % d == 0 or cl.E.length <= ell, \
        f"embedding requires d | ELL or single-row table (d={d}, ell={ell})"
    base = nxt[0]; nxt[0] += L
    _emit_id(rows, cl.x, base, 1, cfg)
    rows_per_w = ell // d
    tok = tuple(int(t) for t in cl.token_ids)
    for ro in range(_nrows(cl.E.length, ell)):
        rows[cl.E.row_start + ro].append(
            (expand_embed, (base, d, tok, ro * rows_per_w, rows_per_w)))


def _rope_cos_sin_real(cfg_r):
    """c, s integer tables at scale s_x, indexed seq·(d_h/2)+k — the public
    twin of claims.py _rope_cos_sin (both sides must compute these identically)."""
    import math
    half = cfg_r.d_h // 2
    cos, sin = [], []
    for seq in range(cfg_r.SEQ):
        pos = seq + cfg_r.position_offset
        for k in range(half):
            theta = pos / (cfg_r.base ** (2 * k / cfg_r.d_h))
            cos.append(int(round(math.cos(theta) * cfg_r.s_x)) % P)
            sin.append(int(round(math.sin(theta) * cfg_r.s_x)) % P)
    return tuple(cos), tuple(sin)


def _rope_decode(f, H, d_h):
    """flat index → (pair_t, e_self, coef_idx) for the split-half rotation."""
    half = d_h // 2
    seq = f // (H * d_h); h = (f // d_h) % H; k = f % d_h
    e_self = k // half; k_in_pair = k % half
    pair_t = seq * H * half + h * half + k_in_pair
    return pair_t, e_self, seq * half + k_in_pair


def expand_rope_xrot(params, cfg):
    """x_rot side: each slot feeds ONE constraint (cid base+2·pair_t+e_self),
    coef 1. params = (base, H, d_h, flat_lo, n_slots)."""
    base, H, d_h, flat_lo, n_slots = params
    for s in range(n_slots):
        pair_t, e_self, _ = _rope_decode(flat_lo + s, H, d_h)
        yield s, base + 2 * pair_t + e_self, 1


def expand_rope_x(params, cfg):
    """x side: each slot feeds TWO constraints (eq1 = 2·pair_t, eq2 = +1) with
    the cos/±sin pair selected by which half it is. params = (base, H, d_h,
    flat_lo, n_slots, cos, sin)."""
    base, H, d_h, flat_lo, n_slots, cos, sin = params
    for s in range(n_slots):
        pair_t, e_self, ci = _rope_decode(flat_lo + s, H, d_h)
        c, sn = cos[ci], sin[ci]
        if e_self == 0:                      # lo half
            yield s, base + 2 * pair_t,     (P - c) % P
            yield s, base + 2 * pair_t + 1, (P - sn) % P
        else:                                # hi half
            yield s, base + 2 * pair_t,     sn
            yield s, base + 2 * pair_t + 1, (P - c) % P


def _c_rope(cl, ch, cfg, rows, rhs, quad, nxt):
    rc = cl.config
    ell = cfg.ELL
    H, d_h = rc.heads, rc.d_h
    assert d_h % 2 == 0, f"RoPE: d_h must be even, got {d_h}"
    L = rc.SEQ * H * d_h
    # The rotation target is x_rot_full when rescale is on (the rescale block
    # then ties x_rot_full → x_rot), else x_rot directly.
    target = cl.x_rot_full if cl.rescale_bits > 0 else cl.x_rot
    base = nxt[0]; nxt[0] += L
    cos, sin = _rope_cos_sin_real(rc)
    for ro in range(_nrows(target.length, ell)):
        flat_lo = ro * ell
        rows[target.row_start + ro].append(
            (expand_rope_xrot, (base, H, d_h, flat_lo, min(ell, L - flat_lo))))
    for ro in range(_nrows(cl.x.length, ell)):
        flat_lo = ro * ell
        rows[cl.x.row_start + ro].append(
            (expand_rope_x, (base, H, d_h, flat_lo, min(ell, L - flat_lo), cos, sin)))
    if cl.rescale_bits > 0:
        # Rotation occupies cids [base, base+L); rescale block follows at base+L.
        cur = nxt[0]                                  # = base + L
        nxt[0] = _emit_rescale(rows, rhs, quad, cur, cfg,
                               high=cl.x_rot, low=cl.x_rot_low,
                               full=cl.x_rot_full, shifted=cl.x_rot_shifted,
                               z_low=cl.z_x_rot_low, z_shifted=cl.z_x_rot_shifted,
                               rescale_bits=cl.rescale_bits, shift_width=cl.output_width,
                               tight=cl.range_rescale, loose=cl.range_output, L=L)


def _c_silu(cl, ch, cfg, rows, rhs, quad, nxt):
    """SiluClaim: output = silu(x) via sign-magnitude decomposition + a paired
    table lookup + a saturation mux. Mirrors claims.py:silu_compile.

    Linear side: 7 identity-scalar families (no new expander), each L wide at
    cids base+i·L:
      0  x        = magnitude + 2·C            (sign-magnitude link)
      1  magnitude= a0 + b·a1 + b2·a2 + b3·a3 + b4·a4   (5-chunk decomp)
      2  g        = b2·a2 + b3·a3 + b4·a4      (saturation indicator)
      3  key      = T_LEN·sign + a1            (paired-lookup index)
      4  x        = output_sat + C             (saturation value)
      5  y        = output + mux_a − mux_b     (output mux)
      6  pt_u     = key + β·y                  (paired-lookup fingerprint input)
    Then 12 quadratics: 7 pure-arithmetic (sign², C, g·inv_g, is_high gating,
    mux_a/b) + 4 range-check LogUps (a0/a2/a3/a4 vs their range tables) + 1
    paired-table-lookup LogUp (pt_u vs silu_table). α/β come from the tables
    (settled separately by the table loop); silu draws no per-claim challenge.

    Optional input rescale (config.rescale_bits>0): the uniform _emit_rescale
    block on x_in = 2^r·x + x_low, with x_low checked vs range_rescale (tight)
    and x_shifted vs range_x≡range_w2 (loose, shift_width = config.width_2)."""
    L = cl.length
    sc = cl.config
    b, T_LEN = sc.b, sc.T_LEN
    b2, b3, b4 = sc.b_2, sc.b_3, sc.b_4
    beta = cl.silu_table.beta

    base = nxt[0]
    families = [
        (cl.x,         [cl.magnitude, cl.C],                       [1, 2]),
        (cl.magnitude, [cl.a_0, cl.a_1, cl.a_2, cl.a_3, cl.a_4],   [1, b, b2, b3, b4]),
        (cl.g,         [cl.a_2, cl.a_3, cl.a_4],                   [b2, b3, b4]),
        (cl.key,       [cl.sign, cl.a_1],                          [T_LEN, 1]),
        (cl.x,         [cl.output_sat, cl.C],                      [1, 1]),
        (cl.y,         [cl.output, cl.mux_a, cl.mux_b],            [1, 1, P - 1]),
        (cl.pt_u,      [cl.key, cl.y],                             [1, beta]),
    ]
    for i, (target, words, coeffs) in enumerate(families):
        _emit_lin_combo(rows, target, words, coeffs, L, base + i * L, cfg)
    nxt[0] = base + 7 * L

    # 7 pure-arithmetic quads (x·y + a·z = b, a = −1, b = 0).
    _emit_quad(quad, cl.sign,    cl.sign,       cl.sign,    P - 1, 0, L, cfg)  # sign² = sign
    _emit_quad(quad, cl.sign,    cl.x,          cl.C,       P - 1, 0, L, cfg)  # sign·x = C
    _emit_quad(quad, cl.g,       cl.inv_g,      cl.is_high, P - 1, 0, L, cfg)  # g·inv_g = is_high
    _emit_quad(quad, cl.is_high, cl.g,          cl.g,       P - 1, 0, L, cfg)  # is_high·g = g
    _emit_quad(quad, cl.is_high, cl.is_high,    cl.is_high, P - 1, 0, L, cfg)  # is_high² = is_high
    _emit_quad(quad, cl.is_high, cl.y,          cl.mux_a,   P - 1, 0, L, cfg)  # is_high·y = mux_a
    _emit_quad(quad, cl.is_high, cl.output_sat, cl.mux_b,   P - 1, 0, L, cfg)  # is_high·output_sat = mux_b

    # 4 range-check LogUps + 1 paired-table lookup (a = −α, b = −1).
    for var, z, tbl in [(cl.a_0, cl.z_a0, cl.range_b),
                        (cl.a_2, cl.z_a2, cl.range_w2),
                        (cl.a_3, cl.z_a3, cl.range_w3),
                        (cl.a_4, cl.z_a4, cl.range_w4)]:
        _emit_quad(quad, var, z, z, (P - tbl.alpha) % P, P - 1, L, cfg)
    _emit_quad(quad, cl.pt_u, cl.pt_z, cl.pt_z, (P - cl.silu_table.alpha) % P, P - 1, L, cfg)

    if sc.rescale_bits > 0:
        # x_in occupies cids [base+7L, base+8L); x_shifted [base+8L, base+9L).
        nxt[0] = _emit_rescale(rows, rhs, quad, base + 7 * L, cfg,
                               high=cl.x, low=cl.x_low, full=cl.x_in, shifted=cl.x_shifted,
                               z_low=cl.z_x_low, z_shifted=cl.z_x_shifted,
                               rescale_bits=sc.rescale_bits, shift_width=sc.width_2,
                               tight=cl.range_rescale, loose=cl.range_x, L=L)


def _c_rmsnorm(cl, ch, cfg, rows, rhs, quad, nxt):
    """RmsNormClaim: output = x ⊙ broadcast(rsqrt(mean(x²)+ε)). Mirrors
    claims.py:rmsnorm_compile. `ch` = ρ ∈ F^d, the per-claim Freivalds challenge
    (rmsnorm_sample draws challenge_vec(d)); used as the −ρ coef vector in the
    u/p row sums.

    7 linear families, each B-wide (one constraint per token b∈[0,B)), at cids
    base+i·B:
      F1 S_total = S + d·ε         (S_total +1, S −1;  RHS b = d·eps_int)
      F2 y_m1    = y − 1           (y_m1 +1, y −1;      RHS b = −1)
      F3 S       = Σ_i X_sq[b·d+i] (S +1; X_sq via expand_rowsum stride=d, coef −1)
      F4 u       = Σ_i ρ_i·x       (u +1; x   via expand_rowsum stride=d, coef −ρ)
      F5 p       = Σ_i ρ_i·output  (p +1; out via expand_rowsum stride=d, coef −ρ)
      F6 s_lo    = Σ_n stride_n·s_lo_chunks[n]   (s_lo +1; chunks −stride_n)
      F7 s_hi    = Σ_n stride_n·s_hi_chunks[n]   (s_hi +1; chunks −stride_n)
    Quadratics: X_sq=x², q1=y², q2=y_m1², the rsqrt bracket
    (q1·S_total − s_lo = magic ; q2·S_total + s_hi = magic−1, sign-flipped),
    Freivalds y·u=p, and per-chunk slack range checks (vs range_slack).

    Rescale: input (s_in>s) and output (s_out) blocks both wired after F7 via
    _emit_rescale; output rescale makes F5 bind output_full. Parity-verified via
    test_rescale.py (rmsnorm_rescale, both blocks)."""
    sc = cl.config
    B, d = sc.B, sc.d
    L = B * d
    eps_int = sc.eps_int
    magic = sc.magic
    # Output rescale binds the raw product output_full (at scale s²) in F5; an
    # output-rescale block then ties output_full → output (at s_out).
    out_target = cl.output_full if sc.output_rescale_bits > 0 else cl.output
    neg_rho = tuple((P - r % P) % P for r in ch)
    neg_ones_d = (P - 1,) * d
    chunk_strides = [1 << (n * sc.slack_chunk_width) for n in range(sc.slack_n_chunks)]

    base = nxt[0]
    f1 = base
    _emit_id(rows, cl.S_total, f1, 1, cfg); _emit_id(rows, cl.S, f1, P - 1, cfg)
    f2 = base + B
    _emit_id(rows, cl.y_m1, f2, 1, cfg);    _emit_id(rows, cl.y, f2, P - 1, cfg)
    f3 = base + 2 * B
    _emit_id(rows, cl.S, f3, 1, cfg);       _emit_rowsum(rows, cl.X_sq, f3, d, neg_ones_d, cfg)
    f4 = base + 3 * B
    _emit_id(rows, cl.u, f4, 1, cfg);       _emit_rowsum(rows, cl.x, f4, d, neg_rho, cfg)
    f5 = base + 4 * B
    _emit_id(rows, cl.p, f5, 1, cfg);       _emit_rowsum(rows, out_target, f5, d, neg_rho, cfg)
    f6 = base + 5 * B
    _emit_id(rows, cl.s_lo, f6, 1, cfg)
    for n, chunk in enumerate(cl.s_lo_chunks):
        _emit_id(rows, chunk, f6, (P - chunk_strides[n] % P) % P, cfg)
    f7 = base + 6 * B
    _emit_id(rows, cl.s_hi, f7, 1, cfg)
    for n, chunk in enumerate(cl.s_hi_chunks):
        _emit_id(rows, chunk, f7, (P - chunk_strides[n] % P) % P, cfg)
    cur = base + 7 * B

    # QUAD ORDER IS LOAD-BEARING (combiner s_t indexed by quad position): emit the
    # op's own quads (arithmetic + slack range checks) FIRST, so the rescale range
    # quads (_emit_rescale, below) come LAST — matching the prover (claims.py).
    _emit_quad(quad, cl.x,    cl.x,      cl.X_sq,   P - 1, 0,         L, cfg)
    _emit_quad(quad, cl.y,    cl.y,      cl.q1,     P - 1, 0,         B, cfg)
    _emit_quad(quad, cl.y_m1, cl.y_m1,   cl.q2,     P - 1, 0,         B, cfg)
    _emit_quad(quad, cl.q1,   cl.S_total, cl.s_lo,  P - 1, magic,     B, cfg)
    _emit_quad(quad, cl.q2,   cl.S_total, cl.s_hi,  1,     magic - 1, B, cfg)
    _emit_quad(quad, cl.y,    cl.u,      cl.p,      P - 1, 0,         B, cfg)
    alpha_T = cl.range_slack.alpha
    for chunk, z in zip(cl.s_lo_chunks, cl.z_lo_chunks):
        _emit_quad(quad, chunk, z, z, (P - alpha_T) % P, P - 1, B, cfg)
    for chunk, z in zip(cl.s_hi_chunks, cl.z_hi_chunks):
        _emit_quad(quad, chunk, z, z, (P - alpha_T) % P, P - 1, B, cfg)

    # Rescale blocks (after F7 cids; their range quads append AFTER the op quads
    # above). Input: x_in = 2^r·x + x_low, loose = range_slack. Output:
    # output_full = 2^r_out·output + output_low, loose = range_output.
    if sc.rescale_bits > 0:
        cur = _emit_rescale(rows, rhs, quad, cur, cfg,
                            high=cl.x, low=cl.x_low, full=cl.x_in, shifted=cl.x_shifted,
                            z_low=cl.z_x_low, z_shifted=cl.z_x_shifted,
                            rescale_bits=sc.rescale_bits, shift_width=sc.slack_chunk_width,
                            tight=cl.range_rescale, loose=cl.range_slack, L=L)
    if sc.output_rescale_bits > 0:
        cur = _emit_rescale(rows, rhs, quad, cur, cfg,
                            high=cl.output, low=cl.output_low, full=cl.output_full,
                            shifted=cl.output_shifted,
                            z_low=cl.z_output_low, z_shifted=cl.z_output_shifted,
                            rescale_bits=sc.output_rescale_bits, shift_width=sc.output_width,
                            tight=cl.range_output_rescale, loose=cl.range_output, L=L)
    nxt[0] = cur

    # RHS: F1 b = d·ε at [base, base+B); F2 b = −1 at [base+B, base+2B).
    _add_rhs(rhs, base, [(0, B, (d * eps_int) % P), (B, B, P - 1)])


def _c_softmax(cl, ch, cfg, rows, rhs, quad, nxt):
    """SoftmaxClaim — full path: causal masking + saturation mux + input rescale.
    Mirrors claims.py:softmax_compile. softmax y = exp(x − LSE) via a two-table
    bracket pinning the LSE candidate c2: prover commits z = c2 − x, looks
    (z → y_A) / (z+δ → y_B) up in the paired exp tables, sums s1/s2, and the
    tight bracket s1 ≤ s_y < s2 forces c2 to the unique integer LSE.

    F0 z-decomp z = c2 − x: non-causal is L_full identity/stride; causal keeps
    only unmasked cells (j ≤ i_qry) → L_u = H·SEQ·(SEQ+1)/2 constraints, via the
    causal expanders (closed-form rank). SATURATE adds: a Z_max·z_high term in
    F0, the y_*_raw = y_* + mux_y_* mux families (the lookup then targets y_*_raw),
    and 5 extra quads (z_high nonzero-gate + mux + range). CAUSAL also puts a
    per-cell Z_max shift on pt_u_A/B's RHS (masked cells pull the table's zero
    half). Input rescale (s_in>s_x) appended last. β/α from the tables; no
    per-claim challenge."""
    sc = cl.config
    B, M, H = sc.B, sc.M, sc.heads
    L = B * M
    sat, causal = sc.saturate, sc.causal
    rescaling = sc.s_in != 0 and sc.s_in != sc.s_x
    Z_max, s_y = sc.Z_max, sc.s_y
    beta_A, beta_B = cl.exp_A.beta, cl.exp_B.beta
    neg_ones_M = (P - 1,) * M
    y_A_look = cl.y_A_raw if sat else cl.y_A      # lookup target (raw under sat)
    y_B_look = cl.y_B_raw if sat else cl.y_B
    L_u = H * (B // H) * ((B // H) + 1) // 2       # causal unmasked-cell count

    base = nxt[0]
    cur = base
    # ---- F0: z = c2 − x  (+ Z_max·z_high under sat) ----
    if causal:
        _emit_causal_id(rows, cl.z, cur, M, H, 1, cfg)
        _emit_causal_c2(rows, cl.c2, cur, H, P - 1, cfg)
        _emit_causal_id(rows, cl.x, cur, M, H, 1, cfg)
        if sat:
            _emit_causal_id(rows, cl.z_high, cur, M, H, Z_max % P, cfg)
        cur += L_u
    else:
        _emit_id(rows, cl.z, cur, 1, cfg)
        _emit_stride_o2m(rows, cl.c2, cur, M, P - 1, cfg)
        _emit_id(rows, cl.x, cur, 1, cfg)
        if sat:
            _emit_id(rows, cl.z_high, cur, Z_max % P, cfg)
        cur += L
    # ---- [sat] y_*_raw = y_* + mux_y_*  (BEFORE pt_u — the prover's order) ----
    if sat:
        _emit_lin_combo(rows, cl.y_A_raw, [cl.y_A, cl.mux_y_A], [1, 1], L, cur, cfg); cur += L
        _emit_lin_combo(rows, cl.y_B_raw, [cl.y_B, cl.mux_y_B], [1, 1], L, cur, cfg); cur += L
    pt_u_A_base, pt_u_B_base = cur, cur + L
    # ---- F1/F2: pt_u_A/B = z + β·y_look ----
    _emit_lin_combo(rows, cl.pt_u_A, [cl.z, y_A_look], [1, beta_A], L, cur, cfg); cur += L
    _emit_lin_combo(rows, cl.pt_u_B, [cl.z, y_B_look], [1, beta_B], L, cur, cfg); cur += L
    # ---- F3/F4: s1/s2 row sums ----
    _emit_id(rows, cl.s1, cur, 1, cfg); _emit_rowsum(rows, cl.y_A, cur, M, neg_ones_M, cfg); cur += B
    _emit_id(rows, cl.s2, cur, 1, cfg); _emit_rowsum(rows, cl.y_B, cur, M, neg_ones_M, cfg); cur += B
    # ---- F5/F6/F7: bracket + c2_shifted ----
    f5 = cur
    _emit_lin_combo(rows, cl.s1,         [cl.r_lo], [-1], B, cur, cfg); cur += B
    _emit_lin_combo(rows, cl.r_hi,       [cl.s2],   [1],  B, cur, cfg); cur += B
    _emit_lin_combo(rows, cl.c2_shifted, [cl.c2],   [1],  B, cur, cfg); cur += B
    # ---- RHS ---- (cid-indexed → order-independent; emit before the quads)
    # Causal: pt_u_A/B get a per-cell Z_max shift on masked cells (j > i_qry).
    if causal:
        for pt_base in (pt_u_A_base, pt_u_B_base):
            for flat in range(L):
                b, j = flat // M, flat % M
                i_qry = b // H
                if j > i_qry:
                    rhs.append((pt_base + flat, Z_max % P))
    _add_rhs(rhs, f5, [(0, B, s_y % P),
                       (B, B, (P - (s_y + 1)) % P),
                       (2 * B, B, (1 << (sc.aux_chunk_width - 1)) % P)])

    # ---- Quadratics: PT_A/PT_B lookups + 3 bracket range checks (+ 6 sat) ----
    # QUAD ORDER load-bearing: op quads FIRST, then the input-rescale range quads
    # (below) — matching the prover (combiner s_t indexed by quad position).
    _emit_quad(quad, cl.pt_u_A, cl.pt_z_A, cl.pt_z_A, (P - cl.exp_A.alpha) % P, P - 1, L, cfg)
    _emit_quad(quad, cl.pt_u_B, cl.pt_z_B, cl.pt_z_B, (P - cl.exp_B.alpha) % P, P - 1, L, cfg)
    alpha_R = cl.range_aux.alpha
    _emit_quad(quad, cl.c2_shifted, cl.z_c2,   cl.z_c2,   (P - alpha_R) % P, P - 1, B, cfg)
    _emit_quad(quad, cl.r_lo,       cl.z_r_lo, cl.z_r_lo, (P - alpha_R) % P, P - 1, B, cfg)
    _emit_quad(quad, cl.r_hi,       cl.z_r_hi, cl.z_r_hi, (P - alpha_R) % P, P - 1, B, cfg)
    if sat:
        _emit_quad(quad, cl.z_high,  cl.inv_z_high, cl.is_high, P - 1, 0, L, cfg)  # z_high·inv = is_high
        _emit_quad(quad, cl.is_high, cl.z_high,     cl.z_high,  P - 1, 0, L, cfg)  # is_high·z_high = z_high
        _emit_quad(quad, cl.is_high, cl.is_high,    cl.is_high, P - 1, 0, L, cfg)  # is_high² = is_high
        _emit_quad(quad, cl.is_high, cl.y_A_raw,    cl.mux_y_A, P - 1, 0, L, cfg)  # is_high·y_A_raw = mux_y_A
        _emit_quad(quad, cl.is_high, cl.y_B_raw,    cl.mux_y_B, P - 1, 0, L, cfg)  # is_high·y_B_raw = mux_y_B
        _emit_quad(quad, cl.z_high,  cl.z_z_high,   cl.z_z_high,
                   (P - cl.range_z_high.alpha) % P, P - 1, L, cfg)                 # RW[z_high]

    # ---- Input rescale (its range quads append AFTER the op quads above) ----
    if rescaling:
        cur = _emit_rescale(rows, rhs, quad, cur, cfg,
                            high=cl.x, low=cl.x_low, full=cl.x_in, shifted=cl.x_shifted,
                            z_low=cl.z_x_low, z_shifted=cl.z_x_shifted,
                            rescale_bits=sc.rescale_bits, shift_width=16,
                            tight=cl.range_rescale, loose=cl.range_aux, L=L)
    nxt[0] = cur


def _c_matmul(cl, ch, cfg, rows, rhs, quad, nxt):
    """MatmulClaim: C = A·B via double-Freivalds. Mirrors claims.py:matmul_compile.
    `ch` = (ρ, λ): ρ ∈ F^{H·n}, λ ∈ F^{H·m} (matmul_sample), the per-head
    Freivalds challenges (h-major). Aux: y = Bρ, u = λ^T A, p = u⊙y.

    cids [base, base + 2k + H):
      LF1 [base,     base+k):    y[i_k] = Σ_j ρ·B   (y identity +1; B expand_freivalds_b)
      LF2 [base+k,   base+2k):   u[i_k] = Σ_i λ·A   (u identity +1; A expand_freivalds_a)
      LF3 [base+2k,  base+2k+H): Σ_r p[h·K+r] = Σ λ·ρ·C  (p stride-many-to-one over K;
                                 C expand_freivalds_c, one cid per head)
    Quadratic: u[i_k]·y[i_k] = p[i_k]  over k (the Freivalds product), per row.

    heads=1 ⇒ H=1, K=k, single head index 0 (the expanders collapse cleanly).
    Output rescale (rescale_bits>0): LF3 binds the RAW product C_full (the matmul
    output at scale s_a·s_b); the uniform _emit_rescale block then ties C_full →
    C (the visible output at s_out), at cids [base+2k+H, base+2k+H+2·L_out).
    Parity-verified via test_rescale.py (matmul_rescale)."""
    H, K = cl.heads, cl.head_dim
    k, m, n = cl.k, cl.m, cl.n
    L_out = m * H * n
    # When rescaling, Freivalds binds the raw product C_full; else C directly.
    C_fv = cl.C_full if cl.rescale_bits > 0 else cl.C
    rho, lam = [int(v) for v in ch[0]], [int(v) for v in ch[1]]   # public lists
    neg_rho = [(P - v % P) % P for v in rho]
    neg_lam = [(P - v % P) % P for v in lam]
    ell = cfg.ELL

    lf1, lf2, lf3 = nxt[0], nxt[0] + k, nxt[0] + 2 * k
    nxt[0] = lf3 + H

    # LF1: y identity (+1) ; B Freivalds (−ρ).
    _emit_id(rows, cl.y, lf1, 1, cfg)
    for ro in range(_nrows(cl.B.length, ell)):
        flat_lo = ro * ell
        rows[cl.B.row_start + ro].append(
            (expand_freivalds_b, (lf1, k, n, H, K, cl.transpose_b, neg_rho,
                                  flat_lo, min(ell, cl.B.length - flat_lo))))
    # LF2: u identity (+1) ; A Freivalds (−λ).
    _emit_id(rows, cl.u, lf2, 1, cfg)
    for ro in range(_nrows(cl.A.length, ell)):
        flat_lo = ro * ell
        rows[cl.A.row_start + ro].append(
            (expand_freivalds_a, (lf2, k, m, H, K, neg_lam,
                                  flat_lo, min(ell, cl.A.length - flat_lo))))
    # LF3: p stride-many-to-one (K p-slots → one head cid, coef +1) ; C_fv outer (−λ·ρ).
    _emit_rowsum(rows, cl.p, lf3, K, (1,) * K, cfg)
    for ro in range(_nrows(C_fv.length, ell)):
        flat_lo = ro * ell
        rows[C_fv.row_start + ro].append(
            (expand_freivalds_c, (lf3, m, n, H, lam, rho,
                                  flat_lo, min(ell, C_fv.length - flat_lo))))

    # Quadratic: u·y = p over k.
    _emit_quad(quad, cl.u, cl.y, cl.p, P - 1, 0, k, cfg)

    # Output rescale: C_full = 2^r·C + C_low, at cids [base+2k+H, +2·L_out).
    if cl.rescale_bits > 0:
        nxt[0] = _emit_rescale(rows, rhs, quad, nxt[0], cfg,
                               high=cl.C, low=cl.C_low, full=cl.C_full, shifted=cl.C_shifted,
                               z_low=cl.z_C_low, z_shifted=cl.z_C_shifted,
                               rescale_bits=cl.rescale_bits, shift_width=cl.output_width,
                               tight=cl.range_rescale, loose=cl.range_output, L=L_out)


def _pub_list(t):
    """A public table tensor → list[int]. Accepts torch tensors (prover side,
    via .tolist()) or plain sequences (already-host). The values ARE public
    table contents, so reading them is not a trust violation."""
    return [int(v) for v in (t.tolist() if hasattr(t, "tolist") else t)]


def _settle_table(table, cfg, rows, rhs, quad, nxt):
    """LogUp table side for one shared table — emitted ONCE per distinct table
    AFTER all operations (the cross-op part of the proof; not an operation, so
    not a claim). Closes the LogUp argument the per-lookup quads (emitted inside
    each op) opened.

    Per row j in [0,T_LEN):  (α − v[j])·w[j] − mult[j] = 0
        v = T (range) or T + β·T_Y (paired); both public.
        w    → expand_weighted, coef_vec = (α − v)
        mult → identity, coef −1
      cids [base, base+T_LEN).
    Sum identity at cid base+T_LEN:
        Σ_z (every lookup z, coef +1) − Σ_j w[j] (coef −1) = 0
        via expand_rowsum with stride = the variable's own length (collapses
        all its slots onto the single sum cid).
    α/β are read off the (public) settled table object — same values the prover
    sampled. Advances nxt by T_LEN + 1."""
    ell = cfg.ELL
    T = _pub_list(table.T)
    T_LEN = len(T)
    alpha = table.alpha
    if table.T_Y is not None:
        TY = _pub_list(table.T_Y)
        v = [(T[j] + mul(table.beta, TY[j])) % P for j in range(T_LEN)]
    else:
        v = T
    w_coef = tuple((alpha - v[j]) % P for j in range(T_LEN))

    base = nxt[0]
    sum_cid = base + T_LEN
    nxt[0] += T_LEN + 1

    # Per-row product constraints. w-side per-slot (α−v); mult-side −1.
    for ro in range(_nrows(table.w_var.length, ell)):
        lo = ro * ell
        n_slots = min(ell, T_LEN - lo)
        rows[table.w_var.row_start + ro].append(
            (expand_weighted, (base + lo, w_coef[lo:lo + n_slots])))
    _emit_id(rows, table.mult_var, base, P - 1, cfg)

    # Sum identity: each z (+1) and w (−1) collapse onto sum_cid.
    for z in table.z_vars:
        ones = (1,) * z.length
        for ro in range(_nrows(z.length, ell)):
            lo = ro * ell
            n_slots = min(ell, z.length - lo)
            rows[z.row_start + ro].append(
                (expand_rowsum, (sum_cid, z.length, ones, lo, n_slots)))
    neg_ones = (P - 1,) * T_LEN
    for ro in range(_nrows(table.w_var.length, ell)):
        lo = ro * ell
        n_slots = min(ell, T_LEN - lo)
        rows[table.w_var.row_start + ro].append(
            (expand_rowsum, (sum_cid, T_LEN, neg_ones, lo, n_slots)))


def _c_word_extraction(cl, ch, cfg, rows, rhs, quad, nxt):
    """x[i] + shift = Σ_n coeffs[n]·words[n][i] — linear-only. Mirrors
    claims.py word_extract_compile."""
    L = cl.length
    base = nxt[0]; nxt[0] += L
    _emit_id(rows, cl.x, base, 1, cfg)
    for w, co in zip(cl.words, cl.coeffs):
        _emit_id(rows, w, base, (P - co % P) % P, cfg)
    if cl.shift % P:
        _add_rhs(rhs, base, [(0, L, (P - cl.shift % P) % P)])


def _c_range_word(cl, ch, cfg, rows, rhs, quad, nxt):
    """(α − x)·z = 1 per slot — quad-only, no cids. Mirrors claims.py
    range_word_compile."""
    _emit_quad(quad, cl.x, cl.z, cl.z, (P - cl.table.alpha) % P, P - 1, cl.length, cfg)


def _c_routing(cl, ch, cfg, rows, rhs, quad, nxt):
    """Top-1 MoE routing (routing_claim.py routing_compile twin). Families:
    F1 rt − 2^L·r = (E−1−e) [T·E] · F2 Σ_e m = 1 [T] · F3 Σ_e mrt − rstar [T] ·
    F4 gap + rt − rstar(bcast) [T·E] · F5 2^L·r_chosen + Σ(E−1−e)·m − rstar [T].
    Quads: m·m = m, m·rt = mrt. gap's range check is a separate composed
    WordExtraction + RangeWord pair, not part of this claim."""
    T, E = cl.T, cl.E
    L = T * E
    neg1 = P - 1
    two_l = (1 << cl.L_bits) % P
    ones_e = [1] * E
    bonus_e = [E - 1 - e for e in range(E)]
    base = nxt[0]; nxt[0] += 2 * L + 3 * T
    cur = base
    # F1: rt − 2^L·r = (E−1−e)
    _emit_id(rows, cl.rt, cur, 1, cfg)
    _emit_id(rows, cl.r, cur, (P - two_l) % P, cfg)
    _add_rhs(rhs, cur, [(f, 1, E - 1 - (f % E)) for f in range(L)])
    cur += L
    # F2: Σ_e m[t,e] = 1
    _emit_rowsum(rows, cl.m, cur, E, ones_e, cfg)
    _add_rhs(rhs, cur, [(0, T, 1)])
    cur += T
    # F3: Σ_e mrt[t,e] − rstar[t] = 0
    _emit_rowsum(rows, cl.mrt, cur, E, ones_e, cfg)
    _emit_id(rows, cl.rstar, cur, neg1, cfg)
    cur += T
    # F4: gap + rt − rstar(broadcast over E) = 0
    _emit_id(rows, cl.gap, cur, 1, cfg)
    _emit_id(rows, cl.rt, cur, 1, cfg)
    _emit_stride_o2m(rows, cl.rstar, cur, E, neg1, cfg)
    cur += L
    # F5: 2^L·r_chosen + Σ_e (E−1−e)·m[t,e] − rstar = 0
    _emit_id(rows, cl.r_chosen, cur, two_l, cfg)
    _emit_rowsum(rows, cl.m, cur, E, bonus_e, cfg)
    _emit_id(rows, cl.rstar, cur, neg1, cfg)
    # quads AFTER the linear families, in prover order
    _emit_quad(quad, cl.m, cl.m, cl.m, neg1, 0, L, cfg)
    _emit_quad(quad, cl.m, cl.rt, cl.mrt, neg1, 0, L, cfg)


def _emit_rowsum_at(rows, var, cid_base, stride, coef_vec, cfg):
    _emit_rowsum(rows, var, cid_base, stride, coef_vec, cfg)


def _c_freivalds_combine(cl, ch, cfg, rows, rhs, quad, nxt):
    """Freivalds-projected masked combine (routing_claim.py fcombine_compile
    twin). C1 s_em binding [E·T] · C2 m_em transpose pin [T·E] · C3 ms_tm
    transpose pin [T·E] · C4 yr binding [T] · C5 seam Σms − yr [T].
    Quad: ms_em = m_em ⊙ s_em. ch = ρ (length F, from s_op by claim index)."""
    T, E, F = cl.T, cl.E, cl.F
    neg1 = P - 1
    neg_rho = [(P - int(r) % P) % P for r in ch]
    ones_e = [1] * E
    base = nxt[0]; nxt[0] += E * T + 2 * T * E + 2 * T
    cur = base
    # C1
    _emit_id(rows, cl.s_em, cur, 1, cfg)
    for e in range(E):
        _emit_rowsum_at(rows, cl.xs[e], cur + e * T, F, neg_rho, cfg)
    cur += E * T
    # C2
    _emit_id(rows, cl.m_em, cur, 1, cfg)
    _emit_transpose_o2m(rows, cl.m, cur, T, E, 1, neg1, cfg)
    cur += T * E
    # C3
    _emit_id(rows, cl.ms_tm, cur, 1, cfg)
    _emit_transpose_o2m(rows, cl.ms_em, cur, E, T, 1, neg1, cfg)
    cur += T * E
    # C4
    _emit_id(rows, cl.yr, cur, 1, cfg)
    _emit_rowsum_at(rows, cl.y, cur, F, neg_rho, cfg)
    cur += T
    # C5
    _emit_rowsum_at(rows, cl.ms_tm, cur, E, ones_e, cfg)
    _emit_id(rows, cl.yr, cur, neg1, cfg)
    _emit_quad(quad, cl.m_em, cl.s_em, cl.ms_em, neg1, 0, E * T, cfg)


def _c_concat(cl, ch, cfg, rows, rhs, quad, nxt):
    """ConcatClaim: dst = srcs concatenated. dst Identity (-1) over the full
    range; each src Identity at base + its segment offset. b = 0."""
    base = nxt[0]
    L = cl.dst.length
    nxt[0] += L
    _emit_id(rows, cl.dst, base, (P - 1) % P, cfg)
    off = 0
    for v in cl.srcs:
        _emit_id(rows, v, base + off, 1, cfg)
        off += v.length
    assert off == L, "concat segments must cover dst"


def _c_masked_combine(cl, ch, cfg, rows, rhs, quad, nxt):
    """y[t,:] = Σ_e m[t,e]·X_e[t,:] (routing_claim.py combine_compile twin).
    G1 replicated-mask pins [E·T·F] (one transpose-fan-out on m covers every
    expert's block) · G2 Σ_e P_e − y = 0 [T·F]. Quads: m_rep_e·X_e = P_e."""
    T, E, F = cl.T, cl.E, cl.F
    LF = T * F
    neg1 = P - 1
    base = nxt[0]; nxt[0] += E * LF + LF
    for e in range(E):
        _emit_id(rows, cl.m_rep[e], base + e * LF, 1, cfg)
    _emit_transpose_o2m(rows, cl.m, base, T, E, F, neg1, cfg)
    cur = base + E * LF
    for e in range(E):
        _emit_id(rows, cl.prods[e], cur, 1, cfg)
    _emit_id(rows, cl.y, cur, neg1, cfg)
    for e in range(E):
        _emit_quad(quad, cl.m_rep[e], cl.xs[e], cl.prods[e], neg1, 0, LF, cfg)


REAL_COMPILE = {
    "AddClaim":             _c_add,
    "HadamardClaim":        _c_hadamard,
    "EmbeddingLookupClaim": _c_embedding,
    "RoPEClaim":            _c_rope,
    "SiluClaim":            _c_silu,
    "RmsNormClaim":         _c_rmsnorm,
    "SoftmaxClaim":         _c_softmax,
    "MatmulClaim":          _c_matmul,
    "WordExtractionClaim":  _c_word_extraction,
    "RangeWordClaim":       _c_range_word,
    "RoutingClaim":         _c_routing,
    "MaskedCombineClaim":   _c_masked_combine,
    "ConcatClaim":          _c_concat,
    "FreivaldsCombineClaim": _c_freivalds_combine,
}


def _op_challenge(cl, claim_index, s_op):
    """The op challenge for one claim, derived from s_op by index — the shared
    definition the prover mirrors. Returns whatever that claim's compile expects:
      MatmulClaim → (ρ, λ)   lengths H·n, H·m
      RmsNormClaim → ρ       length d
      everything else → None (no per-claim challenge)
    Table settlement α/β are NOT here — they are set on the table objects by
    compile_claims (a table is shared, not owned by one op)."""
    name = type(cl).__name__
    if name == "MatmulClaim":
        return (op_vec(s_op, claim_index, "rho", cl.heads * cl.n),
                op_vec(s_op, claim_index, "lam", cl.heads * cl.m))
    if name == "RmsNormClaim":
        return op_vec(s_op, claim_index, "rho", cl.config.d)
    if name == "FreivaldsCombineClaim":
        return op_vec(s_op, claim_index, "rho", cl.F)
    return None


def compile_claims(claim_list, cfg: Config, s_op=None) -> Constraints:
    """Verifier-side compile of an operation list → Constraints, independent of
    the prover. Two passes:
      1. each operation emits its own constraints (incl. its inline per-lookup
         range/quadratic gadgets);
      2. each distinct LogUp table is settled ONCE, after all ops (the cross-op
         sum identity that closes every lookup).
    cids are numbered ops-then-tables; the prover settles in the same order so
    the two sides agree bit-for-bit. Raises for not-yet-ported op types.

    `s_op` is the round-1 seed. All op challenges (matmul ρ,λ; rmsnorm ρ) and all
    table α/β are derived FROM it by index here — the verifier never receives
    challenge values, it expands the seed exactly as the prover does (op_vec /
    challenge by index). claim_index = position in the settled list (ops, then
    one settlement per distinct table). When s_op is None, ops get None
    challenges (only valid for the challenge-free ops — Add/Hadamard/Embed/RoPE)."""
    m_total = _m_total_real(claim_list, cfg)
    rows: List[List[Tuple[Callable, tuple]]] = [[] for _ in range(m_total)]
    rhs:  List[Tuple[int, int]] = []
    quad: List[Quadratic] = []
    nxt = [0]
    tables = _distinct_tables(claim_list)
    n_ops = len(claim_list)
    # Settle each shared table's α/β from s_op, keyed by its settled-list index
    # (settlements come AFTER the ops). v = T (range) or T+β·T_Y (paired).
    if s_op is not None:
        for k, table in enumerate(tables):
            table.alpha = challenge(s_op, 0, f"op{n_ops + k}:alpha")
            if table.T_Y is not None:
                table.beta = challenge(s_op, 0, f"op{n_ops + k}:beta")
    for ci, cl in enumerate(claim_list):        # pass 1: operations
        name = type(cl).__name__
        fn = REAL_COMPILE.get(name)
        if fn is None:
            raise NotImplementedError(f"compile_claims: {name} not ported yet")
        fn(cl, _op_challenge(cl, ci, s_op) if s_op is not None else None,
           cfg, rows, rhs, quad, nxt)
    for table in tables:                         # pass 2: settle shared tables
        _settle_table(table, cfg, rows, rhs, quad, nxt)
    return Constraints(rows, rhs, quad, m_total)


