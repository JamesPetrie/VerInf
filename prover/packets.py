"""
Linear band templates (Level 2 packets).

Each dataclass describes one linear band — a (variable, constraint pattern)
edge: which contiguous constraint ids a variable's slots feed and with what
coefficients, as closed-form index math documented per class. The q_lin fold
LOWERS each template to a CUDA launch pack (core.py `_lower_band`): ten kinds
to the 24-slot descriptor interpreted by `k_interp_band`, the four irregular
kinds (causal ×2, embed, rope-x) to bespoke kernels. Adding a new kind is one
dataclass + one lowering entry (and, for a new index shape, a kernel).

The retired torch expanders (the pre-kernel fold path and the kernels'
one-time bit-equality oracle, `LIGERO_KERNEL_CHECK`) were deleted after gating
the kernels — resurrect from git history at need. Template metadata is
independent of nnz — packets reference Variable layouts and public inputs
only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from cuda_primitives import P


@dataclass(frozen=True, slots=True)
class L2_IdentityScalar:
    """Identity slot mapping with a scalar coefficient.

    Covers any Level 2 contribution where witness row r holds slots that
    map 1:1 to a contiguous range of constraint ids, with a single scalar
    coefficient applied uniformly. Used by AddClaim's a/b/c, Embedding-
    Lookup's x side, TableSettlement's mult side, and most "scalar coef"
    linear constraints in the codebase.

    For witness row r:
      row_off          = r - var_row_start
      n_slots          = clamp(L − row_off · ELL, 0, ELL)
      constraint_id[s] = base + row_off · ELL + s   for s in [0, n_slots)
      target_slot[s]   = s
      coef[s]          = coef                       (scalar, broadcast)
    """
    base: int             # constraint_id_base assigned by _compile_all
    var_row_start: int    # row_start of the participating Variable
    L: int                # total Level 2 length (for clipping last row)
    coef: int             # scalar coefficient mod P


@dataclass(frozen=True, slots=True)
class L2_PerSlotVector:
    """Identity slot mapping with per-slot coefficient from a public vector.

    For witness row r of variable v:
      row_off          = r − var_row_start
      flat[s]          = row_off · ELL + s         for s ∈ [0, n_slots)
      n_slots          = clamp(L − row_off · ELL, 0, ELL)
      constraint_id[s] = base + flat[s]
      coef[s]          = coef_vec[flat[s]]

    Used by TableSettlement (w side, coef_vec = (α − v) over T_LEN entries).
    Reusable for RoPE cos/sin and any per-slot-vector pattern.
    Storage is O(L) — fine for table-sized vectors; for RoPE at long context,
    consider an on-the-fly variant later.
    """
    base: int
    var_row_start: int
    L: int
    coef_vec: torch.Tensor       # (L,) uint64


@dataclass(frozen=True, slots=True)
class L2_RowSumPerSlotVector:
    """Stride aggregation with per-slot vector coef (cyclic in `stride`):
      flat[s]          = row_off · ELL + s
      constraint_id[s] = base + flat[s] // stride
      coef[s]          = coef_vec[flat[s] % stride]

    Used by:
      - RmsNorm row sums (S = Σ X_sq, u = Σ ρ·x, p = Σ ρ·output)
        with stride = d, coef_vec = ones or −ρ
      - Softmax s1, s2 row sums (stride = M, coef_vec = ones or −1)
    """
    base: int
    var_row_start: int
    L: int                       # total variable length
    stride: int                  # period of coef_vec; cid = base + flat//stride
    coef_vec: torch.Tensor       # (stride,) uint64


@dataclass(frozen=True, slots=True)
class L2_StrideOneToManyScalar:
    """One source slot fans out to `stride` consecutive constraints, scalar
    coef applied to all:
      flat[s]         = row_off · ELL + s
      constraint_id   = [base + flat[s]·stride, base + (flat[s]+1)·stride)
      coef            = scalar (broadcast across the stride entries)

    Used by Softmax's z = c2 − x: slot k of c2 broadcasts to M constraints
    (one for each i in [0, M)) with coef −1.
    """
    base: int
    var_row_start: int
    L: int                       # target variable length (source side)
    stride: int                  # fan-out width per source slot
    coef: int


@dataclass(frozen=True, slots=True)
class L2_TransposeO2MScalar:
    """Transposed fan-out: the source variable is a row-major (rows, cols)
    matrix; the slot at flat f = t·cols + e (t ∈ [0, rows), e ∈ [0, cols))
    contributes to `fan` consecutive constraints in TRANSPOSED (col-major)
    order, scalar coef applied to all:
      constraint_id = base + (f % cols)·rows·fan + (f // cols)·fan + k,
                      k ∈ [0, fan)
      coef          = scalar (broadcast across the fan entries)

    With fan = 1 this is a pure transpose-identity — e.g. FreivaldsCombineClaim's
    expert-major mask pin m_em[e, t] = m[t, e] (source m token-major: rows = T
    tokens, cols = E experts) and its token-major ms_tm transpose. With fan > 1
    each source scalar additionally fans out across a contiguous block of `fan`.
    """
    base: int
    var_row_start: int
    L: int                       # source variable length (rows·cols)
    rows: int                    # T  (source major dim)
    cols: int                    # E  (source minor dim; transpose major)
    fan: int                     # F  (fan-out width per source slot)
    coef: int


# ---- Softmax causal patterns ----
# Causal-softmax z-decomp has L_u = H · SEQ · (SEQ+1)/2 constraints
# (unmasked cells only). For cell (b, j) with b = i_qry·H + h:
#   unmasked iff j ≤ i_qry
#   rank     = H·i_qry·(i_qry+1)/2 + h·(i_qry+1) + j     (closed-form)
#
# z, x, z_high (identity slot mapping over the unmasked subset) →
#   L2_CausalFilteredIdScalar
# c2 (each c2[b] fans out to (i_qry+1) consecutive unmasked constraints) →
#   L2_CausalFilteredC2Stride


@dataclass(frozen=True, slots=True)
class L2_CausalFilteredIdScalar:
    """Causal-filtered identity slot mapping with scalar coef. Used by z, x,
    z_high in the causal softmax z-decomp."""
    base: int
    var_row_start: int
    L: int            # full variable length = B·M
    M: int
    H: int
    coef: int


@dataclass(frozen=True, slots=True)
class L2_CausalFilteredC2Stride:
    """c2 in causal softmax z-decomp. c2[b] fans out to (i_qry+1) unmasked
    constraints starting at rank_start = H·i_qry·(i_qry+1)/2 + h·(i_qry+1).
    Variable fan-out per source slot — handled via ragged repeat_interleave."""
    base: int
    c2_row_start: int
    B: int
    H: int
    coef: int


@dataclass(frozen=True, slots=True)
class L2_EmbedE:
    """E-side of EmbeddingLookup: 1·x[i·d+j] − 1·E[token_ids[i]·d+j] = 0.

    Per witness row r of E:
      row_off       = r − E_row_start
      rows_per_w    = ELL // d                                (requires d | ELL)
      vocab_lo      = row_off · rows_per_w
      hits          = i ∈ [0, SEQ) where token_ids[i] ∈ [vocab_lo, vocab_lo+rows_per_w)
      for each hit i and j ∈ [0, d):
        slot_in_row   = (token_ids[i] − vocab_lo) · d + j
        constraint_id = base + i · d + j
        coef          = P − 1                                  (= −1 mod P)

    token_ids is a public reference (small, shared across all this claim's
    L2_EmbedE packets — not copied per packet).
    """
    base: int
    E_row_start: int
    d: int
    token_ids: torch.Tensor   # (SEQ,) int64 — public


# ---- Matmul Freivalds patterns: LF1 B-side, LF2 A-side, LF3 p/C. ----


@dataclass(frozen=True, slots=True)
class L2_FreivaldsLF1B:
    """B side of MatmulClaim LF1: y[i_k] − Σ_j ρ_head[j]·B[…,j] = 0.

    Decoding B's flat index `f` to (i_k, j):
      transpose_b=True:  f = j·k + i_k        →  j = f//k, i_k = f%k
      transpose_b=False: f = (i_k%K)·H·n + (i_k//K)·n + j
                         →  j    = f % n
                            rest = f // n
                            h    = rest % H
                            r    = rest // H        (= i_k % K)
                            i_k  = h·K + r
    constraint_id = base + i_k
    coef          = neg_rho[head_of_k(i_k)·n + j]   (= −ρ_head[j])
    """
    base: int
    B_row_start: int
    k: int                  # H · K
    n: int
    H: int
    K: int                  # k // H (head_dim)
    transpose_b: bool
    neg_rho: torch.Tensor   # (H·n,) uint64 — shared across this claim's LF1B packets


@dataclass(frozen=True, slots=True)
class L2_FreivaldsLF2A:
    """A side of MatmulClaim LF2: u[i_k] − Σ_i λ_head[i]·A[i, h, r] = 0.

    A is laid out row-major as A[i, h, r] flat at i·k + i_k:
      i_k = f % k
      i   = f // k
    constraint_id = base + i_k
    coef          = neg_lam[head_of_k(i_k)·m + i]   (= −λ_head[i])
    """
    base: int
    A_row_start: int
    k: int
    m: int
    H: int
    K: int
    neg_lam: torch.Tensor   # (H·m,) uint64


@dataclass(frozen=True, slots=True)
class L2_StrideManyToOneScalar:
    """Stride-aggregation scalar pattern: `stride` consecutive flat slots
    of `var` collapse onto one constraint:
      slot s at flat f = row_off·ELL + s   →   cid = base + f // stride
      coef                                  =   scalar
    Slots past `L` (the participating-variable's true length) are skipped.

    Used by:
      - MatmulClaim LF3 p side:   stride = K (head_dim), coef = 1, base = LF3_base
        (each block of K p-slots collapses onto one head's LF3 constraint)
      - TableSettlement sum identity: each z (stride = z.length, coef = 1) and
        w (stride = T_LEN, coef = −1) collapse onto the single sum constraint
    """
    base: int
    var_row_start: int
    L: int
    stride: int
    coef: int


@dataclass(frozen=True, slots=True)
class L2_FreivaldsLF3C:
    """C side of MatmulClaim LF3 (one constraint per head):
      Σ_r p[h·K + r] − Σ_{i,j} λ[h·m+i]·ρ[h·n+j]·C[i, h, j] = 0

    C is laid out row-major as C[i, h, j] at flat f = i·H·n + h·n + j:
      j     = f % n
      rest  = f // n
      h     = rest % H
      i     = rest // H
    constraint_id = base + h
    coef          = −λ[h·m + i] · ρ[h·n + j]  (computed on the fly per slot)

    Storing λ and ρ separately (not the outer product) keeps metadata
    O(H·(m + n)) instead of O(H·m·n) — trivially scaling.
    """
    base: int
    C_row_start: int
    m: int
    n: int
    H: int
    L: int                   # total = m · H · n
    lam: torch.Tensor        # (H·m,) uint64 — shared across this claim's LF3C packets
    rho: torch.Tensor        # (H·n,) uint64


# ---- RoPE patterns: x and x_rot sides of the rotation. ----
# RoPE binds two constraints per rotation pair t = (seq, h, k_in_pair):
#   eq1 (cid 2t):   x_rot[lo[t]] − c·x[lo[t]] + s·x[hi[t]] = 0
#   eq2 (cid 2t+1): x_rot[hi[t]] − s·x[lo[t]] − c·x[hi[t]] = 0
# where lo[t] = seq·H·d_h + h·d_h + k_in_pair, hi[t] = lo[t] + d_h/2.


@dataclass(frozen=True, slots=True)
class L2_RoPEXRot:
    """x_rot side: each cell contributes once to its constraint with coef 1.

    For flat f in x_rot:
      seq        = f // (H · d_h)
      h          = (f // d_h) % H
      k          = f % d_h
      half       = d_h // 2
      e_self     = k // half           (0 → lo half, 1 → hi half)
      k_in_pair  = k % half
      pair_t     = seq · H · half + h · half + k_in_pair
      cid        = base + 2 · pair_t + e_self
      coef       = 1
    """
    base: int
    x_rot_row_start: int
    SEQ: int
    H: int
    d_h: int
    L: int                       # = SEQ · H · d_h


@dataclass(frozen=True, slots=True)
class L2_RoPEX:
    """x side: each cell contributes to TWO constraints (eq1 and eq2 of its
    pair). Per flat f in x (decoded as in L2_RoPEXRot):
      pair_t   = seq · H · half + h · half + k_in_pair
      coef_idx = seq · half + k_in_pair          (into cos_t/sin_t)
      c        = cos_t[coef_idx];   s = sin_t[coef_idx]

      e_self = 0 (lo):  to cid 2·pair_t   coef = −c
                         to cid 2·pair_t+1 coef = −s
      e_self = 1 (hi):  to cid 2·pair_t   coef = +s
                         to cid 2·pair_t+1 coef = −c
    """
    base: int
    x_row_start: int
    SEQ: int
    H: int
    d_h: int
    L: int                       # = SEQ · H · d_h
    cos_t: torch.Tensor          # (SEQ · d_h/2,) uint64
    sin_t: torch.Tensor          # (SEQ · d_h/2,) uint64


