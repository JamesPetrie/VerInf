"""
Per-claim protocol implementations.

Each claim type — MatmulClaim, AddClaim, HadamardClaim, RangeWordClaim,
WordExtractionClaim, PairedTlookupClaim, SiluClaim, RmsNormClaim,
SoftmaxClaim, RoPEClaim, EmbeddingLookupClaim — has a
dataclass + sample/compile/aux functions, registered into the
SAMPLE_FNS / COMPILE_FNS / AUX_FNS dicts that the prove/verify framework
in core.py consumes. Tests live in test_claims.py.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import torch

import protocol                       # shared op-challenge derivation (op_vec / challenge)
from core import (
    P, GLOBAL_G,
    Variable, QuadraticConstraint, QuadFamily, abs_slot,
    LigeroConfig,
    SAMPLE_FNS, AUX_FNS, COMPILE_FNS, _build_b_chunk,
    Table, TableSettlement,
)
from packets import (
    L2_IdentityScalar, L2_PerSlotVector, L2_RowSumPerSlotVector, L2_EmbedE,
    L2_FreivaldsLF1B, L2_FreivaldsLF2A,
    L2_StrideManyToOneScalar, L2_StrideOneToManyScalar, L2_FreivaldsLF3C,
    L2_RoPEX, L2_RoPEXRot,
    L2_CausalFilteredIdScalar, L2_CausalFilteredC2Stride,
)
from cuda_primitives import gl_matvec, gl_mul, gl_neg, gl_sub, gl_add, gl_inv_batched


# ===========================================================================
# Toy Ligero parameters.
# ===========================================================================

CFG = LigeroConfig(ELL=8, K_DEG=8, N_LIG=32, T_QUERIES=4)
ELL = CFG.ELL  # convenience for abs_slot calls in compile_fns


# ===========================================================================
# MatmulClaim: Freivalds matmul C = A · B (A is m×k, B is k×n, C is m×n).
# ===========================================================================

M:  int = 2
K:  int = 4
N1: int = 2
N2: int = 3


def matmul_field(A_vals: List[int], B_vals: List[int], m: int, k: int, n: int) -> List[int]:
    """Compute C = A · B mod P. Row-major flat I/O."""
    C = [0] * (m * n)
    for i in range(m):
        for j in range(n):
            s = 0
            for r in range(k):
                s = (s + A_vals[i * k + r] * B_vals[r * n + j]) % P
            C[i * n + j] = s
    return C


def compute_matmul_aux_witness(A_vals, B_vals, rho, lam, m, k, n,
                                transpose_b: bool = False):
    """y = B·ρ (length k); u = λ^T·A (length k); p = u ⊙ y (length k).

    When transpose_b=True, B is committed with shape (n, k) and the claim
    verifies C = A · B^T. The aux witness y becomes y[r] = Σ_j B[j,r]·ρ[j]
    (= the r-th component of B^T·ρ).

    Dispatches on input type: torch.uint64 tensors → gl_matvec on device;
    Python lists → pure-Python loops (toy scale, soundness reference).
    """
    if isinstance(A_vals, torch.Tensor):
        A_t = A_vals.view(m, k).contiguous()
        rho_t = torch.tensor(rho, dtype=torch.uint64, device="cuda")
        lam_t = torch.tensor(lam, dtype=torch.uint64, device="cuda")
        if transpose_b:
            B_t = B_vals.view(n, k).contiguous()
            # y[r] = Σ_j B[j,r]·ρ[j] — i.e., gl_matvec(B^T, ρ) where B is (n,k).
            y = gl_matvec(B_t.T.contiguous(), rho_t)   # (k,)
        else:
            B_t = B_vals.view(k, n).contiguous()
            y = gl_matvec(B_t, rho_t)                  # (k,)
        u = gl_matvec(A_t.T.contiguous(), lam_t)       # (k,)
        p = gl_mul(u, y)
        return y, u, p

    # Convert numpy.uint64 challenge arrays to Python int lists for safe
    # arbitrary-precision arithmetic in the per-element loop below
    # (numpy uint64 silently wraps on multiply).
    rho = [int(v) for v in rho]
    lam = [int(v) for v in lam]
    y = []
    for i in range(k):
        s = 0
        if transpose_b:
            for j in range(n):
                s = (s + B_vals[j * k + i] * rho[j]) % P
        else:
            for j in range(n):
                s = (s + B_vals[i * n + j] * rho[j]) % P
        y.append(s)
    u = []
    for i in range(k):
        s = 0
        for j in range(m):
            s = (s + lam[j] * A_vals[j * k + i]) % P
        u.append(s)
    p = [(u[i] * y[i]) % P for i in range(k)]
    return y, u, p


def compute_matmul_aux_witness_multihead(A_vals, B_vals, rho, lam,
                                          m: int, K: int, n: int, H: int,
                                          transpose_b: bool = False):
    """H-batched aux. Layouts (flat):
        A: (m, H, K)                                      i*H*K + h*K + r
        B: (n, H, K) if transpose_b else (K, H, n)        j*H*K + h*K + r  /  r*H*n + h*n + j
        ρ: (H, n)  λ: (H, m)                              h-major
        y: (H, K)  u: (H, K)  p: (H, K)                   h-major
    Per head: y_h = B_h·ρ_h (or B_h^T·ρ_h with transpose), u_h = λ_h·A_h,
    p_h = u_h ⊙ y_h. Concatenates the H per-head triples h-major."""
    is_torch = isinstance(A_vals, torch.Tensor)
    y_chunks, u_chunks, p_chunks = [], [], []
    for h in range(H):
        rho_h = rho[h * n : (h + 1) * n]
        lam_h = lam[h * m : (h + 1) * m]
        if is_torch:
            # Slice per-head views; .contiguous() materializes a copy so the
            # downstream gl_matvec sees a flat layout it expects.
            A_h = A_vals.view(m, H, K)[:, h, :].contiguous().view(-1)
            if transpose_b:
                B_h = B_vals.view(n, H, K)[:, h, :].contiguous().view(-1)
            else:
                B_h = B_vals.view(K, H, n)[:, h, :].contiguous().view(-1)
        else:
            A_h = [A_vals[i * H * K + h * K + r] for i in range(m) for r in range(K)]
            if transpose_b:
                B_h = [B_vals[j * H * K + h * K + r] for j in range(n) for r in range(K)]
            else:
                B_h = [B_vals[r * H * n + h * n + j] for r in range(K) for j in range(n)]
        y_h, u_h, p_h = compute_matmul_aux_witness(
            A_h, B_h, rho_h, lam_h, m, K, n, transpose_b=transpose_b)
        y_chunks.append(y_h); u_chunks.append(u_h); p_chunks.append(p_h)
    if is_torch:
        return torch.cat(y_chunks), torch.cat(u_chunks), torch.cat(p_chunks)
    y = [v for chunk in y_chunks for v in chunk]
    u = [v for chunk in u_chunks for v in chunk]
    p = [v for chunk in p_chunks for v in chunk]
    return y, u, p


@dataclass
class MatmulClaim:
    """Matmul C = A·B with optional internal output rescale.

    Two modes:
      rescale_bits == 0:  C is the raw matmul output (committed at scale
                          s_a · s_b). Existing behavior, all rescale fields
                          are None. Freivalds verifies C = A·B.
      rescale_bits > 0:   C is the rescaled high word at scale s_out; the
                          raw product C_full = A·B is committed separately
                          at scale s_a·s_b = 2^rescale_bits · s_out. The
                          claim emits:
                            • Freivalds on C_full = A·B  (existing aux y/u/p)
                            • Linear: C_full = (1<<r)·C + C_low
                            • Linear: C_shifted = C + 2^(output_width-1)
                            • Range LogUp on C_low      (tight, 2^r table)
                            • Range LogUp on C_shifted  (loose, 2^output_width
                              table — combined with the offset linear bounds
                              C to signed [-2^(w-1), 2^(w-1)) so signed
                              matmul outputs are admissible).
    The caller sees `C` as their output regardless of mode.

    `transpose_b`: when True, B is committed with shape (n, k) and the
    claim verifies C = A · B^T. The B-indexing in LF1 and y aux witness
    flip from B[r, j] = B[r·n + j] to B[j, r] = B[j·k + r]. Other
    constraints unchanged. Avoids committing a transposed copy for
    Q·K^T-style attention.
    """
    A: Variable
    B: Variable
    C: Variable
    y: Variable
    u: Variable
    p: Variable
    m: int
    k: int
    n: int
    transpose_b: bool = False
    # Multi-head batched matmul. When heads > 1, A/B/C carry a head axis
    # in the middle of their flat layout (h-major within the row):
    #   A:  (m, H, head_dim)  flat at i*H*head_dim + h*head_dim + r
    #   B (transpose_b=True):  (n, H, head_dim) flat at j*H*head_dim + h*head_dim + r
    #   B (transpose_b=False): (head_dim, H, n) flat at r*H*n + h*n + j
    #   C:  (m, H, n)         flat at i*H*n + h*n + j
    # Each head runs an independent matmul; Freivalds aux y/u/p (length
    # H·head_dim) are concatenated h-major; per-head challenges ρ (length
    # H·n) and λ (length H·m) are concatenated h-major. heads=1 reduces
    # to the vanilla case bit-for-bit (head_dim==k, no axis split).
    heads: int = 1
    head_dim: int = 0          # 0 → defaults to k in the factory
    # Optional output rescale (Phase-1 fields are None when no rescale).
    rescale_bits: int = 0
    output_width: int = 24
    C_full: Optional[Variable] = None
    C_low: Optional[Variable] = None
    C_shifted: Optional[Variable] = None
    z_C_low: Optional[Variable] = None
    z_C_shifted: Optional[Variable] = None
    range_rescale: Optional[Table] = None
    range_output: Optional[Table] = None


def matmul_claim(name: str, A: Variable, B: Variable, C: Variable,
                 m: int, k: int, n: int, *,
                 transpose_b: bool = False,
                 heads: int = 1, head_dim: int = 0,
                 rescale_bits: int = 0, output_width: int = 24,
                 C_full: Optional[Variable] = None,
                 C_low: Optional[Variable] = None,
                 C_shifted: Optional[Variable] = None,
                 z_C_low: Optional[Variable] = None,
                 z_C_shifted: Optional[Variable] = None,
                 range_rescale: Optional[Table] = None,
                 range_output: Optional[Table] = None) -> MatmulClaim:
    if head_dim == 0:
        head_dim = k // heads
    assert heads >= 1 and head_dim >= 1, (
        f"matmul_claim: heads={heads}, head_dim={head_dim} must be ≥ 1")
    assert heads * head_dim == k, (
        f"matmul_claim: heads*head_dim ({heads}*{head_dim}={heads*head_dim}) "
        f"must equal k ({k})")
    return MatmulClaim(
        A=A, B=B, C=C,
        y=Variable(f"{name}_y", length=k, phase=2),
        u=Variable(f"{name}_u", length=k, phase=2),
        p=Variable(f"{name}_p", length=k, phase=2),
        m=m, k=k, n=n,
        transpose_b=transpose_b,
        heads=heads, head_dim=head_dim,
        rescale_bits=rescale_bits, output_width=output_width,
        C_full=C_full, C_low=C_low, C_shifted=C_shifted,
        z_C_low=z_C_low, z_C_shifted=z_C_shifted,
        range_rescale=range_rescale, range_output=range_output,
    )


def matmul_sample(c: MatmulClaim, ci: int, s_op) -> Tuple[List[int], List[int]]:
    # Freivalds ρ,λ derived from the round-1 seed by index (shared with the
    # verifier via protocol.op_vec). Multi-head: per-head, h-major, lengths H·n, H·m.
    return (protocol.op_vec(s_op, ci, "rho", c.heads * c.n),
            protocol.op_vec(s_op, ci, "lam", c.heads * c.m))


def matmul_aux_witness(c: MatmulClaim, witness: dict, ch: Tuple[List[int], List[int]]) -> dict:
    rho, lam = ch
    if c.heads == 1:
        y, u, p = compute_matmul_aux_witness(
            witness[c.A], witness[c.B], rho, lam, c.m, c.k, c.n,
            transpose_b=c.transpose_b)
    else:
        y, u, p = compute_matmul_aux_witness_multihead(
            witness[c.A], witness[c.B], rho, lam,
            c.m, c.head_dim, c.n, c.heads, transpose_b=c.transpose_b)
    result = {c.y: y, c.u: u, c.p: p}
    if c.rescale_bits > 0:
        def _t(v):
            if isinstance(v, torch.Tensor): return v.contiguous().view(-1)
            return torch.tensor(v, dtype=torch.uint64, device="cuda")
        c_low_t     = _t(witness[c.C_low])
        c_shifted_t = _t(witness[c.C_shifted])
        α_R = c.range_rescale.alpha
        α_O = c.range_output.alpha
        result[c.z_C_low]     = gl_inv_batched(gl_sub(
            torch.full_like(c_low_t,     α_R), c_low_t))
        result[c.z_C_shifted] = gl_inv_batched(gl_sub(
            torch.full_like(c_shifted_t, α_O), c_shifted_t))
    return result


def matmul_compile(claim: MatmulClaim,
                           ch: Tuple[List[int], List[int]],
                           cfg: LigeroConfig, base: int):
    """Compile MatmulClaim (no rescale yet).

    Emits per-row packets across six variables:
      y (LF1, identity scalar),  B (LF1, FreivaldsLF1B)
      u (LF2, identity scalar),  A (LF2, FreivaldsLF2A)
      p (LF3, stride-many-to-one), C (LF3, FreivaldsLF3C)

    Constraint IDs occupy [base, base + 2·k + H):
      LF1 [base,        base+k):       binds y[i_k] minus B Freivalds row
      LF2 [base+k,      base+2k):      binds u[i_k] minus A Freivalds row
      LF3 [base+2k,     base+2k+H):    one constraint per head, p sum − C outer

    Quadratic constraints emit one per Freivalds row (u·y = p).
    """
    rho, lam = ch
    ell = cfg.ELL
    H, K = claim.heads, claim.head_dim
    k, m, n = claim.k, claim.m, claim.n
    # When rescale is on, LF3 binds the raw product C_full (at scale s_a·s_b);
    # a downstream linear rescales C_full → C at scale s_out.
    C_freivalds = claim.C_full if claim.rescale_bits > 0 else claim.C

    rho_t   = torch.tensor(rho, dtype=torch.uint64, device="cuda")   # (H·n,)
    lam_t   = torch.tensor(lam, dtype=torch.uint64, device="cuda")   # (H·m,)
    neg_rho = gl_neg(rho_t)
    neg_lam = gl_neg(lam_t)

    lf1_base = base
    lf2_base = base + k
    lf3_base = base + 2 * k

    row_pkts: List[Tuple[int, object]] = []

    # ---- LF1 ----
    for row_off in range(claim.y.n_rows(ell)):
        row_pkts.append((claim.y.row_start + row_off,
                          L2_IdentityScalar(base=lf1_base,
                                             var_row_start=claim.y.row_start,
                                             L=k, coef=1)))
    for row_off in range(claim.B.n_rows(ell)):
        row_pkts.append((claim.B.row_start + row_off,
                          L2_FreivaldsLF1B(base=lf1_base,
                                            B_row_start=claim.B.row_start,
                                            k=k, n=n, H=H, K=K,
                                            transpose_b=claim.transpose_b,
                                            neg_rho=neg_rho)))

    # ---- LF2 ----
    for row_off in range(claim.u.n_rows(ell)):
        row_pkts.append((claim.u.row_start + row_off,
                          L2_IdentityScalar(base=lf2_base,
                                             var_row_start=claim.u.row_start,
                                             L=k, coef=1)))
    for row_off in range(claim.A.n_rows(ell)):
        row_pkts.append((claim.A.row_start + row_off,
                          L2_FreivaldsLF2A(base=lf2_base,
                                            A_row_start=claim.A.row_start,
                                            k=k, m=m, H=H, K=K,
                                            neg_lam=neg_lam)))

    # ---- LF3 ----
    for row_off in range(claim.p.n_rows(ell)):
        row_pkts.append((claim.p.row_start + row_off,
                          L2_StrideManyToOneScalar(base=lf3_base,
                                                    var_row_start=claim.p.row_start,
                                                    L=k, stride=K, coef=1)))
    for row_off in range(C_freivalds.n_rows(ell)):
        row_pkts.append((C_freivalds.row_start + row_off,
                          L2_FreivaldsLF3C(base=lf3_base,
                                            C_row_start=C_freivalds.row_start,
                                            m=m, n=n, H=H, L=m * H * n,
                                            lam=lam_t, rho=rho_t)))

    cur = base + 2 * k + H

    # ---- Output rescale (rescale_bits > 0) ----
    # Two identity-scalar families + two range-word quads.
    neg1 = (P - 1) % P
    if claim.rescale_bits > 0:
        L_out = m * H * n
        offset = 1 << (claim.output_width - 1)
        # C_full = (1 << r) · C + C_low
        _emit_lin_csr_idscalar(claim.C_full, [claim.C, claim.C_low],
                                 [1 << claim.rescale_bits, 1], L_out, ell, cur, row_pkts)
        cur += L_out
        # C_shifted = C + 2^(output_width − 1)
        _emit_lin_csr_idscalar(claim.C_shifted, [claim.C], [1], L_out, ell, cur, row_pkts)
        cur += L_out

    # Quadratic: Freivalds u·y = p, length k (one family; rows at expand).
    quads: List[QuadFamily] = [QuadFamily(
        name=f"{claim.C.name}.Q", x_row=claim.u.row_start, y_row=claim.y.row_start,
        z_row=claim.p.row_start, L=k, ell=ell, a=neg1, b=0)]

    # Range LogUps for the rescale chunks (only when rescale_bits > 0).
    nz: List[Tuple[int, int, Any]] = []
    if claim.rescale_bits > 0:
        L_out = m * H * n
        quads += _per_slot_quad(
            f"{claim.C.name}.RW[C_low]", claim.C_low, claim.z_C_low, claim.z_C_low,
            (P - claim.range_rescale.alpha) % P, neg1, L_out, ell)
        quads += _per_slot_quad(
            f"{claim.C.name}.RW[C_shifted]", claim.C_shifted, claim.z_C_shifted, claim.z_C_shifted,
            (P - claim.range_output.alpha) % P, neg1, L_out, ell)
        # C_shifted = C + 2^(output_width − 1) has b = offset uniformly.
        nz.append((2 * k + H + L_out, L_out, 1 << (claim.output_width - 1)))

    return row_pkts, quads, cur - base, _build_b_chunk(cur - base, nz)


COMPILE_FNS[MatmulClaim] = matmul_compile


# ===========================================================================
# LinCombClaim: sum_k coefs[k]·xs[k][i] = rhs[i] (public RHS). Pure linear
# glue for gadget composition (the SHA-256/AES gadgets' recompositions, XOR
# linears, and carry-checked adds — analysis/token-binding.md §12 P2). No aux,
# no challenges, no new witness: every xs[k] is already committed elsewhere.
# Negative coefficients are carried as P - c; rhs is either one value (all
# slots) or one value per slot (e.g. the SHA-256 round constants K[r]).
# ===========================================================================


@dataclass
class LinCombClaim:
    xs: List[Variable]
    coefs: List[int]          # same length as xs, values in [0, P)
    rhs: List[int]            # length 1 (constant) or length `length`
    length: int


def lincomb_sample(c: "LinCombClaim", ci, s_op):
    return None


def lincomb_aux(c: "LinCombClaim", witness: dict, ch) -> dict:
    return {}


def lincomb_compile(claim: "LinCombClaim", _ch, cfg: LigeroConfig, base: int):
    """One L2_IdentityScalar per (variable, witness row); constraint ids
    occupy [base, base+L). The public RHS lands in the b-chunk as runs
    (consecutive equal values compressed)."""
    ell, L = cfg.ELL, claim.length
    assert len(claim.xs) == len(claim.coefs), "xs/coefs length mismatch"
    row_pkts: List[Tuple[int, L2_IdentityScalar]] = []
    for var, coef in zip(claim.xs, claim.coefs):
        assert var.length == L, f"{var.name}: length {var.length} != {L}"
        for row_off in range(var.n_rows(ell)):
            row_pkts.append((var.row_start + row_off,
                              L2_IdentityScalar(base=base,
                                                 var_row_start=var.row_start,
                                                 L=L, coef=int(coef) % P)))
    rhs = claim.rhs
    if len(rhs) == 1:
        runs = [(0, L, int(rhs[0]) % P)]
    else:
        assert len(rhs) == L, f"rhs length {len(rhs)} != {L}"
        runs = []
        i = 0
        while i < L:
            j = i
            while j < L and rhs[j] == rhs[i]:
                j += 1
            runs.append((i, j - i, int(rhs[i]) % P))
            i = j
    return row_pkts, [], L, _build_b_chunk(L, runs)


COMPILE_FNS[LinCombClaim] = lincomb_compile


# ===========================================================================
# AddClaim: c[i] = a[i] + b[i] (elementwise). Pure linear, no aux, no challenges.
# ===========================================================================


@dataclass
class AddClaim:
    a: Variable
    b: Variable
    c: Variable
    length: int
    # public_rhs set => REVEAL pin: assert a == public_rhs (a public constant the
    # verifier reads from the claim), emitting only `a` with that RHS. b/c unused.
    # Used to expose a committed value (e.g. the unexplained-info sum Sz) as a
    # public bound via the existing public-RHS (_build_b_chunk) path — no new
    # claim type, no Merkle open. The value is filled post-witness-pass.
    public_rhs: object = None


def add_sample(c: AddClaim, ci, s_op):
    return None


def add_aux_witness(c: AddClaim, witness: dict, ch) -> dict:
    return {}


def add_compile(claim: AddClaim, _ch, cfg: LigeroConfig, base: int):
    """Compile AddClaim. One L2_IdentityScalar per
    (variable, witness row) — three variables (a, b, c) × ceil(L/ELL) rows.
    Constraint IDs occupy [base, base+L), one per slot in claim order.
    """
    ell, L = cfg.ELL, claim.length
    row_pkts: List[Tuple[int, L2_IdentityScalar]] = []
    if claim.public_rhs is not None:
        # REVEAL pin: 1*a = public_rhs (public). Only `a`; RHS carries the value.
        for row_off in range(claim.a.n_rows(ell)):
            row_pkts.append((claim.a.row_start + row_off,
                              L2_IdentityScalar(base=base,
                                                 var_row_start=claim.a.row_start,
                                                 L=L, coef=1)))
        return row_pkts, [], L, _build_b_chunk(L, [(0, L, int(claim.public_rhs) % P)])
    for var, coef in [(claim.a, 1), (claim.b, 1), (claim.c, (P - 1) % P)]:
        for row_off in range(var.n_rows(ell)):
            row_pkts.append((var.row_start + row_off,
                              L2_IdentityScalar(base=base,
                                                 var_row_start=var.row_start,
                                                 L=L, coef=coef)))
    return row_pkts, [], L, None   # b = 0 throughout


COMPILE_FNS[AddClaim] = add_compile


# ===========================================================================
# ConcatClaim: dst = srcs[0] ‖ srcs[1] ‖ … — segment-stitching with NO new
# packet kind: the dst side is one full-length Identity (coef −1) and each
# src is an Identity whose cid base is shifted by its segment offset. Zero
# challenges, zero quads, b = 0. Used to stitch the hidden-prompt and
# public-continuation embedding segments into the single chain input.
# ===========================================================================
@dataclass(frozen=True, slots=True)
class ConcatClaim:
    srcs: List[Variable]
    dst: Variable

    @property
    def length(self):
        return self.dst.length


def concat_sample(c: ConcatClaim, ci, s_op):
    return None


def concat_aux(c: ConcatClaim, witness: dict, ch) -> dict:
    return {}


def concat_compile(claim: ConcatClaim, _ch, cfg: LigeroConfig, base: int):
    ell = cfg.ELL
    L = claim.dst.length
    assert sum(v.length for v in claim.srcs) == L, "concat segments must cover dst"
    row_pkts: List[Tuple[int, L2_IdentityScalar]] = []
    for row_off in range(claim.dst.n_rows(ell)):
        row_pkts.append((claim.dst.row_start + row_off,
                          L2_IdentityScalar(base=base,
                                             var_row_start=claim.dst.row_start,
                                             L=L, coef=(P - 1) % P)))
    off = 0
    for v in claim.srcs:
        for row_off in range(v.n_rows(ell)):
            row_pkts.append((v.row_start + row_off,
                              L2_IdentityScalar(base=base + off,
                                                 var_row_start=v.row_start,
                                                 L=v.length, coef=1)))
        off += v.length
    return row_pkts, [], L, None   # b = 0 throughout


COMPILE_FNS[ConcatClaim] = concat_compile
SAMPLE_FNS[ConcatClaim] = concat_sample
AUX_FNS[ConcatClaim] = concat_aux


# ===========================================================================
# HadamardClaim: c[i] = a[i] · b[i]. One QuadraticConstraint per ELL-sized row chunk.
# ===========================================================================


@dataclass
class HadamardClaim:
    """Hadamard c[i] = a[i] · b[i] with optional internal output rescale.
    See MatmulClaim's docstring for the rescale-block conventions; the
    only difference is the underlying op (per-slot quadratic vs Freivalds)."""
    a: Variable
    b: Variable
    c: Variable
    length: int
    # Optional output rescale.
    rescale_bits: int = 0
    output_width: int = 16
    c_full: Optional[Variable] = None
    c_low: Optional[Variable] = None
    c_shifted: Optional[Variable] = None
    z_c_low: Optional[Variable] = None
    z_c_shifted: Optional[Variable] = None
    range_rescale: Optional[Table] = None
    range_output: Optional[Table] = None


def hadamard_sample(c: HadamardClaim, ci, s_op):
    return None


def hadamard_compile(claim: HadamardClaim, _ch, cfg: LigeroConfig, base: int):
    """Compile HadamardClaim.

    Quadratic family: a·b − c_target = 0 per slot.
       c_target = c_full when rescale_bits > 0, else c.

    Rescale (optional): two L2_IdentityScalar linear families + two range
    LogUp quads (same shape as legacy)."""
    ell, L = cfg.ELL, claim.length
    neg1 = (P - 1) % P
    c_target = claim.c_full if claim.rescale_bits > 0 else claim.c

    row_pkts: List[Tuple[int, object]] = []
    cur = base
    if claim.rescale_bits > 0:
        offset = 1 << (claim.output_width - 1)
        _emit_lin_csr_idscalar(claim.c_full, [claim.c, claim.c_low],
                                 [1 << claim.rescale_bits, 1], L, ell, cur, row_pkts)
        cur += L
        _emit_lin_csr_idscalar(claim.c_shifted, [claim.c], [1], L, ell, cur, row_pkts)
        cur += L

    quads: List[QuadFamily] = [QuadFamily(
        name="H", x_row=claim.a.row_start, y_row=claim.b.row_start,
        z_row=c_target.row_start, L=L, ell=ell, a=neg1, b=0)]
    nz: List[Tuple[int, int, Any]] = []
    if claim.rescale_bits > 0:
        quads += _per_slot_quad(
            f"{claim.c.name}.RW[c_low]", claim.c_low, claim.z_c_low, claim.z_c_low,
            (P - claim.range_rescale.alpha) % P, neg1, L, ell)
        quads += _per_slot_quad(
            f"{claim.c.name}.RW[c_shifted]", claim.c_shifted, claim.z_c_shifted, claim.z_c_shifted,
            (P - claim.range_output.alpha) % P, neg1, L, ell)
        # c_shifted = c + 2^(output_width-1) → b = offset uniformly.
        # Linear families come BEFORE quads in n_added; layout is c_full (L) then c_shifted (L).
        nz.append((L, L, 1 << (claim.output_width - 1)))
    return row_pkts, quads, cur - base, _build_b_chunk(cur - base, nz)


COMPILE_FNS[HadamardClaim] = hadamard_compile


def hadamard_aux_witness(c: HadamardClaim, witness: dict, ch) -> dict:
    if c.rescale_bits == 0:
        return {}
    def _t(v):
        if isinstance(v, torch.Tensor): return v.contiguous().view(-1)
        return torch.tensor(v, dtype=torch.uint64, device="cuda")
    c_low_t     = _t(witness[c.c_low])
    c_shifted_t = _t(witness[c.c_shifted])
    return {
        c.z_c_low:     gl_inv_batched(gl_sub(
            torch.full_like(c_low_t,     c.range_rescale.alpha), c_low_t)),
        c.z_c_shifted: gl_inv_batched(gl_sub(
            torch.full_like(c_shifted_t, c.range_output.alpha),  c_shifted_t)),
    }


# ===========================================================================
# RangeWordClaim + TableSettlement: shared-table LogUp range check.
#
# Each Tape.range_word(x, table) call records:
#   - one RangeWordClaim (per-slot quadratic (α - x[i])·z[i] = 1)
#   - in-place increment of table.mult_var via lookup_multiplicities_into
#   - the new z_var registered with the table for the sum identity
#
# At prove()/verify() time the Tape prepends one TableSettlement per
# registered Table — its sample_fn samples α (so subsequent RangeWordClaim
# compile_fns can read it), and its compile_fn emits the T_LEN table-side
# constraints plus the cross-claim sum identity tying every z slot to the
# shared w[j] = mult[j]/(α - T[j]).
# ===========================================================================


@dataclass
class RangeWordClaim:
    x: Variable
    z: Variable
    table: Table
    length: int


def range_word_sample(c: RangeWordClaim, ci, s_op):
    return None   # α is owned by the table; sampled by TableSettlement


def range_word_compile(c: RangeWordClaim, _ch, cfg: LigeroConfig, base: int):
    """Compile RangeWordClaim.

    RangeWord has NO linear constraints — only per-slot quadratic
    (α − x[i])·z[i] = 1, expressed as x·z + (−α)·z = −1. Emits the same
    quads as legacy, n_added = 0.
    """
    ell, L = cfg.ELL, c.length
    alpha = c.table.alpha
    neg_alpha, neg1 = (P - alpha) % P, (P - 1) % P
    quads = [QuadFamily(
        name=f"RW[{c.x.name}]", x_row=c.x.row_start, y_row=c.z.row_start,
        z_row=c.z.row_start, L=L, ell=ell, a=neg_alpha, b=neg1)]
    return [], quads, 0, None       # no linear constraints → no b


COMPILE_FNS[RangeWordClaim] = range_word_compile


def range_word_aux(c: RangeWordClaim, witness: dict, _ch) -> dict:
    """z[i] = (α - x[i])^(-1)."""
    x_val = witness[c.x]
    x_t = x_val if isinstance(x_val, torch.Tensor) else torch.tensor(x_val, dtype=torch.uint64, device="cuda")
    x_t = x_t.contiguous().view(-1)
    alpha_t = torch.full_like(x_t, c.table.alpha)
    return {c.z: gl_inv_batched(gl_sub(alpha_t, x_t))}


# ===========================================================================
# WordExtractionClaim: x[i] + shift = Σ_n coeff_n · words[n][i]. One linear
# constraint per slot. Range-checking each word is the caller's job —
# usually via N RangeWordClaim calls against shared range tables.
# ===========================================================================


@dataclass
class WordExtractionClaim:
    """Binds a wide value to a linear combination of narrower committed words:
        x[i] + shift = Σ_n coeffs[n] · words[n][i]
    Coefficients are explicit (no implicit 2^(n·B) default) — the caller picks
    a stride layout that matches their range-check tables (e.g. silu's
    sign-magnitude decomp uses [1, 2^4, 2^18, 2^34, 2^50]).

    Range-checking of individual `words[n]` is the caller's responsibility,
    typically via N RangeWordClaim calls against a shared range table."""
    x: Variable
    words: List[Variable]            # length N, each Variable of length == x.length
    coeffs: List[int]                # explicit per-word coefficient (length N)
    length: int                       # x.length
    shift: int = 0                     # x + shift = Σ coeffs[n] · words[n]


def word_extract_sample(c: WordExtractionClaim, ci, s_op):
    return None


def word_extract_compile(c: WordExtractionClaim, _ch, cfg: LigeroConfig,
                                  base: int):
    """Compile WordExtractionClaim.

    Per slot i: 1·x[i] + Σ_n (P − coeffs[n])·words[n][i] = (P − shift) mod P.
    Linear-only (no quads). Each variable contributes L2_IdentityScalar with
    its own scalar coef; the RHS shift is captured by the b_chunk returned
    to _compile_all.
    """
    ell, L, N = cfg.ELL, c.length, len(c.words)
    row_pkts: List[Tuple[int, object]] = []
    # x side, coef = 1
    for row_off in range(c.x.n_rows(ell)):
        row_pkts.append((c.x.row_start + row_off,
                          L2_IdentityScalar(base=base,
                                             var_row_start=c.x.row_start,
                                             L=L, coef=1)))
    # words[n], coef = (P − coeffs[n]) mod P
    for n, w in enumerate(c.words):
        neg_co = (P - c.coeffs[n] % P) % P
        for row_off in range(w.n_rows(ell)):
            row_pkts.append((w.row_start + row_off,
                              L2_IdentityScalar(base=base,
                                                 var_row_start=w.row_start,
                                                 L=L, coef=neg_co)))
    # b[i] = (P − shift) mod P uniformly when shift != 0; else b = 0.
    b_chunk = _build_b_chunk(L, [(0, L, (P - c.shift % P) % P)] if c.shift % P else [])
    return row_pkts, [], L, b_chunk


COMPILE_FNS[WordExtractionClaim] = word_extract_compile


def word_extract_aux(c: WordExtractionClaim, witness: dict, _ch) -> dict:
    return {}


# ===========================================================================
# PairedTlookupClaim: (x[i], y[i]) ∈ (T, T_Y).
#
# Following design-feasibility.md §B.5 the per-claim work is:
#   u[i] = x[i] + β·y[i]                     (linear, phase-2)
#   (α - u[i]) · z[i] = 1, i.e. u·z + (-α)·z = -1   (quadratic, phase-2)
# z's are added to table.z_vars; the table's shared mult/w and the
# cross-claim sum identity are handled by TableSettlement.
# ===========================================================================


@dataclass
class PairedTlookupClaim:
    x: Variable        # phase-1 input #1, asserted to be in column T of the table
    y: Variable        # phase-1 input #2, asserted to be the matching T_Y entry
    u: Variable        # phase-2, u = (x + shift) + β·y
    z: Variable        # phase-2, z = 1/(α - u)
    table: Table
    length: int
    shift: int = 0     # signed→unsigned shift (design-feasibility §3.4):
                       # effective lookup input is (x + shift), so x can be
                       # signed Q-form while the table indexes from 0


def paired_tlookup_sample(c: PairedTlookupClaim, ci, s_op):
    return None


def paired_tlookup_compile(claim: PairedTlookupClaim, _ch,
                                    cfg: LigeroConfig, base: int):
    """Compile PairedTlookupClaim.

    Linear: u[i] − x[i] − β·y[i] = shift  (3 identity-scalar packets per
            participating-variable row).
    Quadratic: (α − u[i])·z[i] = 1  (one per row chunk, same as legacy).
    """
    ell, L = cfg.ELL, claim.length
    alpha, beta = claim.table.alpha, claim.table.beta
    neg1 = (P - 1) % P
    neg_alpha = (P - alpha) % P
    neg_beta  = (P - beta % P) % P

    row_pkts: List[Tuple[int, object]] = []
    for var, coef in [(claim.u, 1), (claim.x, neg1), (claim.y, neg_beta)]:
        for row_off in range(var.n_rows(ell)):
            row_pkts.append((var.row_start + row_off,
                              L2_IdentityScalar(base=base,
                                                 var_row_start=var.row_start,
                                                 L=L, coef=coef)))

    quads = [QuadFamily(
        name=f"PT[{claim.x.name}]", x_row=claim.u.row_start, y_row=claim.z.row_start,
        z_row=claim.z.row_start, L=L, ell=ell, a=neg_alpha, b=neg1)]
    # b[i] = shift uniformly when shift != 0; else b = 0.
    b_chunk = _build_b_chunk(L, [(0, L, claim.shift % P)] if claim.shift % P else [])
    return row_pkts, quads, L, b_chunk


COMPILE_FNS[PairedTlookupClaim] = paired_tlookup_compile


def paired_tlookup_aux(c: PairedTlookupClaim, witness: dict, _ch) -> dict:
    """u[i] = (x[i] + shift) + β·y[i]; z[i] = (α - u[i])^(-1)."""
    x_val, y_val = witness[c.x], witness[c.y]
    def _t(v):
        return v if isinstance(v, torch.Tensor) else torch.tensor(v, dtype=torch.uint64, device="cuda")
    x_t = _t(x_val).contiguous().view(-1)
    y_t = _t(y_val).contiguous().view(-1)
    beta_t  = torch.full_like(x_t, c.table.beta)
    shift_t = torch.full_like(x_t, c.shift % P)
    u_t = gl_add(gl_add(x_t, shift_t), gl_mul(beta_t, y_t))
    alpha_t = torch.full_like(u_t, c.table.alpha)
    z_t = gl_inv_batched(gl_sub(alpha_t, u_t))
    return {c.u: u_t, c.z: z_t}


# ===========================================================================
# SiluConfig + SiluClaim: full sign-magnitude silu with 2-table lookup and
# saturation multiplexer, compiled directly to CSR + QuadraticConstraints.
#
# Relation: output = silu(x) at Q-format `2^(-r)` quantization, with bounded
# magnitude (|x_signed| < ⌈P/2⌉). Construction in three blocks:
#
#   sign-magnitude link        x = magnitude + 2·C,  C = sign·x
#   5-chunk magnitude decomp   magnitude = a_0 + b·a_1 + b_2·a_2 + b_3·a_3 + b_4·a_4
#                              g = b_2·a_2 + b_3·a_3 + b_4·a_4   (saturation indicator)
#   lookup + multiplexer       key = T_LEN·sign + a_1
#                              y = T_combined[key]   (T_pos ‖ T_neg)
#                              is_high = nonzero?(g)
#                              output_sat = x − C   (= (1−sign)·x)
#                              output = (1−is_high)·y + is_high·output_sat
#                                     = y − is_high·y + is_high·output_sat
#
# Soundness: max magnitude (chunk sum) < ⌈P/2⌉ uniquely determines sign;
# nonzero check on g is sound because the chunk sum can't wrap mod P.
# Strip {x : |x_signed| ≥ max magnitude} is rejected (decomposition fails).
# ===========================================================================


@dataclass(frozen=True, slots=True)
class SiluConfig:
    """Knobs for the silu construction. `b·T_LEN` is the in-range magnitude
    bound (above is saturation). Strides b_2/b_3/b_4 and widths width_2/3/4
    tile bits 18..62 (for production) or smaller (for toy). `r` is the
    Q-format bit shift used to scale T_pos/T_neg from real silu values
    (so the table's "input scale" is `s_x = 1 << r`).

    `s_in` lets the caller commit `x` at a coarser scale than `s_x` (e.g.,
    `s_in = s_x²` straight from a Q3.12 matmul). When `s_in > s_x`, the claim
    internally word-decomposes x_in into (x_low, x), with both range-checked,
    and runs silu on the rescaled `x`."""
    b: int
    T_LEN: int
    b_2: int
    b_3: int
    b_4: int
    width_2: int
    width_3: int
    width_4: int
    r: int
    s_in: int = 0   # 0 → no rescale (s_in == 1<<r implicitly)

    @property
    def s_x(self) -> int:
        return 1 << self.r

    @property
    def rescale_bits(self) -> int:
        if self.s_in == 0 or self.s_in == self.s_x:
            return 0
        ratio = self.s_in // self.s_x
        assert ratio > 0 and (ratio & (ratio - 1)) == 0 and ratio * self.s_x == self.s_in, (
            f"SiluConfig: s_in={self.s_in} must be a power-of-2 multiple of s_x=2^{self.r}")
        return ratio.bit_length() - 1


SILU_TOY = SiluConfig(
    b=2, T_LEN=4,
    b_2=8, b_3=16, b_4=32,
    width_2=1, width_3=1, width_4=1,
    r=0,
)

SILU_14BIT = SiluConfig(
    b=4, T_LEN=1 << 14,
    b_2=1 << 16, b_3=1 << 32, b_4=1 << 48,
    width_2=16, width_3=16, width_4=14,
    r=12,   # s_x = 2^r MUST equal the activation scale S (=2^12). Was 14, a
            # hidden dependency that made silu evaluate silu(x/4). Cascade from S.
)


def _silu_real(x: float) -> float:
    """silu(x) = x · sigmoid(x), numerically stable in both signs."""
    if x >= 0:
        return x / (1.0 + math.exp(-x))
    e = math.exp(x)
    return x * e / (1.0 + e)


def silu_tpos_tneg(cfg: SiluConfig) -> Tuple[List[int], List[int]]:
    """Build the paired silu table values. Both prover and verifier compute
    these identically from the bin centres + math.exp."""
    scale = float(1 << cfg.r) if cfg.r > 0 else 1.0
    T_pos, T_neg = [], []
    for i in range(cfg.T_LEN):
        bin_centre_int = i * cfg.b + cfg.b // 2
        bin_centre_real = bin_centre_int / scale
        T_pos.append(int(round(_silu_real( bin_centre_real) * scale)) % P)
        T_neg.append(int(round(_silu_real(-bin_centre_real) * scale)) % P)
    return T_pos, T_neg


@dataclass
class SiluClaim:
    """High-level relation: output = silu(x) at Q-format `2^(-config.r)`.
    The compile_fn directly emits all CSR rows and quadratic constraints
    that together enforce the sign-magnitude link, 5-chunk magnitude decomp,
    g binding, 2-table lookup, nonzero check, and output multiplexer."""
    # User-facing
    x: Variable
    output: Variable
    length: int
    config: 'SiluConfig'

    # Phase-1 aux witnesses (committed by Tape)
    sign: Variable
    magnitude: Variable
    C: Variable
    a_0: Variable
    a_1: Variable
    a_2: Variable
    a_3: Variable
    a_4: Variable
    g: Variable
    inv_g: Variable
    is_high: Variable
    key: Variable
    output_sat: Variable
    mux_a: Variable
    mux_b: Variable
    y: Variable                # T_combined[key]

    # Phase-2 aux witnesses (computed in silu_aux from challenges)
    pt_u: Variable             # key + β · y
    pt_z: Variable             # 1/(α_pt − pt_u)
    z_a0: Variable             # 1/(α_b  − a_0)
    z_a2: Variable             # 1/(α_w2 − a_2)
    z_a3: Variable             # 1/(α_w3 − a_3)
    z_a4: Variable             # 1/(α_w4 − a_4)

    # Table references (α/β are read off these at compile time)
    silu_table: Table
    range_b: Table
    range_w2: Table
    range_w3: Table
    range_w4: Table

    # Optional rescale plumbing (only when config.rescale_bits > 0).
    x_in: Optional[Variable] = None        # public input at scale s_in
    x_low: Optional[Variable] = None       # low word, tight range-checked
    x_shifted: Optional[Variable] = None   # x + 2^(w-1) (offset trick)
    z_x_low: Optional[Variable] = None     # phase-2 z for x_low
    z_x_shifted: Optional[Variable] = None # phase-2 z for x_shifted (range_x)
    range_rescale: Optional[Table] = None  # 2^rescale_bits table for x_low
    range_x: Optional[Table] = None        # range table for x_shifted
                                           # (typically range_w2 — silu's 16-bit mid table)


def silu_sample(c: SiluClaim, ci, s_op):
    return None        # α/β are sampled by the TableSettlement claims


def silu_aux(c: SiluClaim, witness: dict, _ch) -> dict:
    """Phase-2 witnesses: pt_u, pt_z (paired_tlookup LogUp) and the four
    z_aN (range_word LogUp). All computed once α/β are known."""
    def _t(v):
        if isinstance(v, torch.Tensor): return v.contiguous().view(-1)
        return torch.tensor(v, dtype=torch.uint64, device="cuda")
    key_t = _t(witness[c.key])
    y_t   = _t(witness[c.y])
    a0_t  = _t(witness[c.a_0])
    a2_t  = _t(witness[c.a_2])
    a3_t  = _t(witness[c.a_3])
    a4_t  = _t(witness[c.a_4])

    pt_u = gl_add(key_t, gl_mul(torch.full_like(key_t, c.silu_table.beta), y_t))
    result = {
        c.pt_u: pt_u,
        c.pt_z: gl_inv_batched(gl_sub(torch.full_like(pt_u, c.silu_table.alpha), pt_u)),
        c.z_a0: gl_inv_batched(gl_sub(torch.full_like(a0_t, c.range_b.alpha),  a0_t)),
        c.z_a2: gl_inv_batched(gl_sub(torch.full_like(a2_t, c.range_w2.alpha), a2_t)),
        c.z_a3: gl_inv_batched(gl_sub(torch.full_like(a3_t, c.range_w3.alpha), a3_t)),
        c.z_a4: gl_inv_batched(gl_sub(torch.full_like(a4_t, c.range_w4.alpha), a4_t)),
    }
    if c.config.rescale_bits > 0:
        x_low_t     = _t(witness[c.x_low])
        x_shifted_t = _t(witness[c.x_shifted])
        result[c.z_x_low] = gl_inv_batched(gl_sub(
            torch.full_like(x_low_t, c.range_rescale.alpha), x_low_t))
        result[c.z_x_shifted] = gl_inv_batched(gl_sub(
            torch.full_like(x_shifted_t, c.range_x.alpha), x_shifted_t))
    return result


# ===========================================================================
# Shared compile-fn helpers. Used by silu_compile, rmsnorm_compile, and
# any future direct-CSR claim compiler.
# ===========================================================================


def _per_slot_quad(name: str, x_var: Variable, y_var: Variable, z_var: Variable,
                   a_value: int, b_value: int, L: int, ell: int
                   ) -> List[QuadFamily]:
    """Quadratic constraint x·y + a·z = b, per slot — one QuadFamily (quad
    lift); rows split at expand()."""
    return [QuadFamily(name=name, x_row=x_var.row_start, y_row=y_var.row_start,
                       z_row=z_var.row_start, L=L, ell=ell, a=a_value, b=b_value)]


def _emit_lin_csr_idscalar(target: Variable, words: List[Variable], coeffs: List[int],
                             L: int, ell: int, base: int,
                             row_pkts: List[Tuple[int, object]]) -> None:
    """Helper: emit L2_IdentityScalar packets matching the linear form
    1·target[i] + Σ_n (P − coeffs[n])·words[n][i] = const.

    The RHS constant lives in the caller's b_chunk, so this helper only
    places the LHS contributions."""
    for row_off in range(target.n_rows(ell)):
        row_pkts.append((target.row_start + row_off,
                          L2_IdentityScalar(base=base,
                                             var_row_start=target.row_start,
                                             L=L, coef=1)))
    for word, coef in zip(words, coeffs):
        neg_co = (P - coef % P) % P
        for row_off in range(word.n_rows(ell)):
            row_pkts.append((word.row_start + row_off,
                              L2_IdentityScalar(base=base,
                                                 var_row_start=word.row_start,
                                                 L=L, coef=neg_co)))


def silu_compile(c: SiluClaim, _ch, cfg: LigeroConfig, base: int):
    """Compile SiluClaim.

    Seven base linear families — all _lin_csr without strides — emitted as
    L2_IdentityScalar packets with per-family scalar coefs.

    Optional input rescale: x_in = (1<<r)·x + x_low, plus offset-trick
    x_shifted = x + 2^(width_2 − 1). Two L2_IdentityScalar families + two
    range LogUp quads.

    Quadratics emitted verbatim from legacy."""
    ell, L = cfg.ELL, c.length
    sc = c.config
    b, T_LEN = sc.b, sc.T_LEN
    b_2, b_3, b_4 = sc.b_2, sc.b_3, sc.b_4
    neg1 = (P - 1) % P
    β_pt = c.silu_table.beta

    row_pkts: List[Tuple[int, object]] = []
    families = [
        (c.x,         [c.magnitude, c.C],                     [1, 2]),
        (c.magnitude, [c.a_0, c.a_1, c.a_2, c.a_3, c.a_4],     [1, b, b_2, b_3, b_4]),
        (c.g,         [c.a_2, c.a_3, c.a_4],                  [b_2, b_3, b_4]),
        (c.key,       [c.sign, c.a_1],                        [T_LEN, 1]),
        (c.x,         [c.output_sat, c.C],                    [1, 1]),
        (c.y,         [c.output, c.mux_a, c.mux_b],           [1, 1, P - 1]),
        (c.pt_u,      [c.key, c.y],                           [1, β_pt]),
    ]
    for i, (target, words, coeffs) in enumerate(families):
        _emit_lin_csr_idscalar(target, words, coeffs, L, ell, base + i * L, row_pkts)
    cur = base + 7 * L

    if sc.rescale_bits > 0:
        rescale_offset = 1 << (sc.width_2 - 1)
        _emit_lin_csr_idscalar(
            c.x_in, [c.x, c.x_low], [1 << sc.rescale_bits, 1],
            L, ell, cur, row_pkts); cur += L
        _emit_lin_csr_idscalar(
            c.x_shifted, [c.x], [1], L, ell, cur, row_pkts); cur += L

    quads: List[QuadraticConstraint] = []
    quads += _per_slot_quad("silu.sign²",    c.sign,    c.sign,       c.sign,    neg1, 0, L, ell)
    quads += _per_slot_quad("silu.C",        c.sign,    c.x,          c.C,       neg1, 0, L, ell)
    quads += _per_slot_quad("silu.g_invg",   c.g,       c.inv_g,      c.is_high, neg1, 0, L, ell)
    quads += _per_slot_quad("silu.ish_g",    c.is_high, c.g,          c.g,       neg1, 0, L, ell)
    quads += _per_slot_quad("silu.ish²",     c.is_high, c.is_high,    c.is_high, neg1, 0, L, ell)
    quads += _per_slot_quad("silu.mux_a",    c.is_high, c.y,          c.mux_a,   neg1, 0, L, ell)
    quads += _per_slot_quad("silu.mux_b",    c.is_high, c.output_sat, c.mux_b,   neg1, 0, L, ell)
    quads += _per_slot_quad("silu.RW[a0]",   c.a_0,    c.z_a0, c.z_a0, (P - c.range_b.alpha)  % P, neg1, L, ell)
    quads += _per_slot_quad("silu.RW[a2]",   c.a_2,    c.z_a2, c.z_a2, (P - c.range_w2.alpha) % P, neg1, L, ell)
    quads += _per_slot_quad("silu.RW[a3]",   c.a_3,    c.z_a3, c.z_a3, (P - c.range_w3.alpha) % P, neg1, L, ell)
    quads += _per_slot_quad("silu.RW[a4]",   c.a_4,    c.z_a4, c.z_a4, (P - c.range_w4.alpha) % P, neg1, L, ell)
    quads += _per_slot_quad("silu.PT",       c.pt_u,   c.pt_z, c.pt_z,
                              (P - c.silu_table.alpha) % P, neg1, L, ell)
    if sc.rescale_bits > 0:
        quads += _per_slot_quad(
            "silu.RW[x_low]", c.x_low, c.z_x_low, c.z_x_low,
            (P - c.range_rescale.alpha) % P, neg1, L, ell)
        quads += _per_slot_quad(
            "silu.RW[x_shifted]", c.x_shifted, c.z_x_shifted, c.z_x_shifted,
            (P - c.range_x.alpha) % P, neg1, L, ell)

    # b: 7 base families have b=0. Rescale adds x_in (b=0) then x_shifted
    # (b = rescale_offset = 1 << (width_2−1)). Families occupy [7L..7L+L) and
    # [8L..8L+L) respectively.
    nz: List[Tuple[int, int, Any]] = []
    if sc.rescale_bits > 0:
        nz.append((8 * L, L, 1 << (sc.width_2 - 1)))
    return row_pkts, quads, cur - base, _build_b_chunk(cur - base, nz)


COMPILE_FNS[SiluClaim] = silu_compile


# ===========================================================================
# RmsNormClaim — batched RMSNorm over B rows of length d.
#
# Relation: y[b] = 1/√(mean(x[b]²) + ε), output[b·d+i] = x[b·d+i] · y[b].
#
# Soundness from two parts:
#   1. Algebraic-rsqrt bracket (nonarith-survey §2.10):
#        y² · S_total      ≥ d·s^4               (lower-bracket slack s_lo)
#        (y-1)² · S_total  <  d·s^4               (upper-bracket slack s_hi)
#      where S_total = Σ x² + d·ε. Pins y to one integer step. No rsqrt
#      lookup table — only byte-range LogUps on the slacks.
#   2. Freivalds-folded broadcast multiply (mirrors MatmulClaim):
#        ρ ∈ F^d sampled by verifier
#        u[b] := Σ_i ρ_i · x[b·d+i],   p[b] := Σ_i ρ_i · output[b·d+i]
#        y[b] · u[b] = p[b]                       (one quad per batch row)
#      Avoids committing y_broadcast (B·d slots); collapses B·d Hadamard
#      slots into B.
# ===========================================================================


def _chunk_widths(total_bits: int) -> List[int]:
    """Split a range window of `total_bits` into 16-bit chunks plus one
    narrower top chunk (strides 2^{16n}). Every chunk is range-checked
    against a table of exactly its width, so the recomposition covers
    [0, 2^total_bits) tightly — the B.1 width discipline."""
    ws = [16] * (total_bits // 16)
    if total_bits % 16:
        ws.append(total_bits % 16)
    return ws or [1]


# S_total is decomposed into RMS_N_LIMBS limbs of RMS_LIMB_W bits each; the
# carry-chain products q·S_limb and the final accumulator are sized against
# this cap, so the bracket has exact integer semantics for ANY committed
# S_total the limbs admit — no bound on x is assumed (rmsnorm-bracket-fix.md).
#
# LIMB_W is the one knob for the S_total headroom (= row RMS ≲ 2^(cap/2)/√d).
# Its ceiling is set by two asserts: the product q·S_limb must stay < P
# (2·y_width + LIMB_W ≤ 63) and the slack/G2 accumulator must not wrap
# (3·LIMB_W ≲ 59). LIMB_W=18 → cap 2^54 (row RMS ≲ ~460 at s=2^12), well
# above the ~2^44 seen on real activations. The carry-chain high parts
# (g0h/g1h/G2) stay 16-bit-chunked (independent of LIMB_W); only the limbs
# and the carry lows g0l/g1l live in the 2^LIMB_W table.
RMS_LIMB_W = 18
RMS_N_LIMBS = 3
RMS_S_CAP_BITS = RMS_LIMB_W * RMS_N_LIMBS


@dataclass(frozen=True, slots=True)
class RmsNormConfig:
    """B rows of length d each. Integer scales: input `s_in`, internal `s`.
    When `s_in > s` the claim internally word-decomposes each x_in cell into
    (x_low, x) with x_in = (s_in/s)·x + x_low. x_low is range-checked tight
    on a 2^rescale_bits table; x is range-checked on the slack range table
    (loose but combined with the bracket pins x uniquely). The bracket then
    operates on x at scale s, with magic = d·s^4 unchanged.

    All range-window widths (the slack windows, the y window, and the
    limb-carry windows of the wrap-free bracket) are DERIVED from
    (d, s, eps_int) — see the properties below and rmsnorm-bracket-fix.md.
    The verifier derives the same widths independently from the public
    config, so the prover cannot widen a window."""
    B: int
    d: int
    s: int
    eps_int: int
    s_in: int = 0                # 0 → no input rescale (s_in == s implicitly)
    s_out: int = 0               # 0 → no output rescale (output stays at s²)
    output_width: int = 16

    @property
    def magic(self) -> int:
        return self.d * (self.s ** 4)

    # ---- Derived widths for the wrap-free bracket (rmsnorm-bracket-fix.md).
    # Every bracket operand is independently range-bounded so each identity
    # holds over the integers, not merely mod P.

    @property
    def y_max(self) -> int:
        """Largest honest y: the pinned rsqrt at the smallest valid row
        energy S_min = d·eps_int."""
        s_min = self.d * self.eps_int
        assert s_min >= 1, "rmsnorm needs eps_int >= 1 (S_total floor)"
        y = max(1, math.isqrt(self.magic // s_min))
        while y * y * s_min < self.magic:
            y += 1
        return y

    @property
    def y_width(self) -> int:
        """y − 1 is range-checked into [0, 2^y_width). Bounds q1 = y² and
        q2 = (y−1)² below 2^{2·y_width}, which makes every limb product
        q·S_limb < 2^{2·y_width + LIMB_W} wrap-free."""
        w = max(1, (self.y_max - 1).bit_length())
        assert 2 * w + RMS_LIMB_W <= 63, (
            f"rmsnorm y_width={w}: limb products q·S_limb would wrap at "
            f"LIMB_W={RMS_LIMB_W}; raise eps_int, lower the scale, or narrow LIMB_W")
        return w

    @property
    def slack_width(self) -> int:
        """Window for the bracket slacks s_lo/s_hi. Covers the largest honest
        slack for any S_total the limbs admit (< 2^RMS_S_CAP_BITS): one
        bracket step is ≤ 2√(magic·S) + S. The assert is the B.1 rule — a
        wrapped negative slack (≥ P − magic) must not be decomposable."""
        cap = 1 << RMS_S_CAP_BITS
        w = (2 * math.isqrt(self.magic * cap) + cap).bit_length()
        assert self.magic + (1 << w) < P, (
            "rmsnorm slack window would admit wrapped negatives")
        return w

    @property
    def g0h_width(self) -> int:
        """High part of q·S0 (< 2^{2·y_width + LIMB_W}) shifted down LIMB_W;
        independent of LIMB_W."""
        return 2 * self.y_width

    @property
    def g1h_width(self) -> int:
        """High part of q·S1 + g0h, one bit wider than g0h."""
        return 2 * self.y_width + 1

    @property
    def G2_width(self) -> int:
        """Top accumulator of the carry chain: G2 = (magic + slack) >> 2·LIMB_W.
        Its tight range check is what keeps G2·2^{2·LIMB_W} wrap-free in the
        final bracket identity."""
        w = ((self.magic + (1 << self.slack_width)) >> (2 * RMS_LIMB_W)).bit_length()
        assert w + 2 * RMS_LIMB_W <= 62, "rmsnorm G2 window would wrap"
        return max(1, w)

    @property
    def rescale_bits(self) -> int:
        """log2(s_in / s). Zero when no input rescale is performed."""
        if self.s_in == 0 or self.s_in == self.s:
            return 0
        ratio = self.s_in // self.s
        assert ratio > 0 and (ratio & (ratio - 1)) == 0 and ratio * self.s == self.s_in, (
            f"RmsNormConfig: s_in={self.s_in} must be a power-of-2 multiple of s={self.s}")
        return ratio.bit_length() - 1

    @property
    def output_rescale_bits(self) -> int:
        """log2(s² / s_out). Zero when no output rescale is performed.
        Output is naturally at scale s² (= x·y, both at scale s). Rescaling
        by log2(s²/s_out) brings it to s_out (typically s_out = s)."""
        if self.s_out == 0:
            return 0
        s_sq = self.s * self.s
        ratio = s_sq // self.s_out
        assert s_sq == self.s_out * ratio and ratio > 0 and (ratio & (ratio - 1)) == 0, (
            f"RmsNormConfig: s² ({s_sq}) must be a power-of-2 multiple of s_out ({self.s_out})")
        return ratio.bit_length() - 1


@dataclass
class RmsNormClaim:
    # User-facing
    x: Variable                 # length B·d — internal x at scale `s` (= the
                                #   value the bracket operates on). When rescale_bits == 0
                                #   this is the public input; otherwise it's an internal
                                #   commit and x_in is the public input.
    output: Variable            # length B·d (= x ⊙ broadcast(y), verified via Freivalds)
    config: RmsNormConfig

    # Phase-1 aux witnesses (rsqrt bracket)
    X_sq: Variable              # length B·d:  x ⊙ x
    S: Variable                 # length B:    Σ_i X_sq[b·d+i]
    S_total: Variable           # length B:    S + d·ε
    y: Variable                 # length B:    the rsqrt scalars
    y_m1: Variable              # length B:    y − 1
    q1: Variable                # length B:    y · y
    q2: Variable                # length B:    y_m1 · y_m1
    s_lo: Variable              # length B:    q1·S_total − magic
    s_hi: Variable              # length B:    magic − 1 − q2·S_total
    # Word-decomposed slack chunks: s_lo = Σ_n (1<<(16n))·s_lo_chunks[n], same
    # for s_hi. Chunk n is range-checked against a table of exactly
    # _chunk_widths(config.slack_width)[n] bits (16-bit chunks use
    # range_slack; the narrower top chunk uses range_slack_top).
    s_lo_chunks: List[Variable]
    s_hi_chunks: List[Variable]

    # ---- Wrap-free bracket limbs (rmsnorm-bracket-fix.md). All length B.
    # The bracket products y²·S_total and (y−1)²·S_total are assembled from
    # 16-bit limbs of S_total with range-checked carries, so the two bracket
    # identities hold over the INTEGERS (every operand independently bounded),
    # closing the field-wraparound freedom flagged in
    # degrees-of-freedom-review.md §3.
    ym1_chunks: List[Variable]     # y−1 = Σ 2^{16n}·ym1_chunks[n], tight to y_width
    S_limbs: List[Variable]        # S_total = S0 + 2^16·S1 + 2^32·S2, 16-bit each
    lo_H: List[Variable]           # [q1·S0, q1·S1, q1·S2]
    lo_gl: List[Variable]          # [g0l, g1l]: low 16 bits of the carry chain
    lo_g0h_chunks: List[Variable]  # g0h = (q1·S0) >> 16, chunked to g0h_width
    lo_g1h_chunks: List[Variable]  # g1h = (q1·S1 + g0h) >> 16, chunked to g1h_width
    lo_G2_chunks: List[Variable]   # G2 = q1·S2 + g1h, chunked to G2_width
    hi_H: List[Variable]           # the same chain for q2 = (y−1)²
    hi_gl: List[Variable]
    hi_g0h_chunks: List[Variable]
    hi_g1h_chunks: List[Variable]
    hi_G2_chunks: List[Variable]

    # Phase-2 aux witnesses (Freivalds + per-chunk LogUp z's)
    u: Variable                 # length B
    p: Variable                 # length B
    z_lo_chunks: List[Variable] # one z per s_lo chunk
    z_hi_chunks: List[Variable] # one z per s_hi chunk
    # Phase-2 z's for the limb range checks, one per var, same order.
    z_ym1_chunks: List[Variable]
    z_S_limbs: List[Variable]
    z_lo_gl: List[Variable]
    z_lo_g0h: List[Variable]
    z_lo_g1h: List[Variable]
    z_lo_G2: List[Variable]
    z_hi_gl: List[Variable]
    z_hi_g0h: List[Variable]
    z_hi_g1h: List[Variable]
    z_hi_G2: List[Variable]

    # Range table shared with other slack-using claims
    range_slack: Table
    # 2^LIMB_W table for the S_total limbs and the carry lows g0l/g1l.
    range_limb: Table
    # Top-chunk range tables (None when the role's width is a multiple of 16;
    # 16-bit chunks always check against range_slack).
    range_y_top: Optional[Table]
    range_slack_top: Optional[Table]
    range_g0h_top: Optional[Table]
    range_g1h_top: Optional[Table]
    range_G2_top: Optional[Table]

    # Optional rescale plumbing (only present when config.rescale_bits > 0):
    #   x_in: public input at scale config.s_in
    #   x_low: low word of (x_in = 2^r · x + x_low), range-checked tight
    #   x_shifted: x + 2^(w-1)  (offset trick — see CLAIM_SPECS.md)
    #   z_x_low: phase-2 z for x_low's tight range LogUp
    #   z_x_shifted: phase-2 z for x_shifted's range LogUp (range_slack);
    #     combined with the linear identity x_shifted = x + offset, this
    #     bounds x to [-offset, offset), handling signed values.
    #   range_rescale: 2^rescale_bits range table for x_low
    x_in: Optional[Variable] = None
    x_low: Optional[Variable] = None
    x_shifted: Optional[Variable] = None
    z_x_low: Optional[Variable] = None
    z_x_shifted: Optional[Variable] = None
    range_rescale: Optional[Table] = None
    # Optional OUTPUT rescale (only present when config.output_rescale_bits > 0):
    #   `output` is now the rescaled value at scale s_out; `output_full` is
    #   the raw x·y product at scale s² (what the Freivalds linear binds).
    output_full: Optional[Variable] = None
    output_low: Optional[Variable] = None
    output_shifted: Optional[Variable] = None
    z_output_low: Optional[Variable] = None
    z_output_shifted: Optional[Variable] = None
    range_output_rescale: Optional[Table] = None  # 2^output_rescale_bits table
    range_output: Optional[Table] = None           # 2^output_width loose check


def _rms_limb_range_groups(c: RmsNormClaim):
    """The limb range-checked vars, their z's, and their tables, in the
    FROZEN order shared by rmsnorm_aux, rmsnorm_compile, tape side-effects,
    and the Rust verifier: ym1_chunks, S_limbs, then per bracket (lo, hi):
    gl, g0h chunks, g1h chunks, G2 chunks. The LIMB_W-wide limbs and carry
    lows (S_limbs, gl) check against range_limb (2^LIMB_W); the carry highs'
    16-bit chunks check against range_slack, their narrower top chunks
    against the role's top table."""
    sc = c.config
    def tbls(widths, top):
        return [c.range_slack if w == 16 else top for w in widths]
    yw  = _chunk_widths(sc.y_width)
    g0w = _chunk_widths(sc.g0h_width)
    g1w = _chunk_widths(sc.g1h_width)
    g2w = _chunk_widths(sc.G2_width)
    return [
        (c.ym1_chunks,    c.z_ym1_chunks, tbls(yw,  c.range_y_top)),
        (c.S_limbs,       c.z_S_limbs,    [c.range_limb] * RMS_N_LIMBS),
        (c.lo_gl,         c.z_lo_gl,      [c.range_limb] * 2),
        (c.lo_g0h_chunks, c.z_lo_g0h,     tbls(g0w, c.range_g0h_top)),
        (c.lo_g1h_chunks, c.z_lo_g1h,     tbls(g1w, c.range_g1h_top)),
        (c.lo_G2_chunks,  c.z_lo_G2,      tbls(g2w, c.range_G2_top)),
        (c.hi_gl,         c.z_hi_gl,      [c.range_limb] * 2),
        (c.hi_g0h_chunks, c.z_hi_g0h,     tbls(g0w, c.range_g0h_top)),
        (c.hi_g1h_chunks, c.z_hi_g1h,     tbls(g1w, c.range_g1h_top)),
        (c.hi_G2_chunks,  c.z_hi_G2,      tbls(g2w, c.range_G2_top)),
    ]


def rmsnorm_sample(c: RmsNormClaim, ci: int, s_op) -> List[int]:
    return protocol.op_vec(s_op, ci, "rho", c.config.d)    # ρ ∈ F^d, by index


def rmsnorm_aux(c: RmsNormClaim, witness: dict, rho: List[int]) -> dict:
    """Phase-2 witnesses: Freivalds u, p and one LogUp z per slack chunk."""
    def _t(v):
        if isinstance(v, torch.Tensor): return v.contiguous().view(-1)
        return torch.tensor(v, dtype=torch.uint64, device="cuda")
    B, d = c.config.B, c.config.d
    x_t   = _t(witness[c.x]).view(B, d)
    # Freivalds binds against the raw output (output_full when rescale, else output).
    out_for_p = c.output_full if c.config.output_rescale_bits > 0 else c.output
    out_t = _t(witness[out_for_p]).view(B, d)
    rho_t = torch.tensor(rho, dtype=torch.uint64, device="cuda")
    u = gl_matvec(x_t,   rho_t)                       # (B,)
    p = gl_matvec(out_t, rho_t)                       # (B,)
    α = c.range_slack.alpha
    slack_widths = _chunk_widths(c.config.slack_width)
    def _slack_alpha(n):
        return α if slack_widths[n] == 16 else c.range_slack_top.alpha
    result = {c.u: u, c.p: p}
    for n, (chunk_var, z_var) in enumerate(zip(c.s_lo_chunks, c.z_lo_chunks)):
        chunk_t = _t(witness[chunk_var])
        result[z_var] = gl_inv_batched(
            gl_sub(torch.full_like(chunk_t, _slack_alpha(n)), chunk_t))
    for n, (chunk_var, z_var) in enumerate(zip(c.s_hi_chunks, c.z_hi_chunks)):
        chunk_t = _t(witness[chunk_var])
        result[z_var] = gl_inv_batched(
            gl_sub(torch.full_like(chunk_t, _slack_alpha(n)), chunk_t))
    # Limb range-check z's: each var against the alpha of ITS table (16-bit
    # chunks → range_slack; narrower top chunks → the role's top table).
    for vars_, zs, tbls in _rms_limb_range_groups(c):
        for var, z_var, tbl in zip(vars_, zs, tbls):
            v_t = _t(witness[var])
            result[z_var] = gl_inv_batched(
                gl_sub(torch.full_like(v_t, tbl.alpha), v_t))
    if c.config.rescale_bits > 0:
        α_R = c.range_rescale.alpha
        x_low_t     = _t(witness[c.x_low])
        x_shifted_t = _t(witness[c.x_shifted])
        result[c.z_x_low]     = gl_inv_batched(gl_sub(torch.full_like(x_low_t,     α_R), x_low_t))
        result[c.z_x_shifted] = gl_inv_batched(gl_sub(torch.full_like(x_shifted_t, α),   x_shifted_t))
    if c.config.output_rescale_bits > 0:
        α_OR = c.range_output_rescale.alpha
        α_O  = c.range_output.alpha
        out_low_t     = _t(witness[c.output_low])
        out_shifted_t = _t(witness[c.output_shifted])
        result[c.z_output_low]     = gl_inv_batched(gl_sub(
            torch.full_like(out_low_t,     α_OR), out_low_t))
        result[c.z_output_shifted] = gl_inv_batched(gl_sub(
            torch.full_like(out_shifted_t, α_O),  out_shifted_t))
    return result


def rmsnorm_compile(c: RmsNormClaim, rho: List[int], cfg: LigeroConfig,
                            base: int):
    """Compile RmsNormClaim.

    Seventeen linear families, total 17·B constraints (IDs [base, base+17B)):
      F1  [+0):    S_total = S + d·ε
      F2  [+B):    y_m1 = y − 1
      F3  [+2B):   S = Σ_i X_sq[b·d+i]
      F4  [+3B):   u = Σ_i ρ_i · x[b·d+i]
      F5  [+4B):   p = Σ_i ρ_i · output[b·d+i]
      F6  [+5B):   s_lo = Σ_n 2^{16n} · s_lo_chunks[n]
      F7  [+6B):   s_hi = Σ_n 2^{16n} · s_hi_chunks[n]
      F8  [+7B):   y_m1 = Σ_n 2^{16n} · ym1_chunks[n]
      F9  [+8B):   S_total = S0 + 2^16·S1 + 2^32·S2
      --- wrap-free bracket, lower (q1 = y²); see rmsnorm-bracket-fix.md ---
      F10 [+9B):   H0 = g0l + 2^16·g0h              (g0h from its chunks)
      F11 [+10B):  H1 + g0h = g1l + 2^16·g1h
      F12 [+11B):  H2 + g1h = G2                    (G2 from its chunks)
      F13 [+12B):  2^32·G2 + 2^16·g1l + g0l − s_lo = magic
      --- wrap-free bracket, upper (q2 = (y−1)²) ---
      F14 [+13B):  H0' = g0l' + 2^16·g0h'
      F15 [+14B):  H1' + g0h' = g1l' + 2^16·g1h'
      F16 [+15B):  H2' + g1h' = G2'
      F17 [+16B):  2^32·G2' + 2^16·g1l' + g0l' + s_hi = magic − 1

    With every chunk range-checked to its exact width, F10–F13 force
    y²·S_total = magic + s_lo and F14–F17 force (y−1)²·S_total = magic−1−s_hi
    as INTEGER identities (no term can wrap), which pins y uniquely.
    """
    ell, B, d = cfg.ELL, c.config.B, c.config.d
    L_full = B * d
    neg1 = (P - 1) % P
    magic = c.config.magic
    rho_t = torch.tensor(rho, dtype=torch.uint64, device="cuda")
    neg_rho = gl_neg(rho_t)
    neg_ones_d = torch.full((d,), neg1, dtype=torch.uint64, device="cuda")
    chunk_strides = [1 << (16 * n)
                     for n in range(len(_chunk_widths(c.config.slack_width)))]
    # Freivalds binds the raw product `output_full` when output rescale is on.
    out_target = c.output_full if c.config.output_rescale_bits > 0 else c.output

    row_pkts: List[Tuple[int, object]] = []

    def emit_idscalar(var: Variable, base_id: int, L: int, coef: int):
        for row_off in range(var.n_rows(ell)):
            row_pkts.append((var.row_start + row_off,
                              L2_IdentityScalar(base=base_id,
                                                 var_row_start=var.row_start,
                                                 L=L, coef=coef)))

    def emit_rowsum_vec(var: Variable, base_id: int, L: int, stride: int,
                         coef_vec: torch.Tensor):
        for row_off in range(var.n_rows(ell)):
            row_pkts.append((var.row_start + row_off,
                              L2_RowSumPerSlotVector(base=base_id,
                                                      var_row_start=var.row_start,
                                                      L=L, stride=stride,
                                                      coef_vec=coef_vec)))

    f1 = base
    emit_idscalar(c.S_total, f1, B, 1)
    emit_idscalar(c.S,       f1, B, neg1)

    f2 = base + B
    emit_idscalar(c.y_m1, f2, B, 1)
    emit_idscalar(c.y,    f2, B, neg1)

    f3 = base + 2 * B
    emit_idscalar(c.S, f3, B, 1)
    emit_rowsum_vec(c.X_sq, f3, L_full, d, neg_ones_d)

    f4 = base + 3 * B
    emit_idscalar(c.u, f4, B, 1)
    emit_rowsum_vec(c.x, f4, L_full, d, neg_rho)

    f5 = base + 4 * B
    emit_idscalar(c.p, f5, B, 1)
    emit_rowsum_vec(out_target, f5, L_full, d, neg_rho)

    f6 = base + 5 * B
    emit_idscalar(c.s_lo, f6, B, 1)
    for n, chunk_var in enumerate(c.s_lo_chunks):
        emit_idscalar(chunk_var, f6, B, (P - chunk_strides[n] % P) % P)

    f7 = base + 6 * B
    emit_idscalar(c.s_hi, f7, B, 1)
    for n, chunk_var in enumerate(c.s_hi_chunks):
        emit_idscalar(chunk_var, f7, B, (P - chunk_strides[n] % P) % P)

    # F8: y_m1 tight decomposition (bounds y ≤ 2^y_width, so q1, q2 < 2^{2w}).
    f8 = base + 7 * B
    emit_idscalar(c.y_m1, f8, B, 1)
    for n, chunk_var in enumerate(c.ym1_chunks):
        emit_idscalar(chunk_var, f8, B, (P - (1 << (16 * n)) % P) % P)

    # F9: S_total limb decomposition (bounds S_total < 2^RMS_S_CAP_BITS).
    Lw = RMS_LIMB_W
    f9 = base + 8 * B
    emit_idscalar(c.S_total, f9, B, 1)
    for n, limb_var in enumerate(c.S_limbs):
        emit_idscalar(limb_var, f9, B, (P - (1 << (Lw * n)) % P) % P)

    # F10..F13 / F14..F17: the carry chains assembling q·S_total = magic ± slack
    # over the integers. The limbs and carry lows g0l/g1l are LIMB_W-wide (stride
    # 2^Lw); the highs g0h/g1h/G2 appear only through their 16-bit range-checked
    # chunks (internal stride 2^{16j}), never as standalone committed values.
    def emit_bracket(f0: int, H: List[Variable], gl: List[Variable],
                     g0h: List[Variable], g1h: List[Variable],
                     G2: List[Variable], slack_var: Variable, slack_coef: int):
        # F+0: H0 − g0l − 2^Lw·g0h = 0   (g0h = Σ 2^{16j}·chunk_j)
        emit_idscalar(H[0], f0, B, 1)
        emit_idscalar(gl[0], f0, B, neg1)
        for j, ch in enumerate(g0h):
            emit_idscalar(ch, f0, B, (P - (1 << (Lw + 16 * j)) % P) % P)
        # F+1: H1 + g0h − g1l − 2^Lw·g1h = 0
        f1_ = f0 + B
        emit_idscalar(H[1], f1_, B, 1)
        for j, ch in enumerate(g0h):
            emit_idscalar(ch, f1_, B, (1 << (16 * j)) % P)
        emit_idscalar(gl[1], f1_, B, neg1)
        for j, ch in enumerate(g1h):
            emit_idscalar(ch, f1_, B, (P - (1 << (Lw + 16 * j)) % P) % P)
        # F+2: H2 + g1h − G2 = 0
        f2_ = f0 + 2 * B
        emit_idscalar(H[2], f2_, B, 1)
        for j, ch in enumerate(g1h):
            emit_idscalar(ch, f2_, B, (1 << (16 * j)) % P)
        for j, ch in enumerate(G2):
            emit_idscalar(ch, f2_, B, (P - (1 << (16 * j)) % P) % P)
        # F+3: 2^{2Lw}·G2 + 2^Lw·g1l + g0l + slack_coef·slack = b (b via nz below)
        f3_ = f0 + 3 * B
        for j, ch in enumerate(G2):
            emit_idscalar(ch, f3_, B, (1 << (2 * Lw + 16 * j)) % P)
        emit_idscalar(gl[1], f3_, B, (1 << Lw) % P)
        emit_idscalar(gl[0], f3_, B, 1)
        emit_idscalar(slack_var, f3_, B, slack_coef)

    emit_bracket(base + 9 * B, c.lo_H, c.lo_gl, c.lo_g0h_chunks,
                 c.lo_g1h_chunks, c.lo_G2_chunks, c.s_lo, neg1)
    emit_bracket(base + 13 * B, c.hi_H, c.hi_gl, c.hi_g0h_chunks,
                 c.hi_g1h_chunks, c.hi_G2_chunks, c.s_hi, 1)

    cur = base + 17 * B

    # ---- Input rescale (rescale_bits > 0): x_in = (1<<r)·x + x_low, plus
    # offset-trick x_shifted = x + 2^15. Two L_full families.
    if c.config.rescale_bits > 0:
        rescale_offset = 1 << 15   # signed shift against the 16-bit slack table
        _emit_lin_csr_idscalar(
            c.x_in, [c.x, c.x_low], [1 << c.config.rescale_bits, 1],
            L_full, ell, cur, row_pkts); cur += L_full
        _emit_lin_csr_idscalar(
            c.x_shifted, [c.x], [1], L_full, ell, cur, row_pkts); cur += L_full

    # ---- Output rescale (output_rescale_bits > 0): output_full = (1<<r_out)·output
    # + output_low, plus offset-trick output_shifted = output + 2^(output_width-1).
    if c.config.output_rescale_bits > 0:
        out_offset = 1 << (c.config.output_width - 1)
        _emit_lin_csr_idscalar(
            c.output_full, [c.output, c.output_low],
            [1 << c.config.output_rescale_bits, 1],
            L_full, ell, cur, row_pkts); cur += L_full
        _emit_lin_csr_idscalar(
            c.output_shifted, [c.output], [1], L_full, ell, cur, row_pkts); cur += L_full

    # Quadratics. The bracket products are per-limb (H = q·S_limb) so each is
    # a wrap-free integer; the old whole-product quads (q·S_total vs magic)
    # are replaced by the F10..F17 carry chains above.
    quads: List[QuadraticConstraint] = []
    quads += _per_slot_quad("rms.X_sq",      c.x,    c.x,       c.X_sq,    neg1, 0, L_full, ell)
    quads += _per_slot_quad("rms.q1",        c.y,    c.y,       c.q1,      neg1, 0, B,      ell)
    quads += _per_slot_quad("rms.q2",        c.y_m1, c.y_m1,    c.q2,      neg1, 0, B,      ell)
    for k in range(RMS_N_LIMBS):
        quads += _per_slot_quad(f"rms.loH{k}", c.q1, c.S_limbs[k], c.lo_H[k],
                                 neg1, 0, B, ell)
    for k in range(RMS_N_LIMBS):
        quads += _per_slot_quad(f"rms.hiH{k}", c.q2, c.S_limbs[k], c.hi_H[k],
                                 neg1, 0, B, ell)
    quads += _per_slot_quad("rms.freivalds", c.y,    c.u,       c.p,       neg1, 0, B,      ell)
    α_T = c.range_slack.alpha
    slack_tbl_widths = _chunk_widths(c.config.slack_width)
    def slack_alpha(n):
        if slack_tbl_widths[n] == 16:
            return α_T
        return c.range_slack_top.alpha
    for n, (chunk_var, z_var) in enumerate(zip(c.s_lo_chunks, c.z_lo_chunks)):
        quads += _per_slot_quad(f"rms.RW[s_lo_c{n}]", chunk_var, z_var, z_var,
                                 (P - slack_alpha(n)) % P, neg1, B, ell)
    for n, (chunk_var, z_var) in enumerate(zip(c.s_hi_chunks, c.z_hi_chunks)):
        quads += _per_slot_quad(f"rms.RW[s_hi_c{n}]", chunk_var, z_var, z_var,
                                 (P - slack_alpha(n)) % P, neg1, B, ell)
    # Limb range checks, in the frozen _rms_limb_range_groups order.
    for vars_, zs, tbls in _rms_limb_range_groups(c):
        for var, z_var, tbl in zip(vars_, zs, tbls):
            quads += _per_slot_quad(f"rms.RW[{var.name}]", var, z_var, z_var,
                                     (P - tbl.alpha) % P, neg1, B, ell)
    if c.config.rescale_bits > 0:
        α_R = c.range_rescale.alpha
        quads += _per_slot_quad(
            "rms.RW[x_low]", c.x_low, c.z_x_low, c.z_x_low,
            (P - α_R) % P, neg1, L_full, ell)
        quads += _per_slot_quad(
            "rms.RW[x_shifted]", c.x_shifted, c.z_x_shifted, c.z_x_shifted,
            (P - α_T) % P, neg1, L_full, ell)
    if c.config.output_rescale_bits > 0:
        α_OR = c.range_output_rescale.alpha
        α_O  = c.range_output.alpha
        quads += _per_slot_quad(
            "rms.RW[output_low]", c.output_low, c.z_output_low, c.z_output_low,
            (P - α_OR) % P, neg1, L_full, ell)
        quads += _per_slot_quad(
            "rms.RW[output_shifted]", c.output_shifted, c.z_output_shifted, c.z_output_shifted,
            (P - α_O) % P, neg1, L_full, ell)

    # b for rmsnorm linear families:
    #   F1 [0, B):        S_total = S + d·ε   →  b = d·ε
    #   F2 [B, 2B):       y_m1 = y − 1        →  b = neg1
    #   F13 [12B, 13B):   lower bracket       →  b = magic
    #   F17 [16B, 17B):   upper bracket       →  b = magic − 1
    #   all others:       b = 0
    #   input rescale (if enabled): [17B + L_full, 17B + 2·L_full): b = rescale_offset
    #   output rescale (if enabled): adds at end with b = out_offset
    nz: List[Tuple[int, int, Any]] = [
        (0, B, (c.config.d * c.config.eps_int) % P),
        (B, B, neg1),
        (12 * B, B, magic % P),
        (16 * B, B, (magic - 1) % P),
    ]
    cur_off = 17 * B
    if c.config.rescale_bits > 0:
        # x_in family (b=0): cur_off..cur_off+L_full; x_shifted family with b=rescale_offset.
        cur_off += L_full
        nz.append((cur_off, L_full, 1 << 15))
        cur_off += L_full
    if c.config.output_rescale_bits > 0:
        cur_off += L_full
        nz.append((cur_off, L_full, 1 << (c.config.output_width - 1)))
        cur_off += L_full
    return row_pkts, quads, cur - base, _build_b_chunk(cur - base, nz)


COMPILE_FNS[RmsNormClaim] = rmsnorm_compile


# ===========================================================================
# SoftmaxClaim — batched softmax via §2.13 bracketed shift-invariance
# (nonarith-survey §2.13). Per row b ∈ [0, B), of length M:
#
#   z[b·M + i]   = c2[b] − x[b·M + i]                  (c2 is the LSE candidate)
#   y_A[b·M + i] = T_A[z[b·M + i]] = round(exp(−z/s_c)·s_y)
#   y_B[b·M + i] = T_B[z[b·M + i]] = round(exp((δ−z)/s_c)·s_y)
#                                  ≈ exp(δ/s_c) · y_A
#   s1[b]        = Σ_i y_A[b·M + i]
#   s2[b]        = Σ_i y_B[b·M + i]
#
# Bracket (two non-negative residuals enforced by range LogUp):
#   r_lo[b] = (s_y + slack_max) − s1[b]    ≥ 0    forces s1 ≤ s_y + slack_max
#   r_hi[b] = s2[b] − (s_y − slack_max)    ≥ 0    forces s2 ≥ s_y − slack_max
#
# Pinning: c2 to within ~δ/s_c + slack_max/s_y of LSE. At Q3.12 (s_c=s_y=2^12)
# with δ=1 and slack_max=M, that's a real-unit window of ~0.5 — much tighter
# than §2.3 single-table but not strictly unique due to slack rounding
# tolerance. Output y_A is the softmax result (up to negligible factor of
# exp(±slack_max/s_y) from rounding).
# ===========================================================================


@dataclass(frozen=True, slots=True)
class SoftmaxConfig:
    """B rows of length M each. s_x = s_c by convention (input + LSE scale);
    s_y is output scale. Z_max bounds the paired exp table; out-of-range z
    fails the lookup unless `saturate=True`. The bracket is tight (no slack):
    s1 ≤ s_y AND s2 ≥ s_y+1 pins c2 to the unique integer crossing via the
    s2(c2) = s1(c2−δ) identity built into _softmax_exp_tables.

    `s_in` lets the caller commit `x` at a coarser scale than `s_x` (e.g.,
    `s_in = s_x²` straight from a Q3.12 matmul). When `s_in > s_x`, the claim
    internally word-decomposes each x_in cell into (x_low, x) such that
    x_in = (s_in/s_x)·x + x_low, with both pieces range-checked; the existing
    construction then operates on `x` at scale `s_x`.

    `saturate=True` adds a SiLU-style high-z mux: z = z_low + Z_max·z_high
    with z_low ∈ [0, Z_max) (the lookup key) and z_high ≥ 0 (range-checked
    against a 2^Z_high_width range table). When z_high ≠ 0 the output y is
    forced to 0 (matches T_A[Z_max+] = 0 once Z_max is past natural
    saturation), so Z_max only needs to cover the non-zero region of
    exp(−z/s_c)·s_y (~9·s_c) rather than the full spread of c2 − x. ONE
    direction only — c2 ≥ max(x) is enforced automatically by the bracket
    (negative z values can't be decomposed into nonneg z_low + Z_max·z_high
    within the range tables, so the proof rejects)."""
    B: int
    M: int
    s_x: int
    s_c: int
    s_y: int
    delta: int        # bracket shift in integer units at s_c; nominally 1
    Z_max: int        # paired exp table size; covers the non-zero exp region
    s_in: int = 0     # 0 → no rescale (s_in == s_x implicitly)
    saturate: bool = False        # high-z saturating-mux gadget
    Z_high_width: int = 16        # bit-width of z_high range check (saturate only)
    aux_chunk_width: int = 16     # bit-width of range_aux table (for c2_shifted)
    # Causal attention mask via doubled lookup table. exp_A and exp_B become
    # T_A || T_zero / T_B || T_zero (size 2·Z_max each). Per-cell lookup key
    # is z_low + Z_max·is_masked(b, j), where is_masked is determined entirely
    # by (heads, M, B) at compile time: row b decomposes as (i=b//heads,
    # h=b%heads), and (i, j) is masked iff j > i. No magic constants, no
    # extra constraints — masked cells naturally pull y from the zero half
    # of the table.
    causal: bool = False
    heads: int = 1                # multi-head batching for the causal-mask layout
    # round_up: ceil exp tables + saturate out-of-table weights to 1 (not 0), so
    # every weight >= exact -> the partition over-counts -> c2 is pinned upward
    # (c2 >= true LSE). A sound over-estimate with no external constant. Requires
    # non-causal (no masked cells to disambiguate). Default off (attention path
    # is byte-identical: round() tables, saturate-to-0).
    round_up: bool = False

    @property
    def rescale_bits(self) -> int:
        if self.s_in == 0 or self.s_in == self.s_x:
            return 0
        ratio = self.s_in // self.s_x
        assert ratio > 0 and (ratio & (ratio - 1)) == 0 and ratio * self.s_x == self.s_in, (
            f"SoftmaxConfig: s_in={self.s_in} must be a power-of-2 multiple of s_x={self.s_x}")
        return ratio.bit_length() - 1


def _softmax_exp_tables(cfg: SoftmaxConfig):
    """T_A[k] = round(exp(−k/s_c) · s_y);  T_B[k] = round(exp((δ−k)/s_c) · s_y).
    Returns two np.uint64 arrays of length Z_max.

    SOUNDNESS-CRITICAL: T_A and T_B are computed from the SAME `round(exp(·)·s_y)`
    expression with shifted argument (k vs k−δ). For δ=1 integer unit this makes
    T_B[k] == T_A[k−1] bit-identically, hence s2(c2) == s1(c2−1) as integer sums.
    The bracket's `s1 ≤ s_y ∧ s2 ≥ s_y+1` pins c2 to a unique integer via this
    identity. Do NOT "optimize" T_B[k] := round(T_A[k] · exp(δ/s_c)) — that
    introduces independent rounding and breaks the identity, reopening the c2
    window we just closed.

    Vectorized (numpy): byte-identical to the per-k Python loop (math.exp/round are
    float64, as is numpy's exp/round-half-to-even and ceil), so the T_B[k]==T_A[k−1]
    identity is preserved. Required to build the production tables (Z_max ~ 10^8)
    in seconds instead of a multi-minute Python loop. Every value is ≤ s_y < P, so
    the loop's `% P` was a no-op and is dropped (it can't be vectorized: P > 2^63)."""
    import numpy as np
    assert cfg.s_y < P, f"s_y={cfg.s_y} must be < P for the no-op modulo to hold"
    # round_up ceils BOTH tables from the SAME shifted expression, so the
    # T_B[k] == T_A[k-1] identity (and the unique-c2 bracket) still holds.
    rnd = np.ceil if getattr(cfg, "round_up", False) else np.round
    k = np.arange(cfg.Z_max, dtype=np.float64)
    T_A = rnd(np.exp(-k / cfg.s_c) * cfg.s_y).astype(np.uint64)
    T_B = rnd(np.exp((cfg.delta - k) / cfg.s_c) * cfg.s_y).astype(np.uint64)
    return T_A, T_B


@dataclass
class SoftmaxClaim:
    # User-facing
    x: Variable                 # length B·M — internal x at scale `s_x`. When
                                #   rescale_bits == 0 this is the public input;
                                #   otherwise it's an internal commit derived from x_in.
    y_A: Variable               # length B·M (softmax output)
    config: SoftmaxConfig
    length: int                 # = B·M

    # Phase-1 aux
    c2: Variable                # length B (LSE candidate per row)
    z: Variable                 # length B·M  (c2 − x; checked via T_A/T_B lookup)
    y_B: Variable               # length B·M  (δ-shifted exp table result)
    s1: Variable                # length B
    s2: Variable                # length B
    r_lo: Variable              # length B  (= s_y − s1, ≥ 0     ⇒ s1 ≤ s_y)
    r_hi: Variable              # length B  (= s2 − (s_y + 1), ≥ 0 ⇒ s2 ≥ s_y + 1)

    # Phase-2 aux
    pt_u_A: Variable            # length B·M   (= z + β_A · y_A)
    pt_z_A: Variable            # length B·M   (= 1 / (α_A − pt_u_A))
    pt_u_B: Variable            # length B·M   (= z + β_B · y_B)
    pt_z_B: Variable            # length B·M
    z_c2: Variable              # length B    (range LogUp z for c2_shifted)
    z_r_lo: Variable            # length B
    z_r_hi: Variable            # length B
    # c2 offset trick: c2_shifted = c2 + 2^(aux_chunk_width - 1) lives in
    # [0, 2^aux_chunk_width), letting c2 itself be signed (LSE can be
    # negative — happens for rows where all x_i are negative). Always
    # present; soundness-equivalent to direct range check on c2 when LSE > 0.
    c2_shifted: Variable        # length B

    # Tables
    exp_A: Table                # paired (z, T_A[z]),  Z_max entries
    exp_B: Table                # paired (z, T_B[z]),  Z_max entries
    range_aux: Table            # range LogUp for c2, r_lo, r_hi (16-bit shared)

    # Optional rescale plumbing (only present when config.rescale_bits > 0):
    x_in: Optional[Variable] = None        # public input at scale s_in
    x_low: Optional[Variable] = None       # low word, tight range-checked
    x_shifted: Optional[Variable] = None   # x + 2^(w-1) (offset trick)
    z_x_low: Optional[Variable] = None     # phase-2 z for x_low
    z_x_shifted: Optional[Variable] = None # phase-2 z for x_shifted (range_aux)
    range_rescale: Optional[Table] = None  # 2^rescale_bits table for x_low

    # Optional saturation plumbing (only when config.saturate). The lookup
    # key `z` (above) becomes the LOW word z_low ∈ [0, Z_max); z_high is
    # the high word, range-checked against `range_z_high`. is_high is the
    # SiLU-style "z_high ≠ 0" boolean, gated through inv_z_high. y_A_raw /
    # y_B_raw are the raw exp-table lookups; y_A / y_B are the muxed final
    # values used by the bracket: y_A = y_A_raw − is_high · y_A_raw.
    z_high: Optional[Variable] = None        # length B·M
    inv_z_high: Optional[Variable] = None    # length B·M (Fermat inv or 0)
    is_high: Optional[Variable] = None       # length B·M (boolean)
    y_A_raw: Optional[Variable] = None       # length B·M
    y_B_raw: Optional[Variable] = None       # length B·M
    mux_y_A: Optional[Variable] = None       # length B·M (= is_high · y_A_raw)
    mux_y_B: Optional[Variable] = None       # length B·M (= is_high · y_B_raw)
    z_z_high: Optional[Variable] = None      # phase-2 z for z_high range LogUp
    range_z_high: Optional[Table] = None     # 2^Z_high_width entries


def softmax_sample(c: SoftmaxClaim, ci, s_op):
    return None        # α/β all come from TableSettlements


def softmax_aux(c: SoftmaxClaim, witness: dict, _ch) -> dict:
    """Phase-2: pt_u_A/B = z + β·y, pt_z_A/B = 1/(α − pt_u), plus range z's."""
    def _t(v):
        if isinstance(v, torch.Tensor): return v.contiguous().view(-1)
        return torch.tensor(v, dtype=torch.uint64, device="cuda")
    z_t   = _t(witness[c.z])
    # When saturating, the lookup operates on y_A_raw / y_B_raw, NOT the
    # muxed y_A / y_B — must match the pt_u_A/B linear in softmax_compile.
    y_A_lookup = c.y_A_raw if c.config.saturate else c.y_A
    y_B_lookup = c.y_B_raw if c.config.saturate else c.y_B
    y_A_t = _t(witness[y_A_lookup])
    y_B_t = _t(witness[y_B_lookup])
    c2_shifted_t = _t(witness[c.c2_shifted])
    r_lo_t = _t(witness[c.r_lo])
    r_hi_t = _t(witness[c.r_hi])

    pt_u_A = gl_add(z_t, gl_mul(torch.full_like(z_t, c.exp_A.beta), y_A_t))
    pt_u_B = gl_add(z_t, gl_mul(torch.full_like(z_t, c.exp_B.beta), y_B_t))
    if c.config.causal:
        # Causal lookup key = z + Z_max·is_masked. Match the per-cell shift
        # the compile emits in pt_u_A/B's b vector.
        H_cfg = c.config.heads
        M = c.config.M
        L_full = c.length
        i_full = torch.arange(L_full, dtype=torch.int64, device="cuda")
        b_full = i_full // M
        j_full = i_full % M
        i_qry_full = b_full // H_cfg
        mask_shift = (((j_full > i_qry_full).to(torch.int64) * c.config.Z_max)
                       .to(torch.uint64))
        pt_u_A = gl_add(pt_u_A, mask_shift)
        pt_u_B = gl_add(pt_u_B, mask_shift)
    result = {
        c.pt_u_A: pt_u_A,
        c.pt_z_A: gl_inv_batched(gl_sub(torch.full_like(pt_u_A, c.exp_A.alpha), pt_u_A)),
        c.pt_u_B: pt_u_B,
        c.pt_z_B: gl_inv_batched(gl_sub(torch.full_like(pt_u_B, c.exp_B.alpha), pt_u_B)),
        c.z_c2:   gl_inv_batched(gl_sub(torch.full_like(c2_shifted_t, c.range_aux.alpha), c2_shifted_t)),
        c.z_r_lo: gl_inv_batched(gl_sub(torch.full_like(r_lo_t, c.range_aux.alpha), r_lo_t)),
        c.z_r_hi: gl_inv_batched(gl_sub(torch.full_like(r_hi_t, c.range_aux.alpha), r_hi_t)),
    }
    if c.config.rescale_bits > 0:
        α_R = c.range_rescale.alpha
        α_A = c.range_aux.alpha
        x_low_t     = _t(witness[c.x_low])
        x_shifted_t = _t(witness[c.x_shifted])
        result[c.z_x_low]     = gl_inv_batched(gl_sub(torch.full_like(x_low_t,     α_R), x_low_t))
        result[c.z_x_shifted] = gl_inv_batched(gl_sub(torch.full_like(x_shifted_t, α_A), x_shifted_t))
    if c.config.saturate:
        z_high_t = _t(witness[c.z_high])
        result[c.z_z_high] = gl_inv_batched(gl_sub(
            torch.full_like(z_high_t, c.range_z_high.alpha), z_high_t))
    return result


def _emit_pkt_per_row(var: Variable, ell: int, build: Callable[[], object],
                        row_pkts: List[Tuple[int, object]]) -> None:
    """Per-row helper: append `(var.row_start + r, build())` for each of
    var's rows. Used by all per-claim compile_fns."""
    pkt = build()
    for row_off in range(var.n_rows(ell)):
        row_pkts.append((var.row_start + row_off, pkt))


def softmax_compile(c: SoftmaxClaim, _ch, cfg: LigeroConfig, base: int):
    """Compile SoftmaxClaim — basic, saturate, and causal paths.
    (Rescale path not migrated yet.)

    Constraint families (some only present under saturate / causal):
      F0: z decomp  — L_u under causal, L_full otherwise
                       (sat adds Z_max·z_high term; same constraint count)
      F1: pt_u_A = z + β_A · y_A_lookup                            L_full
      F2: pt_u_B = z + β_B · y_B_lookup                            L_full
      [sat] F1.5:  y_A = y_A_raw − mux_y_A                          L_full
      [sat] F2.5:  y_B = y_B_raw − mux_y_B                          L_full
      F3: s1 = Σ_i y_A[b·M+i]                                       B
      F4: s2 = Σ_i y_B[b·M+i]                                       B
      F5: s1 + r_lo = s_y                                           B
      F6: r_hi − s2 = −(s_y + 1)                                    B
      F7: c2_shifted = c2 + 2^(aux_chunk_width − 1)                 B
    """
    ell = cfg.ELL
    B, M, H = c.config.B, c.config.M, c.config.heads
    L_full = B * M
    neg1   = (P - 1) % P
    Z_max  = c.config.Z_max
    s_y    = c.config.s_y
    sat, causal = c.config.saturate, c.config.causal
    y_A_look = c.y_A_raw if sat else c.y_A
    y_B_look = c.y_B_raw if sat else c.y_B
    β_A, β_B = c.exp_A.beta, c.exp_B.beta

    row_pkts: List[Tuple[int, object]] = []
    cur = base

    # ---- F0: z decomp ----
    # Without causal: L_full identity-scalar/stride constraints.
    # With causal:    L_u = H·SEQ·(SEQ+1)/2 filtered constraints.
    if causal:
        SEQ = B // H
        L_u = H * SEQ * (SEQ + 1) // 2
        _emit_pkt_per_row(c.z,  ell, lambda: L2_CausalFilteredIdScalar(
            base=cur, var_row_start=c.z.row_start,  L=L_full, M=M, H=H, coef=1), row_pkts)
        _emit_pkt_per_row(c.c2, ell, lambda: L2_CausalFilteredC2Stride(
            base=cur, c2_row_start=c.c2.row_start, B=B, H=H, coef=neg1), row_pkts)
        _emit_pkt_per_row(c.x,  ell, lambda: L2_CausalFilteredIdScalar(
            base=cur, var_row_start=c.x.row_start,  L=L_full, M=M, H=H, coef=1), row_pkts)
        if sat:
            _emit_pkt_per_row(c.z_high, ell, lambda: L2_CausalFilteredIdScalar(
                base=cur, var_row_start=c.z_high.row_start, L=L_full, M=M, H=H,
                coef=Z_max % P), row_pkts)
        cur += L_u
    else:
        _emit_pkt_per_row(c.z,  ell, lambda: L2_IdentityScalar(
            base=cur, var_row_start=c.z.row_start,  L=L_full, coef=1), row_pkts)
        _emit_pkt_per_row(c.c2, ell, lambda: L2_StrideOneToManyScalar(
            base=cur, var_row_start=c.c2.row_start, L=B, stride=M, coef=neg1), row_pkts)
        _emit_pkt_per_row(c.x,  ell, lambda: L2_IdentityScalar(
            base=cur, var_row_start=c.x.row_start,  L=L_full, coef=1), row_pkts)
        if sat:
            _emit_pkt_per_row(c.z_high, ell, lambda: L2_IdentityScalar(
                base=cur, var_row_start=c.z_high.row_start, L=L_full,
                coef=Z_max % P), row_pkts)
        cur += L_full

    # ---- [sat] y_A/y_B mux: emitted BEFORE pt_u_A/B (consumer in b_chunk
    #            assumes this order when slicing constraint IDs). ----
    if sat:
        # round_up: y = y_raw - mux + is_high (saturate out-of-table to 1, not 0),
        # i.e. y_raw = y + mux - is_high.  Default: y = y_raw - mux (saturate to 0).
        ru_src = [c.is_high] if c.config.round_up else []
        ru_co  = [-1]        if c.config.round_up else []
        _emit_lin_csr_idscalar(c.y_A_raw, [c.y_A, c.mux_y_A] + ru_src, [1, 1] + ru_co, L_full, ell, cur, row_pkts); cur += L_full
        _emit_lin_csr_idscalar(c.y_B_raw, [c.y_B, c.mux_y_B] + ru_src, [1, 1] + ru_co, L_full, ell, cur, row_pkts); cur += L_full

    # ---- F1, F2: pt_u_A/B (always L_full; b_pt absorbs mask on RHS) ----
    _emit_lin_csr_idscalar(c.pt_u_A, [c.z, y_A_look], [1, β_A], L_full, ell, cur, row_pkts); cur += L_full
    _emit_lin_csr_idscalar(c.pt_u_B, [c.z, y_B_look], [1, β_B], L_full, ell, cur, row_pkts); cur += L_full

    # ---- F3, F4: s1, s2 row sums ----
    neg_ones_M = torch.full((M,), neg1, dtype=torch.uint64, device="cuda")
    _emit_pkt_per_row(c.s1, ell, lambda: L2_IdentityScalar(
        base=cur, var_row_start=c.s1.row_start, L=B, coef=1), row_pkts)
    _emit_pkt_per_row(c.y_A, ell, lambda: L2_RowSumPerSlotVector(
        base=cur, var_row_start=c.y_A.row_start, L=L_full, stride=M,
        coef_vec=neg_ones_M), row_pkts)
    cur += B
    _emit_pkt_per_row(c.s2, ell, lambda: L2_IdentityScalar(
        base=cur, var_row_start=c.s2.row_start, L=B, coef=1), row_pkts)
    _emit_pkt_per_row(c.y_B, ell, lambda: L2_RowSumPerSlotVector(
        base=cur, var_row_start=c.y_B.row_start, L=L_full, stride=M,
        coef_vec=neg_ones_M), row_pkts)
    cur += B

    # ---- F5-F7: bracket linears + c2_shifted ----
    _emit_lin_csr_idscalar(c.s1,         [c.r_lo], [-1], B, ell, cur, row_pkts); cur += B
    _emit_lin_csr_idscalar(c.r_hi,       [c.s2],   [ 1], B, ell, cur, row_pkts); cur += B
    _emit_lin_csr_idscalar(c.c2_shifted, [c.c2],   [ 1], B, ell, cur, row_pkts); cur += B

    # ---- Input rescale (rescale_bits > 0): x_in = (1<<r)·x + x_low,
    # plus offset-trick x_shifted = x + 2^15 (centred for range_aux).
    if c.config.rescale_bits > 0:
        rescale_offset = 1 << 15
        _emit_lin_csr_idscalar(
            c.x_in, [c.x, c.x_low], [1 << c.config.rescale_bits, 1],
            L_full, ell, cur, row_pkts); cur += L_full
        _emit_lin_csr_idscalar(
            c.x_shifted, [c.x], [1], L_full, ell, cur, row_pkts); cur += L_full

    # ---- Quads: same shape as legacy, gated by sat ----
    quads: List[QuadraticConstraint] = []
    quads += _per_slot_quad("sm.PT_A", c.pt_u_A, c.pt_z_A, c.pt_z_A,
                              (P - c.exp_A.alpha) % P, neg1, L_full, ell)
    quads += _per_slot_quad("sm.PT_B", c.pt_u_B, c.pt_z_B, c.pt_z_B,
                              (P - c.exp_B.alpha) % P, neg1, L_full, ell)
    α_R = c.range_aux.alpha
    quads += _per_slot_quad("sm.RW[c2_shifted]", c.c2_shifted, c.z_c2,   c.z_c2,   (P - α_R) % P, neg1, B, ell)
    quads += _per_slot_quad("sm.RW[r_lo]",       c.r_lo,       c.z_r_lo, c.z_r_lo, (P - α_R) % P, neg1, B, ell)
    quads += _per_slot_quad("sm.RW[r_hi]",       c.r_hi,       c.z_r_hi, c.z_r_hi, (P - α_R) % P, neg1, B, ell)
    if sat:
        quads += _per_slot_quad("sm.zh_invzh",  c.z_high,   c.inv_z_high, c.is_high, neg1, 0, L_full, ell)
        quads += _per_slot_quad("sm.ish_zh",    c.is_high,  c.z_high,     c.z_high,  neg1, 0, L_full, ell)
        quads += _per_slot_quad("sm.ish²",      c.is_high,  c.is_high,    c.is_high, neg1, 0, L_full, ell)
        quads += _per_slot_quad("sm.mux_yA",    c.is_high,  c.y_A_raw,    c.mux_y_A, neg1, 0, L_full, ell)
        quads += _per_slot_quad("sm.mux_yB",    c.is_high,  c.y_B_raw,    c.mux_y_B, neg1, 0, L_full, ell)
        α_ZH = c.range_z_high.alpha
        quads += _per_slot_quad("sm.RW[z_high]", c.z_high, c.z_z_high, c.z_z_high,
                                  (P - α_ZH) % P, neg1, L_full, ell)
    if c.config.rescale_bits > 0:
        α_RR = c.range_rescale.alpha
        quads += _per_slot_quad(
            "sm.RW[x_low]", c.x_low, c.z_x_low, c.z_x_low,
            (P - α_RR) % P, neg1, L_full, ell)
        quads += _per_slot_quad(
            "sm.RW[x_shifted]", c.x_shifted, c.z_x_shifted, c.z_x_shifted,
            (P - α_R) % P, neg1, L_full, ell)

    # b for softmax linear families. Track running offset through optional
    # families in the same order the constraints were emitted.
    nz: List[Tuple[int, int, Any]] = []
    # F0 z decomp size: L_u (causal) = H · SEQ · (SEQ+1) / 2  where SEQ = B/H;
    #                   L_full otherwise.
    off = (H * (B // H) * ((B // H) + 1) // 2) if causal else L_full
    if sat:
        off += L_full  # y_A mux  (b=0)
        off += L_full  # y_B mux  (b=0)
    # pt_u_A / pt_u_B: under causal, b_pt = Z_max if masked else 0 (per-cell).
    if causal:
        i_full = torch.arange(L_full, dtype=torch.int64, device="cuda")
        b_full_idx = i_full // M
        j_full     = i_full % M
        i_qry_full = b_full_idx // H
        b_pt = ((j_full > i_qry_full).to(torch.int64) * Z_max).to(torch.uint64).contiguous()
        nz.append((off, L_full, b_pt));     off += L_full  # pt_u_A
        nz.append((off, L_full, b_pt));     off += L_full  # pt_u_B
    else:
        off += 2 * L_full
    off += 2 * B   # s1, s2 row sums (b = 0)
    nz.append((off, B, s_y % P));                off += B   # s1 + r_lo = s_y
    nz.append((off, B, (P - (s_y + 1)) % P));    off += B   # r_hi − s2 = −(s_y+1)
    nz.append((off, B, (1 << (c.config.aux_chunk_width - 1)) % P)); off += B   # c2_shifted
    if c.config.rescale_bits > 0:
        off += L_full  # x_in family, b = 0
        nz.append((off, L_full, (1 << 15) % P));  off += L_full   # x_shifted

    return row_pkts, quads, cur - base, _build_b_chunk(cur - base, nz)


COMPILE_FNS[SoftmaxClaim] = softmax_compile


# ===========================================================================
# RoPEClaim — rotary positional embedding (Llama-style split-half pairing).
#
# Per position `pos ∈ [0, SEQ)` and dim-pair index `k ∈ [0, d_h/2)`, with
# public coefficients c = round(cos(pos·θ_k)·s_x), s = round(sin(pos·θ_k)·s_x)
# at scale s_x, where θ_k = 1 / base^(2k/d_h) (base=10000 in Llama):
#
#   x_rot[seq·d_h + k]            = c · x[seq·d_h + k]      − s · x[seq·d_h + k + d_h/2]
#   x_rot[seq·d_h + k + d_h/2]    = s · x[seq·d_h + k]      + c · x[seq·d_h + k + d_h/2]
#
# Both equations are LINEAR in the committed (x, x_rot) with PUBLIC
# coefficients — no quadratics, no LogUp, no tables. The verifier
# computes the same (c, s) table from `RoPEConfig` and the constraints
# pass iff x_rot is the correct rotation of x.
#
# OUTPUT SCALE: x at scale s_x, c/s at scale s_x ⇒ x_rot at scale s_x².
# Downstream consumers either accept s_x² inputs or chain a rescale.
# A future variant could absorb an internal rescale (high word committed
# at scale s_x, low word range-checked) following the pattern used by
# RmsNorm/Softmax/SiLU's optional rescale block — see CLAIM_SPECS.md.
#
# FUTURE GENERALIZATION (deferred): the entire claim is the n=1 case of
# a `PublicLinearMapClaim(input, output, M_sparse_csr)` primitive that
# emits one linear constraint per output cell with the nonzero coefficients
# of the matching M row. RoPE has 2 nnz per row (the cos/sin pair).
# Other potential users — DFT, fixed-mask projections, public positional
# encodings — could share the same primitive. Worth abstracting once a
# second op needs it; for one user (RoPE) the dedicated claim is simpler.
# ===========================================================================


@dataclass(frozen=True, slots=True)
class RoPEConfig:
    """RoPE for SEQ tokens × H heads × d_h dims-per-head. cos/sin computed
    at compile time at scale s_x; output committed at scale s_x².
    `position_offset` supports non-zero starting positions (packed contexts).

    Multi-head: when heads > 1, x and x_rot have logical shape (SEQ, H, d_h)
    flat at seq*(H*d_h) + h*d_h + k. The same (c, s) table indexed by
    (seq, k) is applied to every head. heads=1 reduces to single-head."""
    SEQ: int
    d_h: int                       # head dim; must be even
    s_x: int                       # input integer scale
    base: float = 10000.0          # RoPE frequency base (Llama default)
    position_offset: int = 0
    heads: int = 1                 # H; cos/sin shared across heads


def _rope_cos_sin(cfg: RoPEConfig) -> Tuple[List[int], List[int]]:
    """Build c, s integer tables at scale s_x, indexed by seq·(d_h/2)+k.
    Both prover and verifier compute these identically; the cross-claim
    soundness of RoPE relies on the same expression here."""
    half = cfg.d_h // 2
    c_l, s_l = [], []
    for seq in range(cfg.SEQ):
        pos = seq + cfg.position_offset
        for k in range(half):
            theta_k = pos / (cfg.base ** (2 * k / cfg.d_h))
            c_l.append(int(round(math.cos(theta_k) * cfg.s_x)) % P)
            s_l.append(int(round(math.sin(theta_k) * cfg.s_x)) % P)
    return c_l, s_l


@dataclass
class RoPEClaim:
    """Rotary positional embedding with optional internal output rescale.

    Without rescale: x_rot at scale s_x² (raw rotation).
    With rescale: x_rot_full at scale s_x² (the raw rotation, internal),
    x_rot at scale s_out = s_x (visible to caller, rescaled high word)."""
    x: Variable                   # length SEQ·d_h, at scale s_x
    x_rot: Variable               # length SEQ·d_h, at scale s_x² (or s_out if rescale)
    config: RoPEConfig
    # Optional output rescale.
    rescale_bits: int = 0
    output_width: int = 16
    x_rot_full: Optional[Variable] = None
    x_rot_low: Optional[Variable] = None
    x_rot_shifted: Optional[Variable] = None
    z_x_rot_low: Optional[Variable] = None
    z_x_rot_shifted: Optional[Variable] = None
    range_rescale: Optional[Table] = None
    range_output: Optional[Table] = None


def rope_sample(c: RoPEClaim, ci, s_op):
    return None       # no per-claim randomness; (c,s) are deterministic


def rope_aux(c: RoPEClaim, witness: dict, _ch) -> dict:
    if c.rescale_bits == 0:
        return {}
    def _t(v):
        if isinstance(v, torch.Tensor): return v.contiguous().view(-1)
        return torch.tensor(v, dtype=torch.uint64, device="cuda")
    low_t     = _t(witness[c.x_rot_low])
    shifted_t = _t(witness[c.x_rot_shifted])
    return {
        c.z_x_rot_low:     gl_inv_batched(gl_sub(
            torch.full_like(low_t,     c.range_rescale.alpha), low_t)),
        c.z_x_rot_shifted: gl_inv_batched(gl_sub(
            torch.full_like(shifted_t, c.range_output.alpha),  shifted_t)),
    }


def rope_compile(c: RoPEClaim, _ch, cfg: LigeroConfig, base: int):
    """Compile RoPEClaim.

    Emits TWO linear constraints per rotation pair t = (seq, h, k_in_pair):
      cid 2t     (eq1): x_rot_target[lo[t]] − c·x[lo[t]] + s·x[hi[t]] = 0
      cid 2t+1   (eq2): x_rot_target[hi[t]] − s·x[lo[t]] − c·x[hi[t]] = 0
    where x_rot_target = c.x_rot_full when rescale_bits > 0, else c.x_rot.

    Per-row packets:
      x_rot_target: L2_RoPEXRot  (1 entry per slot, coef = 1)
      x:            L2_RoPEX     (2 entries per slot, cos/sin coefs)

    Rescale (optional): two L2_IdentityScalar linear families + two range
    LogUp quads (same shape as legacy).
    """
    ell = cfg.ELL
    SEQ, d_h, H = c.config.SEQ, c.config.d_h, c.config.heads
    assert d_h % 2 == 0, f"RoPE: d_h must be even, got {d_h}"
    L = SEQ * H * d_h
    n_rot_constraints = 2 * (SEQ * H * (d_h // 2))   # = L
    x_rot_target = c.x_rot_full if c.rescale_bits > 0 else c.x_rot

    c_l, s_l = _rope_cos_sin(c.config)
    cos_t = torch.tensor(c_l, dtype=torch.uint64, device="cuda")
    sin_t = torch.tensor(s_l, dtype=torch.uint64, device="cuda")

    row_pkts: List[Tuple[int, object]] = []
    for row_off in range(x_rot_target.n_rows(ell)):
        row_pkts.append((x_rot_target.row_start + row_off,
                          L2_RoPEXRot(base=base,
                                       x_rot_row_start=x_rot_target.row_start,
                                       SEQ=SEQ, H=H, d_h=d_h, L=L)))
    for row_off in range(c.x.n_rows(ell)):
        row_pkts.append((c.x.row_start + row_off,
                          L2_RoPEX(base=base,
                                    x_row_start=c.x.row_start,
                                    SEQ=SEQ, H=H, d_h=d_h, L=L,
                                    cos_t=cos_t, sin_t=sin_t)))
    cur = base + n_rot_constraints

    neg1 = (P - 1) % P
    quads: List[QuadraticConstraint] = []
    nz: List[Tuple[int, int, Any]] = []
    if c.rescale_bits > 0:
        offset = 1 << (c.output_width - 1)
        _emit_lin_csr_idscalar(
            c.x_rot_full, [c.x_rot, c.x_rot_low],
            [1 << c.rescale_bits, 1], L, ell, cur, row_pkts); cur += L
        _emit_lin_csr_idscalar(
            c.x_rot_shifted, [c.x_rot], [1], L, ell, cur, row_pkts); cur += L
        quads += _per_slot_quad(
            f"{c.x_rot.name}.RW[low]", c.x_rot_low, c.z_x_rot_low, c.z_x_rot_low,
            (P - c.range_rescale.alpha) % P, neg1, L, ell)
        quads += _per_slot_quad(
            f"{c.x_rot.name}.RW[shifted]", c.x_rot_shifted, c.z_x_rot_shifted, c.z_x_rot_shifted,
            (P - c.range_output.alpha) % P, neg1, L, ell)
        # x_rot_shifted = x_rot + offset → b = offset; placed after rotation (L)
        # and x_rot_full (L), at relative offset 2·L within this claim.
        nz.append((2 * L, L, offset))
    return row_pkts, quads, cur - base, _build_b_chunk(cur - base, nz)


COMPILE_FNS[RoPEClaim] = rope_compile


# ===========================================================================
# EmbeddingLookupClaim — verify x[i, j] = E[token_ids[i], j] for public
# token_ids and committed E (the embedding weight matrix) + committed x.
#
# Public-prompt verified inference setup: the prompt's token IDs are
# known to the verifier (the "what was asked"), the embedding table E is
# a model weight, and the resulting input x to layer 0 is committed.
# The verifier needs to be convinced x was honestly derived from the
# prompt via the table. That's exactly what this claim does.
#
# Constraint per (i, j) cell of x: a single linear identity reading the
# matching slot of E. With public token_ids the structure is fixed at
# compile time — no quadratics, no aux, no LogUp tables. Per cell:
# 2 nnz in the linear-CSR (one for x_slot, one for E_slot at index
# token_ids[i]·d + j). For SEQ tokens × d cols: SEQ·d constraints.
# ===========================================================================


@dataclass
class EmbeddingLookupClaim:
    x: Variable                # length SEQ·d (the embedded input to layer 0)
    E: Variable                # length vocab_size·d (the embedding table; committed weight)
    token_ids: List[int]       # public, length SEQ
    d: int                     # embedding dim per token


def embedding_lookup_sample(c: EmbeddingLookupClaim, ci, s_op):
    return None


def embedding_lookup_aux(c: EmbeddingLookupClaim, witness: dict, _ch) -> dict:
    return {}


def embedding_lookup_compile(c: EmbeddingLookupClaim, _ch, cfg: LigeroConfig,
                                     base: int):
    """Compile EmbeddingLookup.

    x side (regular): L2_IdentityScalar per row of x, coef = 1.
    E side (irregular permutation via public token_ids): L2_EmbedE per row of
    E. Each L2_EmbedE filters token_ids on the fly to find positions hitting
    that row — no per-element storage.

    Constraint IDs occupy [base, base + SEQ*d), one per (i, j) cell of x.
    """
    ell = cfg.ELL
    SEQ = len(c.token_ids)
    d = c.d
    L = SEQ * d
    # d | ELL is required only when vocab rows PACK multiple-per-witness-row
    # (rows_per_w = ell//d > 1). A single-row table (E.length <= ELL — e.g.
    # hadamard_broadcast's vocab=1 gain at Maverick's d=5120) never straddles,
    # so rows_per_w = max(1, ell//d) = 1 is exact.
    assert ell % d == 0 or c.E.length <= ell, (
        f"embedding_lookup_compile requires d | ELL or a single-row table "
        f"(d={d}, ell={ell}, E.length={c.E.length})")
    tok_t = torch.tensor(c.token_ids, dtype=torch.int64, device="cuda")
    row_pkts: List[Tuple[int, object]] = []
    for row_off in range(c.x.n_rows(ell)):
        row_pkts.append((c.x.row_start + row_off,
                          L2_IdentityScalar(base=base,
                                             var_row_start=c.x.row_start,
                                             L=L, coef=1)))
    for row_off in range(c.E.n_rows(ell)):
        row_pkts.append((c.E.row_start + row_off,
                          L2_EmbedE(base=base, E_row_start=c.E.row_start,
                                     d=d, token_ids=tok_t)))
    return row_pkts, [], L, None    # b = 0 throughout


COMPILE_FNS[EmbeddingLookupClaim] = embedding_lookup_compile


# ===========================================================================
# Register protocols.
# ===========================================================================

SAMPLE_FNS.update({
    MatmulClaim:           matmul_sample,
    AddClaim:              add_sample,
    LinCombClaim:          lincomb_sample,
    HadamardClaim:         hadamard_sample,
    RangeWordClaim:        range_word_sample,
    WordExtractionClaim:   word_extract_sample,
    PairedTlookupClaim:    paired_tlookup_sample,
    SiluClaim:             silu_sample,
    RmsNormClaim:          rmsnorm_sample,
    SoftmaxClaim:          softmax_sample,
    RoPEClaim:             rope_sample,
    EmbeddingLookupClaim:  embedding_lookup_sample,
})

AUX_FNS.update({
    MatmulClaim:           matmul_aux_witness,
    AddClaim:              add_aux_witness,
    LinCombClaim:          lincomb_aux,
    HadamardClaim:         hadamard_aux_witness,
    RangeWordClaim:        range_word_aux,
    WordExtractionClaim:   word_extract_aux,
    PairedTlookupClaim:    paired_tlookup_aux,
    SiluClaim:             silu_aux,
    RmsNormClaim:          rmsnorm_aux,
    SoftmaxClaim:          softmax_aux,
    RoPEClaim:             rope_aux,
    EmbeddingLookupClaim:  embedding_lookup_aux,
})

