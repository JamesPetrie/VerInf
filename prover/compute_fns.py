"""Per-claim witness compute functions.

Each function takes (claim, live) and returns {output_var: tensor} for the
phase-1 outputs allocated by the recording tape method. Pure math: no
tape/inputs side effects.

Phase-2 aux witnesses (y/u/p, LogUp z's, etc.) live in claims.py's
AUX_FNS — those run after challenges.

Dispatched via COMPUTE_FNS[ClaimType]. Eager tapes call them inline from
each tape.X method; lazy tapes defer via Tape.run_engine_pass which walks
self._deferred and dispatches the same way.
"""
from __future__ import annotations

from typing import Any, Callable, Dict

import torch

import math

from cuda_primitives import P, gl_add, gl_matmul, gl_mul, gl_sub, gl_inv, gl_inv_batched
from claims import (
    AddClaim, ConcatClaim, HadamardClaim, MatmulClaim,
    WordExtractionClaim, PairedTlookupClaim, RangeWordClaim,
    EmbeddingLookupClaim,
    RoPEClaim, _rope_cos_sin,
    SiluClaim, RmsNormClaim, _chunk_widths, RMS_LIMB_W, RMS_N_LIMBS,
    SoftmaxClaim, _softmax_exp_tables,
)
from core import Variable

# Lifted from tape.py: vectorized signed-floor decomposition + numpy field
# helpers. Imported here (rather than re-imported from tape) so tape can
# later be refactored without compute_fns following along.
from tape import _signed_floor_decomp, _to_signed_np, _to_field_np, _softmax_witness_vec

import numpy as np


COMPUTE_FNS: Dict[type, Callable[[Any, Dict[Variable, torch.Tensor]],
                                  Dict[Variable, torch.Tensor]]] = {}


def add_compute(claim: AddClaim, live):
    if claim.b is None:                # REVEAL pin (a == public_rhs): no output
        return {}
    return {claim.c: gl_add(live[claim.a], live[claim.b])}

COMPUTE_FNS[AddClaim] = add_compute


def lincomb_compute(claim, live):
    return {}                     # pure constraint: all xs already committed

from claims import LinCombClaim
COMPUTE_FNS[LinCombClaim] = lincomb_compute


CONCAT_TAMPER = {}      # dst var name -> override tensor (negative tests)


def concat_compute(claim: ConcatClaim, live):
    if claim.dst.name in CONCAT_TAMPER:
        return {claim.dst: CONCAT_TAMPER[claim.dst.name].to("cuda")}
    # cat via int64 bit-view (uint64 op coverage is spotty across builds)
    parts = [live[v].contiguous().view(-1).view(torch.int64) for v in claim.srcs]
    return {claim.dst: torch.cat(parts).view(torch.uint64)}


COMPUTE_FNS[ConcatClaim] = concat_compute


def range_word_compute(claim: RangeWordClaim, live):
    """RangeWordClaim has no phase-1 outputs — the claim just asserts
    x ∈ table.T (verified via LogUp). Returns {} so the engine-pass
    dispatch is uniform across claim types."""
    return {}

COMPUTE_FNS[RangeWordClaim] = range_word_compute


def _matmul_raw(A: torch.Tensor, B: torch.Tensor, *,
                 m: int, k: int, n: int, heads: int, head_dim: int,
                 transpose_b: bool) -> torch.Tensor:
    """Compute C = A·B (or A·B^T), single- or multi-head. Returns flat (-1,)."""
    H, K = heads, head_dim
    if H == 1:
        if transpose_b:
            return gl_matmul(A.view(m, k), B.view(n, k).t().contiguous()).view(-1)
        return gl_matmul(A.view(m, k), B.view(k, n)).view(-1)
    A3 = A.view(m, H, K)
    B3 = B.view(n, H, K) if transpose_b else B.view(K, H, n)
    C3 = torch.empty((m, H, n), dtype=torch.uint64, device="cuda")
    for h in range(H):
        Ah = A3[:, h, :].contiguous()
        Bh = (B3[:, h, :].t().contiguous() if transpose_b
              else B3[:, h, :].contiguous())
        C3[:, h, :] = gl_matmul(Ah, Bh)
    return C3.contiguous().view(-1)


def matmul_compute(claim: MatmulClaim, live):
    A, B = live[claim.A], live[claim.B]
    c_full = _matmul_raw(A, B, m=claim.m, k=claim.k, n=claim.n,
                          heads=claim.heads, head_dim=claim.head_dim,
                          transpose_b=claim.transpose_b)
    if claim.rescale_bits == 0:
        return {claim.C: c_full}
    k_resc = 1 << claim.rescale_bits
    c_resc, c_low, c_shifted = _signed_floor_decomp(
        c_full, k_resc, claim.output_width)
    return {claim.C: c_resc, claim.C_full: c_full,
            claim.C_low: c_low, claim.C_shifted: c_shifted}

COMPUTE_FNS[MatmulClaim] = matmul_compute


def hadamard_compute(claim: HadamardClaim, live):
    c_full = gl_mul(live[claim.a], live[claim.b])
    if claim.rescale_bits == 0:
        return {claim.c: c_full}
    k_resc = 1 << claim.rescale_bits
    c_resc, c_low, c_shifted = _signed_floor_decomp(
        c_full, k_resc, claim.output_width)
    return {claim.c: c_resc, claim.c_full: c_full,
            claim.c_low: c_low, claim.c_shifted: c_shifted}

COMPUTE_FNS[HadamardClaim] = hadamard_compute


def word_extract_compute(claim: WordExtractionClaim, live):
    """Two directions, distinguished by which side is in `live`:
      - extract: claim.x in live → derive `words` via bit-shifts on x.
      - combine: claim.x not in live → derive x = Σ coeffs·words.
    The extract direction only works when coeffs are powers of two (the
    case tape.word_extract produces); combine direction is fully general.
    """
    L = claim.length
    if claim.x in live:
        x = live[claim.x]
        # Infer B from coeffs[1] (coeffs[0] is always 1 in the extract case).
        N = len(claim.words)
        B = (claim.coeffs[1].bit_length() - 1) if N > 1 else 0
        mask = (1 << B) - 1
        xi = x.contiguous().view(torch.int64)   # torch CUDA has no uint64 >>; bits are identical
        return {claim.words[n]: ((xi >> (n * B)) & mask).to(torch.uint64) for n in range(N)}
    out = torch.zeros(L, dtype=torch.uint64, device="cuda")
    for n, w in enumerate(claim.words):
        coef_t = torch.full((L,), claim.coeffs[n], dtype=torch.uint64, device="cuda")
        out = gl_add(out, gl_mul(live[w], coef_t))
    return {claim.x: out}

COMPUTE_FNS[WordExtractionClaim] = word_extract_compute


def paired_tlookup_compute(claim: PairedTlookupClaim, live):
    x = live[claim.x]
    shift_t = torch.full_like(x, claim.shift % P)
    x_shifted = gl_add(x, shift_t)
    # Clamp the gather index: an out-of-table key (e.g. a wrapped negative gap
    # from a cheating prover) must not crash the prover with an OOB index_select;
    # it gets a garbage y, and the table's LogUp SUM identity then rejects (the
    # multiplicity side counts only in-range keys). No-op for honest, in-range x.
    idx = x_shifted.to(torch.int64).clamp_(0, claim.table.T_Y.numel() - 1)
    y = torch.index_select(claim.table.T_Y, 0, idx)
    return {claim.y: y}

COMPUTE_FNS[PairedTlookupClaim] = paired_tlookup_compute


def embedding_lookup_compute(claim: EmbeddingLookupClaim, live):
    E = live[claim.E]
    tok_t = torch.tensor(claim.token_ids, dtype=torch.int64, device="cuda")
    vocab_size = E.numel() // claim.d
    return {claim.x: E.view(vocab_size, claim.d).index_select(0, tok_t).contiguous().view(-1)}

COMPUTE_FNS[EmbeddingLookupClaim] = embedding_lookup_compute


def rope_compute(claim: RoPEClaim, live):
    """RoPE rotation, optionally followed by signed-floor rescale.

    Pure-torch on device: c/s tables build directly into uint64 cuda
    tensors and the (cb·x_lo − sb·x_hi) / (sb·x_lo + cb·x_hi) products run
    in int64 on GPU. No host round-trip. uint64 arithmetic isn't
    implemented on CUDA in this PyTorch build, so the signed reinterpret
    and field-rep reverse go through int64 view + wraparound add. See
    `_signed_floor_decomp` for the same pattern."""
    sc = claim.config
    SEQ, d_h, heads = sc.SEQ, sc.d_h, sc.heads
    half = d_h // 2
    c_l, s_l = _rope_cos_sin(sc)

    x_u = live[claim.x]
    device = x_u.device
    FIELD_GAP = (1 << 64) - P                                              # 2^32 - 1
    P_HALF    = (P - 1) // 2

    def to_signed(u):
        i = u.view(torch.int64)
        is_neg = (i < 0) | (i > P_HALF)
        return torch.where(is_neg, i + FIELD_GAP, i)

    x_signed = to_signed(x_u)
    c_signed = to_signed(torch.tensor(c_l, dtype=torch.uint64, device=device))
    s_signed = to_signed(torch.tensor(s_l, dtype=torch.uint64, device=device))

    x_3d = x_signed.view(SEQ, heads, d_h)
    x_lo = x_3d[:, :, :half]
    x_hi = x_3d[:, :, half:]
    cb = c_signed.view(SEQ, half).unsqueeze(1)
    sb = s_signed.view(SEQ, half).unsqueeze(1)

    out_lo = cb * x_lo - sb * x_hi
    out_hi = sb * x_lo + cb * x_hi
    out = torch.empty((SEQ, heads, d_h), dtype=torch.int64, device=device)
    out[:, :, :half] = out_lo
    out[:, :, half:] = out_hi

    out_flat = out.view(-1)
    # int64-view select: torch CUDA has no `where` for uint64; bits are identical.
    x_rot_full = torch.where(out_flat >= 0,
                              out_flat,
                              out_flat - FIELD_GAP).view(torch.uint64)

    if claim.rescale_bits == 0:
        return {claim.x_rot: x_rot_full}
    k_resc = 1 << claim.rescale_bits
    x_rot, x_rot_low, x_rot_shifted = _signed_floor_decomp(
        x_rot_full, k_resc, claim.output_width)
    return {claim.x_rot: x_rot, claim.x_rot_full: x_rot_full,
            claim.x_rot_low: x_rot_low, claim.x_rot_shifted: x_rot_shifted}

COMPUTE_FNS[RoPEClaim] = rope_compute


def silu_compute(claim: SiluClaim, live):
    """Compute all phase-1 silu witnesses. Mirrors the body of tape.silu;
    the only difference is reading inputs from `live` instead of WitnessTensor.data
    and returning the values keyed by the claim's Variables."""
    sc = claim.config
    L = claim.length
    b, T_LEN = sc.b, sc.T_LEN
    b_2, b_3, b_4 = sc.b_2, sc.b_3, sc.b_4
    w2_mod = 1 << sc.width_2
    w3_mod = 1 << sc.width_3
    w4_mod = 1 << sc.width_4
    P_half = (P - 1) // 2

    out: Dict[Variable, torch.Tensor] = {}
    if sc.rescale_bits > 0:
        # Rescale: decompose live[x_in] into (x_low, x_internal=claim.x, x_shifted).
        r_resc = sc.rescale_bits
        k_resc = 1 << r_resc
        x_in_cpu = live[claim.x_in].cpu().tolist()
        x_in_signed = [v - P if v > P_half else v for v in x_in_cpu]
        x_low_cpu = [v % k_resc for v in x_in_signed]
        x_internal_cpu = [v // k_resc for v in x_in_signed]
        offset = 1 << (sc.width_2 - 1)
        x_shifted_cpu = [(v + offset) % P for v in x_internal_cpu]
        x_data = torch.tensor([v % P for v in x_internal_cpu],
                               dtype=torch.uint64, device="cuda")
        out[claim.x]         = x_data
        out[claim.x_low]     = torch.tensor([v % P for v in x_low_cpu],
                                             dtype=torch.uint64, device="cuda")
        out[claim.x_shifted] = torch.tensor(x_shifted_cpu, dtype=torch.uint64, device="cuda")
    else:
        x_data = live[claim.x]

    # Bit-exact bulk-numpy sign-magnitude + 5-chunk decomposition. Replaces
    # per-cell Python list-comps over SEQ*d_ff cells (the dominant witness-
    # compute cost, esp. at long context). All ops are uint64; every constant
    # is wrapped in np.uint64 to avoid numpy's uint64+int -> float64 promotion.
    x_np = x_data.cpu().numpy()
    P_np, Ph_np = np.uint64(P), np.uint64(P_half)
    b_np, TL_np = np.uint64(b), np.uint64(T_LEN)
    b2_np, b3_np, b4_np = np.uint64(b_2), np.uint64(b_3), np.uint64(b_4)
    w2_np, w3_np, w4_np = np.uint64(w2_mod), np.uint64(w3_mod), np.uint64(w4_mod)
    sign_np = (x_np > Ph_np)
    mag_np  = np.where(sign_np, (P_np - x_np) % P_np, x_np)
    a0_np = mag_np % b_np
    a1_np = (mag_np // b_np)  % TL_np
    a2_np = (mag_np // b2_np) % w2_np
    a3_np = (mag_np // b3_np) % w3_np
    a4_np = (mag_np // b4_np) % w4_np
    g_np  = b2_np * a2_np + b3_np * a3_np + b4_np * a4_np
    sign_u = sign_np.astype(np.uint64)
    key_np = sign_u * TL_np + a1_np
    is_high_np = (g_np != 0).astype(np.uint64)

    def _t(arr):
        return torch.from_numpy(np.ascontiguousarray(arr)).to("cuda")
    sign_d, mag_d = _t(sign_u), _t(mag_np)
    a0_d, a1_d = _t(a0_np), _t(a1_np)
    a2_d, a3_d, a4_d = _t(a2_np), _t(a3_np), _t(a4_np)
    g_d, key_d, is_high_d = _t(g_np), _t(key_np), _t(is_high_np)
    C_d = gl_mul(sign_d, x_data)
    inv_g_d = gl_inv(g_d)
    output_sat_d = gl_sub(x_data, C_d)
    y_d = torch.index_select(claim.silu_table.T_Y, 0, key_d.to(torch.int64))
    mux_a_d = gl_mul(is_high_d, y_d)
    mux_b_d = gl_mul(is_high_d, output_sat_d)
    output_d = gl_add(gl_sub(y_d, mux_a_d), mux_b_d)

    out.update({
        claim.sign: sign_d, claim.magnitude: mag_d, claim.C: C_d,
        claim.a_0: a0_d, claim.a_1: a1_d, claim.a_2: a2_d,
        claim.a_3: a3_d, claim.a_4: a4_d,
        claim.g: g_d, claim.inv_g: inv_g_d, claim.is_high: is_high_d,
        claim.key: key_d, claim.output_sat: output_sat_d,
        claim.mux_a: mux_a_d, claim.mux_b: mux_b_d, claim.y: y_d,
        claim.output: output_d,
    })
    return out

COMPUTE_FNS[SiluClaim] = silu_compute


def rmsnorm_compute(claim: RmsNormClaim, live):
    """Compute all phase-1 rmsnorm witnesses. Replaces per-cell .cpu().tolist()
    + Python list-comprehensions over 373K-element tensors with bulk numpy
    on CPU after one D2H per tensor.

    Pre-rescale x_in split: uint64 bitwise / shift via numpy (CUDA PyTorch
    doesn't implement uint64 ops). Per-row S = Σ X_sq[b] mod P via
    lo/hi 32-bit split + int64 reductions (fits int64 since d · 2^32 < 2^44).
    The per-batch y_int search stays in Python — B ~ SEQ is small and the
    body is constant work per batch — but starts from math.isqrt(...) for
    an exact ceil(√) seed so the while-loops are normally 0-1 steps.

    Bit-identical to the prior implementation; the verifier sees the same
    witness tensors and ACCEPTs identically."""
    sc = claim.config
    B, d, s, eps_int = sc.B, sc.d, sc.s, sc.eps_int
    slack_max = 1 << sc.slack_width
    magic = sc.magic
    out: Dict[Variable, torch.Tensor] = {}

    def _split(v: int, widths) -> list:
        """Chunk v at 16-bit strides into the given widths (top chunk narrow).
        Tight by construction: asserts v fits the window."""
        assert v < (1 << (16 * (len(widths) - 1) + widths[-1])), \
            f"rmsnorm witness value {v} exceeds its derived window {widths}"
        return [(v >> (16 * n)) & ((1 << wn) - 1) for n, wn in enumerate(widths)]

    if sc.rescale_bits > 0:
        r = sc.rescale_bits
        k_resc = 1 << r
        offset = 1 << 15   # signed shift against the 16-bit slack table
        x_in_d = live[claim.x_in]
        device = x_in_d.device
        x_in_np      = x_in_d.cpu().numpy()                                # uint64
        x_low_np     = x_in_np & np.uint64(k_resc - 1)
        x_high_np    = x_in_np >> np.uint64(r)
        x_shifted_np = x_high_np + np.uint64(offset)
        P_np = np.uint64(P)
        x_shifted_np = np.where(x_shifted_np >= P_np, x_shifted_np - P_np, x_shifted_np)
        x_data = torch.from_numpy(x_high_np).to(device)
        out[claim.x]         = x_data
        out[claim.x_low]     = torch.from_numpy(x_low_np).to(device)
        out[claim.x_shifted] = torch.from_numpy(x_shifted_np).to(device)
    else:
        x_data = live[claim.x]
        device = x_data.device

    X_sq_t = gl_mul(x_data, x_data)
    # Per-row S = sum(X_sq[b]) mod P via numpy CPU lo/hi 32-bit split. Each
    # X_sq cell < P < 2^64; lo and hi each < 2^32; sums fit int64 for d < 2^31.
    X_sq_np = X_sq_t.cpu().numpy().reshape(B, d)
    lo_sum = (X_sq_np & np.uint64(0xFFFFFFFF)).astype(np.int64).sum(axis=1)
    hi_sum = (X_sq_np >> np.uint64(32)).astype(np.int64).sum(axis=1)
    d_eps = (d * eps_int) % P

    S_l, S_total_l, y_l, y_m1_l = [], [], [], []
    q1_l, q2_l, s_lo_l, s_hi_l = [], [], [], []
    S_limbs_l = [[] for _ in range(RMS_N_LIMBS)]
    bracket_l = {t: dict(H=[[] for _ in range(RMS_N_LIMBS)], gl=[[], []],
                         g0h=[], g1h=[], G2=[])
                 for t in ("lo", "hi")}
    ym1_chunks_l = []
    _s_cap = RMS_LIMB_W * RMS_N_LIMBS
    for b in range(B):
        S_b = ((int(hi_sum[b]) << 32) + int(lo_sum[b])) % P
        S_tot_b = (S_b + d_eps) % P
        assert S_tot_b < (1 << _s_cap), (
            f"rmsnorm row {b}: S_total={S_tot_b} exceeds the 2^{_s_cap} limb cap "
            f"(row energy too large for the claim's integer bracket; raise LIMB_W)")
        # Smallest y >= 1 with y² · S_tot >= magic. ceil(sqrt(magic/S_tot))
        # is an exact starting point via isqrt; the while-loops adjust the
        # rare off-by-one from non-square ratios.
        if S_tot_b == 0:
            y_int = 1
        else:
            y_int = max(1, math.isqrt((magic + S_tot_b - 1) // S_tot_b))
        while y_int * y_int * S_tot_b < magic:
            y_int += 1
        while y_int > 1 and (y_int - 1) * (y_int - 1) * S_tot_b >= magic:
            y_int -= 1
        ym1 = y_int - 1
        q1_b = (y_int * y_int) % P
        q2_b = (ym1 * ym1) % P
        slack_lo = (q1_b * S_tot_b - magic) % P
        slack_hi = (magic - 1 - q2_b * S_tot_b) % P
        assert int(slack_lo) < slack_max and int(slack_hi) < slack_max
        S_l.append(S_b);          S_total_l.append(S_tot_b)
        y_l.append(y_int);        y_m1_l.append(ym1)
        q1_l.append(q1_b);        q2_l.append(q2_b)
        s_lo_l.append(int(slack_lo)); s_hi_l.append(int(slack_hi))
        # Wrap-free-bracket limbs (all exact non-negative integers < P; see
        # rmsnorm-bracket-fix.md). Limbs + carry lows are LIMB_W-wide; the
        # carry highs g0h/g1h/G2 are 16-bit-chunked.
        Lw = RMS_LIMB_W
        Lmask = (1 << Lw) - 1
        ym1_chunks_l.append(_split(ym1, _chunk_widths(sc.y_width)))
        limbs = [(S_tot_b >> (Lw * n)) & Lmask for n in range(RMS_N_LIMBS)]
        for n in range(RMS_N_LIMBS):
            S_limbs_l[n].append(limbs[n])
        for tag, q in (("lo", q1_b), ("hi", q2_b)):
            acc = bracket_l[tag]
            G0 = q * limbs[0]
            g0l, g0h = G0 & Lmask, G0 >> Lw
            G1 = q * limbs[1] + g0h
            g1l, g1h = G1 & Lmask, G1 >> Lw
            G2 = q * limbs[2] + g1h
            # Sanity: the chain reassembles the bracket identity exactly.
            ref = magic + int(slack_lo) if tag == "lo" else magic - 1 - int(slack_hi)
            assert (G2 << (2 * Lw)) + (g1l << Lw) + g0l == ref, "rmsnorm limb chain broke"
            for k, hv in enumerate((G0, q * limbs[1], q * limbs[2])):
                acc["H"][k].append(hv % P)
            acc["gl"][0].append(g0l); acc["gl"][1].append(g1l)
            acc["g0h"].append(_split(g0h, _chunk_widths(sc.g0h_width)))
            acc["g1h"].append(_split(g1h, _chunk_widths(sc.g1h_width)))
            acc["G2"].append(_split(G2, _chunk_widths(sc.G2_width)))

    y_t = torch.tensor([v % P for v in y_l], dtype=torch.uint64, device=device)
    y_per_cell = y_t.view(B, 1).expand(B, d).contiguous().view(-1)
    out_full_t = gl_mul(x_data, y_per_cell)

    if sc.output_rescale_bits > 0:
        k_out = 1 << sc.output_rescale_bits
        out_rescaled_d, out_low_d, out_shifted_d = _signed_floor_decomp(
            out_full_t, k_out, sc.output_width)
        out[claim.output_full]    = out_full_t
        out[claim.output_low]     = out_low_d
        out[claim.output_shifted] = out_shifted_d
        out[claim.output] = out_rescaled_d
    else:
        out[claim.output] = out_full_t

    out[claim.X_sq]    = X_sq_t
    out[claim.S]       = torch.tensor(S_l,       dtype=torch.uint64, device=device)
    out[claim.S_total] = torch.tensor(S_total_l, dtype=torch.uint64, device=device)
    out[claim.y]       = y_t
    out[claim.y_m1]    = torch.tensor(y_m1_l,    dtype=torch.uint64, device=device)
    out[claim.q1]      = torch.tensor(q1_l,      dtype=torch.uint64, device=device)
    out[claim.q2]      = torch.tensor(q2_l,      dtype=torch.uint64, device=device)
    out[claim.s_lo]    = torch.tensor(s_lo_l,    dtype=torch.uint64, device=device)
    out[claim.s_hi]    = torch.tensor(s_hi_l,    dtype=torch.uint64, device=device)
    slack_widths = _chunk_widths(sc.slack_width)
    for n, wn in enumerate(slack_widths):
        mask = (1 << wn) - 1
        out[claim.s_lo_chunks[n]] = torch.tensor(
            [(v >> (16 * n)) & mask for v in s_lo_l],
            dtype=torch.uint64, device=device)
        out[claim.s_hi_chunks[n]] = torch.tensor(
            [(v >> (16 * n)) & mask for v in s_hi_l],
            dtype=torch.uint64, device=device)

    def _tens(vals):
        return torch.tensor(vals, dtype=torch.uint64, device=device)
    for n, var in enumerate(claim.ym1_chunks):
        out[var] = _tens([row[n] for row in ym1_chunks_l])
    for n, var in enumerate(claim.S_limbs):
        out[var] = _tens(S_limbs_l[n])
    for tag, (H_vars, gl_vars, g0h_vars, g1h_vars, G2_vars) in (
            ("lo", (claim.lo_H, claim.lo_gl, claim.lo_g0h_chunks,
                    claim.lo_g1h_chunks, claim.lo_G2_chunks)),
            ("hi", (claim.hi_H, claim.hi_gl, claim.hi_g0h_chunks,
                    claim.hi_g1h_chunks, claim.hi_G2_chunks))):
        acc = bracket_l[tag]
        for k, var in enumerate(H_vars):
            out[var] = _tens(acc["H"][k])
        for k, var in enumerate(gl_vars):
            out[var] = _tens(acc["gl"][k])
        for chunks_key, vars_ in (("g0h", g0h_vars), ("g1h", g1h_vars),
                                   ("G2", G2_vars)):
            for j, var in enumerate(vars_):
                out[var] = _tens([row[j] for row in acc[chunks_key]])
    return out

COMPUTE_FNS[RmsNormClaim] = rmsnorm_compute


def softmax_compute(claim: SoftmaxClaim, live):
    """Compute all phase-1 softmax witnesses via the vectorized helper
    `_softmax_witness_vec` (works for both no-rescale and rescale paths;
    the rescale path simply runs the same bracket on the rescaled x)."""
    sc = claim.config
    L = claim.length
    B, M = sc.B, sc.M
    out: Dict[Variable, torch.Tensor] = {}

    if sc.rescale_bits > 0:
        # Decompose live[x_in] into (x_low, x_internal, x_shifted); bracket runs on x_internal.
        r = sc.rescale_bits
        k_resc = 1 << r
        P_half = (P - 1) // 2
        x_in_cpu = live[claim.x_in].cpu().tolist()
        x_in_signed = [v - P if v > P_half else v for v in x_in_cpu]
        x_low_cpu  = [v % k_resc for v in x_in_signed]
        x_high_cpu = [v // k_resc for v in x_in_signed]
        offset = 1 << 15
        x_shifted_cpu = [(v + offset) % P for v in x_high_cpu]
        x_internal_t = torch.tensor([v % P for v in x_high_cpu],
                                     dtype=torch.uint64, device="cuda")
        out[claim.x]         = x_internal_t
        out[claim.x_low]     = torch.tensor([v % P for v in x_low_cpu],
                                             dtype=torch.uint64, device="cuda")
        out[claim.x_shifted] = torch.tensor(x_shifted_cpu, dtype=torch.uint64, device="cuda")
        x_for_bracket = x_internal_t
    else:
        x_for_bracket = live[claim.x]

    T_A_data, T_B_data = _softmax_exp_tables(sc)   # already np.uint64 arrays
    T_A_np = np.asarray(T_A_data, dtype=np.uint64)
    T_B_np = np.asarray(T_B_data, dtype=np.uint64)
    x_in_np = x_for_bracket.detach().cpu().numpy().astype(np.uint64)
    sm = _softmax_witness_vec(
        x_in_np, B=B, M=M, s_x=sc.s_x, s_c=sc.s_c, s_y=sc.s_y,
        T_A_np=T_A_np, T_B_np=T_B_np, Z_max=sc.Z_max,
        aux_chunk_width=sc.aux_chunk_width,
        saturate=sc.saturate, Z_high_width=sc.Z_high_width,
        causal=sc.causal, heads=sc.heads, round_up=getattr(sc, "round_up", False))

    def _u64(arr): return torch.from_numpy(arr).to("cuda")
    out[claim.c2]         = _u64(sm["c2"])
    out[claim.c2_shifted] = _u64(sm["c2_shifted"])
    out[claim.z]          = _u64(sm["z"])
    out[claim.y_A]        = _u64(sm["y_A"])
    out[claim.y_B]        = _u64(sm["y_B"])
    out[claim.s1]         = _u64(sm["s1"])
    out[claim.s2]         = _u64(sm["s2"])
    out[claim.r_lo]       = _u64(sm["r_lo"])
    out[claim.r_hi]       = _u64(sm["r_hi"])
    if sc.saturate:
        zh = _u64(sm["z_high"])
        # Fermat inv: 0 where z_high=0 (gl_inv_batched diverges on 0, so mask).
        # int64-view select: torch CUDA has no `where`/`eq` for uint64; bits are identical.
        zh_i     = zh.view(torch.int64)
        is_zero  = zh_i == 0
        zh_safe  = torch.where(is_zero, torch.ones_like(zh_i), zh_i).view(torch.uint64)
        inv_safe = gl_inv_batched(zh_safe).view(torch.int64)
        inv_zh   = torch.where(is_zero, torch.zeros_like(zh_i), inv_safe).view(torch.uint64)
        out[claim.z_high]     = zh
        out[claim.inv_z_high] = inv_zh
        out[claim.is_high]    = _u64(sm["is_high"])
        out[claim.y_A_raw]    = _u64(sm["y_A_raw"])
        out[claim.y_B_raw]    = _u64(sm["y_B_raw"])
        out[claim.mux_y_A]    = _u64(sm["mux_y_A"])
        out[claim.mux_y_B]    = _u64(sm["mux_y_B"])
    return out

COMPUTE_FNS[SoftmaxClaim] = softmax_compute


# ===========================================================================
# Fold consumers — incremental input absorption for barrier claims.
#
# A claim registered in FOLD_FNS consumes its "foldable" inputs one at a time
# as their producer claims retire, so the streaming engine can free each
# stream immediately instead of holding all of them resident for one atomic
# compute call (FreivaldsCombineClaim at T=1000 × E=128 otherwise pins ~17 GB
# of expert streams). Field addition is exact and commutative, so absorption
# order cannot change a single committed bit — validated by the golden-root
# test (fold on vs off must produce identical Merkle roots).
#
# Registry entry: FOLD_FNS[ClaimType] = dict(
#   foldable=fn(claim) -> [Variables consumed incrementally],
#   init=fn(claim, live) -> state           (anchors like the mask come from live),
#   absorb=fn(claim, state, var, tensor),
#   finalize=fn(claim, state, live) -> outs (MUST absorb any remaining
#       foldables from live itself — committed-rather-than-produced inputs
#       are never offered),
#   aux_absorb=fn(claim, state, var, tensor, ch)   (phase-2, optional),
#   aux_finalize=fn(claim, state, live, ch) -> aux outs)
#
# LIGERO_NO_FOLD=1 disables folding entirely (the A/B switch for the
# golden-root regression).
# ===========================================================================

FOLD_FNS: Dict[type, dict] = {}


class FoldRunner:
    """Per-sweep fold dispatch, shared by run_engine_pass and _stream_sweep.
    Construct fresh per sweep (state must not leak across sound-mode rounds)."""

    def __init__(self, deferred, ch_by_index=None, want_aux=False):
        import os
        self.enabled = not os.environ.get("LIGERO_NO_FOLD")
        self.absorbers = {}        # producer Variable -> [(fold claim idx, claim), ...]
        self.only_consumer = {}    # producer Variable -> safe to free at absorb?
        self.index_of = {}         # id(fold claim) -> deferred index (its ch slot)
        self.states = {}
        self.ch = ch_by_index
        self.want_aux = want_aux
        if not self.enabled:
            return
        consumers: Dict[object, list] = {}
        for i, (c, ivars, _se) in enumerate(deferred):
            for v in ivars:
                consumers.setdefault(v, []).append(i)
        for i, (c, ivars, _se) in enumerate(deferred):
            f = FOLD_FNS.get(type(c))
            if not f:
                continue
            self.index_of[id(c)] = i
            for v in f["foldable"](c):
                self.absorbers.setdefault(v, []).append((i, c))
        for v, ents in self.absorbers.items():
            # Free-at-absorb only when a SINGLE fold claim consumes v and is
            # its only consumer. A var SHARED by several fold claims (e.g. one
            # broadcast `ones` committed once and reused by every MoE layer's
            # gate combine) must instead be absorbed at each claim's FINALIZE:
            # absorbing at land time would init a LATER claim's state, whose
            # mask anchor doesn't exist yet in `live`.
            self.only_consumer[v] = (len(ents) == 1
                                     and consumers.get(v, []) == [ents[0][0]])

    def is_fold(self, claim):
        return self.enabled and type(claim) in FOLD_FNS

    def _state(self, claim, live):
        st = self.states.get(id(claim))
        if st is None:
            st = self.states[id(claim)] = FOLD_FNS[type(claim)]["init"](claim, live)
        return st

    def offer(self, v, live, allow_free=True):
        """Absorb v (just landed in live) if a fold claim wants it; free it
        when this fold is its only consumer. Returns True if freed.

        Only single-absorber vars absorb at land time (the memory-critical
        case: each expert stream feeds exactly one combine, whose mask is
        already live because routing precedes the expert matmuls in tape
        order). A var shared by several fold claims defers to each claim's
        finalize — a later claim's state cannot init yet."""
        ents = self.absorbers.get(v) if self.enabled else None
        if not ents or len(ents) > 1 or v not in live:
            return False
        i, claim = ents[0]
        st = self._state(claim, live)
        self._absorb(claim, st, v, live)
        if allow_free and self.only_consumer.get(v):
            live.pop(v, None)
            return True
        return False

    def _absorb(self, claim, st, v, live):
        """Absorb v into THIS claim's state (never routed via the absorber
        registry — when v is shared, the registry points elsewhere)."""
        f = FOLD_FNS[type(claim)]
        t = live[v]
        t = t() if callable(t) else t
        f["absorb"](claim, st, v, t)
        if self.want_aux and "aux_absorb" in f:
            i = self.index_of[id(claim)]
            f["aux_absorb"](claim, st, v, t, self.ch[i] if self.ch else None)
        st.setdefault("_absorbed", set()).add(v)

    def finalize(self, claim, live):
        f = FOLD_FNS[type(claim)]
        st = self._state(claim, live)
        absorbed = st.get("_absorbed", set())
        for v in f["foldable"](claim):       # committed/never-offered/shared inputs
            if v not in absorbed:
                self._absorb(claim, st, v, live)
        return f["finalize"](claim, st, live)

    def aux_finalize(self, claim, live, ch):
        f = FOLD_FNS[type(claim)]
        return f["aux_finalize"](claim, self._state(claim, live), live, ch)
