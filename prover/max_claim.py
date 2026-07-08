"""MaxClaim: prove v*[t] = max_i l[t,i], emit gap[t,i] = v* - l >= 0, and select
the *committed* output token's gap, gap_o[t] = gap[t, o_t], via a hidden one-hot O.

One new claim type (one verifier handler) for the unexplained-information bound.
All checks are internal, reusing the existing constraint families:
  * A·A = A            (one-hot is boolean)        -- per-slot quad, like Hadamard
  * Σ_i A[t,i] = 1     (exactly one selected)       -- L2_RowSumPerSlotVector = 1
  * Al = A·l, v* = Σ_i Al = l[argmax]               -- quad + rowsum
  * gap + l = v*  (broadcast)                        -- linear (StrideOneToMany)
  * gap ∈ [0, gap_max)  (so v* >= every logit)       -- (α-gap)·z = 1 range LogUp
  * O·O = O, Σ_i O[t,i] = 1                          -- select one-hot (boolean, sums to 1)
  * tok_t = Σ_i i·O[t,i]                             -- O is the one-hot OF the committed token
  * Ogap = O·gap, gap_o = Σ_i Ogap = gap[t, tok_t]  -- quad + rowsum (hidden select)

A one-hot A (boolean, sums to 1) with gap >= 0 pins v* to the exact max: A picks
one logit (v* = l[k]); gap >= 0 forces v* >= all logits; together v* = max.

The OUTPUT is committed as a token-id vector `tok` (length T), blinded exactly
like the model weights (a phase-1 commit hidden behind the Merkle root). A second
committed one-hot O is the select gadget: tok_t = Σ_i i·O[t,i] ties O to the
committed token, and O·O=O / Σ_i O=1 make it a valid one-hot, so gap_o = Σ_i O·gap
= gap[t, tok_t] picks the output token's gap. The claim enforces only that `tok`
is *some* committed token (and O its one-hot); that `tok` is the realized output
is the same trust the committed weights carry. Nothing reveals which token tok_t
is -- the surprisal numerator gap_o^2 is computed over a hidden, committed output.
"""
import math
from dataclasses import dataclass
from typing import List, Tuple

import torch

import core
from core import P, Variable, LigeroConfig, SAMPLE_FNS, AUX_FNS
from claims import (COMPILE_FNS, QuadraticConstraint, QuadFamily, L2_IdentityScalar,
                    L2_RowSumPerSlotVector, L2_StrideOneToManyScalar, _build_b_chunk)
import compute_fns as _cf
from cuda_primitives import gl_mul, gl_sub, gl_neg, gl_inv_batched, lookup_multiplicities_into


@dataclass
class MaxClaim:
    l: Variable           # logits (T*V), committed by the caller
    A: Variable           # one-hot argmax (T*V), committed
    Al: Variable          # A·l (T*V), derived
    vstar: Variable       # max per row (T), derived
    gap: Variable         # v* - l (T*V), derived
    neg_gap: Variable     # -(gap) = l - v* (T*V), derived (for -gap^2 with no pin)
    z_gap: Variable       # range-LogUp aux (T*V)
    tok: Variable         # OUTPUT token ids (T), committed (blinded like weights)
    O: Variable           # one-hot of tok (T*V), committed aux for the select
    Ogap: Variable        # O·gap (T*V), derived
    gap_o: Variable       # Σ_i Ogap = gap[t, tok_t] (T), derived (the output token's gap)
    table: object         # gap range Table [0, gap_max)
    T: int
    V: int

    @property
    def length(self):
        return self.T * self.V


TEST_TAMPER = {}     # "A" -> (T*V,) uint64: override the derived argmax one-hot


def max_sample(c: MaxClaim, ci, s_op):
    return None                       # α owned by the gap table's TableSettlement


def max_aux(c: MaxClaim, witness, _ch):
    g = witness[c.gap]
    g = g if isinstance(g, torch.Tensor) else torch.tensor(g, dtype=torch.uint64, device="cuda")
    g = g.contiguous().view(-1)
    alpha = torch.full_like(g, c.table.alpha)
    return {c.z_gap: gl_inv_batched(gl_sub(alpha, g))}


def max_compute(c: MaxClaim, live):
    T, V = c.T, c.V
    l = live[c.l].contiguous().view(T, V)
    if "A" in TEST_TAMPER:
        A = TEST_TAMPER["A"].to("cuda").view(T, V)
    else:
        am = to_signed(l.reshape(-1)).view(T, V).argmax(dim=1)
        Ai = torch.zeros(T, V, dtype=torch.int64, device="cuda")
        Ai[torch.arange(T, device="cuda"), am] = 1
        A = Ai.to(torch.uint64)
    O = live[c.O].contiguous().view(T, V)
    Al = gl_mul(A.reshape(-1), l.reshape(-1))
    idx = A.to(torch.int64).argmax(dim=1)                 # the one-hot position
    li = l.view(torch.int64)                              # bit-view; gather preserves bits
    vstar = li[torch.arange(T, device="cuda"), idx].contiguous().view(torch.uint64)  # l[argmax]
    vbc = vstar.view(T, 1).expand(T, V).contiguous().reshape(-1)
    gap = gl_sub(vbc, l.reshape(-1))
    # Hidden output select: Ogap = O·gap (one nonzero slot/row), gap_o = Σ_i Ogap.
    # gap >= 0 and < gap_max, so the single selected term is small; the rowsum is
    # exact in int64 (bit-view sum, no field wrap).
    Ogap = gl_mul(O.reshape(-1), gap)
    gap_o = Ogap.view(torch.int64).view(T, V).sum(dim=1).contiguous().view(torch.uint64)
    return {c.A: A.reshape(-1).contiguous(), c.Al: Al, c.vstar: vstar,
            c.gap: gap, c.neg_gap: gl_neg(gap), c.Ogap: Ogap, c.gap_o: gap_o}


def max_compile(c: MaxClaim, _ch, cfg: LigeroConfig, base: int):
    ell, T, V = cfg.ELL, c.T, c.V
    L = T * V
    neg1 = (P - 1) % P
    n_rows = (L + ell - 1) // ell
    ones_v = torch.ones(V, dtype=torch.uint64, device="cuda")
    idx_v = torch.arange(V, dtype=torch.int64, device="cuda").to(torch.uint64)  # [0..V-1] for tok = Σ i·O

    quads: List[QuadFamily] = []
    # A·A=A, Al=A·l, O·O=O, Ogap=O·gap   (per-slot quads; one family each)
    for x_v, y_v, z_v, tag in [(c.A, c.A, c.A, "AA"), (c.A, c.l, c.Al, "Al"),
                                (c.O, c.O, c.O, "OO"), (c.O, c.gap, c.Ogap, "Ogap")]:
        quads.append(QuadFamily(
            name=f"Max.{tag}", x_row=x_v.row_start, y_row=y_v.row_start,
            z_row=z_v.row_start, L=L, ell=ell, a=neg1, b=0))

    row_pkts: List[Tuple[int, object]] = []
    cur = base
    # Σ_i A[t,i] = 1   (rowsum over V -> cid base+t, RHS 1)
    for ro in range(c.A.n_rows(ell)):
        row_pkts.append((c.A.row_start + ro, L2_RowSumPerSlotVector(
            base=cur, var_row_start=c.A.row_start, L=L, stride=V, coef_vec=ones_v)))
    sumA_lo = cur - base
    cur += T
    # v* = Σ_i Al[t,i]    (rowsum(Al) - v* = 0)
    for ro in range(c.Al.n_rows(ell)):
        row_pkts.append((c.Al.row_start + ro, L2_RowSumPerSlotVector(
            base=cur, var_row_start=c.Al.row_start, L=L, stride=V, coef_vec=ones_v)))
    for ro in range(c.vstar.n_rows(ell)):
        row_pkts.append((c.vstar.row_start + ro, L2_IdentityScalar(
            base=cur, var_row_start=c.vstar.row_start, L=T, coef=neg1)))
    cur += T
    # gap + l - v*(broadcast) = 0
    for ro in range(c.gap.n_rows(ell)):
        row_pkts.append((c.gap.row_start + ro, L2_IdentityScalar(
            base=cur, var_row_start=c.gap.row_start, L=L, coef=1)))
    for ro in range(c.l.n_rows(ell)):
        row_pkts.append((c.l.row_start + ro, L2_IdentityScalar(
            base=cur, var_row_start=c.l.row_start, L=L, coef=1)))
    for ro in range(c.vstar.n_rows(ell)):
        row_pkts.append((c.vstar.row_start + ro, L2_StrideOneToManyScalar(
            base=cur, var_row_start=c.vstar.row_start, L=T, stride=V, coef=neg1)))
    cur += L
    # neg_gap + gap = 0   (pins neg_gap = -gap, for -gap^2 with no constant)
    for ro in range(c.neg_gap.n_rows(ell)):
        row_pkts.append((c.neg_gap.row_start + ro, L2_IdentityScalar(
            base=cur, var_row_start=c.neg_gap.row_start, L=L, coef=1)))
    for ro in range(c.gap.n_rows(ell)):
        row_pkts.append((c.gap.row_start + ro, L2_IdentityScalar(
            base=cur, var_row_start=c.gap.row_start, L=L, coef=1)))
    cur += L
    # Σ_i O[t,i] = 1   (output one-hot -> RHS 1)
    sumO_lo = cur - base
    for ro in range(c.O.n_rows(ell)):
        row_pkts.append((c.O.row_start + ro, L2_RowSumPerSlotVector(
            base=cur, var_row_start=c.O.row_start, L=L, stride=V, coef_vec=ones_v)))
    cur += T
    # gap_o = Σ_i Ogap[t,i]    (rowsum(Ogap) - gap_o = 0)
    for ro in range(c.Ogap.n_rows(ell)):
        row_pkts.append((c.Ogap.row_start + ro, L2_RowSumPerSlotVector(
            base=cur, var_row_start=c.Ogap.row_start, L=L, stride=V, coef_vec=ones_v)))
    for ro in range(c.gap_o.n_rows(ell)):
        row_pkts.append((c.gap_o.row_start + ro, L2_IdentityScalar(
            base=cur, var_row_start=c.gap_o.row_start, L=T, coef=neg1)))
    cur += T
    # tok_t = Σ_i i·O[t,i]   (O is the one-hot OF the committed token; rowsum(i·O) - tok = 0)
    for ro in range(c.O.n_rows(ell)):
        row_pkts.append((c.O.row_start + ro, L2_RowSumPerSlotVector(
            base=cur, var_row_start=c.O.row_start, L=L, stride=V, coef_vec=idx_v)))
    for ro in range(c.tok.n_rows(ell)):
        row_pkts.append((c.tok.row_start + ro, L2_IdentityScalar(
            base=cur, var_row_start=c.tok.row_start, L=T, coef=neg1)))
    cur += T

    # gap ∈ [0, gap_max):  (α - gap)·z = 1   (copy the RangeWord pattern)
    neg_alpha = (P - c.table.alpha) % P
    quads.append(QuadFamily(
        name="Max.gap_ge0", x_row=c.gap.row_start, y_row=c.z_gap.row_start,
        z_row=c.z_gap.row_start, L=L, ell=ell, a=neg_alpha, b=neg1))

    n_added = cur - base
    # _build_b_chunk families are (rel_start, LENGTH, value) -- both RHS-1 runs
    # have length T (the ΣA=1 and ΣO=1 rowsums).
    b_chunk = _build_b_chunk(n_added, [(sumA_lo, T, 1), (sumO_lo, T, 1)])
    return row_pkts, quads, n_added, b_chunk


COMPILE_FNS[MaxClaim] = max_compile
SAMPLE_FNS[MaxClaim] = max_sample
AUX_FNS[MaxClaim] = max_aux
_cf.COMPUTE_FNS[MaxClaim] = max_compute


def to_signed(f):
    """uint64 Goldilocks field rep -> int64 signed value (logits are small, so the
    field rep is either a small positive or P-|v| near 2^64; the int64 bit-view is
    negative there and (2^64 - P) un-offsets it). Order-correct for the argmax."""
    fi = f.contiguous().view(torch.int64)
    return fi + (fi < 0).to(torch.int64) * ((1 << 32) - 1)


_BUILD = [0]


def max_gap(tape, logits, tokens, *, T, V, gap_max, force_argmax=None, O_ext=None):
    """Build the MaxClaim on committed `logits` (T,V) and committed output `tokens`
    (length T, blinded like the weights). Returns (gap, gap_o, neg_gap, vstar)
    WitnessTensors, where gap_o[t] = gap[t, tokens[t]] is the selected output gap.

    `force_argmax` (list of indices) builds a consistent witness around a chosen v*
    instead of the real max -- the soundness test uses it to confirm a non-max v* is
    rejected by gap >= 0.

    `O_ext` (WitnessTensor, (T,V)): use an EXISTING variable as the output
    one-hot O instead of committing a fresh one. MaxClaim's booleanity,
    cardinality, and tok = Σ i·O constraints apply to it unchanged. This is
    how the scored tokens are tied to the tokens the model consumed: pass the
    hidden INPUT indicator rows (shifted by one position), and one committed
    token stream drives both the forward pass and the surprisal."""
    from tape import WitnessTensor
    _BUILD[0] += 1
    pfx = f"mx{_BUILD[0]}_"
    table = tape.register_table(f"{pfx}gap", T_data=list(range(gap_max)))
    # A is ENGINE-DERIVED (argmax of signed logits at witness time — lazy
    # tapes have no logits.data at build). force_argmax (soundness tests)
    # rides the tamper hook: a wrong one-hot must REJECT via gap >= 0.
    if force_argmax is not None:
        am = torch.tensor(force_argmax, device="cuda")
        A_data = torch.zeros(T, V, dtype=torch.int64, device="cuda")
        A_data[torch.arange(T, device="cuda"), am] = 1
        TEST_TAMPER["A"] = A_data.reshape(-1).to(torch.uint64)
    A = tape._alloc(f"{pfx}A", T * V)
    # Output committed AS TOKENS (length T), blinded like weights; O is its one-hot.
    tok_t = torch.tensor(list(tokens), dtype=torch.int64, device="cuda")
    tok = tape.commit(f"{pfx}tok", tok_t.to(torch.uint64), (T,))
    if O_ext is not None:
        assert O_ext.var.length == T * V, \
            f"O_ext length {O_ext.var.length} != T*V = {T * V}"
        O = O_ext
    else:
        O_data = torch.zeros(T, V, dtype=torch.int64, device="cuda")
        O_data[torch.arange(T, device="cuda"), tok_t] = 1
        O = tape.commit(f"{pfx}O", O_data.reshape(-1).to(torch.uint64), (T, V))
    Al   = tape._alloc(f"{pfx}Al", T * V)
    vstar = tape._alloc(f"{pfx}vstar", T)
    gap  = tape._alloc(f"{pfx}gap_v", T * V)
    neg_gap = tape._alloc(f"{pfx}neg_gap", T * V)
    Ogap = tape._alloc(f"{pfx}Ogap", T * V)
    gap_o = tape._alloc(f"{pfx}gap_o", T)
    z_gap = Variable(f"{pfx}z_gap", length=T * V, phase=2)
    table.z_vars.append(z_gap)
    claim = MaxClaim(l=logits.var, A=A, Al=Al, vstar=vstar, gap=gap,
                     neg_gap=neg_gap, z_gap=z_gap, tok=tok.var, O=O.var,
                     Ogap=Ogap, gap_o=gap_o, table=table, T=T, V=V)

    def side_effects(values):
        lookup_multiplicities_into(values[gap], table.T, tape.inputs[table.mult_var])

    outs = tape._process_claim(claim, [logits.var, O.var], side_effects)
    tape.claims.append(claim)
    gap_wt    = WitnessTensor(outs[gap] if outs else None, gap, (T, V), tape)
    gap_o_wt  = WitnessTensor(outs[gap_o] if outs else None, gap_o, (T, 1), tape)
    neg_wt    = WitnessTensor(outs[neg_gap] if outs else None, neg_gap, (T, V), tape)
    vstar_wt  = WitnessTensor(outs[vstar] if outs else None, vstar, (T, 1), tape)
    return gap_wt, gap_o_wt, neg_wt, vstar_wt
