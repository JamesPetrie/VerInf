"""InfoFinalizeClaim: the new glue for the unexplained-information upper bound.

Given (per row t):
  e[t,i]   = EXP[gap[t,i]]   (the exp-kernel lookups, from a paired_tlookup)
  pw[t]    = POW[b[t]]       (the log-pin POW lookup, from a paired_tlookup)
  gap_o2[t]= gap_o[t]^2      (the output token's squared gap, from a hadamard)
  b[t]                       (the prover-committed log-pin value)
it proves, with EVERY rounding pushed so U is an upper bound:

  a[t]        = Sum_i e[t,i]                      (rowsum; a in [s_y, V*s_y])
  a[t] <= pw[t]                                   (log-pin: a + d = pw, d >= 0)
  z_o[t]      = ceil(gap_o2[t] / k),  k = s_c/s_b (numerator: k*z_o = gap_o2 + rem)
  surprisal[t]= z_o[t] + b[t]                     (nats * s_b)

then U = (Sum_t surprisal) / (s_b * ln2) bits is computed (rounded up) outside.

Soundness (a + d = pw with d >= 0 forces a <= pw, so b >= s_b*ln(a/s_y)):
  * d >= 0     via word-decomposition d = Sum_j 2^(wb*j) dw_j, each dw_j in [0,2^wb)
  * rem in [0,k) via a range LogUp (k a power of two)
Both directions are LogUp range checks against committed range tables; a wrapped
(negative) d or rem has no nonneg decomposition and is rejected.

The exp lookups (e=EXP[gap]) and pow lookup (pw=POW[b]) are done by the tested
`tape.paired_tlookup`; argmax/gap/select by the tested MaxClaim; the square by
`tape.hadamard`. This claim only carries the irreducible new arithmetic.
"""
from dataclasses import dataclass, field
from typing import List, Tuple

import torch

import core
from core import P, Variable, LigeroConfig, SAMPLE_FNS, AUX_FNS
from claims import (COMPILE_FNS, QuadraticConstraint, QuadFamily, L2_IdentityScalar,
                    L2_RowSumPerSlotVector, _build_b_chunk)
import compute_fns as _cf
from cuda_primitives import gl_mul, gl_sub, gl_add, gl_inv_batched, lookup_multiplicities_into


@dataclass
class InfoFinalizeClaim:
    e: Variable            # EXP lookups (T*V), input
    pw: Variable           # POW[b] (T), DERIVED (bound to ui_pow by a later tlookup)
    gap_o2: Variable       # gap_o^2 (T), input
    b: Variable            # log-pin value (T), DERIVED (searchsorted at witness time)
    a: Variable            # Sum_i e (T), derived
    d: Variable            # pw - a >= 0 (T), derived
    dw: List[Variable]     # word-decomposition of d (N each T), derived
    z_o: Variable          # ceil(gap_o2 / k) (T), derived
    rem: Variable          # k*z_o - gap_o2 in [0,k) (T), derived
    surprisal: Variable    # z_o + b (T), derived  (nats * s_b)
    z_dw: List[Variable]   # phase-2 range-LogUp aux for each dw (N each T)
    z_rem: Variable        # phase-2 range-LogUp aux for rem (T)
    range_wd: object       # range Table [0, 2^wb)  for the dw words
    range_k: object        # range Table [0, k)      for rem
    T: int
    V: int
    k: int                 # = s_c / s_b  (power of two)
    wb: int                # word bits for the d decomposition
    s_y: int               # exp-table output scale (for the POW recompute)
    s_b: int               # log-pin scale
    K: int                 # pow table size

    @property
    def length(self):
        return self.T


def info_sample(c: InfoFinalizeClaim, ci, s_op):
    return None             # alphas owned by the range tables' TableSettlement


def info_compute(c: InfoFinalizeClaim, live):
    T, V, k = c.T, c.V, c.k
    e = live[c.e].contiguous().view(torch.int64).view(T, V)          # in [1, s_y]
    gap_o2 = live[c.gap_o2].contiguous().view(-1)
    # a = Sum_i e[t,i]  (values small, exact int64 sum, no field wrap)
    a = e.sum(dim=1).contiguous().view(torch.uint64)
    # b, pw DERIVED here (lazy tapes have no build-time logits): b = smallest
    # index with POW[b] >= a; the later paired_tlookup binds pw to ui_pow.
    from unexplained_info import pow_table_values
    import numpy as _np
    POW_i64 = torch.tensor(pow_table_values(c.K, c.s_y, c.s_b).astype(_np.int64),
                            device="cuda")
    b = torch.searchsorted(POW_i64, a.view(torch.int64).contiguous()
                            ).clamp(max=c.K - 1)
    pw = POW_i64.index_select(0, b).contiguous().view(torch.uint64)
    b = b.contiguous().view(torch.uint64)
    # d = pw - a  (honest: a <= pw, so d in [0, V*s_y))
    d = gl_sub(pw, a)
    # word-decompose d into N words of wb bits each
    di = d.contiguous().view(torch.int64)
    mask = (1 << c.wb) - 1
    dw = {c.dw[j]: ((di >> (c.wb * j)) & mask).contiguous().view(torch.uint64)
          for j in range(len(c.dw))}
    # z_o = ceil(gap_o2 / k);  rem = k*z_o - gap_o2 in [0,k)
    g2 = gap_o2.view(torch.int64)
    z_o = ((g2 + (k - 1)) // k).contiguous().view(torch.uint64)
    z_o_i = z_o.view(torch.int64)
    rem = (k * z_o_i - g2).contiguous().view(torch.uint64)
    # surprisal = z_o + b   (nats * s_b)
    surprisal = gl_add(z_o, b)
    out = {c.b: b, c.pw: pw, c.a: a, c.d: d, c.z_o: z_o, c.rem: rem,
           c.surprisal: surprisal}
    out.update(dw)
    return out


def info_aux(c: InfoFinalizeClaim, witness, _ch):
    """Phase-2 range-LogUp inverses: z = 1/(alpha - x) for each range-checked x."""
    def inv_against(x_var, table):
        x = witness[x_var]
        x = x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.uint64, device="cuda")
        x = x.contiguous().view(-1)
        alpha = torch.full_like(x, table.alpha)
        return gl_inv_batched(gl_sub(alpha, x))
    out = {c.z_dw[j]: inv_against(c.dw[j], c.range_wd) for j in range(len(c.dw))}
    out[c.z_rem] = inv_against(c.rem, c.range_k)
    return out


def info_compile(c: InfoFinalizeClaim, _ch, cfg: LigeroConfig, base: int):
    ell, T, V, k = cfg.ELL, c.T, c.V, c.k
    L = T * V
    neg1 = (P - 1) % P
    ones_v = torch.ones(V, dtype=torch.uint64, device="cuda")
    row_pkts: List[Tuple[int, object]] = []
    cur = base

    # a = Sum_i e[t,i]    (rowsum(e) - a = 0)
    for ro in range(_nr(c.e, ell)):
        row_pkts.append((c.e.row_start + ro, L2_RowSumPerSlotVector(
            base=cur, var_row_start=c.e.row_start, L=L, stride=V, coef_vec=ones_v)))
    for ro in range(_nr(c.a, ell)):
        row_pkts.append((c.a.row_start + ro, L2_IdentityScalar(
            base=cur, var_row_start=c.a.row_start, L=T, coef=neg1)))
    cur += T

    # a + d - pw = 0     (d = pw - a >= 0)
    for var, coef in [(c.a, 1), (c.d, 1), (c.pw, neg1)]:
        for ro in range(_nr(var, ell)):
            row_pkts.append((var.row_start + ro, L2_IdentityScalar(
                base=cur, var_row_start=var.row_start, L=T, coef=coef)))
    cur += T

    # d - Sum_j 2^(wb*j) dw_j = 0
    for ro in range(_nr(c.d, ell)):
        row_pkts.append((c.d.row_start + ro, L2_IdentityScalar(
            base=cur, var_row_start=c.d.row_start, L=T, coef=1)))
    for j, dwj in enumerate(c.dw):
        coef = (P - ((1 << (c.wb * j)) % P)) % P                       # -2^(wb*j)
        for ro in range(_nr(dwj, ell)):
            row_pkts.append((dwj.row_start + ro, L2_IdentityScalar(
                base=cur, var_row_start=dwj.row_start, L=T, coef=coef)))
    cur += T

    # k*z_o - gap_o2 - rem = 0
    for var, coef in [(c.z_o, k % P), (c.gap_o2, neg1), (c.rem, neg1)]:
        for ro in range(_nr(var, ell)):
            row_pkts.append((var.row_start + ro, L2_IdentityScalar(
                base=cur, var_row_start=var.row_start, L=T, coef=coef)))
    cur += T

    # surprisal - z_o - b = 0
    for var, coef in [(c.surprisal, 1), (c.z_o, neg1), (c.b, neg1)]:
        for ro in range(_nr(var, ell)):
            row_pkts.append((var.row_start + ro, L2_IdentityScalar(
                base=cur, var_row_start=var.row_start, L=T, coef=coef)))
    cur += T

    # range LogUp quads: (alpha - x)*z = 1   for each dw_j and rem
    n_rows_T = (T + ell - 1) // ell
    quads: List[QuadFamily] = []
    neg_alpha_wd = (P - c.range_wd.alpha) % P
    for j, (dwj, zj) in enumerate(zip(c.dw, c.z_dw)):
        quads.append(QuadFamily(
            name=f"Info.dw{j}", x_row=dwj.row_start, y_row=zj.row_start,
            z_row=zj.row_start, L=T, ell=ell, a=neg_alpha_wd, b=neg1))
    neg_alpha_k = (P - c.range_k.alpha) % P
    quads.append(QuadFamily(
        name="Info.rem", x_row=c.rem.row_start, y_row=c.z_rem.row_start,
        z_row=c.z_rem.row_start, L=T, ell=ell, a=neg_alpha_k, b=neg1))

    n_added = cur - base
    return row_pkts, quads, n_added, None     # b = 0 throughout


def _nr(var, ell):
    return var.n_rows(ell)


COMPILE_FNS[InfoFinalizeClaim] = info_compile
SAMPLE_FNS[InfoFinalizeClaim] = info_sample
AUX_FNS[InfoFinalizeClaim] = info_aux
_cf.COMPUTE_FNS[InfoFinalizeClaim] = info_compute


_BUILD = [0]


def info_finalize(tape, e, gap_o2, *, T, V, k, d_max, s_y, s_b, K):
    """Build the InfoFinalizeClaim. Returns the surprisal WitnessTensor (T,1),
    each entry = (z_o + b) = surprisal_t * s_b (nats * s_b)."""
    from tape import WitnessTensor
    _BUILD[0] += 1
    pfx = f"ui{_BUILD[0]}_"
    wb = 12
    n_words = max(1, (d_max.bit_length() + wb - 1) // wb)
    range_wd = tape.register_table(f"{pfx}wd", T_data=list(range(1 << wb)))
    range_k = tape.register_table(f"{pfx}rem", T_data=list(range(k)))

    b = tape._alloc(f"{pfx}b", T)
    pw = tape._alloc(f"{pfx}pw", T)
    a = tape._alloc(f"{pfx}a", T)
    d = tape._alloc(f"{pfx}d", T)
    dw = [tape._alloc(f"{pfx}dw{j}", T) for j in range(n_words)]
    z_o = tape._alloc(f"{pfx}z_o", T)
    rem = tape._alloc(f"{pfx}rem_v", T)
    surprisal = tape._alloc(f"{pfx}surprisal", T)
    z_dw = [Variable(f"{pfx}z_dw{j}", length=T, phase=2) for j in range(n_words)]
    z_rem = Variable(f"{pfx}z_rem", length=T, phase=2)
    for zj in z_dw:
        range_wd.z_vars.append(zj)
    range_k.z_vars.append(z_rem)

    claim = InfoFinalizeClaim(
        e=e.var, pw=pw, gap_o2=gap_o2.var, b=b, a=a, d=d, dw=dw,
        z_o=z_o, rem=rem, surprisal=surprisal, z_dw=z_dw, z_rem=z_rem,
        range_wd=range_wd, range_k=range_k, T=T, V=V, k=k, wb=wb,
        s_y=s_y, s_b=s_b, K=K)

    def side_effects(values):
        for dwj in dw:
            lookup_multiplicities_into(values[dwj], range_wd.T,
                                        tape.inputs[range_wd.mult_var])
        lookup_multiplicities_into(values[rem], range_k.T,
                                    tape.inputs[range_k.mult_var])

    outs = tape._process_claim(claim, [e.var, gap_o2.var], side_effects)
    tape.claims.append(claim)
    return (WitnessTensor(outs[surprisal] if outs else None, surprisal, (T, 1), tape),
            WitnessTensor(outs[b] if outs else None, b, (T, 1), tape),
            WitnessTensor(outs[pw] if outs else None, pw, (T, 1), tape))
