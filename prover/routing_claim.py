"""RoutingClaim + FreivaldsCombineClaim: verified top-1 MoE routing (Llama-4 style).

RoutingClaim proves that a committed one-hot mask `m` (T, E) selects the
argmax expert of committed router logits `r` (T, E), with ties broken by
lowest expert index, and recovers the chosen logit `r_chosen` for the
downstream sigmoid routing weight. It is the top-1 specialization of the
threshold gadget in design-feasibility.md §3.6 / appendix-moe-routing.md §B.3,
structured as a clone of the audited MaxClaim pattern: for topk = 1 the
threshold τ degenerates and "m is the top-1" is exactly "m is the one-hot
argmax", checked by gap ≥ 0 on tiebroken logits.

Tiebroken logits (appendix §B.3 (1)):  rt[t,e] = 2^L·r[t,e] + (E−1−e),
L = ceil(log2 E). All rt in a row are distinct, so the argmax — and hence m —
is unique, matching torch.topk / llama.cpp's lowest-index tie behavior.

Constraint families (emission order is load-bearing for the Rust verifier):
  F1  rt − 2^L·r = (E−1−e)            linear, RHS bonus pattern   (T·E)
  F2  Σ_e m[t,e] = 1                  cardinality, RHS 1          (T)
  F3  Σ_e mrt[t,e] − rstar[t] = 0     chosen tiebroken logit      (T)
  F4  gap + rt − rstar(bcast) = 0     dominance gap               (T·E)
  F5  2^L·r_chosen + Σ_e (E−1−e)·m[t,e] − rstar = 0               (T)
  Q1  m·m = m                         booleanity                  (T·E)
  Q2  m·rt = mrt                                                  (T·E)

gap ∈ [0, 2^{B+L}) is NOT checked here — the route_top1 builder composes the
existing tape.word_extract (WordExtractionClaim + RangeWordClaim) on the gap
output. Soundness precondition (documented in CLAIM_SPECS): `r` must already
be range-bounded to ±2^{B−1} by its producing matmul's output_width rescale;
the word width must satisfy N·B_word ≥ B+L and 2^{N·B_word+1} ≪ P.

Why a wrong mask cannot pass: booleanity + cardinality force a one-hot; if it
selects e' ≠ argmax(rt), then gap[argmax] = rt[e'] − rt[argmax] < 0, whose
field representative is ≈ P and cannot decompose into N·B_word ≪ 64 bits, so
the word-extraction linear constraint (or the word range LogUp) rejects.

FreivaldsCombineClaim proves y[t,:] = Σ_e m[t,e] · X_e[t,:] — the masked expert
combine (appendix §B.4 seam, all-experts-commit form) — by a random projection
ρ ∈ F^F instead of committing the E·T·F per-expert products: it collapses each
expert stream to a per-token scalar and checks the projected identity per token
(witness ~4·E·T, not ~2·E·T·F; soundness error ~1/|F| per token, sound under the
commit-before-challenge ordering). Its constraints (C1–C5 + the ms quadratic)
are documented at the claim below.
"""
import math
from dataclasses import dataclass
from typing import List, Tuple

import torch

import core
from core import P, Variable, LigeroConfig, SAMPLE_FNS, AUX_FNS
from claims import (COMPILE_FNS, QuadraticConstraint, QuadFamily, L2_IdentityScalar,
                    L2_RowSumPerSlotVector, L2_StrideOneToManyScalar,
                    _build_b_chunk)
from packets import L2_TransposeO2MScalar
import compute_fns as _cf
from cuda_primitives import gl_mul, gl_add, gl_sub
from max_claim import to_signed


# ===========================================================================
# RoutingClaim
# ===========================================================================

@dataclass
class RoutingClaim:
    r: Variable            # router logits (T*E), committed by the caller
    m: Variable            # one-hot top-1 mask (T*E), committed
    rt: Variable           # tiebroken logits 2^L·r + (E−1−e) (T*E), derived
    mrt: Variable          # m·rt (T*E), derived
    rstar: Variable        # chosen tiebroken logit per token (T), derived
    gap: Variable          # rstar − rt ≥ 0 (T*E), derived (range via word_extract)
    r_chosen: Variable     # chosen raw logit per token (T), derived (sigmoid input)
    T: int
    E: int
    L_bits: int            # ceil(log2 E), public

    @property
    def length(self):
        return self.T * self.E


def _bonus_vec(E: int) -> torch.Tensor:
    """Tiebreaker bonus (E−1−e) for e in [0, E) — uint64 cuda."""
    return (torch.arange(E - 1, -1, -1, dtype=torch.int64, device="cuda")
            .to(torch.uint64))


def routing_sample(c: RoutingClaim, ci, s_op):
    return None                      # no per-claim challenges


def routing_aux(c: RoutingClaim, witness, _ch):
    return {}                        # range LogUp aux lives in the composed word_extract


def routing_compute(c: RoutingClaim, live):
    """Derives EVERYTHING from r, including the mask: m = one-hot argmax of the
    tiebroken logits (signed comparison; first-max index matches torch.topk /
    llama.cpp ties). The engine computes r before this claim runs, so no
    build-time hint is needed. Downstream values derive from the (possibly
    TEST_TAMPER-overridden) mask so each negative test fires exactly its own
    constraint family."""
    T, E, Lb = c.T, c.E, c.L_bits
    r = live[c.r].contiguous().view(-1)
    two_l = torch.full((T * E,), 1 << Lb, dtype=torch.uint64, device="cuda")
    bonus = _bonus_vec(E).repeat(T)
    rt = gl_add(gl_mul(r, two_l), bonus)
    am = to_signed(rt).view(T, E).argmax(dim=1)
    ar = torch.arange(T, device="cuda")
    m = torch.zeros(T, E, dtype=torch.int64, device="cuda")
    m[ar, am] = 1
    m = m.to(torch.uint64)
    if "m" in TEST_TAMPER:
        m = TEST_TAMPER["m"].to("cuda").view(T, E)
    mrt = gl_mul(m.reshape(-1), rt)
    idx = m.view(torch.int64).argmax(dim=1)
    rstar = rt.view(torch.int64).view(T, E)[ar, idx].contiguous().view(torch.uint64)
    rs_bc = rstar.view(T, 1).expand(T, E).contiguous().reshape(-1)
    gap = gl_sub(rs_bc, rt)
    r_chosen = r.view(torch.int64).view(T, E)[ar, idx].contiguous().view(torch.uint64)
    outs = {c.m: m.reshape(-1), c.rt: rt, c.mrt: mrt, c.rstar: rstar,
            c.gap: gap, c.r_chosen: r_chosen}
    return _apply_tamper(outs, [("rt", c.rt), ("mrt", c.mrt), ("rstar", c.rstar),
                                 ("gap", c.gap), ("r_chosen", c.r_chosen)])


def routing_compile(c: RoutingClaim, _ch, cfg: LigeroConfig, base: int):
    ell, T, E = cfg.ELL, c.T, c.E
    L = T * E
    neg1 = (P - 1) % P
    two_l = (1 << c.L_bits) % P
    neg_two_l = (P - two_l) % P
    ones_e = torch.ones(E, dtype=torch.uint64, device="cuda")
    bonus_e = _bonus_vec(E)
    n_rows = (L + ell - 1) // ell

    row_pkts: List[Tuple[int, object]] = []
    cur = base

    # F1: rt − 2^L·r = (E−1−e)
    f1_lo = cur - base
    for ro in range(c.rt.n_rows(ell)):
        row_pkts.append((c.rt.row_start + ro, L2_IdentityScalar(
            base=cur, var_row_start=c.rt.row_start, L=L, coef=1)))
    for ro in range(c.r.n_rows(ell)):
        row_pkts.append((c.r.row_start + ro, L2_IdentityScalar(
            base=cur, var_row_start=c.r.row_start, L=L, coef=neg_two_l)))
    cur += L
    # F2: Σ_e m[t,e] = 1
    f2_lo = cur - base
    for ro in range(c.m.n_rows(ell)):
        row_pkts.append((c.m.row_start + ro, L2_RowSumPerSlotVector(
            base=cur, var_row_start=c.m.row_start, L=L, stride=E, coef_vec=ones_e)))
    cur += T
    # F3: Σ_e mrt[t,e] − rstar[t] = 0
    for ro in range(c.mrt.n_rows(ell)):
        row_pkts.append((c.mrt.row_start + ro, L2_RowSumPerSlotVector(
            base=cur, var_row_start=c.mrt.row_start, L=L, stride=E, coef_vec=ones_e)))
    for ro in range(c.rstar.n_rows(ell)):
        row_pkts.append((c.rstar.row_start + ro, L2_IdentityScalar(
            base=cur, var_row_start=c.rstar.row_start, L=T, coef=neg1)))
    cur += T
    # F4: gap + rt − rstar(broadcast over E) = 0
    for ro in range(c.gap.n_rows(ell)):
        row_pkts.append((c.gap.row_start + ro, L2_IdentityScalar(
            base=cur, var_row_start=c.gap.row_start, L=L, coef=1)))
    for ro in range(c.rt.n_rows(ell)):
        row_pkts.append((c.rt.row_start + ro, L2_IdentityScalar(
            base=cur, var_row_start=c.rt.row_start, L=L, coef=1)))
    for ro in range(c.rstar.n_rows(ell)):
        row_pkts.append((c.rstar.row_start + ro, L2_StrideOneToManyScalar(
            base=cur, var_row_start=c.rstar.row_start, L=T, stride=E, coef=neg1)))
    cur += L
    # F5: 2^L·r_chosen + Σ_e (E−1−e)·m[t,e] − rstar = 0
    for ro in range(c.r_chosen.n_rows(ell)):
        row_pkts.append((c.r_chosen.row_start + ro, L2_IdentityScalar(
            base=cur, var_row_start=c.r_chosen.row_start, L=T, coef=two_l)))
    for ro in range(c.m.n_rows(ell)):
        row_pkts.append((c.m.row_start + ro, L2_RowSumPerSlotVector(
            base=cur, var_row_start=c.m.row_start, L=L, stride=E, coef_vec=bonus_e)))
    for ro in range(c.rstar.n_rows(ell)):
        row_pkts.append((c.rstar.row_start + ro, L2_IdentityScalar(
            base=cur, var_row_start=c.rstar.row_start, L=T, coef=neg1)))
    cur += T

    quads: List[QuadFamily] = []
    # Q1: m·m = m   Q2: m·rt = mrt   (one family each)
    for x_v, y_v, z_v, tag in [(c.m, c.m, c.m, "mm"),
                                (c.m, c.rt, c.mrt, "mrt")]:
        quads.append(QuadFamily(
            name=f"Route.{tag}", x_row=x_v.row_start, y_row=y_v.row_start,
            z_row=z_v.row_start, L=L, ell=ell, a=neg1, b=0))

    n_added = cur - base
    # RHS: F1 carries the cyclic bonus pattern (E−1−e); F2 is all-ones.
    fam = [(f1_lo + f, 1, int(E - 1 - (f % E))) for f in range(L) if (f % E) != E - 1]
    fam.append((f2_lo, T, 1))
    b_chunk = _build_b_chunk(n_added, fam)
    return row_pkts, quads, n_added, b_chunk


COMPILE_FNS[RoutingClaim] = routing_compile
SAMPLE_FNS[RoutingClaim] = routing_sample
AUX_FNS[RoutingClaim] = routing_aux
_cf.COMPUTE_FNS[RoutingClaim] = routing_compute


_BUILD = [0]

# Test-only hook (audit A1 finding 3): {field_name: uint64 tensor} overrides
# applied to DERIVED witness values inside the compute fns, so each binding
# constraint family has a negative test that commits an inconsistent value —
# the derive-from-committed-inputs design would otherwise mask a missing
# constraint. List fields use indexed keys ("m_rep0", "prods1"). Tests must
# clear this dict (try/finally).
TEST_TAMPER = {}


def _apply_tamper(outs, named_vars):
    for name, var in named_vars:
        if name in TEST_TAMPER:
            outs[var] = TEST_TAMPER[name].to("cuda")
    return outs


def route_top1(tape, r, *, T, E, B_logit, word_bits=11):
    """Build the RoutingClaim on router logits `r` (T, E), plus the gap range
    check via word_extract. Returns (m, r_chosen, gap) WitnessTensors.

    The mask is DERIVED inside the engine (argmax of the tiebroken logits) —
    no build-time hint, works identically for committed or lazily-derived r.
    Negative tests override derived values via routing_claim.TEST_TAMPER
    (key "m" for the mask itself).

    `B_logit` bounds |r| (enforced UPSTREAM by the producing matmul's
    output_width rescale — a documented soundness precondition)."""
    from tape import WitnessTensor
    _BUILD[0] += 1
    pfx = f"rt{_BUILD[0]}_"
    L_bits = max(1, math.ceil(math.log2(E)))
    width = B_logit + L_bits
    n_words = max(1, math.ceil(width / word_bits))
    # SOUNDNESS GUARD (audit A1 finding 1): the wrong-mask argument requires
    # that the words CANNOT represent the field rep of a negative gap.
    assert n_words * word_bits >= width, "word decomposition must cover the gap width"
    assert (1 << (n_words * word_bits)) <= P - (1 << width), \
        (f"route_top1: unsound parameterization — {n_words}x{word_bits}-bit words "
         f"can reach the negative-gap field range (width={width}). "
         f"Reduce B_logit or word_bits so that 2^(n_words*word_bits) <= P - 2^width.")

    m = tape._alloc(f"{pfx}m", T * E)
    rt = tape._alloc(f"{pfx}rt", T * E)
    mrt = tape._alloc(f"{pfx}mrt", T * E)
    rstar = tape._alloc(f"{pfx}rstar", T)
    gap = tape._alloc(f"{pfx}gap", T * E)
    r_chosen = tape._alloc(f"{pfx}r_chosen", T)

    claim = RoutingClaim(r=r.var, m=m, rt=rt, mrt=mrt, rstar=rstar,
                         gap=gap, r_chosen=r_chosen, T=T, E=E, L_bits=L_bits)
    outs = tape._process_claim(claim, [r.var])
    tape.claims.append(claim)

    m_wt = WitnessTensor(outs[m] if outs else None, m, (T, E), tape)
    gap_wt = WitnessTensor(outs[gap] if outs else None, gap, (T, E), tape)
    rch_wt = WitnessTensor(outs[r_chosen] if outs else None, r_chosen, (T, 1), tape)

    # gap ∈ [0, 2^width): word-decompose and range-check each word against a
    # shared range table (the §B.3 (8) range check, composed from existing claims).
    table = tape.register_table(f"{pfx}rng", T_data=list(range(1 << word_bits)))
    tape.word_extract(gap_wt, table, B=word_bits, N=n_words)
    return m_wt, rch_wt, gap_wt


# ===========================================================================
# FreivaldsCombineClaim — the §B.4 seam: y[t,:] = Σ_e m[t,e]·X_e[t,:] proven
# by random projection instead of committed per-expert products.
#
# A naive masked sum that commits the E·T·F replicated-mask and product slots is
# fine at few tokens but catastrophic at T=1000 × E=128 (~10^12 slots). Here the verifier
# challenge ρ ∈ F^F (derived from s_op per claim index, like Freivalds matmul)
# collapses each expert stream to a per-token scalar:
#
#   s[e,t]  = Σ_j X_e[t,j]·ρ[j]          (linear, phase 2)         C1  [E·T]
#   m_em    = mᵀ (expert-major copy)      (TransposeO2M pin)        C2  [T·E]
#   ms_em   = m_em ⊙ s_em                 (quadratic)               Q1
#   ms_tm   = ms_emᵀ (token-major copy)   (TransposeO2M pin)        C3  [T·E]
#   yr[t]   = Σ_j y[t,j]·ρ[j]            (linear, phase 2)         C4  [T]
#   Σ_e ms_tm[t,·] − yr[t] = 0           (the seam)                C5  [T]
#
# If y[t,:] ≠ Σ_e m[t,e]·X_e[t,:], the two projections differ for random ρ
# except with probability 1/|F| per token (m and y are committed in phase 1,
# ρ is sampled after — same ordering argument as the matmul Freivalds).
# Witness cost: ~4·T·E + T slots instead of ~2·E·T·F.
# ===========================================================================

@dataclass
class FreivaldsCombineClaim:
    m: Variable                 # one-hot mask (T*E), token-major (phase 1)
    xs: List[Variable]          # E per-expert streams, each (T*F) (phase 1)
    y: Variable                 # combined output (T*F); derived unless y_committed
    m_em: Variable              # m transposed to expert-major (E*T), derived
    s_em: Variable              # s[e,t] = X_e[t,:]·ρ (E*T), phase 2
    ms_em: Variable             # m_em ⊙ s_em (E*T), phase 2
    ms_tm: Variable             # ms transposed token-major (T*E), phase 2
    yr: Variable                # y[t,:]·ρ (T), phase 2
    y_committed: bool
    T: int
    E: int
    F: int


def fcombine_sample(c: FreivaldsCombineClaim, ci, s_op):
    import protocol
    return protocol.op_vec(s_op, ci, "rho", c.F)


def fcombine_compute(c: FreivaldsCombineClaim, live):
    """Atomic path = the fold path run all-at-once (single source of truth;
    the NO_FOLD toggle changes only WHEN absorbs happen, never the math)."""
    st = _fc_init(c, live)
    for v in c.xs:
        t = live[v]
        _fc_absorb(c, st, v, t() if callable(t) else t)
    return _fc_finalize(c, st, live)


def fcombine_aux(c: FreivaldsCombineClaim, witness, ch):
    st = {"_eidx": {v: e for e, v in enumerate(c.xs)}, "s_rows": {}}
    for v in c.xs:
        t = witness[v]
        t = (t if isinstance(t, torch.Tensor)
             else torch.tensor(t, dtype=torch.uint64, device="cuda"))
        _fc_aux_absorb(c, st, v, t, ch)
    return _fc_aux_finalize(c, st, witness, ch)


def fcombine_compile(c: FreivaldsCombineClaim, ch, cfg: LigeroConfig, base: int):
    ell, T, E, F = cfg.ELL, c.T, c.E, c.F
    neg1 = (P - 1) % P
    neg_rho = torch.tensor([(P - int(r) % P) % P for r in ch],
                            dtype=torch.uint64, device="cuda")
    ones_e = torch.ones(E, dtype=torch.uint64, device="cuda")
    row_pkts: List[Tuple[int, object]] = []
    cur = base

    # C1: s_em[e·T+t] − Σ_j ρ[j]·X_e[t,j] = 0   (per-expert cid blocks)
    for ro in range(c.s_em.n_rows(ell)):
        row_pkts.append((c.s_em.row_start + ro, L2_IdentityScalar(
            base=cur, var_row_start=c.s_em.row_start, L=E * T, coef=1)))
    for e in range(E):
        for ro in range(c.xs[e].n_rows(ell)):
            row_pkts.append((c.xs[e].row_start + ro, L2_RowSumPerSlotVector(
                base=cur + e * T, var_row_start=c.xs[e].row_start,
                L=T * F, stride=F, coef_vec=neg_rho)))
    cur += E * T
    # C2: m_em[e·T+t] − m[t·E+e] = 0
    for ro in range(c.m_em.n_rows(ell)):
        row_pkts.append((c.m_em.row_start + ro, L2_IdentityScalar(
            base=cur, var_row_start=c.m_em.row_start, L=E * T, coef=1)))
    for ro in range(c.m.n_rows(ell)):
        row_pkts.append((c.m.row_start + ro, L2_TransposeO2MScalar(
            base=cur, var_row_start=c.m.row_start, L=T * E,
            rows=T, cols=E, fan=1, coef=neg1)))
    cur += T * E
    # C3: ms_tm[t·E+e] − ms_em[e·T+t] = 0
    for ro in range(c.ms_tm.n_rows(ell)):
        row_pkts.append((c.ms_tm.row_start + ro, L2_IdentityScalar(
            base=cur, var_row_start=c.ms_tm.row_start, L=T * E, coef=1)))
    for ro in range(c.ms_em.n_rows(ell)):
        row_pkts.append((c.ms_em.row_start + ro, L2_TransposeO2MScalar(
            base=cur, var_row_start=c.ms_em.row_start, L=E * T,
            rows=E, cols=T, fan=1, coef=neg1)))
    cur += T * E
    # C4: yr[t] − Σ_j ρ[j]·y[t,j] = 0
    for ro in range(c.yr.n_rows(ell)):
        row_pkts.append((c.yr.row_start + ro, L2_IdentityScalar(
            base=cur, var_row_start=c.yr.row_start, L=T, coef=1)))
    for ro in range(c.y.n_rows(ell)):
        row_pkts.append((c.y.row_start + ro, L2_RowSumPerSlotVector(
            base=cur, var_row_start=c.y.row_start, L=T * F, stride=F,
            coef_vec=neg_rho)))
    cur += T
    # C5: Σ_e ms_tm[t,·] − yr[t] = 0   (the seam)
    for ro in range(c.ms_tm.n_rows(ell)):
        row_pkts.append((c.ms_tm.row_start + ro, L2_RowSumPerSlotVector(
            base=cur, var_row_start=c.ms_tm.row_start, L=T * E, stride=E,
            coef_vec=ones_e)))
    for ro in range(c.yr.n_rows(ell)):
        row_pkts.append((c.yr.row_start + ro, L2_IdentityScalar(
            base=cur, var_row_start=c.yr.row_start, L=T, coef=neg1)))
    cur += T

    quads: List[QuadFamily] = [QuadFamily(
        name="FComb.ms", x_row=c.m_em.row_start, y_row=c.s_em.row_start,
        z_row=c.ms_em.row_start, L=E * T, ell=ell, a=neg1, b=0)]

    return row_pkts, quads, cur - base, None


COMPILE_FNS[FreivaldsCombineClaim] = fcombine_compile
SAMPLE_FNS[FreivaldsCombineClaim] = fcombine_sample
AUX_FNS[FreivaldsCombineClaim] = fcombine_aux
_cf.COMPUTE_FNS[FreivaldsCombineClaim] = fcombine_compute


def freivalds_combine(tape, m, xs, *, T, E, F, force_y=None):
    """Build y[t,:] = Σ_e m[t,e]·xs[e][t,:] via the Freivalds-projected seam
    (§B.4) — the combine: a random projection ρ∈F^F replaces the E·T·F committed
    products, so witness is ~4·E·T not ~2·E·T·F. `force_y` (flat T*F int list,
    test hook) commits a wrong y; the C4/C5 projection seam must then reject."""
    from tape import WitnessTensor
    _BUILD[0] += 1
    pfx = f"fc{_BUILD[0]}_"
    assert m.var.length == T * E, f"mask length {m.var.length} != T*E"
    assert len(xs) == E, f"{len(xs)} expert streams != E"
    for x in xs:
        assert x.var.length == T * F, f"{x.var.name} length {x.var.length} != T*F"
    m_em = tape._alloc(f"{pfx}m_em", E * T)
    if force_y is not None:
        y_data = torch.tensor(list(force_y), dtype=torch.int64,
                              device="cuda").to(torch.uint64)
        y_wt = tape.commit(f"{pfx}y", y_data, (T, F))
        y_var, y_committed = y_wt.var, True
    else:
        y_var, y_committed = tape._alloc(f"{pfx}y", T * F), False
    s_em = Variable(f"{pfx}s_em", length=E * T, phase=2)
    ms_em = Variable(f"{pfx}ms_em", length=E * T, phase=2)
    ms_tm = Variable(f"{pfx}ms_tm", length=T * E, phase=2)
    yr = Variable(f"{pfx}yr", length=T, phase=2)
    claim = FreivaldsCombineClaim(m=m.var, xs=[x.var for x in xs], y=y_var,
                                   m_em=m_em, s_em=s_em, ms_em=ms_em,
                                   ms_tm=ms_tm, yr=yr, y_committed=y_committed,
                                   T=T, E=E, F=F)
    inputs = [m.var] + [x.var for x in xs] + ([y_var] if y_committed else [])
    outs = tape._process_claim(claim, inputs)
    tape.claims.append(claim)
    if y_committed:
        return y_wt
    return WitnessTensor(outs[y_var] if outs else None, y_var, (T, F), tape)


# ── Fold-consumer registration: absorb expert streams as their matmuls retire
#    (frees each X_e immediately — breaks the E·T·F residency barrier).
#    Values are bit-identical to the atomic path (field ops are exact), which
#    test_fc_fold_golden_roots enforces via Merkle-root equality. ──

def _fc_foldable(c: FreivaldsCombineClaim):
    return list(c.xs)


def _fc_init(c: FreivaldsCombineClaim, live):
    T, E, F = c.T, c.E, c.F
    m = live[c.m]
    m = (m() if callable(m) else m).contiguous().view(torch.int64).view(T, E)
    return {"mi": m, "y_acc": torch.zeros(T * F, dtype=torch.uint64, device="cuda"),
            "s_rows": {}, "_eidx": {v: e for e, v in enumerate(c.xs)}}


def _fc_absorb(c: FreivaldsCombineClaim, st, v, t):
    T, F = c.T, c.F
    e = st["_eidx"][v]
    rep = (st["mi"][:, e].contiguous().view(T, 1).expand(T, F).contiguous()
           .view(torch.uint64).reshape(-1))
    st["y_acc"] = gl_add(st["y_acc"], gl_mul(rep, t.contiguous().view(-1)))


def _fc_aux_absorb(c: FreivaldsCombineClaim, st, v, t, ch):
    from cuda_primitives import gl_matvec
    rho = st.get("_rho")
    if rho is None:
        rho = st["_rho"] = torch.tensor(ch, dtype=torch.uint64, device="cuda")
    st["s_rows"][st["_eidx"][v]] = gl_matvec(t.contiguous().view(c.T, c.F), rho)


def _fc_finalize(c: FreivaldsCombineClaim, st, live):
    m_em = st["mi"].T.contiguous().view(torch.uint64).reshape(-1)
    outs = {c.m_em: m_em}
    if not c.y_committed:
        outs[c.y] = st["y_acc"]
    return _apply_tamper(outs, [("m_em", c.m_em)] +
                          ([] if c.y_committed else [("y", c.y)]))


def _fc_aux_finalize(c: FreivaldsCombineClaim, st, live, ch):
    from cuda_primitives import gl_matvec
    T, E, F = c.T, c.E, c.F
    rho = st.get("_rho")
    if rho is None:
        rho = torch.tensor(ch, dtype=torch.uint64, device="cuda")
    s_em = torch.stack([st["s_rows"][e] for e in range(E)]).reshape(-1)
    m_em = live[c.m_em]
    m_em = (m_em() if callable(m_em) else m_em).contiguous().view(-1)
    ms_em = gl_mul(m_em, s_em)
    ms_tm = (ms_em.view(torch.int64).view(E, T).T.contiguous()
             .view(torch.uint64).reshape(-1))
    y = live[c.y]
    y = (y() if callable(y) else y).contiguous()
    yr = gl_matvec(y.view(T, F), rho)
    outs = {c.s_em: s_em, c.ms_em: ms_em, c.ms_tm: ms_tm, c.yr: yr}
    return _apply_tamper(outs, [("s_em", c.s_em), ("ms_em", c.ms_em),
                                 ("ms_tm", c.ms_tm), ("yr", c.yr)])


_cf.FOLD_FNS[FreivaldsCombineClaim] = dict(
    foldable=_fc_foldable, init=_fc_init, absorb=_fc_absorb,
    finalize=_fc_finalize, aux_absorb=_fc_aux_absorb,
    aux_finalize=_fc_aux_finalize)
