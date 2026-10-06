"""Top-k routing with output-side gates (analysis/topk-routing-design.md),
items 1–3 on the toy tape: the threshold selection, the gate bracket, and the
stacked expert chain.

TopkRoutingClaim (§3.1) proves that a committed k-hot mask m (T, E) is the
top-k of the tiebroken selection quantity q̃ = 2^L·(s + b) + (E − 1 − e),
through a free per-token threshold τ:

  F1  q̃ − 2^L·s − 2^L·b(bcast) = (E − 1 − e)   [E·T]  cids expert-major (e·T + t),
                                                       so b fans out by stride
  F2  Σ_e m[t,e] = k                            [T]
  Fd  d − q̃ + τ(bcast) = 0                      [T·E]  d = q̃ − τ
  Fv  v − 2·md + d = 0                          [T·E]  v = (2m − 1)(q̃ − τ)
  Q1  m·m = m      Q2  m·d = md
  v ∈ [0, 2^R) by word_extract in the builder.

Sound under the threshold guard 2^{R+1} − 1 ≤ P − 2^width (topk_params), where
2^width bounds any difference of two q̃: an upstream hypothesis on the score
table and the committed bias. Top-1's guard is not enough (the design's
counterexample).

TopkSlotsClaim (§3.3) splits m into k one-hot slot masks M_i and reads the
slot scores ss[t,i] = s[t, e_i]:

  Sb  Σ_i M_i − m = 0                           [T·E]
  Sc  Σ_e M_i[t,e] = 1                          [k·T]  slot-major (i·T + t)
  Sv  Σ_e MS_i[t,e] − ss[t,i] = 0               [k·T]  ss token-major, transposed in
  Q   M_i·M_i = M_i (each i), then M_i·s = MS_i (each i)

GateBracketClaim (§3.2) pins w[t,i] = ⌊C·ss[t,i] / Z[t]⌋, Z[t] = Σ_i ss[t,i]:

  Bz  Σ_i ss[t,i] − Z[t] = 0                    [T]
  Bb  Zb − Z(bcast) = 0                         [T·k]
  Bd  wZ + rem − C·ss = 0                       [T·k]
  Bg  gr − Zb + rem = −1                        [T·k]  gr = Z − 1 − rem
  Q   w·Zb = wZ
  rem, gr ∈ [0, 2^rem_bits), w ∈ [0, 2^w_bits) by word_extract in the builder.

SplitClaim is ConcatClaim read the other way: the parts are derived from the
whole. The stacked chain (topk_moe_ffn) runs one routed claim per weight set
on the k slots stacked along the token axis (T' = kT), and combines the k
slices of the down projection with the gate weights before one rescale.

Every compile here has its twin in verifier/src/handlers.rs; the emission
order of the linear families and of the quads is load-bearing."""
from dataclasses import dataclass
from typing import List, Tuple

import torch

import compute_fns as _cf
from claims import COMPILE_FNS, QuadFamily, _build_b_chunk
from core import AUX_FNS, P, SAMPLE_FNS, Variable
from cuda_primitives import gl_add, gl_mul, gl_sub
from max_claim import to_signed
from packets import (L2_IdentityScalar, L2_RowSumPerSlotVector,
                     L2_StrideOneToManyScalar, L2_TransposeO2MScalar)
from topk_params import bracket_guard_ok, bracket_sizes, l_bits, threshold_words

NEG1 = (P - 1) % P


def _rows(pkts, var, make, ell):
    """One packet per witness row of `var`, as every compile in the prover
    emits them."""
    for ro in range(var.n_rows(ell)):
        pkts.append((var.row_start + ro, make(var)))


def _id(base, coef):
    return lambda v: L2_IdentityScalar(base=base, var_row_start=v.row_start,
                                       L=v.length, coef=coef % P)


def _rowsum(base, stride, coef_vec):
    return lambda v: L2_RowSumPerSlotVector(base=base, var_row_start=v.row_start,
                                            L=v.length, stride=stride,
                                            coef_vec=coef_vec)


def _o2m(base, stride, coef):
    return lambda v: L2_StrideOneToManyScalar(base=base, var_row_start=v.row_start,
                                              L=v.length, stride=stride, coef=coef % P)


def _transpose(base, rows, cols, coef):
    return lambda v: L2_TransposeO2MScalar(base=base, var_row_start=v.row_start,
                                           L=v.length, rows=rows, cols=cols, fan=1,
                                           coef=coef % P)


def _u64(x):
    return x.to(torch.int64).to(torch.uint64)


def _ones(n):
    return torch.ones(n, dtype=torch.uint64, device="cuda")


def _none_sample(c, ci, s_op):
    return None


def _no_aux(c, witness, ch):
    return {}


# ===========================================================================
# TopkRoutingClaim
# ===========================================================================

@dataclass
class TopkRoutingClaim:
    s: Variable            # selection scores (T*E), token-major, from upstream
    b: Variable            # selection bias (E), committed with the layer
    m: Variable            # k-hot mask (T*E)
    qt: Variable           # 2^L·(s + b) + (E − 1 − e) (T*E)
    tau: Variable          # per-token threshold (T)
    d: Variable            # qt − tau (T*E)
    md: Variable           # m·d (T*E)
    v: Variable            # 2·md − d (T*E), ranged by the builder's word_extract
    T: int
    E: int
    k: int
    L_bits: int
    width: int             # 2^width bounds any difference of two qt (hypothesis)
    range_bits: int        # R = n_words·word_bits of the range on v

    @property
    def length(self):
        return self.T * self.E


def topk_compute(c: TopkRoutingClaim, live):
    T, E, k = c.T, c.E, c.k
    s = live[c.s].contiguous().view(-1)
    b = live[c.b].contiguous().view(-1).view(torch.int64).repeat(T).view(torch.uint64)
    bonus = torch.arange(E - 1, -1, -1, dtype=torch.int64, device="cuda").repeat(T)
    two_l = torch.full((T * E,), 1 << c.L_bits, dtype=torch.uint64, device="cuda")
    qt = gl_add(gl_mul(gl_add(s, b), two_l), _u64(bonus))
    sq = to_signed(qt).view(T, E)
    top = torch.topk(sq, k, dim=1)                 # q̃ are distinct: no tie to break
    m = torch.zeros(T, E, dtype=torch.int64, device="cuda")
    m.scatter_(1, top.indices, 1)
    # τ is the k-th largest q̃, read from q̃ itself: casting the signed value
    # back would give 2^64 + x for a negative x, not its field element
    tau = (qt.view(torch.int64).view(T, E).gather(1, top.indices[:, k - 1:k])
           .contiguous().view(torch.uint64).reshape(-1))
    tau_bc = tau.view(T, 1).expand(T, E).contiguous().reshape(-1)
    d = gl_sub(qt, tau_bc)
    m = _u64(m.reshape(-1))
    md = gl_mul(m, d)
    v = gl_sub(gl_add(md, md), d)
    return {c.m: m, c.qt: qt, c.tau: tau, c.d: d, c.md: md, c.v: v}


def topk_compile(c: TopkRoutingClaim, _ch, cfg, base: int):
    ell = cfg.ELL
    T, E, k = c.T, c.E, c.k
    L = T * E
    two_l = (1 << c.L_bits) % P
    pk: List[Tuple[int, object]] = []
    cur = base
    # F1, expert-major cids e·T + t: the bias fans out along its expert's block
    f1 = cur - base
    _rows(pk, c.qt, _transpose(cur, T, E, 1), ell)
    _rows(pk, c.s, _transpose(cur, T, E, P - two_l), ell)
    _rows(pk, c.b, _o2m(cur, T, P - two_l), ell)
    cur += L
    # F2
    f2 = cur - base
    _rows(pk, c.m, _rowsum(cur, E, _ones(E)), ell)
    cur += T
    # Fd
    _rows(pk, c.d, _id(cur, 1), ell)
    _rows(pk, c.qt, _id(cur, NEG1), ell)
    _rows(pk, c.tau, _o2m(cur, E, 1), ell)
    cur += L
    # Fv
    _rows(pk, c.v, _id(cur, 1), ell)
    _rows(pk, c.md, _id(cur, P - 2), ell)
    _rows(pk, c.d, _id(cur, 1), ell)
    cur += L
    quads = [QuadFamily(name="Topk.mm", x_row=c.m.row_start, y_row=c.m.row_start,
                        z_row=c.m.row_start, L=L, ell=cfg.ELL, a=NEG1, b=0),
             QuadFamily(name="Topk.md", x_row=c.m.row_start, y_row=c.d.row_start,
                        z_row=c.md.row_start, L=L, ell=cfg.ELL, a=NEG1, b=0)]
    rhs = [(f1 + e * T, T, E - 1 - e) for e in range(E - 1)]
    rhs.append((f2, T, k))
    return pk, quads, cur - base, _build_b_chunk(cur - base, rhs)


COMPILE_FNS[TopkRoutingClaim] = topk_compile
SAMPLE_FNS[TopkRoutingClaim] = _none_sample
AUX_FNS[TopkRoutingClaim] = _no_aux
_cf.COMPUTE_FNS[TopkRoutingClaim] = topk_compute

_BUILD = [0]


def route_topk(tape, s, b, *, T, E, k, width, word_bits=12):
    """The TopkRoutingClaim on scores `s` (T, E) and bias `b` (E,), and the
    range on v. `width` bounds any difference of two tiebroken quantities
    (an upstream hypothesis: the score table's range and the committed
    bias); the threshold guard is checked here and again by the verifier.
    Returns (m, tau)."""
    from tape import WitnessTensor
    assert 1 <= k < E, "top-k needs 1 <= k < E"
    L_bits = l_bits(E)
    n_words, R = threshold_words(width, word_bits)
    _BUILD[0] += 1
    pfx = f"tk{_BUILD[0]}_"
    v = {n: tape._alloc(f"{pfx}{n}", T * E) for n in ("m", "qt", "d", "md", "v")}
    tau = tape._alloc(f"{pfx}tau", T)
    claim = TopkRoutingClaim(s=s.var, b=b.var, m=v["m"], qt=v["qt"], tau=tau,
                             d=v["d"], md=v["md"], v=v["v"], T=T, E=E, k=k,
                             L_bits=L_bits, width=width, range_bits=R)
    outs = tape._process_claim(claim, [s.var, b.var])
    tape.claims.append(claim)
    wt = lambda var, shape: WitnessTensor(outs[var] if outs else None, var, shape, tape)
    table = tape.register_table(f"{pfx}rng", T_data=list(range(1 << word_bits)))
    tape.word_extract(wt(v["v"], (T, E)), table, B=word_bits, N=n_words)
    return wt(v["m"], (T, E)), wt(tau, (T, 1))


# ===========================================================================
# TopkSlotsClaim
# ===========================================================================

@dataclass
class TopkSlotsClaim:
    m: Variable            # k-hot mask (T*E)
    s: Variable            # the original scores (T*E), not the biased ones
    M: List[Variable]      # k one-hot slot masks, each (T*E)
    MS: List[Variable]     # M_i·s, each (T*E)
    ss: Variable           # slot scores (T*k), token-major
    T: int
    E: int
    k: int

    @property
    def length(self):
        return self.T * self.E


def slots_compute(c: TopkSlotsClaim, live):
    T, E, k = c.T, c.E, c.k
    m = live[c.m].contiguous().view(T, E).view(torch.int64)
    s = live[c.s].contiguous().view(-1)
    # the i-th selected expert by ascending index: the canonical assignment
    rank = torch.cumsum(m, dim=1) - 1
    outs = {}
    ss = torch.zeros(T, k, dtype=torch.int64, device="cuda")
    for i in range(k):
        Mi = ((m == 1) & (rank == i)).to(torch.int64)
        Mi_u = _u64(Mi.reshape(-1))
        MSi = gl_mul(Mi_u, s)
        outs[c.M[i]] = Mi_u
        outs[c.MS[i]] = MSi
        ss[:, i] = (Mi * s.view(T, E).view(torch.int64)).sum(dim=1)
    outs[c.ss] = _u64(ss.reshape(-1))
    return outs


def slots_compile(c: TopkSlotsClaim, _ch, cfg, base: int):
    ell = cfg.ELL
    T, E, k = c.T, c.E, c.k
    pk: List[Tuple[int, object]] = []
    cur = base
    # Sb
    for Mi in c.M:
        _rows(pk, Mi, _id(cur, 1), ell)
    _rows(pk, c.m, _id(cur, NEG1), ell)
    cur += T * E
    # Sc, slot-major
    sc = cur - base
    for i, Mi in enumerate(c.M):
        _rows(pk, Mi, _rowsum(cur + i * T, E, _ones(E)), ell)
    cur += k * T
    # Sv, slot-major; ss is token-major (t·k + i), transposed in
    for i, MSi in enumerate(c.MS):
        _rows(pk, MSi, _rowsum(cur + i * T, E, _ones(E)), ell)
    _rows(pk, c.ss, _transpose(cur, T, k, NEG1), ell)
    cur += k * T
    quads = [QuadFamily(name=f"Slots.MM{i}", x_row=Mi.row_start, y_row=Mi.row_start,
                        z_row=Mi.row_start, L=T * E, ell=cfg.ELL, a=NEG1, b=0)
             for i, Mi in enumerate(c.M)]
    quads += [QuadFamily(name=f"Slots.MS{i}", x_row=c.M[i].row_start,
                         y_row=c.s.row_start, z_row=c.MS[i].row_start, L=T * E,
                         ell=cfg.ELL, a=NEG1, b=0) for i in range(k)]
    return pk, quads, cur - base, _build_b_chunk(cur - base, [(sc, k * T, 1)])


COMPILE_FNS[TopkSlotsClaim] = slots_compile
SAMPLE_FNS[TopkSlotsClaim] = _none_sample
AUX_FNS[TopkSlotsClaim] = _no_aux
_cf.COMPUTE_FNS[TopkSlotsClaim] = slots_compute


def topk_slots(tape, m, s, *, T, E, k):
    """Split the k-hot mask into k one-hot slot masks (each (T, E)) and read
    the slot scores (T, k). Returns ([M_0 … M_{k-1}], ss)."""
    from tape import WitnessTensor
    _BUILD[0] += 1
    pfx = f"sl{_BUILD[0]}_"
    M = [tape._alloc(f"{pfx}M{i}", T * E) for i in range(k)]
    MS = [tape._alloc(f"{pfx}MS{i}", T * E) for i in range(k)]
    ss = tape._alloc(f"{pfx}ss", T * k)
    claim = TopkSlotsClaim(m=m.var, s=s.var, M=M, MS=MS, ss=ss, T=T, E=E, k=k)
    outs = tape._process_claim(claim, [m.var, s.var])
    tape.claims.append(claim)
    wt = lambda var, shape: WitnessTensor(outs[var] if outs else None, var, shape, tape)
    return [wt(Mi, (T, E)) for Mi in M], wt(ss, (T, k))


# ===========================================================================
# GateBracketClaim
# ===========================================================================

@dataclass
class GateBracketClaim:
    ss: Variable           # slot scores (T*k), token-major
    Z: Variable            # Σ_i ss[t,i] (T)
    Zb: Variable           # Z broadcast (T*k)
    w: Variable            # gate weights ⌊C·ss/Z⌋ (T*k)
    wZ: Variable           # w·Zb (T*k)
    rem: Variable          # C·ss − w·Z (T*k)
    gr: Variable           # Z − 1 − rem (T*k)
    T: int
    k: int
    C: int                 # round(c · S_w), public
    z_bits: int            # Z < 2^z_bits      (from the score bound, a hypothesis)
    cs_bits: int           # C·ss < 2^cs_bits  (likewise)
    rem_bits: int          # range on rem and gr
    w_bits: int            # range on w

    @property
    def length(self):
        return self.T * self.k


def bracket_compute(c: GateBracketClaim, live):
    T, k = c.T, c.k
    ss = live[c.ss].contiguous().view(T, k).view(torch.int64)
    Z = ss.sum(dim=1)
    Zb = Z.view(T, 1).expand(T, k)
    w = (c.C * ss) // Zb
    rem = c.C * ss - w * Zb
    gr = Zb - 1 - rem
    flat = lambda x: _u64(x.contiguous().reshape(-1))
    return {c.Z: flat(Z), c.Zb: flat(Zb), c.w: flat(w), c.wZ: flat(w * Zb),
            c.rem: flat(rem), c.gr: flat(gr)}


def bracket_compile(c: GateBracketClaim, _ch, cfg, base: int):
    ell = cfg.ELL
    T, k = c.T, c.k
    pk: List[Tuple[int, object]] = []
    cur = base
    # Bz
    _rows(pk, c.ss, _rowsum(cur, k, _ones(k)), ell)
    _rows(pk, c.Z, _id(cur, NEG1), ell)
    cur += T
    # Bb
    _rows(pk, c.Zb, _id(cur, 1), ell)
    _rows(pk, c.Z, _o2m(cur, k, NEG1), ell)
    cur += T * k
    # Bd
    _rows(pk, c.wZ, _id(cur, 1), ell)
    _rows(pk, c.rem, _id(cur, 1), ell)
    _rows(pk, c.ss, _id(cur, P - c.C % P), ell)
    cur += T * k
    # Bg
    bg = cur - base
    _rows(pk, c.gr, _id(cur, 1), ell)
    _rows(pk, c.Zb, _id(cur, NEG1), ell)
    _rows(pk, c.rem, _id(cur, 1), ell)
    cur += T * k
    quads = [QuadFamily(name="Bracket.wZ", x_row=c.w.row_start, y_row=c.Zb.row_start,
                        z_row=c.wZ.row_start, L=T * k, ell=cfg.ELL, a=NEG1, b=0)]
    return pk, quads, cur - base, _build_b_chunk(cur - base, [(bg, T * k, NEG1)])


COMPILE_FNS[GateBracketClaim] = bracket_compile
SAMPLE_FNS[GateBracketClaim] = _none_sample
AUX_FNS[GateBracketClaim] = _no_aux
_cf.COMPUTE_FNS[GateBracketClaim] = bracket_compute


def gate_bracket(tape, ss, *, T, k, C, score_bits, word_bits=8):
    """w[t,i] = ⌊C·ss[t,i] / Σ_i ss[t,i]⌋ (T, k). `score_bits` bounds the slot
    scores (the score table's range, a hypothesis), which fixes the bounds on
    Z and C·ss; the ranges are sized from them and the bracket guard is
    checked here and again by the verifier."""
    from tape import WitnessTensor
    sz = bracket_sizes(score_bits, k, C, word_bits)
    z_bits, cs_bits, n_rem, n_w = sz["z_bits"], sz["cs_bits"], sz["n_rem"], sz["n_w"]
    rem_bits, w_bits = sz["rem_bits"], sz["w_bits"]
    assert bracket_guard_ok(z_bits, rem_bits, w_bits, cs_bits), (
        f"unsound gate bracket: z {z_bits}, rem {rem_bits}, w {w_bits}, C·s {cs_bits} bits")
    _BUILD[0] += 1
    pfx = f"gb{_BUILD[0]}_"
    n = {x: tape._alloc(f"{pfx}{x}", T * k) for x in ("Zb", "w", "wZ", "rem", "gr")}
    Z = tape._alloc(f"{pfx}Z", T)
    claim = GateBracketClaim(ss=ss.var, Z=Z, Zb=n["Zb"], w=n["w"], wZ=n["wZ"],
                             rem=n["rem"], gr=n["gr"], T=T, k=k, C=C, z_bits=z_bits,
                             cs_bits=cs_bits, rem_bits=rem_bits, w_bits=w_bits)
    outs = tape._process_claim(claim, [ss.var])
    tape.claims.append(claim)
    wt = lambda var: WitnessTensor(outs[var] if outs else None, var, (T, k), tape)
    table = tape.register_table(f"{pfx}rng", T_data=list(range(1 << word_bits)))
    tape.word_extract(wt(n["rem"]), table, B=word_bits, N=n_rem)
    tape.word_extract(wt(n["gr"]), table, B=word_bits, N=n_rem)
    tape.word_extract(wt(n["w"]), table, B=word_bits, N=n_w)
    return wt(n["w"])


# ===========================================================================
# SplitClaim: parts derived from a whole, the ConcatClaim relation
# ===========================================================================

@dataclass
class SplitClaim:
    whole: Variable
    parts: List[Variable]

    @property
    def length(self):
        return self.whole.length


def split_compute(c: SplitClaim, live):
    x = live[c.whole].contiguous().view(-1)
    out, off = {}, 0
    for p in c.parts:
        out[p] = x[off:off + p.length].contiguous()
        off += p.length
    return out


def split_compile(c: SplitClaim, _ch, cfg, base: int):
    ell = cfg.ELL
    assert sum(p.length for p in c.parts) == c.whole.length, "split parts must cover the whole"
    pk: List[Tuple[int, object]] = []
    _rows(pk, c.whole, _id(base, NEG1), ell)
    off = 0
    for p in c.parts:
        _rows(pk, p, _id(base + off, 1), ell)
        off += p.length
    return pk, [], c.whole.length, None


COMPILE_FNS[SplitClaim] = split_compile
SAMPLE_FNS[SplitClaim] = _none_sample
AUX_FNS[SplitClaim] = _no_aux
_cf.COMPUTE_FNS[SplitClaim] = split_compute


def split(tape, x, n_parts, part_shape):
    """x's flat slots cut into n_parts equal parts, each of part_shape."""
    from tape import WitnessTensor
    L = x.var.length
    assert L % n_parts == 0
    parts = [tape._alloc(f"{x.var.name[:12]}_part{i}", L // n_parts) for i in range(n_parts)]
    claim = SplitClaim(whole=x.var, parts=parts)
    outs = tape._process_claim(claim, [x.var])
    tape.claims.append(claim)
    return [WitnessTensor(outs[p] if outs else None, p, part_shape, tape) for p in parts]


# ===========================================================================
# The stacked chain (§3.3)
# ===========================================================================

def topk_moe_ffn(tape, x, s, b, w, *, T, E, k, d, d_ff, S, S_w, C, score_bits,
                 width, output_width, select_word_bits=12, bracket_word_bits=8,
                 use_bridge=False, trace=None):
    """One top-k MoE FFN with output-side gates.

    x (T, d) at scale S; s (T, E) the scores at their own scale (< 2^score_bits);
    b (E,) the selection bias; w["gate"], w["up"] lists of E shards (d, d_ff),
    w["down"] E shards (d_ff, d), all at scale S. Returns y (T, d) at scale S:
      y[t] = Σ_i w_i[t] · FFN_{e_i}(x[t]),  w_i = ⌊C·s_i / Σ s⌋ at scale S_w.
    `use_bridge`: the shards are external inputs the weight enrollment
    authenticates (routed_projected_matmul's bridge). `trace`, a dict, receives
    the intermediates (m, gw, g, h, D, y_raw, y) for an exact comparison."""
    from rescale_claim import rescale
    from routed_projected import routed_projected_matmul
    from routing_claim import freivalds_combine
    assert tape.silu_config.s_in in (0, tape.silu_config.s_x) and tape.silu_config.s_x == S, (
        f"topk_moe_ffn: the tape's SiLU tables are at scale {tape.silu_config.s_x}, "
        f"the activations at {S}: build the tape with a SiluConfig of r = log2(S)")
    m, _tau = route_topk(tape, s, b, T=T, E=E, k=k, width=width,
                         word_bits=select_word_bits)
    M, ss = topk_slots(tape, m, s, T=T, E=E, k=k)
    gw = gate_bracket(tape, ss, T=T, k=k, C=C, score_bits=score_bits,
                      word_bits=bracket_word_bits)
    Tk = k * T
    M_k = tape.concat(M, (Tk, E))                       # slot-major rows i·T + t
    x_k = tape.concat([x] * k, (Tk, d))
    rp = dict(s_in=S * S, s_out=S, output_width=output_width)
    g = rescale(tape, routed_projected_matmul(tape, x_k, M_k, w["gate"], T=Tk, K=d,
                                              J=d_ff, E=E, use_bridge=use_bridge), **rp)
    u = rescale(tape, routed_projected_matmul(tape, x_k, M_k, w["up"], T=Tk, K=d,
                                              J=d_ff, E=E, use_bridge=use_bridge), **rp)
    h = tape.hadamard(tape.silu(g), u, s_a=S, s_b=S, s_out=S, output_width=output_width)
    D = routed_projected_matmul(tape, h, M_k, w["down"], T=Tk, K=d_ff, J=d, E=E,
                                use_bridge=use_bridge)
    parts = split(tape, D, k, (T, d))                   # D_i: slot i's raw output, S²
    y_raw = freivalds_combine(tape, gw, parts, T=T, E=k, F=d)   # at S_w·S²
    y = rescale(tape, y_raw, s_in=S_w * S * S, s_out=S, output_width=output_width)
    if trace is not None:
        trace.update(m=m, gw=gw, g=g, h=h, D=D, y_raw=y_raw, y=y)
    return y
