"""Kimi K2's integer forward pass, apart from the prover: the two-layer gate's
reference (analysis/k2-session-sizing.md §5–6).

Every value the driver's claims compute is computed here from the claims'
integer definitions, as signed integers, so the gate can require EXACT
agreement with the tape's engine pass, intermediate by named intermediate,
before any float comparison (k2_float_reference.py does those). The
definitions, all on signed integers:

  rescale(v, r)  ⌊v / 2^r⌋, with |output| < 2^(width-1) and the unrescaled v
                 a field representative (|v| < P/2)
  matmul         the exact integer product, then rescale by log2(s_a·s_b/s_out)
  rmsnorm        S_tot = Σx² + d·ε_int; y = the least y ≥ 1 with
                 y²·S_tot ≥ d·s⁴; output = rescale(x·y, log2(s·s/s_out))
  rope           half-split pairs (k, k + d_h/2) with the public integer
                 cos/sin tables, then rescale
  softmax        per row (b = t·H + h, keys u ≤ t), c2 = the least c ≥ max_x
                 with Σ_{0 ≤ c − x_u < Z_max} T_A[c − x_u] ≤ s_y;
                 y = T_A[c2 − x] (0 where masked or past the table)
  silu           |x| ≥ b_2: max(x, 0); else the paired table at
                 sign·T_LEN + (|x| // b) mod T_LEN
  sigmoid        the paired table at r + shift
  top-k, slots,  prover/topk_reference.py (#25): the tiebroken top-k of
  gate bracket   2^L·(s + b) + (E − 1 − e), slots by ascending expert index,
                 w = ⌊C·s_i / Σ s⌋
  combine        Σ_i w_i·D_i exactly, then rescale by log2(S_w·S)

Shared with the prover: the public tables (softmax exp, SiLU, rotary,
sigmoid), the top-k reference, and the weights (k2_loader). The forward
computation is written here independently. The reference stops at the
logits; the unexplained-information tail is the prover's.

Products are exact: float64 BLAS when every partial sum is below 2^53,
int64 below 2^62, limbs and Python integers beyond. Along the way `Ranges`
records each quantity whose window the prover fixes (§6.2's measurements);
a quantity outside its window is reported, not asserted, so one run
measures all of them."""
import math

import numpy as np

import topk_reference as tref
from k2_loader import K2_INT, sigmoid_table
from topk_params import P, l_bits

P_HALF = P // 2


# ---- exact integer arithmetic -------------------------------------------------------

def _maxabs(a) -> int:
    a = np.asarray(a)
    if a.size == 0:
        return 0
    if a.dtype == object:
        return max(abs(int(v)) for v in a.ravel())
    return max(abs(int(a.max())), abs(int(a.min())))


def _narrow(a):
    """An object array back to int64 when every value fits."""
    if a.dtype == object and _maxabs(a) < (1 << 62):
        return a.astype(np.int64)
    return a


def mm(a, b):
    """Exact integer np.matmul(a, b) (batched dims allowed)."""
    a = np.asarray(a); b = np.asarray(b)
    if a.dtype == object or b.dtype == object:
        return _narrow(np.matmul(a.astype(object), b.astype(object)))
    K = a.shape[-1]
    bound = _maxabs(a) * _maxabs(b) * max(K, 1)
    if bound < (1 << 53):
        return np.rint(np.matmul(a.astype(np.float64), b.astype(np.float64))).astype(np.int64)
    if bound < (1 << 62):
        return np.matmul(a.astype(np.int64), b.astype(np.int64))
    # a in base-2^k digits (the last one 0 or -1, an arithmetic shift's
    # floor), each digit's product below 2^61
    k = 61 - (_maxabs(b) * K).bit_length()
    if k < 8:
        return _narrow(np.matmul(a.astype(object), b.astype(object)))
    rest, b, shift, acc = a.astype(np.int64), b.astype(np.int64), 0, 0
    while not ((rest == 0) | (rest == -1)).all():
        acc = acc + (np.matmul(rest & ((1 << k) - 1), b).astype(object) << shift)
        rest, shift = rest >> k, shift + k
    return _narrow(acc + (np.matmul(rest, b).astype(object) << shift))


def mul(a, b):
    """Exact elementwise product."""
    a = np.asarray(a); b = np.asarray(b)
    if a.dtype != object and b.dtype != object and _maxabs(a) * _maxabs(b) < (1 << 62):
        return a.astype(np.int64) * b.astype(np.int64)
    return _narrow(a.astype(object) * b.astype(object))


def floor_shift(v, r: int):
    """⌊v / 2^r⌋ (arithmetic shift), int64 or object."""
    v = np.asarray(v)
    if v.dtype == object:
        return _narrow(np.vectorize(lambda x: int(x) >> r, otypes=[object])(v))
    return v >> r


def log2_ratio(s_in: int, s_out: int) -> int:
    r = (s_in // s_out).bit_length() - 1
    assert s_out << r == s_in, (s_in, s_out)
    return r


class Ranges:
    """name -> (observed, limit, note): observed must stay below limit."""

    def __init__(self):
        self.rows = {}

    def note(self, name, observed, limit, what=""):
        prev = self.rows.get(name)
        if prev is None or observed > prev[0]:
            self.rows[name] = (int(observed), int(limit), what)

    def violations(self):
        return {n: r for n, r in self.rows.items() if not r[0] < r[1]}

    def as_dict(self):
        return {n: dict(observed=o, limit=l, bits=o.bit_length(), limit_bits=l.bit_length(),
                        ok=o < l, what=w) for n, (o, l, w) in self.rows.items()}


class Int:
    """The integer ops, recording ranges as they go."""

    def __init__(self, k=K2_INT, ranges=None):
        self.k = k
        self.S = k["S"]
        self.width = k["output_width"]
        self.ranges = ranges if ranges is not None else Ranges()

    def rescale(self, full, r, name):
        self.ranges.note(f"{name}.full", _maxabs(full), P_HALF + 1, "field representative")
        out = floor_shift(full, r)
        self.ranges.note(f"{name}.out", _maxabs(out), 1 << (self.width - 1),
                         f"{self.width}-bit output")
        return out if np.asarray(out).dtype == object else np.asarray(out, dtype=np.int64)

    def matmul(self, a, w, name):
        return self.rescale(mm(a, w), log2_ratio(self.S * self.S, self.S), name)

    def hadamard(self, a, b, name):
        return self.rescale(mul(a, b), log2_ratio(self.S * self.S, self.S), name)

    def gain(self, x, g, name):
        return self.hadamard(x, np.asarray(g, dtype=np.int64)[None, :], name)

    def rmsnorm(self, x, name):
        """Row-wise over the last axis; Python integers for the bracket."""
        S, eps_int = self.S, self.k["eps_int"]
        x = np.asarray(x, dtype=np.int64)
        B, d = x.shape
        magic = d * S ** 4
        st = [v + d * eps_int for v in _row_energy(x)]
        ys = []
        for s_tot in st:
            assert s_tot > 0, "eps_int >= 1 keeps the row energy positive"
            y = max(1, math.isqrt(magic // s_tot))
            while y * y * s_tot < magic:
                y += 1
            while y > 1 and (y - 1) * (y - 1) * s_tot >= magic:
                y -= 1
            ys.append(y)
        from claims import RMS_S_CAP_BITS, RmsNormConfig
        self.ranges.note(f"{name}.S_tot", max(st), 1 << RMS_S_CAP_BITS,
                         f"row energy against the 2^{RMS_S_CAP_BITS} limb cap (d = {d})")
        y_max = RmsNormConfig(B=1, d=d, s=S, eps_int=eps_int).y_max
        self.ranges.note(f"{name}.y", max(ys), y_max + 1,
                         f"inverse RMS against y_max = {y_max}, the ε floor (d = {d})")
        return self.rescale(mul(x, np.asarray(ys, dtype=np.int64)[:, None]),
                            log2_ratio(S * S, S), name)

    def rope(self, x, cos, sin, *, T, heads, d_h, name):
        half = d_h // 2
        z = np.asarray(x, dtype=np.int64).reshape(T, heads, d_h)
        lo, hi = z[..., :half], z[..., half:]
        c, s = cos[:, None, :], sin[:, None, :]
        rot = np.concatenate([mul(c, lo) - mul(s, hi), mul(s, lo) + mul(c, hi)], axis=-1)
        return self.rescale(rot.reshape(T, heads * d_h), log2_ratio(self.S * self.S, self.S), name)

    def softmax(self, x, *, T, H, name, rows_per_chunk=4096):
        """Causal, rows b = t·H + h over the T keys, by a vectorized bisection
        on the monotone predicate s1(c) <= s_y over [max_x, max_x + Z_max]
        (s1 is non-increasing in c, and 0 at max_x + Z_max)."""
        TA = self.T_A
        Z, s_y = self.k["z_max"], self.S
        xs = np.asarray(x, dtype=np.int64).reshape(T * H, T)
        out = np.zeros_like(xs)
        q_pos = np.arange(T * H) // H
        spread, past, c2_abs = 0, 0, 0
        for r0 in range(0, T * H, rows_per_chunk):
            r1 = min(T * H, r0 + rows_per_chunk)
            xc = xs[r0:r1]
            live = np.arange(T)[None, :] <= q_pos[r0:r1, None]
            mx = np.where(live, xc, np.iinfo(np.int64).min).max(axis=1)
            mn = np.where(live, xc, np.iinfo(np.int64).max).min(axis=1)

            def s1(c):
                z = c[:, None] - xc
                ok = live & (z >= 0) & (z < Z)
                return np.where(ok, TA[np.where(ok, z, 0)], 0).sum(axis=1)
            lo, hi = mx.copy(), mx + Z
            done = s1(lo) <= s_y
            while True:
                act = ~done & (hi - lo > 1)
                if not act.any():
                    break
                mid = (lo + hi) // 2
                fits = s1(mid) <= s_y
                hi = np.where(act & fits, mid, hi)
                lo = np.where(act & ~fits, mid, lo)
            c2 = np.where(done, lo, hi)
            z = c2[:, None] - xc
            ok = live & (z >= 0) & (z < Z)
            out[r0:r1] = np.where(ok, TA[np.where(ok, z, 0)], 0)
            spread = max(spread, int((c2 - mn).max()))
            c2_abs = max(c2_abs, int(np.abs(c2).max()))
            past = max(past, int((live & (z >= Z)).sum(axis=1).max()))
        self.ranges.note(f"{name}.spread", spread, Z << 16,
                         "c2 − min x against Z_max·2^Z_high_width (the saturating window)")
        self.ranges.note(f"{name}.c2", c2_abs, 1 << 23,
                         "|c2| against the signed 24-bit aux chunk (aux_chunk_width 24)")
        self.ranges.note(f"{name}.keys_past_table", past, T + 1,
                         "most keys in a row beyond Z_max (weight 0); informational")
        return out.reshape(T, H * T)

    def silu(self, x):
        cfg = self.silu_cfg
        x = np.asarray(x, dtype=np.int64)
        mag = np.abs(x)
        key = (x < 0).astype(np.int64) * cfg.T_LEN + (mag // cfg.b) % cfg.T_LEN
        return np.where(mag >= cfg.b_2, np.maximum(x, 0), self.silu_TY[key])

    def lookup_sigmoid(self, r, name):
        TY, shift = self.sigmoid
        idx = np.asarray(r, dtype=np.int64) + shift
        self.ranges.note(f"{name}.logit", _maxabs(r), shift,
                         "router logit against the sigmoid table's half-domain")
        assert idx.min() >= 0 and idx.max() < len(TY), "router logit outside the sigmoid table"
        return TY[idx]

    # the public tables, built on first use
    @property
    def T_A(self):
        if not hasattr(self, "_T_A"):
            import claims
            cfg = claims.SoftmaxConfig(B=1, M=1, s_x=self.S, s_c=self.S, s_y=self.S,
                                       delta=1, Z_max=self.k["z_max"], saturate=True)
            self._T_A = np.asarray(claims._softmax_exp_tables(cfg)[0], dtype=np.int64)
        return self._T_A

    @property
    def silu_cfg(self):
        from claims import SILU_14BIT
        assert SILU_14BIT.s_x == self.S
        return SILU_14BIT

    @property
    def silu_TY(self):
        if not hasattr(self, "_silu_TY"):
            from claims import silu_tpos_tneg
            tp, tn = silu_tpos_tneg(self.silu_cfg)
            self._silu_TY = np.asarray([v - P if v > P_HALF else v for v in tp + tn],
                                       dtype=np.int64)
        return self._silu_TY

    @property
    def sigmoid(self):
        if not hasattr(self, "_sig"):
            _, ty, shift = sigmoid_table(self.S, self.k["S_sel"], self.k["sig_bits"])
            self._sig = (np.asarray(ty, dtype=np.int64), shift)
        return self._sig


def _row_energy(x):
    """Exact Σ x² per row, as Python ints."""
    m = _maxabs(x)
    if m * m * x.shape[1] < (1 << 62):
        return [int(v) for v in (x * x).sum(axis=1)]
    return [sum(int(v) * int(v) for v in row) for row in x]


def rope_tables(cfg, *, T, offset, yarn=True, S=K2_INT["S"]):
    """The prover's integer cos/sin tables (T, d_rope/2), signed."""
    from claims import RoPEConfig, _rope_cos_sin
    kw = dict(scale_factor=cfg["factor"], original_max_pos=cfg["original_max_pos"],
              yarn=True, yarn_beta_fast=cfg["beta_fast"], yarn_beta_slow=cfg["beta_slow"],
              yarn_mscale=cfg["mscale"], yarn_mscale_all_dim=cfg["mscale_all_dim"]) if yarn else {}
    rc = RoPEConfig(SEQ=T, d_h=cfg["d_rope"], s_x=S, base=cfg["base"],
                    position_offset=offset, **kw)
    c, s = _rope_cos_sin(rc)
    sg = lambda v: np.asarray([x - P if x > P_HALF else x for x in v], dtype=np.int64)
    half = cfg["d_rope"] // 2
    return sg(c).reshape(T, half), sg(s).reshape(T, half)


# ---- the model --------------------------------------------------------------------

def head_interleave(a, b, *, T, H, w1, w2, shared):
    a = np.asarray(a, dtype=np.int64).reshape(T, H, w1)
    b = np.asarray(b, dtype=np.int64)
    b = np.broadcast_to(b.reshape(T, 1, w2), (T, H, w2)) if shared else b.reshape(T, H, w2)
    return np.concatenate([a, b], axis=2).reshape(T, H * (w1 + w2))


def mla(op: Int, x, w, cfg, *, T, offset, yarn, p, out):
    """One latent-attention block (analysis/mla-attention-design.md §4.2) and
    the post-attention norm. Returns (r1, n2g)."""
    H, dn, dr, dv = cfg["H"], cfg["d_nope"], cfg["d_rope"], cfg["d_v"]
    n1g = op.gain(op.rmsnorm(x, f"{p}.n1"), w["g_attn"], f"{p}.n1g")
    aqg = op.gain(op.rmsnorm(op.matmul(n1g, w["W_qa"], f"{p}.qa"), f"{p}.aq"),
                  w["g_q_a"], f"{p}.aqg")
    ckvg = op.gain(op.rmsnorm(op.matmul(n1g, w["W_ckv"], f"{p}.ckv_raw"), f"{p}.ckv"),
                   w["g_kv_a"], f"{p}.ckvg")
    cos, sin = rope_tables(cfg, T=T, offset=offset, yarn=yarn, S=op.S)
    q_nope = op.matmul(aqg, w["W_q_nope"], f"{p}.q_nope")
    q_pe = op.rope(op.matmul(aqg, w["W_q_pe"], f"{p}.q_pe_raw"), cos, sin,
                   T=T, heads=H, d_h=dr, name=f"{p}.q_pe")
    k_pe = op.rope(op.matmul(n1g, w["W_k_pe"], f"{p}.k_pe_raw"), cos, sin,
                   T=T, heads=1, d_h=dr, name=f"{p}.k_pe")
    k_nope = op.matmul(ckvg, w["W_k_nope"], f"{p}.k_nope")
    v = op.matmul(ckvg, w["W_v"], f"{p}.v")
    query = head_interleave(q_nope, q_pe, T=T, H=H, w1=dn, w2=dr, shared=False)
    key = head_interleave(k_nope, k_pe, T=T, H=H, w1=dn, w2=dr, shared=True)
    dh = dn + dr
    r = log2_ratio(op.S * op.S, op.S)
    raw = mm(query.reshape(T, H, dh).transpose(1, 0, 2),
             key.reshape(T, H, dh).transpose(1, 2, 0))                     # (H, T, T)
    sc = op.rescale(raw.transpose(1, 0, 2).reshape(T, H * T), r, f"{p}.scores")
    sm = op.softmax(sc, T=T, H=H, name=f"{p}.sm")
    raw = mm(sm.reshape(T, H, T).transpose(1, 0, 2),
             v.reshape(T, H, dv).transpose(1, 0, 2))                       # (H, T, dv)
    att = op.rescale(raw.transpose(1, 0, 2).reshape(T, H * dv), r, f"{p}.att")
    r1 = np.asarray(x, dtype=np.int64) + op.matmul(att, w["W_o"], f"{p}.proj")
    n2g = op.gain(op.rmsnorm(r1, f"{p}.n2"), w["g_ffn"], f"{p}.n2g")
    out.update({f"{p}.n1g": n1g, f"{p}.aqg": aqg, f"{p}.ckvg": ckvg, f"{p}.q_pe": q_pe,
                f"{p}.k_pe": k_pe, f"{p}.query": query, f"{p}.key": key, f"{p}.v": v,
                f"{p}.scores": sc, f"{p}.sm": sm, f"{p}.att": att, f"{p}.r1": r1,
                f"{p}.n2g": n2g})
    return r1, n2g


def ffn(op: Int, x, wg, wu, wd, name):
    g = op.matmul(x, wg, f"{name}.g")
    u = op.matmul(x, wu, f"{name}.u")
    return op.matmul(op.hadamard(op.silu(g), u, f"{name}.h"), wd, f"{name}.out")


def topk_width(s_sel: int, bias, E: int) -> int:
    """The bound route_topk takes: any difference of two tiebroken quantities
    2^L·(s + b) + (E − 1 − e), scores in [1, s_sel], is below 2^width."""
    b = np.asarray(bias, dtype=np.int64)
    span = (s_sel - 1) + int(b.max()) - int(b.min())
    return ((span << l_bits(E)) + E).bit_length()


def moe(op: Int, x, W, il, *, T, E, kk, p, out):
    """Router, sigmoid lookup, top-k, slot experts (grouped by expert, each
    loaded once), the gate-weighted combine and its rescale; the shared
    expert separately. Slot-major rows i·T + t, as topk_moe_ffn lays them."""
    k = op.k
    r = op.matmul(x, W.router(il), f"{p}.router")
    s = op.lookup_sigmoid(r, f"{p}.router")
    b = np.asarray(W.bias(il), dtype=np.int64)
    routing = tref.routing_witness(s.tolist(), b.tolist(), kk)
    M, ss = tref.slot_witness(routing["m"], s.tolist(), kk)
    gw = np.asarray(tref.bracket_witness(ss, k["C"])["w"], dtype=np.int64)       # (T, k)
    sel = np.asarray([[M[i][t].index(1) for i in range(kk)] for t in range(T)])  # (T, k)
    g_all = h_all = D_all = None
    for e in sorted(set(sel.ravel().tolist())):
        rows = [(t, i) for t in range(T) for i in range(kk) if sel[t, i] == e]
        xe = x[[t for t, _ in rows]]
        g = op.matmul(xe, W.expert(il, "gate", e), f"{p}.g")
        u = op.matmul(xe, W.expert(il, "up", e), f"{p}.u")
        h = op.hadamard(op.silu(g), u, f"{p}.h")
        D = mm(h, W.expert(il, "down", e))
        op.ranges.note(f"{p}.D", _maxabs(D), P_HALF + 1,
                       "routed down output, field representative")
        if g_all is None:
            g_all = np.zeros((kk * T, g.shape[1]), dtype=np.int64)
            h_all = np.zeros_like(g_all)
            D_all = np.zeros((kk * T, D.shape[1]), dtype=object)
        idx = [i * T + t for t, i in rows]
        g_all[idx], h_all[idx], D_all[idx] = g, h, D
    D_all = _narrow(D_all)
    y_raw = np.zeros(D_all.shape[1], dtype=object)[None, :].repeat(T, axis=0)
    for i in range(kk):
        y_raw = y_raw + mul(gw[:, i:i + 1], D_all[i * T:(i + 1) * T]).astype(object)
    y_raw = _narrow(y_raw)
    y = op.rescale(y_raw, log2_ratio(k["S_w"] * op.S * op.S, op.S), f"{p}.y")
    shw = W.shared(il)
    sh = ffn(op, x, shw["W_gate"], shw["W_up"], shw["W_down"], f"{p}.sh")
    width = topk_width(k["S_sel"], b, E)
    op.ranges.note(f"{p}.topk_width", width, 64,
                   f"q̃ difference width from the bias span {int(b.max()) - int(b.min())} at S_sel")
    out.update({f"{p}.router": r, f"{p}.s": s, f"{p}.mask": np.asarray(routing["m"]),
                f"{p}.gw": gw, f"{p}.g": g_all, f"{p}.h": h_all, f"{p}.D": D_all,
                f"{p}.y_raw": y_raw, f"{p}.y": y, f"{p}.sh": sh})
    return y, sh


def forward(ids, W, cfg, *, layers, offset=0, yarn=True, k=K2_INT, kk=8,
            head_chunk=16384, ranges=None):
    """Token ids to logits through `layers` layers. `W` gives the integer
    weights (k2_loader.GgufWeights(ints=True), or a test's toy provider).
    Returns ({name: int array}, Ranges)."""
    op = Int(k, ranges)
    T = len(ids)
    out = {}
    x = np.asarray(W.embed_rows(ids), dtype=np.int64)
    out["x0"] = x
    for il in range(layers):
        p = f"L{il}"
        r1, n2g = mla(op, x, W.attn(il), cfg, T=T, offset=offset, yarn=yarn, p=p, out=out)
        if W.kind(il) == "dense":
            dw = W.dense(il)
            f = ffn(op, n2g, dw["W_gate"], dw["W_up"], dw["W_down"], f"{p}.ffn")
            out[f"{p}.ffn"] = f
            x = r1 + f
        else:
            y, sh = moe(op, n2g, W, il, T=T, E=W.n_experts(il), kk=kk, p=p, out=out)
            x = (r1 + y) + sh
        out[f"{p}.out"] = x
    nfg = op.gain(op.rmsnorm(x, "final.n"), W.g_out(), "final.ng")
    out["final.ng"] = nfg
    logits = np.empty((T, W.V), dtype=np.int64)
    for v0 in range(0, W.V, head_chunk):
        v1 = min(W.V, v0 + head_chunk)
        logits[:, v0:v1] = op.matmul(nfg, W.head(v0, v1), "logits")
    out["logits"] = logits
    return out, op.ranges
