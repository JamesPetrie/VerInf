"""Kimi K2 precision study (CPU): which scale change could close the fidelity
gap session 10 measured (logits 5.1–5.4% off the float pass at the validated
2^12 scales), with weight precision and activation precision separated and
router selection kept at 2^16 (analysis/b200-session-10-archive.md).

Each configuration is an integer model of the two layers. The implementable
ones use only what the prover's claims already take:
  - matmul and hadamard: any power-of-two (s_a, s_b, s_out);
  - RMSNorm: a coarser input through s_in, its internal scale kept at
    s_norm = 2^12, its output at the activation scale through s_out
    (s_norm^2 >= S_act);
  - SiLU and softmax: their 2^12 tables, the input through s_in;
  - RoPE: its tables at the activation scale;
  - the router: logits rescaled to 2^12 for the sigmoid table, scores at
    S_sel = 2^16.
A configuration marked hypothetical needs a claim changed. At s_norm = 2^14
the RMSNorm bracket's inverse window needs 24 bits against the 22 its limb
carries allow (RmsNormConfig.y_width), so the bracket would need revising.
Raising ε is not studied: it changes the modeled normalization.

The baseline configuration reproduces k2_int_reference.forward exactly
(tests/test_k2_precision_study.py), so every difference measured here is the
scale change's. k2_int_reference stays the validated model; this module only
measures."""
import argparse
import json
import math
import sys
import time

import numpy as np

import k2_float_reference as kf
import k2_int_reference as ki
import topk_reference as tref
from k2_attention import K2_MLA, SIGMA_FOLDED, softmax_scale
from k2_loader import K2_INT, S_SEL, sigmoid_table, to_ints

# exponents: S_act, S_wt, s_norm (2^n); "hypothetical" names the claim that would change
CONFIGS = {
    "baseline": dict(S_act=12, S_wt=12, s_norm=12),
    "w16": dict(S_act=12, S_wt=16, s_norm=12),
    "a16": dict(S_act=16, S_wt=12, s_norm=12),
    "a14w14": dict(S_act=14, S_wt=14, s_norm=12),
    "a16w16": dict(S_act=16, S_wt=16, s_norm=12),
    "a16w16n14": dict(S_act=16, S_wt=16, s_norm=14,
                      hypothetical="RMSNorm bracket at an internal 2^14 (inverse window 24 bits)"),
}
TABLE = 12          # SiLU and softmax internal scale, router-logit scale (2^12)


def mm(a, b):
    """Exact integer matmul through float64 BLAS: when a product could pass
    2^53, `a` is split into base-2^k digits whose products cannot."""
    a = np.asarray(a, dtype=np.int64); b = np.asarray(b, dtype=np.int64)
    K = a.shape[-1]
    mb = ki._maxabs(b) * max(K, 1)
    if ki._maxabs(a) * mb < (1 << 53):
        return np.rint(np.matmul(a.astype(np.float64), b.astype(np.float64))).astype(np.int64)
    k = 52 - mb.bit_length()
    assert k >= 8, "weights too wide for the digit split"
    bf = b.astype(np.float64)
    rest, shift, acc = a, 0, None
    while True:
        last = bool(((rest == 0) | (rest == -1)).all())
        digit = rest if last else rest & ((1 << k) - 1)
        part = np.rint(np.matmul(digit.astype(np.float64), bf)).astype(np.int64).astype(object)
        acc = (part << shift) if acc is None else acc + (part << shift)
        if last:
            break
        rest, shift = rest >> k, shift + k
    return ki._narrow(acc)


class Mixed:
    """The integer ops at mixed scales; ranges recorded as in k2_int_reference
    (rescaled outputs against 2^31, to read the width each would need)."""

    def __init__(self, prec, k=K2_INT):
        self.k = k
        self.A, self.Wt = 1 << prec["S_act"], 1 << prec["S_wt"]
        self.N, self.Tb = 1 << prec["s_norm"], 1 << TABLE
        self.ranges = ki.Ranges()
        # the tables of the 2^12 model (SiLU, softmax exp, sigmoid)
        self.base = ki.Int(k, self.ranges)

    def rescale(self, full, r, name):
        self.ranges.note(f"{name}.full", ki._maxabs(full), ki.P_HALF + 1, "field representative")
        out = ki.floor_shift(full, r)
        self.ranges.note(f"{name}.out", ki._maxabs(out), 1 << 31, "rescaled output (width probe)")
        return np.asarray(out, dtype=np.int64)

    def matmul(self, a, w, name, s_a, s_b, s_out):
        return self.rescale(mm(a, w), ki.log2_ratio(s_a * s_b, s_out), name)

    def mul(self, a, b, name, s_a, s_b, s_out):
        return self.rescale(ki.mul(a, b), ki.log2_ratio(s_a * s_b, s_out), name)

    def coarsen(self, x, s_in, s):
        """A claim's input rescale x_in = (s_in/s)·x + x_low: x = ⌊x_in / (s_in/s)⌋."""
        return np.asarray(x, dtype=np.int64) >> ki.log2_ratio(s_in, s)

    def rmsnorm(self, x, name):
        """Input at S_act, the bracket at s_norm, the output at S_act."""
        A, N = self.A, self.N
        x = self.coarsen(x, A, N) if A > N else np.asarray(x, dtype=np.int64)
        B, d = x.shape
        eps_int = round(self.k["eps"] * N * N)
        magic = d * N ** 4
        st = [v + d * eps_int for v in ki._row_energy(x)]
        ys = []
        for s_tot in st:
            y = max(1, math.isqrt(magic // s_tot))
            while y * y * s_tot < magic:
                y += 1
            while y > 1 and (y - 1) * (y - 1) * s_tot >= magic:
                y -= 1
            ys.append(y)
        from claims import RMS_LIMB_W, RMS_S_CAP_BITS, RmsNormConfig
        y_max = RmsNormConfig(B=1, d=d, s=N, eps_int=eps_int).y_max
        y_width = max(1, (y_max - 1).bit_length())
        self.ranges.note(f"{name}.S_tot", max(st), 1 << RMS_S_CAP_BITS, f"row energy (d = {d})")
        self.ranges.note(f"{name}.y_width", 2 * y_width + RMS_LIMB_W, 64,
                         "2·y_width + LIMB_W against 63: the bracket's carries")
        return self.rescale(ki.mul(x, np.asarray(ys, dtype=np.int64)[:, None]),
                            ki.log2_ratio(N * N, A), name)

    def gain(self, x, g, name):
        return self.mul(x, np.asarray(g, dtype=np.int64)[None, :], name, self.A, self.Wt, self.A)

    def rope(self, x, cfg, *, T, heads, offset, name):
        cos, sin = ki.rope_tables(cfg, T=T, offset=offset, S=self.A)
        dr = cfg["d_rope"]
        z = np.asarray(x, dtype=np.int64).reshape(T, heads, dr)
        half = dr // 2
        lo, hi = z[..., :half], z[..., half:]
        c, s = cos[:, None, :], sin[:, None, :]
        rot = np.concatenate([ki.mul(c, lo) - ki.mul(s, hi), ki.mul(s, lo) + ki.mul(c, hi)], axis=-1)
        return self.rescale(rot.reshape(T, heads * dr), ki.log2_ratio(self.A * self.A, self.A), name)

    def softmax(self, sc, *, T, H, name):
        x = self.coarsen(sc, self.A, self.Tb) if self.A > self.Tb else sc
        return self.base.softmax(x, T=T, H=H, name=name)          # output at 2^12

    def silu(self, g):
        x = self.coarsen(g, self.A, self.Tb) if self.A > self.Tb else g
        return self.base.silu(x)                                   # output at 2^12


def quantized(Wf, il, cfg, prec):
    """One layer's attention variables at the configuration's weight scale,
    σ folded into the query weights as the loader folds it."""
    S = 1 << prec["S_wt"]
    sigma = softmax_scale(cfg)
    return {k: to_ints(v, S, sigma if k in SIGMA_FOLDED else 1.0) for k, v in Wf.attn(il).items()}


def forward(ids, Wf, cfg, prec, *, layers=2, offset=0, kk=8, head_chunk=16384):
    """The two-layer integer forward at one configuration, from float weights
    (`Wf`, k2_loader.GgufWeights(ints=False), cached by the caller)."""
    op = Mixed(prec)
    A, Wt, Tb = op.A, op.Wt, op.Tb
    q = lambda w: to_ints(w, Wt)
    T = len(ids)
    out = {}
    x = to_ints(Wf.embed_rows(ids), A)
    out["x0"] = x
    H, dn, dr, dv = cfg["H"], cfg["d_nope"], cfg["d_rope"], cfg["d_v"]
    for il in range(layers):
        p = f"L{il}"
        w = quantized(Wf, il, cfg, prec)
        n1g = op.gain(op.rmsnorm(x, f"{p}.n1"), w["g_attn"], f"{p}.n1g")
        aqg = op.gain(op.rmsnorm(op.matmul(n1g, w["W_qa"], f"{p}.qa", A, Wt, A), f"{p}.aq"),
                      w["g_q_a"], f"{p}.aqg")
        ckvg = op.gain(op.rmsnorm(op.matmul(n1g, w["W_ckv"], f"{p}.ckv_raw", A, Wt, A), f"{p}.ckv"),
                       w["g_kv_a"], f"{p}.ckvg")
        q_nope = op.matmul(aqg, w["W_q_nope"], f"{p}.q_nope", A, Wt, A)
        q_pe = op.rope(op.matmul(aqg, w["W_q_pe"], f"{p}.q_pe_raw", A, Wt, A), cfg,
                       T=T, heads=H, offset=offset, name=f"{p}.q_pe")
        k_pe = op.rope(op.matmul(n1g, w["W_k_pe"], f"{p}.k_pe_raw", A, Wt, A), cfg,
                       T=T, heads=1, offset=offset, name=f"{p}.k_pe")
        k_nope = op.matmul(ckvg, w["W_k_nope"], f"{p}.k_nope", A, Wt, A)
        v = op.matmul(ckvg, w["W_v"], f"{p}.v", A, Wt, A)
        query = ki.head_interleave(q_nope, q_pe, T=T, H=H, w1=dn, w2=dr, shared=False)
        key = ki.head_interleave(k_nope, k_pe, T=T, H=H, w1=dn, w2=dr, shared=True)
        dh = dn + dr
        raw = mm(query.reshape(T, H, dh).transpose(1, 0, 2), key.reshape(T, H, dh).transpose(1, 2, 0))
        sc = op.rescale(raw.transpose(1, 0, 2).reshape(T, H * T), ki.log2_ratio(A * A, A), f"{p}.scores")
        sm = op.softmax(sc, T=T, H=H, name=f"{p}.sm")                       # at 2^12
        raw = mm(sm.reshape(T, H, T).transpose(1, 0, 2), v.reshape(T, H, dv).transpose(1, 0, 2))
        att = op.rescale(raw.transpose(1, 0, 2).reshape(T, H * dv), ki.log2_ratio(Tb * A, A), f"{p}.att")
        r1 = x + op.matmul(att, w["W_o"], f"{p}.proj", A, Wt, A)
        n2g = op.gain(op.rmsnorm(r1, f"{p}.n2"), w["g_ffn"], f"{p}.n2g")
        out.update({f"{p}.n1g": n1g, f"{p}.r1": r1, f"{p}.n2g": n2g})

        def ffn(xin, wg, wu, wd, name):
            g = op.matmul(xin, q(wg), f"{name}.g", A, Wt, A)
            u = op.matmul(xin, q(wu), f"{name}.u", A, Wt, A)
            h = op.mul(op.silu(g), u, f"{name}.h", Tb, A, A)
            return op.matmul(h, q(wd), f"{name}.out", A, Wt, A)
        if Wf.kind(il) == "dense":
            dw = Wf.dense(il)
            f = ffn(n2g, dw["W_gate"], dw["W_up"], dw["W_down"], f"{p}.ffn")
            out[f"{p}.ffn"] = f
            x = r1 + f
        else:
            y, sh = _moe(op, n2g, Wf, il, T=T, kk=kk, p=p, out=out, ffn=ffn)
            x = (r1 + y) + sh
        out[f"{p}.out"] = x
    nfg = op.gain(op.rmsnorm(x, "final.n"), q(Wf.g_out()), "final.ng")
    out["final.ng"] = nfg
    logits = np.empty((T, Wf.V), dtype=np.int64)
    for v0 in range(0, Wf.V, head_chunk):
        v1 = min(Wf.V, v0 + head_chunk)
        logits[:, v0:v1] = op.matmul(nfg, q(Wf.head(v0, v1)), "logits", A, Wt, A)
    out["logits"] = logits
    return out, op.ranges


def _moe(op, x, Wf, il, *, T, kk, p, out, ffn):
    """Router at 2^12 for the sigmoid table, scores at S_sel; the top-k
    reference; slot experts at the configuration's scales; the combine."""
    A, Wt, k = op.A, op.Wt, op.k
    r = op.matmul(x, to_ints(Wf.router(il), Wt), f"{p}.router", A, Wt, op.Tb)
    s = op.base.lookup_sigmoid(r, f"{p}.router")
    b = to_ints(Wf.bias(il), S_SEL)
    routing = tref.routing_witness(s.tolist(), b.tolist(), kk)
    M, ss = tref.slot_witness(routing["m"], s.tolist(), kk)
    gw = np.asarray(tref.bracket_witness(ss, k["C"])["w"], dtype=np.int64)
    sel = np.asarray([[M[i][t].index(1) for i in range(kk)] for t in range(T)])
    y_raw = np.zeros((T, x.shape[1]), dtype=object)
    for e in sorted(set(sel.ravel().tolist())):
        rows = [(t, i) for t in range(T) for i in range(kk) if sel[t, i] == e]
        xe = x[[t for t, _ in rows]]
        g = op.matmul(xe, to_ints(Wf.expert(il, "gate", e), Wt), f"{p}.g", A, Wt, A)
        u = op.matmul(xe, to_ints(Wf.expert(il, "up", e), Wt), f"{p}.u", A, Wt, A)
        h = op.mul(op.silu(g), u, f"{p}.h", op.Tb, A, A)
        D = mm(h, to_ints(Wf.expert(il, "down", e), Wt))                       # at A·Wt
        op.ranges.note(f"{p}.D", ki._maxabs(D), ki.P_HALF + 1, "routed down output, field representative")
        for j, (t, i) in enumerate(rows):
            y_raw[t] += np.asarray(D[j], dtype=object) * int(gw[t, i])
    y_raw = ki._narrow(y_raw)
    y = op.rescale(y_raw, ki.log2_ratio(k["S_w"] * A * Wt, A), f"{p}.y")
    shw = Wf.shared(il)
    sh = ffn(x, shw["W_gate"], shw["W_up"], shw["W_down"], f"{p}.sh")
    out.update({f"{p}.s": s, f"{p}.mask": np.asarray(routing["m"]), f"{p}.y": y, f"{p}.sh": sh})
    return y, sh


class CachedFloats:
    """GgufWeights(ints=False) with every tensor asked for kept in memory,
    so the configurations share one decode."""

    def __init__(self, Wf):
        self.Wf, self.V, self._c = Wf, Wf.V, {}

    def _get(self, key, fn):
        if key not in self._c:
            self._c[key] = fn()
        return self._c[key]

    def kind(self, il):
        return self.Wf.kind(il)

    def n_experts(self, il):
        return self.Wf.n_experts(il)

    def attn(self, il):
        return self._get(("attn", il), lambda: self.Wf.attn(il))

    def dense(self, il):
        return self._get(("dense", il), lambda: self.Wf.dense(il))

    def shared(self, il):
        return self._get(("shared", il), lambda: self.Wf.shared(il))

    def router(self, il):
        return self._get(("router", il), lambda: self.Wf.router(il))

    def bias(self, il):
        return self._get(("bias", il), lambda: self.Wf.bias(il))

    def expert(self, il, kind, e):
        return self._get(("expert", il, kind, e),
                         lambda: self.Wf.expert(il, kind, e).astype(np.float32))

    def embed_rows(self, ids):
        return self._get(("embed", tuple(ids)), lambda: self.Wf.embed_rows(ids))

    def g_out(self):
        return self._get(("g_out",), self.Wf.g_out)

    def head(self, v0, v1):
        return self._get(("head", v0, v1), lambda: self.Wf.head(v0, v1).astype(np.float32))


def study(gguf, ids, *, configs=tuple(CONFIGS), cfg=K2_MLA, layers=2, offset=0, kk=8,
          log=print):
    """Fidelity, routing and ranges for each configuration against one float pass."""
    from k2_loader import GgufWeights
    Wf = CachedFloats(GgufWeights(gguf, ints=False, cfg=cfg))
    t0 = time.time()
    fo = kf.forward(ids, Wf, cfg, layers=layers, offset=offset, k=kk)
    log(f"float pass {time.time() - t0:.1f}s")
    rows = {}
    for name in configs:
        prec = CONFIGS[name]
        t0 = time.time()
        io, ranges = forward(ids, Wf, cfg, prec, layers=layers, offset=offset, kk=kk)
        fid = kf.fidelity(io, fo, 1 << prec["S_act"], S_SEL)
        rd = {il: kf.routing_differences(io, fo, Wf, il, k=kk, S=1 << prec["S_act"])
              for il in range(layers) if Wf.kind(il) == "moe"}
        rg = ranges.as_dict()
        width = max(r["bits"] for n, r in rg.items() if n.endswith(".out")) + 1
        bracket = max(r["observed"] for n, r in rg.items() if n.endswith(".y_width"))
        rows[name] = dict(prec=prec, s=time.time() - t0, fidelity=fid, routing=rd,
                          output_width_needed=width, rmsnorm_bracket_bits=bracket,
                          rmsnorm_bracket_ok=bracket <= 63,
                          field_ok=all(r["ok"] for n, r in rg.items() if n.endswith((".full", ".D"))),
                          ranges=rg)
        e2e = rd[max(rd)]["end_to_end"] if rd else {}
        log(f"{name:10s} logits rel {fid['logits']['rel_l2']:.4f}  top-1 {fid['logits.top1_agree']:.3f}  "
            f"routing e2e {e2e.get('tokens_differing')}/{e2e.get('tokens')}  "
            f"width {width}  bracket {bracket}/63  ({time.time() - t0:.0f}s)"
            + (f"  [hypothetical: {prec['hypothetical']}]" if "hypothetical" in prec else ""))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--from-gguf", required=True)
    ap.add_argument("--prompt-n", type=int, default=50)
    ap.add_argument("--cont-n", type=int, default=50)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--record", required=True)
    a = ap.parse_args(argv)
    from k2_loader import by_name
    V = int(by_name(a.from_gguf)["token_embd.weight"].n_elements) // K2_MLA["d"]
    rng = np.random.default_rng(11)                     # demo_k2's synthetic ids
    ids = rng.integers(0, V, a.prompt_n).tolist() + rng.integers(0, V, a.cont_n).tolist()
    rows = study(a.from_gguf, ids, configs=a.configs.split(","), offset=a.offset)
    with open(a.record, "x") as fh:
        json.dump(dict(args=vars(a), configs=rows), fh, indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
