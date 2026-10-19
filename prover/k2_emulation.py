"""The rounding emulations of analysis/k2-fidelity-study-design.md §3.2 (K2,
two layers, diagnostic): K2's forward under a precision policy that names
every boundary in the note's tables, starting from the same decoded float64
weights as p64.

A matmul is a float32 GEMM on rounded inputs, which emulates fp32
accumulation; its summation order and blocking differ from any GPU kernel,
so this is a precision policy, not a reproduction of a kernel. Rounding is
to nearest even: bf16 at the bit level from float32, E4M3 on its grid (3
mantissa bits, bias 7, subnormals to 2^-9, saturating at ±448).

Four arms: bf16 or FP8 (E4M3 activations per 1×128 group at the inputs of
the linears that are FP8 in K2's release), each with the router in fp32
(R-fp32) or with its logits and scores in bf16 (R-bf16). Attention, the
router and the LM head never take FP8 inputs. The query scale σ multiplies
the scores in fp32, as a deployment does, rather than being folded into the
weights.

`forward` returns the NumPy reference's names (k2_float_reference.forward),
and records which boundaries it crossed: the coverage test requires that set
to equal BOUNDARIES, the note's table rows."""
import math

import numpy as np

import k2_float_reference as kf
from k2_attention import softmax_scale

ROUTED_SCALING = kf.ROUTED_SCALING
E4M3_MAX = 448.0
GROUP = 128

# the note's §3.2 rows, in composition A order
BOUNDARIES = ("embedding", "rmsnorm", "linear", "rope", "attention", "silu_mul", "residual",
              "router", "gate_combine", "lm_head")


# ---- rounding ---------------------------------------------------------------------

def bf16(x):
    """Round to bfloat16 (nearest even, from float32), returned as float64."""
    a = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
    u = a.view(np.uint32).astype(np.uint64)
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return u.astype(np.uint32).view(np.float32).astype(np.float64)


def e4m3(x):
    """Round to float8 E4M3 (nearest even, saturating at ±448), as float64."""
    x = np.asarray(x, dtype=np.float64)
    a = np.abs(x)
    with np.errstate(divide="ignore"):
        e = np.floor(np.log2(np.where(a > 0, a, 1.0)))
    step = np.exp2(np.maximum(e, -6.0) - 3.0)            # subnormals: 2^-9 below 2^-6
    q = np.minimum(np.rint(a / step) * step, E4M3_MAX)
    return np.where(a > 0, np.sign(x) * q, 0.0)


def fp8_groups(x, group=GROUP):
    """Dynamic E4M3 per 1×`group` block along the last (contraction) axis,
    scale = amax/448, returned dequantized: what the FP8 GEMM multiplies."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    n = x.shape[-1]
    for g0 in range(0, n, group):
        blk = x[..., g0:g0 + group]
        amax = np.abs(blk).max(axis=-1, keepdims=True)
        scale = np.where(amax > 0, amax / E4M3_MAX, 1.0)
        out[..., g0:g0 + group] = e4m3(blk / scale) * scale
    return out


def f32(x):
    return np.asarray(x, dtype=np.float32)


# ---- policies ---------------------------------------------------------------------

class Policy:
    """`activations`: "bf16" or "fp8" (the linears' inputs); `router`: "fp32"
    or "bf16". Records the boundaries the forward crosses."""

    def __init__(self, activations="bf16", router="fp32"):
        assert activations in ("bf16", "fp8") and router in ("fp32", "bf16")
        self.activations, self.router = activations, router
        self.crossed = set()

    @property
    def name(self):
        return f"{self.activations}/R-{self.router}"

    def mark(self, boundary):
        assert boundary in BOUNDARIES, boundary
        self.crossed.add(boundary)

    def linear(self, x, W):
        """bf16 weight; input bf16, or E4M3 per group for FP8; fp32 GEMM; bf16 out."""
        self.mark("linear")
        xi = fp8_groups(x) if self.activations == "fp8" else bf16(x)
        return bf16(f32(xi) @ f32(bf16(W)))

    def rmsnorm(self, x, g, eps):
        """bf16 in; fp32 mean square, inverse root and gain; bf16 out."""
        self.mark("rmsnorm")
        x32 = f32(bf16(x))
        ms = (x32 * x32).mean(axis=-1, keepdims=True, dtype=np.float32)
        inv = np.float32(1.0) / np.sqrt(ms + np.float32(eps))
        return bf16(x32 * inv * f32(g))

    def residual(self, a, b):
        self.mark("residual")
        return bf16(f32(a) + f32(b))


ARMS = {f"{a}/R-{r}": (a, r) for a in ("bf16", "fp8") for r in ("fp32", "bf16")}


# ---- the forward --------------------------------------------------------------------

def _rope_f32(cfg, positions):
    """cos, sin (T, d_rope/2) in float32 with float32 angles, as deployments
    compute them: the reference's YaRN inverse frequencies and mscale, cast."""
    from claims import RoPEConfig, _rope_yarn_inv_freq, _rope_yarn_mscale
    from k2_attention import rope_config_kw
    dr = cfg["d_rope"]
    if cfg.get("yarn", True):
        rc = RoPEConfig(SEQ=1, d_h=dr, s_x=1, base=cfg["base"], **rope_config_kw(cfg))
        inv, m = f32(_rope_yarn_inv_freq(rc)), np.float32(_rope_yarn_mscale(rc))
    else:
        inv, m = f32(1.0 / (cfg["base"] ** (np.arange(0, dr, 2) / dr))), np.float32(1.0)
    theta = np.outer(f32(positions), inv).astype(np.float32)
    return np.cos(theta) * m, np.sin(theta) * m


def _attention(P, x, w, cfg, cos, sin):
    T = x.shape[0]
    H, dn, dr, dv = cfg["H"], cfg["d_nope"], cfg["d_rope"], cfg["d_v"]

    def rope(z):
        P.mark("rope")
        z = f32(z)
        lo, hi = z[..., :dr // 2], z[..., dr // 2:]
        c, s = cos[:, None, :], sin[:, None, :]
        return bf16(np.concatenate([lo * c - hi * s, lo * s + hi * c], axis=-1))
    a_q = P.rmsnorm(P.linear(x, w["W_qa"]), w["g_q_a"], cfg["rms_eps"]) if "W_qa" in w else x
    q_nope = P.linear(a_q, w["W_q_nope"]).reshape(T, H, dn)
    q_pe = rope(P.linear(a_q, w["W_q_pe"]).reshape(T, H, dr))
    ckv = P.rmsnorm(P.linear(x, w["W_ckv"]), w["g_kv_a"], cfg["rms_eps"])
    k_pe = rope(P.linear(x, w["W_k_pe"]).reshape(T, 1, dr))
    k_nope = P.linear(ckv, w["W_k_nope"]).reshape(T, H, dn)
    v = P.linear(ckv, w["W_v"]).reshape(T, H, dv)
    P.mark("attention")
    query = f32(np.concatenate([q_nope, q_pe], axis=-1))
    key = f32(np.concatenate([k_nope, np.broadcast_to(k_pe, (T, H, dr))], axis=-1))
    scores = np.einsum("thc,uhc->htu", query, key).astype(np.float32) * np.float32(softmax_scale(cfg))
    mask = np.tril(np.ones((T, T), dtype=bool))
    scores = np.where(mask, scores, np.float32(-np.inf))
    scores = scores - scores.max(axis=-1, keepdims=True)
    e = np.exp(scores, dtype=np.float32)
    p = bf16(e / e.sum(axis=-1, keepdims=True, dtype=np.float32))
    out = bf16(np.einsum("htu,uhc->thc", f32(p), f32(v)).astype(np.float32)).reshape(T, H * dv)
    return P.linear(out, w["W_o"])


def _ffn(P, x, wg, wu, wd):
    g, u = P.linear(x, wg), P.linear(x, wu)
    P.mark("silu_mul")
    g32 = f32(g)
    h = bf16(g32 / (np.float32(1.0) + np.exp(-g32)) * f32(u))
    return P.linear(h, wd)


def _moe(P, x, W, il, *, k, out, p, scale):
    P.mark("router")
    r = (f32(bf16(x)) @ f32(W.router(il))).astype(np.float32)          # F32 weight as stored
    if P.router == "bf16":
        r = f32(bf16(r))
    s = (np.float32(1.0) / (np.float32(1.0) + np.exp(-r))).astype(np.float32)
    if P.router == "bf16":
        s = f32(bf16(s))
    key = s.astype(np.float32) + f32(W.bias(il))
    order = np.lexsort((np.arange(key.shape[1])[None, :].repeat(key.shape[0], 0), -key), axis=1)
    top = order[:, :k]
    w = np.take_along_axis(s, top, axis=1).astype(np.float32)
    w = w / w.sum(axis=1, keepdims=True, dtype=np.float32) * np.float32(scale)
    P.mark("gate_combine")
    y = np.zeros(x.shape, dtype=np.float32)
    for e in sorted(set(top.ravel().tolist())):
        ts, slots = np.nonzero(top == e)
        h = _ffn(P, x[ts], W.expert(il, "gate", e), W.expert(il, "up", e), W.expert(il, "down", e))
        y[ts] += w[ts, slots][:, None] * f32(h)
    y = bf16(y)
    sh = W.shared(il)
    shared = _ffn(P, x, sh["W_gate"], sh["W_up"], sh["W_down"])
    out.update({f"{p}.s": s.astype(np.float64), f"{p}.top": np.sort(top, axis=1), f"{p}.y": y,
                f"{p}.sh": shared})
    return P.residual(y, shared)


def forward(ids, W, cfg, policy, *, layers, offset=0, k=8, head_chunk=16384):
    """Token ids to logits under `policy`, from the decoded float64 weights of
    `W` (k2_loader.GgufWeights(ints=False)); the NumPy reference's names."""
    P = policy
    eps = cfg["rms_eps"]
    T = len(ids)
    cos, sin = _rope_f32(cfg, offset + np.arange(T))
    out = {}
    P.mark("embedding")
    x = bf16(W.embed_rows(ids))
    out["x0"] = x
    for il in range(layers):
        p = f"L{il}"
        w = W.attn(il)
        n1g = P.rmsnorm(x, w["g_attn"], eps)
        r1 = P.residual(x, _attention(P, n1g, w, cfg, cos, sin))
        n2g = P.rmsnorm(r1, w["g_ffn"], eps)
        out.update({f"{p}.n1g": n1g, f"{p}.r1": r1, f"{p}.n2g": n2g})
        if W.kind(il) == "dense":
            d = W.dense(il)
            f = _ffn(P, n2g, d["W_gate"], d["W_up"], d["W_down"])
            out[f"{p}.ffn"] = f
            x = P.residual(r1, f)
        else:
            x = P.residual(r1, _moe(P, n2g, W, il, k=k, out=out, p=p,
                                    scale=cfg.get("routed_scale", ROUTED_SCALING)))
        out[f"{p}.out"] = x
    nfg = P.rmsnorm(x, W.g_out(), eps)
    out["final.ng"] = nfg
    P.mark("lm_head")
    logits = np.empty((T, W.V))
    xin = f32(bf16(nfg))
    for v0 in range(0, W.V, head_chunk):
        v1 = min(W.V, v0 + head_chunk)
        logits[:, v0:v1] = (xin @ f32(bf16(W.head(v0, v1)))).astype(np.float64)
    out["logits"] = logits
    return out
