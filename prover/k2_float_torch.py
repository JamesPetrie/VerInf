"""Kimi K2's float64 reference (k2_float_reference) in torch on a device: the
full-depth p64 of analysis/k2-fidelity-study-design.md §5.3.

The weights come from the same provider as the NumPy reference
(k2_loader.GgufWeights(ints=False): gguf-py decodes on the CPU), so the two
start from identical decoded tensors. Layers are streamed, experts decoded
one at a time and freed, and only the outputs asked for are kept, moved to
the host as float64. The RoPE tables are the NumPy reference's, computed in
float64 and moved over. Everything else is the same composition in torch
float64; the two agree to float64 summation order
(tests/test_k2_float_torch.py)."""
import numpy as np
import torch

import k2_float_reference as kf
from k2_attention import rope_float_tables, softmax_scale

ROUTED_SCALING = kf.ROUTED_SCALING


def _t(a, dev):
    return torch.as_tensor(np.asarray(a, dtype=np.float64), device=dev)


def rmsnorm(x, g, eps):
    return x / torch.sqrt((x * x).mean(dim=-1, keepdim=True) + eps) * g


def silu(x):
    return x / (1.0 + torch.exp(-x))


def sigmoid(x):
    return 1.0 / (1.0 + torch.exp(-x))


def ffn(x, wg, wu, wd):
    return (silu(x @ wg) * (x @ wu)) @ wd


def rope_tables(cfg, positions):
    """cos and sin (T, d_rope/2), exactly as composition_a_attention builds them."""
    return rope_float_tables(cfg, positions)


def attention(x, w, cfg, cos, sin):
    """composition_a_attention in torch: σ folded into the two query weights,
    half-split RoPE, the shared rotary key, causal softmax."""
    T = x.shape[0]
    H, dn, dr, dv = cfg["H"], cfg["d_nope"], cfg["d_rope"], cfg["d_v"]

    def rope(z):
        lo, hi = z[..., :dr // 2], z[..., dr // 2:]
        c, s = cos[:, None, :], sin[:, None, :]
        return torch.cat([lo * c - hi * s, lo * s + hi * c], dim=-1)
    sigma = softmax_scale(cfg)
    a_q = rmsnorm(x @ w["W_qa"], w["g_q_a"], cfg["rms_eps"]) if "W_qa" in w else x
    q_nope = (a_q @ (w["W_q_nope"] * sigma)).reshape(T, H, dn)
    q_pe = rope((a_q @ (w["W_q_pe"] * sigma)).reshape(T, H, dr))
    ckv = rmsnorm(x @ w["W_ckv"], w["g_kv_a"], cfg["rms_eps"])
    k_pe = rope((x @ w["W_k_pe"]).reshape(T, 1, dr))
    k_nope = (ckv @ w["W_k_nope"]).reshape(T, H, dn)
    v = (ckv @ w["W_v"]).reshape(T, H, dv)
    query = torch.cat([q_nope, q_pe], dim=-1)
    key = torch.cat([k_nope, k_pe.expand(T, H, dr)], dim=-1)
    scores = torch.einsum("thc,uhc->htu", query, key)
    mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
    scores = scores.masked_fill(~mask, float("-inf"))
    scores = scores - scores.max(dim=-1, keepdim=True).values
    e = torch.exp(scores)
    p = e / e.sum(dim=-1, keepdim=True)
    out = torch.einsum("htu,uhc->thc", p, v).reshape(T, H * dv)
    return out @ w["W_o"]


def select(scores, bias, k):
    """Top-k of scores + bias, lowest index on a tie (a stable descending
    sort keeps equal keys in index order), as kf.select."""
    q = scores + bias[None, :]
    order = torch.sort(q, dim=1, descending=True, stable=True).indices
    return order[:, :k], q


def moe(x, W, il, *, k, dev, out, p, scale=ROUTED_SCALING):
    scores = sigmoid(x @ _t(W.router(il), dev))
    bias = _t(W.bias(il), dev)
    top, _ = select(scores, bias, k)
    w = torch.gather(scores, 1, top)
    w = w / w.sum(dim=1, keepdim=True) * scale
    y = torch.zeros_like(x)
    for e in torch.unique(top).tolist():
        ts, slots = torch.nonzero(top == e, as_tuple=True)
        h = ffn(x[ts], _t(W.expert(il, "gate", e), dev), _t(W.expert(il, "up", e), dev),
                _t(W.expert(il, "down", e), dev))
        y.index_add_(0, ts, w[ts, slots][:, None] * h)
    sh = W.shared(il)
    shared = ffn(x, *(_t(sh[n], dev) for n in ("W_gate", "W_up", "W_down")))
    out.update({f"{p}.s": scores, f"{p}.top": torch.sort(top, dim=1).values, f"{p}.y": y,
                f"{p}.sh": shared})
    return y + shared


def forward(ids, W, cfg, *, layers, offset=0, k=8, head_chunk=16384, device="cpu", keep=None):
    """Token ids to logits in torch float64 on `device`. Returns {name: numpy
    float64 array} at the NumPy reference's names; `keep` (a set of names, or
    a predicate) limits what is copied back, None keeps everything."""
    dev = torch.device(device)
    want = (lambda n: True) if keep is None else (keep if callable(keep) else (lambda n: n in keep))
    eps = cfg["rms_eps"]
    T = len(ids)
    cos, sin = (_t(a, dev) for a in rope_tables(cfg, offset + np.arange(T)))
    res = {}

    def put(d):
        for n, v in d.items():
            if want(n):
                res[n] = v.detach().cpu().numpy() if torch.is_tensor(v) else v
    x = _t(W.embed_rows(ids), dev)
    put({"x0": x})
    for il in range(layers):
        p = f"L{il}"
        w = {n: _t(v, dev) for n, v in W.attn(il).items()}
        n1g = rmsnorm(x, w["g_attn"], eps)
        r1 = x + attention(n1g, w, cfg, cos, sin)
        n2g = rmsnorm(r1, w["g_ffn"], eps)
        step = {f"{p}.n1g": n1g, f"{p}.r1": r1, f"{p}.n2g": n2g}
        del w
        if W.kind(il) == "dense":
            d = W.dense(il)
            f = ffn(n2g, *(_t(d[n], dev) for n in ("W_gate", "W_up", "W_down")))
            step[f"{p}.ffn"] = f
            x = r1 + f
        else:
            x = r1 + moe(n2g, W, il, k=k, dev=dev, out=step, p=p,
                         scale=cfg.get("routed_scale", ROUTED_SCALING))
        step[f"{p}.out"] = x
        put(step)
    nfg = rmsnorm(x, _t(W.g_out(), dev), eps)
    put({"final.ng": nfg})
    if want("logits"):
        logits = np.empty((T, W.V))
        for v0 in range(0, W.V, head_chunk):
            v1 = min(W.V, v0 + head_chunk)
            logits[:, v0:v1] = (nfg @ _t(W.head(v0, v1), dev)).cpu().numpy()
        res["logits"] = logits
    return res
