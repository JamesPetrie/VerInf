"""Kimi K2's multi-head latent attention outside the prover: the GGUF weight
transforms and two float64 references (analysis/mla-attention-design.md §4.1, §6).

`mla_weights_from_gguf` turns one layer's GGUF attention tensors into the
variables composition A commits, in the (in, out) orientation tape.matmul
takes: splits, per-head reassembly of the key and value blocks, and the
de-interleave of the rotary rows (llama.cpp and the reference rotate
adjacent pairs; the prover's RoPE rotates half-split pairs). Index
operations only, so they apply alike to float tensors before quantization
and to field tensors after. The softmax scale σ is NOT applied here: the
loader folds it into W_q_nope and W_q_pe when it quantizes them
(`SIGMA_FOLDED`), as Maverick's loader folds 1/√128 into W_Q.

`reference_attention` ports DeepseekV3Attention.forward (Kimi K2's
modeling_deepseek.py at Hugging Face revision fd1984e2, :760-850; YaRN
:226-327) to float64 numpy, on Hugging Face-layout weights.
`composition_a_attention` computes the same layer the way the prover's claims
do (§4.2), in float64, on the transformed weights: split matmuls, half-split
RoPE from the prover's own YaRN frequencies, per-head assembly with the
shared rotary key, σ folded into the query. The two agree to rounding
(prover/tests/test_k2_attention.py). Shapes follow numpy's reading of the
GGUF, which reverses ggml's dimension order (gguf-py GGUFReader)."""
import math

import numpy as np

# Kimi K2 (config.json at Hugging Face revision fd1984e2)
K2_MLA = dict(d=7168, H=64, q_rank=1536, kv_rank=512, d_nope=128, d_rope=64, d_v=128,
              rms_eps=1e-6, base=50000.0, factor=32.0, original_max_pos=4096,
              beta_fast=1.0, beta_slow=1.0, mscale=1.0, mscale_all_dim=1.0)

SIGMA_FOLDED = ("W_q_nope", "W_q_pe")


def _yarn_get_mscale(scale, mscale):
    return 1.0 if scale <= 1 else 0.1 * mscale * math.log(scale) + 1.0


def softmax_scale(cfg) -> float:
    """σ = (d_nope + d_rope)^-1/2 · m², m = yarn_get_mscale(factor,
    mscale_all_dim) when mscale_all_dim is set (:690-696). K2: 0.130861."""
    s = (cfg["d_nope"] + cfg["d_rope"]) ** -0.5
    if cfg.get("mscale_all_dim"):
        m = _yarn_get_mscale(cfg["factor"], cfg["mscale_all_dim"])
        s = s * m * m
    return s


def deinterleave_rows(w: np.ndarray, d_rope: int) -> np.ndarray:
    """Rows in blocks of d_rope, each block's adjacent pairs (2i, 2i+1)
    reordered to half-split (i, i + d_rope/2): new row e·half + i takes old
    row 2i + e — the reference's view(d/2, 2).transpose, and the map of the
    Maverick loader's _unpermute_rows."""
    n = w.shape[0]
    assert n % d_rope == 0 and d_rope % 2 == 0
    idx = np.arange(n).reshape(-1, d_rope // 2, 2).transpose(0, 2, 1).reshape(-1)
    return w[idx]


def mla_weights_from_gguf(t: dict, cfg) -> dict:
    """One layer's committed attention weights from its GGUF tensors, keyed
    by the tensor name without `blk.N.` and `.weight`, as numpy arrays in
    gguf-py's (reversed-dimension) shapes:

      attn_q_a       (q_rank, d)              attn_q_a_norm (q_rank,)
      attn_q_b       (H·(d_nope+d_rope), q_rank)
      attn_kv_a_mqa  (kv_rank + d_rope, d)    attn_kv_a_norm (kv_rank,)
      attn_k_b       (H, kv_rank, d_nope)     attn_v_b (H, d_v, kv_rank)
        — or the legacy attn_kv_b (H·(d_nope+d_v), kv_rank)
      attn_output    (d, H·d_v)               attn_norm (d,)

    Returns {name: array} with every matrix (in, out)."""
    H, dn, dr, dv = cfg["H"], cfg["d_nope"], cfg["d_rope"], cfg["d_v"]
    qr, kr = cfg["q_rank"], cfg["kv_rank"]
    qb = t["attn_q_b"].reshape(H, dn + dr, qr)
    q_pe = deinterleave_rows(qb[:, dn:, :].reshape(H * dr, qr), dr)
    kva = t["attn_kv_a_mqa"]
    assert kva.shape[0] == kr + dr
    if "attn_k_b" in t:
        kb, vb = t["attn_k_b"], t["attn_v_b"]
        assert kb.shape == (H, kr, dn) and vb.shape == (H, dv, kr)
        w_k_nope = kb.transpose(1, 0, 2).reshape(kr, H * dn)
        w_v = vb.transpose(2, 0, 1).reshape(kr, H * dv)
    else:
        kvb = t["attn_kv_b"].reshape(H, dn + dv, kr)
        w_k_nope = kvb[:, :dn, :].transpose(2, 0, 1).reshape(kr, H * dn)
        w_v = kvb[:, dn:, :].transpose(2, 0, 1).reshape(kr, H * dv)
    return {
        "g_attn": t["attn_norm"], "g_q_a": t["attn_q_a_norm"], "g_kv_a": t["attn_kv_a_norm"],
        "W_qa": t["attn_q_a"].T,
        "W_q_nope": qb[:, :dn, :].reshape(H * dn, qr).T,
        "W_q_pe": q_pe.T,
        "W_ckv": kva[:kr].T,
        "W_k_pe": deinterleave_rows(kva[kr:], dr).T,
        "W_k_nope": w_k_nope,
        "W_v": w_v,
        "W_o": t["attn_output"].T,
    }


def gguf_from_hf(hf: dict, cfg) -> dict:
    """What llama.cpp's converter writes for one layer (conversion/deepseek.py
    at c2503049, :433-449: kv_b_proj split per head into k_b, transposed, and
    v_b), from Hugging Face weights (out, in), in gguf-py's shapes. With
    `legacy`, the single attn_kv_b of older conversions. For tests."""
    H, dn, dv, kr = cfg["H"], cfg["d_nope"], cfg["d_v"], cfg["kv_rank"]
    kvb = hf["kv_b_proj"].reshape(H, dn + dv, kr)
    return {
        "attn_q_a": hf["q_a_proj"], "attn_q_a_norm": hf["q_a_layernorm"],
        "attn_q_b": hf["q_b_proj"], "attn_kv_a_mqa": hf["kv_a_proj_with_mqa"],
        "attn_kv_a_norm": hf["kv_a_layernorm"],
        "attn_k_b": kvb[:, :dn, :].transpose(0, 2, 1).copy(),
        "attn_v_b": kvb[:, dn:, :].copy(),
        "attn_kv_b": hf["kv_b_proj"],
        "attn_output": hf["o_proj"], "attn_norm": hf.get("input_layernorm"),
    }


# ---- float64 references -------------------------------------------------------

def _rmsnorm(x, g, eps):
    return x / np.sqrt((x * x).mean(axis=-1, keepdims=True) + eps) * g


def _yarn_inv_freq(cfg):
    """The reference's YaRN inverse frequencies, in float64."""
    dim, base = cfg["d_rope"], cfg["base"]

    def corr(n):
        return (dim * math.log(cfg["original_max_pos"] / (n * 2 * math.pi))) / (2 * math.log(base))
    low = max(math.floor(corr(cfg["beta_fast"])), 0)
    high = min(math.ceil(corr(cfg["beta_slow"])), dim - 1)
    if low == high:
        high += 0.001
    ramp = np.clip((np.arange(dim // 2) - low) / (high - low), 0, 1)
    mask = 1.0 - ramp
    extra = 1.0 / (base ** (np.arange(0, dim, 2) / dim))
    inter = 1.0 / (cfg["factor"] * base ** (np.arange(0, dim, 2) / dim))
    return inter * (1 - mask) + extra * mask


def _causal_softmax(scores):
    T = scores.shape[-1]
    s = np.where(np.tril(np.ones((T, T), dtype=bool)), scores, -np.inf)
    s = s - s.max(axis=-1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis=-1, keepdims=True)


def reference_attention(x, hf, cfg, positions=None):
    """DeepseekV3Attention.forward without a cache, causal, in float64, on
    Hugging Face weights (out, in). x (T, d) is the normed hidden state."""
    T = x.shape[0]
    H, dn, dr, dv = cfg["H"], cfg["d_nope"], cfg["d_rope"], cfg["d_v"]
    pos = np.arange(T) if positions is None else np.asarray(positions)
    q = _rmsnorm(x @ hf["q_a_proj"].T, hf["q_a_layernorm"], cfg["rms_eps"]) @ hf["q_b_proj"].T
    q = q.reshape(T, H, dn + dr)
    q_nope, q_pe = q[..., :dn], q[..., dn:]
    ckv = x @ hf["kv_a_proj_with_mqa"].T
    ckv, k_pe = ckv[:, :cfg["kv_rank"]], ckv[:, cfg["kv_rank"]:]
    kv = (_rmsnorm(ckv, hf["kv_a_layernorm"], cfg["rms_eps"]) @ hf["kv_b_proj"].T)
    kv = kv.reshape(T, H, dn + dv)
    k_nope, v = kv[..., :dn], kv[..., dn:]
    freqs = np.outer(pos, _yarn_inv_freq(cfg))
    emb = np.concatenate([freqs, freqs], axis=-1)
    m = (_yarn_get_mscale(cfg["factor"], cfg["mscale"])
         / _yarn_get_mscale(cfg["factor"], cfg["mscale_all_dim"]))
    cos, sin = np.cos(emb) * m, np.sin(emb) * m              # (T, dr)

    def rotate_half(z):
        return np.concatenate([-z[..., dr // 2:], z[..., :dr // 2]], axis=-1)

    def apply(z):                                           # (T, h, dr), stored order
        z = z.reshape(z.shape[0], z.shape[1], dr // 2, 2).transpose(0, 1, 3, 2).reshape(z.shape)
        return z * cos[:, None, :] + rotate_half(z) * sin[:, None, :]
    q_pe, k_pe = apply(q_pe), apply(k_pe[:, None, :])
    query = np.concatenate([q_nope, q_pe], axis=-1)
    key = np.concatenate([k_nope, np.broadcast_to(k_pe, (T, H, dr))], axis=-1)
    scores = np.einsum("thc,uhc->htu", query, key) * softmax_scale(cfg)
    out = np.einsum("htu,uhc->thc", _causal_softmax(scores), v).reshape(T, H * dv)
    return out @ hf["o_proj"].T


def composition_a_attention(x, w, cfg, positions=None):
    """The same layer as the prover's claims compute it (§4.2), in float64,
    on mla_weights_from_gguf's output: σ folded into the two query weights,
    half-split RoPE at the prover's YaRN frequencies (claims._rope_yarn_inv_freq),
    the per-head assembly with the shared rotary key, scores with no further
    scale."""
    import sys
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from claims import RoPEConfig, _rope_yarn_inv_freq, _rope_yarn_mscale
    T = x.shape[0]
    H, dn, dr, dv = cfg["H"], cfg["d_nope"], cfg["d_rope"], cfg["d_v"]
    pos = np.arange(T) if positions is None else np.asarray(positions)
    rc = RoPEConfig(SEQ=1, d_h=dr, s_x=1, base=cfg["base"], scale_factor=cfg["factor"],
                    original_max_pos=cfg["original_max_pos"], yarn=True,
                    yarn_beta_fast=cfg["beta_fast"], yarn_beta_slow=cfg["beta_slow"],
                    yarn_mscale=cfg["mscale"], yarn_mscale_all_dim=cfg["mscale_all_dim"])
    theta = np.outer(pos, np.array(_rope_yarn_inv_freq(rc)))
    cos, sin = np.cos(theta) * _rope_yarn_mscale(rc), np.sin(theta) * _rope_yarn_mscale(rc)

    def rope(z):                                            # (T, h, dr), half-split
        lo, hi = z[..., :dr // 2], z[..., dr // 2:]
        c, s = cos[:, None, :], sin[:, None, :]
        return np.concatenate([lo * c - hi * s, lo * s + hi * c], axis=-1)
    sigma = softmax_scale(cfg)
    a_q = _rmsnorm(x @ w["W_qa"], w["g_q_a"], cfg["rms_eps"])
    q_nope = (a_q @ (w["W_q_nope"] * sigma)).reshape(T, H, dn)
    q_pe = rope((a_q @ (w["W_q_pe"] * sigma)).reshape(T, H, dr))
    ckv = _rmsnorm(x @ w["W_ckv"], w["g_kv_a"], cfg["rms_eps"])
    k_pe = rope((x @ w["W_k_pe"]).reshape(T, 1, dr))
    k_nope = (ckv @ w["W_k_nope"]).reshape(T, H, dn)
    v = (ckv @ w["W_v"]).reshape(T, H, dv)
    query = np.concatenate([q_nope, q_pe], axis=-1)                       # head_interleave
    key = np.concatenate([k_nope, np.broadcast_to(k_pe, (T, H, dr))], axis=-1)  # shared
    scores = np.einsum("thc,uhc->htu", query, key)
    out = np.einsum("htu,uhc->thc", _causal_softmax(scores), v).reshape(T, H * dv)
    return out @ w["W_o"]
