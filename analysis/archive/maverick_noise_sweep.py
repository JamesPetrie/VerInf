"""Noise amplification of Q4+MoE: U(Q4-with-input-noise vs Q4 baseline).

Baseline AND perturbed runs use the SAME fp32-on-dequantized-Q4 forward (no
llama.cpp-kernel confound): only the embedding inputs differ. Sweep spans
truly tiny sparse perturbations (a fraction `frac` of the T·d embedding
entries shifted by ±`step` quantization units of 1/4096 — frac=1e-4 ≈ 7
entries of 71680 moved by the proof's smallest representable amount) up to
dense significant noise. Per level we report: per-layer router flips vs
baseline, greedy agreement on the baseline's argmax continuation, and
U = Σ −log2 softmax_noisy(o_baseline) over the 12 generated positions.

Dequantized weights are cached in RAM (~80 GB; box has 375), so run 1 pays
~40 min of dequant and each further level costs only minutes.

Run:  ~/miniconda3/bin/python maverick_noise_sweep.py
"""
import numpy as np
from maverick_full_ref import (build_index, fetch as _fetch, unpermute,
                                rms_norm, rope, _silu, _sigmoid, NOPE_STEP)

GGUF = "/home/amodo/maverick-gguf/UD-Q4_K_XL"
IDS = [200000, 954, 2182, 373, 262, 17252, 323, 1092, 954, 6076, 323, 12311, 25, 656]
N_PROMPT = 2
LEVELS = [(1e-4, 1), (1e-3, 1), (1e-2, 1), (1e-1, 1), (1.0, 1), (1.0, 16), (1.0, 256)]
STEP = 1.0 / 4096                     # one integer step at the proof's S = 2^12

BY, KV = build_index(GGUF)
_g = lambda k: KV[f"llama4.{k}"]
L, d = _g("block_count"), _g("embedding_length")
H, Hkv, Dh = _g("attention.head_count"), _g("attention.head_count_kv"), _g("attention.key_length")
theta, eps = _g("rope.freq_base"), _g("attention.layer_norm_rms_epsilon")
_CACHE = {}


def fetch(name, rows=None):
    key = (name, tuple(rows) if rows is not None else None)
    if key not in _CACHE:
        _CACHE[key] = _fetch(BY, name, rows=rows)
    return _CACHE[key]


def forward(x):
    """Returns (logits (T,V), routing (n_moe, T) chosen-expert array)."""
    T = x.shape[0]
    causal = np.tril(np.ones((T, T), bool))
    routing = []
    for il in range(L):
        p = f"blk.{il}."
        h = rms_norm(x, fetch(p + "attn_norm.weight"), eps)
        q = (h @ unpermute(fetch(p + "attn_q.weight"), H).T).reshape(T, H, Dh)
        k = (h @ unpermute(fetch(p + "attn_k.weight"), Hkv).T).reshape(T, Hkv, Dh)
        v = (h @ fetch(p + "attn_v.weight").T).reshape(T, Hkv, Dh)
        if (il + 1) % NOPE_STEP != 0:
            q, k = rope(q, theta), rope(k, theta)
        k, v = np.repeat(k, H // Hkv, 1), np.repeat(v, H // Hkv, 1)
        sc = np.einsum("thd,shd->hts", q, k) / np.sqrt(Dh)
        sc = np.where(causal[None], sc, -np.inf)
        w = np.exp(sc - sc.max(-1, keepdims=True)); w /= w.sum(-1, keepdims=True)
        x = x + np.einsum("hts,shd->thd", w, v).reshape(T, H * Dh) \
            @ fetch(p + "attn_output.weight").T
        h = rms_norm(x, fetch(p + "ffn_norm.weight"), eps)
        if il % 2 == 1:
            lg = h @ fetch(p + "ffn_gate_inp.weight").T
            es = lg.argmax(1); routing.append(es.copy())
            s = _sigmoid(lg[np.arange(T), es])
            ffn = np.empty_like(x)
            for t in range(T):
                e = int(es[t]); xw = s[t] * h[t]
                gw = fetch(p + "ffn_gate_exps.weight", rows=[e])[0]
                uw = fetch(p + "ffn_up_exps.weight", rows=[e])[0]
                dw = fetch(p + "ffn_down_exps.weight", rows=[e])[0]
                ffn[t] = (_silu(xw @ gw.T) * (xw @ uw.T)) @ dw.T
            ffn += (_silu(h @ fetch(p + "ffn_gate_shexp.weight").T)
                    * (h @ fetch(p + "ffn_up_shexp.weight").T)) \
                @ fetch(p + "ffn_down_shexp.weight").T
        else:
            ffn = (_silu(h @ fetch(p + "ffn_gate.weight").T)
                   * (h @ fetch(p + "ffn_up.weight").T)) @ fetch(p + "ffn_down.weight").T
        x = x + ffn
    xf = rms_norm(x, fetch("output_norm.weight"), eps)
    return xf @ fetch("output.weight").T, np.array(routing)


def u_logit_noise(logits, outs, sigma):
    """The repo's estimator (analysis/ui_real_measure.py): Q(o) ∝ exp(−gap²/2σ²)
    over the full vocab, gap = max(ℓ) − ℓ. Prover-chosen σ; we report a sweep."""
    tot = 0.0
    for j, tok in enumerate(outs):
        pos = N_PROMPT + j - 1
        l = logits[pos]
        w = np.exp(-((l.max() - l) ** 2) / (2.0 * sigma * sigma))
        tot += -np.log2(w[tok] / w.sum())
    return tot


def surprisal(logits, outs):
    tot = 0.0; agree = 0
    for j, tok in enumerate(outs):
        pos = N_PROMPT + j - 1
        l = logits[pos] - logits[pos].max()
        tot += -(l[tok] - np.log(np.exp(l).sum())) / np.log(2)
        agree += int(logits[pos].argmax() == tok)
    return tot, agree


x0 = fetch("token_embd.weight", rows=IDS)
print("baseline forward...", flush=True)
base_logits, base_routing = forward(x0)
outs = [int(base_logits[N_PROMPT + j - 1].argmax()) for j in range(len(IDS) - N_PROMPT + 1)]
print(f"baseline continuation (argmax at each position): {outs}", flush=True)
np.save("/tmp/noise_logits_base.npy", base_logits.astype(np.float32))
print(f"\n{'frac':>8} {'step':>5} {'flipped_in':>10} {'router_flips':>12} "
      f"{'agree':>6} {'U_sm':>8} {'U_ln.3':>8} {'U_ln1':>8} {'U_ln3':>8}")
rng = np.random.default_rng(11)
for frac, step in LEVELS:
    xn = x0.copy()
    mask = rng.random(xn.shape) < frac
    xn[mask] += rng.choice([-1.0, 1.0], size=int(mask.sum())).astype(np.float32) * step * STEP
    lg, rt = forward(xn)
    np.save(f"/tmp/noise_logits_{frac:g}_{step}.npy", lg.astype(np.float32))
    flips = int((rt != base_routing).sum())
    u, agree = surprisal(lg, outs)
    uln = [u_logit_noise(lg, outs, sg) for sg in (0.3, 1.0, 3.0)]
    print(f"{frac:8.0e} {step:5d} {int(mask.sum()):10d} "
          f"{flips:7d}/{rt.size:4d} {agree:3d}/{len(outs)} {u:8.2f} "
          f"{uln[0]:8.3f} {uln[1]:8.3f} {uln[2]:8.3f}", flush=True)
