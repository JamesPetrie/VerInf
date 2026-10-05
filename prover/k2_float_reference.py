"""Kimi K2's forward pass in float64 on the dequantized GGUF weights, and the
two comparisons the gate makes after exact integer agreement
(analysis/k2-session-sizing.md §6.1), kept apart:

* fidelity: the integer reference's intermediates (k2_int_reference) against
  this float pass, as relative errors at the same names, and the logits'
  top-1 agreement;
* routing differences: the experts each token selects, (a) end to end, each
  pass routing its own hidden state, and (b) on the same input, the float
  router applied to the integer pass's normalized hidden state, which
  isolates the selection's own quantization (logit rounding, the sigmoid
  table, the bias at S_sel) from the drift upstream of it. Each difference
  carries the float margin at the k-th place, which says whether it is a
  near-tie.

Attention is k2_attention.composition_a_attention (#29), the layer as the
prover's claims compose it, equal to the Hugging Face port to rounding. The
MoE is DeepSeek-V3's MoEGate as K2 configures it (sigmoid scores, the
selection bias added for the choice only, one group, weights normalized over
the k chosen and scaled by routed_scaling_factor 2.827)."""
import numpy as np

from k2_attention import composition_a_attention

ROUTED_SCALING = 2.827


def rmsnorm(x, g, eps):
    return x / np.sqrt((x * x).mean(axis=-1, keepdims=True) + eps) * g


def silu(x):
    return x / (1.0 + np.exp(-x))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def ffn(x, wg, wu, wd):
    return (silu(x @ wg) * (x @ wu)) @ wd


def select(scores, bias, k):
    """Top-k of scores + bias, lowest index on a tie (the prover's tiebreak)."""
    q = scores + bias[None, :]
    order = np.lexsort((np.arange(q.shape[1])[None, :].repeat(q.shape[0], 0), -q), axis=1)
    return order[:, :k], q


def moe(x, W, il, *, k, eps, out, p):
    scores = sigmoid(x @ W.router(il))
    bias = W.bias(il)
    top, _ = select(scores, bias, k)
    T = x.shape[0]
    y = np.zeros_like(x)
    w = np.take_along_axis(scores, top, axis=1)
    w = w / w.sum(axis=1, keepdims=True) * ROUTED_SCALING
    for e in sorted(set(top.ravel().tolist())):
        ts, slots = np.nonzero(top == e)
        h = ffn(x[ts], W.expert(il, "gate", e), W.expert(il, "up", e),
                W.expert(il, "down", e))
        y[ts] += w[ts, slots][:, None] * h
    sh = W.shared(il)
    out.update({f"{p}.s": scores, f"{p}.top": np.sort(top, axis=1), f"{p}.y": y,
                f"{p}.sh": ffn(x, sh["W_gate"], sh["W_up"], sh["W_down"])})
    return y + out[f"{p}.sh"]


def forward(ids, W, cfg, *, layers, offset=0, k=8, head_chunk=16384):
    """Token ids to logits in float64. `W` gives float weights
    (k2_loader.GgufWeights(ints=False)). Returns {name: float array} at the
    integer reference's names where the two compute the same quantity."""
    eps = cfg["rms_eps"]
    T = len(ids)
    pos = offset + np.arange(T)
    out = {}
    x = np.asarray(W.embed_rows(ids), dtype=np.float64)
    out["x0"] = x
    for il in range(layers):
        p = f"L{il}"
        w = W.attn(il)
        n1g = rmsnorm(x, w["g_attn"], eps)
        r1 = x + composition_a_attention(n1g, w, cfg, positions=pos)
        n2g = rmsnorm(r1, w["g_ffn"], eps)
        out.update({f"{p}.n1g": n1g, f"{p}.r1": r1, f"{p}.n2g": n2g})
        if W.kind(il) == "dense":
            d = W.dense(il)
            f = ffn(n2g, d["W_gate"], d["W_up"], d["W_down"])
            out[f"{p}.ffn"] = f
            x = r1 + f
        else:
            x = r1 + moe(n2g, W, il, k=k, eps=eps, out=out, p=p)
        out[f"{p}.out"] = x
    nfg = rmsnorm(x, W.g_out(), eps)
    out["final.ng"] = nfg
    logits = np.empty((T, W.V))
    for v0 in range(0, W.V, head_chunk):
        v1 = min(W.V, v0 + head_chunk)
        logits[:, v0:v1] = nfg @ W.head(v0, v1)
    out["logits"] = logits
    return out


# ---- the comparisons ------------------------------------------------------------

def fidelity(int_out: dict, float_out: dict, S: int, S_sel: int) -> dict:
    """Relative L2 error and max |error| (real units) of each integer
    intermediate against its float twin (the router scores at S_sel, the
    rest at S), and the logits' top-1 agreement."""
    rows = {}
    for name in sorted(set(int_out) & set(float_out)):
        fi, ff = int_out[name], float_out[name]
        if name.endswith((".mask", ".top")) or np.asarray(fi).dtype == object:
            continue
        a = np.asarray(fi, dtype=np.float64) / (S_sel if name.endswith(".s") else S)
        b = np.asarray(ff, dtype=np.float64).reshape(a.shape)
        rows[name] = dict(rel_l2=float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300)),
                          max_abs=float(np.abs(a - b).max()))
    if "logits" in int_out and "logits" in float_out:
        rows["logits.top1_agree"] = float(
            (int_out["logits"].argmax(axis=1) == float_out["logits"].argmax(axis=1)).mean())
    return rows


def _margins(q, top, k):
    """The float gap between the k-th and (k+1)-th of q per token."""
    srt = -np.sort(-q, axis=1)
    return srt[:, k - 1] - srt[:, k]


def routing_differences(int_out: dict, float_out: dict, W, il: int, *, k: int, S: int) -> dict:
    """(a) end to end: the integer mask against the float pass's own choice;
    (b) same input: against the float router applied to the integer n2g."""
    p = f"L{il}"
    mask = np.asarray(int_out[f"{p}.mask"])
    int_top = np.sort(np.argsort(-mask, axis=1, kind="stable")[:, :k], axis=1)
    bias = W.bias(il)

    def compare(top_f, q_f):
        T = top_f.shape[0]
        overlap = np.asarray([len(set(int_top[t]) & set(top_f[t])) for t in range(T)])
        margins = _margins(q_f, top_f, k)
        diff = np.nonzero(overlap < k)[0]
        return dict(tokens=int(T), tokens_differing=int(len(diff)),
                    experts_differing=int((k - overlap).sum()),
                    margins_of_differing=[float(margins[t]) for t in diff[:64]],
                    median_margin=float(np.median(margins)))
    q_end = float_out[f"{p}.s"] + bias[None, :]
    end = compare(float_out[f"{p}.top"], q_end)
    n2g_int = np.asarray(int_out[f"{p}.n2g"], dtype=np.float64) / S
    s_same = sigmoid(n2g_int @ W.router(il))
    top_same, q_same = select(s_same, bias, k)
    same = compare(np.sort(top_same, axis=1), q_same)
    return {"end_to_end": end, "same_input": same}
