"""Per-architecture pieces of the engine arm (analysis/k2-fidelity-study-design.md
§5.2): what llama.cpp must be told so that its model is the published
definition, which tensors to capture, and how their names map to ours.

K2's YaRN betas: llama.cpp's deepseek2 loader (src/models/deepseek2.cpp at
f174553) never reads `rope.scaling.yarn_beta_*` and leaves its defaults,
beta_fast = 32 and beta_slow = 1. K2's config.json states 1 and 1, and so do
our references. On the all-F32 toy, llama.cpp with K2's betas equals the
float64 reference to float32 rounding; with its defaults the layer outputs
are 2% off. The betas are therefore part of the engine's configuration, not a
choice made after results (§3.3). Reported upstream as ggml-org/llama.cpp#30830."""
import numpy as np

KIMI_K2 = dict(name="kimi-k2", engine=dict(yarn_beta_fast=1.0, yarn_beta_slow=1.0),
               routed_scale=2.827, k=8, dense_lead=1)
MOONLIGHT = dict(name="moonlight", engine={}, routed_scale=2.446, k=6, dense_lead=1)


def captures(adapter, layers: int) -> tuple:
    """Every layer's output, and each MoE layer's selected ids and normalized gates."""
    names = [f"l_out-{i}" for i in range(layers)]
    for i in range(adapter["dense_lead"], layers):
        names += [f"ffn_moe_topk-{i}", f"ffn_moe_weights_norm-{i}"]
    return tuple(names)


def ours(adapter, caps: dict, n_experts: int) -> dict:
    """Captured tensors under our names: L{i}.out, L{i}.top (sorted ids),
    L{i}.gates (T, E), the normalized gates (summing to 1 over the selected
    experts, zero elsewhere), which the normalized gate distance compares across
    models, and L{i}.gates_scaled, the same times the routed scale, each
    expert's actual weight on its output."""
    out = {}
    for name, a in caps.items():
        kind, il = name.rsplit("-", 1)
        if kind == "l_out":
            out[f"L{il}.out"] = np.asarray(a, dtype=np.float64)
    for name, a in caps.items():
        kind, il = name.rsplit("-", 1)
        if kind == "ffn_moe_topk":
            ids = np.asarray(a).astype(np.int64)
            w = np.asarray(caps[f"ffn_moe_weights_norm-{il}"], dtype=np.float64)
            g = np.zeros((ids.shape[0], n_experts))
            np.put_along_axis(g, ids, w, axis=1)
            out[f"L{il}.top"] = np.sort(ids, axis=1)
            out[f"L{il}.gates"] = g
            out[f"L{il}.gates_scaled"] = g * adapter["routed_scale"]
    return out
