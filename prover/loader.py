"""Llama-2-7B weight loading for the Ligero prover — quantize HF BF16 weights to
Q3.12 Goldilocks field-rep uint64 CUDA tensors, ready to commit into a Tape.

(Merged from the former llama_loader.py + lazy_loader.py.)

Two modes:
  - Eager: `load_layer_weights()` loads + quantizes a whole transformer block via
    a per-process cached HF model (`_get_model`).
  - Lazy: `LazyHFLoader` reads one tensor at a time from .safetensors shards and
    quantizes on demand — used when the setup must not hold all 32 layers
    (~50 GB) at once (the SEQ=1000 / unified-memory path; at most one ~360 MB
    weight resolved at a time).

Concerns handled (both modes):
  1. Layout: HF nn.Linear weight is (out, in); Tape.matmul wants the right
     operand at (k=in, n=out), so every projection weight is transposed.
  2. Quantization: BF16 → integer at scale S (Q-format); signed reals map to
     Goldilocks field elements (negatives → P − |v|).
  3. The 1/√d_h factor on attention scores is folded into W_Q at quantization
     time (see demo_llama7b.py); the loader applies it via `divide_by`.

KNOWN GAP: per-channel RmsNorm gains ("rms_pre_*_w") are returned but not yet
consumed by RmsNormClaim — loading them is bookkeeping until RmsNormClaim (or a
following HadamardClaim) applies the learned gain.

Requires `transformers` + `torch`.
"""
from __future__ import annotations

import json
import math
import os
from typing import Callable, Dict, Optional, Tuple

import torch

from cuda_primitives import P, gl_sub


_MODEL_CACHE: Dict[str, object] = {}


def _get_model(model_id_or_path: str):
    """Return a (cached) HF model. Loading is heavy (10s+); cache by path."""
    if model_id_or_path not in _MODEL_CACHE:
        from transformers import AutoModelForCausalLM
        _MODEL_CACHE[model_id_or_path] = AutoModelForCausalLM.from_pretrained(
            model_id_or_path, torch_dtype=torch.bfloat16)
    return _MODEL_CACHE[model_id_or_path]


def _signed_to_field(t_int: torch.Tensor) -> torch.Tensor:
    """Signed int64 → uint64 Goldilocks field rep (P − |v| for v < 0)."""
    v_abs = t_int.abs().to(torch.uint64)
    P_t   = torch.full_like(v_abs, P)
    # int64-view select: torch CUDA has no `where` for uint64; bits are identical.
    return torch.where(t_int >= 0, t_int,
                       gl_sub(P_t, v_abs).view(torch.int64)).view(torch.uint64)


def quantize_to_field(t: torch.Tensor, scale: int, *,
                       divide_by: float = 1.0) -> torch.Tensor:
    """Quantize a float tensor to Q-format integers at scale `scale`, optionally
    pre-dividing by a public scalar `divide_by`. Returns a CUDA uint64 Goldilocks
    field tensor.  v_int = round(v_real / divide_by · scale)."""
    t_f = t.to(torch.float64).to("cuda")
    if divide_by != 1.0:
        t_f = t_f / divide_by
    t_int = torch.round(t_f * scale).to(torch.int64)
    return _signed_to_field(t_int)


def load_layer_weights(model_id_or_path: str, layer_idx: int, *,
                        S: int = 2 ** 12,
                        d_h: int = 128,
                        fold_inv_sqrt_d_h_into_W_Q: bool = True,
                        extra_q_k_shrink: float = 1.0,
                        ) -> Dict[str, torch.Tensor]:
    """Load one Llama-2-7B transformer block's weights (eager, via the cached HF
    model), quantize to scale S, return a dict of CUDA uint64 field tensors.

    `extra_q_k_shrink` folds an extra √N into BOTH W_Q and W_K (a stand-in for
    softmax magnitude control — see the original note). Keys/shapes:
      W_Q W_K W_V W_O (d,d) [W_Q has 1/√d_h folded] · W_gate W_up (d,d_ff) ·
      W_down (d_ff,d) · rms_pre_attn_w rms_pre_ffn_w (d,) per-channel gains.
    """
    model = _get_model(model_id_or_path)
    layer = model.model.layers[layer_idx]

    sqrt_d_h = math.sqrt(d_h)
    qk_shrink = math.sqrt(extra_q_k_shrink)
    Q_div = (sqrt_d_h if fold_inv_sqrt_d_h_into_W_Q else 1.0) * qk_shrink
    K_div = qk_shrink

    # HF nn.Linear weight is (out_features, in_features). Our matmul expects the
    # right-operand at (k=in_features, n=out_features), so we transpose.
    out = {
        "W_Q":    quantize_to_field(
            layer.self_attn.q_proj.weight.T.contiguous(), S, divide_by=Q_div),
        "W_K":    quantize_to_field(
            layer.self_attn.k_proj.weight.T.contiguous(), S, divide_by=K_div),
        "W_V":    quantize_to_field(
            layer.self_attn.v_proj.weight.T.contiguous(), S),
        "W_O":    quantize_to_field(
            layer.self_attn.o_proj.weight.T.contiguous(), S),
        "W_gate": quantize_to_field(
            layer.mlp.gate_proj.weight.T.contiguous(), S),
        "W_up":   quantize_to_field(
            layer.mlp.up_proj.weight.T.contiguous(), S),
        "W_down": quantize_to_field(
            layer.mlp.down_proj.weight.T.contiguous(), S),
        "rms_pre_attn_w": quantize_to_field(
            layer.input_layernorm.weight, S),
        "rms_pre_ffn_w":  quantize_to_field(
            layer.post_attention_layernorm.weight, S),
    }
    return out


class LazyHFLoader:
    """Holds shard-map metadata for an HF Llama checkpoint; produces per-weight
    loader callables that read+quantize one tensor at a time (no full-model
    materialization)."""

    def __init__(self, model_id_or_path: str, *,
                  S: int = 2 ** 12,
                  d_h: int = 128,
                  fold_inv_sqrt_d_h_into_W_Q: bool = True,
                  extra_q_k_shrink: float = 1.0):
        self.model_id = model_id_or_path
        self.S = S
        sqrt_d_h = math.sqrt(d_h)
        qk_shrink = math.sqrt(extra_q_k_shrink)
        self.Q_div = (sqrt_d_h if fold_inv_sqrt_d_h_into_W_Q else 1.0) * qk_shrink
        self.K_div = qk_shrink

        self.model_dir = self._find_model_dir(model_id_or_path)
        self.shard_map = self._load_shard_map()

    @staticmethod
    def _find_model_dir(model_id_or_path: str) -> str:
        if os.path.isdir(model_id_or_path):
            return model_id_or_path
        from transformers.utils import cached_file
        config_file = cached_file(model_id_or_path, "config.json")
        return os.path.dirname(config_file)

    def _load_shard_map(self):
        index_file = os.path.join(self.model_dir, "model.safetensors.index.json")
        if os.path.exists(index_file):
            with open(index_file) as f:
                return json.load(f)["weight_map"]
        return None   # single-shard model

    def _shard_for(self, param_name: str) -> str:
        if self.shard_map is None:
            return os.path.join(self.model_dir, "model.safetensors")
        return os.path.join(self.model_dir, self.shard_map[param_name])

    def _load_raw(self, param_name: str) -> torch.Tensor:
        """Load a single tensor in its native dtype (bf16 for HF Llama)."""
        from safetensors.torch import safe_open
        with safe_open(self._shard_for(param_name), framework="pt", device="cpu") as f:
            return f.get_tensor(param_name)

    def _get_shape(self, param_name: str) -> Tuple[int, ...]:
        from safetensors.torch import safe_open
        with safe_open(self._shard_for(param_name), framework="pt", device="cpu") as f:
            return tuple(f.get_slice(param_name).get_shape())

    def make_loader(self, param_name: str, *,
                     transpose: bool = False,
                     divide_by: float = 1.0) -> Callable[[], torch.Tensor]:
        """Return a closure that reads `param_name`, optionally transposes,
        quantizes to scale self.S, and returns a flat CUDA uint64 tensor. No
        caching — each call hits disk."""
        S = self.S

        def load() -> torch.Tensor:
            t = self._load_raw(param_name)
            if transpose:
                t = t.T.contiguous()
            return quantize_to_field(t, S, divide_by=divide_by).view(-1)

        return load

    def load_embedding(self, divide_by: float = 1.0) -> torch.Tensor:
        """Token embedding table (vocab·d,) quantized — read directly from
        safetensors (no full-model materialization)."""
        emb = self._load_raw("model.embed_tokens.weight")
        return quantize_to_field(emb.contiguous(), self.S, divide_by=divide_by)

    def load_final_weights(self) -> Dict[str, torch.Tensor]:
        """Final RmsNorm gain + LM head, read directly from safetensors. Falls
        back to the (tied) embedding for the LM head if the checkpoint has no
        separate lm_head.weight."""
        has_lm = self.shard_map is None or "lm_head.weight" in self.shard_map
        lm_name = "lm_head.weight" if has_lm else "model.embed_tokens.weight"
        return {
            "final_norm_w": quantize_to_field(self._load_raw("model.norm.weight"), self.S),
            "W_lm_head": quantize_to_field(self._load_raw(lm_name).T.contiguous(), self.S),
        }

    def layer_specs(self, layer_idx: int) -> Dict[str, Tuple[str, bool, float]]:
        """Per-weight metadata for one transformer layer:
        {short_name: (hf_param_name, transpose, divide_by)}. Matches the keys +
        transpose convention of load_layer_weights."""
        p = f"model.layers.{layer_idx}"
        return {
            "W_Q":            (f"{p}.self_attn.q_proj.weight",         True,  self.Q_div),
            "W_K":            (f"{p}.self_attn.k_proj.weight",         True,  self.K_div),
            "W_V":            (f"{p}.self_attn.v_proj.weight",         True,  1.0),
            "W_O":            (f"{p}.self_attn.o_proj.weight",         True,  1.0),
            "W_gate":         (f"{p}.mlp.gate_proj.weight",            True,  1.0),
            "W_up":           (f"{p}.mlp.up_proj.weight",              True,  1.0),
            "W_down":         (f"{p}.mlp.down_proj.weight",            True,  1.0),
            "rms_pre_attn_w": (f"{p}.input_layernorm.weight",          False, 1.0),
            "rms_pre_ffn_w":  (f"{p}.post_attention_layernorm.weight", False, 1.0),
        }

    def layer_shapes(self, d: int, d_ff: int) -> Dict[str, Tuple[int, ...]]:
        """Layout shapes for the matmul/hadamard claims, matching
        load_layer_weights' transposed convention (k=in, n=out)."""
        return {
            "W_Q":    (d, d),
            "W_K":    (d, d),
            "W_V":    (d, d),
            "W_O":    (d, d),
            "W_gate": (d, d_ff),
            "W_up":   (d, d_ff),
            "W_down": (d_ff, d),
            "rms_pre_attn_w": (d,),
            "rms_pre_ffn_w":  (d,),
        }


def load_final_weights(model_id_or_path: str, *,
                        S: int = 2 ** 12) -> Dict[str, torch.Tensor]:
    """Final RmsNorm gain + LM head, read directly from safetensors (no
    full-model materialization — that load is ~27 GB and, doubled with the
    embedding load, OOMs the 32-layer SEQ=1000 setup on the GB10's unified pool)."""
    return LazyHFLoader(model_id_or_path, S=S).load_final_weights()


def load_token_embedding(model_id_or_path: str, *,
                          S: int = 2 ** 12,
                          divide_by: float = 1.0) -> torch.Tensor:
    """Token embedding table, quantized to scale S, read straight from
    safetensors (no full-model materialization). `divide_by` applies a public
    scalar division at quantization time. Returns a flat (vocab·d,) CUDA uint64
    tensor."""
    return LazyHFLoader(model_id_or_path, S=S).load_embedding(divide_by=divide_by)


def free_model_cache():
    """Drop cached models; call before phases that need the GPU memory."""
    global _MODEL_CACHE
    _MODEL_CACHE.clear()
    torch.cuda.empty_cache()


def tokenize_prompt(model_id_or_path: str, prompt: str) -> torch.Tensor:
    """Tokenize `prompt` using Llama-2-7B's tokenizer. Returns a 1-D int64 CUDA
    tensor of token ids (integer indices into the embedding table)."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id_or_path)
    ids = tok(prompt, return_tensors="pt").input_ids[0].to("cuda")
    return ids


def _self_test():
    """Quantization roundtrip check. Doesn't require HF weights."""
    src = torch.tensor([-1.5, -0.5, 0.0, 0.5, 1.5], dtype=torch.bfloat16)
    S = 2 ** 12
    q = quantize_to_field(src, S)
    expected = torch.tensor([-6144, -2048, 0, 2048, 6144], dtype=torch.int64,
                             device="cuda")
    got = _signed_to_field(expected)
    assert torch.equal(q, got), (
        f"quantize_to_field mismatch: q={q.cpu().tolist()} vs got={got.cpu().tolist()}")
    print("loader._self_test: OK")
    q_div = quantize_to_field(src, S, divide_by=math.sqrt(128))
    q_no = quantize_to_field(src, S, divide_by=1.0)
    assert q_div[0] != q_no[0], "divide_by didn't change result"
    print("  divide_by=√d_h reduces magnitudes as expected.")


# ═══════════════════════════════════════════════════════════════════════════
# Maverick / GGUF (UD-Q4_K_XL) — M1 plan Phase 0.
#
# Reads ONE MoE layer's tensors from a (memory-mapped) GGUF file, dequantizes
# per tensor type (UD is a MIXED quantization: Q4_K experts, Q5_K/Q6_K/Q8_0/
# F32 elsewhere — gguf.quants.dequantize dispatches on the recorded type),
# and field-quantizes at scale S. Expert slicing happens on the RAW quantized
# memmap (leading data dim = expert), so n_experts=8 dev mode never touches
# the other 120 experts' bytes.
#
# Tensor names follow llama.cpp's MoE conventions; the orientation (transpose
# to this loader's (k=in, n=out) matmul layout) assumes llama.cpp's
# (d_out, d_in) row-major convention — M0 (analysis/maverick_m0_check.py)
# validates both against the actual file + a llama.cpp reference forward.
# ═══════════════════════════════════════════════════════════════════════════

MAVERICK_MOE_TENSORS = {
    # key: (gguf name pattern, stacked-experts?)   numpy dims after dequant:
    "gate_exps": ("blk.{i}.ffn_gate_exps.weight", True),    # (E, d_ff, d)
    "up_exps":   ("blk.{i}.ffn_up_exps.weight",   True),    # (E, d_ff, d)
    "down_exps": ("blk.{i}.ffn_down_exps.weight", True),    # (E, d, d_ff)
    "router":    ("blk.{i}.ffn_gate_inp.weight",  False),   # (E, d)
    "gate_sh":   ("blk.{i}.ffn_gate_shexp.weight", False),  # (d_ff, d)
    "up_sh":     ("blk.{i}.ffn_up_shexp.weight",   False),  # (d_ff, d)
    "down_sh":   ("blk.{i}.ffn_down_shexp.weight", False),  # (d, d_ff)
}


_GGUF_INDEX: Dict[str, dict] = {}


def _gguf_by_name(gguf_path: str) -> dict:
    """Tensor-name index over the shard set, built once per path (the 5 shard
    header parses cost seconds; lazy per-expert loaders call in a loop)."""
    if gguf_path not in _GGUF_INDEX:
        import glob as _glob
        import re as _re
        from gguf import GGUFReader
        if os.path.isdir(gguf_path):
            paths = sorted(_glob.glob(os.path.join(gguf_path, "*.gguf")))
        elif _re.search(r"-\d{5}-of-\d{5}\.gguf$", gguf_path):
            paths = sorted(_glob.glob(_re.sub(r"-\d{5}-of-(\d{5})\.gguf$",
                                               r"-*-of-\1.gguf", gguf_path)))
        else:
            paths = [gguf_path]
        assert paths, f"no .gguf files found at {gguf_path}"
        by_name = {}
        for p in paths:
            for t in GGUFReader(p).tensors:
                by_name[t.name] = t
        _GGUF_INDEX[gguf_path] = by_name
    return _GGUF_INDEX[gguf_path]


def maverick_lazy_expert(gguf_path: str, layer_idx: int, key: str, expert: int,
                          S: int = 2 ** 12):
    """Zero-arg closure for tape.commit_lazy: dequantizes ONE expert's matrix
    (raw-slice → fp32 → transpose to (k_in, n_out) → field at scale S) on each
    call and frees it after — peak memory one expert, not 128."""
    from gguf.quants import dequantize
    pat, stacked = MAVERICK_MOE_TENSORS[key]
    assert stacked, f"{key} is not a stacked expert tensor"
    name = pat.format(i=layer_idx)

    def load():
        import numpy as np
        t = _gguf_by_name(gguf_path)[name]
        qt = t.tensor_type.name
        if qt in ("Q4_K", "Q5_K", "Q6_K"):
            # Fused path: raw block bytes -> field integers on the GPU.
            from kquant_cuda import kquant_to_field
            raw = np.ascontiguousarray(t.data[expert])       # (d_out, row_bytes)
            d_out = int(t.data.shape[1])
            w = kquant_to_field(torch.from_numpy(raw).cuda(), qt, S)
            d_in = w.numel() // d_out
            # transpose to (k_in, n_out) via the int64 bit-view (uint64 lacks .T)
            return (w.view(d_out, d_in).view(torch.int64).T.contiguous()
                    .view(torch.uint64).reshape(-1))
        # Fallback (F32 / exotic types): reference numpy dequant path.
        d = dequantize(t.data[expert:expert + 1], t.tensor_type)[0]
        return quantize_to_field(torch.from_numpy(d.copy()).T.contiguous(),
                                 S).reshape(-1)
    return load


def read_maverick_moe_layer(gguf_path: str, layer_idx: int, *,
                             n_experts: int = None, expert_indices=None,
                             skip_experts: bool = False):
    """Numpy fp32 dict of one MoE layer's tensors (CPU, no CUDA needed).
    Stacked expert tensors are sliced BEFORE dequantization (raw quantized
    rows; leading data dim = expert): `n_experts` takes the first n,
    `expert_indices` (list) takes specific experts — the stacked outputs then
    hold those experts in the given order.

    `gguf_path` may be a single .gguf file, a gguf-split shard set (pass any
    one shard — `-NNNNN-of-NNNNN.gguf` siblings are globbed), or a directory
    containing the shards. The real UD-Q4_K_XL release is 5 shards.
    `skip_experts=True` returns only the non-stacked tensors (router + shared
    expert) — the per-expert matrices then come via maverick_lazy_expert /
    tape.commit_lazy, since materializing all 128 experts (~63 GB fp32 +
    ~100 GB field) exceeds the Spark's 121 GB unified memory."""
    from gguf.quants import dequantize
    by_name = _gguf_by_name(gguf_path)
    out = {}
    for key, (pat, stacked) in MAVERICK_MOE_TENSORS.items():
        if skip_experts and stacked:
            continue
        name = pat.format(i=layer_idx)
        if name not in by_name:
            raise KeyError(
                f"{name} not in GGUF — tensor-name drift; present blk.{layer_idx} "
                f"tensors: {[n for n in by_name if n.startswith(f'blk.{layer_idx}.')]}")
        t = by_name[name]
        data = t.data
        if stacked and expert_indices is not None:
            import numpy as _np
            data = data[_np.asarray(list(expert_indices), dtype=_np.int64)]
        elif stacked and n_experts is not None:
            assert data.shape[0] >= n_experts, \
                f"{name}: leading dim {data.shape[0]} < n_experts={n_experts}"
            data = data[:n_experts]
        out[key] = dequantize(data, t.tensor_type)
    return out


def load_maverick_moe_layer(gguf_path: str, layer_idx: int, *,
                             S: int = 2 ** 12, n_experts: int = None,
                             skip_experts: bool = False):
    """Field-quantized torch dict for the demo builder, in this loader's
    transposed (k=in, n=out) matmul layout:
      W_gate/W_up: list of (d, d_ff) · W_down: list of (d_ff, d) ·
      W_router: (d, E_sliced) · W_{gate,up}_sh: (d, d_ff) · W_down_sh: (d_ff, d)
    Per-expert tensors are quantized one at a time (peak ≈ one expert fp32)."""
    raw = read_maverick_moe_layer(gguf_path, layer_idx, n_experts=n_experts,
                                   skip_experts=skip_experts)

    def q(np_arr):     # (d_out, d_in) → transpose → field at scale S
        return quantize_to_field(torch.from_numpy(np_arr.copy()).T.contiguous(), S)

    out = {
        "W_router":  q(raw["router"][:n_experts] if n_experts is not None
                       else raw["router"]),
        "W_gate_sh": q(raw["gate_sh"]),
        "W_up_sh":   q(raw["up_sh"]),
        "W_down_sh": q(raw["down_sh"]),
    }
    if not skip_experts:
        out["W_gate"] = [q(raw["gate_exps"][e]) for e in range(raw["gate_exps"].shape[0])]
        out["W_up"]   = [q(raw["up_exps"][e])   for e in range(raw["up_exps"].shape[0])]
        out["W_down"] = [q(raw["down_exps"][e]) for e in range(raw["down_exps"].shape[0])]
    return out


if __name__ == "__main__":
    _self_test()
