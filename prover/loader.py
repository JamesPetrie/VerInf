"""Dense-Llama HF weight loading for the Ligero prover — quantize HF BF16
weights to Q3.12 Goldilocks field-rep uint64 CUDA tensors, ready to commit
into a Tape. Dims come from the checkpoint's own config.json (ModelConfig);
Llama-2-7B and Llama-3.2-1B are the same code path.

(Merged from the former llama_loader.py + lazy_loader.py.)

Two modes, one weight-spec table (`layer_specs`) driving both:
  - Eager: `load_layer_weights()` reads + quantizes a whole transformer
    block's tensors from the safetensors shards.
  - Lazy: `LazyHFLoader.make_loader` returns per-weight callables that read
    one tensor at a time and quantize on demand — used when the setup must
    not hold all layers (~50 GB for the 7B) at once (the SEQ=1000 /
    unified-memory path; at most one weight resolved at a time).

Concerns handled (both modes):
  1. Layout: HF nn.Linear weight is (out, in); Tape.matmul wants the right
     operand at (k=in, n=out), so every projection weight is transposed.
  2. Quantization: BF16 → integer at scale S (Q-format); signed reals map to
     Goldilocks field elements (negatives → P − |v|).
  3. The 1/√d_h factor on attention scores is folded into W_Q at quantization
     time (see demo_llama7b.py); the loader applies it via `divide_by`.
  4. GQA (n_kv_heads < n_heads): each KV head's columns are replicated
     kv_groups× after transpose (`replicate_kv_cols`) so K/V commit at full
     (d, n_heads·d_h) width and attention proves as plain MHA — a public
     weight transform, zero new claims (same pattern as the Maverick demo).
  5. Tied embeddings (tie_word_embeddings): checkpoints with no separate
     lm_head.weight fall back to the embedding matrix, detected from the
     actual tensor keys (NOT the shard count — single-shard tied models
     like Llama-3.2-1B have no lm_head.weight anywhere).

KNOWN GAP: per-channel RmsNorm gains ("rms_pre_*_w") are returned but not yet
consumed by RmsNormClaim — loading them is bookkeeping until RmsNormClaim (or a
following HadamardClaim) applies the learned gain.

Requires `transformers` + `torch`.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import torch

from cuda_primitives import P, gl_sub
from model_config import ModelConfig, find_model_dir


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


def replicate_kv_cols(w_t: torch.Tensor, *, d_h: int,
                       groups: int) -> torch.Tensor:
    """GQA public weight transform: (d, n_kv·d_h) → (d, n_kv·groups·d_h),
    repeating each KV head's d_h-column block `groups`× so K/V commit at
    full query-head width and attention proves as plain MHA — zero new
    claims. Exact bit-copy on the field tensor (int64 view; uint64 lacks
    repeat_interleave on CUDA). groups=1 is the identity."""
    if groups == 1:
        return w_t
    d, kv_width = w_t.shape
    n_kv = kv_width // d_h
    assert n_kv * d_h == kv_width, (
        f"replicate_kv_cols: kv width {kv_width} not a multiple of d_h={d_h}")
    v = w_t.view(torch.int64).view(d, n_kv, d_h)
    return (v.repeat_interleave(groups, dim=1).contiguous()
             .view(d, n_kv * groups * d_h).view(torch.uint64))


@dataclass(frozen=True)
class WeightSpec:
    """Per-weight load recipe: HF tensor name + the public transforms applied
    before commit. One table of these (layer_specs) drives BOTH the eager and
    lazy paths, so they can't drift."""
    hf_name: str
    transpose: bool = False
    divide_by: float = 1.0
    kv_groups: int = 1          # >1 ⇒ replicate_kv_cols after transpose (GQA)


def load_layer_weights(model_id_or_path: str, layer_idx: int, *,
                        S: int = 2 ** 12,
                        d_h: Optional[int] = None,
                        fold_inv_sqrt_d_h_into_W_Q: bool = True,
                        extra_q_k_shrink: float = 1.0,
                        ) -> Dict[str, torch.Tensor]:
    """Load one transformer block's weights (eager), quantize to scale S,
    return a dict of CUDA uint64 field tensors shaped per `layer_shapes`.

    Thin wrapper over LazyHFLoader — same spec table, same transforms
    (transpose, 1/√d_h fold into W_Q, GQA KV replication), just resolved
    immediately. `d_h` defaults to the checkpoint's config.json value.

    `extra_q_k_shrink` folds an extra √N into BOTH W_Q and W_K (a stand-in
    for softmax magnitude control — see the original note). Keys/shapes:
      W_Q W_K W_V W_O (d, n_heads·d_h)/(n_heads·d_h, d) [W_Q has 1/√d_h
      folded; K/V replicated to full width under GQA] · W_gate W_up (d,d_ff)
      · W_down (d_ff,d) · rms_pre_attn_w rms_pre_ffn_w (d,) gains.
    """
    ldr = LazyHFLoader(model_id_or_path, S=S, d_h=d_h,
                       fold_inv_sqrt_d_h_into_W_Q=fold_inv_sqrt_d_h_into_W_Q,
                       extra_q_k_shrink=extra_q_k_shrink)
    shapes = ldr.layer_shapes()
    out = {}
    for short, spec in ldr.layer_specs(layer_idx).items():
        flat = ldr.make_loader(spec.hf_name, transpose=spec.transpose,
                               divide_by=spec.divide_by,
                               kv_groups=spec.kv_groups)()
        out[short] = flat.view(shapes[short])
    return out


class LazyHFLoader:
    """Holds shard-map metadata for an HF Llama checkpoint; produces per-weight
    loader callables that read+quantize one tensor at a time (no full-model
    materialization)."""

    def __init__(self, model_id_or_path: str, *,
                  S: int = 2 ** 12,
                  d_h: Optional[int] = None,
                  fold_inv_sqrt_d_h_into_W_Q: bool = True,
                  extra_q_k_shrink: float = 1.0):
        self.model_id = model_id_or_path
        self.S = S
        self.model_dir = find_model_dir(model_id_or_path)
        self.config = ModelConfig.from_hf(self.model_dir)
        self.d_h = self.config.d_h if d_h is None else d_h
        sqrt_d_h = math.sqrt(self.d_h)
        qk_shrink = math.sqrt(extra_q_k_shrink)
        self.Q_div = (sqrt_d_h if fold_inv_sqrt_d_h_into_W_Q else 1.0) * qk_shrink
        self.K_div = qk_shrink

        self.shard_map = self._load_shard_map()

    def _load_shard_map(self):
        index_file = os.path.join(self.model_dir, "model.safetensors.index.json")
        if os.path.exists(index_file):
            with open(index_file) as f:
                return json.load(f)["weight_map"]
        single = os.path.join(self.model_dir, "model.safetensors")
        if not os.path.exists(single):
            raise FileNotFoundError(
                f"no model.safetensors[.index.json] under {self.model_dir} — "
                f"download the checkpoint first (huggingface-cli download)")
        return None   # single-shard model

    def _shard_for(self, param_name: str) -> str:
        if self.shard_map is None:
            return os.path.join(self.model_dir, "model.safetensors")
        return os.path.join(self.model_dir, self.shard_map[param_name])

    def has_tensor(self, param_name: str) -> bool:
        """Whether the checkpoint actually contains `param_name` — from the
        shard index when present, else the single shard's real key set (a
        single-shard TIED model has no lm_head.weight; inferring presence
        from shard count crashes on exactly those checkpoints)."""
        if self.shard_map is not None:
            return param_name in self.shard_map
        from safetensors import safe_open
        with safe_open(self._shard_for(param_name), framework="pt",
                       device="cpu") as f:
            return param_name in f.keys()

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
                     divide_by: float = 1.0,
                     kv_groups: int = 1) -> Callable[[], torch.Tensor]:
        """Return a closure that reads `param_name`, optionally transposes,
        quantizes to scale self.S, replicates KV columns kv_groups× (GQA;
        1 = no-op), and returns a flat CUDA uint64 tensor. No caching — each
        call hits disk."""
        S, d_h = self.S, self.d_h

        def load() -> torch.Tensor:
            t = self._load_raw(param_name)
            if transpose:
                t = t.T.contiguous()
            q = quantize_to_field(t, S, divide_by=divide_by)
            if kv_groups > 1:
                q = replicate_kv_cols(q, d_h=d_h, groups=kv_groups)
            return q.reshape(-1)

        return load

    def load_embedding(self, divide_by: float = 1.0) -> torch.Tensor:
        """Token embedding table (vocab·d,) quantized — read directly from
        safetensors (no full-model materialization)."""
        emb = self._load_raw("model.embed_tokens.weight")
        return quantize_to_field(emb.contiguous(), self.S, divide_by=divide_by)

    def has_separate_lm_head(self) -> bool:
        """False for tied-embedding checkpoints (tie_word_embeddings), which
        ship no lm_head.weight tensor at all — e.g. Llama-3.2-1B."""
        return self.has_tensor("lm_head.weight")

    def load_final_norm(self) -> torch.Tensor:
        """Final RmsNorm gain only — for tied-embedding models that reuse
        the committed embedding table as the LM head (quantizing a
        transposed LM-head copy would be wasted work there)."""
        return quantize_to_field(self._load_raw("model.norm.weight"), self.S)

    def load_final_weights(self) -> Dict[str, torch.Tensor]:
        """Final RmsNorm gain + LM head, read directly from safetensors. Falls
        back to the (tied) embedding for the LM head if the checkpoint has no
        separate lm_head.weight — detected from the actual tensor keys."""
        lm_name = ("lm_head.weight" if self.has_separate_lm_head()
                   else "model.embed_tokens.weight")
        return {
            "final_norm_w": self.load_final_norm(),
            "W_lm_head": quantize_to_field(self._load_raw(lm_name).T.contiguous(), self.S),
        }

    def layer_specs(self, layer_idx: int) -> Dict[str, WeightSpec]:
        """Per-weight load recipes for one transformer layer — the ONE table
        both the eager and lazy paths consume. K/V carry the GQA replication
        factor (1 on MHA models like Llama-2-7B, where it is a no-op)."""
        p = f"model.layers.{layer_idx}"
        g = self.config.kv_groups
        return {
            "W_Q":            WeightSpec(f"{p}.self_attn.q_proj.weight", True, self.Q_div),
            "W_K":            WeightSpec(f"{p}.self_attn.k_proj.weight", True, self.K_div, kv_groups=g),
            "W_V":            WeightSpec(f"{p}.self_attn.v_proj.weight", True, 1.0, kv_groups=g),
            "W_O":            WeightSpec(f"{p}.self_attn.o_proj.weight", True),
            "W_gate":         WeightSpec(f"{p}.mlp.gate_proj.weight",    True),
            "W_up":           WeightSpec(f"{p}.mlp.up_proj.weight",      True),
            "W_down":         WeightSpec(f"{p}.mlp.down_proj.weight",    True),
            "rms_pre_attn_w": WeightSpec(f"{p}.input_layernorm.weight"),
            "rms_pre_ffn_w":  WeightSpec(f"{p}.post_attention_layernorm.weight"),
        }

    def layer_shapes(self, d: Optional[int] = None,
                     d_ff: Optional[int] = None) -> Dict[str, Tuple[int, ...]]:
        """COMMITTED layout shapes for the matmul/hadamard claims, matching
        load_layer_weights' transposed convention (k=in, n=out). K/V are the
        post-replication (full query-head width) shapes under GQA. `d`/`d_ff`
        default from config.json; passing them cross-checks the caller's dims
        against the checkpoint's."""
        cfg = self.config
        if d is None:
            d = cfg.d
        if d_ff is None:
            d_ff = cfg.d_ff
        assert (d, d_ff) == (cfg.d, cfg.d_ff), (
            f"caller dims (d={d}, d_ff={d_ff}) != checkpoint config "
            f"(d={cfg.d}, d_ff={cfg.d_ff})")
        qw = cfg.n_heads * self.d_h        # committed Q/K/V width
        return {
            "W_Q":    (d, qw),
            "W_K":    (d, qw),
            "W_V":    (d, qw),
            "W_O":    (qw, d),
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
    """Release cached GPU memory; call before phases that need it. (The old
    per-process HF model cache is gone — the eager path now reads tensors
    straight from safetensors — but callers still use this as a memory
    checkpoint.)"""
    torch.cuda.empty_cache()


def tokenize_prompt(model_id_or_path: str, prompt: str) -> torch.Tensor:
    """Tokenize `prompt` using the model's tokenizer. Returns a 1-D int64 CUDA
    tensor of token ids (integer indices into the embedding table)."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id_or_path)
    ids = tok(prompt, return_tensors="pt").input_ids[0].to("cuda")
    return ids


def tokenize_chat_prompt(model_id_or_path: str, user_message: str) -> list:
    """Render a single-turn chat via the model's chat template (Instruct
    header tokens, auto system block, trailing generation prompt) and
    tokenize. Returns a plain list of ids. Rendered to TEXT first, then
    tokenized — apply_chat_template(tokenize=True)'s return type varies
    across tokenizers versions (list vs Encoding objects).
    add_special_tokens=False: the template already emits <|begin_of_text|>."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id_or_path)
    text = tok.apply_chat_template(
        [{"role": "user", "content": user_message}],
        tokenize=False, add_generation_prompt=True)
    return [int(t) for t in tok(text, add_special_tokens=False).input_ids]


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
