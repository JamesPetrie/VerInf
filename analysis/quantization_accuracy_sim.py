"""
End-to-end quantization accuracy simulation for Llama-2-7B.

Patches a HuggingFace Llama model to mimic the quantization semantics of the
ZK protocol:

  - matmul: inputs and outputs rounded to a chosen fixed-point scale (Q3.12
    by default); accumulation in int64 (pure integer arithmetic) so the
    result is bit-exact with no FP rounding anywhere on the quantized path.
  - nonlinearities (silu, exp for softmax, 1/sqrt for RMSNorm, 1/sum for
    softmax division): replaced with paired-tlookup-style table lookups at
    T = 2^16. The table is built once at the chosen scale and resolution,
    then applied bit-exactly: input rounded to the nearest table entry,
    output read directly from the (pre-rounded) table.

This gives a bit-exact simulation of what the ZK proof's verified output
would be at our parameters.

Reference baseline: FP16 (not FP32). The inference transcript that the
proof would be generated from runs at FP16 on the GPU; we account for
small FP16-level differences via the zkllm-entropy approach, so what
matters for design purposes is the *additional* divergence introduced by
the Q3.12 + table-lookup scheme on top of FP16's own noise floor. If
Q3.12 + T = 2^16 add only sub-FP16-noise drift, the table size doesn't
need to grow.

Target hardware: DGX Spark (`spark-c191` on Tailscale). NVIDIA GB10 Grace
Blackwell Superchip with unified memory. Setup notes at the bottom of
this file.

Status: v1 skeleton. Quantization primitives and patched module classes are
complete; full WikiText-2 eval loop is sketched but unverified end-to-end.
Run `python3 quantization_accuracy_sim.py --smoke` first to sanity-check
the patch on a single forward pass; expand to WikiText-2 once that's clean.
"""

from __future__ import annotations
import argparse
import math
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----- Defaults (Q3.12 with T = 2^16 lookups) --------------------------------

DEFAULT_SCALE = 2 ** 12          # Q3.12: real ∈ [-8, 8], stored ∈ [-2^15, 2^15]
DEFAULT_TABLE_T = 2 ** 16
DEFAULT_EXP_CLIP = 16.0          # input clip for the post-shift exp table
DEFAULT_RMSNORM_INPUT_RANGE = 64.0     # rsqrt input mean(X²) range
DEFAULT_SILU_X_MAX = 20.0              # silu input range; sweet spot from scale_sweep at scale=2^14 (top-1 99.45%)
                                       # for Llama-7B at T=2^16. Wider (128) loses
                                       # too much resolution at the small-input
                                       # mass; narrower (8) clamps too aggressively
                                       # and propagates a 50% error through layers.
                                       # Proper fix is log-spaced / multi-segment
                                       # silu (same pattern as rsqrt).
#
# Empirical note from first Llama-7B smoke test on spark-c191 (KL ≈ 8.9 with
# the above defaults, KL ≈ 28 with wider ranges): Q3.12 + T = 2^16 does NOT
# produce coherent next-token outputs out of the box on Llama-7B. Widening
# table ranges hurts more than it helps because it costs resolution per
# entry. The fix likely needs one or more of:
#   (a) instrument activation distributions to set ranges from data
#   (b) raise scale to Q5.10 or Q6.10 to give activations more headroom
#   (c) raise T to 2^18 or 2^20 to recover lost resolution from wider ranges
#   (d) multi-segment lookups for rsqrt and silu so resolution scales with
#       the function's local Lipschitz constant
# Debugging path (untouched in this commit): patch only nn.Linear first,
# check KL drift from that alone, then incrementally add RMSNorm / silu /
# softmax patches to isolate which operation's quantization is killing it.


# ----- Quantization primitives -----------------------------------------------
#
# All quantized values flow through these primitives as int64 stored integers.
# A float value x is represented as the int64 round(x * scale); reconstruction
# back to float at the boundary is via int64 -> float -> /scale. All arithmetic
# on quantized values happens in int64 so there is no floating-point error
# anywhere on the quantized path. Integer overflow checks: at Q3.12 with the
# overflow margin from precision_overflow_model.py, every intermediate stays
# well below int64 limits.

def quantize_to_int(x: torch.Tensor, scale: int) -> torch.Tensor:
    """Round x to int64 stored value at the given scale."""
    # round-to-nearest-even via torch.round (banker's rounding); torch.round
    # operates on floats so we go via float, but only for the rounding step.
    # The result is a pure int64 tensor used in all downstream int arithmetic.
    return torch.round(x.float() * scale).to(torch.int64)


def dequantize_from_int(x_int: torch.Tensor, scale: int, dtype: torch.dtype) -> torch.Tensor:
    """Convert int64 stored value back to a float in real units."""
    return (x_int.to(torch.float32) / scale).to(dtype)


def round_div(x_int: torch.Tensor, d: int) -> torch.Tensor:
    """
    Round x_int / d to the nearest int, half-away-from-zero. d positive int.
    Pure int64 arithmetic (no FP).
    """
    half = d // 2
    return torch.where(
        x_int >= 0,
        (x_int + half) // d,
        -((-x_int + half) // d),
    )


@dataclass
class Table:
    """Precomputed paired-lookup table for a nonlinearity, stored as int64.

    Inputs are integer-quantized at `input_scale`; the table covers stored
    inputs in [input_min_int, input_max_int] with T uniformly-spaced entries.
    Outputs are int64 stored at `output_scale`.
    """
    outputs: torch.Tensor   # int64, shape (T,), stored at output_scale
    input_min_int: int
    input_max_int: int
    input_scale: int
    output_scale: int
    T: int


def build_table(
    f: Callable[[torch.Tensor], torch.Tensor],
    x_min: float,
    x_max: float,
    T: int,
    input_scale: int,
    output_scale: int,
    device: str = "cuda",
) -> Table:
    """
    Build a paired-lookup table.

    The table is computed in FP64 once at construction time (for accuracy
    when calling math functions), then rounded to int64 stored values. After
    construction, all uses are int-only.
    """
    inputs_fp = torch.linspace(x_min, x_max, T, dtype=torch.float64, device=device)
    outputs_fp = f(inputs_fp)
    outputs_int = torch.round(outputs_fp * output_scale).to(torch.int64)
    return Table(
        outputs=outputs_int,
        input_min_int=int(round(x_min * input_scale)),
        input_max_int=int(round(x_max * input_scale)),
        input_scale=input_scale,
        output_scale=output_scale,
        T=T,
    )


@dataclass
class LogTable:
    """
    Log-spaced lookup table for functions like rsqrt whose Lipschitz constant
    varies by orders of magnitude across the input range.

    Models the multi-segment lookup pattern (§3.4 "Table size and precision").
    The PyTorch sim uses FP log() for index computation; the ZK protocol would
    realize the same shape via K small tables in base-b digit decomposition.
    """
    outputs: torch.Tensor   # int64, shape (T,), at output_scale
    x_min: float
    x_max: float
    log_min: float
    log_max: float
    input_scale: int
    output_scale: int
    T: int


def build_log_table(
    f: Callable[[torch.Tensor], torch.Tensor],
    x_min: float,
    x_max: float,
    T: int,
    input_scale: int,
    output_scale: int,
    device: str = "cuda",
) -> LogTable:
    """Build a log-spaced lookup table. Inputs log-uniformly spaced in [x_min, x_max]."""
    import math
    log_min = math.log(x_min)
    log_max = math.log(x_max)
    log_inputs = torch.linspace(log_min, log_max, T, dtype=torch.float64, device=device)
    inputs_fp = torch.exp(log_inputs)
    outputs_fp = f(inputs_fp)
    outputs_int = torch.round(outputs_fp * output_scale).to(torch.int64)
    return LogTable(
        outputs=outputs_int,
        x_min=x_min,
        x_max=x_max,
        log_min=log_min,
        log_max=log_max,
        input_scale=input_scale,
        output_scale=output_scale,
        T=T,
    )


def apply_log_table(x_int: torch.Tensor, table: LogTable) -> torch.Tensor:
    """
    Look up a log-spaced table by integer input. NOTE: this uses FP log() at
    runtime, which is a deviation from pure-int semantics. For a faithful ZK
    sim this should be replaced with integer log2 + segment lookup. For now
    this validates that *some* multi-segment-flavored lookup recovers Llama-7B
    accuracy; the int-only translation comes after the model is shown to work.
    """
    x_real = x_int.double() / table.input_scale
    x_clamped = x_real.clamp(table.x_min, table.x_max)
    log_x = torch.log(x_clamped)
    idx = ((log_x - table.log_min) / (table.log_max - table.log_min) * (table.T - 1)).round().long()
    idx = idx.clamp(0, table.T - 1)
    return table.outputs[idx]


def apply_table(x_int: torch.Tensor, table: Table) -> torch.Tensor:
    """
    Apply a paired-lookup table to int64 input. Input is clamped to the
    table's input range, rounded to the nearest table index via integer
    arithmetic, and the stored int64 output is returned.
    """
    x_clamped = x_int.clamp(table.input_min_int, table.input_max_int)
    span = table.input_max_int - table.input_min_int
    # idx in [0, T-1]; integer multiply-then-divide. The multiply
    # (x_clamped - input_min_int) * (T-1) can be large but fits in int64:
    # at our defaults x_clamped <= 2^44, T-1 = 2^16-1 -> product <= 2^60.
    idx = ((x_clamped - table.input_min_int) * (table.T - 1) + span // 2) // span
    return table.outputs[idx]


# ----- Patched module: linear (matmul) ---------------------------------------

class QuantizedLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with quantized semantics.

    The weight matrix and input activations are integer-quantized at the
    chosen scale. The matmul itself runs in FP64 because PyTorch CUDA does
    not implement int64 matmul kernels. This is still bit-exact at our
    magnitudes: FP64 represents integers up to 2^53 exactly, and our matmul
    accumulator stays below 2^44 by the precision_overflow_model.py bound
    (k <= 16384 contracted entries, each input <= 2^15 -> sum <= 2^44).

    Every individual multiplication a*b where |a|, |b| <= 2^15 gives an
    integer-valued FP64 of magnitude <= 2^30 (exact). Every partial sum
    along the contraction stays integer-valued and below 2^53. Therefore
    cuBLAS DGEMM, no matter how it tiles or fuses, produces an exact
    integer-valued FP64 result. We cast the result back to int64 and
    continue in pure integer arithmetic from there.

    The first call in smoke_test runs a verification check (see _verify_int):
    confirms the matmul output is bit-exactly integer-valued by computing
    abs(out - out.round()).max() and asserting it is 0.
    """

    _verify_int = False  # set to True in smoke_test for the one-shot check

    def __init__(self, linear: nn.Linear, scale: int = DEFAULT_SCALE):
        super().__init__()
        self.scale = int(scale)
        with torch.no_grad():
            w_int = quantize_to_int(linear.weight, self.scale)
            self.register_buffer("weight_int", w_int)
            if linear.bias is not None:
                b_int = quantize_to_int(linear.bias, self.scale)
                self.register_buffer("bias_int", b_int)
            else:
                self.bias_int = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_dtype = x.dtype
        x_int = quantize_to_int(x, self.scale)
        # FP64 matmul; result is integer-valued at our bounds.
        out_fp = F.linear(x_int.double(), self.weight_int.double())
        if QuantizedLinear._verify_int:
            err = (out_fp - out_fp.round()).abs().max().item()
            if err > 0.0:
                raise RuntimeError(
                    f"FP64 matmul not bit-exact at these magnitudes: max abs(out - round(out)) = {err}"
                )
        out_int = out_fp.long()  # exact since out_fp is integer-valued
        # out_int is at scale^2; divide by scale to return to Q-scale.
        out_int = round_div(out_int, self.scale)
        if self.bias_int is not None:
            out_int = out_int + self.bias_int
        return dequantize_from_int(out_int, self.scale, out_dtype)


# ----- Patched module: RMSNorm -----------------------------------------------

class QuantizedRMSNorm(nn.Module):
    """
    RMSNorm via pure int64 arithmetic:
      - quantize input X to int64 at scale
      - X² Hadamard: int64 * int64 -> int64 (values at scale^2)
      - sum over embed_dim (int64 sum)
      - mean: round-divide by d (int64 -> int64 at scale^2)
      - rsqrt via paired-lookup int64 table (input at scale^2, output at scale)
      - γ · rsqrt: int64 * int64 -> int64 (at scale^2)
      - round-divide by scale to bring back to scale
      - Hadamard with X: int64 * int64 -> int64 (at scale^2)
      - round-divide by scale, cast to layer dtype

    Note: the rsqrt table input scale is scale^2 because mean(X²) is at
    scale^2 (X² has scale^2 by construction). The output scale matches the
    activation scale.
    """

    def __init__(
        self,
        original: nn.Module,
        scale: int = DEFAULT_SCALE,
        T: int = DEFAULT_TABLE_T,
        input_range: float = DEFAULT_RMSNORM_INPUT_RANGE,
        log_spaced: bool = True,
    ):
        super().__init__()
        self.scale = int(scale)
        self.eps = original.variance_epsilon
        self.input_range = input_range
        self.log_spaced = log_spaced
        with torch.no_grad():
            w_int = quantize_to_int(original.weight, self.scale)
            self.register_buffer("weight_int", w_int)
        # Build the rsqrt table. Llama-7B mean(X²) spans ~7 orders of magnitude
        # (3e-5 to 263), so a linear-spaced table can't give adequate precision
        # at both ends. Default to log-spaced (multi-segment-flavored).
        device = self.weight_int.device
        eps = self.eps
        x_min = eps  # smallest mean(X²) the table needs to handle
        x_max = max(input_range, 512.0)  # cover up to ~263 max + headroom
        if log_spaced:
            self.table = build_log_table(
                lambda x: 1.0 / torch.sqrt(x + eps),
                x_min=x_min,
                x_max=x_max,
                T=T,
                input_scale=self.scale * self.scale,
                output_scale=self.scale,
                device=device,
            )
        else:
            self.table = build_table(
                lambda x: 1.0 / torch.sqrt(x + eps),
                x_min=x_min,
                x_max=x_max,
                T=T,
                input_scale=self.scale * self.scale,
                output_scale=self.scale,
                device=device,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_dtype = x.dtype
        d = x.size(-1)
        x_int = quantize_to_int(x, self.scale)
        # X² at scale^2
        x_sq = x_int * x_int
        # Sum over embed_dim then round-divide by d to get mean.
        sum_sq = x_sq.sum(dim=-1, keepdim=True)
        mean_sq = round_div(sum_sq, d)
        # rsqrt via lookup. table input is at scale^2, output at scale.
        if isinstance(self.table, LogTable):
            rsqrt_int = apply_log_table(mean_sq, self.table)
        else:
            rsqrt_int = apply_table(mean_sq, self.table)
        # γ · rsqrt at scale^2.
        gamma_rsqrt = self.weight_int * rsqrt_int
        # Bring back to scale.
        gamma_rsqrt = round_div(gamma_rsqrt, self.scale)
        # Y = (γ · rsqrt) · X at scale^2.
        y = gamma_rsqrt * x_int
        # Back to scale.
        y = round_div(y, self.scale)
        return dequantize_from_int(y, self.scale, out_dtype)


# ----- Patched activation: SiLU ----------------------------------------------

class QuantizedSiLU:
    """
    SiLU via int64 paired-lookup table. Inputs outside the table range are
    clamped to the boundary (so silu output is bounded by silu(x_max)).

    Empirical finding (see commit log): closed-form "silu(x) = x for large
    positive x" approximations break Llama-7B at layers 30-31 — the model
    appears to depend on silu's output being bounded for stability. Linear
    clamping at x_max=32 is the empirical sweet spot at T=2^16.
    """

    def __init__(
        self,
        scale: int = DEFAULT_SCALE,
        T: int = DEFAULT_TABLE_T,
        x_max: float = DEFAULT_SILU_X_MAX,
    ):
        self.scale = int(scale)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.table = build_table(
            lambda x: F.silu(x.float()).double(),
            x_min=-x_max,
            x_max=x_max,
            T=T,
            input_scale=self.scale,
            output_scale=self.scale,
            device=device,
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        out_dtype = x.dtype
        x_int = quantize_to_int(x, self.scale)
        y_int = apply_table(x_int, self.table)
        return dequantize_from_int(y_int, self.scale, out_dtype)


# ----- Patched softmax (row-max shift + exp table + 1/sum table) -------------

class QuantizedSoftmax:
    """
    Numerically-stable softmax via pure int64 arithmetic:
      - quantize scores to int64
      - row-max shift (z' = z - max(z), so max(z') = 0); pure int subtraction
      - clip z' to [-clip_int, 0]
      - exp via paired-lookup int64 table (input at scale, output at scale)
      - row-sum (int64 sum)
      - 1/sum via paired-lookup int64 table (input at scale, output at scale)
      - softmax = exp * inv_sum / scale (int multiply then round-divide by scale)
    """

    def __init__(
        self,
        scale: int = DEFAULT_SCALE,
        T: int = DEFAULT_TABLE_T,
        exp_clip: float = DEFAULT_EXP_CLIP,
        max_seq_len: int = 8192,
    ):
        self.scale = int(scale)
        self.exp_clip = exp_clip
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # exp table over [-exp_clip, 0]; input at scale, output at scale.
        self.exp_table = build_table(
            lambda x: torch.exp(x),
            x_min=-exp_clip,
            x_max=0.0,
            T=T,
            input_scale=self.scale,
            output_scale=self.scale,
            device=device,
        )
        # 1/sum table: sum range [1, max_seq_len] in real units. Linear-spaced
        # tables struggle here because 1/sum has steep slope at small sums.
        # Use log-spaced lookup (same trick as rsqrt) for adequate precision.
        self.inv_sum_table = build_log_table(
            lambda s: 1.0 / s,
            x_min=1.0,
            x_max=float(max_seq_len),
            T=T,
            input_scale=self.scale,
            output_scale=self.scale,
            device=device,
        )

    def __call__(self, scores: torch.Tensor, dim: int = -1) -> torch.Tensor:
        out_dtype = scores.dtype
        # Row-max shift + clip in FP, then quantize. Doing the shift in int64
        # would overflow on -inf mask values (from causal masks). The protocol's
        # softmax handles masked positions separately via a public mask, so
        # this rearrangement is simulation-only.
        max_scores = scores.max(dim=dim, keepdim=True).values
        shifted_fp = (scores - max_scores).clamp(min=-self.exp_clip, max=0.0)
        clipped = quantize_to_int(shifted_fp, self.scale)
        # exp lookup -> int64 at scale.
        e_int = apply_table(clipped, self.exp_table)
        # Row-sum in int64.
        row_sum_int = e_int.sum(dim=dim, keepdim=True)
        # 1/sum lookup -> int64 at scale. Log-spaced table (built in __init__).
        inv_sum_int = apply_log_table(row_sum_int, self.inv_sum_table)
        # softmax = e * inv_sum, at scale^2; round-divide back to scale.
        y_int = e_int * inv_sum_int
        y_int = round_div(y_int, self.scale)
        return dequantize_from_int(y_int, self.scale, out_dtype)


# ----- Quantized matmul + hadamard helpers -----------------------------------
# Used for the inner-attention torch.matmul calls (QK^T, attn @ V) and the
# silu(gate) * up elementwise multiply in LlamaMLP. These are NOT nn.Linear
# instances, so patch_model_inplace's recursion does not see them. They are
# wired in by patch_inner_ops_inplace below.

def quantized_matmul(a: torch.Tensor, b: torch.Tensor, scale: int = DEFAULT_SCALE) -> torch.Tensor:
    """
    Q-grid matmul. Round inputs to int64 at scale; FP64 matmul (bit-exact at
    Q3.12 magnitudes per the QuantizedLinear bound); cast to int64;
    round-divide by scale; dequantize to original dtype.

    Used for QK^T and attn @ V inside LlamaAttention. Inputs have shape
    (batch, heads, seq, head_dim) so torch.matmul does a batched matmul; the
    accumulator length is head_dim (128 for Llama-2-7b) or seq, both well
    inside the int-exact bound.
    """
    out_dtype = a.dtype
    a_int = quantize_to_int(a, scale)
    b_int = quantize_to_int(b, scale)
    out_fp = torch.matmul(a_int.double(), b_int.double())
    out_int = out_fp.long()
    out_int = round_div(out_int, scale)
    return dequantize_from_int(out_int, scale, out_dtype)


def quantized_hadamard(a: torch.Tensor, b: torch.Tensor, scale: int = DEFAULT_SCALE) -> torch.Tensor:
    """
    Q-grid elementwise multiply. Round both inputs to int64, multiply
    pointwise in int64, round-divide by scale, dequantize.

    Used for silu(gate) * up_proj in LlamaMLP.
    """
    out_dtype = a.dtype
    a_int = quantize_to_int(a, scale)
    b_int = quantize_to_int(b, scale)
    prod_int = a_int * b_int
    prod_int = round_div(prod_int, scale)
    return dequantize_from_int(prod_int, scale, out_dtype)


def _make_quantized_llama_attention_forward(scale: int):
    """
    Returns a forward method for LlamaAttention that mirrors HF's
    eager_attention_forward but routes the two torch.matmul calls (QK^T and
    attn @ V) through quantized_matmul, and calls F.softmax explicitly so
    install_activation_patches's QuantizedSoftmax actually fires.

    Tested against transformers 5.8.0 LlamaAttention signature.
    """
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None,
                past_key_values=None, **kwargs):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx
            )

        k_rep = repeat_kv(key_states, self.num_key_value_groups)
        v_rep = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = quantized_matmul(query_states, k_rep.transpose(2, 3), scale=scale) * self.scaling

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # F.softmax is monkey-patched to QuantizedSoftmax by install_activation_patches.
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        attn_output = quantized_matmul(attn_weights, v_rep, scale=scale)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    return forward


def _make_quantized_llama_mlp_forward(scale: int):
    """
    Returns a forward method for LlamaMLP that routes the gate * up
    elementwise multiply through quantized_hadamard. gate_proj/up_proj/down_proj
    are already QuantizedLinear (via patch_model_inplace), and act_fn is
    F.silu which is monkey-patched to QuantizedSiLU.
    """
    def forward(self, x):
        gate = self.gate_proj(x)
        gate_act = self.act_fn(gate)
        up = self.up_proj(x)
        prod = quantized_hadamard(gate_act, up, scale=scale)
        return self.down_proj(prod)

    return forward


def patch_inner_ops_inplace(model: nn.Module, scale: int = DEFAULT_SCALE) -> None:
    """
    Replace LlamaAttention.forward and LlamaMLP.forward on every instance with
    quantized variants. Combined with patch_model_inplace (Linear, RMSNorm)
    and install_activation_patches (F.silu, F.softmax) this makes every op in
    the transformer block run through the Q-grid -- no more FP fallthrough at
    QK^T, attn @ V, silu, softmax, or silu*up.
    """
    import transformers.models.llama.modeling_llama as ml

    attn_fwd = _make_quantized_llama_attention_forward(scale=scale)
    mlp_fwd = _make_quantized_llama_mlp_forward(scale=scale)

    n_attn = n_mlp = 0
    for module in model.modules():
        if isinstance(module, ml.LlamaAttention):
            module.forward = attn_fwd.__get__(module, type(module))
            n_attn += 1
        elif isinstance(module, ml.LlamaMLP):
            module.forward = mlp_fwd.__get__(module, type(module))
            n_mlp += 1
    print(f"patch_inner_ops_inplace: replaced forward on {n_attn} LlamaAttention and {n_mlp} LlamaMLP instances.")


# ----- Patching helpers ------------------------------------------------------

def patch_model_inplace(
    model: nn.Module,
    scale: float = DEFAULT_SCALE,
    T: int = DEFAULT_TABLE_T,
    patch_linear: bool = True,
    patch_rmsnorm: bool = True,
    patch_inner_ops: bool = True,
) -> None:
    """
    Recursively walk the model and replace:
      - nn.Linear     → QuantizedLinear           (if patch_linear)
      - LlamaRMSNorm  → QuantizedRMSNorm          (if patch_rmsnorm)

    Then, if patch_inner_ops=True (the default), replace
    LlamaAttention.forward and LlamaMLP.forward on each instance with
    versions that route QK^T, attn @ V, and silu(gate)*up through the Q-grid.

    Activation patches (SiLU, softmax) are wired separately via
    install_activation_patches. The model should be loaded with
    attn_implementation="eager" so F.softmax actually fires (SDPA bypasses it).
    """
    for name, child in list(model.named_children()):
        if patch_linear and isinstance(child, nn.Linear):
            setattr(model, name, QuantizedLinear(child, scale=scale))
            continue
        if (
            patch_rmsnorm
            and hasattr(child, "weight")
            and hasattr(child, "variance_epsilon")
            and child.weight.dim() == 1
        ):
            setattr(model, name, QuantizedRMSNorm(child, scale=scale, T=T))
            continue
        patch_model_inplace(
            child, scale=scale, T=T,
            patch_linear=patch_linear, patch_rmsnorm=patch_rmsnorm,
            patch_inner_ops=False,  # only do this once at the top level
        )
    # After recursive replacement of leaves, replace forwards on attention/MLP modules.
    if patch_inner_ops:
        patch_inner_ops_inplace(model, scale=scale)


def install_activation_patches(
    scale: float = DEFAULT_SCALE,
    T: int = DEFAULT_TABLE_T,
    patch_silu: bool = True,
    patch_softmax: bool = True,
    silu_x_max: float = DEFAULT_SILU_X_MAX,
    softmax_exp_clip: float = DEFAULT_EXP_CLIP,
):
    """
    Monkey-patch torch.nn.functional.silu and torch.nn.functional.softmax with the
    quantized versions. Affects all model code that goes through F.silu / F.softmax.
    Returns the originals so the caller can restore them after a run.
    """
    orig_silu = F.silu
    orig_softmax = F.softmax

    if patch_silu:
        qsilu = QuantizedSiLU(scale=scale, T=T, x_max=silu_x_max)

        def patched_silu(x, inplace: bool = False):  # noqa: ARG001
            return qsilu(x)

        F.silu = patched_silu

    if patch_softmax:
        qsoftmax = QuantizedSoftmax(scale=scale, T=T, exp_clip=softmax_exp_clip)

        def patched_softmax(x, dim: int = -1, _stacklevel=None, dtype=None):  # noqa: ARG001
            out = qsoftmax(x, dim=dim)
            if dtype is not None:
                out = out.to(dtype)
            return out

        F.softmax = patched_softmax

    return orig_silu, orig_softmax


def restore_activation_patches(orig_silu, orig_softmax) -> None:
    F.silu = orig_silu
    F.softmax = orig_softmax


# ----- Smoke test ------------------------------------------------------------

def smoke_test():
    """
    Load Llama-2-7B in FP16, patch with quantized modules, run one forward pass
    on a short input, and report the KL divergence between FP16 baseline logits
    and quantized logits at the last position.

    The baseline is FP16 (not FP32) because that's the precision the inference
    transcript is actually generated at. We want to measure the *additional*
    drift introduced by Q3.12 + table lookups on top of FP16's own noise.
    """
    import os
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = os.environ.get("MODEL_PATH", "meta-llama/Llama-2-7b-hf")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Reference: pure FP16 forward pass.
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16
    ).cuda()
    ref_model.eval()

    prompt = "The quick brown fox jumps over the lazy"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        fp16_logits = ref_model(**inputs).logits[0, -1].float()
    del ref_model
    torch.cuda.empty_cache()

    # Quantized: load fresh in FP16, patch with int64-internal quantized modules.
    q_model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16
    ).cuda()
    q_model.eval()
    patch_model_inplace(q_model)
    install_activation_patches()
    # On the first forward pass, verify that FP64 matmul really is bit-exact.
    QuantizedLinear._verify_int = True

    with torch.no_grad():
        q_logits = q_model(**inputs).logits[0, -1].float()

    fp16_probs = F.softmax(fp16_logits, dim=-1)
    q_probs = F.softmax(q_logits, dim=-1)

    kl = (fp16_probs * (fp16_probs.clamp_min(1e-12).log() - q_probs.clamp_min(1e-12).log())).sum().item()
    top_match = (fp16_probs.argmax() == q_probs.argmax()).item()

    print(f"KL(FP16 || quantized) at last position: {kl:.6f}")
    print(f"Top-1 next-token agreement: {top_match}")
    print(f"FP16 top-5: {[tokenizer.decode([t]) for t in fp16_probs.topk(5).indices.tolist()]}")
    print(f"Q top-5:    {[tokenizer.decode([t]) for t in q_probs.topk(5).indices.tolist()]}")


# ----- Ablation: which patch breaks Llama-7B? --------------------------------

def smoke_ablation(prompt: str = "The quick brown fox jumps over the lazy"):
    """
    For each of {linear, rmsnorm, silu, softmax}, run the smoke test with only
    that operation patched (rest left FP16), then a final "everything patched"
    run. Reports KL(FP16 || quantized) and top-1 agreement for each.
    """
    import os, copy
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = os.environ.get("MODEL_PATH", "meta-llama/Llama-2-7b-hf")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print(f"Loading {model_id} in FP16 (reference)...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16
    ).cuda()
    ref_model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        fp16_logits = ref_model(**inputs).logits[0, -1].float()
    fp16_probs = F.softmax(fp16_logits, dim=-1)
    fp16_top5 = [tokenizer.decode([t]) for t in fp16_probs.topk(5).indices.tolist()]
    print(f"FP16 reference top-5: {fp16_top5}")

    del ref_model
    torch.cuda.empty_cache()

    configs = [
        # (name, patch_linear, patch_rmsnorm, patch_silu, patch_softmax)
        ("linear only",                 True,  False, False, False),
        ("rmsnorm only",                False, True,  False, False),
        ("silu only",                   False, False, True,  False),
        ("softmax only",                False, False, False, True),
        ("linear + rmsnorm",            True,  True,  False, False),
        ("linear + silu",               True,  False, True,  False),
        ("linear + softmax",            True,  False, False, True),
        ("all patches (full quantize)", True,  True,  True,  True),
    ]

    for (name, lin, rms, silu, sm) in configs:
        # Fresh model each run so previous patches don't leak
        q_model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16
        ).cuda()
        q_model.eval()
        patch_model_inplace(q_model, patch_linear=lin, patch_rmsnorm=rms)
        orig_silu, orig_softmax = install_activation_patches(
            patch_silu=silu, patch_softmax=sm
        )

        with torch.no_grad():
            q_logits = q_model(**inputs).logits[0, -1].float()
        q_probs = F.softmax(q_logits, dim=-1)

        kl = (fp16_probs * (
            fp16_probs.clamp_min(1e-12).log() - q_probs.clamp_min(1e-12).log()
        )).sum().item()
        match = (fp16_probs.argmax() == q_probs.argmax()).item()
        top5 = [tokenizer.decode([t]) for t in q_probs.topk(5).indices.tolist()]
        print(f"\n[{name}]")
        print(f"  KL: {kl:.4f}    top-1 match: {match}")
        print(f"  top-5: {top5}")

        # Cleanup
        restore_activation_patches(orig_silu, orig_softmax)
        del q_model
        torch.cuda.empty_cache()


# ----- Per-op local precision diagnostic -------------------------------------

def per_op_precision_check(prompts: Optional[list] = None):
    """
    Fast diagnostic: for each patched op, measure local quantization error
    by running both FP16 and quantized versions on the *same input* (capturing
    the FP16 input via a forward hook on the reference model and replaying it
    through the matching quantized op). This isolates per-op contribution
    without error compounding.

    Reports relative MSE per op type (Linear, RMSNorm, SiLU, Softmax) and
    typical magnitudes.
    """
    import os
    from collections import defaultdict
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if prompts is None:
        prompts = [
            "The capital of France is Paris. The capital of Germany is",
            "Once upon a time, in a galaxy far far away, there lived a",
        ]

    model_id = os.environ.get("MODEL_PATH", "meta-llama/Llama-2-7b-hf")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Capture per-op (input, FP16-output) pairs via hooks on the FP16 model.
    fp_model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16
    ).cuda()
    fp_model.eval()

    captures = []  # list of (op_type, input_tensor, fp_output)

    def make_hook(op_type, name):
        def fn(mod, inp, out):
            captures.append((op_type, name, inp[0].detach().clone(), out.detach().clone()))
        return fn

    handles = []
    for name, mod in fp_model.named_modules():
        if isinstance(mod, nn.Linear):
            handles.append(mod.register_forward_hook(make_hook("Linear", name)))
        elif hasattr(mod, "variance_epsilon") and hasattr(mod, "weight") and mod.weight.dim() == 1:
            handles.append(mod.register_forward_hook(make_hook("RMSNorm", name)))

    for p in prompts:
        ids = tokenizer(p, return_tensors="pt").input_ids.cuda()
        with torch.no_grad():
            fp_model(ids)

    for h in handles:
        h.remove()
    del fp_model
    torch.cuda.empty_cache()

    # For each captured (input, fp_output), run the quantized version
    # of that op type on the same input and compare.
    # We need to find the corresponding module in a freshly-patched model.
    q_model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16
    ).cuda()
    q_model.eval()
    patch_model_inplace(q_model)
    install_activation_patches()

    # Map module names to patched module instances
    name_to_mod = dict(q_model.named_modules())

    stats = defaultdict(list)  # op_type -> list of relative MSE
    for op_type, name, x, fp_out in captures:
        q_mod = name_to_mod.get(name)
        if q_mod is None:
            continue
        with torch.no_grad():
            q_out = q_mod(x)
        # relative MSE: |q - fp|^2 / max(|fp|^2, eps)
        rel_mse = ((q_out.float() - fp_out.float()) ** 2).mean() / (fp_out.float() ** 2).mean().clamp_min(1e-12)
        stats[op_type].append(rel_mse.item())

    print(f"\nPer-op local precision (relative MSE, log10):")
    print(f"{'op type':<12} {'n':>6} {'min':>10} {'median':>10} {'max':>10} {'mean':>10}")
    for op_type, vals in stats.items():
        import statistics
        if not vals:
            continue
        m, mn, mx, av = (
            statistics.median(vals),
            min(vals),
            max(vals),
            statistics.mean(vals),
        )
        import math
        def lg(x): return math.log10(max(x, 1e-30))
        print(f"{op_type:<12} {len(vals):>6} {lg(mn):>10.2f} {lg(m):>10.2f} {lg(mx):>10.2f} {lg(av):>10.2f}")


# ----- Per-submodule drift diagnostic ----------------------------------------

def submodule_drift_diagnostic(prompts: Optional[list] = None, ref_dtype: str = "fp32"):
    """
    For each named submodule in each decoder layer (input_layernorm, q_proj,
    k_proj, v_proj, o_proj, self_attn, post_attention_layernorm, gate_proj,
    up_proj, down_proj, mlp), capture both FP reference and quantized
    input/output and report:

      ||fp_out||         typical magnitude of this submodule's output
      ||q_out - fp_out|| absolute output divergence
      relative diff      ||q_out - fp_out|| / ||fp_out||
      input-amp factor   ||q_out - fp_out|| / ||q_in - fp_in||  (>1: amplifies upstream drift; <1: attenuates)

    Aggregates across all 32 decoder layers and all prompts, then groups by
    submodule type to give an "error budget" decomposition: which op introduces
    or amplifies divergence the most.
    """
    import os
    import math
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if prompts is None:
        prompts = [
            "The capital of France is Paris. The capital of Germany is",
            "Once upon a time, in a galaxy far far away, there lived a",
            "Machine learning is a subfield of artificial intelligence that focuses on",
            "The quick brown fox jumps over the lazy dog. The slow gray cat sleeps under",
        ]

    model_id = os.environ.get("MODEL_PATH", "meta-llama/Llama-2-7b-hf")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Submodules to hook (relative to each model.layers.i)
    submodule_suffixes = [
        "input_layernorm",
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "self_attn",
        "post_attention_layernorm",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
        "mlp",
    ]

    def make_hook(target_dict, name):
        def fn(m, i, o):
            # i may be empty tuple if the module was called with only kwargs
            if isinstance(i, tuple) and len(i) > 0:
                inp = i[0]
            elif isinstance(i, torch.Tensor):
                inp = i
            else:
                inp = None
            out = o[0] if isinstance(o, tuple) else o
            inp_clone = inp.detach().float().clone() if inp is not None else None
            target_dict[name] = (inp_clone, out.detach().float().clone())
        return fn

    dtype = {"fp16": torch.float16, "fp32": torch.float32, "fp64": torch.float64}[ref_dtype]
    print(f"Loading reference model ({ref_dtype.upper()}) and quantized model side by side...")
    fp_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).cuda()
    fp_model.eval()
    q_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).cuda()
    q_model.eval()
    patch_model_inplace(q_model)
    install_activation_patches()

    fp_name_to_mod = dict(fp_model.named_modules())
    q_name_to_mod = dict(q_model.named_modules())
    n_layers = len(fp_model.model.layers)

    # Build the list of full module names
    target_names = []
    for li in range(n_layers):
        for suffix in submodule_suffixes:
            target_names.append(f"model.layers.{li}.{suffix}")

    # Aggregate stats: per name, list of (fp_out_norm, out_diff_norm, in_diff_norm, fp_in_norm)
    from collections import defaultdict
    stats = defaultdict(list)

    for p_idx, p in enumerate(prompts):
        ids = tokenizer(p, return_tensors="pt").input_ids.cuda()

        # FP capture
        fp_captures = {}
        fp_handles = []
        for name in target_names:
            mod = fp_name_to_mod.get(name)
            if mod is not None:
                fp_handles.append(mod.register_forward_hook(make_hook(fp_captures, name)))
        with torch.no_grad():
            fp_model(ids)
        for h in fp_handles:
            h.remove()

        # Q capture
        q_captures = {}
        q_handles = []
        for name in target_names:
            mod = q_name_to_mod.get(name)
            if mod is not None:
                q_handles.append(mod.register_forward_hook(make_hook(q_captures, name)))
        with torch.no_grad():
            q_model(ids)
        for h in q_handles:
            h.remove()

        # Compute stats per submodule
        for name in target_names:
            if name not in fp_captures or name not in q_captures:
                continue
            fp_in, fp_out = fp_captures[name]
            q_in, q_out = q_captures[name]
            fp_out_norm = fp_out.norm().item()
            out_diff_norm = (q_out - fp_out).norm().item()
            if fp_in is not None and q_in is not None:
                in_diff_norm = (q_in - fp_in).norm().item()
                fp_in_norm = fp_in.norm().item()
            else:
                in_diff_norm = float("nan")
                fp_in_norm = float("nan")
            stats[name].append((fp_out_norm, out_diff_norm, in_diff_norm, fp_in_norm))

        del fp_captures, q_captures
        torch.cuda.empty_cache()

    # Group by submodule suffix (across all layers)
    by_suffix = defaultdict(list)
    for name in target_names:
        suffix = name.split(f"layers.{name.split('.')[2]}.", 1)[1]
        for tup in stats.get(name, []):
            by_suffix[suffix].append(tup)

    # Report
    print(f"\nSubmodule drift, aggregated across {n_layers} layers × {len(prompts)} prompts:")
    print(f"{'submodule':>30} {'||fp_out||':>12} {'||q-fp||':>12} {'rel diff':>10} {'amp':>10}")
    print("-" * 80)
    for suffix in submodule_suffixes:
        vals = by_suffix.get(suffix, [])
        if not vals:
            continue
        avg_fp = sum(v[0] for v in vals) / len(vals)
        avg_diff = sum(v[1] for v in vals) / len(vals)
        rel_diffs = [v[1] / v[0] for v in vals if v[0] > 0]
        avg_rel = sum(rel_diffs) / len(rel_diffs) if rel_diffs else 0.0
        amps = [v[1] / v[2] for v in vals if v[2] > 0 and not (v[2] != v[2])]  # skip NaN
        avg_amp = sum(amps) / len(amps) if amps else float("nan")
        print(f"{suffix:>30} {avg_fp:>12.3e} {avg_diff:>12.3e} {avg_rel:>10.3e} {avg_amp:>10.3f}")


def relative_noise_growth_diagnostic(prompts: Optional[list] = None, ref_dtype: str = "fp32",
                                     trace_suffix: str = "mlp", mode: str = "quantized"):
    """
    Same capture as submodule_drift_diagnostic but reports *relative* noise
    growth per submodule:

      rel_in  = ||q_in - fp_in||  / ||fp_in||
      rel_out = ||q_out - fp_out|| / ||fp_out||
      rel_amp = rel_out / rel_in    (>1: submodule grows relative noise;
                                     <1: submodule attenuates it)

    Two views:
      (A) aggregated by submodule type — which ops grow rel. noise on average.
      (B) layer-by-layer trace for one chosen submodule (default "mlp") so
          you can see where in the network depth the drift compounds.
    """
    import os
    from collections import defaultdict
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if prompts is None:
        prompts = [
            "The capital of France is Paris. The capital of Germany is",
            "Once upon a time, in a galaxy far far away, there lived a",
            "Machine learning is a subfield of artificial intelligence that focuses on",
            "The quick brown fox jumps over the lazy dog. The slow gray cat sleeps under",
        ]

    model_id = os.environ.get("MODEL_PATH", "meta-llama/Llama-2-7b-hf")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    submodule_suffixes = [
        "input_layernorm",
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "self_attn",
        "post_attention_layernorm",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
        "mlp",
    ]

    def make_hook(target_dict, name):
        def fn(m, i, o):
            if isinstance(i, tuple) and len(i) > 0:
                inp = i[0]
            elif isinstance(i, torch.Tensor):
                inp = i
            else:
                inp = None
            out = o[0] if isinstance(o, tuple) else o
            inp_clone = inp.detach().float().clone() if inp is not None else None
            target_dict[name] = (inp_clone, out.detach().float().clone())
        return fn

    dtype = {"fp16": torch.float16, "fp32": torch.float32, "fp64": torch.float64}[ref_dtype]
    print(f"Loading model once ({ref_dtype.upper()}), eager attention. FP-pass first, then patch, then Q-pass.")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, attn_implementation="eager"
    ).cuda()
    model.eval()

    name_to_mod = dict(model.named_modules())
    n_layers = len(model.model.layers)

    target_names = []
    for li in range(n_layers):
        for suffix in submodule_suffixes:
            target_names.append(f"model.layers.{li}.{suffix}")

    # Stage 1: run FP forward with hooks for every prompt, cache all submodule outputs.
    fp_captures_per_prompt = []
    for p in prompts:
        ids = tokenizer(p, return_tensors="pt").input_ids.cuda()
        fp_captures = {}
        fp_handles = []
        for name in target_names:
            mod = name_to_mod.get(name)
            if mod is not None:
                fp_handles.append(mod.register_forward_hook(make_hook(fp_captures, name)))
        with torch.no_grad():
            model(ids)
        for h in fp_handles:
            h.remove()
        fp_captures_per_prompt.append(fp_captures)
        torch.cuda.empty_cache()

    # Stage 2: either patch the model in place (mode='quantized') OR replace it
    # with a fresh FP16 copy (mode='fp16') for a no-Q-mechanism baseline.
    if mode == "quantized":
        print("Patching model in place (Linear -> QuantizedLinear, RMSNorm -> QuantizedRMSNorm)...")
        patch_model_inplace(model)
        install_activation_patches()
    elif mode in ("fp16", "fp32", "fp64", "bf16"):
        dtype_map = {
            "fp16": torch.float16,
            "fp32": torch.float32,
            "fp64": torch.float64,
            "bf16": torch.bfloat16,
        }
        print(f"Mode='{mode}': discarding {ref_dtype.upper()} model and loading a fresh "
              f"{mode.upper()} copy (no Q-grid mechanisms; pure dtype comparison)...")
        del model
        import gc; gc.collect()
        torch.cuda.empty_cache()
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype_map[mode], attn_implementation="eager"
        ).cuda()
        model.eval()
    else:
        raise ValueError(f"Unknown mode {mode!r}; expected 'quantized', 'fp16', 'fp32', 'fp64', or 'bf16'.")
    # named_modules() dictionary needs to be rebuilt because the model object changed.
    name_to_mod = dict(model.named_modules())

    # Stage 3: run Q forward with hooks, compare against cached FP outputs per submodule.
    per_name = defaultdict(list)
    for p, fp_captures in zip(prompts, fp_captures_per_prompt):
        ids = tokenizer(p, return_tensors="pt").input_ids.cuda()
        q_captures = {}
        q_handles = []
        for name in target_names:
            mod = name_to_mod.get(name)
            if mod is not None:
                q_handles.append(mod.register_forward_hook(make_hook(q_captures, name)))
        with torch.no_grad():
            model(ids)
        for h in q_handles:
            h.remove()

        # Per-layer residual norm = ||fp_input to input_layernorm at that layer||.
        # We normalize per-submodule absolute error by this to get a globally-comparable
        # error metric within each layer.
        residual_norm_by_layer = {}
        for li in range(n_layers):
            ln_name = f"model.layers.{li}.input_layernorm"
            if ln_name in fp_captures and fp_captures[ln_name][0] is not None:
                residual_norm_by_layer[li] = fp_captures[ln_name][0].norm().item()
            else:
                residual_norm_by_layer[li] = float("nan")

        for name in target_names:
            if name not in fp_captures or name not in q_captures:
                continue
            fp_in, fp_out = fp_captures[name]
            q_in, q_out = q_captures[name]
            fp_in_norm = fp_in.norm().item() if fp_in is not None else float("nan")
            fp_out_norm = fp_out.norm().item()
            in_diff = (q_in - fp_in).norm().item() if (fp_in is not None and q_in is not None) else float("nan")
            out_diff = (q_out - fp_out).norm().item()
            rel_in = (in_diff / fp_in_norm) if (fp_in_norm > 0) else float("nan")
            rel_out = (out_diff / fp_out_norm) if (fp_out_norm > 0) else float("nan")
            # Absolute error normalized by residual norm at this layer.
            layer_idx = int(name.split(".")[2])
            r_norm = residual_norm_by_layer.get(layer_idx, float("nan"))
            err_in_per_resid = (in_diff / r_norm) if (r_norm > 0 and r_norm == r_norm) else float("nan")
            err_out_per_resid = (out_diff / r_norm) if (r_norm > 0 and r_norm == r_norm) else float("nan")
            # in_diff and out_diff are the raw absolute errors ||Q - FP||_2.
            per_name[name].append((rel_in, rel_out, err_in_per_resid, err_out_per_resid,
                                   in_diff, out_diff))

        del q_captures
        torch.cuda.empty_cache()

    # (A) aggregated by submodule suffix
    by_suffix = defaultdict(list)
    for name, vals in per_name.items():
        suffix = name.split(f"layers.{name.split('.')[2]}.", 1)[1]
        by_suffix[suffix].extend(vals)

    print(f"\n[A] Relative noise growth by submodule, aggregated across {n_layers} layers x {len(prompts)} prompts:")
    print(f"{'submodule':>30} {'rel_in':>12} {'rel_out':>12} {'rel_amp':>10}")
    print("-" * 70)
    for suffix in submodule_suffixes:
        vals = [v for v in by_suffix.get(suffix, []) if not (v[0] != v[0]) and not (v[1] != v[1])]
        if not vals:
            continue
        avg_in = sum(v[0] for v in vals) / len(vals)
        avg_out = sum(v[1] for v in vals) / len(vals)
        amps = [v[1] / v[0] for v in vals if v[0] > 0]
        avg_amp = sum(amps) / len(amps) if amps else float("nan")
        print(f"{suffix:>30} {avg_in:>12.4e} {avg_out:>12.4e} {avg_amp:>10.3f}")

    # (B) layer-by-layer trace for chosen suffix
    print(f"\n[B] Layer-by-layer rel. noise trace for submodule '{trace_suffix}' (averaged across prompts):")
    print(f"{'layer':>5} {'rel_in':>12} {'rel_out':>12} {'rel_amp':>10}")
    print("-" * 50)
    for li in range(n_layers):
        name = f"model.layers.{li}.{trace_suffix}"
        vals = [v for v in per_name.get(name, []) if not (v[0] != v[0]) and not (v[1] != v[1])]
        if not vals:
            continue
        avg_in = sum(v[0] for v in vals) / len(vals)
        avg_out = sum(v[1] for v in vals) / len(vals)
        amps = [v[1] / v[0] for v in vals if v[0] > 0]
        avg_amp = sum(amps) / len(amps) if amps else float("nan")
        print(f"{li:>5d} {avg_in:>12.4e} {avg_out:>12.4e} {avg_amp:>10.3f}")

    # (C) save CSV of per-name, per-layer rel-noise; produce log-scale plot.
    import csv as _csv
    mode_suffix = "" if mode == "quantized" else f"_{mode}"
    csv_path = os.environ.get("REL_NOISE_CSV", f"/tmp/rel_noise{mode_suffix}.csv")
    rows_for_plot = {}  # suffix -> list of (layer, rel_in, rel_out, err_in_resid, err_out_resid, abs_in, abs_out)
    with open(csv_path, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["layer", "submodule", "rel_in", "rel_out", "err_in_per_resid", "err_out_per_resid",
                    "abs_err_in", "abs_err_out"])
        for li in range(n_layers):
            for suffix in submodule_suffixes:
                name = f"model.layers.{li}.{suffix}"
                vals = [v for v in per_name.get(name, [])
                        if not (v[0] != v[0]) and not (v[1] != v[1])]
                if not vals:
                    continue
                avg_in = sum(v[0] for v in vals) / len(vals)
                avg_out = sum(v[1] for v in vals) / len(vals)
                v_eir = [v[2] for v in vals if len(v) > 2 and v[2] == v[2]]
                v_eor = [v[3] for v in vals if len(v) > 3 and v[3] == v[3]]
                v_ai = [v[4] for v in vals if len(v) > 4 and v[4] == v[4]]
                v_ao = [v[5] for v in vals if len(v) > 5 and v[5] == v[5]]
                avg_eir = sum(v_eir) / len(v_eir) if v_eir else float("nan")
                avg_eor = sum(v_eor) / len(v_eor) if v_eor else float("nan")
                avg_ai = sum(v_ai) / len(v_ai) if v_ai else float("nan")
                avg_ao = sum(v_ao) / len(v_ao) if v_ao else float("nan")
                w.writerow([li, suffix, f"{avg_in:.6e}", f"{avg_out:.6e}",
                            f"{avg_eir:.6e}", f"{avg_eor:.6e}",
                            f"{avg_ai:.6e}", f"{avg_ao:.6e}"])
                rows_for_plot.setdefault(suffix, []).append(
                    (li, avg_in, avg_out, avg_eir, avg_eor, avg_ai, avg_ao))
    print(f"\nSaved per-layer per-submodule rel-noise CSV to {csv_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        png_path = os.environ.get("REL_NOISE_PNG", f"/tmp/rel_noise{mode_suffix}.png")
        # FP16 noise is much smaller than Q noise; lower the log clamp floor.
        log_floor = 1e-8 if mode != "quantized" else 1e-6
        fig, (ax_in, ax_out) = plt.subplots(1, 2, figsize=(14, 6.5), sharey=True)
        for suffix in submodule_suffixes:
            seq = rows_for_plot.get(suffix)
            if not seq:
                continue
            layers = [r[0] for r in seq]
            rel_in = [max(r[1], log_floor) for r in seq]
            rel_out = [max(r[2], log_floor) for r in seq]
            ax_in.plot(layers, rel_in, marker="o", markersize=3, linewidth=1.2, label=suffix)
            ax_out.plot(layers, rel_out, marker="o", markersize=3, linewidth=1.2, label=suffix)
        for ax, title in [
            (ax_in, "rel_in: rel-noise entering each submodule"),
            (ax_out, "rel_out: rel-noise leaving each submodule"),
        ]:
            ax.set_yscale("log")
            ax.set_xlabel("layer index")
            ax.set_title(title)
            ax.grid(True, which="both", alpha=0.3)
        ax_in.set_ylabel("rel-noise (||Q - FP|| / ||FP||)")
        ax_out.legend(fontsize=8, loc="upper right", ncol=2,
                      framealpha=0.85, borderpad=0.4)
        suptitle_label = (
            "Q-vs-FP" if mode == "quantized"
            else f"{mode.upper()}-vs-{ref_dtype.upper()}"
        )
        plt.suptitle(f"{suptitle_label} relative noise per submodule, per layer "
                     f"(averaged across {len(prompts)} prompts)", fontsize=11)
        plt.tight_layout()
        plt.savefig(png_path, dpi=130, bbox_inches="tight")
        print(f"Saved log-scale rel-noise plot to {png_path}")

        # Trajectory plot: sequential ordering through the forward pass,
        # with a line linking consecutive computations and points colored
        # by op type.
        traj_path = os.environ.get("REL_NOISE_TRAJECTORY_PNG", f"/tmp/rel_noise_trajectory{mode_suffix}.png")
        # Within-layer compute order (excluding redundant whole-block entries).
        seq_order = [
            "input_layernorm",
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "post_attention_layernorm",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ]
        sequence = []  # list of (step_idx, layer, submodule, rel_out)
        step_idx = 0
        for li in range(n_layers):
            for sm in seq_order:
                name = f"model.layers.{li}.{sm}"
                vals = [v for v in per_name.get(name, [])
                        if not (v[1] != v[1])]
                if not vals:
                    continue
                avg_out = sum(v[1] for v in vals) / len(vals)
                sequence.append((step_idx, li, sm, avg_out))
                step_idx += 1

        fig2, ax2 = plt.subplots(figsize=(16, 6.5))
        xs = [r[0] for r in sequence]
        ys = [max(r[3], log_floor) for r in sequence]
        ax2.plot(xs, ys, color="gray", linewidth=0.7, alpha=0.55, zorder=1)
        cmap_t = plt.get_cmap("tab10")
        color_map = {sm: cmap_t(i % 10) for i, sm in enumerate(seq_order)}
        for sm in seq_order:
            sm_xs = [r[0] for r in sequence if r[2] == sm]
            sm_ys = [max(r[3], log_floor) for r in sequence if r[2] == sm]
            ax2.scatter(sm_xs, sm_ys, color=color_map[sm], s=22,
                        label=sm, zorder=2, edgecolors="white", linewidths=0.4)
        ax2.set_yscale("log")
        ax2.set_xlabel("compute step (sequence through forward pass)")
        ax2.set_ylabel("rel-noise (||Q - FP|| / ||FP||)")
        ax2.set_title(f"{suptitle_label} rel-noise trajectory through forward pass, "
                      f"colored by op (avg across {len(prompts)} prompts)")
        # Layer boundary markers every 4 layers.
        per_layer = len(seq_order)
        ylim = ax2.get_ylim()
        for li in range(0, n_layers + 1, 4):
            xb = li * per_layer
            ax2.axvline(xb, color="black", linestyle="--", alpha=0.15, zorder=0)
            if li < n_layers:
                ax2.text(xb + per_layer / 2, ylim[1] * 0.7, f"L{li}",
                         fontsize=8, ha="center", va="top", color="dimgray")
        ax2.legend(fontsize=8, loc="lower right", ncol=2,
                   framealpha=0.85, borderpad=0.4)
        ax2.grid(True, which="both", alpha=0.25)
        plt.tight_layout()
        plt.savefig(traj_path, dpi=130, bbox_inches="tight")
        print(f"Saved trajectory plot to {traj_path}")

        # Per-residual trajectory: ||Q - FP|| / ||residual_FP|| at each step.
        # Globally-comparable denominator (||residual|| at the start of each layer)
        # within and across layers, so steps with smaller output shapes don't
        # look artificially noisier.
        resid_traj_path = os.environ.get(
            "REL_NOISE_PER_RESID_TRAJECTORY_PNG",
            f"/tmp/rel_noise_per_resid_trajectory{mode_suffix}.png",
        )
        sequence_resid = []  # (step_idx, layer, submodule, err_out_per_resid)
        step_idx = 0
        for li in range(n_layers):
            for sm in seq_order:
                name = f"model.layers.{li}.{sm}"
                vals = [v for v in per_name.get(name, [])
                        if len(v) > 3 and v[3] == v[3]]
                if not vals:
                    continue
                avg_eor = sum(v[3] for v in vals) / len(vals)
                sequence_resid.append((step_idx, li, sm, avg_eor))
                step_idx += 1

        if sequence_resid:
            fig3, ax3 = plt.subplots(figsize=(16, 6.5))
            xs3 = [r[0] for r in sequence_resid]
            ys3 = [max(r[3], log_floor) for r in sequence_resid]
            ax3.plot(xs3, ys3, color="gray", linewidth=0.7, alpha=0.55, zorder=1)
            for sm in seq_order:
                sm_xs = [r[0] for r in sequence_resid if r[2] == sm]
                sm_ys = [max(r[3], log_floor) for r in sequence_resid if r[2] == sm]
                ax3.scatter(sm_xs, sm_ys, color=color_map[sm], s=22,
                            label=sm, zorder=2, edgecolors="white", linewidths=0.4)
            ax3.set_yscale("log")
            ax3.set_xlabel("compute step (sequence through forward pass)")
            ax3.set_ylabel("||Q - FP||₂ / ||residual_FP||₂ (at start of layer)")
            ax3.set_title(f"{suptitle_label} absolute error normalized by residual norm, "
                          f"colored by op (avg across {len(prompts)} prompts)")
            ylim3 = ax3.get_ylim()
            for li in range(0, n_layers + 1, 4):
                xb = li * per_layer
                ax3.axvline(xb, color="black", linestyle="--", alpha=0.15, zorder=0)
                if li < n_layers:
                    ax3.text(xb + per_layer / 2, ylim3[1] * 0.7, f"L{li}",
                             fontsize=8, ha="center", va="top", color="dimgray")
            ax3.legend(fontsize=8, loc="lower right", ncol=2,
                       framealpha=0.85, borderpad=0.4)
            ax3.grid(True, which="both", alpha=0.25)
            plt.tight_layout()
            plt.savefig(resid_traj_path, dpi=130, bbox_inches="tight")
            print(f"Saved per-residual trajectory plot to {resid_traj_path}")
    except ImportError:
        print("matplotlib not available; skipped plot. CSV saved.")


def submodule_ablation_diagnostic(prompts: Optional[list] = None,
                                  targets: Optional[list] = None):
    """
    Ablation sanity check. For each submodule suffix in `targets`, run the
    quantized model with that submodule's output REPLACED by the FP reference
    output at every decoder layer. Compare the final-logits divergence under
    each ablation to the no-ablation baseline.

    Interpretation:
      large drop  => that submodule is a real error injector (worth
                     spending more bits / per-channel scales there)
      small drop  => that submodule is just transmitting upstream drift
    """
    import os
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if prompts is None:
        prompts = [
            "The capital of France is Paris. The capital of Germany is",
            "Once upon a time, in a galaxy far far away, there lived a",
            "Machine learning is a subfield of artificial intelligence that focuses on",
            "The quick brown fox jumps over the lazy dog. The slow gray cat sleeps under",
        ]
    if targets is None:
        targets = [
            "input_layernorm",
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.o_proj",
            "self_attn",
            "post_attention_layernorm",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
            "mlp",
        ]

    model_id = os.environ.get("MODEL_PATH", "meta-llama/Llama-2-7b-hf")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print("Loading model once (FP32), eager attention. FP-pass + cache, then patch, then Q baseline + ablations.")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32, attn_implementation="eager"
    ).cuda()
    model.eval()
    name_to_mod = dict(model.named_modules())
    n_layers = len(model.model.layers)

    def capture_hook(cache, name):
        def fn(m, i, o):
            out = o[0] if isinstance(o, tuple) else o
            cache[name] = out.detach().float().clone()
        return fn

    def splice_hook(cache, name):
        def fn(m, i, o):
            target = cache[name]
            if isinstance(o, tuple):
                return (target.to(o[0].dtype),) + o[1:]
            return target.to(o.dtype)
        return fn

    all_target_names = [f"model.layers.{li}.{suf}" for li in range(n_layers) for suf in targets]

    # Stage 1: FP forward per prompt, cache submodule outputs and final logits.
    fp_caches = []
    fp_logits_list = []
    for p in prompts:
        ids = tokenizer(p, return_tensors="pt").input_ids.cuda()
        fp_cache = {}
        fp_handles = []
        for name in all_target_names:
            mod = name_to_mod.get(name)
            if mod is not None:
                fp_handles.append(mod.register_forward_hook(capture_hook(fp_cache, name)))
        with torch.no_grad():
            fp_logits = model(ids).logits.detach().float().cpu()
        for h in fp_handles:
            h.remove()
        fp_caches.append(fp_cache)
        fp_logits_list.append(fp_logits)
        torch.cuda.empty_cache()

    # Stage 2: patch in place.
    print(f"Patching model in place. Running quantized baseline + {len(targets)} ablations...")
    patch_model_inplace(model)
    install_activation_patches()
    name_to_mod = dict(model.named_modules())

    # Stage 3: per prompt, Q baseline + one ablation per target suffix.
    from collections import defaultdict
    abl_diffs = defaultdict(list)
    baseline_diffs = []

    for p, fp_cache, fp_logits_cpu in zip(prompts, fp_caches, fp_logits_list):
        ids = tokenizer(p, return_tensors="pt").input_ids.cuda()
        fp_logits = fp_logits_cpu.cuda()
        fp_logits_norm = fp_logits.norm().item()

        with torch.no_grad():
            q_logits = model(ids).logits.detach().float()
        baseline_diffs.append((q_logits - fp_logits).norm().item() / fp_logits_norm)

        for suf in targets:
            handles = []
            for li in range(n_layers):
                name = f"model.layers.{li}.{suf}"
                mod = name_to_mod.get(name)
                if mod is not None and name in fp_cache:
                    handles.append(mod.register_forward_hook(splice_hook(fp_cache, name)))
            with torch.no_grad():
                logits_abl = model(ids).logits.detach().float()
            for h in handles:
                h.remove()
            abl_diffs[suf].append((logits_abl - fp_logits).norm().item() / fp_logits_norm)

        del fp_cache
        torch.cuda.empty_cache()

    avg_baseline = sum(baseline_diffs) / len(baseline_diffs)
    print(f"{'ablated submodule':>25} {'rel. logits drift':>20} {'reduction vs baseline':>24}")
    print("-" * 75)
    print(f"{'(none, baseline)':>25} {avg_baseline:>20.4e} {'-':>24}")
    for suf in targets:
        vals = abl_diffs.get(suf, [])
        if not vals:
            continue
        avg = sum(vals) / len(vals)
        red = 1.0 - avg / avg_baseline if avg_baseline > 0 else 0.0
        print(f"{suf:>25} {avg:>20.4e} {red*100:>22.1f}%")


def attention_breakdown_diagnostic(prompts: Optional[list] = None, ref_dtype: str = "fp32"):
    """
    Fine-grained rel-noise inside the attention block:

      q_proj.out / k_proj.out / v_proj.out   (already hooked as named modules)
      softmax.in  = scaled QK^T              (captured by wrapping F.softmax)
      softmax.out = attention weights        (captured by wrapping F.softmax)
      o_proj.in   = attn_weights @ V reshape (already hooked as named module input)
      o_proj.out                              (already hooked as named module output)

    Notes:
      - Forces attn_implementation="eager" so F.softmax is actually called (SDPA
        bypasses it). Without this, softmax captures would never fire.
      - QK^T and attn @ V are `torch.matmul`, not `nn.Linear`, so they run at the
        model's float dtype in both FP and Q runs. The "rel-noise injection" we
        see at those steps is bilinear amplification of upstream q/k/v noise,
        not new quantization error.
      - call_index of F.softmax == layer_index, since each layer calls softmax
        exactly once in order.
    """
    import os
    import statistics
    from collections import defaultdict
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if prompts is None:
        prompts = [
            "The capital of France is Paris. The capital of Germany is",
            "Once upon a time, in a galaxy far far away, there lived a",
            "Machine learning is a subfield of artificial intelligence that focuses on",
            "The quick brown fox jumps over the lazy dog. The slow gray cat sleeps under",
        ]

    model_id = os.environ.get("MODEL_PATH", "meta-llama/Llama-2-7b-hf")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    dtype = {"fp16": torch.float16, "fp32": torch.float32, "fp64": torch.float64}[ref_dtype]
    print(f"Loading model once ({ref_dtype.upper()}), eager attention so F.softmax is observable.")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, attn_implementation="eager"
    ).cuda()
    model.eval()
    name_to_mod = dict(model.named_modules())
    n_layers = len(model.model.layers)

    submodule_suffixes = [
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    ]
    target_names = [f"model.layers.{li}.{s}" for li in range(n_layers) for s in submodule_suffixes]

    def make_hook(target_dict, name):
        def fn(m, i, o):
            inp = i[0] if isinstance(i, tuple) and len(i) > 0 else None
            out = o[0] if isinstance(o, tuple) else o
            target_dict[name] = (
                inp.detach().float().clone() if inp is not None else None,
                out.detach().float().clone(),
            )
        return fn

    # FP run: wrap F.softmax to capture per call.
    orig_softmax = F.softmax

    def fp_wrap(stash):
        def wrapped(x, dim=-1, _stacklevel=None, dtype=None):
            out = orig_softmax(x, dim=dim, dtype=dtype) if dtype is not None else orig_softmax(x, dim=dim)
            stash.append((x.detach().float().clone(), out.detach().float().clone()))
            return out
        return wrapped

    fp_subhooks_per_p, fp_softmax_per_p = [], []
    for p in prompts:
        fp_caps = {}
        fp_handles = [name_to_mod[n].register_forward_hook(make_hook(fp_caps, n))
                      for n in target_names if n in name_to_mod]
        stash = []
        F.softmax = fp_wrap(stash)
        ids = tokenizer(p, return_tensors="pt").input_ids.cuda()
        with torch.no_grad():
            model(ids)
        F.softmax = orig_softmax
        for h in fp_handles:
            h.remove()
        fp_subhooks_per_p.append(fp_caps)
        fp_softmax_per_p.append(stash)
        torch.cuda.empty_cache()

    # Patch in place, install activation patches (F.softmax now -> QuantizedSoftmax wrapper).
    print("Patching in place + activation patches. Running quantized pass with capture...")
    patch_model_inplace(model)
    install_activation_patches()
    q_softmax = F.softmax  # this is now the patched_softmax wrapper installed by install_activation_patches
    name_to_mod = dict(model.named_modules())

    def q_wrap(stash):
        def wrapped(x, dim=-1, _stacklevel=None, dtype=None):
            out = q_softmax(x, dim=dim, dtype=dtype) if dtype is not None else q_softmax(x, dim=dim)
            stash.append((x.detach().float().clone(), out.detach().float().clone()))
            return out
        return wrapped

    rel_per_step = defaultdict(list)

    def rel(fp_t, q_t):
        n = fp_t.norm().item()
        return ((q_t - fp_t).norm().item() / n) if n > 0 else float("nan")

    for p, fp_caps, fp_sm in zip(prompts, fp_subhooks_per_p, fp_softmax_per_p):
        q_caps = {}
        q_handles = [name_to_mod[n].register_forward_hook(make_hook(q_caps, n))
                     for n in target_names if n in name_to_mod]
        q_stash = []
        F.softmax = q_wrap(q_stash)
        ids = tokenizer(p, return_tensors="pt").input_ids.cuda()
        with torch.no_grad():
            model(ids)
        F.softmax = q_softmax
        for h in q_handles:
            h.remove()

        for li in range(n_layers):
            qn = f"model.layers.{li}.self_attn.q_proj"
            kn = f"model.layers.{li}.self_attn.k_proj"
            vn = f"model.layers.{li}.self_attn.v_proj"
            on = f"model.layers.{li}.self_attn.o_proj"

            if qn in fp_caps and qn in q_caps:
                rel_per_step["q_proj.out"].append(rel(fp_caps[qn][1], q_caps[qn][1]))
            if kn in fp_caps and kn in q_caps:
                rel_per_step["k_proj.out"].append(rel(fp_caps[kn][1], q_caps[kn][1]))
            if vn in fp_caps and vn in q_caps:
                rel_per_step["v_proj.out"].append(rel(fp_caps[vn][1], q_caps[vn][1]))
            if li < len(fp_sm) and li < len(q_stash):
                fp_in, fp_out = fp_sm[li]
                q_in, q_out = q_stash[li]
                rel_per_step["softmax.in (scale*QK^T)"].append(rel(fp_in, q_in))
                rel_per_step["softmax.out (attn weights)"].append(rel(fp_out, q_out))
            if on in fp_caps and on in q_caps:
                fp_oi = fp_caps[on][0]
                q_oi = q_caps[on][0]
                if fp_oi is not None and q_oi is not None:
                    rel_per_step["o_proj.in (attn @ V)"].append(rel(fp_oi, q_oi))
                rel_per_step["o_proj.out"].append(rel(fp_caps[on][1], q_caps[on][1]))

        del q_caps
        torch.cuda.empty_cache()

    print(f"\nAttention-internal rel-noise (averaged across {n_layers} layers x {len(prompts)} prompts):")
    print(f"{'step':>30} {'rel (mean)':>12} {'rel (median)':>14} {'rel (max)':>12}")
    print("-" * 75)
    for step in [
        "q_proj.out", "k_proj.out", "v_proj.out",
        "softmax.in (scale*QK^T)", "softmax.out (attn weights)",
        "o_proj.in (attn @ V)", "o_proj.out",
    ]:
        vals = [v for v in rel_per_step.get(step, []) if v == v]
        if not vals:
            continue
        avg = sum(vals) / len(vals)
        med = statistics.median(vals)
        mx = max(vals)
        print(f"{step:>30} {avg:>12.4f} {med:>14.4f} {mx:>12.4f}")


# ----- Per-layer drift diagnostic --------------------------------------------

def layer_drift_diagnostic(prompts: Optional[list] = None, ref_dtype: str = "fp32"):
    """
    Test the layer-compounding hypothesis. For each LlamaDecoderLayer output,
    compare a high-precision reference (FP32 by default) vs the quantized
    hidden state and report:
      - relative MSE (||q - ref||^2 / ||ref||^2)
      - cosine similarity

    Using FP32 (or FP64) as the reference isolates quantization-specific drift
    from FP16's own accumulation-order noise. FP32 has a 23-bit mantissa which
    is essentially deterministic at our hidden-state magnitudes (well below
    the FP16 ~10^-4 noise floor).

    If the divergence trajectory grows monotonically with depth, layer-
    compounding is the source of the remaining unexplained-info budget.
    """
    import os
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if prompts is None:
        prompts = [
            "The capital of France is Paris. The capital of Germany is",
            "Once upon a time, in a galaxy far far away, there lived a",
            "Machine learning is a subfield of artificial intelligence that focuses on",
            "The quick brown fox jumps over the lazy dog. The slow gray cat sleeps under",
        ]

    model_id = os.environ.get("MODEL_PATH", "meta-llama/Llama-2-7b-hf")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    fp_states = []  # list of (prompt_idx, layer_idx, hidden_state)
    q_states = []

    def capture_hook(target_list, layer_idx, prompt_idx):
        def fn(mod, inp, out):
            # LlamaDecoderLayer output is a tuple; first element is hidden_states
            h = out[0] if isinstance(out, tuple) else out
            target_list.append((prompt_idx, layer_idx, h.detach().float().clone()))
        return fn

    # High-precision reference run (FP32 by default; FP64 optional)
    dtype = {"fp16": torch.float16, "fp32": torch.float32, "fp64": torch.float64}[ref_dtype]
    print(f"Loading reference model in {ref_dtype.upper()}...")
    fp_model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype
    ).cuda()
    fp_model.eval()
    layers = fp_model.model.layers
    handles = [
        layers[i].register_forward_hook(capture_hook(fp_states, i, None))
        for i in range(len(layers))
    ]
    for p_idx, p in enumerate(prompts):
        ids = tokenizer(p, return_tensors="pt").input_ids.cuda()
        # Replace placeholder None prompt_idx via closure trick: use a wrapper
        fp_states_per_prompt = []
        local_handles = [
            l.register_forward_hook(
                (lambda li: (lambda m, i, o: fp_states_per_prompt.append(
                    (p_idx, li, (o[0] if isinstance(o, tuple) else o).detach().float().clone())
                )))(li)
            )
            for li, l in enumerate(layers)
        ]
        with torch.no_grad():
            fp_model(ids)
        for h in local_handles:
            h.remove()
        fp_states.extend(fp_states_per_prompt)
    for h in handles:
        h.remove()
    del fp_model
    torch.cuda.empty_cache()

    # Quantized run
    q_model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16
    ).cuda()
    q_model.eval()
    patch_model_inplace(q_model)
    install_activation_patches()
    layers = q_model.model.layers
    for p_idx, p in enumerate(prompts):
        ids = tokenizer(p, return_tensors="pt").input_ids.cuda()
        q_states_per_prompt = []
        local_handles = [
            l.register_forward_hook(
                (lambda li: (lambda m, i, o: q_states_per_prompt.append(
                    (p_idx, li, (o[0] if isinstance(o, tuple) else o).detach().float().clone())
                )))(li)
            )
            for li, l in enumerate(layers)
        ]
        with torch.no_grad():
            q_model(ids)
        for h in local_handles:
            h.remove()
        q_states.extend(q_states_per_prompt)
    del q_model
    torch.cuda.empty_cache()

    # Sort and align
    fp_dict = {(p, l): h for p, l, h in fp_states}
    q_dict = {(p, l): h for p, l, h in q_states}

    n_layers = max(l for _, l, _ in fp_states) + 1

    print(f"\nPer-layer drift (FP16 vs quantized), averaged across {len(prompts)} prompts:")
    print(f"{'layer':>6} {'rel_mse':>12} {'cosine':>10} {'mean_diff':>12} {'mean_fp':>12}")
    for li in range(n_layers):
        rel_mses = []
        cosines = []
        diffs = []
        mags = []
        for pi in range(len(prompts)):
            fp = fp_dict.get((pi, li))
            q = q_dict.get((pi, li))
            if fp is None or q is None:
                continue
            diff = (q - fp)
            rel_mse = (diff ** 2).mean() / (fp ** 2).mean().clamp_min(1e-12)
            cos = torch.nn.functional.cosine_similarity(
                fp.flatten().unsqueeze(0), q.flatten().unsqueeze(0)
            ).item()
            rel_mses.append(rel_mse.item())
            cosines.append(cos)
            diffs.append(diff.abs().mean().item())
            mags.append(fp.abs().mean().item())
        if rel_mses:
            avg_rel_mse = sum(rel_mses) / len(rel_mses)
            avg_cos = sum(cosines) / len(cosines)
            avg_diff = sum(diffs) / len(diffs)
            avg_mag = sum(mags) / len(mags)
            print(f"{li:>6} {avg_rel_mse:>12.4e} {avg_cos:>10.6f} {avg_diff:>12.4e} {avg_mag:>12.4e}")


# ----- Unexplained information rate (zkllm-entropy framework) ---------------

def _noise_corrected_probs(logits_fp: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Compute q(o) = win(o) / sum_j win(j), where the per-token win probability
    follows the zkllm-entropy code's formula (see verify_entropy.py):

        win(i) = 1 - Phi((v* - l_i) / sigma_eff)

    where v* = max(l) and Phi is the standard normal CDF. For the argmax
    token (v* - l_i = 0), win = 0.5; for tokens far from the max, win → 0.
    Normalized over the vocab gives the noise-corrected selection probability.

    Note: the README on the zkllm-entropy repo currently writes the formula
    without the (1 - ...), which would flip the semantics; the source-of-truth
    is python/verify_entropy.py and src/.../prover code which use 1 - Phi(...).
    """
    v_star = logits_fp.max(dim=-1, keepdim=True).values
    z = (v_star - logits_fp) / sigma
    # 1 - Phi(z) = 0.5 * erfc(z / sqrt(2))
    win = 0.5 * torch.erfc(z / (2.0 ** 0.5))
    return win / win.sum(dim=-1, keepdim=True).clamp_min(1e-30)


def _wikitext_prompts(n_chunks: int = 16, chunk_tokens: int = 128) -> list:
    """Pull WikiText-2 test split, split into chunks for a larger test sample."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    # Concatenate paragraphs, filter trivially short lines
    big = " ".join(line for line in ds["text"] if len(line.strip()) > 32)
    # Cut into chunks of approximately chunk_tokens-worth of characters
    # (4 chars/token rough estimate; we re-tokenize inside the test)
    chars_per_chunk = chunk_tokens * 5
    chunks = []
    for i in range(n_chunks):
        start = i * chars_per_chunk
        end = start + chars_per_chunk
        if end >= len(big):
            break
        chunks.append(big[start:end])
    return chunks


def unexplained_info(
    prompts: Optional[list] = None,
    sigma: float = 0.5,
    n_positions: Optional[int] = None,
    calibrate: bool = False,
    scale: int = DEFAULT_SCALE,
    T: int = DEFAULT_TABLE_T,
    wikitext: bool = False,
    n_chunks: int = 16,
    chunk_tokens: int = 128,
):
    """
    Direct computation of the unexplained-information / covert-channel-capacity
    bound from the zkllm-entropy framework.

    For each position:
      1. FP16 reference model produces logits l_fp.
      2. Quantized model produces logits l_q; we take its argmax as the
         prover's emitted token o.
      3. Compute q(o) under l_fp with Gaussian noise model at sigma.
      4. Surprisal at this position = -log2 q(o).

    Report total surprisal and bits/token. This is the bits-of-covert-capacity
    bound: at most this many bits could be encoded in the output that the FP16
    reference + noise model can't already explain.
    """
    import math
    import os
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if prompts is None:
        if wikitext:
            print(f"Pulling {n_chunks} chunks of ~{chunk_tokens} tokens from WikiText-2 test split...")
            prompts = _wikitext_prompts(n_chunks=n_chunks, chunk_tokens=chunk_tokens)
            print(f"  loaded {len(prompts)} chunks")
        else:
            prompts = [
                "The capital of France is Paris. The capital of Germany is",
                "Once upon a time, in a galaxy far far away, there lived a",
                "Machine learning is a subfield of artificial intelligence that focuses on",
                "The quick brown fox jumps over the lazy dog. The slow gray cat sleeps under",
            ]

    model_id = os.environ.get("MODEL_PATH", "meta-llama/Llama-2-7b-hf")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print(f"Loading reference FP32 model (eager attention)...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32, attn_implementation="eager"
    ).cuda()
    ref_model.eval()

    all_fp_logits = []
    all_input_ids = []
    for p in prompts:
        ids = tokenizer(p, return_tensors="pt").input_ids.cuda()
        with torch.no_grad():
            logits = ref_model(ids).logits[0].float()  # (seq, vocab)
        all_fp_logits.append(logits)
        all_input_ids.append(ids[0])
    del ref_model
    torch.cuda.empty_cache()

    import math
    print(f"Loading quantized model (scale=2^{int(math.log2(scale))}, T=2^{int(math.log2(T))})...")
    q_model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32, attn_implementation="eager"
    ).cuda()
    q_model.eval()
    patch_model_inplace(q_model, scale=scale, T=T)
    install_activation_patches(scale=scale, T=T)

    all_q_logits = []
    for ids in all_input_ids:
        with torch.no_grad():
            logits = q_model(ids.unsqueeze(0)).logits[0].float()  # (seq, vocab)
        all_q_logits.append(logits)
    del q_model
    torch.cuda.empty_cache()

    # Self-calibrate sigma if requested. Gibbs' inequality lets the prover use
    # any estimate Q; minimizing the bound over the Gaussian-σ family gives the
    # tightest valid upper bound on covert capacity for this output sample.
    #
    # The objective is *not* convex in sigma (empirically — small sigma
    # heavily penalizes mismatched tokens, large sigma penalizes correct
    # tokens uniformly, and the crossover region can have multiple local
    # minima depending on the logit landscape). Use a coarse multiplicative
    # grid for global search + scipy's bounded minimize_scalar for local
    # refinement around the grid's best point.
    if calibrate:
        from scipy.optimize import minimize_scalar  # type: ignore

        def total_q_surprisal(s: float) -> float:
            if s <= 0:
                return float("inf")
            tot = 0.0
            for fp_logits, q_logits in zip(all_fp_logits, all_q_logits):
                q_tokens = q_logits.argmax(dim=-1)
                probs = _noise_corrected_probs(fp_logits, s)
                q_of_q = probs.gather(-1, q_tokens.unsqueeze(-1)).squeeze(-1)
                tot += -torch.log2(q_of_q.clamp_min(1e-30)).sum().item()
            return tot

        # Coarse multiplicative grid over [0.05, 10.0]
        grid = []
        s = 0.05
        while s <= 10.0:
            grid.append(s)
            s *= 1.10
        grid_vals = [(s, total_q_surprisal(s)) for s in grid]
        best_s, best_v = min(grid_vals, key=lambda t: t[1])
        # Local refinement: golden-section search in a tight bracket around best_s
        lo = best_s / 1.15
        hi = best_s * 1.15
        result = minimize_scalar(
            total_q_surprisal, bounds=(lo, hi), method="bounded",
            options={"xatol": 1e-4},
        )
        if result.fun < best_v:
            sigma_opt = float(result.x)
            best_v = result.fun
        else:
            sigma_opt = best_s
        print(f"\nCalibrated sigma (minimizes total surprisal):")
        print(f"  global grid argmin:  sigma = {best_s:.4f}")
        print(f"  after refinement:    sigma = {sigma_opt:.4f}  total = {best_v:.3f} bits")
        sigma = sigma_opt

    # Per-position surprisal aggregation. We compute:
    #   - quantized:  surprisal of quantized.argmax under FP16+noise
    #   - baseline:   surprisal of FP16.argmax under FP16+noise (the "noise floor"
    #                 — the cost the framework attributes to a hypothetical FP16
    #                 prover with no quantization)
    #   - delta = quantized − baseline: the quantization-specific cost in bits,
    #                 separated from the FP16 hardware-noise floor.
    total_q_surprisal = 0.0
    total_baseline_surprisal = 0.0
    total_positions = 0
    top1_matches = 0
    print(f"\nPer-prompt surprisal at sigma={sigma}:")
    for i, (fp_logits, q_logits, ids) in enumerate(
        zip(all_fp_logits, all_q_logits, all_input_ids)
    ):
        q_tokens = q_logits.argmax(dim=-1)   # quantized's argmax (the prover's emission)
        fp_tokens = fp_logits.argmax(dim=-1) # FP16's argmax (the noise-floor reference)
        matches = (q_tokens == fp_tokens).sum().item()

        q_probs = _noise_corrected_probs(fp_logits, sigma)
        q_of_q  = q_probs.gather(-1, q_tokens.unsqueeze(-1)).squeeze(-1)
        q_of_fp = q_probs.gather(-1, fp_tokens.unsqueeze(-1)).squeeze(-1)
        surp_q  = -torch.log2(q_of_q.clamp_min(1e-30))
        surp_fp = -torch.log2(q_of_fp.clamp_min(1e-30))

        seq_len = surp_q.numel()
        total_q_surprisal += surp_q.sum().item()
        total_baseline_surprisal += surp_fp.sum().item()
        total_positions += seq_len
        top1_matches += matches
        print(
            f"  prompt {i}: seq_len={seq_len} top1_match={matches}/{seq_len} "
            f"q={surp_q.mean():.4f}  fp16_baseline={surp_fp.mean():.4f}  "
            f"delta={(surp_q.mean() - surp_fp.mean()).item():.4f} bits/tok"
        )

    q_rate = total_q_surprisal / total_positions if total_positions else 0.0
    fp_rate = total_baseline_surprisal / total_positions if total_positions else 0.0
    print(f"\nAggregate across {total_positions} positions in {len(prompts)} prompts at sigma={sigma}:")
    print(f"  top-1 agreement (quantized vs FP16): {top1_matches}/{total_positions} ({top1_matches/total_positions:.1%})")
    print(f"  FP16 baseline rate (no quantization):     {fp_rate:.4f} bits/token")
    print(f"  Quantized rate (full quantized stack):    {q_rate:.4f} bits/token")
    print(f"  Quantization-specific extra:              {q_rate - fp_rate:+.4f} bits/token")


def scale_sweep_diagnostic(
    scales: Optional[list] = None,
    Ts: Optional[list] = None,
    silu_x_maxes: Optional[list] = None,
    exp_clips: Optional[list] = None,
    n_chunks: int = 8,
    chunk_tokens: int = 64,
):
    """
    Sweep Q-scale (and optionally table size T) and report per-config:
    final-logits rel-drift, top-1 match vs FP, and unexplained bits/token
    (= -log2 of Q's softmax probability at the FP top-1 token).

    Loads FP model once for the FP baseline, then re-loads a fresh model per
    (scale, T) config because patch_model_inplace destroys the FP weights.
    """
    import os
    import math
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if scales is None:
        scales = [2**10, 2**12, 2**14, 2**16, 2**18, 2**20]
    if Ts is None:
        Ts = [DEFAULT_TABLE_T]
    if silu_x_maxes is None:
        silu_x_maxes = [DEFAULT_SILU_X_MAX]
    if exp_clips is None:
        exp_clips = [DEFAULT_EXP_CLIP]

    # Snapshot true originals so we can restore between scales
    ORIG_SILU = F.silu
    ORIG_SOFTMAX = F.softmax

    print(f"Loading WikiText prompts: {n_chunks} chunks x {chunk_tokens} tokens")
    prompts = _wikitext_prompts(n_chunks=n_chunks, chunk_tokens=chunk_tokens)

    model_id = os.environ.get("MODEL_PATH", "meta-llama/Llama-2-7b-hf")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    def load_model():
        m = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float32, attn_implementation="eager"
        ).cuda()
        m.eval()
        return m

    # ---- FP run, save logits & top-1 ----
    print("FP forward (single load)...")
    model = load_model()
    fp_logits_per_prompt = []
    fp_top1_per_prompt = []
    for p in prompts:
        ids = tokenizer(p, return_tensors="pt").input_ids.cuda()
        with torch.no_grad():
            fl = model(ids).logits.detach().float().cpu()
        fp_logits_per_prompt.append(fl)
        fp_top1_per_prompt.append(fl.argmax(dim=-1))
    del model
    torch.cuda.empty_cache()

    # ---- Per-config Q runs ----
    print(f"\n{'scale':>10} {'T':>8} {'silu_xmax':>10} {'exp_clip':>9} {'rel-drift':>12} {'top1-match':>12} {'bits/tok':>10}")
    print("-" * 90)

    configs = [(s, T, x, c) for s in scales for T in Ts for x in silu_x_maxes for c in exp_clips]
    for scale, T, silu_xmax, exp_clip in configs:
        F.silu, F.softmax = ORIG_SILU, ORIG_SOFTMAX  # reset before each config
        model = load_model()
        patch_model_inplace(model, scale=scale, T=T)
        install_activation_patches(scale=scale, T=T,
                                   silu_x_max=silu_xmax, softmax_exp_clip=exp_clip)

        rel_drifts = []
        match_count = 0
        pos_count = 0
        bits_sum = 0.0

        for p, fp_logits, fp_top1 in zip(prompts, fp_logits_per_prompt, fp_top1_per_prompt):
            ids = tokenizer(p, return_tensors="pt").input_ids.cuda()
            with torch.no_grad():
                q_logits = model(ids).logits.detach().float().cpu()
            rel_drifts.append((q_logits - fp_logits).norm().item() / fp_logits.norm().item())
            q_top1 = q_logits.argmax(dim=-1)
            match_count += (q_top1 == fp_top1).sum().item()
            pos_count += q_top1.numel()
            log_q = torch.log_softmax(q_logits, dim=-1)
            bits_per_pos = -log_q.gather(-1, fp_top1.unsqueeze(-1)).squeeze(-1) / math.log(2)
            bits_sum += bits_per_pos.sum().item()

        avg_drift = sum(rel_drifts) / len(rel_drifts)
        top1 = match_count / pos_count
        bits = bits_sum / pos_count
        print(f"{scale:>10d} {T:>8d} {silu_xmax:>10.1f} {exp_clip:>9.1f} {avg_drift:>12.4e} {top1:>11.2%} {bits:>10.4f}", flush=True)

        del model
        import gc; gc.collect()
        torch.cuda.empty_cache()

    # Restore patches
    F.silu, F.softmax = ORIG_SILU, ORIG_SOFTMAX


# ----- WikiText-2 perplexity sketch ------------------------------------------

def wikitext_perplexity(model: nn.Module, tokenizer, n_sequences: int = 256, seq_len: int = 1024) -> float:
    """
    Compute perplexity over n_sequences chunks of WikiText-2.
    Loaded via the `datasets` library; expects `pip install datasets`.
    """
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    tokens = tokenizer(text, return_tensors="pt").input_ids[0].to("cuda")

    nll_sum = 0.0
    n_tokens = 0
    with torch.no_grad():
        for i in range(min(n_sequences, len(tokens) // seq_len)):
            chunk = tokens[i * seq_len : (i + 1) * seq_len].unsqueeze(0)
            out = model(chunk)
            logits = out.logits[:, :-1].contiguous()
            targets = chunk[:, 1:].contiguous()
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )
            nll_sum += loss.item() * targets.numel()
            n_tokens += targets.numel()
    return math.exp(nll_sum / n_tokens)


# ----- Entry point -----------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run the smoke test only.")
    parser.add_argument("--ablate", action="store_true",
                        help="Run smoke ablation across patch combinations to isolate which op breaks accuracy.")
    parser.add_argument("--unexplained", action="store_true",
                        help="Compute unexplained information rate (zkllm-entropy framework).")
    parser.add_argument("--sigma", type=float, default=0.5,
                        help="Hardware noise sigma for the noise-corrected probability model.")
    parser.add_argument("--calibrate", action="store_true",
                        help="Self-calibrate sigma by minimizing the total surprisal (Gibbs).")
    parser.add_argument("--wikitext", action="store_true",
                        help="Use WikiText-2 chunks instead of the 4 hardcoded prompts.")
    parser.add_argument("--n-chunks", type=int, default=16,
                        help="Number of WikiText chunks (with --wikitext).")
    parser.add_argument("--chunk-tokens", type=int, default=128,
                        help="Approximate tokens per chunk (with --wikitext).")
    parser.add_argument("--per-op", action="store_true",
                        help="Per-op local precision diagnostic.")
    parser.add_argument("--layer-drift", action="store_true",
                        help="Per-layer divergence trajectory (test layer-compounding hypothesis).")
    parser.add_argument("--submodule-drift", action="store_true",
                        help="Per-submodule drift: error budget by op type.")
    parser.add_argument("--rel-noise", action="store_true",
                        help="Relative noise growth per submodule + layer-by-layer trace (Q vs FP32).")
    parser.add_argument("--rel-noise-fp16", action="store_true",
                        help="Same as --rel-noise but compares FP16 vs FP32 (no Q mechanisms; baseline).")
    parser.add_argument("--rel-noise-fp32", action="store_true",
                        help="Same as --rel-noise but compares FP32 vs FP64 (no Q mechanisms; FP-precision floor).")
    parser.add_argument("--trace-suffix", type=str, default="mlp",
                        help="Submodule suffix to trace layer-by-layer with --rel-noise.")
    parser.add_argument("--ablate-submodule", action="store_true",
                        help="Replace each submodule's quantized output with FP reference and measure logits drift.")
    parser.add_argument("--attn-breakdown", action="store_true",
                        help="Fine-grained rel-noise inside attention math (softmax in/out via F.softmax wrap).")
    parser.add_argument("--scale-sweep", action="store_true",
                        help="Sweep Q-scale: report rel-drift, top-1 match, bits/tok per scale.")
    parser.add_argument("--sweep-scales", type=str, default="1024,4096,16384,65536,262144,1048576",
                        help="Comma-separated list of scales for --scale-sweep (default: 2^10 ... 2^20).")
    parser.add_argument("--sweep-Ts", type=str, default="",
                        help="Comma-separated list of T (table size) values for --scale-sweep. Default: just DEFAULT_TABLE_T.")
    parser.add_argument("--sweep-silu-xmax", type=str, default="",
                        help="Comma-separated list of silu x_max values. Default: just DEFAULT_SILU_X_MAX (32).")
    parser.add_argument("--sweep-exp-clip", type=str, default="",
                        help="Comma-separated list of softmax exp_clip values. Default: just DEFAULT_EXP_CLIP (16).")
    parser.add_argument("--scale", type=int, default=DEFAULT_SCALE)
    parser.add_argument("--T", type=int, default=DEFAULT_TABLE_T)
    parser.add_argument("--seqs", type=int, default=64)
    parser.add_argument("--seqlen", type=int, default=1024)
    args = parser.parse_args()

    if args.ablate:
        smoke_ablation()
        return

    if args.per_op:
        per_op_precision_check()
        return

    if args.layer_drift:
        layer_drift_diagnostic()
        return

    if args.submodule_drift:
        submodule_drift_diagnostic()
        return

    if args.rel_noise:
        relative_noise_growth_diagnostic(trace_suffix=args.trace_suffix, mode="quantized")
        return

    if args.rel_noise_fp16:
        relative_noise_growth_diagnostic(trace_suffix=args.trace_suffix, mode="fp16")
        return

    if args.rel_noise_fp32:
        relative_noise_growth_diagnostic(trace_suffix=args.trace_suffix, mode="fp32", ref_dtype="fp64")
        return

    if args.ablate_submodule:
        submodule_ablation_diagnostic()
        return

    if args.attn_breakdown:
        attention_breakdown_diagnostic()
        return

    if args.scale_sweep:
        scales = [int(s) for s in args.sweep_scales.split(",") if s.strip()]
        Ts = [int(t) for t in args.sweep_Ts.split(",") if t.strip()] or None
        silu_x_maxes = [float(x) for x in args.sweep_silu_xmax.split(",") if x.strip()] or None
        exp_clips = [float(c) for c in args.sweep_exp_clip.split(",") if c.strip()] or None
        scale_sweep_diagnostic(scales=scales, Ts=Ts,
                               silu_x_maxes=silu_x_maxes, exp_clips=exp_clips)
        return

    if args.unexplained:
        unexplained_info(sigma=args.sigma, calibrate=args.calibrate,
                         scale=args.scale, T=args.T,
                         wikitext=args.wikitext,
                         n_chunks=args.n_chunks,
                         chunk_tokens=args.chunk_tokens)
        return

    if args.smoke:
        smoke_test()
        return

    # Full eval path: FP16 baseline, then patched quantized, both on WikiText-2.
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "meta-llama/Llama-2-7b-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print(f"Loading {model_id} in FP16 for baseline...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16
    ).cuda()
    ref_model.eval()

    print(f"Computing FP16 baseline perplexity over {args.seqs} sequences of length {args.seqlen}...")
    fp16_ppl = wikitext_perplexity(ref_model, tokenizer, n_sequences=args.seqs, seq_len=args.seqlen)
    print(f"FP16 baseline perplexity: {fp16_ppl:.4f}")
    del ref_model
    torch.cuda.empty_cache()

    print("Loading fresh model for patched version...")
    q_model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16
    ).cuda()
    q_model.eval()
    print(f"Patching: scale=2^{int(math.log2(args.scale))}, T=2^{int(math.log2(args.T))}...")
    patch_model_inplace(q_model, scale=args.scale, T=args.T)
    install_activation_patches(scale=args.scale, T=args.T)

    print("Computing quantized perplexity...")
    q_ppl = wikitext_perplexity(q_model, tokenizer, n_sequences=args.seqs, seq_len=args.seqlen)
    print(f"Quantized perplexity:    {q_ppl:.4f}")

    print(f"Perplexity ratio (q/fp16): {q_ppl / fp16_ppl:.4f}")
    print(f"Perplexity delta:          {q_ppl - fp16_ppl:+.4f}")


if __name__ == "__main__":
    main()


# ============================================================================
# Setup on DGX Spark (`spark-c191` on Tailscale)
# ============================================================================
#
# Already present on the Spark (user `claude`):
#   - Python venv at /home/claude/venv-hf (has hf-transfer, huggingface_hub,
#     numpy, safetensors, rich, etc., but not torch / transformers / datasets yet)
#   - Llama-2-7B safetensors at /home/claude/models/llama-2-7b-hf/
#     (also under /home/claude/.cache/huggingface/hub/...)
#
# What still needs to be installed (one-time):
#
#   ssh spark-c191 'sudo -u claude /home/claude/venv-hf/bin/pip install \
#     --pre torch --index-url https://download.pytorch.org/whl/nightly/cu126 \
#     transformers datasets accelerate'
#
# (If the cu126 wheel doesn't include the GB10 sm version, try cu128, or fall
# back to the NVIDIA pytorch container at nvcr.io/nvidia/pytorch:25.0X-py3.)
# Verify:
#
#   ssh spark-c191 'sudo -u claude /home/claude/venv-hf/bin/python \
#     -c "import torch; print(torch.__version__, torch.cuda.is_available())"'
#
# Copy this script over:
#
#   scp /root/CC-project-analysis/ligero/quantization_accuracy_sim.py \
#       spark-c191:/home/claude/
#
# Smoke test (one forward pass, prints KL(FP16 || quantized) at last position):
#
#   ssh spark-c191 'sudo -u claude /home/claude/venv-hf/bin/python \
#     /home/claude/quantization_accuracy_sim.py --smoke'
#
# Full eval (~minutes to tens of minutes; int64 matmul is the slow path):
#
#   ssh spark-c191 'sudo -u claude /home/claude/venv-hf/bin/python \
#     /home/claude/quantization_accuracy_sim.py --seqs 16 --seqlen 1024'
#
# Sweeps once the small run looks clean:
#
#   for scale in 1024 4096 16384 65536; do
#     ssh spark-c191 "sudo -u claude /home/claude/venv-hf/bin/python \
#       /home/claude/quantization_accuracy_sim.py --scale $scale --T 65536 --seqs 16"
#   done
#
# Speed note: the int64 matmul path uses non-tensor-core kernels and will be
# the bottleneck. Llama-7B forward at int64 is expected to take tens of seconds
# per sequence. If this becomes painful, the matmul can be swapped to FP64 with
# a check that all intermediates stay integer-valued (FP64 mantissa exceeds
# the 2^44 worst-case accumulator bound, so it's bit-exact at our magnitudes).
# Doing that swap is a "second pass" optimization; int64 is the correctness
# baseline.
# ============================================================================
