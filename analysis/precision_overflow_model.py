"""
Precision and overflow analysis for ZK proof of LLM inference.

Models a transformer layer (attention + FFN) as a pipeline of stages.
Each stage tracks:
  - max_abs_value: upper bound on |stored value| at this stage
  - scale: fixed-point scale, so stored = round(scale * real_value)
  - abs_error: upper bound on |stored - scale * real_value| in stored units

Derived columns in the report:
  - bits = bits required for max_abs_value (signed)
  - margin = 64 (Goldilocks) - bits - safety_margin_bits
  - real_err = abs_error / scale (error expressed in original real units)
  - frac_err = abs_error / max_abs_value (fractional / relative error)

Field: Goldilocks p = 2^64 - 2^32 + 1. A flag (!) marks any stage where
margin drops below zero, indicating the worst-case bound exceeds the
field with the configured safety margin.

Reference defaults (Llama 4 Maverick): see design-feasibility.md §4.1.

Design notes (in flux; observations from working through the protocol):
  - Format range + LogUp table ranges may be sufficient to enforce
    activation bounds at runtime: values flowing outside range fail to
    look up, rejecting the proof. No separate norm-proof step needed.
  - Matmul accumulators are an exception. Goldilocks arithmetic wraps
    mod p silently and Freivalds-reduced matmul checks reduce mod p too,
    so accumulator overflow can't be caught at runtime — the format
    would need to guarantee k · max_int² < p/2 statically for the
    largest contracted dimension k in the network.
  - Accumulator overflow probably isn't a soundness break either way:
    the verifier still sees an internally consistent proof, just of a
    different output than the prover intended. A prover usability issue,
    not a malicious-prover concern.
  - Softmax outputs in [0, 1] reset the analytical bound chain at every
    attention block. Useful when bounds would otherwise compound
    multiplicatively across many layers.
  - One contract option we're considering: verifier publishes
    recommended matrix and vector norm bounds; if a prover's weights and
    activations stay within them the proof is guaranteed to accept, and
    if not it might not. The prover-side discipline would be
    documentation, not an enforced constraint.

V1 LIMITATIONS:
  - matmul error compounds worst-case (assumes all per-element errors
    align). Real i.i.d. errors grow as sqrt(k); use the model to find
    overflow risks, then tighten with statistical bounds where useful.
  - softmax denominator division is approximate. The exp-lookup output
    is bounded by clip; the row-sum and divide are modeled coarsely.
  - No per-layer composition: this models one attention block and one
    FFN, not the full 48-layer stack with residual stream.
  - Quantization-vs-FP drift simulation is not yet implemented; this
    is purely analytical bounds.
"""

from dataclasses import dataclass
from math import ceil, log2
from typing import List


GOLDILOCKS_BITS = 64
DEFAULT_SAFETY_MARGIN_BITS = 8  # require max |value| < 2^(64 - 8) = 2^56


@dataclass
class Stage:
    """One named stage with overflow and precision metadata."""
    name: str
    max_abs_value: float
    scale: float
    abs_error: float

    @property
    def required_bits(self) -> int:
        if self.max_abs_value < 1:
            return 1
        return ceil(log2(self.max_abs_value)) + 1  # +1 for sign bit

    def overflow_margin(self, safety_margin_bits: int = DEFAULT_SAFETY_MARGIN_BITS) -> int:
        return GOLDILOCKS_BITS - self.required_bits - safety_margin_bits

    @property
    def relative_error(self) -> float:
        return self.abs_error / self.scale if self.scale > 0 else 0.0


# ----- Pipeline operations ---------------------------------------------------

def matmul(a: Stage, b: Stage, k: int, name: str) -> Stage:
    """
    c[i, j] = sum over l of a[i, l] * b[l, j], with k contracted entries.

    Output bound: k * |a_max| * |b_max|.
    Output scale: a.scale * b.scale.
    Error per output: k * (a.err * b_max + a_max * b.err + a.err * b.err).
    """
    out_max = k * a.max_abs_value * b.max_abs_value
    out_scale = a.scale * b.scale
    out_err = k * (
        a.abs_error * b.max_abs_value
        + a.max_abs_value * b.abs_error
        + a.abs_error * b.abs_error
    )
    return Stage(name, out_max, out_scale, out_err)


def add(a: Stage, b: Stage, name: str) -> Stage:
    """Pointwise add (assumes equal scales)."""
    if a.scale != b.scale:
        raise ValueError(f"add: scale mismatch {a.scale} vs {b.scale}")
    return Stage(
        name,
        a.max_abs_value + b.max_abs_value,
        a.scale,
        a.abs_error + b.abs_error,
    )


def hadamard(a: Stage, b: Stage, name: str) -> Stage:
    """Pointwise multiply."""
    return Stage(
        name,
        a.max_abs_value * b.max_abs_value,
        a.scale * b.scale,
        a.abs_error * b.max_abs_value
        + a.max_abs_value * b.abs_error
        + a.abs_error * b.abs_error,
    )


def lookup(
    x: Stage,
    name: str,
    T: int,
    input_range: float,
    output_max: float,
    output_scale: float,
    lipschitz: float,
) -> Stage:
    """
    y = f(x) via LogUp on a precomputed (t_in, t_out) table.

    T = table size, input_range = R covered by the table in real units.
    output_max = upper bound on |f| in real units.
    lipschitz = Lipschitz constant of f (for error propagation).

    Resolution at the input is R/T, so the table introduces a worst-case
    rounding error of R/(2T) on the input. Plus the input's existing
    real-units error (x.abs_error / x.scale) propagates with constant
    `lipschitz`.
    """
    input_resolution = input_range / T
    real_input_err = x.abs_error / x.scale
    real_output_err = lipschitz * (real_input_err + input_resolution / 2)
    out_err = output_scale * real_output_err
    return Stage(name, output_max * output_scale, output_scale, out_err)


def requantize(s: Stage, new_scale: float, name: str) -> Stage:
    """
    Rescale to a new (typically smaller) scale. Adds 0.5 to abs_error from
    the rounding step, then the error scales with the rescale factor.
    """
    rescale = new_scale / s.scale
    return Stage(
        name,
        s.max_abs_value * rescale,
        new_scale,
        s.abs_error * abs(rescale) + 0.5,
    )


def sum_reduce(s: Stage, k: int, name: str) -> Stage:
    """
    Sum `k` values of the same scale (e.g., a row-sum across an embedding dim).
    Output bound and error grow linearly in `k`. Scale unchanged.
    """
    return Stage(
        name,
        s.max_abs_value * k,
        s.scale,
        s.abs_error * k,
    )


def softmax_normalize(scores: Stage, name: str, denom_bits_lost: int = 0) -> Stage:
    """
    Output of softmax: bounded in [0, 1] in real units. Stored at output_scale.
    The denom_bits_lost parameter accounts for division precision loss.

    This is a coarse model: the actual softmax involves an exp lookup, a sum,
    and a division. Here we collapse them and assume the output scale is
    chosen so that the unit interval is well-represented.
    """
    output_scale = scores.scale
    # Assume sum normalization preserves order-of-magnitude precision plus a
    # small additive loss from division.
    new_err = scores.abs_error / scores.max_abs_value * output_scale + 2 ** denom_bits_lost
    return Stage(name, output_scale, output_scale, new_err)


# ----- Reporting -------------------------------------------------------------

def report(stages: List[Stage], safety_margin_bits: int = DEFAULT_SAFETY_MARGIN_BITS) -> None:
    headers = ("Stage", "Max |val|", "Bits", "Margin", "Scale", "Real err", "Frac err")
    widths = (44, 12, 6, 8, 12, 12, 12)
    line = " ".join(f"{h:<{w}}" for h, w in zip(headers, widths))
    print(line)
    print("-" * sum(widths))
    for s in stages:
        margin = s.overflow_margin(safety_margin_bits)
        flag = " (!)" if margin < 0 else ""
        frac_err = s.abs_error / s.max_abs_value if s.max_abs_value > 0 else 0.0
        cells = (
            f"{s.name + flag:<{widths[0]}}",
            f"{s.max_abs_value:<{widths[1]}.2e}",
            f"{s.required_bits:<{widths[2]}}",
            f"{margin:<{widths[3]}}",
            f"{s.scale:<{widths[4]}.2e}",
            f"{s.relative_error:<{widths[5]}.2e}",
            f"{frac_err:<{widths[6]}.2e}",
        )
        print(" ".join(cells))


# ----- Pipeline: one Maverick attention block --------------------------------

def maverick_attention_block(
    S: int = 2048,
    d: int = 5120,
    n_q: int = 40,
    n_kv: int = 8,
    d_h: int = 128,
    B_act: int = 16,
    B_weight: int = 8,
    softmax_T: int = 2 ** 16,
) -> List[Stage]:
    """
    Trace one attention block forward and return the list of stages.

    B_act: bits used for activations (signed fixed-point); max |x| ~ 2^(B_act-1).
    B_weight: bits used for weights; max |w| ~ 2^(B_weight-1).
    softmax_T: LogUp table size for exp.
    """
    stages: List[Stage] = []

    # Input from the previous layer's residual stream. Assume per-token RMSNorm
    # has been applied, so per-element |x| is bounded by ~2^(B_act-1).
    activation_max = 2 ** (B_act - 1)
    activation_scale = 1.0  # we measure values in stored integer units
    x = Stage("input activation (post-RMSNorm)", activation_max, activation_scale, 0.5)
    stages.append(x)

    # Q, K, V projections. Treat each weight matrix as Stage with the same
    # convention (its "max_abs_value" represents the per-element bound).
    weight_max = 2 ** (B_weight - 1)
    W = Stage("weight matrix entry", weight_max, activation_scale, 0.5)

    Q_pre = matmul(x, W, k=d, name="Q = x · W_Q (pre-rescale)")
    stages.append(Q_pre)

    # GPUs typically rescale matmul outputs back into activation precision
    # before downstream use; model this as a requantization that brings the
    # scale back to the activation scale.
    Q = requantize(Q_pre, activation_scale, "Q (rescaled to act scale)")
    stages.append(Q)

    # Same shape pattern for K and V, but they have d_kv output dim — same per-
    # element bound, so we reuse Q's profile.
    K = Stage("K (rescaled to act scale)", Q.max_abs_value, Q.scale, Q.abs_error)
    V = Stage("V (rescaled to act scale)", Q.max_abs_value, Q.scale, Q.abs_error)
    stages.append(K)
    stages.append(V)

    # QK^T per head: contract over d_h.
    QK = matmul(Q, K, k=d_h, name="QK^T (per head)")
    stages.append(QK)

    # Scale by 1/sqrt(d_h). Modeled as requantize to a smaller scale.
    QK_scaled = requantize(QK, QK.scale / (d_h ** 0.5), "QK^T / sqrt(d_h)")
    stages.append(QK_scaled)

    # Softmax: exp lookup on QK_scaled, then divide by row sum.
    # exp output bounded by exp(QK_max_real), but in practice we clip — here
    # we assume the ZK protocol clips inputs to [-clip, +clip] and the exp
    # output is bounded by exp(clip).
    clip_real = 16.0  # conservative; in practice tighter via row-max shift
    exp_lipschitz = 2 ** clip_real  # worst-case derivative of exp
    exp_output_max = 2 ** clip_real  # exp is monotone; output range upper-bound
    exp_scale = activation_scale  # store exp values at activation precision
    exp_lookup_input_range = 2 * clip_real
    exp_stage = lookup(
        QK_scaled,
        name="exp(QK / sqrt) [LogUp]",
        T=softmax_T,
        input_range=exp_lookup_input_range,
        output_max=exp_output_max,
        output_scale=exp_scale,
        lipschitz=exp_lipschitz,
    )
    stages.append(exp_stage)

    # Row-sum normalization (softmax denominator). For one row of length S,
    # sum has up to S * exp_max bound; division returns a value in [0, 1].
    # Modeled as accumulator + softmax_normalize.
    softmax_acc = Stage(
        "softmax row-sum (S terms)",
        S * exp_stage.max_abs_value,
        exp_stage.scale,
        S * exp_stage.abs_error,
    )
    stages.append(softmax_acc)

    softmax_p = softmax_normalize(
        softmax_acc, name="softmax probs (post-normalize)", denom_bits_lost=4
    )
    stages.append(softmax_p)

    # Attention output: softmax_p @ V, contracted over S.
    attn_out = matmul(softmax_p, V, k=S, name="softmax · V (per head)")
    stages.append(attn_out)

    # Output projection: concatenated heads -> d.
    attn_proj = matmul(attn_out, W, k=d, name="output projection")
    stages.append(attn_proj)

    # Re-quantize back to activation scale for the residual stream.
    attn_residual = requantize(attn_proj, activation_scale, "attention residual contribution")
    stages.append(attn_residual)

    return stages


def maverick_ffn_dense(
    d: int = 5120,
    d_ff: int = 16384,
    B_act: int = 16,
    B_weight: int = 8,
    silu_T: int = 2 ** 16,
) -> List[Stage]:
    """Trace one dense SwiGLU FFN: gate, up, silu(gate) * up, down."""
    stages: List[Stage] = []

    activation_max = 2 ** (B_act - 1)
    activation_scale = 1.0
    x = Stage("FFN input (post-RMSNorm)", activation_max, activation_scale, 0.5)
    stages.append(x)

    weight_max = 2 ** (B_weight - 1)
    W = Stage("weight matrix entry", weight_max, activation_scale, 0.5)

    gate_pre = matmul(x, W, k=d, name="gate = x · W_gate (pre-rescale)")
    stages.append(gate_pre)
    gate = requantize(gate_pre, activation_scale, "gate (rescaled)")
    stages.append(gate)

    up_pre = matmul(x, W, k=d, name="up = x · W_up (pre-rescale)")
    stages.append(up_pre)
    up = requantize(up_pre, activation_scale, "up (rescaled)")
    stages.append(up)

    # silu(gate) via LogUp. silu is 1-Lipschitz; we cover the gate's range.
    silu = lookup(
        gate,
        name="silu(gate) [LogUp]",
        T=silu_T,
        input_range=2 * gate.max_abs_value / gate.scale,
        output_max=gate.max_abs_value / gate.scale,  # |silu(x)| <= |x|
        output_scale=activation_scale,
        lipschitz=1.0,
    )
    stages.append(silu)

    hidden = hadamard(silu, up, name="silu(gate) · up (Hadamard)")
    stages.append(hidden)

    # Down projection: contract over d_ff.
    down_pre = matmul(hidden, W, k=d_ff, name="down = hidden · W_down (pre-rescale)")
    stages.append(down_pre)
    down = requantize(down_pre, activation_scale, "FFN residual contribution")
    stages.append(down)

    return stages


def maverick_rmsnorm(
    d: int = 5120,
    B_act: int = 16,
    activation_scale: float = 2 ** 12,
    weight_scale: float = 2 ** 12,
    rsqrt_T: int = 2 ** 16,
    rsqrt_input_bits: int = 16,
    eps: float = 1e-5,
) -> List[Stage]:
    """
    RMSNorm: y = γ ⊙ x / sqrt(mean(x²) + ε)

    Verification pattern (paired tlookup for rsqrt):
      1. X² = X ⊙ X                       (Hadamard, d quadratic constraints per token)
      2. S = Σ_k X²[k]                    (linear, one constraint per token, d non-zeros)
      3. rescale S into the rsqrt table's input range
                                          (bit decomposition / range check, §3.5)
      4. R = 1/sqrt(S/d + ε)              (paired tlookup; one LogUp instance shared
                                           across all RMSNorm ops in the model)
      5. γR = γ ⊙ R (broadcast per token) (Hadamard, d quadratic per token)
      6. Y = γR ⊙ X                       (Hadamard, d quadratic per token)

    Notes on the rsqrt step. Input range to rsqrt is `[ε, d · X_max²]`, which is wide
    (up to ~32 bits at our defaults). Single-table lookup with T = 2^16 over the full
    range gives only `R/T = 2^{16}` input-resolution — coarse where it matters most
    (small mean(X²) is precision-critical because rsqrt has steep slope there). Three
    mitigations, in order of complexity:

      a. Pre-rescale (modeled here): drop low-order bits of S so the table covers a
         smaller input range. Cheap (one range check), but loses precision at the
         low-input end where rsqrt is steepest.
      b. Multi-segment lookup: decompose the input across K small tables, each of size
         T = b. Same precision as a single T = b^K table at K-times the LogUp cost.
         Good fit because rsqrt is multiplicatively well-behaved.
      c. Larger T (e.g., T = 2^32 with multiplicities committed once): pays a fixed
         table cost in R_p1, but the M (per-instance query count) stays the same.

    The model below uses (a). The reported error at the rsqrt stage is worst-case at
    input near ε; in practice the dominant mass of mean(X²) is order 1 and the typical
    error is much smaller.
    """
    stages: List[Stage] = []

    activation_max = 2 ** (B_act - 1)
    x = Stage("RMSNorm input (residual stream)", activation_max, activation_scale, 0.5)
    stages.append(x)

    # Step 1: X² (Hadamard).
    x_sq = hadamard(x, x, name="X² (Hadamard)")
    stages.append(x_sq)

    # Step 2: Σ_k X²[k] per token (linear contraction over embed_dim).
    sum_x_sq = sum_reduce(x_sq, k=d, name="Σ X² (per token, k=d)")
    stages.append(sum_x_sq)

    # Step 3: rescale into rsqrt-table input range (one range check via bit decomp).
    target_input_max = 2 ** rsqrt_input_bits
    rescale_factor = target_input_max / sum_x_sq.max_abs_value
    rescaled_sum = requantize(
        sum_x_sq,
        sum_x_sq.scale * rescale_factor,
        name="Σ X² rescaled (for rsqrt table)",
    )
    stages.append(rescaled_sum)

    # Step 4: paired tlookup for 1/sqrt(S/d + ε).
    # Worst-case Lipschitz of 1/sqrt(x+ε) is at x=0: 1/(2·ε^{1.5}). The model uses
    # this worst-case to compute the error column; typical inputs are far from ε and
    # the actual error is much smaller.
    rsqrt_input_range_real = target_input_max / rescaled_sum.scale
    worst_case_lipschitz = 1 / (2 * eps ** 1.5)
    rsqrt_output_max_real = 1 / (eps ** 0.5)  # 1/sqrt(ε) at the small-input edge
    rsqrt = lookup(
        rescaled_sum,
        name="1/sqrt(mean(X²) + ε) [LogUp]",
        T=rsqrt_T,
        input_range=rsqrt_input_range_real,
        output_max=rsqrt_output_max_real,
        output_scale=activation_scale,
        lipschitz=worst_case_lipschitz,
    )
    stages.append(rsqrt)

    # Step 5: γ ⊙ rsqrt (per-token broadcast).
    gamma = Stage("γ (per-channel weight)", 2 ** (B_act - 1), weight_scale, 0.5)
    gamma_rsqrt = hadamard(rsqrt, gamma, name="γ · rsqrt (Hadamard, broadcast)")
    stages.append(gamma_rsqrt)

    # Step 6: rescale γR back to activation scale.
    gamma_rsqrt_rescaled = requantize(
        gamma_rsqrt, activation_scale, name="γ · rsqrt (rescaled)"
    )
    stages.append(gamma_rsqrt_rescaled)

    # Step 7: Y = (γ · rsqrt) ⊙ X (Hadamard).
    y = hadamard(gamma_rsqrt_rescaled, x, name="Y = (γ · rsqrt) · X (Hadamard)")
    stages.append(y)

    # Step 8: final rescale to activation scale for downstream.
    y_out = requantize(y, activation_scale, name="RMSNorm output (rescaled)")
    stages.append(y_out)

    return stages


def main():
    print("=" * 100)
    print("Maverick attention block (S = 2048)")
    print("=" * 100)
    report(maverick_attention_block(S=2048))

    print()
    print("=" * 100)
    print("Maverick attention block (S = 8192)")
    print("=" * 100)
    report(maverick_attention_block(S=8192))

    print()
    print("=" * 100)
    print("Maverick dense FFN")
    print("=" * 100)
    report(maverick_ffn_dense())

    print()
    print("=" * 100)
    print("Maverick RMSNorm (paired tlookup for rsqrt)")
    print("=" * 100)
    report(maverick_rmsnorm())


if __name__ == "__main__":
    main()
