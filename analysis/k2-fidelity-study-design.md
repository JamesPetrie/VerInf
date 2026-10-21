# K2 fidelity study: design and pre-registered acceptance (2026-10-19)

Session 10 found Kimi K2's integer model exact against its integer reference
and 5.1–5.4% off a float pass at the logits after two layers, with the
routed MoE output 18–21% off from expert flips at selection margins near
0.0005 (`analysis/b200-session-10-archive.md`). This note designs the study
that decides whether that integer arithmetic is an acceptable implementation
of K2, and fixes the acceptance criterion before any measurement. It was
settled in review on 2026-10-18 and 2026-10-19. §6 is frozen: it changes
only by a dated amendment written here before the data it would apply to
exist.

A fidelity failure does not invalidate a proof. The proof certifies the
specified integer computation; this study asks whether that computation
stands in for K2 well enough that its unexplained-information bound U can be
compared with what a real inference engine computes on the same tokens.

## 1. Two questions, kept apart

- **Arithmetic drift.** How far each arithmetic lands from a high-precision
  computation on the same decoded weights and tokens. The integer
  configurations and the rounding emulations (§3.2) answer this, and the
  emulations explain where drift arises. They do not reproduce any
  deployment.
- **Deployment equivalence.** Whether the integer model is at least as close
  to the high-precision computation as the engine that writes the
  continuation. Only the engine itself (§3.3) anchors this. If the engine arm
  fails, the emulations still guide precision work, but this question stays
  unanswered.

The decision (§6) concerns the second question, at full depth. Two layers
validate the machinery and rank configurations provisionally; they do not
settle full-depth accuracy.

## 2. Fixed inputs

### 2.1 Weights

Every arm reads the same bytes: `unsloth/Kimi-K2-Instruct-GGUF` at
`23bf90d8`, UD-Q4_K_XL, 13 shards, each pinned by sha256 in the session kit.
The float64 reference, the integer configurations and the emulations all
start from one decode, `k2_loader`'s (gguf-py's `dequantize`, float32,
widened to float64). Each run's manifest records a sha256 per decoded
tensor, so "the same weights" is checked, not assumed. The engine decodes
the same bytes with its own kernels, which is part of what it is.

Drift between distributed checkpoints, for example Moonshot's FP8 release
served by another stack against this Q4 GGUF, mixes weight quantization with
execution error. It is not part of this study. If it is ever measured, it is
reported separately, as total deployment drift.

### 2.2 The reference

p64 is the high-precision float64 reference: `k2_float_reference.forward`
in NumPy, and its torch port (§5.3) once validated against it. It
approximates the mathematical computation on the decoded weights. It does
not recover precision lost when the checkpoint was quantized, and it is not
"exact arithmetic".

### 2.3 Tokens

T is a token count; S is reserved for arithmetic scales. Each passage is a
frozen token sequence (§6.4) that every arm consumes unchanged, at the same
positions with the same causal context.

## 3. Arms

### 3.1 Integer configurations

These are the precision study's `CONFIGS` (`prover/k2_precision_study.py`):
`baseline` (2^12 throughout; equal to the validated reference by test),
`w16`, `a16`, `a14w14`, `a16w16`, and `a16w16n14`. The last is hypothetical:
its RMSNorm bracket would need revising. Selection stays at 2^16, and SiLU
and softmax keep their 2^12 tables. All six run at two layers. At full depth
the primary runs, with the exploratory configurations beside it (§3.5, §7).
Their logits come from the driver's engine pass, or, for the proved passage,
from the proof's own witness.

### 3.2 Emulations (K2-specific, two layers, diagnostic)

Each emulation starts from the decoded float64 tensors and names every
precision boundary below. A matmul is a float32 GEMM on rounded inputs.
That emulates fp32 accumulation; rounding a float64 product would not. Its
summation order and blocking differ from any GPU kernel, so it is an
emulation of a precision policy, not a reproduction of a kernel. bf16 rounds
float32 to nearest even at the bit level. E4M3 rounds to nearest even on its
grid (3 mantissa bits, exponent bias 7, subnormals to 2^-9), saturating at
±448.

| boundary (composition A order) | bf16 policy | FP8 policy |
|---|---|---|
| embedding rows | bf16 | bf16 |
| RMSNorm (attn, q_a, kv_a, ffn, output) | bf16 in; fp32 mean square, inverse root and gain; bf16 out | same |
| linears: q_a, q_b, kv_a, the k_b and v_b split, o; dense, shared and expert gate, up and down | bf16 input and weight, fp32 GEMM, bf16 out | input quantized E4M3 per 1×128 group along the contraction (scale = amax/448), weight bf16 (the decoded GGUF value, §2.1, not FP8), fp32 GEMM, bf16 out |
| RoPE / YaRN | fp32 on bf16 input, bf16 out | same |
| attention scores q·k, probabilities·V | bf16 inputs, fp32 accumulation; softmax fp32; probabilities bf16 before ·V | same: no FP8 on scores or probabilities |
| SiLU(gate)·up | fp32, bf16 out | same |
| residual additions | fp32 sum of bf16 terms, bf16 stream | same |
| router | see the routing table | same (the router is not an FP8 linear) |
| gate × expert output, sum over experts | fp32, bf16 out | same |
| LM head | bf16 input and weight, fp32 GEMM, fp32 logits | same (not FP8) |

The FP8 policy follows K2's release, `quantization_config`
`{fmt: e4m3, activation_scheme: dynamic, weight_block_size: [128, 128]}`
(Kimi-K2-Instruct `config.json`). Activation groups follow DeepSeek-V3's
reference `linear()`, which quantizes activations per 128 only when
`gemm_impl = "fp8"`. Its default is `"bf16"`: weights dequantized, bf16
GEMM. Its default `attn_impl = "absorb"` dequantizes `wkv_b` into bf16
einsums (`inference/model.py`). So the FP8 arm emulates block-FP8 serving of
the linears, not DeepSeek's default reference path, and attention never
receives FP8 inputs. Quantizing attention probabilities would make the
comparison artificially forgiving.

**Routing.** K2 selects on biased scores and weights experts by the original
scores (`analysis/topk-routing-design.md` §1). The emulations state each
stage:

| stage | R-fp32 | R-bf16 |
|---|---|---|
| logits r = x·W_router | bf16 x, F32 weight as stored, fp32 GEMM, fp32 | same GEMM, logits rounded to bf16 |
| scores s = sigmoid(r) | fp32 | bf16 |
| selection key s + b | fp32 (b is F32 as stored) | fp32 sum of the bf16 s and F32 b |
| top-k | on the fp32 key, lowest index on a tie | same |
| gate w = s_sel / Σ s_sel · 2.827 | fp32 on the unbiased s | fp32 on the bf16 s |

That gives four emulation arms (bf16 or FP8, each with R-fp32 or R-bf16).
No ranking of 2^12 against them is predicted. Fixed point has an absolute
step, while a scaled float's step depends on magnitude and block scale, and
their ordering is what the measurement establishes.

### 3.3 The engine

The engine is llama.cpp at a pinned commit. The kit pins, and every manifest
records:
- the build flags, backend and device;
- threads, `n_batch` and `n_ubatch`;
- the context size, pinned at 1,024 for this corpus. The runner requires it
  and never derives it from the sequence, so a capped generation and an
  early-EOS replay cannot run with different contexts;
- the KV cache type and the flash-attention setting;
- greedy sampling.

**The baseline is the generation run itself.** The prompt is prefilled with
the settings generation actually uses, then tokens are decoded one at a
time. The logits that chose each token are captured during generation,
along with the logits at the prompt's last position. The generated
continuation is frozen (token ids, sha256), and every other arm consumes
that frozen sequence.

**The 512-token replay is a separate diagnostic.** The frozen sequence is
prefilled in chunks of 512, with identical positions, causal context, cache
precision and backend. The logical batch and physical microbatch are
recorded. Its difference from the baseline is reported as a measurement. It
is not a floor, and it is not an alternative baseline.

At two layers the engine runs a truncated GGUF (§5.2).

**The engine's model must be K2's published definition** (amendment
2026-10-19, before any result). llama.cpp's deepseek2 loader (at `f174553`)
never reads the GGUF's YaRN beta keys; the unsloth GGUF does not carry them
either. So llama.cpp runs K2 with its defaults, beta_fast = 32 and
beta_slow = 1, where K2's `config.json` and our references have 1 and 1.
That rotates a band of RoPE dimensions differently at every position: a
difference of definition, not of arithmetic.

The K2 adapter (`prover/fidelity_adapters.py`) passes the betas as context
parameters. The check is an all-F32 toy with an f32 KV cache
(`prover/tests/test_llama_engine.py`):
- with K2's betas, llama.cpp equals p64 to float32 rounding: layer outputs
  within 5·10⁻⁷, logits KL 8·10⁻¹⁴, identical routing and gates;
- with the defaults, its layer outputs are 2·10⁻² off;
- at position 130,072, with K2's betas, the outputs are 1.4·10⁻³ off. That
  is consistent with float32 RoPE angles at large positions, which is
  arithmetic.

Reported upstream as ggml-org/llama.cpp#30830 (2026-10-19), framed on the
pinned commit, the GGUF metadata, the override and the toy reproduction.

**The GPU backend, open since session 11** (review of 2026-10-20). On a
card, the all-F32 toys match p64 at position 0 (layer outputs within
1.7·10⁻⁷) and are 2–4·10⁻³ off after it, in K2's form and in Moonlight's.
The CPU path is within float32 rounding.

- **Where it comes from.** The toy test decodes position 0 alone and the
  other positions as one batch. At `f174553`, multi-token float32 matmuls
  run in TF32 on Ampere and later:
  - up to 16 tokens in `mmf`, whose `mma` takes float32 bits as TF32
    operands;
  - beyond that in cuBLAS, which llama.cpp sets to
    `CUBLAS_TF32_TENSOR_OP_MATH`.
  
  Single tokens take float32 vector kernels.
- **What has to be established.** Whether the departure is consistent
  with expected internal precision or shows a semantic mismatch, by
  `tools/engine_precision.py` under a rule fixed before its run (amended
  by the review of 2026-10-21, also before it). The rule weighs evidence
  on two toys at 12 and 40 tokens. Agreement there cannot establish that
  the GPU computes the same function in general.
  - **Unresolved, first:** a CPU run or a one-token GPU run that failed or
    is missing. It is evidence of neither.
  - **Evidence of a semantic mismatch:** those runs all complete, and one is
    outside the float32 bound at some position.
  - **Consistent with internal precision:** they are all within it, and
    every batched GPU run outside it completed and departs first at a
    matmul's output by at most 2⁻⁶.
  - **Unresolved:** anything else, sent back for review.
  
  Session 12's outcome under this rule, and its correction: §3.6.
- **On K2's weights,** `fidelity_study.py engine-paths` scores the two-layer
  engine arm against p64 along each path:
  - the 512-token batch as session 11 replayed it;
  - one-token steps, as generation decodes;
  - cuBLAS's TF32 off;
  - the CPU.
- **Choosing the baseline.** The intended GPU deployment remains the
  preferred baseline, subject to resolving the correctness concerns; the
  CPU path is a diagnostic. The choice between them is made for deployment
  relevance, never for which gives the more favorable fidelity verdict.
  Evidence of a semantic mismatch is fixed before the configuration is
  frozen, and the configuration is frozen before any full-depth result.
  The baseline's generation-time logits are kept with the run for the
  full-depth comparison.
- **Environment in the manifests.** Every engine manifest records the
  environment variables that can change the arithmetic (`GGML_*`,
  `NVIDIA_TF32_*`, `CUBLAS_*`).

### 3.4 The a14w14 driver: combined rounding (2026-10-20)

a14w14 was the nominated K2 primary, pending the implementation and cost
gates (review of 2026-10-20). The amendment of §3.5 makes a16w16 the
primary, built by the same driver at its own scales.
`demo/demo_k2.py --precision` builds the tape at a profile,
`k2_loader.Scales`. Each scale has one meaning:
- `S_act`: activations;
- `S_wt`: checkpoint weights;
- `s_norm`: RMSNorm's internal scale;
- `S_tab`: the SiLU, softmax and router-logit tables;
- `S_sel`: router scores;
- `S_w`: gate weights.

The arithmetic is the precision study's model exactly, by combined rounding:
⌊⌊v/2^a⌋/2^b⌋ = ⌊v/2^(a+b)⌋ for signed v. A producer whose one consumer is
coarser rescales straight to the consumer's scale:
- the attention scores to softmax's;
- the dense, shared and routed gate projections to SiLU's;
- the router logits to the sigmoid table's;
- the query-LoRA and KV latents to RMSNorm's.

The residual stream is also read by its residual adds, so it stays at
`S_act`. It reaches each RMSNorm through one standalone 2-bit rescale: two
per layer and the final norm, 123 sites over K2's 61 layers. No claim takes
an input rescale.

The alternative, the claims' own input rescaling, is rejected for this
study. Its windows bound the coarsened value to [−2^15, 2^15 − 1] (SiLU,
RMSNorm) and [−2^15, 2^24 − 2^15 − 1] (softmax). Moonlight's full-depth
values break them at 109 sites. It would need those claims widened, which is
not implemented unless the combined design meets a concrete limitation.

The tail reads the logits natively at `S_act`, as the fidelity study
measures them (review of 2026-10-20). Its kernel and its gap domain are the
same in real logit units at every scale: `s_c` = 16·S² (σ² = 8) and gaps
strictly below 256 units (`gap_max` = 256·S). Its surprisal stays at the
resolution `s_b` = 2^12, so U = Σ surprisal / (`s_b`·ln 2) at any activation
precision.

At a14w14 that is `s_c` = 2^32, `s_y` = 2^28, `s_b` = 2^12, `gap_max` = 2^22
and k = `s_c`/`s_b` = 2^20. The honest ceiling quotient at the largest gap is
16,777,209 < 2^24 (16,777,185 at 2^12); `tail_params` asserts that bound and
refuses scales beyond 2^16.

The tail's constants are recorded with the proof identity. Each run records
`tail.gap` against the domain; Moonlight's largest gap is 63 units.
`tests/test_k2_tail.py` proves the domain's largest gap (accepted, U sound
and tight) and one past it (rejected). At 2^12 every constant is the
previous one.

Checks:
- **Baseline lock.** The 2^12 tape is locked by digest
  (`tests/test_k2_driver_structure.py`, taken before the profiles existed):
  no zero-shift rescale was added. Maverick shares no changed module.
- **Exactness.** At a14w14 the engine pass must equal the precision study's
  model at every traced name: the consumer inputs at `S_tab` and `s_norm`,
  routing, the gate weights and the logits. The record carries the profile
  with the proof's weight root, statement digest and enrollment identity;
  the bridge enrollment is labeled with `S_wt`.
- **Negative `rescale-witness`.** The first standalone rescale's first
  output is moved one higher and its low word 2^r lower, keeping every
  linear identity. The Rust verifier must reject it: only the low word's
  range check can catch it.

The exactness check, the proof and the negative run on a card (`tests/test_k2_driver.py`).

The a14w14 implementation gate passed in session 11 on shard 1 (two layers,
T = 100, bridged):
- exact at all 42 intermediates, every combined-rounding range inside its
  constraint;
- the honest proof accepted by the Rust verifier;
- the residual rescale tamper and the interleave rejected;
- against the 2^12 baseline on the same pod: proving +10.6%, Rust
  verification +8.4%, peak GPU memory +2.6%, the proof file +2.4%.

These runtimes are at the gate's dimensions. The roughly 2% full-depth
figures of §9.1 are counts, a separate result.

### 3.5 Amendment 2026-10-20: the K2 primary is a16w16

Made after the two-layer diagnostic (session 11, stage C) and before any
full-depth K2 result exists. §6's acceptance criterion is unchanged.

- **Primary:** a16w16, combined rounding.
- **Exploratory:** a14w14 and the 2^12 baseline, reported beside it
  (`--exploratory a14w14,baseline`). As §7 says, an exploratory pass does
  not replace a primary failure or an inconclusive primary.
- **a16w16n14** stays hypothetical, left for later research; it does not run
  at full depth.

**Why.** At two layers, on the four frozen passages, against p64 (record
`session11-2026-10-19/records/run-records/fidelity/two-layer.json`, sha256
`e7905b9d…5d1c24`):

| | pooled mean KL | vs the engine |
|---|---|---|
| a16w16 | 5.84 × 10⁻⁴ | 0.83× |
| engine | 7.05 × 10⁻⁴ | 1× |
| a14w14 | 1.28 × 10⁻³ | 1.81× |

- a14w14 is above the engine on every passage, by 1.65–2.11× in mean KL.
- On Moonlight, at two layers, it was about half the engine (0.53×, that
  engine on the CPU). The K2 evidence, not Moonlight's, decides K2's
  primary, and it favors a16w16 enough to pay its qualification cost.

**What it does not establish.**
- a16w16's mean KL is below the engine's on each passage. Its p99 KL is
  not: on p1 it is 1.115× the engine's (0.907, 0.687 and 0.954 on the
  others).
- §6 bounds p99 at 1.1× at full depth, so the tails decide, and two layers
  are not the criterion.
- The engine figure is llama.cpp on the GPU, replaying each passage as one
  512-token batch. Its float32 kernels for multi-token batches run in TF32
  (§3.3's open question, being settled before the baseline is frozen).
  Full depth scores the baseline's generation-time logits instead.

**Qualification.** a14w14's gate does not qualify a16w16: its tables are
larger and its residual rescale is 4 bits. a16w16 passes its own
implementation gate before the full-depth run:
- exact intermediate agreement with the precision study's a16w16 model;
- the combined-rounding range checks;
- the honest proof accepted by the Rust verifier;
- both tampers rejected: the 4-bit residual rescale and the interleave;
- memory and timings measured beside the baseline.

Its cost, counted (§9.1):
- +15.5% witness and +78% constraints at the two-layer gate, from its
  2^28-entry combine table and the tail's 2^24-entry tables;
- +2.3% witness at full depth.

### 3.6 Amendment 2026-10-21: a16w16's fidelity is an open question

Made after session 12 (2026-10-21) and before any full-depth K2 result. §3.5's
reasoning and results stand as recorded. §6's criterion and the primary are
unchanged; what changes is the expectation.

**a16w16 stays the primary, with no expectation that it passes.** Its
implementation gate is now supported (below). Its fidelity remains open, and
the two-layer evidence is unfavorable.

**The engine along its kernel paths**, at two layers on the four frozen
passages, against p64 (record `session12-2026-10-20/records/records/engine-paths.json`,
sha256 `0793c642…ef2f0b`; the batch replay reproduces session 11's engine arm
to 5·10⁻¹⁴):

| engine path | pooled mean KL | a16w16 against it |
|---|---|---|
| GPU, the 512-token batch replay (session 11's engine arm) | 7.05 × 10⁻⁴ | 0.83× |
| GPU, one token at a time | 3.48 × 10⁻⁴ | 1.68× |
| GPU batch, cuBLAS's TF32 off | 7.11 × 10⁻⁴ | 0.82× |
| labeled CPU batch | 7.48 × 10⁻⁴ | invalid: ran on the GPU (below) |
| CPU, one token at a time | 1.03 × 10⁻³ | 0.57× |

- **Against one-token GPU decoding**, a16w16's mean KL is 1.53–1.80× the
  engine's on every passage, and its p99 KL 1.02–1.18×. a14w14's pooled mean
  KL is 3.67×.
- **§3.5's 0.83×** compares a16w16 with the batch replay, which §3.3 makes a
  diagnostic, not the baseline. It does not support the intended generation
  baseline.
- **The one-token diagnostic is not the generation path either.** It decodes
  position 0 alone (`n_prompt` = 1) and then every token singly. Generation
  prefills the whole prompt in batches and then decodes one token at a time,
  and the prefill sets the cached state. Full-depth acceptance continues to
  score the actual generation-time logits, including the first continuation
  token's prediction from the prefill.
- **Two layers cannot settle depth behavior.** No further precision search
  runs now, and a16w16n14 is not promoted. The batch replay does not become
  the acceptance baseline: that would change the question.

**Next: full-depth fidelity first.** It is the next substantive measurement,
ahead of an expensive full proof where practical, with the criterion and the
primary fixed. The integer computation is reused for proving where possible.
A failure would still be useful evidence about the model, but no proof is
committed on the assumption of a pass.

**a16w16's implementation gate (session 12, shard 1, two layers, T = 100,
bridged).** It passed:
- exact at all 42 intermediates; 185 ranges recorded, every combined-rounding
  range inside its constraint;
- the honest proof accepted by the Rust verifier; the 4-bit residual rescale
  tamper and the interleave rejected; both kept proofs accepted again from
  their files on disk.

Against the 2^12 baseline proved on the same pod:

| | baseline | a16w16 | change |
|---|---|---|---|
| proving | 66.7 s | 100.0 s | **+50%** |
| Rust verification | 150.6 s | 209.5 s | **+39%** |
| peak GPU memory | 50.37 GiB | 57.22 GiB | +13.6% |
| proof file | 837.7 MB | 934.1 MB | +11.5% |

These are at the gate's dimensions. The +2.3% full-depth witness of §9.1 is
a count; amortization at full depth is a prediction until measured.

**The two-layer ranges, recovered whole (session 12, stage R).** Each
configuration's metrics reproduce session 11's within 1.0·10⁻¹³ on every
passage, and the rows outside their constraint are as many as session 11
counted (record `records/ranges/two-layer-ranges.json`, sha256 `73cf6786…1638ac`).
- **Combined rounding**, the drivers' implementation: every range inside
  its constraint, for a16w16, a14w14 and the baseline.
- **Input rescaling**, the rejected alternative (§3.4): three rows outside,
  for a16w16 and a14w14 alike, none for the baseline. a16w16's, over the four
  passages:

| row | signed extremes | constraint | claim |
|---|---|---|---|
| `L0.ckv.x` | [−46,322, 54,340] | −32,768 ≤ v ≤ 32,767 | RMSNorm's s_in: x + 2^15 in the 16-bit slack table |
| `L0.sm.x` | [−160,735, 82,116] | −32,768 ≤ v ≤ 16,744,447 | softmax's s_in: x + 2^15 in the 24-bit aux table |
| `L1.sm.x` | [−137,837, 75,513] | −32,768 ≤ v ≤ 16,744,447 | softmax's s_in, the same table |

**The engine diagnostic on the toys (§3.3), corrected.**
- **The rule's output** was "evidence of a semantic mismatch" for both
  forms. It stands as the rule's output, not as the scientific conclusion.
- **The rule's premise was wrong.** It took one-token GPU execution as a
  float32 control. At `f174553` that fails for MoE layers on NVIDIA:
  `ggml_cuda_mul_mat_id` sends non-quantized experts to `mmf` at any batch
  size (`mmvf` only on AMD), and `mmf`'s `mma` reads float32 bits as TF32.
- **Conclusion: the departure is largely explained by internal precision.**
  The emulation is p64 with the routed experts' operands truncated to TF32:
  - for a one-token evaluation, the down projection's;
  - for batched positions, gate, up and down, as the source predicts.

  The treatment was chosen among 45 candidates on the 12-token one-token
  runs, then applied unchanged to the other runs. The residual it leaves,
  as a share of the GPU's layer-1 error (L2):

  | run | K2's form | Moonlight's form |
  |---|---|---|
  | 12-token one-token runs | 0.1% | 0.5% |
  | 40-token one-token runs | 2.8% | 0.9% |
  | 40-token batch with cuBLAS's TF32 off | 1.6% | 1.7% |

  The rest is unexplained. K2's own routed experts are quantized and take
  `mmvq`/`mmq`, not `mmf`.
- **CPU controls: two invalidated, two audited.** In the CUDA build a
  CPU-labeled condition can run on the GPU: ggml offloads a batch of 32 or
  more tokens whose weights sit in host memory, and the scheduler then places
  nearly the whole graph there. A placement audit (`GGML_SCHED_DEBUG`=2)
  found:
  - **Invalid:** the toys' `cpu-40` (82 of 86 nodes of the batch graph on
    the GPU) and E2's CPU batch (81 of 85). Neither supports a CPU/GPU
    conclusion. With the GPU hidden, `cpu-40` is within float32 on both
    forms.
  - **CPU, audited:** the toys' `cpu-12`, where every node runs on the CPU
    and the logits equal a GPU-hidden run's bit for bit. Also E2's one-token
    CPU path, where every printed graph is on the CPU and the logits equal
    the session's. Graph reuse means three prints cover its evaluations.
  - **K2 with the GPU hidden** aborts in this build (a scheduler assertion),
    so E2's CPU paths have no GPU-hidden twin.

## 4. Metrics

For each scored position t (the position whose next-token distribution
scores continuation token y_{t+1}), each arm's logits go through a float64
log-softmax over the full vocabulary. The engine's float32 logits and the
integer logits at their scale are widened first.

- KL_t = Σ_v p64_t(v) · (log p64_t(v) − log p_arm_t(v)), in nats.
- NLL_t = −log2 p_arm_t(y_{t+1}), in bits.

**Deciding aggregates**, over pooled positions: mean KL, p99 KL, mean NLL,
and max KL as a guard.

**Explaining** (none of these decides anything):
- Routing support: per token and layer, experts selected by the arm and not
  by p64, and the reverse; the number of tokens with any difference.
- Normalized gate distance ½ Σ_e |g64_e − g_arm_e|, on gates normalized to
  sum to 1 over each token's selected experts, so the routed scale (2.827
  for K2, 2.446 for Moonlight) is divided out and models compare on one
  footing. Unselected experts have zero weight. This also captures
  reweighting among unchanged experts. The scaled weights, each expert's
  actual contribution, are reported separately.
- Weight on swapped experts: Σ g64_e over experts p64 selects and the arm
  does not, and the reverse. This is kept separate from the gate distance.
  Neither bounds output error, because expert outputs differ.
- Per-layer residual relative error ‖x_arm − x64‖ / ‖x64‖ at each layer
  output, per position, to locate where divergence starts.
- Selection margins: the 8th-minus-9th gap of s + b per token and layer, as
  a full distribution in every arm, not only for flipped tokens.
- Worst positions: the 10 positions with the highest integer KL and the 10
  with the highest engine KL, each listed with the other arm's value at the
  same position. Otherwise the engine's worst position could hide an
  integer error elsewhere.

Engine routing is reported only where llama.cpp exposes it (its MoE top-k
tensors through the evaluation callback).

## 5. Tooling

### 5.1 The engine runner (reusable)

The runner takes token ids, the prompt/continuation split, the model and the
settings. It emits:
- logits for the requested positions;
- selected layer tensors by name: the layer outputs `l_out-N` and,
  optionally, the MoE top-k ids and weights;
- a reproducibility manifest: the llama.cpp commit and build flags, backend
  and device, settings, GGUF sha256s and tensor list, the token-ids sha256,
  and the capture overhead, measured as wall time with and without the
  callback.

It has two modes: generate (the baseline) and replay. The callback's
synchronization and copying are measured, not assumed free.

Capture changes what is requested, not only how long it takes: with capture
on, the runner asks for outputs at every position, so the last layer's
tensors cover every position. On the CPU toy, capture on and off give
bit-identical tokens and logits (`prover/tests/test_llama_engine.py`). The
kit repeats that check on the target backend before capture is treated as
overhead only.
`analysis/maverick_full_ref.py` already reads `l_out` samples from
llama.cpp's callback log; the runner replaces that with structured
captures.

### 5.2 Architecture adapters (small, per model)

Each adapter holds:
- **A tensor-name mapping**: llama.cpp's names to ours (`l_out-N` ↔
  `L{N}.out`, and so on).
- **Truncated-model construction.** Tensors are extracted by name across the
  whole shard set, never by assuming which shard holds what. The adapter
  requires the embedding, the final norm, the real output head and every
  tensor of blocks 0..n−1. llama.cpp's DeepSeek loader can fall back to tied
  embeddings when the head is absent, which would silently change K2, so a
  missing head is an error here as in our own loader
  (`prover/k2_loader.py: head_name`). Quantized bytes are copied unchanged;
  metadata is copied with the block count rewritten and the split keys
  removed.
- **A loader-branch check.** llama.cpp's DeepSeek loader identifies some
  "lite" variants by layer count, so rewriting the block count can switch
  the loading branch. Moonlight's 27 layers truncated to two is the case at
  hand. The adapter loads the truncated model in the pinned llama.cpp and
  refuses it unless the loaded hyperparameters and branch match the full
  model's: query-LoRA present or absent, head dimensions, expert counts and
  gating.

  **Confirmed on Moonlight** (2026-10-19, `prover/gguf_truncate.py`,
  `prover/tests/test_moonlight_integration.py`). The loader decides "lite"
  as 26 or 27 layers, or 48 with a 128,256 vocabulary. A lite model never
  has its query-LoRA rank read, and Moonlight's GGUF has no such key.
  - Truncated to two layers without a fix, the copy is refused: `key not
    found in model: deepseek2.attention.q_lora_rank`.
  - With `q_lora_rank = 0` written, the copy's print_info hyperparameters
    equal the full model's on every key a truncation keeps.

  K2 truncated to two layers stays in its own branch: 61 layers and two are
  both non-lite, and K2 has query-LoRA.

### 5.3 The torch float64 reference

This is a port of `k2_float_reference` to torch float64 on the GPU, with
layers streamed and expert decoding tiled. To begin with, gguf-py decodes on
the CPU and feeds GPU float64 operations. GPU dequantization follows only
once its output is shown bit-identical to gguf-py's. The port is validated
against NumPy before anything relies on it:
- identical decoded weights (the per-tensor hashes);
- intermediates at every shared name within float64 summation-order
  differences;
- identical routing except at exact float64 ties, which are reported.

It is then benchmarked for memory and runtime while Rust verification
occupies the CPUs on the full-run pod.

**Status (2026-10-19).**
- `prover/k2_float_torch.py` equals the NumPy reference within 10⁻¹²
  relative at every intermediate on the toy, in K2's form and Moonlight's,
  at position 0 and 130,072, with identical routing.
- `prover/gguf_dequant_torch.py` ports gguf-py's dequantizers for F32, F16,
  Q8_0, Q5_0, Q4_K, Q5_K and Q6_K. The port keeps gguf-py's float32
  operations in order, one rounded step each, so eager torch fuses nothing.
  On the CPU it is bit-identical to gguf-py for every bit pattern, from
  random bytes with NaN and infinite scales included.
- `k2_float_torch.DeviceWeights` decodes the experts, the dense and shared
  FFNs and the LM head on the device. It is bit-identical to the CPU
  provider, and so is the forward through it.

The GPU repeats the dequantization test (`VERINF_DEQUANT_DEVICE=cuda`)
before the full-depth reference relies on it.

### 5.4 Emulations

Rounding hooks in K2's NumPy float reference, one per boundary in §3.2's
tables, with a test that the hook set covers every boundary.

## 6. Pre-registered acceptance (frozen 2026-10-19)

### 6.1 What is compared

The chosen integer configuration and the engine baseline (§3.3), each
against p64, at full depth, on the scored continuation positions of the
frozen passages. Only these three arms need full depth.

### 6.2 Bounds

| metric | required bound |
|---|---|
| mean KL | integer ≤ 1.1 × engine + 10⁻⁶ nats |
| p99 KL | integer ≤ 1.1 × engine + 10⁻⁶ nats |
| mean continuation NLL | integer − engine ≤ 0.01 bits/token |
| maximum KL | integer ≤ 2 × engine maximum + 10⁻⁶ nats |

The margins are declared engineering tolerances. They are not derived from
the variation between passages, which belongs in the uncertainty estimate
and does not establish what degradation is acceptable.

### 6.3 Uncertainty and verdicts

**The first three rows use a paired passage bootstrap:**
- B = 10,000 resamples of passages with replacement, from
  `numpy.random.default_rng(20261019)`.
- In each resample, both arms' statistics are recomputed on the same
  resampled passages.
- Excess per resample: integer − (1.1 × engine + 10⁻⁶) for the two KL rows;
  integer − engine − 0.01 for the NLL row.
- **Pass:** the one-sided 95% upper confidence bound on the excess (its
  95th percentile) is ≤ 0.
- **Fail:** the one-sided 95% lower bound (its 5th percentile) is > 0.
- **Inconclusive:** otherwise.

Beating the engine on a point estimate does not override the uncertainty.
The full-corpus point estimates are reported beside the bounds.

**The maximum is a failure guard on the observed corpus,** with no
bootstrap. It is reported with the worst paired positions (§4).

**The study accepts** only when all four rows pass. It fails if any row
fails, and is inconclusive otherwise.

### 6.4 Frozen choices

- **Passages:** N = 4. One is the passage proved in the full run, on the
  prompt the author chooses; three others are chosen by the same rule.
  - Each is a chat-formatted request built with the GGUF's own chat template
    as llama.cpp applies it.
  - Its user turn works from a text written after K2's training data, so the
    continuation is not recall.
  - The four prompts and the sha256 of each prompt's token ids are entered
    in §6.6 before any measurement of any arm.
- **Continuation:** greedy, at most 500 tokens, ending at EOS if EOS comes
  first. A generated EOS is a scored position. Prompts are as tokenized,
  with 400–500 tokens as the target, so T ≤ 1,000.
- **Weighting:** all scored positions are pooled with equal weight, so a
  passage weighs its number of positions. The bootstrap resamples passages,
  never positions, because adjacent tokens are not independent samples.
- **Percentile:** `numpy.percentile(..., method="linear")` over pooled
  positions.
- **Direction:** D_KL(p64 ‖ p_arm), natural logarithm; NLL in bits.
- **Engine configuration:** pinned in the kit before generation (§3.3).

With four passages the bootstrap cannot support a strong generalization
claim. The result speaks for these passages.

### 6.5 Reported beside the criterion

- The 512-token replay against the baseline (§3.3).
- p64's own NLL against the engine's: the raw difference NLL_64 −
  NLL_engine and its excess over 0.01, each with its bootstrap bounds. The
  continuation is the engine's own greedy output, so the NLL row asks
  whether the integer model explains the engine's output as well as the
  engine does. That is the question the U comparison asks, but the engine
  scores its own argmax choices, and a faithful arithmetic can lose NLL to
  it at near-ties for that reason alone. If p64 itself fails this row, the
  report says so.
- **The NLL row scores every continuation position** (decided 2026-10-19,
  before any result). Restricting it to positions where p64 and the engine
  agree would drop exactly the disagreements it must measure. It is not a
  symmetric fidelity measure.
- **If the NLL row alone fails** (or is inconclusive) while the three KL rows
  pass, the report says so: the U-comparison requirement is not met, and
  distributional fidelity, assessed by KL, is. The study's verdict is still
  §6.3's.

### 6.6 Passages (frozen 2026-10-19)

The four user turns are newly written, synthetic prompts, authored for this
study, with provenance and content hashes, in
`analysis/k2-fidelity-prompts-2026-10-19.json`. Each is rendered with the
pinned GGUF's chat template by llama.cpp's Jinja engine, with the assistant
prefix, and tokenized as `llama-completion` does: special tokens added and
parsed. K2 adds no BOS. The template itself inserts its default system
message, since none is given. The tool is `tools/llama_runner/verinf_render`,
run on a metadata-only GGUF built from the pinned shard 1's head. An
independent jinja2 rendering gives the same strings, and no user text holds
special-token text.

**Frozen 2026-10-19, before any measurement**, in
`analysis/k2-fidelity-passages-frozen-2026-10-19.json`. The record holds:
- the token ids;
- the hashes of the prompts file, of each rendered prompt and of the
  template;
- the llama.cpp commit;
- the pinned shard's identity.

The authors accepted p1 (378 tokens) and p2 (398) unchanged on 2026-10-19:
the 400–500 range was a target, and padding the prompts adds no
experimental value. Session 11 re-derives the ids from the full,
sha256-verified shard 1 and requires them equal to these.

| # | id | role | prompt tokens | prompt token ids sha256 (little-endian int32) |
|---|---|---|---|---|
| 1 | p1-research-review | proved | 378 | `f552a8a1355ee532247e4840a694bb15536e744300e85622b27fac3f930683ca` |
| 2 | p2-event-reconciliation | | 398 | `e673b4298273fa2a187d6611c6e457a1b97b7e6f6540ed5450be693b6f93d342` |
| 3 | p3-dependency-schedule | | 400 | `8038e677dc8af4c96f35aad10282b6ded64a88096ac5b4ce7355743c64ac4e3f` |
| 4 | p4-public-notice | | 424 | `9978873e6cc826847ebb0d1ec81b7e4a6d7eb543ee8ac3803aa4116c4e1b16b5` |

All four are within the 1,000-token cap with a 500-token continuation.

## 7. Stages

1. **Two layers** (session 11, after the fixtures pass locally):
   - all six integer configurations, the four emulations, the truncated
     engine model, and NumPy p64 against torch p64;
   - on the four frozen prompts, scoring every prompt position.
   
   Outcome: validated machinery and a provisional ranking. No acceptance.
2. **Full run** (B200):
   - generate each passage's continuation (the baseline, with logits
     captured) and freeze the tokens;
   - the 512 replay;
   - integer logits for each passage;
   - prove the designated passage;
   - torch p64 for each passage on the GPU while Rust verification has the
     CPUs.
3. **Acceptance**, offline, by §6.

If the chosen configuration fails, its proof still stands as a proof of that
integer computation with its fidelity documented, and the next configuration
waits for a later run.

The stages run through `prover/fidelity_study.py`:
- `two-layer`: every arm on each passage's prompt. Each integer
  configuration's ranges are kept whole, per passage and over all four:
  each row's operation, implementation, signed extremes and bounds, and
  the violations listed per implementation, so combined rounding's are
  never counted with the rejected input-rescaling's. Session 11 recorded
  only a count per configuration; the integer configurations run again
  for the rows (`--configs … --arms "" --no-engine`, one process per
  passage, `two-layer-merge`);
- `engine-paths`: the two-layer engine arm along each kernel path (§3.3);
- `full-depth`: the engine's generation, then p64 and the integer arm on
  the frozen sequences, then §6's acceptance and §6.5's calibration. Each
  passage's p64 rows and each configuration's per-position metrics are
  saved as they finish, keyed by the model, layers, GGUF shards, prover
  sources, sequence and configuration, so a stopped run resumes after its
  last whole part. Each configuration's integer ranges over every passage
  are recorded beside the acceptance as proof feasibility. Each quantity's
  signed extremes are checked against the constraint of the claim that
  computes or consumes it: a rescale's signed output window, field safety
  where that is all the claim needs (the residual adds, the products
  before a rescale), the RMSNorm bracket's derived windows, and the
  softmax and sigmoid tables. Where a finer configuration's two
  implementations differ (the claims' input rescaling, or one combined
  rounding), each is checked under its own constraints. Feasibility never
  enters §6's verdict.
- The record's verdict is that of one primary configuration, named before
  the run (`--primary`). The work directory keeps the name, and a resume
  naming another primary is refused. Every other configuration
  (`--exploratory`) is reported beside it. A primary failure or
  inconclusive result remains that result: an exploratory configuration
  passing does not replace it (review of 2026-10-20). The order of the
  configurations plays no part.
- The p64 reference is named too (`--backend numpy|torch`, `--device`). Its
  resolved backend, device, dtype and library version are recorded, and
  they key every part, since each configuration's metrics are scored
  against it. Switching between NumPy and torch, or between the CPU and a
  GPU, recomputes; each reference keeps its own parts.

Engine settings beyond the adapter's are pinned by the caller. It runs end
to end on the toy (`prover/tests/test_llama_engine.py`).

### 7.1 Full depth in two parts (review of 2026-10-21)

Recorded after session 12, before any full-depth K2 run. §6 is unchanged:
this changes how the run is organized, not what it compares. A full proof
waits.

**Reusable artifacts, apart from any arithmetic.** The engine's generation
and the float64 reference are kept as artifacts from which any arithmetic can
be scored later without generating or recomputing either
(`fidelity_study.py` stages `generate`, `reference` and `score`). They run on
different pods.
- **The engine artifact** (F1, `engine-artifact/`). Per passage:
  - the token sequence and the prompt boundary;
  - whether the continuation ended at end-of-generation or at the cap;
  - the logits that chose each token, as generation produced them. The
    scored rows are positions n_prompt − 1 … T − 2, so row 0 is the
    prefill's prediction of the first continuation token.

  With them: the runner's complete configuration, the devices and their
  memory after the context is made, the precision environment, the
  scheduler's placement where it was printed, the GPU topology the kit
  attaches, and the model's and tokenizer's hashes. Flash attention left to
  llama.cpp ("auto") is refused: the frozen configuration names it on or off.
- **The reference artifact** (F2, `reference-artifact/`). p64's float64
  logits at the same rows of every sequence, with the reference's identity
  and the sources' hash. It is completed before any integer arm runs, and
  whatever the arms do. Full depth computes p64 for every passage first.
- **Both artifacts** carry `SHA256SUMS`, checked on loading, together with
  the check that both hold the same sequences.

These artifacts can serve a floating-point prover's arithmetic as well.
Validating such a prover against its own specified arithmetic is separate
work.

**The integer arm at full depth runs on the GPU** (`--implementation
run-only`; `prover/run_only.py`, `demo_k2.run_only`). The CPU model takes
about five minutes per configuration per passage at two layers, which is
days at 61. The run-only pass instead runs the K2 driver's own tape through
the engine pass once, without a proof, keeping only the logits. It reuses
the machinery, not a witness: each value is freed after its last use, and a
proof of the same sequence runs its own pass.

Two checks a proof would otherwise make are recorded as it runs, and form
the arm's feasibility record:
- **Lookups.** Every key is counted at the multiplicity kernel, and those
  outside their table are reported per claim. The engine pass never refuses
  such a key; the proof's LogUp does.
- **Field safety.** A claim's output is trusted once a bound from its
  inputs' magnitudes is within (P − 1)/2. The bound is K·max|a|·max|b|, then
  for a matmul max(|a|·|b|) in float64 when that is not enough. The bounds
  are carried claim by claim from the tape's inputs, since signed extremes
  of reduced values cannot reveal a wrap.

  The claims that only move values, read tables, or compute within windows
  their own lookups enforce are trusted when every input is.

On the toy at two and four layers, at the baseline, a14w14 and a16w16, the
pass's logits equal the NumPy model's and the existing driver's exactly,
with every key inside its table and every bound holding
(`tests/test_k2_run_only.py`). Without a card it runs on Python field
kernels (`tests/_cpu_field.py`); the same cases run on the card's kernels
with `VERINF_RUN_ONLY_GPU=1`.

**Before the engine's configuration is frozen:**
- **The prefill discrepancy is located.** That is the first differing
  intermediate between the batched and one-token paths on K2's weights,
  then the operation tested with identical inputs through the relevant
  kernels. A layer comparison alone cannot tell accumulated rounding from a
  semantic error.
- **The configuration is fixed:** n_ctx 1024, logical batch and microbatch
  512, an f16 KV cache, single-token continuation decoding, and flash
  attention explicit once the diagnostic settles it. No setting is chosen by
  its effect on the integer arm's verdict.
- **The deployment is all-GPU, so multi-GPU execution is qualified on its
  own:** a split run against one GPU on a truncation. The pod is sized from a
  load test with working-memory and KV headroom, not from checkpoint bytes.
- **CPU controls run with the GPU hidden.** For K2 that means a CPU-only
  build, since the CUDA build aborts on K2 without a device.

## 8. Fixtures

- **The tiny synthetic K2 (`prover/tests/_k2_toy.py`),** extended with the
  metadata llama.cpp needs to load it as `deepseek2`: K2's keys from shard
  1's header at the toy's sizes, and tokenizer `none` with `vocab_size`
  dummy tokens. An all-F32 variant isolates definition from arithmetic. The engine runner, the
  adapters, the torch reference, the emulations and the metrics then run end
  to end on a CPU in seconds. The toy keeps covering what Moonlight lacks:
  K2's query-LoRA path, YaRN, and K2's routing shape (one shared expert,
  scale 2.827).
- **Moonlight-16B-A3B-Instruct, Q4_K_M,** for integration:
  - `gabriellarson/Moonlight-16B-A3B-Instruct-GGUF` at `6e9f9f42`, sha256
    `2b113655…df360`, configuration from `moonshotai/Moonlight-16B-A3B-Instruct`
    at `4e735b07`;
  - real weights from K2's architecture family: MLA without query-LoRA, the
    same head dimensions (128 + 64, v 128), 16 heads, d = 2048, 64 experts
    with 6 active and 2 shared, sigmoid selection with a bias, scale 2.446,
    the first layer dense, 27 layers, ε = 1e-5, no YaRN, K2's vocabulary;
  - every arm at two layers and at full depth runs locally, and it is where
    the truncation branch check (§5.2) is first exercised.
  - The references take Moonlight's form by configuration
    (`k2_attention.MOONLIGHT_MLA`, `k2_loader.int_constants`): no
    query-LoRA, plain RoPE, its routed scale and ε. K2's form is unchanged,
    its toy byte-identical.
  - A Moonlight-shaped toy (`_k2_toy.TOY_MOON_MLA`) loads in llama.cpp, and
    in all F32 llama.cpp equals p64 on it as on K2's toy.
  - Moonlight's prompts are the frozen passages' user text rendered with
    its own template. It is an integration rehearsal, not part of the
    study's evidence.

## 9. Cost, provisionally

At two layers session 10 measured T = 100 → 1,000 at 142 → 268 s for the
integer reference and 70 → 149 s for the float pass. Expert decoding
dominates there, not attention. These replace earlier attention-scaling
estimates. The full-depth p64 runtime comes from the §5.3 benchmark. Engine
generation on the full-run pod reads experts from disk; the estimate of 2–4
s per token is unmeasured. The first full-depth generation is timed, and the
remaining passages' estimate updated from it. Four passages were kept with
their roughly $15 increment over one: with a single passage the passage-level
uncertainty would be uninformative.

### 9.1 Proof cost by profile (counted 2026-10-20)

`tools/k2_precision_costs.py` counts the driver's own tapes. They are built
lazily on the CPU at K2's real dimensions, from a sparse stand-in for shard 1
(its real header, no data), and counted by the profiler's extractor. The
routed experts are bridge-held. Full depth is the two-layer tape plus 59
MoE-layer increments. Record: `analysis/k2-precision-costs-2026-10-20.json`.

What each figure is:
- **Witness slots and rows** are exact from the tape.
- **Constraints and quadratic products** come from the profiler's formulas.
  Every one of the tape's 24 claim types now has one; the tail's two were
  added and checked against their compile functions.
- **Proof bytes** are a serialization estimate: T_QUERIES × rows × 11 bytes
  of compact framing, not a measured dump.
- **The live activation set** is the streaming replay's, not total GPU peak
  memory.

The gate (session kit, stage P) measures prove and verify time, the proof's
size and peak GPU memory.

| T = 1000, 61 layers | 2^12 baseline | a14w14 | a16w16 |
|---|---|---|---|
| witness slots | 2.145 × 10^11 | +2.1% | +2.3% |
| linear constraints | 6.73 × 10^10 | +2.7% | +3.1% |
| quadratic products | 8.70 × 10^10 | +2.0% | +2.0% |
| rows at ELL 8192 | 26.21 M | +2.1% | +2.3% |
| proof, compact (estimate, T_QUERIES 54) | 15.6 GB | 15.9 GB | 15.9 GB |
| shared table entries | 104.1 M (30 tables) | 161.8 M (34) | 405.0 M (34) |

- **Per MoE layer, the only difference is the two standalone residual
  rescales:** +71.7 M slots at T = 1000.
- **The rest is tables.**
  - The combine's low-word table is 2^26 entries at a14w14, against 2^24 at
    the baseline.
  - The tail's gap and exp tables are 2^22 for the 256-unit domain at
    2^14, against 2^20.
  - Its remainder table has k = 2^20 entries, against 2^16.
- **The gate is where tables show.** At the two-layer, T = 100 gate a14w14
  is +3.3% in witness and +16% in constraints, most of the latter the
  tables' settlement. a16w16 is +15.5% and +78%, its combine table 2^28.
  At full depth the tables amortize.
- **The live activation set** at T = 1000 over three layers peaks at 32.0
  GiB at the baseline and 33.2 GiB at a14w14, both in the tail's max.

A split low word for the combine's rescale would shrink its table. At full
depth that is under 0.1% of the witness, and it is not done.
