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
only the chosen configuration runs. Its logits come from the driver's engine
pass, or, for the proved passage, from the proof's own witness.

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

### 3.4 The a14w14 driver: combined rounding (2026-10-20)

a14w14 is the nominated K2 primary, pending the implementation and cost
gates (review of 2026-10-20). `demo/demo_k2.py --precision` builds the tape
at a profile, `k2_loader.Scales`. Each scale has one meaning:
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
- `two-layer`: every arm on each passage's prompt;
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
