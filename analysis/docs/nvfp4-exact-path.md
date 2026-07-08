# Increasing the explained fraction: the NVFP4-exact path and other levers

**Status:** design for review (2026-07-04). Synthesizes the measured groundwork in
`analysis/nvfp4-matmul-predictability.md` and `analysis/quantization-evaluation.md`
into a concrete claim-set design for exactly modeling NVFP4 matmuls under
Freivalds, plus a ranked list of the other levers on the unexplained-information
bound. Replaces the dangling `quantization-emulation.md` reference in the
quantization evaluation.

## 1. Where the unexplained bits come from today (measured)

From `quantization-evaluation.md` (Llama-2-7B, 16 WikiText prompts, calibrated σ):

- FP-baseline floor: **0.053 bits/token** — what the logit-noise predictor charges
  even for the FP reference against itself.
- The Int64/Q4.14 committed model adds **+0.33 bits/token** (total 0.383; σ = 0.076;
  top-1 agreement 98.9%).
- The +0.33 is **bimodal**: 11 of 16 prompts contribute < 0.02 bits/token; five
  prompts contribute 0.3–0.8 each — the documented failure mode of single-global-scale
  quantization on outlier activation channels (per-channel scales are unimplemented).

So the quantization divergence between the committed integer model and the FP
deployment is nearly all of the non-floor unexplained information, and it is
concentrated where a single global scale can't follow outlier channels. Maverick's
demonstrated 0.394 bits/token has the same structure.

## 2. The NVFP4-exact path

### 2.1 What is already established

`nvfp4-matmul-predictability.md` measured, on Blackwell (GB10) hardware:

- The NVFP4 matmul accumulator is **bit-exact** against the ideal
  decode-and-dot-product for K ≤ 16384 (max_abs_err = 0.0, every element); first
  deviations (~1–2 ULP on ~0.02% of elements) appear at K = 65536.
- The requantized FP4 **output** matched the software model bit-for-bit at every K
  tested (the coarse output grid absorbs the tiny accumulator noise) — a
  statistical bound, not a proof, above K = 16384.
- The exact integer representation: an E2M1 element times its E4M3 block scale is
  a signed **< 2^23** integer on a common 2^-10 grid; the per-tensor FP32 scale
  factors out as one exact scalar; the exact accumulator obeys P < K·2^44, so
  Goldilocks stays exact for K ≲ 2^19.

This validates the core premise: **the matmul itself is integer-exact; the only
non-integer step is the rounding afterwards.** What follows is the claim-set
design for both halves.

### 2.2 The matmul claim: unchanged

Commit Â, B̂ — the per-element integers with block scales absorbed — and the exact
accumulator Ĉ. `C = A·B` over these integers is *precisely today's MatmulClaim*:
double Freivalds, O(k) auxiliary witness, the same soundness. Nothing in the
Ligero layer, the fold, or the verifier changes. The per-tensor FP32 scalars ride
along as public constants folded into downstream claims (they multiply the
Freivalds check's b-side coefficients exactly).

Two additions around it:

- **Format pin.** The natural committed form of an NVFP4 tensor is its codes:
  E2M1 elements e (16 values) and E4M3 block scales ŝ (≤ 256 values). The proof
  must pin â = e·ŝ_block, or the prover could commit â outside the format. Per
  element: one quadratic â = e·ŝ_rep against a block-broadcast replica of ŝ (the
  MaskedCombine broadcast pin pattern), plus one paired lookup certifying
  e ∈ E2M1 and one certifying ŝ ∈ E4M3 (per block, amortized /16). This also makes
  the *weight commitment* a commitment to the actual deployment artifact (the FP4
  checkpoint), which is what an auditor wants bound.
- **Field-width scope.** K ≤ 2^19 worst-case (2^23 with normal-only scales) covers
  every weight matmul in Maverick (k = d = 5120 or d_ff = 8192) and Llama-7B.
  Attention's AV product has K = S and is excluded (below).

### 2.3 The requantization claim: rounding as brackets and lookups

The deployment's requantization of an exact accumulator block (16 values acc_i,
integers at a common scale) to the next tensor's FP4 codes is:

1. amax = max_i |acc_i| — a 16-way max: the existing MaxClaim gadget (one-hot,
   booleanity/cardinality, range-checked gaps), amortized per block.
2. ŝ = RN(amax / 6) into E4M3 — pinned *without division* by a midpoint bracket:
   a 256-entry paired table holds each representable ŝ with its lower/upper
   rounding boundaries (midpoints to neighbors, pre-multiplied by 6, exact on the
   common grid); two quadratic bracket constraints with range-checked slacks force
   m_lo(ŝ) ≤ amax < m_hi(ŝ). Round-to-nearest-even ties are encoded per table
   entry by making the appropriate boundary inclusive/exclusive (±1 on the integer
   grid — exact, since both midpoints and accumulators live on a common dyadic
   grid).
3. q_i = RN(acc_i / ŝ) into E2M1 — same midpoint-bracket idea against the 8-value
   non-uniform grid: mid_lo(q)·ŝ ≤ 2·acc_i < mid_hi(q)·ŝ, two quadratics (the
   midpoints multiply the *committed* ŝ, keeping it division-free), a 16-entry
   lookup for q, tie handling as above, and a saturating mux at ±6 (the SiLU
   saturation pattern).

Uniqueness (§3.3 of the paper, no degrees of freedom): the max gadget pins amax;
strict monotone brackets pin ŝ and each q_i to single integers; every slack is
range-checked. Estimated witness cost ≈ 10–12 slots per matmul-output element
(vs ~6–8 for today's Int64 rescale block) plus ~2 per element for the format pin —
roughly **1.4–1.6× the per-element cost on matmul outputs**, offset by elements
being 4-bit rather than requiring wide range tables.

### 2.4 What it buys, and the scope boundary

When the deployment itself serves NVFP4 (the Blackwell serving path), the
committed matmul computation **is** the deployed computation, bit for bit, in the
measured regime — the quantization-divergence term of U vanishes for every weight
matmul. Two bonuses:

- **Per-block scaling is native**, which is precisely the per-channel-scale fix
  for the outlier-channel bimodality of §1 — the five bad prompts' failure mode
  disappears by construction rather than by tuning.
- The FP4 output grid **absorbs** upstream computational noise (measured: output
  codes matched even where the accumulator deviated), so bitwise agreement per
  layer is the common case, not a fragile hope.

The boundary: attention (K = S, higher-precision in deployments anyway) and the
elementwise/normalization ops stay on the current integer path or move to
deployment-precision modeling:

- **BF16 unary ops are a gift**: a BF16→BF16 SiLU is *one 2^16-entry paired
  lookup on the raw codes* — cheaper than the current integer SiLU gadget (no
  sign split, no word decomposition, no mux). Same for any unary op.
- **Reductions in FP32** (softmax row sums, RMSNorm energies) round per-add and
  are order-dependent; exact emulation of a *pinned* reduction order is possible
  (a bracket gadget per add) but expensive. The pragmatic hybrid: per-op
  similarity claims (paper §9) — commit the deployment's actual intermediate,
  prove |committed − ideal_int(inputs)| ≤ ε with a range check — which stops
  divergence from compounding across layers without exact emulation.

### 2.5 Open questions (ordered by risk)

1. **Rare-disagreement rate near rounding boundaries.** The output-match result is
   statistical; adversarial sampling near E2M1 midpoints should characterize the
   true rate (est. 1e-7–1e-8 at K = 65536; likely zero in the exact-accumulator
   regime). A residual rate ε just reappears as ~ε·log V in U — quantify, don't
   assume zero.
2. **Kernel K-split behavior**: serving kernels may split K with FP32 partials
   combined in a tree or via atomics; atomics make the output run-to-run
   nondeterministic, in which case no single emulation matches every run. The
   deployment must pin deterministic kernels in the exact regime
   (deployment-controlled).
2b. **The requantization recipe, not the tensor core, is the likeliest
   real-world mismatch source**: frameworks differ on scale formula and
   clamping, on round-to-nearest-even vs other modes for the E4M3/E2M1 casts,
   on zero-block and NaN conventions, and — subtly — on dividing by the scale
   vs multiplying by its FP32 *reciprocal*, which rounds differently near
   midpoints. The emulation must match the deployment's specific recipe,
   characterized the way `nvfp4_matmul_test.py` characterized torch's.
2c. **Bracket-operand bounding**: per the degrees-of-freedom review
   (`degrees-of-freedom-review.md` §3), every operand of the requantization
   brackets (the accumulators, amax) must be independently range-bounded so
   the inequalities have integer semantics in the field.
3. **The elementwise seam**: how much of the remaining divergence lives in
   softmax/norms once matmuls are exact — measure with the ablation harness
   (`--ablate-submodule`) before choosing exact-emulation vs ε-claims per op.

## 3. Other levers, ranked by bits-per-effort

1. **Per-channel weight scales in the current integer path** (S effort). Directly
   targets the measured bimodality (§1) without any format change: per-channel
   scales on weights are *public constants*, so they fold into the Freivalds
   coefficients and rescale constants — no new witness, no new claim type. Worth
   doing even if the NVFP4 path proceeds, as it derisks the near-term Maverick
   numbers. Plausible effect: most of the +0.33 → ≲ +0.1 bits/token (11/16
   prompts are already ≈ 0).
2. **Better predictors Q** (zero proof cost, prover-side only). σ per position or
   as a function of logit-gap/entropy; a full-softmax predictor rather than the
   top-1 gap; held-out calibration (the current σ is in-sample). Any improvement
   multiplies through the whole bound, including the 0.053 FP floor.
3. **Commit the sampler** (M). At temperature > 0, sampling entropy would dwarf
   quantization divergence in U. Committing the sampling seed and proving
   o_i = sample(logits_i, seed_i) — categorical sampling as committed uniforms +
   the existing argmax/one-hot gadget over cumulative sums — removes the sampling
   term entirely. Required before any T > 0 deployment claim is meaningful.
4. **Per-op similarity claims** (M; paper §9). The compounding-stopper — valuable
   independently, and the natural companion of §2.4's hybrid.
5. **Table refinements** (S; already characterized). The evaluation's optimum
   (scale 2^14, silu_x_max = 20, T = 2^16) is implemented; further gains are
   secondary to lever 1.
6. **Integer-exact deployment co-design** (L). Serve with INT4/INT8 integer
   kernels and match exactly by construction — the strongest guarantee, the
   heaviest deployment ask; NVFP4-exact (§2) achieves the same effect on the
   dominant ops *without* changing how anyone serves.

## 4. Suggested sequence

1. Lever 1 (per-channel scales) + lever 2 (predictor) — cheap, measurable on the
   existing 7B harness, likely halves-or-better the non-floor U.
2. §2.5's measurements (boundary sampling; elementwise-seam ablation).
3. The NVFP4 matmul + requant claims behind a flag, gated by the standing
   discipline: exact-emulation differential tests against `_scaled_mm_v2` on
   hardware, then prove→ACCEPT, then a U measurement on the FP4-served model.
4. Lever 3 (sampler) when a T > 0 deployment matters.
