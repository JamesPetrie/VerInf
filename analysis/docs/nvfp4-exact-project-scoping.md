# Exact NVFP4 emulation under Freivalds — project proposal and initial scoping

**Status:** proposal / initial scoping (2026-07-22). Scopes a project to model
NVFP4 matmuls exactly inside VerInf's proof system: motivation, what is already
established (and how much to trust it), the design as it stands, known failure
modes, and a proposed sequence of work items. Everything here is a summary — each
section links to the document that carries the depth. Assumes no prior familiarity
with the codebase; §3 defines the project-specific vocabulary.

## 1. The problem, in one page

VerInf produces zero-knowledge proofs of LLM inference ([`README.md`](../../README.md),
[`paper.md`](../../paper.md)). The proof works over a finite field, but real inference
runs in floating point, whose arithmetic is non-associative: results depend on
summation order, and in deployments whose kernels are not batch-invariant, on
which other requests happen to be batched alongside a given one. So VerInf does
not demand bit-exact reproduction. Instead it proves the
committed outputs are *well explained* by a committed finite-field computation and
certifies a bound on the **unexplained information U** — the output bits per token
the computation does not account for. Lower U = stronger guarantee; U is the
proof's only public value.

The measured state of the art ([`quantization-evaluation.md`](../quantization-evaluation.md)):
the current Int64/Q4.14 integer model with a single global scale adds **+0.33
bits/token** of quantization divergence over the FP floor (0.053), and the excess is
bimodal — 11 of 16 test prompts contribute almost nothing, while five prompts with
outlier activation channels contribute 0.3–0.8 each. A single global scale cannot
follow outlier channels.

The opportunity: Blackwell-class deployments *actually serve* in NVFP4 — 4-bit
E2M1 elements with an E4M3 scale per 16-element block and one FP32 scale per
tensor. If the proof models that arithmetic exactly, then for every weight matmul
the committed computation **is** the deployed computation, bit for bit, and the
quantization-divergence term of U vanishes for those ops. Per-block scaling also
fixes the outlier-channel failure by construction rather than by tuning. That is
the project.

The thesis in one sentence, from the hardware measurements below: **the NVFP4
matmul itself is integer-exact; the only non-integer step is the rounding
afterwards** — so the matmul reuses VerInf's existing Freivalds machinery
unchanged, and the new work is a gadget that pins the rounding.

## 2. What is established, and how much to trust it

[`nvfp4-matmul-predictability.md`](../nvfp4-matmul-predictability.md) is the
measured groundwork — self-contained, defines the format from scratch, and honest
about its limits. The results form a **confidence ladder**; keeping the rungs
distinct is essential (see Research Integrity in the root `CLAUDE.md`: measured ≠
proven, and neither may masquerade as the other).

**Rung 1 — deductive, fully trusted.** An NVFP4 value has an exact integer
representation: element × block scale is a signed integer of magnitude below 2^22
on a common 2^-10 grid; the per-tensor FP32 scale factors out of the matmul as one
exact scalar. The exact accumulator satisfies `P < K·2^44`, so the Goldilocks field
(≈2^64) represents it exactly for K up to ≈2^19 (≈2^23 if block scales are
restricted to normal E4M3). This is arithmetic, not measurement. *If* the hardware
accumulator equals the ideal decode-and-dot, everything downstream is on solid
ground.

**Rung 2 — measured exhaustively, in a narrow regime.** On a GB10 (Blackwell,
sm_121) via `torch._scaled_mm_v2`, the hardware accumulator equalled the ideal
fp64 decode-and-dot on **every element** tested (max_abs_err = 0.0) for K ≤ 16384;
first deviations (~1–2 ULP on ~0.02% of elements) appear at K = 65536. The
requantized FP4 *output* matched the software model bit-for-bit at every K tested,
because the coarse output grid absorbs the tiny accumulator noise.

**Rung 3 — statistical only.** Above K = 16384 the output match is a bound
(mismatch rate below ~1.5×10⁻⁵ in the only regime where mismatch was possible),
not a proof; the estimated true rate (~10⁻⁷–10⁻⁸) is below what the experiment
can resolve.

**What has *not* been established.** These gaps are the project's de-risking work,
not footnotes:

1. **Input-distribution dependence.** The harness
   ([`nvfp4_matmul_test.py`](../nvfp4_matmul_test.py)) tests zero-mean Gaussian
   inputs only, at M = N = 256, 3 seeds. Gaussian inputs cancel — accumulator sums
   grow like √K. Sign-aligned or adversarial inputs grow like K and could exhaust
   the FP32 mantissa at K well below 16384, which would break exactness *inside*
   the currently-trusted range. Real activations are not Gaussian (post-SiLU FFN
   activations are skewed; outlier channels are the whole §1 story). "Bit-exact
   for K ≤ 16384" may be a distribution-dependent result being read as a regime
   result. Nobody has tried to break it.
2. **No mechanistic model.** The agreement is a black-box measurement; NVIDIA does
   not publicly specify the tensor-core accumulation semantics (internal adder
   width, where the block-scale multiply happens, reduction-tree shape). A
   measurement without a mechanism generalizes poorly.
3. **One kernel, one chip, one torch build.** `torch._scaled_mm_v2` on one GB10
   with torch 2.13.0.dev. Real serving stacks (TRT-LLM, cutlass/cuDNN kernels,
   vLLM) use different kernels, possibly with split-K; split-K with atomics would
   make outputs run-to-run nondeterministic, in which case no single emulation
   matches every run. Other Blackwell parts (B200, GB200, RTX 50-series) need not
   behave identically.
4. **The requantization recipe is a separate axis entirely** — see §5.

The defensible claim today: *on GB10 via `_scaled_mm_v2`, with Gaussian inputs at
M = N = 256, the accumulator was bit-exact through K = 16384, and it coincides with
an exact integer semantics.* The claim the project needs — *the deployment's NVFP4
matmuls are exactly predictable* — requires closing gaps 1–3.

## 3. Background and terminology

One line each, with links to the full definitions.

- **Freivalds / MatmulClaim** — verify `C = A·B` by random projection: check
  `r^T C = (r^T A)·B` for random `r`, O(n²) work, never committing the n³
  products. VerInf's workhorse. Paper Appendix B.2; listing-level spec in
  [`CLAIM_SPECS.md`](CLAIM_SPECS.md).
- **Paired lookup** — a table argument certifying a committed value appears in a
  public table (e.g. "this 4-bit code is a valid E2M1 value"). Paper §4.4.
- **MaxClaim / one-hot gadget** — proves a committed value is the max of a set via
  a one-hot selector, booleanity/cardinality checks, and range-checked gaps.
- **Bracket gadget** — pins the result of a rounding/division without performing
  it: table the representable outputs with their rounding boundaries, then force
  `lo(out) ≤ input < hi(out)` with range-checked slacks. Existing examples:
  softmax, RMSNorm rsqrt (paper B.3, B.4).
- **Uniqueness discipline** — every claim must pin its witness to a single valid
  assignment (or prove remaining freedom is one-sided/value-neutral). Paper §2.2,
  Appendix B.1; audit checklist in
  [`degrees-of-freedom-review.md`](degrees-of-freedom-review.md).
- **U and the predictor Q** — U is measured against a prover-chosen predictor Q of
  the deployment's outputs given the committed computation; better Q shrinks U at
  zero proof cost. Paper §2.1, §5.

## 4. The design as it stands

[`nvfp4-exact-path.md`](nvfp4-exact-path.md) is the design doc — a claim-set
sketch, not an implementation. Compressed:

- **Matmul claim: unchanged.** Commit the per-element integers with block scales
  absorbed (rung 1 above); `C = A·B` over these is precisely today's MatmulClaim.
  Per-tensor FP32 scalars ride along as public constants.
- **Format pin (new).** The committed integers must be shown to *be* NVFP4: per
  element one quadratic `â = e·ŝ` against a block-broadcast replica of the scale,
  plus lookups certifying `e ∈ E2M1` and `ŝ ∈ E4M3`. This also binds the weight
  commitment to the actual deployment artifact (the FP4 checkpoint).
- **Requantization claim (new, the heart of the project).** Rounding an exact
  accumulator block back to FP4 codes, division-free: (1) `amax` via the MaxClaim
  gadget; (2) block scale `ŝ = RN(amax/6)` pinned by a 256-entry midpoint-bracket
  table; (3) each code `q = RN(acc/ŝ)` pinned by brackets against the 8-value
  E2M1 grid, with round-to-nearest-even ties encoded as inclusive/exclusive
  boundaries and a saturating mux at ±6. Estimated ~10–12 witness slots per
  output element (~1.4–1.6× the current Int64 rescale block).
- **Scope boundary.** Attention (K = S) and FP32 reductions (softmax sums, RMSNorm
  energies) are excluded — they stay on the current integer path or move to
  per-op similarity claims (paper §9). BF16 unary ops become single 2^16-entry
  lookups, *cheaper* than today's gadgets.

## 5. Known failure modes

Each has an existing scar or writeup:

1. **The requantization recipe, not the tensor core, is the likeliest real-world
   mismatch** (design doc §2.5.2b). Frameworks differ on the scale formula and
   clamping, RNE vs other rounding modes for the E4M3/E2M1 casts, zero-block and
   NaN conventions, and — subtly — dividing by the scale vs multiplying by its
   FP32 *reciprocal*, which rounds differently near midpoints. The emulation must
   match the deployment's specific recipe, characterized empirically the way
   [`nvfp4_scale_layout_bruteforce.py`](../nvfp4_scale_layout_bruteforce.py)
   characterized torch's scale layout.
2. **Vacuous brackets.** Every bracket operand must be independently range-bounded
   or the inequalities lose integer semantics in the field
   ([`degrees-of-freedom-review.md`](degrees-of-freedom-review.md) §3). This is
   not hypothetical: the RMSNorm rsqrt bracket was once vacuous in the production
   configuration and **accepted a forged output** before being fixed
   ([`rmsnorm-bracket-fix.md`](rmsnorm-bracket-fix.md)) — required reading before
   designing any new bracket.
3. **Distribution-dependent exactness** (§2 gap 1) — could invalidate the trusted
   K range itself.
4. **Nondeterministic kernels** (§2 gap 3) — if the serving kernel uses atomics,
   there is nothing to emulate exactly; the deployment must pin deterministic
   kernels.

**The standing discipline** (design doc §4): exact-emulation differential tests
against real hardware first, then prove→ACCEPT, then a U measurement. Exactness is
never claimed from the software model alone, and negative tests (forged-witness
rejections) stay in step with every new check.

## 6. Proposed work items

Roughly in dependency order; items 1–3 need Blackwell hardware (see §8), items
4–5 need none.

1. **Adversarial exactness stress test** (closes §2 gap 1). Sign-aligned,
   max-magnitude, and boundary-adjacent inputs; real-model activation
   distributions; sweep M/N/K shapes. Either exactness survives — a much stronger
   claim — or it breaks, and the design needs the true accumulator semantics.
2. **Characterize a serving kernel, not just torch's** (gap 3). Same differential
   harness against TRT-LLM/cutlass NVFP4 paths; check run-to-run determinism.
3. **Requantization-recipe characterization** (failure mode 1). Pin down one
   deployment's exact recipe, including tie behavior, empirically.
4. **The format-pin gadget** (design §2.2) — the smaller of the two new claims;
   pure proof-system work, no GPU needed.
5. **The requantization gadget** (design §2.3) — the main event: MaxClaim + two
   bracket families + lookups, with a uniqueness argument per
   [`degrees-of-freedom-review.md`](degrees-of-freedom-review.md), differential
   tests, and negative tests.
6. **End-to-end U measurement on an FP4-served model** — the payoff number.

Deliberately *out of scope* (other levers on U, ranked in the design doc §3):
per-channel scales on the current integer path, better predictors Q, committing
the sampler. They are worthwhile but separate; this project should not absorb
them.

## 7. Background reading

In order:

1. [`README.md`](../../README.md) — motivation and the unexplained-information framing.
2. [`paper.md`](../../paper.md) §2 (what VerInf proves), §4.2 (Freivalds), §4.4
   (lookups), Appendix B.1 (shared machinery, uniqueness) — skim B.2–B.5 for the
   gadget idiom.
3. [`quantization-evaluation.md`](../quantization-evaluation.md) §1, §6–7 — where
   +0.33 bits/token comes from; why global scales fail.
4. [`nvfp4-matmul-predictability.md`](../nvfp4-matmul-predictability.md) — the
   measurements (full read).
5. [`nvfp4-exact-path.md`](nvfp4-exact-path.md) — the design (full read).
6. Reference as needed: [`CLAIM_SPECS.md`](CLAIM_SPECS.md),
   [`degrees-of-freedom-review.md`](degrees-of-freedom-review.md),
   [`rmsnorm-bracket-fix.md`](rmsnorm-bracket-fix.md),
   [`ARCHITECTURE.md`](../ARCHITECTURE.md) ("NVFP4: research direction").

## 8. Hardware and resource requirements

The measurement harnesses need a **Blackwell-class GPU** — any of sm_100 (B200),
sm_120 (RTX 50-series), or sm_121 (GB10/DGX Spark); cloud instances of the first
two are widely available and fine for this work. Requirements: a CUDA 13-era
driver and a PyTorch recent enough to have native `float4_e2m1fn_x2` and
`torch._scaled_mm_v2` (the measurements used a 2.13.0-dev build; `pip install
expecttest` for a torch test-helper import). No model weights are needed:

```
python analysis/nvfp4_matmul_test.py             # K/seed sweep: accumulator + output match
python analysis/nvfp4_scale_layout_bruteforce.py # finds the scale-layout convention
```

Two caveats. The validated `_scaled_mm_v2` scale-layout convention (swizzled
`to_blocked` + `SWIZZLE_32_4_4`) was found by brute force on the GB10 and may need
re-brute-forcing on other parts — that's what the second script is for. And
per-chip differences are not noise to route around: they are work item 2, so any
reported number should record exactly which chip/driver/torch produced it.

Gadget-design work (items 4–5) needs no GPU: the prover/verifier and their tests
run on CPU (see [`ARCHITECTURE.md`](../ARCHITECTURE.md) for the codebase map).
