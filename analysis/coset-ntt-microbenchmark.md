# Does the notebook's coset-NTT recommendation actually help? (2026-07-22)

`analysis/verification-parameter-analysis.ipynb` recommends
`Config(K=2**18, pad_proof=128, coset_ntt=True, T_queries=71)` — encoding via
`rho` independent length-`K` coset NTTs instead of one length-`rho*K` NTT —
as its best real design, ~7h prover vs the current K=2^14 design. `coset_ntt`
is a boolean in the notebook's cost-*model* only.

**Confirmed: coset-NTT is not implemented anywhere in the real prover.**
`prover/core.py`'s `_coset_encode_codewords` (the actual Reed-Solomon
row-encoder) does exactly one length-`N_LIG=rho*K_DEG` NTT per batch of
rows. No `coset_ntt` field, env var, or kernel exists outside the notebook.
Testing the recommendation therefore means implementing the encoding, not
flipping a flag.

## Method

Implemented `coset_ntt_encode()` (`analysis/bench/coset_ntt_bench.py`) using
only existing, already-tested primitives (`ntt_forward_batched`, `gl_mul`,
the existing `coset_shift`/`W_N`/`_coset_powers` machinery) — no new CUDA
code, no changes to the real prover. Math: for coset `t` in `[0, rho)`, twist
coefficients by `(coset_shift * W_N^t)^l` and run one length-`K_DEG` NTT;
`codeword[i*rho + t] = coset_t_result[i]` interleaves the `rho` outputs into
the full length-`N_LIG` codeword. Checked bit-exact
(`torch.equal`) against the reference `_coset_encode_codewords` at four
sizes before any timing counted. Deliberately scoped as an isolated
primitive-level A/B, not wired into `prove()` — this answers "does the
intervention help on this GPU" without touching soundness-critical code.

Reproduce: `uv run --project /home/riftuser/VerInf python3 analysis/bench/coset_ntt_bench.py`
(Tesla V100-SXM3-32GB). Results logged to `analysis/bench/prove_runs.jsonl`
(`kind=coset_ntt_ab`); browse with `python3 analysis/bench/show_runs.py --kind coset_ntt_ab`.

## Results

| K_DEG | rho | standard (ns/elem) | coset (ns/elem) | speedup |
|---|---|---|---|---|
| 2^10 (1,024) | 4 | 0.363 | 0.542 | 0.67× (slower) |
| 2^14 (16,384) — current production K | 4 | 0.226 | 0.308 | 0.73× (slower) |
| 2^16 (65,536) — the NTT kernel's Bailey fast-path length | 4 | 0.312 | 0.263 | 1.19× |
| **2^18 (262,144) — the notebook's recommendation** | 4 | 0.443 | 0.351 | **1.26×** |

## Findings

1. **Correct, bit-exact, and real: the coset decomposition works.** Not just
   a cost-model abstraction — a working, verified implementation of the same
   trick, built from pieces already in this codebase.
2. **The benefit is real at K=2^18, but modest on this hardware: ~26%, not
   the multi-x the notebook's abstract `c(n)` curve comparison might
   suggest.** The notebook's own cost model prices coset-NTT by substituting
   `c(K)` for `c(N)` at the *same* total element count — this benchmark
   confirms that substitution is directionally right at K=2^18, just smaller
   in practice than the raw `c(n)` curve gap alone implies (twist +
   interleave add real overhead the notebook's formula doesn't price).
3. **Coset-NTT is a net loss at the current production K=2^14 (0.73×) and
   at this repo's toy/medium K=2^10 (0.67×).** The twist and interleave steps
   are fixed per-call overhead; at small K they cost more than the shorter
   transform saves. It only pays off once K is large enough — empirically
   here, at or above the 2^16 Bailey-fast-path length — which matches the
   notebook's own reasoning for pairing `coset_ntt=True` with a much larger
   K, not adopting it at the current K in isolation.
4. Scope note: this validates the *encoding* step's throughput in isolation.
   It does not measure end-to-end `prove()` impact (would need wiring into
   the real streaming pipeline, a materially larger change touching
   soundness-critical code) or verifier-side implications.

---

# Rigorous A/B: does coset-NTT behave as the theory claims? (authoritative)

The first pass above (5 reps, wall-clock, tiny/mismatched `m`) gave the right
direction but was too crude to answer the real question a small-model test is
*for*: does the optimization behave as the cost-model **theory** predicts, so
we can trust it to scale to large models? Redone properly
(`analysis/bench/coset_ntt_ab.py`): CUDA-event timing, 30 reps, median,
element count held ~2^24 across all K so every point does matched work, and —
the key addition — the measured speedup is compared against the notebook's own
predicted `c(N)/c(K)`, and the full encode is decomposed into
twist / NTT / interleave.

## Test 1 — the theory's core claim (raw NTT, no twist/interleave)

The notebook prices coset-NTT purely by the transform curve: at matched
elements, `rho` length-`K` NTTs vs one length-`N` NTT should cost the ratio
`c(N)/c(K)`. Measured raw batched NTT vs that prediction:

| K | measured speedup | theory `c(N)/c(K)` | agreement |
|---|---|---|---|
| 2^12 | 1.35× | 1.33× | 101% |
| 2^14 | 0.95× (loss) | 0.95× | 100% |
| 2^16 | 1.65× | 1.65× | 100% |
| 2^18 | 1.63× | 1.63× | 100% |
| 2^20 | 1.68× | 1.69× | 99% |

**The theory's core mechanism is validated essentially exactly — 99-101%
agreement at every K.** The coset transform saving *is* `c(N)/c(K)` on this
hardware; the notebook's `c(n)` curve predicts the raw-NTT A/B to within
measurement noise. This is the result that matters for scaling: the mechanism
is real and quantitatively correct, not an artifact of one size.

The `K=2^14` **loss** (0.95×, even on raw NTT) is not noise and not a
contradiction — it is the Bailey 4-step fast path, hard-coded for `N=65536=2^16`
in `prover/kernels/ntt.cuh`. At `K=2^14`, the "big" transform (`N=2^16`) lands
exactly on the optimized path (`c(2^16)=0.351` < `c(2^14)=0.368` ns/elem),
so splitting it into cheap-looking `2^14` NTTs actually costs more. The
theory's measured curve already encodes this dip, so it predicts the loss
correctly (0.95× vs 0.95×). Coset only helps where `N` is off the fast path.

## Test 2 — the full encode primitive (what the theory omits)

The theory prices only the NTT. The real coset encoder also does `rho`× the
twist and one full interleave; the standard encoder does a zero-extend. Full
median times (ms) and the coset breakdown:

| K | m | standard | coset | full speedup | twist | NTT | interleave |
|---|---|---|---|---|---|---|---|
| 2^12 | 1024 | 2.96 | 3.12 | 0.95× | 0.94 | 2.15 | 0.33 |
| 2^14 | 256 | 2.81 | 3.84 | 0.73× | 0.94 | 2.86 | 0.33 |
| 2^16 | 64 | 4.38 | 3.69 | 1.19× | 0.94 | 2.71 | 0.33 |
| 2^18 | 16 | 6.88 | 5.32 | 1.29× | 1.01 | 4.28 | 0.33 |
| 2^20 | 4 | 11.30 | 7.96 | 1.42× | 1.11 | 6.77 | 0.33 |

The twist (~0.94 ms) and interleave (~0.33 ms) are a roughly **fixed ~1.27 ms
overhead** per matched-element batch (they are matched-element work, so
constant across K). The NTT saving grows with `c(N)-c(K)`. So:

- At small K, the fixed overhead swamps a small (or negative, at 2^14) NTT
  saving → full-primitive **loss**.
- At large K, the NTT saving dominates and the overhead becomes a small
  fraction → the full speedup **converges toward** the raw-NTT / theory value
  (2^20: full 1.42× approaching raw 1.68×).

## What this means for scaling to large models

1. **The theory is correct where it makes a claim** (the transform saving =
   `c(N)/c(K)`, 99-101%). Trustworthy to extrapolate: at the large K a big
   model would use, the raw saving is real and matches the curve.
2. **The theory is incomplete as a full-primitive predictor** — it omits the
   twist and interleave. Their cost is a *fixed* per-batch overhead, so it
   matters less and less as K grows, and the full primitive converges to the
   theory. This is exactly the "accurate at scale, pessimistic-to-wrong at
   small K" behavior you want to have *measured* rather than assumed.
3. **The realized full-encode crossover is K ≥ 2^16**, now explained
   mechanistically (fixed twist+interleave vs. a growing NTT saving, plus the
   `N=2^16` fast-path dip), not just observed.
4. Caveat unchanged: this is the encode primitive in isolation. End-to-end
   `prove()` impact also depends on encode's *share* of total time (small —
   witness_recompute dominates; see analysis/toy-transformer-prove-time-formula.md),
   which is a separate axis from "does the transform trick work as theory says"
   (it does).

Reproduce: `uv run --project /home/riftuser/VerInf python3 analysis/bench/coset_ntt_ab.py`.
Logged as `kind=coset_ntt_ab_raw` / `coset_ntt_ab_full` in
`analysis/bench/prove_runs.jsonl`.
