# Improvements roadmap — toward a secure Llama-400B inference proof on DGX Spark

This document catalogues the known gaps between the current Ligero pipeline
and the eventual goal of a clean, secure, performant zero-knowledge proof
of a Llama-400B forward pass on a DGX Spark. It is organised by area and
prioritised by the effort/value trade-off observed so far.

For the *measured* prove-time breakdown and the q_lin (linear-constraint) fold
optimization options — what helps, what's measured-dead, and the sparse-handling
lever — see `prover-optimization-investigation.md`.

Status as of 2026-05-28:
- 32L Llama-2-7B SEQ=2 with random inputs: ACCEPT, 565s prove
- 4L Llama-2-7B SEQ=91 with real "Hello world" prompt: ACCEPT, 901s prove
- 1L Llama-2-7B SEQ=91 with real prompt: ACCEPT, 301s prove
- 32L SEQ=91 with real prompt: not yet validated end-to-end with current code

---

## 1. Streaming gaps

The pipeline advertises "streaming" but only partly delivers it. The current
streaming is at the per-chunk encoding level inside `prove()`. The compute
side caches.

### 1.1 Phase-1 engine pass caches all outputs

**Where:** `Tape.run_engine_pass` in `tape.py:1254`. Walks all claims in
`self._deferred`, computes each `COMPUTE_FN`, accumulates outputs into a
`live` dict, then writes the entire dict back to `tape.inputs` with
`self.inputs.update(live)`.

**Implication:** between the engine pass finishing and `prove()` starting
its encoding work, every phase-1 output for every layer is resident in
memory simultaneously. At 32L Llama-2-7B SEQ=91 this is a few GB — fits
in 121 GB on Spark, doesn't fit at 400B with longer contexts.

**Fix (Phase 2c):** restructure the engine pass to compute each claim's
outputs *just in time* during the encoding loop. Walk claims in order;
encode each claim's phase-1 outputs into the current chunk as they're
produced; evict outputs whose last reader has passed. ~150 LoC, plus a
small `_compute_last_use(claims)` helper and special handling of mult
vars (which aggregate across claims and can only be encoded at the end).

**Live-set bound for Llama-style graphs:** residual + current layer's
working set + persistent mult vars ≈ 1-2 GB regardless of layer count
or sequence length.

### 1.2 Phase-2 witness (AUX_FNS) caches across rounds in real protocol

**Where:** `core.prove()` round 3 + 4 both call `_compile_with_chs` and
re-run `AUX_FNS` from scratch when the function is invoked separately
per round. The output `witness` dict holds ~1 GB at 32L Llama-7B.

For test mode (`returnEverything=True`) this isn't a problem — it lives
in one stack frame. For real interactive protocol where rounds 2, 3, 4
are separate calls, each call recomputes the witness fresh and holds it
through that call. Memory peak per call is bounded but the recompute cost
is paid every round.

**Fix (Phase 2d):** lazy AUX_FNS that fire per-chunk during the encoding
loop, mirroring Phase 2c for phase-1. AUX_FNS is pure (deterministic
from claim + challenges), so re-evaluation per chunk is sound. Cost is
moderate refactor of `compute_p_0_streaming` and the round-3/4 encoding
paths to compute aux on demand instead of upfront.

### 1.3 Real-protocol round 4 redoes round-1 + round-2 work

**Where:** `prove()` with explicit `ch0, ch1, ch2` and
`returnEverything=False`. Currently the stateless rounds re-encode p1 and
p2 in every round. Round 4 specifically re-builds the merkle trees from
scratch in order to extract paths.

**Implication:** real interactive protocol pays ~4× the encoding work
of test-mode fused (because each round re-derives everything).

**Acceptable for now.** At 32L this is 4 × (engine pass + encode) ~ tens
of minutes total — network latency between verifier rounds dominates in
practice. Could be revisited if real deployment latency budget proves
tight; the fix is moderate (cache merkle artifacts between rounds via a
small `ProverState` bundle), but contradicts the current stateless
design and adds complexity.

---

## 2. Efficiency optimisations (with measured numbers)

All numbers from 2026-05-28 benchmarks on Spark GB10 with Llama-2-7B
weights at output_width=24, SEQ=91, real "Hello world" prompt.

### 2.1 RmsNorm aux compute is CPU-numpy bound

**Measured:** per-call cost 10-11 sec at SEQ=91 (1L test, n=3 RmsNorm
calls totalled 30s). Of that, GPU time is < 1 sec; the rest is
`.cpu().tolist()` round-trips and Python list comprehensions over
373K-element arrays.

**Hot path in `compute_fns.py:270` `rmsnorm_compute`:**
```python
x_in_cpu = live[claim.x_in].cpu().tolist()          # 373K-element GPU→CPU + Python list
x_low_cpu  = [v & (k_resc - 1) for v in x_in_cpu]  # Python comp
x_high_cpu = [v >> r for v in x_in_cpu]            # Python comp
x_shifted_cpu = [(v + offset) % P for v in x_high_cpu]  # Python comp
X_sq_2d = X_sq_t.view(B, d).cpu().numpy().astype(np.uint64)  # GPU→CPU+numpy
X_sq_l_view = X_sq_2d.tolist()                      # numpy → Python list
```

**Fix:** rewrite as pure-GPU uint64 torch ops. Binary search for `y_int`
in 18 fixed iterations on GPU; reformulate the comparison as
`y² ≥ ceil(magic / S_total)` to stay in uint64 (avoids 128-bit
intermediates). ~60-80 LoC.

**Expected speedup:** 5-10× on RmsNorm calls.

**Impact at 32L SEQ=91:** RmsNorm currently ~32 × 30s ≈ 16 min. Would
drop to ~2-3 min. Savings ~13 min on a ~hours-long prove.

### 2.2 `_signed_floor_decomp` runs on CPU numpy

**Where:** `tape.py:208`. Called by matmul, hadamard, rope, rmsnorm output
rescale, and softmax's c2 rescale. The function transfers the input from
GPU to CPU, runs numpy operations (bit ops, modular adjust), and transfers
back.

**Cost:** ~1-2 sec per call at 373K-element input (matmul output at
SEQ=91). At 32L with ~10 matmuls per layer = 320 calls × ~1s = ~5 min
just in `_signed_floor_decomp`.

**Fix:** pure-GPU torch implementation. The function does:
1. signed reinterpret (uint64 bit-pattern → int64 sign)
2. floor division and modulo by `k_resc`
3. offset add + mod-P correction for sign

All available as torch ops on uint64 / int64 with care for the
field-rep mapping. ~30-40 LoC.

**Expected speedup:** 5-10× per call.

**Impact at 32L SEQ=91:** saves ~3-4 min.

### 2.3 `gl_matmul` Goldilocks GEMM is slow vs theoretical

**Measured:**
| matmul | shape | wall time | theoretical (FP) | ratio |
|---|---|---|---|---|
| Q/K/V | 91×4096×4096 | 6.7s | ~1ms | ~6700× |
| W_gate, W_up | 91×4096×11008 | 15.8s | ~3ms | ~5000× |
| W_down | 91×11008×4096 | 6.8s | ~3ms | ~2300× |
| lm_head | 91×4096×32000 | 45s | ~8ms | ~5600× |

Two components contributing to the ratio:

1. **Custom uint64 modular GEMM vs Tensor-Core FP16.** Hardware Tensor
   Cores can't accelerate Goldilocks; we run a hand-written CUDA kernel.
   Realistic gap: 50-100× slower than dense FP16 even at the kernel
   level.

2. **Per-matmul rescale overhead** in compute_fns adds ~1-2s on top of
   each matmul (see 2.2).

**Fix:** custom kernel work. Possible improvements:
- Tile-based GEMM with shared-memory accumulation
- Better Goldilocks reduction (Montgomery form? Barrett?)
- Fuse the rescale decomp into the GEMM epilogue

**Expected speedup:** 3-5× on matmul time. Hard work (several hundred
LoC of CUDA), best done after the simpler wins above.

**Impact at 32L SEQ=91:** matmuls are ~half of engine pass time, so
~3-4× engine speedup overall. ~30-60 min savings on a hours-long prove.

### 2.4 Lazy weight re-encoding across rounds

**Where:** `lazy_loader.py` — each weight load re-runs `quantize_to_field`
from the .safetensors mmap. In real interactive mode, each of the 4
rounds re-loads every weight.

**Cost:** ~50ms per weight load × 9 weights/layer × 32 layers × 4 rounds
= ~60 sec per prove.

**Fix options:**
- Cache the quantized uint64 tensor (defeats the lazy memory goal).
- Use the test-mode fused single call so each weight loads once total.
- Accept the cost for real-protocol use; network round-trips dominate
  anyway.

**Recommendation:** keep as-is for real protocol; lean on the fused
single-call test mode for development.

### 2.5 Engine pass / encoding ratio

**Measured at 1L SEQ=91:** engine pass (compute_fns dispatch) = 203s,
prove rounds (encode + q-poly + columns) = ~98s. Engine pass is 67% of
prove.

After 2.1 + 2.2: engine pass drops to ~30-50s, encoding stays roughly
~98s. Encoding becomes the dominant component, and further optimisation
targets shift to NTT throughput and BLAKE3 column hashing.

---

## 3. Code simplification / cuts

### 3.1 `analysis/` directory contains exploration scripts

**Files:** `analysis/compare_accuracy.py`, `analysis/per_op_attribution.py`,
`analysis/shadow_tape.py`, and the other `analysis/*.py` diagnostics.

These were research artefacts. Not part of the production proof
pipeline. Decide which (if any) are still load-bearing and move the
rest to `deprecated/` or delete.

### 3.2 Test-only sample functions in core

`_sample_test_challenges` was added to support test-mode `prove(seed=)`.
It mirrors what `verify(seed=)` does internally. Two-fold duplication
of the challenge sampling order. If we ever refactor verify() to also
take explicit challenges (symmetric with prove), `_sample_test_challenges`
could be shared with verify rather than duplicated.

### 3.3 `_PhaseLogger` and `--verbose` plumbing

`core.py:1764` `_PhaseLogger` was useful when prove() was a 5-round
monolith. After the round-based refactor, the round boundaries are now
expressed by early returns, and the verbose log lines fire at less
useful points. Consider replacing with simpler per-stage `time.time()`
prints OR removing if `--time-ops` is sufficient.

### 3.4 Magic constants and width parameters

The 19 `output_width=24` sites in `demo_llama7b.py` are repetitive.
Could lift to one module-level `OUTPUT_WIDTH = 24` constant referenced
throughout. Same for `RMS_SLACK_N_CHUNKS = 4`. ~20 LoC cleanup, more
discoverable for newcomers.

### 3.5 `Tape.__init__` doc references `prove_lazy`

After the engine-pass refactor, `prove_lazy()` doesn't exist as a
separate method — `tape.prove()` dispatches on `self.lazy`. The
docstring still mentions it. Fixed in one earlier commit but worth
re-auditing across the codebase for similar stale references.

### 3.6 Per-claim `side_effects` closures

Every tape method constructs a `side_effects(values)` closure that
mostly does `lookup_multiplicities_into(...)` calls. Could be factored
into a single helper `_apply_lookup_multiplicities(claim, values)` that
inspects the claim type to know which mult vars to update. ~30 LoC
saving + uniform pattern.

### 3.7 `_check_identities` in verify() is dense

`core.py:_check_identities` is ~250 LoC of polynomial identity checks
with implicit semantic blocks. Could be split into 4-5 named functions
matching the protocol rounds it verifies (e.g., `_check_linear_sum`,
`_check_quadratic_identity`, `_check_range_logups`,
`_check_merkle_paths`). Helps newcomers locate which round a failure
came from.

---

## 4. Security gaps

### 4.1 Stubbed `master_seed` in `prove()`

**Status:** `core.py` `prove()` uses a constant `master_seed = b"\x42" * 32`
with a TODO comment.

**Implication:** ZK blinding rows (u_irs, u_lin, u_quad) and the row-PRG
slack added during encoding are deterministic from this constant. Anyone
observing the proof can derive witness values from the column openings.
**Safe for correctness tests; NOT safe for confidential proofs.**

**Fix:** replace with `secrets.token_bytes(32)` per prove call. For real
interactive protocol, caller must thread the same fresh master_seed
across all 4 rounds (otherwise blinding bytes don't match).

### 4.2 Verifier is not adversarial

The current `Verifier` class is a deterministic PCG64 sampler. In real
deployment the verifier is an external party that picks challenges
freely (or via Fiat-Shamir hash of prior commits). The seed-based
verifier is only valid for test correctness, not for soundness.

**Fix:** wire `core.prove` and `core.verify` to a real interactive
transport (websocket / IPC / Fiat-Shamir). The round-based prove() API
is already shaped for this — challenges come in as args.

### 4.3 No soundness audit of the 14 claim types

`tape.py` has 14 op methods (matmul, hadamard, rmsnorm, softmax, silu,
rope, embed, fingerprint, paired_tlookup, range_word, word_extract,
word_combine, add, hadamard_broadcast). Each emits some combination of
phase-1 vars, phase-2 vars, quadratic constraints, and range LogUps.

`CLAIM_SPECS.md` documents the intended math. Whether the actual claim
construction (in claims.py compile fns and aux fns) faithfully enforces
that math is a separate audit, partially done for matmul (commit 27)
but not exhaustively for all ops.

**Suggestion:** running an independent audit (the protocol from
CLAUDE.md "Independent Audits" section) for each claim type would
catch any silent soundness gaps.

### 4.4 Range table widths

We bumped `output_width=22 → 24` after observing REJECTs at the 22-bit
range. This was an empirical fix. The choice of 24 was based on
measured activations of Llama-2-7B; for Llama-400B the activation
magnitudes may exceed 2^11 real → would need width 25 or 26.

**Suggestion:** at the start of each new-model integration, run the
activation magnitude measurement (similar to what was done with the
fp16 reference probe) and set output_width to give 2× safety margin.

---

## 5. Numerical precision / quantisation

### 5.1 Q-format scale S = 2^12 may be tight at 400B

Llama-7B activations fit comfortably with 4 bits of headroom past the
output_width=24 ceiling. At 400B the activation outliers grow with
model size (well-documented Llama "outlier features"). Need to verify
whether s=2^12 remains adequate or needs s=2^11 with corresponding
output_width adjustment.

### 5.2 EmbeddingLookup is approximate

The embedding lookup currently materialises only `len(token_ids)` rows
of the E matrix as `E_subset`, with a `SOUNDNESS NOTE` warning. The
full vocab × d table would be ~131M field elements at d=4096. Need a
fingerprint-based lookup or batched range-table check to make
EmbeddingLookup committed against the full embedding matrix, not a
subset.

### 5.3 LM head matmul output range

The lm_head matmul output is a `(SEQ, vocab_size)` tensor at scale S
with values up to ~vocab × max_activation magnitudes. At
vocab=32000 the matmul output can hit ~2^28 real, which exceeds the
24-bit window. We currently get away with this because the demo runs
the verifier-side argmax outside the proof (so output range isn't
strictly bounded by the rescale). For a fully-in-proof argmax claim
we'd need a wider output_width or split decomposition.

---

## 6. Test infrastructure

### 6.1 No CI / automated regression tests

`test_claims.py` exists with ~30 per-op correctness tests, but there's
no automated runner. Every commit requires manual `python test_claims.py`
to validate. This makes refactors risky.

**Fix:** set up GitHub Actions or similar to run `test_claims.py` +
`test_pipeline_production.py` on every push. Adds ~30 min of compute
per CI run.

### 6.2 No end-to-end regression for full-prove flows

The 1L / 4L / 32L Llama smoke tests are run by hand on Spark. No
automated check that "32L Llama-2-7B SEQ=2 ACCEPTs" stays green across
commits. Easy to silently regress.

**Fix:** add a nightly job that runs the 32L smoke and posts result to
some channel.

### 6.3 Negative tests are sparse

`test_claims.py` has some negative cases (wrong C1, wrong C2) but
doesn't cover most "what if the witness is wrong here?" scenarios.
Need a systematic negative test for each constraint the verifier
checks.

---

## 7. Documentation gaps for onboarding

### 7.1 No architectural overview

A new contributor reading the codebase has to piece together how
`Tape.X` methods → claims → `_layout` → `prove()` → `verify()`
connect. The deprecated `engine.py` had a docstring trying to do
this; the current pipeline doesn't have an equivalent.

**Fix:** a short `ARCHITECTURE.md` covering:
- The data flow: tape build → engine pass → prove rounds → verify
- The 4 protocol rounds and what each computes
- How to add a new claim type (SAMPLE_FN, COMPILE_FN, AUX_FN, COMPUTE_FN)
- Variable lifecycle (phase-1 vs phase-2, mult vars, blinding rows)

### 7.2 No "how to scale up" guide

For someone trying to take this from 7B to 70B to 400B, the gotchas
encountered (output_width sizing, mem profile, lazy weights vs eager,
fused vs round-based, master_seed handling) are scattered across
commits. A guide that summarises these decisions and lists what to
re-verify per scale-up would help.

### 7.3 Claim spec gaps

`CLAIM_SPECS.md` documents some claim types but not all. Particularly
softmax and rmsnorm have non-obvious aux variable layouts. Filling
out the spec for all 14 claim types would help reviewers verify
soundness independently.

---

## 8. Roadmap to "Llama-400B inference proof on DGX Spark"

A rough sequencing of what work is required, prioritised by readiness:

### Near-term (weeks)
1. RmsNorm + `_signed_floor_decomp` GPU rewrites (§2.1, 2.2) — fast wins
2. 32L SEQ=91 end-to-end ACCEPT validation with current code
3. Logits comparison vs HF fp16 reference run (sanity check the proof
   actually attests to the right computation)
4. Set up CI (§6.1)

### Mid-term (months)
5. Phase 2c streaming engine pass (§1.1) — required for >32L
6. Custom `gl_matmul` improvements (§2.3)
7. `master_seed` from CSPRNG (§4.1) + real interactive protocol wiring
   (§4.2)
8. Independent soundness audit of all claim types (§4.3)
9. EmbeddingLookup full-vocab handling (§5.2)
10. Architecture documentation (§7.1, 7.2)

### Long-term (multiple quarters)
11. Phase 2d lazy AUX_FNS for >100K context (§1.2)
12. Multi-GPU / multi-node scaling (proof of Llama-400B is unlikely to
    fit in a single DGX Spark even with full streaming)
13. Soundness amplification (T_QUERIES tuning, security parameter
    review)
14. Performance: reach within ~10× of theoretical lower bound on
    encoding + GEMM

### Out of scope for now
- Universal Setup / Reusable structured reference string (we use
  Ligero, which is transparent)
- Recursive proofs / proof-of-proofs
- Hardware accelerator (custom ASIC) — assume DGX Spark / similar GPU
  for the foreseeable future

---

## Appendix: measured runtimes summary

| Test | Layers | SEQ | Inputs | width | Prove time | Notes |
|---|---|---|---|---|---|---|
| `--engine` random | 1 | 2 | random | 24 | 37s | smoke baseline |
| `--engine` random | 32 | 2 | random | 22 | 565s | 9.4 min |
| Llama "Hello world" | 4 | 91 | real | 22 | 408s (REJECT) | width too tight |
| Llama "Hello world" | 4 | 91 | real | 24 | 916s | 15.3 min |
| Llama "Hello world" | 4 | 91 | real | 24, fused | 901s | post-refactor |
| Llama "Hello world" | 1 | 91 | real | 24, fused | 301s | per-op timing run |

Extrapolated 32L SEQ=91 (untested): ~2 hours at current code, ~10-15
min after items 1, 2, 5, 6.
