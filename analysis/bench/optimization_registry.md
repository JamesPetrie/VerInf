# Optimization registry — status of every lever tried or known

A durable status board so unproven-but-useful ideas aren't lost. Each entry:
what it is, whether it's SOUND (byte-identical proof + Rust ACCEPT), whether the
speedup is DEMONSTRATED and at what scale, and whether it TRANSFERS to the real
400B production run (the thing that actually matters). Categories by status.

Rule of thumb from this project: **a microbenchmark win is not an in-situ win,
and an in-situ win at test scale is not a production win.** Only the last column
("transfers to 400B") is the real goal.

---

## ✅ Demonstrated AND transferable (real production wins)

| lever | what | sound? | speedup | transfers to 400B? |
|---|---|---|---|---|
| **GPU softmax port** | softmax witness (the O(SEQ²) causal binary search) computed on GPU field-ops instead of CPU numpy | yes (byte-identical + ACCEPT) | seq1024 100→60s in-prove; ~95× on the op in isolation | **YES** — it changes WHERE compute runs, not whether the witness fits; speeds each of the 4 recomputes at any scale. Softmax is O(SEQ²) so this grows with context. |

Flag: `LIGERO_GPU_SOFTMAX` (default on). File: `prover/compute_fns.py`.

---

## ⚠️ Correct but in-situ benefit UNCONFIRMED (sound, opt-in, not a claimed win)

| lever | what | sound? | measured | transfers to 400B? |
|---|---|---|---|---|
| **coset-NTT encode** | encode via ρ length-K NTTs instead of one length-N NTT | yes (byte-identical + ACCEPT) | isolated NTT 1.2× at K≥2^16; **in-prove at K=2^16 (small model): NO encode win, +1.4GB mem** | UNKNOWN — cost model predicts a win at K≥2^18 with a 7TB witness (encode is 32% of the 400B prove and IS NTT-bound there), but unmeasurable here. Microbench overstated the in-prove benefit at runnable scale. |

Flag: `LIGERO_COSET_NTT` (default **off**, opt-in). File: `prover/core.py`
(`_coset_encode_via_coset_ntt`). Kept as a validated-correct lever to A/B at true
production scale; NOT auto-enabled (costs memory for unconfirmed benefit).

---

## 🔬 Real at test/medium scale, does NOT transfer to production

| lever | what | sound? | speedup | transfers to 400B? |
|---|---|---|---|---|
| **witness cache** | reuse softmax/silu compute across the 4 Fiat-Shamir sweeps instead of recomputing | yes (byte-identical + ACCEPT) | seq1024 218→97s at test scale | **NO** — works only because the witness FITS in memory; at 400B the witness is 7TB (the very reason the 4× recompute exists). Degrades to recompute when memory is tight = the production case. |

Flag: `LIGERO_WITNESS_CACHE` (default on — harmless, no-ops at scale). File:
`prover/core.py`. Honest note: this was the loop's biggest headline number but is
a test-scale artifact; do not quote it as a production win.

---

## ⛔ Studied and rejected (soundness- or measurement-blocked)

| lever | what | verdict |
|---|---|---|
| **claim-streaming (witness_passes 4→1)** | merge the 4 witness passes by "finishing each claim's rounds while its rows are live" | **SOUNDNESS-BLOCKED (iter9).** Paper §5.3/§5.4: the 4 rounds fix each commitment BEFORE the next challenge — load-bearing for the 2⁻¹⁶·⁶ bound. Merging needs round-3/4 challenges during round-1 → lets the prover fit a witness to an unseen challenge. Current code only merges via `round_seeds()` pre-derivation, which its own docstring flags as a TEST shortcut (a non-transferable trap like the witness cache). Paper lists it as §9 future work. NOT a drop-in; needs protocol redesign. Did not implement. |
| **witness spill (store-once, re-read)** | the SOUND remnant of claim-streaming: don't merge rounds, just re-read the witness in rounds 2-4 instead of recomputing | **BUILT + SOUND; production benefit MODELED, not measured (iter9+correction).** `LIGERO_WITNESS_SPILL=1`, host-memory backing, byte-identical + Rust ACCEPT. TOY A/B is net-negative (−0.7%/−3.8%) — but that is a SCALE ARTIFACT (toy recompute is trivial, so PCIe overhead dominates). At 400B the crossover INVERTS: effective recompute throughput = 7.5TB/1.08h = **1.93 GB/s**, below any NVMe, so re-read wins. Modeled: **+13% (NVMe) to +34% (RAID/PCIe-cap)** faster; loses only on HDD. See below and spill_costmodel_prod.py. Prod version needs DISK backing (7.5TB ≫ 84GB host) + FULL-witness coverage (prototype = softmax/silu only). |

### 🔬 BabyBear / adaptive field — decision layer built, execution port-blocked (iter10)
Field is hardcoded to Goldilocks across 11 files (Python roots pow(7,…), CUDA
`GL_P`, Rust `field.rs`); a real BabyBear prove+ACCEPT is a multi-file
Python+CUDA+Rust port (new modulus/generator/2^27-adic roots/31-bit packing) —
NOT done. Built the sound DECISION layer instead: `bench/field_policy.py`
(worst-case per-op accumulator bounds) + `bench/run_field_variants.py` (policies
goldilocks|babybear|adaptive-proof|adaptive-op, `--sweep-s` feasibility map,
first-order payoff via prod_lens). Verdict: at the current scale s=2^12 nothing
but elementwise fits BabyBear (30-bit ceiling); weight matmuls need s≤2^8, all
ops need s≤2^4. So the limiter is PRECISION, not field mechanics. Mixed
(adaptive-op) at s=2^6 gives only ~+7% (est.) because the O(seq²) attention-score
witness dominates and can't leave Goldilocks. Real next step is an ACCURACY
experiment (does inference stay correct at reduced s?), then the backend port.

### 🛡️ Guard against this class of error: `bench/prod_lens.py`
Every A/B must end by calling `prod_lens.report(...)`, which prints the toy
number next to the authoritative 400B term shares + the production projection, so
a bare toy % can't be mistaken for a production result. `transfers=False` marks
test-only levers (→ projects 0%). Wired into the spill / gpu_softmax /
witness_cache A/Bs; wire every new one the same way.

### ⚠️ CORRECTION: witness IS ~57% at 400B (I briefly mis-stated it as 6%)
An earlier iter9 note claimed witness was only ~6% of prove and the real levers
were encode+quad. That was drawn from TOY phase timing (d512/L4/seq512) and is
WRONG for production: toy models have negligible model-compute while Ligero
params (K_DEG=1024,N_LIG=4096) are FIXED, so fixed-cost encode dominates and
witness looks tiny. The cost_calculator at production scale (`--S 1093
--witness-mode notebook`, floor scales with W) gives: floor 12380s vs
witness_recompute 16288s -> **witness = 56.8% of the 28668s prove.** The 57% was
RIGHT. Lesson (again): toy phase-shares do NOT transfer; extrapolate the term
that scales with model size via the cost model, don't read it off a toy.

## 📋 Known, available, NOT done (candidate levers)

| lever | what | est. value | risk |
|---|---|---|---|
| **disk-backed full-witness spill** | extend the validated spill prototype to (a) disk backing for 7.5TB, (b) spill the FULL witness not just softmax/silu | HIGH — models to +13-34% at 400B; the sound way to cut the 57% witness term. Benefit confirmable only on prod hardware | MED — the mechanism is proven byte-identical + ACCEPT; the work is disk I/O plumbing + widening coverage, not soundness |
| **encode (NTT) optimization** | encode is the largest part of the identity floor; coset-NTT lever exists (opt-in) but unproven in-situ | MED — floor is ~43% of prove at 400B, encode a chunk of it | MED — NTT/field-op territory, byte-identical gate applies |
| **silu + rmsnorm GPU port** | same trick as GPU softmax, applied to the other CPU-numpy compute_fns; speeds each of the 4 witness recomputes (the 57% term) at any scale | MED — dents the 57% term directly, like GPU softmax did | LOW — mechanical, same byte-identical + ACCEPT gate. silu already ported (GPU_SILU). |

---

## Why TensorRT is not on this list

TensorRT accelerates approximate float/int8 tensor-core NN inference. The prover
computes an EXACT integer witness in the Goldilocks field (mod p = 2⁶⁴−2³²+1);
bit-exactness is what the proof rests on. TensorRT has no primitive for 64-bit
modular field arithmetic, and float approximation would break the proof. The
field-arithmetic equivalent of "TensorRT" here is the project's own CUDA kernels
(gl_mul, NTT, BLAKE3), already on GPU. Porting CPU compute_fns to those kernels
(as the GPU softmax port did) is the correct analog, not TensorRT.
