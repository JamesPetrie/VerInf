# Prover optimization investigation — the q_lin (linear-constraint) fold

Status: **measured 2026-06-16** on `spark-c191` (DGX Spark, GB10). Companion to
`improvements-roadmap.md` (catalogue of gaps) and
`verifier-streaming-architecture.md` (the verifier-side scaling work). This doc
records where the prover's time actually goes, which optimizations were
*measured* to help vs. measured-dead, and the design options for the biggest
remaining lever.

## TL;DR

- **Banked win:** the **inverse-NTT fuse** cuts prove time **−7.5% (−25.7 s)**,
  bit-exact, env-gated (`LIGERO_FUSE_POLYMUL=1`). It's the only NTT-related
  lever that survives, because it does *fewer* NTTs (the NTT itself is
  memory-bound and already efficient).
- **Measured dead ends:** a faster NTT kernel (no headroom — memory-bound;
  sppark slower even raw) and bigger q_lin chunks (slightly *slower*).
- **Biggest remaining lever:** better handling of the **sparse linear
  constraints** (`expand + sort + spmv`, ~90 s ≈ 26% of the prove). The sort is
  *not* fundamental, and the structure is ~95% reusable across chunks.
- **Realistic engineering ceiling ≈ 1.4×.** A clean 2× needs a *protocol*
  change to the linear-constraint argument (sumcheck/GKR-style), not tuning.

## Method (and a measurement caveat)

Benchmark config: reduced Maverick
`demo_maverick_full.py --layers 4 --experts 16 --prompt-n 100 --cont-n 100`,
`LIGERO_T_QUERIES=4`, `N_LIG=2^16`. Baseline prove **≈ 340 s**, peak GPU
28.2 GB. m_total ≈ 1.23 M rows.

**Prefer wall-clock ablation deltas over the phase-bucket timers.** The
`LIGERO_PHASE_TIMING` buckets put a `cuda.synchronize()` at every phase
boundary, which removes kernel overlap and *inflates* the totals — the same
fold reads 168 s as one bucket but 196 s when split into sub-buckets (more sync
points → more inflation), against a real prove of ~340 s. So every lever below
was measured by making a component free/different and reading the **total prove
wall-clock delta** (no phase timer), which is overlap-honest.

All changes are env-gated and **default-off** (baseline is byte-for-byte
unchanged). Reproduction harness at the end.

## 1. Where the prove time goes

Phase shares (single-bucket phase timer; shares meaningful, absolutes inflated):

| phase | share |
|---|---|
| `fold_qlin` (linear-constraint fold) | ~54% |
| `encode` (RS encode NTTs) | ~15% |
| `compile` (Python per-chunk packet/expander regen) | ~11% |
| `quad` (quadratic fold / p_0) | ~10% |
| `merkle` | ~4% |
| `aux` + `witness` | ~5% |

`fold_qlin` dominates. Its sub-structure, via wall-clock ablation (below):
**poly_mul ≈ 74 s**, **expand + sort + spmv ≈ 90 s**, interp/matvec small.
Together ≈ the whole `fold_qlin`.

## 2. Ablation results (real wall-clock deltas vs. 340 s baseline)

| ablation | prove | Δ | reading |
|---|---|---|---|
| baseline | 339.8 s | — | |
| `LIGERO_NTT=sppark` | 389.6 s | **+49.8 s** | sppark is *slower* (see §3) |
| poly_mul → 0 | 265.8 s | **−74.0 s** | poly_mul's true cost (bucket said 75 — accurate) |
| expand+sort+spmv → 0 | 249.6 s | **−90.2 s** | the sparse-handling block (§5) |

The poly_mul bucket was *not* inflated (74 measured vs 75 bucketed) — it's a big
contiguous GPU op with little overlap. The expand block bucket was ~12%
inflated (90 vs 102).

## 3. NTT microbench — faster-kernel and bigger-batch are dead

Timed batched NTT with cuda Events (single end-sync), subtracting clone cost,
bypassing the per-call-sync confound:

- GB10 **practical copy bandwidth: 223 GB/s**.
- Builtin batched NTT is **memory-bound and efficient**: **0.33 ns/elem** at
  n=2^14, **0.42 ns/elem** at 2^15, and **flat across batch m = 256…4096**.
  (m=64 is *faster* per element — cache-resident — confirming memory-bound, not
  launch-bound.)
- **sppark is slower even raw**: 70.8 ms vs builtin 56.6 ms at 2^15/m=4096 — and
  that's *before* counting its shim's one-launch-per-row loop
  (`cuda_primitives.py` `sppark_ntt_batched`).

**Conclusions:** (a) a faster NTT kernel has ~no headroom here — the NTT is
bandwidth-bound and the builtin is already good; (b) bigger batches don't help
NTT throughput (flat ns/elem). The *only* way to cut memory-bound NTT cost is to
do **fewer** NTTs — which is exactly the fuse (§4).

Sanity check: 3 NTTs × 1.23 M rows × 2^15 × 0.42 ns ≈ 50 s of the measured 74 s
poly_mul; the other ~24 s is pad/alloc/pointwise/matvec overhead.

## 4. Optimization sweep + the inverse-NTT fuse (IMPLEMENTED)

| variant | prove | Δ |
|---|---|---|
| baseline | 340.5 s | — |
| **fuse** (`LIGERO_FUSE_POLYMUL=1`) | **314.8 s** | **−25.7 s (−7.5%)** |
| `LIGERO_QLIN_INNER=1024` | 347.9 s | +7.4 s (slower) |
| `LIGERO_QLIN_INNER=2048` | 349.3 s | +8.8 s (slower) |
| fuse + 2048 | 322.6 s | −17.9 s (worse than fuse alone) |

**Bigger q_lin chunks are dead** — the default `inner_chunk_size=256` is
near-optimal; larger chunks help neither the (flat) NTT nor the orchestration.

**The fuse** (`core.py`: `QLinAccumulator` + `_compute_q_lin_inner_chunk`,
`return_eval`): the default path computes, per row, `poly_mul(r_i, a_i)` = 3
NTTs (2 forward + 1 inverse) then sums the products. Since the inverse NTT is
*linear*, the fuse accumulates the pointwise products in the **eval domain**
across all rows/chunks and does **one** global inverse NTT at `finalize()` —
replacing ~m_total per-row inverse NTTs with a single one (1/3 of the fold's
NTT work). `Sum_i INTT(p_i) = INTT(Sum_i p_i)`, so it is **bit-exact**;
verified by the small `demo_maverick_moe.py` smoke (ACCEPT both ways, full
K_DEG=2^14). Memory-neutral (peak identical). At full-model scale the fold is a
larger share (fixed overheads amortize), so the relative win is at least as
large.

**Recommendation: enable/commit the fuse.** Free, safe, prover-side
(a bug yields REJECT, never a forged accept).

## 5. Sparse linear-constraint handling — the biggest remaining lever

The `expand + sort + spmv` block is ~90 s (~26% of the prove). Decomposed:

- **spmv ≈ 10 s** — applying the challenged `r^T A` (a segmented sum of
  `challenge(cid)·coef` per output slot). This is the *irreducible core*.
- **expand ≈ 58 s + sort ≈ 23 s ≈ 80 s** — the *overhead of genericizing*:
  expanders emit `(target, cid, coef)` triples in arbitrary order; the sort
  (`argsort` → `bincount` → `row_ptr`) turns them into CSR so the segmented-sum
  kernel can scan contiguous same-target runs.

So the ceiling for this lever is ~70–80 s (~21–24%): take the block from ~90 s
toward the ~10 s irreducible apply.

### Why a sort at all? (It isn't fundamental.)

The operation is a **group-by-target reduction** — `(r^T A)[i,c]` sums over all
constraints touching slot `(i,c)`. The sort merely groups same-target entries
to enable a contiguous segmented sum. It was chosen because it reuses existing
primitives (`argsort`/`bincount`/segmented `gl_spmv_challenged`) and gives
coalesced reduction — not because the math requires it.

### Options (A/B/C target the *same* ~80 s — pick one, not additive)

- **A. Cache the repeated structure (recommended).** Measured **95% reuse**:
  6316 q_lin chunks → only **322 distinct structures** (one structure = 64% of
  chunks; `LIGERO_REUSE_HASH=1` fingerprints `target_idx`). Cache
  `(target_idx, perm, row_ptr)` keyed by a packet-layout signature; on a hit
  skip the sort and target-generation, recompute only data-dependent
  coefs + spmv. Est. **~50–70 s**. Reuse is *higher* at full scale (48 identical
  layers). Key design point: the cache key must be computable **without** running
  the expander, else you only save the sort, not the expand.
- **B. Counting sort** instead of comparison sort (targets are bounded ints →
  O(n) bucket sort). Cheap, but **subsumed by A** (after caching, the sort runs
  322× not 6316×). ~10–15 s; only worth it without A.
- **C. Field scatter-add — remove the sort entirely.** One kernel:
  `out[target] += challenge(cid)·coef`, accumulated atomically. O(n), no sort,
  no triple reshuffle. Field add is associative+commutative so atomic order is
  bit-exact-irrelevant. Caveats: needs **field-safe atomics** (`atomicCAS`
  field-add loop or a carry/128-bit accumulator — plain `atomicAdd` wraps mod
  2^64, not mod P), and trades sort cost for atomic **contention ∝ per-slot
  fan-in**. Wins if fan-in is bounded. This is a cleaner form of "structured
  accumulate" — one generic kernel, not 13 per-packet kernels. Robust without a
  cache.
- **D. Protocol change (research / TCB).** A sumcheck/GKR-style linear-constraint
  argument avoids materializing `r^T A` densely *and* the downstream poly_mul —
  the only route to a clean 2× — but it reworks the proof system and the
  verifier (TCB risk). Not tuning.

### Open unknowns to measure before building

1. **expand target-gen vs coef-gather split** — how much of the ~58 s expand is
   structural (cacheable) vs data-dependent. Sizes A and C precisely. The
   cleanest probe is the cache prototype itself (correctness-gate decides whether
   the coefs `v` are structural and thus wholesale-cacheable).
2. **fan-in distribution** (triples per target) — decides whether scatter-add (C)
   beats the sort. One-shot `bincount` instrumentation, no kernel work.

### Design follow-on

Two design notes develop the prover-side levers past the options above:

- `qlin-family-object-reorg.md` — realizes option **A** (cache the repeated
  structure) *structurally*: compile to one constraint-family object per
  `(variable, family)` instead of per-row packets, build `r_i` variable-stationary
  with a per-family reduce so the **sort drops out**, and remove `_late_qlin_var`.
  Prover-only; recovers the `compile` + `expand` overhead measured above. Also
  covers the sound-R3 control flow (q_irs/q_lin/p_0) and the don't-hold-encodes /
  no-op-local-p_0 decisions.
- `qlin-evenodd-fused-multiply.md` — a **forward-transform** lever this
  investigation did not consider: the ζ-domain is the even half of the `2K`-th
  roots, so half of `r_i·f_i` is a free elementwise product and the rest is a
  size-`K` coset NTT. **Unmeasured analysis** — the ceiling below predates it;
  whether it pushes past ~1.4× is for the prototype to confirm.

## 6. Combined ceiling and relation to verifier scaling

- Fuse (banked, 7.5%) + sparse-handling (A or C, ~15–24%) → **~1.3–1.45×**.
  A clean 2× requires the protocol change (D).
- The prover fold is **O(m_total)** (chunking caps the sort's log factor), so
  this breakdown is ~scale-invariant along the more-rows axis — it transfers to
  the full model.
- For *very large* models the binding constraints are verifier-side, not prover
  time: **verifier memory ≈ 1 KB/row** (measured 99.6 GB at 99.4 M rows) and
  **proof size ∝ T_QUERIES·m_total**. Both are solvable and *not* hard walls —
  verifier *time* is already sub-prover (3.3 h vs 8 h on mav847) and a CUDA port
  has ample headroom (it also obviates the O(ELL²) eval blowup, removing the
  highest-TCB verifier change); memory is fixed by **streaming the verifier**
  (regenerate per-row constraints, or disk-back the expanders row-major — the
  check-4 pass is sequential, and the accumulation is additive so out-of-order
  fragments sum correctly). See `verifier-streaming-architecture.md`. A CUDA
  verifier enters the TCB, so gate it bit-exact against the simple verifier.

## Reproduction (env knobs + harness)

All in `pipeline/core.py` (on branch `diag/qlin-prover-opt`; not on `main`),
default-off:

| flag | effect |
|---|---|
| `LIGERO_FUSE_POLYMUL=1` | inverse-NTT fuse (the win) |
| `LIGERO_QLIN_INNER=N` | q_lin inner chunk size (default 256; bigger is worse) |
| `LIGERO_REUSE_HASH=1` | print distinct-structure count over q_lin chunks |
| `LIGERO_ABLATE_POLYMUL=1` | poly_mul → 0 (timing ablation; proof REJECTs) |
| `LIGERO_ABLATE_EXPAND=1` | expand+sort+spmv → 0 (timing ablation; proof REJECTs) |
| `LIGERO_PHASE_TIMING=1` | print phase buckets (inflated — read shares only) |

Spark scripts: `~/qlin_opt_sweep.sh` (the sweep), `~/ntt_microbench.py`
(roofline), `~/qlin_ablate.sh` (the ablations). Microbench needs no prover
change; ablations/sweep run the reduced-Maverick config above.
