# Instrumented prover phase split — measured (2026-07-05)

Status: **measured** on `spark-c191` (GB10), branch tip `839bb39`
(post-Phase-4 fold kernels, fuse default-on). Purpose: fill the cost-model
terms of the paper's §8/Appendix A.5 — the column-hashing share (D), the
in-kernel constraint-coefficient share (E), and the full bucket split — with
measured values instead of round-level inference. Companion to
`analysis/maverick-cost-model.md` (machine constants) and
`prover/deprecated/spark-microbench-results.md` (the BLAKE3 primitive:
128 GB/s absorbed = 2.0 Gcompress/s).

## Method

`LIGERO_PHASE_TIMING=1` (cuda-synced wall-clock buckets, accumulated across
the whole four-round prove) plus `LIGERO_EPHASE=1` (CUDA-event GPU timings
for the qlin fold internals). Two caveats from the instrumentation itself:
the syncs serialize the pipeline, so **read shares, not absolutes**; and the
`qlin_*` sub-scopes are **nested inside `fold_qlin`** (their sums match it
exactly in both runs: 25.7 = 16.1+3.6+3.1+2.9 and 43.1 = 27.8+6.2+5.2+3.8),
so the report's BUCKETED total double-counts the fold — shares below are on
the de-duplicated base. JIT warmed by a discarded run per model.
`LIGERO_T_QUERIES=4`.

Two configs, covering both claim mixes:

- **A — dense mix:** Llama-2-7B, 2 layers, seq 1000, real weights
  (`demo_llama7b.py --num-layers 2 --seq 1000 --lazy-weights --engine`).
  Prove 197.1 s, peak 11.6 GB.
- **B — MoE mix:** reduced Maverick, 2 layers (1 dense + 1 MoE), E=16,
  100+100 tokens (`demo_maverick_full.py --layers 2 --experts 16
  --prompt-n 100 --cont-n 100`). Prove 252.7 s, peak 28.4 GB.

Both runs: `root_p1 reproducible across rounds: True`.

## Results (seconds; shares of the de-duplicated bucket total)

| bucket | A: 7B dense | share | B: Maverick MoE | share |
|---|---|---|---|---|
| witness (all 4 rounds' recomputes) | 53.0 s | 29% | 19.8 s | 10% |
| encode (commit + opening re-encode + aux-round) | 51.3 s | 28% | 94.6 s | 46% |
| quadratic fold | 35.8 s | 20% | 20.4 s | 10% |
| linear fold (`fold_qlin`) | 25.7 s | 14% | 43.1 s | 21% |
| — transforms (`qlin_polymul`) | 16.1 s | 63% of fold | 27.8 s | 65% of fold |
| — coefficient work (`qlin_interp`+`qlin_rTA`) | 6.7 s | **3.7% of prove** | 10.0 s | **4.8% of prove** |
| — accumulate (`qlin_rowsum`) | 2.9 s | | 5.2 s | |
| column hashing (`merkle`) | 9.3 s | **5.1%** | 16.0 s | **7.7%** |
| aux | 5.0 s | 3% | 13.0 s | 6% |
| fold_qirs + compile | 0.6 s | | 0.7 s | |
| cols (column extract) | 0.0 s | | 0.0 s | negligible at T=4; its encode cost lands in `encode` |
| de-duplicated total | 180.7 s | | 207.6 s | |

GPU-event confirmation (ephase, single sync): A — polymul 16.07 s, interp
3.54 s, rTA 3.06 s, rowsum 2.90 s (1874 calls each); B — polymul 27.74 s,
interp 6.13 s, rowsum 5.18 s, rTA 3.80 s (3478 calls each).

## Readings

1. **The E term (in-kernel coefficient work) is 4 to 5% of prove** in both
   mixes — the pre-kernel materialize-sort-reduce stage measured ~26%
   (`prover-optimization-investigation.md` §5), so the descriptor kernels
   closed that lever nearly completely; within the fold it is ~25%, with the
   transforms about two-thirds.
2. **The D term (column hashing) is 5 to 8% of prove** — consistent with
   the 2.0 Gcompress/s primitive; a real but secondary single-machine cost.
   Its significance is the scaling law, not the share: it rides scalar-ALU
   compute, and per-GPU scalar throughput on a B200 is only ~2.4× the GB10's
   (both Blackwell; ~31 vs ~75 FP32 TFLOPS), so an NVL72 scales hashing by
   ~170× against ~2,580× for the bandwidth-bound transforms — at 1M tokens
   roughly 1.3 days of hashing vs 3.3 days of transforms (a quarter to a
   third of the cluster floor, growing as the transform side is optimized).
3. **The witness bucket confirms the ×4 accounting** (all rounds recompute
   it): 29% when the model is small relative to context (dense mix), 10% in
   the MoE mix.
4. **Small-scale absolutes are inflated** (small NTT batches + sync
   serialization): per-slot rates here run ~2× the primitive floors. Use the
   full-model run archive for absolute calibration and these runs for the
   split.
5. **Instrumentation note for future runs:** the phase report's BUCKETED sum
   double-counts nested scopes; either de-duplicate as above or un-nest the
   `qlin_*` scopes.
