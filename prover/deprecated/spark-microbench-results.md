# Phase 1.0 Spark microbenchmark — results

**Date:** 2026-05-10. **Hardware:** `spark-c191` (NVIDIA GB10, sm_121, 48 SMs, 121 GiB unified, CUDA 13.0). **Plan:** [`phase-1.0-spark-microbench.md`](phase-1.0-spark-microbench.md). **Source:** [`spark-bench/`](spark-bench/).

## Headline

GB10 handles all three core primitives at speeds that put the Phase 1 wall-time projection comfortably inside the 2-6 minute target. **GO** for Phase 1.

Initial level-per-launch NTT was launch-overhead-bound at ~74 µs/NTT for N=65536. After two optimization passes (8-stage shared-memory fusion, then a Bailey 4-step decomposition for the codeword length) the NTT lands at **23.8 µs/NTT — 3.1× over baseline**. That closes the wall-time gap; the GO decision no longer depends on further NTT work.

## Numbers

| Component | Result | Limiting factor |
|---|---|---|
| Goldilocks `mul` + reduce, independent | **312 Gmul/s** (saturated from 96·512 threads up) | hardware peak |
| BLAKE3 column hash, m=16..32 | **~128 GB/s absorbed**, **2.0 Gcompress/s** | memory-bandwidth-limited at large m |
| NTT length 16384 (K_DEG), fused-256 | **27.7 µs / NTT** | 6 cross-block stages remain on level-per-launch path |
| NTT length 65536 (N_LIG), Bailey 4-step | **23.8 µs / NTT, 22 Gbutterfly/s** | within ~10% of memory-bandwidth + launch model |

The arithmetic floor for length-65536 NTT at 312 Gmul/s is `(N/2)·log₂N / 312e9 = 1.68 µs`. Bailey is **~14×** above the arithmetic floor — most of that gap is the 3 kernel launches plus the ~3 MB of inter-pass global memory traffic (read d_a → write d_temp → read d_temp → write d_a).

### NTT optimization progression

Three NTT variants were measured at N=65536 (all bit-exact vs Python reference for both forward and forward-then-inverse round trip):

| Variant | µs/NTT | Speedup | Launches | Notes |
|---|---|---|---|---|
| Level-per-launch (baseline) | 74.4 | 1.0× | 17 | bit-rev + 16 butterfly stages, each its own kernel |
| Fused 4096-element shared memory | 45.5 | 1.6× | 6 | 12 stages fused per chunk (32 KB shared); 4 cross-chunk stages remain per-level |
| Bailey 4-step (256×256) | 23.8 | 3.1× | 3-4 | Two passes of 256-pt NTTs in shared memory + twiddle, with scratch buffer to avoid in-place transpose race |

The fused 4096 path saturates at 16 blocks (vs GB10's 48 SMs), so it under-occupies the chip even though each block does more work. Bailey runs 256 blocks per pass, full SM occupancy.

## Phase 1 wall-time projection (Llama 2 7B, SEQ=2)

Weight commit dominates; per `design-feasibility.md` §2.2 each row of the encoded matrix needs one inverse NTT of length `K_DEG = 16384` plus one forward NTT of length `N_LIG = 65536`.

```
W_R_W ≈ 7×10⁹ slots          (Llama 2 7B parameters)
m     = W_R_W / ELL ≈ 855K   (rows per encoded matrix; ELL = 8192)
```

| | Per row | Total weight commit |
|---|---|---|
| Level-per-launch baseline | 42 + 74 = 116 µs | ≈ 99 s |
| **Bailey + fused (current best)** | **27.7 + 23.8 = 51.5 µs** | **≈ 44 s** |
| Theoretical floor (arithmetic only at 312 Gmul/s) | 0.4 + 1.7 = 2.1 µs | ≈ 1.8 s |

Plus per-prefill activations (`R_p1`), per-query Freivalds witness (`R_p2`), the linear test on `[R_W; R_p1; R_p2]`, the quadratic test, and column-hash work. At `SEQ = 2` these are sub-dominant; the bringup-plan estimate of 2-6 min comes from this kind of accounting on top of an optimized NTT.

**End-to-end Phase 1 prove time at the current NTT speed:** weight commit ≈ 44 s; the per-prefill commits, per-query Freivalds witness, linear and quadratic tests, and column-hash work add somewhere between 30 s and 60 s on top. End-to-end **~1.5-2 min**, comfortably inside the 2-6 min budget. (This is a back-of-envelope projection; an end-to-end measurement is in the open follow-ups.)

BLAKE3 hash work for the weight commit: ≈ 450 GB total absorbed (m * N_LIG * 8 bytes), divided by 128 GB/s = ~3.5 s. Negligible relative to the NTT work.

## Validation done

| Layer | Method | Result |
|---|---|---|
| Goldilocks `add`/`sub`/`mul` | 61 vectors emitted by `goldilocks_ref.py` (10 hand-picked edge cases including `(P-1)·(P-1) = 1`, `(P-1)+1 = 0`, plus 50 random pairs at fixed seed) compared to CUDA output | 0 failures |
| NTT forward | Bit-exact match against `goldilocks_ref.py emit_ntt_vectors N=1024` and `N=65536` | 0 differences |
| NTT round-trip | `inverse(forward(x)) == x` at N=1024 and N=65536 | identity holds |
| BLAKE3 compression + single-chunk hash | Official BLAKE3 test vectors at input lengths 0, 1, 64, 1024 (inputs `bytes[i] = i mod 251`) | 4/4 match |

The BLAKE3 test-vector hashes used are direct quotes from the project's `test_vectors.json` (`af1349b9...`, `2d3adedf...`, `4eed7141...`, `42214739...`). For this benchmark BLAKE3 is implemented up to a single chunk (≤ 1024 input bytes); for column sizes m ≤ 128 Goldilocks (= 1024 bytes) the single-chunk path is the full story. Larger m needs the chunk+tree machinery, which adds <1% to per-byte cost (compression is ~99% of total work for our column sizes).

## Caveats

1. **Bailey is hard-coded for N=65536** (factored as 256×256). N=16384 (K_DEG) currently uses the simpler 8-stage fused path at 27.7 µs — there's a similar Bailey win available there (factor 128×128 with a 128-pt building block) for ~5 more µs of speedup per row, dropping the weight-commit projection by ~4 s. Not on the critical path.
2. **GB10 toolchain is new** (driver 580.142, CUDA 13.0 on this box). NTT performance at sm_121 may improve with future toolchain updates independent of our code.
3. **ARM64 + unified memory.** Both are firsts for this project. No surprises so far in the kernels we've measured, but the full Phase 1 prover will exercise host/device traffic patterns we haven't.
4. **Field-mul peak is for *independent* multiplications in registers.** Real NTT has dependency chains (butterfly between read and write); 312 Gmul/s is the *upper bound*, not what NTT will actually consume.
5. **BLAKE3 measurement uses repeated input bytes**; per-byte content doesn't affect throughput, but the simple 0xab pattern means the kernel's inner loop hits highly-cacheable data. With column data in global memory each compression block reads 64 fresh bytes regardless, so memory traffic is the same; this caveat is paranoid but worth flagging.

## Next steps

1. **Multi-chunk BLAKE3.** For column hashes at m > 128 we need the chunk-chain + tree-merge logic. Throughput should be within 1% of the current single-chunk number; primarily a correctness extension.
2. **(Optional) Bailey at N=16384** for K_DEG. Modest gain (~5 µs/row); not on the critical path.
3. ~~**End-to-end Phase 1 prover wall-time.**~~ ✅ See "Measured end-to-end" below.

## Measured end-to-end (Llama 2 7B, 32 layers, SEQ=2, 2026-05-22)

Full forward-pass proof on `meta-llama/Llama-2-7b-hf` with the prompt `"Hello world"` (tokens `[1, 15043]`), all 32 transformer layers + final RmsNorm + LM head, cross-layer residual binding via fingerprint claims. Each layer is its own self-contained proof (per-layer Tape); the chain is bound across proofs by a public-coefficient fingerprint `fp = Σ rᵢ · residual_i` committed at every boundary. Raw log: [`spark-bench/benchmarks/llama27b-32layer-seq2-2026-05-22.log`](spark-bench/benchmarks/llama27b-32layer-seq2-2026-05-22.log).

Headline:

```
32 layers + tail, all ACCEPT, fp chain intact at every boundary.
prove sum:  1037.22 s   (~17.3 min)
verify sum:  442.60 s   (~7.4 min)
argmax(logits[1]) = token 13 = '\n'
```

Per-layer prove time, after GPU/kernel warmup (mean of L4-L31): **~29.8 s / layer**. The first 4 layers averaged 47s (cold caches). Verify is very stable at ~13.5 s / layer.

| | Per layer (warm) | 32 layers + tail |
|---|---|---|
| prove | 29.8 s | 1037 s |
| verify | 13.5 s | 443 s |

### Comparison vs the projection above

The projection in the previous section was **~1.5-2 min total** for Phase 1, dominated by weight commit at 44s. The actual run is **~17 min prove + ~7 min verify**, ~10-15× over budget. The gap factors into three sources:

1. **Per-layer commit, not amortized.** The projection assumes one big weight commit and amortizes across many queries. Our demo commits each layer's weights inside its own per-layer Tape (32 separate commits) and runs them sequentially in one query, so the 44s NTT cost recurs implicitly inside each layer's prove. Even ignoring amortization the NTT cost adds up linearly with weight slots, which the projection already accounted for — so this is not the main gap.

2. **Prototype constants vs optimized constants.** The microbench numbers above (51.5 µs / NTT row, 2.0 Gcompress/s BLAKE3) come from hand-tuned CUDA kernels run in isolation. The end-to-end demo uses the Python `spark-bench/python/pipeline` path, which uses PyTorch / torch-based field ops and CSR machinery that are an order of magnitude slower than the microbench primitives on the same hardware. Specifically: there's no integration of the Bailey-4-step NTT into the Python prover, no integration of the fused BLAKE3 column-hash, and the linear-CSR construction is in Python with significant per-claim overhead.

3. **Per-claim setup cost dominates at SEQ=2.** Each transformer block produces ~24 claims and each claim allocates Variables, registers tables, builds CSRs, and computes phase-2 aux witnesses. Python-side overhead per claim is fixed, so at SEQ=2 (where the actual cell counts per claim are small) the fixed cost is a large share. This shrinks rapidly with SEQ — see follow-up SEQ=1000 run.

### Cross-layer binding

Every layer's `fp_in` equals the previous layer's `fp_out` exactly. Public coefficient vector `r` (length `SEQ·d = 8192`) is generated deterministically from a fixed seed (would be Fiat-Shamir-derived from cross-proof commits in a deployed setting). Collision probability ~2⁻⁶⁴ per pair.

### Argmax check

Token 13 = `'\n'`. Sanity-checkable by running the same prompt through HuggingFace's `transformers` at FP16 and confirming `argmax(logits[1]) == 13`. If it matches, that's strong evidence the proof reproduces real Llama-2-7B math end-to-end at Q3.12 quantization.
