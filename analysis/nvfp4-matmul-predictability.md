# NVFP4 matmul predictability — measured on Blackwell (GB10)

**Date:** 2026-06-23. **Hardware:** `spark-c191` (NVIDIA GB10, sm_121, Blackwell, CUDA 13.0),
torch `2.13.0.dev` with native `float4_e2m1fn_x2` + `torch._scaled_mm_v2`. **Harness:**
[`nvfp4_matmul_test.py`](nvfp4_matmul_test.py) (the validated sweep) and
[`nvfp4_scale_layout_bruteforce.py`](nvfp4_scale_layout_bruteforce.py) (how the scale convention was found).

## Why

For verifying FP4 inference we need to know whether a hardware NVFP4 matmul is *predictable* in
software — i.e. whether `decode → ideal dot product → requantize` reproduces what the Blackwell
tensor cores actually emit. If it does, a verifier can model FP4 matmuls cleanly (and exactly, in a
field) without replicating the GPU's reduction tree. This tests that claim against real hardware.

## TL;DR

- **The model holds, and the hardware accumulation is *bit-exact* up to `K = 16384`.** For NVFP4
  inputs (E2M1 elements + E4M3 per-16 block scales), the hardware matmul accumulator equals the
  fp64 dot product of the exactly-decoded inputs with `max_abs_err = 0.0` — every element.
- **At `K = 65536` the fp32 accumulator starts to deviate** by ~1–2 ULP (`~6e-5`) on ~0.02% of
  elements — the expected onset of finite-precision accumulation.
- **The requantized NVFP4 *output* matched the model bit-for-bit at every `K` tested** (output E2M1
  codes and E4M3 scales identical), because the coarse output grid absorbs the tiny accumulator
  noise. **But see "Statistical limits" — this is a bound, not a proof of zero disagreement.**
- **The hard-won detail: the block-scale layout.** `torch._scaled_mm_v2` for NVFP4 needs the per-16
  E4M3 scales in Blackwell's swizzled 128×4 / 32×4×4 layout via `to_blocked(...)` + `SwizzleType.SWIZZLE_32_4_4`.
  Passing the natural `[M, K//16]` layout reads as garbage (~1/16 random agreement); `NO_SWIZZLE`
  errors outright.

## NVFP4 format (as used here)

`value = E2M1_element × E4M3_block_scale × FP32_per_tensor_scale`:

- **E2M1 element** (4 bits): magnitudes `{0, .5, 1, 1.5, 2, 3, 4, 6}`, sign bit. Confirmed packing
  (low nibble = even element): byte `0x21` → `(0.5, 1.0)`. LUT = `[0,.5,1,1.5,2,3,4,6]` ± sign.
- **E4M3 block scale**, shared over **16** elements (this is the NVFP4 vs MXFP4 distinction; MXFP4
  is 32-wide with a power-of-two E8M0 scale).
- **FP32 per-tensor scale**, one scalar per operand (two-level NVFP4).

This torch build **cannot cast `float4_e2m1fn_x2 → float`** (device-side assert), so decoding is done
by unpacking the uint8 nibbles through the E2M1 LUT.

## Validated `_scaled_mm_v2` convention

```
torch._scaled_mm_v2(
    A_fp4, B_fp4,                                            # [M,K//2], [K//2,N] (fp4 packed)
    [to_blocked(A_scale)], [ScalingType.BlockWise1x16], [SwizzleType.SWIZZLE_32_4_4],
    [to_blocked(B_scale)], [ScalingType.BlockWise1x16], [SwizzleType.SWIZZLE_32_4_4],
    None, torch.float32)                                    # bias, out_dtype
```

`A_scale` is the natural `[M, K//16]` E4M3 tensor; `B_scale` is `[N, K//16]` (the free-dim layout,
i.e. `B`-transpose's per-row scale). `to_blocked` (from `torch.testing._internal.common_quantized`)
applies the 128×4 / 32×4×4 swizzle. Brute-forcing all layout × swizzle combinations (the second
script) showed `flat`/`natshape`/`blk2d` of the `to_blocked` buffer all give bit-exact results with
`SWIZZLE_32_4_4`; everything else is wrong or errors.

## Results

`M = N = 256`, random Gaussian inputs quantized to NVFP4 (`scale = amax/6`):

| K | accumulator bit-exact vs ideal | max abs err | output code match | scale match |
|---|---|---|---|---|
| 256 | 100.000% | 0.0 | 100.0000% | 100% |
| 1024 | 100.000% | 0.0 | 100.0000% | 100% |
| 4096 | 100.000% | 0.0 | 100.0000% | 100% |
| 16384 | 100.000% | 0.0 | 100.0000% | 100% |
| 65536 | 99.98% | ~6e-5 | 100.0000% | 100% |

(3 seeds each; values are representative.) The diagnostic with all block scales forced to 1.0 — where
the swizzle is a no-op — was bit-exact, isolating the scale layout as the only convention subtlety.

## Statistical limits (what 1M outputs can and cannot say)

The ~1M outputs across the sweep do **not** establish verifier-grade bit-exactness, for two reasons:

1. **Most samples are non-informative.** A model/hardware disagreement requires the accumulator to
   differ (`hw ≠ ideal`). For `K ≤ 16384` the accumulator was *provably* bit-exact, so
   `requantize(hw) ≡ requantize(ideal)` trivially — those ~786k outputs test nothing about
   boundary-crossing. Only `K = 65536` had any deviation, on ~0.02% of elements (~40 "at-risk"
   elements total), 0 of which flipped an output code.
2. **Power.** Taking the full set at face value, 0 mismatches bounds the rate to `< ~3e-6` (rule of
   three); restricted to the only regime where a mismatch is *possible* (`K = 65536`, ~2×10⁵ outputs)
   the bound is `< ~1.5e-5`. A rough estimate of the true output-mismatch rate at `K = 65536` is
   `~1e-7–1e-8` (accumulator error × E2M1 boundary density) — below what this experiment can resolve.

So the strong, *non-statistical* result is: **for `K ≤ 16384` the accumulator is exact, hence the
requantized output is exactly predictable by construction** (covers most FFN/projection layers,
`d = 5120`, `d_ff ≤ 16384`). For large `K` (e.g. long-context attention, `K = S`) the output match is
only bounded statistically; targeted/adversarial sampling near E2M1 boundaries would be needed to
characterize the rare-disagreement rate.

## Exact integer representation

A decoded NVFP4 value `= e_int × m̂ × 2^(p−1)` with `e_int ∈ {0..12}` (5-bit signed after the ×2 that
clears the half-integer), E4M3 significand `m̂` (4-bit), exponent `p ∈ [−9, +5]`:

- **Bare E2M1 element**: signed **5-bit** integer (block scale kept factored, as a proof would).
- **Element × block scale on a common grid** (LSB `2^−10`, max `6 × 448 = 2688`): signed **23-bit**
  integer (`< 2^22` magnitude); `~20-bit` if block scales are restricted to *normal* E4M3.
- The **FP32 per-tensor scale factors out** of the matmul (one exact scalar multiply at the end).

## Implication for the proof system

An exact integer NVFP4 matmul fits the existing Freivalds design (`design-feasibility.md` §3.3):
factor out the two per-tensor FP32 scalars (one exact scalar product), absorb the per-block E4M3
scales into per-element `< 2^22` integers, verify `Â·B̂` by random projection (**O(n²)** witness and
constraints — the n³ products are never committed; the prover still computes the matmul because it is
running inference), then prove the output requantization via range/LogUp checks (O(n²)).

Field-wrap ceiling: the exact accumulator `P = Σ_k v_A v_B` satisfies `P < K·2^44`, so Goldilocks
(`≈ 2^64`) stays exact for `K ≲ 2^19` (worst case) / `K ≲ 2^23` (normal-only scales). FFN layers
(K ≤ 16k) are comfortable; **1M-context attention (`K = S`) exceeds it** and needs chunked
accumulation or a wider field — the same accumulation-width boundary seen at `K = 65536` above.

## Reproduction

On a Blackwell GPU (`pip install expecttest` for the torch test-helper import chain):

```
python nvfp4_matmul_test.py            # K/seed sweep: accumulation fidelity + output-code match
python nvfp4_scale_layout_bruteforce.py  # tries all scale layout x swizzle combos; finds the convention
```

Both use `torch._scaled_mm_v2` (hardware path) vs a fp64 decode-and-dot reference; no model weights
needed.
