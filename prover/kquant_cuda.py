"""Fused K-quant → Goldilocks-field CUDA kernels.

Takes raw GGUF K-quant super-block bytes (Q4_K / Q5_K / Q6_K) and emits the
committed field integers directly on the GPU — no fp32 intermediate array, no
CPU dequantization. Bit-exact with the reference numpy path
(gguf.quants.dequantize → loader.quantize_to_field): the per-element fp32
expression order mirrors gguf-py's dequantize_blocks exactly, and the final
round((double)v · S) uses round-half-to-even, matching torch.round on f64.
THE BYTE→INTEGER MAP IS THE DECLARED MODEL — any change here must stay
bit-identical to the reference (tests/test_kquant_kernel.py enforces this
against the real file).

Block layouts (from gguf-py quants.py / llama.cpp ggml-quants):
  Q4_K (144 B / 256 elems): d f16 · dmin f16 · scales[12] (6-bit sc/min × 8
        sub-blocks) · qs[128] (4-bit pairs, 32-byte chunks: low nibbles =
        sub-block 2c, high = 2c+1)
  Q5_K (176 B): d · dmin · scales[12] · qh[32] (bit b=e/32 of byte e%32) ·
        qs[128] as Q4_K; q = ql | qh<<4
  Q6_K (210 B): ql[128] (4-bit, 64-byte chunks) · qh[64] (2-bit, shift
        2·((e>>5)&3)) · scales[16] int8 (per 16 elems) · d f16; q ∈ [−32, 31]
"""
import torch
from torch.utils.cpp_extension import load_inline

QK_K = 256
BLOCK_BYTES = {"Q4_K": 144, "Q5_K": 176, "Q6_K": 210}

_CPP = r"""
#include <torch/extension.h>
torch::Tensor q4k_to_field(torch::Tensor raw, double S);
torch::Tensor q5k_to_field(torch::Tensor raw, double S);
torch::Tensor q6k_to_field(torch::Tensor raw, double S);
"""

_CUDA = r"""
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cstdint>

#define GL_P 18446744069414584321ULL

__device__ __forceinline__ uint64_t to_field(float v, double S) {
    long long ll = __double2ll_rn((double)v * S);     // round-half-to-even
    return ll >= 0 ? (uint64_t)ll : GL_P - (uint64_t)(-ll);
}

// get_scale_min: 12 bytes -> 6-bit (sc, min) for sub-block s in [0,8)
__device__ __forceinline__ void scale_min(const uint8_t* sc12, int s,
                                           uint32_t* sc, uint32_t* mn) {
    if (s < 4) {
        *sc = sc12[s] & 0x3F;
        *mn = sc12[s + 4] & 0x3F;
    } else {
        int j = s - 4;
        *sc = (sc12[8 + j] & 0x0F) | ((sc12[j] >> 2) & 0x30);
        *mn = (sc12[8 + j] >> 4)   | ((sc12[j + 4] >> 2) & 0x30);
    }
}

__global__ void q4k_kernel(const uint8_t* __restrict__ raw,
                            uint64_t* __restrict__ out,
                            long n, double S) {
    long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (idx >= n) return;
    long b = idx >> 8; int e = idx & 255;
    const uint8_t* p = raw + b * 144;
    float d    = __half2float(*(const __half*)(p));
    float dmin = __half2float(*(const __half*)(p + 2));
    uint32_t sc, mn; scale_min(p + 4, e >> 5, &sc, &mn);
    const uint8_t* qs = p + 16;
    int q = (qs[(e >> 6) * 32 + (e & 31)] >> (4 * ((e >> 5) & 1))) & 0xF;
    float v = (d * (float)sc) * (float)q - (dmin * (float)mn);
    out[idx] = to_field(v, S);
}

__global__ void q5k_kernel(const uint8_t* __restrict__ raw,
                            uint64_t* __restrict__ out,
                            long n, double S) {
    long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (idx >= n) return;
    long b = idx >> 8; int e = idx & 255;
    const uint8_t* p = raw + b * 176;
    float d    = __half2float(*(const __half*)(p));
    float dmin = __half2float(*(const __half*)(p + 2));
    uint32_t sc, mn; scale_min(p + 4, e >> 5, &sc, &mn);
    const uint8_t* qh = p + 16;
    const uint8_t* qs = p + 48;
    int ql = (qs[(e >> 6) * 32 + (e & 31)] >> (4 * ((e >> 5) & 1))) & 0xF;
    int hb = (qh[e & 31] >> (e >> 5)) & 1;
    int q  = ql | (hb << 4);
    float v = (d * (float)sc) * (float)q - (dmin * (float)mn);
    out[idx] = to_field(v, S);
}

__global__ void q6k_kernel(const uint8_t* __restrict__ raw,
                            uint64_t* __restrict__ out,
                            long n, double S) {
    long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (idx >= n) return;
    long b = idx >> 8; int e = idx & 255;
    const uint8_t* p  = raw + b * 210;
    const uint8_t* ql = p;
    const uint8_t* qh = p + 128;
    const int8_t*  sc = (const int8_t*)(p + 192);
    float d = __half2float(*(const __half*)(p + 208));
    int lo = (ql[(e >> 7) * 64 + (e & 63)] >> (4 * ((e >> 6) & 1))) & 0xF;
    int hi = (qh[(e >> 7) * 32 + (e & 31)] >> (2 * ((e >> 5) & 3))) & 3;
    int q  = (int)(int8_t)(lo | (hi << 4)) - 32;
    float v = (d * (float)sc[e >> 4]) * (float)q;
    out[idx] = to_field(v, S);
}

torch::Tensor q4k_to_field(torch::Tensor raw, double S) {
    long n_blocks = raw.numel() / 144, n = n_blocks * 256;
    TORCH_CHECK(raw.is_cuda() && raw.dtype() == torch::kUInt8 && raw.is_contiguous());
    TORCH_CHECK(n_blocks * 144 == raw.numel());
    auto out = torch::empty({n}, torch::dtype(torch::kUInt64).device(raw.device()));
    q4k_kernel<<<(n + 255) / 256, 256>>>((const uint8_t*)raw.data_ptr(),
                                          (uint64_t*)out.data_ptr(), n, S);
    return out;
}
torch::Tensor q5k_to_field(torch::Tensor raw, double S) {
    long n_blocks = raw.numel() / 176, n = n_blocks * 256;
    TORCH_CHECK(raw.is_cuda() && raw.dtype() == torch::kUInt8 && raw.is_contiguous());
    TORCH_CHECK(n_blocks * 176 == raw.numel());
    auto out = torch::empty({n}, torch::dtype(torch::kUInt64).device(raw.device()));
    q5k_kernel<<<(n + 255) / 256, 256>>>((const uint8_t*)raw.data_ptr(),
                                          (uint64_t*)out.data_ptr(), n, S);
    return out;
}
torch::Tensor q6k_to_field(torch::Tensor raw, double S) {
    long n_blocks = raw.numel() / 210, n = n_blocks * 256;
    TORCH_CHECK(raw.is_cuda() && raw.dtype() == torch::kUInt8 && raw.is_contiguous());
    TORCH_CHECK(n_blocks * 210 == raw.numel());
    auto out = torch::empty({n}, torch::dtype(torch::kUInt64).device(raw.device()));
    q6k_kernel<<<(n + 255) / 256, 256>>>((const uint8_t*)raw.data_ptr(),
                                          (uint64_t*)out.data_ptr(), n, S);
    return out;
}
"""

_module = None


def _arch_flag():
    """nvcc arch for the local GPU (sm_121 = GB10/Spark, sm_90 = H100, ...).
    The wrong arch silently produces wrong kernel results on some pairs —
    the original hardcoded sm_121 broke any non-Spark machine."""
    cap = torch.cuda.get_device_capability()
    return f"-arch=sm_{cap[0]}{cap[1]}"


def _ensure():
    global _module
    if _module is None:
        _module = load_inline(
            name="kquant_to_field",
            cpp_sources=_CPP,
            cuda_sources=_CUDA,
            functions=["q4k_to_field", "q5k_to_field", "q6k_to_field"],
            # -fmad=false: numpy computes mul-then-sub as two rounded fp32 ops;
            # FMA contraction would round differently and break bit-exactness.
            extra_cuda_cflags=[_arch_flag(), "-O3", "-std=c++17", "-fmad=false"],
            extra_cflags=["-std=c++17"],
            verbose=False,
        )
    return _module


def kquant_to_field(raw_u8_cuda: torch.Tensor, qtype: str, S: int) -> torch.Tensor:
    """raw K-quant bytes (any shape, contiguous, CUDA uint8) → flat uint64
    field tensor of n_blocks·256 committed integers (row-major as stored)."""
    m = _ensure()
    fn = {"Q4_K": m.q4k_to_field, "Q5_K": m.q5k_to_field,
          "Q6_K": m.q6k_to_field}.get(qtype)
    assert fn is not None, f"kquant_to_field: unsupported type {qtype}"
    return fn(raw_u8_cuda.reshape(-1), float(S))
