// Pairwise BLAKE3 Merkle internal-node hashing.
//
// One Merkle level: pairs of 32-byte child digests → N/2 parent digests.
// Each parent is plain BLAKE3 of (left || right) — single-block, single-chunk,
// flags = CHUNK_START | CHUNK_END | ROOT, counter = 0.
//
// Lifted unchanged from deprecated/spark-bench/cuda/commit_weights.cu where
// it ran end-to-end at Llama 2 7B q_proj scale. Wrap with a Python loop that
// launches one level per call until N==1; the final level returns the root.

#pragma once

#include <cstdint>
#include "blake3_compress.cuh"

namespace merkle {

__global__ void k_level(const uint32_t* __restrict__ in,
                        uint32_t* __restrict__ out,
                        int n_pairs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_pairs) return;
    uint32_t msg[16];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        msg[i]     = in[(idx * 2 + 0) * 8 + i];
        msg[i + 8] = in[(idx * 2 + 1) * 8 + i];
    }
    uint32_t cv[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) cv[i] = b3::IV[i];
    uint32_t flags = b3::CHUNK_START | b3::CHUNK_END | b3::ROOT;
    uint32_t out16[16];
    b3::compress(cv, msg, /*counter=*/0, /*block_len=*/64, flags, out16);
    #pragma unroll
    for (int i = 0; i < 8; ++i) out[idx * 8 + i] = out16[i];
}

} // namespace merkle
