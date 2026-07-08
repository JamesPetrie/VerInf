// Goldilocks dense matrix kernels: C = A · B mod P, and the matvec
// special case w = M · v mod P.
//
// Lifted from deprecated/spark-bench/cuda/gl_matmul_file.cu (matmul) and
// k_compute_z in deprecated/spark-bench/cuda/commit_weights.cu (matvec).
// Both ran end-to-end against Llama 2 7B q_proj (4096²) in the prior prover.
//
// The kernels are naive per-output-element form: one thread per C[i,j] for
// matmul, one thread per w[i] for matvec, serial dot product over the
// contracted dimension.
//
// FUTURE OPTIMIZATION: the matmul kernel hits 0.30s at 4096³ on GB10,
// which is ~225 Gmul/s vs the 312 Gmul/s peak measured by bench_field_mul.
// Closing that gap needs a tiled / register-blocked / shared-memory-staged
// variant: load (BM × BK) of A and (BK × BN) of B into shared, compute a
// (BM × BN) tile per thread block with each thread accumulating its own
// (TM × TN) micro-tile in registers. Standard CUDA matmul recipe; the
// only Goldilocks-specific complexity is that gl::mul has a 128-bit
// intermediate so register pressure is roughly 2× of an FP32 matmul.

#pragma once

#include <cstdint>
#include "goldilocks.cuh"

namespace gl_dense {

__global__ void k_matmul(
    const uint64_t* __restrict__ A,    // m × k row-major
    const uint64_t* __restrict__ B,    // k × n row-major
    uint64_t* __restrict__ C,          // m × n row-major
    int m, int k, int n
) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m || j >= n) return;
    uint64_t s = 0;
    const uint64_t* a_row = A + (size_t)i * k;
    for (int l = 0; l < k; ++l) {
        s = gl::add(s, gl::mul(a_row[l], B[(size_t)l * n + j]));
    }
    C[(size_t)i * n + j] = s;
}

__global__ void k_matvec(
    const uint64_t* __restrict__ M,    // m × n row-major
    const uint64_t* __restrict__ v,    // n
    uint64_t* __restrict__ w,          // m
    int m, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    uint64_t s = 0;
    const uint64_t* m_row = M + (size_t)i * n;
    for (int j = 0; j < n; ++j) {
        s = gl::add(s, gl::mul(m_row[j], v[j]));
    }
    w[i] = s;
}

} // namespace gl_dense
