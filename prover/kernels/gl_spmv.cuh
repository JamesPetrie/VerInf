// Sparse matrix-vector multiply mod P in CSR format.
//
// y = A · x mod P, where A is (n_rows × n_cols) given as CSR:
//   values    — length nnz, the non-zero entries
//   col_idx   — length nnz, column index of each non-zero
//   row_ptr   — length n_rows + 1, row_ptr[i] is the first nnz of row i
//
// One thread per row; serial dot over the row's nnz. This is the right
// distribution for Ligero constraint matrices, which have similar nnz
// per row in expectation (constraint = matmul column or LogUp sum row).
//
// Generalizes deprecated/spark-bench/cuda/commit_weights.cu's
// k_linear_rTA_zcy, which was specialized to a Freivalds witness layout.
// Same dot-product inner loop; the abstraction is just the standard CSR
// triple instead of a closed-form (m_C, n_C, c_offset, transpose).

#pragma once

#include <cstdint>
#include "goldilocks.cuh"
#include "blake3_hash.cuh"   // b3::hash_bytes for the inline combiner challenge

namespace gl_sparse {

__global__ void k_spmv(
    const uint64_t* __restrict__ values,    // nnz
    const uint64_t* __restrict__ col_idx,   // nnz; uint64 to match torch I/O
    const uint64_t* __restrict__ row_ptr,   // n_rows + 1
    const uint64_t* __restrict__ x,         // n_cols
    uint64_t* __restrict__ y,               // n_rows
    int n_rows
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_rows) return;
    uint64_t start = row_ptr[i];
    uint64_t end   = row_ptr[i + 1];
    uint64_t s = 0;
    for (uint64_t p = start; p < end; ++p) {
        s = gl::add(s, gl::mul(values[p], x[col_idx[p]]));
    }
    y[i] = s;
}

// Inline test-combiner challenge, bit-exact with protocol.challenge:
//   r[cid] = int.from_bytes(BLAKE3(seed(32) || label || cid_le8)[:16], "little") % P
// seed is 32 bytes; label is `label_len` bytes ("lin"/"irs"/"quad"). The hash
// input is <= 48 bytes — a single BLAKE3 chunk (fast path in b3::hash_bytes).
__device__ __forceinline__ uint64_t challenge_inline(
    const uint8_t* __restrict__ seed,
    const uint8_t* __restrict__ label, int label_len,
    uint64_t cid
) {
    uint8_t buf[48];
    #pragma unroll
    for (int i = 0; i < 32; ++i) buf[i] = seed[i];
    for (int i = 0; i < label_len; ++i) buf[32 + i] = label[i];
    #pragma unroll
    for (int i = 0; i < 8; ++i) buf[32 + label_len + i] = (uint8_t)(cid >> (8 * i));
    uint32_t h[8];
    b3::hash_bytes(buf, 32 + label_len + 8, h);
    uint64_t lo = (uint64_t)h[0] | ((uint64_t)h[1] << 32);
    uint64_t hi = (uint64_t)h[2] | ((uint64_t)h[3] << 32);
    return gl::reduce128(lo, hi);
}

// k_spmv with the dense combiner x[cid] replaced by challenge_inline(seed, cid)
// computed in-thread — the (up to ~1.2B-entry) combiner is never materialized.
//   y[i] = Σ_p values[p] · challenge(seed, col_idx[p], label).
__global__ void k_spmv_challenged(
    const uint64_t* __restrict__ values,    // nnz
    const uint64_t* __restrict__ col_idx,   // nnz; the constraint id (cid)
    const uint64_t* __restrict__ row_ptr,   // n_rows + 1
    const uint8_t*  __restrict__ seed,      // 32 bytes (s_comb)
    const uint8_t*  __restrict__ label, int label_len,
    uint64_t* __restrict__ y,               // n_rows
    int n_rows
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_rows) return;
    uint64_t start = row_ptr[i];
    uint64_t end   = row_ptr[i + 1];
    uint64_t s = 0;
    for (uint64_t p = start; p < end; ++p) {
        uint64_t r_cid = challenge_inline(seed, label, label_len, col_idx[p]);
        s = gl::add(s, gl::mul(values[p], r_cid));
    }
    y[i] = s;
}

// Materialize the combiner r[i] = challenge(seed, i, label) for i in [0, n) on
// the GPU — same inline challenge as k_spmv_challenged, for combiners consumed
// by other kernels (irs via gl_matvec, quad). Replaces the Python blake3 loop.
__global__ void k_challenge_vec(
    const uint8_t* __restrict__ seed,
    const uint8_t* __restrict__ label, int label_len,
    uint64_t* __restrict__ out, int64_t n
) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = challenge_inline(seed, label, label_len, (uint64_t)i);
}

} // namespace gl_sparse
