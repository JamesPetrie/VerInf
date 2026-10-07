// New Goldilocks utility kernels for the cuda_primitives wrapper.
//
// Companion to the existing gl:: arithmetic in goldilocks.cuh. These are
// the device kernels that did NOT exist in the deprecated commit_weights.cu
// surface — they target the LogUp side of the protocol (multiplicity
// histograms, batched inverses for z = 1/(α - x)) plus a couple of small
// generic helpers (gl_neg, in-place gl_axpy, polynomial Horner evaluation).
//
// All kernels operate on uint64 buffers in canonical Goldilocks form
// (values in [0, P)). The torch wrapper in cuda_primitives.py enforces
// dtype/device/contiguity.

#pragma once

#include <cstdint>
#include "goldilocks.cuh"

namespace gl_extras {

// y[i] = (P - a[i]) mod P. Returns 0 for input 0 (correct since the
// canonical representative of −0 is 0).
__global__ void k_neg(const uint64_t* __restrict__ a,
                      uint64_t* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint64_t v = a[i];
    y[i] = (v == 0) ? 0 : (gl::P - v);
}

// In-place y[i] += alpha * x[i] mod P. Differs from
// accumulators.cuh::k_scalar_mul_accumulate only by signature: this one
// takes y as both input and output (load+store), the accumulator form
// reads from a separate row_codeword buffer.
__global__ void k_axpy_inplace(uint64_t alpha,
                               const uint64_t* __restrict__ x,
                               uint64_t* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = gl::add(y[i], gl::mul(alpha, x[i]));
}

// Horner polynomial evaluation. For each (row r, point p):
//   out[r, p] = Σ_k coeffs[r, k] · points[p]^k
// One thread per (row, point) cell. 1-D coeffs is the m=1 special case.
__global__ void k_poly_eval(const uint64_t* __restrict__ coeffs,   // (m, d) row-major
                             int m, int d,
                             const uint64_t* __restrict__ points,   // k
                             int k_pts,
                             uint64_t* __restrict__ out) {           // (m, k) row-major
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (p >= k_pts || r >= m) return;
    uint64_t x = points[p];
    uint64_t acc = 0;
    const uint64_t* row = coeffs + (size_t)r * d;
    for (int i = d - 1; i >= 0; --i) {
        acc = gl::add(gl::mul(acc, x), row[i]);
    }
    out[(size_t)r * k_pts + p] = acc;
}

// Multiplicity histogram for LogUp range / functional lookups.
// For each x[i]: if x[i] equals some table[j], atomicAdd 1 into mult[j].
// table is assumed small (≤ 2^16 entries per design-feasibility.md §B);
// each thread does a serial scan. Out-of-range x[i] (no matching j)
// contributes nothing — matches Python compute_multiplicities() semantics.
//
// For range tables that are literally [0, T_LEN), use k_lookup_multiplicities_range
// below — direct indexing, O(1) per witness element instead of O(T_LEN).
__global__ void k_lookup_multiplicities(
    const uint64_t* __restrict__ x, int n_x,
    const uint64_t* __restrict__ table, int n_table,
    unsigned long long* __restrict__ mult       // atomic-friendly type
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_x) return;
    uint64_t xi = x[i];
    for (int j = 0; j < n_table; ++j) {
        if (table[j] == xi) {
            atomicAdd(&mult[j], 1ULL);
            return;
        }
    }
}

// Specialized variant for range tables T = [0, 1, ..., T_LEN-1]. Each
// witness element either hits its slot directly (mult[x[i]]++) or falls
// out of range (no match → no contribution, matching the general kernel).
// Reduces per-thread work from O(T_LEN) to O(1). Caller must guarantee the
// underlying table is actually [0, T_LEN); the Python wrapper probes once
// per unique table pointer and caches the verdict.
__global__ void k_lookup_multiplicities_range(
    const uint64_t* __restrict__ x, int n_x,
    uint64_t T_LEN,
    unsigned long long* __restrict__ mult
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_x) return;
    uint64_t xi = x[i];
    if (xi < T_LEN) atomicAdd(&mult[xi], 1ULL);
}

} // namespace gl_extras
