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

// Forward sweep for the Montgomery batched-inverse trick:
//   prefix[0]  = a[0]
//   prefix[i]  = prefix[i-1] * a[i]
//
// FUTURE OPTIMIZATION: this is a single-threaded GPU loop. Acceptable at
// today's per-column sizes (LogUp z = 1/(α - x) over a witness column,
// n ≤ a few million), where the serial sweep is still ms-scale and one
// launch. At billion-slot witnesses this becomes the bottleneck; replace
// with a multi-block prefix-product scan (standard Blelloch / Brent-Kung
// over a block-strided partial-product → finalize tree).
__global__ void k_inv_batched_forward(const uint64_t* __restrict__ a,
                                       uint64_t* __restrict__ prefix, int n) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    uint64_t acc = 1;
    for (int i = 0; i < n; ++i) {
        acc = gl::mul(acc, a[i]);
        prefix[i] = acc;
    }
}

// Backward sweep: given prefix[] and inv_total = (prefix[n-1])^{-1},
// fill out[i] = a[i]^{-1}. Uses:
//   inv_a[n-1] = inv_total * prefix[n-2]
//   inv_running = inv_total
//   for i = n-1 downto 1:
//     out[i]      = inv_running * prefix[i-1]
//     inv_running = inv_running * a[i]
//   out[0] = inv_running
//
// FUTURE OPTIMIZATION: single-threaded GPU loop, same constraint as the
// forward sweep. Parallel version reuses the same scan infrastructure
// (a suffix-product, or equivalently the same prefix-product algorithm
// applied to the reversed array).
__global__ void k_inv_batched_backward(const uint64_t* __restrict__ a,
                                        const uint64_t* __restrict__ prefix,
                                        uint64_t inv_total,
                                        uint64_t* __restrict__ out, int n) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    if (n == 0) return;
    if (n == 1) { out[0] = inv_total; return; }
    uint64_t inv_running = inv_total;
    for (int i = n - 1; i >= 1; --i) {
        out[i]      = gl::mul(inv_running, prefix[i - 1]);
        inv_running = gl::mul(inv_running, a[i]);
    }
    out[0] = inv_running;
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
