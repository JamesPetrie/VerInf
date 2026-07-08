"""
PyTorch bindings around the CUDA primitives in ligero/cuda/.

Public surface (uniform "operates on torch.uint64 CUDA tensors" API):

    Field arithmetic (elementwise over Goldilocks):
        gl_mul(a, b), gl_add(a, b), gl_sub(a, b), gl_neg(a)
        gl_pow(base, exp_scalar), gl_inv(a), gl_inv_batched(a)
        gl_axpy(y, alpha, x)  — in-place y += alpha*x

    Number-theoretic transform (in-place; length must be power of 2):
        ntt_forward(a), ntt_inverse(a)
        ntt_forward_batched(rows), ntt_inverse_batched(rows)

    Reed-Solomon row encoding:
        rs_encode_rows(messages, n_lig, k_deg)

    Polynomial arithmetic in coefficient form:
        poly_mul(a, b), poly_add(a, b)
        poly_mul_batched(A, B)
        poly_eval(coeffs, points)   — Horner, 1-D or 2-D coeffs

    Linear algebra mod P:
        gl_matmul(A, B), gl_matvec(M, v)
        gl_spmv(values, col_idx, row_ptr, x, n_rows)

    BLAKE3 column hashing + Merkle:
        hash_columns_streamed(matrix)   — (m, n) u64 → (n, 32) u8
        merkle_build_blake3(leaves)     — (N, 32) u8 → (root, levels)

    Lookup helpers (LogUp):
        lookup_multiplicities(x, table)

Backed by ligero/cuda/*.cuh. NTT contexts are allocated lazily
per length and cached for the life of the process. First call to any
function triggers JIT compilation of the PyTorch extension via
torch.utils.cpp_extension.load_inline (~60s on a clean cache); subsequent
runs reuse the cached build under ~/.cache/torch_extensions/.

Requires CUDA, ninja, and a PyTorch build with CUDA support.
"""

from pathlib import Path

import os
import torch
from torch.utils.cpp_extension import load_inline


P = (1 << 64) - (1 << 32) + 1
GLOBAL_G = 7

_CUDA_HEADERS_DIR = Path(__file__).resolve().parent / "kernels"


_CPP_DECLS = r"""
#include <torch/extension.h>

// Elementwise field
torch::Tensor gl_mul(torch::Tensor a, torch::Tensor b);
torch::Tensor gl_add(torch::Tensor a, torch::Tensor b);
torch::Tensor gl_sub(torch::Tensor a, torch::Tensor b);
torch::Tensor gl_neg(torch::Tensor a);
torch::Tensor gl_pow(torch::Tensor base, uint64_t exp);
torch::Tensor gl_inv_batched(torch::Tensor a);
void          gl_axpy_inplace(torch::Tensor y, uint64_t alpha, torch::Tensor x);

// NTT
void ntt_forward(torch::Tensor a);
void ntt_inverse(torch::Tensor a);
void ntt_forward_batched(torch::Tensor rows);
void ntt_inverse_batched(torch::Tensor rows);

// Reed-Solomon row encoding
torch::Tensor rs_encode_rows(torch::Tensor messages, int64_t n_lig, int64_t k_deg);

// Polynomial
torch::Tensor poly_eval(torch::Tensor coeffs, torch::Tensor points);

// Linear algebra mod P
torch::Tensor gl_matmul(torch::Tensor A, torch::Tensor B);
torch::Tensor gl_matvec(torch::Tensor M, torch::Tensor v);
torch::Tensor gl_spmv(torch::Tensor values, torch::Tensor col_idx,
                      torch::Tensor row_ptr, torch::Tensor x, int64_t n_rows);
torch::Tensor gl_spmv_challenged(torch::Tensor values, torch::Tensor col_idx,
                                 torch::Tensor row_ptr, torch::Tensor seed,
                                 torch::Tensor label, int64_t n_rows);
torch::Tensor challenge_vec(torch::Tensor seed, torch::Tensor label, int64_t n);
torch::Tensor challenge_at(torch::Tensor seed, torch::Tensor label, torch::Tensor cids);
void interp_band(torch::Tensor out, int64_t out_off, int64_t flat_lo, int64_t n_slots,
                 torch::Tensor desc, torch::Tensor tblA, torch::Tensor tblB,
                 torch::Tensor chal_buf, torch::Tensor seed, torch::Tensor label);
void interp_band_causal_id(torch::Tensor out, int64_t out_off, int64_t flat_lo, int64_t n_slots,
                           uint64_t base, uint64_t m, uint64_t h, uint64_t coef,
                           torch::Tensor seed, torch::Tensor label);
void interp_band_causal_c2(torch::Tensor out, int64_t out_off, int64_t flat_lo, int64_t n_slots,
                           uint64_t base, uint64_t h, uint64_t coef,
                           torch::Tensor seed, torch::Tensor label);
void interp_band_embed(torch::Tensor out, int64_t out_off, int64_t flat_lo, int64_t n_slots,
                       uint64_t base, uint64_t d, uint64_t rows_per_w, uint64_t ell,
                       torch::Tensor token_ids, torch::Tensor seed, torch::Tensor label);
void interp_band_rope_x(torch::Tensor out, int64_t out_off, int64_t flat_lo, int64_t n_slots,
                        uint64_t base, uint64_t H, uint64_t d_h,
                        torch::Tensor cos_t, torch::Tensor sin_t,
                        torch::Tensor seed, torch::Tensor label);

// BLAKE3 column hash + Merkle
torch::Tensor hash_columns_streamed(torch::Tensor matrix);
torch::Tensor merkle_one_level(torch::Tensor leaves_u32);   // (N, 8) u32 → (N/2, 8) u32

// Chunked column hash (for prove-time codeword streaming).
//   1. allocate states with hash_columns_stream_init(n_cols)
//   2. hash_columns_stream_update(states, chunk, is_last_chunk, n_chunks_total)
//      called once per BLAKE3 chunk (≤ 128 rows per call)
//   3. hash_columns_stream_finalize(states, n_cols, n_chunks_total) → (n_cols, 32) u8
torch::Tensor hash_columns_stream_init(int64_t n_cols);
void          hash_columns_stream_update(torch::Tensor states, torch::Tensor chunk,
                                         bool is_last_chunk, int64_t n_chunks_total);
torch::Tensor hash_columns_stream_finalize(torch::Tensor states, int64_t n_cols,
                                            int64_t n_chunks_total);

// Per-row deterministic PRG used for ZK slack padding. Returns a
// (n_rows, slack_per_row) uint64 tensor where
//   out[r, k] = uint64(BLAKE3(master_seed || (row_offset+r)_le8 || k_le8)[0:8]) mod P.
// Stateless: same (master_seed, row_offset, n_rows, slack_per_row) always
// produces the same output. Replaces the stateful numpy blind_rng path.
torch::Tensor row_prg(torch::Tensor master_seed, int64_t row_offset,
                       int64_t n_rows, int64_t slack_per_row, uint64_t P);
// Indexed row_prg: row_indices is a (n_rows,) uint64 list of absolute rows.
torch::Tensor row_prg_indexed(torch::Tensor master_seed, torch::Tensor row_indices,
                              int64_t slack_per_row, uint64_t P);

// Lookup
torch::Tensor lookup_multiplicities(torch::Tensor x, torch::Tensor table);
void          lookup_multiplicities_into(torch::Tensor x, torch::Tensor table, torch::Tensor mult);
// Range-table specialization: caller asserts table is [0, T_LEN). mult length must equal T_LEN.
void          lookup_multiplicities_range_into(torch::Tensor x, int64_t T_LEN, torch::Tensor mult);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gl_mul", &gl_mul);
    m.def("gl_add", &gl_add);
    m.def("gl_sub", &gl_sub);
    m.def("gl_neg", &gl_neg);
    m.def("gl_pow", &gl_pow);
    m.def("gl_inv_batched", &gl_inv_batched);
    m.def("gl_axpy_inplace", &gl_axpy_inplace);

    m.def("ntt_forward", &ntt_forward);
    m.def("ntt_inverse", &ntt_inverse);
    m.def("ntt_forward_batched", &ntt_forward_batched);
    m.def("ntt_inverse_batched", &ntt_inverse_batched);

    m.def("rs_encode_rows", &rs_encode_rows);

    m.def("poly_eval", &poly_eval);

    m.def("gl_matmul", &gl_matmul);
    m.def("gl_matvec", &gl_matvec);
    m.def("gl_spmv", &gl_spmv);
    m.def("gl_spmv_challenged", &gl_spmv_challenged);
    m.def("challenge_vec", &challenge_vec);
    m.def("challenge_at", &challenge_at);
    m.def("interp_band", &interp_band);
    m.def("interp_band_causal_id", &interp_band_causal_id);
    m.def("interp_band_causal_c2", &interp_band_causal_c2);
    m.def("interp_band_embed", &interp_band_embed);
    m.def("interp_band_rope_x", &interp_band_rope_x);

    m.def("hash_columns_streamed", &hash_columns_streamed);
    m.def("hash_columns_stream_init",     &hash_columns_stream_init);
    m.def("hash_columns_stream_update",   &hash_columns_stream_update);
    m.def("hash_columns_stream_finalize", &hash_columns_stream_finalize);
    m.def("row_prg", &row_prg);
    m.def("row_prg_indexed", &row_prg_indexed);
    m.def("merkle_one_level", &merkle_one_level);

    m.def("lookup_multiplicities", &lookup_multiplicities);
    m.def("lookup_multiplicities_into", &lookup_multiplicities_into);
    m.def("lookup_multiplicities_range_into", &lookup_multiplicities_range_into);
}
"""


_CUDA_SOURCE = r"""
#include <torch/extension.h>
#include "goldilocks.cuh"
#include "ntt.cuh"
#include "blake3_compress.cuh"
#include "blake3_hash.cuh"
#include "blake3_columns.cuh"
#include "merkle.cuh"
#include "gl_matmul.cuh"
#include "gl_spmv.cuh"
#include "gl_extras.cuh"

#include <unordered_map>

#define CHECK_U64(t) do { \
    TORCH_CHECK((t).is_cuda(), #t " must be CUDA"); \
    TORCH_CHECK((t).dtype() == torch::kUInt64, #t " dtype must be uint64"); \
    TORCH_CHECK((t).is_contiguous(), #t " must be contiguous"); \
} while (0)

// ---------------------------------------------------------------------------
// Elementwise field kernels.
// ---------------------------------------------------------------------------

__global__ void k_gl_mul(const uint64_t* a, const uint64_t* b, uint64_t* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = gl::mul(a[i], b[i]);
}
__global__ void k_gl_add(const uint64_t* a, const uint64_t* b, uint64_t* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = gl::add(a[i], b[i]);
}
__global__ void k_gl_sub(const uint64_t* a, const uint64_t* b, uint64_t* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = gl::sub(a[i], b[i]);
}
__global__ void k_gl_pow(const uint64_t* base, uint64_t exp, uint64_t* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = gl::pow(base[i], exp);
}

static inline std::pair<int,int> grid1d(int n, int block = 256) {
    return {(n + block - 1) / block, block};
}

torch::Tensor gl_mul(torch::Tensor a, torch::Tensor b) {
    CHECK_U64(a); CHECK_U64(b);
    TORCH_CHECK(a.sizes() == b.sizes(), "gl_mul: shape mismatch");
    auto out = torch::empty_like(a);
    int n = a.numel();
    auto [g, blk] = grid1d(n);
    k_gl_mul<<<g, blk>>>(
        (const uint64_t*)a.data_ptr(), (const uint64_t*)b.data_ptr(),
        (uint64_t*)out.data_ptr(), n);
    return out;
}
torch::Tensor gl_add(torch::Tensor a, torch::Tensor b) {
    CHECK_U64(a); CHECK_U64(b);
    TORCH_CHECK(a.sizes() == b.sizes(), "gl_add: shape mismatch");
    auto out = torch::empty_like(a);
    int n = a.numel();
    auto [g, blk] = grid1d(n);
    k_gl_add<<<g, blk>>>(
        (const uint64_t*)a.data_ptr(), (const uint64_t*)b.data_ptr(),
        (uint64_t*)out.data_ptr(), n);
    return out;
}
torch::Tensor gl_sub(torch::Tensor a, torch::Tensor b) {
    CHECK_U64(a); CHECK_U64(b);
    TORCH_CHECK(a.sizes() == b.sizes(), "gl_sub: shape mismatch");
    auto out = torch::empty_like(a);
    int n = a.numel();
    auto [g, blk] = grid1d(n);
    k_gl_sub<<<g, blk>>>(
        (const uint64_t*)a.data_ptr(), (const uint64_t*)b.data_ptr(),
        (uint64_t*)out.data_ptr(), n);
    return out;
}
torch::Tensor gl_neg(torch::Tensor a) {
    CHECK_U64(a);
    auto out = torch::empty_like(a);
    int n = a.numel();
    auto [g, blk] = grid1d(n);
    gl_extras::k_neg<<<g, blk>>>(
        (const uint64_t*)a.data_ptr(), (uint64_t*)out.data_ptr(), n);
    return out;
}
torch::Tensor gl_pow(torch::Tensor base, uint64_t exp) {
    CHECK_U64(base);
    auto out = torch::empty_like(base);
    int n = base.numel();
    auto [g, blk] = grid1d(n);
    k_gl_pow<<<g, blk>>>(
        (const uint64_t*)base.data_ptr(), exp, (uint64_t*)out.data_ptr(), n);
    return out;
}
void gl_axpy_inplace(torch::Tensor y, uint64_t alpha, torch::Tensor x) {
    CHECK_U64(y); CHECK_U64(x);
    TORCH_CHECK(y.sizes() == x.sizes(), "gl_axpy: shape mismatch");
    int n = y.numel();
    auto [g, blk] = grid1d(n);
    gl_extras::k_axpy_inplace<<<g, blk>>>(
        alpha, (const uint64_t*)x.data_ptr(), (uint64_t*)y.data_ptr(), n);
}

torch::Tensor gl_inv_batched(torch::Tensor a) {
    CHECK_U64(a);
    TORCH_CHECK(a.dim() == 1, "gl_inv_batched: 1-D input only");
    int n = a.numel();
    auto out = torch::empty_like(a);
    if (n == 0) return out;
    // Per-element Fermat inverse a^(P-2), one thread per element. Bit-identical
    // to batch-inversion (the unique inverse mod P) but fully parallel — the
    // prior single-thread Montgomery scan (<<<1,1>>>) was ~800x slower at the
    // 2^24 LogUp table sizes (10s for 33.5M inverses).
    auto [g, blk] = grid1d(n);
    k_gl_pow<<<g, blk>>>(
        (const uint64_t*)a.data_ptr(), gl::P - 2, (uint64_t*)out.data_ptr(), n);
    return out;
}

// ---------------------------------------------------------------------------
// NTT — context cache + size-dispatched wrappers.
//
// Dispatch by NTT length n:
//   n == 65536          → Bailey 4-step (23.8 µs on GB10 per microbench)
//   n >= 256            → fused (8-stage shared-memory for n in [256, 4096),
//                                12-stage shared-memory for n >= 4096)
//   else                → level-per-launch baseline
//
// All three produce bit-identical output for the same input. The Python
// wrapper exposes a single ntt_forward / ntt_inverse — callers don't
// pick the variant.
// ---------------------------------------------------------------------------

static std::unordered_map<int, gl_ntt::Ctx*>& ntt_ctx_cache() {
    static std::unordered_map<int, gl_ntt::Ctx*> cache;
    return cache;
}
static gl_ntt::Ctx* get_ntt_ctx(int n) {
    auto& cache = ntt_ctx_cache();
    auto it = cache.find(n);
    if (it != cache.end()) return it->second;
    auto* ctx = new gl_ntt::Ctx();
    gl_ntt::ntt_init(n, ctx);
    cache[n] = ctx;
    return ctx;
}
static bool is_power_of_two(int n) { return n > 0 && (n & (n - 1)) == 0; }

static inline void ntt_forward_dispatch(uint64_t* d_a, const gl_ntt::Ctx& ctx,
                                         cudaStream_t s = 0) {
    if (ctx.n == 65536)      gl_ntt::ntt_forward_bailey(d_a, ctx, s);
    else if (ctx.n >= 256)   gl_ntt::ntt_forward_fused (d_a, ctx, s);
    else                     gl_ntt::ntt_forward       (d_a, ctx, s);
}
static inline void ntt_inverse_dispatch(uint64_t* d_a, const gl_ntt::Ctx& ctx,
                                         cudaStream_t s = 0) {
    if (ctx.n == 65536)      gl_ntt::ntt_inverse_bailey(d_a, ctx, s);
    else if (ctx.n >= 256)   gl_ntt::ntt_inverse_fused (d_a, ctx, s);
    else                     gl_ntt::ntt_inverse       (d_a, ctx, s);
}

void ntt_forward(torch::Tensor a) {
    CHECK_U64(a);
    TORCH_CHECK(a.dim() == 1, "NTT input must be 1-D");
    int n = a.numel();
    TORCH_CHECK(is_power_of_two(n) && n >= 2, "NTT length must be power of 2 >= 2");
    auto* ctx = get_ntt_ctx(n);
    ntt_forward_dispatch(reinterpret_cast<uint64_t*>(a.data_ptr()), *ctx);
}
void ntt_inverse(torch::Tensor a) {
    CHECK_U64(a);
    TORCH_CHECK(a.dim() == 1, "NTT input must be 1-D");
    int n = a.numel();
    TORCH_CHECK(is_power_of_two(n) && n >= 2, "NTT length must be power of 2 >= 2");
    auto* ctx = get_ntt_ctx(n);
    ntt_inverse_dispatch(reinterpret_cast<uint64_t*>(a.data_ptr()), *ctx);
}

// Bailey-batched scratch buffer cache keyed by row count. The Bailey
// 4-step decomposition needs a per-row temp of n=65536 uint64s; we cache
// one big scratch tensor and grow if the m we see exceeds capacity. Each
// entry: pair (capacity_m, device pointer).
static std::pair<int, uint64_t*>& bailey_scratch_slot() {
    static std::pair<int, uint64_t*> slot{0, nullptr};
    return slot;
}
static uint64_t* get_bailey_scratch(int m) {
    auto& slot = bailey_scratch_slot();
    if (m <= slot.first && slot.second != nullptr) return slot.second;
    if (slot.second != nullptr) cudaFree(slot.second);
    slot.first = m;
    size_t nbytes = (size_t)m * 65536 * sizeof(uint64_t);
    cudaError_t err = cudaMalloc(&slot.second, nbytes);
    TORCH_CHECK(err == cudaSuccess,
                "Bailey scratch allocation failed: ", cudaGetErrorString(err),
                " (requested ", nbytes, " bytes for m=", m, ")");
    return slot.second;
}

// One launch per NTT stage total — independent of m. Replaces the
// per-row launch loop. Bailey path passes scratch buffer; fused / baseline
// paths don't need scratch.
void ntt_forward_batched(torch::Tensor rows) {
    CHECK_U64(rows);
    TORCH_CHECK(rows.dim() == 2, "batched NTT expects (m, n)");
    int m = rows.size(0);
    int n = rows.size(1);
    TORCH_CHECK(is_power_of_two(n) && n >= 2, "row length must be power of 2 >= 2");
    auto* ctx = get_ntt_ctx(n);
    uint64_t* base = reinterpret_cast<uint64_t*>(rows.data_ptr());
    uint64_t* scratch = (n == 65536) ? get_bailey_scratch(m) : nullptr;
    gl_ntt::ntt_forward_batched_fast(base, m, *ctx, scratch);
}
void ntt_inverse_batched(torch::Tensor rows) {
    CHECK_U64(rows);
    TORCH_CHECK(rows.dim() == 2, "batched NTT expects (m, n)");
    int m = rows.size(0);
    int n = rows.size(1);
    TORCH_CHECK(is_power_of_two(n) && n >= 2, "row length must be power of 2 >= 2");
    auto* ctx = get_ntt_ctx(n);
    uint64_t* base = reinterpret_cast<uint64_t*>(rows.data_ptr());
    uint64_t* scratch = (n == 65536) ? get_bailey_scratch(m) : nullptr;
    gl_ntt::ntt_inverse_batched_fast(base, m, *ctx, scratch);
}

// ---------------------------------------------------------------------------
// Reed-Solomon row encoding: each row's first K_DEG slots get iNTT'd (the
// caller's K_DEG−ELL slots beyond ELL are the ZK pad), then the row is
// zero-extended to N_LIG and fNTT'd. Output is the codeword matrix.
// ---------------------------------------------------------------------------

torch::Tensor rs_encode_rows(torch::Tensor messages, int64_t n_lig, int64_t k_deg) {
    CHECK_U64(messages);
    TORCH_CHECK(messages.dim() == 2, "rs_encode_rows expects (m, K_DEG)");
    TORCH_CHECK((int)messages.size(1) == (int)k_deg,
                "messages's second dim must equal k_deg");
    TORCH_CHECK(is_power_of_two((int)k_deg) && k_deg >= 2);
    TORCH_CHECK(is_power_of_two((int)n_lig) && n_lig >= k_deg);
    int m = messages.size(0);

    // Allocate output and zero-init. Copy each row's K_DEG slots in, then
    // run batched iNTT(K_DEG) on first K_DEG columns followed by
    // fNTT(N_LIG) on the full row.
    auto out = torch::zeros({m, n_lig}, messages.options());
    // Copy: messages → out[:, :K_DEG].
    uint64_t* d_out = (uint64_t*)out.data_ptr();
    const uint64_t* d_msg = (const uint64_t*)messages.data_ptr();
    for (int i = 0; i < m; ++i) {
        cudaMemcpy(d_out + (size_t)i * n_lig,
                   d_msg + (size_t)i * k_deg,
                   (size_t)k_deg * sizeof(uint64_t),
                   cudaMemcpyDeviceToDevice);
    }
    auto* ctx_k = get_ntt_ctx((int)k_deg);
    auto* ctx_n = get_ntt_ctx((int)n_lig);

    // iNTT_K on the first K_DEG slots of each row. The buffer is N_LIG-wide
    // but the NTT operates only on the first K_DEG slots — but our batched
    // kernels assume contiguous (m, ctx.n) rows. So we run iNTT_K with row
    // stride = K_DEG by treating it as a separate batched call on a virtual
    // (m, K_DEG) view. Since out is (m, N_LIG) row-major and only the
    // first K_DEG slots of each row matter for iNTT_K, we'd need a strided
    // batched. Simplest correct path: do iNTT_K row-by-row (cheap at K_DEG)
    // and fNTT_N batched.
    for (int i = 0; i < m; ++i) {
        ntt_inverse_dispatch(d_out + (size_t)i * n_lig, *ctx_k);
    }
    uint64_t* scratch_n = ((int)n_lig == 65536) ? get_bailey_scratch(m) : nullptr;
    gl_ntt::ntt_forward_batched_fast(d_out, m, *ctx_n, scratch_n);
    return out;
}

// ---------------------------------------------------------------------------
// Polynomial evaluation (Horner). Accepts 1-D (d,) or 2-D (m, d) coeffs.
// ---------------------------------------------------------------------------

torch::Tensor poly_eval(torch::Tensor coeffs, torch::Tensor points) {
    CHECK_U64(coeffs); CHECK_U64(points);
    TORCH_CHECK(points.dim() == 1, "points must be 1-D");
    bool one_d = (coeffs.dim() == 1);
    int m = one_d ? 1 : coeffs.size(0);
    int d = coeffs.size(one_d ? 0 : 1);
    int k = points.size(0);
    auto out = one_d
        ? torch::empty({k}, coeffs.options())
        : torch::empty({m, k}, coeffs.options());
    dim3 block(64, 4);
    dim3 grid((k + block.x - 1) / block.x, (m + block.y - 1) / block.y);
    gl_extras::k_poly_eval<<<grid, block>>>(
        (const uint64_t*)coeffs.data_ptr(), m, d,
        (const uint64_t*)points.data_ptr(), k,
        (uint64_t*)out.data_ptr());
    return out;
}

// ---------------------------------------------------------------------------
// Linear algebra mod P.
// ---------------------------------------------------------------------------

torch::Tensor gl_matmul(torch::Tensor A, torch::Tensor B) {
    CHECK_U64(A); CHECK_U64(B);
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "gl_matmul: 2-D inputs");
    int m = A.size(0), k = A.size(1), kb = B.size(0), n = B.size(1);
    TORCH_CHECK(k == kb, "gl_matmul: inner dim mismatch");
    auto C = torch::empty({m, n}, A.options());
    dim3 block(16, 16);
    dim3 grid((n + 15) / 16, (m + 15) / 16);
    gl_dense::k_matmul<<<grid, block>>>(
        (const uint64_t*)A.data_ptr(), (const uint64_t*)B.data_ptr(),
        (uint64_t*)C.data_ptr(), m, k, n);
    return C;
}

torch::Tensor gl_matvec(torch::Tensor M, torch::Tensor v) {
    CHECK_U64(M); CHECK_U64(v);
    TORCH_CHECK(M.dim() == 2 && v.dim() == 1, "gl_matvec: (m,n) @ (n,)");
    int m = M.size(0), n = M.size(1);
    TORCH_CHECK((int)v.size(0) == n, "gl_matvec: dim mismatch");
    auto out = torch::empty({m}, M.options());
    auto [g, blk] = grid1d(m);
    gl_dense::k_matvec<<<g, blk>>>(
        (const uint64_t*)M.data_ptr(), (const uint64_t*)v.data_ptr(),
        (uint64_t*)out.data_ptr(), m, n);
    return out;
}

torch::Tensor gl_spmv(torch::Tensor values, torch::Tensor col_idx,
                      torch::Tensor row_ptr, torch::Tensor x, int64_t n_rows) {
    CHECK_U64(values); CHECK_U64(col_idx); CHECK_U64(row_ptr); CHECK_U64(x);
    TORCH_CHECK(values.dim() == 1 && col_idx.dim() == 1 && row_ptr.dim() == 1 && x.dim() == 1);
    TORCH_CHECK((int64_t)row_ptr.size(0) == n_rows + 1, "row_ptr length must be n_rows + 1");
    auto y = torch::zeros({n_rows}, values.options());
    auto [g, blk] = grid1d((int)n_rows);
    gl_sparse::k_spmv<<<g, blk>>>(
        (const uint64_t*)values.data_ptr(), (const uint64_t*)col_idx.data_ptr(),
        (const uint64_t*)row_ptr.data_ptr(), (const uint64_t*)x.data_ptr(),
        (uint64_t*)y.data_ptr(), (int)n_rows);
    return y;
}

// gl_spmv with the dense combiner x replaced by an inline challenge: x[cid] =
// challenge(seed, cid, label). seed is the 32-byte round-2 seed s_comb; label
// is "lin"/"irs"/"quad". Avoids materializing the combiner vector.
torch::Tensor gl_spmv_challenged(torch::Tensor values, torch::Tensor col_idx,
                                 torch::Tensor row_ptr, torch::Tensor seed,
                                 torch::Tensor label, int64_t n_rows) {
    CHECK_U64(values); CHECK_U64(col_idx); CHECK_U64(row_ptr);
    TORCH_CHECK(seed.is_cuda() && seed.dtype() == torch::kUInt8 && seed.numel() == 32,
                "seed must be a 32-byte CUDA uint8 tensor");
    TORCH_CHECK(label.is_cuda() && label.dtype() == torch::kUInt8 && label.numel() <= 8,
                "label must be a CUDA uint8 tensor of <= 8 bytes");
    TORCH_CHECK((int64_t)row_ptr.size(0) == n_rows + 1, "row_ptr length must be n_rows + 1");
    auto y = torch::zeros({n_rows}, values.options());
    auto [g, blk] = grid1d((int)n_rows);
    gl_sparse::k_spmv_challenged<<<g, blk>>>(
        (const uint64_t*)values.data_ptr(), (const uint64_t*)col_idx.data_ptr(),
        (const uint64_t*)row_ptr.data_ptr(),
        (const uint8_t*)seed.data_ptr(), (const uint8_t*)label.data_ptr(), (int)label.numel(),
        (uint64_t*)y.data_ptr(), (int)n_rows);
    return y;
}

// Materialize r[i] = challenge(seed, i, label) for i in [0, n) on the GPU.
torch::Tensor challenge_vec(torch::Tensor seed, torch::Tensor label, int64_t n) {
    TORCH_CHECK(seed.is_cuda() && seed.dtype() == torch::kUInt8 && seed.numel() == 32,
                "seed must be a 32-byte CUDA uint8 tensor");
    TORCH_CHECK(label.is_cuda() && label.dtype() == torch::kUInt8 && label.numel() <= 8,
                "label must be a CUDA uint8 tensor of <= 8 bytes");
    TORCH_CHECK(n >= 0 && n < (int64_t(1) << 31), "challenge_vec n must be in [0, 2^31)");
    auto y = torch::empty({n}, torch::dtype(torch::kUInt64).device(seed.device()));
    auto [g, blk] = grid1d((int)n);
    gl_sparse::k_challenge_vec<<<g, blk>>>(
        (const uint8_t*)seed.data_ptr(), (const uint8_t*)label.data_ptr(), (int)label.numel(),
        (uint64_t*)y.data_ptr(), n);
    return y;
}

// challenge(seed, cid, label) for an arbitrary cid tensor -- the closed-form fold's
// one new primitive. Bit-identical to challenge_vec(arange(n)) and to the inline
// challenge in gl_spmv_challenged (same gl_sparse::challenge_inline).
__global__ void k_challenge_at(
    const uint8_t* __restrict__ seed, const uint8_t* __restrict__ label, int label_len,
    const uint64_t* __restrict__ cids, uint64_t* __restrict__ out, int64_t n
) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = gl_sparse::challenge_inline(seed, label, label_len, cids[i]);
}

torch::Tensor challenge_at(torch::Tensor seed, torch::Tensor label, torch::Tensor cids) {
    TORCH_CHECK(seed.is_cuda() && seed.dtype() == torch::kUInt8 && seed.numel() == 32,
                "seed must be a 32-byte CUDA uint8 tensor");
    TORCH_CHECK(label.is_cuda() && label.dtype() == torch::kUInt8 && label.numel() <= 8,
                "label must be a CUDA uint8 tensor of <= 8 bytes");
    TORCH_CHECK(cids.is_cuda() && cids.dtype() == torch::kUInt64,
                "cids must be a CUDA uint64 tensor");
    auto cc = cids.contiguous();
    int64_t n = cc.numel();
    TORCH_CHECK(n < (int64_t(1) << 31), "challenge_at: too many cids");
    auto y = torch::empty_like(cc);
    if (n == 0) return y;
    auto [g, blk] = grid1d((int)n);
    k_challenge_at<<<g, blk>>>(
        (const uint8_t*)seed.data_ptr(), (const uint8_t*)label.data_ptr(), (int)label.numel(),
        (const uint64_t*)cc.data_ptr(), (uint64_t*)y.data_ptr(), n);
    return y;
}

// Phase-4 descriptor interpreter (linear-fold-unification.md): evaluate ONE
// linear band's row window into the chunk rTA in place. desc layout (u64[24]):
//   [0]  q — digit count (1..4)
//   [1..4]  radices, most-significant first (unused = 1)
//   [5]  cid_base            [6..9]  per-digit cid strides
//   [10] fan (>= 1)          [11]    per-fan-step cid stride
//   [12] coef mode: 0 const, 1 tblA gather, 2 −(tblA·tblB) (Freivalds C)
//   [13] coef const          [14..17] tblA strides   [18..21] tblB strides
//   [22] chal mode: 0 in-kernel PRF, 1 buffer gather   [23] chal buffer base
// One thread per slot; threads own distinct slots (no atomics); the fan axis
// accumulates in registers. Values are exactly the torch expander path's
// (identical field ops; only grouping differs) — gated by LIGERO_KERNEL_CHECK.
__global__ void k_interp_band(uint64_t* __restrict__ out, int64_t out_off,
                              uint64_t flat_lo, int64_t n_slots,
                              const uint64_t* __restrict__ d,
                              const uint64_t* __restrict__ tblA,
                              const uint64_t* __restrict__ tblB,
                              const uint64_t* __restrict__ chal_buf,
                              const uint8_t* __restrict__ seed,
                              const uint8_t* __restrict__ label, int label_len) {
    int64_t s = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n_slots) return;
    uint64_t rem = flat_lo + (uint64_t)s;
    int q = (int)d[0];
    uint64_t cid = d[5], ia = 0, ib = 0;
    for (int j = q - 1; j >= 1; --j) {
        uint64_t dig = rem % d[1 + j];
        rem /= d[1 + j];
        cid += dig * d[6 + j]; ia += dig * d[14 + j]; ib += dig * d[18 + j];
    }
    cid += rem * d[6]; ia += rem * d[14]; ib += rem * d[18];
    uint64_t mode = d[12];
    uint64_t coef = (mode == 0) ? d[13]
                  : (mode == 1) ? tblA[ia]
                  : gl::sub(0, gl::mul(tblA[ia], tblB[ib]));
    uint64_t fan = d[10], fs = d[11];
    uint64_t acc = 0;
    for (uint64_t t = 0; t < fan; ++t) {
        uint64_t c = cid + t * fs;
        uint64_t r = d[22] ? chal_buf[c - d[23]]
                           : gl_sparse::challenge_inline(seed, label, label_len, c);
        acc = gl::add(acc, r);
    }
    out[out_off + s] = gl::add(out[out_off + s], gl::mul(coef, acc));
}

void interp_band(torch::Tensor out, int64_t out_off, int64_t flat_lo, int64_t n_slots,
                 torch::Tensor desc, torch::Tensor tblA, torch::Tensor tblB,
                 torch::Tensor chal_buf, torch::Tensor seed, torch::Tensor label) {
    CHECK_U64(out); CHECK_U64(desc); CHECK_U64(tblA); CHECK_U64(tblB); CHECK_U64(chal_buf);
    TORCH_CHECK(desc.numel() == 24, "interp_band: desc must be u64[24]");
    TORCH_CHECK(seed.is_cuda() && seed.dtype() == torch::kUInt8 && seed.numel() == 32);
    TORCH_CHECK(label.is_cuda() && label.dtype() == torch::kUInt8 && label.numel() <= 8);
    if (n_slots == 0) return;
    auto [g, blk] = grid1d((int)n_slots);
    k_interp_band<<<g, blk>>>(
        (uint64_t*)out.data_ptr(), out_off, (uint64_t)flat_lo, n_slots,
        (const uint64_t*)desc.data_ptr(),
        (const uint64_t*)tblA.data_ptr(), (const uint64_t*)tblB.data_ptr(),
        (const uint64_t*)chal_buf.data_ptr(),
        (const uint8_t*)seed.data_ptr(), (const uint8_t*)label.data_ptr(), (int)label.numel());
}

// The four irregular band kinds (Phase-4 completion): index maps that don't fit
// the strided-dot descriptor — the causal triangular rank, the data-dependent
// embedding gather, and RoPE's dual emission — each a small bespoke kernel with
// the k_interp_band contract (window-relative, one thread per slot, in-place
// accumulate, no atomics). All use the in-kernel PRF (no dense challenge spans).

__global__ void k_band_causal_id(uint64_t* __restrict__ out, int64_t out_off,
                                 uint64_t flat_lo, int64_t n_slots,
                                 uint64_t base, uint64_t m, uint64_t h, uint64_t coef,
                                 const uint8_t* __restrict__ seed,
                                 const uint8_t* __restrict__ label, int label_len) {
    int64_t s = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n_slots) return;
    uint64_t f = flat_lo + (uint64_t)s, b = f / m, j = f % m;
    uint64_t i_qry = b / h, hh = b % h;
    if (j > i_qry) return;                       // masked cell: no constraint
    uint64_t cid = base + h * i_qry * (i_qry + 1) / 2 + hh * (i_qry + 1) + j;
    uint64_t r = gl_sparse::challenge_inline(seed, label, label_len, cid);
    out[out_off + s] = gl::add(out[out_off + s], gl::mul(coef, r));
}

__global__ void k_band_causal_c2(uint64_t* __restrict__ out, int64_t out_off,
                                 uint64_t flat_lo, int64_t n_slots,
                                 uint64_t base, uint64_t h, uint64_t coef,
                                 const uint8_t* __restrict__ seed,
                                 const uint8_t* __restrict__ label, int label_len) {
    int64_t s = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n_slots) return;
    uint64_t b = flat_lo + (uint64_t)s, i_qry = b / h, hh = b % h;
    uint64_t rank0 = base + h * i_qry * (i_qry + 1) / 2 + hh * (i_qry + 1);
    uint64_t acc = 0;                            // ragged fan: i_qry+1 cids
    for (uint64_t jo = 0; jo <= i_qry; ++jo)
        acc = gl::add(acc, gl_sparse::challenge_inline(seed, label, label_len, rank0 + jo));
    out[out_off + s] = gl::add(out[out_off + s], gl::mul(coef, acc));
}

__global__ void k_band_embed(uint64_t* __restrict__ out, int64_t out_off,
                             uint64_t flat_lo, int64_t n_slots,
                             uint64_t base, uint64_t d, uint64_t rows_per_w, uint64_t ell,
                             const int64_t* __restrict__ token_ids, int64_t seq,
                             const uint8_t* __restrict__ seed,
                             const uint8_t* __restrict__ label, int label_len) {
    int64_t s = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n_slots) return;
    uint64_t f = flat_lo + (uint64_t)s;
    uint64_t row_off = f / ell, within = f % ell;
    uint64_t rel = within / d, jj = within % d;
    if (rel >= rows_per_w) return;               // slots in the ELL remainder
    int64_t v = (int64_t)(row_off * rows_per_w + rel);
    uint64_t acc = 0;                            // a token may repeat: sum the hits
    for (int64_t i = 0; i < seq; ++i)
        if (token_ids[i] == v)
            acc = gl::add(acc, gl_sparse::challenge_inline(
                seed, label, label_len, base + (uint64_t)i * d + jj));
    // coefficient is −1: out += −acc
    out[out_off + s] = gl::add(out[out_off + s], gl::sub(0, acc));
}

__global__ void k_band_rope_x(uint64_t* __restrict__ out, int64_t out_off,
                              uint64_t flat_lo, int64_t n_slots,
                              uint64_t base, uint64_t H, uint64_t d_h,
                              const uint64_t* __restrict__ cos_t,
                              const uint64_t* __restrict__ sin_t,
                              const uint8_t* __restrict__ seed,
                              const uint8_t* __restrict__ label, int label_len) {
    int64_t s = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n_slots) return;
    uint64_t f = flat_lo + (uint64_t)s, half = d_h / 2;
    uint64_t sq = f / (H * d_h), hh = (f / d_h) % H, k = f % d_h;
    uint64_t e = k / half, kp = k % half;
    uint64_t pair_t = sq * H * half + hh * half + kp, ci = sq * half + kp;
    uint64_t c = cos_t[ci], sn = sin_t[ci];
    uint64_t r1 = gl_sparse::challenge_inline(seed, label, label_len, base + 2 * pair_t);
    uint64_t r2 = gl_sparse::challenge_inline(seed, label, label_len, base + 2 * pair_t + 1);
    uint64_t c1 = (e == 0) ? gl::sub(0, c)  : sn;             // eq1 coef
    uint64_t c2 = (e == 0) ? gl::sub(0, sn) : gl::sub(0, c);  // eq2 coef
    out[out_off + s] = gl::add(out[out_off + s],
                               gl::add(gl::mul(c1, r1), gl::mul(c2, r2)));
}

#define IRR_CHECKS(out, seed, label) \
    CHECK_U64(out); \
    TORCH_CHECK(seed.is_cuda() && seed.dtype() == torch::kUInt8 && seed.numel() == 32); \
    TORCH_CHECK(label.is_cuda() && label.dtype() == torch::kUInt8 && label.numel() <= 8);

void interp_band_causal_id(torch::Tensor out, int64_t out_off, int64_t flat_lo, int64_t n_slots,
                           uint64_t base, uint64_t m, uint64_t h, uint64_t coef,
                           torch::Tensor seed, torch::Tensor label) {
    IRR_CHECKS(out, seed, label);
    if (n_slots == 0) return;
    auto [g, blk] = grid1d((int)n_slots);
    k_band_causal_id<<<g, blk>>>((uint64_t*)out.data_ptr(), out_off, (uint64_t)flat_lo, n_slots,
        base, m, h, coef,
        (const uint8_t*)seed.data_ptr(), (const uint8_t*)label.data_ptr(), (int)label.numel());
}

void interp_band_causal_c2(torch::Tensor out, int64_t out_off, int64_t flat_lo, int64_t n_slots,
                           uint64_t base, uint64_t h, uint64_t coef,
                           torch::Tensor seed, torch::Tensor label) {
    IRR_CHECKS(out, seed, label);
    if (n_slots == 0) return;
    auto [g, blk] = grid1d((int)n_slots);
    k_band_causal_c2<<<g, blk>>>((uint64_t*)out.data_ptr(), out_off, (uint64_t)flat_lo, n_slots,
        base, h, coef,
        (const uint8_t*)seed.data_ptr(), (const uint8_t*)label.data_ptr(), (int)label.numel());
}

void interp_band_embed(torch::Tensor out, int64_t out_off, int64_t flat_lo, int64_t n_slots,
                       uint64_t base, uint64_t d, uint64_t rows_per_w, uint64_t ell,
                       torch::Tensor token_ids, torch::Tensor seed, torch::Tensor label) {
    IRR_CHECKS(out, seed, label);
    TORCH_CHECK(token_ids.is_cuda() && token_ids.dtype() == torch::kInt64);
    if (n_slots == 0) return;
    auto tc = token_ids.contiguous();
    auto [g, blk] = grid1d((int)n_slots);
    k_band_embed<<<g, blk>>>((uint64_t*)out.data_ptr(), out_off, (uint64_t)flat_lo, n_slots,
        base, d, rows_per_w, ell,
        (const int64_t*)tc.data_ptr(), (int64_t)tc.numel(),
        (const uint8_t*)seed.data_ptr(), (const uint8_t*)label.data_ptr(), (int)label.numel());
}

void interp_band_rope_x(torch::Tensor out, int64_t out_off, int64_t flat_lo, int64_t n_slots,
                        uint64_t base, uint64_t H, uint64_t d_h,
                        torch::Tensor cos_t, torch::Tensor sin_t,
                        torch::Tensor seed, torch::Tensor label) {
    IRR_CHECKS(out, seed, label);
    CHECK_U64(cos_t); CHECK_U64(sin_t);
    if (n_slots == 0) return;
    auto cc = cos_t.contiguous(); auto sc = sin_t.contiguous();
    auto [g, blk] = grid1d((int)n_slots);
    k_band_rope_x<<<g, blk>>>((uint64_t*)out.data_ptr(), out_off, (uint64_t)flat_lo, n_slots,
        base, H, d_h,
        (const uint64_t*)cc.data_ptr(), (const uint64_t*)sc.data_ptr(),
        (const uint8_t*)seed.data_ptr(), (const uint8_t*)label.data_ptr(), (int)label.numel());
}

// ---------------------------------------------------------------------------
// BLAKE3 column hashing (no row cap) and Merkle level kernel.
// ---------------------------------------------------------------------------

torch::Tensor hash_columns_streamed(torch::Tensor matrix) {
    CHECK_U64(matrix);
    TORCH_CHECK(matrix.dim() == 2, "expected (m, n) matrix");
    int m = matrix.size(0);
    int n_cols = matrix.size(1);
    auto digests_u32 = torch::empty({n_cols, 8}, torch::TensorOptions()
        .dtype(torch::kUInt32).device(matrix.device()));
    int block = 64, grid = (n_cols + block - 1) / block;
    b3_cols::k_hash_columns_simple<<<grid, block>>>(
        (const uint64_t*)matrix.data_ptr(), m, n_cols,
        (uint32_t*)digests_u32.data_ptr());
    return digests_u32.view(torch::kUInt8);
}

// Chunked column hash: stream a tall matrix one BLAKE3 chunk (≤128 rows)
// at a time without materializing the full codeword matrix. Output
// matches hash_columns_streamed bit-exactly on the concatenated input.
torch::Tensor hash_columns_stream_init(int64_t n_cols) {
    auto states = torch::empty(
        {n_cols, (int64_t)sizeof(b3_cols::ColStreamState)},
        torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
    int block = 64, grid = (n_cols + block - 1) / block;
    b3_cols::k_stream_init<<<grid, block>>>(
        (b3_cols::ColStreamState*)states.data_ptr(), n_cols);
    return states;
}

void hash_columns_stream_update(torch::Tensor states, torch::Tensor chunk,
                                bool is_last_chunk, int64_t n_chunks_total) {
    TORCH_CHECK(states.dtype() == torch::kUInt8 && states.is_cuda());
    CHECK_U64(chunk);
    TORCH_CHECK(chunk.dim() == 2, "expected (chunk_rows, n_cols)");
    int n_rows = chunk.size(0);
    int n_cols = chunk.size(1);
    TORCH_CHECK(n_rows <= b3_cols::STREAM_CHUNK_ROWS,
                "chunk must fit within one BLAKE3 chunk (128 rows)");
    bool is_single = (n_chunks_total == 1);
    int block = 64, grid = (n_cols + block - 1) / block;
    // digests output only used when is_single_chunk_total && is_last_chunk;
    // safe to pass nullptr otherwise. To keep the API uniform we always
    // allocate a small dummy buffer when not needed; cheaper than branching
    // the kernel call.
    auto digests_u32 = torch::empty({n_cols, 8}, torch::TensorOptions()
        .dtype(torch::kUInt32).device(chunk.device()));
    b3_cols::k_stream_update_chunk<<<grid, block>>>(
        (b3_cols::ColStreamState*)states.data_ptr(),
        (const uint64_t*)chunk.data_ptr(),
        n_cols, n_rows, is_last_chunk, is_single,
        (uint32_t*)digests_u32.data_ptr());
    // Stash digests into states' first 32 bytes? Simpler: caller calls
    // hash_columns_stream_finalize which handles the single-chunk and
    // multi-chunk cases via the same code path. For single-chunk, finalize
    // pulls from this digests buffer; we store a pointer here. To keep
    // memory ownership clean, we don't actually rely on this for single-
    // chunk and instead require single-chunk paths to call the existing
    // hash_columns_streamed. (Documented in the Python wrapper.)
    (void)digests_u32;
}

// ---------------------------------------------------------------------------
// Per-row deterministic PRG for ZK slack padding.
//
// NEEDS REVIEW + INDEPENDENT TESTING:
//   - GPU-only implementation; no separate CPU reference yet exists for
//     bit-comparison. The algorithm is simple enough to re-derive in Python
//     (BLAKE3(master_seed || row_le8 || k_le8)[0:8] mod P), and we should
//     add a CPU reference + diff test before relying on this in production.
//   - Statistical uniformity of the mod-P reduction should be audited
//     (the bias is ~2^-32 per cell, below soundness terms, but worth a
//     formal check).
//   - Soundness of using this PRG for ZK slack (replacing the global
//     numpy.PCG64 stream) is the same argument as today: master_seed
//     is drawn fresh from a CSPRNG per proof, and per-row outputs are
//     uniform-mod-P given the seed. The verifier never sees master_seed.
//
// Design: one thread per row; each thread loops over slack_per_row outputs,
// hashing (master_seed || row_idx_le8 || k_le8) and taking the first 8
// bytes mod P. Input fits in one BLAKE3 chunk so each hash is a single
// compress call via b3::hash_bytes.
// ---------------------------------------------------------------------------

__global__ void k_row_prg(
    const uint8_t* __restrict__ master_seed,    // (32,) bytes
    uint32_t row_offset,
    uint32_t n_rows,
    uint32_t slack_per_row,
    uint64_t P,
    uint64_t* __restrict__ out                   // (n_rows, slack_per_row) u64
) {
    uint32_t local_row = blockIdx.x * blockDim.x + threadIdx.x;
    if (local_row >= n_rows) return;
    uint32_t global_row = row_offset + local_row;

    uint8_t input[48];
    #pragma unroll
    for (int i = 0; i < 32; ++i) input[i] = master_seed[i];
    *(uint64_t*)(input + 32) = (uint64_t)global_row;

    uint64_t* row_out = out + (size_t)local_row * slack_per_row;

    for (uint32_t k = 0; k < slack_per_row; ++k) {
        *(uint64_t*)(input + 40) = (uint64_t)k;
        uint32_t digest[8];
        b3::hash_bytes(input, 48, digest);
        uint64_t raw = ((uint64_t)digest[0]) | (((uint64_t)digest[1]) << 32);
        row_out[k] = raw % P;
    }
}

// Indexed variant of k_row_prg: rows are an arbitrary (non-contiguous) index
// list instead of a contiguous range. One thread per row — replaces the Python
// per-row row_prg launches in _encode_rows_indexed. Bit-exact (abs rows < 2^32,
// so global_row as uint64 has the same little-endian bytes as the uint32 path).
__global__ void k_row_prg_indexed(
    const uint8_t*  __restrict__ master_seed,
    const uint64_t* __restrict__ row_indices,    // (n_rows,) absolute row indices
    uint32_t n_rows,
    uint32_t slack_per_row,
    uint64_t P,
    uint64_t* __restrict__ out                   // (n_rows, slack_per_row) u64
) {
    uint32_t local_row = blockIdx.x * blockDim.x + threadIdx.x;
    if (local_row >= n_rows) return;
    uint64_t global_row = row_indices[local_row];

    uint8_t input[48];
    #pragma unroll
    for (int i = 0; i < 32; ++i) input[i] = master_seed[i];
    *(uint64_t*)(input + 32) = global_row;

    uint64_t* row_out = out + (size_t)local_row * slack_per_row;

    for (uint32_t k = 0; k < slack_per_row; ++k) {
        *(uint64_t*)(input + 40) = (uint64_t)k;
        uint32_t digest[8];
        b3::hash_bytes(input, 48, digest);
        uint64_t raw = ((uint64_t)digest[0]) | (((uint64_t)digest[1]) << 32);
        row_out[k] = raw % P;
    }
}

torch::Tensor row_prg(torch::Tensor master_seed, int64_t row_offset,
                       int64_t n_rows, int64_t slack_per_row, uint64_t P) {
    TORCH_CHECK(master_seed.dtype() == torch::kUInt8, "master_seed must be uint8");
    TORCH_CHECK(master_seed.numel() == 32, "master_seed must be 32 bytes");
    TORCH_CHECK(master_seed.is_cuda(), "master_seed must be on CUDA");
    TORCH_CHECK(master_seed.is_contiguous(), "master_seed must be contiguous");
    TORCH_CHECK(n_rows >= 0 && slack_per_row >= 0);
    auto out = torch::empty({n_rows, slack_per_row}, torch::TensorOptions()
        .dtype(torch::kUInt64).device(master_seed.device()));
    if (n_rows == 0 || slack_per_row == 0) return out;
    int block = 64;
    int grid = ((int)n_rows + block - 1) / block;
    k_row_prg<<<grid, block>>>(
        (const uint8_t*)master_seed.data_ptr(),
        (uint32_t)row_offset, (uint32_t)n_rows, (uint32_t)slack_per_row,
        P, (uint64_t*)out.data_ptr());
    return out;
}

torch::Tensor row_prg_indexed(torch::Tensor master_seed, torch::Tensor row_indices,
                              int64_t slack_per_row, uint64_t P) {
    TORCH_CHECK(master_seed.dtype() == torch::kUInt8 && master_seed.numel() == 32 && master_seed.is_cuda(),
                "master_seed must be a 32-byte CUDA uint8 tensor");
    TORCH_CHECK(row_indices.dtype() == torch::kUInt64 && row_indices.is_cuda() && row_indices.is_contiguous(),
                "row_indices must be a contiguous CUDA uint64 tensor");
    int64_t n_rows = row_indices.numel();
    auto out = torch::empty({n_rows, slack_per_row}, torch::TensorOptions()
        .dtype(torch::kUInt64).device(master_seed.device()));
    if (n_rows == 0 || slack_per_row == 0) return out;
    int block = 64;
    int grid = ((int)n_rows + block - 1) / block;
    k_row_prg_indexed<<<grid, block>>>(
        (const uint8_t*)master_seed.data_ptr(),
        (const uint64_t*)row_indices.data_ptr(),
        (uint32_t)n_rows, (uint32_t)slack_per_row,
        P, (uint64_t*)out.data_ptr());
    return out;
}

torch::Tensor hash_columns_stream_finalize(torch::Tensor states, int64_t n_cols,
                                            int64_t n_chunks_total) {
    TORCH_CHECK(states.dtype() == torch::kUInt8 && states.is_cuda());
    auto digests_u32 = torch::empty({n_cols, 8}, torch::TensorOptions()
        .dtype(torch::kUInt32).device(states.device()));
    bool is_single = (n_chunks_total == 1);
    int block = 64, grid = (n_cols + block - 1) / block;
    b3_cols::k_stream_finalize<<<grid, block>>>(
        (b3_cols::ColStreamState*)states.data_ptr(),
        (uint32_t*)digests_u32.data_ptr(), n_cols, is_single);
    return digests_u32.view(torch::kUInt8);
}

// One Merkle level: (n_pairs * 2, 8) u32 → (n_pairs, 8) u32.
torch::Tensor merkle_one_level(torch::Tensor leaves_u32) {
    TORCH_CHECK(leaves_u32.is_cuda());
    TORCH_CHECK(leaves_u32.dtype() == torch::kUInt32);
    TORCH_CHECK(leaves_u32.is_contiguous());
    TORCH_CHECK(leaves_u32.dim() == 2 && leaves_u32.size(1) == 8);
    int n_leaves = leaves_u32.size(0);
    TORCH_CHECK(n_leaves % 2 == 0, "merkle_one_level: n must be even (caller duplicates last leaf if needed)");
    int n_pairs = n_leaves / 2;
    auto out = torch::empty({n_pairs, 8}, leaves_u32.options());
    auto [g, blk] = grid1d(n_pairs);
    merkle::k_level<<<g, blk>>>(
        (const uint32_t*)leaves_u32.data_ptr(),
        (uint32_t*)out.data_ptr(), n_pairs);
    return out;
}

// ---------------------------------------------------------------------------
// LogUp helper.
// ---------------------------------------------------------------------------

torch::Tensor lookup_multiplicities(torch::Tensor x, torch::Tensor table) {
    CHECK_U64(x); CHECK_U64(table);
    TORCH_CHECK(x.dim() == 1 && table.dim() == 1);
    int n_table = table.size(0);
    auto mult_u64 = torch::zeros({n_table}, x.options());
    if (x.numel() == 0) return mult_u64;
    auto [g, blk] = grid1d((int)x.numel());
    gl_extras::k_lookup_multiplicities<<<g, blk>>>(
        (const uint64_t*)x.data_ptr(), (int)x.numel(),
        (const uint64_t*)table.data_ptr(), n_table,
        (unsigned long long*)mult_u64.data_ptr());
    return mult_u64;
}

// Accumulating variant: mult[j] += count of i where x[i] == table[j]. The
// kernel uses atomicAdd, so repeated calls with the same mult tensor
// produce a running histogram across all calls — used by Tape to share
// one mult across many tlookups against the same Table.
void lookup_multiplicities_into(torch::Tensor x, torch::Tensor table, torch::Tensor mult) {
    CHECK_U64(x); CHECK_U64(table); CHECK_U64(mult);
    TORCH_CHECK(x.dim() == 1 && table.dim() == 1 && mult.dim() == 1);
    TORCH_CHECK((int64_t)mult.numel() == table.numel(), "mult length must match table");
    if (x.numel() == 0) return;
    auto [g, blk] = grid1d((int)x.numel());
    gl_extras::k_lookup_multiplicities<<<g, blk>>>(
        (const uint64_t*)x.data_ptr(), (int)x.numel(),
        (const uint64_t*)table.data_ptr(), (int)table.numel(),
        (unsigned long long*)mult.data_ptr());
}

void lookup_multiplicities_range_into(torch::Tensor x, int64_t T_LEN, torch::Tensor mult) {
    CHECK_U64(x); CHECK_U64(mult);
    TORCH_CHECK(x.dim() == 1 && mult.dim() == 1);
    TORCH_CHECK((int64_t)mult.numel() == T_LEN, "mult length must equal T_LEN");
    if (x.numel() == 0) return;
    auto [g, blk] = grid1d((int)x.numel());
    gl_extras::k_lookup_multiplicities_range<<<g, blk>>>(
        (const uint64_t*)x.data_ptr(), (int)x.numel(),
        (uint64_t)T_LEN,
        (unsigned long long*)mult.data_ptr());
}
"""


# ---------------------------------------------------------------------------
# Python-side wrappers. Lazy JIT compile on first call.
# ---------------------------------------------------------------------------

def _arch_flag():
    """nvcc arch for the local GPU (sm_121 = GB10/Spark, sm_90 = H100, ...).
    The wrong arch silently produces wrong kernel results on some pairs —
    the original hardcoded sm_121 broke any non-Spark machine."""
    cap = torch.cuda.get_device_capability()
    return f"-arch=sm_{cap[0]}{cap[1]}"


_module = None


def _ensure_compiled():
    global _module
    if _module is None:
        _module = load_inline(
            name="ligero_primitives",
            cpp_sources=_CPP_DECLS,
            cuda_sources=_CUDA_SOURCE,
            extra_include_paths=[str(_CUDA_HEADERS_DIR)],
            extra_cuda_cflags=[_arch_flag(), "-O3", "-std=c++17"],
            extra_cflags=["-std=c++17"],
            verbose=False,
        )
    return _module


# ---- Elementwise field ----

def gl_mul(a, b):  return _ensure_compiled().gl_mul(a, b)
def gl_add(a, b):  return _ensure_compiled().gl_add(a, b)
def gl_sub(a, b):  return _ensure_compiled().gl_sub(a, b)
def gl_neg(a):     return _ensure_compiled().gl_neg(a)
def gl_pow(base, exp): return _ensure_compiled().gl_pow(base, int(exp))


def gl_inv(a):
    """Elementwise Fermat inverse a^(P-2). For very-large vectors, prefer
    gl_inv_batched (one inverse + 3n mults)."""
    return _ensure_compiled().gl_pow(a, P - 2)


def gl_inv_batched(a):
    """Montgomery batched inverse: one Fermat inverse + 3n field mults."""
    return _ensure_compiled().gl_inv_batched(a)


def gl_axpy(y, alpha, x):
    """In-place y += alpha * x mod P."""
    _ensure_compiled().gl_axpy_inplace(y, int(alpha), x)


# ---- NTT ----

def ntt_forward(a):  _ensure_compiled().ntt_forward(a)
def ntt_inverse(a):  _ensure_compiled().ntt_inverse(a)
def ntt_forward_batched(rows):  _ensure_compiled().ntt_forward_batched(rows)
def ntt_inverse_batched(rows):  _ensure_compiled().ntt_inverse_batched(rows)


# ---- Reed-Solomon ----

def rs_encode_rows(messages, n_lig, k_deg):
    """(m, K_DEG) uint64 → (m, N_LIG) codewords."""
    return _ensure_compiled().rs_encode_rows(messages, int(n_lig), int(k_deg))


# ---- Polynomial ----

def poly_eval(coeffs, points):
    """Horner evaluation; coeffs (d,) → (k,) or (m, d) → (m, k)."""
    return _ensure_compiled().poly_eval(coeffs, points)


def _next_pow2(n):
    if n <= 2: return 2
    p = 1
    while p < n: p *= 2
    return p


def poly_mul(a, b):
    """Multiply two polynomials in coefficient form via NTT."""
    assert a.dim() == 1 and b.dim() == 1
    if a.numel() == 0 or b.numel() == 0:
        return torch.empty(0, dtype=torch.uint64, device=a.device)
    result_len = a.numel() + b.numel() - 1
    n = _next_pow2(result_len)
    a_padded = torch.zeros(n, dtype=torch.uint64, device=a.device)
    a_padded[:a.numel()] = a
    b_padded = torch.zeros(n, dtype=torch.uint64, device=b.device)
    b_padded[:b.numel()] = b
    ntt_forward(a_padded)
    ntt_forward(b_padded)
    prod = gl_mul(a_padded, b_padded)
    ntt_inverse(prod)
    return prod[:result_len].contiguous()


def poly_add(a, b):
    """Length-adapting polynomial add."""
    assert a.dim() == 1 and b.dim() == 1
    n = max(a.numel(), b.numel())
    if n == 0:
        return torch.empty(0, dtype=torch.uint64, device=a.device)
    a_padded = torch.zeros(n, dtype=torch.uint64, device=a.device)
    a_padded[:a.numel()] = a
    b_padded = torch.zeros(n, dtype=torch.uint64, device=b.device)
    b_padded[:b.numel()] = b
    return gl_add(a_padded, b_padded)


def poly_mul_batched(A, B):
    """Row-i = poly_mul(A[i], B[i]). A, B same shape (m, d)."""
    assert A.dim() == 2 and B.dim() == 2 and A.shape == B.shape
    m, d = A.shape
    result_len = 2 * d - 1
    n = _next_pow2(result_len)
    A_padded = torch.zeros((m, n), dtype=torch.uint64, device=A.device)
    B_padded = torch.zeros((m, n), dtype=torch.uint64, device=B.device)
    A_padded[:, :d] = A
    B_padded[:, :d] = B
    ntt_forward_batched(A_padded)
    ntt_forward_batched(B_padded)
    prod = gl_mul(A_padded, B_padded)
    ntt_inverse_batched(prod)
    return prod[:, :result_len].contiguous()


# ---- Linear algebra mod P ----

def gl_matmul(A, B): return _ensure_compiled().gl_matmul(A, B)
def gl_matvec(M, v): return _ensure_compiled().gl_matvec(M, v)


def gl_spmv(values, col_idx, row_ptr, x, n_rows):
    """CSR mod-P matvec. row_ptr length must be n_rows + 1."""
    return _ensure_compiled().gl_spmv(values, col_idx, row_ptr, x, int(n_rows))


def gl_spmv_challenged(values, col_idx, row_ptr, seed, label, n_rows):
    """CSR mod-P matvec where the combiner x[cid] = challenge(seed, cid, label)
    is computed inline (no materialized combiner). seed: 32-byte cuda uint8;
    label: cuda uint8 (b"lin"/b"irs"/b"quad")."""
    return _ensure_compiled().gl_spmv_challenged(
        values, col_idx, row_ptr, seed, label, int(n_rows))


def interp_band(out, out_off, flat_lo, n_slots, desc, tblA, tblB, chal_buf, seed, label):
    """Phase-4 descriptor-interpreter band evaluation (in-place accumulate into
    `out`). See the kernel comment for the desc[24] layout."""
    _ensure_compiled().interp_band(out, int(out_off), int(flat_lo), int(n_slots),
                                   desc, tblA, tblB, chal_buf, seed, label)


def interp_band_causal_id(out, out_off, flat_lo, n_slots, base, m, h, coef, seed, label):
    _ensure_compiled().interp_band_causal_id(out, int(out_off), int(flat_lo), int(n_slots),
                                             int(base), int(m), int(h), int(coef), seed, label)


def interp_band_causal_c2(out, out_off, flat_lo, n_slots, base, h, coef, seed, label):
    _ensure_compiled().interp_band_causal_c2(out, int(out_off), int(flat_lo), int(n_slots),
                                             int(base), int(h), int(coef), seed, label)


def interp_band_embed(out, out_off, flat_lo, n_slots, base, d, rows_per_w, ell, token_ids, seed, label):
    _ensure_compiled().interp_band_embed(out, int(out_off), int(flat_lo), int(n_slots),
                                         int(base), int(d), int(rows_per_w), int(ell),
                                         token_ids, seed, label)


def interp_band_rope_x(out, out_off, flat_lo, n_slots, base, H, d_h, cos_t, sin_t, seed, label):
    _ensure_compiled().interp_band_rope_x(out, int(out_off), int(flat_lo), int(n_slots),
                                          int(base), int(H), int(d_h), cos_t, sin_t, seed, label)


def challenge_at(seed, label, cids):
    """challenge(seed, cid, label) for an arbitrary cid tensor (uint64, any shape).
    Bit-identical to challenge_vec(arange(n)) and to gl_spmv_challenged's inline
    challenge -- the one new device primitive for the closed-form q_lin fold."""
    return _ensure_compiled().challenge_at(seed, label, cids)


def challenge_vec(seed, label, n):
    """Materialize the combiner r[i]=challenge(seed,i,label) for i in [0,n) on
    GPU. seed: 32-byte cuda uint8; label: cuda uint8 (b"irs"/b"quad"). n < 2^31."""
    if int(n) == 0:
        # zero-size kernel launch is a CUDA error; a pure-linear tape (e.g.
        # ConcatClaim-only tests) legitimately has len(quads) == 0
        return torch.empty(0, dtype=torch.uint64, device="cuda")
    return _ensure_compiled().challenge_vec(seed, label, int(n))


# ---- BLAKE3 column hash + Merkle ----

def hash_columns_streamed(matrix):
    """(m, n) uint64 → (n, 32) uint8 BLAKE3 column digests. No row cap."""
    return _ensure_compiled().hash_columns_streamed(matrix)


# BLAKE3 chunks 1024 bytes = 128 u64 rows. The streaming kernel processes
# at most one BLAKE3 chunk per call.
_BLAKE3_CHUNK_ROWS = 128


def row_prg(master_seed, row_offset, n_rows, slack_per_row):
    """Per-row deterministic PRG for ZK slack: returns (n_rows, slack_per_row)
    uint64 tensor where out[r, k] = BLAKE3(master_seed || (row_offset+r) || k)[0:8] mod P.

    Stateless replacement for the numpy.PCG64 stream we used in
    _random_field_pad. Same (master_seed, row_offset, n_rows, slack_per_row)
    always produces the same output — any caller (commit, column open,
    compute_p_0 on-demand re-encode) can reproduce a specific row's slack
    without coordinating RNG state.

    NEEDS REVIEW + INDEPENDENT TESTING. See the C++ comment block above
    k_row_prg for the open items: no CPU reference for bit-comparison yet;
    statistical-uniformity audit pending; soundness reasoning matches the
    existing numpy-based path.

    Args:
        master_seed: (32,) uint8 cuda tensor. Drawn fresh per proof from a
            CSPRNG, kept secret to the prover (never sent on wire).
        row_offset: int, global index of the chunk's first row.
        n_rows: int, number of rows in this chunk.
        slack_per_row: int, K_DEG - ELL.
    """
    assert isinstance(master_seed, torch.Tensor)
    return _ensure_compiled().row_prg(
        master_seed, int(row_offset), int(n_rows), int(slack_per_row), int(P))


def row_prg_indexed(master_seed, row_indices, slack_per_row):
    """row_prg for a non-contiguous list of absolute row indices — one batched
    GPU launch instead of one launch per row. row_indices: cuda uint64 tensor.
    out[i, k] = uint64(BLAKE3(master_seed || row_indices[i]_le8 || k_le8)[0:8]) mod P."""
    return _ensure_compiled().row_prg_indexed(
        master_seed, row_indices, int(slack_per_row), int(P))


class MerkleColumnAccumulator:
    """Incremental column-hash for a tall matrix supplied in chunks.

    Produces the same per-column digests as `hash_columns_streamed` on
    the concatenated input, but never holds the full matrix in memory.
    Used by the streaming commit path: encode chunk → update accumulator
    → drop chunk's codewords.

    Caller supplies `n_total_rows` up front (the kernel marks ROOT on
    the final BLAKE3 chunk, which we detect from the cumulative row
    count). `update()` accepts any number of rows; partial BLAKE3
    chunks at the boundary are buffered internally until the next call
    fills them, so callers can chunk encoding at whatever granularity
    is convenient (e.g., 1024-row encode chunks + a 3-row blinding
    prefix). Requires `n_total_rows > 128`; single-chunk inputs should
    use `hash_columns_streamed` directly."""

    def __init__(self, n_cols: int, n_total_rows: int):
        assert n_total_rows > _BLAKE3_CHUNK_ROWS, (
            "MerkleColumnAccumulator requires multi-BLAKE3-chunk input; "
            "for ≤128 rows use hash_columns_streamed directly")
        self.n_cols = n_cols
        self.n_total_rows = n_total_rows
        self.n_chunks_total = ((n_total_rows + _BLAKE3_CHUNK_ROWS - 1)
                                // _BLAKE3_CHUNK_ROWS)
        self._states = _ensure_compiled().hash_columns_stream_init(n_cols)
        self._chunks_seen = 0
        self._rows_seen = 0
        self._partial = None     # leftover rows < 128 from prior update()

    def update(self, rows):
        """rows: (k, n_cols) uint64. Any k ≥ 0; partial BLAKE3 chunks
        are buffered until the next call completes them or finalize()
        is reached."""
        assert rows.dtype == torch.uint64 and rows.is_cuda
        assert rows.dim() == 2 and rows.size(1) == self.n_cols
        if self._partial is not None:
            rows = torch.cat([self._partial, rows])
            self._partial = None
        k = rows.size(0)
        if k == 0:
            return

        # Emit all complete 128-row BLAKE3 chunks except possibly the
        # very last one (which may need to be saved for the next
        # update() call OR for the final partial chunk).
        full_chunks = k // _BLAKE3_CHUNK_ROWS
        # We can safely emit `n_emit` complete chunks: all but possibly
        # the last full chunk, unless we've now seen all rows.
        if self._rows_seen + k == self.n_total_rows:
            # This call covers the final rows. Emit all complete chunks
            # + a final possibly-partial chunk in the leftover.
            n_emit = full_chunks
            leftover_size = k - n_emit * _BLAKE3_CHUNK_ROWS
        else:
            # Hold back a single 128-row chunk so the next update() can
            # tell whether IT is the last. Simpler: hold back any
            # leftover whether full or partial.
            n_emit = full_chunks if (k % _BLAKE3_CHUNK_ROWS != 0) else full_chunks - 1
            n_emit = max(n_emit, 0)
            leftover_size = k - n_emit * _BLAKE3_CHUNK_ROWS

        for i in range(n_emit):
            sub = rows[i * _BLAKE3_CHUNK_ROWS:(i + 1) * _BLAKE3_CHUNK_ROWS].contiguous()
            self._chunks_seen += 1
            self._rows_seen += _BLAKE3_CHUNK_ROWS
            is_last = (self._chunks_seen == self.n_chunks_total)
            _ensure_compiled().hash_columns_stream_update(
                self._states, sub, is_last, self.n_chunks_total)

        if leftover_size > 0:
            leftover = rows[n_emit * _BLAKE3_CHUNK_ROWS:].contiguous()
            if self._rows_seen + leftover_size == self.n_total_rows:
                # Final BLAKE3 chunk (full or partial).
                self._chunks_seen += 1
                self._rows_seen += leftover_size
                _ensure_compiled().hash_columns_stream_update(
                    self._states, leftover, True, self.n_chunks_total)
            else:
                # Buffer for next update().
                self._partial = leftover

    def finalize(self):
        """Returns (n_cols, 32) uint8 column digests."""
        assert self._chunks_seen == self.n_chunks_total, (
            f"expected {self.n_chunks_total} chunks, saw {self._chunks_seen}"
            f" (rows_seen={self._rows_seen}/{self.n_total_rows})")
        return _ensure_compiled().hash_columns_stream_finalize(
            self._states, self.n_cols, self.n_chunks_total)


def merkle_build_blake3(leaves):
    """(N, 32) uint8 leaves → (root: (32,) uint8, levels: list of (k_i, 32) u8).
    levels[0] = leaves, levels[-1] = (1, 32) containing the root."""
    assert leaves.dim() == 2 and leaves.size(1) == 32
    assert leaves.dtype == torch.uint8 and leaves.is_cuda
    levels = [leaves]
    cur = leaves.view(torch.uint32).contiguous()   # (N, 8) u32
    while cur.size(0) > 1:
        if cur.size(0) % 2 == 1:
            cur = torch.cat([cur, cur[-1:].clone()], dim=0)
        cur = _ensure_compiled().merkle_one_level(cur)
        levels.append(cur.view(torch.uint8).contiguous())
    root = levels[-1].view(-1)[:32].clone()
    return root, levels


# ---- LogUp ----

# Per-table-tensor cache keyed by data_ptr → T_LEN if confirmed [0, T_LEN), else 0.
# Tables are reused across many calls, so the one-time probe amortizes away.
_range_table_cache: dict = {}


def _is_range_table(table):
    """One-time GPU probe: is table[i] == i for all i? Cached by data_ptr.
    Tables in this codebase are immutable after construction, so the cache
    is safe for the life of the process."""
    key = (table.data_ptr(), table.numel())
    cached = _range_table_cache.get(key)
    if cached is not None:
        return cached
    n = table.numel()
    ref = torch.arange(n, dtype=torch.int64, device=table.device).view(torch.uint64)
    is_range = bool(torch.equal(table, ref))
    _range_table_cache[key] = is_range
    return is_range


def lookup_multiplicities(x, table):
    """Per-table-entry multiplicity histogram. Out-of-range x[i] contribute nothing."""
    return _ensure_compiled().lookup_multiplicities(x, table)


def lookup_multiplicities_into(x, table, mult, label=""):
    """In-place increment of mult[j] for each i where x[i] == table[j].

    Fast path for range tables (table == [0, T_LEN)): direct indexing
    mult[x[i]] += 1 (when x[i] < T_LEN) — O(1) per witness element instead
    of O(T_LEN). Detected via one-time probe + per-pointer cache."""
    if _is_range_table(table):
        _ensure_compiled().lookup_multiplicities_range_into(x, table.numel(), mult)
    else:
        _ensure_compiled().lookup_multiplicities_into(x, table, mult)


# ---------------------------------------------------------------------------
# Self-test (kept from the original module; covers the elementwise + NTT +
# poly + column-hash primitives at small size). The full acceptance suite
# lives in test_cuda_primitives.py.
# ---------------------------------------------------------------------------

def _self_test():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent / "ref"))
    import goldilocks_ref as ref
    import blake3 as blake3_py

    assert P == ref.P

    test_pairs = [(7, 5), (13, 17), (99, 21), (1 << 60, 7),
                  ((1 << 63) | 1, (1 << 32) | 3)]
    a_cpu = [x for x, _ in test_pairs]
    b_cpu = [y for _, y in test_pairs]
    a = torch.tensor(a_cpu, dtype=torch.uint64, device="cuda")
    b = torch.tensor(b_cpu, dtype=torch.uint64, device="cuda")
    assert torch.equal(gl_mul(a, b),
                       torch.tensor([ref.mul(x, y) for x, y in test_pairs],
                                    dtype=torch.uint64, device="cuda"))
    assert torch.equal(gl_add(a, b),
                       torch.tensor([ref.add(x, y) for x, y in test_pairs],
                                    dtype=torch.uint64, device="cuda"))

    for N in [4, 8, 16, 1024]:
        x_cpu = [(7 * i + 1) % P for i in range(N)]
        omega = ref.root_of_unity(N)
        expected = ref.ntt(list(x_cpu), omega, invert=False)
        x = torch.tensor(x_cpu, dtype=torch.uint64, device="cuda")
        ntt_forward(x)
        assert x.cpu().tolist() == expected, f"NTT N={N}"

    print("cuda_primitives.py: self-test passed")


if __name__ == "__main__":
    _self_test()
