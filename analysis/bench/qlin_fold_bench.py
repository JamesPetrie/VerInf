"""Microbenchmark harness for the q_lin build-r_eval (chunk_rTA) step.

Goal: make per-slot / per-row / multi-packet-per-thread kernel-mapping
decisions for the constraint-family fold by *benchmark*, not by intuition.

Why this is cheap to bench (see analysis/qlin-family-object-reorg.md):
  - chunk-local: build_rTA runs on one (n_chunk x ELL) chunk at a time, so a
    variant is timed on a single representative chunk, not the full prove;
  - witness-independent: chunk_rTA depends only on the constraint structure and
    the challenges, never on activations, so we can synthesize a chunk and
    replay it against every variant offline;
  - bit-exact oracle: the current expand -> argsort -> CSR -> gl_spmv_challenged
    path is the reference; every variant must reproduce its (n_chunk x ELL)
    tensor exactly. Correctness is one `torch.equal`.

This first cut isolates the Freivalds LF1B family (the B-side of a matmul,
i.e. the weight matrices that dominate the witness). The reference and the
variants both produce a *pure-Freivalds* chunk, so the comparison is
apples-to-apples on the family that matters most.

Run on a GPU box (e.g. the Spark):
    ~/venv-hf/bin/python qlin_fold_bench.py --variant all
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import torch

from cuda_primitives import P, gl_spmv_challenged
from packets import (
    L2_FreivaldsLF1B, _expand_freivalds_lf1b,
    L2_IdentityScalar, _expand_identity_scalar,
    L2_StrideOneToManyScalar, _expand_stride_one_to_many,
    L2_FreivaldsLF2A, _expand_freivalds_lf2a,
    L2_FreivaldsLF3C, _expand_freivalds_lf3c,
    L2_StrideManyToOneScalar, _expand_stride_many_to_one,
    L2_PerSlotVector, _expand_per_slot_vector,
    L2_RowSumPerSlotVector, _expand_row_sum_per_slot_vector,
    L2_RoPEXRot, _expand_rope_xrot,
    L2_TransposeO2MScalar, _expand_transpose_o2m,
    L2_CausalFilteredIdScalar, _expand_causal_filtered_id,
    L2_CausalFilteredC2Stride, _expand_causal_filtered_c2,
    L2_RoPEX, _expand_rope_x,
)

ELL_DEFAULT = 8192
_CUDA_DIR = Path(__file__).resolve().parent.parent / "cuda"


def _arch_flag():
    """nvcc arch for the local GPU (sm_121 = GB10/Spark). Wrong arch silently
    produces wrong results, so match the device. (Inlined to avoid depending on
    a particular cuda_primitives version.)"""
    cap = torch.cuda.get_device_capability()
    return f"-arch=sm_{cap[0]}{cap[1]}"

# ---------------------------------------------------------------------------
# Variant CUDA kernels (separate inline module — does NOT touch the production
# build in cuda_primitives.py). They reuse gl_sparse::challenge_inline from
# gl_spmv.cuh and gl::mul from goldilocks.cuh, so the challenge math is
# bit-identical to the reference path.
# ---------------------------------------------------------------------------

_CPP_DECLS = r"""
torch::Tensor lf1b_perslot(torch::Tensor seed, torch::Tensor label,
    torch::Tensor neg_rho, int64_t base, int64_t B_row_start, int64_t chunk_lo,
    int64_t k, int64_t n, int64_t H, int64_t K, int64_t transpose_b,
    int64_t n_chunk, int64_t ELL, int64_t total);
torch::Tensor lf1b_warpcid(torch::Tensor seed, torch::Tensor label,
    torch::Tensor neg_rho, int64_t base, int64_t B_row_start, int64_t chunk_lo,
    int64_t k, int64_t n, int64_t H, int64_t K, int64_t transpose_b,
    int64_t n_chunk, int64_t ELL, int64_t total);
torch::Tensor lf1b_perrow(torch::Tensor seed, torch::Tensor label,
    torch::Tensor neg_rho, int64_t base, int64_t B_row_start, int64_t chunk_lo,
    int64_t k, int64_t n, int64_t H, int64_t K, int64_t transpose_b,
    int64_t n_chunk, int64_t ELL, int64_t total);
torch::Tensor lf1b_multirow(torch::Tensor seed, torch::Tensor label,
    torch::Tensor neg_rho, int64_t base, int64_t B_row_start, int64_t chunk_lo,
    int64_t k, int64_t n, int64_t H, int64_t K, int64_t transpose_b,
    int64_t n_chunk, int64_t ELL, int64_t total, int64_t rows_per_thread);
torch::Tensor challenge_range(torch::Tensor seed, torch::Tensor label,
    int64_t base, int64_t n);
torch::Tensor lf1b_gather(torch::Tensor table, torch::Tensor neg_rho,
    int64_t B_row_start, int64_t chunk_lo, int64_t k, int64_t n, int64_t H, int64_t K,
    int64_t transpose_b, int64_t n_chunk, int64_t ELL, int64_t total);
torch::Tensor id_perslot(torch::Tensor seed, torch::Tensor label, int64_t coef,
    int64_t base, int64_t var_row_start, int64_t chunk_lo, int64_t L,
    int64_t n_chunk, int64_t ELL);
torch::Tensor id_gather(torch::Tensor table, int64_t coef, int64_t var_row_start,
    int64_t chunk_lo, int64_t L, int64_t n_chunk, int64_t ELL);
torch::Tensor stride_perslot(torch::Tensor seed, torch::Tensor label, int64_t coef,
    int64_t base, int64_t var_row_start, int64_t chunk_lo, int64_t L, int64_t stride,
    int64_t n_chunk, int64_t ELL);
torch::Tensor lf2a_perslot(torch::Tensor seed, torch::Tensor label, torch::Tensor neg_lam,
    int64_t base, int64_t A_row_start, int64_t chunk_lo, int64_t k, int64_t m, int64_t K,
    int64_t n_chunk, int64_t ELL, int64_t total);
torch::Tensor lf3c_perslot(torch::Tensor seed, torch::Tensor label, torch::Tensor lam,
    torch::Tensor rho, int64_t base, int64_t C_row_start, int64_t chunk_lo, int64_t m,
    int64_t n, int64_t H, int64_t n_chunk, int64_t ELL, int64_t total);
torch::Tensor s2o_perslot(torch::Tensor seed, torch::Tensor label, int64_t coef,
    int64_t base, int64_t var_row_start, int64_t chunk_lo, int64_t stride, int64_t L,
    int64_t n_chunk, int64_t ELL);
torch::Tensor psv_perslot(torch::Tensor seed, torch::Tensor label, torch::Tensor coef_vec,
    int64_t base, int64_t var_row_start, int64_t chunk_lo, int64_t L,
    int64_t n_chunk, int64_t ELL);
torch::Tensor rsv_perslot(torch::Tensor seed, torch::Tensor label, torch::Tensor coef_vec,
    int64_t base, int64_t var_row_start, int64_t chunk_lo, int64_t stride, int64_t L,
    int64_t n_chunk, int64_t ELL);
torch::Tensor ropexrot_perslot(torch::Tensor seed, torch::Tensor label,
    int64_t base, int64_t x_rot_row_start, int64_t chunk_lo,
    int64_t H, int64_t d_h, int64_t L, int64_t n_chunk, int64_t ELL);
torch::Tensor transpose_o2m_perslot(torch::Tensor seed, torch::Tensor label, int64_t coef,
    int64_t base, int64_t var_row_start, int64_t chunk_lo, int64_t L,
    int64_t rows, int64_t cols, int64_t fan, int64_t n_chunk, int64_t ELL);
torch::Tensor causal_id_perslot(torch::Tensor seed, torch::Tensor label, int64_t coef,
    int64_t base, int64_t var_row_start, int64_t chunk_lo, int64_t L, int64_t M, int64_t H,
    int64_t n_chunk, int64_t ELL);
torch::Tensor causal_c2_perslot(torch::Tensor seed, torch::Tensor label, int64_t coef,
    int64_t base, int64_t c2_row_start, int64_t chunk_lo, int64_t B, int64_t H,
    int64_t n_chunk, int64_t ELL);
torch::Tensor ropex_perslot(torch::Tensor seed, torch::Tensor label,
    torch::Tensor cos_t, torch::Tensor sin_t, int64_t base, int64_t x_row_start,
    int64_t chunk_lo, int64_t H, int64_t d_h, int64_t L, int64_t n_chunk, int64_t ELL);
torch::Tensor lf2a_gather(torch::Tensor table, torch::Tensor neg_lam,
    int64_t A_row_start, int64_t chunk_lo, int64_t k, int64_t m, int64_t K,
    int64_t n_chunk, int64_t ELL, int64_t total);
torch::Tensor lf3c_gather(torch::Tensor table, torch::Tensor lam, torch::Tensor rho,
    int64_t C_row_start, int64_t chunk_lo, int64_t m, int64_t n, int64_t H,
    int64_t n_chunk, int64_t ELL, int64_t total);
torch::Tensor s2o_gather(torch::Tensor table, int64_t coef, int64_t var_row_start,
    int64_t chunk_lo, int64_t stride, int64_t L, int64_t n_chunk, int64_t ELL);
torch::Tensor rsv_gather(torch::Tensor table, torch::Tensor coef_vec,
    int64_t var_row_start, int64_t chunk_lo, int64_t stride, int64_t L,
    int64_t n_chunk, int64_t ELL);
"""

_CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <cstdint>
#include "goldilocks.cuh"
#include "gl_spmv.cuh"   // gl_sparse::challenge_inline

// Decode B's flat index -> (j, i_k) and accumulate one slot's contribution.
// Mirrors packets._expand_freivalds_lf1b exactly.
__device__ __forceinline__ void lf1b_decode(
    int64_t flat, int64_t k, int64_t n, int64_t H, int64_t K, int transpose_b,
    int64_t* j_out, int64_t* ik_out
) {
    int64_t j, i_k;
    if (transpose_b) { j = flat / k; i_k = flat % k; }
    else {
        j = flat % n;
        int64_t rest = flat / n;
        int64_t h = rest % H;
        int64_t r = rest / H;
        i_k = h * K + r;
    }
    *j_out = j; *ik_out = i_k;
}

// --- per-slot: one thread per output slot of the chunk -------------------
__global__ void k_lf1b_perslot(
    const uint8_t* __restrict__ seed, const uint8_t* __restrict__ label, int label_len,
    const uint64_t* __restrict__ neg_rho,
    int64_t base, int64_t B_row_start, int64_t chunk_lo,
    int64_t k, int64_t n, int64_t H, int64_t K, int transpose_b,
    int64_t n_chunk, int64_t ELL, int64_t total, uint64_t* __restrict__ out
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n_out = n_chunk * ELL;
    if (idx >= n_out) return;
    int64_t lr = idx / ELL;
    int64_t s  = idx % ELL;
    int64_t flat = ((chunk_lo + lr) - B_row_start) * ELL + s;
    if (flat >= total) { out[idx] = 0; return; }
    int64_t j, i_k; lf1b_decode(flat, k, n, H, K, transpose_b, &j, &i_k);
    int64_t head = i_k / K;
    uint64_t r_cid = gl_sparse::challenge_inline(seed, label, label_len, (uint64_t)(base + i_k));
    out[idx] = gl::mul(neg_rho[head * n + j], r_cid);
}

// --- warp-collaborative cid hash: one thread per slot (full parallelism +
//     coalescing), but each warp hashes its shared cid ONCE and broadcasts.
//     A run of `n` consecutive slots shares one cid (i_k), and n >> 32, so the
//     common warp is all-one-cid (32 -> 1 hash). The rare straddle warp falls
//     back to per-lane hashing. ----------------------------------------------
__global__ void k_lf1b_warpcid(
    const uint8_t* __restrict__ seed, const uint8_t* __restrict__ label, int label_len,
    const uint64_t* __restrict__ neg_rho,
    int64_t base, int64_t B_row_start, int64_t chunk_lo,
    int64_t k, int64_t n, int64_t H, int64_t K, int transpose_b,
    int64_t n_chunk, int64_t ELL, int64_t total, uint64_t* __restrict__ out
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    int64_t n_out = n_chunk * ELL;
    int64_t lr = idx / ELL;
    int64_t s  = idx % ELL;
    int64_t flat = ((chunk_lo + lr) - B_row_start) * ELL + s;
    bool valid = (idx < n_out) && (flat < total);
    int64_t j = 0, i_k = 0;
    if (valid) lf1b_decode(flat, k, n, H, K, transpose_b, &j, &i_k);
    // distinct negative sentinel for invalid lanes so they never merge a run
    int64_t cid = valid ? (base + i_k) : (int64_t)(-1 - lane);
    int64_t cid0 = __shfl_sync(0xffffffffu, cid, 0);
    unsigned same = __ballot_sync(0xffffffffu, cid == cid0);
    uint64_t chal;
    if (same == 0xffffffffu) {            // whole warp shares one valid cid
        uint64_t c0 = 0;
        if (lane == 0) c0 = gl_sparse::challenge_inline(seed, label, label_len, (uint64_t)cid0);
        chal = __shfl_sync(0xffffffffu, c0, 0);
    } else {                              // straddle warp: per-lane
        chal = valid ? gl_sparse::challenge_inline(seed, label, label_len, (uint64_t)cid) : 0;
    }
    if (idx < n_out) {
        if (valid) { int64_t head = i_k / K; out[idx] = gl::mul(neg_rho[head * n + j], chal); }
        else out[idx] = 0;
    }
}

// --- per-row: one thread per witness row; reuse the challenge across the
//     contiguous run of slots sharing one cid (i_k) ----------------------
__global__ void k_lf1b_perrow(
    const uint8_t* __restrict__ seed, const uint8_t* __restrict__ label, int label_len,
    const uint64_t* __restrict__ neg_rho,
    int64_t base, int64_t B_row_start, int64_t chunk_lo,
    int64_t k, int64_t n, int64_t H, int64_t K, int transpose_b,
    int64_t n_chunk, int64_t ELL, int64_t total, uint64_t* __restrict__ out
) {
    int64_t lr = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (lr >= n_chunk) return;
    int64_t base_flat = ((chunk_lo + lr) - B_row_start) * ELL;
    int64_t cur_ik = -1;
    uint64_t cur_chal = 0;
    for (int64_t s = 0; s < ELL; ++s) {
        int64_t outidx = lr * ELL + s;
        int64_t flat = base_flat + s;
        if (flat >= total) { out[outidx] = 0; continue; }
        int64_t j, i_k; lf1b_decode(flat, k, n, H, K, transpose_b, &j, &i_k);
        if (i_k != cur_ik) {
            cur_ik = i_k;
            cur_chal = gl_sparse::challenge_inline(seed, label, label_len, (uint64_t)(base + i_k));
        }
        int64_t head = i_k / K;
        out[outidx] = gl::mul(neg_rho[head * n + j], cur_chal);
    }
}

// --- multi-row: one thread handles `rows_per_thread` consecutive rows
//     ("several packets in one thread"), reusing the challenge within each. -
__global__ void k_lf1b_multirow(
    const uint8_t* __restrict__ seed, const uint8_t* __restrict__ label, int label_len,
    const uint64_t* __restrict__ neg_rho,
    int64_t base, int64_t B_row_start, int64_t chunk_lo,
    int64_t k, int64_t n, int64_t H, int64_t K, int transpose_b,
    int64_t n_chunk, int64_t ELL, int64_t total, int64_t rows_per_thread,
    uint64_t* __restrict__ out
) {
    int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t lr0 = t * rows_per_thread;
    if (lr0 >= n_chunk) return;
    int64_t lr1 = lr0 + rows_per_thread;
    if (lr1 > n_chunk) lr1 = n_chunk;
    for (int64_t lr = lr0; lr < lr1; ++lr) {
        int64_t base_flat = ((chunk_lo + lr) - B_row_start) * ELL;
        int64_t cur_ik = -1;
        uint64_t cur_chal = 0;
        for (int64_t s = 0; s < ELL; ++s) {
            int64_t outidx = lr * ELL + s;
            int64_t flat = base_flat + s;
            if (flat >= total) { out[outidx] = 0; continue; }
            int64_t j, i_k; lf1b_decode(flat, k, n, H, K, transpose_b, &j, &i_k);
            if (i_k != cur_ik) {
                cur_ik = i_k;
                cur_chal = gl_sparse::challenge_inline(seed, label, label_len, (uint64_t)(base + i_k));
            }
            int64_t head = i_k / K;
            out[outidx] = gl::mul(neg_rho[head * n + j], cur_chal);
        }
    }
}

// --- precompute path: hash the <= k distinct cids once, then pure gather. ---
__global__ void k_challenge_range(
    const uint8_t* __restrict__ seed, const uint8_t* __restrict__ label, int label_len,
    int64_t base, int64_t n, uint64_t* __restrict__ out
) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = gl_sparse::challenge_inline(seed, label, label_len, (uint64_t)(base + i));
}

// gather: one thread per slot, read the precomputed challenge[i_k]. No hashing.
__global__ void k_lf1b_gather(
    const uint64_t* __restrict__ table, const uint64_t* __restrict__ neg_rho,
    int64_t B_row_start, int64_t chunk_lo,
    int64_t k, int64_t n, int64_t H, int64_t K, int transpose_b,
    int64_t n_chunk, int64_t ELL, int64_t total, uint64_t* __restrict__ out
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n_out = n_chunk * ELL;
    if (idx >= n_out) return;
    int64_t lr = idx / ELL;
    int64_t s  = idx % ELL;
    int64_t flat = ((chunk_lo + lr) - B_row_start) * ELL + s;
    if (flat >= total) { out[idx] = 0; return; }
    int64_t j, i_k; lf1b_decode(flat, k, n, H, K, transpose_b, &j, &i_k);
    int64_t head = i_k / K;
    out[idx] = gl::mul(neg_rho[head * n + j], table[i_k]);
}

// ===== Identity family: cid = base + flat, one cid per slot (NO reuse). =====
__global__ void k_id_perslot(
    const uint8_t* __restrict__ seed, const uint8_t* __restrict__ label, int label_len,
    uint64_t coef, int64_t base, int64_t var_row_start, int64_t chunk_lo,
    int64_t L, int64_t n_chunk, int64_t ELL, uint64_t* __restrict__ out
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_chunk * ELL) return;
    int64_t flat = ((chunk_lo + idx / ELL) - var_row_start) * ELL + (idx % ELL);
    if (flat >= L) { out[idx] = 0; return; }
    out[idx] = gl::mul(coef, gl_sparse::challenge_inline(seed, label, label_len, (uint64_t)(base + flat)));
}

__global__ void k_id_gather(
    const uint64_t* __restrict__ table, uint64_t coef, int64_t var_row_start, int64_t chunk_lo,
    int64_t L, int64_t n_chunk, int64_t ELL, uint64_t* __restrict__ out
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_chunk * ELL) return;
    int64_t flat = ((chunk_lo + idx / ELL) - var_row_start) * ELL + (idx % ELL);
    if (flat >= L) { out[idx] = 0; return; }
    out[idx] = gl::mul(coef, table[flat]);
}

// ===== Fan-out: each slot SUMS `stride` distinct challenges (anti-reuse). =====
__global__ void k_stride_perslot(
    const uint8_t* __restrict__ seed, const uint8_t* __restrict__ label, int label_len,
    uint64_t coef, int64_t base, int64_t var_row_start, int64_t chunk_lo,
    int64_t L, int64_t stride, int64_t n_chunk, int64_t ELL, uint64_t* __restrict__ out
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_chunk * ELL) return;
    int64_t flat = ((chunk_lo + idx / ELL) - var_row_start) * ELL + (idx % ELL);
    if (flat >= L) { out[idx] = 0; return; }
    uint64_t acc = 0;
    int64_t c0 = base + flat * stride;
    for (int64_t t = 0; t < stride; ++t)
        acc = gl::add(acc, gl_sparse::challenge_inline(seed, label, label_len, (uint64_t)(c0 + t)));
    out[idx] = gl::mul(coef, acc);
}

// ===== Freivalds LF2A (A side): cid = base + (flat % k); coef = neg_lam[head*m + flat/k] =====
__global__ void k_lf2a_perslot(
    const uint8_t* __restrict__ seed, const uint8_t* __restrict__ label, int label_len,
    const uint64_t* __restrict__ neg_lam,
    int64_t base, int64_t A_row_start, int64_t chunk_lo,
    int64_t k, int64_t m, int64_t K, int64_t n_chunk, int64_t ELL, int64_t total,
    uint64_t* __restrict__ out
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_chunk * ELL) return;
    int64_t flat = ((chunk_lo + idx / ELL) - A_row_start) * ELL + (idx % ELL);
    if (flat >= total) { out[idx] = 0; return; }
    int64_t i_k = flat % k, i_outer = flat / k, head = i_k / K;
    uint64_t r = gl_sparse::challenge_inline(seed, label, label_len, (uint64_t)(base + i_k));
    out[idx] = gl::mul(neg_lam[head * m + i_outer], r);
}

// ===== Freivalds LF3C (C side, per head): cid = base + h; coef = -lam[h*m+i]*rho[h*n+j] =====
__global__ void k_lf3c_perslot(
    const uint8_t* __restrict__ seed, const uint8_t* __restrict__ label, int label_len,
    const uint64_t* __restrict__ lam, const uint64_t* __restrict__ rho,
    int64_t base, int64_t C_row_start, int64_t chunk_lo,
    int64_t m, int64_t n, int64_t H, int64_t n_chunk, int64_t ELL, int64_t total,
    uint64_t* __restrict__ out
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_chunk * ELL) return;
    int64_t flat = ((chunk_lo + idx / ELL) - C_row_start) * ELL + (idx % ELL);
    if (flat >= total) { out[idx] = 0; return; }
    int64_t j = flat % n, rest = flat / n, h = rest % H, i = rest / H;
    uint64_t coef = gl::sub((uint64_t)0, gl::mul(lam[h * m + i], rho[h * n + j]));
    uint64_t r = gl_sparse::challenge_inline(seed, label, label_len, (uint64_t)(base + h));
    out[idx] = gl::mul(coef, r);
}

// ===== StrideManyToOne: cid = base + flat/stride; coef = scalar (high reuse) =====
__global__ void k_s2o_perslot(
    const uint8_t* __restrict__ seed, const uint8_t* __restrict__ label, int label_len,
    uint64_t coef, int64_t base, int64_t var_row_start, int64_t chunk_lo,
    int64_t stride, int64_t L, int64_t n_chunk, int64_t ELL, uint64_t* __restrict__ out
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_chunk * ELL) return;
    int64_t flat = ((chunk_lo + idx / ELL) - var_row_start) * ELL + (idx % ELL);
    if (flat >= L) { out[idx] = 0; return; }
    uint64_t r = gl_sparse::challenge_inline(seed, label, label_len, (uint64_t)(base + flat / stride));
    out[idx] = gl::mul(coef, r);
}

// ===== PerSlotVector: cid = base + flat; coef = coef_vec[flat] (1:1, vec coef) =====
__global__ void k_psv_perslot(
    const uint8_t* __restrict__ seed, const uint8_t* __restrict__ label, int label_len,
    const uint64_t* __restrict__ coef_vec,
    int64_t base, int64_t var_row_start, int64_t chunk_lo, int64_t L,
    int64_t n_chunk, int64_t ELL, uint64_t* __restrict__ out
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_chunk * ELL) return;
    int64_t flat = ((chunk_lo + idx / ELL) - var_row_start) * ELL + (idx % ELL);
    if (flat >= L) { out[idx] = 0; return; }
    uint64_t r = gl_sparse::challenge_inline(seed, label, label_len, (uint64_t)(base + flat));
    out[idx] = gl::mul(coef_vec[flat], r);
}

// ===== RowSumPerSlotVector: cid = base + flat/stride; coef = coef_vec[flat % stride] =====
__global__ void k_rsv_perslot(
    const uint8_t* __restrict__ seed, const uint8_t* __restrict__ label, int label_len,
    const uint64_t* __restrict__ coef_vec,
    int64_t base, int64_t var_row_start, int64_t chunk_lo, int64_t stride, int64_t L,
    int64_t n_chunk, int64_t ELL, uint64_t* __restrict__ out
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_chunk * ELL) return;
    int64_t flat = ((chunk_lo + idx / ELL) - var_row_start) * ELL + (idx % ELL);
    if (flat >= L) { out[idx] = 0; return; }
    uint64_t r = gl_sparse::challenge_inline(seed, label, label_len, (uint64_t)(base + flat / stride));
    out[idx] = gl::mul(coef_vec[flat % stride], r);
}

// ===== RoPEXRot: each x_rot slot -> ONE cid (base + 2*pair_t + e_self), coef 1.
//   seq = flat/(H*d_h); h = (flat/d_h)%H; k = flat%d_h; half = d_h/2;
//   e_self = k/half; k_in_pair = k%half; pair_t = seq*H*half + h*half + k_in_pair.
//   Mirrors packets._expand_rope_xrot exactly. Per-slot, 1 hash, no atomics. =====
__global__ void k_ropexrot_perslot(
    const uint8_t* __restrict__ seed, const uint8_t* __restrict__ label, int label_len,
    int64_t base, int64_t x_rot_row_start, int64_t chunk_lo,
    int64_t H, int64_t d_h, int64_t L, int64_t n_chunk, int64_t ELL,
    uint64_t* __restrict__ out
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_chunk * ELL) return;
    int64_t flat = ((chunk_lo + idx / ELL) - x_rot_row_start) * ELL + (idx % ELL);
    if (flat >= L) { out[idx] = 0; return; }
    int64_t half      = d_h / 2;
    int64_t seq       = flat / (H * d_h);
    int64_t h         = (flat / d_h) % H;
    int64_t kk        = flat % d_h;
    int64_t e_self    = kk / half;
    int64_t k_in_pair = kk % half;
    int64_t pair_t    = seq * H * half + h * half + k_in_pair;
    int64_t cid       = base + 2 * pair_t + e_self;
    uint64_t r = gl_sparse::challenge_inline(seed, label, label_len, (uint64_t)cid);
    out[idx] = gl::mul((uint64_t)1, r);   // coef = 1
}

// ===== TransposeO2MScalar: each slot sums `fan` cids at transposed positions.
//   cid_lo = base + (flat%cols)*rows*fan + (flat/cols)*fan; sum k in [0,fan). =====
__global__ void k_transpose_o2m_perslot(
    const uint8_t* __restrict__ seed, const uint8_t* __restrict__ label, int label_len,
    uint64_t coef, int64_t base, int64_t var_row_start, int64_t chunk_lo,
    int64_t L, int64_t rows, int64_t cols, int64_t fan,
    int64_t n_chunk, int64_t ELL, uint64_t* __restrict__ out
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_chunk * ELL) return;
    int64_t flat = ((chunk_lo + idx / ELL) - var_row_start) * ELL + (idx % ELL);
    if (flat >= L) { out[idx] = 0; return; }
    int64_t cid_lo = base + (flat % cols) * (rows * fan) + (flat / cols) * fan;
    uint64_t acc = 0;
    for (int64_t k = 0; k < fan; ++k)
        acc = gl::add(acc, gl_sparse::challenge_inline(seed, label, label_len,
                                                       (uint64_t)(cid_lo + k)));
    out[idx] = gl::mul(coef, acc);
}

// ===== CausalFilteredIdScalar: causal identity; masked iff j > i_qry -> out=0.
//   b=flat/M, j=flat%M, i_qry=b/H, h=b%H;
//   rank = H*i_qry*(i_qry+1)/2 + h*(i_qry+1) + j; cid = base + rank. =====
__global__ void k_causal_id_perslot(
    const uint8_t* __restrict__ seed, const uint8_t* __restrict__ label, int label_len,
    uint64_t coef, int64_t base, int64_t var_row_start, int64_t chunk_lo,
    int64_t L, int64_t M, int64_t H, int64_t n_chunk, int64_t ELL, uint64_t* __restrict__ out
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_chunk * ELL) return;
    int64_t flat = ((chunk_lo + idx / ELL) - var_row_start) * ELL + (idx % ELL);
    if (flat >= L) { out[idx] = 0; return; }
    int64_t b = flat / M, j = flat % M;
    int64_t i_qry = b / H, h = b % H;
    if (j > i_qry) { out[idx] = 0; return; }            // causal mask
    int64_t rank = H * i_qry * (i_qry + 1) / 2 + h * (i_qry + 1) + j;
    uint64_t r = gl_sparse::challenge_inline(seed, label, label_len, (uint64_t)(base + rank));
    out[idx] = gl::mul(coef, r);
}

// ===== CausalFilteredC2Stride: ragged fan-sum; slot b sums (i_qry+1) cids from
//   rank_start = H*i_qry*(i_qry+1)/2 + h*(i_qry+1). Every slot active (no mask). =====
__global__ void k_causal_c2_perslot(
    const uint8_t* __restrict__ seed, const uint8_t* __restrict__ label, int label_len,
    uint64_t coef, int64_t base, int64_t c2_row_start, int64_t chunk_lo,
    int64_t B, int64_t H, int64_t n_chunk, int64_t ELL, uint64_t* __restrict__ out
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_chunk * ELL) return;
    int64_t b = ((chunk_lo + idx / ELL) - c2_row_start) * ELL + (idx % ELL);
    if (b >= B) { out[idx] = 0; return; }
    int64_t i_qry = b / H, h = b % H;
    int64_t rank_start = H * i_qry * (i_qry + 1) / 2 + h * (i_qry + 1);
    uint64_t acc = 0;
    for (int64_t k = 0; k <= i_qry; ++k)
        acc = gl::add(acc, gl_sparse::challenge_inline(seed, label, label_len,
                                                       (uint64_t)(base + rank_start + k)));
    out[idx] = gl::mul(coef, acc);
}

// ===== RoPEX: each x slot -> two consecutive cids (2*pair_t, +1), coefs +-cos/+-sin
//   by e_self; coef_idx = seq*half + k_in_pair into cos_t/sin_t. neg via gl::sub(0,.). =====
__global__ void k_ropex_perslot(
    const uint8_t* __restrict__ seed, const uint8_t* __restrict__ label, int label_len,
    const uint64_t* __restrict__ cos_t, const uint64_t* __restrict__ sin_t,
    int64_t base, int64_t x_row_start, int64_t chunk_lo,
    int64_t H, int64_t d_h, int64_t L, int64_t n_chunk, int64_t ELL, uint64_t* __restrict__ out
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_chunk * ELL) return;
    int64_t flat = ((chunk_lo + idx / ELL) - x_row_start) * ELL + (idx % ELL);
    if (flat >= L) { out[idx] = 0; return; }
    int64_t half      = d_h / 2;
    int64_t seq       = flat / (H * d_h);
    int64_t kk        = flat % d_h;
    int64_t e_self    = kk / half;
    int64_t k_in_pair = kk % half;
    int64_t h         = (flat / d_h) % H;
    int64_t pair_t    = seq * H * half + h * half + k_in_pair;
    int64_t coef_idx  = seq * half + k_in_pair;
    uint64_t c = cos_t[coef_idx];
    uint64_t s = sin_t[coef_idx];
    uint64_t neg_c = gl::sub((uint64_t)0, c);
    uint64_t neg_s = gl::sub((uint64_t)0, s);
    uint64_t coef_eq1 = (e_self == 0) ? neg_c : s;        // lo: -c, hi: +s
    uint64_t coef_eq2 = (e_self == 0) ? neg_s : neg_c;    // lo: -s, hi: -c
    int64_t cid1 = base + 2 * pair_t;
    int64_t cid2 = base + 2 * pair_t + 1;
    uint64_t r1 = gl_sparse::challenge_inline(seed, label, label_len, (uint64_t)cid1);
    uint64_t r2 = gl_sparse::challenge_inline(seed, label, label_len, (uint64_t)cid2);
    out[idx] = gl::add(gl::mul(coef_eq1, r1), gl::mul(coef_eq2, r2));
}

// ===== Precompute/gather variants for the high-reuse families: hash the family's
//   <= n_distinct cids once (challenge_range), then a pure gather + coef + write
//   (no inline BLAKE3). The per-slot cid index must equal the per-slot kernel's
//   (base + index) so table[index] is bit-identical to the inline hash. =====

// LF2A gather: cid index = flat % k; coef = neg_lam[head*m + flat/k].
__global__ void k_lf2a_gather(
    const uint64_t* __restrict__ table, const uint64_t* __restrict__ neg_lam,
    int64_t A_row_start, int64_t chunk_lo, int64_t k, int64_t m, int64_t K,
    int64_t n_chunk, int64_t ELL, int64_t total, uint64_t* __restrict__ out
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_chunk * ELL) return;
    int64_t flat = ((chunk_lo + idx / ELL) - A_row_start) * ELL + (idx % ELL);
    if (flat >= total) { out[idx] = 0; return; }
    int64_t i_k = flat % k, i_outer = flat / k, head = i_k / K;
    out[idx] = gl::mul(neg_lam[head * m + i_outer], table[i_k]);
}

// LF3C gather: cid index = h; coef = -lam[h*m+i]*rho[h*n+j] (still per slot).
__global__ void k_lf3c_gather(
    const uint64_t* __restrict__ table, const uint64_t* __restrict__ lam,
    const uint64_t* __restrict__ rho, int64_t C_row_start, int64_t chunk_lo,
    int64_t m, int64_t n, int64_t H, int64_t n_chunk, int64_t ELL, int64_t total,
    uint64_t* __restrict__ out
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_chunk * ELL) return;
    int64_t flat = ((chunk_lo + idx / ELL) - C_row_start) * ELL + (idx % ELL);
    if (flat >= total) { out[idx] = 0; return; }
    int64_t j = flat % n, rest = flat / n, h = rest % H, i = rest / H;
    uint64_t coef = gl::sub((uint64_t)0, gl::mul(lam[h * m + i], rho[h * n + j]));
    out[idx] = gl::mul(coef, table[h]);
}

// StrideManyToOne gather: cid index = flat / stride; scalar coef.
__global__ void k_s2o_gather(
    const uint64_t* __restrict__ table, uint64_t coef, int64_t var_row_start,
    int64_t chunk_lo, int64_t stride, int64_t L, int64_t n_chunk, int64_t ELL,
    uint64_t* __restrict__ out
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_chunk * ELL) return;
    int64_t flat = ((chunk_lo + idx / ELL) - var_row_start) * ELL + (idx % ELL);
    if (flat >= L) { out[idx] = 0; return; }
    out[idx] = gl::mul(coef, table[flat / stride]);
}

// RowSumPerSlotVector gather: cid index = flat / stride; coef = coef_vec[flat % stride].
__global__ void k_rsv_gather(
    const uint64_t* __restrict__ table, const uint64_t* __restrict__ coef_vec,
    int64_t var_row_start, int64_t chunk_lo, int64_t stride, int64_t L,
    int64_t n_chunk, int64_t ELL, uint64_t* __restrict__ out
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_chunk * ELL) return;
    int64_t flat = ((chunk_lo + idx / ELL) - var_row_start) * ELL + (idx % ELL);
    if (flat >= L) { out[idx] = 0; return; }
    out[idx] = gl::mul(coef_vec[flat % stride], table[flat / stride]);
}

static uint64_t* u64(torch::Tensor t) { return (uint64_t*)t.data_ptr(); }
static const uint8_t* u8(torch::Tensor t) { return (const uint8_t*)t.data_ptr(); }

torch::Tensor lf1b_perslot(torch::Tensor seed, torch::Tensor label,
    torch::Tensor neg_rho, int64_t base, int64_t B_row_start, int64_t chunk_lo,
    int64_t k, int64_t n, int64_t H, int64_t K, int64_t transpose_b,
    int64_t n_chunk, int64_t ELL, int64_t total) {
    int64_t n_out = n_chunk * ELL;
    auto out = torch::zeros({n_out}, torch::dtype(torch::kUInt64).device(torch::kCUDA));
    int blk = 256; int64_t g = (n_out + blk - 1) / blk;
    k_lf1b_perslot<<<g, blk>>>(u8(seed), u8(label), (int)label.numel(), u64(neg_rho),
        base, B_row_start, chunk_lo, k, n, H, K, (int)transpose_b, n_chunk, ELL, total, u64(out));
    return out;
}

torch::Tensor lf1b_warpcid(torch::Tensor seed, torch::Tensor label,
    torch::Tensor neg_rho, int64_t base, int64_t B_row_start, int64_t chunk_lo,
    int64_t k, int64_t n, int64_t H, int64_t K, int64_t transpose_b,
    int64_t n_chunk, int64_t ELL, int64_t total) {
    int64_t n_out = n_chunk * ELL;
    auto out = torch::zeros({n_out}, torch::dtype(torch::kUInt64).device(torch::kCUDA));
    int blk = 256; int64_t g = (n_out + blk - 1) / blk;
    k_lf1b_warpcid<<<g, blk>>>(u8(seed), u8(label), (int)label.numel(), u64(neg_rho),
        base, B_row_start, chunk_lo, k, n, H, K, (int)transpose_b, n_chunk, ELL, total, u64(out));
    return out;
}

torch::Tensor lf1b_perrow(torch::Tensor seed, torch::Tensor label,
    torch::Tensor neg_rho, int64_t base, int64_t B_row_start, int64_t chunk_lo,
    int64_t k, int64_t n, int64_t H, int64_t K, int64_t transpose_b,
    int64_t n_chunk, int64_t ELL, int64_t total) {
    int64_t n_out = n_chunk * ELL;
    auto out = torch::zeros({n_out}, torch::dtype(torch::kUInt64).device(torch::kCUDA));
    int blk = 256; int64_t g = (n_chunk + blk - 1) / blk;
    k_lf1b_perrow<<<g, blk>>>(u8(seed), u8(label), (int)label.numel(), u64(neg_rho),
        base, B_row_start, chunk_lo, k, n, H, K, (int)transpose_b, n_chunk, ELL, total, u64(out));
    return out;
}

torch::Tensor lf1b_multirow(torch::Tensor seed, torch::Tensor label,
    torch::Tensor neg_rho, int64_t base, int64_t B_row_start, int64_t chunk_lo,
    int64_t k, int64_t n, int64_t H, int64_t K, int64_t transpose_b,
    int64_t n_chunk, int64_t ELL, int64_t total, int64_t rows_per_thread) {
    int64_t n_out = n_chunk * ELL;
    auto out = torch::zeros({n_out}, torch::dtype(torch::kUInt64).device(torch::kCUDA));
    int blk = 256;
    int64_t n_threads = (n_chunk + rows_per_thread - 1) / rows_per_thread;
    int64_t g = (n_threads + blk - 1) / blk;
    k_lf1b_multirow<<<g, blk>>>(u8(seed), u8(label), (int)label.numel(), u64(neg_rho),
        base, B_row_start, chunk_lo, k, n, H, K, (int)transpose_b, n_chunk, ELL, total,
        rows_per_thread, u64(out));
    return out;
}

torch::Tensor challenge_range(torch::Tensor seed, torch::Tensor label,
    int64_t base, int64_t n) {
    auto out = torch::empty({n}, torch::dtype(torch::kUInt64).device(torch::kCUDA));
    int blk = 256; int64_t g = (n + blk - 1) / blk;
    k_challenge_range<<<g, blk>>>(u8(seed), u8(label), (int)label.numel(), base, n, u64(out));
    return out;
}

torch::Tensor lf1b_gather(torch::Tensor table, torch::Tensor neg_rho,
    int64_t B_row_start, int64_t chunk_lo, int64_t k, int64_t n, int64_t H, int64_t K,
    int64_t transpose_b, int64_t n_chunk, int64_t ELL, int64_t total) {
    int64_t n_out = n_chunk * ELL;
    auto out = torch::zeros({n_out}, torch::dtype(torch::kUInt64).device(torch::kCUDA));
    int blk = 256; int64_t g = (n_out + blk - 1) / blk;
    k_lf1b_gather<<<g, blk>>>(u64(table), u64(neg_rho), B_row_start, chunk_lo, k, n, H, K,
        (int)transpose_b, n_chunk, ELL, total, u64(out));
    return out;
}

torch::Tensor id_perslot(torch::Tensor seed, torch::Tensor label, int64_t coef,
    int64_t base, int64_t var_row_start, int64_t chunk_lo, int64_t L,
    int64_t n_chunk, int64_t ELL) {
    int64_t n_out = n_chunk * ELL;
    auto out = torch::zeros({n_out}, torch::dtype(torch::kUInt64).device(torch::kCUDA));
    int blk = 256; int64_t g = (n_out + blk - 1) / blk;
    k_id_perslot<<<g, blk>>>(u8(seed), u8(label), (int)label.numel(), (uint64_t)coef,
        base, var_row_start, chunk_lo, L, n_chunk, ELL, u64(out));
    return out;
}

torch::Tensor id_gather(torch::Tensor table, int64_t coef, int64_t var_row_start,
    int64_t chunk_lo, int64_t L, int64_t n_chunk, int64_t ELL) {
    int64_t n_out = n_chunk * ELL;
    auto out = torch::zeros({n_out}, torch::dtype(torch::kUInt64).device(torch::kCUDA));
    int blk = 256; int64_t g = (n_out + blk - 1) / blk;
    k_id_gather<<<g, blk>>>(u64(table), (uint64_t)coef, var_row_start, chunk_lo, L,
        n_chunk, ELL, u64(out));
    return out;
}

torch::Tensor stride_perslot(torch::Tensor seed, torch::Tensor label, int64_t coef,
    int64_t base, int64_t var_row_start, int64_t chunk_lo, int64_t L, int64_t stride,
    int64_t n_chunk, int64_t ELL) {
    int64_t n_out = n_chunk * ELL;
    auto out = torch::zeros({n_out}, torch::dtype(torch::kUInt64).device(torch::kCUDA));
    int blk = 256; int64_t g = (n_out + blk - 1) / blk;
    k_stride_perslot<<<g, blk>>>(u8(seed), u8(label), (int)label.numel(), (uint64_t)coef,
        base, var_row_start, chunk_lo, L, stride, n_chunk, ELL, u64(out));
    return out;
}

torch::Tensor lf2a_perslot(torch::Tensor seed, torch::Tensor label, torch::Tensor neg_lam,
    int64_t base, int64_t A_row_start, int64_t chunk_lo, int64_t k, int64_t m, int64_t K,
    int64_t n_chunk, int64_t ELL, int64_t total) {
    int64_t n_out = n_chunk * ELL;
    auto out = torch::zeros({n_out}, torch::dtype(torch::kUInt64).device(torch::kCUDA));
    int blk = 256; int64_t g = (n_out + blk - 1) / blk;
    k_lf2a_perslot<<<g, blk>>>(u8(seed), u8(label), (int)label.numel(), u64(neg_lam),
        base, A_row_start, chunk_lo, k, m, K, n_chunk, ELL, total, u64(out));
    return out;
}

torch::Tensor lf3c_perslot(torch::Tensor seed, torch::Tensor label, torch::Tensor lam,
    torch::Tensor rho, int64_t base, int64_t C_row_start, int64_t chunk_lo, int64_t m,
    int64_t n, int64_t H, int64_t n_chunk, int64_t ELL, int64_t total) {
    int64_t n_out = n_chunk * ELL;
    auto out = torch::zeros({n_out}, torch::dtype(torch::kUInt64).device(torch::kCUDA));
    int blk = 256; int64_t g = (n_out + blk - 1) / blk;
    k_lf3c_perslot<<<g, blk>>>(u8(seed), u8(label), (int)label.numel(), u64(lam), u64(rho),
        base, C_row_start, chunk_lo, m, n, H, n_chunk, ELL, total, u64(out));
    return out;
}

torch::Tensor s2o_perslot(torch::Tensor seed, torch::Tensor label, int64_t coef,
    int64_t base, int64_t var_row_start, int64_t chunk_lo, int64_t stride, int64_t L,
    int64_t n_chunk, int64_t ELL) {
    int64_t n_out = n_chunk * ELL;
    auto out = torch::zeros({n_out}, torch::dtype(torch::kUInt64).device(torch::kCUDA));
    int blk = 256; int64_t g = (n_out + blk - 1) / blk;
    k_s2o_perslot<<<g, blk>>>(u8(seed), u8(label), (int)label.numel(), (uint64_t)coef,
        base, var_row_start, chunk_lo, stride, L, n_chunk, ELL, u64(out));
    return out;
}

torch::Tensor psv_perslot(torch::Tensor seed, torch::Tensor label, torch::Tensor coef_vec,
    int64_t base, int64_t var_row_start, int64_t chunk_lo, int64_t L,
    int64_t n_chunk, int64_t ELL) {
    int64_t n_out = n_chunk * ELL;
    auto out = torch::zeros({n_out}, torch::dtype(torch::kUInt64).device(torch::kCUDA));
    int blk = 256; int64_t g = (n_out + blk - 1) / blk;
    k_psv_perslot<<<g, blk>>>(u8(seed), u8(label), (int)label.numel(), u64(coef_vec),
        base, var_row_start, chunk_lo, L, n_chunk, ELL, u64(out));
    return out;
}

torch::Tensor rsv_perslot(torch::Tensor seed, torch::Tensor label, torch::Tensor coef_vec,
    int64_t base, int64_t var_row_start, int64_t chunk_lo, int64_t stride, int64_t L,
    int64_t n_chunk, int64_t ELL) {
    int64_t n_out = n_chunk * ELL;
    auto out = torch::zeros({n_out}, torch::dtype(torch::kUInt64).device(torch::kCUDA));
    int blk = 256; int64_t g = (n_out + blk - 1) / blk;
    k_rsv_perslot<<<g, blk>>>(u8(seed), u8(label), (int)label.numel(), u64(coef_vec),
        base, var_row_start, chunk_lo, stride, L, n_chunk, ELL, u64(out));
    return out;
}

torch::Tensor ropexrot_perslot(torch::Tensor seed, torch::Tensor label,
    int64_t base, int64_t x_rot_row_start, int64_t chunk_lo,
    int64_t H, int64_t d_h, int64_t L, int64_t n_chunk, int64_t ELL) {
    int64_t n_out = n_chunk * ELL;
    auto out = torch::zeros({n_out}, torch::dtype(torch::kUInt64).device(torch::kCUDA));
    int blk = 256; int64_t g = (n_out + blk - 1) / blk;
    k_ropexrot_perslot<<<g, blk>>>(u8(seed), u8(label), (int)label.numel(),
        base, x_rot_row_start, chunk_lo, H, d_h, L, n_chunk, ELL, u64(out));
    return out;
}

torch::Tensor transpose_o2m_perslot(torch::Tensor seed, torch::Tensor label, int64_t coef,
    int64_t base, int64_t var_row_start, int64_t chunk_lo, int64_t L,
    int64_t rows, int64_t cols, int64_t fan, int64_t n_chunk, int64_t ELL) {
    int64_t n_out = n_chunk * ELL;
    auto out = torch::zeros({n_out}, torch::dtype(torch::kUInt64).device(torch::kCUDA));
    int blk = 256; int64_t g = (n_out + blk - 1) / blk;
    k_transpose_o2m_perslot<<<g, blk>>>(u8(seed), u8(label), (int)label.numel(), (uint64_t)coef,
        base, var_row_start, chunk_lo, L, rows, cols, fan, n_chunk, ELL, u64(out));
    return out;
}

torch::Tensor causal_id_perslot(torch::Tensor seed, torch::Tensor label, int64_t coef,
    int64_t base, int64_t var_row_start, int64_t chunk_lo, int64_t L, int64_t M, int64_t H,
    int64_t n_chunk, int64_t ELL) {
    int64_t n_out = n_chunk * ELL;
    auto out = torch::zeros({n_out}, torch::dtype(torch::kUInt64).device(torch::kCUDA));
    int blk = 256; int64_t g = (n_out + blk - 1) / blk;
    k_causal_id_perslot<<<g, blk>>>(u8(seed), u8(label), (int)label.numel(), (uint64_t)coef,
        base, var_row_start, chunk_lo, L, M, H, n_chunk, ELL, u64(out));
    return out;
}

torch::Tensor causal_c2_perslot(torch::Tensor seed, torch::Tensor label, int64_t coef,
    int64_t base, int64_t c2_row_start, int64_t chunk_lo, int64_t B, int64_t H,
    int64_t n_chunk, int64_t ELL) {
    int64_t n_out = n_chunk * ELL;
    auto out = torch::zeros({n_out}, torch::dtype(torch::kUInt64).device(torch::kCUDA));
    int blk = 256; int64_t g = (n_out + blk - 1) / blk;
    k_causal_c2_perslot<<<g, blk>>>(u8(seed), u8(label), (int)label.numel(), (uint64_t)coef,
        base, c2_row_start, chunk_lo, B, H, n_chunk, ELL, u64(out));
    return out;
}

torch::Tensor ropex_perslot(torch::Tensor seed, torch::Tensor label,
    torch::Tensor cos_t, torch::Tensor sin_t, int64_t base, int64_t x_row_start,
    int64_t chunk_lo, int64_t H, int64_t d_h, int64_t L, int64_t n_chunk, int64_t ELL) {
    int64_t n_out = n_chunk * ELL;
    auto out = torch::zeros({n_out}, torch::dtype(torch::kUInt64).device(torch::kCUDA));
    int blk = 256; int64_t g = (n_out + blk - 1) / blk;
    k_ropex_perslot<<<g, blk>>>(u8(seed), u8(label), (int)label.numel(), u64(cos_t), u64(sin_t),
        base, x_row_start, chunk_lo, H, d_h, L, n_chunk, ELL, u64(out));
    return out;
}

torch::Tensor lf2a_gather(torch::Tensor table, torch::Tensor neg_lam,
    int64_t A_row_start, int64_t chunk_lo, int64_t k, int64_t m, int64_t K,
    int64_t n_chunk, int64_t ELL, int64_t total) {
    int64_t n_out = n_chunk * ELL;
    auto out = torch::zeros({n_out}, torch::dtype(torch::kUInt64).device(torch::kCUDA));
    int blk = 256; int64_t g = (n_out + blk - 1) / blk;
    k_lf2a_gather<<<g, blk>>>(u64(table), u64(neg_lam), A_row_start, chunk_lo, k, m, K,
        n_chunk, ELL, total, u64(out));
    return out;
}

torch::Tensor lf3c_gather(torch::Tensor table, torch::Tensor lam, torch::Tensor rho,
    int64_t C_row_start, int64_t chunk_lo, int64_t m, int64_t n, int64_t H,
    int64_t n_chunk, int64_t ELL, int64_t total) {
    int64_t n_out = n_chunk * ELL;
    auto out = torch::zeros({n_out}, torch::dtype(torch::kUInt64).device(torch::kCUDA));
    int blk = 256; int64_t g = (n_out + blk - 1) / blk;
    k_lf3c_gather<<<g, blk>>>(u64(table), u64(lam), u64(rho), C_row_start, chunk_lo,
        m, n, H, n_chunk, ELL, total, u64(out));
    return out;
}

torch::Tensor s2o_gather(torch::Tensor table, int64_t coef, int64_t var_row_start,
    int64_t chunk_lo, int64_t stride, int64_t L, int64_t n_chunk, int64_t ELL) {
    int64_t n_out = n_chunk * ELL;
    auto out = torch::zeros({n_out}, torch::dtype(torch::kUInt64).device(torch::kCUDA));
    int blk = 256; int64_t g = (n_out + blk - 1) / blk;
    k_s2o_gather<<<g, blk>>>(u64(table), (uint64_t)coef, var_row_start, chunk_lo,
        stride, L, n_chunk, ELL, u64(out));
    return out;
}

torch::Tensor rsv_gather(torch::Tensor table, torch::Tensor coef_vec,
    int64_t var_row_start, int64_t chunk_lo, int64_t stride, int64_t L,
    int64_t n_chunk, int64_t ELL) {
    int64_t n_out = n_chunk * ELL;
    auto out = torch::zeros({n_out}, torch::dtype(torch::kUInt64).device(torch::kCUDA));
    int blk = 256; int64_t g = (n_out + blk - 1) / blk;
    k_rsv_gather<<<g, blk>>>(u64(table), u64(coef_vec), var_row_start, chunk_lo,
        stride, L, n_chunk, ELL, u64(out));
    return out;
}
"""

_mod = None


def _module():
    global _mod
    if _mod is None:
        from torch.utils.cpp_extension import load_inline
        _mod = load_inline(
            name="qlin_fold_variants",
            cpp_sources=_CPP_DECLS,
            cuda_sources=_CUDA_SOURCE,
            functions=["lf1b_perslot", "lf1b_warpcid", "lf1b_perrow", "lf1b_multirow",
                       "challenge_range", "lf1b_gather",
                       "id_perslot", "id_gather", "stride_perslot",
                       "lf2a_perslot", "lf3c_perslot", "s2o_perslot",
                       "psv_perslot", "rsv_perslot", "ropexrot_perslot",
                       "transpose_o2m_perslot", "causal_id_perslot",
                       "causal_c2_perslot", "ropex_perslot",
                       "lf2a_gather", "lf3c_gather", "s2o_gather", "rsv_gather"],
            extra_include_paths=[str(_CUDA_DIR)],
            extra_cuda_cflags=[_arch_flag(), "-O3", "-std=c++17"],
            extra_cflags=["-std=c++17"],
            verbose=True,
        )
    return _mod


# ---------------------------------------------------------------------------
# Synthetic Freivalds LF1B chunk
# ---------------------------------------------------------------------------

class Chunk:
    """A pure-LF1B chunk: a contiguous block of a weight B-variable's rows."""
    def __init__(self, k, n, H, transpose_b, n_chunk, ELL, base, B_row_start, chunk_lo):
        assert k % H == 0, "k must be H * head_dim"
        self.k, self.n, self.H, self.K = k, n, H, k // H
        self.transpose_b = transpose_b
        self.n_chunk, self.ELL = n_chunk, ELL
        self.base, self.B_row_start, self.chunk_lo = base, B_row_start, chunk_lo
        self.total = k * n
        # public challenge inputs, drawn once (fixed seed for reproducibility)
        g = torch.Generator(device="cuda").manual_seed(0)
        # field elements in [0, 2^62) < P (P > int64 max, so randint(0, P) is invalid)
        self.neg_rho = torch.randint(0, 2 ** 62, (H * n,), generator=g,
                                     dtype=torch.int64, device="cuda").to(torch.uint64)
        self.seed = torch.arange(32, dtype=torch.uint8, device="cuda")
        self.label = torch.tensor(list(b"lin"), dtype=torch.uint8, device="cuda")

    @property
    def n_out(self):
        return self.n_chunk * self.ELL

    def packets(self):
        pkts, lrows = [], []
        for lr in range(self.n_chunk):
            pkts.append(L2_FreivaldsLF1B(
                base=self.base, B_row_start=self.B_row_start, k=self.k, n=self.n,
                H=self.H, K=self.K, transpose_b=self.transpose_b, neg_rho=self.neg_rho))
            lrows.append(lr)
        return pkts, lrows


def _spmv_reference(t, cid, v, n_out, seed, label):
    """expand triples -> argsort -> CSR -> gl_spmv_challenged (the current path)."""
    cids = cid.to(torch.uint64).contiguous()
    coefs = v.contiguous()
    perm = torch.argsort(t)
    sorted_tgt = t.index_select(0, perm)
    sorted_cid = cids.index_select(0, perm).contiguous()
    sorted_coef = coefs.index_select(0, perm).contiguous()
    counts = torch.bincount(sorted_tgt, minlength=n_out)
    row_ptr = torch.zeros(n_out + 1, dtype=torch.int64, device="cuda")
    row_ptr[1:] = counts.cumsum(0)
    return gl_spmv_challenged(sorted_coef, sorted_cid, row_ptr.to(torch.uint64),
                             seed, label, n_out)


def reference_freivalds(c):
    pkts, lrows = c.packets()
    t, cid, v = _expand_freivalds_lf1b(pkts, lrows, c.chunk_lo, c.ELL)
    return _spmv_reference(t, cid, v, c.n_out, c.seed, c.label).view(c.n_chunk, c.ELL)


def reference_identity(c):
    pkts, lrows = c.packets()
    t, cid, v = _expand_identity_scalar(pkts, lrows, c.chunk_lo, c.ELL)
    return _spmv_reference(t, cid, v, c.n_out, c.seed, c.label).view(c.n_chunk, c.ELL)


def reference_stride(c):
    pkts, lrows = c.packets()
    t, cid, v = _expand_stride_one_to_many(pkts, lrows, c.chunk_lo, c.ELL)
    return _spmv_reference(t, cid, v, c.n_out, c.seed, c.label).view(c.n_chunk, c.ELL)


class _ChunkBase:
    """Common challenge inputs for the non-Freivalds families."""
    def _init_common(self, n_chunk, ELL, base, var_row_start, chunk_lo):
        self.n_chunk, self.ELL = n_chunk, ELL
        self.base, self.var_row_start, self.chunk_lo = base, var_row_start, chunk_lo
        g = torch.Generator(device="cuda").manual_seed(0)
        self.coef = int(torch.randint(0, 2 ** 62, (1,), generator=g, device="cuda").item())
        self.seed = torch.arange(32, dtype=torch.uint8, device="cuda")
        self.label = torch.tensor(list(b"lin"), dtype=torch.uint8, device="cuda")

    @property
    def n_out(self):
        return self.n_chunk * self.ELL


class IdentityChunk(_ChunkBase):
    """One cid per slot (cid = base + flat): the zero-reuse / activation-copy case."""
    def __init__(self, L, n_chunk, ELL, base, var_row_start, chunk_lo):
        self.L = L
        self._init_common(n_chunk, ELL, base, var_row_start, chunk_lo)

    def packets(self):
        pkts = [L2_IdentityScalar(base=self.base, var_row_start=self.var_row_start,
                                  L=self.L, coef=self.coef) for _ in range(self.n_chunk)]
        return pkts, list(range(self.n_chunk))


class StrideChunk(_ChunkBase):
    """Each slot fans out to `stride` distinct cids, summed: the anti-reuse case."""
    def __init__(self, L, stride, n_chunk, ELL, base, var_row_start, chunk_lo):
        self.L, self.stride = L, stride
        self._init_common(n_chunk, ELL, base, var_row_start, chunk_lo)

    def packets(self):
        pkts = [L2_StrideOneToManyScalar(base=self.base, var_row_start=self.var_row_start,
                                         L=self.L, stride=self.stride, coef=self.coef)
                for _ in range(self.n_chunk)]
        return pkts, list(range(self.n_chunk))


class LF2AChunk(_ChunkBase):
    """A-side Freivalds: cid = base + flat%k, coef = neg_lam[head*m + flat//k]."""
    def __init__(self, m, k, H, n_chunk, ELL, base, var_row_start, chunk_lo):
        assert k % H == 0
        self.m, self.k, self.H, self.K = m, k, H, k // H
        self.total = m * k
        self._init_common(n_chunk, ELL, base, var_row_start, chunk_lo)
        g = torch.Generator(device="cuda").manual_seed(1)
        self.neg_lam = torch.randint(0, 2 ** 62, (H * m,), generator=g,
                                     dtype=torch.int64, device="cuda").to(torch.uint64)

    def packets(self):
        pkts = [L2_FreivaldsLF2A(base=self.base, A_row_start=self.var_row_start, k=self.k,
                                 m=self.m, H=self.H, K=self.K, neg_lam=self.neg_lam)
                for _ in range(self.n_chunk)]
        return pkts, list(range(self.n_chunk))


class LF3CChunk(_ChunkBase):
    """C-side Freivalds (per head): cid = base + h, coef = -lam[h*m+i]*rho[h*n+j]."""
    def __init__(self, m, n, H, n_chunk, ELL, base, var_row_start, chunk_lo):
        self.m, self.n, self.H = m, n, H
        self.total = m * H * n
        self._init_common(n_chunk, ELL, base, var_row_start, chunk_lo)
        g = torch.Generator(device="cuda").manual_seed(1)
        self.lam = torch.randint(0, 2 ** 62, (H * m,), generator=g,
                                 dtype=torch.int64, device="cuda").to(torch.uint64)
        self.rho = torch.randint(0, 2 ** 62, (H * n,), generator=g,
                                 dtype=torch.int64, device="cuda").to(torch.uint64)

    def packets(self):
        pkts = [L2_FreivaldsLF3C(base=self.base, C_row_start=self.var_row_start, m=self.m,
                                 n=self.n, H=self.H, L=self.total, lam=self.lam, rho=self.rho)
                for _ in range(self.n_chunk)]
        return pkts, list(range(self.n_chunk))


class S2OChunk(_ChunkBase):
    """StrideManyToOne: cid = base + flat//stride, scalar coef (high reuse)."""
    def __init__(self, L, stride, n_chunk, ELL, base, var_row_start, chunk_lo):
        self.L, self.stride = L, stride
        self._init_common(n_chunk, ELL, base, var_row_start, chunk_lo)

    def packets(self):
        pkts = [L2_StrideManyToOneScalar(base=self.base, var_row_start=self.var_row_start,
                                         L=self.L, stride=self.stride, coef=self.coef)
                for _ in range(self.n_chunk)]
        return pkts, list(range(self.n_chunk))


def reference_lf2a(c):
    pkts, lrows = c.packets()
    t, cid, v = _expand_freivalds_lf2a(pkts, lrows, c.chunk_lo, c.ELL)
    return _spmv_reference(t, cid, v, c.n_out, c.seed, c.label).view(c.n_chunk, c.ELL)


def reference_lf3c(c):
    pkts, lrows = c.packets()
    t, cid, v = _expand_freivalds_lf3c(pkts, lrows, c.chunk_lo, c.ELL)
    return _spmv_reference(t, cid, v, c.n_out, c.seed, c.label).view(c.n_chunk, c.ELL)


def reference_s2o(c):
    pkts, lrows = c.packets()
    t, cid, v = _expand_stride_many_to_one(pkts, lrows, c.chunk_lo, c.ELL)
    return _spmv_reference(t, cid, v, c.n_out, c.seed, c.label).view(c.n_chunk, c.ELL)


class PSVChunk(_ChunkBase):
    """PerSlotVector: cid = base + flat, coef = coef_vec[flat] (length-L vector)."""
    def __init__(self, L, n_chunk, ELL, base, var_row_start, chunk_lo):
        self.L = L
        self._init_common(n_chunk, ELL, base, var_row_start, chunk_lo)
        g = torch.Generator(device="cuda").manual_seed(2)
        self.coef_vec = torch.randint(0, 2 ** 62, (L,), generator=g,
                                      dtype=torch.int64, device="cuda").to(torch.uint64)

    def packets(self):
        pkts = [L2_PerSlotVector(base=self.base, var_row_start=self.var_row_start,
                                 L=self.L, coef_vec=self.coef_vec)
                for _ in range(self.n_chunk)]
        return pkts, list(range(self.n_chunk))


class RSVChunk(_ChunkBase):
    """RowSumPerSlotVector: cid = base + flat//stride, coef = coef_vec[flat%stride]."""
    def __init__(self, L, stride, n_chunk, ELL, base, var_row_start, chunk_lo):
        self.L, self.stride = L, stride
        self._init_common(n_chunk, ELL, base, var_row_start, chunk_lo)
        g = torch.Generator(device="cuda").manual_seed(2)
        self.coef_vec = torch.randint(0, 2 ** 62, (stride,), generator=g,
                                      dtype=torch.int64, device="cuda").to(torch.uint64)

    def packets(self):
        pkts = [L2_RowSumPerSlotVector(base=self.base, var_row_start=self.var_row_start,
                                       L=self.L, stride=self.stride, coef_vec=self.coef_vec)
                for _ in range(self.n_chunk)]
        return pkts, list(range(self.n_chunk))


class RoPEXRotChunk(_ChunkBase):
    """RoPEXRot: each x_rot slot -> one cid (base + 2*pair_t + e_self), coef 1.
    L = SEQ*H*d_h; pick SEQ so L >= n_chunk*ELL to keep every slot valid."""
    def __init__(self, SEQ, H, d_h, n_chunk, ELL, base, var_row_start, chunk_lo):
        assert d_h % 2 == 0, "d_h must be even (RoPE pairs)"
        self.SEQ, self.H, self.d_h = SEQ, H, d_h
        self.L = SEQ * H * d_h
        self._init_common(n_chunk, ELL, base, var_row_start, chunk_lo)

    def packets(self):
        pkts = [L2_RoPEXRot(base=self.base, x_rot_row_start=self.var_row_start,
                            SEQ=self.SEQ, H=self.H, d_h=self.d_h, L=self.L)
                for _ in range(self.n_chunk)]
        return pkts, list(range(self.n_chunk))


class TransposeO2MChunk(_ChunkBase):
    """Transposed fan-out: each slot sums `fan` cids at transposed positions."""
    def __init__(self, rows, cols, fan, n_chunk, ELL, base, var_row_start, chunk_lo):
        self.rows, self.cols, self.fan = rows, cols, fan
        self.L = rows * cols
        self._init_common(n_chunk, ELL, base, var_row_start, chunk_lo)

    def packets(self):
        pkts = [L2_TransposeO2MScalar(base=self.base, var_row_start=self.var_row_start,
                                      L=self.L, rows=self.rows, cols=self.cols,
                                      fan=self.fan, coef=self.coef)
                for _ in range(self.n_chunk)]
        return pkts, list(range(self.n_chunk))


class CausalIdChunk(_ChunkBase):
    """Causal identity: masked iff j > i_qry. M=SEQ keys, B=SEQ*H rows."""
    def __init__(self, SEQ, H, n_chunk, ELL, base, var_row_start, chunk_lo):
        self.SEQ, self.H, self.M = SEQ, H, SEQ
        self.B = SEQ * H
        self.L = self.B * self.M
        self._init_common(n_chunk, ELL, base, var_row_start, chunk_lo)

    def packets(self):
        pkts = [L2_CausalFilteredIdScalar(base=self.base, var_row_start=self.var_row_start,
                                          L=self.L, M=self.M, H=self.H, coef=self.coef)
                for _ in range(self.n_chunk)]
        return pkts, list(range(self.n_chunk))


class CausalC2Chunk(_ChunkBase):
    """Causal c2 ragged fan-sum: slot b sums (i_qry+1) cids. B=SEQ*H rows.
    B is small (one variable), so only the first B slots of the chunk are active;
    the rest are guard-zeroed. This is a correctness gate, not a perf-representative
    shape (most threads early-exit)."""
    def __init__(self, SEQ, H, n_chunk, ELL, base, c2_row_start, chunk_lo):
        self.SEQ, self.H = SEQ, H
        self.B = SEQ * H
        self._init_common(n_chunk, ELL, base, c2_row_start, chunk_lo)

    def packets(self):
        pkts = [L2_CausalFilteredC2Stride(base=self.base, c2_row_start=self.var_row_start,
                                          B=self.B, H=self.H, coef=self.coef)
                for _ in range(self.n_chunk)]
        return pkts, list(range(self.n_chunk))


class RoPEXChunk(_ChunkBase):
    """RoPEX: each x slot -> two cids, coefs +-cos/+-sin selected by e_self."""
    def __init__(self, SEQ, H, d_h, n_chunk, ELL, base, var_row_start, chunk_lo):
        assert d_h % 2 == 0, "d_h must be even (RoPE pairs)"
        self.SEQ, self.H, self.d_h = SEQ, H, d_h
        self.half = d_h // 2
        self.L = SEQ * H * d_h
        self._init_common(n_chunk, ELL, base, var_row_start, chunk_lo)
        g = torch.Generator(device="cuda").manual_seed(3)
        ncoef = SEQ * self.half
        self.cos_t = torch.randint(0, 2 ** 62, (ncoef,), generator=g,
                                   dtype=torch.int64, device="cuda").to(torch.uint64)
        self.sin_t = torch.randint(0, 2 ** 62, (ncoef,), generator=g,
                                   dtype=torch.int64, device="cuda").to(torch.uint64)

    def packets(self):
        pkts = [L2_RoPEX(base=self.base, x_row_start=self.var_row_start, SEQ=self.SEQ,
                         H=self.H, d_h=self.d_h, L=self.L, cos_t=self.cos_t, sin_t=self.sin_t)
                for _ in range(self.n_chunk)]
        return pkts, list(range(self.n_chunk))


def reference_psv(c):
    pkts, lrows = c.packets()
    t, cid, v = _expand_per_slot_vector(pkts, lrows, c.chunk_lo, c.ELL)
    return _spmv_reference(t, cid, v, c.n_out, c.seed, c.label).view(c.n_chunk, c.ELL)


def reference_rsv(c):
    pkts, lrows = c.packets()
    t, cid, v = _expand_row_sum_per_slot_vector(pkts, lrows, c.chunk_lo, c.ELL)
    return _spmv_reference(t, cid, v, c.n_out, c.seed, c.label).view(c.n_chunk, c.ELL)


def reference_ropexrot(c):
    pkts, lrows = c.packets()
    t, cid, v = _expand_rope_xrot(pkts, lrows, c.chunk_lo, c.ELL)
    return _spmv_reference(t, cid, v, c.n_out, c.seed, c.label).view(c.n_chunk, c.ELL)


def reference_transpose(c):
    pkts, lrows = c.packets()
    t, cid, v = _expand_transpose_o2m(pkts, lrows, c.chunk_lo, c.ELL)
    return _spmv_reference(t, cid, v, c.n_out, c.seed, c.label).view(c.n_chunk, c.ELL)


def reference_causal_id(c):
    pkts, lrows = c.packets()
    t, cid, v = _expand_causal_filtered_id(pkts, lrows, c.chunk_lo, c.ELL)
    return _spmv_reference(t, cid, v, c.n_out, c.seed, c.label).view(c.n_chunk, c.ELL)


def reference_causal_c2(c):
    pkts, lrows = c.packets()
    t, cid, v = _expand_causal_filtered_c2(pkts, lrows, c.chunk_lo, c.ELL)
    return _spmv_reference(t, cid, v, c.n_out, c.seed, c.label).view(c.n_chunk, c.ELL)


def reference_ropex(c):
    pkts, lrows = c.packets()
    t, cid, v = _expand_rope_x(pkts, lrows, c.chunk_lo, c.ELL)
    return _spmv_reference(t, cid, v, c.n_out, c.seed, c.label).view(c.n_chunk, c.ELL)


# ---------------------------------------------------------------------------
# Variants (each returns the (n_chunk, ELL) chunk_rTA)
# ---------------------------------------------------------------------------

def _args(c):
    return (c.seed, c.label, c.neg_rho, c.base, c.B_row_start, c.chunk_lo,
            c.k, c.n, c.H, c.K, int(c.transpose_b), c.n_chunk, c.ELL, c.total)


def v_perslot(c):
    return _module().lf1b_perslot(*_args(c)).view(c.n_chunk, c.ELL)


def v_warpcid(c):
    return _module().lf1b_warpcid(*_args(c)).view(c.n_chunk, c.ELL)


def v_precompute(c):
    # COMBINED: hash the <= k distinct cids, then gather. Both timed together,
    # so the per-cid hashing + both kernel launches are inside the measurement.
    m = _module()
    table = m.challenge_range(c.seed, c.label, c.base, c.k)
    out = m.lf1b_gather(table, c.neg_rho, c.B_row_start, c.chunk_lo,
                        c.k, c.n, c.H, c.K, int(c.transpose_b),
                        c.n_chunk, c.ELL, c.total)
    return out.view(c.n_chunk, c.ELL)


def v_perrow(c):
    return _module().lf1b_perrow(*_args(c)).view(c.n_chunk, c.ELL)


def _multirow(rpt):
    def fn(c):
        return _module().lf1b_multirow(*_args(c), rpt).view(c.n_chunk, c.ELL)
    return fn


# --- identity / stride variants ---

def v_id_perslot(c):
    return _module().id_perslot(c.seed, c.label, c.coef, c.base, c.var_row_start,
                                c.chunk_lo, c.L, c.n_chunk, c.ELL).view(c.n_chunk, c.ELL)


def v_id_precompute(c):
    # COMBINED: hash the chunk's distinct cids (cid = base + flat, contiguous),
    # then gather. Both timed. For identity #cids = #slots, so no reuse to exploit.
    m = _module()
    n_distinct = min(c.L, c.n_out)
    table = m.challenge_range(c.seed, c.label, c.base, n_distinct)
    return m.id_gather(table, c.coef, c.var_row_start, c.chunk_lo, c.L,
                       c.n_chunk, c.ELL).view(c.n_chunk, c.ELL)


def v_stride_perslot(c):
    return _module().stride_perslot(c.seed, c.label, c.coef, c.base, c.var_row_start,
                                    c.chunk_lo, c.L, c.stride, c.n_chunk,
                                    c.ELL).view(c.n_chunk, c.ELL)


def v_lf2a_perslot(c):
    return _module().lf2a_perslot(c.seed, c.label, c.neg_lam, c.base, c.var_row_start,
                                  c.chunk_lo, c.k, c.m, c.K, c.n_chunk, c.ELL,
                                  c.total).view(c.n_chunk, c.ELL)


def v_lf3c_perslot(c):
    return _module().lf3c_perslot(c.seed, c.label, c.lam, c.rho, c.base, c.var_row_start,
                                  c.chunk_lo, c.m, c.n, c.H, c.n_chunk, c.ELL,
                                  c.total).view(c.n_chunk, c.ELL)


def v_s2o_perslot(c):
    return _module().s2o_perslot(c.seed, c.label, c.coef, c.base, c.var_row_start,
                                 c.chunk_lo, c.stride, c.L, c.n_chunk,
                                 c.ELL).view(c.n_chunk, c.ELL)


def v_psv_perslot(c):
    return _module().psv_perslot(c.seed, c.label, c.coef_vec, c.base, c.var_row_start,
                                 c.chunk_lo, c.L, c.n_chunk, c.ELL).view(c.n_chunk, c.ELL)


def v_rsv_perslot(c):
    return _module().rsv_perslot(c.seed, c.label, c.coef_vec, c.base, c.var_row_start,
                                 c.chunk_lo, c.stride, c.L, c.n_chunk,
                                 c.ELL).view(c.n_chunk, c.ELL)


def v_ropexrot_perslot(c):
    return _module().ropexrot_perslot(c.seed, c.label, c.base, c.var_row_start,
                                      c.chunk_lo, c.H, c.d_h, c.L, c.n_chunk,
                                      c.ELL).view(c.n_chunk, c.ELL)


def v_transpose_perslot(c):
    return _module().transpose_o2m_perslot(c.seed, c.label, c.coef, c.base, c.var_row_start,
                                           c.chunk_lo, c.L, c.rows, c.cols, c.fan,
                                           c.n_chunk, c.ELL).view(c.n_chunk, c.ELL)


def v_causal_id_perslot(c):
    return _module().causal_id_perslot(c.seed, c.label, c.coef, c.base, c.var_row_start,
                                       c.chunk_lo, c.L, c.M, c.H, c.n_chunk,
                                       c.ELL).view(c.n_chunk, c.ELL)


def v_causal_c2_perslot(c):
    return _module().causal_c2_perslot(c.seed, c.label, c.coef, c.base, c.var_row_start,
                                       c.chunk_lo, c.B, c.H, c.n_chunk,
                                       c.ELL).view(c.n_chunk, c.ELL)


def v_ropex_perslot(c):
    return _module().ropex_perslot(c.seed, c.label, c.cos_t, c.sin_t, c.base, c.var_row_start,
                                   c.chunk_lo, c.H, c.d_h, c.L, c.n_chunk,
                                   c.ELL).view(c.n_chunk, c.ELL)


# --- precompute (gather) variants for the high-reuse families ---
# COMBINED: hash the family's distinct cids once (challenge_range), then a pure
# gather kernel. Both timed together, so the per-cid hashing is not hidden.

def v_lf2a_precompute(c):
    m = _module()
    table = m.challenge_range(c.seed, c.label, c.base, c.k)        # k distinct cids
    return m.lf2a_gather(table, c.neg_lam, c.var_row_start, c.chunk_lo, c.k, c.m, c.K,
                         c.n_chunk, c.ELL, c.total).view(c.n_chunk, c.ELL)


def v_lf3c_precompute(c):
    m = _module()
    table = m.challenge_range(c.seed, c.label, c.base, c.H)        # H distinct cids
    return m.lf3c_gather(table, c.lam, c.rho, c.var_row_start, c.chunk_lo, c.m, c.n, c.H,
                         c.n_chunk, c.ELL, c.total).view(c.n_chunk, c.ELL)


def v_s2o_precompute(c):
    m = _module()
    n_distinct = (c.L + c.stride - 1) // c.stride
    table = m.challenge_range(c.seed, c.label, c.base, n_distinct)
    return m.s2o_gather(table, c.coef, c.var_row_start, c.chunk_lo, c.stride, c.L,
                        c.n_chunk, c.ELL).view(c.n_chunk, c.ELL)


def v_rsv_precompute(c):
    m = _module()
    n_distinct = (c.L + c.stride - 1) // c.stride
    table = m.challenge_range(c.seed, c.label, c.base, n_distinct)
    return m.rsv_gather(table, c.coef_vec, c.var_row_start, c.chunk_lo, c.stride, c.L,
                        c.n_chunk, c.ELL).view(c.n_chunk, c.ELL)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# P2: ConstraintBand interface (one band per (variable, family)).
# contribution(row_lo, row_hi, seed, label, ELL) -> dense (row_hi-row_lo, ELL),
# delegating to the family's validated fused kernel reframed to an arbitrary row
# range (chunk_lo=row_lo, n_chunk=row_hi-row_lo). Structural metadata + per-claim
# coef tensors live on the band (here, the chunk); challenges are passed per call
# (a band carries no challenges of its own). See analysis/qlin-family-object-reorg.md
# (the band object; §1/§3/§11) and qlin-fold-reorg-plan.md P2.
# ---------------------------------------------------------------------------

class _RowRangeView:
    """Presents a chunk as the row range [row_lo, row_hi) with externally supplied
    seed/label/ELL, so the validated v_* launchers run over an arbitrary band of a
    variable's rows. Structural metadata + coef tensors forward from the chunk;
    chunk_lo / n_chunk / seed / label / ELL are the per-call overrides."""
    def __init__(self, chunk, row_lo, row_hi, seed, label, ELL):
        self._c = chunk
        self.chunk_lo = row_lo
        self.n_chunk = row_hi - row_lo
        self.seed = seed
        self.label = label
        self.ELL = ELL

    def __getattr__(self, name):
        return getattr(self._c, name)


# Per-family cid-span dispatch (the recommended kernel from the table in
# qlin-fold-kernels-HANDOFF.md): precompute for the high-reuse families (a cid is
# shared by many slots), per-slot for 1:1 / anti-reuse families. Keyed on the
# family role, NOT on "is a precompute variant present" -- identity HAS a gather
# variant but is 1:1, so per-slot stays optimal there.
_DISPATCH = {
    "freivalds": "precompute", "lf2a": "precompute", "lf3c": "precompute",
    "s2o": "precompute", "rsv": "precompute",
    "identity": "perslot", "stride": "perslot", "psv": "perslot",
    "ropexrot": "perslot", "transpose": "perslot", "causal_id": "perslot",
    "causal_c2": "perslot", "ropex": "perslot",
}


class ConstraintBand:
    """One variable's contribution to one constraint family. Generates a dense
    (n, ELL) block for any row range on demand. contribution() selects the
    recommended kernel (precompute / per-slot) from the family's cid-span via
    _DISPATCH, then runs it over the row range; the per-role logic stays in the
    validated fused kernels (delegated through the variant fns)."""
    def __init__(self, role, chunk, variants):
        self.role = role
        self._chunk = chunk
        self._variants = variants
        want = _DISPATCH.get(role, "perslot")
        self.dispatch = want if want in variants else "perslot"

    def contribution(self, row_lo, row_hi, seed, label, ELL):
        view = _RowRangeView(self._chunk, row_lo, row_hi, seed, label, ELL)
        return self._variants[self.dispatch](view)


def _check_band_contract(header, c, ref, variants, family):
    """Gate the ConstraintBand.contribution(row_lo, row_hi, ...) contract: for
    several row ranges, contribution must equal the reference's matching rows.
    Exercises the per-variable row-range plumbing the per-chunk kernels never hit,
    and uses the band's own cid-span dispatch (precompute / per-slot)."""
    print(header)
    band = ConstraintBand(family, c, variants)
    nch = c.n_chunk
    ranges = [(0, nch), (nch // 4, nch // 2), (nch - 1, nch)]
    allok = True
    for lo, hi in ranges:
        out = band.contribution(lo, hi, c.seed, c.label, c.ELL)
        ok = tuple(out.shape) == (hi - lo, c.ELL) and bool(torch.equal(out, ref[lo:hi]))
        allok = allok and ok
        print(f"  band[{band.dispatch}] rows[{lo:>4},{hi:>4})  {'OK ' if ok else 'BAD'}  "
              f"shape={tuple(out.shape)}  == ref[{lo}:{hi}]")
        if not ok:
            diff = int((out != ref[lo:hi]).sum().item())
            print(f"      !! {diff:,} slots differ")
    print(f"  => ConstraintBand.contribution contract (dispatch={band.dispatch}): "
          f"{'PASS' if allok else 'FAIL'}")


def time_fn(fn, iters, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms per call


def run(header, c, reference_fn, variants, iters, bands=False, family=None):
    ref = reference_fn(c)
    if bands:
        _check_band_contract(header, c, ref, variants, family)
        return
    ref_ms = time_fn(lambda: reference_fn(c), iters)
    n_out = c.n_out
    out_bytes = n_out * 8

    def line(name, ms, ok):
        ns_slot = ms * 1e6 / n_out
        gbs = out_bytes / (ms / 1e3) / 1e9
        tag = "OK " if ok else "BAD"
        print(f"  {name:14s} {tag}  {ms:8.3f} ms  {ns_slot:7.3f} ns/slot  {gbs:8.1f} GB/s")

    print(header)
    line("reference", ref_ms, True)
    for name, fn in variants.items():
        out = fn(c)
        ok = bool(torch.equal(out, ref))
        ms = time_fn(lambda: fn(c), iters) if ok else float("nan")
        line(name, ms, ok)
        if not ok:
            diff = (out != ref).sum().item()
            print(f"      !! {diff:,} / {n_out:,} slots differ vs reference")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="freivalds",
                    choices=["freivalds", "identity", "stride", "lf2a", "lf3c", "s2o",
                             "psv", "rsv", "ropexrot", "transpose", "causal_id",
                             "causal_c2", "ropex"])
    ap.add_argument("--m", type=int, default=4096)
    ap.add_argument("--k", type=int, default=4096)
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--H", type=int, default=1)
    ap.add_argument("--transpose_b", action="store_true")
    ap.add_argument("--L", type=int, default=0, help="variable length (0 = n_chunk*ELL)")
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--d_h", type=int, default=128, help="rope head dim (even)")
    ap.add_argument("--SEQ", type=int, default=0,
                    help="rope/causal seq len (0 = auto)")
    ap.add_argument("--fan", type=int, default=16, help="transpose fan-out width")
    ap.add_argument("--cols", type=int, default=512, help="transpose source minor dim")
    ap.add_argument("--n_chunk", type=int, default=256)
    ap.add_argument("--ELL", type=int, default=ELL_DEFAULT)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--variant", default="all")
    ap.add_argument("--multirow", default="2,4,8")
    ap.add_argument("--bands", action="store_true",
                    help="check the ConstraintBand.contribution(row_lo,row_hi) "
                         "contract (P2) instead of timing variants")
    args = ap.parse_args()

    L = args.L or (args.n_chunk * args.ELL)
    if args.family == "freivalds":
        c = Chunk(k=args.k, n=args.n, H=args.H, transpose_b=args.transpose_b,
                  n_chunk=args.n_chunk, ELL=args.ELL, base=1, B_row_start=0, chunk_lo=0)
        ref = reference_freivalds
        variants = {"perslot": v_perslot, "warpcid": v_warpcid,
                    "precompute": v_precompute, "perrow": v_perrow}
        for r in [int(x) for x in args.multirow.split(",") if x]:
            variants[f"multirow{r}"] = _multirow(r)
        header = (f"[freivalds] k={c.k} n={c.n} H={c.H} transpose_b={c.transpose_b} "
                  f"n_chunk={c.n_chunk} ELL={c.ELL} n_out={c.n_out:,}")
    elif args.family == "identity":
        c = IdentityChunk(L=L, n_chunk=args.n_chunk, ELL=args.ELL,
                          base=1, var_row_start=0, chunk_lo=0)
        ref = reference_identity
        variants = {"perslot": v_id_perslot, "precompute": v_id_precompute}
        header = (f"[identity] L={c.L} n_chunk={c.n_chunk} ELL={c.ELL} "
                  f"n_out={c.n_out:,}  (#cids = #slots, no reuse)")
    elif args.family == "stride":
        c = StrideChunk(L=L, stride=args.stride, n_chunk=args.n_chunk, ELL=args.ELL,
                        base=1, var_row_start=0, chunk_lo=0)
        ref = reference_stride
        variants = {"perslot": v_stride_perslot}
        header = (f"[stride] L={c.L} stride={c.stride} n_chunk={c.n_chunk} "
                  f"ELL={c.ELL} n_out={c.n_out:,}  ({c.stride} hashes/slot, anti-reuse)")
    elif args.family == "lf2a":
        c = LF2AChunk(m=args.m, k=args.k, H=args.H, n_chunk=args.n_chunk, ELL=args.ELL,
                      base=1, var_row_start=0, chunk_lo=0)
        ref = reference_lf2a
        variants = {"perslot": v_lf2a_perslot, "precompute": v_lf2a_precompute}
        header = (f"[lf2a] m={c.m} k={c.k} H={c.H} n_chunk={c.n_chunk} ELL={c.ELL} "
                  f"n_out={c.n_out:,} total={c.total:,}")
    elif args.family == "lf3c":
        c = LF3CChunk(m=args.m, n=args.n, H=args.H, n_chunk=args.n_chunk, ELL=args.ELL,
                      base=1, var_row_start=0, chunk_lo=0)
        ref = reference_lf3c
        variants = {"perslot": v_lf3c_perslot, "precompute": v_lf3c_precompute}
        header = (f"[lf3c] m={c.m} n={c.n} H={c.H} n_chunk={c.n_chunk} ELL={c.ELL} "
                  f"n_out={c.n_out:,} total={c.total:,}")
    elif args.family == "s2o":
        c = S2OChunk(L=L, stride=args.stride, n_chunk=args.n_chunk, ELL=args.ELL,
                     base=1, var_row_start=0, chunk_lo=0)
        ref = reference_s2o
        variants = {"perslot": v_s2o_perslot, "precompute": v_s2o_precompute}
        header = (f"[s2o] L={c.L} stride={c.stride} n_chunk={c.n_chunk} ELL={c.ELL} "
                  f"n_out={c.n_out:,}  (cid=base+flat//stride, high reuse)")
    elif args.family == "psv":
        c = PSVChunk(L=L, n_chunk=args.n_chunk, ELL=args.ELL,
                     base=1, var_row_start=0, chunk_lo=0)
        ref = reference_psv
        variants = {"perslot": v_psv_perslot}
        header = (f"[psv] L={c.L} n_chunk={c.n_chunk} ELL={c.ELL} n_out={c.n_out:,}  "
                  f"(cid=base+flat, vec coef)")
    elif args.family == "ropexrot":
        seq = args.SEQ or ((args.n_chunk * args.ELL + args.H * args.d_h - 1)
                           // (args.H * args.d_h))
        c = RoPEXRotChunk(SEQ=seq, H=args.H, d_h=args.d_h, n_chunk=args.n_chunk,
                          ELL=args.ELL, base=1, var_row_start=0, chunk_lo=0)
        ref = reference_ropexrot
        variants = {"perslot": v_ropexrot_perslot}
        header = (f"[ropexrot] SEQ={c.SEQ} H={c.H} d_h={c.d_h} n_chunk={c.n_chunk} "
                  f"ELL={c.ELL} n_out={c.n_out:,} L={c.L:,}")
    elif args.family == "transpose":
        cols = args.cols
        rows = (args.n_chunk * args.ELL + cols - 1) // cols
        c = TransposeO2MChunk(rows=rows, cols=cols, fan=args.fan, n_chunk=args.n_chunk,
                              ELL=args.ELL, base=1, var_row_start=0, chunk_lo=0)
        ref = reference_transpose
        variants = {"perslot": v_transpose_perslot}
        header = (f"[transpose] rows={c.rows} cols={c.cols} fan={c.fan} n_chunk={c.n_chunk} "
                  f"ELL={c.ELL} n_out={c.n_out:,} L={c.L:,}  ({c.fan} hashes/slot)")
    elif args.family == "causal_id":
        target = (args.n_chunk * args.ELL + args.H - 1) // args.H
        seq = args.SEQ
        if not seq:
            seq = math.isqrt(target)
            while seq * seq < target:
                seq += 1
        c = CausalIdChunk(SEQ=seq, H=args.H, n_chunk=args.n_chunk, ELL=args.ELL,
                          base=1, var_row_start=0, chunk_lo=0)
        ref = reference_causal_id
        variants = {"perslot": v_causal_id_perslot}
        header = (f"[causal_id] SEQ={c.SEQ} M={c.M} H={c.H} B={c.B} n_chunk={c.n_chunk} "
                  f"ELL={c.ELL} n_out={c.n_out:,} L={c.L:,}  (masked iff j>i_qry)")
    elif args.family == "causal_c2":
        seq = args.SEQ or 512
        c = CausalC2Chunk(SEQ=seq, H=args.H, n_chunk=args.n_chunk, ELL=args.ELL,
                          base=1, c2_row_start=0, chunk_lo=0)
        ref = reference_causal_c2
        variants = {"perslot": v_causal_c2_perslot}
        header = (f"[causal_c2] SEQ={c.SEQ} H={c.H} B={c.B} n_chunk={c.n_chunk} ELL={c.ELL} "
                  f"n_out={c.n_out:,}  (ragged fan-sum; active region = B={c.B} slots)")
    elif args.family == "ropex":
        seq = args.SEQ or ((args.n_chunk * args.ELL + args.H * args.d_h - 1)
                           // (args.H * args.d_h))
        c = RoPEXChunk(SEQ=seq, H=args.H, d_h=args.d_h, n_chunk=args.n_chunk, ELL=args.ELL,
                       base=1, var_row_start=0, chunk_lo=0)
        ref = reference_ropex
        variants = {"perslot": v_ropex_perslot}
        header = (f"[ropex] SEQ={c.SEQ} H={c.H} d_h={c.d_h} n_chunk={c.n_chunk} ELL={c.ELL} "
                  f"n_out={c.n_out:,} L={c.L:,}  (2 cids/slot, +-cos/sin)")
    else:  # rsv
        c = RSVChunk(L=L, stride=args.stride, n_chunk=args.n_chunk, ELL=args.ELL,
                     base=1, var_row_start=0, chunk_lo=0)
        ref = reference_rsv
        variants = {"perslot": v_rsv_perslot, "precompute": v_rsv_precompute}
        header = (f"[rsv] L={c.L} stride={c.stride} n_chunk={c.n_chunk} ELL={c.ELL} "
                  f"n_out={c.n_out:,}  (cid=base+flat//stride, cyclic vec coef)")

    if args.variant != "all":
        names = set(args.variant.split(","))
        variants = {k: v for k, v in variants.items() if k in names}
    run(header, c, ref, variants, args.iters, bands=args.bands, family=args.family)


if __name__ == "__main__":
    main()
