// BLAKE3 column hashing for Ligero commits.
//
// Two implementations:
//
//   hash_columns_simple — one thread per column. Walks the entire column
//     chunk-by-chunk, gathering strided u64s out of the row-major encoded
//     matrix. Works without external state, but assumes the full encoded
//     matrix fits in memory.
//
//   hash_columns_streamed — chunked variant designed for the case where the
//     encoded matrix is too large to hold at once. The caller materializes
//     one chunk (STREAM_CHUNK_ROWS rows × N cols) at a time and feeds it to
//     k_stream_update_chunk; per-column BLAKE3 state lives in
//     ColStreamState[N] (~280 bytes per column). After all chunks,
//     k_stream_finalize collapses the per-column merge stacks.
//
// Both kernels lifted from deprecated/spark-bench/cuda/commit_weights.cu
// where they ran end-to-end against Llama 2 7B q_proj scale. Original
// comments preserved.

#pragma once

#include <cstdint>
#include "blake3_compress.cuh"
#include "blake3_hash.cuh"

namespace b3_cols {

constexpr int STREAM_CHUNK_ROWS = 128;
constexpr int STREAM_STACK_MAX  = 54;     // BLAKE3 spec depth (2^54 chunks).
// WAS 16 — off-by-one overflow: the push-before-merge transiently needs index
// B at chunk_counter==2^B, so 16 overflowed at 2^16 chunks = 2^23 rows,
// scribbling the adjacent column's state. Surfaced as a GC-time segfault on
// proofs whose phase-1 row count crossed 8,388,608 (>=~6 Maverick layers).

// Per-column state held across chunks. Excludes the within-chunk CV (kept in
// registers in process_chunk).
struct ColStreamState {
    uint64_t chunk_counter;
    int      stack_ptr;
    uint32_t stack[STREAM_STACK_MAX][8];
};

// One thread per column. Walks the entire column chunk-by-chunk, gathering
// strided u64s out of the row-major encoded matrix. Same state machine as
// b3::hash_bytes minus the byte-buffer abstraction.
__global__ void k_hash_columns_simple(
    const uint64_t* __restrict__ encoded,
    int m_in,
    int N,
    uint32_t* __restrict__ digests
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= N) return;
    int len_bytes = m_in * 8;
    int n_chunks = (len_bytes + b3::CHUNK_LEN - 1) / b3::CHUNK_LEN;
    bool single_chunk = (n_chunks == 1);

    uint32_t stack[b3::STACK_MAX][8];
    int sp = 0;
    uint64_t chunk_counter = 0;
    constexpr int U64_PER_CHUNK = b3::CHUNK_LEN / 8;
    uint32_t out[8];

    for (int c = 0; c < n_chunks; ++c) {
        bool is_last_chunk = (c == n_chunks - 1);
        int u64_start = c * U64_PER_CHUNK;
        int u64_end   = is_last_chunk ? m_in : u64_start + U64_PER_CHUNK;
        int chunk_bytes = (u64_end - u64_start) * 8;

        // Block-by-block compression within this chunk.
        int n_blocks_full = chunk_bytes / 64;
        int rem           = chunk_bytes - n_blocks_full * 64;
        int n_blocks      = n_blocks_full + ((rem > 0 || chunk_bytes == 0) ? 1 : 0);

        uint32_t cv[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) cv[i] = b3::IV[i];

        uint32_t chunk_cv[8];

        for (int b = 0; b < n_blocks; ++b) {
            bool is_first = (b == 0);
            bool is_last  = (b == n_blocks - 1);
            int block_bytes = (b < n_blocks_full) ? 64 : rem;
            if (chunk_bytes == 0) block_bytes = 0;

            uint32_t msg[16] = {0};
            int u64_block_start = u64_start + b * 8;
            int u64_block_count = (b < n_blocks_full) ? 8 : (u64_end - u64_block_start);
            if (u64_block_count > 8) u64_block_count = 8;
            if (u64_block_count < 0) u64_block_count = 0;

            #pragma unroll
            for (int r = 0; r < 8; ++r) {
                if (r < u64_block_count) {
                    uint64_t v = encoded[(size_t)(u64_block_start + r) * N + col];
                    msg[2 * r]     = (uint32_t)v;
                    msg[2 * r + 1] = (uint32_t)(v >> 32);
                }
            }

            uint32_t flags = 0;
            if (is_first) flags |= b3::CHUNK_START;
            if (is_last)  flags |= b3::CHUNK_END;
            if (is_last && single_chunk) flags |= b3::ROOT;

            uint32_t out16[16];
            b3::compress(cv, msg, chunk_counter, (uint32_t)block_bytes, flags, out16);

            if (is_last) {
                #pragma unroll
                for (int i = 0; i < 8; ++i) chunk_cv[i] = out16[i];
            } else {
                #pragma unroll
                for (int i = 0; i < 8; ++i) cv[i] = out16[i];
            }
        }

        if (single_chunk) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) out[i] = chunk_cv[i];
            goto write_digest;
        }

        // Push chunk_cv to merge stack, eager-merge while chunk count even.
        #pragma unroll
        for (int i = 0; i < 8; ++i) stack[sp][i] = chunk_cv[i];
        ++sp;
        ++chunk_counter;

        if (!is_last_chunk) {
            uint64_t cnt = chunk_counter;
            while ((cnt & 1u) == 0u && sp >= 2) {
                cnt >>= 1;
                uint32_t left[8], right[8];
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    right[i] = stack[sp - 1][i];
                    left[i]  = stack[sp - 2][i];
                }
                sp -= 2;
                uint32_t parent[8];
                b3::parent_compress(left, right, /*is_root=*/false, parent);
                #pragma unroll
                for (int i = 0; i < 8; ++i) stack[sp][i] = parent[i];
                ++sp;
            }
        }
    }

    // Final tree merge — the last merge sets ROOT.
    while (sp > 1) {
        uint32_t left[8], right[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            right[i] = stack[sp - 1][i];
            left[i]  = stack[sp - 2][i];
        }
        sp -= 2;
        bool is_root = (sp == 0);
        uint32_t parent[8];
        b3::parent_compress(left, right, is_root, parent);
        #pragma unroll
        for (int i = 0; i < 8; ++i) stack[sp][i] = parent[i];
        ++sp;
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i) out[i] = stack[0][i];

write_digest:
    #pragma unroll
    for (int i = 0; i < 8; ++i) digests[col * 8 + i] = out[i];
}

// ---------- Streaming column hash kernels ----------
//
// Caller materializes one chunk of STREAM_CHUNK_ROWS rows at a time and
// drives the state machine via k_stream_init → k_stream_update_chunk (one
// per chunk) → k_stream_finalize. Output is identical to k_hash_columns_simple
// on the concatenated matrix; this form is for cases where the full encoded
// matrix doesn't fit in memory.

__global__ void k_stream_init(ColStreamState* states, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= N) return;
    states[col].chunk_counter = 0;
    states[col].stack_ptr     = 0;
}

// Process exactly one chunk (≤ STREAM_CHUNK_ROWS rows) for every column.
// `chunk_buffer` is row-major (row * N + col).
//
// is_last_chunk:         this is the last chunk in the column's input
// is_single_chunk_total: total chunks is 1 (apply ROOT to the chunk's last block)
// digests:               output array for the single-chunk-total ROOT case
__global__ void k_stream_update_chunk(
    ColStreamState* __restrict__ states,
    const uint64_t* __restrict__ chunk_buffer,
    int N,
    int n_rows_in_chunk,
    bool is_last_chunk,
    bool is_single_chunk_total,
    uint32_t* __restrict__ digests
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= N) return;

    bool is_root = is_single_chunk_total && is_last_chunk;
    uint64_t chunk_counter = states[col].chunk_counter;

    int chunk_bytes = n_rows_in_chunk * 8;
    int n_blocks_full = chunk_bytes / 64;
    int rem           = chunk_bytes - n_blocks_full * 64;
    int n_blocks      = n_blocks_full + ((rem > 0 || chunk_bytes == 0) ? 1 : 0);

    uint32_t cv[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) cv[i] = b3::IV[i];

    uint32_t chunk_cv[8];

    for (int b = 0; b < n_blocks; ++b) {
        bool is_first = (b == 0);
        bool is_last  = (b == n_blocks - 1);
        int block_bytes = (b < n_blocks_full) ? 64 : rem;
        if (chunk_bytes == 0) block_bytes = 0;

        uint32_t msg[16] = {0};
        int row_start = b * 8;
        int row_count = (b < n_blocks_full) ? 8 : (n_rows_in_chunk - row_start);
        if (row_count > 8) row_count = 8;
        if (row_count < 0) row_count = 0;

        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            if (r < row_count) {
                uint64_t v = chunk_buffer[(size_t)(row_start + r) * N + col];
                msg[2 * r]     = (uint32_t)v;
                msg[2 * r + 1] = (uint32_t)(v >> 32);
            }
        }

        uint32_t flags = 0;
        if (is_first) flags |= b3::CHUNK_START;
        if (is_last)  flags |= b3::CHUNK_END;
        if (is_last && is_root) flags |= b3::ROOT;

        uint32_t out16[16];
        b3::compress(cv, msg, chunk_counter, (uint32_t)block_bytes, flags, out16);

        if (is_last) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) chunk_cv[i] = out16[i];
        } else {
            #pragma unroll
            for (int i = 0; i < 8; ++i) cv[i] = out16[i];
        }
    }

    if (is_root) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) digests[col * 8 + i] = chunk_cv[i];
        return;
    }

    int sp = states[col].stack_ptr;
    #pragma unroll
    for (int i = 0; i < 8; ++i) states[col].stack[sp][i] = chunk_cv[i];
    ++sp;
    ++chunk_counter;

    if (!is_last_chunk) {
        uint64_t cnt = chunk_counter;
        while ((cnt & 1u) == 0u && sp >= 2) {
            cnt >>= 1;
            uint32_t left[8], right[8];
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                right[i] = states[col].stack[sp - 1][i];
                left[i]  = states[col].stack[sp - 2][i];
            }
            sp -= 2;
            uint32_t parent[8];
            b3::parent_compress(left, right, /*is_root=*/false, parent);
            #pragma unroll
            for (int i = 0; i < 8; ++i) states[col].stack[sp][i] = parent[i];
            ++sp;
        }
    }

    states[col].stack_ptr     = sp;
    states[col].chunk_counter = chunk_counter;
}

// Final tree merge: collapse remaining CVs on each column's stack.
// The last merge sets ROOT. Skipped when total chunks == 1 (digest was
// already written by k_stream_update_chunk).
__global__ void k_stream_finalize(
    ColStreamState* __restrict__ states,
    uint32_t* __restrict__ digests,
    int N,
    bool is_single_chunk_total
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= N) return;
    if (is_single_chunk_total) return;

    int sp = states[col].stack_ptr;
    while (sp > 1) {
        uint32_t left[8], right[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            right[i] = states[col].stack[sp - 1][i];
            left[i]  = states[col].stack[sp - 2][i];
        }
        sp -= 2;
        bool is_root = (sp == 0);
        uint32_t parent[8];
        b3::parent_compress(left, right, is_root, parent);
        #pragma unroll
        for (int i = 0; i < 8; ++i) states[col].stack[sp][i] = parent[i];
        ++sp;
    }

    #pragma unroll
    for (int i = 0; i < 8; ++i) digests[col * 8 + i] = states[col].stack[0][i];
}

} // namespace b3_cols
