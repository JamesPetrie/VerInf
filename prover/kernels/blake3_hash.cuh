// BLAKE3 multi-chunk hashing on top of the compression primitive in
// blake3_compress.cuh.
//
// Builds the chunk + tree-merge state machine described in the BLAKE3
// specification:
//   - Input is split into chunks of CHUNK_LEN = 1024 bytes (last possibly short).
//   - Each chunk hashes its own block chain with chunk_counter set to the
//     chunk index. First block has CHUNK_START flag; last has CHUNK_END.
//   - Each chunk produces a 32-byte chaining value (CV).
//   - CVs form leaves of a binary merge tree. Each parent compression
//     concatenates left||right CVs as a 64-byte message and is run with
//     PARENT flag set.
//   - The very last (root) compression — single-chunk's last block, or
//     the final parent merge in the multi-chunk case — additionally has
//     ROOT flag set; its first 8 output words are the final 32-byte hash.
//
// Each thread runs an independent column hash. Per-thread CV stack is
// shallow (≤ 16) for any column size we care about (a 16-deep stack
// covers up to 2^16 chunks = 64 MB per column), so the stack can live in
// thread-local registers / local memory without measurable cost.

#pragma once

#include "blake3_compress.cuh"

namespace b3 {

// Maximum CV-stack depth, in entries of 8 u32 words. BLAKE3 spec depth (54)
// covers 2^54 chunks. WAS 16 (2^16 chunks = 2^23 rows): the push-before-merge
// transiently needs index B at the 2^B-chunk boundary, so 16 overflowed by one
// at exactly 2^16 chunks — a real OOB, latent here (small single-shot inputs)
// but fatal in the streaming column-hash at >=2^23 rows.
constexpr int STACK_MAX = 54;

// Pack up to 64 bytes from a buffer into a 16-word message block, zero-
// padding any unused tail. Caller passes the actual byte count via
// `block_bytes`.
__device__ __forceinline__ void pack_block(
    const uint8_t* src, int block_bytes, uint32_t msg[16]
) {
    #pragma unroll
    for (int i = 0; i < 16; ++i) msg[i] = 0;
    for (int i = 0; i < block_bytes; ++i) {
        ((uint8_t*)msg)[i] = src[i];
    }
}

// Hash one chunk (≤ CHUNK_LEN bytes). Sets ROOT on the final compression
// only when this is also the entire input (single-chunk hash), per spec.
__device__ __forceinline__ void hash_chunk(
    const uint8_t* data,
    int chunk_len,                   // bytes; 0..CHUNK_LEN
    uint64_t chunk_counter,
    bool is_single_chunk_root,
    uint32_t out_cv[8]
) {
    uint32_t cv[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) cv[i] = IV[i];

    int n_blocks_full = chunk_len / 64;
    int rem           = chunk_len - n_blocks_full * 64;
    int n_blocks      = n_blocks_full + ((rem > 0 || chunk_len == 0) ? 1 : 0);

    int offset = 0;
    for (int b = 0; b < n_blocks; ++b) {
        bool is_first = (b == 0);
        bool is_last  = (b == n_blocks - 1);
        int block_bytes = (b < n_blocks_full) ? 64 : rem;
        if (chunk_len == 0) block_bytes = 0;

        uint32_t flags = 0;
        if (is_first) flags |= CHUNK_START;
        if (is_last)  flags |= CHUNK_END;
        if (is_last && is_single_chunk_root) flags |= ROOT;

        uint32_t msg[16];
        pack_block(data + offset, block_bytes, msg);

        uint32_t out16[16];
        compress(cv, msg, chunk_counter, (uint32_t)block_bytes, flags, out16);

        if (is_last) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) out_cv[i] = out16[i];
        } else {
            #pragma unroll
            for (int i = 0; i < 8; ++i) cv[i] = out16[i];
        }
        offset += block_bytes;
    }
}

// Parent compression: compress(IV, left||right, counter=0, block_len=64,
// flags=PARENT[|ROOT]).
__device__ __forceinline__ void parent_compress(
    const uint32_t left[8],
    const uint32_t right[8],
    bool is_root,
    uint32_t out_cv[8]
) {
    uint32_t msg[16];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        msg[i]     = left[i];
        msg[i + 8] = right[i];
    }
    uint32_t cv[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) cv[i] = IV[i];

    uint32_t flags = PARENT;
    if (is_root) flags |= ROOT;

    uint32_t out16[16];
    compress(cv, msg, /*counter=*/0, /*block_len=*/64, flags, out16);
    #pragma unroll
    for (int i = 0; i < 8; ++i) out_cv[i] = out16[i];
}

// Hash an arbitrary-length byte buffer with BLAKE3, producing a 32-byte
// (8 u32) digest. Single-thread implementation; intended for one
// thread = one column-hash use.
__device__ void hash_bytes(
    const uint8_t* data,
    int len_bytes,
    uint32_t out[8]
) {
    int n_chunks = (len_bytes + CHUNK_LEN - 1) / CHUNK_LEN;
    if (n_chunks == 0) n_chunks = 1;          // empty input → one empty chunk
    bool single_chunk = (n_chunks == 1);

    // Single-chunk fast path: chunk's own ROOT-flagged compression IS the hash.
    if (single_chunk) {
        hash_chunk(data, len_bytes, /*counter=*/0, /*is_single_chunk_root=*/true, out);
        return;
    }

    // Multi-chunk: stack-based merge, BLAKE3 reference rules.
    uint32_t stack[STACK_MAX][8];
    int sp = 0;
    int offset = 0;
    uint64_t chunk_counter = 0;

    for (int c = 0; c < n_chunks; ++c) {
        bool is_last_chunk = (c == n_chunks - 1);
        int chunk_bytes = is_last_chunk ? (len_bytes - offset) : CHUNK_LEN;

        uint32_t cv[8];
        hash_chunk(data + offset, chunk_bytes, chunk_counter,
                   /*is_single_chunk_root=*/false, cv);
        offset += chunk_bytes;
        ++chunk_counter;

        // Push.
        #pragma unroll
        for (int i = 0; i < 8; ++i) stack[sp][i] = cv[i];
        ++sp;

        // Eager merge: while the just-incremented chunk_counter has a
        // trailing zero bit (i.e. is even), merge top two.
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
                parent_compress(left, right, /*is_root=*/false, parent);
                #pragma unroll
                for (int i = 0; i < 8; ++i) stack[sp][i] = parent[i];
                ++sp;
            }
        }
    }

    // Final cleanup: collapse remaining stack with parent compressions.
    // The last merge — when sp drops to 1 — sets ROOT.
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
        parent_compress(left, right, is_root, parent);
        #pragma unroll
        for (int i = 0; i < 8; ++i) stack[sp][i] = parent[i];
        ++sp;
    }

    #pragma unroll
    for (int i = 0; i < 8; ++i) out[i] = stack[0][i];
}

} // namespace b3
