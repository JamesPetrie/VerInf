// BLAKE3 compression function in CUDA.
//
// This is the core block primitive of BLAKE3: take a chaining value
// (8 words), a 16-word message block, a counter and flags, run 7 rounds
// of mixing, and produce 16 output words (or 8 when used as a chaining
// value). All higher-level structure (chunk chain, tree merge) is built
// on top of this single function.
//
// Reference: https://github.com/BLAKE3-team/BLAKE3-specs (Section 2.1).
// Source-of-truth quotes embedded inline at the constants below.

#pragma once

#include <cstdint>

namespace b3 {

// "The IV of BLAKE3 is the same as for BLAKE2 and SHA-256." — spec §2.1.
__device__ __constant__ uint32_t IV[8] = {
    0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
    0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u,
};

// "MSG_PERMUTATION is applied to the 16 message words at the start of each round
//  except the first." — spec §2.1.
__device__ __constant__ uint8_t MSG_PERMUTATION[16] = {
    2,6,3,10,7,0,4,13,1,11,12,5,9,14,15,8
};

// Flag bits.
constexpr uint32_t CHUNK_START = 1u << 0;
constexpr uint32_t CHUNK_END   = 1u << 1;
constexpr uint32_t PARENT      = 1u << 2;
constexpr uint32_t ROOT        = 1u << 3;

constexpr uint32_t BLOCK_LEN  = 64;
constexpr uint32_t CHUNK_LEN  = 1024;

__device__ __forceinline__ uint32_t rotr(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

__device__ __forceinline__ void g(uint32_t s[16],
                                   int a, int b, int c, int d,
                                   uint32_t mx, uint32_t my) {
    s[a] = s[a] + s[b] + mx;
    s[d] = rotr(s[d] ^ s[a], 16);
    s[c] = s[c] + s[d];
    s[b] = rotr(s[b] ^ s[c], 12);
    s[a] = s[a] + s[b] + my;
    s[d] = rotr(s[d] ^ s[a], 8);
    s[c] = s[c] + s[d];
    s[b] = rotr(s[b] ^ s[c], 7);
}

__device__ __forceinline__ void round(uint32_t s[16], const uint32_t m[16]) {
    g(s, 0, 4,  8, 12, m[ 0], m[ 1]);
    g(s, 1, 5,  9, 13, m[ 2], m[ 3]);
    g(s, 2, 6, 10, 14, m[ 4], m[ 5]);
    g(s, 3, 7, 11, 15, m[ 6], m[ 7]);
    g(s, 0, 5, 10, 15, m[ 8], m[ 9]);
    g(s, 1, 6, 11, 12, m[10], m[11]);
    g(s, 2, 7,  8, 13, m[12], m[13]);
    g(s, 3, 4,  9, 14, m[14], m[15]);
}

// Compress one 64-byte block. Writes 16 output words in `out`.
//   cv       : 8-word input chaining value
//   block    : 16-word message block (zero-pad if input is short)
//   counter  : chunk counter (low 32 bits → s[12], high 32 → s[13])
//   block_len: bytes used in this block (≤ BLOCK_LEN)
//   flags    : OR of CHUNK_START / CHUNK_END / PARENT / ROOT
//
// "compress() takes... and produces 16 output words. The first 8 are used
//  as the next chaining value; the full 16 are used as extended output
//  (for XOF) when ROOT is set." — spec §2.1.
__device__ __forceinline__ void compress(
    const uint32_t cv[8],
    const uint32_t block[16],
    uint64_t counter,
    uint32_t block_len,
    uint32_t flags,
    uint32_t out[16]
) {
    uint32_t s[16] = {
        cv[0], cv[1], cv[2], cv[3],
        cv[4], cv[5], cv[6], cv[7],
        IV[0], IV[1], IV[2], IV[3],
        (uint32_t)counter,
        (uint32_t)(counter >> 32),
        block_len,
        flags
    };
    uint32_t m[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) m[i] = block[i];

    // Round 1 uses the message as-is. Rounds 2..7 each apply MSG_PERMUTATION.
    round(s, m);
    #pragma unroll
    for (int r = 1; r < 7; ++r) {
        uint32_t mp[16];
        #pragma unroll
        for (int i = 0; i < 16; ++i) mp[i] = m[MSG_PERMUTATION[i]];
        #pragma unroll
        for (int i = 0; i < 16; ++i) m[i] = mp[i];
        round(s, m);
    }

    // "After the rounds, the upper half of the state is xor'd into the lower
    //  half (and vice versa) to produce the 16 output words." — spec §2.1.
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        out[i]     = s[i] ^ s[i + 8];
        out[i + 8] = s[i + 8] ^ cv[i];
    }
}

} // namespace b3
