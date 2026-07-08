// Goldilocks field arithmetic for CUDA.
//
// Field: Goldilocks prime p = 2^64 - 2^32 + 1.
// - 2-adicity: p - 1 = 2^32 * (2^32 - 1), so we can do NTTs up to length 2^32.
// - Primitive root: g = 7. (Standard Plonky2 choice.)
//
// Reduction strategy is the canonical Plonky2 form:
//   x = x_lo + 2^64 * x_hi  (128-bit input)
//   2^64    ≡  2^32 - 1   (mod p)   [since p = 2^64 - (2^32 - 1)]
//   2^96    ≡  -1         (mod p)   [since 2^96 = 2^64 · 2^32 ≡ (2^32-1)·2^32 ≡ 2^64 - 2^32 ≡ -1]
//   So with x_hi = x_hi_hi · 2^32 + x_hi_lo:
//   x ≡ x_lo - x_hi_hi + (2^32 - 1) · x_hi_lo  (mod p)
//   Implemented in three steps with overflow-aware u64 arithmetic; see reduce128.
//
// All non-trivial constants and identities below are derivable from
// p = 2^64 - 2^32 + 1; no external library dependency.

#pragma once

#include <cstdint>

namespace gl {

constexpr uint64_t P       = 0xFFFFFFFF00000001ULL;   // 2^64 - 2^32 + 1
constexpr uint64_t EPSILON = 0xFFFFFFFFULL;           // 2^32 - 1; equals 2^64 mod P

// Reduce 128-bit (hi:lo) to canonical Goldilocks in [0, P).
__device__ __forceinline__ uint64_t reduce128(uint64_t lo, uint64_t hi) {
    uint64_t hi_hi = hi >> 32;             // top 32 bits of hi
    uint64_t hi_lo = hi & EPSILON;         // low 32 bits of hi

    // t0 = lo - hi_hi  (mod p, accounting for u64 borrow)
    uint64_t t0   = lo - hi_hi;
    bool   borrow = (lo < hi_hi);
    if (borrow) t0 -= EPSILON;             // -2^64 ≡ -EPSILON (mod p)

    // t1 = hi_lo * (2^32 - 1)  — fits in u64 since hi_lo < 2^32.
    uint64_t t1 = hi_lo * EPSILON;

    // res = t0 + t1  (mod p, accounting for u64 carry)
    uint64_t res = t0 + t1;
    bool   carry = (res < t0);
    if (carry) res += EPSILON;             // +2^64 ≡ +EPSILON (mod p)

    // Final canonical step: at most one P to subtract.
    if (res >= P) res -= P;
    return res;
}

__device__ __forceinline__ uint64_t add(uint64_t a, uint64_t b) {
    uint64_t s = a + b;
    bool carry = (s < a);
    if (carry) s += EPSILON;
    if (s >= P) s -= P;
    return s;
}

__device__ __forceinline__ uint64_t sub(uint64_t a, uint64_t b) {
    // Assumes a, b in [0, P). Result canonical in [0, P).
    uint64_t d   = a - b;
    bool   borrow = (a < b);
    if (borrow) d -= EPSILON;              // d - EPSILON = (a - b) + p when borrow
    return d;
}

__device__ __forceinline__ uint64_t mul(uint64_t a, uint64_t b) {
    uint64_t lo = a * b;                   // low 64 bits
    uint64_t hi = __umul64hi(a, b);        // high 64 bits
    return reduce128(lo, hi);
}

// Modular exponentiation via square-and-multiply.
__device__ __forceinline__ uint64_t pow(uint64_t base, uint64_t exp) {
    uint64_t r = 1;
    while (exp) {
        if (exp & 1) r = mul(r, base);
        base = mul(base, base);
        exp >>= 1;
    }
    return r;
}

} // namespace gl
