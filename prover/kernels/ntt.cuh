// Goldilocks NTT — iterative Cooley-Tukey, decimation-in-time, in-place.
//
// Implementation strategy: level-per-kernel-launch.
//   - One bit-reversal kernel.
//   - log2(N) butterfly kernels, one per level (length = 2, 4, ..., N).
//   - Precomputed twiddle table of N/2 entries holding ω^0 .. ω^(N/2-1)
//     for a primitive N-th root of unity ω. At level "length", the
//     in-group index k uses twiddle index `k * (N/length)`.
//
// This is deliberately the simplest correct version. It pays log2(N)
// kernel-launch latencies per NTT plus log2(N) full-array global-memory
// passes. A "stages-fused-in-shared-memory" variant (cuFFT-style) would
// reduce both, but we want the clean baseline first so we can tell how
// much of any future speedup comes from the optimization.
//
// API:
//   ntt_init(n, &ctx)          — allocate device twiddles for size N.
//   ntt_forward(d_a, ctx)      — in-place forward NTT of length N.
//   ntt_inverse(d_a, ctx)      — in-place inverse NTT (incl. divide by n).
//   ntt_destroy(&ctx)
//
// Inputs and outputs are canonical Goldilocks elements (< P).

#pragma once

#include "goldilocks.cuh"
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace gl_ntt {

struct Ctx {
    int        n;            // NTT length, power of 2
    int        log2n;        // log2(n)
    uint64_t*  d_twid_fwd;   // size n/2: ω^0, ω^1, ..., ω^(n/2-1)
    uint64_t*  d_twid_inv;   // size n/2: ω^{-0}, ω^{-1}, ..., ω^{-(n/2-1)}
    uint64_t   n_inv;        // 1/n mod p, for inverse normalization
    uint64_t*  d_temp;       // Bailey scratch buffer (size n if Bailey enabled)
};

// ---------- kernels ----------

__global__ void k_bit_reverse(uint64_t* a, int n, int log2n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned ui = (unsigned)i;
    unsigned uj = __brev(ui) >> (32 - log2n);
    int j = (int)uj;
    if (i < j) {
        uint64_t t = a[i];
        a[i] = a[j];
        a[j] = t;
    }
}

// One Cooley-Tukey level. `half` = length/2. `twid_stride` = n/length.
// The butterfly index ranges 0 .. n/2 - 1; group = idx / half, k = idx % half.
// Base = group * length + k; pair = (base, base + half).
__global__ void k_butterfly_level(
    uint64_t* a,
    const uint64_t* twiddles,
    int n_half,        // n / 2
    int half,          // level length / 2
    int twid_stride    // n / length
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_half) return;
    int group = idx / half;
    int k     = idx - group * half;
    int base  = group * (half << 1) + k;

    uint64_t w = twiddles[k * twid_stride];
    uint64_t u = a[base];
    uint64_t v = gl::mul(a[base + half], w);
    a[base]        = gl::add(u, v);
    a[base + half] = gl::sub(u, v);
}

__global__ void k_scale(uint64_t* a, int n, uint64_t c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    a[i] = gl::mul(a[i], c);
}

// ---------- host helpers ----------

static inline uint64_t host_addmod(uint64_t a, uint64_t b) {
    __uint128_t s = (__uint128_t)a + b;
    if (s >= gl::P) s -= gl::P;
    return (uint64_t)s;
}
static inline uint64_t host_mulmod(uint64_t a, uint64_t b) {
    __uint128_t prod = (__uint128_t)a * b;
    return (uint64_t)(prod % gl::P);
}
static inline uint64_t host_powmod(uint64_t base, uint64_t exp) {
    uint64_t r = 1; base %= gl::P;
    while (exp) {
        if (exp & 1) r = host_mulmod(r, base);
        base = host_mulmod(base, base);
        exp >>= 1;
    }
    return r;
}
static inline uint64_t host_invmod(uint64_t a) { return host_powmod(a, gl::P - 2); }

// Primitive root g = 7 for Goldilocks.
static inline uint64_t host_root_of_unity(int n) {
    // (P-1) % n == 0 holds for all n | 2^32, which our power-of-two N
    // satisfies up to N = 2^32.
    return host_powmod(7ULL, (gl::P - 1) / (uint64_t)n);
}

// Build the n/2-element twiddle table on host: out[k] = base^k.
static void host_fill_twiddles(uint64_t* out, int n_half, uint64_t base) {
    out[0] = 1;
    for (int k = 1; k < n_half; ++k) out[k] = host_mulmod(out[k - 1], base);
}

// Fused-stages kernel: each block does 8 butterfly stages in shared memory
// on a 256-element chunk. Replaces stages 1-8 of the level-per-launch
// pipeline (which still handles stages 9..log2(N) for N > 256).
//
// Precondition: input is already globally bit-reversed at log2(N) bits.
// After global bit-reversal, the elements paired by stages 1..8 lie
// within the same 256-element chunk, so they can be processed entirely
// in shared memory with no cross-block traffic.
//
// We use 128 threads per block (one butterfly per thread per stage).
// Each thread loads/stores 2 elements; chunk size = 256 = 128 × 2.
__global__ void k_butterfly_fused256(
    uint64_t* __restrict__ a,
    const uint64_t* __restrict__ twid_full,   // length n/2: ω^0, ω^1, ..., ω^(n/2-1)
    int n_total                                // overall NTT size, used for twiddle stride
) {
    __shared__ uint64_t s[256];

    int tid   = threadIdx.x;
    int chunk = blockIdx.x;
    int gbase = chunk * 256;

    // Load 2 elements per thread.
    s[tid]       = a[gbase + tid];
    s[tid + 128] = a[gbase + tid + 128];
    __syncthreads();

    // 8 fused levels. At level L (0-indexed), half = 2^L (1, 2, ..., 128).
    // Each thread does one butterfly: idx = tid in [0,128).
    // group = idx / half; k = idx % half; sbase = group*length + k.
    // Twiddle: w = ω_n^(k * (n_total / length)) where length = 2*half.
    #pragma unroll
    for (int L = 0; L < 8; ++L) {
        int half   = 1 << L;
        int length = half << 1;
        int twid_stride_in_full = n_total / length;

        int group = tid >> L;            // tid / half
        int k     = tid - (group << L);  // tid % half
        int sbase = (group << (L + 1)) + k;

        uint64_t w = twid_full[k * twid_stride_in_full];
        uint64_t u = s[sbase];
        uint64_t v = gl::mul(s[sbase + half], w);
        s[sbase]        = gl::add(u, v);
        s[sbase + half] = gl::sub(u, v);
        __syncthreads();
    }

    a[gbase + tid]       = s[tid];
    a[gbase + tid + 128] = s[tid + 128];
}

// Larger fused kernel: 12 stages over 4096-element chunks in 32 KB of
// shared memory. 1024 threads per block × 4 elements per thread; each
// thread does 2 butterflies per stage. Same precondition as
// k_butterfly_fused256 (input is globally bit-reversed first).
__global__ void k_butterfly_fused4096(
    uint64_t* __restrict__ a,
    const uint64_t* __restrict__ twid_full,
    int n_total
) {
    constexpr int N_LOCAL = 4096;
    __shared__ uint64_t s[N_LOCAL];

    int tid   = threadIdx.x;          // 0..1023
    int chunk = blockIdx.x;
    int gbase = chunk * N_LOCAL;

    // Load 4 contiguous elements per thread.
    int load_base = tid * 4;
    s[load_base + 0] = a[gbase + load_base + 0];
    s[load_base + 1] = a[gbase + load_base + 1];
    s[load_base + 2] = a[gbase + load_base + 2];
    s[load_base + 3] = a[gbase + load_base + 3];
    __syncthreads();

    // 12 fused levels. Each level: 2048 butterflies → 2 per thread.
    #pragma unroll
    for (int L = 0; L < 12; ++L) {
        int half = 1 << L;
        int length = half << 1;
        int twid_stride_full = n_total / length;
        int half_mask = half - 1;

        #pragma unroll
        for (int boff = 0; boff < 2; ++boff) {
            int idx   = (tid << 1) + boff;
            int group = idx >> L;
            int k     = idx & half_mask;
            int sbase = (group << (L + 1)) + k;

            uint64_t w = twid_full[k * twid_stride_full];
            uint64_t u = s[sbase];
            uint64_t v = gl::mul(s[sbase + half], w);
            s[sbase]        = gl::add(u, v);
            s[sbase + half] = gl::sub(u, v);
        }
        __syncthreads();
    }

    a[gbase + load_base + 0] = s[load_base + 0];
    a[gbase + load_base + 1] = s[load_base + 1];
    a[gbase + load_base + 2] = s[load_base + 2];
    a[gbase + load_base + 3] = s[load_base + 3];
}

// 256-point NTT in shared memory, with internal bit-reversal at load and
// caller-controlled strided I/O. Each block does ONE 256-pt NTT.
//
// Building block of the Bailey two-pass decomposition for N=65536. The
// caller supplies the FULL twiddle table (n_total/2 entries) and we use
// stride n_total/length for each butterfly stage; that lets a single
// twiddle table serve both the 256-pt building blocks and any other
// kernel that targets the same n_total.
__global__ void k_ntt256_strided(
    const uint64_t* __restrict__ a_in,
    uint64_t* __restrict__       a_out,
    int read_block_step,
    int read_stride,
    int write_block_step,
    int write_stride,
    const uint64_t* __restrict__ twid_full,
    int n_total
) {
    __shared__ uint64_t s[256];
    int tid   = threadIdx.x;
    int chunk = blockIdx.x;
    int read_base  = chunk * read_block_step;
    int write_base = chunk * write_block_step;

    // 8-bit reversal on load.
    int i0 = tid;
    int i1 = tid + 128;
    int j0 = (int)(__brev((unsigned)i0) >> 24);
    int j1 = (int)(__brev((unsigned)i1) >> 24);
    s[j0] = a_in[read_base + i0 * read_stride];
    s[j1] = a_in[read_base + i1 * read_stride];
    __syncthreads();

    #pragma unroll
    for (int L = 0; L < 8; ++L) {
        int half = 1 << L;
        int twid_stride_full = n_total / (half << 1);
        int half_mask = half - 1;
        int group = tid >> L;
        int k     = tid & half_mask;
        int sbase = (group << (L + 1)) + k;

        uint64_t w = twid_full[k * twid_stride_full];
        uint64_t u = s[sbase];
        uint64_t v = gl::mul(s[sbase + half], w);
        s[sbase]        = gl::add(u, v);
        s[sbase + half] = gl::sub(u, v);
        __syncthreads();
    }

    a_out[write_base + i0 * write_stride] = s[i0];
    a_out[write_base + i1 * write_stride] = s[i1];
}

// Bailey twiddle: a[k_a * n2 + j_b] *= ω_N^(k_a * j_b).
// Twiddle indices range up to (n1-1)*(n2-1); since the table only holds
// n_total/2 entries we extend via ω^(N/2) = -1 for indices >= N/2.
__global__ void k_bailey_twiddle(
    uint64_t* __restrict__ a,
    const uint64_t* __restrict__ twid_full,
    int n1, int n2, int n_total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_total) return;
    int k_a = idx / n2;
    int j_b = idx - k_a * n2;

    int t_idx  = k_a * j_b;
    int n_half = n_total / 2;

    uint64_t w;
    if (t_idx < n_half) {
        w = twid_full[t_idx];
    } else {
        uint64_t v = twid_full[t_idx - n_half];
        w = (v == 0) ? 0 : (gl::P - v);   // ω^(N/2) = -1
    }
    a[idx] = gl::mul(a[idx], w);
}

// ---------- public API ----------

inline void ntt_init(int n, Ctx* ctx) {
    if (n < 2 || (n & (n - 1)) != 0) {
        fprintf(stderr, "ntt_init: n must be a power of 2 >= 2; got %d\n", n);
        std::exit(1);
    }
    int log2n = 0; for (int t = n; t > 1; t >>= 1) ++log2n;

    ctx->n     = n;
    ctx->log2n = log2n;
    ctx->n_inv = host_invmod((uint64_t)n);

    int n_half = n / 2;
    uint64_t* h = (uint64_t*)std::malloc((size_t)n_half * sizeof(uint64_t));

    uint64_t omega     = host_root_of_unity(n);
    uint64_t omega_inv = host_invmod(omega);

    cudaMalloc(&ctx->d_twid_fwd, (size_t)n_half * sizeof(uint64_t));
    cudaMalloc(&ctx->d_twid_inv, (size_t)n_half * sizeof(uint64_t));

    host_fill_twiddles(h, n_half, omega);
    cudaMemcpy(ctx->d_twid_fwd, h, (size_t)n_half * sizeof(uint64_t), cudaMemcpyHostToDevice);

    host_fill_twiddles(h, n_half, omega_inv);
    cudaMemcpy(ctx->d_twid_inv, h, (size_t)n_half * sizeof(uint64_t), cudaMemcpyHostToDevice);

    std::free(h);

    // Allocate Bailey scratch only for sizes where Bailey applies (currently
    // hard-coded to N=65536 = 256x256). Other sizes set d_temp to null.
    ctx->d_temp = nullptr;
    if (n == 65536) {
        cudaMalloc(&ctx->d_temp, (size_t)n * sizeof(uint64_t));
    }
}

inline void ntt_destroy(Ctx* ctx) {
    if (ctx->d_twid_fwd) cudaFree(ctx->d_twid_fwd);
    if (ctx->d_twid_inv) cudaFree(ctx->d_twid_inv);
    if (ctx->d_temp)     cudaFree(ctx->d_temp);
    ctx->d_twid_fwd = nullptr;
    ctx->d_twid_inv = nullptr;
    ctx->d_temp     = nullptr;
}

// Whether to fuse the first 8 butterfly stages into a single shared-memory
// kernel. Off by default for parity with the historical baseline; turn on
// by passing fuse=true to ntt_run_inplace.
inline void ntt_run_inplace(uint64_t* d_a, const Ctx& ctx, bool invert,
                            cudaStream_t stream = 0, bool fuse = false) {
    int n = ctx.n;
    int n_half = n / 2;
    const uint64_t* twid = invert ? ctx.d_twid_inv : ctx.d_twid_fwd;

    // bit-reverse permutation
    {
        int t = 256;
        int b = (n + t - 1) / t;
        k_bit_reverse<<<b, t, 0, stream>>>(d_a, n, ctx.log2n);
    }

    int length = 2;
    if (fuse) {
        // Use the largest shared-memory chunk that fits into N. 4096 covers
        // 12 stages (32 KB shared); 256 covers 8 stages (2 KB shared).
        if (n >= 4096) {
            int chunks = n / 4096;
            k_butterfly_fused4096<<<chunks, 1024, 0, stream>>>(d_a, twid, n);
            length = 1 << 13;        // stages 1..12 done; resume at length=8192.
        } else if (n >= 256) {
            int chunks = n / 256;
            k_butterfly_fused256<<<chunks, 128, 0, stream>>>(d_a, twid, n);
            length = 1 << 9;         // stages 1..8 done; resume at length=512.
        }
    }

    while (length <= n) {
        int half = length / 2;
        int twid_stride = n / length;
        int t = 256;
        int b = (n_half + t - 1) / t;
        k_butterfly_level<<<b, t, 0, stream>>>(d_a, twid, n_half, half, twid_stride);
        length <<= 1;
    }

    if (invert) {
        int t = 256;
        int b = (n + t - 1) / t;
        k_scale<<<b, t, 0, stream>>>(d_a, n, ctx.n_inv);
    }
}

inline void ntt_forward(uint64_t* d_a, const Ctx& ctx, cudaStream_t s = 0) {
    ntt_run_inplace(d_a, ctx, false, s, /*fuse=*/false);
}
inline void ntt_inverse(uint64_t* d_a, const Ctx& ctx, cudaStream_t s = 0) {
    ntt_run_inplace(d_a, ctx, true, s, /*fuse=*/false);
}
inline void ntt_forward_fused(uint64_t* d_a, const Ctx& ctx, cudaStream_t s = 0) {
    ntt_run_inplace(d_a, ctx, false, s, /*fuse=*/true);
}
inline void ntt_inverse_fused(uint64_t* d_a, const Ctx& ctx, cudaStream_t s = 0) {
    ntt_run_inplace(d_a, ctx, true, s, /*fuse=*/true);
}

// Bailey 4-step decomposition, hard-coded for N=65536 = 256x256.
//
// View input as M[j_a, j_b] = a[j_a*256 + j_b].
// Pass 1 (column NTT, size 256): for each j_b, NTT down column j_b. Strided
//   read/write across blocks — but we use d_temp so reads from d_a never
//   alias writes.
// Pass 2 (twiddle): a[k_a*256+j_b] *= ω^(k_a*j_b). In-place on d_temp.
// Pass 3 (row NTT + transposed write, size 256): for each k_a, NTT row k_a
//   contiguous in d_temp; write transposed to d_a so output A[k_b*256+k_a]
//   sits at a[k_b*256+k_a]. Bailey output convention.
//
// Output indexing: A[k] for k in [0, N) in standard linear order — matches
// what the level-per-launch baseline produces.
inline void ntt_run_bailey(uint64_t* d_a, const Ctx& ctx, bool invert,
                           cudaStream_t stream = 0) {
    constexpr int N1 = 256, N2 = 256;
    const int N = N1 * N2;
    if (ctx.n != N || ctx.d_temp == nullptr) {
        fprintf(stderr, "ntt_run_bailey: only supported at N=65536 with allocated scratch.\n");
        std::exit(1);
    }
    const uint64_t* twid = invert ? ctx.d_twid_inv : ctx.d_twid_fwd;

    // Pass 1: 256 column NTTs. Read d_a strided by N2, write d_temp same.
    k_ntt256_strided<<<256, 128, 0, stream>>>(
        d_a, ctx.d_temp,
        /*read_block_step=*/  1,
        /*read_stride=*/      N2,
        /*write_block_step=*/ 1,
        /*write_stride=*/     N2,
        twid, N);

    // Pass 2: twiddle on d_temp, in place.
    {
        int t = 256;
        int b = (N + t - 1) / t;
        k_bailey_twiddle<<<b, t, 0, stream>>>(ctx.d_temp, twid, N1, N2, N);
    }

    // Pass 3: 256 row NTTs with transposed write. Read d_temp contiguous,
    // write d_a strided by N1.
    k_ntt256_strided<<<256, 128, 0, stream>>>(
        ctx.d_temp, d_a,
        /*read_block_step=*/  N2,
        /*read_stride=*/      1,
        /*write_block_step=*/ 1,
        /*write_stride=*/     N1,
        twid, N);

    if (invert) {
        int t = 256;
        int b = (N + t - 1) / t;
        k_scale<<<b, t, 0, stream>>>(d_a, N, ctx.n_inv);
    }
}

inline void ntt_forward_bailey(uint64_t* d_a, const Ctx& ctx, cudaStream_t s = 0) {
    ntt_run_bailey(d_a, ctx, false, s);
}
inline void ntt_inverse_bailey(uint64_t* d_a, const Ctx& ctx, cudaStream_t s = 0) {
    ntt_run_bailey(d_a, ctx, true, s);
}

// ============================================================================
// Batched kernels: process m rows in one launch each. blockIdx.y is the row;
// the buffer is offset by row * n. Everything else is identical to the
// single-row kernel above. Replaces a per-row C++ for-loop (which paid
// kernel-launch overhead per row) with one launch per NTT stage total.
// ============================================================================

__global__ void k_bit_reverse_b(uint64_t* a, int n, int log2n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y;
    if (i >= n) return;
    uint64_t* ra = a + (size_t)row * n;
    unsigned uj = __brev((unsigned)i) >> (32 - log2n);
    int j = (int)uj;
    if (i < j) {
        uint64_t t = ra[i];
        ra[i] = ra[j];
        ra[j] = t;
    }
}

__global__ void k_butterfly_level_b(
    uint64_t* a, const uint64_t* twiddles,
    int n_half, int half, int twid_stride, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y;
    if (idx >= n_half) return;
    uint64_t* ra = a + (size_t)row * n;
    int group = idx / half;
    int k     = idx - group * half;
    int base  = group * (half << 1) + k;
    uint64_t w = twiddles[k * twid_stride];
    uint64_t u = ra[base];
    uint64_t v = gl::mul(ra[base + half], w);
    ra[base]        = gl::add(u, v);
    ra[base + half] = gl::sub(u, v);
}

__global__ void k_scale_b(uint64_t* a, int n, uint64_t c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y;
    if (i >= n) return;
    a[(size_t)row * n + i] = gl::mul(a[(size_t)row * n + i], c);
}

__global__ void k_butterfly_fused256_b(
    uint64_t* __restrict__ a, const uint64_t* __restrict__ twid_full, int n_total
) {
    __shared__ uint64_t s[256];
    int tid   = threadIdx.x;
    int chunk = blockIdx.x;
    int row   = blockIdx.y;
    size_t gbase = (size_t)chunk * 256 + (size_t)row * n_total;

    s[tid]       = a[gbase + tid];
    s[tid + 128] = a[gbase + tid + 128];
    __syncthreads();

    #pragma unroll
    for (int L = 0; L < 8; ++L) {
        int half   = 1 << L;
        int length = half << 1;
        int twid_stride_in_full = n_total / length;
        int group = tid >> L;
        int k     = tid - (group << L);
        int sbase = (group << (L + 1)) + k;
        uint64_t w = twid_full[k * twid_stride_in_full];
        uint64_t u = s[sbase];
        uint64_t v = gl::mul(s[sbase + half], w);
        s[sbase]        = gl::add(u, v);
        s[sbase + half] = gl::sub(u, v);
        __syncthreads();
    }

    a[gbase + tid]       = s[tid];
    a[gbase + tid + 128] = s[tid + 128];
}

__global__ void k_butterfly_fused4096_b(
    uint64_t* __restrict__ a, const uint64_t* __restrict__ twid_full, int n_total
) {
    constexpr int N_LOCAL = 4096;
    __shared__ uint64_t s[N_LOCAL];

    int tid   = threadIdx.x;
    int chunk = blockIdx.x;
    int row   = blockIdx.y;
    size_t gbase = (size_t)chunk * N_LOCAL + (size_t)row * n_total;
    int load_base = tid * 4;

    s[load_base + 0] = a[gbase + load_base + 0];
    s[load_base + 1] = a[gbase + load_base + 1];
    s[load_base + 2] = a[gbase + load_base + 2];
    s[load_base + 3] = a[gbase + load_base + 3];
    __syncthreads();

    #pragma unroll
    for (int L = 0; L < 12; ++L) {
        int half = 1 << L;
        int length = half << 1;
        int twid_stride_full = n_total / length;
        int half_mask = half - 1;
        #pragma unroll
        for (int boff = 0; boff < 2; ++boff) {
            int idx   = (tid << 1) + boff;
            int group = idx >> L;
            int k     = idx & half_mask;
            int sbase = (group << (L + 1)) + k;
            uint64_t w = twid_full[k * twid_stride_full];
            uint64_t u = s[sbase];
            uint64_t v = gl::mul(s[sbase + half], w);
            s[sbase]        = gl::add(u, v);
            s[sbase + half] = gl::sub(u, v);
        }
        __syncthreads();
    }

    a[gbase + load_base + 0] = s[load_base + 0];
    a[gbase + load_base + 1] = s[load_base + 1];
    a[gbase + load_base + 2] = s[load_base + 2];
    a[gbase + load_base + 3] = s[load_base + 3];
}

__global__ void k_ntt256_strided_b(
    const uint64_t* __restrict__ a_in, uint64_t* __restrict__ a_out,
    int read_block_step, int read_stride,
    int write_block_step, int write_stride,
    const uint64_t* __restrict__ twid_full, int n_total
) {
    __shared__ uint64_t s[256];
    int tid   = threadIdx.x;
    int chunk = blockIdx.x;
    int row   = blockIdx.y;
    size_t read_base  = (size_t)chunk * read_block_step  + (size_t)row * n_total;
    size_t write_base = (size_t)chunk * write_block_step + (size_t)row * n_total;

    int i0 = tid;
    int i1 = tid + 128;
    int j0 = (int)(__brev((unsigned)i0) >> 24);
    int j1 = (int)(__brev((unsigned)i1) >> 24);
    s[j0] = a_in[read_base + i0 * read_stride];
    s[j1] = a_in[read_base + i1 * read_stride];
    __syncthreads();

    #pragma unroll
    for (int L = 0; L < 8; ++L) {
        int half = 1 << L;
        int twid_stride_full = n_total / (half << 1);
        int half_mask = half - 1;
        int group = tid >> L;
        int k     = tid & half_mask;
        int sbase = (group << (L + 1)) + k;
        uint64_t w = twid_full[k * twid_stride_full];
        uint64_t u = s[sbase];
        uint64_t v = gl::mul(s[sbase + half], w);
        s[sbase]        = gl::add(u, v);
        s[sbase + half] = gl::sub(u, v);
        __syncthreads();
    }

    a_out[write_base + i0 * write_stride] = s[i0];
    a_out[write_base + i1 * write_stride] = s[i1];
}

__global__ void k_bailey_twiddle_b(
    uint64_t* __restrict__ a, const uint64_t* __restrict__ twid_full,
    int n1, int n2, int n_total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y;
    if (idx >= n_total) return;
    int k_a = idx / n2;
    int j_b = idx - k_a * n2;
    int t_idx = k_a * j_b;
    int n_half = n_total / 2;
    uint64_t w;
    if (t_idx < n_half) {
        w = twid_full[t_idx];
    } else {
        uint64_t v = twid_full[t_idx - n_half];
        w = (v == 0) ? 0 : (gl::P - v);
    }
    size_t pos = (size_t)row * n_total + idx;
    a[pos] = gl::mul(a[pos], w);
}

// ---------- batched run functions ----------

inline void ntt_run_inplace_batched(
    uint64_t* d_a, int m, const Ctx& ctx, bool invert,
    cudaStream_t stream = 0, bool fuse = false
) {
    int n = ctx.n;
    int n_half = n / 2;
    const uint64_t* twid = invert ? ctx.d_twid_inv : ctx.d_twid_fwd;

    {
        int t = 256;
        int bx = (n + t - 1) / t;
        k_bit_reverse_b<<<dim3(bx, m), t, 0, stream>>>(d_a, n, ctx.log2n);
    }

    int length = 2;
    if (fuse) {
        if (n >= 4096) {
            int chunks = n / 4096;
            k_butterfly_fused4096_b<<<dim3(chunks, m), 1024, 0, stream>>>(d_a, twid, n);
            length = 1 << 13;
        } else if (n >= 256) {
            int chunks = n / 256;
            k_butterfly_fused256_b<<<dim3(chunks, m), 128, 0, stream>>>(d_a, twid, n);
            length = 1 << 9;
        }
    }

    while (length <= n) {
        int half = length / 2;
        int twid_stride = n / length;
        int t = 256;
        int bx = (n_half + t - 1) / t;
        k_butterfly_level_b<<<dim3(bx, m), t, 0, stream>>>(
            d_a, twid, n_half, half, twid_stride, n);
        length <<= 1;
    }

    if (invert) {
        int t = 256;
        int bx = (n + t - 1) / t;
        k_scale_b<<<dim3(bx, m), t, 0, stream>>>(d_a, n, ctx.n_inv);
    }
}

// Bailey batched: needs scratch of size m * n uint64s passed in.
inline void ntt_run_bailey_batched(
    uint64_t* d_a, int m, const Ctx& ctx, bool invert,
    uint64_t* scratch, cudaStream_t stream = 0
) {
    constexpr int N1 = 256, N2 = 256;
    const int N = N1 * N2;
    if (ctx.n != N) {
        fprintf(stderr, "ntt_run_bailey_batched: only N=65536 supported\n");
        std::exit(1);
    }
    const uint64_t* twid = invert ? ctx.d_twid_inv : ctx.d_twid_fwd;

    // Pass 1: column NTTs into scratch.
    k_ntt256_strided_b<<<dim3(256, m), 128, 0, stream>>>(
        d_a, scratch, 1, N2, 1, N2, twid, N);

    // Pass 2: twiddle on scratch.
    {
        int t = 256;
        int bx = (N + t - 1) / t;
        k_bailey_twiddle_b<<<dim3(bx, m), t, 0, stream>>>(scratch, twid, N1, N2, N);
    }

    // Pass 3: row NTTs with transposed write back to d_a.
    k_ntt256_strided_b<<<dim3(256, m), 128, 0, stream>>>(
        scratch, d_a, N2, 1, 1, N1, twid, N);

    if (invert) {
        int t = 256;
        int bx = (N + t - 1) / t;
        k_scale_b<<<dim3(bx, m), t, 0, stream>>>(d_a, N, ctx.n_inv);
    }
}

inline void ntt_forward_batched_fast(
    uint64_t* d_a, int m, const Ctx& ctx, uint64_t* bailey_scratch = nullptr,
    cudaStream_t s = 0
) {
    if (ctx.n == 65536) {
        ntt_run_bailey_batched(d_a, m, ctx, false, bailey_scratch, s);
    } else if (ctx.n >= 256) {
        ntt_run_inplace_batched(d_a, m, ctx, false, s, /*fuse=*/true);
    } else {
        ntt_run_inplace_batched(d_a, m, ctx, false, s, /*fuse=*/false);
    }
}

inline void ntt_inverse_batched_fast(
    uint64_t* d_a, int m, const Ctx& ctx, uint64_t* bailey_scratch = nullptr,
    cudaStream_t s = 0
) {
    if (ctx.n == 65536) {
        ntt_run_bailey_batched(d_a, m, ctx, true, bailey_scratch, s);
    } else if (ctx.n >= 256) {
        ntt_run_inplace_batched(d_a, m, ctx, true, s, /*fuse=*/true);
    } else {
        ntt_run_inplace_batched(d_a, m, ctx, true, s, /*fuse=*/false);
    }
}

} // namespace gl_ntt
