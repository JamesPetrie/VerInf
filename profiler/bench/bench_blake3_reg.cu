// BLAKE3 compression throughput, REGISTER-RESIDENT.
//
// Question this answers: how many b3::compress calls per second can the
// chip sustain when state and message live in registers — i.e. the
// ALU-bound rate, with no global-memory traffic in the loop.
//
// Why it exists: the cost model's B constant (ns per constraint id) is one
// challenge-hash compress per cid and is REGISTER/ALU-bound. The existing
// bench_blake3_columns absorbs column data from global memory and is
// memory-bandwidth-limited at large m (spark-microbench-results.md), so its
// Gcompress/s rate cannot calibrate B's cross-machine scaling. This bench
// isolates the compute side. Run it on BOTH boxes of a ratio (the GB10
// baseline lacks this number until someone runs it there).
//
// Method: mirrors bench_field_mul — each thread runs a dependent chain of
// compress calls (out chains back into cv and message), parallelism comes
// from thread count; each compress carries ~7x16 G-function rounds of
// internal ILP, so occupancy hides the chain latency. Ops reported are
// blocks * threads * iters compress calls.
//
// Build (from repo root; arch must match the card):
//   nvcc -arch=<sm_XXX> -std=c++17 -O3 -Iprover/kernels \
//        profiler/bench/bench_blake3_reg.cu -o bench_blake3_reg
//
// Usage:
//   ./bench_blake3_reg                 # defaults
//   ./bench_blake3_reg --warmup 5 --iters 2048

#include "blake3_compress.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

__global__ void bench_compress_kernel(uint32_t* sink, uint32_t seed, int iters) {
    uint64_t tid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;

    // Per-thread state seeded from tid so threads run distinct traces
    // (no closed form for the compiler to find).
    uint32_t cv[8];
    uint32_t block[16];
    uint32_t out[16];
    #pragma unroll
    for (int i = 0; i < 8; ++i)
        cv[i] = seed * 0x9E3779B9u + (uint32_t)tid + (uint32_t)i * 0x85EBCA6Bu;
    #pragma unroll
    for (int i = 0; i < 16; ++i)
        block[i] = seed * 0xC2B2AE35u + (uint32_t)tid + (uint32_t)i * 0x27D4EB2Fu;

    // Dependent chain: feed the output back into cv and half the message.
    // Don't unroll the outer loop — register pressure.
    #pragma unroll 1
    for (int i = 0; i < iters; ++i) {
        b3::compress(cv, block, (uint64_t)i, 64u, 0u, out);
        #pragma unroll
        for (int k = 0; k < 8; ++k) cv[k] = out[k];
        #pragma unroll
        for (int k = 0; k < 8; ++k) block[k] ^= out[k + 8];
    }

    uint32_t r = 0;
    #pragma unroll
    for (int k = 0; k < 16; ++k) r ^= out[k];
    sink[tid] = r;   // anti-DCE
}

int main(int argc, char** argv) {
    int warmup_runs = 3;
    int timed_runs  = 5;
    int iters       = 2048;
    int blocks      = 256;
    int threads     = 256;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> int {
            if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", argv[i]); exit(2); }
            return atoi(argv[++i]);
        };
        if      (a == "--warmup")  warmup_runs = next();
        else if (a == "--runs")    timed_runs  = next();
        else if (a == "--iters")   iters       = next();
        else if (a == "--blocks")  blocks      = next();
        else if (a == "--threads") threads     = next();
        else { fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 2; }
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("device: %s  sm=%d.%d  SMs=%d\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    printf("config: blocks=%d threads=%d iters=%d  warmup=%d runs=%d  "
           "(register-resident chained compress)\n",
           blocks, threads, iters, warmup_runs, timed_runs);

    size_t n_threads = (size_t)blocks * threads;
    uint32_t* d_sink;
    cudaMalloc(&d_sink, n_threads * sizeof(uint32_t));

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    for (int w = 0; w < warmup_runs; ++w) {
        bench_compress_kernel<<<blocks, threads>>>(d_sink, 0xC0FFEEu + w, iters);
    }
    cudaDeviceSynchronize();

    double best_gcps = 0.0;
    for (int r = 0; r < timed_runs; ++r) {
        cudaEventRecord(t0);
        bench_compress_kernel<<<blocks, threads>>>(d_sink, 0x1234u + r, iters);
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, t0, t1);

        double total_ops = (double)n_threads * (double)iters;
        double gcps = total_ops / (ms * 1e6);   // ms->s, ops->Gops
        if (gcps > best_gcps) best_gcps = gcps;
        printf("  run %d: %.3f ms  -> %.2f Gcompress/s\n", r, ms, gcps);
    }
    printf("best: %.2f Gcompress/s (register-resident)\n", best_gcps);

    uint32_t scratch;
    cudaMemcpy(&scratch, d_sink, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    printf("sink[0] = %08x (anti-DCE)\n", scratch);

    cudaFree(d_sink);
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
