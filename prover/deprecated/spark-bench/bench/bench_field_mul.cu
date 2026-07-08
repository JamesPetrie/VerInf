// Goldilocks multiplication throughput benchmark.
//
// Question this answers: how many independent Goldilocks (mul + 128->64
// reduce) operations per second can the GB10 sustain in registers?
//
// This is the cleanest signal for whether the chip is fast at the field
// arithmetic Ligero hammers in NTT and pointwise products. Numbers from
// here feed directly into the wall-time projection in the Phase 1.0
// kickoff doc.
//
// Method: each thread keeps K=8 independent multiplicative chains and
// runs ITERS iterations of "x_i = mul(x_i, c_i)". K is chosen to overlap
// the mul + reduce latency (~12 cycles) so we measure throughput, not
// latency. Each iteration is K=8 muls; total reported ops are
//   blocks * threads * K * ITERS.
//
// Output is XORed into a global array to defeat dead-code elimination.
//
// Usage:
//   ./bench_field_mul                  # default size
//   ./bench_field_mul --warmup 5 --iters 4096

#include "goldilocks.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

constexpr int K_LANES = 8;

__global__ void bench_mul_kernel(uint64_t* sink, uint64_t seed, int iters) {
    uint64_t tid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;

    // K independent chains with K independent multipliers, all canonical < P.
    uint64_t x[K_LANES];
    uint64_t c[K_LANES];
    #pragma unroll
    for (int k = 0; k < K_LANES; ++k) {
        // Mix tid into both operands so different threads run different
        // numerical traces (avoids a degenerate constant-input case where
        // the compiler could find a closed form).
        uint64_t a = (seed * 0x9E3779B97F4A7C15ULL) + tid + (uint64_t)k * 0xBF58476D1CE4E5B9ULL;
        uint64_t b = (seed * 0xD1B54A32D192ED03ULL) + tid + (uint64_t)k * 0x94D049BB133111EBULL;
        if (a >= gl::P) a -= gl::P;
        if (b >= gl::P) b -= gl::P;
        x[k] = a;
        c[k] = b;
    }

    // Tight loop. Don't unroll — we want the compiler to keep the chain
    // tight, but unrolling explodes register pressure.
    #pragma unroll 1
    for (int i = 0; i < iters; ++i) {
        #pragma unroll
        for (int k = 0; k < K_LANES; ++k) {
            x[k] = gl::mul(x[k], c[k]);
        }
    }

    uint64_t r = 0;
    #pragma unroll
    for (int k = 0; k < K_LANES; ++k) r ^= x[k];
    sink[tid] = r;
}

int main(int argc, char** argv) {
    int warmup_runs = 3;
    int timed_runs  = 5;
    int iters       = 4096;
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

    // Print device summary.
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("device: %s  sm=%d.%d  SMs=%d  totalGlobalMem=%.2f GB\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount,
           prop.totalGlobalMem / 1024.0 / 1024.0 / 1024.0);
    printf("config: blocks=%d threads=%d K_LANES=%d iters=%d  warmup=%d runs=%d\n",
           blocks, threads, K_LANES, iters, warmup_runs, timed_runs);

    size_t n_threads = (size_t)blocks * threads;
    uint64_t* d_sink;
    cudaMalloc(&d_sink, n_threads * sizeof(uint64_t));

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    for (int w = 0; w < warmup_runs; ++w) {
        bench_mul_kernel<<<blocks, threads>>>(d_sink, 0xC0FFEEULL + w, iters);
    }
    cudaDeviceSynchronize();

    double best_gops = 0.0;
    for (int r = 0; r < timed_runs; ++r) {
        cudaEventRecord(t0);
        bench_mul_kernel<<<blocks, threads>>>(d_sink, 0x1234ULL + r, iters);
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, t0, t1);

        double total_ops = (double)n_threads * (double)K_LANES * (double)iters;
        double gops = total_ops / (ms * 1e6);   // ms→s, ops→Gops
        if (gops > best_gops) best_gops = gops;
        printf("  run %d: %.3f ms  -> %.2f Gmul/s\n", r, ms, gops);
    }
    printf("best: %.2f Gmul/s\n", best_gops);

    // Read sink to ensure the kernel actually committed.
    uint64_t scratch;
    cudaMemcpy(&scratch, d_sink, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    printf("sink[0] = %016lx (anti-DCE)\n", scratch);

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
