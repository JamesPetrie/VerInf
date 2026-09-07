// Batched Goldilocks NTT throughput — the prover's actual path.
//
// Question this answers: what does an NTT cost per element when transforms
// are BATCHED, as the prover's encode/fold path runs them — not one 512 KB
// transform at a time (which fits entirely in a B200's L2 and times kernel
// launches, the failure mode the first calibration session measured as a
// bogus 1.55x).
//
// The single-transform bench_ntt stays as the launch-overhead probe; THIS
// number is the one the A-constant's bandwidth-scaling story rests on.
//
// Build (from repo root; arch must match the card):
//   nvcc -arch=<sm_XXX> -std=c++17 -O3 -Iprover/kernels \
//        profiler/bench/bench_ntt_batched.cu -o bench_ntt_batched
//
// Usage:
//   ./bench_ntt_batched                       # n=65536, batch sweep
//   ./bench_ntt_batched --n 65536 --batch 512 --warmup 3 --runs 5

#include "ntt.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

static void bench_one(int n, int m, int warmup, int runs) {
    gl_ntt::Ctx ctx;
    gl_ntt::ntt_init(n, &ctx);
    size_t elems = (size_t)n * m;
    uint64_t *d_a, *d_scratch;
    if (cudaMalloc(&d_a, elems * sizeof(uint64_t)) != cudaSuccess ||
        cudaMalloc(&d_scratch, elems * sizeof(uint64_t)) != cudaSuccess) {
        fprintf(stderr, "alloc failed for n=%d m=%d (%.2f GB)\n",
                n, m, 2.0 * elems * 8 / 1e9);
        gl_ntt::ntt_destroy(&ctx);
        return;
    }
    cudaMemset(d_a, 0x11, elems * sizeof(uint64_t));

    for (int w = 0; w < warmup; ++w) {
        gl_ntt::ntt_forward_batched_fast(d_a, m, ctx, d_scratch);
        gl_ntt::ntt_inverse_batched_fast(d_a, m, ctx, d_scratch);
    }
    cudaDeviceSynchronize();

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int r = 0; r < runs; ++r) {
        gl_ntt::ntt_forward_batched_fast(d_a, m, ctx, d_scratch);
        gl_ntt::ntt_inverse_batched_fast(d_a, m, ctx, d_scratch);
    }
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);

    double total_ntts = 2.0 * runs * m;               // fwd + inv
    double us_per = ms * 1000.0 / total_ntts;
    double ns_per_elem = us_per * 1000.0 / n;
    printf("n=%6d m=%5d  fwd+inv x %d  total=%9.3f ms  -> %8.3f us/NTT  "
           "%.4f ns/elem\n", n, m, runs, ms, us_per, ns_per_elem);

    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    cudaFree(d_a);
    cudaFree(d_scratch);
    gl_ntt::ntt_destroy(&ctx);
}

int main(int argc, char** argv) {
    int n = 65536, batch = -1, warmup = 3, runs = 5;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> int {
            if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", argv[i]); exit(2); }
            return atoi(argv[++i]);
        };
        if      (a == "--n")      n = next();
        else if (a == "--batch")  batch = next();
        else if (a == "--warmup") warmup = next();
        else if (a == "--runs")   runs = next();
        else { fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 2; }
    }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("device: %s  sm=%d.%d  SMs=%d\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    if (batch > 0) {
        bench_one(n, batch, warmup, runs);
    } else {
        int ms[] = {1, 8, 64, 512, 2048};
        for (int m : ms) bench_one(n, m, warmup, runs);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
