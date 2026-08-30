// HBM random-access characteristics: gather throughput and chase latency.
//
// Two numbers a sequential-copy bench cannot give:
//   gather — y[i] = x[idx[i]] with random idx over a buffer far larger
//            than L2: the achievable random 8B-granule read bandwidth.
//   chase  — a dependent pointer walk (next = x[next]): raw HBM access
//            LATENCY, with W independent walkers to show how much of it
//            parallelism hides.
//
// Consumers: the machine profile (gpu.hbm_random_GBps, gpu.hbm_chase_ns)
// and the HBM-state challenge protocol's f(x, nonce) design — both the
// "requires random access" discriminator and the response-window floor
// are set by these rates.
//
// Build (from repo root):
//   nvcc -arch=<sm_XXX> -std=c++17 -O3 -Iprover/kernels \
//        profiler/bench/bench_hbm_random.cu -o bench_hbm_random
//
// Usage:
//   ./bench_hbm_random                    # 8 GiB buffer, defaults
//   ./bench_hbm_random --gib 4 --hops 20000 --walkers 65536

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>

__global__ void k_fill_perm(uint64_t* x, uint64_t n, uint64_t seed) {
    // Pseudo-random single-cycle permutation via an affine map with an
    // odd multiplier: next = (a*i + c) mod n visits cells in a stride
    // pattern incoherent to the cache once a is large and odd.
    uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    uint64_t stride = gridDim.x * (uint64_t)blockDim.x;
    uint64_t a = (seed | 1uLL);
    for (uint64_t j = i; j < n; j += stride)
        x[j] = (a * j + 12345uLL) % n;
}

__global__ void k_gather(const uint64_t* x, uint64_t* y, uint64_t n,
                         uint64_t reads, uint64_t seed) {
    uint64_t tid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    uint64_t nthreads = gridDim.x * (uint64_t)blockDim.x;
    uint64_t per = reads / nthreads;
    uint64_t idx = (seed * 0x9E3779B97F4A7C15uLL + tid) % n;
    uint64_t acc = 0;
    for (uint64_t j = 0; j < per; ++j) {
        // LCG hop: uncorrelated 8B reads across the whole buffer
        idx = (6364136223846793005uLL * idx + 1442695040888963407uLL) % n;
        acc ^= x[idx];
    }
    y[tid] = acc;                        // anti-DCE
}

__global__ void k_chase(const uint64_t* x, uint64_t* y, int hops,
                        uint64_t n, uint64_t seed) {
    uint64_t tid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    uint64_t p = (seed + tid * 0xBF58476D1CE4E5B9uLL) % n;
    #pragma unroll 1
    for (int h = 0; h < hops; ++h)
        p = x[p];                        // DEPENDENT: latency-bound
    y[tid] = p;
}

int main(int argc, char** argv) {
    double gib = 8.0;
    long long hops = 20000, walkers = 65536, reads = 1LL << 28;
    int warmup = 2, runs = 3;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> long long {
            if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", argv[i]); exit(2); }
            return atoll(argv[++i]);
        };
        if      (a == "--gib")     gib = (double)next();
        else if (a == "--hops")    hops = next();
        else if (a == "--walkers") walkers = next();
        else if (a == "--reads")   reads = next();
        else if (a == "--warmup")  warmup = (int)next();
        else if (a == "--runs")    runs = (int)next();
        else { fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 2; }
    }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("device: %s  sm=%d.%d  SMs=%d  L2=%.0f MB\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount,
           prop.l2CacheSize / 1e6);

    uint64_t n = (uint64_t)(gib * (1uLL << 30)) / 8;
    uint64_t *d_x, *d_y;
    if (cudaMalloc(&d_x, n * 8) != cudaSuccess) {
        fprintf(stderr, "alloc failed (%.1f GiB)\n", gib);
        return 1;
    }
    cudaMalloc(&d_y, (size_t)walkers * 8);
    k_fill_perm<<<1024, 256>>>(d_x, n, 0x2545F4914F6CDD1DuLL);
    cudaDeviceSynchronize();
    printf("buffer: %.1f GiB (%llu cells)\n", gib, (unsigned long long)n);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    // ---- gather throughput -------------------------------------------
    for (int w = 0; w < warmup; ++w)
        k_gather<<<256, 256>>>(d_x, d_y, n, reads, w);
    cudaDeviceSynchronize();
    double best_gbps = 0.0;
    for (int r = 0; r < runs; ++r) {
        cudaEventRecord(t0);
        k_gather<<<256, 256>>>(d_x, d_y, n, reads, 0x1234 + r);
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        float ms = 0;
        cudaEventElapsedTime(&ms, t0, t1);
        double gbps = (double)reads * 8.0 / (ms * 1e6);
        if (gbps > best_gbps) best_gbps = gbps;
        printf("  gather run %d: %.1f ms -> %.2f GB/s\n", r, ms, gbps);
    }
    printf("gather best: %.2f GB/s (random 8B reads)\n", best_gbps);

    // ---- chase latency ------------------------------------------------
    int blocks = (int)((walkers + 255) / 256);
    for (int w = 0; w < warmup; ++w)
        k_chase<<<blocks, 256>>>(d_x, d_y, (int)hops, n, w);
    cudaDeviceSynchronize();
    double best_ns = 1e18;
    for (int r = 0; r < runs; ++r) {
        cudaEventRecord(t0);
        k_chase<<<blocks, 256>>>(d_x, d_y, (int)hops, n, 0x77 + r);
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        float ms = 0;
        cudaEventElapsedTime(&ms, t0, t1);
        double ns = ms * 1e6 / hops;     // per-hop wall across all walkers
        if (ns < best_ns) best_ns = ns;
        printf("  chase run %d: %.1f ms (%lld hops x %lld walkers)\n",
               r, ms, hops, walkers);
    }
    printf("chase best: %.2f ns/hop (%lld parallel walkers)\n",
           best_ns, walkers);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
