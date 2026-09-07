// Kernel-launch overhead: synced round trip vs back-to-back stream rate.
//
// Two numbers:
//   sync   — launch + cudaDeviceSynchronize per iteration: the full
//            host-visible round trip (what a challenge-response loop or a
//            naive per-claim orchestrator pays).
//   stream — N launches then one sync: the amortized enqueue cost (what a
//            well-pipelined per-claim schedule pays).
//
// Consumers: the machine profile (gpu.launch_us_sync / launch_us_stream),
// the orchestration analysis (Python-vs-launch bound), and the HBM-state
// challenge response-window floor.
//
// Build (from repo root):
//   nvcc -arch=<sm_XXX> -std=c++17 -O3 \
//        profiler/bench/bench_launch_latency.cu -o bench_launch_latency

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <chrono>

__global__ void k_empty(uint64_t* sink) {
    if (blockIdx.x == 0 && threadIdx.x == 0) *sink += 1;   // anti-DCE
}

int main(int argc, char** argv) {
    int iters = 2000, warmup = 200;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> int {
            if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", argv[i]); exit(2); }
            return atoi(argv[++i]);
        };
        if      (a == "--iters")  iters = next();
        else if (a == "--warmup") warmup = next();
        else { fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 2; }
    }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("device: %s  sm=%d.%d\n", prop.name, prop.major, prop.minor);

    uint64_t* d_sink;
    cudaMalloc(&d_sink, sizeof(uint64_t));
    for (int i = 0; i < warmup; ++i) k_empty<<<1, 32>>>(d_sink);
    cudaDeviceSynchronize();

    using clk = std::chrono::steady_clock;

    auto a0 = clk::now();
    for (int i = 0; i < iters; ++i) {
        k_empty<<<1, 32>>>(d_sink);
        cudaDeviceSynchronize();
    }
    auto a1 = clk::now();
    double sync_us = std::chrono::duration<double, std::micro>(a1 - a0)
                         .count() / iters;

    auto b0 = clk::now();
    for (int i = 0; i < iters; ++i) k_empty<<<1, 32>>>(d_sink);
    cudaDeviceSynchronize();
    auto b1 = clk::now();
    double stream_us = std::chrono::duration<double, std::micro>(b1 - b0)
                           .count() / iters;

    printf("sync: %.2f us/launch (launch + device sync round trip)\n", sync_us);
    printf("stream: %.3f us/launch (back-to-back enqueue, one sync)\n", stream_us);

    cudaFree(d_sink);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
