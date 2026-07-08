// NTT throughput benchmark on Goldilocks.
//
// Reports two numbers per length N:
//   - ns / NTT (single-NTT latency)
//   - butterflies / second  ( = (N/2) * log2(N) per NTT, divided by time )
//
// Method: warm-up runs first (allocate, prime caches), then a long batch
// where every NTT runs back-to-back on the same buffer. We ensure each
// NTT depends on the previous one's output by alternating forward and
// inverse on the same input — this prevents the launch loop from
// pipelining away the work.
//
// Usage:
//   ./bench_ntt                                # default sweep
//   ./bench_ntt --n 65536 --runs 200 --warmup 5

#include "ntt.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static double ns_per(double ms, int runs) { return (ms * 1e6) / runs; }

static void bench_one(int n, int warmup, int runs) {
    gl_ntt::Ctx ctx;
    gl_ntt::ntt_init(n, &ctx);

    // Random-ish input: use the additive hash of the index. Doesn't have to
    // be canonical-mod-p since p > 2^63; but we mask a bit to be safe.
    std::vector<uint64_t> h(n);
    for (int i = 0; i < n; ++i) {
        uint64_t x = (uint64_t)i * 0x9E3779B97F4A7C15ULL ^ (uint64_t)n * 0xBF58476D1CE4E5B9ULL;
        if (x >= gl::P) x -= gl::P;
        h[i] = x;
    }

    uint64_t* d_a;
    cudaMalloc(&d_a, (size_t)n * sizeof(uint64_t));
    cudaMemcpy(d_a, h.data(), (size_t)n * sizeof(uint64_t), cudaMemcpyHostToDevice);

    for (int w = 0; w < warmup; ++w) {
        gl_ntt::ntt_forward(d_a, ctx);
        gl_ntt::ntt_inverse(d_a, ctx);
    }
    cudaDeviceSynchronize();

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    enum Variant { BASELINE, FUSED, BAILEY };
    auto run_timed = [&](Variant v, const char* label) {
        cudaEventRecord(t0);
        for (int r = 0; r < runs; ++r) {
            switch (v) {
                case BASELINE:
                    gl_ntt::ntt_forward(d_a, ctx);
                    gl_ntt::ntt_inverse(d_a, ctx); break;
                case FUSED:
                    gl_ntt::ntt_forward_fused(d_a, ctx);
                    gl_ntt::ntt_inverse_fused(d_a, ctx); break;
                case BAILEY:
                    gl_ntt::ntt_forward_bailey(d_a, ctx);
                    gl_ntt::ntt_inverse_bailey(d_a, ctx); break;
            }
        }
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, t0, t1);

        int total_ntts = 2 * runs;
        double per_ns = ns_per(ms, total_ntts);
        double bf_per_ntt = (double)(n / 2) * (double)ctx.log2n;
        double bf_per_s = (bf_per_ntt * total_ntts) / (ms * 1e-3);

        printf("n=%6d log2n=%2d  %-8s  fwd+inv x %d  total=%7.3f ms  -> %.2f us/NTT  %.2f Gbutterfly/s\n",
               n, ctx.log2n, label, runs, ms, per_ns / 1e3, bf_per_s / 1e9);
    };

    run_timed(BASELINE, "baseline");
    if (n >= 256)   run_timed(FUSED,  "fused");
    if (n == 65536) run_timed(BAILEY, "bailey");

    cudaFree(d_a);
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    gl_ntt::ntt_destroy(&ctx);
}

int main(int argc, char** argv) {
    int n_override = -1;
    int warmup = 5;
    int runs = 100;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> int {
            if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", argv[i]); exit(2); }
            return atoi(argv[++i]);
        };
        if      (a == "--n")      n_override = next();
        else if (a == "--warmup") warmup     = next();
        else if (a == "--runs")   runs       = next();
        else { fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 2; }
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("device: %s  sm=%d.%d  SMs=%d\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount);

    if (n_override > 0) {
        bench_one(n_override, warmup, runs);
    } else {
        // Sweep relevant Ligero sizes plus context.
        int sizes[] = {1024, 4096, 16384, 65536};
        for (int n : sizes) bench_one(n, warmup, runs);
    }
    return 0;
}
