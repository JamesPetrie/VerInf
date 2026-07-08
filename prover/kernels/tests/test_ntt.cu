// NTT correctness test: compare CUDA forward NTT against the Python
// reference, then check that inverse(forward(x)) == x for size 65536.
//
// Usage:
//   python3 goldilocks_ref.py emit_ntt_vectors N=1024 | ./test_ntt
//
// Input format (from goldilocks_ref.py):
//   line 1: n
//   line 2: n hex u64 values (input)
//   line 3: n hex u64 values (expected forward NTT output)

#include "../ntt.cuh"
#include <cstdio>
#include <cstdlib>
#include <vector>

static int read_hex_line(std::vector<uint64_t>& out, int n) {
    out.resize(n);
    for (int i = 0; i < n; ++i) {
        if (scanf(" %lx", &out[i]) != 1) return -1;
    }
    return 0;
}

int main() {
    int n = 0;
    if (scanf(" %d", &n) != 1) { fprintf(stderr, "couldn't read n\n"); return 1; }
    std::vector<uint64_t> h_in, h_expect;
    if (read_hex_line(h_in, n)     != 0) { fprintf(stderr, "couldn't read input\n");  return 1; }
    if (read_hex_line(h_expect, n) != 0) { fprintf(stderr, "couldn't read output\n"); return 1; }

    gl_ntt::Ctx ctx;
    gl_ntt::ntt_init(n, &ctx);

    uint64_t* d_a;
    cudaMalloc(&d_a, n * sizeof(uint64_t));
    cudaMemcpy(d_a, h_in.data(), n * sizeof(uint64_t), cudaMemcpyHostToDevice);

    enum Variant { BASELINE, FUSED, BAILEY };
    auto run_variant = [&](Variant v, const char* label) -> int {
        cudaMemcpy(d_a, h_in.data(), n * sizeof(uint64_t), cudaMemcpyHostToDevice);
        switch (v) {
            case BASELINE: gl_ntt::ntt_forward(d_a, ctx); break;
            case FUSED:    gl_ntt::ntt_forward_fused(d_a, ctx); break;
            case BAILEY:   gl_ntt::ntt_forward_bailey(d_a, ctx); break;
        }
        cudaDeviceSynchronize();

        std::vector<uint64_t> h_fwd(n);
        cudaMemcpy(h_fwd.data(), d_a, n * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        int fwd_diffs = 0;
        for (int i = 0; i < n; ++i) if (h_fwd[i] != h_expect[i]) ++fwd_diffs;
        if (fwd_diffs > 0) {
            for (int i = 0, shown = 0; i < n && shown < 3; ++i) {
                if (h_fwd[i] != h_expect[i]) {
                    fprintf(stderr, "  [%s] fwd diff at i=%d: got %016lx want %016lx\n",
                            label, i, h_fwd[i], h_expect[i]); ++shown;
                }
            }
        }

        switch (v) {
            case BASELINE: gl_ntt::ntt_inverse(d_a, ctx); break;
            case FUSED:    gl_ntt::ntt_inverse_fused(d_a, ctx); break;
            case BAILEY:   gl_ntt::ntt_inverse_bailey(d_a, ctx); break;
        }
        cudaDeviceSynchronize();
        std::vector<uint64_t> h_rt(n);
        cudaMemcpy(h_rt.data(), d_a, n * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        int rt_diffs = 0;
        for (int i = 0; i < n; ++i) if (h_rt[i] != h_in[i]) ++rt_diffs;
        if (rt_diffs > 0) {
            for (int i = 0, shown = 0; i < n && shown < 3; ++i) {
                if (h_rt[i] != h_in[i]) {
                    fprintf(stderr, "  [%s] rt diff at i=%d: got %016lx want %016lx\n",
                            label, i, h_rt[i], h_in[i]); ++shown;
                }
            }
        }

        printf("test_ntt n=%d  %-8s  fwd diffs=%d  round-trip diffs=%d\n",
               n, label, fwd_diffs, rt_diffs);
        return (fwd_diffs == 0 && rt_diffs == 0) ? 0 : 1;
    };

    int rc_baseline = run_variant(BASELINE, "baseline");
    int rc_fused    = run_variant(FUSED,    "fused");
    int rc_bailey   = (n == 65536) ? run_variant(BAILEY, "bailey") : 0;

    cudaFree(d_a);
    gl_ntt::ntt_destroy(&ctx);
    return (rc_baseline | rc_fused | rc_bailey);
}
