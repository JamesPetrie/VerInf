// Validate goldilocks.cuh add/sub/mul against the Python reference.
//
// Usage:
//   python3 goldilocks_ref.py emit_field_vectors | ./test_field
//
// Each input line is "a b a+b a-b a*b" in hex u64. We run the CUDA op and
// compare. One launch per vector — slow, but with ~60 vectors it's <1 s and
// the simplicity is worth more than the throughput.

#include "../goldilocks.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>

struct Vec { uint64_t a, b, sum, diff, prod; };

__global__ void run_one(uint64_t a, uint64_t b, uint64_t* out) {
    out[0] = gl::add(a, b);
    out[1] = gl::sub(a, b);
    out[2] = gl::mul(a, b);
}

int main() {
    char buf[256];
    int line = 0, fails = 0;
    uint64_t* d_out = nullptr;
    cudaMalloc(&d_out, 3 * sizeof(uint64_t));

    while (fgets(buf, sizeof(buf), stdin)) {
        ++line;
        Vec v;
        if (sscanf(buf, "%lx %lx %lx %lx %lx",
                   &v.a, &v.b, &v.sum, &v.diff, &v.prod) != 5) {
            fprintf(stderr, "line %d: parse error: %s", line, buf);
            return 1;
        }

        run_one<<<1, 1>>>(v.a, v.b, d_out);
        uint64_t h_out[3];
        cudaMemcpy(h_out, d_out, 3 * sizeof(uint64_t), cudaMemcpyDeviceToHost);

        if (h_out[0] != v.sum || h_out[1] != v.diff || h_out[2] != v.prod) {
            fprintf(stderr,
                "FAIL line %d  a=%016lx b=%016lx\n"
                "   got  add=%016lx sub=%016lx mul=%016lx\n"
                "   want add=%016lx sub=%016lx mul=%016lx\n",
                line, v.a, v.b,
                h_out[0], h_out[1], h_out[2],
                v.sum, v.diff, v.prod);
            ++fails;
        }
    }

    cudaFree(d_out);
    printf("test_field: %d cases, %d failures\n", line, fails);
    return fails == 0 ? 0 : 1;
}
