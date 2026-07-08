// Validate the multi-chunk BLAKE3 hash in blake3_hash.cuh against the
// Python `blake3` reference (which wraps the official BLAKE3 Rust impl).
//
// Usage:
//   python3 emit_blake3_vectors.py | ./test_blake3_multi
//
// The Python script emits, one per line:
//   <length_bytes> <expected_64-char_hex_digest>
//
// We construct the standard BLAKE3 test pattern (bytes[i] = i mod 251)
// of the requested length, hash with hash_bytes, compare.

#include "../blake3_hash.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

__global__ void run_one(const uint8_t* data, int len, uint32_t* out) {
    b3::hash_bytes(data, len, out);
}

static void words_to_hex(const uint32_t* w, char* hex) {
    static const char* H = "0123456789abcdef";
    for (int i = 0; i < 8; ++i) {
        uint32_t x = w[i];
        for (int b = 0; b < 4; ++b) {
            uint8_t byte = (x >> (8 * b)) & 0xff;
            *hex++ = H[byte >> 4];
            *hex++ = H[byte & 0xf];
        }
    }
    *hex = 0;
}

int main() {
    char buf[256];
    int line = 0;
    int fails = 0;

    // Reusable scratch buffers, sized for the largest vector we expect.
    constexpr int MAX_LEN = 1 << 20;     // 1 MB
    uint8_t* d_data; cudaMalloc(&d_data, MAX_LEN);
    uint32_t* d_out; cudaMalloc(&d_out, 8 * sizeof(uint32_t));

    while (fgets(buf, sizeof(buf), stdin)) {
        ++line;
        int len = -1;
        char want_hex[65] = {0};
        if (sscanf(buf, "%d %64s", &len, want_hex) != 2) {
            fprintf(stderr, "line %d: parse error: %s", line, buf);
            return 1;
        }
        if (len < 0 || len > MAX_LEN) {
            fprintf(stderr, "line %d: length %d out of range [0,%d]\n", line, len, MAX_LEN);
            return 1;
        }

        // Build input pattern.
        std::vector<uint8_t> input(len);
        for (int i = 0; i < len; ++i) input[i] = (uint8_t)(i % 251);
        cudaMemcpy(d_data, input.data(), len, cudaMemcpyHostToDevice);

        run_one<<<1, 1>>>(d_data, len, d_out);
        uint32_t h_out[8];
        cudaMemcpy(h_out, d_out, 8 * sizeof(uint32_t), cudaMemcpyDeviceToHost);

        char got_hex[65];
        words_to_hex(h_out, got_hex);

        bool ok = (strcmp(got_hex, want_hex) == 0);
        printf("len=%-8d  %s%s\n", len, got_hex, ok ? "  OK" : "  MISMATCH");
        if (!ok) {
            printf("                want=%s\n", want_hex);
            ++fails;
        }
    }

    cudaFree(d_data);
    cudaFree(d_out);
    printf("test_blake3_multi: %d cases, %d failures\n", line, fails);
    return fails == 0 ? 0 : 1;
}
