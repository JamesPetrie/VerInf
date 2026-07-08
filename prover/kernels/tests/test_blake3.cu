// Validate b3::compress() and the single-chunk hash against the BLAKE3
// project's published test vectors.
//
// Vectors taken from
// https://github.com/BLAKE3-team/BLAKE3/blob/master/test_vectors/test_vectors.json
// (committed snapshot; the relevant subset is small enough to embed).
//
// We test:
//   1. Hash of 0-byte input.
//   2. Hash of 1-byte input (0x00).
//   3. Hash of 64-byte input (single block, all 0x00).
//   4. Hash of 1024-byte input (full chunk = 16 blocks).
//
// These cover (a) compression with no message bytes, (b) partial-block,
// (c) one full block, (d) chained compressions within a chunk.

#include "../blake3_compress.cuh"
#include <cstdio>
#include <cstring>
#include <vector>

// Hash a single chunk (input length ≤ 1024 bytes). For the chunk's first
// block flags|=CHUNK_START; for the last flags|=CHUNK_END; if it's also the
// root (no tree merge), flags|=ROOT.
__device__ void hash_chunk_root(
    const uint8_t* data, int len,
    uint32_t out[8]
) {
    uint32_t cv[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) cv[i] = b3::IV[i];

    int n_blocks_full = len / 64;
    int rem           = len - n_blocks_full * 64;
    int n_blocks      = n_blocks_full + ((rem > 0 || len == 0) ? 1 : 0);

    for (int b = 0; b < n_blocks; ++b) {
        bool is_first = (b == 0);
        bool is_last  = (b == n_blocks - 1);
        uint32_t flags = 0;
        if (is_first) flags |= b3::CHUNK_START;
        if (is_last)  flags |= b3::CHUNK_END | b3::ROOT;

        // Pack 64 bytes (zero-padded) into 16 words.
        uint32_t m[16] = {0};
        int avail = (b < n_blocks_full) ? 64 : rem;
        for (int i = 0; i < avail; ++i) {
            ((uint8_t*)m)[i] = data[b * 64 + i];
        }
        uint32_t blen = is_last ? (uint32_t)(len - b * 64) : 64u;
        if (len == 0) blen = 0;

        uint32_t outwords[16];
        b3::compress(cv, m, /*counter=*/0, blen, flags, outwords);
        if (is_last) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) out[i] = outwords[i];
        } else {
            #pragma unroll
            for (int i = 0; i < 8; ++i) cv[i] = outwords[i];
        }
    }
}

__global__ void run_hash(const uint8_t* data, int len, uint32_t* out) {
    hash_chunk_root(data, len, out);
}

// Convert 8 little-endian words to a 32-byte digest, as a hex string.
static void words_to_hex(const uint32_t* w, char* hex) {
    static const char* H = "0123456789abcdef";
    for (int i = 0; i < 8; ++i) {
        uint32_t x = w[i];
        for (int b = 0; b < 4; ++b) {
            uint8_t byte = (x >> (8 * b)) & 0xff;   // little-endian
            *hex++ = H[byte >> 4];
            *hex++ = H[byte & 0xf];
        }
    }
    *hex = 0;
}

struct Case {
    const char* name;
    int len;
    const char* want_hex;   // first 32 bytes of the published BLAKE3 test vector
};

int main() {
    // Published BLAKE3 test vectors (extended-output truncated to 32 bytes).
    // Inputs: zero-byte, 1 byte (0x00), 64 bytes, 1024 bytes — each input is
    // bytes[i] = i mod 251 per the test_vectors.json convention.
    Case cases[] = {
        {"len=0",    0,
         "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262"},
        {"len=1",    1,
         "2d3adedff11b61f14c886e35afa036736dcd87a74d27b5c1510225d0f592e213"},
        {"len=64",   64,
         "4eed7141ea4a5cd4b788606bd23f46e212af9cacebacdc7d1f4c6dc7f2511b98"},
        {"len=1024", 1024,
         "42214739f095a406f3fc83deb889744ac00df831c10daa55189b5d121c855af7"},
    };

    int fails = 0;
    uint8_t* d_data;
    uint32_t* d_out;
    cudaMalloc(&d_data, 1024);
    cudaMalloc(&d_out, 8 * sizeof(uint32_t));

    for (const Case& c : cases) {
        std::vector<uint8_t> input(c.len);
        for (int i = 0; i < c.len; ++i) input[i] = (uint8_t)(i % 251);
        cudaMemcpy(d_data, input.data(), c.len, cudaMemcpyHostToDevice);

        run_hash<<<1, 1>>>(d_data, c.len, d_out);
        uint32_t h_out[8];
        cudaMemcpy(h_out, d_out, 8 * sizeof(uint32_t), cudaMemcpyDeviceToHost);

        char got[65];
        words_to_hex(h_out, got);

        bool ok = (strcmp(got, c.want_hex) == 0);
        printf("%-9s  got=%s  %s\n", c.name, got, ok ? "OK" : "MISMATCH");
        if (!ok) {
            printf("           want=%s\n", c.want_hex);
            ++fails;
        }
    }

    cudaFree(d_data);
    cudaFree(d_out);
    printf("test_blake3: %d failures\n", fails);
    return fails == 0 ? 0 : 1;
}
