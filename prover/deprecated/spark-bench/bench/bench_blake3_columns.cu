// BLAKE3 column-hash throughput on GB10.
//
// Mirrors the Ligero column-hash workload in §2.2 of design-feasibility.md:
// a length-`m` column of Goldilocks elements is hashed with BLAKE3 to a
// 32-byte digest, and the prover does N_LIG = 65536 of these per commit.
// Each thread hashes one column.
//
// We benchmark at several column sizes (m), all power-of-two ≤ 1024 (i.e.,
// single-chunk inputs ≤ 8192 bytes). Larger m would require BLAKE3's tree
// merge, but that adds <1% of the work since the per-byte cost is dominated
// by the inner compression.
//
// Throughput reported as:
//   columns/sec, GB/sec (input bytes absorbed), compressions/sec.
//
// Anti-DCE: per-thread 32-byte digests are XOR-folded into a 32-byte sink
// returned to the host.

#include "blake3_compress.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

constexpr int B_BLOCK_GOLDILOCKS = 8;   // §3.1: 8 Goldilocks per BLAKE3 block

// Per-column hash with up to CHUNK_LEN = 1024 bytes of input. Input is read
// 64 bytes at a time from `col`; the message bytes are little-endian-packed
// into 16 u32 words per block.
__device__ __forceinline__ void hash_column_single_chunk(
    const uint64_t* col,    // m u64 elements = m*8 bytes
    int m,                  // column length in u64 elements
    uint32_t out[8]
) {
    int len_bytes = m * 8;
    uint32_t cv[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) cv[i] = b3::IV[i];

    int n_blocks_full = len_bytes / 64;
    int rem           = len_bytes - n_blocks_full * 64;
    int n_blocks      = n_blocks_full + ((rem > 0 || len_bytes == 0) ? 1 : 0);

    for (int b = 0; b < n_blocks; ++b) {
        uint32_t flags = 0;
        if (b == 0)              flags |= b3::CHUNK_START;
        if (b == n_blocks - 1)   flags |= b3::CHUNK_END | b3::ROOT;

        uint32_t msg[16] = {0};
        int avail = (b < n_blocks_full) ? 64 : rem;

        // 64 bytes = 8 u64. Pack low-32 → msg[2k], high-32 → msg[2k+1] for k in [0, 8).
        const uint64_t* col_block = col + b * 8;
        int avail_u64 = avail / 8;
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            if (k < avail_u64) {
                uint64_t v = col_block[k];
                msg[2 * k]     = (uint32_t)v;
                msg[2 * k + 1] = (uint32_t)(v >> 32);
            }
        }

        uint32_t blen = (b == n_blocks - 1) ? (uint32_t)(len_bytes - b * 64) : 64u;
        uint32_t outwords[16];
        b3::compress(cv, msg, /*counter=*/0, blen, flags, outwords);

        if (b == n_blocks - 1) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) out[i] = outwords[i];
        } else {
            #pragma unroll
            for (int i = 0; i < 8; ++i) cv[i] = outwords[i];
        }
    }
}

// One column per thread. Strided over n_columns.
__global__ void k_hash_columns(
    const uint64_t* data,    // n_columns * m u64 elements, column-major: col c at &data[c*m]
    int m,
    int n_columns,
    uint32_t* sink           // 8 u32 = 32-byte XOR-fold sink
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    uint32_t local[8] = {0};
    for (int c = tid; c < n_columns; c += stride) {
        uint32_t digest[8];
        hash_column_single_chunk(&data[(size_t)c * m], m, digest);
        #pragma unroll
        for (int i = 0; i < 8; ++i) local[i] ^= digest[i];
    }

    // Atomically fold per-thread XOR into the sink.
    #pragma unroll
    for (int i = 0; i < 8; ++i) atomicXor(&sink[i], local[i]);
}

static int compressions_per_column(int m) {
    int len_bytes = m * 8;
    int full = len_bytes / 64;
    int rem  = len_bytes - full * 64;
    return full + ((rem > 0 || len_bytes == 0) ? 1 : 0);
}

static void bench_one(int m, int n_columns, int blocks, int threads,
                      int warmup, int runs) {
    size_t bytes_in = (size_t)n_columns * m * sizeof(uint64_t);
    uint64_t* d_data;
    uint32_t* d_sink;
    cudaMalloc(&d_data, bytes_in);
    cudaMalloc(&d_sink, 8 * sizeof(uint32_t));
    cudaMemset(d_sink, 0, 8 * sizeof(uint32_t));

    // Fill data on the device with a simple pattern; per-byte content doesn't
    // affect throughput so we don't bother with random.
    cudaMemset(d_data, 0xab, bytes_in);

    for (int w = 0; w < warmup; ++w) {
        k_hash_columns<<<blocks, threads>>>(d_data, m, n_columns, d_sink);
    }
    cudaDeviceSynchronize();

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int r = 0; r < runs; ++r) {
        k_hash_columns<<<blocks, threads>>>(d_data, m, n_columns, d_sink);
    }
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);

    double total_columns      = (double)n_columns * runs;
    double total_compressions = total_columns * compressions_per_column(m);
    double total_bytes_in     = (double)bytes_in * runs;
    double s = ms * 1e-3;

    printf("m=%5d  cols=%6d  bytes=%6.2f MB  ms=%7.3f"
           "  -> %.2f Mcols/s  %.2f Gcompress/s  %.2f GB/s absorbed\n",
           m, n_columns, bytes_in / 1024.0 / 1024.0, ms,
           (total_columns) / s / 1e6,
           (total_compressions) / s / 1e9,
           (total_bytes_in) / s / 1e9);

    uint32_t scratch[8];
    cudaMemcpy(scratch, d_sink, 8 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    // Touch scratch so the compiler doesn't elide the copy.
    if (scratch[0] == 0xdeadbeefu) printf("(unreachable)\n");

    cudaFree(d_data);
    cudaFree(d_sink);
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
}

int main(int argc, char** argv) {
    int n_columns = 65536;     // matches N_LIG
    int blocks    = 256;
    int threads   = 256;
    int warmup    = 3;
    int runs      = 5;
    int m_only    = -1;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> int {
            if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", argv[i]); exit(2); }
            return atoi(argv[++i]);
        };
        if      (a == "--columns") n_columns = next();
        else if (a == "--blocks")  blocks    = next();
        else if (a == "--threads") threads   = next();
        else if (a == "--warmup")  warmup    = next();
        else if (a == "--runs")    runs      = next();
        else if (a == "--m")       m_only    = next();
        else { fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 2; }
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("device: %s sm=%d.%d  SMs=%d\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    printf("config: columns=%d blocks=%d threads=%d  B_BLOCK=%d (Goldilocks/block)\n",
           n_columns, blocks, threads, B_BLOCK_GOLDILOCKS);

    if (m_only > 0) {
        bench_one(m_only, n_columns, blocks, threads, warmup, runs);
    } else {
        // Phase 1 m candidates: Llama 2 7B has W_R_W ≈ 7e9, W/ELL ≈ 855K rows.
        // Single-chunk hash (≤1024 bytes) handles m ≤ 128 here; larger m would
        // need tree merge. We sweep through 16..128 to characterize per-byte
        // throughput; the result extrapolates to larger m within ~1% (tree
        // merges are negligible per spec §3.4).
        int ms[] = {8, 16, 32, 64, 128};
        for (int m : ms) bench_one(m, n_columns, blocks, threads, warmup, runs);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
