// Dense Goldilocks matmul benchmark on GB10.
//
// Question: at Maverick-scale dimensions, how fast can the prover do the
// forward-pass simulation in the field? Per matmul, the prover computes
// Y = A · X (or equivalents) in Goldilocks arithmetic so the witness
// values are well-defined for the Ligero constraints.
//
// Default shape: A is (16384, 5120) — Maverick dense FFN gate/up weight.
// B is (n, 5120). Output Y = A · B^T is (16384, n). We sweep n.
//
// Kernel: naive per-output-thread. Each thread computes one Y[i, j] by
// looping over the contracted dimension k. Coalesced reads on B (threads
// in the same warp share j → same B row); A reads broadcast across the
// row direction. Memory-bandwidth-bound at small n; arithmetic-bound at
// large n once A is reused enough.

#include "goldilocks.cuh"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>

__global__ void k_gl_matmul_at_bt(
    const uint64_t* __restrict__ A,    // m × k row-major
    const uint64_t* __restrict__ B,    // n × k row-major (B^T against A)
    uint64_t* __restrict__ Y,          // m × n row-major
    int m, int k, int n
) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m || j >= n) return;
    uint64_t s = 0;
    const uint64_t* a_row = A + (size_t)i * k;
    const uint64_t* b_row = B + (size_t)j * k;
    #pragma unroll 8
    for (int l = 0; l < k; ++l) {
        s = gl::add(s, gl::mul(a_row[l], b_row[l]));
    }
    Y[(size_t)i * n + j] = s;
}

static void bench(int m, int k, int n, int warmup, int runs) {
    size_t A_sz = (size_t)m * (size_t)k * sizeof(uint64_t);
    size_t B_sz = (size_t)n * (size_t)k * sizeof(uint64_t);
    size_t Y_sz = (size_t)m * (size_t)n * sizeof(uint64_t);

    uint64_t *d_A, *d_B, *d_Y;
    if (cudaMalloc(&d_A, A_sz) != cudaSuccess
        || cudaMalloc(&d_B, B_sz) != cudaSuccess
        || cudaMalloc(&d_Y, Y_sz) != cudaSuccess) {
        fprintf(stderr, "alloc failed for n=%d (A+B+Y = %.2f GB)\n",
                n, (A_sz + B_sz + Y_sz) / 1024.0 / 1024.0 / 1024.0);
        return;
    }
    cudaMemset(d_A, 0xab, A_sz);
    cudaMemset(d_B, 0xcd, B_sz);

    dim3 block(16, 16);
    dim3 grid((n + 15) / 16, (m + 15) / 16);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);

    for (int w = 0; w < warmup; ++w) {
        k_gl_matmul_at_bt<<<grid, block>>>(d_A, d_B, d_Y, m, k, n);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(t0);
    for (int r = 0; r < runs; ++r) {
        k_gl_matmul_at_bt<<<grid, block>>>(d_A, d_B, d_Y, m, k, n);
    }
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms_total = 0;
    cudaEventElapsedTime(&ms_total, t0, t1);
    double per_run = ms_total / runs;

    double ops_per_run = (double)m * (double)k * (double)n;     // # mul-adds
    double tput_gops   = ops_per_run / (per_run * 1e-3) / 1e9;  // Gmul-add/s
    double floor_ms    = ops_per_run / 312e9 * 1000.0;          // floor at peak Gmul/s

    printf("n=%5d  ops=%6.2e  time/run=%8.2f ms  throughput=%6.2f Gmul/s  (%.1fx peak floor)\n",
           n, ops_per_run, per_run, tput_gops, per_run / floor_ms);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_Y);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
}

int main(int argc, char** argv) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("device: %s sm=%d.%d  SMs=%d  total mem=%.1f GB\n\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount,
           prop.totalGlobalMem / 1024.0 / 1024.0 / 1024.0);

    int m = 16384, k = 5120;
    printf("matmul sweep: A(%d, %d) @ B^T(n, %d) = Y(%d, n)\n", m, k, k, m);
    printf("Arithmetic floor at 312 Gmul/s (measured peak from bench_field_mul);\n");
    printf("naive kernel expected ~2-4x slower from memory traffic.\n\n");

    int n_values[] = {1, 2, 8, 32, 128, 512, 2048, 8192};
    for (int n : n_values) {
        bench(m, k, n, /*warmup=*/2, /*runs=*/3);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
