# Spark microbenchmark Makefile.
# Target: NVIDIA GB10 (DGX Spark, Blackwell, sm_121).
# CUDA toolkit lives at /usr/local/cuda; PATH is set in ~claude/.bashrc.
#
# Architecture note: GB10 is sm_121. If you build this for sm_90 (H100)
# the kernels will run but produce wrong numbers — same failure mode that
# bit zkllm-entropy with sm_100 vs sm_90.
#
# Scope after the 2026-05-16 deprecation: builds only the primitive
# validators (test_*) and microbenchmarks (bench_*). The full prover +
# verifier pipeline (commit_weights, gl_matmul_file, freivalds_at_scale,
# etc.) lives under ligero/deprecated/ — the prior architecture being
# replaced. Primitives (Goldilocks, NTT, BLAKE3) are kept here as
# reusable building blocks for any future Ligero-shape prover.
#
# Layout (relative to this Makefile, which lives at ligero/spark-bench/):
#   ../cuda/   — CUDA headers (goldilocks/ntt/blake3) + validator programs
#   bench/     — CUDA microbenchmarks (this dir)
#   ../ref/    — Python references (goldilocks_ref, splitmix64, polynomials)
#   drivers/   — emit_blake3_vectors.py (test-vector emitter for test_blake3_multi)
# Binaries land at the spark-bench root.

NVCC ?= /usr/local/cuda/bin/nvcc
ARCH ?= sm_121
CUDA_DIR := ../cuda
REF_DIR  := ../ref
# -I$(CUDA_DIR) so .cu files in bench/ can `#include "goldilocks.cuh"`
# etc. by bare name. (../cuda/test_*.cu live next to the headers so don't
# strictly need this, but it's harmless for them.)
NVCC_FLAGS := -arch=$(ARCH) -std=c++17 -O3 -lineinfo -I$(CUDA_DIR)

TARGETS := test_field bench_field_mul test_ntt bench_ntt test_blake3 test_blake3_multi bench_blake3_columns bench_goldilocks_matmul

all: $(TARGETS)

test_field: $(CUDA_DIR)/test_field.cu $(CUDA_DIR)/goldilocks.cuh
	$(NVCC) $(NVCC_FLAGS) $(CUDA_DIR)/test_field.cu -o test_field

bench_field_mul: bench/bench_field_mul.cu $(CUDA_DIR)/goldilocks.cuh
	$(NVCC) $(NVCC_FLAGS) bench/bench_field_mul.cu -o bench_field_mul

test_ntt: $(CUDA_DIR)/test_ntt.cu $(CUDA_DIR)/goldilocks.cuh $(CUDA_DIR)/ntt.cuh
	$(NVCC) $(NVCC_FLAGS) $(CUDA_DIR)/test_ntt.cu -o test_ntt

bench_ntt: bench/bench_ntt.cu $(CUDA_DIR)/goldilocks.cuh $(CUDA_DIR)/ntt.cuh
	$(NVCC) $(NVCC_FLAGS) bench/bench_ntt.cu -o bench_ntt

test_blake3: $(CUDA_DIR)/test_blake3.cu $(CUDA_DIR)/blake3_compress.cuh
	$(NVCC) $(NVCC_FLAGS) $(CUDA_DIR)/test_blake3.cu -o test_blake3

test_blake3_multi: $(CUDA_DIR)/test_blake3_multi.cu $(CUDA_DIR)/blake3_hash.cuh $(CUDA_DIR)/blake3_compress.cuh
	$(NVCC) $(NVCC_FLAGS) $(CUDA_DIR)/test_blake3_multi.cu -o test_blake3_multi

bench_blake3_columns: bench/bench_blake3_columns.cu $(CUDA_DIR)/blake3_compress.cuh
	$(NVCC) $(NVCC_FLAGS) bench/bench_blake3_columns.cu -o bench_blake3_columns

bench_goldilocks_matmul: bench/bench_goldilocks_matmul.cu $(CUDA_DIR)/goldilocks.cuh
	$(NVCC) $(NVCC_FLAGS) bench/bench_goldilocks_matmul.cu -o bench_goldilocks_matmul

# Convenience: validate field arithmetic, NTT, and BLAKE3 against references.
check: test_field test_ntt test_blake3
	python3 $(REF_DIR)/goldilocks_ref.py emit_field_vectors | ./test_field
	python3 $(REF_DIR)/goldilocks_ref.py emit_ntt_vectors N=1024 | ./test_ntt
	python3 $(REF_DIR)/goldilocks_ref.py emit_ntt_vectors N=65536 | ./test_ntt
	./test_blake3

clean:
	rm -f $(TARGETS)

.PHONY: all check clean
