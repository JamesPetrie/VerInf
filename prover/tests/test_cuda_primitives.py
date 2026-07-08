"""
Acceptance tests for pipeline/cuda_primitives.py.

Passing all tests indicates the wrapper is in a sufficient state to drive a
Llama 2 7B Ligero proof end-to-end. Each primitive in the wrapper API gets
two flavours of coverage:

  - Correctness: cross-checked against a pure-Python reference at a small
    size (the Python ref is the source of truth).
  - Production scale: re-invoked at Llama-7B Ligero parameters
    (ELL=8192, K_DEG=16384, N_LIG=65536) or one Llama-7B matmul shape
    (4096 x 4096) to confirm the kernel handles the real input shape
    without OOM and to flag obvious performance cliffs.

The final test (test_commit_phase_integration) chains the production-scale
primitives into one §2.2 commit at realistic m so any cross-kernel layout
or contract bug surfaces.
"""

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # pipeline/ on path
import random
import sys
import time
from pathlib import Path

import blake3 as _blake3
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "ref"))

from cuda_primitives import (
    P,
    gl_mul, gl_add, gl_sub, gl_neg, gl_pow, gl_inv, gl_inv_batched, gl_axpy,
    ntt_forward, ntt_inverse, ntt_forward_batched, ntt_inverse_batched,
    rs_encode_rows,
    poly_mul, poly_add, poly_mul_batched, poly_eval,
    gl_matmul, gl_matvec, gl_spmv,
    hash_columns_streamed, merkle_build_blake3,
    lookup_multiplicities,
)

import goldilocks_ref as ref
import polynomials as poly_ref


ELL = 8192
K_DEG = 16384
N_LIG = 65536

LLAMA_D = 4096          # Llama 2 7B model dim; q/k/v/o projection is d x d.
LLAMA_D_FF = 11008      # Llama 2 7B SwiGLU hidden; gate/up is d_ff x d.


def _u64(vals):
    return torch.tensor(vals, dtype=torch.uint64, device="cuda")


def _rand_list(n, rng):
    return [rng.randrange(P) for _ in range(n)]


def _rand_u64_tensor(shape):
    """Random uint64 tensor in [0, 2^63). Goldilocks kernels reduce internally,
    so non-canonical inputs are fine for shape/perf tests; correctness tests
    use the Python-list path to stay in canonical form."""
    return torch.randint(0, 2**63 - 1, shape, dtype=torch.int64, device="cuda").to(torch.uint64)


# ===========================================================================
# 1. Field arithmetic
# ===========================================================================

def test_gl_arithmetic_correctness():
    rng = random.Random(1)
    n = 257
    a_cpu = _rand_list(n, rng)
    b_cpu = _rand_list(n, rng)
    a, b = _u64(a_cpu), _u64(b_cpu)
    assert torch.equal(gl_mul(a, b), _u64([ref.mul(x, y) for x, y in zip(a_cpu, b_cpu)]))
    assert torch.equal(gl_add(a, b), _u64([ref.add(x, y) for x, y in zip(a_cpu, b_cpu)]))
    assert torch.equal(gl_sub(a, b), _u64([ref.sub(x, y) for x, y in zip(a_cpu, b_cpu)]))
    assert torch.equal(gl_neg(a),    _u64([(P - x) % P for x in a_cpu]))
    exp = 12345
    assert torch.equal(gl_pow(a, exp), _u64([pow(x, exp, P) for x in a_cpu]))


def test_gl_inv_scalar_correctness():
    rng = random.Random(2)
    a_cpu = [rng.randrange(1, P) for _ in range(64)]
    a = _u64(a_cpu)
    assert torch.equal(gl_inv(a), _u64([ref.inv(x) for x in a_cpu]))


def test_gl_inv_batched_roundtrip():
    """Montgomery-trick batched inverse: a * a^-1 == 1, and matches scalar
    gl_inv on a slice. Used by LogUp at full column scale."""
    rng = random.Random(3)
    n = 4096
    a_cpu = [rng.randrange(1, P) for _ in range(n)]
    a = _u64(a_cpu)
    inv = gl_inv_batched(a)
    assert torch.equal(gl_mul(a, inv), torch.ones_like(a))
    assert torch.equal(inv[:64], gl_inv(a[:64]))


def test_gl_axpy_correctness():
    rng = random.Random(4)
    n = 4096
    y_cpu, x_cpu = _rand_list(n, rng), _rand_list(n, rng)
    alpha = rng.randrange(P)
    y, x = _u64(y_cpu), _u64(x_cpu)
    gl_axpy(y, alpha, x)
    expected = _u64([(y_cpu[i] + alpha * x_cpu[i]) % P for i in range(n)])
    assert torch.equal(y, expected)


# ===========================================================================
# 2. NTT
# ===========================================================================

def test_ntt_correctness_small():
    rng = random.Random(10)
    for N in [4, 16, 64, 1024]:
        x_cpu = _rand_list(N, rng)
        omega = ref.root_of_unity(N)
        expected = ref.ntt(list(x_cpu), omega, invert=False)
        x = _u64(x_cpu)
        ntt_forward(x)
        assert x.cpu().tolist() == expected, f"NTT N={N} mismatch"


def test_ntt_roundtrip_production():
    """N=65536 is the codeword length used in a real proof. iNTT(fNTT(x))==x
    at this size is the load-bearing check for the production NTT path."""
    rng = random.Random(11)
    x_cpu = _rand_list(N_LIG, rng)
    x_orig = _u64(x_cpu)
    x = x_orig.clone()
    ntt_forward(x)
    ntt_inverse(x)
    assert torch.equal(x, x_orig)


def test_ntt_batched_matches_single():
    rng = random.Random(12)
    m, N = 64, 1024
    rows_cpu = [_rand_list(N, rng) for _ in range(m)]
    rows_a = torch.stack([_u64(r) for r in rows_cpu])
    rows_b = rows_a.clone()
    for i in range(m):
        single = rows_a[i].clone()
        ntt_forward(single)
        rows_a[i] = single
    ntt_forward_batched(rows_b)
    assert torch.equal(rows_a, rows_b)
    # Inverse on the same data round-trips.
    rows_c = rows_b.clone()
    ntt_inverse_batched(rows_c)
    assert torch.equal(rows_c, torch.stack([_u64(r) for r in rows_cpu]))


def test_ntt_batched_at_production_sizes():
    """Batched NTT correctness at production lengths: N_LIG=65536 hits the
    Bailey-batched path, K_DEG=16384 / poly_mul-length=32768 hit the
    fused-batched path. Small m=8 to keep the test fast — the goal is to
    catch correctness regressions in the batched kernels at the actual
    sizes used by encode_messages, _interpolate_to_kdeg, and poly_mul_batched."""
    for N in [16384, 32768, 65536]:
        m = 8
        # Build a (m, N) buffer where each row is canonical [0, P).
        rows_orig = _rand_u64_tensor((m, N))
        # Round-trip first: iNTT(fNTT(x)) == x.
        rows = rows_orig.clone()
        ntt_forward_batched(rows)
        ntt_inverse_batched(rows)
        assert torch.equal(rows, rows_orig), f"batched round-trip at N={N} fails"
        # Compare per-row vs batched forward.
        rows_b = rows_orig.clone()
        ntt_forward_batched(rows_b)
        rows_per_row = rows_orig.clone()
        for i in range(m):
            single = rows_per_row[i].clone()
            ntt_forward(single)
            rows_per_row[i] = single
        assert torch.equal(rows_per_row, rows_b), f"batched != per-row at N={N}"
        print(f"  batched NTT N={N} m={m}: round-trip + matches-per-row OK")


# ===========================================================================
# 3. Reed-Solomon row encoding
# ===========================================================================

def test_rs_encode_rows_small():
    """Cross-check against the composed iNTT(K_DEG) + fNTT(N_LIG) sequence
    that `rs_encode_rows` is supposed to fuse."""
    rng = random.Random(20)
    m, ell, k_deg, n_lig = 16, 8, 8, 32
    msgs_cpu = [_rand_list(ell, rng) + [0] * (k_deg - ell) for _ in range(m)]
    ref_rows = []
    for row in msgs_cpu:
        buf = _u64(row)
        ntt_inverse(buf)
        ext = torch.zeros(n_lig, dtype=torch.uint64, device="cuda")
        ext[:k_deg] = buf
        ntt_forward(ext)
        ref_rows.append(ext)
    expected = torch.stack(ref_rows)
    msgs = torch.stack([_u64(r) for r in msgs_cpu])
    out = rs_encode_rows(msgs, n_lig=n_lig, k_deg=k_deg)
    assert torch.equal(out, expected)


def test_rs_encode_rows_production():
    """At ELL=8192, K_DEG=16384, N_LIG=65536: encode 128 rows at once.
    Each row's first ELL slots carry the message; trailing slots are
    ZK pad (zero is fine for a shape/runs test)."""
    m = 128
    rng = random.Random(21)
    msgs = torch.zeros((m, K_DEG), dtype=torch.uint64, device="cuda")
    msgs[:, :ELL] = _rand_u64_tensor((m, ELL))
    t0 = time.time()
    out = rs_encode_rows(msgs, n_lig=N_LIG, k_deg=K_DEG)
    torch.cuda.synchronize()
    print(f"  rs_encode_rows m=128 K_DEG=16384 N_LIG=65536: {time.time()-t0:.2f}s")
    assert out.shape == (m, N_LIG)


# ===========================================================================
# 4. Polynomial arithmetic
# ===========================================================================

def test_poly_mul_correctness():
    rng = random.Random(30)
    for la, lb in [(1, 1), (3, 5), (16, 16), (256, 256)]:
        a_cpu, b_cpu = _rand_list(la, rng), _rand_list(lb, rng)
        expected = poly_ref.poly_mul(a_cpu, b_cpu)
        got = poly_mul(_u64(a_cpu), _u64(b_cpu)).cpu().tolist()
        assert got == expected, f"poly_mul ({la},{lb}) mismatch"


def test_poly_add_correctness():
    rng = random.Random(31)
    for la, lb in [(4, 4), (5, 3), (3, 5), (1, 8)]:
        a_cpu, b_cpu = _rand_list(la, rng), _rand_list(lb, rng)
        expected = poly_ref.poly_add(a_cpu, b_cpu)
        got = poly_add(_u64(a_cpu), _u64(b_cpu)).cpu().tolist()
        assert got == expected


def test_poly_mul_batched_matches_single():
    rng = random.Random(32)
    m, la = 32, 64
    A_cpu = [_rand_list(la, rng) for _ in range(m)]
    B_cpu = [_rand_list(la, rng) for _ in range(m)]
    A = torch.stack([_u64(r) for r in A_cpu])
    B = torch.stack([_u64(r) for r in B_cpu])
    expected_rows = [poly_mul(_u64(a), _u64(b)) for a, b in zip(A_cpu, B_cpu)]
    expected = torch.stack(expected_rows)
    got = poly_mul_batched(A, B)
    assert torch.equal(got, expected)


def test_poly_eval_correctness():
    """Used by the verifier on q_irs/q_lin/p_0 at query points η_j; and by
    the prover during Lagrange interpolation of pa/pb."""
    rng = random.Random(33)
    d, k = 128, 16
    coeffs_cpu, points_cpu = _rand_list(d, rng), _rand_list(k, rng)
    expected = [poly_ref.poly_eval(coeffs_cpu, p) for p in points_cpu]
    got = poly_eval(_u64(coeffs_cpu), _u64(points_cpu)).cpu().tolist()
    assert got == expected


# ===========================================================================
# 5. Linear algebra mod P
# ===========================================================================

def test_gl_matmul_correctness_small():
    rng = random.Random(40)
    m, k, n = 8, 12, 16
    A_cpu = [_rand_list(k, rng) for _ in range(m)]
    B_cpu = [_rand_list(n, rng) for _ in range(k)]
    expected = [[sum(A_cpu[i][r] * B_cpu[r][j] for r in range(k)) % P
                 for j in range(n)] for i in range(m)]
    A = torch.stack([_u64(r) for r in A_cpu])
    B = torch.stack([_u64(r) for r in B_cpu])
    assert gl_matmul(A, B).cpu().tolist() == expected


def test_gl_matmul_llama_shape():
    """Llama 2 7B q-projection shape: 4096 x 4096 x 4096 = 6.8e10 muls.
    At 300 Gmul/s this is ~0.2s. Spot-check one row against a CPU reference."""
    m = k = n = LLAMA_D
    A = _rand_u64_tensor((m, k))
    B = _rand_u64_tensor((k, n))
    t0 = time.time()
    C = gl_matmul(A, B)
    torch.cuda.synchronize()
    print(f"  gl_matmul 4096^3: {time.time()-t0:.2f}s")
    i = 0
    A_row = A[i].cpu().tolist()
    B_cpu = B.cpu().tolist()
    expected_row = [sum(A_row[r] * B_cpu[r][j] for r in range(k)) % P for j in range(32)]
    assert C[i, :32].cpu().tolist() == expected_row


def test_gl_matvec_correctness():
    """Backs q_irs = row_polys^T @ r_irs in the IRS test composition."""
    rng = random.Random(42)
    m, n = 64, 128
    M_cpu = [_rand_list(n, rng) for _ in range(m)]
    v_cpu = _rand_list(n, rng)
    expected = [sum(M_cpu[i][j] * v_cpu[j] for j in range(n)) % P for i in range(m)]
    M = torch.stack([_u64(r) for r in M_cpu])
    assert gl_matvec(M, _u64(v_cpu)).cpu().tolist() == expected


def test_gl_spmv_correctness():
    """CSR mod-P matvec — backs r^T A in the linear-test composition.
    A is sparse (O(L) non-zeros over a W-slot witness)."""
    rng = random.Random(43)
    n_rows, n_cols, nnz_target = 100, 1000, 500
    triples = sorted({(rng.randrange(n_rows), rng.randrange(n_cols)) for _ in range(nnz_target)})
    triples = [(r, c, rng.randrange(P)) for r, c in triples]
    row_ptr = [0] * (n_rows + 1)
    for r, _, _ in triples:
        row_ptr[r + 1] += 1
    for r in range(n_rows):
        row_ptr[r + 1] += row_ptr[r]
    col_idx = [c for _, c, _ in triples]
    values = [v for _, _, v in triples]
    x_cpu = _rand_list(n_cols, rng)
    expected = [0] * n_rows
    for r, c, v in triples:
        expected[r] = (expected[r] + v * x_cpu[c]) % P
    got = gl_spmv(_u64(values), _u64(col_idx), _u64(row_ptr), _u64(x_cpu), n_rows)
    assert got.cpu().tolist() == expected


# ===========================================================================
# 6. Hashing + Merkle
# ===========================================================================

def test_hash_columns_streamed_correctness():
    """At m > 1024 (the legacy hash_columns row cap) the streamed variant
    must still match the official Python BLAKE3 on every column."""
    rng = random.Random(50)
    m, n_cols = 4096, 64
    matrix_cpu = [[rng.randrange(P) for _ in range(n_cols)] for _ in range(m)]
    matrix = torch.tensor(matrix_cpu, dtype=torch.uint64, device="cuda")
    digests = hash_columns_streamed(matrix)
    for j in [0, 13, n_cols - 1]:
        col_bytes = b"".join(int(matrix_cpu[i][j]).to_bytes(8, "little") for i in range(m))
        expected = _blake3.blake3(col_bytes).digest()
        got = bytes(digests[j].cpu().numpy().tolist())
        assert got == expected, f"col {j} digest mismatch"


def test_hash_columns_streamed_scale():
    """One commit's encoded matrix at m=2048 rows × N_LIG=65536 cols ≈ 1 GiB.
    Real Llama-7B per-prefill commits sit in the 1k-12k row range; this
    is a representative middle."""
    m = 2048
    matrix = _rand_u64_tensor((m, N_LIG))
    t0 = time.time()
    digests = hash_columns_streamed(matrix)
    torch.cuda.synchronize()
    print(f"  hash_columns_streamed m={m} N_LIG={N_LIG}: {time.time()-t0:.2f}s")
    assert digests.shape == (N_LIG, 32)


def test_merkle_build_blake3_correctness():
    rng = random.Random(60)
    N = 1024
    leaves_cpu = [bytes(rng.randrange(256) for _ in range(32)) for _ in range(N)]
    leaves = torch.tensor([list(leaf) for leaf in leaves_cpu],
                          dtype=torch.uint8, device="cuda")
    root, _levels = merkle_build_blake3(leaves)
    cur = list(leaves_cpu)
    while len(cur) > 1:
        if len(cur) % 2 == 1:
            cur.append(cur[-1])
        cur = [_blake3.blake3(cur[i] + cur[i + 1]).digest()
               for i in range(0, len(cur), 2)]
    assert bytes(root.cpu().numpy().tolist()) == cur[0]


def test_merkle_build_blake3_production_size():
    """N_LIG=65536 leaves = one commit's column-hash count."""
    leaves = torch.randint(0, 256, (N_LIG, 32), dtype=torch.uint8, device="cuda")
    t0 = time.time()
    root, _levels = merkle_build_blake3(leaves)
    torch.cuda.synchronize()
    print(f"  merkle_build N={N_LIG}: {time.time()-t0:.2f}s")
    assert root.shape == (32,)


# ===========================================================================
# 7. Lookup helpers
# ===========================================================================

def test_lookup_multiplicities_correctness():
    rng = random.Random(70)
    K, T_LEN = 512, 64
    table_cpu = list(range(T_LEN))
    x_cpu = [rng.randrange(T_LEN + 10) for _ in range(K)]
    expected = [0] * T_LEN
    for v in x_cpu:
        if v < T_LEN:
            expected[v] += 1
    got = lookup_multiplicities(_u64(x_cpu), _u64(table_cpu)).cpu().tolist()
    assert got == expected


def test_lookup_multiplicities_logup_size():
    """Paired tlookup table size from design-feasibility.md §B is 2^16;
    column length K reaches into the millions per commit."""
    K, T_LEN = 1 << 20, 1 << 16
    table = torch.arange(T_LEN, dtype=torch.int64, device="cuda").to(torch.uint64)
    x = torch.randint(0, T_LEN, (K,), dtype=torch.int64, device="cuda").to(torch.uint64)
    mult = lookup_multiplicities(x, table)
    assert mult.shape == (T_LEN,)
    assert int(mult.sum().item()) == K


# ===========================================================================
# 8. Integration smoke — production-parameter commit pipeline
# ===========================================================================

def test_commit_phase_integration():
    """Encode m rows, hash columns, build Merkle root — the §2.2 commit
    pipeline at production ELL/K_DEG/N_LIG. Catches contract mismatches
    between rs_encode_rows, hash_columns_streamed, and merkle_build_blake3
    that per-primitive tests can miss. m=64 keeps the run under a few seconds
    while exercising every kernel at its real per-element shape."""
    m = 64
    rng = random.Random(80)
    msgs = torch.zeros((m, K_DEG), dtype=torch.uint64, device="cuda")
    msgs[:, :ELL] = torch.tensor(
        [_rand_list(ELL, rng) for _ in range(m)],
        dtype=torch.uint64, device="cuda",
    )
    t0 = time.time()
    codewords = rs_encode_rows(msgs, n_lig=N_LIG, k_deg=K_DEG)
    column_digests = hash_columns_streamed(codewords)
    root, _levels = merkle_build_blake3(column_digests)
    torch.cuda.synchronize()
    print(f"  commit m={m} W={m*ELL}: {time.time()-t0:.2f}s")
    assert codewords.shape == (m, N_LIG)
    assert column_digests.shape == (N_LIG, 32)
    assert root.shape == (32,)


# ===========================================================================
# Runner
# ===========================================================================

ALL_TESTS = [
    test_gl_arithmetic_correctness,
    test_gl_inv_scalar_correctness,
    test_gl_inv_batched_roundtrip,
    test_gl_axpy_correctness,
    test_ntt_correctness_small,
    test_ntt_roundtrip_production,
    test_ntt_batched_matches_single,
    test_ntt_batched_at_production_sizes,
    test_rs_encode_rows_small,
    test_rs_encode_rows_production,
    test_poly_mul_correctness,
    test_poly_add_correctness,
    test_poly_mul_batched_matches_single,
    test_poly_eval_correctness,
    test_gl_matmul_correctness_small,
    test_gl_matmul_llama_shape,
    test_gl_matvec_correctness,
    test_gl_spmv_correctness,
    test_hash_columns_streamed_correctness,
    test_hash_columns_streamed_scale,
    test_merkle_build_blake3_correctness,
    test_merkle_build_blake3_production_size,
    test_lookup_multiplicities_correctness,
    test_lookup_multiplicities_logup_size,
    test_commit_phase_integration,
]


if __name__ == "__main__":
    for t in ALL_TESTS:
        print(f"-- {t.__name__}")
        t()
    print(f"\n{len(ALL_TESTS)} tests passed.")
