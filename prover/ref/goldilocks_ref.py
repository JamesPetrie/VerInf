"""
Goldilocks reference in pure Python (arbitrary-precision ints).

Used to generate test vectors that the CUDA kernels validate against.
This file is the source of truth for correctness; the CUDA code is the
performance implementation.

Goldilocks prime: P = 2^64 - 2^32 + 1.
  - p - 1 = 2^32 * (2^32 - 1), so the multiplicative group has 2-adicity 32.
  - Primitive root g = 7 (Plonky2's choice).

Run:
    python3 goldilocks_ref.py emit_field_vectors > field_vectors.txt
    python3 goldilocks_ref.py emit_ntt_vectors N=1024 > ntt_1024.txt
"""

import random
import struct
import sys

P = (1 << 64) - (1 << 32) + 1     # 0xFFFFFFFF00000001
G = 7                              # primitive root


def add(a, b):
    return (a + b) % P


def sub(a, b):
    return (a - b) % P


def mul(a, b):
    return (a * b) % P


def pow_(base, exp):
    return pow(base, exp, P)


def inv(a):
    return pow(a, -1, P)


def root_of_unity(n):
    """Primitive n-th root of unity. Requires n | (P-1)."""
    if (P - 1) % n != 0:
        raise ValueError(f"{n} does not divide P-1; can't build a primitive {n}-th root")
    return pow(G, (P - 1) // n, P)


def ntt(a, omega, invert=False):
    """Iterative Cooley-Tukey NTT, decimation-in-time, in-place semantics."""
    n = len(a)
    if n & (n - 1):
        raise ValueError(f"length must be a power of 2, got {n}")
    a = list(a)

    # Bit-reversal permutation.
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            a[i], a[j] = a[j], a[i]

    # Cooley-Tukey butterflies, level by level.
    length = 2
    while length <= n:
        w_len = pow(omega, n // length, P)
        if invert:
            w_len = inv(w_len)
        half = length // 2
        for i in range(0, n, length):
            w = 1
            for k in range(half):
                u = a[i + k]
                v = (a[i + k + half] * w) % P
                a[i + k]        = (u + v) % P
                a[i + k + half] = (u - v) % P
                w = (w * w_len) % P
        length <<= 1

    if invert:
        n_inv = inv(n)
        a = [(x * n_inv) % P for x in a]
    return a


def ntt_bailey(a, n1, n2):
    """
    Bailey 4-step NTT: factor N = n1 * n2.
      View a as M[j_a, j_b] = a[j_a*n2 + j_b], j_a in [0, n1), j_b in [0, n2).
      Step 1: column-NTT (size n1) for each j_b. Result B[k_a, j_b].
      Step 2: twiddle  C[k_a, j_b] = B[k_a, j_b] * w_N^{k_a * j_b}.
      Step 3: row-NTT (size n2) for each k_a, with TRANSPOSED output.
              Output A[k] where k = k_b * n1 + k_a (i.e., column-major in a flat
              array sized by (k_b, k_a)).

    Used here to verify the math before committing to the CUDA port.
    """
    assert n1 * n2 == len(a)
    n = n1 * n2
    w_n  = root_of_unity(n)
    w_n1 = root_of_unity(n1)
    w_n2 = root_of_unity(n2)

    # Step 1: column NTT of size n1 for each j_b.
    after1 = list(a)
    for jb in range(n2):
        col = [a[ja * n2 + jb] for ja in range(n1)]
        col_ntt = ntt(col, w_n1)
        for ka in range(n1):
            after1[ka * n2 + jb] = col_ntt[ka]

    # Step 2: twiddle.
    after2 = list(after1)
    for ka in range(n1):
        for jb in range(n2):
            after2[ka * n2 + jb] = (after1[ka * n2 + jb] * pow(w_n, ka * jb, P)) % P

    # Step 3: row NTT of size n2 for each k_a, transposed output.
    out = [0] * n
    for ka in range(n1):
        row = [after2[ka * n2 + jb] for jb in range(n2)]
        row_ntt = ntt(row, w_n2)
        for kb in range(n2):
            out[kb * n1 + ka] = row_ntt[kb]
    return out


def pmul_goldilocks(a, b):
    """Polynomial multiplication over Goldilocks via NTT.
    Returns coefficients of a*b (length len(a) + len(b) - 1)."""
    if not a or not b:
        return []
    deg = len(a) + len(b) - 1
    n = 1
    while n < deg:
        n *= 2
    omega = root_of_unity(n)
    A = list(a) + [0] * (n - len(a))
    B = list(b) + [0] * (n - len(b))
    A = ntt(A, omega)
    B = ntt(B, omega)
    C = [(A[i] * B[i]) % P for i in range(n)]
    C = ntt(C, omega, invert=True)
    return C[:deg]


def test_bailey():
    """Verify Bailey matches the direct NTT for a few sizes."""
    rng = random.Random(7)
    for n1, n2 in [(2, 2), (4, 4), (4, 8), (8, 4), (16, 16), (32, 32)]:
        n = n1 * n2
        a = [rng.randrange(P) for _ in range(n)]
        direct = ntt(a, root_of_unity(n))
        bailey = ntt_bailey(a, n1, n2)
        assert direct == bailey, f"Bailey mismatch at n1={n1}, n2={n2}"
    print("ntt_bailey matches direct NTT")


def emit_field_vectors():
    """
    Print a list of (a, b, a+b, a-b, a*b) tuples in hex, all canonical mod p.
    The CUDA test reads these and checks each operation matches.

    Format: one tuple per line, five hex u64 values space-separated.
    """
    rng = random.Random(42)
    cases = [
        (0, 0),
        (1, 0),
        (0, 1),
        (1, 1),
        (P - 1, 1),                    # add wrap to 0
        (P - 1, P - 1),                # add wrap to P-2
        (0, 1),                        # sub borrow to P-1
        (1, 2),                        # sub borrow
        (P - 1, P - 1),                # mul: (-1)*(-1) = 1
        (1 << 32, 1 << 32),            # mul: 2^64 mod p = 2^32 - 1
        (1 << 63, 1 << 63),            # mul: 2^126 mod p
    ]
    # plus 50 random pairs
    for _ in range(50):
        cases.append((rng.randrange(P), rng.randrange(P)))

    for a, b in cases:
        print(f"{a:016x} {b:016x} {add(a,b):016x} {sub(a,b):016x} {mul(a,b):016x}")


def emit_ntt_vectors(n):
    """
    Emit an NTT round-trip test vector for length n.
    Format:
      Line 1: n in decimal
      Line 2: n hex u64 values (input)
      Line 3: n hex u64 values (forward NTT output at root_of_unity(n))
    """
    rng = random.Random(123)
    a = [rng.randrange(P) for _ in range(n)]
    omega = root_of_unity(n)
    fwd = ntt(a, omega, invert=False)

    print(n)
    print(" ".join(f"{x:016x}" for x in a))
    print(" ".join(f"{x:016x}" for x in fwd))


def self_test():
    """Sanity tests on the Python reference itself."""
    rng = random.Random(0)

    # Field axioms
    assert add(P - 1, 1) == 0
    assert sub(0, 1) == P - 1
    assert mul(P - 1, P - 1) == 1
    # 2^64 mod p = 2^32 - 1
    assert (1 << 64) % P == (1 << 32) - 1

    # NTT round trip at several sizes
    for n in [2, 4, 8, 64, 1024]:
        a = [rng.randrange(P) for _ in range(n)]
        omega = root_of_unity(n)
        b = ntt(a, omega)
        c = ntt(b, omega, invert=True)
        assert a == c, f"round-trip failed at n={n}"

    # Bailey decomposition
    test_bailey()
    print("goldilocks_ref self-tests passed")


def main():
    if len(sys.argv) == 1:
        self_test()
        return
    cmd = sys.argv[1]
    if cmd == "emit_field_vectors":
        emit_field_vectors()
    elif cmd.startswith("emit_ntt_vectors"):
        n = None
        for arg in sys.argv[2:]:
            if arg.startswith("N="):
                n = int(arg.split("=", 1)[1])
        if n is None:
            sys.exit("emit_ntt_vectors requires N=<size>")
        emit_ntt_vectors(n)
    elif cmd == "self_test":
        self_test()
    else:
        sys.exit(f"unknown command: {cmd}")


if __name__ == "__main__":
    main()
