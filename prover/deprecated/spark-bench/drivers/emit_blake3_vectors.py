"""
Emit BLAKE3 test vectors using the official Python `blake3` package
(which wraps the canonical Rust impl). Inputs follow the standard
BLAKE3 test pattern: bytes[i] = i mod 251.

Output: one "<length> <hex_digest>" line per vector. Lengths span
single-chunk (≤ 1024 bytes), single-chunk-boundary, several chunks,
and column-sized inputs we'll actually use in commit_weights.

Run:
    pip install blake3
    python3 emit_blake3_vectors.py | ./test_blake3_multi
"""

import sys

try:
    import blake3
except ImportError:
    print("error: pip install blake3", file=sys.stderr)
    sys.exit(1)


# Coverage: 0/1/64/1024 reproduce the single-chunk tests; 1025 onwards
# exercises the multi-chunk merge tree at 2, 3, 4, 8, 16, 64-chunk depths.
LENGTHS = [0, 1, 64, 1023, 1024, 1025, 2048, 4096, 8192, 16384, 65536]


def vector_input(n):
    return bytes(i % 251 for i in range(n))


def main():
    for n in LENGTHS:
        digest = blake3.blake3(vector_input(n)).hexdigest()
        print(f"{n} {digest}")


if __name__ == "__main__":
    main()
