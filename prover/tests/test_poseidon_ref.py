"""Gates for the Poseidon reference (prover/ref/poseidon_gl.py).

This module is the spec, so these tests are about internal soundness of the
construction, not agreement with a third party (there is no published KAT for
this instantiation -- see the module docstring).
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "ref"))
import poseidon_gl as pg


def _msg(seed=b"k"):
    import hashlib
    out = b""
    i = 0
    while len(out) < pg.MSG_BYTES:
        out += hashlib.sha256(seed + bytes([i])).digest()
        i += 1
    return out[:pg.MSG_BYTES]


def test_sbox_is_a_bijection():
    """x^7 permutes the field iff gcd(7, p-1) = 1. Check the exponent condition
    directly, then spot-check invertibility."""
    import math
    assert math.gcd(pg.ALPHA, pg.P - 1) == 1, "x^ALPHA is not a permutation"
    d = pow(pg.ALPHA, -1, pg.P - 1)
    for v in (0, 1, 2, 12345, pg.P - 1, pg.P - 2):
        assert pow(pg.sbox(v), d, pg.P) == v % pg.P, "sbox not invertible at %d" % v
    print("    x^7 is a bijection on Goldilocks; inverse verified")


def test_mds_is_actually_mds():
    """Every square submatrix of a Cauchy matrix must be non-singular. Checking
    ALL of them is exponential, so check every 1x1..3x3 exhaustively plus the
    full matrix -- enough to catch a construction blunder (a repeated x_i or a
    zero denominator collapses small minors immediately)."""
    from itertools import combinations

    def det(m):
        n = len(m)
        m = [row[:] for row in m]
        d = 1
        for c in range(n):
            piv = next((r for r in range(c, n) if m[r][c] % pg.P), None)
            if piv is None:
                return 0
            if piv != c:
                m[c], m[piv] = m[piv], m[c]
                d = (-d) % pg.P
            d = d * m[c][c] % pg.P
            inv = pow(m[c][c], pg.P - 2, pg.P)
            for r in range(c + 1, n):
                f = m[r][c] * inv % pg.P
                for k in range(c, n):
                    m[r][k] = (m[r][k] - f * m[c][k]) % pg.P
        return d % pg.P

    n_checked = 0
    for k in (1, 2, 3):
        for rows in combinations(range(pg.T), k):
            for cols in combinations(range(pg.T), k):
                sub = [[pg.MDS[i][j] for j in cols] for i in rows]
                assert det(sub) != 0, "singular %dx%d submatrix %s %s" % (k, rows, cols, k)
                n_checked += 1
    assert det(pg.MDS) != 0, "full MDS matrix is singular"
    print("    Cauchy MDS: %d small submatrices + full matrix all non-singular" % n_checked)


def test_permutation_is_invertible_and_mixes():
    """A permutation must be injective; and one flipped input bit must move
    essentially every output element (no dead lanes)."""
    a = pg.permute([i for i in range(pg.T)])
    b = pg.permute([i for i in range(pg.T)])
    assert a == b, "permute is not deterministic"
    c = pg.permute([i for i in range(pg.T)][:-1] + [pg.T])
    assert a != c, "permute collapsed two distinct states"
    diff = sum(1 for i in range(pg.T) if a[i] != c[i])
    assert diff == pg.T, "only %d/%d output lanes changed -- poor diffusion" % (diff, pg.T)
    print("    deterministic, injective on the tested pair, all %d lanes diffuse" % pg.T)


def test_digest_shape_and_determinism():
    m = _msg()
    d1, d2 = pg.hash_bytes(m), pg.hash_bytes(m)
    assert d1 == d2, "hash not deterministic"
    assert len(d1) == 32, "digest is %d bytes, expected 32" % len(d1)
    assert pg.digest_elems(m) == [int.from_bytes(d1[8 * i:8 * i + 8], "little")
                                  for i in range(4)], "elem/byte encodings disagree"
    print("    32-byte digest, deterministic, elem<->byte encoding consistent")


def test_avalanche_every_input_bit():
    """Flipping ANY single bit of the 40-byte key material must change the
    digest. A bit that cannot affect the output would be a bit an attacker
    could vary freely while keeping KEY_COMMIT fixed."""
    m = _msg()
    base = pg.hash_bytes(m)
    for byte in range(pg.MSG_BYTES):
        for bit in range(8):
            mm = bytearray(m)
            mm[byte] ^= 1 << bit
            assert pg.hash_bytes(bytes(mm)) != base, \
                "bit %d of byte %d does not reach the digest" % (bit, byte)
    print("    all %d input bits reach the digest" % (pg.MSG_BYTES * 8))


def test_packing_is_injective():
    """5-byte little-endian packing must be lossless: two different key
    materials must never pack to the same field elements."""
    seen = {}
    for i in range(300):
        m = _msg(bytes([i % 256, i // 256]))
        k = tuple(pg.pack(m))
        assert k not in seen or seen[k] == m, "packing collision"
        seen[k] = m
    m = _msg()
    for e in pg.pack(m):
        assert 0 <= e < 1 << (8 * pg.BYTES_PER_ELEM), "packed element out of range"
        assert e < pg.P, "packed element exceeds the field"
    print("    packing injective over 300 samples; elements < 2^40 << p")


def test_trace_matches_permute_and_selfchecks():
    """The trace generator and the plain implementation must agree, and the
    independent constraint checker must accept the honest trace."""
    m = _msg(b"trace")
    t = pg.trace(m)
    assert t["digest"] == pg.digest_elems(m), "trace digest != permute digest"
    bad = pg.check_constraints(t)
    assert not bad, "honest trace failed its own checker: %s" % bad[:5]
    print("    trace == permute; constraint checker accepts the honest trace")


def test_checker_catches_every_tampered_class():
    """The checker must have no blind spots: bump one slot of each committed
    class and confirm it is caught. A class the checker ignores is a class the
    circuit could leave unconstrained."""
    m = _msg(b"tamper")
    for cls in ("x", "a", "s"):
        t = pg.trace(m)
        t[cls][3][5] = (t[cls][3][5] + 1) % pg.P
        assert pg.check_constraints(t), "tampered %s went undetected" % cls
    t = pg.trace(m)
    t["xR"][2] = (t["xR"][2] + 1) % pg.P
    assert pg.check_constraints(t), "tampered final state went undetected"
    print("    tampering x / a / s / xR each caught by the checker")


if __name__ == "__main__":
    fails = 0
    names = sorted(n for n in dir() if n.startswith("test_"))
    for n in names:
        try:
            globals()[n]()
            print("  PASS %s" % n)
        except Exception as e:
            print("  FAIL %s: %s: %s" % (n, type(e).__name__, e))
            fails += 1
    print("=== poseidon ref: %d/%d %s ===" % (len(names) - fails, len(names),
                                              "PASS" if not fails else "FAIL"))
    raise SystemExit(fails)
