"""Reference recorder for the token-binding construction (paper Appendix E,
analysis/token-binding.md §9/§12 P0).

This file IS the byte-level spec the circuit must match: it maps
(tokens_in, tokens_out, key material) to the three public digests

    H1_in  = SHA256( AES128_CTR(key, iv_in,  serialize(tokens_in))  )
    H1_out = SHA256( AES128_CTR(key, iv_out, serialize(tokens_out)) )
    H2     = SHA256( key || iv_in || iv_out )

exactly as the recording side (e.g. the interlock frontend) computes them,
and it is the test-vector source for every implementation phase (P1-P5).

Pinned conventions (v0) — every constant here is part of the public claim
format; a mismatch anywhere silently breaks the binding (B1/B2):

  * Token serialization: each token id is 4 little-endian bytes
    (`TOKEN_BYTES = 4`); the payload length is always a multiple of 4, and
    a token unit never straddles a 16-byte AES block (4 | 16).
  * Cipher: AES-128 in counter mode with the AES-GCM counter layout, so
    hardware that encrypts with AES-GCM produces byte-identical ciphertext:
    with a 96-bit IV, GCM's J0 = IV || 0x00000001 and the first ciphertext
    counter block is inc32(J0) = IV || 0x00000002; block j uses
    IV || BE32(2 + j). PAYLOAD is ciphertext only — the GCM tag is never
    hashed (v6 interlock: nothing verifies tags; integrity comes from the
    certificate chain).
  * Key material: `key (16 bytes) || iv_in (12 bytes) || iv_out (12 bytes)`,
    40 bytes total, hashed as-is for H2. One key per request/response
    exchange (v6 spec decision 1), with DISTINCT per-direction IVs so the
    two streams never reuse keystream. H2 is fixed with the request, before
    the response exists.
  * Packetization: single-packet-per-stream stub. The multi-packet CTR
    counter-offset rule is deferred (token-binding.md §9.4, open) and must
    be pinned jointly with the recomputation design before multi-packet
    streams are recorded.

The AES here is pure Python from FIPS-197 (S-box generated algebraically,
not typed), validated by test_token_recorder.py against the FIPS-197
Appendix C vector and cross-checked against the `cryptography` library's
AES-GCM ciphertext when that library is available. SHA-256 is hashlib
(the reference by definition).
"""
import hashlib
import json
import sys

TOKEN_BYTES = 4          # little-endian bytes per token id
KEY_BYTES = 16           # AES-128
IV_BYTES = 12            # 96-bit IV, GCM-compatible
AES_BLOCK = 16
CTR_START = 2            # GCM: ciphertext counter blocks start at inc32(J0)


# ---------------------------------------------------------------- AES-128

def _gf_mul(a, b):
    """Multiply in GF(2^8) mod x^8 + x^4 + x^3 + x + 1 (0x11b)."""
    r = 0
    for _ in range(8):
        if b & 1:
            r ^= a
        hi = a & 0x80
        a = (a << 1) & 0xFF
        if hi:
            a ^= 0x1B
        b >>= 1
    return r


def _build_sbox():
    """S-box from the definition: multiplicative inverse then affine map —
    generated, not transcribed, so there is no 256-entry table to typo."""
    # inverses by brute force (256^2 once at import; fine for a reference)
    inv = [0] * 256
    for x in range(1, 256):
        for y in range(1, 256):
            if _gf_mul(x, y) == 1:
                inv[x] = y
                break
    sbox = []
    for x in range(256):
        b = inv[x]
        s = 0
        for i in range(8):
            bit = ((b >> i) ^ (b >> ((i + 4) % 8)) ^ (b >> ((i + 5) % 8)) ^
                   (b >> ((i + 6) % 8)) ^ (b >> ((i + 7) % 8)) ^ (0x63 >> i)) & 1
            s |= bit << i
        sbox.append(s)
    return sbox


SBOX = _build_sbox()
assert SBOX[0x00] == 0x63 and SBOX[0x53] == 0xED, "S-box generation broken"


def _key_expand(key):
    """AES-128 key schedule: 11 round keys of 16 bytes."""
    assert len(key) == KEY_BYTES
    w = [list(key[4 * i:4 * i + 4]) for i in range(4)]
    rcon = 1
    for i in range(4, 44):
        t = list(w[i - 1])
        if i % 4 == 0:
            t = t[1:] + t[:1]                      # RotWord
            t = [SBOX[b] for b in t]               # SubWord
            t[0] ^= rcon                           # Rcon
            rcon = _gf_mul(rcon, 2)
        w.append([a ^ b for a, b in zip(w[i - 4], t)])
    return [bytes(sum(w[4 * r:4 * r + 4], [])) for r in range(11)]


def _encrypt_block(rks, block):
    """One AES-128 block encryption. State is column-major per FIPS-197:
    state[r][c] = in[r + 4c]."""
    s = [[block[r + 4 * c] for c in range(4)] for r in range(4)]

    def add_rk(rk):
        for c in range(4):
            for r in range(4):
                s[r][c] ^= rk[r + 4 * c]

    add_rk(rks[0])
    for rnd in range(1, 11):
        for r in range(4):                          # SubBytes
            for c in range(4):
                s[r][c] = SBOX[s[r][c]]
        for r in range(1, 4):                       # ShiftRows
            s[r] = s[r][r:] + s[r][:r]
        if rnd < 10:                                # MixColumns
            for c in range(4):
                a = [s[r][c] for r in range(4)]
                s[0][c] = _gf_mul(a[0], 2) ^ _gf_mul(a[1], 3) ^ a[2] ^ a[3]
                s[1][c] = a[0] ^ _gf_mul(a[1], 2) ^ _gf_mul(a[2], 3) ^ a[3]
                s[2][c] = a[0] ^ a[1] ^ _gf_mul(a[2], 2) ^ _gf_mul(a[3], 3)
                s[3][c] = _gf_mul(a[0], 3) ^ a[1] ^ a[2] ^ _gf_mul(a[3], 2)
        add_rk(rks[rnd])
    return bytes(s[r % 4][r // 4] for r in range(16))


def aes128_ctr_gcm(key, iv, plaintext):
    """AES-128-CTR with the GCM counter layout: block j of the keystream is
    E_K( iv || BE32(CTR_START + j) ). Identical bytes to AES-GCM ciphertext
    for the same (key, iv, plaintext)."""
    assert len(key) == KEY_BYTES and len(iv) == IV_BYTES
    rks = _key_expand(key)
    out = bytearray()
    for j in range((len(plaintext) + AES_BLOCK - 1) // AES_BLOCK):
        counter = iv + (CTR_START + j).to_bytes(4, "big")
        ks = _encrypt_block(rks, counter)
        chunk = plaintext[AES_BLOCK * j: AES_BLOCK * (j + 1)]
        out.extend(p ^ k for p, k in zip(chunk, ks))
    return bytes(out)


# ---------------------------------------------------- serialization + record

def serialize_tokens(tokens):
    """Fixed-width serialization: 4 little-endian bytes per token id."""
    out = bytearray()
    for t in tokens:
        if not (0 <= t < 1 << (8 * TOKEN_BYTES)):
            raise ValueError(f"token id {t} out of range")
        out.extend(int(t).to_bytes(TOKEN_BYTES, "little"))
    return bytes(out)


def record(tokens_in, tokens_out, key, iv_in, iv_out):
    """The recorder: (tokens, key material) -> the three public digests,
    plus the intermediates the circuit difftests need."""
    if iv_in == iv_out:
        raise ValueError("iv_in must differ from iv_out (keystream reuse)")
    ct_in = aes128_ctr_gcm(key, iv_in, serialize_tokens(tokens_in))
    ct_out = aes128_ctr_gcm(key, iv_out, serialize_tokens(tokens_out))
    key_material = key + iv_in + iv_out
    return {
        "H1_in": hashlib.sha256(ct_in).hexdigest(),
        "H1_out": hashlib.sha256(ct_out).hexdigest(),
        "H2": hashlib.sha256(key_material).hexdigest(),
        "ct_in": ct_in.hex(),
        "ct_out": ct_out.hex(),
        "key_material": key_material.hex(),
    }


# --------------------------------------------------------------- vectors

def _deterministic_bytes(seed, n):
    """Deterministic expandable bytes for test vectors (no RNG state)."""
    out = bytearray()
    counter = 0
    while len(out) < n:
        out.extend(hashlib.sha256(seed + counter.to_bytes(4, "big")).digest())
        counter += 1
    return bytes(out[:n])


def make_vectors():
    """Deterministic test vectors for the circuit phases (P1-P5)."""
    vectors = []
    for name, n_in, n_out in [("tiny", 1, 1), ("blocky", 4, 4),
                              ("uneven", 3, 7), ("prompt50", 50, 13)]:
        seed = b"token-binding-v0/" + name.encode()
        key = _deterministic_bytes(seed + b"/key", KEY_BYTES)
        iv_in = _deterministic_bytes(seed + b"/iv_in", IV_BYTES)
        iv_out = _deterministic_bytes(seed + b"/iv_out", IV_BYTES)
        toks_in = [int.from_bytes(_deterministic_bytes(seed + b"/ti%d" % i, 3),
                                  "little") % 202048 for i in range(n_in)]
        toks_out = [int.from_bytes(_deterministic_bytes(seed + b"/to%d" % i, 3),
                                   "little") % 202048 for i in range(n_out)]
        rec = record(toks_in, toks_out, key, iv_in, iv_out)
        vectors.append({
            "name": name, "tokens_in": toks_in, "tokens_out": toks_out,
            "key": key.hex(), "iv_in": iv_in.hex(), "iv_out": iv_out.hex(),
            **rec,
        })
    return {"spec": "token-binding-v0", "token_bytes": TOKEN_BYTES,
            "ctr_start": CTR_START,
            "key_material_layout": "key16 || iv_in12 || iv_out12",
            "vectors": vectors}


if __name__ == "__main__":
    out = make_vectors()
    if len(sys.argv) > 1:
        with open(sys.argv[1], "w") as f:
            json.dump(out, f, indent=1)
        print(f"wrote {len(out['vectors'])} vectors to {sys.argv[1]}")
    else:
        print(json.dumps(out, indent=1))
