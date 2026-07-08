"""P0 gates for the token-binding reference recorder (ref/token_recorder.py).

Run: python tests/run_tests.py test_token_recorder   (pure Python, no GPU)

Covers: the AES core against the FIPS-197 Appendix C vector; the pinned
GCM-counter-layout claim (our CTR output == AES-GCM ciphertext byte-for-byte)
against the `cryptography` library when available; the serialization, key-
material, and keystream-separation invariants; and vector determinism.
"""
import hashlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ref"))
import token_recorder as tr


def test_fips197_appendix_c_block():
    # FIPS-197 Appendix C.1: AES-128, key 000102...0f, pt 00112233...ff
    key = bytes(range(16))
    pt = bytes.fromhex("00112233445566778899aabbccddeeff")
    rks = tr._key_expand(key)
    ct = tr._encrypt_block(rks, pt)
    assert ct.hex() == "69c4e0d86a7b0430d8cdb78070b4c55a", ct.hex()


def test_gcm_ciphertext_equality():
    # The pinned compatibility claim: AES-CTR with the GCM counter layout
    # reproduces AES-GCM ciphertext exactly (tag excluded). Uses the
    # cryptography library as the independent reference when installed.
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError:
        print("    (cryptography not installed — GCM cross-check skipped)")
        return
    key = hashlib.sha256(b"tb-gcm-key").digest()[:16]
    iv = hashlib.sha256(b"tb-gcm-iv").digest()[:12]
    for n in (1, 15, 16, 17, 64, 100):
        pt = hashlib.sha256(b"tb-gcm-pt%d" % n).digest() * 4
        pt = pt[:n]
        ref = AESGCM(key).encrypt(iv, pt, None)[:n]   # strip the tag
        ours = tr.aes128_ctr_gcm(key, iv, pt)
        assert ours == ref, f"GCM mismatch at len {n}"


def test_ctr_roundtrip():
    key = bytes(range(16))
    iv = bytes(range(12))
    pt = bytes(range(256)) * 2
    ct = tr.aes128_ctr_gcm(key, iv, pt)
    assert tr.aes128_ctr_gcm(key, iv, ct) == pt      # CTR is an involution
    assert ct != pt


def test_serialization():
    assert tr.serialize_tokens([0]) == b"\x00\x00\x00\x00"
    assert tr.serialize_tokens([1, 258]) == b"\x01\x00\x00\x00\x02\x01\x00\x00"
    assert len(tr.serialize_tokens(list(range(7)))) == 28   # PLD_LEN = 4*T
    try:
        tr.serialize_tokens([1 << 32])
        assert False, "out-of-range token accepted"
    except ValueError:
        pass


def test_record_shape_and_h2_layout():
    key, iv_in, iv_out = bytes(16), bytes(11) + b"\x01", bytes(11) + b"\x02"
    rec = tr.record([1, 2, 3], [4, 5], key, iv_in, iv_out)
    km = bytes.fromhex(rec["key_material"])
    assert km == key + iv_in + iv_out and len(km) == 40
    assert rec["H2"] == hashlib.sha256(km).hexdigest()
    assert rec["H1_in"] == hashlib.sha256(bytes.fromhex(rec["ct_in"])).hexdigest()
    assert len(bytes.fromhex(rec["ct_in"])) == 12    # 3 tokens * 4 bytes


def test_keystream_separation():
    # Same tokens both directions must yield different ciphertexts (distinct
    # IVs), and identical IVs must be rejected outright.
    key = hashlib.sha256(b"k").digest()[:16]
    iv_a = hashlib.sha256(b"a").digest()[:12]
    iv_b = hashlib.sha256(b"b").digest()[:12]
    rec = tr.record([9, 9, 9], [9, 9, 9], key, iv_a, iv_b)
    assert rec["ct_in"] != rec["ct_out"]
    try:
        tr.record([1], [2], key, iv_a, iv_a)
        assert False, "iv reuse accepted"
    except ValueError:
        pass


def test_vectors_deterministic_and_consistent():
    v1, v2 = tr.make_vectors(), tr.make_vectors()
    assert v1 == v2
    for v in v1["vectors"]:
        rec = tr.record(v["tokens_in"], v["tokens_out"],
                        bytes.fromhex(v["key"]), bytes.fromhex(v["iv_in"]),
                        bytes.fromhex(v["iv_out"]))
        assert rec["H1_in"] == v["H1_in"] and rec["H2"] == v["H2"]


def test_vectors_match_checked_in_file():
    path = os.path.join(os.path.dirname(__file__), "vectors",
                        "token_binding_v0.json")
    if not os.path.exists(path):
        print("    (no checked-in vector file yet — regenerate gate skipped)")
        return
    import json
    with open(path) as f:
        on_disk = json.load(f)
    assert on_disk == tr.make_vectors(), \
        "checked-in vectors diverge from the recorder — spec drift"
