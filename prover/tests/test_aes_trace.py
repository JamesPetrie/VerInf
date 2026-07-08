"""P3 gates for the AES-128-CTR constraint-system reference (ref/aes_trace.py).

Pure Python, no GPU: run with  python tests/run_tests.py test_aes_trace

Gates: the trace's ciphertext equals token_recorder.aes128_ctr_gcm's on the P0
vectors (both stream directions) and on random lengths; multi-block traces
advance the GCM counter 2, 3, 4, ...; the key schedule is committed once per
key regardless of block count; the constraint system is satisfied on every
honest trace; and an EXHAUSTIVE tamper fuzz — every committed slot of every
class, bumped one at a time — shows the layout has no unconstrained slots
(the completeness property the tape compile and Rust handler will inherit by
construction from this layout).
"""
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ref"))
import aes_trace as at
import token_recorder as tr


def _vectors():
    path = os.path.join(os.path.dirname(__file__), "vectors",
                        "token_binding_v0.json")
    with open(path) as f:
        return json.load(f)["vectors"]


def test_ciphertext_matches_recorder_on_p0_vectors():
    # Both directions of every P0 vector: the trace ciphertext must equal the
    # frozen ct_in/ct_out bytes (aes_ctr additionally asserts the constraint
    # system is satisfied and re-checks against the live oracle).
    for v in _vectors():
        key = bytes.fromhex(v["key"])
        for iv_hex, toks, ct_hex in ((v["iv_in"], v["tokens_in"], v["ct_in"]),
                                     (v["iv_out"], v["tokens_out"], v["ct_out"])):
            pt = tr.serialize_tokens(toks)
            t = at.aes_ctr(key, bytes.fromhex(iv_hex), pt)
            assert t["ciphertext"].hex() == ct_hex


def test_ciphertext_random_lengths():
    # Odd lengths around the block boundary, including partial final blocks.
    for n in (1, 4, 15, 16, 17, 32, 40, 100):
        seed = b"aes-trace-len%d" % n
        key = hashlib.sha256(seed + b"/k").digest()[:16]
        iv = hashlib.sha256(seed + b"/iv").digest()[:12]
        pt = (hashlib.sha256(seed + b"/pt").digest() * 4)[:n]
        t = at.aes_ctr(key, iv, pt)
        assert t["ciphertext"] == tr.aes128_ctr_gcm(key, iv, pt)


def test_multi_block_counters_advance():
    # 40 bytes = 3 blocks (16 + 16 + 8): the pinned counter suffixes must be
    # BE32(2), BE32(3), BE32(4) — the GCM layout token_recorder pins.
    key = bytes(range(16))
    iv = bytes(range(100, 112))
    pt = bytes(range(40))
    t = at.aes_ctr(key, iv, pt)
    assert t["nblocks"] == 3
    assert [bytes(c) for c in t["ctr_bytes"]] == \
        [(2 + j).to_bytes(4, "big") for j in range(3)]
    assert len(t["pt_bytes"][2]) == 8 and len(t["ct_bytes"][2]) == 8


def test_key_schedule_committed_once():
    # The schedule classes are per-KEY, not per-block: identical between a
    # 1-block and a 3-block trace, sized by rounds, and equal to the oracle's
    # expansion. Only the per-block classes scale with the block count.
    key = hashlib.sha256(b"ks-share/key").digest()[:16]
    iv = hashlib.sha256(b"ks-share/iv").digest()[:12]
    t1 = at.trace(key, iv, b"A" * 16)
    t3 = at.trace(key, iv, b"B" * 48)
    assert len(t3["rk"]) == 11 and t1["rk"] == t3["rk"]
    assert [bytes(r) for r in t3["rk"]] == tr._key_expand(key)
    for cls in ("ks_sub", "ks_rcon_b", "ks_rcon_k", "ks_rcon_z", "ks_xor_k"):
        assert t1[cls] == t3[cls]
        assert len(t3[cls]) == 10        # schedule rounds, not blocks
    assert len(t1["s_ark"]) == 1 and len(t3["s_ark"]) == 3
    assert len(t1["iv_bytes"]) == 12 and t1["iv_bytes"] == t3["iv_bytes"]


def _fuzz_all_slots(key, iv, pt):
    """Bump every committed slot by +1 (mutate, check, restore) and return
    the (class, path) pairs whose tamper is NOT caught."""
    t = at.trace(key, iv, pt)
    assert not at.check_constraints(t)
    missed = []
    for cls in at.COMMITTED:
        for cont, i, path in at._slots(t[cls]):
            cont[i] += 1
            if not at.check_constraints(t):
                missed.append((cls, path))
            cont[i] -= 1
    return missed


def test_constraints_reject_every_tamper_single_block():
    key = hashlib.sha256(b"fuzz1/key").digest()[:16]
    iv = hashlib.sha256(b"fuzz1/iv").digest()[:12]
    missed = _fuzz_all_slots(key, iv, b"\x00" * 16)   # exhaustive, ~2300 slots
    assert not missed, f"tampers NOT caught (unconstrained slots): {missed[:10]}"


def test_constraints_reject_every_tamper_multi_block_partial():
    # 2 blocks with a partial final block (28 bytes = 16 + 12): covers the
    # cross-block classes (shared iv, per-block counters) and the unused
    # keystream tail of the partial block.
    key = hashlib.sha256(b"fuzz2/key").digest()[:16]
    iv = hashlib.sha256(b"fuzz2/iv").digest()[:12]
    pt = (hashlib.sha256(b"fuzz2/pt").digest() * 2)[:28]
    missed = _fuzz_all_slots(key, iv, pt)
    assert not missed, f"tampers NOT caught (unconstrained slots): {missed[:10]}"


def test_sites_reproduce_checker():
    """The gadget-facing sites()/pool_layout() enumeration must flag exactly
    what check_constraints flags: agreement on honest traces and on random
    single-slot tampers, across shapes (the gadget composes from sites(), so
    this is the bridge between the checker's completeness and the circuit)."""
    import copy
    import random

    def eval_sites(t):
        idx, pool = at.pool_layout(t)
        S = at.sites(t)
        kv = at.key_values(t, S["xor"])
        deref = lambda ref: pool[idx[ref]]
        bad = []
        for s in S["sbox"]:
            if not (0 <= deref(s["x"]) < 256) or deref(s["y"]) != at.SBOX[deref(s["x"]) & 0xFF]:
                bad.append(("sbox", s))
        for s in S["xtime"]:
            if not (0 <= deref(s["x"]) < 256) or deref(s["y"]) != at.XTIME[deref(s["x"]) & 0xFF]:
                bad.append(("xtime", s))
        for k, s in zip(kv, S["xor"]):
            if k != 256 * deref(s["a"]) + deref(s["b"]):
                bad.append(("key", s))
            elif deref(s["z"]) != ((k >> 8) ^ (k & 0xFF)) % 256:
                bad.append(("xor", s))
        for p in S["public"]:
            if deref(p["ref"]) != p["value"]:
                bad.append(("public", p))
        bad += [("range", i) for i, v in enumerate(pool) if not (0 <= v < 256)]
        return bad

    key, iv = bytes(range(16)), bytes(range(12))
    rng = random.Random(1)
    for pt_len in (16, 40, 7):
        pt = bytes((i * 7 + 3) % 256 for i in range(pt_len))
        t = at.trace(key, iv, pt)
        assert not at.check_constraints(t) and not eval_sites(t)
        slots = [(cls, path) for cls in at.COMMITTED
                 for cont, i, path in at._slots(t[cls])]
        for cls, path in rng.sample(slots, min(300, len(slots))):
            t2 = copy.deepcopy(t)
            obj = t2[cls]
            for p in path[:-1]:
                obj = obj[p]
            obj[path[-1]] += 1
            assert at.check_constraints(t2) and eval_sites(t2), (cls, path)
