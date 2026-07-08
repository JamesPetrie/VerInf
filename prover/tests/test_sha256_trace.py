"""P2 gates for the SHA-256 constraint-system reference (ref/sha256_trace.py).

Pure Python, no GPU: run with  python tests/run_tests.py test_sha256_trace

Gates: the trace's digest equals hashlib on the P0 key materials and on random
inputs; the constraint system is satisfied on every honest trace; the PUBLIC
digest pin rejects a wrong H2; and a tamper fuzz shows the layout has no
unconstrained slots — bumping ANY committed value class violates at least one
constraint (the completeness property the tape gadget inherits from this
layout).
"""
import copy
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ref"))
import sha256_trace as st
import token_recorder as tr


def _vec_key_materials():
    path = os.path.join(os.path.dirname(__file__), "vectors",
                        "token_binding_v0.json")
    with open(path) as f:
        data = json.load(f)
    return [bytes.fromhex(v["key_material"]) for v in data["vectors"]]


def test_digest_matches_hashlib_on_p0_vectors():
    for km in _vec_key_materials():
        assert len(km) == 40
        t = st.sha256_one_block(km)          # asserts digest + constraints
        assert t["digest"] == hashlib.sha256(km).digest()


def test_h2_equals_recorder():
    # The full B2 statement at reference level: the trace digest of the key
    # material IS the recorder's H2, and the constraint system accepts it as
    # the public pin.
    key = bytes(range(16))
    iv_in = bytes(range(12))
    iv_out = bytes(range(1, 13))
    rec = tr.record([1, 2], [3], key, iv_in, iv_out)
    t = st.sha256_one_block(key + iv_in + iv_out)
    assert t["digest"].hex() == rec["H2"]
    assert not st.check_constraints(t, digest=bytes.fromhex(rec["H2"]))


def test_wrong_public_digest_rejected():
    km = _vec_key_materials()[0]
    t = st.trace(km)
    wrong = bytearray(t["digest"])
    wrong[5] ^= 1
    bad = st.check_constraints(t, digest=bytes(wrong))
    assert bad and all(n.startswith("digest") for n, _, _ in bad), bad


def test_digest_various_lengths():
    for n in (0, 4, 8, 32, 40, 52):          # multiples of 4, single block
        msg = (hashlib.sha256(b"len%d" % n).digest() * 2)[:n]
        st.sha256_one_block(msg)


def test_constraints_reject_every_tamper():
    km = _vec_key_materials()[0]
    t = st.trace(km)
    assert not st.check_constraints(t)

    def tampered(mutate):
        t2 = copy.deepcopy(t)
        mutate(t2)
        return st.check_constraints(t2)

    flip = lambda cls, r, j: (lambda t2: t2[cls][r].__setitem__(j, 1 - t2[cls][r][j]))
    cases = {
        "W word":        lambda t2: t2["W"].__setitem__(20, t2["W"][20] ^ 1),
        "W pad word":    lambda t2: t2["W"].__setitem__(12, t2["W"][12] ^ 1),
        "A state":       lambda t2: t2["A"].__setitem__(30, t2["A"][30] ^ 4),
        "E state":       lambda t2: t2["E"].__setitem__(50, t2["E"][50] ^ 1),
        "A prefix":      lambda t2: t2["A"].__setitem__(0, t2["A"][0] ^ 1),
        "msg stride byte": lambda t2: t2["msg_stride"][2].__setitem__(5, t2["msg_stride"][2][5] ^ 1),
        "msg byte range":  lambda t2: t2["msg_stride"][1].__setitem__(3, t2["msg_stride"][1][3] + 256),
        "a bit":  flip("a_bit", 10, 3),   "b bit":  flip("b_bit", 10, 3),
        "c bit":  flip("c_bit", 10, 3),   "e bit":  flip("e_bit", 10, 3),
        "f bit":  flip("f_bit", 10, 3),   "g bit":  flip("g_bit", 10, 3),
        "w_m bit": flip("w_bit_m", 10, 3), "w_n bit": flip("w_bit_n", 10, 3),
        "w_n range bit (word 62)": flip("w_bit_n", 62 - st.N_LO, 0),
        "non-bool bit":  lambda t2: t2["e_bit"][7].__setitem__(0, 2),
        "x12 xor":  flip("x12", 12, 5),   "y12 xor":  flip("y12", 12, 5),
        "tbc xor":  flip("tbc", 12, 5),   "xm xor":   flip("xm", 12, 5),
        "xn xor":   flip("xn", 12, 5),
        "p1 prod":  flip("p1", 12, 5),    "p2 prod":  flip("p2", 12, 5),
        "q1 prod":  flip("q1", 12, 5),    "q2 prod":  flip("q2", 12, 5),
        "ef prod":  flip("ef", 12, 5),    "eg prod":  flip("eg", 12, 5),
        "bc prod":  flip("bc", 12, 5),    "u prod":   flip("u", 12, 5),
        "m1 prod":  flip("m1", 12, 5),    "m2 prod":  flip("m2", 12, 5),
        "n1 prod":  flip("n1", 12, 5),    "n2 prod":  flip("n2", 12, 5),
        "e carry":  lambda t2: t2["ce"].__setitem__(9, t2["ce"][9] + 1),
        "a carry":  lambda t2: t2["ca"].__setitem__(9, t2["ca"][9] + 1),
        "w carry":  lambda t2: t2["cw"].__setitem__(9, t2["cw"][9] + 1),
        "carry range escape": lambda t2: (
            t2["ce"].__setitem__(9, t2["ce"][9] + 8),
            t2["E"].__setitem__(4 + 9, (t2["E"][4 + 9] - 8 * st.MOD32) % (1 << 64))),
        "digest carry":  lambda t2: t2["cd"].__setitem__(3, 1 - t2["cd"][3]),
    }
    fails = [name for name, mutate in cases.items() if not tampered(mutate)]
    assert not fails, f"tampers NOT caught (unconstrained slots): {fails}"
