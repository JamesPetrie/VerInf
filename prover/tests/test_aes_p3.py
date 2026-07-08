"""P3 end-to-end: AES-128-CTR proven and Rust-verified against the P0
recorder's ciphertext (token-binding.md §12 P3).

The gadget commits key/IV/plaintext bytes and proves the CTR ciphertext
(GCM counter layout) equals the recorder's, pinned public for this
standalone gate. Positive on a one-block stream and on the 40-byte
key-material shape (3 blocks, partial final). Cheats (want REJECT): a wrong
public ciphertext byte; a tampered key byte; a tampered plaintext byte; a
tampered MixColumns intermediate; a tampered IV byte.

Run on the Spark:  ~/venv-hf/bin/python tests/test_aes_p3.py
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch  # noqa: F401

import core
import claims as _C        # noqa: F401
import packets as _PK      # noqa: F401
from tape import Tape
from _rust_verify import rust_verify_tape

import token_binding as tb
from ref.token_recorder import serialize_tokens, aes128_ctr_gcm

CFG = core.LigeroConfig(ELL=64, K_DEG=64, N_LIG=256, T_QUERIES=4)
SEED = b"aes-p3"
VEC_PATH = pathlib.Path(__file__).parent / "vectors" / "token_binding_v0.json"


def _vec(name):
    with open(VEC_PATH) as f:
        return next(v for v in json.load(f)["vectors"] if v["name"] == name)


def _build(key, iv, pt, ct_public, _tamper=None):
    core._COSET_POWERS_K_CACHE.clear()
    tape = Tape(CFG, lazy=True)
    tables = tb.register_binding_tables(tape, with_xor=True)
    tb.aes_ctr_gadget(tape, tables, key, iv, pt, ct_public=ct_public,
                      _tamper=_tamper)
    return tape


def run_case(label, key, iv, pt, ct_public, want_accept, _tamper=None):
    tape = _build(key, iv, pt, ct_public, _tamper)
    acc, msg = rust_verify_tape(tape, tape.prove(seed=SEED), seed=SEED)
    ok = acc == want_accept
    print(f"[{'OK ' if ok else 'XX '}] {label}: "
          f"verify={'ACCEPT' if acc else 'REJECT'} "
          f"(want {'ACCEPT' if want_accept else 'REJECT'}) ({msg})", flush=True)
    return ok


def main():
    v = _vec("blocky")                       # 4 tokens = 16 bytes = 1 block
    key = bytes.fromhex(v["key"])
    iv = bytes.fromhex(v["iv_in"])
    pt = serialize_tokens(v["tokens_in"])
    ct = bytes.fromhex(v["ct_in"])
    assert aes128_ctr_gcm(key, iv, pt) == ct

    km = bytes.fromhex(v["key_material"])    # 40 bytes = 3 blocks (partial)
    iv2 = bytes.fromhex(v["iv_out"])
    ct_km = aes128_ctr_gcm(key, iv2, km)

    wrong_ct = bytearray(ct)
    wrong_ct[3] ^= 1

    results = [
        run_case("P3 positive, 1 block (P0 vector)", key, iv, pt, ct, True),
        run_case("P3 positive, 3 blocks partial final", key, iv2, km, ct_km, True),
        run_case("cheat: wrong public ct byte", key, iv, pt, bytes(wrong_ct), False),
        run_case("cheat: tampered key byte", key, iv, pt, ct, False,
                 _tamper=lambda t: t["rk"][0].__setitem__(3, t["rk"][0][3] ^ 1)),
        run_case("cheat: tampered plaintext byte", key, iv, pt, ct, False,
                 _tamper=lambda t: t["pt_bytes"][0].__setitem__(
                     5, t["pt_bytes"][0][5] ^ 1)),
        run_case("cheat: tampered MixColumns mid", key, iv, pt, ct, False,
                 _tamper=lambda t: t["mc_mid"][0][4][7].__setitem__(
                     1, t["mc_mid"][0][4][7][1] ^ 1)),
        run_case("cheat: tampered IV byte", key, iv, pt, ct, False,
                 _tamper=lambda t: t["iv_bytes"].__setitem__(
                     2, t["iv_bytes"][2] ^ 1)),
    ]
    fails = results.count(False)
    print(f"=== aes P3: {len(results) - fails}/{len(results)} "
          f"{'PASS' if not fails else 'FAIL'} ===")
    return fails


if __name__ == "__main__":
    raise SystemExit(main())
