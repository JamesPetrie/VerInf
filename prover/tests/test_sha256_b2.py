"""B2 end-to-end: SHA256(key material) == H2, proven and Rust-verified —
the first full slice of the token binding (token-binding.md §12 P2).

The gadget commits the 40-byte key material (as byte-position strides) and
proves its SHA-256 digest equals the PUBLIC H2 from the reference recorder,
composed entirely from existing claims + LinCombClaim against the frozen
layout of ref/sha256_trace.py.

Positive: P0-vector key material + its recorded H2 -> ACCEPT.
Negatives (all want REJECT): a wrong public H2; a tampered committed message
byte; a bumped round carry; a flipped state bit; a tampered state word.

Run on the Spark:  ~/venv-hf/bin/python tests/test_sha256_b2.py
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

CFG = core.LigeroConfig(ELL=64, K_DEG=64, N_LIG=256, T_QUERIES=4)
SEED = b"sha256-b2"
VEC_PATH = pathlib.Path(__file__).parent / "vectors" / "token_binding_v0.json"


def _vec():
    with open(VEC_PATH) as f:
        return json.load(f)["vectors"][0]


def _build(km, digest, _tamper=None):
    core._COSET_POWERS_K_CACHE.clear()
    tape = Tape(CFG, lazy=True)
    byte_tbl = tape.register_table("tb_byte", list(range(256)))
    sha_tables = tb.register_sha_tables(tape)
    tb.sha256_h2_gadget(tape, sha_tables, byte_tbl, km, digest, _tamper=_tamper)
    return tape


def run_case(label, km, digest, want_accept, _tamper=None):
    tape = _build(km, digest, _tamper)
    acc, msg = rust_verify_tape(tape, tape.prove(seed=SEED), seed=SEED)
    ok = acc == want_accept
    print(f"[{'OK ' if ok else 'XX '}] {label}: "
          f"verify={'ACCEPT' if acc else 'REJECT'} "
          f"(want {'ACCEPT' if want_accept else 'REJECT'}) ({msg})", flush=True)
    return ok


def main():
    v = _vec()
    km = bytes.fromhex(v["key_material"])
    h2 = bytes.fromhex(v["H2"])
    wrong_h2 = bytearray(h2)
    wrong_h2[7] ^= 1

    results = [
        run_case("B2 positive (P0 vector)", km, h2, True),
        run_case("cheat: wrong public H2", km, bytes(wrong_h2), False),
        run_case("cheat: tampered message byte", km, h2, False,
                 _tamper=lambda t: t["msg_stride"][2].__setitem__(
                     5, t["msg_stride"][2][5] ^ 1)),
        run_case("cheat: bumped round carry", km, h2, False,
                 _tamper=lambda t: t["ce"].__setitem__(9, t["ce"][9] + 1)),
        run_case("cheat: flipped state bit", km, h2, False,
                 _tamper=lambda t: t["e_bit"][10].__setitem__(
                     3, 1 - t["e_bit"][10][3])),
        run_case("cheat: tampered state word", km, h2, False,
                 _tamper=lambda t: t["A"].__setitem__(30, t["A"][30] ^ 4)),
    ]
    fails = results.count(False)
    print(f"=== sha256 B2: {len(results) - fails}/{len(results)} "
          f"{'PASS' if not fails else 'FAIL'} ===")
    return fails


if __name__ == "__main__":
    raise SystemExit(main())
