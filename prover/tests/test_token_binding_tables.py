"""P1 gates for the token-binding tables + byte plumbing (token_binding.py).

Positive: the byte decomposition matches the reference recorder's
serialization byte-for-byte on the checked-in P0 vectors, and the
S-box / xtime / XOR8 paired lookups prove and Rust-verify ACCEPT.
Negative: an oversized token (>= 2^32) cannot satisfy the 4-byte
decomposition; a value outside a table's domain breaks its LogUp
settlement.

Run on the Spark:  ~/venv-hf/bin/python tests/test_token_binding_tables.py
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch

import core
import claims as _C        # noqa: F401  (registers COMPILE_FNS)
import packets as _PK      # noqa: F401
from tape import Tape
from _rust_verify import rust_verify_tape

import token_binding as tb
from ref.token_recorder import serialize_tokens

CFG = core.LigeroConfig(ELL=64, K_DEG=64, N_LIG=256, T_QUERIES=4)
SEED = b"token-binding-p1"
VEC_PATH = pathlib.Path(__file__).parent / "vectors" / "token_binding_v0.json"


def _vec(name):
    with open(VEC_PATH) as f:
        data = json.load(f)
    return next(v for v in data["vectors"] if v["name"] == name)


def _commit(tape, name, vals):
    t = torch.tensor(vals, dtype=torch.int64, device="cuda").to(torch.uint64)
    return tape.commit(name, t, (len(vals),))


def run_positive_token_bytes():
    """Byte decomposition == reference serialization, and Rust ACCEPTs."""
    core._COSET_POWERS_K_CACHE.clear()
    toks = _vec("blocky")["tokens_in"] + _vec("blocky")["tokens_out"]
    ser = serialize_tokens(toks)

    tape = Tape(CFG, lazy=True)
    tables = tb.register_binding_tables(tape, with_xor=False)
    x = _commit(tape, "toks", toks)
    bytes_le = tb.token_bytes(tape, x, tables)
    live = tape.run_engine_pass()
    got = [live[b.var].to(torch.int64).cpu().tolist() for b in bytes_le]
    exp = [[ser[4 * t + n] for t in range(len(toks))] for n in range(4)]
    values_ok = got == exp

    tape2 = Tape(CFG, lazy=True)
    tables2 = tb.register_binding_tables(tape2, with_xor=False)
    tb.token_bytes(tape2, _commit(tape2, "toks", toks), tables2)
    acc, msg = rust_verify_tape(tape2, tape2.prove(seed=SEED), seed=SEED)
    ok = acc and values_ok
    print(f"[{'OK ' if ok else 'XX '}] token_bytes: values_ok={values_ok} "
          f"verify={'ACCEPT' if acc else 'REJECT'} ({msg})")
    return ok


def run_positive_lookups():
    """S-box, xtime, and XOR8 lookups: correct values + Rust ACCEPT."""
    core._COSET_POWERS_K_CACHE.clear()
    xs = list(range(256))
    pairs = [(a, b) for a in (0x00, 0x53, 0xAA, 0xFF) for b in (0x01, 0x53, 0x7E)]
    keys = [a * 256 + b for a, b in pairs]

    tape = Tape(CFG, lazy=True)
    tables = tb.register_binding_tables(tape, with_xor=True)
    xb = _commit(tape, "xb", xs)
    kb = _commit(tape, "kb", keys)
    ys = tape.paired_tlookup(xb, tables["sbox"])
    yt = tape.paired_tlookup(xb, tables["xtime"])
    yx = tape.paired_tlookup(kb, tables["xor8"])
    live = tape.run_engine_pass()
    ok_vals = (live[ys.var].to(torch.int64).cpu().tolist() == list(tb.SBOX)
               and live[yt.var].to(torch.int64).cpu().tolist() == tb.XTIME
               and live[yx.var].to(torch.int64).cpu().tolist() == [a ^ b for a, b in pairs])

    tape2 = Tape(CFG, lazy=True)
    tables2 = tb.register_binding_tables(tape2, with_xor=True)
    tape2.paired_tlookup(_commit(tape2, "xb", xs), tables2["sbox"])
    tape2.paired_tlookup(_commit(tape2, "kb", keys), tables2["xor8"])
    acc, msg = rust_verify_tape(tape2, tape2.prove(seed=SEED), seed=SEED)
    ok = acc and ok_vals
    print(f"[{'OK ' if ok else 'XX '}] lookups: values_ok={ok_vals} "
          f"verify={'ACCEPT' if acc else 'REJECT'} ({msg})")
    return ok


def run_cheat_oversized_token():
    """token >= 2^32 has no valid 4-byte decomposition -> REJECT."""
    core._COSET_POWERS_K_CACHE.clear()
    tape = Tape(CFG, lazy=True)
    tables = tb.register_binding_tables(tape, with_xor=False)
    tb.token_bytes(tape, _commit(tape, "toks", [(1 << 32) + 5, 7]), tables)
    acc, msg = rust_verify_tape(tape, tape.prove(seed=SEED), seed=SEED)
    ok = not acc
    print(f"[{'OK ' if ok else 'XX '}] cheat oversized token: "
          f"verify={'ACCEPT' if acc else 'REJECT'} (want REJECT) ({msg})")
    return ok


def run_cheat_off_table_range():
    """A committed 'byte' of 300 is outside [0,256): the range LogUp cannot
    settle -> REJECT."""
    core._COSET_POWERS_K_CACHE.clear()
    tape = Tape(CFG, lazy=True)
    tables = tb.register_binding_tables(tape, with_xor=False)
    bad = _commit(tape, "bad", [3, 300, 7])
    tape.range_word(bad, tables["byte"])
    acc, msg = rust_verify_tape(tape, tape.prove(seed=SEED), seed=SEED)
    ok = not acc
    print(f"[{'OK ' if ok else 'XX '}] cheat off-table byte: "
          f"verify={'ACCEPT' if acc else 'REJECT'} (want REJECT) ({msg})")
    return ok


def run_cheat_off_table_paired():
    """A paired-lookup key outside the XOR8 domain -> REJECT."""
    core._COSET_POWERS_K_CACHE.clear()
    tape = Tape(CFG, lazy=True)
    tables = tb.register_binding_tables(tape, with_xor=True)
    kb = _commit(tape, "kb", [tb.XOR8_LEN + 3])
    tape.paired_tlookup(kb, tables["xor8"])
    acc, msg = rust_verify_tape(tape, tape.prove(seed=SEED), seed=SEED)
    ok = not acc
    print(f"[{'OK ' if ok else 'XX '}] cheat off-table paired key: "
          f"verify={'ACCEPT' if acc else 'REJECT'} (want REJECT) ({msg})")
    return ok


def main():
    tests = [run_positive_token_bytes, run_positive_lookups,
             run_cheat_oversized_token, run_cheat_off_table_range,
             run_cheat_off_table_paired]
    fails = 0
    for t in tests:
        try:
            if not t():
                fails += 1
        except Exception as e:
            fails += 1
            print(f"[XX ] {t.__name__}: {type(e).__name__}: {e}")
    n = len(tests)
    print(f"=== token-binding P1: {n - fails}/{n} {'PASS' if not fails else 'FAIL'} ===")
    return fails


if __name__ == "__main__":
    raise SystemExit(main())
