"""Byte-identical diff-test for the LDE skip: same tape, same pinned ZK seed,
proof dumped with the skip ON (current code) and OFF (forced), compared byte
for byte."""
import sys, pathlib, hashlib, tempfile, os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, "/home/riftuser/VerInf/prover")
import torch, core, claims as _C, packets as _PK, protocol as pr
from tape import Tape
from proof_dump import dump_proof

CFG = core.LigeroConfig(ELL=8, K_DEG=8, N_LIG=32, T_QUERIES=4)
ZK = b"pinned-zk-seed-for-difftest-0001"

def build():
    core._COSET_POWERS_K_CACHE.clear()
    t = Tape(CFG, lazy=True)
    u = lambda xs: torch.tensor(xs, dtype=torch.int64, device="cuda").to(torch.uint64)
    a = t.commit("a", u([41]), (1,)); b = t.commit("b", u([1]), (1,))
    t.reveal(t.add(a, b), value=42)
    return t

def dump(force_codewords):
    real = core.encode_messages
    if force_codewords:
        core.encode_messages = lambda *a, **k: real(*a, **{**k, "need_codewords": True})
    try:
        tape = build()
        p = tape.prove(zk_seed=ZK)
        fd, path = tempfile.mkstemp(suffix=".json"); os.close(fd)
        dump_proof(path, pr.claims_to_json(tape.claims, CFG), None, p, None, None)
        h = hashlib.sha256(open(path, "rb").read()).hexdigest()
        os.unlink(path)
        return h
    finally:
        core.encode_messages = real

off = dump(False)   # current code: LDE skipped where unread
on  = dump(True)    # forced: LDE always computed (the old behaviour)
print("skip-on ", off)
print("skip-off", on)
print("BYTE-IDENTICAL" if off == on else "DIFFERENT — the skip changed the proof")
raise SystemExit(0 if off == on else 1)
