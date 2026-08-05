"""Phase-3 witness block and the fifth transcript message (S1b).

The routed-projected matmul needs auxiliaries that may only be sampled AFTER
the phase-2 rows are committed (Q = M·P has two inputs that were not both
fixed in R1), so the prover grew a third commitment epoch:

    R1 -> s_op -> R2 -> s_bind -> R3 -> s_comb -> test polys -> s_col

These tests exercise the plumbing itself — layout, the p3 Merkle tree, the p3
openings, and the transcript — using an ordinary matmul whose Freivalds
auxiliaries are moved into phase 3. That is deliberately NOT the routed claim
(which arrives in S2): it isolates "does a third block commit, open and verify"
from "is the new relation sound".

The Rust verifier needs no claim-side change for this: a claim's JSON carries
row_start/length, never the epoch, so a row moving to a later block is just a
different position in the joint column — which is exactly why the fifth message
can be added without touching the compile.
"""
import json
import os
import pathlib
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch
import core
import claims as _C       # noqa: F401
import packets as _PK     # noqa: F401
import protocol as pr
from tape import Tape
from _rust_verify import _verify_proof_bin, rust_verify_tape
from proof_dump import dump_proof

CFG = core.LigeroConfig(ELL=8, K_DEG=8, N_LIG=32, T_QUERIES=4)


def _matmul_tape(late: bool):
    """A 2x2 · 2x2 matmul; `late` moves its Freivalds aux into phase 3."""
    core._COSET_POWERS_K_CACHE.clear()
    tape = Tape(CFG, lazy=True)
    u64 = lambda xs: torch.tensor(xs, dtype=torch.int64, device="cuda").to(torch.uint64)
    A = tape.commit("A", u64([1, 2, 3, 4]), (2, 2))
    B = tape.commit("B", u64([5, 6, 7, 8]), (2, 2))
    tape.matmul(A, B)
    if late:
        for c in tape.claims:
            for f in ("y", "u", "p"):
                v = getattr(c, f, None)
                if isinstance(v, core.Variable):
                    v.phase = 3
    return tape


def _dump(tape, proof):
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    dump_proof(path, pr.claims_to_json(tape.claims, CFG), None, proof, None, None)
    return path


def test_phase3_block_commits_opens_and_verifies():
    tape = _matmul_tape(late=True)
    proof = tape.prove()
    assert proof.blocks[-1] == "p3", f"p3 block missing: {proof.blocks}"
    assert proof.root_p3 is not None and proof.root_p3 != pr.EMPTY_COMMIT_ROOT
    assert proof.opened_p3 and all(t.numel() for t in proof.opened_p3.values()), \
        "phase-3 columns were not opened"
    acc, msg = rust_verify_tape(tape, proof, seed=None)
    assert acc, f"phase-3 proof: expected ACCEPT ({msg})"
    print(f"    p3 block: root={proof.root_p3.hex()[:12]}… "
          f"{len(proof.opened_p3)} columns opened, ACCEPT")


def test_phase3_transcript_is_five_messages():
    tape = _matmul_tape(late=True)
    proof = tape.prove()
    s_op = pr.fs_s_op(proof.statement_digest, proof.blocks[:-2],
                      [getattr(proof, "root_%s" % b) for b in proof.blocks[:-2]])
    s_bind = pr.fs_s_bind(s_op, proof.root_p2)
    s_comb = pr.fs_s_comb(s_bind, proof.root_p3)
    assert proof.seeds["s_op"] == s_op
    assert proof.seeds["s_bind"] == s_bind, "s_bind is not H(s_op || root_p2)"
    assert proof.seeds["s_comb"] == s_comb, "s_comb is not H(s_bind || root_p3)"
    print("    transcript: s_op -> s_bind(root_p2) -> s_comb(root_p3) -> s_col")


def test_empty_phase3_frames_the_round():
    """A tape with no late stage still hashes an R3 message, so a proof cannot
    drop the round to reuse an old s_comb."""
    tape = _matmul_tape(late=False)
    proof = tape.prove()
    assert "p3" not in proof.blocks and proof.root_p3 is None
    s_bind = pr.fs_s_bind(proof.seeds["s_op"], proof.root_p2)
    assert proof.seeds["s_comb"] == pr.fs_s_comb(s_bind, pr.EMPTY_COMMIT_ROOT)
    acc, msg = rust_verify_tape(tape, proof, seed=None)
    assert acc, f"no-p3 proof: expected ACCEPT ({msg})"
    print("    empty p3: framed with the empty-commit root, ACCEPT")


def test_tampered_phase3_column_rejects():
    tape = _matmul_tape(late=True)
    path = _dump(tape, tape.prove())
    try:
        def bump(d):
            cols = d["proof"]["opened_p3"]
            j = sorted(cols)[0]
            cols[j][0] = (cols[j][0] + 1) % ((1 << 64) - (1 << 32) + 1)
        doc = json.load(open(path))
        bump(doc)
        json.dump(doc, open(path, "w"))
        r = subprocess.run([_verify_proof_bin(), path], capture_output=True, text=True)
        assert "rust_verify: ACCEPT" not in r.stdout, "tampered p3 column: expected REJECT"
        print("    tampered p3 column: REJECT ok")
    finally:
        os.unlink(path)


def test_dropped_phase3_message_rejects():
    """Deleting the R3 message from the proof must not verify: s_comb (and so
    every later coin) is hashed over root_p3, and the joint column loses rows."""
    tape = _matmul_tape(late=True)
    path = _dump(tape, tape.prove())
    try:
        doc = json.load(open(path))
        doc["proof"]["blocks"] = [b for b in doc["proof"]["blocks"] if b != "p3"]
        for k in ("root_p3", "opened_p3", "paths_p3"):
            doc["proof"].pop(k, None)
        json.dump(doc, open(path, "w"))
        r = subprocess.run([_verify_proof_bin(), path], capture_output=True, text=True)
        assert "rust_verify: ACCEPT" not in r.stdout, "dropped R3: expected REJECT"
        print("    dropped R3 message: REJECT ok")
    finally:
        os.unlink(path)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    fails = 0
    for t in tests:
        try:
            t(); print(f"[OK ] {t.__name__}")
        except Exception as e:
            fails += 1; print(f"[XX ] {t.__name__}: {type(e).__name__}: {e}")
    print(f"=== phase3: {len(tests)-fails}/{len(tests)} "
          f"{'PASS' if not fails else 'FAIL'} ===")
    return fails


if __name__ == "__main__":
    raise SystemExit(main())
