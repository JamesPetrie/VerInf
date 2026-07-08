"""P3 gate (analysis/persistent-weights.md): a proof can REFERENCE a
pre-committed W tree instead of rebuilding it. commit the weights once → save
→ load → prove with the commitment → the proof carries the loaded R_W and the
Rust verifier ACCEPTs. Because R_W is context-independent (layout B), the
re-extracted weight columns reproduce the persisted leaves, so the opening
paths from the stored tree check out.

Run on the Spark:  ~/venv-hf/bin/python run_tests.py test_persistent_weights_p3
"""
import sys, pathlib, tempfile, os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch
import core
from tape import Tape
from _rust_verify import rust_verify_tape

CFG = core.LigeroConfig(ELL=8192, K_DEG=16384, N_LIG=65536, T_QUERIES=4)


def _t(vals):
    return torch.tensor(vals, dtype=torch.int64, device="cuda").to(torch.uint64)


def _build():
    """A small verifying tape with persistent weights referenced by a claim.
    Two DISTINCT weights added (c = w1 + w2) — distinct operands so the linear
    fold has no repeated-variable coefficient collapse; both land in the W
    block, c in the p1 (activations) block, so the AddClaim spans W↔p1."""
    tape = Tape(CFG, lazy=True)
    w1 = tape.commit("W1", _t(list(range(12000))), (12000,), persistent=True)
    w2 = tape.commit("W2", _t([v * 2 for v in range(12000)]), (12000,), persistent=True)
    tape.add(w1, w2)                           # c = w1 + w2 → W block + AddClaim
    return tape


def test_reference_matches_and_verifies():
    tape = _build()
    wc = core.WeightCommitment.from_tape(tape, CFG)
    fd, path = tempfile.mkstemp(suffix=".wc"); os.close(fd)
    try:
        wc.save(path)
        loaded = core.WeightCommitment.load(path)
        assert loaded.root == wc.root and loaded.m_w == wc.m_w, "save/load round-trip changed R_W"

        # Prove REFERENCING the loaded commitment.
        tape2 = _build()
        proof = tape2.prove(seed=b"p3", weight_commitment=loaded)
        assert proof.root_w == loaded.root, "referenced proof's root_w != committed R_W"
        acc, msg = rust_verify_tape(tape2, proof, seed=b"p3")
        print(f"    referenced prove+verify: {'ACCEPT' if acc else 'REJECT'} "
              f"(m_w={loaded.m_w}, {msg.splitlines()[-1] if msg else ''})")
        assert acc, "referenced-W proof must Rust-verify ACCEPT"
    finally:
        os.unlink(path)


def test_baseline_and_reference_agree():
    """The referenced proof and a rebuilt-W proof commit the SAME R_W and both
    verify — referencing is transparent to the verdict."""
    tape_a = _build()
    p_rebuild = tape_a.prove(seed=b"p3b")                 # no commitment → rebuild W
    tape_b = _build()
    wc = core.WeightCommitment.from_tape(tape_b, CFG)
    tape_c = _build()
    p_ref = tape_c.prove(seed=b"p3b", weight_commitment=wc)
    assert p_rebuild.root_w == p_ref.root_w == wc.root, "rebuild vs reference R_W disagree"
    a1, _ = rust_verify_tape(tape_a, p_rebuild, seed=b"p3b")
    a2, _ = rust_verify_tape(tape_c, p_ref, seed=b"p3b")
    print(f"    rebuild ACCEPT={a1}  reference ACCEPT={a2}  (same R_W={p_rebuild.root_w[:6].hex()}…)")
    assert a1 and a2, "both rebuild-W and reference-W proofs must ACCEPT"


def test_mismatched_commitment_rejected():
    """A commitment for DIFFERENT weights (different m_w) is refused by the guard."""
    tape = _build()
    other = Tape(CFG, lazy=True)                                # different W size → different m_w
    w1 = other.commit("W1", _t(list(range(5000))), (5000,), persistent=True)
    w2 = other.commit("W2", _t(list(range(5000))), (5000,), persistent=True)
    other.add(w1, w2)
    wc_other = core.WeightCommitment.from_tape(other, CFG)
    try:
        tape.prove(seed=b"p3", weight_commitment=wc_other)
        raise AssertionError("prove accepted a mismatched weight commitment")
    except AssertionError as e:
        if "mismatch" not in str(e):
            raise
        print("    mismatched commitment: refused by guard")


if __name__ == "__main__":
    ok = True
    for fn in (test_reference_matches_and_verifies,
               test_baseline_and_reference_agree,
               test_mismatched_commitment_rejected):
        try:
            fn(); print(f"[OK ] {fn.__name__}")
        except Exception as e:
            ok = False; print(f"[XX ] {fn.__name__}: {e}")
    sys.exit(0 if ok else 1)
