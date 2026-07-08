"""P5 step-2 gate (analysis/persistent-weights.md): the ZK padding's
(seed, logical row offset) is decoupled from physical placement.

The padding PRG is keyed by (seed, row), but the column Merkle tree hashes
codeword VALUES in emission order — so a weight block padded under its
commitment's (seed, logical offset) reproduces the committed root wherever
it physically sits. Gates:

1. Displacement: the W block emitted at a DIFFERENT physical offset, padded
   under (seed B, logical NUM_BLINDING_ROWS), reproduces R_W'(B)
   bit-for-bit. Controls: physical-keyed padding, or the right seed at the
   wrong offset, do NOT.
2. Refresh reference: a proof referencing a commitment REFRESHED under
   seed B carries root_w == R_W'(B) and Rust-verifies ACCEPT (the
   re-extracted weight columns pad under the commitment's seed, so the
   persisted opening paths check out).
3. The commitment's seed survives save/load; pre-P5 pickles (no seed field)
   load with the default MASTER_SEED.

Run on the Spark:  ~/venv-hf/bin/python run_tests.py test_persistent_weights_p5
"""
import sys, pathlib, tempfile, os, pickle
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch
import core
from tape import Tape
from _rust_verify import rust_verify_tape

CFG = core.LigeroConfig(ELL=8192, K_DEG=16384, N_LIG=65536, T_QUERIES=4)
SEED_B = b"\x07" * 32          # the refresh seed (≠ core.MASTER_SEED)


def _t(vals):
    return torch.tensor(vals, dtype=torch.int64, device="cuda").to(torch.uint64)


def _build():
    """Same tape shape as the P3 gate: two distinct persistent weights and
    c = w1 + w2 (distinct operands — no repeated-var band-dedup collapse),
    so the AddClaim spans W↔p1 and the tape is linear in W (no quads on
    weight rows, as refresh referencing requires)."""
    tape = Tape(CFG, lazy=True)
    w1 = tape.commit("W1", _t(list(range(12000))), (12000,), persistent=True)
    w2 = tape.commit("W2", _t([v * 2 for v in range(12000)]), (12000,), persistent=True)
    tape.add(w1, w2)
    return tape


def test_displaced_block_reproduces_refreshed_root():
    tape = _build()
    art_b, weight_vars, m_w = core.commit_weights(tape, CFG, master_seed=SEED_B)
    assert m_w > 0

    master_t = core._master_seed_to_cuda(core.MASTER_SEED)
    seed_b_t = core._master_seed_to_cuda(SEED_B)
    X = core.NUM_BLINDING_ROWS + 1000     # a displaced physical placement

    def emit_at(phys, pad_seed=None, pad_row_offset=None):
        acc = core._make_merkle_acc(CFG.N_LIG, m_w)
        core._stream_phase(weight_vars, tape.inputs, CFG,
                           master_seed=master_t, abs_row_offset=phys,
                           pad_seed=pad_seed, pad_row_offset=pad_row_offset,
                           merkle_acc=acc)
        return core._finalize_merkle_artifact(acc).root

    # Physical X, padded as (seed B, logical NUM_BLINDING_ROWS) → R_W'(B).
    root = emit_at(X, pad_seed=seed_b_t, pad_row_offset=core.NUM_BLINDING_ROWS)
    assert root == art_b.root, "displaced block with decoupled pad must reproduce R_W'"
    print(f"    displaced emit at row {X} reproduces R_W' = {art_b.root[:8].hex()}…")

    # Controls: physical-keyed padding, and right-seed-wrong-offset.
    assert emit_at(X) != art_b.root, "physical-keyed padding must NOT reproduce R_W'"
    assert emit_at(X, pad_seed=seed_b_t, pad_row_offset=X) != art_b.root, \
        "seed B at the physical offset must NOT reproduce R_W'"
    print("    controls: physical-keyed pad and wrong-offset pad both differ")


def test_refreshed_commitment_reference_verifies():
    # Refresh: commit under seed B, then prove REFERENCING that commitment.
    wc_b = core.WeightCommitment.from_tape(_build(), CFG, master_seed=SEED_B)
    wc_a = core.WeightCommitment.from_tape(_build(), CFG)      # default seed
    assert wc_b.root != wc_a.root, "refresh must change the root"

    tape = _build()
    proof = tape.prove(seed=b"p5", weight_commitment=wc_b)
    assert proof.root_w == wc_b.root, "referenced proof's root_w != refreshed R_W'"
    acc, msg = rust_verify_tape(tape, proof, seed=b"p5")
    print(f"    refreshed-reference prove+verify: {'ACCEPT' if acc else 'REJECT'} "
          f"(R_W'={wc_b.root[:6].hex()}…, {msg.splitlines()[-1] if msg else ''})")
    assert acc, "proof referencing a refreshed commitment must Rust-verify ACCEPT"


def test_commitment_seed_roundtrip():
    wc = core.WeightCommitment.from_tape(_build(), CFG, master_seed=SEED_B)
    fd, path = tempfile.mkstemp(suffix=".wc"); os.close(fd)
    try:
        wc.save(path)
        loaded = core.WeightCommitment.load(path)
        assert loaded.master_seed == SEED_B, "save/load lost the commitment seed"
        assert loaded.root == wc.root
        # Pre-P5 pickle (no master_seed key) loads with the default seed.
        with open(path, "wb") as f:
            pickle.dump({"root": wc.root, "levels": wc.levels,
                         "m_w": wc.m_w, "n_lig": wc.n_lig}, f)
        legacy = core.WeightCommitment.load(path)
        assert legacy.master_seed == core.MASTER_SEED, "pre-P5 pickle must default to MASTER_SEED"
        print("    seed round-trips; pre-P5 pickle defaults to MASTER_SEED")
    finally:
        os.unlink(path)


if __name__ == "__main__":
    ok = True
    for fn in (test_displaced_block_reproduces_refreshed_root,
               test_refreshed_commitment_reference_verifies,
               test_commitment_seed_roundtrip):
        try:
            fn(); print(f"[OK ] {fn.__name__}")
        except Exception as e:
            ok = False; print(f"[XX ] {fn.__name__}: {e}")
    sys.exit(0 if ok else 1)
