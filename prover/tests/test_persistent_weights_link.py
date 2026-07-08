"""P5 steps 3+4 gate (analysis/persistent-weights.md): the linking proof.

When the per-commitment column budget nears exhaustion, the prover refreshes:
re-commits the same weights under a fresh seed → R_W′, and produces ONE proof
that links R_W′ to the trusted R_W — two weight blocks in one tape (W_old →
"w" tree reproducing R_W, W_new → "wnew" tree reproducing R_W′) plus the
LinCombClaim equality W_old[i] − W_new[i] = 0. The verifier adopts R_W′ after
(a) the proof ACCEPTs, (b) root_w == trusted R_W, (c) root_wnew == claimed
R_W′.

Gates:
1. Honest link: ACCEPT with root_w == R_W and root_wnew == R_W′.
2. NEGATIVE (the load-bearing soundness check): linking R_W′ committed over
   DIFFERENT weights → REJECT (the equality fails; RS distance / lin_sum
   catches it) — both a fully-different block and a single-element tamper.
3. Wrong refresh seed → root_wnew ≠ R_W′ (the adoption comparison fails even
   though the proof itself is internally consistent).

Run on the Spark:  ~/venv-hf/bin/python run_tests.py test_persistent_weights_link
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch
import core
from tape import Tape
from _rust_verify import rust_verify_tape

CFG = core.LigeroConfig(ELL=8192, K_DEG=16384, N_LIG=65536, T_QUERIES=4)
SEED_B = b"\x07" * 32          # the refresh seed (≠ core.MASTER_SEED)

W1 = list(range(12000))
W2 = [v * 2 for v in range(12000)]


def _t(vals):
    return torch.tensor(vals, dtype=torch.int64, device="cuda").to(torch.uint64)


def _weights_tape(w1_vals, w2_vals):
    """The canonical weight tape (the shape commit_weights sees in a normal
    proof): two persistent weights referenced by an activation-producing
    claim. Its W block order is [W1, W2]."""
    tape = Tape(CFG, lazy=True)
    w1 = tape.commit("W1", _t(w1_vals), (len(w1_vals),), persistent=True)
    w2 = tape.commit("W2", _t(w2_vals), (len(w2_vals),), persistent=True)
    tape.add(w1, w2)
    return tape


def _linking_tape(old1, old2, new1, new2):
    """The linking proof's tape: W_old in the "w" block (order [W1, W2],
    matching _weights_tape), W_new in the "wnew" block (same order), the
    per-pair LinComb equalities, and one activation claim so p1 is
    non-empty. Linear in both weight blocks (no quads on weight rows), as
    the decoupled padding requires."""
    tape = Tape(CFG, lazy=True)
    w1o = tape.commit("W1", _t(old1), (len(old1),), persistent=True)
    w2o = tape.commit("W2", _t(old2), (len(old2),), persistent=True)
    w1n = tape.commit("W1n", _t(new1), (len(new1),), persistent="new")
    w2n = tape.commit("W2n", _t(new2), (len(new2),), persistent="new")
    tape.add(w1o, w2o)                       # claim 0: places the W block
    tape.lincomb([w1o, w1n], [1, -1], 0)     # claim 1: W1_old == W1_new
    tape.lincomb([w2o, w2n], [1, -1], 0)     # claim 2: W2_old == W2_new
    return tape


def test_honest_link_accepts_and_binds_both_roots():
    wc_a = core.WeightCommitment.from_tape(_weights_tape(W1, W2), CFG)   # trusted R_W
    wc_b = core.WeightCommitment.from_tape(_weights_tape(W1, W2), CFG,
                                           master_seed=SEED_B)           # refreshed R_W'
    tape = _linking_tape(W1, W2, W1, W2)
    proof = tape.prove(seed=b"link", wnew_seed=SEED_B)
    assert proof.root_w == wc_a.root, "linking proof's root_w != trusted R_W"
    assert proof.root_wnew == wc_b.root, "linking proof's root_wnew != refreshed R_W'"
    acc, msg = rust_verify_tape(tape, proof, seed=b"link")
    print(f"    honest link: {'ACCEPT' if acc else 'REJECT'} "
          f"(R_W={wc_a.root[:6].hex()}… ↔ R_W'={wc_b.root[:6].hex()}…, "
          f"{msg.splitlines()[-1] if msg else ''})")
    assert acc, "honest linking proof must Rust-verify ACCEPT"


def test_link_to_different_weights_rejected():
    """The attack the linking proof must kill: adopt an R_W' that commits
    DIFFERENT weights. The prover commits W' under the refresh seed (so
    root_wnew really is that R_W'), links it to the true W — the equality is
    false → REJECT."""
    W1p = [v + 1 for v in W1]                                # different weights
    wc_bp = core.WeightCommitment.from_tape(_weights_tape(W1p, W2), CFG,
                                            master_seed=SEED_B)
    tape = _linking_tape(W1, W2, W1p, W2)
    proof = tape.prove(seed=b"link", wnew_seed=SEED_B)
    assert proof.root_wnew == wc_bp.root, \
        "proof must genuinely bind the different-weights R_W' (else the test is vacuous)"
    acc, _ = rust_verify_tape(tape, proof, seed=b"link")
    print(f"    different-weights link: {'ACCEPT' if acc else 'REJECT'} (want REJECT)")
    assert not acc, "linking R_W' to different weights MUST be rejected"


def test_link_single_element_tamper_rejected():
    """Minimal-distance variant: W' differs from W in ONE element. lin_sum's
    random fold catches any false equality w.p. 1 - 1/P, independent of how
    many columns are opened."""
    W1p = list(W1); W1p[7] += 1
    tape = _linking_tape(W1, W2, W1p, W2)
    proof = tape.prove(seed=b"link", wnew_seed=SEED_B)
    acc, _ = rust_verify_tape(tape, proof, seed=b"link")
    print(f"    one-element tamper: {'ACCEPT' if acc else 'REJECT'} (want REJECT)")
    assert not acc, "a single differing weight element MUST be rejected"


def test_wrong_refresh_seed_fails_root_adoption():
    """Proving with a seed other than the refresh's: the proof is internally
    consistent (ACCEPT) but root_wnew ≠ R_W', so the verifier's adoption
    comparison — part (c) of the link check — fails."""
    wc_b = core.WeightCommitment.from_tape(_weights_tape(W1, W2), CFG,
                                           master_seed=SEED_B)
    tape = _linking_tape(W1, W2, W1, W2)
    proof = tape.prove(seed=b"link", wnew_seed=b"\x09" * 32)
    assert proof.root_wnew != wc_b.root, \
        "a different pad seed must yield a different wnew root"
    acc, _ = rust_verify_tape(tape, proof, seed=b"link")
    print(f"    wrong-seed link: proof {'ACCEPT' if acc else 'REJECT'}, "
          f"root_wnew != R_W' → adoption refused")
    assert acc, "the wrong-seed proof itself is consistent — the ROOT comparison is the gate"


if __name__ == "__main__":
    ok = True
    for fn in (test_honest_link_accepts_and_binds_both_roots,
               test_link_to_different_weights_rejected,
               test_link_single_element_tamper_rejected,
               test_wrong_refresh_seed_fails_root_adoption):
        try:
            fn(); print(f"[OK ] {fn.__name__}")
        except Exception as e:
            ok = False; print(f"[XX ] {fn.__name__}: {e}")
    sys.exit(0 if ok else 1)
