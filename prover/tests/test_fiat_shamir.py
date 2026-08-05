"""Sequential Fiat-Shamir transcript (S1 of the routed-projected plan).

What these tests are for: the prover used to expand all three verifier coins
from one base seed (`protocol.round_seeds`), which its own docstring flagged as
a TEST shortcut — a prover could know the opened columns before it committed
anything. Coins are now hashed from the transcript, and the Rust verifier
RECOMPUTES each one instead of reading it out of the proof.

Each test below therefore attacks the transcript, not the arithmetic:
  * an honest proof verifies and the recomputation agrees;
  * rewriting any coin in the proof file is caught;
  * rewriting the claim bytes (the statement) is caught;
  * a trusted statement digest supplied as policy must match;
  * no coin exists before the message it follows (structural check on the
    prover: the R1 sweep runs with ch0 still unset).
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
import claims as _C        # noqa: F401  (registers claim handlers)
import packets as _PK      # noqa: F401
import protocol as pr
from tape import Tape
from _rust_verify import _verify_proof_bin
from proof_dump import dump_proof

CFG = core.LigeroConfig(ELL=8, K_DEG=8, N_LIG=32, T_QUERIES=4)


def _build():
    core._COSET_POWERS_K_CACHE.clear()
    tape = Tape(CFG, lazy=True)
    u64 = lambda xs: torch.tensor(xs, dtype=torch.int64, device="cuda").to(torch.uint64)
    a = tape.commit("a", u64([41]), (1,))
    b = tape.commit("b", u64([1]), (1,))
    tape.reveal(tape.add(a, b), value=42)
    return tape


def _proof_file(tape=None):
    tape = tape or _build()
    proof = tape.prove()
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    dump_proof(path, pr.claims_to_json(tape.claims, CFG), None, proof, None, None)
    return path, proof


def _run(path, *policy):
    r = subprocess.run([_verify_proof_bin(), path, *policy],
                       capture_output=True, text=True)
    return ("rust_verify: ACCEPT" in r.stdout), (r.stdout + r.stderr).strip()


def _rewrite(path, mutate):
    doc = json.load(open(path))
    mutate(doc)
    with open(path, "w") as f:
        json.dump(doc, f)


def test_honest_proof_accepts_with_recomputed_coins():
    path, proof = _proof_file()
    try:
        acc, msg = _run(path)
        assert acc, f"honest FS proof: expected ACCEPT ({msg})"
        assert "seeds in file = recomputed transcript" in msg, msg
        assert "statement_digest = H(claim bytes)" in msg, msg
        # the coins really are transcript-derived, not seed-derived
        blocks = proof.blocks[:-1]
        roots = [getattr(proof, "root_%s" % b) for b in blocks]
        want = pr.fs_s_op(proof.statement_digest, blocks, roots)
        assert proof.seeds["s_op"] == want, "s_op is not H(statement || R1 roots)"
        print(f"    honest: ACCEPT, s_op={proof.seeds['s_op'].hex()[:12]}… "
              f"recomputed by the verifier")
    finally:
        os.unlink(path)


def test_rewritten_column_coin_rejects():
    """The attack the shortcut allowed: choose the opened columns."""
    path, proof = _proof_file()
    try:
        _rewrite(path, lambda d: d["seeds"].__setitem__("s_col", "00" * 32))
        acc, msg = _run(path)
        assert not acc, "rewritten s_col: expected REJECT"
        print("    rewritten s_col: REJECT ok")
    finally:
        os.unlink(path)


def test_rewritten_op_coin_rejects():
    path, _ = _proof_file()
    try:
        _rewrite(path, lambda d: d["seeds"].__setitem__("s_op", "11" * 32))
        acc, msg = _run(path)
        assert not acc, "rewritten s_op: expected REJECT"
        print("    rewritten s_op: REJECT ok")
    finally:
        os.unlink(path)


def test_rewritten_statement_rejects():
    """Changing the claim set must break its digest, hence the whole transcript."""
    path, _ = _proof_file()
    try:
        def bump(d):
            d["claims"]["table_order"] = d["claims"].get("table_order", []) + [7]
        _rewrite(path, bump)
        acc, msg = _run(path)
        assert not acc, "rewritten claims: expected REJECT"
        print("    rewritten claim set: REJECT ok")
    finally:
        os.unlink(path)


def test_policy_statement_digest_enforced():
    path, proof = _proof_file()
    try:
        # "-": this toy proof has no persistent weight block to police.
        acc, _ = _run(path, "-", proof.statement_digest.hex())
        assert acc, "matching policy digest: expected ACCEPT"
        acc, _ = _run(path, "-", "ab" * 32)
        assert not acc, "wrong policy digest: expected REJECT"
        print("    policy statement digest: matching ACCEPT / wrong REJECT")
    finally:
        os.unlink(path)


def test_no_coin_before_its_message():
    """Structural: the R1 sweep must run with the op challenges still unset.

    Wrapping _stream_sweep lets us observe what the prover knew at each round —
    a regression that pre-derives s_op would show ch0 already populated in the
    first (want_aux=False) sweep."""
    seen = []
    real = core._stream_sweep

    def spy(tape, cfg, master_seed_t, groups, n_ops, p1, p2, m_p1, tables, ch0, **kw):
        seen.append((kw.get("want_aux"), ch0 is None, kw.get("Q_cols") is None))
        return real(tape, cfg, master_seed_t, groups, n_ops, p1, p2, m_p1,
                    tables, ch0, **kw)

    core._stream_sweep = spy
    try:
        _build().prove()
    finally:
        core._stream_sweep = real
    assert seen[0] == (False, True, True), (
        f"R1 must run with no op challenge and no column set, got {seen[0]}")
    assert seen[1][1] is False, "R2 must have the op challenges"
    assert seen[-1][2] is False, "the opening round must have the columns"
    print(f"    round order ok: {seen}")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    fails = 0
    for t in tests:
        try:
            t(); print(f"[OK ] {t.__name__}")
        except Exception as e:
            fails += 1; print(f"[XX ] {t.__name__}: {type(e).__name__}: {e}")
    print(f"=== fiat-shamir: {len(tests)-fails}/{len(tests)} "
          f"{'PASS' if not fails else 'FAIL'} ===")
    return fails


if __name__ == "__main__":
    raise SystemExit(main())
