"""Verify a proof with the independent Rust verifier (verify_proof).

The unit + negative tests used to verify via core.verify — the Python
co-simulation, which shares the prover's own COMPILE_FNS/EXPANDERS and so is a
circular check. This helper instead dumps the proof to the verify_proof JSON
format and shells out to the Rust binary (the real TCB), returning
(accepted, output) so the existing `acc, msg = verify(...)` sites keep working.

Both test provers (tests/test_prover.prove and the streaming tape.prove) return
the same Proof object, so one helper covers every test. The dump format mirrors
tests/dump_routing_proof.py and the demos' --dump-proof.
"""
import json
import os
import pathlib
import subprocess
import tempfile

import protocol as pr

_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _verify_proof_bin():
    env = os.environ.get("LIGERO_VERIFY_PROOF")
    if env:
        return env
    for profile in ("release", "debug"):
        p = _ROOT / "verifier" / "target" / profile / "verify_proof"
        if p.exists():
            return str(p)
    raise RuntimeError(
        "verify_proof binary not found — build it first:\n"
        "  (cd verifier && cargo build --release --bin verify_proof)\n"
        "or point LIGERO_VERIFY_PROOF at the binary.")


# (proof serialization is now the single block-driven writer proof_dump.dump_proof;
#  rust_verify below calls it directly.)


def rust_verify(claims, proof, seed, cfg):
    """Dump `proof` via the single writer (proof_dump.dump_proof) and check it
    with the Rust verifier. Returns (accepted, output)."""
    from proof_dump import dump_proof
    # A Fiat-Shamir proof carries its own coins/columns (dump_proof prefers
    # them); the legacy test prover (tests/test_prover.prove) does not, so the
    # base-seed expansion stays as the fallback for it.
    if getattr(proof, "seeds", None):
        s_col = proof.seeds["s_col"]
        seeds = {k: v.hex() for k, v in proof.seeds.items()}
    else:
        s_op, s_comb, s_col = pr.round_seeds(seed)
        seeds = {"s_op": s_op.hex(), "s_comb": s_comb.hex(), "s_col": s_col.hex()}
    Q = list(pr.random_columns(s_col, cfg))
    # The verifier is fail-closed on policy: a proof carrying a statement
    # digest must be given a trusted one, and a proof with a weight block must
    # be given the enrolled root. In these tests the policy values come from
    # the proof itself — that is deliberately circular and only checks the
    # claim mechanics; policy ENFORCEMENT (wrong or missing digest/root) is
    # tested in test_fiat_shamir.py.
    stmt = getattr(proof, "statement_digest", None)
    root_w = getattr(proof, "root_w", None)
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    argv = [_verify_proof_bin(), path,
            root_w.hex() if root_w else "-",
            stmt.hex() if stmt else "-"]
    try:
        dump_proof(path, pr.claims_to_json(claims, cfg), seeds, proof, Q, None)
        r = subprocess.run(argv, capture_output=True, text=True)
    finally:
        os.unlink(path)
    accepted = "rust_verify: ACCEPT" in r.stdout
    return accepted, (r.stdout + r.stderr).strip()


def rust_verify_tape(tape, proof, seed):
    """Convenience for the tape-based tests: pulls claims + cfg off the tape."""
    return rust_verify(tape.claims, proof, seed, tape.cfg)
