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
    s_op, s_comb, s_col = pr.round_seeds(seed)
    Q = list(pr.random_columns(s_col, cfg))
    seeds = {"s_op": s_op.hex(), "s_comb": s_comb.hex(), "s_col": s_col.hex()}
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        dump_proof(path, pr.claims_to_json(claims, cfg), seeds, proof, Q, None)
        r = subprocess.run([_verify_proof_bin(), path],
                           capture_output=True, text=True)
    finally:
        os.unlink(path)
    accepted = "rust_verify: ACCEPT" in r.stdout
    return accepted, (r.stdout + r.stderr).strip()


def rust_verify_tape(tape, proof, seed):
    """Convenience for the tape-based tests: pulls claims + cfg off the tape."""
    return rust_verify(tape.claims, proof, seed, tape.cfg)
