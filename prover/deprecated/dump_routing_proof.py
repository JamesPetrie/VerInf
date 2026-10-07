"""Dump toy routing+combine proofs for the standalone Rust verifier.

Builds the test_routing_claim positive case (expect ACCEPT) and the
wrong-expert cheat (expect REJECT), proves each, and dumps proof JSONs in the
demo's verify_proof format. Public data only — no witness.

Run on the Spark:
    PATH=~/venv-hf/bin:$PATH python tests/dump_routing_proof.py
Then:
    verifier-rs/target/release/verify_proof /tmp/routing_proof_ok.json     # ACCEPT
    verifier-rs/target/release/verify_proof /tmp/routing_proof_cheat.json  # REJECT
"""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json
import core
import claims as _C         # noqa: F401
import packets as _PK        # noqa: F401
import protocol as pr
from test_routing_claim import _build, LOGITS, SEED, CFG


def dump(path, **build_kw):
    tape, *_ = _build(LOGITS, with_combine=True, **build_kw)
    proof = tape.prove(seed=SEED)
    s_op, s_comb, s_col = pr.round_seeds(SEED)
    Q = pr.random_columns(s_col, CFG)
    _i = lambda t: [int(v) for v in t.cpu().tolist()]
    _pj = lambda p: [[sib.hex(), int(side)] for sib, side in p]
    with open(path, "w") as f:
        json.dump({"claims": pr.claims_to_json(tape.claims, CFG),
                   "seeds": {"s_op": s_op.hex(), "s_comb": s_comb.hex(), "s_col": s_col.hex()},
                   "proof": {"root_p1": proof.root_p1.hex(), "root_p2": proof.root_p2.hex(),
                             "q_irs": _i(proof.q_irs), "q_lin": _i(proof.q_lin), "p_0": _i(proof.p_0),
                             "opened_p1": {str(j): _i(proof.opened_p1[j]) for j in Q},
                             "opened_p2": {str(j): _i(proof.opened_p2[j]) for j in Q},
                             "paths_p1": {str(j): _pj(proof.paths_p1[j]) for j in Q},
                             "paths_p2": {str(j): _pj(proof.paths_p2[j]) for j in Q}}}, f)
    print(f"{path}: proof dumped (verify with verify_proof)")


if __name__ == "__main__":
    dump("/tmp/routing_proof_ok.json")
    dump("/tmp/routing_proof_cheat.json", force_expert=[2, 2])
