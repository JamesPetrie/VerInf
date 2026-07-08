"""Write the PUBLIC claims.json the Rust verifier loads (trusted statement).

The verifier must obtain the statement from a trusted source, NOT the untrusted
prover. This step assigns layout (row_start) and serializes only the public claim
structure — no witness. The prover (prover_server.py) builds the same statement
and additionally holds the witness; layout is deterministic from claim shapes, so
the row_starts here match the prover's.

Usage:  python setup_statement.py [matmul|chain|rmsnorm]  ->  /tmp/claims.json
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # pipeline/ on path
import json, sys
import core
import protocol as pr
from statements import STATEMENTS


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "matmul"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "/tmp/claims.json"
    claims, _inputs, cfg = STATEMENTS[name]()
    # Assign row_start to every Variable (value-independent), so the serialized
    # claims match the prover's layout.
    core._layout(core._with_synthesized_settlements(list(claims)), cfg)
    with open(out_path, "w") as f:
        json.dump(pr.claims_to_json(claims, cfg), f)
    print(f"wrote {out_path} for statement '{name}'", file=sys.stderr)


if __name__ == "__main__":
    main()
