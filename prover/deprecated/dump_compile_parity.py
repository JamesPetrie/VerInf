"""Dump per-op compile-parity cases (claims + s_op + protocol.py's canonical
constraint system) to JSON for the Rust compile differential test.

Witness-free: compile_claims needs only the public claim list + s_op, so this
covers ALL 8 op handlers + 13 expanders without constructing any witness. The
Rust side (bin/compile_difftest) reads each case, runs its own compile_claims,
canonicalizes identically, and asserts equality. Public data only.

Run on the Spark (cases() builds tape-based ops needing torch/CUDA):
    ~/venv-hf/bin/python dump_compile_parity.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # pipeline/ on path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[0]))  # deprecated/ (python_verifier_compile)
import json, sys
import torch              # noqa: F401 — cases() builds tape ops on CUDA
import core               # noqa: F401
import claims as C        # noqa: F401 — registers COMPILE/SAMPLE/AUX
import packets as PK      # noqa: F401 — registers EXPANDERS
import protocol as pr
import python_verifier_compile as pvc
from test_compile_parity import cases, canon_mine, canon_quads_mine, canon_rhs_mine

S_OP = b"parity-s_op"     # same op seed compare() uses


def dump_case(tag, claim_list, cfg):
    cl = list(claim_list)
    # Assign each Variable's row_start by running the layout (it mutates the
    # Variable objects in place), exactly as compare() / the prover do — else the
    # vars are unbound (row_start = -1) and both the serialized row_starts and
    # compile_claims' row indexing would be wrong. Layout runs on the settled
    # list (ops + synthesized table settlements), then we compile/serialize the
    # ops (compile_claims rediscovers + settles the tables itself).
    core._layout(core._with_synthesized_settlements(list(cl)), cfg)
    cons = pvc.compile_claims(cl, cfg, S_OP)
    lin = canon_mine(cons, cfg)              # {(r,slot,cid): Σcoef}, nonzero
    quad = canon_quads_mine(cons.quadratic)  # sorted [(x,y,z,n,a,b)]
    rhs = canon_rhs_mine(cons.rhs)           # {cid: b}, nonzero
    return {
        "tag": tag,
        "claims": pr.claims_to_json(cl, cfg),
        "s_op": S_OP.hex(),
        "lin": sorted([list(k) + [v] for k, v in lin.items()]),
        "quad": [list(q) for q in quad],
        "rhs": sorted([[cid, b] for cid, b in rhs.items()]),
        "m_total": cons.m_total,
    }


def main():
    out = []
    for tag, claim_list, cfg in cases():
        d = dump_case(tag, claim_list, cfg)
        out.append(d)
        print(f"dumped {tag}: lin {len(d['lin'])} quad {len(d['quad'])} "
              f"rhs {len(d['rhs'])} m_total {d['m_total']}", file=sys.stderr)
    with open("/tmp/compile_parity.json", "w") as f:
        json.dump(out, f)
    print(f"wrote /tmp/compile_parity.json ({len(out)} cases)", file=sys.stderr)


if __name__ == "__main__":
    main()
