"""Build the unexplained-information construction, prove, and dump proof.json for
the standalone Rust verifier (verify_proof) -- exercises compile_max (incl. the
hidden output-token select) in the TCB. Public data only (no witness); the output
tokens are committed + blinded, never in the public claim fields.

Run on the Spark:
  ~/venv-hf/bin/python analysis/dump_ui_proof.py /tmp/ui_proof.json
"""
import sys
import pathlib
import json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "prover"))

import torch
import core
import claims as _C        # noqa: F401
import packets as _PK       # noqa: F401
import max_claim as _MX     # noqa: F401 -- registers MaxClaim
import ui_claim as _UI      # noqa: F401 -- registers InfoFinalizeClaim
from tape import Tape
import protocol as pr
from unexplained_info import prove_unexplained_info, bound_bits, unexplained_info_reference

CFG = core.LigeroConfig(ELL=8, K_DEG=8, N_LIG=32, T_QUERIES=4)
SEED = b"ui-rust-test"
T, V = 2, 8
S_C, S_Y, S_B, GAP_MAX = 256, 1 << 12, 16, 128
LOGITS = [[40, 8, 32, 0, 24, 16, 36, 4], [4, 44, 12, 36, 20, 8, 40, 28]]
TOKENS = [2, 3]
PARAMS = dict(T=T, V=V, s_c=S_C, s_y=S_Y, s_b=S_B, gap_max=GAP_MAX)


def _t(rows):
    flat = [v for row in rows for v in row]
    return torch.tensor(flat, dtype=torch.int64, device="cuda").to(torch.uint64)


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/ui_proof.json"
    mode = sys.argv[2] if len(sys.argv) > 2 else ""
    cheat = mode == "cheat"          # v* = output token (non-max)
    tape = Tape(CFG, lazy=True)
    logits = tape.commit("logits", _t(LOGITS), (T, V))
    Sz, _ = prove_unexplained_info(tape, logits, TOKENS,
                                   force_argmax=(TOKENS if cheat else None), **PARAMS)
    proof = tape.prove(seed=SEED)
    accepted, msg = tape.verify(proof, seed=SEED)
    print(f"python verify: {'ACCEPT' if accepted else 'REJECT'}  ({msg})")

    s_op, s_comb, s_col = pr.round_seeds(SEED)
    Q = pr.random_columns(s_col, CFG)
    _i = lambda t: [int(v) for v in t.cpu().tolist()]
    _pj = lambda p: [[sib.hex(), int(side)] for sib, side in p]
    with open(out_path, "w") as f:
        json.dump({"claims": pr.claims_to_json(tape.claims, CFG),
                   "seeds": {"s_op": s_op.hex(), "s_comb": s_comb.hex(), "s_col": s_col.hex()},
                   "proof": {"root_p1": proof.root_p1.hex(), "root_p2": proof.root_p2.hex(),
                             "q_irs": _i(proof.q_irs), "q_lin": _i(proof.q_lin), "p_0": _i(proof.p_0),
                             "opened_p1": {str(j): _i(proof.opened_p1[j]) for j in Q},
                             "opened_p2": {str(j): _i(proof.opened_p2[j]) for j in Q},
                             "paths_p1": {str(j): _pj(proof.paths_p1[j]) for j in Q},
                             "paths_p2": {str(j): _pj(proof.paths_p2[j]) for j in Q}},
                   "python_accept": bool(accepted)}, f)
    print(f"dumped proof -> {out_path}")


if __name__ == "__main__":
    main()
