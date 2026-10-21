"""Unexplained-information bound at LLAMA VOCAB SCALE (V=32000) with the
PRODUCTION Ligero config, on sane synthetic logits, dumping proof.json for the
standalone Rust verifier. Demonstrates the hidden-output construction (output
tokens committed + blinded like weights) scales to the real vocab and verifies
in the TCB -- the piece demo_llama7b.py wires after the LM head.

Synthetic (not live-forward) logits are used ON PURPOSE: this pipeline's int
forward currently emits massive-activation outliers spanning ~2^32, so gap^2
overflows Goldilocks (P ~ 2^64) and the range tables explode (a pre-existing
int-forward scale issue; see project_seq100_range_overflow). Here the logits are
in a sane range so gap^2 stays in-field and the tables are small.

  ~/venv-hf/bin/python analysis/ui_llama_scale.py /tmp/ui_llama.json [cheat]
  verifier-rs/target/release/verify_proof /tmp/ui_llama.json
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
from unexplained_info import (prove_unexplained_info, bound_bits,
                               unexplained_info_reference)

CFG = core.LigeroConfig(ELL=8192, K_DEG=16384, N_LIG=65536, T_QUERIES=80)
SEED = b"ui-llama-scale"
T, V = 4, 32000
S_C, S_Y, S_B = 1 << 17, 1 << 20, 1 << 12     # s_c=2*sigma^2 (sigma~256); k=s_c/s_b=32
GAP_MAX = 3402                                 # background gap 3400 (kernel ~0 there)
# All logits >= 0 (so the uint64 commit is the field rep). Max logit 3400; a few
# "candidate" tokens sit at gaps {0,50,120,200,350}; the other ~32k tokens are at
# 0 (gap 3400 -> weight ~0). Output token = the gap-120 one.
VMAX = 3400
CANDIDATES = [(0, VMAX), (7, VMAX - 50), (42, VMAX - 120),
              (99, VMAX - 200), (123, VMAX - 350)]                   # (token, logit)
TOKENS = [42, 7, 0, 99]                                              # hidden outputs (per row)
PARAMS = dict(T=T, V=V, s_c=S_C, s_y=S_Y, s_b=S_B, gap_max=GAP_MAX)


def _logits():
    rows = torch.zeros((T, V), dtype=torch.int64, device="cuda")     # background gap = VMAX
    for tok, val in CANDIDATES:
        rows[:, tok] = val
    return rows


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/ui_llama.json"
    cheat = len(sys.argv) > 2 and sys.argv[2] == "cheat"
    rows = _logits()
    tape = Tape(CFG, lazy=True)
    logits = tape.commit("logits", rows.reshape(-1).to(torch.uint64), (T, V))
    Sz, _ = prove_unexplained_info(
        tape, logits, TOKENS, force_argmax=(TOKENS if cheat else None), **PARAMS)

    live = tape.run_engine_pass()
    U = bound_bits(int(live[Sz.var].cpu().item()), s_b=S_B)
    U_ref = unexplained_info_reference(rows.cpu().tolist(), TOKENS, S_C)
    print(f"  V={V}, T={T}, s_c={S_C}: U={U:.4f} bits  U_ref={U_ref:.4f}  "
          f"d={U - U_ref:+.4f}  (output tokens hidden)")

    core._COSET_POWERS_K_CACHE.clear()
    tape2 = Tape(CFG, lazy=True)
    l2 = tape2.commit("logits", rows.reshape(-1).to(torch.uint64), (T, V))
    prove_unexplained_info(
        tape2, l2, TOKENS, force_argmax=(TOKENS if cheat else None), **PARAMS)
    proof = tape2.prove(seed=SEED)
    accepted, msg = tape2.verify(proof, seed=SEED)
    print(f"  python verify: {'ACCEPT' if accepted else 'REJECT'}  ({msg})")

    s_op, s_comb, s_col = pr.round_seeds(SEED)
    Q = pr.random_columns(s_col, CFG)
    _i = lambda t: [int(v) for v in t.cpu().tolist()]
    _pj = lambda p: [[sib.hex(), int(side)] for sib, side in p]
    with open(out_path, "w") as f:
        json.dump({"claims": pr.claims_to_json(tape2.claims, CFG),
                   "seeds": {"s_op": s_op.hex(), "s_comb": s_comb.hex(), "s_col": s_col.hex()},
                   "proof": {"root_p1": proof.root_p1.hex(), "root_p2": proof.root_p2.hex(),
                             "q_irs": _i(proof.q_irs), "q_lin": _i(proof.q_lin), "p_0": _i(proof.p_0),
                             "opened_p1": {str(j): _i(proof.opened_p1[j]) for j in Q},
                             "opened_p2": {str(j): _i(proof.opened_p2[j]) for j in Q},
                             "paths_p1": {str(j): _pj(proof.paths_p1[j]) for j in Q},
                             "paths_p2": {str(j): _pj(proof.paths_p2[j]) for j in Q}},
                   "python_accept": bool(accepted)}, f)
    print(f"  dumped proof -> {out_path}")


if __name__ == "__main__":
    main()
