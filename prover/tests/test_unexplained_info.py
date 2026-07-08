"""Unexplained-information UPPER bound (explicit a = Sum exp(-gap^2/s_c) + log-pin).

POSITIVE: prove + verify (ACCEPT); U is a sound over-estimate of the float
reference (U >= U_ref) and tight (within tol).

SOUNDNESS: force_argmax = the output token (a non-max v*). The gaps for the true
max then go negative (field P-x), which has no [0,gap_max) entry in the EXP table,
so the paired lookup's range LogUp REJECTS.

Run on the Spark:  PYTHONPATH=~/ligero/pipeline ~/venv-hf/bin/python tests/test_unexplained_info.py
"""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import core
import claims as _C        # noqa: F401
import packets as _PK       # noqa: F401
import max_claim as _MX     # noqa: F401
import ui_claim as _UI      # noqa: F401
from tape import Tape
from _rust_verify import rust_verify_tape
from unexplained_info import (prove_unexplained_info, unexplained_info_reference,
                              bound_bits)

CFG = core.LigeroConfig(ELL=8, K_DEG=8, N_LIG=32, T_QUERIES=4)
SEED = b"unexplained-info-test"

T, V = 2, 8
S_C = 256            # = 2*sigma^2  (sigma ~ 11.3); kernel exp(-gap^2/256)
S_Y = 1 << 12        # >> V, far-token over-count negligible
S_B = 16             # surprisal fixed-point (nats * s_b); k = s_c/s_b = 16
GAP_MAX = 128
PARAMS = dict(T=T, V=V, s_c=S_C, s_y=S_Y, s_b=S_B, gap_max=GAP_MAX)

#        gaps 0..40 (some saturate to ~0 -> floor-1)
COMPACT = [[40, 8, 32, 0, 24, 16, 36, 4], [4, 44, 12, 36, 20, 8, 40, 28]]


def _t(rows):
    flat = [v for row in rows for v in row]
    return torch.tensor(flat, dtype=torch.int64, device="cuda").to(torch.uint64)


def run_positive(label, rows, tokens, tol=0.3):
    core._COSET_POWERS_K_CACHE.clear()
    tape = Tape(CFG, lazy=True)
    logits = tape.commit("logits", _t(rows), (T, V))
    Sz, _ = prove_unexplained_info(tape, logits, tokens, **PARAMS)
    live = tape.run_engine_pass()
    U = bound_bits(int(live[Sz.var].cpu().item()), s_b=S_B)
    U_ref = unexplained_info_reference(rows, tokens, S_C)

    core._COSET_POWERS_K_CACHE.clear()
    tape2 = Tape(CFG, lazy=True)
    l2 = tape2.commit("logits", _t(rows), (T, V))
    prove_unexplained_info(tape2, l2, tokens, **PARAMS)
    proof = tape2.prove(seed=SEED)
    acc, msg = rust_verify_tape(tape2, proof, seed=SEED)

    ok = acc and (U_ref - 0.05) <= U < U_ref + tol         # sound (>=) and tight
    print(f"[{'OK ' if ok else 'XX '}] {label}: verify={'ACCEPT' if acc else 'REJECT'}  "
          f"U={U:.3f}  U_ref={U_ref:.3f}  d={U - U_ref:+.3f}  ({msg})")
    return ok


def run_cheat(label, rows, tokens):
    core._COSET_POWERS_K_CACHE.clear()
    tape = Tape(CFG, lazy=True)
    logits = tape.commit("logits", _t(rows), (T, V))
    prove_unexplained_info(tape, logits, tokens, force_argmax=tokens, **PARAMS)
    proof = tape.prove(seed=SEED)
    acc, msg = rust_verify_tape(tape, proof, seed=SEED)
    ok = not acc
    print(f"[{'OK ' if ok else 'XX '}] {label}: verify={'ACCEPT' if acc else 'REJECT'}  "
          f"(want REJECT)  ({msg})")
    return ok


def main():
    results = [
        run_positive("compact, tokens=[2,3]", COMPACT, [2, 3]),
        run_positive("compact, max tokens=[0,1]", COMPACT, [0, 1]),
        run_cheat("cheat: v* = output token", COMPACT, [2, 3]),
    ]
    ok = all(results)
    print(f"\n=== unexplained_info: {sum(results)}/{len(results)} {'PASS' if ok else 'FAIL'} ===")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
