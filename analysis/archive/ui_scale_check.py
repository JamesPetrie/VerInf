"""Witness-level check of the unexplained-information bound at large context.

Runs the circuit WITNESS only (build the tape, run_engine_pass) -- the exact
quantised arithmetic the prover would commit -- and compares to the vectorised
float reference, skipping prove/verify. The proof's encode/Merkle cost is what
we skip; the witness is just GPU tensor ops, so this validates the computation
at realistic T x V in seconds.

Synthetic logits are peaked (a flat background plus one boosted token per row),
so the greedy output (= argmax) has low U and a random output has high U -- a
range that exercises both the near-1 win prob and the far tail.

Run on the Spark:  ~/venv-hf/bin/python analysis/ui_scale_check.py
"""
import sys
import pathlib
import time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "prover"))  # pipeline/

import torch
import core
import claims as _C        # noqa: F401 -- registers COMPILE/SAMPLE/AUX
import packets as _PK       # noqa: F401 -- registers EXPANDERS
from tape import Tape
from unexplained_info import witness_bound, reference_bits

CFG = core.LigeroConfig(ELL=8, K_DEG=8, N_LIG=32, T_QUERIES=4)
WIN_SCALE = 1 << 10
LOG_SCALE = 1 << 10
D_MAX = 256


def _peaked_logits(T, V, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randint(0, 60, (T, V), generator=g, device="cuda", dtype=torch.int64)
    peak = torch.randint(0, V, (T,), generator=g, device="cuda")
    val = torch.randint(120, D_MAX, (T,), generator=g, device="cuda")
    logits[torch.arange(T, device="cuda"), peak] = val
    return logits


def run(T, V, sigma, greedy, win_scale=WIN_SCALE, seed=0):
    logits = _peaked_logits(T, V, seed)
    if greedy:
        tokens = logits.argmax(dim=1)                              # output = argmax -> low U
    else:
        g = torch.Generator(device="cuda").manual_seed(seed + 1)
        tokens = torch.randint(0, V, (T,), generator=g, device="cuda")   # -> high U

    params = dict(T=T, V=V, sigma_eff=sigma, d_max=D_MAX,
                  win_scale=win_scale, log_scale=LOG_SCALE, slack_bits=24)
    tape = Tape(CFG, lazy=True)
    lw = tape.commit("logits", logits.reshape(-1).to(torch.uint64), (T, V))
    t0 = time.time()
    U_q = witness_bound(tape, lw, tokens.cpu().tolist(), **params)
    dt = time.time() - t0
    U_f = reference_bits(logits, tokens, sigma)
    tag = "greedy" if greedy else "random"
    print(f"T={T:5d} V={V:6d} sigma={sigma:4.0f} ws=2^{win_scale.bit_length()-1:<2d} {tag:6s}: "
          f"U_wit={U_q:11.3f}  U_flt={U_f:11.3f}  |d|/tok={abs(U_q - U_f) / T:.4f}  ({dt:.1f}s)")


def main():
    print("# error vs sigma (flatter effective distribution -> bigger dropped tail):")
    for sigma in (5.0, 20.0, 40.0):
        run(128, 32000, sigma, greedy=True)
    print("# same sigma=20, but raise win_scale to resolve the tail:")
    for ws in (1 << 10, 1 << 16, 1 << 22):
        run(128, 32000, 20.0, greedy=True, win_scale=ws)


if __name__ == "__main__":
    main()
