"""Show the proven U converging to the float U as s_y grows (the floor-1 partition
over-count shrinks as ~V/s_y). Runs the U-bound WITNESS on the saved real int logits
(no 32-layer forward), summed over the 50 output positions, at a few s_y; compares to
the float reference at the same s_c.

  PYTHONPATH=~/ligero/pipeline ~/venv-hf/bin/python analysis/ui_sy_sweep.py
"""
import sys
import pathlib
import json
import math

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "prover"))
import torch
import numpy as np
import core
import claims as _C        # noqa: F401
import packets as _PK       # noqa: F401
import max_claim as _MX     # noqa: F401
import ui_claim as _UI      # noqa: F401
from tape import Tape
from cuda_primitives import P, gl_sub
from unexplained_info import prove_unexplained_info, bound_bits

CFG = core.LigeroConfig(ELL=8192, K_DEG=16384, N_LIG=65536, T_QUERIES=80)
T, V, S_C, S_B, N_IN = 100, 32000, 1 << 18, 1 << 12, 50
positions = list(range(N_IN - 1, T - 1))                  # 49..98


def load():
    seq = json.load(open("/tmp/ui_real/seq.json"))
    a = np.load("/tmp/ui_real/int_logits.npy").astype(np.int64)
    signed = a + (a < 0) * ((1 << 32) - 1)                # unwrap field-rep -> signed
    out = (seq[1:] + [seq[-1]])                           # next-token target per position
    return signed, out


def to_field(signed):
    st = torch.tensor(signed.reshape(-1), device="cuda")  # int64
    absu = st.abs().to(torch.uint64)
    negf = gl_sub(torch.full_like(absu, P), absu)
    return torch.where(st >= 0, st, negf.view(torch.int64)).view(torch.uint64)


def float_U(signed, out):
    U = 0.0
    for p in positions:
        l = signed[p].astype(np.float64)
        w = np.exp(-((l.max() - l) ** 2) / S_C)
        U += -math.log2(w[out[p]] / w.sum())
    return U


def main():
    signed, out = load()
    gap_max = int((signed.max(1, keepdims=True) - signed).max()) + 2
    print(f"  real int logits: gap_max={gap_max:,}, float U (s_c={S_C}, 50 outputs) "
          f"= {float_U(signed, out):.3f} bits")
    for s_y_bits in (20, 24, 28):
        core._COSET_POWERS_K_CACHE.clear()
        tape = Tape(CFG, lazy=True)
        logits = tape.commit("logits", to_field(signed), (T, V))
        Sz, _ = prove_unexplained_info(tape, logits, out, T=T, V=V, s_c=S_C,
                                       s_y=1 << s_y_bits, s_b=S_B, gap_max=gap_max,
                                       sum_positions=positions)
        live = tape.run_engine_pass()
        U = bound_bits(int(live[Sz.var].cpu().item()), s_b=S_B)
        print(f"    s_y=2^{s_y_bits}: proven U = {U:.3f} bits over 50 outputs")


if __name__ == "__main__":
    main()
