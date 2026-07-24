"""Fresh prove() phase decomposition at seq1024 with gpu-softmax + cache ON,
so iter9 attacks the CURRENT top bucket (witness is now ~6%, the old shares are
stale)."""
import sys, os
from pathlib import Path
R = Path("/home/riftuser/VerInf")
sys.path.insert(0, str(R / "prover")); sys.path.insert(0, str(R / "demo"))
os.environ["LIGERO_PHASE_TIMING"] = "1"
import _uint64_compat  # noqa
import torch
import demo_toy_transformer as dt
import core as C
import compute_fns as CF
from core import LigeroConfig
from tape import Tape

CFG = LigeroConfig(ELL=512, K_DEG=1024, N_LIG=4096, T_QUERIES=16)
D, DFF, DH, SEQ, NL = 512, 2048, 64, 1024, 4
dt.d, dt.d_ff, dt.d_h, dt.SEQ = D, DFF, DH, SEQ
H = D // DH
C._WITNESS_CACHE_ON = True
CF._GPU_SOFTMAX_ON = True


def build():
    tape = Tape(CFG, silu_config=dt.SILU_CFG, lazy=True)
    x = dt._rand_signed(SEQ * D, half=dt.HALF_X)
    resid = tape.commit("x_input", x, (SEQ, D))
    for L in range(NL):
        w = dt._commit_weights_random(tape, layer_idx=L)
        resid = dt._run_block(tape, resid, w, H=H)
    vocab = 64
    fn = torch.full((D,), dt.S, dtype=torch.uint64, device="cuda")
    lm = dt._rand_signed(D * vocab, half=dt.HALF)
    fnw = tape.commit("final_norm_w", fn, (D,))
    lmw = tape.commit("W_lm_head", lm, (D, vocab))
    dt._run_tail(tape, resid, fnw, lmw, vocab_size=vocab)
    return tape


torch.manual_seed(1234)
build().prove(seed=b"phase-dump")
print("DONE")
