"""End-to-end ACCEPT gate for the mem-gated witness cache: build the SMALL toy
transformer (1 layer -> contains the cached softmax + silu ops), prove with the
cache ON (new default), and verify with the standalone Rust verifier. Must
print ACCEPT."""
import sys
from pathlib import Path
R = Path("/home/riftuser/VerInf")
sys.path.insert(0, str(R / "prover")); sys.path.insert(0, str(R / "prover/tests"))
sys.path.insert(0, str(R / "demo"))
import _uint64_compat  # noqa
import torch
import demo_toy_transformer as dt
import core as C
from core import LigeroConfig
from tape import Tape
from _rust_verify import rust_verify_tape

torch.manual_seed(1234)
CFG = LigeroConfig(ELL=512, K_DEG=1024, N_LIG=4096, T_QUERIES=16)
SEED = b"accept-toy-cache"
C._WITNESS_CACHE_ON = True  # explicit: new mem-gated cache active


def build():
    tape = Tape(CFG, silu_config=dt.SILU_CFG, lazy=True)
    x = dt._rand_signed(dt.SEQ * dt.d, half=dt.HALF_X)
    resid = tape.commit("x_input", x, (dt.SEQ, dt.d))
    w = dt._commit_weights_random(tape, layer_idx=0)
    resid = dt._run_block(tape, resid, w, H=dt.d // dt.d_h)
    vocab = 64
    fn = torch.full((dt.d,), dt.S, dtype=torch.uint64, device="cuda")
    lm = dt._rand_signed(dt.d * vocab, half=dt.HALF)
    fnw = tape.commit("final_norm_w", fn, (dt.d,))
    lmw = tape.commit("W_lm_head", lm, (dt.d, vocab))
    dt._run_tail(tape, resid, fnw, lmw, vocab_size=vocab)
    return tape


tape = build()
proof = tape.prove(seed=SEED)
acc, msg = rust_verify_tape(tape, proof, seed=SEED)
print(f"Rust verify_proof: {'ACCEPT' if acc else 'REJECT'}")
if not acc:
    print(msg)
sys.exit(0 if acc else 1)
