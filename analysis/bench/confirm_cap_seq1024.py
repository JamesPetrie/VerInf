"""Confirm the seq1024 win collapse is caused by _WITNESS_CACHE_MAX_ELEMS
engaging (not the cache being ineffective at scale). Re-run cache ON at seq1024
with the cap lifted; if prove_s drops toward the ~42% seen at seq512/768 and
peak GPU mem stays well under 32GB, the cap is the culprit and is too tight.
Also prints the cumulative cached-element count to show where the cap bites."""
import sys, time, os, gc
from pathlib import Path
R = Path("/home/riftuser/VerInf")
sys.path.insert(0, str(R / "prover")); sys.path.insert(0, str(R / "demo"))
sys.path.insert(0, str(R / "analysis/bench"))
os.environ["LIGERO_PHASE_TIMING"] = "1"
import _uint64_compat  # noqa
import torch
import demo_toy_transformer as dt
import core as C
from core import LigeroConfig
from tape import Tape
from run_log import log_run

CFG = LigeroConfig(ELL=512, K_DEG=1024, N_LIG=4096, T_QUERIES=16)
D, DFF, DH, SEQ, NL = 512, 2048, 64, 1024, 4
dt.d, dt.d_ff, dt.d_h, dt.SEQ = D, DFF, DH, SEQ
H = D // DH


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


def run(cap):
    C._WITNESS_CACHE_ON = True
    C._WITNESS_CACHE_MAX_ELEMS = cap
    torch.manual_seed(1234)
    gc.collect(); torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    tape = build()
    C._PHASE_TIMES.clear()
    torch.cuda.synchronize(); t0 = time.time()
    tape.prove(seed=b"cap-confirm")
    torch.cuda.synchronize(); dt_s = time.time() - t0
    wit = dict(C._PHASE_TIMES).get("witness", 0.0)
    peak = torch.cuda.max_memory_allocated() / 1e9
    del tape; gc.collect(); torch.cuda.empty_cache()
    return dt_s, wit, peak


# Default cap (2e8) vs lifted cap (5e9 => effectively uncapped at this scale).
for name, cap in [("cap=2e8 (default)", 200_000_000), ("cap=5e9 (lifted)", 5_000_000_000)]:
    s, wit, peak = run(cap)
    want = int(SEQ * SEQ * H * NL)  # rough softmax-matrix element scale, for context
    print(f"{name:<20} prove {s:7.2f}s  witness {wit:7.2f}s  peak {peak:.2f}GB")
    log_run(kind="witness_cache_cap_confirm", label=f"d{D},ff{DFF},dh{DH},seq{SEQ},L{NL}",
            params=dict(ELL=CFG.ELL, K_DEG=CFG.K_DEG, N_LIG=CFG.N_LIG,
                        T_QUERIES=CFG.T_QUERIES, d=D, d_ff=DFF, d_h=DH, SEQ=SEQ,
                        num_layers=NL, cap=cap),
            measured=dict(prove_s=s, witness_s=wit, peak_gb=peak),
            notes=f"seq1024 cap-confirm ({name}); OFF baseline from sweep = 214.89s.")
print("DONE")
