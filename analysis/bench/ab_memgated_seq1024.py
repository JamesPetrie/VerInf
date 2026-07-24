"""Confirm the mem-gated cache (new default fraction 0.25 of free GPU mem)
recovers the full seq1024 win that the old fixed 2e8 cap blocked. Same-process
OFF vs ON, fixed seed, CUDA-synced prove wall-clock + witness phase + peak mem."""
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


def run(cache_on):
    C._WITNESS_CACHE_ON = cache_on  # ON uses the new mem-gated default fraction
    torch.manual_seed(1234)
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    tape = build()
    C._PHASE_TIMES.clear()
    torch.cuda.synchronize(); t0 = time.time()
    tape.prove(seed=b"ab-memgated")
    torch.cuda.synchronize(); s = time.time() - t0
    wit = dict(C._PHASE_TIMES).get("witness", 0.0)
    peak = torch.cuda.max_memory_allocated() / 1e9
    del tape; gc.collect(); torch.cuda.empty_cache()
    return s, wit, peak


off_s, off_wit, off_peak = run(False)
on_s, on_wit, on_peak = run(True)
pct = 100 * (off_s - on_s) / off_s
print(f"seq1024 mem-gated (frac={C._WITNESS_CACHE_MEM_FRACTION})")
print(f"  OFF: prove {off_s:.2f}s  witness {off_wit:.2f}s  peak {off_peak:.2f}GB")
print(f"  ON : prove {on_s:.2f}s  witness {on_wit:.2f}s  peak {on_peak:.2f}GB")
print(f"  -> {pct:.1f}% faster ({off_s/on_s:.2f}x)")
log_run(kind="witness_cache_memgated_ab", label=f"d{D},ff{DFF},dh{DH},seq{SEQ},L{NL}",
        params=dict(ELL=CFG.ELL, K_DEG=CFG.K_DEG, N_LIG=CFG.N_LIG, T_QUERIES=CFG.T_QUERIES,
                    d=D, d_ff=DFF, d_h=DH, SEQ=SEQ, num_layers=NL,
                    mem_fraction=C._WITNESS_CACHE_MEM_FRACTION),
        measured=dict(prove_off_s=off_s, prove_on_s=on_s, pct_faster=pct,
                      speedup=off_s/on_s, witness_off_s=off_wit, witness_on_s=on_wit,
                      peak_gb_off=off_peak, peak_gb_on=on_peak),
        notes="mem-gated cache (free-mem fraction) vs OFF at seq1024; replaces "
              "the fixed 2e8 cap that limited this config to -10.8%.")
print("DONE")
