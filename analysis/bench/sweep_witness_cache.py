"""Sweep the witness cache OFF vs ON across the ledger's config range.
Confirms the iter2 win generalizes, checks peak GPU memory / whether the
element cap engages at the top, and produces cache-ON prove_s to replace the
stale pre-cache ledger rows. Same-process, fixed seed => identical model per
config; cache flipped via core._WITNESS_CACHE_ON. Logs each config as it
finishes so partial progress is durable if interrupted."""
import sys, time, os, gc
from pathlib import Path
R = Path(__file__).resolve().parents[2]
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

# (d, d_ff, d_h, SEQ, layers, reps) — reps kept low at the expensive top end.
CONFIGS = [
    (16,   32,  8,    4, 1, 3),   # toy: cache should ~no-op (negligible witness)
    (512, 1024, 64, 256, 4, 2),
    (512, 2048, 64, 512, 4, 1),
    (512, 1536, 64, 768, 4, 1),
    (512, 2048, 64, 1024, 4, 1),
]


def build(D, DFF, DH, SEQ, NL):
    dt.d, dt.d_ff, dt.d_h, dt.SEQ = D, DFF, DH, SEQ
    H = D // DH
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


def measure(D, DFF, DH, SEQ, NL, reps, cache_on):
    C._WITNESS_CACHE_ON = cache_on
    times, wit, peak, elems = [], [], 0, 0
    for _ in range(reps):
        torch.manual_seed(1234)
        gc.collect(); torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        tape = build(D, DFF, DH, SEQ, NL)
        C._PHASE_TIMES.clear()
        torch.cuda.synchronize()
        t0 = time.time()
        tape.prove(seed=b"sweep-cache")
        torch.cuda.synchronize()
        times.append(time.time() - t0)
        wit.append(dict(C._PHASE_TIMES).get("witness", 0.0))
        peak = max(peak, torch.cuda.max_memory_allocated())
        del tape
        gc.collect(); torch.cuda.empty_cache()
    times.sort(); wit.sort()
    return times[len(times) // 2], wit[len(wit) // 2], peak / 1e9


print(f"{'config':<26} {'off_s':>8} {'on_s':>8} {'faster':>8} "
      f"{'wit_off':>8} {'wit_on':>8} {'peakGB_on':>9}")
for (D, DFF, DH, SEQ, NL, reps) in CONFIGS:
    label = f"d{D},ff{DFF},dh{DH},seq{SEQ},L{NL}"
    off_s, off_wit, _ = measure(D, DFF, DH, SEQ, NL, reps, False)
    on_s, on_wit, peak_on = measure(D, DFF, DH, SEQ, NL, reps, True)
    speedup = off_s / on_s if on_s else 0.0
    saved = off_s - on_s
    pct = 100 * saved / off_s if off_s else 0.0
    print(f"{label:<26} {off_s:8.2f} {on_s:8.2f} {pct:7.1f}% "
          f"{off_wit:8.2f} {on_wit:8.2f} {peak_on:9.2f}")
    log_run(kind="witness_cache_sweep", label=label,
            params=dict(ELL=CFG.ELL, K_DEG=CFG.K_DEG, N_LIG=CFG.N_LIG,
                        T_QUERIES=CFG.T_QUERIES, d=D, d_ff=DFF, d_h=DH, SEQ=SEQ,
                        num_layers=NL, reps=reps,
                        cached_types="SoftmaxClaim,SiluClaim"),
            measured=dict(prove_off_s=off_s, prove_on_s=on_s, speedup=speedup,
                          saved_s=saved, pct_faster=pct, witness_off_s=off_wit,
                          witness_on_s=on_wit, peak_gb_on=peak_on),
            notes="cache OFF vs ON sweep across ledger range; peak_gb_on is "
                  "torch peak alloc with cache on (OOM/cap check).")
    sys.stdout.flush()
print("DONE")
