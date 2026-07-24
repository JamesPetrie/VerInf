"""A/B: prove() with GPU softmax OFF (numpy) vs ON, at seq512 and seq1024.
Same-process, fixed seed, CUDA-synced. Cache stays ON both sides (isolates the
softmax-path variable). Reports prove_s, witness phase, peak mem, and logs."""
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
import compute_fns as CF
from core import LigeroConfig
from tape import Tape
from run_log import log_run
import prod_lens

CFG = LigeroConfig(ELL=512, K_DEG=1024, N_LIG=4096, T_QUERIES=16)
CONFIGS = [(512, 2048, 64, 512, 4, 2), (512, 2048, 64, 1024, 4, 1)]
C._WITNESS_CACHE_ON = True


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


def measure(cfg, gpu_on, reps):
    D, DFF, DH, SEQ, NL, _ = cfg
    CF._GPU_SOFTMAX_ON = gpu_on
    times, wit, peak = [], [], 0
    for _ in range(reps):
        torch.manual_seed(1234)
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        tape = build(D, DFF, DH, SEQ, NL)
        C._PHASE_TIMES.clear()
        torch.cuda.synchronize(); t0 = time.time()
        tape.prove(seed=b"ab-gpu-sm")
        torch.cuda.synchronize()
        times.append(time.time() - t0)
        wit.append(dict(C._PHASE_TIMES).get("witness", 0.0))
        peak = max(peak, torch.cuda.max_memory_allocated() / 1e9)
        del tape; gc.collect(); torch.cuda.empty_cache()
    times.sort(); wit.sort()
    return times[len(times)//2], wit[len(wit)//2], peak


print(f"{'config':<22}{'off_s':>9}{'on_s':>9}{'faster':>9}{'wit_off':>9}{'wit_on':>9}{'peakGB':>8}")
for cfg in CONFIGS:
    D, DFF, DH, SEQ, NL, reps = cfg
    label = f"d{D},ff{DFF},dh{DH},seq{SEQ},L{NL}"
    off_s, off_w, _ = measure(cfg, False, reps)
    on_s, on_w, peak = measure(cfg, True, reps)
    pct = 100 * (off_s - on_s) / off_s
    print(f"{label:<22}{off_s:9.2f}{on_s:9.2f}{pct:8.1f}%{off_w:9.2f}{on_w:9.2f}{peak:8.2f}")
    log_run(kind="gpu_softmax_ab", label=label,
            params=dict(ELL=CFG.ELL, K_DEG=CFG.K_DEG, N_LIG=CFG.N_LIG, T_QUERIES=CFG.T_QUERIES,
                        d=D, d_ff=DFF, d_h=DH, SEQ=SEQ, num_layers=NL, reps=reps),
            measured=dict(prove_off_s=off_s, prove_on_s=on_s, pct_faster=pct,
                          speedup=off_s/on_s, witness_off_s=off_w, witness_on_s=on_w,
                          peak_gb=peak),
            notes="GPU softmax (numpy->torch int64 on cuda) vs numpy path; cache on both. "
                  "Byte-identical proof + Rust ACCEPT (validate_gpu_softmax.py).")
    # production lens: GPU softmax speeds the softmax part of the witness
    # recompute at ANY scale (it changes WHERE compute runs, not whether the
    # witness fits) -> transfers. Its per-term cut is the witness-phase fraction;
    # softmax is O(SEQ^2) so its witness share GROWS with context (this toy
    # projection is thus conservative for long-context production).
    wit_frac = (off_w - on_w) / off_w if off_w else 0.0
    prod_lens.report(f"GPU softmax [{label}]", pct,
                     prod_lens.LeverEffect(term="witness", toy_frac=wit_frac,
                                           note="(transfers; softmax share of witness "
                                                "grows with SEQ, so conservative)"))
    sys.stdout.flush()
print("DONE")
