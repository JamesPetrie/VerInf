"""A/B: prove() wall time with the witness cache OFF vs ON, on a medium config.
Same process, fixed seed (identical model), cache flipped via core._WITNESS_CACHE_ON.
Reports median of a few reps + the witness-phase share, and logs the result."""
import sys, time, os
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
D, DFF, DH, SEQ, NL = 512, 1536, 64, 384, 4
dt.d, dt.d_ff, dt.d_h, dt.SEQ = D, DFF, DH, SEQ
H = D // DH
REPS = 2


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


def measure(cache_on):
    C._WITNESS_CACHE_ON = cache_on
    times, wit = [], []
    for _ in range(REPS):
        torch.manual_seed(1234)
        tape = build()
        C._PHASE_TIMES.clear()
        torch.cuda.synchronize()
        t0 = time.time()
        tape.prove(seed=b"ab-cache")
        torch.cuda.synchronize()
        times.append(time.time() - t0)
        wit.append(dict(C._PHASE_TIMES).get("witness", 0.0))
        del tape
        torch.cuda.empty_cache()
    times.sort(); wit.sort()
    return times[len(times) // 2], wit[len(wit) // 2]


off_s, off_wit = measure(False)
on_s, on_wit = measure(True)
speedup = off_s / on_s
saved = off_s - on_s
print(f"config d{D},ff{DFF},seq{SEQ},L{NL}  (median of {REPS} reps each)")
print(f"  cache OFF: prove {off_s:.2f}s  (witness phase {off_wit:.2f}s)")
print(f"  cache ON : prove {on_s:.2f}s  (witness phase {on_wit:.2f}s)")
print(f"  saved    : {saved:.2f}s  = {100*saved/off_s:.1f}% faster  ({speedup:.2f}x)")
print(f"  witness phase dropped {off_wit:.2f}s -> {on_wit:.2f}s "
      f"({100*(off_wit-on_wit)/off_wit:.0f}% of the witness term)")

# production lens: the GPU-memory cache is TEST-SCALE ONLY — the 7.5TB witness
# cannot be held in GPU memory at 400B, so the mechanism does not exist there.
sys.path.insert(0, str(R / "analysis/bench"))
import prod_lens
prod_lens.report("witness cache (GPU-mem)", 100 * saved / off_s, transfers=False)

log_run(kind="witness_cache_ab", label=f"d{D},ff{DFF},seq{SEQ},L{NL}",
        params=dict(ELL=CFG.ELL, K_DEG=CFG.K_DEG, N_LIG=CFG.N_LIG, T_QUERIES=CFG.T_QUERIES,
                    d=D, d_ff=DFF, d_h=DH, SEQ=SEQ, num_layers=NL, reps=REPS,
                    cached_types="SoftmaxClaim,SiluClaim"),
        measured=dict(prove_off_s=off_s, prove_on_s=on_s, speedup=speedup,
                       saved_s=saved, pct_faster=100 * saved / off_s,
                       witness_off_s=off_wit, witness_on_s=on_wit),
        notes="witness cache (softmax+silu compute_fn outputs reused across the 4 "
              "sweeps). Proof byte-identical (validate_witness_cache.py) + Rust ACCEPT.")
