"""Iteration 1 probe: is the witness forward pass genuinely recomputed ~4x
(once per Fiat-Shamir sweep), and how much is that worth?

Theory (notebook): prove_streaming runs FOUR sweeps, each regenerating the
witness, because at 400B scale the witness can't be stored. At our test scale
it CAN (peak ~4GB), so if the per-sweep witness compute is identical and
deterministic, caching it across rounds could save ~3/4 of witness_recompute.

This probe does NOT change the prover. It measures, on one medium config:
  A) forward_only wall time  = ONE witness computation (engine pass)
  B) full prove witness-phase bucket (LIGERO_PHASE_TIMING) = the 4-sweep sum
If B ~= 4*A, the 4x recompute is confirmed and the cache ceiling is 3*A.
"""
import sys, time, os
from pathlib import Path
R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R / "prover")); sys.path.insert(0, str(R / "demo"))
sys.path.insert(0, str(R / "analysis/bench"))
os.environ["LIGERO_PHASE_TIMING"] = "1"
import _uint64_compat  # noqa
import torch
import demo_toy_transformer as dt
from core import LigeroConfig
from tape import Tape
import core as C
from run_log import log_run

CFG = LigeroConfig(ELL=512, K_DEG=1024, N_LIG=4096, T_QUERIES=16)
# medium config where witness is ~59% of prove (metric_ledger)
D, DFF, DH, SEQ, NL = 512, 1536, 64, 384, 4
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
    logits = dt._run_tail(tape, resid, fnw, lmw, vocab_size=vocab)
    return tape, logits


# --- A) one witness pass via a single engine pass (forward only, no proof) ---
tape, logits = build()
torch.cuda.synchronize()
t0 = time.time()
tape.run_engine_pass(free_intermediates=True, keep={logits.var})
torch.cuda.synchronize()
fwd_s = time.time() - t0
del tape
torch.cuda.empty_cache()

# --- B) full prove, read the witness phase bucket (sum over 4 sweeps) ---
tape, _ = build()
C._PHASE_TIMES.clear()
torch.cuda.synchronize()
t0 = time.time()
tape.prove(seed=b"witness-probe")
torch.cuda.synchronize()
prove_s = time.time() - t0
phases = dict(C._PHASE_TIMES)
witness_s = phases.get("witness", 0.0)

ratio = witness_s / fwd_s if fwd_s > 0 else float("nan")
print(f"config d{D},ff{DFF},seq{SEQ},L{NL}")
print(f"  A) one forward pass (engine)      : {fwd_s:.2f}s")
print(f"  B) prove witness-phase (4 sweeps) : {witness_s:.2f}s  ({100*witness_s/prove_s:.0f}% of {prove_s:.1f}s prove)")
print(f"  ratio B/A                         : {ratio:.2f}x  (theory says ~4x)")
print(f"  cache ceiling (save 3 of 4 passes): {3*fwd_s:.2f}s "
      f"= {100*3*fwd_s/prove_s:.0f}% of prove, IF witness is cacheable across rounds")

log_run(kind="witness_probe", label=f"d{D},ff{DFF},seq{SEQ},L{NL}",
        params=dict(ELL=CFG.ELL, K_DEG=CFG.K_DEG, N_LIG=CFG.N_LIG, T_QUERIES=CFG.T_QUERIES,
                    d=D, d_ff=DFF, d_h=DH, SEQ=SEQ, num_layers=NL),
        measured=dict(forward_one_pass_s=fwd_s, witness_phase_s=witness_s,
                       prove_s=prove_s, ratio_B_over_A=ratio,
                       cache_ceiling_s=3 * fwd_s),
        notes="iter1 probe: confirm the 4x witness recompute and quantify the "
              "ceiling of caching the witness across the 4 Fiat-Shamir sweeps")
