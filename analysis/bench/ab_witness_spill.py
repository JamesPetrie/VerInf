"""A/B: prove() wall time across the three witness backing strategies, same
process + fixed seed (identical model), phase timing on:
    recompute  — no reuse, forward pass redone every sweep (SPILL off, CACHE off)
    gpu-cache  — deterministic softmax/silu reused from GPU memory (capacity-limited)
    host-spill — same reuse but from pinned HOST memory, re-read over PCIe (SPILL on)
The spill's claim is "re-read from host beats recompute"; this measures it
directly (host-spill vs recompute) and shows the GPU-cache ceiling for context."""
import sys, time, os
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
import prod_lens

_ELL = int(os.environ.get("AB_ELL", "512"))  # must be >= d (embedding lookup); pow2
CFG = LigeroConfig(ELL=_ELL, K_DEG=2 * _ELL, N_LIG=8 * _ELL, T_QUERIES=16)
D = int(os.environ.get("AB_D", "512"))
DFF = int(os.environ.get("AB_DFF", str(3 * D)))
DH = int(os.environ.get("AB_DH", "64"))
SEQ = int(os.environ.get("AB_SEQ", "512"))
NL = int(os.environ.get("AB_NL", "4"))
dt.d, dt.d_ff, dt.d_h, dt.SEQ = D, DFF, DH, SEQ
H = D // DH
REPS = int(os.environ.get("AB_REPS", "3"))


def build():
    tape = Tape(CFG, silu_config=dt.SILU_CFG, lazy=True)
    x = dt._rand_signed(SEQ * D, half=dt.HALF_X)
    resid = tape.commit("x_input", x, (SEQ, D))
    for L in range(NL):
        w = dt._commit_weights_random(tape, layer_idx=L)
        resid = dt._run_block(tape, resid, w, H=H)
    vocab = int(os.environ.get("AB_VOCAB", "64"))
    fn = torch.full((D,), dt.S, dtype=torch.uint64, device="cuda")
    lm = dt._rand_signed(D * vocab, half=dt.HALF)
    fnw = tape.commit("final_norm_w", fn, (D,))
    lmw = tape.commit("W_lm_head", lm, (D, vocab))
    dt._run_tail(tape, resid, fnw, lmw, vocab_size=vocab)
    return tape


def measure(spill_on, cache_on):
    C._WITNESS_SPILL_ON = spill_on
    C._WITNESS_CACHE_ON = cache_on
    times, wit = [], []
    for _ in range(REPS):
        torch.manual_seed(1234)
        tape = build()
        C._PHASE_TIMES.clear()
        torch.cuda.synchronize()
        t0 = time.time()
        tape.prove(seed=b"ab-spill")
        torch.cuda.synchronize()
        times.append(time.time() - t0)
        wit.append(dict(C._PHASE_TIMES).get("witness", 0.0))
        del tape
        torch.cuda.empty_cache()
    times.sort(); wit.sort()
    return times[len(times) // 2], wit[len(wit) // 2]


print(f"config d{D},ff{DFF},seq{SEQ},L{NL}  (median of {REPS} reps each)\n")
rc_s, rc_w = measure(False, False)
gc_s, gc_w = measure(False, True)
sp_s, sp_w = measure(True, False)
print(f"  recompute : prove {rc_s:6.2f}s   witness {rc_w:6.2f}s")
print(f"  gpu-cache : prove {gc_s:6.2f}s   witness {gc_w:6.2f}s   "
      f"({100*(rc_s-gc_s)/rc_s:+.1f}% vs recompute)")
print(f"  host-spill: prove {sp_s:6.2f}s   witness {sp_w:6.2f}s   "
      f"({100*(rc_s-sp_s)/rc_s:+.1f}% vs recompute)")
print(f"\n  spill vs recompute: witness {rc_w:.2f}s -> {sp_w:.2f}s "
      f"({100*(rc_w-sp_w)/max(rc_w,1e-9):+.0f}% of the witness term)")
print(f"  spill vs gpu-cache: prove {gc_s:.2f}s vs {sp_s:.2f}s "
      f"(spill trades GPU mem for host mem; {'slower' if sp_s>gc_s else 'faster'} by "
      f"{abs(sp_s-gc_s):.2f}s here)")

# --- production lens: never let the toy % stand alone (see prod_lens.py) ---
# Spill is a CROSSOVER-type lever: its per-term effect is NOT scale-invariant
# (toy cut != prod cut), so the toy witness fraction does NOT transfer. The
# production per-term reduction comes from the recompute-vs-reread crossover at
# 400B (spill_costmodel_prod.py), computed here from the same lpd constants.
import ligero_param_derivation as lpd
_T = lpd.T_WIT_S; _WB = lpd.W_of(1093) * 8
_eff_bw = min(7.0, 11.0) * 1e9              # NVMe 7 GB/s, capped by PCIe ~11
_io = _WB / _eff_bw
_wit_raw = 4 * _T                            # 4 recompute passes
_wit_spill = _T + 4 * _io                    # 1 compute + 1 write + 3 reads
_prod_frac = (_wit_raw - _wit_spill) / _wit_raw
toy_pct = 100 * (rc_s - sp_s) / rc_s
prod_lens.report("witness spill", toy_pct,
                 prod_lens.LeverEffect(term="witness", toy_frac=_prod_frac,
                                       note="(NVMe 7GB/s; crossover-type — prod "
                                            "per-term cut, NOT the toy fraction)"))

log_run(kind="witness_spill_ab", label=f"d{D},ff{DFF},seq{SEQ},L{NL}",
        params=dict(ELL=CFG.ELL, K_DEG=CFG.K_DEG, N_LIG=CFG.N_LIG, T_QUERIES=CFG.T_QUERIES,
                    d=D, d_ff=DFF, d_h=DH, SEQ=SEQ, num_layers=NL, reps=REPS),
        measured=dict(recompute_s=rc_s, gpu_cache_s=gc_s, spill_s=sp_s,
                       witness_recompute_s=rc_w, witness_gpu_cache_s=gc_w,
                       witness_spill_s=sp_w,
                       spill_vs_recompute_pct=100*(rc_s-sp_s)/rc_s),
        notes="host-memory witness spill vs recompute vs gpu-cache. Proof "
              "byte-identical (validate_witness_spill.py) + Rust ACCEPT (accept_toy_spill.py).")
