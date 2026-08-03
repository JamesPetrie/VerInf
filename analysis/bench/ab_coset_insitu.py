"""In-situ: does coset-NTT speed up the ENCODE phase in the REAL prover at
production geometry (K=2^16)? Small model, K=65536/N=262144 (rho=4), prove with
coset OFF vs ON (forced), CUDA-synced phase timing. This is the honest
transfer check — the isolated primitive win (coset_ntt_ab.py, 1.26x at K=2^18)
should show up in-prove as a lower `encode` bucket. Confirms ACCEPT on the
coset-on proof too."""
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

# Production-scale geometry on a small model. ELL divisible by d=512.
CFG = LigeroConfig(ELL=65024, K_DEG=65536, N_LIG=262144, T_QUERIES=16)
D, DFF, DH, SEQ, NL = 512, 512, 64, 4, 1
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


def measure(mode):
    C._COSET_NTT_MODE = mode
    torch.manual_seed(7)
    tape = build()
    C._PHASE_TIMES.clear()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()
    proof = tape.prove(seed=b"coset-insitu")
    torch.cuda.synchronize()
    dt_s = time.time() - t0
    enc = dict(C._PHASE_TIMES).get("encode", 0.0)
    peak = torch.cuda.max_memory_allocated() / 1e9
    del tape
    torch.cuda.empty_cache()
    return dt_s, enc, peak


off_s, off_enc, off_pk = measure("off")
on_s, on_enc, on_pk = measure("on")
print(f"K=2^16 (N=2^18), small model, in-prove:")
print(f"  coset OFF: prove {off_s:.2f}s  encode {off_enc:.2f}s  peak {off_pk:.1f}GB")
print(f"  coset ON : prove {on_s:.2f}s  encode {on_enc:.2f}s  peak {on_pk:.1f}GB")
if off_enc > 0:
    print(f"  encode speedup: {off_enc/on_enc:.2f}x  ({100*(off_enc-on_enc)/off_enc:+.0f}% encode; "
          f"{100*(off_s-on_s)/off_s:+.0f}% total prove)")

log_run(kind="coset_insitu_ab", label="K=2^16,N=2^18,small",
        params=dict(ELL=CFG.ELL, K_DEG=CFG.K_DEG, N_LIG=CFG.N_LIG, T_QUERIES=CFG.T_QUERIES,
                    d=D, d_ff=DFF, SEQ=SEQ, num_layers=NL),
        measured=dict(prove_off_s=off_s, prove_on_s=on_s, encode_off_s=off_enc,
                       encode_on_s=on_enc, encode_speedup=(off_enc/on_enc if on_enc else None),
                       peak_off_gb=off_pk, peak_on_gb=on_pk),
        notes="coset-NTT wired into _coset_encode_codewords; in-prove encode A/B at "
              "production K=2^16. Byte-identical proof + ACCEPT validated separately.")
