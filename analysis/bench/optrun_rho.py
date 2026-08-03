"""FULL optimization-validation run: prove the phone model under the OPTIMIZED
Ligero config (rho=2, T_QUERIES=17) vs the BASELINE (rho=4, T_QUERIES=40), same
model + seed, and Rust-verify each. Answers the single open question behind the
~4.4h 400B projection: does rho=2 produce a proof the standalone Rust verifier
ACCEPTs (i.e. is the (1-1/rho)^T soundness legal at rho=2 in practice), and does
the prove floor move in the predicted direction? Rich per-phase logging.

Shape from env (AB_D/AB_DFF/AB_DH/AB_SEQ/AB_NL/AB_VOCAB); defaults = Qwen-0.5B.
"""
import sys, os, time, gc, json
from pathlib import Path
R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R / "prover")); sys.path.insert(0, str(R / "prover/tests"))
sys.path.insert(0, str(R / "demo")); sys.path.insert(0, str(R / "analysis/bench"))
os.environ["LIGERO_PHASE_TIMING"] = "1"          # per-phase prints from prove()
import _uint64_compat  # noqa
import torch
import demo_toy_transformer as dt
import core as C
from core import LigeroConfig
from tape import Tape
from _rust_verify import rust_verify_tape

_t0 = time.time()
def lg(m): print(f"[optrun +{time.time()-_t0:7.1f}s] {m}", flush=True)

D     = int(os.environ.get("AB_D", "896"))
DFF   = int(os.environ.get("AB_DFF", "4864"))
DH    = int(os.environ.get("AB_DH", "64"))
SEQ   = int(os.environ.get("AB_SEQ", "256"))
NL    = int(os.environ.get("AB_NL", "24"))
VOCAB = int(os.environ.get("AB_VOCAB", "151936"))
ELL   = 1 << max(D, 512).bit_length()            # >= d, power of two (K=2*ELL, ZK pad = ELL)
if ELL // 2 >= D and ELL // 2 >= 512:
    ELL //= 2
K     = 2 * ELL
SEED  = b"optrun-rho"
C._WITNESS_CACHE_ON = True


def build(cfg):
    dt.d, dt.d_ff, dt.d_h, dt.SEQ = D, DFF, DH, SEQ
    tape = Tape(cfg, silu_config=dt.SILU_CFG, lazy=True)
    x = dt._rand_signed(SEQ * D, half=dt.HALF_X)
    resid = tape.commit("x_input", x, (SEQ, D))
    for L in range(NL):
        w = dt._commit_weights_random(tape, layer_idx=L)
        resid = dt._run_block(tape, resid, w, H=D // DH)
    fn = torch.full((D,), dt.S, dtype=torch.uint64, device="cuda")
    lm = dt._rand_signed(D * VOCAB, half=dt.HALF)
    fnw = tape.commit("final_norm_w", fn, (D,))
    lmw = tape.commit("W_lm_head", lm, (D, VOCAB))
    dt._run_tail(tape, resid, fnw, lmw, vocab_size=VOCAB)
    return tape


def run(label, rho, T, spill=False):
    N = rho * K
    irs = T * (1 if rho == 2 else __import__("math").log2(rho / (rho - 1)))
    lg(f"===== {label}: rho={rho}  T_QUERIES={T}  spill={'ON' if spill else 'off'}  |  "
       f"ELL={ELL} K_DEG={K} N_LIG={N} (pad={K-ELL}, IRS~{irs:.1f} bit) =====")
    cfg = LigeroConfig(ELL=ELL, K_DEG=K, N_LIG=N, T_QUERIES=T)
    C._WITNESS_SPILL_ON = bool(spill)          # host-memory witness spill (re-read vs recompute)
    torch.manual_seed(1234)
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    t = time.time(); tape = build(cfg); bt = time.time() - t
    lg(f"{label}: build {bt:.1f}s")
    t = time.time(); proof = tape.prove(seed=SEED); pt = time.time() - t
    peak = torch.cuda.max_memory_allocated() / 1e9
    lg(f"{label}: PROVE {pt:.1f}s  peak {peak:.2f} GB")
    t = time.time(); acc, msg = rust_verify_tape(tape, proof, seed=SEED); vt = time.time() - t
    lg(f"{label}: Rust verify_proof {vt:.1f}s  ->  {'ACCEPT' if acc else 'REJECT'}")
    if not acc:
        lg(f"{label}: verifier says: {str(msg)[:500]}")
    C._WITNESS_SPILL_ON = False
    return dict(label=label, rho=rho, T=T, ELL=ELL, K=K, N=N, spill=bool(spill),
                prove_s=round(pt, 2), verify_s=round(vt, 2), accept=bool(acc),
                peak_gb=round(peak, 3))


lg(f"model: D={D} DFF={DFF} DH={DH} SEQ={SEQ} NL={NL} VOCAB={VOCAB}")
lg(f"geometry: ELL={ELL} (>= d={D}), K_DEG={K}, ZK pad={K-ELL}")
lg(f"GPU: {torch.cuda.get_device_name(0)}")
SPILL = os.environ.get("AB_SPILL", "0") == "1"
res = [run("BASELINE  rho=4 T=40", 4, 40, spill=False),
       run(f"OPTIMIZED rho=2 T=17{' +SPILL' if SPILL else ''}", 2, 17, spill=SPILL)]

print("\n=== SUMMARY (optrun: rho=2 optimization validation) ===", flush=True)
for r in res:
    print(f"  {r['label']:22s}: prove {r['prove_s']:8.1f}s · verify {r['verify_s']:7.1f}s · "
          f"peak {r['peak_gb']:.2f}GB · {'ACCEPT' if r['accept'] else 'REJECT ***'}")
b, o = res
dp = o['prove_s'] - b['prove_s']
print(f"\n  rho=2 legal (Rust ACCEPT)?  {'YES' if o['accept'] else 'NO — rho=2 REJECTED at this geometry'}")
print(f"  prove delta (opt vs base):  {dp:+.1f}s ({100*dp/max(b['prove_s'],1e-9):+.0f}%)")
print(f"  both ACCEPT?  {'YES — optimization is sound-mechanism-valid at phone scale' if (b['accept'] and o['accept']) else 'NO'}")
Path("optrun_results.json").write_text(json.dumps(res, indent=2))
sys.exit(0 if (b['accept'] and o['accept']) else 1)
