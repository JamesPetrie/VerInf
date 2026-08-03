"""Run the REAL Llama-4-Maverick MoE layer (synthetic random weights, NO gguf)
under BASELINE (rho=4, T=40) vs OPTIMIZED (rho=2, T=17, + witness SPILL), and
Rust-verify each. Exercises the ACTUAL 400B claim types — router/RoutingClaim,
top-1 mask, sigmoid lookup, per-expert gate/up/down MatmulClaims, FreivaldsCombine
x3, shared SwiGLU, AddClaim — not a dense toy. Scale via env MOE_E/SEQ/D/DFF.
"""
import sys, os, time, gc
from pathlib import Path
R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R / "prover")); sys.path.insert(0, str(R / "prover/tests"))
sys.path.insert(0, str(R / "demo"))
import _uint64_compat  # noqa
import torch
import core as C
from core import LigeroConfig
from tape import Tape
from _rust_verify import rust_verify_tape
import demo_maverick_moe as mav

E   = int(os.environ.get("MOE_E", "8"))
SEQ = int(os.environ.get("MOE_SEQ", "4"))
D   = int(os.environ.get("MOE_D", "2048"))
DFF = int(os.environ.get("MOE_DFF", "4096"))
ELL, K = 8192, 16384                      # real 400B Ligero geometry
_t0 = time.time()
def lg(m): print(f"[optrun-moe +{time.time()-_t0:7.1f}s] {m}", flush=True)


DISK = os.environ.get("MOE_DISK", "0") == "1"        # optimized run uses DISK spill


VERIFY = os.environ.get("MOE_VERIFY", "both")        # both | opt | none (verify cost control)


def run(label, rho, T, spill, verify=True):
    cfg = LigeroConfig(ELL=ELL, K_DEG=K, N_LIG=rho * K, T_QUERIES=T)
    mode = ("disk" if DISK else "host") if spill else "off"
    lg(f"===== {label}: rho={rho} T_QUERIES={T} spill={mode} "
       f"| N_LIG={rho*K} (pad={K-ELL}) =====")
    C._WITNESS_SPILL_DISK = bool(spill and DISK)
    C._WITNESS_SPILL_ON = bool(spill and not DISK)
    torch.manual_seed(7)
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    tape = Tape(cfg, silu_config=mav.SILU_CFG, lazy=True)
    mav.build(tape, T=SEQ, E=E, d=D, d_ff=DFF, real=None)
    lg(f"{label}: {len(tape.claims)} claims recorded")
    t = time.time(); proof = tape.prove(seed=mav.SEED); pt = time.time() - t
    peak = torch.cuda.max_memory_allocated() / 2**30
    lg(f"{label}: PROVE {pt:.1f}s  peak {peak:.2f} GB")
    acc, vt = None, 0.0
    if verify:
        t = time.time(); acc, msg = rust_verify_tape(tape, proof, seed=mav.SEED); vt = time.time() - t
        lg(f"{label}: Rust verify_proof {vt:.1f}s  ->  {'ACCEPT' if acc else 'REJECT'}")
        if not acc:
            lg(f"{label}: verifier says: {str(msg)[:500]}")
    else:
        lg(f"{label}: verify skipped (cost control)")
    C._WITNESS_SPILL_ON = False
    C._WITNESS_SPILL_DISK = False
    return dict(label=label, rho=rho, T=T, spill=spill, prove_s=round(pt, 2),
                verify_s=round(vt, 2), accept=acc, peak_gb=round(peak, 3),
                claims=len(tape.claims))


lg(f"REAL Maverick MoE layer (synthetic weights): E={E} seq={SEQ} d={D} d_ff={DFF} "
   f"| ELL={ELL} K={K}")
lg(f"GPU: {torch.cuda.get_device_name(0)}")
def _vd(name): return VERIFY == "both" or (VERIFY == "opt" and name == "opt")
res = [run("BASELINE  rho=4 T=40", 4, 40, False, verify=_vd("base")),
       run("OPTIMIZED rho=2 T=17 +SPILL", 2, 17, True, verify=_vd("opt"))]

def _av(a): return "ACCEPT" if a else ("REJECT ***" if a is False else "verify-skipped")
print("\n=== SUMMARY: REAL Maverick MoE architecture (synthetic weights) ===", flush=True)
for r in res:
    print(f"  {r['label']:28s}: prove {r['prove_s']:7.1f}s · verify {r['verify_s']:6.1f}s · "
          f"peak {r['peak_gb']:.2f}GB · claims {r['claims']} · {_av(r['accept'])}")
b, o = res
print(f"\n  rho=2 ACCEPT on REAL MoE claims?  {_av(o['accept'])}")
print(f"  prove delta (opt+spill vs base):  {o['prove_s']-b['prove_s']:+.1f}s "
      f"({100*(o['prove_s']-b['prove_s'])/max(b['prove_s'],1e-9):+.0f}%)")
# pass if nothing REJECTED (skipped verify is not a failure)
sys.exit(1 if (b['accept'] is False or o['accept'] is False) else 0)
