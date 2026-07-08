"""Single Llama-4-Maverick MoE layer, few tokens — the M1 keystone demo.

Builds the full MoE FFN layer as claims and proves it end to end:

    x ─┬─ router matmul ─ RoutingClaim (top-1 mask m, r_chosen) ─ gap range
       │                   └ sigmoid paired lookup s = σ(r_chosen)
       ├─ x_r = s ⊙ x      (input-side routing weight, HF/llama.cpp semantics)
       ├─ gate_e/up_e = x_r @ W_{gate,up}[e]  ∀e   (2E MatmulClaim)
       ├─ g_sum/up_sum = Σ_e m_e·{gate,up}_e        (FreivaldsCombineClaim ×2)
       ├─ hidden = silu(g_sum) ⊙ up_sum             (sum-before-nonlinearity, §B.7.6)
       ├─ out_e = hidden @ W_down[e]  ∀e            (E MatmulClaim)
       ├─ ffn  = Σ_e m_e·out_e                       (FreivaldsCombineClaim)
       ├─ shared SwiGLU on the unscaled x            (3 MatmulClaim + silu + ⊙)
       └─ y = ffn + shared                           (AddClaim)

Weights are SYNTHETIC (random ints, demo_llama7b conventions) until the
UD-Q4_K_XL GGUF loader lands (M1 plan Phase 0); dims default to the real
Maverick MoE layer (d=5120, d_ff=8192), experts default to the E=8 dev mode.

Run (Spark):
    PATH=~/venv-hf/bin:$PATH python demo_maverick_moe.py \
        --experts 8 --seq 4 --dump-proof /tmp/maverick_moe_proof.json
    ../verifier-rs/target/release/verify_proof /tmp/maverick_moe_proof.json

Smoke test (small dims, fast):
    LIGERO_T_QUERIES=4 python demo_maverick_moe.py --experts 4 --seq 2 \
        --d 256 --d-ff 512
"""
import argparse
import json
import math
import os
import time

import sys
import pathlib
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch

import sys as _s, pathlib as _pl
_R = _pl.Path(__file__).resolve().parents[1]
_s.path.insert(0, str(_R / "prover")); _s.path.insert(0, str(_R / "demo"))
import _uint64_compat  # noqa: F401 — patch uint64 CUDA op gaps before any prover op

import core
from core import P, LigeroConfig
import claims as _C          # noqa: F401
import packets as _PK         # noqa: F401
from tape import Tape
from claims import SiluConfig
from routing_claim import route_top1, freivalds_combine
from cuda_primitives import gl_sub

CFG = LigeroConfig(ELL=8192, K_DEG=16384, N_LIG=65536,
                   T_QUERIES=int(os.environ.get("LIGERO_T_QUERIES", "80")))
SEED = b"maverick-moe-demo"

# Scale cascade — identical to demo_llama7b (one knob, derived constants).
SCALE_BITS = 12
S          = 1 << SCALE_BITS
OUTPUT_WIDTH = 26                    # matmul/hadamard rescale range width
SILU_CFG   = SiluConfig(b=4, T_LEN=1 << 14, b_2=1 << 16, b_3=1 << 32, b_4=1 << 48,
                        width_2=16, width_3=16, width_4=14, r=SCALE_BITS)
SIG_BITS   = 19                      # sigmoid table: r_chosen + 2^18 ∈ [0, 2^19), ±64 real
                                     # units at S=2^12. Full-model reference measured
                                     # max|r| = 11.1 (layer 47), so ±16 was tight; the
                                     # table costs ~2 slots/entry → ±64 is ~1M witness
                                     # slots, ~0.0003% of the full-model witness. Free.
SIG_SHIFT  = 1 << (SIG_BITS - 1)
WORD_BITS  = 11                      # gap range word width (shared 2^11 table)
HALF_W, HALF_X = 8, 4                # synthetic weight / activation magnitudes


def _to_field(v):
    """Signed int64 → uint64 Goldilocks rep (P − |v| for v < 0)."""
    v_abs = v.abs().to(torch.uint64)
    neg = gl_sub(torch.full_like(v_abs, P), v_abs)
    return torch.where(v >= 0, v, neg.view(torch.int64)).view(torch.uint64)


def _rand_int(*shape, half):
    return torch.randint(-half, half, shape, dtype=torch.int64, device="cuda")


def _sigmoid_table():
    """Paired table (k, round(sigmoid((k − SIG_SHIFT)/S)·S)) for k ∈ [0, 2^SIG_BITS)."""
    k = torch.arange(1 << SIG_BITS, dtype=torch.float64)
    y = torch.sigmoid((k - SIG_SHIFT) / S) * S
    return list(range(1 << SIG_BITS)), [int(v) for v in y.round().to(torch.int64)]


def build(tape, *, T, E, d, d_ff, real=None):
    """real: field-tensor dict from loader.load_maverick_moe_layer (UD-Q4_K_XL
    blk weights, first E experts); None -> synthetic random weights."""
    mm = dict(s_a=S, s_b=S, s_out=S, output_width=OUTPUT_WIDTH)

    _EXP_KEY = {"W_gate": "gate_exps", "W_up": "up_exps", "W_down": "down_exps"}

    def wcommit(key, name, shape, idx=None):
        if real is not None:
            if idx is not None:
                # Per-expert matrix: LAZY commit — dequantized on demand per
                # sweep, freed after. All 128 experts eager would be ~163 GB
                # (63 fp32 + 100 field) > the Spark's 121 GB unified pool.
                from loader import maverick_lazy_expert
                ld = maverick_lazy_expert(real["_gguf"], real["_layer"],
                                           _EXP_KEY[key], idx, S)
                return tape.commit_lazy(name, ld, shape, shape[0] * shape[1])
            return tape.commit(name, real[key].contiguous().reshape(-1), shape)
        n = shape[0] * shape[1]
        return tape.commit(name, _to_field(_rand_int(n, half=HALF_W)), shape)

    x_i = _rand_int(T * d, half=HALF_X)
    x = tape.commit("x", _to_field(x_i), (T, d))

    # Router + routing decision (mask hidden; gap range composed inside).
    # The mask is engine-derived from r — no build-time hint needed.
    W_router = wcommit("W_router", "W_router", (d, E))
    r = tape.matmul(x, W_router, **mm)
    m, r_chosen, _gap = route_top1(tape, r, T=T, E=E,
                                    B_logit=OUTPUT_WIDTH, word_bits=WORD_BITS)

    # Input-side sigmoid weight: s = σ(r_chosen); x_r = s ⊙ x.
    t_in, t_out = _sigmoid_table()
    sig_tbl = tape.register_table("sigmoid", T_data=t_in, T_Y_data=t_out)
    s_val = tape.paired_tlookup(r_chosen, sig_tbl, shift=SIG_SHIFT)
    ones = tape.commit("bc_ones", torch.ones(T * d, dtype=torch.uint64, device="cuda"),
                        (T, d))
    s_rep = freivalds_combine(tape, s_val, [ones], T=T, E=1, F=d)   # broadcast pin
    x_r = tape.hadamard(s_rep, x, **mm)

    # All-E expert matmuls on the same committed x_r (privacy: §B.4).
    W_gate = [wcommit("W_gate", f"W_gate{e}", (d, d_ff), idx=e) for e in range(E)]
    W_up   = [wcommit("W_up",   f"W_up{e}",   (d, d_ff), idx=e) for e in range(E)]
    gates = [tape.matmul(x_r, W_gate[e], **mm) for e in range(E)]
    ups   = [tape.matmul(x_r, W_up[e],   **mm) for e in range(E)]

    # Sum-before-nonlinearity (top-1, §B.7.6): one silu/hadamard stream.
    g_sum  = freivalds_combine(tape, m, gates, T=T, E=E, F=d_ff)
    up_sum = freivalds_combine(tape, m, ups,   T=T, E=E, F=d_ff)
    silu_g = tape.silu(g_sum)
    hidden = tape.hadamard(silu_g, up_sum, **mm)

    W_down = [wcommit("W_down", f"W_down{e}", (d_ff, d), idx=e) for e in range(E)]
    outs = [tape.matmul(hidden, W_down[e], **mm) for e in range(E)]
    ffn = freivalds_combine(tape, m, outs, T=T, E=E, F=d)

    # Shared expert: SwiGLU on the UNSCALED x (HF Llama4TextMoe semantics).
    Wg_s = wcommit("W_gate_sh", "W_gate_sh", (d, d_ff))
    Wu_s = wcommit("W_up_sh",   "W_up_sh",   (d, d_ff))
    Wd_s = wcommit("W_down_sh", "W_down_sh", (d_ff, d))
    g_s  = tape.matmul(x, Wg_s, **mm)
    u_s  = tape.matmul(x, Wu_s, **mm)
    h_s  = tape.hadamard(tape.silu(g_s), u_s, **mm)
    sh   = tape.matmul(h_s, Wd_s, **mm)

    return ffn + sh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--seq", type=int, default=4)
    ap.add_argument("--d", type=int, default=5120)
    ap.add_argument("--d-ff", type=int, default=8192)
    ap.add_argument("--dump-proof", type=str, default=None)
    ap.add_argument("--from-gguf", type=str, default=None,
                    help="UD-Q4_K_XL file/dir: REAL blk weights (first --experts)")
    ap.add_argument("--layer", type=int, default=1)
    a = ap.parse_args()
    torch.manual_seed(7)

    print(f"[maverick-moe] E={a.experts} T={a.seq} d={a.d} d_ff={a.d_ff} "
          f"T_QUERIES={CFG.T_QUERIES}")
    real = None
    if a.from_gguf:
        from loader import load_maverick_moe_layer
        real = load_maverick_moe_layer(a.from_gguf, a.layer, S=S,
                                        n_experts=a.experts, skip_experts=True)
        real["_gguf"], real["_layer"] = a.from_gguf, a.layer
    tape = Tape(CFG, silu_config=SILU_CFG, lazy=True)
    y = build(tape, T=a.seq, E=a.experts, d=a.d, d_ff=a.d_ff, real=real)
    print(f"[maverick-moe] {len(tape.claims)} claims recorded")

    t0 = time.time()
    proof = tape.prove(seed=SEED)
    t_prove = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"[maverick-moe] prove={t_prove:.1f}s peakGPU={peak:.2f}GB "
          f"(verify with verifier-rs/.../verify_proof on the dump)")

    if a.dump_proof:
        import protocol as pr
        from proof_dump import dump_proof   # single block-driven writer
        s_op, s_comb, s_col = pr.round_seeds(SEED)
        Q = list(pr.random_columns(s_col, CFG))
        dump_proof(a.dump_proof, pr.claims_to_json(tape.claims, CFG),
                   {"s_op": s_op.hex(), "s_comb": s_comb.hex(), "s_col": s_col.hex()},
                   proof, Q, None)
        print(f"[maverick-moe] proof dumped to {a.dump_proof}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
