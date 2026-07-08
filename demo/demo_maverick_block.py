"""Full Llama-4-Maverick transformer BLOCK (attention + MoE FFN) — M2 demo.

Every operation type: RMSNorm (×2, with per-channel gains), GQA attention
(40 q / 8 kv heads — KV weight columns replicated 5× in the loader, a public
deterministic transform, so no new claims), RoPE at θ from the file
(NoPE layers skip it), causal softmax, the full MoE FFN of demo_maverick_moe
(router → RoutingClaim → input-side sigmoid → all-E expert matmuls), with the
three big combines on the Freivalds-projected seam (§B.4) so T=1000 × E=128
stays ~10⁶ combine slots instead of ~10¹².

The routing mask is derived inside the engine (argmax of the tiebroken
logits at witness-generation time), so there is no build-time hint and no
pre-pass — the builder composes claims; the engine computes values.

Run (Spark):
    PATH=~/venv-hf/bin:$PATH LIGERO_T_QUERIES=4 python demo_maverick_block.py \
        --experts 128 --seq 1000 --from-gguf ~/maverick-gguf/UD-Q4_K_XL \
        --dump-proof /tmp/maverick_block.json
"""
import argparse
import json
import math
import os
import sys
import pathlib
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch

import sys as _s, pathlib as _pl
_R = _pl.Path(__file__).resolve().parents[1]
_s.path.insert(0, str(_R / "prover")); _s.path.insert(0, str(_R / "demo"))
import _uint64_compat  # noqa: F401

import core
from core import P, LigeroConfig
import claims as _C          # noqa: F401
import packets as _PK         # noqa: F401
from tape import Tape
from claims import SiluConfig
from routing_claim import route_top1, freivalds_combine
from max_claim import to_signed
from demo_maverick_moe import (_to_field, _rand_int, _sigmoid_table, CFG, S,
                                SCALE_BITS, OUTPUT_WIDTH, SILU_CFG, SIG_SHIFT,
                                WORD_BITS, HALF_X)

SEED = b"maverick-block-demo"
EPS_INT = round(1e-5 * S * S)                 # rms_norm_eps from the GGUF metadata
Z_NONZERO_REAL = 40000 / 4096
Z_MAX = round(Z_NONZERO_REAL * S)
RMS_SLACK_N_CHUNKS = 4
H, HKV, DH = 40, 8, 128


def _unpermute_rows(w, n_head):
    """Inverse of convert_hf_to_gguf's q/k permute, as an exact row reorder on
    the FIELD tensor (int64 bit-view index_select — no arithmetic)."""
    d_out = w.shape[0]
    half = d_out // n_head // 2
    idx = torch.arange(d_out).view(n_head, 2, half).transpose(1, 2).reshape(-1)
    inv = torch.argsort(idx).to(w.device)
    return w.view(torch.int64).index_select(0, inv).view(torch.uint64)


def _replicate_kv_cols(w_t, groups=H // HKV):
    """(d, HKV·DH) → (d, H·DH): repeat each KV head's columns `groups`×
    (public weight transform — GQA with zero new claims)."""
    d = w_t.shape[0]
    v = w_t.view(torch.int64).view(d, HKV, DH)
    return (v.repeat_interleave(groups, dim=1).contiguous()
            .view(d, H * DH).view(torch.uint64))


def load_attention(gguf_path, layer, S_=S):
    from loader import _gguf_by_name, quantize_to_field
    from kquant_cuda import kquant_to_field
    from gguf.quants import dequantize
    import numpy as np
    by = _gguf_by_name(gguf_path)

    def field(name, divide_by=1.0):
        t = by[name]
        qt = t.tensor_type.name
        if qt in ("Q4_K", "Q5_K", "Q6_K") and divide_by == 1.0:
            d_out = int(t.data.shape[0])
            w = kquant_to_field(torch.from_numpy(
                np.ascontiguousarray(t.data)).cuda(), qt, S_)
            return w.view(d_out, -1)
        d = dequantize(np.ascontiguousarray(t.data), t.tensor_type)
        return quantize_to_field(torch.from_numpy(d.copy()), S_,
                                 divide_by=divide_by).view(d.shape[0], -1)

    def T_(w):     # (d_out, d_in) → (d_in, d_out), exact bit-view transpose
        return (w.view(torch.int64).T.contiguous().view(torch.uint64))

    p = f"blk.{layer}."
    wq = T_(_unpermute_rows(field(p + "attn_q.weight",
                                   divide_by=math.sqrt(DH)), H))
    wk = _replicate_kv_cols(T_(_unpermute_rows(field(p + "attn_k.weight"), HKV)))
    wv = _replicate_kv_cols(T_(field(p + "attn_v.weight")))
    wo = T_(field(p + "attn_output.weight"))
    g_attn = field(p + "attn_norm.weight").reshape(-1)
    g_ffn = field(p + "ffn_norm.weight").reshape(-1)
    return dict(W_Q=wq, W_K=wk, W_V=wv, W_O=wo,
                g_attn=g_attn, g_ffn=g_ffn)


def build_attn_chain(tape, x, attn, *, T, d, theta, use_rope):
    """x → norm·gain → GQA attention → resid → norm·gain. Returns
    (resid1, norm2_g)."""
    mm = dict(s_a=S, s_b=S, s_out=S, output_width=OUTPUT_WIDTH)
    n1 = tape.rmsnorm(x, d=d, s=S, eps_int=EPS_INT,
                       slack_n_chunks=RMS_SLACK_N_CHUNKS,
                       s_out=S, output_width=OUTPUT_WIDTH)
    n1g = tape.hadamard_broadcast(n1, attn["g_attn_wt"], SEQ=T, d=d,
                                   s_a=S, s_b=S, s_out=S,
                                   output_width=OUTPUT_WIDTH)
    q = tape.matmul(n1g, attn["W_Q_wt"], **mm)
    k = tape.matmul(n1g, attn["W_K_wt"], **mm)
    v = tape.matmul(n1g, attn["W_V_wt"], **mm)
    if use_rope:
        q = tape.rope(q, SEQ=T, d_h=DH, heads=H, base=theta,
                       s_x=S, s_out=S, output_width=OUTPUT_WIDTH)
        k = tape.rope(k, SEQ=T, d_h=DH, heads=H, base=theta,
                       s_x=S, s_out=S, output_width=OUTPUT_WIDTH)
    sc = tape.matmul(q, k, transpose_b=True, heads=H, head_dim=DH, **mm)
    sm = tape.softmax(sc, M=T, s_x=S, s_c=S, s_y=S, Z_max=Z_MAX,
                       saturate=True, Z_high_width=16, aux_chunk_width=24,
                       causal=True, heads=H)
    att = tape.matmul(sm, v, heads=H, head_dim=T, **mm)
    proj = tape.matmul(att, attn["W_O_wt"], **mm)
    resid1 = x + proj
    n2 = tape.rmsnorm(resid1, d=d, s=S, eps_int=EPS_INT,
                       slack_n_chunks=RMS_SLACK_N_CHUNKS,
                       s_out=S, output_width=OUTPUT_WIDTH)
    n2g = tape.hadamard_broadcast(n2, attn["g_ffn_wt"], SEQ=T, d=d,
                                   s_a=S, s_b=S, s_out=S,
                                   output_width=OUTPUT_WIDTH)
    return resid1, n2g


def build_moe_ffn(tape, n2g, m_args, *, T, E, d, d_ff, real):
    """The MoE FFN of demo_maverick_moe, with Freivalds-projected combines."""
    mm = dict(s_a=S, s_b=S, s_out=S, output_width=OUTPUT_WIDTH)

    def wcommit(key, name, shape, idx=None):
        if idx is not None:
            from loader import maverick_lazy_expert
            ld = maverick_lazy_expert(real["_gguf"], real["_layer"],
                                       {"W_gate": "gate_exps", "W_up": "up_exps",
                                        "W_down": "down_exps"}[key], idx, S)
            return tape.commit_lazy(name, ld, shape, shape[0] * shape[1])
        return tape.commit(name, real[key].contiguous().reshape(-1), shape)

    W_router = wcommit("W_router", "W_router", (d, E))
    r = tape.matmul(n2g, W_router, **mm)
    m, r_chosen, _gap = route_top1(tape, r, T=T, E=E, B_logit=OUTPUT_WIDTH,
                                    word_bits=WORD_BITS)
    t_in, t_out = _sigmoid_table()
    sig_tbl = tape.register_table("sigmoid", T_data=t_in, T_Y_data=t_out)
    s_val = tape.paired_tlookup(r_chosen, sig_tbl, shift=SIG_SHIFT)
    ones = tape.commit("bc_ones", torch.ones(T * d, dtype=torch.uint64,
                                              device="cuda"), (T, d))
    s_rep = freivalds_combine(tape, s_val, [ones], T=T, E=1, F=d)
    x_r = tape.hadamard(s_rep, n2g, **mm)

    W_gate = [wcommit("W_gate", f"W_gate{e}", (d, d_ff), idx=e) for e in range(E)]
    W_up = [wcommit("W_up", f"W_up{e}", (d, d_ff), idx=e) for e in range(E)]
    gates = [tape.matmul(x_r, W_gate[e], **mm) for e in range(E)]
    ups = [tape.matmul(x_r, W_up[e], **mm) for e in range(E)]
    g_sum = freivalds_combine(tape, m, gates, T=T, E=E, F=d_ff)
    up_sum = freivalds_combine(tape, m, ups, T=T, E=E, F=d_ff)
    silu_g = tape.silu(g_sum)
    hidden = tape.hadamard(silu_g, up_sum, **mm)
    W_down = [wcommit("W_down", f"W_down{e}", (d_ff, d), idx=e) for e in range(E)]
    outs = [tape.matmul(hidden, W_down[e], **mm) for e in range(E)]
    ffn = freivalds_combine(tape, m, outs, T=T, E=E, F=d)

    Wg_s = wcommit("W_gate_sh", "W_gate_sh", (d, d_ff))
    Wu_s = wcommit("W_up_sh", "W_up_sh", (d, d_ff))
    Wd_s = wcommit("W_down_sh", "W_down_sh", (d_ff, d))
    g_s = tape.matmul(n2g, Wg_s, **mm)
    u_s = tape.matmul(n2g, Wu_s, **mm)
    h_s = tape.hadamard(tape.silu(g_s), u_s, **mm)
    sh = tape.matmul(h_s, Wd_s, **mm)
    return ffn + sh


def build(tape, *, T, E, d, d_ff, real, attn_w, theta, use_rope, x_data):
    x = tape.commit("x", x_data, (T, d))
    attn = dict(attn_w)
    attn["g_attn_wt"] = tape.commit("g_attn", attn_w["g_attn"], (d,))
    attn["g_ffn_wt"] = tape.commit("g_ffn", attn_w["g_ffn"], (d,))
    attn["W_Q_wt"] = tape.commit("W_Q", attn_w["W_Q"].reshape(-1), (d, H * DH))
    attn["W_K_wt"] = tape.commit("W_K", attn_w["W_K"].reshape(-1), (d, H * DH))
    attn["W_V_wt"] = tape.commit("W_V", attn_w["W_V"].reshape(-1), (d, H * DH))
    attn["W_O_wt"] = tape.commit("W_O", attn_w["W_O"].reshape(-1), (H * DH, d))
    resid1, n2g = build_attn_chain(tape, x, attn, T=T, d=d, theta=theta,
                                    use_rope=use_rope)
    moe = build_moe_ffn(tape, n2g, None, T=T, E=E, d=d, d_ff=d_ff, real=real)
    return resid1 + moe, n2g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", type=int, default=128)
    ap.add_argument("--seq", type=int, default=1000)
    ap.add_argument("--d", type=int, default=5120)
    ap.add_argument("--d-ff", type=int, default=8192)
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--from-gguf", type=str, required=True)
    ap.add_argument("--dump-proof", type=str, default=None)
    a = ap.parse_args()
    torch.manual_seed(7)

    from loader import load_maverick_moe_layer, _gguf_by_name
    theta = 500000.0
    use_rope = (a.layer + 1) % 4 != 0
    print(f"[maverick-block] E={a.experts} T={a.seq} d={a.d} d_ff={a.d_ff} "
          f"layer={a.layer} rope={use_rope} θ={theta} "
          f"T_QUERIES={CFG.T_QUERIES}", flush=True)
    real = load_maverick_moe_layer(a.from_gguf, a.layer, S=S,
                                    n_experts=a.experts, skip_experts=True)
    real["_gguf"], real["_layer"] = a.from_gguf, a.layer
    attn_w = load_attention(a.from_gguf, a.layer)
    x_data = _to_field(_rand_int(a.seq * a.d, half=HALF_X))

    tape = Tape(CFG, silu_config=SILU_CFG, lazy=True)
    y, _ = build(tape, T=a.seq, E=a.experts, d=a.d, d_ff=a.d_ff, real=real,
                 attn_w=attn_w, theta=theta, use_rope=use_rope, x_data=x_data)
    print(f"[maverick-block] {len(tape.claims)} claims recorded", flush=True)
    t0 = time.time()
    proof = tape.prove(seed=SEED)
    t_prove = time.time() - t0
    print(f"[maverick-block] prove returned ({t_prove:.1f}s)", flush=True)
    # Verification is the standalone Rust verifier's job (run verify_proof on the
    # dump); the demo only proves + dumps.
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"[maverick-block] prove={t_prove:.1f}s peakGPU={peak:.2f}GB", flush=True)

    if a.dump_proof:
        import protocol as pr
        from proof_dump import dump_proof
        s_op, s_comb, s_col = pr.round_seeds(SEED)
        Q = pr.random_columns(s_col, CFG)
        t0 = time.time()
        dump_proof(a.dump_proof, pr.claims_to_json(tape.claims, CFG),
                   {"s_op": s_op.hex(), "s_comb": s_comb.hex(),
                    "s_col": s_col.hex()},
                   proof, list(Q), None)
        print(f"[maverick-block] proof dumped to {a.dump_proof} "
              f"({time.time()-t0:.1f}s, streaming)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
