"""Full 48-layer Llama-4-Maverick proof with unexplained-information claim.

Input binding (analysis/full-model-v1-design.md, updated 2026-07-08):
  * ALL positions HIDDEN — committed one-hot indicators, RoutingClaim
    constrains each row one-hot (booleanity + cardinality hold regardless of
    the committed indicator), x = m @ E selects the rows. Token ids never
    appear as numbers; only the unexplained-information bound is public.
  * The indicator rows for positions 1..T-1 are SHARED with the UI output
    select (O = ind_mid ‖ o_last via ConcatClaim), so one committed token
    stream drives both the forward pass and the surprisal: the scored token
    at position t is provably the input token at position t+1.

Layers: even = dense FFN (d_ff 16384), odd = MoE (128 experts + shared,
Freivalds-projected combines); NoPE (no rope) when (il+1) % 4 == 0; attention
temperature tuning is exactly 1 below position 8192 (llama-graph.cpp) — not
modeled. Tail: final RMSNorm·gain → LM head → prove_unexplained_info summed
over the continuation positions.

Modes:
  --witness-only   run_engine_pass(free_intermediates) — no proof; prints the
                   REAL UI number + argmax-vs-continuation agreement and dumps
                   logits for the llama.cpp cross-check. (H100 safety run.)
  default          full streaming prove + streaming dump.   (Spark run.)

Run:
  PATH=~/venv-hf/bin:$PATH LIGERO_T_QUERIES=4 \\
  LIGERO_STREAM_DBG=1 python demo_maverick_full.py \\
      --from-gguf ~/maverick-gguf/UD-Q4_K_XL --tokens ~/inference-1k/tokens.json \\
      --dump-proof /tmp/maverick_full.json
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
from core import P
import claims as _C          # noqa: F401
import packets as _PK         # noqa: F401
from tape import Tape, WitnessTensor
from routing_claim import route_top1, freivalds_combine
from unexplained_info import prove_unexplained_info, bound_bits
from demo_maverick_moe import (_to_field, _rand_int, _sigmoid_table, CFG, S,
                                SCALE_BITS, OUTPUT_WIDTH, SILU_CFG, SIG_SHIFT,
                                WORD_BITS, HALF_X)
from demo_maverick_block import (load_attention, build_attn_chain, EPS_INT,
                                  H, HKV, DH)

SEED = b"maverick-full-demo"
# s_y = 2^18 > V: the proven EXP table floors every entry at 1 (soundness:
# LogUp entries nonzero, round-up keeps U an upper bound), so the partition
# carries a log2(V/s_y) floor — 5.62 bits/token at s_y=2^12 (measured: proven
# 5.87 vs float 0.60). At 2^18 the floor is ~0 and proven ~= float.
# s_y >> V (202048) so the LogUp exp-table floor log2(1+V/s_y) ~= 0.001
# bits is negligible (2^18 left ~0.8 bits of floor; 2^28 is clean).
UI = dict(s_c=1 << 28, s_y=1 << 28, s_b=1 << 12, gap_max=1 << 20)


def _log(msg):
    print(f"[maverick-full] {msg}", flush=True)


def _field_loader(gguf, name, S_=S, transpose=False):
    """Closure: GGUF tensor -> field ints, fused kernel when possible."""
    def load():
        from loader import _gguf_by_name, quantize_to_field
        from kquant_cuda import kquant_to_field
        from gguf.quants import dequantize
        import numpy as np
        t = _gguf_by_name(gguf)[name]
        qt = t.tensor_type.name
        if qt in ("Q4_K", "Q5_K", "Q6_K"):
            d_out = int(t.data.shape[0])
            w = kquant_to_field(torch.from_numpy(
                np.ascontiguousarray(t.data)).cuda(), qt, S_).view(d_out, -1)
        else:
            d = dequantize(np.ascontiguousarray(t.data), t.tensor_type)
            w = quantize_to_field(torch.from_numpy(d.copy()), S_).view(d.shape[0], -1)
        if transpose:
            w = w.view(torch.int64).T.contiguous().view(torch.uint64)
        return w.reshape(-1)
    return load


def build_inputs(tape, gguf, E_wt, prompt_ids, cont_ids, *, V, d):
    """ALL tokens hidden: committed one-hot indicators, never revealed,
    one-hot enforced by claim, x = m @ E. Returns (x, ind_mid, o_last).

    The indicator is committed in two pieces so its rows can be SHARED with
    the unexplained-information output select: `ind_mid` holds positions
    1..T-1, which are simultaneously the input tokens at those positions
    and (shifted by one) the scored output tokens at positions 0..T-2. The
    caller concats (ind_mid, o_last) into the UI's O, so ONE committed
    token stream provably drives both the forward pass and the surprisal."""
    ids = list(prompt_ids) + list(cont_ids)
    T = len(ids)

    def _onehot(rows_ids):
        n = len(rows_ids)
        ind = torch.zeros(n, V, dtype=torch.int64, device="cuda")
        ind[torch.arange(n), torch.tensor(rows_ids)] = 1
        return ind.reshape(-1).to(torch.uint64)

    ind0 = tape.commit("tok_ind0", _onehot(ids[:1]), (1, V))
    ind_mid = tape.commit("tok_ind_mid", _onehot(ids[1:]), (T - 1, V))
    # the UI's last row: targets[T-1] = ids[-1] (a repeat, excluded from the sum)
    o_last = tape.commit("tok_olast", _onehot(ids[-1:]), (1, V))

    r_ind = tape.concat([ind0, ind_mid], (T, V))
    m, _rc, _gap = route_top1(tape, r_ind, T=T, E=V, B_logit=1, word_bits=11)
    # exact select (one-hot row sum, no rescale): scale-free matmul claim
    x = tape.matmul(m, E_wt)
    return x, ind_mid, o_last


def build_dense_ffn(tape, n2g, gguf, il, *, T, d):
    mm = dict(s_a=S, s_b=S, s_out=S, output_width=OUTPUT_WIDTH)
    from loader import _gguf_by_name
    by = _gguf_by_name(gguf)
    d_ff = int(by[f"blk.{il}.ffn_gate.weight"].data.shape[0])
    Wg = tape.commit_lazy(f"L{il}_Wg", _field_loader(gguf, f"blk.{il}.ffn_gate.weight",
                                                      transpose=True), (d, d_ff), d * d_ff)
    Wu = tape.commit_lazy(f"L{il}_Wu", _field_loader(gguf, f"blk.{il}.ffn_up.weight",
                                                      transpose=True), (d, d_ff), d * d_ff)
    Wd = tape.commit_lazy(f"L{il}_Wd", _field_loader(gguf, f"blk.{il}.ffn_down.weight",
                                                      transpose=True), (d_ff, d), d_ff * d)
    g = tape.matmul(n2g, Wg, **mm)
    u = tape.matmul(n2g, Wu, **mm)
    h = tape.hadamard(tape.silu(g), u, **mm)
    return tape.matmul(h, Wd, **mm)


def _moe_part(gguf, il, key, E):
    """Lazy loader for one router/shared tensor (resident only near its use)."""
    def load():
        from loader import load_maverick_moe_layer
        real = load_maverick_moe_layer(gguf, il, S=S, n_experts=E, skip_experts=True)
        return real[key].contiguous().reshape(-1)
    return load


def build_moe_ffn(tape, n2g, gguf, il, sig_tbl, ones_bc, *, T, E, d, d_ff):
    """MoE FFN (all E experts, Freivalds combines, shared expert)."""
    from loader import maverick_lazy_expert
    mm = dict(s_a=S, s_b=S, s_out=S, output_width=OUTPUT_WIDTH)

    def wexp(kind, name, shape, e):
        ld = maverick_lazy_expert(gguf, il, kind, e, S)
        return tape.commit_lazy(name, ld, shape, shape[0] * shape[1])

    W_router = tape.commit_lazy(f"L{il}_Wr", _moe_part(gguf, il, "W_router", E), (d, E), d * E)
    r = tape.matmul(n2g, W_router, **mm)
    m, r_chosen, _g = route_top1(tape, r, T=T, E=E, B_logit=OUTPUT_WIDTH,
                                  word_bits=WORD_BITS)
    s_val = tape.paired_tlookup(r_chosen, sig_tbl, shift=SIG_SHIFT)
    s_rep = freivalds_combine(tape, s_val, [ones_bc], T=T, E=1, F=d)
    x_r = tape.hadamard(s_rep, n2g, **mm)

    gates = [tape.matmul(x_r, wexp("gate_exps", f"L{il}_Wg{e}", (d, d_ff), e), **mm)
             for e in range(E)]
    ups = [tape.matmul(x_r, wexp("up_exps", f"L{il}_Wu{e}", (d, d_ff), e), **mm)
           for e in range(E)]
    g_sum = freivalds_combine(tape, m, gates, T=T, E=E, F=d_ff)
    up_sum = freivalds_combine(tape, m, ups, T=T, E=E, F=d_ff)
    hidden = tape.hadamard(tape.silu(g_sum), up_sum, **mm)
    outs = [tape.matmul(hidden, wexp("down_exps", f"L{il}_Wd{e}", (d_ff, d), e), **mm)
            for e in range(E)]
    ffn = freivalds_combine(tape, m, outs, T=T, E=E, F=d)

    Wg_s = tape.commit_lazy(f"L{il}_Wgs", _moe_part(gguf, il, "W_gate_sh", E), (d, d_ff), d * d_ff)
    Wu_s = tape.commit_lazy(f"L{il}_Wus", _moe_part(gguf, il, "W_up_sh", E), (d, d_ff), d * d_ff)
    Wd_s = tape.commit_lazy(f"L{il}_Wds", _moe_part(gguf, il, "W_down_sh", E), (d_ff, d), d_ff * d)
    h_s = tape.hadamard(tape.silu(tape.matmul(n2g, Wg_s, **mm)),
                         tape.matmul(n2g, Wu_s, **mm), **mm)
    sh = tape.matmul(h_s, Wd_s, **mm)
    return ffn + sh


def build_model(tape, gguf, prompt_ids, cont_ids, *, V, d, n_layers, E, d_ff):
    from loader import _gguf_by_name
    by = _gguf_by_name(gguf)
    T = len(prompt_ids) + len(cont_ids)
    theta = 500000.0

    E_wt = tape.commit_lazy("token_embd", _field_loader(gguf, "token_embd.weight"),
                             (V, d), V * d)
    x, ind_mid, o_last = build_inputs(tape, gguf, E_wt, prompt_ids, cont_ids, V=V, d=d)
    _log(f"input bound: all {T} tokens hidden ({len(prompt_ids)} prompt + "
         f"{len(cont_ids)} continuation)")

    t_in, t_out = _sigmoid_table()
    sig_tbl = tape.register_table("sigmoid", T_data=t_in, T_Y_data=t_out)
    ones_bc = tape.commit("bc_ones", torch.ones(T * d, dtype=torch.uint64,
                                                 device="cuda"), (T, d))

    def _attn_part(il, key):
        def load():
            return load_attention(gguf, il)[key].reshape(-1)
        return load

    for il in range(n_layers):
        attn = {}
        # gains are tiny (d,) — eager; projection weights lazy (1.7 GB/layer
        # eager across 48 layers OOMs the H100's 80 GB VRAM)
        attn["g_attn_wt"] = tape.commit(f"L{il}_gA", _attn_part(il, "g_attn")(), (d,))
        attn["g_ffn_wt"] = tape.commit(f"L{il}_gF", _attn_part(il, "g_ffn")(), (d,))
        for kk, nm, sh in [("W_Q_wt", "W_Q", (d, H * DH)), ("W_K_wt", "W_K", (d, H * DH)),
                            ("W_V_wt", "W_V", (d, H * DH)), ("W_O_wt", "W_O", (H * DH, d))]:
            attn[kk] = tape.commit_lazy(f"L{il}_{nm}", _attn_part(il, nm),
                                         sh, sh[0] * sh[1])
        use_rope = (il + 1) % 4 != 0
        resid1, n2g = build_attn_chain(tape, x, attn, T=T, d=d, theta=theta,
                                        use_rope=use_rope)
        if il % 2 == 1:
            ffn = build_moe_ffn(tape, n2g, gguf, il, sig_tbl, ones_bc,
                                 T=T, E=E, d=d, d_ff=d_ff)
            kind = f"moe E={E}"
        else:
            ffn = build_dense_ffn(tape, n2g, gguf, il, T=T, d=d)
            kind = "dense"
        x = resid1 + ffn
        _log(f"layer {il} built ({kind}, rope={use_rope}) — "
             f"{len(tape.claims)} claims so far")

    # final norm + gain -> LM head
    n_f = tape.rmsnorm(x, d=d, s=S, eps_int=EPS_INT,
                        s_out=S, output_width=OUTPUT_WIDTH)
    g_out = tape.commit("g_out", _field_loader(gguf, "output_norm.weight")(), (d,))
    n_fg = tape.hadamard_broadcast(n_f, g_out, SEQ=T, d=d, s_a=S, s_b=S,
                                    s_out=S, output_width=OUTPUT_WIDTH)
    lm_name = "output.weight" if "output.weight" in by else "token_embd.weight"
    W_lm = tape.commit_lazy("W_lm", _field_loader(gguf, lm_name, transpose=True),
                             (d, V), d * V)
    logits = tape.matmul(n_fg, W_lm, s_a=S, s_b=S, s_out=S,
                          output_width=OUTPUT_WIDTH)
    _log(f"LM head from {lm_name} ({'tied' if lm_name != 'output.weight' else 'untied'})")

    # UI over the continuation: position t predicts ids[t+1]. The output
    # select O is the INPUT indicator rows shifted by one (ind_mid holds
    # one-hots of ids[1:]), plus a repeat of the last row for position T-1
    # (excluded from the sum) — so the scored tokens are, by shared
    # committed variable, the tokens the model consumed.
    ids = list(prompt_ids) + list(cont_ids)
    targets = ids[1:] + [ids[-1]]
    O_ext = tape.concat([ind_mid, o_last], (T, V))
    sum_pos = list(range(len(prompt_ids) - 1, T - 1))
    Sz, handles = prove_unexplained_info(tape, logits, targets, T=T, V=V,
                                          sum_positions=sum_pos, reveal=True,
                                          O_ext=O_ext, **UI)
    _log(f"UI claim over {len(sum_pos)} continuation positions; "
         f"{len(tape.claims)} claims total")
    return logits, Sz, handles, sum_pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-gguf", required=True)
    ap.add_argument("--tokens", default=None, help="tokens.json (prompt+continuation)")
    ap.add_argument("--layers", type=int, default=48)
    ap.add_argument("--experts", type=int, default=128)
    ap.add_argument("--d", type=int, default=5120)
    ap.add_argument("--d-ff", type=int, default=8192)
    ap.add_argument("--vocab", type=int, default=202048)
    ap.add_argument("--prompt-n", type=int, default=0, help="synthetic prompt len")
    ap.add_argument("--cont-n", type=int, default=0, help="synthetic continuation len")
    ap.add_argument("--witness-only", action="store_true")
    ap.add_argument("--dump-proof", default=None)
    ap.add_argument("--logits-out", default=None)
    ap.add_argument("--ui-abort-above", type=float, default=None,
                    help="bits/token threshold: if the reveal engine pass "
                         "(run before the prove sweep) computes a bound above "
                         "this, abort BEFORE the ~8h sweep instead of proving "
                         "a useless number")
    a = ap.parse_args()
    torch.manual_seed(7)

    if a.tokens:
        tk = json.load(open(a.tokens))
        prompt_ids, cont_ids = tk["prompt"], tk["continuation"]
    else:
        g = torch.Generator().manual_seed(11)
        prompt_ids = torch.randint(0, a.vocab, (a.prompt_n or 4,), generator=g).tolist()
        cont_ids = torch.randint(0, a.vocab, (a.cont_n or 4,), generator=g).tolist()
    T = len(prompt_ids) + len(cont_ids)
    _log(f"layers={a.layers} E={a.experts} T={T} V={a.vocab} "
         f"T_QUERIES={CFG.T_QUERIES} witness_only={a.witness_only}")

    tape = Tape(CFG, silu_config=SILU_CFG, lazy=True)
    t0 = time.time()
    logits, Sz, handles, sum_pos = build_model(
        tape, a.from_gguf, prompt_ids, cont_ids, V=a.vocab, d=a.d,
        n_layers=a.layers, E=a.experts, d_ff=a.d_ff)
    _log(f"build {time.time()-t0:.1f}s, {len(tape.claims)} claims")

    if a.witness_only:
        t0 = time.time()
        keep = {logits.var, Sz.var, handles["surprisal"].var}
        live = tape.run_engine_pass(free_intermediates=True, keep=keep)
        _log(f"witness pass {time.time()-t0:.1f}s "
             f"peakGPU={torch.cuda.max_memory_allocated()/2**30:.2f}GB")
        from max_claim import to_signed
        lg = to_signed(live[logits.var].reshape(-1)).view(T, a.vocab)
        ids = list(prompt_ids) + list(cont_ids)
        agree = sum(int(lg[t].argmax().item() == ids[t + 1])
                    for t in range(len(prompt_ids) - 1, T - 1))
        Sz_v = int(live[Sz.var].cpu()[0])
        bits_total = bound_bits(Sz_v, s_b=UI["s_b"])
        _log(f"UI: Sz={Sz_v}  ->  {bits_total:.1f} bits total = "
             f"{bits_total/len(sum_pos):.4f} bits/token over {len(sum_pos)} "
             f"continuation positions")
        _log(f"greedy agreement: {agree}/{len(sum_pos)} continuation argmax matches")
        if a.logits_out:
            import numpy as np
            np.save(a.logits_out, lg.cpu().numpy().astype("int64"))
            _log(f"logits saved to {a.logits_out}")
        return 0

    # Reveal the bound: compute Sz (engine pass), pin it as the PUBLIC value
    # the verifier reads, then re-zero LogUp mults so the prove sweep
    # re-accumulates cleanly.
    if handles.get("reveal_pin") is not None:
        t0 = time.time()
        _live = tape.run_engine_pass(free_intermediates=True, keep={Sz.var})
        _sz = int(_live[Sz.var].cpu().item())
        _bits = bound_bits(_sz, s_b=UI["s_b"])
        _bpt = _bits / len(sum_pos)
        _log(f"reveal engine pass {time.time()-t0:.1f}s: Sz={_sz} -> "
             f"{_bits:.1f} bits total ({_bpt:.4f} bits/token over "
             f"{len(sum_pos)} positions)")
        if a.ui_abort_above is not None and _bpt > a.ui_abort_above:
            _log(f"ABORT before prove sweep: {_bpt:.4f} bits/token > "
                 f"{a.ui_abort_above} threshold. Skipping the ~8h prove "
                 f"(the number would be useless). No proof written.")
            return 0
        handles["reveal_pin"].public_rhs = _sz
        for _v in list(tape.inputs):
            if getattr(_v, "name", "").endswith("_mult"):
                tape.inputs[_v].zero_()
        _log(f"reveal: Sz={_sz} pinned as PUBLIC bound; "
             f"verifier reads {_bpt:.4f} bits/token from the claim")

    t0 = time.time()
    proof = tape.prove(seed=SEED)
    t_prove = time.time() - t0
    _log(f"prove returned ({t_prove:.1f}s) "
         f"peakGPU={torch.cuda.max_memory_allocated()/2**30:.2f}GB")
    if a.dump_proof:
        import protocol as pr
        from proof_dump import dump_proof
        s_op, s_comb, s_col = pr.round_seeds(SEED)
        Q = pr.random_columns(s_col, CFG)
        t0 = time.time()
        dump_proof(a.dump_proof, pr.claims_to_json(tape.claims, CFG),
                   {"s_op": s_op.hex(), "s_comb": s_comb.hex(), "s_col": s_col.hex()},
                   proof, list(Q), None)
        _log(f"proof dumped to {a.dump_proof} ({time.time()-t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
