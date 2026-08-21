#!/usr/bin/env python3
"""Token-layer subsampled challenge — DEMO MODE, NOT A PROOF.

Same CLI contract as interlock_challenge.py, but instead of proving the whole
forward pass it proves ONE randomly-chosen (token, layer) transition against
committed activations, plus the tail (LM head + U).

    SOUNDNESS: detection probability is ~k/N over (layers x positions)
    token-layers. With k=1 and TinyLlama (22 layers x 21 positions = 462) a
    cheating prover passes ~99.8% of the time, and --t-queries 1 weakens even
    the one transition that is opened. This demonstrates the shape of the
    protocol; it does not prove the inference. Never present a verdict from
    this path as a proof.

    THE SEED MUST COME FROM THE VERIFIER. Without --seed the pick is derived
    from the request and response bytes -- and the prover chooses the response,
    so it can grind until the pick lands on a transition it computed honestly,
    taking detection from 1/462 to ~0. infcli.do_challenge now draws 16 random
    bytes after the response and both certs are fixed and sends them in band;
    model_backend forwards them here. If you drive this script directly, pass
    --seed or the number above is fiction.

    WHAT IS PROVEN IN FULL: the key binding (B1/B2) is not subsampled, and the
    response tokens ARE welded to the model's committed output tokens, so
    keybind=OK means here what it means on the sound path. Only the forward
    pass is sampled -- `pick=` reports how much of it was opened.

How it works:
  Phase A  run the real forward on a tape (engine pass, no Ligero work) and
           capture, per layer, the input residual and the post-RoPE K / V.
           This is the "commit KV caches and activations" stage.
  Phase B  derive (t, L) from a verifier-supplied seed, then build a fresh tape
           that COMMITS the captured x_t / K[0..t] / V[0..t] and proves only
           that transition, plus the tail over the committed final residual.
  Phase C  prove, dump, and check with the independent Rust verifier.

Two existing features make the transition expressible with no protocol change:
tape.rope takes `position_offset`, and MatmulClaim carries independent (m,k,n)
so q(1,d) x K(t+1,d)^T is an ordinary claim.

stdout contract matches interlock_challenge.py's last line, with mode=subsample
added so a caller can never mistake it for the sound path.
"""
import argparse
import hashlib
import json
import os
import struct
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT + "/prover")
sys.path.insert(0, ROOT + "/demo")
# The interlock app (wire crypto) as a sibling checkout; ILK_APP overrides.
ILK_APP = os.environ.get("ILK_APP") or os.path.join(
    os.path.dirname(ROOT), "interlock", "app")
sys.path.insert(0, ILK_APP)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

MODEL = os.environ.get("VERINF_MODEL", ROOT + "/models/llama-160m")

# Calibrated fine-scale unexplained-information settings (same as
# interlock_challenge.py and demo/gate.py).
UI_LM_SOUT, UI_LM_OW, UI_S_C = 4096, 24, 1 << 18


def _ids(s):
    return [int(v) for v in s.split(",") if v.strip() != ""]


def _payload(ids):
    return b"".join(struct.pack("<I", i & 0xFFFFFFFF) for i in ids)


def _emit(verdict, U="NA", verify="?", out_bind="?", hreq="", hrsp="", pick="",
          keybind="n/a"):
    print("CHALLENGE_RESULT verdict=%s U=%s verify=%s out_bind=%s hreq=%s hrsp=%s "
          "keybind=%s mode=subsample pick=%s" % (verdict, U, verify, out_bind, hreq,
                                                 hrsp, keybind, pick),
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--request", required=True)
    ap.add_argument("--response", required=True)
    ap.add_argument("--t-queries", default="10")
    # Same key-binding inputs as the sound path. Subsampling weakens the MODEL
    # statement (one token-layer of ~460), but it does not weaken the key
    # binding: B1/B2 are proven in full here, over the whole request and
    # response ciphertexts, because they cost ~1.5 s and there is nothing to
    # gain by sampling them.
    ap.add_argument("--nonce")
    ap.add_argument("--key-commit")
    ap.add_argument("--ct-in")
    ap.add_argument("--ct-out")
    ap.add_argument("--seed", default="",
                    help="verifier-supplied seed selecting (token, layer); "
                         "empty = derive from the request/response bytes")
    a = ap.parse_args()

    req_ids, rsp_ids = _ids(a.request), _ids(a.response)
    hreq = hashlib.sha256(_payload(req_ids)).hexdigest()[:16]
    hrsp = hashlib.sha256(_payload(rsp_ids)).hexdigest()[:16]
    if not req_ids or not rsp_ids:
        _emit("FAIL", verify="ERROR", out_bind="MISMATCH", hreq=hreq, hrsp=hrsp)
        return 1

    os.environ["LIGERO_T_QUERIES"] = str(a.t_queries)
    import torch
    import demo_llama7b as D
    import protocol as pr
    from tape import Tape
    from loader import LazyHFLoader, load_final_weights, free_model_cache
    from proof_dump import dump_proof as write_proof

    cfg = json.load(open(MODEL + "/config.json"))
    d, d_ff = cfg["hidden_size"], cfg["intermediate_size"]
    H = cfg["num_attention_heads"]
    d_h, V = d // H, cfg["vocab_size"]
    n_layers = cfg["num_hidden_layers"]

    full = req_ids + rsp_ids
    SEQ = len(full)
    D.d, D.d_ff, D.d_h = d, d_ff, d_h
    D.ROPE_BASE = cfg.get("rope_theta", 10000.0)
    D.ROPE_SCALING = cfg.get("rope_scaling")
    S, EPS_INT, OW, Z_MAX = D.S, D.EPS_INT, D.OUTPUT_WIDTH, D.Z_MAX
    mm = dict(s_a=S, s_b=S, s_out=S, output_width=OW)

    # -- verifier's coin: which token-layer gets checked -----------------------
    seed_src = (a.seed or "").encode() or _payload(req_ids) + _payload(rsp_ids)
    h = hashlib.sha256(b"token-layer-pick|" + seed_src).digest()
    LAYER = int.from_bytes(h[:4], "big") % n_layers
    POS = int.from_bytes(h[4:8], "big") % SEQ
    pick = "L%d/t%d" % (LAYER, POS)
    n_tl = n_layers * SEQ
    print("[subsample] DEMO MODE -- spot check, not a proof", flush=True)
    print("[subsample] model=%s layers=%d SEQ=%d -> %d token-layers"
          % (os.path.basename(MODEL), n_layers, SEQ, n_tl), flush=True)
    print("[subsample] verifier picked %s; detection probability ~1/%d = %.2f%%"
          % (pick, n_tl, 100.0 / n_tl), flush=True)

    # ---------------- Phase A: real forward, capture activations + KV --------
    t_a = time.time()
    D.SEQ = SEQ
    tapeA = Tape(D.CFG, silu_config=D.SILU_CFG, lazy=True)
    loaderA = LazyHFLoader(MODEL, S=S, d_h=d_h)
    from loader import load_token_embedding
    E_full = load_token_embedding(MODEL, S=S)
    uniq = sorted(set(full))
    idx = {t: i for i, t in enumerate(uniq)}
    E_sub = E_full.view(-1, d).index_select(
        0, torch.tensor(uniq, dtype=torch.int64, device="cuda")).contiguous().view(-1)
    del E_full
    free_model_cache()
    E_wt = tapeA.commit("E_embedding_subset", E_sub, (len(uniq), d))
    resid = tapeA.embed(E_wt, token_ids=[idx[t] for t in full], d=d)

    caps = []                     # per layer: (x_in, k_rope, v, x_out)
    for L in range(n_layers):
        W = D._commit_weights_from_hf_lazy(tapeA, loaderA, L)
        x = resid
        n1 = tapeA.rmsnorm(x, d=d, s=S, eps_int=EPS_INT, s_out=S, output_width=OW)
        n1g = tapeA.hadamard_broadcast(n1, W["rms_pre_attn_w"], SEQ=SEQ, d=d,
                                       s_a=S, s_b=S, s_out=S, output_width=OW)
        q = tapeA.matmul(n1g, W["W_Q"], **mm)
        k = tapeA.matmul(n1g, W["W_K"], **mm)
        v = tapeA.matmul(n1g, W["W_V"], **mm)
        qr = tapeA.rope(q, SEQ=SEQ, d_h=d_h, heads=H, base=D.ROPE_BASE,
                        rope_scaling=D.ROPE_SCALING, s_x=S, s_out=S, output_width=OW)
        kr = tapeA.rope(k, SEQ=SEQ, d_h=d_h, heads=H, base=D.ROPE_BASE,
                        rope_scaling=D.ROPE_SCALING, s_x=S, s_out=S, output_width=OW)
        sc = tapeA.matmul(qr, kr, transpose_b=True, heads=H, head_dim=d_h, **mm)
        sm = tapeA.softmax(sc, M=SEQ, s_x=S, s_c=S, s_y=S, Z_max=Z_MAX, saturate=True,
                           Z_high_width=16, aux_chunk_width=D.AUX_CHUNK_WIDTH, causal=True, heads=H)
        att = tapeA.matmul(sm, v, heads=H, head_dim=SEQ, **mm)
        r1 = x + tapeA.matmul(att, W["W_O"], **mm)
        n2 = tapeA.rmsnorm(r1, d=d, s=S, eps_int=EPS_INT, s_out=S, output_width=OW)
        n2g = tapeA.hadamard_broadcast(n2, W["rms_pre_ffn_w"], SEQ=SEQ, d=d,
                                       s_a=S, s_b=S, s_out=S, output_width=OW)
        g = tapeA.matmul(n2g, W["W_gate"], **mm)
        u = tapeA.matmul(n2g, W["W_up"], **mm)
        inter = tapeA.hadamard(tapeA.silu(g), u, **mm)
        out = r1 + tapeA.matmul(inter, W["W_down"], **mm)
        caps.append((x, kr, v, out))
        resid = out

    keep = set()
    for (x, kr, v, o) in caps:
        keep |= {x.var, kr.var, v.var, o.var}
    live = tapeA.run_engine_pass(free_intermediates=True, keep=keep)
    xs = [live[c[0].var].clone() for c in caps]
    ks = [live[c[1].var].clone() for c in caps]
    vs = [live[c[2].var].clone() for c in caps]
    final_resid = live[caps[-1][3].var].clone()
    del tapeA, live
    torch.cuda.empty_cache()
    t_a = time.time() - t_a
    print("[subsample] phase A (forward + capture): %.1fs" % t_a, flush=True)

    # ---------------- Phase B: prove the picked transition + tail ------------
    t_b = time.time()
    _bt = {}

    def _mark(k, t0):
        _bt[k] = _bt.get(k, 0.0) + (time.time() - t0)
    KV = POS + 1
    _t = time.time()
    tape = Tape(D.CFG, silu_config=D.SILU_CFG, lazy=True)
    loader = LazyHFLoader(MODEL, S=S, d_h=d_h)
    _mark("tape+loader ctor", _t)
    _t = time.time()
    W = D._commit_weights_from_hf_lazy(tape, loader, LAYER)
    _mark("commit layer weights", _t)

    D.SEQ = 1
    _t = time.time()
    x_t = tape.commit("x_t", xs[LAYER].view(SEQ, d)[POS].contiguous().view(-1), (1, d))
    Kc = tape.commit("K_cache", ks[LAYER].view(SEQ, d)[:KV].contiguous().view(-1), (KV, d))
    Vc = tape.commit("V_cache", vs[LAYER].view(SEQ, d)[:KV].contiguous().view(-1), (KV, d))
    _mark("commit x_t/K/V", _t)

    _t = time.time()
    n1 = tape.rmsnorm(x_t, d=d, s=S, eps_int=EPS_INT, s_out=S, output_width=OW)
    n1g = tape.hadamard_broadcast(n1, W["rms_pre_attn_w"], SEQ=1, d=d,
                                  s_a=S, s_b=S, s_out=S, output_width=OW)
    q = tape.matmul(n1g, W["W_Q"], **mm)
    qr = tape.rope(q, SEQ=1, d_h=d_h, heads=H, base=D.ROPE_BASE,
                   rope_scaling=D.ROPE_SCALING, position_offset=POS,
                   s_x=S, s_out=S, output_width=OW)
    sc = tape.matmul(qr, Kc, transpose_b=True, heads=H, head_dim=d_h, **mm)
    sm = tape.softmax(sc, M=KV, s_x=S, s_c=S, s_y=S, Z_max=Z_MAX, saturate=True,
                      Z_high_width=16, aux_chunk_width=D.AUX_CHUNK_WIDTH, causal=False, heads=H)
    att = tape.matmul(sm, Vc, heads=H, head_dim=KV, **mm)
    r1 = x_t + tape.matmul(att, W["W_O"], **mm)
    n2 = tape.rmsnorm(r1, d=d, s=S, eps_int=EPS_INT, s_out=S, output_width=OW)
    n2g = tape.hadamard_broadcast(n2, W["rms_pre_ffn_w"], SEQ=1, d=d,
                                  s_a=S, s_b=S, s_out=S, output_width=OW)
    g = tape.matmul(n2g, W["W_gate"], **mm)
    u = tape.matmul(n2g, W["W_up"], **mm)
    inter = tape.hadamard(tape.silu(g), u, **mm)
    blk_out = r1 + tape.matmul(inter, W["W_down"], **mm)
    _mark("layer graph", _t)

    sum_pos = list(range(len(req_ids) - 1, SEQ - 1))
    # MEASURED 2026-08-19 -- this knob buys nothing; leave it off.
    #
    # The comment here used to claim the UI machinery was "the bulk of the tail,
    # which is the bulk of phase B", and that this was the only lever on the
    # tail floor. Fine-grained instrumentation says otherwise: the tail is 0.2s
    # of an 8.5s phase B, and subsampling 8 positions down to 2 changed the
    # total by 0.1s -- inside noise.
    #
    # The reason is that phase B is dominated by the LM head WEIGHTS (d x V =
    # 32k rows at ELL=2048, larger than the sampled layer), and those are the
    # same size whatever subset of positions U sums over. Only the logits
    # shrink (125 -> 31 rows), which is noise.
    #
    # Meanwhile the cost is real and it flatters: U fell from 0.1478 (the
    # turn's actual bound) to 0.0004 over 2 cherry-picked positions. LOWER U
    # reads as less unexplained information, i.e. less exfiltration bandwidth
    # than the truth -- the wrong direction for the threat model. A knob that
    # improves the headline number by 370x while saving nothing is a trap.
    #
    # The real lever on the LM head is the vocabulary axis, not the position
    # axis: commit all logits (cheap, 125 rows) and prove only a verifier-chosen
    # subset of vocabulary COLUMNS. That needs a soundness argument first --
    # U is dominated by a few columns (argmax, observed token, the head of the
    # distribution), so uniform column sampling is weakest exactly where an
    # attacker would push.
    #
    # SUBSAMPLE_U_POSITIONS=n keeps n positions, chosen by the same verifier
    # seed. The reported bound is then over those positions only -- it is NOT
    # the turn's U, and the result line says so via u_positions=.
    _n_u = int(os.environ.get("SUBSAMPLE_U_POSITIONS", "0"))
    if 0 < _n_u < len(sum_pos):
        step = max(1, len(sum_pos) // _n_u)
        sum_pos = sum_pos[::step][:_n_u]
        print("[subsample] U spot-checked over %d of %d response positions"
              % (len(sum_pos), SEQ - len(req_ids)), flush=True)

    # The LM head is d x V (2048 x 32000 for TinyLlama = 65.5M params, bigger
    # than several transformer layers) and it runs at EVERY position -- 21 x
    # 32000 = 672k logit values, all committed. But U is summed only over the
    # response positions, and the output binding only covers those too: the
    # logits at prompt positions are computed, committed, and never read.
    #
    # So slice the committed final residual to the positions U needs before the
    # head. That cuts the head's work by SEQ/len(sum_pos) -- here 21/8 = 2.6x --
    # and phase B is mostly the head.
    #
    # Position p of the slice is original position sum_pos[0]+p, whose observed
    # output token is full[sum_pos[0]+p+1]; re-index out_tokens to match.
    _p0 = sum_pos[0]
    _n = len(sum_pos)
    D.SEQ = _n

    # ---- LM-head column subsampling (SUBSAMPLE_LM_COLS=n, 0 = off) --------
    # The LM head is d x V and is the single largest block on this tape --
    # 32k rows at ELL=2048 for TinyLlama, larger than the sampled layer -- while
    # the LOGITS it produces are only n x V (~125 rows). So commit every logit
    # and prove the head's matmul on a random subset of vocabulary COLUMNS.
    #
    # The column set comes from the same verifier-supplied seed that picks the
    # (token, layer), under its own domain separator. That ordering is what
    # makes the sample meaningful at all: the prover commits, THEN learns which
    # columns are checked, so it cannot place a lie where nobody looks.
    #
    # MEASURED 2026-08-19 -- IT DOES NOT PAY. Leave it off (the default).
    #   columns   coverage   prove   total
    #   all 32000   100%      6.8s   14.0s
    #        4096   12.8%     5.3s   13.7s
    #        1024    3.2%     5.1s   13.4s
    #          64    0.2%     5.0s   13.3s
    # Cutting coverage 64x (4096 -> 64) buys 0.4s; the ceiling on the whole
    # idea is ~0.7s, about 5%. Two reasons the row count misled: the logits
    # must still be computed under the ENGINE's arithmetic, which costs a
    # throwaway pass of ~1.1s no matter how few columns are proven; and what
    # remains of prove is dominated by the sampled LAYER's 21k W-block rows,
    # which this does not touch. Trading LM-head attestation down to 3% for a
    # 4% speedup is a bad bargain -- recorded here so the estimate is not
    # re-derived from row counts, which is what overestimated it the first time.
    #
    # SOUNDNESS, PLAINLY: this is uniform sampling, and uniform is weakest
    # exactly where an attacker would push. U is dominated by a handful of
    # columns -- the argmax (sets v*, hence every gap) and the observed token's
    # column (sets gap_o) -- and each is ONE column out of V. A prover who
    # inflates just the observed token's logit is caught with probability
    # |J|/V, so at |J|=1024 of 32000 that is ~3%. The attack direction is to
    # UNDER-claim U, which is the wrong direction for the threat model. Pinning
    # those two columns via the existing one-hot machinery would remove the
    # dominant attack cheaply; until that lands, this is a speed knob on a path
    # already labelled NOT A PROOF, and the result line reports lm_cols= so the
    # coverage travels with the verdict.
    _n_lm = int(os.environ.get("SUBSAMPLE_LM_COLS", "0"))
    _lm_cols = None
    if 0 < _n_lm < V:
        import random as _rnd
        _r = _rnd.Random(hashlib.sha256(seed_src + b"lm-cols").digest())
        _lm_cols = sorted(_r.sample(range(V), _n_lm))
        print("[subsample] LM head: proving %d of %d vocabulary columns "
              "(%.2f%%); logits all committed"
              % (_n_lm, V, 100.0 * _n_lm / V), flush=True)
    _t = time.time()
    fw = load_final_weights(MODEL, S=S)
    _mark("load final weights (LM head)", _t)
    _resid_slice = final_resid.view(SEQ, d)[_p0:_p0 + _n].contiguous().view(-1)
    _logit_vals = None
    if _lm_cols is not None:
        # The true logits, under the ENGINE's fixed-point arithmetic rather than
        # torch's float, are needed to commit them. Get them from a throwaway
        # tape that carries the full head: it is engine-evaluated only, never
        # proven, so it costs compute (a 8 x 2048 x 32000 matmul, milliseconds)
        # and none of the Ligero row cost that committing the head would.
        _t = time.time()
        _tapeL = Tape(D.CFG, silu_config=D.SILU_CFG, lazy=True)
        _fnL = _tapeL.commit("final_norm_w", fw["final_norm_w"], (d,))
        _lmL = _tapeL.commit("W_lm_head", fw["W_lm_head"], (d, V))
        _rsL = _tapeL.commit("resid_final", _resid_slice, (_n, d))
        _lgL = D._run_tail(_tapeL, _rsL, _fnL, _lmL, vocab_size=V,
                           lm_s_out=UI_LM_SOUT, lm_ow=UI_LM_OW)
        _lvL = _tapeL.run_engine_pass(free_intermediates=True, keep={_lgL.var})
        _logit_vals = _lvL[_lgL.var].clone()
        del _tapeL, _lgL, _lvL
        _mark("logits via throwaway engine pass", _t)

    _t = time.time()
    fn_w = tape.commit("final_norm_w", fw["final_norm_w"], (d,))
    if _lm_cols is None:
        lm_w = tape.commit("W_lm_head", fw["W_lm_head"], (d, V))
    else:
        import torch as _th
        _cols = _th.tensor(_lm_cols, dtype=_th.int64,
                           device=fw["W_lm_head"].device)
        lm_w = tape.commit("W_lm_head_J",
                           fw["W_lm_head"].view(d, V).index_select(1, _cols)
                           .contiguous().view(-1), (d, _n_lm))
    _mark("commit LM head (d x V)", _t)
    _t = time.time()
    free_model_cache()
    _mark("free_model_cache", _t)
    _t = time.time()
    resid_f = tape.commit("resid_final",
                          final_resid.view(SEQ, d)[_p0:_p0 + _n].contiguous().view(-1),
                          (_n, d))
    _mark("commit resid_final", _t)
    _t_tail = time.time()
    print("[subsample] LM head over %d of %d positions (the ones U sums)"
          % (_n, SEQ), flush=True)
    # Calibrated fine-scale UI settings -- the SAME values interlock_challenge.py
    # uses. demo_llama7b's own UI_* defaults are coarser (s_c=2^28), which makes
    # the exp/gap tables larger AND reports a meaningless bound (~15 bits/token
    # instead of ~0.017).
    if _lm_cols is not None:
        # Same first two steps as _run_tail (final RmsNorm + per-channel gain),
        # then the head restricted to the sampled columns.
        _fnorm = tape.rmsnorm(resid_f, d=d, s=D.S, eps_int=D.EPS_INT,
                              s_out=D.S, output_width=D.OUTPUT_WIDTH)
        _fnormg = tape.hadamard_broadcast(_fnorm, fn_w, SEQ=_n, d=d,
                                          s_a=D.S, s_b=D.S, s_out=D.S,
                                          output_width=D.OUTPUT_WIDTH)
        logits_J = tape.matmul(_fnormg, lm_w, s_a=D.S, s_b=D.S,
                               s_out=UI_LM_SOUT, output_width=UI_LM_OW)
        # Every logit is committed; U is computed over all of them.
        logits = tape.commit("logits_full", _logit_vals, (_n, V))
        # ... and the sampled columns are pinned to the proven matmul output,
        # which is the only thing tying the committed logits to the model.
        from crypto_binding import pool_gather as _gather
        _col_ids = [r * V + j for r in range(_n) for j in _lm_cols]
        _g = _gather(tape, logits, "lm_col_sample", _col_ids)
        tape.lincomb([_g, logits_J], [1, -1], 0)
    else:
        logits = D._run_tail(tape, resid_f, fn_w, lm_w, vocab_size=V,
                             lm_s_out=UI_LM_SOUT, lm_ow=UI_LM_OW)
    if logits.data is None:
        _t = time.time()
        lv = tape.run_engine_pass(free_intermediates=True,
                                  keep={logits.var, blk_out.var})
        _mark("engine pass (logits)", _t)
        logits._data = lv[logits.var]
        for _v in list(tape.inputs):
            if _v.name.endswith("_mult"):
                tape.inputs[_v].zero_()
    # Re-indexed to the slice: position p holds full[_p0+p+1].
    out_tokens = [full[_p0 + p + 1] for p in range(_n)]
    ui_Sz, ui_info = D._run_unexplained_info(
        tape, logits, vocab_size=V, seq=_n, output_tokens=out_tokens,
        s_c=UI_S_C, sum_positions=list(range(_n)))
    if ui_info.get("reveal_pin") is not None:
        from unexplained_info import bound_bits
        _t = time.time()
        lv = tape.run_engine_pass(free_intermediates=True,
                                  keep={ui_Sz.var, blk_out.var})
        _mark("engine pass (Sz reveal)", _t)
        sz = int(lv[ui_Sz.var].cpu().item())
        ui_info["reveal_pin"].public_rhs = sz
        for _v in list(tape.inputs):
            if _v.name.endswith("_mult"):
                tape.inputs[_v].zero_()
        u_total = bound_bits(sz, s_b=ui_info["s_b"])
    else:
        u_total = float("nan")
    u_per_tok = u_total / max(len(sum_pos), 1)
    print("[subsample] claims=%d  U=%.4f bits over %d positions = %.4f bits/token"
          % (len(tape.claims), u_total, len(sum_pos), u_per_tok), flush=True)
    _upos = len(sum_pos)

    _mark("tail (LM head + U machinery)", _t_tail)
    _t = time.time()
    # ---- key binding (full, not subsampled) --------------------------------
    keybind = "n/a"
    _welded = 0
    _kb = [a.nonce, a.key_commit, a.ct_in, a.ct_out]
    if all(_kb):
        import ilk_crypto as _ic
        import crypto_binding as _cbnd
        _km = _ic.derive(_ic.load_psk(), bytes.fromhex(a.nonce))[0]
        _kc = bytes.fromhex(a.key_commit)
        if _ic.key_commit(_km) != _kc:
            print("[subsample] KEY_COMMIT does not match the PSK-derived key material",
                  flush=True)
            _emit("FAIL", verify="ERROR", hreq=hreq, hrsp=hrsp, pick=pick,
                  keybind="COMMIT-MISMATCH")
            return 1
        _t0 = time.time()
        _res = _cbnd.bind(tape, keymat=_km, key_commit=_kc,
                          streams=[{"name": "in", "tokens": req_ids,
                                    "ct": bytes.fromhex(a.ct_in)},
                                   {"name": "out", "tokens": rsp_ids,
                                    "ct": bytes.fromhex(a.ct_out)}],
                          reveal_tokens={"in"})
        # WELD (token-binding.md P5). The comment that used to sit here said this
        # path "does not build the model's output-token commitment (it proves one
        # token-layer, not the LM head over every position)". That was stale: the
        # tail above runs the LM head over exactly the positions U sums, and
        # _run_unexplained_info hands back the same committed `tok` vector the
        # sound path welds to. So the link is available, and it costs one gather
        # plus one LinComb -- against a tail that was already being paid for.
        #
        # Note the indexing difference from interlock_challenge.py: there, ui_tok
        # spans the whole sequence and the gather uses the original sum_positions.
        # Here the residual was sliced to [_p0 : _p0+_n] before the head, so ui_tok
        # is already response-local and the gather is range(_n).
        #
        # What this does and does not buy: it ties the decrypted wire tokens to the
        # tokens the committed logits were scored on. The logits' own provenance is
        # still only the one sampled transition, so this closes the token<->wire
        # seam, not the subsampling. keybind=OK now means what it means in sound
        # mode; `pick=` still says how much of the forward pass was opened.
        _wtok = (ui_info or {}).get("tok")
        _welded = 0
        if _wtok is None:
            print("[subsample] weld skipped: no committed output tokens "
                  "(unexplained_info off)", flush=True)
        elif _n != len(rsp_ids):
            # U was subsampled, so only part of the response has a committed token
            # behind it. Welding the covered subset and still reporting OK would
            # overclaim, so refuse and keep the honest label.
            print("[subsample] weld skipped: U covers %d of %d response positions"
                  % (_n, len(rsp_ids)), flush=True)
        else:
            _g = _cbnd.pool_gather(tape, _wtok, "cb_weld_out", list(range(_n)))
            _cbnd.bind_tokens_to(tape, _res["tok"]["out"], _g)
            _welded = _n
        print("[subsample] key binding: +claims in %.1fs (welded %d response tokens)"
              % (time.time() - _t0, _welded), flush=True)
        keybind = "PENDING"
    elif any(_kb):
        print("[subsample] partial key-binding arguments; refusing to guess", flush=True)
        _emit("FAIL", verify="ERROR", hreq=hreq, hrsp=hrsp, pick=pick, keybind="ERROR")
        return 1

    proof = tape.prove(seed=b"xformer-single-tape")
    torch.cuda.synchronize()
    _mark("prove", _t)
    t_b = time.time() - t_b
    print("[subsample] phase B breakdown: %s"
          % "  ".join("%s=%.1fs" % (k, v) for k, v in _bt.items()), flush=True)

    # ---------------- Phase C: dump + independent verify ---------------------
    path = os.path.join(tempfile.gettempdir(), "subsample_%d.json" % os.getpid())
    s_op, s_comb, s_col = pr.round_seeds(b"xformer-single-tape")
    Q = list(pr.random_columns(s_col, D.CFG))
    write_proof(path, pr.claims_to_json(tape.claims, D.CFG),
                {"s_op": s_op.hex(), "s_comb": s_comb.hex(), "s_col": s_col.hex()},
                proof, Q, None)
    t_c = time.time()
    res = subprocess.run([ROOT + "/verifier/target/release/verify_proof", path],
                         capture_output=True, text=True)
    t_c = time.time() - t_c
    ok = res.returncode == 0 and "rust_verify: ACCEPT" in res.stdout
    try:
        # KEEP_PROOF=1 leaves the dump on disk for profiling. Off by default:
        # the proof is large and never leaves this box.
        if os.environ.get("KEEP_PROOF") == "1":
            print("[subsample] kept proof: %s" % path, flush=True)
            raise OSError("kept")
        os.unlink(path)
    except OSError:
        pass
    print("[subsample] phase B (prove): %.1fs   phase C (verify): %.1fs   total %.1fs"
          % (t_b, t_c, t_a + t_b + t_c), flush=True)
    print("[subsample] rust verifier: %s" % ("ACCEPT" if ok else "REJECT"), flush=True)
    _emit("PASS" if ok else "FAIL", U="%.4f" % u_per_tok,
          verify="ACCEPT" if ok else "REJECT", out_bind="OK",
          hreq=hreq, hrsp=hrsp,
          pick="%s,u=%d/%d,lm_cols=%s" % (pick, _upos, SEQ - len(req_ids),
                                          ("%d/%d" % (_n_lm, V)) if _lm_cols
                                          else "all"),
          # OK only when the response tokens were actually welded to the
          # model's committed output tokens. If the weld was skipped the old
          # OK-NOWELD label stands -- B1/B2 are still proven in full, but the
          # decrypted stream is not tied to the forward pass, and reporting
          # plain OK would claim a link that does not exist.
          keybind=(("OK" if _welded else "OK-NOWELD") if ok else "FAIL")
                  if keybind == "PENDING" else keybind)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
