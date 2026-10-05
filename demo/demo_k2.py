"""Kimi K2 through the prover: the two-layer gate's driver
(analysis/k2-session-sizing.md §5–6). A RESEARCH driver, like
profiler/instrumented_prove.py: no admission gate, an in-process enrollment,
the public bound discovered by a reveal pass, the Rust check under the same
run's anchors. Cite its checks and timings, not its proofs.

The tape: the token-select embedding over hidden one-hot inputs
(demo_maverick_full.build_inputs); per layer, latent attention as
composition A (analysis/mla-attention-design.md §4.2: σ folded into the
query weights, YaRN RoPE #27, HeadInterleaveClaim #28), then layer 0's dense
FFN or a top-k MoE (router, the sigmoid table at S_sel, topk_moe_ffn #25,
the shared expert); the final norm, gain and LM head; the
unexplained-information tail over the continuation. Every weight is a
persistent (enrolled) variable, the RMSNorm gains included; under --bridge
the routed experts are external inputs the streaming enrollment
authenticates.

Modes, in the order the gate runs them:
  check     the engine pass, then the integer reference (k2_int_reference)
            on the CPU from the same GGUF: every named intermediate must
            agree EXACTLY. Prints and records the ranges of §6.2.
  fidelity  CPU only: the integer reference against the float64 reference
            (k2_float_reference): relative errors, then routing differences,
            end to end and on the same router input.
  prove     enrollment (the dense W block, and the bridge's), reveal pass,
            prove, Rust check; then each --negatives case proved on a fresh
            tape against the HONEST enrollment (the same WeightCommitment
            and bridge enrollment objects, never re-enrolled) and checked
            under the honest weight root and enrollment identity:
              interleave      one HeadInterleaveClaim slot swapped (witness)
              yarn-witness    the rotary witness from the unscaled tables
                              under the YaRN statement
              yarn-statement  the claims without YaRN, under the honest
                              statement digest (and, for contrast, its own)
              wrong-slice     expert 0's gate shard decoded from expert 1's
                              rows (--bridge)
            each must be rejected.

    python3 demo/demo_k2.py --mode check --from-gguf <UD-Q4_K_XL dir> \\
        --prompt-n 50 --cont-n 50 --record /workspace/k2-s100-check.json
"""
import argparse
import contextlib
import dataclasses
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time

_R = pathlib.Path(__file__).resolve().parents[1]
for _p in (_R / "prover", _R / "demo", _R / "profiler", _R / "prover" / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from k2_attention import K2_MLA          # noqa: E402  (numpy only)
from k2_loader import K2_INT, S_SEL      # noqa: E402

S = K2_INT["S"]
OW = K2_INT["output_width"]
NEGATIVES = ("interleave", "yarn-witness", "yarn-statement", "wrong-slice")

_T0 = time.time()


def log(msg):
    print(f"[demo_k2 +{time.time() - _T0:.1f}s] {msg}", flush=True)


def yarn_scaling(cfg):
    from model_config import YarnScaling
    return YarnScaling(factor=cfg["factor"], original_max_position_embeddings=cfg["original_max_pos"],
                       beta_fast=cfg["beta_fast"], beta_slow=cfg["beta_slow"],
                       mscale=cfg["mscale"], mscale_all_dim=cfg["mscale_all_dim"])


# ---- the tape ---------------------------------------------------------------------

def _mm():
    return dict(s_a=S, s_b=S, s_out=S, output_width=OW)


def _norm_gain(tape, x, gain, *, T, d):
    n = tape.rmsnorm(x, d=d, s=S, eps_int=K2_INT["eps_int"], s_out=S, output_width=OW)
    return tape.hadamard_broadcast(n, gain, SEQ=T, d=d, **_mm())


def build_mla(tape, x, gguf, il, *, T, cfg, offset, yarn, trace):
    """Latent attention and the post-attention norm. Returns (r1, n2g)."""
    from head_interleave import head_interleave
    from k2_loader import attention_loader
    d, H, qr, kr = cfg["d"], cfg["H"], cfg["q_rank"], cfg["kv_rank"]
    dn, dr, dv = cfg["d_nope"], cfg["d_rope"], cfg["d_v"]
    shapes = {"g_attn": (d,), "g_q_a": (qr,), "g_kv_a": (kr,), "g_ffn": (d,),
              "W_qa": (d, qr), "W_q_nope": (qr, H * dn), "W_q_pe": (qr, H * dr),
              "W_ckv": (d, kr), "W_k_pe": (d, dr), "W_k_nope": (kr, H * dn),
              "W_v": (kr, H * dv), "W_o": (H * dv, d)}
    w = {k: tape.commit_lazy(f"L{il}_{k}", attention_loader(gguf, il, k, S, cfg), sh,
                             sh[0] * (sh[1] if len(sh) > 1 else 1))
         for k, sh in shapes.items()}
    mm = _mm()
    p = f"L{il}"
    n1g = _norm_gain(tape, x, w["g_attn"], T=T, d=d)
    aqg = _norm_gain(tape, tape.matmul(n1g, w["W_qa"], **mm), w["g_q_a"], T=T, d=qr)
    ckvg = _norm_gain(tape, tape.matmul(n1g, w["W_ckv"], **mm), w["g_kv_a"], T=T, d=kr)
    rope = dict(SEQ=T, d_h=dr, s_x=S, base=cfg["base"], position_offset=offset,
                rope_scaling=yarn_scaling(cfg) if yarn else None, s_out=S, output_width=OW)
    q_nope = tape.matmul(aqg, w["W_q_nope"], **mm)
    q_pe = tape.rope(tape.matmul(aqg, w["W_q_pe"], **mm), heads=H, **rope)
    k_pe = tape.rope(tape.matmul(n1g, w["W_k_pe"], **mm), heads=1, **rope)
    k_nope = tape.matmul(ckvg, w["W_k_nope"], **mm)
    v = tape.matmul(ckvg, w["W_v"], **mm)
    query = head_interleave(tape, q_nope, q_pe, T=T, H=H, w1=dn, w2=dr)
    key = head_interleave(tape, k_nope, k_pe, T=T, H=H, w1=dn, w2=dr, shared=True)
    sc = tape.matmul(query, key, transpose_b=True, heads=H, head_dim=dn + dr, **mm)
    sm = tape.softmax(sc, M=T, s_x=S, s_c=S, s_y=S, Z_max=K2_INT["z_max"], saturate=True,
                      Z_high_width=16, aux_chunk_width=24, causal=True, heads=H)
    att = tape.matmul(sm, v, heads=H, head_dim=T, **mm)
    r1 = x + tape.matmul(att, w["W_o"], **mm)
    n2g = _norm_gain(tape, r1, w["g_ffn"], T=T, d=d)
    trace.update({f"{p}.n1g": n1g, f"{p}.aqg": aqg, f"{p}.ckvg": ckvg, f"{p}.q_pe": q_pe,
                  f"{p}.k_pe": k_pe, f"{p}.query": query, f"{p}.key": key, f"{p}.v": v,
                  f"{p}.scores": sc, f"{p}.sm": sm, f"{p}.att": att, f"{p}.r1": r1,
                  f"{p}.n2g": n2g})
    return r1, n2g


def _dense_ffn(tape, x, gguf, names, *, d, d_ff, tag):
    from demo_maverick_full import _field_loader
    gate, up, down = names
    Wg = tape.commit_lazy(f"{tag}_Wg", _field_loader(gguf, gate, transpose=True), (d, d_ff), d * d_ff)
    Wu = tape.commit_lazy(f"{tag}_Wu", _field_loader(gguf, up, transpose=True), (d, d_ff), d * d_ff)
    Wd = tape.commit_lazy(f"{tag}_Wd", _field_loader(gguf, down, transpose=True), (d_ff, d), d_ff * d)
    mm = _mm()
    h = tape.hadamard(tape.silu(tape.matmul(x, Wg, **mm)), tape.matmul(x, Wu, **mm), **mm)
    return tape.matmul(h, Wd, **mm)


def _rows(t):
    """The leading dimension of a GGUF matrix (gguf-py's shape)."""
    return int(t.data.shape[0])


def build_moe(tape, n2g, gguf, il, sig_tbl, *, T, d, kk, bridge, wrong_slice, trace):
    """Router, sigmoid lookup, top-k MoE (topk_moe_ffn), shared expert."""
    from demo_maverick_full import _field_loader
    from k2_int_reference import topk_width
    from k2_loader import bias_ints, by_name, ints_loader
    from loader import gguf_provenance, maverick_lazy_expert
    from topk_routing import topk_moe_ffn
    by = by_name(gguf)
    E = int(by[f"blk.{il}.ffn_gate_inp.weight"].n_elements) // d
    d_ff = int(by[f"blk.{il}.ffn_gate_exps.weight"].data.shape[1])
    mm = _mm()
    p = f"L{il}"
    W_r = tape.commit_lazy(f"{p}_Wr", _field_loader(gguf, f"blk.{il}.ffn_gate_inp.weight",
                                                    transpose=True), (d, E), d * E)
    r = tape.matmul(n2g, W_r, **mm)
    s = tape.paired_tlookup(r, sig_tbl, shift=1 << (K2_INT["sig_bits"] - 1))
    bias_name = f"blk.{il}.exp_probs_b.bias"
    b = tape.commit_lazy(f"{p}_bias", ints_loader(lambda: bias_ints(by_name(gguf), il, S_SEL),
                                                  provenance=gguf_provenance(gguf, bias_name)),
                         (E,), E)
    width = topk_width(S_SEL, bias_ints(by, il, S_SEL), E)

    def shard(kind, e, shape):
        src = 1 if (wrong_slice and kind == "gate_exps" and e == 0) else e
        ld = maverick_lazy_expert(gguf, il, kind, src, S)
        name = f"{p}_{kind}{e}"
        if bridge:
            return tape.external_lazy(name, ld, shape, shape[0] * shape[1])
        return tape.commit_lazy(name, ld, shape, shape[0] * shape[1])
    w = {"gate": [shard("gate_exps", e, (d, d_ff)) for e in range(E)],
         "up": [shard("up_exps", e, (d, d_ff)) for e in range(E)],
         "down": [shard("down_exps", e, (d_ff, d)) for e in range(E)]}
    tr = {}
    y = topk_moe_ffn(tape, n2g, s, b, w, T=T, E=E, k=kk, d=d, d_ff=d_ff, S=S, S_w=K2_INT["S_w"],
                     C=K2_INT["C"], score_bits=S_SEL.bit_length(), width=width,
                     output_width=OW, select_word_bits=K2_INT["select_word_bits"],
                     bracket_word_bits=K2_INT["bracket_word_bits"], use_bridge=bridge, trace=tr)
    d_sh = _rows(by[f"blk.{il}.ffn_gate_shexp.weight"])
    sh = _dense_ffn(tape, n2g, gguf, [f"blk.{il}.ffn_{k}_shexp.weight" for k in ("gate", "up", "down")],
                    d=d, d_ff=d_sh, tag=f"{p}_sh")
    trace.update({f"{p}.router": r, f"{p}.s": s, f"{p}.mask": tr["m"], f"{p}.gw": tr["gw"],
                  f"{p}.g": tr["g"], f"{p}.h": tr["h"], f"{p}.D": tr["D"],
                  f"{p}.y_raw": tr["y_raw"], f"{p}.y": tr["y"], f"{p}.sh": sh})
    return y, sh, dict(E=E, d_ff=d_ff, width=width)


@dataclasses.dataclass
class Built:
    tape: object
    trace: dict
    logits: object
    Sz: object
    handles: dict
    sum_pos: list
    layer_of: dict          # id(claim) -> layer index (or "embed" / "tail")
    info: dict


def build(gguf, prompt_ids, cont_ids, *, cfg=K2_MLA, layers=2, kk=8, offset=0, yarn=True,
          bridge=False, tamper=None, ligero=None):
    """The K2 tape over `layers` layers. `tamper` = "yarn-statement" builds the
    claims without YaRN; "wrong-slice" decodes expert 0's gate shard from
    expert 1's rows. Witness tampers are applied at prove time (tampering)."""
    from demo_maverick_full import UI, _field_loader, build_inputs
    from demo_maverick_moe import CFG, SILU_CFG
    from k2_loader import by_name, sigmoid_table
    from tape import Tape
    from unexplained_info import prove_unexplained_info
    assert SILU_CFG.s_x == S
    tape = Tape(ligero or CFG, silu_config=SILU_CFG, lazy=True)
    by = by_name(gguf)
    d = cfg["d"]
    V = int(by["token_embd.weight"].n_elements) // d
    T = len(prompt_ids) + len(cont_ids)
    layer_of, trace, info = {}, {}, dict(V=V, T=T, layers={})

    def mark(tag, n0):
        for c in tape.claims[n0:]:
            layer_of.setdefault(id(c), tag)
    n0 = len(tape.claims)
    E_wt = tape.commit_lazy("token_embd", _field_loader(gguf, "token_embd.weight"), (V, d), V * d)
    x, ind_mid, o_last = build_inputs(tape, gguf, E_wt, prompt_ids, cont_ids, V=V, d=d)
    trace["x0"] = x
    mark("embed", n0)
    sig_tbl = None
    for il in range(layers):
        n0 = len(tape.claims)
        p = f"L{il}"
        r1, n2g = build_mla(tape, x, gguf, il, T=T, cfg=cfg, offset=offset,
                            yarn=(yarn and tamper != "yarn-statement"), trace=trace)
        if f"blk.{il}.ffn_gate_inp.weight" in by:
            if sig_tbl is None:
                keys, ty, _ = sigmoid_table(S, S_SEL, K2_INT["sig_bits"])
                sig_tbl = tape.register_table("k2_sigmoid", T_data=keys, T_Y_data=ty)
            y, sh, li = build_moe(tape, n2g, gguf, il, sig_tbl, T=T, d=d, kk=kk, bridge=bridge,
                                  wrong_slice=(tamper == "wrong-slice"), trace=trace)
            x = (r1 + y) + sh
            info["layers"][il] = dict(kind="moe", **li)
        else:
            d_ff = _rows(by[f"blk.{il}.ffn_gate.weight"])
            f = _dense_ffn(tape, n2g, gguf, [f"blk.{il}.ffn_{k}.weight" for k in ("gate", "up", "down")],
                           d=d, d_ff=d_ff, tag=p)
            trace[f"{p}.ffn"] = f
            x = r1 + f
            info["layers"][il] = dict(kind="dense", d_ff=d_ff)
        trace[f"{p}.out"] = x
        mark(il, n0)
        log(f"layer {il} built ({info['layers'][il]['kind']}): {len(tape.claims)} claims so far")
    n0 = len(tape.claims)
    g_out = tape.commit_lazy("g_out", _field_loader(gguf, "output_norm.weight"), (d,), d)
    nfg = _norm_gain(tape, x, g_out, T=T, d=d)
    head = "output.weight" if "output.weight" in by else "token_embd.weight"
    W_lm = tape.commit_lazy("W_lm", _field_loader(gguf, head, transpose=True), (d, V), d * V)
    logits = tape.matmul(nfg, W_lm, **_mm())
    trace["final.ng"], trace["logits"] = nfg, logits
    ids = list(prompt_ids) + list(cont_ids)
    O_ext = tape.concat([ind_mid, o_last], (T, V))
    sum_pos = list(range(len(prompt_ids) - 1, T - 1))
    Sz, handles = prove_unexplained_info(tape, logits, ids[1:] + [ids[-1]], T=T, V=V,
                                         sum_positions=sum_pos, reveal=True, O_ext=O_ext, **UI)
    mark("tail", n0)
    log(f"built: {len(tape.claims)} claims, T={T}, V={V}, offset={offset}, "
        f"yarn={'off' if tamper == 'yarn-statement' or not yarn else 'on'}, "
        f"bridge={'on' if bridge else 'off'}" + (f", tamper={tamper}" if tamper else ""))
    return Built(tape, trace, logits, Sz, handles, sum_pos, layer_of, info)


# ---- check: exact agreement with the integer reference -----------------------------

def engine_values(b: Built):
    """The engine pass, keeping every traced intermediate, as signed int64 numpy."""
    import torch
    from max_claim import to_signed
    keep = {wt.var for wt in b.trace.values()} | {b.Sz.var}
    live = b.tape.run_engine_pass(free_intermediates=True, keep=keep)
    out = {n: to_signed(live[wt.var].reshape(-1)).cpu().numpy() for n, wt in b.trace.items()}
    sz = int(live[b.Sz.var].cpu().reshape(-1)[0])
    del live
    torch.cuda.empty_cache()
    return out, sz


def compare_exact(got: dict, ref: dict) -> list:
    """One row per traced name, in build order: equal, or the mismatch count
    and the first mismatch."""
    import numpy as np
    rows = []
    for name, g in got.items():
        if name not in ref:
            rows.append(dict(name=name, ok=False, why="not in the reference"))
            continue
        r = np.asarray(ref[name])
        r = (np.asarray([int(v) for v in r.ravel()], dtype=np.int64) if r.dtype == object
             else r.astype(np.int64).ravel())
        if r.size != g.size:
            rows.append(dict(name=name, ok=False, why=f"size {g.size} vs reference {r.size}"))
            continue
        bad = np.nonzero(g != r)[0]
        row = dict(name=name, ok=not len(bad), n=int(g.size), mismatches=int(len(bad)),
                   max_abs=int(np.abs(g).max()) if g.size else 0)
        if len(bad):
            i = int(bad[0])
            row.update(first=i, tape=int(g[i]), reference=int(r[i]))
        rows.append(row)
    return rows


def check(gguf, prompt_ids, cont_ids, *, cfg=K2_MLA, layers=2, kk=8, offset=0, ligero=None):
    """Engine pass against the integer reference. Returns the record."""
    import k2_int_reference as ki
    from k2_loader import GgufWeights
    b = build(gguf, prompt_ids, cont_ids, cfg=cfg, layers=layers, kk=kk, offset=offset,
              ligero=ligero)
    t0 = time.time()
    got, sz = engine_values(b)
    t_engine = time.time() - t0
    log(f"engine pass {t_engine:.1f}s, Sz={sz}")
    t0 = time.time()
    ref, ranges = ki.forward(list(prompt_ids) + list(cont_ids), GgufWeights(gguf, cfg=cfg), cfg,
                             layers=layers, offset=offset, kk=kk)
    t_ref = time.time() - t0
    rows = compare_exact(got, ref)
    first_bad = next((r for r in rows if not r["ok"]), None)
    log(f"integer reference {t_ref:.1f}s: {sum(r['ok'] for r in rows)}/{len(rows)} "
        f"intermediates EXACT" + (f"; first difference at {first_bad['name']}: {first_bad}"
                                  if first_bad else ""))
    viol = ranges.violations()
    log(f"ranges: {len(ranges.rows)} recorded, {len(viol)} outside their window"
        + (f": {sorted(viol)}" if viol else ""))
    return dict(mode="check", exact=rows, all_exact=first_bad is None, Sz=sz,
                ranges=ranges.as_dict(), engine_s=t_engine, reference_s=t_ref, info=b.info)


def fidelity(gguf, ids, *, cfg=K2_MLA, layers=2, kk=8, offset=0):
    """CPU only: the integer reference against the float reference."""
    import k2_float_reference as kf
    import k2_int_reference as ki
    from k2_loader import GgufWeights
    t0 = time.time()
    io, ranges = ki.forward(ids, GgufWeights(gguf, cfg=cfg), cfg, layers=layers,
                            offset=offset, kk=kk)
    Wf = GgufWeights(gguf, ints=False, cfg=cfg)
    fo = kf.forward(ids, Wf, cfg, layers=layers, offset=offset, k=kk)
    fid = kf.fidelity(io, fo, S, S_SEL)
    rd = {il: kf.routing_differences(io, fo, Wf, il, k=kk, S=S)
          for il in range(layers) if Wf.kind(il) == "moe"}
    log(f"fidelity ({time.time() - t0:.1f}s): logits rel {fid['logits']['rel_l2']:.3g}, "
        f"top-1 agreement {fid['logits.top1_agree']:.3f}")
    for il, r in rd.items():
        log(f"routing L{il}: end to end {r['end_to_end']['tokens_differing']}/"
            f"{r['end_to_end']['tokens']} tokens differ; same input "
            f"{r['same_input']['tokens_differing']}/{r['same_input']['tokens']}")
    return dict(mode="fidelity", fidelity=fid, routing=rd, ranges=ranges.as_dict())


# ---- prove: the honest proof, then the negatives against its enrollment -------------

@contextlib.contextmanager
def tampering(kind, cfg):
    """Witness tampers, applied wherever the claim's witness is computed."""
    import compute_fns
    if kind == "interleave":
        import torch
        w1 = cfg["d_nope"]

        def swap(t):
            u = t.contiguous().view(torch.int64).clone()
            a, c = int(u[0]), int(u[w1])
            u[0], u[w1] = c, a
            return u.view(torch.uint64)
        compute_fns.WITNESS_TAMPER[("HeadInterleaveClaim", "dst")] = swap
        try:
            yield
        finally:
            compute_fns.WITNESS_TAMPER.pop(("HeadInterleaveClaim", "dst"), None)
    elif kind == "yarn-witness":
        import claims
        orig = compute_fns._rope_cos_sin
        compute_fns._rope_cos_sin = lambda rc: claims._rope_cos_sin(dataclasses.replace(
            rc, yarn=False, scale_factor=1.0, original_max_pos=0))
        try:
            yield
        finally:
            compute_fns._rope_cos_sin = orig
    else:
        yield


def reveal(b: Built):
    """Discover Sz and pin it; re-zero the LogUp multiplicities."""
    import torch
    live = b.tape.run_engine_pass(free_intermediates=True, keep={b.Sz.var})
    sz = int(live[b.Sz.var].cpu().reshape(-1)[0])
    b.handles["reveal_pin"].public_rhs = sz
    for v in list(b.tape.inputs):
        if getattr(v, "name", "").endswith("_mult"):
            b.tape.inputs[v].zero_()
    del live
    torch.cuda.empty_cache()
    return sz


def rust_verify_anchored(claims, proof, cfg, *, root_w, stmt, wc_identity):
    """The Rust verifier under GIVEN anchors (not the proof's own)."""
    import protocol as pr
    from _rust_verify import _verify_proof_bin
    from proof_dump import dump_proof
    seeds = {k: v.hex() for k, v in proof.seeds.items()}
    Q = list(pr.random_columns(proof.seeds["s_col"], cfg))
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    os.unlink(path)
    argv = [_verify_proof_bin(), path, root_w.hex() if root_w else "-",
            stmt.hex() if stmt else "-", wc_identity.hex() if wc_identity else "-"]
    try:
        dump_proof(path, pr.claims_to_json(claims, cfg), seeds, proof, Q, None)
        r = subprocess.run(argv, capture_output=True, text=True)
    finally:
        if os.path.exists(path):
            os.unlink(path)
    return "rust_verify: ACCEPT" in r.stdout, (r.stdout + r.stderr).strip()


def claim_memory(b: Built):
    """core.CLAIM_MEM_LOG summarized per sweep: the peak, and the allocation
    and projection bytes at the first claim of each layer (the growth §3
    extrapolates)."""
    import core
    per = {}
    for sweep, label, i, typ, cid, alloc, held in core.CLAIM_MEM_LOG:
        rec = per.setdefault(sweep, dict(label=label, peak=0, at_layer={}))
        rec["peak"] = max(rec["peak"], alloc)
        tag = b.layer_of.get(cid)
        if tag is not None and str(tag) not in rec["at_layer"]:
            rec["at_layer"][str(tag)] = dict(allocated=alloc, projections=held)
    return per


def research_prove(gguf, prompt_ids, cont_ids, *, cfg=K2_MLA, layers=2, kk=8, offset=0,
                   bridge=False, negatives=(), dump=None, ligero=None, wc_params=None):
    import torch
    import core
    from demo_maverick_moe import CFG
    lig = ligero or CFG
    kw = dict(cfg=cfg, layers=layers, kk=kk, offset=offset, bridge=bridge, ligero=lig)
    rec = dict(mode="prove", bridge=bridge, offset=offset, T=len(prompt_ids) + len(cont_ids))
    t0 = time.time()
    hon = build(gguf, prompt_ids, cont_ids, **kw)
    rec["build_s"] = time.time() - t0
    t0 = time.time()
    wc = core.WeightCommitment.from_tape(hon.tape, lig)
    rec["enroll_s"] = time.time() - t0
    wc_enr = None
    if bridge:
        import wc_bridge as wcb
        t0 = time.time()
        wc_enr = wcb.lazy_enroll_tape(hon.tape, b"k2-gate-mask",
                                      f"kimi-k2|{gguf}|S={S}".encode(),
                                      wc_params or wcb.WcParams())
        rec["wc_enroll_s"] = time.time() - t0
        log(f"bridge enrollment {rec['wc_enroll_s']:.1f}s, identity {wc_enr.identity().hex()[:16]}…")
    log(f"enrolled {wc.m_w} weight rows in {rec['enroll_s']:.1f}s, root {wc.root.hex()[:16]}…")
    t0 = time.time()
    rec["Sz"] = reveal(hon)
    rec["reveal_s"] = time.time() - t0
    core.CLAIM_MEM_LOG.clear()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    proof = hon.tape.prove(weight_commitment=wc, weight_enrollment=wc_enr)
    rec["prove_s"] = time.time() - t0
    rec["peak_gpu_GiB"] = torch.cuda.max_memory_allocated() / 2 ** 30
    rec["claim_memory"] = claim_memory(hon)
    log(f"PROVE {rec['prove_s']:.1f}s, peak GPU {rec['peak_gpu_GiB']:.2f} GiB")
    anchors = dict(root_w=proof.root_w, stmt=proof.statement_digest,
                   wc_identity=(proof.wc_bridge["identity"] if bridge else None))
    assert anchors["root_w"] == wc.root, "the proof's weight root is not the enrollment's"
    if dump:
        from instrumented_prove import write_dump_policy
        from proof_dump import dump_proof
        t0 = time.time()
        dump_proof(dump, None, None, proof, None, None, u64_encoding="u64le-base64")
        rec["dump"] = dict(path=dump, bytes=os.path.getsize(dump), s=time.time() - t0,
                           policy=write_dump_policy(dump, proof))
    t0 = time.time()
    acc, msg = rust_verify_anchored(hon.tape.claims, proof, lig, **anchors)
    rec["verify"] = dict(accept=acc, s=time.time() - t0)
    log(f"rust verify_proof (honest, same-run anchors): {'ACCEPT' if acc else 'REJECT'} "
        f"in {rec['verify']['s']:.1f}s")
    if not acc:
        log(msg[-3000:])
        rec["verify"]["tail"] = msg[-3000:]
        return rec
    del proof
    torch.cuda.empty_cache()
    rec["negatives"] = {}
    for neg in negatives:
        if neg == "wrong-slice" and not bridge:
            raise SystemExit("wrong-slice needs --bridge: the bridge enrollment is its anchor")
        t0 = time.time()
        bt = build(gguf, prompt_ids, cont_ids, tamper=neg, **kw)
        with tampering(neg, cfg):
            sz = reveal(bt)
            try:
                p = bt.tape.prove(weight_commitment=wc, weight_enrollment=wc_enr)
            except Exception as e:                     # the prover refused: also a rejection
                rec["negatives"][neg] = dict(rejected=True, by="prover",
                                             why=f"{type(e).__name__}: {e}"[:2000])
                log(f"negative {neg}: the prover refused ({type(e).__name__})")
                continue
        stmt = anchors["stmt"] if neg == "yarn-statement" else p.statement_digest
        acc, msg = rust_verify_anchored(bt.tape.claims, p, lig, root_w=anchors["root_w"],
                                        stmt=stmt, wc_identity=anchors["wc_identity"])
        row = dict(rejected=not acc, by="rust", Sz=sz, s=time.time() - t0,
                   tail=msg[-1500:], same_statement=(p.statement_digest == anchors["stmt"]))
        if neg == "yarn-statement":                    # contrast: its own digest
            own, _ = rust_verify_anchored(bt.tape.claims, p, lig, root_w=anchors["root_w"],
                                          stmt=p.statement_digest,
                                          wc_identity=anchors["wc_identity"])
            row["accept_under_own_digest"] = own
        rec["negatives"][neg] = row
        log(f"negative {neg}: {'REJECT' if not acc else 'ACCEPT (!)'} under the honest "
            f"anchors in {row['s']:.1f}s")
        del p, bt
        torch.cuda.empty_cache()
    return rec


# ---- the command line ----------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mode", choices=("check", "fidelity", "prove"), required=True)
    ap.add_argument("--from-gguf", required=True)
    ap.add_argument("--prompt-n", type=int, default=50)
    ap.add_argument("--cont-n", type=int, default=50)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--offset", type=int, default=0, help="first position (long-position arm: 130072)")
    ap.add_argument("--t-queries", type=int, default=54)
    ap.add_argument("--bridge", action="store_true")
    ap.add_argument("--weight-cache", action="store_true")
    ap.add_argument("--routed-cache", action="store_true")
    ap.add_argument("--sweep-timing", action="store_true")
    ap.add_argument("--negatives", default="", help=f"comma list of {', '.join(NEGATIVES)}")
    ap.add_argument("--dump-proof", default=None)
    ap.add_argument("--record", default=None, help="write the run's record (JSON) here")
    ap.add_argument("--allow-env-geometry", action="store_true")
    a = ap.parse_args(argv)
    negs = [n for n in a.negatives.split(",") if n]
    bad = [n for n in negs if n not in NEGATIVES]
    if bad:
        raise SystemExit(f"unknown negatives {bad}; choose from {NEGATIVES}")
    if a.record and os.path.exists(a.record):
        raise SystemExit(f"{a.record} exists; name a new record")
    if a.dump_proof and os.path.exists(a.dump_proof):
        raise SystemExit(f"{a.dump_proof} exists; not overwriting a proof")

    # before any demo import: the configs read the geometry at import time
    os.environ["LIGERO_T_QUERIES"] = str(a.t_queries)
    for k, v in (("LIGERO_ELL", "8192"), ("LIGERO_K_DEG", "16384"), ("LIGERO_N_LIG", "65536")):
        if a.allow_env_geometry:
            os.environ.setdefault(k, v)
        else:
            os.environ[k] = v
    os.environ.setdefault("LIGERO_PHASE_TIMING", "1")
    if a.sweep_timing:
        os.environ["LIGERO_SWEEP_TIMING"] = "1"
    os.environ["LIGERO_ROUTED_Y_CACHE"] = "1" if a.routed_cache else "0"
    os.environ["LIGERO_WEIGHT_CACHE"] = "1" if a.weight_cache else "0"
    os.environ["LIGERO_CLAIM_MEM"] = "1" if a.mode == "prove" else "0"

    from k2_loader import by_name
    V = int(by_name(a.from_gguf)["token_embd.weight"].n_elements) // K2_MLA["d"]
    import numpy as np
    rng = np.random.default_rng(11)
    prompt_ids = rng.integers(0, V, a.prompt_n).tolist()
    cont_ids = rng.integers(0, V, a.cont_n).tolist()
    log(f"RESEARCH run, mode {a.mode}: T={a.prompt_n + a.cont_n} (synthetic token ids), "
        f"layers={a.layers}, k={a.top_k}, offset={a.offset}, T_QUERIES={a.t_queries}")
    kw = dict(layers=a.layers, kk=a.top_k, offset=a.offset)
    if a.mode == "check":
        rec = check(a.from_gguf, prompt_ids, cont_ids, **kw)
        ok = rec["all_exact"]
    elif a.mode == "fidelity":
        rec = fidelity(a.from_gguf, prompt_ids + cont_ids, **kw)
        ok = True
    else:
        rec = research_prove(a.from_gguf, prompt_ids, cont_ids, bridge=a.bridge,
                             negatives=negs, dump=a.dump_proof, **kw)
        ok = rec["verify"]["accept"] and all(r["rejected"] for r in rec.get("negatives", {}).values())
    rec["args"] = vars(a)
    if a.record:
        with open(a.record, "x") as fh:
            json.dump(rec, fh, indent=1, default=str)
        log(f"record written to {a.record}")
    log("RESULT " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
