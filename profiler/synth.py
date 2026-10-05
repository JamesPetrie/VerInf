"""Synthetic manifest builders: Maverick, Llama-2-7B and Kimi K2, closed form.

Mirrors the claim structure of demo_maverick_full.py / demo_llama7b.py the way
analysis/maverick_cost_model.py does, but emits one ClaimRecord per tape op
(per-expert matmuls individually) with producer/consumer variable edges, so
the DAG and live-set analyses see the real fan-out — notably the E-way expert
parallelism per MoE layer.

This is the no-torch path: it exists so prediction can run anywhere and as a
cross-check of the tape extractor. The extractor (extract.py, run where the
prover runs) is the ground truth; when the two disagree, trust the tape.
Model shapes here are the team-standard workloads; any other model should
come in through the extractor, not by adding builders here. Kimi K2 is here
before its demo exists because the group agreed it as the next model
(2026-09-23) and the profiler is how it is sized; its builder carries the
modeling assumptions its docstring lists, which the K2 demo replaces.
"""
from __future__ import annotations

from manifest import Manifest, ClaimRecord, VariableRecord

LIGERO = dict(ELL=8192, K_DEG=16384, N_LIG=65536)


class _Builder:
    """Mini-tape: emit(claim) allocates output vars and records edges."""

    def __init__(self):
        self.claims: list[ClaimRecord] = []
        self.vars: dict[str, VariableRecord] = {}

    def input_var(self, name, length, *, persistent=False, external=False):
        self.vars[name] = VariableRecord(name=name, length=length,
                                         persistent=persistent, producer=None,
                                         external=external)
        return name

    def emit(self, type_, params, *, label="", layer=None, inputs=(),
             out=None, out_len=0, more=(), phase2=()):
        """`out` is the claim's main output; `more` lists further phase-1
        outputs as (name, length); `phase2` lists phase-2 outputs."""
        idx = len(self.claims)
        outputs = []
        if out is not None:
            self.vars[out] = VariableRecord(name=out, length=int(out_len),
                                            producer=idx)
            outputs.append(out)
        for name, length in more:
            self.vars[name] = VariableRecord(name=name, length=int(length),
                                             producer=idx)
            outputs.append(name)
        for name, length in phase2:
            self.vars[name] = VariableRecord(name=name, length=int(length),
                                             phase=2, producer=idx)
            outputs.append(name)
        for v in inputs:
            self.vars[v].consumers.append(idx)
        self.claims.append(ClaimRecord(
            idx=idx, type=type_, label=label, layer=layer, params=params,
            inputs=list(inputs), outputs=outputs))
        return out


def _attention(b: _Builder, x, il, S, d, H, dh, use_rope, prefix):
    """Norm + gain + QKV + (RoPE) + scores + softmax + AV + O + residual.
    Returns the post-attention residual variable."""
    Ld = S * d
    # Gains are plain (non-persistent) commits in the demos — witness rows,
    # not streamed weights.
    g1 = b.input_var(f"{prefix}.gain1", d)
    g2 = b.input_var(f"{prefix}.gain2", d)
    n1 = b.emit("rmsnorm", dict(B=S, d=d), label=f"{prefix}.norm1", layer=il,
                inputs=[x], out=f"{prefix}.n1", out_len=Ld)
    b.emit("embed_lookup", dict(L=Ld), label=f"{prefix}.gain1.bcast", layer=il,
           inputs=[g1], out=f"{prefix}.g1b", out_len=Ld)
    n1g = b.emit("hadamard", dict(L=Ld), label=f"{prefix}.gain1", layer=il,
                 inputs=[n1, f"{prefix}.g1b"], out=f"{prefix}.n1g", out_len=Ld)
    qkv = {}
    for w in ("q", "k", "v"):
        wv = b.input_var(f"{prefix}.W_{w}", d * d, persistent=True)
        qkv[w] = b.emit("matmul", dict(m=S, k=d, n=d), layer=il,
                        label=f"{prefix}.{w}_proj", inputs=[n1g, wv],
                        out=f"{prefix}.{w}", out_len=Ld)
    if use_rope:
        qkv["q"] = b.emit("rope", dict(L=Ld), label=f"{prefix}.rope_q",
                          layer=il, inputs=[qkv["q"]],
                          out=f"{prefix}.q_rope", out_len=Ld)
        qkv["k"] = b.emit("rope", dict(L=Ld), label=f"{prefix}.rope_k",
                          layer=il, inputs=[qkv["k"]],
                          out=f"{prefix}.k_rope", out_len=Ld)
    scores = b.emit("matmul", dict(m=S, k=d, n=S, heads=H), layer=il,
                    label=f"{prefix}.scores", inputs=[qkv["q"], qkv["k"]],
                    out=f"{prefix}.scores", out_len=H * S * S)
    sm = b.emit("softmax", dict(B=H * S, M=S, causal=True), layer=il,
                label=f"{prefix}.softmax", inputs=[scores],
                out=f"{prefix}.sm", out_len=H * S * S)
    av = b.emit("matmul", dict(m=S, k=H * S, n=dh, heads=H), layer=il,
                label=f"{prefix}.attnV", inputs=[sm, qkv["v"]],
                out=f"{prefix}.av", out_len=Ld)
    wo = b.input_var(f"{prefix}.W_o", d * d, persistent=True)
    proj = b.emit("matmul", dict(m=S, k=d, n=d), layer=il,
                  label=f"{prefix}.o_proj", inputs=[av, wo],
                  out=f"{prefix}.proj", out_len=Ld)
    r1 = b.emit("add", dict(L=Ld), label=f"{prefix}.resid1", layer=il,
                inputs=[x, proj], out=f"{prefix}.r1", out_len=Ld)
    n2 = b.emit("rmsnorm", dict(B=S, d=d), label=f"{prefix}.norm2", layer=il,
                inputs=[r1], out=f"{prefix}.n2", out_len=Ld)
    b.emit("embed_lookup", dict(L=Ld), label=f"{prefix}.gain2.bcast", layer=il,
           inputs=[g2], out=f"{prefix}.g2b", out_len=Ld)
    n2g = b.emit("hadamard", dict(L=Ld), label=f"{prefix}.gain2", layer=il,
                 inputs=[n2, f"{prefix}.g2b"], out=f"{prefix}.n2g", out_len=Ld)
    return r1, n2g


def _dense_ffn(b, r1, n2g, il, S, d, d_ff, prefix):
    Ld = S * d
    wg = b.input_var(f"{prefix}.W_gate", d * d_ff, persistent=True)
    wu = b.input_var(f"{prefix}.W_up", d * d_ff, persistent=True)
    wd = b.input_var(f"{prefix}.W_down", d_ff * d, persistent=True)
    gate = b.emit("matmul", dict(m=S, k=d, n=d_ff), label=f"{prefix}.gate",
                  layer=il, inputs=[n2g, wg], out=f"{prefix}.gate", out_len=S * d_ff)
    up = b.emit("matmul", dict(m=S, k=d, n=d_ff), label=f"{prefix}.up",
                layer=il, inputs=[n2g, wu], out=f"{prefix}.up", out_len=S * d_ff)
    sg = b.emit("silu", dict(L=S * d_ff), label=f"{prefix}.silu", layer=il,
                inputs=[gate], out=f"{prefix}.sg", out_len=S * d_ff)
    inter = b.emit("hadamard", dict(L=S * d_ff), label=f"{prefix}.inter",
                   layer=il, inputs=[sg, up], out=f"{prefix}.inter",
                   out_len=S * d_ff)
    down = b.emit("matmul", dict(m=S, k=d_ff, n=d), label=f"{prefix}.down",
                  layer=il, inputs=[inter, wd], out=f"{prefix}.down", out_len=Ld)
    return b.emit("add", dict(L=Ld), label=f"{prefix}.resid2", layer=il,
                  inputs=[r1, down], out=f"{prefix}.r2", out_len=Ld)


def _moe_ffn(b, r1, n2g, il, S, d, d_ff, E, bc_ones, prefix):
    """Router + all-E committed expert matmuls + Freivalds sums + shared expert."""
    Ld = S * d
    wr = b.input_var(f"{prefix}.W_router", d * E, persistent=True)
    router = b.emit("matmul", dict(m=S, k=d, n=E), label=f"{prefix}.router",
                    layer=il, inputs=[n2g, wr], out=f"{prefix}.router",
                    out_len=S * E)
    b.emit("routing", dict(T=S, E=E, n_words=3), label=f"{prefix}.route",
           layer=il, inputs=[router], out=f"{prefix}.mask", out_len=S * E)
    b.emit("ptlookup", dict(L=S), label=f"{prefix}.sigma", layer=il,
           inputs=[router], out=f"{prefix}.sig", out_len=S)
    # sigma broadcast via freivalds_combine(E=1) over bc_ones, per the demo.
    b.emit("freivalds_combine", dict(T=S, E=1, F=d), label=f"{prefix}.s_rep",
           layer=il, inputs=[f"{prefix}.sig", bc_ones],
           out=f"{prefix}.srep", out_len=Ld)
    xr = b.emit("hadamard", dict(L=Ld), label=f"{prefix}.x_r", layer=il,
                inputs=[n2g, f"{prefix}.srep"], out=f"{prefix}.xr", out_len=Ld)
    # All-E expert matmuls on the same committed x_r — the E-way parallel fan.
    for kind, kk, nn in (("gate", d, d_ff), ("up", d, d_ff)):
        for e in range(E):
            we = b.input_var(f"{prefix}.e{e}.W_{kind}", kk * nn, persistent=True)
            b.emit("matmul", dict(m=S, k=kk, n=nn), layer=il,
                   label=f"{prefix}.e{e}.{kind}", inputs=[xr, we],
                   out=f"{prefix}.e{e}.{kind}", out_len=S * nn)
        b.emit("freivalds_combine", dict(T=S, E=E, F=nn), layer=il,
               label=f"{prefix}.{kind}_sum",
               inputs=[f"{prefix}.mask"] + [f"{prefix}.e{e}.{kind}" for e in range(E)],
               out=f"{prefix}.{kind}_sum", out_len=S * nn)
    sg = b.emit("silu", dict(L=S * d_ff), label=f"{prefix}.silu", layer=il,
                inputs=[f"{prefix}.gate_sum"], out=f"{prefix}.sg", out_len=S * d_ff)
    hidden = b.emit("hadamard", dict(L=S * d_ff), label=f"{prefix}.hidden",
                    layer=il, inputs=[sg, f"{prefix}.up_sum"],
                    out=f"{prefix}.hidden", out_len=S * d_ff)
    for e in range(E):
        we = b.input_var(f"{prefix}.e{e}.W_down", d_ff * d, persistent=True)
        b.emit("matmul", dict(m=S, k=d_ff, n=d), layer=il,
               label=f"{prefix}.e{e}.down", inputs=[hidden, we],
               out=f"{prefix}.e{e}.down", out_len=Ld)
    ffn = b.emit("freivalds_combine", dict(T=S, E=E, F=d), layer=il,
                 label=f"{prefix}.ffn_sum",
                 inputs=[f"{prefix}.mask"] + [f"{prefix}.e{e}.down" for e in range(E)],
                 out=f"{prefix}.ffn", out_len=Ld)
    # Shared expert (always active).
    swg = b.input_var(f"{prefix}.sh.W_gate", d * d_ff, persistent=True)
    swu = b.input_var(f"{prefix}.sh.W_up", d * d_ff, persistent=True)
    swd = b.input_var(f"{prefix}.sh.W_down", d_ff * d, persistent=True)
    g = b.emit("matmul", dict(m=S, k=d, n=d_ff), label=f"{prefix}.sh.gate",
               layer=il, inputs=[n2g, swg], out=f"{prefix}.sh.g", out_len=S * d_ff)
    u = b.emit("matmul", dict(m=S, k=d, n=d_ff), label=f"{prefix}.sh.up",
               layer=il, inputs=[n2g, swu], out=f"{prefix}.sh.u", out_len=S * d_ff)
    sgs = b.emit("silu", dict(L=S * d_ff), label=f"{prefix}.sh.silu", layer=il,
                 inputs=[g], out=f"{prefix}.sh.sg", out_len=S * d_ff)
    hs = b.emit("hadamard", dict(L=S * d_ff), label=f"{prefix}.sh.hidden",
                layer=il, inputs=[sgs, u], out=f"{prefix}.sh.h", out_len=S * d_ff)
    shd = b.emit("matmul", dict(m=S, k=d_ff, n=d), label=f"{prefix}.sh.down",
                 layer=il, inputs=[hs, swd], out=f"{prefix}.sh.d", out_len=Ld)
    a1 = b.emit("add", dict(L=Ld), label=f"{prefix}.resid2a", layer=il,
                inputs=[r1, ffn], out=f"{prefix}.r2a", out_len=Ld)
    return b.emit("add", dict(L=Ld), label=f"{prefix}.resid2b", layer=il,
                  inputs=[a1, shd], out=f"{prefix}.r2", out_len=Ld)


def _moe_ffn_projected(b, r1, n2g, il, S, d, d_ff, E, bc_ones, prefix):
    """MoE FFN under the routed-projected protocol, mirroring the current
    demo_maverick_full.build_moe_ffn: router/route/sigma/s_rep/x_r and the
    shared expert are byte-identical to _moe_ffn; the all-E expert fan and
    its three freivalds_combine calls are replaced by one
    routed_projected + rescale_claim pair per expert matrix (gate/up/down),
    with silu/hadamard on the SELECTED (S x d_ff) outputs (as _moe_ffn
    already had them — the projected builder changes the expert matmuls
    and combines, not those). Per-expert
    weight variables stay: one persistent var per expert shard, exactly
    what the enrolled weight block holds."""
    Ld = S * d
    wr = b.input_var(f"{prefix}.W_router", d * E, persistent=True)
    router = b.emit("matmul", dict(m=S, k=d, n=E), label=f"{prefix}.router",
                    layer=il, inputs=[n2g, wr], out=f"{prefix}.router",
                    out_len=S * E)
    b.emit("routing", dict(T=S, E=E, n_words=3), label=f"{prefix}.route",
           layer=il, inputs=[router], out=f"{prefix}.mask", out_len=S * E)
    b.emit("ptlookup", dict(L=S), label=f"{prefix}.sigma", layer=il,
           inputs=[router], out=f"{prefix}.sig", out_len=S)
    b.emit("freivalds_combine", dict(T=S, E=1, F=d), label=f"{prefix}.s_rep",
           layer=il, inputs=[f"{prefix}.sig", bc_ones],
           out=f"{prefix}.srep", out_len=Ld)
    xr = b.emit("hadamard", dict(L=Ld), label=f"{prefix}.x_r", layer=il,
                inputs=[n2g, f"{prefix}.srep"], out=f"{prefix}.xr", out_len=Ld)

    def routed(kind, x_in, K, J):
        ws = [b.input_var(f"{prefix}.e{e}.W_{kind}", K * J, persistent=True)
              for e in range(E)]
        raw = b.emit("routed_projected", dict(T=S, K=K, J=J, E=E), layer=il,
                     label=f"{prefix}.{kind}_rp",
                     inputs=[x_in, f"{prefix}.mask"] + ws,
                     out=f"{prefix}.{kind}_raw", out_len=S * J)
        # s_in = S*S -> s_out = S at S = 2^12: rescale_bits 12; demo
        # OUTPUT_WIDTH = 26
        return b.emit("rescale_claim",
                      dict(length=S * J, rescale_bits=12, output_width=26),
                      layer=il, label=f"{prefix}.{kind}_rs", inputs=[raw],
                      out=f"{prefix}.{kind}", out_len=S * J)

    g_sum = routed("gate", xr, d, d_ff)
    sg = b.emit("silu", dict(L=S * d_ff), label=f"{prefix}.silu", layer=il,
                inputs=[g_sum], out=f"{prefix}.sg", out_len=S * d_ff)
    up_sum = routed("up", xr, d, d_ff)
    hidden = b.emit("hadamard", dict(L=S * d_ff), label=f"{prefix}.hidden",
                    layer=il, inputs=[sg, up_sum],
                    out=f"{prefix}.hiddenv", out_len=S * d_ff)
    ffn = routed("down", hidden, d_ff, d)
    # Shared expert (always active) — unchanged from _moe_ffn.
    swg = b.input_var(f"{prefix}.sh.W_gate", d * d_ff, persistent=True)
    swu = b.input_var(f"{prefix}.sh.W_up", d * d_ff, persistent=True)
    swd = b.input_var(f"{prefix}.sh.W_down", d_ff * d, persistent=True)
    g = b.emit("matmul", dict(m=S, k=d, n=d_ff), label=f"{prefix}.sh.gate",
               layer=il, inputs=[n2g, swg], out=f"{prefix}.sh.g", out_len=S * d_ff)
    u = b.emit("matmul", dict(m=S, k=d, n=d_ff), label=f"{prefix}.sh.up",
               layer=il, inputs=[n2g, swu], out=f"{prefix}.sh.u", out_len=S * d_ff)
    sgs = b.emit("silu", dict(L=S * d_ff), label=f"{prefix}.sh.silu", layer=il,
                 inputs=[g], out=f"{prefix}.sh.sg", out_len=S * d_ff)
    hs = b.emit("hadamard", dict(L=S * d_ff), label=f"{prefix}.sh.hidden",
                inputs=[sgs, u], layer=il, out=f"{prefix}.sh.h", out_len=S * d_ff)
    shd = b.emit("matmul", dict(m=S, k=d_ff, n=d), label=f"{prefix}.sh.down",
                 layer=il, inputs=[hs, swd], out=f"{prefix}.sh.d", out_len=Ld)
    a1 = b.emit("add", dict(L=Ld), label=f"{prefix}.resid2a", layer=il,
                inputs=[r1, ffn], out=f"{prefix}.r2a", out_len=Ld)
    return b.emit("add", dict(L=Ld), label=f"{prefix}.resid2b", layer=il,
                  inputs=[a1, shd], out=f"{prefix}.r2", out_len=Ld)


def maverick(seq: int, t_queries: int = 40) -> Manifest:
    """48-layer Llama-4-Maverick: 24 dense (even, RoPE) + 24 MoE (odd,
    alternating RoPE/NoPE), all E=128 experts committed, shared expert,
    token-select embed + final norm/gain + LM head. Shapes per
    demo_maverick_full.py."""
    d, dff_d, dff_e, E, H, dh, V = 5120, 16384, 8192, 128, 40, 128, 202048
    b = _Builder()
    emb = b.input_var("embed.W", V * d, persistent=True)
    x = b.emit("matmul", dict(m=seq, k=V, n=d, rescale=False),
               label="embed.select", inputs=[emb], out="x0", out_len=seq * d)
    ones = b.input_var("bc_ones", seq * d)   # shared sigma-broadcast operand
    for il in range(48):
        prefix = f"L{il}"
        if il % 2 == 0:
            r1, n2g = _attention(b, x, il, seq, d, H, dh, True, prefix)
            x = _dense_ffn(b, r1, n2g, il, seq, d, dff_d, prefix)
        else:
            use_rope = (il % 4 == 1)   # il = 1,5,9,... RoPE; 3,7,11,... NoPE
            r1, n2g = _attention(b, x, il, seq, d, H, dh, use_rope, prefix)
            x = _moe_ffn(b, r1, n2g, il, seq, d, dff_e, E, ones, prefix)
    gf = b.input_var("final.gain", d)
    fn = b.emit("rmsnorm", dict(B=seq, d=d), label="final.norm", inputs=[x],
                out="final.n", out_len=seq * d)
    b.emit("embed_lookup", dict(L=seq * d), label="final.gain.bcast",
           inputs=[gf], out="final.gb", out_len=seq * d)
    fng = b.emit("hadamard", dict(L=seq * d), label="final.gain",
                 inputs=[fn, "final.gb"], out="final.ng", out_len=seq * d)
    head = b.input_var("head.W", d * V, persistent=True)
    b.emit("matmul", dict(m=seq, k=d, n=V), label="lm_head",
           inputs=[fng, head], out="logits", out_len=seq * V)
    return Manifest(
        source=dict(kind="synth", generator="synth.maverick"),
        model=dict(name="llama4-maverick", d=d, d_ff_dense=dff_d,
                   d_ff_expert=dff_e, experts=E, heads=H, head_dim=dh,
                   vocab=V, layers=48),
        run=dict(seq=seq, ligero=dict(T_QUERIES=t_queries, **LIGERO)),
        claims=b.claims, variables=list(b.vars.values()))


def llama7b(seq: int, layers: int = 32, t_queries: int = 80) -> Manifest:
    """Llama-2-7B per demo_llama7b.py: 32 dense RoPE layers + final norm +
    LM head; every matmul/hadamard/rope/rmsnorm output-rescaled."""
    d, d_ff, H, dh, V = 4096, 11008, 32, 128, 32000
    b = _Builder()
    x = b.input_var("x0", seq * d)   # committed embeddings enter as run input
    for il in range(layers):
        prefix = f"L{il}"
        r1, n2g = _attention(b, x, il, seq, d, H, dh, True, prefix)
        x = _dense_ffn(b, r1, n2g, il, seq, d, d_ff, prefix)
    gf = b.input_var("final.gain", d)
    fn = b.emit("rmsnorm", dict(B=seq, d=d), label="final.norm", inputs=[x],
                out="final.n", out_len=seq * d)
    b.emit("embed_lookup", dict(L=seq * d), label="final.gain.bcast",
           inputs=[gf], out="final.gb", out_len=seq * d)
    fng = b.emit("hadamard", dict(L=seq * d), label="final.gain",
                 inputs=[fn, "final.gb"], out="final.ng", out_len=seq * d)
    head = b.input_var("head.W", d * V, persistent=True)
    b.emit("matmul", dict(m=seq, k=d, n=V), label="lm_head",
           inputs=[fng, head], out="logits", out_len=seq * V)
    return Manifest(
        source=dict(kind="synth", generator="synth.llama7b"),
        model=dict(name="llama2-7b", d=d, d_ff=d_ff, heads=H, head_dim=dh,
                   vocab=V, layers=layers),
        run=dict(seq=seq, ligero=dict(T_QUERIES=t_queries, **LIGERO)),
        claims=b.claims, variables=list(b.vars.values()))


def maverick_projected(seq: int, t_queries: int = 54) -> Manifest:
    """Maverick under the routed-projected protocol (kept-trees main):
    identical to maverick() except every MoE layer uses
    _moe_ffn_projected. t_queries defaults to 54, the demonstrated 400B
    run's setting. Pair with predict/partition --enrolled-weights —
    per-expert weight vars are the enrolled block."""
    d, dff_d, dff_e, E, H, dh, V = 5120, 16384, 8192, 128, 40, 128, 202048
    b = _Builder()
    emb = b.input_var("embed.W", V * d, persistent=True)
    x = b.emit("matmul", dict(m=seq, k=V, n=d, rescale=False),
               label="embed.select", inputs=[emb], out="x0", out_len=seq * d)
    ones = b.input_var("bc_ones", seq * d)
    for il in range(48):
        prefix = f"L{il}"
        if il % 2 == 0:
            r1, n2g = _attention(b, x, il, seq, d, H, dh, True, prefix)
            x = _dense_ffn(b, r1, n2g, il, seq, d, dff_d, prefix)
        else:
            use_rope = (il % 4 == 1)
            r1, n2g = _attention(b, x, il, seq, d, H, dh, use_rope, prefix)
            x = _moe_ffn_projected(b, r1, n2g, il, seq, d, dff_e, E, ones,
                                   prefix)
    gf = b.input_var("final.gain", d)
    fn = b.emit("rmsnorm", dict(B=seq, d=d), label="final.norm", inputs=[x],
                out="final.n", out_len=seq * d)
    b.emit("embed_lookup", dict(L=seq * d), label="final.gain.bcast",
           inputs=[gf], out="final.gb", out_len=seq * d)
    fng = b.emit("hadamard", dict(L=seq * d), label="final.gain",
                 inputs=[fn, "final.gb"], out="final.ng", out_len=seq * d)
    head = b.input_var("head.W", d * V, persistent=True)
    b.emit("matmul", dict(m=seq, k=d, n=V), label="lm_head",
           inputs=[fng, head], out="logits", out_len=seq * V)
    return Manifest(
        source=dict(kind="synth", generator="synth.maverick_projected"),
        model=dict(name="llama4-maverick-projected", d=d, d_ff_dense=dff_d,
                   d_ff_expert=dff_e, experts=E, heads=H, head_dim=dh,
                   vocab=V, layers=48),
        run=dict(seq=seq, ligero=dict(T_QUERIES=t_queries, **LIGERO)),
        claims=b.claims, variables=list(b.vars.values()))


# ---------------------------------------------------------------- top-k

def threshold_words(width: int, word_bits: int) -> int:
    """Word count of the dominance range in route_topk: ceil(width /
    word_bits) (prover/topk_params.threshold_words, which also enforces the
    threshold guard)."""
    return max(1, -(-width // word_bits))


def bracket_words(score_bits: int, k: int, C: int, word_bits: int):
    """(words on rem and on Z - 1 - rem, words on w), as gate_bracket sizes
    them (prover/topk_routing.py): Z < 2^(score_bits + ceil(log2(k + 1))),
    w < 2^C.bit_length()."""
    z_bits = score_bits + max(1, _ceil_log2(k + 1))
    n_rem = max(1, -(-z_bits // word_bits))
    n_w = max(1, -(-C.bit_length() // word_bits))
    return n_rem, n_w


def _ceil_log2(n: int) -> int:
    return (n - 1).bit_length()


def _ranged(b, x_name, L, n_words, il, label):
    """word_extract + one range_word per word, as tape.word_extract emits."""
    words = [f"{label}.w{i}" for i in range(n_words)]
    b.emit("word_extract", dict(L=L, n_words=n_words), layer=il, label=label,
           inputs=[x_name], out=words[0], out_len=L,
           more=[(w, L) for w in words[1:]])
    for i, w in enumerate(words):
        b.emit("range_word", dict(L=L), layer=il, label=f"{label}.range{i}",
               inputs=[w], out=f"{label}.z{i}", out_len=L)


def _topk_experts(b, x, s_sel, s, bias, il, S, d, d_ff, E, k, prefix, *,
                  width, select_word_bits, score_bits, C, bracket_word_bits,
                  bridge=False):
    """The claims topk_routing.topk_moe_ffn records, one for one: selection
    (TopkRoutingClaim + the range on v), the slot masks and slot scores, the
    gate bracket (+ ranges on rem, Z - 1 - rem and w), the stacked slot masks
    and inputs (two concats), routed gate/up at T' = kT with their rescales,
    SiLU, the rescaled Hadamard, the routed down projection, the split into
    the k slot outputs, the gate-weighted combine and its one rescale.
    Selection reads `s_sel` with the bias; the slot scores and gate weights
    read `s` (the same variable when one score scale serves both, as on the
    toy tape). Returns the FFN output variable (S x d)."""
    TE, Tk, Tp = S * E, S * k, k * S
    b.emit("topk_routing", dict(T=S, E=E, k=k), layer=il, label=f"{prefix}.topk",
           inputs=[s_sel, bias], out=f"{prefix}.mask", out_len=TE,
           more=[(f"{prefix}.topk.{n}", TE) for n in ("qt", "d", "md", "v")]
           + [(f"{prefix}.topk.tau", S)])
    _ranged(b, f"{prefix}.topk.v", TE, threshold_words(width, select_word_bits),
            il, f"{prefix}.topk.v")
    slots = [f"{prefix}.M{i}" for i in range(k)]
    b.emit("topk_slots", dict(T=S, E=E, k=k), layer=il, label=f"{prefix}.slots",
           inputs=[f"{prefix}.mask", s], out=f"{prefix}.ss", out_len=Tk,
           more=[(m, TE) for m in slots] + [(f"{prefix}.MS{i}", TE) for i in range(k)])
    b.emit("gate_bracket", dict(T=S, k=k), layer=il, label=f"{prefix}.bracket",
           inputs=[f"{prefix}.ss"], out=f"{prefix}.gw", out_len=Tk,
           more=[(f"{prefix}.bracket.{n}", Tk) for n in ("Zb", "wZ", "rem", "gr")]
           + [(f"{prefix}.bracket.Z", S)])
    n_rem, n_w = bracket_words(score_bits, k, C, bracket_word_bits)
    _ranged(b, f"{prefix}.bracket.rem", Tk, n_rem, il, f"{prefix}.bracket.rem")
    _ranged(b, f"{prefix}.bracket.gr", Tk, n_rem, il, f"{prefix}.bracket.gr")
    _ranged(b, f"{prefix}.gw", Tk, n_w, il, f"{prefix}.bracket.w")
    b.emit("concat", dict(L=Tp * E), layer=il, label=f"{prefix}.M_k",
           inputs=slots, out=f"{prefix}.M_k", out_len=Tp * E)
    b.emit("concat", dict(L=Tp * d), layer=il, label=f"{prefix}.x_k",
           inputs=[x], out=f"{prefix}.x_k", out_len=Tp * d)

    def routed(kind, x_in, K, J, rescaled=True):
        ws = [b.input_var(f"{prefix}.e{e}.W_{kind}", K * J, persistent=True,
                          external=bridge) for e in range(E)]
        raw = b.emit("routed_projected", dict(T=Tp, K=K, J=J, E=E), layer=il,
                     label=f"{prefix}.{kind}_rp", inputs=[x_in, f"{prefix}.M_k"] + ws,
                     out=f"{prefix}.{kind}_raw", out_len=Tp * J)
        if not rescaled:
            return raw
        return b.emit("rescale_claim", dict(length=Tp * J, rescale_bits=12,
                                            output_width=26),
                      layer=il, label=f"{prefix}.{kind}_rs", inputs=[raw],
                      out=f"{prefix}.{kind}", out_len=Tp * J)

    g = routed("gate", f"{prefix}.x_k", d, d_ff)
    u = routed("up", f"{prefix}.x_k", d, d_ff)
    sg = b.emit("silu", dict(L=Tp * d_ff), layer=il, label=f"{prefix}.silu",
                inputs=[g], out=f"{prefix}.sg", out_len=Tp * d_ff)
    h = b.emit("hadamard", dict(L=Tp * d_ff), layer=il, label=f"{prefix}.hidden",
               inputs=[sg, u], out=f"{prefix}.h", out_len=Tp * d_ff)
    D = routed("down", h, d_ff, d, rescaled=False)
    parts = [f"{prefix}.D{i}" for i in range(k)]
    b.emit("split", dict(L=Tp * d), layer=il, label=f"{prefix}.split", inputs=[D],
           out=parts[0], out_len=S * d, more=[(n, S * d) for n in parts[1:]])
    y_raw = b.emit("freivalds_combine", dict(T=S, E=k, F=d), layer=il,
                   label=f"{prefix}.combine", inputs=[f"{prefix}.gw"] + parts,
                   out=f"{prefix}.y_raw", out_len=S * d)
    return b.emit("rescale_claim", dict(length=S * d, rescale_bits=24,
                                        output_width=26),
                  layer=il, label=f"{prefix}.y_rs", inputs=[y_raw],
                  out=f"{prefix}.ffn", out_len=S * d)


def _mla_attention(b, x, il, S, d, H, q_rank, kv_rank, d_nope, d_rope, d_v,
                   prefix):
    """DeepSeek-V3 multi-head latent attention, composed from existing claims
    WITHOUT weight absorption (the reference modeling code's form): the query
    through its low-rank pair with a norm between; the compressed KV and the
    shared rope key from one down projection (two matmuls here, the weight's
    column split); the per-head key and value from the compressed KV;
    YaRN RoPE on the rope parts (a RoPE claim with other tables); the rope
    key broadcast to the heads; query and key assembled per head (a pin of
    the concat's cost); scores, softmax and AV over all H heads; the output
    projection. ASSUMPTION until the K2 demo fixes the composition (design
    item 5): the pins and the split matmuls are one plausible layout. None
    of them is quadratic in S, so the S^2 term — 21 slots per score cell
    over H heads, as for every attention here — does not depend on it.
    Returns (post-attention residual, normed and gained FFN input)."""
    Ld, dqk = S * d, d_nope + d_rope

    def norm(x_in, width, tag):
        g = b.input_var(f"{prefix}.{tag}.gain", width)
        n = b.emit("rmsnorm", dict(B=S, d=width), label=f"{prefix}.{tag}",
                   layer=il, inputs=[x_in], out=f"{prefix}.{tag}.n",
                   out_len=S * width)
        b.emit("embed_lookup", dict(L=S * width), label=f"{prefix}.{tag}.bcast",
               layer=il, inputs=[g], out=f"{prefix}.{tag}.gb", out_len=S * width)
        return b.emit("hadamard", dict(L=S * width), label=f"{prefix}.{tag}.gain",
                      layer=il, inputs=[n, f"{prefix}.{tag}.gb"],
                      out=f"{prefix}.{tag}.ng", out_len=S * width)

    def mm(x_in, k, n, tag, heads=1):
        w = b.input_var(f"{prefix}.W_{tag}", k * n, persistent=True)
        return b.emit("matmul", dict(m=S, k=k, n=n), label=f"{prefix}.{tag}",
                      layer=il, inputs=[x_in, w], out=f"{prefix}.{tag}",
                      out_len=S * n)

    xn = norm(x, d, "norm1")
    qa = norm(mm(xn, d, q_rank, "q_a"), q_rank, "q_a_norm")
    q_nope = mm(qa, q_rank, H * d_nope, "q_nope")
    q_pe = mm(qa, q_rank, H * d_rope, "q_pe")
    ckv = norm(mm(xn, d, kv_rank, "kv_a"), kv_rank, "kv_a_norm")
    k_pe = mm(xn, d, d_rope, "k_pe")
    k_nope = mm(ckv, kv_rank, H * d_nope, "k_nope")
    v = mm(ckv, kv_rank, H * d_v, "v")
    q_pe = b.emit("rope", dict(L=S * H * d_rope), label=f"{prefix}.rope_q",
                  layer=il, inputs=[q_pe], out=f"{prefix}.q_pe_r",
                  out_len=S * H * d_rope)
    k_pe = b.emit("rope", dict(L=S * d_rope), label=f"{prefix}.rope_k",
                  layer=il, inputs=[k_pe], out=f"{prefix}.k_pe_r",
                  out_len=S * d_rope)
    k_pe = b.emit("embed_lookup", dict(L=S * H * d_rope), label=f"{prefix}.k_pe.bcast",
                  layer=il, inputs=[k_pe], out=f"{prefix}.k_pe_h",
                  out_len=S * H * d_rope)
    q = b.emit("concat", dict(L=S * H * dqk), label=f"{prefix}.q", layer=il,
               inputs=[q_nope, q_pe], out=f"{prefix}.q", out_len=S * H * dqk)
    k = b.emit("concat", dict(L=S * H * dqk), label=f"{prefix}.k", layer=il,
               inputs=[k_nope, k_pe], out=f"{prefix}.k", out_len=S * H * dqk)
    scores = b.emit("matmul", dict(m=S, k=H * dqk, n=S, heads=H), layer=il,
                    label=f"{prefix}.scores", inputs=[q, k],
                    out=f"{prefix}.scores", out_len=H * S * S)
    sm = b.emit("softmax", dict(B=H * S, M=S, causal=True), layer=il,
                label=f"{prefix}.softmax", inputs=[scores],
                out=f"{prefix}.sm", out_len=H * S * S)
    av = b.emit("matmul", dict(m=S, k=H * S, n=d_v, heads=H), layer=il,
                label=f"{prefix}.attnV", inputs=[sm, v],
                out=f"{prefix}.av", out_len=S * H * d_v)
    proj = mm(av, H * d_v, d, "o")
    r1 = b.emit("add", dict(L=Ld), label=f"{prefix}.resid1", layer=il,
                inputs=[x, proj], out=f"{prefix}.r1", out_len=Ld)
    return r1, norm(r1, d, "norm2")


def _moe_ffn_topk(b, r1, n2g, il, S, d, d_ff, E, k, prefix, *, bridge=False,
                  **topk):
    """A DeepSeek-V3 MoE FFN with output-side gates: the router and its
    sigmoid lookups over all T x E logits (two: the selection scale and the
    gate scale — ASSUMPTION, design 3.1 leaves the finer selection scale to a
    second lookup or a shift), the committed per-layer selection bias, the
    top-k chain of topk_moe_ffn, the shared expert, two residual adds."""
    Ld, TE = S * d, S * E
    wr = b.input_var(f"{prefix}.W_router", d * E, persistent=True)
    bias = b.input_var(f"{prefix}.bias", E, persistent=True)
    router = b.emit("matmul", dict(m=S, k=d, n=E), label=f"{prefix}.router",
                    layer=il, inputs=[n2g, wr], out=f"{prefix}.router",
                    out_len=TE)
    s_sel = b.emit("ptlookup", dict(L=TE), label=f"{prefix}.sigma_sel", layer=il,
                   inputs=[router], out=f"{prefix}.s_sel", out_len=TE)
    s_gate = b.emit("ptlookup", dict(L=TE), label=f"{prefix}.sigma", layer=il,
                    inputs=[router], out=f"{prefix}.s", out_len=TE)
    ffn = _topk_experts(b, n2g, s_sel, s_gate, bias, il, S, d, d_ff, E, k,
                        prefix, bridge=bridge, **topk)
    sh = _dense_ffn_out(b, n2g, il, S, d, d_ff, f"{prefix}.sh")
    a1 = b.emit("add", dict(L=Ld), label=f"{prefix}.resid2a", layer=il,
                inputs=[r1, ffn], out=f"{prefix}.r2a", out_len=Ld)
    return b.emit("add", dict(L=Ld), label=f"{prefix}.resid2b", layer=il,
                  inputs=[a1, sh], out=f"{prefix}.r2", out_len=Ld)


def _dense_ffn_out(b, n2g, il, S, d, d_ff, prefix):
    """gate/up matmuls, SiLU, Hadamard, down: the FFN without its residual."""
    wg = b.input_var(f"{prefix}.W_gate", d * d_ff, persistent=True)
    wu = b.input_var(f"{prefix}.W_up", d * d_ff, persistent=True)
    wd = b.input_var(f"{prefix}.W_down", d_ff * d, persistent=True)
    g = b.emit("matmul", dict(m=S, k=d, n=d_ff), label=f"{prefix}.gate",
               layer=il, inputs=[n2g, wg], out=f"{prefix}.g", out_len=S * d_ff)
    u = b.emit("matmul", dict(m=S, k=d, n=d_ff), label=f"{prefix}.up",
               layer=il, inputs=[n2g, wu], out=f"{prefix}.u", out_len=S * d_ff)
    sg = b.emit("silu", dict(L=S * d_ff), label=f"{prefix}.silu", layer=il,
                inputs=[g], out=f"{prefix}.sg", out_len=S * d_ff)
    h = b.emit("hadamard", dict(L=S * d_ff), label=f"{prefix}.hidden",
               layer=il, inputs=[sg, u], out=f"{prefix}.h", out_len=S * d_ff)
    return b.emit("matmul", dict(m=S, k=d_ff, n=d), label=f"{prefix}.down",
                  layer=il, inputs=[h, wd], out=f"{prefix}.d", out_len=S * d)


# Kimi K2's top-k parameters as the K2 demo is expected to set them
# (analysis/topk-routing-design.md 3.1-3.2): selection differences under
# 2^26 in two 13-bit words, scores at the 2^12 scale (13 bits), C =
# round(2.827 * 2^12), the bracket's 8-bit words. The word counts are fixed
# once the router-logit and bias ranges are measured (design item 5).
K2_TOPK = dict(width=26, select_word_bits=13, score_bits=13,
               C=round(2.827 * 4096), bracket_word_bits=8)

# Kimi K2 (moonshotai/Kimi-K2-Instruct config.json at Hugging Face revision
# fd1984e2b7a3350dbf7305fe73a4ede25c14de50): DeepseekV3ForCausalLM.
K2 = dict(d=7168, layers=61, first_k_dense=1, d_ff_dense=18432, d_ff_expert=2048,
          experts=384, top_k=8, shared_experts=1, heads=64, q_lora_rank=1536,
          kv_lora_rank=512, qk_nope_head_dim=128, qk_rope_head_dim=64,
          v_head_dim=128, vocab=163840)


def kimi_k2(seq: int, t_queries: int = 54, *, bridge: bool = False) -> Manifest:
    """Kimi K2 under the routed-projected protocol with top-k routing: the
    token-select embedding, 61 layers of multi-head latent attention, the
    first with a dense FFN and the other 60 with 384 routed experts (8
    active) and one shared expert, then the final norm, gain and LM head.

    Modeling assumptions, each replaced by the K2 demo (design item 5): the
    attention composition (_mla_attention), the two sigmoid lookups
    (_moe_ffn_topk), and the word counts of K2_TOPK. `bridge` marks the
    routed experts' weights bridge-held (external), as the session-6 proof
    held Maverick's; their own cost is then unmodeled, as for Maverick.
    t_queries defaults to Maverick's demonstrated 54."""
    c = K2
    d, V = c["d"], c["vocab"]
    b = _Builder()
    emb = b.input_var("embed.W", V * d, persistent=True)
    x = b.emit("matmul", dict(m=seq, k=V, n=d, rescale=False),
               label="embed.select", inputs=[emb], out="x0", out_len=seq * d)
    for il in range(c["layers"]):
        prefix = f"L{il}"
        r1, n2g = _mla_attention(b, x, il, seq, d, c["heads"], c["q_lora_rank"],
                                 c["kv_lora_rank"], c["qk_nope_head_dim"],
                                 c["qk_rope_head_dim"], c["v_head_dim"], prefix)
        if il < c["first_k_dense"]:
            f = _dense_ffn_out(b, n2g, il, seq, d, c["d_ff_dense"], prefix)
            x = b.emit("add", dict(L=seq * d), label=f"{prefix}.resid2",
                       layer=il, inputs=[r1, f], out=f"{prefix}.r2",
                       out_len=seq * d)
        else:
            x = _moe_ffn_topk(b, r1, n2g, il, seq, d, c["d_ff_expert"],
                              c["experts"], c["top_k"], prefix, bridge=bridge,
                              **K2_TOPK)
    gf = b.input_var("final.gain", d)
    fn = b.emit("rmsnorm", dict(B=seq, d=d), label="final.norm", inputs=[x],
                out="final.n", out_len=seq * d)
    b.emit("embed_lookup", dict(L=seq * d), label="final.gain.bcast",
           inputs=[gf], out="final.gb", out_len=seq * d)
    fng = b.emit("hadamard", dict(L=seq * d), label="final.gain",
                 inputs=[fn, "final.gb"], out="final.ng", out_len=seq * d)
    head = b.input_var("head.W", d * V, persistent=True)
    b.emit("matmul", dict(m=seq, k=d, n=V), label="lm_head",
           inputs=[fng, head], out="logits", out_len=seq * V)
    return Manifest(
        source=dict(kind="synth", generator="synth.kimi_k2",
                    config="moonshotai/Kimi-K2-Instruct@fd1984e2"),
        model=dict(name="kimi-k2", **c),
        run=dict(seq=seq, ligero=dict(T_QUERIES=t_queries, **LIGERO)),
        claims=b.claims, variables=list(b.vars.values()))


def topk_toy(seq: int, t_queries: int = 4, *, d: int = 16, d_ff: int = 16,
             E: int = 8, k: int = 3, **topk) -> Manifest:
    """One top-k expert chain on committed inputs — exactly what
    topk_routing.topk_moe_ffn records — for crosscheck.py's topk-toy entry,
    which builds the real toy tape at the same shape and diffs the two."""
    params = dict(K2_TOPK, **topk)
    b = _Builder()
    x = b.input_var("x", seq * d)
    s = b.input_var("s", seq * E)
    bias = b.input_var("b", E)
    _topk_experts(b, x, s, s, bias, 0, seq, d, d_ff, E, k, "L0", **params)
    return Manifest(
        source=dict(kind="synth", generator="synth.topk_toy"),
        model=dict(name="topk-toy", d=d, d_ff_expert=d_ff, experts=E, top_k=k),
        run=dict(seq=seq, ligero=dict(T_QUERIES=t_queries, **LIGERO)),
        claims=b.claims, variables=list(b.vars.values()))


BUILDERS = {"maverick": maverick, "llama7b": llama7b,
            "maverick-projected": maverick_projected, "kimi-k2": kimi_k2}
