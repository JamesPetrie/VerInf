"""Synthetic manifest builders: Maverick and Llama-2-7B, closed form.

Mirrors the claim structure of demo_maverick_full.py / demo_llama7b.py the way
analysis/maverick_cost_model.py does, but emits one ClaimRecord per tape op
(per-expert matmuls individually) with producer/consumer variable edges, so
the DAG and live-set analyses see the real fan-out — notably the E-way expert
parallelism per MoE layer.

This is the no-torch path: it exists so prediction can run anywhere and as a
cross-check of the tape extractor. The extractor (extract.py, run where the
prover runs) is the ground truth; when the two disagree, trust the tape.
Model shapes here are the two team-standard workloads; any other model should
come in through the extractor, not by adding builders here.
"""
from __future__ import annotations

from manifest import Manifest, ClaimRecord, VariableRecord

LIGERO = dict(ELL=8192, K_DEG=16384, N_LIG=65536)


class _Builder:
    """Mini-tape: emit(claim) allocates output vars and records edges."""

    def __init__(self):
        self.claims: list[ClaimRecord] = []
        self.vars: dict[str, VariableRecord] = {}

    def input_var(self, name, length, *, persistent=False):
        self.vars[name] = VariableRecord(name=name, length=length,
                                         persistent=persistent, producer=None)
        return name

    def emit(self, type_, params, *, label="", layer=None, inputs=(),
             out=None, out_len=0, phase2=()):
        idx = len(self.claims)
        outputs = []
        if out is not None:
            self.vars[out] = VariableRecord(name=out, length=int(out_len),
                                            producer=idx)
            outputs.append(out)
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


BUILDERS = {"maverick": maverick, "llama7b": llama7b}
