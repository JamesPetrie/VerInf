"""HeadInterleaveClaim: one (w1 + w2)-wide vector per head from two parts
(analysis/mla-attention-design.md §4.2–4.3).

Multi-head latent attention scores each head over a query and a key whose
first w1 = 128 dimensions come from one projection and whose last w2 = 64
are rotary; the key's rotary part is ONE vector shared by all heads. The
scores matmul reads per-head blocks (heads=H, head_dim=w1+w2), so the claim
assembles them:

  dst[t, h, j]      = a[t, h, j]                   j < w1
  dst[t, h, w1 + j] = b[t, h, j]   (per head)      j < w2
                    = b[t, j]      (shared=1)

One linear constraint per dst slot, cid base + (t·H + h)·W + c with
W = w1 + w2; dst enters with coef −1, each source slot with +1 at its
position (a shared b slot at H positions, one per head). No quads. Every dst
slot equals exactly one source slot, so dst is unique given a and b.

Packets: dst as L2_IdentityScalar; a and b as L2_BlockStrideScalar (the
fold's regular band descriptor, core._lower_geometry). Rust twin:
handlers.rs compile_head_interleave and Expander::BlockStride. Cost row:
profiler/claimcosts.py head_interleave, (L, L, 0) with L = T·H·W."""
from dataclasses import dataclass

import torch

import compute_fns as _cf
from claims import COMPILE_FNS, _emit_pkt_per_row
from core import AUX_FNS, P, SAMPLE_FNS, Variable
from packets import L2_BlockStrideScalar, L2_IdentityScalar


@dataclass
class HeadInterleaveClaim:
    a: Variable            # (T, H, w1)
    b: Variable            # (T, H, w2), or (T, w2) when shared
    dst: Variable          # (T, H, w1 + w2)
    T: int
    H: int
    w1: int
    w2: int
    shared: int            # 1: b is one vector per token, fanned out to the heads

    @property
    def length(self):
        return self.dst.length


def head_interleave_compute(c: HeadInterleaveClaim, live):
    T, H, w1, w2 = c.T, c.H, c.w1, c.w2
    a = live[c.a].contiguous().view(torch.int64).view(T, H, w1)
    if c.shared:
        b = live[c.b].contiguous().view(torch.int64).view(T, 1, w2).expand(T, H, w2)
    else:
        b = live[c.b].contiguous().view(torch.int64).view(T, H, w2)
    dst = torch.cat([a, b], dim=2).contiguous().view(-1).view(torch.uint64)
    return {c.dst: dst}


def head_interleave_compile(c: HeadInterleaveClaim, _ch, cfg, base: int):
    ell = cfg.ELL
    T, H, w1, w2 = c.T, c.H, c.w1, c.w2
    W = w1 + w2
    assert c.a.length == T * H * w1 and c.dst.length == T * H * W
    assert c.b.length == (T * w2 if c.shared else T * H * w2)
    pk = []
    _emit_pkt_per_row(c.dst, ell, lambda: L2_IdentityScalar(
        base=base, var_row_start=c.dst.row_start, L=c.dst.length, coef=(P - 1) % P), pk)
    _emit_pkt_per_row(c.a, ell, lambda: L2_BlockStrideScalar(
        base=base, var_row_start=c.a.row_start, L=c.a.length, inner=w1, outer=W,
        fan=1, fan_stride=0, coef=1), pk)
    if c.shared:
        mk = lambda: L2_BlockStrideScalar(
            base=base + w1, var_row_start=c.b.row_start, L=c.b.length, inner=w2,
            outer=H * W, fan=H, fan_stride=W, coef=1)
    else:
        mk = lambda: L2_BlockStrideScalar(
            base=base + w1, var_row_start=c.b.row_start, L=c.b.length, inner=w2,
            outer=W, fan=1, fan_stride=0, coef=1)
    _emit_pkt_per_row(c.b, ell, mk, pk)
    return pk, [], c.dst.length, None


COMPILE_FNS[HeadInterleaveClaim] = head_interleave_compile
SAMPLE_FNS[HeadInterleaveClaim] = lambda c, ci, s_op: None
AUX_FNS[HeadInterleaveClaim] = lambda c, witness, ch: {}
_cf.COMPUTE_FNS[HeadInterleaveClaim] = head_interleave_compute

_BUILD = [0]


def head_interleave(tape, a, b, *, T, H, w1, w2, shared=False):
    """dst (T, H·(w1 + w2)) with each head's w1 dimensions from a (T, H·w1)
    and its w2 from b: b (T, H·w2) per head, or b (T, w2) shared by all
    heads. The result is laid out for tape.matmul(..., heads=H,
    head_dim=w1 + w2)."""
    from tape import WitnessTensor
    assert a.var.length == T * H * w1, "a must be (T, H·w1)"
    assert b.var.length == (T * w2 if shared else T * H * w2), (
        "b must be (T, w2) when shared, else (T, H·w2)")
    _BUILD[0] += 1
    dst = tape._alloc(f"hi{_BUILD[0]}_{a.var.name[:12]}", T * H * (w1 + w2))
    claim = HeadInterleaveClaim(a=a.var, b=b.var, dst=dst, T=T, H=H, w1=w1, w2=w2,
                                shared=int(bool(shared)))
    outs = tape._process_claim(claim, [a.var, b.var])
    tape.claims.append(claim)
    return WitnessTensor(outs[dst] if outs else None, dst, (T, H * (w1 + w2)), tape)
