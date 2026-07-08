"""ShadowTape — subclass of Tape that records per-op LOCAL quantization
error vs an FP shadow computed from the same quantized inputs.

Lives in analysis/ so production code (pipeline/*.py) never
imports it. Inherits from Tape so any Tape user can switch to ShadowTape
with one constructor swap and get per-op error attribution for free.

The local-error model:

  Each op call (matmul, rmsnorm, …) receives quantized inputs (uint64
  Goldilocks tensors). ShadowTape converts those SAME inputs to float64
  and computes a pure-FP version of the op. The local error is

        local_err = | quantized_output_as_float − fp_output |

  isolated from any upstream noise that the inputs may already carry.
  This answers "which op kind introduces the most local rounding /
  table-lookup error" — exact ops like Goldilocks add should show
  effectively zero local error; lookup-table ops (silu, exp inside
  softmax, 1/√x inside rmsnorm) should dominate.

Scales (Q-format) are tracked per WitnessTensor: each op records the
scale its output sits at so downstream conversions are correct.

Currently shadowed: matmul, hadamard, hadamard_broadcast, rmsnorm,
softmax, silu, rope, add. Other ops (embed, fingerprint, range_word,
word_extract, paired_tlookup) propagate through unchanged.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "prover"))

from cuda_primitives import P
from tape import Tape, WitnessTensor


# ---------------------------------------------------------------------------
# Goldilocks → signed float conversion.
# ---------------------------------------------------------------------------

_P_U = np.uint64(P)
_HALF_U = np.uint64(P // 2)


def gl_to_float(t_u64: torch.Tensor, scale: int = 1) -> torch.Tensor:
    """Goldilocks uint64 → signed float64 cuda tensor, divided by `scale`.

    The sign flip is done in integer arithmetic (uint64 wraparound +
    int64 view) before converting to float, to avoid the ~2^11 of
    precision lost by `float(P) − v` when v is near P. Same trick as
    pipeline._signed_floor_decomp."""
    arr = t_u64.detach().cpu().numpy().astype(np.uint64)
    pos = arr.view(np.int64)                          # correct when arr < P/2 < 2^63
    neg = (arr - _P_U).view(np.int64)                 # correct when arr >= P/2 (gives v - P)
    signed = np.where(arr < _HALF_U, pos, neg)
    return torch.from_numpy(signed.astype(np.float64) / float(scale)).cuda()


# ---------------------------------------------------------------------------
# Per-op delta record.
# ---------------------------------------------------------------------------

@dataclass
class OpDelta:
    op:         str
    max_abs:    float
    mean_abs:   float
    output_max: float
    n_cells:    int

    @property
    def rel_max(self) -> float:
        return self.max_abs / max(self.output_max, 1e-12)

    @property
    def rel_mean(self) -> float:
        return self.mean_abs / max(self.output_max, 1e-12)


# ---------------------------------------------------------------------------
# ShadowTape.
# ---------------------------------------------------------------------------

class ShadowTape(Tape):
    """Tape that records each op's local quantization error vs an FP shadow."""

    def __init__(self, *args, default_scale: int = 1 << 12, **kw):
        super().__init__(*args, **kw)
        # id(WitnessTensor) → Q-format scale of its .data values.
        self._scale: Dict[int, int] = {}
        self.deltas: List[OpDelta] = []
        self.default_scale = default_scale

    # -----------------------------------------------------------------------
    # Scale tracking.
    # -----------------------------------------------------------------------

    def _scale_of(self, wt: WitnessTensor) -> int:
        return self._scale.get(id(wt), self.default_scale)

    def _set_scale(self, wt: WitnessTensor, s: int):
        self._scale[id(wt)] = s

    def _record(self, op: str, out_q: torch.Tensor, out_fp: torch.Tensor,
                 out_scale: int):
        out_q_fp = gl_to_float(out_q, out_scale)
        diff = (out_q_fp - out_fp).abs()
        self.deltas.append(OpDelta(
            op=op,
            max_abs=float(diff.max().item()),
            mean_abs=float(diff.mean().item()),
            output_max=float(out_fp.abs().max().item()),
            n_cells=int(diff.numel()),
        ))

    # -----------------------------------------------------------------------
    # Commit: tracks scale for the committed data.
    # -----------------------------------------------------------------------

    def commit(self, name, data, shape, *, scale: Optional[int] = None):
        # Production Tape.commit signature doesn't take scale; we add it
        # as an optional ShadowTape-only kwarg for explicit scale-setting.
        wt = super().commit(name, data, shape)
        self._set_scale(wt, scale if scale is not None else self.default_scale)
        return wt

    # -----------------------------------------------------------------------
    # Per-op overrides.
    # Each: call super() for the quantized result, convert THIS op's
    # quantized inputs to FP, compute the FP version, record the delta,
    # set the output's scale.
    # -----------------------------------------------------------------------

    # ---- matmul ----
    def matmul(self, a: WitnessTensor, b: WitnessTensor, *,
                transpose_b: bool = False,
                heads: int = 1, head_dim: int = 0,
                s_a: Optional[int] = None, s_b: Optional[int] = None,
                s_out: Optional[int] = None, output_width: int = 24
                ) -> WitnessTensor:
        out = super().matmul(a, b, transpose_b=transpose_b,
                              heads=heads, head_dim=head_dim,
                              s_a=s_a, s_b=s_b, s_out=s_out,
                              output_width=output_width)
        # FP inputs reconstructed from THE QUANTIZED .data the op received.
        sa = s_a if s_a is not None else self._scale_of(a)
        sb = s_b if s_b is not None else self._scale_of(b)
        a_fp = gl_to_float(a.data, sa)
        b_fp = gl_to_float(b.data, sb)
        m, k = a.shape
        if head_dim == 0:
            head_dim = k // heads
        H, K = heads, head_dim
        if transpose_b:
            n, _ = b.shape
        else:
            _, n_times_h = b.shape
            n = n_times_h // H
        if heads == 1:
            if transpose_b:
                out_fp = (a_fp.view(m, k) @ b_fp.view(n, k).t()).contiguous().view(-1)
            else:
                out_fp = (a_fp.view(m, k) @ b_fp.view(k, n)).contiguous().view(-1)
        else:
            a3 = a_fp.view(m, H, K)
            if transpose_b:
                b3 = b_fp.view(n, H, K)
                c3 = torch.einsum('mhk,nhk->mhn', a3, b3)
            else:
                b3 = b_fp.view(K, H, n)
                c3 = torch.einsum('mhk,khn->mhn', a3, b3)
            out_fp = c3.contiguous().view(-1)
        out_scale = s_out if s_out is not None else sa * sb
        self._record("matmul", out.data, out_fp, out_scale)
        self._set_scale(out, out_scale)
        return out

    # ---- hadamard ----
    def hadamard(self, a: WitnessTensor, b: WitnessTensor, *,
                  s_a: Optional[int] = None, s_b: Optional[int] = None,
                  s_out: Optional[int] = None, output_width: int = 24
                  ) -> WitnessTensor:
        out = super().hadamard(a, b, s_a=s_a, s_b=s_b, s_out=s_out,
                                output_width=output_width)
        sa = s_a if s_a is not None else self._scale_of(a)
        sb = s_b if s_b is not None else self._scale_of(b)
        a_fp = gl_to_float(a.data, sa); b_fp = gl_to_float(b.data, sb)
        out_fp = (a_fp * b_fp).contiguous()
        out_scale = s_out if s_out is not None else sa * sb
        self._record("hadamard", out.data, out_fp, out_scale)
        self._set_scale(out, out_scale)
        return out

    # ---- hadamard_broadcast ----
    def hadamard_broadcast(self, x: WitnessTensor, gain: WitnessTensor, *,
                            SEQ: int, d: int,
                            s_a: Optional[int] = None, s_b: Optional[int] = None,
                            s_out: Optional[int] = None, output_width: int = 24
                            ) -> WitnessTensor:
        out = super().hadamard_broadcast(x, gain, SEQ=SEQ, d=d,
                                          s_a=s_a, s_b=s_b, s_out=s_out,
                                          output_width=output_width)
        sa = s_a if s_a is not None else self._scale_of(x)
        sb = s_b if s_b is not None else self._scale_of(gain)
        x_fp = gl_to_float(x.data, sa); g_fp = gl_to_float(gain.data, sb)
        out_fp = (x_fp.view(SEQ, d) * g_fp.view(1, d)).contiguous().view(-1)
        out_scale = s_out if s_out is not None else sa * sb
        self._record("hadamard_broadcast", out.data, out_fp, out_scale)
        self._set_scale(out, out_scale)
        return out

    # ---- rmsnorm ----
    def rmsnorm(self, x: WitnessTensor, *, d: int, s: int = 4, eps_int: int = 1,
                 slack_chunk_width: int = 16, slack_n_chunks: int = 1,
                 s_in: Optional[int] = None,
                 s_out: Optional[int] = None, output_width: int = 16):
        out = super().rmsnorm(x, d=d, s=s, eps_int=eps_int,
                               slack_chunk_width=slack_chunk_width,
                               slack_n_chunks=slack_n_chunks,
                               s_in=s_in, s_out=s_out, output_width=output_width)
        sx = s_in if s_in is not None else self._scale_of(x)
        x_fp = gl_to_float(x.data, sx).view(-1, d)
        # eps stored in pipeline as eps_int with respect to s² scale. In
        # float units this is eps_int / s² (matches the pipeline's
        # internal computation). The output (1/√(mean(x²)+ε)) · x then
        # gets rescaled to s_out by the pipeline's signed-floor decomp.
        eps_real = float(eps_int) / float(s * s)
        mean_sq = (x_fp * x_fp).mean(dim=-1, keepdim=True)
        inv_rms = (mean_sq + eps_real).rsqrt()
        out_fp = (x_fp * inv_rms).contiguous().view(-1)
        out_scale = s_out if s_out is not None else 1
        self._record("rmsnorm", out.data, out_fp, out_scale)
        self._set_scale(out, out_scale)
        return out

    # ---- silu ----
    def silu(self, x: WitnessTensor, *, s_in: Optional[int] = None):
        out = super().silu(x, s_in=s_in)
        sx = s_in if s_in is not None else self._scale_of(x)
        x_fp = gl_to_float(x.data, sx)
        out_fp = (x_fp * torch.sigmoid(x_fp)).contiguous()
        # SiLU output sits at same scale as input in our pipeline.
        self._record("silu", out.data, out_fp, sx)
        self._set_scale(out, sx)
        return out

    # ---- softmax ----
    def softmax(self, x: WitnessTensor, *, M: int, s_x: int = 4,
                 s_c: Optional[int] = None, s_y: Optional[int] = None,
                 Z_max: int = 0, saturate: bool = False,
                 Z_high_width: int = 16, aux_chunk_width: int = 20,
                 causal: bool = False, heads: int = 1):
        out = super().softmax(x, M=M, s_x=s_x, s_c=s_c, s_y=s_y,
                               Z_max=Z_max, saturate=saturate,
                               Z_high_width=Z_high_width,
                               aux_chunk_width=aux_chunk_width,
                               causal=causal, heads=heads)
        sx = self._scale_of(x) if s_x is None else s_x
        x_fp = gl_to_float(x.data, sx)
        B = x_fp.numel() // M
        x_2d = x_fp.view(B, M)
        if causal:
            # Row layout: b = m*H + h where m is the query position, h is the
            # head index. Same convention as tape.softmax's causal-mask claim
            # (`i_qry_flat = b_flat // heads`). Using `b % SEQ` is wrong
            # whenever heads != SEQ.
            iq = torch.arange(B, device="cuda") // heads
            j  = torch.arange(M, device="cuda").unsqueeze(0)
            mask = j > iq.unsqueeze(1)
            x_2d = x_2d.masked_fill(mask, float("-inf"))
        e = torch.softmax(x_2d, dim=-1)
        out_fp = e.contiguous().view(-1)
        out_scale = s_y if s_y is not None else 1
        self._record("softmax", out.data, out_fp, out_scale)
        self._set_scale(out, out_scale)
        return out

    # ---- add (Goldilocks: exact) ----
    def add(self, a: WitnessTensor, b: WitnessTensor) -> WitnessTensor:
        out = super().add(a, b)
        sa = self._scale_of(a)
        a_fp = gl_to_float(a.data, sa); b_fp = gl_to_float(b.data, sa)
        out_fp = (a_fp + b_fp).contiguous()
        self._record("add", out.data, out_fp, sa)
        self._set_scale(out, sa)
        return out

    # ---- rope (cos/sin from precomputed Q-quantized tables) ----
    def rope(self, x: WitnessTensor, *, SEQ: int, d_h: int,
              s_x: int, heads: int = 1, s_out: Optional[int] = None,
              output_width: int = 24):
        out = super().rope(x, SEQ=SEQ, d_h=d_h, s_x=s_x, heads=heads,
                            s_out=s_out, output_width=output_width)
        x_fp = gl_to_float(x.data, s_x)
        H = heads; half = d_h // 2
        x3 = x_fp.view(SEQ, H, d_h)
        x_a = x3[..., :half]; x_b = x3[..., half:]
        pos = torch.arange(SEQ, device="cuda", dtype=torch.float64).view(SEQ, 1, 1)
        i = torch.arange(half, device="cuda", dtype=torch.float64).view(1, 1, half)
        theta = pos * 10000.0 ** (-2.0 * i / float(d_h))
        c_t = torch.cos(theta); s_t = torch.sin(theta)
        out_fp = torch.cat([x_a * c_t - x_b * s_t,
                             x_a * s_t + x_b * c_t], dim=-1).contiguous().view(-1)
        out_scale = s_out if s_out is not None else s_x
        self._record("rope", out.data, out_fp, out_scale)
        self._set_scale(out, out_scale)
        return out

    # -----------------------------------------------------------------------
    # Reporting.
    # -----------------------------------------------------------------------

    def summary(self) -> Dict[str, Dict[str, float]]:
        bucket: Dict[str, List[OpDelta]] = defaultdict(list)
        for d in self.deltas:
            bucket[d.op].append(d)
        report: Dict[str, Dict[str, float]] = {}
        for op, ds in bucket.items():
            tot = sum(d.n_cells for d in ds)
            report[op] = dict(
                calls=len(ds), cells=tot,
                max_abs=max(d.max_abs for d in ds),
                mean_abs=sum(d.mean_abs * d.n_cells for d in ds) / tot,
                max_rel=max(d.rel_max for d in ds),
                mean_rel=sum(d.rel_mean * d.n_cells for d in ds) / tot,
            )
        return report

    def print_summary(self):
        rep = self.summary()
        print(f"{'op':<20} {'calls':>6} {'cells':>14} "
              f"{'max|Δ|':>12} {'mean|Δ|':>12} {'max rel':>10} {'mean rel':>10}")
        for op, r in sorted(rep.items(), key=lambda kv: -kv[1]['mean_abs']):
            print(f"{op:<20} {r['calls']:>6} {r['cells']:>14,} "
                  f"{r['max_abs']:>12.4g} {r['mean_abs']:>12.4g} "
                  f"{r['max_rel']:>10.2e} {r['mean_rel']:>10.2e}")
