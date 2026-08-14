"""RescaleClaim — the signed-floor output rescale as a standalone claim.

Every matmul/hadamard in this tree carries an INTERNAL rescale: it commits the
raw product at scale s_a*s_b and binds the caller-visible output at s_out with
two linears and two range LogUps. The routed-projected matmul cannot do that:
its committed output Y is the raw routed accumulator, because Y is what the
projection yr = Y*rho is taken over. So the rescale becomes its own claim that
follows every routed raw output — omitting it is a forbidden regression
(demo/4h-production-runbook.md), since the whole quantized-arithmetic statement
depends on the same signed floor and range bounds the old path enforced.

The relation is EXACTLY the old one, with the same field arithmetic:

    x_full   = (1 << r) * x + x_low            (r = rescale_bits)
    x_shifted = x + 2^(w-1)                    (w = output_width)
    x_low     in [0, 2^r)                      LogUp range, tight table
    x_shifted in [0, 2^w)                      LogUp range, loose table

Together these pin x to the signed floor of x_full / 2^r inside
[-2^(w-1), 2^(w-1)). Constraint ids from `base`: [0, L) the first linear,
[L, 2L) the second; two per-slot quadratic families for the LogUp inverses.
This is the same layout the Rust verifier's `Build::emit_rescale` emits, which
is what the standalone handler calls.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from claims import _emit_lin_csr_idscalar, _per_slot_quad
from core import (
    AUX_FNS, COMPILE_FNS, SAMPLE_FNS, LigeroConfig, Table, Variable,
    _build_b_chunk,
)
from cuda_primitives import P as FIELD_P, gl_inv_batched, gl_sub


@dataclass
class RescaleClaim:
    """x = signed_floor(x_full / 2^rescale_bits), range-bounded to
    output_width bits. x_full is the committed raw accumulator."""
    x_full: Variable
    x: Variable
    x_low: Variable
    x_shifted: Variable
    z_low: Variable
    z_shifted: Variable
    range_rescale: Table
    range_output: Table
    length: int
    rescale_bits: int
    output_width: int = 24


def rescale_sample(c: RescaleClaim, ci: int, s_op):
    """No op challenge: the relation is public linear + LogUp."""
    return None


def rescale_aux(c: RescaleClaim, witness, _ch) -> dict:
    """The two LogUp inverses, z = 1/(alpha - value)."""
    def flat(v):
        t = witness[v]
        return t.contiguous().view(-1)
    low, shifted = flat(c.x_low), flat(c.x_shifted)
    return {
        c.z_low: gl_inv_batched(gl_sub(
            torch.full_like(low, c.range_rescale.alpha), low)),
        c.z_shifted: gl_inv_batched(gl_sub(
            torch.full_like(shifted, c.range_output.alpha), shifted)),
    }


def rescale_compile(c: RescaleClaim, _ch, cfg: LigeroConfig, base: int):
    ell = cfg.ELL
    L = c.length
    neg1 = (FIELD_P - 1) % FIELD_P
    offset = 1 << (c.output_width - 1)
    row_pkts: List[Tuple[int, object]] = []
    # x_full = (1<<r)*x + x_low
    _emit_lin_csr_idscalar(c.x_full, [c.x, c.x_low],
                           [1 << c.rescale_bits, 1], L, ell, base, row_pkts)
    # x_shifted = x + 2^(w-1)
    _emit_lin_csr_idscalar(c.x_shifted, [c.x], [1], L, ell, base + L, row_pkts)
    quads = (_per_slot_quad(f"{c.x.name}.RS[low]", c.x_low, c.z_low, c.z_low,
                            (FIELD_P - c.range_rescale.alpha) % FIELD_P, neg1, L, ell)
             + _per_slot_quad(f"{c.x.name}.RS[shifted]", c.x_shifted, c.z_shifted,
                              c.z_shifted,
                              (FIELD_P - c.range_output.alpha) % FIELD_P, neg1, L, ell))
    return row_pkts, quads, 2 * L, _build_b_chunk(2 * L, [(L, L, offset)])


def rescale(tape, x_full, *, s_in: int, s_out: int, output_width: int = 24):
    """Record a standalone rescale of the raw accumulator `x_full`.

    s_in/s_out are the fixed-point scales; their ratio must be a power of two,
    exactly as the in-matmul rescale required."""
    from tape import WitnessTensor
    from cuda_primitives import lookup_multiplicities_into

    L = x_full.var.length
    ratio = s_in // s_out
    assert s_in == s_out * ratio and ratio > 0 and (ratio & (ratio - 1)) == 0, (
        f"rescale: s_in ({s_in}) must be a power-of-2 multiple of s_out ({s_out})")
    rescale_bits = ratio.bit_length() - 1
    range_rescale = tape._range_table("rescale", rescale_bits)
    range_output = tape._range_table("output", output_width)
    name = f"{x_full.var.name}_rs"
    x = tape._alloc(name, L)
    low, shifted, z_low, z_shifted = tape._emit_rescale_aux(
        name, L, range_rescale, range_output)
    claim = RescaleClaim(
        x_full=x_full.var, x=x, x_low=low, x_shifted=shifted,
        z_low=z_low, z_shifted=z_shifted,
        range_rescale=range_rescale, range_output=range_output,
        length=L, rescale_bits=rescale_bits, output_width=output_width)

    def side_effects(values):
        lookup_multiplicities_into(values[low], range_rescale.T,
                                   tape.inputs[range_rescale.mult_var])
        lookup_multiplicities_into(values[shifted], range_output.T,
                                   tape.inputs[range_output.mult_var],
                                   label=shifted.name)

    outs = tape._process_claim(claim, [x_full.var], side_effects)
    tape.claims.append(claim)
    return WitnessTensor(outs[x] if outs else None, x, x_full.shape, tape)


def rescale_compute(c: RescaleClaim, live) -> dict:
    from tape import _signed_floor_decomp
    x, low, shifted = _signed_floor_decomp(
        live[c.x_full].contiguous().view(-1), 1 << c.rescale_bits, c.output_width)
    return {c.x: x, c.x_low: low, c.x_shifted: shifted}


SAMPLE_FNS[RescaleClaim] = rescale_sample
AUX_FNS[RescaleClaim] = rescale_aux
COMPILE_FNS[RescaleClaim] = rescale_compile

import compute_fns as _cf                      # noqa: E402  (registry wiring)
_cf.COMPUTE_FNS[RescaleClaim] = rescale_compute
