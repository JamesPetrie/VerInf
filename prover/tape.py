"""
Demo: Tape abstraction running a SwiGLU FFN at Llama 2 7B dims.

Real claims, fully wired:
    @  (matmul)    → MatmulClaim
    *  (Hadamard)  → HadamardClaim    (multi-row quadratic split)
    +  (residual)  → AddClaim          (linear, c = a + b)

Stubbed with warnings (identity passthrough, no constraint):
    rmsnorm, softmax, rope

tape.prove() now covers all arithmetic ops in the FFN. The remaining
warnings — rmsnorm — are the genuine paired-tlookup work; silu is full.
"""
import math
import time
import warnings
from typing import Optional

import numpy as np
import torch

from cuda_primitives import (
    P, gl_matmul, gl_mul, gl_add, gl_sub, gl_inv, lookup_multiplicities_into,
)


# ============================================================
# Vectorized witness helpers (avoid Python list-comp at large SEQ).
# ============================================================

_P_NP        = np.uint64(P)
_P_HALF_NP   = np.uint64((P - 1) // 2)
_FIELD_GAP   = np.uint64((1 << 64) - int(P))    # 2^64 − P = 2^32 − 1


def _to_signed_np(uint_arr):
    """uint64 numpy array (values in [0, P)) → signed int64 numpy array."""
    minus_P = (uint_arr - _P_NP)                                           # wraparound
    return np.where(uint_arr <= _P_HALF_NP, uint_arr, minus_P).view(np.int64)


def _to_field_np(signed_arr):
    """signed int64 numpy array → uint64 field-rep."""
    u = signed_arr.view(np.uint64)
    # Adjust: negative int64 viewed as uint64 = 2^64 + signed; field rep wants P + signed.
    # So subtract _FIELD_GAP = 2^64 − P from the uint64 view for negative inputs.
    return np.where(signed_arr >= 0, u, u - _FIELD_GAP)


def _softmax_witness_vec(x_in_uint64, *,
                          B: int, M: int, s_x: int, s_c: int, s_y: int,
                          T_A_np, T_B_np, Z_max: int, aux_chunk_width: int,
                          saturate: bool, Z_high_width: int,
                          causal: bool, heads: int, round_up: bool = False):
    """Vectorized softmax witness construction. Replaces a Python `for b in
    range(B)` loop that does per-row binary search + per-cell aux assembly —
    at SEQ=1000 (B=32k, M=1000) the loop was 32M+ Python ops per layer and
    dominated wall time. All work moves to numpy (C-backed) on CPU.

    Returns a dict of uint64 numpy arrays for every witness this softmax
    needs to commit. Caller converts to torch and registers."""
    # ---- shape input as (B, M) signed int64 ----
    x_signed = _to_signed_np(x_in_uint64).reshape(B, M)

    # ---- causal mask (B, M) ----
    if causal:
        i_qry = (np.arange(B) // heads)[:, None]                           # (B, 1)
        j_idx = np.arange(M)[None, :]                                       # (1, M)
        mask_2d = (j_idx > i_qry)                                           # (B, M) bool
    else:
        mask_2d = np.zeros((B, M), dtype=bool)
    unmasked_2d = ~mask_2d

    # Replace masked x with sentinels so per-row max/min ignore them.
    INT_MIN = np.iinfo(np.int64).min
    INT_MAX = np.iinfo(np.int64).max
    x_for_max = np.where(unmasked_2d, x_signed, INT_MIN)
    x_for_min = np.where(unmasked_2d, x_signed, INT_MAX)
    max_x = x_for_max.max(axis=1)                                           # (B,)
    min_x = x_for_min.min(axis=1)

    def s1_at(c2_b):
        """c2_b: (B,) int64. Returns Σ_{j unmasked, z in [0, Z_max)} T_A[z] per row."""
        z = c2_b[:, None] - x_signed                                        # (B, M)
        in_range = (z >= 0) & (z < Z_max) & unmasked_2d
        z_clamped = np.where(in_range, z, 0)
        T_A_vals = T_A_np[z_clamped]                                        # (B, M) uint64
        # Sum as int64 (T_A values << 2^62; M·max ≤ s_y·M which fits int64 easily).
        s = np.where(in_range, T_A_vals.astype(np.int64), 0)
        if round_up:           # far tokens (unmasked, z >= Z_max) saturate to 1
            s = s + (unmasked_2d & (z >= Z_max)).astype(np.int64)
        return s.sum(axis=1)

    # ---- vectorized binary search for first c2 with s1(c2) ≤ s_y ----
    c2_lo = max_x.astype(np.int64).copy()
    if saturate:
        c2_hi = (max_x + Z_max).astype(np.int64)
    else:
        c2_hi = (min_x + Z_max - 1).astype(np.int64)
        assert (c2_hi >= c2_lo).all(), (
            "softmax: Z_max too small for spread max_x − min_x; pass saturate=True.")
    # Edge: rows where s1(c2_lo) ≤ s_y already. Mark to skip search.
    s1_lo = s1_at(c2_lo)
    skip_search = s1_lo <= s_y
    if not saturate:
        s1_hi = s1_at(c2_hi)
        if (~skip_search & (s1_hi > s_y)).any():
            raise ValueError(
                "softmax: c2 search exceeded Z_max window without s1 ≤ s_y; "
                "pass saturate=True or raise Z_max.")
    # Iterate log2(Z_max) + 1 times — enough for any (B,) row to converge.
    n_iter = max(1, (int(Z_max) - 1).bit_length()) + 2
    for _ in range(n_iter):
        active = (c2_lo + 1 < c2_hi) & ~skip_search
        if not active.any():
            break
        c2_mid = (c2_lo + c2_hi) // 2
        s1_mid = s1_at(c2_mid)
        update_hi = (s1_mid <= s_y) & active
        update_lo = (~(s1_mid <= s_y)) & active
        c2_hi = np.where(update_hi, c2_mid, c2_hi)
        c2_lo = np.where(update_lo, c2_mid, c2_lo)
    c2 = np.where(skip_search, c2_lo, c2_hi)                                # (B,) int64
    s1 = s1_at(c2)                                                          # (B,)

    # ---- per-cell y_A, y_B (0 for masked or out-of-range z) ----
    z_2d = c2[:, None] - x_signed                                           # (B, M)
    in_range_2d = (z_2d >= 0) & (z_2d < Z_max) & unmasked_2d
    z_clamped_2d = np.where(in_range_2d, z_2d, 0)
    y_A_2d = np.where(in_range_2d, T_A_np[z_clamped_2d], np.uint64(0))      # (B, M) uint64
    y_B_2d = np.where(in_range_2d, T_B_np[z_clamped_2d], np.uint64(0))
    if round_up:               # far tokens saturate to 1 (matches s1_at + the mux)
        far_2d = unmasked_2d & (z_2d >= Z_max)
        y_A_2d = np.where(far_2d, np.uint64(1), y_A_2d)
        y_B_2d = np.where(far_2d, np.uint64(1), y_B_2d)
    s2 = y_B_2d.astype(np.int64).sum(axis=1)                                # (B,)

    # ---- bracket residuals + assertions ----
    r_lo = s_y - s1                                                         # (B,)
    r_hi = s2 - s_y - 1
    w_mask = (1 << aux_chunk_width)
    if not ((r_lo >= 0) & (r_lo < w_mask)).all():
        bad = np.where((r_lo < 0) | (r_lo >= w_mask))[0]
        b = int(bad[0])
        raise AssertionError(
            f"softmax[{b}]: r_lo {int(r_lo[b])} out of [0, 2^{aux_chunk_width})")
    if not ((r_hi >= 0) & (r_hi < w_mask)).all():
        bad = np.where((r_hi < 0) | (r_hi >= w_mask))[0]
        b = int(bad[0])
        raise AssertionError(
            f"softmax[{b}]: r_hi {int(r_hi[b])} out of [0, 2^{aux_chunk_width}); "
            "per-step s1 jump exceeds chunk width — raise aux_chunk_width.")
    c2_half_range = 1 << (aux_chunk_width - 1)
    if not ((c2 >= -c2_half_range) & (c2 < c2_half_range)).all():
        bad = np.where((c2 < -c2_half_range) | (c2 >= c2_half_range))[0]
        b = int(bad[0])
        raise AssertionError(
            f"softmax[{b}]: c2 {int(c2[b])} out of "
            f"[-2^{aux_chunk_width-1}, 2^{aux_chunk_width-1}); raise aux_chunk_width.")

    out = {
        "c2":          _to_field_np(c2),                                    # (B,) uint64
        "c2_shifted":  _to_field_np(c2 + c2_half_range),
        "z":           None,                                                 # filled below
        "y_A":         y_A_2d.reshape(-1).astype(np.uint64),
        "y_B":         y_B_2d.reshape(-1).astype(np.uint64),
        "s1":          _to_field_np(s1),
        "s2":          _to_field_np(s2),
        "r_lo":        _to_field_np(r_lo),
        "r_hi":        _to_field_np(r_hi),
    }

    # ---- saturate path: z_low, z_high, is_high, y_*_raw, mux_y_* ----
    if saturate:
        z_low_unmasked  = z_2d % np.int64(Z_max)
        z_high_unmasked = z_2d // np.int64(Z_max)
        # Causal: masked cells get z_low = z_high = 0 (the decomp constraint is
        # skipped for them in compile; their lookup at key=Z_max returns 0).
        if causal:
            z_low_2d  = np.where(mask_2d, np.int64(0), z_low_unmasked)
            z_high_2d = np.where(mask_2d, np.int64(0), z_high_unmasked)
        else:
            z_low_2d, z_high_2d = z_low_unmasked, z_high_unmasked
        if not ((z_high_2d >= 0) & (z_high_2d < (1 << Z_high_width))).all():
            zh_max = int(z_high_2d.max())
            raise AssertionError(
                f"softmax: z_high max {zh_max} out of [0, 2^{Z_high_width}); "
                "raise Z_high_width.")
        is_high_2d = (z_high_2d != 0).astype(np.int64)
        # y_*_raw = T_*[z_low] if not masked, else 0.
        zl_clamped = np.where(mask_2d, 0, z_low_2d).astype(np.intp)
        yA_raw_2d = np.where(mask_2d, np.uint64(0), T_A_np[zl_clamped])
        yB_raw_2d = np.where(mask_2d, np.uint64(0), T_B_np[zl_clamped])
        # mux_y_* = is_high · y_*_raw. is_high ∈ {0, 1}, so this is just a
        # masked passthrough — no multiplication or mod-P needed.
        is_high_bool = is_high_2d.astype(bool)
        mux_yA_2d = np.where(is_high_bool, yA_raw_2d, np.uint64(0))
        mux_yB_2d = np.where(is_high_bool, yB_raw_2d, np.uint64(0))
        # When saturating, c.z is the LOW word.
        out["z"] = z_low_2d.reshape(-1).astype(np.uint64)
        out["z_high"]   = z_high_2d.reshape(-1).astype(np.uint64)
        out["is_high"]  = is_high_2d.reshape(-1).astype(np.uint64)
        out["y_A_raw"]  = yA_raw_2d.reshape(-1).astype(np.uint64)
        out["y_B_raw"]  = yB_raw_2d.reshape(-1).astype(np.uint64)
        out["mux_y_A"]  = mux_yA_2d.reshape(-1).astype(np.uint64)
        out["mux_y_B"]  = mux_yB_2d.reshape(-1).astype(np.uint64)
        # inv_z_high: Fermat inv where z_high>0, 0 otherwise. Computed on GPU
        # (gl_inv_batched) by the caller since numpy can't do mod-P pow.
    else:
        out["z"] = _to_field_np(z_2d.reshape(-1))                            # signed z when not saturating
    return out


def _signed_floor_decomp(c_full_data, k_resc: int, output_width: int):
    """Vectorized signed-floor rescale decomposition on device.

    Inputs: c_full_data — uint64 torch tensor on cuda, length L_out.
    Returns three uint64 cuda tensors: c_rescaled_d, c_low_d, c_shifted_d
    (all field-rep, ready to commit). Bit-identical to the prior numpy
    implementation; stays on device throughout — no host round-trip.

    Implementation note: PyTorch (≤ 2.13 nightly) does not implement
    uint64 arithmetic / comparisons / bitwise / shifts on CUDA, so we
    reinterpret to int64 on entry and stay there until the final
    .view(torch.uint64) cast. int64 add wraps on overflow (verified on
    this build), which is exactly what we need for v in (P_HALF, 2^63).

    Field-rep / signed conversion:
    - Forward: signed = (cf_i ∈ [0, P_HALF]) ? cf_i : cf_i + FIELD_GAP
      where cf_i = c_full_data.view(int64). The `+ FIELD_GAP` is bit-
      identical to the uint64 wraparound `v - P` for v > P_HALF, with
      int64 overflow doing the wraparound for v ∈ (P_HALF, 2^63).
    - Backward: c_signed ≥ 0 ⇒ uint64 view = value; c_signed < 0 ⇒
      uint64 view of (c_signed - FIELD_GAP) = c_signed + P (since
      -FIELD_GAP is the int64 bit pattern of P).

    c_low is always in [0, k_resc) (Python-style remainder with positive
    divisor), so its uint64 view is field-valid without correction.
    c_rescaled and c_shifted may be out-of-range (negative) when the input
    overflows; the range LogUp on c_shifted catches that soundly afterwards.
    """
    FIELD_GAP = (1 << 64) - P                                              # 2^32 - 1, fits in int64
    P_HALF    = (P - 1) // 2                                               # < 2^63, fits in int64

    cf_i      = c_full_data.view(torch.int64)
    # `v > P_HALF` (uint64) ≡ `(cf_i < 0) | (cf_i > P_HALF)` (int64 view).
    is_neg    = (cf_i < 0) | (cf_i > P_HALF)
    cf_signed = torch.where(is_neg, cf_i + FIELD_GAP, cf_i)

    c_rescaled_i = torch.div(cf_signed, k_resc, rounding_mode='floor')
    c_low_i      = torch.remainder(cf_signed, k_resc)
    offset       = 1 << (output_width - 1)
    c_shifted_i  = c_rescaled_i + offset

    # int64-view select: torch CUDA has no `where` for uint64; bits are identical.
    c_rescaled_fld = torch.where(c_rescaled_i >= 0,
                                  c_rescaled_i,
                                  c_rescaled_i - FIELD_GAP).view(torch.uint64)
    c_low_fld      = c_low_i.view(torch.uint64)                            # always ≥ 0
    c_shifted_fld  = torch.where(c_shifted_i >= 0,
                                  c_shifted_i,
                                  c_shifted_i - FIELD_GAP).view(torch.uint64)
    return c_rescaled_fld, c_low_fld, c_shifted_fld
from core import Variable, LigeroConfig, Table
from claims import (
    matmul_claim, AddClaim, HadamardClaim,
    RangeWordClaim, WordExtractionClaim, PairedTlookupClaim,
    SiluConfig, SILU_TOY, SILU_14BIT, SiluClaim, silu_tpos_tneg,
    RmsNormConfig, RmsNormClaim, _chunk_widths, _rms_limb_range_groups, RMS_LIMB_W,
    SoftmaxConfig, SoftmaxClaim, _softmax_exp_tables,
    RoPEConfig, RoPEClaim, _rope_cos_sin,
    EmbeddingLookupClaim,
)
# Imported late (after claims) to avoid the circular tape↔compute_fns import.
import compute_fns as _compute_fns


# ============================================================
# Tape abstraction
# ============================================================

# Rescale convention used by matmul, hadamard, silu, rmsnorm, softmax, rope:
# when an operation's natural output lives at a higher Q-format scale than the
# downstream consumer wants, we commit the high-scale value (`_full` for
# matmul/hadamard/rope; `_internal` for silu/rmsnorm/softmax) and split it
# into low/shifted parts so both halves are range-checkable via LogUp:
#
#   full = shifted * 2^rescale_bits + low
#         where low ∈ [0, 2^rescale_bits)         → range_rescale table
#               shifted ∈ [0, 2^output_width)     → range_output table
#
# Signed values: handled inside _signed_floor_decomp (offset by 2^(w-1)).
# Phase-2 LogUp witnesses for the two range checks are named `_z_low` and
# `_z_shifted`. Tape._emit_rescale_aux allocates the four (low, shifted,
# z_low, z_shifted) Variables and registers the z's against the right tables.

class WitnessTensor:
    def __init__(self, data, var, shape, tape):
        self._data, self.var, self.shape, self.tape = data, var, shape, tape

    @property
    def data(self):
        """Tensor for this Variable, resolving lazy paths transparently.

        Eager mode: _data is set at tape build time.
        Lazy mode (Tape(lazy=True)): _data is None until tape.prove() runs
        the engine pass, which writes computed values back into tape.inputs.
        Either way, `tape.inputs[self.var]` carries the live tensor (or a
        loader callable for commit_lazy weights); resolve and return it."""
        if self._data is not None:
            return self._data
        v = self.tape.inputs.get(self.var)
        return v() if callable(v) else v

    def __matmul__(self, b): return self.tape.matmul(self, b)
    def __mul__(self, b):    return self.tape.hadamard(self, b)
    def __add__(self, b):    return self.tape.add(self, b)


class Tape:
    def __init__(self, cfg, silu_config: SiluConfig = SILU_TOY, lazy: bool = False,
                  time_ops: bool = False):
        """`lazy=True` defers compute_fn dispatch and per-claim side effects
        (mult accumulation) to a single engine pass after tape build. tape.X
        methods just allocate Variables, build the Claim, and record a
        (claim, input_vars, side_effects_closure) entry in self._deferred.
        tape.prove() runs run_engine_pass first, producing a `live` dict
        that plays the role of self.inputs in the eager path.

        Default (lazy=False) keeps the existing eager behavior — tests and
        the unchanged verifier should produce identical results either way.

        `time_ops=True` prints one line per claim with cuda-synchronised wall
        time for compute_fn + side_effects (eager: inside _process_claim;
        lazy: inside run_engine_pass). Useful for finding hot per-op work."""
        self.cfg, self.inputs, self.claims, self._n = cfg, {}, [], 0
        self.silu_config = silu_config
        self.lazy = lazy
        self.time_ops = time_ops
        self._op_idx = 0   # monotonic counter for per-op log labels
        # When lazy: list of (claim, input_var_list, side_effects_callable).
        # side_effects(values) runs lookup_multiplicities_into etc. with
        # `values` = {var: tensor} for both inputs AND outputs of the claim.
        self._deferred: list = []

    def _alloc(self, name, length, phase=1):
        """Allocate a Variable without populating tape.inputs (compute_fn fills it)."""
        self._n += 1
        return Variable(f"{name}#{self._n}", length=length, phase=phase)

    def _log_op_time(self, claim, outs, input_vars, elapsed_s):
        """Emit one line of per-claim timing when time_ops=True. Uses the
        first output Variable's name as a label (falls back to first input
        for claims with no phase-1 output, e.g. RangeWordClaim)."""
        label = (next(iter(outs)).name if outs
                 else (input_vars[0].name if input_vars else "?"))
        print(f"  [op#{self._op_idx:4d} {type(claim).__name__:20s}] "
              f"{elapsed_s:7.4f}s  -> {label}", flush=True)
        self._op_idx += 1

    def _range_table(self, kind: str, width: int):
        """Get-or-create a range LogUp table for `kind` ("rescale" or "output")
        at the given bit width. Tables are shared across all rescale sites
        with matching (kind, width) — registered once, reused everywhere.
        See the "Rescale convention" comment above the WitnessTensor class."""
        cache = f"_range_{kind}_w{width}"
        if not hasattr(self, cache):
            tbl = self.register_table(f"{kind}_w{width}",
                                       T_data=list(range(1 << width)))
            setattr(self, cache, tbl)
        return getattr(self, cache)

    def _emit_rescale_aux(self, prefix: str, L: int, low_tbl, shifted_tbl):
        """Allocate the four standard rescale-decomp aux Variables
        ({prefix}_low, {prefix}_shifted, {prefix}_z_low, {prefix}_z_shifted)
        and register the two phase-2 z's against their range tables.
        Returns (low, shifted, z_low, z_shifted)."""
        low      = self._alloc(f"{prefix}_low",      L)
        shifted  = self._alloc(f"{prefix}_shifted",  L)
        z_low    = Variable(f"{prefix}_z_low",       length=L, phase=2)
        z_shift  = Variable(f"{prefix}_z_shifted",   length=L, phase=2)
        low_tbl.z_vars.append(z_low)
        shifted_tbl.z_vars.append(z_shift)
        return low, shifted, z_low, z_shift

    def _process_claim(self, claim, input_vars, side_effects=None):
        """Eager path: resolve inputs (calling lazy loaders), dispatch via
        COMPUTE_FNS, populate self.inputs for output Variables, then run
        side_effects(values) with both inputs and outputs in `values`.
        Returns the outs dict.

        Lazy path: record (claim, input_vars, side_effects) in self._deferred
        and return None. run_engine_pass will compute and run side effects.

        `input_vars`: iterable of Variable objects this claim reads from
        (lookup keys for tape.inputs / live).
        `side_effects(values)`: optional callable; `values` is a dict
        containing both input and output variables → tensors. Side effects
        typically call lookup_multiplicities_into on output tensors
        (rescale aux, range checks) or input tensors (range_word)."""
        if self.lazy:
            self._deferred.append((claim, tuple(input_vars), side_effects))
            return None
        if self.time_ops:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        # Resolve lazy loaders (commit_lazy): callable values fire here, and
        # the loaded tensor lives only for this dispatch — freed when
        # `resolved` drops out of scope before the next compute_fn call.
        resolved = {v: (val() if callable(val) else val)
                     for v, val in ((v, self.inputs[v]) for v in input_vars)}
        outs = _compute_fns.COMPUTE_FNS[type(claim)](claim, resolved)
        for v, t in outs.items():
            self.inputs[v] = t
        if side_effects is not None:
            side_effects({**resolved, **outs})
        if self.time_ops:
            torch.cuda.synchronize()
            self._log_op_time(claim, outs, input_vars, time.perf_counter() - t0)
        return outs

    def commit(self, name, data, shape, *, persistent=False):
        """`persistent=True` marks a model weight for the persistent W block
        (its own root R_W across proofs; analysis/persistent-weights.md).
        `persistent="new"` places it in the SECOND weight block "wnew" of a
        linking proof (the refreshed commitment's tree; P5)."""
        flat = data.contiguous().view(-1)
        v = Variable(name, length=flat.numel(), phase=1,
                     persistent=bool(persistent), w_new=(persistent == "new"))
        self.inputs[v] = flat
        return WitnessTensor(flat, v, shape, self)

    def commit_lazy(self, name, loader, shape, length, *, persistent: bool = True):
        """Register a Variable whose data is loaded on demand via `loader`.
        tape.inputs[v] stores the callable, not a tensor — used for weight
        commits where holding all weights in memory simultaneously would
        exceed the available pool (e.g. Llama-2-7B 32L single-tape on
        a 121 GB unified-memory system).

        Lazy commits are model weights by default (`persistent=True`) → the
        persistent W block (analysis/persistent-weights.md). Set
        `persistent=False` for a lazily-loaded non-weight input.

        Returns a WitnessTensor whose .data IS the loader callable; tape
        methods that pass it on to compute_fns go through _process_claim,
        which resolves callables before dispatch."""
        v = Variable(name, length=length, phase=1,
                     persistent=bool(persistent), w_new=(persistent == "new"))
        self.inputs[v] = loader
        return WitnessTensor(loader, v, shape, self)

    def matmul(self, a, b, *,
                transpose_b: bool = False,
                heads: int = 1, head_dim: int = 0,
                s_a: Optional[int] = None, s_b: Optional[int] = None,
                s_out: Optional[int] = None, output_width: int = 24):
        """Matmul C = A·B (or A·B^T when transpose_b=True).

        `transpose_b`: when True, B is taken at its committed shape (n, k)
        and the claim verifies C = A · B^T. Avoids committing a transposed
        copy of B for Q·K^T-style attention.

        `heads`, `head_dim`: multi-head batched matmul. The reduction axis
        k splits into (H, K=head_dim) and the matmul becomes H independent
        per-head contractions. Layouts (h-major within row):
          A:  (m, H, K)              flat at i*H*K + h*K + r
          B (transpose_b):  (n, H, K)  flat at j*H*K + h*K + r
          B (non-transpose): (K, H, n) flat at r*H*n + h*n + j
          C:  (m, H, n)              flat at i*H*n + h*n + j
        heads=1 reduces to vanilla matmul; head_dim defaults to k.

        `s_a`, `s_b`, `s_out`: when all set and s_a·s_b > s_out, internally
        rescales the output: caller sees C at scale s_out; raw product
        committed and Freivalds-verified internally."""
        m, k = a.shape
        if head_dim == 0:
            head_dim = k // heads
        assert heads * head_dim == k, (
            f"matmul: heads*head_dim ({heads}*{head_dim}) must equal k ({k})")
        H, K = heads, head_dim
        if transpose_b:
            # B logical (n, H, K). Flat shape: (n, H*K) — second dim = k.
            n, k2 = b.shape
            assert k2 == k, (
                f"matmul transpose_b: b's second dim must equal k={k}; got {k2}")
        else:
            # B logical (K, H, n). Flat shape: (K, H*n) — first dim = K, second = H*n.
            # For heads=1 this is just (k, n), the existing convention.
            k2, n_times_h = b.shape
            assert k2 == K, (
                f"matmul non-transpose: b's first dim must equal head_dim={K}; got {k2}")
            assert n_times_h % H == 0, (
                f"matmul non-transpose: b's second dim must be a multiple of heads={H}; got {n_times_h}")
            n = n_times_h // H
        name = f"{a.var.name}@{b.var.name}{('^T' if transpose_b else '')}"
        # Output shape: (m, n) single-head, (m, H*n) multi-head (flat, with
        # head in the middle of the layout).
        out_shape = (m, n) if heads == 1 else (m, heads * n)
        L_out = m * heads * n

        if s_a is None or s_b is None or s_out is None:
            c_var = self._alloc(name, L_out)
            claim = matmul_claim(
                c_var.name, a.var, b.var, c_var,
                m=m, k=k, n=n, transpose_b=transpose_b,
                heads=heads, head_dim=head_dim)
            outs = self._process_claim(claim, [a.var, b.var])
            self.claims.append(claim)
            return WitnessTensor(outs[c_var] if outs else None, c_var, out_shape, self)

        ratio = (s_a * s_b) // s_out
        assert s_a * s_b == s_out * ratio and ratio > 0 and (ratio & (ratio - 1)) == 0, (
            f"matmul rescale: s_a*s_b ({s_a*s_b}) must be a power-of-2 "
            f"multiple of s_out ({s_out}); got ratio {s_a*s_b/s_out}")
        rescale_bits = ratio.bit_length() - 1
        range_rescale = self._range_table("rescale", rescale_bits)
        range_output  = self._range_table("output",  output_width)

        c_full_var = self._alloc(f"{name}_full", L_out)
        c_var      = self._alloc(name,           L_out)
        c_low_var, c_shifted_var, z_C_low, z_C_shifted = self._emit_rescale_aux(
            name, L_out, range_rescale, range_output)

        claim = matmul_claim(
            c_var.name, a.var, b.var, c_var, m=m, k=k, n=n,
            transpose_b=transpose_b,
            heads=heads, head_dim=head_dim,
            rescale_bits=rescale_bits, output_width=output_width,
            C_full=c_full_var, C_low=c_low_var, C_shifted=c_shifted_var,
            z_C_low=z_C_low, z_C_shifted=z_C_shifted,
            range_rescale=range_rescale, range_output=range_output,
        )
        def side_effects(values):
            lookup_multiplicities_into(values[c_low_var],     range_rescale.T,
                                        self.inputs[range_rescale.mult_var])
            lookup_multiplicities_into(values[c_shifted_var], range_output.T,
                                        self.inputs[range_output.mult_var], label=c_shifted_var.name)
        outs = self._process_claim(claim, [a.var, b.var], side_effects)
        self.claims.append(claim)
        return WitnessTensor(outs[c_var] if outs else None, c_var, out_shape, self)

    def hadamard_broadcast(self, x, gain, *, SEQ: int, d: int,
                            s_a: Optional[int] = None,
                            s_b: Optional[int] = None,
                            s_out: Optional[int] = None,
                            output_width: int = 16):
        """Per-channel multiply: out[seq, j] = x[seq, j] · gain[j].
        Used for the per-channel RmsNorm gain (`rms_pre_*_w`) — gain has
        length d, x has shape (SEQ, d), output has shape (SEQ, d).

        Broadcasts gain via EmbeddingLookupClaim with token_ids=[0]*SEQ
        (so the lookup binds gain_broadcast[seq, j] = gain[j]), then runs
        a regular Hadamard. Inherits Hadamard's rescale plumbing."""
        assert gain.var.length == d, (
            f"hadamard_broadcast: gain length {gain.var.length} != d={d}")
        assert x.var.length == SEQ * d, (
            f"hadamard_broadcast: x length {x.var.length} != SEQ*d={SEQ*d}")
        # gain_bcast = SEQ copies of gain, bound via an EmbeddingLookupClaim.
        # In eager mode compute_fn for that claim materializes the broadcast;
        # in lazy mode it's deferred to the engine pass — same numerics
        # either way, and avoids accessing gain.data at tape-build time
        # (which is None for lazy-mode intermediates / a callable for
        # lazy-committed weights).
        gain_bcast_var = self._alloc(f"{gain.var.name}_bcast", SEQ * d)
        embed_claim = EmbeddingLookupClaim(
            x=gain_bcast_var, E=gain.var, token_ids=[0] * SEQ, d=d)
        outs = self._process_claim(embed_claim, [gain.var])
        self.claims.append(embed_claim)
        gain_bcast = WitnessTensor(
            outs[gain_bcast_var] if outs else None,
            gain_bcast_var, (SEQ, d), self)
        return self.hadamard(x, gain_bcast,
                              s_a=s_a, s_b=s_b, s_out=s_out,
                              output_width=output_width)

    def hadamard(self, a, b, *,
                  s_a: Optional[int] = None, s_b: Optional[int] = None,
                  s_out: Optional[int] = None, output_width: int = 16):
        """Hadamard c[i] = a[i]·b[i]. When s_a, s_b, s_out are all set with
        s_a·s_b > s_out, internally rescales: caller sees c at scale s_out;
        raw product c_full at scale s_a·s_b committed and verified internally."""
        assert a.shape == b.shape, f"hadamard shape mismatch: {a.shape} vs {b.shape}"
        L = a.var.length
        name = f"{a.var.name}*{b.var.name}"

        if s_a is None or s_b is None or s_out is None:
            c_var = self._alloc(name, L)
            claim = HadamardClaim(a=a.var, b=b.var, c=c_var, length=L)
            outs = self._process_claim(claim, [a.var, b.var])
            self.claims.append(claim)
            return WitnessTensor(outs[c_var] if outs else None, c_var, a.shape, self)

        ratio = (s_a * s_b) // s_out
        assert s_a * s_b == s_out * ratio and ratio > 0 and (ratio & (ratio - 1)) == 0, (
            f"hadamard rescale: s_a*s_b ({s_a*s_b}) must be a power-of-2 multiple of s_out ({s_out})")
        rescale_bits = ratio.bit_length() - 1
        range_rescale = self._range_table("rescale", rescale_bits)
        range_output  = self._range_table("output",  output_width)

        c_full_var = self._alloc(f"{name}_full", L)
        c_var      = self._alloc(name,           L)
        c_low_var, c_shifted_var, z_c_low, z_c_shifted = self._emit_rescale_aux(
            name, L, range_rescale, range_output)

        claim = HadamardClaim(
            a=a.var, b=b.var, c=c_var, length=L,
            rescale_bits=rescale_bits, output_width=output_width,
            c_full=c_full_var, c_low=c_low_var, c_shifted=c_shifted_var,
            z_c_low=z_c_low, z_c_shifted=z_c_shifted,
            range_rescale=range_rescale, range_output=range_output,
        )
        def side_effects(values):
            lookup_multiplicities_into(values[c_low_var],     range_rescale.T,
                                        self.inputs[range_rescale.mult_var])
            lookup_multiplicities_into(values[c_shifted_var], range_output.T,
                                        self.inputs[range_output.mult_var], label=c_shifted_var.name)
        outs = self._process_claim(claim, [a.var, b.var], side_effects)
        self.claims.append(claim)
        return WitnessTensor(outs[c_var] if outs else None, c_var, a.shape, self)

    def add(self, a, b):
        assert a.shape == b.shape, f"add shape mismatch: {a.shape} vs {b.shape}"
        L = a.var.length
        c_var = self._alloc(f"{a.var.name}+{b.var.name}", L)
        claim = AddClaim(a=a.var, b=b.var, c=c_var, length=L)
        outs = self._process_claim(claim, [a.var, b.var])
        self.claims.append(claim)
        return WitnessTensor(outs[c_var] if outs else None, c_var, a.shape, self)

    def lincomb(self, xs, coefs, rhs):
        """Public linear pin: sum_k coefs[k]·xs[k][i] = rhs[i] over aligned
        committed vars. `coefs` are ints (negatives allowed, stored mod P);
        `rhs` is one int (all slots) or a per-slot list. Emits no witness —
        pure constraint glue (LinCombClaim)."""
        from claims import LinCombClaim
        L = xs[0].var.length
        rhs_list = rhs if isinstance(rhs, (list, tuple)) else [rhs]
        claim = LinCombClaim(xs=[x.var for x in xs],
                             coefs=[int(c) % P for c in coefs],
                             rhs=[int(v) % P for v in rhs_list], length=L)
        self._process_claim(claim, [x.var for x in xs])
        self.claims.append(claim)
        return claim

    def reveal(self, x, value=None):
        """Expose committed `x` as a PUBLIC value via the equality pin
        x == public_rhs (reuses AddClaim's public-RHS path; no new claim type,
        no Merkle open). `value` is the public constant; in lazy mode it's
        filled AFTER the witness pass (set the returned claim's .public_rhs).
        Returns the AddClaim so the caller can set .public_rhs = <witness value>."""
        claim = AddClaim(a=x.var, b=None, c=None, length=x.var.length,
                         public_rhs=value)
        self._process_claim(claim, [x.var])
        self.claims.append(claim)
        return claim

    def register_table(self, name: str, T_data, T_Y_data=None) -> Table:
        """Allocate a shared-mult LogUp table. T_Y_data set → paired table
        (PairedTlookupClaim against it); T_Y_data None → range table
        (RangeWordClaim against it)."""
        T = torch.tensor(T_data, dtype=torch.uint64, device="cuda")
        T_Y = (torch.tensor(T_Y_data, dtype=torch.uint64, device="cuda")
               if T_Y_data is not None else None)
        T_LEN = T.numel()
        mult_var = Variable(f"{name}_mult", length=T_LEN, phase=1)
        w_var    = Variable(f"{name}_w",    length=T_LEN, phase=2)
        self.inputs[mult_var] = torch.zeros(T_LEN, dtype=torch.uint64, device="cuda")
        return Table(name=name, T=T, T_Y=T_Y, mult_var=mult_var, w_var=w_var)

    def range_word(self, x, table: Table):
        """Assert x ∈ table.T. Returns x itself (now range-proven)."""
        z = Variable(f"{x.var.name}_z", length=x.var.length, phase=2)
        table.z_vars.append(z)
        claim = RangeWordClaim(x=x.var, z=z, table=table, length=x.var.length)
        def side_effects(values):
            lookup_multiplicities_into(values[x.var], table.T,
                                        self.inputs[table.mult_var])
        self._process_claim(claim, [x.var], side_effects)
        self.claims.append(claim)
        return x

    def word_extract(self, x, table: Table, B: int, N: int):
        """Decompose x into N words of B bits each, all range-checked via `table`.
        Returns a list of N WitnessTensors (one per word, low → high)."""
        L = x.var.length
        word_vars = [self._alloc(f"{x.var.name}_w{n}", L) for n in range(N)]
        claim = WordExtractionClaim(
            x=x.var, words=word_vars,
            coeffs=[(1 << (n * B)) % P for n in range(N)], length=L)
        outs = self._process_claim(claim, [x.var])
        self.claims.append(claim)
        word_wts = [WitnessTensor(outs[v] if outs else None, v, x.shape, self)
                     for v in word_vars]
        for wt in word_wts:
            self.range_word(wt, table)
        return word_wts

    def concat(self, srcs, shape):
        """dst = srcs[0] ‖ srcs[1] ‖ … (ConcatClaim — per-slot identity pins)."""
        from claims import ConcatClaim
        L = sum(w.var.length for w in srcs)
        dst = self._alloc("cat_" + "_".join(w.var.name[:8] for w in srcs), L)
        claim = ConcatClaim(srcs=[w.var for w in srcs], dst=dst)
        outs = self._process_claim(claim, [w.var for w in srcs])
        self.claims.append(claim)
        return WitnessTensor(outs[dst] if outs else None, dst, shape, self)

    def paired_tlookup(self, x, table: Table, shift: int = 0, y_var=None):
        """Look up (x[i] + shift, y[i]) ∈ (T, T_Y). The shift lets `x` be a
        signed Q-form value while the public table indexes from 0; the
        protocol constraint folds shift in as the b of the u-binding
        linear constraint (no extra claim needed). Computes y eagerly via
        T_Y[x + shift], commits y, records the PairedTlookupClaim, and
        accumulates mult against x + shift (not raw x)."""
        assert table.T_Y is not None, "paired_tlookup requires a Table with T_Y set"
        L = x.var.length
        # y_var reuse: bind an EXISTING derived variable to the table (the UI
        # log-pin pw is derived by InfoFinalize, then bound here; both compute
        # fns write the same value).
        if y_var is None:
            y_var = self._alloc(f"{x.var.name}_lookup", L)
        u = Variable(f"{x.var.name}_pt_u", length=L, phase=2)
        z = Variable(f"{x.var.name}_pt_z", length=L, phase=2)
        table.z_vars.append(z)
        claim = PairedTlookupClaim(
            x=x.var, y=y_var, u=u, z=z, table=table, length=L, shift=shift)
        def side_effects(values):
            x_val = values[x.var]
            shift_t = torch.full_like(x_val, shift % P)
            lookup_multiplicities_into(gl_add(x_val, shift_t), table.T,
                                        self.inputs[table.mult_var])
        outs = self._process_claim(claim, [x.var], side_effects)
        self.claims.append(claim)
        return WitnessTensor(outs[y_var] if outs else None, y_var, x.shape, self)

    def silu(self, x, *, s_in: Optional[int] = None):
        """Eager-compute all silu witnesses, then emit one `SiluClaim`.

        The Tape's job is just witness creation + table registration. All
        constraint emission (7 linear + 12 quadratic per slot) happens inside
        SiluClaim.silu_compile in claims.py. See SiluClaim for the
        full relation and soundness argument.

        `s_in` lets the caller commit x at a coarser scale than s_x = 2^r
        (e.g., post-matmul at scale s_x²). When `s_in > s_x`, the claim
        internally word-decomposes x_in into (x_low, x) such that
        x_in = (s_in/s_x)·x + x_low, with both pieces range-checked. The
        existing sign-magnitude + lookup machinery then runs on `x`."""
        if not hasattr(self, "_silu_state"):
            sc = self.silu_config
            range_b = self.register_table(
                f"silu_range_{sc.b}", T_data=list(range(sc.b)))
            range_w2 = self.register_table(
                f"silu_range_w{sc.width_2}", T_data=list(range(1 << sc.width_2)))
            range_w3 = (range_w2 if sc.width_3 == sc.width_2 else
                        self.register_table(
                            f"silu_range_w{sc.width_3}",
                            T_data=list(range(1 << sc.width_3))))
            range_w4 = (range_w2 if sc.width_4 == sc.width_2 else
                        range_w3 if sc.width_4 == sc.width_3 else
                        self.register_table(
                            f"silu_range_w{sc.width_4}",
                            T_data=list(range(1 << sc.width_4))))
            T_pos, T_neg = silu_tpos_tneg(sc)
            silu_table = self.register_table(
                "silu_paired",
                T_data=list(range(2 * sc.T_LEN)),
                T_Y_data=T_pos + T_neg,
            )
            self._silu_state = (range_b, range_w2, range_w3, range_w4, silu_table)
        range_b, range_w2, range_w3, range_w4, silu_table = self._silu_state
        # Build a per-call SiluConfig that carries the (optional) s_in. Use
        # self.silu_config's existing knobs unchanged.
        base_sc = self.silu_config
        s_x = 1 << base_sc.r
        s_in_eff = s_x if s_in is None else s_in
        sc = SiluConfig(
            b=base_sc.b, T_LEN=base_sc.T_LEN,
            b_2=base_sc.b_2, b_3=base_sc.b_3, b_4=base_sc.b_4,
            width_2=base_sc.width_2, width_3=base_sc.width_3, width_4=base_sc.width_4,
            r=base_sc.r, s_in=s_in_eff,
        )
        rescale_bits = sc.rescale_bits
        L = x.var.length

        # Rescale path: register rescale table; allocate x_low/x_internal/x_shifted Vars.
        if rescale_bits > 0:
            r_resc = rescale_bits
            k_resc = 1 << r_resc
            range_rescale = self._range_table("rescale", r_resc)
            x_internal_var = self._alloc(f"{x.var.name}_silu_x_internal", L)
            x_low_var, x_shifted_var, z_x_low_v, z_x_shifted_v = self._emit_rescale_aux(
                f"{x.var.name}_silu_x", L, range_rescale, range_w2)
            x_in_var = x.var
        else:
            x_low_var = x_shifted_var = None
            x_internal_var = x.var
            x_in_var = None
            z_x_low_v = z_x_shifted_v = None
            range_rescale = None

        # Allocate phase-1 Variables in the same order tape used to commit them
        # (so #counter suffixes match the legacy proof for debugging).
        sign_var       = self._alloc(f"{x.var.name}_silu_sign", L)
        magnitude_var  = self._alloc(f"{x.var.name}_silu_mag",  L)
        C_var          = self._alloc(f"{x.var.name}_silu_C",    L)
        a_0_var        = self._alloc(f"{x.var.name}_silu_a0",   L)
        a_1_var        = self._alloc(f"{x.var.name}_silu_a1",   L)
        a_2_var        = self._alloc(f"{x.var.name}_silu_a2",   L)
        a_3_var        = self._alloc(f"{x.var.name}_silu_a3",   L)
        a_4_var        = self._alloc(f"{x.var.name}_silu_a4",   L)
        g_var          = self._alloc(f"{x.var.name}_silu_g",    L)
        inv_g_var      = self._alloc(f"{x.var.name}_silu_invg", L)
        is_high_var    = self._alloc(f"{x.var.name}_silu_ish",  L)
        key_var        = self._alloc(f"{x.var.name}_silu_key",  L)
        output_sat_var = self._alloc(f"{x.var.name}_silu_osat", L)
        mux_a_var      = self._alloc(f"{x.var.name}_silu_muxa", L)
        mux_b_var      = self._alloc(f"{x.var.name}_silu_muxb", L)
        y_var          = self._alloc(f"{x.var.name}_silu_y",    L)
        output_var     = self._alloc(f"{x.var.name}_silu_out",  L)

        # Phase-2 LogUp z slots — values filled in by silu_aux after α/β.
        pt_u = Variable(f"{x.var.name}_silu_pt_u", length=L, phase=2)
        pt_z = Variable(f"{x.var.name}_silu_pt_z", length=L, phase=2)
        z_a0 = Variable(f"{x.var.name}_silu_z_a0", length=L, phase=2)
        z_a2 = Variable(f"{x.var.name}_silu_z_a2", length=L, phase=2)
        z_a3 = Variable(f"{x.var.name}_silu_z_a3", length=L, phase=2)
        z_a4 = Variable(f"{x.var.name}_silu_z_a4", length=L, phase=2)
        silu_table.z_vars.append(pt_z)
        range_b.z_vars.append(z_a0)
        range_w2.z_vars.append(z_a2)
        range_w3.z_vars.append(z_a3)
        range_w4.z_vars.append(z_a4)

        claim = SiluClaim(
            x=x_internal_var, output=output_var, length=L, config=sc,
            sign=sign_var, magnitude=magnitude_var, C=C_var,
            a_0=a_0_var, a_1=a_1_var, a_2=a_2_var, a_3=a_3_var, a_4=a_4_var,
            g=g_var, inv_g=inv_g_var, is_high=is_high_var,
            key=key_var, output_sat=output_sat_var,
            mux_a=mux_a_var, mux_b=mux_b_var, y=y_var,
            pt_u=pt_u, pt_z=pt_z,
            z_a0=z_a0, z_a2=z_a2, z_a3=z_a3, z_a4=z_a4,
            silu_table=silu_table, range_b=range_b,
            range_w2=range_w2, range_w3=range_w3, range_w4=range_w4,
            x_in=x_in_var,
            x_low=x_low_var, x_shifted=x_shifted_var,
            z_x_low=z_x_low_v, z_x_shifted=z_x_shifted_v,
            range_rescale=range_rescale,
            range_x=(range_w2 if rescale_bits > 0 else None),
        )
        def side_effects(values):
            lookup_multiplicities_into(values[key_var], silu_table.T,
                                        self.inputs[silu_table.mult_var])
            for chunk_var, tbl in [(a_0_var, range_b), (a_2_var, range_w2),
                                    (a_3_var, range_w3), (a_4_var, range_w4)]:
                lookup_multiplicities_into(values[chunk_var], tbl.T,
                                            self.inputs[tbl.mult_var])
            if rescale_bits > 0:
                lookup_multiplicities_into(values[x_low_var], range_rescale.T,
                                            self.inputs[range_rescale.mult_var])
                lookup_multiplicities_into(values[x_shifted_var], range_w2.T,
                                            self.inputs[range_w2.mult_var])
        outs = self._process_claim(claim, [x.var], side_effects)
        self.claims.append(claim)
        return WitnessTensor(outs[output_var] if outs else None, output_var, x.shape, self)

    def rmsnorm(self, x, *, d: int, s: int = 4, eps_int: int = 1,
                s_in: Optional[int] = None,
                s_out: Optional[int] = None, output_width: int = 16):
        """Batched RMSNorm: y[b] = 1/√(mean(x[b]²) + ε), output = x·broadcast(y).

        `s_in` rescales the input (s_in > s ⇒ internal word decomp).
        `s_out` rescales the output (output naturally at scale s²; rescale
        to s_out via word decomp). With s_out, caller sees output at s_out;
        the raw product `output_full` is committed internally and the
        Freivalds linear binds to output_full.

        All range-window widths (slack, y, limb carries) are DERIVED from
        (d, s, eps_int) — see RmsNormConfig and rmsnorm-bracket-fix.md."""
        L = x.var.length
        assert L % d == 0, f"rmsnorm: x length {L} not divisible by d={d}"
        B = L // d
        s_in_eff = s if s_in is None else s_in
        s_out_eff = 0 if s_out is None else s_out
        sc = RmsNormConfig(B=B, d=d, s=s, eps_int=eps_int,
                            s_in=s_in_eff,
                            s_out=s_out_eff,
                            output_width=output_width)
        rescale_bits = sc.rescale_bits
        output_rescale_bits = sc.output_rescale_bits
        magic = sc.magic

        # Cache the 16-bit slack range table on first call.
        cache_key = "_rmsnorm_range_w16"
        if not hasattr(self, cache_key):
            range_slack = self.register_table(
                "rmsnorm_range_w16",
                T_data=list(range(1 << 16)),
            )
            setattr(self, cache_key, range_slack)
        range_slack = getattr(self, cache_key)

        # 2^LIMB_W table for the S_total limbs and the carry lows g0l/g1l.
        range_limb = self._range_table("rms", RMS_LIMB_W)

        # Top-chunk tables for the derived windows (None when the window is a
        # multiple of 16 — then every chunk checks against range_slack).
        def top_table(width):
            rem = width % 16 if width >= 16 else width
            return self._range_table("rms", rem) if rem else None
        range_y_top     = top_table(sc.y_width)
        range_slack_top = top_table(sc.slack_width)
        range_g0h_top   = top_table(sc.g0h_width)
        range_g1h_top   = top_table(sc.g1h_width)
        range_G2_top    = top_table(sc.G2_width)
        K = len(_chunk_widths(sc.slack_width))

        # Input rescale aux (when s_in > s).
        if rescale_bits > 0:
            range_rescale = self._range_table("rescale", rescale_bits)
            x_internal_var = self._alloc(f"{x.var.name}_rms_x_internal", L)
            x_low_var, x_shifted_var, z_x_low, z_x_shifted = self._emit_rescale_aux(
                f"{x.var.name}_rms_x", L, range_rescale, range_slack)
            x_in_var = x.var
        else:
            x_low_var = x_shifted_var = None
            x_internal_var = x.var
            x_in_var = None
            z_x_low = z_x_shifted = None
            range_rescale = None

        # Output rescale tables (when s_out is set).
        if output_rescale_bits > 0:
            range_output_rescale = self._range_table("rescale", output_rescale_bits)
            range_output         = self._range_table("output",  output_width)
        else:
            range_output_rescale = range_output = None

        # Allocate phase-1 Variables (order matches the phase-1 commit order).
        X_sq_var     = self._alloc(f"{x.var.name}_rms_Xsq",     L)
        S_var        = self._alloc(f"{x.var.name}_rms_S",       B)
        S_total_var  = self._alloc(f"{x.var.name}_rms_S_total", B)
        y_var        = self._alloc(f"{x.var.name}_rms_y",       B)
        y_m1_var     = self._alloc(f"{x.var.name}_rms_y_m1",    B)
        q1_var       = self._alloc(f"{x.var.name}_rms_q1",      B)
        q2_var       = self._alloc(f"{x.var.name}_rms_q2",      B)
        s_lo_var     = self._alloc(f"{x.var.name}_rms_s_lo",    B)
        s_hi_var     = self._alloc(f"{x.var.name}_rms_s_hi",    B)
        output_var   = self._alloc(f"{x.var.name}_rms_out",     L)
        s_lo_chunks_vars = [self._alloc(f"{x.var.name}_rms_s_lo_c{n}", B)
                             for n in range(K)]
        s_hi_chunks_vars = [self._alloc(f"{x.var.name}_rms_s_hi_c{n}", B)
                             for n in range(K)]
        # Wrap-free-bracket limb witnesses (rmsnorm-bracket-fix.md).
        pfx = x.var.name
        ym1_chunks_vars = [self._alloc(f"{pfx}_rms_ym1_c{n}", B)
                            for n in range(len(_chunk_widths(sc.y_width)))]
        S_limbs_vars = [self._alloc(f"{pfx}_rms_Slimb{n}", B) for n in range(3)]
        def _alloc_bracket(tag):
            return dict(
                H=[self._alloc(f"{pfx}_rms_{tag}_H{k}", B) for k in range(3)],
                gl=[self._alloc(f"{pfx}_rms_{tag}_g{k}l", B) for k in range(2)],
                g0h=[self._alloc(f"{pfx}_rms_{tag}_g0h_c{j}", B)
                     for j in range(len(_chunk_widths(sc.g0h_width)))],
                g1h=[self._alloc(f"{pfx}_rms_{tag}_g1h_c{j}", B)
                     for j in range(len(_chunk_widths(sc.g1h_width)))],
                G2=[self._alloc(f"{pfx}_rms_{tag}_G2_c{j}", B)
                    for j in range(len(_chunk_widths(sc.G2_width)))])
        lo = _alloc_bracket("lo")
        hi = _alloc_bracket("hi")
        if output_rescale_bits > 0:
            output_full_var = self._alloc(f"{x.var.name}_rms_out_full", L)
            output_low_var, output_shifted_var, z_output_low, z_output_shifted = \
                self._emit_rescale_aux(f"{x.var.name}_rms_out", L,
                                        range_output_rescale, range_output)
        else:
            output_full_var = output_low_var = output_shifted_var = None
            z_output_low = z_output_shifted = None

        # Phase-2 slots.
        u = Variable(f"{x.var.name}_rms_u", length=B, phase=2)
        p = Variable(f"{x.var.name}_rms_p", length=B, phase=2)
        z_lo_chunks = [Variable(f"{x.var.name}_rms_z_lo_c{n}", length=B, phase=2)
                        for n in range(K)]
        z_hi_chunks = [Variable(f"{x.var.name}_rms_z_hi_c{n}", length=B, phase=2)
                        for n in range(K)]
        slack_widths = _chunk_widths(sc.slack_width)
        for n, z in enumerate(z_lo_chunks):
            (range_slack if slack_widths[n] == 16 else range_slack_top).z_vars.append(z)
        for n, z in enumerate(z_hi_chunks):
            (range_slack if slack_widths[n] == 16 else range_slack_top).z_vars.append(z)
        def _z_list(vars_):
            return [Variable(f"z_{v.name}", length=B, phase=2) for v in vars_]
        z_ym1   = _z_list(ym1_chunks_vars)
        z_Slimb = _z_list(S_limbs_vars)
        z_lo = {k: _z_list(lo[k]) for k in ("gl", "g0h", "g1h", "G2")}
        z_hi = {k: _z_list(hi[k]) for k in ("gl", "g0h", "g1h", "G2")}

        claim = RmsNormClaim(
            x=x_internal_var, output=output_var, config=sc,
            X_sq=X_sq_var, S=S_var, S_total=S_total_var,
            y=y_var, y_m1=y_m1_var, q1=q1_var, q2=q2_var,
            s_lo=s_lo_var, s_hi=s_hi_var,
            s_lo_chunks=s_lo_chunks_vars, s_hi_chunks=s_hi_chunks_vars,
            ym1_chunks=ym1_chunks_vars, S_limbs=S_limbs_vars,
            lo_H=lo["H"], lo_gl=lo["gl"], lo_g0h_chunks=lo["g0h"],
            lo_g1h_chunks=lo["g1h"], lo_G2_chunks=lo["G2"],
            hi_H=hi["H"], hi_gl=hi["gl"], hi_g0h_chunks=hi["g0h"],
            hi_g1h_chunks=hi["g1h"], hi_G2_chunks=hi["G2"],
            u=u, p=p,
            z_lo_chunks=z_lo_chunks, z_hi_chunks=z_hi_chunks,
            z_ym1_chunks=z_ym1, z_S_limbs=z_Slimb,
            z_lo_gl=z_lo["gl"], z_lo_g0h=z_lo["g0h"],
            z_lo_g1h=z_lo["g1h"], z_lo_G2=z_lo["G2"],
            z_hi_gl=z_hi["gl"], z_hi_g0h=z_hi["g0h"],
            z_hi_g1h=z_hi["g1h"], z_hi_G2=z_hi["G2"],
            range_slack=range_slack, range_limb=range_limb,
            range_y_top=range_y_top, range_slack_top=range_slack_top,
            range_g0h_top=range_g0h_top, range_g1h_top=range_g1h_top,
            range_G2_top=range_G2_top,
            x_in=x_in_var, x_low=x_low_var, x_shifted=x_shifted_var,
            z_x_low=z_x_low, z_x_shifted=z_x_shifted,
            range_rescale=range_rescale,
            output_full=output_full_var,
            output_low=output_low_var,
            output_shifted=output_shifted_var,
            z_output_low=z_output_low,
            z_output_shifted=z_output_shifted,
            range_output_rescale=range_output_rescale,
            range_output=range_output,
        )
        # Register the limb z's on their tables (frozen group order).
        limb_groups = _rms_limb_range_groups(claim)
        for vars_, zs, tbls in limb_groups:
            for z, tbl in zip(zs, tbls):
                tbl.z_vars.append(z)

        def side_effects(values):
            for n, var in enumerate(s_lo_chunks_vars):
                tbl = range_slack if slack_widths[n] == 16 else range_slack_top
                lookup_multiplicities_into(values[var], tbl.T,
                                            self.inputs[tbl.mult_var])
            for n, var in enumerate(s_hi_chunks_vars):
                tbl = range_slack if slack_widths[n] == 16 else range_slack_top
                lookup_multiplicities_into(values[var], tbl.T,
                                            self.inputs[tbl.mult_var])
            for vars_, _zs, tbls in limb_groups:
                for var, tbl in zip(vars_, tbls):
                    lookup_multiplicities_into(values[var], tbl.T,
                                                self.inputs[tbl.mult_var])
            if rescale_bits > 0:
                lookup_multiplicities_into(values[x_low_var], range_rescale.T,
                                            self.inputs[range_rescale.mult_var])
                lookup_multiplicities_into(values[x_shifted_var], range_slack.T,
                                            self.inputs[range_slack.mult_var])
            if output_rescale_bits > 0:
                lookup_multiplicities_into(values[output_low_var], range_output_rescale.T,
                                            self.inputs[range_output_rescale.mult_var])
                lookup_multiplicities_into(values[output_shifted_var], range_output.T,
                                            self.inputs[range_output.mult_var], label=output_shifted_var.name)
        outs = self._process_claim(claim, [x.var], side_effects)
        self.claims.append(claim)
        return WitnessTensor(outs[output_var] if outs else None, output_var, x.shape, self)

    def softmax(self, x, *, M: int, s_x: int = 4, s_c: Optional[int] = None,
                s_y: int = 16, delta: int = 1, Z_max: int = 16,
                aux_chunk_width: int = 16,
                s_in: Optional[int] = None,
                saturate: bool = False,
                Z_high_width: int = 16,
                causal: bool = False,
                heads: int = 1,
                round_up: bool = False):
        """Batched softmax via §2.13 two-table bracket. The tight bracket
        s1 ≤ s_y AND s2 ≥ s_y+1 pins c2 to the unique integer crossing —
        s2(c2) = s1(c2−δ) bit-identically via the table construction (see
        _softmax_exp_tables), so monotonicity of integer s1 in c2 gives
        uniqueness with no slack.

        `s_in` lets the caller commit x at a coarser scale than s_x (e.g.,
        directly post-matmul). When `s_in > s_x`, the claim internally word-
        decomposes x_in into (x_low, x) such that x_in = (s_in/s_x)·x + x_low,
        with both pieces range-checked. The bracket then operates on `x`.

        `saturate=True` enables the SiLU-style high-z mux: z = z_low +
        Z_max·z_high (z_low ∈ [0, Z_max), z_high ∈ [0, 2^Z_high_width)),
        with y_A/y_B forced to 0 when z_high ≠ 0. Lets Z_max be sized to
        the non-zero exp region (~9·s_c) rather than the full c2 − min(x)
        spread. One-sided — c2 ≥ max(x) is enforced automatically (a
        too-small c2 makes z negative, which can't be decomposed into
        nonneg z_low + Z_max·z_high under the range tables → REJECT)."""
        if s_c is None:
            s_c = s_x        # by convention
        if round_up:
            assert saturate and not causal, \
                "softmax round_up requires saturate=True and non-causal"
        L = x.var.length
        assert L % M == 0, f"softmax: x length {L} not divisible by M={M}"
        B = L // M
        s_in_eff = s_x if s_in is None else s_in
        sc = SoftmaxConfig(B=B, M=M, s_x=s_x, s_c=s_c, s_y=s_y,
                            delta=delta, Z_max=Z_max, s_in=s_in_eff,
                            saturate=saturate, Z_high_width=Z_high_width,
                            aux_chunk_width=aux_chunk_width,
                            causal=causal, heads=heads, round_up=round_up)
        rescale_bits = sc.rescale_bits

        # Cache the two exp tables + the shared aux range table.
        # When causal=True, the tables are doubled: T_combined = T_A || T_zero
        # (size 2·Z_max). Masked cells look up in the upper [Z_max, 2·Z_max)
        # range and naturally get y = 0; unmasked cells use the lower half
        # exactly as in non-causal softmax.
        cache_attr = f"_softmax_state_{Z_max}_{s_c}_{s_y}_{delta}_c{int(causal)}"
        if not hasattr(self, cache_attr):
            T_A_data, T_B_data = _softmax_exp_tables(sc)   # np.uint64 arrays
            if causal:
                zeros = np.zeros(Z_max, dtype=np.uint64)
                T_A_combined = np.concatenate([T_A_data, zeros])
                T_B_combined = np.concatenate([T_B_data, zeros])
                Z_table = 2 * Z_max
                tab_suffix = f"Z{Z_max}_causal"
            else:
                T_A_combined = T_A_data
                T_B_combined = T_B_data
                Z_table = Z_max
                tab_suffix = f"Z{Z_max}"
            # np.arange key (not list(range(...))): at Z_max ~ 10^8 a Python list
            # would dominate the build time + memory.
            keys = np.arange(Z_table, dtype=np.uint64)
            exp_A = self.register_table(
                f"sm_exp_A_{tab_suffix}", T_data=keys, T_Y_data=T_A_combined)
            exp_B = self.register_table(
                f"sm_exp_B_{tab_suffix}", T_data=keys, T_Y_data=T_B_combined)
            setattr(self, cache_attr, (exp_A, exp_B))
        exp_A, exp_B = getattr(self, cache_attr)
        range_aux_attr = f"_rmsnorm_range_w{aux_chunk_width}"
        if not hasattr(self, range_aux_attr):
            range_aux = self.register_table(
                f"sm_range_w{aux_chunk_width}",
                T_data=list(range(1 << aux_chunk_width)))
            setattr(self, range_aux_attr, range_aux)
        range_aux = getattr(self, range_aux_attr)

        if saturate:
            zh_attr = f"_softmax_range_zhigh_w{Z_high_width}"
            if not hasattr(self, zh_attr):
                range_z_high_t = self.register_table(
                    f"sm_range_zhigh_w{Z_high_width}",
                    T_data=list(range(1 << Z_high_width)))
                setattr(self, zh_attr, range_z_high_t)
            range_z_high = getattr(self, zh_attr)
        else:
            range_z_high = None

        L = x.var.length

        # Internal rescale aux (when s_in > s_x).
        if rescale_bits > 0:
            range_rescale = self._range_table("rescale", rescale_bits)
            x_internal_var = self._alloc(f"{x.var.name}_sm_x_internal", L)
            x_low_var, x_shifted_var, z_x_low_v, z_x_shifted_v = self._emit_rescale_aux(
                f"{x.var.name}_sm_x", L, range_rescale, range_aux)
            x_in_var = x.var
        else:
            x_low_var = x_shifted_var = None
            x_internal_var = x.var
            x_in_var = None
            z_x_low_v = z_x_shifted_v = None
            range_rescale = None

        # Allocate phase-1 Variables (order matches legacy commit order).
        c2_var         = self._alloc(f"{x.var.name}_sm_c2",         B)
        c2_shifted_var = self._alloc(f"{x.var.name}_sm_c2_shifted", B)
        z_var   = self._alloc(f"{x.var.name}_sm_z",    L)
        y_A_var = self._alloc(f"{x.var.name}_sm_y_A",  L)
        y_B_var = self._alloc(f"{x.var.name}_sm_y_B",  L)
        s1_var  = self._alloc(f"{x.var.name}_sm_s1",   B)
        s2_var  = self._alloc(f"{x.var.name}_sm_s2",   B)
        r_lo_var = self._alloc(f"{x.var.name}_sm_r_lo", B)
        r_hi_var = self._alloc(f"{x.var.name}_sm_r_hi", B)
        if saturate:
            zh_var     = self._alloc(f"{x.var.name}_sm_z_high",   L)
            inv_zh_var = self._alloc(f"{x.var.name}_sm_inv_zh",   L)
            ish_var    = self._alloc(f"{x.var.name}_sm_is_high",  L)
            yA_raw_var = self._alloc(f"{x.var.name}_sm_y_A_raw",  L)
            yB_raw_var = self._alloc(f"{x.var.name}_sm_y_B_raw",  L)
            mux_yA_var = self._alloc(f"{x.var.name}_sm_mux_y_A",  L)
            mux_yB_var = self._alloc(f"{x.var.name}_sm_mux_y_B",  L)
        else:
            zh_var = inv_zh_var = ish_var = None
            yA_raw_var = yB_raw_var = mux_yA_var = mux_yB_var = None

        # Phase-2 slots.
        pt_u_A = Variable(f"{x.var.name}_sm_pt_u_A", length=L, phase=2)
        pt_z_A = Variable(f"{x.var.name}_sm_pt_z_A", length=L, phase=2)
        pt_u_B = Variable(f"{x.var.name}_sm_pt_u_B", length=L, phase=2)
        pt_z_B = Variable(f"{x.var.name}_sm_pt_z_B", length=L, phase=2)
        z_c2_v   = Variable(f"{x.var.name}_sm_z_c2",   length=B, phase=2)
        z_r_lo_v = Variable(f"{x.var.name}_sm_z_r_lo", length=B, phase=2)
        z_r_hi_v = Variable(f"{x.var.name}_sm_z_r_hi", length=B, phase=2)
        exp_A.z_vars.append(pt_z_A)
        exp_B.z_vars.append(pt_z_B)
        range_aux.z_vars.append(z_c2_v)
        range_aux.z_vars.append(z_r_lo_v)
        range_aux.z_vars.append(z_r_hi_v)
        if saturate:
            z_zh_v = Variable(f"{x.var.name}_sm_z_z_high", length=L, phase=2)
            range_z_high.z_vars.append(z_zh_v)
        else:
            z_zh_v = None

        claim = SoftmaxClaim(
            x=x_internal_var, y_A=y_A_var, config=sc, length=L,
            c2=c2_var, c2_shifted=c2_shifted_var, z=z_var, y_B=y_B_var,
            s1=s1_var, s2=s2_var, r_lo=r_lo_var, r_hi=r_hi_var,
            pt_u_A=pt_u_A, pt_z_A=pt_z_A,
            pt_u_B=pt_u_B, pt_z_B=pt_z_B,
            z_c2=z_c2_v, z_r_lo=z_r_lo_v, z_r_hi=z_r_hi_v,
            exp_A=exp_A, exp_B=exp_B, range_aux=range_aux,
            x_in=x_in_var,
            x_low=x_low_var, x_shifted=x_shifted_var,
            z_x_low=z_x_low_v, z_x_shifted=z_x_shifted_v,
            range_rescale=range_rescale,
            z_high=zh_var, inv_z_high=inv_zh_var, is_high=ish_var,
            y_A_raw=yA_raw_var, y_B_raw=yB_raw_var,
            mux_y_A=mux_yA_var, mux_y_B=mux_yB_var,
            z_z_high=z_zh_v,
            range_z_high=range_z_high,
        )
        def side_effects(values):
            if causal:
                i_flat = torch.arange(L, dtype=torch.int64, device="cuda")
                mask_shift_d = ((i_flat % M > i_flat // M // heads).to(torch.int64) * Z_max).to(torch.uint64)
                exp_key_d = gl_add(values[z_var], mask_shift_d)
            else:
                exp_key_d = values[z_var]
            lookup_multiplicities_into(exp_key_d, exp_A.T, self.inputs[exp_A.mult_var])
            lookup_multiplicities_into(exp_key_d, exp_B.T, self.inputs[exp_B.mult_var])
            if saturate:
                lookup_multiplicities_into(values[zh_var], range_z_high.T,
                                            self.inputs[range_z_high.mult_var])
            for v in (values[c2_shifted_var], values[r_lo_var], values[r_hi_var]):
                lookup_multiplicities_into(v, range_aux.T, self.inputs[range_aux.mult_var])
            if rescale_bits > 0:
                lookup_multiplicities_into(values[x_low_var], range_rescale.T,
                                            self.inputs[range_rescale.mult_var])
                lookup_multiplicities_into(values[x_shifted_var], range_aux.T,
                                            self.inputs[range_aux.mult_var])
        outs = self._process_claim(claim, [x.var], side_effects)
        self.claims.append(claim)
        return WitnessTensor(outs[y_A_var] if outs else None, y_A_var, x.shape, self)

    def embed(self, E_wt, token_ids: list, d: int):
        """Look up embedding rows: x[i, :] = E[token_ids[i], :] for public
        token_ids and committed embedding table `E_wt` (shape (vocab_size, d)).
        Commits the resulting x and emits an EmbeddingLookupClaim binding
        x to E at the public token-id positions.

        Returns the committed x as a WitnessTensor of shape (len(token_ids), d)."""
        SEQ = len(token_ids)
        vocab_size = E_wt.var.length // d
        assert E_wt.var.length == vocab_size * d, (
            f"embed: E_wt has length {E_wt.var.length}, not divisible by d={d}")
        x_var = self._alloc(f"x_embed_{'_'.join(map(str, token_ids))}", SEQ * d)
        claim = EmbeddingLookupClaim(
            x=x_var, E=E_wt.var, token_ids=list(token_ids), d=d)
        outs = self._process_claim(claim, [E_wt.var])
        self.claims.append(claim)
        return WitnessTensor(outs[x_var] if outs else None, x_var, (SEQ, d), self)

    def rope(self, x, *, SEQ: int, d_h: int, s_x: int,
             heads: int = 1,
             base: float = 10000.0, position_offset: int = 0,
             s_out: Optional[int] = None, output_width: int = 16):
        """Apply RoPE (Llama split-half) to x at scale s_x. With no s_out,
        returns x_rot at scale s_x² (raw). With s_out specified (typically
        s_out = s_x), internally rescales by log2(s_x² / s_out) bits and
        returns x_rot at scale s_out.

        Multi-head: when heads > 1, x is interpreted as (SEQ, H, d_h) flat;
        the same (cos, sin) table is applied to every head. heads=1 is the
        original single-head behavior."""
        L = x.var.length
        d_total = heads * d_h
        assert L == SEQ * d_total, (
            f"rope: expected length {SEQ*d_total} (SEQ*H*d_h), got {L}")
        assert d_h % 2 == 0, f"rope: d_h must be even, got {d_h}"
        sc = RoPEConfig(SEQ=SEQ, d_h=d_h, s_x=s_x, base=base,
                         position_offset=position_offset, heads=heads)

        if s_out is None:
            x_rot_var = self._alloc(f"{x.var.name}_rope", L)
            claim = RoPEClaim(x=x.var, x_rot=x_rot_var, config=sc)
            outs = self._process_claim(claim, [x.var])
            self.claims.append(claim)
            return WitnessTensor(outs[x_rot_var] if outs else None, x_rot_var, x.shape, self)

        ratio = (s_x * s_x) // s_out
        assert s_x * s_x == s_out * ratio and ratio > 0 and (ratio & (ratio - 1)) == 0, (
            f"rope rescale: s_x² ({s_x*s_x}) must be a power-of-2 multiple of s_out ({s_out})")
        rescale_bits = ratio.bit_length() - 1
        range_rescale = self._range_table("rescale", rescale_bits)
        range_output  = self._range_table("output",  output_width)

        x_rot_full_var = self._alloc(f"{x.var.name}_rope_full", L)
        x_rot_var      = self._alloc(f"{x.var.name}_rope",      L)
        x_rot_low_var, x_rot_shifted_var, z_low, z_shifted = self._emit_rescale_aux(
            f"{x.var.name}_rope", L, range_rescale, range_output)

        claim = RoPEClaim(
            x=x.var, x_rot=x_rot_var, config=sc,
            rescale_bits=rescale_bits, output_width=output_width,
            x_rot_full=x_rot_full_var,
            x_rot_low=x_rot_low_var,
            x_rot_shifted=x_rot_shifted_var,
            z_x_rot_low=z_low, z_x_rot_shifted=z_shifted,
            range_rescale=range_rescale, range_output=range_output,
        )
        def side_effects(values):
            lookup_multiplicities_into(values[x_rot_low_var], range_rescale.T,
                                        self.inputs[range_rescale.mult_var])
            lookup_multiplicities_into(values[x_rot_shifted_var], range_output.T,
                                        self.inputs[range_output.mult_var], label=x_rot_shifted_var.name)
        outs = self._process_claim(claim, [x.var], side_effects)
        self.claims.append(claim)
        return WitnessTensor(outs[x_rot_var] if outs else None, x_rot_var, x.shape, self)

    def prove(self, seed, *, verbose=False, weight_commitment=None, wnew_seed=None):
        """Streaming prover — the sound four-round protocol (the single path).
        Requires a lazy tape (streaming replays the tape's deferred ops).

        `weight_commitment` (a core.WeightCommitment): reference a pre-committed
        W tree instead of rebuilding it (persistent-weights P3).
        `wnew_seed` (bytes): required for linking proofs (persistent="new" vars)
        — the refresh seed the new commitment was made under (P5)."""
        if not self.lazy:
            raise RuntimeError(
                "tape.prove requires Tape(cfg, lazy=True): the streaming prover "
                "replays the tape's deferred ops.")
        from core import prove_streaming
        return prove_streaming(self, self.cfg, seed,
                               weight_commitment=weight_commitment,
                               wnew_seed=wnew_seed)

    def run_engine_pass(self, free_intermediates: bool = False, keep=None):
        """Process self._deferred (recorded by tape.X in lazy mode): for
        each claim, compute via COMPUTE_FNS and run its side effects.
        Returns a `live` dict {Variable: tensor} that mirrors what
        self.inputs would contain in eager mode.

        Bootstraps live with self.inputs (which in lazy mode only holds
        externally-committed values — tape.commit / tape.commit_lazy
        entries). Lazy callables are resolved on demand.

        Eager-mode tapes can also call this; for them self._deferred is
        empty and live just copies self.inputs.

        free_intermediates=True (forward-only / accuracy checks): drop each
        witness as soon as its last consumer claim (and that claim's side
        effects) have run, keeping only `keep` (e.g. the logits Variable).
        Peak memory falls from O(all layers) to O(one layer) — long contexts
        fit. MUST NOT be used when the witnesses are needed afterwards (a
        proof commits/opens every row); the proof path calls with no args."""
        keep = set(keep or ())
        last_use, consumed = {}, set()
        if free_intermediates:
            for i, (_claim, input_vars, _se) in enumerate(self._deferred):
                for v in input_vars:
                    last_use[v] = i
            consumed = set(last_use)                       # any Variable a claim reads
        live: dict = {}
        for v, val in self.inputs.items():
            live[v] = val
        fold = _compute_fns.FoldRunner(self._deferred)
        def fetch(v):
            val = live[v]
            return val() if callable(val) else val
        for i, (claim, input_vars, side_effects) in enumerate(self._deferred):
            if self.time_ops:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
            if fold.is_fold(claim):
                input_data = {}
                outs = fold.finalize(claim, live)
            else:
                input_data = {v: fetch(v) for v in input_vars}
                outs = _compute_fns.COMPUTE_FNS[type(claim)](claim, input_data)
            for v, t in outs.items():
                live[v] = t
            if side_effects is not None:
                side_effects({**input_data, **outs})
            for v in list(outs):
                fold.offer(v, live, allow_free=free_intermediates)
            if self.time_ops:
                torch.cuda.synchronize()
                self._log_op_time(claim, outs, input_vars,
                                   time.perf_counter() - t0)
            if free_intermediates:
                # Dead inputs (this is their last consumer) and aux outputs
                # no later claim reads (rescale/lookup aux — needed only by the
                # side effect just run, or by a proof we aren't doing).
                for v in input_vars:
                    if last_use.get(v) == i and v not in keep:
                        live.pop(v, None)
                for v in outs:
                    if v not in consumed and v not in keep:
                        live.pop(v, None)
                del input_data, outs
        # Mirror live → self.inputs so WitnessTensor.data resolves correctly
        # after prove() returns (caller may read e.g. logits.data for argmax
        # or save). Eager mode already writes computed outputs into
        # self.inputs inside _process_claim; this brings the lazy path in
        # line with that contract.
        self.inputs.update(live)
        return live

