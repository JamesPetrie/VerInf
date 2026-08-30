"""
Ligero framework: encoding, commits, the three test computations
(IRS, linear, quadratic), prove/verify orchestration. Generic over
per-claim registries SAMPLE_FNS / COMPILE_FNS / AUX_FNS keyed
by claim_type.

All field-element data lives on device as torch.uint64 tensors via the
cuda_primitives wrapper. Python only orchestrates: building per-row
Level-2 packet lists, the Verifier RNG, and the Merkle path bookkeeping
for the small T_QUERIES path data.

Ligero parameters (ELL, K_DEG, N_LIG, T_QUERIES) flow through a
LigeroConfig passed in — toy tests use small values for fast iteration;
production uses the design-feasibility §3.1 values.

Polynomial-interpolation convention: r_i, pa, pb are degree-<K_DEG
polynomials obtained by zero-padding their ELL-slot values to K_DEG and
running iNTT_K. This differs from a degree-<ELL Lagrange interpolation
but agrees with it at ζ_0..ζ_{ELL−1}, which is all the protocol uses
them at. Switching from Lagrange to iNTT_K turns an O(ELL²) per-row
verifier loop into a single batched iNTT — critical for production
verifier perf — at the cost of bumping q_lin/p_0 from K_DEG+ELL−1 to
2·K_DEG−1 coefficients (~1.3× larger).
"""
from __future__ import annotations

import bisect
import random
import secrets
import time
import warnings
from dataclasses import dataclass, field, fields
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Type, Union

import numpy as np
import os
import torch
import blake3 as _blake3

from cuda_primitives import (
    P, GLOBAL_G,
    gl_mul, gl_add, gl_sub, gl_neg, gl_inv_batched,
    ntt_forward, ntt_inverse, ntt_forward_batched, ntt_inverse_batched,
    rs_encode_rows,
    poly_mul, poly_add, poly_mul_batched, poly_eval,
    gl_matmul, gl_matvec, gl_spmv, challenge_vec, challenge_at,
    interp_band, interp_band_causal_id, interp_band_causal_c2,
    interp_band_embed, interp_band_rope_x,
    hash_columns_streamed, merkle_build_blake3,
    MerkleColumnAccumulator, row_prg, row_prg_indexed,
)

# The packet classes core itself constructs (table settlement bands); all other
# kinds are constructed by their claims and reach core only as opaque templates
# for the lowering.
from packets import L2_IdentityScalar, L2_PerSlotVector, L2_StrideManyToOneScalar
import protocol as pr   # shared challenge PRF / domains / column sampler (no torch)


# ============================================================
# Data types
# ============================================================

@dataclass(eq=False)
class Variable:
    """A named contiguous block of witness slots.

    `phase` is the commitment epoch a variable belongs to:
      1 — committed in R1, before any challenge exists;
      2 — committed in R2, may depend on the R1 coin s_op;
      3 — committed in R3, may depend on the R2 coin s_bind (the late
          Freivalds auxiliaries of the routed-projected matmul live here).
    `row_start` is assigned by the framework when prove/verify is called —
    variables are laid out phase 1 first, then 2, then 3, in declaration
    order, with each variable taking n_rows(ELL) = ceil(length/ELL) rows.

    eq=False so identity-based equality/hash hold: two Variables are
    distinct unless they're the same Python object, which lets us use
    Variable instances as dict keys in the witness map.
    """
    name: str
    length: int
    phase: int = 1
    row_start: int = -1
    persistent: bool = False   # model weight → the persistent W block (its own
                               # Merkle root R_W, committed once across proofs;
                               # see analysis/persistent-weights.md)
    w_new: bool = False        # linking proofs (P5): persistent var belongs to
                               # the SECOND weight block "wnew" (the refreshed
                               # commitment's tree) instead of "w"

    def __post_init__(self):
        # Variables derived by chaining tape ops (e.g. residual x + proj across
        # many layers) build names like "{a.name}+{b.name}" — at multi-layer
        # depth the names double per layer and reach hundreds of MB each,
        # consuming all system memory while cuda_a stays flat. Truncate to a
        # bounded suffix so Python string memory stays O(n_vars).
        if len(self.name) > 80:
            self.name = "..." + self.name[-77:]

    def n_rows(self, ell: int) -> int:
        return (self.length + ell - 1) // ell


@dataclass
class QuadraticConstraint:
    """For i ∈ [0, n):  x_row[i] · y_row[i] + a · z_row[i] = b, where the a/b
    coefficients are UNIFORM across the constraint's n slots.

    x_row/y_row/z_row are absolute row indices in the joint encoded matrix.
    n ≤ ELL — a single quadratic constraint fits in one row's ELL message
    slots. Multi-row quadratic constraints must be split by the caller.

    a_values/b_values therefore hold a SINGLE element (the uniform a/b); every
    consumer reads [0]. They were formerly [a]*n / [b]*n — at 32-layer SEQ=1000
    that duplicated tens of GB of identical ints across the SEQ²-sized softmax
    LogUp quads (the compile's dominant memory cost). Kept as 1-element lists
    (not bare scalars) so the [0]-indexing consumers stay untouched.
    """
    name: str
    x_row: int
    y_row: int
    z_row: int
    n: int
    a_values: List[int]      # length 1; the uniform per-slot a-coefficient
    b_values: List[int]      # length 1; the uniform per-slot b-coefficient


@dataclass
class QuadFamily:
    """One quadratic constraint FAMILY (the quad lift,
    linear-fold-unification.md): row t in [0, nrows) is the per-row constraint
    x_row+t . y_row+t + a . z_row+t = b over n = min(ell, L - t*ell) slots; the
    positional r_quad index of row t is (consumer-assigned base) + t. Replaces
    per-row QuadraticConstraint lists in COMPILE_FNS returns: O(#emit calls)
    stored instead of O(rows); consumers expand() transiently per claim.
    expand() reproduces the retired per-row loop exactly (rows, n, a, b and
    the flat index order), so the r_quad challenge pairing is unchanged."""
    name: str
    x_row: int      # row_start of each operand; row t adds +t
    y_row: int
    z_row: int
    L: int          # total slots; the last row may be partial
    ell: int
    a: int          # uniform per-slot coefficients
    b: int

    def nrows(self) -> int:
        return (self.L + self.ell - 1) // self.ell

    def expand(self) -> List[QuadraticConstraint]:
        nr = self.nrows()
        return [QuadraticConstraint(
            name=f"{self.name}[{t}]" if nr > 1 else self.name,
            x_row=self.x_row + t, y_row=self.y_row + t, z_row=self.z_row + t,
            n=min(self.ell, self.L - t * self.ell),
            a_values=[self.a], b_values=[self.b]) for t in range(nr)]


def abs_slot(var: Variable, flat_idx: int, ell: int) -> int:
    """(variable, flat_idx) → absolute slot in the joint witness."""
    return (var.row_start + flat_idx // ell) * ell + flat_idx % ell


@dataclass
class Table:
    """A LogUp table with shared multiplicity, used by RangeWordClaim and
    PairedTlookupClaim. The Tape's `register_table` allocates the mult_var
    (phase-1, zero-init, incremented on-the-fly by lookups via
    lookup_multiplicities_into) and the w_var (phase-2,
    w[j]=mult[j]/(α-v[j]), computed at aux-witness time).

    If T_Y is set, this is a *paired* table — the per-table-entry v[j] is
    T[j] + β·T_Y[j] (with β a separate per-table challenge), and
    PairedTlookupClaim proves (x[i], y[i]) ∈ (T, T_Y). If T_Y is None, this
    is a *range* table: v[j] = T[j], no β.

    `alpha` and (optionally) `beta` are sampled by TableSettlement.sample_fn
    and read by every using-claim's compile_fn. `z_vars` collects every
    per-claim z so the settlement can emit the cross-claim sum identity."""
    name: str
    T: torch.Tensor                       # (T_LEN,) uint64 on device — public
    mult_var: Variable                    # phase=1, length=T_LEN
    w_var: Variable                       # phase=2, length=T_LEN
    T_Y: Optional[torch.Tensor] = None    # (T_LEN,) for paired tables, else None
    alpha: int = 0
    beta: int = 0                          # used only when T_Y is set
    z_vars: List[Variable] = field(default_factory=list)


# ============================================================
# Ligero parameters
# ============================================================

@dataclass
class LigeroConfig:
    ELL: int
    K_DEG: int
    N_LIG: int
    T_QUERIES: int

    def __post_init__(self):
        assert self.ELL <= self.K_DEG <= self.N_LIG
        assert self.K_DEG & (self.K_DEG - 1) == 0
        assert self.N_LIG & (self.N_LIG - 1) == 0
        assert (P - 1) % self.K_DEG == 0
        assert (P - 1) % self.N_LIG == 0
        # Linear / quadratic blinding rows are length 2·K_DEG (Ligero §4.7 /
        # ligero-prover include/zkp/nonbatch_context.hpp::mask_callback); we
        # need 2·K_DEG ≤ N_LIG so their codewords fit in the encoded matrix.
        assert 2 * self.K_DEG <= self.N_LIG, (
            f"2 * K_DEG = {2 * self.K_DEG} must be ≤ N_LIG = {self.N_LIG} "
            f"(needed for 2K-sized blinding rows for q_lin / p_0)")

    @property
    def W_K(self) -> int:
        return pow(GLOBAL_G, (P - 1) // self.K_DEG, P)

    @property
    def W_N(self) -> int:
        return pow(GLOBAL_G, (P - 1) // self.N_LIG, P)

    @property
    def coset_shift(self) -> int:
        """Coset shift γ for the evaluation domain (Ligero §4.4 / standard FRI
        coset-LDE convention).

        We need γ ∉ ⟨ω_N⟩ so that η_j = γ · ω_N^j is disjoint from ζ_c = ω_K^c
        (which lies in ⟨ω_K⟩ ⊂ ⟨ω_N⟩ since K | N). GLOBAL_G = 7 is a primitive
        root of F_P with order P − 1, which doesn't divide N, so γ = GLOBAL_G
        is outside ⟨ω_N⟩.

        Matches the STARK/FRI ecosystem convention (ethSTARK §3, Plonky2's
        Field::coset_shift() at field/src/types.rs:441, Plonky3's coset_lde_batch,
        Anatomy of a STARK §3 — all use the multiplicative group generator as
        the FRI coset shift)."""
        return GLOBAL_G

    def zeta(self, c: int) -> int:
        """Interpolation/message points: K-th roots of unity, c ∈ [0, ELL).
        Lies in ⟨ω_K⟩ ⊂ ⟨ω_N⟩."""
        return pow(self.W_K, c, P)

    def eta(self, j: int) -> int:
        """Evaluation/codeword points: η_j = γ · ω_N^j ∈ γ · ⟨ω_N⟩, disjoint
        from ζ_c by construction (γ ∉ ⟨ω_N⟩)."""
        return (self.coset_shift * pow(self.W_N, j, P)) % P

    def zeta_points(self) -> List[int]:
        return [self.zeta(c) for c in range(self.ELL)]


# ============================================================
# Challenges — every challenge is protocol.challenge(seed, index, label).
#
# There is no Verifier/stream object any more. Each protocol round has its own
# seed (the verifier's fresh coins, delivered AFTER that round's commitment;
# in test mode derived from one base via protocol.round_seeds). From a seed,
# both prover and verifier expand the exact values they need BY INDEX:
#   op challenges (matmul ρ,λ; rmsnorm ρ; table α,β)  → protocol.op_vec /
#       challenge(s_op, idx, "op{ci}:…"), keyed by settled-list claim index;
#   test combiners r_irs/r_lin/r_quad                 → _combiner_vec(s_comb,…);
#   opened columns Q                                  → protocol.random_columns.
# No challenge values are ever sent between the parties — they share only seeds.
# ============================================================

def _combiner_vec(seed, label: str, n: int) -> np.ndarray:
    """Materialize a test-combiner vector r_label[0:n] from a round seed via the
    shared indexable PRF: r[i] = protocol.challenge(seed, i, label). Values are
    identical to the streaming verifier's on-demand challenge(seed, i, label);
    core materializes only because its GPU identity checks consume dense tensors.
    Hoists the per-call glue (challenge()/_seed_bytes/label.encode) out of the
    loop so only the blake3 hash + 128-bit reduce runs per index. Bit-identical
    to challenge(seed, i, label) = blake3(seed_bytes||label||i_le8)[:16] % P."""
    prefix    = pr._seed_bytes(seed) + label.encode()
    digest    = _blake3.blake3
    frombytes = int.from_bytes
    out = np.empty(n, dtype=np.uint64)
    for i in range(n):
        out[i] = frombytes(digest(prefix + i.to_bytes(8, "little")).digest()[:16],
                           "little") % P
    return out


# ============================================================
# Merkle helpers — BLAKE3 throughout (matches design-feasibility.md §3.1
# and the GPU hash_columns_streamed leaf hash).
# ============================================================

def _b3(left: bytes, right: bytes) -> bytes:
    return _blake3.blake3(left + right).digest()


def _levels_to_bytes(levels_dev: List[torch.Tensor]) -> List[List[bytes]]:
    """levels_dev[i] is a (k_i, 32) uint8 device tensor. Pull them all to
    host as List[List[bytes]] (small data — at N_LIG=65536 the full tree
    is ~4 MB)."""
    out = []
    for lvl in levels_dev:
        arr = lvl.cpu().numpy()
        out.append([bytes(arr[r].tolist()) for r in range(arr.shape[0])])
    return out


def merkle_path(levels: List[List[bytes]], idx: int) -> List[Tuple[bytes, int]]:
    """Extract opening path for leaf `idx`. Each element is (sibling, side)
    where side ∈ {0, 1} indicates whether `idx` was the right child (0)
    or left child (1) at that level."""
    path: List[Tuple[bytes, int]] = []
    for level in levels[:-1]:
        sibling_idx = idx ^ 1
        if sibling_idx >= len(level):
            sibling_idx = idx
        side = 1 if (idx & 1) == 0 else 0
        path.append((level[sibling_idx], side))
        idx //= 2
    return path


def merkle_verify(leaf: bytes, path: List[Tuple[bytes, int]],
                  claimed_root: bytes) -> bool:
    h = leaf
    for sibling, side in path:
        h = _b3(sibling, h) if side == 0 else _b3(h, sibling)
    return h == claimed_root


@dataclass
class CommitArtifact:
    root: bytes
    levels: List[List[bytes]]            # host-side bytes for path extraction
    column_hashes: List[bytes]           # also host-side; len = N_LIG
    matrix: torch.Tensor                 # (m_rows, N_LIG) uint64 on device


# Sentinel root used when a commit has zero rows (e.g., a claim list with
# no phase-2 variables — AddClaim, HadamardClaim). The verifier checks
# this sentinel and skips Merkle verification for that commit.
EMPTY_COMMIT_ROOT = b"\x00" * 32


# ============================================================
# Witness layout: build the message-domain (m_total, ELL) tensor.
# ============================================================

InputVal = Union[List[int], torch.Tensor]


def _to_device_u64(val: InputVal) -> torch.Tensor:
    """Accept either a Python list or a tensor; produce a 1-D uint64 CUDA tensor."""
    if isinstance(val, torch.Tensor):
        return val.to(device="cuda", dtype=torch.uint64).contiguous().reshape(-1)
    return torch.tensor(val, dtype=torch.uint64, device="cuda")


def _fetch(inputs: Dict[Variable, InputVal], v: Variable):
    """Resolve inputs[v] — calls it if it's a callable (lazy loader),
    else returns the stored value as-is. Lets the streaming primitives
    accept either eagerly-committed tensors or LazyHFLoader callables
    without per-call dispatch logic."""
    val = inputs[v]
    return val() if callable(val) else val


class _LazyResolvingDict:
    """Dict view that resolves callables to tensors on first access and
    caches the result per-instance. Used to feed AUX_FNS (which may read
    weight tensors via witness[c.A] / witness[c.B]) without modifying
    every AUX_FN to call _fetch explicitly. Cache lifetime = one AUX call,
    so weights load once per claim, not per access within the claim."""

    def __init__(self, base: Dict[Variable, InputVal]):
        self._base = base
        self._cache: Dict[Variable, InputVal] = {}

    def __getitem__(self, k: Variable):
        if k in self._cache:
            return self._cache[k]
        v = self._base[k]
        if callable(v):
            v = v()
        self._cache[k] = v
        return v

    def __contains__(self, k):
        return k in self._base


def _iter_message_chunks(vars_list: List[Variable],
                          inputs: Dict[Variable, InputVal],
                          cfg: LigeroConfig,
                          row_offset_start: int,
                          chunk_size: int):
    """Yield (abs_row_offset, chunk_tensor) pairs for the message matrix
    (vars_list packed row by row into ELL-wide rows from row_offset_start).

    Replaces the full (m_total, ELL) matrix allocation with chunk-by-chunk
    iteration. chunk_tensor is freshly allocated (cloned) per yield, so
    consumers may freely hold or drop without affecting later chunks.

    Each variable is padded/reshaped into its own (v_rows, ELL) tensor in
    one vectorized op (pad tail with zeros, then .view) instead of a
    Python loop over rows -- v_rows can be in the hundreds of thousands
    for a single lookup-table variable, and a per-row torch.Tensor
    __setitem__ pays ~10-20us of dispatch overhead regardless of how
    little data it moves (see analysis/ in the "toy transformer" formula-
    vs-reality investigation: this loop was ~69% of prove() wall-clock on
    a small witness, dwarfing the actual NTT/hash kernels it feeds). A
    `carry` buffer holds any rows left over after slicing off whole
    chunk_size pieces, so chunk boundaries are identical to the old
    per-row version even when a variable's row count doesn't align to
    chunk_size, or a chunk is filled by the tail of one variable plus the
    head of the next -- this does NOT materialize the full (m_total, ELL)
    matrix; peak extra memory is one variable's own rows plus one chunk.

    Last chunk may have fewer rows. abs_row_offset is row_offset_start +
    cumulative-rows-yielded-so-far."""
    if not vars_list:
        return
    ELL = cfg.ELL
    carry = None                          # (k, ELL) leftover rows, k < chunk_size, or None
    abs_offset = row_offset_start
    padded = None
    for v in vars_list:
        data = _to_device_u64(_fetch(inputs, v)).reshape(-1)
        v_len = data.numel()
        assert v_len == v.length, (
            f"variable '{v.name}' length {v.length} != input numel {v_len}")
        v_rows = v.n_rows(ELL)
        pad_len = v_rows * ELL - v_len
        if pad_len:
            padded = torch.zeros(v_rows * ELL, dtype=torch.uint64, device="cuda")
            padded[:v_len] = data
            rows_v = padded.view(v_rows, ELL)
        else:
            rows_v = data.view(v_rows, ELL)
        if carry is not None:
            rows_v = torch.cat([carry, rows_v], dim=0)
            carry = None
        n = rows_v.shape[0]
        pos = 0
        while n - pos >= chunk_size:
            yield abs_offset, rows_v[pos:pos + chunk_size].clone()
            abs_offset += chunk_size
            pos += chunk_size
        if pos < n:
            # .clone(), not a view: a view keeps the WHOLE source variable
            # alive until the next chunk boundary, so the previous expert
            # shard would still be resident while the next one loads (~336 MB
            # each at Maverick shapes, and the routed claim walks 128 of
            # them). The copy is at most one partial chunk.
            carry = rows_v[pos:].clone()
        # Drop this variable's tensors BEFORE the next iteration resolves the
        # next one: plain rebinding evaluates the new loader while the old
        # tensor is still bound, which means two shards resident at the peak.
        data = rows_v = padded = None
    if carry is not None and carry.shape[0] > 0:
        yield abs_offset, carry.clone()


# ============================================================
# Reed-Solomon encoding + commit phase.
# ============================================================

# ============================================================
# Zero-knowledge blinding (Ligero §4.6 + §4.7).
#
# Three blinding rows live at the start of R_p1, at fixed row indices
# 0, 1, 2. They are not modeled as claim-level Variables — they're
# protocol-level scaffolding handled directly in prove/verify.
#
#   row 0: u_irs   — degree-<K polynomial; mixes into q_irs (degree < K).
#   row 1: u_lin   — degree-<2K polynomial; mixes into q_lin (degree < 2K).
#   row 2: u_quad  — degree-<2K polynomial; mixes into p_0   (degree < 2K).
#
# Matches the ligero-prover ZKIPCP implementation
# (include/zkp/nonbatch_context.hpp::mask_callback): the IRS blinding is
# length K, but the linear and quadratic blindings are length 2K (encoded
# via iNTT_2K) so they fully cover the 2K-coefficient test polynomials
# without relying on witness-slack entropy for masking high-degree bits.
#
# The structural constraints on u_lin (Σ_c γ_c = 0 at ζ-points c ∈ [0, ELL))
# and u_quad (= 0 at ζ-points c ∈ [0, ELL)) preserve the verifier's
# test-condition checks:
#   - Σ_c q_lin(ζ_c) = expected: u_lin contributes 0   ✓
#   - p_0(ζ_c) = 0 for c ∈ [ELL): u_quad contributes 0 ✓
#
# Codeword evaluation at the coset η_j = γ · ω_N^j (see coset_shift) uses
# a twist-by-γ^l on the polynomial coefficients before applying the standard
# NTT_N. For the 2K rows this is a twist by γ^l for l ∈ [0, 2K).
# ============================================================

NUM_BLINDING_ROWS = 3
_BLIND_ROW_IRS  = 0   # u_irs  (length-K polynomial)
_BLIND_ROW_LIN  = 1   # u_lin  (length-2K polynomial)
_BLIND_ROW_QUAD = 2   # u_quad (length-2K polynomial)

# The prover's fixed seed for the ZK-padding PRG (row_prg) and the blinding
# messages. Constant across proofs ON PURPOSE: the W block's padding must
# reproduce bit-for-bit for R_W to be context-independent
# (analysis/persistent-weights.md). A REFRESHED weight commitment is made
# under a different seed; prove_streaming then pads the W block under the
# commitment's own seed (P5).
MASTER_SEED = b"\x42" * 32


def _make_blinding_messages(cfg: 'LigeroConfig',
                             master_seed: bytes
                             ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate the three blinding-row messages, returning a tuple
    (u_irs_msg, u_lin_msg, u_quad_msg).

    u_irs_msg:  (ELL,)        — random; encoded via the standard length-K path.
    u_lin_msg:  (2 · K_DEG,)  — encoded via iNTT_2K. Structure:
        - Even-indexed positions [0, 2·ELL) (i.e., values at ζ_c for c ∈ [0, ELL))
          are random γ_c with Σ γ_c = 0.
        - Even-indexed positions [2·ELL, 2·K_DEG) (values at K-th roots beyond
          the message slots) are random (slack at ζ-positions).
        - Odd-indexed positions [0, 2·K_DEG) (values at non-K-th 2K-th roots)
          are uniform random (provides high-degree masking entropy for q_lin).
    u_quad_msg: (2 · K_DEG,)  — encoded via iNTT_2K. Structure:
        - Even-indexed positions [0, 2·ELL) are 0 (u_quad encodes 0^ℓ).
        - All other positions are random.

    Asserts the message-level invariants; a mismatch would manifest as a
    completeness failure (verifier rejects honest proofs).
    """
    P_np = np.uint64(P)
    # Derive a per-proof numpy RNG from master_seed via blake3, so that
    # _make_blinding_messages is a pure function of (cfg, master_seed)
    # — same property as encode_messages with row_prg.
    bm_key = _blake3.blake3(master_seed + b"blinding_messages").digest(length=16)
    rng = np.random.default_rng(np.frombuffer(bm_key, dtype=np.uint64))

    # u_irs: length ELL random.
    u_irs_msg = (rng.integers(0, 2**64, size=cfg.ELL, dtype=np.uint64)
                 % P_np)

    # u_lin: length 2K random, then constrain Σ_c γ_c = 0 at ζ-positions.
    u_lin_msg = (rng.integers(0, 2**64, size=2 * cfg.K_DEG, dtype=np.uint64)
                 % P_np)
    # ζ_c maps to even-indexed positions [0, 2c, 4c, ...] of the 2K vector
    # (since ζ_c = ω_K^c = ω_{2K}^{2c}, the 2K-th root at index 2c).
    zeta_idx = np.arange(0, 2 * cfg.ELL, 2, dtype=np.int64)   # [0, 2, 4, ..., 2*ELL-2]
    # Sum γ_c at these positions in Python int (numpy .sum() wraps mod 2^64),
    # then adjust the last γ to make Σ = 0 mod P.
    head_sum = sum(int(u_lin_msg[i]) for i in zeta_idx[:-1]) % P
    u_lin_msg[zeta_idx[-1]] = (P - head_sum) % P

    # u_quad: length 2K random, then zero out ζ-positions for c ∈ [0, ELL).
    u_quad_msg = (rng.integers(0, 2**64, size=2 * cfg.K_DEG, dtype=np.uint64)
                  % P_np)
    u_quad_msg[zeta_idx] = 0

    # Structural assertions.
    lin_zeta_sum = sum(int(u_lin_msg[i]) for i in zeta_idx) % P
    assert lin_zeta_sum == 0, \
        f"_make_blinding_messages: u_lin Σ at ζ-positions = {lin_zeta_sum}, expected 0 mod P"
    assert bool((u_quad_msg[zeta_idx] == 0).all()), \
        "_make_blinding_messages: u_quad values at ζ-positions must be 0 for c ∈ [0, ELL)"

    return (torch.from_numpy(u_irs_msg.copy()).to("cuda"),
            torch.from_numpy(u_lin_msg.copy()).to("cuda"),
            torch.from_numpy(u_quad_msg.copy()).to("cuda"))


def _encode_2k_blinding_rows(msgs_2k: torch.Tensor,
                              cfg: 'LigeroConfig'
                              ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode (m_blind, 2·K_DEG) length-2K blinding messages.

    Returns:
      polys: (m_blind, 2·K_DEG) — coefficient form of degree-<2K polynomial
             (iNTT_2K of the input values at 2K-th roots).
      codewords: (m_blind, N_LIG) — codeword evaluated at coset η_j = γ · ω_N^j.

    Polynomial values at 2K-th roots equal `msgs_2k`. After coset twist by
    γ^l and forward NTT_N, the codeword positions are at γ · ω_N^j.
    """
    m = msgs_2k.size(0)
    if m == 0:
        return (torch.empty((0, 2 * cfg.K_DEG), dtype=torch.uint64, device="cuda"),
                torch.empty((0, cfg.N_LIG),     dtype=torch.uint64, device="cuda"))

    polys = msgs_2k.clone()
    ntt_inverse_batched(polys)                    # in-place iNTT_2K → (m, 2K) coeffs

    g_powers = _coset_powers_2k(cfg)              # (2K,) γ^l for l ∈ [0, 2K)
    g_powers_2d = g_powers.unsqueeze(0).expand(m, 2 * cfg.K_DEG).contiguous()
    twisted = gl_mul(polys, g_powers_2d)          # (m, 2K)

    extended = torch.zeros((m, cfg.N_LIG), dtype=torch.uint64, device="cuda")
    extended[:, :2 * cfg.K_DEG] = twisted
    ntt_forward_batched(extended)                 # in-place NTT_N → (m, N) codeword
    return polys, extended


def _mix_blinding_into_tests(q_irs: torch.Tensor,
                              q_lin: torch.Tensor,
                              p_0:   torch.Tensor,
                              u_irs_poly:  torch.Tensor,   # (K_DEG,)
                              u_lin_poly:  torch.Tensor,   # (2·K_DEG,)
                              u_quad_poly: torch.Tensor,   # (2·K_DEG,)
                              cfg: 'LigeroConfig'
                              ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mix the three blinding row polynomials into the test polynomials
    with coefficient 1 (Ligero §4.7). Mirrored in verify() at the
    column-identity reconstruction.

    Input shapes:
      q_irs (K_DEG,)         — IRS test polynomial; degree < K.
      q_lin (2·K_DEG - 1,)   — linear test poly from witness, degree < 2K - 1.
      p_0   (2·K_DEG - 1,)   — quadratic test poly from witness, degree < 2K - 1.

    Output shapes:
      q_irs (K_DEG,)         — q_irs + u_irs_poly.
      q_lin (2·K_DEG,)       — q_lin (padded) + u_lin_poly. Degree < 2K.
      p_0   (2·K_DEG,)       — p_0   (padded) + u_quad_poly. Degree < 2K.
    """
    q_irs = gl_add(q_irs, u_irs_poly)

    # Pad q_lin / p_0 from length 2K - 1 to length 2K and add the 2K
    # blinding polynomials.
    K2 = 2 * cfg.K_DEG
    q_lin_full = torch.zeros(K2, dtype=torch.uint64, device="cuda")
    q_lin_full[:K2 - 1] = q_lin
    q_lin_full = gl_add(q_lin_full, u_lin_poly)

    p_0_full = torch.zeros(K2, dtype=torch.uint64, device="cuda")
    p_0_full[:K2 - 1] = p_0
    p_0_full = gl_add(p_0_full, u_quad_poly)

    return q_irs, q_lin_full, p_0_full


def _master_seed_to_cuda(master_seed: bytes) -> torch.Tensor:
    """Pack a 32-byte master_seed into a CUDA uint8 tensor for the row_prg
    kernel. Called once per prove(); the resulting tensor is passed to
    every encode_messages call."""
    assert isinstance(master_seed, (bytes, bytearray)) and len(master_seed) == 32
    return torch.frombuffer(bytes(master_seed), dtype=torch.uint8).cuda()


# ============================================================
# Coset shift powers (for coset-LDE codeword evaluation).
#
# To evaluate a polynomial of degree < K at η_j = γ · ω_N^j (the coset
# η-domain), we twist its K coefficients by γ^l (l ∈ [0, K)) and then
# apply standard NTT_N. Since γ^l is the same for every row of the
# encoded matrix, we precompute the length-K twist vector once per
# (K_DEG, γ) and reuse.
# ============================================================

_COSET_POWERS_K_CACHE: Dict[Tuple[int, int], torch.Tensor] = {}


def _coset_powers_k(cfg: 'LigeroConfig') -> torch.Tensor:
    """Cached (K_DEG,) uint64 tensor of γ^l for l ∈ [0, K_DEG). One-time
    O(K_DEG) Python multiplications per (K_DEG, γ) pair, cached on device."""
    return _coset_powers(cfg.K_DEG, cfg.coset_shift)


def _coset_powers_2k(cfg: 'LigeroConfig') -> torch.Tensor:
    """Cached (2·K_DEG,) uint64 tensor of γ^l for l ∈ [0, 2·K_DEG). Used
    for the coset twist of length-2K blinding polynomials before NTT_N."""
    return _coset_powers(2 * cfg.K_DEG, cfg.coset_shift)


def _coset_powers(n: int, gamma: int) -> torch.Tensor:
    """Cached length-n uint64 tensor of γ^l for l ∈ [0, n)."""
    key = (n, gamma)
    cached = _COSET_POWERS_K_CACHE.get(key)
    if cached is not None:
        return cached
    powers_np = np.empty(n, dtype=np.uint64)
    curr = 1
    for l in range(n):
        powers_np[l] = curr
        curr = (curr * gamma) % P
    tensor = torch.from_numpy(powers_np).to("cuda")
    _COSET_POWERS_K_CACHE[key] = tensor
    return tensor


# --- coset-NTT encode mode (autoresearch: encode is ~32% of the 400B prove) ---
# The notebook's coset_ntt=True: evaluate at the N=rho*K coset by doing rho
# separate length-K NTTs instead of one length-N NTT. Byte-identical to the
# single-N path (validated bit-exact at K=64..16384 in
# analysis/bench/coset_ntt_bench.py + a byte-identical PROOF + Rust ACCEPT in
# validate_coset_ntt.py). The isolated NTT primitive is ~1.2x faster at
# K >= 2^16 (coset_ntt_ab.py), BUT an in-situ prove A/B at K=2^16 on a small
# model (ab_coset_insitu.py) showed NO encode-time win and +1.4GB peak memory
# — at that scale encode isn't NTT-bound, so the microbenchmark overstated the
# in-prove benefit. Its production benefit (huge witness at K>=2^18) is a
# cost-model PREDICTION, not a measured fact. So this is OPT-IN ONLY (default
# off): a validated-correct lever to A/B at true production scale, never
# auto-enabled (it costs memory for unconfirmed benefit). LIGERO_COSET_NTT=on.
_COSET_NTT_MODE = os.environ.get("LIGERO_COSET_NTT", "off")   # off | on | auto
_COSET_NTT_MIN_K = 1 << 16


def _coset_encode_via_coset_ntt(coeffs: torch.Tensor, cfg: 'LigeroConfig') -> torch.Tensor:
    """rho length-K coset NTTs, interleaved -> (m, N_LIG) codeword. Peak memory
    = out (m*N) + one length-K transform (m*K), same as the single-N path (no
    3x expand); the rho=N/K cosets run as a small Python loop. Bit-exact
    equivalent of the single-N encode below."""
    m, K, N = coeffs.size(0), cfg.K_DEG, cfg.N_LIG
    rho = N // K
    out = torch.empty((m, N), dtype=torch.uint64, device="cuda")
    for t in range(rho):
        gamma_t = (cfg.coset_shift * pow(cfg.W_N, t, P)) % P
        gp = _coset_powers(K, gamma_t)                             # (K,), cached
        twisted = gl_mul(coeffs, gp.unsqueeze(0).expand(m, K).contiguous())
        ntt_forward_batched(twisted)                               # in-place length-K
        # codeword[row, i*rho + t] = coset_t_result[row, i]; strided write via the
        # int64 view (pure data movement -> bit-identical, dodges the uint64 gap).
        out.view(torch.int64)[:, t::rho] = twisted.view(torch.int64)
    return out


def _coset_encode_codewords(coeffs: torch.Tensor,
                              cfg: 'LigeroConfig') -> torch.Tensor:
    """Evaluate (m, K_DEG) polynomials (coefficient form, degree < K_DEG)
    at the coset η_j = γ · ω_N^j for j ∈ [0, N_LIG). Returns (m, N_LIG)
    codeword.

    Equivalent to standard NTT_N of γ^l-twisted coefficients, zero-extended
    from K_DEG to N_LIG."""
    m = coeffs.size(0)
    if m == 0:
        return torch.empty((0, cfg.N_LIG), dtype=torch.uint64, device="cuda")
    use_coset = (_COSET_NTT_MODE == "on"
                 or (_COSET_NTT_MODE == "auto" and cfg.K_DEG >= _COSET_NTT_MIN_K))
    if use_coset and cfg.N_LIG > cfg.K_DEG:
        return _coset_encode_via_coset_ntt(coeffs, cfg)
    g_powers = _coset_powers_k(cfg)                                # (K_DEG,)
    # gl_mul requires same shapes; broadcast g_powers to (m, K_DEG).
    g_powers_2d = g_powers.unsqueeze(0).expand(m, cfg.K_DEG).contiguous()
    twisted = gl_mul(coeffs, g_powers_2d)                          # (m, K_DEG)
    extended = torch.zeros((m, cfg.N_LIG), dtype=torch.uint64, device="cuda")
    extended[:, :cfg.K_DEG] = twisted
    ntt_forward_batched(extended)
    return extended


def encode_messages(messages: torch.Tensor, cfg: LigeroConfig,
                    *,
                    master_seed: torch.Tensor,
                    row_offset: int = 0,
                    need_codewords: bool = True,
                    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """messages: (m, ELL). Returns (row_polys: (m, K_DEG), codewords: (m, N_LIG)).

    `need_codewords=False` returns (row_polys, None) and skips the N_LIG-point
    coset LDE. The q-poly round consumes only row_polys — it has no Merkle tree
    and opens no column — so computing the codewords there is work whose result
    nobody reads: 402.7G weight slots' worth of it at Maverick scale. Skipping
    output that is never consumed cannot change the proof.

    The K_DEG - ELL slack slots are filled by `row_prg(master_seed,
    row_offset + i, slack)` for row i. Deterministic: same (master_seed,
    row_offset, messages) always produces the same output, regardless of
    caller. Replaces the stateful blind_rng / numpy.PCG64 stream that
    required snapshot/restore for the 4-pass chunked prover.

    Required for the proof to be zero-knowledge: a column-query open
    reveals p_i(η_j), a linear function of the polynomial's K_DEG
    coefficients; randomizing K_DEG - ELL of them means each opened
    value is statistically uniform conditioned on the message.

    Soundness is unaffected: the IRS test enforces that each row is a
    valid codeword of degree < K_DEG, which random pad satisfies.

    Statistical ZK (against column queries) requires T_QUERIES < slack.
    The blinding-row construction for the test polynomials (q_lin,
    q_quad, q_irs) is a separate fix not yet implemented.
    """
    m = messages.size(0)
    if m == 0:
        return (torch.empty((0, cfg.K_DEG), dtype=torch.uint64, device="cuda"),
                torch.empty((0, cfg.N_LIG), dtype=torch.uint64, device="cuda"))

    padded = torch.empty((m, cfg.K_DEG), dtype=torch.uint64, device="cuda")
    padded[:, :cfg.ELL] = messages

    slack = cfg.K_DEG - cfg.ELL
    if slack > 0:
        if slack <= cfg.T_QUERIES:
            warnings.warn(
                f"ZK pad too small for column-query confidentiality: "
                f"slack=K_DEG-ELL={slack}, T_QUERIES={cfg.T_QUERIES}. "
                f"Proof is sound but not zero-knowledge under this config.",
                UserWarning, stacklevel=2)
        padded[:, cfg.ELL:] = row_prg(master_seed, row_offset, m, slack)
    # else: K_DEG == ELL, no slack to fill (toy/soundness-only configs).

    row_polys = padded.clone()
    ntt_inverse_batched(row_polys)
    if not need_codewords:
        return row_polys, None
    codewords = _coset_encode_codewords(row_polys, cfg)
    return row_polys, codewords


def _pack_column_for_hash(column_vals: List[int]) -> bytes:
    return b"".join(int(v).to_bytes(8, "little") for v in column_vals)


# BLAKE3 chunk = 1024 bytes = 128 u64 rows. The chunked commit / open
# helpers pack input rows into multiples of this for the streaming
# kernel (see MerkleColumnAccumulator).
_BLAKE3_CHUNK_ROWS = 128
_ENCODE_CHUNK_ROWS = 1024   # default: 8 BLAKE3 chunks per encode call


class _SmallMerkleAcc:
    """One-shot column-hash for total_rows ≤ 128 (single BLAKE3 chunk).
    Same interface as MerkleColumnAccumulator so _stream_phase can use
    either without branching."""

    def __init__(self, n_cols: int, n_total_rows: int):
        self.n_cols = n_cols
        self.n_total_rows = n_total_rows
        self._chunks: List[torch.Tensor] = []

    def update(self, rows: torch.Tensor) -> None:
        self._chunks.append(rows)

    def finalize(self) -> torch.Tensor:
        if not self._chunks:
            return torch.empty((self.n_cols, 32), dtype=torch.uint8, device="cuda")
        all_codes = torch.cat(self._chunks) if len(self._chunks) > 1 else self._chunks[0]
        return hash_columns_streamed(all_codes)


def _make_merkle_acc(n_cols: int, n_total_rows: int):
    """Pick the appropriate merkle column accumulator for the input size."""
    if n_total_rows > _BLAKE3_CHUNK_ROWS:
        return MerkleColumnAccumulator(n_cols, n_total_rows)
    return _SmallMerkleAcc(n_cols, n_total_rows)


def _finalize_merkle_artifact(merkle_acc) -> CommitArtifact:
    """Finalize column digests → BLAKE3 merkle tree → CommitArtifact."""
    if merkle_acc.n_total_rows == 0:
        return CommitArtifact(root=EMPTY_COMMIT_ROOT, levels=[[]],
                              column_hashes=[], matrix=None)
    digests = merkle_acc.finalize()
    root_dev, levels_dev = merkle_build_blake3(digests)
    levels_host = _levels_to_bytes([digests] + levels_dev[1:])
    root = bytes(root_dev.cpu().numpy().tolist())
    return CommitArtifact(root=root, levels=levels_host,
                          column_hashes=levels_host[0], matrix=None)


# ============================================================
# Persistent weight block (analysis/persistent-weights.md, P1∪P3 core).
# ============================================================

def collect_weight_vars(tape) -> List[Variable]:
    """The persistent (model-weight) variables of a tape, in stable
    declaration order — the W block. Deduped by identity, order fixed by
    first appearance in tape.inputs, which is independent of the per-proof
    activation/aux vars, so the block is context-independent."""
    seen, out = set(), []
    for v in tape.inputs:
        if (isinstance(v, Variable) and v.persistent and not v.w_new
                and id(v) not in seen):
            seen.add(id(v))
            out.append(v)
    return out


def commit_weights(tape, cfg: LigeroConfig, master_seed: bytes = MASTER_SEED):
    """Commit the persistent W block by its OWN sweep and return
    (artifact, weight_vars, m_w_rows).

    Separate from the per-proof op-order sweep on purpose: weights are
    committed at their op position, interleaved with activations, so the
    W block cannot be a row-range reshuffle of that sweep without breaking
    row-order==op-order (see analysis/persistent-weights.md, the 2026-07-07
    finding). Weights don't depend on activations, so a dedicated sweep
    emits the W rows in weight-row order into their own Merkle tree.

    Row layout: weight vars occupy rows [NUM_BLINDING_ROWS, NUM_BLINDING_ROWS
    + m_w) — the W block leads the witness right after the blinding tree
    (layout B, analysis/persistent-weights.md), so this standalone R_W
    matches prove_streaming's in-proof W tree bit-for-bit. The W tree has no
    per-proof blinding of its own. Deterministic given the model +
    master_seed, since the ZK padding is seeded by absolute row index — so
    the same weights reproduce R_W byte-for-byte across proofs of different
    prompts (the P1 gate). Returns m_w = the weight row COUNT (not the end
    row).

    Uses the SAME `_layout` as prove_streaming — walking the tape's CLAIMS —
    so the weight set and their row order are identical to the in-proof W
    tree. (A weight committed but never referenced by a claim is not in the
    proof's witness, so it is excluded here too.)"""
    claims = _with_synthesized_settlements(tape.claims)
    (_all, _p1, _p2, _p3, _m_p1, _m_p2, _m_total,
     weight_vars, m_w, _wn, _m_wn) = _layout(claims, cfg)
    master_seed_t = _master_seed_to_cuda(master_seed)
    acc = _make_merkle_acc(cfg.N_LIG, m_w) if m_w else None
    if acc is not None:
        _stream_phase(weight_vars, tape.inputs, cfg,
                      master_seed=master_seed_t,
                      abs_row_offset=NUM_BLINDING_ROWS, merkle_acc=acc)
    art = _finalize_merkle_artifact(acc) if acc is not None else _finalize_merkle_artifact(
        _make_merkle_acc(cfg.N_LIG, 0))
    return art, weight_vars, m_w


@dataclass
class WeightCommitment:
    """A persisted commitment to a model's W block (analysis/persistent-weights.md,
    P3): the root R_W, the Merkle tree levels (for opening-path generation), and
    the weight row count. Committed once by `commit_weights`, then referenced by
    many proofs — `prove_streaming(..., weight_commitment=wc)` uses this root and
    these levels instead of rebuilding the W tree, so the per-proof cost drops by
    the weight column-hashing (the compute-bound D·W term, paper §8). The encoded
    weight matrix is NOT stored (it is re-encoded on demand from the tape's
    weights at open time); levels are a few MB at N_LIG=65536.

    `master_seed` is the ZK-padding seed the commitment was made under. The
    default MASTER_SEED is the prover's fixed seed; a REFRESHED commitment
    (P5) uses a fresh seed, and prove_streaming pads the W block under the
    commitment's own seed so the re-extracted columns reproduce these
    leaves."""
    root: bytes
    levels: List[List[bytes]]
    m_w: int
    n_lig: int
    master_seed: bytes = MASTER_SEED
    # Opening ledger. Every proof that references this commitment opens
    # T_QUERIES columns of the SAME weight rows under the SAME padding, so the
    # leakage that matters is the CUMULATIVE set of distinct columns, not the
    # per-proof count. Each row's polynomial carries K_DEG-ELL random pad
    # coefficients; once the union of opened columns approaches that, the
    # padding stops hiding and the enrollment must be REFRESHED (the P5 path:
    # re-commit under a fresh seed and prove the link).
    opened_columns: List[int] = field(default_factory=list)

    @staticmethod
    def from_tape(tape, cfg: LigeroConfig, master_seed: bytes = None
                  ) -> "WeightCommitment":
        # A fresh secret enrollment seed by default: the padding of the
        # enrolled model is what hides it in the opened columns, so a public
        # constant would give the weights away. Callers that need a
        # reproducible R_W (diff-tests) pass one explicitly.
        if master_seed is None:
            master_seed = new_zk_seed()
        art, _wv, m_w = commit_weights(tape, cfg, master_seed)
        return WeightCommitment(root=art.root, levels=art.levels, m_w=m_w,
                                n_lig=cfg.N_LIG, master_seed=bytes(master_seed))

    # Wire format: a plain length-framed binary blob, NOT pickle. The handle
    # carries the enrollment seed (a secret) and the Merkle levels, and it is
    # read before anything about it has been checked — pickle.load would
    # execute whatever the file says. Layout:
    #   magic "VERINFWC" | u32 version | u32 m_w | u32 n_lig | 32B root
    #   | 32B master_seed | u32 n_levels | per level: u32 n_nodes, nodes*32B
    MAGIC = b"VERINFWC"
    VERSION = 2
    # Fraction of the ZK pad we are willing to spend before demanding a
    # refresh. Half leaves the same margin again for the refresh proof itself.
    OPENING_BUDGET_FRACTION = 0.5

    def save(self, path: str) -> None:
        import os
        import struct
        # The ledger is security state, not a cache.  Publish it atomically so
        # a crash cannot roll the enrollment back to an earlier opening set.
        part = path + ".part"
        with open(part, "wb") as f:
            f.write(self.MAGIC)
            f.write(struct.pack("<III", self.VERSION, self.m_w, self.n_lig))
            f.write(self.root)
            f.write(self.master_seed)
            f.write(struct.pack("<I", len(self.levels)))
            for lvl in self.levels:
                f.write(struct.pack("<I", len(lvl)))
                for node in lvl:
                    assert len(node) == 32, "merkle node must be 32 bytes"
                    f.write(node)
            cols = sorted(set(self.opened_columns))
            f.write(struct.pack("<I", len(cols)))
            for c in cols:
                f.write(struct.pack("<I", c))
            f.flush()
            os.fsync(f.fileno())
        os.replace(part, path)
        dfd = os.open(os.path.dirname(os.path.abspath(path)) or ".", os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)

    @staticmethod
    def load(path: str) -> "WeightCommitment":
        try:
            return WeightCommitment._parse(path)
        except AssertionError:
            raise
        except Exception as e:                      # truncated / scrambled file
            raise AssertionError(
                f"corrupt weight commitment {path}: {type(e).__name__}: {e}")

    @staticmethod
    def _parse(path: str) -> "WeightCommitment":
        import struct
        with open(path, "rb") as f:
            blob = f.read()
        assert blob[:8] == WeightCommitment.MAGIC, "not a weight commitment file"
        ver, m_w, n_lig = struct.unpack("<III", blob[8:20])
        assert ver == WeightCommitment.VERSION, f"unsupported version {ver}"
        root, seed = blob[20:52], blob[52:84]
        (n_levels,) = struct.unpack("<I", blob[84:88])
        off = 88
        levels: List[List[bytes]] = []
        for _ in range(n_levels):
            (n_nodes,) = struct.unpack("<I", blob[off:off + 4]); off += 4
            end = off + 32 * n_nodes
            levels.append([blob[i:i + 32] for i in range(off, end, 32)])
            off = end
        (n_cols,) = struct.unpack("<I", blob[off:off + 4]); off += 4
        cols = list(struct.unpack(f"<{n_cols}I", blob[off:off + 4 * n_cols]))
        off += 4 * n_cols
        assert off == len(blob), "trailing bytes in weight commitment file"
        wc = WeightCommitment(root=root, levels=levels, m_w=m_w, n_lig=n_lig,
                              master_seed=seed, opened_columns=cols)
        wc.check_topology()
        return wc

    def opening_budget(self, cfg: 'LigeroConfig') -> int:
        """How many DISTINCT columns this enrollment may ever reveal."""
        slack = cfg.K_DEG - cfg.ELL
        return int(slack * self.OPENING_BUDGET_FRACTION)

    def check_openings(self, cfg: 'LigeroConfig', new_columns) -> None:
        """Refuse a proof that would spend more of the pad than the ledger
        allows. Called once the column challenge exists and BEFORE any weight
        column is produced, so an exhausted enrollment costs nothing.

        A config with no usable slack (K_DEG-ELL <= T_QUERIES) is not zero
        knowledge to begin with — encode_messages already warns about it — so
        there is no pad to ration and the ledger stands down. Production runs
        at K_DEG=2*ELL, where the budget is real."""
        if cfg.K_DEG - cfg.ELL <= cfg.T_QUERIES:
            return
        after = len(set(self.opened_columns) | set(new_columns))
        budget = self.opening_budget(cfg)
        if after > budget:
            raise AssertionError(
                f"weight-commitment opening budget exhausted: this proof would "
                f"take the enrollment to {after} distinct opened columns, past "
                f"the {budget} its ZK padding covers "
                f"(K_DEG-ELL={cfg.K_DEG - cfg.ELL}). REFRESH the enrollment "
                f"under a fresh seed and prove the link (persistent-weights P5) "
                f"before proving again.")

    def record_openings(self, new_columns) -> None:
        """Book the columns a completed proof revealed. The caller persists the
        handle afterwards — an unsaved ledger is a silently reusable pad."""
        self.opened_columns = sorted(set(self.opened_columns) | set(new_columns))

    def check_topology(self) -> None:
        """Recompute the tree from its leaves: a handle is trusted for the
        root it hands to the verifier, so a file whose levels do not build
        that root must not be usable at all."""
        assert self.levels, "empty commitment"
        assert len(self.levels[0]) == self.n_lig, (
            f"leaf count {len(self.levels[0])} != N_LIG {self.n_lig}")
        cur = list(self.levels[0])
        depth = 1
        while len(cur) > 1:
            nxt = [_b3(cur[i], cur[i + 1]) for i in range(0, len(cur), 2)]
            assert depth < len(self.levels) and self.levels[depth] == nxt, (
                f"merkle level {depth} does not follow from level {depth - 1}")
            cur = nxt
            depth += 1
        assert depth == len(self.levels), "extra levels in commitment"
        assert cur[0] == self.root, "levels do not build the stored root"



def _opened_columns_match_leaves(opened: Dict[int, torch.Tensor],
                                 Q_cols: List[int],
                                 art: CommitArtifact,
                                 n_total_rows: int) -> bool:
    """Prover self-check: re-hash the T extracted columns and compare each
    digest against the commit-round leaf. Detects witness-regeneration drift
    between the commit sweep and the round-4 column extraction (the PRG-seeded
    ZK padding must reproduce bit for bit) without rebuilding the tree. Feeds
    the same accumulator the commit used, in row blocks, so no second full
    column matrix is ever resident."""
    if n_total_rows == 0 or not Q_cols:
        return True
    acc = _make_merkle_acc(len(Q_cols), n_total_rows)
    step = 1 << 20
    for lo in range(0, n_total_rows, step):
        hi = min(lo + step, n_total_rows)
        # ColumnSink serves the openings from pre-sized HOST buffers; the
        # merkle accumulator hashes on the GPU, so move each row block over.
        acc.update(torch.stack([opened[j][lo:hi] for j in Q_cols], dim=1)
                   .to("cuda", non_blocking=True))
    digests = acc.finalize().cpu().numpy()
    return all(bytes(digests[k].tolist()) == art.column_hashes[j]
               for k, j in enumerate(Q_cols))


# Optional prove-time phase breakdown (env LIGERO_PHASE_TIMING=1). cuda-synced
# wall-clock buckets accumulated across the sweep, printed at prove end. Pure
# diagnostic: a no-op when off (zero overhead, no behavior change). The syncs
# serialize the GPU pipeline, so the bucket SUM slightly overstates wall-clock —
# read the SHARES, not the absolute total.
from contextlib import contextmanager as _contextmanager
_PHASE_ON = bool(os.environ.get("LIGERO_PHASE_TIMING"))
_PHASE_TIMES: Dict[str, float] = {}


@_contextmanager
def _phase(name):
    if not _PHASE_ON:
        yield
        return
    torch.cuda.synchronize()
    _t0 = time.time()
    try:
        yield
    finally:
        torch.cuda.synchronize()
        _PHASE_TIMES[name] = _PHASE_TIMES.get(name, 0.0) + (time.time() - _t0)


# Precise GPU-kernel timing (env LIGERO_EPHASE=1): async CUDA events summed with a
# SINGLE sync at report time -- no per-call host-sync inflation. Compare to the _phase
# wall-clock for the same scope: wall >> gpu means host-bound (Python loop / .item()
# stalls), wall ~= gpu means GPU-bound.
_EPHASE_ON = bool(os.environ.get("LIGERO_EPHASE"))
_EPHASE: Dict[str, list] = {}


@_contextmanager
def _ephase(name):
    if not _EPHASE_ON:
        yield
        return
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    try:
        yield
    finally:
        e.record()
        _EPHASE.setdefault(name, []).append((s, e))


def _ephase_report():
    if not (_EPHASE_ON and _EPHASE):
        return
    torch.cuda.synchronize()
    tot = {n: sum(a.elapsed_time(b) for a, b in evs) for n, evs in _EPHASE.items()}
    print("  [ephase] GPU-kernel time per scope (cuda events, single sync):", flush=True)
    for n, ms in sorted(tot.items(), key=lambda kv: -kv[1]):
        print(f"    {n:14s} {ms / 1000:7.3f}s  ({len(_EPHASE[n])} calls)", flush=True)


class ColumnSink:
    """Opened columns, written straight to host memory as they are produced.

    The old path buffered every chunk's slice on the GPU and joined them with
    torch.cat at the end. At the target geometry the openings are tens of GB
    and the join briefly doubles that — the peak that decides whether a run
    fits in VRAM at all. Here each column is one pre-sized pinned host tensor
    (the block's row count is known before the sweep) and every chunk is
    copied into its own slice, so there is no accumulation and no join.
    """

    __slots__ = ("cols", "n_rows", "row_base", "_filled", "_ranges")

    def __init__(self, n_rows: int, columns: List[int], row_base: int = 0):
        self.n_rows = n_rows
        self.row_base = row_base          # absolute row of this block's row 0
        self.cols = {j: torch.empty(n_rows, dtype=torch.uint64, device="cpu")
                     for j in columns}
        self._filled = 0
        self._ranges: List[Tuple[int, int]] = []   # block-local [lo, hi) written

    def write(self, abs_row: int, chunk: torch.Tensor, columns: List[int]):
        """chunk: (rows, len(columns)) device slice of this chunk's codewords,
        starting at ABSOLUTE row `abs_row`."""
        lo = abs_row - self.row_base
        n = chunk.size(0)
        assert 0 <= lo and lo + n <= self.n_rows, (
            f"column sink overflow: [{lo}, {lo + n}) outside [0, {self.n_rows})")
        host = chunk.cpu()
        for k, j in enumerate(columns):
            self.cols[j][lo:lo + n] = host[:, k]
        self._filled += n
        self._ranges.append((lo, lo + n))

    def write_host(self, abs_row: int, cols: Dict[int, torch.Tensor]):
        """Scatter a finished piece of this block produced elsewhere (a
        weight-split worker's run): {column: (rows,) HOST tensor}, all the
        same length, starting at ABSOLUTE row `abs_row`. Every column of the
        sink must be supplied."""
        lo = abs_row - self.row_base
        n = None
        for j in self.cols:
            t = cols[j]
            if n is None:
                n = int(t.numel())
            assert int(t.numel()) == n, "ragged column piece"
        n = n or 0
        assert 0 <= lo and lo + n <= self.n_rows, (
            f"column sink overflow: [{lo}, {lo + n}) outside [0, {self.n_rows})")
        for j in self.cols:
            self.cols[j][lo:lo + n] = cols[j].to("cpu", torch.uint64)
        self._filled += n
        self._ranges.append((lo, lo + n))

    def finish(self) -> Dict[int, torch.Tensor]:
        assert self._filled == self.n_rows, (
            f"column sink filled {self._filled} of {self.n_rows} rows")
        # Exact coverage, not just the count: a duplicated piece and a
        # missing one would cancel in _filled but not here.
        pos = 0
        for lo, hi in sorted(self._ranges):
            assert lo == pos, (f"column sink coverage gap/overlap at block row "
                               f"{pos}: next piece starts at {lo}")
            pos = hi
        assert pos == self.n_rows, f"column sink coverage ends at {pos}, not {self.n_rows}"
        return self.cols


def _stream_phase(
    vars_list: List[Variable],
    inputs: Dict[Variable, InputVal],
    cfg: LigeroConfig,
    *,
    master_seed: torch.Tensor,
    abs_row_offset: int,
    pad_seed: Optional[torch.Tensor] = None,
    pad_row_offset: Optional[int] = None,
    prefix_codewords: Optional[torch.Tensor] = None,
    merkle_acc=None,
    q_irs_acc: Optional['QIrsAccumulator'] = None,
    q_lin_acc: Optional['QLinAccumulator'] = None,
    columns_at: Optional[List[int]] = None,
    column_sink: Optional['ColumnSink'] = None,
    chunk_size: int = _ENCODE_CHUNK_ROWS,
) -> Dict[str, Any]:
    """One streaming pass over (vars_list, inputs). For each chunk:
        encode → feed every accumulator that was provided.

    The unified building block for round 1 (merkle commit), round 3
    (q-poly accumulate), round 4 (merkle rebuild + column extract),
    and the fused test-mode prover (all accumulators active at once).

    pad_seed / pad_row_offset decouple the ZK-padding PRG from physical
    placement (analysis/persistent-weights.md, P5): row i of the pass pads
    from (pad_seed, pad_row_offset + i) while the merkle/q accumulators keep
    the physical abs_row_offset. Defaults (None) fall back to
    (master_seed, abs_row_offset) — padding keyed by physical row, the
    standard case. Because the column Merkle tree hashes codeword VALUES in
    emission order, a weight block padded under its commitment's
    (seed, logical offset) reproduces the committed root wherever it
    physically sits — the enabler for refresh + linking proofs.

    Accumulators are mutated in place; caller finalizes them (one
    accumulator may consume both p1 and p2 — e.g., q_irs_acc).

    Returns a dict containing whatever ONLY this helper can finalise:
      - 'opened_columns': Dict[int, Tensor] if columns_at given
    Caller-owned accumulators are not included; caller finalises them.
    """
    out: Dict[str, Any] = {}
    # No sink: nothing to compute. This is the referenced-weight case (a
    # persistent commitment supplies root and paths, so R1 has no merkle
    # accumulator for the W block) — encoding it anyway would cost a full
    # RS pass over every weight slot, 402.7G slots at Maverick scale, for
    # output nobody reads.
    if (merkle_acc is None and q_irs_acc is None and q_lin_acc is None
            and columns_at is None):
        return out

    col_buf: Optional[Dict[int, List[torch.Tensor]]] = None
    Q_set: Optional[torch.Tensor] = None
    if columns_at is not None:
        # With a sink the slices go straight to host; without one (unit tests,
        # the in-process verifier) the old device buffer is kept.
        col_buf = None if column_sink is not None else {j: [] for j in columns_at}
        Q_set = torch.tensor(list(columns_at), dtype=torch.long, device="cuda")

    # Optional prefix codewords (round 1: blinding rows for phase-1).
    if prefix_codewords is not None and prefix_codewords.size(0) > 0:
        if merkle_acc is not None:
            merkle_acc.update(prefix_codewords)
        if column_sink is not None:
            column_sink.write(abs_row_offset,
                              prefix_codewords.index_select(1, Q_set), columns_at)
        if col_buf is not None:
            slice_ = prefix_codewords.index_select(1, Q_set)
            for k, j in enumerate(columns_at):
                col_buf[j].append(slice_[:, k].clone())

    # Stream encode + feed every active accumulator.
    #
    # Only the codeword sinks need the LDE: the Merkle tree hashes codeword
    # values and an opening slices codeword columns. The q-poly round has
    # neither, and its rows are the whole enrolled weight block, so computing
    # the codewords there was the single largest piece of unread work in the
    # prover.
    seed_for_pad = master_seed if pad_seed is None else pad_seed
    need_codes = (merkle_acc is not None or column_sink is not None
                  or col_buf is not None)
    for chunk_abs_row, chunk_msg in _iter_message_chunks(
            vars_list, inputs, cfg, abs_row_offset, chunk_size):
        with _phase('encode'):
            polys, codes = encode_messages(
                chunk_msg, cfg, master_seed=seed_for_pad,
                row_offset=(chunk_abs_row if pad_row_offset is None
                            else pad_row_offset + (chunk_abs_row - abs_row_offset)),
                need_codewords=need_codes)
        if merkle_acc is not None:
            with _phase('merkle'):
                merkle_acc.update(codes)
        if q_irs_acc is not None:
            with _phase('fold_qirs'):
                q_irs_acc.update(chunk_abs_row, polys)
        if q_lin_acc is not None:
            with _phase('fold_qlin'):
                q_lin_acc.update(chunk_abs_row, polys)
        if column_sink is not None:
            with _phase('cols'):
                column_sink.write(chunk_abs_row,
                                  codes.index_select(1, Q_set), columns_at)
        elif col_buf is not None:
            with _phase('cols'):
                slice_ = codes.index_select(1, Q_set)
                for k, j in enumerate(columns_at):
                    col_buf[j].append(slice_[:, k].clone())

    if col_buf is not None:
        out['opened_columns'] = {
            j: (torch.cat(col_buf[j]) if col_buf[j]
                else torch.empty(0, dtype=torch.uint64, device="cuda"))
            for j in columns_at
        }
    return out








# ============================================================
# Polynomial helpers built on the wrapper.
# ============================================================

def _interpolate_to_kdeg(values_2d: torch.Tensor, cfg: LigeroConfig) -> torch.Tensor:
    """Zero-pad each row from ELL to K_DEG and run iNTT_K batched.

    Result row i is the coefficients (degree < K_DEG) of the unique
    polynomial that takes values[i, c] at ζ_c for c ∈ [0, ELL) and 0
    at ζ_c for c ∈ [ELL, K_DEG). At ELL=K_DEG (toy) this equals the
    Lagrange interpolation through (ζ_c, values[i, c])."""
    m = values_2d.size(0)
    padded = torch.zeros((m, cfg.K_DEG), dtype=torch.uint64, device="cuda")
    padded[:, :cfg.ELL] = values_2d
    ntt_inverse_batched(padded)
    return padded


# ============================================================
# Test-response polynomial computations.
# ============================================================

class QIrsAccumulator:
    """Chunk-driven q_irs accumulator: row_polys is streamed and never
    materialized as one tensor.

    q_irs[k] = Σᵢ r_irs[i] · row_polys[i, k]  (sum over witness rows)

    Caller supplies each chunk with its ABSOLUTE row offset (including the
    blinding-rows prefix), so chunks may arrive in any order — the streaming
    prover feeds an op's phase-1 then phase-2 rows, jumping the phase boundary.
    r_irs is witness-indexed (blinding excluded), so index = abs_row - NUM_BLINDING_ROWS."""

    def __init__(self, r_irs_witness: torch.Tensor, cfg: LigeroConfig):
        self.r_irs = r_irs_witness                  # (m_witness_total,)
        self.cfg   = cfg
        self.q     = torch.zeros(cfg.K_DEG, dtype=torch.uint64, device="cuda")

    def update(self, abs_row: int, witness_polys_chunk: torch.Tensor) -> None:
        """abs_row: absolute row of witness_polys_chunk[0] (blinding included).
        witness_polys_chunk: (n, K_DEG)."""
        n = witness_polys_chunk.size(0)
        if n == 0:
            return
        lo = abs_row - NUM_BLINDING_ROWS
        r_slice = self.r_irs[lo:lo + n]
        contrib = gl_matmul(r_slice.unsqueeze(0), witness_polys_chunk).squeeze(0)
        self.q = gl_add(self.q, contrib)

    def merge(self, q_partial: torch.Tensor) -> None:
        """Add another accumulator's partial sum (a weight-split worker's fold
        over its rows). Exact field addition, so the merged result equals
        one accumulator having seen every row, in any order."""
        assert q_partial.shape == self.q.shape
        self.q = gl_add(self.q, q_partial.to(self.q.device))

    def finalize(self) -> torch.Tensor:
        return self.q


class _PktRange(NamedTuple):
    """O(1) alternative to emitting one (row, pkt) tuple per row: a
    compile_fn that knows its packet is IDENTICAL across a whole
    row range (the common case -- see _index_bands's docstring: "All rows
    of a (variable, family) share an identical template; only the row
    range varies") can return one of these instead of row_end - row_start
    separate tuples. _index_bands folds it into a band in O(1); the eager
    (non-streaming) compile path expands it back to per-row tuples via
    _expand_row_pkts, since that path's per_row list genuinely needs one
    entry per row."""
    pkt: object
    row_start: int
    row_end: int   # exclusive


def _expand_row_pkts(row_pkts):
    """Normalize row_pkts entries to a flat (row, pkt) iterator, expanding
    any _PktRange entries. O(rows) -- for the eager compile path
    (_compile_with_chs), which needs a genuine per-row list; the streaming
    path (_StreamingPackets._index_bands) consumes _PktRange directly in
    O(1) instead and should not go through this."""
    for item in row_pkts:
        if isinstance(item, _PktRange):
            for r in range(item.row_start, item.row_end):
                yield r, item.pkt
        else:
            yield item


def _compile_one(claim, ch, cfg, base, late_ch=None):
    """COMPILE_FNS dispatch, passing the late challenge to claims that have a
    phase-3 stage. Every other claim keeps the original 4-argument call."""
    fn = COMPILE_FNS[type(claim)]
    if late_ch is not None and type(claim) in LATE_SAMPLE_FNS:
        return fn(claim, ch, cfg, base, late_ch=late_ch)
    return fn(claim, ch, cfg, base)


class _StreamingPackets:
    """Lazy per-claim compile for the streaming prover.

    Replaces the all-claims-upfront per_row list (~45 GB of packet objects
    at 48-layer Maverick scale — the binding memory constraint). Regular op
    claims compile when the sweep reaches them; packets sit in a row-keyed
    dict and are POPPED as q_lin consumes those rows.

    TableSettlement claims are the exception and are PRE-compiled here: their
    sum-side packets land on every LogUp-inverse row across the whole tape
    (those rows emit at their owning lookup ops, far before the settlements
    compile), so their packets must be in the dict from the start. They are
    few and small relative to the op packets. Their compile cursor (cid base
    after all ops) comes from the same count pass that sizes r_quad.

    Quads fire at their compiling op (operands are that claim's own fields,
    all live there) with global indices from the running base — p_0 is
    term-identical to the eager path. The fold consumes the band index via
    bands_overlapping (variable-major; no per-row lists)."""

    def __init__(self, claims, ch0, cfg, n_ops, chs_late=None):
        self.claims, self.ch0, self.cfg = claims, ch0, cfg
        self.n_ops = n_ops
        # Late (phase-3) challenges, for claims that own a second stage. The
        # compile runs ONCE, after s_bind, so a late claim's bands and its
        # early ones are emitted together and the constraint ids never depend
        # on when the compile happened.
        self.chs_late = chs_late if chs_late is not None else [None] * len(claims)
        # Build the per-(variable, family) band index UPFRONT from the count pass:
        # one template + row range per band (~MB), replacing the per-row packet
        # store (~45 GB). All bands are known before the sweep -> no lazy compile and
        # no late patch. Quads are still re-derived per op at the sweep and fired.
        self.bands: List[list] = []           # each: [template_pkt, row_start, row_end]
        self._band_seen: Dict[Any, int] = {}
        cur, nq = 0, 0
        self.base_c = []
        # Quad lift: store each claim's QuadFamily descriptors with their
        # positional index bases (O(#emit_quad) total, not O(rows)); the sweep
        # expands per claim -- compile_op no longer re-runs COMPILE_FNS.
        self.quad_fams: Dict[int, list] = {}
        for i in range(n_ops):
            self.base_c.append(cur)
            pk, q, na, _ = _compile_one(claims[i], ch0[i], cfg, cur,
                                        self.chs_late[i])
            self._index_bands(pk)
            cur += na
            fams = []
            for fam in q:
                fams.append((nq, fam)); nq += fam.nrows()
            self.quad_fams[i] = fams
        self.base_c.append(cur)
        self.ops_end_constraints, self.ops_end_quads = cur, nq
        # settlements: their packets land on rows across the tape -> indexed here too.
        for i in range(n_ops, len(claims)):
            pk, q, na, _ = _compile_one(claims[i], ch0[i], cfg, cur,
                                        self.chs_late[i])
            cur += na
            self._index_bands(pk)
            fams = []
            for fam in q:
                fams.append((nq, fam)); nq += fam.nrows()
            self.quad_fams[i] = fams
        self.total_constraints, self.total_quads = cur, nq
        self._build_row_lookup()
        self.next_op = 0

    def _index_bands(self, row_pkts):
        """Dedup a claim's per-row packets into band records. All rows of a
        (variable, family) share an identical template; only the row range
        varies -- a compile_fn that already knows its whole range shares one
        packet can hand it over as a _PktRange and skip the O(rows) walk
        entirely (folded in O(1) below); anything still emitting bare
        (row, pkt) tuples is handled exactly as before, one row at a time."""
        for item in row_pkts:
            if isinstance(item, _PktRange):
                pkt, rs, re = item.pkt, item.row_start, item.row_end
            else:
                r, pkt = item
                rs, re = r, r + 1
            key = _band_key(pkt)
            idx = self._band_seen.get(key)
            if idx is None:
                self._band_seen[key] = len(self.bands)
                self.bands.append([pkt, rs, re])
            else:
                b = self.bands[idx]
                if rs < b[1]: b[1] = rs
                if re > b[2]: b[2] = re

    def _build_row_lookup(self):
        """Group bands by their row range (disjoint across variables -- each
        variable owns a distinct range) and sort. The fold queries the ranges
        OVERLAPPING a chunk (bands_overlapping) instead of per-row template
        lists -- the walk restructure: the variable hands over its bands."""
        ranges: Dict[Tuple[int, int], List[Any]] = {}
        for tmpl, rs, re in self.bands:
            ranges.setdefault((rs, re), []).append(tmpl)
        self._range_keys = sorted(ranges)
        self._range_pkts = [ranges[k] for k in self._range_keys]
        self._range_ends = [k[1] for k in self._range_keys]

    def bands_overlapping(self, lo, hi):
        """[(templates, row_start, row_end)] for every (variable, family) band
        range intersecting [lo, hi), in row order. Ranges are disjoint (a row
        belongs to exactly one variable), so the overlap set is one contiguous
        slice of the sorted table, found by one bisect per chunk."""
        out = []
        i = bisect.bisect_right(self._range_ends, lo)   # first range ending past lo
        while i < len(self._range_keys) and self._range_keys[i][0] < hi:
            out.append((self._range_pkts[i],) + self._range_keys[i])
            i += 1
        return out

    def reset(self):
        """Sound-mode re-sweep: restart the op cursor. The band index is immutable
        and read-only, so there is nothing to rebuild or clear."""
        self.next_op = 0

    def compile_op(self, i):
        """Fire claim i's quads with their global positional indices, expanded
        transiently from the stored QuadFamily descriptors (quad lift -- no
        COMPILE_FNS re-run; every band already exists in the upfront index)."""
        if i < self.n_ops:
            assert i == self.next_op, f"stream compile out of order: {i} != {self.next_op}"
            self.next_op += 1
        return [(b0 + t, qc)
                for b0, fam in self.quad_fams.get(i, ())
                for t, qc in enumerate(fam.expand())]


class _EagerBands:
    """Adapter: the eager per-row packet list (tests/test_prover.py's compile
    path) behind the band-index interface — each row is its own singleton
    range, reproducing the retired per-row semantics exactly."""
    def __init__(self, per_row):
        self.per_row = per_row

    def bands_overlapping(self, lo, hi):
        return [(self.per_row[r], r, r + 1)
                for r in range(lo, min(hi, len(self.per_row))) if self.per_row[r]]


class QLinAccumulator:
    """Chunk-driven q_lin accumulator. Replaces the full-matrix
    q_lin fold when row_polys is being streamed.

    Wraps _compute_q_lin_inner_chunk: caller provides chunks of rows
    (with their ABSOLUTE row indices in the joint witness, including
    blinding offset); each chunk asks the band index for the overlapping
    (variable, family) bands and folds them into q_lin.

    Internally re-chunks caller's input into inner_chunk_size pieces to
    bound the prods intermediate (inner_chunk_size × 2K)."""

    def __init__(self, seed_u8: torch.Tensor, band_index,
                 cfg: LigeroConfig, inner_chunk_size: int = 256):
        self.seed_u8 = seed_u8                       # 32-byte s_comb (cuda uint8)
        self.label_u8 = torch.tensor(list(b"lin"), dtype=torch.uint8, device="cuda")
        if isinstance(band_index, list):             # eager per-row list (tests)
            band_index = _EagerBands(band_index)
        self.band_index = band_index                 # exposes bands_overlapping
        self.cfg = cfg
        # Inverse-NTT fuse (default ON; LIGERO_FUSE_POLYMUL=0 falls back):
        # accumulate the per-row products in the EVAL domain (Σ_i NTT(r_i)·
        # NTT(p_i)) and do ONE inverse NTT at finalize, instead of an inverse
        # NTT per row inside poly_mul. Valid since the inverse NTT is linear:
        # Σ_i INTT(prod_i) = INTT(Σ_i prod_i) — bit-exact with the unfused
        # path (measured −7.5% prove; analysis/prover-optimization-
        # investigation.md §4). LIGERO_FUSE_CHECK=1 runs BOTH paths and
        # asserts bit-equality at finalize (the gate).
        self.fuse = os.environ.get("LIGERO_FUSE_POLYMUL", "1") != "0"
        self.fuse_check = os.environ.get("LIGERO_FUSE_CHECK") == "1"
        n_eval = 1
        while n_eval < 2 * cfg.K_DEG - 1:
            n_eval <<= 1
        self.n_eval = n_eval
        self.q_eval = (torch.zeros(n_eval, dtype=torch.uint64, device="cuda")
                       if (self.fuse or self.fuse_check) else None)
        self.q_coeff = (torch.zeros(2 * cfg.K_DEG - 1, dtype=torch.uint64, device="cuda")
                        if (not self.fuse or self.fuse_check) else None)
        self.inner_chunk = inner_chunk_size
        # Phase 2 (linear-fold-unification.md): Freivalds challenge dedup+cache,
        # one instance per fold so the cache spans all chunks and layers.
        self.chal_src = ChalSource(self.seed_u8, self.label_u8)

    def update(self, abs_row_lo: int, polys_chunk: torch.Tensor) -> None:
        """abs_row_lo: absolute row index of polys_chunk[0] in joint witness
        (including the blinding-rows offset).
        polys_chunk: (n, K_DEG)."""
        n_outer = polys_chunk.size(0)
        for inner_lo in range(0, n_outer, self.inner_chunk):
            inner_hi = min(inner_lo + self.inner_chunk, n_outer)
            abs_lo = abs_row_lo + inner_lo
            abs_hi = abs_row_lo + inner_hi
            inner_polys = polys_chunk[inner_lo:inner_hi]
            if self.q_eval is not None:
                self.q_eval = gl_add(self.q_eval, _compute_q_lin_inner_chunk(
                    abs_lo, abs_hi, inner_polys,
                    self.seed_u8, self.label_u8, self.band_index, self.cfg,
                    chal_src=self.chal_src, return_eval=True))
            if self.q_coeff is not None:
                self.q_coeff = gl_add(self.q_coeff, _compute_q_lin_inner_chunk(
                    abs_lo, abs_hi, inner_polys,
                    self.seed_u8, self.label_u8, self.band_index, self.cfg,
                    chal_src=self.chal_src, return_eval=False))

    def partials(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """(q_eval, q_coeff) as accumulated so far, whichever this mode keeps
        — the un-finalised state a weight-split worker hands back."""
        return self.q_eval, self.q_coeff

    def merge(self, q_eval: Optional[torch.Tensor] = None,
              q_coeff: Optional[torch.Tensor] = None) -> None:
        """Add a worker's partials before finalize. Both representations are
        plain field sums over rows (the eval-domain one is summed BEFORE its
        single inverse NTT, which is linear), so the merged fold is
        bit-identical to one accumulator having seen every row. The worker
        must run in the same fuse mode: every representation this
        accumulator keeps must be supplied."""
        if self.q_eval is not None:
            assert q_eval is not None, "fused q_lin merge needs the worker's q_eval"
            assert q_eval.shape == self.q_eval.shape
            self.q_eval = gl_add(self.q_eval, q_eval.to(self.q_eval.device))
        if self.q_coeff is not None:
            assert q_coeff is not None, "coeff-domain q_lin merge needs the worker's q_coeff"
            assert q_coeff.shape == self.q_coeff.shape
            self.q_coeff = gl_add(self.q_coeff, q_coeff.to(self.q_coeff.device))

    def finalize(self) -> torch.Tensor:
        if self.q_eval is not None:
            q = self.q_eval.view(1, -1).contiguous()
            ntt_inverse_batched(q)               # ONE inverse for the whole fold
            fused = q[0, :2 * self.cfg.K_DEG - 1].contiguous()
            if self.fuse_check:
                assert bool(torch.equal(fused, self.q_coeff)), \
                    "[fuse-check] eval-domain fold != coeff-domain fold"
                print("[fuse-check] PASS: fused q_lin bit-identical to unfused",
                      flush=True)
            return fused if self.fuse else self.q_coeff
        return self.q_coeff



class ChalSource:
    """r_lin access for the q_lin fold (linear-fold-unification.md, Phase 2 —
    the prover side of the verifier's Chal). Dedups challenge hashing per
    DISTINCT cid for the strided-repeat Freivalds bands, whose cids (base +
    f%k) recur on every row: one batched hash over the band's small static
    span [base, base+k) (or +H for the C side), cached for the WHOLE prove
    (s_comb is fixed once per round-3 fold), then gathered per nonzero.
    Everything else keeps the direct challenge_at path — for those kinds each
    cid appears once per chunk, so a buffer saves nothing.

    Bit-exact by construction: challenge_at is a deterministic PRF, so hashing
    a cid once and gathering equals hashing it per nonzero (validated by the
    retired LIGERO_QLIN_BANDCHK oracle before its deletion).

    Spans are STATIC (from the band template) — no cid min/max device syncs.
    Cache size: one (span,)-u64 tensor per distinct Freivalds band ≈ 8·k B
    each; MBs over a full model."""

    _SPANS = {
        "L2_FreivaldsLF1B": lambda p: (p.base, p.base + p.k),   # cid = base + i_k
        "L2_FreivaldsLF2A": lambda p: (p.base, p.base + p.k),   # cid = base + i_k
        "L2_FreivaldsLF3C": lambda p: (p.base, p.base + p.H),   # cid = base + h
    }

    def __init__(self, seed_u8, label_u8):
        self.seed_u8, self.label_u8 = seed_u8, label_u8
        self.cache: Dict[Tuple[int, int], torch.Tensor] = {}
        self.interp_cache: Dict[int, Any] = {}   # Phase-4 launch packs, per fold

    def buffer_for(self, kind, template):
        """(lo, buffer) for a strided-repeat band's dense challenge span, or
        None for kinds with no span (each cid used ~once). Cached per fold."""
        span_fn = self._SPANS.get(kind.__name__)
        if span_fn is None:
            return None
        lo, hi = span_fn(template)
        buf = self.cache.get((lo, hi))
        if buf is None:
            span = torch.arange(lo, hi, dtype=torch.int64, device="cuda")
            buf = challenge_at(self.seed_u8, self.label_u8,
                               span.to(torch.uint64).contiguous())
            self.cache[(lo, hi)] = buf
        return lo, buf

_ROW_START_FIELD_NAME_CACHE: Dict[type, Optional[str]] = {}
_NO_ROW_START_FIELD = object()   # sentinel: "checked, this type has no *row_start field"


def _row_start_field_name(pkt_type: type) -> Optional[str]:
    """The field name ending in 'row_start' for a packet dataclass TYPE,
    memoized -- dataclasses.fields() is the same for every instance of a
    type, so re-deriving it per packet (this is called once per witness
    row during the one-time band-index pass -- hundreds of thousands of
    times on a real run) was pure repeated reflection overhead."""
    name = _ROW_START_FIELD_NAME_CACHE.get(pkt_type, _NO_ROW_START_FIELD)
    if name is not _NO_ROW_START_FIELD:
        return name
    name = None
    for f in fields(pkt_type):
        if f.name.endswith("row_start"):
            name = f.name
            break
    _ROW_START_FIELD_NAME_CACHE[pkt_type] = name
    return name


def _band_key(pkt):
    """A (variable, family) band identity from a packet's scalar fields. Any valid
    partition keeps the decomposition bit-exact (gl_add is linear); this keys on
    (kind, base, *_row_start) so it mirrors the real (variable, family) bands."""
    pkt_type = type(pkt)
    name = _row_start_field_name(pkt_type)
    rs = getattr(pkt, name) if name is not None else None
    return (pkt_type, getattr(pkt, "base", None), rs)


# --- Phase 4: descriptor-interpreter kernels (linear-fold-unification.md) ---
# Every band kind lowers to a CUDA launch pack: ten regular kinds to a 24-slot
# u64 descriptor interpreted by ONE kernel (k_interp_band — mixed-radix digit
# decode -> cid/coef by strided dots -> accumulate coef * r_lin[cid] over the
# fan axis, in place into the chunk rTA), and the four irregular kinds (causal
# x2, embed, rope-x) to small bespoke kernels on the same contract. The torch
# expander path that predated the kernels (and served as their bit-equality
# oracle, LIGERO_KERNEL_CHECK) was DELETED after gating the kernels — the
# standing correctness anchor is the independent Rust verifier; resurrect the
# oracle from git history for future kernel work.
_DUMMY_U64 = None
def _dummy_u64():
    global _DUMMY_U64
    if _DUMMY_U64 is None:
        _DUMMY_U64 = torch.zeros(1, dtype=torch.uint64, device="cuda")
    return _DUMMY_U64


def _ceil_div(a, b):
    return (a + b - 1) // b


def _lower_geometry(pkt):
    """(q, shape[4], cid_strides[4], fan, fan_stride, coef_mode, coef_const,
    A_strides[4], tblA, B_strides[4], tblB, L) for a kernel-eligible band
    template, or None for the irregular kinds. Shapes are msd-first."""
    k = type(pkt).__name__
    Z4 = [0, 0, 0, 0]
    if k == "L2_IdentityScalar":
        return (1, [pkt.L, 1, 1, 1], [1] + Z4[:3], 1, 0, 0, pkt.coef % P,
                Z4, None, Z4, None, pkt.L)
    if k == "L2_PerSlotVector":
        return (1, [pkt.L, 1, 1, 1], [1] + Z4[:3], 1, 0, 1, 0,
                [1, 0, 0, 0], pkt.coef_vec, Z4, None, pkt.L)
    if k == "L2_RowSumPerSlotVector":
        return (2, [_ceil_div(pkt.L, pkt.stride), pkt.stride, 1, 1],
                [1, 0, 0, 0], 1, 0, 1, 0,
                [0, 1, 0, 0], pkt.coef_vec, Z4, None, pkt.L)
    if k == "L2_StrideManyToOneScalar":
        return (2, [_ceil_div(pkt.L, pkt.stride), pkt.stride, 1, 1],
                [1, 0, 0, 0], 1, 0, 0, pkt.coef % P, Z4, None, Z4, None, pkt.L)
    if k == "L2_StrideOneToManyScalar":
        return (1, [pkt.L, 1, 1, 1], [pkt.stride, 0, 0, 0], pkt.stride, 1,
                0, pkt.coef % P, Z4, None, Z4, None, pkt.L)
    if k == "L2_TransposeO2MScalar":
        return (2, [pkt.rows, pkt.cols, 1, 1],
                [pkt.fan, pkt.rows * pkt.fan, 0, 0], pkt.fan, 1,
                0, pkt.coef % P, Z4, None, Z4, None, pkt.L)
    if k == "L2_FreivaldsLF1B":
        if pkt.transpose_b:
            n_j = _ceil_div(pkt.n * pkt.k, pkt.k)   # = n (source cols)
            return (3, [n_j, pkt.H, pkt.K, 1], [0, pkt.K, 1, 0], 1, 0, 1, 0,
                    [1, pkt.n, 0, 0], pkt.neg_rho, Z4, None, pkt.k * pkt.n)
        return (3, [pkt.K, pkt.H, pkt.n, 1], [1, pkt.K, 0, 0], 1, 0, 1, 0,
                [0, pkt.n, 1, 0], pkt.neg_rho, Z4, None, pkt.k * pkt.n)
    if k == "L2_FreivaldsLF2A":
        return (3, [pkt.m, pkt.H, pkt.K, 1], [0, pkt.K, 1, 0], 1, 0, 1, 0,
                [1, pkt.m, 0, 0], pkt.neg_lam, Z4, None, pkt.m * pkt.k)
    if k == "L2_FreivaldsLF3C":
        return (3, [pkt.m, pkt.H, pkt.n, 1], [0, 1, 0, 0], 1, 0, 2, 0,
                [1, pkt.m, 0, 0], pkt.lam, [0, pkt.n, 1, 0], pkt.rho, pkt.L)
    if k == "L2_RoPEXRot":
        half = pkt.d_h // 2
        return (4, [pkt.SEQ, pkt.H, 2, half],
                [2 * pkt.H * half, 2 * half, 1, 2], 1, 0, 0, 1,
                Z4, None, Z4, None, pkt.L)
    return None


def _var_row_start_of(pkt):
    for f in fields(pkt):
        if f.name.endswith("row_start"):
            return getattr(pkt, f.name)
    raise AssertionError(f"band template {type(pkt).__name__} has no row_start")


def _lower_band(pkt, chal_src, ell):
    """Launch pack {"L", "var_rs", "launch"} for one band template — every kind
    lowers: ten regular kinds to the 24-slot descriptor interpreted by
    k_interp_band; the four irregular kinds (causal ×2, embed, rope-x) to their
    bespoke kernels. Cached on chal_src (per fold — the Freivalds challenge
    buffers bind to s_comb)."""
    cache = chal_src.interp_cache
    key = id(pkt)
    if key in cache:
        return cache[key]
    kname = type(pkt).__name__
    if kname == "L2_CausalFilteredIdScalar":
        pack = {"L": pkt.L, "var_rs": pkt.var_row_start,
                "launch": lambda out, oo, fl, ns, sd, lb, p=pkt:
                    interp_band_causal_id(out, oo, fl, ns, p.base, p.M, p.H,
                                          p.coef % P, sd, lb)}
    elif kname == "L2_CausalFilteredC2Stride":
        pack = {"L": pkt.B, "var_rs": pkt.c2_row_start,
                "launch": lambda out, oo, fl, ns, sd, lb, p=pkt:
                    interp_band_causal_c2(out, oo, fl, ns, p.base, p.H,
                                          p.coef % P, sd, lb)}
    elif kname == "L2_EmbedE":
        # No stored length: the per-slot vocab-row guard in the kernel makes
        # window overshoot a no-op, so the clip bound is effectively infinite.
        rows_per_w = max(1, ell // pkt.d)
        pack = {"L": 1 << 62, "var_rs": pkt.E_row_start,
                "launch": lambda out, oo, fl, ns, sd, lb, p=pkt, rw=rows_per_w:
                    interp_band_embed(out, oo, fl, ns, p.base, p.d, rw, ell,
                                      p.token_ids, sd, lb)}
    elif kname == "L2_RoPEX":
        pack = {"L": pkt.L, "var_rs": pkt.x_row_start,
                "launch": lambda out, oo, fl, ns, sd, lb, p=pkt:
                    interp_band_rope_x(out, oo, fl, ns, p.base, p.H, p.d_h,
                                       p.cos_t, p.sin_t, sd, lb)}
    else:
        geo = _lower_geometry(pkt)
        assert geo is not None, f"no lowering for band kind {kname}"
        (q, shape, cs, fan, fs, mode, c0, a_str, tblA, b_str, tblB, L) = geo
        got = chal_src.buffer_for(type(pkt), pkt)
        chal_mode, chal_base, chal_buf = (1, got[0], got[1]) if got else (0, 0, _dummy_u64())
        desc = torch.tensor(
            [q] + shape + [pkt.base] + cs + [fan, fs, mode, c0] + a_str + b_str
            + [chal_mode, chal_base],
            dtype=torch.uint64, device="cuda")   # uint64 direct: coef consts exceed int64
        tA = tblA if tblA is not None else _dummy_u64()
        tB = tblB if tblB is not None else _dummy_u64()
        pack = {"L": L, "var_rs": _var_row_start_of(pkt),
                "launch": lambda out, oo, fl, ns, sd, lb,
                                 d=desc.contiguous(), a=tA, b=tB, cb=chal_buf:
                    interp_band(out, oo, fl, ns, d, a, b, cb, sd, lb)}
    cache[key] = pack
    return pack


def _compute_q_lin_inner_chunk(
    chunk_lo: int, chunk_hi: int,
    row_polys_chunk: torch.Tensor,
    seed_u8: torch.Tensor, label_u8: torch.Tensor,
    band_index,
    cfg: LigeroConfig,
    chal_src: 'ChalSource' = None,
    return_eval: bool = False,
) -> torch.Tensor:
    """Process one inner chunk for q_lin; returns the chunk's contribution
    (2K-1,) polynomial. VARIABLE-MAJOR (the walk restructure,
    linear-fold-unification.md item 4): the chunk asks the band index for the
    (variable, family) bands overlapping its rows and evaluates each band over
    the intersection window directly -- no per-row template lists, no per-chunk
    regroup, one bisect per chunk. Bit-exact vs the retired per-row grouping:
    identical expander inputs (one template per covered row) and a commutative
    sum (validated by the retired LIGERO_QLIN_BANDCHK oracle before its
    deletion; e2e gates: test_claims, prove->ACCEPT, the seq-1000 runs)."""
    n_chunk = chunk_hi - chunk_lo
    K, ELL = cfg.K_DEG, cfg.ELL
    n_targets = n_chunk * ELL

    overlaps = band_index.bands_overlapping(chunk_lo, chunk_hi)
    with _phase('qlin_rTA'), _ephase('qlin_rTA'):
        chunk_rTA_flat = torch.zeros(n_targets, dtype=torch.uint64, device="cuda")
        for tmpls, rs, re in overlaps:
            r_lo, r_hi = max(rs, chunk_lo), min(re, chunk_hi)
            for tmpl in tmpls:
                # Every kind lowers to a kernel launch pack. Window is
                # variable-relative; targets are the contiguous slots at out_off.
                pack = _lower_band(tmpl, chal_src, ELL)
                var_rs = pack["var_rs"]
                flat_lo_v = (r_lo - var_rs) * ELL
                flat_hi_v = min((r_hi - var_rs) * ELL, pack["L"])
                n_sl = flat_hi_v - flat_lo_v
                if n_sl <= 0:
                    continue
                out_off = (r_lo - chunk_lo) * ELL
                assert out_off + n_sl <= n_targets, "interp window exceeds chunk"
                pack["launch"](chunk_rTA_flat, out_off, flat_lo_v, n_sl,
                               seed_u8, label_u8)
    assert chunk_rTA_flat.numel() == n_chunk * ELL, (
        f"band fold returned {chunk_rTA_flat.numel()} != "
        f"n_chunk*ELL={n_chunk * ELL} — view would alias OOB")
    chunk_rTA  = chunk_rTA_flat.view(n_chunk, ELL)
    with _phase('qlin_interp'), _ephase('qlin_interp'):
        r_i_coeffs = _interpolate_to_kdeg(chunk_rTA, cfg)
    if return_eval:
        # FUSED path: forward-NTT both factors, multiply pointwise, and return
        # the EVAL-domain row-sum (size n_eval = next_pow2(2K−1)) — NO per-row
        # inverse NTT; the caller sums across chunks and inverts once at
        # finalize. Bit-exact with poly_mul (inverse NTT is linear; the
        # dropped coefficient at index 2K−1 is zero since deg(r·p) = 2K−2).
        with _phase('qlin_polymul'), _ephase('qlin_polymul'):
            kd = r_i_coeffs.size(1)
            n_eval = 1
            while n_eval < 2 * kd - 1:
                n_eval <<= 1
            a_pad = r_i_coeffs.new_zeros((n_chunk, n_eval)); a_pad[:, :kd] = r_i_coeffs
            b_pad = row_polys_chunk.new_zeros((n_chunk, n_eval))
            b_pad[:, :kd] = row_polys_chunk.contiguous()
            ntt_forward_batched(a_pad); ntt_forward_batched(b_pad)
            prod = gl_mul(a_pad, b_pad)
        with _phase('qlin_rowsum'), _ephase('qlin_rowsum'):
            return gl_matvec(prod.T.contiguous(),
                             torch.ones(n_chunk, dtype=torch.uint64, device="cuda"))
    with _phase('qlin_polymul'), _ephase('qlin_polymul'):
        prods = poly_mul_batched(r_i_coeffs, row_polys_chunk.contiguous())
    with _phase('qlin_rowsum'), _ephase('qlin_rowsum'):
        out = gl_matvec(prods.T.contiguous(),
                        torch.ones(n_chunk, dtype=torch.uint64, device="cuda"))
    return out


def _encode_rows_indexed(msgs: torch.Tensor, abs_row_indices: List[int],
                          cfg: LigeroConfig, master_seed: torch.Tensor,
                          ) -> torch.Tensor:
    """Encode rows at the given (possibly non-contiguous) absolute indices.

    msgs: (n, ELL), one row per index in abs_row_indices.
    abs_row_indices: n absolute indices in the joint witness — used to
        derive each row's slack via row_prg.

    Slower than encode_messages's contiguous-batch path (one row_prg
    launch per row instead of one big batched launch); intended for
    sparse access patterns like compute_p_0_streaming.
    """
    n = msgs.size(0)
    if n == 0:
        return torch.empty((0, cfg.K_DEG), dtype=torch.uint64, device="cuda")
    padded = torch.zeros((n, cfg.K_DEG), dtype=torch.uint64, device="cuda")
    padded[:, :cfg.ELL] = msgs
    slack = cfg.K_DEG - cfg.ELL
    if slack > 0:
        idx = torch.tensor(abs_row_indices, dtype=torch.uint64, device="cuda")
        padded[:, cfg.ELL:] = row_prg_indexed(master_seed, idx, slack)
    polys = padded.clone()
    ntt_inverse_batched(polys)
    return polys


class _RowMap:
    """abs_row -> (Variable, local_row), same lookup contract as the plain
    dict _build_row_map used to return (`row_map[abs_row]`, KeyError if out
    of range), but built from O(n_vars) sorted ranges instead of one dict
    entry per row -- same restructure _build_row_lookup/bands_overlapping
    already applies to the band index, applied here too since a variable
    list can be hundreds of thousands of rows."""
    __slots__ = ("_starts", "_ends", "_vars")

    def __init__(self, starts: List[int], ends: List[int], vars_: List[Variable]):
        self._starts, self._ends, self._vars = starts, ends, vars_

    def __getitem__(self, abs_row: int) -> Tuple[Variable, int]:
        i = bisect.bisect_right(self._starts, abs_row) - 1
        if i < 0 or abs_row >= self._ends[i]:
            raise KeyError(abs_row)
        return self._vars[i], abs_row - self._starts[i]


def _build_row_map(vars_list: List[Variable], cfg: LigeroConfig,
                    row_offset_start: int) -> '_RowMap':
    """Build abs_row → (Variable, local_row) for the given var list, walking
    vars in declaration order (one range per variable, not one entry per row)."""
    starts: List[int] = []
    ends: List[int] = []
    vars_: List[Variable] = []
    abs_offset = row_offset_start
    for v in vars_list:
        n = v.n_rows(cfg.ELL)
        if n == 0:
            continue
        starts.append(abs_offset)
        ends.append(abs_offset + n)
        vars_.append(v)
        abs_offset += n
    return _RowMap(starts, ends, vars_)


def _gather_rows(inputs: Dict[Variable, InputVal],
                  cfg: LigeroConfig,
                  row_map: Dict[int, Tuple[Variable, int]],
                  needed_abs_rows: List[int]) -> torch.Tensor:
    """Return a (len(needed_abs_rows), ELL) tensor of message rows at the
    requested absolute indices. `row_map` is pre-built by `_build_row_map`
    so callers in a loop don't pay the build cost per call.

    Per-variable, the needed rows are gathered with one index_select instead
    of a Python loop doing one __setitem__ per row -- found via profiling a
    medium-scale (SEQ=384, 4 layers) run, where this was ~17% of prove()
    wall-clock (the Freivalds-style spot-checks in compute_p_0_streaming
    call this often, with growing needed_abs_rows lists as SEQ grows), the
    same pattern as _iter_message_chunks (see that function's docstring)."""
    ELL = cfg.ELL
    out = torch.zeros((len(needed_abs_rows), ELL), dtype=torch.uint64, device="cuda")
    # Group needed rows by Variable so each weight loads once even if
    # multiple rows are needed from it (matters under lazy loaders).
    by_var: Dict[Variable, List[Tuple[int, int]]] = {}  # var → [(out_i, local_row)]
    for i, abs_row in enumerate(needed_abs_rows):
        v, r = row_map[abs_row]
        by_var.setdefault(v, []).append((i, r))
    for v, items in by_var.items():
        data = _to_device_u64(_fetch(inputs, v)).reshape(-1)
        v_len = data.numel()
        v_rows = v.n_rows(ELL)
        pad_len = v_rows * ELL - v_len
        if pad_len:
            padded = torch.zeros(v_rows * ELL, dtype=torch.uint64, device="cuda")
            padded[:v_len] = data
            rows_v = padded.view(v_rows, ELL)
        else:
            rows_v = data.view(v_rows, ELL)
        out_idx = torch.tensor([i for i, _ in items], dtype=torch.long, device="cuda")
        row_idx = torch.tensor([r for _, r in items], dtype=torch.long, device="cuda")
        gathered = rows_v.index_select(0, row_idx)
        # index_copy_ isn't in _uint64_compat's patched op list (index_select
        # is); pure data movement (row copy, sign never inspected), so the
        # int64 view is bit-identical -- same rule _uint64_compat itself uses.
        out.view(torch.int64).index_copy_(0, out_idx, gathered.view(torch.int64))
        del data
    return out


def compute_p_0_streaming(
    p1_vars: List[Variable],              # phase-1 vars (witness, starting at NUM_BLINDING_ROWS)
    p2_vars: List[Variable],              # phase-2 vars (starting at m_p1_rows)
    inputs: Dict[Variable, InputVal],
    m_p1_rows: int,                       # row index where phase-2 starts
    r_quad: torch.Tensor,
    quad_constraints: List[QuadraticConstraint],
    cfg: LigeroConfig,
    master_seed: torch.Tensor,
    chunk_size: int = 256,
    maps=None,
) -> torch.Tensor:
    """Compute p_0 with a per-chunk working set independent of the total
    quad-constraint count or the size of the witness.

    For each chunk of `chunk_size` constraints: collect the rows that
    chunk references (≤ 3·chunk_size unique), gather them from inputs,
    encode just those rows, run the inner loop, drop everything before
    the next chunk. Peak memory per chunk is ~few hundred MB regardless
    of SEQ — versus the prior implementation, which materialized one
    `(|needed_rows|, K_DEG)` cache for the whole T loop (multi-GB at
    SEQ ≳ 200).

    Trade-off: a row referenced by k chunks gets re-encoded k times.
    For Llama-shape quad sets that averages ~2×; absolute encoding cost
    is dominated by the inner loop work anyway."""
    K = cfg.K_DEG
    T = len(quad_constraints)
    p_0 = torch.zeros(2 * K - 1, dtype=torch.uint64, device="cuda")
    if T == 0:
        return p_0

    # Multiplication is performed in the 2K evaluation domain.  The previous
    # implementation inverse-transformed every product row and only then took
    # the r_quad-weighted row sum.  The inverse NTT is linear, so that did
    # exactly the same work as
    #
    #   iNTT(sum_i r_i * (NTT(px_i)NTT(py_i) +
    #                     NTT(pa_i)NTT(pz_i)))
    #
    # but paid one 2K inverse transform per constraint row.  Accumulate the
    # products in the evaluation domain and invert ONCE at the end.  This is a
    # pure reassociation in F_p: transcript, polynomial coefficients and the
    # verifier are unchanged.
    n_eval = 1
    while n_eval < 2 * K - 1:
        n_eval <<= 1
    eval_acc = torch.zeros(n_eval, dtype=torch.uint64, device="cuda")
    public_acc = torch.zeros(K, dtype=torch.uint64, device="cuda")
    saw_public = False

    # Public a/b rows are uniform on the first n message positions and zero on
    # the rest of the K interpolation domain.  A full chunk normally repeats
    # the same (n,a,b) hundreds of times (notably LogUp's a=-1,b=0).  Interpolate
    # each distinct public row once instead of running an identical iNTT for
    # every constraint.  The cache is local to this call and therefore bounded
    # by the number of distinct public coefficients in one claim.
    public_poly_cache = {}

    def public_poly(n: int, coef: int) -> torch.Tensor:
        key = (int(n), int(coef) % P)
        hit = public_poly_cache.get(key)
        if hit is not None:
            return hit
        mask = (slot_grid < int(n)).to(torch.int64).unsqueeze(0)
        c = torch.tensor([key[1]], dtype=torch.uint64, device="cuda")
        vals = (c.view(torch.int64).unsqueeze(1) * mask).view(torch.uint64)
        hit = _interpolate_to_kdeg(vals.contiguous(), cfg)
        public_poly_cache[key] = hit
        return hit

    # Build row maps once — walking the var lists per chunk would be O(M·rows·T/chunk_size).
    # The streaming prover passes prebuilt maps so per-op calls don't rebuild them.
    if maps is not None:
        p1_row_map, p2_row_map = maps
    else:
        p1_row_map = _build_row_map(p1_vars, cfg, NUM_BLINDING_ROWS)
        p2_row_map = _build_row_map(p2_vars, cfg, m_p1_rows)

    slot_grid = torch.arange(cfg.ELL, dtype=torch.int64, device="cuda")

    for t_lo in range(0, T, chunk_size):
        t_hi = min(t_lo + chunk_size, T)
        chunk_qcs = quad_constraints[t_lo:t_hi]
        chunk = len(chunk_qcs)

        # Rows referenced by this chunk only.
        chunk_needed = set()
        for qc in chunk_qcs:
            chunk_needed.add(qc.x_row); chunk_needed.add(qc.y_row); chunk_needed.add(qc.z_row)
        sorted_needed = sorted(chunk_needed)
        p1_abs = [r for r in sorted_needed if NUM_BLINDING_ROWS <= r < m_p1_rows]
        p2_abs = [r for r in sorted_needed if r >= m_p1_rows]

        # Gather + encode this chunk's rows only.
        if p1_abs:
            p1_msgs = _gather_rows(inputs, cfg, p1_row_map, p1_abs)
            p1_polys = _encode_rows_indexed(p1_msgs, p1_abs, cfg, master_seed)
            del p1_msgs
        else:
            p1_polys = torch.empty((0, K), dtype=torch.uint64, device="cuda")
        if p2_abs:
            p2_msgs = _gather_rows(inputs, cfg, p2_row_map, p2_abs)
            p2_polys = _encode_rows_indexed(p2_msgs, p2_abs, cfg, master_seed)
            del p2_msgs
        else:
            p2_polys = torch.empty((0, K), dtype=torch.uint64, device="cuda")
        polys_cache = torch.cat([p1_polys, p2_polys], dim=0)
        del p1_polys, p2_polys

        # Compact index for this chunk.
        compact_idx_of: Dict[int, int] = {}
        for i, r in enumerate(p1_abs): compact_idx_of[r] = i
        base = len(p1_abs)
        for i, r in enumerate(p2_abs): compact_idx_of[r] = base + i

        x_compact = torch.tensor([compact_idx_of[qc.x_row] for qc in chunk_qcs],
                                  dtype=torch.int64, device="cuda")
        y_compact = torch.tensor([compact_idx_of[qc.y_row] for qc in chunk_qcs],
                                  dtype=torch.int64, device="cuda")
        z_compact = torch.tensor([compact_idx_of[qc.z_row] for qc in chunk_qcs],
                                  dtype=torch.int64, device="cuda")
        px = polys_cache.index_select(0, x_compact)
        py = polys_cache.index_select(0, y_compact)
        pz = polys_cache.index_select(0, z_compact)

        a_keys = [(qc.n, qc.a_values[0] % P) for qc in chunk_qcs]
        b_keys = [(qc.n, qc.b_values[0] % P) for qc in chunk_qcs]
        have_a = any(a != 0 for _, a in a_keys)
        have_b = any(b != 0 for _, b in b_keys)

        # One pair of batched forward NTT launches for both products.  Rows
        # with a=0 need no a*z product at all.
        lhs_parts, rhs_parts = [px], [py]
        if have_a:
            pa = torch.cat([public_poly(n, a) for n, a in a_keys], dim=0)
            lhs_parts.append(pa)
            rhs_parts.append(pz)
        lhs_k = torch.cat(lhs_parts, dim=0)
        rhs_k = torch.cat(rhs_parts, dim=0)
        lhs = torch.zeros((lhs_k.size(0), n_eval), dtype=torch.uint64, device="cuda")
        rhs = torch.zeros_like(lhs)
        lhs[:, :K] = lhs_k
        rhs[:, :K] = rhs_k
        ntt_forward_batched(lhs)
        ntt_forward_batched(rhs)
        products = gl_mul(lhs, rhs)
        product_sum = products[:chunk]
        if have_a:
            product_sum = gl_add(product_sum, products[chunk:])
        weighted_eval = gl_matvec(product_sum.T.contiguous(), r_quad[t_lo:t_hi])
        eval_acc = gl_add(eval_acc, weighted_eval)

        # The public b polynomial is already in coefficient form.  Accumulate
        # it separately and subtract after the single inverse transform.  The
        # overwhelmingly common b=0 family avoids this path entirely.
        if have_b:
            saw_public = True
            pb = torch.cat([public_poly(n, b) for n, b in b_keys], dim=0)
            public_acc = gl_add(
                public_acc,
                gl_matvec(pb.T.contiguous(), r_quad[t_lo:t_hi]))

        del (polys_cache, px, py, pz, lhs_k, rhs_k, lhs, rhs, products,
             product_sum, weighted_eval)
        if have_a:
            del pa
        if have_b:
            del pb

    ntt_inverse(eval_acc)
    p_0 = eval_acc[:2 * K - 1].contiguous()
    if saw_public:
        public_pad = torch.zeros(2 * K - 1, dtype=torch.uint64, device="cuda")
        public_pad[:K] = public_acc
        p_0 = gl_sub(p_0, public_pad)
    return p_0


def gl_sum_mod_p(vec: torch.Tensor) -> int:
    """Σ vec mod P. gl_matvec(v.unsqueeze(0), ones) is single-threaded
    (parallelizes per output row, of which there's only one) — too slow for
    multi-million-element vectors. Reshape to (~√n, ~√n) for parallelism.
    Returns a Python int."""
    n = vec.numel()
    if n == 0:
        return 0
    if n < 4096:
        # Small enough that single-threaded matvec is fine.
        return int(gl_matvec(vec.unsqueeze(0).contiguous(),
                              torch.ones(n, dtype=torch.uint64, device=vec.device)).item())
    # Two-level reduction: chunk sums via parallel matvec, then final reduction.
    block_size = int(n ** 0.5) + 1
    block_count = (n + block_size - 1) // block_size
    padded_len = block_count * block_size
    if padded_len == n:
        chunks = vec.view(block_count, block_size)
    else:
        padded = torch.zeros(padded_len, dtype=torch.uint64, device=vec.device)
        padded[:n] = vec
        chunks = padded.view(block_count, block_size)
    ones_in = torch.ones(block_size,  dtype=torch.uint64, device=vec.device)
    chunk_sums = gl_matvec(chunks.contiguous(), ones_in)        # (block_count,)
    ones_out = torch.ones(block_count, dtype=torch.uint64, device=vec.device)
    return int(gl_matvec(chunk_sums.unsqueeze(0).contiguous(), ones_out).item())


# ============================================================
# Proof dataclass.
# ============================================================

@dataclass
class Proof:
    root_p1: bytes
    root_p2: bytes
    q_irs: torch.Tensor       # (K_DEG,) on device
    q_lin: torch.Tensor       # (2K-1,) on device
    p_0:   torch.Tensor       # (2K-1,) on device
    opened_p1: Dict[int, torch.Tensor]   # j → (m_p1_rows,) on device
    opened_p2: Dict[int, torch.Tensor]
    paths_p1:  Dict[int, List[Tuple[bytes, int]]]
    paths_p2:  Dict[int, List[Tuple[bytes, int]]]
    # Persistent W block (analysis/persistent-weights.md). None/empty when no
    # weight vars are marked persistent.
    root_w:    Optional[bytes] = None
    opened_w:  Dict[int, torch.Tensor] = field(default_factory=dict)
    paths_w:   Dict[int, List[Tuple[bytes, int]]] = field(default_factory=dict)
    # Second weight block "wnew" (linking proofs, P5): the refreshed
    # commitment's tree — the verifier adopts root_wnew as the new trusted
    # R_W′ once the proof (whose LinComb equality binds it to root_w == the
    # currently-trusted R_W) verifies.
    root_wnew:    Optional[bytes] = None
    opened_wnew:  Dict[int, torch.Tensor] = field(default_factory=dict)
    paths_wnew:   Dict[int, List[Tuple[bytes, int]]] = field(default_factory=dict)
    # Blinding block (layout B): its own tree at the front rows. When set, the
    # proof carries an explicit `blocks` order and root_p1 is the activations-
    # only tree; when None, root_p1 is the legacy blinding+phase-1 tree.
    root_blind:   Optional[bytes] = None
    opened_blind: Dict[int, torch.Tensor] = field(default_factory=dict)
    paths_blind:  Dict[int, List[Tuple[bytes, int]]] = field(default_factory=dict)
    # Phase-3 block: the late auxiliaries committed in R3, after the coin
    # s_bind. None when no claim on the tape has a late stage.
    root_p3:    Optional[bytes] = None
    opened_p3:  Dict[int, torch.Tensor] = field(default_factory=dict)
    paths_p3:   Dict[int, List[Tuple[bytes, int]]] = field(default_factory=dict)
    blocks:       Optional[List[str]] = None
    # Sequential Fiat-Shamir transcript (S1). `seeds` are the coins the prover
    # DERIVED (never chose): the verifier recomputes each one from the same
    # transcript and ignores these. `statement_digest` is the digest of
    # `claims_bytes`, the exact canonical claim-set bytes the proof file
    # carries; `Q_cols` are the columns s_col selected.
    seeds:            Optional[Dict[str, bytes]] = None
    statement_digest: Optional[bytes] = None
    claims_bytes:     Optional[bytes] = None
    Q_cols:           Optional[List[int]] = None


def _u64_le_bytes(t: torch.Tensor) -> bytes:
    """A device u64 field vector as little-endian u64 bytes — the transcript
    framing the Rust verifier hashes (protocol.u64_bytes)."""
    return t.cpu().contiguous().view(torch.uint8).numpy().tobytes()


# ============================================================
# Framework: per-claim registries, _layout, prove, verify.
# ============================================================

# Per-claim challenge sampler: (claim, verifier) → challenge (any shape).
SAMPLE_FNS: Dict[Type, Callable] = {}
# Per-claim auxiliary-witness computer: (claim, witness, challenge) → dict.
AUX_FNS: Dict[Type, Callable] = {}
# LATE (phase-3) counterparts, for claims that need a second, independent coin
# sampled AFTER their phase-2 rows are committed — the routed-projected
# matmul's Freivalds check of Q = M·P is the reason these exist. A claim
# registers here only if it owns phase-3 variables; every other claim is
# untouched and its transcript position is unchanged.
LATE_SAMPLE_FNS: Dict[Type, Callable] = {}
LATE_AUX_FNS: Dict[Type, Callable] = {}
# Callables run at the start of every proof, for per-proof state that lives
# outside core (currently the routed claim's challenge-keyed projection cache).
# A hook must only DROP state: nothing here may affect what is proved.
PROVE_START_HOOKS: List[Callable] = []
# Claim types whose compute/aux resolve their OWN inputs, one shard at a time.
# The sweep must not pre-fetch their inputs and must not hand them a caching
# view: a routed MoE matmul references 128 expert shards (~43 GB for one
# Maverick matrix), so materializing them together is an OOM, not a slowdown.
# Their compute takes (claim, live, ch) — `live` still holds unresolved
# loaders — and is responsible for releasing each shard before the next.
STREAMING_INPUT_CLAIMS: set = set()


# Set True by the streaming provers around _compile_with_chs. They DISCARD the
# returned b_chunks (the verifier re-derives the RHS itself), so building these
# dense per-constraint tensors is pure waste — and at 32L SEQ=1000 they
# accumulate to ~114 GB (matmul 10.6 + softmax 9.2 + hadamard ... at 8L → ×4),
# which is the prover's peak. Skipping them drops the peak to the sweep's ~6 GB.
_SKIP_B_CHUNK = False


def _build_b_chunk(n_added: int,
                    nz_families: List[Tuple[int, int, Any]]) -> Optional[torch.Tensor]:
    """Build a per-claim b_chunk of length n_added with the listed non-zero
    families filled in (other entries stay zero). Returns None if there are
    no non-zero families — most claims fall here, so the framework can skip
    the chunk entirely.

    nz_families: list of (rel_start, length, b_value) tuples. b_value is
    either a Python int (scalar broadcast across the range) or a uint64
    tensor of length `length` (per-cell).
    """
    if _SKIP_B_CHUNK or not nz_families:
        return None
    b = torch.zeros(n_added, dtype=torch.uint64, device="cuda")
    for rel_start, length, value in nz_families:
        if isinstance(value, torch.Tensor):
            b[rel_start:rel_start + length] = value
        else:
            # Direct slice assignment from a Python int routes through
            # long long, which overflows for values ≥ 2^63 (common — most
            # b's are encoded as P − k near 2^64). Build a uint64 tensor
            # via numpy, which handles the wraparound losslessly.
            b[rel_start:rel_start + length] = torch.from_numpy(
                np.array(value % P, dtype=np.uint64)).cuda()
    return b

# Per-claim compile_fn registry.
# Signature: (claim, ch, cfg, constraint_id_base) → (List[(witness_row, packet)],
#                                                    List[QuadraticConstraint],
#                                                    n_new_constraints,
#                                                    Optional[b_chunk uint64 tensor]).
# Every claim type passed to prove()/verify() must have an entry here.
COMPILE_FNS: Dict[Type, Callable] = {}


def _sample_chs(claims: List, s_op) -> List:
    """One op challenge per claim, each derived from the round-1 seed s_op by the
    claim's index in this (settled) list — the same scheme protocol.op_vec and the
    Rust verifier use, so prover and verifier agree bit-for-bit with no challenge
    values sent. No compile work."""
    return [SAMPLE_FNS[type(c)](c, ci, s_op) for ci, c in enumerate(claims)]


def _sample_late_chs(claims: List, s_bind) -> List:
    """The phase-3 coins, from the R2 seed. None for claims with no late
    stage, so the index stays the claim's position in the settled list."""
    return [LATE_SAMPLE_FNS[type(c)](c, ci, s_bind)
            if type(c) in LATE_SAMPLE_FNS else None
            for ci, c in enumerate(claims)]


def _compile_with_chs(claims: List, chs_per_claim: List, cfg: LigeroConfig,
                       m_total: int) -> Tuple[List[List[Any]],
                                               List[QuadraticConstraint],
                                               List[Tuple[int, torch.Tensor]],
                                               int]:
    """Compile claims given already-sampled challenges. Returns
    (per_row_packets, quadratic_all, b_chunks, n_constraints).

    Each compile_fn returns (row_pkts, quads, n_added, b_chunk). b_chunk
    is Optional[Tensor] of length n_added (RHS of each constraint);
    None means all-zero. Non-None b_chunks accumulate as
    (constraint_base, b_chunk) pairs consumed by the verifier's linear
    sum identity check."""
    per_row: List[List[Any]] = [[] for _ in range(m_total)]
    quadratic_all: List[QuadraticConstraint] = []
    b_chunks: List[Tuple[int, torch.Tensor]] = []
    n_constraints = 0
    import os as _os
    _prof = _os.environ.get("LIGERO_COMPILE_PROFILE")
    _agg = {}; _gpeak = 0; _gtype = None
    for c, ch in zip(claims, chs_per_claim):
        compile_fn = COMPILE_FNS.get(type(c))
        if compile_fn is None:
            raise NotImplementedError(
                f"no COMPILE_FNS entry for {type(c).__name__}")
        if _prof:
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            _b = torch.cuda.memory_allocated()
        row_pkts, quads, n_added, b_chunk = compile_fn(c, ch, cfg, n_constraints)
        if _prof:
            torch.cuda.synchronize()
            _tn = type(c).__name__; _mp = torch.cuda.max_memory_allocated()
            _a = _agg.setdefault(_tn, [0, 0, 0])
            _a[0] += torch.cuda.memory_allocated() - _b      # net retained (its b_chunk)
            _a[1] = max(_a[1], _mp - _b)                     # transient above baseline
            _a[2] += 1
            if _mp > _gpeak: _gpeak = _mp; _gtype = _tn
        for r, pkt in _expand_row_pkts(row_pkts):
            per_row[r].append(pkt)
        for fam in quads:                        # quad lift: families -> per-row
            quadratic_all.extend(fam.expand())
        if b_chunk is not None:
            b_chunks.append((n_constraints, b_chunk))
        n_constraints += n_added
    if _prof:
        print(f"=== compile GPU profile: global_peak={_gpeak/1e9:.2f}GB (at {_gtype})  "
              f"cur_after={torch.cuda.memory_allocated()/1e9:.2f}GB ===", flush=True)
        for _tn in sorted(_agg, key=lambda t: -_agg[t][1]):
            _r, _tp, _n = _agg[_tn]
            print(f"  {_tn:22s} n={_n:>4} retained={_r/1e9:7.3f}GB  transient={_tp/1e9:7.3f}GB", flush=True)
        if _os.environ.get("LIGERO_COMPILE_PROFILE_EXIT", "1") == "1":
            import sys; sys.exit(0)
    return per_row, quadratic_all, b_chunks, n_constraints


def _compile_all(claims: List, s_op, cfg: LigeroConfig,
                  m_total: int) -> Tuple[List[List[Any]],
                                          List[QuadraticConstraint],
                                          List,
                                          List[Tuple[int, torch.Tensor]],
                                          int]:
    """Sample + compile in one call (used by verify() and
    _sample_test_challenges). op challenges derived from the round-1 seed s_op."""
    chs = _sample_chs(claims, s_op)
    per_row, quads, b_chunks, n_lin = _compile_with_chs(claims, chs, cfg, m_total)
    return per_row, quads, chs, b_chunks, n_lin


# _sample_test_challenges moved to tests/test_prover.py with prove() (see note
# above prove_streaming / Tape.prove). verify() derives its own per-round
# challenges via pr.round_seeds + _compile_all.


# ============================================================
# TableSettlement — framework-internal LogUp "table side" protocol.
# Auto-synthesized by prove()/verify() for each unique Table referenced
# by any user claim. Users normally don't construct it directly; tests
# that build claim lists by hand may include it explicitly and the
# auto-synthesis will dedup.
# ============================================================


@dataclass
class TableSettlement:
    table: 'Table'


def table_settlement_sample(c: TableSettlement, ci: int, s_op):
    # α (and β for paired tables) derived from the round-1 seed by the table's
    # settled-list index — shared with the Rust verifier (handlers.rs derives the
    # same values the same way).
    c.table.alpha = pr.challenge(s_op, 0, f"op{ci}:alpha")
    if c.table.T_Y is not None:
        c.table.beta = pr.challenge(s_op, 0, f"op{ci}:beta")
    return None


def table_settlement_aux(c: TableSettlement, witness: dict, _ch) -> dict:
    """w[j] = mult[j] / (α - v[j]) with v[j] = T[j] + β·T_Y[j] (paired) or T[j] (range)."""
    table = c.table
    mult_val = witness[table.mult_var]
    mult_t = mult_val if isinstance(mult_val, torch.Tensor) else torch.tensor(
        mult_val, dtype=torch.uint64, device="cuda")
    mult_t = mult_t.contiguous().view(-1)
    if table.T_Y is not None:
        beta_t = torch.full_like(table.T, table.beta)
        v = gl_add(table.T, gl_mul(beta_t, table.T_Y))
    else:
        v = table.T
    alpha_t = torch.full_like(table.T, table.alpha)
    return {table.w_var: gl_mul(mult_t, gl_inv_batched(gl_sub(alpha_t, v)))}


SAMPLE_FNS[TableSettlement] = table_settlement_sample
AUX_FNS[TableSettlement]    = table_settlement_aux


def table_settlement_compile(c: TableSettlement, _ch, cfg: LigeroConfig,
                                     base: int):
    """Compile for TableSettlement.

    Emits T_LEN per-row product constraints (IDs [base, base+T_LEN)) plus
    one sum-identity constraint (ID base+T_LEN):

      Per row j ∈ [0, T_LEN):  (α − v[j])·w[j] − mult[j] = 0
        w   → L2_PerSlotVector with coef_vec = (α − v)
        mult → L2_IdentityScalar with coef = P−1

      Sum:  Σ_{z over using-claims} z[i] − Σ_j w[j] = 0
        each z → L2_StrideManyToOneScalar(stride = z.length, coef = 1)
        w     → L2_StrideManyToOneScalar(stride = T_LEN,    coef = P−1)
        (stride = full-var-length collapses every slot of that variable onto
         the single sum constraint at base + T_LEN.)
    """
    table = c.table
    ell = cfg.ELL
    T_LEN = table.T.numel()
    alpha = table.alpha
    neg1 = (P - 1) % P

    # coef_vec[j] = (α − v[j]) for the w-side, where v = T (range) or T + β·T_Y (paired).
    if table.T_Y is not None:
        beta_vec = torch.full((T_LEN,), table.beta, dtype=torch.uint64, device="cuda")
        v = gl_add(table.T, gl_mul(beta_vec, table.T_Y))
    else:
        v = table.T
    alpha_vec = torch.full((T_LEN,), alpha, dtype=torch.uint64, device="cuda")
    w_coef_vec = gl_sub(alpha_vec, v).contiguous()

    sum_cid = base + T_LEN

    # Every packet below is IDENTICAL across its whole row range (only the
    # row_off used to vary was never actually read by the packet fields --
    # var_row_start is the VARIABLE's start, not the row's own index), so
    # each is handed to _index_bands as one O(1) _PktRange instead of
    # row_end - row_start separate (row, pkt) tuples. table.w_var alone can
    # be a lookup-table-sized variable (hundreds of thousands of rows), so
    # this is the difference between one range object and one Python object
    # + list-append per row. The eager (non-streaming) compile path expands
    # these back to per-row tuples via _expand_row_pkts, unchanged for it.
    w_rows = table.w_var.n_rows(ell)
    mult_rows = table.mult_var.n_rows(ell)

    row_pkts: List[object] = [
        _PktRange(
            L2_PerSlotVector(base=base, var_row_start=table.w_var.row_start,
                              L=T_LEN, coef_vec=w_coef_vec),
            table.w_var.row_start, table.w_var.row_start + w_rows),
        _PktRange(
            L2_IdentityScalar(base=base, var_row_start=table.mult_var.row_start,
                               L=T_LEN, coef=neg1),
            table.mult_var.row_start, table.mult_var.row_start + mult_rows),
    ]
    # Sum identity (single constraint at sum_cid). Stride = full var length
    # → every slot of that variable collapses onto base + 0 = sum_cid, so
    # this is still one range per z / for w_var. Order matches the old
    # per-row emission exactly (z_vars, THEN w_var's own sum term) --
    # per_row[r] for a w_var row accumulates both its L2_PerSlotVector and
    # this term, and the eager path's per-row list order is observable.
    for z in table.z_vars:
        z_rows = z.n_rows(ell)
        row_pkts.append(_PktRange(
            L2_StrideManyToOneScalar(base=sum_cid, var_row_start=z.row_start,
                                      L=z.length, stride=z.length, coef=1),
            z.row_start, z.row_start + z_rows))
    row_pkts.append(_PktRange(
        L2_StrideManyToOneScalar(base=sum_cid, var_row_start=table.w_var.row_start,
                                  L=T_LEN, stride=T_LEN, coef=neg1),
        table.w_var.row_start, table.w_var.row_start + w_rows))

    return row_pkts, [], T_LEN + 1, None


COMPILE_FNS[TableSettlement] = table_settlement_compile


def _collect_tables(claims: List) -> List['Table']:
    """Find every Table referenced from any claim's fields (one level deep),
    *minus* tables already covered by an explicit TableSettlement in the
    claim list. Used by prove()/verify() to auto-synthesize settlements
    without duplicating them when callers also pass them explicitly."""
    explicit = {id(c.table) for c in claims if isinstance(c, TableSettlement)}
    seen: set = set()
    tables: List[Table] = []
    for c in claims:
        for f in fields(c):
            v = getattr(c, f.name)
            if isinstance(v, Table) and id(v) not in explicit and id(v) not in seen:
                seen.add(id(v))
                tables.append(v)
    return tables


def _with_synthesized_settlements(claims: List) -> List:
    """Return claims with one TableSettlement per unique referenced Table
    APPENDED after the operations (deduped against any explicit settlements).

    Settle-after-ops matches the Rust verifier (handlers.rs), which
    runs the per-op constraints first and settles each shared table last (the
    LogUp sum identity naturally closes after every lookup z exists). Both sides
    use THIS function, so layout (row_start) and cid numbering stay mutually
    consistent; the ordering is what the trusted protocol.py mirrors."""
    return list(claims) + [TableSettlement(table=t) for t in _collect_tables(claims)]


def _layout(claims: List, cfg: LigeroConfig):
    """Walk claims, collect unique Variables, assign row_start by phase.
    Returns (all_vars, p1_vars, p2_vars, m_p1_rows, m_total_rows).

    Rows 0..NUM_BLINDING_ROWS-1 of R_p1 are reserved for ZK blinding
    (managed directly by prove/verify, not via Variables). Witness
    Variables start at row NUM_BLINDING_ROWS.

    Also walks one level into Table-typed fields to find shared mult/w
    Variables and the Table's accumulated z_vars list, so shared-table
    claims (RangeWordClaim, TableSettlement) get correctly laid out."""
    all_vars: List[Variable] = []
    seen: set = set()
    def add(v):
        if isinstance(v, Variable) and id(v) not in seen:
            seen.add(id(v))
            all_vars.append(v)
    for c in claims:
        # Streaming relayout: a table's shared mult/w and its collected z's are
        # placed only at the table's APPENDED TableSettlement, not at the first
        # using-claim. So mult/w fall to the bottom of their phase, and each
        # per-lookup z stays at its own using-claim (the z is a direct field,
        # already added above). This keeps row-order == op-order so the prover
        # can stream the witness in one forward sweep; the settlement's z-walk
        # is idempotent (z's already placed). The verifier reads row_start from
        # the claim JSON and its only var-walk is order-independent, so this
        # needs no verifier change.
        is_settlement = isinstance(c, TableSettlement)
        for f in fields(c):
            v = getattr(c, f.name)
            add(v)
            if isinstance(v, Table):
                if is_settlement:
                    add(v.mult_var)
                    add(v.w_var)
                    for z in v.z_vars:
                        add(z)
            elif isinstance(v, list):
                # WordExtractionClaim.words is List[Variable]; walk one level.
                for item in v:
                    add(item)
    # Row-blocks in order (analysis/persistent-weights.md, layout B):
    #   [ blind 0..NUM_BLINDING | W (persistent phase-1) | Wnew (linking
    #     proofs only) | p1 (other phase-1) | p2 ]
    # Blinding is its own tree at the front rows, so the IRS witness index
    # (abs-NUM_BLINDING) is UNCHANGED. The W block leads the witness at the
    # FIXED offset NUM_BLINDING, so R_W is context-independent — it reproduces
    # across proofs of different prompts and matches commit_weights bit-for-bit.
    # The Wnew block (P5) holds the REFRESHED copy of the weights in a linking
    # proof; it sits after W physically but pads at logical NUM_BLINDING under
    # the refresh seed, so its root reproduces the refreshed R_W′.
    # Each block's vars are assigned row_starts in all_vars (op) order, so the
    # streaming sweep feeds each block's tree in row order even though weights
    # and activations interleave in op order.
    weight_vars = [v for v in all_vars if v.phase == 1 and v.persistent and not v.w_new]
    wnew_vars   = [v for v in all_vars if v.phase == 1 and v.persistent and v.w_new]
    p1_vars     = [v for v in all_vars if v.phase == 1 and not v.persistent]
    p2_vars     = [v for v in all_vars if v.phase == 2]
    p3_vars     = [v for v in all_vars if v.phase == 3]
    next_row = NUM_BLINDING_ROWS    # rows 0..NUM_BLINDING_ROWS-1 → blinding (its own tree)
    for v in weight_vars:
        v.row_start = next_row
        next_row += v.n_rows(cfg.ELL)
    m_w_rows = next_row - NUM_BLINDING_ROWS
    for v in wnew_vars:
        v.row_start = next_row
        next_row += v.n_rows(cfg.ELL)
    m_wnew_rows = next_row - NUM_BLINDING_ROWS - m_w_rows
    for v in p1_vars:
        v.row_start = next_row
        next_row += v.n_rows(cfg.ELL)
    m_p1_rows = next_row                # end of phase-1 (W + Wnew + activations), start of p2
    for v in p2_vars:
        v.row_start = next_row
        next_row += v.n_rows(cfg.ELL)
    m_p2_rows = next_row                # end of phase-2, start of phase-3
    for v in p3_vars:
        v.row_start = next_row
        next_row += v.n_rows(cfg.ELL)
    return (all_vars, p1_vars, p2_vars, p3_vars, m_p1_rows, m_p2_rows, next_row,
            weight_vars, m_w_rows, wnew_vars, m_wnew_rows)


def _claim_var_groups(claims, cfg):
    """Mirror _layout's var-walk, returning [(claim, p1_vars, p2_vars, p3_vars)]
    in claim order with each list in row order. The streaming prover encodes a
    claim's own vars right after generating them, so it needs this per-claim
    split of the (relaid-out) layout. The walk MUST match _layout exactly
    (same TableSettlement guard) or row-order != op-order and roots diverge."""
    seen = set()
    groups = []
    def collect(v, buckets):
        if isinstance(v, Variable) and id(v) not in seen:
            seen.add(id(v))
            buckets[min(v.phase, 3) - 1].append(v)
    for c in claims:
        buckets = ([], [], [])
        is_settlement = isinstance(c, TableSettlement)
        for f in fields(c):
            v = getattr(c, f.name)
            collect(v, buckets)
            if isinstance(v, Table):
                if is_settlement:
                    collect(v.mult_var, buckets)
                    collect(v.w_var, buckets)
                    for z in v.z_vars:
                        collect(z, buckets)
            elif isinstance(v, list):
                for item in v:
                    collect(item, buckets)
        groups.append((c, *buckets))
    return groups


# --- Witness cache (autoresearch iter 2) --------------------------------------
# The sound prover re-runs the witness forward pass once per Fiat-Shamir sweep
# (4x). Each compute_fn output is a DETERMINISTIC function of committed inputs,
# so it is byte-identical on every sweep; caching it across sweeps yields a
# byte-identical witness -> byte-identical proof -> ACCEPT unchanged (the whole
# point of the 4 rounds — fixing each commitment before the next challenge — is
# preserved: we recompute nothing challenge-dependent, only reuse deterministic
# values). We cache ONLY the expensive-to-recompute, small-in-memory ops
# (softmax's s1_at CPU binary search; silu's numpy decomposition), NEVER the
# aux/Freivalds witnesses (those depend on the round's challenges and are the
# `aux` phase, not touched here). A cumulative-element cap bounds memory at
# large scale and degrades gracefully to recompute. LIGERO_WITNESS_CACHE=0 off.
_WITNESS_CACHE_ON = os.environ.get("LIGERO_WITNESS_CACHE", "1") != "0"
_WITNESS_CACHE_TYPES = frozenset({"SoftmaxClaim", "SiluClaim"})
# Memory budget for the cache. The old fixed 2e8-element (~1.6GB) cap was far
# too tight: at seq1024 it engaged partway through sweep 1 and gated most
# caching out, dropping the win from ~50% to 10.8% while peak mem was only
# 5.7GB / 32GB (iter3 sweep, measured). We now size the budget to a fraction of
# the *actually free* GPU memory at cache-init, so it auto-scales to the card
# and still degrades to recompute only when memory is genuinely tight. Which ops
# get cached may thus vary with free memory, but cached and recomputed outputs
# are byte-identical (validated), so this affects ONLY timing, never the proof.
# LIGERO_WITNESS_CACHE_MEM_FRACTION overrides the fraction; the legacy
# LIGERO_WITNESS_CACHE_MAX_ELEMS still applies as an optional hard ceiling.
_WITNESS_CACHE_MEM_FRACTION = float(
    os.environ.get("LIGERO_WITNESS_CACHE_MEM_FRACTION", "0.25"))
_WITNESS_CACHE_MAX_ELEMS = int(
    os.environ.get("LIGERO_WITNESS_CACHE_MAX_ELEMS", "0")) or None  # None = no hard cap


def _witness_cache_budget_bytes():
    """Bytes the cache may use: a fraction of currently-free GPU memory, leaving
    the rest for prove()'s later working set (encode/merkle/etc.). Returns 0 if
    memory can't be queried, which degrades the cache to a no-op (safe)."""
    try:
        free, _total = torch.cuda.mem_get_info()
    except Exception:
        return 0
    return int(free * _WITNESS_CACHE_MEM_FRACTION)


# --- Witness SPILL: store-once, re-read (host memory) instead of recompute. ---
# Same soundness argument as the GPU witness cache (only deterministic,
# challenge-INDEPENDENT compute_fn outputs are reused; nothing challenge-derived
# is touched, so commit-before-challenge is untouched). The difference is WHERE
# the reused witness lives: pinned HOST memory instead of scarce GPU memory, so
# it holds far more than the card and covers the regime "witness > GPU but the
# forward-pass recompute is slower than a host->device re-read" (measured on this
# box: recompute ~8 GB/s vs H2D ~11 GB/s -> re-read wins narrowly). Whether it
# wins depends on the deployment (recompute throughput vs storage bandwidth), so
# it is OPT-IN (LIGERO_WITNESS_SPILL=1) and validated byte-identical.
_WITNESS_SPILL_ON = os.environ.get("LIGERO_WITNESS_SPILL", "0") != "0"
_WITNESS_SPILL_MEM_FRACTION = float(
    os.environ.get("LIGERO_WITNESS_SPILL_MEM_FRACTION", "0.5"))
# DISK-backed spill: for the production case where the witness (7.5 TB at 400B)
# exceeds host RAM, so it must land on disk, not pinned host memory. Same
# soundness as host spill (identical int64 bit pattern re-read), but covers the
# FULL witness (every claim type, not just softmax/silu) and its budget is disk
# free space. Re-reads over storage BW instead of recomputing. LIGERO_WITNESS_
# SPILL_DISK=1 + LIGERO_WITNESS_SPILL_DIR=/path (a big/fast filesystem).
_WITNESS_SPILL_DISK = os.environ.get("LIGERO_WITNESS_SPILL_DISK", "0") != "0"
_WITNESS_SPILL_DIR = os.environ.get("LIGERO_WITNESS_SPILL_DIR", "/tmp")
# BENCHMARK-ONLY, default OFF: evict each spilled range from the page cache
# (fdatasync + posix_fadvise DONTNEED) so re-reads hit the DISK, not RAM. Lets a
# small model measure true disk-spill I/O without needing witness > RAM. Changes
# ONLY timing -- the bytes read back are identical -> proof byte-for-byte unchanged.
# NEVER engages unless LIGERO_SPILL_FADVISE=1 AND disk-spill is on, so it cannot
# affect any run that does not explicitly ask for it.
_SPILL_FADVISE = os.environ.get("LIGERO_SPILL_FADVISE", "0") != "0"


def _host_spill_budget_bytes():
    """Bytes the host spill may use: a fraction of available host RAM (from
    /proc/meminfo MemAvailable). Returns 0 if unqueryable -> spill degrades to
    recompute (safe)."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    avail = int(line.split()[1]) * 1024
                    return int(avail * _WITNESS_SPILL_MEM_FRACTION)
    except Exception:
        pass
    return 0


def _spill_store(t):
    """Copy a witness tensor to pinned host memory for later re-read. The int64
    view sidesteps uint64 CUDA/pin gaps; the bit pattern (hence field value) is
    identical, so re-reading is byte-for-byte the same as recomputing."""
    src = t.contiguous()
    src_i = src.view(torch.int64) if src.dtype == torch.uint64 else src
    host = torch.empty(src_i.shape, dtype=src_i.dtype, pin_memory=True)
    host.copy_(src_i)
    return (host, t.dtype)


def _spill_load(entry):
    """Re-read a spilled witness tensor back onto the GPU (fresh, independent
    tensor -> no clone needed)."""
    host, dt = entry
    dev = host.to("cuda", non_blocking=True)
    return dev.view(dt) if dt == torch.uint64 else dev


def _disk_spill_open(wc):
    """Open a per-proof append-only spill file on disk; budget = free space on
    the spill dir's filesystem (the 7.5 TB witness cannot fit host RAM)."""
    import tempfile, shutil
    os.makedirs(_WITNESS_SPILL_DIR, exist_ok=True)
    fd, path = tempfile.mkstemp(prefix="ligero_wspill_", dir=_WITNESS_SPILL_DIR)
    wc['_disk_fd'], wc['_disk_path'], wc['_disk_off'] = fd, path, 0
    wc['_budget_bytes'] = int(shutil.disk_usage(_WITNESS_SPILL_DIR).free * 0.9)
    return wc


def _disk_spill_store(t, wc):
    """Append a witness tensor's int64 bytes to the disk file; return an
    (offset, nbytes, shape, dtype) handle. The bit pattern is identical to
    recompute, so re-reading is byte-for-byte the same -> proof unchanged."""
    src = t.contiguous()
    src_i = src.view(torch.int64) if src.dtype == torch.uint64 else src
    buf = src_i.cpu().numpy().tobytes()
    off = wc['_disk_off']; n = 0
    while n < len(buf):
        n += os.pwrite(wc['_disk_fd'], buf[n:], off + n)
    if _SPILL_FADVISE:                     # bench-only: flush + drop from cache -> cold reads
        try:
            os.fdatasync(wc['_disk_fd'])
            os.posix_fadvise(wc['_disk_fd'], off, len(buf), os.POSIX_FADV_DONTNEED)
        except (OSError, AttributeError):
            pass
    wc['_disk_off'] = off + len(buf)
    return (off, len(buf), tuple(src_i.shape), t.dtype)


def _disk_spill_load(entry, wc):
    """Read a spilled tensor back from disk onto the GPU (fresh, independent)."""
    import numpy as _np
    off, nb, shape, dt = entry
    buf = bytearray(nb); got = 0
    while got < nb:
        chunk = os.pread(wc['_disk_fd'], nb - got, off + got)
        if not chunk:
            raise IOError("disk-spill short read")
        buf[got:got + len(chunk)] = chunk; got += len(chunk)
    if _SPILL_FADVISE:                     # bench-only: drop the just-read pages -> next re-read also cold
        try: os.posix_fadvise(wc['_disk_fd'], off, nb, os.POSIX_FADV_DONTNEED)
        except (OSError, AttributeError): pass
    arr = _np.frombuffer(bytes(buf), dtype=_np.int64).reshape(shape)
    dev = torch.from_numpy(arr.copy()).to("cuda", non_blocking=True)
    return dev.view(dt) if dt == torch.uint64 else dev


def _disk_spill_close(wc):
    """Close and delete the per-proof spill file."""
    if wc and wc.get('_disk_fd') is not None:
        try: os.close(wc['_disk_fd'])
        except Exception: pass
        try: os.remove(wc['_disk_path'])
        except Exception: pass
        wc['_disk_fd'] = None


def _stream_sweep(tape, cfg, master_seed_t, groups, n_ops, p1_vars, p2_vars, m_p1_rows,
                  tables, ch0, *, want_aux, ch1=None, p3_vars=(), m_p2_rows=None,
                  merkle_p1=None, merkle_p2=None, merkle_p3=None, merkle_w=None,
                  merkle_wnew=None, merkle_blind=None, q_irs=None, q_lin=None,
                  col_p1=None, col_p2=None, col_p3=None,
                  col_w=None, col_wnew=None, col_blind=None, p_0=None,
                  w_pad=None, wnew_pad=None,
                  stream_pk=None, r_quad=None, p_maps=None, Q_cols=None, p1_prefix=None,
                  witness_cache=None, w_owned=None):
    """One op-order streaming pass: regenerate the witness, encode each op's rows
    into whichever accumulators are non-None, fire its quads into p_0 (if given)
    from `live`, freeing per op. want_aux=False does phase-1 only (the commit
    round before α exists). Fast mode calls this once with every accumulator;
    sound mode calls it once per round with the round's subset. Returns p_0.

    `ch1` (the phase-3 coins, sampled after the R2 commitment) enables the LATE
    aux stage: claims registered in LATE_AUX_FNS produce their phase-3 rows
    only when it is present, so the R2 sweep physically cannot compute a value
    that depends on a coin it has not seen.

    w_pad / wnew_pad: optional (pad_seed_tensor, logical_offset,
    block_phys_start) for the W / Wnew block's ZK padding (P5) — the block
    pads as if its first row sat at logical_offset, under pad_seed, regardless
    of physical placement (a group starting at physical row r pads at
    logical_offset + (r - block_phys_start)). None → master seed at physical
    rows (identical padding, the standard case).

    w_owned (weight-split, shard_plan.py): a set of id(Variable) for the W
    variables THIS device folds/opens in this sweep; the others are left to
    the workers, whose partials the caller merges. None → all of them. Only
    the W emit is filtered — the witness, aux and quads are untouched, so the
    semantic sweep is exactly the single-device one."""
    import compute_fns as _cf
    import os as _os
    import resource as _res
    _dbg = _os.environ.get("LIGERO_STREAM_DBG")
    _dbg_every = max(1, len(groups) // 20)
    # Always-on lightweight progress cadence (~50 ticks/sweep) + sweep start
    # time, so a multi-hour sweep reports % done + ETA instead of going dark.
    _prog_every = max(1, len(groups) // 50)
    _t_sweep0 = time.time()
    for t in tables:
        if t.mult_var in tape.inputs:
            tape.inputs[t.mult_var].zero_()
    # The sound prover re-sweeps; LogUp multiplicity tables accumulate via
    # side_effects and MUST be re-zeroed each sweep, or counts stack (2x, 3x, …)
    # and the commitment diverges across rounds (the cause of the sound-path
    # REJECT). The loop above no-ops here — its `tables` mult_var instances don't
    # match the live tape.inputs keys — so reset by name as a robust backstop.
    for _v in tape.inputs:
        if getattr(_v, 'name', '').endswith('_mult'):
            tape.inputs[_v].zero_()
    Q_set = torch.tensor(Q_cols, dtype=torch.long, device="cuda") if Q_cols else None
    # Blinding is its own tree (layout B), committed/opened at the front rows.
    if p1_prefix is not None and p1_prefix.size(0) > 0:
        if merkle_blind is not None:
            merkle_blind.update(p1_prefix)
        if col_blind is not None:
            pslice = p1_prefix.index_select(1, Q_set)
            if isinstance(col_blind, ColumnSink):
                col_blind.write(0, pslice, Q_cols)
            else:
                for k, j in enumerate(Q_cols):
                    col_blind[j].append(pslice[:, k].clone())
    last_use = {}
    for i, (_c, ivars, _se) in enumerate(tape._deferred):
        for v in ivars:
            last_use[v] = i
    # Quad-ref hardening (linear-fold-unification.md): a quad operand's VALUE
    # must live to its DECLARING claim even if that claim's compute never reads
    # it — a structural guarantee replacing the per-claim-type convention that
    # quad operands ⊆ inputs ∪ own vars. Behavior-identical for the current
    # claim set; the outs-free below uses last_use (not the consumed set) so an
    # own-var operand still frees at its declaring claim rather than leaking.
    if stream_pk is not None:
        start2var = {v.row_start: v for v in (*p1_vars, *p2_vars, *p3_vars)}
        for qi_claim, fams in stream_pk.quad_fams.items():
            for _b0, fam in fams:
                for rs in (fam.x_row, fam.y_row, fam.z_row):
                    v = start2var.get(rs)
                    if v is not None and last_use.get(v, -1) < qi_claim:
                        last_use[v] = qi_claim
    live = dict(tape.inputs)
    fold = _cf.FoldRunner(tape._deferred, ch_by_index=ch0, want_aux=want_aux)
    use_pk = stream_pk is not None and (q_lin is not None or p_0 is not None)
    if use_pk:
        stream_pk.reset()
    def fetch(v):
        val = live[v]
        return val() if callable(val) else val
    def emit(vg, merkle, abs0, colbuf, pad=None):
        if not vg:
            return
        sink = colbuf if isinstance(colbuf, ColumnSink) else None
        res = _stream_phase(vg, live, cfg, master_seed=master_seed_t, abs_row_offset=abs0,
                            pad_seed=(None if pad is None else pad[0]),
                            pad_row_offset=(None if pad is None else pad[1]),
                            merkle_acc=merkle, q_irs_acc=q_irs, q_lin_acc=q_lin,
                            columns_at=(Q_cols if colbuf is not None else None),
                            column_sink=sink)
        if colbuf is not None and sink is None:
            for j in Q_cols:
                colbuf[j].append(res['opened_columns'][j])
    do_p1 = any(x is not None for x in (merkle_p1, q_irs, q_lin, col_p1))
    do_p2 = any(x is not None for x in (merkle_p2, q_irs, q_lin, col_p2))
    # Phase-3 rows exist only once their coin does, so a sweep without ch1
    # emits nothing for them however its accumulators are set.
    do_p3 = ch1 is not None and any(x is not None
                                    for x in (merkle_p3, q_irs, q_lin, col_p3))
    # Optional periodic allocator release during the sweep (LIGERO_SWEEP_GC=N):
    # diagnostic/workaround for sustained-churn faults on unified-memory GPUs.
    _sweep_gc = int(os.environ.get("LIGERO_SWEEP_GC", "0") or "0")
    for i in range(len(groups)):
        if _sweep_gc and i and i % _sweep_gc == 0:
            torch.cuda.empty_cache()
        claim = groups[i][0]
        input_vars, outs = (), {}
        if i < n_ops:
            _c, input_vars, side_effects = tape._deferred[i]
            if fold.is_fold(claim):
                input_data = {}
                with _phase('witness'):
                    outs = fold.finalize(claim, live)
            elif type(claim) in STREAMING_INPUT_CLAIMS:
                # Shard-streaming claim: hand it the raw live map (loaders
                # unresolved) and the round's op challenge, so one pass over a
                # shard can serve both the semantic output and the projection.
                input_data = {}
                with _phase('witness'):
                    outs = _cf.COMPUTE_FNS[type(claim)](
                        claim, live, ch0[i] if (want_aux and ch0) else None)
            else:
                input_data = {v: fetch(v) for v in input_vars}
                _spill = witness_cache.get('_spill') if witness_cache else False
                _disk = witness_cache.get('_disk') if witness_cache else False
                # disk-spill covers the FULL witness (every compute claim); host
                # spill / GPU cache cover only the expensive softmax/silu ops.
                _cacheable = (witness_cache is not None
                              and (_disk or type(claim).__name__ in _WITNESS_CACHE_TYPES))
                if _cacheable and i in witness_cache:
                    # Reuse the deterministic compute_fn output from sweep 1.
                    # GPU cache: clone so consumers can't mutate the cached copy.
                    # Spill: _spill_load returns a fresh device tensor (from the
                    # host copy) each time -> already independent, no clone.
                    if _disk:
                        outs = {v: _disk_spill_load(e, witness_cache) for v, e in witness_cache[i].items()}
                    elif _spill:
                        outs = {v: _spill_load(e) for v, e in witness_cache[i].items()}
                    else:
                        outs = {v: t.clone() for v, t in witness_cache[i].items()}
                else:
                    with _phase('witness'):
                        outs = _cf.COMPUTE_FNS[type(claim)](claim, input_data)
                    if _cacheable:
                        _nb = sum(t.numel() * t.element_size() for t in outs.values())
                        _ne = sum(t.numel() for t in outs.values())
                        _budget = witness_cache.get('_budget_bytes', 0)
                        _under_bytes = witness_cache.get('_bytes', 0) + _nb <= _budget
                        _under_elems = (_WITNESS_CACHE_MAX_ELEMS is None
                                        or witness_cache.get('_elems', 0) + _ne
                                        <= _WITNESS_CACHE_MAX_ELEMS)
                        if _under_bytes and _under_elems:
                            if _disk:
                                witness_cache[i] = {v: _disk_spill_store(t, witness_cache) for v, t in outs.items()}
                            elif _spill:
                                witness_cache[i] = {v: _spill_store(t) for v, t in outs.items()}
                            else:
                                witness_cache[i] = {v: t.clone() for v, t in outs.items()}
                            witness_cache['_bytes'] = witness_cache.get('_bytes', 0) + _nb
                            witness_cache['_elems'] = witness_cache.get('_elems', 0) + _ne
            for v, t in outs.items():
                live[v] = t
            if side_effects is not None:
                side_effects({**input_data, **outs})
        if want_aux:
            with _phase('aux'):
                if i < n_ops and fold.is_fold(claim):
                    live.update(fold.aux_finalize(claim, live, ch0[i]))
                else:
                    _wit = (live if type(claim) in STREAMING_INPUT_CLAIMS
                            else _LazyResolvingDict(live))
                    live.update(AUX_FNS[type(claim)](claim, _wit, ch0[i]))
                # Late (phase-3) aux: only once the R2 coin exists.
                if ch1 is not None and type(claim) in LATE_AUX_FNS:
                    _wit_l = (live if type(claim) in STREAMING_INPUT_CLAIMS
                              else _LazyResolvingDict(live))
                    live.update(LATE_AUX_FNS[type(claim)](claim, _wit_l, ch1[i]))
        own_quads = None
        if use_pk:
            with _phase('compile'):
                own_quads = stream_pk.compile_op(i)
        p1g, p2g, p3g = groups[i][1], groups[i][2], groups[i][3]
        if do_p1:
            # Split the claim's phase-1 vars — activations → p1 tree,
            # persistent weights → W tree, refreshed weights (linking proofs)
            # → Wnew tree. Each subset is row-contiguous (the claim's vars are
            # consecutive in op order, and _layout assigns each block's rows
            # in op order), so one emit each. All feed q_irs/q_lin (all
            # witness rows), so the tests are unchanged; only the merkle tree
            # + column buffer + padding differ per block.
            p1g_a = [v for v in p1g if not v.persistent]
            p1g_w = [v for v in p1g if v.persistent and not v.w_new]
            p1g_wn = [v for v in p1g if v.persistent and v.w_new]
            if p1g_a:
                emit(p1g_a, merkle_p1, p1g_a[0].row_start, col_p1)
            if p1g_w and w_owned is not None:
                # Weight-split: keep only this device's W variables. An
                # ownership cut can fall inside a claim's weight group, so
                # the kept subset is emitted as row-contiguous runs, each at
                # its own absolute row (never compacted across a gap).
                p1g_w = [v for v in p1g_w if id(v) in w_owned]
            for run_w in _row_contiguous_runs(p1g_w, cfg):
                # Pad translation: under w_pad this run's rows pad at the
                # block-local position shifted to the commitment's logical
                # offset, under the commitment's seed.
                wp = None if w_pad is None else (
                    w_pad[0], w_pad[1] + (run_w[0].row_start - w_pad[2]))
                emit(run_w, merkle_w, run_w[0].row_start, col_w, pad=wp)
            if p1g_wn:
                wp = None if wnew_pad is None else (
                    wnew_pad[0], wnew_pad[1] + (p1g_wn[0].row_start - wnew_pad[2]))
                emit(p1g_wn, merkle_wnew, p1g_wn[0].row_start, col_wnew, pad=wp)
        if want_aux and do_p2:
            emit(p2g, merkle_p2, p2g[0].row_start if p2g else m_p1_rows, col_p2)
        if want_aux and do_p3:
            emit(p3g, merkle_p3, p3g[0].row_start if p3g else m_p2_rows, col_p3)
        if want_aux and p_0 is not None and own_quads:
            with _phase('quad'):
                idx = torch.tensor([gi for gi, _ in own_quads], dtype=torch.long, device="cuda")
                p_0 = gl_add(p_0, compute_p_0_streaming(
                    p1_vars, [*p2_vars, *p3_vars], live, m_p1_rows,
                    r_quad.index_select(0, idx),
                    [qc for _, qc in own_quads], cfg, master_seed_t, maps=p_maps))
        if i < n_ops:                              # free: dead inputs / unread outputs
            for v in input_vars:
                if last_use.get(v) == i:
                    live.pop(v, None)
            if i != n_ops - 1:                     # keep last forward op's outputs (logits)
                for v in outs:
                    if last_use.get(v, i) <= i:    # never read later (quads incl.)
                        live.pop(v, None)
            # Fold absorption: after this claim's rows were emitted and its quads
            # fired, stream-consume outputs destined for a fold claim and free them
            # now instead of at the fold claim's index. The fold claim's bands already
            # exist in the upfront band index, so no early-compile is needed.
            for v in list(outs):
                fold.offer(v, live)
        if want_aux:
            for v in p2g:
                live.pop(v, None)
            if do_p3:
                for v in p3g:
                    live.pop(v, None)
        # Always-on progress line: op count, % done, elapsed, and a linear-rate
        # ETA. Just a print — correctness-neutral. The verbose memory dump below
        # stays behind LIGERO_STREAM_DBG.
        if i % _prog_every == 0 or i == len(groups) - 1:
            _done = i + 1
            _el = time.time() - _t_sweep0
            _eta = _el / _done * (len(groups) - _done)
            print(f"[sweep] op {_done}/{len(groups)} ({100 * _done / len(groups):.0f}%) "
                  f"elapsed={_el / 60:.1f}m eta={_eta / 60:.1f}m "
                  f"cur={torch.cuda.memory_allocated() / 1e9:.1f}GB", flush=True)
        if _dbg and (i % _dbg_every == 0 or i == len(groups) - 1):
            c1 = sum(t.numel() for L in (col_p1 or {}).values() for t in L)
            c2 = sum(t.numel() for L in (col_p2 or {}).values() for t in L)
            with open("/proc/self/statm") as _f:
                rss = int(_f.read().split()[1]) * 4096 / 1e9          # current VmRSS
            print(f"[sweep] op {i}/{len(groups)} cur={torch.cuda.memory_allocated()/1e9:.1f}GB "
                  f"peak={torch.cuda.max_memory_allocated()/1e9:.1f}GB "
                  f"res={torch.cuda.memory_reserved()/1e9:.1f}GB rss={rss:.1f}GB "
                  f"nlive={len(live)} col={8*(c1+c2)/1e9:.1f}GB "
                  f"bands={len(stream_pk.bands) if use_pk else 0}",
                  flush=True)
            if i == len(groups) - 1:
                lt = [(t.numel() * t.element_size(), str(v)[:48]) for v, t in live.items()
                      if torch.is_tensor(t)]
                tot = sum(s for s, _ in lt)
                print(f"[sweep] live tensors total={tot/1e9:.1f}GB over {len(lt)} tensors; top:",
                      flush=True)
                for sz, name in sorted(lt, reverse=True)[:12]:
                    print(f"   {sz/1e9:.3f}GB {name}", flush=True)
    tape.inputs.update(live)                        # mirror remainder (logits)
    return p_0


def _row_contiguous_runs(vars_list, cfg):
    """Split a row-ordered variable list into maximal runs whose rows are
    physically adjacent (v.row_start == prev.row_start + prev.n_rows). One
    _stream_phase call per run keeps every row at its true absolute index;
    _iter_message_chunks packs its list as ONE contiguous sequence, so a
    list with a gap would be compacted and every later row misplaced."""
    runs = []
    cur = []
    for v in vars_list:
        if cur and v.row_start != cur[-1].row_start + cur[-1].n_rows(cfg.ELL):
            runs.append(cur)
            cur = []
        cur.append(v)
    if cur:
        runs.append(cur)
    return runs


def _stream_setup(tape, cfg, zk_seed=None):
    """Shared setup for the streaming provers: layout, quad grouping, row-maps
    and blinding — everything that is CHALLENGE-INDEPENDENT.

    Nothing derived from a verifier coin is computed here: under sequential
    Fiat-Shamir each coin exists only after the prover message it follows, so
    the op challenges (s_op), the test-combiner vectors (s_comb) and the opened
    columns (s_col) are derived inside prove_streaming at their own round."""
    import os as _os
    def _m(tag):
        if not _os.environ.get("LIGERO_STREAM_DBG"):
            return
        with open("/proc/self/statm") as _f:
            rss = int(_f.read().split()[1]) * 4096 / 1e9
        print(f"  [setup] {tag}: rss={rss:.1f}GB gpu={torch.cuda.memory_allocated()/1e9:.1f}GB "
              f"peak={torch.cuda.max_memory_allocated()/1e9:.1f}GB", flush=True)
        torch.cuda.reset_peak_memory_stats()   # peak is per-step (since prev checkpoint)
    master_seed = zk_seed if zk_seed is not None else MASTER_SEED
    master_seed_t = _master_seed_to_cuda(master_seed)
    claims = _with_synthesized_settlements(tape.claims)
    _m("enter")
    (_all, p1_vars, p2_vars, p3_vars, m_p1_rows, m_p2_rows, m_total,
     weight_vars, m_w_rows, wnew_vars, m_wnew_rows) = _layout(claims, cfg)
    if _os.environ.get("LIGERO_LAYOUT_BREAKDOWN"):
        import sys
        from collections import defaultdict
        agg = defaultdict(lambda: [0, 0])          # claim type -> [rows, elements]
        seen_b = set()
        def _acct(v, tn):
            if isinstance(v, Variable) and id(v) not in seen_b:
                seen_b.add(id(v)); agg[tn][0] += v.n_rows(cfg.ELL); agg[tn][1] += v.length
        for _c in claims:
            _tn = type(_c).__name__
            _settle = isinstance(_c, TableSettlement)
            for _f in fields(_c):
                _v = getattr(_c, _f.name)
                if isinstance(_v, Variable):
                    _acct(_v, _tn)
                elif isinstance(_v, Table) and _settle:
                    _acct(_v.mult_var, _tn); _acct(_v.w_var, _tn)
                    for _z in _v.z_vars: _acct(_z, _tn)
                elif isinstance(_v, list):
                    for _it in _v:
                        if isinstance(_it, Variable): _acct(_it, _tn)
        W = m_total * cfg.ELL
        print(f"=== witness layout by claim type (m_total={m_total:,}, W={W:,} elements) ===", flush=True)
        for _t, (_r, _e) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
            print(f"  {_t:22s} rows={_r:>11,}  elements={_e:>15,}  {100*_e/W:5.1f}%", flush=True)
        sys.exit(0)
    groups = _claim_var_groups(claims, cfg)
    n_ops = len(tape.claims)
    # Phase-1 row map covers weights, then the linking proof's Wnew block, then
    # activations (layout B row order), so a quad operand in any of them lands
    # correctly; p2 map starts at the phase-1 end.
    # Phase 3 continues straight after phase 2, so one map covers both: p_0's
    # row split is "< m_p1_rows" vs "the rest", and the rest is p2 then p3 in
    # row order.
    p_maps = (_build_row_map(weight_vars + wnew_vars + p1_vars, cfg, NUM_BLINDING_ROWS),
              _build_row_map(p2_vars + p3_vars, cfg, m_p1_rows))
    _m("row_maps")
    u_irs_msg, u_lin_msg, u_quad_msg = _make_blinding_messages(cfg, master_seed)
    u_irs_polys_K, u_irs_codes = encode_messages(
        u_irs_msg.unsqueeze(0), cfg, master_seed=master_seed_t, row_offset=_BLIND_ROW_IRS)
    polys_2k, codes_2k = _encode_2k_blinding_rows(
        torch.stack([u_lin_msg, u_quad_msg], dim=0), cfg)
    return dict(
        master_seed_t=master_seed_t, claims=claims, p1_vars=p1_vars, p2_vars=p2_vars,
        p3_vars=p3_vars, weight_vars=weight_vars, m_w_rows=m_w_rows,
        wnew_vars=wnew_vars, m_p1_rows=m_p1_rows, m_p2_rows=m_p2_rows,
        m_total=m_total, groups=groups, n_ops=n_ops,
        p_maps=p_maps, tables=_collect_tables(claims),
        u_irs_poly=u_irs_polys_K[0], u_lin_poly=polys_2k[0], u_quad_poly=polys_2k[1],
        p1_prefix=torch.cat([u_irs_codes, codes_2k], dim=0),
        n_blind_total=NUM_BLINDING_ROWS,          # blinding is its own tree (layout B)
        n_p1_total=sum(v.n_rows(cfg.ELL) for v in p1_vars),   # activations only, no blinding
        n_w_total=m_w_rows, n_wnew_total=m_wnew_rows,
        n_p2_total=sum(v.n_rows(cfg.ELL) for v in p2_vars),
        n_p3_total=sum(v.n_rows(cfg.ELL) for v in p3_vars))


def _build_stream_packets(claims, ch0, ch1, cfg, n_ops):
    """The streaming compile: band index + quad families for the whole tape.

    Runs once, after BOTH coins that any claim's bands depend on (s_op and
    s_bind), so a late claim's phase-3 bands are emitted in the same pass as
    its early ones and constraint ids are independent of compile timing.
    Regular ops then stream their rows against this index; settlements are
    pre-indexed here because their sum-side packets land on rows emitted long
    before they would compile. The count pass also yields the global quad
    total for r_quad."""
    globals()['_SKIP_B_CHUNK'] = True
    stream_pk = _StreamingPackets(claims, ch0, cfg, n_ops, chs_late=ch1)
    globals()['_SKIP_B_CHUNK'] = False
    # Unified-memory hygiene: release the allocator's reserved high-water
    # before the long-lived phase begins.
    torch.cuda.empty_cache()
    return stream_pk


def _derive_test_challenges(s_comb, m_total, total_quads):
    """s_comb -> the IRS / linear / quadratic combiner vectors."""
    seed_u8 = torch.tensor(list(s_comb), dtype=torch.uint8, device="cuda")
    _lbl = lambda b: torch.tensor(list(b), dtype=torch.uint8, device="cuda")
    r_irs_t = challenge_vec(seed_u8, _lbl(b"irs"), m_total - NUM_BLINDING_ROWS)
    r_quad_t = challenge_vec(seed_u8, _lbl(b"quad"), total_quads)
    return r_irs_t, seed_u8, r_quad_t


def new_zk_seed() -> bytes:
    """A fresh secret seed for one proof's ZK padding and blinding rows.

    The old fixed MASTER_SEED is public (it is a constant in this file), and a
    verifier who knows it reconstructs every mask and subtracts it from the
    openings: the proof stays sound but stops being zero-knowledge. Online
    rows therefore pad under a per-proof secret; the persistent W block keeps
    padding under its own ENROLLMENT seed, which must be kept secret in the
    commitment handle, or its committed leaves would not reproduce."""
    import secrets
    return secrets.token_bytes(32)


def prove_streaming(tape, cfg, seed=None, weight_commitment=None, wnew_seed=None,
                    claims_bytes=None, zk_seed=None, shard_plan=None):
    """Streaming prover — the single production path (the sound four-round protocol).

    `shard_plan` (a shard_plan.ShardPlan, weight-split M1): split the ENROLLED
    W block's two per-proof passes — the test-polynomial fold and the column
    opening — across devices. This process is the coordinator (device 0): it
    runs every sweep as usual but folds/opens only its own runs of weight
    variables; the plan's workers fold/open theirs (shard_worker.py) and the
    partials are merged here — exact field sums before the single inverse NTT,
    and column pieces scattered by absolute row — so the proof is
    BYTE-IDENTICAL to the unsharded one and the verifier is unchanged.
    Requires `weight_commitment` (the split is defined on the enrolled block,
    which does no commitment work in R1-R3).

    `weight_commitment` (a WeightCommitment, P3): reference a pre-committed W
    tree instead of committing it here. The R1 weight commit and R4 weight
    rebuild are skipped; root_w and the opening paths come from the persisted
    commitment, while the opened weight columns are still re-extracted (their
    codewords reproduce the persisted leaves because R_W is context-independent
    under layout B). Saves the weight column-hashing (the compute-bound D·W
    term, §8) across every proof that shares one model.

    `wnew_seed` (bytes, P5): required iff the tape has a Wnew block
    (persistent="new" vars — a linking proof). The Wnew tree pads under
    (wnew_seed, logical NUM_BLINDING_ROWS) so its root reproduces the
    refreshed commitment R_W′ = commit_weights(seed=wnew_seed).

    FOUR streaming sweeps, the witness regenerated each round, as the staged
    interactive protocol requires (commit before the challenges that determine
    what is revealed).

    Each coin is SEQUENTIAL Fiat-Shamir over the transcript so far
    (analysis/routed-projected-protocol.md):

        R1 (blind|W|Wnew|p1 roots) -> s_op
        R2 (p2 root)               -> s_comb
        test polynomials           -> s_col -> openings

    so no coin can be derived before the prover message it follows. `seed` no
    longer determines any coin — it is accepted for call compatibility and
    ignored. The derived seeds travel on the returned Proof (`proof.seeds`)
    together with the statement digest they are bound to; the Rust verifier
    recomputes all of them and trusts none of them from the wire.

    `claims_bytes`: the exact canonical claim-set JSON bytes that will be
    written into the proof file. Pass the driver's copy so it is built once;
    None recomputes it here. Returns a full Proof; Rust-verified.
    """
    torch.cuda.reset_peak_memory_stats()
    if _PHASE_ON:
        _PHASE_TIMES.clear()
    for _hook in PROVE_START_HOOKS:
        _hook()
    # Fresh secret padding/blinding entropy per proof unless a caller pins it
    # (diff-tests that compare two proofs byte-for-byte do pin it).
    s = _stream_setup(tape, cfg, zk_seed=(zk_seed or new_zk_seed()))
    if claims_bytes is None:
        claims_bytes = pr.claims_canonical_bytes(tape.claims, cfg)
    # The row-block layout is part of the statement, not something the proof
    # gets to choose: it follows from the tape (persistent weights? a linking
    # block? a late-stage claim?) and is fixed before R1.
    blocks = (["blind"] + (["w"] if s['n_w_total'] else [])
              + (["wnew"] if s['n_wnew_total'] else []) + ["p1", "p2"]
              + (["p3"] if s['n_p3_total'] else []))
    stmt_digest = pr.statement_digest(claims_bytes, blocks)
    # The streaming sweeps recompile each claim and DISCARD its b_chunk (RHS) —
    # only row packets + quads are consumed (_compile_at: `..., _b = COMPILE_FNS`).
    # Keep _SKIP_B_CHUNK on so the sweep doesn't waste an O(T*V) dense RHS build
    # per claim (the V=202048 hidden-routing claim's b_chunk was ~89M entries in
    # Python — a ~40-min stall before op 0). Value-neutral: the RHS is unused.
    globals()['_SKIP_B_CHUNK'] = True
    # Coins and everything derived from them are filled in at their own round
    # below; nothing challenge-dependent may exist before R1 is committed.
    ch0 = ch1 = stream_pk = Q_cols = None
    q_irs_acc = q_lin_acc = None
    col_p1 = col_p2 = col_p3 = None

    w_pad = None      # (pad_seed_t, logical_offset, block_phys_start) for the W /
    wnew_pad = None   # Wnew block — set below (P5 refresh / linking support).

    # Witness cache shared across the 4 sweeps: populated on sweep 1, reused on
    # sweeps 2-4 for the deterministic softmax/silu compute_fn outputs. None
    # disables (LIGERO_WITNESS_CACHE=0). Fresh per prove call -> no staleness.
    # Two backing stores, same soundness: default caches in GPU memory; opt-in
    # LIGERO_WITNESS_SPILL=1 caches in pinned HOST memory (larger budget, covers
    # witness > GPU) and re-reads over PCIe instead of recomputing.
    if _WITNESS_SPILL_DISK:
        witness_cache = _disk_spill_open({'_spill': True, '_disk': True})
    elif _WITNESS_SPILL_ON:
        witness_cache = {'_budget_bytes': _host_spill_budget_bytes(), '_spill': True}
    elif _WITNESS_CACHE_ON:
        witness_cache = {'_budget_bytes': _witness_cache_budget_bytes()}
    else:
        witness_cache = None

    def sweep(**kw):
        return _stream_sweep(tape, cfg, s['master_seed_t'], s['groups'], s['n_ops'],
                             s['p1_vars'], s['p2_vars'], s['m_p1_rows'], s['tables'],
                             ch0, ch1=ch1, p3_vars=s['p3_vars'],
                             m_p2_rows=s['m_p2_rows'], w_pad=w_pad, wnew_pad=wnew_pad,
                             witness_cache=witness_cache, **kw)

    def _p0_zero():
        return torch.zeros(2 * cfg.K_DEG - 1, dtype=torch.uint64, device="cuda")

    # FOUR blocks (analysis/persistent-weights.md, layout B): blinding tree,
    # W tree (persistent weights), p1 tree (activations), p2 tree (aux). All of
    # blind/W/p1 are phase-1 (committed in R1); p2 in R2. Roots, trees, and
    # opening paths all come from the R1/R2 commits (the trees are a few MB
    # and are kept); R4 only re-extracts the challenged columns, whose
    # identity is not known until the round-3 column challenge. R_W is
    # context-independent (W at the fixed offset), so it matches
    # commit_weights.
    has_w = bool(s['n_w_total'])
    # P3: reference a persisted W commitment — skip the R1 weight commit + R4
    # weight rebuild, take root_w and the opening paths from it. Guard that it
    # is for THIS model's W block (same row count and codeword length).
    wc = weight_commitment if has_w else None
    # Weight-split plan (M1): validated here, before R1, so a malformed plan
    # (gap, overlap, wrong count) fails before any sweep runs. The split is
    # defined on the ENROLLED block, whose R1-R3 work is nil.
    plan = None
    if shard_plan is not None:
        import shard_plan as _sp
        import shard_worker as _sw
        assert wc is not None, (
            "shard_plan needs weight_commitment: the split is defined on the "
            "enrolled W block")
        plan = _sp.as_plan(shard_plan, len(s['weight_vars']))

    # The quad-placement guards below need the compiled quad families, which
    # exist only after s_op (the R1 coin) — so they are QUEUED here and run the
    # moment the compile lands, still before any work that depends on them.
    _pending_quad_checks = []

    def _assert_no_quads_in(lo, hi, what):
        _pending_quad_checks.append((lo, hi, what))

    def _run_quad_checks(stream_pk):
        for lo, hi, what in _pending_quad_checks:
            _assert_no_quads_now(stream_pk, lo, hi, what)

    def _assert_no_quads_now(stream_pk, lo, hi, what):
        # Completeness guard: p_0's sparse re-encode (compute_p_0_streaming)
        # pads by PHYSICAL row under MASTER_SEED, so a quad constraint touching
        # a row padded under a different (seed, logical offset) would make p_0
        # inconsistent with the commitment and the proof would REJECT.
        # Refresh/linking tapes are linear in the weight blocks; fail loudly if
        # not. Band anchors are exact at variable granularity (a band lives
        # inside one variable, and a variable is entirely in or out of a block).
        for fams in stream_pk.quad_fams.values():
            for _b0, fam in fams:
                for rs in (fam.x_row, fam.y_row, fam.z_row):
                    assert not (lo <= rs < hi), (
                        f"{what} with a quad constraint on its rows (band row "
                        f"{rs}): p_0 would re-encode under the master seed at "
                        "physical rows and the proof would not verify")

    if wc is not None:
        assert wc.m_w == s['n_w_total'] and wc.n_lig == cfg.N_LIG, (
            f"weight commitment mismatch: m_w {wc.m_w} vs {s['n_w_total']}, "
            f"N_LIG {wc.n_lig} vs {cfg.N_LIG}")
        # P5: pad the W block under the COMMITMENT's seed at the block's
        # logical offset, so the re-extracted columns reproduce the persisted
        # leaves even when the commitment was REFRESHED under a fresh seed.
        # Identity when the seed is the default MASTER_SEED (W physically sits
        # at NUM_BLINDING_ROWS, same as its logical offset).
        w_pad = (_master_seed_to_cuda(wc.master_seed), NUM_BLINDING_ROWS,
                 NUM_BLINDING_ROWS)
        if wc.master_seed != MASTER_SEED:
            _assert_no_quads_in(NUM_BLINDING_ROWS,
                                NUM_BLINDING_ROWS + s['n_w_total'],
                                "refreshed-seed weight commitment")
    # P5 linking proof: the Wnew block pads under (wnew_seed, logical
    # NUM_BLINDING_ROWS) — as if it led the witness, the position the refresh
    # commit_weights(seed=wnew_seed) used — so its root reproduces R_W′
    # despite physically sitting after W.
    has_wnew = bool(s['n_wnew_total'])
    if has_wnew:
        assert wnew_seed is not None, (
            "tape has a Wnew block (persistent='new' vars) — pass wnew_seed, "
            "the refresh seed the new commitment was made under")
        wnew_phys = NUM_BLINDING_ROWS + s['n_w_total']
        wnew_pad = (_master_seed_to_cuda(wnew_seed), NUM_BLINDING_ROWS, wnew_phys)
        _assert_no_quads_in(wnew_phys, wnew_phys + s['n_wnew_total'],
                            "linking proof's Wnew block")
    _acc = lambda n: _make_merkle_acc(cfg.N_LIG, n) if n else None
    merkle_blind = _acc(s['n_blind_total'])                              # R1: commit phase-1
    merkle_w = None if wc is not None else _acc(s['n_w_total'])          #   (W referenced → skip)
    merkle_wnew = _acc(s['n_wnew_total'])                                #   (linking proofs)
    merkle_p1 = _acc(s['n_p1_total'])
    sweep(want_aux=False, merkle_blind=merkle_blind, merkle_w=merkle_w,
          merkle_wnew=merkle_wnew, merkle_p1=merkle_p1,
          p1_prefix=s['p1_prefix'])
    art_blind = _finalize_merkle_artifact(merkle_blind)
    art_w = _finalize_merkle_artifact(merkle_w) if merkle_w is not None else None
    art_wnew = _finalize_merkle_artifact(merkle_wnew) if merkle_wnew is not None else None
    art_p1 = _finalize_merkle_artifact(merkle_p1) if merkle_p1 is not None else None
    # ---- coin after R1: the op challenges (rho and friends) --------------
    # root_w of a REFERENCED persistent commitment is part of R1 exactly as an
    # in-proof weight tree would be, so the same statement/model binds either
    # way. Block order is framed too: a proof cannot re-label its blocks.
    root_w_r1 = (wc.root if wc is not None else (art_w.root if has_w else None))
    blocks_r1 = (["blind"] + (["w"] if has_w else []) + (["wnew"] if has_wnew else [])
                 + ["p1"])
    roots_r1 = ([art_blind.root] + ([root_w_r1] if has_w else [])
                + ([art_wnew.root] if has_wnew else []) + [art_p1.root])
    s_op = pr.fs_s_op(stmt_digest, blocks_r1, roots_r1)
    ch0 = _sample_chs(s['claims'], s_op)
    merkle_p2 = _make_merkle_acc(cfg.N_LIG, s['n_p2_total'])              # R2: commit phase-2
    sweep(want_aux=True, merkle_p2=merkle_p2)
    art_p2 = _finalize_merkle_artifact(merkle_p2)
    # ---- coin after R2: the late (phase-3) challenges --------------------
    s_bind = pr.fs_s_bind(s_op, art_p2.root)
    has_p3 = bool(s['n_p3_total'])
    ch1 = _sample_late_chs(s['claims'], s_bind)
    if has_p3:                                                            # R3: commit phase-3
        merkle_p3 = _acc(s['n_p3_total'])
        sweep(want_aux=True, merkle_p3=merkle_p3)
        art_p3 = _finalize_merkle_artifact(merkle_p3)
        root_p3 = art_p3.root
    else:
        # No late-stage claim on this tape: R3 is the empty message, and the
        # transcript still frames it so a proof cannot silently drop a round.
        art_p3, root_p3 = None, pr.EMPTY_COMMIT_ROOT
    # Compile once, now that every coin a band can depend on exists.
    stream_pk = _build_stream_packets(s['claims'], ch0, ch1, cfg, s['n_ops'])
    _run_quad_checks(stream_pk)
    # ---- coin after R3: the test combiners -------------------------------
    s_comb = pr.fs_s_comb(s_bind, root_p3)
    r_irs_t, r_lin_seed, r_quad_t = _derive_test_challenges(
        s_comb, s['m_total'], stream_pk.total_quads)
    q_irs_acc = QIrsAccumulator(r_irs_t, cfg)
    q_lin_acc = QLinAccumulator(r_lin_seed, stream_pk, cfg)
    # Weight-split (M1): the coordinator folds only its own run of W
    # variables in this sweep; the workers' partial folds are merged below.
    w_owned = None if plan is None else plan.owned_ids(0, "fold", s['weight_vars'])
    p_0 = sweep(want_aux=True, q_irs=q_irs_acc, q_lin=q_lin_acc,          # R3: q-polys + p_0
                p_0=_p0_zero(), stream_pk=stream_pk,
                r_quad=r_quad_t, p_maps=s['p_maps'], w_owned=w_owned)
    if plan is not None:
        for dev, lo, hi in plan.worker_runs("fold"):
            part = _sw.fold_run(s['weight_vars'], tape.inputs, cfg,
                                s['master_seed_t'], w_pad, lo, hi,
                                r_irs_t, r_lin_seed, stream_pk,
                                device=plan.device_of(dev))
            q_irs_acc.merge(part["q_irs"])
            q_lin_acc.merge(q_eval=part["q_eval"], q_coeff=part["q_coeff"])
    # ---- coin after the test polynomials: the opened columns -------------
    # Mixed with the blinding rows FIRST: s_col must hash the polynomials the
    # verifier actually receives, not the pre-blinding accumulators.
    q_irs, q_lin, p_0 = _mix_blinding_into_tests(
        q_irs_acc.finalize(), q_lin_acc.finalize(), p_0,
        s['u_irs_poly'], s['u_lin_poly'], s['u_quad_poly'], cfg)
    s_col = pr.fs_s_col(s_comb, _u64_le_bytes(q_irs), _u64_le_bytes(q_lin),
                        _u64_le_bytes(p_0))
    Q_cols = list(pr.random_columns(s_col, cfg))
    if wc is not None:
        # Fail before a single weight column is extracted.
        wc.check_openings(cfg, Q_cols)
    # One pre-sized HOST buffer per column per block: the openings are written
    # through as they are produced, so nothing accumulates on the GPU and
    # there is no final torch.cat (which briefly doubled tens of GB).
    _w_base = NUM_BLINDING_ROWS
    _wnew_base = _w_base + s['n_w_total']
    _p1_base = _wnew_base + s['n_wnew_total']
    col_blind = ColumnSink(s['n_blind_total'], Q_cols, 0)
    col_w = ColumnSink(s['n_w_total'], Q_cols, _w_base)
    col_wnew = ColumnSink(s['n_wnew_total'], Q_cols, _wnew_base)
    col_p1 = ColumnSink(s['n_p1_total'], Q_cols, _p1_base)
    col_p2 = ColumnSink(s['n_p2_total'], Q_cols, s['m_p1_rows'])
    col_p3 = ColumnSink(s['n_p3_total'], Q_cols, s['m_p2_rows'])
    # Opening round: columns only. Roots, trees and opening paths all come
    # from the R1/R2/R3 commits (the trees are a few MB and are kept); this
    # sweep only re-extracts the challenged columns, whose identity is not
    # known until s_col.
    w_owned = None if plan is None else plan.owned_ids(0, "open", s['weight_vars'])
    sweep(want_aux=True, col_blind=col_blind, col_w=col_w,
          col_wnew=col_wnew, col_p1=col_p1, col_p2=col_p2, col_p3=col_p3,
          Q_cols=Q_cols, p1_prefix=s['p1_prefix'], w_owned=w_owned)
    if plan is not None:
        # Workers open their runs; the pieces are scattered into the W sink
        # by absolute row, and finish() asserts exact, non-overlapping
        # coverage of the block.
        for dev, lo, hi in plan.worker_runs("open"):
            piece = _sw.open_run(s['weight_vars'], tape.inputs, cfg,
                                 s['master_seed_t'], w_pad, lo, hi, Q_cols,
                                 device=plan.device_of(dev))
            if piece is not None:
                col_w.write_host(piece["abs_row"], piece["cols"])

    def _opened(colbuf):
        if isinstance(colbuf, ColumnSink):
            return colbuf.finish()
        return {j: (torch.cat(colbuf[j]) if colbuf[j]
                    else torch.empty(0, dtype=torch.uint64, device="cuda")) for j in Q_cols}
    def _paths(art):
        return {j: merkle_path(art.levels, j) for j in Q_cols} if art is not None else {}
    opened_blind = _opened(col_blind)
    opened_w     = _opened(col_w) if has_w else {}
    opened_wnew  = _opened(col_wnew) if has_wnew else {}
    opened_p1    = _opened(col_p1)
    opened_p2    = _opened(col_p2)
    opened_p3    = _opened(col_p3) if has_p3 else {}
    # W block root + opening paths: from the referenced commitment (P3) or the
    # in-proof W tree kept from R1. Either way opened_w is the re-extracted
    # weight columns, which reproduce the committed leaves (context-independent
    # R_W; under a refreshed seed, the P5 padding rule).
    if wc is not None:
        root_w = wc.root
        paths_w = {j: merkle_path(wc.levels, j) for j in Q_cols}
        w_leaf_art = CommitArtifact(root=wc.root, levels=wc.levels,
                                    column_hashes=wc.levels[0], matrix=None)
    else:
        root_w = art_w.root if has_w else None
        paths_w = _paths(art_w) if has_w else {}
        w_leaf_art = art_w
    # Cross-round consistency self-check, replacing the old full opening-round
    # tree rebuild + root comparison: the re-extracted columns must hash to
    # the leaves committed in R1/R2/R3 (for a referenced W block, to the
    # persisted commitment's leaves). Same witness-regeneration-drift
    # detection on exactly the data being served, now covering every block
    # including phases 2 and 3, at T column hashes per block instead of N_LIG.
    # NOTE: q_irs/q_lin/p_0 were already blinding-mixed BEFORE s_col above —
    # the s_col coin must hash the polynomials the verifier receives.
    repro = (_opened_columns_match_leaves(opened_blind, Q_cols, art_blind, s['n_blind_total'])
             and (not has_w or _opened_columns_match_leaves(opened_w, Q_cols, w_leaf_art, s['n_w_total']))
             and (not has_wnew or _opened_columns_match_leaves(opened_wnew, Q_cols, art_wnew, s['n_wnew_total']))
             and _opened_columns_match_leaves(opened_p1, Q_cols, art_p1, s['n_p1_total'])
             and _opened_columns_match_leaves(opened_p2, Q_cols, art_p2, s['n_p2_total'])
             and (not has_p3 or _opened_columns_match_leaves(opened_p3, Q_cols, art_p3, s['n_p3_total'])))
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"  [stream-sound] {5 if has_p3 else 4} rounds done; opened columns "
          f"match committed leaves: {repro}; W-block rows {s['n_w_total']}; "
          f"W-ref {wc is not None}; Wnew rows {s['n_wnew_total']}; "
          f"p3 rows {s['n_p3_total']}; peak {peak:.2f} GB", flush=True)
    if _PHASE_ON and _PHASE_TIMES:
        _tot = sum(_PHASE_TIMES.values())
        print("  [phase] prove-time breakdown (cuda-synced buckets; shares, not "
              "absolute — syncs remove overlap):", flush=True)
        for _k, _v in sorted(_PHASE_TIMES.items(), key=lambda kv: -kv[1]):
            print(f"    {_k:10s} {_v:8.1f}s  {100 * _v / _tot:5.1f}%", flush=True)
        print(f"    {'BUCKETED':10s} {_tot:8.1f}s  (vs total prove wall-clock; "
              f"remainder = setup + un-bucketed)", flush=True)
    _ephase_report()
    # Reset the sweep's b_chunk skip so a subsequent in-process verify() (which
    # DOES need the public RHS — e.g. the reveal pin) recompiles it. Leaking
    # True here silently zeroed every nonzero-RHS constraint in verify.
    globals()['_SKIP_B_CHUNK'] = False
    _disk_spill_close(witness_cache)          # delete the per-proof disk-spill file
    assert blocks == (["blind"] + (["w"] if has_w else [])
                      + (["wnew"] if has_wnew else []) + ["p1", "p2"]
                      + (["p3"] if has_p3 else [])), (
        "row-block layout diverged from the one the statement digest fixed")
    return Proof(
        q_irs=q_irs, q_lin=q_lin, p_0=p_0, blocks=blocks,
        seeds={"s_op": s_op, "s_bind": s_bind, "s_comb": s_comb, "s_col": s_col},
        statement_digest=stmt_digest, claims_bytes=claims_bytes, Q_cols=Q_cols,
        root_blind=art_blind.root, opened_blind=opened_blind, paths_blind=_paths(art_blind),
        root_w=root_w,
        opened_w=opened_w, paths_w=paths_w,
        root_wnew=(art_wnew.root if has_wnew else None),
        opened_wnew=opened_wnew,
        paths_wnew=(_paths(art_wnew) if has_wnew else {}),
        root_p1=art_p1.root, opened_p1=opened_p1, paths_p1=_paths(art_p1),
        root_p2=art_p2.root, opened_p2=opened_p2, paths_p2=_paths(art_p2),
        root_p3=(root_p3 if has_p3 else None),
        opened_p3=(opened_p3 if has_p3 else {}),
        paths_p3=(_paths(art_p3) if has_p3 else {}))


class _PhaseLogger:
    """Phase-by-phase wall-time + GPU memory logger for prove/verify.

    Output looks like:
        [prove +12.3s] [alloc=4.2G reserved=8.1G peak=6.5G] Round 1 done — m=4523
    where alloc = currently-allocated GPU memory by live tensors,
    reserved = total reserved by PyTorch caching allocator, and peak =
    max allocated since last reset_peak()."""

    def __init__(self, label: str, verbose: bool):
        self.label = label
        self.verbose = verbose
        self.t0 = time.time()

    def log(self, msg: str):
        if not self.verbose:
            return
        elapsed = time.time() - self.t0
        try:
            alloc    = torch.cuda.memory_allocated()     / 1e9
            reserved = torch.cuda.memory_reserved()      / 1e9
            peak     = torch.cuda.max_memory_allocated() / 1e9
            mem = f"[alloc={alloc:.1f}G reserved={reserved:.1f}G peak={peak:.1f}G]"
        except Exception:
            mem = ""
        print(f"  [{self.label} +{elapsed:6.1f}s] {mem} {msg}", flush=True)

    def reset_peak(self):
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass


# The flat unit/negative-test prover prove() and its _sample_test_challenges
# helper moved to tests/test_prover.py (test harness). core.py is the production
# prover only (prove_streaming) — see Tape.prove.
