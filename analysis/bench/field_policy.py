"""Per-proof / per-op FIELD POLICY — the decision layer for running the prover
in a smaller field (BabyBear) where it is provably safe, and only falling back
to Goldilocks where an operation's worst-case accumulator would overflow.

IMPORTANT SCOPE / HONESTY: the prover's field arithmetic is hardcoded to
Goldilocks across Python (protocol.py roots pow(7,(P-1)/K,P)), the CUDA kernels
(kquant_cuda.py `#define GL_P`), and the Rust verifier (field.rs `P`). A real
BabyBear PROVE+ACCEPT is a multi-file port (new modulus, generator, two-adic
roots (BabyBear is 2^27-adic vs Goldilocks 2^32), 31-bit packing) — NOT done
here. This module is the PREREQUISITE and the brain: it decides, from SOUND
worst-case bounds, whether a given proof (or which of its ops) can use BabyBear
at all, and projects the payoff — so we know if that port is even worth doing.

Soundness of the bound: the field must hold every partial sum of the fixed-point
accumulator. For a dot product of length k with operands bounded by |value| <=
s*R (scale s, magnitude clip R enforced by the prover's range checks):
  - generic (both operands adversarial, e.g. attention scores Q·K^T):
        max|acc| <= k * (s*R)^2                 -> the BINDING case
  - weight matmul (one operand is a COMMITTED, fixed weight W): the adversary
    controls only the activation, so the true worst case is the tight
        max|acc| <= s_a*s_b * R * ||W_col||_1   (<< k*(sR)^2)
bits = floor(log2(max|acc|)) + 1  (the +1 is the sign bit).
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field as _dc_field
from enum import Enum


@dataclass(frozen=True)
class FieldSpec:
    name: str
    p: int
    two_adic: int        # max NTT size = 2^two_adic
    @property
    def bits(self) -> int:            # usable signed capacity (p ~ 2^bits)
        return int(math.log2(self.p))
    @property
    def word_bytes(self) -> int:      # storage word: 4B for <=32-bit, 8B else
        return 4 if self.bits <= 32 else 8


BABYBEAR   = FieldSpec("BabyBear",   (1 << 31) - (1 << 27) + 1, two_adic=27)
GOLDILOCKS = FieldSpec("Goldilocks", (1 << 64) - (1 << 32) + 1, two_adic=32)
FIELDS = [BABYBEAR, GOLDILOCKS]   # smallest first


class OpKind(Enum):
    WEIGHT_MATMUL = "weight_matmul"      # act · committed weight  (tight bound)
    ATTENTION_SCORE = "attention_score"  # act · act (Q·K^T)       (generic bound)
    ELEMENTWISE = "elementwise"          # silu/rmsnorm/add/rope   (small)


@dataclass(frozen=True)
class Op:
    name: str
    kind: OpKind
    k: int                 # contraction / dot-product length
    s_a: int = 4096        # 1st operand scale (2^12)
    s_b: int = 4096        # 2nd operand scale
    R: float = 32.0        # 1st operand magnitude clip
    R2: float | None = None  # 2nd operand magnitude (default = R). For P·V the
                             # softmax operand is <=1, so set R=1.0 there.
    w_l1: float | None = None   # max committed weight column L1 norm (tight path)

    def acc_bits(self) -> float:
        """Sound worst-case accumulator bit-width (max over partial sums)."""
        R2 = self.R if self.R2 is None else self.R2
        if self.kind is OpKind.ELEMENTWISE:
            # no length-k accumulation; a single scaled value (+ headroom)
            v = self.s_a * self.R
            return math.log2(max(v, 1)) + 1
        if self.kind is OpKind.WEIGHT_MATMUL and self.w_l1 is not None:
            # weight is committed/fixed: only the activation is adversarial ->
            # tight  s_a*s_b * R_act * ||W_col||_1  (<< k*(sR)^2)
            v = self.s_a * self.s_b * self.R * self.w_l1
            return math.log2(max(v, 1)) + 1
        # generic: both operands adversarial, bounded by s*R over k terms
        v = self.k * (self.s_a * self.R) * (self.s_b * R2)
        return math.log2(max(v, 1)) + 1

    def min_field(self) -> FieldSpec:
        b = self.acc_bits()
        for f in FIELDS:                 # smallest field that holds it
            if b < f.bits and self.k * 4 < (1 << f.two_adic):
                return f
        return GOLDILOCKS


# ---------------- policies -----------------------------------------------
class Policy(Enum):
    GOLDILOCKS = "goldilocks"        # baseline: everything in Goldilocks
    BABYBEAR_FORCE = "babybear"      # force BabyBear; flag overflow ops (unsound if any)
    ADAPTIVE_PROOF = "adaptive-proof"  # one field for the whole proof: smallest safe
    ADAPTIVE_OP = "adaptive-op"      # mixed: each op in its own minimal safe field


def assign(ops: list[Op], policy: Policy) -> dict[str, FieldSpec]:
    if policy is Policy.GOLDILOCKS:
        return {o.name: GOLDILOCKS for o in ops}
    if policy is Policy.BABYBEAR_FORCE:
        return {o.name: BABYBEAR for o in ops}
    if policy is Policy.ADAPTIVE_OP:
        return {o.name: o.min_field() for o in ops}
    # ADAPTIVE_PROOF: BabyBear iff every op fits it, else Goldilocks
    all_fit = all(o.min_field() is BABYBEAR for o in ops)
    f = BABYBEAR if all_fit else GOLDILOCKS
    return {o.name: f for o in ops}


def overflow_ops(ops: list[Op], f: FieldSpec) -> list[Op]:
    return [o for o in ops if o.acc_bits() >= f.bits]
