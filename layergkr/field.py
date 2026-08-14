"""Goldilocks field ops that COUNT themselves.

Same field and same semantics as `prover/protocol.py` -- these are not a second
implementation, they are the same operations with `counters.charge` attached, so
a proof run reports exactly how many multiplications, additions, inversions and
hashed bytes it performed. That count is what the cost model predicts and what
`bench/run_toy.py` checks it against.

Counting convention, chosen so nothing is double-charged or silently free:
  * `mul`/`add` are single field operations.
  * `inv` charges one inversion AND the ~2*63 multiplications Fermat's little
    theorem actually costs (square-and-multiply over a 64-bit exponent), because
    on the real prover an inversion is not free and the model must see it.
  * batched helpers (`dot`, `vec_add`, ...) charge in bulk rather than per
    element call, which is both faster and closer to how the GPU prover works.
"""
from typing import Iterable, List, Sequence

from prover.protocol import P

from .counters import charge

# Cost of one Fermat inversion in multiplications: square-and-multiply over a
# 64-bit exponent. Measured against the reference implementation's behaviour,
# not guessed: exponent P-2 has 64 bits, 32 of them set.
INV_MULS = 95


def add(a: int, b: int) -> int:
    charge(add=1)
    return (a + b) % P


def sub(a: int, b: int) -> int:
    charge(add=1)
    return (a - b) % P


def mul(a: int, b: int) -> int:
    charge(mul=1)
    return (a * b) % P


def inv(a: int) -> int:
    charge(inv=1, mul=INV_MULS)
    return pow(a % P, P - 2, P)


def neg(a: int) -> int:
    charge(add=1)
    return (-a) % P


# ── batched helpers: charge in bulk ──────────────────────────────────────────
def dot(xs: Sequence[int], ys: Sequence[int]) -> int:
    n = len(xs)
    charge(mul=n, add=max(n - 1, 0))
    acc = 0
    for x, y in zip(xs, ys):
        acc += x * y
    return acc % P


def vec_scale(xs: Sequence[int], k: int) -> List[int]:
    charge(mul=len(xs))
    return [(x * k) % P for x in xs]


def vec_add(xs: Sequence[int], ys: Sequence[int]) -> List[int]:
    charge(add=len(xs))
    return [(x + y) % P for x, y in zip(xs, ys)]


def vec_sub(xs: Sequence[int], ys: Sequence[int]) -> List[int]:
    charge(add=len(xs))
    return [(x - y) % P for x, y in zip(xs, ys)]


def hadamard(xs: Sequence[int], ys: Sequence[int]) -> List[int]:
    charge(mul=len(xs))
    return [(x * y) % P for x, y in zip(xs, ys)]


def batch_inv(xs: Sequence[int]) -> List[int]:
    """Montgomery's trick: one inversion for the whole batch, 3n muls. This is
    what makes LogUp's reciprocals affordable, and the model counts it this way
    rather than as n independent inversions."""
    n = len(xs)
    if n == 0:
        return []
    pref = [1] * (n + 1)
    for i, x in enumerate(xs):
        pref[i + 1] = pref[i] * x % P
    charge(mul=n, red_op=n)
    acc = inv(pref[n])                       # the single real inversion
    out = [0] * n
    for i in range(n - 1, -1, -1):
        out[i] = acc * pref[i] % P
        acc = acc * xs[i] % P
    charge(mul=2 * n, red_op=2 * n)
    return out


def horner(coeffs: Sequence[int], x: int) -> int:
    n = len(coeffs)
    charge(mul=max(n - 1, 0), add=max(n - 1, 0))
    acc = 0
    for c in reversed(coeffs):
        acc = (acc * x + c) % P
    return acc


def sum_all(xs: Iterable[int]) -> int:
    xs = list(xs)
    charge(add=max(len(xs) - 1, 0))
    return sum(xs) % P
