"""
Polynomial primitives over Goldilocks — pure-Python reference.

Five helpers extracted from the U.0 unified-Ligero prototype:

  - lagrange_eval(xs, ys, x_target):
        Evaluate at `x_target` the unique polynomial of degree < len(xs)
        that passes through {(xs[i], ys[i])}.

  - lagrange_poly_coeffs(xs, ys):
        Recover the coefficient list [c_0, c_1, ..., c_{d-1}] of that
        same polynomial (degree < d = len(xs)).

  - poly_eval(coeffs, x):
        Horner evaluation of a polynomial in coefficient form at x.

  - poly_mul(a, b):
        Convolution: returns the coefficients of a(X) · b(X).

  - poly_add(a, b):
        Sum of two polynomials in coefficient form (length-adapting).

All arithmetic is mod P = 2^64 − 2^32 + 1 (Goldilocks). The helpers
import `add`, `sub`, `mul`, `inv`, `P` from `goldilocks_ref` so this
module has no field-arithmetic surface of its own.

Used as the verifier-side polynomial machinery: recovering polynomial
coefficients from sampled values (Lagrange), evaluating committed
polynomials at queried column points (Horner). The CUDA prover does
the same operations via NTT in `cuda/ntt.cuh`; this file is the
slow-but-obvious oracle for cross-validation.
"""

from typing import List

from goldilocks_ref import P, add, sub, mul, inv


def _neg(a: int) -> int:
    """Negation in Goldilocks. Equivalent to sub(0, a)."""
    return (-a) % P


def lagrange_eval(xs: List[int], ys: List[int], x_target: int) -> int:
    """Evaluate at `x_target` the unique polynomial of degree < len(xs)
    passing through {(xs[i], ys[i])}.
    """
    assert len(xs) == len(ys)
    result = 0
    for c in range(len(xs)):
        num, den = 1, 1
        for i in range(len(xs)):
            if i == c:
                continue
            num = mul(num, sub(x_target, xs[i]))
            den = mul(den, sub(xs[c], xs[i]))
        result = add(result, mul(ys[c], mul(num, inv(den))))
    return result


def lagrange_poly_coeffs(xs: List[int], ys: List[int]) -> List[int]:
    """Recover coefficients [c_0, c_1, ..., c_{d-1}] of the polynomial
    of degree < d = len(xs) that passes through (xs[i], ys[i]).

    Used by the verifier to recover polynomial coefficients from sample
    evaluations — e.g., recovering q_add's coefficients from values at
    ζ_c, to evaluate at η_j for the column-identity check.
    """
    d = len(xs)
    coeffs = [0] * d
    for c in range(d):
        # Build numerator polynomial prod_{i != c} (X - xs[i]).
        num_poly = [1]
        for i in range(d):
            if i == c:
                continue
            # Multiply num_poly by (X - xs[i]).
            new_poly = [0] * (len(num_poly) + 1)
            for j in range(len(num_poly)):
                new_poly[j] = sub(new_poly[j], mul(num_poly[j], xs[i]))
                new_poly[j + 1] = add(new_poly[j + 1], num_poly[j])
            num_poly = new_poly
        # Denominator scalar.
        den = 1
        for i in range(d):
            if i == c:
                continue
            den = mul(den, sub(xs[c], xs[i]))
        scale = mul(ys[c], inv(den))
        for j in range(d):
            coeffs[j] = add(coeffs[j], mul(num_poly[j], scale))
    return coeffs


def poly_eval(coeffs: List[int], x: int) -> int:
    """Horner evaluation of a polynomial in coefficient form at x.

    coeffs = [c_0, c_1, ..., c_{d-1}] represents c_0 + c_1·x + ... + c_{d-1}·x^{d-1}.
    """
    result = 0
    for c in reversed(coeffs):
        result = add(mul(result, x), c)
    return result


def poly_mul(a: List[int], b: List[int]) -> List[int]:
    """Convolution of two polynomials in coefficient form: returns
    coefficients of a(X) · b(X).

    O((len(a)) · len(b)) field operations. For large operands use
    NTT-based multiplication in goldilocks_ref.py (`pmul_goldilocks`).
    """
    if not a or not b:
        return []
    result = [0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        if ai == 0:
            continue
        for j, bj in enumerate(b):
            result[i + j] = add(result[i + j], mul(ai, bj))
    return result


def poly_add(a: List[int], b: List[int]) -> List[int]:
    """Sum of two polynomials in coefficient form (length-adapting)."""
    n = max(len(a), len(b))
    return [add(a[i] if i < len(a) else 0, b[i] if i < len(b) else 0) for i in range(n)]


# ---------------------------------------------------------------------------
# Self-test: cross-check Lagrange against Horner on small random instances.
# ---------------------------------------------------------------------------

def _self_test() -> None:
    import random
    rng = random.Random(0)

    # Random polynomial of degree < d; eval at d points; recover coeffs;
    # verify they match the original.
    for d in [1, 2, 5, 10]:
        coeffs = [rng.randrange(P) for _ in range(d)]
        xs = [rng.randrange(P) for _ in range(d)]
        # Make xs distinct.
        while len(set(xs)) != d:
            xs = [rng.randrange(P) for _ in range(d)]
        ys = [poly_eval(coeffs, x) for x in xs]
        recovered = lagrange_poly_coeffs(xs, ys)
        assert recovered == coeffs, f"d={d}: recovered {recovered}, expected {coeffs}"
        # Lagrange eval at a new point should match Horner.
        x_new = rng.randrange(P)
        assert lagrange_eval(xs, ys, x_new) == poly_eval(coeffs, x_new)

    # poly_mul cross-check via Horner.
    a = [3, 1, 4]   # 3 + x + 4x²
    b = [1, 5]      # 1 + 5x
    prod = poly_mul(a, b)
    # Expected: (3+x+4x²)(1+5x) = 3 + 16x + 9x² + 20x³ → [3, 16, 9, 20]
    expected = [3, 16, 9, 20]
    assert prod == expected, f"poly_mul: {prod} != {expected}"

    # poly_add length-adapting.
    assert poly_add([1, 2, 3], [10, 20]) == [11, 22, 3]

    print("polynomials.py: self-test passed")


if __name__ == "__main__":
    _self_test()
