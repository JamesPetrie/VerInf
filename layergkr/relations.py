"""The relation zoo, and the tagged ragged batching of doc §4 L3.

Every non-matmul constraint of a layer is a ZERO RELATION: a polynomial
expression that must vanish at every point of its domain.

    hadamard   c = a*b            ->  a*b - c
    booleanity b in {0,1}         ->  b*b - b
    affine     y = sum c_k x_k    ->  sum c_k x_k - y
    rescale    raw = q*s + r      ->  raw - s*q - r      (r range-checked by LogUp)

All of them are `sum_k coeff_k * prod_j factor_j`, which is exactly what
`sumcheck.prove_terms` proves. So one sumcheck batches the whole layer:

  * gates are padded to a common power-of-two domain. Padding a ZERO relation
    with zeros is free -- 0 = 0 holds -- unlike LogUp's reciprocal constraint,
    where padding had to carry real values (see logup.py).
  * gates are combined with verifier batching coefficients drawn AFTER their
    witnesses are committed, so a prover cannot fit a gate to its coefficient.
  * the batch is eq-weighted at a random z, turning "vanishes everywhere" into
    one scalar claim: sum_x eq(z,x) * sum_k lambda_k * G_k(x) == 0.

`node_id/kind/port` go into the batching domain tag, so the two operand ports of
x*x are not conflated (doc §4 L3).
"""
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

from prover.protocol import P as FIELD_P

from . import sumcheck as sc
from .counters import charge
from .logup import eq_vector
from .transcript import Transcript


@dataclass
class Gate:
    """One zero relation over a vector domain. `terms` are (coeff, factors)."""
    kind: str
    node_id: str
    terms: List[Tuple[int, List[List[int]]]] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.terms[0][1][0])

    @property
    def degree(self) -> int:
        return max(len(f) for _, f in self.terms)


def _vec(x):
    """A gate factor, copied if it is a Python sequence and kept as-is if it is
    already a device tensor. `list()` on a tensor would pull the whole witness
    back to the host one element at a time, which is exactly the cost the tensor
    semantics path exists to remove."""
    return x if hasattr(x, "shape") else list(x)


def hadamard(node_id: str, a: Sequence[int], b: Sequence[int],
             c: Sequence[int]) -> Gate:
    return Gate("hadamard", node_id, [(1, [_vec(a), _vec(b)]), (FIELD_P - 1, [_vec(c)])])


def booleanity(node_id: str, b: Sequence[int]) -> Gate:
    return Gate("boolean", node_id, [(1, [_vec(b), _vec(b)]), (FIELD_P - 1, [_vec(b)])])


def affine(node_id: str, coeffs: Sequence[int], xs: Sequence[Sequence[int]],
           y: Sequence[int]) -> Gate:
    terms = [(c % FIELD_P, [_vec(x)]) for c, x in zip(coeffs, xs)]
    terms.append((FIELD_P - 1, [_vec(y)]))
    return Gate("affine", node_id, terms)


def rescale(node_id: str, raw: Sequence[int], q: Sequence[int], r: Sequence[int],
            scale: int) -> Gate:
    """raw = scale*q + r. The bound 0 <= r < scale is NOT here -- it is a range
    lookup, because that is what makes it cheap; see semantics.py."""
    return Gate("rescale", node_id,
                [(1, [_vec(raw)]), ((-scale) % FIELD_P, [_vec(q)]),
                 (FIELD_P - 1, [_vec(r)])])


def _pad(vec, size: int):
    if hasattr(vec, "shape"):
        import torch
        if int(vec.shape[0]) == size:
            return vec
        out = torch.zeros(size, dtype=vec.dtype, device=vec.device)
        out[: int(vec.shape[0])] = vec
        return out
    return list(vec) + [0] * (size - len(vec))


def batch_domain(gates: Sequence[Gate]) -> int:
    n = max(g.size for g in gates)
    p = 1
    while p < n:
        p *= 2
    return p


def prove_batch(gates: Sequence[Gate], tr: Transcript, label: str = "gates",
                tape=None) -> Tuple[sc.SumcheckProof, List[int], List[int]]:
    """One eq-weighted sumcheck for every gate in the layer.

    Returns (proof, z, lambdas). The claim is 0 for an honest witness: the sum
    of eq-weighted, randomly-batched zero relations."""
    size = batch_domain(gates)
    n_vars = max(size.bit_length() - 1, 1)
    lambdas = tr.coin(f"{label}_batch", len(gates))
    z = tr.coin(f"{label}_z", n_vars)
    eq_z = eq_vector(z)
    charge(mul=len(eq_z))

    terms: List[Tuple[int, List[List[int]]]] = []
    for lam, g in zip(lambdas, gates):
        for coeff, factors in g.terms:
            terms.append(((lam * coeff) % FIELD_P,
                          [eq_z] + [_pad(f, size) for f in factors]))
    coins = tr.coin(f"{label}_sc", n_vars)
    proof = sc.prove_terms(terms, lambda i: coins[i], tape=tape)
    return proof, z, lambdas


def verify_batch(proof: sc.SumcheckProof, gates: Sequence[Gate],
                 z: Sequence[int], lambdas: Sequence[int],
                 mu0: int = 0) -> Tuple[bool, str]:
    """Recheck the batch. The claim must equal the CARRIED-IN mask mu0 (zero for
    an unmasked proof): the relations sum to zero, so anything else means some
    gate does not vanish. mu0 is authenticated elsewhere -- it is the previous
    masked claim, not something the prover may choose here."""
    size = batch_domain(gates)
    eq_z = eq_vector(z)
    terms = []
    for lam, g in zip(lambdas, gates):
        for coeff, factors in g.terms:
            terms.append(((lam * coeff) % FIELD_P,
                          [eq_z] + [_pad(f, size) for f in factors]))
    if proof.claim % FIELD_P != mu0 % FIELD_P:
        return False, "batched gate claim is not zero -- some relation does not vanish"
    ok, why = sc.verify_terms(proof, terms, lambda i: proof.challenges[i])
    if not ok:
        return False, why
    return True, "ok"
