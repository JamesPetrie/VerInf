"""Local proof bridge for claims selected by the one-pass sampled audit.

The production Tape uses CUDA tensors while :mod:`layergkr.sampled_audit`
uses small Python wire messages.  This module is the narrow bridge between
them.  A proof is generated only after the verifier's per-block challenge and
is copied to host memory, so verification consumes a prover message rather
than reusing an intermediate CUDA result.

``MatmulClaim`` has a Freivalds argument; ``AddClaim`` and the raw-product
part of ``HadamardClaim`` have eq-weighted sumchecks.  The registry also
records the intended family for every production claim, but callers must keep
exact-recomputation fallbacks visibly separate from materialized proofs.
"""
from __future__ import annotations

import dataclasses
import json
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import blake3
import protocol
import torch
from claims import (
    AddClaim,
    ConcatClaim,
    HadamardClaim,
    LinCombClaim,
    MatmulClaim,
    WordExtractionClaim,
)
from cuda_primitives import P, gl_matvec, gl_mul, gl_sub
from rescale_claim import RescaleClaim

from layergkr import sumcheck as sc

# This is a compiler-coverage registry, not a claim that every entry already
# has a production proof implementation.  Keeping it explicit makes a new
# model opcode fail closed instead of silently receiving an exact-check label.
CLAIM_PROOF_FAMILIES = {
    "MatmulClaim": "freivalds",
    "RoutedProjectedMatmulClaim": "freivalds",
    "FreivaldsCombineClaim": "freivalds",
    "AddClaim": "sumcheck",
    "ConcatClaim": "sumcheck",
    "HadamardClaim": "sumcheck",
    "InfoFinalizeClaim": "sumcheck",
    "LinCombClaim": "sumcheck",
    "MaxClaim": "sumcheck",
    "RescaleClaim": "sumcheck",
    "RmsNormClaim": "sumcheck",
    "RoPEClaim": "sumcheck",
    "RoutingClaim": "sumcheck",
    "SiluClaim": "sumcheck",
    "SoftmaxClaim": "sumcheck",
    "WordExtractionClaim": "sumcheck",
    "EmbeddingLookupClaim": "product-tree",
    "PairedTlookupClaim": "product-tree",
    "RangeWordClaim": "product-tree",
}

MATERIALIZED_PROOF_CLAIMS = frozenset({
    "AddClaim", "ConcatClaim", "HadamardClaim", "LinCombClaim",
    "MatmulClaim", "RescaleClaim", "WordExtractionClaim"})


def proof_family(claim: object) -> str:
    name = type(claim).__name__
    try:
        return CLAIM_PROOF_FAMILIES[name]
    except KeyError as exc:
        raise ValueError(f"no sampled local-proof family for {name}") from exc


def manifest_family_counts(claims: Iterable[object]) -> dict[str, int]:
    counts = Counter(proof_family(claim) for claim in claims)
    return dict(sorted(counts.items()))


def has_materialized_proof(claim: object) -> bool:
    return type(claim).__name__ in MATERIALIZED_PROOF_CLAIMS


def _host(tensor: torch.Tensor) -> torch.Tensor:
    flat = tensor.detach().contiguous().view(-1)
    if flat.dtype != torch.uint64:
        flat = flat.to(torch.uint64)
    return flat.cpu()


def _cuda_vec(values: list[int]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.uint64, device="cuda")


@dataclass
class MatmulFreivaldsProof:
    """Standard right-projection proof for all heads of one matmul.

    ``w_projection`` is ``B_h rho_h`` and ``c_projection`` is
    ``C_h rho_h``.  Both are host-resident proof messages.  The verifier
    authenticates them against B/C and checks ``A_h (B_h rho_h) = C_h rho_h``.
    """

    claim_index: int
    challenge: bytes
    w_projection: torch.Tensor
    c_projection: torch.Tensor

    @property
    def byte_size(self) -> int:
        return (self.w_projection.numel() + self.c_projection.numel()) * 8

    @property
    def digest(self) -> bytes:
        h = blake3.blake3(b"verinf/sampled/matmul-freivalds/v1")
        h.update(self.claim_index.to_bytes(8, "little"))
        h.update(len(self.challenge).to_bytes(8, "little"))
        h.update(self.challenge)
        for tensor in (self.w_projection, self.c_projection):
            h.update(tensor.numel().to_bytes(8, "little"))
            h.update(tensor.contiguous().numpy().tobytes())
        return h.digest()


def _rho(challenge: bytes, claim_index: int, length: int) -> list[int]:
    return protocol.op_vec(
        challenge, claim_index, "sampled-freivalds-rho", length)


def _matrices(claim: MatmulClaim, live: dict):
    heads, width = claim.heads, claim.head_dim
    a = live[claim.A].reshape(claim.m, heads, width)
    if claim.transpose_b:
        b = live[claim.B].reshape(claim.n, heads, width)
    else:
        b = live[claim.B].reshape(width, heads, claim.n)
    c_var = claim.C_full if claim.rescale_bits > 0 else claim.C
    c = live[c_var].reshape(claim.m, heads, claim.n)
    return a, b, c


def prove_matmul(claim: MatmulClaim, live: dict, *, claim_index: int,
                 challenge: bytes) -> MatmulFreivaldsProof:
    a, b, c = _matrices(claim, live)
    del a
    rho = _rho(challenge, claim_index, claim.heads * claim.n)
    w_parts, c_parts = [], []
    for head in range(claim.heads):
        rho_h = _cuda_vec(rho[head * claim.n:(head + 1) * claim.n])
        if claim.transpose_b:
            b_h = b[:, head, :].T.contiguous()
        else:
            b_h = b[:, head, :].contiguous()
        c_h = c[:, head, :].contiguous()
        w_parts.append(gl_matvec(b_h, rho_h))
        c_parts.append(gl_matvec(c_h, rho_h))
    return MatmulFreivaldsProof(
        claim_index=claim_index,
        challenge=bytes(challenge),
        w_projection=_host(torch.cat(w_parts)),
        c_projection=_host(torch.cat(c_parts)),
    )


def verify_matmul(claim: MatmulClaim, live: dict,
                  proof: MatmulFreivaldsProof, *, claim_index: int,
                  challenge: bytes) -> tuple[bool, str]:
    if proof.claim_index != claim_index or proof.challenge != challenge:
        return False, "Freivalds transcript challenge mismatch"
    if (proof.w_projection.numel() != claim.k
            or proof.c_projection.numel() != claim.heads * claim.m):
        return False, "Freivalds projection shape mismatch"

    a, b, c = _matrices(claim, live)
    rho = _rho(challenge, claim_index, claim.heads * claim.n)
    w_message = proof.w_projection.to("cuda")
    c_message = proof.c_projection.to("cuda")
    for head in range(claim.heads):
        rho_h = _cuda_vec(rho[head * claim.n:(head + 1) * claim.n])
        w_lo, w_hi = head * claim.head_dim, (head + 1) * claim.head_dim
        c_lo, c_hi = head * claim.m, (head + 1) * claim.m
        if claim.transpose_b:
            b_h = b[:, head, :].T.contiguous()
        else:
            b_h = b[:, head, :].contiguous()
        expected_w = gl_matvec(b_h, rho_h)
        expected_c = gl_matvec(c[:, head, :].contiguous(), rho_h)
        if not torch.equal(expected_w, w_message[w_lo:w_hi]):
            return False, "Freivalds B projection is not witness-bound"
        if not torch.equal(expected_c, c_message[c_lo:c_hi]):
            return False, "Freivalds C projection is not witness-bound"
        contraction = gl_matvec(
            a[:, head, :].contiguous(), w_message[w_lo:w_hi])
        if not torch.equal(contraction, c_message[c_lo:c_hi]):
            return False, "Freivalds contraction failed"
    return True, "ok"


@dataclass
class RelationSumcheckProof:
    """A host-resident sumcheck transcript for one eq-weighted relation."""

    claim_index: int
    challenge: bytes
    relation: str
    sumcheck: sc.SumcheckProof

    @property
    def byte_size(self) -> int:
        samples = sum(len(poly) for poly in self.sumcheck.round_polys)
        scalars = (1 + 2 * samples + len(self.sumcheck.challenges)
                   + len(self.sumcheck.final_point) + 1)
        return scalars * 8

    @property
    def digest(self) -> bytes:
        payload = json.dumps(
            dataclasses.asdict(self), sort_keys=True,
            separators=(",", ":"), default=lambda value: value.hex()
            if isinstance(value, bytes) else value).encode()
        return blake3.blake3(
            b"verinf/sampled/relation-sumcheck/v1" + payload).digest()


def _pad_pow2(tensor: torch.Tensor, length: int) -> torch.Tensor:
    flat = tensor.detach().contiguous().view(-1).to(torch.uint64)
    size = 1 << max(0, (length - 1).bit_length())
    if flat.numel() == size:
        return flat
    out = torch.zeros(size, dtype=torch.uint64, device=flat.device)
    out[:length] = flat[:length]
    return out


def _eq_weights(tau: list[int], device) -> torch.Tensor:
    weights = torch.ones(1, dtype=torch.uint64, device=device)
    # Reverse construction makes tau[0] the most-significant MLE variable,
    # matching sumcheck.mle_eval's first-half/second-half fold order.
    for value in reversed(tau):
        r = torch.tensor([value], dtype=torch.uint64, device=device)
        rr = r.expand(weights.numel()).contiguous()
        one = torch.ones_like(rr)
        weights = torch.cat([
            gl_mul(weights, gl_sub(one, rr)),
            gl_mul(weights, rr),
        ])
    return weights


def _relation_terms(claim: object, live: dict, *, claim_index: int,
                    challenge: bytes):
    length = claim.length
    size = 1 << max(0, (length - 1).bit_length())
    rounds = size.bit_length() - 1
    tau = protocol.op_vec(
        challenge, claim_index, "sampled-sumcheck-eq", rounds)
    if isinstance(claim, (AddClaim, HadamardClaim)):
        anchor = claim.a
    elif isinstance(claim, ConcatClaim):
        anchor = claim.dst
    elif isinstance(claim, LinCombClaim):
        anchor = claim.xs[0]
    elif isinstance(claim, WordExtractionClaim):
        anchor = claim.x
    elif isinstance(claim, RescaleClaim):
        anchor = claim.x_full
    else:
        raise TypeError(
            f"no sumcheck relation builder for {type(claim).__name__}")
    eq = _eq_weights(tau, live[anchor].device)

    if isinstance(claim, AddClaim):
        a = _pad_pow2(live[claim.a], length)
        if claim.public_rhs is not None:
            rhs = torch.zeros(size, dtype=torch.uint64, device=a.device)
            rhs[:length] = int(claim.public_rhs) % P
            return "add-public", [
                (1, [a, eq]), (P - 1, [rhs, eq])]
        b = _pad_pow2(live[claim.b], length)
        c = _pad_pow2(live[claim.c], length)
        return "add", [
            (1, [a, eq]), (1, [b, eq]), (P - 1, [c, eq])]

    if isinstance(claim, HadamardClaim):
        a = _pad_pow2(live[claim.a], length)
        b = _pad_pow2(live[claim.b], length)
        target = claim.c_full if claim.rescale_bits > 0 else claim.c
        c = _pad_pow2(live[target], length)
        return "hadamard", [
            (1, [a, b, eq]), (P - 1, [c, eq])]

    if isinstance(claim, ConcatClaim):
        joined = torch.cat([
            live[var].detach().contiguous().view(-1) for var in claim.srcs])
        src = _pad_pow2(joined, length)
        dst = _pad_pow2(live[claim.dst], length)
        return "concat", [
            (1, [src, eq]), (P - 1, [dst, eq])]

    if isinstance(claim, LinCombClaim):
        terms = [(int(coef) % P,
                  [_pad_pow2(live[var], length), eq])
                 for var, coef in zip(claim.xs, claim.coefs)]
        rhs = torch.zeros(size, dtype=torch.uint64, device=eq.device)
        if len(claim.rhs) == 1:
            rhs[:length] = int(claim.rhs[0]) % P
        else:
            rhs[:length] = torch.tensor(
                claim.rhs, dtype=torch.uint64, device=eq.device)
        terms.append((P - 1, [rhs, eq]))
        return "lincomb", terms

    if isinstance(claim, WordExtractionClaim):
        terms = [(1, [_pad_pow2(live[claim.x], length), eq])]
        for word, coef in zip(claim.words, claim.coeffs):
            terms.append(((P - int(coef)) % P,
                          [_pad_pow2(live[word], length), eq]))
        if claim.shift % P:
            shift = torch.zeros(
                size, dtype=torch.uint64, device=eq.device)
            shift[:length] = claim.shift % P
            terms.append((1, [shift, eq]))
        return "word-extraction", terms

    # Batch the two public rescale linears with an independent random tag so
    # errors in one cannot cancel errors in the other except with 1/|F| chance.
    gamma = protocol.op_vec(
        challenge, claim_index, "sampled-rescale-batch", 1)[0]
    x_full = _pad_pow2(live[claim.x_full], length)
    x = _pad_pow2(live[claim.x], length)
    low = _pad_pow2(live[claim.x_low], length)
    shifted = _pad_pow2(live[claim.x_shifted], length)
    offset = torch.zeros(size, dtype=torch.uint64, device=eq.device)
    offset[:length] = 1 << (claim.output_width - 1)
    return "rescale-linear", [
        (1, [x_full, eq]),
        ((P - (1 << claim.rescale_bits)) % P, [x, eq]),
        (P - 1, [low, eq]),
        (gamma, [shifted, eq]),
        ((P - gamma) % P, [x, eq]),
        ((P - gamma) % P, [offset, eq]),
    ]


def _sumcheck_coins(challenge: bytes, claim_index: int, rounds: int):
    coins = protocol.op_vec(
        challenge, claim_index, "sampled-sumcheck-round", rounds)
    return coins, lambda index: coins[index]


def prove_sumcheck(claim: object, live: dict, *, claim_index: int,
                   challenge: bytes) -> RelationSumcheckProof:
    relation, terms = _relation_terms(
        claim, live, claim_index=claim_index, challenge=challenge)
    rounds = len(terms[0][1][0]).bit_length() - 1
    _coins, coin = _sumcheck_coins(challenge, claim_index, rounds)
    work = len(terms[0][1][0]) * sum(len(factors)
                                     for _coef, factors in terms)
    if work < sc.GPU_MIN_SUMCHECK_WORK:
        prover_terms = [(coef, [factor.cpu().tolist() for factor in factors])
                        for coef, factors in terms]
    else:
        prover_terms = terms
    proof = sc.prove_terms(prover_terms, coin)
    return RelationSumcheckProof(
        claim_index, bytes(challenge), relation, proof)


def verify_sumcheck(claim: object, live: dict,
                    proof: RelationSumcheckProof, *, claim_index: int,
                    challenge: bytes) -> tuple[bool, str]:
    if proof.claim_index != claim_index or proof.challenge != challenge:
        return False, "sumcheck transcript challenge mismatch"
    relation, terms = _relation_terms(
        claim, live, claim_index=claim_index, challenge=challenge)
    if proof.relation != relation:
        return False, "sumcheck relation mismatch"
    if proof.sumcheck.claim % P != 0:
        return False, "sumcheck relation claim is not zero"
    rounds = len(terms[0][1][0]).bit_length() - 1
    _coins, coin = _sumcheck_coins(challenge, claim_index, rounds)
    ok, why = sc.verify_terms(proof.sumcheck, terms, coin)
    return ok, why if not ok else "ok"
