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
    EmbeddingLookupClaim,
    HadamardClaim,
    LinCombClaim,
    MatmulClaim,
    PairedTlookupClaim,
    RangeWordClaim,
    WordExtractionClaim,
)
from cuda_primitives import P, gl_add, gl_matvec, gl_mul, gl_sub
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
    "AddClaim", "ConcatClaim", "EmbeddingLookupClaim", "HadamardClaim",
    "LinCombClaim", "MatmulClaim", "PairedTlookupClaim", "RangeWordClaim",
    "RescaleClaim", "WordExtractionClaim"})


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


@dataclass
class RangeProductTreeProof:
    """Two compact roots of the query and indexed-table product trees."""

    claim_index: int
    challenge: bytes
    query_root: int
    indexed_table_root: int

    @property
    def byte_size(self) -> int:
        return 16

    @property
    def digest(self) -> bytes:
        return blake3.blake3(
            b"verinf/sampled/range-product-tree/v1"
            + self.claim_index.to_bytes(8, "little")
            + self.challenge
            + int(self.query_root).to_bytes(8, "little")
            + int(self.indexed_table_root).to_bytes(8, "little")).digest()


def _product_tree_root(values: torch.Tensor) -> int:
    level = values.detach().contiguous().view(-1).to(torch.uint64)
    if not level.numel():
        return 1
    while level.numel() > 1:
        if level.numel() & 1:
            level = torch.cat([
                level, torch.ones(1, dtype=torch.uint64,
                                   device=level.device)])
        level = gl_mul(level[0::2].contiguous(),
                       level[1::2].contiguous())
    return int(level[0].item())


def _range_products(claim: RangeWordClaim, live: dict, *, claim_index: int,
                    challenge: bytes):
    if claim.local_indices is None:
        raise ValueError("RangeWordClaim has no pre-commit local index wire")
    alpha = protocol.op_vec(
        challenge, claim_index, "sampled-lookup-alpha", 1)[0]
    query = live[claim.x].detach().contiguous().view(-1)
    indices = live[claim.local_indices].detach().contiguous().view(-1)
    signed_indices = indices.view(torch.int64)
    table = claim.table.T.detach().contiguous().view(-1)
    safe = signed_indices.clamp(0, table.numel() - 1)
    indexed_table = table.index_select(0, safe)
    alpha_t = torch.full_like(query, alpha)
    return (
        _product_tree_root(gl_sub(alpha_t, query)),
        _product_tree_root(gl_sub(alpha_t, indexed_table)),
        signed_indices,
        table.numel(),
    )


def prove_range_product_tree(
        claim: RangeWordClaim, live: dict, *, claim_index: int,
        challenge: bytes) -> RangeProductTreeProof:
    query_root, table_root, _indices, _table_len = _range_products(
        claim, live, claim_index=claim_index, challenge=challenge)
    return RangeProductTreeProof(
        claim_index, bytes(challenge), query_root, table_root)


def verify_range_product_tree(
        claim: RangeWordClaim, live: dict, proof: RangeProductTreeProof, *,
        claim_index: int, challenge: bytes) -> tuple[bool, str]:
    if proof.claim_index != claim_index or proof.challenge != challenge:
        return False, "product-tree transcript challenge mismatch"
    query_root, table_root, indices, table_len = _range_products(
        claim, live, claim_index=claim_index, challenge=challenge)
    valid = bool(((indices >= 0) & (indices < table_len)).all().item())
    if not valid:
        return False, "product-tree lookup index is outside the public table"
    if query_root != proof.query_root:
        return False, "query product-tree root is not witness-bound"
    if table_root != proof.indexed_table_root:
        return False, "table product-tree root is not index-bound"
    if proof.query_root != proof.indexed_table_root:
        return False, "lookup product-tree roots differ"
    return True, "ok"


@dataclass
class EmbeddingProductTreeProof:
    """Compact multiset equality for public-index rows of committed E."""

    claim_index: int
    challenge: bytes
    output_root: int
    embedding_root: int

    @property
    def byte_size(self) -> int:
        return 16

    @property
    def digest(self) -> bytes:
        return blake3.blake3(
            b"verinf/sampled/embedding-product-tree/v1"
            + self.claim_index.to_bytes(8, "little")
            + self.challenge
            + int(self.output_root).to_bytes(8, "little")
            + int(self.embedding_root).to_bytes(8, "little")).digest()


def _embedding_products(claim: EmbeddingLookupClaim, live: dict, *,
                        claim_index: int, challenge: bytes):
    alpha, beta = protocol.op_vec(
        challenge, claim_index, "sampled-embedding-alpha-beta", 2)
    embedding = live[claim.E].detach().contiguous().view(-1)
    output = live[claim.x].detach().contiguous().view(-1)
    vocab = embedding.numel() // claim.d
    indices = torch.tensor(
        claim.token_ids, dtype=torch.int64, device=embedding.device)
    valid = bool(((indices >= 0) & (indices < vocab)).all().item())
    safe = indices.clamp(0, vocab - 1)
    selected = embedding.view(vocab, claim.d).index_select(
        0, safe).contiguous().view(-1)
    alpha_t = torch.full_like(output, alpha)
    beta_t = torch.full_like(output, beta)
    positions = torch.arange(
        output.numel(), dtype=torch.int64,
        device=output.device).to(torch.uint64)
    output_fp = gl_add(output, gl_mul(beta_t, positions))
    selected_fp = gl_add(selected, gl_mul(beta_t, positions))
    return (
        _product_tree_root(gl_sub(alpha_t, output_fp)),
        _product_tree_root(gl_sub(alpha_t, selected_fp)),
        valid,
    )


def prove_embedding_product_tree(
        claim: EmbeddingLookupClaim, live: dict, *, claim_index: int,
        challenge: bytes) -> EmbeddingProductTreeProof:
    output_root, embedding_root, _valid = _embedding_products(
        claim, live, claim_index=claim_index, challenge=challenge)
    return EmbeddingProductTreeProof(
        claim_index, bytes(challenge), output_root, embedding_root)


def verify_embedding_product_tree(
        claim: EmbeddingLookupClaim, live: dict,
        proof: EmbeddingProductTreeProof, *, claim_index: int,
        challenge: bytes) -> tuple[bool, str]:
    if proof.claim_index != claim_index or proof.challenge != challenge:
        return False, "embedding product-tree transcript mismatch"
    output_root, embedding_root, valid = _embedding_products(
        claim, live, claim_index=claim_index, challenge=challenge)
    if not valid:
        return False, "embedding token index is outside the public table"
    if output_root != proof.output_root:
        return False, "embedding output product root is not witness-bound"
    if embedding_root != proof.embedding_root:
        return False, "embedding table product root is not witness-bound"
    if proof.output_root != proof.embedding_root:
        return False, "embedding product-tree roots differ"
    return True, "ok"


@dataclass
class PairedLookupProductTreeProof:
    """Compact equality of position-tagged committed and public pairs."""

    claim_index: int
    challenge: bytes
    query_root: int
    table_root: int

    @property
    def byte_size(self) -> int:
        return 16

    @property
    def digest(self) -> bytes:
        return blake3.blake3(
            b"verinf/sampled/paired-lookup-product-tree/v1"
            + self.claim_index.to_bytes(8, "little")
            + self.challenge
            + int(self.query_root).to_bytes(8, "little")
            + int(self.table_root).to_bytes(8, "little")).digest()


def _paired_lookup_products(
        claim: PairedTlookupClaim, live: dict, *, claim_index: int,
        challenge: bytes):
    alpha, beta, gamma = protocol.op_vec(
        challenge, claim_index, "sampled-paired-lookup-challenges", 3)
    x = live[claim.x].detach().contiguous().view(-1)
    y = live[claim.y].detach().contiguous().view(-1)
    shift_t = torch.full_like(x, claim.shift % P)
    keys = gl_add(x, shift_t)
    indices = keys.view(torch.int64)
    table_x = claim.table.T.detach().contiguous().view(-1)
    table_y = claim.table.T_Y.detach().contiguous().view(-1)
    safe = indices.clamp(0, table_x.numel() - 1)
    selected_x = table_x.index_select(0, safe)
    selected_y = table_y.index_select(0, safe)
    positions = torch.arange(
        x.numel(), dtype=torch.int64, device=x.device).to(torch.uint64)
    alpha_t = torch.full_like(x, alpha)
    beta_t = torch.full_like(x, beta)
    gamma_t = torch.full_like(x, gamma)
    position_tags = gl_mul(gamma_t, positions)
    query_fp = gl_add(gl_add(keys, gl_mul(beta_t, y)), position_tags)
    table_fp = gl_add(
        gl_add(selected_x, gl_mul(beta_t, selected_y)), position_tags)
    return (
        _product_tree_root(gl_sub(alpha_t, query_fp)),
        _product_tree_root(gl_sub(alpha_t, table_fp)),
        indices,
        table_x.numel(),
    )


def prove_paired_lookup_product_tree(
        claim: PairedTlookupClaim, live: dict, *, claim_index: int,
        challenge: bytes) -> PairedLookupProductTreeProof:
    query_root, table_root, _indices, _table_len = _paired_lookup_products(
        claim, live, claim_index=claim_index, challenge=challenge)
    return PairedLookupProductTreeProof(
        claim_index, bytes(challenge), query_root, table_root)


def verify_paired_lookup_product_tree(
        claim: PairedTlookupClaim, live: dict,
        proof: PairedLookupProductTreeProof, *, claim_index: int,
        challenge: bytes) -> tuple[bool, str]:
    if proof.claim_index != claim_index or proof.challenge != challenge:
        return False, "paired lookup product-tree transcript mismatch"
    query_root, table_root, indices, table_len = _paired_lookup_products(
        claim, live, claim_index=claim_index, challenge=challenge)
    valid = bool(((indices >= 0) & (indices < table_len)).all().item())
    if not valid:
        return False, "paired lookup index is outside the public table"
    if query_root != proof.query_root:
        return False, "paired lookup query root is not witness-bound"
    if table_root != proof.table_root:
        return False, "paired lookup table root is not index-bound"
    if proof.query_root != proof.table_root:
        return False, "paired lookup product-tree roots differ"
    return True, "ok"
