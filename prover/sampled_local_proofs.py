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
    RMS_LIMB_W,
    AddClaim,
    ConcatClaim,
    EmbeddingLookupClaim,
    HadamardClaim,
    LinCombClaim,
    MatmulClaim,
    PairedTlookupClaim,
    RangeWordClaim,
    RmsNormClaim,
    RoPEClaim,
    SiluClaim,
    SoftmaxClaim,
    WordExtractionClaim,
    _chunk_widths,
    _rms_limb_range_groups,
    _rope_cos_sin,
)
from cuda_primitives import P, gl_add, gl_matmul, gl_matvec, gl_mul, gl_sub
from max_claim import MaxClaim
from rescale_claim import RescaleClaim
from routed_projected import RoutedProjectedMatmulClaim
from routing_claim import FreivaldsCombineClaim, RoutingClaim
from ui_claim import InfoFinalizeClaim

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
    "FreivaldsCombineClaim", "InfoFinalizeClaim", "LinCombClaim",
    "MatmulClaim", "MaxClaim",
    "PairedTlookupClaim", "RangeWordClaim", "RescaleClaim", "RmsNormClaim",
    "RoPEClaim", "RoutedProjectedMatmulClaim", "RoutingClaim", "SiluClaim",
    "SoftmaxClaim", "WordExtractionClaim"})


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
                 challenge: bytes):
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
    freivalds = MatmulFreivaldsProof(
        claim_index=claim_index,
        challenge=bytes(challenge),
        w_projection=_host(torch.cat(w_parts)),
        c_projection=_host(torch.cat(c_parts)),
    )
    if claim.rescale_bits == 0:
        return freivalds
    return MatmulAuditProof(
        freivalds=freivalds,
        rounding=prove_rounding_sumcheck(
            claim, live, claim_index=claim_index, challenge=challenge),
    )


def _verify_matmul_freivalds(
        claim: MatmulClaim, live: dict, proof: MatmulFreivaldsProof, *,
        claim_index: int, challenge: bytes) -> tuple[bool, str]:
    if not isinstance(proof, MatmulFreivaldsProof):
        return False, "matmul Freivalds proof type mismatch"
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


def verify_matmul(claim: MatmulClaim, live: dict, proof, *,
                  claim_index: int,
                  challenge: bytes) -> tuple[bool, str]:
    if claim.rescale_bits == 0:
        if not isinstance(proof, MatmulFreivaldsProof):
            return False, "matmul Freivalds proof type mismatch"
        return _verify_matmul_freivalds(
            claim, live, proof, claim_index=claim_index,
            challenge=challenge)
    if not isinstance(proof, MatmulAuditProof):
        return False, "rescaled matmul proof has no rounding argument"
    ok, why = _verify_matmul_freivalds(
        claim, live, proof.freivalds, claim_index=claim_index,
        challenge=challenge)
    if not ok:
        return ok, why
    return verify_rounding_sumcheck(
        claim, live, proof.rounding, claim_index=claim_index,
        challenge=challenge)


@dataclass
class CombineFreivaldsProof:
    """Host messages for the masked expert-stream contraction."""

    claim_index: int
    challenge: bytes
    expert_projections: torch.Tensor
    output_projection: torch.Tensor

    @property
    def byte_size(self) -> int:
        return (self.expert_projections.numel()
                + self.output_projection.numel()) * 8

    @property
    def digest(self) -> bytes:
        h = blake3.blake3(b"verinf/sampled/combine-freivalds/v1")
        h.update(self.claim_index.to_bytes(8, "little"))
        h.update(len(self.challenge).to_bytes(8, "little"))
        h.update(self.challenge)
        for tensor in (self.expert_projections, self.output_projection):
            h.update(tensor.numel().to_bytes(8, "little"))
            h.update(tensor.contiguous().numpy().tobytes())
        return h.digest()


def _combine_rho(challenge: bytes, claim_index: int, length: int) -> list[int]:
    return protocol.op_vec(
        challenge, claim_index, "sampled-combine-freivalds-rho", length)


def _combine_projections(claim: FreivaldsCombineClaim, live: dict,
                         rho_t: torch.Tensor):
    expert = []
    for var in claim.xs:
        value = live[var]
        if callable(value):
            value = value()
        expert.append(gl_matvec(value.reshape(claim.T, claim.F), rho_t))
    output = gl_matvec(live[claim.y].reshape(claim.T, claim.F), rho_t)
    return torch.stack(expert), output


def prove_freivalds_combine(
        claim: FreivaldsCombineClaim, live: dict, *, claim_index: int,
        challenge: bytes) -> CombineFreivaldsProof:
    rho_t = _cuda_vec(_combine_rho(challenge, claim_index, claim.F))
    expert, output = _combine_projections(claim, live, rho_t)
    return CombineFreivaldsProof(
        claim_index=claim_index,
        challenge=bytes(challenge),
        expert_projections=_host(expert),
        output_projection=_host(output),
    )


def verify_freivalds_combine(
        claim: FreivaldsCombineClaim, live: dict,
        proof: CombineFreivaldsProof, *, claim_index: int,
        challenge: bytes) -> tuple[bool, str]:
    if proof.claim_index != claim_index or proof.challenge != challenge:
        return False, "combine Freivalds transcript challenge mismatch"
    if (proof.expert_projections.numel() != claim.E * claim.T
            or proof.output_projection.numel() != claim.T):
        return False, "combine Freivalds projection shape mismatch"
    rho_t = _cuda_vec(_combine_rho(challenge, claim_index, claim.F))
    expected_expert, expected_output = _combine_projections(
        claim, live, rho_t)
    expert_message = proof.expert_projections.to("cuda").reshape(
        claim.E, claim.T)
    output_message = proof.output_projection.to("cuda")
    if not torch.equal(expected_expert, expert_message):
        return False, "combine expert projections are not witness-bound"
    if not torch.equal(expected_output, output_message):
        return False, "combine output projection is not witness-bound"
    mask = live[claim.m].reshape(claim.T, claim.E)
    projected_by_token = expert_message.T.contiguous()
    masked = gl_mul(mask, projected_by_token)
    ones = torch.ones(claim.E, dtype=torch.uint64, device=mask.device)
    contraction = gl_matvec(masked.contiguous(), ones)
    if not torch.equal(contraction, output_message):
        return False, "combine Freivalds contraction failed"
    return True, "ok"


@dataclass
class RoutedFreivaldsProof:
    """Projected expert weights and routed output for one selected claim."""

    claim_index: int
    challenge: bytes
    weight_projections: torch.Tensor
    output_projection: torch.Tensor

    @property
    def byte_size(self) -> int:
        return (self.weight_projections.numel()
                + self.output_projection.numel()) * 8

    @property
    def digest(self) -> bytes:
        h = blake3.blake3(b"verinf/sampled/routed-freivalds/v1")
        h.update(self.claim_index.to_bytes(8, "little"))
        h.update(len(self.challenge).to_bytes(8, "little"))
        h.update(self.challenge)
        for tensor in (self.weight_projections, self.output_projection):
            h.update(tensor.numel().to_bytes(8, "little"))
            h.update(tensor.contiguous().numpy().tobytes())
        return h.digest()


def _routed_rho(challenge: bytes, claim_index: int, length: int) -> list[int]:
    return protocol.op_vec(
        challenge, claim_index, "sampled-routed-freivalds-rho", length)


def _routed_projections(claim: RoutedProjectedMatmulClaim, live: dict,
                        rho_t: torch.Tensor):
    projected_weights = []
    for var in claim.W:
        value = live[var]
        if callable(value):
            value = value()
        projected_weights.append(
            gl_matvec(value.reshape(claim.K, claim.J), rho_t))
    output = gl_matvec(live[claim.Y].reshape(claim.T, claim.J), rho_t)
    return torch.stack(projected_weights), output


def prove_routed_matmul(
        claim: RoutedProjectedMatmulClaim, live: dict, *, claim_index: int,
        challenge: bytes) -> RoutedFreivaldsProof:
    rho_t = _cuda_vec(_routed_rho(challenge, claim_index, claim.J))
    weights, output = _routed_projections(claim, live, rho_t)
    return RoutedFreivaldsProof(
        claim_index=claim_index,
        challenge=bytes(challenge),
        weight_projections=_host(weights),
        output_projection=_host(output),
    )


def verify_routed_matmul(
        claim: RoutedProjectedMatmulClaim, live: dict,
        proof: RoutedFreivaldsProof, *, claim_index: int,
        challenge: bytes) -> tuple[bool, str]:
    if proof.claim_index != claim_index or proof.challenge != challenge:
        return False, "routed Freivalds transcript challenge mismatch"
    if (proof.weight_projections.numel() != claim.E * claim.K
            or proof.output_projection.numel() != claim.T):
        return False, "routed Freivalds projection shape mismatch"
    rho_t = _cuda_vec(_routed_rho(challenge, claim_index, claim.J))
    expected_weights, expected_output = _routed_projections(
        claim, live, rho_t)
    weight_message = proof.weight_projections.to("cuda").reshape(
        claim.E, claim.K)
    output_message = proof.output_projection.to("cuda")
    if not torch.equal(expected_weights, weight_message):
        return False, "routed weight projections are not model-bound"
    if not torch.equal(expected_output, output_message):
        return False, "routed output projection is not witness-bound"
    mask = live[claim.M].reshape(claim.T, claim.E)
    routed_projection = gl_matmul(mask, weight_message)
    products = gl_mul(live[claim.X].reshape(claim.T, claim.K),
                      routed_projection)
    ones = torch.ones(claim.K, dtype=torch.uint64, device=mask.device)
    contraction = gl_matvec(products.contiguous(), ones)
    if not torch.equal(contraction, output_message):
        return False, "routed Freivalds contraction failed"
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


def _scale(value: torch.Tensor, coefficient: int) -> torch.Tensor:
    return gl_mul(torch.full_like(value, coefficient % P), value)


def _batch_residuals(residuals, coefficients, length: int) -> torch.Tensor:
    """Challenge-batch ragged committed-wire residuals before MLE padding.

    The verifier derives the same vector from authenticated selected wires.
    Padding only this vector, rather than every source wire, is what keeps a
    selected 202M-slot UI/LM claim below the A100 memory ceiling.
    """
    first = residuals[0].reshape(-1)
    aggregate = torch.zeros(
        length, dtype=torch.uint64, device=first.device)
    for coefficient, residual in zip(coefficients, residuals):
        flat = residual.reshape(-1)
        count = flat.numel()
        aggregate[:count] = gl_add(
            aggregate[:count].contiguous(), _scale(flat, coefficient))
    return aggregate


def _relation_terms(claim: object, live: dict, *, claim_index: int,
                    challenge: bytes):
    length = (claim.x.length
              if isinstance(claim, (RmsNormClaim, RoPEClaim))
              else claim.length)
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
    elif isinstance(claim, RoPEClaim):
        anchor = claim.x
    elif isinstance(claim, RoutingClaim):
        anchor = claim.r
    elif isinstance(claim, SiluClaim):
        anchor = claim.x
    elif isinstance(claim, RmsNormClaim):
        anchor = claim.x
    elif isinstance(claim, SoftmaxClaim):
        anchor = claim.x
    elif isinstance(claim, MaxClaim):
        anchor = claim.l
    elif isinstance(claim, InfoFinalizeClaim):
        anchor = claim.a
    else:
        raise TypeError(
            f"no sumcheck relation builder for {type(claim).__name__}")
    eq = _eq_weights(tau, live[anchor].device)

    if isinstance(claim, AddClaim):
        a = live[claim.a].detach().contiguous().view(-1)
        if claim.public_rhs is not None:
            residual = gl_sub(
                a, torch.full_like(a, int(claim.public_rhs) % P))
            return "add-public", [
                (1, [_pad_pow2(residual, length), eq])]
        residual = gl_sub(
            gl_add(a, live[claim.b].detach().contiguous().view(-1)),
            live[claim.c].detach().contiguous().view(-1))
        return "add", [
            (1, [_pad_pow2(residual, length), eq])]

    if isinstance(claim, HadamardClaim):
        target = claim.c_full if claim.rescale_bits > 0 else claim.c
        raw = gl_sub(
            gl_mul(live[claim.a].detach().contiguous().view(-1),
                   live[claim.b].detach().contiguous().view(-1)),
            live[target].detach().contiguous().view(-1))
        if claim.rescale_bits == 0:
            return "hadamard", [
                (1, [_pad_pow2(raw, length), eq])]
        gamma, delta = protocol.op_vec(
            challenge, claim_index, "sampled-hadamard-rescale-batch", 2)
        full = live[claim.c_full].detach().contiguous().view(-1)
        out = live[claim.c].detach().contiguous().view(-1)
        low = live[claim.c_low].detach().contiguous().view(-1)
        shifted = live[claim.c_shifted].detach().contiguous().view(-1)
        rescale = gl_sub(gl_sub(full, _scale(
            out, 1 << claim.rescale_bits)), low)
        shift = gl_sub(
            gl_sub(shifted, out),
            torch.full_like(out, 1 << (claim.output_width - 1)))
        aggregate = _batch_residuals(
            [raw, rescale, shift], [1, gamma, delta], length)
        return "hadamard-rescale", [
            (1, [_pad_pow2(aggregate, length), eq])]

    if isinstance(claim, ConcatClaim):
        joined = torch.cat([
            live[var].detach().contiguous().view(-1) for var in claim.srcs])
        residual = gl_sub(
            joined, live[claim.dst].detach().contiguous().view(-1))
        return "concat", [
            (1, [_pad_pow2(residual, length), eq])]

    if isinstance(claim, LinCombClaim):
        lhs = torch.zeros(length, dtype=torch.uint64, device=eq.device)
        for var, coefficient in zip(claim.xs, claim.coefs):
            lhs = gl_add(lhs, _scale(
                live[var].detach().contiguous().view(-1), coefficient))
        if len(claim.rhs) == 1:
            rhs = torch.full_like(lhs, int(claim.rhs[0]) % P)
        else:
            rhs = torch.tensor(
                claim.rhs, dtype=torch.uint64, device=eq.device)
        return "lincomb", [
            (1, [_pad_pow2(gl_sub(lhs, rhs), length), eq])]

    if isinstance(claim, WordExtractionClaim):
        residual = live[claim.x].detach().contiguous().view(-1).clone()
        for word, coef in zip(claim.words, claim.coeffs):
            residual = gl_sub(
                residual, _scale(
                    live[word].detach().contiguous().view(-1), coef))
        if claim.shift % P:
            residual = gl_add(
                residual, torch.full_like(residual, claim.shift % P))
        return "word-extraction", [
            (1, [_pad_pow2(residual, length), eq])]

    if isinstance(claim, RoPEClaim):
        cfg = claim.config
        seq, heads, d_h = cfg.SEQ, cfg.heads, cfg.d_h
        half = d_h // 2
        x = live[claim.x].reshape(seq, heads, d_h)
        x_lo, x_hi = x[:, :, :half], x[:, :, half:]
        paired = torch.cat([x_hi, x_lo], dim=2).reshape(-1)
        cos_l, sin_l = _rope_cos_sin(cfg)
        cos = torch.tensor(
            cos_l, dtype=torch.uint64, device=x.device).reshape(
                seq, 1, half).expand(seq, heads, half).contiguous()
        sin = torch.tensor(
            sin_l, dtype=torch.uint64, device=x.device).reshape(
                seq, 1, half).expand(seq, heads, half).contiguous()
        zero = torch.zeros_like(cos)
        neg_cos = gl_sub(zero, cos)
        self_coeff = torch.cat([neg_cos, neg_cos], dim=2).reshape(-1)
        pair_coeff = torch.cat(
            [sin, gl_sub(zero, sin)], dim=2).reshape(-1)
        target_var = (claim.x_rot_full if claim.rescale_bits > 0
                      else claim.x_rot)
        target = _pad_pow2(live[target_var], length)
        terms = [
            (1, [target, eq]),
            (1, [_pad_pow2(x.reshape(-1), length),
                 _pad_pow2(self_coeff, length), eq]),
            (1, [_pad_pow2(paired, length),
                 _pad_pow2(pair_coeff, length), eq]),
        ]
        if claim.rescale_bits == 0:
            return "rope-rotation", terms
        gamma, delta = protocol.op_vec(
            challenge, claim_index, "sampled-rope-rescale-batch", 2)
        rotated = _pad_pow2(live[claim.x_rot], length)
        low = _pad_pow2(live[claim.x_rot_low], length)
        shifted = _pad_pow2(live[claim.x_rot_shifted], length)
        offset = torch.zeros(size, dtype=torch.uint64, device=eq.device)
        offset[:length] = 1 << (claim.output_width - 1)
        terms.extend([
            (gamma, [target, eq]),
            ((P - gamma * (1 << claim.rescale_bits) % P) % P,
             [rotated, eq]),
            ((P - gamma) % P, [low, eq]),
            (delta, [shifted, eq]),
            ((P - delta) % P, [rotated, eq]),
            ((P - delta) % P, [offset, eq]),
        ])
        return "rope-rotation-rescale", terms

    if isinstance(claim, RoutingClaim):
        t_count, experts = claim.T, claim.E
        r = live[claim.r].reshape(t_count, experts)
        m = live[claim.m].reshape(t_count, experts)
        rt = live[claim.rt].reshape(t_count, experts)
        mrt = live[claim.mrt].reshape(t_count, experts)
        gap = live[claim.gap].reshape(t_count, experts)
        rstar = live[claim.rstar].reshape(t_count)
        r_chosen = live[claim.r_chosen].reshape(t_count)
        ones_e = torch.ones(experts, dtype=torch.uint64, device=r.device)
        bonus_e = torch.arange(
            experts - 1, -1, -1, dtype=torch.int64,
            device=r.device).to(torch.uint64)
        bonus = bonus_e.view(1, experts).expand_as(r).contiguous()
        rstar_broadcast = rstar.view(
            t_count, 1).expand_as(rt).contiguous()
        scale = torch.full_like(r, 1 << claim.L_bits)
        residuals = [
            gl_sub(gl_sub(rt, gl_mul(scale, r)), bonus),
            gl_sub(gl_matvec(m.contiguous(), ones_e),
                   torch.ones(t_count, dtype=torch.uint64, device=r.device)),
            gl_sub(gl_matvec(mrt.contiguous(), ones_e), rstar),
            gl_sub(gl_add(gap, rt), rstar_broadcast),
            gl_sub(gl_add(
                gl_mul(torch.full_like(r_chosen, 1 << claim.L_bits),
                       r_chosen),
                gl_matvec(gl_mul(m, bonus).contiguous(),
                          ones_e)), rstar),
            gl_sub(gl_mul(m, m), m),
            gl_sub(gl_mul(m, rt), mrt),
        ]
        gammas = protocol.op_vec(
            challenge, claim_index, "sampled-routing-relation-batch", 7)
        aggregate = torch.zeros(
            length, dtype=torch.uint64, device=r.device)
        for gamma, residual in zip(gammas, residuals):
            flat = residual.reshape(-1)
            padded = torch.zeros_like(aggregate)
            padded[:flat.numel()] = flat
            aggregate = gl_add(
                aggregate,
                gl_mul(torch.full_like(aggregate, gamma), padded))
        return "routing-all-relations", [
            (1, [_pad_pow2(aggregate, length), eq])]

    if isinstance(claim, SiluClaim):
        cfg = claim.config

        def scale(value, scalar):
            return gl_mul(torch.full_like(value, scalar % P), value)

        x = live[claim.x].reshape(-1)
        sign = live[claim.sign].reshape(-1)
        magnitude = live[claim.magnitude].reshape(-1)
        c_value = live[claim.C].reshape(-1)
        a0 = live[claim.a_0].reshape(-1)
        a1 = live[claim.a_1].reshape(-1)
        a2 = live[claim.a_2].reshape(-1)
        a3 = live[claim.a_3].reshape(-1)
        a4 = live[claim.a_4].reshape(-1)
        g = live[claim.g].reshape(-1)
        inv_g = live[claim.inv_g].reshape(-1)
        is_high = live[claim.is_high].reshape(-1)
        key = live[claim.key].reshape(-1)
        output_sat = live[claim.output_sat].reshape(-1)
        mux_a = live[claim.mux_a].reshape(-1)
        mux_b = live[claim.mux_b].reshape(-1)
        lookup_y = live[claim.y].reshape(-1)
        output = live[claim.output].reshape(-1)
        residuals = [
            gl_sub(gl_sub(x, magnitude), scale(c_value, 2)),
            gl_sub(gl_sub(gl_sub(gl_sub(gl_sub(
                magnitude, a0), scale(a1, cfg.b)),
                scale(a2, cfg.b_2)), scale(a3, cfg.b_3)),
                scale(a4, cfg.b_4)),
            gl_sub(gl_sub(gl_sub(
                g, scale(a2, cfg.b_2)), scale(a3, cfg.b_3)),
                scale(a4, cfg.b_4)),
            gl_sub(gl_sub(key, scale(sign, cfg.T_LEN)), a1),
            gl_sub(gl_sub(x, output_sat), c_value),
            gl_add(gl_sub(gl_sub(lookup_y, output), mux_a), mux_b),
            gl_sub(gl_mul(sign, sign), sign),
            gl_sub(gl_mul(sign, x), c_value),
            gl_sub(gl_mul(g, inv_g), is_high),
            gl_sub(gl_mul(is_high, g), g),
            gl_sub(gl_mul(is_high, is_high), is_high),
            gl_sub(gl_mul(is_high, lookup_y), mux_a),
            gl_sub(gl_mul(is_high, output_sat), mux_b),
        ]
        if cfg.rescale_bits > 0:
            x_in = live[claim.x_in].reshape(-1)
            x_low = live[claim.x_low].reshape(-1)
            x_shifted = live[claim.x_shifted].reshape(-1)
            residuals.extend([
                gl_sub(gl_sub(x_in, scale(x, 1 << cfg.rescale_bits)),
                       x_low),
                gl_sub(gl_sub(x_shifted, x), torch.full_like(
                    x, 1 << (cfg.width_2 - 1))),
            ])
        gammas = protocol.op_vec(
            challenge, claim_index, "sampled-silu-relation-batch",
            len(residuals))
        aggregate = torch.zeros_like(x)
        for gamma, residual in zip(gammas, residuals):
            aggregate = gl_add(aggregate, scale(residual, gamma))
        return "silu-all-relations", [
            (1, [_pad_pow2(aggregate, length), eq])]

    if isinstance(claim, RmsNormClaim):
        cfg = claim.config
        batch, width = cfg.B, cfg.d

        def scale(value, scalar):
            return gl_mul(torch.full_like(value, scalar % P), value)

        def recompose(values, stride_bits):
            acc = torch.zeros(
                batch, dtype=torch.uint64, device=live[claim.x].device)
            for index, value in enumerate(values):
                acc = gl_add(
                    acc, scale(live[value].reshape(-1),
                               1 << (stride_bits * index)))
            return acc

        x = live[claim.x].reshape(batch, width)
        out_var = (claim.output_full if cfg.output_rescale_bits > 0
                   else claim.output)
        out = live[out_var].reshape(batch, width)
        x_sq = live[claim.X_sq].reshape(batch, width)
        s_value = live[claim.S].reshape(-1)
        s_total = live[claim.S_total].reshape(-1)
        y = live[claim.y].reshape(-1)
        y_m1 = live[claim.y_m1].reshape(-1)
        q1 = live[claim.q1].reshape(-1)
        q2 = live[claim.q2].reshape(-1)
        s_lo = live[claim.s_lo].reshape(-1)
        s_hi = live[claim.s_hi].reshape(-1)
        ones_d = torch.ones(width, dtype=torch.uint64, device=x.device)
        ones_b = torch.ones(batch, dtype=torch.uint64, device=x.device)
        residuals = [
            gl_sub(gl_sub(s_total, s_value),
                   torch.full_like(s_total, width * cfg.eps_int)),
            gl_add(gl_sub(y_m1, y), ones_b),
            gl_sub(s_value, gl_matvec(x_sq.contiguous(), ones_d)),
            gl_sub(s_lo, recompose(claim.s_lo_chunks, 16)),
            gl_sub(s_hi, recompose(claim.s_hi_chunks, 16)),
            gl_sub(y_m1, recompose(claim.ym1_chunks, 16)),
            gl_sub(s_total, recompose(claim.S_limbs, RMS_LIMB_W)),
        ]

        def chunk_value(values):
            return recompose(values, 16)

        def bracket_residuals(H, lows, g0_chunks, g1_chunks, g2_chunks,
                              slack, upper):
            g0h = chunk_value(g0_chunks)
            g1h = chunk_value(g1_chunks)
            g2 = chunk_value(g2_chunks)
            h = [live[var].reshape(-1) for var in H]
            glows = [live[var].reshape(-1) for var in lows]
            result = [
                gl_sub(gl_sub(h[0], glows[0]),
                       scale(g0h, 1 << RMS_LIMB_W)),
                gl_sub(gl_sub(gl_add(h[1], g0h), glows[1]),
                       scale(g1h, 1 << RMS_LIMB_W)),
                gl_sub(gl_add(h[2], g1h), g2),
            ]
            final = gl_add(
                gl_add(scale(g2, 1 << (2 * RMS_LIMB_W)),
                       scale(glows[1], 1 << RMS_LIMB_W)), glows[0])
            if upper:
                final = gl_sub(gl_add(final, slack),
                               torch.full_like(final, cfg.magic - 1))
            else:
                final = gl_sub(gl_sub(final, slack),
                               torch.full_like(final, cfg.magic))
            result.append(final)
            return result

        residuals.extend(bracket_residuals(
            claim.lo_H, claim.lo_gl, claim.lo_g0h_chunks,
            claim.lo_g1h_chunks, claim.lo_G2_chunks, s_lo, False))
        residuals.extend(bracket_residuals(
            claim.hi_H, claim.hi_gl, claim.hi_g0h_chunks,
            claim.hi_g1h_chunks, claim.hi_G2_chunks, s_hi, True))
        residuals.extend([
            gl_sub(gl_mul(x.reshape(-1), x.reshape(-1)),
                   x_sq.reshape(-1)),
            gl_sub(gl_mul(y, y), q1),
            gl_sub(gl_mul(y_m1, y_m1), q2),
        ])
        for limb, lo_h, hi_h in zip(
                claim.S_limbs, claim.lo_H, claim.hi_H):
            limb_value = live[limb].reshape(-1)
            residuals.append(gl_sub(
                gl_mul(q1, limb_value), live[lo_h].reshape(-1)))
            residuals.append(gl_sub(
                gl_mul(q2, limb_value), live[hi_h].reshape(-1)))
        rho_t = _cuda_vec(protocol.op_vec(
            challenge, claim_index, "sampled-rmsnorm-freivalds-rho", width))
        x_projection = gl_matvec(x.contiguous(), rho_t)
        out_projection = gl_matvec(out.contiguous(), rho_t)
        residuals.append(gl_sub(gl_mul(y, x_projection), out_projection))
        if cfg.rescale_bits > 0:
            x_flat = x.reshape(-1)
            residuals.extend([
                gl_sub(gl_sub(live[claim.x_in].reshape(-1),
                              scale(x_flat, 1 << cfg.rescale_bits)),
                       live[claim.x_low].reshape(-1)),
                gl_sub(gl_sub(live[claim.x_shifted].reshape(-1), x_flat),
                       torch.full_like(x_flat, 1 << 15)),
            ])
        if cfg.output_rescale_bits > 0:
            output = live[claim.output].reshape(-1)
            residuals.extend([
                gl_sub(gl_sub(live[claim.output_full].reshape(-1),
                              scale(output, 1 << cfg.output_rescale_bits)),
                       live[claim.output_low].reshape(-1)),
                gl_sub(gl_sub(live[claim.output_shifted].reshape(-1), output),
                       torch.full_like(
                           output, 1 << (cfg.output_width - 1))),
            ])
        gammas = protocol.op_vec(
            challenge, claim_index, "sampled-rmsnorm-relation-batch",
            len(residuals))
        aggregate = torch.zeros(length, dtype=torch.uint64, device=x.device)
        for gamma, residual in zip(gammas, residuals):
            flat = residual.reshape(-1)
            count = flat.numel()
            aggregate[:count] = gl_add(
                aggregate[:count].contiguous(), scale(flat, gamma))
        return "rmsnorm-all-relations", [
            (1, [_pad_pow2(aggregate, length), eq])]

    if isinstance(claim, SoftmaxClaim):
        cfg = claim.config
        batch, width = cfg.B, cfg.M

        def scale(value, scalar):
            return gl_mul(torch.full_like(value, scalar % P), value)

        x = live[claim.x].reshape(batch, width)
        c2 = live[claim.c2].reshape(-1)
        c2_broadcast = c2.view(batch, 1).expand_as(x).contiguous()
        z = live[claim.z].reshape(batch, width)
        y_a = live[claim.y_A].reshape(batch, width)
        y_b = live[claim.y_B].reshape(batch, width)
        s1 = live[claim.s1].reshape(-1)
        s2 = live[claim.s2].reshape(-1)
        r_lo = live[claim.r_lo].reshape(-1)
        r_hi = live[claim.r_hi].reshape(-1)
        ones_m = torch.ones(width, dtype=torch.uint64, device=x.device)
        z_full = z
        if cfg.saturate:
            z_full = gl_add(z, scale(
                live[claim.z_high].reshape(batch, width), cfg.Z_max))
        z_residual = gl_add(gl_sub(z_full, c2_broadcast), x)
        if cfg.causal:
            positions = torch.arange(
                claim.length, dtype=torch.int64, device=x.device)
            row = positions // width
            column = positions % width
            query = row // cfg.heads
            active = (column <= query).to(torch.uint64).reshape(batch, width)
            z_residual = gl_mul(z_residual, active)
        residuals = [
            z_residual.reshape(-1),
            gl_sub(s1, gl_matvec(y_a.contiguous(), ones_m)),
            gl_sub(s2, gl_matvec(y_b.contiguous(), ones_m)),
            gl_sub(gl_add(s1, r_lo), torch.full_like(s1, cfg.s_y)),
            gl_add(gl_sub(r_hi, s2), torch.full_like(r_hi, cfg.s_y + 1)),
            gl_sub(gl_sub(live[claim.c2_shifted].reshape(-1), c2),
                   torch.full_like(c2, 1 << (cfg.aux_chunk_width - 1))),
        ]
        if cfg.saturate:
            is_high = live[claim.is_high].reshape(-1)
            z_high = live[claim.z_high].reshape(-1)
            y_a_raw = live[claim.y_A_raw].reshape(-1)
            y_b_raw = live[claim.y_B_raw].reshape(-1)
            mux_a = live[claim.mux_y_A].reshape(-1)
            mux_b = live[claim.mux_y_B].reshape(-1)
            line_a = gl_sub(gl_sub(y_a_raw, y_a.reshape(-1)), mux_a)
            line_b = gl_sub(gl_sub(y_b_raw, y_b.reshape(-1)), mux_b)
            if cfg.round_up:
                line_a = gl_add(line_a, is_high)
                line_b = gl_add(line_b, is_high)
            residuals.extend([
                line_a, line_b,
                gl_sub(gl_mul(z_high, live[claim.inv_z_high].reshape(-1)),
                       is_high),
                gl_sub(gl_mul(is_high, z_high), z_high),
                gl_sub(gl_mul(is_high, is_high), is_high),
                gl_sub(gl_mul(is_high, y_a_raw), mux_a),
                gl_sub(gl_mul(is_high, y_b_raw), mux_b),
            ])
        if cfg.rescale_bits > 0:
            x_flat = x.reshape(-1)
            residuals.extend([
                gl_sub(gl_sub(live[claim.x_in].reshape(-1),
                              scale(x_flat, 1 << cfg.rescale_bits)),
                       live[claim.x_low].reshape(-1)),
                gl_sub(gl_sub(live[claim.x_shifted].reshape(-1), x_flat),
                       torch.full_like(x_flat, 1 << 15)),
            ])
        gammas = protocol.op_vec(
            challenge, claim_index, "sampled-softmax-relation-batch",
            len(residuals))
        aggregate = torch.zeros(length, dtype=torch.uint64, device=x.device)
        for gamma, residual in zip(gammas, residuals):
            flat = residual.reshape(-1)
            count = flat.numel()
            aggregate[:count] = gl_add(
                aggregate[:count].contiguous(), scale(flat, gamma))
        return "softmax-all-relations", [
            (1, [_pad_pow2(aggregate, length), eq])]

    if isinstance(claim, MaxClaim):
        batch, width = claim.T, claim.V
        logits = live[claim.l].reshape(batch, width)
        a_mask = live[claim.A].reshape(batch, width)
        al = live[claim.Al].reshape(batch, width)
        vstar = live[claim.vstar].reshape(-1)
        gap = live[claim.gap].reshape(batch, width)
        neg_gap = live[claim.neg_gap].reshape(batch, width)
        output_mask = live[claim.O].reshape(batch, width)
        output_gap = live[claim.Ogap].reshape(batch, width)
        gap_o = live[claim.gap_o].reshape(-1)
        token = live[claim.tok].reshape(-1)
        ones = torch.ones(width, dtype=torch.uint64, device=logits.device)
        indices = torch.arange(
            width, dtype=torch.int64, device=logits.device).to(torch.uint64)
        vstar_bc = vstar.view(batch, 1).expand_as(logits).contiguous()
        residuals = [
            gl_sub(gl_mul(a_mask, a_mask), a_mask),
            gl_sub(gl_mul(a_mask, logits), al),
            gl_sub(gl_mul(output_mask, output_mask), output_mask),
            gl_sub(gl_mul(output_mask, gap), output_gap),
            gl_sub(gl_matvec(a_mask.contiguous(), ones),
                   torch.ones(batch, dtype=torch.uint64,
                              device=logits.device)),
            gl_sub(gl_matvec(al.contiguous(), ones), vstar),
            gl_sub(gl_add(gap, logits), vstar_bc),
            gl_add(neg_gap, gap),
            gl_sub(gl_matvec(output_mask.contiguous(), ones),
                   torch.ones(batch, dtype=torch.uint64,
                              device=logits.device)),
            gl_sub(gl_matvec(output_gap.contiguous(), ones), gap_o),
            gl_sub(gl_matvec(
                gl_mul(output_mask, indices.view(1, width).expand_as(
                    output_mask).contiguous()).contiguous(), ones), token),
        ]
        gammas = protocol.op_vec(
            challenge, claim_index, "sampled-max-relation-batch",
            len(residuals))
        aggregate = torch.zeros(length, dtype=torch.uint64,
                                device=logits.device)
        for gamma, residual in zip(gammas, residuals):
            flat = residual.reshape(-1)
            count = flat.numel()
            weighted = gl_mul(torch.full_like(flat, gamma), flat)
            aggregate[:count] = gl_add(
                aggregate[:count].contiguous(), weighted)
        return "max-all-relations", [
            (1, [_pad_pow2(aggregate, length), eq])]

    if isinstance(claim, InfoFinalizeClaim):
        batch, width = claim.T, claim.V
        e = live[claim.e].reshape(batch, width)
        a_value = live[claim.a].reshape(-1)
        d_value = live[claim.d].reshape(-1)
        pw = live[claim.pw].reshape(-1)
        z_o = live[claim.z_o].reshape(-1)
        gap_o2 = live[claim.gap_o2].reshape(-1)
        rem = live[claim.rem].reshape(-1)
        surprisal = live[claim.surprisal].reshape(-1)
        b_value = live[claim.b].reshape(-1)
        ones = torch.ones(width, dtype=torch.uint64, device=e.device)
        words = torch.zeros(batch, dtype=torch.uint64, device=e.device)
        for index, var in enumerate(claim.dw):
            value = live[var].reshape(-1)
            words = gl_add(words, gl_mul(
                torch.full_like(value, 1 << (claim.wb * index)), value))
        residuals = [
            gl_sub(gl_matvec(e.contiguous(), ones), a_value),
            gl_sub(gl_add(a_value, d_value), pw),
            gl_sub(d_value, words),
            gl_sub(gl_sub(gl_mul(torch.full_like(z_o, claim.k), z_o),
                          gap_o2), rem),
            gl_sub(gl_sub(surprisal, z_o), b_value),
        ]
        gammas = protocol.op_vec(
            challenge, claim_index, "sampled-info-relation-batch",
            len(residuals))
        aggregate = torch.zeros(length, dtype=torch.uint64, device=e.device)
        for gamma, residual in zip(gammas, residuals):
            aggregate = gl_add(aggregate, gl_mul(
                torch.full_like(residual, gamma), residual))
        return "info-finalize-all-relations", [
            (1, [_pad_pow2(aggregate, length), eq])]

    # Batch the two public rescale linears with an independent random tag so
    # errors in one cannot cancel errors in the other except with 1/|F| chance.
    gamma = protocol.op_vec(
        challenge, claim_index, "sampled-rescale-batch", 1)[0]
    x_full = live[claim.x_full].detach().contiguous().view(-1)
    x = live[claim.x].detach().contiguous().view(-1)
    low = live[claim.x_low].detach().contiguous().view(-1)
    shifted = live[claim.x_shifted].detach().contiguous().view(-1)
    rescale = gl_sub(
        gl_sub(x_full, _scale(x, 1 << claim.rescale_bits)), low)
    shift = gl_sub(
        gl_sub(shifted, x),
        torch.full_like(x, 1 << (claim.output_width - 1)))
    aggregate = _batch_residuals(
        [rescale, shift], [1, gamma], length)
    return "rescale-linear", [
        (1, [_pad_pow2(aggregate, length), eq]),
    ]


def _sumcheck_transcript(challenge: bytes, claim_index: int,
                         relation: str) -> sc.RoundTranscript:
    """Per-round coins: each round's challenge follows its polynomial, bound
    to the post-commit block challenge (which hashes the window root over the
    block commitments), the claim and the relation. Expanding every round's
    coin from the challenge up front let a prover steer a false claim onto
    the true terminal value."""
    return sc.RoundTranscript(b"verinf/sampled-local/relation/v1", bytes(challenge),
                              int(claim_index).to_bytes(8, "little"),
                              relation.encode())


def prove_sumcheck(claim: object, live: dict, *, claim_index: int,
                   challenge: bytes) -> RelationSumcheckProof:
    relation, terms = _relation_terms(
        claim, live, claim_index=claim_index, challenge=challenge)
    work = len(terms[0][1][0]) * sum(len(factors)
                                     for _coef, factors in terms)
    if work < sc.GPU_MIN_SUMCHECK_WORK:
        prover_terms = [(coef, [factor.cpu().tolist() for factor in factors])
                        for coef, factors in terms]
    else:
        prover_terms = terms
    proof = sc.prove_terms(
        prover_terms, _sumcheck_transcript(challenge, claim_index, relation))
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
    # the local argument is unmasked: a carried mask would be a free term
    if proof.sumcheck.masked:
        return False, "masked transcript in an unmasked local argument"
    if proof.sumcheck.claim % P != 0:
        return False, "sumcheck relation claim is not zero"
    ok, why = sc.verify_terms(proof.sumcheck, terms,
                              _sumcheck_transcript(challenge, claim_index, relation))
    return ok, why if not ok else "ok"


@dataclass
class RoPESumcheckProof:
    """Rotation/rescale sumcheck plus both internal range product roots."""

    relation: RelationSumcheckProof
    low_query_root: int = 1
    low_table_root: int = 1
    shifted_query_root: int = 1
    shifted_table_root: int = 1

    @property
    def byte_size(self) -> int:
        extra = 32 if self.relation.relation.endswith("-rescale") else 0
        return self.relation.byte_size + extra

    @property
    def digest(self) -> bytes:
        h = blake3.blake3(b"verinf/sampled/rope-sumcheck/v1")
        h.update(self.relation.digest)
        for root in (self.low_query_root, self.low_table_root,
                     self.shifted_query_root, self.shifted_table_root):
            h.update(int(root).to_bytes(8, "little"))
        return h.digest()


def _range_value_products(value: torch.Tensor, table: torch.Tensor,
                          alpha: int):
    flat = value.detach().contiguous().view(-1)
    indices = flat.view(torch.int64)
    public = table.detach().contiguous().view(-1)
    safe = indices.clamp(0, public.numel() - 1)
    indexed = public.index_select(0, safe)
    alpha_t = torch.full_like(flat, alpha)
    return (_product_tree_root(gl_sub(alpha_t, flat)),
            _product_tree_root(gl_sub(alpha_t, indexed)), indices,
            public.numel())


@dataclass
class RoundingSumcheckProof:
    """Signed-floor linears plus compact roots for both range relations."""

    relation: RelationSumcheckProof
    low_query_root: int
    low_table_root: int
    shifted_query_root: int
    shifted_table_root: int

    @property
    def byte_size(self) -> int:
        return self.relation.byte_size + 32

    @property
    def digest(self) -> bytes:
        h = blake3.blake3(b"verinf/sampled/rounding-sumcheck/v1")
        h.update(self.relation.digest)
        for root in (self.low_query_root, self.low_table_root,
                     self.shifted_query_root, self.shifted_table_root):
            h.update(int(root).to_bytes(8, "little"))
        return h.digest()


@dataclass
class MatmulAuditProof:
    """Raw-product Freivalds proof and its signed-floor output proof."""

    freivalds: MatmulFreivaldsProof
    rounding: RoundingSumcheckProof

    @property
    def byte_size(self) -> int:
        return self.freivalds.byte_size + self.rounding.byte_size

    @property
    def digest(self) -> bytes:
        return blake3.blake3(
            b"verinf/sampled/matmul-audit/v1"
            + self.freivalds.digest + self.rounding.digest).digest()


def _rounding_parts(claim: object):
    if isinstance(claim, MatmulClaim):
        return (claim.C_full, claim.C, claim.C_low, claim.C_shifted,
                claim.range_rescale, claim.range_output, claim.C.length,
                claim.rescale_bits, claim.output_width)
    if isinstance(claim, HadamardClaim):
        return (claim.c_full, claim.c, claim.c_low, claim.c_shifted,
                claim.range_rescale, claim.range_output, claim.length,
                claim.rescale_bits, claim.output_width)
    if isinstance(claim, RescaleClaim):
        return (claim.x_full, claim.x, claim.x_low, claim.x_shifted,
                claim.range_rescale, claim.range_output, claim.length,
                claim.rescale_bits, claim.output_width)
    raise TypeError(f"no rounding proof for {type(claim).__name__}")


def _matmul_rounding_terms(claim: MatmulClaim, live: dict, *,
                           claim_index: int, challenge: bytes):
    (full_var, out_var, low_var, shifted_var, _low_table, _out_table,
     length, rescale_bits, output_width) = _rounding_parts(claim)
    size = 1 << max(0, (length - 1).bit_length())
    rounds = size.bit_length() - 1
    tau = protocol.op_vec(
        challenge, claim_index, "sampled-sumcheck-eq", rounds)
    eq = _eq_weights(tau, live[full_var].device)
    gamma = protocol.op_vec(
        challenge, claim_index, "sampled-matmul-rescale-batch", 1)[0]
    full = live[full_var].detach().contiguous().view(-1)
    out = live[out_var].detach().contiguous().view(-1)
    low = live[low_var].detach().contiguous().view(-1)
    shifted = live[shifted_var].detach().contiguous().view(-1)
    rescale = gl_sub(gl_sub(full, _scale(
        out, 1 << rescale_bits)), low)
    shift = gl_sub(
        gl_sub(shifted, out),
        torch.full_like(out, 1 << (output_width - 1)))
    aggregate = _batch_residuals(
        [rescale, shift], [1, gamma], length)
    return "matmul-rescale", [
        (1, [_pad_pow2(aggregate, length), eq]),
    ]


def _prove_explicit_relation(
        relation: str, terms, *, claim_index: int,
        challenge: bytes) -> RelationSumcheckProof:
    work = len(terms[0][1][0]) * sum(len(factors)
                                     for _coef, factors in terms)
    if work < sc.GPU_MIN_SUMCHECK_WORK:
        prover_terms = [(coef, [factor.cpu().tolist() for factor in factors])
                        for coef, factors in terms]
    else:
        prover_terms = terms
    return RelationSumcheckProof(
        claim_index, bytes(challenge), relation,
        sc.prove_terms(prover_terms,
                       _sumcheck_transcript(challenge, claim_index, relation)))


def _verify_explicit_relation(
        proof: RelationSumcheckProof, relation: str, terms, *,
        claim_index: int, challenge: bytes) -> tuple[bool, str]:
    if proof.claim_index != claim_index or proof.challenge != challenge:
        return False, "rounding sumcheck transcript challenge mismatch"
    if proof.relation != relation:
        return False, "rounding sumcheck relation mismatch"
    if proof.sumcheck.masked:
        return False, "masked transcript in an unmasked local argument"
    if proof.sumcheck.claim % P != 0:
        return False, "rounding relation claim is not zero"
    ok, why = sc.verify_terms(proof.sumcheck, terms,
                              _sumcheck_transcript(challenge, claim_index, relation))
    return ok, why if not ok else "ok"


def prove_rounding_sumcheck(
        claim: object, live: dict, *, claim_index: int,
        challenge: bytes) -> RoundingSumcheckProof:
    (_full, _out, low_var, shifted_var, low_table, output_table,
     _length, rescale_bits, _output_width) = _rounding_parts(claim)
    if rescale_bits <= 0:
        raise ValueError("rounding proof requires rescale_bits > 0")
    if isinstance(claim, MatmulClaim):
        relation_name, terms = _matmul_rounding_terms(
            claim, live, claim_index=claim_index, challenge=challenge)
        relation = _prove_explicit_relation(
            relation_name, terms, claim_index=claim_index,
            challenge=challenge)
    else:
        relation = prove_sumcheck(
            claim, live, claim_index=claim_index, challenge=challenge)
    alpha_low, alpha_shifted = protocol.op_vec(
        challenge, claim_index, "sampled-rounding-range-alpha", 2)
    low_query, low_public, _low_indices, _low_len = _range_value_products(
        live[low_var], low_table.T, alpha_low)
    shifted_query, shifted_public, _shifted_indices, _shifted_len = (
        _range_value_products(
            live[shifted_var], output_table.T, alpha_shifted))
    return RoundingSumcheckProof(
        relation, low_query, low_public, shifted_query, shifted_public)


def verify_rounding_sumcheck(
        claim: object, live: dict, proof: RoundingSumcheckProof, *,
        claim_index: int, challenge: bytes) -> tuple[bool, str]:
    (_full, _out, low_var, shifted_var, low_table, output_table,
     _length, _rescale_bits, _output_width) = _rounding_parts(claim)
    if isinstance(claim, MatmulClaim):
        relation_name, terms = _matmul_rounding_terms(
            claim, live, claim_index=claim_index, challenge=challenge)
        ok, why = _verify_explicit_relation(
            proof.relation, relation_name, terms, claim_index=claim_index,
            challenge=challenge)
    else:
        ok, why = verify_sumcheck(
            claim, live, proof.relation, claim_index=claim_index,
            challenge=challenge)
    if not ok:
        return ok, why
    alpha_low, alpha_shifted = protocol.op_vec(
        challenge, claim_index, "sampled-rounding-range-alpha", 2)
    low_query, low_public, low_indices, low_len = _range_value_products(
        live[low_var], low_table.T, alpha_low)
    shifted_query, shifted_public, shifted_indices, shifted_len = (
        _range_value_products(
            live[shifted_var], output_table.T, alpha_shifted))
    if not bool(((low_indices >= 0) & (low_indices < low_len)).all().item()):
        return False, "rounding low value is outside the public table"
    if not bool(((shifted_indices >= 0)
                 & (shifted_indices < shifted_len)).all().item()):
        return False, "rounding shifted value is outside the public table"
    actual = (low_query, low_public, shifted_query, shifted_public)
    message = (proof.low_query_root, proof.low_table_root,
               proof.shifted_query_root, proof.shifted_table_root)
    if actual != message:
        return False, "rounding range roots are not witness-bound"
    if low_query != low_public or shifted_query != shifted_public:
        return False, "rounding range product-tree roots differ"
    return True, "ok"


def prove_rope_sumcheck(
        claim: RoPEClaim, live: dict, *, claim_index: int,
        challenge: bytes) -> RoPESumcheckProof:
    relation = prove_sumcheck(
        claim, live, claim_index=claim_index, challenge=challenge)
    if claim.rescale_bits == 0:
        return RoPESumcheckProof(relation)
    alpha_low, alpha_shifted = protocol.op_vec(
        challenge, claim_index, "sampled-rope-range-alpha", 2)
    low_query, low_table, _low_indices, _low_len = _range_value_products(
        live[claim.x_rot_low], claim.range_rescale.T, alpha_low)
    shifted_query, shifted_table, _shifted_indices, _shifted_len = (
        _range_value_products(
            live[claim.x_rot_shifted], claim.range_output.T, alpha_shifted))
    return RoPESumcheckProof(
        relation, low_query, low_table, shifted_query, shifted_table)


def verify_rope_sumcheck(
        claim: RoPEClaim, live: dict, proof: RoPESumcheckProof, *,
        claim_index: int, challenge: bytes) -> tuple[bool, str]:
    ok, why = verify_sumcheck(
        claim, live, proof.relation, claim_index=claim_index,
        challenge=challenge)
    if not ok or claim.rescale_bits == 0:
        return ok, why
    alpha_low, alpha_shifted = protocol.op_vec(
        challenge, claim_index, "sampled-rope-range-alpha", 2)
    low_query, low_table, low_indices, low_len = _range_value_products(
        live[claim.x_rot_low], claim.range_rescale.T, alpha_low)
    shifted_query, shifted_table, shifted_indices, shifted_len = (
        _range_value_products(
            live[claim.x_rot_shifted], claim.range_output.T, alpha_shifted))
    if not bool(((low_indices >= 0) & (low_indices < low_len)).all().item()):
        return False, "RoPE low range index is outside the public table"
    if not bool(((shifted_indices >= 0)
                 & (shifted_indices < shifted_len)).all().item()):
        return False, "RoPE shifted range index is outside the public table"
    actual = (low_query, low_table, shifted_query, shifted_table)
    message = (proof.low_query_root, proof.low_table_root,
               proof.shifted_query_root, proof.shifted_table_root)
    if actual != message:
        return False, "RoPE range product roots are not witness-bound"
    if low_query != low_table or shifted_query != shifted_table:
        return False, "RoPE range product-tree roots differ"
    return True, "ok"


@dataclass
class SiluSumcheckProof:
    """All algebraic SiLU relations plus its range/paired lookups."""

    relation: RelationSumcheckProof
    range_roots: tuple[tuple[int, int], ...]
    paired_query_root: int
    paired_table_root: int

    @property
    def byte_size(self) -> int:
        return self.relation.byte_size + 16 * (len(self.range_roots) + 1)

    @property
    def digest(self) -> bytes:
        h = blake3.blake3(b"verinf/sampled/silu-sumcheck/v1")
        h.update(self.relation.digest)
        for query, table in self.range_roots:
            h.update(int(query).to_bytes(8, "little"))
            h.update(int(table).to_bytes(8, "little"))
        h.update(int(self.paired_query_root).to_bytes(8, "little"))
        h.update(int(self.paired_table_root).to_bytes(8, "little"))
        return h.digest()


def _silu_range_items(claim: SiluClaim, live: dict):
    items = [
        (live[claim.a_0], claim.range_b.T),
        (live[claim.a_2], claim.range_w2.T),
        (live[claim.a_3], claim.range_w3.T),
        (live[claim.a_4], claim.range_w4.T),
    ]
    if claim.config.rescale_bits > 0:
        items.extend([
            (live[claim.x_low], claim.range_rescale.T),
            (live[claim.x_shifted], claim.range_x.T),
        ])
    return items


def _silu_paired_products(claim: SiluClaim, live: dict, *,
                          claim_index: int, challenge: bytes):
    alpha, beta, gamma = protocol.op_vec(
        challenge, claim_index, "sampled-silu-paired-challenges", 3)
    key = live[claim.key].detach().contiguous().view(-1)
    value = live[claim.y].detach().contiguous().view(-1)
    indices = key.view(torch.int64)
    table_x = claim.silu_table.T.detach().contiguous().view(-1)
    table_y = claim.silu_table.T_Y.detach().contiguous().view(-1)
    safe = indices.clamp(0, table_x.numel() - 1)
    selected_x = table_x.index_select(0, safe)
    selected_y = table_y.index_select(0, safe)
    positions = torch.arange(
        key.numel(), dtype=torch.int64,
        device=key.device).to(torch.uint64)
    alpha_t = torch.full_like(key, alpha)
    beta_t = torch.full_like(key, beta)
    gamma_t = torch.full_like(key, gamma)
    tags = gl_mul(gamma_t, positions)
    query_fp = gl_add(gl_add(key, gl_mul(beta_t, value)), tags)
    table_fp = gl_add(
        gl_add(selected_x, gl_mul(beta_t, selected_y)), tags)
    return (_product_tree_root(gl_sub(alpha_t, query_fp)),
            _product_tree_root(gl_sub(alpha_t, table_fp)), indices,
            table_x.numel())


def prove_silu_sumcheck(
        claim: SiluClaim, live: dict, *, claim_index: int,
        challenge: bytes) -> SiluSumcheckProof:
    relation = prove_sumcheck(
        claim, live, claim_index=claim_index, challenge=challenge)
    items = _silu_range_items(claim, live)
    alphas = protocol.op_vec(
        challenge, claim_index, "sampled-silu-range-alpha", len(items))
    roots = []
    for (value, table), alpha in zip(items, alphas):
        query, public, _indices, _length = _range_value_products(
            value, table, alpha)
        roots.append((query, public))
    paired_query, paired_table, _indices, _length = _silu_paired_products(
        claim, live, claim_index=claim_index, challenge=challenge)
    return SiluSumcheckProof(
        relation, tuple(roots), paired_query, paired_table)


def verify_silu_sumcheck(
        claim: SiluClaim, live: dict, proof: SiluSumcheckProof, *,
        claim_index: int, challenge: bytes) -> tuple[bool, str]:
    ok, why = verify_sumcheck(
        claim, live, proof.relation, claim_index=claim_index,
        challenge=challenge)
    if not ok:
        return ok, why
    items = _silu_range_items(claim, live)
    if len(proof.range_roots) != len(items):
        return False, "SiLU range product-tree shape mismatch"
    alphas = protocol.op_vec(
        challenge, claim_index, "sampled-silu-range-alpha", len(items))
    actual_roots = []
    for (value, table), alpha in zip(items, alphas):
        query, public, indices, table_len = _range_value_products(
            value, table, alpha)
        if not bool(((indices >= 0) & (indices < table_len)).all().item()):
            return False, "SiLU range index is outside the public table"
        actual_roots.append((query, public))
    if tuple(actual_roots) != proof.range_roots:
        return False, "SiLU range product roots are not witness-bound"
    if any(query != public for query, public in actual_roots):
        return False, "SiLU range product-tree roots differ"
    paired_query, paired_table, indices, table_len = _silu_paired_products(
        claim, live, claim_index=claim_index, challenge=challenge)
    if not bool(((indices >= 0) & (indices < table_len)).all().item()):
        return False, "SiLU lookup index is outside the public table"
    if (paired_query, paired_table) != (
            proof.paired_query_root, proof.paired_table_root):
        return False, "SiLU paired roots are not witness-bound"
    if paired_query != paired_table:
        return False, "SiLU paired product-tree roots differ"
    return True, "ok"


@dataclass
class RmsNormSumcheckProof:
    """RMS bracket/output sumcheck plus all compact range product roots."""

    relation: RelationSumcheckProof
    range_roots: tuple[tuple[int, int], ...]

    @property
    def byte_size(self) -> int:
        return self.relation.byte_size + 16 * len(self.range_roots)

    @property
    def digest(self) -> bytes:
        h = blake3.blake3(b"verinf/sampled/rmsnorm-sumcheck/v1")
        h.update(self.relation.digest)
        for query, table in self.range_roots:
            h.update(int(query).to_bytes(8, "little"))
            h.update(int(table).to_bytes(8, "little"))
        return h.digest()


def _rmsnorm_range_items(claim: RmsNormClaim, live: dict):
    widths = _chunk_widths(claim.config.slack_width)
    items = []
    for values in (claim.s_lo_chunks, claim.s_hi_chunks):
        for index, var in enumerate(values):
            table = (claim.range_slack if widths[index] == 16
                     else claim.range_slack_top)
            items.append((live[var], table.T))
    for variables, _zs, tables in _rms_limb_range_groups(claim):
        items.extend((live[var], table.T)
                     for var, table in zip(variables, tables))
    if claim.config.rescale_bits > 0:
        items.extend([
            (live[claim.x_low], claim.range_rescale.T),
            (live[claim.x_shifted], claim.range_slack.T),
        ])
    if claim.config.output_rescale_bits > 0:
        items.extend([
            (live[claim.output_low], claim.range_output_rescale.T),
            (live[claim.output_shifted], claim.range_output.T),
        ])
    return items


def prove_rmsnorm_sumcheck(
        claim: RmsNormClaim, live: dict, *, claim_index: int,
        challenge: bytes) -> RmsNormSumcheckProof:
    relation = prove_sumcheck(
        claim, live, claim_index=claim_index, challenge=challenge)
    items = _rmsnorm_range_items(claim, live)
    alphas = protocol.op_vec(
        challenge, claim_index, "sampled-rmsnorm-range-alpha", len(items))
    roots = []
    for (value, table), alpha in zip(items, alphas):
        query, public, _indices, _length = _range_value_products(
            value, table, alpha)
        roots.append((query, public))
    return RmsNormSumcheckProof(relation, tuple(roots))


def verify_rmsnorm_sumcheck(
        claim: RmsNormClaim, live: dict, proof: RmsNormSumcheckProof, *,
        claim_index: int, challenge: bytes) -> tuple[bool, str]:
    ok, why = verify_sumcheck(
        claim, live, proof.relation, claim_index=claim_index,
        challenge=challenge)
    if not ok:
        return ok, why
    items = _rmsnorm_range_items(claim, live)
    if len(proof.range_roots) != len(items):
        return False, "RMSNorm range product-tree shape mismatch"
    alphas = protocol.op_vec(
        challenge, claim_index, "sampled-rmsnorm-range-alpha", len(items))
    actual = []
    for (value, table), alpha in zip(items, alphas):
        query, public, indices, table_len = _range_value_products(
            value, table, alpha)
        if not bool(((indices >= 0) & (indices < table_len)).all().item()):
            return False, "RMSNorm range index is outside the public table"
        actual.append((query, public))
    if tuple(actual) != proof.range_roots:
        return False, "RMSNorm range product roots are not witness-bound"
    if any(query != public for query, public in actual):
        return False, "RMSNorm range product-tree roots differ"
    return True, "ok"


@dataclass
class SoftmaxSumcheckProof:
    """Softmax algebraic sumcheck with ranges and two paired lookups."""

    relation: RelationSumcheckProof
    range_roots: tuple[tuple[int, int], ...]
    paired_roots: tuple[tuple[int, int], tuple[int, int]]

    @property
    def byte_size(self) -> int:
        return self.relation.byte_size + 16 * (len(self.range_roots) + 2)

    @property
    def digest(self) -> bytes:
        h = blake3.blake3(b"verinf/sampled/softmax-sumcheck/v1")
        h.update(self.relation.digest)
        for query, table in self.range_roots + self.paired_roots:
            h.update(int(query).to_bytes(8, "little"))
            h.update(int(table).to_bytes(8, "little"))
        return h.digest()


def _softmax_range_items(claim: SoftmaxClaim, live: dict):
    items = [
        (live[claim.c2_shifted], claim.range_aux.T),
        (live[claim.r_lo], claim.range_aux.T),
        (live[claim.r_hi], claim.range_aux.T),
    ]
    if claim.config.saturate:
        items.append((live[claim.z_high], claim.range_z_high.T))
    if claim.config.rescale_bits > 0:
        items.extend([
            (live[claim.x_low], claim.range_rescale.T),
            (live[claim.x_shifted], claim.range_aux.T),
        ])
    return items


def _softmax_lookup_key(claim: SoftmaxClaim, live: dict):
    key = live[claim.z].detach().contiguous().view(-1)
    if not claim.config.causal:
        return key
    cfg = claim.config
    positions = torch.arange(
        claim.length, dtype=torch.int64, device=key.device)
    row = positions // cfg.M
    column = positions % cfg.M
    query = row // cfg.heads
    shift = ((column > query).to(torch.int64) * cfg.Z_max).to(torch.uint64)
    return gl_add(key, shift)


def _softmax_paired_products(
        claim: SoftmaxClaim, live: dict, table, value: torch.Tensor, *,
        claim_index: int, challenge: bytes, label: str):
    alpha, beta, gamma = protocol.op_vec(
        challenge, claim_index, label, 3)
    key = _softmax_lookup_key(claim, live)
    value = value.detach().contiguous().view(-1)
    indices = key.view(torch.int64)
    table_x = table.T.detach().contiguous().view(-1)
    table_y = table.T_Y.detach().contiguous().view(-1)
    safe = indices.clamp(0, table_x.numel() - 1)
    selected_x = table_x.index_select(0, safe)
    selected_y = table_y.index_select(0, safe)
    positions = torch.arange(
        key.numel(), dtype=torch.int64, device=key.device).to(torch.uint64)
    alpha_t = torch.full_like(key, alpha)
    beta_t = torch.full_like(key, beta)
    tags = gl_mul(torch.full_like(key, gamma), positions)
    query_fp = gl_add(gl_add(key, gl_mul(beta_t, value)), tags)
    table_fp = gl_add(
        gl_add(selected_x, gl_mul(beta_t, selected_y)), tags)
    return (_product_tree_root(gl_sub(alpha_t, query_fp)),
            _product_tree_root(gl_sub(alpha_t, table_fp)), indices,
            table_x.numel())


def prove_softmax_sumcheck(
        claim: SoftmaxClaim, live: dict, *, claim_index: int,
        challenge: bytes) -> SoftmaxSumcheckProof:
    relation = prove_sumcheck(
        claim, live, claim_index=claim_index, challenge=challenge)
    items = _softmax_range_items(claim, live)
    alphas = protocol.op_vec(
        challenge, claim_index, "sampled-softmax-range-alpha", len(items))
    ranges = []
    for (value, table), alpha in zip(items, alphas):
        query, public, _indices, _length = _range_value_products(
            value, table, alpha)
        ranges.append((query, public))
    value_a = live[claim.y_A_raw] if claim.config.saturate else live[claim.y_A]
    value_b = live[claim.y_B_raw] if claim.config.saturate else live[claim.y_B]
    pair_a = _softmax_paired_products(
        claim, live, claim.exp_A, value_a, claim_index=claim_index,
        challenge=challenge, label="sampled-softmax-pair-a")[:2]
    pair_b = _softmax_paired_products(
        claim, live, claim.exp_B, value_b, claim_index=claim_index,
        challenge=challenge, label="sampled-softmax-pair-b")[:2]
    return SoftmaxSumcheckProof(relation, tuple(ranges), (pair_a, pair_b))


def verify_softmax_sumcheck(
        claim: SoftmaxClaim, live: dict, proof: SoftmaxSumcheckProof, *,
        claim_index: int, challenge: bytes) -> tuple[bool, str]:
    ok, why = verify_sumcheck(
        claim, live, proof.relation, claim_index=claim_index,
        challenge=challenge)
    if not ok:
        return ok, why
    items = _softmax_range_items(claim, live)
    if len(proof.range_roots) != len(items):
        return False, "Softmax range product-tree shape mismatch"
    alphas = protocol.op_vec(
        challenge, claim_index, "sampled-softmax-range-alpha", len(items))
    actual_ranges = []
    for (value, table), alpha in zip(items, alphas):
        query, public, indices, table_len = _range_value_products(
            value, table, alpha)
        if not bool(((indices >= 0) & (indices < table_len)).all().item()):
            return False, "Softmax range index is outside the public table"
        actual_ranges.append((query, public))
    if tuple(actual_ranges) != proof.range_roots:
        return False, "Softmax range product roots are not witness-bound"
    if any(query != public for query, public in actual_ranges):
        return False, "Softmax range product-tree roots differ"
    value_a = live[claim.y_A_raw] if claim.config.saturate else live[claim.y_A]
    value_b = live[claim.y_B_raw] if claim.config.saturate else live[claim.y_B]
    actual_pairs = []
    for table, value, label in (
            (claim.exp_A, value_a, "sampled-softmax-pair-a"),
            (claim.exp_B, value_b, "sampled-softmax-pair-b")):
        query, public, indices, table_len = _softmax_paired_products(
            claim, live, table, value, claim_index=claim_index,
            challenge=challenge, label=label)
        if not bool(((indices >= 0) & (indices < table_len)).all().item()):
            return False, "Softmax lookup index is outside the public table"
        actual_pairs.append((query, public))
    if tuple(actual_pairs) != proof.paired_roots:
        return False, "Softmax paired roots are not witness-bound"
    if any(query != public for query, public in actual_pairs):
        return False, "Softmax paired product-tree roots differ"
    return True, "ok"


@dataclass
class MaxSumcheckProof:
    relation: RelationSumcheckProof
    gap_query_root: int
    gap_table_root: int

    @property
    def byte_size(self) -> int:
        return self.relation.byte_size + 16

    @property
    def digest(self) -> bytes:
        return blake3.blake3(
            b"verinf/sampled/max-sumcheck/v1" + self.relation.digest
            + int(self.gap_query_root).to_bytes(8, "little")
            + int(self.gap_table_root).to_bytes(8, "little")).digest()


def prove_max_sumcheck(
        claim: MaxClaim, live: dict, *, claim_index: int,
        challenge: bytes) -> MaxSumcheckProof:
    relation = prove_sumcheck(
        claim, live, claim_index=claim_index, challenge=challenge)
    alpha = protocol.op_vec(
        challenge, claim_index, "sampled-max-range-alpha", 1)[0]
    query, public, _indices, _length = _range_value_products(
        live[claim.gap], claim.table.T, alpha)
    return MaxSumcheckProof(relation, query, public)


def verify_max_sumcheck(
        claim: MaxClaim, live: dict, proof: MaxSumcheckProof, *,
        claim_index: int, challenge: bytes) -> tuple[bool, str]:
    ok, why = verify_sumcheck(
        claim, live, proof.relation, claim_index=claim_index,
        challenge=challenge)
    if not ok:
        return ok, why
    alpha = protocol.op_vec(
        challenge, claim_index, "sampled-max-range-alpha", 1)[0]
    query, public, indices, table_len = _range_value_products(
        live[claim.gap], claim.table.T, alpha)
    if not bool(((indices >= 0) & (indices < table_len)).all().item()):
        return False, "Max gap index is outside the public table"
    if (query, public) != (proof.gap_query_root, proof.gap_table_root):
        return False, "Max gap product roots are not witness-bound"
    if query != public:
        return False, "Max gap product-tree roots differ"
    return True, "ok"


@dataclass
class InfoSumcheckProof:
    relation: RelationSumcheckProof
    range_roots: tuple[tuple[int, int], ...]

    @property
    def byte_size(self) -> int:
        return self.relation.byte_size + 16 * len(self.range_roots)

    @property
    def digest(self) -> bytes:
        h = blake3.blake3(b"verinf/sampled/info-sumcheck/v1")
        h.update(self.relation.digest)
        for query, table in self.range_roots:
            h.update(int(query).to_bytes(8, "little"))
            h.update(int(table).to_bytes(8, "little"))
        return h.digest()


def _info_range_items(claim: InfoFinalizeClaim, live: dict):
    return ([(live[var], claim.range_wd.T) for var in claim.dw]
            + [(live[claim.rem], claim.range_k.T)])


def prove_info_sumcheck(
        claim: InfoFinalizeClaim, live: dict, *, claim_index: int,
        challenge: bytes) -> InfoSumcheckProof:
    relation = prove_sumcheck(
        claim, live, claim_index=claim_index, challenge=challenge)
    items = _info_range_items(claim, live)
    alphas = protocol.op_vec(
        challenge, claim_index, "sampled-info-range-alpha", len(items))
    roots = []
    for (value, table), alpha in zip(items, alphas):
        query, public, _indices, _length = _range_value_products(
            value, table, alpha)
        roots.append((query, public))
    return InfoSumcheckProof(relation, tuple(roots))


def verify_info_sumcheck(
        claim: InfoFinalizeClaim, live: dict, proof: InfoSumcheckProof, *,
        claim_index: int, challenge: bytes) -> tuple[bool, str]:
    ok, why = verify_sumcheck(
        claim, live, proof.relation, claim_index=claim_index,
        challenge=challenge)
    if not ok:
        return ok, why
    items = _info_range_items(claim, live)
    if len(proof.range_roots) != len(items):
        return False, "Info range product-tree shape mismatch"
    alphas = protocol.op_vec(
        challenge, claim_index, "sampled-info-range-alpha", len(items))
    actual = []
    for (value, table), alpha in zip(items, alphas):
        query, public, indices, table_len = _range_value_products(
            value, table, alpha)
        if not bool(((indices >= 0) & (indices < table_len)).all().item()):
            return False, "Info range index is outside the public table"
        actual.append((query, public))
    if tuple(actual) != proof.range_roots:
        return False, "Info range product roots are not witness-bound"
    if any(query != public for query, public in actual):
        return False, "Info range product-tree roots differ"
    return True, "ok"


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
