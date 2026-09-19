"""Reproducible GPU preflight for the largest sampled relation sumcheck.

The real Vast statement has T=1,000 and V=202,048, so its LM-head output has
202,048,000 slots and pads to 2^28.  Large relations are challenge-batched into
one residual vector before padding, leaving the two factors ``residual * eq``.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "prover"), str(ROOT)]

import torch  # noqa: E402

from layergkr import sumcheck as sc  # noqa: E402


def run(elements: int = 1 << 28) -> dict:
    if elements <= 0 or elements & (elements - 1):
        raise ValueError("elements must be a positive power of two")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not sc._sumcheck_gpu_ok():
        raise RuntimeError("GPU sumcheck bit-exactness gate rejected")

    zero = torch.zeros(elements, dtype=torch.uint64, device="cuda")
    one = torch.ones(elements, dtype=torch.uint64, device="cuda")
    terms = [(1, [zero, one])]
    rounds = elements.bit_length() - 1
    coins = list(range(1, rounds + 1))

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    proof = sc.prove_terms(terms, lambda index: coins[index])
    torch.cuda.synchronize()
    prove_s = time.perf_counter() - started

    started = time.perf_counter()
    accepted, reason = sc.verify_terms(
        proof, terms, lambda index: coins[index])
    torch.cuda.synchronize()
    verify_s = time.perf_counter() - started
    result = {
        "kind": "sampled-local-proof-preflight-v1",
        "gpu": torch.cuda.get_device_name(),
        "elements": elements,
        "maverick_tokens": 1_000,
        "maverick_vocab": 202_048,
        "maverick_lm_output_elements": 1_000 * 202_048,
        "factor_occurrences": sum(len(factors) for _coef, factors in terms),
        "rounds": rounds,
        "prove_s": prove_s,
        "verify_s": verify_s,
        "accepted": accepted,
        "reason": reason,
        "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    if not accepted:
        raise RuntimeError(f"preflight verifier rejected: {reason}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--elements", type=int, default=1 << 28)
    args = parser.parse_args()
    print(json.dumps(run(args.elements), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
