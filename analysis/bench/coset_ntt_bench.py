#!/usr/bin/env python3
"""Isolated microbenchmark: does the notebook's coset-NTT encoding trick
(analysis/verification-parameter-analysis.ipynb, `Config.coset_ntt`) actually
save time on this hardware?

Confirmed this session: coset-NTT is NOT implemented anywhere in the real
prover (prover/core.py's `_coset_encode_codewords` does one length-N=rho*K
NTT; grep for "coset_ntt" outside the notebook returns nothing). Testing it
for real means implementing the encoding, not flipping a flag -- done here,
but scoped as an ISOLATED primitive-level benchmark against the existing
reference encoder, not wired into the real prove() pipeline. That's a
deliberate scope limit: it answers "is the intervention effective on this
GPU" without touching soundness-critical code paths.

The math (derived from the existing `_coset_encode_codewords`/`coset_shift`/
`W_N` machinery -- nothing new, just recombined): evaluating a degree-<K
polynomial at N=rho*K points {gamma*W_N^j} is equivalent to rho separate
K-point evaluations, one per coset t in [0,rho): twist coefficients by
(gamma*W_N^t)^l for l in [0,K), NTT_K, and interleave codeword[i*rho+t] =
result_t[i]. Implemented below as `coset_ntt_encode`, checked bit-exact
against the reference `_coset_encode_codewords` before any timing counts.

Run:
    uv run --project /home/riftuser/VerInf python3 coset_ntt_bench.py
"""
from __future__ import annotations

import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "prover"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "demo"))
import _uint64_compat  # noqa
import torch

from cuda_primitives import P, gl_mul, ntt_forward_batched
import core as C
from core import LigeroConfig, _coset_encode_codewords, _coset_powers

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_log import log_run


def coset_ntt_encode(coeffs: torch.Tensor, cfg: LigeroConfig) -> torch.Tensor:
    """rho separate length-K_DEG coset NTTs instead of one length-N_LIG NTT.
    Bit-exact equivalent of _coset_encode_codewords, checked below."""
    m, K, N = coeffs.size(0), cfg.K_DEG, cfg.N_LIG
    rho = N // K
    if m == 0:
        return torch.empty((0, N), dtype=torch.uint64, device="cuda")

    W_N = cfg.W_N
    gamma_t = [(cfg.coset_shift * pow(W_N, t, P)) % P for t in range(rho)]
    # (rho, K) twist-power table, one row per coset -- each row cached by
    # _coset_powers keyed on (K, gamma_t), same cache _coset_encode_codewords uses.
    twist_table = torch.stack([_coset_powers(K, g) for g in gamma_t])  # (rho, K)

    # Twist all rho cosets for all m rows in one batched multiply: broadcast
    # coeffs to (rho, m, K), twist_table to (rho, 1, K) -> (rho, m, K).
    coeffs_r = coeffs.unsqueeze(0).expand(rho, m, K).contiguous()
    twist_r = twist_table.unsqueeze(1).expand(rho, m, K).contiguous()
    twisted = gl_mul(coeffs_r.view(rho * m, K), twist_r.view(rho * m, K))  # (rho*m, K)

    ntt_forward_batched(twisted)  # in-place, (rho*m, K)
    results = twisted.view(rho, m, K)
    # codeword[row, i*rho + t] = results[t, row, i]
    return results.permute(1, 2, 0).contiguous().view(m, N)


def check_correctness(K: int, rho: int, m: int = 4, seed: int = 0) -> bool:
    cfg = LigeroConfig(ELL=max(1, K // 4), K_DEG=K, N_LIG=K * rho, T_QUERIES=1)
    torch.manual_seed(seed)
    coeffs = torch.randint(0, 1 << 62, (m, K), dtype=torch.int64, device="cuda").to(torch.uint64)
    ref = _coset_encode_codewords(coeffs, cfg)
    got = coset_ntt_encode(coeffs, cfg)
    ok = torch.equal(ref, got)
    print(f"  correctness K={K} rho={rho} m={m}: {'OK bit-exact' if ok else 'MISMATCH'}", flush=True)
    return ok


def bench_one(K: int, rho: int, m: int, reps: int = 5):
    cfg = LigeroConfig(ELL=max(1, K // 4), K_DEG=K, N_LIG=K * rho, T_QUERIES=1)
    torch.manual_seed(0)
    coeffs = torch.randint(0, 1 << 62, (m, K), dtype=torch.int64, device="cuda").to(torch.uint64)

    # Neither path mutates `coeffs` (both twist into a freshly-allocated
    # tensor before the in-place NTT), confirmed by check_correctness()
    # reusing the same input for both -- no per-rep clone needed.
    _coset_encode_codewords(coeffs, cfg)  # warm up (JIT/alloc/cache)
    coset_ntt_encode(coeffs, cfg)
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(reps):
        _coset_encode_codewords(coeffs, cfg)
    torch.cuda.synchronize()
    standard_s = (time.time() - t0) / reps

    t0 = time.time()
    for _ in range(reps):
        coset_ntt_encode(coeffs, cfg)
    torch.cuda.synchronize()
    coset_s = (time.time() - t0) / reps

    n_elements = m * K * rho
    return dict(K=K, rho=rho, N=K * rho, m=m, n_elements=n_elements,
                standard_s=standard_s, coset_s=coset_s,
                standard_ns_per_elem=standard_s * 1e9 / n_elements,
                coset_ns_per_elem=coset_s * 1e9 / n_elements,
                speedup=standard_s / coset_s if coset_s > 0 else float("nan"))


def main():
    print("=== correctness (bit-exact vs the reference single-N-NTT encoder) ===")
    for K, rho in [(64, 4), (256, 4), (1024, 8), (4096, 4)]:
        assert check_correctness(K, rho), f"coset_ntt_encode diverged from reference at K={K}, rho={rho}"

    print("\n=== throughput: standard (one length-N NTT) vs coset (rho length-K NTTs) ===")
    # K=1024: this repo's current toy/medium CFG. K=16384: notebook's current
    # production K. K=262144=2^18: the notebook's recommended ceiling.
    configs = [
        (1024, 4, 64),
        (16384, 4, 16),
        (65536, 4, 8),     # the Bailey-fast-path length (ntt.cuh) -- standard should look artificially good here
        (262144, 4, 4),    # notebook's recommended K=2^18
    ]
    results = []
    for K, rho, m in configs:
        r = bench_one(K, rho, m)
        print(f"  K=2^{K.bit_length()-1:<2} ({K:>7,}) rho={rho}  m={m:<3}  "
              f"standard={r['standard_ns_per_elem']:.3f}ns/elem  "
              f"coset={r['coset_ns_per_elem']:.3f}ns/elem  "
              f"speedup={r['speedup']:.2f}x", flush=True)
        results.append(r)
        log_run(
            kind="coset_ntt_ab", label=f"K={K},rho={rho},m={m}",
            params=dict(K=K, rho=rho, N=K * rho, m=m, n_elements=r["n_elements"],
                        hardware="Tesla V100-SXM3-32GB"),
            measured=dict(standard_s=r["standard_s"], coset_s=r["coset_s"],
                           standard_ns_per_elem=r["standard_ns_per_elem"],
                           coset_ns_per_elem=r["coset_ns_per_elem"], speedup=r["speedup"]),
            notes="isolated primitive-level A/B, not wired into the real prove() pipeline; "
                  "coset_ntt_encode checked bit-exact against _coset_encode_codewords",
        )

    print(f"\n{len(results)} points logged to analysis/bench/prove_runs.jsonl "
          f"(kind=coset_ntt_ab). View with: python3 show_runs.py --kind coset_ntt_ab")


if __name__ == "__main__":
    main()
