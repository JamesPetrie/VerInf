#!/usr/bin/env python3
"""Rigorous A/B: does coset-NTT behave as the cost-model theory claims?

The theory (analysis/verification-parameter-analysis.ipynb `price()`,
`coset_ntt`): encoding via `rho` length-K coset NTTs instead of one length-N=
rho*K NTT trades one expensive transform for `rho` cheap ones. At MATCHED
total element count (rho*m length-K NTTs vs m length-N NTTs both touch m*N
elements), the theory prices the difference purely by the NTT cost curve:
speedup = c(N)/c(K), because c(n) (ns/element) grows with n (DRAM passes grow
with log n). It prices NOTHING for the twist and interleave the coset path
also needs.

This bench tests that claim with statistical rigor, and separates the two
questions the notebook conflates:

  TEST 1 -- the pure c(n) claim: raw batched NTT, one length-N vs rho*m
    length-K, matched elements, no twist/interleave. Measured speedup vs the
    c(N)/c(K) the notebook's own curve predicts. Agreement here = the theory's
    CORE claim holds on this hardware.

  TEST 2 -- the real encode primitive: full standard vs full coset (with
    twist + interleave), decomposed into twist/NTT/interleave so the overhead
    the theory omits is visible and attributable.

Timing: CUDA events, 30 reps after warmup, median reported (robust to the
occasional scheduler blip) with the interquartile spread. Total element count
held ~2^24 across all K so every config does comparable work and the ns/elem
numbers are directly comparable.

Run: uv run --project /home/riftuser/VerInf python3 analysis/bench/coset_ntt_ab.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "prover"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "demo"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))       # analysis/
sys.path.insert(0, str(Path(__file__).resolve().parent))            # analysis/bench/
import _uint64_compat  # noqa
import torch

import ligero_param_derivation as lpd
from cuda_primitives import P, gl_mul, ntt_forward_batched
from core import LigeroConfig, _coset_encode_codewords, _coset_powers
from coset_ntt_bench import coset_ntt_encode
from run_log import log_run

REPS = 30
WARMUP = 5
TARGET_ELEMENTS = 1 << 24   # hold m*N roughly constant across K


def _time_ms(fn, reps=REPS, warmup=WARMUP):
    """Median GPU ms per call over `reps`, plus (q1, q3) spread. CUDA events,
    one sync per rep -- robust per-call timing, not a wall-clock loop."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    n = len(times)
    return times[n // 2], times[n // 4], times[(3 * n) // 4]


def _rand(m, n):
    return torch.randint(0, 1 << 62, (m, n), dtype=torch.int64, device="cuda").to(torch.uint64)


def check_correctness():
    print("=== correctness: coset_ntt_encode bit-exact vs _coset_encode_codewords ===")
    for K, rho in [(64, 4), (256, 4), (1024, 8), (4096, 4), (16384, 4)]:
        cfg = LigeroConfig(ELL=max(1, K // 4), K_DEG=K, N_LIG=K * rho, T_QUERIES=1)
        coeffs = _rand(4, K)
        ok = torch.equal(_coset_encode_codewords(coeffs, cfg), coset_ntt_encode(coeffs, cfg))
        assert ok, f"MISMATCH at K={K}, rho={rho}"
        print(f"  K={K:>6,} rho={rho}: OK")
    print()


def test1_raw_ntt(K, rho, m):
    """Pure c(n) claim: one length-N batched NTT (m rows) vs rho*m length-K
    batched NTTs, matched total elements. NTT is in-place and value-independent,
    so repeated NTT of the same buffer measures transform cost correctly."""
    N = rho * K
    buf_N = _rand(m, N)
    buf_K = _rand(rho * m, K)
    std_ms, std_q1, std_q3 = _time_ms(lambda: ntt_forward_batched(buf_N))
    cos_ms, cos_q1, cos_q3 = _time_ms(lambda: ntt_forward_batched(buf_K))
    elements = m * N
    return dict(
        std_ms=std_ms, cos_ms=cos_ms,
        std_ns_elem=std_ms * 1e6 / elements, cos_ns_elem=cos_ms * 1e6 / elements,
        std_spread=(std_q3 - std_q1) / std_ms, cos_spread=(cos_q3 - cos_q1) / cos_ms,
        measured_speedup=std_ms / cos_ms,
        theory_speedup=lpd.c(N) / lpd.c(K),   # the notebook's own prediction
    )


def _coset_components(coeffs, cfg):
    """coset_ntt_encode split into (twist, ntt, interleave) timed separately."""
    m, K, N = coeffs.size(0), cfg.K_DEG, cfg.N_LIG
    rho = N // K
    gamma_t = [(cfg.coset_shift * pow(cfg.W_N, t, P)) % P for t in range(rho)]
    twist_table = torch.stack([_coset_powers(K, g) for g in gamma_t])

    def do_twist():
        coeffs_r = coeffs.unsqueeze(0).expand(rho, m, K).contiguous()
        twist_r = twist_table.unsqueeze(1).expand(rho, m, K).contiguous()
        return gl_mul(coeffs_r.view(rho * m, K), twist_r.view(rho * m, K))

    twisted = do_twist()

    def do_ntt():
        b = twisted.clone()
        ntt_forward_batched(b)
        return b

    ntt_out = do_ntt()

    def do_interleave():
        return ntt_out.view(rho, m, K).permute(1, 2, 0).contiguous().view(m, N)

    tw = _time_ms(do_twist)[0]
    nt = _time_ms(do_ntt)[0]
    il = _time_ms(do_interleave)[0]
    return dict(twist_ms=tw, ntt_ms=nt, interleave_ms=il)


def test2_full_encode(K, rho, m):
    cfg = LigeroConfig(ELL=max(1, K // 4), K_DEG=K, N_LIG=K * rho, T_QUERIES=1)
    coeffs = _rand(m, K)
    std_ms = _time_ms(lambda: _coset_encode_codewords(coeffs, cfg))[0]
    cos_ms = _time_ms(lambda: coset_ntt_encode(coeffs, cfg))[0]
    comp = _coset_components(coeffs, cfg)
    return dict(std_ms=std_ms, cos_ms=cos_ms, measured_speedup=std_ms / cos_ms, **comp)


def main():
    check_correctness()

    rho = 4
    Ks = [1 << lg for lg in (12, 14, 16, 18, 20)]

    print("=== TEST 1 -- raw NTT (pure c(n) claim, matched elements ~2^24, no twist/interleave) ===")
    print(f"{'K':>9} {'m':>6} {'std ns/el':>10} {'cos ns/el':>10} {'measured':>9} {'theory c(N)/c(K)':>17} {'agree':>7}")
    for K in Ks:
        N = rho * K
        m = max(1, TARGET_ELEMENTS // N)
        r = test1_raw_ntt(K, rho, m)
        agree = r["measured_speedup"] / r["theory_speedup"]
        print(f"2^{K.bit_length()-1:<2}{'':>4} {m:>6} {r['std_ns_elem']:>10.4f} {r['cos_ns_elem']:>10.4f} "
              f"{r['measured_speedup']:>8.2f}x {r['theory_speedup']:>16.2f}x {agree:>6.0%}")
        log_run(kind="coset_ntt_ab_raw", label=f"K={K},rho={rho}",
                params=dict(K=K, rho=rho, N=N, m=m, elements=m * N, reps=REPS,
                            hardware="Tesla V100-SXM3-32GB"),
                measured=dict(**r),
                notes="raw batched NTT, matched elements, CUDA-event median of 30 -- "
                      "tests the pure c(n) theory claim (no twist/interleave)")

    print()
    print("=== TEST 2 -- full encode primitive (standard vs coset, coset decomposed) ===")
    print(f"{'K':>9} {'m':>6} {'std ms':>8} {'cos ms':>8} {'speedup':>8}  | coset breakdown: "
          f"{'twist':>7} {'ntt':>7} {'interlv':>7}")
    for K in Ks:
        N = rho * K
        m = max(1, TARGET_ELEMENTS // N)
        r = test2_full_encode(K, rho, m)
        print(f"2^{K.bit_length()-1:<2}{'':>4} {m:>6} {r['std_ms']:>8.3f} {r['cos_ms']:>8.3f} "
              f"{r['measured_speedup']:>7.2f}x  | {r['twist_ms']:>7.3f} {r['ntt_ms']:>7.3f} "
              f"{r['interleave_ms']:>7.3f}")
        log_run(kind="coset_ntt_ab_full", label=f"K={K},rho={rho}",
                params=dict(K=K, rho=rho, N=N, m=m, reps=REPS, hardware="Tesla V100-SXM3-32GB"),
                measured=dict(**r),
                notes="full encode standard vs coset (with twist+interleave), decomposed; "
                      "CUDA-event median of 30")

    print()
    print("Logged: kind=coset_ntt_ab_raw and coset_ntt_ab_full "
          "(python3 show_runs.py --kind coset_ntt_ab_raw)")


if __name__ == "__main__":
    main()
