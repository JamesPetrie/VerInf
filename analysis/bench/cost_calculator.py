#!/usr/bin/env python3
"""Standalone prover-cost calculator -- the "turn this knob" cell from
analysis/verification-parameter-analysis.ipynb (`Config` / `price()`),
pulled out of the notebook so it can be run and versioned without a kernel,
and corrected against this session's measured results rather than the
notebook's own assumptions alone.

What changed vs. the notebook's price():

1. `witness_recompute` is no longer assumed negligible or read off the
   demonstrated run's T_WIT_S constant. It is fit from this codebase's own
   measurements (LIGERO_PHASE_TIMING's `witness` bucket across 5 medium-scale
   toy-transformer runs, m_total 604k-2.75M -- see
   analysis/toy-transformer-prove-time-formula.md and
   analysis/bench/prove_runs.jsonl). Pass --witness-mode notebook to use the
   original T_WIT_S-based term instead (useful for reproducing the notebook's
   own numbers on the demonstrated 400B run).
2. `coset_ntt=True` is a THEORETICAL toggle only -- confirmed (this session)
   that rho independent length-K coset NTTs are NOT implemented anywhere in
   the real prover (prover/core.py's _coset_encode_codewords does one
   length-N=rho*K NTT). Passing --coset-ntt still reprices using the
   notebook's `c = lambda n: lpd.c(min(n, K))` approximation, but the CLI
   labels it clearly as unimplemented/theoretical in the output.
3. Adds a fixed per-process floor (~3-4s on this machine: CUDA context, tape
   setup) that dominates below m_total~500k, where neither the identity floor
   nor witness_recompute predicts much -- see the toy-scale finding in the
   same report.

Usage:
    python3 cost_calculator.py --W 168744448 --Q 117 --L 84301051 --m-total 329579
    python3 cost_calculator.py --S 1093                      # reproduce the notebook's demonstrated run
    python3 cost_calculator.py --K 262144 --rho 4 --coset-ntt --S 1093   # notebook's "K=2^18, coset-NTT" recommendation
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # analysis/ on path
import ligero_param_derivation as lpd

FIELD_P = (1 << 64) - (1 << 32) + 1

# --- our own measurements, this session (Tesla V100-SXM3-32GB) -------------
# Fit across 5 toy-transformer runs (d=512, d_ff 1024-2048, SEQ 256-1024, 4
# layers; ELL=512/K_DEG=1024/N_LIG=4096/T=16 throughout), witness_s = the
# LIGERO_PHASE_TIMING `witness` bucket (compute_fn execution -- rmsnorm,
# matmul, softmax's causal-range-proof binary search, ...), regressed
# against m_total. R^2=0.994 excluding the smallest (closest-to-toy-scale)
# point. NOT a protocol constant -- dominated by softmax cost in THIS sweep,
# where SEQ (the real driver of softmax cost) happened to correlate tightly
# with m_total because ELL/K/N/rho were held fixed. Re-fit if your workload's
# claim mix differs a lot from a stack of RmsNorm/Matmul/RoPE/Softmax/SiLU
# blocks at this scale.
WITNESS_RECOMPUTE_US_PER_ROW = 74.57
WITNESS_RECOMPUTE_INTERCEPT_S = -42.13
WITNESS_RECOMPUTE_VALID_M_TOTAL = (604_267, 2_747_117)  # fit range; extrapolate with caution

# Toy-scale (m_total < ~500k) fixed per-process floor: CUDA context init,
# tape/layout setup -- measured mean 3.9s, std 1.5s across 9 configs where
# m_total barely moved (330k-456k) yet prove_s did (2.8-7.5s). Below the
# witness-recompute fit's valid range, THIS dominates instead.
TOY_SCALE_FIXED_FLOOR_S = 3.9
TOY_SCALE_M_TOTAL_CEILING = 500_000

# Per-row OVERHEAD (Python dispatch + kernel launch per committed row), the term
# the asymptotic c(n)*size floor omits. It DOMINATES at small scale (2-6x of the
# measured floor at toy) and saturates once the GPU is full, so it stays
# negligible at 400B. Fit against 39 saved runs (toy d16-512 through phone
# d896/d2048): floor+overhead lands within +-30% of measured floor across the
# whole range, vs 0.15-0.47x for the bare asymptotic floor. Replaces the old
# crude TOY_SCALE_FIXED_FLOOR hack with a term that scales correctly.
OVERHEAD_NS_PER_ROW = 22_000        # ~22 us per committed row (measured constant)
OVERHEAD_SAT_ROWS = 5_000_000       # saturates here (GPU full) -> flat above


@dataclass(frozen=True)
class Config:
    """Same fields as the notebook's Config, defaults matching its
    demonstrated-run values. `S` drives Llama-shaped W(S)/Q(S)/L(S) closed
    forms (lpd.W_of etc) for reproducing the notebook's own analysis;
    pass W/Q/L/m_total directly instead for an arbitrary (non-Llama-shaped)
    model, e.g. this repo's demo_toy_transformer.py."""
    S: int = 1093
    K: int = 16384
    rho: int = 4
    pad_weights: int = 8192
    pad_proof: int = 8192
    T_queries: int = 40
    field_bits: int = 64
    logup_repeats: int = 1
    coset_ntt: bool = False
    witness_passes: int = 4

    @property
    def N(self) -> int:
        return self.rho * self.K

    @property
    def ELL_weights(self) -> int:
        return self.K - self.pad_weights

    @property
    def ELL_proof(self) -> int:
        return self.K - self.pad_proof


_LPD_MIN_LG, _LPD_MAX_LG = 12, 24  # lpd.c()'s calibrated range (C_V100 dict keys)


def _c_safe(n: int, flag: list) -> float:
    """lpd.c(n), clamped to the calibrated range instead of extrapolated
    past it -- lpd.c's linear-in-log2(n) extrapolation goes NEGATIVE below
    2^12 (confirmed this session: K_DEG=1024 gives c(1024) < 0), which is
    physically meaningless, not just imprecise. Clamping to the nearest
    calibrated endpoint is itself only a rough stand-in outside the
    calibrated range -- `flag[0]` is set so callers can surface a warning
    rather than silently trusting an unclamped or clamped number equally."""
    lg = round(math.log2(n))
    if lg < _LPD_MIN_LG or lg > _LPD_MAX_LG:
        flag[0] = True
        lg = min(max(lg, _LPD_MIN_LG), _LPD_MAX_LG)
        return lpd.c(1 << lg)
    return lpd.c(n)


def identity_floor_s(cfg: Config, *, W: float, Q: float, L: float,
                      W_weights: Optional[float] = None) -> dict:
    """The notebook's row_ns()-based NTT/hash/quad/lin cost, unchanged in
    shape from price(). Returns the per-term breakdown (hours -> converted
    to seconds here) plus the total."""
    W_weights = lpd.W_WEIGHTS if W_weights is None else W_weights
    W_proof = max(W - W_weights, 0.0)
    out_of_range = [False]
    c = (lambda n: _c_safe(min(n, cfg.K), out_of_range)) if cfg.coset_ntt \
        else (lambda n: _c_safe(n, out_of_range))

    rows_weights = W_weights / cfg.ELL_weights
    rows_proof = W_proof / cfg.ELL_proof
    rows = rows_weights + rows_proof

    def row_ns(commit_encode: bool) -> float:
        encode = cfg.K * c(cfg.K) + cfg.N * c(cfg.N)
        reencode = encode
        fold = 4 * cfg.K * c(2 * cfg.K)
        hashing = (cfg.N / 8) * lpd.HASH_NS_PER_COMPRESSION
        return (encode if commit_encode else 0) + reencode + fold + hashing

    weight_lifetime = max(1, cfg.pad_weights // cfg.T_queries)
    encode_weight = cfg.K * c(cfg.K) + cfg.N * c(cfg.N)
    streaming_ns = (rows_proof * row_ns(True)
                    + rows_weights * (row_ns(False) + encode_weight / weight_lifetime))
    quad_ns = (Q / cfg.ELL_proof) * lpd.QUAD_TRANSFORMS_PER_ROW * (2 * cfg.K) * c(2 * cfg.K)
    lin_ns = lpd.B_NS_PER_CID * L

    to_s = lambda ns: ns / 1e9 / (1 - lpd.E_FRACTION)
    terms = dict(streaming_s=to_s(streaming_ns), quadratic_s=to_s(quad_ns), lin_s=to_s(lin_ns))
    return dict(**terms, rows=rows, floor_s=sum(terms.values()), c_out_of_range=out_of_range[0])


def witness_recompute_s(cfg: Config, *, m_total: float, mode: str = "measured") -> dict:
    """Two modes:
    - "measured" (default): this session's fitted linear law in m_total.
      Flags out_of_range=True if m_total falls outside the fit's support.
    - "notebook": the original price() term, cfg.witness_passes * T_WIT_S
      (a constant measured on the demonstrated 400B run's specific witness
      -- appropriate only when reproducing/extending that analysis, not this
      repo's toy transformer)."""
    if mode == "notebook":
        s = cfg.witness_passes * lpd.T_WIT_S / (1 - lpd.E_FRACTION)
        return dict(witness_recompute_s=s, mode=mode, out_of_range=False)
    lo, hi = WITNESS_RECOMPUTE_VALID_M_TOTAL
    s = max(0.0, WITNESS_RECOMPUTE_US_PER_ROW * 1e-6 * m_total + WITNESS_RECOMPUTE_INTERCEPT_S)
    return dict(witness_recompute_s=s, mode=mode, out_of_range=not (lo <= m_total <= hi))


def predict(cfg: Config, *, W: Optional[float] = None, Q: Optional[float] = None,
            L: Optional[float] = None, m_total: Optional[float] = None,
            W_weights: Optional[float] = None, witness_mode: str = "measured") -> dict:
    """Full prediction: identity floor + witness_recompute (+ the toy-scale
    fixed floor when m_total is small enough that it, not either formula
    term, is what actually dominates).

    W_weights: size of the persistent weight block. Defaults to the
    notebook's demonstrated-run constant (lpd.W_WEIGHTS = 4.00e11) ONLY when
    reproducing that run via --S; for a directly-specified W/Q/L (e.g. this
    repo's demo_toy_transformer.py, which never uses a persistent weight
    block -- every run prints "W-block rows 0") it defaults to 0. Override
    explicitly if your model does split off a persistent weight commitment."""
    from_S = W is None
    if from_S:  # Llama-shaped closed forms, reproducing the notebook
        W, Q, L = lpd.W_of(cfg.S), lpd.Q_of(cfg.S), lpd.L_of(cfg.S)
    if W_weights is None:
        W_weights = lpd.W_WEIGHTS if from_S else 0.0
    if m_total is None:
        W_proof = max(W - W_weights, 0.0)
        m_total = (W_weights / cfg.ELL_weights if W_weights else 0.0) + W_proof / cfg.ELL_proof

    floor = identity_floor_s(cfg, W=W, Q=Q, L=L, W_weights=W_weights)
    witness = witness_recompute_s(cfg, m_total=m_total, mode=witness_mode)

    # per-row overhead (dispatch + kernel launch), saturating once the GPU fills.
    # This is the term the asymptotic floor omits; it dominates at toy scale and
    # is negligible at 400B. Replaces the old fixed-floor toy hack.
    overhead_s = OVERHEAD_NS_PER_ROW * min(m_total, OVERHEAD_SAT_ROWS) / 1e9
    small_scale = m_total < TOY_SCALE_M_TOTAL_CEILING
    predicted_total = floor["floor_s"] + witness["witness_recompute_s"] + overhead_s

    field = FIELD_P * 2.0 ** (cfg.field_bits - 64)
    proof_bytes = cfg.T_queries * floor["rows"] * lpd.FIELD_BYTES + \
        (4 * cfg.K + cfg.ELL_proof) * lpd.FIELD_BYTES
    verify_s = cfg.T_queries * floor["rows"] * lpd.VERIFY_NS_PER_CELL / 1e9

    return dict(
        W=W, Q=Q, L=L, m_total=m_total,
        identity_floor_s=floor["floor_s"], floor_terms=floor,
        c_out_of_range=floor["c_out_of_range"],
        witness_recompute_s=witness["witness_recompute_s"],
        witness_mode=witness["mode"], witness_out_of_range=witness["out_of_range"],
        small_scale_regime=small_scale,
        overhead_s=overhead_s,
        predicted_total_s=predicted_total,
        proof_GB=proof_bytes / 1e9, verify_s=verify_s,
    )


def _print_report(cfg: Config, r: dict) -> None:
    print(f"=== Config ===")
    print(f"  K={cfg.K:,}  rho={cfg.rho}  N={cfg.N:,}  T_queries={cfg.T_queries}  "
          f"pad_proof={cfg.pad_proof:,}  coset_ntt={cfg.coset_ntt}"
          f"{'  <-- THEORETICAL: not implemented in the real prover' if cfg.coset_ntt else ''}")
    print(f"  W={r['W']:,.0f}  Q={r['Q']:,.0f}  L={r['L']:,.0f}  m_total={r['m_total']:,.0f}")
    print()
    print(f"=== Prediction ===")
    print(f"  identity floor (NTT/hash/quad/lin): {r['identity_floor_s']:.3f}s"
          + ("  <-- K/N/2K fell outside lpd.c()'s calibrated 2^12-2^24 range; "
             "clamped to the nearest endpoint, not a reliable number" if r["c_out_of_range"] else ""))
    print(f"  per-row overhead (dispatch/launch): {r['overhead_s']:.3f}s"
          + ("  <-- saturated (GPU full); negligible at this scale" if r["m_total"] >= OVERHEAD_SAT_ROWS else "  <-- dominates the floor at this scale"))
    print(f"  witness_recompute ({r['witness_mode']:>8s})      : {r['witness_recompute_s']:.3f}s"
          + ("  <-- outside the fitted m_total range, extrapolated" if r["witness_out_of_range"] else ""))
    print(f"  predicted total (floor+ovh+witness): {r['predicted_total_s']:.2f}s")
    print(f"  (floor+overhead lands within +-30% of the measured floor across 39 saved "
          f"runs, toy d16 through phone d2048; see research_journal.md iter13)")
    print()
    print(f"=== Proof / verify (geometry only, same as notebook) ===")
    print(f"  proof size  : {r['proof_GB']:.2f} GB")
    print(f"  verify time : {r['verify_s']:.2f}s ({r['verify_s']/3600:.2f}h)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--S", type=int, default=None, help="context length (Llama-shaped W/Q/L via lpd)")
    ap.add_argument("--W", type=float, default=None, help="witness elements (overrides --S)")
    ap.add_argument("--Q", type=float, default=None, help="quadratic product count")
    ap.add_argument("--L", type=float, default=None, help="linear constraint count")
    ap.add_argument("--m-total", type=float, default=None, help="row count (else derived from W/ELL)")
    ap.add_argument("--W-weights", type=float, default=None,
                     help="persistent weight-block size (default: notebook's demonstrated-run "
                          "constant if using --S, else 0 -- most models, including this repo's "
                          "toy transformer, don't split off a persistent weight block)")
    ap.add_argument("--K", type=int, default=16384)
    ap.add_argument("--rho", type=int, default=4)
    ap.add_argument("--pad-proof", type=int, default=8192)
    ap.add_argument("--pad-weights", type=int, default=8192)
    ap.add_argument("--T-queries", type=int, default=40)
    ap.add_argument("--field-bits", type=int, default=64)
    ap.add_argument("--logup-repeats", type=int, default=1)
    ap.add_argument("--coset-ntt", action="store_true")
    ap.add_argument("--witness-passes", type=int, default=4)
    ap.add_argument("--witness-mode", choices=["measured", "notebook"], default="measured")
    args = ap.parse_args()

    cfg = Config(S=args.S or 1093, K=args.K, rho=args.rho, pad_weights=args.pad_weights,
                 pad_proof=args.pad_proof, T_queries=args.T_queries, field_bits=args.field_bits,
                 logup_repeats=args.logup_repeats, coset_ntt=args.coset_ntt,
                 witness_passes=args.witness_passes)

    if args.W is not None:
        assert args.Q is not None and args.L is not None, "--W requires --Q and --L too"
        r = predict(cfg, W=args.W, Q=args.Q, L=args.L, m_total=args.m_total,
                    W_weights=args.W_weights, witness_mode=args.witness_mode)
    else:
        r = predict(cfg, m_total=args.m_total, W_weights=args.W_weights,
                    witness_mode=args.witness_mode)
    _print_report(cfg, r)


if __name__ == "__main__":
    main()
