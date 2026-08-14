"""Closed operation ledger and 4-hour admission test for RoutedProjectedMoE.

This file deliberately does not use a global calibration multiplier.  Counts
come from the same full-witness ledger as ``maverick_cost_model.py``; each
executed kernel has its own maximum admissible rate.  A production run is
admitted only when its measured p99 bound is no worse than every rate below.

The structural replacement is:
  * persistent weights leave the ordinary Ligero witness matrix and are
    authenticated by their enrolled roots plus projected/opened columns;
  * all-expert MoE output witnesses are replaced by active-only outputs and the
    two-stage projected relation P = W rho, Q = M P;
  * the current global five-epoch implementation is charged for all five
    active-only witness regenerations; no layer-local pass elimination is
    assumed by this model.
"""

from dataclasses import dataclass


S = 1000
E = 128
LAYERS_MOE = 24
D = 5120
DFF = 8192

# Exact S=1000 ledger emitted by analysis/maverick_cost_model.py.
W_OLD = 888_249_981_888
L_OLD = 162_237_276_010
Q_OLD = 173_106_423_296

W_PERSISTENT_FLAT = 400_000_000_000
W_ALL_EXPERT = 396_361_728_000
LQ_ALL_EXPERT = 132_120_576_000
# Old per-expert MatmulClaim Freivalds auxiliaries disappear as well; the new
# claim has its own R3 vectors counted in W_ROUTE/L_ROUTE/Q_ROUTE.
W_ALL_EXPERT_FV = 169_869_312
L_ALL_EXPERT_FV = 113_255_424
Q_ALL_EXPERT_FV = 56_623_104
W_SELECTED_OLD = 3_096_576_000
LQ_SELECTED_OLD = 1_032_192_000

# RoutedProjectedMatmul layout. N is selected token/input cells; R is projected
# expert/input cells, summed over gate/up/down and 24 MoE layers.  The new
# protocol commits P[R], Q[N], H[N], one yr[T] per matmul and three length-E
# late-Freivalds vectors.  The two small families are charged by full ELL row
# capacity because every claim owns a separate variable.
N = LAYERS_MOE * S * (2 * D + DFF)
R = LAYERS_MOE * E * (2 * D + DFF)
N_MATS = LAYERS_MOE * 3
ELL = 8192
W_ROUTE = 2 * N + R + N_MATS * ELL + 3 * N_MATS * ELL
L_ROUTE = R + 2 * N_MATS * S + N_MATS * (2 * E + 1)
Q_ROUTE = N + N_MATS * E

# The builder also removed the old three per-layer freivalds_combine calls.
W_OLD_COMBINE = 553_032_000
L_OLD_COMBINE = 27_792_000
Q_OLD_COMBINE = 9_216_000
W = (W_OLD - W_PERSISTENT_FLAT - W_ALL_EXPERT - W_ALL_EXPERT_FV
     - W_OLD_COMBINE + W_SELECTED_OLD + W_ROUTE)
L = (L_OLD - LQ_ALL_EXPERT - L_ALL_EXPERT_FV - L_OLD_COMBINE
     + LQ_SELECTED_OLD + L_ROUTE)
Q = (Q_OLD - LQ_ALL_EXPERT - Q_ALL_EXPERT_FV - Q_OLD_COMBINE
     + LQ_SELECTED_OLD + Q_ROUTE)

# Admission prices physical RS row capacity, not the sum of logical variable
# lengths.  The target driver exports the exact layout; these rounded-up caps
# cover every per-variable tail without pretending aggregate packing.
FRESH_ROW_CAPACITY = 100_000_000_000
LINEAR_COUNT_CAP = 32_000_000_000
QUADRATIC_COUNT_CAP = 45_000_000_000

# Executed flat-Variable layout.  Persistent gains are excluded because the
# builder commits them as fresh p1 values.  Every remaining length is divisible
# by ELL, so message capacity equals the physical parameter slots exactly.
P_WEIGHT = 402_724_618_240
WEIGHT_ROW_CAPACITY = 402_724_618_240

# Production proof transport is JSON-framed u64le/base64, not decimal integers.
# The exact field payload is unchanged; 4/3 base64 expansion plus roots, paths,
# claims and a conservative metadata allowance stays below this cap.  Legacy
# decimal JSON remains accepted by the verifier but is not the 4-hour path.
PROOF_BYTES_COMPACT = 52_000_000_000

# ...and the cap is DERIVED, not asserted.  One opened column carries one value
# per committed row, over T_QUERIES columns; base64 of 8 canonical bytes is
# 10.67 B/value, and 11 B/value covers the JSON envelope (quotes, commas, keys).
# Merkle paths, roots and the claim document are megabytes against this.
T_QUERIES = 54
BYTES_PER_VALUE_B64 = 11
OPENED_VALUES = ((WEIGHT_ROW_CAPACITY + FRESH_ROW_CAPACITY) // ELL) * T_QUERIES
PROOF_BYTES_DERIVED = OPENED_VALUES * BYTES_PER_VALUE_B64

# The current global transcript regenerates the witness in five commitment /
# test epochs.  This is NOT hidden behind a one-pass fiction: every pass uses
# active-only expert execution on the real GGUF shards.  Shape-exact selected
# forward work includes embedding gather-as-matmul and QK^T/AV attention.
# It is 19.68898048T MACs/pass, hence 98.4449024T total.  Random-weight
# benchmarks cannot satisfy the real-GGUF decode/page-migration admission.
SEMANTIC_SWEEPS = 5
ACTIVE_MACS_PER_SWEEP = 19_688_980_480_000
ACTIVE_MACS_TOTAL = SEMANTIC_SWEEPS * ACTIVE_MACS_PER_SWEEP


@dataclass(frozen=True)
class SLO:
    # Seconds or ns/unit upper bounds.  These are admission limits, not fitted
    # averages: each production kernel benchmark must establish a simultaneous
    # p99 upper confidence bound below the corresponding number.
    model_load_s: float = 400.0
    semantic_all_sweeps_s: float = 3609.0
    fresh_commit_fold_ns: float = 9.5
    # RESTATED 2026-08-06 by the model's owner, on measurement rather than
    # estimate. `linear` is the CONSTRAINT-side half of the q_lin fold;
    # fresh_commit_fold and persistent_weight_qlin price the row/witness side.
    # The measured fold rate is 3.56 ns per constraint id (V100 and A100
    # agree), so the 0.8 ns estimate understated it by 4.5x. At 3.5625 ns the
    # stage costs 114.0 s and the envelope still closes at 13,046.3 s.
    linear_ns: float = 3.5625
    quadratic_ns: float = 17.0
    fresh_hash_coef_ns: float = 1.4
    # Complete persistent q sweep: message->polynomial inverse NTT, IRS/linear
    # folds and coefficient generation.  The N-point LDE is skipped in code.
    persistent_q_sweep_ns: float = 9.0
    persistent_open_ns: float = 4.5
    fresh_open_ns: float = 4.5
    proof_bytes_per_s: float = 108_000_000.0
    rtt_s: float = 80.0
    tail_s: float = 20.0
    orchestration_refresh_s: float = 600.0


def seconds(slo: SLO = SLO()):
    # Executed serializer writes base64 of canonical u64le field bytes inside
    # the same JSON envelope. Exact weight openings are 21.237GB raw; the cap
    # covers 4/3 expansion, fresh rows, paths, polynomials and metadata.
    # Local verifier reads this file; remote transfer
    # is outside this prover SLO unless it is overlapped explicitly.
    proof_bytes = PROOF_BYTES_COMPACT
    parts = {
        "model_load": slo.model_load_s,
        "semantic_5_active_sweeps": slo.semantic_all_sweeps_s,
        "fresh_commit_fold": slo.fresh_commit_fold_ns * 1e-9 * FRESH_ROW_CAPACITY,
        "linear": slo.linear_ns * 1e-9 * LINEAR_COUNT_CAP,
        "quadratic": slo.quadratic_ns * 1e-9 * QUADRATIC_COUNT_CAP,
        "fresh_hash_coef": slo.fresh_hash_coef_ns * 1e-9 * FRESH_ROW_CAPACITY,
        # The P=W*rho matvec is fused into the R2 active semantic weight read
        # and its 56.6M-field result is cached.  This separate scan is the
        # post-s_comb persistent-weight q_lin fold and its secret pad work.
        "persistent_weight_qlin": slo.persistent_q_sweep_ns * 1e-9 * WEIGHT_ROW_CAPACITY,
        "persistent_open": slo.persistent_open_ns * 1e-9 * WEIGHT_ROW_CAPACITY,
        "fresh_open": slo.fresh_open_ns * 1e-9 * FRESH_ROW_CAPACITY,
        "proof_egress": proof_bytes / slo.proof_bytes_per_s,
        "rtt": slo.rtt_s,
        "tail": slo.tail_s,
        "orchestration_refresh": slo.orchestration_refresh_s,
    }
    return parts, sum(parts.values())


def main():
    assert N == 442_368_000 and R == 56_623_104
    assert W == 95_205_646_976
    assert L == 31_064_630_194
    assert Q == 42_394_577_408
    assert ACTIVE_MACS_TOTAL / SLO().semantic_all_sweeps_s >= 27_000_000_000
    # The compact-wire cap must cover the layout it is charged for, or the
    # egress stage would be priced for a file smaller than the one written.
    assert PROOF_BYTES_DERIVED <= PROOF_BYTES_COMPACT, (
        f"proof cap {PROOF_BYTES_COMPACT:,} < derived {PROOF_BYTES_DERIVED:,}")
    parts, total = seconds()
    print(f"old ledger W/L/Q = {W_OLD:,} / {L_OLD:,} / {Q_OLD:,}")
    print(f"new ledger W/L/Q = {W:,} / {L:,} / {Q:,}")
    print(f"ratios             = {W/W_OLD:.4%} / {L/L_OLD:.4%} / {Q/Q_OLD:.4%}")
    for name, value in parts.items():
        print(f"{name:24s} {value:9.3f} s")
    print(f"{'TOTAL':24s} {total:9.3f} s = {total/3600:.4f} h")
    print(f"{'MARGIN':24s} {14_400-total:9.3f} s")
    assert total <= 14_400, "4-hour admission envelope is not closed"


if __name__ == "__main__":
    main()
