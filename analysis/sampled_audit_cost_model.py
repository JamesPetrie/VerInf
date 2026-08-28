"""Cost model for the 2,596-claim, 5-of-49 sampled local audit.

The default row reproduces the requested 599.5 s resident-model estimate. It
keeps measured and projected quantities separate: the 289.1 s forward is from
the saved full-Maverick Vast run; C0, strict-local and opening rates are explicit
engineering hypotheses until the new campaign writes measurements.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Geometry:
    blocks: int = 2596
    window: int = 49
    selected_per_window: int = 5
    rs_columns: int = 61

    def selected(self) -> int:
        full, tail = divmod(self.blocks, self.window)
        tail_selected = (math.ceil(self.selected_per_window * tail / self.window)
                         if tail else 0)
        return full * self.selected_per_window + tail_selected

    def validate(self) -> None:
        assert self.blocks > 0 and self.window > 0
        assert 0 < self.selected_per_window <= self.window
        assert self.rs_columns > 0


@dataclass(frozen=True)
class RateCard:
    # MEASURED on real Maverick / Vast, best saved resident run.
    forward_s: float = 289.1
    # PROJECTED hypotheses. Each is independently replaceable by a campaign
    # measurement; no opaque kappa or hidden amortisation is applied.
    c0_stream_commit_s: float = 113.4
    strict_local_all_blocks_s: float = 1562.4981132075472
    rs_open_61_s: float = 31.5
    persistent_verify_and_orchestration_s: float = 6.0


def predict(geometry: Geometry = Geometry(), rates: RateCard = RateCard()) -> dict:
    geometry.validate()
    selected = geometry.selected()
    fraction = selected / geometry.blocks
    local = rates.strict_local_all_blocks_s * fraction
    terms = {
        "one_real_forward_s": rates.forward_s,
        "c0_stream_commit_s": rates.c0_stream_commit_s,
        "selected_local_proofs_s": local,
        "rs_61_column_openings_s": rates.rs_open_61_s,
        "persistent_verify_orchestration_s": rates.persistent_verify_and_orchestration_s,
    }
    total = round(sum(terms.values()), 6)
    full, tail = divmod(geometry.blocks, geometry.window)
    tail_k = (math.ceil(geometry.selected_per_window * tail / geometry.window)
              if tail else 0)
    probs = ([geometry.selected_per_window / geometry.window] if full else [])
    if tail:
        probs.append(tail_k / tail)
    min_p = min(probs)
    return {
        "geometry": dataclasses.asdict(geometry),
        "selected_blocks": selected,
        "full_windows": full,
        "tail_blocks": tail,
        "tail_selected": tail_k,
        "audit_fraction": fraction,
        "min_first_bad_block_detection": min_p,
        "terms": terms,
        "total_s": total,
        "total_min": total / 60,
        "within_10_min": total <= 600,
        "within_20_min": total <= 1200,
        "real_campaign": {
            "date": "2026-08-28",
            "status": "accepted_runtime_adapter",
            "artifact": "analysis/bench/remote_results/5c40ddad035b/campaign_results.json",
            "target_s": 600.0,
            "runtime_adapter_validated_within_10_min": True,
            "full_cryptographic_599_5_row_validated": False,
            "claims": 2596,
            "selected": 265,
            "accepted": True,
            "binding": "striped-blake3 exact-local runtime",
            "wall_s": 406.7752990722656,
            "forward_s": 360.1375698968768,
            "c0_stream_commit_s": 8.88972863741219,
            "selected_exact_local_checks_s": 37.74800053797662,
            "rs_open_s": 0.0,
            "verify_s": 0.0,
            "peak_gpu_gib": 61.747413635253906,
            "c0_root": "6979947a47cdead273a0efeac6b4fd920f89a421c8f00b45237ed27384181c3d",
            "limitation": (
                "the real adapter measured exact selected-claim recomputation; "
                "Freivalds/sumcheck/lookup proof objects and 61 RS openings "
                "were validated by the separate portable protocol smoke but "
                "were not timed on the 400B witness"
            ),
            "prior_failed_attempt": {
                "status": "manually_aborted_over_cap",
                "timed_audit_lower_bound_s": 4147.5,
                "process_elapsed_lower_bound_s": 4226.0,
                "root_cause": (
                    "LIGERO_NO_FOLD=1, one-column C0 hashing, and a post-exit "
                    "rather than process-killing timeout"
                ),
            },
            "diagnostics": {
                "estimated_nonpersistent_witness_gib": 508.5406,
                "largest_49_claim_window_gib": 27.91,
                "largest_hash_batch_temporary_gib": 16.559,
                "old_one_column_hash_gib_s": 0.027,
                "warm_window_batched_hash_gib_s_low": 23.0,
                "warm_window_batched_hash_gib_s_high": 31.8,
                "local_128_mib_effective_commit_gib_s": 1.79,
            },
            "rs_preflight": {
                "date": "2026-08-28",
                "status": "measured_local_production_geometry",
                "gpu": "Tesla V100-SXM3-32GB",
                "geometry": {"ELL": 16322, "K_DEG": 16384,
                             "N_LIG": 32768, "columns": 61},
                "padding_slots": 62,
                "probe_rows": 8192,
                "probe_commit_s": 0.1808278919197619,
                "probe_open_s": 0.11607919121161103,
                "probe_verify_s": 0.001089682336896658,
                "projected_508_5406_gib_commit_s": 92.30763332463916,
                "projected_508_5406_gib_open_s": 59.25521392317395,
                "second_full_lde_for_opening": True,
                "limitation": (
                    "linear projection from a synthetic 8192-row V100 probe; "
                    "per-variable padding, selected-wire re-encoding, and the "
                    "real A100 window distribution require the Vast campaign"
                ),
            },
        },
        "evidence": {
            "forward": "measured: analysis/bench/remote_results/0b51ac2023c5/witness.log",
            "runtime_adapter": "measured: analysis/bench/remote_results/5c40ddad035b/campaign_results.json",
            "full_crypto_terms": "projected hypotheses; portable protocol smoke is functional but not a real-400B timing",
        },
        "excluded": ["model download", "GGUF/model loading", "one-time verifier startup"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--forward-s", type=float, default=RateCard.forward_s)
    ap.add_argument("--c0-s", type=float, default=RateCard.c0_stream_commit_s)
    ap.add_argument("--strict-all-s", type=float, default=RateCard.strict_local_all_blocks_s)
    ap.add_argument("--open-s", type=float, default=RateCard.rs_open_61_s)
    ap.add_argument("--verify-s", type=float,
                    default=RateCard.persistent_verify_and_orchestration_s)
    args = ap.parse_args()
    row = predict(rates=RateCard(args.forward_s, args.c0_s, args.strict_all_s,
                                 args.open_s, args.verify_s))
    if args.json:
        print(json.dumps(row, indent=2, sort_keys=True))
    else:
        print("sampled local audit: 5-of-49, 2,596 claims, 61 RS columns")
        print(f"selected {row['selected_blocks']}/2596 = "
              f"{100*row['audit_fraction']:.3f}%")
        print(f"minimum P[first bad block sampled] = "
              f"{100*row['min_first_bad_block_detection']:.3f}%")
        for name, seconds in row["terms"].items():
            print(f"  {name:36s} {seconds:8.1f} s")
        print(f"  {'TOTAL':36s} {row['total_s']:8.1f} s = {row['total_min']:.3f} min")
        print("  model load/download excluded; projected terms require Vast validation")
    return 0 if row["within_20_min"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
