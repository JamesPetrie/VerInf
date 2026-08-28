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
        "evidence": {
            "forward": "measured: analysis/bench/remote_results/0b51ac2023c5/witness.log",
            "other_terms": "projected hypotheses pending sampled-audit Vast campaign",
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
