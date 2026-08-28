"""Freeze the headline geometry and 599.5 s cost row."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "analysis"))

from sampled_audit_cost_model import Geometry, predict


def test_headline_cost_row():
    row = predict()
    assert Geometry().selected() == 265
    assert row["audit_fraction"] == 265 / 2596
    assert row["min_first_bad_block_detection"] == 5 / 49
    assert row["total_s"] == 599.5
    assert row["within_10_min"] and row["within_20_min"]
    campaign = row["real_campaign"]
    assert campaign["status"] == "accepted_rs_bound_runtime_adapter"
    assert campaign["accepted"] is True
    assert campaign["claims"] == 2596 and campaign["selected"] == 265
    assert campaign["wall_s"] < 600
    assert campaign["runtime_adapter_validated_within_10_min"] is True
    assert campaign["full_cryptographic_599_5_row_validated"] is False
    assert campaign["rs_openings_materialized"] is True
    assert campaign["rs_rows"] == 4205517
    assert campaign["rs_opened_values"] == campaign["rs_rows"] * 61
    assert campaign["rs_commit_s"] > 0 and campaign["rs_open_s"] > 0
    assert campaign["prior_failed_attempt"]["timed_audit_lower_bound_s"] > 3600
    bridge = campaign["post_campaign_local_bridge"]
    assert bridge["real_maverick_timing_measured"] is False
    assert bridge["manifest_claims_materialized"] == 1330
    assert bridge["manifest_fraction_materialized"] == 1330 / 2596
    assert bridge["family_coverage"]["product-tree"]["materialized"] == 0
    diagnostics = campaign["diagnostics"]
    assert diagnostics["estimated_nonpersistent_witness_gib"] > 500
    assert diagnostics["warm_window_batched_hash_gib_s_low"] > 20
