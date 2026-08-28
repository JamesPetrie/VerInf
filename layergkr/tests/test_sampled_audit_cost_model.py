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
