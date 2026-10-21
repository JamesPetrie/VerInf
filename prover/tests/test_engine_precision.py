"""tools/engine_precision.py's verdict rule (review of 2026-10-21), on
records built by hand: no llama.cpp. A CPU run or a one-token GPU run that
failed or is missing leaves the verdict unresolved, never a mismatch; a
completed one outside the bound is evidence of a mismatch; batched
departures are consistent with internal precision only when every batched
run completed and departs first at a matmul's output within 2^-6."""
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "prover"))
sys.path.insert(0, str(REPO / "prover" / "tests"))
sys.path.insert(0, str(REPO / "tools"))

import engine_precision as ep               # noqa: E402

IN, OUT = 1e-7, 2e-3


def _row(within, layer0=None):
    row = dict(within_bound=within)
    if layer0 is not None:
        row["layer0_vs_cpu"] = {n: [v] for n, v in layer0.items()}
    return row


def _departs_at(name, size=OUT):
    """Layer 0 within float32 until `name`, which leaves it by `size`."""
    i = ep.LAYER0.index(name)
    return {n: (IN if j < i else size) for j, n in enumerate(ep.LAYER0)}


def _record(**changes):
    rec = {c: _row(True) for c in ep.AGREEMENT}
    rec["gpu-12"] = _row(False, _departs_at("kqv_out-0"))
    rec["gpu-40"] = _row(False, _departs_at("ffn_inp-0"))
    rec["gpu-12-no-tf32"] = _row(True)
    rec.update(changes)
    return {k: v for k, v in rec.items() if v is not None}


def test_matmul_departures_are_consistent_with_internal_precision():
    v = ep.verdict(_record())
    assert v["verdict"] == "consistent with internal precision"
    assert v["departures"]["gpu-12"]["first"] == "kqv_out-0"
    assert "gpu-12-no-tf32" not in v["departures"]


@pytest.mark.parametrize("cond", ep.AGREEMENT)
def test_a_failed_cpu_or_one_token_run_is_unresolved(cond):
    v = ep.verdict(_record(**{cond: dict(error="runner exited 1")}))
    assert v["verdict"] == "unresolved" and v["missing"] == [cond]


@pytest.mark.parametrize("cond", ep.AGREEMENT)
def test_a_missing_cpu_or_one_token_run_is_unresolved(cond):
    v = ep.verdict(_record(**{cond: None}))
    assert v["verdict"] == "unresolved" and v["missing"] == [cond]


def test_a_completed_run_outside_the_bound_is_evidence_of_a_mismatch():
    v = ep.verdict(_record(**{"gpu-12-steps": _row(False)}))
    assert v["verdict"] == "evidence of a semantic mismatch" and v["outside"] == ["gpu-12-steps"]


def test_a_departure_at_a_norm_is_unresolved():
    assert ep.verdict(_record(**{"gpu-12": _row(False, _departs_at("attn_norm-0"))}))["verdict"] == "unresolved"


def test_a_departure_past_the_share_is_unresolved():
    big = _row(False, _departs_at("ffn_out-0", 2 * ep.MATMUL_SHARE))
    assert ep.verdict(_record(**{"gpu-40": big}))["verdict"] == "unresolved"


def test_a_failed_batched_run_is_unresolved():
    v = ep.verdict(_record(**{"gpu-40-no-fusion": dict(error="runner exited 1")}))
    assert v["verdict"] == "unresolved" and v["departures"]["gpu-40-no-fusion"]["completed"] is False


def test_a_batched_run_without_its_layer0_capture_is_unresolved():
    assert ep.verdict(_record(**{"gpu-12": _row(False)}))["verdict"] == "unresolved"
