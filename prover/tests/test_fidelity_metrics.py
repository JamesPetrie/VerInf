"""The fidelity metrics and the pre-registered acceptance
(prover/fidelity_metrics.py, analysis/k2-fidelity-study-design.md §4, §6), CPU:

- KL and NLL per position equal the direct float64 formulas, chunking does
  not change them, and float32 logits are widened, not used as they are;
- the verdicts: equal arms accept; an integer twice the engine fails; one
  bad passage makes the bound cross zero, inconclusive; a better point
  estimate with an interval crossing the bound is inconclusive, not a pass;
  zero against zero passes through the allowance; a spike past twice the
  engine's maximum fails on its own; worst positions are reported paired;
- the routing explainers separate support changes from reweighting;
- the frozen constants are the note's."""
import math
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import fidelity_metrics as fm                       # noqa: E402

NOTE = pathlib.Path(__file__).resolve().parents[2] / "analysis" / "k2-fidelity-study-design.md"


def _direct(ref, arm, tgt):
    kl, nll = [], []
    for r, a, y in zip(ref.astype(np.float64), arm.astype(np.float64), tgt):
        p = np.exp(r - r.max()); p /= p.sum()
        q = np.exp(a - a.max()); q /= q.sum()
        kl.append(float((p * (np.log(p) - np.log(q))).sum()))
        nll.append(-math.log2(q[y]))
    return np.asarray(kl), np.asarray(nll)


def test_position_metrics_are_the_direct_formulas_in_any_chunking():
    rng = np.random.default_rng(1)
    ref = rng.normal(0, 3, (37, 50))
    arm = ref + rng.normal(0, 0.2, ref.shape)
    tgt = rng.integers(0, 50, 37)
    kl, nll = _direct(ref, arm, tgt)
    for chunk in (1, 5, 64):
        got = fm.position_metrics(ref, arm, tgt, chunk=chunk)
        assert np.allclose(got["kl"], kl, rtol=1e-12, atol=1e-15)
        assert np.allclose(got["nll"], nll, rtol=1e-12)
    assert np.all(fm.position_metrics(ref, ref, tgt)["kl"] == 0)


def test_large_logits_are_stable_and_float32_is_widened():
    rng = np.random.default_rng(2)
    ref = rng.normal(0, 1, (4, 30)) + 1e4
    arm32 = (ref + 0.01).astype(np.float32)
    got = fm.position_metrics(ref, arm32, [0, 1, 2, 3])
    assert np.all(np.isfinite(got["kl"])) and np.all(got["kl"] >= 0)
    # the float32 values themselves, widened: equal to computing on their float64 copies
    want = fm.position_metrics(ref, arm32.astype(np.float64), [0, 1, 2, 3])
    assert np.array_equal(got["kl"], want["kl"])


def test_shapes_must_pair():
    with pytest.raises(ValueError):
        fm.position_metrics(np.zeros((3, 4)), np.zeros((3, 5)), [0, 0, 0])
    with pytest.raises(ValueError):
        fm.position_metrics(np.zeros((3, 4)), np.zeros((3, 4)), [0, 0])


def _arms(int_kl, eng_kl, n=50, int_nll=1.0, eng_nll=1.0):
    """One entry per passage: constant KL and NLL over n positions."""
    integer = [dict(kl=np.full(n, k), nll=np.full(n, int_nll)) for k in int_kl]
    engine = [dict(kl=np.full(n, k), nll=np.full(n, eng_nll)) for k in eng_kl]
    return integer, engine


def test_equal_arms_accept():
    integer, engine = _arms([0.3, 0.5, 0.4, 0.2], [0.3, 0.5, 0.4, 0.2])
    out = fm.acceptance(integer, engine, B=2000)
    assert out["overall"] == "accept"
    assert all(r["verdict"] == "pass" for r in out["rows"].values())


def test_an_integer_twice_the_engine_fails():
    integer, engine = _arms([0.6, 1.0, 0.8, 0.4], [0.3, 0.5, 0.4, 0.2])
    out = fm.acceptance(integer, engine, B=2000)
    assert out["rows"]["mean_kl"]["verdict"] == "fail" and out["overall"] == "fail"


def test_one_bad_passage_is_inconclusive():
    integer, engine = _arms([1.0, 1.0, 1.0, 1.6], [1.0, 1.0, 1.0, 1.0])
    row = fm.acceptance(integer, engine, B=4000)["rows"]["mean_kl"]
    assert row["excess"] > 0 and row["lower95"] <= 0 < row["upper95"]
    assert row["verdict"] == "inconclusive"


def test_a_better_point_estimate_does_not_override_the_interval():
    integer, engine = _arms([0.5, 0.5, 0.5, 2.0], [1.0, 1.0, 1.0, 1.0])
    out = fm.acceptance(integer, engine, B=4000)
    row = out["rows"]["mean_kl"]
    assert row["excess"] < 0 and row["upper95"] > 0 and row["verdict"] == "inconclusive"
    assert out["overall"] == "inconclusive"


def test_the_allowance_near_zero():
    zero = fm.acceptance(*_arms([0.0] * 4, [0.0] * 4), B=500)
    assert zero["rows"]["mean_kl"]["verdict"] == "pass" and zero["rows"]["max_kl"]["verdict"] == "pass"
    tiny = fm.acceptance(*_arms([1e-7] * 4, [0.0] * 4), B=500)
    assert tiny["rows"]["mean_kl"]["verdict"] == "pass"
    past = fm.acceptance(*_arms([1e-5] * 4, [0.0] * 4), B=500)
    assert past["rows"]["mean_kl"]["verdict"] == "fail"


def test_nll_row():
    worse = fm.acceptance(*_arms([0.1] * 4, [0.1] * 4, int_nll=1.02, eng_nll=1.0), B=500)
    assert worse["rows"]["mean_nll"]["verdict"] == "fail"
    inside = fm.acceptance(*_arms([0.1] * 4, [0.1] * 4, int_nll=1.005, eng_nll=1.0), B=500)
    assert inside["rows"]["mean_nll"]["verdict"] == "pass"


def test_a_spike_past_twice_the_engine_maximum_fails_alone_and_is_reported_paired():
    integer, engine = _arms([0.1] * 4, [0.1] * 4)
    integer[2]["kl"] = integer[2]["kl"].copy()
    integer[2]["kl"][17] = 0.5                       # mean and p99 stay inside
    out = fm.acceptance(integer, engine, B=1000)
    assert out["rows"]["mean_kl"]["verdict"] == "pass"
    assert out["rows"]["max_kl"]["verdict"] == "fail" and out["overall"] == "fail"
    top = out["worst_integer"][0]
    assert (top["passage"], top["position"]) == (2, 17)
    assert top["integer_kl"] == 0.5 and top["engine_kl"] == 0.1


def test_the_arms_must_score_the_same_positions_and_the_bootstrap_is_seeded():
    integer, engine = _arms([0.1] * 3, [0.1] * 3)
    engine[1]["kl"] = engine[1]["kl"][:-1]
    with pytest.raises(ValueError):
        fm.acceptance(integer, engine, B=10)
    a = fm.acceptance(*_arms([1.0, 1.0, 1.6], [1.0] * 3), B=300)
    b = fm.acceptance(*_arms([1.0, 1.0, 1.6], [1.0] * 3), B=300)
    assert a["rows"] == b["rows"]


def test_nll_calibration():
    out = fm.nll_calibration([np.full(10, 1.1)] * 3, [np.full(10, 1.0)] * 3, B=200)
    assert out["verdict"] == "fail" and abs(out["excess"] - 0.09) < 1e-12


def test_routing_explainers_separate_support_from_reweighting():
    E = 6
    top_ref, w_ref = np.array([[0, 1]]), np.array([[0.6, 0.4]])
    same_support = fm.gate_matrix(top_ref, [[0.5, 0.5]], E)
    g_ref = fm.gate_matrix(top_ref, w_ref, E)
    assert fm.gate_distance(g_ref, same_support)[0] == pytest.approx(0.1)
    assert fm.swapped_weight(g_ref, same_support)["dropped"][0] == 0
    swapped = fm.gate_matrix(np.array([[0, 5]]), [[0.6, 0.4]], E)
    assert fm.gate_distance(g_ref, swapped)[0] == pytest.approx(0.4)
    assert fm.swapped_weight(g_ref, swapped)["dropped"][0] == pytest.approx(0.4)
    assert fm.support_differences(top_ref, np.array([[0, 5]]))[0] == 1


def test_margins_and_residuals():
    q = np.array([[0.9, 0.1, 0.5, 0.45]])
    assert fm.selection_margins(q, 2)[0] == pytest.approx(0.05)
    x = np.array([[3.0, 4.0]])
    assert fm.residual_error(x, x + [[0.3, 0.4]])[0] == pytest.approx(0.1)


def test_the_frozen_constants_are_the_notes():
    assert (fm.RATIO, fm.ABS, fm.NLL_MARGIN, fm.MAX_RATIO) == (1.1, 1e-6, 0.01, 2.0)
    assert (fm.PERCENTILE, fm.PERCENTILE_METHOD) == (99.0, "linear")
    assert (fm.BOOTSTRAP_B, fm.BOOTSTRAP_SEED, fm.UPPER_Q, fm.LOWER_Q) == (10_000, 20261019, 95.0, 5.0)
    note = NOTE.read_text()
    for row in ("| mean KL | integer ≤ 1.1 × engine + 10⁻⁶ nats |",
                "| p99 KL | integer ≤ 1.1 × engine + 10⁻⁶ nats |",
                "| mean continuation NLL | integer − engine ≤ 0.01 bits/token |",
                "| maximum KL | integer ≤ 2 × engine maximum + 10⁻⁶ nats |",
                "B = 10,000", "default_rng(20261019)", 'method="linear"'):
        assert row in note, row
