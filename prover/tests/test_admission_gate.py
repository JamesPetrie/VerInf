"""The admission gate is only worth having if it refuses.

Every test here feeds it a report that is wrong in exactly one way and checks
it says no — an over-cap stage, a report from another machine, another build,
another model, another statement, another geometry, too few runs, averages
instead of upper bounds, and random weights. The positive case is one honest
report that passes.
"""
import copy
import json
import os
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import admission


class _Cfg:
    ELL, K_DEG, N_LIG, T_QUERIES = 8192, 16384, 65536, 54


MODEL_ROOT = bytes.fromhex("aa" * 32)
STMT = bytes.fromhex("bb" * 32)
MANIFEST = {"m_total": 1000, "m_w": 800, "m_wnew": 0, "n_p1": 100,
            "n_p2": 80, "n_p3": 20, "n_claims": 42,
            "ELL": 8192, "N_LIG": 65536, "T_QUERIES": 54}


def _honest_report():
    return {
        "source_digest": admission.source_digest(),
        "model_root": MODEL_ROOT.hex(),
        "statement_digest": STMT.hex(),
        "machine": admission.machine_fingerprint(),
        "row_manifest": dict(MANIFEST),
        "runs": admission.MIN_RUNS,
        "bound_kind": admission.BOUND_KIND,
        "weights": "real_gguf",
        # every stage exactly at its cap: admissible, with nothing to spare
        "stages": {k: v for k, v in admission.STAGE_CAPS.items()},
        "stage_samples": {
            k: [v] * admission.MIN_RUNS
            for k, v in admission.STAGE_CAPS.items()
        },
        "egress_filesystem": admission.filesystem_fingerprint("."),
    }


def _check(report):
    admission.check(report, cfg=_Cfg, model_root=MODEL_ROOT,
                    statement_digest=STMT, manifest=MANIFEST)


def _refuses(report, expect):
    try:
        _check(report)
    except SystemExit as e:
        assert expect in str(e), f"refused for the wrong reason: {e}"
        return str(e)
    raise AssertionError(f"gate ACCEPTED a report it should refuse ({expect})")


def test_honest_report_passes():
    _check(_honest_report())
    print("    honest report at cap: admitted")


def test_over_cap_stage_refused():
    r = _honest_report()
    r["stages"]["persistent_weight_qlin"] += 0.001
    _refuses(r, "over its cap")
    print("    stage 1 ms over cap: refused")


def test_wrong_machine_refused():
    r = _honest_report()
    r["machine"] = dict(r["machine"], gpu_name="Some Other GPU")
    _refuses(r, "machine mismatch")
    print("    report from another GPU: refused")


def test_wrong_build_refused():
    r = _honest_report()
    r["source_digest"] = "0" * 64
    _refuses(r, "source digest differs")
    print("    report from another build: refused")


def test_wrong_model_or_statement_refused():
    r = _honest_report()
    r["model_root"] = "cc" * 32
    _refuses(r, "model root differs")
    r = _honest_report()
    r["statement_digest"] = "dd" * 32
    _refuses(r, "statement digest differs")
    print("    report for another model / another statement: refused")


def test_smaller_geometry_refused():
    """The classic way to make a benchmark look good."""
    r = _honest_report()
    r["row_manifest"]["ELL"] = 1024
    _refuses(r, "row manifest mismatch on ELL")
    r = _honest_report()
    r["row_manifest"]["m_total"] = MANIFEST["m_total"] // 2
    _refuses(r, "row manifest mismatch on m_total")
    print("    smaller ELL / half the rows: refused")


def test_weak_statistics_refused():
    r = _honest_report()
    r["runs"] = admission.MIN_RUNS - 1
    _refuses(r, "runs per stage")
    r = _honest_report()
    r["bound_kind"] = "mean"
    _refuses(r, "bound_kind")
    print("    insufficient runs / averages instead of upper bounds: refused")


def _mixed_report():
    """A report under the mixed policy: kernels keep the 714-run bound, the
    single-shot stages carry one measured run and the required margin."""
    r = _honest_report()
    r["bound_kind"] = admission.BOUND_KIND_MIXED
    r["single_shot_stages"] = sorted(admission.SINGLE_SHOT_STAGES)
    for stage in admission.SINGLE_SHOT_STAGES:
        # measured once, at the cap divided by the margin, so the bound (= cap)
        # is exactly SINGLE_SHOT_SAFETY above what was observed
        observed = admission.STAGE_CAPS[stage] / admission.SINGLE_SHOT_SAFETY
        r["stage_samples"][stage] = [observed]
    return r


def test_mixed_policy_admits_single_shot_stages():
    """The long stages are the production run itself: one measurement plus a
    stated margin, and the report has to say that is what it is."""
    _check(_mixed_report())
    print(f"    mixed policy: {len(admission.SINGLE_SHOT_STAGES)} single-shot "
          f"stages measured once at {admission.SINGLE_SHOT_SAFETY}x margin: admitted")


def test_single_shot_margin_enforced():
    r = _mixed_report()
    # observed max creeps up so the cap is no longer a full margin above it
    r["stage_samples"]["semantic_5_active_sweeps"] = [
        admission.STAGE_CAPS["semantic_5_active_sweeps"] / 1.05]
    _refuses(r, "margin over its observed max")
    print("    single-shot bound without its margin: refused")


def test_single_shot_class_cannot_be_widened():
    """A kernel must not be moved into the weaker class to dodge 714 runs."""
    r = _mixed_report()
    r["single_shot_stages"] = sorted(
        set(admission.SINGLE_SHOT_STAGES) | {"quadratic"})
    r["stage_samples"]["quadratic"] = [admission.STAGE_CAPS["quadratic"] / 2]
    _refuses(r, "single_shot_stages must be exactly")
    print("    kernel relabelled as single-shot: refused")


def test_kernel_stages_still_need_the_full_campaign():
    """Under the mixed policy the repeatable stages keep the 714-run rule."""
    r = _mixed_report()
    r["stage_samples"]["quadratic"] = [admission.STAGE_CAPS["quadratic"]] * 30
    _refuses(r, "raw samples, need >= 714")
    print("    30 samples on a kernel stage, even under the mixed policy: refused")


def test_nonfinite_or_negative_stage_refused():
    for bad in (float("nan"), float("inf"), -1.0):
        r = _honest_report()
        r["stages"]["quadratic"] = bad
        _refuses(r, "finite and non-negative")
    print("    NaN / infinity / negative stage bounds: refused")


def test_missing_or_understated_raw_samples_refused():
    r = _honest_report()
    del r["stage_samples"]["quadratic"]
    _refuses(r, "no raw sample list")
    r = _honest_report()
    r["stage_samples"]["quadratic"][-1] += 1.0
    _refuses(r, "below observed max")
    print("    missing / understated raw measurements: refused")


def test_random_weights_refused():
    r = _honest_report()
    r["weights"] = "random"
    _refuses(r, "random weights cannot satisfy it")
    print("    random-weight benchmark: refused")


def test_missing_stage_and_unpriced_stage_refused():
    r = _honest_report()
    del r["stages"]["linear"]
    _refuses(r, "is not in the report")
    r = _honest_report()
    r["stages"]["free_lunch"] = 0.0
    _refuses(r, "unpriced stages")
    print("    missing stage / unpriced extra stage: refused")


def test_incomplete_file_refused():
    r = _honest_report()
    del r["machine"]
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        json.dump(r, open(path, "w"))
        try:
            admission.load_report(path)
            raise AssertionError("a report with no machine binding was loaded")
        except SystemExit as e:
            assert "machine" in str(e), e
    finally:
        os.unlink(path)
    print("    report without a machine binding: refused at load")


def test_dev_config_refused_by_default():
    class Dev:
        ELL, K_DEG, N_LIG, T_QUERIES = 8, 8, 32, 4
    try:
        admission.check_config(Dev)
        raise AssertionError("a dev geometry was admitted")
    except SystemExit as e:
        assert "not the target" in str(e), e
    admission.check_config(Dev, allow_dev=True)      # explicit opt-out is fine
    admission.check_config(_Cfg)                     # the target passes
    print("    dev geometry refused unless explicitly allowed")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    fails = 0
    for t in tests:
        try:
            t(); print(f"[OK ] {t.__name__}")
        except Exception as e:
            fails += 1; print(f"[XX ] {t.__name__}: {type(e).__name__}: {e}")
    print(f"=== admission-gate: {len(tests)-fails}/{len(tests)} "
          f"{'PASS' if not fails else 'FAIL'} ===")
    return fails


if __name__ == "__main__":
    raise SystemExit(main())


def test_caps_match_the_model():
    """The gate's caps are a COPY of the model's stage seconds.

    Nothing kept them in sync, so restating a cap in the model would have left
    the gate quietly enforcing the old one — which is how a run gets authorized
    against numbers nobody agreed to."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "rp4h", pathlib.Path(admission._ANALYSIS) / "routed_projected_4h_model.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    parts, total = m.seconds()
    assert set(parts) == set(admission.STAGE_CAPS), (
        f"stage sets differ: model {sorted(parts)} vs gate "
        f"{sorted(admission.STAGE_CAPS)}")
    for stage, cap in parts.items():
        assert abs(cap - admission.STAGE_CAPS[stage]) < 1e-3, (
            f"cap drift on '{stage}': model {cap} vs gate "
            f"{admission.STAGE_CAPS[stage]}")
    assert total <= admission.TOTAL_CAP_S, total
    print(f"    caps match the model on all {len(parts)} stages; "
          f"total {total:.1f}s of {admission.TOTAL_CAP_S:.0f}s")
