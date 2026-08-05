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
        "runs": 30,
        "bound_kind": "p99_upper",
        "weights": "real_gguf",
        # every stage exactly at its cap: admissible, with nothing to spare
        "stages": {k: v for k, v in admission.STAGE_CAPS.items()},
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
    r["runs"] = 29
    _refuses(r, "runs per stage")
    r = _honest_report()
    r["bound_kind"] = "mean"
    _refuses(r, "UPPER confidence bounds")
    print("    29 runs / averages instead of upper bounds: refused")


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
