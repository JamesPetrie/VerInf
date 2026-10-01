"""tools/bench_static_check.py, the CPU gate's static check of the scripts
that build through the toy transformer demo (CPU).

The July 2026 break (a keyword removed from Tape.rmsnorm, still passed by the
toy demo) must be reported in each shape a script can reach it, valid calls
must pass, findings must fail the exit status, and an empty or missing file
list is a usage error, never a pass."""
import pathlib
import re
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
TOOL = REPO / "tools" / "bench_static_check.py"
FIX = pathlib.Path(__file__).resolve().parent / "fixtures" / "bench_static"
sys.path.insert(0, str(REPO / "tools"))

import bench_static_check as bsc  # noqa: E402


def _run(*files):
    r = subprocess.run([sys.executable, str(TOOL), *map(str, files)],
                       capture_output=True, text=True, cwd=REPO)
    findings = [l for l in r.stdout.splitlines() if not l.startswith("bench_static_check:")]
    return r.returncode, findings, r.stdout + r.stderr


def _marked_lines(path):
    return {i for i, l in enumerate(path.read_text().splitlines(), 1) if "# REPORT" in l}


def _reported_lines(findings):
    return {int(re.match(r"[^:]+:(\d+):", f).group(1)) for f in findings}


def test_the_maintained_scripts_are_clean():
    rc, findings, out = _run()
    assert rc == 0 and not findings, out
    assert f"{len(bsc.MAINTAINED)} files" in out


def test_the_july_regression_is_reported_in_every_shape():
    rc, findings, out = _run(FIX / "july_regression.py")
    assert rc == 1, out
    assert len(findings) == 3 and all("slack_n_chunks" in f for f in findings), out
    shapes = {f.split(": ")[1] for f in findings}
    assert shapes == {"Tape.rmsnorm", "dt._run_block", "_run_block"}, shapes
    assert _reported_lines(findings) == _marked_lines(FIX / "july_regression.py")


def test_valid_calls_pass():
    rc, findings, out = _run(FIX / "valid_calls.py")
    assert rc == 0 and not findings, out


def test_each_other_defect_is_reported_where_marked():
    rc, findings, out = _run(FIX / "other_defects.py")
    assert rc == 1, out
    assert _reported_lines(findings) == _marked_lines(FIX / "other_defects.py"), out


def test_an_empty_list_is_a_usage_error(monkeypatch, capsys):
    monkeypatch.setattr(bsc, "MAINTAINED", [])
    assert bsc.main([]) == 2
    assert "no files" in capsys.readouterr().err


def test_a_missing_file_is_a_usage_error():
    rc, _, out = _run(FIX / "no_such_fixture.py")
    assert rc == 2 and "no such file" in out


def test_a_bench_outside_the_list_is_reported(tmp_path):
    (tmp_path / "new_bench.py").write_text("import demo_toy_transformer as dt\n")
    (tmp_path / "unrelated.py").write_text("import json\n")
    out = bsc.unlisted(tmp_path, bsc.MAINTAINED)
    assert len(out) == 1 and "new_bench.py" in out[0]


@pytest.mark.parametrize("rel", bsc.MAINTAINED)
def test_every_maintained_file_exists(rel):
    assert (REPO / rel).is_file()
