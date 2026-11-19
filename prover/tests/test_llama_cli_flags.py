# Copyright (c) 2026 Levi Neuwirth. Released under the MIT License (see LICENSE).
"""The Llama driver's fixed-bound, scoring, statement and enrollment flags.

Each flag reaches main() as its keyword, a flag left out keeps main()'s
default, and a combination that cannot run is refused by the parser before
any model is loaded. CPU only.
"""
import argparse
import pathlib
import subprocess
import sys

import pytest

_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
for p in (_HERE, _ROOT / "prover", _ROOT / "demo"):
    sys.path.insert(0, str(p))

import demo_llama7b as D  # noqa: E402

_SCRIPT = _ROOT / "demo" / "demo_llama7b.py"


def _args(**over):
    base = dict(unexplained_info=False, ui_s_c=None, ui_s_b=None, ui_gap_max=None,
                ui_positions=None, ui_lm_sout=None, ui_lm_ow=None,
                statement_out=None, persistent_head=False, enroll_out=None,
                weight_commitment=None, dump_proof=None)
    base.update(over)
    return argparse.Namespace(**base)


class Refused(Exception):
    pass


def _refuse(msg):
    raise Refused(msg)


def test_flags_left_out_keep_main_s_defaults():
    kw = D._bound_and_enrollment_kwargs(_args(), _refuse)
    assert kw == dict(statement_out=None, persistent_head=False,
                      enroll_out=None, weight_commitment=None)


def test_each_flag_reaches_its_keyword():
    kw = D._bound_and_enrollment_kwargs(_args(
        unexplained_info=True, ui_s_c=262144, ui_s_b=128, ui_gap_max=185364,
        statement_out="s.claims", persistent_head=True, weight_commitment="e.wc"), _refuse)
    assert kw == dict(ui_s_c=262144, ui_s_b=128, ui_gap_max=185364,
                      statement_out="s.claims", persistent_head=True,
                      enroll_out=None, weight_commitment="e.wc")
    kw = D._bound_and_enrollment_kwargs(_args(
        unexplained_info=True, ui_positions="37:137", ui_lm_sout=4096, ui_lm_ow=26), _refuse)
    assert kw["ui_positions"] == list(range(37, 137))
    assert (kw["ui_lm_sout"], kw["ui_lm_ow"]) == (4096, 26)
    kw = D._bound_and_enrollment_kwargs(_args(unexplained_info=True, ui_positions="3,5,9"), _refuse)
    assert kw["ui_positions"] == [3, 5, 9]
    kw = D._bound_and_enrollment_kwargs(_args(persistent_head=True, enroll_out="e.wc"), _refuse)
    assert kw["enroll_out"] == "e.wc" and kw["weight_commitment"] is None


@pytest.mark.parametrize("over, says", [
    (dict(ui_s_b=128), "--unexplained-info"),
    (dict(ui_gap_max=1000), "--unexplained-info"),
    (dict(ui_positions="0:4"), "--unexplained-info"),
    (dict(ui_positions=""), "--unexplained-info"),
    (dict(ui_lm_ow=26), "--unexplained-info"),
    (dict(unexplained_info=True, ui_gap_max=0), "positive"),
    (dict(unexplained_info=True, ui_lm_sout=0), "positive"),
    (dict(unexplained_info=True, ui_positions="5:5"), "non-empty"),
    (dict(unexplained_info=True, ui_positions=""), "non-empty"),
    (dict(unexplained_info=True, ui_positions="3,3"), "distinct"),
    (dict(unexplained_info=True, ui_positions="5,2"), "increasing"),
    (dict(unexplained_info=True, ui_positions="-1,2"), "non-negative"),
    (dict(unexplained_info=True, ui_positions="a:b"), "START:STOP"),
    (dict(unexplained_info=True, ui_s_b=-128), "positive"),
    (dict(unexplained_info=True, ui_s_c=1000, ui_s_b=128), "multiple"),
    (dict(unexplained_info=True, ui_s_b=3), "multiple"),
    (dict(enroll_out="e.wc", weight_commitment="f.wc"), "only enrolls"),
    (dict(enroll_out="e.wc", statement_out="s.claims"), "only enrolls"),
    (dict(enroll_out="e.wc", dump_proof="p.json"), "only enrolls"),
    (dict(statement_out="s.claims", dump_proof="p.json"), "no proof to dump"),
])
def test_combinations_that_cannot_run_are_refused(over, says):
    with pytest.raises(Refused, match=says):
        D._bound_and_enrollment_kwargs(_args(**over), _refuse)


def test_the_command_line_lists_the_flags_and_refuses_before_loading():
    run = lambda *a: subprocess.run([sys.executable, str(_SCRIPT), *a],
                                    capture_output=True, text=True, timeout=300)
    help_text = run("--help").stdout
    for flag in ("--ui-s-c", "--ui-s-b", "--ui-gap-max", "--statement-out",
                 "--persistent-head", "--enroll-out", "--weight-commitment",
                 "--ui-positions", "--ui-lm-sout", "--ui-lm-ow"):
        assert flag in help_text
    # A nonexistent model path: reaching main() would fail differently.
    r = run("--from-hf", "/nonexistent", "--ui-s-b", "128")
    assert r.returncode == 2 and "--unexplained-info" in r.stderr
    r = run("--from-hf", "/nonexistent", "--enroll-out", "e.wc", "--weight-commitment", "f.wc")
    assert r.returncode == 2 and "only enrolls" in r.stderr
