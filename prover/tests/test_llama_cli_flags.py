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
    base = dict(from_hf="fixture", unexplained_info=False, ui_s_c=None, ui_s_b=None, ui_gap_max=None,
                ui_positions=None, ui_lm_sout=None, ui_lm_ow=None,
                statement_out=None, persistent_head=False, enroll_out=None,
                weight_commitment=None, dump_proof=None, forward_only=False)
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


@pytest.mark.parametrize("option", [
    {"ui_positions": [0]}, {"ui_gap_max": 128}, {"ui_s_c": D.UI_S_C},
    {"ui_s_b": 1 << D.UI_S_B_BITS}, {"ui_lm_sout": D.UI_COARSE_S}, {"ui_lm_ow": D.UI_COARSE_OW},
])
def test_python_preview_rejects_scoring_options_without_scoring(tmp_path, option):
    with pytest.raises(ValueError, match="requires --unexplained-info"):
        D.main(from_hf="/nonexistent", statement_out=tmp_path / "claims", **option)
    assert list(tmp_path.iterdir()) == []


def test_each_flag_reaches_its_keyword():
    kw = D._bound_and_enrollment_kwargs(_args(
        unexplained_info=True, ui_s_c=262144, ui_s_b=128, ui_gap_max=185364,
        statement_out="s.claims", persistent_head=True, weight_commitment="e.wc"), _refuse)
    assert kw == dict(ui_s_c=262144, ui_s_b=128, ui_gap_max=185364,
                      statement_out="s.claims", persistent_head=True,
                      enroll_out=None, weight_commitment="e.wc")
    kw = D._bound_and_enrollment_kwargs(_args(
        unexplained_info=True, ui_positions="37:137", ui_lm_sout=4096, ui_lm_ow=26), _refuse)
    assert list(kw["ui_positions"]) == list(range(37, 137))
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
    (dict(unexplained_info=True, ui_lm_sout=3), "power of two"),
    (dict(unexplained_info=True, ui_lm_sout=D.S * D.S), "power of two"),
    (dict(unexplained_info=True, ui_positions="5:5"), "non-empty"),
    (dict(unexplained_info=True, ui_positions=""), "non-empty"),
    (dict(unexplained_info=True, ui_positions="3,3"), "distinct"),
    (dict(unexplained_info=True, ui_positions="5,2"), "increasing"),
    (dict(unexplained_info=True, ui_positions="-1,2"), "non-negative"),
    (dict(unexplained_info=True, ui_positions="a:b"), "START:STOP"),
    (dict(unexplained_info=True, ui_s_b=-128), "positive"),
    (dict(unexplained_info=True, ui_s_c=1000, ui_s_b=128), "multiple"),
    (dict(unexplained_info=True, ui_s_c=12288), "power-of-two"),
    (dict(unexplained_info=True, ui_s_b=3), "multiple"),
    (dict(from_hf=None, enroll_out="e.wc"), "require --from-hf"),
    (dict(from_hf=None, weight_commitment="e.wc"), "require --from-hf"),
    (dict(enroll_out="e.wc", weight_commitment="f.wc"), "only enrolls"),
    (dict(enroll_out="e.wc", statement_out="s.claims"), "only enrolls"),
    (dict(enroll_out="e.wc", dump_proof="p.json"), "only enrolls"),
    (dict(statement_out="s.claims", dump_proof="p.json"), "no proof to dump"),
    (dict(forward_only=True, statement_out="s.claims"), "--forward-only"),
    (dict(forward_only=True, enroll_out="e.wc"), "--forward-only"),
    (dict(forward_only=True, weight_commitment="e.wc"), "--forward-only"),
    (dict(forward_only=True, dump_proof="p.json"), "--forward-only"),
])
def test_combinations_that_cannot_run_are_refused(over, says):
    with pytest.raises(Refused, match=says):
        D._bound_and_enrollment_kwargs(_args(**over), _refuse)


def test_the_command_line_lists_the_flags_and_refuses_before_loading(monkeypatch):
    monkeypatch.delenv("LIGERO_DUMP_PROOF", raising=False)
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
    r = run("--from-hf", "/nonexistent", "--weight-commitment", "e.wc")
    assert r.returncode == 2 and "requires --dump-proof or LIGERO_DUMP_PROOF" in r.stderr


@pytest.mark.parametrize("options, length", [
    ({}, 5), ({"token_ids": [1, 2]}, 2), ({"chat": True, "prompt": "hello"}, 3),
])
def test_positions_use_effective_sequence_before_model_loading(monkeypatch, options, length):
    import loader
    monkeypatch.setattr(D, "SEQ", 5)
    monkeypatch.setattr(loader, "tokenize_chat_prompt", lambda *a: [4, 2, 1])
    with pytest.raises(ValueError, match=f"outside the {length}-token sequence"):
        D.main(from_hf="/nonexistent", unexplained_info=True, ui_positions=[length], **options)
    D._check_positions(range(length), length)


@pytest.mark.parametrize("text", ["", "1,,2", "1,", "1.5", "abc"])
def test_explicit_output_list_cannot_be_parsed_as_greedy(text):
    with pytest.raises(argparse.ArgumentTypeError, match="non-empty"):
        D._parse_output_tokens(text)
    assert D._parse_output_tokens("0, 1") == [0, 1]


@pytest.mark.parametrize("tokens, message", [
    ([], "exactly 2"), ([0], "exactly 2"), ([0, 1, 2], "exactly 2"),
    ([-1, 0], "outside vocabulary"), ([0, 32000], "outside vocabulary"),
    ([0, 1.5], "integer IDs"), ([False, 1], "integer IDs"),
])
def test_output_tokens_rejected_before_model_tensors(monkeypatch, tokens, message):
    from max_claim import max_gap
    monkeypatch.setattr(D, "SEQ", 10)
    monkeypatch.setattr(D.ModelConfig, "from_hf", lambda _: D.RANDOM_WEIGHTS_CFG)
    with pytest.raises(ValueError, match=message):
        D.main(from_hf="fixture", token_ids=[2, 3], unexplained_info=True, ui_output_tokens=tokens)
    with pytest.raises(ValueError, match=message):
        D._run_unexplained_info(None, None, seq=2, vocab_size=32000, output_tokens=tokens)
    with pytest.raises(ValueError, match=message):
        max_gap(None, None, tokens, T=2, V=32000, gap_max=8)


def test_command_line_refuses_empty_output_list_before_loading():
    run = subprocess.run([sys.executable, str(_SCRIPT), "--from-hf", "/nonexistent",
                          "--unexplained-info", "--ui-output-tokens", ""],
                         capture_output=True, text=True, timeout=60)
    assert run.returncode == 2 and "--ui-output-tokens" in run.stderr and "non-empty" in run.stderr


def test_output_tokens_require_scoring():
    with pytest.raises(ValueError, match="requires --unexplained-info"):
        D.main(from_hf="/nonexistent", ui_output_tokens=[0, 1])


@pytest.mark.parametrize("positions", [[0.1, 0.9], [0, 1.0], [False, 1]])
def test_noninteger_positions_refused_before_model_loading(positions):
    with pytest.raises(ValueError, match="integer indices"):
        D.main(from_hf="/nonexistent", unexplained_info=True, ui_positions=positions)


def test_strided_positions_match_explicit_indices():
    import numpy as np

    positions = range(0, 6, 2)
    expected = [0, 2, 4]
    assert list(D._check_positions(positions, seq=5)) == expected
    indices = D._check_positions(np.array(expected), seq=5)
    assert indices == expected and all(type(i) is int for i in indices)
    with pytest.raises(ValueError, match="outside"):
        D._check_positions(positions, seq=4)


def test_position_range_checked_before_expansion():
    positions = D._parse_positions("0:1000000000000", _refuse)
    with pytest.raises(ValueError, match="outside"):
        D.main(from_hf="/nonexistent", unexplained_info=True, ui_positions=positions)


def test_main_rejects_invalid_lm_scale_before_model_loading():
    with pytest.raises(ValueError, match="--ui-lm-sout"):
        D.main(from_hf="/nonexistent", unexplained_info=True, ui_lm_sout=D.S * D.S)


def test_main_rejects_forward_only_statement_before_loading():
    with pytest.raises(ValueError, match="--forward-only"):
        D.main(from_hf="/nonexistent", forward_only=True, statement_out="s.claims")


def test_main_rejects_unsupported_bound_scale_before_loading():
    with pytest.raises(ValueError, match="power-of-two"):
        D.main(from_hf="/nonexistent", unexplained_info=True, ui_s_c=12288)
