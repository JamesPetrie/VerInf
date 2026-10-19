"""Moonlight-16B-A3B-Instruct, the integration fixture of
analysis/k2-fidelity-study-design.md §8: VERINF_MOONLIGHT names the pinned
Q4_K_M GGUF and VERINF_LLAMA_RUNNER the runner; without them the suite is
skipped. About two minutes (the full model loads once, memory-mapped).

- the file is the pinned one;
- llama.cpp's deepseek2 loader treats 27 layers as "lite" and never reads
  the query-LoRA rank. Truncated to two layers, Moonlight leaves that branch:
  without q_lora_rank = 0 llama.cpp refuses it, and with the fix its
  hyperparameters are the full model's on every key a truncation keeps."""
import os
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import gguf_truncate as gt                  # noqa: E402
import llama_runner as lr                   # noqa: E402

MOONLIGHT = os.environ.get("VERINF_MOONLIGHT")
RUNNER = os.environ.get("VERINF_LLAMA_RUNNER")
pytestmark = pytest.mark.skipif(not (MOONLIGHT and RUNNER),
                                reason="VERINF_MOONLIGHT and VERINF_LLAMA_RUNNER not both set")
PIN_SHA256 = "2b113655999f11730a6dc31b7ba93463e6523a323f46074b01c55d96101df360"


@pytest.fixture(scope="module")
def full_fingerprint(tmp_path_factory):
    return lr.run(RUNNER, MOONLIGHT, tmp_path_factory.mktemp("full") / "info",
                  mode="info")["fingerprint"]


def test_the_fixture_is_the_pinned_file():
    assert lr.file_sha256(MOONLIGHT) == PIN_SHA256


def test_full_moonlight_is_lite_without_query_lora(full_fingerprint):
    assert full_fingerprint["n_layer"] == "27" and full_fingerprint["n_lora_q"] == "0"
    assert "deepseek2.attention.q_lora_rank" not in gt.metadata(MOONLIGHT)


def test_two_layers_load_on_the_full_branch_only_with_the_fix(full_fingerprint, tmp_path):
    rec = gt.truncate(MOONLIGHT, tmp_path / "fixed.gguf", 2)
    assert rec["added_keys"] == {"deepseek2.attention.q_lora_rank": 0}
    fixed = lr.run(RUNNER, tmp_path / "fixed.gguf", tmp_path / "fixed", mode="info")["fingerprint"]
    assert fixed["n_layer"] == "2" and lr.branch_differences(full_fingerprint, fixed) == {}
    gt.truncate(MOONLIGHT, tmp_path / "unfixed.gguf", 2, branch_fix=lambda md, t: {})
    with pytest.raises(RuntimeError):
        lr.run(RUNNER, tmp_path / "unfixed.gguf", tmp_path / "unfixed", mode="info")
    log = (tmp_path / "unfixed" / "load.log").read_text()
    assert "key not found in model: deepseek2.attention.q_lora_rank" in log


def test_render_applies_the_gguf_template_as_jinja2_does(tmp_path):
    """verinf_render (llama.cpp's Jinja engine) against an independent jinja2
    rendering of the same template; special tokens become single ids."""
    import json
    import subprocess
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    render = pathlib.Path(RUNNER).with_name("verinf_render")
    content = "Name three prime numbers above 100, one per line."
    (tmp_path / "user.txt").write_text(content)
    subprocess.run([str(render), "--model", MOONLIGHT, "--content", str(tmp_path / "user.txt"),
                    "--out", str(tmp_path)], check=True)
    rj = json.loads((tmp_path / "render.json").read_text())
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    want = env.from_string(rj["template"]).render(
        messages=[{"role": "user", "content": content}], add_generation_prompt=True)
    assert (tmp_path / "prompt.txt").read_text() == want
    ids = [int(x) for x in (tmp_path / "ids.txt").read_text().split()]
    assert rj["n_tokens"] == len(ids) and not rj["content_has_special_tokens"]
    assert len(ids) < len(want) // 2                  # tags are single tokens, not characters
