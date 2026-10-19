"""The passage freeze (prover/fidelity_passages.py), CPU: lengths outside the
target need the authors' recorded acceptance, the cap and special-token text
never pass, one template for all, and a freeze is never overwritten. The
rendering itself (llama.cpp) is test_moonlight_integration.py's."""
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import fidelity_passages as fp             # noqa: E402


def _row(id_, n, special=False, template="t"):
    return dict(id=id_, role="additional", n_prompt_tokens=n, content_has_special_tokens=special,
                template_sha256=template, ids=list(range(n)), ids_sha256=fp.ids_sha256(range(n)))


def test_problems():
    assert fp.problems([_row("a", 450), _row("b", 400), _row("c", 500)]) == []
    assert fp.problems([_row("a", 378)]) == ["a: 378 prompt tokens, outside the 400-500 target"]
    assert any("cap" in p for p in fp.problems([_row("a", 501)]))
    assert any("special-token" in p for p in fp.problems([_row("a", 450, special=True)]))
    assert any("templates" in p for p in fp.problems([_row("a", 450), _row("b", 450, template="u")]))


def test_freeze_needs_acceptance_for_short_prompts_and_never_overwrites(tmp_path):
    prompts = tmp_path / "prompts.json"
    prompts.write_text("{}")
    rows = [_row("a", 378), _row("b", 450)]
    with pytest.raises(ValueError, match="not frozen"):
        fp.freeze(prompts, rows, tmp_path / "f.json", model_provenance={})
    rec = fp.freeze(prompts, rows, tmp_path / "f.json", model_provenance={"k": 1}, accept_lengths=True)
    assert rec["lengths_accepted_outside_target"] and rec["prompts_sha256"] == fp.sha256(b"{}")
    assert json.loads((tmp_path / "f.json").read_text())["passages"][0]["ids_sha256"] == rows[0]["ids_sha256"]
    with pytest.raises(FileExistsError):
        fp.freeze(prompts, rows, tmp_path / "f.json", model_provenance={}, accept_lengths=True)


def test_the_cap_is_never_accepted(tmp_path):
    prompts = tmp_path / "prompts.json"
    prompts.write_text("{}")
    with pytest.raises(ValueError, match="cap"):
        fp.freeze(prompts, [_row("a", 520)], tmp_path / "f.json", model_provenance={}, accept_lengths=True)


def test_a_changed_user_text_is_refused(tmp_path):
    doc = {"passages": [{"id": "a", "role": "proved", "user_content_sha256": "0" * 64,
                         "messages": [{"role": "user", "content": "changed"}]}]}
    (tmp_path / "p.json").write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="does not match its recorded sha256"):
        fp.render_all(tmp_path / "p.json", "unused", "unused", tmp_path / "w")


def test_verify_names_every_difference(tmp_path):
    rows = [_row("a", 450), _row("b", 420)]
    frozen = tmp_path / "frozen.json"
    frozen.write_text(json.dumps({"passages": [dict(id="a", ids_sha256=rows[0]["ids_sha256"]),
                                               dict(id="b", ids_sha256="0" * 64),
                                               dict(id="c", ids_sha256="1" * 64)]}))
    assert fp.verify(rows, frozen) == [("b", "0" * 64, rows[1]["ids_sha256"]), ("c", "1" * 64, None)]
    frozen.write_text(json.dumps({"passages": [dict(id=r["id"], ids_sha256=r["ids_sha256"]) for r in rows]}))
    assert fp.verify(rows, frozen) == []
