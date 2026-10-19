"""Truncated models for the engine arm (prover/gguf_truncate.py), CPU, on the
toy K2 GGUF (tests/_k2_toy.py):

- the first n blocks and the globals are kept, quantized bytes unchanged, the
  block count rewritten, split keys dropped and per-layer arrays cut;
- a model without its output head is refused, never given the embedding;
- a block missing a tensor (a shard absent) is refused, not loaded short;
- the deepseek2 branch fix writes q_lora_rank = 0 only for a model without
  query-LoRA whose GGUF lacks the key.
The load in llama.cpp and the branch fingerprints are test_llama_engine.py's."""
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import _k2_toy as toy                       # noqa: E402
import gguf_truncate as gt                  # noqa: E402


@pytest.fixture(scope="module")
def toy_path(tmp_path_factory):
    p = tmp_path_factory.mktemp("trunc") / "toy.gguf"
    toy.write(p)
    return p


def _tensors(path):
    from gguf import GGUFReader
    return {t.name: t for t in GGUFReader(str(path)).tensors}


def test_one_block_kept_bytes_unchanged_and_metadata_rewritten(toy_path, tmp_path):
    out = tmp_path / "one.gguf"
    rec = gt.truncate(toy_path, out, 1)
    src, dst = _tensors(toy_path), _tensors(out)
    want = set(gt.GLOBALS_REQUIRED) | gt.deepseek2_block_names(0, dense_lead=1, q_lora=True)
    assert set(dst) == want == set(rec["tensors"])
    for n in dst:
        assert dst[n].tensor_type == src[n].tensor_type
        assert np.asarray(dst[n].data).tobytes() == np.asarray(src[n].data).tobytes(), n
    md = gt.metadata(out)
    assert md["deepseek2.block_count"] == 1 and md["deepseek2.leading_dense_block_count"] == 1
    assert md["deepseek2.attention.q_lora_rank"] == toy.TOY_MLA["q_rank"]
    assert not any(k.startswith("split.") for k in md) and rec["added_keys"] == {}


def test_both_blocks_is_the_whole_toy(toy_path, tmp_path):
    rec = gt.truncate(toy_path, tmp_path / "two.gguf", 2)
    assert set(rec["tensors"]) == set(_tensors(toy_path))


def test_a_model_without_its_head_is_refused(tmp_path):
    p = tmp_path / "headless.gguf"
    toy.write(p, drop=("output.weight",))
    with pytest.raises(ValueError, match="output.weight"):
        gt.truncate(p, tmp_path / "x.gguf", 1)


def test_a_block_missing_a_tensor_is_refused(tmp_path):
    p = tmp_path / "short.gguf"
    toy.write(p, drop=("blk.1.ffn_up_exps.weight",))
    gt.truncate(p, tmp_path / "ok.gguf", 1)                     # block 0 is whole
    with pytest.raises(ValueError, match=r"block 1: missing \['blk.1.ffn_up_exps.weight'\]"):
        gt.truncate(p, tmp_path / "x.gguf", 2)


def test_layer_count_bounds(toy_path, tmp_path):
    for n in (0, 3):
        with pytest.raises(ValueError):
            gt.truncate(toy_path, tmp_path / "x.gguf", n)


def test_the_deepseek2_branch_fix():
    with_q = {"blk.0.attn_q_a.weight": None}
    without_q = {"blk.0.attn_q.weight": None}
    md = {"general.architecture": "deepseek2"}
    assert gt.deepseek2_branch_fix(md, with_q) == {}
    assert gt.deepseek2_branch_fix(md, without_q) == {"deepseek2.attention.q_lora_rank": 0}
    stated = dict(md, **{"deepseek2.attention.q_lora_rank": 0})
    assert gt.deepseek2_branch_fix(stated, without_q) == {}


def test_shard_sets_are_found_by_pattern(tmp_path):
    for i in (1, 2, 3):
        (tmp_path / f"m-{i:05d}-of-00003.gguf").write_bytes(b"")
    assert [p.name for p in gt.shard_paths(tmp_path / "m-00002-of-00003.gguf")] == \
        [f"m-{i:05d}-of-00003.gguf" for i in (1, 2, 3)]
    assert gt.shard_paths(tmp_path / "single.gguf") == [tmp_path / "single.gguf"]


def test_metadata_only_from_a_head_keeps_every_key_but_split(toy_path, tmp_path):
    """The key-values come from the head of the file alone (the tensor data
    cut off), as from the pinned K2 shard's first bytes."""
    data = toy_path.read_bytes()
    head = tmp_path / "head.bin"
    head.write_bytes(data[: len(data) // 4])
    _, n_tensors, kv = gt.read_head_metadata(head)
    assert n_tensors == len(_tensors(toy_path))
    out = tmp_path / "meta.gguf"
    gt.metadata_only(head, out)
    full, meta = gt.metadata(toy_path), gt.metadata(out)
    assert meta == {k: v for k, v in full.items() if k not in gt.SPLIT_KEYS}
    assert _tensors(out) == {}


def test_a_head_cut_inside_the_key_values_is_refused(toy_path, tmp_path):
    head = tmp_path / "short.bin"
    head.write_bytes(toy_path.read_bytes()[:200])
    with pytest.raises(ValueError, match="ends inside the key-values"):
        gt.read_head_metadata(head)


def test_the_expected_fingerprint_from_metadata(toy_path, tmp_path):
    md = gt.metadata(toy_path)
    exp = gt.deepseek2_expected_fingerprint(md)
    assert exp["n_lora_q"] == str(toy.TOY_MLA["q_rank"]) and exp["n_layer_dense_lead"] == "1"
    assert exp["expert_gating_func"] == "sigmoid" and exp["rope scaling"] == "yarn"
    assert exp["n_expert"] == str(toy.TOY["E"]) and exp["n_expert_used"] == str(toy.TOY["k"])
    moon = tmp_path / "moon.gguf"
    toy.write(moon, mla=toy.TOY_MOON_MLA)
    assert gt.deepseek2_expected_fingerprint(gt.metadata(moon))["n_lora_q"] == "0"
