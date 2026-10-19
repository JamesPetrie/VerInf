"""The precision study's integer models (prover/k2_precision_study.py), CPU:
the baseline configuration reproduces the validated integer reference
(k2_int_reference.forward) exactly at every name both compute, so any
difference the study reports is the scale change's; the digit-split matmul is
exact; every configuration runs on the toy GGUF, and finer scales do not lose
fidelity there."""
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import _k2_toy as toy                       # noqa: E402
import k2_float_reference as kf             # noqa: E402
import k2_int_reference as ki               # noqa: E402
import k2_loader as kl                      # noqa: E402
import k2_precision_study as ps             # noqa: E402

IDS = [3, 17, 0, 41, 9, 9, 22, 30]


@pytest.fixture(scope="module")
def gguf(tmp_path_factory):
    path = tmp_path_factory.mktemp("k2prec") / "toy.gguf"
    toy.write(path)
    return str(path)


@pytest.mark.parametrize("bits", [10, 30, 45])
def test_mm_is_exact(bits):
    rng = np.random.default_rng(bits)
    a = rng.integers(-(1 << bits), 1 << bits, (4, 33))
    b = rng.integers(-(1 << 20), 1 << 20, (33, 5))
    want = np.matmul(a.astype(object), b.astype(object))
    got = ps.mm(a, b)
    assert all(int(x) == int(y) for x, y in zip(np.ravel(got), np.ravel(want)))


@pytest.mark.parametrize("offset", [0, 130_072])
def test_the_baseline_is_the_validated_reference(gguf, offset):
    cfg, kk = toy.TOY_MLA, toy.TOY["k"]
    ref, _ = ki.forward(IDS, kl.GgufWeights(gguf, cfg=cfg), cfg, layers=2, offset=offset,
                        kk=kk, head_chunk=20)
    _equal_at_shared_names(ref, gguf, cfg, kk, offset)


def test_without_query_lora_or_yarn_the_baseline_is_the_integer_reference(tmp_path):
    """Moonlight's variant: the integer reference with the model's constants."""
    cfg, kk = toy.TOY_MOON_MLA, toy.TOY["k"]
    path = tmp_path / "moon.gguf"
    toy.write(path, mla=cfg)
    ref, _ = ki.forward(IDS, kl.GgufWeights(str(path), cfg=cfg), cfg, layers=2, kk=kk,
                        head_chunk=20, k=kl.int_constants(cfg))
    assert "L0.aqg" not in ref or np.array_equal(ref["L0.aqg"], ref["L0.n1g"])
    _equal_at_shared_names(ref, str(path), cfg, kk, 0)


def _equal_at_shared_names(ref, gguf, cfg, kk, offset):
    Wf = ps.CachedFloats(kl.GgufWeights(gguf, ints=False, cfg=cfg))
    got, _ = ps.forward(IDS, Wf, cfg, ps.CONFIGS["baseline"], offset=offset, kk=kk, head_chunk=20)
    shared = sorted(set(ref) & set(got))
    assert {"x0", "L0.n1g", "L0.r1", "L0.ffn", "L1.n2g", "L1.s", "L1.mask", "L1.y",
            "L1.sh", "final.ng", "logits"} <= set(shared)
    for name in shared:
        a = np.asarray(ref[name]).astype(object).ravel()
        b = np.asarray(got[name]).astype(object).ravel()
        assert a.shape == b.shape and all(int(x) == int(y) for x, y in zip(a, b)), name


def test_every_configuration_runs_and_finer_scales_do_not_lose_fidelity(gguf):
    cfg, kk = toy.TOY_MLA, toy.TOY["k"]
    rows = ps.study(gguf, IDS, cfg=cfg, kk=kk, log=lambda *_: None)
    assert set(rows) == set(ps.CONFIGS)
    rel = {n: r["fidelity"]["logits"]["rel_l2"] for n, r in rows.items()}
    assert rel["a16w16"] < rel["baseline"] and rel["w16"] <= rel["baseline"] * 1.05
    assert rows["baseline"]["rmsnorm_bracket_ok"] and not rows["a16w16n14"]["rmsnorm_bracket_ok"]
    assert all(r["field_ok"] for r in rows.values())
