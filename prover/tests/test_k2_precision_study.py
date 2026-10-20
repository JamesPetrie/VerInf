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


def test_consecutive_floors_combine_exactly():
    """⌊⌊v/2^a⌋/2^b⌋ = ⌊v/2^(a+b)⌋ for signed v: combined-rounding's values are input-rescale's."""
    v = np.random.default_rng(5).integers(-(1 << 40), 1 << 40, 100_000)
    for a, b in ((2, 2), (12, 2), (14, 2), (16, 4)):
        assert np.array_equal((v >> a) >> b, v >> (a + b))


def test_finer_activations_record_both_implementations(gguf):
    cfg, kk = toy.TOY_MLA, toy.TOY["k"]
    Wf = ps.CachedFloats(kl.GgufWeights(gguf, ints=False, cfg=cfg))
    _, base = ps.forward(IDS, Wf, cfg, ps.CONFIGS["baseline"], kk=kk, head_chunk=20)
    _, a14 = ps.forward(IDS, Wf, cfg, ps.CONFIGS["a14w14"], kk=kk, head_chunk=20)
    assert not any(r.implementation for r in base.rows.values()) and "L0.scores.out" in base.rows
    ir, cr = ps.IMPLEMENTATIONS
    for name in ("L0.n1.x", "L0.sm.x", "L0.ffn.silu.x", "L1.silu.x", "L1.sh.silu.x"):
        a, b = a14.rows[f"{name} [{ir}]"], a14.rows[f"{name} [{cr}]"]
        assert (a.min, a.max) == (b.min, b.max) and (a.lo, a.hi) != (b.lo, b.hi), name
    # the scores and gates at 2^14 exist only where softmax and SiLU take them through s_in
    for name in ("L0.scores.out", "L0.ffn.g.out", "L1.g.out", "L1.sh.g.out"):
        assert f"{name} [{ir}]" in a14.rows and name not in a14.rows, name
    for name in ("L0.r1", "L0.x_out", "L1.r1_y", "L1.x_out", "x0", "L1.D"):
        assert name in a14.rows and ki.FIELD in a14.rows[name].claim, name


def test_the_input_rescale_windows_are_the_claims():
    op = ps.Mixed(ps.CONFIGS["a16"], kl.K2_INT)
    op.rmsnorm(np.full((1, 64), 9 << 16, dtype=np.int64), "n")     # 9.0: 9·2^12 > 2^15 at s_norm
    assert set(op.ranges.violations("input-rescale")) == {"n.x [input-rescale]"}
    assert not op.ranges.violations("combined-rounding")
    assert ps.softmax_s_in(kl.K2_INT)[:2] == (-(1 << 15), (1 << 24) - (1 << 15) - 1)
    f = ps.feasibility(op.ranges)
    assert not f["implementations"]["input-rescale"]["ok"] and f["implementations"]["combined-rounding"]["ok"]
