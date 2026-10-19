"""The torch float64 reference (prover/k2_float_torch.py) against the NumPy
one (k2_float_reference), CPU, on the toy K2 (analysis/k2-fidelity-study-design.md
§5.3): the same decoded weights by construction (one provider), every
intermediate both compute equal to float64 summation order, the routing
identical, at position 0 and at 130,072 (YaRN); `keep` limits what comes
back; a tie in the selection goes to the lowest index in both."""
import pathlib
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import _k2_toy as toy                       # noqa: E402
import k2_float_reference as kf             # noqa: E402
import k2_float_torch as kt                 # noqa: E402
import k2_loader as kl                      # noqa: E402

IDS = [3, 17, 0, 41, 9, 9, 22, 30, 5, 12, 47, 1]


@pytest.fixture(scope="module", params=["k2", "moonlight"])
def variant(request, tmp_path_factory):
    cfg = toy.TOY_MLA if request.param == "k2" else toy.TOY_MOON_MLA
    p = tmp_path_factory.mktemp("k2torch") / f"{request.param}.gguf"
    toy.write(p, mla=cfg)
    return kl.GgufWeights(str(p), ints=False, cfg=cfg), cfg


@pytest.fixture(scope="module")
def weights(tmp_path_factory):
    p = tmp_path_factory.mktemp("k2torch") / "toy.gguf"
    toy.write(p)
    return kl.GgufWeights(str(p), ints=False, cfg=toy.TOY_MLA)


@pytest.mark.parametrize("offset", [0, 130_072])
def test_every_intermediate_is_the_numpy_references(variant, offset):
    weights, cfg = variant
    k = toy.TOY["k"]
    ref = kf.forward(IDS, weights, cfg, layers=2, offset=offset, k=k, head_chunk=20)
    got = kt.forward(IDS, weights, cfg, layers=2, offset=offset, k=k, head_chunk=20)
    assert set(ref) == set(got)
    for name in ref:
        a, b = np.asarray(ref[name]), np.asarray(got[name])
        assert a.shape == b.shape, name
        if name.endswith(".top"):
            assert np.array_equal(a, b), name
        else:
            scale = max(np.abs(a).max(), 1e-300)
            assert np.abs(a - b).max() <= 1e-12 * scale, (name, np.abs(a - b).max() / scale)


def test_keep_limits_what_comes_back(weights):
    got = kt.forward(IDS, weights, toy.TOY_MLA, layers=2, k=toy.TOY["k"],
                     keep={"L0.out", "L1.out", "logits"})
    assert set(got) == {"L0.out", "L1.out", "logits"}
    only = kt.forward(IDS, weights, toy.TOY_MLA, layers=2, k=toy.TOY["k"],
                      keep=lambda n: n.endswith(".top"))
    assert set(only) == {"L1.top"}


def test_a_tie_goes_to_the_lowest_index_in_both():
    s = np.array([[0.5, 0.7, 0.7, 0.7, 0.1]])
    b = np.zeros(5)
    top_np, _ = kf.select(s, b, 2)
    top_t, _ = kt.select(torch.as_tensor(s), torch.as_tensor(b), 2)
    assert top_np.tolist() == top_t.tolist() == [[1, 2]]


def test_device_decoding_gives_the_cpu_providers_weights_and_the_same_forward(variant):
    """DeviceWeights (gguf_dequant_torch) against the CPU provider: every
    accessor bit-identical, so the torch forward is bit-identical too."""
    weights, cfg = variant
    D = kt.DeviceWeights(weights, "cpu")
    for il in (0, 1):
        pairs = list(weights.dense(il).items()) if weights.kind(il) == "dense" else \
            list(weights.shared(il).items()) + [(f"{k}{e}", weights.expert(il, k, e))
                                                for k in ("gate", "up", "down") for e in (0, 5)]
        dev = D.dense(il) if weights.kind(il) == "dense" else \
            {**D.shared(il), **{f"{k}{e}": D.expert(il, k, e) for k in ("gate", "up", "down") for e in (0, 5)}}
        for name, a in pairs:
            assert np.array_equal(dev[name].numpy(), a), (il, name)
    assert np.array_equal(D.head(3, 40).numpy(), weights.head(3, 40))
    a = kt.forward(IDS, weights, cfg, layers=2, k=toy.TOY["k"])
    b = kt.forward(IDS, D, cfg, layers=2, k=toy.TOY["k"])
    assert all(np.array_equal(a[n], b[n]) for n in a)


def test_the_p64_check_on_the_cpu(tmp_path):
    import fidelity_study as fs
    p = tmp_path / "toy.gguf"
    toy.write(p)
    rec = fs.p64_check(dict(cfg=toy.TOY_MLA, k=toy.TOY["k"]), p, IDS, device="cpu")
    assert rec["max_relative"] < 1e-12 and rec["routing_tokens_differing"] == 0
