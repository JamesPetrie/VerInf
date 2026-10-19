"""The rounding emulations (prover/k2_emulation.py, analysis/k2-fidelity-study-design.md
§3.2), CPU:

- bf16 and E4M3 rounding equal torch's own casts bit for bit (nearest even,
  subnormals, saturation at ±448);
- FP8 activations are scaled per 1×128 group, each group's largest element
  landing on ±448 times its scale;
- every arm crosses exactly the note's boundaries, returns the NumPy
  reference's names and runs on the toy K2 at position 0 and 130,072;
- R-bf16 leaves the router's scores bf16-representable; R-fp32 does not
  round them."""
import pathlib
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import _k2_toy as toy                       # noqa: E402
import fidelity_metrics as fm               # noqa: E402
import k2_emulation as em                   # noqa: E402
import k2_float_reference as kf             # noqa: E402
import k2_loader as kl                      # noqa: E402

IDS = [3, 17, 0, 41, 9, 9, 22, 30, 5, 12, 47, 1]


def _values(seed=0):
    rng = np.random.default_rng(seed)
    x = np.concatenate([rng.normal(0, 1, 4000), rng.normal(0, 1e-3, 2000), rng.normal(0, 300, 2000),
                        np.exp2(rng.integers(-12, 10, 500)) * (1 + rng.integers(0, 16, 500) / 16),
                        [0.0, 1.0625, -1.0625, 1.1875, 448.0, 449.0, 464.0, 1000.0, -1000.0,
                         2.0 ** -9, 2.0 ** -10, 3 * 2.0 ** -11, 1.00390625, 1.01171875]])
    return x.astype(np.float32).astype(np.float64)


def test_bf16_is_torchs_cast():
    x = _values(1)
    want = torch.as_tensor(x, dtype=torch.float32).to(torch.bfloat16).to(torch.float64).numpy()
    assert np.array_equal(em.bf16(x), want)


def test_e4m3_is_torchs_cast():
    x = _values(2)
    want = torch.as_tensor(np.clip(x, -448, 448), dtype=torch.float32) \
        .to(torch.float8_e4m3fn).to(torch.float64).numpy()
    got = em.e4m3(x)
    assert np.array_equal(got, want), x[got != want][:10]
    assert em.e4m3(np.array([1e6, -1e6])).tolist() == [448.0, -448.0]


def test_fp8_groups_scale_per_128():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, (5, 300))
    x[:, 130] = 50.0
    q = em.fp8_groups(x)
    for g0 in (0, 128, 256):
        blk, qb = x[:, g0:g0 + 128], q[:, g0:g0 + 128]
        scale = np.abs(blk).max(axis=1) / 448.0
        assert np.allclose(np.abs(qb).max(axis=1), 448.0 * scale)
        assert np.abs(qb - blk).max() <= np.abs(blk).max() * 2 ** -4 + 1e-12
    assert np.array_equal(em.fp8_groups(np.zeros((2, 4))), np.zeros((2, 4)))


@pytest.fixture(scope="module")
def weights(tmp_path_factory):
    p = tmp_path_factory.mktemp("emu") / "toy.gguf"
    toy.write(p)
    return kl.GgufWeights(str(p), ints=False, cfg=toy.TOY_MLA)


@pytest.mark.parametrize("arm", sorted(em.ARMS))
@pytest.mark.parametrize("offset", [0, 130_072])
def test_every_arm_crosses_every_boundary_and_stays_near_p64(weights, arm, offset):
    cfg, k = toy.TOY_MLA, toy.TOY["k"]
    ref = kf.forward(IDS, weights, cfg, layers=2, offset=offset, k=k)
    P = em.Policy(*em.ARMS[arm])
    got = em.forward(IDS, weights, cfg, P, layers=2, offset=offset, k=k)
    assert P.crossed == set(em.BOUNDARIES)
    assert set(got) == set(ref)
    for name in ref:
        assert np.asarray(got[name]).shape == np.asarray(ref[name]).shape, name
    kl_t = fm.position_metrics(ref["logits"][:-1], got["logits"][:-1], IDS[1:])["kl"]
    assert np.all(np.isfinite(kl_t)) and kl_t.max() < 0.05
    assert fm.residual_error(ref["L1.out"], got["L1.out"]).max() < 0.2
    s = got["L1.s"]
    if P.router == "bf16":
        assert np.array_equal(s, em.bf16(s))
    else:
        assert not np.array_equal(s, em.bf16(s))


@pytest.mark.parametrize("arm", sorted(em.ARMS))
def test_moonlights_variant_runs_every_arm(tmp_path, arm):
    cfg = toy.TOY_MOON_MLA
    p = tmp_path / "moon.gguf"
    toy.write(p, mla=cfg)
    W = kl.GgufWeights(str(p), ints=False, cfg=cfg)
    ref = kf.forward(IDS, W, cfg, layers=2, k=toy.TOY["k"])
    P = em.Policy(*em.ARMS[arm])
    got = em.forward(IDS, W, cfg, P, layers=2, k=toy.TOY["k"])
    assert P.crossed == set(em.BOUNDARIES) and set(got) == set(ref)
    kl_t = fm.position_metrics(ref["logits"][:-1], got["logits"][:-1], IDS[1:])["kl"]
    assert np.all(np.isfinite(kl_t))
    # nothing mixes positions after the last MoE in two layers: a large KL at a
    # position is explained by an expert flip at that position (a bf16 router
    # flips near-ties on this random toy)
    flipped = (ref["L1.top"] != got["L1.top"]).any(axis=1)[:-1]
    assert np.all(flipped[kl_t > 0.01]), (kl_t, flipped)
