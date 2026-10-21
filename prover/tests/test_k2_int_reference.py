"""Kimi K2's integer reference (prover/k2_int_reference.py) on the CPU.

Exactness first, where the prover's own computation runs without CUDA:
- the softmax equals the prover's witness builder (tape._softmax_witness_vec)
  on causal multi-head rows, saturating spreads included;
- RoPE equals the prover's compute function (compute_fns.rope_compute) with
  K2's YaRN at short and at long positions;
- the products are exact on all three paths (BLAS, int64, limbs);
- the routed combine equals topk_reference.combine.
Then the definitions where the prover's code needs CUDA: the RMSNorm y is
the least y with y²·S_tot ≥ d·s⁴, SiLU's two branches, the top-k width
bounds every tiebroken difference, and an out-of-window value is recorded
as a violation, not raised.
Last, the two-layer forward on the toy GGUF against the float reference:
fidelity and routing differences, reported apart (GPU exactness against the
tape is tests/test_k2_driver.py)."""
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import _k2_toy as toy                                     # noqa: E402
import k2_float_reference as kf                           # noqa: E402
import k2_int_reference as ki                             # noqa: E402
import k2_loader as kl                                    # noqa: E402
import topk_reference as tref                             # noqa: E402
from k2_attention import K2_MLA                           # noqa: E402
from topk_params import P                                 # noqa: E402

S = kl.S_ACT


def _signed(u):
    u = np.asarray(u, dtype=np.uint64).astype(object)
    return np.asarray([int(v) - P if int(v) > P // 2 else int(v) for v in u.ravel()],
                      dtype=np.int64).reshape(np.shape(u))


@pytest.mark.parametrize("bits", [10, 22, 40])
def test_mm_is_exact_on_every_path(bits):
    rng = np.random.default_rng(bits)
    a = rng.integers(-(1 << bits), 1 << bits, (5, 37))
    b = rng.integers(-(1 << bits), 1 << bits, (37, 4))
    want = np.matmul(a.astype(object), b.astype(object))
    got = ki.mm(a, b)
    assert all(int(x) == int(y) for x, y in zip(np.ravel(got), np.ravel(want)))


def test_softmax_equals_the_prover_witness():
    from tape import _softmax_witness_vec, _to_field_np
    import claims
    T, H = 9, 3
    rng = np.random.default_rng(1)
    x = rng.normal(0, 3, (T, H * T)) * S
    x[2, :] += rng.normal(0, 150, H * T) * S           # spreads past Z_max: saturating
    x = np.rint(x).astype(np.int64)
    op = ki.Int()
    got = op.softmax(x, T=T, H=H, name="sm", rows_per_chunk=5)
    cfg = claims.SoftmaxConfig(B=1, M=1, s_x=S, s_c=S, s_y=S, delta=1,
                               Z_max=kl.K2_INT["z_max"], saturate=True)
    T_A, T_B = claims._softmax_exp_tables(cfg)
    w = _softmax_witness_vec(_to_field_np(x.reshape(-1)), B=T * H, M=T, s_x=S, s_c=S, s_y=S,
                             T_A_np=T_A, T_B_np=T_B, Z_max=kl.K2_INT["z_max"],
                             aux_chunk_width=24, saturate=True, Z_high_width=16,
                             causal=True, heads=H)
    assert np.array_equal(got.reshape(-1), w["y_A"].astype(np.int64))
    assert op.ranges.rows["sm.keys_past_table"][0] > 0      # the saturating row was exercised


@pytest.mark.parametrize("offset", [0, 130_072])
def test_rope_equals_the_prover_compute(offset):
    import torch
    import compute_fns
    from claims import RoPEClaim, RoPEConfig
    from core import Variable
    T, heads, dr = 5, 3, K2_MLA["d_rope"]
    rng = np.random.default_rng(offset)
    x = rng.integers(-(1 << 20), 1 << 20, (T, heads * dr))
    cfg = RoPEConfig(SEQ=T, d_h=dr, s_x=S, base=K2_MLA["base"], position_offset=offset,
                     heads=heads, scale_factor=K2_MLA["factor"],
                     original_max_pos=K2_MLA["original_max_pos"], yarn=True,
                     yarn_beta_fast=K2_MLA["beta_fast"], yarn_beta_slow=K2_MLA["beta_slow"],
                     yarn_mscale=K2_MLA["mscale"], yarn_mscale_all_dim=K2_MLA["mscale_all_dim"])
    from tape import _to_field_np
    xv, rv = Variable("x", length=x.size), Variable("xr", length=x.size)
    x_field = torch.from_numpy(_to_field_np(x.reshape(-1)))      # signed values mod P
    full = compute_fns.rope_compute(RoPEClaim(x=xv, x_rot=rv, config=cfg), {xv: x_field})[rv]
    want = _signed(full.numpy()) >> 12
    cos, sin = ki.rope_tables(K2_MLA, T=T, offset=offset)
    got = ki.Int().rope(x, cos, sin, T=T, heads=heads, d_h=dr, name="r")
    assert np.array_equal(got.reshape(-1), want)
    stripped = ki.rope_tables(K2_MLA, T=T, offset=offset, yarn=False)
    if offset:                                   # YaRN changes the long-position tables
        assert not np.array_equal(stripped[0], cos)


def test_rmsnorm_is_the_least_y_and_close_to_float():
    rng = np.random.default_rng(3)
    d = 512
    x = np.rint(rng.normal(0, 2.0, (6, d)) * S).astype(np.int64)
    x[5] = 0                                       # the ε floor: y = y_max
    import math
    op = ki.Int()
    out = op.rmsnorm(x, "n")
    magic = d * S ** 4
    for row in range(6):
        st = int((x[row].astype(object) ** 2).sum()) + d * kl.K2_INT["eps_int"]
        # the least y >= 1 with y² >= ⌈magic / st⌉, in closed form
        y = math.isqrt(-(-magic // st) - 1) + 1
        assert y * y * st >= magic and (y - 1) ** 2 * st < magic
        assert np.array_equal(out[row], (x[row] * y) >> 12)
    real = x[:5] / S
    want = real / np.sqrt((real ** 2).mean(axis=1, keepdims=True) + kl.K2_INT["eps"])
    assert np.abs(out[:5] / S - want).max() < 2e-3
    assert op.ranges.rows["n.y"][0] == op.ranges.rows["n.y"][1] - 1     # the zero row hit y_max


def test_silu_branches():
    op = ki.Int()
    cfg = op.silu_cfg
    x = np.arange(-cfg.b_2 - 40, cfg.b_2 + 40, 37, dtype=np.int64)
    y = op.silu(x)
    hi = np.abs(x) >= cfg.b_2
    assert np.array_equal(y[hi], np.maximum(x[hi], 0))
    real = x[~hi] / S
    assert np.abs(y[~hi] / S - real / (1 + np.exp(-real))).max() < 2e-3


def test_topk_width_bounds_every_tiebroken_difference():
    rng = np.random.default_rng(4)
    E, s_sel = 384, kl.S_SEL
    b = rng.integers(-40_000, 90_000, E)
    w = ki.topk_width(s_sel, b, E)
    lo = tref.tiebroken((np.ones(E, dtype=np.int64) + b).tolist(), E)
    hi = tref.tiebroken((np.full(E, s_sel) + b).tolist(), E)
    assert max(hi) - min(lo) < (1 << w)


def test_out_of_window_values_are_recorded_not_raised():
    op = ki.Int()
    big = np.full((1, 4), 1 << 30, dtype=np.int64)
    op.rmsnorm(big, "hot")                                   # row energy 2^62 > 2^54
    op.matmul(np.full((1, 2), 1 << 24), np.full((2, 1), 1 << 15), "wide")
    v = op.ranges.violations()
    assert "hot.S_tot" in v and "wide.out" in v
    assert all(r["ok"] for n, r in op.ranges.as_dict().items() if n not in v)


@pytest.fixture(scope="module")
def toy_gguf(tmp_path_factory):
    path = tmp_path_factory.mktemp("k2ref") / "toy.gguf"
    toy.write(path)
    return str(path)


def test_routed_combine_equals_the_topk_reference(toy_gguf):
    W = kl.GgufWeights(toy_gguf, cfg=toy.TOY_MLA)
    rng = np.random.default_rng(5)
    T, kk = 6, toy.TOY["k"]
    x = np.rint(rng.normal(0, 1, (T, toy.TOY_MLA["d"])) * S).astype(np.int64)
    out = {}
    ki.moe(ki.Int(), x, W, 1, T=T, E=toy.TOY["E"], kk=kk, p="L1", out=out)
    D = out["L1.D"].tolist()
    want = tref.combine(out["L1.gw"].tolist(), D, kk, T)
    assert np.array_equal(np.asarray(out["L1.y_raw"], dtype=object), np.asarray(want, dtype=object))
    assert (np.asarray(out["L1.mask"]).sum(axis=1) == kk).all()


@pytest.mark.parametrize("offset", [0, 130_072])
def test_two_layers_against_the_float_reference(toy_gguf, offset):
    cfg, kk = toy.TOY_MLA, toy.TOY["k"]
    Wi = kl.GgufWeights(toy_gguf, cfg=cfg)
    Wf = kl.GgufWeights(toy_gguf, ints=False, cfg=cfg)
    ids = [3, 17, 0, 41, 9, 9, 22, 30]
    io, ranges = ki.forward(ids, Wi, cfg, layers=2, offset=offset, kk=kk, head_chunk=20)
    fo = kf.forward(ids, Wf, cfg, layers=2, offset=offset, k=kk, head_chunk=20)
    assert not ranges.violations(), ranges.violations()
    fid = kf.fidelity(io, fo, S, kl.S_SEL)
    for name in ("L0.n1g", "L0.r1", "L1.n2g", "L1.s", "L1.y", "final.ng", "logits"):
        assert fid[name]["rel_l2"] < 2e-2, (name, fid[name])
    assert fid["logits.top1_agree"] >= 7 / 8
    rd = kf.routing_differences(io, fo, Wf, 1, k=kk, S=S)
    assert rd["same_input"]["tokens_differing"] <= 1, rd
    assert rd["end_to_end"]["tokens_differing"] <= 2, rd
