"""YaRN RoPE frequency scaling (rope_scaling type "yarn": DeepSeek-V3, Kimi K2):
golden vectors, the step K2's config produces, backward compatibility, the
serialized config, and the f64 tables against the reference's float32
formula at short and long positions under stated tolerances (CPU).

The tables are public constants computed independently by the Python prover
(claims._rope_cos_sin) and the Rust verifier (handlers.rs rope_cos_sin). The
golden vectors here are asserted bit-for-bit by both suites (Rust:
rope_yarn_tests in handlers.rs). They were generated from the Python
implementation: K2's configuration at positions 1, 4,095 and 131,071, and a
smooth-ramp configuration (factor 40, beta_fast 32, mscale_all_dim 0) that
exercises a blended pair and a cos/sin factor other than 1.

The reference is DeepseekV3YarnRotaryEmbedding in K2's modeling_deepseek.py
at Hugging Face revision fd1984e2 (:226-327), which computes in float32.
The prover and verifier compute in f64, so the rounded tables differ from
the reference by float32's error in position x frequency: at most 1 unit
below position 4,096 and, measured, 35 units up to position 131,071 (19 of
the 64 entries at 131,071 move, by up to 22). The tolerances are 1 and 64
units at scale 4,096, the second about 2^-23 x 131,071 rad; a wrong ramp or
frequency misses by thousands."""
import math
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from claims import (P, RoPEConfig, _rope_cos_sin, _rope_yarn_inv_freq,   # noqa: E402
                    _rope_yarn_mscale)
from model_config import YarnScaling                                     # noqa: E402

K2 = dict(d_h=64, s_x=4096, base=50000.0, scale_factor=32.0, original_max_pos=4096,
          yarn=True, yarn_beta_fast=1.0, yarn_beta_slow=1.0, yarn_mscale=1.0,
          yarn_mscale_all_dim=1.0)
RAMP = dict(d_h=8, s_x=4096, base=10000.0, scale_factor=40.0, original_max_pos=4096,
            yarn=True, yarn_beta_fast=32.0, yarn_beta_slow=1.0, yarn_mscale=1.0,
            yarn_mscale_all_dim=0.0)

K2_1_COS = [
    2213, 3098, 3578, 3830, 3960, 4027, 4061, 4078, 4087, 4091, 4094, 4095,
    4095, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096,
    4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096,
]
K2_1_SIN = [
    3447, 2680, 1994, 1453, 1047, 751, 537, 384, 274, 195, 139, 99, 71, 51,
    36, 26, 18, 13, 9, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
]
K2_4095_COS = [
    18446744069414584051, 325, 18446744069414580644, 18446744069414582028,
    18446744069414580353, 1563, 18446744069414583232, 3008,
    18446744069414580788, 3582, 2110, 1395, 18446744069414583784, 3983,
    18446744069414583848, 3496, 3521, 3608, 18446744069414580251, 3836,
    4051, 4073, 4084, 4090, 4093, 4094, 4095, 4096, 4096, 4096, 4096, 4096,
]
K2_4095_SIN = [
    18446744069414580234, 18446744069414580238, 1804, 3394,
    18446744069414583304, 3786, 18446744069414580372, 2780,
    18446744069414582248, 1986, 3511, 18446744069414580470, 4061, 955,
    18446744069414580252, 2135, 18446744069414582229, 1939, 457, 1435, 604,
    431, 308, 220, 157, 112, 80, 57, 41, 29, 21, 15,
]
K2_131071_COS = [
    18446744069414580971, 3630, 1607, 1625, 18446744069414580453, 3536,
    18446744069414583596, 18446744069414584008, 4012, 1578,
    18446744069414580833, 3150, 142, 18446744069414584096,
    18446744069414580394, 1817, 18446744069414583267, 18446744069414580284,
    18446744069414580501, 1993, 101, 18446744069414580339,
    18446744069414581276, 18446744069414583721, 1388, 2630, 3327, 3698,
    3892, 3992, 4043, 4069,
]
K2_131071_SIN = [
    18446744069414581965, 18446744069414582424, 3768, 18446744069414580561,
    18446744069414582973, 18446744069414582254, 4031, 4084, 824,
    18446744069414580541, 2148, 18446744069414581703, 18446744069414580227,
    4090, 1166, 18446744069414580650, 3958, 18446744069414583626, 1478,
    18446744069414580743, 18446744069414580226, 18446744069414583362, 2740,
    4052, 3854, 3140, 2390, 1760, 1276, 917, 657, 469,
]
RAMP_COS = [
    5607, 5607, 5607, 5607, 3029, 5579, 5607, 5607, 18446744069414581988,
    5495, 5607, 5607,
]
RAMP_SIN = [
    0, 0, 0, 0, 4718, 560, 29, 0, 5098, 1114, 57, 0,
]


def test_golden_vectors():
    for off, cos, sin in ((1, K2_1_COS, K2_1_SIN), (4095, K2_4095_COS, K2_4095_SIN),
                          (131071, K2_131071_COS, K2_131071_SIN)):
        c, s = _rope_cos_sin(RoPEConfig(SEQ=1, position_offset=off, **K2))
        assert (c, s) == (cos, sin), f"K2 at position {off} drifted"
    assert _rope_cos_sin(RoPEConfig(SEQ=3, **RAMP)) == (RAMP_COS, RAMP_SIN)


def test_k2_is_a_step():
    """K2's beta_fast = beta_slow = 1 put both correction bounds at 19 and 20:
    pairs 0-19 keep the base frequency, 20-31 divide it by 32, and the cos/sin
    factor is exactly 1."""
    cfg = RoPEConfig(SEQ=1, **K2)
    inv = _rope_yarn_inv_freq(cfg)
    base = [1.0 / (50000.0 ** (2 * k / 64)) for k in range(32)]
    assert all(inv[k] == base[k] for k in range(20))
    assert all(math.isclose(inv[k], base[k] / 32, rel_tol=1e-15) for k in range(20, 32))
    assert _rope_yarn_mscale(cfg) == 1.0


def test_k2s_step_is_the_llama3_ramp_with_unit_factors():
    """An independent check of K2's step: with beta_fast = beta_slow = 1, YaRN
    interpolates exactly the pairs whose wavelength exceeds the original
    context, which is the Llama-3 ramp with both frequency factors 1 (pair
    19's wavelength is about 3,890, pair 20's about 5,422, against 4,096).
    Two separately written formulas give the same tables."""
    llama3 = {k: v for k, v in K2.items() if not k.startswith("yarn")}
    for off in (0, 4093, 131069):
        assert _rope_cos_sin(RoPEConfig(SEQ=3, position_offset=off, **K2)) == \
            _rope_cos_sin(RoPEConfig(SEQ=3, position_offset=off, low_freq_factor=1.0,
                                     high_freq_factor=1.0, **llama3))


def test_the_ramp_configuration_blends_and_scales():
    cfg = RoPEConfig(SEQ=1, **RAMP)
    inv = _rope_yarn_inv_freq(cfg)
    ratios = [inv[k] * 10000.0 ** (2 * k / 8) for k in range(4)]
    assert [round(r, 12) for r in ratios] == [1.0, 1.0, 0.5125, 0.025]
    assert math.isclose(_rope_yarn_mscale(cfg), 0.1 * math.log(40) + 1.0)
    assert RAMP_COS[0] == round(4096 * (0.1 * math.log(40) + 1.0))


def test_yarn_off_leaves_tables_unchanged():
    """yarn=False ignores the YaRN fields: existing tables stay byte-identical."""
    plain = dict(SEQ=4, d_h=8, s_x=4096, base=500000.0, position_offset=2)
    assert _rope_cos_sin(RoPEConfig(**plain)) == _rope_cos_sin(RoPEConfig(
        **plain, yarn_beta_fast=7.0, yarn_beta_slow=3.0, yarn_mscale=2.0,
        yarn_mscale_all_dim=5.0))


def test_the_serialized_config_carries_yarn():
    """What the Rust verifier reads (rope_yarn_tests reproduces this dict)."""
    from protocol import _ser_value
    d = _ser_value(RoPEConfig(SEQ=1, position_offset=131071, **K2))["config"]
    assert d["yarn"] == 1 and d["yarn_beta_fast"] == 1.0 and d["yarn_beta_slow"] == 1.0
    assert d["yarn_mscale"] == 1.0 and d["yarn_mscale_all_dim"] == 1.0
    assert d["scale_factor"] == 32.0 and d["original_max_pos"] == 4096


def test_yarn_scaling_from_k2_config():
    y = YarnScaling.from_config({"beta_fast": 1.0, "beta_slow": 1.0, "factor": 32.0,
                                 "mscale": 1.0, "mscale_all_dim": 1.0,
                                 "original_max_position_embeddings": 4096,
                                 "type": "yarn"})
    assert (y.factor, y.original_max_position_embeddings, y.beta_fast, y.beta_slow,
            y.mscale, y.mscale_all_dim) == (32.0, 4096, 1.0, 1.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        YarnScaling.from_config({"type": "llama3", "factor": 8.0,
                                 "original_max_position_embeddings": 8192})


def _reference(positions, *, d_h, base, scale_factor, original_max_pos, yarn_beta_fast,
               yarn_beta_slow, yarn_mscale, yarn_mscale_all_dim, **_):
    """modeling_deepseek.py @ fd1984e2 :226-327, verbatim in float32."""
    import torch

    def corr_dim(n):
        return (d_h * math.log(original_max_pos / (n * 2 * math.pi))) / (2 * math.log(base))

    def get_mscale(scale, mscale):
        return 1.0 if scale <= 1 else 0.1 * mscale * math.log(scale) + 1.0

    low = max(math.floor(corr_dim(yarn_beta_fast)), 0)
    high = min(math.ceil(corr_dim(yarn_beta_slow)), d_h - 1)
    if low == high:
        high += 0.001
    ramp = torch.clamp((torch.arange(d_h // 2, dtype=torch.float32) - low) / (high - low), 0, 1)
    mask = 1.0 - ramp
    fe = 1.0 / (base ** (torch.arange(0, d_h, 2, dtype=torch.float32) / d_h))
    fi = 1.0 / (scale_factor * base ** (torch.arange(0, d_h, 2, dtype=torch.float32) / d_h))
    inv = fi * (1 - mask) + fe * mask
    f = torch.outer(torch.tensor(positions, dtype=torch.float32), inv)
    m = float(get_mscale(scale_factor, yarn_mscale) / get_mscale(scale_factor, yarn_mscale_all_dim))
    return f.cos() * m, f.sin() * m


def _max_diff(lo, n):
    import torch
    c, s = _rope_cos_sin(RoPEConfig(SEQ=n, position_offset=lo, **K2))
    rc, rs = _reference(list(range(lo, lo + n)), **K2)
    ref = torch.round(torch.cat([rc.reshape(-1), rs.reshape(-1)]).double() * 4096).long().tolist()
    ours = [v - P if v > P // 2 else v for v in c + s]
    return max(abs(a - b) for a, b in zip(ours, ref))


def test_against_the_float32_reference():
    """Below position 4,096 within 1 unit; at 65,536 and up to 131,071 within
    64 (positions 65,536-65,599 and 131,008-131,071 in full)."""
    for lo, n, tol in ((0, 4096, 1), (65536, 64, 64), (131008, 64, 64)):
        worst = _max_diff(lo, n)
        assert worst <= tol, (lo, worst, tol)


def test_against_the_float32_reference_across_the_context():
    """Every 1,021st position up to 131,071, under the long tolerance."""
    worst = max(_max_diff(p, 1) for p in range(0, 131072, 1021))
    assert worst <= 64, worst
