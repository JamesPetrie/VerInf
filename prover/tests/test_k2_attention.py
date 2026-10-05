"""Kimi K2's latent attention outside the prover (CPU): the GGUF weight
transforms and composition A against the reference (prover/k2_attention.py;
analysis/mla-attention-design.md §4.1, §6).

- The transforms give the same committed weights from llama.cpp's split
  k_b/v_b tensors and from the legacy single kv_b.
- Composition A — split matmuls, half-split RoPE at the prover's YaRN
  frequencies, the per-head assembly with the shared rotary key, σ folded
  into the query — computes the same layer as reference_attention, the
  float64 port of DeepseekV3Attention.forward, at K2's head dimensions and
  at short and long positions.
- The rotary de-interleave is the reference's reshape and the Maverick
  loader's _unpermute_rows map.

reference_attention itself was checked once against the real module
(modeling_deepseek.py at Hugging Face fd1984e2, transformers 4.48.3, K2's
config at hidden size 96 and 3 heads, float64): 9.3e-8 relative at
positions 0-6, the reference's float32 rotary frequencies, and rotary tables
within 22.7 units at scale 4,096 up to position 131,071, the float32 effect
test_rope_yarn.py bounds."""
import math
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import k2_attention as ka   # noqa: E402

SMALL = dict(ka.K2_MLA, d=96, H=3, q_rank=40, kv_rank=24)   # K2's head dims, small model


def _hf(cfg, seed=0):
    rng = np.random.default_rng(seed)
    H, dn, dr, dv = cfg["H"], cfg["d_nope"], cfg["d_rope"], cfg["d_v"]
    r = lambda *s: rng.normal(0, 0.05, s)
    return {"q_a_proj": r(cfg["q_rank"], cfg["d"]), "q_a_layernorm": 1 + r(cfg["q_rank"]),
            "q_b_proj": r(H * (dn + dr), cfg["q_rank"]),
            "kv_a_proj_with_mqa": r(cfg["kv_rank"] + dr, cfg["d"]),
            "kv_a_layernorm": 1 + r(cfg["kv_rank"]),
            "kv_b_proj": r(H * (dn + dv), cfg["kv_rank"]), "o_proj": r(cfg["d"], H * dv),
            "input_layernorm": 1 + r(cfg["d"])}


def test_split_and_legacy_gguf_give_the_same_weights():
    hf = _hf(SMALL)
    g = ka.gguf_from_hf(hf, SMALL)
    split = ka.mla_weights_from_gguf({k: v for k, v in g.items() if k != "attn_kv_b"}, SMALL)
    legacy = ka.mla_weights_from_gguf({k: v for k, v in g.items()
                                       if k not in ("attn_k_b", "attn_v_b")}, SMALL)
    assert split.keys() == legacy.keys()
    for k in split:
        assert np.array_equal(split[k], legacy[k]), k


def test_the_gguf_shapes_are_llama_cpps():
    """gguf-py reverses ggml's dims: attn_k_b {128, 512, 64} reads as
    (64, 512, 128), attn_v_b {512, 128, 64} as (64, 128, 512)."""
    g = ka.gguf_from_hf(_hf(SMALL), SMALL)
    assert g["attn_k_b"].shape == (3, 24, 128) and g["attn_v_b"].shape == (3, 128, 24)


@pytest.mark.parametrize("start", [0, 131060])
def test_composition_a_computes_the_reference_layer(start):
    hf = _hf(SMALL, seed=1)
    w = ka.mla_weights_from_gguf(ka.gguf_from_hf(hf, SMALL), SMALL)
    x = np.random.default_rng(2).normal(0, 1, (7, SMALL["d"]))
    pos = np.arange(start, start + 7)
    ref = ka.reference_attention(x, hf, SMALL, positions=pos)
    got = ka.composition_a_attention(x, w, SMALL, positions=pos)
    assert np.allclose(got, ref, rtol=1e-10, atol=1e-12), np.abs(got - ref).max()


def test_a_wrong_rotary_layout_is_caught():
    """Without the de-interleave the layer differs: the check above has teeth."""
    hf = _hf(SMALL, seed=3)
    w = ka.mla_weights_from_gguf(ka.gguf_from_hf(hf, SMALL), SMALL)
    H, dn, dr, qr = SMALL["H"], SMALL["d_nope"], SMALL["d_rope"], SMALL["q_rank"]
    w["W_q_pe"] = hf["q_b_proj"].reshape(H, dn + dr, qr)[:, dn:, :].reshape(H * dr, qr).T
    x = np.random.default_rng(4).normal(0, 1, (5, SMALL["d"]))
    assert not np.allclose(ka.composition_a_attention(x, w, SMALL),
                           ka.reference_attention(x, hf, SMALL), rtol=1e-6)


def test_the_deinterleave_is_the_references_reshape_and_maveriks_map():
    dr, H = 64, 3
    w = np.arange(H * dr * 2).reshape(H * dr, 2)
    got = ka.deinterleave_rows(w, dr)
    # the reference: view(d/2, 2).transpose(-1, -2).reshape(d) along each head's rows
    ref = w.reshape(H, dr // 2, 2, 2).transpose(0, 2, 1, 3).reshape(H * dr, 2)
    assert np.array_equal(got, ref)
    # demo_maverick_block._unpermute_rows with n_head = H: index_select by argsort(idx)
    idx = np.arange(H * dr).reshape(H, 2, dr // 2).transpose(0, 2, 1).reshape(-1)
    assert np.array_equal(got, w[np.argsort(idx)])


def test_sigma_is_k2s_softmax_scale_and_folds_into_the_query_only():
    m = 0.1 * math.log(32) + 1.0
    assert math.isclose(ka.softmax_scale(ka.K2_MLA), 192 ** -0.5 * m * m)
    assert round(ka.softmax_scale(ka.K2_MLA), 6) == 0.130861
    assert ka.SIGMA_FOLDED == ("W_q_nope", "W_q_pe")
