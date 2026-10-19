"""Kimi K2's GGUF weights as committed (prover/k2_loader.py), on the CPU,
against a toy GGUF with K2's names, layout and types (tests/_k2_toy.py):

- every committed tensor is the source tensor in the (in, out) orientation,
  rounded half to even at its scale: the router and the shared and dense
  FFNs transposed, the bias at S_sel, the expert slice e of each stacked
  tensor (and not its neighbour), the embedding rows and head columns;
- the integer and float providers are one mapping: ints = rint(floats·S),
  σ folded into the two query weights only;
- the attention loader's provenance shares split each source tensor whole;
- the sigmoid table is monotone, floored at 1, and centred."""
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import _k2_toy as toy                                     # noqa: E402
import k2_loader as kl                                    # noqa: E402
from k2_attention import SIGMA_FOLDED, mla_weights_from_gguf, softmax_scale   # noqa: E402

S = kl.S_ACT


@pytest.fixture(scope="module")
def gguf(tmp_path_factory):
    path = tmp_path_factory.mktemp("k2") / "toy.gguf"
    return str(path), toy.write(path)


def _rint(a, scale):
    return np.rint(np.asarray(a, dtype=np.float64) * scale).astype(np.int64)


def test_dense_router_bias_and_shared_orientation(gguf):
    path, back = gguf
    W = kl.GgufWeights(path, cfg=toy.TOY_MLA)
    assert W.kind(0) == "dense" and W.kind(1) == "moe" and W.n_experts(1) == toy.TOY["E"]
    for k, n in (("W_gate", "gate"), ("W_up", "up"), ("W_down", "down")):
        assert np.array_equal(W.dense(0)[k], _rint(back[f"blk.0.ffn_{n}.weight"].T, S))
        assert np.array_equal(W.shared(1)[k], _rint(back[f"blk.1.ffn_{n}_shexp.weight"].T, S))
    assert W.router(1).shape == (toy.TOY_MLA["d"], toy.TOY["E"])
    assert np.array_equal(W.router(1), _rint(back["blk.1.ffn_gate_inp.weight"].T, S))
    assert np.array_equal(W.bias(1), _rint(back["blk.1.exp_probs_b.bias"], kl.S_SEL))


@pytest.mark.parametrize("kind", ["gate", "up", "down"])
def test_expert_slice_is_its_own(gguf, kind):
    path, back = gguf
    W = kl.GgufWeights(path, cfg=toy.TOY_MLA)
    stacked = back[f"blk.1.ffn_{kind}_exps.weight"]
    for e in range(toy.TOY["E"]):
        got = W.expert(1, kind, e)
        assert np.array_equal(got, _rint(stacked[e].T, S)), (kind, e)
        assert not np.array_equal(got, _rint(stacked[(e + 1) % toy.TOY["E"]].T, S))


def test_embedding_rows_and_head_columns(gguf):
    path, back = gguf
    W = kl.GgufWeights(path, cfg=toy.TOY_MLA)
    ids = [5, 0, 47, 5]
    assert np.array_equal(W.embed_rows(ids), _rint(back["token_embd.weight"][ids], S))
    assert W.V == toy.TOY["V"] and W.head_name == "output.weight"
    full = _rint(back["output.weight"].T, S)
    assert np.array_equal(np.concatenate([W.head(0, 20), W.head(20, 48)], axis=1), full)
    assert np.array_equal(W.g_out(), _rint(back["output_norm.weight"], S))


@pytest.mark.parametrize("ints", [True, False])
def test_a_gguf_without_the_head_is_refused(tmp_path, ints):
    """K2 is untied: no fallback to the embedding, in either provider."""
    path = tmp_path / "headless.gguf"
    toy.write(path, drop=("output.weight",))
    with pytest.raises(ValueError, match="output.weight is missing"):
        kl.GgufWeights(str(path), ints=ints, cfg=toy.TOY_MLA)


def test_attention_ints_are_the_transforms_rounded_with_sigma_folded(gguf):
    path, back = gguf
    cfg = toy.TOY_MLA
    Wi = kl.GgufWeights(path, cfg=cfg)
    Wf = kl.GgufWeights(path, ints=False, cfg=cfg)
    for il in range(2):
        t = {k: back[f"blk.{il}.{k}.weight"] for k in kl.ATTN_TENSORS
             if f"blk.{il}.{k}.weight" in back}
        want = mla_weights_from_gguf(t, cfg)
        want["g_ffn"] = back[f"blk.{il}.ffn_norm.weight"]
        ai, af = Wi.attn(il), Wf.attn(il)
        assert set(ai) == set(want) == set(af)
        sigma = softmax_scale(cfg)
        for k, v in want.items():
            mult = sigma if k in SIGMA_FOLDED else 1.0
            assert np.array_equal(ai[k], np.rint(np.asarray(v, np.float64) * mult * S)), k
            assert np.array_equal(af[k], np.asarray(v, np.float64)), k


def test_int_and_float_providers_are_one_mapping(gguf):
    path, _ = gguf
    Wi = kl.GgufWeights(path, cfg=toy.TOY_MLA)
    Wf = kl.GgufWeights(path, ints=False, cfg=toy.TOY_MLA)
    pairs = [(Wi.router(1), Wf.router(1), S), (Wi.bias(1), Wf.bias(1), kl.S_SEL),
             (Wi.expert(1, "down", 3), Wf.expert(1, "down", 3), S),
             (Wi.embed_rows([1, 2]), Wf.embed_rows([1, 2]), S), (Wi.head(0, 7), Wf.head(0, 7), S)]
    for a, f, scale in pairs:
        assert np.array_equal(a, _rint(f, scale))


def test_attention_provenance_shares_cover_each_source(gguf, monkeypatch):
    path, _ = gguf
    import loader
    monkeypatch.setattr(loader, "gguf_provenance",
                        lambda p, name, **kw: {"quant": "x", "packed_bytes": 1_000_000, "name": name})
    shares = {}
    for key in kl._attn_sources(toy.TOY_MLA):
        prov = kl.attention_loader(path, 1, key, cfg=toy.TOY_MLA).provenance
        assert prov["transform"] == key
        shares[prov["name"]] = shares.get(prov["name"], 0) + prov["packed_bytes"]
    assert set(shares.values()) == {1_000_000}
    assert len(shares) == 10            # the nine split-form attention tensors + ffn_norm


def test_sigmoid_table():
    keys, ty, shift = kl.sigmoid_table(S, kl.S_SEL, bits=19)
    assert keys[0] == 0 and len(keys) == len(ty) == 1 << 19 and shift == 1 << 18
    y = np.asarray(ty)
    assert (np.diff(y) >= 0).all() and y.min() == 1 and y.max() == kl.S_SEL
    assert y[shift] == kl.S_SEL // 2
    assert y[shift + 4 * S] == round(kl.S_SEL / (1 + np.exp(-4.0)))


def test_constants():
    k = kl.K2_INT
    assert k["eps_int"] == 17 and k["C"] == 11579 and k["S"] == 1 << 12


def test_int_constants_are_k2s_for_k2_and_the_models_own_otherwise():
    from k2_attention import K2_MLA, MOONLIGHT_MLA
    assert kl.int_constants(K2_MLA) == kl.K2_INT
    m = kl.int_constants(MOONLIGHT_MLA)
    assert m["C"] == round(2.446 * 4096) and m["eps"] == 1e-5 and m["eps_int"] == round(1e-5 * 2 ** 24)
    assert {k: v for k, v in m.items() if k not in ("C", "eps", "eps_int")} == \
        {k: v for k, v in kl.K2_INT.items() if k not in ("C", "eps", "eps_int")}


def test_without_query_lora_the_query_weights_read_the_input(tmp_path):
    path = tmp_path / "moon.gguf"
    back = toy.write(path, mla=toy.TOY_MOON_MLA)
    cfg = toy.TOY_MOON_MLA
    W = kl.GgufWeights(str(path), ints=False, cfg=cfg)
    w = W.attn(0)
    assert "W_qa" not in w and "g_q_a" not in w
    H, dn, dr, d = cfg["H"], cfg["d_nope"], cfg["d_rope"], cfg["d"]
    q = back["blk.0.attn_q.weight"].reshape(H, dn + dr, d)
    assert np.array_equal(w["W_q_nope"], q[:, :dn, :].reshape(H * dn, d).T.astype(np.float64))
    assert w["W_q_pe"].shape == (d, H * dr)
