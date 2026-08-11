"""HF loader on synthetic checkpoints: ModelConfig parsing, GQA KV-column
replication, eager/lazy spec agreement, and tied-LM-head detection.

Writes tiny fake checkpoints (config.json + safetensors, bf16, HF Llama
tensor names) to a temp dir — no real weights needed:
  - "gqa":  single-shard, tied embeddings, n_kv_heads < n_heads, llama3
            rope_scaling — the Llama-3.2-1B shape class.
  - "mha":  two-shard with index.json, separate lm_head, n_kv = n_heads —
            the Llama-2-7B shape class; guards that the GQA path is a
            byte-exact no-op for existing models.

Needs CUDA (quantize_to_field). Run: python tests/run_tests.py test_model_loader
"""
import json
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

# Tiny dims. n_heads·d_h == d in both, like every real Llama so far.
D, DFF, VOCAB, HEADS, DH = 32, 48, 40, 8, 4
GQA_KV, MHA_KV = 2, 8
S = 2 ** 12


def _layer_tensors(gen, kv_heads):
    def r(*shape):
        return torch.randn(*shape, generator=gen, dtype=torch.float32).to(torch.bfloat16)
    p = "model.layers.0"
    return {
        f"{p}.self_attn.q_proj.weight": r(HEADS * DH, D),
        f"{p}.self_attn.k_proj.weight": r(kv_heads * DH, D),
        f"{p}.self_attn.v_proj.weight": r(kv_heads * DH, D),
        f"{p}.self_attn.o_proj.weight": r(D, HEADS * DH),
        f"{p}.mlp.gate_proj.weight":    r(DFF, D),
        f"{p}.mlp.up_proj.weight":      r(DFF, D),
        f"{p}.mlp.down_proj.weight":    r(D, DFF),
        f"{p}.input_layernorm.weight":          r(D),
        f"{p}.post_attention_layernorm.weight": r(D),
        "model.norm.weight":         r(D),
        "model.embed_tokens.weight": r(VOCAB, D),
    }


def _write_ckpt(td, *, kv_heads, tied, sharded, rope_scaling=None, seed=0):
    from safetensors.torch import save_file
    d = pathlib.Path(td)
    cfg = {
        "hidden_size": D, "intermediate_size": DFF, "num_hidden_layers": 1,
        "num_attention_heads": HEADS, "num_key_value_heads": kv_heads,
        "head_dim": DH, "vocab_size": VOCAB, "rms_norm_eps": 1e-5,
        "rope_theta": 500000.0, "tie_word_embeddings": tied,
    }
    if rope_scaling is not None:
        cfg["rope_scaling"] = rope_scaling
    (d / "config.json").write_text(json.dumps(cfg))

    gen = torch.Generator().manual_seed(seed)
    tensors = _layer_tensors(gen, kv_heads)
    if not tied:
        tensors["lm_head.weight"] = torch.randn(
            VOCAB, D, generator=gen, dtype=torch.float32).to(torch.bfloat16)
    if sharded:
        names = sorted(tensors)
        half = len(names) // 2
        shards = {"model-00001-of-00002.safetensors": names[:half],
                  "model-00002-of-00002.safetensors": names[half:]}
        weight_map = {}
        for fname, keys in shards.items():
            save_file({k: tensors[k].contiguous() for k in keys}, str(d / fname))
            weight_map.update({k: fname for k in keys})
        (d / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": weight_map}))
    else:
        save_file({k: v.contiguous() for k, v in tensors.items()},
                  str(d / "model.safetensors"))
    return tensors


LLAMA3_SCALING = {"rope_type": "llama3", "factor": 32.0, "low_freq_factor": 1.0,
                  "high_freq_factor": 4.0,
                  "original_max_position_embeddings": 8192}


def _gqa_ckpt(td):
    return _write_ckpt(td, kv_heads=GQA_KV, tied=True, sharded=False,
                       rope_scaling=LLAMA3_SCALING, seed=1)


def _mha_ckpt(td):
    return _write_ckpt(td, kv_heads=MHA_KV, tied=False, sharded=True, seed=2)


def _i64(t):
    return t.view(torch.int64)


def test_model_config_parses():
    from model_config import ModelConfig
    with tempfile.TemporaryDirectory() as td:
        _gqa_ckpt(td)
        c = ModelConfig.from_hf(td)
        assert (c.d, c.d_ff, c.vocab) == (D, DFF, VOCAB)
        assert (c.n_heads, c.n_kv_heads, c.d_h) == (HEADS, GQA_KV, DH)
        assert c.kv_groups == HEADS // GQA_KV and c.q_width == HEADS * DH
        assert c.tied_embeddings and c.rope_theta == 500000.0
        assert c.rope_scaling is not None and c.rope_scaling.factor == 32.0
        assert c.rope_scaling.original_max_position_embeddings == 8192
    with tempfile.TemporaryDirectory() as td:
        _mha_ckpt(td)
        c = ModelConfig.from_hf(td)
        assert c.kv_groups == 1 and not c.tied_embeddings
        assert c.rope_scaling is None


def test_model_config_rejects_unknown_rope_scaling():
    from model_config import ModelConfig
    with tempfile.TemporaryDirectory() as td:
        _write_ckpt(td, kv_heads=MHA_KV, tied=False, sharded=False,
                    rope_scaling={"type": "linear", "factor": 2.0})
        try:
            ModelConfig.from_hf(td)
        except ValueError as e:
            assert "linear" in str(e)
        else:
            raise AssertionError("unknown rope_scaling type must raise")


def test_replicate_kv_cols_pattern():
    from loader import replicate_kv_cols, quantize_to_field
    base = quantize_to_field(     # includes negatives → field-rep P−|v| path
        torch.randn(D, GQA_KV * DH, generator=torch.Generator().manual_seed(3)), S)
    groups = HEADS // GQA_KV
    rep = replicate_kv_cols(base, d_h=DH, groups=groups)
    assert rep.shape == (D, HEADS * DH)
    for hq in range(HEADS):
        src = _i64(base)[:, (hq // groups) * DH:(hq // groups + 1) * DH]
        dst = _i64(rep)[:, hq * DH:(hq + 1) * DH]
        assert torch.equal(src, dst), f"head {hq}: replicated block mismatch"
    assert replicate_kv_cols(base, d_h=DH, groups=1) is base   # identity


def test_lazy_gqa_replication_matches_reference():
    from loader import LazyHFLoader, quantize_to_field
    with tempfile.TemporaryDirectory() as td:
        tensors = _gqa_ckpt(td)
        ldr = LazyHFLoader(td, S=S)
        spec = ldr.layer_specs(0)["W_K"]
        assert spec.kv_groups == HEADS // GQA_KV
        got = ldr.make_loader(spec.hf_name, transpose=spec.transpose,
                              divide_by=spec.divide_by,
                              kv_groups=spec.kv_groups)().view(D, HEADS * DH)
        base = quantize_to_field(
            tensors["model.layers.0.self_attn.k_proj.weight"].T.contiguous(),
            S, divide_by=spec.divide_by)
        for hq in range(HEADS):
            kv = hq // spec.kv_groups
            assert torch.equal(_i64(got)[:, hq * DH:(hq + 1) * DH],
                               _i64(base)[:, kv * DH:(kv + 1) * DH]), f"head {hq}"


def test_eager_matches_lazy():
    from loader import LazyHFLoader, load_layer_weights
    with tempfile.TemporaryDirectory() as td:
        _gqa_ckpt(td)
        eager = load_layer_weights(td, 0, S=S)
        ldr = LazyHFLoader(td, S=S)
        shapes = ldr.layer_shapes()
        assert set(eager) == set(shapes)
        for short, spec in ldr.layer_specs(0).items():
            lazy = ldr.make_loader(spec.hf_name, transpose=spec.transpose,
                                   divide_by=spec.divide_by,
                                   kv_groups=spec.kv_groups)()
            assert eager[short].shape == shapes[short], short
            assert torch.equal(_i64(eager[short].reshape(-1)), _i64(lazy)), short


def test_mha_replication_is_identity():
    """Llama-2-7B-class regression: on an MHA checkpoint the new GQA path
    must be byte-identical to plain transpose+quantize."""
    from loader import LazyHFLoader, quantize_to_field
    with tempfile.TemporaryDirectory() as td:
        tensors = _mha_ckpt(td)
        ldr = LazyHFLoader(td, S=S)
        for short in ("W_K", "W_V"):
            spec = ldr.layer_specs(0)[short]
            assert spec.kv_groups == 1
            got = ldr.make_loader(spec.hf_name, transpose=spec.transpose,
                                  divide_by=spec.divide_by,
                                  kv_groups=spec.kv_groups)()
            hf_name = spec.hf_name
            ref = quantize_to_field(tensors[hf_name].T.contiguous(), S,
                                    divide_by=spec.divide_by).reshape(-1)
            assert torch.equal(_i64(got), _i64(ref)), short
        assert ldr.layer_shapes()["W_K"] == (D, D)


def test_tied_lm_head_fallback():
    from loader import LazyHFLoader, quantize_to_field
    with tempfile.TemporaryDirectory() as td:      # tied, single-shard
        tensors = _gqa_ckpt(td)
        ldr = LazyHFLoader(td, S=S)
        assert not ldr.has_separate_lm_head()
        fw = ldr.load_final_weights()              # crashed before the fix
        ref = quantize_to_field(
            tensors["model.embed_tokens.weight"].T.contiguous(), S)
        assert torch.equal(_i64(fw["W_lm_head"]), _i64(ref))
    with tempfile.TemporaryDirectory() as td:      # untied, sharded
        tensors = _mha_ckpt(td)
        ldr = LazyHFLoader(td, S=S)
        assert ldr.has_separate_lm_head()
        fw = ldr.load_final_weights()
        ref = quantize_to_field(tensors["lm_head.weight"].T.contiguous(), S)
        assert torch.equal(_i64(fw["W_lm_head"]), _i64(ref))


def test_layer_shapes_cross_check():
    from loader import LazyHFLoader
    with tempfile.TemporaryDirectory() as td:
        _gqa_ckpt(td)
        ldr = LazyHFLoader(td, S=S)
        assert ldr.layer_shapes(D, DFF)["W_K"] == (D, HEADS * DH)
        try:
            ldr.layer_shapes(D + 1, DFF)
        except AssertionError:
            pass
        else:
            raise AssertionError("dim mismatch with config.json must raise")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"=== model_loader: {len(fns)}/{len(fns)} PASS ===")
