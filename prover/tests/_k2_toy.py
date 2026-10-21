"""A toy GGUF with Kimi K2's tensor names, layout and mixed types, for the
loader and reference tests (CPU) and the driver suite (GPU).

Two layers as K2's first two: layer 0 latent attention and a dense FFN,
layer 1 latent attention and a top-k MoE (router, selection bias, stacked
experts, shared expert); then the final norm and an untied head. `layers`
> 2 appends further MoE layers like layer 1, drawn after the two-layer
toy's tensors, so those are unchanged. K2's
attention head dimensions are scaled down to multiples of 32 so the Q8_0
tensors (attn_k_b, attn_v_b, attn_q_b and the experts, as in UD-Q4_K_XL's
Q8_0 set) quantize; norms, router and bias are F32 as in the real file.
K-quants are exercised on the real GGUF only (gguf-py cannot write them)."""
import numpy as np

import k2_attention as ka

TOY_MLA = dict(ka.K2_MLA, d=64, H=2, q_rank=32, kv_rank=32, d_nope=32, d_rope=32, d_v=32)
TOY = dict(V=48, d_ff_dense=64, E=8, k=3, d_ff_expert=32, d_ff_shared=32)
# Moonlight's variant of the family (k2_attention.MOONLIGHT_MLA): no query-LoRA,
# plain RoPE, its routed scale and rms_norm_eps
TOY_MOON_MLA = dict(TOY_MLA, q_rank=None, yarn=False, routed_scale=2.446, rms_eps=1e-5)


def tensors(seed=0, mla=TOY_MLA, toy=TOY, layers=2):
    """{name: float32 array in gguf-py's shape} for the toy at `layers`
    (at least 2)."""
    assert layers >= 2
    rng = np.random.default_rng(seed)
    d, H, dn, dr, dv = mla["d"], mla["H"], mla["d_nope"], mla["d_rope"], mla["d_v"]
    qr, kr = mla["q_rank"], mla["kv_rank"]
    V, E = toy["V"], toy["E"]
    n = lambda std, *s: rng.normal(0.0, std, s).astype(np.float32)
    gain = lambda m: (1.0 + rng.normal(0.0, 0.1, m)).astype(np.float32)
    t = {"token_embd.weight": n(0.5, V, d), "output_norm.weight": gain(d),
         "output.weight": n(d ** -0.5, V, d)}
    def attention(il):
        p = f"blk.{il}."
        t[p + "attn_norm.weight"] = gain(d)           # first, as before: K2's draws unchanged
        if qr is None:
            t[p + "attn_q.weight"] = n(d ** -0.5, H * (dn + dr), d)
        else:
            t.update({p + "attn_q_a.weight": n(d ** -0.5, qr, d),
                      p + "attn_q_a_norm.weight": gain(qr),
                      p + "attn_q_b.weight": n(qr ** -0.5, H * (dn + dr), qr)})
        t.update({
            p + "attn_kv_a_mqa.weight": n(d ** -0.5, kr + dr, d),
            p + "attn_kv_a_norm.weight": gain(kr),
            p + "attn_k_b.weight": n(kr ** -0.5, H, kr, dn),
            p + "attn_v_b.weight": n(kr ** -0.5, H, dv, kr),
            p + "attn_output.weight": n((H * dv) ** -0.5, d, H * dv),
            p + "ffn_norm.weight": gain(d),
        })

    def moe(il):
        p = f"blk.{il}."
        t.update({p + "ffn_gate_inp.weight": n(d ** -0.5, E, d),
                  p + "exp_probs_b.bias": rng.uniform(-0.3, 0.3, E).astype(np.float32),
                  p + "ffn_gate_exps.weight": n(d ** -0.5, E, fe, d),
                  p + "ffn_up_exps.weight": n(d ** -0.5, E, fe, d),
                  p + "ffn_down_exps.weight": n(fe ** -0.5, E, d, fe),
                  p + "ffn_gate_shexp.weight": n(d ** -0.5, fs, d),
                  p + "ffn_up_shexp.weight": n(d ** -0.5, fs, d),
                  p + "ffn_down_shexp.weight": n(fs ** -0.5, d, fs)})
    for il in range(2):
        attention(il)
    f = toy["d_ff_dense"]
    t.update({"blk.0.ffn_gate.weight": n(d ** -0.5, f, d),
              "blk.0.ffn_up.weight": n(d ** -0.5, f, d),
              "blk.0.ffn_down.weight": n(f ** -0.5, d, f)})
    fe, fs = toy["d_ff_expert"], toy["d_ff_shared"]
    moe(1)
    for il in range(2, layers):
        attention(il)
        moe(il)
    return t


Q8_0_NAMES = ("attn_q_b", "attn_q.", "attn_k_b", "attn_v_b", "_exps", "ffn_down.", "output.weight")


def llama_metadata(w, mla=TOY_MLA, toy=TOY, layers=2):
    """The deepseek2 keys llama.cpp reads, as K2's UD-Q4_K_XL shard 1 states
    them (read 2026-10-19), at the toy's sizes; no tokenizer, `vocab_size`
    dummy tokens instead. llama.cpp's loader takes the YaRN betas from the
    context, not from these keys (analysis/k2-fidelity-study-design.md §3.3)."""
    import gguf
    d, H, dn, dr, dv, kr = (mla[k] for k in ("d", "H", "d_nope", "d_rope", "d_v", "kv_rank"))
    w.add_block_count(layers)
    w.add_context_length(131072)
    w.add_embedding_length(d)
    w.add_feed_forward_length(toy["d_ff_dense"])
    w.add_head_count(H)
    w.add_head_count_kv(1)
    w.add_rope_freq_base(mla["base"])
    w.add_layer_norm_rms_eps(mla["rms_eps"])
    w.add_expert_used_count(toy["k"])
    w.add_leading_dense_block_count(1)
    w.add_vocab_size(toy["V"])
    w.add_q_lora_rank(mla["q_rank"] or 0)        # 0: no query-LoRA, two layers are not "lite"
    w.add_kv_lora_rank(kr)
    w.add_key_length(kr + dr)
    w.add_value_length(kr)
    w.add_key_length_mla(dn + dr)
    w.add_value_length_mla(dv)
    w.add_expert_feed_forward_length(toy["d_ff_expert"])
    w.add_expert_count(toy["E"])
    w.add_expert_shared_count(1)
    w.add_expert_weights_scale(mla.get("routed_scale", 2.827))
    w.add_expert_weights_norm(True)
    w.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)
    w.add_rope_dimension_count(dr)
    if mla.get("yarn", True):
        w.add_rope_scaling_type(gguf.RopeScalingType.YARN)
        w.add_rope_scaling_factor(mla["factor"])
        w.add_rope_scaling_orig_ctx_len(mla["original_max_pos"])
        w.add_rope_scaling_yarn_log_mul(0.1 * mla["mscale_all_dim"])
    w.add_tokenizer_model("none")


def write(path, seed=0, drop=(), quantize_q8=True, mla=TOY_MLA, layers=2):
    """Write the toy and return {name: the float32 values gguf-py reads back}.
    Tensors named in `drop` are left out; `quantize_q8=False` writes every
    tensor F32; `mla=TOY_MOON_MLA` writes Moonlight's variant. The metadata
    lets llama.cpp load it."""
    from gguf import GGUFWriter
    from gguf.constants import GGMLQuantizationType as Q
    from gguf.quants import dequantize, quantize
    w = GGUFWriter(str(path), "deepseek2")
    llama_metadata(w, mla, layers=layers)
    back = {}
    for name, a in tensors(seed, mla, layers=layers).items():
        if name in drop:
            continue
        if quantize_q8 and any(k in name for k in Q8_0_NAMES):
            qd = quantize(a, Q.Q8_0)
            w.add_tensor(name, qd, raw_shape=qd.shape, raw_dtype=Q.Q8_0)
            back[name] = dequantize(qd, Q.Q8_0).reshape(a.shape)
        else:
            w.add_tensor(name, a, raw_dtype=Q.F32)
            back[name] = a
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file()
    w.close()
    return back
