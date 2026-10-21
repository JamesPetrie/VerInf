"""A toy GGUF with Kimi K2's tensor names, layout and mixed types, for the
loader and reference tests (CPU) and the driver suite (GPU).

Two layers as K2's first two: layer 0 latent attention and a dense FFN,
layer 1 latent attention and a top-k MoE (router, selection bias, stacked
experts, shared expert); then the final norm and an untied head. K2's
attention head dimensions are scaled down to multiples of 32 so the Q8_0
tensors (attn_k_b, attn_v_b, attn_q_b and the experts, as in UD-Q4_K_XL's
Q8_0 set) quantize; norms, router and bias are F32 as in the real file.
K-quants are exercised on the real GGUF only (gguf-py cannot write them)."""
import numpy as np

import k2_attention as ka

TOY_MLA = dict(ka.K2_MLA, d=64, H=2, q_rank=32, kv_rank=32, d_nope=32, d_rope=32, d_v=32)
TOY = dict(V=48, d_ff_dense=64, E=8, k=3, d_ff_expert=32, d_ff_shared=32)


def tensors(seed=0, mla=TOY_MLA, toy=TOY):
    """{name: float32 array in gguf-py's shape} for the two-layer toy."""
    rng = np.random.default_rng(seed)
    d, H, dn, dr, dv = mla["d"], mla["H"], mla["d_nope"], mla["d_rope"], mla["d_v"]
    qr, kr = mla["q_rank"], mla["kv_rank"]
    V, E = toy["V"], toy["E"]
    n = lambda std, *s: rng.normal(0.0, std, s).astype(np.float32)
    gain = lambda m: (1.0 + rng.normal(0.0, 0.1, m)).astype(np.float32)
    t = {"token_embd.weight": n(0.5, V, d), "output_norm.weight": gain(d),
         "output.weight": n(d ** -0.5, V, d)}
    for il in range(2):
        p = f"blk.{il}."
        t.update({
            p + "attn_norm.weight": gain(d),
            p + "attn_q_a.weight": n(d ** -0.5, qr, d),
            p + "attn_q_a_norm.weight": gain(qr),
            p + "attn_q_b.weight": n(qr ** -0.5, H * (dn + dr), qr),
            p + "attn_kv_a_mqa.weight": n(d ** -0.5, kr + dr, d),
            p + "attn_kv_a_norm.weight": gain(kr),
            p + "attn_k_b.weight": n(kr ** -0.5, H, kr, dn),
            p + "attn_v_b.weight": n(kr ** -0.5, H, dv, kr),
            p + "attn_output.weight": n((H * dv) ** -0.5, d, H * dv),
            p + "ffn_norm.weight": gain(d),
        })
    f = toy["d_ff_dense"]
    t.update({"blk.0.ffn_gate.weight": n(d ** -0.5, f, d),
              "blk.0.ffn_up.weight": n(d ** -0.5, f, d),
              "blk.0.ffn_down.weight": n(f ** -0.5, d, f)})
    fe, fs = toy["d_ff_expert"], toy["d_ff_shared"]
    t.update({"blk.1.ffn_gate_inp.weight": n(d ** -0.5, E, d),
              "blk.1.exp_probs_b.bias": rng.uniform(-0.3, 0.3, E).astype(np.float32),
              "blk.1.ffn_gate_exps.weight": n(d ** -0.5, E, fe, d),
              "blk.1.ffn_up_exps.weight": n(d ** -0.5, E, fe, d),
              "blk.1.ffn_down_exps.weight": n(fe ** -0.5, E, d, fe),
              "blk.1.ffn_gate_shexp.weight": n(d ** -0.5, fs, d),
              "blk.1.ffn_up_shexp.weight": n(d ** -0.5, fs, d),
              "blk.1.ffn_down_shexp.weight": n(fs ** -0.5, d, fs)})
    return t


Q8_0_NAMES = ("attn_q_b", "attn_k_b", "attn_v_b", "_exps", "ffn_down.", "output.weight")


def write(path, seed=0):
    """Write the toy and return {name: the float32 values gguf-py reads back}."""
    from gguf import GGUFWriter
    from gguf.constants import GGMLQuantizationType as Q
    from gguf.quants import dequantize, quantize
    w = GGUFWriter(str(path), "deepseek2")
    back = {}
    for name, a in tensors(seed).items():
        if any(k in name for k in Q8_0_NAMES):
            qd = quantize(a, Q.Q8_0)
            w.add_tensor(name, qd, raw_shape=qd.shape, raw_dtype=Q.Q8_0)
            back[name] = dequantize(qd, Q.Q8_0).reshape(a.shape)
        else:
            w.add_tensor(name, a, raw_dtype=Q.F32)
            back[name] = a
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file()
    w.close()
    return back
