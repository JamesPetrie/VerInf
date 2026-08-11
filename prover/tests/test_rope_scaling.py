"""Llama-3 RoPE frequency scaling (rope_scaling, rope_type "llama3"):
golden vectors + backward compatibility + HF formula agreement.

The cos/sin tables are public constants computed INDEPENDENTLY by the Python
prover (claims._rope_cos_sin) and the Rust verifier (handlers.rs
rope_cos_sin). The golden vectors here are asserted bit-for-bit by BOTH test
suites (Rust: rope_scaling_tests in handlers.rs) so any cross-language drift
fails a unit test instead of rejecting whole proofs with no diagnostic.
Vectors were generated from the Python implementation on the Spark; the
d_h=8 config exercises all three ramp branches (keep / interpolate / ÷factor).

Run: python tests/run_tests.py test_rope_scaling
"""
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from claims import RoPEConfig, _rope_cos_sin, _rope_scaled_inv_freq, P

SCALED = dict(base=500000.0, scale_factor=32.0, low_freq_factor=1.0,
              high_freq_factor=4.0, original_max_pos=8192)

# Config A: SEQ=3, d_h=8, s_x=4096 — hits all three ramp branches
# (k=0,1 keep; k=2 interpolate; k=3 ÷32).
A_COS = [4096, 4096, 4096, 4096, 2213, 4093, 4096, 4096,
         18446744069414582616, 4084, 4096, 4096]
A_SIN = [0, 0, 0, 0, 3447, 154, 2, 0, 3724, 308, 4, 0]

# Config B: SEQ=1 at position_offset=3, d_h=64 (the Llama-3.2-1B head dim).
B_COS = [18446744069414580266, 18446744069414582651, 1012, 2620, 3422, 3795,
         3962, 4037, 4070, 4085, 4091, 4094, 4095, 4096, 4096, 4096, 4096,
         4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096,
         4096, 4096, 4096, 4096]
B_SIN = [578, 3740, 3969, 3148, 2251, 1542, 1038, 693, 461, 306, 203, 135,
         90, 59, 39, 16, 5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


def test_golden_vectors():
    c, s = _rope_cos_sin(RoPEConfig(SEQ=3, d_h=8, s_x=4096, **SCALED))
    assert c == A_COS and s == A_SIN, "config A drifted from golden vectors"
    c, s = _rope_cos_sin(RoPEConfig(SEQ=1, d_h=64, s_x=4096,
                                    position_offset=3, **SCALED))
    assert c == B_COS and s == B_SIN, "config B drifted from golden vectors"


def test_unscaled_matches_legacy_expression():
    """original_max_pos=0 (and all defaults) must reproduce the pre-scaling
    tables byte-for-byte — old proof dumps must verify unchanged."""
    cfg = RoPEConfig(SEQ=5, d_h=8, s_x=4096, base=10000.0, position_offset=2)
    c, s = _rope_cos_sin(cfg)
    exp_c, exp_s = [], []
    for seq in range(cfg.SEQ):                     # the ORIGINAL expression
        pos = seq + cfg.position_offset
        for k in range(cfg.d_h // 2):
            theta = pos / (cfg.base ** (2 * k / cfg.d_h))
            exp_c.append(int(round(math.cos(theta) * cfg.s_x)) % P)
            exp_s.append(int(round(math.sin(theta) * cfg.s_x)) % P)
    assert c == exp_c and s == exp_s


def test_scaled_matches_hf_formula():
    """Our scalar ramp must agree with the transformers vectorized formula
    (_compute_llama3_parameters, quoted verbatim below in torch) to float32
    precision on the real Llama-3.2-1B parameters (d_h=64, θ=500000)."""
    import torch
    base, dim = 500000.0, 64
    factor, low_freq_factor, high_freq_factor, old_context_len = 32.0, 1.0, 4.0, 8192
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64)
                               .to(dtype=torch.float) / dim))
    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    wavelen = 2 * math.pi / inv_freq
    inv_freq_llama = torch.where(wavelen > low_freq_wavelen,
                                 inv_freq / factor, inv_freq)
    smooth_factor = ((old_context_len / wavelen - low_freq_factor)
                     / (high_freq_factor - low_freq_factor))
    smoothed_inv_freq = ((1 - smooth_factor) * inv_freq_llama / factor
                         + smooth_factor * inv_freq_llama)
    is_medium_freq = (~(wavelen < high_freq_wavelen)) & (~(wavelen > low_freq_wavelen))
    inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)

    cfg = RoPEConfig(SEQ=1, d_h=dim, s_x=4096, **SCALED)
    ours = [_rope_scaled_inv_freq(cfg, k) for k in range(dim // 2)]
    hf = inv_freq_llama.tolist()
    for k, (a, b) in enumerate(zip(ours, hf)):
        assert math.isclose(a, b, rel_tol=1e-5), (
            f"k={k}: ours={a!r} vs HF={b!r} — ramp formula drifted from HF")


def test_config_serializes_scaling_fields():
    """The generic dataclass config dump must carry the new fields (floats as
    floats), and the defaults must match what the Rust side assumes for old
    dumps (scale/low/high=1.0, original_max_pos=0)."""
    from protocol import _ser_value
    d = _ser_value(RoPEConfig(SEQ=2, d_h=8, s_x=4096, **SCALED), {})["config"]
    assert d["scale_factor"] == 32.0 and isinstance(d["scale_factor"], float)
    assert d["original_max_pos"] == 8192
    d0 = _ser_value(RoPEConfig(SEQ=2, d_h=8, s_x=4096), {})["config"]
    assert (d0["scale_factor"], d0["low_freq_factor"],
            d0["high_freq_factor"], d0["original_max_pos"]) == (1.0, 1.0, 1.0, 0)


def _build_rope_claim(cfg_rope, seed_val: int):
    """RoPE fixture at an arbitrary RoPEConfig (generalizes test_claims'
    _build_rope): x random field elements, x_rot computed with the SAME
    cos/sin tables the claim compiles, so the rotation relation holds
    bit-for-bit."""
    import random
    from core import Variable
    from claims import RoPEClaim
    half = cfg_rope.d_h // 2
    H = cfg_rope.heads
    rng = random.Random(seed_val)
    L = cfg_rope.SEQ * H * cfg_rope.d_h
    x_vals = [rng.randrange(P) for _ in range(L)]
    c_l, s_l = _rope_cos_sin(cfg_rope)
    x_rot_vals = [0] * L
    for seq in range(cfg_rope.SEQ):
        for h in range(H):
            for k in range(half):
                idx_lo = seq * H * cfg_rope.d_h + h * cfg_rope.d_h + k
                idx_hi = idx_lo + half
                ci = seq * half + k
                c, s = c_l[ci], s_l[ci]
                x_rot_vals[idx_lo] = (c * x_vals[idx_lo]
                                     + (P - s * x_vals[idx_hi] % P) % P) % P
                x_rot_vals[idx_hi] = (s * x_vals[idx_lo]
                                     + c * x_vals[idx_hi]) % P
    x     = Variable("rope_x",    length=L)
    x_rot = Variable("rope_xrot", length=L)
    return ([RoPEClaim(x=x, x_rot=x_rot, config=cfg_rope)],
            {x: x_vals, x_rot: x_rot_vals})


SCALED_E2E = RoPEConfig(SEQ=4, d_h=8, s_x=4096, **SCALED)


def test_honest_scaled_rope_rust_verifies():
    """End-to-end cross-language: prove a RoPE claim whose tables use the
    llama3 ramp, verify with the independent Rust binary (which recomputes
    the tables from the serialized config). ACCEPT ⇒ serialization + both
    table implementations agree."""
    from test_prover import prove
    from _rust_verify import rust_verify
    from claims import CFG
    claims, inputs = _build_rope_claim(SCALED_E2E, seed_val=43)
    proof = prove(claims, inputs, seed=b"rope-scaled-h", cfg=CFG)
    acc, msg = rust_verify(claims, proof, seed=b"rope-scaled-h", cfg=CFG)
    assert acc, f"honest scaled rope should ACCEPT: {msg}"


def test_scaling_stripped_from_claim_rejects():
    """Tamper: prove with the ramp ON, then present a claim that says
    scaling is OFF. The verifier recomputes different cos/sin tables and
    must REJECT — i.e. the scaling parameters are soundness-bearing, not
    advisory."""
    import dataclasses
    from test_prover import prove
    from _rust_verify import rust_verify
    from claims import CFG, RoPEClaim
    stripped = dataclasses.replace(
        SCALED_E2E, scale_factor=1.0, low_freq_factor=1.0,
        high_freq_factor=1.0, original_max_pos=0)
    assert _rope_cos_sin(stripped) != _rope_cos_sin(SCALED_E2E), (
        "fixture too coarse: scaled and unscaled tables coincide, "
        "the tamper below would be vacuous")
    claims, inputs = _build_rope_claim(SCALED_E2E, seed_val=44)
    proof = prove(claims, inputs, seed=b"rope-scaled-t", cfg=CFG)
    tampered = [dataclasses.replace(claims[0], config=stripped)]
    acc, _ = rust_verify(tampered, proof, seed=b"rope-scaled-t", cfg=CFG)
    assert not acc, "verifier ACCEPTed a proof whose rope_scaling was stripped"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"=== rope_scaling: {len(fns)}/{len(fns)} PASS ===")
