"""GGUF Maverick loader: synthetic-file round trip (CPU-only, no CUDA).

Writes a tiny GGUF with llama.cpp's Llama-4 MoE tensor names (Q8_0 experts,
F32 elsewhere — same mixed-type dispatch path the real UD-Q4_K_XL file
exercises with K-quants), then checks read_maverick_moe_layer returns the
right shapes/values and that the n_experts raw-slice matches a full dequant.

Run anywhere with `pip install gguf`:  python tests/test_gguf_loader.py
"""
import sys
import pathlib
import tempfile
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

E, D, DFF, LAYER = 4, 64, 96, 1


def _write_toy(path):
    from gguf import GGUFWriter
    from gguf.quants import quantize
    from gguf.constants import GGMLQuantizationType as Q
    rng = np.random.default_rng(7)
    full = {
        f"blk.{LAYER}.ffn_gate_exps.weight": rng.standard_normal((E, DFF, D)),
        f"blk.{LAYER}.ffn_up_exps.weight":   rng.standard_normal((E, DFF, D)),
        f"blk.{LAYER}.ffn_down_exps.weight": rng.standard_normal((E, D, DFF)),
        f"blk.{LAYER}.ffn_gate_inp.weight":  rng.standard_normal((E, D)),
        f"blk.{LAYER}.ffn_gate_shexp.weight": rng.standard_normal((DFF, D)),
        f"blk.{LAYER}.ffn_up_shexp.weight":   rng.standard_normal((DFF, D)),
        f"blk.{LAYER}.ffn_down_shexp.weight": rng.standard_normal((D, DFF)),
    }
    w = GGUFWriter(path, "llama4")
    for name, a in full.items():
        a = a.astype(np.float32)
        if "exps" in name:                      # quantized experts (Q8_0 stand-in)
            qd = quantize(a, Q.Q8_0)
            w.add_tensor(name, qd, raw_shape=qd.shape, raw_dtype=Q.Q8_0)
        else:
            w.add_tensor(name, a, raw_dtype=Q.F32)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file()
    w.close()
    return full


def test_read_shapes_and_slice():
    from loader import read_maverick_moe_layer
    with tempfile.TemporaryDirectory() as td:
        path = f"{td}/toy.gguf"
        full = _write_toy(path)
        out = read_maverick_moe_layer(path, LAYER)
        assert out["gate_exps"].shape == (E, DFF, D), out["gate_exps"].shape
        assert out["down_exps"].shape == (E, D, DFF)
        assert out["router"].shape == (E, D)
        assert out["gate_sh"].shape == (DFF, D)
        # F32 tensors exact; Q8_0 within block-quant tolerance
        assert np.array_equal(out["router"],
                              full[f"blk.{LAYER}.ffn_gate_inp.weight"].astype(np.float32))
        ref = full[f"blk.{LAYER}.ffn_gate_exps.weight"].astype(np.float32)
        rel = np.abs(out["gate_exps"] - ref).max() / np.abs(ref).max()
        assert rel < 0.01, f"Q8_0 round-trip rel err {rel}"
        # raw-slice (n_experts) must equal the full dequant's prefix
        sl = read_maverick_moe_layer(path, LAYER, n_experts=2)
        assert sl["gate_exps"].shape == (2, DFF, D)
        assert np.array_equal(sl["gate_exps"], out["gate_exps"][:2])
        assert np.array_equal(sl["router"], out["router"])    # non-stacked: unsliced
        print("    shapes, values, and raw expert slice all match")


def test_name_drift_is_loud():
    from loader import read_maverick_moe_layer
    with tempfile.TemporaryDirectory() as td:
        path = f"{td}/toy.gguf"
        _write_toy(path)
        try:
            read_maverick_moe_layer(path, LAYER + 1)     # wrong layer → missing names
            raise AssertionError("expected KeyError on tensor-name drift")
        except KeyError as e:
            assert "tensor-name drift" in str(e)
            print("    missing tensor names raise loudly")


if __name__ == "__main__":
    test_read_shapes_and_slice()
    test_name_drift_is_loud()
    print("=== gguf_loader: 2/2 PASS ===")
