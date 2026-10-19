"""The torch dequantizer (prover/gguf_dequant_torch.py) against gguf-py's,
CPU: random bytes, so every bit pattern of scales, mins and quants is
exercised, NaN and infinite float16 scales included, and the float32 bits
must be equal. VERINF_DEQUANT_DEVICE runs the same on a GPU (the session kit
sets it to cuda)."""
import os
import pathlib
import sys

import numpy as np
import pytest
from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType as Q
from gguf.quants import dequantize

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import gguf_dequant_torch as gd             # noqa: E402

DEVICE = os.environ.get("VERINF_DEQUANT_DEVICE", "cpu")


def _bits_equal(a, b):
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    na, nb = np.isnan(a), np.isnan(b)
    return a.shape == b.shape and np.array_equal(na, nb) and \
        np.array_equal(a[~na].view(np.uint32), b[~nb].view(np.uint32))


@pytest.mark.parametrize("qtype", [Q.Q8_0, Q.Q5_0, Q.Q4_K, Q.Q5_K, Q.Q6_K])
def test_every_bit_pattern_dequantizes_as_gguf_py_does(qtype):
    block_size, type_size = GGML_QUANT_SIZES[qtype]
    rng = np.random.default_rng(int(qtype))
    raw = rng.integers(0, 256, (7, 23 * type_size), dtype=np.uint8)       # 7 rows of 23 blocks
    with np.errstate(all="ignore"):                  # NaN and infinite scales, as intended
        want = dequantize(raw, qtype)
    got = gd.dequantize(raw, qtype, device=DEVICE).cpu().numpy()
    assert got.shape == want.shape == (7, 23 * block_size)
    assert _bits_equal(got, want)


def test_plain_types_and_refusals():
    x = np.random.default_rng(0).normal(0, 1, (3, 8)).astype(np.float32)
    assert _bits_equal(gd.dequantize(x.view(np.uint8), Q.F32, device=DEVICE).cpu().numpy().reshape(3, 8), x)
    h = x.astype(np.float16)
    assert _bits_equal(gd.dequantize(h.view(np.uint8), Q.F16, device=DEVICE).cpu().numpy().reshape(3, 8),
                       h.astype(np.float32))
    with pytest.raises(NotImplementedError):
        gd.dequantize(np.zeros((1, 66), np.uint8), Q.IQ4_XS)
    with pytest.raises(ValueError):
        gd.dequantize(np.zeros((1, 33), np.uint8), Q.Q8_0)
