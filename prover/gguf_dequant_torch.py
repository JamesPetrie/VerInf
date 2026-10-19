"""GGUF block dequantization in torch, on any device, bit-identical to
gguf-py's numpy (gguf.quants) for the types K2's UD-Q4_K_XL and Moonlight's
Q4_K_M use: F32, F16, Q8_0, Q5_0, Q4_K, Q5_K, Q6_K
(analysis/k2-fidelity-study-design.md §5.3).

Each port keeps gguf-py's float32 operations in its order, one rounded
operation per step: eager torch runs each as its own kernel, so no fused
multiply-add can change a rounding. Equality with gguf-py is tested on every
bit pattern on the CPU (tests/test_gguf_dequant_torch.py); the session kit
repeats that on the GPU before the full-depth reference relies on it."""
import numpy as np
import torch
from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType as Q

QK_K = 256


def _f16(b):
    """Two bytes per row as float16, widened to float32."""
    return b.contiguous().view(torch.float16).to(torch.float32)


def _u8_shift(x, shifts):
    """x (…, 1, n) >> shifts along a new axis, uint8."""
    return x >> torch.as_tensor(shifts, dtype=torch.uint8, device=x.device).reshape(1, 1, -1, 1)


def _q8_0(b):
    d, x = b[:, :2], b[:, 2:]
    return x.contiguous().view(torch.int8).to(torch.float32) * _f16(d)


def _q5_0(b):
    n = b.shape[0]
    d = _f16(b[:, :2])
    qh = b[:, 2:6].contiguous().view(torch.int32).reshape(n, 1)
    qs = b[:, 6:]
    qh = (qh >> torch.arange(32, dtype=torch.int32, device=b.device).reshape(1, 32)) & 1
    ql = _u8_shift(qs.reshape(n, -1, 1, 16), [0, 4])
    ql = (ql & 0x0F).reshape(n, -1)
    q = (ql | (qh.to(torch.uint8) << 4)).to(torch.int8) - 16
    return d * q.to(torch.float32)


def _scale_min_k4(scales):
    n = scales.shape[0]
    s = scales.reshape(n, 3, 4)
    d, m, m_d = s[:, 0:1, :], s[:, 1:2, :], s[:, 2:3, :]
    sc = torch.cat([d & 0x3F, (m_d & 0x0F) | ((d >> 2) & 0x30)], dim=-1)
    mn = torch.cat([m & 0x3F, (m_d >> 4) | ((m >> 2) & 0x30)], dim=-1)
    return sc.reshape(n, 8), mn.reshape(n, 8)


def _q4_k(b):
    n = b.shape[0]
    d, dmin, scales, qs = _f16(b[:, 0:2]), _f16(b[:, 2:4]), b[:, 4:16], b[:, 16:]
    sc, m = _scale_min_k4(scales)
    d = (d * sc.to(torch.float32)).reshape(n, -1, 1)
    dm = (dmin * m.to(torch.float32)).reshape(n, -1, 1)
    q = _u8_shift(qs.reshape(n, -1, 1, 32), [0, 4])
    q = (q & 0x0F).reshape(n, -1, 32).to(torch.float32)
    return (d * q - dm).reshape(n, QK_K)


def _q5_k(b):
    n = b.shape[0]
    d, dmin, scales = _f16(b[:, 0:2]), _f16(b[:, 2:4]), b[:, 4:16]
    qh, qs = b[:, 16:16 + QK_K // 8], b[:, 16 + QK_K // 8:]
    sc, m = _scale_min_k4(scales)
    d = (d * sc.to(torch.float32)).reshape(n, -1, 1)
    dm = (dmin * m.to(torch.float32)).reshape(n, -1, 1)
    ql = _u8_shift(qs.reshape(n, -1, 1, 32), [0, 4])
    qh = _u8_shift(qh.reshape(n, -1, 1, 32), list(range(8)))
    ql = (ql & 0x0F).reshape(n, -1, 32)
    qh = (qh & 0x01).reshape(n, -1, 32)
    q = (ql | (qh << 4)).to(torch.float32)
    return (d * q - dm).reshape(n, QK_K)


def _q6_k(b):
    n = b.shape[0]
    ql, qh = b[:, :QK_K // 2], b[:, QK_K // 2:QK_K // 2 + QK_K // 4]
    scales = b[:, QK_K // 2 + QK_K // 4:QK_K // 2 + QK_K // 4 + QK_K // 16]
    d = _f16(b[:, -2:])
    scales = scales.contiguous().view(torch.int8).to(torch.float32)
    d = (d * scales).reshape(n, QK_K // 16, 1)
    ql = _u8_shift(ql.reshape(n, -1, 1, 64), [0, 4])
    ql = (ql & 0x0F).reshape(n, -1, 32)
    qh = _u8_shift(qh.reshape(n, -1, 1, 32), [0, 2, 4, 6])
    qh = (qh & 0x03).reshape(n, -1, 32)
    q = (ql | (qh << 4)).to(torch.int8) - 32
    q = q.reshape(n, QK_K // 16, -1).to(torch.float32)
    return (d * q).reshape(n, QK_K)


_BLOCKS = {Q.Q8_0: _q8_0, Q.Q5_0: _q5_0, Q.Q4_K: _q4_k, Q.Q5_K: _q5_k, Q.Q6_K: _q6_k}
SUPPORTED = (Q.F32, Q.F16) + tuple(_BLOCKS)


def dequantize(data, qtype, device="cpu"):
    """gguf.quants.dequantize in torch: `data` as a ReaderTensor holds it
    (uint8 rows of packed blocks, or F32/F16 values), returned float32 on
    `device` in gguf-py's shape."""
    qtype = Q(qtype)
    arr = np.array(data, copy=True, order="C")     # writable: torch will not wrap a read-only memmap
    if qtype == Q.F32:
        return torch.as_tensor(arr.view(np.float32), device=device)
    if qtype == Q.F16:
        return torch.as_tensor(arr.view(np.float16), device=device).to(torch.float32)
    if qtype not in _BLOCKS:
        raise NotImplementedError(f"{qtype.name} is not ported")
    block_size, type_size = GGML_QUANT_SIZES[qtype]
    rows = arr.shape[:-1]
    if arr.shape[-1] % type_size:
        raise ValueError("the last axis is not whole blocks")
    blocks = torch.as_tensor(arr.reshape(-1, type_size), device=device)
    out = _BLOCKS[qtype](blocks)
    return out.reshape(*rows, arr.shape[-1] // type_size * block_size)
