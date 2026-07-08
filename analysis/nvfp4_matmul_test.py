#!/usr/bin/env python
"""Validated NVFP4 matmul vs 'decode -> ideal fp64 dot -> requantize' model.
Correct convention: to_blocked(natural per-16 E4M3 scale) + SWIZZLE_32_4_4.
Sweeps K and seeds; reports accumulation fidelity and output-code agreement."""
import torch
from torch.nn.functional import ScalingType, SwizzleType
from torch.testing._internal.common_quantized import _bfloat16_to_float4_e2m1fn_x2, to_blocked

dev = "cuda"; BLK = 16; F4MAX = 6.0; E4M3 = torch.float8_e4m3fn
_LUT = torch.tensor([0,.5,1,1.5,2,3,4,6, 0,-.5,-1,-1.5,-2,-3,-4,-6], device=dev, dtype=torch.float64)

def quant(x):
    R, C = x.shape
    xb = x.float().reshape(R, C//BLK, BLK)
    amax = xb.abs().amax(-1, keepdim=True)
    sc = (amax/F4MAX).to(E4M3).float(); sc = torch.where(sc==0, torch.ones_like(sc), sc)
    fp4 = _bfloat16_to_float4_e2m1fn_x2((xb/sc).reshape(R, C).clamp(-F4MAX, F4MAX).to(torch.bfloat16))
    return fp4, sc.squeeze(-1).to(E4M3)

def decode(fp4, s):
    b = fp4.view(torch.uint8).long(); R, Ch = b.shape
    v = torch.empty(R, Ch*2, dtype=torch.float64, device=b.device)
    v[:,0::2] = _LUT[b & 0xF]; v[:,1::2] = _LUT[b >> 4]
    return v * s.float().double().repeat_interleave(BLK, dim=1)

def nvfp4_mm(A_fp4, A_s, B_fp4, B_s_free):
    return torch._scaled_mm_v2(A_fp4, B_fp4,
        [to_blocked(A_s)],      [ScalingType.BlockWise1x16], [SwizzleType.SWIZZLE_32_4_4],
        [to_blocked(B_s_free)], [ScalingType.BlockWise1x16], [SwizzleType.SWIZZLE_32_4_4],
        None, torch.float32)

print("torch", torch.__version__, "cap", torch.cuda.get_device_capability())
print(f"{'K':>6} {'seed':>4} | {'frac_acc_bitexact':>17} {'max_abs_err':>12} | {'out_code_match':>14} {'scale_match':>11}")
M = N = 256
for K in [256, 1024, 4096, 16384, 65536]:
    for seed in [0, 1, 2]:
        torch.manual_seed(seed)
        A = torch.randn(M, K, device=dev); B = torch.randn(K, N, device=dev)
        A_fp4, A_s = quant(A)
        Bt_fp4, Bt_s = quant(B.t().contiguous())   # Bt_s [N,K//16] = B scale in free-dim layout
        ideal = decode(A_fp4, A_s) @ decode(Bt_fp4, Bt_s).t()
        hw = nvfp4_mm(A_fp4, A_s, Bt_fp4.t(), Bt_s).double()
        err = (hw - ideal).abs()
        frac = (hw == ideal).double().mean().item()
        fo, so = quant(hw.float()); fi, si = quant(ideal.float())
        cm = (fo.view(torch.uint8) == fi.view(torch.uint8)).double().mean().item()
        sm = (so.view(torch.uint8) == si.view(torch.uint8)).double().mean().item()
        print(f"{K:>6} {seed:>4} | {frac:>17.5f} {err.max().item():>12.3e} | {cm:>14.6f} {sm:>11.5f}")
