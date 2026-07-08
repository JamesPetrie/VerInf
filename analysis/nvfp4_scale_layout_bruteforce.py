#!/usr/bin/env python
"""Brute-force the NVFP4 block-scale layout/swizzle convention for _scaled_mm_v2.
Decode/ideal are fixed (natural, already validated bit-exact at unit scale).
Vary only how the block-scale tensors are laid out + the SwizzleType flag."""
import torch
from torch.nn.functional import ScalingType, SwizzleType
from torch.testing._internal.common_quantized import _bfloat16_to_float4_e2m1fn_x2, to_blocked

dev = "cuda"; BLK = 16; F4MAX = 6.0; E4M3 = torch.float8_e4m3fn
torch.manual_seed(0)
_LUT = torch.tensor([0,.5,1,1.5,2,3,4,6, 0,-.5,-1,-1.5,-2,-3,-4,-6], device=dev, dtype=torch.float64)

def quant(x):
    R, C = x.shape
    xb = x.float().reshape(R, C//BLK, BLK)
    amax = xb.abs().amax(-1, keepdim=True)
    sc = (amax/F4MAX).to(E4M3).float(); sc = torch.where(sc==0, torch.ones_like(sc), sc)
    fp4 = _bfloat16_to_float4_e2m1fn_x2((xb/sc).reshape(R, C).clamp(-F4MAX, F4MAX).to(torch.bfloat16))
    return fp4, sc.squeeze(-1).to(E4M3)            # [R,C//2], [R,C//16] natural

def decode(fp4, s):
    b = fp4.view(torch.uint8).long(); R, Ch = b.shape
    v = torch.empty(R, Ch*2, dtype=torch.float64, device=b.device)
    v[:,0::2] = _LUT[b & 0xF]; v[:,1::2] = _LUT[b >> 4]
    return v * s.float().double().repeat_interleave(BLK, dim=1)

def build(nat, is_b, layout):
    H, W = nat.shape                                # nat = [free, K//16]
    if layout == "nat":
        return (nat.t() if is_b else nat).contiguous()
    bl = to_blocked(nat)                            # 1D swizzled
    if layout == "flat":
        return bl.contiguous()
    if layout == "natshape":
        return bl.reshape(W, H).contiguous() if is_b else bl.reshape(H, W).contiguous()
    if layout == "blk2d":
        r = 32*((H+127)//128); c = 16*((W+3)//4)
        return bl.reshape(r, c).contiguous()

M, K, N = 256, 1024, 512
A = torch.randn(M, K, device=dev); B = torch.randn(K, N, device=dev)
A_fp4, A_s = quant(A)                               # A_s [M,K//16]
Bt_fp4, Bt_s = quant(B.t().contiguous())            # Bt_s [N,K//16]
B_fp4 = Bt_fp4.t()
ideal = decode(A_fp4, A_s) @ decode(Bt_fp4, Bt_s).t()

print(f"[{M}x{K}x{N}]  (looking for code_match ~1.0)")
for layout in ["nat", "flat", "natshape", "blk2d"]:
    for sw in [SwizzleType.NO_SWIZZLE, SwizzleType.SWIZZLE_32_4_4]:
        try:
            sa = build(A_s, False, layout); sb = build(Bt_s, True, layout)
            hw = torch._scaled_mm_v2(A_fp4, B_fp4,
                [sa], [ScalingType.BlockWise1x16], [sw],
                [sb], [ScalingType.BlockWise1x16], [sw],
                None, torch.float32).double()
            err = (hw - ideal).abs().max().item()
            fp4o, _ = quant(hw.float()); fp4i, _ = quant(ideal.float())
            cm = (fp4o.view(torch.uint8) == fp4i.view(torch.uint8)).float().mean().item()
            print(f"  layout={layout:9s} swizzle={sw.name:15s} max_abs={err:10.3e} code_match={cm:.4f}")
        except Exception as e:
            print(f"  layout={layout:9s} swizzle={sw.name:15s} ERR {repr(e)[:70]}")
