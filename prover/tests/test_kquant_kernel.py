"""Fused K-quant→field kernel vs the reference numpy path: BIT-EXACT gate.

The byte→integer map IS the declared model, so the CUDA kernel must produce
identical field integers to gguf.quants.dequantize → quantize_to_field for
every quant type it claims. Runs against the real UD-Q4_K_XL file (covers
Q4_K gate/up experts, Q5_K down experts, Q6_K shared-expert down).

Run on the Spark:
    PATH=~/venv-hf/bin:$PATH python tests/test_kquant_kernel.py [gguf_path]
"""
import os
import sys
import pathlib
import time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import torch

GGUF = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/maverick-gguf/UD-Q4_K_XL")
LAYER = 1
S = 1 << 12
CASES = [  # (tensor name, expert slice or None)
    (f"blk.{LAYER}.ffn_gate_exps.weight", 0),      # Q4_K
    (f"blk.{LAYER}.ffn_down_exps.weight", 0),      # Q5_K
    (f"blk.{LAYER}.ffn_down_shexp.weight", None),  # Q6_K
    (f"blk.{LAYER}.ffn_gate_exps.weight", 5),      # Q4_K, different expert
]


def main():
    from gguf.quants import dequantize
    from loader import _gguf_by_name, quantize_to_field
    from kquant_cuda import kquant_to_field
    by = _gguf_by_name(GGUF)
    failed = 0
    for name, e in CASES:
        t = by[name]
        qt = t.tensor_type.name
        raw = np.ascontiguousarray(t.data[e] if e is not None else t.data)
        # reference: numpy dequant → fp32 → field (row-major, no transpose here)
        ref_f = dequantize(raw, t.tensor_type)
        ref = quantize_to_field(torch.from_numpy(ref_f.copy()), S).reshape(-1)
        # fused kernel
        t0 = time.time()
        got = kquant_to_field(torch.from_numpy(raw).cuda(), qt, S)
        torch.cuda.synchronize()
        dt = time.time() - t0
        eq = torch.equal(ref.view(torch.int64), got.view(torch.int64))
        n = got.numel()
        if not eq:
            bad = (ref.view(torch.int64) != got.view(torch.int64)).sum().item()
            print(f"[XX ] {name}{'' if e is None else f'[{e}]'} ({qt}): "
                  f"{bad}/{n} integers differ")
            failed += 1
        else:
            print(f"[OK ] {name}{'' if e is None else f'[{e}]'} ({qt}): "
                  f"bit-exact over {n} values, kernel {dt*1e3:.0f} ms "
                  f"({n/dt/1e6:.0f}M elem/s)")
    print(f"=== kquant_kernel: {len(CASES)-failed}/{len(CASES)} "
          f"{'PASS' if not failed else 'FAIL'} ===")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
