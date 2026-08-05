"""Measure the per-primitive RATES of this machine.

Rates are the only machine-dependent input to the cost model. They are MEASURED
here, once, and never fitted to a prove run -- so a modelling error cannot hide
inside them (which is exactly what kappa allowed).

  .venv/bin/python layergkr/bench/rates.py            # this box (python path)
  .venv/bin/python layergkr/bench/rates.py --torch    # also the GPU field path

The python numbers describe the prototype. The torch/GPU numbers describe the
path the production prover actually uses, and are the ones to carry to a 400B
projection; both are reported so the difference is visible rather than assumed.
"""
import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import blake3

from prover.protocol import P


def time_op(fn, n: int) -> float:
    """Seconds per operation, best of three, loop overhead subtracted."""
    best = float("inf")
    for _ in range(3):
        t0 = time.perf_counter()
        fn(n)
        best = min(best, time.perf_counter() - t0)
    empty = float("inf")
    for _ in range(3):
        t0 = time.perf_counter()
        _spin(n)
        empty = min(empty, time.perf_counter() - t0)
    return max(best - empty, 0.0) / n


def _spin(n):
    x = 0
    for _ in range(n):
        x += 1
    return x


def _muls(n):
    a, b, acc = 0x1234567890ABCDEF, 0xFEDCBA0987654321, 0
    for _ in range(n):
        acc = (a * b) % P
    return acc


def _muls_defer(n):
    """How the hot loops actually multiply: accumulate a*b into a wide Python int
    and reduce once. Counting these as reduced multiplies over-prices them ~2x."""
    a, b, acc = 0x1234567890ABCDEF, 0xFEDCBA0987654321, 0
    for _ in range(n):
        acc += a * b
    return acc % P


def _adds(n):
    a, b, acc = 0x1234567890ABCDEF, 0xFEDCBA0987654321, 0
    for _ in range(n):
        acc = (a + b) % P
    return acc


def measure_python(n: int = 200_000) -> dict:
    mul_s = time_op(_muls, n)
    mul_defer_s = time_op(_muls_defer, n)
    add_s = time_op(_adds, n)
    buf = b"\x5a" * (1 << 20)
    t0 = time.perf_counter()
    reps = 64
    for _ in range(reps):
        blake3.blake3(buf).digest()
    hash_GBps = (reps * len(buf)) / (time.perf_counter() - t0) / 1e9
    return {"name": "python-ref", "mul_ns": mul_s * 1e9,
            "mul_defer_ns": mul_defer_s * 1e9, "add_ns": add_s * 1e9,
            "hash_GBps": hash_GBps}


def measure_torch(n: int = 1 << 22) -> dict:
    """The GPU field path the production prover uses. Reduction is the same
    Goldilocks mul the existing kernels do; this is a floor, not a claim about
    the tuned kernels."""
    try:
        import torch
    except ImportError:
        return {}
    if not torch.cuda.is_available():
        return {}
    dev = "cuda"
    a = torch.randint(0, 1 << 62, (n,), dtype=torch.int64, device=dev)
    b = torch.randint(0, 1 << 62, (n,), dtype=torch.int64, device=dev)

    def bench(fn, reps=20):
        fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / reps / n

    mul_s = bench(lambda: torch.remainder(a * b, P))
    add_s = bench(lambda: torch.remainder(a + b, P))
    name = torch.cuda.get_device_name(0)
    return {"name": f"torch:{name}", "mul_ns": mul_s * 1e9,
            "mul_defer_ns": mul_s * 1e9,   # SIMD: no deferred-reduction saving
            "add_ns": add_s * 1e9,
            "hash_GBps": measure_python(20_000)["hash_GBps"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--torch", action="store_true", help="also measure the GPU field path")
    ap.add_argument("--out", default=str(pathlib.Path(__file__).parent / "rates.json"))
    args = ap.parse_args()

    cards = [measure_python()]
    if args.torch:
        t = measure_torch()
        if t:
            cards.append(t)
        else:
            print("(no CUDA torch available; python rates only)")

    for c in cards:
        print(f"{c['name']:<28} mul {c['mul_ns']:7.2f} ns  mul-defer "
              f"{c['mul_defer_ns']:7.2f} ns  add {c['add_ns']:7.2f} ns  "
              f"blake3 {c['hash_GBps']:5.2f} GB/s")
    pathlib.Path(args.out).write_text(json.dumps(cards, indent=2))
    print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
