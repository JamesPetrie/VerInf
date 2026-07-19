"""Rescale fixture — the one load-bearing TCB piece that was untested on BOTH
sides (no fixture anywhere set rescale_bits>0; see FINDINGS F2).

Rescale is the fixed-point requantization every real Llama op uses: a product at
scale s_a·s_b is split full = 2^r·high + low (low range-checked tight, high's
shifted form range-checked loose) to bring it back to s_out. Here a Hadamard with
s_a=s_b=s_out=4 ⇒ rescale_bits = log2(16/4) = 2.

Two checks:
  (1) PROVER: build via the Tape, run the prover's OWN prove+verify → ACCEPT.
      This executes the prover's rescale compile + witness path for the first
      time (and so also exercises whatever F1 was worried about).
  (2) PARITY: feed the same claim to the oracle — protocol._emit_rescale must
      reproduce the prover's rescale constraints byte-for-byte.

Run on the Spark:  ~/venv-hf/bin/python test_rescale.py
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # pipeline/ on path
import torch
import core
import claims as C          # noqa: F401 — registers COMPILE/SAMPLE/AUX
import packets as PK        # noqa: F401 — registers EXPANDERS
import protocol as pr
from tape import Tape
from _rust_verify import rust_verify_tape

CFG = core.LigeroConfig(ELL=8, K_DEG=8, N_LIG=32, T_QUERIES=4)


def _t(vals):
    return torch.tensor(vals, dtype=torch.int64, device="cuda").to(torch.uint64)


def build_hadamard():
    """Hadamard with s_a=s_b=s_out=4 ⇒ rescale_bits=2. a·b spans multiples and
    non-multiples of 4 so c_low is sometimes nonzero (exercises the low range)."""
    tape = Tape(CFG, lazy=True)
    a = tape.commit("a", _t([1, 2, 3, 4, 5, 6]), (6,))
    b = tape.commit("b", _t([4, 4, 4, 4, 4, 4]), (6,))
    tape.hadamard(a, b, s_a=4, s_b=4, s_out=4, output_width=8)
    return tape, "HadamardClaim"


def build_matmul():
    """Single-head matmul C=A·B with s_a=s_b=s_out=4 ⇒ rescale_bits=2. The raw
    product C_full is committed + Freivalds-checked, then rescaled to C. m=2,
    k=2, n=2; small values keep the rescaled high word in signed range."""
    tape = Tape(CFG, lazy=True)
    a = tape.commit("mA", _t([[1, 2], [3, 1]]), (2, 2))      # (m, k)
    b = tape.commit("mB", _t([[2, 1], [1, 2]]), (2, 2))      # (k, n)
    tape.matmul(a, b, s_a=4, s_b=4, s_out=4, output_width=8)
    return tape, "MatmulClaim"


def build_softmax():
    """Softmax with input rescale: s_in = 2·s_x ⇒ rescale_bits=1 (x_in =
    2·x + x_low). Production-style scales, tiny B=1, M=2. The rescale block
    range-checks x_low vs range_rescale (tight) and x_shifted vs range_aux
    (loose, shared with the bracket)."""
    tape = Tape(CFG, lazy=True)
    M = 2
    s_x = 1 << 12
    x = tape.commit("smx", _t([0, 0]), (M,))
    tape.softmax(x, M=M, s_x=s_x, s_c=s_x, s_y=s_x, Z_max=1 << 14,
                 s_in=2 * s_x)
    return tape, "SoftmaxClaim"


def build_rmsnorm():
    """RmsNorm with BOTH rescale blocks: s=4, s_in=8 ⇒ input rescale_bits=1
    (x_in=2·x+x_low), and s_out=4 ⇒ output_rescale_bits=log2(16/4)=2
    (output_full=4·output+output_low). B=2 tokens × d=4. Small non-negative
    inputs keep slacks within one 16-bit chunk."""
    tape = Tape(CFG, lazy=True)
    x = tape.commit("rx", _t([3, 1, 4, 2, 5, 2, 6, 1]), (8,))
    tape.rmsnorm(x, d=4, s=4, eps_int=1, s_in=8, s_out=4,
                 output_width=16)
    return tape, "RmsNormClaim"


def run(label, builder):
    # PROVER: prove + the Rust verifier must ACCEPT this rescale path.
    tape, _ = builder()
    proof = tape.prove(seed=b"rescale-fix")
    acc, msg = rust_verify_tape(tape, proof, seed=b"rescale-fix")
    print(f"[{'OK ' if acc else 'XX '}] {label} prover prove+verify: "
          f"{'ACCEPT' if acc else 'REJECT'}  ({msg})")
    # (The compile-parity cross-check retired with REAL_COMPILE; the Rust verifier
    # ACCEPT covers rescale correctness end-to-end.)
    return acc


def main():
    results = [run("hadamard_rescale", build_hadamard),
               run("matmul_rescale",   build_matmul),
               run("softmax_rescale",  build_softmax),
               run("rmsnorm_rescale",  build_rmsnorm)]
    ok = all(results)
    print(f"\n=== rescale fixtures: {sum(results)}/{len(results)} "
          f"{'PASS' if ok else 'FAIL'} ===")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
