"""Production-scale regression: matmul + silu shapes, end-to-end prove + verify
under the design-feasibility §3.1 cfg (ELL=8192, K_DEG=16384, N_LIG=65536,
T_QUERIES=80). Reports wall-clock for each shape; ACCEPTs all.

Matmul shapes — every one Llama 2 7B SwiGLU needs:
  attention q/k/v/o_proj   m=2 k=4096  n=4096   (k ≤ ELL)
  FFN gate_proj / up_proj  m=2 k=4096  n=11008  (k ≤ ELL, n > ELL)
  FFN down_proj            m=2 k=11008 n=4096   (k > ELL → multi-row aux)

Silu shapes — exercises the full sign-magnitude + saturation construction
at Q-format-realistic scale, one paired silu_table at 14-bit + range tables:
  silu_14bit  L=4096  (one FFN layer's worth of silu calls)

Each shape is one prove + verify on independent random witness data.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Tuple

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from core import P, Variable, LigeroConfig, prove, verify
from cuda_primitives import gl_matmul, gl_neg
from test_pipeline import matmul_claim, MatmulClaim   # noqa: F401  (registers protocol)
from tape import Tape, SILU_14BIT, SiluConfig, RmsNormConfig, SoftmaxConfig


CFG = LigeroConfig(ELL=8192, K_DEG=16384, N_LIG=65536, T_QUERIES=80)


def _rand_canonical(shape) -> torch.Tensor:
    return torch.randint(0, 1 << 62, shape, dtype=torch.int64, device="cuda").to(torch.uint64)


def run_one(name: str, m: int, k: int, n: int) -> Tuple[float, float]:
    A = Variable("A", length=m * k)
    B = Variable("B", length=k * n)
    C = Variable("C", length=m * n)
    claim = matmul_claim("mm", A, B, C, m=m, k=k, n=n)

    A_t = _rand_canonical((m, k))
    B_t = _rand_canonical((k, n))
    C_t = gl_matmul(A_t, B_t)
    torch.cuda.synchronize()
    inputs = {A: A_t.view(-1), B: B_t.view(-1), C: C_t.view(-1)}

    t0 = time.time(); proof = prove([claim], inputs, seed=name.encode(), cfg=CFG)
    torch.cuda.synchronize(); t_prove = time.time() - t0

    t0 = time.time(); acc, msg = verify([claim], proof, seed=name.encode(), cfg=CFG)
    t_verify = time.time() - t0

    aux_rows = claim.y.n_rows(CFG.ELL)
    print(f"  {name:>12}  m={m:5d} k={k:5d} n={n:5d}  aux_rows={aux_rows}  "
          f"prove={t_prove:.2f}s verify={t_verify:.2f}s  "
          f"{'ACCEPT' if acc else 'REJECT'}  ({msg})")
    assert acc, msg
    return t_prove, t_verify


SHAPES = [
    ("attn_qkvo",  2,  4096,  4096),
    ("ffn_gate",   2,  4096, 11008),
    ("ffn_down",   2, 11008,  4096),
]


def _rand_silu_input(L: int, sc: SiluConfig) -> torch.Tensor:
    """Mix of in-range positive, positive saturation, negative-wrap small, and
    negative-wrap saturation — exercises all four mux branches."""
    lookup_bound = sc.b * sc.T_LEN                       # in-range magnitude bound
    sat_bound    = lookup_bound * 4                      # exemplary positive overflow
    q = L // 4
    # Magnitudes fit cleanly in int64; convert to uint64 first, then use gl_neg
    # to get the Goldilocks-wrapped representation (avoids int64 overflow on P).
    in_pos  = torch.randint(0, lookup_bound,         (q,), dtype=torch.int64, device="cuda").to(torch.uint64)
    sat_pos = torch.randint(lookup_bound, sat_bound, (q,), dtype=torch.int64, device="cuda").to(torch.uint64)
    in_neg  = gl_neg(torch.randint(1, lookup_bound,         (q,), dtype=torch.int64, device="cuda").to(torch.uint64))
    sat_neg = gl_neg(torch.randint(lookup_bound, sat_bound, (L - 3*q,), dtype=torch.int64, device="cuda").to(torch.uint64))
    pieces = torch.cat([in_pos, sat_pos, in_neg, sat_neg])
    perm = torch.randperm(pieces.numel(), device="cuda")
    return torch.index_select(pieces, 0, perm)


def run_silu(name: str, L: int, silu_config: SiluConfig) -> Tuple[float, float]:
    """One prove + verify of `L` independent silu calls under `silu_config`,
    using the Tape (which handles witness commitment, table registration,
    and claim ordering)."""
    tape = Tape(CFG, silu_config=silu_config)
    x_data = _rand_silu_input(L, silu_config)
    x = tape.commit("x", x_data, (L,))
    _ = tape.silu(x)
    torch.cuda.synchronize()

    t0 = time.time(); proof = tape.prove(seed=name.encode())
    torch.cuda.synchronize(); t_prove = time.time() - t0

    t0 = time.time(); acc, msg = tape.verify(proof, seed=name.encode())
    t_verify = time.time() - t0

    n_claims = len(tape.claims)
    print(f"  {name:>12}  L={L:5d}  claims={n_claims:3d}  "
          f"prove={t_prove:.2f}s verify={t_verify:.2f}s  "
          f"{'ACCEPT' if acc else 'REJECT'}  ({msg})")
    assert acc, msg
    return t_prove, t_verify


SILU_SHAPES = [
    ("silu_14bit", 4096, SILU_14BIT),
]


def _rand_rmsnorm_input(B: int, d: int, x_max_int: int) -> torch.Tensor:
    """Random non-negative integer inputs in [0, x_max_int). At Llama-shape
    parameters, x_max_int=1000 keeps slacks well within the 64-bit chunk budget
    while exercising a realistic range of magnitudes."""
    return torch.randint(0, x_max_int, (B * d,), dtype=torch.int64, device="cuda").to(torch.uint64)


def run_rmsnorm(name: str, B: int, d: int, s: int, eps_int: int,
                 slack_n_chunks: int, x_max_int: int = 1000) -> Tuple[float, float]:
    tape = Tape(CFG)
    x_data = _rand_rmsnorm_input(B, d, x_max_int)
    x = tape.commit("x", x_data, (B * d,))
    _ = tape.rmsnorm(x, d=d, s=s, eps_int=eps_int,
                      slack_n_chunks=slack_n_chunks)
    torch.cuda.synchronize()

    t0 = time.time(); proof = tape.prove(seed=name.encode())
    torch.cuda.synchronize(); t_prove = time.time() - t0
    t0 = time.time(); acc, msg = tape.verify(proof, seed=name.encode())
    t_verify = time.time() - t0

    print(f"  {name:>12}  B={B:5d} d={d:5d} s=2^{s.bit_length()-1:<2d}  claims={len(tape.claims):2d}  "
          f"prove={t_prove:.2f}s verify={t_verify:.2f}s  "
          f"{'ACCEPT' if acc else 'REJECT'}  ({msg})")
    assert acc, msg
    return t_prove, t_verify


# Llama 2 7B prefill rmsnorm: per-token normalization over hidden_dim=4096.
# At Q3.12, worst-case slack ~2^49 → 4 chunks × 16 bits = 64-bit budget.
RMSNORM_SHAPES = [
    # (name, B,    d,    s,         eps_int, slack_n_chunks)
    ("rms_llama7B", 2048, 4096, 1 << 12, 168,     4),
]


def _rand_softmax_input(B: int, M: int, x_max_int: int) -> torch.Tensor:
    """Non-negative integers in [0, x_max_int). With x_max_int chosen so that
    honest LSE plus the input range stays inside Z_max, every row's z-vector
    lands in the exp-table domain."""
    return torch.randint(0, x_max_int, (B * M,), dtype=torch.int64,
                         device="cuda").to(torch.uint64)


def run_softmax(name: str, B: int, M: int, s_x: int, s_c: int, s_y: int,
                Z_max: int, delta: int = 1,
                x_max_int: int = 2048) -> Tuple[float, float]:
    tape = Tape(CFG)
    x_data = _rand_softmax_input(B, M, x_max_int)
    x = tape.commit("x", x_data, (B * M,))
    _ = tape.softmax(x, M=M, s_x=s_x, s_c=s_c, s_y=s_y,
                     delta=delta, Z_max=Z_max)
    torch.cuda.synchronize()

    t0 = time.time(); proof = tape.prove(seed=name.encode())
    torch.cuda.synchronize(); t_prove = time.time() - t0
    t0 = time.time(); acc, msg = tape.verify(proof, seed=name.encode())
    t_verify = time.time() - t0

    print(f"  {name:>14}  B={B:5d} M={M:5d} s=2^{s_x.bit_length()-1:<2d}  "
          f"Z=2^{Z_max.bit_length()-1:<2d}  claims={len(tape.claims):2d}  "
          f"prove={t_prove:.2f}s verify={t_verify:.2f}s  "
          f"{'ACCEPT' if acc else 'REJECT'}  ({msg})")
    assert acc, msg
    return t_prove, t_verify


# Llama 2 7B attention softmax at Q3.12: paired exp tables sized for the
# z = c2 − x range. Tight bracket pins c2 uniquely (see _softmax_exp_tables).
SOFTMAX_SHAPES = [
    # (name,        B,    M,    s_x,    s_c,    s_y,    Z_max)
    ("sm_llama7B",  512, 2048, 1 << 12, 1 << 12, 1 << 12, 1 << 17),
]


def main():
    print(f"=== production regression (cfg: ELL={CFG.ELL} K_DEG={CFG.K_DEG} "
          f"N_LIG={CFG.N_LIG} T_QUERIES={CFG.T_QUERIES}) ===")
    total_prove = 0.0
    total_verify = 0.0

    print("\n--- matmul shapes ---")
    for name, m, k, n in SHAPES:
        p, v = run_one(name, m, k, n)
        total_prove  += p
        total_verify += v

    print("\n--- silu shapes ---")
    for name, L, sc in SILU_SHAPES:
        p, v = run_silu(name, L, sc)
        total_prove  += p
        total_verify += v

    print("\n--- rmsnorm shapes ---")
    for name, B, d, s, eps_int, n_chunks in RMSNORM_SHAPES:
        p, v = run_rmsnorm(name, B, d, s, eps_int, n_chunks)
        total_prove  += p
        total_verify += v

    print("\n--- softmax shapes ---")
    for name, B, M, s_x, s_c, s_y, Z_max in SOFTMAX_SHAPES:
        p, v = run_softmax(name, B, M, s_x, s_c, s_y, Z_max)
        total_prove  += p
        total_verify += v

    print(f"\n  totals: prove={total_prove:.2f}s verify={total_verify:.2f}s")


if __name__ == "__main__":
    main()
