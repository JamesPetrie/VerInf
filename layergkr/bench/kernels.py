"""Measure the actual INNER LOOPS, not abstract primitives.

Why this file exists. The first machine model priced work as "field muls" and
"field adds" measured in isolated loops. It did not match the stopwatch, and a
least-squares fit over the real workload could not separate the two primitives at
all (it returned negative costs) because in this code they occur in a fixed ratio
-- they are collinear, so "mul" and "add" are not independently identifiable and
therefore are the WRONG UNIT OF ACCOUNT.

The right unit is the loop body the code actually executes. There are only four
that matter, and each is measured here exactly as written in the source:

  encode      acc += v * L[c]            over ELL slots, one reduction at the end
              (rs.encode_row)             -- accumulator grows to 128 + log2(ELL) bits
  combine     out[j] = (out[j] + cw[j]*a) % P
              (rs.linear_combination, projection.commit_projection)
  fold        (f[i] + x*(f[half+i] - f[i])) % P
              (sumcheck round evaluation and folding)
  reduce      (a * b) % P                 -- an isolated reduced multiply

`ns_per_iter` for each is what the cost model should multiply by. The scaling
sweep also answers the question the earlier model got wrong: whether the encode
loop's cost per slot is CONSTANT or grows with the accumulator width.

  .venv/bin/python layergkr/bench/kernels.py
"""
import json
import pathlib
import sys
import time
from typing import Dict, List

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import blake3

from prover.protocol import P

REPS = 5


def _best(fn, n_iter: int) -> float:
    best = float("inf")
    for _ in range(REPS):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best / n_iter


def k_encode(ell: int, rows: int = 200) -> float:
    """rs.encode_row's inner loop, verbatim."""
    vals = [(i * 2654435761 + 12345) % P for i in range(ell)]
    L = [(i * 40503 + 7) % P for i in range(ell)]

    def run():
        for _ in range(rows):
            acc = 0
            for c, v in enumerate(vals):
                if v:
                    acc += v * L[c]
            acc % P
    return _best(run, rows * ell)


def k_encode_skip(ell: int, rows: int = 200) -> float:
    """The SAME loop with all-zero values: `if v:` fails every time, so this is
    the pure scan cost -- loop, enumerate, truth test -- with no arithmetic.

    This is the term the first model missed entirely. rs.encode_row is
    sparsity-aware, and most rows it encodes are mostly padding (a LogUp
    multiplicity row is one value in an ELL-wide message), so charging a dense
    ELL*N_LIG multiply-accumulate over-prices encoding several-fold."""
    vals = [0] * ell
    L = [(i * 40503 + 7) % P for i in range(ell)]

    def run():
        for _ in range(rows):
            acc = 0
            for c, v in enumerate(vals):
                if v:
                    acc += v * L[c]
            acc % P
    return _best(run, rows * ell)


def k_combine(n: int, rows: int = 200) -> float:
    """rs.linear_combination's inner loop."""
    out = [(i * 7919) % P for i in range(n)]
    cw = [(i * 104729 + 13) % P for i in range(n)]
    a = 0x9E3779B97F4A7C15 % P

    def run():
        for _ in range(rows):
            for j in range(n):
                out[j] = (out[j] + cw[j] * a) % P
    return _best(run, rows * n)


def k_fold(n: int, rows: int = 200) -> float:
    """sumcheck's fold / round-evaluation body."""
    f = [(i * 2246822519) % P for i in range(2 * n)]
    x = 0x1234_5678_9ABC_DEF0 % P

    def run():
        for _ in range(rows):
            for i in range(n):
                (f[i] + x * (f[n + i] - f[i])) % P
    return _best(run, rows * n)


def k_transpose(rows: int, n: int = 256, reps: int = 20) -> float:
    """rs.Commit's column build: [[cw[j] for cw in codewords] for j in range(N)].
    Pure list work, no arithmetic -- and completely unmodelled at first, which is
    part of why the model under-predicted the bigger instances."""
    cw = [[(i * 7 + j) % P for j in range(n)] for i in range(rows)]

    def run():
        for _ in range(reps):
            [[c[j] for c in cw] for j in range(n)]
    return _best(run, reps * rows * n)


def k_hash_call(n: int = 20_000, size: int = 256) -> float:
    """Per-call cost of hashing one column: pack + BLAKE3 on a small buffer.
    Throughput in GB/s is the wrong unit here -- the buffers are tiny, so the
    call overhead dominates."""
    col = [(i * 2654435761) % P for i in range(size // 8)]

    def run():
        for _ in range(n):
            blake3.blake3(b"".join(int(v).to_bytes(8, "little") for v in col)).digest()
    return _best(run, n)


def k_fold_list(n: int, rows: int = 200) -> float:
    """sumcheck._fold AS WRITTEN: a list comprehension, which allocates a fresh
    list of `n` elements on top of the arithmetic. The first version of this file
    measured the bare expression in a `for` loop and missed the allocation -- one
    of the two reasons the model under-predicted the big instances by 2x."""
    f = [(i * 2246822519) % P for i in range(2 * n)]
    r = 0x1234_5678_9ABC_DEF0 % P

    def run():
        for _ in range(rows):
            [(f[i] + r * (f[n + i] - f[i])) % P for i in range(n)]
    return _best(run, rows * n)


def k_round_eval(n: int, n_fac: int = 2, rows: int = 60) -> float:
    """sumcheck.prove_terms' round-evaluation body, as written: for each factor
    it folds AND multiplies into a running product -- TWO reductions and an extra
    multiply per factor, inside a triple-nested loop. Charging it at the plain
    fold rate under-prices it. Returns ns per (position x factor)."""
    fs = [[(i * 2246822519 + k) % P for i in range(2 * n)] for k in range(n_fac)]
    x, c = 3, 0x9E3779B97F4A7C15 % P

    def run():
        for _ in range(rows):
            acc = 0
            for i in range(n):
                prod = c
                for f in fs:
                    prod = prod * ((f[i] + x * (f[n + i] - f[i])) % P) % P
                acc += prod
    return _best(run, rows * n * n_fac)


def k_pack(n: int, reps: int = 200) -> float:
    """protocol.pack_column, as written: one `int.to_bytes(8)` per value inside a
    generator, then a join. A profile of a real prove showed this to be the single
    largest cost -- ~6.3 s of 19.8 s -- and the model charged it only per HASH
    CALL. It is per VALUE."""
    col = [(i * 2654435761) % P for i in range(n)]

    def run():
        for _ in range(reps):
            b"".join(int(v).to_bytes(8, "little") for v in col)
    return _best(run, reps * n)


def k_reduce(n: int = 200_000) -> float:
    a, b = 0x1234567890ABCDEF, 0xFEDCBA0987654321

    def run():
        for _ in range(n):
            (a * b) % P
    return _best(run, n)


def k_gpu_encode(ell: int, n: int, rows: int, reps: int = 20) -> float:
    """The GPU encode path: gl_matmul of (rows x ELL) by (ELL x N), i.e. the same
    work rs.encode_row does slot by slot. Returns ns per SLOT SCANNED
    (rows*ELL*N), so it is directly comparable with the CPython scan rate.

    A GPU is a different machine, so it needs its own measured card -- the same
    counts priced with CPython rates over-predict by more than an order of
    magnitude, which is what the first near-real run showed."""
    try:
        import torch
        from prover import cuda_primitives as cp
        if not torch.cuda.is_available():
            return float("nan")
    except Exception:
        return float("nan")
    A = torch.randint(0, 1 << 62, (rows, ell), dtype=torch.int64).to(torch.uint64).cuda()
    B = torch.randint(0, 1 << 62, (ell, n), dtype=torch.int64).to(torch.uint64).cuda()
    cp.gl_matmul(A, B); torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        cp.gl_matmul(A, B)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps / (rows * ell * n)


def k_gpu_marshal(rows: int, n: int, reps: int = 10) -> float:
    """Device->host transfer plus the Python list conversion the backend does:
    `[[int(v) for v in row] for row in C.cpu().tolist()]`.

    Once encoding moves to the GPU this becomes the dominant term, and it was
    unmodelled -- the model under-predicted by 2-3x until it was added. A
    production prover keeps the data on the device and would not pay it; this
    implementation does, so the model must count it."""
    try:
        import torch
        if not torch.cuda.is_available():
            return float("nan")
    except Exception:
        return float("nan")
    C = torch.randint(0, 1 << 62, (rows, n), dtype=torch.int64).to(torch.uint64).cuda()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        [[int(v) for v in row] for row in C.cpu().tolist()]
    return (time.perf_counter() - t0) / reps / (rows * n)


def main() -> int:
    print("inner-loop cost, ns per iteration (best of %d)\n" % REPS)

    print("encode: does cost per slot depend on ELL (accumulator width)?")
    enc: Dict[int, float] = {}
    for ell in (8, 16, 32, 64, 128, 256, 512):
        enc[ell] = k_encode(ell) * 1e9
        print(f"  ELL={ell:4d}   {enc[ell]:7.2f} ns/slot")
    skip: Dict[int, float] = {}
    for ell in (8, 64, 512):
        skip[ell] = k_encode_skip(ell) * 1e9
    print("  scan-only cost (all slots zero -> `if v:` fails, no arithmetic):")
    for ell, v in skip.items():
        print(f"    ELL={ell:4d}   {v:7.2f} ns/slot")
    print(f"  -> a nonzero slot costs ~{enc[64]:.0f} ns, a zero slot ~{skip[64]:.0f} ns."
          f" Encoding must be")
    print(f"     priced as scan(ELL) + mac(nonzeros), not as a dense ELL*N product.\n")
    drift = enc[512] / enc[8]
    print(f"  -> {drift:.2f}x from ELL=8 to ELL=512: the accumulator widens, so a "
          f"CONSTANT")
    print(f"     ns-per-slot is wrong; the model must price encode as a function "
          f"of ELL.\n")

    print("combine / fold / reduce: cost per iteration")
    comb = {n: k_combine(n) * 1e9 for n in (64, 256, 1024)}
    fold = {n: k_fold(n) * 1e9 for n in (64, 256, 1024)}
    red = k_reduce() * 1e9
    for n in (64, 256, 1024):
        print(f"  combine n={n:5d}  {comb[n]:7.2f} ns/iter     "
              f"fold n={n:5d}  {fold[n]:7.2f} ns/iter")
    print(f"  reduce (isolated a*b % P)   {red:7.2f} ns/iter")
    pk = {n: k_pack(n) * 1e9 for n in (32, 128, 512)}
    print("  -- column packing (per VALUE, profile says this dominates) --")
    for n_, v in pk.items():
        print(f"  pack rows={n_:4d}   {v:7.2f} ns/value")
    fl = {n: k_fold_list(n) * 1e9 for n in (256, 1024, 4096)}
    ev = {n: k_round_eval(n) * 1e9 for n in (256, 1024, 4096)}
    print("  -- the two sumcheck bodies, measured as written --")
    for n in (256, 1024, 4096):
        print(f"  _fold list-comp n={n:5d}  {fl[n]:7.2f} ns/elem     "
              f"round-eval n={n:5d}  {ev[n]:7.2f} ns/(pos*factor)")
    xp = {r: k_transpose(r) * 1e9 for r in (8, 32, 128)}
    hc = k_hash_call() * 1e9
    for r, v in xp.items():
        print(f"  transpose rows={r:4d}         {v:7.2f} ns/element")
    print(f"  hash one column (pack+blake3) {hc:7.2f} ns/call "
          f"-- call overhead, not GB/s")

    print("\nGPU encode (production gl_matmul), ns per slot:")
    gpu = {}
    for rows, ell, n in ((64, 256, 1024), (128, 512, 2048), (256, 512, 4096)):
        v = k_gpu_encode(ell, n, rows) * 1e9
        gpu[f"{rows}x{ell}x{n}"] = v
        print(f"  rows={rows:4d} ELL={ell:4d} N={n:5d}   {v:8.4f} ns/slot")
    if gpu:
        g = min(gpu.values())
        print(f"  -> ~{enc[64]/g:.0f}x faster per slot than the CPython loop. The same")
        print(f"     COUNTS priced with the wrong card over-predict by that factor;")
        print(f"     the counts are backend-independent, the rates are not.")

    mar = {}
    for rows, n in ((64, 1024), (256, 2048)):
        mar[f"{rows}x{n}"] = k_gpu_marshal(rows, n) * 1e9
    print("\nGPU->host transfer + list marshalling, ns per element:")
    for k_, v in mar.items():
        print(f"  {k_:>12}   {v:8.2f} ns/elem")

    out = {"encode_ns_per_slot": enc, "encode_scan_ns_per_slot": skip,
           "gpu_marshal_ns_per_elem": mar,
           "fold_list_ns_per_elem": fl, "round_eval_ns_per_unit": ev,
           "pack_ns_per_value": pk,
           "gpu_encode_ns_per_slot": gpu,
           "combine_ns_per_iter": comb, "fold_ns_per_iter": fold, "reduce_ns": red,
           "transpose_ns_per_elem": xp, "hash_call_ns": hc}
    path = pathlib.Path(__file__).parent / "kernels.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwritten to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
