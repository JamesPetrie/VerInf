"""Gates for the GPU backend (gpu.py).

The only thing that matters here: the GPU path must produce BIT-IDENTICAL
codewords to the CPU path. A faster backend that computes something slightly
different would silently invalidate every proof and every measurement, so the
backend refuses to enable itself until this passes, and these tests pin that.

Skipped cleanly when there is no CUDA.

Run:  .venv/bin/python layergkr/tests/run_tests.py test_gpu
"""
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from layergkr import gpu, rs
from prover.protocol import P

CONFIGS = [rs.Config(ELL=16, K_DEG=32, N_LIG=64, T_QUERIES=4),
           rs.Config(ELL=64, K_DEG=128, N_LIG=256, T_QUERIES=8),
           rs.Config(ELL=256, K_DEG=512, N_LIG=1024, T_QUERIES=8)]


def _skip() -> bool:
    if not gpu.available():
        print("    (no CUDA — skipped)")
        return True
    return False


def test_encode_is_bit_identical():
    if _skip():
        return
    for cfg in CONFIGS:
        err = gpu.selftest(cfg, n_rows=6)
        assert err is None, f"ELL={cfg.ELL} N={cfg.N_LIG}: {err}"


def test_sparse_rows_are_bit_identical():
    """Most real rows are mostly padding; the matmul path must handle the zeros
    the CPU path skips."""
    if _skip():
        return
    cfg = CONFIGS[1]
    r = random.Random(3)
    msgs = [[r.randrange(P) if j < 3 else 0 for j in range(cfg.ELL)] for _ in range(5)]
    got = gpu.encode_batch(cfg, msgs)
    for i, m in enumerate(msgs):
        assert got[i] == rs.encode_row(cfg, m), f"row {i}"


def test_commit_root_matches_cpu_backend():
    """End to end: the same messages must give the same Merkle root whichever
    backend encoded them."""
    if _skip():
        return
    cfg = CONFIGS[2]
    r = random.Random(9)
    msgs = [[r.randrange(P) for _ in range(cfg.ELL)] for _ in range(8)]
    gpu_commit = rs.Commit(cfg, gpu.encode_batch(cfg, msgs))
    cpu_commit = rs.Commit(cfg, [rs.encode_row(cfg, m) for m in msgs])
    assert gpu_commit.root == cpu_commit.root


def test_linear_combination_is_bit_identical():
    if _skip():
        return
    cfg = CONFIGS[1]
    r = random.Random(5)
    cw = [rs.encode_row(cfg, [r.randrange(P) for _ in range(cfg.ELL)]) for _ in range(6)]
    co = [r.randrange(P) for _ in range(6)]
    assert rs.linear_combination(cfg, cw, co) == gpu.linear_combination(cfg, cw, co)


def test_backend_is_gated_on_the_selftest():
    """rs._gpu_ok must not enable a backend that has not proved itself."""
    if _skip():
        return
    cfg = CONFIGS[1]
    assert rs._gpu_ok(cfg) is True
    assert (cfg.ELL, cfg.K_DEG, cfg.N_LIG) in rs._GPU_OK


def test_chunked_encode_is_bit_identical_and_passes_the_grid_limit():
    """A CUDA grid dimension is capped at 65535 and the batched NTT kernels use
    one block-row per message, so an unchunked encode simply fails past that --
    measured exactly: 65535 encodes, 65536 raises `invalid configuration
    argument`. LogUp commits one RS row per lookup QUERY, so the cap is reached
    around d=256, S=8. Chunking is only legitimate if it changes nothing, which
    is what this pins."""
    if _skip():
        return
    import torch

    cfg = rs.Config(ELL=64, K_DEG=128, N_LIG=256, T_QUERIES=8)
    g = torch.Generator(device="cuda").manual_seed(3)
    msgs = torch.randint(0, 1 << 62, (300, cfg.ELL), generator=g,
                         dtype=torch.int64, device="cuda").view(torch.uint64)
    ref = gpu.encode_batch_ntt(cfg, msgs).clone()
    saved = gpu.MAX_GRID_ROWS
    try:
        gpu.MAX_GRID_ROWS = 37              # several uneven chunks
        got = gpu.encode_batch_ntt(cfg, msgs)
        assert torch.equal(ref, got), "chunked encode differs from unchunked"
    finally:
        gpu.MAX_GRID_ROWS = saved

    # and a row count past the hardware cap now completes at all
    wide = rs.Config(ELL=256, K_DEG=512, N_LIG=1024, T_QUERIES=8)
    big = torch.zeros((gpu.MAX_GRID_ROWS + 65, wide.ELL), dtype=torch.uint64,
                      device="cuda")
    out = gpu.encode_batch_ntt(wide, big)
    assert int(out.shape[0]) == gpu.MAX_GRID_ROWS + 65
