"""Differential tests for _gather_rows: used a Python `for i, r in items:
out[i, :hi-lo] = data[lo:hi]` loop -- one PyTorch dispatch per requested row
-- to assemble scattered/needed rows for a Variable. Found via profiling a
medium-scale (d=512, d_ff=1536, SEQ=384, 4 layers) run in the formula-vs-
reality investigation this session: ~17% of prove() wall-clock, the same
per-row-dispatch pattern as _iter_message_chunks (fixed earlier in the same
session), just triggered by compute_p_0_streaming's Freivalds-style spot
checks rather than the witness-commit sweep. Replaced with one
index_select per Variable.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # prover/ on path

import random
import torch

from core import Variable, _build_row_map, _gather_rows, _to_device_u64


def _reference_gather_rows(inputs, cfg, row_map, needed_abs_rows):
    """The exact old per-row implementation, preserved here only, as ground
    truth for the new index_select-based version in core.py."""
    ell = cfg.ELL
    out = torch.zeros((len(needed_abs_rows), ell), dtype=torch.uint64, device="cuda")
    by_var = {}
    for i, abs_row in enumerate(needed_abs_rows):
        v, r = row_map[abs_row]
        by_var.setdefault(v, []).append((i, r))
    for v, items in by_var.items():
        data = _to_device_u64(inputs[v]).reshape(-1)
        v_len = data.numel()
        for i, r in items:
            lo = r * ell
            hi = min(lo + ell, v_len)
            out[i, :hi - lo] = data[lo:hi]
    return out


def _rand_u64(n):
    return torch.randint(0, 1 << 62, (n,), dtype=torch.int64, device="cuda").to(torch.uint64)


def _check(vars_lengths, ell, needed_abs_rows, row_offset_start=0, seed=0):
    torch.manual_seed(seed)
    inputs = {}
    vars_list = []
    for i, length in enumerate(vars_lengths):
        v = Variable(name=f"v{i}", length=length)
        inputs[v] = _rand_u64(length)
        vars_list.append(v)
    cfg = type("FakeCfg", (), {"ELL": ell})()
    row_map = _build_row_map(vars_list, cfg, row_offset_start)

    ref = _reference_gather_rows(inputs, cfg, row_map, needed_abs_rows)
    new = _gather_rows(inputs, cfg, row_map, needed_abs_rows)
    assert ref.shape == new.shape
    assert torch.equal(ref, new)
    return vars_list, row_map


def test_single_var_contiguous_rows():
    _check(vars_lengths=[8 * 5], ell=8, needed_abs_rows=[0, 1, 2, 3, 4])


def test_single_var_scattered_rows():
    _check(vars_lengths=[8 * 10], ell=8, needed_abs_rows=[7, 2, 9, 0, 5])


def test_single_var_repeated_rows():
    # Same source row requested for multiple output positions.
    _check(vars_lengths=[8 * 5], ell=8, needed_abs_rows=[2, 2, 2, 0, 4, 4])


def test_var_needs_padding_last_row_requested():
    # length 19 at ell=8 -> 3 rows, last one partial; request it specifically.
    _check(vars_lengths=[19], ell=8, needed_abs_rows=[2, 0, 2, 1])


def test_multiple_vars_interleaved_requests():
    v_lens = [8 * 4, 8 * 3 + 2, 8 * 5]
    # abs rows: v0 -> [0,4), v1 -> [4,8) (last partial), v2 -> [8,13)
    _check(vars_lengths=v_lens, ell=8, needed_abs_rows=[9, 1, 5, 12, 0, 7, 3])


def test_empty_needed_rows():
    _check(vars_lengths=[8 * 3], ell=8, needed_abs_rows=[])


def test_nonzero_row_offset_start():
    _check(vars_lengths=[8 * 6], ell=8, needed_abs_rows=[1003, 1000, 1005],
           row_offset_start=1000)


def test_realistic_mix_matches_reference():
    lengths = [16, 32, 8, 512 * 40, 4, 16 * 3]
    ell = 512
    rng = random.Random(7)
    total_rows = sum((L + ell - 1) // ell for L in lengths)
    needed = [rng.randrange(total_rows) for _ in range(50)]
    _check(vars_lengths=lengths, ell=ell, needed_abs_rows=needed)


def test_gather_rows_randomized():
    rng = random.Random(20260722)
    for trial in range(40):
        ell = rng.choice([1, 4, 8, 512])
        n_vars = rng.randint(1, 4)
        lengths = [rng.randint(1, 3 * ell) for _ in range(n_vars)]
        total_rows = sum((L + ell - 1) // ell for L in lengths)
        n_needed = rng.randint(0, min(30, total_rows * 2))
        needed = [rng.randrange(total_rows) for _ in range(n_needed)] if total_rows else []
        try:
            _check(vars_lengths=lengths, ell=ell, needed_abs_rows=needed, seed=trial)
        except AssertionError as e:
            raise AssertionError(
                f"trial {trial} failed: lengths={lengths} ell={ell} needed={needed}: {e}")
