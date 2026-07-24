"""Differential tests for _RowMap: _build_row_map used to build a plain
{abs_row: (Variable, local_row)} dict with one insertion per row (0.47s of
6.0s profiled prove() time on the toy transformer, 2 calls -- see the
formula-vs-reality investigation this session). Replaced with O(n_vars)
sorted (row_start, row_end, Variable) ranges + bisect, the same restructure
already used elsewhere in core.py for the band index (_build_row_lookup /
bands_overlapping). The only consumer is _gather_rows, which only ever does
`row_map[abs_row]` -- these tests pin that lookup to match the old dict
exactly, including KeyError on an out-of-range row.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # prover/ on path

import random
import pytest
from core import Variable, LigeroConfig, _build_row_map


def _reference_row_map(vars_list, ell, row_offset_start):
    """The exact old per-row dict-building implementation, preserved here
    only, as ground truth."""
    row_map = {}
    abs_offset = row_offset_start
    for v in vars_list:
        n = v.n_rows(ell)
        for r in range(n):
            row_map[abs_offset] = (v, r)
            abs_offset += 1
    return row_map


def _check(lengths, ell, row_offset_start=0):
    vars_list = [Variable(name=f"v{i}", length=n) for i, n in enumerate(lengths)]
    cfg = type("FakeCfg", (), {"ELL": ell})()
    ref = _reference_row_map(vars_list, ell, row_offset_start)
    new = _build_row_map(vars_list, cfg, row_offset_start)

    # Every row the old dict knows about must resolve identically.
    for abs_row, (v, r) in ref.items():
        got_v, got_r = new[abs_row]
        assert got_v is v and got_r == r, (abs_row, (v, r), (got_v, got_r))

    # Out-of-range rows (just past the end, and row_offset_start - 1 if
    # positive) must KeyError exactly like the dict does.
    total_rows = sum(v.n_rows(ell) for v in vars_list)
    past_end = row_offset_start + total_rows
    assert past_end not in ref
    with pytest.raises(KeyError):
        new[past_end]
    if row_offset_start > 0:
        before = row_offset_start - 1
        assert before not in ref
        with pytest.raises(KeyError):
            new[before]
    return vars_list


def test_single_var_exact_multiple():
    _check(lengths=[24], ell=8)


def test_single_var_needs_padding_row():
    _check(lengths=[19], ell=8)  # 19/8 -> 3 rows, last partial (n_rows only cares about count)


def test_multiple_vars_contiguous():
    _check(lengths=[16, 8, 40, 3], ell=8)


def test_nonzero_row_offset_start():
    _check(lengths=[16, 24], ell=8, row_offset_start=1000)


def test_empty_vars_list():
    cfg = type("FakeCfg", (), {"ELL": 8})()
    rm = _build_row_map([], cfg, 0)
    with pytest.raises(KeyError):
        rm[0]


def test_zero_length_variable_is_skipped_like_old_dict():
    # A variable with length 0 -> n_rows(ell) == 0 -> contributes nothing to
    # either the old dict or the new ranges; a variable AFTER it must still
    # land at the same abs_row as if the empty one weren't there.
    v0 = Variable(name="empty", length=0)
    v1 = Variable(name="v1", length=16)
    cfg = type("FakeCfg", (), {"ELL": 8})()
    ref = _reference_row_map([v0, v1], 8, 0)
    new = _build_row_map([v0, v1], cfg, 0)
    for abs_row, (v, r) in ref.items():
        got_v, got_r = new[abs_row]
        assert got_v is v and got_r == r


def test_realistic_mix_matches_reference():
    lengths = [16, 32, 8, 512 * 400, 4, 16 * 3, 1, 999]
    _check(lengths=lengths, ell=512, row_offset_start=3)


def test_row_map_randomized():
    rng = random.Random(20260722)
    for trial in range(50):
        ell = rng.choice([1, 2, 8, 16, 512])
        n_vars = rng.randint(0, 6)
        lengths = [rng.choice([0, rng.randint(1, 3 * ell)]) for _ in range(n_vars)]
        row_offset_start = rng.choice([0, rng.randint(1, 5000)])
        try:
            _check(lengths=lengths, ell=ell, row_offset_start=row_offset_start)
        except AssertionError as e:
            raise AssertionError(
                f"trial {trial} failed: lengths={lengths} ell={ell} "
                f"row_offset_start={row_offset_start}: {e}")
