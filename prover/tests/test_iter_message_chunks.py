"""Differential tests for the _iter_message_chunks vectorization and the
_band_key field-name cache (both in core.py) -- added while root-causing why
the toy-transformer prove() was ~28x slower than the notebook's identity-
floor cost formula predicted. cProfile found _iter_message_chunks's old
per-row Python loop was ~69% of prove() wall-clock (see
analysis/verification-parameter-analysis.ipynb investigation, this session);
_band_key's uncached dataclasses.fields() call was a second, smaller instance
of the same pattern. Both were rewritten to avoid per-row/per-call Python
overhead. These tests pin the new code to the OLD code's exact output on a
battery of chunk-boundary edge cases, so the optimization can't silently
change what gets committed.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # prover/ on path

import random
import torch
from dataclasses import dataclass, fields
from typing import Optional

import core
from core import Variable, LigeroConfig, _iter_message_chunks, _band_key


# ---------------------------------------------------------------------------
# Reference implementation: the exact old per-row loop, preserved here only
# (not in production code) as the ground truth the new version must match.
# ---------------------------------------------------------------------------
def _iter_message_chunks_reference(vars_list, inputs, cfg, row_offset_start, chunk_size):
    if not vars_list:
        return
    chunk = torch.zeros((chunk_size, cfg.ELL), dtype=torch.uint64, device="cuda")
    chunk_row = 0
    abs_offset = row_offset_start
    for v in vars_list:
        data = core._to_device_u64(core._fetch(inputs, v)).reshape(-1)
        v_len = data.numel()
        assert v_len == v.length
        v_rows = v.n_rows(cfg.ELL)
        for r in range(v_rows):
            lo = r * cfg.ELL
            hi = min(lo + cfg.ELL, v_len)
            chunk[chunk_row, :hi - lo] = data[lo:hi]
            if hi - lo < cfg.ELL:
                chunk[chunk_row, hi - lo:].zero_()
            chunk_row += 1
            if chunk_row == chunk_size:
                yield abs_offset, chunk
                chunk = torch.zeros((chunk_size, cfg.ELL), dtype=torch.uint64, device="cuda")
                chunk_row = 0
                abs_offset += chunk_size
    if chunk_row > 0:
        yield abs_offset, chunk[:chunk_row]


def _rand_u64(n):
    return torch.randint(0, 1 << 62, (n,), dtype=torch.int64, device="cuda").to(torch.uint64)


def _check(vars_lengths, ELL, chunk_size, row_offset_start=0, seed=0):
    """Build variables of the given lengths with random data, run both
    implementations, and assert identical (abs_offset, chunk) sequences."""
    torch.manual_seed(seed)
    inputs = {}
    vars_list = []
    for i, length in enumerate(vars_lengths):
        v = Variable(name=f"v{i}", length=length)
        inputs[v] = _rand_u64(length)
        vars_list.append(v)
    cfg_ell = ELL  # only cfg.ELL is read by _iter_message_chunks
    cfg = type("FakeCfg", (), {"ELL": cfg_ell})()

    ref = list(_iter_message_chunks_reference(vars_list, inputs, cfg, row_offset_start, chunk_size))
    new = list(_iter_message_chunks(vars_list, inputs, cfg, row_offset_start, chunk_size))

    assert len(ref) == len(new), f"chunk count differs: ref={len(ref)} new={len(new)}"
    for (ro, rc), (no, nc) in zip(ref, new):
        assert ro == no, f"abs_offset differs: ref={ro} new={no}"
        assert rc.shape == nc.shape, f"shape differs at offset {ro}: ref={rc.shape} new={nc.shape}"
        assert torch.equal(rc, nc), f"values differ at offset {ro}"
    return len(ref)


def test_single_var_exact_multiple_of_ell():
    n = _check(vars_lengths=[3 * 8], ELL=8, chunk_size=2)
    assert n == 2, n  # 3 rows / chunk_size 2 -> chunks of 2, then 1


def test_single_var_needs_padding():
    # length 10 at ELL=8 -> 2 rows, second row padded with 6 zero slots
    _check(vars_lengths=[10], ELL=8, chunk_size=4)


def test_var_spans_many_chunks():
    # 37 rows at chunk_size=5 -> exercises the multi-chunk `while` loop
    # inside a single variable.
    _check(vars_lengths=[37 * 16], ELL=16, chunk_size=5)


def test_multiple_vars_carry_across_chunk_boundary():
    # Row counts of 3, 5, 2 at chunk_size=4: chunks don't align to variable
    # boundaries, so a chunk is filled by the tail of one var + head of next.
    _check(vars_lengths=[3 * 8, 5 * 8, 2 * 8], ELL=8, chunk_size=4)


def test_multiple_vars_with_padding_and_carry():
    # Non-multiple-of-ELL lengths mixed with carry-across-boundary.
    _check(vars_lengths=[8 + 3, 8 * 2 + 5, 8 * 4], ELL=8, chunk_size=3)


def test_chunk_size_one():
    _check(vars_lengths=[8 * 5], ELL=8, chunk_size=1)


def test_chunk_size_larger_than_total_rows():
    _check(vars_lengths=[8 * 3], ELL=8, chunk_size=100)


def test_empty_vars_list():
    cfg = type("FakeCfg", (), {"ELL": 8})()
    out = list(_iter_message_chunks([], {}, cfg, 0, 4))
    assert out == []


def test_nonzero_row_offset_start():
    _check(vars_lengths=[8 * 6], ELL=8, chunk_size=4, row_offset_start=1000)


def test_many_small_vars_realistic_mix():
    # A shape closer to the real claim list: several small vars plus one
    # large one (like a lookup-table multiplicities array), at the toy
    # run's actual ELL.
    _check(vars_lengths=[16, 32, 8, 512 * 400, 4, 16 * 3], ELL=512, chunk_size=64)


# ---------------------------------------------------------------------------
# _band_key: the cache must not change which field is picked, or leak a
# lookup across unrelated types.
# ---------------------------------------------------------------------------
@dataclass
class _FakePktWithRowStart:
    base: int
    x_row_start: int
    other: int = 0


@dataclass
class _FakePktNoRowStart:
    base: int
    other: int = 0


@dataclass
class _FakePktDifferentRowStartName:
    base: int
    y_row_start: int


def test_band_key_finds_row_start_field():
    pkt = _FakePktWithRowStart(base=7, x_row_start=42, other=99)
    assert _band_key(pkt) == (_FakePktWithRowStart, 7, 42)


def test_band_key_no_row_start_field():
    pkt = _FakePktNoRowStart(base=3, other=1)
    assert _band_key(pkt) == (_FakePktNoRowStart, 3, None)


def test_band_key_caching_does_not_cross_contaminate_types():
    # Interleave calls across three distinct types (some with, some without
    # a *row_start field) to make sure the per-type cache keys correctly.
    a = _FakePktWithRowStart(base=1, x_row_start=10)
    b = _FakePktNoRowStart(base=2)
    c = _FakePktDifferentRowStartName(base=3, y_row_start=30)
    for _ in range(3):  # repeat to exercise both the cold and warm cache paths
        assert _band_key(a) == (_FakePktWithRowStart, 1, 10)
        assert _band_key(b) == (_FakePktNoRowStart, 2, None)
        assert _band_key(c) == (_FakePktDifferentRowStartName, 3, 30)


def test_band_key_matches_uncached_reference():
    """Ground truth: re-derive the field name the slow way (fields() every
    call, no cache) and compare, against a real packet type from packets.py
    -- catches a cache bug that happens to agree on the fakes above but not
    on the real dataclasses (real ones use slots=True/frozen=True, unlike
    the plain fakes above)."""
    import packets

    def _uncached_band_key(pkt):
        rs = None
        for f in fields(pkt):
            if f.name.endswith("row_start"):
                rs = getattr(pkt, f.name)
                break
        return (type(pkt), getattr(pkt, "base", None), rs)

    real_pkt = packets.L2_IdentityScalar(base=5, var_row_start=17, L=100, coef=3)
    assert _band_key(real_pkt) == _uncached_band_key(real_pkt) == (packets.L2_IdentityScalar, 5, 17)


# ---------------------------------------------------------------------------
# Randomized: the hand-picked edge cases above are chosen to be adversarial,
# but a fixed set can't cover the interaction of (n_vars, each length, ELL,
# chunk_size, offset) that a real claim list produces. Fixed seed -> a
# failure is exactly reproducible by re-running (print the seed on failure).
# ---------------------------------------------------------------------------
def test_iter_message_chunks_randomized():
    rng = random.Random(20260722)
    n_trials = 60
    for trial in range(n_trials):
        ell = rng.choice([1, 2, 3, 8, 16, 512])
        n_vars = rng.randint(0, 5)
        # Lengths deliberately span sub-ELL, exact-multiple, and many-row cases.
        lengths = [rng.choice([
            rng.randint(1, ell),                    # sub-ELL (needs padding, or exact if ==ell)
            ell * rng.randint(1, 4),                 # exact multiple, 1-4 rows
            ell * rng.randint(1, 4) + rng.randint(1, ell - 1) if ell > 1 else ell,  # multi-row + remainder
        ]) for _ in range(n_vars)]
        chunk_size = rng.randint(1, 20)
        row_offset_start = rng.choice([0, rng.randint(1, 10_000)])
        try:
            _check(vars_lengths=lengths, ELL=ell, chunk_size=chunk_size,
                   row_offset_start=row_offset_start, seed=trial)
        except AssertionError as e:
            raise AssertionError(
                f"trial {trial} failed: lengths={lengths} ELL={ell} "
                f"chunk_size={chunk_size} row_offset_start={row_offset_start}: {e}")
