"""Differential tests for the _PktRange fast path: _index_bands folding a
whole (variable, family) row range in O(1) instead of walking it row by row,
and table_settlement_compile emitting ranges instead of one Python object
per row. Second round of the same investigation as
test_iter_message_chunks.py -- re-profiling after that fix found
table_settlement_compile (the lookup-table settlement compile, called once
per table but each call looping over that table's full row count) was the
new #1 cost, same per-row-Python-object pattern.

Correctness must be exact: _index_bands's bands drive which witness rows
get folded into which linear-test contribution, so a range that's off by
one row silently drops or duplicates a constraint. These tests diff the new
range-based path against the old row-by-row path on both an isolated
_index_bands harness and a real TableSettlement.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # prover/ on path

import torch
from dataclasses import dataclass

import random

import core
from core import (
    Variable, LigeroConfig, Table, TableSettlement, table_settlement_compile,
    _PktRange, _expand_row_pkts, _band_key, P,
)


# ---------------------------------------------------------------------------
# Isolated _index_bands harness: the real method, bound to a bare namespace
# object instead of a full _StreamingPackets (whose __init__ does far more
# than band-indexing).
# ---------------------------------------------------------------------------
class _FakeIndexer:
    def __init__(self):
        self.bands = []
        self._band_seen = {}
    _index_bands = core._StreamingPackets._index_bands


def _bands_as_set(bands):
    """Order-insensitive view: (packet content key, row_start, row_end)."""
    return {(_band_key(pkt), rs, re) for pkt, rs, re in bands}


@dataclass
class _FakePkt:
    base: int
    var_row_start: int
    tag: str = "x"   # lets us make "different" packets that still share a band key on purpose


def test_index_bands_single_range_matches_per_row():
    pkt = _FakePkt(base=1, var_row_start=100)
    old = _FakeIndexer()
    old._index_bands([(100 + i, pkt) for i in range(50)])
    new = _FakeIndexer()
    new._index_bands([_PktRange(pkt, 100, 150)])
    assert _bands_as_set(old.bands) == _bands_as_set(new.bands)
    assert new.bands == [[pkt, 100, 150]]


def test_index_bands_range_mixed_with_plain_tuples():
    pkt_a = _FakePkt(base=1, var_row_start=0)
    pkt_b = _FakePkt(base=2, var_row_start=200)
    old = _FakeIndexer()
    old._index_bands([(i, pkt_a) for i in range(0, 20)] + [(200 + i, pkt_b) for i in range(5)])
    new = _FakeIndexer()
    new._index_bands([_PktRange(pkt_a, 0, 20), (200, pkt_b), (201, pkt_b), (202, pkt_b),
                       (203, pkt_b), (204, pkt_b)])
    assert _bands_as_set(old.bands) == _bands_as_set(new.bands)


def test_index_bands_two_ranges_same_key_get_merged():
    # Same (type, base, row_start-field) key emitted as two separate range
    # calls (e.g. two claims referencing the same table) must merge into one
    # band spanning the union, exactly like two separate row-by-row passes
    # over a disjoint-then-adjacent range would.
    pkt = _FakePkt(base=9, var_row_start=0)
    old = _FakeIndexer()
    old._index_bands([(i, pkt) for i in range(0, 10)])
    old._index_bands([(i, pkt) for i in range(10, 25)])
    new = _FakeIndexer()
    new._index_bands([_PktRange(pkt, 0, 10)])
    new._index_bands([_PktRange(pkt, 10, 25)])
    assert _bands_as_set(old.bands) == _bands_as_set(new.bands) == {((_FakePkt, 9, 0), 0, 25)}


def test_index_bands_disjoint_ranges_same_key_span_the_gap():
    # _band_key only sees (type, base, row_start-field) -- two ranges with a
    # gap between them still collapse to one band spanning min..max, same as
    # the old row-by-row walk would (it never checked contiguity either).
    pkt = _FakePkt(base=3, var_row_start=0)
    old = _FakeIndexer()
    old._index_bands([(i, pkt) for i in range(0, 5)] + [(i, pkt) for i in range(50, 55)])
    new = _FakeIndexer()
    new._index_bands([_PktRange(pkt, 0, 5), _PktRange(pkt, 50, 55)])
    assert _bands_as_set(old.bands) == _bands_as_set(new.bands)


def test_expand_row_pkts_range_matches_flat():
    pkt = _FakePkt(base=1, var_row_start=7)
    expanded = list(_expand_row_pkts([_PktRange(pkt, 7, 12)]))
    assert expanded == [(7, pkt), (8, pkt), (9, pkt), (10, pkt), (11, pkt)]


def test_expand_row_pkts_passthrough_for_plain_tuples():
    pkt = _FakePkt(base=1, var_row_start=0)
    items = [(0, pkt), (1, pkt), (2, pkt)]
    assert list(_expand_row_pkts(items)) == items


def test_expand_row_pkts_mixed():
    pkt_a, pkt_b = _FakePkt(base=1, var_row_start=0), _FakePkt(base=2, var_row_start=5)
    out = list(_expand_row_pkts([_PktRange(pkt_a, 0, 2), (5, pkt_b), _PktRange(pkt_a, 10, 11)]))
    assert out == [(0, pkt_a), (1, pkt_a), (5, pkt_b), (10, pkt_a)]


# ---------------------------------------------------------------------------
# table_settlement_compile: reference (old, per-row) vs current (range-based),
# through _index_bands, on a real Table/TableSettlement.
# ---------------------------------------------------------------------------
def _table_settlement_compile_reference(c, _ch, cfg, base):
    """The exact old per-row implementation, preserved here only, as ground
    truth for the new range-based version in core.py."""
    from core import (L2_PerSlotVector, L2_IdentityScalar, L2_StrideManyToOneScalar,
                       gl_add, gl_mul, gl_sub)
    table = c.table
    ell = cfg.ELL
    T_LEN = table.T.numel()
    alpha = table.alpha
    neg1 = (P - 1) % P
    if table.T_Y is not None:
        beta_vec = torch.full((T_LEN,), table.beta, dtype=torch.uint64, device="cuda")
        v = gl_add(table.T, gl_mul(beta_vec, table.T_Y))
    else:
        v = table.T
    alpha_vec = torch.full((T_LEN,), alpha, dtype=torch.uint64, device="cuda")
    w_coef_vec = gl_sub(alpha_vec, v).contiguous()
    sum_cid = base + T_LEN
    row_pkts = []
    for row_off in range(table.w_var.n_rows(ell)):
        row_pkts.append((table.w_var.row_start + row_off,
                          L2_PerSlotVector(base=base, var_row_start=table.w_var.row_start,
                                            L=T_LEN, coef_vec=w_coef_vec)))
    for row_off in range(table.mult_var.n_rows(ell)):
        row_pkts.append((table.mult_var.row_start + row_off,
                          L2_IdentityScalar(base=base, var_row_start=table.mult_var.row_start,
                                             L=T_LEN, coef=neg1)))
    for z in table.z_vars:
        for row_off in range(z.n_rows(ell)):
            row_pkts.append((z.row_start + row_off,
                              L2_StrideManyToOneScalar(base=sum_cid, var_row_start=z.row_start,
                                                        L=z.length, stride=z.length, coef=1)))
    for row_off in range(table.w_var.n_rows(ell)):
        row_pkts.append((table.w_var.row_start + row_off,
                          L2_StrideManyToOneScalar(base=sum_cid, var_row_start=table.w_var.row_start,
                                                    L=T_LEN, stride=T_LEN, coef=neg1)))
    return row_pkts, [], T_LEN + 1, None


def _make_table_settlement(T_LEN=37, n_z=2, ell=8, paired=False):
    """A real Table + TableSettlement + row_start-assigned Variables, small
    enough to run fast but with T_LEN not a multiple of ell (exercises the
    partial last row) and n_z z_vars of different lengths (exercises the
    per-z range in the sum-identity loop)."""
    torch.manual_seed(0)
    T = torch.randint(0, 1 << 62, (T_LEN,), dtype=torch.int64, device="cuda").to(torch.uint64)
    mult_var = Variable(name="mult", length=T_LEN, row_start=0)
    w_var = Variable(name="w", length=T_LEN, row_start=mult_var.n_rows(ell))
    z_vars = []
    row_cursor = w_var.row_start + w_var.n_rows(ell)
    for i in range(n_z):
        zlen = T_LEN - i * 3  # different lengths -> different row counts
        zv = Variable(name=f"z{i}", length=zlen, row_start=row_cursor)
        row_cursor += zv.n_rows(ell)
        z_vars.append(zv)
    T_Y = None
    beta = 0
    if paired:
        T_Y = torch.randint(0, 1 << 62, (T_LEN,), dtype=torch.int64, device="cuda").to(torch.uint64)
        beta = 123
    table = Table(name="t", T=T, mult_var=mult_var, w_var=w_var, T_Y=T_Y,
                  alpha=456, beta=beta, z_vars=z_vars)
    return TableSettlement(table=table)


def _compile_and_index(compile_fn, settlement, cfg, base=1000):
    row_pkts, quads, n_added, b_chunk = compile_fn(settlement, None, cfg, base)
    idx = _FakeIndexer()
    idx._index_bands(row_pkts)
    return idx.bands, quads, n_added, b_chunk


def test_table_settlement_compile_matches_reference_range_table():
    cfg = LigeroConfig(ELL=8, K_DEG=16, N_LIG=64, T_QUERIES=4)
    settlement = _make_table_settlement(T_LEN=37, n_z=2, ell=cfg.ELL, paired=False)
    old_bands, old_quads, old_n, old_b = _compile_and_index(
        _table_settlement_compile_reference, settlement, cfg)
    new_bands, new_quads, new_n, new_b = _compile_and_index(
        table_settlement_compile, settlement, cfg)
    assert old_quads == new_quads == []
    assert old_n == new_n
    assert old_b == new_b
    assert _bands_as_set(old_bands) == _bands_as_set(new_bands)


def test_table_settlement_compile_matches_reference_paired_table():
    cfg = LigeroConfig(ELL=8, K_DEG=16, N_LIG=64, T_QUERIES=4)
    settlement = _make_table_settlement(T_LEN=41, n_z=3, ell=cfg.ELL, paired=True)
    old_bands, *_ = _compile_and_index(_table_settlement_compile_reference, settlement, cfg)
    new_bands, *_ = _compile_and_index(table_settlement_compile, settlement, cfg)
    assert _bands_as_set(old_bands) == _bands_as_set(new_bands)


def test_table_settlement_compile_exact_multiple_of_ell():
    # T_LEN a clean multiple of ELL -> no partial last row anywhere.
    cfg = LigeroConfig(ELL=8, K_DEG=16, N_LIG=64, T_QUERIES=4)
    settlement = _make_table_settlement(T_LEN=32, n_z=1, ell=cfg.ELL, paired=False)
    old_bands, *_ = _compile_and_index(_table_settlement_compile_reference, settlement, cfg)
    new_bands, *_ = _compile_and_index(table_settlement_compile, settlement, cfg)
    assert _bands_as_set(old_bands) == _bands_as_set(new_bands)


def test_table_settlement_compile_expand_matches_reference_flat_list():
    """The eager-path compatibility shim: expanding the new range-based
    output must equal the old flat per-row list exactly, element for
    element (same order), since _compile_with_chs indexes per_row[r] by
    position and must land identically."""
    cfg = LigeroConfig(ELL=8, K_DEG=16, N_LIG=64, T_QUERIES=4)
    settlement = _make_table_settlement(T_LEN=29, n_z=2, ell=cfg.ELL, paired=False)
    old_row_pkts, _, _, _ = _table_settlement_compile_reference(settlement, None, cfg, 1000)
    new_row_pkts, _, _, _ = table_settlement_compile(settlement, None, cfg, 1000)
    expanded_new = list(_expand_row_pkts(new_row_pkts))
    # Compare by (row, band_key) since the packet objects themselves are
    # distinct instances (fresh coef_vec tensors each call) -- band_key is
    # exactly what downstream code (_index_bands) treats as identity.
    old_keyed = [(r, _band_key(pkt)) for r, pkt in old_row_pkts]
    new_keyed = [(r, _band_key(pkt)) for r, pkt in expanded_new]
    assert old_keyed == new_keyed


# ---------------------------------------------------------------------------
# Randomized: vary table shape (T_LEN, ell, n_z, z lengths, paired) and
# _index_bands input shape (random mixes of ranges / plain tuples / repeated
# keys) rather than only the hand-picked cases above. Fixed seed for
# reproducibility.
# ---------------------------------------------------------------------------
def test_table_settlement_compile_randomized():
    rng = random.Random(20260722)
    for trial in range(25):
        ell = rng.choice([1, 4, 8, 64])
        T_LEN = rng.randint(1, 200)
        n_z = rng.randint(0, 4)
        paired = rng.choice([True, False])
        try:
            cfg = LigeroConfig(ELL=ell, K_DEG=ell * 2, N_LIG=ell * 8, T_QUERIES=4)
            settlement = _make_table_settlement(T_LEN=T_LEN, n_z=n_z, ell=ell, paired=paired)
            old_bands, old_quads, old_n, old_b = _compile_and_index(
                _table_settlement_compile_reference, settlement, cfg)
            new_bands, new_quads, new_n, new_b = _compile_and_index(
                table_settlement_compile, settlement, cfg)
            assert old_n == new_n
            assert old_b == new_b
            assert _bands_as_set(old_bands) == _bands_as_set(new_bands)
        except AssertionError as e:
            raise AssertionError(
                f"trial {trial} failed: ell={ell} T_LEN={T_LEN} n_z={n_z} paired={paired}: {e}")


def _random_range_or_tuple(rng, pkt, lo, hi):
    """Randomly represent [lo, hi) rows of `pkt` as either one _PktRange or
    a list of individual (row, pkt) tuples -- _index_bands must treat both
    identically."""
    if rng.random() < 0.5:
        return [_PktRange(pkt, lo, hi)]
    return [(r, pkt) for r in range(lo, hi)]


def test_index_bands_randomized_mixed_representations():
    rng = random.Random(4242)
    for trial in range(40):
        n_keys = rng.randint(1, 4)
        pkts = [_FakePkt(base=k, var_row_start=k * 1000) for k in range(n_keys)]
        # Each key gets 1-3 row spans (possibly overlapping/adjacent), each
        # independently emitted as a range or as flat tuples.
        old = _FakeIndexer()
        new = _FakeIndexer()
        for pkt in pkts:
            n_spans = rng.randint(1, 3)
            cursor = 0
            for _ in range(n_spans):
                lo = cursor + rng.randint(0, 5)
                hi = lo + rng.randint(1, 15)
                cursor = hi
                flat = [(r, pkt) for r in range(lo, hi)]
                old._index_bands(flat)
                new._index_bands(_random_range_or_tuple(rng, pkt, lo, hi))
        try:
            assert _bands_as_set(old.bands) == _bands_as_set(new.bands)
        except AssertionError as e:
            raise AssertionError(f"trial {trial} failed (n_keys={n_keys}): {e}")
