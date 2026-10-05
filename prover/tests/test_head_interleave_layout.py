"""HeadInterleaveClaim on the CPU: the witness, the constraints the prover's
fold will actually apply, and the profiler's cost row.

The constraint check goes through core._lower_geometry, the descriptor the
fold's kernel interprets (mixed-radix digits -> cid by strided dots, plus a
fan axis), and evaluates it here: every dst constraint must hold the dst
slot with coef -1 and exactly one source slot with coef +1, per head and
shared (handlers.rs head_interleave_tests checks the same map in the
verifier). The GPU suite test_head_interleave.py proves it end to end."""
import pathlib
import sys

import pytest
import torch

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "prover"))
sys.path.insert(0, str(REPO / "profiler"))

import claimcosts                                           # noqa: E402
import head_interleave as hi                                # noqa: E402
from core import P, LigeroConfig, Variable, _lower_geometry  # noqa: E402

CFG = LigeroConfig(ELL=4, K_DEG=8, N_LIG=32, T_QUERIES=4)


def _var(name, length, row):
    v = Variable(name, length=length)
    v.row_start = row
    return v


def _claim(T, H, w1, w2, shared):
    a = _var("a", T * H * w1, 0)
    b = _var("b", T * w2 if shared else T * H * w2, 100)
    dst = _var("dst", T * H * (w1 + w2), 200)
    return hi.HeadInterleaveClaim(a=a, b=b, dst=dst, T=T, H=H, w1=w1, w2=w2,
                                  shared=int(shared))


def _apply(geo, base, f):
    """The descriptor's cids for flat slot f (q <= 2 digits here)."""
    q, shape, strides, fan, fan_stride = geo[0], geo[1], geo[2], geo[3], geo[4]
    digits, rem = [], f
    for r in reversed(shape[:q]):
        digits.append(rem % r)
        rem //= r
    digits = digits[::-1]
    cid0 = base + sum(d * s for d, s in zip(digits, strides[:q]))
    return [cid0 + h * fan_stride for h in range(fan)], geo[6]


def _constraint_map(c):
    pk, quads, n_added, b_chunk = hi.head_interleave_compile(c, None, CFG, 1000)
    by_var = {c.a.row_start: c.a, c.b.row_start: c.b, c.dst.row_start: c.dst}
    seen, m = set(), {}
    for _, pkt in pk:
        if id(pkt) in seen:
            continue                       # one packet object serves all its rows
        seen.add(id(pkt))
        geo = _lower_geometry(pkt)
        assert geo is not None, type(pkt).__name__
        var = by_var[pkt.var_row_start]
        for f in range(var.length):
            cids, coef = _apply(geo, pkt.base, f)
            for cid in cids:
                m.setdefault(cid, []).append((var.name, f, coef))
    return m, quads, n_added, b_chunk


@pytest.mark.parametrize("T,H,w1,w2,shared", [(2, 3, 2, 1, False), (2, 3, 2, 1, True),
                                               (3, 4, 128, 64, False), (3, 4, 128, 64, True)])
def test_every_dst_slot_is_pinned_to_exactly_one_source(T, H, w1, w2, shared):
    c = _claim(T, H, w1, w2, shared)
    m, quads, n_added, b_chunk = _constraint_map(c)
    W = w1 + w2
    assert n_added == T * H * W and quads == [] and b_chunk is None
    assert sorted(m) == list(range(1000, 1000 + T * H * W))
    for t in range(T):
        for h in range(H):
            for col in range(W):
                cid = 1000 + (t * H + h) * W + col
                if col < w1:
                    src = ("a", (t * H + h) * w1 + col)
                elif shared:
                    src = ("b", t * w2 + col - w1)
                else:
                    src = ("b", (t * H + h) * w2 + col - w1)
                want = sorted([("dst", cid - 1000, (P - 1) % P), (src[0], src[1], 1)])
                assert sorted(m[cid]) == want, (t, h, col)


@pytest.mark.parametrize("shared", [False, True])
def test_the_witness_interleaves_per_head(shared):
    T, H, w1, w2 = 2, 3, 4, 2
    c = _claim(T, H, w1, w2, shared)
    a = torch.arange(T * H * w1, dtype=torch.int64)
    b = 1000 + torch.arange(c.b.length, dtype=torch.int64)
    out = hi.head_interleave_compute(c, {c.a: a.view(torch.uint64),
                                         c.b: b.view(torch.uint64)})[c.dst]
    got = out.view(torch.int64).view(T, H, w1 + w2).tolist()
    for t in range(T):
        for h in range(H):
            want_b = (b.view(T, w2)[t] if shared else b.view(T, H, w2)[t, h]).tolist()
            assert got[t][h] == a.view(T, H, w1)[t, h].tolist() + want_b


@pytest.mark.parametrize("shared", [False, True])
def test_the_cost_row_matches_the_compile(shared):
    T, H, w1, w2 = 3, 4, 128, 64
    c = _claim(T, H, w1, w2, shared)
    _, quads, n_added, _ = hi.head_interleave_compile(c, None, CFG, 0)
    assert claimcosts.cost("HeadInterleaveClaim",
                           dict(T=T, H=H, w1=w1, w2=w2, shared=int(shared))) == \
        (float(c.dst.length), float(n_added), float(sum(q.L for q in quads)))


def test_the_evaluator_reads_an_existing_descriptor_as_documented():
    """_apply's reading of the descriptor, checked on L2_TransposeO2MScalar's
    documented map (cid = base + (f % cols)·rows·fan + (f // cols)·fan + k),
    so the checks above test the lowering, not the evaluator."""
    from packets import L2_TransposeO2MScalar
    rows, cols, fan = 3, 4, 2
    pkt = L2_TransposeO2MScalar(base=50, var_row_start=0, L=rows * cols, rows=rows,
                                cols=cols, fan=fan, coef=7)
    geo = _lower_geometry(pkt)
    for f in range(rows * cols):
        cids, coef = _apply(geo, pkt.base, f)
        assert cids == [50 + (f % cols) * rows * fan + (f // cols) * fan + k for k in range(fan)]
        assert coef == 7
