"""The profiler's cost rows for the top-k claims, against the prover (CPU).

profiler/claimcosts.py prices each claim as (W, cids, Q): the slots of the
claim's own outputs, the linear constraint ids its compile advances over, and
the lengths of its quadratic families. For TopkRoutingClaim, TopkSlotsClaim,
GateBracketClaim, SplitClaim and ConcatClaim this suite builds the claim with
laid-out variables, runs the prover's own compile function, and compares
cur - base and the quad lengths with the row, at several shapes. The two
compile helpers that touch the GPU (the all-ones vector and the RHS chunk)
are replaced by CPU stand-ins; neither changes a count."""
import pathlib
import sys

import pytest
import torch

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "prover"))
sys.path.insert(0, str(REPO / "profiler"))

import claimcosts                                          # noqa: E402
import topk_routing as tr                                  # noqa: E402
from claims import ConcatClaim, concat_compile             # noqa: E402
from core import LigeroConfig, Variable                    # noqa: E402

CFG = LigeroConfig(ELL=16, K_DEG=16, N_LIG=64, T_QUERIES=4)


@pytest.fixture(autouse=True)
def _cpu_compile(monkeypatch):
    monkeypatch.setattr(tr, "_ones", lambda n: torch.ones(n, dtype=torch.uint64))
    monkeypatch.setattr(tr, "_build_b_chunk", lambda n, fam: None)


class _Layout:
    """Variables laid out row after row, as the prover's layout would."""

    def __init__(self):
        self.row = 0

    def var(self, name, length):
        v = Variable(name, length=length)
        v.row_start = self.row
        self.row += v.n_rows(CFG.ELL)
        return v


def _counts(compile_fn, claim, outputs):
    pkts, quads, n_added, _ = compile_fn(claim, None, CFG, 1000)
    return (float(sum(v.length for v in outputs)), float(n_added),
            float(sum(q.L for q in quads)))


SHAPES = [(2, 8, 3), (5, 16, 4), (3, 384, 8), (7, 33, 1)]


@pytest.mark.parametrize("T,E,k", SHAPES)
def test_topk_routing_row(T, E, k):
    lay = _Layout()
    s, b = lay.var("s", T * E), lay.var("b", E)
    outs = {n: lay.var(n, T * E) for n in ("m", "qt", "d", "md", "v")}
    tau = lay.var("tau", T)
    c = tr.TopkRoutingClaim(s=s, b=b, tau=tau, T=T, E=E, k=k, L_bits=9, width=20,
                            range_bits=24, **outs)
    got = _counts(tr.topk_compile, c, list(outs.values()) + [tau])
    assert got == claimcosts.cost("TopkRoutingClaim", dict(T=T, E=E, k=k))


@pytest.mark.parametrize("T,E,k", SHAPES)
def test_topk_slots_row(T, E, k):
    lay = _Layout()
    m, s = lay.var("m", T * E), lay.var("s", T * E)
    M = [lay.var(f"M{i}", T * E) for i in range(k)]
    MS = [lay.var(f"MS{i}", T * E) for i in range(k)]
    ss = lay.var("ss", T * k)
    c = tr.TopkSlotsClaim(m=m, s=s, M=M, MS=MS, ss=ss, T=T, E=E, k=k)
    got = _counts(tr.slots_compile, c, M + MS + [ss])
    assert got == claimcosts.cost("TopkSlotsClaim", dict(T=T, E=E, k=k))


@pytest.mark.parametrize("T,E,k", SHAPES)
def test_gate_bracket_row(T, E, k):
    lay = _Layout()
    ss, Z = lay.var("ss", T * k), lay.var("Z", T)
    outs = {n: lay.var(n, T * k) for n in ("Zb", "w", "wZ", "rem", "gr")}
    c = tr.GateBracketClaim(ss=ss, Z=Z, T=T, k=k, C=11579, z_bits=17, cs_bits=27,
                            rem_bits=24, w_bits=16, **outs)
    got = _counts(tr.bracket_compile, c, [Z] + list(outs.values()))
    assert got == claimcosts.cost("GateBracketClaim", dict(T=T, k=k))


@pytest.mark.parametrize("parts,each", [(3, 12), (8, 7168 * 2), (1, 5)])
def test_split_and_concat_rows(parts, each):
    lay = _Layout()
    whole = lay.var("whole", parts * each)
    pieces = [lay.var(f"p{i}", each) for i in range(parts)]
    sc = tr.SplitClaim(whole=whole, parts=pieces)
    assert _counts(tr.split_compile, sc, pieces) == claimcosts.cost(
        "SplitClaim", dict(length=sc.length))
    cc = ConcatClaim(srcs=pieces, dst=whole)
    assert _counts(concat_compile, cc, [whole]) == claimcosts.cost(
        "ConcatClaim", dict(length=cc.length))


def test_the_extractor_records_concat_and_split_lengths():
    from extract import _claim_params
    lay = _Layout()
    whole = lay.var("whole", 24)
    pieces = [lay.var(f"p{i}", 8) for i in range(3)]
    assert _claim_params(tr.SplitClaim(whole=whole, parts=pieces))["length"] == 24
    assert _claim_params(ConcatClaim(srcs=pieces, dst=whole))["length"] == 24
