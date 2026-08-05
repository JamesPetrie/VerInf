"""Gates for the tensor semantics path (semantics.forward_tensor).

There are now two implementations of the layer, and the fast one is the one that
will be used. The single failure mode that matters is a fast prover that proves a
slightly different statement, so these tests pin equivalence rather than speed:

  * the trace is equal field by field, in order, including gate node ids;
  * `check_trace` -- which shares no arithmetic with either path -- accepts both;
  * both traces produce the SAME proof, byte for byte;
  * the range guard, which is what makes the int64 representation legitimate,
    actually fires instead of merely existing.

Skipped cleanly when there is no CUDA.

Run:  .venv/bin/python layergkr/tests/run_tests.py test_semantics
"""
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from layergkr import full_layer as fl, gpu, rs, semantics as sem
from layergkr.bench.validate_semantics import compare_traces, digest

CFG_RS = rs.Config(ELL=64, K_DEG=128, N_LIG=256, T_QUERIES=8)


def _skip() -> bool:
    if not gpu.available():
        print("    (no CUDA — skipped)")
        return True
    return False


def _pair(S, d, d_ff, E, seed=7, table_bits=6):
    """The same weights through both paths. Drawn on the device and converted
    down, never drawn twice from a seed -- equal seeds only give equal weights
    while both implementations consume the stream identically, which is the
    assumption under test."""
    cfg = sem.ToyConfig(S=S, d=d, d_ff=d_ff, E=E, table_bits=table_bits,
                        scale_bits=6)
    w_dev = sem.LayerWeights.draw_tensor(cfg, seed)
    ref = sem.forward(cfg, None, weights=w_dev.to_lists())
    got = sem.to_python(sem.forward_tensor(cfg, None, weights=w_dev))
    return cfg, ref, got


def test_tensor_trace_equals_reference_field_by_field():
    if _skip():
        return
    for S, d, d_ff, E in ((4, 8, 16, 1), (6, 16, 32, 3), (8, 16, 32, 5),
                          (5, 32, 64, 2)):
        _, ref, got = _pair(S, d, d_ff, E)
        where = compare_traces(ref, got)
        assert where == "", f"S={S} d={d} E={E}: first difference at {where}"


def test_independent_arbiter_accepts_the_tensor_trace():
    if _skip():
        return
    _, ref, got = _pair(6, 16, 32, 3)
    for name, tr in (("reference", ref), ("tensor", got)):
        ok, why = sem.check_trace(tr)
        assert ok, f"{name} trace inconsistent: {why}"


def test_check_trace_reads_a_device_trace_without_materialising_it_first():
    """A tensor trace handed straight to the arbiter must be checked, not waved
    through: 0-dim tensors are hashable by identity, so the table membership
    test would pass on anything if the trace were indexed elementwise."""
    if _skip():
        return
    cfg = sem.ToyConfig(S=5, d=16, d_ff=32, E=2, table_bits=6, scale_bits=6)
    trace = sem.forward_tensor(cfg, None,
                               weights=sem.LayerWeights.draw_tensor(cfg, 3))
    ok, why = sem.check_trace(trace)
    assert ok, why
    # and a corrupted one is caught
    bad = sem.to_python(trace)
    bad.matmuls[0].Y[0][0] = (bad.matmuls[0].Y[0][0] + 1) % (1 << 62)
    ok, why = sem.check_trace(bad)
    assert not ok and "matmul" in why, f"corruption not caught: {why}"


def test_both_traces_produce_the_same_proof():
    if _skip():
        return
    for S, d, d_ff, E in ((4, 8, 16, 1), (6, 16, 32, 3)):
        _, ref, got = _pair(S, d, d_ff, E)
        a = digest(fl.prove_full_layer(ref, CFG_RS, fl.Enrollment(CFG_RS), 8,
                                       random.Random(7)))
        b = digest(fl.prove_full_layer(got, CFG_RS, fl.Enrollment(CFG_RS), 8,
                                       random.Random(7)))
        assert a == b, f"S={S} d={d} E={E}: proof digests differ, {a} != {b}"


def test_the_range_guard_fires_instead_of_wrapping():
    """int64 carries the TRUE value, so the two paths agree only below 2^63.
    A configuration that would cross it must stop, and say where."""
    if _skip():
        return
    cfg = sem.ToyConfig(S=16, d=4096, d_ff=8192, E=2, table_bits=10, scale_bits=6)
    try:
        sem.forward_tensor(cfg, None, weights=sem.LayerWeights.draw_tensor(cfg, 7))
    except sem.RangeOverflow as e:
        assert "2^63" in str(e) and ">=" in str(e), f"unhelpful message: {e}"
        return
    raise AssertionError("a configuration past 2^63 was accepted silently")


def test_headroom_shrinks_as_the_layer_widens():
    """The wall is not hypothetical: the toy's values grow by ~n_in per matmul
    because the rescale only divides by `scale`. Pinning the DIRECTION here
    means a future change that quietly removes the growth gets noticed."""
    if _skip():
        return
    seen = []
    for d in (256, 1024):
        cfg = sem.ToyConfig(S=8, d=d, d_ff=2 * d, E=2, table_bits=6, scale_bits=6)
        tr = sem.forward_tensor(cfg, None,
                                weights=sem.LayerWeights.draw_tensor(cfg, 7))
        seen.append(tr.headroom_bits)
    assert seen[1] < seen[0], f"headroom did not shrink with width: {seen}"
    assert all(h > 0 for h in seen), f"a passing run must have headroom: {seen}"
