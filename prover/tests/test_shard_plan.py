"""Torch-free checks for the weight-split shard plans (shard_plan.py):
validation rejects gaps, overlaps, reversed/out-of-range runs and a device
with two runs; accepted plans tile the block exactly; from_pairs /
two_way / as_plan agree; owned_ids and worker_runs read off the plan.

    python3 prover/tests/run_tests.py test_shard_plan     # no GPU needed
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from shard_plan import ShardPlan, as_plan, validate_runs


class _V:
    def __init__(self, i):
        self.i = i


def _raises(fn, needle):
    try:
        fn()
    except ValueError as e:
        assert needle in str(e), (needle, str(e))
        return
    raise AssertionError(f"expected ValueError containing {needle!r}")


def test_validate_runs_tiles_exactly():
    n = 10
    ok = validate_runs([(1, 4, 10), (0, 0, 4)], n)
    assert ok == [(0, 0, 4), (1, 4, 10)]                       # sorted by lo
    assert validate_runs([(0, 0, 10), (1, 10, 10)], n)[1] == (1, 10, 10)   # empty run ok
    assert validate_runs([(0, 0, 0), (1, 0, 10)], n)[0] == (0, 0, 0)       # coordinator idle ok
    _raises(lambda: validate_runs([(0, 0, 4), (1, 5, 10)], n), "owned by nobody")
    _raises(lambda: validate_runs([(0, 0, 6), (1, 4, 10)], n), "owned twice")
    _raises(lambda: validate_runs([(0, 0, 4), (1, 4, 9)], n), "[9, 10) are owned by nobody")
    _raises(lambda: validate_runs([(0, 0, 4), (1, 4, 11)], n), "outside")
    _raises(lambda: validate_runs([(0, 4, 0), (1, 4, 10)], n), "reversed")
    _raises(lambda: validate_runs([(0, 0, 4), (0, 4, 10)], n), "two runs")
    _raises(lambda: validate_runs([(0, 0, 4), (-1, 4, 10)], n), "bad device")
    _raises(lambda: validate_runs([(0, 0, 4)], n), "owned by nobody")
    # non-index values are rejected here, not in a slice much later
    _raises(lambda: validate_runs([(0, 0.0, 3.0), (1, 3.0, 10.0)], n), "lo must be an int")
    _raises(lambda: validate_runs([(0, 0, 4), (1, 4, 10.0)], n), "hi must be an int")
    _raises(lambda: validate_runs([(0.0, 0, 4), (1, 4, 10)], n), "device must be an int")
    _raises(lambda: validate_runs([(0, False, 4), (1, 4, 10)], n), "lo must be an int")
    _raises(lambda: validate_runs([(True, 0, 4), (1, 4, 10)], n), "device must be an int")
    _raises(lambda: validate_runs([(0, 0), (1, 4, 10)], n), "must be (device, lo, hi)")
    _raises(lambda: validate_runs([7, (1, 4, 10)], n), "must be (device, lo, hi)")
    print("    validate_runs: gaps, overlaps, ranges, duplicates rejected")


def test_plan_constructors_and_views():
    n = 7
    p = ShardPlan.two_way(3, cut_open=5).validated(n)
    assert p.fold == [(0, 0, 3), (1, 3, 7)] and p.open == [(0, 0, 5), (1, 5, 7)]
    assert p.coordinator_run("fold") == (0, 3) and p.coordinator_run("open") == (0, 5)
    assert p.worker_runs("fold") == [(1, 3, 7)] and p.worker_runs("open") == [(1, 5, 7)]
    assert p.n_devices() == 2
    q = as_plan(([(0, 2), (2, 5), (5, 7)], [(0, 1), (1, 6), (6, 7)]), n)
    assert q.fold == [(0, 0, 2), (1, 2, 5), (2, 5, 7)]
    assert q.open == [(0, 0, 1), (1, 1, 6), (2, 6, 7)]
    assert q.worker_runs("open") == [(1, 1, 6), (2, 6, 7)] and q.n_devices() == 3
    vs = [_V(i) for i in range(n)]
    assert q.owned_ids(0, "fold", vs) == {id(vs[0]), id(vs[1])}
    assert q.owned_ids(2, "open", vs) == {id(vs[6])}
    assert q.owned_ids(5, "open", vs) == set()
    # empty worker runs are skipped by worker_runs
    r = as_plan(([(0, 7), (7, 7)], [(0, 7), (7, 7)]), n)
    assert r.worker_runs("fold") == [] and r.worker_runs("open") == []
    # the open plan defaults to the fold plan
    s = ShardPlan.from_pairs([(0, 4), (4, 7)]).validated(n)
    assert s.open == s.fold
    # the two_way -1 sentinel expands ONLY as a true int: a float -1.0 is
    # rejected by validation, and a malformed raw run gets the intended
    # ValueError rather than a tuple-unpacking exception
    _raises(lambda: ShardPlan(fold=[(0, 0, 3), (1, 3, -1.0)],
                              open=[(0, 0, 3), (1, 3, -1.0)]).validated(n),
            "hi must be an int")
    _raises(lambda: ShardPlan(fold=[(0, 0, 3), (1, 3)],
                              open=[(0, 0, 3), (1, 3, 7)]).validated(n),
            "must be (device, lo, hi)")
    _raises(lambda: ShardPlan(fold=[7, (1, 0, 7)],
                              open=[(0, 0, 7), (1, 7, 7)]).validated(n),
            "must be (device, lo, hi)")
    _raises(lambda: ShardPlan(fold=[(0, 0, 3), (True, 3, -1)],
                              open=[(0, 0, 7), (1, 7, 7)]).validated(n),
            "device must be an int")
    # bad shapes
    try:
        as_plan("nope", n)
        raise AssertionError("accepted a string")
    except TypeError:
        pass
    _raises(lambda: as_plan(([(0, 4), (4, 7)], [(0, 4), (5, 7)]), n), "open plan")
    _raises(lambda: as_plan(([(0, 4.0), (4.0, 7)], [(0, 4), (4, 7)]), n), "fold plan: hi must be an int")
    _raises(lambda: p.runs("commit"), "unknown stage")
    print("    constructors/views: two_way, from_pairs, as_plan, owned_ids, worker_runs")


if __name__ == "__main__":
    test_validate_runs_tiles_exactly()
    test_plan_constructors_and_views()
    print("test_shard_plan OK")
