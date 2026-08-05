"""Gates for the MoE sort/segment argument (moe.py, doc §5).

Positive: the scan reproduces the direct segment sums; the payoff identity holds;
the degenerate routes the doc calls out (all tokens to one expert, empty experts)
work through the same equations with no special case.

Negative, one per cheat the argument is supposed to stop:
  * edit a value in the sorted stream        -> permutation product differs
  * drop / duplicate a record                -> permutation product differs
  * reorder within a segment (break stability)-> strict key not increasing
  * put a token in the wrong expert's segment -> a_q(e'_q - (c_q - 1)) = 0 fails
  * relabel a delimiter                       -> d_q(label_q - c_q) = 0 fails
  * drop a delimiter                          -> final counter != E+1
  * lie about an emitted accumulator          -> emission check fails

Run:  .venv/bin/python layergkr/tests/run_tests.py test_moe
"""
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from layergkr import moe
from prover.protocol import P as FIELD_P

S, E, D_IN, D_OUT = 7, 3, 4, 3
LAYER, KIND = 5, 1


def _setup(route=None, seed=11):
    r = random.Random(seed)
    route = route if route is not None else [r.randrange(E) for _ in range(S)]
    X = [[r.randrange(FIELD_P) for _ in range(D_IN)] for _ in range(S)]
    W = [[[r.randrange(FIELD_P) for _ in range(D_OUT)] for _ in range(D_IN)]
         for _ in range(E)]
    tau = [r.randrange(FIELD_P) for _ in range(S)]
    rho = [r.randrange(FIELD_P) for _ in range(D_OUT)]
    return route, X, W, tau, rho


def _lane(route, X, tau, j):
    src = moe.source_records(LAYER, KIND, route, X)
    dst = moe.stable_sorted_records(src)
    return src, dst, [r for r in dst if r.j == j]


# ── positive ─────────────────────────────────────────────────────────────────
def test_scan_reproduces_direct_segment_sums():
    route, X, W, tau, rho = _setup()
    A_direct = moe.segment_sums(route, X, tau, E)
    for j in range(D_IN):
        _, _, lane = _lane(route, X, tau, j)
        trace, A_lane = moe.build_lane(lane, E, tau)
        ok, why = moe.scan_constraints_ok(trace, E, tau)
        assert ok, why
        for e in range(E):
            assert A_lane[e] == A_direct[e][j], f"expert {e}, coord {j}"


def test_payoff_identity_holds():
    """sum_{t,i} tau_t rho_i Y[t,i] == sum_{e,j} A[e,j] P[e,j] -- the whole point
    of §5.3: the selected matmul collapses to one scalar with no S*E*K tensor."""
    route, X, W, tau, rho = _setup()
    lhs, rhs = moe.moe_identity_sides(route, X, W, tau, rho)
    assert lhs == rhs, f"{lhs} != {rhs}"


def test_all_tokens_to_one_expert():
    """Degenerate route: every other segment is empty, so delimiters land back to
    back and must emit zero accumulators."""
    route, X, W, tau, rho = _setup(route=[1] * S)
    lhs, rhs = moe.moe_identity_sides(route, X, W, tau, rho)
    assert lhs == rhs
    _, _, lane = _lane(route, X, tau, 0)
    trace, A_lane = moe.build_lane(lane, E, tau)
    ok, why = moe.scan_constraints_ok(trace, E, tau)
    assert ok, why
    assert A_lane[0] == 0 and A_lane[2] == 0, A_lane


def test_permutation_accepts_the_honest_sort():
    route, X, W, tau, rho = _setup()
    src, dst, _ = _lane(route, X, tau, 0)
    r = random.Random(2)
    beta, z = r.randrange(FIELD_P), r.randrange(FIELD_P)
    assert moe.permutation_ok(src, dst, beta, z)
    ok, why = moe.strict_order_ok(dst, S)
    assert ok, why


def test_cell_counts_match_the_doc_scale():
    """The §5.3 accounting at Maverick's numbers: 56.623B naive cells vs
    442.368M segmented."""
    per_layer = moe.cell_counts(S=1000, E=128, d=5120, d_ff=8192)
    n_moe_layers = 24
    assert n_moe_layers * per_layer["naive"] == 56_623_104_000     # doc: 56.623B
    assert n_moe_layers * per_layer["segmented"] == 442_368_000    # doc: 442.368M
    assert per_layer["ratio"] == 128                               # exactly E


# ── negative ─────────────────────────────────────────────────────────────────
def _beta_z(seed=3):
    r = random.Random(seed)
    return r.randrange(FIELD_P), r.randrange(FIELD_P)


def test_edited_value_breaks_the_permutation():
    route, X, W, tau, rho = _setup()
    src, dst, _ = _lane(route, X, tau, 0)
    beta, z = _beta_z()
    bad = list(dst)
    bad[2] = moe.Record(bad[2].layer, bad[2].kind, bad[2].t, bad[2].e, bad[2].j,
                        (bad[2].x + 1) % FIELD_P)
    assert not moe.permutation_ok(src, bad, beta, z)


def test_dropped_and_duplicated_records_break_the_permutation():
    route, X, W, tau, rho = _setup()
    src, dst, _ = _lane(route, X, tau, 0)
    beta, z = _beta_z()
    assert not moe.permutation_ok(src, dst[:-1], beta, z), "drop went undetected"
    assert not moe.permutation_ok(src, dst[:-1] + [dst[0]], beta, z), "dup undetected"


def test_relabelled_expert_breaks_the_permutation():
    """Changing which expert a record claims is inside the fingerprint."""
    route, X, W, tau, rho = _setup()
    src, dst, _ = _lane(route, X, tau, 0)
    beta, z = _beta_z()
    bad = list(dst)
    bad[0] = moe.Record(bad[0].layer, bad[0].kind, bad[0].t,
                        (bad[0].e + 1) % E, bad[0].j, bad[0].x)
    assert not moe.permutation_ok(src, bad, beta, z)


def test_unstable_order_is_rejected():
    """Swapping two records of the same expert breaks the strictly-increasing key."""
    route, X, W, tau, rho = _setup(route=[0] * S)
    _, _, lane = _lane(route, X, tau, 0)
    swapped = list(lane)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    ok, why = moe.strict_order_ok(swapped, S)
    assert not ok and "not >" in why, why


def test_token_in_the_wrong_segment_is_rejected():
    route, X, W, tau, rho = _setup()
    _, _, lane = _lane(route, X, tau, 0)
    trace, _ = moe.build_lane(lane, E, tau)
    tok = next(i for i, r in enumerate(trace) if not r.is_delim)
    trace[tok].e = (trace[tok].e + 1) % E
    ok, why = moe.scan_constraints_ok(trace, E, tau)
    assert not ok and "segment" in why, why


def test_relabelled_delimiter_is_rejected():
    route, X, W, tau, rho = _setup()
    _, _, lane = _lane(route, X, tau, 0)
    trace, _ = moe.build_lane(lane, E, tau)
    d_idx = [i for i, r in enumerate(trace) if r.is_delim][1]
    trace[d_idx].label += 1
    ok, why = moe.scan_constraints_ok(trace, E, tau)
    assert not ok and "delimiter label" in why, why


def test_dropped_delimiter_is_rejected():
    route, X, W, tau, rho = _setup()
    _, _, lane = _lane(route, X, tau, 0)
    trace, _ = moe.build_lane(lane, E, tau)
    d_idx = [i for i, r in enumerate(trace) if r.is_delim][-1]
    del trace[d_idx]
    ok, why = moe.scan_constraints_ok(trace, E, tau)
    assert not ok and "final counter" in why, why


def test_lied_emission_is_rejected():
    route, X, W, tau, rho = _setup()
    _, _, lane = _lane(route, X, tau, 0)
    trace, _ = moe.build_lane(lane, E, tau)
    d_idx = [i for i, r in enumerate(trace) if r.is_delim][-1]
    trace[d_idx].emitted = (trace[d_idx].emitted + 1) % FIELD_P
    ok, why = moe.scan_constraints_ok(trace, E, tau)
    assert not ok and "emitted" in why, why
