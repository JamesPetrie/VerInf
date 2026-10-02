"""Top-k routing, the relations themselves (CPU): the threshold guard, the
review's counterexample, and the selection, slot and gate-weight relations
over integers mod P (prover/topk_reference.py). The GPU suite
test_topk_routing.py proves the same witnesses through the Rust verifier.

The counterexample (review of 2026-10-01, analysis/topk-routing-design.md
§3.1): E = 8, k = 3, B_q = 40, three 21-bit words, every score zero; the
cheat selects {5, 6, 7} with τ = (P − 1)/2 + 3. Every relation and range
holds at R = 63, which top-1's guard allows; the threshold guard refuses
R = 63, and at a guarded R the cheat's v leaves the range."""
import pathlib
import random
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import topk_params as tp          # noqa: E402
import topk_reference as ref      # noqa: E402
from topk_params import P         # noqa: E402

E, K = 8, 3


def _cheat(sel, tau, T=1):
    """The full alternative witness selecting `sel` with threshold `tau` on
    all-zero scores, every derived value consistent with it."""
    w = {"m": [], "qt": [], "tau": [], "d": [], "md": [], "v": []}
    for _ in range(T):
        qt = ref.tiebroken([0] * E, E)
        m = [1 if e in sel else 0 for e in range(E)]
        d = [(q - tau) % P for q in qt]
        md = [(m[e] * d[e]) % P for e in range(E)]
        for n, row in (("m", m), ("qt", qt), ("d", d), ("md", md),
                       ("v", [(2 * md[e] - d[e]) % P for e in range(E)])):
            w[n].append(row)
        w["tau"].append(tau)
    return w


# ---- the guard ---------------------------------------------------------------

def test_the_counterexample_parameters_pass_top1_and_fail_the_threshold_guard():
    width = 40 + tp.l_bits(E)                 # B_q = 40, L = 3
    assert tp.top1_guard_ok(width, 63)        # three 21-bit words
    assert not tp.threshold_guard_ok(width, 63)
    with pytest.raises(ValueError, match="unsound top-k"):
        tp.threshold_words(width, 21)
    assert tp.threshold_words(width, 12) == (4, 48)


def test_the_guard_caps_the_window_at_62_bits():
    assert tp.threshold_guard_ok(43, 62) and not tp.threshold_guard_ok(43, 63)
    with pytest.raises(ValueError, match="cover"):
        tp.threshold_words(43, 8, n_words=5)  # 40 < 43


# ---- the counterexample ---------------------------------------------------------

def test_the_counterexample_satisfies_every_relation_at_63_bits():
    s, b = [[0] * E], [0] * E
    cheat = _cheat({5, 6, 7}, (P - 1) // 2 + 3)
    assert ref.routing_failures(cheat, s, b, K, range_bits=63) == set()
    honest = ref.routing_witness(s, b, K)
    assert [e for e in range(E) if honest["m"][0][e]] == [0, 1, 2]


def test_the_counterexample_fails_the_range_at_a_guarded_width():
    s, b = [[0] * E], [0] * E
    cheat = _cheat({5, 6, 7}, (P - 1) // 2 + 3)
    _, R = tp.threshold_words(40 + tp.l_bits(E), 12)
    assert ref.routing_failures(cheat, s, b, K, range_bits=R) == {"R"}


def test_no_threshold_admits_a_wrong_selection_under_the_guard():
    """Under the guard, the wrong selection fails for τ at every wrap point the
    argument names: near 0, near the selected and unselected values, near
    P/2 and near P."""
    s, b = [[0] * E], [0] * E
    _, R = tp.threshold_words(3 + tp.l_bits(E), 4)
    cands = {0, 1, 2, 3, 4, 5, 7, (P - 1) // 2, (P - 1) // 2 + 3, (P + 1) // 2,
             P - 1, P - 3, P - (1 << R), (1 << R) - 1, 1 << R}
    for sel in ({5, 6, 7}, {0, 1, 3}, {2, 3, 4}):
        for tau in cands:
            assert ref.routing_failures(_cheat(sel, tau % P), s, b, K, R), (sel, tau)


# ---- honest witnesses --------------------------------------------------------

def _scores(seed, T, lo=0, hi=1 << 10):
    rnd = random.Random(seed)
    return [[rnd.randrange(lo, hi) for _ in range(E)] for _ in range(T)]


def test_honest_selection_with_ties_and_a_signed_bias():
    s = _scores(1, 4, hi=8)                   # small values: ties are common
    b = [3, -2, 0, 0, 5, -7, 1, 0]
    width = 11 + tp.l_bits(E)                 # |s + b| < 2^10 with headroom
    _, R = tp.threshold_words(width, 8)
    w = ref.routing_witness(s, b, K)
    assert ref.routing_failures(w, s, b, K, R) == set()
    for t in range(4):
        q = [s[t][e] + b[e] for e in range(E)]
        want = sorted(range(E), key=lambda e: (-q[e], e))[:K]   # lowest index on a tie
        assert sorted(e for e in range(E) if w["m"][t][e]) == sorted(want)


@pytest.mark.parametrize("drop,add", [(0, 0), (2, 4)])
def test_a_swapped_expert_breaks_the_range(drop, add):
    """Drop the drop-th selected expert (by index) for the add-th unselected
    one, with the threshold most favorable to the cheat."""
    s, b = _scores(2, 1), [0] * E
    width = 11 + tp.l_bits(E)
    _, R = tp.threshold_words(width, 8)
    w = ref.routing_witness(s, b, K)
    sel = sorted(e for e in range(E) if w["m"][0][e])
    uns = sorted(e for e in range(E) if not w["m"][0][e])
    cheat_sel = (set(sel) - {sel[drop]}) | {uns[add]}
    qt = w["qt"][0]
    tau = min(qt[e] for e in cheat_sel)       # the cheat's best threshold
    cheat = {"m": [[1 if e in cheat_sel else 0 for e in range(E)]], "qt": [qt],
             "tau": [tau]}
    cheat["d"] = [[(q - tau) % P for q in qt]]
    cheat["md"] = [[(cheat["m"][0][e] * cheat["d"][0][e]) % P for e in range(E)]]
    cheat["v"] = [[(2 * cheat["md"][0][e] - cheat["d"][0][e]) % P for e in range(E)]]
    assert ref.routing_failures(cheat, s, b, K, R) == {"R"}


def test_cardinality_and_booleanity_are_enforced():
    s, b = _scores(3, 1), [0] * E
    w = ref.routing_witness(s, b, K)
    four = {n: [list(r) for r in w[n]] for n in ("m", "qt", "d", "md", "v")}
    four["tau"] = list(w["tau"])
    four["m"][0][next(e for e in range(E) if not w["m"][0][e])] = 1
    assert "F2" in ref.routing_failures(four, s, b, K, 24)
    two = {n: [list(r) for r in w[n]] for n in ("m", "qt", "d", "md", "v")}
    two["tau"] = list(w["tau"])
    e0, e1 = [e for e in range(E) if w["m"][0][e]][:2]
    two["m"][0][e0], two["m"][0][e1] = 2, 0     # still sums to k
    assert "Q1" in ref.routing_failures(two, s, b, K, 24)


# ---- slots -------------------------------------------------------------------

def test_honest_slots_and_their_cheats():
    s, b = _scores(4, 3), [0] * E
    m = ref.routing_witness(s, b, K)["m"]
    M, ss = ref.slot_witness(m, s, K)
    assert ref.slot_failures(M, ss, m, s) == set()
    dup = [[list(r) for r in Mi] for Mi in M]   # slots 0 and 1 on the same expert
    e1 = next(e for e in range(E) if M[1][0][e])
    e0 = next(e for e in range(E) if M[0][0][e])
    dup[1][0][e1], dup[1][0][e0] = 0, 1
    assert "S" in ref.slot_failures(dup, ss, m, s)
    off = [[list(r) for r in Mi] for Mi in M]   # a slot on an unselected expert
    eu = next(e for e in range(E) if not m[0][e])
    off[2][0] = [1 if e == eu else 0 for e in range(E)]
    assert "S" in ref.slot_failures(off, ss, m, s)
    bad_ss = [list(r) for r in ss]
    bad_ss[0][0] += 1
    assert ref.slot_failures(M, bad_ss, m, s) == {"V"}


# ---- the gate bracket -------------------------------------------------------

C = round(2.827 * (1 << 12))                  # K2's routed scale at w-scale 2^12
REM_BITS, W_BITS = 16, 15


def test_honest_weights_are_the_floor_quotients():
    ss = [[300, 200, 100], [1, 1, 1], [1023, 5, 7]]
    br = ref.bracket_witness(ss, C)
    assert ref.bracket_failures(br, ss, C, REM_BITS, W_BITS) == set()
    assert br["w"][1] == [C // 3] * 3
    assert tp.bracket_guard_ok(z_bits=12, rem_bits=REM_BITS, w_bits=W_BITS, cs_bits=27)


@pytest.mark.parametrize("delta,broken", [(+1, "Rr"), (-1, "Rg")])
def test_a_shifted_quotient_breaks_a_range(delta, broken):
    ss = [[300, 200, 100]]
    br = ref.bracket_witness(ss, C)
    Z = br["Z"][0]
    br["w"][0][0] += delta                     # keep w·Z + rem = C·ss
    br["rem"][0][0] -= delta * Z
    br["gr"][0][0] = Z - 1 - br["rem"][0][0]
    assert ref.bracket_failures(br, ss, C, REM_BITS, W_BITS) == {broken}


def test_the_combine_weights_each_slot_by_its_token():
    T, F = 2, 3
    D = [[10 * r + j for j in range(F)] for r in range(K * T)]
    w = [[1, 2, 3], [4, 5, 6]]
    y = ref.combine(w, D, K, T)
    assert y[1][0] == 4 * D[1][0] + 5 * D[3][0] + 6 * D[5][0]
