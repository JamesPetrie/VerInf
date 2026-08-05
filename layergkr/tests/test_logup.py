"""Gates for logup.py (doc §6).

Positive: an honest lookup proves and verifies; the two sides of the identity
agree; repeated queries are handled by multiplicities.

Ordering (the whole reason §6 exists): the reciprocal point alpha must not be
reachable before the compressed tuples are committed, and the compression must
not happen before the raw tuples are bound. Both are refused.

Negative: forged reciprocals, a wrong multiplicity, and a query outside the
table are each rejected.

Run:  .venv/bin/python layergkr/tests/run_tests.py test_logup
"""
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from layergkr import logup, rs
from layergkr.transcript import Transcript
from prover.protocol import P as FIELD_P

CFG = rs.Config(ELL=16, K_DEG=32, N_LIG=64, T_QUERIES=4)
TABLE_ID = 3


def _silu_table(n=8):
    """A stand-in unary table: (input, output) pairs."""
    return [(x, (x * x + 1) % FIELD_P) for x in range(n)]


def _instance(seed=5, n_queries=6, table_n=8, bad_query=False):
    r = random.Random(seed)
    table = _silu_table(table_n)
    qs = [table[r.randrange(table_n)] for _ in range(n_queries)]
    if bad_query:
        qs[0] = (999, 999)
    return table, qs


def _run(table, qs, tr=None):
    tr = tr or Transcript()
    lu = logup.LogUp(CFG, TABLE_ID, qs, table)
    lu.commit_raw(tr)
    beta = tr.coin("beta")[0]
    lu.compress(beta, tr)
    alpha = tr.coin("alpha")[0]
    lu.build(alpha)
    proof = lu.prove(tr)
    return lu, proof, beta, alpha


# ── positive ─────────────────────────────────────────────────────────────────
def test_honest_lookup_verifies():
    table, qs = _instance()
    lu, proof, beta, alpha = _run(table, qs)
    ok, why = logup.verify(proof, lu.q_vals, lu.t_vals, lu.mult, alpha)
    assert ok, why


def test_identity_sides_agree():
    """Use the NORMALISED query/table lists (LogUp pads both to a power of two
    before committing); mixing padded multiplicities with unpadded lists would
    compare two different instances."""
    table, qs = _instance()
    lu, proof, beta, alpha = _run(table, qs)
    lhs, rhs = logup.identity_sides(lu.queries, lu.table, lu.mult, beta, alpha, TABLE_ID)
    assert lhs == rhs, "logup identity does not hold on an honest instance"


def test_padding_does_not_change_the_lookup_semantics():
    """Query padding repeats a real entry and is counted; table padding is
    sentinel tuples with multiplicity zero. So the padded instance proves the
    same statement as the unpadded one."""
    table = _silu_table(6)          # 6 -> padded to 8
    qs = [table[2], table[5], table[2]]   # 3 -> padded to 4
    lu, proof, beta, alpha = _run(table, qs)
    assert len(lu.queries) == 4 and len(lu.table) == 8
    assert sum(lu.mult) == len(lu.queries)
    for k in range(6, 8):
        assert lu.mult[k] == 0, "a sentinel table row was queried"
    ok, why = logup.verify(proof, lu.q_vals, lu.t_vals, lu.mult, alpha)
    assert ok, why


def test_multiplicities_are_counted():
    table = _silu_table(4)
    qs = [table[1], table[1], table[1], table[2]]
    lu, proof, beta, alpha = _run(table, qs)
    assert lu.mult == [0, 3, 1, 0], lu.mult
    ok, why = logup.verify(proof, lu.q_vals, lu.t_vals, lu.mult, alpha)
    assert ok, why


def test_all_queries_same_entry():
    table = _silu_table(6)
    qs = [table[4]] * 7
    lu, proof, beta, alpha = _run(table, qs)
    ok, why = logup.verify(proof, lu.q_vals, lu.t_vals, lu.mult, alpha)
    assert ok, why


# ── ordering ─────────────────────────────────────────────────────────────────
def test_compress_before_raw_commit_is_refused():
    table, qs = _instance()
    lu = logup.LogUp(CFG, TABLE_ID, qs, table)
    raised = False
    try:
        lu.compress(12345, Transcript())
    except RuntimeError as e:
        raised = "out of order" in str(e)
    assert raised, "compression ran before the raw tuples were bound"


def test_reciprocals_before_compression_are_refused():
    """alpha must not be usable until R_cmp exists -- otherwise the prover fits
    the lookup witness to the sampled point."""
    table, qs = _instance()
    tr = Transcript()
    lu = logup.LogUp(CFG, TABLE_ID, qs, table)
    lu.commit_raw(tr)
    raised = ""
    try:
        lu.build(tr.coin("alpha")[0])
    except RuntimeError as e:
        raised = str(e)
    assert "after the compressed tuples" in raised, raised


def test_prove_before_build_is_refused():
    table, qs = _instance()
    tr = Transcript()
    lu = logup.LogUp(CFG, TABLE_ID, qs, table)
    lu.commit_raw(tr)
    lu.compress(tr.coin("beta")[0], tr)
    raised = False
    try:
        lu.prove(tr)
    except RuntimeError:
        raised = True
    assert raised


# ── negative ─────────────────────────────────────────────────────────────────
def test_query_outside_the_table_is_rejected():
    table, qs = _instance(bad_query=True)
    raised = False
    try:
        logup.LogUp(CFG, TABLE_ID, qs, table)
    except KeyError:
        raised = True
    assert raised, "a query outside the table was accepted"


def test_forged_reciprocal_fails_the_constraint():
    table, qs = _instance()
    lu, proof, beta, alpha = _run(table, qs)
    proof.r_query[0] = (proof.r_query[0] + 1) % FIELD_P
    ok, why = logup.verify(proof, lu.q_vals, lu.t_vals, lu.mult, alpha)
    assert not ok and "reciprocal" in why, why


def test_wrong_multiplicity_breaks_the_identity():
    table, qs = _instance()
    lu, proof, beta, alpha = _run(table, qs)
    bad = list(lu.mult)
    idx = next(i for i, m in enumerate(bad) if m > 0)
    bad[idx] += 1
    ok, why = logup.verify(proof, lu.q_vals, lu.t_vals, bad, alpha)
    assert not ok, "an inflated multiplicity was accepted"


def test_tampered_compressed_value_breaks_the_identity():
    table, qs = _instance()
    lu, proof, beta, alpha = _run(table, qs)
    bad_q = list(lu.q_vals)
    bad_q[0] = (bad_q[0] + 7) % FIELD_P
    ok, why = logup.verify(proof, bad_q, lu.t_vals, lu.mult, alpha)
    assert not ok and "reciprocal" in why, why
