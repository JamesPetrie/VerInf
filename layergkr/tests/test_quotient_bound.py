"""Regression cases for protocol review F12 (25 September 2026).

The rescale and bracket gates state raw = n*q + r in the field and range r to
[0, n). The quotient q was left free, and n is invertible mod P, so every r in
[0, n) had a field q: the review's example is raw = 0, n = 16, satisfied by
(q, r) = (0, 0) and by (-1/16, 1). A bracket's r is the next lookup key, so the
freedom reached nonlinear outputs.

The repair (semantics.quotient_words, _Builder._bound_quotient) ranges q by
words of the range table, below 2^(wb*k) with nb + wb*k <= 63, so n*q + r stays
below P and the field equation is an integer identity. These tests keep the
counterexample as a case that must now fail.

Run:  .venv/bin/python layergkr/tests/run_tests.py test_quotient_bound
"""
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from prover.protocol import P as FIELD_P

from layergkr import relations as rel, semantics as sem


def _vanishes(gate: rel.Gate, idx: int) -> bool:
    acc = 0
    for coeff, factors in gate.terms:
        prod = coeff % FIELD_P
        for f in factors:
            prod = prod * (f[idx] % FIELD_P) % FIELD_P
        acc = (acc + prod) % FIELD_P
    return acc == 0


def _field_quotient(raw: int, r: int, n: int) -> int:
    return (raw - r) * pow(n, -1, FIELD_P) % FIELD_P


def test_review_counterexample_satisfies_the_bare_gate():
    """The defect itself: without the quotient bound both witnesses pass."""
    n = 16
    alt_q = (-pow(n, -1, FIELD_P)) % FIELD_P
    gate = rel.rescale("bare", [0, 0], [0, alt_q], [0, 1], n)
    assert _vanishes(gate, 0) and _vanishes(gate, 1)


def test_the_counterexample_quotient_has_no_ranged_words():
    n_bits, wb = 4, 6
    k = sem.quotient_words(n_bits, wb)
    alt_q = (-pow(1 << n_bits, -1, FIELD_P)) % FIELD_P
    assert alt_q >= 1 << (wb * k), "the review's quotient fits the word range"


def test_a_ranged_quotient_leaves_exactly_one_remainder():
    """For every raw the bound admits, exactly one r in [0, n) has a field
    quotient inside the word range, and it is raw mod n."""
    rng = random.Random(12)
    for n_bits, wb in ((4, 6), (6, 6), (6, 10), (10, 10)):
        n, k = 1 << n_bits, sem.quotient_words(n_bits, wb)
        limit = 1 << (wb * k)
        assert n_bits + wb * k <= 63 < FIELD_P.bit_length()
        raws = [0, 1, n - 1, n, n * limit - 1] + [rng.randrange(n * limit)
                                                   for _ in range(40)]
        for raw in raws:
            ok = [r for r in range(n) if _field_quotient(raw, r, n) < limit]
            assert ok == [raw % n], f"n=2^{n_bits} wb={wb} raw={raw}: {ok}"
        # and a raw past the bound has no ranged decomposition at all
        raw = n * limit
        assert not [r for r in range(n) if _field_quotient(raw, r, n) < limit]


def _trace():
    cfg = sem.ToyConfig(S=3, d=8, d_ff=16, E=2, table_bits=6, scale_bits=6)
    return cfg, sem.forward(cfg, random.Random(5))


def test_honest_trace_carries_a_quotient_bound_for_every_rescale_and_bracket():
    cfg, tr = _trace()
    ok, why = sem.check_trace(tr)
    assert ok, why
    kinds = [g.node_id.split("#")[0] for g in tr.gates]
    for i, g in enumerate(tr.gates):
        if g.kind == "rescale":
            assert kinds[i + 1] == "qbound", f"{g.node_id} has no quotient bound"
            assert tr.gates[i + 1].terms[-1][1][0] == g.terms[1][1][0], \
                f"{g.node_id}: the bounded quotient is not the gate's"


def test_alternative_bracket_witness_is_rejected():
    """The review's move on a real trace: shift a bracket's low word by one and
    take the field quotient that keeps the bracket gate satisfied. The bracket
    gate still vanishes; the quotient bound does not, whether the words are
    kept or recomputed from the new quotient."""
    cfg, tr = _trace()
    i = next(j for j, g in enumerate(tr.gates) if g.node_id.startswith("bracket"))
    br, qb = tr.gates[i], tr.gates[i + 1]
    n, wb = cfg.table_size, cfg.table_bits
    raw, hi, lo = (br.terms[0][1][0], br.terms[1][1][0], br.terms[2][1][0])
    idx = 0
    lo_alt = (lo[idx] + 1) % n
    hi_alt = _field_quotient(raw[idx], lo_alt, n)
    hi[idx], lo[idx] = hi_alt, lo_alt
    assert _vanishes(br, idx), "the shifted witness should satisfy the bare gate"

    q_vec = qb.terms[-1][1][0]
    q_vec[idx] = hi_alt
    assert not _vanishes(qb, idx), "the quotient bound accepted the old words"
    words = [t[1][0] for t in qb.terms[:-1]]
    for j, w in enumerate(words):
        w[idx] = (hi_alt >> (wb * j)) & ((1 << wb) - 1)
    assert not _vanishes(qb, idx), "a ranged recomposition reached the field quotient"
    ok, _ = sem.check_trace(tr)
    assert not ok


def test_python_path_refuses_a_raw_past_the_bound():
    cfg = sem.ToyConfig(S=1, d=8, d_ff=16, E=1, table_bits=6, scale_bits=6)
    b = sem._Builder(cfg, sem.build_tables(cfg.table_bits, cfg.scale))
    k = sem.quotient_words(cfg.scale_bits, cfg.table_bits)
    b.rescale([(1 << (cfg.scale_bits + cfg.table_bits * k)) - 1])
    try:
        b.rescale([1 << (cfg.scale_bits + cfg.table_bits * k)])
    except sem.RangeOverflow as e:
        assert ">=" in str(e) and "2^" in str(e), f"unhelpful message: {e}"
        return
    raise AssertionError("a quotient past the word range was accepted")
