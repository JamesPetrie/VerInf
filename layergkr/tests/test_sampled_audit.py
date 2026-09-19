"""Protocol/order/binding gates for the 5-of-49 sampled local audit."""
import copy
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from layergkr import rs
from layergkr.sampled_audit import (
    AuditParams,
    AuditStatement,
    LookupBlock,
    MatmulBlock,
    RelationBlock,
    VerifierSession,
    Wire,
    detection_probability,
    prove,
    verify,
)
from prover.protocol import P

CFG = rs.Config(ELL=8, K_DEG=16, N_LIG=64, T_QUERIES=4)
SECRET = b"test-only persistent verifier secret"


def _matmul(index, bad=False):
    x = [[1, 2, 3], [4, 5, 6]]
    w = [[2, 0, 1], [1, 3, 2]]
    y = [[sum(a * b for a, b in zip(row, wr)) for wr in w] for row in x]
    if bad:
        y[0][0] += 1
    return MatmulBlock(index, f"mm{index}", Wire(f"x{index}", x),
                       Wire(f"w{index}", w), Wire(f"y{index}", y))


def _relation(index, bad=False):
    a, b = [2, 3, 4, 5], [7, 8, 9, 10]
    c = [(x * y) % P for x, y in zip(a, b)]
    if bad:
        c[0] += 1
    wires = {"a": Wire(f"a{index}", [a]), "b": Wire(f"b{index}", [b]),
             "c": Wire(f"c{index}", [c])}
    return RelationBlock(index, f"round{index}", "rounding", wires,
                         [(1, ["a", "b"]), (P - 1, ["c"])])


def _lookup(index, bad=False):
    table = [(0,), (1,), (2,), (3,)]
    queries = [[0], [1], [1], [3]]
    if bad:
        queries[0] = [9]
    mult = [1, 2, 0, 1]
    return LookupBlock(index, f"lookup{index}", Wire(f"q{index}", queries),
                       Wire(f"m{index}", [mult]), table, table_id=7)


def _blocks(n=6):
    out = []
    for i in range(n):
        out.append((_matmul, _relation, _lookup)[i % 3](i))
    return out


def _run(blocks, q=4, window=3, sample=1):
    params = AuditParams(len(blocks), window, sample, q)
    statement = AuditStatement.from_blocks(params, b"public input/output", blocks)
    session = VerifierSession(SECRET)
    proof = prove(blocks, CFG, statement, session)
    return statement, session, proof


def test_production_sampling_geometry_and_detection_bound():
    p = AuditParams()
    assert p.selected_blocks == 265
    assert abs(p.audit_fraction - 265 / 2596) < 1e-15
    assert p.min_detection_probability == 5 / 49
    assert detection_probability(p) > 0.10


def test_every_local_proof_family_accepts():
    statement, session, proof = _run(_blocks())
    ok, why = verify(proof, CFG, statement, session)
    assert ok, why


def test_sample_is_after_and_depends_on_window_commitment():
    blocks = _blocks(9)
    statement, session, proof = _run(blocks, window=3, sample=1)
    changed = copy.deepcopy(proof)
    changed.block_commitments[0] = b"x" * 32
    ok, why = verify(changed, CFG, statement, session)
    assert not ok and ("selected" in why or "commitment" in why), why


def test_61_distinct_post_proof_columns():
    blocks = _blocks(3)
    statement, session, proof = _run(blocks, q=61, window=3, sample=1)
    for selected in proof.selected:
        for opening in selected.openings.values():
            assert len(opening.columns) == len(set(opening.columns)) == 61
    ok, why = verify(proof, CFG, statement, session)
    assert ok, why


def test_tampered_c0_opening_rejects():
    statement, session, proof = _run(_blocks())
    opening = next(iter(proof.selected[0].openings.values()))
    opening.values[0][0] = (opening.values[0][0] + 1) % P
    ok, why = verify(proof, CFG, statement, session)
    assert not ok and ("Merkle" in why or "bound to C0" in why), why


def test_tampered_local_messages_reject_before_local_check():
    statement, session, proof = _run(_blocks())
    opening = next(iter(proof.selected[0].openings.values()))
    opening.messages[0][0] = (opening.messages[0][0] + 1) % P
    ok, why = verify(proof, CFG, statement, session)
    assert not ok and "digest" in why, why


def test_first_bad_selected_matmul_is_detected():
    blocks = [_matmul(0, bad=True)]
    statement, session, proof = _run(blocks, window=1, sample=1)
    ok, why = verify(proof, CFG, statement, session)
    assert not ok and "Freivalds" in why, why


def test_selected_bad_sumcheck_is_detected():
    blocks = [_relation(0, bad=True)]
    statement, session, proof = _run(blocks, window=1, sample=1)
    ok, why = verify(proof, CFG, statement, session)
    assert not ok and "sumcheck" in why, why


def test_selected_bad_lookup_is_detected():
    blocks = [_lookup(0, bad=True)]
    statement, session, proof = _run(blocks, window=1, sample=1)
    ok, why = verify(proof, CFG, statement, session)
    assert not ok and "lookup" in why, why


def test_wire_rebinding_is_refused():
    first = _matmul(0)
    second = _matmul(1)
    second.x.wire_id = first.x.wire_id
    second.x.messages[0][0] += 1
    blocks = [first, second]
    params = AuditParams(2, 2, 1, 4)
    statement = AuditStatement.from_blocks(params, b"io", blocks)
    try:
        prove(blocks, CFG, statement, VerifierSession(SECRET))
    except ValueError as e:
        assert "rebound" in str(e)
    else:
        raise AssertionError("a global wire id was rebound to another witness")


def test_public_manifest_cannot_be_replaced():
    blocks = _blocks(3)
    statement, session, proof = _run(blocks, window=3, sample=1)
    other = list(statement.descriptors)
    other[0] = dict(other[0], name="attacker manifest")
    bad_statement = AuditStatement(statement.params, statement.public_io_digest, tuple(other))
    ok, why = verify(proof, CFG, bad_statement, session)
    assert not ok and "statement" in why, why
