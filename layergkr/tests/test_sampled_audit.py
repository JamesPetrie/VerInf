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


# -- review 2026-09-23 findings 3, 4 and 7: the relation's local sumcheck ----
# One elementwise-product block, every block selected, 61 RS columns; each
# forgery replaces only the local proof the prover sends, and is built with
# the verifier's own coins where it can be, so each test fails for one reason.
from layergkr import sampled_audit as audit
from layergkr import sumcheck as sc


def _product_block(c):
    return RelationBlock(0, "elementwise-product", "rounding", {
        "a": Wire("a", [[2, 3, 4, 5]]),
        "b": Wire("b", [[7, 8, 9, 10]]),
        "c": Wire("c", [c]),
    }, [(1, ["a", "b"]), (P - 1, ["c"])])


def _audit(block, producer=None):
    params = AuditParams(1, 1, 1, 61)
    statement = AuditStatement.from_blocks(params, b"fixed public IO", [block])
    verifier = VerifierSession(b"private verifier entropy for testing")
    original = audit._prove_local
    if producer is not None:
        audit._prove_local = producer
    try:
        proof = prove([block], CFG, statement, verifier)
    finally:
        audit._prove_local = original
    return verify(proof, CFG, statement, verifier)


def _terms(block, challenge):
    return audit._relation_terms(block.terms,
                                 {r: w.messages for r, w in block.wires.items()},
                                 challenge)


def _terminal(terms, point):
    total = 0
    for coef, factors in terms:
        v = coef
        for f in factors:
            v = v * sc.mle_eval(f, point) % P
        total = (total + v) % P
    return total


def _zero_rounds(block, challenge, block_com):
    """Round polynomials that are identically zero (the claim is 0), with the
    coins the verifier's transcript would draw for them."""
    tr = audit._relation_transcript(challenge, block.index, block_com)
    polys, coins = [], []
    for rnd in range(2):
        samples = [(x, 0) for x in range(4)]          # degree 3 with eq
        polys.append(samples)
        coins.append(sc.draw_coin(tr, rnd, samples))
    return polys, coins


def test_honest_relation_accepts_and_a_single_error_rejects():
    assert _audit(_product_block([14, 24, 36, 50]))[0]
    ok, why = _audit(_product_block([14, 25, 36, 50]))
    assert not ok and "not zero" in why, why


def test_cancelling_elementwise_errors_reject():
    # finding 7: +1 and -1 at two coordinates summed to zero before the eq weight
    ok, why = _audit(_product_block([15, 23, 36, 50]))
    assert not ok and "not zero" in why, why


def test_round_coins_known_in_advance_do_not_verify():
    # finding 3: with every coin expanded from the block challenge up front,
    # linear round polynomials walk a false zero claim onto the true terminal
    def known_coins(block, challenge, block_com):
        terms = _terms(block, challenge)
        coins = audit._field_stream(challenge, b"sumcheck", 2)
        target, current, polys = _terminal(terms, coins), 0, []
        for i, r in enumerate(coins):
            nxt = target if i == len(coins) - 1 else 0
            a = (r * current - nxt) * pow((2 * r - 1) % P, P - 2, P) % P
            slope = (current - 2 * a) % P
            polys.append([(x, (a + slope * x) % P) for x in range(4)])
            current = nxt
        return audit.RelationLocalProof(sc.SumcheckProof(
            claim=0, round_polys=polys, challenges=coins, final_point=coins,
            n_terms=len(terms)))
    ok, why = _audit(_product_block([14, 25, 36, 50]), known_coins)
    assert not ok and "challenge mismatch" in why, why


def test_a_proof_chosen_terminal_point_rejects():
    # finding 4: zero rounds, the verifier's own coins, and a terminal point at
    # a vertex where the residual vanishes
    def moved_point(block, challenge, block_com):
        polys, coins = _zero_rounds(block, challenge, block_com)
        return audit.RelationLocalProof(sc.SumcheckProof(
            claim=0, round_polys=polys, challenges=coins, final_point=[0, 0],
            n_terms=2))
    ok, why = _audit(_product_block([14, 25, 36, 50]), moved_point)
    assert not ok and "terminal point" in why, why


def test_an_unauthenticated_mask_rejects():
    # finding 4: the right point and coins, the terminal cancelled by a mask
    def masked(block, challenge, block_com):
        polys, coins = _zero_rounds(block, challenge, block_com)
        return audit.RelationLocalProof(sc.SumcheckProof(
            claim=0, round_polys=polys, challenges=coins, final_point=coins,
            n_terms=2, masked=True,
            final_mask=(-_terminal(_terms(block, challenge), coins)) % P))
    ok, why = _audit(_product_block([14, 25, 36, 50]), masked)
    assert not ok and "masked" in why, why
