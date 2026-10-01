"""The runtime's local relation sumcheck takes one coin per round, after that
round's polynomial (CPU; review 2026-09-23 findings 3 and 4).

Every round coin used to be expanded from the block challenge before the
prover built any polynomial; the shared checker also accepted a proof-chosen
terminal point and a proof-supplied mask. The portable audit's end-to-end
counterexamples are in layergkr/tests/test_sampled_audit.py; these lock the
runtime's entrypoints (sampled_local_proofs) the same way."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch

import protocol
import sampled_local_proofs as slp
from layergkr import sumcheck as sc
from layergkr.logup import eq_vector

P = protocol.P
CHALLENGE = b"\x5a" * 32
A, B = [2, 3, 4, 5], [7, 8, 9, 10]


def _terms(c, as_tensor):
    eq = eq_vector(protocol.op_vec(CHALLENGE, 0, "test-eq", 2))
    vec = (lambda v: torch.tensor(v, dtype=torch.uint64)) if as_tensor else list
    return [(1, [vec(A), vec(B), vec(eq)]), (P - 1, [vec(c), vec(eq)])]


def _verify(proof, c):
    return slp._verify_explicit_relation(proof, "test-product", _terms(c, False),
                                         claim_index=3, challenge=CHALLENGE)


def _prove(c):
    return slp._prove_explicit_relation("test-product", _terms(c, True),
                                        claim_index=3, challenge=CHALLENGE)


def _terminal(terms, point):
    total = 0
    for coef, factors in terms:
        v = coef
        for f in factors:
            v = v * sc.mle_eval(f, point) % P
        total = (total + v) % P
    return total


def test_honest_relation_verifies():
    good = [14, 24, 36, 50]
    ok, why = _verify(_prove(good), good)
    assert ok, why


def test_coins_known_before_the_polynomials_do_not_verify():
    bad = [14, 25, 36, 50]
    terms = _terms(bad, False)
    # the old schedule: every round coin from the challenge, up front
    coins = protocol.op_vec(CHALLENGE, 3, "sampled-sumcheck-round", 2)
    target, current, polys = _terminal(terms, coins), 0, []
    for i, r in enumerate(coins):
        nxt = target if i == 1 else 0
        a = (r * current - nxt) * pow((2 * r - 1) % P, P - 2, P) % P
        polys.append([(x, (a + (current - 2 * a) * x) % P) for x in range(4)])
        current = nxt
    forged = slp.RelationSumcheckProof(3, CHALLENGE, "test-product", sc.SumcheckProof(
        claim=0, round_polys=polys, challenges=coins, final_point=coins, n_terms=2))
    ok, why = _verify(forged, bad)
    assert not ok and "challenge mismatch" in why, why


def _zero_rounds():
    tr = slp._sumcheck_transcript(CHALLENGE, 3, "test-product")
    polys = [[(x, 0) for x in range(4)] for _ in range(2)]
    return polys, [sc.draw_coin(tr, r, polys[r]) for r in range(2)]


def test_a_proof_chosen_terminal_point_rejects():
    bad = [14, 25, 36, 50]
    polys, coins = _zero_rounds()
    forged = slp.RelationSumcheckProof(3, CHALLENGE, "test-product", sc.SumcheckProof(
        claim=0, round_polys=polys, challenges=coins, final_point=[0, 0], n_terms=2))
    ok, why = _verify(forged, bad)
    assert not ok and "terminal point" in why, why


def test_a_masked_transcript_rejects():
    bad = [14, 25, 36, 50]
    polys, coins = _zero_rounds()
    forged = slp.RelationSumcheckProof(3, CHALLENGE, "test-product", sc.SumcheckProof(
        claim=0, round_polys=polys, challenges=coins, final_point=coins, n_terms=2,
        masked=True, final_mask=(-_terminal(_terms(bad, False), coins)) % P))
    ok, why = _verify(forged, bad)
    assert not ok and "masked" in why, why


def test_a_round_coin_cannot_precede_its_polynomial():
    tr = sc.RoundTranscript(b"ctx")
    try:
        tr(0)
    except ValueError:
        pass
    else:
        raise AssertionError("a coin was drawn before its round was absorbed")
