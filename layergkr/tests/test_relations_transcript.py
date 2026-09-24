"""The batched gate sumcheck's coins are recomputed by the verifier, one per
round after its polynomial (review 2026-09-23 follow-up to finding 3).

prove_batch drew every round coin from the layer transcript up front, and
verify_batch checked the proof against its own challenges; a prover could fit
the round polynomials to known coins and walk a false zero claim onto the
true terminal value."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from layergkr import relations as rel
from layergkr import sumcheck as sc
from layergkr.logup import eq_vector
from layergkr.transcript import Transcript
from prover.protocol import P


def _gates(c):
    return [rel.hadamard("n0", [2, 3, 4, 5], [7, 8, 9, 10], c)]


def test_honest_batch_verifies():
    gates = _gates([14, 24, 36, 50])
    proof, z, lambdas = rel.prove_batch(gates, Transcript(b"t"))
    ok, why = rel.verify_batch(proof, gates, z, lambdas)
    assert ok, why


def test_known_coins_do_not_verify():
    gates = _gates([14, 25, 36, 50])                # one wrong product
    tr = Transcript(b"t")
    lambdas = tr.coin("gates_batch", 1)
    z = tr.coin("gates_z", 2)
    coins = tr.coin("gates_sc", 2)                  # the old up-front schedule
    eq = eq_vector(z)
    terms = [((lambdas[0] * c) % P, [eq] + list(f)) for c, f in gates[0].terms]
    target = 0
    for c, fs in terms:
        v = c
        for f in fs:
            v = v * sc.mle_eval(f, coins) % P
        target = (target + v) % P
    polys, current = [], 0
    for i, r in enumerate(coins):
        nxt = target if i == 1 else 0
        a = (r * current - nxt) * pow((2 * r - 1) % P, P - 2, P) % P
        polys.append([(x, (a + (current - 2 * a) * x) % P) for x in range(4)])
        current = nxt
    forged = sc.SumcheckProof(claim=0, round_polys=polys, challenges=coins,
                              final_point=coins, n_terms=len(terms))
    ok, why = rel.verify_batch(forged, gates, z, lambdas)
    assert not ok and "challenge mismatch" in why, why


def test_replaced_challenges_reject():
    gates = _gates([14, 24, 36, 50])
    proof, z, lambdas = rel.prove_batch(gates, Transcript(b"t"))
    proof.challenges = [(c + 1) % P for c in proof.challenges]
    proof.final_point = list(proof.challenges)
    ok, why = rel.verify_batch(proof, gates, z, lambdas)
    assert not ok and "challenge mismatch" in why, why
