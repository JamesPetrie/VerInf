"""Gates for sumcheck.py: the sumcheck itself, and the §7 ZK mask compiler.

Positive: honest proofs verify, bare and masked, for degree-2 and degree-3
instances; a masked proof carries the same claim structure as the bare one.

Hiding: the tape -> round-polynomial map is inverted, showing every polynomial in
the affine space {p : p(0)+p(1) = masked claim} is produced by exactly one tape.
That is the doc's triangular-full-rank claim, executed.

Negative: a wrong claim, an edited round polynomial, a mismatched challenge and a
tampered terminal are each rejected -- masked and bare alike, which is the point
of the 'subtract the authenticated masks' soundness argument.

Run:  .venv/bin/python layergkr/tests/run_tests.py test_sumcheck
"""
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from layergkr import sumcheck as sc
from prover.protocol import P as FIELD_P

N_VARS = 4


def _factors(k=2, seed=17):
    r = random.Random(seed)
    return [[r.randrange(FIELD_P) for _ in range(1 << N_VARS)] for _ in range(k)]


def _coin(seed=99):
    r = random.Random(seed)
    vals = [r.randrange(FIELD_P) for _ in range(N_VARS)]
    return lambda i: vals[i]


def _tape(n, seed=5):
    r = random.Random(seed)
    return sc.MaskTape([r.randrange(FIELD_P) for _ in range(n)])


# ── positive ─────────────────────────────────────────────────────────────────
def test_bare_sumcheck_verifies():
    f = _factors(2)
    proof = sc.prove(f, _coin())
    ok, why = sc.verify(proof, f, _coin())
    assert ok, why


def test_degree_three_sumcheck_verifies():
    f = _factors(3, seed=23)
    proof = sc.prove(f, _coin())
    ok, why = sc.verify(proof, f, _coin())
    assert ok, why


def test_masked_sumcheck_verifies_and_hides_the_claim():
    f = _factors(2)
    mu0 = 123456789
    bare = sc.prove(f, _coin())
    masked = sc.prove(f, _coin(), tape=_tape(2 * N_VARS), mu0=mu0)
    ok, why = sc.verify(masked, f, _coin())
    assert ok, why
    assert masked.claim == (bare.claim + mu0) % FIELD_P
    # the transmitted polynomials genuinely differ from the bare ones
    assert masked.round_polys[0] != bare.round_polys[0]


def test_mask_polynomial_splits_the_carried_mask():
    """h(0) + h(1) = mu, for every degree."""
    r = random.Random(4)
    for d in (1, 2, 3, 4):
        mu = r.randrange(FIELD_P)
        u = [r.randrange(FIELD_P) for _ in range(d)]
        h = sc.mask_poly_coeffs(mu, u)
        assert (sc.poly_eval_coeffs(h, 0) + sc.poly_eval_coeffs(h, 1)) % FIELD_P == mu


def test_tape_to_polynomial_map_is_a_bijection_onto_the_affine_space():
    """Hiding, checked: pick any polynomial in the affine space cut out by the
    masked claim; exactly one tape produces it. So a uniform tape gives a uniform
    transcript polynomial, revealing nothing beyond the previous masked claim."""
    r = random.Random(8)
    d = 3
    mu = r.randrange(FIELD_P)
    for _ in range(20):
        # a target in the space: free top coefficients, a_0 forced by the claim
        top = [r.randrange(FIELD_P) for _ in range(d)]
        target = sc.mask_poly_coeffs(mu, top)
        u = sc.solve_tape_for_target(mu, target)
        assert u == [c % FIELD_P for c in top], "map is not injective/onto as claimed"
    # a polynomial outside the space has no preimage
    bad = list(sc.mask_poly_coeffs(mu, [1, 2, 3]))
    bad[0] = (bad[0] + 1) % FIELD_P
    assert sc.solve_tape_for_target(mu, bad) is None


def test_masked_product_identity():
    """§7.2: Z - c = XY - Xb - Ya + ab for the carried masks."""
    r = random.Random(12)
    for _ in range(50):
        x, y, a, b, c = (r.randrange(FIELD_P) for _ in range(5))
        assert sc.masked_product_ok(x, y, a, b, c)


# ── negative ─────────────────────────────────────────────────────────────────
def test_wrong_claim_is_rejected():
    f = _factors(2)
    proof = sc.prove(f, _coin())
    proof.claim = (proof.claim + 1) % FIELD_P
    ok, why = sc.verify(proof, f, _coin())
    assert not ok and "claim" in why, why


def test_edited_round_polynomial_is_rejected():
    f = _factors(2)
    proof = sc.prove(f, _coin())
    x, y = proof.round_polys[1][0]
    proof.round_polys[1][0] = (x, (y + 1) % FIELD_P)
    ok, why = sc.verify(proof, f, _coin())
    assert not ok, why


def test_replayed_challenge_is_rejected():
    f = _factors(2)
    proof = sc.prove(f, _coin())
    proof.challenges[2] = (proof.challenges[2] + 1) % FIELD_P
    ok, why = sc.verify(proof, f, _coin())
    assert not ok and "challenge" in why, why


def test_tampered_terminal_is_rejected_masked_and_bare():
    """The masks must not buy the prover anything: the same lie fails on both
    paths, which is the 'subtract the authenticated masks' argument in action."""
    f = _factors(2)
    for tape, mu0 in ((None, 0), (_tape(2 * N_VARS), 4242)):
        proof = sc.prove(f, _coin(), tape=tape, mu0=mu0)
        bad = [list(f[0]), list(f[1])]
        bad[0][3] = (bad[0][3] + 1) % FIELD_P
        ok, why = sc.verify(proof, bad, _coin())
        assert not ok and "terminal" in why, (tape is not None, why)


def test_exhausted_tape_raises():
    f = _factors(2)
    raised = False
    try:
        sc.prove(f, _coin(), tape=_tape(2), mu0=1)   # needs 2 per round x 4 rounds
    except RuntimeError:
        raised = True
    assert raised, "short tape silently accepted"
