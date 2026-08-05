"""Gates for one layer proof and the composition lemma (layer.py, doc §8.1).

Positive: a single layer verifies; a 3-layer chain verifies; each layer's input
root is the previous output root byte-for-byte.

Negative:
  * splice a foreign root between layers          -> chain rejects
  * tamper the weight root a layer is checked against -> weight seam rejects
  * tamper a revealed small vector                -> binding rejects
  * tamper an opened column value                 -> merkle rejects
  * feed the next layer a different input state   -> chain rejects

Run:  .venv/bin/python layergkr/tests/run_tests.py test_layer
"""
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from layergkr import layer as lg, projection as pj, rs
from prover.protocol import P as FIELD_P

CFG = rs.Config(ELL=16, K_DEG=32, N_LIG=128, T_QUERIES=8)
S, N_IN, N_OUT, Q = 3, 8, 4, 8


def _layer(X, seed):
    r = random.Random(seed)
    W = [[r.randrange(FIELD_P) for _ in range(N_IN)] for _ in range(N_OUT)]
    pad = [[r.randrange(FIELD_P) for _ in range(CFG.ELL - N_IN)] for _ in range(N_OUT)]
    return lg.Layer(CFG, pj.PersistentWeights.enroll(CFG, W, pad), X)


def _stack(n=3, seed=41):
    """n layers with n_out == n_in so the output state feeds the next input."""
    r = random.Random(seed)
    X = [[r.randrange(FIELD_P) for _ in range(N_IN)] for _ in range(S)]
    layers = []
    for l in range(n):
        rr = random.Random(seed + l)
        W = [[rr.randrange(FIELD_P) for _ in range(N_IN)] for _ in range(N_IN)]
        pad = [[rr.randrange(FIELD_P) for _ in range(CFG.ELL - N_IN)]
               for _ in range(N_IN)]
        lay = lg.Layer(CFG, pj.PersistentWeights.enroll(CFG, W, pad), X)
        lay.compute()
        layers.append(lay)
        X = lay.Y                        # output state becomes the next input
    return layers


# ── positive ─────────────────────────────────────────────────────────────────
def test_single_layer_verifies():
    r = random.Random(2)
    X = [[r.randrange(FIELD_P) for _ in range(N_IN)] for _ in range(S)]
    lay = _layer(X, seed=9)
    proofs = lg.prove_chain([lay], Q)
    ok, why = lg.verify_layer(CFG, lay.weights.root, proofs[0], N_IN)
    assert ok, why


def test_chain_verifies_and_roots_link():
    layers = _stack(3)
    proofs = lg.prove_chain(layers, Q)
    roots = [l.weights.root for l in layers]
    ok, why = lg.verify_chain(CFG, roots, proofs, [N_IN] * 3)
    assert ok, why
    for l in range(1, 3):
        assert proofs[l].in_root == proofs[l - 1].out_root, f"link {l} broken"


def test_contraction_identity_is_what_the_sumcheck_claims():
    """The sumcheck's claim equals the tau/rho-weighted output computed directly
    -- prove_layer asserts it internally, this pins it from outside too."""
    layers = _stack(1)
    proofs = lg.prove_chain(layers, Q)
    p, lay = proofs[0], layers[0]
    direct = 0
    for t in range(S):
        for i in range(N_IN):
            direct = (direct + p.tau[t] * p.rho[i] * lay.Y[t][i]) % FIELD_P
    assert p.sumcheck.claim == direct


# ── negative ─────────────────────────────────────────────────────────────────
def test_spliced_root_is_rejected():
    layers = _stack(3)
    proofs = lg.prove_chain(layers, Q)
    roots = [l.weights.root for l in layers]
    proofs[1].in_root = b"\x42" * 32
    ok, why = lg.verify_chain(CFG, roots, proofs, [N_IN] * 3)
    assert not ok, "a spliced input root was accepted"
    assert "seam" in why or "splice" in why, why


def test_wrong_weight_root_is_rejected():
    layers = _stack(2)
    proofs = lg.prove_chain(layers, Q)
    roots = [layers[1].weights.root, layers[0].weights.root]      # swapped
    ok, why = lg.verify_chain(CFG, roots, proofs, [N_IN] * 2)
    assert not ok and "weight seam" in why, why


def test_tampered_small_vector_is_rejected():
    layers = _stack(1)
    proofs = lg.prove_chain(layers, Q)
    proofs[0].p_blocks[0][0] = (proofs[0].p_blocks[0][0] + 1) % FIELD_P
    ok, why = lg.verify_layer(CFG, layers[0].weights.root, proofs[0], N_IN)
    assert not ok and "does not match its root" in why, why


def test_tampered_opened_column_is_rejected():
    layers = _stack(1)
    proofs = lg.prove_chain(layers, Q)
    proofs[0].w_opening.w_values[0][0] = \
        (proofs[0].w_opening.w_values[0][0] + 1) % FIELD_P
    ok, why = lg.verify_layer(CFG, layers[0].weights.root, proofs[0], N_IN)
    assert not ok and "merkle" in why, why


def test_layer_proved_on_a_different_input_state_breaks_the_chain():
    """Prove layer 1 on an input state that is NOT layer 0's output: the layer
    proof itself is fine, but the composition link fails -- which is exactly what
    §8.1 says the root equality is for."""
    layers = _stack(2)
    r = random.Random(77)
    layers[1].X = [[r.randrange(FIELD_P) for _ in range(N_IN)] for _ in range(S)]
    layers[1].out_commit = None
    layers[1].compute()
    proofs = lg.prove_chain(layers, Q)
    roots = [l.weights.root for l in layers]
    ok0, _ = lg.verify_layer(CFG, roots[1], proofs[1], N_IN)
    assert ok0, "the standalone layer proof should still be valid"
    ok, why = lg.verify_chain(CFG, roots, proofs, [N_IN] * 2)
    assert not ok and "splice" in why, why
