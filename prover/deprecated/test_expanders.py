"""Diff-test: each row-expander turns its small params into exactly the
(slot, cid, coef) terms you'd write by hand. Torch-free, no prover — drives the
expanders DIRECTLY (the trust-critical primitives), independent of any claim
model. The end-to-end "expanders match the real prover" check lives in
test_compile_parity.py (on the Spark); this pins the expanders' local semantics.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # pipeline/ on path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[0]))  # deprecated/ (python_verifier_compile)
import protocol as pr
import python_verifier_compile as pvc

cfg = pr.Config(ELL=4, K_DEG=8, N_LIG=16, T_QUERIES=3)
ch2 = 999


def accumulate(emissions):
    """Fold a list of (row, fn, params) into r^T A = {(row,col,cid): Σ S·coef}."""
    R = {}
    for row, fn, params in emissions:
        for col, cid, coef in fn(params, cfg):
            assert 0 <= col < cfg.ELL, f"slot {col} out of row at row {row}"
            key = (row, col)
            R[key] = pr.add(R.get(key, 0), pr.mul(pr.challenge(ch2, cid, "lin"), coef))
    return {k: v for k, v in R.items() if v}


# ---- expand_identity: a length-4 Copy dst[s] − src[s] = 0, cids [0,4) ----
# dst at row 4 (coef +1), src at row 3 (coef −1), one row each (n=4 = ELL).
emit = [(4, pvc.expand_identity, (0, 4, 1)),
        (3, pvc.expand_identity, (0, 4, pr.P - 1))]
ref = {}
for s in range(4):
    S = pr.challenge(ch2, s, "lin")
    ref[(4, s)] = pr.mul(S, 1)
    ref[(3, s)] = pr.mul(S, pr.P - 1)
ref = {k: v for k, v in ref.items() if v}
assert accumulate(emit) == ref, "expand_identity != hand reference"
print("expand_identity: Copy r^T A matches reference  ok")

# ---- expand_rope: x_rot[o] − ca·x[a[o]] − cb·x[b[o]] = 0, cids [0,4), one row ----
# Single-row x (n=4 = ELL) at row 5, x_rot at row 6. a/b/ca/cb are public terms.
a  = (0, 1, 0, 1)            # pair lo-halves
b  = (2, 3, 2, 3)            # pair hi-halves
ca = (11, 22, 33, 44)
cb = (55, 66, 77, 88)
neg_ca = tuple(pr.sub(0, c) for c in ca)
neg_cb = tuple(pr.sub(0, c) for c in cb)
emit = [(6, pvc.expand_identity, (0, 4, 1)),                       # x_rot side
        (5, pvc.expand_rope, (0, a, b, neg_ca, neg_cb, 0, 4))]     # x side (row window [0,4))
ref = {}
for o in range(4):
    S = pr.challenge(ch2, o, "lin")
    ref[(6, o)]    = pr.add(ref.get((6, o), 0), pr.mul(S, 1))
    ref[(5, a[o])] = pr.add(ref.get((5, a[o]), 0), pr.mul(S, neg_ca[o]))
    ref[(5, b[o])] = pr.add(ref.get((5, b[o]), 0), pr.mul(S, neg_cb[o]))
ref = {k: v for k, v in ref.items() if v}
assert accumulate(emit) == ref, "expand_rope != hand reference"
print("expand_rope: rotation r^T A matches reference  ok")

print("EXPANDER DIFF-TEST PASSED (expand_identity + expand_rope, direct)")
