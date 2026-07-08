"""Cross-row RoPE: every rotation pair x[a]↔x[b] straddles a row boundary
(ELL=2, the head spans rows 3,4). Each row's expand_rope emits only the terms
whose source falls in its window [row_lo,row_hi); the two halves share one cid.
r^T A must still match a reference that routes each term to its row by global
slot — i.e. shared-cid-across-rows works, no row-aligned layout required.
Torch-free; drives expand_rope directly per row."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # pipeline/ on path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[0]))  # deprecated/ (python_verifier_compile)
import protocol as pr
import python_verifier_compile as pvc

cfg = pr.Config(ELL=2, K_DEG=4, N_LIG=8, T_QUERIES=2)
ch2 = 77
ell = cfg.ELL

# x spans rows 3,4 (n=4 > ELL=2); x_rot spans rows 5,6.
x_row, xr_row = 3, 5
a  = (0, 1, 0, 1)            # lo-halves — all in row 3 (slot < ELL=2)
b  = (2, 3, 2, 3)            # hi-halves — all in row 4 (slot ≥ ELL)
ca = (11, 22, 33, 44)
cb = (55, 66, 77, 88)
neg_ca = tuple(pr.sub(0, c) for c in ca)
neg_cb = tuple(pr.sub(0, c) for c in cb)
assert all(av < ell for av in a) and all(bv >= ell for bv in b), (a, b)  # confirm straddle

m_total = 7
rows = [[] for _ in range(m_total)]
# x_rot side: identity over n=4, sliced across rows 5,6 (cids [0,4)).
pvc._emit_id(rows, type("V", (), {"row_start": xr_row, "length": 4})(), 0, 1, cfg)
# x side: one offset-aware expand_rope per x-row, window [ro·ELL, ro·ELL+ELL).
for ro in range(2):
    lo = ro * ell
    rows[x_row + ro].append((pvc.expand_rope, (0, a, b, neg_ca, neg_cb, lo, lo + ell)))

R = [[0] * ell for _ in range(m_total)]
for i, exps in enumerate(rows):
    for fn, params in exps:
        for col, cid, coef in fn(params, cfg):
            assert 0 <= col < ell, f"slot {col} out of row at row {i}"
            R[i][col] = pr.add(R[i][col], pr.mul(pr.challenge(ch2, cid, "lin"), coef))

# Reference: route each term to (row, col) by its global slot.
ref = [[0] * ell for _ in range(m_total)]
def put(var_row, g, cid, coef):
    ro, s = divmod(g, ell)
    ref[var_row + ro][s] = pr.add(ref[var_row + ro][s], pr.mul(pr.challenge(ch2, cid, "lin"), coef))
for o in range(4):
    put(xr_row, o,    o, 1)
    put(x_row,  a[o], o, neg_ca[o])      # row 3
    put(x_row,  b[o], o, neg_cb[o])      # row 4

assert R == ref, "cross-row RoPE r^T A != reference"
print("CROSS-ROW ROPE TEST PASSED (straddling pairs, shared cid across rows)")
