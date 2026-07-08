"""Accept/reject test for the row-outer irs / quadratic column checks.

Builds opened columns + a response poly that makes each check ACCEPT (the poly
is interpolated to hit the honest lhs at the queried points), then tampers one
opened value and confirms REJECT. A wrong loop reorder would compute a
different lhs and fail the ACCEPT case, so this catches the flip going wrong.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # pipeline/ on path
import random
import protocol as pr
import verify as vf

cfg = pr.Config(ELL=4, K_DEG=8, N_LIG=16, T_QUERIES=3)
ch2 = 7
rng = random.Random(0)
m_total = 6
Q = pr.random_columns(123, cfg)
etas = [cfg.eta(j) for j in Q]


# --- field Lagrange interpolation: coeffs of the degree-<len(pts) poly through (pts, vals) ---
def _mul_lin(c, a0):                          # multiply poly c by (x + a0)
    out = [0] * (len(c) + 1)
    for k in range(len(c)):
        out[k]     = pr.add(out[k], pr.mul(c[k], a0))
        out[k + 1] = pr.add(out[k + 1], c[k])
    return out

def interp(pts, vals):
    coeffs = [0] * len(pts)
    for i in range(len(pts)):
        num, den = [1], 1
        for m in range(len(pts)):
            if m == i:
                continue
            num = _mul_lin(num, pr.sub(0, pts[m]))
            den = pr.mul(den, pr.sub(pts[i], pts[m]))
        sc = pr.mul(vals[i], pr.inv(den))
        for k in range(len(num)):
            coeffs[k] = pr.add(coeffs[k], pr.mul(num[k], sc))
    return coeffs


def cols_from(subcol):                        # single-commit OpenedColumns
    return vf.OpenedColumns(subcols=[subcol], paths=[{}])

subcol = {j: [rng.randrange(pr.P) for _ in range(m_total)] for j in Q}
cols = cols_from(subcol)
C = [cols.joint(j) for j in Q]

# ---------------- IRS ----------------
mw = m_total - pr.NUM_BLINDING_ROWS
irs_target = []
for jx in range(len(Q)):
    s = C[jx][pr.BLIND_IRS]
    for i in range(mw):
        s = pr.add(s, pr.mul(pr.challenge(ch2, i, "irs"), C[jx][pr.NUM_BLINDING_ROWS + i]))
    irs_target.append(s)
irs_poly = interp(etas, irs_target)
assert vf.irs_column_test(cols, irs_poly, Q, ch2, cfg) is True, "IRS honest must ACCEPT"

bad = cols_from({j: list(subcol[j]) for j in Q})
bad.subcols[0][Q[0]][pr.NUM_BLINDING_ROWS] = pr.add(bad.subcols[0][Q[0]][pr.NUM_BLINDING_ROWS], 1)
assert vf.irs_column_test(bad, irs_poly, Q, ch2, cfg) is False, "IRS tampered must REJECT"
print("irs_column_test: ACCEPT honest, REJECT tampered  ok")

# ---------------- QUADRATIC ----------------
quad = [pr.Quadratic(3, 4, 5, a=rng.randrange(pr.P), b=rng.randrange(pr.P), n=4),
        pr.Quadratic(3, 5, 4, a=rng.randrange(pr.P), b=rng.randrange(pr.P), n=3)]
cons = pr.Constraints(rows=[[] for _ in range(m_total)], rhs=[], quadratic=quad, m_total=m_total)

quad_target = []
for jx in range(len(Q)):
    col, eta = C[jx], etas[jx]
    s = col[pr.BLIND_QUAD]
    for t, qc in enumerate(cons.quadratic):
        Ux, Uy, Uz = col[qc.x_row], col[qc.y_row], col[qc.z_row]
        mask = pr.eval_zeta_form(cfg, [1] * qc.n, eta)
        term = (pr.mul(Ux, Uy) + pr.mul(pr.mul(qc.a, mask), Uz) - pr.mul(qc.b, mask)) % pr.P
        s = pr.add(s, pr.mul(pr.challenge(ch2, t, "quad"), term))
    quad_target.append(s)
quad_poly = interp(etas, quad_target)
assert vf.quadratic_column_test(cols, quad_poly, Q, cons, ch2, cfg) is True, "QUAD honest must ACCEPT"

bad2 = cols_from({j: list(subcol[j]) for j in Q})
bad2.subcols[0][Q[0]][3] = pr.add(bad2.subcols[0][Q[0]][3], 1)   # tamper an x_row value
assert vf.quadratic_column_test(bad2, quad_poly, Q, cons, ch2, cfg) is False, "QUAD tampered must REJECT"
print("quadratic_column_test: ACCEPT honest, REJECT tampered  ok")

print("COLUMN-CHECK FLIP TEST PASSED (row-outer, all-columns-at-once)")
