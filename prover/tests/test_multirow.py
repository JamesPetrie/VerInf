"""Multi-row variables (n > ELL): _emit_id must slice a variable across the rows
it occupies — every expander's slots stay in-row, cids offset by row_offset·ELL.
Torch-free; drives the real emit-helpers (_emit_id, _emit_quad) directly with a
duck-typed Var (the real claim path reads .row_start/.length the same way)."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # pipeline/ on path
import protocol as pr
from collections import namedtuple

Var = namedtuple("Var", "row_start length")
cfg = pr.Config(ELL=4, K_DEG=8, N_LIG=16, T_QUERIES=3)
ch2 = 55
ell = cfg.ELL


def rTA(rows, m_total):
    R = [[0] * ell for _ in range(m_total)]
    for i, exps in enumerate(rows):
        for fn, params in exps:
            for col, cid, coef in fn(params, cfg):
                assert 0 <= col < ell, f"slot {col} out of row at row {i}"
                R[i][col] = pr.add(R[i][col], pr.mul(pr.challenge(ch2, cid, "lin"), coef))
    return R


# Copy dst[g] − src[g] = 0 for g in [0,6).  src spans rows 3,4; dst spans 5,6.
src = Var(3, 6); dst = Var(5, 6)
m_total = 7
rows = [[] for _ in range(m_total)]
pr._emit_id(rows, dst, 0, 1, cfg)          # dst coef +1, cids [0,6) sliced across rows 5,6
pr._emit_id(rows, src, 0, pr.P - 1, cfg)   # src coef −1, sliced across rows 3,4

ref = [[0] * ell for _ in range(m_total)]
for g in range(6):
    ro, s = divmod(g, ell)
    S = pr.challenge(ch2, g, "lin")
    ref[dst.row_start + ro][s] = pr.add(ref[dst.row_start + ro][s], pr.mul(S, 1))
    ref[src.row_start + ro][s] = pr.add(ref[src.row_start + ro][s], pr.mul(S, pr.P - 1))
assert rTA(rows, m_total) == ref, "multi-row Copy r^T A != reference"
print("multi-row Copy: r^T A correct across 2 rows, slots stay in-row  ok")

# Multi-row quadratic: a Hadamard over n=6 must emit one Quadratic per row chunk
# (n=4 then n=2). _emit_quad does the per-row split.
x = Var(3, 6); y = Var(5, 6); z = Var(7, 6)
quad = []
pr._emit_quad(quad, x, y, z, pr.P - 1, 0, 6, cfg)
assert len(quad) == 2, len(quad)
q0, q1 = quad
assert (q0.x_row, q0.n) == (3, 4) and (q1.x_row, q1.n) == (4, 2), (q0, q1)
print("multi-row Hadamard: one quadratic per row (n=4 then n=2)  ok")

print("MULTI-ROW TEST PASSED")
