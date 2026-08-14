"""Bit-exact gate for the evaluation-domain quadratic accumulator.

The optimized prover delays the inverse NTT until after the r_quad row sum.
This test compares it with the literal per-constraint polynomial identity,
including partial rows and non-zero public a/b coefficients.
"""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import core
from cuda_primitives import gl_add, gl_matvec, gl_sub, poly_mul_batched

CFG = core.LigeroConfig(ELL=8, K_DEG=16, N_LIG=64, T_QUERIES=4)


def _u64(xs):
    return torch.tensor(xs, dtype=torch.int64, device="cuda").to(torch.uint64)


def _reference(p1, p2, inputs, m_p1, r, quads, seed, maps):
    K = CFG.K_DEG
    out = torch.zeros(2 * K - 1, dtype=torch.uint64, device="cuda")
    grid = torch.arange(CFG.ELL, dtype=torch.int64, device="cuda")
    for i, q in enumerate(quads):
        rows = sorted({q.x_row, q.y_row, q.z_row})
        p1_abs = [x for x in rows if core.NUM_BLINDING_ROWS <= x < m_p1]
        p2_abs = [x for x in rows if x >= m_p1]
        polys, loc = [], {}
        for vs, mp, rr in ((p1, maps[0], p1_abs), (p2, maps[1], p2_abs)):
            if not rr: continue
            msg = core._gather_rows(inputs, CFG, mp, rr)
            enc = core._encode_rows_indexed(msg, rr, CFG, seed)
            for row, poly in zip(rr, enc):
                loc[row] = len(polys); polys.append(poly)
        pc = torch.stack(polys)
        px = pc[loc[q.x_row]].unsqueeze(0)
        py = pc[loc[q.y_row]].unsqueeze(0)
        pz = pc[loc[q.z_row]].unsqueeze(0)
        mask = (grid < q.n).to(torch.int64).unsqueeze(0)
        av = torch.tensor([q.a_values[0]], dtype=torch.uint64, device="cuda")
        bv = torch.tensor([q.b_values[0]], dtype=torch.uint64, device="cuda")
        pa = core._interpolate_to_kdeg((av.view(torch.int64).unsqueeze(1) * mask)
                                       .view(torch.uint64), CFG)
        pb = core._interpolate_to_kdeg((bv.view(torch.int64).unsqueeze(1) * mask)
                                       .view(torch.uint64), CFG)
        inner = gl_sub(gl_add(poly_mul_batched(px, py),
                             poly_mul_batched(pa, pz)),
                       torch.cat([pb, torch.zeros((1, K - 1), dtype=torch.uint64,
                                                  device="cuda")], dim=1))
        out = gl_add(out, gl_matvec(inner.T.contiguous(), r[i:i + 1]))
    return out


def test_eval_domain_accumulator_matches_literal_polynomials():
    core._COSET_POWERS_K_CACHE.clear()
    v1 = core.Variable("x", 16, phase=1); v1.row_start = core.NUM_BLINDING_ROWS
    v2 = core.Variable("y", 16, phase=2); v2.row_start = v1.row_start + 2
    m_p1 = v2.row_start
    inputs = {v1: _u64(range(1, 17)), v2: _u64(range(21, 37))}
    quads = [
        core.QuadraticConstraint("full-b0", v1.row_start, v2.row_start,
                                 v2.row_start, 8, [core.P - 1], [0]),
        core.QuadraticConstraint("partial", v1.row_start + 1, v2.row_start + 1,
                                 v2.row_start + 1, 5, [7], [11]),
    ]
    r = _u64([13, 17])
    seed = core._master_seed_to_cuda(b"q" * 32)
    maps = (core._build_row_map([v1], CFG, core.NUM_BLINDING_ROWS),
            core._build_row_map([v2], CFG, m_p1))
    got = core.compute_p_0_streaming([v1], [v2], inputs, m_p1, r, quads,
                                     CFG, seed, chunk_size=2, maps=maps)
    want = _reference([v1], [v2], inputs, m_p1, r, quads, seed, maps)
    assert torch.equal(got, want), "evaluation-domain reassociation changed p_0"
    print("    optimized p_0 == literal per-row polynomial p_0")


def main():
    test_eval_domain_accumulator_matches_literal_polynomials()
    print("=== quad-eval-accumulator: 1/1 PASS ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
