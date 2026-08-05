"""RoutedProjectedMatmulClaim (S2): the routed expert matmul proved by
projection instead of by committing all E expert outputs.

Positive: an honest routed matmul verifies with the independent Rust verifier,
which compiles the claim itself and re-derives rho from s_op and (sigma,
lambda) from s_bind.

Negative: three MALICIOUS-PROVER simulations, one per relation that can fail
independently — a fabricated output (caught by yr = Y*rho and sum_k H = yr), an
output served by an expert the route matrix does not name (caught only by the
late Q = M*P), and a committed projection that is not W*rho.

What these tests deliberately do NOT cover: proving against different weights.
Swapping W makes the whole witness self-consistent, so no relation in the claim
can see it — that is the enrolled weight root's job (policy), not the claim's.
Tampering with the committed phase-3 columns is covered in test_phase3_block.
"""
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch
import core
import claims as _C       # noqa: F401
import packets as _PK     # noqa: F401
from tape import Tape
from routed_projected import routed_projected_matmul, RoutedProjectedMatmulClaim
from _rust_verify import rust_verify_tape

CFG = core.LigeroConfig(ELL=8, K_DEG=8, N_LIG=32, T_QUERIES=4)
T, K, J, E = 3, 2, 2, 2


def _u64(t):
    return t.to(torch.int64).to(torch.uint64).cuda()


def _build(routes=(0, 1, 0), x_scale=1, w_bump=0):
    """A T x K input, E experts of K x J weights, one-hot routes."""
    core._COSET_POWERS_K_CACHE.clear()
    tape = Tape(CFG, lazy=True)
    X = torch.arange(1, T * K + 1).reshape(T, K) * x_scale
    W = torch.arange(1, E * K * J + 1).reshape(E, K * J) + w_bump
    M = torch.zeros(T, E, dtype=torch.int64)
    for t, e in enumerate(routes):
        M[t, e] = 1
    x = tape.commit("X", _u64(X.reshape(-1)), (T, K))
    m = tape.commit("M", _u64(M.reshape(-1)), (T, E))
    w = tape.commit("W", _u64(W.reshape(-1)), (E, K * J))
    y = routed_projected_matmul(tape, x, m, w, T=T, K=K, J=J, E=E)
    return tape, y, X, W, M


def _reference(X, W, M):
    routes = M.argmax(dim=1)
    Y = torch.zeros(T, J, dtype=torch.int64)
    for t in range(T):
        Y[t] = X[t] @ W[routes[t]].reshape(K, J)
    return Y


def test_honest_routed_matmul_verifies():
    tape, y, X, W, M = _build()
    proof = tape.prove()
    assert "p3" in proof.blocks, f"late Freivalds rows missing: {proof.blocks}"
    acc, msg = rust_verify_tape(tape, proof, seed=None)
    assert acc, f"honest routed matmul: expected ACCEPT ({msg})"
    print(f"    honest routed matmul: ACCEPT (blocks={proof.blocks})")


def test_active_only_output_matches_dense_reference():
    """The committed Y must equal the dense per-token reference, even though
    it is produced one expert at a time over that expert's tokens only."""
    tape, y, X, W, M = _build()
    live = tape.run_engine_pass()
    got = live[y.var].view(torch.int64).cpu().reshape(T, J)
    want = _reference(X, W, M)
    assert torch.equal(got, want), f"active-only output {got} != dense {want}"
    print(f"    active-only == dense reference: {got.tolist()}")


def test_wrong_output_rejects():
    """Commit an output the routed contraction does not produce. The witness
    is otherwise consistent, so only yr = Y*rho and sum_k H = yr can catch it."""
    import compute_fns as cf
    real = cf.COMPUTE_FNS[RoutedProjectedMatmulClaim]

    def lying(claim, live):
        out = real(claim, live)
        out[claim.Y] = (out[claim.Y].view(torch.int64) + 1).view(torch.uint64)
        return out

    tape, y, X, W, M = _build()
    cf.COMPUTE_FNS[RoutedProjectedMatmulClaim] = lying
    try:
        proof = tape.prove()
    finally:
        cf.COMPUTE_FNS[RoutedProjectedMatmulClaim] = real
    acc, msg = rust_verify_tape(tape, proof, seed=None)
    assert not acc, "wrong routed output: expected REJECT"
    print("    wrong output (yr = Y*rho breaks): REJECT ok")


def _prove_with_patch(registry, key, patched):
    tape, y, X, W, M = _build()
    real = registry[key]
    registry[key] = patched(real)
    try:
        return tape, tape.prove()
    finally:
        registry[key] = real


def test_route_not_matching_the_output_rejects():
    """A prover that serves token t from a different expert than M claims.
    Everything else stays consistent, so only the late Q = M*P check sees it —
    precisely the relation the fifth transcript message exists for."""
    def lying_router(real):
        def f(claim, live):
            from cuda_primitives import gl_matmul
            M = live[claim.M].reshape(claim.T, claim.E).view(torch.int64)
            X = live[claim.X].reshape(claim.T, claim.K)
            W = live[claim.W].reshape(claim.E, claim.K, claim.J)
            routes = (M.argmax(dim=1) + 1) % claim.E          # wrong expert
            Y = torch.zeros((claim.T, claim.J), dtype=torch.int64, device="cuda")
            for t_i in range(claim.T):
                Y[t_i] = gl_matmul(X[t_i:t_i + 1].contiguous(),
                                   W[routes[t_i]].contiguous()).view(torch.int64)[0]
            return {claim.Y: Y.view(torch.uint64).reshape(-1)}
        return f

    import compute_fns as cf
    tape, proof = _prove_with_patch(cf.COMPUTE_FNS, RoutedProjectedMatmulClaim,
                                    lying_router)
    acc, msg = rust_verify_tape(tape, proof, seed=None)
    assert not acc, "output from an unrouted expert: expected REJECT"
    print("    output served by the wrong expert (Q = M*P breaks): REJECT ok")


def test_lying_projection_rejects():
    """A prover that commits a P which is not W*rho — the projection is the
    only thing standing between the proof and arbitrary expert weights."""
    def lying_projection(real):
        def f(claim, witness, rho):
            out = real(claim, witness, rho)
            out[claim.Pj] = (out[claim.Pj].view(torch.int64) + 1).view(torch.uint64)
            return out
        return f

    tape, proof = _prove_with_patch(core.AUX_FNS, RoutedProjectedMatmulClaim,
                                    lying_projection)
    acc, msg = rust_verify_tape(tape, proof, seed=None)
    assert not acc, "P != W*rho: expected REJECT"
    print("    P not equal to W*rho: REJECT ok")


def test_constraint_count_matches_the_ledger():
    """E*K + 2T + 2E + 1 — the L_route term of the admission model."""
    tape, y, X, W, M = _build()
    c = next(c for c in tape.claims if isinstance(c, RoutedProjectedMatmulClaim))
    import protocol as pr
    from routed_projected import routed_compile
    for v, start in ((c.X, 0), (c.Y, 8), (c.M, 16), (c.W, 24), (c.Pj, 32),
                     (c.Qm, 40), (c.Hd, 48), (c.yr, 56), (c.f_y, 64),
                     (c.f_u, 72), (c.f_p, 80)):
        v.row_start = start
    rho = pr.op_vec(b"x" * 32, 0, "rho", J)
    late = (pr.op_vec(b"y" * 32, 0, "sig", K), pr.op_vec(b"y" * 32, 0, "lam", T))
    _pk, quads, n_added, _b = routed_compile(c, rho, CFG, 0, late_ch=late)
    assert n_added == E * K + 2 * T + 2 * E + 1, n_added
    assert sum(f.L for f in quads) == T * K + E, [f.L for f in quads]
    print(f"    ledger: {n_added} linear ids, {T*K + E} quads "
          f"(= E*K+2T+2E+1, T*K+E)")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    fails = 0
    for t in tests:
        try:
            t(); print(f"[OK ] {t.__name__}")
        except Exception as e:
            fails += 1; print(f"[XX ] {t.__name__}: {type(e).__name__}: {e}")
    print(f"=== routed-projected: {len(tests)-fails}/{len(tests)} "
          f"{'PASS' if not fails else 'FAIL'} ===")
    return fails


if __name__ == "__main__":
    raise SystemExit(main())
