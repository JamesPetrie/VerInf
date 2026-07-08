"""
verify.py — the Ligero verifier, written as the staged protocol.

Read it top-to-bottom: `run_verification` IS the protocol — four rounds, each
drawing fresh challenges AFTER the prover commits, then the checks that become
possible at that round. Everything is plain Python ints (via protocol.py); no
torch, no CUDA, no batching. Every line is meant to be checkable by eye.

The prover is an external callable with core.py's staged signature:
    prove(claims, inputs, cfg)                 -> root_p1           (round 1)
    prove(claims, inputs, cfg, ch0)            -> root_p2           (round 2)
    prove(claims, inputs, cfg, ch0, ch1)       -> (q_irs,q_lin,p_0) (round 3)
    prove(claims, inputs, cfg, ch0, ch1, ch2)  -> (opened_p1, opened_p2,
                                                   paths_p1, paths_p2) (round 4)
We never read its code: the verifier derives constraints itself (compile_claims)
and the 6 checks are authoritative. Soundness comes from drawing each round's
challenge only after that round's commitment is in hand — which is exactly the
round structure of run_verification.

The 6 check functions mirror core.py:_check_identities, unbatched. The test
combiners r_irs/r_lin/r_quad are not stored — each is derived on demand from
the round-2 (combiner) seed via pr.challenge(seed, index, label). The expensive
spots are the per-query double loops in the *_column_test functions (the O(W·T)
cost); those loops, and the per-query challenge derivation inside them, are what
a numpy/cuda backend would later batch and hoist.
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # pipeline/ on path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[0]))  # deprecated/ (python_verifier_compile)
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import protocol as pr
import python_verifier_compile as pvc
from protocol import P, mul, add, sub, poly_eval, eval_zeta_form

# ----------------------------------------------------------------------
# Vectorized Goldilocks field over numpy uint64 arrays. The scalar int ops in
# protocol.py (add/mul/…) are the readable spec/oracle; these are the fast path,
# kept bit-identical to them (proven by test_field.py). gmul is the one dense
# function: numpy has no uint128, so we multiply via 32-bit limbs and apply the
# Goldilocks reduction (2^64 ≡ 2^32−1, 2^96 ≡ −1 mod P).
# ----------------------------------------------------------------------
_M, _E, _S32, _Pu = (np.uint64(0xFFFFFFFF), np.uint64(0xFFFFFFFF),
                     np.uint64(32), np.uint64(P))

def gadd(a, b):
    s = a + b
    s = np.where(s < a, s + _E, s)                   # undo a 2^64 wrap
    return np.where(s >= _Pu, s - _Pu, s)

def gsub(a, b):
    d = a - b
    return np.where(a < b, d - _E, d)                # undo a borrow

def gmul(a, b):
    al, ah = a & _M, a >> _S32
    bl, bh = b & _M, b >> _S32
    ll, lh, hl, hh = al * bl, al * bh, ah * bl, ah * bh    # 32×32 → exact uint64
    mid  = lh + hl
    cmid = (mid < lh).astype(np.uint64)
    lo   = ll + ((mid & _M) << _S32)
    clo  = (lo < ll).astype(np.uint64)
    hi   = hh + (mid >> _S32) + (cmid << _S32) + clo
    hl32, hh32 = hi & _M, hi >> _S32                      # reduce (hi:lo) mod P
    r = gadd(lo % _Pu, (hl32 << _S32) % _Pu)
    return gsub(gsub(r, hl32), hh32)

def gsum(a, axis=0):                                      # modular sum (tree-reduce)
    a = np.moveaxis(a, axis, 0)
    while a.shape[0] > 1:
        if a.shape[0] & 1:
            a = np.concatenate([a, np.zeros((1,) + a.shape[1:], np.uint64)])
        a = gadd(a[0::2], a[1::2])
    return a[0]


# ----------------------------------------------------------------------
# What the prover opens at stage 4 (numpy/torch tensors get converted to lists).
# ----------------------------------------------------------------------
@dataclass
class OpenedColumns:
    """Columns opened at the queried indices Q, one sub-column per commit.
    `subcols` and `paths` are in joint-row order; row 0 of the joint column is
    the first row of the first commit (which holds the 3 blinding rows)."""
    subcols: List[Dict[int, List[int]]]            # subcols[commit][j] -> column
    paths:   List[Dict[int, List[Tuple[bytes, int]]]]
    def joint(self, j: int) -> List[int]:
        return [v for sc in self.subcols for v in sc[j]]


# ----------------------------------------------------------------------
# The driver — two entry points over one check surface.
#
# `prover` is a callable with the staged signature (the prover party; it owns the
# witness, the verifier never does):
#     prover(1)                       -> root_p1
#     prover(2, s_op)                 -> root_p2
#     prover(3, s_op, s_comb)         -> (q_irs, q_lin, p_0)
#     prover(4, s_op, s_comb, s_col)  -> (opened_p1, opened_p2, paths_p1, paths_p2)
#     prover(0, s_op, s_comb, s_col)  -> all of the above at once (fused, ~4× faster)
# Each q/p poly is ascending int coeffs; opened_pX is {j: column}; paths_pX is
# {j: merkle path}. `rand` is the verifier's coin source — it just returns a
# fresh seed (no args); both parties expand each seed by index identically.
# The verifier receives only `claims` (public), never the witness.
# ----------------------------------------------------------------------
def _checks(claims, cfg, s_op, s_comb, s_col,
            root_p1, root_p2, q_irs, q_lin, p_0, opened_p1, opened_p2, paths_p1, paths_p2):
    cons = pvc.compile_claims(claims, cfg, s_op)           # verifier compiles its OWN constraints
    Q    = pr.random_columns(s_col, cfg)
    cols = OpenedColumns([opened_p1, opened_p2], [paths_p1, paths_p2])
    return (merkle_test(cols, Q, [root_p1, root_p2])
            and irs_column_test(cols, q_irs, Q, s_comb, cfg)
            and linear_constraint_test(q_lin, cons, s_comb, cfg)
            and linear_column_test(cols, q_lin, Q, cons, s_comb, cfg)
            and quadratic_constraint_test(p_0, cfg)
            and quadratic_column_test(cols, p_0, Q, cons, s_comb, cfg))

def run_verification(prover, claims, cfg, rand):
    """INTERACTIVE protocol (SOUND): draw each round's seed only AFTER the prover
    has committed that round, so the prover can't see a challenge before it has
    bound what the challenge tests. Four prover round-trips."""
    root_p1            = prover(1)
    s_op               = rand()                  # only now — after seeing R_p1
    root_p2            = prover(2, s_op)
    s_comb             = rand()                  # only now — after R_p2
    q_irs, q_lin, p_0  = prover(3, s_op, s_comb)
    s_col              = rand()                  # only now — after the polys are fixed
    opened1, opened2, paths1, paths2 = prover(4, s_op, s_comb, s_col)
    return _checks(claims, cfg, s_op, s_comb, s_col,
                   root_p1, root_p2, q_irs, q_lin, p_0, opened1, opened2, paths1, paths2)

def run_verification_fast(prover, claims, cfg, rand):
    """FUSED variant (~4× faster, ONE streaming pass): all three seeds up front,
    prover computes everything in a single pass. Same checks. NOT sound on its
    own — the prover sees all challenges before committing — so use only when
    soundness is provided externally, or as a quick self-check."""
    s_op, s_comb, s_col = rand(), rand(), rand()
    root_p1, root_p2, q_irs, q_lin, p_0, opened1, opened2, paths1, paths2 = \
        prover(0, s_op, s_comb, s_col)
    return _checks(claims, cfg, s_op, s_comb, s_col,
                   root_p1, root_p2, q_irs, q_lin, p_0, opened1, opened2, paths1, paths2)


# ----------------------------------------------------------------------
# 1. Merkle: every opened sub-column hashes to its commit's root.
# ----------------------------------------------------------------------
def merkle_test(cols: OpenedColumns, Q, roots) -> bool:
    for subcols, paths, root in zip(cols.subcols, cols.paths, roots):
        if root == pr.EMPTY_COMMIT_ROOT:
            continue
        for j in Q:
            if j not in subcols:
                return False
            if not pr.merkle_verify(pr.merkle_leaf(subcols[j]), paths[j], root):
                return False
    return True


# ----------------------------------------------------------------------
# 2. IRS column identity:
#    q_irs(η_j) == Σ_i r_irs[i]·col[NUM_BLINDING_ROWS+i] + col[BLIND_IRS]
# ----------------------------------------------------------------------
def irs_column_test(cols, irs_poly, Q, ch2, cfg) -> bool:
    etas = [cfg.eta(j) for j in Q]
    C    = [cols.joint(j) for j in Q]
    lhs  = [col[pr.BLIND_IRS] for col in C]                # blind row, added once
    m_witness = len(C[0]) - pr.NUM_BLINDING_ROWS
    for i in range(m_witness):                            # per witness row
        ri = pr.challenge(ch2, i, "irs")                 # row combiner, computed once
        for jx in range(len(Q)):                         # update all queried columns now
            lhs[jx] = add(lhs[jx], mul(ri, C[jx][pr.NUM_BLINDING_ROWS + i]))
    return all(lhs[jx] == poly_eval(irs_poly, etas[jx]) for jx in range(len(Q)))


# ----------------------------------------------------------------------
# 3. Linear sum identity (poly-vs-constraint):
#    Σ_c q_lin(ζ_c) == Σ_g r_lin[g]·rhs_g
# ----------------------------------------------------------------------
def linear_constraint_test(lin_poly, cons: pr.Constraints, ch2, cfg) -> bool:
    sum_q = 0
    for c in range(cfg.ELL):
        sum_q = (sum_q + poly_eval(lin_poly, cfg.zeta(c))) % P
    rhs = 0
    for g, b_g in cons.rhs:                        # sparse: only the few nonzero RHS terms
        rhs = (rhs + mul(pr.challenge(ch2, g, "lin"), b_g)) % P
    return sum_q == rhs


# ----------------------------------------------------------------------
# 4. Linear column identity (poly-vs-column):
#    q_lin(η_j) == Σ_i r_i(η_j)·col[i] + col[BLIND_LIN]
# where r_i(η_j) is row i of (r^T A), in message form, evaluated at η_j.
# THIS double loop is the O(W·T) cost; it is the only thing worth batching.
# ----------------------------------------------------------------------
def linear_column_test(cols, lin_poly, Q, cons: pr.Constraints, ch2, cfg) -> bool:
    etas = [cfg.eta(j) for j in Q]
    C    = [cols.joint(j) for j in Q]                  # C[jx] = opened column for query Q[jx]
    lhs  = [col[pr.BLIND_LIN] for col in C]            # blind row, added once
    for i, expanders in enumerate(cons.rows):          # stream rows; r^T A never materialized
        ri = {}                                        # col -> row i of r^T A (sparse)
        for fn, params in expanders:                   # this row's constraint families
            for col, cid, coef in fn(params, cfg):     # expand on the fly (incl. RoPE cos/sin)
                S = pr.challenge(ch2, cid, "lin")
                ri[col] = pr.add(ri.get(col, 0), pr.mul(S, coef))
        if not ri:                                     # blinding / unconstrained rows → 0
            continue
        for jx in range(len(Q)):
            r_eta = 0
            for c, v in ri.items():                    # only this row's nonzeros
                r_eta = pr.add(r_eta, pr.mul(v, pr.lagrange(cfg, c, etas[jx])))
            lhs[jx] = pr.add(lhs[jx], pr.mul(r_eta, C[jx][i]))
    return all(lhs[jx] == poly_eval(lin_poly, etas[jx]) for jx in range(len(Q)))


# ----------------------------------------------------------------------
# 5. Quadratic zero identity (poly-vs-constraint): p_0(ζ_c) == 0 ∀ c.
# ----------------------------------------------------------------------
def quadratic_constraint_test(quad_poly, cfg) -> bool:
    return all(poly_eval(quad_poly, cfg.zeta(c)) == 0 for c in range(cfg.ELL))


# ----------------------------------------------------------------------
# 6. Quadratic column identity (poly-vs-column):
#    p_0(η_j) == Σ_t r_quad[t]·(Ux·Uy + a·Uz − b)|η_j + col[BLIND_QUAD]
# a, b are uniform scalars over the first n positions, so e.g.
#    a(η_j) = a · Σ_{c<n} L_c(η_j).
# ----------------------------------------------------------------------
def quadratic_column_test(cols, quad_poly, Q, cons: pr.Constraints, ch2, cfg) -> bool:
    etas = [cfg.eta(j) for j in Q]
    C    = [cols.joint(j) for j in Q]
    lhs  = [col[pr.BLIND_QUAD] for col in C]              # blind row, added once
    for t, qc in enumerate(cons.quadratic):              # per quadratic constraint (row triple)
        S = pr.challenge(ch2, t, "quad")                 # constraint combiner, computed once
        for jx in range(len(Q)):                         # update all queried columns now
            col, eta = C[jx], etas[jx]
            Ux, Uy, Uz = col[qc.x_row], col[qc.y_row], col[qc.z_row]
            mask = eval_zeta_form(cfg, [1] * qc.n, eta)      # Σ_{c<n} L_c(η_j)
            term = (mul(Ux, Uy) + mul(mul(qc.a, mask), Uz) - mul(qc.b, mask)) % P
            lhs[jx] = add(lhs[jx], mul(S, term))
    return all(lhs[jx] == poly_eval(quad_poly, etas[jx]) for jx in range(len(Q)))
