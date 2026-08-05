"""RoutedProjectedMatmulClaim — the routed MoE expert matmul, proved by
projection instead of by committing all E expert outputs.

The relation (analysis/routed-projected-protocol.md), for one MoE matrix:

    Y[t,j] = sum_k X[t,k] * W[e_t,k,j]

with e_t the unique top-1 route already proved by RoutingClaim (M is its
one-hot matrix). Committing Y for all E experts is what makes the ordinary
tape cost 396 G witness slots at S=1000; here the verifier instead projects
the output axis with rho (sampled after R1) and the degree-three contraction
reassociates into

    P[e,k] = sum_j W[e,k,j] rho_j          (a matvec over the enrolled weights)
    Q[t,k] = sum_e M[t,e] P[e,k]           (a matrix product, checked late)
    H[t,k] = X[t,k] * Q[t,k]               (Hadamard)
    yr[t]  = sum_j Y[t,j] rho_j            (projection of the committed output)
    sum_k H[t,k] = yr[t]

Only Q = M*P has two inputs that were not both fixed in R1, so it is checked
by an ordinary Freivalds contraction whose challenges (sigma, lambda) are
sampled AFTER P and Q are committed — the phase-3 rows f_y/f_u/f_p and the
fifth transcript message exist for exactly this.

Nothing here is a new proof system: every band below is one of the packet
templates the MatmulClaim already uses, so the Rust verifier's existing
lowering covers them and the same q_lin/q_quad/IRS/Merkle checks discharge the
claim. Concretely:

    P = W rho          LF1 (identity on P, FreivaldsLF1B on W)
    yr = Y rho         LF1 (identity on yr, FreivaldsLF1B on Y)
    sum_k H = yr       stride-many-to-one on H, identity on yr
    H = X * Q          one quadratic family
    f_y = P sigma      LF1 with sigma
    f_u = lambda^T M   LF2 with lambda
    f_p = f_u * f_y    one quadratic family
    sum_e f_p = sum_{t,k} lambda_t Q[t,k] sigma_k
                       stride-many-to-one on f_p, LF3C on Q

Constraint ids, from `base`:
    [0,            E*K)            P     = W rho
    [E*K,          E*K + T)        yr    = Y rho
    [E*K + T,      E*K + 2T)       sum_k H = yr
    [E*K + 2T,     E*K + 2T + E)   f_y   = P sigma
    [E*K + 2T + E, E*K + 2T + 2E)  f_u   = lambda^T M
    [E*K + 2T + 2E, +1)            the final Freivalds scalar
i.e. E*K + 2T + 2E + 1 constraints, which is the L_route term of the ledger in
analysis/routed_projected_4h_model.py.
"""
from dataclasses import dataclass
from typing import List, Tuple

import torch

import protocol
from core import (
    AUX_FNS, COMPILE_FNS, LATE_AUX_FNS, LATE_SAMPLE_FNS, SAMPLE_FNS,
    LigeroConfig, QuadFamily, Variable, _build_b_chunk,
)
from cuda_primitives import P as FIELD_P, gl_matmul, gl_mul, gl_neg
from packets import (
    L2_FreivaldsLF1B, L2_FreivaldsLF2A, L2_FreivaldsLF3C, L2_IdentityScalar,
    L2_StrideManyToOneScalar,
)


@dataclass
class RoutedProjectedMatmulClaim:
    """One routed expert matmul. X/Y/M/W are phase 1, P/Q/H/yr phase 2,
    f_y/f_u/f_p phase 3.

    Shapes (all row-major, flat):
      X  (T, K)      layer input
      Y  (T, J)      raw routed output (before rescale — RescaleClaim follows)
      M  (T, E)      one-hot routes, from RoutingClaim
      W  E vars of (K, J)   enrolled expert weights, ONE VARIABLE PER EXPERT
      P  (E, K)      Q (T, K)   H (T, K)   yr (T,)
      f_y f_u f_p (E,)

    W is a list, not a single (E, K, J) variable, because the prover must never
    hold more than one expert shard at a time: at Maverick shapes one layer's
    128 experts are ~43 GB. Per-expert variables let both the witness pass and
    the projection stream one shard, and they are what the enrolled weight
    block already contains.
    """
    X: Variable
    Y: Variable
    M: Variable
    W: List[Variable]     # one variable PER EXPERT, each (K, J) flat
    Pj: Variable          # projected weights   (E, K)
    Qm: Variable          # routed projection   (T, K)
    Hd: Variable          # X * Q               (T, K)
    yr: Variable          # Y rho               (T,)
    f_y: Variable
    f_u: Variable
    f_p: Variable
    T: int
    K: int
    J: int
    E: int


# ---------------------------------------------------------------- challenges
def routed_sample(c: RoutedProjectedMatmulClaim, ci: int, s_op):
    """R1 coin: the output-axis projection rho (length J)."""
    return protocol.op_vec(s_op, ci, "rho", c.J)


def routed_late_sample(c: RoutedProjectedMatmulClaim, ci: int, s_bind):
    """R2 coin: the Freivalds challenges for Q = M*P. Sampled from s_bind,
    which does not exist until P and Q are committed."""
    return (protocol.op_vec(s_bind, ci, "sig", c.K),
            protocol.op_vec(s_bind, ci, "lam", c.T))


def _vec(xs) -> torch.Tensor:
    return torch.tensor(xs, dtype=torch.uint64, device="cuda")


# ------------------------------------------------------------ aux witnesses
# P = W*rho is the one pass over the enrolled 400B weights. The prover
# regenerates the witness in five epochs, so without a cache the projection
# would be recomputed four more times; caching it turns four identical 400B
# passes into one (demo/4h-production-runbook.md). The key includes a digest of
# rho, so a different challenge cannot hit a stale entry — recomputing is
# always sound, reusing a P from another rho would not be.
_P_CACHE: dict = {}
# Observability for the gate: how many 400B projections actually ran.
P_CACHE_STATS = {"hits": 0, "misses": 0}


def clear_p_cache():
    """Drop the projected-weight cache. Runs at the start of each proof."""
    _P_CACHE.clear()
    P_CACHE_STATS.update(hits=0, misses=0)


def _rho_key(c, rho) -> tuple:
    import blake3
    digest = blake3.blake3(
        b"".join(int(v).to_bytes(8, "little") for v in rho)).digest()
    return (id(c), digest)


def _resolve(live, var):
    """Read ONE shard, resolving its loader. Deliberately not cached: the
    caller drops the tensor before touching the next expert."""
    val = live[var]
    return val() if callable(val) else val


def _project_weights(c: RoutedProjectedMatmulClaim, live, rho) -> torch.Tensor:
    """P[e,k] = sum_j W[e,k,j] rho_j, one expert shard at a time."""
    key = _rho_key(c, rho)
    hit = _P_CACHE.get(key)
    if hit is not None:
        P_CACHE_STATS["hits"] += 1
        return hit
    P_CACHE_STATS["misses"] += 1
    rho_t = _vec(rho).view(c.J, 1)
    out = torch.empty(c.E * c.K, dtype=torch.uint64, device="cuda")
    for e, w_var in enumerate(c.W):
        w_e = _resolve(live, w_var).reshape(c.K, c.J)
        out[e * c.K:(e + 1) * c.K] = gl_matmul(w_e, rho_t).reshape(-1)
        del w_e
    _P_CACHE[key] = out
    return out


def routed_aux(c: RoutedProjectedMatmulClaim, witness, rho) -> dict:
    """Phase-2 rows. The projection is normally already in the cache: the same
    sweep's witness pass computed it while each shard was resident (see
    routed_compute), so no expert is read twice per epoch."""
    rho_t = _vec(rho).view(c.J, 1)
    Pj = _project_weights(c, witness, rho)                     # (E*K,)
    M = witness[c.M].reshape(c.T, c.E)
    Qm = gl_matmul(M, Pj.reshape(c.E, c.K)).reshape(-1)        # (T*K,)
    Hd = gl_mul(witness[c.X].reshape(-1), Qm)
    Y = witness[c.Y].reshape(c.T, c.J)
    yr = gl_matmul(Y, rho_t).reshape(-1)                       # (T,)
    return {c.Pj: Pj, c.Qm: Qm, c.Hd: Hd, c.yr: yr}


def routed_late_aux(c: RoutedProjectedMatmulClaim, witness, ch) -> dict:
    """Phase-3 rows: the late Freivalds auxiliaries of Q = M*P."""
    sig, lam = ch
    Pj = witness[c.Pj].reshape(c.E, c.K)
    f_y = gl_matmul(Pj, _vec(sig).view(c.K, 1)).reshape(-1)            # (E,)
    M = witness[c.M].reshape(c.T, c.E)
    f_u = gl_matmul(_vec(lam).view(1, c.T), M).reshape(-1)             # (E,)
    return {c.f_y: f_y, c.f_u: f_u, c.f_p: gl_mul(f_u, f_y)}


# ------------------------------------------------------------------ compile
def routed_compile(c: RoutedProjectedMatmulClaim, rho, cfg: LigeroConfig,
                   base: int, late_ch=None):
    """Emit the bands and quads. `late_ch` is (sigma, lambda); it is None only
    before R2, when the phase-3 constraints do not yet exist — the verifier
    always compiles with it, since by then the whole transcript is fixed."""
    ell = cfg.ELL
    T, K, J, E = c.T, c.K, c.J, c.E
    rho_t = _vec(rho)
    neg_rho = gl_neg(rho_t)
    neg1 = (FIELD_P - 1) % FIELD_P

    b_P = base                      # P    = W rho          (E*K constraints)
    b_yr = b_P + E * K              # yr   = Y rho          (T)
    b_sum = b_yr + T                # sum_k H = yr          (T)
    b_fy = b_sum + T                # f_y  = P sigma        (E)
    b_fu = b_fy + E                 # f_u  = lambda^T M     (E)
    b_fin = b_fu + E                # the final scalar      (1)
    n_added = b_fin + 1 - base

    row_pkts: List[Tuple[int, object]] = []

    def rows(var, pkt):
        for off in range(var.n_rows(ell)):
            row_pkts.append((var.row_start + off, pkt))

    # ---- P = W rho : identity on P, LF1-B on the enrolled weights ----------
    rows(c.Pj, L2_IdentityScalar(base=b_P, var_row_start=c.Pj.row_start,
                                 L=E * K, coef=1))
    # One band per expert shard: expert e owns constraint ids [b_P + e*K,
    # b_P + (e+1)*K), so the projection streams shard by shard.
    for e, w_var in enumerate(c.W):
        rows(w_var, L2_FreivaldsLF1B(base=b_P + e * K, B_row_start=w_var.row_start,
                                     k=K, n=J, H=1, K=K,
                                     transpose_b=False, neg_rho=neg_rho))
    # ---- yr = Y rho -------------------------------------------------------
    rows(c.yr, L2_IdentityScalar(base=b_yr, var_row_start=c.yr.row_start,
                                 L=T, coef=1))
    rows(c.Y, L2_FreivaldsLF1B(base=b_yr, B_row_start=c.Y.row_start,
                               k=T, n=J, H=1, K=T,
                               transpose_b=False, neg_rho=neg_rho))
    # ---- sum_k H[t,k] - yr[t] = 0 ----------------------------------------
    rows(c.Hd, L2_StrideManyToOneScalar(base=b_sum, var_row_start=c.Hd.row_start,
                                        L=T * K, stride=K, coef=1))
    rows(c.yr, L2_IdentityScalar(base=b_sum, var_row_start=c.yr.row_start,
                                 L=T, coef=neg1))

    # Both quadratic families are emitted unconditionally: a QuadFamily is
    # pure geometry (rows and length), so the constraint COUNT never depends on
    # a challenge — which is what keeps constraint ids and the r_quad vector
    # aligned no matter when the compile runs.
    quads: List[QuadFamily] = [
        QuadFamily(name=f"{c.Y.name}.RP[H=X*Q]", x_row=c.X.row_start,
                   y_row=c.Qm.row_start, z_row=c.Hd.row_start, L=T * K,
                   ell=ell, a=neg1, b=0),
        QuadFamily(name=f"{c.Y.name}.RP[f_p=f_u*f_y]", x_row=c.f_u.row_start,
                   y_row=c.f_y.row_start, z_row=c.f_p.row_start, L=E,
                   ell=ell, a=neg1, b=0),
    ]

    if late_ch is not None:
        sig, lam = late_ch
        sig_t, lam_t = _vec(sig), _vec(lam)
        # ---- f_y = P sigma ------------------------------------------------
        rows(c.f_y, L2_IdentityScalar(base=b_fy, var_row_start=c.f_y.row_start,
                                      L=E, coef=1))
        rows(c.Pj, L2_FreivaldsLF1B(base=b_fy, B_row_start=c.Pj.row_start,
                                    k=E, n=K, H=1, K=E,
                                    transpose_b=False, neg_rho=gl_neg(sig_t)))
        # ---- f_u = lambda^T M ---------------------------------------------
        rows(c.f_u, L2_IdentityScalar(base=b_fu, var_row_start=c.f_u.row_start,
                                      L=E, coef=1))
        rows(c.M, L2_FreivaldsLF2A(base=b_fu, A_row_start=c.M.row_start,
                                   k=E, m=T, H=1, K=E, neg_lam=gl_neg(lam_t)))
        # ---- sum_e f_p = sum_{t,k} lambda_t Q[t,k] sigma_k -----------------
        rows(c.f_p, L2_StrideManyToOneScalar(base=b_fin,
                                             var_row_start=c.f_p.row_start,
                                             L=E, stride=E, coef=1))
        rows(c.Qm, L2_FreivaldsLF3C(base=b_fin, C_row_start=c.Qm.row_start,
                                    m=T, n=K, H=1, L=T * K,
                                    lam=lam_t, rho=sig_t))

    return row_pkts, quads, n_added, _build_b_chunk(n_added, [])


# ------------------------------------------------------------------ witness
def routed_compute(c: RoutedProjectedMatmulClaim, live, rho=None) -> dict:
    """The raw routed output, executed ACTIVE-ONLY: one expert at a time, and
    for each expert only the tokens routed to it. No all-expert output tensor
    is ever allocated — that is the structural requirement in
    demo/4h-production-runbook.md, not just an optimisation.

    `live` arrives with its loaders UNRESOLVED (core.STREAMING_INPUT_CLAIMS):
    exactly one expert shard is in memory at a time, which is what makes a
    128-expert Maverick matrix (~43 GB) fit at all.

    When `rho` is given (every sweep from R2 on), the SAME resident shard also
    contributes its slice of P = W*rho, so the projection costs no extra read
    or decode of the enrolled weights."""
    T, K, J, E = c.T, c.K, c.J, c.E
    X = _resolve(live, c.X).reshape(T, K)
    M = _resolve(live, c.M).reshape(T, E).view(torch.int64)
    routes = M.argmax(dim=1)
    fuse_projection = rho is not None and _rho_key(c, rho) not in _P_CACHE
    if fuse_projection:
        P_CACHE_STATS["misses"] += 1
        rho_t = _vec(rho).view(J, 1)
        proj = torch.empty(E * K, dtype=torch.uint64, device="cuda")
    # int64 view for the scatter: torch has no index_put for uint64, and the
    # values are field elements either way — the view is a reinterpretation,
    # not a conversion.
    Y = torch.zeros((T, J), dtype=torch.int64, device="cuda")
    for e in range(E):
        idx = (routes == e).nonzero(as_tuple=True)[0]
        if idx.numel() == 0 and not fuse_projection:
            continue                       # shard never touched: no load at all
        w_e = _resolve(live, c.W[e]).reshape(K, J)
        if idx.numel():
            Y[idx] = gl_matmul(X.index_select(0, idx).contiguous(),
                               w_e).view(torch.int64)
        if fuse_projection:
            proj[e * K:(e + 1) * K] = gl_matmul(w_e, rho_t).reshape(-1)
        del w_e                            # released before the next shard
    if fuse_projection:
        _P_CACHE[_rho_key(c, rho)] = proj
    return {c.Y: Y.view(torch.uint64).reshape(-1)}


def routed_projected_matmul(tape, x, m_routes, w_experts, *, T, K, J, E):
    """Record one routed expert matmul on `tape`.

    x:         (T, K) WitnessTensor — the layer input
    m_routes:  (T, E) WitnessTensor — one-hot routes (RoutingClaim's M)
    w_experts: list of E WitnessTensors, each (K, J) — the enrolled expert
               shards, kept separate so the prover streams one at a time
    Returns the raw output (T, J); a RescaleClaim follows it in the model
    builder, exactly as the old in-matmul rescale did.
    """
    assert len(w_experts) == E, f"expected {E} expert shards, got {len(w_experts)}"
    w_vars = [w.var for w in w_experts]
    name = f"rp[{x.var.name}@{w_vars[0].name}..]"
    Y = tape._alloc(name, T * J, phase=1)
    Pj = tape._alloc(f"{name}.P", E * K, phase=2)
    Qm = tape._alloc(f"{name}.Q", T * K, phase=2)
    Hd = tape._alloc(f"{name}.H", T * K, phase=2)
    yr = tape._alloc(f"{name}.yr", T, phase=2)
    f_y = tape._alloc(f"{name}.f_y", E, phase=3)
    f_u = tape._alloc(f"{name}.f_u", E, phase=3)
    f_p = tape._alloc(f"{name}.f_p", E, phase=3)
    claim = RoutedProjectedMatmulClaim(
        X=x.var, Y=Y, M=m_routes.var, W=w_vars,
        Pj=Pj, Qm=Qm, Hd=Hd, yr=yr, f_y=f_y, f_u=f_u, f_p=f_p,
        T=T, K=K, J=J, E=E)
    # The expert shards are NOT declared as claim inputs: the generic sweep
    # pre-fetches every input_var, which for 128 Maverick shards is ~43 GB in
    # one go. They are claim fields (so they are laid out, compiled and
    # emitted), and the compute/aux resolve them one at a time.
    outs = tape._process_claim(claim, [x.var, m_routes.var])
    tape.claims.append(claim)
    from tape import WitnessTensor
    return WitnessTensor(outs[Y] if outs else None, Y, (T, J), tape)


import core as _core                          # noqa: E402  (registry wiring)
if clear_p_cache not in _core.PROVE_START_HOOKS:
    _core.PROVE_START_HOOKS.append(clear_p_cache)
_core.STREAMING_INPUT_CLAIMS.add(RoutedProjectedMatmulClaim)

SAMPLE_FNS[RoutedProjectedMatmulClaim] = routed_sample
AUX_FNS[RoutedProjectedMatmulClaim] = routed_aux
LATE_SAMPLE_FNS[RoutedProjectedMatmulClaim] = routed_late_sample
LATE_AUX_FNS[RoutedProjectedMatmulClaim] = routed_late_aux
COMPILE_FNS[RoutedProjectedMatmulClaim] = routed_compile

import compute_fns as _cf                      # noqa: E402  (registry wiring)
_cf.COMPUTE_FNS[RoutedProjectedMatmulClaim] = routed_compute
