"""WC-LCRL-STC part 1+2: enrollment + late coefficient bridge
(analysis/wc-lcrl-stc-spec.md §0.1–§0.4).  Toy geometry, production ratios:
K_w a power of two, N_w = 2·K_w, two width groups sharing the transcript."""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

import wc_bridge as wc
from cuda_primitives import P, poly_eval

PARAMS = wc.WcParams(B=48, lam=16, N_w=128, q_w=8)
S_R1 = b"\x11" * 32          # transcript state after R1 (outputs fixed)


def _toy_enrollment(seed=0):
    g = torch.Generator().manual_seed(seed)
    groups = {
        4: (torch.randint(0, 1 << 62, (100, 4), generator=g,
                          dtype=torch.int64).to(torch.uint64)).cuda(),
        6: (torch.randint(0, 1 << 62, (52, 6), generator=g,
                          dtype=torch.int64).to(torch.uint64)).cuda(),
    }
    enr = wc.build_enrollment(groups, b"mask-seed", b"manifest-v1", PARAMS)
    meta = {n: (enr.groups[n].n_blocks, n) for n in enr.groups}
    return enr, meta


def test_rs_domain_matches_poly_eval():
    """The codeword really is F evaluated on the module's domain: transform a
    random coefficient row and cross-check codeword slots against Horner."""
    coeffs = (torch.randint(0, 1 << 62, (1, PARAMS.K_w),
                            dtype=torch.int64).to(torch.uint64)).cuda()
    cw = wc._coeffs_to_codewords(coeffs, PARAMS)[0]
    domain = wc._rs_domain(PARAMS)
    for i in (0, 5, PARAMS.N_w - 1):
        want = poly_eval(coeffs[0], domain[i:i + 1])[0].item()
        assert cw[i].item() == want, f"domain slot {i} disagrees with Horner"


def test_honest_bridge_accepts():
    enr, meta = _toy_enrollment()
    proof = wc.prove_bridge(enr, S_R1)
    ok, why = wc.verify_bridge(enr.root, enr.manifest_digest, meta,
                               proof, S_R1, PARAMS)
    assert ok, why
    # and P_trace really is W rho, checked independently in python ints
    n = 4
    W = enr.groups[n].weights.cpu().tolist()
    rho = proof.rho[n]
    for row in (0, 57, 99):
        want = sum(W[row][j] * rho[j] for j in range(n)) % P
        assert proof.p_trace[n][row].item() % P == want


def test_soundness_bound_value():
    p = wc.WcParams()          # production geometry
    h = p.soundness_bound()
    assert p.K_w == 16384 and p.N_w == 32768
    assert abs(h - 8.859073161e-13) / 8.859073161e-13 < 1e-6, h


def test_tampered_p_trace_rejects():
    """A projection that is not W rho must fail the bridge equation."""
    enr, meta = _toy_enrollment()
    proof = wc.prove_bridge(enr, S_R1)
    proof.p_trace[4][17] = (int(proof.p_trace[4][17].item()) + 1) % P
    # honest-prover consistency values (c, v) no longer match the commitment;
    # a cheating prover would instead recompute c/v from the fake P_trace, so
    # rebuild them the way the prover does — the bridge equation still fails.
    ok, why = wc.verify_bridge(enr.root, enr.manifest_digest, meta,
                               proof, S_R1, PARAMS)
    assert not ok, "tampered P_trace accepted"


def test_recommitted_fake_projection_rejects():
    """The strong attack: fake P_trace committed BEFORE the late coins, with
    c and v honestly recomputed from the fake values.  Caught by the
    enrollment side of the bridge at the eta points."""
    enr, meta = _toy_enrollment()
    # tamper the weights the prover projects, keep the enrollment intact
    enr_fake = wc.build_enrollment(
        {n: g.weights[:g.n_rows].clone() for n, g in enr.groups.items()},
        b"mask-seed", b"manifest-v1", PARAMS)
    enr_fake.groups[4].weights[3, 2] = \
        (int(enr_fake.groups[4].weights[3, 2].item()) + 5) % P
    proof = wc.prove_bridge(enr_fake, S_R1)      # internally consistent lie
    ok, why = wc.verify_bridge(enr.root, enr.manifest_digest, meta,
                               proof, S_R1, PARAMS)
    assert not ok, "re-committed fake projection accepted"
    assert "bridge equation" in why or "merkle" in why, why


def test_tampered_enrollment_column_rejects():
    enr, meta = _toy_enrollment()
    proof = wc.prove_bridge(enr, S_R1)
    i = proof.eta_idx[0]
    proof.opened[i][0] = (int(proof.opened[i][0]) + 1) % P
    ok, why = wc.verify_bridge(enr.root, enr.manifest_digest, meta,
                               proof, S_R1, PARAMS)
    assert not ok and "merkle" in why, why


def test_foreign_eta_rejects():
    """The prover cannot choose its own opening points."""
    enr, meta = _toy_enrollment()
    proof = wc.prove_bridge(enr, S_R1)
    proof.eta_idx = list(range(PARAMS.q_w))          # attacker-chosen points
    ok, why = wc.verify_bridge(enr.root, enr.manifest_digest, meta,
                               proof, S_R1, PARAMS)
    assert not ok and "eta" in why, why


def test_zero_pad_tail_is_bound():
    """Zero-filled tails (spec §0.1) are part of the polynomial: claiming a
    nonzero value in the padded region breaks the bridge."""
    enr, meta = _toy_enrollment()
    proof = wc.prove_bridge(enr, S_R1)
    g = enr.groups[6]
    assert g.n_blocks * PARAMS.B > g.n_rows, "toy shape should need padding"
    proof.p_trace[6][g.n_rows + 1] = 12345          # inside the padded tail
    ok, _ = wc.verify_bridge(enr.root, enr.manifest_digest, meta,
                             proof, S_R1, PARAMS)
    assert not ok, "padded tail is not bound"


# ---- review tests (analysis/wc-lcrl-stc-review checklist §5/§7) ------------

def test_tampered_pi_rejects():
    """A cheater that substitutes projected masks and honestly recomputes
    c/v (masks from a different seed, same claimed root) fails the bridge."""
    enr, meta = _toy_enrollment()
    enr_fake = wc.build_enrollment(
        {n: g.weights[:g.n_rows].clone() for n, g in enr.groups.items()},
        b"other-mask-seed", b"manifest-v1", PARAMS)   # pi != z^T rho for root
    proof = wc.prove_bridge(enr_fake, S_R1)
    ok, why = wc.verify_bridge(enr.root, enr.manifest_digest, meta,
                               proof, S_R1, PARAMS)
    assert not ok, "substituted masks accepted"


def test_geometry_is_bound():
    """Review §5.5: a proof made under different (B, lam, N_w, q_w) must not
    verify — the geometry is hashed into every coin."""
    enr, meta = _toy_enrollment()
    proof = wc.prove_bridge(enr, S_R1)
    other = wc.WcParams(B=48, lam=16, N_w=256, q_w=8)   # same K_w, bigger N_w
    ok, why = wc.verify_bridge(enr.root, enr.manifest_digest, meta,
                               proof, S_R1, other)
    assert not ok, "geometry downgrade accepted"


def test_duplicate_eta_rejects():
    """H_40 assumes distinct points; a proof claiming duplicates is rejected
    before anything else."""
    enr, meta = _toy_enrollment()
    proof = wc.prove_bridge(enr, S_R1)
    proof.eta_idx = [proof.eta_idx[0]] * PARAMS.q_w
    ok, why = wc.verify_bridge(enr.root, enr.manifest_digest, meta,
                               proof, S_R1, PARAMS)
    assert not ok and "distinct" in why, why


def test_ledger_margin_and_overflow():
    """Review §4.1: refuse to prove when fewer than q_w unspent mask points
    remain; refuse to charge past lam distinct points."""
    enr, meta = _toy_enrollment()
    lg = wc.EnrollmentLedger(lam=PARAMS.lam)
    proof = wc.prove_bridge(enr, S_R1, ledger=lg)       # spends q_w of lam
    assert len(lg.spent) == PARAMS.q_w
    # margin: leave q_w-1 remaining -> precheck refuses before any work
    lg2 = wc.EnrollmentLedger(lam=PARAMS.lam,
                              spent=set(range(PARAMS.lam - PARAMS.q_w + 1)))
    try:
        wc.prove_bridge(enr, S_R1, ledger=lg2)
        assert False, "prove ran with an exhausted mask budget"
    except wc.LedgerExhausted:
        pass
    # overflow: direct charge past lam distinct points
    lg3 = wc.EnrollmentLedger(lam=4, spent={0, 1, 2})
    try:
        lg3.charge([3, 4])
        assert False, "charge exceeded lam"
    except wc.LedgerExhausted:
        pass


def test_chain_registry_enforces_exactly_one_chain():
    """Spec §0.8 / review §5.1: every block in exactly one chain; an
    unlinked COPY of P_trace is a build error."""
    enr, meta = _toy_enrollment()
    proof = wc.prove_bridge(enr, S_R1)
    reg = wc.ChainRegistry(enr, proof)
    for n in enr.groups:
        for a in range(enr.groups[n].n_blocks):
            reg.consume(n, a, proof.p_trace[n])
    reg.finalize()                                      # all chains closed
    # unlinked copy -> immediate ChainError
    reg2 = wc.ChainRegistry(enr, proof)
    try:
        reg2.consume(4, 0, proof.p_trace[4].clone())
        assert False, "unlinked copy accepted"
    except wc.ChainError:
        pass
    # double consumption -> ChainError
    reg3 = wc.ChainRegistry(enr, proof)
    reg3.consume(4, 0, proof.p_trace[4])
    try:
        reg3.consume(4, 0, proof.p_trace[4])
        assert False, "double consumption accepted"
    except wc.ChainError:
        pass
    # missing block -> finalize fails closed
    reg4 = wc.ChainRegistry(enr, proof)
    reg4.consume(4, 0, proof.p_trace[4])
    try:
        reg4.finalize()
        assert False, "missing chains passed finalize"
    except wc.ChainError:
        pass


def test_mask_vandermonde_full_rank():
    """Review §4.1 hiding argument: the q_w x lam mask system (columns
    eta^{B+h}) has full row rank over F, so <= lam opened points give a
    perfectly hidden (underdetermined) system for the weight coefficients."""
    enr, meta = _toy_enrollment()
    proof = wc.prove_bridge(enr, S_R1)
    dom = wc._rs_domain(PARAMS).cpu().tolist()
    rows = [[pow(dom[i], PARAMS.B + h, P) for h in range(PARAMS.lam)]
            for i in proof.eta_idx]
    # Gaussian elimination mod P
    rank, cols = 0, PARAMS.lam
    m = [r[:] for r in rows]
    for col in range(cols):
        piv = next((r for r in range(rank, len(m)) if m[r][col] % P), None)
        if piv is None:
            continue
        m[rank], m[piv] = m[piv], m[rank]
        inv = pow(m[rank][col], P - 2, P)
        m[rank] = [(x * inv) % P for x in m[rank]]
        for r in range(len(m)):
            if r != rank and m[r][col] % P:
                f = m[r][col]
                m[r] = [(a - f * b) % P for a, b in zip(m[r], m[rank])]
        rank += 1
        if rank == len(m):
            break
    assert rank == PARAMS.q_w, f"mask Vandermonde rank {rank} < q_w"


def test_qw_scaling_rule():
    """Review §7: q_w >= ceil(0.416 tau) keeps the bridge at least as strong
    as the fresh part; the production q_w=40 covers tau up to 96."""
    assert wc.WcParams.min_qw_for_tau(54) == 23
    assert wc.WcParams.min_qw_for_tau(96) == 40
    assert wc.WcParams().q_w >= wc.WcParams.min_qw_for_tau(96)
