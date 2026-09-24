"""WC-LCRL-STC integration brick 2: the weight_enrollment flag inside the
real 5-round prove_streaming.

With the flag: the routed claim's own R1 coin (ch0 rho) drives bridge_r2,
the hosted late coin is derived from the real s_bind + the bridge R2
commitment, and the proof carries a python sidecar (proof.wc_bridge).  The
Rust wire proof is UNCHANGED (twin lands in brick 4), so the existing
ACCEPT gate must stay green alongside the hosted bridge verify.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

import core
import wc_bridge as wc
from routed_projected import (RoutedProjectedMatmulClaim, routed_sample,
                              routed_projected_matmul)
from tape import Tape
from tests.test_routed_projected import _build, _u64, CFG, T, K, J, E
from tests._rust_verify import rust_verify_tape


def _build_bridged(routes=(0, 1, 0), w_bump=0):
    """test_routed_projected._build with use_bridge=1 on the claim."""
    core._COSET_POWERS_K_CACHE.clear()
    tape = Tape(CFG, lazy=True)
    X = torch.arange(1, T * K + 1).reshape(T, K)
    W = torch.arange(1, E * K * J + 1).reshape(E, K * J) + w_bump
    M = torch.zeros(T, E, dtype=torch.int64)
    for t, e in enumerate(routes):
        M[t, e] = 1
    x = tape.commit("X", _u64(X.reshape(-1)), (T, K))
    m = tape.commit("M", _u64(M.reshape(-1)), (T, E))
    w = [tape.commit(f"W{e}", _u64(W[e]), (K, J)) for e in range(E)]
    routed_projected_matmul(tape, x, m, w, T=T, K=K, J=J, E=E,
                            use_bridge=True)
    return tape

PARAMS = wc.WcParams(B=48, lam=16, N_w=128, q_w=8)


def _enroll(tape, claim):
    def _flat(wv):
        val = tape.inputs[wv]
        return (val() if callable(val) else val)
    w_rows = torch.cat([_flat(wv).reshape(K, J) for wv in claim.W])
    return wc.build_enrollment({J: w_rows.cuda()}, b"flag-mask",
                               b"flag-manifest", PARAMS)


def _layout(tape):
    """The verifier's own layout: from the claim set, never the proof."""
    return wc.claim_layout(wc.bridged_claim_map(tape.claims))


def _prove_with_flag():
    tape = _build_bridged()
    claim = next(c for c in tape.claims
                 if isinstance(c, RoutedProjectedMatmulClaim))
    enr = _enroll(tape, claim)
    proof = tape.prove(weight_enrollment=enr)
    return tape, proof, enr


def test_flagged_proof_keeps_rust_accept_and_bridge_accepts():
    tape, proof, enr = _prove_with_flag()
    acc, msg = rust_verify_tape(tape, proof, seed=None)
    assert acc, f"rust ACCEPT lost under the flag: {msg}"
    sc = proof.wc_bridge
    ci = sc["claim_index"]
    claim = tape.claims[ci]
    rho = routed_sample(claim, ci, proof.seeds["s_op"])
    ok, why = wc.verify_bridge_hosted(sc["root"], sc["manifest_digest"],
                                      sc["group_meta"], sc["bridge"],
                                      proof.seeds["s_bind"], {J: rho},
                                      sc["params"],
                                      trusted_identity=sc["identity"],
                                      layout=_layout(tape),
                                      t_cols=CFG.T_QUERIES)
    assert ok, f"hosted bridge REJECT: {why}"
    # and the bridge P_trace is the claim's projection (link invariant)
    assert sc["bridge"].p_trace[J].numel() >= E * K


def test_flag_off_has_no_sidecar():
    tape, y, X, W, M = _build()
    proof = tape.prove()
    assert not hasattr(proof, "wc_bridge"), \
        "flag off must leave the proof exactly as before"


def test_hosted_bridge_rejects_foreign_rho():
    tape, proof, enr = _prove_with_flag()
    sc = proof.wc_bridge
    bad_rho = {J: [1] * J}
    ok, why = wc.verify_bridge_hosted(sc["root"], sc["manifest_digest"],
                                      sc["group_meta"], sc["bridge"],
                                      proof.seeds["s_bind"], bad_rho,
                                      sc["params"],
                                      trusted_identity=sc["identity"],
                                      layout=_layout(tape),
                                      t_cols=CFG.T_QUERIES)
    assert not ok and "rho" in why, why


def test_hosted_bridge_rejects_tampered_p_trace():
    tape, proof, enr = _prove_with_flag()
    sc = proof.wc_bridge
    br = sc["bridge"]
    br.p_trace[J].view(torch.int64)[3] ^= 1      # bit-flip tamper, no overflow
    ci = sc["claim_index"]
    rho = routed_sample(tape.claims[ci], ci, proof.seeds["s_op"])
    ok, why = wc.verify_bridge_hosted(sc["root"], sc["manifest_digest"],
                                      sc["group_meta"], br,
                                      proof.seeds["s_bind"], {J: rho},
                                      sc["params"],
                                      trusted_identity=sc["identity"],
                                      layout=_layout(tape),
                                      t_cols=CFG.T_QUERIES)
    assert not ok, "tampered hosted P_trace accepted"


def test_hosted_bridge_rejects_foreign_enrollment():
    """A proof whose bridge ran against different weights must fail against
    the true enrollment root (model-substitution, hosted mode)."""
    tape = _build_bridged()
    claim = next(c for c in tape.claims
                 if isinstance(c, RoutedProjectedMatmulClaim))
    enr_true = _enroll(tape, claim)
    tape2 = _build_bridged(w_bump=3)              # different weights
    claim2 = next(c for c in tape2.claims
                  if isinstance(c, RoutedProjectedMatmulClaim))
    enr_fake = _enroll(tape2, claim2)
    proof = tape2.prove(weight_enrollment=enr_fake)
    sc = proof.wc_bridge
    ci = sc["claim_index"]
    rho = routed_sample(tape2.claims[ci], ci, proof.seeds["s_op"])
    ok, why = wc.verify_bridge_hosted(enr_true.root, enr_true.manifest_digest,
                                      sc["group_meta"], sc["bridge"],
                                      proof.seeds["s_bind"], {J: rho},
                                      sc["params"],
                                      trusted_identity=enr_true.identity(),
                                      layout=_layout(tape2),
                                      t_cols=CFG.T_QUERIES)
    assert not ok, "bridge against a substituted model accepted"


def test_two_bridged_claims_share_one_enrollment():
    """Spec 0.2: two width-J routed matmuls, ONE shared rho, ONE enrollment,
    one q_w-point opening — end-to-end through the Rust twin."""
    core._COSET_POWERS_K_CACHE.clear()
    tape = Tape(CFG, lazy=True)
    X = torch.arange(1, T * K + 1).reshape(T, K)
    W1 = torch.arange(1, E * K * J + 1).reshape(E, K * J)
    W2 = W1 + 7
    M = torch.zeros(T, E, dtype=torch.int64)
    for t, e in enumerate((0, 1, 0)):
        M[t, e] = 1
    x = tape.commit("X", _u64(X.reshape(-1)), (T, K))
    m = tape.commit("M", _u64(M.reshape(-1)), (T, E))
    wa = [tape.commit(f"Wa{e}", _u64(W1[e]), (K, J)) for e in range(E)]
    wb = [tape.commit(f"Wb{e}", _u64(W2[e]), (K, J)) for e in range(E)]
    routed_projected_matmul(tape, x, m, wa, T=T, K=K, J=J, E=E,
                            use_bridge=True)
    routed_projected_matmul(tape, x, m, wb, T=T, K=K, J=J, E=E,
                            use_bridge=True)
    enr = wc.enroll_tape(tape, b"multi-mask", b"multi-manifest", PARAMS)
    assert enr.groups[J].n_rows == 2 * E * K
    proof = tape.prove(weight_enrollment=enr)
    # both claims sampled the SAME rho (shared per width)
    cis = [ci for ci, c in enumerate(tape.claims)
           if isinstance(c, RoutedProjectedMatmulClaim)]
    r0 = routed_sample(tape.claims[cis[0]], cis[0], proof.seeds["s_op"])
    r1 = routed_sample(tape.claims[cis[1]], cis[1], proof.seeds["s_op"])
    assert r0 == r1, "bridged claims of one width must share rho"
    acc, msg = rust_verify_tape(tape, proof, seed=None)
    assert acc, f"two-claim bridged proof rejected by Rust: {msg}"
    # ledger economics: one opening set for BOTH claims
    assert len(proof.wc_bridge["bridge"].eta_idx) == PARAMS.q_w


def test_external_weights_leave_the_witness():
    """The W-block removal: bridged weights as tape.external() — prover
    inputs, NOT committed rows. The proof shrinks and still ACCEPTs through
    the Rust twin (the enrollment, not the commitment, authenticates W)."""
    core._COSET_POWERS_K_CACHE.clear()
    tape = Tape(CFG, lazy=True)
    X = torch.arange(1, T * K + 1).reshape(T, K)
    W = torch.arange(1, E * K * J + 1).reshape(E, K * J)
    M = torch.zeros(T, E, dtype=torch.int64)
    for t, e in enumerate((0, 1, 0)):
        M[t, e] = 1
    x = tape.commit("X", _u64(X.reshape(-1)), (T, K))
    m = tape.commit("M", _u64(M.reshape(-1)), (T, E))
    w = [tape.external(f"W{e}", _u64(W[e]), (K, J)) for e in range(E)]
    routed_projected_matmul(tape, x, m, w, T=T, K=K, J=J, E=E,
                            use_bridge=True)
    enr = wc.enroll_tape(tape, b"ext-mask", b"ext-manifest", PARAMS)
    proof = tape.prove(weight_enrollment=enr)
    acc, msg = rust_verify_tape(tape, proof, seed=None)
    assert acc, f"external-weights proof rejected: {msg}"
    # the committed witness really shrank: compare p1 rows against the
    # committed-W variant of the same tape
    tape2 = _build_bridged()
    enr2 = _enroll(tape2, next(c for c in tape2.claims
                               if isinstance(c, RoutedProjectedMatmulClaim)))
    proof2 = tape2.prove(weight_enrollment=enr2)
    rows = lambda p: max(v for cols in (p.opened_p1,) for v in
                         [next(iter(cols.values())).numel()])
    assert rows(proof) < rows(proof2), (
        f"witness did not shrink: {rows(proof)} vs {rows(proof2)} p1 rows")


def test_external_weights_without_bridge_refuse():
    """Fail-closed: uncommitted weights under a non-bridged claim are a
    soundness hole and must not even build."""
    core._COSET_POWERS_K_CACHE.clear()
    tape = Tape(CFG, lazy=True)
    X = torch.arange(1, T * K + 1).reshape(T, K)
    W = torch.arange(1, E * K * J + 1).reshape(E, K * J)
    M = torch.zeros(T, E, dtype=torch.int64)
    for t, e in enumerate((0, 1, 0)):
        M[t, e] = 1
    x = tape.commit("X", _u64(X.reshape(-1)), (T, K))
    m = tape.commit("M", _u64(M.reshape(-1)), (T, E))
    w = [tape.external(f"W{e}", _u64(W[e]), (K, J)) for e in range(E)]
    try:
        routed_projected_matmul(tape, x, m, w, T=T, K=K, J=J, E=E)
        assert False, "external weights accepted without the bridge"
    except AssertionError as e:
        assert "use_bridge" in str(e)


def test_lazy_enrollment_end_to_end():
    """Production path: LazyEnrollment (no resident weights, streaming root
    + streaming openings, GPU BLAKE3 inner digests) must produce the same
    ACCEPT through the Rust twin as the resident enrollment."""
    core._COSET_POWERS_K_CACHE.clear()
    tape = Tape(CFG, lazy=True)
    X = torch.arange(1, T * K + 1).reshape(T, K)
    W = torch.arange(1, E * K * J + 1).reshape(E, K * J)
    M = torch.zeros(T, E, dtype=torch.int64)
    for t, e in enumerate((0, 1, 0)):
        M[t, e] = 1
    x = tape.commit("X", _u64(X.reshape(-1)), (T, K))
    m = tape.commit("M", _u64(M.reshape(-1)), (T, E))
    w = [tape.external(f"W{e}", _u64(W[e]), (K, J)) for e in range(E)]
    routed_projected_matmul(tape, x, m, w, T=T, K=K, J=J, E=E,
                            use_bridge=True)
    enr = wc.lazy_enroll_tape(tape, b"lazy-mask", b"lazy-manifest", PARAMS)
    # same manifest/mask/weights => same root as the resident build
    enr_res = wc.enroll_tape(tape, b"lazy-mask", b"lazy-manifest", PARAMS)
    assert enr.root == enr_res.root, "lazy root != resident root"
    # one manifest digest and one layout for both builds: one identity
    assert enr.identity() == enr_res.identity(), "lazy identity != resident"
    proof = tape.prove(weight_enrollment=enr)
    acc, msg = rust_verify_tape(tape, proof, seed=None)
    assert acc, f"lazy-enrollment proof rejected: {msg}"


def test_repeated_bridged_proofs_keep_one_statement():
    """Review 2026-09-23 finding 8: the first proof attaches its P_trace pin to
    the claim; a second proof of the same tape must state the same claims
    (the pin is prover state, not statement), and still verify."""
    tape, first, enr = _prove_with_flag()
    second = tape.prove(weight_enrollment=enr)
    assert first.statement_digest == second.statement_digest, \
        "a tape's statement digest changed after its first bridged proof"
    acc, msg = rust_verify_tape(tape, second, seed=None)
    assert acc, f"second bridged proof rejected: {msg}"
