"""The Python bridge verifier's own checks, on CPU (review of the fix pass,
2026-09-23, findings 1 and 2).

The enrollment is built by Horner evaluation on the host and only the CUDA
computation of the fixed public RS domain is replaced by the same values, so
the verifier, its hashes, transcripts, identity and path checks run
unmodified. Two false accepts are locked here: a group_meta that repeats a
smaller width (the bridge equation then combined one output of four), and
q_w = 0 (no openings, so nothing checked). The rest are the Rust twin's shape
checks, which the Python twin now makes too: a malformed proof rejects
instead of raising."""
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

import wc_bridge as wc

P = wc.P
PARAMS = wc.WcParams(B=12, lam=4, N_w=64, q_w=40)
LAYOUT = {4: [12]}
MANIFEST = b"m" * 32
S_OP, S_BIND = b"o" * 32, b"b" * 32
RHO = {4: wc.pr.op_vec(S_OP, 0, "rho-w4", 4)}
OMEGA = pow(7, (P - 1) // PARAMS.N_w, P)
DOMAIN = [pow(OMEGA, i, P) for i in range(PARAMS.N_w)]
COEFFS = [[1 + 100 * j + k for k in range(PARAMS.K_w)] for j in range(4)]
T_COLS = 54


def _eval(c, x):
    out = 0
    for a in reversed(c):
        out = (out * x + a) % P
    return out


COLUMNS = [torch.tensor([_eval(c, x) for c in COEFFS], dtype=torch.uint64) for x in DOMAIN]
LEVELS = wc._tree([wc._leaf(col) for col in COLUMNS])
ROOT = LEVELS[-1][0]
TRUSTED = wc.enrollment_identity(ROOT, MANIFEST, PARAMS, LAYOUT)


def _proof(used_width=4, params=PARAMS):
    """A hosted-mode proof whose projection combines the first used_width
    outputs (all four is the honest one)."""
    proj = [sum(RHO[4][j] * COEFFS[j][k] for j in range(used_width)) % P
            for k in range(params.K_w)]
    pt = {4: torch.tensor(proj[:params.B], dtype=torch.uint64)}
    pi = {4: torch.tensor([proj[params.B:]], dtype=torch.uint64)}
    late = wc.hosted_s_late(S_BIND, ROOT, MANIFEST, pt, pi, params)
    alpha = wc.pr.challenge(late, 0, "alpha")
    c = [(alpha * x) % P for x in proj]
    eta = wc.pr.random_columns_n(wc.pr.fs_seed("wc/eta", late), params.q_w, params.N_w)
    return wc.BridgeProof({n: list(r) for n, r in RHO.items()}, pt, pi, c,
                          [_eval(c, DOMAIN[i]) for i in eta], eta,
                          {i: COLUMNS[i] for i in eta},
                          {i: wc._path(LEVELS, i) for i in eta})


def _verify(pf, meta_width=4, params=PARAMS, t_cols=T_COLS):
    with patch.object(wc, "_rs_domain",
                      return_value=torch.tensor(DOMAIN, dtype=torch.uint64)):
        return wc.verify_bridge_hosted(ROOT, MANIFEST, {4: (1, meta_width)}, pf,
                                       S_BIND, RHO, params, trusted_identity=TRUSTED,
                                       layout=LAYOUT, t_cols=t_cols)


def test_honest_projection_accepts_and_a_partial_one_rejects():
    assert _verify(_proof()) == (True, "ACCEPT")
    ok, why = _verify(_proof(used_width=1))
    assert not ok and "bridge equation" in why, why


def test_group_metadata_cannot_shrink_the_width():
    # finding 1: {4: (1, 1)} made the equation combine output 0 only
    ok, why = _verify(_proof(used_width=1), meta_width=1)
    assert not ok and "group metadata" in why, why


def test_opening_count_is_checked_though_outside_the_identity():
    # finding 2: q_w is per proof, so the identity matches at any q_w
    for q_w, reason in ((0, "out of range"), (20, "below the floor"),
                        (65, "out of range")):
        params = wc.WcParams(B=12, lam=4, N_w=64, q_w=q_w)
        assert wc.enrollment_identity(ROOT, MANIFEST, params, LAYOUT) == TRUSTED
        pf = _proof(used_width=1, params=params) if q_w <= 64 else _proof(used_width=1)
        ok, why = _verify(pf, params=params)
        assert not ok and reason in why, (q_w, why)
    # the floor follows the verifier's own column count, not the proof's
    params = wc.WcParams(B=12, lam=4, N_w=64, q_w=23)
    assert _verify(_proof(params=params), params=params, t_cols=54)[0]
    assert "below the floor" in _verify(_proof(params=params), params=params, t_cols=56)[1]


def test_malformed_arrays_reject_instead_of_raising():
    pf = _proof()
    pf.p_trace = {4: pf.p_trace[4][:-1]}
    assert _verify(pf) == (False, "p_trace shape")
    pf = _proof()
    pf.pi = {4: torch.zeros(2, PARAMS.lam, dtype=torch.uint64)}
    assert _verify(pf) == (False, "pi shape")
    pf = _proof()
    pf.v = pf.v[:-1]
    assert _verify(pf) == (False, "v/eta are not q_w field elements")
    pf = _proof()
    del pf.opened[pf.eta_idx[3]]
    assert _verify(pf) == (False, "column or path missing at eta[3]")
    pf = _proof()
    i = pf.eta_idx[0]
    pf.opened[i] = torch.cat([pf.opened[i], torch.zeros(1, dtype=torch.uint64)])
    assert _verify(pf) == (False, "column length at eta[0]")


def test_malformed_forms_reject_before_identity_or_transcript_work():
    """Review of the second round, 2026-09-23: the form is checked first, so
    none of these reaches the identity hash, the R2 commitment or the
    aggregation (they raised ZeroDivisionError, KeyError and MemoryError)."""
    # B = 0 divided in the identity's block count
    zero_b = wc.WcParams(B=0, lam=16, N_w=64, q_w=40)
    assert _verify(_proof(), params=zero_b) == (
        False, "lam and B must be positive: the masks are the hiding")
    # an absent group was read by the R2 commitment
    pf = _proof()
    del pf.pi[4]
    assert _verify(pf) == (False, "pi groups are not the enrolled widths")
    pf = _proof()
    pf.p_trace[5] = pf.p_trace[4]
    assert _verify(pf) == (False, "p_trace groups are not the enrolled widths")
    # the standalone entrypoint reads rho before the transcript
    pf = _proof()
    del pf.rho[4]
    ok, why = wc.verify_bridge(ROOT, MANIFEST, {4: (1, 4)}, pf, b"r" * 32, PARAMS,
                               trusted_identity=TRUSTED, layout=LAYOUT, t_cols=T_COLS)
    assert (ok, why) == (False, "rho groups are not the enrolled widths")
    # the right element count in the wrong rank reached the aggregation
    pf = _proof()
    pf.p_trace = {4: pf.p_trace[4].reshape(2, 6)}
    assert _verify(pf) == (False, "p_trace shape")
    pf = _proof()
    pf.p_trace = {4: pf.p_trace[4].to(torch.int64)}
    assert _verify(pf) == (False, "p_trace shape")
    # nested or non-integer scalars
    pf = _proof()
    pf.c = [[x] for x in pf.c]
    assert _verify(pf) == (False, "c is not K_w field elements")
    pf = _proof()
    i = pf.eta_idx[2]
    pf.opened[i] = pf.opened[i].reshape(-1, 1)
    assert _verify(pf) == (False, "column shape at eta[2]")
    # malformed group metadata values are a reject too
    assert _verify(_proof(), meta_width=None)[1].startswith("width 4: group metadata")
