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


S_R1 = b"r" * 32            # the standalone entrypoint's transcript state after R1


def _proof(used_width=4, params=PARAMS, mode="hosted"):
    """A proof whose projection combines the first used_width outputs (all
    four is the honest one). Hosted: rho is the host's (RHO) and the late seed
    the hosted one; standalone: both derived from S_R1 as verify_bridge does."""
    if mode == "hosted":
        rho = {4: list(RHO[4])}
    else:
        s_rho = wc.pr.fs_seed("wc/rho", S_R1, ROOT, MANIFEST, wc._params_bytes(params))
        rho = {4: wc.pr.op_vec(s_rho, 0, "rho", 4)}
    proj = [sum(rho[4][j] * COEFFS[j][k] for j in range(used_width)) % P
            for k in range(params.K_w)]
    pt = {4: torch.tensor(proj[:params.B], dtype=torch.uint64)}
    pi = {4: torch.tensor([proj[params.B:]], dtype=torch.uint64)}
    if mode == "hosted":
        late = wc.hosted_s_late(S_BIND, ROOT, MANIFEST, pt, pi, params)
    else:
        late = wc.pr.fs_seed("wc/late", s_rho, wc._commit_r2(pt, pi))
    alpha = wc.pr.challenge(late, 0, "alpha")
    c = [(alpha * x) % P for x in proj]
    eta = wc.pr.random_columns_n(wc.pr.fs_seed("wc/eta", late), params.q_w, params.N_w)
    return wc.BridgeProof(rho, pt, pi, c,
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
    ok, why = wc.verify_bridge(ROOT, MANIFEST, {4: (1, 4)}, pf, S_R1, PARAMS,
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


# -- every input, every malformed form: a reject, never an exception ---------
# Review of the third round, 2026-09-24: a scalar rho, a malformed path step
# and a short root still raised. Rather than lock those three, replace each
# input the verifier reads, whole and element by element, with each of these
# and require a clean (False, reason) from both entrypoints.
BAD = [None, 0, 1, -1, 2 ** 70, 1.5, 1.0, 4.0, True, "x", b"", b"x" * 31, b"x" * 33,
       [], [1], [[1]], (1,), {}, {4: None},
       torch.zeros(0, dtype=torch.uint64), torch.zeros(3, dtype=torch.int64),
       torch.zeros(2, 2, dtype=torch.uint64), torch.zeros(3, dtype=torch.float32)]


HONEST_META = {4: (1, 4)}


def _hosted(pf, root=ROOT, manifest=MANIFEST, meta=HONEST_META, params=PARAMS,
            trusted=TRUSTED, layout=LAYOUT, t_cols=T_COLS):
    with patch.object(wc, "_rs_domain",
                      return_value=torch.tensor(DOMAIN, dtype=torch.uint64)):
        return wc.verify_bridge_hosted(root, manifest, meta,
                                       pf, S_BIND, RHO, params, trusted_identity=trusted,
                                       layout=layout, t_cols=t_cols)


def _standalone(pf, root=ROOT, manifest=MANIFEST, meta=HONEST_META, params=PARAMS,
                trusted=TRUSTED, layout=LAYOUT, t_cols=T_COLS):
    with patch.object(wc, "_rs_domain",
                      return_value=torch.tensor(DOMAIN, dtype=torch.uint64)):
        return wc.verify_bridge(root, manifest, meta,
                                pf, S_R1, params, trusted_identity=trusted,
                                layout=layout, t_cols=t_cols)


def _rejects(label, run):
    try:
        out = run()
    except Exception as exc:            # the property under test
        raise AssertionError(f"{label}: raised {type(exc).__name__}: {exc}") from exc
    assert isinstance(out, tuple) and len(out) == 2 and out[0] is False \
        and isinstance(out[1], str), f"{label}: {out!r}"


def _same(a, b) -> bool:
    """Equal in type and value (a mutation to an equal value is no mutation)."""
    if type(a) is not type(b) or isinstance(a, torch.Tensor):
        return False
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def _hashable(x) -> bool:
    try:
        hash(x)
    except TypeError:
        return False
    return True


def _params_with(**kw):
    """A WcParams carrying arbitrary field values (its own __post_init__
    asserts would stop them; a proof's geometry is not so polite)."""
    p = object.__new__(wc.WcParams)
    for f in ("B", "lam", "N_w", "q_w"):
        object.__setattr__(p, f, kw.get(f, getattr(PARAMS, f)))
    return p


N_CASES_FLOOR = 1700          # 1,767 today: the loops ran (the count prints with -s)


def test_both_honest_baselines_accept():
    # the fuzz below starts from these: every rejection there is the mutation's
    assert _hosted(_proof(mode="hosted")) == (True, "ACCEPT")
    assert _standalone(_proof(mode="standalone")) == (True, "ACCEPT")


def test_no_malformed_input_raises():
    test_both_honest_baselines_accept()
    n_cases = 0
    for entry, mode in ((_hosted, "hosted"), (_standalone, "standalone")):
        def check(label, mutate=None, **args):
            nonlocal n_cases
            pf = _proof(mode=mode)
            if mutate is not None and mutate(pf) is False:
                return                      # the "bad" value was the honest one
            _rejects(f"{entry.__name__}: {label}", lambda: entry(pf, **args))
            n_cases += 1
        for bad in BAD:
            r = repr(bad)[:24]
            for field in ("rho", "p_trace", "pi", "c", "v", "eta_idx", "opened", "paths"):
                check(f"{field} = {r}", lambda pf, f=field: setattr(pf, f, bad))
            for field in ("rho", "p_trace", "pi"):
                check(f"{field}[4] = {r}", lambda pf, f=field: getattr(pf, f).__setitem__(4, bad))
            for field in ("c", "v", "eta_idx"):
                def one(pf, f=field):
                    seq = list(getattr(pf, f))
                    if _same(seq[1], bad):
                        return False
                    seq[1] = bad
                    setattr(pf, f, seq)
                check(f"{field}[1] = {r}", one)
            def rho0(pf):
                if _same(pf.rho[4][0], bad):
                    return False
                pf.rho[4] = [bad] + pf.rho[4][1:]
            check(f"rho[4][0] = {r}", rho0)
            check(f"opened[eta0] = {r}", lambda pf: pf.opened.__setitem__(pf.eta_idx[0], bad))
            check(f"paths[eta0] = {r}", lambda pf: pf.paths.__setitem__(pf.eta_idx[0], bad))
            for k, form in ((0, lambda s: bad), (1, lambda s: (bad, s[1])),
                            (2, lambda s: (s[0], bad))):
                def step(pf, form=form):
                    path = list(pf.paths[pf.eta_idx[0]])
                    new = form(path[2])
                    if _same(new, path[2]):
                        return False
                    path[2] = new
                    pf.paths[pf.eta_idx[0]] = path
                check(f"path step form {k} = {r}", step)
            # the inputs the Rust verifier reads off the wire: root, manifest,
            # group metadata and geometry
            check(f"root = {r}", root=bad)
            check(f"manifest = {r}", manifest=bad)
            check(f"group_meta = {r}", meta=bad)
            check(f"group_meta[4] = {r}", meta={4: bad})
            if not _same(bad, 1):
                check(f"group_meta[4][0] = {r}", meta={4: (bad, 4)})
            check(f"group_meta[4][1] = {r}", meta={4: (1, bad)})
            # a key of the wrong type (4.0 and True compare equal to ints)
            if _hashable(bad):
                check(f"group_meta key = {r}", meta={bad: (1, 4)})
                for field in ("rho", "p_trace", "pi"):
                    def rekey(pf, f=field):
                        d = getattr(pf, f)
                        d[bad] = d.pop(4)
                    check(f"{field} key = {r}", rekey)
                for field in ("opened", "paths"):
                    def rekey_eta(pf, f=field):
                        d = getattr(pf, f)
                        i = pf.eta_idx[0]
                        if _same(bad, i):
                            return False
                        d[bad] = d.pop(i)
                    check(f"{field} key eta0 = {r}", rekey_eta)
            for f in ("B", "lam", "N_w", "q_w"):
                check(f"params.{f} = {r}", params=_params_with(**{f: bad}))
            # and the verifier's own policy, malformed by a caller
            check(f"trusted = {r}", trusted=bad)
            check(f"layout = {r}", layout=bad)
            check(f"layout[4] = {r}", layout={4: bad})
            if not (type(bad) is int and 0 <= bad < 1 << 63):   # 0 and 1 are valid counts
                check(f"t_cols = {r}", t_cols=bad)
    print(f"{n_cases} malformed inputs, each a clean reject")
    assert n_cases >= N_CASES_FLOOR, n_cases
