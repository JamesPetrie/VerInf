"""The K2 driver's verdicts on the CPU (demo/demo_k2.py), the GPU work faked
where it would run:

- check passes only with every intermediate exact AND every range inside its
  window: exact agreement with a range outside its window fails;
- a reference stopped by an undefined sigmoid lookup fails the check before
  any tape is built, and its record keeps the ranges measured up to the stop;
- an exception in a negative's prove (an injected "CUDA out of memory")
  propagates and is never counted as a rejection, the record written with
  what was measured;
- a negative passes only when its change was applied and the Rust verifier
  rejected it; the statement without YaRN also needs its own-digest
  acceptance."""
import json
import pathlib
import sys

import numpy as np
import pytest

_HERE = pathlib.Path(__file__).resolve().parent
for _p in (_HERE.parent, _HERE, _HERE.parents[1] / "demo"):
    sys.path.insert(0, str(_p))

import _k2_toy as toy                  # noqa: E402
import demo_k2 as dk                   # noqa: E402
import k2_int_reference as ki          # noqa: E402
import k2_loader as kl                 # noqa: E402


@pytest.fixture(scope="module")
def gguf(tmp_path_factory):
    path = tmp_path_factory.mktemp("k2cpu") / "toy.gguf"
    toy.write(path)
    return str(path)


@pytest.fixture
def driver(monkeypatch, gguf, tmp_path):
    """main() on the toy, its environment changes undone afterwards."""
    for k in dk._ENV:
        monkeypatch.setenv(k, "0")
    monkeypatch.setattr(dk, "MLA", toy.TOY_MLA)
    counter = iter(range(100))

    def run(mode, *extra, raises=None):
        """(exit code, record); with `raises`, the exception must propagate
        and the code is None."""
        record = tmp_path / f"rec{next(counter)}.json"
        args = ["--mode", mode, "--from-gguf", gguf, "--prompt-n", "3", "--cont-n", "3",
                "--top-k", str(toy.TOY["k"]), "--record", str(record), *extra]
        if raises is None:
            rc = dk.main(args)
        else:
            with pytest.raises(raises):
                dk.main(args)
            rc = None
        return rc, json.loads(record.read_text())
    return run


def _no_tape(*a, **k):
    raise AssertionError("no tape may be built here")


def test_a_stopped_reference_fails_the_check_and_keeps_its_ranges(driver, monkeypatch):
    monkeypatch.setitem(kl.K2_INT, "sig_bits", 6)       # a 64-entry table: logits fall outside
    monkeypatch.setattr(dk, "build", _no_tape)
    rc, rec = driver("check")
    assert rc == 1 and rec["ok"] is False
    assert "outside the sigmoid table" in rec["reference_error"]
    assert rec["ranges"]["L1.router.logit"]["ok"] is False
    assert rec["ranges"]["L0.n1.S_tot"]["ok"] is True          # measured before the stop
    assert "L1.router.logit" in rec["range_violations"] and "exact" not in rec


def _engine_equal_to_the_reference(monkeypatch):
    """The engine pass faked as the reference's own values: exact by construction."""
    seen = {}
    forward = ki.forward

    def spy(*a, **k):
        out, rg = forward(*a, **k)
        seen.update(out)
        return out, rg
    monkeypatch.setattr(ki, "forward", spy)
    monkeypatch.setattr(dk, "build", lambda *a, **k: dk.Built(None, {}, None, None, {}, [], {},
                                                              {"layers": {}}))
    monkeypatch.setattr(dk, "engine_values", lambda b: (
        {n: np.asarray([int(x) for x in np.ravel(v)], dtype=np.int64) for n, v in seen.items()},
        0))


def test_exact_agreement_with_every_range_inside_passes(driver, monkeypatch):
    _engine_equal_to_the_reference(monkeypatch)
    rc, rec = driver("check")
    assert rc == 0 and rec["ok"] and rec["all_exact"] and not rec["range_violations"]


def test_exact_agreement_with_a_range_outside_its_window_fails(driver, monkeypatch):
    _engine_equal_to_the_reference(monkeypatch)
    monkeypatch.setitem(kl.K2_INT, "output_width", 12)  # rescaled outputs past 2^11
    rc, rec = driver("check")
    assert rec["all_exact"] is True
    assert rec["range_violations"] and rc == 1 and rec["ok"] is False


# ---- prove, with the prover and the verifier faked ------------------------------------

class _Proof:
    def __init__(self, stmt=b"s" * 32):
        self.root_w, self.statement_digest, self.wc_bridge = b"\x01" * 32, stmt, None


class _Tape:
    def __init__(self, prove):
        self.claims, self._prove = [], prove

    def prove(self, **kw):
        return self._prove()


def _fake_prover(monkeypatch, negative_prove, negative_verify=None):
    import core
    import torch

    class WC:
        root, m_w = b"\x01" * 32, 3
    monkeypatch.setattr(core.WeightCommitment, "from_tape", staticmethod(lambda t, c: WC()))
    for f in ("reset_peak_memory_stats", "empty_cache"):
        monkeypatch.setattr(torch.cuda, f, lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(dk, "reveal", lambda b: 7)
    monkeypatch.setattr(dk, "claim_memory", lambda b: {})
    built = iter(range(100))

    def build(*a, tamper=None, **k):
        n = next(built)
        tape = _Tape(_Proof if n == 0 else negative_prove)
        return dk.Built(tape, {}, None, None, {}, [], {}, {"layers": {}})
    monkeypatch.setattr(dk, "build", build)
    verdicts = []

    def verify(claims, proof, cfg, *, root_w, stmt, wc_identity):
        honest = len(verdicts) == 0
        verdicts.append(stmt)
        if not honest and negative_verify is not None:
            return negative_verify()
        verdict = "ACCEPT" if honest else "REJECT"
        return dict(verdict=verdict, returncode=0, output=f"rust_verify: {verdict}",
                    checks=["[OK ] merkle"] + ([] if honest else ["[XX ] lin"]))
    monkeypatch.setattr(dk, "rust_verify_anchored", verify)


def test_a_prover_exception_in_a_negative_propagates_with_the_record(driver, monkeypatch):
    def oom():
        raise RuntimeError("CUDA out of memory")
    _fake_prover(monkeypatch, oom)
    _, rec = driver("prove", "--negatives", "interleave", raises=RuntimeError)
    assert rec["ok"] is False and rec["error"] == "RuntimeError: CUDA out of memory"
    assert rec["verify"]["accept"] is True and rec["negatives"] == {}


def test_a_killed_verifier_in_a_negative_propagates_and_never_counts(driver, monkeypatch):
    """The reviewer's case: a verifier exiting 137 was a passing negative."""
    import torch
    import compute_fns
    from _rust_verify import VerifierFailure

    def tampered_prove():
        hook = compute_fns.WITNESS_TAMPER[("HeadInterleaveClaim", "dst")]
        hook(torch.arange(2 * toy.TOY_MLA["d_nope"], dtype=torch.int64).view(torch.uint64))
        return _Proof()

    def killed():
        raise VerifierFailure(137, "partial output")
    _fake_prover(monkeypatch, tampered_prove, killed)
    _, rec = driver("prove", "--negatives", "interleave", raises=VerifierFailure)
    assert rec["ok"] is False and "exit status 137" in rec["error"]
    assert rec["negatives"] == {} and rec["verify"]["verdict"] == "ACCEPT"


def test_a_negative_with_no_change_applied_fails(driver, monkeypatch):
    _fake_prover(monkeypatch, _Proof)                    # rejected, but nothing was tampered
    rc, rec = driver("prove", "--negatives", "interleave")
    row = rec["negatives"]["interleave"]
    assert row["rejected_by_rust"] and row["applied"] == 0 and not row["passed"]
    assert rc == 1 and rec["ok"] is False


def test_an_applied_and_rejected_negative_passes(driver, monkeypatch):
    import torch
    import compute_fns

    def tampered_prove():
        hook = compute_fns.WITNESS_TAMPER[("HeadInterleaveClaim", "dst")]
        hook(torch.arange(2 * toy.TOY_MLA["d_nope"], dtype=torch.int64).view(torch.uint64))
        return _Proof()
    _fake_prover(monkeypatch, tampered_prove)
    rc, rec = driver("prove", "--negatives", "interleave")
    row = rec["negatives"]["interleave"]
    assert row["applied"] == 1 and row["verdict"] == "REJECT" and row["exit_status"] == 0
    assert row["passed"] and rc == 0 and rec["ok"] is True and row["failed_checks"] == ["[XX ] lin"]
    assert rec["verify"]["verdict"] == "ACCEPT" and rec["verify"]["exit_status"] == 0
    assert ("HeadInterleaveClaim", "dst") not in compute_fns.WITNESS_TAMPER


REJ = dict(verdict="REJECT", exit_status=0)
OWN = dict(own_digest_verdict="ACCEPT", own_digest_exit_status=0)


@pytest.mark.parametrize("row,neg,want", [
    (dict(applied=1, **REJ), "interleave", True),
    (dict(applied=0, **REJ), "interleave", False),
    (dict(applied=2, verdict="ACCEPT", exit_status=0), "wrong-slice", False),
    (dict(applied=2, verdict="REJECT", exit_status=137), "wrong-slice", False),
    (dict(applied=2, rejected_by_rust=True), "wrong-slice", False),       # no explicit verdict
    (dict(applied=4, **REJ, **OWN), "yarn-statement", True),
    (dict(applied=4, **REJ, own_digest_verdict="REJECT", own_digest_exit_status=0),
     "yarn-statement", False),
    (dict(applied=4, **REJ), "yarn-statement", False),
])
def test_negative_verdicts(row, neg, want):
    assert dk.negative_passed(neg, row) is want


def test_prove_verdict_needs_every_planned_negative():
    good = dict(passed=True)
    assert dk.prove_verdict({"verify": {"accept": True}, "negatives": {"interleave": good}},
                            ["interleave"])
    assert not dk.prove_verdict({"verify": {"accept": True}, "negatives": {}}, ["interleave"])
    assert not dk.prove_verdict({"verify": {"accept": False}, "negatives": {}}, [])
