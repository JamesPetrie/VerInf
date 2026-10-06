"""The Rust-verifier harness gives a verdict only when verify_proof exited
normally with exactly one explicit `rust_verify: ACCEPT|REJECT` line (CPU,
with stand-in binaries). A verifier killed (status 137 from a shell, or a
SIGKILL), panicking (101), exiting non-zero after printing REJECT, printing
no verdict or two, raises VerifierFailure: before this, "no ACCEPT in the
output" counted as a rejection, so a killed verifier passed a negative."""
import os
import pathlib
import stat
import sys

import pytest

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE))

import _rust_verify as rv          # noqa: E402


def _bin(tmp_path, name, body):
    p = tmp_path / name
    p.write_text("#!/bin/sh\n" + body + "\n")
    p.chmod(p.stat().st_mode | stat.S_IXUSR)
    return str(p)


VERDICTS = {
    "accept": ('echo "  [OK ] merkle"; echo "rust_verify: ACCEPT"; exit 0', "ACCEPT"),
    "reject": ('echo "  [XX ] lin"; echo "rust_verify: REJECT"; echo "python_accept: (none)"; exit 0',
               "REJECT"),
}
FAILURES = {
    "exit 137": ("echo partial output; exit 137", 137),
    "SIGKILL": ("echo partial output; kill -9 $$", -9),
    "panic": ("echo \"thread 'main' panicked at src/handlers.rs: topk: threshold guard\" >&2; exit 101",
              101),
    "silent": ("exit 0", 0),
    "two verdicts": ('echo "rust_verify: REJECT"; echo "rust_verify: ACCEPT"; exit 0', 0),
    "REJECT then exit 1": ('echo "rust_verify: REJECT"; exit 1', 1),
    "unknown verdict": ('echo "rust_verify: MAYBE"; exit 0', 0),
}


@pytest.mark.parametrize("name", sorted(VERDICTS))
def test_an_explicit_verdict_after_a_normal_exit(tmp_path, name):
    body, want = VERDICTS[name]
    v = rv.run_verify_proof([_bin(tmp_path, "vp", body), "proof.json"])
    assert v["verdict"] == want and v["returncode"] == 0
    assert f"rust_verify: {want}" in v["output"]


@pytest.mark.parametrize("name", sorted(FAILURES))
def test_no_verdict_is_a_failure_never_a_rejection(tmp_path, name):
    body, status = FAILURES[name]
    with pytest.raises(rv.VerifierFailure) as e:
        rv.run_verify_proof([_bin(tmp_path, "vp", body), "proof.json"])
    assert e.value.returncode == status
    assert f"exit status {status}" in str(e.value)


class _Proof:
    seeds = {"s_op": b"\x01" * 32, "s_comb": b"\x02" * 32, "s_col": b"\x03" * 32}


def _through(monkeypatch, tmp_path, body, fn):
    """rust_verify with the dump stubbed and the stand-in binary on
    LIGERO_VERIFY_PROOF: the path a negative takes."""
    import core
    import proof_dump
    monkeypatch.setattr(proof_dump, "dump_proof",
                        lambda path, *a, **k: pathlib.Path(path).write_text("{}"))
    monkeypatch.setenv("LIGERO_VERIFY_PROOF", _bin(tmp_path, "vp", body))
    cfg = core.LigeroConfig(ELL=16, K_DEG=16, N_LIG=64, T_QUERIES=4)
    return rv.rust_verify([], _Proof(), None, cfg)


@pytest.mark.parametrize("fn", ["rust_verify"])
def test_a_killed_verifier_raises_through_the_helpers(monkeypatch, tmp_path, fn):
    with pytest.raises(rv.VerifierFailure) as e:
        _through(monkeypatch, tmp_path, FAILURES["exit 137"][0], fn)
    assert e.value.returncode == 137
    assert not list(tmp_path.glob("*.json"))             # the dump is removed either way


def test_the_helpers_return_explicit_verdicts(monkeypatch, tmp_path):
    acc, out = _through(monkeypatch, tmp_path, VERDICTS["reject"][0], "rust_verify")
    assert acc is False and "rust_verify: REJECT" in out
    acc, out = _through(monkeypatch, tmp_path, VERDICTS["accept"][0], "rust_verify")
    assert acc is True and "rust_verify: ACCEPT" in out
