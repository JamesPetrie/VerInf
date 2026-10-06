"""The b_chunk skip is scoped, and the EPHASE switch parses like the others.

`_SKIP_B_CHUNK` used to be set and reset by hand around the prove, with the
reset placed after an assertion: any failure in between leaked True, and the
next in-process compile silently zeroed every nonzero-RHS constraint. The
flag is now held by a context manager that restores the previous value on
every exit. CPU-only (no kernels are launched)."""
import os
import pathlib
import subprocess
import sys
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import core

PROVER_DIR = pathlib.Path(__file__).resolve().parents[1]


def test_skip_is_restored_after_an_exception():
    with patch.object(core, "_SKIP_B_CHUNK", False):
        try:
            with core._skipping_b_chunk():
                assert core._SKIP_B_CHUNK is True
                raise RuntimeError("mid-prove failure")
        except RuntimeError:
            pass
        assert core._SKIP_B_CHUNK is False


def test_nested_skip_restores_the_outer_setting():
    with patch.object(core, "_SKIP_B_CHUNK", False):
        with core._skipping_b_chunk():
            with core._skipping_b_chunk():          # _build_stream_packets inside the prove
                assert core._SKIP_B_CHUNK is True
            assert core._SKIP_B_CHUNK is True       # still on for the rest of the prove
        assert core._SKIP_B_CHUNK is False


def test_prove_streaming_holds_the_skip_and_releases_it_on_failure():
    seen = []

    def body(*_a, **_k):
        seen.append(core._SKIP_B_CHUNK)
        raise RuntimeError("the prove failed before its own reset would have run")

    with (patch.object(core, "_SKIP_B_CHUNK", False),
          patch.object(core, "_prove_streaming_body", side_effect=body)):
        try:
            core.prove_streaming(object(), object())
        except RuntimeError:
            pass
        assert seen == [True]
        assert core._SKIP_B_CHUNK is False


def _ephase_in_subprocess(value):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    if value is None:
        env.pop("LIGERO_EPHASE", None)
    else:
        env["LIGERO_EPHASE"] = value
    out = subprocess.run(
        [sys.executable, "-c", "import core; print(core._EPHASE_ON)"],
        cwd=PROVER_DIR, env=env, capture_output=True, text=True, check=True)
    return out.stdout.strip()


def test_ephase_zero_is_off():
    assert _ephase_in_subprocess(None) == "False"
    assert _ephase_in_subprocess("0") == "False"
    assert _ephase_in_subprocess("off") == "False"
    assert _ephase_in_subprocess("1") == "True"
