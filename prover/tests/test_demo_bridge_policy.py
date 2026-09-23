"""The production driver keeps both anchors under the bridge (CPU; review
2026-09-23 finding 9 and its dense-weight note).

Bridge mode used to drop --weight-commitment: the dense weights, still
persistent rows of the W block, were committed fresh on every proof under a
root nothing outside the proof vouched for, and the run's end read the absent
commitment's ledger after publishing the proof. The policy check now asks for
the dense enrollment in both modes, before anything is loaded."""
import os
import subprocess
import sys

REPO = os.path.join(os.path.dirname(__file__), "..", "..")


def _demo(*args):
    r = subprocess.run([sys.executable, os.path.join(REPO, "demo", "demo_maverick_full.py"),
                        "--from-gguf", "/nonexistent.gguf", "--allow-dev-config", *args],
                       capture_output=True, text=True, timeout=300)
    return r.returncode, r.stdout + r.stderr


def test_bridge_mode_refuses_without_the_dense_enrollment():
    rc, out = _demo("--wc-bridge", "--public-sz", "1",
                    "--admission-report", "/nonexistent.json",
                    "--dump-proof", "/nonexistent-proof.json")
    assert "refusing to prove: missing --weight-commitment, --expected-weight-root" in out, out
    assert "enrolled root" not in out          # refused before any model work


def test_both_modes_ask_for_the_same_anchors():
    for mode in ((), ("--wc-bridge",)):
        rc, out = _demo(*mode, "--public-sz", "1", "--dump-proof", "/nonexistent-proof.json")
        assert ("missing --weight-commitment, --expected-weight-root, "
                "--admission-report") in out, (mode, out)
