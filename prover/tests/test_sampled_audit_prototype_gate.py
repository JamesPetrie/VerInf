"""The runtime sampled audit runs only as a labeled prototype (CPU; review
2026-09-23 findings 5 and 6).

Two properties a verified audit of model inference needs are not checked by
the runtime: that the weights its local checks read are the enrolled model's,
and that a wire reused across windows kept its value. Until they are, the
runtime refuses to start unless the caller acknowledges that it is a
prototype, before any expensive setup, and every result says what it is. The
label on a real run's result is checked on a card in
test_sampled_claim_runtime.test_result_is_labeled_a_prototype."""
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import sampled_claim_runtime as rt

REPO = os.path.join(os.path.dirname(__file__), "..", "..")


def test_runtime_refuses_without_the_acknowledgement():
    try:
        # nothing is touched before the refusal: no tape, no card
        rt.ClaimWindowAudit(None, None, b"v" * 32, b"p" * 32, b"m" * 32)
    except ValueError as exc:
        msg = str(exc)
        assert "prototype" in msg and "weight-to-enrollment binding" in msg \
            and "cross-window value consistency" in msg, msg
    else:
        raise AssertionError("the prototype ran without acknowledgement")


def test_labels_name_both_unchecked_properties():
    labels = rt.prototype_labels()
    assert labels["prototype"] is True and labels["verified_inference"] is False
    assert "not verified model inference" in labels["notice"]
    assert [u.split(":")[0] for u in labels["unchecked"]] == [
        "weight-to-enrollment binding", "cross-window value consistency"]


def _demo(*args):
    r = subprocess.run([sys.executable, os.path.join(REPO, "demo", "demo_maverick_full.py"),
                        "--from-gguf", "/nonexistent.gguf", "--allow-dev-config", *args],
                       capture_output=True, text=True, timeout=300)
    return r.stdout + r.stderr


def test_driver_refuses_before_any_setup():
    out = _demo("--sampled-audit-out", "/nonexistent-audit.json")
    assert "refusing sampled audit: the runtime is a prototype" in out, out
    assert "--sampled-audit-prototype" in out
    assert "build" not in out and "enrolled root" not in out, out


def test_driver_flag_passes_the_gate_only():
    # with the acknowledgement the next policy check speaks (missing inputs)
    out = _demo("--sampled-audit-out", "/nonexistent-audit.json",
                "--sampled-audit-prototype")
    assert "the runtime is a prototype" not in out, out
    assert "refusing sampled audit: missing" in out, out
