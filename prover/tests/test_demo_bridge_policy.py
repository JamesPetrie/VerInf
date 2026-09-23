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


def test_bridge_mode_saves_the_proof_after_both_ledgers(tmp_path):
    """Review of the fix pass, 2026-09-23, finding 3: the run past the prove.
    Model loading, proving, admission and the output reservation are stubbed;
    main's own control flow runs from the policy check to the dump, so a
    bookkeeping error after a long proof cannot leave no proof file behind."""
    from contextlib import ExitStack
    from types import SimpleNamespace as NS
    from unittest.mock import Mock, patch
    sys.path[:0] = [os.path.join(REPO, "demo"), os.path.join(REPO, "prover")]
    import demo_maverick_full as demo
    import admission
    import proof_dump
    import wc_bridge

    (tmp_path / "weights").write_text("stub")
    (tmp_path / "admission.json").write_text("{}")
    (tmp_path / "proof.part").touch()
    root = b"w" * 32
    dense = NS(root=root, m_w=1, opened_columns={0}, record_openings=Mock(),
               save=Mock(), opening_budget=lambda cfg: 100)
    enrollment = NS(root=b"e" * 32, identity=lambda: b"i" * 32)
    proof = NS(Q_cols=[0], wc_bridge={"root": enrollment.root,
                                      "bridge": NS(eta_idx=[3, 5])})
    tape = NS(claims=[], prove=Mock(return_value=proof))
    dump = Mock()
    out = tmp_path / "proof.json"
    argv = ["demo", "--from-gguf", str(tmp_path / "model.gguf"), "--allow-dev-config",
            "--wc-bridge", "--public-sz", "1", "--weight-commitment", str(tmp_path / "weights"),
            "--expected-weight-root", root.hex(),
            "--admission-report", str(tmp_path / "admission.json"), "--dump-proof", str(out)]
    with ExitStack() as stack:
        for obj, name, value in [
                (sys, "argv", argv),
                (demo, "Tape", Mock(return_value=tape)),
                (demo, "build_model", Mock(return_value=(None, None, {"reveal_pin": NS()}, [0]))),
                (demo.core.WeightCommitment, "load", Mock(return_value=dense)),
                (wc_bridge, "lazy_enroll_tape", Mock(return_value=enrollment)),
                (admission, "prepare", Mock(return_value=(b"{}", {}, b"s" * 32))),
                (admission, "load_report", Mock(return_value={"machine": {"gpu_name": "stub"},
                                                              "runs": 1})),
                (admission, "check", Mock()),
                (proof_dump, "reserve_output", Mock(return_value=str(tmp_path / "proof.part"))),
                (proof_dump, "dump_proof", dump)]:
            stack.enter_context(patch.object(obj, name, value))
        assert demo.main() == 0
    assert tape.prove.called
    dense.record_openings.assert_called_once_with([0])
    dense.save.assert_called_once()
    assert dump.called, "the proof was never written"
    import json
    ledger = json.loads((tmp_path / "proof.json.wc-ledger.json").read_text())
    assert ledger["eta_spent"] == [3, 5] and ledger["root"] == (b"e" * 32).hex()
