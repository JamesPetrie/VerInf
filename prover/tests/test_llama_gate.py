"""Exercise the quickstart's pre-proof policy and calibration checks on CPU."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "demo"), str(ROOT / "prover")]


@pytest.mark.parametrize("sz, verdict, expected", [
    (4, "ACCEPT", 0), (4, "REJECT", 1), (100, "ACCEPT", 1),
])
def test_quickstart_verifies_preproof_policy(tmp_path, monkeypatch, sz, verdict, expected):
    import core
    import demo_llama7b
    for name in ("LIGERO_T_QUERIES", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.setenv(name, "1")
    import gate

    monkeypatch.setattr(gate, "_check_model", lambda: None)
    binary = ROOT / "verifier" / "target" / "release" / "verify_proof"
    monkeypatch.setattr(gate, "_verifier_bin", lambda: binary)
    monkeypatch.setattr(gate.tempfile, "tempdir", str(tmp_path))
    root = bytes(range(32))
    digest = "17" * 32
    phases = []

    def driver(**options):
        assert options["persistent_head"] and options["token_ids"] == gate.TOKENS
        if "enroll_out" in options:
            phases.append("enroll")
            wc = core.WeightCommitment(root=root, levels=[[root]], m_w=1, n_lig=1)
            wc.save(options["enroll_out"], overwrite=False)
            return
        assert options["weight_commitment"].is_file() and options["unexplained_info"]
        if "statement_out" in options:
            phases.append("preview")
            policy = dict(statement_digest=digest, sz=sz, ui=dict(n=len(gate.TOKENS), s_b=1))
            options["statement_out"].with_suffix(".json").write_text(json.dumps(policy))
            return
        phases.append("prove")
        options["dump_proof"].write_text("opaque proof")
        # Policy must already have been read before calling the prover.
        options["dump_proof"].with_name("claims.json").unlink()

    def verify(argv, **kwargs):
        phases.append("verify")
        executable, proof, expected_root, expected_digest = argv
        assert executable == str(binary) and Path(proof).read_text() == "opaque proof"
        assert (expected_root, expected_digest) == (root.hex(), digest)
        return subprocess.CompletedProcess(argv, 0, stdout=f"rust_verify: {verdict}", stderr="")

    monkeypatch.setattr(demo_llama7b, "main", driver)
    monkeypatch.setattr(gate.subprocess, "run", verify)
    assert gate.main() == expected
    assert phases == ["enroll", "preview", "prove", "verify"]
    assert list(tmp_path.iterdir()) == []
