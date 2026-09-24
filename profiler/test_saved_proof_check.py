"""The S=1000 proof handoff: same-run policy, Rust verdict, durable receipt."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from profiler.instrumented_prove import write_dump_policy
from tools.check_dumped_proof import check
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "prover"))
from proof_dump import estimated_bytes


def _fixture(tmp_path, verdict="ACCEPT"):
    proof = tmp_path / "mavp-s1000.bin"
    proof.write_bytes(b"compact proof fixture")
    obj = SimpleNamespace(
        root_w=b"w" * 32,
        statement_digest=b"s" * 32,
        wc_bridge={"identity": b"i" * 32},
    )
    policy_path = Path(write_dump_policy(str(proof), obj))
    revision = tmp_path / "session6-rev"
    revision.write_text("e488910\n")
    verifier = tmp_path / "verify_proof"
    verifier.write_text(
        "#!/bin/sh\n"
        f"[ \"$#\" -eq 4 ] && [ \"$2\" = '{'77' * 32}' ] && "
        f"[ \"$3\" = '{'73' * 32}' ] && "
        f"[ \"$4\" = '{'69' * 32}' ] || exit 19\n"
        f"echo 'rust_verify: {verdict}'\n"
    )
    verifier.chmod(0o700)
    return proof, policy_path, revision, verifier


def test_accept_writes_receipt_for_the_saved_proof(tmp_path):
    proof, policy_path, revision, verifier = _fixture(tmp_path)
    assert json.loads(policy_path.read_text())["policy_source"] == \
        "same-run prover (mechanism check only)"
    receipt = check(proof, verifier, revision)
    assert receipt["verdict"] == "ACCEPT"
    assert receipt["proof_bytes"] == proof.stat().st_size
    assert receipt["proof_sha256"] == hashlib.sha256(proof.read_bytes()).hexdigest()
    assert receipt["revision"] == "e488910"
    assert json.loads(Path(str(proof) + ".verify.json").read_text()) == receipt


def test_reject_does_not_create_a_receipt(tmp_path):
    proof, _policy_path, revision, verifier = _fixture(tmp_path, "REJECT")
    with pytest.raises(RuntimeError, match="did not ACCEPT"):
        check(proof, verifier, revision)
    assert not Path(str(proof) + ".verify.json").exists()


def test_dump_estimate_includes_bridge_openings():
    one = torch.zeros(1, dtype=torch.uint64)
    proof = SimpleNamespace(
        blocks=["p1", "p2"], opened_p1={0: one}, opened_p2={0: one},
        paths_p1={0: []}, paths_p2={0: []},
        q_irs=one, q_lin=one, p_0=one, wc_bridge=None,
    )
    without_bridge = estimated_bytes(proof, [0], u64_encoding="u64le-base64")
    opened = torch.zeros(100, dtype=torch.uint64)
    proof.wc_bridge = {"bridge": SimpleNamespace(
        p_trace={4: one}, pi={4: one}, c=[0], v=[0], eta_idx=[0],
        opened={0: opened}, paths={0: [(b"x" * 32, 0)]},
    )}
    with_bridge = estimated_bytes(proof, [0], u64_encoding="u64le-base64")
    assert with_bridge - without_bridge >= 100 * 11 + 80
