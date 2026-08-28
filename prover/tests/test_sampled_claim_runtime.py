"""The real Tape -> 5-of-49 runtime adapter uses one engine pass."""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import admission
import core
import compute_fns
import pytest
import torch
from cuda_primitives import hash_columns_streamed
from routing_claim import freivalds_combine
from sampled_claim_runtime import (ClaimWindowAudit, _finish_tensor_digest,
                                   _tensor_digest_parts)
from tape import Tape

CFG = core.LigeroConfig(ELL=8, K_DEG=8, N_LIG=32, T_QUERIES=4)


def _case():
    tape = Tape(CFG, lazy=True)
    a = tape.commit(
        "a", torch.arange(5, dtype=torch.int64, device="cuda").to(torch.uint64),
        (5,))
    b = tape.commit(
        "b", (10 + torch.arange(3, dtype=torch.int64, device="cuda")).to(
            torch.uint64), (3,))
    dst = tape.concat([a, b], (8,))
    admission.prepare(tape, CFG)
    audit = ClaimWindowAudit(
        tape, CFG, b"v" * 32, b"public" * 4, b"model" * 6 + b"xx",
        expected_claims=0, window_size=1, sample_per_window=1)
    return tape, dst, audit


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime test")
def test_real_tape_observer_accepts_without_replay():
    tape, dst, audit = _case()
    live = tape.run_engine_pass(
        free_intermediates=True, keep={dst.var}, observer=audit)
    result = audit.finish()

    assert live[dst.var].to(torch.int64).cpu().tolist() == [0, 1, 2, 3, 4, 10, 11, 12]
    assert result["accepted"] is True
    assert result["claims"] == result["selected"] == 1
    assert len(result["c0_root"]) == 64
    assert result["local_argument"] == "exact-recomputation"
    assert result["cryptographic_local_proofs"] is False
    assert result["rs_openings_materialized"] is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime test")
def test_c0_is_independent_of_window_batching():
    tape, dst, per_claim = _case()
    tape.run_engine_pass(free_intermediates=True, keep={dst.var},
                         observer=per_claim)
    first = per_claim.finish()

    one_window = ClaimWindowAudit(
        tape, CFG, b"v" * 32, b"public" * 4, b"model" * 6 + b"xx",
        expected_claims=0, window_size=len(tape.claims),
        sample_per_window=len(tape.claims))
    tape.run_engine_pass(free_intermediates=True, keep={dst.var},
                         observer=one_window)
    second = one_window.finish()

    assert first["c0_root"] == second["c0_root"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime test")
def test_real_tape_observer_rejects_tampered_selected_output():
    tape, dst, audit = _case()

    def tamper_then_observe(index, claim, input_vars, input_data, outs, live):
        bad = outs[dst.var].clone()
        bad[0] = 99
        outs[dst.var] = bad
        live[dst.var] = bad
        audit(index, claim, input_vars, input_data, outs, live)

    tape.run_engine_pass(observer=tamper_then_observe)
    result = audit.finish()

    assert result["accepted"] is False
    assert len(result["failures"]) == 1
    assert result["failures"][0].startswith("claim 0: cat_a_b#")
    assert result["failures"][0].endswith(": local operation mismatch")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime test")
def test_foldable_boundary_wires_are_retained_without_disabling_fold(
        tmp_path, monkeypatch):
    monkeypatch.delenv("LIGERO_NO_FOLD", raising=False)
    tape = Tape(CFG, lazy=True)
    mask = tape.commit(
        "mask", torch.tensor([1, 0, 0, 0, 0, 1, 0, 0],
                             dtype=torch.uint64, device="cuda"), (2, 4))
    xs = []
    for expert in range(4):
        a = tape.commit(
            f"x{expert}a", torch.full((3,), 10 + expert,
                                      dtype=torch.uint64, device="cuda"), (3,))
        b = tape.commit(
            f"x{expert}b", torch.full((3,), 20 + expert,
                                      dtype=torch.uint64, device="cuda"), (3,))
        xs.append(tape.concat([a, b], (6,)))
    out = freivalds_combine(tape, mask, xs, T=2, E=4, F=3)
    admission.prepare(tape, CFG)
    assert compute_fns.FoldRunner(tape._deferred).is_fold(tape.claims[-1])
    progress = tmp_path / "progress.jsonl"
    audit = ClaimWindowAudit(
        tape, CFG, b"v" * 32, b"public" * 4, b"model" * 6 + b"xx",
        expected_claims=0, window_size=len(tape.claims),
        sample_per_window=len(tape.claims), progress_path=str(progress),
        heartbeat_every=1000)

    live = tape.run_engine_pass(
        free_intermediates=True, keep={out.var}, observer=audit)
    result = audit.finish()
    events = [json.loads(line) for line in progress.read_text().splitlines()]

    assert result["accepted"] is True
    assert result["fold_retained_peak_bytes"] > 0
    assert audit.fold_values == {}
    assert live[out.var].numel() == 6
    assert any(e["event"] == "window_complete" for e in events)
    assert events[-1]["event"] == "audit_complete"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime test")
def test_striped_wire_digest_matches_portable_cpu_and_detects_tamper():
    value = torch.arange(10003, dtype=torch.int64, device="cuda").to(torch.uint64)

    def finish(tensor):
        n, columns, payload = _tensor_digest_parts(tensor)
        if isinstance(payload, torch.Tensor):
            payload = hash_columns_streamed(payload).cpu().numpy().tobytes()
        return _finish_tensor_digest(n, columns, payload)

    gpu_digest = finish(value)
    assert gpu_digest == finish(value.cpu())
    tampered = value.clone()
    tampered.view(torch.int64)[917] += 1
    assert finish(tampered) != gpu_digest
