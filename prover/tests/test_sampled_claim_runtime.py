"""The real Tape -> 5-of-49 runtime adapter uses one engine pass."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import admission
import core
import pytest
import torch
from sampled_claim_runtime import ClaimWindowAudit
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
