"""CPU regression tests for the Llama driver's enrollment and ledger."""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "demo"), str(ROOT / "prover")]

import core
import demo_llama7b as driver
import loader


@pytest.mark.parametrize("options, persistent", [
    ({}, False),
    ({"enroll_out": "model.wc"}, True),
    ({"weight_commitment": "model.wc"}, True),
    ({"persistent_head": True}, True),
])
def test_eager_layer_weights_in_driver(monkeypatch, options, persistent):
    weights = {name: torch.ones(8) for name in ("W_Q", "rms_pre_attn_w")}
    monkeypatch.setattr(loader, "load_layer_weights", lambda *a, **kw: weights)
    monkeypatch.setattr(driver.ModelConfig, "from_hf", lambda _: driver.RANDOM_WEIGHTS_CFG)
    monkeypatch.setattr(driver, "_rand_signed", lambda n, **kw: torch.zeros(n))

    class BlockReached(Exception):
        pass

    def inspect_block(tape, resid, layer_weights, **kwargs):
        assert set(layer_weights) == set(weights)
        assert all(w.var.persistent == persistent for w in layer_weights.values())
        raise BlockReached

    monkeypatch.setattr(driver, "_run_block", inspect_block)
    with pytest.raises(BlockReached):
        driver.main(from_hf="fixture", engine=True, **options)


def test_empty_enrollment_is_refused(monkeypatch):
    monkeypatch.setattr(core, "commit_weights", lambda *a: (None, [], 0))
    with pytest.raises(ValueError, match="no persistent weight rows"):
        core.WeightCommitment.from_tape(None, driver.CFG)
