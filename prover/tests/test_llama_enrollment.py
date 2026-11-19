"""CPU regression tests for the Llama driver's enrollment and ledger."""
import json
import os
import stat
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "demo"), str(ROOT / "prover")]

import core
import demo_llama7b as driver
import loader
from statement_preview import write_statement


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


def test_tied_weight_rows_match_with_and_without_prompt(monkeypatch):
    cfg = replace(driver.RANDOM_WEIGHTS_CFG, d=8, vocab=16, tied_embeddings=True)
    embedding = torch.arange(cfg.vocab * cfg.d).to(torch.uint64)
    monkeypatch.setattr(driver.ModelConfig, "from_hf", lambda _: cfg)
    monkeypatch.setattr(driver, "SEQ", 2)
    monkeypatch.setattr(loader, "load_token_embedding", lambda *a, **kw: embedding)
    monkeypatch.setattr(loader.LazyHFLoader, "__init__", lambda *a, **kw: None)
    monkeypatch.setattr(loader.LazyHFLoader, "load_final_norm", lambda _: torch.ones(cfg.d))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    class TailReached(Exception):
        pass

    def inspect_tail(tape, resid, norm, head, **kwargs):
        assert kwargs["lm_transpose_b"] and head.shape == (cfg.vocab, cfg.d)
        lookup, = tape.claims
        assert lookup.E is head.var
        tape.embed(norm, [0], cfg.d)
        tape.matmul(resid, head, transpose_b=True)
        layout = core._layout(tape.claims, tape.cfg)
        rows = [(w.name, w.row_start, tape.inputs[w].tolist()) for w in layout[7]]
        raise TailReached(lookup.token_ids, rows)

    monkeypatch.setattr(driver, "_run_tail", inspect_tail)
    def weight_rows(ids):
        with pytest.raises(TailReached) as exc:
            driver.main(from_hf="fixture", num_layers=0, token_ids=ids,
                        persistent_head=True, engine=True)
        used_ids, rows = exc.value.args
        assert used_ids == ([0, 0] if ids is None else ids)
        assert [name for name, _, _ in rows] == ["E_embedding", "final_norm_w"]
        return rows

    assert weight_rows(None) == weight_rows([1, 2])


def _handle():
    leaf = bytes(32)
    return core.WeightCommitment(root=leaf, levels=[[leaf]], m_w=1, n_lig=1)


def test_statement_sidecar_contains_only_public_ui_fields(tmp_path):
    tape = driver.Tape(driver.CFG, lazy=True)
    public = dict(s_b=16, s_c=256, s_y=256, gap_max=128, n=2)
    info = dict(public, observed_gap=60, tokens=[1, 2], reveal_pin=object())
    write_statement(tape, tape.cfg, str(tmp_path / "claims"), ui_info=info, sz=7)
    sidecar = json.loads((tmp_path / "claims.json").read_text())
    assert sidecar["ui"] == public
    assert sidecar["sz"] == 7


def test_lock_survives_atomic_save_and_releases_on_error(tmp_path):
    path = str(tmp_path / "model.wc")
    _handle().save(path)
    with pytest.raises(BlockingIOError, match="already in use"):
        with driver._locked_enrollment(path) as wc:
            wc.record_openings([0])
            wc.save(path)
            with driver._locked_enrollment(path):
                pytest.fail("another prover acquired the enrollment lock")
    with driver._locked_enrollment(path) as wc:
        assert wc.opened_columns == [0]


def test_handle_and_temporary_file_are_private_on_every_save(tmp_path, monkeypatch):
    path = tmp_path / "model.wc"
    wc = _handle()
    fsync = os.fsync
    modes = []

    def inspect_fsync(fd):
        mode = os.fstat(fd).st_mode
        if stat.S_ISREG(mode):
            modes.append(stat.S_IMODE(mode))
        fsync(fd)

    monkeypatch.setattr(os, "fsync", inspect_fsync)
    old_mask = os.umask(0o022)
    try:
        for _ in range(2):
            wc.save(path)
            assert stat.S_IMODE(path.stat().st_mode) == 0o600
    finally:
        os.umask(old_mask)
    assert modes == [0o600, 0o600]
    assert list(tmp_path.iterdir()) == [path]
