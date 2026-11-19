"""CPU regression tests for the Llama driver's enrollment and ledger."""
import json
import os
import stat
import subprocess
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


@pytest.mark.parametrize("queries", [0, 5, 8, 9])
def test_ineligible_query_count_refused_before_enrolled_proof(monkeypatch, queries):
    cfg = core.LigeroConfig(ELL=8, K_DEG=16, N_LIG=64, T_QUERIES=queries)
    monkeypatch.setattr(driver, "CFG", cfg)
    with pytest.raises(ValueError, match="padding budget"):
        driver.main(from_hf="unused", weight_commitment="unused.wc")
    with pytest.raises(ValueError, match="padding budget"):
        core.prove_streaming(None, cfg, weight_commitment=_handle())


def test_opening_budget_counts_the_union_at_the_limit():
    cfg = core.LigeroConfig(ELL=8, K_DEG=16, N_LIG=64, T_QUERIES=4)
    wc = _handle()
    wc.check_query_budget(cfg)
    wc.record_openings([0, 1, 2, 3])
    wc.check_openings(cfg, [0, 1, 2, 3])
    with pytest.raises(AssertionError, match="opening budget exhausted"):
        wc.check_openings(cfg, [0, 1, 2, 4])


@pytest.mark.parametrize("flag", ["enroll_out", "weight_commitment"])
def test_enrollment_requires_reloadable_weights(tmp_path, flag):
    with pytest.raises(ValueError, match="require --from-hf"):
        driver.main(**{flag: tmp_path / "model.wc"})
    assert list(tmp_path.iterdir()) == []


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
    def weight_rows(ids, **options):
        with pytest.raises(TailReached) as exc:
            driver.main(from_hf="fixture", num_layers=0, token_ids=ids,
                        persistent_head=True, engine=True, **options)
        used_ids, rows = exc.value.args
        assert used_ids == ([0, 0] if ids is None else ids)
        assert [name for name, _, _ in rows] == ["E_embedding", "final_norm_w"]
        return rows

    rows = weight_rows(None)
    assert rows == weight_rows([1, 2])
    assert rows == weight_rows([1, 2], unexplained_info=True, public_input_tokens=True)


@pytest.mark.parametrize("output", ["statement_out", "dump_proof"])
def test_tied_scoring_requires_public_input_consent(tmp_path, monkeypatch, capsys, output):
    cfg = replace(driver.RANDOM_WEIGHTS_CFG, tied_embeddings=True)
    monkeypatch.setattr(driver.ModelConfig, "from_hf", lambda _: cfg)
    monkeypatch.setattr(driver, "SEQ", 2)
    with pytest.raises(ValueError, match="tied embeddings publish every input token ID"):
        driver.main(from_hf="fixture", token_ids=[12345, 9876], unexplained_info=True,
                    **{output: tmp_path / "result"})
    assert "9876" not in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == []


def test_tied_transcript_file_refused_before_reading_weights(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(dict(
        hidden_size=8, intermediate_size=16, num_hidden_layers=1,
        num_attention_heads=2, vocab_size=10000, tie_word_embeddings=True)))
    transcript = tmp_path / "tokens.json"
    transcript.write_text(json.dumps({"prompt": [2, 3], "continuation": [9876]}))
    statement = tmp_path / "claims"
    run = subprocess.run([sys.executable, str(ROOT / "demo" / "demo_llama7b.py"),
                          "--from-hf", str(tmp_path), "--tokens-file", str(transcript),
                          "--unexplained-info", "--statement-out", str(statement)],
                         capture_output=True, text=True, timeout=60)
    assert run.returncode and "--public-input-tokens" in run.stderr
    assert "9876" not in run.stdout + run.stderr
    assert not statement.exists() and not statement.with_suffix(".json").exists()


def _handle():
    leaf = bytes(32)
    return core.WeightCommitment(root=leaf, levels=[[leaf]], m_w=1, n_lig=1)


@pytest.mark.parametrize("output", ["statement_out", "dump_proof"])
def test_requested_bound_is_required_for_published_outputs(tmp_path, monkeypatch, output):
    cfg = replace(driver.RANDOM_WEIGHTS_CFG, d=2, vocab=2)
    monkeypatch.setattr(driver.ModelConfig, "from_hf", lambda _: cfg)
    monkeypatch.setattr(driver, "SEQ", 2)
    monkeypatch.setattr(loader, "load_token_embedding", lambda *a, **kw: torch.ones(4))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(loader.LazyHFLoader, "__init__", lambda *a, **kw: None)
    monkeypatch.setattr(loader.LazyHFLoader, "load_final_norm", lambda _: torch.ones(2))
    monkeypatch.setattr(loader.LazyHFLoader, "load_final_weights",
                        lambda _: {"W_lm_head": torch.ones(4)})
    data = torch.tensor([0, 1 << 26, 0, 1], dtype=torch.uint64)
    monkeypatch.setattr(driver, "_run_tail",
                        lambda tape, *a, **kw: tape.commit("logits", data, (2, 2)))
    with pytest.raises(ValueError, match="too large for the exp table"):
        driver.main(from_hf="fixture", num_layers=0, unexplained_info=True,
                    engine=True, **{output: str(tmp_path / "result")})
    assert list(tmp_path.iterdir()) == []


@pytest.fixture
def tiny_scoring_model(monkeypatch):
    """Checkpoint fixture with an identity tail and a witness-dependent reveal on CPU."""
    import compute_fns
    import unexplained_info

    cfg = replace(driver.RANDOM_WEIGHTS_CFG, d=2, vocab=2)
    monkeypatch.setattr(driver.ModelConfig, "from_hf", lambda _: cfg)
    monkeypatch.setattr(driver, "SEQ", 2)
    monkeypatch.setattr(loader, "load_token_embedding",
                        lambda *a, **kw: torch.tensor([3, 1, 0, 4], dtype=torch.uint64))
    monkeypatch.setattr(loader.LazyHFLoader, "__init__", lambda *a, **kw: None)
    monkeypatch.setattr(loader.LazyHFLoader, "load_final_norm", lambda _: torch.ones(2))
    monkeypatch.setattr(loader.LazyHFLoader, "load_final_weights",
                        lambda _: {"W_lm_head": torch.eye(2)})
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(driver, "_run_tail", lambda tape, resid, *a, **kw: resid)
    monkeypatch.setattr(compute_fns, "gl_add", torch.add)

    def score(tape, logits, tokens, **kw):
        rows = logits.data.view(2, 2).to(torch.int64)
        gaps = rows.max(1).values - rows[range(2), tokens]
        parts = [tape.commit(f"gap_{i}", v, (1,)) for i, v in enumerate(gaps)]
        sz = tape.add(*parts)
        return sz, {"reveal_pin": tape.reveal(sz)}

    monkeypatch.setattr(unexplained_info, "prove_unexplained_info", score)
    return dict(from_hf="fixture", num_layers=0, engine=True, unexplained_info=True,
                ui_output_tokens=[1, 1], ui_gap_max=8)


def test_promptless_scoring_preview_matches_prover_input(tmp_path, monkeypatch, tiny_scoring_model):
    statement = tmp_path / "claims"
    driver.main(**tiny_scoring_model, statement_out=statement)
    assert json.loads((tmp_path / "claims.json").read_bytes())["sz"] == 4

    class ProverReached(Exception):
        pass

    def check_claims(tape, *, claims_bytes, **kw):
        assert claims_bytes == statement.read_bytes()
        raise ProverReached

    monkeypatch.setattr(driver.Tape, "prove", check_claims)
    with pytest.raises(ProverReached):
        driver.main(**tiny_scoring_model, dump_proof=tmp_path / "proof")


@pytest.mark.parametrize("gap_max", [8, None])
def test_forward_engine_reports_revealed_score(tiny_scoring_model, capsys, gap_max):
    options = dict(tiny_scoring_model, ui_gap_max=gap_max)
    logits = driver.main(**options, forward_only=True)
    assert logits.data.tolist() == [3, 1, 3, 1]
    assert "U = 0.0014 bits over 2 scored output token(s)" in capsys.readouterr().out


@pytest.mark.parametrize("tokens, expected_sz", [(None, 0), ([0, 1], 2)])
def test_default_greedy_and_explicit_scoring(tiny_scoring_model, tokens, expected_sz, tmp_path):
    options = dict(tiny_scoring_model, ui_output_tokens=tokens)
    driver.main(**options, statement_out=tmp_path / "claims")
    assert json.loads((tmp_path / "claims.json").read_bytes())["sz"] == expected_sz


def test_random_weights_cannot_preview_witness_dependent_bound(tmp_path):
    with pytest.raises(ValueError, match="preview requires --from-hf"):
        driver.main(unexplained_info=True, statement_out=tmp_path / "claims")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("persistent, rows, columns, error", [
    (False, 1, 64, "without persistent"),
    (True, 2, 64, "m_w 2 vs 1"),
    (True, 1, 32, "N_LIG 32 vs 64"),
    (True, 1, 64, None),
])
def test_preview_checks_enrollment_geometry(tmp_path, persistent, rows, columns, error):
    cfg = core.LigeroConfig(ELL=8, K_DEG=16, N_LIG=64, T_QUERIES=4)
    tape = driver.Tape(cfg, lazy=True)
    w = tape.commit("w", torch.ones(8), (8,), persistent=persistent)
    x = tape.commit("x", torch.ones(8), (8,))
    tape.add(w, x)
    wc = replace(_handle(), m_w=rows, n_lig=columns)
    path = str(tmp_path / "claims")
    if error:
        with pytest.raises((ValueError, AssertionError), match=error):
            write_statement(tape, cfg, path, weight_commitment=wc)
        assert list(tmp_path.iterdir()) == []
    else:
        meta = write_statement(tape, cfg, path, weight_commitment=wc)
        assert "w" in meta["blocks"] and meta["weight_root"] == wc.root.hex()


def test_statement_sidecar_contains_only_public_ui_fields(tmp_path):
    tape = driver.Tape(driver.CFG, lazy=True)
    public = dict(s_b=16, s_c=256, s_y=256, gap_max=128, n=2)
    info = dict(public, observed_gap=60, tokens=[1, 2], reveal_pin=object())
    write_statement(tape, tape.cfg, str(tmp_path / "claims"), ui_info=info, sz=7)
    sidecar = json.loads((tmp_path / "claims.json").read_text())
    assert sidecar["ui"] == public
    assert sidecar["sz"] == 7


def test_failed_preview_publication_can_be_retried(tmp_path, monkeypatch):
    tape = driver.Tape(driver.CFG, lazy=True)
    path = tmp_path / "statement"
    side = tmp_path / "statement.json"
    link = os.link

    def fail_claims(source, dest):
        if dest == path:
            assert side.is_file() and not path.exists()
            raise OSError("cannot publish claims")
        link(source, dest)

    with monkeypatch.context() as m:
        m.setattr(os, "link", fail_claims)
        with pytest.raises(OSError, match="cannot publish claims"):
            write_statement(tape, tape.cfg, path)
    assert list(tmp_path.iterdir()) == []
    side.write_bytes(b"someone else's sidecar")
    with pytest.raises(FileExistsError):
        write_statement(tape, tape.cfg, path)
    assert side.read_bytes() == b"someone else's sidecar"
    side.unlink()
    write_statement(tape, tape.cfg, path)
    assert path.is_file() and side.is_file()


def test_interrupted_preview_recovers_its_own_sidecar(tmp_path):
    path = tmp_path / "statement"
    child = """
import os, sys
from pathlib import Path
sys.path[:0] = sys.argv[2:]
from statement_preview import _publish_statement
link = os.link
def stop_after_link(source, dest):
    link(source, dest)
    os._exit(7)
os.link = stop_after_link
_publish_statement(Path(sys.argv[1]), b"old claims", b"old sidecar")
"""
    result = subprocess.run([sys.executable, "-c", child, str(path),
                             str(ROOT / "demo"), str(ROOT / "prover")], timeout=60)
    assert result.returncode == 7
    side = tmp_path / "statement.json"
    assert side.read_bytes() == b"old sidecar" and not path.exists()
    tape = driver.Tape(driver.CFG, lazy=True)
    meta = write_statement(tape, tape.cfg, path)
    assert json.loads(side.read_bytes()) == meta
    assert set(tmp_path.iterdir()) == {path, side}


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


def test_hard_linked_enrollment_refused_at_load_and_update(tmp_path):
    path = tmp_path / "model.wc"
    alias = tmp_path / "alias.wc"
    wc = _handle()
    wc.save(path)
    os.link(path, alias)
    for name in (path, alias):
        with pytest.raises(ValueError, match="hard-linked weight commitments"):
            with driver._locked_enrollment(name):
                pytest.fail("accepted an ambiguous enrollment")
    wc.record_openings([0])
    with pytest.raises(ValueError, match="hard-linked weight commitments"):
        wc.save(path)
    assert path.samefile(alias)
    alias.unlink()
    assert core.WeightCommitment.load(path).opened_columns == []
    wc.save(path)
    assert core.WeightCommitment.load(path).opened_columns == [0]


def test_handle_save_through_symlink_updates_the_target(tmp_path):
    path = tmp_path / "model.wc"
    _handle().save(path)
    alias = tmp_path / "current.wc"
    alias.symlink_to(path.name)

    wc = core.WeightCommitment.load(alias)
    wc.record_openings([0])
    wc.save(alias)

    assert alias.is_symlink() and alias.samefile(path)
    assert core.WeightCommitment.load(path).opened_columns == [0]


def test_proof_through_symlink_updates_the_locked_target(tmp_path, monkeypatch):
    path = tmp_path / "model.wc"
    _handle().save(path)
    alias = tmp_path / "current.wc"
    alias.symlink_to(path.name)
    monkeypatch.setattr(driver.ModelConfig, "from_hf", lambda _: driver.RANDOM_WEIGHTS_CFG)
    monkeypatch.setattr(driver, "_rand_signed", lambda n, **kw: torch.zeros(n))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    def prove(tape, **kwargs):
        assert tape.lazy
        with pytest.raises(BlockingIOError):
            with driver._locked_enrollment(path):
                pytest.fail("the canonical enrollment lock must be held")
        from types import SimpleNamespace
        return SimpleNamespace(Q_cols=[0])

    monkeypatch.setattr(driver.Tape, "prove", prove)
    driver.main(from_hf="fixture", num_layers=0, with_lm_head=False,
                weight_commitment=alias)
    assert alias.is_symlink()
    assert core.WeightCommitment.load(path).opened_columns == [0]


@pytest.mark.parametrize("suffix", ["", ".part"])
def test_output_collision_precedes_proving_and_ledger_update(tmp_path, monkeypatch, suffix):
    handle = tmp_path / "weights.wc"
    _handle().save(handle)
    target = tmp_path / ("proof.json" + suffix)
    target.write_bytes(b"original")
    monkeypatch.setattr(driver.ModelConfig, "from_hf", lambda _: driver.RANDOM_WEIGHTS_CFG)
    monkeypatch.setattr(driver, "_rand_signed", lambda n, **kw: torch.zeros(n))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(driver.Tape, "prove", lambda *a, **kw: pytest.fail("started proof"))
    monkeypatch.setenv("LIGERO_DUMP_PROOF", str(tmp_path / "proof.json"))
    with pytest.raises(FileExistsError):
        driver.main(from_hf="fixture", num_layers=0, with_lm_head=False, weight_commitment=handle)
    assert target.read_bytes() == b"original"
    assert core.WeightCommitment.load(handle).opened_columns == []


def test_failed_proof_releases_its_reservation(tmp_path):
    from proof_dump import reserved_dense_output
    tape = driver.Tape(driver.CFG, lazy=True)
    path = tmp_path / "proof.json"
    with pytest.raises(RuntimeError, match="prover stopped"):
        with reserved_dense_output(path, tape) as (part, claims):
            assert Path(part).stat().st_size > len(claims) and not path.exists()
            with pytest.raises(FileExistsError):
                with reserved_dense_output(path, tape):
                    pytest.fail("another writer reserved this path")
            raise RuntimeError("prover stopped")
    assert list(tmp_path.iterdir()) == []


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


def test_initial_enrollment_preserves_existing_handle(tmp_path):
    path = tmp_path / "model.wc"
    first = _handle()
    first.record_openings([0])
    first.save(path, overwrite=False)
    original = path.read_bytes()

    with pytest.raises(FileExistsError, match="refusing to overwrite enrollment"):
        driver.main(from_hf="unused", enroll_out=path)
    # The atomic publish must also refuse a file created after the preflight.
    with pytest.raises(FileExistsError):
        _handle().save(path, overwrite=False)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]
