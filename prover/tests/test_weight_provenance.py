"""CPU checks of GGUF source aliases and linking metadata through a real Tape.

Synthetic GGUF headers feed the production demo's lazy loader adapters;
no loader resolution, CUDA work, or full model checkpoint is needed.

    python3 prover/tests/run_tests.py test_weight_provenance
"""
import pathlib
import sys
import tempfile
from unittest.mock import patch

_ROOT = pathlib.Path(__file__).resolve().parents[2]
for path in (_ROOT / "prover", _ROOT / "profiler", _ROOT / "demo"):
    sys.path.insert(0, str(path))

import numpy as np
from gguf import GGUFWriter
import core
from tape import Tape
from demo_maverick_full import _field_loader
from extract import extract_tape
from manifest import Manifest
from shard_plan import ShardPlan
import weightsplit
from machine import MachineProfile


CFG = core.LigeroConfig(ELL=8, K_DEG=16, N_LIG=64, T_QUERIES=4)


def _write_weights(path):
    writer = GGUFWriter(str(path), "llama4")
    values = np.arange(64, dtype=np.float16).reshape(8, 8)
    writer.add_tensor("token_embd.weight", values)
    # Equal values under a different name are not an alias: these are two
    # independent stored sources, regardless of their current contents.
    writer.add_tensor("output.weight", values.copy())
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def test_tied_source_and_linking_metadata_survive_real_tape_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        path = pathlib.Path(td) / "weights.gguf"
        _write_weights(path)
        with patch("gguf.quants.dequantize", side_effect=AssertionError(
                "provenance extraction decoded a weight")):
            embedding = _field_loader(str(path), "token_embd.weight")
            head = _field_loader(str(path), "token_embd.weight", transpose=True)
            assert embedding.provenance == head.provenance
            assert head.provenance["packed_bytes"] == 128
            assert head.provenance["packed_source"]
            tape = Tape(CFG, lazy=True)
            a = tape.commit_lazy("embedding", embedding, (8, 8), 64)
            b = tape.commit_lazy("head", head, (8, 8), 64)
            new = tape.commit_lazy("Wnew", head, (8, 8), 64, persistent="new")
            tape.concat([a, b, new], (192,))
            man = extract_tape(tape, model=dict(name="tied-link"), seq=1)
            enrolled = core._layout(tape.claims, CFG)[7]
            assert [v.name for v in enrolled] == ["embedding", "head"]
            for suffix in ("json", "json.gz"):
                manifest_path = pathlib.Path(td) / f"manifest.{suffix}"
                man.save(manifest_path)
                loaded = Manifest.load(manifest_path)
                assert loaded == man
                assert loaded.var_by_name()["Wnew"].w_new
                blk = weightsplit._Block(loaded, None, CFG.ELL)
                assert blk.n == len(enrolled) == 2
                assert blk.bytes(0, 2) == blk.bytes(0, 1) == blk.bytes(1, 2) == 128
                assert blk.stream_bytes(0, 2) == 256
                ev = weightsplit.evaluate(loaded, MachineProfile.load("gb10-spark"),
                                          2, resident=True, x_fold=0.5, x_open=0.5)
                assert ev["hold_bytes"] == [128, 128]
                ShardPlan.from_pairs(ev["plan_fold"], ev["plan_open"]).validated(len(enrolled))


def test_source_identity_distinguishes_tensor_names_and_files():
    with tempfile.TemporaryDirectory() as td:
        paths = [pathlib.Path(td) / f"weights{i}.gguf" for i in range(2)]
        for path in paths:
            _write_weights(path)
        loaders = [_field_loader(str(path), name)
                   for path in paths for name in ("token_embd.weight", "output.weight")]
        assert len({ld.provenance["packed_source"] for ld in loaders}) == 4
        assert all(ld.provenance["packed_bytes"] == 128 for ld in loaders)
