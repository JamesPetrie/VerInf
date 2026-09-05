"""CPU regression for worker openings sharing the coordinator's host sink.

The real variable chunk iterator, opening pass, and ColumnSink execute here;
only CUDA placement and encoding are replaced with deterministic CPU work.
The GPU proof-byte gates remain in test_weight_split.

    python3 prover/tests/run_tests.py test_shard_worker
"""
import pathlib
import sys
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import core
import shard_worker
from shard_plan import ShardPlan


CFG = core.LigeroConfig(ELL=4, K_DEG=8, N_LIG=32, T_QUERIES=2)
COLUMNS = [3, 11]
ROW_BASE = 101
PAD_BASE = 17


def _fixture(lengths):
    variables, inputs = [], {}
    row = ROW_BASE
    for i, length in enumerate(lengths):
        v = core.Variable(f"W{i}", length, row_start=row, persistent=True)
        variables.append(v)
        inputs[v] = torch.arange(1 + i * 10000, 1 + i * 10000 + length,
                                 dtype=torch.int64).to(torch.uint64)
        row += v.n_rows(CFG.ELL)
    sink = core.ColumnSink(row - ROW_BASE, COLUMNS, ROW_BASE)
    return variables, inputs, sink


@contextmanager
def _cpu_openings():
    tensor, zeros = torch.tensor, torch.zeros
    chunks = []

    def on_cpu(fn):
        def call(*args, **kwargs):
            if kwargs.get("device") == "cuda":
                kwargs["device"] = "cpu"
            return fn(*args, **kwargs)
        return call

    def encode(messages, cfg, *, master_seed, row_offset, need_codewords):
        assert master_seed == "enrolled" and need_codewords
        n = messages.size(0)
        chunks.append((row_offset, n))
        # Depend on both the message and translated padding row so a worker
        # starting at the wrong offset cannot pass the placement assertions.
        codes = (messages[:, :1].to(torch.int64) * 100000
                 + (row_offset + torch.arange(n)).unsqueeze(1) * 100
                 + torch.arange(cfg.N_LIG))
        return messages, codes

    with (patch.object(core, "_to_device_u64", side_effect=lambda data: data),
          patch.object(core, "encode_messages", side_effect=encode),
          patch.object(core, "_phase", side_effect=lambda name: nullcontext()),
          patch.object(torch, "tensor", side_effect=on_cpu(tensor)),
          patch.object(torch, "zeros", side_effect=on_cpu(zeros)),
          patch.object(torch, "empty", side_effect=AssertionError(
              "worker allocated another opening buffer"))):
        yield chunks


def _open(variables, inputs, sink, lo, hi):
    return shard_worker.open_run(
        variables, inputs, CFG, None, ("enrolled", PAD_BASE, ROW_BASE),
        lo, hi, COLUMNS, column_sink=sink)


def test_open_runs_write_chunks_into_one_host_buffer():
    # The worker begins inside the W block and spans two variables and two
    # encode chunks. Partial final rows exercise the real padding iterator.
    variables, inputs, sink = _fixture([5, (core._ENCODE_CHUNK_ROWS + 1) * 4 + 1, 7])
    pointers = {j: col.data_ptr() for j, col in sink.cols.items()}
    with _cpu_openings() as chunks:
        assert _open(variables, inputs, sink, 1, 3) is None
        # The sink is still incomplete; open_run must not finish it early.
        assert sink._filled == sink.n_rows - variables[0].n_rows(CFG.ELL)
        assert _open(variables, inputs, sink, 0, 1) is None
    opened = sink.finish()
    assert len(chunks) == 3
    assert max(n for _, n in chunks) == core._ENCODE_CHUNK_ROWS
    assert {j: col.data_ptr() for j, col in opened.items()} == pointers
    for j, col in opened.items():
        expected = []
        for i, v in enumerate(variables):
            for r in range(v.n_rows(CFG.ELL)):
                first = 1 + i * 10000 + r * CFG.ELL
                pad_row = PAD_BASE + v.row_start - ROW_BASE + r
                expected.append(first * 100000 + pad_row * 100 + j)
        assert col.tolist() == expected


def test_shared_sink_still_rejects_duplicate_and_missing_runs():
    variables, inputs, sink = _fixture([5, 7])
    with _cpu_openings() as chunks:
        assert _open(variables, inputs, sink, 0, 0) is None
        assert chunks == [] and sink._filled == 0
        _open(variables, inputs, sink, 0, 1)
        try:
            sink.finish()
        except AssertionError as exc:
            assert "filled 2 of 4" in str(exc)
        else:
            raise AssertionError("missing worker rows were accepted")
        _open(variables, inputs, sink, 0, 1)
    # Total row count now matches, but the second run duplicated the first.
    try:
        sink.finish()
    except AssertionError as exc:
        assert "coverage" in str(exc)
    else:
        raise AssertionError("duplicate worker rows were accepted")


class _SweepReached(Exception):
    pass


@contextmanager
def _proof_preflight():
    """Run prove_streaming through validation, stopping at its first sweep.

    Setup/allocators are replaced so this exercises the real orchestration
    on CPU, including whether validation runs before disk-spill allocation.
    """
    variables = [core.Variable(f"W{i}", CFG.ELL, persistent=True) for i in range(2)]
    state = dict(weight_vars=variables, n_w_total=2, n_wnew_total=0,
                 n_blind_total=3, n_p1_total=1, n_p2_total=0, n_p3_total=0,
                 master_seed_t=None, groups=[], n_ops=0, p1_vars=[],
                 p2_vars=[], p3_vars=[], m_p1_rows=6, m_p2_rows=6,
                 tables=[], p1_prefix=None)
    with (patch.object(torch.cuda, "reset_peak_memory_stats"),
          patch.object(torch.cuda, "current_device", return_value=2),
          patch.object(core, "PROVE_START_HOOKS", []),
          patch.object(core, "_stream_setup", return_value=state),
          patch.object(core, "_master_seed_to_cuda", return_value=None),
          patch.object(core, "_make_merkle_acc", return_value=object()),
          patch.object(core, "_WITNESS_SPILL_DISK", True),
          patch.object(core, "_disk_spill_open", return_value={}) as spill,
          patch.object(core, "_SKIP_B_CHUNK", False),
          patch.object(core, "_stream_sweep", side_effect=_SweepReached) as sweep):
        yield spill, sweep


def _prove_with_plan(plan):
    wc = SimpleNamespace(m_w=2, n_lig=CFG.N_LIG, master_seed=core.MASTER_SEED)
    return core.prove_streaming(SimpleNamespace(claims=[]), CFG,
                                weight_commitment=wc, shard_plan=plan,
                                claims_bytes=b"[]", zk_seed=b"\x22" * 32)


def test_foreign_fold_and_open_only_devices_fail_before_sweeps():
    active, idle = [(0, 1), (1, 2)], [(0, 2), (2, 2)]
    for fold, open_ in ((active, idle), (idle, active)):
        for device in ("cuda:7", "cpu"):
            plan = ShardPlan.from_pairs(fold, open_, devices={1: device})
            with _proof_preflight() as (spill, sweep):
                try:
                    _prove_with_plan(plan)
                except NotImplementedError as exc:
                    assert "milestone M1b" in str(exc)
                else:
                    raise AssertionError("unsupported worker device was accepted")
                sweep.assert_not_called()
                spill.assert_not_called()
                assert core._SKIP_B_CHUNK is False


def test_current_device_and_empty_worker_plans_reach_first_sweep():
    active, idle = [(0, 1), (1, 2)], [(0, 2), (2, 2)]
    plans = [ShardPlan.from_pairs(active, devices={1: device})
             for device in (None, "cuda", "cuda:2")]
    plans += [ShardPlan.from_pairs(idle, active, devices={1: "cuda:2"}),
              ShardPlan.from_pairs(active, idle, devices={1: "cuda:2"}),
              ShardPlan.from_pairs(idle, devices={1: "cuda:7"}),
              ShardPlan.from_pairs([(0, 2)]), None]
    for plan in plans:
        with _proof_preflight() as (spill, sweep):
            try:
                _prove_with_plan(plan)
            except _SweepReached:
                pass
            else:
                raise AssertionError("valid plan did not reach its first sweep")
            spill.assert_called_once()
            sweep.assert_called_once()
