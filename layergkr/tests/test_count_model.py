"""Gates for the cost model's two lowest levels: L1 (shape) and L4 (bytes).

These live in the TEST SUITE, not in a bench script, and that is the point. L1's
exactness was validated by `bench/run_toy.py`, which was not re-run after the MoE
work replaced per-token expert matmuls with three hidden-route nodes. So the
model went on predicting `5 + 3*S` matmuls and deriving range checks from matmul
CELLS, and was wrong by 8x on lookups, 3.4x on gate slots and 2.4x on cells --
in a level the standing document called exact. A validation nobody runs is not a
validation.

L4 exists because until 2026-08-05 the model ended in seconds and nothing else,
so it could not say the one thing that stopped two runs that day: that they would
not fit. Both failures are pinned below as regression cases.

Run:  .venv/bin/python layergkr/tests/run_tests.py test_count_model
"""
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from layergkr import count_model as cm, rs, semantics as sem

SHAPES = [(6, 16, 32, 3), (5, 32, 64, 2), (4, 64, 128, 2), (7, 24, 48, 4),
          (3, 48, 96, 1), (8, 16, 32, 5)]

QUANTITIES = ("matmuls", "matmul_cells", "moe_nodes", "moe_cells", "gates",
              "gate_slots", "lookup_queries", "lookup_table_rows")


def test_l1_shape_is_exact_against_the_emitted_trace():
    """Exact, not close. L1 is a statement about the protocol and the geometry;
    any gap is a missing term, never a tolerance."""
    for S, d, d_ff, E in SHAPES:
        toy = sem.ToyConfig(S=S, d=d, d_ff=d_ff, E=E, table_bits=6, scale_bits=6)
        real = sem.forward(toy, random.Random(7)).counts()
        pred = cm.predict_trace_shape(toy)
        for q in QUANTITIES:
            assert real[q] == getattr(pred, q), (
                f"S={S} d={d} d_ff={d_ff} E={E}: {q} real {real[q]:,} != "
                f"predicted {getattr(pred, q):,}")


def test_l1_counts_moe_nodes_not_per_token_matmuls():
    """The specific staleness that went unnoticed. Pinned on its own so a future
    change back to per-token expert matmuls fails loudly here."""
    toy = sem.ToyConfig(S=9, d=16, d_ff=32, E=3, table_bits=6, scale_bits=6)
    pred = cm.predict_trace_shape(toy)
    assert pred.matmuls == 5, f"expected 5 named matmuls, got {pred.matmuls}"
    assert pred.moe_nodes == 3, f"expected 3 MoE nodes, got {pred.moe_nodes}"


def test_forward_memory_tracks_measured_peaks():
    """Measured on a V100-SXM3-32GB by `bench/semantics_ladder.py`. The model is
    within 5% from d=256 up; the smallest point under-predicts, which is the
    usual fixed-overhead signature and is the harmless direction."""
    measured_gb = {(128, 256, 8, 4): 0.01, (256, 512, 16, 4): 0.03,
                   (384, 768, 16, 4): 0.07, (512, 1024, 32, 4): 0.13,
                   (1024, 2048, 32, 8): 0.88, (2048, 4096, 64, 8): 3.53,
                   (4096, 8192, 64, 8): 13.90}
    for (d, d_ff, S, E), meas in measured_gb.items():
        toy = sem.ToyConfig(S=S, d=d, d_ff=d_ff, E=E, table_bits=6, scale_bits=6)
        pred = sum(cm.predict_forward_memory(toy).values()) / 1e9
        ratio = pred / meas
        lo = 0.70 if d <= 128 else 0.90
        assert lo <= ratio <= 1.10, (
            f"d={d} E={E}: predicted {pred:.3f} GB against measured {meas:.3f} "
            f"GB, ratio {ratio:.2f}x")


def test_the_model_would_have_refused_both_runs_that_crashed():
    """Regression for 2026-08-05. Before L4 existed the model predicted hours
    for both of these and said nothing about the fact that neither would start."""
    card = 34.07e9                       # V100-SXM3-32GB, as reported by torch

    # 1. the forward pass at d=5120, E=16 -- died with CUDA OOM
    toy = sem.ToyConfig(S=128, d=5120, d_ff=10240, E=16, table_bits=6,
                        scale_bits=6)
    need = sum(cm.predict_forward_memory(toy).values())
    assert need > card, f"predicted {need/1e9:.1f} GB, which would have fitted"

    # 2. the LogUp commit at production geometry -- died inside the Bailey NTT
    #    scratch allocation, asking for 17.2 GB on a chunk of 32,832 rows
    cfg = rs.Config(ELL=8192, K_DEG=16384, N_LIG=65536, T_QUERIES=54)
    trace = sem.forward(sem.ToyConfig(S=8, d=128, d_ff=256, E=4, table_bits=6,
                                      scale_bits=6), random.Random(7))
    verdict = cm.will_it_fit(trace, cfg, card, chunk=32832)
    assert not verdict["fits"], "predicted the LogUp commit would fit; it did not"
    assert verdict["largest_term"] == "encode_transient", (
        f"blamed {verdict['largest_term']}, but the failure was the encode "
        f"scratch")

    # and with a chunk chosen to fit, the same run is predicted to succeed --
    # which is what actually happened once `gpu._chunk_rows` sized it
    ok = cm.will_it_fit(trace, cfg, card, chunk=cm.max_encode_rows(cfg, 18e9))
    assert ok["fits"], (
        f"a fitted chunk still predicted {ok['predicted_bytes']/1e9:.1f} GB")


def test_bailey_scratch_is_only_charged_at_n_65536():
    """The 4-step path in `prover/cuda_primitives.py` dispatches on NTT length
    exactly 65536, and its scratch is a raw cudaMalloc outside torch's allocator
    -- so it is invisible to `max_memory_allocated` and has to be modelled."""
    big = rs.Config(ELL=8192, K_DEG=16384, N_LIG=65536, T_QUERIES=54)
    small = rs.Config(ELL=1024, K_DEG=2048, N_LIG=4096, T_QUERIES=16)
    assert cm.ntt_scratch_bytes(big, 1000) == 1000 * 65536 * 8
    assert cm.ntt_scratch_bytes(small, 1000) == 0
