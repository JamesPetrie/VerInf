"""Fixture for tools/bench_static_check.py: the July 2026 break, written in the
three shapes a script can reach it. The toy demo as merged on 2026-08-11
(fc02dcc, demo/demo_toy_transformer.py:230) still passed slack_n_chunks to
Tape.rmsnorm, which the wrap-free bracket had removed. Never run: the checker
reads it. Each of the three marked calls must be reported."""
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(R / "prover")); sys.path.insert(0, str(R / "demo"))
from tape import Tape                                   # noqa: E402
import demo_toy_transformer as dt                       # noqa: E402
from demo_toy_transformer import _run_block             # noqa: E402

RMS_SLACK_N_CHUNKS = 4


def block(tape, x):
    return tape.rmsnorm(x, d=dt.d, s=dt.S, eps_int=dt.EPS_INT,     # REPORT: Tape method
                        slack_n_chunks=RMS_SLACK_N_CHUNKS,
                        s_out=dt.S, output_width=dt.OUTPUT_WIDTH)


def run(x, weights):
    tape = Tape(dt.CFG, silu_config=dt.SILU_CFG, lazy=True)
    dt._run_block(tape, x, weights, H=dt.d // dt.d_h,             # REPORT: module-qualified
                  slack_n_chunks=RMS_SLACK_N_CHUNKS)
    _run_block(tape, x, weights, H=dt.d // dt.d_h,                # REPORT: imported by name
               slack_n_chunks=RMS_SLACK_N_CHUNKS)
