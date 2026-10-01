"""Fixture for tools/bench_static_check.py: one of each other defect the
checker reports. Never run. Each marked line must be reported."""
import os
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(R / "prover")); sys.path.insert(0, str(R / "demo"))
import demo_toy_transformer as dt                       # noqa: E402
from tape import NoSuchTapeHelper                       # REPORT: not defined  # noqa: E402,F401
import no_such_bench_module                             # REPORT: unresolved  # noqa: E402,F401
from demo_toy_transformer import _run_block             # noqa: E402

os.environ["LIGERO_NO_SUCH_KNOB"] = "1"                 # REPORT: knob read nowhere


def run(tape, x, weights):
    dt._run_block(tape, x)                              # REPORT: missing weights, H
    _run_block(tape, x, weights, 4, H=1)                # REPORT: too many positional
    return dt.NO_SUCH_CONSTANT                          # REPORT: not defined in the demo
