"""Fixture for tools/bench_static_check.py: the same calls as
july_regression.py, valid on the current tree, plus an import from
prover/tests (as analysis/bench/ab_routed_cache.py makes) and a dataclass
built by keyword. Never run: the checker reads it and must report nothing."""
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(R / "prover")); sys.path.insert(0, str(R / "demo"))
sys.path.insert(0, str(R / "prover/tests"))
from tape import Tape                                   # noqa: E402
from claims import SiluConfig                           # noqa: E402
import demo_toy_transformer as dt                       # noqa: E402
from demo_toy_transformer import _run_block             # noqa: E402
import test_shard_streaming as tss                      # noqa: E402


def block(tape, x):
    return tape.rmsnorm(x, d=dt.d, s=dt.S, eps_int=dt.EPS_INT,
                        s_out=dt.S, output_width=dt.OUTPUT_WIDTH)


def run(x, weights):
    tape = Tape(dt.CFG, silu_config=dt.SILU_CFG, lazy=True)
    dt._run_block(tape, x, weights, H=dt.d // dt.d_h)
    _run_block(tape, x, weights, H=dt.d // dt.d_h)
    tss.test_one_expert_shard_resident_at_a_time()
    return SiluConfig(b=4, T_LEN=1 << 14, b_2=1 << 16, b_3=1 << 32, b_4=1 << 48,
                      width_2=16, width_3=16, width_4=14, r=12)
