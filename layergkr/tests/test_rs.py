"""Gates for rs.py: the cached/batched encoding path against the reference.

The Lagrange matrix has a fast build (one `eta^K` per column, one Montgomery
inversion per column instead of ELL Fermat inversions -- ~50x). Fast is only
useful if it is the SAME matrix, so every gate here compares against
`prover.protocol.lagrange` and `eval_zeta_form` directly.

Run:  .venv/bin/python layergkr/tests/run_tests.py test_rs
"""
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from layergkr import rs
from prover.protocol import eval_zeta_form, lagrange

CONFIGS = [rs.Config(ELL=16, K_DEG=32, N_LIG=64, T_QUERIES=4),
           rs.Config(ELL=64, K_DEG=128, N_LIG=256, T_QUERIES=8)]


def test_batched_lagrange_matches_the_reference():
    """Every cell, on the small config; a spread of cells on the larger one."""
    cfg = CONFIGS[0]
    M = rs.lagrange_matrix(cfg)
    for j in range(cfg.N_LIG):
        for c in range(cfg.ELL):
            assert M[j][c] == lagrange(cfg, c, cfg.eta(j)), f"cell ({j},{c})"

    cfg = CONFIGS[1]
    M = rs.lagrange_matrix(cfg)
    r = random.Random(5)
    for _ in range(200):
        j, c = r.randrange(cfg.N_LIG), r.randrange(cfg.ELL)
        assert M[j][c] == lagrange(cfg, c, cfg.eta(j)), f"cell ({j},{c})"


def test_encode_matches_eval_zeta_form():
    """The cached path must equal the reference evaluation, message by message."""
    for cfg in CONFIGS:
        r = random.Random(3)
        msg = [r.randrange(rs.P) for _ in range(cfg.ELL)]
        cw = rs.encode_row(cfg, msg)
        for j in (0, 1, cfg.N_LIG // 3, cfg.N_LIG - 1):
            assert cw[j] == eval_zeta_form(cfg, msg, cfg.eta(j)), f"column {j}"


def test_sparse_message_encodes_like_a_padded_one():
    """encode_row skips zero slots; the result must equal the dense evaluation."""
    cfg = CONFIGS[0]
    r = random.Random(9)
    msg = [r.randrange(rs.P) if i % 3 == 0 else 0 for i in range(cfg.ELL)]
    cw = rs.encode_row(cfg, msg)
    for j in (0, 2, cfg.N_LIG - 1):
        assert cw[j] == eval_zeta_form(cfg, msg, cfg.eta(j))


def test_short_message_is_zero_padded():
    cfg = CONFIGS[0]
    r = random.Random(11)
    short = [r.randrange(rs.P) for _ in range(cfg.ELL // 2)]
    assert rs.encode_row(cfg, short) == rs.encode_row(
        cfg, short + [0] * (cfg.ELL - len(short)))
