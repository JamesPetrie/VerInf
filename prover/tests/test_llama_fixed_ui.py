# Copyright (c) 2026 Levi Neuwirth. Released under the MIT License (see LICENSE).
"""Fixed bound parameters in the Llama driver, and the statement preview.

demo_llama7b._run_unexplained_info with an explicit s_b and a fixed gap_max:
the values reach InfoFinalizeClaim, the revealed Sz matches an independent
NumPy computation, a run outside the fixed range is refused before any claim
is appended, and the legacy call is unchanged. statement_preview writes the
exact claim bytes, block order and statement digest the proof then carries.

GPU (CUDA tape) and the Rust verifier binary (see _rust_verify.py).
"""
import math
import pathlib
import sys

import numpy as np
import pytest
import torch

_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
for p in (_HERE, _ROOT / "prover", _ROOT / "demo"):
    sys.path.insert(0, str(p))

import core                                   # noqa: E402
import demo_llama7b as D                      # noqa: E402
from statement_preview import write_statement  # noqa: E402
from tape import Tape                         # noqa: E402
from _rust_verify import rust_verify_tape     # noqa: E402

CFG = core.LigeroConfig(ELL=8, K_DEG=8, N_LIG=32, T_QUERIES=4)
SEED = b"llama-fixed-ui-test"
T, V = 4, 8
S_C, S_B, GAP_MAX = 256, 16, 128             # k = 16
ROWS = [[40, 8, 32, 0, 24, 16, 36, 4],
        [4, 44, 12, 36, 20, 8, 40, 28],
        [10, 10, 0, 5, 7, 30, 2, 1],
        [0, 1, 2, 3, 4, 5, 6, 60]]
TOKENS = [2, 6, 0, 7]                        # hidden outputs: gaps 8, 4, 20, 0
POSITIONS = [1, 2, 3]


def _commit(tape, rows):
    flat = torch.tensor([v for r in rows for v in r], dtype=torch.int64,
                        device="cuda").to(torch.uint64)
    return tape.commit("logits", flat, (len(rows), V))


def _reference_sz(rows, tokens, positions, s_c, s_b, s_y):
    """Independent of the prover's code: same formulas, written out per row."""
    k = s_c // s_b
    total = 0
    for t in positions:
        row = np.array(rows[t], dtype=np.int64)
        gaps = row.max() - row
        e = np.maximum(1.0, np.ceil(np.exp(-(gaps.astype(np.float64) ** 2) / s_c) * s_y))
        a = int(e.astype(np.uint64).sum())
        b = 0
        while int(np.floor(s_y * np.exp(np.float64(b) / s_b))) < a:
            b += 1
        g = int(gaps[tokens[t]])
        total += (g * g + k - 1) // k + b
    return total


def _build(rows=ROWS, **kw):
    core._COSET_POWERS_K_CACHE.clear()
    tape = Tape(CFG, lazy=True)
    logits = _commit(tape, rows)
    Sz, info = D._run_unexplained_info(tape, logits, vocab_size=V, seq=len(rows),
                                       output_tokens=TOKENS, sum_positions=POSITIONS, **kw)
    return tape, Sz, info


def _reveal(tape, Sz, info):
    """The driver's reveal step: engine pass, pin Sz, re-zero the LogUp tables."""
    live = tape.run_engine_pass(free_intermediates=True, keep={Sz.var})
    sz = int(live[Sz.var].cpu().item())
    info["reveal_pin"].public_rhs = sz
    for v in list(tape.inputs):
        if v.name.endswith("_mult"):
            tape.inputs[v].zero_()
    return sz


def test_fixed_parameters_reach_the_claim_and_the_sum():
    tape, Sz, info = _build(s_c=S_C, s_b=S_B, gap_max=GAP_MAX)
    assert (info["s_b"], info["gap_max"], info["observed_gap"]) == (S_B, GAP_MAX, 60)
    fin = [c for c in tape.claims if type(c).__name__ == "InfoFinalizeClaim"]
    assert len(fin) == 1 and fin[0].s_b == S_B and fin[0].k == S_C // S_B
    sz = _reveal(tape, Sz, info)
    assert sz == _reference_sz(ROWS, TOKENS, POSITIONS, S_C, S_B, 1 << D.UI_S_Y_BITS)


def test_out_of_range_run_is_refused_before_any_claim():
    rows = [list(r) for r in ROWS]
    rows[3][7] = GAP_MAX                      # largest gap becomes GAP_MAX
    core._COSET_POWERS_K_CACHE.clear()
    tape = Tape(CFG, lazy=True)
    logits = _commit(tape, rows)
    before = len(tape.claims)
    with pytest.raises(ValueError, match="outside the fixed range"):
        D._run_unexplained_info(tape, logits, vocab_size=V, seq=T, output_tokens=TOKENS,
                                s_c=S_C, s_b=S_B, gap_max=GAP_MAX, sum_positions=POSITIONS)
    assert len(tape.claims) == before


def test_legacy_call_derives_the_range_from_the_run():
    tape, Sz, info = _build(s_c=1 << 18)
    assert info["s_b"] == 1 << D.UI_S_B_BITS
    assert info["gap_max"] == info["observed_gap"] + 2 == 62


def test_statement_preview_is_what_the_proof_carries(tmp_path):
    tape, Sz, info = _build(s_c=S_C, s_b=S_B, gap_max=GAP_MAX)
    sz = _reveal(tape, Sz, info)
    path = str(tmp_path / "statement.claims")
    meta = write_statement(tape, CFG, path, ui_info={"s_b": S_B}, sz=sz)
    raw = pathlib.Path(path).read_bytes()
    proof = tape.prove(seed=SEED)
    assert proof.claims_bytes == raw
    assert list(proof.blocks) == meta["blocks"]
    assert proof.statement_digest.hex() == meta["statement_digest"]
    acc, msg = rust_verify_tape(tape, proof, seed=SEED)
    assert acc, msg
    with pytest.raises(FileExistsError):
        write_statement(tape, CFG, path)


def test_a_different_fixed_range_is_a_different_statement(tmp_path):
    digests = []
    for gap_max in (GAP_MAX, 2 * GAP_MAX):
        tape, Sz, info = _build(s_c=S_C, s_b=S_B, gap_max=gap_max)
        sz = _reveal(tape, Sz, info)
        meta = write_statement(tape, CFG, str(tmp_path / f"s{gap_max}"), sz=sz)
        digests.append((meta["statement_digest"], sz))
    assert digests[0][0] != digests[1][0]
    assert digests[0][1] == digests[1][1]     # same rows: the range changes the statement, not Sz
