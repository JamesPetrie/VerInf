"""core.layout_breakdown — the LIGERO_LAYOUT_BREAKDOWN table as a function.

The profiler's crosscheck probes a Maverick tape's row layout IN-PROCESS
with it (the demo's main() is fail-closed on main and can no longer serve
as a subprocess probe), so three things must stay true:

1. it is value-free: a lazy tape whose weight loaders would RAISE if
   resolved lays out fine (extraction and the probe never touch weights);
2. format_layout_breakdown prints exactly what _stream_setup printed —
   the crosscheck parser and the archived probe files share that text;
3. its aggregation equals the profiler's manifest mirror row for row.

CPU-only (no CUDA): the tape is concat-only with CPU tensors.
"""
import sys
import pathlib

_HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))                      # prover/
sys.path.insert(0, str(_HERE.parents[2] / "profiler"))         # profiler/
import torch
import core
from tape import Tape

CFG = core.LigeroConfig(ELL=8192, K_DEG=16384, N_LIG=65536, T_QUERIES=4)
LENS = [12000, 8192, 20000, 5, 16384, 9000, 3000]


def _t(vals):
    return torch.tensor(vals, dtype=torch.int64).to(torch.uint64)


def _raising(name):
    def load():
        raise RuntimeError(f"{name}: a layout probe resolved a weight loader")
    return load


def _build(lazy_weights):
    tape = Tape(CFG, lazy=True)
    ws = []
    for i, n in enumerate(LENS):
        if lazy_weights:
            ws.append(tape.commit_lazy(f"W{i}", _raising(f"W{i}"), (n,), n))
        else:
            ws.append(tape.commit(f"W{i}", _t([(v * (i + 3)) % 1000003 for v in range(n)]),
                                  (n,), persistent=True))
    a = tape.concat([ws[0], ws[1]], (LENS[0] + LENS[1],))
    b = tape.concat([ws[2], ws[3], ws[4]], (sum(LENS[2:5]),))
    c = tape.concat([ws[5], ws[6]], (LENS[5] + LENS[6],))
    tape.concat([a, b, c], (sum(LENS),))
    return tape


def test_breakdown_is_value_free_and_matches_layout():
    tape = _build(lazy_weights=True)              # loaders raise if touched
    m_total, table = core.layout_breakdown(tape, CFG)
    claims = core._with_synthesized_settlements(tape.claims)
    lay = core._layout(claims, CFG)
    assert m_total == lay[6]
    w_rows = sum(-(-n // CFG.ELL) for n in LENS)
    act_rows = sum(-(-n // CFG.ELL) for n in (LENS[0] + LENS[1], sum(LENS[2:5]),
                                              LENS[5] + LENS[6], sum(LENS)))
    assert table == {"ConcatClaim": (w_rows + act_rows,
                                     sum(LENS) + (LENS[0] + LENS[1]) + sum(LENS[2:5])
                                     + (LENS[5] + LENS[6]) + sum(LENS))}
    assert m_total == core.NUM_BLINDING_ROWS + w_rows + act_rows
    print(f"    value-free: {m_total} rows laid out with raising loaders")


def test_printed_form_round_trips_through_crosscheck():
    import crosscheck
    from extract import extract_tape
    tape = _build(lazy_weights=False)
    m_total, table = core.layout_breakdown(tape, CFG)
    text = core.format_layout_breakdown(m_total, table, CFG)
    assert text.startswith("=== witness layout by claim type (m_total=")
    mt, parsed = crosscheck.parse_layout(text + "\n")
    assert mt == m_total and parsed == table
    man = extract_tape(tape, model=dict(name="toy"), seq=1)
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        flags = crosscheck.diff_layout(mt, parsed, man)
    assert flags == [], flags
    assert mt - crosscheck.rows_total(man) == core.NUM_BLINDING_ROWS
    print("    printed table parses back and matches the manifest mirror")
