"""LinCombClaim gates: sum_k coefs[k]·xs[k][i] = rhs[i] with a public RHS —
the one new TCB surface of the token-binding gadgets (token-binding.md §12 P2).

Positive: per-slot RHS (exercises the run compression) and constant RHS both
ACCEPT. Negative: a lie in any slot, or a wrong public RHS, REJECTs.

Run on the Spark:  ~/venv-hf/bin/python tests/test_lincomb.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch
import core
from core import P
import claims as _C        # noqa: F401
import packets as _PK      # noqa: F401
from tape import Tape
from _rust_verify import rust_verify_tape

CFG = core.LigeroConfig(ELL=8, K_DEG=8, N_LIG=32, T_QUERIES=4)
SEED = b"lincomb-test"

A = [1, 2, 3, 4, 5, 6]
B = [10, 20, 30, 40, 50, 60]


def _commit(tape, name, vals):
    t = torch.tensor([v % P for v in vals], dtype=torch.uint64, device="cuda")
    return tape.commit(name, t, (len(vals),))


def _build(rhs, lie_slot=None, rhs_claim=None):
    """Constrain 3a + 5b - y = rhs_claim (defaults to rhs), with y honestly
    computed from `rhs` (optionally off by one in lie_slot). Passing a
    different rhs_claim tests the public-RHS path itself."""
    core._COSET_POWERS_K_CACHE.clear()
    tape = Tape(CFG, lazy=True)
    rhs_list = rhs if isinstance(rhs, list) else [rhs] * len(A)
    y = [3 * a + 5 * b - r for a, b, r in zip(A, B, rhs_list)]
    if lie_slot is not None:
        y[lie_slot] += 1
    a = _commit(tape, "a", A)
    b = _commit(tape, "b", B)
    yv = _commit(tape, "y", y)
    tape.lincomb([a, b, yv], [3, 5, -1], rhs_claim if rhs_claim is not None else rhs)
    return tape


def run_case(label, rhs, lie_slot, want_accept, rhs_claim=None):
    tape = _build(rhs, lie_slot, rhs_claim)
    acc, msg = rust_verify_tape(tape, tape.prove(seed=SEED), seed=SEED)
    ok = acc == want_accept
    print(f"[{'OK ' if ok else 'XX '}] {label}: "
          f"verify={'ACCEPT' if acc else 'REJECT'} "
          f"(want {'ACCEPT' if want_accept else 'REJECT'}) ({msg})")
    return ok


def main():
    per_slot = [7, 7, 100, 0, P - 3, 7]        # runs of equal + distinct values
    results = [
        run_case("per-slot rhs", per_slot, None, True),
        run_case("constant rhs", 5, None, True),
        run_case("cheat: slot lie (per-slot rhs)", per_slot, 2, False),
        run_case("cheat: slot lie (constant rhs)", 5, 0, False),
        run_case("cheat: wrong public rhs", per_slot, None, False,
                 rhs_claim=[8, 7, 100, 0, P - 3, 7]),
    ]
    fails = results.count(False)
    print(f"=== lincomb: {len(results) - fails}/{len(results)} "
          f"{'PASS' if not fails else 'FAIL'} ===")
    return fails


if __name__ == "__main__":
    raise SystemExit(main())
