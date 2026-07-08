"""Reveal pin: expose a committed value as a PUBLIC bound via AddClaim's
public_rhs (x == public_rhs). Verifier reads the value from the claim and
ACCEPTs iff it matches the committed value. Wrong value -> REJECT.
Run on the Spark:  ~/venv-hf/bin/python tests/test_reveal.py"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch
import core
import claims as _C        # noqa
import packets as _PK      # noqa
from tape import Tape
from _rust_verify import rust_verify_tape

CFG = core.LigeroConfig(ELL=8, K_DEG=8, N_LIG=32, T_QUERIES=4)
SEED = b"reveal-test"


def _build_scalar(reveal_value):
    core._COSET_POWERS_K_CACHE.clear()
    tape = Tape(CFG, lazy=True)
    a = tape.commit("a", torch.tensor([41], dtype=torch.int64, device="cuda").to(torch.uint64), (1,))
    b = tape.commit("b", torch.tensor([1], dtype=torch.int64, device="cuda").to(torch.uint64), (1,))
    s = tape.add(a, b)                  # committed = 42
    tape.reveal(s, value=reveal_value)  # assert s == reveal_value (public)
    return tape


def test_reveal_correct():
    tape = _build_scalar(42)
    acc, msg = rust_verify_tape(tape, tape.prove(seed=SEED), seed=SEED)
    assert acc, f"correct reveal: expected ACCEPT ({msg})"
    pub = [c.public_rhs for c in tape.claims if getattr(c, "public_rhs", None) is not None]
    assert pub == [42], pub
    print(f"    reveal correct: ACCEPT, public bound readable = {pub[0]}")


def test_reveal_wrong():
    tape = _build_scalar(99)            # lie: claim s == 99 (really 42)
    acc, msg = rust_verify_tape(tape, tape.prove(seed=SEED), seed=SEED)
    assert not acc, "wrong reveal: expected REJECT"
    print(f"    reveal wrong: REJECT ok ({msg})")


def main():
    fails = 0
    for t in [test_reveal_correct, test_reveal_wrong]:
        try:
            t(); print(f"[OK ] {t.__name__}")
        except Exception as e:
            fails += 1; print(f"[XX ] {t.__name__}: {e}")
    print(f"=== reveal: {2-fails}/2 {'PASS' if not fails else 'FAIL'} ===")
    return fails


if __name__ == "__main__":
    raise SystemExit(main())
