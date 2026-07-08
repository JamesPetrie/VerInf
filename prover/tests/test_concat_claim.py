"""ConcatClaim: dst = srcs concatenated — positive + tamper REJECT.
Run on the Spark:  ~/venv-hf/bin/python tests/test_concat_claim.py"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch
import core
import claims as _C        # noqa: F401
import packets as _PK      # noqa: F401
import compute_fns as _cf
from tape import Tape
from _rust_verify import rust_verify_tape

CFG = core.LigeroConfig(ELL=8, K_DEG=8, N_LIG=32, T_QUERIES=4)
SEED = b"concat-test"


def _build(tamper=None):
    core._COSET_POWERS_K_CACHE.clear()
    _cf.CONCAT_TAMPER.clear()
    tape = Tape(CFG, lazy=True)
    a = tape.commit("a", torch.arange(5, dtype=torch.int64, device="cuda").to(torch.uint64), (5,))
    b = tape.commit("b", (10 + torch.arange(9, dtype=torch.int64, device="cuda")).to(torch.uint64), (9,))
    dst = tape.concat([a, b], (14,))
    if tamper is not None:
        _cf.CONCAT_TAMPER[dst.var.name] = torch.tensor(tamper, dtype=torch.int64,
                                                        device="cuda").to(torch.uint64)
    return tape, dst


def test_positive():
    tape, dst = _build()
    live = tape.run_engine_pass()
    vals = live[dst.var].to(torch.int64).cpu().tolist()
    assert vals == list(range(5)) + list(range(10, 19)), vals
    tape2, _ = _build()
    acc, msg = rust_verify_tape(tape2, tape2.prove(seed=SEED), seed=SEED)
    assert acc, f"expected ACCEPT ({msg})"
    print(f"    positive: ACCEPT ok, dst={vals}")


def test_tamper_reject():
    try:
        tape, _ = _build(tamper=[7] * 14)
        acc, msg = rust_verify_tape(tape, tape.prove(seed=SEED), seed=SEED)
    finally:
        _cf.CONCAT_TAMPER.clear()
    assert not acc, "tampered concat: expected REJECT"
    print(f"    tamper: REJECT ok ({msg})")


def main():
    fails = 0
    for t in [test_positive, test_tamper_reject]:
        try:
            t(); print(f"[OK ] {t.__name__}")
        except Exception as e:
            fails += 1; print(f"[XX ] {t.__name__}: {e}")
    print(f"=== concat_claim: {2-fails}/2 {'PASS' if not fails else 'FAIL'} ===")
    return fails


if __name__ == "__main__":
    raise SystemExit(main())
