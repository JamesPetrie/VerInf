"""Test runner for the layergkr prototype — same shape as prover/tests/run_tests.py
(no pytest dependency). Runs every top-level `test_*` in the named module, or all
layergkr test modules when called with no argument.

  .venv/bin/python layergkr/tests/run_tests.py                # everything
  .venv/bin/python layergkr/tests/run_tests.py test_projection # one module

Exit code is the number of failures.
"""
import importlib
import pathlib
import sys
import traceback

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

ALL = sorted(p.stem for p in HERE.glob("test_*.py"))


def run_module(name: str) -> tuple:
    mod = importlib.import_module(name)
    tests = sorted(n for n in dir(mod)
                   if n.startswith("test_") and callable(getattr(mod, n)))
    passed = failed = 0
    print(f"--- {name} ({len(tests)} tests)")
    for n in tests:
        try:
            getattr(mod, n)()
            print(f"  PASS {n}")
            passed += 1
        except Exception as e:
            print(f"  FAIL {n}: {type(e).__name__}: {e}")
            traceback.print_exc()
            failed += 1
    return passed, failed


def main() -> int:
    mods = [sys.argv[1]] if len(sys.argv) > 1 else ALL
    tp = tf = 0
    for m in mods:
        p, f = run_module(m)
        tp += p
        tf += f
    print(f"=== {tp} passed, {tf} failed (of {tp + tf}) ===")
    return tf


if __name__ == "__main__":
    sys.exit(main())
