"""Minimal test runner (no pytest on the Spark). Imports a test module and
calls every top-level `test_*` function, reporting pass/fail counts. Exit code
is the number of failures."""
import importlib
import sys
import traceback

mod = importlib.import_module(sys.argv[1] if len(sys.argv) > 1 else "test_claims")
names = sorted(n for n in dir(mod) if n.startswith("test_") and callable(getattr(mod, n)))
passed = failed = 0
for n in names:
    try:
        getattr(mod, n)()
        print(f"  PASS {n}")
        passed += 1
    except Exception as e:
        print(f"  FAIL {n}: {type(e).__name__}: {e}")
        traceback.print_exc()
        failed += 1
print(f"=== {passed} passed, {failed} failed (of {len(names)}) ===")
sys.exit(failed)
