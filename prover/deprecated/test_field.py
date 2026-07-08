"""Diff-test verify.py's vectorized numpy field against protocol.py's int oracle.

A wrong gmul is a silent soundness bug, so this is the safety net: the numpy
field must be bit-identical to plain (a*b)%P on random + edge-case inputs.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # pipeline/ on path
import numpy as np
import protocol as pr
import verify as vf

rng = np.random.default_rng(0)
N = 200_000
edges = np.array([0, 1, 2, pr.P - 1, pr.P - 2, 1 << 32, (1 << 32) - 1, 1 << 63],
                 dtype=np.uint64)
a = np.concatenate([rng.integers(0, pr.P, N, dtype=np.uint64), edges, np.flip(edges)])
b = np.concatenate([rng.integers(0, pr.P, N, dtype=np.uint64), np.flip(edges), edges])

for name, fn, oracle in [("gmul", vf.gmul, pr.mul),
                         ("gadd", vf.gadd, pr.add),
                         ("gsub", vf.gsub, pr.sub)]:
    got = fn(a, b)
    ref = np.array([oracle(int(x), int(y)) for x, y in zip(a, b)], dtype=np.uint64)
    if not np.array_equal(got, ref):
        i = int(np.argmax(got != ref))
        raise AssertionError(
            f"{name} mismatch at {i}: {int(a[i])},{int(b[i])} -> "
            f"{int(got[i])} != {int(ref[i])}")
    print(name, "ok")

s_got = int(vf.gsum(a))
s_ref = sum(int(x) for x in a) % pr.P
assert s_got == s_ref, f"gsum mismatch: {s_got} != {s_ref}"
print("gsum ok")
print(f"FIELD DIFF-TEST PASSED ({len(a)} ops each)")
