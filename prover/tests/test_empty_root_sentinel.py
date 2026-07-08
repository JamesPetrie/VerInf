"""Negative gate for the empty-root sentinel (P6 audit finding S2).

The all-zeros root is the prover's sentinel for a zero-row block (no tree
exists — e.g. p2 on a tape with no phase-2 aux). merkle_test must accept the
sentinel ONLY for a genuinely empty block: before the fix it skipped merkle
verification for ANY block declaring the zero root, so a malicious prover
could zero out root_p1 (which no deployment comparison pins, unlike root_w)
and supply arbitrary opened columns — commitment binding off, forged
witnesses ACCEPT.

Gates:
1. Baseline ACCEPT, and the toy tape's p2 block is genuinely empty with the
   zero root — the sentinel's legitimate use still works.
2. For each non-empty block (blind, w, p1): zeroing its declared root while
   keeping the real opened columns/paths → REJECT.

Run on the Spark:  ~/venv-hf/bin/python run_tests.py test_empty_root_sentinel
"""
import json, os, subprocess, sys, pathlib, tempfile
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch
import core
import protocol as pr
from tape import Tape
from proof_dump import dump_proof
from _rust_verify import _verify_proof_bin

CFG = core.LigeroConfig(ELL=8192, K_DEG=16384, N_LIG=65536, T_QUERIES=4)
SEED = b"empty-root"


def _t(vals):
    return torch.tensor(vals, dtype=torch.int64, device="cuda").to(torch.uint64)


def _build():
    tape = Tape(CFG, lazy=True)
    w1 = tape.commit("W1", _t(list(range(12000))), (12000,), persistent=True)
    w2 = tape.commit("W2", _t([v * 2 for v in range(12000)]), (12000,), persistent=True)
    tape.add(w1, w2)
    return tape


def _dump(tape, proof, path):
    s_op, s_comb, s_col = pr.round_seeds(SEED)
    Q = list(pr.random_columns(s_col, CFG))
    seeds = {"s_op": s_op.hex(), "s_comb": s_comb.hex(), "s_col": s_col.hex()}
    dump_proof(path, pr.claims_to_json(tape.claims, CFG), seeds, proof, Q, None)


def _run(path):
    r = subprocess.run([_verify_proof_bin(), path], capture_output=True, text=True)
    return "rust_verify: ACCEPT" in r.stdout


def test_zero_root_sentinel():
    tape = _build()
    proof = tape.prove(seed=SEED)
    fd, path = tempfile.mkstemp(suffix=".json"); os.close(fd)
    try:
        _dump(tape, proof, path)
        d = json.load(open(path))

        # Gate 1: baseline ACCEPT, with p2 genuinely empty under the sentinel
        # root — the legitimate use of the zero root keeps working.
        assert d["proof"]["root_p2"] == "00" * 32, \
            "toy tape's p2 should be empty (sentinel root) — test premise"
        assert all(len(col) == 0 for col in d["proof"]["opened_p2"].values())
        assert _run(path), "baseline proof (with a genuinely empty p2) must ACCEPT"
        print("    baseline ACCEPT (p2 empty under the sentinel root)")

        # Gate 2: zeroing a NON-empty block's root must REJECT — the sentinel
        # may not stand in for a real root.
        for blk in ("blind", "w", "p1"):
            t = json.loads(json.dumps(d))          # deep copy
            assert any(len(c) > 0 for c in t["proof"][f"opened_{blk}"].values()), \
                f"{blk} should be non-empty — test premise"
            t["proof"][f"root_{blk}"] = "00" * 32
            fd2, path2 = tempfile.mkstemp(suffix=".json"); os.close(fd2)
            try:
                json.dump(t, open(path2, "w"))
                acc = _run(path2)
                print(f"    zeroed root_{blk} (opened columns intact): "
                      f"{'ACCEPT' if acc else 'REJECT'} (want REJECT)")
                assert not acc, f"zero root_{blk} with non-empty columns MUST be rejected"
            finally:
                os.unlink(path2)
    finally:
        os.unlink(path)


if __name__ == "__main__":
    try:
        test_zero_root_sentinel(); print("[OK ] test_zero_root_sentinel")
    except Exception as e:
        print(f"[XX ] test_zero_root_sentinel: {e}"); sys.exit(1)
