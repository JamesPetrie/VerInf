"""MaxClaim: prove v* = max_i l, emit gap = v* - l >= 0. Correctness + soundness.

POSITIVE: build, prove + verify (ACCEPT), confirm gap matches v* - l.
SOUNDNESS: build a consistent witness around a non-max v* (force_argmax = a
non-max index); gap then has negative entries, so the gap >= 0 range LogUp must
REJECT -- this is the exact-max binding.

Run on the Spark:  ~/venv-hf/bin/python tests/test_max_claim.py
"""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import core
import claims as _C        # noqa: F401
import packets as _PK       # noqa: F401
from tape import Tape
from _rust_verify import rust_verify_tape
from max_claim import max_gap

CFG = core.LigeroConfig(ELL=8, K_DEG=8, N_LIG=32, T_QUERIES=4)
SEED = b"max-claim-test"
T, V = 2, 4
GAP_MAX = 64
LOGITS = [[40, 20, 30, 10], [15, 45, 25, 5]]   # row maxes: 40@0, 45@1
TOKENS = [2, 0]                                # hidden output tokens -> gap_o = [40-30, 45-15] = [10, 30]


def _t(rows):
    flat = [v for row in rows for v in row]
    return torch.tensor(flat, dtype=torch.int64, device="cuda").to(torch.uint64)


def run_positive():
    core._COSET_POWERS_K_CACHE.clear()
    tape = Tape(CFG, lazy=True)
    logits = tape.commit("logits", _t(LOGITS), (T, V))
    gap, gap_o, neg_gap, vstar = max_gap(tape, logits, TOKENS, T=T, V=V, gap_max=GAP_MAX)
    live = tape.run_engine_pass()
    gap_w = live[gap.var].view(T, V).to(torch.int64).cpu().tolist()
    vs_w = live[vstar.var].to(torch.int64).cpu().tolist()
    go_w = live[gap_o.var].to(torch.int64).cpu().tolist()
    exp_gap = [[max(r) - x for x in r] for r in LOGITS]
    exp_vs = [max(r) for r in LOGITS]
    exp_go = [LOGITS[t][TOKENS[t]] for t in range(T)]    # gap_o = v* - l[tok]
    exp_go = [exp_vs[t] - exp_go[t] for t in range(T)]

    tape2 = Tape(CFG, lazy=True)
    l2 = tape2.commit("logits", _t(LOGITS), (T, V))
    max_gap(tape2, l2, TOKENS, T=T, V=V, gap_max=GAP_MAX)
    acc, msg = rust_verify_tape(tape2, tape2.prove(seed=SEED), seed=SEED)

    ok = acc and gap_w == exp_gap and vs_w == exp_vs and go_w == exp_go
    print(f"[{'OK ' if ok else 'XX '}] positive: verify={'ACCEPT' if acc else 'REJECT'}  "
          f"gap_ok={gap_w == exp_gap}  vstar_ok={vs_w == exp_vs}  "
          f"gap_o_ok={go_w == exp_go} (={go_w})  ({msg})")
    return ok


def run_cheat(label, force_argmax):
    core._COSET_POWERS_K_CACHE.clear()
    tape = Tape(CFG, lazy=True)
    logits = tape.commit("logits", _t(LOGITS), (T, V))
    max_gap(tape, logits, TOKENS, T=T, V=V, gap_max=GAP_MAX, force_argmax=force_argmax)
    acc, msg = rust_verify_tape(tape, tape.prove(seed=SEED), seed=SEED)
    ok = not acc
    print(f"[{'OK ' if ok else 'XX '}] {label}: verify={'ACCEPT' if acc else 'REJECT'}  "
          f"(want REJECT)  ({msg})")
    return ok


def main():
    results = [
        run_positive(),
        run_cheat("cheat: v* = non-max token", force_argmax=[2, 2]),   # not the argmax
    ]
    ok = all(results)
    print(f"\n=== max_claim: {sum(results)}/{len(results)} {'PASS' if ok else 'FAIL'} ===")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
