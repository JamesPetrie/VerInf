"""P1 gate (analysis/persistent-weights.md): the persistent W block's root
R_W is deterministic and CONTEXT-INDEPENDENT — the same weights reproduce
R_W byte-for-byte regardless of the per-proof activation/aux vars around
them. That reproducibility is what lets a later proof reference an earlier
weight commitment.

Run on the Spark:  ~/venv-hf/bin/python run_tests.py test_persistent_weights
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch
import core
from tape import Tape

CFG = core.LigeroConfig(ELL=8192, K_DEG=16384, N_LIG=65536, T_QUERIES=4)


def _t(vals):
    return torch.tensor(vals, dtype=torch.int64, device="cuda").to(torch.uint64)


def _weights(tape):
    """Commit three 'model weight' blocks (persistent) of varying size, and
    reference each in a claim (add(w,w)) so it enters the witness — commit_weights
    walks the CLAIMS, like the in-proof W tree, so an unreferenced weight is
    excluded from both."""
    torch.manual_seed(0)
    for name, n in (("W_a", 5000), ("W_b", 12000), ("W_c", 3333)):
        data = (torch.randint(0, 1 << 20, (n,), dtype=torch.int64)).to(torch.uint64).cuda()
        wt = tape.commit(name, data, (n,), persistent=True)
        tape.add(wt, wt)                       # reference it → in the claim graph


def test_deterministic():
    """Same weights, committed twice → identical R_W."""
    t1 = Tape(CFG, lazy=True); _weights(t1)
    t2 = Tape(CFG, lazy=True); _weights(t2)
    a1, _, m1 = core.commit_weights(t1, CFG)
    a2, _, m2 = core.commit_weights(t2, CFG)
    assert m1 == m2 and a1.root == a2.root, "R_W not deterministic across runs"
    print(f"    deterministic: R_W={a1.root[:6].hex()}… m_w={m1} rows")


def test_context_independent():
    """Same weights but DIFFERENT activation vars interleaved → same R_W.
    Weights are committed at their op position amongst activations, so this
    checks the W-block sweep ignores non-persistent vars and their ordering."""
    ta = Tape(CFG, lazy=True)
    ta.commit("act0", _t(list(range(64))), (64,))          # activation (not persistent)
    _weights(ta)
    ta.commit("act1", _t(list(range(100))), (100,))        # more activations

    tb = Tape(CFG, lazy=True)
    _weights(tb)                                            # weights only, no activations
    tb.commit("actX", _t(list(range(7))), (7,))

    aa, _, ma = core.commit_weights(ta, CFG)
    ab, _, mb = core.commit_weights(tb, CFG)
    assert ma == mb, f"m_w differs with activations present: {ma} vs {mb}"
    assert aa.root == ab.root, "R_W depends on surrounding activations (NOT context-independent)"
    print(f"    context-independent: R_W stable across differing activations (m_w={ma})")


def test_changes_with_weights():
    """Sanity: a different weight value DOES change R_W (the commit binds)."""
    t1 = Tape(CFG, lazy=True); _weights(t1)
    t2 = Tape(CFG, lazy=True); _weights(t2)
    # perturb one weight in t2
    for v in list(t2.inputs):
        if getattr(v, "name", "") == "W_b":
            t2.inputs[v] = t2.inputs[v].clone()
            t2.inputs[v][0] = (int(t2.inputs[v][0]) + 1) % ((1 << 64) - (1 << 32) + 1)
            break
    a1, _, _ = core.commit_weights(t1, CFG)
    a2, _, _ = core.commit_weights(t2, CFG)
    assert a1.root != a2.root, "R_W unchanged after perturbing a weight (not binding)"
    print("    binding: perturbing one weight changes R_W")


if __name__ == "__main__":
    ok = True
    for fn in (test_deterministic, test_context_independent, test_changes_with_weights):
        try:
            fn(); print(f"[OK ] {fn.__name__}")
        except Exception as e:
            ok = False; print(f"[XX ] {fn.__name__}: {e}")
    sys.exit(0 if ok else 1)
