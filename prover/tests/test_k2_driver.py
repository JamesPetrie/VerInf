"""The K2 driver (demo/demo_k2.py) end to end on the toy GGUF (GPU).

In the order the gate validates:
1. exact integer agreement: the engine pass equals the integer reference
   (prover/k2_int_reference.py) at every named intermediate, embedding to
   logits, at short and at long positions, every range inside its window;
2. the honest two-layer proof is accepted by the Rust verifier, bridged and
   not;
3. the negatives, each proved on a fresh tape against the honest enrollment
   (the same WeightCommitment and bridge enrollment, never re-enrolled) and
   checked under the honest weight root and enrollment identity, are
   rejected: a HeadInterleaveClaim slot swap, the rotary witness from the
   unscaled tables, expert 0's gate shard from expert 1's rows; the claims
   without YaRN are rejected under the honest statement digest and accepted
   under their own (the digest is what binds YaRN into the statement).
Float fidelity and routing differences are CPU checks
(tests/test_k2_int_reference.py, and demo_k2.py --mode fidelity on the real
GGUF). Plain test functions, for tests/run_tests.py."""
import pathlib
import sys
import tempfile

_HERE = pathlib.Path(__file__).resolve().parent
for _p in (_HERE.parent, _HERE, _HERE.parents[1] / "demo"):
    sys.path.insert(0, str(_p))

import _uint64_compat  # noqa: E402,F401
import core            # noqa: E402
import wc_bridge as wcb  # noqa: E402

import _k2_toy as toy  # noqa: E402
import demo_k2 as dk   # noqa: E402

CFG = core.LigeroConfig(ELL=16, K_DEG=16, N_LIG=64, T_QUERIES=4)
PARAMS = wcb.WcParams(B=48, lam=16, N_w=128, q_w=8)
PROMPT, CONT = [3, 17, 0, 41], [9, 22, 30, 5]
KK = toy.TOY["k"]
LONG = 130_072
_DIR = []


def _gguf():
    if not _DIR:
        _DIR.append(tempfile.TemporaryDirectory())
        toy.write(pathlib.Path(_DIR[0].name) / "toy.gguf")
    return str(pathlib.Path(_DIR[0].name) / "toy.gguf")


def _exact(offset):
    rec = dk.check(_gguf(), PROMPT, CONT, cfg=toy.TOY_MLA, layers=2, kk=KK, offset=offset,
                   ligero=CFG)
    bad = [r for r in rec["exact"] if not r["ok"]]
    assert not bad, bad[:3]
    names = {r["name"] for r in rec["exact"]}
    assert {"x0", "L0.query", "L0.key", "L0.sm", "L0.ffn", "L1.mask", "L1.gw", "L1.D",
            "L1.y", "L1.sh", "logits"} <= names
    assert all(r["ok"] for r in rec["ranges"].values()), \
        {n: r for n, r in rec["ranges"].items() if not r["ok"]}


def test_engine_pass_equals_the_integer_reference_at_position_0():
    _exact(0)


def test_engine_pass_equals_the_integer_reference_at_long_positions():
    _exact(LONG)


def test_unbridged_proof_accepts_and_witness_tampers_reject():
    core._CLAIM_MEM_ON = True
    try:
        rec = dk.research_prove(_gguf(), PROMPT, CONT, cfg=toy.TOY_MLA, layers=2, kk=KK,
                                offset=LONG, bridge=False,
                                negatives=("interleave", "yarn-witness"), ligero=CFG)
    finally:
        core._CLAIM_MEM_ON = False
    assert rec["verify"]["accept"], rec["verify"].get("tail")
    for neg, row in rec["negatives"].items():
        assert row["rejected"], (neg, row)
    mem = rec["claim_memory"]
    assert mem and all({"0", "1"} <= set(r["at_layer"]) for r in mem.values()), mem


def test_bridged_proof_accepts_and_negatives_reject_against_the_honest_enrollment():
    rec = dk.research_prove(_gguf(), PROMPT, CONT, cfg=toy.TOY_MLA, layers=2, kk=KK,
                            offset=LONG, bridge=True, negatives=dk.NEGATIVES,
                            ligero=CFG, wc_params=PARAMS)
    assert rec["verify"]["accept"], rec["verify"].get("tail")
    negs = rec["negatives"]
    assert set(negs) == set(dk.NEGATIVES)
    for neg, row in negs.items():
        assert row["rejected"], (neg, row)
    assert negs["yarn-statement"]["accept_under_own_digest"], negs["yarn-statement"]
    assert not negs["yarn-statement"]["same_statement"]
    assert negs["wrong-slice"]["by"] == "rust", negs["wrong-slice"]
