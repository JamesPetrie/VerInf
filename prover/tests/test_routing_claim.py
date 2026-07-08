"""RoutingClaim + FreivaldsCombineClaim: correctness + soundness.

POSITIVE: top-1 mask matches the argmax (incl. tie broken by lowest index,
matching torch.topk), r_chosen is the chosen raw logit, the combine
returns the chosen expert's stream, and prove + verify ACCEPTs.

SOUNDNESS (each must REJECT):
  * wrong expert        — one-hot at a non-argmax index → gap < 0 → the
                          word-decomposition / range LogUp of gap rejects
  * cardinality 0 / 2   — Σ m ≠ 1 → F2 rejects
  * non-boolean mask    — m = [2, −1, 0, 0] sums to 1 (cardinality passes)
                          but m(m−1) ≠ 0 → booleanity quad rejects
  * combine tamper      — committed wrong y → the Freivalds projection seam rejects

Run on the Spark:  ~/venv-hf/bin/python tests/test_routing_claim.py
"""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
import core
import claims as _C         # noqa: F401
import packets as _PK        # noqa: F401
from tape import Tape
from _rust_verify import rust_verify_tape
from routing_claim import route_top1, freivalds_combine

CFG = core.LigeroConfig(ELL=8, K_DEG=8, N_LIG=32, T_QUERIES=4)
SEED = b"routing-claim-test"
T, E, F = 2, 4, 3
B_LOGIT = 6                  # |r| < 64 in these tests
WORD_BITS = 4                # gap width = 6 + 2 → two 4-bit words
LOGITS = [[40, 20, 30, 10], [15, 45, 25, 5]]    # argmax experts: 0, 1
CHOSEN = [0, 1]
# Per-expert streams: expert e's stream is filled with (10·(e+1) + t).
XS = [[[10 * (e + 1) + t] * F for t in range(T)] for e in range(E)]


def _t(rows):
    flat = [v for row in rows for v in row]
    return torch.tensor(flat, dtype=torch.int64, device="cuda").to(torch.uint64)


def _build(logits, *, with_combine=False, force_expert=None, force_mask=None,
           force_y=None):
    # force_expert / force_mask now ride on TEST_TAMPER["m"] (the mask is
    # engine-derived; the tamper overrides it so constraints must reject).
    import routing_claim as rc
    if force_expert is not None:
        md = torch.zeros(T, E, dtype=torch.int64)
        md[torch.arange(T), torch.tensor(force_expert)] = 1
        rc.TEST_TAMPER["m"] = md.reshape(-1).cuda().to(torch.uint64)
    if force_mask is not None:
        rc.TEST_TAMPER["m"] = torch.tensor(list(force_mask), dtype=torch.int64,
                                            device="cuda").to(torch.uint64)
    core._COSET_POWERS_K_CACHE.clear()
    tape = Tape(CFG, lazy=True)
    r = tape.commit("r", _t(logits), (T, E))
    m, r_chosen, gap = route_top1(tape, r, T=T, E=E, B_logit=B_LOGIT,
                                   word_bits=WORD_BITS)
    y = None
    if with_combine or force_y is not None:
        xs = [tape.commit(f"x{e}", _t(XS[e]), (T, F)) for e in range(E)]
        y = freivalds_combine(tape, m, xs, T=T, E=E, F=F, force_y=force_y)
    return tape, m, r_chosen, gap, y


def _expect_reject(label, **kw):
    import routing_claim as rc
    try:
        tape, *_ = _build(LOGITS, **kw)
        acc, msg = rust_verify_tape(tape, tape.prove(seed=SEED), seed=SEED)
    finally:
        rc.TEST_TAMPER.clear()
    assert not acc, f"{label}: expected REJECT, got ACCEPT"
    print(f"    {label}: REJECT ok ({msg})")


def test_positive():
    tape, m, r_chosen, gap, y = _build(LOGITS, with_combine=True)
    live = tape.run_engine_pass()
    rch = live[r_chosen.var].to(torch.int64).cpu().tolist()
    assert rch == [LOGITS[t][CHOSEN[t]] for t in range(T)], f"r_chosen={rch}"
    m_w = live[m.var].to(torch.int64).view(T, E).cpu()
    assert all(m_w[t].argmax().item() == CHOSEN[t] and m_w[t].sum().item() == 1
               for t in range(T)), f"mask={m_w.tolist()}"
    y_w = live[y.var].to(torch.int64).view(T, F).cpu().tolist()
    exp_y = [XS[CHOSEN[t]][t] for t in range(T)]
    assert y_w == exp_y, f"y={y_w} want {exp_y}"

    tape2, *_ = _build(LOGITS, with_combine=True)
    acc, msg = rust_verify_tape(tape2, tape2.prove(seed=SEED), seed=SEED)
    assert acc, f"positive: expected ACCEPT ({msg})"
    print(f"    positive: ACCEPT ok, r_chosen={rch}, y={y_w}")


def test_tiebreak_lowest_index():
    # Row 0 all-equal → expert 0; row 1 ties at experts 1,2 → expert 1.
    logits = [[7, 7, 7, 7], [3, 9, 9, 1]]
    tape, m, r_chosen, gap, _ = _build(logits)
    live = tape.run_engine_pass()
    expect = [logits[t].index(max(logits[t])) for t in range(T)]   # torch.topk tie rule
    m_w = live[m.var].to(torch.int64).view(T, E).cpu()
    assert [m_w[t].argmax().item() for t in range(T)] == expect, m_w.tolist()
    tape2, *_ = _build(logits)
    acc, msg = rust_verify_tape(tape2, tape2.prove(seed=SEED), seed=SEED)
    assert acc, f"tiebreak: expected ACCEPT ({msg})"
    print(f"    tiebreak: ACCEPT ok, chosen={expect}")


def test_cheat_wrong_expert():
    _expect_reject("wrong expert (gap<0)", force_expert=[2, 2])


def test_cheat_cardinality_two():
    _expect_reject("two-hot mask", force_mask=[1, 1, 0, 0, 0, 1, 0, 0])


def test_cheat_cardinality_zero():
    _expect_reject("all-zero mask", force_mask=[0] * (T * E))


def test_cheat_nonboolean_mask():
    # [2, −1, 0, 0] sums to 1 in the field (cardinality passes); booleanity must
    # catch it. −1 ≡ P−1 enters via int64 −2^32 (the uint64 cast wraps mod 2^64,
    # and 2^64 − 2^32 = P − 1).
    neg1 = -(1 << 32)
    _expect_reject("non-boolean mask", force_mask=[2, neg1, 0, 0, 0, 1, 0, 0])


def test_guard_rejects_unsound_word_params():
    # Audit A1 finding 1: if the words can represent the field rep of a
    # negative gap (2^(n_words*word_bits) > P − 2^width), the wrong-mask
    # argument is vacuous — route_top1 must refuse to build at all.
    import routing_claim as rc
    core._COSET_POWERS_K_CACHE.clear()
    tape = Tape(CFG, lazy=True)
    r = tape.commit("r", _t(LOGITS), (T, E))
    try:
        rc.route_top1(tape, r, T=T, E=E, B_logit=60, word_bits=11)
        raise AssertionError("expected the soundness guard to raise")
    except AssertionError as e:
        assert "unsound parameterization" in str(e), e
    print("    guard: unsound (B_logit=60, w=11) refused ok")


def _expect_reject_tamper(label, key, tensor_vals):
    # Audit A1 finding 3: commit an inconsistent DERIVED value so each binding
    # family has a covering negative test (the honest builder satisfies them
    # all trivially, masking a hypothetically-removed constraint).
    import routing_claim as rc
    rc.TEST_TAMPER[key] = torch.tensor(tensor_vals, dtype=torch.int64,
                                        device="cuda").to(torch.uint64)
    try:
        _expect_reject(label, with_combine=True)
    finally:
        rc.TEST_TAMPER.clear()


def test_tamper_rt():        # F1 (tiebroken-logit pin)
    _expect_reject_tamper("tampered rt", "rt", [0] * (T * E))


def test_tamper_mrt():       # Q2 / F3
    _expect_reject_tamper("tampered mrt", "mrt", [0] * (T * E))


def test_tamper_rstar():     # F3 / F4
    _expect_reject_tamper("tampered rstar", "rstar", [999] * T)


def test_tamper_gap():       # F4 (gap pin; values in range, so only F4 fires)
    _expect_reject_tamper("tampered gap", "gap", [1] * (T * E))


def test_tamper_r_chosen():  # F5 (chosen-logit recovery)
    _expect_reject_tamper("tampered r_chosen", "r_chosen", [0] * T)


# ── FreivaldsCombineClaim (the §B.4 projected seam) ──

def test_fc_positive():
    tape, m, r_chosen, gap, y = _build(LOGITS, with_combine=True)
    live = tape.run_engine_pass()
    y_w = live[y.var].to(torch.int64).view(T, F).cpu().tolist()
    exp_y = [XS[CHOSEN[t]][t] for t in range(T)]
    assert y_w == exp_y, f"y={y_w} want {exp_y}"
    tape2, *_ = _build(LOGITS, with_combine=True)
    acc, msg = rust_verify_tape(tape2, tape2.prove(seed=SEED), seed=SEED)
    assert acc, f"fc positive: expected ACCEPT ({msg})"
    print(f"    fc positive: ACCEPT ok, y={y_w}")


def test_fc_wrong_y():
    wrong = [[99] * F for _ in range(T)]
    _expect_reject("fc wrong y (projection seam)",
                   force_y=[v for row in wrong for v in row])


def _fc_tamper(label, key, vals):
    import routing_claim as rc
    rc.TEST_TAMPER[key] = torch.tensor(vals, dtype=torch.int64,
                                        device="cuda").to(torch.uint64)
    try:
        _expect_reject(label, with_combine=True)
    finally:
        rc.TEST_TAMPER.clear()


def test_fc_tamper_m_em():   # C2 transpose pin
    _fc_tamper("fc tampered m_em", "m_em", [0] * (T * E))


def test_fc_tamper_s_em():   # C1 projection binding
    _fc_tamper("fc tampered s_em", "s_em", [3] * (T * E))


def test_fc_tamper_ms_em():  # quad / C3
    _fc_tamper("fc tampered ms_em", "ms_em", [5] * (T * E))


def test_fc_tamper_yr():     # C4 / C5 seam
    _fc_tamper("fc tampered yr", "yr", [9] * T)


def test_fc_fold_golden_roots():
    # The fold-consumer path (incremental expert-stream absorption) must be a
    # pure scheduling change: with LIGERO_NO_FOLD toggled, the SAME tape and
    # seed must produce bit-identical Merkle roots. Field ops are exact, so
    # any divergence is a fold bug, caught here at the commitment level.
    import os
    roots = {}
    for mode in ("fold", "nofold"):
        if mode == "nofold":
            os.environ["LIGERO_NO_FOLD"] = "1"
        else:
            os.environ.pop("LIGERO_NO_FOLD", None)
        try:
            tape, *_ = _build(LOGITS, with_combine=True)
            proof = tape.prove(seed=SEED)
            acc, msg = rust_verify_tape(tape, proof, seed=SEED)
            assert acc, f"{mode}: expected ACCEPT ({msg})"
            roots[mode] = (proof.root_p1.hex(), proof.root_p2.hex())
        finally:
            os.environ.pop("LIGERO_NO_FOLD", None)
    assert roots["fold"] == roots["nofold"], \
        f"fold changed commitments!\n  fold:   {roots['fold']}\n  nofold: {roots['nofold']}"
    print(f"    fold golden roots: identical ({roots['fold'][0][:16]}…)")


def main():
    fns = [test_positive, test_tiebreak_lowest_index, test_cheat_wrong_expert,
           test_cheat_cardinality_two, test_cheat_cardinality_zero,
           test_cheat_nonboolean_mask,
           test_guard_rejects_unsound_word_params,
           test_tamper_rt, test_tamper_mrt, test_tamper_rstar, test_tamper_gap,
           test_tamper_r_chosen,
           test_fc_positive, test_fc_wrong_y, test_fc_tamper_m_em,
           test_fc_tamper_s_em, test_fc_tamper_ms_em, test_fc_tamper_yr,
           test_fc_fold_golden_roots]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"[OK ] {fn.__name__}")
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[XX ] {fn.__name__}: {e}")
            failed += 1
    print(f"\n=== routing_claim: {len(fns) - failed}/{len(fns)} "
          f"{'PASS' if not failed else 'FAIL'} ===")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
