"""Top-k routing on the toy tape (E = 8, k = 3): honest proofs the Rust verifier
ACCEPTs, and targeted alternative witnesses it must REJECT
(analysis/topk-routing-design.md §3.1–3.3; prover/topk_routing.py).

Selection (TopkRoutingClaim + the range on v):
  honest                  ties broken by the lowest index, a signed bias
  wrong selection         one selected expert swapped for an unselected one
  the review's counterexample  all-zero scores, {5, 6, 7} with τ = (P − 1)/2 + 3
  guard bypassed          R = 63 at width 43 built past the Python guard: the
                          verifier's own guard refuses the statement
  cardinality k ± 1       F2
  non-boolean mask        Q1
  q̃ / md tamper           F1 / Q2 and Fv
Slots and gate weights (TopkSlotsClaim, GateBracketClaim):
  honest                  slot scores and ⌊C·s/Z⌋ equal the integer reference
  two slots on one expert, a slot on an unselected expert   Sb
  slot score tamper       Sv
  w + 1 with rem − Z      the range on rem
  w − 1 with rem + Z      the range on Z − 1 − rem
One-word ranges (N = 1) for routing and the bracket: honest ACCEPTs.
The stacked chain (topk_moe_ffn): honest, its whole output against an
independent integer reference (a SiLU table at the activation scale, both
branches, signed values, nonzero outputs), and tampers of the stacked
concatenations, a split slice and the combine (through its folded path).

Each alternative witness is written in full (every derived value consistent
with the cheat), so the relation that must catch it is the one that fires;
every tamper is counted and must have been applied, so a hook the prover
bypasses fails the test instead of passing it.
Needs a card. Run:  python prover/tests/run_tests.py test_topk_routing"""
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import torch

import core
import claims as _C        # noqa: F401
import packets as _PK      # noqa: F401
import topk_params as tp
import topk_reference as ref
import topk_routing as tr
from _rust_verify import rust_verify_tape
from claims import SiluConfig
from compute_fns import WITNESS_TAMPER
from tape import Tape

P = tp.P
CFG = core.LigeroConfig(ELL=16, K_DEG=16, N_LIG=64, T_QUERIES=4)
SEED = b"topk-routing"
T, E, K = 2, 8, 3
SCORE_BITS = 8
WIDTH, WORD_BITS = 13, 7                 # |q̃ − q̃'| < 2^13: s < 2^8, |b| < 2^7, L = 3
C = round(2.827 * (1 << 12))             # K2's routed scale, gate weights at 2^12
S = 1 << 4
# token 0: s + b ties experts 6 and 7 at 125 for third place, and the lowest
# index must win; the bias is signed
SCORES = [[5, 200, 37, 37, 200, 9, 120, 125], [250, 3, 77, 190, 12, 190, 45, 101]]
BIAS = [0, 3, -4, 6, 0, -2, 5, 0]


def _u64(vals):
    return torch.from_numpy(np.array([v % P for v in vals], dtype=np.uint64)).to("cuda")


def _override(cells):
    def fn(t):
        t = t.clone().contiguous()
        signed = t.view(torch.int64)
        for i, v in cells.items():
            v %= P
            signed[i] = v - (1 << 64) if v >= (1 << 63) else v
        return t
    return fn


def _plus_one(i):
    """Add one to slot i of whatever the claim computed."""
    return lambda t: _override({i: int(t.view(torch.int64)[i].item()) + 1})(t)


_HITS = {}


def _tamper_fn(claim_name, field, fn):
    """Register a witness tamper that counts its own applications."""
    key = (claim_name, field)

    def counted(t):
        _HITS[key] = _HITS.get(key, 0) + 1
        return fn(t)
    WITNESS_TAMPER[key] = counted


def _tamper(claim_name, fields):
    WITNESS_TAMPER.clear()
    _HITS.clear()
    for field, cells in fields.items():
        _tamper_fn(claim_name, field, _override(cells))


def _verdict(tape):
    """Prove and verify; every registered tamper must have been applied."""
    try:
        proof = tape.prove(seed=SEED)
        result = rust_verify_tape(tape, proof, seed=SEED)
        unapplied = [k for k in WITNESS_TAMPER if not _HITS.get(k)]
    finally:
        WITNESS_TAMPER.clear()
        _HITS.clear()
    assert not unapplied, f"tamper(s) never applied, the test proves nothing: {unapplied}"
    return result


def _flat(rows):
    return {i: v for i, v in enumerate(x for r in rows for x in r)}


def _selection_tape(scores=SCORES, bias=BIAS, width=WIDTH, word_bits=WORD_BITS):
    core._COSET_POWERS_K_CACHE.clear()
    tape = Tape(CFG, lazy=True)
    s = tape.commit("s", _u64([x for r in scores for x in r]), (T, E))
    b = tape.commit("b", _u64(bias), (E,))
    m, tau = tr.route_topk(tape, s, b, T=T, E=E, k=K, width=width, word_bits=word_bits)
    return tape, s, m, tau


def _cheat_fields(scores, bias, sel_rows, tau_rows):
    """Every TopkRoutingClaim field for a mask selecting sel_rows[t] with
    threshold tau_rows[t], the derived values consistent with it."""
    qt = [ref.tiebroken([scores[t][e] + bias[e] for e in range(E)], E) for t in range(T)]
    m = [[1 if e in sel_rows[t] else 0 for e in range(E)] for t in range(T)]
    d = [[(qt[t][e] - tau_rows[t]) % P for e in range(E)] for t in range(T)]
    md = [[(m[t][e] * d[t][e]) % P for e in range(E)] for t in range(T)]
    v = [[(2 * md[t][e] - d[t][e]) % P for e in range(E)] for t in range(T)]
    return {"m": _flat(m), "tau": {t: tau_rows[t] % P for t in range(T)},
            "d": _flat(d), "md": _flat(md), "v": _flat(v)}


# ---- selection -----------------------------------------------------------

def test_selection_honest_accepts():
    tape, _, m, _ = _selection_tape()
    live = tape.run_engine_pass()
    got = live[m.var].view(torch.int64).view(T, E).cpu().tolist()
    want = ref.routing_witness(SCORES, BIAS, K)["m"]
    assert got == want, (got, want)
    assert [e for e in range(E) if got[0][e]] == [1, 4, 6], got[0]   # 6 wins the tie with 7
    acc, msg = _verdict(_selection_tape()[0])
    assert acc, f"honest top-k selection: expected ACCEPT ({msg})"
    print("    honest top-3 of 8 (a tie for third, signed bias): ACCEPT")


def test_wrong_selection_rejects():
    w = ref.routing_witness(SCORES, BIAS, K)
    sel = [set(e for e in range(E) if w["m"][t][e]) for t in range(T)]
    sel[1] = (sel[1] - {3}) | {2}                         # 190 out, 77 in
    qt1 = ref.tiebroken([SCORES[1][e] + BIAS[e] for e in range(E)], E)
    taus = [w["tau"][0], min(qt1[e] for e in sel[1])]
    tape = _selection_tape()[0]
    _tamper("TopkRoutingClaim", _cheat_fields(SCORES, BIAS, sel, taus))
    acc, msg = _verdict(tape)
    assert not acc, "a non-top-k selection accepted"
    print(f"    wrong selection: REJECT ({msg.splitlines()[-1]})")


def test_the_review_counterexample_rejects():
    zero, nob = [[0] * E for _ in range(T)], [0] * E
    tau = (P - 1) // 2 + 3
    tape = _selection_tape(zero, nob, width=3 + tp.l_bits(E), word_bits=4)[0]
    _tamper("TopkRoutingClaim", _cheat_fields(zero, nob, [{5, 6, 7}] * T, [tau] * T))
    acc, msg = _verdict(tape)
    assert not acc, "the review's counterexample accepted under the guard"
    print(f"    counterexample {{5,6,7}}, tau = (P-1)/2 + 3: REJECT ({msg.splitlines()[-1]})")


def test_the_builder_refuses_the_counterexample_parameters():
    try:
        _selection_tape(width=40 + tp.l_bits(E), word_bits=21)
    except ValueError as e:
        assert "unsound top-k" in str(e)
        print("    B_q = 40 with three 21-bit words: refused by the builder")
        return
    raise AssertionError("route_topk accepted R = 63 at width 43")


def test_the_verifier_refuses_the_counterexample_parameters():
    """Built past the Python guard (seven 9-bit words, R = 63, width 43), the
    statement must still be refused: the verifier's compile asserts its own
    threshold guard."""
    real = tr.threshold_words
    tr.threshold_words = lambda width, word_bits: (7, 63)
    try:
        tape = _selection_tape(width=43, word_bits=9)[0]
    finally:
        tr.threshold_words = real
    acc, msg = _verdict(tape)
    assert not acc, "the verifier accepted a statement past the threshold guard"
    assert "threshold guard" in msg, msg[-400:]
    print("    R = 63 at width 43 past the Python guard: the verifier refuses it")


def _honest_fields():
    w = ref.routing_witness(SCORES, BIAS, K)
    return w, [set(e for e in range(E) if w["m"][t][e]) for t in range(T)]


def test_cardinality_k_plus_one_rejects():
    w, sel = _honest_fields()
    qt0 = ref.tiebroken([SCORES[0][e] + BIAS[e] for e in range(E)], E)
    extra = max((e for e in range(E) if e not in sel[0]), key=lambda e: qt0[e])
    sel[0] = sel[0] | {extra}
    taus = [qt0[extra], w["tau"][1]]                      # every range still holds
    tape = _selection_tape()[0]
    _tamper("TopkRoutingClaim", _cheat_fields(SCORES, BIAS, sel, taus))
    acc, _ = _verdict(tape)
    assert not acc, "k + 1 experts accepted"
    print("    cardinality k + 1 (ranges consistent): REJECT")


def test_cardinality_k_minus_one_rejects():
    w, sel = _honest_fields()
    qt0 = ref.tiebroken([SCORES[0][e] + BIAS[e] for e in range(E)], E)
    drop = min(sel[0], key=lambda e: qt0[e])
    sel[0] = sel[0] - {drop}
    taus = [min(qt0[e] for e in sel[0]), w["tau"][1]]
    tape = _selection_tape()[0]
    _tamper("TopkRoutingClaim", _cheat_fields(SCORES, BIAS, sel, taus))
    acc, _ = _verdict(tape)
    assert not acc, "k − 1 experts accepted"
    print("    cardinality k - 1 (ranges consistent): REJECT")


def test_non_boolean_mask_rejects():
    w, sel = _honest_fields()
    f = _cheat_fields(SCORES, BIAS, sel, w["tau"])
    a, b2 = sorted(sel[0])[:2]
    f["m"][a], f["m"][b2] = 2, 0                          # Σ m still k
    for e in (a, b2):
        f["md"][e] = (f["m"][e] * f["d"][e]) % P
        f["v"][e] = (2 * f["md"][e] - f["d"][e]) % P
    tape = _selection_tape()[0]
    _tamper("TopkRoutingClaim", f)
    acc, _ = _verdict(tape)
    assert not acc, "a non-boolean mask accepted"
    print("    non-boolean mask (sum still k): REJECT")


def test_tamper_qt_and_md_reject():
    for field, cell in (("qt", 3), ("md", 1)):
        tape = _selection_tape()[0]
        _tamper_fn("TopkRoutingClaim", field, _plus_one(cell))
        acc, _ = _verdict(tape)
        assert not acc, f"a tampered {field} accepted"
        print(f"    tampered {field}: REJECT")


# ---- slots and the gate bracket ---------------------------------------------

def _bracket_tape():
    tape, s, m, _ = _selection_tape()
    M, ss = tr.topk_slots(tape, m, s, T=T, E=E, k=K)
    gw = tr.gate_bracket(tape, ss, T=T, k=K, C=C, score_bits=SCORE_BITS)
    return tape, M, ss, gw


def _ref_slots():
    m = ref.routing_witness(SCORES, BIAS, K)["m"]
    M, ss = ref.slot_witness(m, SCORES, K)
    return m, M, ss


def test_slots_and_bracket_honest_accept():
    tape, M, ss, gw = _bracket_tape()
    live = tape.run_engine_pass()
    _, Mr, ssr = _ref_slots()
    assert live[ss.var].view(torch.int64).view(T, K).cpu().tolist() == ssr
    for i in range(K):
        assert live[M[i].var].view(torch.int64).view(T, E).cpu().tolist() == Mr[i]
    want = ref.bracket_witness(ssr, C)["w"]
    assert live[gw.var].view(torch.int64).view(T, K).cpu().tolist() == want
    acc, msg = _verdict(_bracket_tape()[0])
    assert acc, f"honest slots and gate weights: expected ACCEPT ({msg})"
    print(f"    slot scores {ssr}, gate weights {want}: ACCEPT")


def _slot_cheat(Mr, s):
    f = {}
    for i in range(K):
        f[f"M[{i}]"] = _flat(Mr[i])
        f[f"MS[{i}]"] = _flat([[Mr[i][t][e] * s[t][e] for e in range(E)] for t in range(T)])
    f["ss"] = _flat([[sum(Mr[i][t][e] * s[t][e] for e in range(E)) for i in range(K)]
                     for t in range(T)])
    return f


def test_two_slots_on_one_expert_reject():
    m, Mr, _ = _ref_slots()
    e0 = Mr[0][0].index(1)
    Mr[1][0] = [1 if e == e0 else 0 for e in range(E)]
    tape = _bracket_tape()[0]
    _tamper("TopkSlotsClaim", _slot_cheat(Mr, SCORES))
    acc, _ = _verdict(tape)
    assert not acc, "two slots on one expert accepted"
    print("    two slots on one expert: REJECT")


def test_a_slot_on_an_unselected_expert_rejects():
    m, Mr, _ = _ref_slots()
    eu = next(e for e in range(E) if not m[0][e])
    Mr[2][0] = [1 if e == eu else 0 for e in range(E)]
    tape = _bracket_tape()[0]
    _tamper("TopkSlotsClaim", _slot_cheat(Mr, SCORES))
    acc, _ = _verdict(tape)
    assert not acc, "a slot on an unselected expert accepted"
    print("    a slot on an unselected expert: REJECT")


def test_slot_score_tamper_rejects():
    tape = _bracket_tape()[0]
    _, _, ssr = _ref_slots()
    _tamper("TopkSlotsClaim", {"ss": {0: ssr[0][0] + 1}})
    acc, _ = _verdict(tape)
    assert not acc, "a tampered slot score accepted"
    print("    tampered slot score: REJECT")


def _bracket_shift(delta):
    """w ± 1 at cell 0 with rem ∓ Z, every other bracket value consistent."""
    _, _, ssr = _ref_slots()
    br = ref.bracket_witness(ssr, C)
    Z = br["Z"][0]
    w = [list(r) for r in br["w"]]
    rem = [list(r) for r in br["rem"]]
    w[0][0] += delta
    rem[0][0] -= delta * Z
    gr = [[br["Z"][t] - 1 - rem[t][i] for i in range(K)] for t in range(T)]
    wZ = [[w[t][i] * br["Z"][t] for i in range(K)] for t in range(T)]
    return {"w": _flat(w), "rem": _flat(rem), "gr": _flat(gr), "wZ": _flat(wZ)}


def test_a_quotient_one_too_high_rejects():
    tape = _bracket_tape()[0]
    _tamper("GateBracketClaim", _bracket_shift(+1))
    acc, _ = _verdict(tape)
    assert not acc, "w + 1 with a negative remainder accepted"
    print("    w + 1, rem - Z: REJECT")


def test_a_quotient_one_too_low_rejects():
    tape = _bracket_tape()[0]
    _tamper("GateBracketClaim", _bracket_shift(-1))
    acc, _ = _verdict(tape)
    assert not acc, "w - 1 with rem >= Z accepted"
    print("    w - 1, rem + Z: REJECT")


# ---- one-word ranges ---------------------------------------------------------

def _word_counts(tape):
    from claims import WordExtractionClaim
    return [len(c.words) for c in tape.claims if isinstance(c, WordExtractionClaim)]


def test_one_word_selection_range_accepts():
    """width 13 in one 16-bit word: the extraction has no stride to infer the
    word width from, and must still give the honest word."""
    tape = _selection_tape(word_bits=16)[0]
    assert _word_counts(tape) == [1]
    acc, msg = _verdict(tape)
    assert acc, f"one-word selection range: expected ACCEPT ({msg})"
    print("    selection with a one-word range: ACCEPT")


def test_one_word_bracket_ranges_accept():
    tape, s, m, _ = _selection_tape()
    _, ss = tr.topk_slots(tape, m, s, T=T, E=E, k=K)
    tr.gate_bracket(tape, ss, T=T, k=K, C=C, score_bits=SCORE_BITS, word_bits=16)
    assert _word_counts(tape)[1:] == [1, 1, 1]          # rem, gr, w
    acc, msg = _verdict(tape)
    assert acc, f"one-word bracket ranges: expected ACCEPT ({msg})"
    print("    gate bracket with one-word ranges: ACCEPT")


# ---- the stacked chain --------------------------------------------------------

D_MODEL, D_FF = 4, 4
# SiLU tables at the activation scale S = 2^4: bins of 2, 32 entries below
# |x| = 64 (real 4.0), saturation above; the tiles cover |x| < 4096
SILU_S16 = SiluConfig(b=2, T_LEN=32, b_2=64, b_3=128, b_4=256,
                      width_2=1, width_3=1, width_4=4, r=4)


def _silu_real(x):
    if x >= 0:
        return x / (1.0 + math.exp(-x))
    e = math.exp(x)
    return x * e / (1.0 + e)


def _silu_ref(x, cfg=SILU_S16):
    """The SiLU claim's function, from its specification: below b_2 the
    table value at the centre of the bin |x| // b, the sign choosing the
    branch; at and above b_2, max(x, 0)."""
    mag = abs(x)
    if mag >= cfg.b_2:
        return x if x > 0 else 0
    c = ((mag // cfg.b) * cfg.b + cfg.b // 2) / (1 << cfg.r)
    return int(round(_silu_real(c if x >= 0 else -c) * (1 << cfg.r)))


def _ffn_fixture():
    g = torch.Generator().manual_seed(7)
    X = torch.randint(-48, 48, (T, D_MODEL), generator=g).tolist()
    W = {n: torch.randint(-16, 16, (E,) + shape, generator=g).tolist()
         for n, shape in (("gate", (D_MODEL, D_FF)), ("up", (D_MODEL, D_FF)),
                          ("down", (D_FF, D_MODEL)))}
    return X, W


def _ffn_ref(X, W):
    """The whole FFN over integers, independent of the claims: top-k on
    s + b, slots by ascending expert, w = ⌊C·s/Σs⌋, per slot the expert's
    gate and up matmuls floored to S, SiLU, the floored Hadamard, the raw
    down projection, then the weighted sum floored by S_w·S."""
    m = ref.routing_witness(SCORES, BIAS, K)["m"]
    M, ss = ref.slot_witness(m, SCORES, K)
    gw = ref.bracket_witness(ss, C)["w"]
    y = []
    for t in range(T):
        acc = [0] * D_MODEL
        for i in range(K):
            e = M[i][t].index(1)
            g = [sum(X[t][a] * W["gate"][e][a][j] for a in range(D_MODEL)) >> 4
                 for j in range(D_FF)]
            u = [sum(X[t][a] * W["up"][e][a][j] for a in range(D_MODEL)) >> 4
                 for j in range(D_FF)]
            h = [(_silu_ref(g[j]) * u[j]) >> 4 for j in range(D_FF)]
            for c in range(D_MODEL):
                acc[c] += gw[t][i] * sum(h[j] * W["down"][e][j][c] for j in range(D_FF))
        y.append([v >> 16 for v in acc])
    return y


def _ffn_tape():
    core._COSET_POWERS_K_CACHE.clear()
    X, W = _ffn_fixture()
    tape = Tape(CFG, silu_config=SILU_S16, lazy=True)
    x = tape.commit("x", _u64([v for r in X for v in r]), (T, D_MODEL))
    s = tape.commit("s", _u64([v for r in SCORES for v in r]), (T, E))
    b = tape.commit("b", _u64(BIAS), (E,))
    w = {n: [tape.commit(f"{n}{e}", _u64([v for r in W[n][e] for v in r]),
                         (len(W[n][e]), len(W[n][e][0]))) for e in range(E)]
         for n in ("gate", "up", "down")}
    y = tr.topk_moe_ffn(tape, x, s, b, w, T=T, E=E, k=K, d=D_MODEL, d_ff=D_FF, S=S,
                        S_w=1 << 12, C=C, score_bits=SCORE_BITS, width=WIDTH,
                        output_width=16, select_word_bits=WORD_BITS)
    return tape, y


def _claim(tape, cls):
    return [c for c in tape.claims if isinstance(c, cls)]


def test_the_ffn_refuses_a_silu_table_at_another_scale():
    tape = Tape(CFG, lazy=True)                       # SILU_TOY: scale 1, not S
    try:
        tr.topk_moe_ffn(tape, *[None] * 4, T=T, E=E, k=K, d=D_MODEL, d_ff=D_FF, S=S,
                        S_w=1 << 12, C=C, score_bits=SCORE_BITS, width=WIDTH,
                        output_width=16)
    except AssertionError as e:
        assert "SiLU tables" in str(e)
        print("    a SiLU table at scale 1 under activations at 16: refused")
        return
    raise AssertionError("topk_moe_ffn accepted a SiLU table at the wrong scale")


def test_stacked_chain_honest_accepts():
    X, W = _ffn_fixture()
    want = _ffn_ref(X, W)
    assert any(v for r in want for v in r), "the reference output is all zero"
    tape, y = _ffn_tape()
    live = tape.run_engine_pass()
    field = [[v % (1 << 64) for v in r]                  # int64 bits -> the field element
             for r in live[y.var].view(torch.int64).view(T, D_MODEL).cpu().tolist()]
    got = [[v - P if v > P // 2 else v for v in r] for r in field]
    assert got == want, (got, want)
    acc, msg = _verdict(_ffn_tape()[0])
    assert acc, f"honest stacked top-k FFN: expected ACCEPT ({msg})"
    print(f"    stacked top-3 FFN == independent reference {want}: ACCEPT")


def test_stacked_chain_tampers_reject():
    from claims import ConcatClaim
    for label, cls, field in (("the stacked concatenations", ConcatClaim, "dst"),
                              ("a split slice", tr.SplitClaim, "parts[1]")):
        tape, _ = _ffn_tape()
        _tamper_fn(cls.__name__, field, _plus_one(0))
        acc, _ = _verdict(tape)
        assert not acc, f"a tampered {label} accepted"
        print(f"    tampered {label}: REJECT")


class _CountingTamper(dict):
    """routing_claim.TEST_TAMPER, counting reads: the combine's folded
    finalizer applies this hook, not WITNESS_TAMPER."""
    hits = 0

    def __getitem__(self, key):
        self.hits += 1
        return super().__getitem__(key)


def test_stacked_combine_tamper_rejects():
    import routing_claim as rc
    from routing_claim import FreivaldsCombineClaim
    tape, _ = _ffn_tape()
    comb = _claim(tape, FreivaldsCombineClaim)[0]
    bad = tape.run_engine_pass()[comb.y].contiguous().view(-1).clone()
    bad.view(torch.int64)[0] += 1
    hook = _CountingTamper(y=bad)
    real, rc.TEST_TAMPER = rc.TEST_TAMPER, hook
    try:
        acc, _ = _verdict(_ffn_tape()[0])
    finally:
        rc.TEST_TAMPER = real
    assert hook.hits > 0, "the combine tamper never reached the folded path"
    assert not acc, "a tampered combine output accepted"
    print(f"    tampered combine output (folded path, {hook.hits} applications): REJECT")
