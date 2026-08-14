"""Gates for the project-before-sumcheck seam (projection.py).

Positive: linearity of the projection through the RS code (including the secret
padding), and an honest seam verifying at the opened columns.

Negative, one per way the seam could be broken:
  * projection built for a DIFFERENT rho than the transcript's,
  * a forged R_P that is a legitimate codeword of a tampered message -- the case
    the (K/N)^q bound is about; the disagreement count is measured, not assumed,
  * a value edited at an opened column without moving the root (Merkle),
  * committing R_P only AFTER the challenge (the §4 causal counterexample).

Run:  .venv/bin/python layergkr/tests/run_tests.py test_projection
"""
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from layergkr import projection as pj, rs
from layergkr.transcript import CausalityError, LAYER_SCHEDULE, Schedule, Transcript

# Rate K/N = 1/4, matching the doc's rho=4 IRS geometry, so the >= N-K bound
# below is the same statement as its (1/4)^q.
CFG = rs.Config(ELL=16, K_DEG=32, N_LIG=128, T_QUERIES=8)
N_IN, N_OUT, Q_COLS = 12, 6, 8


def _weights(seed=7):
    r = random.Random(seed)
    W = [[r.randrange(rs.P) for _ in range(N_IN)] for _ in range(N_OUT)]
    pad = [[r.randrange(rs.P) for _ in range(CFG.ELL - N_IN)] for _ in range(N_OUT)]
    return pj.PersistentWeights.enroll(CFG, W, pad)


def _fresh_transcript():
    return Transcript(schedule=Schedule(LAYER_SCHEDULE))


def _absorb_l1(t):
    for label in ("R_out", "R_lk", "R_sort", "R_mask"):
        t.absorb_root(label, blake_stub(label))
    t.coin("beta")
    t.absorb_root("R_cmp", blake_stub("R_cmp"))
    t.coin("alpha")


def blake_stub(label: str) -> bytes:
    return (label.encode() * 8)[:32].ljust(32, b"\x00")


# ── positive ─────────────────────────────────────────────────────────────────
def test_projection_is_linear_through_the_code():
    """F_P computed on codewords == encode(P computed on messages). This is the
    identity the whole seam rests on; padding is included on both sides."""
    pw = _weights()
    chi = [random.Random(1).randrange(rs.P) for _ in range(N_OUT)]
    assert pj.project_codeword(pw, chi) == rs.encode_row(
        CFG, pj.project_message(pw, chi)[0])


def test_honest_seam_verifies():
    pw = _weights()
    t = _fresh_transcript()
    _absorb_l1(t)
    chi, p_commit, opening, cols = pj.run_seam(CFG, pw, t, Q_COLS)
    ok, why = pj.verify_projection(CFG, pw.root, p_commit.root, chi, opening)
    assert ok, why
    assert len(set(cols)) == Q_COLS, "columns must be distinct"


def test_projected_message_matches_the_definition():
    """P[j] = sum_i chi_i W[i][j] on the model entries (padding aside)."""
    pw = _weights()
    chi = [random.Random(3).randrange(rs.P) for _ in range(N_OUT)]
    got = pj.project_message(pw, chi)[0]
    for j in range(N_IN):
        want = sum(chi[i] * pw.messages[i][j] for i in range(N_OUT)) % rs.P
        assert got[j] == want, f"slot {j}"


# ── negative ─────────────────────────────────────────────────────────────────
def test_projection_for_wrong_rho_is_rejected():
    pw = _weights()
    t = _fresh_transcript()
    _absorb_l1(t)
    chi, p_commit, opening, cols = pj.run_seam(CFG, pw, t, Q_COLS)
    wrong = list(chi)
    wrong[0] = (wrong[0] + 1) % rs.P
    ok, why = pj.verify_projection(CFG, pw.root, p_commit.root, wrong, opening)
    assert not ok and "projection equality" in why, why


def test_forged_codeword_disagrees_in_at_least_N_minus_K_columns():
    """The (K/N)^q bound, measured. A cheating prover commits a VALID codeword of
    a tampered message; the difference is a nonzero codeword of degree < K, so it
    must be nonzero in at least N - K positions."""
    pw = _weights()
    chi = [random.Random(5).randrange(rs.P) for _ in range(N_OUT)]
    tampered = pj.project_message(pw, chi)[0]
    tampered[0] = (tampered[0] + 12345) % rs.P
    forged = rs.encode_row(CFG, tampered)
    disagreements = pj.count_column_disagreements(pw, chi, forged)
    assert disagreements >= CFG.N_LIG - CFG.K_DEG, (
        f"{disagreements} < N-K = {CFG.N_LIG - CFG.K_DEG}")
    # and therefore the seam rejects it at the opened columns
    forged_commit = rs.Commit(CFG, [forged])
    opening = pj.open_projection(pw, forged_commit, list(range(Q_COLS)))
    ok, why = pj.verify_projection(CFG, pw.root, forged_commit.root, chi, opening)
    assert not ok and "projection equality" in why, why


def test_edited_opened_value_fails_merkle():
    pw = _weights()
    t = _fresh_transcript()
    _absorb_l1(t)
    chi, p_commit, opening, cols = pj.run_seam(CFG, pw, t, Q_COLS)
    opening.p_values[0][0] = (opening.p_values[0][0] + 1) % rs.P
    ok, why = pj.verify_projection(CFG, pw.root, p_commit.root, chi, opening)
    assert not ok and "merkle" in why, why


def test_contraction_coin_before_R_P_is_refused():
    """The §4 causal counterexample: a prover that wants to choose the projection
    once the contraction challenge is known. The schedule refuses one step
    earlier than the cheat itself -- it will not hand out the contraction coin
    while R_P is still uncommitted, so the cheating state is unreachable."""
    pw = _weights()
    t = _fresh_transcript()
    _absorb_l1(t)
    t.coin("rho", N_OUT)
    raised = ""
    try:
        t.coin("contraction", 4)      # jumping the gun: R_P not absorbed yet
    except CausalityError as e:
        raised = str(e)
    assert "expected absorb:R_P" in raised, f"schedule allowed it: {raised!r}"


def test_late_R_P_breaks_fiat_shamir_even_without_the_schedule():
    """Belt and braces: strip the schedule guard and the ordering is still
    enforced cryptographically. An honest transcript absorbs R_P before the
    contraction coin; a cheat absorbs it after. The contraction challenges
    differ, so the cheat's proof is checked against coins it never used."""
    honest, cheat = Transcript(), Transcript()
    for t in (honest, cheat):
        t.absorb_root("R_out", blake_stub("R_out"))
        t.coin("rho", N_OUT)
    honest.absorb_root("R_P", b"\x22" * 32)
    c_honest = honest.coin("contraction", 4)
    c_cheat = cheat.coin("contraction", 4)
    cheat.absorb_root("R_P", b"\x22" * 32)
    assert c_honest != c_cheat, "R_P placement did not affect the contraction coin"


def test_reordered_transcript_changes_the_coins():
    """Even without the schedule guard, Fiat-Shamir catches reordering: the same
    absorbs in a different order yield different challenges."""
    a, b = Transcript(), Transcript()
    a.absorb_root("R_out", b"\x01" * 32)
    a.absorb_root("R_P", b"\x02" * 32)
    b.absorb_root("R_P", b"\x02" * 32)
    b.absorb_root("R_out", b"\x01" * 32)
    assert a.coin("rho", 3) != b.coin("rho", 3)


if __name__ == "__main__":
    sys.exit(__import__("run_tests").__name__ and 0)


# ── multi-row layout (n_in > ELL) ────────────────────────────────────────────
def _wide_weights(n_in, n_out, seed=21):
    r = random.Random(seed)
    W = [[r.randrange(rs.P) for _ in range(n_in)] for _ in range(n_out)]
    return pj.PersistentWeights.enroll(CFG, W)


def test_multi_row_layout_blocks_the_contraction():
    """A contraction wider than ELL spans ceil(n_in/ELL) rows per output
    coordinate -- required at production geometry (Maverick's FFN contracts over
    16384 against ELL=8192) and the ceil() factor in the theorem's N_pad."""
    n_in = CFG.ELL * 2 + 5
    pw = _wide_weights(n_in, 4)
    assert pw.n_blocks == 3, pw.n_blocks
    assert pw.commit.n_rows == 3 * 4, pw.commit.n_rows


def test_multi_row_projection_matches_the_definition():
    n_in = CFG.ELL * 2 + 5
    pw = _wide_weights(n_in, 4)
    chi = [random.Random(2).randrange(rs.P) for _ in range(4)]
    flat = pj.flatten_projection(pw, pj.project_message(pw, chi))
    assert len(flat) == n_in
    for b in range(pw.n_blocks):
        lo, hi = pw.block_slice(b)
        for j in range(lo, hi):
            want = sum(chi[i] * pw.messages[b * pw.n_out + i][j - lo]
                       for i in range(4)) % rs.P
            assert flat[j] == want, f"slot {j}"


def test_multi_row_seam_verifies_and_rejects_a_tampered_block():
    n_in = CFG.ELL * 2 + 5
    pw = _wide_weights(n_in, 4)
    chi = [random.Random(6).randrange(rs.P) for _ in range(4)]
    pc = pj.commit_projection(pw, chi)
    opening = pj.open_projection(pw, pc, list(range(Q_COLS)))
    ok, why = pj.verify_projection(CFG, pw.root, pc.root, chi, opening)
    assert ok, why
    # break only the SECOND block: the per-block check must catch it
    opening.p_values[0][1] = (opening.p_values[0][1] + 1) % rs.P
    ok, why = pj.verify_projection(CFG, pw.root, pc.root, chi, opening)
    assert not ok and "merkle" in why, why


def test_multi_row_padding_is_sized_by_the_block_tail():
    """Regression: the tail exists only on the LAST block and is (-n_in) % ELL.
    Sizing it as ELL - n_in goes negative once n_in > ELL, which silently produced
    empty padding rows and blew up at enrolment."""
    n_in = CFG.ELL * 2 + 5
    tail = (-n_in) % CFG.ELL
    assert tail == CFG.ELL - 5
    r = random.Random(31)
    W = [[r.randrange(rs.P) for _ in range(n_in)] for _ in range(3)]
    pad = [[r.randrange(rs.P) for _ in range(tail)] for _ in range(3)]
    pw = pj.PersistentWeights.enroll(CFG, W, pad)
    assert pw.n_blocks == 3
    # full blocks carry no padding; only the last one does
    for b in range(2):
        for i in range(3):
            assert pw.messages[b * 3 + i] == W[i][b * CFG.ELL:(b + 1) * CFG.ELL]
    assert pw.messages[2 * 3 + 0][5:] == pad[0]
    # padding shorter than the tail must be refused, not silently zero-filled
    raised = False
    try:
        pj.PersistentWeights.enroll(CFG, W, [[1, 2] for _ in range(3)])
    except ValueError as e:
        raised = "block tail needs" in str(e)
    assert raised
