# Prover degrees-of-freedom review (claim-by-claim, from the verifier's side)

**Date:** 2026-07-04. A systematic pass over `verifier/src/handlers.rs` — the
authoritative constraint compile — asking, per claim type: given the claim's
inputs, do the constraints admit exactly one satisfying assignment of its own
variables (§3.3's no-degrees-of-freedom requirement)? Findings are grouped by
severity. Summary: **no exploitable freedom found in the constraint logic; one
gadget (the RMSNorm rsqrt bracket) needs either a written wraparound-exclusion
argument or a cheap range pin; a handful of benign free slots should be
documented or canonicalized.**

## 1. Sound by randomness, not by uniqueness (by design, accounted)

These do not pin the witness absolutely; they pin it up to the soundness error
of §5.4, with the commit-before-challenge order doing the work:

- **Freivalds products** (MatmulClaim's C; RMSNorm's broadcast; the
  FreivaldsCombine seam): any C′ ≠ AB survives with probability ≤ 2/|F| over
  (ρ, λ), drawn after C′ is committed.
- **LogUp multiplicities and lookups**: binding with error (M+T+1)/|F| over
  (α, β), drawn after the tables' query sides are committed.
- **The blinding rows** (BLIND_IRS/LIN/QUAD): committed rows the prover chooses
  freely — but committed before the combiner challenges, so using them to
  absorb a wrong witness would require predicting r_lin; the honest
  construction zeroes their ζ-sums, an adversary gains nothing post-commit.

## 2. Verified clean (uniqueness holds; the arguments)

- **Add / Hadamard / RoPE / Concat / rescale blocks**: outputs are linear or
  quadratic images of inputs; the rescale word decompositions are unique
  because BOTH parts are range-checked and the composed range 2^(r+w) fits the
  field — the injectivity of bounded mixed-radix decomposition.
- **Softmax shift**: the two-table bracket pins c even across plateaus — on a
  flat stretch s1(c0) = … = s1(c1) = s_y, only the smallest c satisfies
  s2(c) = s1(c−δ) ≥ s_y + 1, so rounding plateaus do not create freedom. The
  lookup's own range constraint forces z ≥ 0. Operands (s1, s2) are sums of
  lookup outputs, hence integer-bounded — the bracket has honest integer
  semantics.
- **SiLU sign split**: the magnitude's word decomposition bounds it below
  ⌈P/2⌉, which forces the sign bit for every x ≠ 0 (the wrong sign would need
  a magnitude > P/2).
- **Routing argmax**: the public index tiebreak makes the argmax unique; a
  non-maximal one-hot makes some gap a near-P field element that cannot be
  recomposed from range-checked words.
- **InfoFinalize ceiling division**: `range_k` is a table of length exactly k,
  so rem ∈ [0, k) exactly and z_o = ⌈gap_o2 / k⌉ is the unique solution of
  k·z_o = gap_o2 + rem — no slack to shift z_o (which would directly deflate
  the reported surprisal; this was the highest-stakes check in the review).
- **MaxClaim output select**: tok_t = Σ_i i·O_i with O boolean and Σ O = 1
  pins O's support to exactly the committed token id, so gap_o is evaluated at
  the committed output token, not a prover-chosen one. (Binding tok to the
  *real-world* transcript remains the known Appendix-E gap — an anchoring
  limitation, not a claim-level freedom.)

## 3. Flagged: the RMSNorm rsqrt bracket needs a wrap-exclusion argument

**RESOLVED 2026-07-06 — and the flag understated the severity.** The
production configs used slack windows of 4×16 bits = 2^64 ≥ P, so EVERY
field element decomposed and the bracket was fully vacuous: any forged y′
was accepted, no search needed (the ~2⁻¹⁵ estimate below assumed W ≈ 2²⁴).
The recommended lone range check on y is also insufficient — the honest
slack needs a window (~2⁵³) wider than one-wrap exclusion tolerates
(~2⁴¹). Fixed instead by assembling both bracket products from 16-bit
limbs of S_total with tight range-checked carries, giving the identities
integer semantics outright; all windows are derived from (d, s, eps_int)
on both sides. See `rmsnorm-bracket-fix.md` and `test_rmsnorm_bracket.py`
(the forgery is a negative test). The original finding follows.

The bracket pins y by y²·S ≥ magic and (y−1)²·S < magic (slacks s_lo, s_hi
range-checked into a window of width W). Over the *integers* this pins a unique
y ≥ 0. Over the *field*, the constraints read y²·S ≡ magic + s_lo (mod P) etc.,
and — unlike every other bracket gadget in the system — **y itself carries no
range check**; it is not a lookup output (softmax) and not word-decomposed
(rescale). Subtracting the brackets gives (2y−1)·S mod P ∈ [1, 2W), which for
invertible S admits ~2W candidate y values of arbitrary magnitude; each
spurious candidate must then also land y²·S mod P in the width-W window, a
~W/P event per candidate — heuristically ~2W²/P expected spurious solutions.
Whether that is negligible depends on the slack window width
(W = 2^(chunks·chunk_width)) against |F| ≈ 2^64: at W ≈ 2^24 it is ~2^-15 per
row (small but searchable over many rows); at wider windows it degrades
quadratically. A wrapped y′ would change the normalized stream and hence
downstream values — in principle U-relevant.

**Recommendation (cheap):** add one range check on y per row (its honest
magnitude is ≤ √(magic/S_min), far below any wrap ambiguity) — bsz lookups per
claim, negligible cost — or write the number-theoretic exclusion argument into
Appendix B.4 with the concrete slack widths. The same discipline applies to any
future bracket gadget (including the NVFP4 requantization brackets of
`nvfp4-exact-path.md`): **every bracket operand must be independently
range-bounded**, so the inequalities have integer semantics; the accumulator
inputs to the requant claim should be range-decomposed for the same reason.

## 4. Benign free slots (no effect on any constrained value or on U)

Catalogued for completeness; each is either unconsumed or output-invariant:

- **Inverse helpers when the operand is zero**: the booleanization pattern
  g·inv_g = is_high, is_high·g = g forces is_high but leaves inv_g
  unconstrained when g = 0 (SiLU's inv_g, softmax's inv_z_high). The free slot
  feeds nothing. Canonicalize with one extra quad (is_high·inv_g = inv_g,
  forcing inv_g = 0 when is_high = 0) if a strictly unique witness is wanted.
- **SiLU at x = 0**: both sign assignments satisfy the split and both table
  halves return silu(0) = 0 — two witnesses, identical outputs.
- **MaxClaim's A under exact logit ties**: A may select either tied position;
  v* and every consumed quantity are invariant. (RoutingClaim, where the
  selection itself matters, has the tiebreak; MaxClaim deliberately does not
  need one.)
- **Last-row padding slots** beyond a variable's length L (and the K−ELL ZK
  slack columns): referenced by no constraint id; free by design — the slack
  is the hiding mechanism, the padding is layout.

## 5. Cross-cutting observations

- The system's uniqueness arguments come in exactly three flavors — linear
  images, range-bounded decompositions, and monotone brackets — and the bracket
  flavor is the only one whose field-vs-integer semantics needs per-gadget
  care. A one-line rule for new claims: *a bracket is sound iff every operand
  in it is independently bounded (by lookup, decomposition, or being a sum of
  bounded terms).* RMSNorm's y is the one current operand missing that bound.
- Witness variables that are inputs to no constraint at all would be silent
  freedom; the compile-difftest's whole-variable span checks plus the policy
  audit make these visible today, but an automated "every committed row is
  referenced by ≥ 1 constraint or documented as padding/blinding" lint would
  close the class structurally — a small addition to the §9 static-analysis
  item.
