# RmsNorm bracket fix: integer semantics for the rsqrt pin

**Status: IMPLEMENTED** (2026-07-06). Closes the gadget flagged in
`degrees-of-freedom-review.md` §3 — and the flag understated the problem
(§1 below): at the production chunk configuration the bracket was not
merely heuristically weak but fully vacuous.

## 1. The vulnerability, as-built

The rsqrt scalar `y` was pinned by two quadratic brackets

```
q1·S_total = magic + s_lo        q1 = y²,  s_lo ≥ 0   (lower)
q2·S_total = magic − 1 − s_hi    q2 = (y−1)², s_hi ≥ 0 (upper)
```

with the slacks word-decomposed into `slack_n_chunks × slack_chunk_width`
range-checked chunks, and **no range bound on y, q1, q2, or S_total**. Over
the integers the brackets pin a unique `y`; over the field they are
congruences mod P, and the "s ≥ 0" direction exists only if the slack
window excludes wrapped negatives.

Both production demos (`demo_llama7b.py`, `demo_maverick_full.py`) used
`slack_n_chunks=4 × 16` bits — a window of `2^64 − 1 ≥ P`. **Every field
element is decomposable in that window**, including wrapped negatives, so
the bracket constrained nothing: for ANY `y′` (no search, no grinding), a
prover could commit `q1′ = y′² mod P`, `s_lo′ = (q1′·S − magic) mod P`,
decompose the wrapped slacks into four in-range 16-bit chunks, forge
`output = x ⊙ y′` so the Freivalds row binds, and every constraint of the
claim was satisfied. `test_rmsnorm_bracket.py::test_forged_y_rejected`
constructs exactly this witness; the pre-fix verifier ACCEPTs it.

Impact: rsqrt uniqueness — an upstream-of-logits requirement (§2.3 of the
paper) — was absent in the demonstrated runs. The proofs remain valid
records of the honest computation, but "every forward-pass claim admits a
unique witness" was false as-built: a dishonest prover had a free scalar
per normalized row, propagating into the logits and hence the U bound.

The review's per-row estimate (~2⁻¹⁵ spurious solutions) assumed a 2²⁴
window; the deployed window was 2⁶⁴. A range check on `y` alone does NOT
repair it: with `y ≤ 2^21` and `S_total ≤ 2^42`, the product `y²·S_total`
still spans ~2⁸⁴, and for realistic row energies there exist wrapped
`(y′, k)` with `y′²·S = magic + kP + s`, `s` inside any window wide enough
for the honest slack (~2⁵³). The window tension is irreconcilable: honest
slack needs ≥ 2⁵³, exclusion of one-wrap solutions needs ≤ ~2⁴¹.

## 2. The fix: assemble the products from range-checked limbs

Bound every operand and build each bracket product limb-by-limb so no
intermediate can wrap — the identities then hold over ℤ, not merely mod P.

Committed per batch row (all length B), with `q ∈ {q1, q2}` per bracket:

```
y−1   = Σ 2^{16n}·ym1_chunks[n]      chunks tight to y_width      (F8)
S_total = S0 + 2^L·S1 + 2^{2L}·S2    three LIMB_W-wide limbs      (F9)
Hk    = q·Sk                          three quads, k ∈ {0,1,2}
H0    = g0l + 2^L·g0h                g0h 16-bit-chunked            (F10)
H1 + g0h = g1l + 2^L·g1h            g1h 16-bit-chunked            (F11)
H2 + g1h = G2                        G2 16-bit-chunked            (F12)
2^{2L}·G2 + 2^L·g1l + g0l ∓ slack = magic (−1)                    (F13/F17)
```

with `L = LIMB_W = 18` (see §3 for the choice). The limbs and carry lows
`g0l/g1l` are LIMB_W-wide; the carry highs `g0h/g1h/G2` are 16-bit-chunked
(their strides are independent of LIMB_W).

The splits telescope: `2^{2L}·G2 + 2^L·g1l + g0l = H2·2^{2L} + H1·2^L + H0
= q·S_total` exactly, so F13/F17 are the bracket identities — now over the
integers, because every term is individually bounded below P (values for
7B/Maverick at LIMB_W=18):

| term | bound | why |
|---|---|---|
| `q` | `2^{2·y_width}` = 2^42 | y−1 tight-chunked (F8), y = y_m1 + 1 |
| `S_total` | `2^{3·LIMB_W}` = 2^54 | three tight LIMB_W-bit limbs (F9) |
| `Hk = q·Sk` | `2^{2·y_width+LIMB_W}` = 2^60 | asserted `2·y_width + LIMB_W ≤ 63` |
| `g0h` | `2^{2·y_width}` = 2^42 | tight chunks |
| `H1 + g0h` | `< 2^61` | sum of the above |
| `G2` | `2^{G2_width}` = 2^25 | tight chunks — the key magnitude gate |
| `2^{2L}·G2 + …` | `< 2^61` | so F13/F17 compare integers < P |
| `s_lo, s_hi` | `2^slack_width` = 2^59 | `magic + 2^slack_width < P` asserted |

Given integer semantics, `y²·S_total ≥ magic` and `(y−1)²·S_total < magic`
pin `y` uniquely (monotonicity in `y ≥ 1`; `y ≥ 1` from the y_m1 range),
restoring B.4's uniqueness argument with no unwritten assumptions.

`S_total` is pinned mod P by `x` (F1/F3); if its value has no
representation below 2^{3·LIMB_W} the limb decomposition is unsatisfiable
and the proof rejects. The bracket therefore pins `y` as the rounded rsqrt
of the *committed* `S_total` — upstream integrity of `S_total` itself is
F1/F3's job, unchanged.

## 3. Derived widths — no prover-chosen windows

All windows are computed from `(d, s, eps_int)` in `RmsNormConfig`
(properties) and INDEPENDENTLY in the Rust verifier
(`handlers.rs::rms_widths`), so a prover cannot ship a widened window
(values for 7B/Maverick, s=2^12, eps=168):

- `y_width` = bit-length of `y_max − 1`, `y_max` = the pinned rsqrt at
  `S_min = d·eps_int` (asserts `2·y_width + LIMB_W ≤ 63`; **60 ≤ 63**).
  y_max = 1,294,391 → **21 bits** (d-independent: `y_max = ⌈s²/√eps⌉`).
- `slack_width` = bit-length of `2√(magic·2^{3·LIMB_W}) + 2^{3·LIMB_W}` —
  the largest honest bracket step for any limb-representable `S_total`
  (asserts `magic + 2^slack_width < P`). **59 bits** → [16,16,16,11].
- `g0h_width = 2·y_width` (42), `g1h_width = 2·y_width + 1` (43),
  `G2_width = bitlen((magic + 2^slack_width) >> 2·LIMB_W)` (**25**; asserts
  `G2_width + 2·LIMB_W ≤ 62`; **61 ≤ 62**).

**LIMB_W = 18 is the single S_total-headroom knob.** Its ceiling is set by
the two asserts above: the product `q·S_limb < P` caps `LIMB_W ≤ 21`, and
the slack/G2 no-wrap caps `3·LIMB_W ≲ 59` (i.e. `LIMB_W ≤ 19`). 18 gives
cap `2^54` (row RMS ≲ ~460 at 7B/Maverick — ~29× the 2^44 seen on real
activations) with a clean bit of margin on every assert. Raising to 19
(cap 2^57, RMS ~650) is valid but tightens G2 to exactly the limit; 20+
violates the slack bound. More headroom than that needs a fourth limb (a
new carry stage), not a wider one.

The limbs and carry lows (`S_limbs`, `g0l/g1l`) are range-checked against
`range_limb` (a `2^LIMB_W` table); the 16-bit chunks share
`rmsnorm_range_w16`, the narrow top chunks share `rms_w{k}` tables (all
via `Tape._range_table`). This is the B.1 discipline — the recomposition
ceiling of every decomposition sits far below P — which the old 4×16
slack decomposition itself violated.

## 4. Completeness

Honest values fit the windows by construction: `y ≤ y_max` at `S_min`,
slack ≤ one bracket step at `S_total < 2^{3·LIMB_W}`, carries bounded by
the limb algebra. The one new completeness bound is
`S_total < 2^{3·LIMB_W} = 2^54` (witness assert in `rmsnorm_compute`),
i.e. row RMS ≲ 512 real units at 7B (d=4096, s=2^12) / ≲ 458 at Maverick
(d=5120).

**How LIMB_W=18 was chosen.** The first cut used three 16-bit limbs
(cap 2^48, RMS ≲ 64). Measuring the 7B gate (SEQ=6, "The capital of
France is"): max `S_total` over all 65 rmsnorm claims was **2^44**
(RMS ≈ 16) — only **4 bits** under the 2^48 cap. Too thin to assume a
1000-token Maverick run, with 48 layers and a wider residual stream,
clears it (the assert fails LOUD prover-side, so it can only block a run,
never corrupt one — but blocking a 19 h run is worth avoiding). Widening
the limbs to 18 bits — same three limbs, same carry-chain depth — lifts
the cap to 2^54 (RMS ≲ ~460), ~29× the observed 2^44, at the cost of a
2^18 range table (4× the 2^16, still a few MB) and no new constraint
family. See §3 for why 18 and not more.

The **high-energy honest test** (`test_honest_high_energy`, S_total ≈
2^53) exercises the top limb in its >16-bit regime — the coverage the
widening exists for, which the SEQ=6 gate (2^44) does not reach.

Bit-compatibility: the witness values of all pre-existing variables
(`y`, `q1`, `s_lo`, …) are unchanged — the fix adds constraints the
honest witness already satisfies. Accuracy of the proven computation is
identical; proof format is not (new variables and families), so proofs
must be re-generated to verify against the new verifier.

## 5. Cost

Per batch row (B = tokens per claim): ~30 phase-1 committed slots, ~22
phase-2 z slots, 10 new linear families' worth of B-length rows, 6
product quads + ~22 range quads. At SEQ=1000 × ~65 rmsnorm claims at 7B
that is ~3M witness slots — noise against the 3×10^10 witness.

## 6. Tests

`prover/tests/test_rmsnorm_bracket.py`:
- `test_honest_toy` — toy config ACCEPT (Rust verifier).
- `test_honest_production_scales` — d=4096, s=2^12, eps=168, Gaussian
  activations: real derived widths, ACCEPT.
- `test_forged_y_rejected` — the §1 forgery, mod-P-consistent everywhere;
  must REJECT (pre-fix code ACCEPTs it).
`test_rescale.py::rmsnorm_rescale` continues to cover the rescale blocks.
