# Persistent weight commitment — design

Reference doc for the persistent-weight-commitment work (agent-plans
`plans/infproof/persistent-weight-commitment.md`). This is the spec the
implementation phases difftest against; it fixes the contract, not the code.

## Goal

Commit a model's weights once and reuse that commitment across many
inference proofs, then refresh (re-commit with fresh randomness) and link
(prove same weights) when the per-commitment column budget nears
exhaustion. Paper §5.1 (three commitment blocks, weights "committed once
and persistent across queries") and §9 (the linking proof) are the
publication form.

## Current state (baseline)

Two commitment blocks, laid out by `Variable.phase` in `core._layout`:

- **phase 1** — activations, routing masks, norm auxiliaries, AND weights
  (weights are ordinary `commit_lazy` vars, `phase=1`). Rows
  `[0, m_p1_rows)`, Merkle root `root_p1`. Rows `0..NUM_BLINDING_ROWS`
  are the test-polynomial ZK blinding.
- **phase 2** — challenge-dependent aux (Freivalds projections, LogUp
  inverses). Rows `[m_p1_rows, m_total)`, root `root_p2`.

The proof carries `root_p1, root_p2` and two opened-column maps
(`opened_p1, opened_p2`); the verifier checks each map against its root
(`verify.rs::merkle_test`), joins them in row order (commit 0 ‖ commit 1,
`OpenedColumns::into_joint`), and runs the six checks on the joint columns.
Weights are re-committed every proof.

## Target block structure

Three blocks, in row order:

1. **W (weights)** — rows `[0, m_w)`, root `R_W`. Persistent: committed
   once, referenced by later proofs. NO per-proof blinding rows here (the
   block must be deterministic; see below).
2. **p1 (activations)** — rows `[m_w, m_w + m_p1)`, root `R_p1`. The
   `NUM_BLINDING_ROWS` test-blinding rows move to the head of THIS block
   (they are per-proof, so they belong with the per-proof block).
3. **p2 (aux)** — rows `[m_w + m_p1, m_total)`, root `R_p2`.

All three share ONE column-query set (the verifier's `s_col` draw), so an
opened column carries three Merkle paths (one per tree) and the joint
column is `W-rows ‖ p1-rows ‖ p2-rows`, in that row order.

### Blinding-row placement: two prover layouts (decision for the emission step)

The `NUM_BLINDING_ROWS` (=3) test-polynomial blinding rows are today at
absolute rows `[0,3)`, prepended to the p1 tree, and the IRS witness index
is `abs − NUM_BLINDING_ROWS` (blinding at the front, all witness after). A W
block that must be context-independent (fixed `row_start`s) has to sit
before the activation vars, which conflicts with the blinding prefix. Two
resolutions, with different risk:

- **(A) 3 roots, W first, blinding moves to p1.** Layout
  `[W 0..m_w][p1: blinding m_w..m_w+3, activations][p2]`. Matches the
  paper's three blocks exactly, but the IRS index FORMULA changes to
  `abs if abs < m_w else abs − 3`, on BOTH prover and verifier — IRS
  soundness surgery.
- **(B) 4 roots, blinding its own tree, W second.** Layout
  `[blind 0..3][W 3..3+m_w][p1: activations][p2]`. The IRS index formula
  `abs − NUM_BLINDING_ROWS` is UNCHANGED (blinding stays at the front); only
  `row_start` ASSIGNMENTS change (weights grouped). W is still
  context-independent (fixed `+3` offset). Cost: a fourth root (the
  N-generic verifier handles it for free) and the blinding becomes an
  explicit tiny (3-row) block rather than a p1 prefix.

**Chosen: (B) — IMPLEMENTED (2026-07-07, commit fb05156).** It avoids the
IRS index formula change — the single riskiest edit — for the price of one
extra root the verifier already supports. `commit_weights` lays weights at
`row_start = NUM_BLINDING_ROWS` (reserving `[0,3)`) so the standalone R_W
matches `prove_streaming`'s W tree bit-for-bit. Join order:
`blind ‖ W ‖ p1 ‖ p2` (row order). The paper's conceptual three blocks
(weights / activations / aux) are unchanged; the blinding tree is an
internal detail.

(A layout-C detour — blinding kept in the p1 tree, W appended after the
activations, 3 blocks — was implemented and 7B-verified first as the minimal
P2 step, then replaced by B on preference: B is cleaner per-block and yields
a context-independent R_W immediately, folding P3's reconciliation into P2.
The "own sweep" finding below still holds for the STANDALONE `commit_weights`
path; inside `prove_streaming` the streaming sweep already feeds each block's
tree in row order via per-block row_start assignment, so no separate sweep is
needed there.)

### Why weights lead and carry no blinding

`R_W` must reproduce bit-for-bit across proofs of different prompts, or a
later proof cannot reference the earlier commitment. Two requirements:

- **Context-independent rows.** Weight vars are laid out FIRST, in a stable
  order independent of the per-proof activation/aux vars (which scale with
  sequence length). Their absolute `row_start`s are then fixed by the model
  alone. Because the ZK padding is `row_prg(master_seed, row_offset, ·)`,
  seeded by absolute row index, fixed rows ⇒ reproducible codewords ⇒
  reproducible `R_W`.
- **No per-proof blinding in W.** The test-polynomial blinding rows are
  regenerated per proof; putting them in W would change `R_W` each proof.
  They live in p1. The weights' own per-row padding (deterministic) is what
  hides the weights at opened columns.

## The generic multi-root contract (P2)

Replace the hard-coded two roots with N labelled roots + a per-variable
block assignment:

- Prover emits `roots = [R_W, R_p1, R_p2]` and opened-column maps per block.
- Verifier accepts `roots: &[[u8;32]]` (not `[_;2]`), checks each block's
  opened map against its root, and joins the per-block columns in block
  order to form the joint column.
- Block membership per row is explicit in the proof (which opened map a
  column value sits in), exactly as the two-block case is today — the
  generalization is 2 → N maps/roots, not a new mechanism.

Build it generically (N, not 3) so the same extension serves a future
second commitment (e.g. the token-binding recorder). This is the only
change that grows the audited verifier TCB.

### Value-parity gate

The 2→3 split must be a pure regrouping: the joint columns and every one of
the six check values must be identical to the two-root path on the same
witness (weights simply moved from the p1 tree into their own tree, same
rows in the same join order). Gate with `VERIFY_DUMP_CHECK_VALUES` (the
verifier's per-check value dump) comparing a stored proof across the two
builds, plus prove→Rust-ACCEPT on the real models.

## Confidentiality budget

Opening a fixed commitment at columns C₁ in proof 1 and C₂ in proof 2
reveals exactly the evaluations at C₁ ∪ C₂ — identical to opening the union
in one proof. So Ligero's single-proof hiding bound applies to the union:
**`K_DEG − ELL = 16384 − 8192 = 8192` distinct columns are perfectly
hiding** (the 8192 random padding values give a full-rank mask for any
≤ 8192 openings, by the Vandermonde/NTT structure). The budget is the count
of DISTINCT columns opened against `R_W` across all proofs; a deployment
may hold a margin under 8192. Per-test blinding rows are per-proof and
refreshed each proof, so they do not consume the weight budget. Ledger
(P4) is prover-side: the verifier always benefits from more columns, so
budget enforcement protects only the prover's own weight secrecy.

## Refresh + linking (P5) — a standard proof

When the union nears the budget, `refresh_weights` re-commits the same
weights under a fresh seed → `R_W′`, restoring the full budget. The linking
proof rides the existing claim infrastructure:

- Two weight blocks in one tape: `W_old` (committed under the old seed, so
  its root equals the trusted `R_W`) and `W_new` (fresh seed → `R_W′`).
- Equality via `LinCombClaim`: `W_old[i] − W_new[i] = 0`, Freivalds-
  compressed to O(weight rows) (random λ: `λᵀ(W_old − W_new) = 0`), checked
  by the existing `lin_sum`/`lin_col` tests. NO new gadget.
- The verifier anchors `W_old` to the `R_W` it already trusts and adopts
  `R_W′` once the proof verifies; the equality carries the binding, so a
  prover cannot substitute different weights (the equality would have to
  hold against the real `R_W`).

Soundness: RS distance, identical to Ligero's linear test — a differing
weight row makes the difference codeword far from the zero-message subcode,
caught in a constant fraction of columns. A **negative test** (link `R_W′`
to different weights → REJECT) is mandatory. ZK: per-proof blinding, as
usual. Cost: the refresh re-folds both weight blocks (~one extra
weight-fold), one-time every ~100 proofs at T=80.

### The construction subtlety: decoupled logical row offset (2026-07-07)

"Two weight blocks under different seeds" is not quite a plain proof,
because the padding PRG is keyed by `(master_seed, ABSOLUTE row)`. `R_W`
and `R_W′` are both commitments to weights at rows `[3, 3+m_w)` — same
rows, different SEED. In a single linking proof the two blocks must sit at
different physical rows (they can't both be `[3, 3+m_w)`), yet each must
reproduce a root computed at rows `[3, …)`. Two facts make it work:

- **The Merkle tree is over COLUMNS** (each leaf hashes a column's values in
  row order), so a block's root depends only on its codeword VALUES and
  their order, NOT on the block's absolute row indices. So `W_new` placed at
  physical rows `[3+m_w, …)` still yields root `R_W′` as long as its
  codeword values match — i.e. as long as it is padded as if at the logical
  offset the refresh used.
- **The padding therefore needs a LOGICAL row offset** decoupled from
  physical placement: encode `W_old` with `(seed_A, logical 3)` and `W_new`
  with `(seed_B, logical 3)`, wherever they physically land. Same message,
  different padding → different roots (`R_W`, `R_W′`), and the `LinComb`
  equality on the message slots links them.

Enabling change: `encode_messages` / the sweep take a per-weight-block
`(seed, logical_row_offset)` for the ZK padding, independent of the physical
`row_start`. `commit_weights(seed=…)` already varies the seed (the refresh
primitive); the linking proof adds the second block + the logical offset.
This is the one place the encode's seed/offset is not the global master.

### P5 build order

1. **Refresh primitive** — `commit_weights(tape, cfg, seed=B)` → `R_W′`
   (already supported; `WeightCommitment.from_tape(..., master_seed=B)`).
   Gate: `R_W(A) ≠ R_W(B)`, same weights. **DONE + verified 2026-07-07.**
2. **Per-block (seed, logical offset) padding** in `encode_messages` + the
   sweep's weight emit. Gate: a block emitted at physical rows `[X, …)` with
   logical offset 3 + seed B reproduces `R_W′` from step 1.
   **DONE + Spark-verified 2026-07-08 (commit 60b5cc2;
   `test_persistent_weights_p5` 3/3, and the P3/P1 suites re-passed 3/3+3/3 —
   the defaults are an identity):**
   `_stream_phase(pad_seed=, pad_row_offset=)` decouples the padding PRG from
   the physical `abs_row_offset` (which the merkle/q accumulators keep);
   `_stream_sweep(w_pad=(seed_t, logical_offset))` applies it to the weight
   emit. Bonus fallout: `WeightCommitment` records the seed it was committed
   under, and `prove_streaming(weight_commitment=wc)` pads the W block under
   `wc.master_seed` — so proofs can reference a REFRESHED commitment (the
   deployment story after each refresh). Completeness guard: a refreshed-seed
   reference asserts no quad constraint touches W rows, because `p_0`'s
   sparse re-encode (`compute_p_0_streaming`/`_encode_rows_indexed`) pads by
   PHYSICAL row under the master seed — a quad on differently-padded W rows
   would make `p_0` inconsistent (REJECT, not unsound). Weight-touching quads
   (matmuls) are unaffected on the standard path (default seed ⇒ identity).
   Gates passed (`test_persistent_weights_p5`): a block emitted at physical
   row 1003 padded as (seed B, logical 3) reproduces `R_W′` bit-for-bit,
   with two negative controls (physical-keyed pad, right-seed-wrong-offset);
   refreshed-reference prove → Rust ACCEPT with `root_w == R_W′`; seed
   save/load with pre-P5 pickle compat.
3. **Linking proof** — a tape/mode with `W_old (A, logical 3)`,
   `W_new (B, logical 3)`, and the `LinComb` equality; run the standard
   prover/verifier (N-root already supports two weight roots). Gate:
   ACCEPT, `root_wold == R_W`, `root_wnew == R_W′`.
   **DONE + Spark-verified 2026-07-08 (commit 66a5667).** Construction:
   `persistent="new"` marks a var for a SECOND weight block "wnew" (row
   order `[blind | W | Wnew | p1 | p2]`); `prove_streaming(wnew_seed=B)`
   builds the wnew tree padded under `(B, logical 3)` at physical rows
   `[3+m_w, …)`; the equality is `tape.lincomb([w_old, w_new], [1, -1], 0)`
   per weight var. `verify_proof.rs` parses the `wnew` block; the N-generic
   `verify()` needed no change. The completeness guard extends to the wnew
   block (no quads on either differently-padded weight block). Adoption
   contract (deployment): accept `R_W′` iff proof ACCEPTs ∧
   `root_w == trusted R_W` ∧ `root_wnew == claimed R_W′`.
4. **Negative test** — link `R_W′` to DIFFERENT weights → REJECT (the
   load-bearing soundness check).
   **DONE + Spark-verified 2026-07-08** (`test_persistent_weights_link`
   4/4): honest link ACCEPT with both roots binding; DIFFERENT weights →
   REJECT (both a full-block difference and a SINGLE-element tamper —
   lin_sum's random fold catches any false equality w.p. 1−1/P regardless
   of opened-column count); wrong refresh seed → `root_wnew ≠ R_W′`, so
   the adoption comparison refuses it (the proof itself stays consistent).
   Regressions all green on the same run: p5 3/3, p3 3/3, p1 3/3,
   lincomb 5/5, test_claims 21/21.

## P6 independent audit (2026-07-08) — findings & disposition

Fresh-context audit (opus, no implementation context, checklist per the
project audit procedure) of P1–P5 on `feat/persistent-weights`. Verdict: the
cryptographic mechanism (multi-root, decoupled padding, RS linear-equality
link) is correctly implemented for an honest prover; the gaps are at the
trust boundary. Findings:

- **S1 (soundness, DEFERRED by decision 2026-07-08):** the linking statement
  lives entirely in the prover-supplied claims JSON; a prover can omit the
  LinComb equalities and get an ACCEPTing proof binding `root_w == R_W` and
  an arbitrary-weights `root_wnew`. The three-part adoption check is
  therefore insufficient — the deployment must ALSO validate that the claims
  are the canonical linking statement (exactly m_w equalities `[1,−1]·
  [w,wnew]=0` spanning both blocks, expected block list/sizes). This is the
  same trust boundary as the deferred "verifier validates the inference
  claims themselves" work and will be addressed with it. Until then the
  adoption contract is: ACCEPT ∧ root_w == R_W ∧ root_wnew == R_W′ ∧
  **claims validated as the canonical link statement (currently manual)**.
- **S2 (soundness, FIXED 2026-07-08):** `merkle_test` skipped ANY block
  declaring the all-zeros `EMPTY_COMMIT_ROOT` (the zero-row-block sentinel,
  e.g. an empty p2) without checking the block was actually empty —
  declaring a zero root for a NON-empty block (root_p1 has no external
  comparison, unlike root_w) disabled merkle binding for its opened columns
  entirely, admitting forged witnesses even for a validated statement. Fix:
  the sentinel is accepted only when the block's opened sub-columns are all
  empty (`verify.rs::merkle_test`); negative gate
  `test_empty_root_sentinel` (zeroed blind/w/p1 roots → REJECT; genuinely
  empty p2 still ACCEPTs).
- **H1 (open):** `WeightCommitment.load` is pickle — do not load untrusted
  `.wc` files; replace with a non-executable format before the commitment
  crosses a trust boundary.
- **H3 (spec note):** a linking proof opens T columns against the OLD
  commitment, so refresh at a margin of ≥ T under the 8192 budget.
- **H2 (spec correction):** the test-poly blinding rows are fixed-seed
  (identical across proofs), not "regenerated per proof" — the pre-existing
  fixed-MASTER_SEED ZK caveat, tracked separately.
- **C1 (hardening, open):** the quad guard checks band ANCHOR rows only —
  exact for the current claim set; tighten to full band spans if a
  multi-variable-span family is ever added. Completeness-only either way.
- **C2 (known):** the staged interactive verifier is still 2-root; the
  offline `verify_proof` path (the shipping path) is N-generic.
- Confirmed by the audit: every listed block is merkle-checked, the LinComb
  binds messages per-slot, padding is prover-side only (decoupling cannot
  affect soundness), serializer/parser block handling matches name-by-name,
  legacy 2-block proofs parse byte-identically.

## Implementation finding (2026-07-07): weights need their own sweep

`core._layout` deliberately keeps **row-order == op-order** ("so the prover
can stream the witness in one forward sweep"): the streaming merkle
accumulator hashes each column's entries in the order rows are *emitted*,
which is op (tape) order, and that must equal row order or the committed
tree diverges from the row-ordered layout the verifier reconstructs. But
weights are committed at their op position — interleaved with activations
across layers (each `commit_lazy` fires during its layer's build). So a
weights-first block is **not** a row-range reshuffle of the existing sweep;
emitting weights first while computing activations in forward order would
break the op==row invariant.

Resolution (refines the P1/P3 split): the W block gets its **own sweep**.
Weights don't depend on activations, so a dedicated weight sweep emits all
weight rows (loaded from disk) into the W tree, in weight-row order, before
the per-proof activation sweep runs the forward pass into the p1 tree.
This is exactly the `commit_weights` operation P3 already wants — so P1 (the
W block existing with its own root) and P3 (that block committed by a
standalone, persistable operation) are one unit, not two. The per-proof
prover then runs the activation+aux sweeps (weights excluded) and, for the
weight columns, opens against the pre-committed W tree. `_layout` still
assigns weight vars their own contiguous row range (block W), but the
COMMIT of that range is a separate pass, not part of the op-order sweep.

Consequence for sequencing: implement the weight sweep + R_W first (P1∪P3
core), gated by "same weights → byte-identical R_W across two runs," then
the generic multi-root verifier (P2), then wire the per-proof path to
reference R_W. The budget ledger (P4) and refresh/link (P5) sit on top
unchanged.

## Persistent artifact (P3)

`commit_weights(model) → {R_W, tree levels, master seed, model/scale
metadata}`, persisted to disk. The encoded matrix (≈60 TB for Maverick) is
NOT stored; columns are re-encoded on demand from the on-disk weights at
open time (the padding PRG makes re-encode bit-reproducible — the same path
round 4 already uses). Per-proof, the prover loads `R_W` and opens the
weight columns from the stored tree instead of rebuilding it. This saves
the weight column-hash + Merkle build (the compute-bound `D·W` weight term,
§8) — NOT the fold re-encode, since weights still fold into each proof's
linear/quad tests.

## What is NOT new research

Recorded because the plan's scope rests on it (design discussion
2026-07-07): the budget is the standard Ligero bound (via the union
reduction above), and the linking is the existing equality claim, not a new
gadget. The only genuinely new code is the generic multi-root verifier
support.
