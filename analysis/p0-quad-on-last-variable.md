# Quadratic fold on the last-instantiated variable (keeping `p_0` in the per-variable pass)

Status: **design note, not yet implemented** (2026-06-26). Companion to
`qlin-family-object-reorg.md` and `qlin-fold-reorg-plan.md`. Those cover the
**linear** fold (`q_lin`); this covers the **quadratic** test (`p_0`). It resolves
one deferral in that reorg: when the linear fold becomes variable-stationary, where
does the quadratic fold go?

## 1. The problem

The prover's round-3 sweep builds three test polynomials together — `q_irs`,
`q_lin`, and `p_0` (the quadratic / Hadamard test). Today the sweep is **per-op**: it
walks the tape in op order, and as each op's rows become live it folds that op's
linear contribution into `q_lin`/`q_irs` and fires that op's **owned quads** into
`p_0` (`core.py:2111`). A quad fires *from the live witness* — there is no separate
quadratic pass.

The `qlin-family-object-reorg.md` reorg changes the linear fold from per-op to
**variable-stationary** (one `ConstraintFamily` per `(variable, role)`; encode a
variable's rows once; feed `q_irs` and `q_lin`). That note explicitly **defers**
`p_0`: "`p_0` stays constraint-stationary (its 3-row coupling does not fit a
per-variable pass)." Taken literally, that means pulling `p_0` out of the main sweep
into a **separate** pass over all quadratic constraints that re-gathers and
re-encodes their rows — a regression from today's "fired in the sweep, from live."

This note shows how to keep `p_0` folded into the (now variable-stationary) main
pass, so the reorg does not reintroduce a separate quadratic sweep.

## 2. What a quad is, and how it fires today

A `QuadraticConstraint` (`core.py:88`) is, for `i ∈ [0, n)` with `n ≤ ELL` (single
row), `x[i]·y[i] + a·z[i] = b`, with `a`, `b` uniform scalars. It couples **three**
encoded rows: `x_row`, `y_row`, `z_row`.

Folding it into `p_0` needs all three rows as **polynomials**:
`poly_mul(px, py) + a·pz − b`, weighted by the per-quad challenge `r_quad[gi]`, and
accumulated (`compute_p_0_streaming`, `core.py:1275`). The product `px·py` is a real
polynomial multiplication (degree `2K`), which is why the quadratic fold is the
NTT-heavy sibling of the linear one.

Today the three rows are available because:

- **Ownership** (`_quads_by_op`, `core.py:1946`): each quad is assigned to the op
  that owns one of its rows; a **relayout co-locates the quad's three rows in/near
  that one op**, so — in the function's own words — "that op's processing has every
  operand live."
- **Firing** (`core.py:2111`): when the owning op is processed,
  `compute_p_0_streaming` gathers the three rows from `live`, re-encodes them
  (deterministically per `(master_seed, row)`), and folds them into `p_0`.

So the invariant the reorg must preserve is: **when a quad fires, all three of its
rows' witnesses are resident.** Today that rests on the relayout. The
variable-stationary pass processes *variables* in order, not co-located op groups, so
the relayout assumption no longer holds for free.

## 3. The rule: fire a quad at its last-instantiated variable

Assign each quad to the **variable that instantiates the last of its three rows** in
the pass order. That instant is exactly when all three rows are first simultaneously
live: the two earlier rows are already materialized, and the third has just been
produced. When the variable-stationary pass reaches a variable `V`, after encoding
`V`'s rows it fires every quad whose last row belongs to `V`, gathering the three
(live) rows and folding them into `p_0` with the existing kernel.

This is the principled form of today's "the op that has every operand live." It
replaces the relayout's *co-location* with explicit *liveness*: instead of reordering
ops so a quad's rows are adjacent, keep the two earlier rows alive until the last is
processed.

## 4. Obligations

Four conditions; (a)–(b) are correctness/soundness, (c) is memory tuning, and (d) is
the iteration-order constraint the row layout imposes.

**(a) Every quad fires exactly once (coverage).** Soundness requires every quadratic
constraint to be checked — a skipped quad is a hole, exactly as a skipped linear
constraint is. Assign each quad to the unique row with the latest pass position among
`{x_row, y_row, z_row}`; break ties (two rows in the same variable / same position)
deterministically. A quad whose rows include public/constant operands with no witness
variable fires at the latest *witness* row among the three. This is the prover-side
analogue of the verifier's "for each quad in `cons.quadratic`" coverage — gate it so
the multiset of fired `gi` is exactly `range(len(quads))`.

**(b) The three rows are live when the quad fires (liveness).** A row's witness must
survive not only its last *data* use but the quads that read it. The clean way to get
this (§9) is to treat a quad operand as a **use of that variable by the emitting
claim**: a variable's `last_use` is then simply the latest claim that touches it — as a
data input *or* a quad operand — and freeing is claim-by-claim. A quad is just another
consumer; there is no separate per-quad liveness analysis. For the quads that actually
occur — Freivalds `p[i] = u[i]·y[i]` (three rows from one matmul), the SiLU / MoE
booleans (`m^2 = m`) — all three rows belong to one claim, so the resident window is
one claim wide; only a quad reaching back to an earlier claim's variable extends a
lifetime, bounded by that span.

**(c) Re-encode at firing, or hold the encoded rows (a knob).** The fold needs the
rows as polynomials. Two options: re-encode the three rows from the live witness at
firing — what `compute_p_0_streaming` already does, deterministic and exact (same
codewords as at emission; the `_late_qlin_var` precedent, `core.py:1969`), keeping
only the witness values resident but re-encoding the two earlier rows a second time;
or keep the earlier rows' encoded polynomials resident to skip the re-encode. For
local quads either is cheap; re-encode-at-firing is the simpler default and matches
the existing kernel.

**(d) The row layout demands the claim-grouped order — which (b) already gives.** Rows
are laid out **phase-major** (`_layout`, `core.py:1901`):
`[blinding][all phase-1 vars, claim order][all phase-2 vars, claim order]`, so a single
claim's phase-1 and phase-2 variables sit in two *distant* row blocks. A **cross-phase**
quad — a phase-1 operand times a phase-2 inverse — therefore spans those blocks in row
order, and a pure phase/row-major walk would split its operands, blowing the one-claim
window of (b) up to a whole-block one. This needs no separate rule: it *is* the
claim-as-unit model of (b) and §9. Processing and freeing **by claim** keeps a claim's
p1 *and* p2 families together — as the current op-major sweep already does
(`core.py:2105`+`2108`) — so cross-phase quads stay local automatically and the
implementation never audits which operands are which phase. (Whether genuine
cross-phase quads exist — e.g. the UI `dw_j` (phase-1) × its `z_dw_j` inverse
(phase-2) — was not exhaustively confirmed; claim-as-unit is robust either way.)

## 5. Why this beats deferring to a separate pass

The deferred (constraint-stationary) alternative is a separate sweep over all quads
that re-gathers and re-encodes their rows — a full extra pass with its own row I/O.
Firing at the last variable instead:

- rides on the variable-stationary pass the linear reorg already runs (no extra
  sweep);
- reuses the rows the pass already has live (and, with the 4c "hold" option, the
  polynomials it already encoded);
- preserves today's "`p_0` in the main sweep, from live" property through the reorg
  rather than regressing it.

The fold *kernel* (`compute_p_0_streaming`'s inner: `poly_mul(px,py) + a·pz − b`,
`r_quad`-weighted) is unchanged. Only **when** each quad fires and **how** its rows
are kept resident change.

## 6. Relationship to the relayout

The current relayout exists to co-locate a quad's rows so per-op firing finds them
live. Under last-variable firing the relayout is no longer *required* (liveness
guarantees residency), but it stays a useful *optimization*: ordering so a quad's
three rows are instantiated close together minimizes how long the earlier rows must
be held. Keep the relayout as a memory optimization, not a correctness dependency.

## 7. Validation

The seam is bit-exact, like the linear reorg's: the `p_0` produced by last-variable
firing must equal today's per-op `p_0` byte-for-byte (`torch.equal`), env-gated
default-off. Gate in order:

1. **Coverage:** the multiset of fired `gi` equals `range(len(quads))` — no skip, no
   double — independent of any proof.
2. **`p_0` equality** vs the current per-op fold on the routing / E=8 / single-layer
   proofs.
3. **End-to-end ACCEPT** on `demo_maverick_moe.py`, with the shipped verifier
   (`quad_zero`: `p_0(ζ_c) = 0` for all `c`; `quad_col`: the quadratic column
   identity) as the cross-impl oracle — the same verifier that gates the linear
   reorg.

## 8. Scope and sequencing

This is an **extension to `qlin-fold-reorg-plan.md` P3**, not a standalone project: it
only makes sense once the linear fold is variable-stationary (before that, the per-op
firing already works and there is nothing to preserve). It is also **optional** — the
deferred constraint-stationary `p_0` is correct and memory-bounded; this note's value
is avoiding a second full pass over the witness.

Recommended sequencing: land the linear reorg (plan P1–P4) with `p_0` deferred to the
constraint-stationary pass as the plan states; **measure** what that separate pass
costs once the linear side is variable-stationary; adopt last-variable firing only if
eliminating it is worth it. There is almost no genuinely new machinery: the ownership
rule (§3) and the fold kernel (`compute_p_0_streaming`) already exist, and the liveness
is just "count a quad operand as a use by its claim, free by claim" (§9) — an extension
of the existing free-at-boundary path, not a new analysis. This note reuses all of it
and only changes when each quad fires.

## 9. What the q_lin reorg should leave (the seams)

This fold rides on the variable-stationary q_lin reorg (`qlin-fold-reorg-plan.md`), so
the cheapest path is for that reorg — at its P3, which already reworks the lifetime
system — to leave a few seams. Most cost nothing: they are choices about *how* to
structure work it is already doing.

**The one that matters — make the claim the unit of processing and freeing.** Bump a
variable's `last_use` on **every** reference during construction — data input *or* quad
operand — so `last_use[v]` is just the latest claim that touches `v`. Walk
claim-by-claim; within a claim, fire its quads, then free the variables whose
`last_use` is that claim (an input a later claim still uses has `last_use` beyond it and
stays). This single choice delivers **both** outstanding obligations at once: the
liveness (b) — a quad's operands are live at its claim because the claim referenced
them — and the claim-grouped order (d) — a claim's p1 and p2 families are processed and
freed together, so cross-phase quads never split. It is the existing free-at-boundary
pattern (`core.py:2111` fire, `2123` free) with the claim as the unit and quad operands
counted as uses — **no bespoke quad-liveness code**. The failure mode to avoid: baking a
data-only `last_use` deep into the free path, which forces a re-open later. So the seam
is exactly: *keep `last_use` a per-variable latest-touch value, and free by claim.*

**Cheap hooks for the later transform fusion** (so `p_0` can ride q_lin's NTT instead
of re-encoding — see §4c, and `qlin-evenodd-fused-multiply.md`):

- Keep a variable's transformed (eval-domain) rows a **retainable handle**, not a fused
  transform–multiply–free step, so `p_0` can reuse them. Holding them for the one-claim
  window of (b) is cheap.
- Set up the eval-domain q_lin accumulator and the global-inverse-at-finalize (the
  inverse-NTT fuse) so `p_0`'s products can be added into the **same** accumulation and
  share the one inverse. Then fusing `p_0` is "also accumulate here," not a new finalize.

**Smaller conveniences:**

- Build the count-pass + reverse index generically (`variable → [attachment]`) so
  `variable → [owned quads]` attaches the same way `variable → [family]` does.
- Leave `compute_p_0_streaming`'s inner kernel intact when deleting `_StreamingPackets`
  / `_late_qlin_var`; only its scheduling moves.
- While re-measuring (plan P4), instrument two numbers: the max quad span (`last − first`
  operand claim-position) and `p_0`'s transform share — the first confirms the
  one-claim-window assumption, the second sizes the fusion win.
