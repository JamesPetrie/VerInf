# Softmax bracket spike — findings

**Branch:** `lean-bracket-spike`
**Target (from the launch-risk analysis):** *"the bracket pins a unique `c` at S = 2, saturation included"* — build the field→integer lift library against the hardest consumer, and read the timeline off the result.
**Toolchain:** Lean 4 (v4.32.0) + Mathlib. `lean/BracketSpike/BracketSpike/Bracket.lean`, 224 lines.

## Result

The mathematical core of Lemma B.3 is **proven with no `sorry`** — every theorem below
depends only on Lean's three standard axioms (`propext`, `Classical.choice`, `Quot.sound`),
confirmed by `#print axioms`:

| Theorem | What it establishes | Maps to |
|---|---|---|
| `lift_cell` | field recomposition eq + range bounds + width condition ⟹ the genuine **integer** identity `c = x + z + Zmax·zhigh` | the Lemma B.1a lift (**Risk #1**) |
| `cell_value_neutral` | *any* saturation witness `(z, zhigh)` emits output `= g(c−x)` — full case split (below Zmax forces `zhigh=0`; at/above Zmax the table is already 0) | the compressed prose of Lemma B.3 (**Risk #3**) |
| `threshold_unique` | a non-increasing integer function is pinned by a two-sided adjacent bracket | the reusable monotone-pin schema (softmax **and** RMSNorm) |
| `g_noninc`, `Row.s1_noninc` | the ideal cell output, and the row sum, are non-increasing in `c` | monotonicity step |
| `shift_unique_S2` | two satisfying witnesses at S = 2 share the same shift `c` | the spike's stated target |

The three pieces that were *most uncertain* going in — the field↔ℤ lift, the saturation
value-neutrality case split, and the monotone-threshold argument — all went through
**as expected mathematically**. No hidden mathematical surprise surfaced. That is the
single most important read from this spike: the softmax bracket's difficulty is *plumbing*,
not *mathematics*.

## What is modelled / abstracted (honest limits of the spike)

This is **not** yet an end-to-end "raw ZMod constraint system ⟹ unique `c`" theorem. The
following are assumed via the model rather than derived, in rough increasing order of
remaining work:

1. **Table certificate.** `TableCert` carries the table's properties (nonneg,
   non-increasing, zero-tail before Zmax) as hypotheses. In the real gate these come from
   a `native_decide` scan of the concrete 80 000-entry table — a known-cheap discharge, not
   done here.
2. **δ-shift step.** `s2(c) = s1(c−1)` is baked into the `Row` model (the bracket uses
   `s1(c−1)` directly) rather than derived from the `T_B[k] = T_A[k−δ]` table pair. This is
   the "the tables are bit-identical up to δ" clause of Lemma B.3.
3. **Boolean-flag derivation.** `cell_value_neutral` takes `t = [zhigh ≠ 0]` as given. The
   actual construction pins `t` through the quadratics `t·zhigh = zhigh`, `t² = t`.
   **Finding:** deriving `t ∈ {0,1}` from `t² = t` over `ZMod P` needs `P` to be an integral
   domain — i.e. **it pulls in primality of Goldilocks**, which the lift (`lift_cell`)
   deliberately avoids. Proving Goldilocks prime in Lean is a one-time task (`native_decide`
   or a Pratt certificate) that the bracket-flag step forces but the lift does not.
4. **End-to-end glue.** The pieces' types line up (lift → cell-neutrality → sum →
   threshold) but are not yet composed into one statement quantified over the raw witness.
5. **Masked cells.** Only causal/unmasked cells are modelled; the `i > q` doubled-zero-table
   value-neutrality (a separate, simpler argument) is not here.

## Where the friction actually was

100% of the iteration cost was Lean plumbing, none of it mathematical:

- `omega` + numeric atoms: `omega` does not chain a variable bound (`Zmax ≤ 2^16`) into a
  concrete `< P` fact on its own; each such step needs a `norm_num [P]` numeric lemma fed in
  explicitly. This recurred throughout the lift.
- `ZMod.val_*` API (`val_add`, `val_mul`, `val_natCast_of_lt`) + `Nat.mod_eq_of_lt` — the
  standard lift toolkit, straightforward once the pattern (`val_add_lt`) was factored out.
- `split_ifs` / `simp only` ergonomics for the `g` case analysis.

The `lift_cell` / `val_add_lt` pair **is** the reusable lift library the launch plan calls
for; it amortizes across RMSNorm's carry chains and every `range`/`rescale` line.

## Timeline read

- **Risk #1 (softmax bracket hides a math surprise): retired.** The hard mathematical
  content is proven and behaved. The remaining items (1–5 above) are mechanical or
  one-time, not research.
- **The two-week estimate for a full, `sorry`-free, S = 2 softmax bracket looks credible,
  arguably conservative** for the S = 2 case: the hard core took a fraction of a session of
  focused work plus a handful of compile iterations. The bulk of the remaining two weeks is
  the glue (item 4), the flag/primality step (item 3), the δ-shift (item 2), and masked
  cells (item 5).
- **New concrete finding for the plan:** budget a one-time "Goldilocks is prime" proof — the
  bracket's boolean-flag step needs it (item 3), even though the lift does not.
- **Not tested by this spike:** the parametric-in-`S` version (the ∀S induction over row
  length). That remains the genuinely open estimate; this spike only exercised fixed S = 2.

## Reproduce

```
cd lean/BracketSpike
lake exe cache get && lake build          # clean build, 8660 jobs
# axiom ledger (shows only propext / Classical.choice / Quot.sound):
#   see BracketSpike/Audit.lean
```
