# Verifier per-family representation (design + staging)

Status: **design, implementation starting 2026-06-24.** Builds on the per-Expander
eval seam + shortcuts (PR #7) and the row-streaming verifier (PR #6), which this
representation *consolidates and partly supersedes*. Companion to
`verifier-constraint-family-streaming.md` and the prover-side
`qlin-family-object-reorg.md`.

## The key finding: it *simplifies* the compile

Reading the verifier's emit helpers (`handlers.rs`), every family is emitted by the
**same uniform per-row loop**:

```rust
for ro in 0..nrows(var.length, ell) {
    let flat_lo = ro * ell;
    let n_slots = ell.min(var.length - flat_lo);
    self.push(var.row_start + ro, Expander::Kind { /* params */, flat_lo, n_slots });
}
```

The params are **constant across the rows** (only `Identity` advances `cid_base` by
`ro*ell`); the only per-row variation is `flat_lo = ro*ell` and the last-row
`n_slots` clamp. So a per-`(variable, family)` object plus a **uniform
reconstruction rule** captures the whole per-row set. That means the per-family
representation **removes the per-row `for ro` loops** from the compile (a genuine
simplification), shrinks the constraint store from `N` Expanders to one descriptor
per family, *and* is the structure that lets challenges cache across rows.

## The `Family` model

```rust
/// One (variable, role) constraint family. The per-row Expander is reconstructed
/// on demand: row `ro` uses flat_lo = ro*ell, n_slots clamped on the last row
/// (and, for Identity, cid_base advanced by ro*ell).
struct Family {
    row_start: usize,    // var.row_start  — first witness row
    length: usize,       // var.length     — total slots across the family's rows
    ell: usize,
    template: Expander,  // the row-0 expander (flat_lo = 0)
}
impl Family {
    fn nrows(&self) -> usize { (self.length + self.ell - 1) / self.ell }
    /// Reconstruct row `ro`'s Expander — the exact inverse of the emit_* loops.
    fn row_expander(&self, ro: usize) -> Expander { /* per-kind: set flat_lo/n_slots */ }
    /// [cid_lo, cid_hi) this family's constraints occupy — for the challenge cache.
    fn cid_range(&self) -> (usize, usize) { /* per-kind */ }
}
```

## Compile change

`emit_id`/`emit_rowsum`/… drop the `for ro` loop and push **one** `Family`:

```rust
fn emit_rowsum(&mut self, var, cid_base, stride, coef_vec, ell) {
    self.families.push(Family { row_start: var.row_start, length: var.length, ell,
        template: Expander::Rowsum { cid_base, stride, coef_vec, flat_lo: 0,
                                     n_slots: ell.min(var.length) } });
}
```

`Constraints` carries `families: Vec<Family>` instead of `rows: Vec<Vec<Expander>>`
(quadratic side already compact). Memory: one descriptor per family, not per row.

## Fold change (the cross-row win)

```rust
for fam in families {
    let chal = compute_challenges(fam.cid_range());   // ONCE per family — cross-row dedup
    for ro in 0..fam.nrows() {                          // stream rows (chunk for memory)
        let e = fam.row_expander(ro);
        // fold e's terms using `chal` (gather, no re-hash) into acc, at cj[qi][row_start+ro]
    }
}
```

- **Cross-row dedup:** `challenge(base+i)` is identical for every row of a family
  (same `base`); compute the `k`-entry cache once and reuse across all `nrows` rows —
  the `R×` saving the per-Expander seam (PR #7) cannot get, which is the dominant
  remaining verify cost for Freivalds (the matmul). Captures `FreivaldsA` /
  `FreivaldsB`-transpose automatically (their challenges just gather from the cache).
- **Bounded memory:** stream a family's rows in chunks; the cache is `k` (small).
  This *replaces* PR #6's row-windowing with family-iteration + within-family
  chunking — same bounded footprint, cleaner unit.
- The per-Expander shortcuts (#7) fold in as how a family contributes its row, now
  reading cached challenges instead of recomputing.

## Gating (oracle preserved throughout)

1. **`row_expander` difftest:** for every kind, `Family::row_expander(ro)` emits the
   *identical* `(slot, cid, coef)` terms as the current per-row compile, for all `ro`
   — proves the reconstruction is exact, independent of any proof.
2. **Eager oracle:** keep the per-row `compile_claims` + `lin_contrib_emit` until the
   family fold is gated; compare verdict + per-check on the routing, E=8, and
   single-layer proofs.
3. **Cross-impl:** the Python verifier remains the end-to-end oracle.
4. **Delete last:** `Expander::emit`, `lin_contrib_emit`, the per-row `rows` store
   and the `for ro` loops go only after (1)–(3) pass at scale (the Spark A/B).

## Staging

- S0 (foundation): `Family` type + `row_expander`/`cid_range` + the difftest, for
  the uniform kinds. No live-path change.
- S1: `compile_families` producing `Vec<Family>` (drop the per-row loops), gated
  against `compile_claims` (expanded families == per-row Expanders).
- S2: family fold with the per-family challenge cache + within-family row chunking;
  gate vs eager on routing/E=8/single-layer.
- S3: delete the per-row store + `emit` + `lin_contrib_emit`; the strided-Freivalds
  cross-row dedup is now free. Run the full-model 92 GB verify once on the final
  verifier.

## Payoff

Simpler compile (no per-row loops), smaller constraint store (one descriptor per
family), the cross-row challenge dedup that finishes the matmul speedup, and a
single bounded streaming model — consolidating #6 and #7 rather than extending them.
