# Verifier streaming via a per-constraint-family fold

Status: **design note, not yet implemented** (2026-06-24). Synthesized from a
design discussion. The code references below are read from the current tree
(`verifier-rs/src/verify.rs`, `verifier-rs/src/handlers.rs`); the proposed
architecture is not yet built. Companion to `verifier-streaming-architecture.md`
(the higher-level streaming plan — this note is the concrete "lazy per-claim
compile" half of it) and to `qlin-family-object-reorg.md` (the prover-side
analog of the same family abstraction).

## Problem: the per-row constraint wall

The verifier recompiles the public claim list into a `Constraints` value whose
linear side is

```
rows: Vec<Vec<Expander>>      // one Vec<Expander> per witness row
```

i.e. **one `Vec<Expander>` per witness row, with all `m_total` rows resident at
once**. Measured ~99.6 GB at `m_total = 99.4M` rows (`full-model-v1-design.md`);
~100–130 GB projected for the seq=1093 run. It is `O(witness-rows)`,
**query-count-independent**, and it is exactly what `lin_col` iterates
(`(0..cons.rows.len()).into_par_iter()`). On a 121 GB box that floor is what
OOMs a sound full-model verify — and the dumped T=40 proof piles its opened
columns (~40 GB, see Memory) on top.

## Two facts that make a lean fix possible

1. **The linear test is additive.** `linear_column_test` is
   `par_iter().fold(..).reduce(+)` over the rows; the per-column accumulators are
   summed. So contributions can be accumulated **per constraint / per family**
   and the rows **streamed** — there is no need to hold them all at once.

2. **The quadratic test already does this.** `quadratic_column_test` iterates
   `cons.quadratic` **per constraint** over compact descriptors
   (`QuadraticConstraint { x_row, y_row, z_row, a, b, n }`), computes
   `challenge(ti, "quad")` **once per constraint**, and evaluates a **precomputed
   prefix-sum of the Lagrange basis**:

   ```
   prefix[qi][n] = Σ_{c < n} L_c(η_qi)        // built once
   mask          = prefix[qi][qc.n]            // sum of Lagrange over the first n slots
   ```

   It is the in-tree existence proof for the representation this note proposes
   for the linear side.

The quadratic side is per-constraint and compact because a quadratic constraint
is *local* — it touches a fixed handful of rows. A linear constraint can be
*dense* — a Freivalds row sums an entire contraction dimension — which is why
the linear side currently falls back to per-row Expanders. **The reorg makes the
linear side as compact as the quadratic side already is.**

## Design: a per-constraint-family fold, streamed

Replace the materialized `Vec<Vec<Expander>>` with **compact family descriptors,
folded as they are compiled** ("fold-as-you-compile"), along the natural
claim/layer seams. Rows are laid out per variable, in declaration order, so a
claim's rows are contiguous — there are no arbitrary `[lo, hi)` row windows to
pick; the unit is one claim (optionally batched up to a layer). Each family is
folded into the `T`-sized accumulators and then dropped.

### Precompute once per verify (shared by all families)

```
lag[qi][c] = L_c(η_qi)            // already built (lagrange_table)
pre[qi][n] = Σ_{c < n} lag[qi][c] // one scan -> any contiguous Lagrange-range sum,
                                  //   and causal/triangular (varying n) for free
```

### The trusted eval surface: three one-liners

```rust
fn at_point(qi, slot, coef)   -> F { coef * lag[qi][slot] }                 // identity / copy
fn at_range(qi, lo, hi, coef) -> F { coef * (pre[qi][hi] - pre[qi][lo]) }   // rowsum / causal (hi = i+1)
fn at_dense(qi, coefs: &[F])  -> F { dot(coefs, &lag[qi]) }                 // Freivalds / RoPE / strided
```

### Per-family fold (streamed)

```rust
fn fold_family(fam, cols, acc) {
    let chal = challenge_over(fam.cid_base, fam.k);   // BLAKE3 once per cid in the family's range
    for r in fam.rows() {                             // contiguous witness rows of one variable
        let s = chal[fam.cid_offset(r)];
        for qi in 0..T {
            let rval = match fam.shape(r) {
                Point{slot, coef}   => at_point(qi, slot, coef),
                Range{lo, hi, coef} => at_range(qi, lo, hi, coef),
                Dense{coefs}        => at_dense(qi, &coefs),
            };
            acc[qi] = add(acc[qi], mul(s, mul(rval, cols[qi][r])));
        }
    }
}   // fam's transient data dropped here -> bounded memory

let mut acc = vec![0; T];
for fam in families(claims) { fold_family(fam, &cols, &mut acc); }   // stream, fold, drop
for qi in 0..T {
    assert!(add(acc[qi], cols[qi][BLIND_LIN]) == poly_eval(q_lin, eta[qi]));
}
```

The math: each family's contribution to `acc[qi]` is
`challenge(cid) · (Σ_slot coef · L_slot(η_qi)) · cols[qi][row]`. The inner
`Σ_slot coef · L_slot(η)` is the only thing that varies by family shape, and it
reduces to the three evaluators above. Reading `cols[qi][r]` from more than one
family that touches row `r` is fine — the test is additive and per-slot fan-in
is small (1–3).

### Structural taxonomy (shape → evaluator)

| family kind | shape | evaluator |
|---|---|---|
| identity / copy | one slot, one cid | `at_point` |
| rowsum / stride-one-to-many (const coef, contiguous range) | range | `at_range` (prefix diff) |
| causal / triangular | range with varying `hi` | `at_range` (varying n — quad's `mask`) |
| Freivalds / RoPE (per-slot varying coef) | dense vector | `at_dense` (dot) |
| transpose-o2m / strided | strided set | `at_dense` (as a vector) — or an optional strided-prefix |
| fan-out (one slot, many cids) | challenge-range | `Σ chal[range]` (reuses `challenge_over`) |

So "structured-but-not-contiguous" is always one of: a varying-`n` range, a
dense dot, or a challenge-range sum — all covered by the three evaluators plus
the challenge table. There is no zoo of bespoke trusted evaluators.

## Memory model

- **Linear constraints:** ~100 GB of per-row Expanders → compact family
  descriptors + the `lag`/`pre` tables (`T × ncols`, small) + the `T`-sized
  accumulators. Bounded to one family/chunk at a time.
- **Opened columns (the proof data):** the `T` joint columns are `T × m_total`
  field elements ≈ `40 × ~130M × 8 B ≈ ~40 GB` at T=40. This is the *other* term
  that must fit (or also be streamed row-major). On a 121 GB box it fits
  alongside bounded constraint streaming; on a small box it does not — see
  prototyping.

## TCB analysis

- **The eval surface shrinks.** Today `handlers.rs` has a distinct `emit_*` per
  Expander kind plus per-kind evaluation; this collapses evaluation onto
  `{ at_point, at_range, at_dense }` — three auditable one-liners.
- **No new trust in the structural mapping.** `fam.shape(r)` / `fam.cid_offset(r)`
  is the same claim→constraint logic the current compile already contains,
  expressed as descriptors instead of materialized Expanders.
- **No new crypto.** Field, BLAKE3, Merkle, PRF unchanged. A bug yields REJECT,
  never a forged accept.
- **Gate bit-exact** against the current eager verifier (same proof → same
  verdict, byte-for-byte). Keep `at_dense` as the trusted reference for any
  strided fast-path: the optimization must equal the dense dot.

This is the rare case where the leaner, faster design is also the more
trustworthy one: less trusted eval code, bounded memory, and challenges
de-duplicated (computed once per cid-range via `challenge_over`, not once per
nonzero as `lin_col` does today).

## Relationship to other work

- `verifier-streaming-architecture.md` — this note is the concrete
  implementation of its "lazy per-claim compile (the substantive half)"; the
  family abstraction + Lagrange-eval dispatch is what makes the streaming both
  memory-bounded *and* lean.
- `qlin-family-object-reorg.md` — the prover-side analog (same family
  abstraction). The verifier compiles independently, so this is a separate Rust
  implementation of the same idea, not shared code.
- `quad_col` in `verifier-rs/src/verify.rs` — the in-tree existence proof for
  the per-constraint, prefix-Lagrange representation.

## Validation / prototyping plan

- **Implement behind a flag**, default to the current eager verifier.
- **Bit-exact gate** on small proofs: eager vs streaming → identical verdict,
  byte-for-byte, on the single-layer sound proof and a reduced-Maverick config,
  including the negative-test proofs (must still REJECT).
- **Dev runs on a CPU-only box** (the verifier needs no GPU): build `verifier-rs`
  on a small box (e.g. the 7 GB Hetzner orchestration host), develop the fold,
  and gate against the small local proofs there — leaving the Spark's prove run
  untouched.
- **The full-model verify needs a large-RAM host**, not the 7 GB box: even with
  streamed constraints, the ~40 GB of opened columns must fit. Run it on the
  Spark (121 GB, where the dumped proof already lives — no transfer) once the
  prove finishes, or on another large-RAM host. To run the full verify on a
  *small* box as well, the opened-column data would also have to be streamed
  row-major — a further extension.

## Open items

- The structural-mapping refactor (`emit_*` → family descriptors with
  `shape(r)`) is the bulk of the diff: mechanical, per-kind, and in the TCB.
- Confirm the fold granularity (per-claim vs per-layer) for the memory/time
  trade.
- Add a strided-prefix fast-path only if a hot strided family makes the
  dense-dot path too slow; gate it against the dense reference.
- Opened-column row-major streaming, if a small-box full verify is ever wanted.
