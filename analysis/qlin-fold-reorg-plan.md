# Plan: CUDA constraint-fold kernels + prover integration

Status: **design + partial prototype** (2026-06-26). Companion to
`qlin-family-object-reorg.md` (the build-`r_i` reorg design), `qlin-evenodd-fused-multiply.md`
(the multiply side), `prover-optimization-investigation.md` (the measured
prove-time breakdown), and `qlin-fold-kernel-bench-results.md` (the kernel
microbenchmark results from this work).

## 1. Goal

Replace the q_lin build-`r_eval` (`chunk_rTA`) step — today
`expand (torch) → cat → argsort → CSR → gl_spmv_challenged` — with a
constraint-family-direct CUDA fold that writes each family's contribution
straight into the dense `(n_chunk × ELL)` block, **no sort, no COO
materialization**. The output is bit-identical to today's `chunk_rTA`, so the
downstream `_interpolate_to_kdeg → poly_mul_batched` is unchanged and the change
is a pure prover-side optimization (cids/layout/verifier untouched).

Measured target: the `expand + sort` overhead is ~81 s of the ~340 s prove at
32L/SEQ-1000 (`prover-optimization-investigation.md`); removing it plus the
per-cid hashing wins are the prize.

## 2. Work done so far (this session)

A standalone microbench harness, `pipeline/qlin_fold_bench.py`, was built and run
on the Spark (GB10, sm_121). It is cheap to iterate because the build step is
**chunk-local**, **witness-independent**, and has a **bit-exact oracle** (the
current path). It synthesizes a representative chunk for one family, runs each
candidate kernel, checks `torch.equal` against the reference, and reports
ns/slot + GB/s. Full results in `qlin-fold-kernel-bench-results.md`.

Variants implemented + benchmarked, all **bit-exact** vs the reference:

- **per-slot** — one thread per output slot, closed-form `cid` + inline hash.
- **per-row / multirow** — one thread per witness row (or several): *loses* to
  per-slot (starves parallelism). Rejected.
- **warpcid** — one thread per slot, but each warp hashes its shared `cid` once
  and `__shfl`-broadcasts (per-lane fallback on the straddle warp). ~2× per-slot.
- **precompute** — hash the chunk's ≤`k` distinct cids once (`challenge_range`),
  then a pure `gather + mul + write` kernel. Bandwidth-bound. **~12× per-slot**
  for wide-reuse families. Timed as the *combined* cost (hashing + both launches),
  so the precompute is not hidden.

Families covered by the bench (bit-exact kernels, 13 of 14 expanders): **Freivalds
LF1B / LF2A / LF3C**, **Identity**, **stride-one-to-many fan-out**,
**stride-many-to-one**, **PerSlotVector**, **RowSumPerSlotVector**, **RoPEXRot**,
**RoPEX**, **TransposeO2M**, **CausalFilteredId**, and **CausalFilteredC2**. Only
**`L2_EmbedE`** remains (needs the inverse-index-vs-scatter design call). The
high-reuse ones (LF2A/LF3C/StrideManyToOne/RSV) now also have bit-exact **precompute**
(gather) variants — the recommended dispatch (LF2A/LF3C ~0.061 ns/slot, bandwidth-
bound; StrideManyToOne/RSV ~0.14, stride-16 reuse; vs ~0.69 per-slot).

### The dispatch rule (the key finding)

For the **build** step, cost ≈ memory(write `chunk_rTA`) + hashing(per cid).
Which dominates depends on the family's **cid-span** (how many slots share a cid):

| reuse (slots per cid) | example | best kernel | measured ns/slot | bound |
|---|---|---|---|---|
| high (`n`:1) | Freivalds on weights | **precompute** | 0.06 | bandwidth |
| none (1:1) | Identity (activation copies) | **per-slot** | 0.70 | hash |
| anti (1:`stride`) | softmax `z=c2−x`, routing gap | **per-slot** | ~0.70·stride | hash |
| warp-local | medium reuse | **warpcid** | 0.35 | mixed |

**precompute wins iff a cid is shared by many slots.** Zero/anti-reuse families
are irreducibly hash-bound, so per-slot is optimal there. So the production fold
dispatches by cid-span: precompute for the wide weight Freivalds (the witness
bulk, bandwidth-bound), per-slot for identity / fan-out / 1:1, warpcid for the
in-between. This also gives the per-family cost constants (`A ≈ 0.06` ns/witness
slot, `B ≈ 0.62` ns/distinct cid on GB10) that feed the analytical cost model.

## 3. Family coverage

`EXPANDERS` has 14 expander classes (`packets.py`; role table in
`qlin-family-object-reorg.md` §5). Status + planned dispatch:

| kind | role | reuse | dispatch | status |
|---|---|---|---|---|
| `L2_FreivaldsLF1B` | Freivalds B | high | precompute | **done (bench)** |
| `L2_IdentityScalar` | identity | none | per-slot | **done (bench)** |
| `L2_StrideOneToManyScalar` | fan-out | anti | per-slot | **done (bench)** |
| `L2_FreivaldsLF2A` | Freivalds A | high | precompute | **done (per-slot + precompute)** |
| `L2_FreivaldsLF3C` | Freivalds C | high | precompute | **done (per-slot + precompute)** |
| `L2_StrideManyToOneScalar` | shared cid (`cid//stride`) | high | precompute | **done (per-slot + precompute)** |
| `L2_PerSlotVector` | identity + vec coef | none | per-slot | **done (per-slot bench)** |
| `L2_RowSumPerSlotVector` | strided + vec coef | high | precompute | **done (per-slot + precompute)** |
| `L2_TransposeO2MScalar` | fan-out (transposed) | anti | per-slot (fan-sum, no atomics) | **done (per-slot bench)** |
| `L2_CausalFilteredIdScalar` | causal identity | none | per-slot (+mask) | **done (per-slot bench)** |
| `L2_CausalFilteredC2Stride` | causal fan-out (ragged) | anti | per-slot (ragged fan-sum) | **done (per-slot bench)** |
| `L2_EmbedE` | token-gated identity | none | legacy dispatch | **deferred** (see note) |
| `L2_RoPEX` | 2 cids/slot, ±cos/sin coef | none | per-slot | **done (per-slot bench)** |
| `L2_RoPEXRot` | 1 cid/slot, coef 1 | none | per-slot | **done (per-slot bench)** |

**`L2_EmbedE` — deferred (do not kernelize).** It is cold: one input-binding claim,
`SEQ·d` *linear* constraints (~3 ms at the demonstrated 1093 tokens, <1% of the
fold). The dispatch routes it to the legacy `expand→sort→spmv` path — bit-exact, and
it does not block end-to-end testing (random-init proves emit no `EmbeddingLookup`;
prompt-bound proves run it on the legacy path and still ACCEPT). **Eventual plan
(required, not just an optimization):** the current form assumes a publicly-shared /
subset-committed embedding (public `token_ids` + only-the-used-rows commit; the
`demo_llama7b.py:385` soundness note already flags that it needs a full-`E` Merkle
anchor). A deployable proof cannot assume the prover shares the embedding matrix, so
the lookup must eventually be verified against a **fully committed, hidden** `E` — a
Freivalds `x = S·E` (public selection `S`, hidden full `E`) or a committed
indexed-gather. That moves the embedding cost out of the cheap `L` term into `W`
(commit the full `V·d` table) and `Q`; it is a **claim + verifier redesign** tracked
separately from this prover-side fold port, and it needs verifier support for a
public-selection / committed-gather form (open question).

## 4. Phased plan

**P1 — remaining family kernels into the bench. Status: COMPLETE — 13 of 14
expanders gated bit-exact on GB10; `L2_EmbedE` deferred (note above).** For each
kind: read its expander, write the per-slot kernel (and precompute kernel where reuse
is high), add a synthetic chunk, gate bit-exact against the reference in
`qlin_fold_bench.py`. Lowest risk (isolated, gated). Deliverable: every (non-deferred)
family has a bit-exact direct kernel + a measured ns/slot, completing the dispatch
table.

**P2 — the `ConstraintBand` interface + dispatch.** Define the
`contribution(row_lo, row_hi, seed, label, ELL) → (n,ELL)` contract (one band per
`(variable, family)`; see `qlin-family-object-reorg.md` §1, §11); port each kernel
behind it; the cid-span dispatch (precompute / warpcid / per-slot) chosen at compile
from family metadata. Factor `challenge_at` / `challenge_range` out of
`gl_spmv_challenged` into reusable device primitives. Env-gated, default-off.

**P3 — prover integration.** Replace the compile's per-row packet store with the
count-pass + reverse-index (`variable → [band]`); run the fold variable-
stationary (`qlin-family-object-reorg.md` §9); replace `expand→sort→spmv` with
the dispatched per-family kernels; delete `_StreamingPackets` lazy compile and
`_late_qlin_var`. **Audit checkpoint A-pre (before) and A-post (after):** this
touches the streaming memory model and the lifetime systems; verify (a) bit-exact
`chunk_rTA` vs the current path on the full `K_DEG`, (b) no per-row packet store
(memory), (c) end-to-end ACCEPT on `demo_maverick_moe.py`.

*First moves (non-destructive, before any fold edit):* (i) a **premise check** —
build `variable → [band]` for the target config and confirm it is MB-not-GB and
fast (the per-row store was the reason for lazy compile; `qlin-family-object-reorg.md`
§7); (ii) the **A-pre oracle** — capture the current per-row path's `chunk_rTA`
(row-keyed field fingerprint, chunking-independent) as the regression gate every
later edit diffs against, env-gated default-off (`LIGERO_QLIN_AUDIT`).

*Forward-compat seams to leave for the quadratic fold (`p0-quad-on-last-variable.md`
§9; `p_0` itself stays deferred to its constraint-stationary pass for P3):*
1. **Claim-as-unit `last_use`** — bump a variable's `last_use` on *every* reference
   (data input *or* quad operand) so it is the latest claim touching it, and free
   **by claim**. Gives the quad fold its liveness + claim-grouped order with no new
   analysis. Avoid baking a data-only `last_use` into the free path (forces a re-open).
2. **Generic attachment** — `variable → [attachment]` so owned quads later attach the
   same way linear bands do (typed lists on the variable, not a hardcoded `.bands`).
3. **Retainable eval-domain rows** — don't write the fold as transform→multiply→free;
   keep a variable's transformed rows a retainable handle so `p_0` can reuse them and
   share the inverse-NTT (`qlin-evenodd-fused-multiply.md`). Leave
   `compute_p_0_streaming`'s inner kernel intact — only its scheduling moves.

(Pending: `p0-quad-on-last-variable.md` on `main` still uses the old
`ConstraintFamily` / `variable → [family]` names — rename to `ConstraintBand` /
`[band]` when this branch merges.)

**P4 — full-prover validation + re-measure constants.** End-to-end bit-exact +
ACCEPT on the reduced-Maverick config; wall-clock delta vs baseline; re-measure
`A`/`B`/`C` (the integrated `A` now includes the downstream NTT) and refresh the
cost model (`qlin-fold-cost-model.md`).

**P5 — structural cleanup / consolidation (after P1–P4 land and verify).** Goal: pay
down the structural debt the reorg exposes (and some that predates it), so the end
state is genuinely *simpler* than today — one way to do each thing, not two. The reorg
is net-subtractive; this phase makes sure the subtraction actually happens and the
residual forks are removed. High-level targets, each to be scoped as its own gated
change later:

- **Eliminate the sort / `spmv` from the prover.** `compute_r_at_eta` builds the same
  `chunk_rTA` as the fold (its `partial`), so it should share the bands' `build_r_eval`
  rather than its own `expand→sort→gl_spmv`; after that, `gl_spmv`/`gl_spmv_challenged`
  survive only inside the EmbedE fallback band — removed when EmbedE is resolved.
- **Resolve EmbedE.** Replace the token-gated / subset-committed embedding with the
  fully-committed hidden-`E` form (a claim+verifier change), removing the only family
  that still needs the legacy `expand→sort→spmv` fallback — one fold path for all.
- **Consolidate duplicate claims.** Confirm and merge the apparently two softmax claim
  implementations; audit the other claim types for similar drift.
- **Consolidate the verifier's claim logic**, currently spread across several places,
  into one per-claim location so prover and verifier share a single source of truth
  per family.
- **Delete the transitional scaffolding** — the band-decomposition gate and the A-pre
  audit oracle (keep at most one as a regression gate) — and any dead code the
  contraction leaves behind.

Compose with the multiply-side note (`qlin-evenodd-fused-multiply.md`) and the
already-merged inverse-NTT fuse (`LIGERO_FUSE_POLYMUL`), which are orthogonal
(they optimize `poly_mul`, P-here optimizes the build).

## 5. Validation strategy

The seam is the cleanest possible: `r_eval` is shape- and value-identical to
today's `chunk_rTA`, so every gate is a direct `torch.equal` at the fold
boundary, env-gated default-off, independent of the downstream multiply. The
verifier's per-family `lin_col` (already shipped, `feat/verifier-per-family`) is
the correctness oracle for the per-family contribution logic — the prover and
verifier expanders are kept bit-exact by the compile difftests.

## 6. Risks / open items

- **Expander → `contribution` rewrite is the bulk of the diff** — mechanical,
  per-role, 14 expanders. P1 de-risks it one family at a time.
- **Field-safe atomics** for the sparse/scatter kinds (causal, embed) and any
  cross-family overlap — Goldilocks add needs a CAS loop or a per-kind reduce;
  modular add is order-independent so it's bit-exact regardless of race.
- **Warp divergence** is mostly a non-issue (big families are single-type per
  warp); group descriptors by type per launch if needed (cheap — over
  descriptors, not nnz).
- **Upfront compile must be cheap once compacted** — verify the count-pass
  timing before relying on it (the per-row store was the reason for lazy compile).
- **No new prover observables** — keep any debug counters compile-gated out of
  the production fold.
