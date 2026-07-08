# Linear-fold unification — one run taxonomy, pluggable challenge access

**Status: IMPLEMENTED through Phase 4** (2026-07-04; designed 2026-07-02).
Phases 0–3 landed as PRs #15–#18 (gated by the two seq-1000 runs), the
inverse-NTT fuse as #20, and Phase 4 — the descriptor-interpreter kernel plus
the four bespoke irregular kernels, followed by deletion of the torch fold
path — in the irregular-kernels PR. Covers BOTH sides: the Rust verifier's
`lin_col`/`lin_sum` and the Python prover's q_lin fold. Absorbed Piece 1 of
`qlin-fold-simplification.md` (the dedup challenge buffer → `ChalSource`) and
implemented its Piece 2 as the interpreter kernel.

## Problem

The linear-constraint geometry (flat slot index → constraint id, coefficient) is
written three times in three taxonomies that don't align:

| copy | where | form |
|---|---|---|
| prover expand | `prover/packets.py` | 14 `L2_*` dataclasses + 14 torch `_expand_*` fns |
| verifier compile | `verifier/src/compile.rs` | 15 `Expander` variants + per-term `emit` |
| verifier fold | `verifier/src/verify.rs::row_contrib` | 4 hand-duplicated fast paths + emit fallback |

The fast paths exist to hoist the BLAKE3 challenge out of slot runs that share a
cid; the prover's `_BAND_DISPATCH` (scatter/fan/spmv) is a second, different
partition of the same kinds; and the challenge PRF is invoked once per *nonzero*
on the prover and on every verifier kind without a fast path. Each new claim type
or optimization adds another divergent copy.

## Key observation

Every linear family decomposes into **maximal homogeneous runs**, and each run has
one of four shapes. The shape is a static property of the family kind — no runtime
probing — and it dictates both the optimal challenge access and the memory story:

| run shape | slots ↔ cids | families | challenge access | dedup win |
|---|---|---|---|---|
| **repeat** | run of slots → 1 cid | rowsum(-const), table-settlement sum, FreivaldsB (¬T, per run), FreivaldsC (per run) | `at(cid)` once per run | terms/cid = run length |
| **strided repeat** | cid = base + f mod k; k distinct cids recur m× with no contiguous runs | FreivaldsA, FreivaldsB (transposed), FreivaldsC (across runs: h distinct) | `preload([base, base+k))` + gather | terms/cid = m (or n), span ≤ 16 K cids |
| **1:1** | slot i → cid base+i | identity, weighted, causal-id, rope-rot, embed | `at` per term | none possible (terms/cid = 1) |
| **fan** | 1 slot → contiguous cid range | strideO2m, transposeO2m, causal-c2 | `sum(lo, hi)` streamed | none possible (terms/cid = 1); needs only the range **sum** |

Consequences:

- **Repeat + strided-repeat** carry all the achievable hash dedup. Strided repeat
  is the Freivalds case run structure alone misses — its spans are small and dense
  (`[base, base+k)`, k ≤ H·K ≈ 16 K → ≤ 128 KB), so `preload` handles it at
  negligible memory. (The ρ/λ projection vectors themselves are *already* deduped:
  `op_vec` once per claim, `Arc`/tensor-shared. Not in scope.)
- **1:1** gains nothing from precomputation on the CPU verifier (one hash per
  distinct cid is the floor either way); on the GPU prover a windowed buffer is
  still worthwhile purely for *batching* (one `challenge_range` kernel vs
  per-element PRF) — that is Piece 1 of the qlin doc, unchanged.
- **Fan** is the only shape where a naive "buffer the span" blows up (MaxClaim's
  `vstar` broadcast: T·V cids ≈ 1.3×10⁸ at Maverick scale ≈ 1.05 GB). But each
  fan cid is consumed exactly once and only the range **sum** is needed, so a
  tiled/streamed `sum(lo, hi)` does it in O(tile) memory with the same
  (irreducible) hash count.

So the generic interface is not "buffer vs hash-on-the-fly" — it is a challenge
source with three primitives, and a per-kind static choice of which to use.

## Interface sketch

### Rust verifier

```rust
/// Challenge access for the linear fold. One implementation; the *call pattern*
/// varies per family kind (statically, like _BAND_DISPATCH).
struct Chal<'a> { s_comb: &'a [u8] }
impl Chal<'_> {
    fn at(&self, cid: usize) -> u64;                  // 1 BLAKE3
    fn sum(&self, lo: usize, hi: usize) -> u64;       // streamed range sum, O(1) mem
    fn preload(&self, lo: usize, hi: usize) -> Vec<u64>;  // small dense spans only
}

/// Coefficient source within a run.
enum CoefSrc<'a> {
    Const(u64),
    Slice(&'a [u64]),                  // per-slot vector (weighted, rowsum coef_vec)
    Sep { a: u64, b: &'a [u64] },      // coef[s] = a·b[s]  (FreivaldsC: λ_i × ρ_j —
}                                      //  hoists the λ mul out of the slot loop)

/// One maximal homogeneous piece of a family's row window.
enum Run<'a> {
    Repeat   { slot_lo: usize, len: usize, cid: usize,     coef: CoefSrc<'a> },
    OneToOne { slot_lo: usize, len: usize, cid_lo: usize,  coef: CoefSrc<'a> },
    Fan      { slot: usize,    cid_lo: usize, len: usize,  coef: u64 },
    Term     { slot: usize,    cid: usize,                 coef: u64 },  // embed, rope-x
}

impl Expander {
    /// NEW: yield the window's runs. Same index math as emit, coarser granularity.
    fn for_runs(&self, flat_lo: usize, n_slots: usize, f: &mut impl FnMut(Run));
    /// KEPT verbatim as the difftest oracle (line-for-line with Python).
    fn emit(&self, flat_lo: usize, n_slots: usize, f: &mut impl FnMut(usize, usize, u64));
}
```

`row_contrib` becomes ONE generic fold over `for_runs`:

## TCB impact

The verdict path today carries TWO copies of the family geometry: `Expander::emit`
(~120 lines, the fallback arm) and the four `row_contrib` fast paths (~80 lines of
the trickiest index math in the crate), with their equivalence established only by
tests. After Phase 1 the verdict path has exactly ONE copy (`for_runs`) + a
~40-line generic fold + ~30 lines of `Chal`. `emit` survives only as the difftest
oracle, off the verdict path — it stops being TCB. No new crate dependencies
(`Chal` is pure functions over the existing `challenge()`).

The run taxonomy also exposes variant merges that emit identical (cid, coef)
streams, shrinking the enum itself (15 → ~12), each merge bit-exact by
construction and landed as its own gated micro-commit:

- `Identity` ≡ `RowsumConst { stride: 1 }`
- `StrideO2m` ≡ `TransposeO2m { cols: 1, fan: stride }`
- `FreivaldsA` ≡ transposed-`FreivaldsB` with λ↔ρ (both decode `i_k = f % k`,
  coef vector indexed by `f / k`) → one "strided Freivalds side" variant

Honest accounting: total geometry lines drop modestly (~200 → ~170 on the verdict
path); the real reduction is structural — one path instead of two per kind, so an
audit reads one fold rather than four hand-optimizations plus a fallback plus the
argument that they agree.

- `Repeat`: challenge once (from `at` or the family's preload); slot side via the
  existing prefix-Lagrange table (`Const`) or a coef·Lagrange dot (`Slice`/`Sep`).
- `OneToOne`: per-slot `at`·coef·L — today's fallback, now recognized statically.
- `Fan`: `coef · L_slot · sum(lo, hi)`.
- `Term`: as today.

The four hand-written fast paths are deleted; `linear_constraint_test`'s rhs loop
(`verify.rs:275-281`) is already a contiguous challenge-range sum and switches to
`Chal::sum` for free. Whether a kind uses `preload` is a static per-kind table
(Freivalds A/B/C: preload `[base, base+k)` / `[base, base+h)` once per family
visit).

### Python prover

```python
class ChalSource:                      # backs the s_comb "lin" challenges on GPU
    def range(self, lo, hi) -> Tensor          # one batched-BLAKE3 kernel (Piece 1)
    def gather(self, cids: Tensor) -> Tensor   # reads the window/preload buffer
    def range_sums(self, lo, hi, stride) -> Tensor  # tiled fan sums
```

`_band_contribution`'s three modes read the source instead of hashing inline
(`challenge_at`); Freivalds band templates cache their `preload` across chunks
and layers (valid for the whole prove: `s_comb` is fixed once per protocol run).
The 14 packet kinds get the same run-shape annotation; the `_BAND_DISPATCH`
scatter/fan/spmv split is re-derived from it rather than maintained by hand.

## Back of the envelope

Assumptions (all ±3×; treat ratios as the reliable part): CPU BLAKE3 challenge
≈ 80 ns/core, rayon ~20 cores → ~4 ns amortized; CPU field mul 1–2 ns/core; GPU
batched challenge ≈ 0.6 ns (measured, paper §A.5). Reference configs:
**R1** = Llama-2-7B, 32 layers, S = 1000 (W ≈ 3×10¹⁰, linear nnz ≈ 5–8×10¹⁰);
**R2** = Maverick 48 layers, S = 1093 (W ≈ 9.4×10¹¹, L ≈ 1.8×10¹¹,
nnz ≈ 2–3×10¹²).

### Hash counts

- **Verifier hash floor** (unchanged by design): one hash per distinct cid ≈ L.
  R1: ~10¹⁰ → tens of seconds. R2: 1.8×10¹¹ × 4 ns ≈ 12 min.
- **FreivaldsA dedup (strided repeat → preload).** R1: Σ m·k over matmuls
  ≈ 7×10⁷/layer × 32 ≈ 2.3×10⁹ hashes today → Σk ≈ 2.3×10⁶ after (~1000× on that
  share; ~9 s @ 4 ns). R2: the expert matmuls' A-side (the S×d token stream read
  by all 128 experts × 3 matmuls × 24 layers) ≈ 4.7×10¹⁰ hashes → 4.7×10⁷
  (~3 min saved @ 4 ns).
- **Prover (per 256-row chunk, R1):** nnz/chunk ≈ 4–8 M hashes today →
  distinct ≈ 2–4 M, one batched kernel (~2–4 ms), with Freivalds bands ~500×
  smaller on their share and weight-side B bands ~n× smaller (already run-level
  on the verifier, not on the prover).

### Memory

| item | size | note |
|---|---|---|
| verifier preload buffers | ≤ 128 KB per family, one live at a time | span ≤ H·K ≈ 16 K cids |
| verifier fan streaming | O(tile) ≈ 0.5 MB | avoids the 1.05 GB naive MaxClaim buffer |
| verifier Lagrange/prefix tables | ~5 MB at T = 80 | existing, unchanged |
| prover R window (1:1 kinds) | ≤ 17 MB/chunk (2.1 M cids × 8 B) | Piece 1 buffer |
| prover Freivalds preload cache | ≤ ~128 KB × live bands | cached across chunks/layers |
| prover triples (today) | ~100–200 MB transient/chunk (24 B × 4–8 M nnz) | removed only by Phase 4 (fused kernels) |

Net: verifier memory unchanged; prover Piece-1 memory ≈ neutral (+≤20 MB, triples
still present), Phase 4 removes the triples.

### Relative runtime

| component | today | expected after | basis |
|---|---|---|---|
| prover q_lin fold | 25–50 % of prove, dominated by organizing overhead (qlin doc §Motivation) | approaches NTT-bound; ~1.15–1.35× whole-prove with Phase 4, less with Piece 1 alone | banked measurements + qlin doc estimate |
| verifier lin_col @ T=4 | hash-dominated (~2 h of the 3 h Maverick T=4 verify) | ~1.2–1.5× | A-side + FreivaldsC-run dedup are the duplicated share; 1:1 floor unchanged |
| verifier lin_col @ T=80 | T·mul-per-term dominated | ~1.0–1.1× | hashing is a minor share |
| verifier code | 4 fast paths + fallback + difftests per path | 1 fold + 1 oracle | — |

The honest summary: **the prover gets the runtime win, the verifier gets the
structure win** (plus a real but secondary speedup at low T, which is the
development configuration).

## What this deliberately leaves alone

- **λ·ρ per-slot products** in FreivaldsC: field muls, not hashes; ~10⁹–10¹⁰ at
  scale but ns-each and GPU-trivial. The `Sep` coef source hoists the λ factor per
  run for free; nothing further.
- **Cross-family duplicate hashing** of shared cid ranges (e.g. MaxClaim's gap
  identity + vstar fan walk the same range twice): constant factor 2–3× on a small
  slice; a claim-scoped shared buffer would reintroduce the big-span problem
  exactly where spans are big. Accept.
- **Python↔Rust twin-ness** stays a convention enforced by difftests — but with
  one shared kind taxonomy and run vocabulary, auditable field-by-field.
- The larger LinOp / per-variable restructuring and the apply→iNTT fusion
  (qlin doc, out of scope there too).

## Correctness gating — bit-exactness is the gate

Bit-exactness is not just desirable here, it is *available*: every transformation
in this plan is a reassociation or hoisting of Goldilocks field ops, which are
exactly associative/commutative/distributive mod P — unlike floats, no reordering
can change a bit. The challenge PRF is untouched. Any value drift anywhere is a
bug by definition, so the gates demand equality, not "still accepts."

Asymmetry that sets the gate weight per side: the Python prover is NOT in the TCB
— a prover bug yields a spurious REJECT (liveness, caught by gate.py), while a
verifier bug is a potential soundness hole. Verifier phases carry the full stack
(1–3 below); prover phases lean on 4–5.

1. **Property difftests (all shapes):** `for_runs ≡ emit` per kind over
   adversarial geometries (partial last rows, windows straddling run boundaries,
   transpose_b, causal partial triangles, h/kk edge cases). `emit` stays
   hand-written, line-for-line with the Python generators — an oracle independent
   of the new code. During Phase 1 the four deleted fast paths are temporarily
   kept as ADDITIONAL test oracles (new fold ≡ old fast path on random inputs),
   deleted one release later.
2. **Bit-exact corpus gate (real shapes):** re-verify archived proof dumps
   comparing not just the verdict but intermediate check values — a debug env
   flag dumps lin_col's per-query accumulators and the poly_eval targets; a
   harness diffs old-binary vs new-binary output on the corpus. (Verdict-only
   comparison is too weak on accepting proofs.)
3. **Negative gate (rejects still reject):** tampered proofs — corrupt one opened
   column value, one q_lin coefficient, one rhs value — must REJECT with
   identical per-check booleans before and after. This catches the failure mode
   accept-side bit-exactness cannot see: a fold bug that makes a check vacuous.
4. **Prover tensor + byte gates:** `LIGERO_QLIN_BANDCHK` (chunk_rTA equality vs
   the legacy path), plus determinism end-to-end: with a fixed seed the dumped
   proof bytes must be IDENTICAL before/after (hash the JSON).
5. **End-to-end:** `demo/gate.py` PASS, and one full-scale re-verify before the
   legacy path is deleted.

## Migration plan

Each step lands separately, gated as listed. Rust first — most self-contained,
and it fixes the canonical taxonomy the Python rename then follows.

**Phase 0 — dead code + docs (zero risk).**
Strip `protocol.py`'s ghost banners of the retired compile (lines ~180–330);
fix stale docstrings (`L2_StrideManyToOneScalar`'s "future" table-settlement
note); mark difftest-only items (`Constraints::m_total`, `Family::emit_global`)
and the unused `eval_zeta_form`. *Gate:* `cargo test`, `run_tests.py test_claims`.

**Phase 1 — Rust verifier.**
1. `Chal` (`at`/`sum`/`preload`); switch `linear_constraint_test`'s rhs loop to
   `sum`; add the debug check-value dump flag (gate 2's harness). *Gates 1–2.*
2. `Expander::for_runs` for all kinds. *Gate 1* (`for_runs ≡ emit`).
3. Rewrite `row_contrib` as the generic run fold (old fast paths retained as
   test oracles, off the verdict path); Freivalds kinds get preload; matmul
   p-side switches `Rowsum{ones}` → `RowsumConst` (same cids/coefs, drops the
   materialized ones-vector). *Gates 1–3 + 5*; benchmark lin_col before/after
   (guard against per-run dispatch overhead).
4. Variant merges (`Identity`→`RowsumConst{1}`, `StrideO2m`→`TransposeO2m`,
   `FreivaldsA`→strided-`FreivaldsB`), one micro-commit each. *Gates 1–2 each.*

**Phase 2 — Python prover (Piece 1 of the qlin doc, in run vocabulary).**
1. `ChalSource.range` kernel + window buffer; scatter/spmv read it instead of
   `challenge_at`. *Gate:* `LIGERO_QLIN_BANDCHK` (band fold vs legacy, final
   chunk_rTA equality), `gate.py`, negative tests.
2. Freivalds band preload, cached across chunks/layers within a prove.
   *Gate:* BANDCHK + gate.py; measure hash-time drop.
3. Fan-mode tiled `range_sums` where coef is constant. *Gate:* BANDCHK.

**Phase 3 — taxonomy alignment + deletion.**
Rename both sides' kinds to one canonical list (documented in `CLAIM_SPECS.md`);
one mechanical commit per side, difftests between; then delete
`_chunk_rTA_legacy` + the BANDCHK path once a full-scale run (≥ 7B seq-1000
prove→verify) is green.

*Phase-3 outcome (2026-07-03):* deletions done (prover `_chunk_rTA_legacy` /
`_by_kind_of` / `LIGERO_QLIN_BANDCHK`; verifier `row_contrib_fastpaths`); mass
RENAMES consciously skipped — true name convergence requires the Python-side
variant merges (Identity/StrideO2m/LF2A equivalents), which belong with the
Phase-4 kernel work; renaming without merging churns every file while leaving
the taxonomies structurally different. The cross-language mapping is instead
pinned here:

| Python (packets.py) | Rust (compile.rs) | run shape | challenge tier | coef source |
|---|---|---|---|---|
| `L2_IdentityScalar` | `RowsumConst{stride:1}` | 1:1 | per-term / window | const |
| `L2_PerSlotVector` | `Weighted` | 1:1 | per-term / window | vector |
| `L2_RowSumPerSlotVector` | `Rowsum` | repeat | one per run | periodic vector |
| `L2_StrideManyToOneScalar` | `RowsumConst` | repeat | one per run | const |
| `L2_StrideOneToManyScalar` | `TransposeO2m{cols:1}` | fan | range-sum | const |
| `L2_TransposeO2MScalar` | `TransposeO2m` | fan | range-sum | const |
| `L2_CausalFilteredIdScalar` | `CausalId` | 1:1 (triangular segments) | per-term | const |
| `L2_CausalFilteredC2Stride` | `CausalC2` | fan (ragged) | range-sum | const |
| `L2_EmbedE` | `Embed` | 1:1 (token-scattered) | per-term | const −1 |
| `L2_FreivaldsLF1B` | `FreivaldsB{¬transpose}` | repeat | preload [base, base+k) | ρ slice |
| `L2_FreivaldsLF2A` | `FreivaldsB{transpose}` | strided repeat | preload [base, base+k) | λ, head-const |
| `L2_FreivaldsLF3C` | `FreivaldsC` | repeat | preload [base, base+h) | −λ·ρ (Sep) |
| `L2_RoPEXRot` | `RopeXrot` | 1:1 (cid step 2) | per-term | const 1 |
| `L2_RoPEX` | `RopeX` | 1:1 ×2 (cid step 2) | per-term | ±cos/±sin |

**Phase 4 — deferred.** Fused CUDA kernels (qlin doc Piece 2) consuming the run
descriptors; the future verifier CUDA port shares the same taxonomy.

*Phase-4 outcome (2026-07-04):* implemented in the DESCRIPTOR-ALGEBRA form.
Ten regular kinds lower to a 24-slot u64 descriptor interpreted by ONE kernel
(`k_interp_band`: ≤4-digit mixed-radix decode → cid/coef by strided dots and
table gathers, incl. −(λ·ρ) for Freivalds C → fan-axis accumulation in
registers → in-place rTA, one thread per slot, no atomics); the four irregular
kinds (causal ×2, embed, rope-x) got small bespoke kernels on the same
contract. Challenges: preloaded spans for the Freivalds kinds (ChalSource),
in-kernel PRF otherwise. The inverse-NTT fuse (banked `205f66c`) was ported
and defaulted on. After gating every kind against the torch oracle
(`LIGERO_KERNEL_CHECK`, both model shapes), the ENTIRE torch fold path was
deleted (~850 → 339 lines in packets.py; the expander fns, `_band_contribution`,
`_spmv_one_band`, `_BAND_DISPATCH`, and both env flags) — the standing
correctness anchor is the independent Rust verifier + its emit oracle;
resurrect the torch oracle from git history for future kernel work. The
taxonomy renames became moot: the canonical cross-language form IS the
descriptor; the packet dataclasses remain as the compile-side band templates
that lower to it.

## Proposed target execution model: liveness-driven fold (2026-07-03, James)

Proposal (rough form, refined below): the tape holds variables + claims; each
claim instantiates its aux variables and lists its linear/quadratic constraints;
each variable carries a `last_use` field (last claim involving it). Round 3
walks the tape in order, generating witness variables as they appear and
attaching small `constraintBand` objects (compact descriptor + an expander,
Rust or CUDA) to each involved variable; quads fire at their declaring claim
(when the last operand exists); at the end of each claim, any live variable
whose `last_use` is this claim evaluates its accumulated bands into q_lin and
frees.

Assessment: sound, and it is the endpoint the run-taxonomy work builds toward.
It fixes claim-major folding's flaw (rows touched once per consumer) by
accumulating bands and folding ONCE at variable death. Completeness: all of a
variable's contributions come from claims involving it, so the band set is
complete at last use; quad operands are fields of the declaring claim, so quads
fire before any operand dies. Bit-exactness gates survive: cids stay in tape
order, so every r_lin value is unchanged, and field add commutes — death-order
folding yields the same q_lin/p_0 bits (BANDCHK-style equality still applies).

Refinements required:

1. **Settlement exception** (already latent in `_StreamingPackets`): naive
   last_use keeps every LogUp-inverse variable alive to the settlements at tape
   end. Sum-side bands are constant-coef RowsumConst with count-pass-computable
   cids → attach eagerly at variable creation; last_use excludes settlements
   for sum-side vars. (w/mult vars genuinely live at the settlement — small.)
2. **Decouple fold-eligibility from batch flush**: liveness decides when rows
   BECOME foldable; dead rows buffer and flush at ≥256 for healthy iNTT/poly-mul
   batches. Resolves the qlin doc's fragmentation objection to variable-aligned
   work.
3. **One metadata-only count pass survives** (settlement cid bases + last_use);
   cid numbering itself becomes incremental with tape-order execution.
4. **Value retention is mostly already paid**: witness values must live
   producer→last-consumer regardless; the new state is the compact bands (tiny)
   plus optionally cached encodings for pending quads (or re-encode on demand,
   as p_0 streaming does today). Audit for pathologically long liveness spans
   at tape build.

Consequences: kills the global band index and per-chunk regroup; unifies linear
and quad folding under one lifecycle (= the quad lift below, from the execution
side); the STREAMING VERIFIER runs the same walk with bands evaluated at the T
opened points at death — prover and streaming verifier become one program shape
with different band evaluators; and it is the natural driver for the
prove-claims-one-at-a-time protocol change (README future work). Phases 1–2 are
unchanged and prerequisite — the run descriptors + Chal/ChalSource are exactly
the band-evaluation layer this consumes; this reframes Phase 4.

### Variant: variable-major with an upfront band pass (recommended first)

Same band/evaluator layer, different driver: a metadata-only first pass attaches
every band to its variable (settlement bands included — no special case), then
the sweep folds each variable's LINEAR side AT GENERATION (all bands already
known) and frees its rows immediately. This is the current `_StreamingPackets`
architecture's shape with the cleanup applied — the smallest delta from code
proven at 400B scale.

Liveness accounting (three separate lifetimes; only the third differs between
drivers):

1. **Values** — needed to compute later claims' witness. Liveness-managed in
   EVERY scheme, and already implemented exactly as the proposal sketches:
   `tape.py` computes `last_use[v]` over the claim list and pops the `live`
   dict at the last consuming claim. A witness-generation concern, not a fold
   concern; identical across drivers.
2. **Encodings** — needed at generation (commit) and again for quad products.
   Quads fire at their DECLARING claim in both drivers (core.py fires quads
   at :1947 BEFORE the claim's frees at :1953). Operand values are live there
   by a per-claim-type CONVENTION, verified across the current claim set:
   every quad operand is either an external input the claim's COMPUTE_FN
   reads (∈ input_vars → last_use ≥ this claim) or an own variable the claim
   computes (unique function of inputs + challenges, §3.3). The quad shapes
   themselves vary (product pins, booleanity m·m=m, LogUp inverses, slack
   brackets, muxes) — "x·y = z with x,y known" is only the common case.
   The convention is not framework-enforced; a violation is a CRASH-class
   prover bug, not a soundness bug (the verifier recompiles independently).
   Hardening, free in the variable-major first pass: last_use[v] =
   max(input_vars use, last claim whose QUADS reference v). Do NOT extend to
   all variable fields — settlements reference every LogUp-inverse var at
   tape end but never need their VALUES (sum side is linear over committed
   rows; multiplicities accumulate as a histogram), so all-fields liveness
   reintroduces the keep-alive explosion. Encodings are either re-derived on
   demand (deterministic: ZK slack is PRG-seeded by absolute row index —
   today's p_0 streaming) or retained via a STATIC keep-mark from the first
   pass — a memory-vs-iNTT trade, no runtime tracking either way.
3. **Fold state** (bands + rTA) — the only genuine driver difference:
   variable-major keeps it liveness-free (fold at generation); the liveness
   walk schedules it by death. The liveness proposal is thus "extend the
   existing per-value last_use to also schedule the fold" — a small step to
   take later if the per-claim protocol wants it.

Memory of the upfront index: band descriptors ≈ 10–15/claim × ~10⁴ claims
(Maverick) × ~100 B ≈ **10–20 MB** (7B: ~1 MB) — noise against the 20–80 GB
witness working set. One trap: eagerly materializing op-challenge vectors (ρ/λ)
inside bands is ~0.6–1 GB at MoE scale (9,216 expert matmuls × ~75 KB); store
`(claim_index, label, length)` and re-derive per row-range entry (PRF, one
batched kernel) or cache per active range. LogUp w-side coef vectors (128 MB at
2²⁴) derive at settlement-sweep time, one at a time.

Hash dedup is UNAFFECTED by the driver choice — per-run hashing, preload spans,
and the chunk-window buffer all live in the band evaluator; variable-contiguous
rows give preload caches their best possible locality (each band's challenge
window touched once, contiguously). Keep the fixed-size batch flush across
variable boundaries for healthy NTT batches.

Decision: land variable-major first; the liveness walk remains the evolution
path when the per-claim interactive protocol or the streaming verifier wants
it — swapping drivers does not touch the band layer, and both drivers produce
bit-identical q_lin/p_0 (commutative fold, same cids).

### Synthesis: constraint objects on variables (2026-07-03, James — settled target)

The refined form merging both variants. Tape construction records variables +
claims; each claim instantiates its aux variables and lists its linear and
quadratic constraints. A metadata-only NUMBERING PASS after tape construction
assigns cid bases and (positional) quad indices, attaches to each VARIABLE the
`linearConstraintBand` objects it participates in, stores each quadratic
constraint's single copy on its LAST-INSTANTIATED operand, and computes
per-variable `last_use` = max(positions of quads referencing v, positions of
variables whose compute-or-aux reads v) — linear bands deliberately excluded
(their fold needs only the variable's own rows + PRF-derivable coefficients).

Round 3 walks the tape constructing variables in order (per-claim compute in
practice, variable-ordered within); at each variable's instantiation it fires
the stored bands (rows → fold-eligible; flush at ≥256 for batch health) and any
quads stored on it (operand values live by the last_use rule; encodings
re-derived on demand — deterministic PRG slack — or retained via static
keep-marks from the numbering pass); then frees values whose last_use has
passed.

Two structural consequences (2026-07-03 follow-up):
- **Family dissolves as a freestanding record.** With bands owned by variables,
  the geometry fields (row_start/length) are redundant — a band reduces to the
  expander part (kind, cid_base, coef source), an EDGE in the variable graph.
  The global family list stops being the fold's iteration structure; band
  evaluation becomes variable-relative (drops the chunk-absolute coordinate
  gymnastics; only last-row clipping remains). The expander itself — the
  irreducible index math — is what survives. (The non-streaming Rust verifier
  keeps its compiled Constraints list as the audit seam; the streaming verifier
  adopts the variable-attached form.)
- **Individually scheduled, batch-executed.** Cross-variable chunking is
  removed as an ORGANIZING principle (no chunk regroup / row-keyed lookup /
  chunk coordinates) but survives demoted to a flush buffer in the arithmetic
  backend: fired rows accumulate to ≥256 before the batched iNTT/poly-mul —
  aggregating tiny variables (per-claim scalars: 1 row) and splitting huge ones
  (softmax S² witness: unbounded at frontier context). Invisible to the
  semantic layer; bit-exact either way (commutative accumulation).

Storage and execution decisions (2026-07-03 follow-up):
- **Bands ARE expander descriptors** — no wrapper object. One enum/dataclass:
  kind tag + cid_base + kind params, ~60–100 B. Challenge vectors stored as
  PRF RECIPES `(claim_index, label, length)`, re-derived at evaluation (kills
  the Arc-sharing and the ~1 GB eager-ρ/λ trap); preloads/materializations are
  transient evaluation state, never stored. Shared public objects (token_ids,
  T_Y) stay references; RoPE cos/sin derive from config scalars.
- **Execution via one row-stream sink** (`push(rows, polys)` → internal
  ≥256-row buffer → batched iNTT/poly-mul/rowsum; ~50–100 lines, ~48 MB held).
  Large variables force the SPLIT direction anyway (frontier S² rTA can't
  materialize at once), so the aggregate direction is marginal lines, not a
  second architecture. Per-variable-immediate is also correct and only costs
  O(#variables) launch/dispatch overhead (~seconds-to-a-minute at Maverick) —
  a performance knob, not architecture. Large variables are runtime-unaffected
  either way; the pre-existing S² VALUE materialization binder (paper §6.1) is
  orthogonal and unchanged by this restructure.

Execution details (2026-07-03 follow-up): the fold splits into two stages with
opposite uniformity. STAGE A (band evaluation, rTA[slot] += chal(cid)·coef) is
kind-specific but per-band and never cross-variable — each launch is internally
uniform; today's `by_kind` regroup exists only because the current chunk fold
batches stage A ACROSS variables, and the target deletes the need. STAGE B (the
iNTT → poly-mul → accumulate) is kind- and variable-agnostic — the flush buffer
batches only this stage, so CUDA uniformity is free. Cross-variable-chunking
complexities that remain: sink owns pushed rows until flush (~48 MB pin);
bands must evaluate over row windows (required anyway for huge variables);
debug tracing wants a flush-on-demand. Do NOT batch stage A across variables —
that reintroduces the regroup.

Challenge-hash timing (expanders name cids; the challenge source serves them;
tier chosen statically per kind):
1. **Band-scoped** — once per band activation, cached across the variable's
   windows: strided-repeat preloads (Freivalds ≤128 KB), single-cid bands
   (settlement sum: one hash reused by millions of slots).
2. **Window-scoped** — per flush window: one `at` per contiguous run;
   `challenge_range` over the window's cid span for 1:1 kinds.
3. **Boundary duplication (accepted)** — runs straddling window edges re-hash
   one cid per crossing (stride ∤ ELL, e.g. softmax M=seq); O(#windows) total.
   Cross-band cid-range sharing stays duplicated 2–3× on a small slice.

Sink pseudocode (the whole flush machinery — note how little there is):

```python
class QLinSink:
    """Stage B, variable-agnostic. Bit-exact for ANY push order/granularity
    (field add commutes), so flush timing is pure performance policy."""
    def __init__(self, cfg, chunk=256):
        self.rta   = alloc(chunk, cfg.ELL)      # pending rTA rows
        self.polys = alloc(chunk, cfg.K_DEG)    # matching encoded rows
        self.n, self.chunk = 0, chunk
        self.q_lin = zeros(2 * cfg.K_DEG - 1)

    def push(self, rta_rows, poly_rows):        # any number of rows
        while rta_rows:                          # split OR aggregate, same loop
            take = min(self.chunk - self.n, len(rta_rows))
            self.rta[self.n:self.n+take], self.polys[self.n:self.n+take] = \
                rta_rows[:take], poly_rows[:take]
            self.n += take
            rta_rows, poly_rows = rta_rows[take:], poly_rows[take:]
            if self.n == self.chunk:
                self._flush()

    def _flush(self):                            # 3 batched kernels, in-stream
        coeffs = intt_batched(self.rta[:self.n])            # ELL vals → K coeffs
        prods  = poly_mul_batched(coeffs, self.polys[:self.n])   # degree < 2K
        self.q_lin = gl_add(self.q_lin, sum_rows(prods))
        self.n = 0

    def finalize(self):
        if self.n: self._flush()                 # partial last batch
        return self.q_lin                        # caller adds the blinding row

def fire_variable(v, live, chal, sink, merkle, q_irs, p0):
    vals = live[v]                               # just built by the walk
    for lo, hi in windows(v.n_rows, WINDOW):     # WINDOW ≥ chunk; splits big vars
        rows = encode_rows(vals, v.abs_row(lo))  # values + PRG slack (by abs idx)
        merkle.update(rows); q_irs.update(rows)  # commit-path accumulators
        rta = zeros(hi - lo, ELL)
        for band in v.bands:                     # stage A: per band, uniform
            band.eval((lo, hi), chal, out=rta)   # preloads cached on band state
        sink.push(rta, rows)                     # stage B: uniform batch
    for (t, quad) in v.quads:                    # stored on last operand
        p0.accum(r_quad_at(t), quad)             # operands re-encoded on demand
```

The "extra logic for building chunks and deciding when to flush" is the
`while` loop in `push` (~10 lines): rows arrive in walk order and the buffer
slices itself; flush fires at buffer-full and at `finalize`, nothing else. No
policy, no scheduler, no host sync — flushes are ordinary in-stream kernel
launches. This REPLACES QLinAccumulator's re-chunking + `qlin_group`'s per-row
dict regroup + `_StreamingPackets.__getitem__`'s bisect — the current
QLinAccumulator.update() already works this way internally; the sink is the
same loop minus the packet lookup, taking precomputed rTA instead of building
it inside.

Two simplifications (2026-07-03 follow-up):
- **Aux variables are first-class.** With aux (Freivalds y/u/p, LogUp inverses)
  as ordinary tape variables whose builders read their operands (+ challenges),
  the aux clause of last_use falls out — the rule is uniformly "variables whose
  BUILDER reads v, plus quads referencing v." Aux keeps only a block tag
  (R_p2 → second root, committed round 2). Every protocol round becomes the
  SAME walk with a tag filter and different accumulators active (round 1:
  build+commit p1; round 2: build all, commit p2; round 3: build all, fold;
  round 4: build all, open).
- **Construction granularity is orthogonal.** Per-claim builders are fine:
  firing is per-variable at its tape position regardless of when the value
  materialized; constructing a claim's outputs together costs at most one
  claim's output set of extra peak memory. Full per-variable builders also
  work but recompute shared intra-claim work (softmax's shift search) unless
  shared intermediates are themselves variables/memoized — defer.

Why this form wins:
- **The settlement exception vanishes**: bands sit on variables before the walk
  starts, so a LogUp-inverse's settlement sum band fires at the variable's own
  instantiation — no eager-attach special case.
- **The liveness convention gap closes structurally**: last_use derives from
  explicit tape structure (quad membership + dataflow), not from the
  "compute reads every constrained operand" convention.
- **Quad firing matches today's behavior**: every current quad has an own-aux
  operand of its declaring claim (§3.3 derived value), so the last-instantiated
  operand is created there — while the rule also handles all-external quads.
- **Claims become compile-time-only**: the runtime walk consults variables and
  their attached constraint objects; claims survive at runtime only as compute
  functions (dataflow).
- **Gates survive**: cids and quad indices are assigned positionally by the
  numbering pass (verifier parity); firing order differs but field add
  commutes → bit-identical q_lin/p_0.

### Gap analysis: current code → target model (effort)

The current streaming prover is ~⅔ of the target walk already: core.py's sweep
is claim-ordered with live/last_use freeing, per-claim variable emission, quads
before frees, and flag-differentiated rounds; QLinAccumulator already folds at
emission (only the band DISCOVERY is row-keyed); _StreamingPackets' count pass
is the numbering pass. Unchanged: wire format, cid numbering, PRF, proof
format, serializer, tape API, and the constraint definitions — only packaging
moves, which is what keeps every step bit-exact-gateable.

| # | item | size | notes |
|---|---|---|---|
| 1 | verifier evaluator (Phases 1.2–1.4) | M (~2–4 d) | oracle difftests per kind; runs on the local proof corpus |
| 2 | prover ChalSource (Phase 2) | S–M (~1–2 d) | BANDCHK + byte-identical dumps; needs Spark |
| 3 | quad-family lift (both sides) | M (~2 d) | emit_quad loops → generative descriptors with POSITIONAL indices; removes per-row quad structs; gate: bit-exact p_0 |
| 4 | variable-attached bands + walk restructure | L (~3–5 d) | numbering pass attaches bands/quads to variables; emission hands over band lists (deletes qlin_group / _band_key / bisect lookup); last_use += quad refs; keep-marks; builder/tag unification deferrable |
| 5 | Phase 3 cleanup (renames, delete legacy) | S (~0.5–1 d) | after a full-scale green run |
| 6 | (optional perf) Phase 4 fused kernels | L (~3–5 d) | orthogonal to the model |

Total for the target model (1–5): **~8–14 focused days**, ordered 1→2→3→4→5
(3 before 4 so the walk restructure is mostly deletion). Binding constraint is
Spark access for the prover gates, not code. Out of scope: the 4-round
re-sweep (removing it is the per-claim interactive protocol change).

## Outlook: level-of-abstraction questions (2026-07-03 discussion)

Two conclusions from asking whether folding could operate "at the level of
constraints instead of claims", recorded here because they shape Phases 4+:

- **The generative middle layer is forced by scale; the claim layer never
  reaches the folds anyway.** A materialized constraint-level linear system is
  O(nnz) ≈ 2–4×W (tens of TB at Maverick scale), and even a global run list is
  O(W) (fan families have one run per slot). Families/packets are the O(params)
  compression that lets both sides regenerate any window on demand. The claim
  layer's jobs (public statement / policy audit, witness generation, cid
  numbering, serialization) are all outside the folds.
- **Lift quadratics up, don't flatten linear down.** Quads are today a FLAT
  per-row list (`Quadratic` structs, no generative layer) — fine at ~1 GB for
  Maverick-1093, but O(W/ELL): ~25 GB at ~10k context, absurd at frontier, and
  a verifier-memory binder the streaming-verifier work will hit. The row
  structure is trivially generative (row indices advance by 1; a, b constant),
  so the unification is one `Family` carrying a linear run-generator AND a quad
  descriptor over its row range — same machinery, both folds.
- **Descriptor algebra (deferred with Phase 4).** Most decode kinds are
  mixed-radix strided index maps; one generic (shape, digit-permutation, cid
  strides, coef source) descriptor covers all but causal (triangular rank),
  embed (data-dependent gather), and rope (pair, two terms/slot) — ~15 kinds →
  ~4–5, and ONE fused kernel instead of ~14. Trade: index math becomes data,
  auditability shifts to descriptor values + difftests. This is the qlin doc's
  deferred LinOp idea; the run-fold interface is what it would slot under.
- **Families are not a removable layer — only their materialization timing is a
  choice.** A Family is the (row range × run generator) pairing; a generator
  without geometry is meaningless, so "claims → expanders directly" still
  constructs exactly these records. The PROVER must materialize them upfront as
  a row index: its fold is row-major (streaming frees each row's polynomial
  after one visit) and rows carry generators from several claims at once (e.g.
  `g` feeds the `W_Q`/`W_K`/`W_V` matmuls → three FreivaldsA families from
  three claims on the same rows; settlements scatter across the tape), so
  claim-major folding would re-encode or hold rows — breaking the streaming
  memory bound. The VERIFIER could fuse compile→fold claim-major (families
  transient, `Constraints` never materialized); the gain is small (MBs) and it
  costs the compile seam (`compile_difftest`, "compile then check"), so keep
  the seam — but note the fused single pass is the natural shape for the future
  STREAMING verifier, where holding the compiled system is the memory problem.
  Making claims emit their own runs (methods on claim structs) relocates
  construction without removing anything and blurs the wire-data /
  own-interpretation trust boundary; keep handlers separate.
- **Invariant regardless:** the per-claim cid cursor stays the numbering
  authority. cids index `challenge(s_comb, cid, "lin")`, so renumbering changes
  every r_lin value — it would break proof comparability across versions and
  every bit-exactness gate by construction.

## Risks

- `for_runs` windowing bugs at run boundaries — the property difftests against
  the untouched per-term `emit` are the net; `emit` stays line-for-line with the
  Python generators, so the oracle is independent of the new code.
- Rust fold regression from enum dispatch — runs are coarse (thousands of slots),
  so per-run overhead amortizes; the Phase-1 benchmark gates it.
- Cross-language drift during Phase-3 renames — mechanical, one side per commit,
  compile_difftest + gate.py between.
