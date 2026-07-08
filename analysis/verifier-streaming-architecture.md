# Streaming verifier architecture (scaling to the sound, full-model regime)

Status: **proposal** (2026-06-14). Motivated by the 48-layer Maverick proof
verify OOMing at >120 GB. Supersedes the parsing-only fix sketch — the real
bottleneck is the *constraint materialization*, not the JSON parser.

## Problem

The standalone Rust verifier (`verify_proof`) loads the entire proof and the
entire compiled constraint system into RAM, then runs the 6 checks. At
full-model scale this does not fit:

- The 48L/T=847 proof is a **12.5 GB JSON** file (`T_QUERIES=4`, test-grade).
- The sound configuration (`T_QUERIES=80`) is ~20× the opened-column data,
  ~250 GB of proof — does not fit in any single box's RAM as a whole.

So "load everything" fails by design in the regime we ultimately need. We want
a verifier whose peak memory is **independent of `T_QUERIES`** and bounded by
model size, not witness size.

## Measured facts (2026-06-14)

Byte split of the 12.5 GB proof JSON (stream-scan of top-level keys):

| section | size | notes |
|---|---|---|
| `claims` | 4.36 GB | the public statement: 11,433 claims, long op-history variable names, 128-var combine lists |
| `opened_p1` | 6.76 GB | opened columns (decimal text) |
| `opened_p2` | 1.35 GB | opened columns |
| `seeds`, `paths_*` | ~KB | negligible |

Two memory hogs, both confirmed:

1. **`serde_json::Value` DOM bloat.** A `Value` boxes every integer as a
   ~24-byte enum node and every object as a map allocation; the 12.5 GB JSON
   parsed to a ~120 GB DOM → OOM. *This is a parser choice, not inherent to
   the data* — a typed parse of the same bytes is ~file-size or smaller
   (the 8 GB of opened columns → ~3 GB of packed `Vec<u64>`).

2. **`Constraints` is O(witness rows), not O(model).** From `compile.rs`:
   ```rust
   pub struct Constraints {
       pub rows: Vec<Vec<Expander>>,   // ONE inner Vec PER WITNESS ROW
       ...
   }
   ```
   At 48L, `rows` has **m_total = 99,370,590 entries** → order tens of GB
   resident (≈2.4 GB of inner-Vec headers alone, plus ~2–4 `Expander`s/row;
   per-matmul coefficient vectors are `Arc`-shared, so not duplicated per row).
   This is the same per-row packet store the *prover* had before it was made
   to stream (`_StreamingPackets`). It does NOT shrink with `T_QUERIES`; it is
   a dominant resident cost at every scale.

**Measured 2026-06-15:** the standalone `verify_proof` on the 48L/T=847
fast proof held **~99.6 GB resident** (process RSS == HWM, plateaued before
`lin_col`) — confirming `Constraints` over m_total = 99,370,592 rows is the
dominant cost, not the proof JSON (the typed `from_reader` parse keeps that at a
few GB). This is query-count-independent, so the sound `T=80` artifact (~250 GB
of opened columns on top of this same ~100 GB floor) does not fit any single box —
the lazy/streaming verifier below is the prerequisite, not an optimization.

## Target architecture

The verifier's checks are **per opened column**: Merkle path, and the
IRS/linear/quadratic column evaluations, each operate on one opened column at a
time, and the column contributions are summed. So the scalable shape is:

```
spec = parse_claims(header)               # compact, resident, O(model)
for j in query_columns:                   # one column at a time
    col = next_column(source)             # streamed, dropped after use
    fold constraint contributions at col  # expanders GENERATED per-claim, on the fly
    check Merkle path of col vs root
finalize sum identities
```

Two changes, in order of importance:

### 1. Lazy per-claim compile (the substantive half)

Do **not** materialize `Vec<Vec<Expander>>` over all m_total rows. Keep the
per-claim expander descriptors (O(claims) ≈ 11k, genuinely bounded by model)
and generate each row's `(row, cid, coef)` contribution on the fly inside the
per-column fold. This mirrors the prover's `_StreamingPackets` transformation
exactly (which cut the prover's setup from ~45 GB to streaming).

Resident floor after this: the compact claim spec, O(model ops) — a few GB at
48L, and the SAME at sound `T=80` (constraints don't grow with queries).

### 2. Per-column streaming proof format (the easy half)

Dump the proof as **JSON Lines** (newline-delimited), not one object:

- line 0: the small header — `{claims, seeds, roots, q_irs, q_lin, p_0}`
- one line per opened column — `{"side","j","col":[...],"path":[...]}`

The verifier streams with the standard line iterator:
```rust
for line in BufReader::new(file).lines() {
    let rec: Value = serde_json::from_str(&line?)?;  // ONE column's Value
    fold(&spec, &rec, &mut acc);                     // existing accessors, dropped each iter
}
```
This reuses the existing `u64s()`/`paths_map()` accessors on one small
line-Value at a time — *less* parsing code than today's whole-file
`from_str::<Value>` + nested accessors, no binary format, no offset index, no
streaming-JSON state machine. Peak = one column (~1–2 GB), independent of
query count. (`proof_dump.py` already writes incrementally — it emits a header
line then one line per column.)

Sequential streaming suffices: the verifier needs each column exactly once. An
offset index is a later, optional speed knob (to restore rayon per-column
parallelism) and stays outside the TCB.

## Memory model

| | current | target |
|---|---|---|
| claims | ~95 GB (Value DOM) | ~few GB (compact spec, resident) |
| constraints | tens of GB (per-row materialized) | ~few GB (per-claim descriptors, resident) |
| opened columns | all `T_QUERIES` resident | one column streamed |
| **peak** | >120 GB (OOM) | **O(model) + one column**, query-count-independent |

## TCB analysis

The trusted computing base = code that, if buggy, could make the verifier
wrongly **ACCEPT** a false proof: field arithmetic, Blake3/Merkle, the
challenge PRF, the `Expander` definitions + their expansion rule, and the 6
checks. The JSON/proof parsing is **outside** the TCB (a wrong value →
failed Merkle/identity check → REJECT, never a forged accept).

Effect of this change on the TCB:

- **Trusted content unchanged.** Same expanders, same check math, same
  field/Merkle/PRF. No new trusted dependencies. The change is control-flow
  (materialize → stream), not new math.
- **Trusted surface shrinks slightly.** The materialized `Vec<Vec<Expander>>`
  table is deleted; it existed only to be read back.
- **Completeness guarantee preserved.** Soundness needs *every implied
  constraint checked* (a skipped constraint is a hole). Today that rests on
  `compile_claims` doing `for claim in claims { expand }`. The lazy version is
  `for claim in claims { expand-and-fold }` — the SAME loop, so the same
  coverage argument; no harder obligation on the auditor.
- **Honest cost: auditability, not TCB size.** An interleaved
  compile-and-fold is intrinsically a bit harder to read than straight-line
  "materialize, then check." The formal TCB is the same set of facts; the code
  an auditor reads is marginally more involved. This is the real trade, and
  it's worth it only because "load everything" does not fit the sound regime
  at all.

To hold the line: keep the `Expander` expansion functions and the check
functions **byte-for-byte unchanged** — called per-column instead of
populating a table — so the new code is a thin scheduling layer. Gate with the
existing differential test (byte-identical verdicts to the materialize
verifier on real proofs); the no-skip guarantee itself comes from the
unchanged `for claim in claims` structure, not from the test.

## Migration / gating

1. JSONL dump in `proof_dump.py` (keep the single-object path for back-compat
   / small proofs and differential testing).
2. Lazy compile + per-column fold in `verify.rs`, expander/check math
   untouched. Read the current rayon check loop first — preserving its
   per-column parallelism cleanly is where "thin scheduling layer" stays thin.
3. Differential test: streamed verdict == materialize verdict, byte-identical,
   on existing proofs (2L/4L and the 48L once converted).
4. Only then retire the materialize path for large proofs.

## Open items (measure, don't guess)

- ~~Instrument `compile_claims` to report the actual `Constraints` footprint at
  48L (ground the "tens of GB" estimate).~~ Measured 2026-06-15: ~99.6 GB
  resident for the whole verify (RSS==HWM). Still worth a per-type breakdown
  (which `Expander` variants dominate) to size the streaming rewrite's savings.
- Stream-scan the 4.36 GB `claims` section to find what dominates it (long
  variable names? embedded LogUp-table arrays?). If large arrays, those move
  to the streamed bulk too, not the resident header — the measurement decides
  the header/bulk split precisely.
- The existing 12.5 GB single-object proof verifies via the one-time typed
  parse (the `verify_proof` move-not-clone path); JSONL is for proofs going
  forward and is the prerequisite for verifying the sound `T=80` artifact at
  all.
