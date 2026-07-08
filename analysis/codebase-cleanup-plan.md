# Prover cleanup + repo reorg plan

Status: **plan for review (2026-06-27).** Supersedes the earlier survey-style cleanup
plan. Scope is set by three locked-in decisions from the maintainer:

1. **Target protocol: one sound-mode proof, file hand-off for now.** Drop the
   "fast"/Fiat-Shamir single-pass path. Keep today's working shape — the Python/CUDA
   prover runs the 4-round sound mode and dumps `proof.json`; the Rust `verify_proof`
   batch-checks it — **but keep the switch to a live interactive transport
   (verifier-supplied seeds) easy** (§3).
2. **The Rust verifier is locked in and is the validation oracle.** `verifier-rs`
   recompiles every constraint from the public claim list and returns ACCEPT/REJECT,
   so it is a sound bit-exactness check on any prover-side refactor. **This round does
   not touch `verifier-rs`.** Every prover change is gated by a verifier ACCEPT.
3. **Prover first.** Do the Python/CUDA cleanup (§1) and the repo reorg (§2) now;
   defer everything that touches the verifier or the interactive transport (§4).

The repo is a research prototype; the goal of this pass is to make it **efficient,
low-abstraction, and scannable by a newcomer** — not to add abstractions.

---

## 0. Validation protocol (the locked-in verifier as oracle)

Every step below is gated by these, run on a GPU box (Spark / H100). A step lands only
when all three stay green.

- **GATE-U (unit):** `cd pipeline && python tests/run_tests.py test_claims
  test_compile_parity test_routing_claim test_max_claim test_unexplained_info
  test_rescale test_reveal test_concat_claim` all pass.
- **GATE-E (end-to-end ACCEPT):** a small sound prove → Rust ACCEPT:
  - Llama, no checkpoint: `python demo_llama7b.py --num-layers 1 --seq 4 --no-lm-head
    --engine --sound --dump-proof /tmp/p.json` then
    `../verifier-rs/target/release/verify_proof /tmp/p.json` → `rust_verify: ACCEPT`.
  - Maverick MoE, `E=2` sound: the `demo_maverick_*` sound dump → ACCEPT.
- **GATE-N (negative):** `pipeline/tests/tamper_proof.py` (or a known-bad field flip)
  still makes `verify_proof` print `REJECT`. Confirms we did not weaken the check.

Rationale: the prover has a Python in-process co-simulation verifier (`core.verify`),
but it shares the prover's own `COMPILE_FNS`/`EXPANDERS`, so it is **not** an
independent check. Only `verifier-rs` is. GATE-E is therefore the real gate.

---

## 1. Prover cleanup (Python/CUDA), in order

### P0 — zero-risk deletions (no behavior change)

Dead code with zero live callers (all confirmed by grep across `pipeline/` + `analysis/`):

- **Dead leaf functions:** `Tape._wt` (tape.py:348), `Tape.word_combine` (tape.py:681),
  `SiluConfig.max_magnitude` (claims.py:884), `_quads_by_op` (core.py:2065),
  `_make_blinding_rng` (core.py:452), `_build_message_matrix` (core.py:383).
- **`pipeline/ref/splitmix64.py`** — imported nowhere.
- **`pipeline/run_all_checks.sh`** — references test files that have since moved to
  `deprecated/`; stale.
- **Stale string references to `deprecated/staged_prove.py`** (a file that does not
  exist) in core.py:2363, tape.py:1294, tape.py:1300 — fix the docstrings.
- **Doc drift:** `_StreamingPackets` docstring still describes a removed "late q_lin
  update" path (core.py:998-1017); `compute_q_lin` is named in dead comments
  (packets.py:5/28, core.py:1100/1285/1624) but no longer exists. Correct or drop.

Gate: GATE-U + GATE-E.

### P1 — strip debug/tamper scaffold out of production modules

Investigation refined this from the original "strip everything" framing — not every flagged
knob is pure cruft, and the tamper hooks turned out to be load-bearing test infrastructure.

**Done:**
- **The witness-value leak — REMOVED.** `LIGERO_RANGE_CHECK` / `LIGERO_RANGE_STRICT`
  (cuda_primitives.py) printed up to six actual signed **witness values** on overflow. The
  whole default-off block is gone; the functional `lookup_multiplicities_into` dispatch is
  unchanged. (This is the one genuine confidential-data leak in the set.)
- **The spent GC segfault-probe — REMOVED.** `LIGERO_GC_PROBE` / `LIGERO_GCPROBE_FROM` +
  `_fine_probe`/`_GCPROBE_FINE` + the in-loop `_probe(...)` calls were one-time tooling for
  a resolved 2026-06-11 unified-memory segfault. They cluttered the hot encode/sweep path;
  removing them declutters it. Default-off, so production behavior is unchanged.

**Kept deliberately (active tooling for the perf/fold work, not cruft):**
- `LIGERO_SWEEP_GC` — a real unified-memory allocator-release workaround (functional).
- `LIGERO_PHASE_TIMING` / `LIGERO_EPHASE` — the just-added (HEAD `4afc0f6`) fold profiling.
- `LIGERO_NO_FOLD` — the A/B fold-disable switch; `LIGERO_QLIN_BANDCHK` — the band-vs-legacy
  regression gate the q_lin reorg still relies on.
- `LIGERO_COMPILE_PROFILE` / `LIGERO_LAYOUT_BREAKDOWN` / `LIGERO_STREAM_DBG` — opt-in
  analysis tools (the `sys.exit(0)` paths only fire when explicitly enabled). Left for now;
  a later pass can fold them behind one `DEBUG` guard if desired.

**Tamper hooks — KEPT, decision deferred (see note):** `TEST_TAMPER` (max_claim.py,
routing_claim.py) and `CONCAT_TAMPER` (compute_fns.py) are how the **negative tests inject a
wrong witness** (a non-argmax mask, a tampered concat) to confirm the constraints REJECT it.
Proof-tampering — the original plan's suggested replacement — can't reproduce this: flipping
a committed value in `proof.json` trips the Merkle check, not the constraint under test, so
removing the hooks would **lose soundness-test coverage**. They are not a soundness hole
either (the prover is already untrusted; substituting a witness is exactly what the verifier
catches). **Decided (maintainer): keep now, isolate later** — tracked follow-up F1 below:
move the injection into a test-only seam (a witness-override callback or test subclass) so the
prod claim modules no longer carry module-global tamper dicts. Not a P1 strip.

**Tracked follow-ups (not blocking the cleanup):**
- **F1 — test-only tamper seam.** Replace `TEST_TAMPER`/`CONCAT_TAMPER` module-global dicts in
  `max_claim.py`/`routing_claim.py`/`compute_fns.py` with a test-only witness-override
  mechanism; migrate `test_routing_claim`/`test_max_claim`/`test_concat_claim` onto it.
- **F2 — fast-mode doc sweep.** Reconcile the design docs (paper.md §5.3, any remaining
  ARCHITECTURE mentions) with the fast-mode removal; keep the historical "the 400B run was
  fast-mode `T=4`" statements accurate as past record.
- **F3 — combine doc sweep.** Update the design docs that still describe `MaskedCombineClaim`
  as the MoE combine (`CLAIM_SPECS.md` §MoE routing, paper.md §4.5, `maverick-cost-model.md`
  framing) to the Freivalds combine — best done with the `→ CombineClaim` rename in the
  verifier round so the name settles once.

- **Note:** `LIGERO_QLIN_AUDIT` / `LIGERO_EAGER` from the old plan do not exist — nothing to
  remove. `master_seed = b"\x42"*32` (the ZK-blinding stub) is a securing-ZK fix, not this
  cleanup.

Gate: GATE-U + GATE-E (the GC-probe removal edited the hot `_stream_sweep`/encode loop, so a
prove → Rust ACCEPT confirms the loop still works).

### P2 — delete FingerprintClaim (closes the one prover/verifier parity gap)

**Decided: delete.** `FingerprintClaim` (claims.py:2164) + `Tape.fingerprint`
(tape.py:1206) + `fingerprint_compute` (compute_fns.py:190) are dead end-to-end (no demo
or model emits them; only one positive unit test in test_claims.py). They are also the
**only** claim the prover can compile that the Rust verifier cannot —
`handlers.rs:231` `panic!`s on it. Removing them deletes dead code and closes the gap
(safe: no dumped proof contains the claim, so GATE-E is unaffected).

- **Done.** Removed the claim class + `fingerprint_sample/aux/compile`, `Tape.fingerprint`,
  `fingerprint_compute`, all `SAMPLE_FNS`/`AUX_FNS`/`COMPILE_FNS`/`COMPUTE_FNS` entries, the
  import-list mentions, the `packets.py` example refs, and the `test_honest_fingerprint`
  fixture + test. (The `protocol.py` REAL_COMPILE-region comment refs are left for P5, which
  relocates that region to `deprecated/`.)
- **Optional hardening — skipped.** A prover-side assertion refusing claim types absent from
  the verifier's handler set would prevent future gaps, but it adds a maintained claim-type
  list/coupling; the gap is closed, so this is left out per the low-abstraction goal.

Gate: GATE-U + GATE-E.

### P3 — collapse fast → sound (the mode simplification)

**Done.** `prove_streaming` is now always the four-round sound path (fast branch +
`sound=` param removed); `Tape.prove` dropped `sound=`/`LIGERO_STREAM_SOUND`; the
`--sound` flag + `sound=` threading removed from `demo_llama7b`/`demo_maverick_block`/
`demo_maverick_moe` (`demo_maverick_full` passed no `sound`, so it is sound now by
default). The seed seam is preserved — `prove_streaming(tape, cfg, seed)` still derives
the per-round `s_op`/`s_comb`/`s_col` at the points a transport would inject them. Tests
are unaffected (they use the separate `test_prover.prove`, not `Tape.prove`). README run
instructions + the ARCHITECTURE "two modes" / "fast mode" claims updated (historical "the
400B run was fast-mode" caveats left intact). The Rust unsound `run_verification_fast`/
`round0` mirror stays for §4. **Follow-up F2:** a fuller design-doc sweep (paper.md, any
remaining ARCHITECTURE mentions) for the fast-mode removal.

Make the 4-round sound prove the only path. The fast and sound paths already share
`_stream_sweep`; only the round orchestration differs (core.py:2389-2419).

- Delete the fast branch `if not sound:` (core.py:2389-2399) and the `repro is None`
  print path (core.py:2431-2432).
- Drop the `sound=` parameter from `prove_streaming` (core.py:2349) and `Tape.prove`
  (tape.py:1288), and the `LIGERO_STREAM_SOUND` env fallback (tape.py:1301-1302).
  Always run the four sweeps.
- Rust: the unsound fused driver `run_verification_fast` + `round0` in `verify.rs` is the
  mirror of this mode. It is **verifier-side**, so per decision (2) it is **deferred to
  §4**, not removed now — but note it here so the two are dropped together later.

**Preserve the future-interactive seam (§3): do not fuse the per-round seed derivation
into one master seed.** Keep `s_op`, `s_comb`, `s_col` as the three inputs the sweeps
consume (today via `protocol.round_seeds(seed)`), so a future transport can supply each
one live, after the matching commit, with no structural change.

Gate: GATE-U + GATE-E (both Llama and Maverick-E2 in sound mode) + GATE-N.

### P4 — retire the duplicate combine claim

**Done.** `MaskedCombineClaim` and `FreivaldsCombineClaim` proved the same seam
`y = Σ_e m_e·X_e`. Deleted Masked; everything now uses `freivalds_combine`.

- Removed `MaskedCombineClaim` + `masked_combine` + `combine_sample/aux/compute/compile` +
  its 4 registry entries (routing_claim.py).
- Migrated `demo_maverick_moe`'s real combines **and** the `E=1` broadcast pin (all three
  demos) onto `freivalds_combine`; fixed the module docstrings/diagram/packet comment.
- Tests: `test_routing_claim` collapsed to Freivalds-only (dropped `use_fc`; removed the
  masked-specific `test_cheat_combine_wrong_y`/`test_tamper_m_rep`/`test_tamper_prods` — the
  `test_fc_*` equivalents preserve coverage for the surviving claim); `test_compile_parity`'s
  masked case merged into the Freivalds case (which already exercises `route_top1`).

**Security note (why this is sound).** The two are NOT the same security argument:
Masked is **exact** (commits the per-expert products, pins `y` by exact identities, zero
combine-specific error); Freivalds is a **probabilistic** Schwartz–Zippel/Freivalds check
(`(Σ m·X)·ρ = y·ρ` for random `ρ`), error `~1/|F|` per token — the *same* negligible,
already-accounted-for class `MatmulClaim` uses for every matmul (paper §5.4). Cost model
confirms Freivalds dominates for any `F>2` (`maverick-cost-model.md:45-46`). The catch:
Freivalds's soundness *requires* commit-before-challenge (`ρ` drawn after `y` is committed),
so **P4 leans on P3** (sound-only) — fine, since the whole proof already needs that ordering
for matmul. The only thing genuinely given up is an exact combine that would survive a
non-interactive setting; the target design does not need it.

**Deferred to the verifier round** (decision: verifier untouched here): the Rust
`compile_masked_combine` + dispatch arm and `protocol.py` `_c_masked_combine` go dead (prover
just stops emitting the tag) — removed with the other dead verifier code. The
**`FreivaldsCombineClaim` → `CombineClaim` rename** is also queued there (the class name is
the wire tag the Rust verifier matches on, so the rename is a coordinated cross-boundary
relabel, not Python-only).

Gate: GATE-U (`test_routing_claim`, `test_compile_parity`, …) + GATE-E on `demo_maverick_moe`
`E=2` sound dump → Rust ACCEPT.

### P5 — retire both Python verifiers (`core.verify` and the `REAL_COMPILE` oracle)

**DONE (P5a/P5b/P5c). The Rust `verifier-rs` is now the single verifier.**
- **P5a** — every test verifies through the Rust binary (`tests/_rust_verify.py` →
  `verify_proof`), not the co-sim. (Surfaced 3 layers of `test_claims` coupling — explicit
  `TableSettlement`, settlement ordering, Python-message assertions — all fixed.)
- **P5b** — deleted `core.verify`/`_check_identities`/`compute_r_at_eta`/`_lagrange_zeta_at_eta`
  + `Tape.verify`; demos prove-and-dump only. (A too-wide splice first deleted interleaved
  `Proof`/`SAMPLE_FNS`/`AUX_FNS`/`_SKIP_B_CHUNK` — caught by the Spark GATE-U, fixed with a
  surgical redo + val.sh hardened against stale-proof false positives.)
- **P5c** — extracted the compile half (`Constraints`, `expand_*`, `_emit_*`, `_c_*`,
  `REAL_COMPILE`, `_op_challenge`, `compile_claims` — 50 nodes, by AST span) from `protocol.py`
  into `deprecated/python_verifier_compile.py`; **`protocol.py` 1413 → 446 lines** (the
  byte-exact prover↔verifier contract only). Moved the 4 difftest tests to `deprecated/`. The
  Rust `compile_difftest` bin is now dead (verifier round removes it).

**Decided: retire both. The Rust `verifier-rs` becomes the single source of truth for the
constraint compile; end-to-end ACCEPT (GATE-E) is the standing check.** Deprecate rather
than hard-delete, so the references stay easy to find.

**(a) `core.verify` (the co-simulation) — remove.** `core.verify` / `_check_identities`
(core.py:2494) reuses the prover's own `COMPILE_FNS`/`EXPANDERS`, so it is not an
independent check, just a circular one. Remove it, `_check_identities`, and the co-sim-only
helpers reachable only from it (`compute_r_at_eta` at core.py:1591 — confirm each with grep
first), the `LIGERO_SKIP_GPU_VERIFY` gate, and the demo plumbing that calls the in-process
verify.

**(b) `protocol.REAL_COMPILE` + `compile_difftest` (the port's golden reference) — move to
`deprecated/`.** This is the surviving compile half of the old standalone Python verifier
(the one the Rust verifier was bit-exact-ported from; its checks half, `verify.py`, is
already in `deprecated/`). It is used at runtime by **nothing** in the live prover — the
`protocol.compile_claims` mentions in core.py:1756/1849/1971 are comments; the only real
callers are the difftest tests. Extraction boundary (verified):
  - **Move to `pipeline/deprecated/python_verifier_compile.py`:** from `protocol.py`, the
    `Constraints` dataclass (:210), all pure row expanders `expand_*` (:232-686), the
    `_c_*`/`_emit_*` handlers, `REAL_COMPILE` (:1215), `_op_challenge` (:1243), and
    `compile_claims` (:1261).
  - **Move the dependent tests to `deprecated/`:** `tests/dump_compile_parity.py`,
    `tests/test_compile_parity.py`, `tests/test_expanders.py`, `tests/test_rope_straddle.py`
    (the last two drive `pr.expand_*` directly).
  - **Rust:** the `verify-rs/src/bin/compile_difftest.rs` bin loses its oracle → it joins
    the dead-bin group in §4 (recoverable via git; note it in the `deprecated/README`).
  - **`protocol.py` stays** as the trusted-primitives-only module: `challenge`, `op_vec`,
    `random_columns`, `round_seeds`, `_seed_bytes`, `Config`/domains, Merkle verify,
    poly/Lagrange eval, and `claims_to_json` (:1394) — all still used by the live prover.

Once both are gone, `SAMPLE_FNS` (15/18 trivial `return None` stubs) can be simplified to
default-missing-keys.

Gate: GATE-U (remaining suite green; the four moved tests no longer run) + GATE-E (prove →
Rust ACCEPT, unaffected — the live path never called `compile_claims`). This step edits the
trusted `protocol.py`, so validate it on a GPU box, not by reasoning alone.

---

## 2. Repository reorganization (file moves) — do this LAST

Done after §1 so fewer files move and the logic is settled. The goal is that a newcomer
opening the repo sees the **important parts** immediately and the research/scratch
material is set aside. Use `git mv` to preserve history. Each move batch is gated by
GATE-U + GATE-E (moving Python rewrites imports, so the verifier ACCEPT is the proof the
wiring still holds).

### The problem today

- `pipeline/` is 21 flat modules mixing the trusted engine, the claim library, runnable
  demos, weight loaders, and a stray benchmark.
- `analysis/` is 25 one-off research scripts + 21 docs (referenced design specs,
  transient plans, handoffs, archives) + figures, all flat — the worst scannability
  offender.

### Target tree (decided)

Four role-based top dirs — `prover` / `verifier` / `demo` / `analysis` — with `README.md`
and `ARCHITECTURE.md` at root (the entry docs). Tests live **within each** (`prover/tests/`
Python; the verifier's tests stay in the Rust crate — Cargo can't host them elsewhere), not
a top-level `test/`. Flat-import bootstrap so `import core` still works after grouping.

```
infproof/
  README.md            # start here (root)
  ARCHITECTURE.md      # system overview — first thing a newcomer reads (root)
  prover/              # was pipeline/ + cuda/  — the Python + CUDA prover (a library)
    engine/            # core, tape, protocol, packets, compute_fns, cuda_primitives
    claims/            # claims, routing_claim, max_claim, ui_claim, unexplained_info
    kernels/           # was cuda/  (goldilocks/ntt/blake3/gl_*/merkle .cuh + their .cu tests)
    ref/               # pure-python validators (goldilocks_ref, polynomials)
    loader.py, kquant_cuda.py, proof_dump.py, _uint64_compat.py, _bootstrap.py
    tests/             # the Python prover tests
  verifier/            # was verifier-rs/  (Rust crate; tests stay in-crate)
  demo/                # demo_llama7b, demo_maverick_{moe,block,full}, demo_prompt.txt
  analysis/            # research scripts + deep design docs + figures (name KEPT — see R1)
    docs/              # the README/ARCHITECTURE-referenced specs + CLAIM_SPECS (from prover)
    bench/             # qlin_fold_bench.py
    figures/
  deprecated/          # was pipeline/deprecated/  (archive, hoisted to top level)
```

Notes on the placements: CUDA kernels live under `prover/` (only the prover uses them; this
also turns the `../cuda` sibling dependency into `../kernels`). `verifier` drops the `-rs`
suffix — but if the planned CUDA verifier port lands, restore it (`verifier-rs` +
`verifier-cuda`) to avoid ambiguity. `demo/` is top-level for discoverability, but the demos
import prover internals (`from core import …`, `loader`, …), so demo scripts use the prover
bootstrap to reach `prover/engine` + `prover/claims` + the prover root — the coupling is
real, just absorbed by the bootstrap. `analysis/` **keeps its name** so the live
`from analysis.cv_check import cv_check` in `test_prover.py` (and intra-analysis imports)
survive the reorg.

### Move batches (each gated by GATE-U + GATE-E — moving Python rewrites imports)

- **R1 — organize `analysis/` (highest value, lowest code risk; name unchanged).** Create
  `analysis/docs/` for the referenced design specs (`design-feasibility`, `paper.md`,
  `verifier-streaming-architecture`, `qlin-*-reorg`, `quantization-evaluation`, `token-binding`,
  `nvfp4-*`, … + `CLAIM_SPECS` once it moves in R3) and `analysis/figures/`; leave the one-off
  research scripts (`ui_*`, `*_sweep`, quant sims, `cv_check`, …) at `analysis/` root. Update
  the README "Further reading" + ARCHITECTURE §11 links. Because the dir name stays `analysis`,
  no Python imports break.
- **R2 — hoist the archive + the bench.** `git mv pipeline/deprecated deprecated`;
  `git mv pipeline/qlin_fold_bench.py analysis/bench/` (or delete — orphaned branch-local
  microbench). Refresh the README "Repository layout" section.
- **R3 — the structural move (churny; gated hard by GATE-E).** Rename and regroup:
  - `git mv pipeline prover`; `git mv cuda prover/kernels`; `git mv verifier-rs verifier`;
    `git mv` the prover modules into `engine/` + `claims/`; pull the demos out to top-level
    `demo/`; move `pipeline/CLAIM_SPECS.md` → `analysis/docs/`.
  - **Flat-import bootstrap:** `prover/_bootstrap.py` appends `engine/` + `claims/` to
    `sys.path`; the entry points (`demo/` scripts, `prover/tests/`, the prover-importing
    `analysis/` scripts) import it so `import core` / `from claims import …` keep working with
    no dotted-path rewrite. (Slightly hacky, but the lowest-churn landing.)
  - **CUDA path fix:** `cuda_primitives.py` resolves the kernel-header dir relative to its own
    file (`cuda_primitives.py:53`). After it moves to `prover/engine/`, the kernels are at
    `../kernels` — update that computation (re-run GATE-E: a wrong path crashes the JIT prove,
    it does not silently mis-verify).
  - **Doc/path sweep:** README run-commands (`cd pipeline` → `cd prover`, demo paths),
    ARCHITECTURE references, `verifier-rs/…` → `verifier/…` paths (README, `tools/`, the
    Spark `val.sh`, the dead interactive bin's `LIGERO_PIPELINE` default), the
    prover-importing `analysis/` scripts, and the `spark-infproof-setup` agent note.
  - Validate heavily with GATE-E — an import or path mistake surfaces as a REJECT or a crash,
    not a silent error.

---

## 3. The future-interactive seam (design constraint to preserve)

Today (file hand-off) the prover dumps `proof.json` including the seeds it chose, and
`verify_proof` reads them. As a standalone artifact that is **not adversarially sound**
— the prover picked its own challenges. True interactive soundness needs the **verifier**
to draw each of `s_op`, `s_comb`, `s_col` from its own randomness and hand it to the
prover **after** the matching commit. The skeleton for that already exists, dead, on the
Rust side (`prover.rs` `SubprocessProver` + `verify.rs` `staged`/`run_verification`,
which draw `/dev/urandom` coins after each commit) and the Python side
(`deprecated/prover_server.py`).

To keep the switch cheap, the §1 cleanup must:

- Keep the 4-round structure in `prove_streaming` intact (P3) — never collapse the
  rounds back into one pass.
- Keep `s_op`/`s_comb`/`s_col` as **three injectable inputs** to the sweeps, not values
  derived inside the prover from a single fused seed. The only change to go live later is
  *where those three bytes come from* (a transport message instead of `round_seeds`).
- Do not delete the Rust interactive cluster or `prover_server.py` in this round (they
  are the reference for the wire protocol) — see §4.

## 4. Deferred to a later "transport + verifier" round (not now)

Per decision (2), nothing here is touched this round; listed so the couplings are
explicit:

- **`verifier-rs` internals:** drop the unsound `run_verification_fast`/`round0`
  (mirror of P3); remove the dead `compile_masked_combine` handler (after P4);
  consolidate the per-family geometry written twice (`Expander::emit` +
  `verify.rs::row_contrib`).
- **Dead Rust bins (remove together):** `verify_interactive.rs` (spawns the deprecated
  `prover_server.py`), `difftest_foundation.rs` (pairs with `deprecated/difftest_foundation_py.py`),
  and `compile_difftest.rs` (oracle moved to `deprecated/` in P5). All three are recoverable
  via git; note them in `deprecated/README`.
- **`deprecated/` deletion is coupled to those bins.** `pipeline/deprecated/` (now larger
  after P5 adds the Python verifier-compile + its tests) has no live Python importer, **but**
  `prover_server.py` is the Python end of the future interactive transport (§3). So keep
  `deprecated/` rather than deleting it — it is exactly the "easy to find later" archive the
  maintainer asked for. A later pass can prune the genuinely-dead benches once the transport
  is rebuilt.
- **`cuda/tests/*.cu`** (4 standalone tests, no build automation) — orphaned; remove or
  wire into a build. Low priority.
- **Reviving the live interactive transport** (verifier-supplied seeds end to end at
  scale) — the actual feature behind §3.

## 5. Maintainer decisions (resolved 2026-06-27)

1. **P2 FingerprintClaim:** **delete.**
2. **P5 Python verifiers:** **retire both.** Remove `core.verify` (co-sim); move
   `protocol.REAL_COMPILE` + `compile_difftest` + their tests to `deprecated/` (not
   hard-deleted). Rust `verifier-rs` becomes the single compile source of truth.
3. **R2 `qlin_fold_bench.py`:** **move to `research/bench/`.**
4. **R3 packaging style:** **flat-import bootstrap** (group into subfolders, keep
   `import core` working via a `sys.path` bootstrap — no dotted-path rewrite).
5. **Top-level rename:** **yes — `pipeline/`→`prover/`**, done in the final reorg phase
   (R3) with the doc/path sweep above. (`cuda/`/`verifier-rs/` names kept.)

---

## Appendix: carried-over context

- **q_lin fold reorg** (the prover's dominant cost) is tracked in
  `qlin-fold-reorg-plan.md` / `qlin-family-object-reorg.md`; steps 1-3 and 5 are landed
  (the per-row packet store, lazy compile, and watermark apparatus are already gone).
  Measured A/B (BASE `f486bd5^` vs HEAD): ~10% faster prove at every scale, **peak GPU
  identical** (24L: 44.89 vs 44.92 GB) — the reorg buys prove-time + code simplicity,
  not streaming peak memory. That work is orthogonal to this cleanup and continues on its
  own branch.
- **Earned abstraction, do not collapse:** the claim polymorphism, the packet classes,
  the streaming accumulators, and the prover/verifier TCB split are all pulling their
  weight. The removable surface is dead code, debug scaffold, the fast-mode branch, the
  one duplicate combine claim, and the flat-directory sprawl — not the core design.
