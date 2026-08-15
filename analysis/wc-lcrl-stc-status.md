# WC-LCRL-STC — status (the only authority on what is implemented)

Spec: [wc-lcrl-stc-spec.md](wc-lcrl-stc-spec.md). Branch: `wc-lcrl-stc`
(private repo, testing).

## Implemented and tested

- **S1 — coefficient-RS enrollment** (`prover/wc_bridge.py`): per-width
  packing into B-blocks, lam mask coefficients, coefficient-RS via zero-pad
  + batched forward NTT (probed: `rs_encode_rows` is evaluation-based and
  is the WRONG primitive for coefficient enrollment), column-major store,
  sha256 Merkle root, manifest digest. Zero-padded tails are bound.
- **S2 — late coefficient bridge**: shared per-width `rho` derived only
  from the post-R1 transcript state; `P_trace = W rho` and projected masks
  `pi` committed; `alpha`/`eta` derived only after that commitment;
  aggregated `c`, `v = c(eta)`; CPU python-int verifier (stand-in for the
  Rust twin) recomputes every coin and checks the bridge equation against
  Merkle-verified enrollment columns.
- **Tests** (`prover/tests/test_wc_bridge.py`, 8/8): domain/Horner
  cross-check; honest ACCEPT (+ independent `W rho` recomputation); exact
  `H_40 = C(16383,40)/C(32768,40) ≈ 8.859e-13`; negatives: tampered
  `P_trace`, internally consistent re-committed fake projection, tampered
  enrollment column, attacker-chosen `eta`, nonzero claim in the padded
  tail.
- **Bench** (`analysis/bench/wc_bench.py`): production geometry on a
  configurable block slice. V100, width 4096, one block (62.9M weights):
  enrollment 4.65 s (74 ns/param, one-time; dominated by the CPU sha256
  Merkle), bridge prove 0.07 s (1.13 ns/param), CPU verify 0.26 s, ACCEPT.

## Integrated (bricks 1-4, 2026-08-15)

- **§0.2/0.3/0.5 wiring is DONE for the single-claim case**: a
  `use_bridge=1` routed claim (part of the statement) proves `P = W rho`
  by the bridge inside the real 5-round transcript. The per-expert
  FreivaldsLF1B folds over the weights are NOT emitted by either
  compiler; the committed Pj rows are pinned (b_chunk) to the public
  P_trace that the bridge authenticates against the enrollment root.
  `tape.prove(weight_enrollment=...)`; fail-closed both ways (bridged
  claim without enrollment refuses to prove; without a verified wc
  section the Rust verifier rejects).
- **Rust twin is DONE**: `verify_proof.rs` recomputes every bridge coin
  from its own transcript, checks eta/c/v/merkle/bridge-equation and
  only then feeds the pin to compile. Wire: the `wc` proof section;
  enrollment merkle is blake3; the NTT domain order is protocol-pinned
  (natural powers of 7^((P-1)/N_w)) with an assert against kernel drift.
- Gate: 79 tests green across 13 suites, incl. the flagged proof
  end-to-end through Rust and the bridge negatives.
- Toy prove A/B: 0.02 s vs 0.02 s — NO measurable delta at toy shapes,
  as expected: the deleted work is linear in weight count (the toy-scale
  lesson). The magnitude of the deletion at scale is the measured
  Maverick bridge (B+C 1,051 s) vs the persistent fold+open (5,437 s).

## Integrated (bricks 5-6, 2026-08-15)

- **Multi-claim, shared per-width rho (§0.2)**: every bridged claim of
  one width samples ONE rho (`rho-w<J>`, both compilers) — 72 Maverick
  matmuls spend one q_w=40 opening set, not 2,880 mask points. The
  claim→P_trace-slice map is canonical (recomputed from the claim set on
  both sides, never wire). `wc_bridge.enroll_tape` builds the enrollment
  from a tape; prove fail-closed checks coverage.
- **W-block removal (brick 6)**: `tape.external()` weights are prover
  inputs, never committed rows — layout skips them, wire carries null
  row_start, the Rust parser poisons it (any accidental use overflows).
  External weights refuse to build without use_bridge. Proof measurably
  shrinks; Rust ACCEPT end-to-end. Together with the fold removal this
  deletes BOTH production weight components: 3,625 s (fold) + 1,812 s
  (commit/open) per proof — pending a full-scale measured run.

## NOT implemented (do not claim it)

- **pi as committed R2 rows proved by fresh qLin** (interim: bound via
  the hosted late coin).
- **§0.6 message cache** (two-pass q/opening over cached fresh rows).
- **Embedding late-lookup (review §5.7)**, gain blocks, enrollment
  ledger persistence across proofs (in-memory only), Maverick-tape
  driver switch to use_bridge + enroll_tape.
- **§0.6 message cache**: the existing `LIGERO_WITNESS_CACHE`/`SPILL`
  caches compute_fn outputs only; canonical fresh message rows + pad
  metadata caching and the two-pass q/opening structure are not built.
- **§0.8 compiler chain invariant**: not enforced anywhere yet.
- **Rust twin** of `verify_bridge`; GPU (BLAKE3) enrollment Merkle;
  gain-block groups; enrollment-from-GGUF (current enrollment takes
  already-decoded tensors and does not verify them against a GGUF digest).

## Remote validation (vast, 2026-08-14)

Quadro RTX 8000 ($0.241/h, instance 47736613, 445 s wall ≈ $0.03,
destroyed after download). `wc_remote.sh`: gate_failures = 0
(wc_bridge 8/8 + fiat_shamir + routed_projected on the box). Production
geometry, growing slices:

| slice | weights | enroll ns/param | bridge prove ns/param |
|---|---|---|---|
| 4096 × 1 blk | 62.9M | 62.9 | 1.00 |
| 4096 × 4 blk | 251.7M | 50.6 | 0.41 |
| 4096+11008 × 2 blk | 464.0M | 49.3 | 0.38 |

Per-param cost still falling with slice size (fixed overhead amortizes),
so linear extrapolation from the largest slice is an upper bound.

## MEASURED: the full Maverick (vast L40S, 2026-08-14, run #3)

`wc_maverick.py` over the real UD-Q4_K_XL GGUF, ALL persistent maps —
400,711,352,320 weights, 9,578 units, 26.16M polynomials, 3,830 blocks
(widths 128/1024/5120/8192/16384/202048; embed+lm-head through the lean
wide path):

| pass | time | notes |
|---|---|---|
| A — enrollment (one-time) | 648.8 s | dequant 324.7 + pack/NTT 353.5; GPU BLAKE3 merkle; **1.62 ns/param** |
| B — bridge (per proof) | 470.9 s | W rho + pi + c + v; dominated by the one unavoidable full weight read |
| C — eta columns (per proof) | 580.0 s | second NTT sweep; drift=False vs pass A digests |
| verify | 6.1 s | coins + v=c(eta) python ints + GPU bridge equation + spot checks |

**ACCEPT=True, no fails; root 5607ebd0…; ledger spend 40/1024.**
Total compute 28.4 min end-to-end (4.24 ns/param); the earlier modeled
numbers are superseded by these measurements. Per proof the bridge side
costs B+C ≈ 1,051 s on an L40S against the 3,625 s persistent q_lin fold
+ 1,812 s persistent openings it is meant to replace — a measured ~5.2x
on the replaced component, PENDING the transcript integration (the
replacement is real only once the 5-round wiring deletes those stages).

Run history: #1 killed by a false `offline` API reading (lesson: verdict
from suite.log only); #2 OOM on the width-202048 group + cold slow disk
(both fixed); #3 clean. Raw results:
`analysis/bench/remote_results/l40s-47743564/`.
