# RoutedProjected-MoE — gap list and staged plan

Target: `analysis/routed-projected-protocol.md` + `demo/4h-production-runbook.md`.
This file is the ONLY authority on what is actually implemented.  Every stage
below is DONE only when its own gate passed; nothing is marked from a design
document.

Verified state of the tree on 2026-08-05 (before any work):

| target artifact | state |
|---|---|
| `RoutedProjectedMatmulClaim` | absent (0 grep hits repo-wide) |
| standalone `RescaleClaim` | absent |
| five-message transcript (`s_bind`, `p3`) | absent — 3 seeds, all pre-derived from one base by `pr.round_seeds(SEED)` (`prover/protocol.py:167`, `prover/core.py:2798`, `demo/demo_maverick_full.py:358`); the docstring itself marks this a TEST shortcut |
| active-only MoE builder | absent — `demo/demo_maverick_full.py:172-181` builds 128 gate + 128 up + 128 down `tape.matmul` outputs plus three `freivalds_combine`, i.e. exactly the runbook's first forbidden regression |
| CLI `--enroll-weights` / `--weight-commitment` / `--expected-weight-root` / `--admission-report` / `--public-sz` | absent (only `--from-gguf/--tokens/--layers/--experts/--d/--d-ff/--vocab/--prompt-n/--cont-n/--witness-only/--dump-proof/--logits-out/--ui-abort-above`) |
| verifier policy inputs (trusted weight root + statement digest, required) | absent — `verifier/src/bin/verify_proof.rs` takes 3 seeds and no required policy |
| streaming JSON proof writer | `prover/proof_dump.py` present, 68 lines — must be checked against the "no Python-int materialization" requirement |
| admission report + fail-closed gate | absent |
| reusable base that stays | persistent weights (`commit_weights`, `WeightCommitment`, `tests/test_persistent_weights*.py`), `prover/routing_claim.py` top-1 route, tape/claims/CUDA kernels |

Ledger cross-check (done): the document's baseline
`(W,L,Q) = (888,249,981,888 / 162,237,276,010 / 173,106,423,296)` is exactly what
`analysis/maverick_cost_model.py` emits at S=1000, so the new ledger is derived
against this tree and not against an abstraction.
`analysis/routed_projected_4h_model.py` now runs and closes at 12,957.864 s =
3.5994 h after the compact proof-wire change described in S4f.
Exact ratios are 10.7183% / 19.1477% / 24.4905% (the protocol note said 10.78%
for W — corrected in the copy checked in here).

## Hardware decision

Development and all toy gates: the local V100-SXM3-32GB box (4 vCPU, 83 GB RAM).
Production 400B run: a **rented (vast.ai) card**, so the admission report is
bound to the rented machine's fingerprint, never to this box.  Rental follows
the standing rule: hard budget cap + watchdog, destroy on finish.

Indicative isolated-kernel rates measured on the local V100 (NOT admission
evidence — the runbook forbids counting isolated timings; recorded only to show
the commit path is not the first thing to fail):

```
encode (pad + inverse NTT + coset LDE)   3.40 ns / message slot   (caps 9.5 / 9.0)
blake3 column hashing                    0.34 ns / message slot   (cap 1.4)
ELL=8192, K_DEG=16384, N_LIG=65536, 1024 rows, 8.4M slots
```

## Stages

Each stage ends with its own gate.  A stage that fails its gate is reverted, not
carried.  The full 400B proof is not started until every stage below is DONE and
`admission.json` passes every per-kernel cap.

- [x] **S0 — documents + admission model in-tree.**
      Gate: `analysis/routed_projected_4h_model.py` runs, all asserts pass, and
      its baseline matches `maverick_cost_model.py` at S=1000.  PASSED.
- [x] **S1a — sequential Fiat--Shamir over the existing four rounds.**
      `round_seeds(SEED)` is gone from the prover: `s_op = H(statement ||
      R1 block labels || R1 roots)`, `s_comb = H(s_op || root_p2)`,
      `s_col = H(s_comb || q_irs || q_lin || p_0)`, length-framed
      (`prover/protocol.py`), mirrored in `verifier/src/fs.rs`.  The verifier
      recomputes every coin from the raw claim bytes it read and uses the file's
      seeds only to report agreement.  `verify_proof` gained optional policy
      arguments (trusted weight root, trusted statement digest; `-` skips one).
      Gate PASSED: `tests/test_fiat_shamir.py` 6/6 (honest ACCEPT, rewritten
      s_col / s_op / claim set REJECT, policy digest enforced, and a structural
      test that R1 runs with no op challenge and no column set), plus
      test_claims 21/21, test_routing_claim 19/19, persistent-weights 3+3+3,
      test_reveal 2/2, test_empty_root_sentinel 1/1 unchanged.
      (`tests/test_multirow.py` fails on `pr._emit_id` — pre-existing, it calls
      the retired Python compile; verified broken before this work too.)
- [x] **S1b — fifth message: `s_bind` + the `p3` witness phase.**
      Third witness epoch through `_layout` / `_claim_var_groups` /
      `_stream_sweep` / `_stream_setup`, its own Merkle tree and `p3` block, a
      fifth sweep, and the coins `s_bind = H(s_op || root_p2)`,
      `s_comb = H(s_bind || root_p3)`.  Claims opt in through the new
      `LATE_SAMPLE_FNS` / `LATE_AUX_FNS` registries; a sweep without `ch1`
      cannot emit a phase-3 row at all, so no value can depend on a coin the
      prover has not seen.  A tape with no late-stage claim still frames R3
      with `EMPTY_COMMIT_ROOT`, so dropping the message changes the transcript.
      p_0's row map and the quad-ref hardening cover p3 rows.
      Gate PASSED: `tests/test_phase3_block.py` 5/5 (p3 commits, opens and
      Rust-ACCEPTs; the transcript really is `s_op -> s_bind(root_p2) ->
      s_comb(root_p3) -> s_col`; empty-p3 framing; tampered p3 column REJECT;
      dropped R3 message REJECT), with test_claims 21/21, test_fiat_shamir
      6/6, test_routing_claim 19/19, persistent-weights 3+3+3, test_reveal
      2/2, test_empty_root_sentinel 1/1 unchanged.
      Note: that suite exercises the BLOCK (an ordinary matmul with its
      Freivalds aux moved to phase 3), not a late-sampled relation — the first
      real `LATE_SAMPLE_FNS` user, with its own soundness tests, is S2.
- [x] **S2a — `RoutedProjectedMatmulClaim`.**
      `prover/routed_projected.py` (claim, early/late samplers, phase-2 and
      phase-3 aux, compile, active-only witness) and the Rust twin
      `compile_routed_projected` in `verifier/src/handlers.rs`, with `s_bind`
      threaded through `verify_bound` / `compile_claims_bound`.  Every band is
      an EXISTING packet template (identity, FreivaldsB, FreivaldsC,
      RowsumConst), so no new lowering or kernel was needed — the claim is a
      re-association, not a new proof system.  Measured constraint count is
      `E*K + 2T + 2E + 1` linear ids and `T*K + E` quads per matmul, i.e.
      exactly the `L_route`/`Q_route` terms of the admission ledger.
      One prover-wide change came with it: the streaming compile now runs ONCE
      after `s_bind`, so a late claim's early and late bands are emitted in the
      same pass and constraint ids never depend on compile timing.
      Gate PASSED: `tests/test_routed_projected.py` 6/6 — honest proof
      Rust-ACCEPTs; active-only output equals the dense reference; three
      malicious-prover simulations REJECT (fabricated output, output served by
      an unrouted expert, committed P != W*rho); the ledger count is asserted.
      Full suite green (test_claims 21, fiat_shamir 6, phase3 5, routing 19,
      persistent-weights 3+3+3, reveal 2, empty-root 1).
      Note: proving against SWAPPED weights is not a claim-level failure — the
      witness stays self-consistent — it is the enrolled root's job, checked by
      policy in S4.
- [x] **S2b — standalone `RescaleClaim`.**
      `prover/rescale_claim.py`: the same signed-floor relation the in-matmul
      rescale enforced (`x_full = 2^r*x + x_low`, `x_shifted = x + 2^(w-1)`,
      tight and loose range LogUps), now its own claim so it can follow the
      routed raw accumulator.  The Rust handler reuses the existing
      `Build::emit_rescale`, so both sides emit an identical layout.
      Gate PASSED: `tests/test_rescale_claim.py` 4/4 — routed output +
      standalone rescale Rust-ACCEPTs, the rescaled value equals the
      signed-floor reference, wrong rounding REJECTs (first linear), and an
      `x_low` pushed outside `[0, 2^r)` while keeping the linear satisfiable
      REJECTs (tight LogUp).  Full suite green.
- [x] **S3 — active-only MoE block + challenge-keyed `P` cache.**
      `demo/moe_routed.py` builds the MoE FFN from three
      `RoutedProjectedMatmulClaim` + `RescaleClaim` pairs instead of 128 gate,
      128 up and 128 down `tape.matmul` outputs and three `freivalds_combine`
      folds.  The claim now carries the expert weights as ONE VARIABLE PER
      EXPERT (a single (E,K,J) variable would be ~43 GB per layer at Maverick
      shapes), so both the witness pass and the projection stream shard by
      shard; the Rust handler emits one band per shard to match.
      `P = W*rho` is cached under a digest of rho (`clear_p_cache` runs from
      the new `core.PROVE_START_HOOKS` at the start of every proof), which is
      what turns four identical passes over the enrolled weights into one.
      Gate PASSED: `tests/test_moe_routed.py` 4/4 — the routed block's output
      equals the all-expert builder's element for element; growing E by 4 adds
      exactly `T*E + E*K + 3E` activation slots and NO `T x d_ff` term (the
      old builder pays that E times); the five witness epochs perform exactly
      1 projection with 3 cache hits and still Rust-ACCEPT; a different rho
      misses the cache.  Full suite green (claims 21, fiat_shamir 6, phase3 5,
      routed 6, rescale 4, moe 4, routing 19, persistent 3+3+3, reveal 2,
      empty-root 1).
      Not yet done here (landed in S4c): wiring this block into `demo_maverick_full.py` itself
      and the GGUF per-shard loader — that lands with the driver work in S4,
      where it can be exercised end to end.
- [x] **S4a — shard streaming, fused projection, no sink-less encodes.**
      Three review findings, all reproduced before they were fixed:
      (a) the routed claim declared its 128 expert shards as claim INPUTS, and
      the generic sweep pre-fetches every input — ~43 GB in one go for one
      Maverick matrix.  The shards are now claim FIELDS only
      (`core.STREAMING_INPUT_CLAIMS`); compute and aux get the raw `live` map
      with loaders unresolved and release each shard before the next.  The
      caching `_LazyResolvingDict` is bypassed for these claims for the same
      reason.
      (b) `_iter_message_chunks` kept the previous variable alive while the
      next loaded (a `carry` VIEW into the old tensor, plus plain rebinding
      evaluating the new loader before dropping the old value) — measured peak
      of 2 resident shards, now 1.
      (c) `_stream_phase` encoded rows even when every sink was None, so a
      REFERENCED weight commitment still paid a full RS pass over the enrolled
      model in R1 (402.7G slots, ~3625 s at the model's own rate).  It now
      returns immediately.
      Also fused: the sweep that projects reads each shard once and computes
      both the active output and its slice of `P = W*rho`.
      Gate PASSED: `tests/test_shard_streaming.py` 4/4 with 128 lazy loaders —
      peak resident shards = 1; exactly 5 semantic sweeps, 1 projection, 4
      cache hits, Rust ACCEPT; fusing saves exactly E shard loads (one whole
      weight pass); referencing an enrolled commitment saves exactly `m_w`
      encoded rows.  Full suite green.
- [x] **S4b — ZK entropy, canonical statement, fail-closed policy, safe handle.**
      Four more review findings:
      (a) padding and the three blinding rows came from `MASTER_SEED`, a public
      constant in `core.py` — a verifier who knows it reconstructs every mask
      and strips it off the openings, so the proof was sound but NOT zero
      knowledge.  Proofs now draw fresh secret entropy per run (`new_zk_seed`,
      `secrets.token_bytes`); the enrolled weight block keeps padding under its
      own ENROLLMENT seed, which `WeightCommitment` now generates fresh and
      secret by default.  Tests that compare two proofs' roots pin the seed
      explicitly.
      (b) the "static" statement digest covered `Table.alpha/beta`, which are
      per-proof LogUp challenges and MUTATE on the tape after the first proof —
      two proofs of one claim set could disagree on the digest.  They are out
      of the canonical bytes now (the verifier re-derives them from `s_op` and
      already overwrote whatever the JSON said), and the digest additionally
      covers the row-block layout, so a proof cannot relabel its own blocks.
      (c) `verify_proof` was fail-OPEN: with no trusted statement digest the
      prover picks what it proves, and with no enrolled root it picks its own
      weights.  Both are now required — a proof carrying a statement digest
      without a policy digest, or a weight block without an enrolled root,
      REJECTs.
      (d) `WeightCommitment.load` was `pickle.load` — arbitrary code execution
      from a file read before anything about it is checked.  Replaced by a
      framed binary format plus `check_topology()`, which rebuilds the tree
      from its leaves and refuses a handle whose levels do not produce the
      stored root.
      Gate PASSED: `tests/test_fiat_shamir.py` 7/7 (adds missing-policy
      REJECT), `test_persistent_weights_p5` 3/3 (adds pickle-refused and
      broken-tree-refused), full suite green.
- [x] **S4c — driver, admission gate, no reveal pass, streamed openings.**
      `demo/demo_maverick_full.py` now builds the MoE FFN from three routed
      claims + their rescales (the 128-output lists and three
      `freivalds_combine` folds are gone) and its expert shards are enrolled
      persistent variables.  New modes and policy:
      `--enroll-weights` commits the model once and prints the root;
      a proof REFUSES to start without `--weight-commitment`,
      `--expected-weight-root`, `--public-sz` and `--admission-report`, and
      those are checked before the model is even built.
      The pre-proof reveal pass is gone: `--public-sz` comes from the already
      fixed serving statement and is pinned into the reveal claim before R1,
      so there is no sixth semantic sweep.
      The Ligero geometry is pinned (`admission.TARGET`, `T_QUERIES=54`); a
      non-target config refuses unless `--allow-dev-config`.
      `prover/admission.py` is the fail-closed gate: the report must bind to
      the source digest, model root, statement digest, machine fingerprint and
      the row manifest of the layout just built, carry the admission module's
      distribution-free sample count and bound kind,
      bounds measured on real GGUF, and come in under every per-stage cap.
      Openings no longer accumulate on the GPU: `ColumnSink` writes each
      chunk's slice straight into a pre-sized host buffer, so there is no
      final `torch.cat` (which briefly doubled tens of GB of openings).
      Gate PASSED on the pre-S4f tree: `tests/test_admission_gate.py` 11/11 —
      an honest report at
      cap is admitted, and a report that is over cap by 1 ms, from another
      GPU, another build, another model, another statement, a smaller ELL or
      half the rows, insufficient runs, averages instead of upper bounds, random
      weights, a missing stage, an unpriced stage or without a machine binding
      is refused, as is a dev geometry.  Driver refusals checked directly
      (wrong config, missing policy).  Full suite green.
      `--enroll-weights`, `--weight-commitment`, `--expected-weight-root`,
      `--public-sz`, `--admission-report`; drop the hidden pre-proof reveal
      pass; `verify_proof` takes the trusted weight root and statement digest as
      REQUIRED arguments.
      Gate: proof with a wrong `--expected-weight-root` or a wrong statement
      digest rejects; missing policy rejects; streaming writer verified not to
      materialize Python ints.
- [x] **S4e — opening ledger and atomic proof output.**
      The last two review items.  Every proof against one enrollment opens
      columns of the SAME weight rows under the SAME padding, so what leaks is
      the CUMULATIVE set of distinct columns; nothing in the system noticed,
      because the proofs keep verifying.  `WeightCommitment` now carries an
      opening ledger (persisted in the handle), refuses a proof that would
      spend past `(K_DEG-ELL)/2` distinct columns — 4096 at the production
      geometry, about 75 proofs at 54 per proof — and says to refresh the
      enrollment.  On a config with no usable slack the ledger stands down,
      because such a config is not zero knowledge to begin with.  The driver
      books and saves the ledger after every proof.
      Proof output is atomic: the size is estimated, the free space checked,
      the document written to `<path>.part`, fsynced, and renamed — so a run
      that dies out of disk cannot leave a truncated file that looks like a
      proof.  The commitment loader reports any malformed handle as a
      corruption instead of a struct traceback.
      Gate PASSED on the pre-S4f tree: `tests/test_opening_ledger.py` 5/5
      (ledger records and
      survives save/load; an exhausted budget refuses BEFORE any weight column
      is produced and names the remedy; the budget scales with the pad;
      atomic write leaves no `.part`; a proof larger than the free space is
      refused with nothing written).
- [~] **S4f — quadratic reassociation, compact proof wire, transactional output.**
      Implemented in this handoff, but NOT marked DONE until the new CUDA and
      Rust gates run on the rented machine. `compute_p_0_streaming` now uses
      linearity of the inverse NTT: it accumulates the random linear
      combination of quadratic products in the 2K evaluation domain and pays
      one inverse NTT at the end, instead of one per constraint row. Public
      constant rows are interpolated once per distinct `(n,c)`. This changes
      neither `p_0` nor the verifier equation; `test_quad_eval_accumulator`
      compares it coefficient-for-coefficient to the literal construction.

      Production field vectors are JSON strings containing canonical `u64le`
      bytes in base64. The Rust verifier accepts this form and legacy decimal
      arrays and reconstructs the same `Vec<u64>` before every check. At the
      exact cap this changes modeled drain from 879.630 s to 481.481 s; model
      total is 12,957.864 s (3.5994 h), margin 1,442.136 s.

      Output is reserved with `posix_fallocate` before proving. Access to an
      enrollment is locked across the run; after proving, its opening ledger
      is atomically saved and fsynced BEFORE the proof is atomically published.
      Failure after ledger save spends budget conservatively instead of
      publishing unrecorded leakage.

      Admission rejects NaN, infinity and negative durations. The old “30
      samples imply simultaneous p99/99%” statement was false: without a
      parametric tail assumption the required stagewise-max count is 714 after
      Bonferroni over 13 stages. Thirty-run campaigns are now exploratory and
      cannot authorize production.
      Reports additionally bind GPU UUID/driver/PCI id/power limit and the
      actual proof-output filesystem; timing `/tmp` cannot authorize a write
      to a different disk.

      **Local CUDA/Rust gates RUN (2026-08-06, V100-SXM3-32GB).**  Everything
      the handoff listed as the load-bearing next action, except the rented
      card: `cargo build --release` clean, `cargo test --release --bin
      verify_proof` 1/1 (compact-wire roundtrip + bad padding), and
      test_quad_eval_accumulator 1/1, test_opening_ledger 6/6,
      test_admission_gate 13/13, test_pipeline_integration 3/3,
      test_fiat_shamir 7/7, test_phase3_block 5/5, test_routed_projected 6/6,
      test_rescale_claim 4/4, test_moe_routed 4/4, test_shard_streaming 4/4,
      test_claims 21/21.

      Both stages that failed S5a now clear their caps ON THE SAME BOX as the
      earlier V100 measurement, so this is a like-for-like A/B, 30 runs
      (exploratory, not a report):

      | stage | V100 pre-S4f | V100 S4f | cap |
      |---|---|---|---|
      | quadratic | 821.7 s (18.28 ns/product) | **703.8 s (15.64 ns)** | 765 |
      | proof_egress | 2114 s (44.9 MB/s) | **455.3 s (114.2 MB/s)** | 879.6 |

      Other kernel stages unchanged (fresh_commit_fold 492.0, fresh_hash_coef
      32.5, persistent_open 97.2, fresh_open 24.1).  The quadratic win is the
      reassociation; the egress win is the compact wire (52 GB modeled instead
      of 95 GB) plus base64 in C rather than decimal formatting in Python.

      What this does NOT show: `persistent_weight_qlin` (cap 3624.5 s, the
      largest single kernel stage) is still `NOT MEASURED` by the kernel
      campaign, as are `linear` and every model-dependent stage.  So "the two
      failing stages now pass" is exactly that claim and not "the envelope
      closes".

      **CONFIRMED on a rented A100-SXM4-40GB** (2026-08-06, $0.61/h, 476 s
      total ≈ $0.08, instance 46975810 destroyed and confirmed gone).  All 14
      gate suites ran on the rented card with `gate_failures: 0`, so nothing in
      S4f is V100-specific.  Kernel rates, 30 runs (exploratory):

      | stage | A6000 pre-S4f | V100 S4f | A100 S4f | cap |
      |---|---|---|---|---|
      | quadratic | 800.2 | 703.8 | **686.0** | 765 |
      | proof_egress | 979.8 | 455.3 | **165.1** | 879.6 |
      | fresh_commit_fold | 428.4 | 492.0 | 511.9 | 950 |
      | fresh_hash_coef | 29.8 | 32.5 | 22.9 | 140 |
      | persistent_open | 68.7 | 97.2 | 78.7 | 1812 |
      | fresh_open | 17.1 | 24.1 | 19.5 | 450 |

      Both S5a failures are therefore fixed on both cards measured, with the
      A100 at 0.90x and 0.19x of cap.  One honest oddity: encode is SLOWER on
      the A100 than on the A6000 (5.12 vs 4.28 ns/slot) despite twice the
      memory bandwidth, so the commit path is not bandwidth-limited on that
      card — untuned for sm_80, or clock/power capped.  It has 1.9x margin, so
      it is a note, not a blocker.

      The compact-wire cap is now derived rather than asserted: (weight rows +
      fresh rows) x 54 columns x 11 B/value = 36.45 GB against the 52 GB cap,
      checked by an assert in the model.
- [~] **S5 — admission harness (partial: kernel stages measured, model stages not).**
      `analysis/bench/admission_bench.py` measures the EXECUTED loop bodies at
      the target geometry (ELL=8192, K_DEG=16384, N_LIG=65536). A 30-run
      campaign is an exploratory optimization check; production needs the
      sample count and method required by `prover/admission.py`. It converts
      each per-slot rate into the stage seconds the model caps.
      It deliberately leaves `model_load` and the five semantic sweeps as
      `null` — they need real GGUF shards — so the report it writes is
      incomplete and the gate refuses it, which is the correct outcome.

      MEASURED on the local V100-SXM3-32GB (p99-conservative bounds,
      mean+3sd floored at the observed max):

      | stage | measured | cap | verdict |
      |---|---|---|---|
      | fresh_commit_fold | 491.6 s | 950 | ok |
      | quadratic | 821.7 s | 765 | **OVER 1.07x** |
      | fresh_hash_coef | 32.7 s | 140 | ok |
      | persistent_open | 94.1 s | 1812 | ok |
      | fresh_open | 23.4 s | 450 | ok |
      | proof_egress | 2114 s | 879.6 | **OVER 2.40x** |

      Per-slot rates: encode 4.92 ns, hash 0.33 ns, open 0.23 ns,
      quad 18.28 ns/product, egress 44.9 MB/s.

      HISTORICAL result for the pre-S4f tree. The egress failure was CPU-bound:
      the streaming decimal
      JSON writer. Rendering each chunk with json.dumps (the C encoder)
      instead of a Python `",".join(str(v) ...)` is byte-identical output and
      measures 79 MB/s vs 47 MB/s single-run — but the p99 bound only moved
      44.9 from 42.0, because run-to-run I/O variance dominates the bound on
      this 4-vCPU box. Byte-level alternatives were slower
      (b",".join of %d: 38.5 MB/s; numpy.savetxt: 14.9 MB/s).
      So on THIS box the 4-hour envelope does not close, and the two failing
      stages are known before any money is spent.
      MEASURED on a rented RTX A6000 (vast, $0.404/h, 654 s total = $0.07,
      instance destroyed and confirmed gone). All 12 gate suites passed on the
      rented card (`gate_failures: 0`), so nothing in the build is
      V100-specific.

      | stage | V100 | A6000 | cap | A6000 verdict |
      |---|---|---|---|---|
      | fresh_commit_fold | 491.6 | 428.4 | 950 | ok |
      | quadratic | 821.7 | 800.2 | 765 | **OVER 1.05x** |
      | fresh_hash_coef | 32.7 | 29.8 | 140 | ok |
      | persistent_open | 94.1 | 68.7 | 1812 | ok |
      | fresh_open | 23.4 | 17.1 | 450 | ok |
      | proof_egress | 2114 | 979.8 | 879.6 | **OVER 1.11x** |

      Rates: encode 4.28 ns/slot, hash 0.30, open 0.17, quad 17.78 ns/product,
      egress 97.0 MB/s.

      The finding for commit `0279686`: BOTH stages that failed on the dev box
      also failed on a modern
      card, by 5% and 11%. The kernels are bandwidth-bound and the A6000
      (768 GB/s GDDR6) is not a step up from a V100 SXM3 on that axis — encode
      moved only 4.92 -> 4.28 ns/slot. So the 4-hour envelope does not close on
      either card measured so far, and the gap is small enough that it is a
      real engineering question, not a rounding error. S4f changes exactly
      those two bodies, so this report remains a baseline but is invalid as
      admission evidence for the new source digest.
- [ ] **S5b — the model-dependent stages, on the rented card.**
      Benchmarks the exact production loop bodies (no random matrices, no
      isolated modmul), with the sample count/method required by
      `prover/admission.py`, writes
      `admission.json` bound to source digest, model root, statement digest,
      CUDA machine fingerprint and the actual row manifest.  Driver refuses to
      start when any stage bound exceeds its cap.
      Gate: the gate itself is tested by feeding it a deliberately over-cap
      report and a report from the wrong machine — both must refuse.
- [ ] **S6 — production run.**
      Enroll -> admission -> proof -> standalone `verify_proof`.  Only after S5
      is green on the rented machine.

## Open risks (stated now, not discovered later)

1. The five-sweep semantic cap (27.278G decoded real-GGUF MAC/s aggregate) is
   now the largest completely unmeasured term. Compact proof egress and the
   optimized quadratic accumulator must be remeasured on the rented machine;
   neither is inferred from the historical A6000 report.
2. GGUF availability: only shard 1 of 5 (21 GB) is present locally as of
   2026-08-05.
3. S1 is a protocol change to a prover whose soundness rule is
   commit-before-challenge.  It is the highest-risk stage and gets its own
   tamper tests before anything is built on top of it.
