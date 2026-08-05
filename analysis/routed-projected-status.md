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
`analysis/routed_projected_4h_model.py` runs and closes at 13,356.012 s = 3.7100 h.
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
      Not yet done here: wiring this block into `demo_maverick_full.py` itself
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
- [ ] **S4b — driver: enrollment, policy, streaming proof.**
      `--enroll-weights`, `--weight-commitment`, `--expected-weight-root`,
      `--public-sz`, `--admission-report`; drop the hidden pre-proof reveal
      pass; `verify_proof` takes the trusted weight root and statement digest as
      REQUIRED arguments.
      Gate: proof with a wrong `--expected-weight-root` or a wrong statement
      digest rejects; missing policy rejects; streaming writer verified not to
      materialize Python ints.
- [ ] **S5 — admission harness on real GGUF.**
      Benchmarks the exact production loop bodies (no random matrices, no
      isolated modmul), >=30 runs, simultaneous >=99% upper bounds, writes
      `admission.json` bound to source digest, model root, statement digest,
      CUDA machine fingerprint and the actual row manifest.  Driver refuses to
      start when any stage bound exceeds its cap.
      Gate: the gate itself is tested by feeding it a deliberately over-cap
      report and a report from the wrong machine — both must refuse.
- [ ] **S6 — production run.**
      Enroll -> admission -> proof -> standalone `verify_proof`.  Only after S5
      is green on the rented machine.

## Open risks (stated now, not discovered later)

1. The five-sweep semantic cap (27.278G decoded real-GGUF MAC/s aggregate) and
   the 95 GB decimal-JSON drain at 108 MB/s are the two caps most likely to
   fail; both are CPU/loader bound, and the local box has 4 vCPU.  They must be
   measured on the rented machine before anything is promised.
2. GGUF availability: only shard 1 of 5 (21 GB) is present locally as of
   2026-08-05.
3. S1 is a protocol change to a prover whose soundness rule is
   commit-before-challenge.  It is the highest-risk stage and gets its own
   tamper tests before anything is built on top of it.
