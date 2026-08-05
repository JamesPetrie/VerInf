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
- [ ] **S1 — sequential Fiat--Shamir, five messages, `p3` phase.**
      Replace the pre-derived `round_seeds(SEED)` with a framed transcript hash:
      `R1 -> s_op -> R2 -> s_bind -> R3 -> s_comb -> test polys -> s_col`.
      Adds a third witness phase to `_layout`/`_stream_sweep` and a fourth seed
      to the Rust verifier, which recomputes every seed itself.
      Gate: toy proof still `verify_proof=ACCEPT`; tamper tests reject; a test
      that proves `s_bind` cannot be derived before the R2 root exists.
- [ ] **S2 — `RoutedProjectedMatmulClaim` + standalone `RescaleClaim`.**
      Python claim (sample/aux/compile) + Rust handler, relations `P=W rho`,
      `H=X*Q`, `yr=Y rho`, `sum_k H = yr`, late Freivalds `sum_e f_p =
      sum_{t,k} lam_t Q sig_k`.
      Gate: unit tests on a toy MoE, Rust ACCEPT, and tamper tests for each of
      the five relations plus a wrong-route witness.
- [ ] **S3 — active-only MoE builder + challenge-keyed `P` cache.**
      One GGUF expert shard at a time, only tokens routed to it; delete the
      128-output lists and the three `freivalds_combine` calls.
      Gate: byte-identical model outputs vs. the current builder on a small
      real-GGUF slice, and a structural test asserting no all-expert tensor is
      ever allocated.
- [ ] **S4 — driver: enrollment, policy, streaming proof.**
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
