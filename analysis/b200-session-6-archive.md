# B200 session 6 — archive (2026-09-30)

The session for the saved S=1000 proof: the bridged S=1000 prove with its
proof written out and checked by the Rust verifier from PR 21's final head,
the bridged S=100 arm under the thread cap session 5 asked for, and the
gates on the tree that became PR 21. One NVIDIA B200 (RunPod secure cloud,
US-NC-2, 180 GB HBM, a 2 TB host with a 251 GB cgroup and a 20.4-CPU quota
against 192 visible cores), 400 GB container disk, host CUDA 13.2 (driver
595.91) under the cu128 image, 17:45 to about 21:25 UTC, about 3.6 hours
at $6.79/h. Repository state: the prover tree is `62c1cf4` (a tarball of
`7bb00c2` plus the three test files fixed during phase 0, below); the
S=1000 check ran `verify_proof` built on the pod from `c5d72e5`, the head
PR 21 merged (verifier/ tree `2846cd7dd65c`), with
`tools/check_dumped_proof.py` from the same revision. The thread pools
were capped to the quota at bootstrap (`OMP_NUM_THREADS`,
`MKL_NUM_THREADS`, `RAYON_NUM_THREADS` = 20), and the host sampler ran
from bootstrap to the end (`logs/sampler.log`).

## What ran

1. Bootstrap (`logs/phase0.log`), then the twelve GGUF-free suites before
   the pull. Two failures, both in tests and both fixed and shipped during
   the session: the weight-cache suite's verifier helper dumped proofs
   onto an existing temporary file, which the dump's overwrite refusal
   rejects (`aedaa7b`); the bridge suite expected the identity reason for
   a redeclared boundary that the form check now refuses first
   (`62c1cf4`). The third run passed every suite: the bridge 17/17, the
   streaming flag 10/10, our Rust negatives 8/8, Fiat–Shamir 7/7, shard
   streaming 7/7, weight cache 5/5, phase-3 block 5/5, instrument
   bookkeeping 3/3, pipeline integration 3/3, the empty-root sentinel
   1/1, the tape link 1/1, the sampled-audit runtime 85 (`logs/phase0b.log`
   and the `*.fail*.log` files).
2. The pull: 232.13 GB in 5.3 minutes (724 MB/s), every shard hashing to
   its pin. No calibration this session.
3. The gate script (`logs/gates.log`): all thirteen gates; the three
   two-layer real-GGUF proofs accepted by the Rust verifier.
4. One bridged S=100 arm (`logs/mavp-s100-bridge6.log`): session 5's
   command, now with decode-once and the compact wire in the tree.
5. The bridged S=1000 prove (`logs/mavp-s1000-bridge.log`), routed and
   weight caches on, with `--dump-proof`; then the Rust check of the saved
   file (`logs/mavp-s1000-rust-check.log`), its receipt
   `mavp-s1000.bin.verify.json` and the same run's policy sidecar
   `mavp-s1000.bin.policy.json`. The proof itself (19,667,647,609 bytes)
   is kept off the repository; the copy's SHA-256 matches the receipt.

## Results

The bridged S=100 arm across three hosts (sessions 4 and 5 uncapped):

| item | session 4 (US-CA-2) | session 5 (US-NE-1) | session 6 (US-NC-2) |
|---|---|---|---|
| CPU quota; torch threads | 23.8; 112 | 30.6; 144 | 20.4; 20 (capped) |
| throttled CPU-seconds over the arm | 20 | 8,120 (build 581, enrollments 1,559, reveal 1,723, prove 4,256) | 0 |
| streaming enrollment of every expert shard | 273.4 s | 133.9 s | 119.8 s (shard decode 54.4 s / 9,218) |
| the bridge's per-proof pass | not separated: 592.3 s outside the sweeps altogether (987.8 s wall less 395.5 s of sweeps) | 198.5 s | 126.6 s: NTT 41.6, shard decode 36.7 / 9,218, columns concat 15.6, columns to host 6.9, projected-mask matvec 4.9, gather 4.5, masks 4.4 + 4.3, pack 4.1 |
| prove wall | 988.0 s | 1,271.2 s | 809.9 s |
| five sweeps | 395 s (fetch 109, encode 136, compile 22) | 633 s (fetch 290, encode 136, compile 65) | 642 s (fetch 367, encode 136, compile 1.4) |
| outside the sweeps, not the bridge | unmeasured (within the 592.3 s) | ~440 s | 41 s |
| build, dense enrollment, reveal | 19.1, 48.0, 37.3 s | 41.2, 134.1, 96.4 s | 39.4, 83.6, 104.2 s |
| Rust verify of the three gate proofs (plain, wc, bridged) | 272, 272, 65 s | 230, 230, 67 s | 583, 546, 131 s |
| leaf check | true | true | true |

The S=1000 prove against session 2's unbridged arm with the caches off on
an H200 (a different card and configuration, not an A/B):

| item | session 2 (H200, caches off, no bridge) | session 6 (B200, bridge, routed and weight caches) |
|---|---|---|
| prove wall | 5,333.5 s | 1,780.2 s (29.7 min) |
| sweeps R1 / R2 / R3 / fold / open | 401 / 536 / 438 / 2,173 / 1,550 s | 302 / 178 / 62 / 701 / 280 s |
| loaders, encode | 2,062 s, 1,599 s | 123 s (every dense load after R1 a cache hit, 3,620), 543 s |
| witness, quad, fold_qlin | 384, 365, 463 s | 293, 340, 89 s |
| enrollment, reveal | 926 s, 246 s | dense 138.7 s + streaming 116.4 s, 183.1 s |
| peak GPU | 79.7 GiB | 152.64 GiB |
| proof, dump | 35.46 GB in 43.1 s | 19.67 GB in 30.4 s (646 MB/s) |
| Rust verifier | not run | ACCEPT: 5,385 s on 20 threads (`lin_col` 88 of its 90 minutes, 18,899,057 rows), 5,661.7 s with the parse; peak RSS 27.2 GB |

The S=1000 bound: 21.7194 bits/token over 500 scored positions. Host
memory at S=1000: anonymous memory peaked at 32.7 GB beside 206 GB of the
mapped model in page cache, under the 251 GB limit; over the arm the
memory-pressure counter recorded 29 s of stall and the IO counter 0.1 s.

## What it means

The S=1000 result is a Rust-accepted S=1000 proof on synthetic token IDs,
checked against policy values produced by the same run. It is not an
independent enrollment of the model and not the real-token 0.880
bits/token result. That remains August's production run on an A100
(`analysis/routed-projected-status.md`, S5c/S6): real tokens, an
independent enrollment, 2,596 claims, a 7,352.9 s prove and a 35.46 GB
proof accepted in 6,661 s on 30 threads. What this session adds is the
bridge at S=1000 and a saved proof checked, after the prover exited, by a
verifier built from the merged revision rather than from the prover's
tree.

The thread cap holds the throttling at zero: 16 throttled CPU-seconds
across the whole 3.5-hour session, against 9,608 in session 5's 70
minutes. Alongside it, the time outside the sweeps that is not the bridge
was 41 s against about 440 s on session 5's host, the fold sweep's claim
compile 1.4 s against 65 s, and the S=100 prove wall is the lowest bridged
one so far, on the smallest quota of the three hosts. Session 4's
uncapped host throttled for only 20 CPU-seconds, so the cap is shown safe
and sufficient here, not necessary on every host; a capped and an
uncapped arm on one host would separate the two.

What the cap did not help is CPU-bound work that scales with the quota.
The loaders took 367 s against session 5's 290 s and session 4's 109 s,
with no IO or memory stall to explain it (0.1 s and 5.7 s over the arm):
they are decodes on the CPU, and this host had the fewest CPUs. The build
and the reveal stayed near 40 s and 100 s, as on session 5's host and two
to three times session 4's. The Rust verifier ran the gate proofs two to
two and a half times slower than in session 5, and at S=1000 its
`lin_col` pass dominates: 90 minutes of verification against a 30-minute
prove (August's run verified in about the time it proved). The verifier,
not the prover, is now the long pole of an S=1000 session on a 20-CPU
host. The host's CPU model was not recorded; the environment record
(`tools/session2_env.sh`) now prints `lscpu`'s model line and the visible
core count, which `nproc` hid once the cap was set.

Decode-once and the compact wire did what they were for: shard decodes
halved (18,434 to 9,218), the 46 s conversion to Python integers is gone,
and the bridge's per-proof pass fell from 198.5 s to 126.6 s. It did not
reach the projected 100 s: the NTT (41.6 s) is unchanged and the decode
and the column concatenation remain.

## Caveats

- The protocol review of 2026-09-25 found soundness gaps in the causal
  softmax mask and the SiLU branch index and zero sign (F01, F03, F04)
  that every proof of this session shares, the S=1000 proof included.
  The repairs change the constraint set, and these proofs do not verify
  under it. The session's tail ran the repair branch's GPU suites; their
  logs travel with that branch.
- The three S=100 hosts differ in quota and in CPUs. The throttling
  counts are direct measurements; the walls compare hosts, not changes.
- The S=1000 comparison with session 2 crosses cards and configurations.
- The bridge numbers remain for the interim form (folds in the clear).
