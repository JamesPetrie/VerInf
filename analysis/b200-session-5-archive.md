# B200 session 5 — archive (2026-09-19/20)

A short session for one question: where the bridge's per-proof pass goes,
and whether moving the mask coefficients onto the card removes most of it.
One NVIDIA B200 (RunPod secure cloud, US-NE-1, 180 GB HBM, a 3 TB host with
a 377 GB cgroup and a 30.6-CPU quota against 288 visible cores — session
3's host class, not session 4's), 400 GB container disk, host CUDA 13.0,
the cu128 image, 22:47 to 23:58 UTC, about 1.2 hours at $6.79/h.
Repository state: `3e26e98` on `weight-split-model` as a tarball — the
pushed tip plus the two commits under test: stage timers in
`prover/wc_bridge.py` and the masks from `cuda_primitives.row_prg`. The
host sampler ran throughout (`logs/sampler.log`).

## What ran

1. Bootstrap (`logs/phase0.log`), then before the pull the seven GPU-free
   suites: instrument bookkeeping 3/3, weight cache 5/5, shard streaming
   7/7, the bridge's standalone suite 15/15 (its roots recomputed under the
   new masks), the streaming-flag suite 9/9, the tape link 1/1, our Rust
   negatives 6/6 — every suite on the first run.
2. The pull (probe 124 MB/s, 217 GB in twenty minutes, every shard hashing
   to its pin). No calibration this session.
3. The gate script (`logs/gates.log`): all gates, the bridged two-layer
   proof printing its stage lines and accepted (prove 69.0 s, enrollment
   4.9 s against 11.2 s in session 4, the bridge pass 6.6 s).
4. One bridged S=100 arm (`logs/mavp-s100-bridge5.log`): the same command
   as session 4's D4 arm, the routed cache on, the weight cache off.

## Results

| item | session 4 (US-CA-2 host) | session 5 (US-NE-1 host) |
|---|---|---|
| streaming enrollment of every expert shard | 273.4 s | 133.9 s: shard decode 71.2, NTT 41.0, Merkle 11.5, masks 4.2, pack 4.1 |
| the bridge's per-proof pass | 592 s (uninstrumented; the whole "outside the sweeps") | 198.5 s: shard decode 66.5, columns to Python ints 46.0, NTT 41.0, columns to host 5.9, projected-mask matvec 4.9, gather 4.8, masks 4.3 + 4.2, pack 4.2, aggregate 0.1, eval 0.0 |
| prove wall | 988.0 s | 1,271.2 s |
| five sweeps | 395 s (fetch 109, encode 136, compile 22) | 633 s (fetch 290, encode 136, compile 65) |
| outside the sweeps, not the bridge | ~110 s (as in every session-4 arm) | ~440 s |
| build, dense enrollment, reveal | 19.1, 48.0, 37.3 s | 41.2, 134.1, 96.4 s |
| leaf check, verifier on the gate proof | true, ACCEPT | true, ACCEPT |

## What it means

The mask fix works and the timers say what is left. The CPU random draw
that produced the mask coefficients three times per proof is gone from
the accounting: 4.2 s per pass on the card against roughly 115 s before.
The bridge's per-proof pass is 198.5 s, of which two thirds are now two
things this session did not touch: the column pass decodes every expert
shard twice, once per output-width pass of the stream, 66.5 s for 18,434
decodes of 9,216 shards; and it ends by converting about a billion column
values to Python integers, 46 s, which is the same conversion that makes
the wire section plain JSON. The batched NTT is 41 s and close to the
kernel's own rate. Decoding each shard once and keeping the columns as
tensors until the compact dump would put the pass near 100 s.

The prove wall did not fall, and the host explains it. The sampler's
cgroup counters show the process throttled for thousands of CPU-seconds
during the enrollments and the reveal pass and for over a thousand during
the prove: torch started 144 threads against a 30.6-CPU quota (112 against
23.8 on session 4's host), and every CPU-side phase paid — the build
41 s against 19, the dense enrollment 134 s against 48, the reveal 96 s
against 37, the claim compile 65 s against 22, the loaders 290 s against
109, and 440 s of proof assembly outside the sweeps against 110. This is
the mechanism behind session 3's 303 s build on the same host class. The
remedy is an environment line, not code: cap the thread pools to the
quota at bootstrap (`OMP_NUM_THREADS`, `torch.set_num_threads`), which
the runbook now states as the first thing the next session measures.

## Caveats

- Cross-host comparisons with session 4 are for the bridge pass and the
  enrollment, which the timers measure directly; the prove wall is not
  comparable across the two hosts.
- The mask change redefines the enrollment root; no registration
  predates it.
- The bridge numbers remain for the interim form (folds in the clear).
- The thread cap is untested: its effect is the next session's first
  measurement, on a bridged S=100 arm with the cap set.
