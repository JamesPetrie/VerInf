# B200 session 3 — archive (2026-09-16)

The third rented session: the Blackwell calibration deferred from session
2, the gates with the decoded-weight cache, and the D3 loader A/B at
S=100, on one NVIDIA B200 (RunPod secure cloud, US-NE-1, 180 GB HBM, a
3 TB host with a 377 GB cgroup and 36 vCPUs allotted while the container
sees 288, 400 GB container disk, host CUDA 13.2 under the cu128 image
`runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`), about 4.3 hours at
$6.79/h, $29.00. The session ended when the account balance ran out
after the D3 arms, with the author asleep; nothing planned after D3
was lost except the S=1000 prove with both caches, which the D3 result
had already made moot. Repository state: `5d438fe` on
`weight-split-model` as a tarball on the pod, plus two files shipped
during the session (a test constant and the host-allocator stats key,
both in the follow-up commit) and, at the end, the cache's GPU tier.
The session followed `profiler/RUNBOOK-blackwell.md` "Session 3" with
one change: the GPU-free gates ran right after the bootstrap, before
the pull, so a wrong count assertion would cost minutes, not an hour.

## What ran

1. Bootstrap on the image's torch 2.8.0+cu128 (capability 10.0), the
   verifier built, the CUDA primitives JIT-compiled and their 25 tests
   passed (`logs/phase0.log`). Record: `session3-env.txt`; packages:
   `session3-pins.txt`. Then, before the pull, `test_weight_cache` (4/5
   on the first run: my boundary test's budget constant was wrong, the
   cache itself passed; 5/5 after the fix, `logs/early-weight-cache*.log`)
   and `test_shard_streaming` 7/7.
2. `profiler/calibrate.py --name b200-runpod-s3 --skip-io`, overlapped
   with the pull, then `--io-only` on idle storage. Output:
   `profiler/machines/b200-runpod-s3.json`, raw bench logs in
   `profiler/machines/calibrate-raw-b200-runpod-s3/`.
3. `tools/gguf_pull.sh`: the first probe measured 87.9 MB/s over eight
   streams and refused the pull below the 100 MB/s floor
   (`logs/gguf-pull.log`); the relaunch probed 129.1 MB/s and pulled the
   five shards at the pinned revision `41032e5`, every size and sha256
   matching the pins (`logs/gguf-pull2.log`).
4. `tools/session2_gates.sh`: the K-quant kernel 4/4, shard streaming
   7/7, routed 6/6, the weight cache 5/5, the toy A/B's counts, and TWO
   two-layer real-GGUF proofs through `profiler/instrumented_prove.py
   --verify`, cache off and on, both accepted (`logs/gates.log`,
   `logs/gate-*.log`).
5. D3, the loader A/B at S=100 (50+50), the routed cache and the group
   memo on both arms, the decoded-weight cache off then on with its
   pinned host tier, each arm with its own throwaway enrollment and
   reveal pass, the per-sweep table with the kind lines on
   (`logs/mavp-s100-wc-off.log`, `logs/mavp-s100-wc-on.log`,
   `ab-s100-compare.txt` from `ab_compare.py`).
6. After the on arm: a store-cost micro-benchmark on the pod (pinned,
   pageable and GPU stores of a 105 MB and a 168 MB int32 tensor) and
   the cache's GPU tier written, shipped and gated 5/5 (`prover/tests/
   test_weight_cache.py`); its A/B did not run.

## Results

| item | result |
|---|---|
| memory bandwidth (torch d2d, read+write) | 6,497.4 GB/s |
| BLAKE3 register-resident compress | 18.77 Gc/s (column bench 11.58 Gc/s, 741.0 GB/s) |
| field mul / NTT single / NTT batched (largest batch) | 678.1 Gmul/s / 0.173 ns/elem / 0.0553 ns/elem — batched is 3.84x the bandwidth-scaled expectation: encode is NOT bandwidth-scaled on this part either |
| derived constants | A 0.309 ns/slot, B 0.0533 ns/cid, C 0.515 ns/product (h200-runpod-s2: 0.471 / 0.0961 / 0.785; the 2026-08-19 b200-runpod: 0.308 / 0.0677 / 0.513) |
| storage (container overlay) | sequential read 1.21 GB/s, compact dump 748 MB/s (probe), proxy 288 MB/s, H2D 55.5 GB/s |
| two-layer proofs (gates), prove wall | off 118.6 s, on 109.1 s; dense loader 24.9 s over 187 calls off, 4.9 s over 17 calls plus 170 hits on; 17 weights 10.6 GB packed, 12.1 GB pinned; both ACCEPT |
| D3, S=100, prove wall | off 2,492.8 s, on 2,966.6 s (+19.0%); peak 48.4 GiB both; enrollment 851.5 / 839.3 s, reveal 82.9 / 76.2 s, same public bound; build 41.8 / 303.3 s; leaf checks true |
| D3 off arm, loader by kind | dense weights 631.0 s over 3,982 resolutions (362 in R1, 724 in R2 and R3, 1,086 in the fold and the opening) = 0.16 s each WITH the group memo; shards 114.4 s over 29,988 |
| D3 on arm, the cache | 362 weights decoded once, 64.7 GB packed, 92.4 GB pinned (the allocator's own count agreed), 0 refused, 3,620 hits; R1's 362 first resolutions 693.7 s = 1.9 s each (off: 0.22 s); shards 277.9 s (2.4x the off arm); R3's fetch 2.8 s for 724 hits |
| D3, where the off-arm prove goes | sweeps 2,378.5 s: encode 1,194.1, fetch 724.9, fold_qlin 277.0, witness 54.9, aux 8.1; fold sweep 1,001 s, open 999 s |
| store micro-benchmark (pod, in isolation) | 105 MB int32: pinned store 115 ms, pageable 59 ms, GPU clone 0.2 ms; load pinned 3.5 ms, pageable 13.1 ms. 168 MB: 101 / 106 / 0.2 ms; 3.2 / 17.3 ms |
| cgroup during the on arm | memory.current 327 GB of 377; memory.stat file 324 GB of which shmem 92 GB (the pinned tier), anon 1 GB |

## What D3 means

The decoded-weight cache is correct and does what it was built to do:
after R1 every dense resolution is a 4 ms hit and the proof is the same
bytes. It lost anyway, and the kind lines say where. First, the group
memo (`demo_maverick_block.memo_group`, on in both arms) had already cut
a dense resolution from the 0.45 s session 2 inferred to a measured
0.16 s, so the whole dense loader column was 631 s of the 2,493 s prove,
a quarter rather than the half of session 2, and the cache's ceiling was
that 631 s minus R1's 362 decodes. Second, the pinned host tier cost
more than it could save on this container: its cgroup held the GGUF's
page cache, and every pinned block forced reclaim, so R1's stores took
1.9 s each (a pinned store measures 100 ms in isolation on the same
box) and the reclaimed shard pages came back from the 1.2 GB/s disk in
every later sweep, 164 s more over the proof. The on arm's build, 303 s
against 41.8 s for the same tape, PRECEDES any pinning and is not
explained by this account; the container sees 288 CPUs against a
36-vCPU quota, so throttling of the numpy dequantize path is the other
candidate, and the session's logs cannot tell the two apart. The fix in
the tree is a GPU tier above the host one: the packed int32 set is 65 GB,
which fits beside the 48 GiB peak at S=100 (and beside the 80 GiB peak
of S=1000 on this card), a store is 0.2 ms, and HBM never touches the
cgroup; gated, not yet measured at S=100. With the loader at a quarter
of the prove, the enrolled block's encode, 1,194 s and context-
independent, is the largest term, the weight split's target.

## Caveats

- Every wall is an instrumented one (LIGERO_SWEEP_TIMING syncs around
  every loader resolution); the two arms carry the same instrumentation.
- The kind seconds are wall time around the loader call, inclusive of
  its CUDA sync, and can exceed the exclusive `fetch` column slightly.
- The on arm ran after the off arm on the same container; the build
  anomaly may be state the off arm left behind or a neighbour's load on
  a shared host. The session did not sample the host during the arms
  (the launcher's memwatch files were not copied home; the sampler that
  would have settled it, `tools/host_sampler.sh`, was written after).
- The 0.16 s per dense resolution is with the memo; an arm with
  LIGERO_GROUP_MEMO=0 on this host was not budgeted, so the memo's own
  saving against session 2's 0.45 s is not separated from the host's.
- The S=1000 prove with both caches did not run; the GPU tier's A/B is
  the next session's one arm (runbook, "D3 as measured ... and the
  rerun").
- The b200-runpod-s3 profile's `aggregate_ns_per_slot` is left null as
  the h200's was (needs a measured run on this box); `interconnect` is
  null on a single card.
