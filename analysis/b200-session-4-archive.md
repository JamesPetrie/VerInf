# B200 session 4 — archive (2026-09-19)

The fourth rented session: the merged tree's GPU suites, the bridge's
first measurement on the same card and tape as our baseline, and the
decoded-weight cache's GPU tier, on one NVIDIA B200 (RunPod secure cloud,
US-CA-2, 180 GB HBM, a 2.2 TB host with a 283 GB cgroup and a 23.8-CPU
quota against 224 visible cores, 400 GB container disk, host CUDA 13.2
under the cu128 image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`),
19:54 to 22:24 UTC, about 2.5 hours at $6.79/h. Repository state:
`41ddcf6` on `weight-split-model` as a tarball, plus one verifier fix
shipped during the session (below). Every phase through its gate; the
host sampler (`tools/host_sampler.sh`) ran from bootstrap to the end
(`logs/sampler.log`), and the launcher's memwatch files came home.

## What ran

1. Bootstrap (`logs/phase0.log`): verifier built, primitives 25/25, then
   before the pull the seven GPU-free suites on the merged tree —
   instrument bookkeeping 3/3, weight cache 5/5, shard streaming 7/7, the
   bridge's standalone suite 15/15, its streaming-flag suite 9/9, its
   tape-link test 1/1, and our six Rust negatives at 3/6: the verifier
   rejected the three cases, but by a panic in the compiler (a bridged
   claim with no authenticated pin) before printing its verdict line. Fixed
   on the spot — a failed bridge check now prints the policy table and
   REJECT — rebuilt on the pod, 6/6 (`logs/early-negatives-2.log`).
2. Calibration `b200-runpod-s4` (A 0.313 ns/slot, B 0.0533, C 0.522;
   batched NTT 3.79x the bandwidth-scaled expectation), overlapped with
   the pull (probe 172 MB/s, 217 GB in eight minutes, every shard hashing
   to its pin), then storage on idle disk: read 3.49 GB/s, compact dump
   1,271 MB/s, H2D 57.7 GB/s — about three times session 3's box.
3. The gate script (`logs/gates.log`): K-quant 4/4, nine suites, the toy
   A/B, and three two-layer real-GGUF proofs through the driver with
   `--verify`: committed weights (prove 108.4 s, Rust 272 s), the GPU-tier
   cache (94.4 s; 10.6 GB packed on the card, nothing pinned), and the
   bridge (67.5 s, Rust 64.7 s; the enrolled block 324 k rows against
   2.29 M). All accepted.
4. Three S=100 arms (50+50 tokens, the routed cache on in each, own
   throwaway enrollment and reveal pass, per-sweep table with the kind
   lines): the weight cache's GPU tier (`logs/mavp-s100-wc-on2.log`), the
   bridge (`logs/mavp-s100-bridge.log`), and the same-host baseline with
   neither (`logs/mavp-s100-wc-off2.log`). Comparisons in
   `ab-s100-weight-cache.txt` and `ab-s100-bridge.txt`; figures from
   `figures.py`.

## Results

| item | result |
|---|---|
| baseline, same host | prove 1,890.8 s; dense loader 117 s over 3,982 decodes (0.029 s each); shard loader 53 s; encode 1,194 s, fold 272 s; enrollment 842 s; peak 48.4 GiB |
| GPU-tier weight cache | prove 1,787.0 s (−5.5%); 362 decodes then 3,620 hits; 64.7 GB packed on the GPU, 0 pinned, 0 refused; dense loader 23 s; peak 108.7 GiB; enrollment 766 s |
| bridge | prove 988.0 s (−47.7%); the five sweeps 395 s; encode 136 s, fold 17 s; shard loads 11,556 (R1 and R2 only); dense loader 94 s; enrollment 48 s + 273 s streaming coefficient-RS of all expert shards; 592 s outside the sweeps: the bridge's own per-proof pass |
| session-3 baseline, other host | prove 2,492.8 s; the same encode and fold to the second; dense loader 631 s (0.16 s per decode); the loader columns are host-bound and not comparable |
| build | 18.6 to 19.1 s in every arm; session 3's 303 s did not recur |
| leaf checks | true in every arm; every gate proof accepted by the Rust verifier |

## What it means

The weight bridge takes the enrolled block out of the prove: on the same
card and tape the bridged proof is 52 percent of the baseline at a
hundred tokens, the two per-proof passes over enrolled rows fall from
1,466 s to 153 s (the dense weights only), the shards are read in R1 and
R2 alone, and the verifier at two layers is four times faster because it
no longer folds the weight rows. What the bridge adds is its own pass,
592 s here and 580 s on the collaborator's L40S bench, so it is not
GPU-bound as written and it is now the largest single term of a bridged
prove; that pass is where instrumentation goes next, and where the
two-level enrollment's cost would land. The five semantic sweeps, 395 s,
are the second term and the witness-cache line's target.

The GPU tier of the decoded-weight cache is correct and worth 5.5 percent
on this host, where a dense decode costs 0.029 s; session 3's 28-percent
reading was two thirds the host. It keeps the pinned-memory failure of
session 3 from recurring and costs 65 GB of HBM at a hundred tokens,
which is affordable on this card and not on a 141 GB one at a thousand.

## Caveats

- Instrumented walls, the same instrumentation in every arm.
- The bridge numbers are for its interim form: the folds, the projected
  masks and the aggregate travel in the proof in the clear. The committed
  form adds tiny rows and is the zero-knowledge claim's prerequisite.
- The bridge section of the proof is plain JSON (about 8 GB at this
  scale); no proof was dumped this session.
- The enrollment is rebuilt every process (273 s); persistence and the
  rotation linking proof are the agreed follow-ups.
- The GPU tier's peak (108.7 GiB) leaves 70 GB on this card at S=100; at
  S=1000 the witness cache's budget shrinks accordingly and was not
  measured.
- One session-3 comparison remains cross-host by construction; the
  same-host baseline is the one to cite.
