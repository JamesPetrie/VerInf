# H200 session 2 — archive (2026-09-15/16)

The second rented session: the projected protocol's hardware crosscheck,
the first witness composition of a real routed tape, and the routed-output
cache A/B, on one NVIDIA H200 (RunPod secure cloud, US-NC-1, 141 GB HBM,
2 TB host RAM, 400 GB container disk, host CUDA 12.8,
`runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`), about 5.4 hours at
$4.59/h. No B200 was in stock; the Blackwell calibration is deferred to a
short later session. Repository state: `e51eb21` on `weight-split-model`,
clean; the session followed `profiler/RUNBOOK-blackwell.md` "Session 2"
phase by phase, every phase through its gate.

## What ran

1. Bootstrap on the image's torch 2.8.0+cu128 (capability 9.0), the
   verifier built, the CUDA primitives JIT-compiled and their 25 tests
   passed. Record: `session2-env.txt`; packages: `session2-pins.txt`.
2. `profiler/calibrate.py --name h200-runpod-s2 --skip-io`, overlapped
   with the pull, then `--io-only` on idle storage. Output:
   `profiler/machines/h200-runpod-s2.json`, raw bench logs in
   `profiler/machines/calibrate-raw-h200-runpod-s2/`.
3. `tools/gguf_pull.sh`: the probe at 237 MB/s over eight streams, the
   five UD-Q4_K_XL shards at the pinned revision `41032e5`, 232.13 GB,
   every size and sha256 matching `profiler/data/maverick-ud-q4_k_xl-shards.json`
   (`logs/gguf-pull.log`).
4. `tools/session2_gates.sh`: the K-quant kernel against the real shards
   (4/4), `test_shard_streaming` 7/7 with the three routed-cache gates,
   `test_routed_projected` 6/6, the toy A/B's counts, and a two-layer
   real-GGUF proof through `profiler/instrumented_prove.py --verify`,
   accepted by the Rust verifier in 391 s (`logs/gates.log`,
   `logs/gate-*.log`).
5. `profiler/crosscheck.py maverick` at 2+2 tokens with the selftest, and
   at 500+500, the timed prove's own split, with the layout probe and the
   witness composition (`crosscheck-out/`).
6. The A/B at S=100 (50+50), the routed-output cache off then on, same
   tape and settings, each arm with its own throwaway enrollment and
   reveal pass, the per-sweep table on (`logs/mavp-s100-off.log`,
   `logs/mavp-s100-on.log`, `ab-s100-compare.txt`).
7. The S=1000 (500+500) prove, cache off, both timing modes, the proof
   dumped (`logs/mavp-s1000-off.log`). The S=1000 on arm was not run.

## Results

| item | result |
|---|---|
| memory bandwidth (torch d2d, read+write) | 4,259.5 GB/s — 19.10x gb10-spark |
| BLAKE3 register-resident compress | 10.41 Gc/s (column bench 7.92 Gc/s, 506.7 GB/s) |
| field mul / NTT single / NTT batched (largest batch) | 607.4 Gmul/s / 0.213 ns/elem / 0.0622 ns/elem — batched is 2.83x the bandwidth-scaled expectation: encode is NOT bandwidth-scaled on this part |
| derived constants | A 0.471 ns/slot, B 0.0961 ns/cid, C 0.785 ns/product (b200-runpod: 0.308 / 0.0677 / 0.513) |
| storage (container overlay) | sequential read 1.52 GB/s, compact dump 771 MB/s (probe), 822 MB/s (the real 35.46 GB proof), proxy 246 MB/s, H2D 55.3 GB/s |
| llama7b crosscheck | clean |
| maverick crosscheck, T=4 and S=1000 | every modeled claim type and the weight block exact; three FLAGs on the shared-type totals (W +3.1% at T=4, +2.3% at S=1000; cids and Q similar), all from the routing-bundle and unexplained-information records the synth `maverick-projected` builder does not count (routing[+aux] 1.65 G slots, MaxClaim 1.21 G, ptlookup 0.61 G, Concat 0.40 G at S=1000) |
| witness composition, S=1000, real routed tape | produced phase-1 outputs 67.44 G elements (539.5 GB); committed 2.4 GB; enrolled weights 402.72 G slots (3,221.8 GB as field elements); phase-2 aux 31.52 G (252.2 GB); phase-3 27,648; routed Y 516.10 M elements (4.13 GB). The regeneration note's estimates were 517 and 243 GB |
| A/B, S=100, prove wall | off 3,833.3 s, on 3,788.9 s (−1.2%); peak 47.1 / 47.5 GiB; enrollment 951 / 923 s, reveal 185 / 184 s, same public bound; leaf checks true |
| A/B, S=100, what the cache removed | 2,340 routed shard loads and 785 GB of decoded bytes in each of R3, fold, open (7,020 loads, 2.36 TB per proof); witness compute in those sweeps 11.1 → 6.0 s; loader time in those sweeps −4.7, −7.6, −6.0 s |
| A/B counters | routed_wr 72 in R1, routed_rd 72 in R2/R3/fold/open on the on arm, 0 on the off arm; proj 0,72,0,0,0 both; witness cache 120 writes and 120 reads per sweep both |
| S=100 off arm, where the prove goes | loader calls 1,967 s (dense weights: 4,344 resolutions at 0.45 s), encode 1,191 s, fold_qlin 365 s, witness 77 s, aux 9 s; fold sweep 1,469 s, open 1,289 s |
| S=1000 off arm | prove 5,333.5 s, peak 79.7 GiB; enrollment 926 s, reveal 246 s; sweeps R1 401 / R2 536 / R3 438 / fold 2,173 / open 1,550 s; loader 2,062 s, encode 1,599 s, fold_qlin 463 s, witness 384 s, quad 365 s; proof 35.46 GB dumped in 43.1 s |
| witness cache at S=1000 | 16 claims stored of 120 (the quarter-of-free-HBM budget fills after the first layers) |

## What the A/B means

The routed-output cache removes exactly the reads it targets, and they
were cheap: a routed shard resolution slices a 23 MB packed Q4 block from
the page-cached GGUF and dequantizes it on the GPU, about 2 ms
(`prover/loader.py`, `maverick_lazy_expert`). The loader time that is
half the prove is the dense weights: `demo_maverick_block.load_attention`
decodes a layer's whole six-tensor group on every call for any one of
them, with the query projection through the CPU numpy path (its scale
carries a division the GPU kernel does not take), and the sweep resolves
each dense weight twice per aux sweep (the compute fetch, then the aux's
lazy dict) and once more in each encode pass — 4,344 resolutions per
proof, which at 0.45 s (R3's 724 dense loads in 335.6 s on the on arm)
account for the 1,967 s loader column. The witness compute the cache and
its proposed extensions target is 77 s at S=100 and 384 s at S=1000.
The next levers in order of size on this box: a decode-once cache of the
48 attention groups (about 41 GB) or a GPU path for the query
projection, worth about half the prove; the two enrolled-block passes
(the weight split's target), about half the rest; the witness cache's
budget at S=1000. The argument is in the vault pass of 2026-09-16.

## Caveats

- Every wall is an instrumented one: LIGERO_SWEEP_TIMING adds a cuda
  sync around every loader resolution. The two arms carry the same
  instrumentation, so the delta is fair; the absolute walls overstate an
  uninstrumented prove by an unmeasured amount.
- The per-sweep table counts loads and decoded bytes but does not time
  them by loader type; the 0.45 s per dense resolution is inferred from
  the two arms' R3 rows, not measured per loader.
- The routed shards were page-cached on a 2 TB host. On a smaller host
  the 2 ms would grow, and the cache's value with it.
- On a B200 the compute side is faster and the loader side is not, so
  the loader share is larger there; the ranking of the levers should
  survive, the numbers will not.
- The H200 profile's constants are not comparable to the B200's; the
  254 s floor and the weight-split ratios remain B200 numbers.
- The synth `maverick-projected` builder omits the routing-bundle and
  unexplained-information records (2.3% of W at S=1000); a profiler
  follow-up.
