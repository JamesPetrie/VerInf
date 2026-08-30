# Blackwell session 1 — archive (2026-08-19)

First run of the profiler tooling on the target hardware class: one rented
NVIDIA B200 (RunPod secure cloud, US-NC-2, 183 GB HBM, host CUDA 13.0,
`runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`), ~100 minutes, ~$11.
Repository state: profiler as of PR #18 (base 530283a) plus the untracked
session tooling later folded into the sync PR.

## What ran

1. `profiler/calibrate.py --name b200-runpod` — all five CUDA benches
   compiled at sm_100 (including `bench_blake3_reg`'s first compile), torch
   d2d/H2D, disk and dump probes. Output: `profiler/machines/b200-runpod.json`
   with raw bench logs in `profiler/machines/calibrate-raw-b200-runpod/`.
2. `profiler/crosscheck.py llama7b --seq 100 --layout` — extraction
   cross-check, weight-free.
3. `profiler/crosscheck.py maverick --prompt-n 2 --cont-n 2 --layout` and
   `--seq 1000` against the UD-Q4_K_XL GGUF.
4. `predict` / `partition` on the extracted S=1000 manifest with the new
   profile.

## Results

| item | result |
|---|---|
| memory bandwidth (torch d2d, read+write) | 6,522.7 GB/s — 29.25x gb10-spark |
| BLAKE3 register-resident compress | 14.78 Gc/s (column bench: 11.36 Gc/s, 727 GB/s) |
| field mul / NTT (single transform) | 559.9 Gmul/s / 0.271 ns/elem — both look launch-bound on this part; see caveats |
| llama7b cross-check | zero flags; layout probe identical across 9 claim types, m_total gap +3 |
| maverick cross-check (T=4) | zero flags; matmul 9,674 exact, persistent slots 402,724,618,240 exact, every UI cap hit exactly, layout identical across 17 types, +3 |
| maverick S=1000 extraction | 109,273,513 rows vs the archived production run's measured 109.27 M |
| first Blackwell floor, full protocol, S=1000 | 376 s (A*W 276 + C*Q 89 + B*cids 11) — ~28x gb10, tracking the bandwidth ratio |

Note: the S=1000 extraction inherited the demo's default `T_QUERIES=80`
(the archived run used 40); T-linear outputs (proof size, opened columns)
are the archive's x2.

## Caveats and later corrections

- The single-transform NTT and field-mul ratios (1.55x, 1.79x) are
  under-measured: a 64K transform fits in the B200's L2, so the bench times
  kernel launches. `bench_ntt_batched` (session 2) is the prover-path
  number.
- `disk_read_GBps` (0.3) and `proof_dump_MBps` (33.8, proxy) reflect the
  pod's MooseFS network volume, not local NVMe — session 2 uses local disk.
- The session's partition scorecard reported ~950 GB/sweep of cross-shard
  traffic and a BINDING comms verdict. That was retracted on 2026-08-20 as
  ~99.7% model artifact (settlement z vectors shipped whole); with the
  corrected model traffic is 2.4-2.8 GB/sweep and comms are negligible at
  every swept bandwidth. See the sync PR.

## Files

- `crosscheck-out/*.json.gz` — the three extracted manifests (gzipped;
  `Manifest.load` reads `.gz` directly), plus both `LIGERO_LAYOUT_BREAKDOWN`
  probe outputs.
- `xchk-llama.log`, `xchk-mav-small.log`, `calibrate.log` — session logs.
- `b200-pins.txt` — `pip freeze` on the pod.
