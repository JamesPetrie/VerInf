# Full-model sound proof run — result archive (paper-critical)

Recorded 2026-06-24. **Purpose:** preserve every metric from the first full-model
*sound* (four-round, commit-before-challenge) proof of LLM inference, so this can
be cited/written up without re-running it (the run is ~19 h of GPU time). All
numbers below are from the run log `/tmp/full_sound_t40.log` on `spark-c191`
unless marked otherwise.

## Headline

A complete **sound** Ligero proof was generated for a **1093-token forward pass
of the full 48-layer Llama-4-Maverick (128 experts/MoE layer)** at **T=40 query
columns**, on a single DGX Spark (GB10, 121 GiB unified). The four-round
commit-before-challenge consistency check **passed** (`root_p1 reproducible
across rounds: True`): the prover demonstrably used one consistent witness
between the round-1 commitment and the round-4 opening. An independent `verifier-rs`
`verify_proof` run then **`ACCEPT`ed** the proof at T=30 of the 40 opened columns
(ε_IRS ≈ 2^-12.5; see Verification).

## Configuration

| field | value |
|---|---|
| Model | Llama-4-Maverick, GGUF `UD-Q4_K_XL` (k-quant weights → field) |
| Layers | 48 (alternating dense / MoE; MoE layers E=128) |
| Vocab | 202048 |
| Sequence length T | 1093 = **442 prompt (hidden bound)** + **651 continuation (public)** |
| Claims | 11926 total |
| Protocol | **sound** — four-round, commit-before-challenge |
| Query columns T_QUERIES | 40 (soundness ε_IRS = (3/4)^40 ≈ 2^-16.6, LogUp-capped ~2^-28) |
| Hardware | DGX Spark `spark-c191`: NVIDIA GB10 (sm_121), 121 GiB unified, ARM64, CUDA 13.0 |

## Runtimes

| phase | wall-clock |
|---|---|
| Reveal / witness engine pass (first pass, yields the UI) | 3888.8 s (64.8 min) |
| Round 1 — commit phase 1 (incl. ~20 min compile) | 160.5 min |
| Round 2 — commit phase 2 | 101.6 min |
| Round 3 — q-poly fold (q_lin + p_0; the heavy round) | 552.6 min (~9.2 h) |
| Round 4 — merkle rebuild + column extract | 321.3 min (~5.4 h) |
| **Total prove (`prove returned`)** | **69390.0 s = 19.27 h** |

(Per-round figures are the in-round `[sweep]` elapsed at 100%; the ~20 min residual
vs. the 69390 s total is inter-round setup/compile. Round 3 ≫ others because it is
the q_lin **and** p_0 polynomial fold — the NTT-bound work.)

## Memory

- **Peak GPU: 77.56 GB** (`peakGPU` at `prove returned`).
- **Peak unified (CPU+GPU pool): 83.28 GB** (`[stream-sound] ... peak 83.28 GB`).
- Fit the 121 GiB Spark with ~38 GiB headroom; no OOM, streaming prover throughout.

## Unexplained information (UI) — the entropy claim

- **UI = 0.3941 bits/token**, over the 651 continuation positions → **256.6 bits total**.
- `Sz=728395` pinned as the PUBLIC bound; the verifier reads 0.3941 bits/token from
  the claim (the prover cannot inflate it post hoc).
- Interpretation: the proven forward pass is near-greedy (very low per-token
  unexplained information), consistent with the FP/GGUF model being effectively
  deterministic on this transcript.

## Soundness check

`[stream-sound] 4 rounds done; root_p1 reproducible across rounds: True` — the
round-1 Merkle root was reproduced identically when round 4 re-derived it, i.e.
the committed witness is consistent across the commit→challenge→open rounds. This
is the core integrity property the four-round protocol exists to enforce.

## Artifact (the dumped proof)

- Path on `spark-c191`: `/tmp/maverick_full_sound_t40.json`
- **Size: 92 GB** (dump took **851.7 s ≈ 14.2 min**).
- **md5: `07a1e14c1029af07b64f54d1cc3017d4`**
- Format: single JSON (claims + seeds + per-round replies + opened columns + Merkle
  paths) consumable by `verifier-rs` `verify_proof` (typed streaming serde parse).
- Verification status: **verified — `ACCEPT` at T=30 of 40 columns** (ε_IRS ≈ 2^-12.5;
  interactive → meaningful soundness). Also `ACCEPT` at T=4 (earlier smoke check).
  See the **Verification** section below.

## Verification (the verifier side)

Run on `spark-c191` with `verifier-rs` `verify_proof` on branch
`feat/verifier-per-family` — the per-family / windowed-`emit` verifier plus the
column-memory fixes below (*not* the row-streaming path originally planned). The
prover opened **40** columns; the verifier was run at two query counts.

**Result: `ACCEPT` at both — all six checks pass** (`merkle`, `irs_col`, `lin_sum`,
`lin_col`, `quad_zero`, `quad_col` all `[OK]`).

| field | T=30 (primary) | T=4 (earlier smoke check) |
|---|---|---|
| Verdict | **ACCEPT** | **ACCEPT** |
| Query columns checked | **30 of 40** (`LIGERO_MAX_COLS=30`) | 4 of 40 (`LIGERO_MAX_COLS=4`) |
| IRS soundness ε_IRS = (3/4)^T | **≈ 2^-12.5** | ≈ 2^-1.66 |
| Wall-clock | **14:02:22** (50,302,708 ms verify timer; 20 threads) | 3:02:09 (10,691,024 ms) |
| Peak RSS | **78.6 GB** (82,447,948 KB) | 78.6 GB (82,447,920 KB) |
| Date | 2026-06-26 | 2026-06-25 |

Constraints (both): 115,235,029 rows; 23,554,246 quadratic; 275,681 families.

**The peak RSS is identical at T=4 and T=30** (78.6 GB), which pins it as the **parse
of the 92 GB proof** (opened columns + serde over-allocation), *not* the
column-checking — the resident set is independent of how many columns are checked.
Both fit the 121 GiB Spark with headroom.

**Soundness reading.** The protocol is interactive (commit-before-challenge), so the
prover cannot grind: the per-challenge error bounds a *single* live attempt, not the
best of many. At **T=30, ε_IRS ≈ 2^-12.5 is a meaningful soundness statement** (a
cheating prover passes about once in 5,800), not a placeholder. T=4 (ε ≈ 2^-1.66 ≈
1-in-3) is genuinely just a "valid proof passes" smoke check. More margin is a
deployment knob: T=40 → 2^-16.6, T=80 → 2^-33 (past ~T=80 the LogUp term ~2^-28
binds and needs parallel repetition of its challenge). A higher-T verify on this CPU
box is proportionally longer (T=30 took 14 h); GPU is the practical path to higher T
in less wall-clock.

**What made it fit** (the eager verifier OOM'd at 119 GB; two bit-exact fixes → 78.6 GB):
1. Build the joint opened-columns **once** and free the raw per-commit subcolumns
   (were held ~2× and the join was rebuilt 3×).
2. **Move** the queried columns out of the parsed proof instead of cloning (frees the
   parse copy).
Measured parse breakdown: 37 GB opened columns + ~17 GB serde `Vec` over-allocation;
merkle paths ≈ 0; constraint families only **6.5 GB** total (`Embed` 3.3 GB is the
largest — *not* the dominant memory; the columns are).

**`lin_col` time by constraint kind** (from the **T=4** run — ~178 of its 182 min;
per-kind timing wasn't instrumented at T=30, where the `O(T)` kinds scale up and
`lin_col` ran ~13 h of the 14 h total):

| kind | time | rows | |
|---|---|---|---|
| Identity | 5616 s (60%) | 55.7M | activation/copy constraints — the bottleneck at full scale |
| FreivaldsB | 2030 s (22%) | 49.2M | matmul weights |
| FreivaldsA + C | 1601 s (17%) | 17.7M | matmul λ / combined |
| all other kinds | < 500 s | | |

`lin_col` is inherent `O(trace · T)` dense field arithmetic, parallelizing across all
20 cores (not a parallelism bug). The bottleneck **shifts with scale**: on the small
`m1` proof `FreivaldsB` dominated (88%), but at full scale `Identity` (the activation
copies) dominates. The dense per-family arithmetic is the natural GPU target.

## Provenance

- Machine: `spark-c191` (Tailscale); user `claude`; repo `~/infproof`.
- Driver: `pipeline/demo_maverick_full.py --sound --from-gguf <maverick UD-Q4_K_XL>
  --tokens <tokens_1k.json> --dump-proof /tmp/maverick_full_sound_t40.json
  --ui-abort-above 5.0`, `LIGERO_T_QUERIES=40`.
- Log: `/tmp/full_sound_t40.log`; pidfile `/tmp/full_sound_t40.pid`.
- Date completed: 2026-06-24.

## TODO to finalize this archive

- [x] Record final proof size + md5 once the dump completes. (92 GB, md5 `07a1e14c…`, 2026-06-24.)
- [x] Record the verify outcome (ACCEPT/REJECT), verify wall-clock, and verifier
      peak RSS once run. (**ACCEPT at T=30**, 14:02:22, 78.6 GB peak, ε_IRS ≈ 2^-12.5,
      2026-06-26; also ACCEPT at T=4, 3:02:09 — see Verification. T=40/T=80 for more
      soundness margin remain optional.)
- [ ] If kept long-term, copy the proof off `/tmp` to durable storage.
