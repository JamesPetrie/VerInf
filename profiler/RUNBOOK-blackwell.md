# First rented-Blackwell session runbook

One short session on a rented Blackwell box closes profiler roadmap items
**1** (extraction cross-check — the gate for trusting extracted manifests)
and **3** (calibration — a measured machine profile for the actual target
hardware instead of gb10 ratio extrapolation). Optional stretch: first
instrumented timing on Blackwell. Written 2026-08 for the first rental;
update from experience.

## What you come home with

- [ ] `profiler/machines/<sku>.json` — measured Blackwell profile
      (+ `calibrate-raw-<sku>/` bench logs for provenance)
- [ ] `crosscheck-out/` — extracted manifests (llama7b, maverick small-T,
      maverick S=1000) + layout probe outputs + the diff verdicts
- [ ] first `predict` reports priced on real Blackwell constants
- [ ] (stretch) `LIGERO_PHASE_TIMING` output from an instrumented prove

## Before renting

- **SKU**: ask James which chip the eventual cluster targets (B200 vs
  GB200) and rent that — the profile should describe the machine the
  predictions are for. Unknown → B200. One GPU is enough for everything
  here. If the choice is genuinely open, the calibration suite makes SKU
  comparison cheap: ~1 hour on each candidate → one profile per SKU →
  `predict` gives $/proof on each.
- **Image**: CUDA toolkit new enough for the SKU (sm_100 needs CUDA
  12.8+; the repo's dev box runs 13.0), recent driver, Python 3.10+.
  `nvcc` must be present for the CUDA microbenches (`--skip-cuda` runs the
  rest without it, but field-mul/NTT/BLAKE3 are the point).
- **Disk**: ≥ 60 GB without Maverick; ≥ 350 GB with the Maverick GGUF.
  Know which mount is real disk vs tmpfs — probe files on tmpfs measure
  RAM (calibrate warns about this; it once took out a dev box's /tmp).
- **Getting the repo there**: clone once the PR branches are pushed, or
  `rsync` the working tree. The GGUF (UD-Q4_K_XL, ~250 GB) is the long
  pole — start that download first, in the background.

## Session order

Steps 1–2 need no downloads — run them while the GGUF transfers.

### 0. Bootstrap (~10 min)

```sh
python3 -m venv venv && . venv/bin/activate
pip install -r requirements.txt        # torch wheel must match the CUDA toolkit
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
nvcc --version
```

The verifier build (cargo) is NOT needed — nothing here verifies.

### 1. Calibrate (~15 min)

```sh
python3 profiler/calibrate.py --name <sku> --tmpdir /path/on/real/disk
```

Compiles all four spark-bench microbenches (field-mul, NTT, BLAKE3
columns, matmul) plus the profiler's own `bench_blake3_reg` with the
**detected** arch (the old Makefile hardcodes sm_121 = GB10; wrong-arch
builds run and silently print wrong numbers), runs torch copy/H2D benches,
disk + dump probes (unique temp files, page-cache eviction, cleaned up on
failure), and writes the profile with per-field provenance. Each bench is
isolated — one failure skips one number, not the run.

Prove constants: A/C are ratio-derived from gb10-spark by measured
bandwidth; B comes from `bench_blake3_reg` (ALU-bound compress) — DIRECT
at 1 compress/cid for now, upgraded to a cross-machine ratio once someone
runs that bench on the Spark (gb10's column-hash 2.0 Gc/s is
bandwidth-limited and can't carry B; direct-B leans floor-ward
structurally — raw ALU compress, no launch/orchestration margin, the size
of which stays unquantified until the reg bench runs on a box with a
calibrated B). All labeled in provenance; validation
mode (roadmap 2) replaces them. **If the reg bench fails to build, B is
null and step 4's floor prints UNAVAILABLE** — predict/partition need all
of A/B/C, so fix the bench (or fill B by hand from the provenance note)
before moving on. `aggregate` stays null on purpose.

Sanity-check the printed ratios-vs-gb10 table on the spot: a B200 should
show roughly memory-bandwidth ~8x, compute well above that. A ratio near
1x means a bench mis-set (arch? clocks? MIG slice?) — fix before trusting
anything downstream.

### 2. Cross-check, weight-free half (~10 min)

```sh
python3 profiler/crosscheck.py llama7b --seq 100 --layout
```

Runs `extract.py --selftest` (first time extract.py touches real hardware),
builds the random-weights Llama-7B lazy tape, extracts, diffs against synth
(claim counts, W/cids/Q, persistent slots exact), then reruns the demo as a
subprocess to the prover's `LIGERO_LAYOUT_BREAKDOWN` probe and diffs the
row layout table row for row. Exit 0 = clean; any FLAG line = eyeball it —
divergence means a stale formula, and the tape is the ground truth.

Also eyeball the extracted labels/layers (README caveat: name-parsing is
best-effort): `python3 profiler/cli.py partition crosscheck-out/llama7b-*.json
--shards 4` should score `layers` sensibly, not refuse.

### 3. Cross-check, Maverick (~20 min once the GGUF is down)

```sh
python3 profiler/crosscheck.py maverick --from-gguf <path> \
    --prompt-n 2 --cont-n 2 --layout
python3 profiler/crosscheck.py maverick --from-gguf <path> --seq 1000 \
    --skip-selftest        # big-S extraction; no --layout (engine pass costs hours at S=1000)
```

Small-T first: full 48-layer structure, MoE fan, UI chain, settlements —
everything the differ checks — at minutes of runtime. The layout probe runs
the demo's reveal engine pass before the layout print, which is why big-S
skips it. Expected extracted-vs-synth deltas are labeled in the output
(UI chain ~107 fixed + ~2/position, settlements); unexpected ones FLAG.

The S=1000 manifest is the keeper: the first extracted (not synthetic)
Maverick manifest, for the partition scorecard on real labels.

### 4. First Blackwell predictions (~5 min)

```sh
python3 profiler/cli.py predict crosscheck-out/maverick-s1000-extracted.json \
    --machine <sku>
python3 profiler/cli.py partition crosscheck-out/maverick-s1000-extracted.json \
    --shards 4 --weight-bytes-per-param 0.7
```

First floor priced on measured Blackwell constants (A/C bandwidth-ratioed,
B direct — the aggregate row stays unavailable until a measured run
calibrates it, which is honest: it doesn't transfer across hardware), and
the phase-1 scorecard on an extracted manifest. Save the reports.

### 5. Stretch, budget permitting: instrumented prove

```sh
LIGERO_PHASE_TIMING=1 LIGERO_T_QUERIES=40 tools/spark_run.sh mav-s100 \
    python3 demo/demo_maverick_full.py --from-gguf <path> --prompt-n 50 --cont-n 50
```

An S=100 Maverick prove (~6 h on GB10; Blackwell should be several times
faster — the measurement is the point) with the per-phase breakdown printed
at the end. This confirms the R2 weight-restream prediction (~33 min,
S-independent — the small-context archive's open question) and starts the
wall-residual attribution. The **S=1000** phase-timing run is the
expensive headline ask (per ROADMAP.local: the residual's S-dependent term)
— decide on it after seeing the S=100 shares, don't default into it.

Use `tools/spark_run.sh` for anything long — detached, logged, survives
SSH drops. Kill via the pidfile, never `pkill -f`.

## Copy home before releasing the box

```
profiler/machines/<sku>.json  +  profiler/machines/calibrate-raw-<sku>/
crosscheck-out/               (manifests, layout probes)
predict/partition reports     (stdout captures)
~/mav-s100.log                (if the stretch ran)
pip freeze > blackwell-pins.txt   (requirements.txt asks for this)
```

## Gotchas

- `LIGERO_T_QUERIES` is read at **import time** by the demo configs — set
  it in the environment before any demo import (crosscheck's `--t-queries`
  does this; for manual runs, prefix the command).
- The GB10 unified-memory notes scattered in the demos (lazy-weights
  driver bug, pytorch #174358) do not apply on discrete-HBM parts; nothing
  needs changing, just don't be alarmed by the comments.
- Tape construction materializes LogUp tables eagerly on CUDA (~1 GB for a
  width-26 table pair) — that's why even "no-witness" extraction needs a
  GPU at all.
- If `nvcc` compiles but a bench prints absurd numbers, check the arch line
  in `device:` output matches the card before anything else.

---

# Session 2 — hardware crosscheck of the projected protocol + bench extensions

Goals: close the last validation gap of the sync milestone (extraction
of the new claim types on real hardware), fill the five new calibration
fields, and run the first instrumented PROJECTED prove on Blackwell.
Prepped 2026-08-23; the extraction contract is already regression-locked
against fake-core stubs (test_projected_extraction), so phase C is
expected-clean, not exploratory.

Rent: one B200 as before, but request ~400 GB CONTAINER disk and skip
the network volume — session 1 measured the network FS starving the
weight lane (0.3 GB/s); everything lives on local NVMe this time.
Budget: phases A-D ~2.5-4.5 h ≈ $17-31.

A (~45 min, no downloads):
   python3 profiler/calibrate.py --name b200-runpod-s2 --tmpdir /workspace
   # now also compiles bench_ntt_batched / bench_hbm_random /
   # bench_launch_latency (first-ever compiles — graceful null on
   # failure) and runs the u64le/base64 compact-dump bench.
   # SANITY: ntt_batched_ns_per_elem is THE number — if it lands near
   # the bandwidth-scaled ~0.014-0.05 ns/elem, the A-constant story
   # holds; near the single-transform 0.27, encode is launch-bound even
   # batched and the floor needs a rethink.
   python3 profiler/crosscheck.py llama7b --seq 100 --layout \
       -o /workspace/crosscheck-out   # validates the ModelConfig mirror

B (~10 min): GGUF to LOCAL disk (same hf snapshot command, local_dir
   under the container disk).

C (~30-60 min): the projected crosscheck —
   python3 profiler/crosscheck.py maverick --from-gguf <local> \
       --prompt-n 2 --cont-n 2 --layout -o /workspace/crosscheck-out
   python3 profiler/crosscheck.py maverick --from-gguf <local> \
       --seq 1000 --skip-selftest -o /workspace/crosscheck-out
   # The tape now contains RoutedProjectedMatmulClaim / RescaleClaim;
   # the diff runs against synth maverick-projected. Expect the UI caps
   # and the routing-bundle expansion to hold as at session 1; any FLAG
   # on the routed types is the finding this session exists to catch.

D (~30-90 min): instrumented projected prove, the strategy-grade run —
   LIGERO_PHASE_TIMING=1 tools/spark_run.sh mavp-s1000 \
       python3 demo/demo_maverick_full.py --from-gguf <local> \
       --prompt-n 500 --cont-n 500 --dump-proof /workspace/mavp.bin
   # Deliverables: measured wall vs our 254 s floor / 30-45 min pipeline
   # projection; per-stage timings for the admission-model unification
   # with Ed; enrollment exercised on Blackwell; a real compact-dump
   # rate (cross-check io.proof_dump_compact_MBps from phase A).

Copy home: profile + raw logs, crosscheck-out/, phase-timing log, the
proof file size (not the proof), pip freeze.
