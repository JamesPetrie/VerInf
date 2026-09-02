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

Compiles the four spark-bench microbenches (field-mul, NTT, BLAKE3
columns, matmul) plus the profiler's own `bench_blake3_reg`,
`bench_ntt_batched`, `bench_hbm_random` and `bench_launch_latency` with the
**detected** arch (the old Makefile hardcodes sm_121 = GB10; wrong-arch
builds run and silently print wrong numbers), runs torch copy/H2D benches,
disk + both dump probes (unique temp files, page-cache eviction, cleaned up
on failure), and writes the profile with per-field provenance. Each bench
is isolated — one failure skips one number, not the run — and if anything
later in the run dies (MemoryError in the dump proxy, Ctrl-C, a disk probe)
the profile is still written, marked PARTIAL, with everything measured so
far. The chained-ALU benches (field-mul, blake3_reg) get a grid sized to
the SM count (full occupancy); their built-in 256x256 default is 13.8
warps/SM on a B200, which is what made session 1's per-SM rates look
"launch-bound" — they were under-occupied. `gpu.mem_GB` is GiB.

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

Sanity-check the printed ratios-vs-gb10 table on the spot. Session 1
measured memory bandwidth 29.25x gb10 (the torch d2d copy, 2x-counted on
both boxes) and field-mul 1.79x, blake3-reg-vs-column 1.55x — so "~8x
bandwidth, compute well above" was the wrong expectation; a ratio near
1x on bandwidth, or field-mul BELOW 1x, means a bench mis-set (arch?
clocks? MIG slice? grid?) — fix before trusting anything downstream. The
run also prints the batched-NTT check: the batched ns/elem against the
value gb10's NTT would have if encode scaled with bandwidth (that is the
assumption behind A); "NOT bandwidth-scaled" there means the floor is
optimistic and should be read as such.

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
python3 profiler/crosscheck.py maverick --from-gguf <path> --t-queries 54 \
    --prompt-n 2 --cont-n 2 --layout
python3 profiler/crosscheck.py maverick --from-gguf <path> --t-queries 54 \
    --seq 1000 --skip-selftest --layout      # big-S extraction + layout
```

Small-T first: full 48-layer structure, MoE fan, UI chain, settlements —
everything the differ checks — at minutes of runtime. The synth builder is
chosen from the tape (routed-projected claims present → `maverick-
projected`, else the legacy `maverick`), and the Maverick layout probe is
in-process (`core.layout_breakdown`: no engine pass, no weight resolution),
so `--layout` is cheap at S=1000 too. Expected extracted-vs-synth deltas
are labeled in the output (UI chain ~107 fixed + ~2/position,
settlements); unexpected ones FLAG. `--t-queries 54` is the target
geometry; the manifest's T_QUERIES feeds the proof-size and opening lines.

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
LIGERO_PHASE_TIMING=1 tools/spark_run.sh mav-s100 \
    python3 profiler/instrumented_prove.py --from-gguf <path> \
    --t-queries 54 --prompt-n 50 --cont-n 50
```

**Not `demo/demo_maverick_full.py`**: on current main its proof path is
fail-closed (enrollment + trusted root + public Sz + a 714-run admission
report with measured five-sweep semantics + target T_QUERIES, per
demo/4h-production-runbook.md) and refuses to start otherwise. The
research driver runs the same tape through the same prover with a
throwaway in-process enrollment and a reveal pass for Sz, and prints the
per-phase breakdown, enrollment time and proof size — timings, not a
production proof. An S=100 Maverick prove is ~6 h on GB10; Blackwell
should be several times faster — the measurement is the point. This
confirms the R2 weight-restream prediction (~33 min, S-independent — the
small-context archive's open question) and starts the wall-residual
attribution. The **S=1000** phase-timing run is the expensive headline
ask (per ROADMAP.local: the residual's S-dependent term) — decide on it
after seeing the S=100 shares, don't default into it.

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
  it in the environment before any demo import (crosscheck's and
  instrumented_prove's `--t-queries` do this; for manual runs, prefix the
  command). The demo's own main() now refuses any geometry but the target
  (T_QUERIES=54, ELL 8192, K_DEG 16384, N_LIG 65536) without
  `--allow-dev-config`; the demo default is 80, so an unprefixed demo run
  dies at the admission check before building anything.
- The profile's `io.disk_read_GBps` says which mount it measured. Session
  1's 0.3 GB/s is the RunPod NETWORK VOLUME, not the B200 class; every
  streaming model on that profile is I/O-bound by it (weightsplit prints
  an N=1 wall 4x the floor). Re-measure on container disk and say so in
  the provenance.
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
   # Compiles the four session-2 benches too (first-ever compiles of
   # ntt_batched / hbm_random / launch_latency on sm_100 — graceful null
   # on failure), runs the u64le/base64 compact-dump bench, sizes the
   # ALU-bench grids to the SM count, and writes a PARTIAL profile if
   # any later probe dies. --tmpdir must be the CONTAINER disk: the
   # profile's disk/dump numbers describe whatever mount this is, and
   # session 1's network-volume numbers (0.3 GB/s) are what made every
   # streaming model I/O-bound.
   # SANITY: the run prints "ntt BATCHED ... vs expected if encode scaled
   # with bandwidth" — that IS the A-constant check (A is bandwidth-
   # ratioed from gb10). Holds → the floor story stands; "NOT bandwidth-
   # scaled" → A is optimistic, read the floor with that caveat (the
   # A100 data in routed-projected-status.md already hint this way:
   # encode 5.12 ns/slot on A100 vs 4.28 on A6000 despite 2x bandwidth).
   python3 profiler/crosscheck.py llama7b --seq 100 --layout \
       -o /workspace/crosscheck-out   # validates the ModelConfig mirror
   # (llama7b's layout probe is still the demo subprocess — that demo has
   # no admission gate — and it is what validates build_llama7b's hand
   # mirror; note its weight-free build commits placeholders as
   # persistent, so the persistent-slot line checks the mirror against
   # synth, not the demo's own --lazy-weights persistence choice.)

B (~10 min): GGUF to LOCAL disk (same hf snapshot command, local_dir
   under the container disk).

C (~30-60 min): the projected crosscheck —
   python3 profiler/crosscheck.py maverick --from-gguf <local> --t-queries 54 \
       --prompt-n 2 --cont-n 2 --layout -o /workspace/crosscheck-out
   python3 profiler/crosscheck.py maverick --from-gguf <local> --t-queries 54 \
       --seq 1000 --skip-selftest --layout -o /workspace/crosscheck-out
   # The tape contains RoutedProjectedMatmulClaim / RescaleClaim; the
   # script detects that and diffs against synth maverick-projected
   # (it prints "protocol on the tape: maverick-projected" — if it says
   # "maverick", the GGUF build fell back to the legacy fan; stop and
   # look). The layout probe is in-process (core.layout_breakdown), so
   # it runs at S=1000 as well — the first extracted routed manifest
   # WITH a layout check; the session-1 S=1000 manifest is the legacy
   # protocol and never had one. Expect the UI caps and the routing-
   # bundle expansion to hold as at session 1; any FLAG on the routed
   # types is the finding this session exists to catch. Save the
   # S=1000 manifest gzipped (Manifest.save gzips on a .gz name now):
   #   python3 -c "import sys; sys.path.insert(0,'profiler'); from manifest import Manifest; \
   #     Manifest.load('/workspace/crosscheck-out/maverick-s1000-extracted.json').save('/workspace/crosscheck-out/maverick-s1000-extracted.json.gz')"

D (~30-90 min): instrumented projected prove, the strategy-grade run —
   LIGERO_PHASE_TIMING=1 tools/spark_run.sh mavp-s1000 \
       python3 profiler/instrumented_prove.py --from-gguf <local> \
       --t-queries 54 --prompt-n 500 --cont-n 500 --dump-proof /workspace/mavp.bin
   # NOT demo_maverick_full.py: its proof path is fail-closed on current
   # main (enrollment + trusted root + public Sz + 714-run admission
   # report with measured five-sweep semantics; the admission bench
   # emits those semantic stages as null by design, so no report can
   # pass on a rented box). instrumented_prove.py is the research
   # harness: same tape, same prover, throwaway in-process enrollment,
   # reveal pass for Sz, LIGERO_PHASE_TIMING on — a timing run, never a
   # production proof; it says so in its banner. It has NOT run on a GPU
   # before this session (the box it was written on has no CUDA): if it
   # dies, the traceback is the deliverable and the fallback is the
   # demo's --enroll-weights run (enrollment timing only, which the
   # policy path does allow) plus admission_bench.py --runs 30 for the
   # kernel stages at production geometry.
   # Deliverables: measured wall vs our 254 s kernel floor (the floor
   # EXCLUDES the five semantic sweeps — the 4h model's A100 run spent
   # 2024 s there; expect the wall to be dominated by them, not by the
   # floor) / 30-45 min pipeline projection; per-stage timings for the
   # admission-model unification with Ed; enrollment time on Blackwell;
   # a real compact-dump rate (cross-check io.proof_dump_compact_MBps
   # from phase A; predict's fallback is the A100 egress bound, 245 MB/s).

Copy home: profile + raw logs, crosscheck-out/ (manifests gzipped, layout
probes, diff verdicts), phase-timing log, the proof file size (not the
proof), pip freeze. Then run `python3 profiler/cli.py weightsplit
<S=1000 manifest> --machine b200-runpod-s2 --resident` at home: the first
weight-split numbers on an EXTRACTED projected manifest (the branch's
133.4 s / 1.90x headline is from the synth builder).
