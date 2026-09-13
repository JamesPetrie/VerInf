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
# (session 1's recipe; on the Runpod cu128/torch-2.8 image use the image's
#  torch and the Session 2 bootstrap below — a venv does not see it)
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
SSH drops. Kill with `kill -- -$(cat ~/<name>.pid)` (the process group;
the pid alone is the supervisor), never `pkill -f`.

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

# Session 2 — hardware crosscheck of the projected protocol + the routed-cache A/B

Goals: close the last validation gap of the sync milestone (extraction of
the new claim types on real hardware), fill the five new calibration
fields, run the first instrumented PROJECTED prove on Blackwell, and answer
the witness-cache question per sweep (the routed-output cache A/B; witness
cache handoff, 2026-09-09). Prepped 2026-08-23; revised 2026-09-13 after
the author's readiness audits: every phase below has a gate, long steps run
through the checked launcher and are waited on, short steps run through a
helper that returns the command's own status, the download is a checked
tool, and the timed proves pin their geometry and flags so the arms cannot
differ by an environment variable.

Rent: one B200, secure (US-NE-1 had stock on 2026-09-13, on CUDA 13.0
hosts — the cu128 image runs under a newer driver and its nvcc already
targets sm_100), image runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404,
400 GB CONTAINER disk, no network volume (session 1's network FS starved
the weight lane at 0.3 GB/s), minRamPerGpu 256 (the retained openings
alone need ~26-29 GB of host RAM at S=1000, on top of loader buffers and
the routed cache's host tier), ports 22/tcp, PUBLIC_KEY in env. Budget:
roughly 4-5 h ≈ $27-34 at $6.79/h plus $6.79 per extra hour — an estimate
until an arm is measured. Shards: 232.13 GB; proof dump: ~36 GB.

## 0. Bootstrap — gate: the record, the verifier build and the primitives test all exit 0

Look at the storage first, because it decides where everything goes.
`/workspace` is where Runpod mounts a VOLUME when one exists; with none it
sits on the container overlay, and only findmnt says which. The model,
the probe dir and the proof must be on the 400 GB container disk:

```sh
findmnt -T /workspace -o SOURCE,FSTYPE,TARGET,SIZE,AVAIL; findmnt -T / -o SOURCE,FSTYPE,TARGET,SIZE,AVAIL; df -h / /workspace
```

Then set the environment ONCE, in the interactive shell — never inside a
piped or braced block, whose exports die with its subshell. Every later
phase inherits these, and the two helpers are used by every phase:
`waitfor <name>` returns when a launcher job has written its EXIT line and
returns the job's success; `logrun <name> <cmd...>` logs a foreground
command to $VERINF_LOGS/<name>.log and returns the COMMAND's status, not
tee's.

```sh
export VERINF_ROOT=/workspace                 # or a dir under / if /workspace is a separate volume
export VERINF_GGUF_DIR=$VERINF_ROOT/gguf
export VERINF_GGUF=$VERINF_GGUF_DIR/UD-Q4_K_XL/Llama-4-Maverick-17B-128E-Instruct-UD-Q4_K_XL-00001-of-00005.gguf
export VERINF_OUT=$VERINF_ROOT/crosscheck-out; export VERINF_PROOF=$VERINF_ROOT/proofs/mavp-s1000.bin
export VERINF_LOGS=$VERINF_ROOT/logs
export PATH=/usr/local/cuda/bin:$PATH         # nvcc is not on the login PATH
mkdir -p "$VERINF_OUT" "$(dirname "$VERINF_PROOF")" "$VERINF_ROOT/probe" "$VERINF_LOGS"
waitfor() { until grep -q '^EXIT=' ~/"$1".log 2>/dev/null; do sleep 60; done
            grep -qx 'EXIT=0' ~/"$1".log && return 0
            echo "$1 FAILED: $(grep '^EXIT=' ~/"$1".log)"; return 1; }
logrun()  { local name=$1; shift; "$@" 2>&1 | tee "$VERINF_LOGS/$name.log"
            local rc=${PIPESTATUS[0]}; [ "$rc" -eq 0 ] || echo "$name FAILED rc=$rc"; return "$rc"; }
pip install --break-system-packages blake3 gguf safetensors ninja transformers hf_transfer
command -v cargo >/dev/null || curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal; . ~/.cargo/env
```
The image's torch is 2.8.0+cu128 and that is the environment. NEVER
`python3 -m venv` (it does not see the image's torch; requirements.txt is
unpinned) and never `uv sync` (the lock resolves torch 2.13+cu126 for the
V100 dev box). Now the record and the two builds, each required to succeed:

```sh
logrun env-record tools/session2_env.sh && cp $VERINF_LOGS/env-record.log $VERINF_ROOT/session2-env.txt \
  && logrun cargo cargo build --release --manifest-path verifier/Cargo.toml --bin verify_proof \
  && logrun cuda-primitives python3 prover/tests/run_tests.py test_cuda_primitives \
  && git diff > $VERINF_ROOT/session2-dirty.diff && pip freeze > $VERINF_ROOT/session2-pins.txt \
  && echo "BOOTSTRAP OK"
```
(the record covers torch/CUDA/nvcc versions, device and capability, host
RAM and the cgroup limit, the filesystems under the chosen paths, the
revision and dirty state; the primitives test JITs the CUDA extension)

## A1 + B. Compute calibration and the shards, overlapped — then a barrier

A1's compute benches and B's download overlap; nothing else does. The
llama crosscheck uses the GPU (its layout subprocess runs the model
engine), so it waits for A1's completion — the barrier is in the commands,
not only in the prose.

```sh
tools/spark_run.sh calib-a1 python3 profiler/calibrate.py --name b200-runpod-s2 \
    --tmpdir $VERINF_ROOT/probe --skip-io
tools/spark_run.sh gguf-pull tools/gguf_pull.sh $VERINF_GGUF_DIR
waitfor calib-a1                                       # no other GPU work before this returns
logrun xchk-llama7b python3 profiler/crosscheck.py llama7b --seq 100 --layout -o $VERINF_OUT
waitfor gguf-pull
```
A1 compiles the four session-2 benches (first compiles of ntt_batched /
hbm_random / launch_latency on sm_100 — graceful null on failure), sizes
the ALU-bench grids to the SM count, writes a PARTIAL profile if any later
probe dies. SANITY: "ntt BATCHED ... vs expected if encode scaled with
bandwidth" IS the A-constant check; it now uses the LARGEST batch of the
sweep and the provenance lists every batch — read the m=512/2048 values,
never m=1, which is cache-resident. Gate: every gpu.* field in the printed
profile is non-null (a null is a skipped bench; exit 0 is not
completeness). The llama crosscheck validates the ModelConfig mirror.

B's probe validates eight parallel range requests (exit code, 206, exact
bytes, timeout) and rates the bytes that arrived; below 100 MB/s, or on
any bad range, it fails and nothing is pulled. The pull is the pinned
revision 41032e5 from profiler/data/maverick-ud-q4_k_xl-shards.json;
every size and every sha256 is checked against the pins. Gate: "every
shard hashes to its pin" in ~/gguf-pull.log. Storage probes never overlap
the pull; compute benches may.

## A2. Storage calibration — after B, storage idle (~10 min)

```sh
logrun calib-a2 python3 profiler/calibrate.py --name b200-runpod-s2 --tmpdir $VERINF_ROOT/probe --io-only
```
Merges the sequential-read, compact-dump and dump-proxy rates into the A1
profile and records the probed directory with its filesystem (provenance
`io_tmpdir`). The three fields are cleared before probing: a probe that
fails leaves null with "FAILED" in its provenance and the command returns
1, so an earlier mount's rate cannot survive under this mount's
description. Gate: logrun's status 0, io.* non-null, io_tmpdir names the
container disk, not a volume.

## G. Gates before any timed prove (~20 min)

```sh
logrun gates tools/session2_gates.sh
```
The script runs the gates in order and stops at the first failure with its
status (a pipeline cannot hide it; each gate also logs to
$VERINF_LOGS/gate-<name>.log): the K-quant kernel against the real shards
(the file runs directly, since its check is main() and run_tests.py would
find nothing; it must print `=== kquant_kernel: 4/4 PASS ===`), the
shard-streaming suite at 7/7 with the three routed-cache gates, the routed
suite at 6/6, the toy A/B's counts, and a two-layer REAL-GGUF proof through
the research driver with `--verify`, which must print "rust verify_proof:
ACCEPT" and "opened columns match committed leaves: True" — the prover now
asserts the latter on every path, so a False can no longer produce EXIT=0
and a timing. The driver had never run on a GPU before this session: if
it dies here, the traceback is the deliverable and the fallback is the
demo's --enroll-weights run (enrollment timing only) plus
admission_bench.py --runs 30 for the kernel stages; neither answers the
A/B question. Gate: "== all gates passed" and logrun's status 0.

## C. The projected crosscheck (~45-60 min)

```sh
logrun xchk-mav-small python3 profiler/crosscheck.py maverick --from-gguf $VERINF_GGUF --t-queries 54 \
    --prompt-n 2 --cont-n 2 --layout -o $VERINF_OUT
tools/spark_run.sh xchk-mav-s1000 python3 profiler/crosscheck.py maverick --from-gguf $VERINF_GGUF \
    --t-queries 54 --prompt-n 500 --cont-n 500 --skip-selftest --layout -o $VERINF_OUT
waitfor xchk-mav-s1000
```
The keeper is 500+500, the timed prove's own split — NOT `--seq 1000`,
which means 2+998 and a different UI chain (996 extra selection and
addition claims); crosscheck refuses `--seq` with any explicit split. The
tape carries RoutedProjectedMatmulClaim / RescaleClaim and the script
diffs against synth maverick-projected ("protocol on the tape:
maverick-projected"; "maverick" means the GGUF build fell back to the
legacy fan — stop). The layout probe is in-process and value-free, so it
runs at S=1000; `--layout` also writes maverick-s1000-composition.txt, the
witness by phase and origin (produced / committed / weights / p2 / p3),
the measurement behind the witness-regeneration note's 517 GB and 243 GB
estimates. The diff verdict is console output: it lives in
~/xchk-mav-s1000.log. Gate: RESULT: clean, the composition file present,
and the manifest gzipped and reloadable:
```sh
python3 -c "import sys; sys.path.insert(0,'profiler'); from manifest import Manifest; \
  Manifest.load('$VERINF_OUT/maverick-s1000-extracted.json').save('$VERINF_OUT/maverick-s1000-extracted.json.gz'); \
  print(len(Manifest.load('$VERINF_OUT/maverick-s1000-extracted.json.gz').claims), 'claims reload')"
```

## D2. The routed-cache A/B at S=100, per sweep — BEFORE D

Two arms, one after the other, same tape (the driver's token seeds are
fixed; enrollment and ZK entropy are fresh per arm, so the two proofs are
not byte-identical — that property is the toy gate's, with a shared
enrollment). The driver pins ELL=8192, K_DEG=16384, N_LIG=65536 and
T_QUERIES itself and sets LIGERO_ROUTED_Y_CACHE 0 or 1 from --routed-cache,
so no inherited environment can decide either; the ordinary witness-cache
and spill flags are named so both arms hold them fixed. An S=100 arm is
NOT ten times cheaper than S=1000: the enrolled-weight fold and open are
S-independent (about 190 s of floor at S=100 against 254 s at S=1000) and
each arm pays a fresh enrollment and reveal pass outside PROVE WALL —
budget the whole arm from LAUNCH to EXIT.

```sh
ARM="python3 -u profiler/instrumented_prove.py --from-gguf $VERINF_GGUF --t-queries 54 --prompt-n 50 --cont-n 50 --sweep-timing"
if LIGERO_WITNESS_CACHE=1 LIGERO_WITNESS_SPILL=0 LIGERO_WITNESS_SPILL_DISK=0 \
       tools/spark_run.sh mavp-s100-off $ARM && waitfor mavp-s100-off; then
    LIGERO_WITNESS_CACHE=1 LIGERO_WITNESS_SPILL=0 LIGERO_WITNESS_SPILL_DISK=0 \
        tools/spark_run.sh mavp-s100-on $ARM --routed-cache && waitfor mavp-s100-on
else
    echo "D2: the off arm did not succeed; the on arm was not launched"; false
fi
```
spark_run.sh detaches into its own session and ALWAYS writes an EXIT line,
"EXIT=killed ..." included, so `waitfor` terminates on a started job and
never waits for one that was not launched; the block's status is the last
arm's, or failure when the off arm failed. To cancel a run:
`kill -- -$(cat ~/<name>.pid)` (the whole process group — the plain pid is
the supervisor). Gate per arm: EXIT=0; "opened columns match committed
leaves: True"; five sweep rows (R1, R2, R3, fold, open); proj = 0,72,0,0,0
(one projection per routed claim, all in R2: 24 MoE layers × three routed
matrices); on the ON arm the R3, fold and open rows lose the routed shard
reads while R2 keeps them for the projection, and fold and open keep the
enrolled block's encode-path reads on both arms; the routed cache has its
OWN columns — routed_wr = 72 in R1 and routed_rd = 72 in each of R2, R3,
fold and open on the ON arm, all zero on the OFF arm — while cache_rd and
cache_wr belong to the ordinary witness cache (48 softmax and 72 silu
claims on this model) and are the same on both arms. Reading the columns:
fetch is the WHOLE loader call (storage read, host preparation, decode,
transfer) taken out of the bucket it ran in; witness is compute with the
fetches removed; loads and load_GB are loader calls and the DECODED field
bytes they returned, not packed or disk bytes (a group loader resolves a
whole attention group to return one member); the walls are instrumented
(cuda-synced) — an uninstrumented speedup is not established by these
tables. The dense claims' weights resolve more than once per sweep
(compute fetch, then the aux's lazy dict); the loads column shows it.

## D. The S=1000 instrumented prove, cache OFF, proof dumped (~30-60 min, unmeasured)

```sh
LIGERO_WITNESS_CACHE=1 LIGERO_WITNESS_SPILL=0 LIGERO_WITNESS_SPILL_DISK=0 \
    tools/spark_run.sh mavp-s1000-off \
    python3 -u profiler/instrumented_prove.py --from-gguf $VERINF_GGUF --t-queries 54 \
    --prompt-n 500 --cont-n 500 --sweep-timing --dump-proof $VERINF_PROOF \
    && waitfor mavp-s1000-off
```
NOT demo_maverick_full.py: its proof path is fail-closed on current main
(enrollment + trusted root + public Sz + a 714-run admission report with
measured five-sweep semantics). instrumented_prove.py is the research
harness: same tape, same prover, throwaway in-process enrollment, reveal
pass for Sz, both timing modes — a timing run, never a production proof.
It refuses to start unless the dump directory has 60 GB free (the
projected proof is ~36 GB), creates and fsyncs the proof's temporary file
by exclusive creation — an existing .part is refused, never truncated —
refuses to overwrite an existing proof, and refuses early-exit diagnostics
(LIGERO_LAYOUT_BREAKDOWN, LIGERO_COMPILE_PROFILE_EXIT). Deliverables:
measured wall against the 254 s kernel floor (the floor EXCLUDES the five
semantic sweeps — the A100 admission bound put 2,024 s there; expect the
wall to be dominated by them); the per-sweep table at S=1000; separate
build / enrollment / reveal / prove / dump times; enrollment time on
Blackwell; a real compact-dump rate and fsynced file size (cross-check
io.proof_dump_compact_MBps from A2). Gate: EXIT=0, leaf check True,
five sweep rows, the SUMMARY line.

Optional D-on: the same command with `--routed-cache`, WITHOUT
--dump-proof, name mavp-s1000-on, only after D's EXIT=0 and only if D2's
measured savings and D's full elapsed cost justify another hour.

## Copy home — gate: each item present before the pod is terminated

The revision and dirty diff (session2-dirty.diff), session2-env.txt and
session2-pins.txt, $VERINF_LOGS/ (the logrun logs and the per-gate logs)
and every ~/<name>.log the launcher wrote with its .memwatch (calib-a1,
gguf-pull, xchk-mav-s1000, the arms: the per-sweep tables, phase reports,
EXIT lines and the job group's memory trace), the effective commands as
run, profiler/machines/b200-runpod-s2.json plus
calibrate-raw-b200-runpod-s2/, $VERINF_OUT/ (manifests gzipped, layout
probes, composition, diff verdicts), the proof's size and dump rate (not
the proof), and any traceback. Then terminate the pod. At home:
`python3 profiler/cli.py weightsplit <S=1000 manifest> --machine
b200-runpod-s2 --resident`, the first weight-split numbers on an EXTRACTED
projected manifest.
