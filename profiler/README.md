# VerInf dry-run profiler

First deliverable of the multi-GPU (Blackwell) parallelization effort: predict
a proving run's exact costs — time, memory, bandwidth, proof size — *before*
running it, on hardware we haven't touched yet, and expose the dependency
structure a dispatcher/scheduler will consume. Everything here is measurement
scaffolding for the future dispatch architecture; nothing touches the
soundness-critical `prover/` path.

## Architecture

One contract, two producers, several consumers:

```
                    ┌─ extract.py  (tape walker; runs where the prover runs; EXACT,
                    │              model-agnostic — anything that builds a Tape)
  manifest.json  ◄──┤
                    └─ synth.py    (closed-form builders: maverick, maverick-projected, llama7b;
                                   pure Python — runs anywhere, cross-checks extract)
        │
        ├─► predict.py   × machines/<name>.json  →  cost report
        └─► dag.py       →  dependency DAG + parallelism profile
```

- **`manifest.py`** — the workload manifest: one record per tape op (type,
  shape params, produced/consumed variables, exact witness slots when
  extracted). The scheduler's future work-list format.
- **`claimcosts.py`** — per-claim `(W, #cids, Q)` accounting, the
  active-in-production forms from `analysis/maverick-cost-model.md`.
- **`machines/`** — measured hardware constants with provenance strings.
  `gb10-spark.json` is seeded from `analysis/` measurements;
  `_template-blackwell.json` is the calibration checklist for cluster access.
  `interconnect` is null by design — topology is unknown as of 2026-07; the
  schema grows when it lands.
- **`predict.py`** — totals, bracketed prove time, memory terms, proof/verify
  estimates, ideal N-GPU what-ifs.
- **`dag.py`** — claim-level dependency graph (weight variables excluded —
  they're streamed inputs, not dataflow edges), critical path, width profile.
- **`partition.py`** — the strategy scorecard: maps claims to N shards
  (`rows` contiguous tape ranges, `layers` pipeline blocks, `experts`
  round-robin over the MoE fan with a layer backbone), computes per-shard
  work/imbalance, reduction-aware cross-shard traffic (Freivalds combines
  send one output-sized partial per remote shard, not their inputs; LogUp
  multiplicity vectors likewise reduce as one table-sized partial per
  participating remote shard — extracted manifests only), weight
  streaming, per-shard opened-column memory — and prices comms across a
  swept interconnect bandwidth since topology is unknown. Conclusions stable
  across the sweep are actionable now; ones that flip wait for topology.

## Usage

```sh
python3 profiler/demo.py        # guided tour of everything below (no torch)
python3 profiler/cli.py synth   --model maverick --seq 1000 --t-queries 40 -o man.json
python3 profiler/cli.py predict man.json --machine gb10-spark
python3 profiler/cli.py predict man.json --machine gb10-spark --gpus 8      # ideal scaling
python3 profiler/cli.py dag     man.json -o dag.json
python3 profiler/cli.py partition man.json --shards 8 --weight-bytes-per-param 0.7   # compare strategies
python3 profiler/cli.py partition man.json --shards 8 --strategy experts             # one in detail
python3 profiler/crosscheck.py maverick --from-gguf <gguf> --t-queries 54 --seq 1000 --layout
python3 profiler/calibrate.py --name <sku> --tmpdir /local/disk
python3 profiler/instrumented_prove.py --from-gguf <gguf> --t-queries 54 --prompt-n 500 --cont-n 500
#   the hardware-session tools (GPU box): extraction cross-check against
#   synth + the prover's own layout, machine-profile calibration, and the
#   research timing prove (the demo's proof path is fail-closed on main) —
#   RUNBOOK-blackwell.md is the session script.
python3 profiler/cli.py weightsplit man.json --machine b200-runpod --resident --intervals 2
#   stage-aware wall for the weight-split prover (coordinator + enrolled-block
#   workers) from executable whole-variable plans with exact cuts: commit +
#   max(fold) + max(open) across the s_col barrier; physical (row-padded) slots;
#   resident (union hold vs HBM; capped tied plans exact at N=2, labelled
#   -heuristic at N>=3) or streaming lower bound with --disk-mode
#   shared|per-device and --io-overlap none|perfect; reports the kernel-floor
#   ratio and the same-mode speedup (n/a when N=1 is not executable). Enrolled only.
```

First findings from the evaluator (Maverick, 8 shards, floor model):

- **Interconnect bandwidth does not bind** — for every strategy, at every
  swept bandwidth (25–900 GB/s), at S=1000 *and* S=100k, cross-shard
  traffic is <1% of wall. Even `rows` at S=100k, which naively ships ~19
  TB/sweep by cutting through S² attention matrices, amortizes against
  S²-growing compute. The real multi-GPU risks are chain serialization,
  barrier idle, and memory capacity — not comms volume.
- **Memory capacity binds first at frontier context**: the opened-column
  payload at S=100k, T=40 is ~2.2 TB *per shard* across 8 GPUs. R4/proof
  handling must stream opened columns off-device as they're produced —
  a protocol-adjacent design item to raise well before big-context runs.
- All three strategies load-balance to within a few percent; `layers` has
  the least traffic (~0.3 GB/sweep at S=1000 — thin residual handoffs),
  `experts` also shards the weight-streaming I/O per GPU.

On a GPU box, the exact path (see `extract.py` docstring): build any model's
`Tape(cfg, lazy=True)` exactly as the demos do — claim recording is
shape-only, no weights load, no witness computes — then
`extract_tape(tape, ...).save("man.json")`. `python profiler/extract.py
--selftest` smoke-tests the walker on a toy tape. `python
profiler/test_profiler.py` (or pytest) runs the no-torch regression suite —
extractor semantics (including property-backed mode flags), cost identities
and mode rejection, partition traffic, and CLI/manifest validation — on
any box.

## Validation (synthetic Maverick S=1000 T=40 vs the archived hidden run)

| quantity | predicted | measured (`analysis/full-model-hidden-run-archive.md`) |
|---|---|---|
| witness rows `m_total` | 108.7 M | 109.27 M |
| proof size | 93.1 GB | 93.6 GB |
| opened-column GPU payload | 34.8 GB | ~35 GB (derived) |
| verifier peak RSS | 76.1 GB | 75.7 GB |
| proof dump time (legacy decimal JSON) | 751 s | 756 s |
| prove wall-clock | 3.0 h floor / 9.9 h aggregate | 14.26 h |

Prove time is reported as a **bracket**, per `analysis/maverick-cost-model.md`:
the *floor* (`A·W + B·#cids + C·Q`) is the NTT-bound, post-fold-reorg target
where `A`/`C` ride memory bandwidth and `B` rides compute — the reason they
scale to Blackwell by *different* ratios; the *aggregate* (`~40 ns/slot · W`)
is calibrated on today's code and undercounts fixed overheads (UI machinery,
hidden-prompt one-hot, weight-commit I/O — none in the synthetic model yet).
The gap between bracket and measurement is not noise: it is the profiler's
work-list.

Same for memory: the report itemizes known terms (opened columns, encode
working set, streaming-order activation live-set) and says explicitly that
measured peaks exceed them — attributing the remainder (fold accumulators,
in-flight weights, allocator slack) is a measurement task, done with
`LIGERO_STREAM_DBG` / `LIGERO_COMPILE_PROFILE` traces on real runs.

## Roadmap

1. **On-Spark cross-check** — run `extract.py --selftest`, extract the real
   Llama-7B and Maverick tapes, diff against the synth manifests and against
   `LIGERO_LAYOUT_BREAKDOWN=1`. Divergence = stale formula; trust the tape.
2. **Validation mode** — parse `LIGERO_PHASE_TIMING` / `LIGERO_EPHASE`
   output from instrumented runs and diff predicted vs measured per phase;
   recalibrate `prove_constants` from the residuals.
3. **Blackwell calibration** — port `prover/deprecated/spark-bench/`
   (field-mul, NTT, BLAKE3, matmul) plus memcpy/H2D/disk benches into a
   suite that fills a machine profile in one run. First thing to run when
   cluster access lands.
4. **Interconnect model** — when topology is known: extend the machine
   schema (link bandwidth, domain size, hierarchy), replace the swept
   bandwidth in `partition.py` with the real hierarchy.
5. **Scheduler sim** — the partition evaluator scores work and traffic but
   not dependency-chain serialization or barrier idle; a discrete-event sim
   over the DAG with per-claim times (from a `time_ops=True` run) closes
   that gap and doubles as the dispatcher's dry-run test harness.
6. **Scheduler handoff** — the manifest + DAG is the dispatcher's input:
   per-claim work vectors, variable lifetimes, round barriers. The
   Tomasulo-style core schedules against exactly this structure.

## Caveats

- Synth builders cover the team-standard workloads (Maverick, Llama-2-7B,
  and `maverick-projected` — the routed-projected protocol, mirroring the
  current demo; pair it with `--enrolled-weights`) and omit UI/MaxClaim and the hidden-prompt one-hot chain
  (~1,100 claims in the archived run — flagged in `maverick-cost-model.md`
  as structural, sub-1% of W at current scale; `embed.select` is therefore
  a zero-in-degree source here where the real tape derives it from the
  one-hot routing mask). The synth `routing` row bundles the
  RangeWord/WordExtraction aux, matching the cost model's form; extracted
  tapes cost the pieces separately (`routing_core` + per-claim word/range
  formulas — the sum is identical). LogUp table commitments (mult/w,
  ~2·T_LEN each) and their `TableSettlement` claims (T_LEN+1 cids) are
  extracted from tapes but not modeled by synth.
- `claimcosts` prices the production claim modes (rescale ⓡ on
  matmul/hadamard/rope/rmsnorm, saturating causal softmax, rescale-free
  silu). matmul and hadamard accept both rescale settings; for the rest, a
  manifest carrying a non-production mode flag is rejected with a clear
  error rather than priced with the wrong formula — the formula table grows
  when such a mode ships. Extracted W is always exact regardless (tape
  `w_slots` beats the formula); the rejection protects cids/Q.
- Synthetic row totals (`m_total` and everything derived: proof size,
  opened-column memory, verifier RSS) are labeled approximate when
  formula-only aux slots are row-packed at W/ELL density; core layout
  rounds every variable up independently. Extracted manifests itemize
  every variable, so their rows are exact.
- The DAG's ideal-speedup figure covers *witness generation* parallelism
  only; encode/hash work is per-row shardable regardless of the DAG, and the
  the streaming sweeps are hard barriers: four (R1, R2, test
  polynomials, openings), five when the tape has a phase-3 late-aux
  commitment — routed-projected tapes always do (`partition.n_sweeps`).
- `extract.py` is written against the tape API but has not yet run on GPU
  hardware — item 1 above is the gate before trusting extracted manifests.
  Claim labels/layers on that path are parsed from output-variable names
  (`L{n}_`/`_L{n}`/`blk.{n}` conventions, best-effort) — the first real
  extraction should eyeball them before trusting `layers`/`experts`
  partition scores; both strategies refuse manifests with no layer metadata
  rather than silently piling everything on shard 0.
