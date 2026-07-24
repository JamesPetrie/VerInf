# Session handoff — claim-streaming study + witness spill (2026-07-23)

Interactive session AFTER the overnight autoresearch loop (see
`autoresearch-report.md` for the loop). Purpose of this doc: let the next run
resume without the full transcript. Durable technical detail lives in
`bench/optimization_registry.md` and `bench/research_journal.md` (iter9 +
iter9-CORRECTION); this is the map.

## Operating constraints (still in force)
- **No git** — no commits/branches/push. All prover changes stay uncommitted in
  the working tree. (Git is denied in settings; user said "без коммитов пока".)
- **ACCEPT is the gate** — every prover change must keep: byte-identical proof
  diff-test + Rust `verify_proof=ACCEPT` on a toy proof + fast suite green, else
  revert.
- **No cheating / be self-critical** — no fabricated numbers; every speedup a
  logged reproducible A/B; dead-ends are valid; never oversell a test-scale win
  as a production win. (This session I violated it once — see the LESSON — and
  the user caught it.)
- `Write` is NOT in the settings allowlist (causes prompts); tmux-armed
  ScheduleWakeup once fired unintentionally — don't arm wakeups in this session.

## THE HARNESS (enforced guard, not a lesson) — `bench/prod_lens.py`
Toy phase-shares do NOT transfer to 400B. On toy configs (d512/L4/seq512) the
Ligero params (K_DEG=1024, N_LIG=4096) are FIXED while model-compute is
negligible, so fixed-cost `encode` dominates and `witness` looks like ~6%. That
is an ARTIFACT — at 400B `witness_recompute` is **56.8%** of prove. I made exactly
this error this session (concluded "witness is 6%, spill is dead"); the fix is a
HARNESS, not a note:

**`analysis/bench/prod_lens.py`** — every A/B must end by calling
`prod_lens.report(title, toy_pct, effect=...)`. It prints the toy number labeled
"do NOT read as production" beside the authoritative 400B term shares (witness
56.8% / streaming 32.3% / quad 10.4% / lin 0.4%, from the cost model) and the
lever's production projection. `transfers=False` for test-only levers (GPU-mem
cache → projects 0%). Self-check asserts witness ~57%. Already wired into
`ab_witness_spill.py`, `ab_gpu_softmax.py`, `ab_witness_cache.py`; wire every NEW
A/B the same way. The lens gets its numbers from:
```
uv run python3 analysis/bench/cost_calculator.py --S 1093 --witness-mode notebook
  -> identity floor (encode+quad+lin) 12380s | witness_recompute 16288s (56.8%)
     | total 28668s (7.96h);  witness = 7.5TB, one pass T_WIT_S=3889s (1.08h)
```

## Claim-streaming (the 57% lever) — STUDIED, SOUNDNESS-BLOCKED
Merging the 4 witness passes ("finish each claim's rounds while its rows are
live") is NOT soundly implementable. Paper §5.3/§5.4: the 4 rounds fix each
commitment BEFORE the next challenge — load-bearing for the 2⁻¹⁶·⁶ soundness
(prevents fitting a witness to a seen challenge). Merging needs round-3/4
challenges during round-1. The current code only merges via `round_seeds()`
pre-deriving all challenges — its own docstring flags this as a TEST shortcut
(a non-transferable trap). Paper lists it as §9 future work. Verdict: needs a
protocol redesign; did not implement.

## Witness SPILL (the sound remnant) — BUILT + SOUND; prod benefit MODELED
Don't merge rounds; just re-read the witness in rounds 2-4 instead of
recomputing. Store-once, read-3×.
- **Code** (`prover/core.py`, opt-in `LIGERO_WITNESS_SPILL=1`, default OFF):
  `_WITNESS_SPILL_ON`, `_host_spill_budget_bytes`, `_spill_store`/`_spill_load`
  (pinned int64-view → byte-identical, sidesteps uint64 CUDA/pin gaps), wired
  through the existing `witness_cache` with a `_spill` flag.
- **Sound**: `bench/validate_witness_spill.py` (byte-identical) +
  `bench/accept_toy_spill.py` (Rust ACCEPT) — both PASS.
- **Toy A/B is net-negative** (`bench/ab_witness_spill.py`: −0.7% seq512,
  −3.8% seq1024) — a SCALE ARTIFACT (toy recompute trivial → PCIe overhead wins).
- **Production crossover INVERTS** (`bench/spill_costmodel_prod.py`): effective
  recompute throughput = 7.5TB/1.08h = **1.93 GB/s**, below any NVMe. Modeled
  win: **+13% (NVMe 3.5) → +28% (NVMe 7) → +34% (RAID/PCIe-cap)**; loses only on
  HDD (1.5 GB/s). This is a PREDICTION (can't run 400B here), like coset-NTT —
  not a measured win. Don't oversell, don't dismiss.
- **Prototype gaps for production** (the next step, user paused it — "Пока нет"):
  (1) DISK backing — 7.5TB ≫ 84GB host RAM, so host-memory backing won't hold it;
  (2) FULL-witness coverage — prototype spills only softmax/silu, need all rows
  for the full 57%.

## BabyBear / adaptive field — decision layer BUILT, port-blocked (iter10)
Field is hardcoded to Goldilocks in 11 files (Python roots pow(7,(P-1)/K,P), CUDA
`GL_P`, Rust `field.rs`); a real BabyBear prove is a multi-file Python+CUDA+Rust
port (modulus, generator, 2^27-adic roots vs Goldilocks 2^32, 31-bit packing).
NOT done, not faked. Built the sound decision layer:
- **`bench/field_policy.py`** — sound worst-case per-op accumulator bounds
  (weight-matmul tight `s_a·s_b·R·||W||₁`; attention generic `k·(sR)²`; P·V with
  softmax operand ≤1) + policies + field assignment.
- **`bench/run_field_variants.py`** — run variants: `--policy
  {goldilocks,babybear,adaptive-proof,adaptive-op}`, `--all`, `--sweep-s`
  feasibility map, first-order payoff via prod_lens.
Verdict: at s=2^12 only elementwise fits BabyBear (30-bit ceiling). Weight
matmuls fit at s≤2^8, all ops at s≤2^4. **The limiter is PRECISION, not field
mechanics** — fitting BabyBear means shrinking s (fewer fractional bits).
adaptive-op (mixed) at s=2^6 ≈ +7% (est.), capped because the O(seq²)
attention-score witness dominates at long context and can't leave Goldilocks.
**Real next step: an ACCURACY experiment** — prove a small model at reduced
s (2^8, 2^6) and check inference output still matches — before any backend port.

## Field-size / BabyBear thread (earlier in session)
- Field-size worst-case (SOUND, not typical): `b = log₂k + 2·log₂s + 2·log₂R +1`.
  Tighter sound bound for weight matmuls: `s²·R_clip·||W_j||₁` (weights are
  committed/fixed; only input is adversarial) — much tighter than notebook's
  `k·(sR)²`. Scripts: scratchpad `worstcase_field.py`.
- **SmoothQuant does NOT reduce the field requirement** — product invariance
  (the per-channel scales cancel term-by-term, accumulator identical). Verified:
  scratchpad `smoothquant_field.py`. So it doesn't unlock BabyBear.
- Attention scores (act·act, no fixed operand) are the BabyBear-binding
  constraint; weight matmuls fit BabyBear under the tight bound, scores don't.
- Open idea (not built): a COMBINATION of fields — run the rare non-typical
  case on a larger field, bulk on BabyBear. `babybear_map.html` (published
  artifact) is the 3D feasibility map over (k, s, R).

## Demonstrated production-transferable wins so far
Only the **GPU softmax port** (loop iter8, `compute_fns.py`, `LIGERO_GPU_SOFTMAX`)
truly transfers — it speeds each of the 4 witness recomputes at any scale, denting
the 57% term; grows with context (O(SEQ²)). **GPU silu** also ported
(`LIGERO_GPU_SILU`). The witness CACHE (GPU-mem) is test-scale-only (doesn't fit
at 7TB). coset-NTT and spill are unmeasured-at-prod predictions.

## Suggested next steps (user has NOT chosen — ask)
1. Disk-backed, full-witness spill (the real +13-34% lever; mechanism already
   sound-proven; work is I/O plumbing + coverage, not protocol).
2. silu/rmsnorm remaining GPU work / re-measure seq768 post-GPU-softmax loose end.
3. BabyBear combination-of-fields prototype.

## Key files created this session
- **`bench/prod_lens.py`** — the production-lens harness (MANDATORY for every A/B)
- **`bench/field_policy.py`**, **`bench/run_field_variants.py`** — BabyBear /
  adaptive-field decision layer + run variants (iter10)
- `bench/spill_costmodel.py`, `bench/spill_costmodel_prod.py` (crossover math)
- `bench/ab_witness_spill.py`, `bench/validate_witness_spill.py`,
  `bench/accept_toy_spill.py` (spill A/B + gates)
- `prover/core.py` spill code (see above)
- scratchpad: `worstcase_field.py`, `smoothquant_field.py`
- Artifacts (published): `dashboard.html`, `babybear_map.html`
