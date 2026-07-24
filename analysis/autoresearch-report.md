# Autoresearch loop — prover optimization report (2026-07-22 → 07-23)

An autonomous cold-restart loop (`prover-autoresearch` skill) ran overnight to
drive down VerInf prover `prove()` time. It picked one bottleneck per iteration,
prototyped on small/medium transformers, A/B'd against the notebook's cost-model
theory, validated with byte-identical proofs + Rust `verify_proof=ACCEPT` + the
fast test suite, and applied changes only when green. This is the record of what
it did. Every applied change was re-verified as a stack at the end: the final
toy proof with all three changes on (default flags) is **Rust ACCEPT**.

## Headline: prove() speedups (all soundness-verified)

| config (d,d_ff,d_h,SEQ,L) | before | after | speedup |
|---|---|---|---|
| 512,1024,64,256,4  | 17.84s | **12.12s** | 1.47× |
| 512,1536,64,384,4  | 36.17s | **21.39s** | 1.69× |
| 512,2048,64,512,4  | 62.13s | **24.44s** | 2.54× |
| 512,1536,64,768,4  | 75.39s* | (not remeasured post-GPU-softmax) | — |
| 512,2048,64,1024,4 | 218.28s | **59.92s** | **3.64×** |

\*seq768 got the cache (iter3) but was not re-measured after the GPU-softmax port
(iter8), so its real current number is lower than 75.39s — an open loose end.

The win **grows with context length** because softmax witness cost is O(SEQ²);
at seq1024 the two changes together turned a 3.6-minute prove into ~1 minute.

## What was changed (all in `prover/`, uncommitted per the no-git instruction)

1. **`core.py` — witness cache** (iter2). The sound prover re-runs the witness
   forward pass once per Fiat-Shamir sweep (4×). The compute_fn outputs are
   deterministic, so `_stream_sweep` now shares a `witness_cache` across the 4
   sweeps: softmax/silu outputs are computed on sweep 1 and reused on 2–4. Only
   challenge-independent outputs are cached; the aux/Freivalds witnesses are not.
2. **`core.py` — memory-fraction budget** (iter4). The first cut used a fixed
   `2e8`-element cap, which (found in iter3) was ~3× too small at seq1024 and
   silently stopped caching mid-sweep, collapsing the win to 10.8%. Replaced with
   a budget = 0.25 × free GPU memory (`torch.cuda.mem_get_info`), so it
   auto-scales and only degrades to recompute when memory is genuinely tight.
3. **`compute_fns.py` — GPU softmax port** (iter8). `_softmax_witness_vec` ran on
   CPU numpy (an O(SEQ²) causal-bracket binary search + host↔device copies).
   Ported to resident GPU int64 tensors behind `LIGERO_GPU_SOFTMAX` (default on);
   the numpy path is kept as fallback. ~95× faster in isolation; the biggest
   single lever at large SEQ.

## The iterations (chronological, including the dead-ends)

- **iter1** — Confirmed the 4× witness recompute by measurement (one forward
  pass 5.19s vs the 4-sweep witness bucket 21.9s = ratio 4.22×). Quantified the
  cache ceiling at ~42% of prove. *No code change.*
- **iter2** — Implemented the witness cache. seq384: 36.17→21.39s (42.7%).
  Proof byte-identical, ACCEPT. **Applied.**
- **iter3** — Swept the cache across configs; found the seq1024 win collapsed to
  10.8% instead of the predicted ~50%. Root-caused it to the fixed element cap
  engaging mid-sweep; confirmed by lifting the cap (214.89→98.10s). *Diagnosis,
  no code change.* (A prediction-vs-measurement disagreement chased to its cause,
  not tuned away.)
- **iter4** — Applied the memory-fraction budget fix. seq1024: 200.51→97.30s
  (2.06×). Byte-identical, ACCEPT, 71 tests green. **Applied.** Cumulative with
  iter2 at seq1024: 218.28→97.30 = 2.24×.
- **iter5** — Probed whether extending the cache to matmul/rmsnorm/rope helps.
  **Dead-end, recorded honestly:** measured that 85% of the remaining witness is
  softmax's *own one-time* compute (already cached — caching can't remove a cost
  paid once), while matmul is cheap-to-recompute but memory-heavy. Cache lever
  exhausted. *No code change.*
- **iter6** — Profiled softmax internals before porting: 65% binary search + 26%
  output/saturate, all vectorized numpy, GPU-friendly; built a byte-exact timed
  reference. *No code change.*
- **iter7** — Prototyped the GPU softmax port (95× isolated). *Prototype.*
- **iter8** — Applied the GPU softmax port. seq512 33.23→24.44s (26.4%), seq1024
  100.77→59.92s (40.5%). Proof BYTE-IDENTICAL (numpy vs GPU), Rust ACCEPT,
  branch coverage green. **Applied.** Cumulative seq1024: 218.28→59.92 = 3.64×.
- **iter9** — Was re-running the phase breakdown at seq1024 (witness is now only
  ~6%, so the frontier moved to quad ~14% and encode ~13%) when the loop was
  stopped. *In progress, nothing applied.*

## Validation discipline (the anti-cheat gate held)

- Every **applied** change: proof **byte-identical** to the pre-change path
  (`root_p1/root_p2/q_irs/q_lin/p_0/opened columns` all equal, cache off-vs-on
  and numpy-vs-GPU) AND standalone Rust `verify_proof=ACCEPT` AND the fast suite
  (71 tests) green.
- Predictions were written **before** measuring, so measured-vs-predicted gaps
  were findings (iter3's collapse), not post-hoc rationalizations.
- Two honest dead-ends recorded so a future run won't re-walk them (iter5 cache
  extension; the iter3 anomaly diagnosed rather than papered over).
- Everything logged to `analysis/bench/prove_runs.jsonl` (browse with
  `show_runs.py`); full narrative in `analysis/bench/research_journal.md`.

## Still open (honest loose ends)

- **`prover/tests/test_witness_cache.py` was never formalized** — the
  byte-identical soundness gates live as scripts in `analysis/bench/`
  (`validate_witness_cache.py`, `validate_gpu_softmax.py`), not in the suite that
  `run_tests.py` runs. They should be promoted so the gate runs on every change.
- **seq768 not remeasured** after the GPU softmax port.
- **iter9 unfinished**: the next frontier (now that witness is ~6%) is the quad
  and encode phases at large SEQ — untouched.
- **Scale caveat**: all measured on one V100 at small/medium configs. The
  cost-model theory (`cost_calculator.py`) says these generalize, but nothing was
  verified at the notebook's 400B scale. The soundness argument (byte-identical
  proof) *does* hold at any scale — it's a determinism argument, not a
  scale-dependent one.
- All changes are **uncommitted** (no-git instruction) — they live in the working
  tree of `prover/core.py` and `prover/compute_fns.py`.

## Bottom line

The loop took the user's named lever ("multiple forward passes") and drove it to
a **3.64× prove speedup at seq1024**, entirely within the soundness gate (every
step produced a byte-identical, Rust-ACCEPTed proof). It found and fixed a real
bug in its own first implementation (the too-small cache cap), correctly
abandoned a lever that didn't pay (cache extension), and ported the true
bottleneck (CPU softmax) to GPU. The witness term — 75% of prove at seq1024 at
the start — is now ~6%, and the frontier has moved to the quad/encode phases.
