# Witness scaling vs context length — measured (single Maverick block)

**Date:** 2026-06-23. **Hardware:** `spark-c191` (NVIDIA GB10, sm_121, 121 GiB
unified, CUDA 13.0). **Config:** Llama-4-Maverick UD-Q4_K_XL, layer 1 (MoE, RoPE),
`d=5120, n_q=40, d_h=128, d_ff_exp=8192`; Ligero `ELL=8192, K_DEG=16384,
N_LIG=65536, T_QUERIES=4`, quant scale `2^12`. **Harness:**
[`sweep_block.py`](sweep_block.py) (reuses `demo_maverick_block.build`; adds
`../pipeline` to `sys.path`, the standard `analysis/` convention).

## Why

`design-feasibility.md` §4.3/§4.6 models the per-prefill witness as
`W(S) = A·S + B·S²`, with the quadratic term coming from attention (the `S×S`
score/probability matrices). That `B` coefficient drives every long-context
projection, but it was an architectural estimate (`B = 2·n_q·L`, i.e. scores +
probabilities only), and the softmax LogUp accounting in §4.6 was flagged as a
first pass. This measures `A` and `B` directly by sweeping the sequence length on
a single block and fitting the committed-witness-slot count, then checks that
prove time tracks witness size.

## TL;DR

- **The quadratic coefficient is ~10× larger than the design doc.** Measured
  `B = 840` committed slots per block per `S²`, vs the doc's `2·n_q = 80`. The fit
  is exact (a clean `c + a·S + b·S²`, max relative residual 0.000% across
  `S = 512…32768`).
- **It is dominated by the softmax *proof* witness, not the score activations.**
  Per-claim-type attribution: `~600` SoftmaxClaim + `~160` MatmulClaim + `~80`
  RmsNorm. The doc counted only the scores+probs activations (`80`); the bulk is
  the per-entry softmax witness — `exp` value, score range-check word
  decompositions, and LogUp inverses.
- **`B` is expert-count-independent.** Identical `B = 840` at `E=8` and `E=128`
  (attention is the same regardless of expert count) — the consistency check we
  wanted.
- **Prove time tracks witness size at ~40 ns/slot** (marginal), consistent across
  the sweep and matching the full 48-layer `mav847` run's 35.6 ns/slot. Since the
  witness is quadratic in `S`, prove time is quadratic in `S`.
- **Memory grows with `S²` and binds early.** Peak GPU 7.7 → 12.4 → 32.5 GB at
  `S = 512/1024/2048`; `S=4096` OOM'd on a *single* block (>121 GB unified) — the
  softmax `S²` witness is materialized, not streamed.

## Method

`sweep_block.py` builds one Maverick block (`demo_maverick_block.build`) at each
sequence length, then walks `tape.claims` — including into LogUp `Table` objects
to capture phase-2 aux — to tally distinct committed `Variable` slots, deduped by
identity and attributed to the first claim type that references them. Witness size
is read from the built tape (no prove needed; expert weights stay lazy). Prove
runs are a subset, timed, no verify (the goal is prover cost). All fits are
`numpy.polyfit(seq, slots, 2)`.

Caveat on the per-type split: phase-2 range/LogUp `z`'s live on **shared** range
tables, so they are attributed to whichever claim first touched the table. The
**total** `B = 840` and the phase-1/phase-2 split are exact; the
Softmax/Matmul/RmsNorm split is approximate (the `RmsNorm` `S²` entry is almost
certainly mis-attributed softmax range `z`'s — RmsNorm is per-token and has no
genuine `S²` work).

## Results — witness size

Fitted `W(S) = c + a·S + b·S²`, max relative residual **0.000%** (exact quadratic):

| sweep | `b` (per `S²`) | `a` (per `S`) | `c` (const) |
|---|---|---|---|
| E=8, total | **840** | 2,088,700 | 1.41×10⁹ |
| E=8, phase-1 (committed) | 560 | 1,447,299 | 1.32×10⁹ |
| E=8, phase-2 (aux/LogUp) | 280 | 641,401 | 8.5×10⁷ |
| E=128, total | **840** (identical) | 17,574,940 | 1.65×10¹⁰ |

`b` is unchanged between `E=8` and `E=128`; only the linear floor `a` scales with
experts. Per-claim-type `b` (E=8): SoftmaxClaim `600`, MatmulClaim `160`,
RmsNorm `80` (see caveat), all others `0` (linear in `S`).

Raw E=8 witness slots (total / m_total rows / phase-1 / phase-2):

| seq | total slots | m_total | phase-1 | phase-2 |
|---|---|---|---|---|
| 512 | 2,697,220,616 | 329,360 | 2,210,010,884 | 487,209,732 |
| 1024 | 4,427,237,896 | 540,520 | 3,391,429,892 | 1,035,808,004 |
| 2048 | 9,208,478,216 | 1,124,160 | 6,635,071,748 | 2,573,406,468 |
| 4096 | 24,055,781,896 | 2,936,560 | 16,645,570,820 | 7,410,211,076 |
| 8192 | 74,889,681,416 | 9,141,840 | 50,759,430,404 | 24,130,251,012 |
| 16384 | 261,114,649,096 | 31,874,380 | 175,358,595,332 | 85,756,053,764 |
| 32768 | 971,793,259,016 | 118,627,140 | 650,042,708,228 | 321,750,550,788 |

## Results — prove time and memory

E=8, prove only (no verify):

| seq | witness slots | prove (s) | ns/slot | peak GPU |
|---|---|---|---|---|
| 512 | 2.70×10⁹ | 89.4 | 33.1 | 7.7 GB |
| 1024 | 4.43×10⁹ | 157.7 | 35.6 | 12.4 GB |
| 2048 | 9.21×10⁹ | 348.2 | 37.8 | 32.5 GB |
| 4096 | 2.41×10¹⁰ | — (OOM) | — | >121 GB |

Prove time is linear in witness slots — marginal rate **39.8 ns/slot**, consistent
across both intervals — so it inherits the `S²` scaling (prove ratios 1.76×, 2.21×
track slot ratios 1.64×, 2.08×). The rate matches the full 48-layer `mav847` run
(35.6 ns/slot; `analysis/full-model-v1-design.md`). `S=4096` was killed at the
softmax op with no Python traceback (kernel OOM-killer; GB10 unified memory is
shared CPU/GPU) — a single block cannot prove beyond ~3–4K context on this box
without streaming the softmax `S²` witness.

## Implication for the cost model

`design-feasibility.md` §4.3 gives `B = 2·n_q·L` (scores + probs activations). The
measured per-block quadratic coefficient is **`840`, ~10.5× the `80` that formula
predicts**, because it omits the per-entry softmax *proof* witness. The §4.6 LogUp
accounting narrows but does not close the gap. **Every long-context projection in
that document that uses the `2·n_q` quadratic term is a ~10× underestimate** and
should be re-derived against `B ≈ 840` per layer (softmax-dominated).

### Corrected full-model projection

Extrapolating per-layer `B = 840` across all 48 layers (attention is identical per
layer, so this is sound) gives `B_full = 840 × 48 = 40,320` slots per `S²`. For
**dense** attention at `S = 1,000,000`:

| Quantity | Doc-`B` (`2·n_q·L`=3,840) | **Measured-`B` (40,320)** |
|---|---|---|
| Witness | 46 PB | **323 PB** |
| Spark prove @ ~40 ns/slot | ~6.5 yr | **~50 years** |
| NVL72 memory-bound floor (41% mem-bound, ÷2580 BW ratio) | ~9 h | **~3 days** (×2–2.5 overhead → ~6–7 days) |

The single-block (E=128) crossover where `B·S²` overtakes `a·S` is `≈ 21,000`; for
the full model the crossover is `≈ 11,000` context, so 1M is ~90× past it —
deeply quadratic-dominated.

## Caveats

- **Dense attention is the load-bearing assumption.** The `S²` term *is* the
  witness at long context. A sliding-window / sparse 1M model replaces `B·S²` with
  `~B·S·w` and collapses all of the above by ~100×. The real 1M number depends on
  the attention pattern, not on `B`.
- **Per-layer × 48 is an extrapolation** from one measured block (the attention
  `S²` is identical per layer; the linear/weight terms differ between dense and MoE
  layers but do not affect `B`).
- **Resident regime only.** These runs fit in 121 GB (peak ≤ 32.5 GB at S=2048).
  The 1M projection is far past where the witness must spill; the per-slot rate
  there is not validated and the memory wall (above) binds first.
- **Small-seq absolute times are overhead-inflated** (fixed per-claim/build cost);
  use the fitted coefficients, not the raw small-run wall times.

## Reproduction

On `spark-c191`:

```
cd ~/infproof/analysis
# witness size only (fast, ~6 s/point), E-independence check:
PATH=~/venv-hf/bin:$PATH LIGERO_T_QUERIES=4 python sweep_block.py \
    --from-gguf ~/maverick-gguf/UD-Q4_K_XL --experts 8 \
    --seqs 512,1024,2048,4096,8192,16384,32768 --out /tmp/ws_e8.json
PATH=~/venv-hf/bin:$PATH LIGERO_T_QUERIES=4 python sweep_block.py \
    --from-gguf ~/maverick-gguf/UD-Q4_K_XL --experts 128 \
    --seqs 512,1024,2048,4096,8192 --out /tmp/ws_e128.json
# prove time (slower; S>=4096 OOMs on GB10):
PATH=~/venv-hf/bin:$PATH LIGERO_T_QUERIES=4 python sweep_block.py \
    --from-gguf ~/maverick-gguf/UD-Q4_K_XL --experts 8 \
    --seqs 512,1024,2048 --prove --out /tmp/pt_e8.json
```

Fit with `numpy.polyfit(seq, total_slots, 2)`; the `b` coefficient is the
per-block `S²` term.
