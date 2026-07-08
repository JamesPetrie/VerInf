# Maverick Ligero cost model (analytical)

Purely analytical: the three cost drivers — **W** (committed witness slots),
**#cids** (distinct linear constraint ids), **Q** (quadratic products) — for each
claim type, as polynomials `c₀ + c₁·S + c₂·S²` in the context length `S`, for the
fixed Llama-4-Maverick config. Machine throughput (turning these into wall-clock
via `T ≈ A·W + B·#cids + C·Q`) is a separate step.

Reproduced by `maverick_cost_model.py` (pure Python). Grounded in
`claims.py` / `routing_claim.py` / `CLAIM_SPECS.md` (per-claim counts) and
`demo_maverick_full.py` (model structure).

**Validation:** the model's `W` S²-coefficient is **40320 = 840/layer × 48**,
matching the measured fit in `witness-scaling-measurement.md` (840 slots/block/S²).
The 840 decomposes as **240** (scores `QKᵀ` matmul, via its output-rescale block)
+ **600** (the saturating-causal softmax witness) — the old `2·n_q = 80` estimate
missed both the rescale and the softmax witness, hence its ~10× miss.

## Config

`d=5120`, `d_ff_dense=16384`, `d_ff_exp=8192`, `E=128` experts (top-1, **all
committed**), `n_q=40` heads × `head_dim=128`, `V=202048`. 48 layers = **24
dense** (even index, all RoPE) + **24 MoE** (odd; 12 RoPE + 12 NoPE). Every
`matmul`/`hadamard`/`rope`/`rmsnorm` carries the **output-rescale block**
(`rescale_bits=12`, `output_width=26`); softmax runs **saturate+causal**; silu has
no rescale. The big MoE expert sums use `freivalds_combine` (`~4ET`), not
`masked_combine` (`~2ETF`).

## Per-claim formulas (active-in-Maverick forms)

For shapes: matmul `(m,k,n,H)`; rmsnorm/softmax `(B,M or d)`; elementwise `(L)`;
routing/combine `(T,E,F)`. `ⓡ` = the output-rescale delta (present on every
Maverick matmul/hadamard/rope/rmsnorm).

| claim | W | #cids | Q |
|---|---|---|---|
| **matmul** `(m,k,n,H)` ⓡ | `6·mHn + 3k` | `2k + H + 2·mHn` | `k + 2·mHn` |
| **rmsnorm** `(B,d)` ⓡ, K=4 | `7Bd + 26B` | `7B + 2Bd` | `3Bd + 13B` |
| **softmax** `(B,M)` sat+causal | `15BM + 9B` | `½B(M+1) + 4BM + 5B` | `8BM + 3B` |
| **silu** `(L)` | `23L` | `7L` | `12L` |
| **hadamard** `(L)` ⓡ | `6L` | `2L` | `3L` |
| **rope** `(L)` ⓡ | `6L` | `3L` | `2L` |
| **add** `(L)` | `L` | `L` | `0` |
| **routing** `(T,E)`, n_words=3 | `10TE + 2T` | `3TE + 3T` | `5TE` |
| **masked_combine** `(T,E,F)` | `2ETF + TF` | `ETF + TF` | `ETF` |
| **freivalds_combine** `(T,E,F)` | `TF + 4ET + T` | `3ET + 2T` | `ET` |

`mHn = m·H·n`. The matmul `ⓡ` term `2·mHn` (cids) / `2·mHn` (Q) is the rescale,
which is what makes a rescaled matmul cost `O(m·n)` cids instead of `O(k)` — the
branch that decides whether "weights are free." Attention specializes these:
scores `QKᵀ` is `matmul(S, 5120, S, H=40)` → output `40·S²`; softmax is
`(B=40S, M=S)`; scores@V is `matmul(S, 40S, 128, H=40)` → output `5120·S`.

## Result: per claim type, summed over the model (`c₀ + c₁·S + c₂·S²`)

| claim type | W: c₁ (per S) | W: c₂ (per S²) | #cids: c₁ | #cids: c₂ | Q: c₁ | Q: c₂ |
|---|---|---|---|---|---|---|
| matmul | 4.12×10⁸ | 11520 | 1.37×10⁸ | 3840 | 1.37×10⁸ | 3840 |
| softmax | 17280 | **28800** | 10560 | **8640** | 5760 | **15360** |
| silu | 1.81×10⁷ | — | 5.51×10⁶ | — | 9.44×10⁶ | — |
| hadamard | 8.40×10⁶ | — | 2.80×10⁶ | — | 4.20×10⁶ | — |
| rmsnorm | 3.44×10⁶ | — | 9.84×10⁵ | — | 1.48×10⁶ | — |
| rope | 2.21×10⁶ | — | 1.11×10⁶ | — | 7.37×10⁵ | — |
| LM head | 1.21×10⁶ | — | 4.04×10⁵ | — | 4.04×10⁵ | — |
| add | 6.14×10⁵ | — | 6.14×10⁵ | — | — | — |
| freivalds_combine | 5.53×10⁵ | — | 2.78×10⁴ | — | 9216 | — |
| embed_lookup | 4.92×10⁵ | — | 4.92×10⁵ | — | — | — |
| masked_combine | 3.69×10⁵ | — | 2.46×10⁵ | — | 1.23×10⁵ | — |
| routing | 3.08×10⁴ | — | 9288 | — | 15360 | — |

(`c₀` is dominated by the **weights floor** for W, and by the matmul Freivalds aux
`3k`/`2k+H` for #cids/Q — both S-independent; full numbers in the script output.)

## Totals

```
W(S)    ≈  4.00×10¹¹  +  4.48×10⁸ · S  +  40320 · S²
#cids(S) ≈  1.19×10⁸   +  1.50×10⁸ · S  +  12480 · S²
Q(S)    ≈  5.93×10⁷   +  1.54×10⁸ · S  +  19200 · S²
```

- **`matmul` dominates the `S` (linear) term** of all three — the QKV/O projections,
  the 128 expert matmuls per MoE layer, the FFN, and the LM head.
- **`softmax` dominates the `S²` term** of all three (with the scores matmul) — this
  is the attention quadratic.

## Regime structure (why you need all three)

Each quantity has a different profile, so a single per-slot constant can't capture
the cost:

| quantity | S-independent floor | linear catches floor | quadratic catches linear |
|---|---|---|---|
| **W** | **4.0×10¹¹ (weights)** | `S ≈ 900` | `S ≈ 11100` |
| **#cids** | ~1.2×10⁸ (small) | `S ≈ 1` | `S ≈ 12000` |
| **Q** | ~5.9×10⁷ (small) | `S ≈ 0.4` | `S ≈ 8000` |

- **W** is **weight-dominated below ~900 tokens** (the 400 B committed params,
  inactive experts included), linear (activations + experts) from ~900 to ~11 000,
  and **attention-quadratic above ~11 000**. The Maverick run (`S=1093`) sits right
  at the weights↔linear crossover; **1M tokens is ~90× into the quadratic regime**.
- **#cids** and **Q** have *no* weight floor (Freivalds compresses each weight
  matmul to `O(k)` cids), so they're linear from the start and go quadratic at
  ~12 000 / ~8 000 tokens respectively.

So the three crossovers differ (`W`~11k, `#cids`~12k, `Q`~8k) and only `W` carries
the weight floor — which is exactly why the cost model is these three numbers, not
one per-slot constant.

## Intuitive approximation — two claims carry it

You don't need every claim. Only two set the leading behavior, and for the
top-order term the reduction is *exact*, not a hand-wave.

**The S² term is attention, full stop.** Softmax and the scores `QKᵀ` matmul are
the *only* claims with an S² term — every other claim is ≤ linear in S. So the
quadratic coefficient is exactly attention, with a clean closed form (per head,
per layer):

| quantity | S² coefficient | value |
|---|---|---|
| W | `21 · n_q · n_layers` | 40320 (= 15 softmax + 6 scores-rescale, ×40×48) |
| #cids | `6.5 · n_q · n_layers` | 12480 |
| Q | `10 · n_q · n_layers` | 19200 |

**The S (linear) term is the committed experts.** ~89% of the W linear
coefficient is the 128 expert matmuls — committed even though only one runs (the
all-experts-commit privacy requirement) — so **MoE width, not context, sets the
linear cost**: `W_linear ≈ 6 · E · (2·d_ff_exp + d) · n_moe · S ≈ 4×10⁸·S`.

**The W constant is the weights** — the 400 B committed params. (`#cids`/`Q` have
no floor; Freivalds compresses each weight matmul to `O(k)` cids.)

### Where these coefficients come from (exact)

**Attention.** Per layer, attention forms an `S×S` score matrix for each of
`n_q` heads — so **`n_q·S²` cells**. Exactly two claims touch every cell:

- the **scores `QKᵀ` matmul** commits **6** values per score entry — the score
  `C`, its raw product `C_full`, the rescale words `C_low`/`C_shifted`, and 2
  range-check inverses → `6·n_q·S²`;
- **softmax** commits **15** values per cell — the two exp-table lookups, their
  LogUp inverses, the paired-lookup combinations, and the saturating-mux gadget
  → `15·n_q·S²`.

So `W` per layer `= (6 + 15)·n_q·S² = 21·n_q·S²`, hence `21·n_q·n_layers =
21·40·48 = 40320`. The same two claims give the `#cids` and `Q` coefficients —
counting *constraints* and *quadratic products* per cell instead of witness
slots:

| per cell | scores matmul | softmax | total | `× n_q·n_layers` |
|---|---|---|---|---|
| W | 6 | 15 | 21 | 40320 |
| #cids | 2 (rescale) | 4½ (2 lookups + 2 mux + ½ causal) | 6½ | 12480 |
| Q | 2 (range) | 8 (2 LogUp + 6 mux) | 10 | 19200 |

(The ½ on softmax `#cids` is the causal mask — only the lower triangle of each
`S×S` block is constrained, so that one constraint family is `½·n_q·S²` not
`n_q·S²`.)

**Committed experts.** Each MoE layer runs all `E=128` experts — committed even
though only one fires. Each expert is 3 matmuls on the shared `S`-token input,
with output sizes `S·d_ff_exp` (gate), `S·d_ff_exp` (up), `S·d` (down); each
output carries the 6× rescale block. So per expert `6·S·(2·d_ff_exp + d)`, and
over `E` experts × `n_moe = 24` layers:

```
6 · E · (2·d_ff_exp + d) · n_moe  =  6 · 128 · 21504 · 24  =  3.96×10⁸  per token
```

— 89% of the linear coefficient (the remaining 11% is the attention QKVO
projections, the silu/hadamard nonlinearities, the RMS norms, and the LM head).

So the ballpark model, each term tied to one cause:

```
W(S)     ≈   4×10¹¹    +   4×10⁸·S   +   4×10⁴·S²
              weights       experts       attention
#cids(S) ≈                 1.5×10⁸·S  +   1.2×10⁴·S²
Q(S)     ≈                 1.5×10⁸·S  +   1.9×10⁴·S²
```

**Which term wins** (W):

| S (context) | W | weights | experts | attention |
|---|---|---|---|---|
| 1 093 (demonstrated run) | 9.4×10¹¹ | 43% | 46% | 5% |
| 100 000 | 4.5×10¹⁴ | 0% | 9% | 90% |
| 1 000 000 | 4.1×10¹⁶ | 0% | 1% | 99% |

Two sentences capture it:
- **At the demonstrated 1093-token run, the proof is model-dominated** — weights
  + committed experts are ~90% of the witness, attention only ~5%.
- **At frontier context (~1M), the proof is ~99% attention** — you can size a
  machine off `W ≈ 21·n_q·n_layers·S²` and ignore weights and experts entirely.

Small context → you pay for the *model* (weights + all experts); large context →
you pay for *attention* (softmax²). The exact totals above and this two-claim
ballpark agree to the percentages in the table.

## Not modeled here (flagged)

- **Weights floor** is set to `R_W = 400×10⁹` (the param count, all experts
  committed); a few-% refinement would compute it from the layer shapes.
- **Embedding hidden-prompt one-hot** (proving the hidden input tokens are valid
  one-hots over `V`) is an `O(T_hidden · V)` *fixed* cost, not ∝ S; it's omitted
  from the per-S model (the LM head and the token-select *are* included).
- **UI / MaxClaim** machinery (the unexplained-information bound on the logits) is
  not a per-layer arithmetic claim and is not counted.

## Machine throughput — from `(W, #cids, Q)` to wall-clock

Built ground-up from **two measured machine primitives**, not from an
all-together prove rate (the ~40 ns/slot aggregate is held aside for a separate
end-to-end comparison, below). `T ≈ A·W + B·#cids + C·Q`, where each constant is
`(protocol NTT/hash count) × (primitive rate)`.

**The two primitives (measured, GB10):**

- **NTT: ~0.42 ns/element** at length 2¹⁵ (0.33 at 2¹⁴), flat across batch,
  **memory-bandwidth-bound** at 223 GB/s — and a faster kernel was *measured* to
  have no headroom; the only lever is fewer NTTs (`prover-optimization-investigation.md` §3).
- **BLAKE3 challenge: ~0.6 ns/cid** (2.0 Gcompress/s), **compute-bound** — the
  input `seed‖cid‖label` is built in registers, so it's ALU, not bandwidth
  (`spark-microbench-results.md`; our fold bench confirms: per-cid hashing ran at
  11–22 GB/s ≪ 223, collapsing to a 135 GB/s floor only once hashing was removed).

**The three constants:**

| constant | NTT / hash work (per unit) | GB10 | bound by |
|---|---|---|---|
| `A` (per witness slot) | encode (`iNTT_K`=16384 + `NTT_N`=65536) + linear-fold `poly_mul` (~3× `2·K_DEG`=32768), per ELL=8192 row | **≈ 9 ns/slot** (encode ~4 + linear fold ~5) | NTT → **bandwidth** |
| `B` (per linear cid) | 1 `BLAKE3` compress | **≈ 0.6 ns/cid** | hash → **compute** |
| `C` (per quad product) | ~2 `poly_mul`s of `2·K_DEG` + operand re-encode, per ELL-packed quad descriptor | **≈ 15 ns/product** | NTT → **bandwidth** |

All NTTs are one family at three lengths (commit `iNTT_K=16384` / `NTT_N=65536`;
both folds' `poly_mul` `2·K_DEG=32768`), so **`A` and `C` are bandwidth-bound and
`B` is compute-bound** — which is what makes them scale to bigger machines by
*different* ratios.

**Scaling across machines.** `A`/`C` (NTT) ride **aggregate memory bandwidth**:
an NVL72 has ~2580× the GB10's bandwidth, so its NTT-bound floor is the Spark time
**÷2580** (× 2–2.5 real-world overhead; `witness-scaling-measurement.md`,
`ARCHITECTURE.md §8`). `B` rides the **compute** ratio instead, but `B·#cids` is
a rounding error vs the NTT terms at frontier, so bandwidth governs the total.

**Worked number (1M dense context), ground-up.** All attention, so
`W ≈ 40320·S²`, `Q ≈ 19200·S²`:

```
A·W ≈ 9 ns  · 40320 · 10¹²  ≈ 3.6×10⁸ s ≈ 11 yr
C·Q ≈ 15 ns · 19200 · 10¹²  ≈ 2.9×10⁸ s ≈  9 yr      (B·#cids ≈ 0.3 yr — negligible)
total ≈ ~20 years on one Spark  →  ÷2580 ≈ ~3 days on an NVL72 (×overhead → ~a week)
```

**Separate end-to-end check (not a model input).** The all-together measured
prove rate is **~40 ns/slot** (`witness-scaling-measurement.md`), which on `W`
alone gives ~50 yr at 1M — ~2.5× the ground-up NTT-bound floor above. The gap is
the current prover's `expand + sort` and Python orchestration overhead (~26%+ of
prove), which the constraint-fold reorg (`qlin-fold-reorg-plan.md`) removes — so
the ground-up model is effectively the *post-reorg, NTT-bound* estimate, and
comparing it against the 40-ns/slot aggregate is the validation step.

Two caveats: (1) the exact NTT-count-per-unit is the one protocol factor still to
pin precisely — the `~9` / `~15` ns figures use the *un-fused* `poly_mul` count,
so the inverse-NTT fuse (which removes ~1 NTT per `poly_mul`) makes them an upper
bound; (2) dense attention is load-bearing — sparse/sliding-window replaces
`B·S²` with `~B·S·w`, collapsing the frontier numbers ~100×.
