# Prover overhead relative to inference (back of the envelope)

How much more expensive is proving a transcript than computing it? This note
derives the ratio at a physics level: the dominating term in each regime is
justified from first principles; the smaller terms are carried through the
final numbers without derivation. Companion to `maverick-cost-model.md`, which
supplies the witness polynomials `W(S), #cids(S), Q(S)`; wall-clock anchors are
the measured runs of paper §9 and the cost identity of Appendix A.5.

**Takeaway numbers** (derived below; against fp8 flash-attention inference at
40% MFU on the same hardware, weight commitment amortized):

- **Headline: overhead ≈ 10⁵, within a factor of a few**, across both regimes
  and both machines (Spark measured, NVL72 floor).
- Expert-dominated regime (900 ≲ S ≲ 11k): **≈ 1×10⁵** (34 s of proving per
  token vs 0.34 ms of inference, measured).
- Attention-dominated regime (S ≳ 11k): **≈ 5×10⁵** on the Spark measured,
  **≈ 1×10⁵** on the NVL72 floor.
- Fixed cost: weight commitment, **5.4 h measured**, amortizing to minutes per
  proof over the ~100-proof refresh cycle.
- Floor under everything: the 4 witness passes, **≈ 4×10⁴**; a large dense
  model (no MoE route-hiding, bigger inner dims) would sit near **≈ 10³**.

## Definition and convention

```
overhead(S)  =  prover wall-clock  /  GPU-time to recompute the same transcript
                on the same hardware
```

Baseline convention (each toggle is a stated factor of ~2):

- **fp8** forward pass (Maverick deployments serve fp8; bf16 halves the overhead),
- **flash attention**: attention is compute-bound, the S×S matrix is never
  materialized (a naive-attention baseline would flatter the overhead ~4–8×),
- **40% MFU** (overhead scales linearly in the assumed utilization),
- **causal** FLOP counting (the prover commits the full saturated score matrix,
  inference computes the lower triangle; non-causal counting halves the ratio),
- the **fixed weight commitment is split out** and amortized (paper §6.4), as a
  serving operator would.

Spark (GB10) numbers: fp8 dense peak ≈ 250 TFLOPS → `c_flop = 1×10⁻¹⁴ s` at
40%. NVL72 aggregate: fp8 ≈ 324 PFLOPS → `c_flop = 7.7×10⁻¹⁸ s` at 40%.

## Master factorization

Marginal prover cost is proportional to the committed witness,
`T_prove ≈ c_slot · W(S)`, where `c_slot` absorbs encode, folds, hashing, and
the witness passes per slot. Inference is `T_inf ≈ c_flop · F(S)`. So

```
overhead(S)  =  [ W(S) / F(S) ]      ×      [ c_slot / c_flop ]
                slots per FLOP              machine factor
                (protocol property)         (hardware property)
```

- `c_slot` **measured all-in on the Spark: 58 ns** (14.3 h / 8.9×10¹¹ slots);
  the A.5 identity floor is ≈ 35 ns. Machine factor: **5.8×10⁶** measured,
  3.5×10⁶ floor.
- Both `W(S)` and `F(S)` are quadratics in `S`, so slots-per-FLOP is
  piecewise-constant — one value per regime of paper §A.4 — and the whole
  estimate is three coefficient comparisons.

**`c_slot` is an all-in rate, not a commitment rate.** It is total wall-clock
over witness slots, so constraint cost is inside it: the linear fold where the
constraint system is evaluated (transforms plus constraint-coefficient work),
the quadratic fold over the Hadamard products, and the per-constraint challenge
hashes, alongside the commit encode, column hashing, and the four witness
passes. By the instrumented bucket shares of A.5, the constraint-side work is
roughly 30–40% of proving time. The per-slot normalization assumes `Q ∝ W`;
that ratio shifts from ≈ 0.20 at S=1000 (where the rate was measured) to
19200/40320 ≈ 0.48 in the pure attention regime, so the Regime-2 overhead is
understated by ~10% — inside the error bars, and carried explicitly (the
`15 ns × Q` line) in the NVL72 floor below. Excluded from the overhead
entirely: verifier time (17.7 h for the demonstrated run), proof
storage/transmission, and the amortized weight commitment.

`F(S) ≈ 2·N_active·S + 2·d·n_layers·S²` with `N_active = 17×10⁹`, i.e.
`3.4×10¹⁰·S + 4.9×10⁵·S²` FLOP (causal attention).

## Regime 1 — linear, `900 ≲ S ≲ 11k`: the committed experts

Prover: `4.48×10⁸` slots/token (cost model `c₁`), of which 89% is the MoE
experts. The physics: a rescaled matmul commits **6 slots per output element**
(raw product, kept word, dropped word, range inverses — the ⓡ block), and the
route-hiding of §3.3 commits **all `E = 128` experts** though one fires:

```
6 slots/element × E(2·d_ff + d) elements/token/layer × n_moe
  = 6 × 128 × 21504 × 24  ≈  4.0×10⁸ slots/token
```

Inference computes only the active expert: `2 × 17×10⁹` FLOP/token.

```
slots/FLOP = 4.48×10⁸ / 3.4×10¹⁰ ≈ 0.013   →   overhead ≈ 0.013 × 5.8×10⁶ ≈ 8×10⁴
```

Measured anchor: the increment between the S=100 and S=1000 runs is
**34 s of proving per token**, against 0.34 ms of inference → **1.0×10⁵**.

## Regime 2 — quadratic, `S ≳ 11k`: attention

Per score cell (head × layer × position²), inference does `4·d_h = 512` FLOP
(`QKᵀ` plus `A·V`), halved by causality, and under flash attention writes
nothing to DRAM. The prover must commit the cell: **6 slots** from the scores
matmul plus **15 from softmax** (the bracketed exponential lookups, LogUp
inverses, saturating mux — softmax has no inner dimension to amortize over,
which is why it, not the matmul, dominates). That is the 840/layer × 48 =
40320·S² of the cost model.

```
slots/FLOP = 21 / 256 ≈ 0.082   →   overhead ≈ 0.082 × 5.8×10⁶ ≈ 5×10⁵  (floor ≈ 3×10⁵)
```

The ~5× worsening over Regime 1 is softmax's per-cell table machinery.

## Regime 0 — constant, `S ≲ 900`: the weights

`4×10¹¹` slots, one per parameter — **5.4 h measured** (the S=10 run).
Inference has no matching term (weights cost nothing beyond being read), so
this is not expressible as a ratio: it is a fixed cost, amortized to minutes
per proof over the ~100-proof refresh cycle of §6.4.

## The floor under everything: witness generation

Four streaming passes recompute the integer forward pass (paper §A.5). On the
Spark that is ≈ 1 h/pass at S=1000 — int64 on CUDA cores against fp8 on tensor
cores, times `E` for the hidden routing:

```
witness-only overhead ≈ 4 × 3600 s / 0.34 s ≈ 4×10⁴
```

Already ~half the marginal cost at S=1000. This term is the asymptote: as
matrices grow, commit cost (∝ outputs) vanishes relative to arithmetic
(∝ MACs), and the overhead limits to *doing the model's math in integer, times
the pass count*. Levers are protocol/engineering, not size: claim-streaming
(4 passes → ~1, §10) and integer-tensor-core witness matmuls via limb
decomposition (int-vs-fp gap ~400× → ~10×).

## Cross-machine: why the number survives the move to a cluster

`c_slot` rides memory bandwidth (with a hash term scaling only ~170× against
bandwidth's ~2580×); `c_flop` rides tensor cores. The overhead therefore
transforms by the machines' bytes-per-FLOP ratio: a B200 has ≈ 3.6 GB/s per
bf16 TFLOPS against the GB10's measured 1.8, a 2× prover-favoring shift,
partially clawed back by the hash term growing from ~5% to ~25% of the floor.
At S = 10⁶ on an NVL72 (all floor, no implementation gap):

```
prover:    bandwidth 3.0×10⁵ s + hash 1.2×10⁵ s + witness 0.5×10⁵ s ≈ 5 days
inference: 5.3×10¹⁷ FLOP / 1.3×10¹⁷ FLOPS ≈ 4 s
overhead ≈ 1.1×10⁵
```

Machine factor ≈ 1.5×10⁶ — within 4× of the Spark's, on hardware 2,500× apart.

## Summary

| term | dominating physics | slots/FLOP | overhead (Spark, measured) |
|---|---|---|---|
| weights (`S ≲ 900`) | 1 slot per parameter | — (fixed cost) | 5.4 h once; ~min/proof amortized |
| experts (`900–11k`) | 6 slots/element × all 128 experts vs 1 active | 0.013 | ≈ 1×10⁵ |
| attention (`S ≳ 11k`) | 21 slots/cell (15 = softmax) vs 4·d_h FLOP | 0.082 | ≈ 5×10⁵ (NVL72 floor ≈ 1×10⁵) |
| witness floor | 4 passes, int64 vs fp8 tensor | — | ≈ 4×10⁴ |

**Headline: ≈ 10⁵, within a factor of a few, across both regimes and both
machines** (against fp8 flash-attention inference at 40% MFU, weights
amortized). The overhead is not intrinsic to the argument but to what must be
committed: for matmuls slots/FLOP falls as `3/k` with the inner dimension, so a
large dense model would sit near the witness floor of ~10³; the demonstrated
system is held at ~10⁵ by MoE route-hiding (~×10) and per-cell softmax at long
context (~×5). One coupling: growing `k` costs ~1 bit of accumulator headroom
per doubling (§3.6), an accuracy trade, not a wall.

## Provenance

Measured: 14.3 h / 5.8 h / 5.4 h runs, 58 ns/slot, 34 s/token, 840
slots/block/S². From the paper's cost identity (A.5): the 35 ns floor, the
NVL72 scaling ratios (2580× bandwidth, 170× hash), ~1 h/witness pass. Assumed:
peak FLOPS from spec sheets, 40% MFU, fp8 serving, and the GB10↔B200 scalar
throughput ratio (~2.4×) the paper also uses. The int64-vs-fp8 gap (~400×) is
back-calculated from the measured witness pass and is the softest number here.
