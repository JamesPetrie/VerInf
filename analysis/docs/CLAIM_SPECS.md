# Non-arithmetic claim specifications

Reference for the non-arithmetic claims currently implemented in this
directory: **RmsNorm**, **Softmax** (§2.13 two-table bracket), **SiLU**, and
**MoE routing** (`RoutingClaim` + `MaskedCombineClaim`, top-1, after SiLU —
tracked by `../analysis/maverick-moe-implementation-plan.md`).

For each claim:

- **I/O** lists the public input(s) and output(s).
- **Phase-1 vars** are committed before the verifier samples challenges.
- **Phase-2 vars** are committed after; their values depend on the
  per-claim challenges (typically `α, β` from the table settlements).
- **Tables** lists every LogUp table the claim uses; the per-table
  `TableSettlement` (auto-synthesized by `core.prove` / `core.verify`)
  samples `α`, optionally `β`, and emits the cross-claim sum identity.
- **Constraints** lists every linear (CSR row) and quadratic
  (pairwise-product) family the claim emits.
- **Intuition** is the one-paragraph reading.

Italicized rows are the optional rescale block (active only when the
config's `s_in > native_scale`); see [the rescale block](#cross-cutting-the-rescale-block) at
the bottom.

---

## RmsNorm

**Function.** `output[b,i] = x[b,i] / √(mean(x[b]²) + ε)` for `b ∈ [0, B)`,
`i ∈ [0, d)`. Integer-quantized at scale `s` throughout.

**Config.** `B, d, s, eps_int, slack_n_chunks=K, slack_chunk_width=w (=16),
s_in (optional, ≥ s)`. Derived: `magic = d·s⁴`, `rescale_bits r = log2(s_in/s)`.

**I/O.**

- Input: `x` (public, length `B·d`) at scale `s_in`. When `s_in == s`,
  this is also the internal `x`; otherwise it's `x_in` and the internal
  `x` is derived.
- Output: `output` (length `B·d`), `output[b·d+i] = x_internal[b·d+i] · y[b]`.

**Phase-1 vars** (committed before challenges):

| var | length | what it is |
|---|---|---|
| `x` | B·d | internal x (at scale s); equals public input when no rescale |
| `output` | B·d | x ⊙ broadcast(y) |
| `X_sq` | B·d | x ⊙ x |
| `S` | B | Σᵢ X_sq[b·d+i] |
| `S_total` | B | S + d·ε |
| `y` | B | rsqrt scalar (smallest int with y²·S_total ≥ magic) |
| `y_m1` | B | y − 1 |
| `q1, q2` | B each | y², (y−1)² |
| `s_lo` | B | q1·S_total − magic (lower-bracket slack) |
| `s_hi` | B | magic − 1 − q2·S_total (upper-bracket slack) |
| `s_lo_chunks[n], s_hi_chunks[n]` | B each (K of each) | word decomp of slacks |
| *rescale (r > 0):* `x_in, x_low` | B·d each | public input, low word |

**Phase-2 vars** (committed after challenges):

| var | length | what it is |
|---|---|---|
| `u, p` | B each | Freivalds row aggs: `u = Σ ρᵢ·x[b·d+i]`, `p = Σ ρᵢ·output[b·d+i]` |
| `z_lo_chunks[n], z_hi_chunks[n]` | B each | 1/(α_slack − chunk) |
| *rescale:* `z_x_low, z_x` | B·d each | 1/(α_rescale − x_low), 1/(α_slack − x) |

**Tables.** `range_slack` (2^w = 2^16 entries, shared across slack chunks
and the rescale's loose check); `range_rescale` (2^r entries, when
rescale active).

**Constraints.**

Linear (per row unless noted):

- `S_total = S + d·ε`, `y_m1 = y − 1`
- `S = Σᵢ X_sq[b·d+i]` (row-sum, d nnz)
- `u = Σᵢ ρᵢ · x[b·d+i]`, `p = Σᵢ ρᵢ · output[b·d+i]` (Freivalds)
- `s_lo = Σₙ (1<<(n·w))·s_lo_chunks[n]`, same for `s_hi`
- *rescale (per cell):* `x_in = (1<<r)·x + x_low`

Quadratic:

- `X_sq = x ⊙ x` (per cell)
- `q1 = y², q2 = y_m1²` (per row)
- **Lower bracket:** `q1·S_total − s_lo = magic` (per row)
- **Upper bracket:** `q2·S_total + s_hi = magic − 1` (per row)
- **Freivalds check:** `y · u = p` (per row)
- Range LogUps: `(α_slack − chunk)·z = 1` for every slack chunk
- *rescale:* `(α_rescale − x_low)·z_x_low = 1`, `(α_slack − x)·z_x = 1`

**Intuition.** Algebraic-rsqrt bracket pins `y` to `⌈√(magic / S_total)⌉`
without an rsqrt lookup table. Slacks accommodate the integer width
and are word-decomposed so we don't need a giant range table.
Freivalds-folded broadcast (`y · u = p`) avoids committing `B·d` cells
of `y_broadcast`.

---

## Softmax (§2.13 two-table bracket)

**Function.** `y_A[b,i] ≈ exp(x[b,i]/s_c) / Σⱼ exp(x[b,j]/s_c)`,
output scaled by `s_y`.

**Config.** `B, M, s_x, s_c (=s_x), s_y, delta=1, Z_max, s_in (optional,
≥ s_x)`. Derived: `r = log2(s_in/s_x)`.

**I/O.**

- Input: `x` (public, length `B·M`) at scale `s_in`. May be signed.
- Output: `y_A` (length `B·M`) at scale `s_y`.

**Phase-1 vars.**

| var | length | what it is |
|---|---|---|
| `x` | B·M | internal x at scale s_x |
| `y_A` | B·M | `T_A[z]` — softmax output |
| `c2` | B | per-row LSE candidate |
| `z` | B·M | `c2[b] − x[b·M+i]` |
| `y_B` | B·M | `T_B[z] = T_A[z−δ]` (δ-shifted) |
| `s1, s2` | B each | Σ y_A, Σ y_B |
| `r_lo` | B | `s_y − s1` (≥ 0 ⇒ s1 ≤ s_y) |
| `r_hi` | B | `s2 − (s_y + 1)` (≥ 0 ⇒ s2 ≥ s_y + 1) |
| *rescale:* `x_in, x_low` | B·M each | |

**Phase-2 vars.**

| var | length | what it is |
|---|---|---|
| `pt_u_A, pt_u_B` | B·M each | `z + β·y_{A,B}` (paired-tlookup u) |
| `pt_z_A, pt_z_B` | B·M each | `1/(α_{A,B} − pt_u_{A,B})` |
| `z_c2, z_r_lo, z_r_hi` | B each | `1/(α_aux − {c2, r_lo, r_hi})` |
| *rescale:* `z_x_low, z_x` | B·M each | |

**Tables.**

- `exp_A`: paired `(k, T_A[k])` where `T_A[k] = round(exp(−k/s_c)·s_y)`,
  `Z_max` entries.
- `exp_B`: paired `(k, T_B[k])` where `T_B[k] = round(exp((δ−k)/s_c)·s_y)`.
  **Computed from the same `round(exp(·)·s_y)` expression →
  `T_B[k] == T_A[k−δ]` bit-identically.** This is the soundness-critical
  identity that makes `s2(c2) == s1(c2−δ)` as integer sums.
- `range_aux`: 2^16 entries, shared.
- `range_rescale` (when r > 0).

**Constraints.**

Linear (per cell unless noted):

- `z = c2[i//M] − x` (stride-M access to c2)
- `pt_u_A = z + β_A·y_A`, `pt_u_B = z + β_B·y_B`
- `s1 = Σᵢ y_A[b·M+i]`, `s2 = Σᵢ y_B[b·M+i]` (row sums, per row)
- **Tight upper bracket:** `s1 + r_lo = s_y` (per row, no slack)
- **Tight lower bracket:** `r_hi − s2 = −(s_y + 1)` (per row, no slack)
- *rescale:* `x_in = (1<<r)·x + x_low`

Quadratic:

- **Paired LogUp on exp_A:** `(α_A − pt_u_A)·pt_z_A = 1` (per cell)
- **Paired LogUp on exp_B:** `(α_B − pt_u_B)·pt_z_B = 1` (per cell)
- Range LogUps: `(α_aux − {c2,r_lo,r_hi})·z_{...} = 1` (per row)
- *rescale:* `(α_rescale − x_low)·z_x_low = 1`, `(α_aux − x)·z_x = 1`

**Intuition.** Integer `s1(c2) = Σ T_A[c2 − xᵢ]` is monotone non-increasing
in `c2`. The bracket `s1(c2) ≤ s_y AND s2(c2) ≥ s_y + 1` (recalling
`s2(c2) = s1(c2−1)` by the table identity) pins `c2` to the unique
integer where `s1` crosses `s_y` from above. Two paired-LogUp lookups
against `T_A, T_B` (sharing the `z` key) certify that `y_A, y_B` are
the right exp values. No slack — table identity + monotonicity make
the bracket exact.

### Optional: high-z saturating mux (`saturate=True`)

`T_A[k]` rounds to 0 once `k > s_c · log(2·s_y)` (~9·s_c at s_y=s_c=S);
beyond that index the table is just zeros. Without saturation, `Z_max`
must still cover the full `c2 − min(x)` spread (LogUp requires every
key match an entry), so most of the table is wasted zeros. The
saturating-mux gadget — same construction as SiLU's high-magnitude
mux — lets us size `Z_max` to the non-zero exp region only:

| extra var | length | what it is |
|---|---|---|
| `z_high` | B·M | high word of z (z = z_low + Z_max·z_high) |
| `inv_z_high` | B·M | Fermat inverse of z_high (0 when z_high=0) |
| `is_high` | B·M | boolean (= 1 iff z_high ≠ 0) |
| `y_A_raw, y_B_raw` | B·M each | raw `T_{A,B}[z_low]` (lookup result) |
| `mux_y_A, mux_y_B` | B·M each | `is_high · y_*_raw` (mux output) |
| `z_z_high` | B·M | phase-2 inv for z_high range LogUp |

The `z` field of the claim now holds **z_low** (the lookup key into T_A/T_B).
Extra linear: `z = c2 − x − Z_max·z_high`; `y_A_raw = y_A + mux_y_A` (and
B). The `pt_u_A/B` linear binds the raw lookup: `pt_u_A = z + β·y_A_raw`.
Extra quadratics (one per cell): `z_high·inv_z_high = is_high`,
`is_high·z_high = z_high`, `is_high² = is_high`, `is_high·y_A_raw = mux_y_A`
(and B), `(α_zh − z_high)·z_z_high = 1` (range LogUp on `range_z_high`,
size `2^Z_high_width`).

**One-sided.** Only the upper tail (large z, small exp) is saturated.
The lower tail — c2 < max(x) — would make z negative; in the field, that
means `c2 − x ≡ P − k` (huge). The range tables cap `z_low + Z_max·z_high`
at a small value (≈ Z_max · 2^Z_high_width), so the decomposition linear
can't be satisfied for huge values: the proof rejects. So `c2 ≥ max(x)`
is enforced automatically by witness-range constraints; no separate gadget
needed.

---

## SiLU

**Function.** `output = silu(x) = x · sigmoid(x)`, input/output at
Q-format `2^r`. Magnitudes above `b·T_LEN` saturate to `x` (positive)
or `0` (negative).

**Config.** `b, T_LEN, b_2, b_3, b_4, width_2, width_3, width_4, r,
s_in (optional, ≥ 2^r)`. Derived: `s_x = 1<<r`, `rescale_bits = log2(s_in/s_x)`.

**I/O.**

- Input: `x` (public, length `L`) at scale `s_in`. Signed; field rep
  maps negative reals to large field elements.
- Output: `output` (length `L`).

**Phase-1 vars.**

| var | length | what it is |
|---|---|---|
| `x` | L | internal signed x at scale s_x |
| `output` | L | silu(x), with saturation mux applied |
| `sign` | L | 0 (x ≥ 0) or 1 (x < 0) |
| `magnitude` | L | \|x\| |
| `C` | L | sign · x |
| `a_0` | L | magnitude mod b (sub-bin position, range [0, b)) |
| `a_1` | L | (magnitude / b) mod T_LEN (table input) |
| `a_2, a_3, a_4` | L each | high-word saturation chunks (widths width_2/3/4) |
| `g` | L | b_2·a_2 + b_3·a_3 + b_4·a_4 (sat indicator value) |
| `inv_g` | L | 1/g if g ≠ 0 else 0 (Fermat) |
| `is_high` | L | g·inv_g (≡ saturation flag ∈ {0,1}) |
| `key` | L | sign·T_LEN + a_1 (paired lookup index) |
| `output_sat` | L | x − C (signed-saturated value: x if pos, 0 if neg) |
| `mux_a, mux_b` | L each | is_high · y_lookup, is_high · output_sat |
| `y` | L | T_combined[key] (paired lookup result) |
| *rescale:* `x_in, x_low` | L each | |

**Phase-2 vars.**

| var | length | what it is |
|---|---|---|
| `pt_u` | L | key + β · y |
| `pt_z` | L | 1/(α_pt − pt_u) |
| `z_a0, z_a2, z_a3, z_a4` | L each | range-LogUp z's for word range checks |
| *rescale:* `z_x_low, z_x` | L each | |

**Tables.**

- `silu_table`: paired `(k, T_combined[k])` where `T_combined = T_pos || T_neg`,
  `T_pos[i] = round(silu(bin_center_i·2^r) · 2^r)`, similar for `T_neg`. Size `2·T_LEN`.
- `range_b`: `b` entries (for a_0).
- `range_w2, range_w3, range_w4`: 2^width entries each (for a_2/3/4).
- `range_rescale` (when active).

**Constraints.**

Linear (per cell):

- **Sign-magnitude link:** `x = magnitude + 2·C`
- **Magnitude decomp:** `magnitude = a_0 + b·a_1 + b_2·a_2 + b_3·a_3 + b_4·a_4`
- **Saturation accumulator:** `g = b_2·a_2 + b_3·a_3 + b_4·a_4`
- **Lookup key:** `key = sign·T_LEN + a_1`
- **Saturation value:** `x = output_sat + C` (= x if sign=0, = 0 if sign=1)
- **Output mux:** `y = output + mux_a − mux_b`
- **Paired-tlookup u:** `pt_u = key + β·y`
- *rescale:* `x_in = (1<<r')·x + x_low`

Quadratic (per cell):

- **Sign indicator:** `sign² = sign` (forces sign ∈ {0,1})
- **C definition:** `sign · x = C`
- **Saturation flag:** `g · inv_g = is_high`, `is_high · g = g`,
  `is_high² = is_high` (forces is_high ∈ {0,1}; commits to `is_high = (g ≠ 0)`)
- **Mux components:** `is_high · y = mux_a`, `is_high · output_sat = mux_b`
- **Range LogUps:** `(α_b − a_0)·z_a0 = 1`, same for a_2/a_3/a_4
- **Paired tlookup:** `(α_pt − pt_u)·pt_z = 1`
- *rescale:* `(α_rescale − x_low)·z_x_low = 1`, `(α_w2 − x)·z_x = 1`

**Intuition.** Sign-magnitude split + 5-word magnitude decomp. The bin
index `a_1` indexes the silu lookup table (`T_pos` for non-negative
inputs, `T_neg` for negative — concatenated and selected by the
`sign·T_LEN` offset). The high words `a_2, a_3, a_4` are zero when the
input is in lookup range; non-zero when saturated, in which case
`is_high = 1` and the output mux replaces the lookup result with
`output_sat` (which is `x` for positive saturation, `0` for negative).

---

## MoE routing (top-1) — `RoutingClaim` + `MaskedCombineClaim`

> **Status: implemented** (`routing_claim.py`; verifier handlers in
> `protocol.py` `_c_routing`/`_c_masked_combine` and `handlers.rs`
> `compile_routing`/`compile_masked_combine`; difftest case `routing_combine`).
> Implements `design-feasibility.md §3.6` / `appendix-moe-routing.md §B.3`
> **specialized to `topk = 1`** (Maverick's config): the threshold τ
> degenerates, and "m is the top-1" is exactly "m is the one-hot argmax of the
> tiebroken logits" — the audited MaxClaim pattern. The τ-threshold form in
> appendix §B.3 remains the documented path for general top-k.

**Function.** Given per-token router logits `r ∈ F^{T×E}` (a committed matmul
output), pin a one-hot mask `m ∈ {0,1}^{T×E}` to the argmax of the
tiebreaker-adjusted logits

```
rt[t,e] = r[t,e]·2^L + (E−1−e)        L = ⌈log₂ E⌉
```

(ties → lowest expert index, matching `torch.topk` and llama.cpp), and recover
the chosen raw logit `r_chosen[t]` for the input-side sigmoid routing weight.

**Config.** `T` (tokens), `E` (experts), `L_bits = ⌈log₂ E⌉`, and at the
builder: `B_logit` (bit-width bound on `|r|`) and `word_bits` for the gap
range decomposition. **Soundness precondition:** `r` must be range-bounded to
`±2^{B_logit−1}` by its producing matmul's `output_width` rescale, and
`2^{B_logit + L_bits + 1} ≪ P`.

**Layout.** Token-major throughout (`slot = t·E + e`), matching the router
matmul's output order — no transpose on the routing side. The per-slot RHS
bonus pattern `(E−1−e)` costs `T·E` singleton runs; fine at few tokens. (The
expert-major layout the earlier draft proposed is NOT used; the combine's
layout bridge is `L2_TransposeO2MScalar`, below.)

**Phase-1 vars** (`m` committed; the rest derived):

| var | length | what it is |
|---|---|---|
| `m` | T·E | one-hot top-1 mask ∈ {0,1} |
| `rt` | T·E | tiebroken logits `2^L·r + (E−1−e)` |
| `mrt` | T·E | `m · rt` |
| `rstar` | T | chosen tiebroken logit `Σ_e mrt[t,·]` |
| `gap` | T·E | `rstar − rt ≥ 0` |
| `r_chosen` | T | chosen raw logit (sigmoid input) |

**Constraints** (cid order F1→F5; quads after, in order):

```
F1  rt − 2^L·r = (E−1−e)              linear, RHS bonus pattern   (T·E)
F2  Σ_e m[t,e] = 1                    cardinality, RHS 1          (T)
F3  Σ_e mrt[t,e] − rstar[t] = 0       rowsum + id                 (T)
F4  gap + rt − rstar(broadcast) = 0   id + id + stride-O2M        (T·E)
F5  2^L·r_chosen + Σ_e (E−1−e)·m[t,e] − rstar[t] = 0              (T)
Q1  m·m = m                           booleanity
Q2  m·rt = mrt
```

**Gap range check — composed, not inlined.** The builder (`route_top1`)
appends a standalone `WordExtractionClaim` (`gap = Σ_n 2^{n·w}·word_n`) plus
one `RangeWordClaim` per word against a shared `2^word_bits` range table.
**Soundness guard (audit A1):** the wrong-mask argument requires the words to
be UNABLE to represent the field rep of a negative gap; `route_top1` asserts
`2^{n_words·word_bits} ≤ P − 2^{width}` and refuses to build otherwise (a
`B_logit`/`word_bits` choice with `n_words·word_bits ≥ 64` would otherwise
verify a wrong-expert proof).
Those two claims now have their own verifier handlers (`_c_word_extraction` /
`_c_range_word`, `"WordExtractionClaim"` / `"RangeWordClaim"` arms in
`handlers.rs`), so any claim can compose them.

**Witness (`routing_compute`).** Everything — including `m` itself — is
derived from `r` inside the engine: honest `m` = one-hot argmax of signed
`rt` (first-max index; ties match `torch.topk` and llama.cpp). No build-time
mask commitment or hint exists; the builder composes claims, the engine
computes values. Downstream values (`mrt`, `rstar`, `gap`, `r_chosen`)
derive from the possibly-TAMPERED mask (`TEST_TAMPER["m"]`), so a wrong mask
yields a consistent-looking witness and the *constraints* must reject — this
is what the negative tests exercise.

**Intuition / soundness.** Booleanity + cardinality force a one-hot. If it
selects `e' ≠ argmax(rt)`, then `gap[argmax] = rt[e'] − rt[argmax] < 0`, whose
field representative is `≈ P` and cannot recompose from `N·word_bits ≪ 64`
bits of range-checked words — the word-extraction linear constraint (or the
word range LogUp) rejects. All-distinct `rt` (tiebreaker) makes the argmax —
and hence `m` — unique. F5 is exact: `rstar = 2^L·r_chosen + (tiebreak bonus
of the chosen expert)`, and `Σ_e (E−1−e)·m[t,e]` is that bonus, linearly.
No τ exists in this form, so there is no threshold-freedom to reason about.

**Sigmoid routing weight (input-side, composed downstream).** Llama-4 scales
the expert *input*: `x_r = sigmoid(r_chosen) ⊙ x` (HF `Llama4TextMoe`,
llama.cpp `weight_before_ffn`). Composition: one `paired_tlookup` on
`r_chosen` into a sigmoid table (the `shift` parameter handles the signed
input), one stride-broadcast pin, one `HadamardClaim`. All experts then run on
the same committed `x_r`, so the all-experts-commit privacy argument is
unchanged and the masked combine stays unweighted.

### `MaskedCombineClaim` — y[t,:] = Σ_e m[t,e]·X_e[t,:]

**I/O.** Input: `m` (from `RoutingClaim`), `xs` = E per-expert streams
(each T·F). Output: `y` (T·F). Derived per expert: `m_rep[e]` (the mask
scalar replicated across the feature axis) and `prods[e] = m_rep[e] ⊙ X_e`.

**Constraints:**

```
G1  m_rep[e][t,j] − m[t,e] = 0     replicated-mask pin             (E·T·F)
G2  Σ_e prods[e] − y = 0           masked sum                      (T·F)
Q   m_rep[e] · X_e = prods[e]      per expert, in expert order     (E·T·F)
```

G1 is one `L2_TransposeO2MScalar` packet on the token-major `m` (cid =
`base + e·T·F + t·F + j` for source slot `t·E + e` — transpose + fan-out in
one emission; `protocol.py expand_transpose_o2m` / `Expander::TransposeO2m`
are the verifier twins) plus per-expert identity packets on each `m_rep[e]`.

**Privacy.** All `E` experts' streams are committed even at top-1 (`§B.4`: a
sparse witness leaks routing); inactive experts are zeroed by the mask in the
combine. The Freivalds-compressed `§B.4` commit form is a later optimization.

### Negative tests (`tests/test_routing_claim.py`)

| test | tamper | rejects via |
|---|---|---|
| `test_cheat_wrong_expert` | one-hot at a non-argmax index | gap < 0 → word recomposition / range LogUp |
| `test_cheat_cardinality_two` | two-hot mask | F2 |
| `test_cheat_cardinality_zero` | all-zero mask | F2 |
| `test_cheat_nonboolean_mask` | `m = [2, −1, 0, 0]` (Σ = 1 in F, so F2 passes) | Q1 booleanity (F3 co-fires) |
| `test_cheat_combine_wrong_y` | committed wrong combine output | G2 |
| `test_guard_rejects_unsound_word_params` | `B_logit=60, w=11` (words reach the negative-gap field range) | builder guard refuses |
| `test_tamper_{rt,mrt,rstar,gap,r_chosen,m_rep,prods}` | inconsistent derived value via `TEST_TAMPER` | F1 / Q2·F3 / F3·F4 / F4 / F5 / G1 / combine quad |

Positives: `test_positive` (ACCEPT + mask/r_chosen/combine values),
`test_tiebreak_lowest_index` (equal logits pin the lowest index). End-to-end
Rust: `tests/dump_routing_proof.py` dumps an ACCEPT and a wrong-expert REJECT
proof for `verify_proof`; compile parity via the `routing_combine` case in
`test_compile_parity.py` / `compile_difftest`.

---

## Cross-cutting

**TableSettlement.** Every `α_*` and `β_*` is sampled by the
auto-synthesized `TableSettlement` for its `Table`. Every `(α − v)·z = 1`
quadratic on a phase-2 `z`, combined with the cross-claim sum identity
from the settlement, is the LogUp membership proof for that variable's row.

**Freivalds aux.** RmsNorm's `u, p` use a verifier-sampled challenge
`ρ ∈ F^d` to fold the per-cell broadcast multiply
`y[b]·x[b·d+i] = output[b·d+i]` into B scalar checks
`y[b]·u[b] = p[b]`. Avoids committing `y_broadcast` of length `B·d`.

**The rescale block.** When `s_in > native_scale`, every non-arith claim
emits the same internal word-decomposition:

- Linear (per cell): `x_in = (1<<r)·x + x_low`
- Quadratic (per cell, tight): `(α_rescale − x_low)·z_x_low = 1` against a
  fresh 2^r-entry `range_rescale` table.
- Quadratic (per cell, loose): `(α_aux − x)·z_x = 1` against the existing
  16-bit `range_aux` / `range_slack` table (or `range_w2` for SiLU).

The two checks together pin the `(x, x_low)` decomposition uniquely so
the prover can't equivocate. The loose check fails for negative `x`
under the current implementation (field rep of a negative integer is
outside `[0, 2^16)`) — the **offset trick** (commit `x_shifted = x + 2^15`,
range-check `x_shifted ∈ [0, 2^16)`) is the standard fix and is not yet
landed.
