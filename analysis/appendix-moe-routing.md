# Appendix: Verified Mixture-of-Experts Routing in Ligero

This appendix specifies a Ligero-compatible construction for proving a single
forward pass through a Mixture-of-Experts (MoE) feed-forward layer, with both
routing correctness and matmul correctness verified, and with routing privacy
preserved.  It uses bracket notation throughout (no subscripts).

## B.1 Setting

Per token.  Let `E` be the number of routed experts, `k` the number of active
experts per token (e.g. `k = 6` for DeepSeek-V4), and `(d_in, d_out)` the
input/output dimensions of each expert's matmul.  Each expert `e` has a weight
matrix `W[e]` of shape `(d_out, d_in)`, committed once at deploy time in a
weight commitment `R_W`.

Per-token inputs (committed in `P1`):
- `x ∈ F^{d_in}`         — the activation arriving at the FFN
- `r ∈ F^E`              — the router logits for this token

Per-token output (committed in `P1`, consumed by the next layer):
- `y_active ∈ F^{d_out}` — the gated FFN output

The honest computation:
```
y_active[k] = sum over (e, j) of  m[e] * W[e, k, j] * x[j]                (★)
```
where `m ∈ {0, 1}^E` is the top-k mask of `r`: `m[e] = 1` iff `r[e]` is among
the `k` largest entries of `r` (with index tiebreaker, see §B.3), and
`sum_e m[e] = k`.

The proof certifies both (★) and the identity of `m` as the top-k of `r`.

## B.2 Structure of (★) — what Freivalds compresses

(★) is a contraction of three tensors over indices `(e, j)` with free index
`k`.  Two observations drive the construction:

1.  **Freivalds compresses `k`.**  Sampling `rho ∈ F^{d_out}` and dotting both
    sides reduces (★) to a scalar identity per token, with soundness error
    `d_out / |F|`.

2.  **`m` is rank-1 in `e`.**  The residual `e`-contraction after the
    `k`-compression is `sum_e m[e] * s[e]` for a per-expert scalar `s[e]` —
    an inner product realized by `E` quadratic gates and one linear sum.  No
    further random projection is required.  Higher-rank mask structures
    (e.g. a per-token permutation matrix) would require additional reductions.

## B.3 Top-k mask gadget

Let `B` be the bit-width of `r`, and let `L = ceil(log_2 E)`.  Define a
public tiebreaker-adjusted logit:
```
rtilde[e] := r[e] * 2^L + (E - 1 - e)                                      (1)
```
(`rtilde` is computed inside the constraint system as one linear
constraint per `e`; the constants `2^L` and `(E - 1 - e)` are public.)

Because the low `L` bits of `rtilde[e]` encode the index `e`, all entries of
`rtilde` are distinct.  This makes the top-k uniquely defined and the mask
deterministic in `r`.

The prover commits the mask `m ∈ {0, 1}^E` and a scalar threshold
`tau ∈ F`.  The constraints:

```
m[e] * (m[e] - 1)         = 0                  (booleanity, E quadratics)   (2)
sum_e m[e]                = k                  (cardinality, 1 linear)      (3)

mr[e]   = m[e] * rtilde[e]                     (E quadratics)                (4)
mtau[e] = m[e] * tau                           (E quadratics)                (5)
u[e]    = mr[e] - mtau[e]                      (E linear)                    (6)
v[e]    = 2*u[e] - rtilde[e] + tau             (E linear)                    (7)
v[e]   in [0, 2^{B+L})                         (E LogUp range queries)       (8)
                                                                                  // TODO: revisit. For (B+L) ≈ 37 bits the LogUp table T = 2^{B+L} is too
                                                                                  //  large to be tractable. Standard practice is bit decomposition
                                                                                  //  (commit B+L booleanity-checked bits per range query, plus one
                                                                                  //  binding linear constraint v = Σ b_i · 2^i), or multi-segment
                                                                                  //  LogUp with K small per-digit tables. Choose the cheaper option
                                                                                  //  and update this constraint accordingly.
```

Constraints (4)–(7) compute `v[e] = (2*m[e] - 1) * (rtilde[e] - tau)`
using two auxiliary quadratics.  Constraint (8) enforces non-negativity via
LogUp into a public range table.

**Why this forces the unique top-k.**  When `m[e] = 1`, `v[e] = rtilde[e] - tau`,
forcing `rtilde[e] >= tau`.  When `m[e] = 0`, `v[e] = tau - rtilde[e]`, forcing
`rtilde[e] <= tau`.  Combined with `sum_e m[e] = k` and the all-distinct
`rtilde`, the only satisfying `(m, tau)` has `m` as the unique top-k indicator
of `rtilde`, with `tau` in the open interval between the `k`-th and
`(k+1)`-th largest `rtilde[e]`.  The threshold `tau` has freedom inside that
interval, but `m` is uniquely pinned.

## B.4 Freivalds-compressed matmul with masked gating

After committing `P1`, the verifier samples `rho ∈ F^{d_out}` uniformly at
random.  The prover then commits in `P2`:

| Slot | Shape | Role |
|---|---|---|
| `y[e, j]`     | `E · d_in`     | per-expert Freivalds vector       |
| `tt[e, t, j]` | `E · N · d_in` | per-(expert, token, j) product    |
| `s[e, t]`     | `E · N`        | per-expert per-token scalar       |
| `gate[e, t]`  | `E · N`        | masked scalar (mask applied here) |

(For the per-token statement, drop `t` from the indices; we keep it visible
when discussing prefill batches in §B.6.)

### Constraints

```
y[e, j]      = sum_k W[e, k, j] * rho[k]                  (linear, E · d_in)   (9)
tt[e, t, j]  = x[t, j] * y[e, j]                          (quadratic)         (10)
s[e, t]      = sum_j tt[e, t, j]                          (linear, E · N)     (11)
gate[e, t]   = m[t, e] * s[e, t]                          (quadratic, E · N)  (12)
sum_e gate[e, t]  =  sum_k rho[k] * y_active[t, k]        (linear, N)         (13)
```

By (9)–(11), `s[e, t]` is bound to `(W[e], x[t,:], rho)` and equals
`rho · (W[e] · x[t,:])`.  By (12), `gate[e, t] = m[t, e] * rho · Y[e][t,:]`.
The seam (13) is the Freivalds identity for (★): if `y_active[t, :]` is not
the masked combination of expert outputs, (13) fails for random `rho` with
probability at least `1 - d_out/|F|`.

### What Freivalds compresses and what it leaves behind

It is worth being explicit about which dimensions the Freivalds challenge
`rho` actually collapses, because the bulk of the witness lives in what
`rho` does **not** touch.

Constraint (9) is the Freivalds step: `rho` projects the matrix `W[e]`
along its output axis, replacing it with a length-`d_in` vector `y[e]`.
The `d_out` axis is gone after this.  What remains per token is an
**inner product over `d_in`**:

```
s[e, t]  =  x[t, :] · y[e]  =  sum_j x[t, j] * y[e, j]
```

Freivalds does nothing to this inner product itself.  The `d_in` axis is
the contracted dimension of the original matmul — it never appeared on
both sides of an equality, so no random-projection identity helps with it.
Constraints (10)–(11) verify the inner product the only way Ligero allows:
by committing each per-`j` pointwise product `tt[e, t, j]` and summing.
This is the source of the `E · N · d_in` cost — not the matmul, but the
residual inner product Ligero is forced to spell out pointwise.

A second Freivalds projection along the token axis (`sigma ∈ F^N`) can
compress this further for the **non-MoE** single-matmul case:

```
sum_t sigma[t] * s[t]  =  <y, x_sigma>     where  x_sigma[j] = sum_t sigma[t] * x[t, j]
```

Because `sigma` is public after the challenge, `x_sigma` is a linear
combination of committed `x[t, j]` and costs no extra commits.  The full
inner product collapses to `d_in` pointwise products instead of `N · d_in`
— an `N×` saving, with soundness `(N · d_out) / |F|` (an extension field
is needed for large `N`).

This trick **does not rescue the MoE case**.  The masked seam expands to:

```
sum_t sigma[t] * sum_e m[t, e] * sum_j x[t, j] * y[e, j]
```

The factor `m[t, e] * x[t, j]` is a product of two committed values.  No
public-coefficient projection can absorb it; Ligero must materialize the
per-`(t, e, j)` product as a quadratic intermediate.  We land back at
`N · E · d_in` regardless of how the projections are arranged — see
§B.7.1.

### Why all `E` experts must commit

The mask `m` is hidden, so the verifier cannot tell which experts are active.
Were the prover to commit `tt`, `s`, `gate` only for the `k` active experts,
the witness shape itself would leak the routing.  Privacy forces a uniform
witness: every expert commits its full Freivalds chain whether or not it
contributes.  The mask zeros out inactive experts at constraint (12).

The `y[e, j]` vectors do not depend on `t` and are shared across all tokens
in a prefill batch — see §B.6.

## B.5 Routing-weight gadget (for weighted top-k)

If the model uses unweighted top-k (mask in `{0, 1}^E`), §B.3 suffices.  If
the model uses softmax-weighted top-k (e.g. Mixtral), or per-logit affinity
weights (e.g. DeepSeek-V4's `Sqrt(Softplus(r[e]))`), replace `m[t, e]` in
constraint (12) with a weight `w[t, e] ∈ F` derived from `r` and the support:

- **Mixtral-style** (softmax over the top-k subset):
  ```
  exp[e]  = m[e] * Exp(r[e])                         (LogUp into exp table)
  Z       = sum_e exp[e]                              (linear)
  w[e] * Z = exp[e]                                   (quadratic, per e)
  ```
  Yields `w[e] = softmax(r restricted to support)[e]` for `e` in the support
  and `0` otherwise.  Renormalization to sum to 1 is automatic from the
  division.

- **DeepSeek-V4 style** (per-logit `Sqrt(Softplus(·))` affinity, not
  renormalized):
  ```
  sp[e] = m[e] * Softplus(r[e])                       (LogUp into softplus table)
  w[e]  = Sqrt(sp[e])                                 (LogUp into sqrt table)
  ```
  Two table lookups per expert; the affinity is monotonic in `r[e]`, so the
  threshold gadget of §B.3 still operates on `r` directly to identify the
  support — the activation is computed only for weighting, not ranking.

In either case, the seam (13) is unchanged in shape; only the meaning of
`gate[e, t]` becomes "weight × scalar" instead of "indicator × scalar".

## B.6 Soundness

Let `epsL`, `epsQ` be Ligero's linear and quadratic test soundness errors,
and `epsR` the LogUp range-check error.  Per token, a prover producing
accepted commitments with either (i) `y_active` violating (★), or (ii) `m`
not the top-k of `r`, succeeds with probability at most:

```
d_out / |F|       (Freivalds error on (13))
  + epsL          (catches lies in (1), (3), (6), (7), (9), (11), (13))
  + epsQ          (catches lies in (2), (4), (5), (10), (12))
  + epsR          (catches range violations in (8))
```

Across `N` tokens the Freivalds term sums to `N · d_out / |F|`.  For
small fields (e.g. `|F| = 65537`) and large `N`, an extension field is
required to maintain soundness.

## B.7 Cost and privacy tradeoff

### B.7.1 Why the `E · N · d_in` cost appears

The dominant cost is constraint (10) — the t-tensor `tt[e, t, j]` —
contributing `E · N · d_in` quadratic constraints and committed slots per
matmul.  Note that this is **not** the matmul itself; Freivalds already
collapsed the `d_out` axis (see "What Freivalds compresses" above).  What
remains is the masked inner product `sum_e m[t, e] · sum_j x[t, j] · y[e, j]`
— a degree-3 multilinear contraction in the committed witnesses
(`m`, `x`, `y`), where `y` is itself bound to `(W, rho)` by linear
constraint (9).  Ligero supports only degree-2 (pointwise quadratic)
constraints, so decomposing the degree-3
contraction forces an explicit commitment to one of the three pairwise
products at every `(t, e, j)`:

| Choose to commit | Slot | Same `E · N · d_in` cost |
|---|---|---|
| `x[t,j] * y[e,j]` | t-tensor (used here) | yes |
| `m[t,e] * x[t,j]` | mask-input intermediate | yes |
| `m[t,e] * y[e,j]` | mask-Freivalds intermediate | yes |

Any factoring of (★) within Ligero pays the same `E · N · d_in` price.
Moving the intermediate around does not eliminate it.

### B.7.2 What does and does not amortize

| Witness | Per-token cost | Shared across N tokens (prefill)? |
|---|---|---|
| `y[e, j]`           | —              | yes (one `E · d_in` block per layer/matmul) |
| `tt[e, t, j]`       | `E · d_in`     | no |
| `s[e, t]`, `gate[e,t]` | `2 E`       | no |
| `y_active[t, k]`    | `d_out`        | no |
| Routing gadget (m, tau, mr, mtau, u, v) | `~5E + 1` | no |
| LogUp range table multiplicities | — | yes (single shared table) |

Per-token marginal cost is dominated by `tt`: `E · d_in` field elements per
token regardless of batch size.  For DeepSeek-V4-Pro
(`E = 384`, `d_in ≈ 7168`), this is ≈ 2.75M slots per token per matmul,
compared to ≈ 7168 slots for `y_active` per token — a factor of `E`
between Freivalds-intermediate cost and forward-flowing data cost.

### B.7.3 Routing privacy as the cost lever

The `E×` blowup over `y_active` is the price of routing privacy.  Two
relaxations recover most of it:

1.  **Reveal the active set per token.**  The witness shape becomes
    `k · d_in` instead of `E · d_in`.  For V4-Pro: `k/E = 6/384 ≈ 64×`
    reduction.  The verifier learns which experts handled which tokens; the
    weights remain hidden if private.  For most inference-verification
    deployments, routing patterns are not load-bearing for confidentiality.

2.  **Switch proof system to sumcheck.**  Multilinear-IP protocols (GKR,
    HyperPlonk) verify (★) without committing per-`(t, e, j)` intermediates.
    Witness drops from `O(N · E · d_in)` to `O(log(N · E · d_in))`-style
    succinct openings, but the proof system, deployment story, and
    aggregation properties differ from Ligero.

Within Ligero with full routing privacy, `E · N · d_in` is a hard floor.

### B.7.4 Routing gadget overhead

The routing gadget itself (§B.3) is negligible.  Per token: `~5E` witness
slots and `~5E` constraints, plus `E` LogUp queries.  Compared to the
matmul side (`E · d_in` ≈ `7168 · E` slots), routing is ~0.025% of the
proof.  Speeding up routing is not a useful optimization target.

### B.7.5 Lookup-based "load the active expert" — an alternative cost shape

The mask-and-gate construction in §B.4 expresses expert selection as a
**circuit** that touches every expert: every `(t, e, j)` triple gets a
quadratic intermediate, and the mask zeros out the inactive ones.  An
alternative is to express selection as a **lookup**: commit the active
expert index per token and prove that a "loaded" Freivalds vector matches
the corresponding row of `y[e, :]` via LogUp.

#### Construction sketch

Replace constraints (10)–(13) with the following.  The prover commits:

- `e_t ∈ [0, E)` for each token (the active expert index, range-checked).
- `y_loaded[t, j]` for each `(t, j)` — claimed value of `y[e_t, j]`.
- `tt_loaded[t, j] = x[t, j] * y_loaded[t, j]` — Freivalds product (no
  `e` index any more).
- `s[t] = sum_j tt_loaded[t, j]` — per-token scalar.

The lookup constraint:
```
(e_t, j, y_loaded[t, j])  ∈  { (e, j, y[e, j]) : e ∈ [0, E), j ∈ [0, d_in) }
```
verified via the standard LogUp polynomial identity (inverse columns
commit, fractional-sum check).  Inverse-column constraints are pointwise
quadratic (`inv[i] * (X - q[i]) = 1`), so this remains within Ligero's
constraint class.

The seam becomes:
```
s[t]  =  sum_k rho[k] * y_active[t, k]                                   (linear)
```

with no `gate` and no per-expert intermediates.

#### Cost (top-1)

| Class | Count |
|---|---|
| `tt_loaded[t, j]` quadratic (Freivalds product) | `N · d_in` |
| LogUp inverse-column quadratic | `N · d_in + E · d_in` |
| `s[t]` linear (sum) | `N` |
| Range check on `e_t` | `N` queries into `[0, E)` |
| Lookup table size | `E · d_in` (shared across tokens) |

Dominant per-token cost: `~N · d_in` quadratic — an **`E×` reduction** from
`N · E · d_in`.  For DeepSeek-V4-Pro top-1 hypothetical: `N · 7168` versus
`N · 384 · 7168`.

#### Cost (top-k, k > 1)

Commit `k` indices per token (`e_t_1, ..., e_t_k`) and `k` loaded vectors;
sum the `k` Freivalds products with their routing weights.  Per-token cost
scales as `k · N · d_in` — still `E/k`-fold cheaper than the mask
approach.  For V4-Pro top-6: `~64×` reduction.

#### Privacy tradeoff

The mask-and-gate approach hides the active set entirely.  The lookup
approach hides each individual `e_t` (it is committed and only verified
through a polynomial identity), but the **table multiplicities** —
how many tokens routed to each `(e, j)` entry of the lookup table — are
themselves committed and entangled in the LogUp identity.  In the
standard ZK-LogUp construction these multiplicities are blinded so the
verifier learns nothing about them at queried points, but soundness
requires that they exist as honest counts in the prover's witness.

What this means in practice:

- **Per-token routing privacy: preserved** under standard ZK-LogUp.
  The verifier cannot tell which expert handled which specific token.
- **Aggregate routing distribution: leaks weakly.**  An adversarial
  verifier with side-channel access to multiplicity-related commitments
  may infer which experts received zero tokens (those table entries
  unused), which received many, etc.  Whether this is acceptable depends
  on whether the routing distribution is sensitive — e.g. for confidential
  inference where the input domain (and thus routing pattern) is private,
  this could leak input characteristics.

For the verify-this-inference-was-honest threat model, where the model and
its routing strategy are public and the only secret is the input, the
aggregate distribution leakage is usually acceptable.  For
confidential-inference deployments it merits careful analysis.

#### Comparison

| Approach | Quadratic cost per matmul | Per-token routing | Aggregate distribution |
|---|---|---|---|
| Mask-and-gate (§B.4)        | `N · E · d_in`     | hidden | hidden |
| LogUp loading (this section) | `N · k · d_in`     | hidden | weakly leaked |
| Public routing               | `N · k · d_in`     | revealed | revealed |
| Sumcheck-based proof system  | `O(log(N · E · d_in))` | hidden | hidden |

The lookup approach sits between mask-and-gate (full privacy, full cost)
and public routing (no privacy, minimal cost), recovering most of the
cost savings without revealing the per-token routing decision.  For
DeepSeek-V4-scale deployments, this is likely the right operating point
inside Ligero unless the threat model specifically requires hiding the
aggregate routing distribution.

### B.7.6 Sum-before-nonlinearity for chained ops (top-1 only)

§B.4 verifies a single matmul `(W[e], x, rho)` with masked gating. Real FFN
chains like SwiGLU place non-arithmetic operations and Hadamard products
between matmuls — `silu(gate_proj(x)) · up_proj(x)` then `down_proj(...)`.
The question is whether these chained operations need per-expert
intermediates `silu(gate_i)`, `silu(gate_i) · up_i`, etc., or whether they
can operate on a single summed forward tensor.

**Under top-1 routing only (`k = 1`), the chained operations collapse to
the summed forward stream.**  Under top-1, exactly one expert has a
nonzero routing score per token, so the sum of per-expert pre-nonlinearity
values is itself a single nonzero term:

```
Σ_e gate_i  =  Σ_e m[t, e] · (X · W_gate[e])  =  m[t, chosen] · (X · W_gate[chosen])
```

Applying the nonlinearity to this single term is identical to applying it
per-expert and summing:

```
silu(Σ_e gate_i)  =  silu(m[t, chosen] · gate[chosen])  =  Σ_e silu(gate_i)
```

The second equality holds because `gate[i] = 0` for non-chosen `i` (via
the input masking in `routed_in *= router_scores`) and `silu(0) = 0`. The
**reason this works is that the sum has exactly one nonzero term**, not
that it is sparse: nonlinearities do not distribute over sums in general
(`silu(a + b) ≠ silu(a) + silu(b)`), so the trick fails for top-k routing
with `k > 1`.

The same observation applies to the Hadamard product `silu_gate · up`:
under top-1, both `silu_gate` and `up` collapse to single nonzero terms
(at the same chosen expert), and the pointwise product preserves this.

**Witness savings (top-1 SwiGLU FFN, per MoE layer):**

| Forward tensor | Per-expert | Summed (top-1) |
|---|---|---|
| `gate`         | `E · d_ff_exp · N` | `d_ff_exp · N` |
| `up`           | `E · d_ff_exp · N` | `d_ff_exp · N` |
| `silu_gate`    | `E · d_ff_exp · N` | `d_ff_exp · N` |
| `hidden = silu_gate · up` | `E · d_ff_exp · N` | `d_ff_exp · N` |

Per layer this is `4 · (E - 1) · d_ff_exp · N` slots saved. For Llama 4
Maverick (`E = 128, d_ff_exp = 8192, N = 2048`):
`4 · 127 · 8192 · 2048 ≈ 8.6 × 10^9` slots per MoE layer × 24 layers
`≈ 2.1 × 10^{11}` slots saved over the full prefill.

The corresponding LogUp and Hadamard reductions:

- SiLU LogUp queries: `E · d_ff_exp · N` → `d_ff_exp · N` per layer
  (`E×` reduction).
- Hadamard quadratic constraints (`silu_gate · up`): same `E×` reduction.

**Down-projection still uses per-expert outputs.**  Each `W_down[e]` is
different, so `out[e] = hidden_summed · W_down[e]` produces a different
result per expert. To recover the chosen expert's contribution without
revealing the routing decision, we commit a binary indicator
`mind[e] ∈ {0, 1}` (in addition to the sigmoid scores `m`) and compute:

```
out_summed = Σ_e mind[e] · (hidden_summed · W_down[e])
           = hidden_summed · W_down[chosen]
```

`mind` is verified against `m` via a small auxiliary check (a LogUp into a
"sigmoid-score-positive" range, or a constraint that `mind[e] = 1 iff
m[e] > 0`).  Cost is `E · N` extra slots per layer plus a modest LogUp —
negligible relative to the savings.

**Why ZK is preserved.**  `gate_summed[t, :]` reveals the chosen expert's
scaled gate values *to anyone who can read the witness*, but Ligero's
per-row ZK padding (`K_DEG − ELL` random fillers per row) hides committed
values from column queries by construction. The same property protects
the mask `m` and per-expert scalars `s, gate` in §B.4; nothing additional
is leaked by replacing per-expert forward tensors with summed forward
tensors. The §B.4 per-expert Freivalds intermediates (`tt`, `s`, `gate`,
`y`) are unchanged — those still pay the `E × N × d_in` cost analyzed in
§B.7.1.

**Net effect.**  This optimization eliminates the per-expert duplication
*in the forward chain between matmuls* but does not affect the per-expert
Freivalds intermediates *within each matmul*. It is a `~E×` reduction on
the FFN-intermediate witness in `R_p1` and on the SiLU LogUp / Hadamard
quadratic costs, but leaves the dominant `E · N · d_in` per-matmul cost
identified in §B.7.1 in place. To attack that, see §B.7.5 (lookup-based
loading) or §B.7.3 (routing-privacy relaxations).

## B.8 Worked example (E = 2, top-1, two tokens)

```
X            = [[1, 0], [0, 1]]                  (two tokens, d_in = 2)
W[0]         = [[3, 1], [2, 4]]
W[1]         = [[1, 5], [2, 0]]
m[0, :]      = [1, 0]                            (token 0 → expert 0)
m[1, :]      = [0, 1]                            (token 1 → expert 1)
rho          = [1, 1]                            (Freivalds challenge)

Honest outputs:
y_active[0, :] = W[0] · X[0, :] = [3, 2]
y_active[1, :] = W[1] · X[1, :] = [5, 0]

Phase 2 commitments:
y[0, :]   = W[0]^T · rho = [5, 5]
y[1, :]   = W[1]^T · rho = [3, 5]
tt[0,0,:] = X[0,:] * y[0,:] = [5, 0]      (elementwise)
tt[0,1,:] = X[1,:] * y[0,:] = [0, 5]
tt[1,0,:] = X[0,:] * y[1,:] = [3, 0]
tt[1,1,:] = X[1,:] * y[1,:] = [0, 5]
s[0, 0]   = 5    s[0, 1] = 5
s[1, 0]   = 3    s[1, 1] = 5
gate[0, 0] = 1 * 5 = 5    gate[1, 0] = 0 * 3 = 0
gate[0, 1] = 0 * 5 = 0    gate[1, 1] = 1 * 5 = 5

Seam:
 token 0: gate[0,0] + gate[1,0] = 5  vs  rho · y_active[0,:] = 1*3 + 1*2 = 5  ✓
 token 1: gate[0,1] + gate[1,1] = 5  vs  rho · y_active[1,:] = 1*5 + 1*0 = 5  ✓
```

Note that `y_active` (the data flowing forward) is committed; the per-expert
output tensors `Y[0]`, `Y[1]` are not — only the Freivalds-collapsed scalars
`s[e, t]` carry their information into the seam check.

## B.9 Mapping to DeepSeek-V4

§2.1 and §4.2.1 of the V4 technical report give:

- MoE in **FFN layers only** (not attention).  V4-Pro: 384 routed experts +
  1 shared expert, top-6 active per token, expert intermediate dim 3072,
  hidden dim 7168.
- Routing weights are the per-logit affinity `Sqrt(Softplus(r[e]))`, not
  a softmax over the top-k subset (different from Mixtral).  Use the
  DeepSeek-V4 form of the routing-weight gadget in §B.5.
- The shared expert is always-on: add its Freivalds chain unconditionally
  (no mask, no gate); its output adds directly to `y_active`.
- The first 3 MoE layers use **hash routing** (a public function of the
  token ID).  The threshold gadget of §B.3 is replaced by a single
  hash-evaluation constraint binding `m` to the public hash of the
  (committed) token ID.  Top-k machinery is unnecessary in those layers.
- SwiGLU has three matmuls (gate, up, down).  Each gets its own Freivalds
  challenge `rho` and its own `y[e]` block; the `E · N · d_in` cost
  applies per matmul.

For per-layer per-token witness on V4-Pro, the matmul side dominates at
~`3 · E · d_in ≈ 8.3M` field elements (across the three SwiGLU matmuls).
Routing gadget contribution is ~`2K` slots — negligible.  Across 61 layers
and a 1M-token prefill, the total witness footprint is ~`5 × 10^14` field
elements within Ligero with full routing privacy — the regime where the
levers in §B.7.3 must be pulled to make verification feasible.
