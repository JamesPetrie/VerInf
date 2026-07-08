# Design Feasibility: Ligero Zero-Knowledge Proofs for LLM Inference

## 1 Overview

We aim to verify frontier-scale LLM inference in zero knowledge: a verifier confirms that the prover ran a specific model on a specific input and produced a specific output, without learning the model weights or intermediate states. The target scale is models such as Llama 4 Maverick (approximately 400 billion total parameters, 17 billion activated per token). This document examines the feasibility using Ligero polynomial commitments.

### 1.1 Benefits

**Simplicity.** Ligero is simpler than proof systems based on elliptic curve pairings or recursive composition. It uses only Reed-Solomon codes, Merkle trees, and a small number of polynomial constraint tests. This makes the protocol easier to audit and easier to extend with custom gadgets.

**Prover cost less than inference work.** Committing a witness of `W` Goldilocks field elements costs approximately 234·W field operations and W hash compressions. Inference itself, for a model of `P` activated parameters over a sequence of length `S`, requires approximately 2·P·S field operations for the matrix multiplications. When the witness scales with the inference work, the proof system adds only a modest constant-factor overhead.

> **TODO:** Update cost figures once the Background section is finalized.

**Post-quantum security.** Ligero relies only on the collision resistance of a hash function and the proximity properties of Reed-Solomon codes. It does not use any cryptographic assumption known to be broken by quantum algorithms. Provided the hash function is quantum resistant, the proofs remain sound under quantum adversaries.

### 1.2 Downsides

**Proof size and verifier cost.** Ligero proof size and verifier work scale as O(sqrt(W)) in the witness size rather than logarithmically. For a Maverick prefill of a few thousand tokens, this means proof size on the order of tens of megabytes and verifier work in the billions of field operations. If this becomes binding, Ligero can be composed with a sumcheck-based outer protocol such as Ligerito.

**Interactive protocol.** The protocol requires multiple rounds of communication between prover and verifier. A non-interactive version via Fiat-Shamir is possible but introduces additional soundness assumptions and degrades soundness for multi-round protocols. For our use case the prover and verifier maintain a continuous communication channel, so interactive operation is acceptable.

## 2 Ligero background

### 2.1 Three-test structure

Ligero represents the witness as a matrix `U` with `m` rows, each row a Reed-Solomon codeword of length `N_LIG` encoding `K_DEG` polynomial coefficients. The prover hashes columns of `U` into a Merkle tree, producing a single root that binds the entire witness. The verifier then issues three tests over `U`:

- **Interleaved Reed-Solomon (IRS) test.** Every row is close to a Reed-Solomon codeword.
- **Linear test.** A specified system of linear constraints over the encoded values holds.
- **Quadratic test.** A specified system of quadratic (pointwise multiplicative) constraints over the encoded values holds.

All three tests share a single random column query set `Q ⊂ [N_LIG]` with `|Q| = T_QUERIES`. The prover responds to each test with one short polynomial and the columns at indices `Q` (along with Merkle paths from the column hashes to the committed root).

Our protocol partitions the witness across three separate Merkle-rooted commits (`R_W` for model weights, `R_p1` for per-prefill activations, `R_p2` for per-query randomness-dependent values), but the three tests operate on the concatenated virtual matrix `[R_W; R_p1; R_p2]` with column queries shared across all three commits. Soundness is analyzed once on the combined matrix, not via union bound over three separate IRS tests.

### 2.2 Polynomial commitments

A commit operation Reed-Solomon-encodes a witness of `W` field elements into `m = ⌈W / ELL⌉` rows of `N_LIG` codeword positions, hashes the columns of the resulting matrix, and builds a Merkle tree over the column hashes. The prover keeps the encoded matrix (`N_LIG` field elements per row) plus the Merkle tree, so queried codeword positions are returned by direct lookup.

Encoding one row uses an inverse NTT of length `K_DEG` (interpolating `K_DEG` values, including `K_DEG − ELL` random ZK padding values, to polynomial coefficients) followed by a forward NTT of length `N_LIG` evaluating the polynomial at all codeword points. A radix-2 Cooley-Tukey NTT of length `n` performs `(n/2) · log_2 n` butterflies, each consisting of one twiddle-factor multiplication, one addition, and one subtraction, giving `3` field operations per butterfly and `(3/2) · n · log_2 n` field operations per NTT.[^ntt-const] Subleading additive costs (bit-reversal permutation, final modular reduction) grow as `O(n)` and are dominated by the `O(n · log_2 n)` butterfly cost at our parameters, so we drop them.

Cost is measured in two units throughout the document: `F_*` quantities are field operations, `H_*` quantities are hash compressions.

```
F_commit(W) = (3 / (2 · ELL)) · (K_DEG · log_2 K_DEG + ρ · K_DEG · log_2(ρ · K_DEG)) · W

H_commit(W) = (ρ · K_DEG) / (B_block · ELL) · W
```

After committing, the prover keeps the encoded matrix in memory at `(ρ · K_DEG / ELL) · W` field elements per commit. The `ρ` factor accounts for the codeword expansion; the `K_DEG / ELL` factor covers the ZK padding rows. This memory is what makes column queries near-free.

Commitment-related costs are linear in `W` for fixed column dimensions and add across separate commits up to single-row rounding. Growing `K_DEG = O(sqrt(W))` reduces proof size and verifier cost to `O(sqrt(W) · log W)` (Ligero §5.3).

With the encoded matrix held, queried columns are returned by direct memory lookup at negligible compute cost. If memory is tight, the prover can drop the encoded matrix and hold only the upstream data the witness was derived from, raising per-opening cost from approximately `0` to `F_commit(W)`, since the prover repeats the encoding to recover the queried columns.

[^ntt-const]: This count assumes one field operation per modular multiplication, addition, and subtraction. A more conservative count that treats modular multiplication as two operations (multiply plus reduction) and counts conditional subtractions from lazy reduction in full gives roughly `5/2` operations per butterfly, scaling the numerical estimates by up to `5/3 ≈ 1.67×`. This does not affect any feasibility argument in this document.

[^ab-todo]: TODO: revisit this claim after the rest of the document is complete. The claim that `a` and `b` are always constants or scalar broadcasts in our protocol holds for the constraint families enumerated so far (Freivalds quadratic, LogUp inverses), but should be re-checked against the full set of quadratic-test uses once §3 (Our protocol design) is finalized.

### 2.3 Linear test

The linear test verifies a system of linear constraints `A · x = b` over the encoded witness, where `A` is sparse with `O(W)` non-zero entries. The verifier samples a random vector `r`, and the prover responds with the coefficients of a single polynomial

```
q(·) = Σ_{i ∈ [m]} r_i(·) · p_i(·)
```

of degree `< K_DEG + ELL − 1`, where `r_i(·)` is the polynomial encoding the i-th block of `r^T A` and `p_i(·)` is the row-i polynomial of the witness commit.

The dominant prover cost is computing `q`. The prover already has `p_i` evaluated at the `N_LIG` codeword points (§2.2). Per row, the prover extends `r_i` from `ELL` evaluations to `N_LIG` evaluations (inverse NTT of length `ELL` followed by forward NTT of length `N_LIG`), pointwise multiplies with `p_i`'s codeword evaluations, and accumulates into a running sum. After all `m` rows, one inverse NTT of length `N_LIG` recovers `q`'s coefficients. Per-row work is `F_NTT(ELL) + F_NTT(N_LIG) + 2 · N_LIG` (the last term covers pointwise multiplication and accumulation); this dominates the final inverse NTT. The cost of computing `r^T A` scales with the number of non-zeros `L` in `A`:

```
F_lin_prove(W, L) = ((3/2) · log_2(ELL) + (3/2) · (N_LIG / ELL) · log_2(N_LIG) + 2 · N_LIG / ELL) · W + 2 · L
```

The prover transmits `K_DEG + ELL − 1` field elements (the coefficients of `q`).

Verifier work is dominated by evaluating each `r_i` at the queried column points. Per query, the verifier evaluates each `r_i` (degree `< ELL`) at `η_j` via Horner at `~2 · ELL` field ops, then accumulates `m` products against the queried column (`~2 · m` ops, subleading). Per-query work is `~2 · m · ELL = 2 · W`, giving total

```
F_lin_verify(W) ≈ 2 · T_QUERIES · W
```

A single linear test handles arbitrarily many linear constraints simultaneously by aggregating them into `A`. The prover cost scales with the witness size `W` and the constraint non-zeros `L`. Linear constraints in our protocol include Freivalds matmul checks (§3.3), LogUp linear identities (§3.4), and structural copy constraints between commits.

Structurally, the linear test is Freivalds applied to a constraint system: a random vector `r` reduces many constraints to a single weighted combination, caught with probability `1 − 1/|F|`. What Ligero adds on top of plain Freivalds is the polynomial encoding and column-query consistency, which let the verifier check `Ax = b` over a Reed-Solomon committed witness (zero-knowledge) rather than over plaintext `x`. The Freivalds matmul checks (§3.3) we layer onto this are themselves linear constraints, fed into the same linear test.

### 2.4 Quadratic test

The quadratic test verifies pointwise constraints `x ⊙ y + a ⊙ z = b` over committed vectors `x, y, z` (each of length `Q`) and known public vectors `a, b` (also of length `Q`). Here `Q` is the number of pairwise products checked, equivalently the number of "filled positions" in any one of `x, y, z`, with `Q = m_quad · ELL` for `m_quad` rows in the quadratic-test encoded matrices. The verifier samples a random vector `r ∈ F^{m_quad}` (one scalar per row), and the prover responds with the coefficients of a single polynomial

```
p_0(·) = Σ_{i ∈ [m_quad]} r_i · (px_i · py_i + pa_i · pz_i − pb_i)
```

of degree `< 2 · K_DEG − 1`, where `px_i, py_i, pz_i` are the polynomials encoding row `i` of the witness commits `Ux, Uy, Uz` (each of degree `< K_DEG`) and `pa_i, pb_i` encode row `i` of the public `a, b` (each of degree `< ELL`). The verifier checks `p_0(ζ_c) = 0` for every `c ∈ [ELL]` (the constraint check, since `p_i(ζ_c) = x_{i,c} · y_{i,c} + a_{i,c} · z_{i,c} − b_{i,c} = 0` if the constraint holds) and `p_0(η_j) = Σ_i r_i · (Ux_{i,j} · Uy_{i,j} + Ua_{i,j} · Uz_{i,j} − Ub_{i,j})` at queried columns `j ∈ Q_cols` (binding `p_0` to the committed witness).

The dominant prover cost is computing `p_0`. The prover already has `Ux, Uy, Uz` in `N_LIG` evaluation form (§2.2). The vectors `a` and `b` are fixed before the test's randomness `r` is sampled but may incorporate verifier challenges from earlier rounds; in our protocol they are always constants or scalar broadcasts (e.g., `β` from a LogUp challenge), so `Ua, Ub` are either precomputed once or formed at negligible per-query cost from a precomputed all-ones codeword.[^ab-todo] Per row, the prover does pointwise multiplications and additions on the `N_LIG` codeword domain: `2 · N_LIG` mults (for `px · py` and `pa · pz`), `2 · N_LIG` adds and subs (combining the products and subtracting `pb`), plus `2 · N_LIG` for the `r_i`-scaled accumulation into `p_0`. The final inverse NTT of length `N_LIG` recovers `p_0`'s coefficients and is dominated by the per-row sum:

```
F_quad_prove(Q) = (6 · N_LIG / ELL) · Q
```

The prover transmits `2 · K_DEG − 1` field elements (the coefficients of `p_0`).

Verifier work has two components:

- **Polynomial evaluations on `p_0`** at `ELL` message-domain points (check (a)) and `T_QUERIES` codeword points (check (b)), giving `~F_NTT(2 · K_DEG) + T_QUERIES · 2 · K_DEG` field ops, independent of `Q`.
- **Per-query column dot products** at `T_QUERIES` queries: `~6 · m_quad` field ops per query, summing to `6 · T_QUERIES · m_quad`.

```
F_quad_verify(Q) = (6 · T_QUERIES / ELL) · Q + F_NTT(2 · K_DEG) + 2 · T_QUERIES · K_DEG
```

The quadratic test is structurally similar to the linear test (a random combination reduces many constraints to one polynomial), but the per-row weight is a scalar `r_i` rather than a polynomial `r_i(·)`. The prover skips the `r_i` extension step (which dominates linear test cost), and the verifier skips per-query polynomial evaluations of `r_i`. The quadratic test cost per pairwise product is therefore much smaller than the per-witness-slot linear test cost.

A new pairwise product adds one slot to each of `x, y, z`, so up to `3` committed witness slots per `ΔQ` (less if layout reuses slots, e.g., `x = z` in LogUp). When tallying end-to-end cost for a feature, count both: `ΔW` for the commit and linear-test contributions, and `ΔQ` for the quadratic-test contribution.

**Witness layout.** The Hadamard structure pairs values at matching column indices but does not restrict where those values come from in the underlying witness. Linear constraints (§2.3) can copy or broadcast witness slots into matching positions of the three test vectors, so a single row of `ELL` positions can host many short independent pairwise multiplications laid out side by side. This makes the quadratic test efficient for batched short products: as long as a family of constraints is Hadamard-shaped and fits within `ELL` positions, packing them into one row lets the test handle all of them in parallel with a single per-row scalar weight.

### 2.5 Zero-knowledge masking

Ligero achieves zero knowledge through two complementary masking mechanisms (Ligero §4.6, Appendix B.4):

**Per-row ZK padding.** Each row has `K_DEG` slots, of which `ELL` are constrained by the witness and `K_DEG − ELL` are filled with uniformly random field elements. Column queries reveal evaluations of polynomials whose values at the `K_DEG − ELL` random points hide the witness slots. Sustaining `T_QUERIES` queries per IRS test for `Q_max` proofs requires `K_DEG − ELL ≥ Q_max · T_QUERIES`, giving the query lifetime `Q_max = (K_DEG − ELL) / T_QUERIES`.

**Test-level affine blinding rows.** Each test (linear, quadratic, IRS) is augmented with one auxiliary row encoding a structured message (zero message for IRS, sum-zero for linear, all-zeros for quadratic). For example, in the linear test (§2.3) the prover replaces the response polynomial with the affine variant

```
q(·) = Σ_{i ∈ [m]} r_i(·) · p_i(·) + r_Blind(·)
```

where `r_Blind(·)` of degree `< K_DEG + ELL − 1` is the polynomial corresponding to the blinding row `u'`. The blinding term is added with coefficient one. Unlike the witness rows it is not weighted by `r`, so the blinding has full effect regardless of the verifier's choice of randomness. The structured-message constraint on `u'` (sum-zero in this case) ensures the verifier's consistency check `Σ_c q(ζ_c) = Σ r_{ic} · b_{ic}` is unaffected: `Σ_c r_Blind(ζ_c) = 0` by construction, so the blinding contributes nothing to the sum check while still hiding the witness-derived part of `q`. The quadratic and IRS tests use analogous constructions described in Ligero Appendix B.4 and B.5. Each blinding row adds one row of work to the test cost, negligible compared to the `m`-row sum for our parameters.

### 2.6 Soundness as a function of parameters

The combined Appendix C analysis of the extended Ligero paper bounds the verifier's rejection probability against any cheating prover by

```
ε ≤ ε_IRS + ε_lin + ε_quad + ε_field
```

with the four error terms

```
ε_IRS   = (1 − e / N_LIG)^T_QUERIES
ε_lin   = ((K_DEG + ELL) / N_LIG)^T_QUERIES
ε_quad  = (2 · K_DEG / N_LIG)^T_QUERIES
ε_field = (N_LIG + 3) / |F|
```

where `e` is the IRS proximity bound (the maximum number of corrupted columns the test must catch, a parameter of the analysis) subject to `e < (N_LIG − K_DEG + 1) / 2`. Following Ligero §5.3 we take `e = K_DEG`, giving `e / N_LIG = 1 / ρ` and

```
ε_IRS   = (1 − 1/ρ)^T_QUERIES
ε_lin   = ((K_DEG + ELL) / (ρ · K_DEG))^T_QUERIES
ε_quad  = (2 / ρ)^T_QUERIES
ε_field ≈ N_LIG / |F|
```

For reasonable parameter choices (`ρ > 3` and `ELL ≤ K_DEG`) the IRS term dominates the other query-based errors, so achieving query-based soundness `2^{−S}` against the IRS test requires `T_QUERIES = O(S · ρ / (ρ − 1))`. The field term `ε_field` is set by `N_LIG / |F|` independently of `T_QUERIES`; tightening it requires parallel repetition of the random-challenge phase or a larger field. Concrete values for our default constants are tabulated in §3.1.

## 3 Our protocol design

> **TODO:** Section needs further development. The skeleton below captures the high-level design choices, with details to be filled in as the Background section is completed and as we make detailed parameter and structural decisions.

The constraint families and cost accounting in §3 are written in terms of standard transformer architecture parameters: `L_dense`, `L_moe` (dense and MoE layer counts), `d` (model dimension), `n_q`, `n_kv`, `d_h` (attention head counts and per-head dimension), `d_ff_dense`, `d_ff_exp` (FFN hidden dimensions), `E` (experts per MoE layer), `topk` (active experts per token), and `S` (sequence length). Concrete values for Maverick are tabulated in §4.1.

### 3.1 Protocol constants

```
K_DEG     = 16384              polynomial degree bound
ρ         = 4                  Reed-Solomon inverse rate (rate 1/4)
T_QUERIES = 80                 column queries per IRS test (placeholder)
ELL       = 8192               constrained slots per row
B_block   = 8                  field elements per hash block
|F|       = 2^64 − 2^32 + 1    field size (Goldilocks prime)
```

Derived:

```
N_LIG     = ρ · K_DEG = 65536              codeword length
Q_max     = (K_DEG − ELL) / T_QUERIES = 102   queries per persistent commit
```

The hash function is Blake3 with 32-byte output. Blake3 absorbs 64 bytes per compression block, giving `B_block = 8` Goldilocks elements per block. We chose Blake3 over SHA-256 for SIMD throughput and over arithmetization-friendly hashes (Poseidon, etc.) because the prover hashes on the order of `10^{11}` bytes per query and a native-speed hash function dominates the design under that load.

Substituting these constants into the §2 formulas gives the per-witness-slot prover costs we use throughout §3 and §4:

```
F_commit(W)         ≈ 234 · W              [§2.2]
H_commit(W)         ≈ W
memory(W)           ≈ 8 · W                field elements
F_lin_prove(W, L)   ≈ 228 · W + 2 · L      [§2.3]
F_quad_prove(Q)     ≈ 48 · Q               [§2.4]
```

The Ligero soundness terms (§2.6) at these constants:

```
ε_IRS    = (3/4)^T_QUERIES
ε_lin    = (3/8)^T_QUERIES        (much smaller than ε_IRS)
ε_quad   = (1/2)^T_QUERIES        (smaller than ε_IRS)
ε_field  ≈ N_LIG / |F| ≈ 2^{-48}
```

Achieving `2^{-S}` against IRS requires `T_QUERIES ≥ S / log_2(4/3) ≈ 2.4 · S`; the placeholder `T_QUERIES = 80` gives `~2^{-33}`. The field term floors at `~2^{-48}`; tightening it requires parallel repetition.

### 3.2 Three commits

The underlying ZK proof system is Ligero (§2) with three Merkle-rooted commits:

- `R_W`: model weights, committed once at deploy time, persistent across many queries.
- `R_p1`: per-prefill activations (intermediate matmul outputs, routing decisions, normalization auxiliaries), committed per query before the verifier samples test challenges.
- `R_p2`: per-query randomness-dependent values (Freivalds intermediates, LogUp inverses), committed after the verifier samples challenges.

The three Ligero tests (linear, quadratic, IRS) run on the concatenated virtual matrix `[R_W; R_p1; R_p2]` with shared column queries, so a single `T_QUERIES`-column proof certifies all three commits jointly.

**ZK lifetime and re-commit.** The per-prefill commits `R_p1` and `R_p2` are fresh per query, so their column reveals don't accumulate across queries. Only `R_W` is at risk of running out of randomness: each `R_W` commit has a finite ZK lifetime determined by per-row padding, with `Q_max = (K_DEG − ELL) / T_QUERIES` queries (§2.5) before the accumulated column reveals exhaust the random budget. We extend `R_W`'s lifetime by periodic re-commit with a linear-equality test linking the new and old `R_W` codewords.

> **TODO:**
> - Detail the soundness reasoning for shared challenges across commits (one combined IRS bound applies, no `3 × ε_IRS` union bound). Cross-reference `security-and-performance.md` §1.7.
> - Compare against a Hyrax-style commitment scheme (Pedersen-based polynomial commitment, used in zkLLM): in Hyrax the prover does not need to pre-commit linear-combination intermediates because Pedersen homomorphism (`Commit(a) + Commit(b) = Commit(a + b)`) lets the verifier check linear relations directly, and multiplications are handled with prover-supplied "hints" rather than committed intermediate products. Our Ligero approach commits all intermediates explicitly, which is conceptually simpler and post-quantum but increases witness size. Discuss the trade-offs (post-quantum security, hash-only assumptions, prover speed, witness blow-up) and why Ligero is the right choice for our setting.
> - Detail re-commit cost and frequency, cross-reference `prover-feasibility.md` §10.
> - Discuss the R_p1 / R_p2 witness-generation split as an implementation simplification. R_p1 (committed activations: matmul outputs, RMSNorm outputs, mask values, FFN intermediates) consists entirely of fixed-point values bounded by the overflow analysis (§3.1) and can be generated by standard quantized inference at int64 or FP64 precision, then converted to field elements via modular reduction. FP64 is bit-exact at our magnitudes because every value at every stage fits in the 53-bit mantissa with margin; int32 is too narrow (matmul accumulators reach `~2^{44}`, overflowing int32 at `2^{31}`); int64 is the conceptually-clean default with FP64 as a faster H100-friendly alternative. R_p2 (Freivalds aux, LogUp query inverses) consists of field-random values without a fixed-point interpretation and requires native Goldilocks kernels (the smaller, post-challenge phase). The prover then reuses standard inference infrastructure (cuBLAS/cuDNN paths) for the bulk of witness generation and only pays for custom field kernels in R_p2.

### 3.3 Matmuls via Freivalds

Direct verification of a matmul `C = A·B` (shapes `m × k`, `k × n`, `m × n`) via per-element quadratic constraints would require `m·n·k` auxiliary witness slots (one per product `t[i,j,k] = A[i,k] · B[k,j]`) and the same number of quadratic constraints. At frontier scale this is infeasible: a single matmul with shapes `(m, k, n) ≈ (5000, 5000, 5000)` would need `~10^{11}` slots. Freivalds reduces the check to a small number of scalar tests via random reduction.

**Standard Freivalds (single vector).** The verifier samples `ρ ∈ F^n` and checks `Cρ = ABρ`. Auxiliary witness slots: `y = Bρ` (`k` slots) and `t[i,k] = A[i,k]·y[k]` (`m·k` slots). Constraints:

- Linear: `y[k] = Σ_j B[k,j]·ρ[j]` (`k` constraints) and `Σ_k t[i,k] = Σ_j C[i,j]·ρ[j]` (`m` constraints); total `m + k`.
- Quadratic: `t[i,k] = A[i,k]·y[k]` (`m·k` constraints).
- Soundness: `1/|F|` per matmul.

**Double Freivalds (two vectors).** The verifier samples `ρ ∈ F^n` and `λ ∈ F^m` (both derived from a single seed) and checks `λ^T C ρ = λ^T A B ρ`. Auxiliary witness slots: `y = Bρ` (`k` slots), `u = λ^T A` (`k` slots), and `p[k] = u[k]·y[k]` (`k` slots). Constraints:

- Linear: `y[k] = Σ_j B[k,j]·ρ[j]` (`k`), `u[k] = Σ_i λ[i]·A[i,k]` (`k`), and `Σ_k p[k] = Σ_{i,j} λ[i]·ρ[j]·C[i,j]` (1); total `2k + 1`.
- Quadratic: `p[k] = u[k]·y[k]` (`k` constraints).
- Soundness: `2/|F|` per matmul (`1/|F|` per random vector, union bound).

**Comparison.**

| | Naive per-element | Single Freivalds | Double Freivalds |
|---|---|---|---|
| Auxiliary witness slots | `m·n·k` | `m·k + k` | `3k` |
| Quadratic constraints | `m·n·k` | `m·k` | `k` |
| Linear non-zeros (`L`) | `O(m·n·k)` | `O(kn + mk + mn)` | `O(kn + km + mn)` |
| Soundness | exact | `1/\vert F\vert` | `2/\vert F\vert` |

Single Freivalds saves a factor of `n` (the contracted matmul dimension) in auxiliary witness and quadratic constraints. Double Freivalds saves another factor of `m/3` in witness and `m` in quadratic constraints, at the cost of a `2×` increase in soundness error per matmul (still `~2^{-63}` at Goldilocks). Linear non-zeros stay the same in absolute terms between single and double Freivalds, but are reduced by a factor of `n` (or `m`, for the larger of the two contracted dimensions) compared to naive.

**Choice.** We use double Freivalds for every matmul in the model.

> **TODO:** Plan witness layout to pack the per-`k` quadratic constraints `p[k] = u[k] · y[k]` densely into rows of `ELL` positions (§2.4 note on witness layout). Target `3` rows per matmul (one each for `u`, `y`, `p`); cross-feature packing where multiple matmuls or other Hadamard families share rows.

### 3.4 Non-arithmetic operations via LogUp

> **TODO:** Restructure this section so the paired lookup `(x, f(x))` form is introduced early, without dwelling on the single-value lookup. The plain multiset-inclusion form still belongs first (it's the cleanest way to state the LogUp identity), but the section currently spends too much real estate on it — the witness/verification breakdown, cost equations, and table-sharing discussion are all written around plain LogUp before the paired form appears. Trim the plain-LogUp treatment to just what's needed to state the identity, then introduce paired LogUp early and run the cost equations and downstream discussion on the paired form, since that's what almost every non-arithmetic op in our protocol actually uses.

LogUp (Habök 2022) reduces multiset inclusion `S ⊆ T` to a polynomial identity over multiplicative inverses. For a query multiset `S = {s_1, ..., s_M}` and a public table `T = {t_1, ..., t_T}` with multiplicities `m_j` counting how many queries equal `t_j`:

```
Σ_{i ∈ [M]} 1 / (β + s_i)  =  Σ_{j ∈ [T]} m_j / (β + t_j)
```

The identity holds at every `β` (where defined) iff the multiset inclusion is correct. The verifier samples a random `β` and the prover commits auxiliary inverses to verify the identity at `β`. zkLLM's tlookup is functionally equivalent (same identity, same witness shape), but uses sumcheck instead of Ligero's linear and quadratic tests for verification.

**Witness and verification.** The prover commits `M` query inverses `A_q[i] = 1/(β + s_i)` (in `R_p2` after `β` is sampled) and `T` multiplicities `m[j]` (in `R_p1`). Table inverses `B_t[j] = 1/(β + t_j)` are *not* committed. `T` and `β` are public, so both parties compute `B_t` locally and treat the values as public coefficients in the linear test (saving `T` slots vs sumcheck variants which must commit them; zkLLM and other sumcheck-based LogUp constructions commit `B_t` because sumcheck operates over committed multilinear extensions). The identity at `β` is then checked using Ligero's existing primitives:

- **Linear** (the sum identity): `Σ_i A_q[i] − Σ_j B_t[j] · m[j] = 0`, a single constraint with `M + T` non-zero coefficients (all `1` on the `A_q` slots, the publicly computed `B_t[j]` values on the `m` slots), fed into the linear test (§2.3).
- **Quadratic** (the query-side inverse definitions): `A_q[i] · (β + s_i) = 1` for each `i ∈ [M]`, fed into the quadratic test (§2.4). The table-side identity `B_t[j] · (β + t_j) = 1` is checked by the verifier directly when computing `B_t` from `T` and `β`; no committed-witness machinery is needed for it.

**Per-instance cost equations.** Per LogUp instance with `M` queries against a table of size `T`:

```
ΔW_logup(M, T) = M + T                 query inverses + multiplicities
ΔL_logup(M, T) = M + T                 one linear constraint with M + T non-zeros
ΔQ_logup(M, T) = M                     one quadratic constraint per query inverse
```

Flowing through the §2.2-§2.4 cost formulas at our default constants:

```
ΔF_commit(M, T)     = 234 · (M + T)
ΔF_lin_prove(M, T)  = 228 · (M + T) + 2 · (M + T) = 230 · (M + T)
ΔF_quad_prove(M, T) = 48 · M
ΔH_commit(M, T)     = M + T

F_logup_prove(M, T) ≈ 512 · M + 464 · T
H_logup(M, T)       = M + T
```

The verifier additionally computes `T` modular inverses for `B_t` (`~64 · T` field ops; negligible).

**Table sharing.** If `L` instances share the same table (e.g., all softmax operations across all layers use one `exp` table), we batch them into a single LogUp with combined queries `M = Σ_i M_i` and one shared multiplicity vector `m`. This pays the `T` cost once instead of `L` times, saving `(L − 1) · 464 · T` field ops per shared table (`464 = 234 + 230` is the per-T commit and linear-test cost from above).

**Paired lookup `(x, f(x))`.** To verify `y_i = f(x_i)` against a public table of `(t_in_j, t_out_j)` pairs, the prover commits `x_i, y_i` in `R_p1` along with multiplicities `m[j]` counting how many `i` satisfy `(x_i, y_i) = (t_in_j, t_out_j)` (well-defined before per-query challenges, since the multiplicity is over the unordered set of pairs). The verifier samples random `α, β` per query after `R_p1` commits. Both parties compute `t_combined[j] = t_in_j + α · t_out_j` and `B_t[j] = 1/(β + t_combined[j])` locally. The prover commits query inverses `A_q[i] = 1/(β + x_i + α · y_i)` in `R_p2`. The linear and quadratic tests check

- Linear: `Σ_i A_q[i] − Σ_j B_t[j] · m[j] = 0`.
- Quadratic: `A_q[i] · (β + x_i + α · y_i) = 1` for each `i ∈ [M]`.

With `α` random, the combined-value LogUp accepts iff the pairs `(x_i, y_i)` actually lie in the public `(t_in, t_out)` table (collision error `1/|F|`). The trick is the same random-linear-combination idea Freivalds uses to compress a matmul check (§3.3): one random scalar collapses a multi-dimensional consistency check into a single one. A single paired-LogUp instance certifies both membership and the function relation `y = f(x)` at the plain-LogUp cost (`ΔW = M + T`, `ΔL = M + T`, `ΔQ = M`). Almost every non-arithmetic op in our protocol uses this form.

**Table size and precision.** Resolution per entry is `R / T` for input range `R`. For multiplicative functions like `exp`, zkLLM's multi-segment trick (decompose `x = Σ_k b^k · x_k`, use one small LogUp per digit, `K` instances with `T = b` instead of one with `T = b^K`) gives exponentially smaller `T` at fixed precision. Non-multiplicative functions like `silu` and `1/√(x+ε)` need single-table LogUp; `T` grows linearly with target precision.

The single-table parameters (`T`, `silu` input range, `exp` clip range) are validated empirically by the Q-grid simulation. See `quantization-evaluation.md` for the measured rel-noise, top-1 agreement, and Gibbs unexplained-information bound on Llama-2-7B at each choice. Headline finding: at `T = 2^{16}` and a Q3.12 (or Q4.14) scale, the dominant accuracy lever is the `silu` input range, not the matmul scale or table size.

**Soundness.** Each paired LogUp instance contributes `~(M + T + 1) / |F|` to the rejection error.[^logup-soundness] At Goldilocks `|F| ≈ 2^{64}`, an instance with `M ~ 10^{10}` queries gives `~2^{-30}` per-instance.

[^logup-soundness]: TODO: validate whether per-instance and aggregate soundness suffice for our threat model.

The set of LogUp instances and aggregate witness contribution depend on which non-arithmetic operations the model uses (typically `exp`, `1/√(x+ε)`, `silu`, etc., each with its own table and `M`); see §4.6 for the Maverick instantiation.

> **TODO:** Plan witness layout to pack the per-query inverse constraints `A_q[i] · (β + s_i) = 1` densely into rows of `ELL` positions (§2.4 note on witness layout). Identify natural batching across LogUp instances to minimize padding overhead.

> **TODO:** Replace the single-big-table treatment with the construction-per-op recommended in `nonarith-survey.md`:
>
> - **`1/√(x+ε)` (RMSNorm):** algebraic-relation check (`nonarith-survey.md` §2.10). No rsqrt-specific lookup table at all — soundness comes from two quadratic-test products (`y²·(x+ε)`, `(y−1)²·(x+ε)`) bracketing `d·s^4` plus byte range-checks on the slacks. Per-instance cost is ~14 byte-range LogUp queries + 4 quadratic constraints, no `R_p1` table material. Strictly cleaner than the originally-considered single huge linear table or exponent-mantissa decomposition. Validated end-to-end at toy scale in `spark-bench/tests/test_rmsnorm.py`. **Field-fit caveat: sound at activation scale `s ≤ 2^{12}` (Q3.12 — the design-doc default) without rescaling.** At Llama-2-7B parameters (`d = 4096`, `ε ≈ 10^{-5}`), the integer product `y² · S_total` is at most `2 · d · s^4`; for `s = 2^{12}` that's `~2^{61}`, comfortably under Goldilocks `P ≈ 2^{64}`. **At `s = 2^{14}` (Q4.14, the §7.1 recommended scale in `quantization-evaluation.md`) the product reaches `~2^{68}` and wraps mod P**, breaking the algebraic-rsqrt soundness because a cheating prover can hit small-residue field equalities that don't correspond to integer ones. **TODO: build a pre-rescale gadget** (bit-decompose `S_total`, drop low bits, run rsqrt on the rescaled value — see `analysis/precision_overflow_model.py:408-426`) if/when Q4.14 rsqrt becomes worth the ~0.4 percentage-point top-1 gain over Q3.12 reported in `quantization-evaluation.md` §5.5. For now we use Q3.12 for the rsqrt step (other ops can still use Q4.14 independently — per-op scale choice).
> - **`1/sum` (softmax denominator):** shift-invariance trick (`nonarith-survey.md` §2.3). Prover supplies the per-row log-sum-exp shift `ẑ` as auxiliary witness; protocol verifies `y = exp(x − ẑ)` via the existing `exp` lookup, and binds `ẑ` via one linear constraint per row `Σ_j y_j = scale`. Drops the `1/sum` LogUp instance entirely. Fallback if the row-sum constraint is awkward: §2.10 algebraic check with degree-1 inner relation (`y·s ≥ 2^{2k}`, `(y−1)·s < 2^{2k}`).
> - **`exp` (softmax numerator):** multi-segment digit decomposition (`nonarith-survey.md` §2.2; zkLLM's tlookup). Decompose `x_shifted` in mixed base, use `exp(a+b) = exp(a)·exp(b)` to chain `K` small per-segment lookups via quadratic-test products. Asymptotically smaller per-op table than a single big lookup.
> - **`silu` (FFN gate):** word-decomposition aligned lookup (`nonarith-survey.md` §2.11). Decompose `x_shifted = a_0 + b·a_1 + b·T·a_high`; `a_1` is the lookup index, `a_0` is sub-bin slop, `a_high` enforces saturation. One linear constraint for the decomposition, one paired LogUp for the lookup, byte range-checks for `a_0` and `a_high`. At the recommended `x_max = 20`, `T = 2^{16}` config (`quantization-evaluation.md` §5.3) the in-range guarantee from upstream constraints makes `a_high = 0` always, and the cost reduces to one LogUp + small range checks per query.

> **TODO (softmax follow-up, 2026-05-14):** the `1/sum` and `exp` bullets above were written before the **unified bracketed shift-invariance** construction (`nonarith-survey.md` §2.13) was worked out. §2.13 replaces both bullets with a single construction:
>
> - Prover supplies the per-row LSE shift `c2` as auxiliary witness; protocol verifies `y_i = exp(x_i − c2)` via a paired exp lookup keyed on `z_i = c2 − x_i`, and binds `c2` via the integer-friendly bracket `s1 = Σ y_i ≤ 1·s_y` and `s2 = Σ exp(−z_i + δ)·s_y ≥ 1·s_y` for `δ = 1` integer unit at scale `s_c`. Two paired exp tables (the second is the first shifted by `exp(δ)`), one `c2` range-check per row, two slack range-checks per row.
> - The paired LogUp's multiset inclusion implicitly enforces `z_i ∈ [0, Z_max)` — no separate per-input range check needed.
> - Strictly cheaper per query than the multi-segment `exp` above (2 lookups vs 3, no quadratic chaining) and integer-friendly where the `Σ_j y_j = scale` row-sum constraint above fails (it can't hold exactly under integer quantization without an explicit slack).
>
> When this is validated end-to-end (toy `test_softmax.py` in spark-bench, mirroring `test_rmsnorm.py`'s algebraic-bracket pattern), **rewrite this §3.4 to lead with §2.13 for softmax**, replacing the two bullets above. Multi-segment `exp` (§2.2) remains a useful primitive for `exp` outside softmax contexts and should stay as a separate technique reference.

### 3.5 Range checks via bit decomposition

For range constraints `x ∈ [0, 2^k)` that don't carry a functional dependency (no `y = f(x)` to verify), bit decomposition is often cheaper than LogUp and avoids the precomputed-table cost. The prover commits the binary representation of `x`:

```
b_0, b_1, ..., b_{k-1}    (k bits)
```

Constraints:

- **Bit validity** (quadratic): `b_i · (b_i − 1) = 0` for each `i ∈ [k]` (`k` quadratic constraints).
- **Composition** (linear): `x = Σ_i 2^i · b_i` (one linear constraint with `k + 1` non-zeros).

**Per-instance cost equations.** Per range-check on a single value of width `k`:

```
ΔW_range(k) = k                  bits
ΔL_range(k) = k + 1 non-zeros    in 1 linear constraint
ΔQ_range(k) = k                  bit-validity constraints
```

For `N` independent range-checks of width `k`: `ΔW = N · k`, `ΔL ≈ N · k` (in `N` constraints), `ΔQ = N · k`.

**Comparison with LogUp.** A LogUp range-check on the same range needs a table of size `T = 2^k`:

```
ΔW_logup = N + 2^k,    ΔL_logup = N + 2^k,    ΔQ_logup = N
```

LogUp wins when `N` is large enough that the `T = 2^k` table cost amortizes, typical for the non-arithmetic ops in §3.4 where `N` reaches `10^{10}` and `T = 2^{16}`. Bit decomposition wins when the table cost wouldn't amortize: pure bound-style range checks on a small number of values, or ranges where `k` is small enough that `N · k < 2^k`.

> **TODO:** Discuss bytewise (or more generally `b`-ary) decomposition via LogUp as the efficient middle ground. Splitting `x ∈ [0, 2^k)` into `k/8` byte-limbs and range-checking each limb with a single shared LogUp against a `T = 256` table costs `ΔW = N · k/8 + 256`, `ΔL = N · k/8 + 256`, `ΔQ = N · k/8` — an `8×` reduction in `ΔW` and `ΔQ` vs bitwise, with the table cost amortized across all range-checks of any width. Work out the crossover width `k` where bytewise beats bitwise once the shared `T = 256` table cost is folded in, and identify which range-checks in the protocol (e.g., the threshold-dominance check in §3.6) should switch to bytewise.

**Soundness.** Bit decomposition is exact: every `x ∈ [0, 2^k)` has a unique binary representation, and the constraints fully characterize it. No probabilistic-soundness term beyond what Ligero's tests contribute.

### 3.6 Mixture-of-experts routing

In an MoE layer each token routes to its top-`topk` experts by routing logit. Verifying MoE inference adds two challenges over a dense FFN: (1) the routing decision must be checked (the right experts are selected from the logits), and (2) the per-expert computation must be verified for active experts only, ideally without committing every expert's intermediates per token.

**Mask-and-gate construction.** Per (token, expert) the prover commits a binary mask `m_e ∈ {0, 1}` with `m_e = 1` on the chosen experts. The forward output is the linear combination `y = Σ_e m_e · expert_e(x)`, certified by one linear constraint per output coordinate once `m` is pinned.

To pin `m` to the top-`topk` of the routing logits `r ∈ F^E`, the prover additionally commits a per-token threshold `τ` and proves it separates active from inactive logits.

**Tiebreaker.** Top-`topk` must be uniquely defined to pin `m`; if two logits are equal, either could sit on the boundary between active and inactive, leaving `m` ambiguous. The protocol replaces each logit with a public tiebreaker-adjusted version

```
r̃_e := r_e · 2^L + (E − 1 − e)        L = ⌈log_2 E⌉
```

This is the logit shifted `L` bits left, with the expert index packed into the low `L` bits and ordered so lower indices win on ties (the bonus `E − 1 − e` is largest at `e = 0` and decreases with `e`, so `r̃_e > r̃_f` whenever `r_e = r_f` and `e < f`). All `r̃_e` are now distinct, and the top-`topk` of `r̃` is uniquely determined by `r` and the public index ordering.

Three checks pin `m`:

1. **Booleanity:** `m_e · (m_e − 1) = 0` for each `e` (`E` quadratic constraints per token).
2. **Cardinality:** `Σ_e m_e = topk` (one linear constraint per token).
3. **Threshold dominance:** `v_e := (2 m_e − 1) · (r̃_e − τ) ≥ 0` for each `e`, enforced by a bit-decomposition range check (§3.5) of width `B + L`, where `B` is the bit-width of `r`.

When `m_e = 1` the threshold-dominance constraint forces `r̃_e ≥ τ`; when `m_e = 0` it forces `r̃_e ≤ τ`. Combined with cardinality and the all-distinct `r̃`, this uniquely pins `m` to the top-`topk` of `r`. The threshold `τ` has freedom inside the open interval between the `topk`-th and `(topk+1)`-th largest `r̃`, but every valid `τ` certifies the same `m`, so this freedom doesn't leak the routing decision.

A malicious prover cannot satisfy all three constraints with a wrong mask: booleanity and cardinality fix `|supp(m)| = topk`; threshold dominance then forces that support to be the top-`topk` indices of `r̃` (i.e., the top-`topk` of `r` by tiebreaker). See `analysis/appendix-moe-routing.md` §B.4 for the full construction including matmul integration.

In its simplest form, mask-and-gate commits all `E` experts' FFN intermediates per token. This is a worst case for witness size but conceptually clean: every expert is checked uniformly, and an incorrect mask cannot elide a computation that would have been required. Particular routing patterns (e.g., top-1) admit further compression of the per-expert intermediates; see §4.8 for one such optimization applied to Maverick.

### 3.7 Composed soundness

The total prover-cheating probability bound combines Ligero's test errors (§2.6) with the per-claim errors from each constraint family. The two layers are independent under our commit ordering: per-query challenges (Freivalds `ρ`, `λ`; LogUp `β`, `α`) are sampled after `R_p1` is committed but before `R_p2`, so the prover cannot adaptively choose a witness against the challenges.

The composed bound is:

```
ε_total  ≤  ε_IRS + ε_lin + ε_quad + ε_field           [Ligero, §2.6]
          + Σ_matmuls 2 / |F|                            [Freivalds, per matmul]
          + Σ_logup_instances (M + T + 1) / |F|          [LogUp, per instance]
          + Σ_range_checks 0                             [bit decomposition is exact]
```

Each term contributes additively under union bound. The events ("Freivalds reduction false-accepts", "LogUp reduction false-accepts", "Ligero tests pass on a witness violating reduced constraints") are disjoint, so no double-counting.

The Ligero side is dominated by `ε_IRS = (1 − 1/ρ)^T_QUERIES`, controlled by the column-query count from §2.2. Per-matmul Freivalds error is `2/|F|`, summing to a vanishing total at any realistic matmul count when `|F| = 2^{64}`. LogUp's per-instance `(M + T + 1) / |F|` is typically the binding constraint at frontier scale, where `M` (queries per LogUp instance) scales with the per-prefill compute of a non-arithmetic operation and can reach `10^{10}` or more, giving per-instance error around `2^{-30}`.

To tighten beyond what one round of LogUp delivers: parallel repetition of the `β` challenge (`r` independent samples reduce per-instance error to `((M + T) / |F|)^r`, multiplying LogUp prove cost by `r`); multi-segment decomposition for multiplicative-function lookups (smaller per-instance `M`); or moving to a larger field. Concrete numbers for Maverick are derived in §4.7.

## 4 Prover Costs (Llama 4 Maverick Example)

> **TODO:** This section is a first pass. It accounts for committed activation values, double-Freivalds matmul auxiliary witness, and LogUp lookup contributions across the 48 layers of Llama 4 Maverick prefill. Excluded from the present pass: copy constraints between commits, IRS test affine blinding rows, ZK blinding rows for the linear and quadratic tests, RoPE auxiliaries, residual-stream re-commits, and re-commit amortization for `R_W`. The MoE expert accounting assumes the mask-and-gate construction (`security-and-performance.md` §B.4 / `analysis/appendix-moe-routing.md`) with all 128 experts' FFN intermediates committed (the worst case for witness size); if §B.4 admits a sparser commit form, MoE numbers will drop accordingly.

### 4.1 Architecture

Llama 4 Maverick (`meta-llama/Llama-4-Maverick-17B-128E-Instruct`):

| Symbol | Description | Value |
|---|---|---|
| `L_dense` | dense transformer layers | 24 |
| `L_moe` | MoE transformer layers | 24 |
| `L = L_dense + L_moe` | total layers | 48 |
| `d` | model dimension (hidden size) | 5120 |
| `n_q` | query attention heads | 40 |
| `n_kv` | KV heads (GQA) | 8 |
| `d_h` | per-head dimension (`d / n_q`) | 128 |
| `d_ff_dense` | dense FFN hidden (`intermediate_size_mlp`) | 16384 |
| `d_ff_exp` | per-expert FFN hidden (`intermediate_size`) | 8192 |
| `E` | experts per MoE layer | 128 |
| `topk` | active experts per token | 1 |
| `S` | sequence length | 2048 / 8192 |

Derived: `d_q = n_q · d_h = 5120` (concatenated query dimension), `d_kv = n_kv · d_h = 1024` (concatenated KV dimension). GQA replicates each KV head `n_q / n_kv = 5` times for compute.

### 4.2 Per-layer matmul shapes

Attention block (same in dense and MoE layers):

| Matmul | Shapes `(m, k, n)` |
|---|---|
| Q proj | `(S, d, d_q)` |
| K proj | `(S, d, d_kv)` |
| V proj | `(S, d, d_kv)` |
| `QK^T` per head | `(S, d_h, S)`, `n_q` heads |
| `AV` per head | `(S, S, d_h)`, `n_q` heads |
| Output proj | `(S, d_q, d)` |

Total attention matmuls per layer: `4 + 2·n_q = 84`.

FFN (dense, SwiGLU):

| Matmul | Shapes `(m, k, n)` |
|---|---|
| Gate proj | `(S, d, d_ff_dense)` |
| Up proj | `(S, d, d_ff_dense)` |
| Down proj | `(S, d_ff_dense, d)` |

FFN (MoE, SwiGLU per expert × `E`): same three shapes per expert, with `d_ff_exp` in place of `d_ff_dense`. `3·E = 384` matmuls per MoE layer.

### 4.3 Per-prefill committed activations (`R_p1`)

Per-layer outputs (newly produced activations, excluding inherited residual stream).

**Dense layer:**

```
W_dense(S) = (d_q + 2·d_kv + d_q + d + 3·d_ff_dense + d) · S  +  2·n_q · S^2
           = (5120 + 2048 + 5120 + 5120 + 49152 + 5120) · S  +  80·S^2
           = 71680 · S  +  80·S^2
```

**MoE layer** (mask-and-gate, all `E` experts committed):

```
W_moe(S) = (d_q + 2·d_kv + d_q + d) · S  +  2·n_q · S^2  +  E · (3·d_ff_exp + d) · S  +  E · S
        = 17408 · S  +  80·S^2  +  128 · (24576 + 5120) · S  +  128 · S
        = 3818624 · S  +  80·S^2
```

(The trailing `E · S` covers the per-token routing mask `m[E][t]`.)

**Total per-prefill activations** (`R_p1`):

```
W_R_p1(S) = n_dense · W_dense(S)  +  n_moe · W_moe(S)
          = 24 · (71680·S + 80·S^2)  +  24 · (3818624·S + 80·S^2)
          = 24 · 3,890,304 · S  +  3840·S^2
          ≈ 9.34 × 10^7 · S  +  3840 · S^2
```

### 4.4 Per-query Freivalds auxiliary witness (`R_p2`)

Each matmul of shape `(m, k, n)` contributes `3k` auxiliary witness slots and `k` quadratic constraints under double Freivalds (§3.3).

**Per dense layer:**

| Matmul | `3k` aux | `k` quad |
|---|---|---|
| Q proj | `3 · d` | `d` |
| K proj | `3 · d` | `d` |
| V proj | `3 · d` | `d` |
| `QK^T` (`n_q` heads) | `n_q · 3·d_h` | `n_q · d_h` |
| `AV` (`n_q` heads) | `n_q · 3·S` | `n_q · S` |
| Output proj | `3 · d_q` | `d_q` |
| Gate proj | `3 · d` | `d` |
| Up proj | `3 · d` | `d` |
| Down proj | `3 · d_ff_dense` | `d_ff_dense` |

```
A_dense(S) = 6·(3·d) + 3·d_q + 3·d_ff_dense + n_q · 3·d_h + n_q · 3·S
           = 18·d + 3·d_q + 3·d_ff_dense + 3·n_q · d_h + 3·n_q · S
           = 92160 + 15360 + 49152 + 15360 + 120·S
           = 171,072 + 120·S            aux witness slots

Q_dense(S) = 6·d + d_q + d_ff_dense + n_q·d_h + n_q·S
           = 30720 + 5120 + 16384 + 5120 + 40·S
           = 57,344 + 40·S              quadratic constraints
```

**Per MoE layer** (attention same as dense; FFN expanded per expert × `E`):

```
A_moe(S) = (attention aux) + E · (2·3·d + 3·d_ff_exp)
         = 92160 + 15360 + 120·S + E · (6·d + 3·d_ff_exp)
         = 107520 + 120·S + 128 · (30720 + 24576)
         = 107520 + 120·S + 7,077,888
         = 7,185,408 + 120·S            aux witness slots

Q_moe(S) = (attention quad) + E · (2·d + d_ff_exp)
        = 25,600 + 40·S + 128 · (10240 + 8192)
        = 25,600 + 40·S + 2,359,296
        = 2,384,896 + 40·S              quadratic constraints
```

**Total `R_p2` aux witness and quadratic constraints:**

```
W_R_p2(S) = 24 · A_dense(S) + 24 · A_moe(S)
          = 24 · (171,072 + 7,185,408) + 24 · 240·S
          = 1.77 × 10^8 + 5760 · S       aux witness slots

Q_total(S) = 24 · Q_dense(S) + 24 · Q_moe(S)
           = 24 · (57,344 + 2,384,896) + 24 · 80·S
           = 5.86 × 10^7 + 1920 · S      quadratic constraints
```

### 4.5 Concrete numbers

Substituting `S = 2048`:

```
W_R_p1 ≈ 9.34 × 10^7 · 2048 + 3840 · 2048^2
       ≈ 1.91 × 10^11 + 1.61 × 10^10
       ≈ 2.07 × 10^11 slots         ≈ 1.66 TB at 8 bytes/slot

W_R_p2 ≈ 1.77 × 10^8 + 5760 · 2048
       ≈ 1.89 × 10^8 slots          ≈ 1.5 GB

Q_total ≈ 5.86 × 10^7 + 1920 · 2048
        ≈ 6.25 × 10^7 constraints
```

For `S = 8192`:

```
W_R_p1 ≈ 9.34 × 10^7 · 8192 + 3840 · 8192^2
       ≈ 7.65 × 10^11 + 2.58 × 10^11
       ≈ 1.02 × 10^12 slots         ≈ 8.2 TB
```

The per-prefill commit `W_R_p1` dominates. At `S = 2048` it is dominated by MoE FFN intermediates (`128 · 3 · d_ff_exp · S` per MoE layer ≈ `6.5 × 10^9` per layer × 24 = `1.55 × 10^{11}`); at `S = 8192` the attention `S^2` term grows to comparable magnitude.

### 4.6 Field operations and hash compressions per prefill

Plug §4.3-§4.5 witness counts into the §2 cost formulas. The relevant per-query work is:

- **Commit** on per-query data `R_p1 + R_p2`: `F_commit(W_R_p1 + W_R_p2)` and `H_commit(W_R_p1 + W_R_p2)`.
- **Re-commit on `R_W`** amortized over `Q_max` queries: `F_commit(W_R_W) / Q_max` and `H_commit(W_R_W) / Q_max`.
- **Linear test** over the concatenated `[R_W; R_p1; R_p2]`: `F_lin_prove(W_total, L)`.
- **Quadratic test**: `F_quad_prove(Q)`.

`R_W` (weight commit). With Maverick's `~4 × 10^{11}` total parameters at one Goldilocks slot per scalar:

```
W_R_W ≈ 4 × 10^11 slots
```

`L` (linear constraint non-zeros from matmuls). For each matmul `(m, k, n)`, double Freivalds contributes `kn + km + mn + 3k ≈ kn + km + mn` non-zeros. Summing across all matmuls in §4.2:

```
L_dense(S) ≈ 3.15 × 10^8 + 1.18 × 10^5 · S + 80 · S^2
L_moe(S)   ≈ 1.62 × 10^10 + 5.16 × 10^6 · S + 80 · S^2
L_total(S) = 24 · L_dense(S) + 24 · L_moe(S)
           ≈ 4.0 × 10^11 + 1.27 × 10^8 · S + 3840 · S^2
```

For `S = 2048`: `L_total ≈ 4.0 × 10^{11} + 2.6 × 10^{11} + 1.6 × 10^{10} ≈ 6.7 × 10^{11}` non-zeros.

**LogUp contributions per prefill.** Three classes of non-arithmetic operation in Maverick's prefill use LogUp lookups: softmax (via `exp`), RMSNorm (via `1/√(x+ε)`), and FFN gating (via SiLU). Per-prefill query counts in the architecture parameters from §4.1:

| Operation | Where used | Per-prefill `M` |
|---|---|---|
| `exp` | softmax across `n_q` heads in every layer | `(L_dense + L_moe) · n_q · S²` |
| `1 / √(x + ε)` | RMSNorm, two per layer (pre-attention, pre-FFN) | `2 · (L_dense + L_moe) · d · S` |
| SiLU | FFN gate path; dense layers contribute `d_ff_dense · S`, MoE layers contribute `E · d_ff_exp · S` (mask-and-gate commits all experts) | `(L_dense · d_ff_dense + L_moe · E · d_ff_exp) · S` |

Each operation uses one batched LogUp instance with the paired-lookup trick (§3.4) against a precomputed `(t_in, f(t_in))` table of size `T = 2^{16}`. Aggregating across the three operations, with `M_total = M_softmax + M_rmsnorm + M_silu`:

```
ΔW_logup = M_total + 3 · T
ΔL_logup = M_total + 3 · T
ΔQ_logup = M_total
```

The `T` contribution is negligible at frontier scale (`T = 2^{16}`, `M_total > 10^{10}`).

Concrete numbers for Maverick:

| | `S = 2048` | `S = 8192` |
|---|---|---|
| Softmax queries | `8.05 × 10^9` | `1.29 × 10^{11}` |
| RMSNorm queries | `1.01 × 10^9` | `4.03 × 10^9` |
| SiLU queries | `5.23 × 10^{10}` | `2.09 × 10^{11}` |
| `M_total` (`= ΔW_logup ≈ ΔL_logup ≈ ΔQ_logup`) | `~6.1 × 10^{10}` | `~3.4 × 10^{11}` |

LogUp witness sits in `R_p1` (multiplicities, `3 · T ≈ 2 × 10^5` slots, negligible) and `R_p2` (query inverses, `≈ M_total`). At `S = 2048` MoE SiLU dominates (mask-and-gate commits all 128 experts' FFN intermediates and verifies SiLU on each); at `S = 8192` the softmax `S²` and SiLU contributions are comparable.

> **TODO:**
> - Verify the `T = 2^{16}` choice gives sufficient precision for Maverick reference inference. At `R ≈ ±128`, this is `~2^{-8}` input resolution per entry. If higher precision is needed, evaluate scaling up to `T = 2^{32}` (gives `~2^{-24}` precision at the same input range; adds `~6 × 10^{12}` field ops total across 3 batched instances, ~`4%` of `F_total`; ~100 GB committed multiplicities, ~210 GB precomputed table data). For multiplicative functions like `exp`, multi-segment lookup (zkLLM tlookup decomposition) gives the same precision exponentially cheaper.
> - For SiLU under mask-and-gate, evaluate whether unrouted-token SiLU computations can be elided from the LogUp queries (or zero-padded efficiently). If yes, the per-prefill LogUp witness drops by `~E×` for the SiLU contribution.
> - Rework the per-op LogUp accounting once the constructions in §3.4's TODO (algebraic-relation check for rsqrt, shift trick for softmax denom, multi-segment for `exp`, word-decomposition for `silu`) are adopted. Expected directional changes: rsqrt drops from a `M_rmsnorm`-query LogUp on a multi-million-entry table to ~14 byte-range LogUp queries per rsqrt against the shared `T=256` table (`M_rmsnorm` term redistributes into byte-range queries plus quadratic constraints); softmax denominator term drops entirely if the shift trick is used; SiLU term unchanged in `M`, but the per-query cost gains a linear-decomposition constraint and small range-checks on `a_0`, `a_high`. The headline `M_total` figure will be dominated by `exp` and `silu`; rsqrt moves out of the LogUp ledger into byte-range and quadratic-test accounting.

**Per-query field operations** (applying §2.2-§2.4 formulas with default constants, including LogUp):

```
W_R_p2(total) = W_R_p2(matmul) + ΔW_logup
              ≈ 1.89 × 10^8 + 6.1 × 10^10
              ≈ 6.12 × 10^10

W_total = W_R_W + W_R_p1 + W_R_p2
        ≈ 4.0 × 10^11 + 2.07 × 10^11 + 6.12 × 10^10
        ≈ 6.69 × 10^11

L_total(total) = L_matmul + ΔL_logup
               ≈ 6.7 × 10^11 + 6.1 × 10^10
               ≈ 7.31 × 10^11

Q_total(total) = Q_matmul + ΔQ_logup
               ≈ 6.25 × 10^7 + 6.1 × 10^10
               ≈ 6.15 × 10^10

F_commit_per_query  = 234 · (W_R_p1 + W_R_p2) + 234 · W_R_W / Q_max
                    ≈ 234 · 2.69 × 10^11 + 9.4 × 10^11
                    ≈ 6.39 × 10^13

F_lin_prove         = 228 · W_total + 2 · L_total
                    ≈ 228 · 6.69 × 10^11 + 2 · 7.31 × 10^11
                    ≈ 1.54 × 10^14

F_quad_prove        = 48 · Q_total
                    ≈ 48 · 6.15 × 10^10
                    ≈ 2.95 × 10^12

F_total_per_query   ≈ 2.2 × 10^14
```

**Per-query hash compressions:**

```
H_commit_per_query  = (W_R_p1 + W_R_p2) + W_R_W / Q_max
                    ≈ 2.69 × 10^11 + 4 × 10^9
                    ≈ 2.73 × 10^11
```

Plus negligible per-query Merkle path verification (`T_QUERIES · log_2(N_LIG) · 3` hashes per query for the three commits, ~`4 × 10^3` total).

**Sanity check against inference cost.** Maverick prefill compute is roughly `2 · P_active · S ≈ 2 · 1.7 × 10^{10} · 2048 ≈ 7 × 10^{13}` field-equivalent operations. The prover does `~2.2 × 10^{14}` per query, i.e., `~3.1×` the inference compute. Order-of-magnitude consistent with §1.1's "modest constant-factor overhead" claim, with the dominant terms being the linear test on `R_W` (`9.1 × 10^{13}` ops), the per-prefill commit (`6.4 × 10^{13}` ops), and the LogUp contribution (`~3 × 10^{13}` ops).

**For `S = 8192`** (the `S²` attention terms become a meaningful share):

```
W_R_p1 ≈ 9.34 × 10^7 · 8192 + 3840 · 8192^2
       ≈ 7.65 × 10^11 + 2.58 × 10^11
       ≈ 1.02 × 10^12

W_R_p2 ≈ 2.2 × 10^8 + 3.4 × 10^11
       ≈ 3.4 × 10^11           (LogUp dominates; matmul aux negligible)

L_total ≈ 4.0 × 10^11 + 1.27 × 10^8 · 8192 + 3840 · 8192^2 + 3.4 × 10^11
        ≈ 1.70 × 10^12 + 3.4 × 10^11
        ≈ 2.04 × 10^12

W_total ≈ 1.76 × 10^12

F_total_per_query ≈ 234·(W_R_p1 + W_R_p2) + 228·W_total + 2·L_total + (R_W amortized) + 48·Q_total
                  ≈ 3.18 × 10^14 + 4.01 × 10^14 + 4.08 × 10^12 + 9.4 × 10^11 + 1.6 × 10^13
                  ≈ 7.4 × 10^14

H_per_query       ≈ 1.36 × 10^12
```

Inference compute at `S = 8192` is `~2 · P_active · S ≈ 2.8 × 10^{14}` field-equivalent ops, so prover overhead is `~2.6×`, between `3.1×` at `S = 2048` and the asymptotic regime where per-prefill terms grow faster than the constant `R_W`.

The `S²` contribution is `~25%` of `W_R_p1` at `S = 8192` (vs `~8%` at `S = 2048`) and `~15%` of `L_total` from matmuls. At very long context (`S ≥ 32K`) the `S²` terms would dominate `W_R_p1`, and the softmax LogUp `S²` quadratic constraints (`(L_dense + L_moe) · n_q · S²`) would dominate `Q_total`.

### 4.7 Soundness

Plugging the §4 quantities into the composed bound (§3.7):

```
ε_IRS         = (3/4)^80                          ≈ 2^{-33}    (dominates Ligero side)
ε_field       ≈ N_LIG / |F|                       ≈ 2^{-48}
Σ Freivalds   ≈ 5000 · 2/2^{64}                   ≈ 2^{-50}    (~5000 matmuls per prefill)
Σ LogUp       ≈ M_total / 2^{64}                  ≈ 2^{-28}    (M_total ≈ 6.1 × 10^{10}, S = 2048)
Σ range       = 0                                              (bit decomposition is exact)

ε_total       ≈ 2^{-28}                                        (S = 2048)
              ≈ 2^{-26}                                        (S = 8192, M_total ≈ 3.4 × 10^{11})
```

The total bound is dominated by the LogUp instance with the largest `M`, which at Maverick scale is the FFN gating nonlinearity (`M ≈ 5.2 × 10^{10}` at `S = 2048`, growing linearly with `S`). The softmax instance is next (`M ≈ 8 × 10^9` at `S = 2048`, growing as `S²` and overtaking SiLU at `S ≈ 6500`). Per-instance LogUp error scales as `M/|F|`; reducing it requires parallel repetition of `β` (multiplies prove cost) or smaller `M` per instance via multi-segment lookup.

> **TODO:** Confirm whether `~2^{-28}` is acceptable for the deployment threat model. If not, the cheapest knob is parallel repetition of the LogUp `β` challenge: doubling `r` from 1 to 2 takes the LogUp term from `~2^{-28}` to `~2^{-56}`, with `2×` LogUp prove cost (a small fraction of `F_total` since LogUp is `~16%` of the total at `S = 2048`).

### 4.8 Sum-before-nonlinearity optimization (preliminary)

> **TODO:** This subsection records a preliminary estimate for the §B.7.6 optimization in `analysis/appendix-moe-routing.md`. Numbers below need to be re-validated once the §B.4 mask-and-gate construction is reviewed against this optimization, and once the rest of §4 is updated to reflect the §B.4 forward-stream accounting (the current §4.3 may be over-counting per-expert FFN intermediates).

For top-1 routing (`topk = 1`), there is exactly one nonzero `m_e` per token. For any elementwise nonlinearity `φ` applied to a per-expert intermediate `g_e(x)`, the masked sum `Σ_e m_e · g_e(x)` equals the chosen expert's `g_{*}(x)`, so applying `φ` once after the sum yields the same result as applying it per-expert and summing. This collapses the per-expert nonlinearity chain to a single forward stream, reducing committed intermediates and nonlinearity LogUp queries by an `E − 1` factor. See `analysis/appendix-moe-routing.md` §B.7.6 for the construction.

In Maverick the eligible chain is the SwiGLU FFN (`gate, up, silu_gate, hidden = silu_gate · up`), where SiLU is the elementwise nonlinearity that gets pulled past the sum.

Estimated savings at `S = 2048`:

- `W_R_p1` drops by `~4 · (E − 1) · d_ff_exp · S · 24 ≈ 2.1 × 10^{11}` slots (the FFN-intermediate term in the current §4.3 estimate, which assumed per-expert forward intermediates).
- `W_logup` (SiLU portion) drops by `~(E − 1) · d_ff_exp · S · 24 ≈ 5.1 × 10^{10}` slots, since SiLU is now applied to the summed gate stream once per token-position rather than per (token, expert) pair.
- Hadamard quadratic constraint count drops by the same factor.

The down-projection step still uses per-expert outputs (each `W_down[e]` differs); a binary indicator `mind[e]` is committed alongside the sigmoid mask `m` to recover the chosen expert's contribution. Negligible additional cost.

Combined effect on per-query `F_total`: drops by `~234 · 2.1 × 10^{11} + 228 · 2.1 × 10^{11} + 48 · 5.1 × 10^{10} ≈ 1.0 × 10^{14}` field ops at `S = 2048`. Subtracted from the §4.6 baseline of `~2.2 × 10^{14}`, the new `F_total` would be approximately `1.2 × 10^{14}` (roughly halved), with the remaining cost dominated by the linear test on `R_W` and the per-expert Freivalds intermediates inside §B.4.

> **TODO:** validate the §B.4 forward-stream accounting (the current §4 numbers may already implicitly incorporate the summed forward chain since §B.4 commits `y_active = gate_summed`, not per-expert gate values). If so, the savings here may already be reflected in §4.3-§4.6 and this subsection is just clarifying intent rather than reducing numbers further. Re-derive after reviewing `analysis/appendix-moe-routing.md` §B.4 and §B.7.6 in detail.

### 4.9 Open items

- **MoE construction.** Confirm against `analysis/appendix-moe-routing.md` §B.4 whether all `E` experts' FFN intermediates are committed (worst case used here) or whether a sparser commit form is admissible. If sparser, MoE FFN witness drops by up to `E ×`.
- **Copy constraints between commits.** Inputs / outputs threaded across `R_W`, `R_p1`, `R_p2`. Tally pending.
- **IRS test affine blinding rows and ZK blinding rows** for the linear and quadratic tests.
- **RoPE coefficients and residual-stream re-commits** for layer interfaces.
- **Re-commit overhead** for `R_W` when its `Q_max` lifetime is hit.
- **R_W contribution** (weights). Currently treated as amortized; revisit per the §3.2 re-commit TODO.
- **Comparison to single Freivalds and naive baselines** at the same scale.
