# VerInf architecture

VerInf produces zero-knowledge proofs of large language model inference,
including inference that is nondeterministic. Production inference runs in
floating point and need not be reproducible bit-for-bit (it can be made
deterministic, as DeepSeek has shown, but proprietary deployments may not be), so
VerInf neither re-runs the model exactly nor needs to reproduce its
floating-point arithmetic. Instead it verifies that the information in a sequence
of tokens is well explained by a policy-compliant finite-field computation by a
committed model on committed inputs, and bounds the information the computation
does not account for. That bound is the only thing the proof makes public, and it
is what lets VerInf target a real, already-trained model directly rather than
only computations contrived to match an integer circuit.

The argument reveals nothing about the model's weights, and in a future version
will not reveal the input or output tokens either, instead verifying that the
hidden tokens match a cryptographic commitment. A small, separate verifier reads
the proof and answers `ACCEPT` or `REJECT`. Applications include verifying AI
compute agreements (the bound caps the bandwidth left for hidden work such as
covert training), detecting model-weight exfiltration, auditing that a deployed
model matches an approved version, and checking high-stakes inference for
tampering.

VerInf is an end-to-end prototype. It has proven and verified Llama-4 Maverick
(400B parameters, all 128 experts) on consumer hardware, a single NVIDIA DGX
Spark, which is, to our knowledge, the first proof of a model this size. The
demonstrated run is 847 tokens at development soundness (about 8 hours to prove
and 3 hours to verify); the soundness-grade target is roughly 1000 tokens in
about 32 hours and is in progress (see §8 and §10). It manages memory with a
streaming architecture (commit and fold the witness row by row, so peak memory
tracks the working set, not the model or proof), and uses CUDA to accelerate the
proof on the GPU. The verifier is written in Rust to run on a CPU, prioritizing
readability and trading off a smaller soundness margin; a high-performance
version could be written in CUDA.

Design strategy: aim for as much simplicity as possible with sufficient
performance. The target use case is high-assurance settings, for example
international agreements, where strong guarantees matter and large amounts of
prover and verifier compute are acceptable. That is why VerInf chooses Ligero
with an interactive proof: hash-based commitments only (no trusted setup,
plausibly post-quantum) and an interactive protocol together make a
self-contained proof system with a small trusted base, favoring a simple
construction and a small verifier over proof size.

Caveats. Research prototype; not security-audited. The demonstrated 400B run is
development soundness (`T_QUERIES=4`, one fused pass), not the sound four-round
protocol, and zero-knowledge blinding is not yet secured, so this is not yet a
public hiding-and-soundness claim (§10). Early runs explain about 90% of the
information in an FP4-Maverick token stream (0.597 unexplained bits/token), which
may not suffice for every application. Proofs are large (megabytes to gigabytes),
and token binding (the input/output anchor) is not yet implemented.

This document is the architecture overview, one place to understand the whole
system end to end. It describes the target architecture (the state we are
building toward) and tags each component so you can tell what runs today from
what is planned. It is self-contained; deeper dives and exact cost tables live in
the docs in [§11](#11-further-reading), and where those disagree with current
reality it follows the measured or latest source (see
[§12](#12-document-status-and-known-inconsistencies)).

---

## 1. What this proves, and the trust model

Precisely, the verifier confirms an untrusted prover ran a specific model on a
specific input and produced a specific output, learning nothing about the
weights, activations, or tokens, and returns `ACCEPT` or `REJECT`. The rest of
this section makes the overview precise: the bound the proof makes public, and
how the input and output are anchored. `[Implemented]` unless noted.

### Similarity, not a bit-exact rerun

Inference in floating point is not necessarily deterministic. It can be made
deterministic (DeepSeek has shown this), but a proprietary deployment may not
be, and summation order alone can change the result, so VerInf supports
nondeterminism for flexibility. A useful consequence is that it does not need to
model floating-point arithmetic exactly. Rather than re-running the model
bit-for-bit, it proves the committed output is well explained by a
policy-compliant finite-field computation, and bounds the unexplained
information, the output bits that computation does not account for. This is what
lets it target a real, already-trained model directly rather than only
computations that match an integer circuit exactly.

The bound (the "unexplained information" or UI framework):

```
unexplained information:  U(O) = H(O | D(x))
Gibbs upper bound:        U(O) ≤ − E_P[ log2 Q(O | D(x)) ]      for any predictor Q
prover's estimate:        U(o) = − Σ_i log2 Q_i(o_i | D(x), o_<i)
logit-noise model:        Q_i(o_i) ∝ exp(−(v* − l_i)² / σ²),    v* = max_i l_i
```

`D` is the declared computation; `O` is what the hardware could have produced. A
poor predictor `Q` only inflates the prover's own number, so `Q` can be left to
the prover. Only the bound `U` is revealed; the weights, inputs, and output
tokens stay committed and hidden. `[Implemented]`, measured at 0.597 unexplained
bits/token on a 400B model (§8). Formulation and security analysis are in the
unexplained-information paper (PDF in this repo).

### Token binding (the input/output anchor) `[Planned]`

The proof keeps both input and output tokens hidden, so on its own they float
free of the real run: a prover could commit different output tokens to
under-claim `U`, or a fake input `x` to certify nothing about the real prompt.
The planned fix binds each token stream to a commitment recorded independently,
before the proof, by an external recorder publishing `H1 = Hash(AES(tokens,
key))` and `H2 = Hash(key)`. In-proof, the prover commits the tokens and key and
proves both hashes match; AES (under a high-entropy key) keeps `H1` hiding
despite low-entropy tokens, and `H2` pins the key so the decryption is unique.
The bound tokens must be the same witness variables the UI claim consumes, wired
by equality constraints. Root-of-trust caveat: this is only meaningful if `H1`
and `H2` are recorded by a party the verifier trusts, independently and prior to
the proof; if the prover controls the recording, the binding is circular. Cost
is negligible (about 10⁶ constraints, under 0.01% of the forward proof).

---

## 2. The big picture

The end-to-end pipeline:

```
model (PyTorch-like tape)
   └─ claims ──compile──▶ linear + quadratic constraints over a witness matrix
                              │
witness values ──encode──▶ Reed-Solomon rows ──hash columns──▶ BLAKE3 Merkle root   (COMMIT)
                              │
random combiners ──fold──▶ IRS / linear / quadratic test polynomials                (TEST)
                              │
random column subset ──open──▶ revealed columns + Merkle paths                       (OPEN)
                              │
                          verifier re-derives constraints, re-checks ──▶ ACCEPT / REJECT
```

There are exactly two objects: a public claim list (what was computed, in the
clear) and the proof (the commitment roots, the test polynomials, and the opened
columns). The verifier recompiles the constraints from the public claim list; it
never trusts prover-supplied constraints.

The prover runs one mode: the sound four-round protocol, each commitment fixed
before the verifier draws the next challenge, so the prover cannot fit a witness
to a challenge it has already seen. `[Implemented]`, but see the soundness caveat
in §5 and §10; the public soundness claim is not yet met. (The challenges are
seed-derived today, at the points a future interactive transport would supply
them between rounds.) An earlier single-fused-pass "fast" development mode has
been removed.

---

## 3. The model and the claim language

The forward pass is written as ordinary tensor code against a tape, like
PyTorch. Each `tape.<op>` call records one claim, a public statement of what was
computed, and returns a handle; handles overload `@`, `*`, `+`. The same claim
list drives witness generation, the constraint compile, and verification. Adding
a model from existing ops needs no prover or verifier change; only a new
operation needs a new claim type (the UI bound was added this way, as two claim
types reusing existing lookups). `[Implemented]`

```python
norm = tape.rmsnorm(x, d=d, s=S, eps_int=EPS_INT)
g    = tape.hadamard_broadcast(norm, rms_w, SEQ=SEQ, d=d)
q    = tape.matmul(g, W_Q, s_a=S, s_b=S, s_out=S)
sc   = tape.matmul(qr, kr, transpose_b=True)      # attention scores
sm   = tape.softmax(sc, M=SEQ, s_x=S)
out  = tape.matmul(sm, v)
```

### The witness is fixed-point integers

Values are quantized to Int64 Q-format at a scale `S` (e.g. `2^14`); the prover
commits integers, the verifier checks them with field constraints, and lookup
tables handle the nonlinearities (§9). The implemented scheme is Int64, not
NVFP4; NVFP4 is a research direction (§8, §10).

### Claim types `[Implemented]`

Each claim compiles to linear and quadratic constraints (§4):

- Matmul: verified by Freivalds (random projection), not entrywise.
- Elementwise (Hadamard): per-slot quadratics.
- RMSNorm: an algebraic rsqrt bracket (pins the rsqrt scalar `y` to
  `⌈√(magic/S_total)⌉` with `magic = d·s⁴`, no rsqrt table) plus a
  Freivalds-folded broadcast multiply.
- Softmax: a two-table monotonicity bracket (§2.13). Two exp tables `T_A` and
  `T_B = T_A[·−δ]` pin the per-row log-sum-exp candidate; paired LogUp certifies
  each output. No slack.
- SiLU: sign-magnitude split plus a 5-word magnitude decomposition into a paired
  silu lookup, with a saturation mux for out-of-range inputs.
- Embedding: public lookup (and a one-hot path for hidden tokens, §8).
- RoPE, range/word-extraction, and an optional rescale gadget (word-decomposition
  pinning `x_in = 2^r·x + x_low`) shared by the non-arith claims.
- MoE routing (top-1): `RoutingClaim` pins a one-hot mask to the argmax of
  tiebroken logits; `MaskedCombineClaim` combines expert streams. All `E`
  experts' streams are committed even at top-1, since a sparse witness would leak
  the routing decision; inactive experts are zeroed by the mask (this is the
  information-flow policy).

> Catalog note: `pipeline/CLAIM_SPECS.md` fully specifies RmsNorm, Softmax, SiLU,
> and MoE routing; matmul, embedding, rope, and rescale are referenced there but
> not yet fully written up.

### Information-flow policy

The verifier does not just check arithmetic; it checks that the claim graph
meets the policy: the expected kind of computation with compliant information
flow (e.g. each weight read only in a forward pass; causal masking; all experts
committed). Automated semantic analysis of the claim graph is future work.

---

## 4. From claims to constraints

The compile walks the claim list and turns each claim into two kinds of
constraint over the witness, assigning each a contiguous range of constraint ids
(cids):

- Linear constraints: `A·x = b`, one row of `A` per constraint, `A` sparse.
- Quadratic constraints: pointwise `x ⊙ y + a ⊙ z = b` over committed `x,y,z`
  and public `a,b`.

The witness is a matrix: each Variable occupies a contiguous block of rows, each
row holds `ELL` message slots (`= 8192`). Variables are laid out phase-1
(committed before challenges) then phase-2 (auxiliary witnesses committed after
challenges).

### Matmul via double Freivalds

Checking `C = A·B` entrywise is infeasible (`m·n·k` slots). Instead the verifier
samples `ρ ∈ Fⁿ`, `λ ∈ Fᵐ` and checks `λᵀCρ = λᵀABρ`. The prover commits three
small projections, `y = Bρ` (length `k`), `u = λᵀA` (`k`), and `p[i]=u[i]·y[i]`
(`k`), so the claim emits about `2k+1` linear and `k` quadratic constraints, and
one length-`k` dot product replaces the whole matmul. Soundness `2/|F|` per
matmul. Used for every matmul in the model. `[Implemented]`

### Non-arithmetic ops via LogUp

The uniform pattern: to prove a value is in a table, commit a phase-2 inverse `z`
and emit a quadratic `(α − v)·z = 1` plus a cross-claim sum identity; `α` (and
`β` for paired `(key,value)` tables) is sampled by an auto-synthesized
`TableSettlement`. Paired LogUp (`pt_u = key + β·value`) certifies membership and
a function relation `y = f(x)` in one lookup. Table entries are public, so the
verifier computes the table side directly. `[Implemented]`

### Range checks

Either bit/word decomposition (`x = Σ 2^i·bᵢ`, plus booleanity `bᵢ²=bᵢ`; exact,
no probabilistic term) or a LogUp range table, whichever amortizes.

The point of compiling to two aggregate tests is that one linear test handles
arbitrarily many linear constraints (a random combiner folds them all), and one
quadratic test handles all the pointwise products (see §5).

---

## 5. The Ligero argument

VerInf uses Ligero over the Goldilocks field (`|F| = 2^64 − 2^32 + 1`). The
setting tolerates large proofs and occasional offline verification, so it favors
a simple construction with a small verifier over proof size. Hash-based
commitments only, so it is transparent (no trusted setup) and plausibly
post-quantum.

### Parameters

```
ELL       = 8192            constrained message slots per row
K_DEG     = 16384           polynomial degree bound (coeffs per row)
ρ         = 4               Reed-Solomon inverse rate
N_LIG     = ρ·K_DEG = 65536 codeword length (columns)
T_QUERIES = 80 (sound) / ~4 (dev)   random column queries
hash      = BLAKE3, 32-byte
```

### Commit

Each witness row is Reed-Solomon encoded: the `ELL` message values plus
`K_DEG − ELL` random ZK-padding values are interpolated (inverse NTT of length
`K_DEG`) to polynomial coefficients, then evaluated on a coset of length `N_LIG`
(forward NTT) to the codeword. The codeword columns are hashed into a BLAKE3
Merkle tree, whose single root binds the whole witness. The witness is three
stacked blocks: weights `R_W` (committed once, persistent), per-prefill
activations `R_p1`, and per-query auxiliary values `R_p2`, checked as one virtual
matrix with a shared column-query set. `[Implemented]`

### Test: the three test polynomials

Random combiners fold all constraints into a few short polynomials:

- IRS test (`q_irs`): "every row is close to a Reed-Solomon codeword." A random
  per-row combiner; `q_irs = Σ_i r_irs[i]·f_i` is a weighted sum of the row
  polynomials (no product, no NTT of its own).
- Linear test (`q_lin`): the linear constraints. This is Freivalds applied to the
  constraint system: a random `r` reduces `A·x = b` to one weighted combination,
  caught with probability `1 − 1/|F|`. `q_lin = Σ_i r_i·f_i`, where `r_i` encodes
  the i-th block of `rᵀA` and `f_i` is the row's witness polynomial; degree below
  `K_DEG + ELL − 1`.
- Quadratic test (`p_0`): the pointwise products. `p_0 = Σ_t r_quad[t]·(px·py +
  pa·pz − pb)`; degree below `2·K_DEG − 1`. The per-row weight is a scalar (not a
  polynomial), so the quadratic test is much cheaper per product than the linear
  test is per slot.

### Open and verify

The verifier names `T_QUERIES` random columns; the prover reveals them with
Merkle paths. The verifier then runs six checks (§7): Merkle, the IRS column
identity, the linear sum and column identities, and the quadratic zero and column
identities. Soundness comes from Reed-Solomon distance: any inconsistency shows
up in a constant fraction of columns, so a few random column checks make missing
it negligible.

### Zero-knowledge

Two mechanisms: per-row ZK padding (`K_DEG − ELL` random slots per row; sustains
a query lifetime `Q_max = (K_DEG − ELL)/T_QUERIES` per weight commit), and three
test-level blinding rows (`u_irs`, `u_lin`, `u_quad`) mixed into the test
polynomials with structural constraints that leave the verifier's checks
unaffected.

### Soundness, and the honest caveat

```
ε_IRS   = (1 − 1/ρ)^T_QUERIES = (3/4)^T_QUERIES   (dominates the Ligero side)
ε_lin   = (3/8)^T_QUERIES ,  ε_quad = (1/2)^T_QUERIES
ε_field ≈ N_LIG/|F| ≈ 2^-48
+ Σ_matmuls 2/|F|  + Σ_logup (M+T+1)/|F|  + Σ_range 0
```

At `T_QUERIES = 80` the IRS term is only `(3/4)^80 ≈ 2⁻³³` (reaching `2⁻ˢ` needs
`T ≈ 2.4·S`), so 80 is a development placeholder, not a high-soundness setting.
The LogUp `(M+T+1)/|F|` term (with `M` up to about `10¹⁰` at frontier scale) is
typically the binding constraint, about `2⁻²⁸` to `2⁻³⁰`, tightened by parallel
repetition of `β`. The interactive (sound) mode tolerates lower per-challenge
soundness without enabling grinding.

---

## 6. The prover

The prover runs on GPU: field arithmetic, Reed-Solomon NTTs, and Merkle hashing
are CUDA kernels over Goldilocks, JIT-compiled for the local card on first use.
`[Implemented]`

### Streaming

The full witness exceeds device memory, so the prover streams it one operation
at a time: encode each row, fold it into the column hashes and the
test-polynomial accumulators, then free it. Peak memory tracks the working set,
not the model or proof size. The same row-by-row structure parallelizes: NTT work
splits across rows and GPUs with hashing accumulated in dedicated nodes
(multi-GPU is `[Planned]`).

### The five phases / four sound rounds

1. Computation claims: the tape records the relationships.
2. Intermediate witness: walk the claim list, compute each value, RS-encode and
   update the column hashes, send the commitment.
3. Auxiliary witness: the verifier issues challenges (Freivalds `ρ,λ`; LogUp
   `α,β`); the prover computes and commits the phase-2 auxiliaries.
4. Polynomial tests: combiner challenges in; the prover sends the folded IRS,
   linear, and quadratic test polynomials.
5. Column opening: the verifier names columns; the prover reveals them with
   Merkle paths.

The prover runs phases 2 to 5 as four rounds (commit, challenge, recommit, open).

### The q-poly fold (the prover's dominant cost): current and target

Round 3 (the q-poly accumulate) builds `q_irs`, `q_lin`, and `p_0` from the
streamed witness. `q_lin` dominates (about 54% of prove). It is the largest
single prover cost and the focus of active optimization.

- Today `[Implemented]`: per chunk, expand per-row constraint packets into
  `(target, cid, coef)` triples, sort to group by target, segmented-sum to build
  `rᵀA`, interpolate, and `poly_mul` against the witness rows (three NTTs per
  row). A per-row packet store plus lazy compile plus a late-fold path handle the
  streaming.
- Target `[Planned]` (design notes `qlin-family-object-reorg.md`,
  `qlin-evenodd-fused-multiply.md`): replace per-row packets with one
  constraint-family object per `(variable, family)` (compile becomes a count pass
  plus reverse index, megabytes not gigabytes); build `rᵀA` per-family so the
  sort drops out; and do the multiply with an even/odd split, since the ζ-domain
  is the even half of the `2K`-th roots, so half the product is a free
  elementwise multiply and the rest is a size-`K` coset NTT. The fold becomes
  per-variable: encode a variable's rows once, feed both `q_irs` (a matvec, no
  transform) and `q_lin`; `p_0` stays constraint-stationary (its 3-row coupling
  does not fit a per-variable pass). An inverse-NTT fuse (one global inverse
  instead of per-row) is implemented on a branch (about 7.5%), not yet on `main`.

Prover cost is dominated by the witness computation (large Int64 matmuls), the
Reed-Solomon NTTs (memory-bandwidth bound on the GB10, so a faster kernel does
not help; fewer NTTs do), and the column hashing.

---

## 7. The verifier

The verifier runs different code from the prover and trusts only its own;
soundness protects it as zero-knowledge protects the prover. It recompiles the
constraints from the public claim list and checks the proof against its own
derivation; it never trusts prover-supplied constraints. A small Rust binary
(`verify_proof`), differential-tested bit-for-bit against a Python reference.
`[Implemented]`

### The six checks

1. `merkle`: re-hash each opened column and its Merkle path against the roots.
2. `irs_col`: the IRS column identity.
3. `lin_sum`: the linear constraint test (`q_lin` vs the compiled RHS).
4. `lin_col`: the linear column identity (the heavy check: per column,
   `Σ_rows challenge(cid)·coef·L_slot(η_j)·col[row] + blind == q_lin(η_j)`).
5. `quad_zero`: `p_0(ζ_c) = 0` for all `c`.
6. `quad_col`: the quadratic column identity (via a precomputed prefix-sum of the
   Lagrange basis, making each mask O(1)).

ACCEPT if all six pass, else REJECT.

### Trusted computing base (TCB)

What, if buggy, could cause a wrong ACCEPT: the field arithmetic, BLAKE3 and
Merkle, the challenge PRF, the constraint compile (the expander definitions), and
the six checks. Crate dependencies: `blake3`, `rayon`, `serde_json`. Everything
else, including all JSON and proof parsing, is outside the TCB: a wrong value
yields a failed check and so a REJECT, never a forged accept.

### Streaming the verifier `[In flight]`

The binding cost is the compiled constraints, which are `O(witness rows)`, not
`O(model)`: at the 48-layer Maverick run the verifier held about 99.6 GB resident
over 99.4 M rows `[measured]`, the wall that blocks the sound `T=80` config (its
roughly 250 GB of opened columns sit on top of that floor). The fix mirrors the
prover's family reorg: a per-constraint-family fold (the trusted eval surface
shrinks to three primitives: point, range-via-prefix-diff, dense-dot) and
row-streaming, so constraint memory is bounded. Currently implemented on
`feat/verifier-family-eval`: row-streaming `lin_col`, the per-family seam for two
expander variants, a typed (non-DOM) JSON parse that fits the 12 GB proof, and a
difftest gating shortcuts bit-exact against the oracle. The full
family-descriptor refactor and opened-column streaming are still `[Planned]`. A
CUDA verifier port is `[Planned]` (the Rust version stays for bit-exactness). The
verifier needs no GPU; the full-model verify needs a large-RAM host.

---

## 8. Cost and scaling

### The cost model `[projected]`

`design-feasibility.md` develops a per-claim cost model. At the protocol
constants, per witness element: `F_commit ≈ 234·W` field ops plus `W` hash
compressions; `F_lin ≈ 228·W + 2·L` (L = linear non-zeros); `F_quad ≈ 48·Q`. For
Maverick at `S = 2048` the model puts the prover at about 3× the inference
compute (dominated by the linear test on the weights, the per-prefill commit, and
LogUp). These are analytical field-op counts, not wall-clock.

### Measured runs `[measured]`

- 48-layer Llama-4 Maverick (400B, all 128 experts), 847 tokens (442 hidden
  prompt plus 405 public continuation), `T_QUERIES=4` (test-grade): proved and
  verified end-to-end on one DGX Spark. UI bound 0.597 bits/token, matching the
  float reference to four sig figs. Prove 8.04 h (peak GPU 42.3 GB), verify about
  197 min (resident about 99.6 GB), proof 12.5 GB. This is the first
  verifier-confirmed UI bound on a 400B-class model, but it is fast-mode,
  test-grade soundness, not the sound `T=80` protocol (§10).
- 32-layer Llama-2-7B, 1000 tokens: proves and verifies on one DGX Spark, prover
  peak about 20 GB.

### Scaling `[measured below about S=2048; projected beyond]`

Prove time is about linear in witness slots (about 40 ns/slot `[measured]`).
Witness size is `W(S) = c + a·S + b·S²`; the `S²` term is attention. The measured
quadratic coefficient is about 840 slots/block/S², roughly 10× the design-doc
estimate, and is dominated by the softmax proof witness, not the score
activations, and is expert-count-independent. Memory grows with `S²` and binds
early: a single block OOM'd at `S=4096` on a 121 GB box (the softmax `S²` witness
is materialized, not streamed). Concrete long-context projections follow from the measured coefficients (tagged
`[projected]`). The per-block quadratic coefficient (about 840 slots/S²) holds
for every layer, since attention is identical per layer, so across 48 layers
`B_full ≈ 40,320` slots/S². Past the full-model crossover where `a·S = b·S²`
(about 11,000 tokens) the witness is quadratic-dominated, so witness ≈
`B_full·S²`; the linear `a·S` term adds roughly 10% at 100k and less at 1M. Prove
time then follows at about 40 ns/slot, and the NVL72 entry is a memory-bound
floor (bytes moved over aggregate bandwidth) with a 2 to 2.5× overhead applied:

| context (dense) | witness | Spark prove (~40 ns/slot) | NVL72 (mem-bound floor; with overhead) |
|---|---|---|---|
| 100k | ~3.2 PB | ~6 months | ~40 min; ~1.5 to 2 h |
| 1M | ~323 PB | ~50 years | ~3 days; ~6 to 7 days |

Worked for 100k: `B_full·S² = 40,320·(10⁵)² ≈ 4.0×10¹⁴` slots, times 8 bytes is
~3.2 PB; at 40 ns/slot that is ~1.6×10⁷ s ≈ 6 months; the NVL72 floor is the 1M
figure divided by 100 (witness, and so bytes moved, scale as S²). Both rows rest
on a dense-attention assumption: sliding-window or sparse attention replaces
`b·S²` with about `b·S·w` and collapses them by roughly 100×. They are also past
the resident-memory regime, where the multi-PB witness must stream and the
per-slot rate is unvalidated (the memory wall binds first).

Proof size and verifier work scale as `O(√W)` (Ligero), so tens of MB to GB;
opened columns are `T·m_total`. Hardware in the cost model: DGX Spark (121 GB
unified), H100 clusters, NVL72 (`[projected]`).

### Optimization levers (`prover-optimization-investigation.md`)

The NTT is bandwidth-bound, so the lever is fewer NTTs: the inverse-NTT fuse
(about 7.5%, banked on a branch), and better handling of the sparse linear
constraints (about a quarter of the prove, with structure that repeats across
about 95% of chunks), cached or restructured by the q_lin family reorg, or a
scatter-add that removes the sort (§6). The combined realistic engineering
ceiling is about 1.3 to 1.4×; a clean 2× needs a protocol change (sumcheck or
GKR-style linear argument).

### NVFP4: research direction `[Planned/research]`

A hardware NVFP4 matmul on Blackwell is bit-exact-predictable in software up to
`K = 16384` `[measured]`, so an exact-integer NVFP4 matmul fits the existing
Freivalds design (factor out the per-tensor scales, absorb per-block scales into
small integers, verify by random projection). This is the path to committing FP4
intermediates instead of Int64 so FP4 errors do not propagate through the proof;
not yet implemented.

---

## 9. Numerics

The implemented scheme is Int64 fixed-point (Q-format) at an activation scale `S`
(recommended `2^14`, i.e. Q4.14); matmuls run in FP64 (bit-exact at these
magnitudes, accumulators below `2^53`) or Int64, then quantize. `[Implemented]`

Overflow is the central numerics constraint: values must stay below the
Goldilocks wrap, which bounds scales and contraction depths. It is also why the
older "rsqrt only sound at `2^12`" concern was retired: the algebraic rsqrt
bracket in the current RMSNorm claim replaces the rsqrt table. The softmax mask
is handled by the protocol's public attention mask.

Accuracy (Llama-2-7B simulation): at the recommended config, top-1 match about
99% `[measured]`; quantization adds about +0.33 bits/token to the UI bound over
the FP baseline, concentrated in a few outlier-channel prompts; error comes from
intrinsic per-block injection, residual-stream accumulation (about 7 to 13%), and
bilinear amplification at `silu(gate)·up` and `softmax(QK)·V`. The precision tax
sits about 25× above the FP16-vs-FP32 floor. Headroom: per-channel weight scales,
an FP8 or NVFP4 reference, multi-segment tables.

---

## 10. Status and roadmap

### Implemented (runs today)

- The streaming prover (fast and sound modes) and the claim types in §3.
- The Rust verifier (six checks), difftested against the Python reference.
- The 400B Maverick end-to-end run (fast mode) and the 7B 1000-token run.

### In flight

- Verifier streaming and per-family eval (`feat/verifier-family-eval`):
  row-streaming `lin_col`, the per-family seam, typed parse, difftest, needed to
  verify the full model within memory and to reach the sound `T=80` config.
- Sound-grade full run (`T=80`, four rounds): estimated about 32 h prove,
  currently bounded by verifier memory at full scale.

### Planned and known gaps (these block the public claim)

- Prover q_lin family-object reorg and even/odd multiply: design notes only.
- Token binding (§1): design only.
- Security: `master_seed` is a fixed stub (`b"\x42"*32`), so blinding rows are
  derivable, which is safe for correctness tests but not for confidential proofs;
  the fix is a CSPRNG seed plus a real adversarial or interactive transport. No
  soundness audit of the claim types yet.
- EmbeddingLookup is approximate: only the referenced vocab rows are
  materialized; a full-vocab fingerprint check is not yet landed (soundness
  note).
- NVFP4 intermediates, multi-GPU parallelization, a CUDA verifier, a wider column
  format, and stream-one-claim-at-a-time (up to 4× by not running the proof up to
  four times).

Bottom line on soundness: the headline end-to-end result is fast-mode,
test-grade (`T=4`), with ZK not yet secured. The full hiding-and-soundness claim
(sound `T=80` four-round protocol, CSPRNG blinding, an audited claim set, and
token binding) is not yet met.

---

## 11. Further reading

Although this document is self-contained, the deep dives are:

- Unexplained-information paper (PDF, this repo): the UI formulation and security
  analysis.
- `design-feasibility.md`: the protocol spec and the analytical cost model (MoE
  routing, Maverick example). Draft; see §12.
- `pipeline/CLAIM_SPECS.md`: claim-type definitions (RmsNorm, Softmax, SiLU, MoE).
- `analysis/prover-optimization-investigation.md`: measured prover-time breakdown
  and optimization options.
- `analysis/qlin-family-object-reorg.md`, `analysis/qlin-evenodd-fused-multiply.md`:
  the prover q_lin reorg design.
- `analysis/verifier-streaming-architecture.md`,
  `analysis/verifier-constraint-family-streaming.md`: the streaming verifier.
- `analysis/witness-scaling-measurement.md`: measured scaling vs context.
- `analysis/full-model-v1-design.md`: the verified Maverick result.
- `analysis/quantization-evaluation.md`: the per-token quantization study.
- `analysis/nvfp4-matmul-predictability.md`: the FP4-matmul predictability test.
- `analysis/token-binding.md`: the token-binding construction.
- Code: `verifier-rs/` (verifier), `pipeline/core.py` (streaming prover),
  `pipeline/demo_llama7b.py` and `pipeline/demo_maverick_moe.py` (examples),
  `cuda/` (Goldilocks, NTT, BLAKE3 kernels).

---

## 12. Document status and known inconsistencies

This doc follows the measured or latest source where older docs are stale:

- `design-feasibility.md` is a draft and partly superseded. Its S²-witness
  coefficient (about 80/block) is roughly 10× too low vs the measured 840/block,
  so its long-context cost projections are optimistic. Its §3.4 "single big LogUp
  table" approach (and the "rsqrt only at `2^12`" concern) is superseded by the
  per-op constructions in `CLAIM_SPECS.md`. Its §4 numbers are analytical
  field-op counts, not measured throughput.
- The verifier streaming docs' "not yet implemented / proposal" headers are out
  of date; that work is partially implemented on `feat/verifier-family-eval`
  (treated as in flight here).
- `improvements-roadmap.md` (2026-05-28) predates the Maverick result and the
  scaling and verifier work; its status baseline is stale, but its security-gap
  list remains valid.
- `CLAIM_SPECS.md` is incomplete (non-arith claims only).
- Do not overstate soundness: the headline run is `T_QUERIES=4` test-grade, not
  sound `T=80`; even `T=80` is only about `2⁻³³` (§5).
