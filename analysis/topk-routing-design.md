# Top-k routing with output-side gates: design note (2026-09-19, revised after review)

Generalizes the mixture-of-experts routing proof from Maverick's top-1
(`prover/routing_claim.py`, `prover/routed_projected.py`, appendix
`analysis/appendix-moe-routing.md` §B.3–B.7) to the DeepSeek-V3 family:
Kimi K2 (8 of 384, one shared expert, gate scale 2.827) and Kimi K3 (16 of
896, two shared experts, gate scale 1.0, experts in a 3,584-wide latent).
A design, not an implementation; every code reference is to the tree at
`41ddcf6`. Revised once after review: the gate bracket gained the range
check that makes it sound, the k slot claims became one stacked claim,
the cost table was recomputed with the profiler's formulas, and the
review's second-order points are folded in where they belong.

Sources for the model figures: Kimi K2's `config.json`
(huggingface.co/moonshotai/Kimi-K2-Instruct) and DeepSeek-V3's
`modeling_deepseek.py` (huggingface.co/deepseek-ai/DeepSeek-V3) for the
gate semantics; Kimi K3's `config.json` (huggingface.co/moonshotai/Kimi-K3)
and technical report §2.3 (github.com/MoonshotAI/Kimi-K3,
`k3_tech_report.pdf`) for the K3 figures, read on 2026-09-19.

## 1. Target semantics

From `MoEGate.forward` in DeepSeek-V3, which Kimi K2 uses unchanged
(`n_group = topk_group = 1`, `scoring_func = sigmoid`, `topk_method =
noaux_tc`, `norm_topk_prob = true`, `routed_scaling_factor = 2.827`);
K3's config states the same method with renormalization on and factor
1.0:

    r[t,e]  = x[t,:] · W_router[:,e]                       logits
    s[t,e]  = sigmoid(r[t,e])                              scores
    select  = top-k over  s[t,e] + b[e]                    b: per-expert bias, SELECTION ONLY
    w[t,e]  = m[t,e] · s[t,e] / Σ_e' m[t,e'] s[t,e']  · c  original scores, renormalized, scaled
    y[t,:]  = Σ_e w[t,e] · FFN_e(x[t,:])  + Σ_shared FFN_s(x[t,:])

Three departures from Maverick: the selection quantity is the *score plus
a bias*, not the logit; there are k winners; and the gate multiplies the
expert *output*. Maverick scales the expert input (`CLAIM_SPECS.md:404-410`),
which is why one silu and one Hadamard serve all experts there
(`appendix-moe-routing.md` §B.7.6). The expert is degree two in its
input, so output scaling is not input scaling, and the collapse is gone:
each selected expert needs its own nonlinearity.

Group-limited routing (`n_group > 1`) is not needed for K2; K3's config
does not name it and its report does not describe one. If K3 has it, it
is a second selection gadget over group sums, not covered here.

## 2. What the tree already proves, and what it does not

- `RoutedProjectedMatmulClaim` proves `Y[t,:] = Σ_e M[t,e] · (X[t,:] W_e)`
  for **any field-valued M**: the compile bands (`routed_projected.py:242-306`)
  and the late Freivalds argument (`analysis/routed-projected-protocol.md:45-53`,
  error 2/|F|) make no structural assumption. One-hot-ness lives only in
  the witness path (`routed_compute`, `:339-357`: argmax and assign) and
  in `RoutingClaim`. The claim proves a sum, with one input X per claim,
  and its fused projection is cached per claim (`_rho_key(c, rho)`).
- `freivalds_combine` proves `y[t,:] = Σ_e m[t,e] · X_e[t,:]` for any m,
  with no booleanity or range on m (`routing_claim.py:285-294, 338-398`);
  its fold accumulates (`:460-465`); the Rust twin is likewise
  structure-agnostic (`handlers.rs:1015-1053`).
- `RoutingClaim` is top-1 in exactly two ways: the cardinality RHS
  literal `1` (`routing_claim.py:198`, `handlers.rs:941`), and the families
  F3/F5 that name *the* chosen logit. Q1 (booleanity), Q2 (m·rt), F1
  (tiebreak) and the composed gap range check (`route_top1`, `:227-272`,
  with the soundness guard `:245-250`) carry over.
- `paired_tlookup` takes any length (`tape.py:735`), so a (T, E) sigmoid
  lookup is one claim over the existing table (`demo_maverick_moe.py:68-93`).

## 3. Design

### 3.1 Selection: the appendix's threshold form, on scores plus bias

Commit per token a threshold τ[t] and a k-hot mask m[t,·]; commit the
selection quantity

    q[t,e]  = s[t,e] + b[e]                     fixed point at scale S_sel
    q̃[t,e] = 2^L · q[t,e] + (E − 1 − e)        tiebreak, L = ⌈log2 E⌉ (9 for 384, 10 for 896)

and prove (appendix §B.3, (2)–(8), the range check by word decomposition
as `route_top1` does today):

    Q1   m ⊙ m = m                                       booleanity        T·E quads
    F2   Σ_e m[t,e] = k                                  cardinality       T linear, public RHS k
    F1   q̃ − 2^L·s − 2^L·b(bcast) = (E−1−e)             tiebreak          T·E linear
    Q2   mq = m ⊙ q̃                                                        T·E quads
    Q3   mτ = m ⊙ τ(bcast)                               τ fanned out by L2_StrideOneToManyScalar
    F6   v = 2·(mq − mτ) − q̃ + τ(bcast)                 = (2m−1)(q̃ − τ)   T·E linear
    R    v ∈ [0, 2^{B_q + L})                            word_extract + range words, as route_top1

The uniqueness lemma is the appendix's (`:86-92`): booleanity and
cardinality make m a k-subset; dominance forces every selected q̃ ≥ τ
and every unselected q̃ ≤ τ; the tiebreak makes all q̃ distinct; so m is
the unique top-k of q̃ and τ is free only inside the gap between the
k-th and (k+1)-th, which certifies the same m and leaks nothing. F3 and
F5 disappear.

Four points the review fixed here:

- **The bias is a model parameter.** The router and the experts sit
  behind the weight and enrollment roots; publishing b would expose about
  23 k parameters across K2's 60 layers. The design commits b as a
  persistent per-layer variable (E slots) and broadcasts it into F1 over
  the T tokens by the one-to-many packet, so F1 stays one linear family
  with the bonus pattern as its only public RHS. A public b would instead
  be a per-slot RHS list, which needs a scalar-list accessor the Rust
  claim reader does not have. Which of the two the deployment wants is a
  policy choice for the group; the committed form is the default here.
- **Selection resolution.** The sigmoid's slope is at most a quarter, so
  ranking on scores at scale 2^12 is about four times coarser than
  Maverick's ranking on logits, and a near-tie flipped against the
  reference selects a different expert, which the information bound
  pays for as a wrong computation, not a rounding. The selection lookup
  therefore uses its own output scale `S_sel` (2^14 or finer): the same
  2^19-entry domain, more bits per entry, and the extra bits appear only
  in the gap width. The gate weights use the S-scale score from a second
  lookup of the same input, or a shift of the finer one.
- **Table domain.** Every logit of every expert is now looked up, not
  only the chosen one, so an honest prover fails on a single logit
  outside the table's ±64 real units. K2's router logit range must be
  measured from the checkpoint before the table is sized; the router
  matmul's 26-bit output width bounds nothing that tight.
- **Gap width.** With q at 14 to 16 bits plus the bias's headroom and L,
  the gap is 24 to 27 bits: two 12-bit or three 9-bit words under the
  unchanged guard `2^{n_words·word_bits} ≤ P − 2^{width}`. The word
  count is fixed after the bias magnitudes are measured; the sentence
  "cheaper than today's three words" is conditional on that.

### 3.2 Gate weights: a division bracket over the k slots

The scores of the k selected experts are the only ones with nonzero
weight, so the bracket runs over the k slots, T·k relations, not T·E.
With the slot scores `s_i[t]` (defined in 3.3):

    Z[t]        = Σ_i s_i[t]                                T linear
    w_i[t]·Z[t] = C·s_i[t] − rem_i[t]                       T·k quads + linear, C = round(c · S) public
    0 ≤ rem_i[t] < Z[t]                                     range on rem_i and on Z − 1 − rem_i
    0 ≤ w_i[t]  < 2^{15}                                    range on the quotient

The last line is what makes the bracket sound: without it every
remainder in [0, Z) yields some field element w through Z's inverse, and
only one of the Z candidates is the integer quotient; with it the
division algorithm pins (w, rem) uniquely, `w = ⌊C·s/Z⌋` at scale S.
Magnitudes: s ≤ S, Z ≤ k·S, w ≤ C < 2^15, products below 2^31.
**Precondition:** Z ≥ 1. The S-scale table rounds the sigmoid to zero
below about −9 real units, so a token whose k selected scores all round
to zero would make the bracket unsatisfiable; the table floors at one,
which changes no selection and keeps every weight defined.

### 3.3 Expert streams: one stacked claim per weight set

The routed claim proves a sum with one input per claim and the expert is
nonlinear, so the k selected experts need k separate streams. The first
draft used k claims with one-hot slot masks; the review showed that the
witness path resolves every expert shard per claim and the projection
cache is keyed by claim, so k claims read each shard up to k times per
sweep on the loader-bound half of the prove, and would need the bridge's
claim-to-slice map changed. The stacked form avoids all of that:

    x_k  = x stacked k times                     (kT, d): a repetition pin of x, T·(k−1)·d linear
    M_k  = the k slot masks stacked              (kT, E): one-hot rows, Σ over the k blocks = m
                                                 (Q1' booleanity, F2' cardinality 1 per row, block-sum pin)
    G    = routed(x_k, M_k, W_gate)              one claim at T' = kT, output (kT, d_ff)
    U    = routed(x_k, M_k, W_up)
    H    = hadamard(silu(rescale(G)), rescale(U))   on (kT, d_ff)
    D    = routed(H, M_k, W_down)                raw, (kT, d), at scale S²·S
    y    = rescale( freivalds_combine(w_slots, [D_1 … D_k], T, E = k, F = d) )   Σ_i w_i[t]·D_i[t,:], one rescale by 24 bits
    s_i[t] = Σ_e M_i[t,e]·s[t,e]                 the slot scores, quads + rowsum, feeding 3.2

One routed claim per weight matrix, so one shard load per sweep per
matrix, one fused projection, and no change to `bridged_claim_map`.
The slot assignment is the prover's (canonical: ascending expert index),
committed, and it does not change the sum. The combine's raw inputs are
at S³ (h at S, W at S, w at S); combining before rescaling drops the k
per-slot down rescales and pays one T·d rescale by 24 bits after the
sum; the field headroom is about 2^57 at k = 8 against 2^63, to be
checked against the measured magnitudes for K3's k = 16. The routed
witness path is correct as it stands for one-hot rows; the
accumulate-not-assign form is a hardening for any future k-hot mask.

### 3.4 What the Rust verifier gains

One generalized `compile_routing`: the `topk` scalar (scalars serialize
off the dataclass, `protocol.py:385-404`), τ and b as variables, F6 and
Q3 in place of F3/F5, the same word/range composition. The bracket, the
slot masks, the stacked pin and the slot scores compose from existing
claims (`PairedTlookupClaim`, `HadamardClaim`, `LinCombClaim`,
`WordExtractionClaim`, `RangeWordClaim`, the broadcast packets); if the
broadcast-and-rowsum patterns do not fit `LinCombClaim` as it stands, a
`WeightBracketClaim` of a few packets closes it. `freivalds_combine`,
the routed claim's compile and the bridge map do not change.

## 4. Cost, Kimi K2 at S = 1000, per MoE layer

Recomputed with `profiler/claimcosts.py` (routed W = T·J + E·K + 2·T·K +
T + 3E; rescale 5·L; silu 23·L; hadamard with rescale 6·L; combine T·F +
4·E·T + T), d = 7,168, expert hidden 2,048, E = 384, k = 8, T = 1,000,
T' = 8,000:

| part | witness slots |
|---|---|
| stacked-x pin | 57 M |
| gate + up routed claims at T' | 268 M |
| gate + up rescales | 164 M |
| silu, hadamard with rescale | 475 M |
| down routed claim, raw | 91 M |
| combine (raw) + one rescale | 43 M |
| routing gadget, lookups, slot masks, bracket | 4 M |
| **per layer, stacked form** | **1.10 G slots** |
| per layer, k separate claims (first draft's structure) | 1.38 G |
| Maverick's MoE layer, same formulas | 0.41 G |

The stacked form saves the k down rescales; the stacked pin roughly
cancels the saved E·K terms otherwise. Sixty MoE layers give about 66 G
slots, some 530 GB of produced witness at S = 1000 for the MoE part
alone, about one Maverick witness (540 GB measured); with MLA at
d = 7,168 over 61 layers the K2 witness is roughly two to three times
Maverick's. K3 in its 3,584 latent with k = 16 is about 2.5 G slots per
layer, 2.3 times K2's. The five semantic sweeps regenerate all of it,
which is what the witness-cache line addresses, and the routed-output
cache's value grows with k. The enrolled weights under the bridge are
1 T parameters, 2.5 times Maverick's; per proof that is the column
re-derivation pass and, without the two-level enrollment, 21 GB of
opened columns on the wire.

## 5. Soundness and privacy

Selection is exact: booleanity and cardinality are quadratics and one
linear pin; dominance rests on the bit-decomposition range check, which
adds no probabilistic term (`design-feasibility.md:368`) under the same
guard as today. The gate bracket is exact integer arithmetic under its
three range checks. Each routed claim contributes 3/|F|
(`routed-projected-protocol.md:~93`), three per layer; the combine 1/|F|
per token (`routing_claim.py:291-294`). Composed as in
`design-feasibility.md` §3.7.

Privacy is Maverick's: every expert's enrollment is opened alike by the
bridge; the masks, τ, the bias, the weights and the slot assignment are
committed rows; τ's freedom certifies the same mask; the canonical slot
order is a function of the mask. Routing decisions the integer model
makes differently from the fp32 reference are wrong computations, not
roundings, and the finer selection scale is there to make them rare;
their cost shows in the unexplained-information bound.

## 6. Implementation plan and gates

1. `RoutingClaim(topk=k)` with the committed bias: the generalized
   families in Python and Rust, `route_topk` builder with the same
   word/range composition and guard; negatives: a wrong k-subset, a
   non-boolean mask, a τ outside the gap, a tampered bias row, a tie
   broken the wrong way. Toy E = 8, k = 3.
2. The gate bracket over the slots: builder and tests for the rounding
   identity and the three range checks; negatives with rem ≥ Z and with
   a non-quotient w.
3. The stacked pin, the slot masks with their block-sum pin, the
   per-slot chain and the raw combine in a toy builder (`demo/moe_topk.py`);
   byte-identity and Rust ACCEPT gates as `test_shard_streaming.py` does.
4. `claimcosts.py` rows for the new shapes; a `kimi-k2` synth builder; the
   crosscheck entry.
5. Then the K2 loader and driver, the numpy reference against llama.cpp,
   the measured router-logit and bias ranges that fix the tables and
   word counts, and a two-layer real-GGUF proof through the verifier.

Items 1–3 need no GGUF and gate on a toy tape; they are the next pod's
work after session 4.

## 7. Open questions for the group

- Public or committed router bias (3.1): a policy choice.
- Whether K3 uses group-limited routing.
- The witness-cache budget at a K2 witness two to three times
  Maverick's, and the routed-output cache at k = 8.
- Whether the stacked pin should be a claim of its own (one packet
  family) or the embedding-lookup broadcast reused with a repeated index.
