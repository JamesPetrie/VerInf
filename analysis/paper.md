# VerInf: Zero-Knowledge Proofs of Inference for Frontier-Scale Models on a Single Machine

James Petrie

---

---


## 1. Introduction
VerInf verifies how much of the information in a token stream is explained by inference of a committed large language model, in zero knowledge. Its one public quantity is a bound on the unexplained information in the output, the bits that a permitted computation on the measured inputs does not account for. The committed witness at frontier scale exceeds any single machine's memory, so our implementation streams it operation by operation. This is what makes massive proofs feasible on small hardware: VerInf produced a proof of a 1000-token forward pass of Llama-4-Maverick, a 400-billion-parameter mixture of experts with all 128 experts committed per layer, in 14.3 hours on a single consumer NVIDIA DGX Spark, streaming a 7.2-terabyte witness at a peak working set of 83.9 GB. Prior published zero-knowledge inference work reaches models of about 13 billion parameters (Sun, Li, and Zhang 2024), roughly thirty times smaller.


Deployed inference typically runs in floating point and is often nondeterministic across executions. Prior inference proofs require the computation to match an integer circuit exactly, so they attest to a computation contrived to fit the proof; how easily a given serving stack could be made bit-exact is difficult to judge from outside. VerInf requires no changes to the deployment. Rather than reproduce its output, the proof certifies how well a committed integer version of the model predicts it, and the cost of every approximation appears in the reported bound rather than in a failed proof. A deployment that does run deterministically needs no new machinery and simply certifies a tighter bound.


The target use case is sampling packets under high-stakes AI agreements. In the companion framework [cite], a compute operator (the prover) demonstrates to an external party (the verifier) that it runs only permitted workloads, where neither party trusts the other's hardware; the framework gives three confidentiality-preserving ways to perform the recomputation, and this paper builds the zero-knowledge one. Three properties of the setting shape the design. Only randomly sampled outputs are proven, so proving cost amortizes over the workload it deters rather than gating it. Both parties have substantial resources, so slow proving, slow verification, and large proofs are all acceptable, while a small trusted base matters greatly; the demonstrated proof is 93.6 GB and takes hours to check. And the parties are in live contact, so the proof can be interactive: the prover commits before each challenge is drawn and cannot grind over challenges offline, which lets a modest per-challenge soundness level carry real deterrent weight.


The bound itself is a sum of per-token surprisals under a predictor of the deployment's outputs that the prover supplies. By Gibbs' inequality the sum is a valid upper bound on the unexplained information for any predictor, so the choice can be left to the prover, which is best placed to make it accurate; a poor predictor inflates only the prover's own reported number. Beyond the bound and the public claim list, the proof reveals nothing: not the weights, the activations, or the input and output tokens. The bound places an unusual demand on the proof system: the prover must have no freedom it can use to deflate it. Witness slack is a liability for any inference proof, since slack in an intermediate value propagates through the remaining layers in ways that are hard to analyze; for a bound on unexplained information it is directly exploitable, since the prover selects among valid witnesses in the direction that inflates its predicted probabilities. VerInf therefore meets a two-tier requirement: upstream of the logits every claim admits a unique witness, and downstream of the logits every prover freedom provably inflates the reported bound (§2.3).

VerInf uses Ligero (Ames et al. 2017) as its argument system because it is transparent, needing no trusted setup, plausibly post-quantum, with commitments that are hash-based only, and among the simplest constructions available. The trade Ligero makes is proof size and verifier work that grow as the square root of the witness rather than polylogarithmically, which the setting tolerates: verification is occasional and offline, and the verifier is well resourced. A cost model fitted to measured hardware primitives and validated against the demonstrated runs projects that a one-million-token context could be proven in days on an NVL72-class cluster; the projection assumes dense attention in every layer, and windowed or sparse attention reduces the dominant quadratic term in proportion to the layers and span it covers (§8). The verifier recompiles every constraint from the public claim list and checks the proof only against its own derivation, never against constraints supplied with the proof, so its trusted base is short and auditable.

A model is written as ordinary tensor code against a tape, in the style of PyTorch; each operation records a public claim, and each claim type carries the machinery to prove it, some of it using an additional challenge round. Matrix multiplications are proven by Freivalds' check: the prover commits the input and output matrices, the verifier supplies random projection vectors, and the prover shows the projection of the committed output equals the product of the projections of the committed inputs, reducing each matmul to a single short dot product. Nonlinearities such as softmax are proven against public lookup tables with LogUp (Haböck 2022), whose cost amortizes across the many operations sharing a table. Softmax is where the unique-witness requirement trades against efficiency: its per-row shift is not an integer, and constraining it only to a tolerance band would leave the prover a choice among valid witnesses. VerInf instead pins the shift to a unique integer using the monotonicity of the row sum, at the cost of a second table and the bracket constraints (§4.4). The bound is meaningful only for the transcript the run actually produced, so the committed token streams are bound to digests recorded independently at generation time (§2.4); the circuits for this binding are implemented, and demonstrating it end to end against recorded digests is future work.

The demonstrated system has two main limitations. The committed integer model leaves 0.880 bits per token unexplained, explaining about 95% of the information a token from the 202,048-token vocabulary can carry; future work can tighten this by modeling the deployment's floating-point computation more closely. And the public claim list reveals the model architecture, though not the weights; a second proof stage, showing that the verifier's own architecture checks ran and accepted, could hide it (§9).

Our contributions are:
* A proof design that leaves the prover no freedom to deflate the reported bound, with a unique witness pinned at every step of the forward pass.
* A complete implementation on a hash-based proof system (Ligero), with no trusted setup: a streaming prover whose peak memory tracks one operation rather than the witness, and a verifier that recompiles every constraint from the public claim list.
* A demonstration at frontier scale: a proof of a 1000-token forward pass of a 400-billion-parameter mixture of experts, generated in 14.3 hours on a single consumer machine by streaming the 7.2-terabyte witness at a working set of 83.9 GB, and accepted by an independently implemented verifier.
* A cost model fitted to measured hardware primitives, projecting that a million-token context with dense attention in every layer could be proven in days on an NVL72-class cluster (§8).

The paper is organized as follows. Section 2 states what VerInf proves and what the bound requires of the proof system. Section 3 compares with prior work. Section 4 compiles a model to constraints, Section 5 gives the Ligero argument and its soundness, and Section 6 describes the streaming prover and the small verifier. Section 7 reports the results, Section 8 the cost model and scaling, and Section 9 the limitations and future work.
## 2. What VerInf proves

VerInf attests to a single public quantity: a bound on the unexplained information in a committed output, given a committed model and committed inputs. The quantity and its security analysis are developed in the companion paper [cite]; this section recaps the definition and the estimate the proof certifies (§2.1), states the trust model (§2.2), and then sets out what the bound demands of the proof system itself (§2.3), the requirements the claim design of §4 and Appendix B is built to satisfy. Section 2.4 describes how the committed token streams are anchored to the transcript the run actually produced.

### 2.1 The unexplained-information bound

Model the declared channel as discrete: a sequence of input tokens and a sequence of output tokens. Let $x$ denote the measured inputs and $o$ the measured outputs, and let $D$ be the declared computation, which on input $x$ produces $D(x)$, including intermediate values such as logits. The unexplained information is the information in the output not determined by $D(x)$. Any covert payload must hide there: the more of the output $D(x)$ accounts for, the less room remains.

The same computation can produce different outputs across runs, so let $O$ be the random variable over outputs the hardware could produce given $D(x)$. The unexplained information is the conditional entropy

$$U(O) = H(O \mid D(x)).$$

The verifier cannot evaluate $U(O)$ directly, since it depends on the true distribution over hardware outcomes. But for any predictor $Q(O \mid D(x))$, Gibbs' inequality gives an upper bound,

$$U(O) \le -\,\mathbb{E}_P\!\left[\log_2 Q(O \mid D(x))\right],$$

valid for every $Q$. Factoring $Q$ autoregressively and evaluating on the measured output $o$ gives the estimate the proof certifies, a sum of per-token surprisals,

$$U(o) = -\sum_i \log_2 Q_i\!\left(o_i \mid D(x),\, o_{\lt i}\right).$$

A better predictor only tightens the bound, so the choice of $Q$ is left to the prover, which is also best placed to make it accurate: a poor $Q$ inflates only the prover's own reported number. Our prototype models hardware nondeterminism as Gaussian noise on the logits,

$$Q_i(o_i) \propto \exp\!\left(-(v^{*} - \ell_i)^2 / \sigma^2\right), \qquad v^{*} = \max_i \ell_i,$$

with $\ell_i$ the logit for token $i$ and $\sigma$ calibrated empirically. Only $U(o)$ is revealed; the weights, inputs, and output tokens stay committed and hidden. The full derivation, the extension to non-zero-temperature sampling, and the security analysis are in the companion paper.

In this paper the declared computation $D$ is the committed integer model of §4, and the cost of that approximation appears directly in the bound: the worse the integer model predicts the deployment's tokens, the larger the reported $U(o)$. The same accounting covers every design choice downstream, from the quantization scales to the noise model, so tightening the bound is an engineering trade rather than a soundness question (§9).

### 2.2 Trust model

The two parties want different guarantees. The prover wants confidentiality: the model weights, the activations, and the input and output token streams must not leak. The verifier wants soundness: the reported $U(o)$ must be a genuine upper bound for the committed tokens under the committed weights. Their interface is the public claim list, a statement of what kind of computation was performed (§4); it reveals the model architecture, though not the weights, and hiding the architecture as well is future work (§9). The proof reveals nothing beyond $U(o)$ and the claim list, so a dishonest verifier learns nothing more, and a dishonest prover cannot produce an accepting proof for a deflated bound except with the soundness error of §5. Each side trusts only its own code: the verifier's trusted base is short (§6.2), and a fault anywhere on the prover's side can only cause a proof to fail, never to falsely verify.

### 2.3 What the bound requires of the proof system

For $U(o)$ to be an upper bound, the prover must have no freedom it can use to deflate it. The requirement takes two forms, split at the logits. Upstream of the logits, in the forward pass that produces them, every claim must admit exactly one satisfying assignment. Slack in an intermediate value propagates through the remaining layers in directions that cannot be analyzed, so any freedom there could cascade into the token probabilities arbitrarily. Downstream of the logits, in the short computation from logits to the reported bound, freedom is permitted provided every free direction increases the reported value. This weaker property can be established directly, because the downstream computation is a few steps of analyzable arithmetic. The construction meets the first requirement claim by claim (Appendix B.2 to B.5) and the second by pushing every rounding in the surprisal computation upward (Appendix B.6).

Softmax is where the first requirement trades against efficiency. Its row-wise normalization uses a per-row shift (the log-sum-exp) that is not an integer, and recent work speeds up the proof by leaving the shift unverified, checking only that the normalized outputs sum to the expected total within a tolerance for quantization error (Sun, Li, and Zhang 2024). The tolerance admits several shifts, which give slightly different output probabilities. For reproducing a fixed output this is arguably adequate, provided accumulated tolerances cannot be steered through the remaining layers, which has not been established; for a bound on $U$ it is immediately exploitable, since the prover selects among the valid witnesses in the direction that inflates its predicted probabilities. VerInf pins the shift to a single integer instead, using the monotonicity of the row sum in the shift, at the cost of a somewhat larger proof (§4.4, Appendix B.3).

A second requirement is causality: the prediction of each output token must depend only on earlier tokens, so a later token cannot be used to lower the surprisal of an earlier one. This is enforced by the public attention mask compiled into the claims.

These are properties of the claim graph, which is public and recorded with the proof. The verifier checks the proof against this claim list; that the claim list itself has the required structure, for example that each weight is read only in a forward pass and never updated so that no gradient step is hidden in the computation, is established by auditing it. Confirming this automatically, by static analysis of the claim graph, is future work (§9).

### 2.4 Anchoring to the real transcript

The bound is conditioned on the input and scored on the output, so it is meaningful only when both token streams are the ones the run actually used. Inside the proof they are hidden witness, which on its own does not tie them to any external record: a prover could commit a lower-surprisal transcript and deflate the bound, or condition on a fabricated prompt and certify nothing. VerInf closes this by binding both committed streams to digests recorded independently at generation time: a recorder hashes the encrypted token streams as they pass and receives a commitment to the key material, fixed with the request before the response exists, and the proof shows that the committed tokens encrypt and hash to the recorded digests, without revealing them (Appendix E). The record itself must come from a process the verifier trusts, independently of and prior to the proof; in the setting of §1 this is the verifier's network-boundary hardware.


## 3. Related work

**Zero-knowledge proofs of inference.** *[TODO: survey and cite additional zero-knowledge inference work beyond zkLLM, e.g. the broader zkML line and any post-2024 LLM inference proofs, and position each against scale, statement proven, and commitment assumptions.]* zkLLM (Sun, Li, and Zhang 2024) is the closest prior work and the largest published zero-knowledge proof of LLM inference. It proves transformer inference with a sumcheck-based argument and a Hyrax-style commitment, with a tailored proof of attention, and reports about 13 minutes for LLaMA-2 13B at 2,048 tokens on an A100. VerInf differs in three ways. It reaches roughly thirty times the parameter count, with a full prover and verifier run end to end. It proves a different statement: zkLLM, like other inference proofs, requires the computation to match its integer circuit exactly, so it attests to a computation contrived to fit the proof, where VerInf proves that the committed output is well explained by the committed integer model and bounds what goes unexplained, which is what lets it target a deployment as it runs (§1). And its commitments are hash-based only, needing no trusted setup and remaining plausibly post-quantum, where the Hyrax commitment rests on the discrete-logarithm assumption that a quantum adversary breaks.

The zkLLM witness also leaves the prover freedom of the kind §2.3 forbids, in at least three places: the softmax shift is constrained only by a tolerance band on the row sum, the rule for rounding real-valued table entries to integers is unspecified, and some setup parameters are prover-chosen. For reproducing a fixed output this is arguably adequate, provided accumulated tolerances cannot be steered through the remaining layers, which has not been established; for a bound on the unexplained information each freedom is immediately a lever, since the prover selects among valid witnesses in the direction that inflates its predicted probabilities and deflates the reported bound. VerInf closes all three: the tables, scales, and noise parameters are public constants of the claim list rather than prover choices, every table entry is computed by a deterministic rounding rule the verifier reproduces, and each tolerance band is replaced by a bracket that pins its value to a unique integer (§4.4, Appendix B).

**Verified inference with a trusted verifier.** Two recent works verify inference in the unilateral setting, where the verifier is trusted and sees plaintext. Rinberg et al. (2025) verify inference to detect exfiltration of model weights, and Karvonen et al. (2025) verify inference despite nondeterminism. VerInf shares with this line the per-token surprisal measure under a logit-noise model, but targets the bilateral setting where the verifier trusts neither the prover's hardware nor its software, and where confidentiality must hold against the verifier as well: the recomputation is replaced by a zero-knowledge proof, and the verifier learns only the bound.


## 4. From a model to constraints

VerInf compiles a forward pass into a flat list of linear and quadratic constraints over a committed witness, which the Ligero argument of §5 then proves all at once. This section describes the compilation: how a model is written (§4.1), how its values are represented as field elements (§4.2), how matrix multiplication (§4.3), the nonlinearities (§4.4), and mixture-of-experts routing (§4.5) become constraints, and how the unexplained-information bound is added as further claims (§4.6).

### 4.1 The claim language

A model is written as ordinary tensor code against a tape, in the style of PyTorch. Each `tape.<op>` call records one claim, a public statement of what was computed, and returns a handle; handles overload the usual `@`, `*`, and `+`. A few lines of an attention block read:

```python
norm = tape.rmsnorm(x, d=d, s=S, eps_int=EPS_INT)
g    = tape.hadamard_broadcast(norm, rms_w, SEQ=SEQ, d=d)
q    = tape.matmul(g, W_Q, s_a=S, s_b=S, s_out=S)
sc   = tape.matmul(qr, kr, transpose_b=True)   # attention scores
sm   = tape.softmax(sc, M=SEQ, s_x=S)
out  = tape.matmul(sm, v)
```

The same claim list drives both witness generation, computing each value in the forward pass, and the constraint compile, emitting the linear and quadratic constraints that define a correct computation. Verification recompiles those constraints from the public claim list and checks the proof against them, so it depends only on the claims, never on constraints supplied with the proof. Everything a claim depends on beyond the committed witness, the lookup tables, the quantization scales, and the noise-model parameters, is a public constant of the claim list, fixed before any challenge is drawn; nothing the prover chooses privately enters a constraint (§3). A model built from existing operations needs no change to the prover or verifier; only a genuinely new operation needs a new claim type. The unexplained-information claims were added this way, reusing existing table lookups and elementwise steps (§4.6).

### 4.2 The fixed-point witness

The proof works over a finite field, so every value is quantized to a 64-bit fixed-point integer at a scale $S$ (we use $S = 2^{14}$). The prover commits integers; the verifier checks them with field constraints, and lookup tables handle the nonlinearities. Matrix multiplications are carried out in Int64 or in FP64, which is bit-exact at these magnitudes because every accumulator stays below $2^{53}$, then requantized.

The central numeric constraint is overflow: every committed value and every intermediate product must stay below the Goldilocks modulus, which bounds the usable scales and contraction depths. This is what fixes the scale choices and what forces a rescaling step into some of the nonlinearity claims when an input arrives at a higher scale than the claim can absorb (§4.4). The accuracy cost of the representation is priced by the bound (§2.1).

If a value nevertheless exceeds its budget, one of two things happens. Where it feeds a word decomposition or range check, a wrapped field element has no valid decomposition into range-checked words and the proof rejects (Appendix B.1). Elsewhere the constraints are field identities and the proof accepts, but what it then attests is the wrapped field computation: still a deterministic function of the committed inputs with a unique witness, so the requirement of §2.3 is unaffected, and the bound remains genuine for that computation; in practice wrapped logits predict the deployment's tokens poorly and the divergence surfaces as a large $U(o)$. Care is needed only where a uniqueness or exclusion argument itself assumes a magnitude bound, since a bracket whose operand is not independently bounded can admit a wrapped second solution. Every such operand therefore carries a range check or a written width argument tracing to one; Appendix B applies this discipline claim by claim, and it is most load-bearing in the surprisal claims of B.6, where slack is deliberately permitted and the safe direction must be established in the field (§9).

### 4.3 Matrix multiplication via Freivalds

Checking $C = AB$ of shapes $(m, k)$ and $(k, n)$ entrywise would need $mnk$ product constraints, which is infeasible at frontier scale. VerInf uses Freivalds' randomized check instead, with a projection on each side: the verifier samples $\rho \in F^n$ and $\lambda \in F^m$, and the identity $C = AB$ is checked through its projection $\lambda^\top C \rho = \lambda^\top A B \rho$. We call this two-sided form double Freivalds. The prover commits three short projections, $y = B\rho$ and $u = \lambda^\top A$ (each of length $k$) and their pointwise product $p[i] = u[i]\, y[i]$, so the claim emits about $2k + 1$ linear constraints and $k$ quadratic constraints, and one length-$k$ dot product stands in for the whole matmul. The soundness error is $2/|F|$ per matmul, negligible at $|F| \approx 2^{64}$.

|  | naive per-element | single Freivalds | double Freivalds |
|---|---|---|---|
| auxiliary witness slots | $mnk$ | $mk + k$ | $3k$ |
| quadratic constraints | $mnk$ | $mk$ | $k$ |
| soundness error | exact | $1/\vert F\vert$ | $2/\vert F\vert$ |

Relative to the usual single-projection form, the second projection reduces the auxiliary witness and the quadratic-constraint count by further factors of $m/3$ and $m$, at the cost of doubling an already-negligible soundness error. VerInf uses the double form for every matmul in the model.

### 4.4 Nonlinearities via lookup tables

Each nonlinearity is verified against a public table with a lookup argument. LogUp (Haböck 2022) reduces a lookup to an identity over multiplicative inverses: to prove a functional relationship $y_i = f(x_i)$, the table is a set of input-output pairs $(x^{(j)}, f(x^{(j)}))$, each query and table entry is folded into one field element with a random challenge $\alpha$, and the check is the single identity

$$
\sum_{i} \frac{1}{\beta + x_i + \alpha\, y_i} \;=\; \sum_{j} \frac{m_j}{\beta + x^{(j)} + \alpha\, f\!\big(x^{(j)}\big)},
$$

over random $\alpha, \beta$, with multiplicities $m_j$ counting how often each table entry is queried, tallied during witness generation and committed before $\alpha, \beta$ are drawn. Because $\alpha$ is random, a query matches a table entry only when both its key $x_i$ and its value $y_i$ agree, so the identity holds if and only if every committed $(x_i, y_i)$ is a genuine input-output pair of $f$: one lookup certifies both that $x_i$ is in range and that $y_i = f(x_i)$. The prover commits the inputs, the outputs, the inverses, and the multiplicities; a per-table settlement, synthesized automatically in the compile, samples the lookup challenges and emits the cross-claim sum identity, and the constraints feed the same linear and quadratic tests that handle the arithmetic (§5). Table entries are public and computed by a deterministic rounding rule, so the verifier evaluates the right-hand side itself. The three nonlinearities in the model are arithmetized as follows.

**RMSNorm.** Rather than look up a reciprocal square root, the claim pins the rsqrt scalar $y$ algebraically, with two quadratic brackets that force $y$ to the unique integer with $y^2\, S_{\text{total}} \ge \text{magic}$ and $(y - 1)^2\, S_{\text{total}} \lt \text{magic}$, where $\text{magic} = d\, s^4$, and the bracket slacks are word-decomposed and range-checked so no large table is needed. The broadcast multiply is folded by Freivalds rather than committed cellwise.

**Softmax.** The claim pins the per-row log-sum-exp shift with a two-table monotonicity bracket, rather than leaving it constrained only by a tolerance on the row sum. Two exponential tables $T_A$ and $T_B$, the second the first shifted by one integer unit $\delta$, are computed from the same rounded expression, so $T_B[k] = T_A[k-\delta]$ bit-for-bit and the row sums satisfy $s_2(c) = s_1(c-\delta)$ exactly. Since $s_1(c)$ is monotone non-increasing in the shift $c$, bracketing it between $s_1 \le s_y$ and $s_2 \ge s_y + 1$ pins $c$ to the unique integer where $s_1$ crosses $s_y$, with no tolerance band. Two paired lookups against $T_A$ and $T_B$ then certify the outputs. Pinning the shift this way removes the deflating freedom of §2.3, at the cost of a second table and the bracket constraints. An optional saturating mux sizes the table to the nonzero region of the exponential.

**SiLU.** The input is split into sign and magnitude, the magnitude is decomposed into a low word that indexes the table and high words that detect saturation, and a paired lookup returns the table value; when the high words are nonzero a mux replaces the lookup with the saturated value (the input itself for large positive inputs, zero for large negative).

When an input arrives at a higher scale than a claim can absorb without overflow (§4.2), the claim emits a shared rescale block, a word decomposition that drops the low bits, before its main constraints. The full per-claim constraint listings are in Appendix B.

### 4.5 Mixture-of-experts routing

In a top-1 MoE layer each token routes to one expert by routing logit. The routing claim pins a one-hot mask $m$ to the argmax of the routing logits, made unique by a public tiebreaker that packs the expert index into the low bits of each logit so no two are equal. Booleanity ($m_e^2 = m_e$) and cardinality ($\sum_e m_e = 1$) force $m$ to be one-hot, and a range-checked gap constraint forces its support to be the argmax: if the mask selected a non-maximal expert, the gap to the true maximum would be a negative field element, which cannot be recomposed from the range-checked words, and the proof rejects. A masked-combine claim then forms the layer output as $\sum_e m_e\, \text{expert}_e(x)$.

All $E$ experts' streams are committed even though only one is active. This is a hiding requirement, not an inefficiency: a witness that committed only the active expert would reveal the routing decision, so the inactive experts are committed and zeroed by the mask. (A top-1 simplification applies the elementwise nonlinearity once after the masked sum rather than per expert, since the sum already selects the chosen expert's stream; this reduces the committed intermediates without changing what is proven.)

### 4.6 The unexplained-information bound as claims

The bound of §2.1 is computed from the LM-head logits by four claims per output position, reusing the gap gadget of the routing claim, the paired table lookup, and the elementwise steps; Appendix B.6 gives the full specification, and Appendix E binds the same committed tokens to the digests recorded at generation time (§2.4). Its soundness property is the weaker downstream one of §2.3: the witness is deliberately not unique, and instead every prover freedom provably inflates the reported value. Normalization is where the asymmetry between the two requirements pays. Softmax pins its per-row shift exactly, because upstream slack is unanalyzable, while the bound replaces normalization with a one-sided logarithm pin, because downstream slack can be shown to only inflate. The output tokens enter only as committed witness consumed by these claims; they never appear in the public claim list, so the proof reveals the bound and nothing about which tokens were produced. Because the bound folds onto the logits inside the proof, certifying it costs little beyond the forward pass it sits on top of.


## 5. The Ligero argument and soundness

Section 4 produced a witness and a flat list of linear and quadratic constraints. This section gives the argument that proves them all at once: the Ligero construction and our parameters (§5.1), the zero-knowledge masking (§5.2), the four interactive rounds across which the stages run (§5.3), and the soundness analysis (§5.4).

### 5.1 Commit, test, open

Ligero (Ames, Hazay, Ishai, and Venkitasubramaniam 2017) is a zero-knowledge argument built from Reed-Solomon codes and a Merkle commitment, with no trusted setup and security resting only on the collision resistance of a hash function. It proceeds in three stages. In the commit stage it arranges the committed values as a matrix whose rows are Reed-Solomon codewords and hashes the columns into a Merkle tree, whose single root binds the whole witness. In the test stage it folds the constraints into a few short polynomials with random combiners: a linear test that a system $Ax = b$ holds, a quadratic (Hadamard) test that a system of pointwise products holds, and an interleaved Reed-Solomon test that every row is close to a codeword. In the open stage the verifier names a random subset of columns; the prover reveals them with Merkle paths, and the verifier checks that they hash to the root and are consistent with the test polynomials. Soundness comes from Reed-Solomon distance: any inconsistency appears in a constant fraction of columns, so a few random column checks catch it with high probability.

The trade Ligero makes is proof size and verifier work, both growing with the witness rather than logarithmically, in exchange for a simple construction, a small trusted base, and transparent, plausibly post-quantum security. In the setting of §1 this is the right trade: verification is occasional and offline, the verifier is well resourced, and a small auditable verifier with no trusted setup matters more than proof size.

**Parameters.** VerInf works over the Goldilocks field, $|F| = 2^{64} - 2^{32} + 1$, which admits fast number-theoretic transforms and fits the fixed-point magnitudes of §4.2 without wraparound. The constants are

$$\mathrm{ELL} = 8192, \quad \mathrm{K\_DEG} = 16384, \quad \rho = 4, \quad \mathrm{N\_LIG} = \rho \cdot \mathrm{K\_DEG} = 65536,$$

where `ELL` is the number of constrained message slots per row, `K_DEG` the polynomial degree bound, $\rho$ the Reed-Solomon inverse rate, and `N_LIG` the codeword length (the number of columns). The number of columns opened, `T_QUERIES`, is a deployment choice that sets the soundness level (§5.4); the demonstrated runs are reported with their values in §7. The hash is BLAKE3.

**Commit.** The witness is laid out as a matrix, each variable occupying a contiguous block of rows of `ELL` slots. Each row is Reed-Solomon encoded: the `ELL` message values, padded with `K_DEG − ELL` random slots for zero-knowledge, are interpolated to polynomial coefficients (an inverse NTT of length `K_DEG`) and evaluated on a coset of length `N_LIG` (a forward NTT) to give the codeword. The codeword columns are hashed into a BLAKE3 Merkle tree whose single root binds the witness. The witness is committed as three stacked blocks, sharing one column-query set: the weights $R_W$, committed once and persistent across queries; the per-prefill activations $R_{p1}$; and the per-query auxiliary values $R_{p2}$, such as the Freivalds projections and LogUp inverses, which depend on challenges and so are committed after them.

**Test.** Random combiners fold all constraints into three short polynomials: an interleaved Reed-Solomon test that every row is close to a codeword, a linear test that aggregates the entire linear system $Ax = b$ into one weighted combination (Freivalds applied to the constraint system, caught with probability $1 - 1/|F|$), and a quadratic test that aggregates all the pointwise products. The linear test dominates the prover's cost (§6.1).

**Open.** The verifier names `T_QUERIES` random columns; the prover reveals them with Merkle paths. The verifier re-hashes the opened columns against the root and recomputes the three test polynomials at those columns, checking they match.

### 5.2 Zero-knowledge

Two mechanisms hide the witness. Each row carries `K_DEG − ELL` random padding slots, so the values revealed at the opened columns are evaluations whose dependence on the witness is masked; this sustains a finite number of openings per persistent commitment (8,192 distinct columns at these parameters) before the random budget is spent, after which the commitment is refreshed: the same weights are re-committed under fresh randomness, and a linking proof, a per-slot equality claim between the two weight blocks, ties the new commitment to the old (the ledger that decides when to refresh is future work, §9). Each of the three tests additionally mixes in a structured blinding row that leaves the verifier's checks unaffected while hiding the witness-derived part of the test polynomial. The challenges are drawn from a seed; with a randomly generated seed the proof is hiding, and the demonstrated runs use a fixed seed for reproducibility, which a one-line change replaces.

### 5.3 Protocol rounds

The three stages of §5.1 run across four interactive rounds, so that each commitment is fixed before the verifier draws the next challenge. After the tape records the claims, the prover and verifier exchange:

| Round | Prover sends | Verifier replies |
|---|---|---|
| 1 | Commitment to the intermediate witness ($R_{p1}$: activations, routing masks, normalization auxiliaries) | Per-claim challenges (Freivalds projections $\rho, \lambda$; LogUp $\alpha, \beta$) |
| 2 | Commitment to the challenge-dependent auxiliary witness ($R_{p2}$: Freivalds projections, LogUp inverses) | Combiner challenges for the test polynomials |
| 3 | The folded interleaved Reed-Solomon, linear, and quadratic test polynomials | The random column set |
| 4 | The named columns, with Merkle paths | ACCEPT or REJECT |

Splitting the commit stage across rounds 1 and 2, with the auxiliary witness committed only after its challenges, and fixing each commitment before the next challenge, is what prevents the prover from fitting a witness to a challenge it has already seen. The interactivity assumed available in §1 is used here: because the prover commits before each challenge is drawn, it cannot search over challenges offline, so a per-challenge soundness level bounds the chance of passing a live attempt rather than the best of many.

### 5.4 Soundness

The total probability that a cheating prover is accepted is bounded by the sum of the Ligero test errors and the per-claim reduction errors:

$$\varepsilon \;\le\; \underbrace{\varepsilon_{\text{IRS}} + \varepsilon_{\text{lin}} + \varepsilon_{\text{quad}} + \varepsilon_{\text{field}}}_{\text{Ligero}} \;+\; \sum_{\text{matmuls}} \frac{2}{|F|} \;+\; \sum_{\text{LogUp}} \frac{M + T + 1}{|F|},$$

with the range checks contributing nothing (bit and word decompositions are exact). The Ligero side is dominated by the interleaved Reed-Solomon term $\varepsilon_{\text{IRS}} = (1 - 1/\rho)^{\mathrm{T\_QUERIES}} = (3/4)^{\mathrm{T\_QUERIES}}$; reaching $2^{-s}$ needs `T_QUERIES` $\approx 2.4\,s$. The field term floors at $\varepsilon_{\text{field}} \approx \mathrm{N\_LIG}/|F| \approx 2^{-48}$, and the Freivalds terms are negligible at $|F| \approx 2^{64}$. The binding term in practice is the LogUp $(M + T + 1)/|F|$, where $M$, the number of lookup queries in a batched instance, reaches $10^{10}$ at frontier scale, giving a per-instance error around $2^{-28}$ to $2^{-30}$; it is tightened where needed by parallel repetition of the LogUp challenge.

`T_QUERIES` is therefore the dial. Opening more columns drives the interleaved Reed-Solomon term down geometrically until, past roughly $\mathrm{T\_QUERIES} = 80$, the LogUp term becomes the binding one and raising soundness further also requires parallel repetition of the LogUp challenge. The demonstrated configuration sits below that point, at a per-challenge bound of about $2^{-16.6}$ (§7). In the setting of §1 this is a meaningful level rather than a compromise: the protocol is interactive, so the bound holds per live attempt with no offline grinding, and the intended parties are deterred by any non-negligible chance of being caught even once, since a detected violation is diplomatically costly. The relevant quantity is the probability of escaping detection on a given challenge, not an asymptotically small forgery probability, and the configuration is a deployment choice trading proof size and verifier time for a smaller error rather than a property of the construction.


## 6. Implementation

### 6.1 The streaming prover

The prover is implemented in CUDA: the field arithmetic, the Reed-Solomon transforms, and the Merkle hashing of §5.1 are GPU kernels over the Goldilocks field, compiled for the local device on first use.

At frontier scale the committed witness is far larger than any single machine's memory: for §7's 400B-parameter run it is about 7.2 terabytes ($9.0{\times}10^{11}$ field elements, Appendix A). The prover therefore streams the witness, committing one operation at a time: each row is Reed-Solomon encoded, folded into the column hashes and into running accumulators for the test polynomials, then freed before the next is computed. Because the Merkle tree is built over columns and the test polynomials are linear and quadratic accumulations over rows, no later step needs the full encoded matrix resident, so peak memory tracks the working set of a single operation rather than the witness or the proof. Streaming bounds working-set memory only; proof size and proving time are unaffected. The same row-by-row structure admits parallelization across GPUs, with transform work split across rows and the column hashes partitioned across nodes, each accumulating its assigned columns (§9).

[Todo: fix writing] Peak memory is therefore set by the largest single working set. At long context that is softmax's: its witness grows quadratically with context length, and the implementation currently proves each softmax over its full score matrix at once. This is a choice rather than a necessity, since the rows could be split into chunks and proven piece by piece, capping the working set at the chunk size. One small piece of state also persists across the whole proof: the lookup argument of §4.4 needs per-table counts of how often each entry was queried, kept as one resident histogram per table and fixed in size by the tables.

Appendix C specifies how the sparse constraint system is regenerated and evaluated against the streamed witness during the test folds; the resulting cost profile is analyzed in §8.

### 6.2 The verifier

The verifier is a Rust program that reads a proof and decides whether to accept it. It shares no code with the prover: it recompiles the constraint system from the public claim list (§4.1) and checks the proof against its own derivation, never against constraints supplied with the proof. Its work is, first, to confirm that the claim list meets the requirements of §2.3, and then to check, at the columns opened in the final round (§5.3), that they re-hash to the committed Merkle root and that the three test polynomials of §5.1, recomputed at those columns, match the prover's. A proof is accepted only if every check passes.

The trusted base comprises the field arithmetic, the hash and Merkle check, the challenge derivation, the constraint compile, and these checks. Proof parsing and all other handling sit outside it, since a malformed or dishonest value fails a check and the proof is rejected. The verifier depends on three crates (`blake3`, `rayon`, `serde_json`) and is differential-tested bit-for-bit against a Python reference implementation. It needs no GPU, but at full model scale it needs a large-memory host, because the compiled constraint system grows with the witness; the heaviest step, the linear identity at the opened columns, is dense field arithmetic over the constraints, parallelizes across cores, and is the natural candidate for a GPU port (§9).

This division of labor follows the trust model of §2.3: only the verifier requires review, and prover-side code can be modified freely without enlarging the trusted base, since a fault there causes a proof to fail rather than to verify falsely.


## 7. Results

We report two runs, both produced by the streaming prover of §6.1 on a single NVIDIA DGX Spark (GB10, 121 GB unified memory) and checked by the verifier of §6.2.

**Llama-4-Maverick, 1000 tokens, every token hidden.** The full 48-layer, 400B-parameter mixture of experts, with all 128 experts committed per MoE layer, proven in the four-round protocol of §5.3 with 40 columns opened. The transcript is a 500-token prompt and the model's own 500-token greedy continuation; all 1000 tokens are hidden, entering the proof only as committed one-hot indicators, and the indicator rows are shared between the input selection and the surprisal claims of §4.6, so the scored tokens are, by shared committed variable, the tokens the model consumed. The surprisal claims run inside the proof and the bound is its only public value: $U(o) = 0.880$ bits per token over the 500-token continuation, explaining about 95% of the information a token from the 202,048-token vocabulary can carry. The proof took 14.3 hours to generate at a prover peak of 78.1 GB GPU memory (83.9 GB unified); the committed witness is about 7.2 terabytes, streamed at the working set (§6.1). The verifier accepted the 93.6 GB proof, checking all 40 opened columns, in 17.7 hours on 20 CPU cores at a peak of 75.7 GB; the per-challenge soundness bound is about $2^{-16.6}$ (§5.4). An earlier 1093-token run (19.3 hours, $U(o) = 0.394$ bits per token, continuation tokens public) predated two prover soundness fixes, a vacuous RMSNorm bracket and a constraint-fold defect, and is superseded by this run; its lower bound reflects a transcript generated by a different backend with higher integer-model agreement, a predictor-side difference that §2.1 prices, not a change in the proof system. Raising the configuration to 80 opened columns is a deployment choice whose measured cost is verification runtime, since the per-column work grows with the count (a GPU verifier is the identified path, §9); verifier memory is dominated by parsing the opened columns and does not depend on how many are checked.

**Llama-2-7B, 1000 tokens.** All 32 layers on real checkpoint weights: the forward-pass proof generates in about 44 minutes at a prover peak of 11.2 GB with the weights streamed from disk, producing a 1.44 GB proof at 10 opened columns that the verifier accepts in about 23 minutes on 20 CPU cores.

**Token binding.** The runs above do not include the transcript binding of Appendix E. Both committed token streams are hidden and internally consistent: the input tokens and the scored output tokens are one committed stream, so the bound is certified over the transcript the proven forward pass actually consumed. What remains open is anchoring that committed transcript to a record produced at generation time (§2.4): the AES and SHA-256 circuits of Appendix E are implemented and tested claim types, and demonstrating the binding end to end against recorded digests is future work (§9).

**Negative controls.** Tampered proofs are rejected: a modified opened column, a modified test polynomial, and a cheating routing witness each fail verification. A claim-level negative suite applies one targeted tamper per verifier check and passes on every claim type. **[TODO: add the Appendix B.6 negative test to the suite and report it: with the paired lookup binding $\mathrm{POW}[b]$ removed, a deflated bound must be accepted, demonstrating that the lookup is load-bearing; in the shipped configuration the same tamper must be rejected.]**

The cost model of §8 was fitted to measured hardware primitives and checked against these runs (Appendix A.2); a companion benchmarks document tabulating predicted against measured times is planned.

## 8. Cost and scaling

The prover's cost is governed by three quantities, each a polynomial in the context length $S$: the committed witness size $W$, the number of distinct linear constraints $L$, and the number of quadratic products $Q$. For Llama-4-Maverick, summing the per-claim contributions of Appendix A.1 gives

$$
\begin{aligned}
W(S) &\approx 4.00{\times}10^{11} + 4.48{\times}10^{8}\,S + 40320\,S^2, \\
L(S) &\approx 1.19{\times}10^{8} + 1.50{\times}10^{8}\,S + 12480\,S^2, \\
Q(S) &\approx 5.93{\times}10^{7} + 1.54{\times}10^{8}\,S + 19200\,S^2.
\end{aligned}
$$

The totals are validated at full scale: the verifier's independent compile of the demonstrated runs yields witness and quadratic row counts within about 1% of the model at both $S = 1000$ and $S = 1093$ (Appendix A.2).

Both leading coefficients have closed forms, so the cost can be understood from two claim types. The $S^2$ term is attention exactly: the scores matmul and softmax are the only claims with a quadratic term, committing 21 witness slots per score cell, and $21\, n_q\, n_{\text{layers}} = 40320$ (Appendix A.3). The $S$ term is dominated, about 89%, by the mixture-of-experts matmuls, committed for all $E = 128$ experts per MoE layer even though one fires (§4.5), about $4{\times}10^{8}$ slots per token. The constant term of $W$ is the committed weights; $L$ and $Q$ have almost none, because Freivalds compresses each weight matmul to $O(k)$ constraints (§4.3).

The three terms dominate the witness in turn as context grows: the weights below about 900 tokens, the committed experts from there to about 11,100, and attention above.

| $S$ | $W$ | weights | experts | attention |
|---|---|---|---|---|
| 1,000 | $8.9{\times}10^{11}$ | 45% | 45% | 5% |
| 11,100 | $1.0{\times}10^{13}$ | 4% | 43% | 48% |
| 1,000,000 | $4.1{\times}10^{16}$ | 0% | 1% | 99% |

The demonstrated run sits at the first crossover, its witness about 90% weights and committed experts; at frontier context the witness is almost entirely attention and the cost can be sized from $W \approx 40320\,S^2$ alone.

Wall-clock time follows from these drivers through a per-primitive cost identity (Appendix A.5), with each heavy step priced by a measured hardware rate; the transforms are memory-bandwidth bound, so the identity's leading terms ride bandwidth, except the column hashing, which rides hash throughput. The four-round protocol multiplies only the witness recomputation, the cheapest term. At $S = 1000$ the identity gives a floor of roughly 8 to 10 hours against the measured 14.3: the implementation is within about 1.5x of its own floor, with the gap in the fold's remaining memory traffic and orchestration (Appendix C), not in the protocol. (The pre-optimization run measured 19.3 hours at $S = 1093$, about 2x the floor; the difference is the fold work of Appendix C landing at full scale.)

At a context of one million tokens the same identity prices a dense-attention proof at roughly 25 years on the demonstrated machine. The bandwidth-riding terms divide by a cluster's aggregate-bandwidth ratio, on the order of 2,500 for an NVL72-class machine, bringing them to days; the column hashing divides by hash throughput instead and joins the leading order at that scale [confirm: per-GPU BLAKE3 rate], which is one motivation for the hash and verifier engineering of §9. Two structural reductions are available. The projection assumes dense attention in every layer: a model whose layers are all windowed or sparse cuts the attention term by the ratio of context to window, after which the committed-experts term is the floor, an overall reduction of one to two orders of magnitude; a model that interleaves, keeping dense attention in a fraction of layers, cuts the quadratic term only by that fraction. And peak memory at long context, set by the softmax residency, is removable by chunking (§6.1) independently of these witness reductions.

The remaining prover levers are the streaming schedule and the argument itself. Proving claims one at a time rather than in four passes removes the repeated witness sweeps and the column-opening re-encode, worth roughly a third of the floor (Appendix A.5); beyond that, a larger speedup requires changing the argument, for example replacing Ligero's linear test with a sumcheck- or GKR-style protocol.


## 9. Limitations and future work

VerInf is a research prototype and has not had a security review. The demonstrated system has four main caveats. The proofs are large, gigabytes at full model scale (§7). The demonstrated run leaves 0.880 bits per token unexplained, which may not suffice for every application. The soundness of a demonstrated configuration is a per-challenge bound (§5.4), adequate in the setting of §1 but a deployment choice rather than a fixed property. And the public claim list reveals the model architecture (§2.2). The main directions for future work are:

**Tightening the bound.**
- Commit the low-precision floating-point intermediates directly, rather than an Int64 approximation, and prove a similarity claim per operation, so quantization error does not propagate through the proof; proving exact low-precision integer inference is a natural first step. Where a deployment runs deterministic kernels, modeling them exactly in the proof is the limiting case of the same direction, driving the bound toward zero.
- Tune the quantization scales, table sizes, and noise-model parameters, which trade unexplained information against proving time; §2.1 prices every such choice through the bound.

**Prover cost.**
- Reduce the committed intermediates that set the leading cost (§8): the quadratic attention witness and the mixture-of-experts commitment.
- Chunk the softmax witness to remove the one large residency (§6.1), decoupling peak memory from context length.
- Prove claims one at a time rather than in four passes, worth roughly a third of the cost floor (Appendix A.5), and parallelize the prover across GPUs along the row structure of §6.1.

**Proof size and verification.**
- Port the verifier's column checks to a GPU, keeping the CPU version as the bit-exact reference; verification runtime is the measured cost of higher soundness levels (§7).
- Reduce proof size with a wider column format, and grow the polynomial degree with the witness so that proof size and verifier work scale as the square root of the witness, as Ligero permits; the demonstrated runs hold the degree fixed, which is simpler but makes the proof grow linearly.
- A budget ledger for the persistent weight commitment: the refresh, and the linear proof linking an old commitment to its replacement, are implemented and tested (§5.2); the remaining piece is the prover-side bookkeeping that tracks the opened-column budget and triggers the refresh.

**Assurance.**
- Formally verify the two properties of §2.3: that the forward-pass claims admit unique witnesses, including a systematic magnitude and width audit of every bracket and exclusion operand (§4.2), with the surprisal claims of Appendix B.6 first; and that the causal mask prevents a later token from influencing an earlier one.
- Replace the manual audit of the claim graph with automated analysis, for example confirming that each weight is read only in a forward pass so that no gradient step is hidden in the computation.
- Hide the model architecture with a second proof stage showing that the verifier's own check ran and accepted (§2.2), so architecture-dependent conditions never surface.
- A security review of the construction and the claim set.

The same claim types support further model architectures with no change to the prover or verifier (§4.1), and demonstrating the token binding of Appendix E at full scale, against digests from real recording hardware, would complete the chain from network capture to certified bound (§2.4, §7).


## Appendix A. Cost model

This appendix derives the cost figures of §8: the per-claim contributions to the three drivers $W$ (committed witness slots), $L$ (distinct linear constraints), and $Q$ (quadratic products); their sums over Llama-4-Maverick as polynomials in the context length $S$; the closed forms for the leading terms; and the machine constants that turn $(W, L, Q)$ into wall-clock.

### A.1 Per-claim contributions

Each claim type contributes to $W$, $L$, and $Q$ as a function of its size parameters. The forms below are for the Maverick configuration, in which every matmul, hadamard, rope, and rmsnorm carries an output-rescale block (marked ⓡ) and softmax runs in its saturating-causal form. Matmul shapes are written $(m, k, n, H)$ for an $H$-head batch of $(m,k)\times(k,n)$ products, with $mHn \equiv m\,H\,n$; elementwise claims have length $N$; routing and combine are over $T$ tokens, $E$ experts, feature width $F$.

| claim | $W$ | $L$ | $Q$ |
|---|---|---|---|
| matmul $(m,k,n,H)$ ⓡ | $6\,mHn + 3k$ | $2k + H + 2\,mHn$ | $k + 2\,mHn$ |
| rmsnorm $(B,d)$ ⓡ | $7Bd + 26B$ | $7B + 2Bd$ | $3Bd + 13B$ |
| softmax $(B,M)$ sat+causal | $15BM + 9B$ | $\tfrac{1}{2}B(M{+}1) + 4BM + 5B$ | $8BM + 3B$ |
| silu $(N)$ | $23N$ | $7N$ | $12N$ |
| hadamard $(N)$ ⓡ | $6N$ | $2N$ | $3N$ |
| rope $(N)$ ⓡ | $6N$ | $3N$ | $2N$ |
| add $(N)$ | $N$ | $N$ | $0$ |
| paired lookup $(N)$ | $3N$ | $N$ | $N$ |
| routing $(T,E)$ | $10TE + 2T$ | $3TE + 3T$ | $5TE$ |
| masked-combine $(T,E,F)$ | $2ETF + TF$ | $ETF + TF$ | $ETF$ |
| freivalds-combine $(T,E,F)$ | $TF + 4ET + T$ | $3ET + 2T$ | $ET$ |

The rescale term in the matmul row, $2\,mHn$ in both $L$ and $Q$, is the output-rescale block; it is what makes a rescaled matmul cost $O(mn)$ constraints rather than $O(k)$, and so determines whether the weights are effectively free in the constraint counts. Attention specializes these claims: the scores $QK^\top$ are a matmul $(S, d, S, n_q)$, an $n_q S^2$ output; softmax is $(B{=}n_q S, M{=}S)$; the value product is a matmul $(S, n_q S, d_h, n_q)$, an $O(S)$ output. In these attention shapes the inner dimension $k$ is written summed over heads ($k = n_q d_h = d$ for the scores), so the $O(k)$ Freivalds terms count all heads at once; the per-head claims total identically. The expert sums use freivalds-combine ($\approx 4ET$) rather than masked-combine ($\approx 2ETF$).

### A.2 Summed totals

Summing the per-claim contributions over the 48-layer model (24 dense, 24 MoE; $d = 5120$, $d_{\text{ff,exp}} = 8192$, $E = 128$ top-1 with all experts committed, $n_q = 40$, $V = 202048$) gives the totals quoted in §8:

$$
\begin{aligned}
W(S) &\approx 4.00{\times}10^{11} + 4.48{\times}10^{8}\,S + 40320\,S^2, \\
L(S) &\approx 1.19{\times}10^{8} + 1.50{\times}10^{8}\,S + 12480\,S^2, \\
Q(S) &\approx 5.93{\times}10^{7} + 1.54{\times}10^{8}\,S + 19200\,S^2.
\end{aligned}
$$

The matmul claims dominate the linear ($S$) coefficient of all three: the QKVO projections, the 128 expert matmuls per MoE layer, the FFN, and the LM head. Softmax, together with the scores matmul, dominates the $S^2$ coefficient. The constant term of $W$ is the committed weights, taken as the parameter count $4{\times}10^{11}$; $L$ and $Q$ have only a small constant from the Freivalds auxiliaries, because Freivalds compresses each weight matmul to $O(k)$ constraints.

The totals are validated at full scale by the demonstrated runs. The verifier's independent compile of the 1093-token proof reports 115,235,029 witness rows and 23,554,246 quadratic rows; at $\mathrm{ELL} = 8192$ slots per row that is $W = 9.44{\times}10^{11}$ and $Q = 1.93{\times}10^{11}$, within 1% of the model's $W(1093) = 9.38{\times}10^{11}$ and $Q(1093) = 1.91{\times}10^{11}$. The 1000-token all-hidden run of §7 checks the same way: 109,267,016 witness rows and 21,370,360 quadratic rows, $W = 8.95{\times}10^{11}$ and $Q = 1.75{\times}10^{11}$ against $W(1000) = 8.88{\times}10^{11}$ and $Q(1000) = 1.73{\times}10^{11}$, again within about 1% (the hidden-token indicator claims it adds sit below the percent level). The $S^2$ coefficient is separately anchored by measurement: single-block runs across context lengths fit 840 committed slots per block per $S^2$ — identical at $E = 8$ and $E = 128$, confirming the quadratic term is attention alone — and $840 \times 48 = 40320$.

### A.3 The leading terms, exactly

Both leading coefficients have closed forms, so the cost can be understood from two claim types.

**The $S^2$ term is attention.** Softmax and the scores matmul are the only claims with an $S^2$ term, and each layer forms an $S \times S$ score matrix for each of $n_q$ heads, so $n_q S^2$ score cells per layer. Per cell, the scores matmul commits 6 values (the score, its raw product, the two rescale words, and two range-check inverses) and softmax commits 15 (the two exponential-table lookups, their LogUp inverses, the paired-lookup combinations, and the saturating mux), giving $(6 + 15)\,n_q\,n_{\text{layers}} = 21 \times 40 \times 48 = 40320$ witness slots per $S^2$. Counting constraints and quadratic products per cell instead gives the $L$ and $Q$ coefficients:

| per score cell | scores matmul | softmax | total | $\times\, n_q\, n_{\text{layers}}$ |
|---|---|---|---|---|
| $W$ | 6 | 15 | 21 | 40320 |
| $L$ | 2 | 4½ | 6½ | 12480 |
| $Q$ | 2 | 8 | 10 | 19200 |

The half in the softmax $L$ count is the causal mask: only the lower triangle of each $S \times S$ block is constrained, so that constraint family is $\tfrac{1}{2}\,n_q S^2$.

**The $S$ term is the committed experts.** Each MoE layer runs all $E$ experts, committed even though one fires (§4.5). Each expert is three matmuls on the shared $S$-token input with output sizes $S\,d_{\text{ff,exp}}$, $S\,d_{\text{ff,exp}}$, and $S\,d$, each carrying the $6\times$ rescale block, so $6\,S\,(2 d_{\text{ff,exp}} + d)$ per expert, and over $E$ experts and $n_{\text{moe}} = 24$ layers,

$$
6\,E\,(2 d_{\text{ff,exp}} + d)\,n_{\text{moe}} = 6 \times 128 \times 21504 \times 24 \approx 3.96{\times}10^{8}
$$

slots per token, about 89% of the linear coefficient; the remainder is the attention projections, the nonlinearities, the norms, and the LM head. The constant term of $W$ is the committed weights.

### A.4 Which term dominates

The three terms dominate $W$ in turn as context grows: weights below about 900 tokens, the committed experts (linear) from there to about 11,100, and attention (quadratic) above. The demonstrated 1000-token run sits at the first crossover.

| $S$ | $W$ | weights | experts | attention |
|---|---|---|---|---|
| 1,000 | $8.9{\times}10^{11}$ | 45% | 45% | 5% |
| 4,000 | $2.8{\times}10^{12}$ | 14% | 56% | 23% |
| 11,100 | $1.0{\times}10^{13}$ | 4% | 43% | 48% |
| 100,000 | $4.5{\times}10^{14}$ | 0% | 9% | 90% |
| 1,000,000 | $4.1{\times}10^{16}$ | 0% | 1% | 99% |

Shares do not sum to 100%: the remainder is the non-expert linear work (attention projections, norms, nonlinearities, the LM head), about 6% at the low end. The 11,100-token row is the second crossover, where the linear and quadratic terms are equal by construction. At small context the proof is model-dominated, with weights and committed experts about 90% of the witness; at frontier context it is almost entirely attention, and the cost can be sized from $W \approx 40320\,S^2$ alone.

### A.5 Wall-clock from $(W, L, Q)$: pass accounting

The streaming prover never stores the witness (the encoded matrix at $S = 1000$ is 57 TB), so the four-round protocol recomputes the witness values once per round. That recomputation is the *only* cost the rounds multiply; every heavy step runs once. The cost identity is

$$T \;\approx\; 4\,T_{\text{wit}} \;+\; (A_c + A_f + A_x)\,W \;+\; D\,W \;+\; E\,W \;+\; C\,Q \;+\; B\,L \;+\; T_{\text{aux}},$$

with each term priced by a measured GB10 primitive (Reed-Solomon NTT: 0.42 ns per element at length $2^{15}$, memory-bandwidth bound at the measured 223 GB/s; a faster NTT kernel was measured to have no headroom, so the only transform lever is fewer transforms):

- $T_{\text{wit}}$ — one witness-computation sweep (the integer forward pass and derived values), compute-bound, about an hour per pass at $S = 1000$. The only ×4 term, and the cheapest; it stays negligible at every scale.
- $A_c \approx 4.2$ ns/slot — the commit encode: an inverse NTT of length $\mathrm{K\_DEG}$ plus a forward NTT of length $\mathrm{N\_LIG}$ is ten transform elements per slot.
- $A_f \approx 3.4$ ns/slot — the linear fold's own transforms (two per row with the fused inverse NTT, Appendix C). These are not a redundant re-encode: the coefficient polynomial does not exist until the round-3 challenges arrive.
- $A_x \approx 4.2$ ns/slot — the column-opening re-encode in round 4. Producing the $T$ opened columns by one more full encode is cheaper than direct evaluation at $T$ points once $T \gtrsim 10$. No re-hashing is required — the column-hash tree from round 1 is a few megabytes and is kept (the implementation re-derives the root anyway as a cheap cross-round consistency check).
- $D$ — column hashing: the encoded matrix is $8W$ elements and BLAKE3 absorbs eight per compression, so one compression per witness slot. The measured primitive is 2.0 gigacompressions per second (about 0.5 ns per slot), and instrumented runs of both claim mixes put the hashing bucket at 5 to 8% of prove time. Compute-bound — the one leading term that rides hash throughput rather than memory bandwidth.
- $E$ — constraint-coefficient work for the linear fold, proportional to the nonzeros of the constraint matrix, $O(W)$. The fold kernels compute each target slot's coefficients in registers directly from its band descriptor (Appendix C) — one thread per slot, nothing materialized or sorted — so the term is arithmetic fused into the fold: 4 to 5% of prove time measured across both claim mixes.
- $C \approx 15$ ns per quadratic product (the products and re-encode of the quadratic fold, bandwidth-bound); $B \approx 0.6$ ns per linear constraint (one challenge hash), negligible at scale.

Calibration against the demonstrated runs: round 1 measured 10.3 ns per witness slot against its $A_c{+}D$ budget of about 5.5; the pre-fold-optimization run (19.3 hours at $S = 1093$) was within about 2× of the identity's total of 8 to 10 hours, and the demonstrated run (14.3 hours at $S = 1000$, with the Appendix C fold optimizations landed) is within about 1.5×. Instrumented two-layer runs of both claim mixes (dense and mixture-of-experts; cuda-synced bucket shares) decompose the prove into: witness 29%/10%, encode 28%/46%, quadratic fold 20%/10%, linear fold 14%/21% — of which the transforms are about two-thirds and the coefficient work a quarter — column hashing 5%/8%, and auxiliary commit 3%/6%. The residual against the identity sits in the fold's last DRAM round trip and chunk orchestration — implementation, not protocol. Collapsing the identity to a single per-slot transform constant underprices the same run by about 6×: the hash, coefficient, and witness terms are load-bearing, not overhead.

Scaling behavior differs by term. $A_c, A_f, A_x, C$ ride aggregate memory bandwidth and divide by a cluster's bandwidth ratio (about 2,500× for a 72-GPU NVL-class machine). $D$ rides hash compute: it divides by GPU count and per-GPU scalar-ALU rate, not bandwidth — and since a B200's scalar throughput is only about 2.4× a GB10's, the hash term scales by roughly 170× where the transforms scale by 2,580×, leaving it a quarter to a third of the cluster floor at one million tokens. $T_{\text{wit}}$ rides matmul compute and stays negligible. Claim-streaming (§9) removes three of the four witness passes and the $A_x$ term by completing each claim's rounds while its rows are live — roughly a third of the floor, not the 4× a per-round count suggests. The projection assumes dense attention, which a sliding-window or sparse pattern replaces with an $O(S w)$ term.

### A.6 Not modeled

The weights floor is taken as the parameter count rather than recomputed from layer shapes (a few-percent refinement). The one-hot validity proof for hidden prompt tokens costs $O(V)$ slots per hidden token — about 0.1% of the per-token witness — and is omitted; it scales with the hidden-prompt length, not the full context. The unexplained-information machinery on the logits is not a per-layer arithmetic claim and is not counted.


## Appendix B. Claim specifications

This appendix specifies the non-arithmetic claim types referenced in §4.4 and §4.5. For each we give the function computed, the gadget, and the constraints that pin its witness. The shared decomposition primitive (B.1) is described first, since the others reuse it; then SiLU (B.2), softmax (B.3), RMSNorm (B.4), and top-1 mixture-of-experts routing (B.5). Each subsection ends on why its witness is uniquely determined, the no-degrees-of-freedom property §3.3 requires: on a valid input, each claim admits exactly one satisfying assignment, so the prover cannot choose a witness that would deflate the bound. Throughout, values are Int64 fixed-point at scale $s$, the paired lookup is the LogUp form of §2.2, and $P$ is the field modulus.

### B.1 Word decomposition and range checks

Two primitives recur in every claim below. A range check proves a committed value lies in $[0, 2^w)$ by a LogUp lookup against a public $2^w$-entry table, committing the inverse $z = 1/(\alpha - x)$ and emitting the per-slot quadratic $(\alpha - x)\,z = 1$; the multiplicities live on the public table side. A word decomposition proves a wide value is a known combination of narrow ones,

$$
x + \text{shift} = \sum_n \text{coeffs}_n \cdot \text{word}_n ,
$$

as one linear constraint, with each $\text{word}_n$ separately range-checked. The coefficients are explicit strides chosen to match the range-table widths. Together these pin the decomposition uniquely: the linear constraint fixes the weighted sum and the range checks fix each word to its window, so there is one satisfying set of words.

Range checks reject negative field elements, whose representatives lie near $P$ rather than in $[0, 2^w)$. A signed value is therefore shifted before checking: the claim commits $x_{\text{shifted}} = x + 2^{w-1}$, range-checks $x_{\text{shifted}} \in [0, 2^w)$, and recovers the signed range $x \in [-2^{w-1}, 2^{w-1})$ from the linear offset. The same gadget serves as the rescale block: when an input arrives at a higher scale $s_{\text{in}}$ than a claim's working scale $s$, the claim decomposes $x_{\text{in}} = 2^r x + x_{\text{low}}$ with $r = \log_2(s_{\text{in}}/s)$, range-checking both the low word (against a $2^r$ table) and $x$, dropping the surplus low bits before its main constraints.

### B.2 SiLU

SiLU computes $\text{out} = x\,\sigma(x)$, saturating to $x$ for large positive $x$ and to $0$ for large negative $x$. It is the canonical tabulated nonlinearity: a sign split, a magnitude decomposition, a paired lookup, and a saturation mux.

The claim commits the sign and magnitude of $x$, pinned by $\text{sign}^2 = \text{sign}$ (boolean) and $\text{sign}\cdot x = C$ with $x = \text{magnitude} + 2C$, so the sign-magnitude split is forced once the magnitude is bounded below $\lceil P/2 \rceil$. The magnitude is word-decomposed (B.1) into a low word that indexes the table and higher words that are nonzero only when the input is out of table range; their weighted sum $g$ is the saturation indicator. A paired lookup on the key $\text{sign}\cdot T_{\text{LEN}} + \text{(low word)}$ returns the in-range SiLU value $y$ from a table holding the positive and negative branches concatenated. A saturation flag $\text{is\_high}$ is committed as the booleanized indicator that $g \neq 0$, pinned by $g\cdot\text{inv}\_g = \text{is\_high}$, $\text{is\_high}\cdot g = g$, and $\text{is\_high}^2 = \text{is\_high}$; it drives a mux that replaces $y$ with the saturated value ($x$ for positive, $0$ for negative) when the input is out of range.

The witness is unique: the magnitude bound forces the sign, the word decomposition is exact (B.1), the boolean flag is determined by whether $g$ is zero, and the lookup and mux are then fixed.

### B.3 Softmax

Softmax computes $y_i \propto \exp(x_i/s)$ normalized over a row, output at scale $s_y$. The difficulty is the per-row normalization, which needs the log-sum-exp shift; the claim pins that shift with a two-table bracket and no slack.

The claim commits the per-row shift $c$ and the shifted inputs $z_i = c - x_i$, and certifies the outputs with two exponential tables, $T_A$ and $T_B$, where $T_B$ is $T_A$ shifted by one integer unit $\delta$. Crucially $T_A$ and $T_B$ are computed from the same rounded expression with shifted argument, so $T_B[k] = T_A[k-\delta]$ bit-for-bit; hence the row sums satisfy $s_2(c) = s_1(c-\delta)$ as exact integer sums, where $s_1 = \sum_i T_A[z_i]$ and $s_2 = \sum_i T_B[z_i]$. The shift $c$ is then bracketed by

$$
s_1 \le s_y \qquad\text{and}\qquad s_2 \ge s_y + 1 ,
$$

each emitted as an exact equality with a non-negative slack, $s_1 + r_{\text{lo}} = s_y$ and $r_{\text{hi}} - s_2 = -(s_y+1)$, the slacks range-checked. Because $s_1(c)$ is monotone non-increasing in $c$ and $s_2(c) = s_1(c-\delta)$, the two inequalities pin $c$ to the unique integer where $s_1$ crosses $s_y$ from above. Two paired lookups against $T_A$ and $T_B$, sharing the key $z_i$, then fix the outputs, and the lookup's own range constraint forces $z_i \ge 0$, i.e. $c \ge \max_i x_i$. A saturating mux handles the high-$z$ tail where the exponential rounds to zero: the lookup key is split into a low word (the table index) and a high word, range-checked, that forces the output to zero when nonzero, so the table need cover only the nonzero region. For attention the same machinery applies the causal mask by doubling the tables with a zero half and keying masked cells into it, so a later token contributes nothing to an earlier token's row.

The witness is unique: the bit-exact table identity makes the bracket an exact integer crossing, monotonicity makes the crossing unique, and the two lookups fix the outputs from $c$ with no tolerance.

### B.4 RMSNorm

RMSNorm computes $\text{out}_{b,i} = x_{b,i} / \sqrt{\text{mean}_i(x_{b,i}^2) + \varepsilon}$. The reciprocal square root is the hard part; rather than tabulate it, the claim pins the integer rsqrt scalar $y_b$ algebraically, the one nonlinearity here that needs no lookup table.

The claim commits the row energy $S_{\text{total}} = \sum_i x_{b,i}^2 + d\varepsilon$ and the scalar $y_b$, and pins $y_b$ to the unique integer satisfying

$$
y_b^2\, S_{\text{total}} \ge \text{magic} \qquad\text{and}\qquad (y_b-1)^2\, S_{\text{total}} \lt \text{magic}, \qquad \text{magic} = d\, s^4 .
$$

These are the rounded definition of $y_b = \lceil \sqrt{\text{magic}/S_{\text{total}}} \rceil$; since $y \mapsto y^2 S_{\text{total}}$ is strictly increasing in $y \ge 0$, exactly one integer satisfies both. The brackets are quadratic constraints with non-negative slacks, the slacks word-decomposed and range-checked (B.1), so no rsqrt table is needed. The broadcast multiply $\text{out}_{b,i} = x_{b,i}\, y_b$ is checked by Freivalds: a random projection folds the $B \times d$ products into $B$ scalar checks $y_b\, u_b = p_b$, avoiding a committed broadcast of $y$.

The witness is unique: the strict monotonicity makes the bracket pin a single $y_b$, and the Freivalds-folded multiply fixes the output.

### B.5 Mixture-of-experts routing

For top-1 routing the claim pins a one-hot mask $m \in \{0,1\}^E$ to the argmax of the routing logits and combines the expert streams under it.

To make the argmax unique, the public logits are tiebroken by packing the expert index into the low bits, $\tilde r_e = 2^L r_e + (E-1-e)$ with $L = \lceil \log_2 E \rceil$, so all $\tilde r_e$ are distinct and lower indices win ties. Three constraints pin $m$: booleanity $m_e^2 = m_e$, cardinality $\sum_e m_e = 1$, and a gap constraint that the chosen logit dominates, $\tilde r_{\text{chosen}} - \tilde r_e \ge 0$ for all $e$, enforced by a range-checked word decomposition of the gap (B.1). A non-maximal choice would make some gap a negative field element near $P$, which cannot be recomposed from the range-checked words, so the proof rejects; with the distinct $\tilde r_e$, this pins $m$ to the unique argmax. The layer output is $\sum_e m_e\, \text{expert}_e(x)$, formed by a combine claim; all $E$ expert streams are committed and the inactive ones zeroed by the mask, which is a hiding requirement (§4.5), not an optimization.

The witness is unique: booleanity and cardinality force a one-hot, the gap constraint forces its support to be the argmax, and the tiebreaker makes the argmax unique.

### B.6: The unexplained-information bound
The bound of §2.1 is computed from the LM-head logits by four claims per output position. An argmax claim yields the per-vocabulary logit gaps $g_i = v^{*} - \ell_i$ and the observed token's gap $g_o$, using the same range-checked gap gadget as routing (B.5). A paired lookup against a public exponential table gives $e_i = \mathrm{EXP}[g_i]$, where $\mathrm{EXP}[g] = \max\!\big(1,\, \lceil s_y\, e^{-g^2/s_c} \rceil\big)$; the lookup's own range constraint bounds each $g_i$. A Hadamard claim forms the squared gap $g_o^2$. A finalize claim then assembles the reported surprisal. All in-circuit arithmetic is in nats at scale $s_b$. The public object is the revealed sum $S_z$; the conversion to bits, $U(o) = \lceil S_z / (s_b \ln 2) \rceil$, happens outside the proof.
The finalize claim commits, per position, the normalizer $a = \sum_i e_i$, a row sum over the vocabulary; a logarithm-pin value $b$ satisfying $a \le \mathrm{POW}[b]$, enforced as the exact identity $a + d = \mathrm{POW}[b]$ with the slack $d$ word-decomposed and range-checked (B.1), and with $\mathrm{POW}[b]$ bound to the public table $\mathrm{POW}[j] = \lfloor s_y\, e^{j/s_b} \rfloor$ by a paired lookup; and the ceiling $z_o = \lceil g_o^2 / k \rceil$ with $k = s_c / s_b$ a power of two, enforced as $k\, z_o = g_o^2 + r$ with the remainder $r$ range-checked in $[0, k)$. The reported surprisal is $z_o + b$. A chain of additions accumulates the positions into $S_z$, which a public pin reveals.
Unlike the claims of B.2 to B.5, the witness here is deliberately not unique, and the soundness property is the weaker one of §2.3: no free direction deflates the bound. Every rounding is pushed upward. Each table entry $e_i$ rounds the true exponential up and is floored at one, so $a$ over-counts the true normalizer. $\mathrm{POW}$ rounds down, so $a \le \mathrm{POW}[b]$ forces $b \ge s_b \ln(a / s_y)$ with no fractional escape. And $z_o$ is a ceiling. The prover's one genuine freedom, choosing $b$ above the least valid index (the honest prover takes the smallest, clamped to the table end), only raises the reported value. With $Q_i(o) = e_o / a$, which is a genuine distribution since $a$ normalizes the committed table values exactly, $z_o + b \ge s_b\,(-\ln Q_i(o))$ follows term by term, so $S_z$ upper-bounds the true surprisal sum and a poor witness penalizes only the prover.
Two freedoms require arguments over the field rather than over the integers. The word decomposition of $d$ is safe by width: at $w_b = 12$ and $d_{\max} = V s_y$, the maximum recomposable value lies far below the modulus, so no wrapped negative $d$ has a valid decomposition, the argument of B.1. The ceiling constraint is subtler. Over the integers, $z_o$ is unique given $g_o^2$; over the field, every range-valid remainder $r$ admits a solution $z_o = k^{-1}(g_o^2 + r) \bmod P$, and $z_o$ itself carries no range check. The reachable perturbations of the public sum are $S_z' = S_z + k^{-1} s \bmod P$ for integer $s \in [0, T k)$. Deflating the bound by any amount requires $s = P - k\Delta$, on the order of $2^{64}$ and unreachable by roughly twenty orders of magnitude, while every reachable perturbation either inflates the bound by less than $T$ scaled-nat units (about 1.5 bits in total at $T = 1093$, the safe direction) or lands $S_z$ near the modulus, an absurd self-reported bound that is useless to a deflating prover. The claim is therefore sound by exclusion rather than by uniqueness; this is the one place in the construction where that argument is load-bearing, and it is written out here for that reason.

## Appendix C. Constraint compilation and evaluation

This appendix specifies how the flat constraint system of §4 is represented and evaluated: the generative constraint level shared by the prover and the verifier, the run structure both evaluation loops exploit, the challenge-access discipline, and the two folds themselves. The governing constraint is scale. The linear system has $\Theta(\text{nnz})$ nonzeros with $\text{nnz} \approx 2\text{–}4\,W$ — tens of terabytes at the scale of §7 — so no materialized form of the constraints ever exists on either side. Everything below is regenerated on demand from descriptors whose total size is $O(\#\text{claims})$, a few tens of megabytes at 400B scale.

### C.1 The constraint level: bands and quadratic descriptors

Each claim compiles — independently on each side; the verifier recompiles from the public claim list and never reads prover-supplied constraints (§6.2) — into three kinds of object:

- **Linear bands.** Each variable carries a small list of bands, one per constraint pattern it participates in: a kind tag, a constraint-id base, and the kind's parameters (roughly 100 bytes). A band maps each flat slot $f$ of its variable to constraint ids and coefficients by closed-form index arithmetic; the map is a pure function of the descriptor and the variable's geometry, so any row window can be evaluated without state. Parameter vectors (Freivalds $\rho, \lambda$; lookup-table coefficients; RoPE tables) are sized $O(k)$ or $O(\text{table})$, never $O(\text{nnz})$, and are shared, not copied per row.
- **Quadratic descriptors.** One per quadratic emission: the three operands' starting rows, the uniform $(a, b)$ coefficients, the slot count $L$, and a positional index base. Row $t$ of a descriptor is the per-row constraint $w_{x+t} \circ w_{y+t} + a\,w_{z+t} = b$ over $\min(\mathrm{ELL},\, L - t\cdot\mathrm{ELL})$ slots, and its combiner index is $\text{base} + t$. This replaces a per-row constraint list of size $O(W/\mathrm{ELL})$ — a verifier-memory binder at long context — with $O(\#\text{emissions})$.
- **Right-hand sides**, kept as compact runs $(\text{start}, \text{length}, \text{value})$.

Constraint ids and quadratic indices advance in claim order; this positional numbering is the entire cross-side contract. The fold combiners $r_{\text{lin}}[g]$ and $r_{\text{quad}}[t]$ are values of a hash PRF on $(s_{\text{comb}}, \text{index}, \text{label})$ and are never materialized globally: both sides derive any combiner in $O(1)$ from the round seed, so no challenge vectors cross the wire. Because a quadratic descriptor carries its index base, firing order is immaterial — each row fetches its own combiner, and field addition commutes.

### C.2 Run structure and challenge access

A band's slot-to-(id, coefficient) map decomposes into maximal homogeneous runs of four shapes, and the shape is a static property of the band kind:

| shape | structure | challenge access |
|---|---|---|
| repeat | a run of slots shares one id | one PRF call per run |
| strided repeat | $\text{id} = \text{base} + (f \bmod k)$: $k$ distinct ids recur on every row | preload $[\text{base}, \text{base}+k)$, cached |
| one-to-one | $\text{id} = \text{base} + f$ (RoPE: stride 2) | one PRF call per slot |
| fan | one slot feeds a contiguous id range | streamed range sum |

Row sums and the Freivalds $B$/$C$ sides are repeats; the Freivalds $A$ side is the strided repeat, whose ids have no contiguous runs but span only $k \le H\cdot K$ ids ($\le 128$ KB preloaded — the prover caches these buffers for the whole proof, across all chunks and layers, since $s_{\text{comb}}$ is fixed once per round). Identity pins and lookups are one-to-one, where one hash per distinct id is the floor; broadcasts are fans, where only the range *sum* is needed, so the range is never buffered. The effect is that challenge hashing costs $O(\text{distinct ids})$ on the duplication-heavy bands — a weight matmul's $B$-side reuses each id $n$ times, an expert matmul's $A$-side $m$ times — rather than $O(\text{nnz})$.

### C.3 The prover's fold (round 3)

The linear test polynomial is $q_{\text{lin}} = \sum_i R_i \cdot p_i$, where $p_i$ is row $i$'s committed codeword polynomial and $R_i$ interpolates row $i$'s slice of $r_{\text{lin}}^{\top} A$. The prover computes it during the same tape-order sweep that regenerates the witness (§6.1), in two stages per 256-row chunk:

1. **Band evaluation** (kind-specific, per band, internally uniform): the band index — descriptors sorted by their variables' disjoint row ranges — yields the bands overlapping the chunk in one binary search; each band evaluates its intersection window into the chunk's $r^{\top}A$ rows, reading challenges per its shape.
2. **The transform fold** (kind- and variable-agnostic, batched): interpolate the chunk's $r^{\top}A$ rows to coefficients (inverse NTT), forward-transform both factors, multiply pointwise, and accumulate the products in the evaluation domain; one inverse NTT at the end of the fold recovers $q_{\text{lin}}$ — exact, since the inverse transform is linear — replacing a per-row inverse transform. Rows are freed afterwards, preserving the working-set memory bound.

Quadratic descriptors fire at their declaring claim, where the sweep's value liveness guarantees all three operands are resident: their rows are re-encoded on demand — exact, because the zero-knowledge padding of §5.2 is generated by a PRG seeded with the absolute row index, so re-encoding reproduces the committed polynomial bit for bit — and the pointwise products fold into $p_0$ under the positionally indexed combiners.

### C.4 The verifier's evaluation

The verifier recompiles the same bands and quadratic descriptors from the public claim list, then evaluates them at the opened columns rather than over full polynomials. The linear sum check needs no witness at all: $\sum_c q_{\text{lin}}(\zeta_c)$ is compared against the right-hand-side runs, each run one PRF range sum. The linear column check reconstructs, for every opened point $\eta_j$, each row's $R_i(\eta_j)$ through the closed form for a message slot's contribution to a codeword value (no NTT), as one generic fold over the runs: a repeat run takes one challenge and a prefix-summed-Lagrange difference (constant coefficients) or a coefficient dot (vector coefficients, with the Freivalds $\lambda$ factored out per run); strided repeats read the band's preload; a fan takes one challenge range sum shared across all $T$ opened points. The quadratic column check walks the quadratic descriptors' rows with their positional combiners.

One implementation of the index arithmetic sits on the verdict path. A second, per-term generator — line-for-line with the prover's — exists only as a test oracle: property tests compare the two as complete (slot, id, coefficient) triple sets over adversarial row windows for every band kind, so the trusted base carries a single copy of the geometry while its correctness is checked against an independent one.

### C.5 Equivalence discipline

Every representation change above — per-row structures to bands and descriptors, per-term evaluation to runs, inline hashing to preloads — is a regrouping of exact field operations, so equality of results is exact, not approximate, and the development gates demand it: verifier builds are compared by per-check *values* (not verdicts) on stored proofs; the prover's fold was compared chunk-by-chunk against an unmodified reference path until its retirement; tampered proofs must still be rejected after every change; and cross-language agreement is tested by expanding both compilers' outputs to canonical triples. The positional numbering of C.1 is what makes this possible: no reorganization changes which challenge multiplies which constraint, so any drift in any bit is a defect by definition.

## Appendix E. Token binding

The bound of §3.1 is conditioned on the input tokens and scored on the output tokens, so it certifies the real run only if the committed streams are the streams the run actually used (§3.4). Inside the proof they are ordinary hidden witness: the claims force the output tokens to be *some* valid selections, not the ones the deployment emitted, and a prover could commit a lower-surprisal transcript and deflate the bound — or condition on a fabricated prompt and certify nothing. This appendix gives the construction that closes the gap by binding both committed streams to a commitment recorded independently at generation time, outside the proof.

### E.1 The recorded commitment

At generation time, an independent recording process computes, for each request/response exchange,

$$H_{1,\text{in}} = H(\mathrm{AES}(\text{key}, \text{tokens}_{\text{in}})), \qquad H_{1,\text{out}} = H(\mathrm{AES}(\text{key}, \text{tokens}_{\text{out}})), \qquad H_2 = H(\text{key material}),$$

with $H$ = SHA-256 and AES in counter mode. These three digests are public inputs to the proof. One key covers the exchange, and $H_2$ is fixed with the request — before the response exists — so the key material cannot be chosen after the fact to fit a covert payload. The prover commits the tokens and the key material in the first round, hidden behind the Merkle root like the weights, and proves

$$H(\mathrm{AES}(\text{key}, \text{tokens}_s)) = H_{1,s} \;\; \text{for } s \in \{\text{in}, \text{out}\}, \qquad H(\text{key material}) = H_2.$$

### E.2 Soundness and the root of trust

Both digests are required. $H_1$ alone is vacuous: it fixes the ciphertext $C$, but for any key the prover grinds, decrypting $C$ under it yields *some* token stream that re-encrypts to $C$, so the tokens would be free. $H_2$ pins the key by collision resistance; with the ciphertext and the key both fixed, the token stream is the unique decryption. The binding therefore has the stronger of §3.3's two properties — a unique satisfying assignment — obtained from collision resistance rather than from constraint structure.

What the binding then certifies is exactly this: *the committed tokens equal the ones that produced the pre-recorded digests*. That is meaningful only if the record is produced by a process the verifier trusts, independently of and prior to the proof — for example, network-boundary hardware that hashes traffic as it passes and certifies the digests on a fixed schedule. A record the prover can rewrite after the fact makes the binding circular, and the assumption should be stated wherever the bound is reported.

### E.3 Confidentiality

Token streams are low-entropy — about 18 bits per token against a 202,048-token vocabulary — so a bare hash of the tokens would be a dictionary-searchable commitment and would leak them. Encrypting under a high-entropy committed key first makes $H_1$ a hiding commitment, preserving zero-knowledge: the digests reveal nothing about the tokens beyond their length. A deployment that chooses to reveal one stream simply publishes it alongside the proof and drops the hiding for that side; the binding equations are unchanged.

### E.4 One committed integer per token

The committed token integer $t_i$ is the single interface between the model side and the wire side of the proof, and every connection is a copy constraint on shared witness slots. On the input side, a select claim commits an indicator row $M_i$ over the vocabulary with booleanity $M_i \circ M_i = M_i$, cardinality $\sum_j M_{ij} = 1$, and the index binding $t_i = \sum_j j \cdot M_{ij}$ — the three together pin $M_i$ uniquely as the indicator of $t_i$ — and the embedded stream is $x = M E$ by a Freivalds matmul against the embedding matrix, which is a committed model weight in any case. On the output side, the argmax claim's select gadget (B.6) carries the same index binding, so the observed-token gap that feeds the surprisal is evaluated at $t_i$ by construction. On the wire side, a word decomposition splits each $t_i$ into a fixed serialization of four little-endian bytes, which feed the cipher. Each of these links is load-bearing: without any one of them, the binding would attest to a different token set than the one the bound is computed over.

### E.5 The cipher and hash in constraints

Both primitives are lookup arguments over the existing table machinery, and all values stay below $2^{32}$, so the width argument of B.1 rules out field wraparound throughout. AES-128-CTR costs, per 16-byte block: sixteen S-box paired lookups per round against a 256-entry table; MixColumns as an *xtime* lookup plus linear constraints (doubling in $\mathrm{GF}(2^8)$); AddRoundKey and all other XORs against a $2^{16}$-entry byte-pair table; ShiftRows as wiring. The key schedule reuses the S-box table and is proven once per key. SHA-256 costs, per 64-byte block: the $\sigma$/$\Sigma$/Ch/Maj functions on 16-bit-limb XOR and AND tables, rotations as decomposition rewiring, and mod-$2^{32}$ additions with one range-checked carry word each; message padding is public structure compiled into the constraints.

Counter mode is chosen for compatibility in both directions. Hardware that encrypts with AES-GCM needs no additional circuit support: GCM's ciphertext is exactly counter-mode output, with counter blocks derived from the IV, which rides in the committed key material behind $H_2$ — so no $\mathrm{GF}(2^{128})$ authentication arithmetic enters the proof. The 16-byte GCM tag cannot be partially explained inside a hash preimage, so either the recorded payload is defined as ciphertext-only, or the tag enters the preimage as unconstrained witness and is charged to the reported bound at 128 bits per packet. The fixed four-byte serialization keeps every token position-addressable and inside a single keystream block, so the same byte layout serves the circuit, the recorder, and any external process that re-derives per-token ciphertext units from the record.

### E.6 Cost

The binding runs over kilobytes of tokens, not the model. A few-thousand-token transcript costs on the order of $10^6$ constraint rows for both streams together — under 0.01% of the forward proof's witness (Appendix A) and invisible in the cost model's terms. SHA-256 and AES are deliberately standard rather than arithmetization-friendly choices: at token scale their circuit cost is negligible, and standard primitives are what independent recording hardware produces.
