# VerInf: Zero-Knowledge Proofs of Inference for Frontier-Scale Models on a Single Machine

James Petrie

---

---


## 1. Introduction
VerInf verifies how much of the information in a token stream is explained by inference of a committed large language model, in zero knowledge. Its one public quantity is a bound on the unexplained information in the output, the bits that a permitted computation on the measured inputs does not account for. The committed witness at frontier scale exceeds any single machine's memory, so our implementation streams it operation by operation. This is what makes massive proofs feasible on small hardware: VerInf produced a proof of a 1000-token forward pass of Llama-4-Maverick, a 400-billion-parameter mixture of experts with all 128 experts committed per layer, in 14.3 hours on a single consumer NVIDIA DGX Spark, streaming a 7.2-terabyte witness at a peak working set of 83.9 GB. Prior published zero-knowledge inference work reaches models of about 13 billion parameters (Sun, Li, and Zhang 2024), roughly thirty times smaller.


Deployed inference typically runs in floating point and is typically nondeterministic across executions. Prior inference proofs require the computation to match an integer circuit exactly, so they attest to a computation contrived to fit the proof; how easily a given serving stack could be made bit-exact is difficult to judge from outside. VerInf requires no changes to the deployment. Rather than reproduce its output, the proof certifies how well a committed integer version of the model predicts it, and the cost of every approximation appears in the reported bound rather than in a failed proof. A deployment that does run deterministically needs no new machinery and simply certifies a tighter bound.


The target use case is sampling packets under high-stakes AI agreements (Shavit 2023; Wasil et al. 2024; Scher and Thiergart 2024; Baker et al. 2025). In the companion framework [cite], a compute operator (the prover) demonstrates to an external party (the verifier) that it runs only permitted workloads, where neither party trusts the other's hardware; the framework gives three confidentiality-preserving ways to perform the recomputation, and this paper builds the zero-knowledge one. Three properties of the setting shape the design. Only randomly sampled outputs are proven, so proving cost amortizes over the workload it deters rather than gating it. Both parties have substantial resources, so slow proving, slow verification, and large proofs are all acceptable, while a small trusted base matters greatly; the demonstrated proof is 93.6 GB and takes hours to check. And the parties are in live contact, so the proof can be interactive: the prover commits before each challenge is drawn and cannot grind over challenges offline, which lets a modest per-challenge soundness level carry real deterrent weight.


The bound itself is a sum of per-token surprisals under a predictor of the deployment's outputs that the prover supplies. By Gibbs' inequality the sum is a valid upper bound on the unexplained information for any predictor, so the design of the predictor can be left to the prover, which is best placed to make it accurate; a poor predictor inflates only the prover's own reported number. Beyond the bound and the public claim list, the proof reveals nothing: not the weights, the activations, or the input and output tokens. The bound places an unusual demand on the proof system: the prover must have no freedom it can use to deflate it. Witness slack is a liability for any inference proof, since slack in an intermediate value propagates through the remaining layers in ways that are hard to analyze; for a bound on unexplained information it is directly exploitable, since the prover selects among valid witnesses in the direction that inflates its predicted probabilities. VerInf therefore meets a two-tier requirement: upstream of the logits every claim admits a unique witness, and downstream of the logits every prover freedom provably inflates the reported bound (§2.2).

VerInf uses Ligero (Ames et al. 2017) as its argument system because it is transparent, needing no trusted setup, plausibly post-quantum, with commitments that are hash-based only, and among the simplest constructions available. The trade Ligero makes is proof size and verifier work that grow as the square root of the witness rather than polylogarithmically, which the setting tolerates: verification is occasional and offline, and the verifier is well resourced. A cost model fitted to measured hardware primitives and validated against the demonstrated runs projects that a one-million-token context could be proven in days on an NVL72-class cluster; the projection assumes dense attention in every layer, and windowed or sparse attention reduces the dominant quadratic term in proportion to the layers and span it covers (§8). The verifier recompiles every constraint from the public claim list and checks the proof only against its own derivation, never against constraints supplied with the proof, so its trusted base is short and auditable.

A model is written as ordinary tensor code against a tape, in the style of PyTorch; each operation records a public claim, and each claim type carries the machinery to prove it, some of it using an additional challenge round. Matrix multiplications are proven by Freivalds' check: the prover commits the input and output matrices, the verifier supplies random projection vectors, and the prover shows the projection of the committed output equals the product of the projections of the committed inputs, reducing each matmul to a single short dot product. Nonlinearities such as softmax are proven against public lookup tables with LogUp (Haböck 2022), whose cost amortizes across the many operations sharing a table. Softmax is where the unique-witness requirement trades against efficiency: its per-row shift is not an integer, and constraining it only to a tolerance band would leave the prover a choice among valid witnesses. VerInf instead pins the shift to a unique integer using the monotonicity of the row sum, at the cost of a second table and the bracket constraints (§4.4). The bound is meaningful only for the transcript the run actually produced, so the committed token streams are bound to digests recorded independently at generation time (§2.4); the circuits for this binding are implemented, and demonstrating it end to end against recorded digests is future work.

The demonstrated system has two main limitations. The committed integer model leaves 0.880 bits per token unexplained, explaining about 95% of the information a token from the 202,048-token vocabulary can carry; future work can tighten this by modeling the deployment's floating-point computation more closely. And the public claim list reveals the model architecture, though not the weights; a second proof stage, showing that the verifier's own architecture checks ran and accepted, could hide it (§9).

Our contributions are:
* A proof design that leaves the prover no freedom to deflate the reported bound, with a unique witness pinned at every step of the forward pass.
* A complete implementation on a hash-based proof system (Ligero), with no trusted setup: a streaming prover whose peak memory tracks one operation rather than the witness, and a verifier that recompiles every constraint from the public claim list.
* A demonstration at frontier scale: a proof of a 1000-token forward pass of a 400-billion-parameter mixture of experts, generated in 14.3 hours on a single consumer machine by streaming the 7.2-terabyte witness at a working set of 83.9 GB, and accepted by an independently implemented verifier.
* A cost model fitted to measured hardware primitives, projecting that a million-token context with dense attention in every layer could be proven in days on an NVL72-class cluster (§8).

The paper is organized as follows. Section 2 states what VerInf proves and what the bound requires of the proof system. Section 3 compares with prior work. Section 4 compiles a model to constraints, Section 5 gives the Ligero argument and its soundness, and Section 6 describes the streaming prover and the verifier. Section 7 reports the results, Section 8 the cost model and scaling, and Section 9 the limitations and future work.


## 2. What VerInf proves

VerInf attests to a single public quantity: a bound on the unexplained information in a committed output, given a committed model and committed inputs. The quantity and its security analysis are developed in the companion paper [cite]; this section states the certified quantity (§2.1), sets out what the bound demands of the proof system itself (§2.2), the requirements the claim design of §4 and Appendix B is built to satisfy, and then states the trust model (§2.3). Section 2.4 describes how the committed token streams are anchored to the transcript the run actually produced.

### 2.1 The unexplained-information bound

The certified quantity measures how well the declared computation explains the observed output: the fewer bits of the output stream left unexplained, the more closely the stream matches what the committed model, run on the committed inputs, would produce. The declared computation $D$ is the workload the prover claims to be running; in this paper, it is the committed integer model of §4. Running $D$ on the measured inputs $x$ produces $D(x)$, which includes intermediate values such as the logits. The companion paper [cite] bounds the unexplained information by a sum of per-token surprisals:

$$U(o) = -\sum_i \log_2 Q_i\!\left(o_i \mid D(x),\, o_{\lt i}\right).$$

Here $Q$ is any probability distribution predicting the outputs from $D(x)$, and $o$ is the measured output. This is the quantity the proof certifies, and it is its only public value; the weights, inputs, and output tokens stay hidden. In the setting of §1 the bound is what caps covert bandwidth: a payload must hide in the unexplained bits, so the less the output leaves unexplained, the less room remains. The derivation, the extension to non-zero-temperature sampling, and the security analysis are in the companion paper.

The design of $Q$ is left to the prover: any $Q$ gives a valid bound, and a poor one inflates only the prover's own reported number. The predictor is committed before any challenge is drawn, and the proof certifies that the reported value is at least the surprisal under it (§4.6). Our prototype models hardware nondeterminism as Gaussian noise on the logits, $Q_i(o_i) \propto \exp(-(v^{*} - \ell_i)^2/\sigma^2)$, with $v^{*}$ the maximum logit and $\sigma$ calibrated empirically. Every approximation, from the integer model to the noise parameters, is priced the same way: a worse predictor of the deployment's tokens reports a larger $U(o)$, so tightening the bound is an engineering trade, not a soundness question (§9).

### 2.2 What the bound requires of the proof system

For $U(o)$ to be an upper bound, the prover must have no freedom it can use to deflate it. The requirement takes two forms, split at the logits. Upstream of the logits, in the forward pass that produces them, every claim must admit exactly one satisfying assignment. Slack in an intermediate value propagates through the remaining layers in directions that cannot be analyzed, so any freedom there could cascade into the token probabilities arbitrarily. Downstream of the logits, in the short computation from logits to the reported bound, freedom is permitted provided every free direction increases the reported value. This weaker property can be established directly, because the downstream computation is a few steps of analyzable arithmetic. The construction meets the first requirement claim by claim (Appendix B.2 to B.6) and the second by pushing every rounding in the surprisal computation upward (Appendix B.7).

Softmax is where the first requirement trades against efficiency. Its row-wise normalization uses a per-row shift (the log-sum-exp) that is not an integer, and recent work speeds up the proof by leaving the shift unverified, checking only that the normalized outputs sum to the expected total within a tolerance for quantization error (Sun, Li, and Zhang 2024). The tolerance admits several shifts, which give slightly different output probabilities. For reproducing a fixed output this is arguably adequate, provided accumulated tolerances cannot be steered through the remaining layers, which has not been established; for a bound on $U$ it is immediately exploitable, since the prover selects among the valid witnesses in the direction that inflates its predicted probabilities. VerInf pins the shift to a single integer instead, using the monotonicity of the row sum in the shift, at the cost of a somewhat larger proof (§4.4, Appendix B.3).

A second requirement is causality: the prediction of each output token must depend only on earlier tokens, so a later token cannot be used to lower the surprisal of an earlier one. This is enforced by the public attention mask compiled into the claims.

These are properties of the claim graph, which is public and recorded with the proof. The verifier checks the proof against this claim list; that the claim list itself has the required structure, for example that each weight is read only in a forward pass and never updated so that no gradient step is hidden in the computation, is established by auditing it. Confirming this automatically, by static analysis of the claim graph, is future work (§9).

### 2.3 Trust model

The two parties want different guarantees. The prover wants confidentiality: the model weights, the activations, and the input and output token streams must not leak. The verifier wants soundness: the reported $U(o)$ must be a genuine upper bound for the committed tokens under the committed weights. Their interface is the public claim list, a statement of what kind of computation was performed (§4); it reveals the model architecture, though not the weights, and hiding the architecture as well is future work (§9). The proof reveals nothing beyond $U(o)$ and the claim list, so a dishonest verifier learns nothing more, and a dishonest prover cannot produce an accepting proof for a deflated bound except with the soundness error of §5. Each side trusts only its own code: the verifier's trusted base is short (§6.2), and a fault anywhere on the prover's side can only cause a proof to fail, never to falsely verify.

### 2.4 Anchoring to the real transcript

The bound is conditioned on the input and scored on the output, so it is meaningful only when both token streams are the ones the run actually used. Inside the proof they are hidden witness, which on its own does not tie them to any external record: a prover could commit a lower-surprisal transcript and deflate the bound, or condition on a fabricated prompt and certify nothing. VerInf closes this by binding both committed streams to digests recorded independently at generation time: a recorder hashes the encrypted token streams as they pass and receives a commitment to the key material, fixed with the request before the response exists, and the proof shows that the committed tokens encrypt and hash to the recorded digests, without revealing them (Appendix E). The record itself must come from a process the verifier trusts, independently of and prior to the proof; in the setting of §1 this is the verifier's network-boundary hardware (Petrie et al. 2025).


## 3. Related work

**Zero-knowledge proofs of LLM inference.** zkLLM (Sun, Li, and Zhang 2024) is the closest prior work: a sumcheck-based argument with a tailored proof of attention, proving LLaMA-2-13B at a 2,048-token sequence in about 13 minutes on an A100. VerInf differs on three axes. On scale, its demonstrated run is roughly thirty times larger; the reported zkLLM system demonstrates the prover, where the demonstrated VerInf runs are end to end, with an independently implemented verifier accepting the proof and the zero-knowledge masking active. On the statement proven, zkLLM, like the other systems below, requires the computation to match its integer circuit exactly, so it attests to a computation contrived to fit the proof; VerInf proves that the committed output is well explained by the committed integer model and bounds what goes unexplained, which is what lets it target a deployment as it runs (§1). And on commitments, zkLLM instantiates its polynomial commitment with Hyrax (Wahby et al. 2018), which rests on the discrete-logarithm assumption; VerInf's commitments are hash-based only, with no trusted setup and plausibly post-quantum.

zkGPT (Qu et al. 2025) proves GPT-2 inference in under 25 seconds; zkPyTorch (Xie et al. 2025) compiles quantized models to a GKR pipeline, reporting Llama-3-8B at 150 seconds per token on one CPU thread; ZKTorch (Chen, Tang, and Kang 2025) compiles inference to an accumulation scheme at the several-billion-parameter scale. The line descends from pre-LLM zkML systems such as zkCNN (Liu, Xie, and Zhang 2021) and ZKML (Chen et al. 2024); Peng et al. (2025) survey the area.

The exact-circuit systems also leave the prover freedom in the witness. zkLLM constrains the softmax normalization only to a tolerance band on the row sum, leaves the rounding rule for table entries unspecified, and lets the prover choose some setup parameters; each admits multiple satisfying witnesses, so the proven relation is weaker than exact execution of the committed circuit, and how much the accumulated slack admits has not been analyzed. For the statement VerInf proves the question cannot be left open, since any such freedom is a direct lever on the reported bound (§2.2). The construction therefore replaces each with a public constant or a unique-witness gadget: the tables, scales, and noise parameters are constants of the claim list, every table entry follows a deterministic rounding rule the verifier reproduces, and the softmax shift is pinned by a two-table bracket (§4.4, Appendix B).

**Verified inference with a trusted verifier.** A separate line verifies inference in the unilateral setting, where the verifier is trusted and sees plaintext. Rinberg et al. (2025) and Karvonen et al. (2025), companion works, verify recomputation against hardware nondeterminism to detect weight exfiltration and inference-specification violations; VerInf shares their per-token surprisal measure under a logit-noise model. TOPLOC (Ong et al. 2025) attests with locality-sensitive hashes of activations, and Verde (Arun et al. 2025) instead makes inference bitwise reproducible so that arbitration can settle disputes. VerInf targets the bilateral setting, where the verifier trusts neither the prover's hardware nor its software and must not see the tokens: the recomputation is replaced by a zero-knowledge proof, and the verifier learns only the bound.


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

The proof works over a finite field, so every value is quantized to a 64-bit fixed-point integer at a scale $S$ (we use $S = 2^{12}$). The prover commits integers; the verifier checks them with field constraints, and lookup tables handle the nonlinearities. Matrix multiplications are carried out in Int64 or in FP64, which is bit-exact at these magnitudes because every accumulator stays below $2^{53}$, then requantized.

The central numeric constraint is overflow: every committed value and every intermediate product must stay below the Goldilocks modulus, which bounds the usable scales and contraction depths. This is what fixes the scale choices and what forces a rescaling step into some of the nonlinearity claims when an input arrives at a higher scale than the claim can absorb (§4.4). The accuracy cost of the representation is priced by the bound (§2.1).

If a value nevertheless exceeds its budget, one of two things happens. Where it feeds a word decomposition or range check, a wrapped field element has no valid decomposition into range-checked words and the proof rejects (Appendix B.1). Elsewhere the constraints are field identities and the proof accepts, but what it then attests is the wrapped field computation: still a deterministic function of the committed inputs with a unique witness, so the requirement of §2.2 is unaffected, and the bound remains genuine for that computation; in practice wrapped logits predict the deployment's tokens poorly and the divergence surfaces as a large $U(o)$. Care is needed only where a uniqueness or exclusion argument itself assumes a magnitude bound, since a bracket whose operand is not independently bounded can admit a wrapped second solution. Every such operand therefore carries a range check or a written width argument tracing to one; Appendix B applies this discipline claim by claim, and it is most load-bearing in the surprisal claims of B.7, where slack is deliberately permitted and the safe direction must be established in the field (§9).

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

**RMSNorm.** Rather than look up a reciprocal square root, the claim pins the rsqrt scalar $y$ algebraically, with two quadratic brackets that force $y$ to the unique integer with $y^2\, S_{\text{total}} \ge \text{magic}$ and $(y - 1)^2\, S_{\text{total}} \lt \text{magic}$, where $\text{magic} = d\, s^4$, with the bracket products assembled from range-checked limbs so the inequalities hold over the integers, and no large table is needed. The broadcast multiply is folded by Freivalds rather than committed cellwise.

**Softmax.** The claim pins the per-row log-sum-exp shift with a two-table monotonicity bracket, rather than leaving it constrained only by a tolerance on the row sum. Two exponential tables $T_A$ and $T_B$, the second the first shifted by one integer unit $\delta$, are computed from the same rounded expression, so $T_B[k] = T_A[k-\delta]$ bit-for-bit and the row sums satisfy $s_2(c) = s_1(c-\delta)$ exactly. Since $s_1(c)$ is monotone non-increasing in the shift $c$, bracketing it between $s_1 \le s_y$ and $s_2 \ge s_y + 1$ pins $c$ to the unique integer where $s_1$ crosses $s_y$, with no tolerance band. Two paired lookups against $T_A$ and $T_B$ then certify the outputs. Pinning the shift this way removes the deflating freedom of §2.2, at the cost of a second table and the bracket constraints. An optional saturating mux sizes the table to the nonzero region of the exponential.

**SiLU.** The input is split into sign and magnitude, the magnitude is decomposed into a low word that indexes the table and high words that detect saturation, and a paired lookup returns the table value; when the high words are nonzero a mux replaces the lookup with the saturated value (the input itself for large positive inputs, zero for large negative).

When an input arrives at a higher scale than a claim can absorb without overflow (§4.2), the claim emits a shared rescale block, a word decomposition that drops the low bits, before its main constraints. The full per-claim constraint listings are in Appendix B.

### 4.5 Mixture-of-experts routing

In a top-1 MoE layer each token routes to one expert by routing logit. The routing claim pins a one-hot mask $m$ to the argmax of the routing logits, made unique by a public tiebreaker that packs the expert index into the low bits of each logit so no two are equal. Booleanity ($m_e^2 = m_e$) and cardinality ($\sum_e m_e = 1$) force $m$ to be one-hot, and a range-checked gap constraint forces its support to be the argmax: if the mask selected a non-maximal expert, the gap to the true maximum would be a negative field element, which cannot be recomposed from the range-checked words, and the proof rejects. A masked-combine claim then forms the layer output as $\sum_e m_e\, \text{expert}_e(x)$.

All $E$ experts' streams are committed even though only one is active. This is a hiding requirement, not an inefficiency: a witness that committed only the active expert would reveal the routing decision, so the inactive experts are committed and zeroed by the mask. (A top-1 simplification applies the elementwise nonlinearity once after the masked sum rather than per expert, since the sum already selects the chosen expert's stream; this reduces the committed intermediates without changing what is proven.)

### 4.6 The unexplained-information bound as claims

The bound of §2.1 is computed from the LM-head logits by four claims per output position, reusing the gap gadget of the routing claim, the paired table lookup, and the elementwise steps; Appendix B.7 gives the full specification, and Appendix E binds the same committed tokens to the digests recorded at generation time (§2.4). Its soundness property is the weaker downstream one of §2.2: the witness is deliberately not unique, and instead every prover freedom provably inflates the reported value. Normalization is where the asymmetry between the two requirements pays. Softmax pins its per-row shift exactly, because upstream slack is unanalyzable, while the bound replaces normalization with a one-sided logarithm pin, because downstream slack can be shown to only inflate. The output tokens enter only as committed witness consumed by these claims; they never appear in the public claim list, so the proof reveals the bound and nothing about which tokens were produced. Because the bound folds onto the logits inside the proof, certifying it costs little beyond the forward pass it sits on top of.


## 5. The Ligero argument and soundness

Section 4 produced a witness and a flat list of linear and quadratic constraints. This section gives the argument that proves them all at once: the Ligero construction and our parameters (§5.1), the zero-knowledge masking (§5.2), the four interactive rounds across which the stages run (§5.3), and the soundness analysis (§5.4).

### 5.1 Commit, test, open

Ligero (Ames, Hazay, Ishai, and Venkitasubramaniam 2017) is a zero-knowledge argument built from Reed-Solomon codes and a Merkle commitment, with no trusted setup and security resting only on the collision resistance of a hash function. It proceeds in three stages. In the commit stage it arranges the committed values as a matrix whose rows are Reed-Solomon codewords and hashes the columns into a Merkle tree, whose single root binds the whole witness. In the test stage it folds the constraints into a few short polynomials with random combiners: a linear test that a system $Ax = b$ holds, a quadratic (Hadamard) test that a system of pointwise products holds, and an interleaved Reed-Solomon test that every row is close to a codeword. In the open stage the verifier names a random subset of columns; the prover reveals them with Merkle paths, and the verifier checks that they hash to the root and are consistent with the test polynomials. Soundness comes from Reed-Solomon distance: any inconsistency appears in a constant fraction of columns, so a few random column checks catch it with high probability.

The trade Ligero makes is proof size and verifier work, both growing as the square root of the witness rather than polylogarithmically, in exchange for a simple construction, a small trusted base, and transparent, plausibly post-quantum security. In the setting of §1 this is the right trade: verification is occasional and offline, the verifier is well resourced, and a short auditable trusted base with no trusted setup matters more than proof size.

**Parameters.** VerInf works over the Goldilocks field, $|F| = 2^{64} - 2^{32} + 1$, which admits fast number-theoretic transforms and fits the fixed-point magnitudes of §4.2 without wraparound. The constants are

$$\mathrm{ELL} = 8192, \quad \mathrm{K\_DEG} = 16384, \quad \rho = 4, \quad \mathrm{N\_LIG} = \rho \cdot \mathrm{K\_DEG} = 65536,$$

where `ELL` is the number of constrained message slots per row, `K_DEG` the polynomial degree bound, $\rho$ the Reed-Solomon inverse rate, and `N_LIG` the codeword length (the number of columns). The demonstrated runs hold these constants fixed as the witness grows, which is simpler but makes proof size and verifier work grow linearly rather than as the square root Ligero permits (§9). The number of columns opened, `T_QUERIES`, is a deployment choice that sets the soundness level (§5.4); the demonstrated runs are reported with their values in §7. The hash is BLAKE3.

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

We implemented the prover in CUDA and the verifier in Rust; the two share no code.

### 6.1 The streaming prover

In the prover, the field arithmetic, the Reed-Solomon transforms, and the Merkle hashing of §5.1 are GPU kernels over the Goldilocks field, compiled for the local device on first use.

At frontier scale the committed witness is far larger than any single machine's memory: for §7's 400B-parameter run it is about 7.2 terabytes ($9.0{\times}10^{11}$ field elements, Appendix A). The prover therefore streams the witness, committing one operation at a time: each row is Reed-Solomon encoded, folded into the column hashes and into running accumulators for the test polynomials, then freed before the next is computed. Because the Merkle tree is built over columns and the test polynomials are linear and quadratic accumulations over rows, no later step needs the full encoded matrix resident, so peak memory tracks the working set of a single operation rather than the witness or the proof. Streaming bounds working-set memory only; proof size and proving time are unaffected. The same row-by-row structure admits parallelization across GPUs, with transform work split across rows and the column hashes partitioned across nodes, each accumulating its assigned columns (§9).

[Todo: fix writing] Peak memory is therefore set by the largest single working set. At long context that is softmax's: its witness grows quadratically with context length, and the implementation currently proves each softmax over its full score matrix at once. This is a choice rather than a necessity, since the rows could be split into chunks and proven piece by piece, capping the working set at the chunk size. One small piece of state also persists across the whole proof: the lookup argument of §4.4 needs per-table counts of how often each entry was queried, kept as one resident histogram per table and fixed in size by the tables.

Appendix C specifies how the sparse constraint system is regenerated and evaluated against the streamed witness during the test folds; the resulting cost profile is analyzed in §8.

### 6.2 The verifier

The verifier is a Rust program that reads a proof and decides whether to accept it. Sharing no code with the prover, it recompiles the constraint system from the public claim list (§4.1) and checks the proof against its own derivation, never against constraints supplied with the proof. Its work is, first, to confirm that the claim list meets the requirements of §2.2, and then to check, at the columns opened in the final round (§5.3), that they re-hash to the committed Merkle root and that the three test polynomials of §5.1, recomputed at those columns, match the prover's. A proof is accepted only if every check passes.

The trusted base comprises the field arithmetic, the hash and Merkle check, the challenge derivation, the constraint compile, and these checks. Proof parsing and all other handling sit outside it, since a malformed or dishonest value fails a check and the proof is rejected. The verifier depends on three crates (`blake3`, `rayon`, `serde_json`) and is differential-tested bit-for-bit against a Python reference implementation. It needs no GPU, but at full model scale it needs a large-memory host, because the compiled constraint system grows with the witness; the heaviest step, the linear identity at the opened columns, is dense field arithmetic over the constraints, parallelizes across cores, and is the natural candidate for a GPU port (§9).

This division of labor follows the trust model of §2.3: only the verifier requires review, and prover-side code can be modified freely without enlarging the trusted base, since a fault there causes a proof to fail rather than to verify falsely.


## 7. Results

We report two runs, both produced by the streaming prover of §6.1 on a single NVIDIA DGX Spark (GB10, 121 GB unified memory) and checked by the verifier of §6.2, which shares no code with the prover.

**Llama-4-Maverick, 1000 tokens, every token hidden.** The full 48-layer, 400B-parameter mixture of experts, with all 128 experts committed per MoE layer, proven in the four-round protocol of §5.3 with 40 columns opened. The transcript is a 500-token prompt and the model's own 500-token greedy continuation; all 1000 tokens are hidden, entering the proof only as committed one-hot indicators, and the indicator rows are shared between the input selection and the surprisal claims of §4.6, so the scored tokens are, by shared committed variable, the tokens the model consumed. The surprisal claims run inside the proof and the bound is its only public value: $U(o) = 0.880$ bits per token over the 500-token continuation, explaining about 95% of the information a token from the 202,048-token vocabulary can carry. The proof took 14.3 hours to generate at a prover peak of 78.1 GB GPU memory (83.9 GB unified); the committed witness is about 7.2 terabytes, streamed at the working set (§6.1). The verifier accepted the 93.6 GB proof, checking all 40 opened columns, in 17.7 hours on 20 CPU cores at a peak of 75.7 GB; the per-challenge soundness bound is about $2^{-16.6}$ (§5.4). An earlier 1093-token run (19.3 hours, $U(o) = 0.394$ bits per token, continuation tokens public) predated two prover soundness fixes, a vacuous RMSNorm bracket and a constraint-fold defect, and is superseded by this run; its lower bound reflects a transcript generated by a different backend with higher integer-model agreement, a predictor-side difference that §2.1 prices, not a change in the proof system. Raising the configuration to 80 opened columns is a deployment choice whose measured cost is verification runtime, since the per-column work grows with the count (a GPU verifier is the identified path, §9); verifier memory is dominated by parsing the opened columns and does not depend on how many are checked.

**Llama-2-7B, 1000 tokens.** All 32 layers on real checkpoint weights: the forward-pass proof generates in about 44 minutes at a prover peak of 11.2 GB with the weights streamed from disk, producing a 1.44 GB proof at 10 opened columns that the verifier accepts in about 23 minutes on 20 CPU cores.

**Token binding.** The runs above do not include the transcript binding of Appendix E. Both committed token streams are hidden and internally consistent: the input tokens and the scored output tokens are one committed stream, so the bound is certified over the transcript the proven forward pass actually consumed. What remains open is anchoring that committed transcript to a record produced at generation time (§2.4): the AES and SHA-256 circuits of Appendix E are implemented and tested claim types, and demonstrating the binding end to end against recorded digests is future work (§9).

**Negative controls.** Tampered proofs are rejected: a modified opened column, a modified test polynomial, and a cheating routing witness each fail verification. A claim-level negative suite applies one targeted tamper per verifier check and passes on every claim type. **[TODO: add the Appendix B.7 negative test to the suite and report it: with the paired lookup binding $\mathrm{POW}[b]$ removed, a deflated bound must be accepted, demonstrating that the lookup is load-bearing; in the shipped configuration the same tamper must be rejected.]**

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

VerInf is a research prototype and has not had a security review. The demonstrated system has four main caveats. The proofs are large, gigabytes at full model scale (§7). The demonstrated run leaves 0.880 bits per token unexplained, which may not suffice for every application. The soundness of a demonstrated configuration is a per-challenge bound (§5.4), adequate in the setting of §1 but a deployment choice rather than a fixed property. And the public claim list reveals the model architecture (§2.3). The main directions for future work are:

**Tightening the bound.**
- Commit the low-precision floating-point intermediates directly, rather than an Int64 approximation, and prove a similarity claim per operation, so quantization error does not propagate through the proof; proving exact low-precision integer inference is a natural first step. Where a deployment runs deterministic kernels, modeling them exactly in the proof is the limiting case of the same direction, driving the bound toward zero; bitwise-reproducible inference has been demonstrated across GPU variants (Cankaya 2026), and a deployment that adopts it needs only this limiting case.
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
- Formally verify the two properties of §2.2: that the forward-pass claims admit unique witnesses, including a systematic magnitude and width audit of every bracket and exclusion operand (§4.2), with the surprisal claims of Appendix B.7 first; and that the causal mask prevents a later token from influencing an earlier one.
- Replace the manual audit of the claim graph with automated analysis, for example confirming that each weight is read only in a forward pass so that no gradient step is hidden in the computation.
- Hide the model architecture with a second proof stage showing that the verifier's own check ran and accepted (§2.3), so architecture-dependent conditions never surface.
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

Each claim is specified by a listing, followed by its counts, a soundness lemma, and a generalization note. Every listing below is reconciled against the compiler; emitted counts match Appendix A.1's rows exactly except where a deviation is stated in place.

**Public constants.** Every claim's size parameters and scales are public constants of the claim list; a claim's header names only its further public constants, such as lookup tables and their rounding rules. In the demonstrated configuration every claim's inputs and outputs sit at the working scale $S = 2^{12}$, with each multiplicative claim rescaling its output to $S$ (B.1); the compiler supports per-claim scales, and the surprisal claims use distinct scales for table values and nats (B.7).

**Line kinds.** Commitment lines carry no equation: `in` names values inherited from upstream claims or the committed weights, `decl` values the prover asserts, and `chal` verifier randomness. Constraint lines carry exactly one equation each: `lin comp` and `quad comp` define a value by its constraint (given everything above, exactly one assignment satisfies it), feeding the linear and quadratic tests; `lkp comp` defines a value by a LogUp lookup against a named public table, certifying both that the key is in the table's range and that the value is the table's entry (B.1); `lin pin` (and the one-sided `lin pin≤`) constrains already-committed values without introducing new ones. A `gadget` line invokes a B.1 gadget at a stated size. Italic comment rows name the phases of longer listings and carry no constraints. A bound index carries its range in the sum; a free index carries it in a trailing quantifier.

**What may be factored out.** A gadget line, or an omitted expansion, is permitted only for machinery that is soundness-inert: no lemma in this appendix may ever need its internals. Under this rule the LogUp bookkeeping is factored everywhere. The per-table challenges, the committed inverses and folded keys, and the settlements are identical boilerplate whose soundness is the per-table LogUp term of §5.4's error sum, cited by no claim lemma, and each `lkp comp` or range-check line implies them (B.1). Anything a lemma touches stays inline: the softmax saturation mechanics, the bracket slacks, and every pin appear in their listings explicitly.

**Turn boundaries.** Each horizontal rule is a message boundary: the lines between two rules are one party's message, committed together, and a `chal` line is drawn only after everything above its rule is committed. This ordering is what each lemma's "committed before the challenge" step refers to. The factored LogUp turns follow the same discipline (folded keys and inverses commit after their table's challenges) without appearing in the listings. The block above the first rule (`in` lines) is inherited context, not part of the claim's transcript: an `in` line promises nothing, and each claim's guarantees are conditional on its inputs, the weights included.

**Witness and coefficients.** Every `decl` and every value defined by a `comp` line is committed. A value used only linearly may additionally be left uncommitted, its defining sum folded into the constraints that use it; listings mark such virtual values where they use them (the broadcast product inside RMSNorm, B.4, is the worked example), and the compiler commits everything else, including values with purely linear participation such as row sums. Constraint coefficients may be arbitrary public functions of the challenges (for example, the product coefficients in the matmul pin, or the public per-entry coefficients $\alpha - v[j]$ in the LogUp settlements), derived identically by both sides and never materialized.

**Counts.** Each claim ends with a labeled totals line attributing every term to the line that emits it. The rules: a `comp` line contributes its multiplicity to $W$ and to its tagged test; `lkp comp` contributes three slots per query and one linear and one quadratic (B.1); a `decl` contributes to $W$ only; a `pin` to its test only; `chal` and `in` contribute nothing, each value counted once by the claim that creates it; a `gadget` line contributes B.1's per-element counts times its size. LogUp settlements and multiplicity histograms are per public table, shared across all invocations, outside any per-element count.

**Soundness obligation.** Each claim's lemma first discharges the comp lines (unique by definition given the commitments and challenges), then argues every `decl`: pinned to a unique value, or, where the pin is marked one-sided, bounded from the safe direction. Declarations may be discharged in any order that avoids circularity. The lemma classifies the claim's overflow behavior under the cases of §4.2. A declaration without an argued pin is a defect by definition. Gadget invocations carry B.1's lemmas at the invoked size.

### B.1 Shared gadgets

Four gadgets recur across the claims. The LogUp lookup is the underlying mechanism; the range check and the paired lookup are its two specializations, and the word decomposition and rescale build on the range check. Each entry depends only on entries above it.

**LogUp lookups.** Two of the gadgets below are lookup arguments, built on LogUp as described in §4.4. A query against a public table is certified by committing, after the table's challenges $(\alpha, \beta)$ are drawn, a folded key that combines the query's components under $\beta$ and the inverse $z = 1/(\alpha - \text{key})$; a per-table settlement then proves that the multiset of all queries matches the table. The settlement commits a multiplicity histogram (before the challenges) and per-entry weights $w[j] = m[j]/(\alpha - v[j])$ (after), and emits one linear constraint per entry, with the public coefficient $\alpha - v[j]$, plus a single cross-claim sum identity equating the query inverses with the entry weights. Because the challenge folds the query's components together, a query matches an entry only when every component agrees. Shared by all lookups, and per the factoring rule implied rather than listed: the challenge turns, the folded keys and inverses, the settlement, and the histogram. The challenges are sampled per table from the round seed.

**Range check.** A range check proves that a committed value $x$ lies in $[0, 2^w)$: a LogUp lookup against the table containing every integer in that range, keyed on $x$ alone, so no folded key is needed and the per-query constraint is the single quadratic $(\alpha - x)\,z = 1$. Counts per checked slot: one witness slot ($z$) and one quadratic.

A range check as stated rejects negative values, whose field representatives lie near $P$ rather than in $[0, 2^w)$. To check a signed value $x \in [-2^{w-1}, 2^{w-1})$, the prover commits the shifted form $x_{\text{shifted}} = x + 2^{w-1}$, one linear constraint ties it to $x$, and the range check runs on $x_{\text{shifted}}$; the signed range follows from the offset. The signed form adds one slot and one linear constraint.

**Paired lookup.** A paired lookup proves a functional relation $y = T[x]$ against a table of input-output pairs: the LogUp query is the pair, folded as $u = (x + \text{shift}) + \beta\,y$, so one lookup certifies both that $x$ is in the table's key range and that $y$ is the table's value there. Committed per query: the value $y$, the folded key $u$, and the inverse $z$; the input $x$ belongs to its producer. Per-query constraints: one linear (the key folding) and one quadratic (the inverse). Counts per query: three slots, one linear, one quadratic, matching A.1's paired-lookup row.

**Word decomposition.** A word decomposition proves that a wide value is built from narrow pieces in a fixed way: the single linear constraint

$$x = \textstyle\sum_{n} \text{coeff}_n \cdot \text{word}_n,$$

with each word separately range-checked (signed words via the shifted form above). The coefficients are public constants of the invoking claim, powers of two chosen so the words' windows tile the intended range, as in splitting a product into a kept high word and a dropped low word.

The point of the gadget is uniqueness, and it is worth seeing why it holds. The linear constraint fixes the weighted sum; each range check confines its word to a window; and because the windows tile, no two distinct word assignments produce the same sum. So exactly one assignment satisfies all the constraints, with one proviso. The argument runs over the integers, but the constraints run over the field, and a wrapped negative value has a representative near $P$. Uniqueness therefore additionally requires that the largest value the words can recompose, $\sum_n \text{coeff}_n (2^{w_n} - 1)$ plus any shift, lies below $P$: then no wrapped value is reachable by any valid assignment. This is the **width condition**, and every claim that leans on a decomposition for an exclusion argument must state it with its actual widths. Overflow of a decomposed operand is thus the rejecting case of §4.2: excluded by construction rather than priced by the bound.

Counts for a $t$-word decomposition: $t$ word slots plus their range checks, one linear constraint plus any signed shifts, $t$ quadratics.

**Soundness (Lemma B.1a).** As argued above: given the linear constraint, the range checks, and the width condition, exactly one word assignment satisfies the constraints.

**Rescale.** A rescale returns a raw product to the working scale. A product of two values at scale $S$ arrives at scale $S^2$; the rescale keeps the signed high word $x$ and drops the $r = \log_2(s_a s_b / s_{\text{out}})$ low bits, which are the quantization error the bound prices. It is the two-word instance of the decomposition, in the form the compiler emits: the decomposition constraint $x_{\text{full}} - 2^{r} x - x_{\text{low}} = 0$, the signed-shift relation $x_{\text{shifted}} - x = 2^{w-1}$, and the two range-check quadratics on $x_{\text{low}}$ and $x_{\text{shifted}}$.

Counts per element: five witness slots ($x$, $x_{\text{low}}$, $x_{\text{shifted}}$, two inverses), two linear, two quadratic; the input $x_{\text{full}}$ is counted by the invoking claim. The demonstrated widths are $r = 12$ and $w = 26$, so the kept word satisfies $x \in [-2^{25}, 2^{25})$.

**Soundness (Lemma B.1b).** Lemma B.1a at two words with the signed form. The width condition at the demonstrated parameters: the maximum recomposable value is $2^{12} \cdot 2^{25} + (2^{12} - 1) \approx 2^{37}$, against $P \approx 2^{64}$, about 27 bits of margin, so no wrapped negative has a valid decomposition and the kept word is the unique high part of the committed product.

### B.2 Matmul

Matmul proves $C_{\text{full}} = AB$ for $A \in F^{m \times k}$ and $B \in F^{k \times n}$.

$$
\begin{array}{lll}
\texttt{in} & A \in F^{m \times k},\; B \in F^{k \times n} & \\
\hline
\texttt{decl} & C_{\text{full}} \in F^{m \times n} & \text{raw product at } s_a s_b \\
\hline
\texttt{chal} & \rho \in F^{n},\; \lambda \in F^{m} & \\
\hline
\texttt{lin comp} & y[i] = \textstyle\sum_{j \in [n]} B[i,j]\,\rho[j] & \forall\, i \in [k] \\
\texttt{lin comp} & u[i] = \textstyle\sum_{j \in [m]} \lambda[j]\,A[j,i] & \forall\, i \in [k] \\
\texttt{quad comp} & p[i] = u[i]\,y[i] & \forall\, i \in [k] \\
\texttt{lin pin} & \textstyle\sum_{a \in [m],\, b \in [n]} \lambda[a]\,\rho[b]\,C_{\text{full}}[a,b] = \sum_{i \in [k]} p[i] & \\
\end{array}
$$

The output consumed downstream is $C = \texttt{rescale}(C_{\text{full}})$ at the working scale (B.1), included in the counts; the rescale's commitments ride the same pre-challenge turn as $C_{\text{full}}$.

**Counts.** $W = mn\,(C_{\text{full}}) + 5mn\,(\text{rescale}) + 3k\,(y, u, p)$; $\;L = 2mn\,(\text{rescale}) + 2k\,(y, u) + 1\,(\text{pin})$; $\;Q = 2mn\,(\text{rescale}) + k\,(p)$. Matches A.1's $6mn + 3k$, $2k + 1 + 2mn$, $k + 2mn$ at $H = 1$.

**Soundness (Lemma B.2).** The one declaration is $C_{\text{full}}$. The comp constraints define $y$, $u$, $p$ exactly: given the commitments and challenges, each has one satisfying value, and by associativity $\sum_i p[i] = (\lambda^\top A)(B\rho) = \lambda^\top (AB)\rho$. The pin therefore enforces $\lambda^\top C_{\text{full}}\,\rho = \lambda^\top (AB)\rho$, i.e. $\lambda^\top E \rho = 0$ for $E = C_{\text{full}} - AB$. If $E \neq 0$, then $E\rho \neq 0$ except with probability $1/|F|$ over $\rho$: a nonzero matrix has a nonzero row, and that row's inner product with a uniform $\rho$ is uniform. Conditioned on $E\rho \neq 0$, $\lambda^\top(E\rho) = 0$ with probability $1/|F|$ over $\lambda$. A false $C_{\text{full}}$ survives with probability at most $2/|F|$, and since it is committed before $(\rho, \lambda)$ are drawn, the prover cannot select it against the challenge; each matmul adds $2/|F|$ to the error sum of §5.4. Given $C_{\text{full}}$, the output $C$ is unique by Lemma B.1b. Overflow in the projection constraints is the accepting case of §4.2: field identities, wrapped values unique but wrong, priced by the bound; the magnitude exclusion lives in the rescale (Lemma B.1b).

**Generalization.** $H$-head batches share one challenge pair, with one pin constraint per head (coefficients $\lambda[i]\rho[j]$ on $C_{\text{full}}[i,h,j]$), the $O(k)$ terms counted once across heads: A.1's $+H$ term. `transpose_b` reindexes $B$ only.

### B.3 Softmax

Softmax proves, for each row of a $B \times M$ input block, the exponentiated outputs at scale $s_y$, normalized by pinning the per-row shift. Public constants beyond geometry: the exponential tables, computed by the deterministic rounding rule

$$T_A[j] = \mathrm{round}\big(s_y\, e^{-j/s_c}\big), \qquad T_B[j] = \mathrm{round}\big(s_y\, e^{(\delta - j)/s_c}\big),$$

with round-half-to-even, so that $T_B[j] = T_A[j - \delta]$ bit for bit; the shift unit $\delta = 1$; the table length $Z_{\max}$, doubled with a zero half under the causal variant; the target row sum $s_y$; and the saturation width $w_{\text{zh}}$ and slack width $w_{\text{aux}}$. In the demonstrated attention configuration $s_x = s_c = s_y = S$, $Z_{\max} = 40000$, $w_{\text{zh}} = 16$, $w_{\text{aux}} = 24$.

$$
\begin{array}{lll}
\texttt{in} & x \in F^{B \times M} \\
\hline
& \textit{--- shift and exponentiate ---} \\
\texttt{decl} & c \in F^{B} & \text{per-row shift} \\
\texttt{lin comp} & z[b,i] + Z_{\max}\, z_{\text{high}}[b,i] = c[b] - x[b,i] & \forall\, b,i \text{ unmasked} \;\; z_{\text{high}} \text{ range-checked (B.1)} \\
\texttt{lkp comp} & e_1[b,i] = T_A[z[b,i]] & \forall\, b,i \\
\texttt{lkp comp} & e_2[b,i] = T_B[z[b,i]] & \forall\, b,i \\
& \textit{--- saturate the tail ---} \\
\texttt{quad comp} & t[b,i]: \;\; z_{\text{high}}\cdot\mathit{inv} = t, \;\; t\cdot z_{\text{high}} = z_{\text{high}}, \;\; t^2 = t & \forall\, b,i \\
\texttt{quad comp} & \mathit{mux}_1[b,i] = t[b,i]\, e_1[b,i] & \forall\, b,i \;\; \text{(and likewise } \mathit{mux}_2 \text{ from } e_2\text{)} \\
\texttt{lin comp} & y_1[b,i] = e_1[b,i] - \mathit{mux}_1[b,i] & \forall\, b,i \;\; \text{(and likewise } y_2\text{)} \\
& \textit{--- bracket the shift ---} \\
\texttt{lin comp} & s_1[b] = \textstyle\sum_{i \in [M]} y_1[b,i] & \forall\, b \\
\texttt{lin comp} & s_2[b] = \textstyle\sum_{i \in [M]} y_2[b,i] & \forall\, b \\
\texttt{decl} & r_{\text{lo}}, r_{\text{hi}} \in F^{B} & \text{slacks, single-word range-checked (B.1)} \\
\texttt{lin pin} & s_1[b] + r_{\text{lo}}[b] = s_y & \forall\, b \\
\texttt{lin pin} & r_{\text{hi}}[b] - s_2[b] = -(s_y + 1) & \forall\, b \\
\end{array}
$$

The shift $c$ is committed via its signed form (B.1). The saturated outputs $y_1, y_2$ are what the row sums bracket, which is what the tables' zero tail requires; the listing groups the compiler's constraint families in expository order without changing the set.

**Counts.** Per cell, $W{:}\,15$: $z$, $z_{\text{high}}$, the range inverse of $z_{\text{high}}$, the flag $t$ and its defining inverse ($5$), the two lookups' values, folded keys, and inverses ($6$, B.1), the two mux products and two muxed outputs ($4$). $L{:}\,4\tfrac12$: the key decomposition on unmasked cells only ($\tfrac12$, see the causal note), the two lookup foldings, the two mux recoveries. $Q{:}\,8$: the two lookup inverses, the three flag constraints, the two mux products, the $z_{\text{high}}$ range check. Per row, $W{:}\,9$: $c$, its shifted form and inverse, $s_1$, $s_2$, the two slacks and their inverses. $L{:}\,5$: the two row sums, the two bracket pins, the signed shift. $Q{:}\,3$: the three range inverses. Totals $W = 15BM + 9B$, $L = \tfrac12 B(M{+}1) + 4BM + 5B$, $Q = 8BM + 3B$, matching A.1's row exactly.

**Soundness (Lemma B.3).** The declarations are $c$, $r_{\text{lo}}$, $r_{\text{hi}}$; discharge the slacks given $c$, then argue $c$. Each slack is fixed by its linear equality once $s_1, s_2$ are (comps), and its range check forces it non-negative, so the pins state $s_1(c) \le s_y$ and $s_2(c) \ge s_y + 1$. For $c$: the lookups force every table index in range and $z_{\text{high}}$ is range-checked, so the key decomposition gives $c[b] - x[b,i] \ge 0$ for every unmasked cell, i.e. $c \ge \max_i x[b,i]$; the tables are bit-identical up to the shift, and because $Z_{\max}$ is sized past natural saturation ($T_A[j] = 0$ for $j \ge Z_{\max}$), a muxed cell equals what an unbounded table would return, so $s_2(c) = s_1(c - \delta)$ holds over the muxed sums as exact integers; $s_1$ is monotone non-increasing in $c$; hence the two inequalities hold for exactly one integer $c$, the crossing point, with no tolerance. Given $c$, everything downstream is comp: the key decomposition (Lemma B.1a), the lookups, the flag ($t$ is determined by whether $z_{\text{high}}$ is zero), the mux products and recoveries. Overflow is the rejecting case of §4.2 at the bracket, whose monotonicity argument presumes genuine integer sums: the slacks are range-checked at $w_{\text{aux}} = 24$, and since $s_2 \le M\, s_y$, the honest-fit condition is $M s_y \lesssim 2^{w_{\text{aux}}}$, which at $s_y = 2^{12}$ limits $M \lesssim 4096$ and must grow with context; all checked values sit below $2^{24} \ll P$, so recomposition exclusion holds with about 40 bits of margin. Elsewhere the accepting case.

**Generalization.** The causal variant doubles each table with a zero half; the effective key is $z + Z_{\max} \cdot \mathrm{mask}(b,i)$, with the mask a public constant baked into the constraint right-hand sides, nothing committed, so a masked cell looks up zero and a later token contributes nothing to an earlier row. The masked triangle still runs the lookups and quadratics over all $BM$ cells; only the key-decomposition linear family is filtered to unmasked cells, giving the $\tfrac12 B(M{+}1)$ term, whose $+1$ is the diagonal (query $i$ has $i + 1$ unmasked cells). Attention instantiates $B = n_q S$, $M = S$.

### B.4 RMSNorm

RMSNorm proves $\text{out}[b,i] = x[b,i]\, y[b]$ with $y[b]$ the rounded reciprocal square root $\lceil \sqrt{\text{magic}/S_{\text{total}}[b]} \rceil$, where $S_{\text{total}}[b] = \sum_i x[b,i]^2 + d\varepsilon$ and $\text{magic} = d s^4$, over $B$ rows of width $d$. It is the one nonlinearity with no lookup table: the rsqrt is pinned by an algebraic bracket. Because the bracket compares products of committed values, its operands are assembled from range-checked limbs so the comparison holds over the integers; all limb windows are derived from $(d, s, \varepsilon)$ by both sides independently, never chosen by the prover. At the demonstrated parameters the limb width is $L_{\text{limb}} = 18$, the $y$ window is 21 bits, the slack window 59 bits, and the carry windows 42, 43, and 25 bits.

$$
\begin{array}{lll}
\texttt{in} & x \in F^{B \times d} & \\
\hline
\texttt{quad comp} & X_{\text{sq}}[b,i] = x[b,i]^2 & \forall\, b,i \\
\texttt{lin comp} & S_{\text{total}}[b] = \textstyle\sum_{i \in [d]} X_{\text{sq}}[b,i] + d\varepsilon & \forall\, b \\
\texttt{decl} & y \in F^{B} & \text{the rsqrt scalars} \\
\texttt{lin comp} & y_{m1}[b] = y[b] - 1 & \forall\, b \\
\texttt{gadget} & \text{word-decompose } y_{m1} \text{ (21-bit window), } S_{\text{total}} \text{ into 3 limbs of } L_{\text{limb}} \text{ bits (B.1)} & \\
\texttt{quad comp} & q_1[b] = y[b]^2, \quad q_2[b] = y_{m1}[b]^2 & \forall\, b \\
\texttt{quad comp} & H_k[b] = q[b]\, S_k[b] \quad k \in \{0,1,2\}, \; q \in \{q_1, q_2\} & \forall\, b \\
\texttt{lin comp} & H_0 = g_{0l} + 2^{L_{\text{limb}}} g_{0h}; \;\; H_1 + g_{0h} = g_{1l} + 2^{L_{\text{limb}}} g_{1h}; \;\; H_2 + g_{1h} = G_2 & \text{carry chain, per } q; \text{ all parts range-checked} \\
\texttt{decl} & s_{\text{lo}}, s_{\text{hi}} \in F^{B} & \text{slacks, word-decomposed (B.1)} \\
\texttt{lin pin} & 2^{2 L_{\text{limb}}} G_2 + 2^{L_{\text{limb}}} g_{1l} + g_{0l} - s_{\text{lo}} = \text{magic} & \text{for } q_1 \;\; \forall\, b \\
\texttt{lin pin} & 2^{2 L_{\text{limb}}} G_2 + 2^{L_{\text{limb}}} g_{1l} + g_{0l} + s_{\text{hi}} = \text{magic} - 1 & \text{for } q_2 \;\; \forall\, b \\
\texttt{decl} & \text{out}_{\text{full}} \in F^{B \times d} & \text{broadcast product at } s^2 \\
\hline
\texttt{chal} & \rho \in F^{d} & \\
\hline
\texttt{lin comp} & u[b] = \textstyle\sum_{i \in [d]} \rho[i]\, x[b,i], \quad p[b] = \textstyle\sum_{i \in [d]} \rho[i]\, \text{out}_{\text{full}}[b,i] & \forall\, b \\
\texttt{quad pin} & y[b]\, u[b] = p[b] & \forall\, b \\
\end{array}
$$

The output consumed downstream is $\texttt{rescale}(\text{out}_{\text{full}})$ at the working scale (B.1). The per-cell broadcast products $x[b,i]\,y[b]$ are never committed: they are the virtual values of the preamble's convention, folded into the projection comps, which is what collapses $Bd$ Hadamard slots to $B$.

**Counts.** Per cell: $X_{\text{sq}}$, $\text{out}_{\text{full}}$, and the rescale's five, matching A.1's $7Bd$; the quadratics $X_{\text{sq}}$ plus the two rescale checks match $3Bd$, and the rescale's two linears the $2Bd$. Per row the counts follow the derived windows: at the demonstrated widths the bracket machinery (limbs, carry parts, chunked windows, slacks, and their range inverses, plus $u$, $p$) comes to roughly eighty slots per row. A.1's per-row constant of $26B$ predates the limb-assembled bracket and understates it; the difference is far below the resolution of A.2's validation, and the per-cell terms, which carry the cost, are unchanged.

**Soundness (Lemma B.4).** The declarations are $y$, the slacks, and $\text{out}_{\text{full}}$. The $y_{m1}$ decomposition bounds $y \in [1, 2^{21}]$, the limb decomposition bounds $S_{\text{total}} \lt 2^{3 L_{\text{limb}}} = 2^{54}$, and each carry part is range-checked, so every term in the carry chain sits below $2^{61} \lt P$: the chain telescopes to $q\,S_{\text{total}}$ exactly, and the two pins read $y^2 S_{\text{total}} \ge \text{magic}$ and $(y-1)^2 S_{\text{total}} \le \text{magic} - 1$ as integer inequalities, not congruences. Since $q \mapsto q\,S_{\text{total}}$ is strictly increasing in $y \ge 1$, exactly one integer $y$ satisfies both, and the slacks are then fixed by their equalities (their windows are sized to the largest honest bracket step, with the width condition $\text{magic} + 2^{59} \lt P$). Given $y$, the projection comps and the quad pin force $\text{out}_{\text{full}} = x \odot y$ row by row except with probability $1/|F|$ per row over $\rho$, drawn after everything above is committed. Overflow is the rejecting case of §4.2 throughout the bracket: an $S_{\text{total}}$ with no limb representation below $2^{54}$ has no satisfying decomposition and the proof rejects (the honest completeness cap is a row RMS of about 460 at the demonstrated scales).

**Generalization.** The elementwise gain multiply that follows the normalization in the transformer block is a separate Hadamard claim (B.8). The pre-fix bracket, which range-checked the slacks in a window wide enough to admit every field element, is unsound and documented in the negative test suite; the listing above is the repaired construction the demonstrated runs used.

### B.5 SiLU

SiLU proves $\text{out} = x\,\sigma(x)$, saturating to $x$ for large positive $x$ and to $0$ for large negative $x$: a sign split, a magnitude decomposition, a paired lookup, and a saturation mux. Public constants: the paired table, holding the positive and negative branches concatenated, computed by the rounding rule $T[\pm i] = \mathrm{round}(\mathrm{silu}(\pm(ib + b/2)/s_x)\, s_x)$ at bin centres of width $b$; in the demonstrated configuration $b = 4$, table length $2^{14}$ per branch, $s_x = S$, and the magnitude windows are $2$, $14$ (the table index), $16$, $16$, and $14$ bits with strides $1, b, 2^{16}, 2^{32}, 2^{48}$.

$$
\begin{array}{lll}
\texttt{in} & x \in F^{N} & \\
\hline
\texttt{decl} & \mathit{sign} \in F^{N} & \\
\texttt{quad comp} & \mathit{sign}^2 = \mathit{sign}, \quad C = \mathit{sign}\cdot x & \forall\, n \\
\texttt{lin comp} & \mathit{mag} = x - 2C & \forall\, n \\
\texttt{gadget} & \mathit{mag} = a_0 + b\,a_1 + 2^{16} a_2 + 2^{32} a_3 + 2^{48} a_4 \text{ (B.1; } a_1 \text{ range-proved by the lookup)} & \\
\texttt{lin comp} & g = 2^{16} a_2 + 2^{32} a_3 + 2^{48} a_4 & \forall\, n \\
\texttt{quad comp} & t: \;\; g \cdot \mathit{inv} = t, \;\; t \cdot g = g, \;\; t^2 = t & \forall\, n \\
\texttt{lin comp} & \mathit{key} = T_{\text{LEN}}\,\mathit{sign} + a_1; \quad \mathit{sat} = x - C & \forall\, n \\
\texttt{lkp comp} & y = T[\mathit{key}] & \forall\, n \\
\texttt{quad comp} & \mathit{mux}_a = t\, y, \quad \mathit{mux}_b = t\,\mathit{sat} & \forall\, n \\
\texttt{lin comp} & \text{out} = y - \mathit{mux}_a + \mathit{mux}_b & \forall\, n \\
\end{array}
$$

**Counts.** Per element, $W{:}\,23$: sign, $C$, magnitude, five words, four word-range inverses ($a_1$ needs none), $g$, its inverse, $t$, key, sat, the two mux products, the lookup's value, folded key, and inverse, and the output. $L{:}\,7$: the sign split, the decomposition, $g$, the key and sat pair, the mux recovery, the lookup folding. $Q{:}\,12$: the two sign quads, the three flag quads, the two mux products, four word range checks, the lookup inverse. Matches A.1's SiLU row exactly.

**Soundness (Lemma B.5).** The one declaration is $\mathit{sign}$. The magnitude decomposition bounds $\mathit{mag} \lt 2^{63}$ (width condition: the maximum recomposable magnitude is about $2^{62}$, below $\lceil P/2 \rceil$), and with $\mathit{sign}$ boolean and $C = \mathit{sign}\cdot x$, the pair $(x = \mathit{mag} + 2C,\ \mathit{mag}$ bounded$)$ admits exactly one sign for each $x$: the wrong sign would need a magnitude of the wrong parity of $P$, above the bound. Everything else is comp: the words by Lemma B.1a with the lookup bounding $a_1$, the flag by whether $g$ is zero, the mux and output linearly. Overflow: the decomposition is the rejecting case of §4.2; the saturated path returns $x$ or $0$ exactly.

### B.6 Mixture-of-experts routing and combine

Routing pins a one-hot mask $m \in \{0,1\}^{T \times E}$ to the argmax of committed router logits $r$, tiebroken by expert index; the combine forms $y[t,:] = \sum_e m[t,e]\, X_e[t,:]$ over the $E$ committed expert streams without committing the $E\,T\,F$ masked products. Public constants: the tiebreak stride $2^{L}$ with $L = \lceil \log_2 E \rceil$, and the gap word width $w_g$ (demonstrated: $E = 128$, $L = 7$, $w_g = 11$, three words covering the $B + L = 33$-bit gap window).

$$
\begin{array}{lll}
\texttt{in} & r \in F^{T \times E}, \quad X_e \in F^{T \times F} \; \forall\, e & \\
\hline
\texttt{decl} & m \in F^{T \times E} & \text{the mask} \\
\texttt{lin comp} & \mathit{rt}[t,e] = 2^{L} r[t,e] + (E{-}1{-}e) & \forall\, t,e \;\; \text{tiebroken logits} \\
\texttt{quad comp} & m^2 = m; \quad \mathit{mrt} = m \cdot \mathit{rt} & \forall\, t,e \\
\texttt{lin pin} & \textstyle\sum_e m[t,e] = 1 & \forall\, t \\
\texttt{lin comp} & r^{*}[t] = \textstyle\sum_e \mathit{mrt}[t,e]; \quad \mathit{gap}[t,e] = r^{*}[t] - \mathit{rt}[t,e] & \forall\, t \;(,e) \\
\texttt{gadget} & \text{word-decompose each } \mathit{gap} \text{ into } 3 \times 11\text{-bit words (B.1)} & \\
\texttt{lin comp} & 2^{L} r_{\text{chosen}}[t] = r^{*}[t] - \textstyle\sum_e (E{-}1{-}e)\, m[t,e] & \forall\, t \\
\hline
\texttt{chal} & \rho \in F^{F} & \\
\texttt{decl} & y \in F^{T \times F} & \text{the combined output} \\
\texttt{lin comp} & s[e,t] = X_e[t,:]\cdot\rho; \quad y_\rho[t] = y[t,:]\cdot\rho & \forall\, e,t \;(,t) \\
\texttt{quad comp} & \mathit{ms}[e,t] = m[t,e]\, s[e,t] & \forall\, e,t \\
\texttt{lin pin} & \textstyle\sum_e \mathit{ms}[e,t] = y_\rho[t] & \forall\, t \\
\end{array}
$$

**Counts.** Routing: $W = 10TE + 2T$ ($m$, $\mathit{rt}$, $\mathit{mrt}$, $\mathit{gap}$, three words and three range inverses per gap; $r^{*}$ and $r_{\text{chosen}}$ per token), $L = 3TE + 3T$, $Q = 5TE$ (booleanity, $\mathit{mrt}$, three word checks). Combine: $W = TF + 4ET + T$ ($y$; $s$, $\mathit{ms}$, the mask and product transposes; $y_\rho$), $L = 3ET + 2T$, $Q = ET$. Both match A.1's rows (the compiler carries expert-major and token-major copies of the mask and products as pure re-indexings, included in the $4ET$).

**Soundness (Lemma B.6).** Routing's declaration is $m$. The tiebreak makes all $\mathit{rt}$ in a row distinct, booleanity and cardinality force $m$ one-hot, and $r^{*}$ is then the selected tiebroken logit. If the selection were not the argmax, some $\mathit{gap}[t,e]$ would be a negative field element near $P$, which the $3 \times 11$-bit decomposition cannot recompose (width condition: $2^{33} \ll P$, given that $r$ is bounded to $\pm 2^{B-1}$ by its producing matmul's rescale, $B = 26$); so $m$ is the unique argmax mask and $r_{\text{chosen}}$ follows linearly. The combine's declaration is $y$: for each token, $\sum_e m[t,e]\,s[e,t]$ equals the projection of $\sum_e m[t,e] X_e[t,:]$ by the one-hotness of $m$, so the pin forces $y[t,:]\cdot\rho$ to match the combined stream's projection, and a false $y$ row survives with probability $1/|F|$ over $\rho$, drawn after $y$ and the streams are committed. Overflow: the gap gadget is the rejecting case of §4.2; the projections are the accepting case.

**Generalization.** All $E$ expert streams are committed even though one fires, a hiding requirement (§4.5), and the elementwise nonlinearity is applied once after the combine in the top-1 form. The same gap gadget, at $E = V$, is the argmax and hidden-select machinery of the surprisal claims (B.7).

### B.7 The unexplained-information bound

The bound of §2.1 is computed from the LM-head logits by a short chain of claims per output position, reusing the gap gadget (B.6), the paired lookup (B.1), and the elementwise claims; Appendix E binds the same committed tokens to the digests recorded at generation time (§2.4). Its soundness property is the weaker downstream one of §2.2: the witness is deliberately not unique, and instead every prover freedom provably inflates the reported value. Public constants: the exponential-kernel table $\mathrm{EXP}[g] = \max(1, \lceil s_y\, e^{-g^2/s_c} \rceil)$ over $[0, \mathit{gap}_{\max})$, the logarithm table $\mathrm{POW}[j] = \lfloor s_y\, e^{j/s_b} \rfloor$ over $[0, K)$ with $K = \lceil s_b \ln V \rceil + 4$, and the ceiling divisor $k = s_c/s_b$, a power of two. Demonstrated: $s_c = s_y = 2^{28}$, $s_b = 2^{12}$, $\mathit{gap}_{\max} = 2^{20}$, $k = 2^{16}$, and the slack words $w_b = 12$ against $d_{\max} = V s_y$. All in-circuit arithmetic is in nats at scale $s_b$; the public value is the revealed sum $S_z$, and the conversion to bits, $U(o) = S_z / (s_b \ln 2)$ rounded up, happens outside the proof.

Per position $t$, with $\ell \in F^{V}$ the logit row:

$$
\begin{array}{lll}
\texttt{in} & \ell \in F^{V}, \quad \mathit{tok}[t] & \text{logits; the committed token stream} \\
\hline
\texttt{decl} & A, O \in F^{V} & \text{argmax and output-select one-hots} \\
\texttt{quad comp} & A^2 = A, \; O^2 = O; \quad A\ell = A \cdot \ell, \; O\mathit{gap} = O \cdot \mathit{gap} & \forall\, i \\
\texttt{lin pin} & \textstyle\sum_i A[i] = 1, \quad \textstyle\sum_i O[i] = 1, \quad \textstyle\sum_i i\,O[i] = \mathit{tok}[t] & \\
\texttt{lin comp} & v^{*} = \textstyle\sum_i A\ell[i]; \quad \mathit{gap}[i] = v^{*} - \ell[i]; \quad \mathit{gap}_o = \textstyle\sum_i O\mathit{gap}[i] & \\
\texttt{lkp comp} & e[i] = \mathrm{EXP}[\mathit{gap}[i]] & \forall\, i \;\; \text{also bounds } \mathit{gap}[i] \ge 0 \\
\texttt{quad comp} & g_2 = \mathit{gap}_o \cdot \mathit{gap}_o & \\
\texttt{lin comp} & a = \textstyle\sum_i e[i] & \text{the normalizer} \\
\texttt{decl} & b & \text{the log-pin index} \\
\texttt{lkp comp} & \mathit{pw} = \mathrm{POW}[b] & \\
\texttt{lin pin≤} & a + \mathit{d} = \mathit{pw}, \quad \mathit{d} \text{ word-decomposed (B.1)} & a \le \mathrm{POW}[b] \\
\texttt{lin comp} & k\, z_o = g_2 + \mathit{rem}, \quad \mathit{rem} \text{ range-checked in } [0, k) & z_o = \lceil g_2 / k \rceil \\
\texttt{lin comp} & \mathit{surprisal}[t] = z_o + b & \\
\hline
& \textit{--- across positions ---} & \\
\texttt{lin comp} & S_z = \textstyle\sum_{t} \mathit{surprisal}[t] & \text{chained adds over the scored positions} \\
\texttt{lin pin} & S_z = \text{the revealed public value} & \\
\end{array}
$$

The output tokens enter only as the committed stream $\mathit{tok}$ consumed by the select pins; they never appear in the public claim list, and the indicator rows $O$ are shared with the input selection (Appendix E.4), so the scored tokens are the tokens the model consumed.

**Counts.** Per position the $V$-length families dominate: $A$, $A\ell$, $O$, $O\mathit{gap}$, $\mathit{gap}$, and the EXP lookup's three slots, roughly $8V$ slots with $O(1)$ scalars (the finalize's $a$, $b$, $\mathit{pw}$, $\mathit{d}$ and its four words and inverses, $z_o$, $\mathit{rem}$, and the surprisal). These claims are excluded from the cost model by construction (A.6); they are about 0.1% of the per-token witness.

**Soundness (Lemma B.7, one-sided).** The declarations are $A$, $O$, and $b$; $O$ is pinned to the unique indicator of $\mathit{tok}[t]$ by its three pins (Lemma B.6's argument at $E = V$), and $A$ with the gap non-negativity forces $v^{*} = \max_i \ell[i]$ exactly. The remaining freedom is $b$, and every free direction inflates. Each table entry $e[i]$ rounds the true exponential up and is floored at one, so $a$ over-counts the true normalizer; $\mathrm{POW}$ rounds down, so $a \le \mathrm{POW}[b]$ forces $b \ge s_b \ln(a/s_y)$ with no fractional escape, and choosing $b$ above the least valid index only raises the reported value; $z_o$ is a ceiling. With $Q_t(o) = e_o/a$, a genuine distribution since $a$ normalizes the committed table values exactly, $z_o + b \ge s_b(-\ln Q_t(o))$ follows term by term, so $S_z$ upper-bounds the true surprisal sum and a poor witness penalizes only the prover. Two freedoms need field arguments rather than integer ones. The word decomposition of $\mathit{d}$ is safe by width: at $w_b = 12$ and $d_{\max} = V s_y$ the maximum recomposable value lies far below the modulus, so no wrapped negative $\mathit{d}$ has a valid decomposition (Lemma B.1a). The ceiling constraint is subtler: over the integers $z_o$ is unique given $g_2$, but over the field every range-valid remainder admits a solution $z_o = k^{-1}(g_2 + \mathit{rem}) \bmod P$, and $z_o$ carries no range check. The reachable perturbations of the public sum are $S_z' = S_z + k^{-1} s \bmod P$ for integer $s \in [0, Tk)$; deflating by any amount requires $s = P - k\Delta$, on the order of $2^{64}$ and unreachable by roughly twenty orders of magnitude, while every reachable perturbation either inflates the bound by less than $T$ scaled-nat units (about 1.5 bits in total at $T = 1000$, the safe direction) or lands $S_z$ near the modulus, an absurd self-reported bound useless to a deflating prover. The claim is sound by exclusion rather than by uniqueness; this is the one place in the construction where that argument is load-bearing.

**Generalization.** Normalization is where the asymmetry between §2.2's two requirements pays: softmax pins its shift exactly because upstream slack is unanalyzable, while the bound replaces normalization with the one-sided logarithm pin because downstream slack provably only inflates. Summing over a subset of positions bounds the unexplained information of just those outputs; the demonstrated runs score the 500-token continuation.

### B.8 Remaining claims

The remaining claim types are compositions of B.1 gadgets with no soundness argument beyond Lemma B.1a and the comp rule; their listings are omitted and their counts, matching A.1, are:

| claim | gadgets | $W$ | $L$ | $Q$ |
|---|---|---|---|---|
| add $(N)$ | none | $N$ | $N$ | $0$ |
| hadamard $(N)$ ⓡ | rescale | $6N$ | $2N$ | $3N$ |
| rope $(N)$ ⓡ | rescale; public cos/sin coefficients | $6N$ | $3N$ | $2N$ |
| paired lookup $(N)$ | paired lookup | $3N$ | $N$ | $N$ |
| word extraction $(N, t)$ | word decomposition | $2tN$ | $N$ | $tN$ |
| embedding select $(T, V)$ | one-hot select (B.7's pins); Freivalds against the embedding matrix | $O(TV)$ | $O(TV)$ | $O(TV)$ |
| masked combine $(T,E,F)$ | committed products form | $2ETF + TF$ | $ETF + TF$ | $ETF$ |

The masked combine is the committed-products alternative the demonstrated runs replace with B.6's projected form. The token-binding circuits (AES, SHA-256) are specified in Appendix E at their own granularity.

## Appendix C. Constraint compilation and evaluation

This appendix specifies how the flat constraint system of §4 is represented and evaluated: the generative constraint level shared by the prover and the verifier, the run structure both evaluation loops exploit, the challenge-access discipline, and the two folds themselves. The governing constraint is scale. The linear system has $\Theta(\text{nnz})$ nonzeros with $\text{nnz} \approx 2\text{–}4\,W$, tens of terabytes at the scale of §7, so no materialized form of the constraints ever exists on either side. Everything below is regenerated on demand from descriptors whose total size is $O(\#\text{claims})$, a few tens of megabytes at 400B scale.

### C.1 The constraint level: bands and quadratic descriptors

Each claim compiles, independently on each side (the verifier recompiles from the public claim list and never reads prover-supplied constraints, §6.2), into three kinds of object:

- **Linear bands.** Each variable carries a small list of bands, one per constraint pattern it participates in: a kind tag, a constraint-id base, and the kind's parameters (roughly 100 bytes). A band maps each flat slot $f$ of its variable to constraint ids and coefficients by closed-form index arithmetic; the map is a pure function of the descriptor and the variable's geometry, so any row window can be evaluated without state. Parameter vectors (Freivalds $\rho, \lambda$; lookup-table coefficients; RoPE tables) are sized $O(k)$ or $O(\text{table})$, never $O(\text{nnz})$, and are shared, not copied per row.
- **Quadratic descriptors.** One per quadratic emission: the three operands' starting rows, the uniform $(a, b)$ coefficients, the slot count $L$, and a positional index base. Row $t$ of a descriptor is the per-row constraint $w_{x+t} \circ w_{y+t} + a\,w_{z+t} = b$ over $\min(\mathrm{ELL},\, L - t\cdot\mathrm{ELL})$ slots, and its combiner index is $\text{base} + t$. This replaces a per-row constraint list of size $O(W/\mathrm{ELL})$, a verifier-memory binder at long context, with $O(\#\text{emissions})$.
- **Right-hand sides**, kept as compact runs $(\text{start}, \text{length}, \text{value})$.

Constraint ids and quadratic indices advance in claim order; this positional numbering is the entire cross-side contract. The fold combiners $r_{\text{lin}}[g]$ and $r_{\text{quad}}[t]$ are values of a hash PRF on $(s_{\text{comb}}, \text{index}, \text{label})$ and are never materialized globally: both sides derive any combiner in $O(1)$ from the round seed, so no challenge vectors cross the wire. Because a quadratic descriptor carries its index base, firing order is immaterial: each row fetches its own combiner, and field addition commutes.

### C.2 Run structure and challenge access

A band's slot-to-(id, coefficient) map decomposes into maximal homogeneous runs of four shapes, and the shape is a static property of the band kind:

| shape | structure | challenge access |
|---|---|---|
| repeat | a run of slots shares one id | one PRF call per run |
| strided repeat | $\text{id} = \text{base} + (f \bmod k)$: $k$ distinct ids recur on every row | preload $[\text{base}, \text{base}+k)$, cached |
| one-to-one | $\text{id} = \text{base} + f$ (RoPE: stride 2) | one PRF call per slot |
| fan | one slot feeds a contiguous id range | streamed range sum |

Row sums and the Freivalds $B$/$C$ sides are repeats; the Freivalds $A$ side is the strided repeat, whose ids have no contiguous runs but span only $k \le H\cdot K$ ids ($\le 128$ KB preloaded; the prover caches these buffers for the whole proof, across all chunks and layers, since $s_{\text{comb}}$ is fixed once per round). Identity pins and lookups are one-to-one, where one hash per distinct id is the floor; broadcasts are fans, where only the range *sum* is needed, so the range is never buffered. The effect is that challenge hashing costs $O(\text{distinct ids})$ on the duplication-heavy bands rather than $O(\text{nnz})$: a weight matmul's $B$-side reuses each id $n$ times, an expert matmul's $A$-side $m$ times.

### C.3 The prover's fold (round 3)

The linear test polynomial is $q_{\text{lin}} = \sum_i R_i \cdot p_i$, where $p_i$ is row $i$'s committed codeword polynomial and $R_i$ interpolates row $i$'s slice of $r_{\text{lin}}^{\top} A$. The prover computes it during the same tape-order sweep that regenerates the witness (§6.1), in two stages per 256-row chunk:

1. **Band evaluation** (kind-specific, per band, internally uniform): the band index (descriptors sorted by their variables' disjoint row ranges) yields the bands overlapping the chunk in one binary search; each band evaluates its intersection window into the chunk's $r^{\top}A$ rows, reading challenges per its shape.
2. **The transform fold** (kind- and variable-agnostic, batched): interpolate the chunk's $r^{\top}A$ rows to coefficients (inverse NTT), forward-transform both factors, multiply pointwise, and accumulate the products in the evaluation domain; one inverse NTT at the end of the fold recovers $q_{\text{lin}}$ (exact, since the inverse transform is linear), replacing a per-row inverse transform. Rows are freed afterwards, preserving the working-set memory bound.

Quadratic descriptors fire at their declaring claim, where the sweep's value liveness guarantees all three operands are resident: their rows are re-encoded on demand (exact, because the zero-knowledge padding of §5.2 is generated by a PRG seeded with the absolute row index, so re-encoding reproduces the committed polynomial bit for bit), and the pointwise products fold into $p_0$ under the positionally indexed combiners.

### C.4 The verifier's evaluation

The verifier recompiles the same bands and quadratic descriptors from the public claim list, then evaluates them at the opened columns rather than over full polynomials. The linear sum check needs no witness at all: $\sum_c q_{\text{lin}}(\zeta_c)$ is compared against the right-hand-side runs, each run one PRF range sum. The linear column check reconstructs, for every opened point $\eta_j$, each row's $R_i(\eta_j)$ through the closed form for a message slot's contribution to a codeword value (no NTT), as one generic fold over the runs: a repeat run takes one challenge and a prefix-summed-Lagrange difference (constant coefficients) or a coefficient dot (vector coefficients, with the Freivalds $\lambda$ factored out per run); strided repeats read the band's preload; a fan takes one challenge range sum shared across all $T$ opened points. The quadratic column check walks the quadratic descriptors' rows with their positional combiners.

One implementation of the index arithmetic sits on the verdict path. A second, per-term generator, line-for-line with the prover's, exists only as a test oracle: property tests compare the two as complete (slot, id, coefficient) triple sets over adversarial row windows for every band kind, so the trusted base carries a single copy of the geometry while its correctness is checked against an independent one.

### C.5 Equivalence discipline

Every representation change above (per-row structures to bands and descriptors, per-term evaluation to runs, inline hashing to preloads) is a regrouping of exact field operations, so equality of results is exact, not approximate, and the development gates demand it: verifier builds are compared by per-check *values* (not verdicts) on stored proofs; the prover's fold was compared chunk-by-chunk against an unmodified reference path until its retirement; tampered proofs must still be rejected after every change; and cross-language agreement is tested by expanding both compilers' outputs to canonical triples. The positional numbering of C.1 is what makes this possible: no reorganization changes which challenge multiplies which constraint, so any drift in any bit is a defect by definition.

## Appendix E. Token binding

The bound of §2.1 is conditioned on the input tokens and scored on the output tokens, so it certifies the real run only if the committed streams are the streams the run actually used (§2.4). Inside the proof they are ordinary hidden witness: the claims force the output tokens to be *some* valid selections, not the ones the deployment emitted, and a prover could commit a lower-surprisal transcript and deflate the bound, or condition on a fabricated prompt and certify nothing. This appendix gives the construction that closes the gap by binding both committed streams to a commitment recorded independently at generation time, outside the proof.

### E.1 The recorded commitment

At generation time, an independent recording process computes, for each request/response exchange,

$$H_{1,\text{in}} = H(\mathrm{AES}(\text{key}, \text{tokens}_{\text{in}})), \qquad H_{1,\text{out}} = H(\mathrm{AES}(\text{key}, \text{tokens}_{\text{out}})), \qquad H_2 = H(\text{key material}),$$

with $H$ = SHA-256 and AES in counter mode. These three digests are public inputs to the proof. One key covers the exchange, and $H_2$ is fixed with the request, before the response exists, so the key material cannot be chosen after the fact to fit a covert payload. The prover commits the tokens and the key material in the first round, hidden behind the Merkle root like the weights, and proves

$$H(\mathrm{AES}(\text{key}, \text{tokens}_s)) = H_{1,s} \;\; \text{for } s \in \{\text{in}, \text{out}\}, \qquad H(\text{key material}) = H_2.$$

### E.2 Soundness and the root of trust

Both digests are required. $H_1$ alone is vacuous: it fixes the ciphertext $C$, but for any key the prover grinds, decrypting $C$ under it yields *some* token stream that re-encrypts to $C$, so the tokens would be free. $H_2$ pins the key by collision resistance; with the ciphertext and the key both fixed, the token stream is the unique decryption. The binding therefore has the stronger of §2.2's two properties, a unique satisfying assignment, obtained from collision resistance rather than from constraint structure.

What the binding then certifies is exactly this: *the committed tokens equal the ones that produced the pre-recorded digests*. That is meaningful only if the record is produced by a process the verifier trusts, independently of and prior to the proof: for example, network-boundary hardware that hashes traffic as it passes and certifies the digests on a fixed schedule. A record the prover can rewrite after the fact makes the binding circular, and the assumption should be stated wherever the bound is reported.

### E.3 Confidentiality

Token streams are low-entropy, about 18 bits per token against a 202,048-token vocabulary, so a bare hash of the tokens would be a dictionary-searchable commitment and would leak them. Encrypting under a high-entropy committed key first makes $H_1$ a hiding commitment, preserving zero-knowledge: the digests reveal nothing about the tokens beyond their length. A deployment that chooses to reveal one stream simply publishes it alongside the proof and drops the hiding for that side; the binding equations are unchanged.

### E.4 One committed integer per token

The committed token integer $t_i$ is the single interface between the model side and the wire side of the proof, and every connection is a copy constraint on shared witness slots. On the input side, a select claim commits an indicator row $M_i$ over the vocabulary with booleanity $M_i \circ M_i = M_i$, cardinality $\sum_j M_{ij} = 1$, and the index binding $t_i = \sum_j j \cdot M_{ij}$ (the three together pin $M_i$ uniquely as the indicator of $t_i$), and the embedded stream is $x = M E$ by a Freivalds matmul against the embedding matrix, which is a committed model weight in any case. On the output side, the argmax claim's select gadget (B.7) carries the same index binding, so the observed-token gap that feeds the surprisal is evaluated at $t_i$ by construction. On the wire side, a word decomposition splits each $t_i$ into a fixed serialization of four little-endian bytes, which feed the cipher. Each of these links is load-bearing: without any one of them, the binding would attest to a different token set than the one the bound is computed over.

### E.5 The cipher and hash in constraints

Both primitives are lookup arguments over the existing table machinery, and all values stay below $2^{32}$, so the width argument of B.1 rules out field wraparound throughout. AES-128-CTR costs, per 16-byte block: sixteen S-box paired lookups per round against a 256-entry table; MixColumns as an *xtime* lookup plus linear constraints (doubling in $\mathrm{GF}(2^8)$); AddRoundKey and all other XORs against a $2^{16}$-entry byte-pair table; ShiftRows as wiring. The key schedule reuses the S-box table and is proven once per key. SHA-256 costs, per 64-byte block: the $\sigma$/$\Sigma$/Ch/Maj functions on 16-bit-limb XOR and AND tables, rotations as decomposition rewiring, and mod-$2^{32}$ additions with one range-checked carry word each; message padding is public structure compiled into the constraints.

Counter mode is chosen for compatibility in both directions. Hardware that encrypts with AES-GCM needs no additional circuit support: GCM's ciphertext is exactly counter-mode output, with counter blocks derived from the IV, which rides in the committed key material behind $H_2$, so no $\mathrm{GF}(2^{128})$ authentication arithmetic enters the proof. The 16-byte GCM tag cannot be partially explained inside a hash preimage, so either the recorded payload is defined as ciphertext-only, or the tag enters the preimage as unconstrained witness and is charged to the reported bound at 128 bits per packet. The fixed four-byte serialization keeps every token position-addressable and inside a single keystream block, so the same byte layout serves the circuit, the recorder, and any external process that re-derives per-token ciphertext units from the record.

### E.6 Cost

The binding runs over kilobytes of tokens, not the model. A few-thousand-token transcript costs on the order of $10^6$ constraint rows for both streams together, under 0.01% of the forward proof's witness (Appendix A) and invisible in the cost model's terms. SHA-256 and AES are deliberately standard rather than arithmetization-friendly choices: at token scale their circuit cost is negligible, and standard primitives are what independent recording hardware produces.
