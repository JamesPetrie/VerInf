# Automated Uniqueness Verification for Untrusted Claim Types

**Status:** strategy document — nothing implemented yet.
**Goal:** an automated admission gate for **untrusted claim lists**: a check that a submitted claim type pins its outputs uniquely and preserves causality (§6), so that new claims — including contestant submissions in the prediction competition (§7) — need no per-claim hand proof or human review.
**Setting:** a competition to improve the efficiency and explained information of a ZKP predicting an LLM's output tokens. Contestants submit their own claim lists; the claim list *is* the predictor. There is therefore no reference architecture to check submissions against, and no "is this the right model" obligation — the integrity properties (uniqueness, causality, one-sided scoring) are the entire spec.
**Scoring path (B.7):** the one-sided surprisal claims are *not* verified by the uniqueness pipeline (their property is "every free direction inflates the reported bound," an inequality, not uniqueness). They are **harness-owned**: contestants do not submit scoring claims. Their one-sidedness check (encode "a satisfying witness reports less than the honest value," prove UNSAT) uses the same solver back-end and must land before submissions open, since the scoring arithmetic is precisely where a rational contestant attacks.

**Trust boundary (deliberate):** Ligero soundness, commitment binding, and the Rust verifier implementation are *assumed* for now, as named hypotheses of every statement below. The formal effort concentrates on the claim layer, because that is the only layer untrusted content flows through. The competition itself doubles as a red team for the trusted layers: an exploited vulnerability is a welcome submission. One caveat to keep in view: soundness is exactly the property that makes "strong predictor" and "silent exploit" indistinguishable from an accepted proof alone, so this dynamic works only if disclosed breaks are rewarded at least as well as high scores, and submissions are archived (claim list, commitments, transcript) so a suspect score can be audited after the fact.

This document records the strategy discussed on 2026-07-21: what the property is, why naive approaches fail, the symbolic-challenge reduction that makes automation possible, how it stays sound when dimensions are parameters, how the same machinery checks causality (later tokens must not influence earlier logits — §6), the competition anti-cheat surface (§7), and the concrete components to build.

The protocol is interactive: challenges are fresh verifier randomness drawn after the relevant commitments, not Fiat–Shamir-derived from the transcript.

## 1. The property, split three ways

"No degrees of freedom for the prover" (§2.2 of the paper) is not one property. Appendix B's claims divide into three classes, and the right verification tool differs per class:

1. **Deterministic gadgets** — decompositions/rescale (B.1), the softmax bracket (B.3), RMSNorm carry chains (B.4), SiLU (B.5), routing (B.6). Property: given the inputs and the ideal lookup relations, the constraints admit exactly one witness, up to declared value-neutral freedom (masked `z`, `inv` at zero). Machine-checkable as a uniqueness (UNSAT) query.
2. **Probabilistic pins** — the Freivalds/matmul pins (B.2) and anything challenge-weighted. These are *not* deterministic for a fixed challenge: for any fixed (ρ, λ), many false `C_full` satisfy the pin. The true statement is Schwartz–Zippel-shaped: a false value committed *before* the challenge survives with probability at most `d/|F|`. Handled by the symbolic reduction of §3.
3. **One-sided freedom** — B.7 surprisal. Held constant as part of the fixed scoring harness (§7): contestants cannot modify it, and it is verified once, with an inequality query rather than a uniqueness query, before submissions open.

The LogUp layer itself (challenges, folded keys, inverses, settlements) is also probabilistic. It is *not* fed to the automated checker; it is axiomatized as an ideal relation (§4) whose soundness is a hand-proven lemma.

This is the "under-constrained circuit detection" problem from the ZK literature. Known tools and results we lean on conceptually: Picus (Veridise; automated determinism checking for R1CS-like systems, PLDI 2023), cvc5's finite-field theory (Ozdemir et al., CAV 2023), and parametric/certified approaches in proof assistants (e.g. Coda for circom in Coq; the Verified zkEVM project's Lean gadget framework). Our constraint language (linear + quadratic over Goldilocks + LogUp lookups) is unusually close to what these tools consume.

## 2. Why the naive check fails — and must

A determinism query on the *compiled* constraint system checks the wrong thing, for two reasons:

- In the implementation, challenges are concrete `u64`s baked into constraint coefficients at compile time (`op_vec(s_op, ci, "rho", ...)` in `verifier/src/handlers.rs`, table alphas from the round seed via `verifier/src/protocol.rs`). A solver looking at the compiled system sees one fixed challenge draw.
- For a fixed challenge the Freivalds pin genuinely does not pin `C_full`. Example (2×2, ρ = (7, 3)): the cheating matrix

  ```
  C' = C + Δ,   Δ = [ 3  -7 ]      (Δ·ρ = 0, so λᵀΔρ = 0 for every λ)
                    [ 0   0 ]
  ```

  satisfies the pin for all λ. A determinism checker will correctly report "under-constrained" — for the honest matmul claim. The soundness argument lives in commit *timing* (C was committed before ρ existed), which no fixed-challenge query can see.

So symbolic treatment of challenges is mandatory, not an optimization.

## 3. The reduction: symbolic challenges + one Schwartz–Zippel schema

Treat each challenge as a formal indeterminate, and make the checker round-aware (claims already declare which values commit with the forward pass and which after a challenge; Appendix B's turn boundaries encode the same thing). One hand-proven metatheorem then covers every challenge-weighted pin, present and future:

> **Schema (Schwartz–Zippel).** If values committed before a challenge draw satisfy a constraint that is a polynomial of total degree `d` in the challenge indeterminates, then either the constraint holds *identically as a polynomial* — i.e. coefficient-wise — or it survives with probability at most `d/|F|`.

The automated pipeline for a claim:

1. **Taint challenges.** Lift challenge-derived coefficients to indeterminates.
2. **Turn check.** Every variable appearing with a challenge-dependent coefficient in a pin must be committed before those challenges are drawn. Mechanical, from the declared commit schedule. This catches the classic bug: a "random" pin over a value chosen after the randomness is visible.
3. **Eliminate post-challenge defined variables.** Arrow-defined values committed after the challenge (matmul's `y`, `u`, `p`) are unique given the challenges by construction; substitute them symbolically. Licensed by a small trusted metatheorem.
4. **Expand the polynomial identity.** Require the pin to hold identically; equate coefficients per challenge monomial. For the Freivalds pin the coefficient of `λ[a]·ρ[b]` is exactly `C_full[a,b] − (AB)[a,b]`, so the pin collapses to the honest relation `C_full = AB`, cell by cell — mechanically reproducing Lemma B.2.
5. **Run the uniqueness query** on the now challenge-free system (cvc5-ff or Picus), on *outputs only*, so declared value-neutral internal freedom passes.
6. **Emit the error budget:** sum of `degree/|F|` over eliminated pins plus per-table LogUp terms from the registry — the claim's contribution to the soundness error sum, computed rather than transcribed.

### Worked example: the payoff on a broken claim

Suppose a claim author "optimizes" the 2×2 pin by reusing one challenge for both columns, ρ = (r, r). The symbolic pass doesn't pattern-match; it expands. The monomials are now `λ0·r` and `λ1·r`, and collecting coefficients yields only two constraints:

```
coeff(λ0·r):  (C[0,0] − (AB)[0,0]) + (C[0,1] − (AB)[0,1]) = 0
coeff(λ1·r):  (C[1,0] − (AB)[1,0]) + (C[1,1] − (AB)[1,1]) = 0
```

Only the *row sums* of C are pinned. The uniqueness query returns SAT with the counterexample

```
C' = C + [ 1  -1 ]
         [ 0   0 ]
```

— one residual degree of freedom per row, found automatically, with a concrete witness pair to show the author. Note that the broken claim behaves identically to the correct one under fixed-challenge checking and under any amount of honest-prover testing; only expand-then-check-uniqueness distinguishes them. The turn check catches the other classic bug the same way: commit C after the challenge turn and step 2 rejects outright (a post-challenge prover can always solve `λᵀΔρ = 0` for Δ).

## 4. Lookups: the trusted interface

LogUp's final step — "the settlement identity holds as a rational function of α, therefore the query multiset matches the table" — rests on uniqueness of partial-fraction decompositions with a multiplicity bound below the field characteristic. That is genuine mathematics, parametric in the table, and not recoverable by coefficient expansion. It is proven once, by hand (later mechanized in Lean if desired), and exposed through the line kinds that already exist:

- The linter replaces each `range` / `lookup` / `rescale` line with its **ideal relation**: `key ∈ [0, 2^w)`, `(key, value)` is a row of table T. No LogUp inverses, folded keys, or settlements ever reach the solver.
- Each use charges the registry's `(M + T_len + 1)/|F|` term to the error budget.
- The per-query inverse `z = 1/(α − key)` is post-challenge and deterministic given α (the `α = key` case has probability `T_len/|F|`, inside the error term), so it folds into the schema via step 3.

Untrusted claim authors compose `decl` / arrows / pins / `range` / `lookup` / `rescale` freely. Everything expressible either reduces to challenge-free uniqueness or is rejected by the linter and escalated to a human — the correct failure mode for an interface between trusted machinery and untrusted claim code.

## 5. Parametric dimensions

The pipeline must stay sound when sizes (S, d, m, n, k, ...) are parameters. One naive answer — "check at S = 2, 3, 4 and trust uniformity" — is **unsound for this system specifically**: the width conditions depend on the parameters and fail only at large sizes (the softmax bracket is honest-fit only while `S·s_y ≲ 2^24`; RMSNorm's windows derive from (d, S, ε) and its chunk structure changes discretely with d). The fix is the split Lemma B.1a already makes in prose:

1. **Structural uniqueness** — *given* every value confined to its stated window, the constraint pattern pins the witness. Size-uniform; checkable on small instances.
2. **Width conditions** — the windows actually tile below P at the deployed parameters. Closed-form integer inequalities in the size parameters ("max recomposable value below P", "S·s_y below 2^24"), which the linter *emits as explicit verification conditions* and evaluates at the actual (or maximal supported) parameters. Trivial arithmetic, fully automatic.

"Verified at small sizes + width VCs discharged at deployment parameters" is a sound composite claim; small-size checking alone is not.

What parameterizes cleanly vs. not:

| Obligation | Parametric? | Mechanism |
|---|---|---|
| Turn discipline, challenge freshness, pin collapse | yes | linter on the symbolic listing form (extents and summations as data, never unrolled) |
| Width conditions | yes | closed-form VCs, evaluated at max supported parameters |
| Cell-local uniqueness | yes, via separability | one-cell SMT query + lift metatheorem |
| Sum-coupled cores (softmax bracket, carry chains) | needs induction | small-S SMT as interim evidence; 2–3 one-time Lean lemmas to close |

**Separability:** if the constraint family's variable-dependency graph is a disjoint union over an extent index (cell i's constraints touch only cell i's variables plus pinned inputs), uniqueness of one cell implies uniqueness of the family at every size. Separability itself is a syntactic check on the listing. Most lines pass it (rescale, SiLU, elementwise, routing). The residue — constraints that sum or chain across an index — is a short list: the softmax bracket (monotonicity of the row sum in the shift, an induction over row length) and the RMSNorm carry chains (a fold over limbs). Those are the natural Lean targets; a *new* claim that is cell-local and uses fresh-challenge pins — which is most of them — gets a fully parametric guarantee with zero new hand proof. The linter detects a novel coupling pattern and flags it rather than guessing.

Note: the field cannot be shrunk to ease solving — every exclusion argument leans on the ~2^64 headroom of Goldilocks, so the formal model must keep the real P.

## 6. Causality: prefix-determinism

The second §2.2-adjacent obligation — later tokens must not influence earlier positions' logits — is a *generalization of the uniqueness query*, not a new kind of analysis. The paper sketches the argument shape in Lemma B.3 ("causality holds globally by induction: attention carries the filter, everything else is position-local"); this section makes it a machine check.

**Formal statement.** Uniqueness says: two satisfying witnesses that agree on all inputs agree on the outputs. Causality is the same statement with a weaker agreement precondition:

> Two satisfying witnesses that agree on inputs at positions ≤ q (but may differ arbitrarily at positions above q) agree on all outputs at positions ≤ q.

Same two-witness UNSAT encoding, same solver back-end — the only change is which variables the "agree" clause covers. A SAT result is a concrete witness pair demonstrating a future token influencing an earlier logit: the right rejection artifact.

**Why a semantic query, not taint tracking.** The tempting cheap approach is syntactic information-flow labeling: give every variable a position, check definitions only flow forward. Softmax's masked cells show why that fails. The masked `z[h,q,i]` for `i` above `q` is a free declaration, and a malicious prover *can* set those cells to functions of future tokens — nothing stops them. Taint analysis flags a violation; there is none, because the doubled table forces `e1 = e2 = 0` on masked cells regardless of the key, so the freedom never *influences* anything (the value-neutrality of Lemma B.3). The property is not "no dependence exists in the witness" but "no dependence propagates to outputs," which is inherently semantic. The prefix-determinism query handles it natively: the two witnesses may differ wildly on masked cells, and the check passes exactly when those differences cannot reach the row-q outputs. The same query correctly *fails* on a tolerance-band softmax, where slack in the normalization can be steered.

Because the protocol is interactive, challenge values are fresh verifier randomness independent of the witness, so masked-cell freedom offers no side channel through challenges; and the logits are committed before any challenge is drawn regardless, per the turn discipline the linter already enforces.

**Reuse from the pipeline.** Almost everything:

- *Separability along the position index is position-locality.* Component 2's separability analysis already computes which extent indices a claim's constraint graph is disjoint over. A claim separable in the sequence-position index (RMSNorm, SiLU, elementwise, hidden-dimension matmuls, routing, embedding select) is prefix-deterministic for free, as a corollary of its ordinary uniqueness. No new solving.
- *Only position-coupled claims need the prefix query* — in the current claim set, the attention-shaped ones (scores matmul, softmax, AV matmul), which mix positions and are supposed to mix them only backwards. They run the prefix query after challenge elimination, so the Freivalds pins are already collapsed to the honest relations.
- *One new trusted lemma: DAG composition.* If every claim is prefix-preserving and the claim graph is acyclic with inputs wired from upstream outputs or committed weights, the logits at position q depend only on tokens ≤ q. A three-line induction over the graph — the smallest lemma in the trusted base.

**Format addition.** Each tensor needs an annotation for which extent index is the sequence position (softmax's `q` and `i` both are; RMSNorm's row index is; hidden-dimension indices are not). Without it the linter cannot know which axis causality is about; it is also the natural place to declare intended exceptions such as a sliding-window predicate (which preserves causality — the upper edge is unchanged).

**Failure mode.** A claim that smuggles in position coupling — a "sequence-level normalization" dividing each position by a whole-sequence sum, a cross-position statistic feeding back into a layer — fails separability along the position index, carries no recognized causal filter, and is routed to the prefix query, which returns SAT with a witness pair showing position 0's output moving when a later position's input changes. Rejected with evidence; the check is on what the constraints entail, not what the claim is named.

**Parametric status.** Position-local claims: fully parametric via the separability lift, unchanged. Attention claims: the prefix query at small S (2, 3, 4 — enough for masked cells, a nonempty prefix, and a nontrivial suffix) is strong evidence; the ∀S statement is one induction over row length, essentially the same induction as the bracket lemma, so the Lean cost is shared. No new probabilistic content: causality is exact conditional on the pins holding, whose error the uniqueness budget already counts — the causality check adds no `1/|F|` terms.

## 7. The competition setting and anti-cheat surface

### Ownership split

- **Fixed harness (organizer-owned, verified once):** the surprisal/scoring claims (B.7) and their tables, token binding (Appendix E), and the logits-to-score interface — including the position pairing: logits at position q are scored against token q+1. Contestants cannot modify any of it.
- **Contestant-owned (untrusted, linter-gated):** the forward-pass claim list — everything from token embeddings to the committed logits. This is the only layer untrusted content flows through, and the linter is its admission gate.
- **Prover implementation: arbitrary.** Soundness holds against any prover by construction — nothing in any statement here depends on prover code — so contestants may rewrite the prover freely for efficiency, with no verification obligation attached.

In this setting **causality is co-equal with uniqueness**, not a secondary property. The competition thesis — that a low unexplained-information score requires something functionally equivalent to an LLM — holds only if prediction is genuinely from the strict past. Uniqueness without causality is worthless (a unique function that peeks still cheats); causality without uniqueness likewise (witness freedom deflates the score directly).

### Anti-cheat enumeration

1. **Uniqueness upstream of logits** — linter (§§2–5).
2. **Causality** — linter (§6).
3. **Scoring pairing** — fixed harness. The q-versus-(q+1) pairing lives in the harness, not in submissions; the linter checks only that a contestant graph's declared logit outputs plug into the harness interface with the declared position indexing. (If the pairing were submission-controlled, a claim list could score logits at q against token q — causal, and trivially perfect.)
4. **Input provenance** — linter. Source nodes of a contestant claim graph may only be: pre-committed parameters, prefix token embeddings, and public constants. No auxiliary committed input streams — an "advice" input is a channel for smuggled answers.
5. **Parameters committed before the evaluation sample is revealed** — protocol/harness; an outer-level turn check, or the tokens get encoded into the "weights." Distinguishing memorization of public data from generalization is then competition design (fresh, wide samples), not a proof-system property.
6. **Token binding** — fixed harness (Appendix E); the committed stream must be the real transcript, or contestants predict tokens of their own choosing.
7. **One-sidedness of the scoring path** — fixed harness, verified once before launch (§1, class 3).

### The theorem the gate certifies

> For **every** contestant claim list that passes the linter: an accepting transcript implies, except with probability ε (the computed error budget), that the committed logits are the unique causal function of the committed prefix determined by that claim list, and that the reported unexplained information is a true upper bound for the committed token stream. Hypotheses: Ligero soundness, commitment binding, verifier-implementation correctness, and uniform verifier randomness (the trust boundary above).

The universal quantifier over submissions is the point: it is what permits admitting claim lists nobody has reviewed. Published scores should carry their ε.

## 8. Architecture: six components

**End state.** A claim author submits (a) a machine-readable listing and (b) an implementation. `claim-lint` outputs **PASS** with an error budget and checked width VCs, or **FAIL** with a named rule violation or a concrete two-witness counterexample.

1. **Machine-readable claim format.** A schema capturing what an Appendix B listing already contains: line kinds (`input`/`decl`/`chal`/`lin`/`quad`/`==`/`range`/`lookup`/`rescale`), expressions with symbolic index variables, extents with filters, turn boundaries, declared outputs, and a per-tensor annotation of which extent index is the sequence position (§6). Plus the table registry (B.1.0) as data: generating rule, width, length per table. This is the load-bearing artifact — the grammar exists; it currently lives in LaTeX instead of a parseable format.
2. **Structural checks** (syntactic, all sizes): well-formedness, turn discipline, challenge freshness (one independent challenge per extent element — flags the ρ-reuse bug before any solving), separability classification (cell-local vs. coupled), input-provenance typing (source nodes limited to pre-committed parameters, prefix token embeddings, and public constants — §7 item 4), and the harness-interface check (declared logit outputs wired to the fixed scoring harness with the declared position indexing — §7 item 3).
3. **Challenge elimination:** substitute post-challenge arrow variables, expand pins in challenge monomials with extent indices kept symbolic, output a challenge-free system plus a degree per pin. Rejects pins that are not polynomial in the challenges or that weigh a late-committed `decl`. Needs a small computer-algebra layer (polynomial expansion, summation reindexing).
4. **Lookup idealization + width VC emission:** replace lookup lines with ideal relations, charge LogUp error terms, emit every decomposition/bracket/exclusion width inequality and evaluate it at deployment parameters.
5. **Uniqueness solver:** for each cell-local component, instantiate one cell at real bit-widths, run the two-witness UNSAT query (cvc5-ff / Picus) on declared outputs; SAT prints the counterexample pair. The query takes an **agreement-set parameter**: all inputs for the uniqueness check, prefix inputs for the causality check (§6). Coupled components run at small S (evidence, honestly labeled) pending their Lean lemmas.
6. **Conformance check:** instantiate the listing at a small shape and seed; run `compile_claims` (`verifier/src/handlers.rs`) at the same shape and seed; diff the constraint systems entry-by-entry (the difftest infrastructure is most of this). Ties "the listing is sound" to "the code emits the listing," so an author cannot pass the linter with one system and ship another.

## 9. Trusted base

Proven by hand once, ever (candidates for later Lean mechanization):

1. The Schwartz–Zippel schema (licenses coefficient expansion).
2. The substitution metatheorem for post-challenge arrow-defined variables.
3. The LogUp multiset lemma (licenses lookup idealization).
4. The separability lift (licenses the one-cell reduction).
5. The DAG composition lemma (prefix-preserving claims over an acyclic graph compose to global causality; §6).

Plus, over time, the 2–3 inductive lemmas for the coupled cores (softmax bracket, RMSNorm carries), and — unavoidably — the linter's own implementation. By the trust decision of §7, Ligero soundness, commitment binding, and the Rust verifier implementation are **named hypotheses**, not proof obligations; protocol-level facts (interactive round composition) are handled once in the paper's soundness section, not per claim. Everything a claim author writes is untrusted and machine-checked.

## 10. Milestones

1. **M1 — skeleton:** format schema + matmul listing transcribed + components 2, 3, 5. Acceptance: matmul PASSes with budget `2/|F|`; the challenge-reuse variant FAILs with the row-sum counterexample. Also tells us early whether cvc5-ff handles the cell queries comfortably or Picus-style propagation is needed.
2. **M2 — lookups:** registry as data + component 4; matmul *with rescale* passes end-to-end; width VCs appear.
3. **M3 — easy sweep:** SiLU, elementwise, routing (mostly cell-local; should pass or expose transcription drift between paper and code).
4. **M4 — coupled path:** softmax and RMSNorm at small S; honest-caveat reporting; enumerate the residual induction lemmas. This is also where causality lands: the prefix-determinism query on the attention-shaped claims, the position-axis annotations, and the claim-graph composition check (§6).
5. **M5 — conformance:** diff instantiated listings against `compile_claims` output.
6. **M6 — scoring harness (pre-launch gate):** one-sidedness of the fixed B.7 claims via the inequality query, plus the outer-level checks (parameters-before-sample ordering, token binding wired in). Required before contestant submissions open; the competition cannot soundly launch on M1–M5 alone.

### Extraction note for M5 and beyond

`compile_op` bakes challenges in as `u64`s, so recovering polynomial structure from the compiler has three options, cheapest first: (a) run `compile_claims` under several seeds and interpolate — challenge polynomials are low degree (the `λ[a]ρ[b]` pins are degree 2), so a handful of evaluations recovers coefficients exactly, and coefficients constant across seeds are challenge-free; (b) make `Build`'s coefficient type generic over `u64` vs. a symbolic tag (cleaner, modest refactor); (c) do extraction on the Python prover side and difftest against the Rust. Start with (a) to validate the pipeline, then decide whether (b) is worth it.

## 11. Relation to the paper

- §2.2 states the two-sided requirement this pipeline discharges the first half of; the paper's future-work section names formal verification of exactly these properties — both the unique-witness property and the causal-mask property, the latter now covered by §6's prefix-determinism check.
- The pipeline mechanically reproduces the structure of the Appendix B lemmas: challenge elimination re-derives Lemma B.2's coefficient argument; the width-VC split mirrors Lemma B.1a; lookup idealization is the factoring rule ("soundness-inert expansions") made executable.
- The claim-graph obligations remain: per-claim uniqueness composes only over an acyclic wiring where each claim's inputs are upstream outputs or committed weights, and cross-claim width dependencies (e.g. B.6's gap exclusion relying on B.2's rescale bound) should become explicit magnitude contracts — each claim's verified statement carries "outputs in [−2^25, 2^25)" as a postcondition consumed as a precondition downstream — so the §2.2 audit becomes contract-checking.

---

## 12. Architecture options and how the design evolved

Sections 1–11 describe the **first** design considered: a bespoke checker (the "linter") that runs symbolic-challenge elimination and an SMT uniqueness query. That reasoning is all still valid and reused, but subsequent analysis moved the recommended endpoint. This section records the option space and the current recommendation; §§1–11 remain the detailed treatment of the machinery the later options also depend on.

### 12.1 Two structural simplifications that survive every option

**Freivalds and LogUp are one pattern.** Both are *random-point equality testing of committed algebraic fingerprints*: encode a big object as a (multi)linear or rational function, commit it before a random point is drawn, check the fingerprint identity at that point, and conclude by Schwartz–Zippel plus a per-encoding **injectivity lemma** (bilinear-form-vanishes ⟹ matrix zero, for Freivalds; equal rational fingerprints ⟹ equal multisets, for LogUp). Consequence for the checker: the matmul pin, the RMSNorm broadcast/projection pin (B.4's ρ-projection, degree 1, one constraint per free row), the MoE combine pin, and per-head batching are all instances of **one** `contract_pin` composite — an equality of multilinear expressions in committed tensors with a declared set of challenge-contracted indices, error `(#contracted challenge vectors)/|F|` per free-index constraint. So the challenge surface is exactly two composites: `contract_pin` and the lookup family.

**Challenge-by-construction.** The Rust already draws challenges internally (`op_vec` in `handlers.rs`); no one writes raw challenge constraints today. Make that explicit: the contestant-visible grammar has **no `chal` line and no challenge symbols at all**. Composites (`contract_pin`, `lookup`/`range`/`rescale`) are compiler-owned expansions that draw fresh challenges, keyed by (claim, line, label), at a turn placed after their operands' commitment. Then turn-order and freshness hold *by construction* rather than being checked — the ρ-reuse bug and the commit-after-challenge bug become **inexpressible** rather than detectable, and the entire symbolic-elimination pass (§3, component 3) drops out of the per-claim path.

Under both simplifications the checker's job reduces to: consume each composite's *conclusion* as an ideal fact (`contract_pin` → "the tensor identity holds exactly, + degree/|F|"; `lookup` → "(key, value) ∈ T, + registry term") and then verify the *deterministic* remainder pins every output.

### 12.2 The options for the deterministic remainder

The remaining question is how the deterministic part (decompositions, carry chains, brackets, gap exclusions) is verified. Four options, in the order they were considered:

- **(A) SMT uniqueness query** (§§1–11). A two-witness UNSAT query per cell-local component, small-S for coupled ones. *Risk:* finite-field/range encoding feasibility of cvc5-ff is unproven; solver timeouts make gate verdicts non-deterministic; needs a computer-algebra layer for the symbolic pass. Good as a **development aid** (counterexample finding); shaky as a competition admission gate.
- **(B) Composite library with per-pattern lemmas.** ~7 composites (contraction pin, lookup/range/rescale, word split, exact product, monotone bracket, gap exclusion), each a compiler expansion with one hand-proven soundness lemma; claims are data composing them; a deterministic checker consumes their conclusions. *Risk:* contestants want composites we didn't ship — a fixed ceiling, organizer-in-the-loop for anything new.
- **(C) Generic rule kernel.** Replace per-pattern lemmas with ~6 generic inference rules (interval propagation, wrap-lifting, tiling determination, lifted linear consequence, a monotonicity calculus, reachability); each `decl` carries a checkable `pinned_by:` derivation hint that the sugar generates automatically. Absorbs most unforeseen *patterns* (they re-derive from the rules); only a new *reasoning principle* needs a new rule. *Risk:* the kernel (interval arithmetic + exact linear algebra + monotonicity calculus) is subtle to implement and maintain, and its correctness joins the trusted base.
- **(D) Proof-carrying claims (recommended).** The contestant submits the claim listing **plus a machine-checkable Lean proof** of its uniqueness/causality goals; the gate compiles the proof and checks (via `#print axioms`) that only two sanctioned axioms — the `contract_pin`/Freivalds fact and the LogUp fact — appear, with no `sorry`. Generality is maximal: anything provable is admissible, so a novel gadget's inventor proves it themselves with **no organizer in the loop**. The bespoke reasoning engine of (A)/(C) disappears — the checking engine is the Lean kernel, and Mathlib's `omega`/`nlinarith`/`decide` do what the rule kernel would have. Composite libraries (B) and rule lemmas (C) survive as a **convenience library** of Lean lemmas/tactics so tier-A claims discharge in a line or two, with the sugar functions emitting the proof so those contestants never open Lean.

### 12.3 The recommended layering

The options are not exclusive; the recommended system layers them by contestant population:

1. **DSL fast path** — architecture-search contestants write a forward pass in a small functional DSL; a compiler emits listing + auto-generated proof; zero proof burden. Most submissions live here.
2. **Schema path** — contestants writing raw listings from existing mechanisms get auto-generated certificates/proofs from the convenience library.
3. **Kernel path** — gadget inventors submit a novel primitive *with its Lean proof*; once checked, it joins the DSL primitive set and the library, so the trusted vocabulary is **extensible by proof-carrying submission**, benefiting every later contestant.

This resolves the "N composites is too many, and contestants will want more" problem structurally: the gate has no fixed ceiling and no human-in-the-loop even for novel techniques, while the common case stays turnkey. Trust decision unchanged from §7: Ligero, commitment binding, and the verifier implementation are assumed; the two composite facts are named axioms justified by their papers and Appendix B.

### 12.4 Trusted base under proof-carrying

Kernel (standard) + the two sanctioned axioms (`contract_pin`/Freivalds, LogUp) + the **translator** (listing JSON → Lean goal statements — the one load-bearing new piece, kept small, golden-tested, and difftested against `compile_claims`) + the axiom-guard script + an organizer lemma library (the lift family, tiling/B.1a, the monotone-threshold schema). Smaller than the rule-kernel base and strictly more general. The generic verifier compiler (the untrusted-listing interpreter) is a separate concern; see §14.

### 12.5 Table discharge as a pluggable backend

Claim proofs should consume the tables only through a **property interface** (monotone, zero-tail, pair-shift relation, bounds) — never mentioning `exp`. Two backends discharge that interface: **data** (`native_decide`/scan over the concrete array — cheap, robust, non-parametric; the gate's default) and **analytic** (prove the properties from the generating equation — more work, parametric in the scales, and a prerequisite only for the harness-side B.7 "true bound in nats" statement, which genuinely compares table entries to real exponentials). Building data-discharge first unblocks every claim proof; the certified-evaluation layer waits for B.7. A concrete side-benefit of the analytic route: it audits the deployed float64-generated tables for near-tie rounding disagreements the Python generator has never been checked for.

## 13. The generic verifier compiler (untrusted-listing interpreter)

Independent of the gate: the *verifier* must compile an arbitrary contestant listing to constraints, since it cannot run contestant code. Today `handlers.rs` compiles known claim types as monolithic handlers over a fixed `Expander` vocabulary (`Weighted`, `Rowsum`, `CausalId`, `FreivaldsB/C`, `Embed`, `RopeX`, …). Findings:

- **Arbitrary listings do not map onto the current expanders as-is** — the vocabulary is ten hand-compiled special cases.
- **But all ten are specializations of one pattern:** decompose the flat slot into a mixed-radix multi-index; the constraint id is an affine function of the components (a dropped axis = a reduction), possibly filtered by a fixed predicate; the coefficient is a product of per-axis factors (constant, public vector, public gather, or — inside `pin` only — a challenge vector). A single `AffineMap` family subsumes the lot. The **listing's index expressions are the interface**: the compiler lowers each line to `AffineMap` parameters; contestants never name an expander. The admission rule is then a grammar rule — index arithmetic affine in the extent indices, filters from the fixed set, coefficients of the allowed kinds — enforced at submission, and it doubles as the discipline the causality/separability analyses want.
- **Deliberately out of scope** (rejected at submission, correctly): witness-dependent indexing (that is what lookups are for), ragged/dynamic extents (use masks, as routing does), and author-supplied mask *semantics* (fixed predicate set only). Misaligned quads and per-slot quad coefficients are handled by a `lin`-copy into alignment — a witness cost, not an expressiveness limit.
- **The real risk is performance, not correctness** (see §15): the generic `AffineMap::emit` must match the hand-tuned expanders' streaming throughput at Maverick scale, or keep them as fast paths.

## 14. Findings from the softmax-bracket Lean spike

Run on branch `lean-bracket-spike` (`lean/BracketSpike/`, see `lean/FINDINGS.md` there). Target: prove the softmax shift `c` is uniquely pinned at S = 2, **saturation included**, through the honest field→integer lift — the "hardest consumer" for the lift library, chosen to de-risk the bracket formalization before committing to option (D).

**Result — the mathematical core is proven with no `sorry`** (every theorem depends only on `propext`, `Classical.choice`, `Quot.sound`, confirmed by `#print axioms`):

- `lift_cell` — field recomposition + range bounds + width condition ⟹ the genuine **integer** identity. This is the reusable lift library primitive (Lemma B.1a), amortized across RMSNorm carries and every `range`/`rescale`.
- `cell_value_neutral` — *any* saturation witness emits output `= g(c − x)`; full case split (below Zmax forces `zhigh = 0`; at/above Zmax the table is already 0). This is the clause of Lemma B.3 whose prose is most compressed.
- `threshold_unique` — a non-increasing integer function is pinned by a two-sided adjacent bracket. The reusable monotone-pin schema, shared with RMSNorm's rsqrt.
- `shift_unique_S2` — assembled: two satisfying witnesses at S = 2 share the same `c`.

**Key reads:**

- **No mathematical surprise.** The three most-uncertain pieces (lift, saturation value-neutrality, monotone threshold) went through as the paper argues. All friction was Lean *plumbing* (`omega` needing explicit `norm_num` numeric facts to chain a variable bound into `< P`; the `ZMod.val_*` API), none of it mathematical. The largest fear — that the bracket hides an unstated side condition — did not materialize at S = 2.
- **New concrete finding for the plan:** deriving the boolean saturation flag from its `t² = t` quadratic needs `ZMod P` to be an integral domain — i.e. it **pulls in a Goldilocks-primality proof**, which the lift itself avoids. Budget a one-time `native_decide`/Pratt-certificate primality proof; the bracket-flag step forces it, the lift does not.
- **Timeline:** the two-week estimate for a full `sorry`-free S = 2 bracket looks credible, arguably conservative — the hard core took a fraction of a session. Remaining work is mechanical/one-time, not research.

**Honest limits of the spike** (assumed via the model, not yet derived): the table is abstracted by a property certificate (data-discharge not run); the δ-shift step `s2(c) = s1(c−1)` is baked into the model rather than derived from the `T_B[k] = T_A[k−δ]` pair; the boolean flag is taken as given rather than derived from its quadratics (see primality finding); the end-to-end "raw ZMod constraints ⟹ unique `c`" composition is not assembled; masked cells (`i > q`) are not modelled. **Not exercised at all:** the parametric-in-`S` induction — this spike is fixed S = 2, so the ∀S estimate remains the open one.

## 15. Launch inventory and risk

**Ownership recap:** the B.7 scoring harness and token binding are organizer-owned and fixed; contestants modify only the forward-pass claim list and may rewrite the prover arbitrarily (soundness never depends on prover code). Note a likely simplification: because the organizer publishes the evaluation tokens, Appendix E transcript binding is probably out of scope — the anti-cheat need reduces to the commit-before-reveal ceremony.

**Workstreams to a launchable gate (~3–4 months focused; de-scope levers below):**
- **A. Generic verifier compiler** (§13) — the biggest hidden item; `AffineMap` lowering + difftest against existing handlers.
- **B. The gate** — prelude, translator (JSON→Lean), lemma library, guards (pinned toolchain, axiom check, statement-integrity, sandboxed builds), structural checks.
- **C. Incumbent claims through their own gate** — the baseline must pass the gate it is judged by; dominated by the softmax and RMSNorm bracket proofs (the §14 spike is the first slice).
- **D. Scoring harness + outer protocol** — B.7 one-sidedness in Lean, the commit-reveal ceremony, ε attached to every score.
- **E. Ops** — rules doc, SDK + worked example, leaderboard, submission archiving, disclosure-reward policy, local gate distribution.

**Launch bar:** baseline passes end-to-end (listings → kernel-checked proofs → generic verifier accepts a real run → score + ε); scoring one-sidedness proven; an internal **red-team round** against your own gate; a second machine reproduces a verdict from archived artifacts.

**De-scope levers:** round 1 = SDK-only (no contestant Lean); smaller baseline model; attention claim fixed as harness-owned in round 1 (removes the hardest contestant-facing proof). With all three, a credible round-1 launch is ~6–8 weeks.

### 15.1 Riskiest parts

- **Soundness-critical (silent-failure) risk:** the generic verifier compiler (§13) upholding, for *every* adversarial listing, the by-construction side conditions the Lean layer takes as granted facts (turn order, challenge freshness, correct expansion). A miss here makes a kernel-checked proof true of a statement that is false of the running system — PASS with a fake score, below the specification boundary where more Lean cannot reach. *Mitigation:* enforce the invariants as protocol-driver runtime assertions and compile-time challenge-tuple uniqueness (not compiler intentions); adversarially fuzz the compiler; aim the red-team round here; keep the grammar minimal.
- **Timeline (harder-than-expected) risk, ranked:** (1) the bracket formalizations + the lift library API — highest variance, though §14 retired the *mathematical* surprise and the lift is now built; (2) the generic compiler's **performance** at frontier scale (correctness is ordinary; the streaming-throughput match to hand-tuned expanders is the open question — one benchmarking spike settles it); (3) **discovery risk** — formalization or the B.7 pass surfacing a genuine soundness gap that cascades through paper/Rust/Python (precedent: the pre-fix RMSNorm bracket). Sequence the discovery-prone items (softmax bracket — started; B.7 rounding directions) early so any redesign lands before the launch machinery is built on the current construction.
