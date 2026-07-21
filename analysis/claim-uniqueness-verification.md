# Automated Uniqueness Verification for Untrusted Claim Types

**Status:** strategy document — nothing implemented yet.
**Goal:** an automated check that a proposed claim type pins its outputs uniquely — and preserves causality (§6) — so that new claims (written by untrusted authors, human or agent) need no per-claim hand proof.
**Out of scope for now:** the one-sided surprisal claims (B.7). Their property is "every free direction inflates the reported bound," not uniqueness; the pipeline below can be extended to it later (encode "a satisfying witness reports less than the honest value" and prove UNSAT), but we defer it.

This document records the strategy discussed on 2026-07-21: what the property is, why naive approaches fail, the symbolic-challenge reduction that makes automation possible, how it stays sound when dimensions are parameters, how the same machinery checks causality (later tokens must not influence earlier logits — §6), and the concrete components to build.

The protocol is interactive: challenges are fresh verifier randomness drawn after the relevant commitments, not Fiat–Shamir-derived from the transcript.

## 1. The property, split three ways

"No degrees of freedom for the prover" (§2.2 of the paper) is not one property. Appendix B's claims divide into three classes, and the right verification tool differs per class:

1. **Deterministic gadgets** — decompositions/rescale (B.1), the softmax bracket (B.3), RMSNorm carry chains (B.4), SiLU (B.5), routing (B.6). Property: given the inputs and the ideal lookup relations, the constraints admit exactly one witness, up to declared value-neutral freedom (masked `z`, `inv` at zero). Machine-checkable as a uniqueness (UNSAT) query.
2. **Probabilistic pins** — the Freivalds/matmul pins (B.2) and anything challenge-weighted. These are *not* deterministic for a fixed challenge: for any fixed (ρ, λ), many false `C_full` satisfy the pin. The true statement is Schwartz–Zippel-shaped: a false value committed *before* the challenge survives with probability at most `d/|F|`. Handled by the symbolic reduction of §3.
3. **One-sided freedom** — B.7 surprisal. Deferred (see scope note above).

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

## 7. Architecture: six components

**End state.** A claim author submits (a) a machine-readable listing and (b) an implementation. `claim-lint` outputs **PASS** with an error budget and checked width VCs, or **FAIL** with a named rule violation or a concrete two-witness counterexample.

1. **Machine-readable claim format.** A schema capturing what an Appendix B listing already contains: line kinds (`input`/`decl`/`chal`/`lin`/`quad`/`==`/`range`/`lookup`/`rescale`), expressions with symbolic index variables, extents with filters, turn boundaries, declared outputs, and a per-tensor annotation of which extent index is the sequence position (§6). Plus the table registry (B.1.0) as data: generating rule, width, length per table. This is the load-bearing artifact — the grammar exists; it currently lives in LaTeX instead of a parseable format.
2. **Structural checks** (syntactic, all sizes): well-formedness, turn discipline, challenge freshness (one independent challenge per extent element — flags the ρ-reuse bug before any solving), separability classification (cell-local vs. coupled).
3. **Challenge elimination:** substitute post-challenge arrow variables, expand pins in challenge monomials with extent indices kept symbolic, output a challenge-free system plus a degree per pin. Rejects pins that are not polynomial in the challenges or that weigh a late-committed `decl`. Needs a small computer-algebra layer (polynomial expansion, summation reindexing).
4. **Lookup idealization + width VC emission:** replace lookup lines with ideal relations, charge LogUp error terms, emit every decomposition/bracket/exclusion width inequality and evaluate it at deployment parameters.
5. **Uniqueness solver:** for each cell-local component, instantiate one cell at real bit-widths, run the two-witness UNSAT query (cvc5-ff / Picus) on declared outputs; SAT prints the counterexample pair. The query takes an **agreement-set parameter**: all inputs for the uniqueness check, prefix inputs for the causality check (§6). Coupled components run at small S (evidence, honestly labeled) pending their Lean lemmas.
6. **Conformance check:** instantiate the listing at a small shape and seed; run `compile_claims` (`verifier/src/handlers.rs`) at the same shape and seed; diff the constraint systems entry-by-entry (the difftest infrastructure is most of this). Ties "the listing is sound" to "the code emits the listing," so an author cannot pass the linter with one system and ship another.

## 8. Trusted base

Proven by hand once, ever (candidates for later Lean mechanization):

1. The Schwartz–Zippel schema (licenses coefficient expansion).
2. The substitution metatheorem for post-challenge arrow-defined variables.
3. The LogUp multiset lemma (licenses lookup idealization).
4. The separability lift (licenses the one-cell reduction).
5. The DAG composition lemma (prefix-preserving claims over an acyclic graph compose to global causality; §6).

Plus, over time, the 2–3 inductive lemmas for the coupled cores (softmax bracket, RMSNorm carries), and — unavoidably — the linter's own implementation. Protocol-level facts (interactive round composition, commitment binding) are handled once in the paper's soundness section, not per claim. Everything a claim author writes is untrusted and machine-checked.

## 9. Milestones

1. **M1 — skeleton:** format schema + matmul listing transcribed + components 2, 3, 5. Acceptance: matmul PASSes with budget `2/|F|`; the challenge-reuse variant FAILs with the row-sum counterexample. Also tells us early whether cvc5-ff handles the cell queries comfortably or Picus-style propagation is needed.
2. **M2 — lookups:** registry as data + component 4; matmul *with rescale* passes end-to-end; width VCs appear.
3. **M3 — easy sweep:** SiLU, elementwise, routing (mostly cell-local; should pass or expose transcription drift between paper and code).
4. **M4 — coupled path:** softmax and RMSNorm at small S; honest-caveat reporting; enumerate the residual induction lemmas. This is also where causality lands: the prefix-determinism query on the attention-shaped claims, the position-axis annotations, and the claim-graph composition check (§6).
5. **M5 — conformance:** diff instantiated listings against `compile_claims` output.

### Extraction note for M5 and beyond

`compile_op` bakes challenges in as `u64`s, so recovering polynomial structure from the compiler has three options, cheapest first: (a) run `compile_claims` under several seeds and interpolate — challenge polynomials are low degree (the `λ[a]ρ[b]` pins are degree 2), so a handful of evaluations recovers coefficients exactly, and coefficients constant across seeds are challenge-free; (b) make `Build`'s coefficient type generic over `u64` vs. a symbolic tag (cleaner, modest refactor); (c) do extraction on the Python prover side and difftest against the Rust. Start with (a) to validate the pipeline, then decide whether (b) is worth it.

## 10. Relation to the paper

- §2.2 states the two-sided requirement this pipeline discharges the first half of; the paper's future-work section names formal verification of exactly these properties — both the unique-witness property and the causal-mask property, the latter now covered by §6's prefix-determinism check.
- The pipeline mechanically reproduces the structure of the Appendix B lemmas: challenge elimination re-derives Lemma B.2's coefficient argument; the width-VC split mirrors Lemma B.1a; lookup idealization is the factoring rule ("soundness-inert expansions") made executable.
- The claim-graph obligations remain: per-claim uniqueness composes only over an acyclic wiring where each claim's inputs are upstream outputs or committed weights, and cross-claim width dependencies (e.g. B.6's gap exclusion relying on B.2's rescale bound) should become explicit magnitude contracts — each claim's verified statement carries "outputs in [−2^25, 2^25)" as a postcondition consumed as a precondition downstream — so the §2.2 audit becomes contract-checking.
