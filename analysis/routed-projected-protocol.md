# RoutedProjected-MoE for VerInf

## Statement and non-negotiable compatibility

The public statement is the existing VerInf statement: the canonical claim-set
digest, enrolled Maverick weight roots, exact quantized arithmetic/rounding and
tables, input/transcript commitment, and final UI claim.  This construction
changes only how the expert matmuls are proved.  It does not replace the model
with a parameter-count abstraction and it does not introduce a second verifier.
Every new witness column is an ordinary row of the existing joint Ligero
commitment and is discharged by the existing IRS, `q_lin`, `q_quad`, Merkle and
Rust-verifier checks.

For one routed matmul, the relation is

$$Y_{t,j}=\sum_k X_{t,k}W_{e_t,k,j}.$$

The route $e_t$ is the unique top-1 route already proved by `RoutingClaim`,
including booleanity, cardinality, tie-breaking, dominance and range checks.

## Algebraic reduction

After $X,Y,M$ are committed, the verifier samples
$\rho\in\mathbb F^J$ and projects only the output dimension.  Associativity
turns the degree-three routed contraction into one projected weight matrix, one
ordinary matrix product $Q=MP$, and one Hadamard product.  The challenge for
checking $Q=MP$ is sampled in a later epoch, after $P,Q$ are committed.

## Private routing without sort or lookup

Reassociate the contraction after $\rho$ is sampled:

$$P_{e,k}=\sum_j W_{e,k,j}\rho_j,\qquad
Q_{t,k}=\sum_e M_{t,e}P_{e,k}.$$

Commit also

$$H_{t,k}=X_{t,k}Q_{t,k},\qquad
y^\rho_t=\sum_jY_{t,j}\rho_j.$$

Ordinary Ligero relations bind $P=W\rho$, $H=X\odot Q$,
$y^\rho=Y\rho$, and $\sum_kH_{t,k}=y^\rho_t$.  Only $Q=MP$ has two
inputs that were not both fixed in R1.  After R2, sample fresh
$\sigma\in\mathbb F^K$, $\lambda\in\mathbb F^T$, and commit the standard
Freivalds auxiliaries

$$f_y[e]=\sum_kP[e,k]\sigma_k,\quad
f_u[e]=\sum_t\lambda_tM[t,e],\quad f_p[e]=f_u[e]f_y[e].$$

Ligero checks

$$\sum_e f_p[e]=\sum_{t,k}\lambda_tQ[t,k]\sigma_k.$$

This uses only $TK+EK$ fields, leaks neither selected experts nor their
multiplicities, and needs no dynamic table, private sort, GKR or new PCS.

## Transcript

The exact five-message order is:

1. `R1`: commit model inputs/outputs/routes and challenge-independent rows;
   reference the trusted enrolled weight roots.
2. Sample `s_op`: derive $\rho$.
3. `R2`: commit $P,Q,H,y^\rho$.
4. Sample `s_bind`: derive $\sigma,\lambda$.
5. `R3`: commit $f_y,f_u,f_p$.
6. Sample `s_comb`; send the existing `q_irs`, `q_lin`, `p_0` polynomials.
7. Sample `s_col`; open exactly the verifier-derived distinct columns of the
   block order `blind | W | p1 | p2 | p3`.

Every verifier coin is sampled after the preceding prover message.  Fiat-Shamir
mode hashes the same framed transcript.  The verifier receives the trusted
statement/model roots externally; it never accepts roots, shapes, challenges or
column indices chosen by the proof.

## Security inheritance

Conditioned on binding Merkle roots and valid Ligero codewords:

* a false projected output survives $\rho$ with probability at most
  $1/|\mathbb F|$;
* a false $Q=MP$ survives the two-sided late Freivalds check with probability
  at most $2/|\mathbb F|$;
* all linear, quadratic and proximity errors are exactly the existing Ligero
  error terms because the new relations use the same joint tests;
* the top-1 route semantics remain those of `RoutingClaim`;
* zero knowledge is inherited from fresh secret Ligero padding/blinding for
  `p1/p2/p3`, the persistent-weight opening ledger and refresh rule.  No
  challenge projection or auxiliary vector is a public proof field.

Thus

$$\epsilon_{new}\le \epsilon_{Ligero}+\frac{3}{|\mathbb F|}+\epsilon_{hash}.$$

At the Maverick dimensions these field terms are far below the existing
$48(3/4)^{54}$ proximity budget.

## Exact conservative ledger at S=1000

The original full-witness ledger is

$$(W,L,Q)=(888{,}249{,}981{,}888,
162{,}237{,}276{,}010,
173{,}106{,}423{,}296).$$

For 24 MoE layers and the three expert matrices,

$$N=24\cdot1000\cdot(2\cdot5120+8192)=442{,}368{,}000,$$

$$R=24\cdot128\cdot(2\cdot5120+8192)=56{,}623{,}104.$$

A row-exact representation (including separate ELL padding for the 72 `yr`
variables and 216 late auxiliary vectors) charges

$$W_{route}=2N+R+4\cdot72\cdot8192=943{,}718{,}400,$$

$$L_{route}=R+2\cdot72\cdot1000+72(2E+1)=56{,}785{,}608,$$

$$Q_{route}=N+72E=442{,}377{,}216.$$

After deleting the 400B ordinary weight-witness block, all 128 expert output
streams and their old Freivalds auxiliaries, while retaining exact selected
outputs/rescale gadgets, the safe ledger used by the admission model is

$$(W,L,Q)=(95{,}205{,}646{,}976,
31{,}064{,}630{,}194,
42{,}394{,}577{,}408).$$

This is 10.72%, 19.15% and 24.49% of the original full-witness counts
(exact ratios printed by `analysis/routed_projected_4h_model.py`).

## Four-hour admission envelope

`analysis/routed_projected_4h_model.py` prices every executed kernel separately;
there is no global calibration multiplier.  It includes GGUF load/decode,
fresh rows, linear/quadratic work, the exact
executed flattened-variable weight geometry including row padding, the post-combiner persistent
weight `q_lin` fold, persistent and
fresh openings, the currently implemented compact u64le/base64 JSON proof drain, RTT,
refresh and orchestration.  It also
charges the current global prover's five complete witness regenerations:
$5\cdot19.68898048$T active-model MACs must fit in 3609 seconds (at least 27.278G
decoded real-GGUF MAC/s, including loader/page-migration overhead).  Random
weights cannot satisfy this admission test.

The admission limits sum to

$$T=12{,}957.864\text{ s}=3.5994\text{ h},$$

leaving $1442.136$ seconds.  A production build rejects startup unless
simultaneous p99 bounds satisfy every per-kernel limit in that file.  The
critical structural requirements are exactly five active-only regenerations
and no sixth reveal pass, no all-expert activation tensor, no ordinary
flat-weight commit/fold/open, no
resident encoded-weight matrix, and incremental proof serialization without a
Python-int materialization.
