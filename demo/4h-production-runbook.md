# 400B / S=1000 production runbook

This runbook is deliberately fail-closed.  It describes the executed path in
this tree, not a future GKR/PCS implementation and not a random-weight proxy.

> **Implementation status (2026-08-06).** S0--S4e are implemented and gated.
> S4f (quadratic accumulator, compact proof wire, transactional ledger/output)
> is implemented but awaits the new GPU/Rust campaign. S5 still lacks real
> GGUF load and five-sweep measurements. See
> `analysis/routed-projected-status.md`, the only implementation authority.
> Do not start the 400B proof until S4f and S5 are marked DONE there.

## What changed

The 24 Maverick MoE layers no longer build 128 gate, 128 up and 128 down output
tensors.  `RoutedProjectedMatmulClaim` streams one real GGUF expert shard at a
time, computes only tokens routed to that expert, and commits one `T x J` raw
output.  After `R1`, the verifier challenge `rho` defines

```
P[e,k] = sum_j W[e,k,j] rho[j]
Q[t,k] = sum_e M[t,e] P[e,k]
H[t,k] = X[t,k] Q[t,k]
yr[t]  = sum_j Y[t,j] rho[j].
```

Existing Ligero qlin/qquad proves `P=W rho`, `H=X*Q`, `yr=Y rho` and
`sum_k H=yr`.  A challenge issued after `R2` proves `Q=M P` by an ordinary
Freivalds contraction committed in `R3`.  The exact old signed-floor/range
rescale is now a standalone `RescaleClaim` and follows every routed raw output.

The projected `P` tensors are challenge-keyed and retained for the proof
lifetime.  Their total is 56,623,104 fields (~453 MB).  This is essential: it
turns four identical 400B projections into one.  A changed `rho` invalidates
the cache.

## Required transcript and policy

The offline artifact uses sequential Fiat--Shamir:

```
R1 -> s_op -> R2 -> s_bind -> R3 -> s_comb -> test polynomials -> s_col -> openings
```

`verify_proof` recomputes every seed.  It additionally requires the trusted
enrolled weight root and trusted static statement digest as command-line policy
inputs.  Missing either makes a persistent-model proof reject.  Online p1/p2/p3
padding is fresh secret entropy.  Weight padding uses the fresh secret enrollment
seed stored in the private commitment handle.

## One-time enrollment

```
LIGERO_T_QUERIES=54 python demo/demo_maverick_full.py \
  --from-gguf MODEL.gguf --tokens tokens.json \
  --layers 48 --experts 128 --d 5120 --d-ff 8192 --vocab 202048 \
  --enroll-weights maverick.wcommit
```

Record the printed `root=` in verifier policy.  Enrollment is not an online
proof stage; the proof command refuses to silently rebuild it.

## Proof

`PUBLIC_SZ` must come from the already-fixed serving statement.  The driver no
longer performs a hidden pre-proof reveal pass.

```
LIGERO_T_QUERIES=54 python demo/demo_maverick_full.py \
  --from-gguf MODEL.gguf --tokens tokens.json \
  --layers 48 --experts 128 --d 5120 --d-ff 8192 --vocab 202048 \
  --weight-commitment maverick.wcommit --expected-weight-root EXPECTED_R_W_HEX \
  --admission-report admission.json --public-sz PUBLIC_SZ \
  --dump-proof maverick-proof.json
```

Record the printed trusted statement digest.  Verify with:

```
cargo run --release --bin verify_proof -- \
  maverick-proof.json EXPECTED_R_W_HEX EXPECTED_STATEMENT_DIGEST_HEX
```

## Admission gates

The exact ledger is computed by `analysis/routed_projected_4h_model.py`:

```
W =  95,205,646,976
L =  31,064,630,194
Q =  42,394,577,408
```

The current global protocol performs five active-only witness regenerations,
not one.  Their exact shape count is `5 * 19.68898048T = 98.4449024T`
real-model MACs (including embedding and QK/AV attention).
All five, including GGUF decode/page movement, must complete in <=3609s
(>=27.278G decoded MAC/s aggregate).  The remaining hard caps are the fields in
`SLO`; the complete executed compact-wire model is:

```
model cold/load                   400.000 s
five active witness sweeps      3609.000 s
fresh commit/fold                950.000 s
linear                            25.600 s
quadratic                        765.000 s
fresh hash/coefficient           140.000 s
persistent-weight qlin          3624.522 s
persistent opening              1812.261 s
fresh opening                    450.000 s
compact u64le/base64 proof       481.481 s
RTT/tail/orchestration           700.000 s
TOTAL                          12,957.864 s = 3.5994 h
margin                         1,442.136 s
```

There is no global kappa.  Before the full run, benchmark the exact production
CUDA loop bodies on real GGUF shards and create `admission.json`; the driver
rejects reports that do not satisfy the distribution-free simultaneous bound
and sample count in `prover/admission.py`,
do not match the model root/statement, or exceed any stage cap. Do not
substitute random weights, Python primitive timings,
isolated modular multiplication, or a smaller ELL/N geometry.
The report contains the raw seconds list for every stage; the gate recomputes
that the declared bound is at least its observed maximum. It is also bound to
the GPU/driver/power identity and to the filesystem used by `--dump-proof`.
Benchmarking egress to `/tmp` does not authorize a proof written elsewhere.

## Forbidden regressions

The following each invalidates either soundness or the time theorem:

- reintroducing lists of 128 `tape.matmul` expert outputs;
- recomputing `P=W rho` after its challenge-keyed cache exists;
- accepting `Sz`, model roots, claims, seeds or columns from the proof as policy;
- deriving `s_bind` before `R2`, or `s_col` before all test polynomials;
- using the public deterministic ZK seed in production enrollment/proofs;
- omitting `RescaleClaim` after a routed raw accumulator;
- rebuilding the persistent model commitment online;
- adding a sixth witness/reveal pass;
- materializing proof JSON as Python integers instead of using the compact
  streaming writer;
- publishing a proof before atomically saving its opening-ledger update.
