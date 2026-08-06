# RoutedProjected-MoE

Ligero proofs of top-1 MoE inference that commit **only the selected expert's
activation**, while still binding the result to every enrolled expert weight.

For 400B Maverick at `E=128` the online witness drops to 10.7% of the
all-expert layout: a 1000-token proof runs in **2 h 02 min** and an independent
Rust verifier **ACCEPTs** it in 1 h 51 min.

## The idea

Proving `Y[t,j] = Σ_e Σ_k M[t,e] X[t,k] W[e,k,j]` naively materializes outputs
for all `E` experts, then proves the private route `M` picked one. Instead:
after `X`, `Y`, `M` are committed, the transcript samples `rho`, and the prover
commits

```
P[e,k] = Σ_j W[e,k,j] rho[j]      Q[t,k] = Σ_e M[t,e] P[e,k]
H[t,k] = X[t,k] Q[t,k]            yr[t]  = Σ_j Y[t,j] rho[j]
```

Ordinary Ligero relations bind `P = W rho`, `H = X⊙Q`, `yr = Y rho` and
`Σ_k H = yr`. Only `Q = MP` has two inputs that were not both fixed in R1, so a
*second* challenge `(sigma, lambda)` — sampled only after `P,Q` are committed —
checks it by Freivalds:

```
Σ_e f_u[e]·f_y[e] = Σ_{t,k} lambda_t · Q[t,k] · sigma_k
```

The prover computes only selected activations, but `P = W rho` reads every
enrolled weight, so the model stays bound. It is an algebraic reduction *inside*
Ligero — every new value is an ordinary witness row, discharged by the existing
IRS, `q_lin`, `q_quad`, Merkle and Rust-verifier checks. No GKR, no new
commitment scheme, no trusted setup.

Transcript, each coin hashed from everything before it:

```
R1 → s_op(rho) → R2 → s_bind(sigma,lambda) → R3 → s_comb → test polys → s_col → openings
```

## Measured, 2026-08-06

A100-SXM4-80GB, real GGUF `UD-Q4_K_XL`, 48 layers, `E=128`, `S=1000`,
`T_QUERIES=54`.

| | |
|---|---:|
| Prove | **7,352.9 s = 2 h 02 min** |
| Verify (independent Rust, 30 threads) | **1 h 51 min → `ACCEPT`** |
| Proof size | 35.46 GB, `u64le`/base64 |
| Peak GPU | 65.11 GB |
| Admission total | 6,939.8 s of the 14,400 s envelope |
| Enrollment, one-time, 49.2M weight rows | 1,401.2 s |

Ledger at `S=1000`: `W` 888.2G → 95.2G (10.72%), `L` 162.2G → 31.1G (19.15%),
`Q` 173.1G → 42.4G (24.49%).

Four of the ten verifier checks are what make it a *policy* check and not a
self-check: the statement digest recomputed from the claim bytes, that digest
against an externally supplied one, Fiat–Shamir coins recomputed rather than
read from the proof, and the weight root against the trusted enrolled root.

For scale, not as a controlled comparison (different card, 40 columns, 1093
tokens): the all-expert prover took 19.27 h to prove and 13.97 h to verify.

## Gates

```bash
uv sync && (cd verifier && cargo build --release && cargo test --release --bin verify_proof)
cd prover
for t in fiat_shamir phase3_block routed_projected rescale_claim moe_routed \
         shard_streaming opening_ledger quad_eval_accumulator admission_gate \
         pipeline_integration; do python tests/run_tests.py test_$t; done
```

Mostly negative tests: fabricated routed output, output from an unrouted
expert, `P ≠ W rho`, rewritten transcript coins, tampered phase-3 columns,
missing policy, exhausted opening budget, and admission reports from another
tree, GPU, filesystem, model or row layout.

## Production

Commands in [demo/4h-production-runbook.md](demo/4h-production-runbook.md).
Enroll once → build the claim tape → measure an admission report on the same
source, model, machine, layout and filesystem → prove → save the opening ledger
→ verify with externally supplied root and statement digest. Geometry is pinned
at `ELL=8192, K_DEG=16384, N_LIG=65536, T_QUERIES=54`; the driver refuses a
cheaper dev geometry, random weights, a stale report or missing policy.

## Limitations

- One run on one rented machine, not reproduced.
- Synthetic token set — full protocol and cost, but the statement is not about
  a served conversation.
- Every enrolled weight is still read once for `P = W rho`: inactive
  *activations* are removed, not dependence on parameter count.
- Verification is real and unpriced — 1 h 51 min, no stage cap governs it.
- Zero knowledge is stateful: proofs against one root spend a cumulative column
  budget (4096 ≈ 75 proofs), and the ledger must not be rolled back.
- Research prototype, no external audit. Fiat–Shamir in the ROM.

## Details

| | |
|---|---|
| [routed-projected-protocol.md](analysis/routed-projected-protocol.md) | Normative construction and security reduction |
| [routed-projected-status.md](analysis/routed-projected-status.md) | The only authority on what is implemented |
| [routed_projected_4h_model.py](analysis/routed_projected_4h_model.py) | Operation ledger and the envelope |
| `prover/routed_projected.py`, `verifier/src/handlers.rs` | The claim and its independent Rust twin |
| `prover/admission.py` | Fail-closed startup gate |
