# VerInf: zero-knowledge proofs of LLM inference

VerInf produces zero-knowledge proofs of LLM inference for large models. It has
proven a 400B-parameter model, runs the prover on a single consumer NVIDIA DGX Spark,
and relies only on hash-based commitments, so there is no trusted setup and the
construction is plausibly post-quantum. The proof reveals nothing about the weights,
activations, or the tokens kept hidden. Input and output tokens are committed, and
either side can be revealed or hidden as the deployment requires (the full-model run
hides the prompt and reveals the continuation; hiding both at once is future work, see
the encrypted token-stream commitment under Future work). The prover is untrusted; a
small Rust verifier checks the proof.

Real inference runs in floating point, outside the finite field the proof works over,
and is nondeterministic (summation order alone changes the result). Instead of
requiring a bit-exact rerun, VerInf proves that the committed outputs are well
explained by a finite-field computation, and bounds the unexplained information: the
output bits that computation does not account for. Proving similarity rather than
exact reproduction is what lets it target a real, already-trained model, not just
computations built to match a finite-field circuit exactly.

This could be used to verify AI compute agreements (the bound caps the bandwidth
available for hidden work such as covert training), to detect model-weight
exfiltration, to audit that a deployed model matches an approved version, and to check
high-stakes inference for bugs or tampering.

## Status

VerInf has produced a sound, four-round Ligero proof of a 1093-token forward pass of
the full 48-layer Llama 4 Maverick (a 400B-parameter mixture of experts, all 128
experts committed per MoE layer): 19.3 hours to prove on a single DGX Spark at a
prover peak of 77.6 GB GPU memory, certifying an unexplained-information bound of
0.394 bits per token over the 651-token continuation. The 442-token prompt stays
hidden; the continuation tokens are public in this run. The prover opened 40 columns
(`LIGERO_T_QUERIES=40`) and the independent Rust verifier accepted at T=30 of the 40
(a per-challenge bound of about 2^-12.5), taking 14.0 hours on the Spark's 20 CPU
cores at 78.6 GB peak RSS over the 92 GB proof
(`analysis/full-model-sound-run-archive.md`).

Two soundness-relevant bugs were found and fixed after that run: the RMSNorm rsqrt
bracket turned out to be vacuous in the production configuration (a forged output was
accepted before the fix), and the constraint fold mishandled an operand shared across
MoE layers. A re-run of the full-model proof on the current prover, with a 1000-token
transcript (500 hidden prompt, 500 public continuation), is in progress and will
replace the numbers above. The smaller gates have already re-run clean on the current
code: the 32-layer Llama-2-7B at 1000 tokens proves in about 47 minutes at a 13 GB
peak, and its 1.4 GB proof verifies in about 24 minutes on 20 CPU cores.

## What is new

To our knowledge, published zero-knowledge inference work has reached about 13B
parameters (Sun, Li, and Zhang 2024 report 13 minutes for LLaMA-2 13B at 2,048 tokens
on an A100). VerInf reaches 400B, about thirty times larger, with a full prover and
verifier checked end to end.

VerInf also differs in what it proves. Earlier inference proofs require the
computation to match the integer proof structure exactly, so they cannot target a real
model whose inference runs in floating point. VerInf instead proves that the
committed output tokens are well explained by a policy-compliant integer computation
on the measured inputs, and bounds the output information that computation does not
account for. Floating-point inference is nondeterministic (summation order alone
changes the result), so instead of a bit-exact rerun the prover supplies its own
prediction Q of each output token. Writing D for the declared computation and O for
the outputs the hardware could produce:

```
unexplained information:  U(O) = H(O | D(x))
Gibbs upper bound:        U(O) <= - E_P[ log2 Q(O | D(x)) ]   for any predictor Q
prover's estimate:        U(o) = - sum_i log2 Q_i(o_i | D(x), o_<i)
logit-noise model:        Q_i(o_i) proportional to exp(-(v* - l_i)^2 / sigma^2),  v* = max_i l_i
```

A poor prediction only inflates the prover's own number, so Q can be left to the
prover, who is best placed to make it accurate. Checking prediction quality rather
than exact reproduction lets VerInf certify the bound for a real, previously trained
model: the weights, inputs, and output tokens stay committed and hidden, and only the
bound is revealed. The formulation and security analysis are in the
unexplained-information paper (PDF in this repository).

## The argument: Ligero over Goldilocks

VerInf uses Ligero because the setting tolerates large proofs and verifier work
(verification is occasional and offline), so we can favor a simple construction with a
small verifier over proof size. Ligero arranges the committed values as a 2D array,
supports a linear test and a Hadamard (quadratic) test over them, and opens a random
subset of columns to bind the commitment. VerInf stacks the weights, intermediate
activations, and auxiliary witnesses as separate blocks of one such array, each with
its own Merkle root.

Each operation compiles to a flat list of linear and quadratic constraints. Matmuls
use Freivalds: instead of checking C = A·B entrywise, the prover commits three small
projections, and the claim emits three linear and one quadratic constraint, so one
length-k dot product replaces the whole matmul. Elementwise operations use per-slot
quadratics; softmax, SiLU, and RMSNorm use LogUp table lookups. Ligero proves the
whole list at once:

- Commit: encode each row of the witness with a Reed-Solomon code (ELL = 8192 values
  to N_LIG = 65536 columns) and hash the columns into a BLAKE3 Merkle tree, whose root
  binds the witness.
- Test: random combiners fold all linear constraints into one polynomial and all
  quadratics into another, and the verifier checks that each vanishes where required.
- Open: the verifier names T random columns, and the prover reveals them with Merkle
  paths; the verifier checks they hash to the root and lie on low-degree codewords.

Soundness comes from Reed-Solomon distance: any inconsistency appears in a constant
fraction of columns, so a few random column checks make the chance of missing it
negligible. The soundness-grade configuration opens T = 80 columns; the development
path drops to around 4 for speed.

## Persistent weight commitment

The weight block sits at a fixed position, so its Merkle root depends only on the
model, not on the prompt. This means the weights can be committed once
(`WeightCommitment`, saved to disk) and referenced by later proofs instead of being
rebuilt, which removes the weight column-hashing (the compute-bound term at short
context) from every proof of the same model. Opening columns of a fixed commitment
gradually spends its zero-knowledge budget (8192 distinct columns at these
parameters), so the commitment can be refreshed: re-commit the same weights under a
fresh seed, and produce one linking proof that the old and new roots commit the same
weights, using an ordinary per-slot equality claim. Negative tests confirm that
linking a refreshed root to different weights is rejected, down to a single changed
element. An independent audit of the feature found one verifier bug (a block declaring
the empty-root sentinel skipped Merkle binding entirely), fixed with its own negative
test.

## Writing a model

The forward pass is built as ordinary tensor code against a tape, like PyTorch. Each
`tape.<op>` call records one claim (a public statement of what was computed) and
returns a handle; handles overload `@`, `*`, and `+`:

```python
norm = tape.rmsnorm(x, d=d, s=S, eps_int=EPS_INT)
g    = tape.hadamard_broadcast(norm, rms_w, SEQ=SEQ, d=d)
q    = tape.matmul(g, W_Q, s_a=S, s_b=S, s_out=S)
sc   = tape.matmul(qr, kr, transpose_b=True)      # attention scores
sm   = tape.softmax(sc, M=SEQ, s_x=S)
out  = tape.matmul(sm, v)
```

The same claim list drives witness generation, the constraint compile, and
verification. A model built from existing operations needs no change to the prover or
verifier; only a new operation requires a new claim type. The unexplained-information
bound was added this way, as two claim types reusing existing table lookups and
elementwise steps. Examples: `demo/demo_llama7b.py` and `demo/demo_maverick_moe.py`.

## The prover

The prover proceeds in five phases. Phases 2 through 5 run as four rounds, so each
commitment is fixed before the verifier draws the next challenge; otherwise the prover
could fit a witness to a challenge it has already seen.

1. Computation claims. The tape records the relationships between the values to be
   committed.
2. Intermediate witness. The prover walks the claim list, computes each value, commits
   it row by row (Reed-Solomon encode, then update the column hashes), and sends the
   commitment.
3. Auxiliary witness. The verifier issues challenges for the claim types that need
   them (such as the Freivalds projection vectors); the prover computes the auxiliary
   witnesses, commits them the same way, and sends the commitment.
4. Polynomial tests. The verifier provides combiner challenges; the prover sends the
   folded IRS, linear, and quadratic test polynomials (Ligero).
5. Column opening. The verifier names a random subset of columns; the prover reveals
   their plaintext and Merkle paths over the stacked array.

The prover runs on GPU: field arithmetic, the Reed-Solomon NTTs, Merkle hashing, and
the constraint-fold band kernels are CUDA kernels over the Goldilocks field,
JIT-compiled for the local card on first use. The full witness would exceed device
memory, so the prover streams it one operation at a time, encoding and hashing each
piece before freeing it; peak memory tracks the working set, not the model or proof
size.

## The verifier

The verifier runs different code from the prover and trusts only its own. Its tasks:

0. During the prover phases, collect commitments and sample challenges (very little
   work).
1. Parse the claim list and check it meets the policy: the expected kind of
   computation, with compliant information flow.
2. Re-hash the opened columns and their Merkle paths and check they match the
   committed root.
3. Recompute the IRS, linear, and quadratic test polynomials at the opened columns and
   check they match the prover's.

The verifier never trusts prover-supplied constraints; it recompiles them from the
public claim list and checks the proof against its own derivation. Its trusted base is
the field, the evaluation domains, BLAKE3, the constraint compile, and the checks
(crate dependencies: `blake3`, `rayon`, `serde_json`). It returns `ACCEPT` or
`REJECT`.

## Cost and benchmarks

Prover cost is dominated by the witness computation (large Int64 matmuls), the
Reed-Solomon NTTs, and the column hashing. The witness cost is Goldilocks field
arithmetic in the matmuls, a fixed factor more expensive than a low-precision
(e.g. NVFP4) multiply-accumulate; since proving prefill and batched decode are compute
bound, that factor carries into the estimate. The rest depends on NTT and hashing
throughput.

`design-feasibility.md` develops a cost model: per claim type, the witness size and
the number of linear and quadratic constraints as a function of the size parameters,
combined with measured NTT and hashing throughput on a few platforms (DGX Spark,
clusters of H100s and NVL72s) at a few NTT dimensions. This gives runtime, proof size,
and witness size across models, context lengths, hardware, and column widths. The
measured runs to compare against are archived in
`analysis/full-model-sound-run-archive.md` and
`analysis/prover-optimization-investigation.md`, which also has a measured breakdown
of where prover time goes and which optimizations help.

## Architecture: streaming and parallelization

The streaming prover commits row by row and accumulates the column hashes
incrementally, so values can be flushed once folded into the hashes; peak memory stays
at the working set. The same structure parallelizes: NTT work splits across rows and
GPUs, with hashing accumulated in dedicated nodes. This is how we expect to reach
larger models and longer contexts within a fixed time budget.

## Caveats

- Research prototype; no security audit of the full construction (the persistent
  weight commitment had an internal one, see above).
- The demonstrated run explains about 98% of the per-token information in an FP4
  Maverick token stream (0.394 unexplained bits per token against a 202,048-token
  vocabulary), which may not suffice for all applications.
- Proofs are large, megabytes to gigabytes (92 GB for the 1093-token full-model run at
  40 opened columns; higher soundness opens more columns and grows the proof).
- The soundness-grade protocol is interactive, which lets us tolerate lower
  per-challenge soundness without enabling grinding attacks.
- The demonstrated full-scale soundness is T=30 (a per-challenge bound of about
  2^-12.5, adequate for interactively-deterred parties). The gap to the T=80
  deployment grade is verify runtime, not memory: verifier RSS is dominated by parsing
  the opened columns (78.6 GB, identical at T=4 and T=30), while the per-column check
  took 14 hours at T=30 on 20 CPU cores and grows with T. A GPU verifier is the
  planned fix (`analysis/full-model-sound-run-archive.md`).
- The headline numbers predate the two fixes described in Status; the re-run on the
  current prover is in progress.

## Future work

### Engineering

- Parallelize the prover across multiple GPUs.
- Port the verifier to CUDA, keeping the Rust version for bit-exactness checks.
- Reduce prover time further. The two levers identified in
  `analysis/prover-optimization-investigation.md` have landed (the inverse-NTT fuse
  and the descriptor-kernel restructuring of the sparse linear constraints, see
  `analysis/docs/linear-fold-unification.md`); a faster NTT kernel was measured to
  have no headroom (memory-bandwidth bound on the GB10), so the remaining levers are
  fusing the fold's remaining DRAM round trip into shared memory and, past that,
  replacing Ligero's linear test with a sumcheck- or GKR-style protocol.
- Tune the claim parameters for a better tradeoff between performance and unexplained
  information.
- Support a wider column format for smaller proofs.
- Support more models, for example DeepSeek V4.
- Hide both input and output tokens at once: a parallel proof that
  Hash(AES(key, tokens)) matches a public token-stream hash and Hash(key) matches a
  public key hash, so the token commitments can be produced with standard network
  hardware while both token streams stay hidden. The AES lookup tables and
  token-to-byte plumbing have landed (`prover/token_binding.py`, difftested against a
  reference recorder); the end-to-end binding remains.
- Stream by proving claims one at a time rather than running through the same proof up
  to four times, for up to a 4x speedup, at the cost of more complexity and many more
  prover-verifier messages.

### Research

- Explain more of the information in low-precision (NVFP4) inference. A concrete
  design for exactly modeling NVFP4 matmuls under Freivalds, with the other levers on
  the bound ranked by bits-per-effort (per-channel scales and predictor improvements
  first), is in `analysis/docs/nvfp4-exact-path.md`, building on the measured
  groundwork in `analysis/nvfp4-matmul-predictability.md`.
- Investigate architecture changes so fewer intermediate values need to be committed,
  especially the squared attention matrix with attention applied to it, and MoE
  routing.
- A security review.
- Formal verification that the prover has no degrees of freedom in the witness. A
  manual claim-by-claim review plus targeted negative tests found one exploitable
  gadget (the RMSNorm bracket, since fixed); a formal treatment remains open. Also
  that causal attention masking prevents later tokens from influencing earlier ones.
- Automated semantic analysis of the claim graph, for example checking that each
  weight is only ever read in a forward pass, so no gradient descent is happening.
- Further investigate related projects, for better architectures, performance, or
  specialized hardware.

## Build and run

**Quickstart: one command.** With the gated `Llama-2-7b-hf` checkpoint downloaded
(`huggingface-cli login` then `huggingface-cli download meta-llama/Llama-2-7b-hf`, or
set `VERINF_GATE_MODEL` to a local path), a CUDA GPU, and Rust installed, run the
prove-then-verify gate:
```sh
python demo/gate.py
```
It builds the verifier if needed, proves a small real-weights forward with the
calibrated unexplained-information bound (output tokens hidden), checks it with the
Rust verifier, and prints `PASS` on success. It exits 0 only if the verifier ACCEPTs
*and* the bound is non-degenerate, so it doubles as the regression gate for large
changes (it opens `LIGERO_T_QUERIES=10` columns for speed; production soundness is
80). The steps below are the full manual walk-through.

### 1. Build the verifier (Rust, CPU-only)
```sh
cd verifier && cargo build --release
```
(`Cargo.lock` is regenerated on first build; commit it for a reproducible TCB.)

### 2. Set up the prover (Python + CUDA)
Needs an NVIDIA GPU and a CUDA toolkit (`nvcc` on `PATH`); the CUDA primitives
JIT-compile for the local GPU on first use (needs `ninja`).
```sh
python -m venv venv && . venv/bin/activate && pip install -r requirements.txt
cd prover
python tests/run_tests.py test_claims          # unit and differential tests
```

### 3. Prove the unexplained-information bound
Prove a bound on the unexplained information of a real Llama-2-7B generation, output
tokens committed and hidden. `ui_real_measure.py` generates 50 tokens in fp16,
recomputes the quantized int logits, and reports the bound U with the fp-vs-int
comparison; `ui_real_proof.py` proves the bound over the 32-layer forward and dumps
it. Both read a local `Llama-2-7b-hf` checkpoint (set `MODEL` in the scripts);
`--lazy-weights` streams the weights from disk (prover peak ~20 GB).
```sh
python ../analysis/ui_real_measure.py    # fp16 generate, int logits, U; saves both logit sets
python ../analysis/ui_real_proof.py      # prove U over the 32-layer forward (output tokens hidden)
../verifier/target/release/verify_proof /tmp/ui_real_proof.json   # prints rust_verify: ACCEPT
```
The bound folds onto the LM-head logits inside the proof; the output tokens never
enter the public claims, only U is revealed.

### 4. Variations
- Forward correctness only (without the bound):
  ```sh
  python ../demo/demo_llama7b.py --from-hf meta-llama/Llama-2-7b-hf --num-layers 32 --seq 10 \
      --lazy-weights --engine --prompt "Hello world" --dump-proof /tmp/proof.json
  ../verifier/target/release/verify_proof /tmp/proof.json
  ```
- Longer context: `--seq 1000 --prompt-file demo_prompt.txt` for the 1000-token run.
- The prover always runs the four-round commit-before-challenge protocol; soundness is
  set by `LIGERO_T_QUERIES` (the number of opened columns): the soundness-grade
  default is 80; `LIGERO_T_QUERIES=4` opens 4 instead, for a faster verify and dump
  during development.
- No checkpoint: drop `--from-hf` and `--lazy-weights` to run on random weights with
  nothing to download, e.g. `--num-layers 1 --seq 4 --no-lm-head --engine
  --dump-proof /tmp/proof.json`.

## Repository layout

- `prover/` is the prover (Python + CUDA): the tape, the claim/constraint/protocol
  code, the CUDA bridge, and the weight loader, with `kernels/` (the
  Goldilocks/NTT/BLAKE3 `.cuh` headers, JIT-compiled on first use), `tests/`, `ref/`
  (pure-Python validators), and `deprecated/` alongside.
- `verifier/` is the trusted Rust verifier (`verify_proof`); its tests live in the
  crate.
- `demo/` holds the runnable examples (`demo_llama7b.py`, `demo_maverick_*.py`).
- `analysis/` holds the design docs (`docs/`), the quantization-accuracy study and
  prover diagnostics (research scripts), and `bench/`.

## Further reading

- Unexplained-information paper (PDF, this repository): the formulation and security
  analysis.
- `design-feasibility.md`: the full specification and the cost model, including the
  mixture-of-experts routing and Llama 4 Maverick cost analysis.
- `analysis/full-model-v1-design.md`: the verified Maverick result.
- `analysis/full-model-sound-run-archive.md`: every metric of the full-model sound run
  (prove, verify, memory, soundness, the artifact).
- `analysis/persistent-weights.md`: the persistent weight commitment, refresh and
  linking design, and the audit findings.
- `analysis/docs/degrees-of-freedom-review.md`: the claim-by-claim prover
  degrees-of-freedom review.
- `analysis/docs/nvfp4-exact-path.md`: the NVFP4-exact claim design and the ranked
  levers on the unexplained-information bound.
- `analysis/prover-optimization-investigation.md`: the measured prover-time breakdown
  and optimization options.
- `analysis/docs/linear-fold-unification.md`: the constraint-fold architecture (bands,
  descriptors, the interpreter kernels) and its bit-exactness gating discipline; the
  paper's Appendix C is the concise specification.
- `analysis/quantization-evaluation.md`: the per-token quantization study.
- `analysis/docs/CLAIM_SPECS.md`: the claim type definitions.
- Code: `verifier/` (verifier), `demo/demo_llama7b.py` and
  `demo/demo_maverick_moe.py` (model examples), `prover/core.py` (the streaming
  prover), `prover/kernels/` (the Goldilocks, NTT, and BLAKE3 kernels).
