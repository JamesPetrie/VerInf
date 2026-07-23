# VerInf: zero-knowledge proofs of LLM inference

VerInf produces zero-knowledge proofs of large-language-model inference. It has proven a 400B-parameter model with the prover running on a single consumer NVIDIA DGX Spark, and it uses only hash-based commitments, so there is no trusted setup and the construction is plausibly post-quantum.

VerInf certifies an observed output stream against a committed integer model and bounds the *unexplained information*: the output bits that computation does not account for. Proving similarity rather than exact reproduction is what lets it target a real, already-trained model whose inference runs in floating point. The proof reveals nothing beyond this bound and the public claim list, which discloses the model architecture but not the weights, activations, or hidden tokens.

This can verify high-stakes AI compute agreements between parties who do not trust each other's hardware, and unilaterally detect model-weight exfiltration, audit that a deployed model matches an approved version, or check inference for bugs or tampering.

The formulation, the construction, the soundness analysis, and the cost model are in the paper ([PDF](paper.pdf), generated from [`paper.md`](paper.md)); this README covers what the system does and how to run it.

## Status

VerInf has produced a sound, four-round proof of a 1000-token forward pass of the full 48-layer Llama 4 Maverick (a 400B-parameter mixture of experts, all 128 experts committed per layer), with every token hidden. Proving took 14.3 hours on a single DGX Spark (78.1 GB GPU, 83.9 GB unified), certifying an unexplained-information bound of 0.880 bits per token over the 500-token continuation, the proof's only public value. The independent Rust verifier accepted at 40 opened columns in 17.7 hours on 20 CPU cores over the 93.6 GB proof. The smaller gate, a 32-layer Llama-2-7B at 1000 tokens, proves in about 44 minutes and verifies in about 23 minutes.

See [`analysis/full-model-hidden-run-archive.md`](analysis/full-model-hidden-run-archive.md) for every metric of the full-model run.

## Caveats

- Research prototype; no security audit of the full construction.
- The demonstrated run explains about 95% of the per-token information in the token stream (0.880 unexplained bits per token against a 202,048-token vocabulary), which may not suffice for all applications.
- Proofs are large (93.6 GB for the 1000-token full-model run at 40 opened columns; higher soundness opens more columns and grows the proof).
- The protocol is interactive, which allows a lower per-challenge soundness level without enabling grinding. The demonstrated full-scale run opened 40 columns; raising it to the deployment grade of 80 costs verifier runtime, not memory (a GPU verifier is the planned fix).

## Build and run

**Quickstart.** With the gated `Llama-2-7b-hf` checkpoint downloaded (`huggingface-cli login` then `huggingface-cli download meta-llama/Llama-2-7b-hf`, or set `VERINF_GATE_MODEL` to a local path), a CUDA GPU, and Rust installed:

```sh
python demo/gate.py
```

This builds the verifier, proves a small real-weights forward with the calibrated unexplained-information bound (output tokens hidden), checks it with the Rust verifier, and prints `PASS` on success. It exits 0 only if the verifier ACCEPTs and the bound is non-degenerate, so it also serves as the regression gate. It opens 10 columns for speed; production soundness is 80.

The manual walk-through:

```sh
# 1. Build the verifier (Rust, CPU-only). Cargo.lock regenerates on first build; commit it for a reproducible TCB.
cd verifier && cargo build --release

# 2. Set up the prover (Python + CUDA). Needs an NVIDIA GPU, nvcc on PATH, and ninja (primitives JIT-compile on first use).
python -m venv venv && . venv/bin/activate && pip install -r requirements.txt
cd prover && python tests/run_tests.py test_claims          # unit and differential tests

# 3. Prove the unexplained-information bound over a 32-layer Llama-2-7B forward, output tokens hidden.
python ../analysis/ui_real_measure.py    # fp16 generate, recompute int logits, report the bound U
python ../analysis/ui_real_proof.py      # prove U over the 32-layer forward
../verifier/target/release/verify_proof /tmp/ui_real_proof.json   # prints rust_verify: ACCEPT
```

Both step-3 scripts read a local `Llama-2-7b-hf` checkpoint (set `MODEL` in the scripts) and stream the weights from disk lazily (prover peak ~20 GB).

**Variations.**

- Forward correctness only, without the bound:
  ```sh
  python ../demo/demo_llama7b.py --from-hf meta-llama/Llama-2-7b-hf --num-layers 32 --seq 10 \
      --lazy-weights --engine --prompt "Hello world" --dump-proof /tmp/proof.json
  ../verifier/target/release/verify_proof /tmp/proof.json
  ```
- Longer context: add `--seq 1000 --prompt-file demo_prompt.txt`.
- Soundness is set by `LIGERO_T_QUERIES`, the number of opened columns: the default is 80; use `LIGERO_T_QUERIES=4` for a faster verify during development. The prover always runs the four-round commit-before-challenge protocol.
- No checkpoint: drop `--from-hf` and `--lazy-weights` to run on random weights, e.g. `--num-layers 1 --seq 4 --no-lm-head --engine --dump-proof /tmp/proof.json`.

## Writing a model

Models are written as ordinary tensor code against a tape, in the style of PyTorch. Each `tape.<op>` call records one claim and returns a handle; handles overload `@`, `*`, and `+`:

```python
q   = tape.matmul(g, W_Q, s_a=S, s_b=S, s_out=S)
sc  = tape.matmul(qr, kr, transpose_b=True)   # attention scores
sm  = tape.softmax(sc, M=SEQ, s_x=S)
out = tape.matmul(sm, v)
```

The same claim list drives witness generation, the constraint compile, and verification. A model built from existing operations needs no change to the prover or verifier; only a new operation requires a new claim type. See [`demo/demo_llama7b.py`](demo/demo_llama7b.py) and [`demo/demo_maverick_moe.py`](demo/demo_maverick_moe.py).

## Repository layout

- [`prover/`](prover/) -- the prover (Python + CUDA): the tape, the claim/constraint/protocol code, the CUDA bridge, and the weight loader, with `kernels/`, `tests/`, and `ref/` (pure-Python validators) alongside.
- [`verifier/`](verifier/) -- the trusted Rust verifier (`verify_proof`).
- [`demo/`](demo/) -- runnable examples (`demo_llama7b.py`, `demo_maverick_*.py`).
- [`analysis/`](analysis/) -- design docs ([`docs/`](analysis/docs/)), the quantization-accuracy study, and prover diagnostics.

## Further reading

- **Paper ([PDF](paper.pdf), this repository)** -- the formulation, the construction, the soundness analysis, and the cost model.
- [`analysis/design-feasibility.md`](analysis/design-feasibility.md) -- the full specification and cost model, including the mixture-of-experts routing and Llama 4 Maverick analysis.
- [`analysis/full-model-hidden-run-archive.md`](analysis/full-model-hidden-run-archive.md) -- every metric of the all-hidden full-model run.
- [`analysis/docs/nvfp4-exact-path.md`](analysis/docs/nvfp4-exact-path.md) -- the NVFP4-exact claim design and the ranked levers on the bound.
- [`analysis/prover-optimization-investigation.md`](analysis/prover-optimization-investigation.md) -- the measured prover-time breakdown and optimization options.
