# `layergkr` — implementation of the Layer-GKR-LF protocol

Implements the protocol of `analysis/VerInf_LayerGKR_4h_theorem_ru.md` over real
integer layer semantics, with a cost model that has **no calibration factor** and
is validated against instrumented runs.

Self-contained: nothing outside this directory is modified. It imports
`prover/protocol.py` for the field, the message→codeword map and the Merkle
conventions, so everything here is stated in the encoding the production Rust
verifier already speaks.

```
.venv/bin/python layergkr/tests/run_tests.py        # 71 tests
.venv/bin/python layergkr/bench/kernels.py          # measure this box's loop bodies
.venv/bin/python layergkr/bench/run_toy.py          # prove, verify, check the counts
.venv/bin/python layergkr/bench/validate_time.py    # check the MACHINE model
.venv/bin/python layergkr/bench/run_toy.py --set real     # near-real widths (GPU)
```

**The cost model has its own document: [`analysis/layergkr-cost-model.md`](../analysis/layergkr-cost-model.md).**
It carries the three-level structure, every measured rate, and a change log of
each machine characteristic that was found and added. Improvements go there.

## What is implemented

| Doc | Mechanism | Module |
|---|---|---|
| §4 L3–L5 | project-before-sumcheck seam, `F_P = Σ χ_ρ(i) F_{W_i}`, same-column equality | `projection.py` |
| §3.2, §6 | commit-before-challenge as an **enforced schedule**, not prose | `transcript.py` |
| §4 L3 | relation zoo (hadamard, booleanity, affine/RoPE, rescale bracket) batched into one tagged ragged sumcheck | `relations.py`, `sumcheck.py` |
| §6 | LogUp with `β → R_cmp → α` ordering, multiplicities, reciprocals proved by sumcheck | `logup.py` |
| §7 | affine mask compiler, masked products — **wired into the prover** | `sumcheck.py`, `full_layer.py` |
| §5 | MoE hidden stable sort + delimiters + segmented scan + permutation fingerprint | `moe.py` |
| §8.1 | layer composition by exact root equality | `layer.py`, `full_layer.py` |
| §3 | persistent weight enrolment, deduplicated across tokens | `full_layer.py` |
| — | exact integer layer semantics: RMSNorm, RoPE, causal softmax, SiLU, residuals, every rescale bracket | `semantics.py` |
| §9 | count-based cost model, prover **and verifier** | `count_model.py` |

`semantics.forward` computes a real layer — sum-of-squares → isqrt lookup,
QKV matmuls, RoPE, causal scores, exp lookup, reciprocal-normalised softmax,
SiLU, SwiGLU hadamard, both residuals — with a raw accumulator and a
deterministic range-checked rescale after every multiply. `check_trace`
re-derives every relation independently of the emitter.

## The cost model has no kappa

Three levels, each falsifiable on its own. Full detail and the change log live in
[`analysis/layergkr-cost-model.md`](../analysis/layergkr-cost-model.md).

| level | maps | validated against | status |
|---|---|---|---|
| L1 | geometry → relation counts | the emitted trace | **exact** |
| L2 | relations → loop-body iterations | op counters | **exact (0.00%)** |
| L3 | iterations → seconds | the stopwatch | **0.80×–1.01×** |

L3 did not start there. It started at 0.69×–2.48×, and each improvement was a
machine characteristic that had been left out — never a coefficient:

```
mul/add rate card                          0.69x .. 2.48x
+ deferred vs reduced multiplication       0.65x .. 2.12x
+ price loop bodies, not primitives        (a least-squares fit returned a
                                            NEGATIVE cost per multiply: mul and
                                            add are collinear here, so they were
                                            never identifiable — the wrong unit)
+ model encode sparsity (scan vs mac)      0.47x .. 0.95x
+ model transpose and hash-call overhead   0.80x .. 1.01x
```

The largest of those: `rs.encode_row` tests `if v:` and skips zero slots, and
almost every row it encodes is mostly padding. A nonzero slot costs 100 ns, a zero
slot 33 ns. Pricing all `ELL·N` at the nonzero rate was the single biggest error.
**That has a protocol consequence too** — at Maverick geometry a 5120-wide row in
an `ELL = 8192` message is 37% padding, and encode + opening is ~75% of the
theorem's budget.

## Verifier cost — which the theorem does not model at all

The source document's only verification number is a conditional corollary in
§9.4 (`< 420 s` for a hypothetical streaming GPU verifier) explicitly excluded
from the theorem. Here it is modelled, predicted exactly, and measured:

```
run          S    d   E  prove s  verify s  v/p time  proof KB   opened
toy-xs       2    4   1     0.05      0.02     0.366       3.1      392
small-1      6   16   2     1.09      0.08     0.078      40.9    5,232
large-1      8   32   4     8.67      0.24     0.028     146.4   18,744
large-2     12   32   8    11.79      0.41     0.035     211.7   27,096
```

Verification is a **shrinking** fraction of proving as the instance grows
(37% → 3%), because the verifier's work is dominated by the `q`-column openings
and the per-round interpolations, neither of which grows with `d` the way the
prover's encode and projection do. That is a favourable result for the scheme —
and it is now a measurement rather than an omission.

## What is still not here

* **The MoE segmented path is not wired into the prover.** `moe.py` implements
  and gates the sort/delimiter/segment argument, but `full_layer.py` proves the
  routed FFN as per-token matmuls. So a 400B projection from this code would
  price the `S·E·K` structure the scheme exists to avoid. Wiring `moe.py` into
  `full_layer.py` is the next step, and until it is done **no 400B number is
  quoted here**.
* Small projected vectors (`A`, `P`) and the LogUp reciprocals travel in the
  clear, bound by re-encoding rather than by a local LF proof over the small
  roots. Soundness is intact; hiding of those specific vectors is not. The mask
  machinery that would carry them is built and gated, and is already carrying the
  gate batch.
* No enrolment ledger, no streamed binary proof format, no memory-phase liveness
  (doc §9.4, §10).
* The tables in `semantics.py` have the right shape (unary, bounded domain, one
  output per input) but not Llama's numeric content; proof cost depends on the
  shape.

## Status of the theorem's headline

`cost_model.py` (the earlier, κ-based model) still reproduces the document's
§9.2 geometry exactly from tensor shapes — `P = 402,725,114,880`,
`N_pad = 564,632,231,936` — so its arithmetic is sound. Its κ is the disputed
part: the doc adopts 1.5 (→3.95 h, break-even 1.537); our production prover
exhibits 1.83 (→4.57 h). The work in `count_model.py` is the constructive answer
to that dispute: a model that predicts counts exactly does not need a κ at all.

## Map

```
counters.py    op counters + the measured rate card (KernelRates)
gpu.py         GPU backend for encode/combine via the PRODUCTION Goldilocks
               kernels (prover/cuda_primitives). Enabled only after proving
               itself BIT-IDENTICAL to the CPU path, per Config.
field.py       Goldilocks ops that count themselves
rs.py          RS encode (cached Lagrange) + column Merkle
transcript.py  Fiat-Shamir + enforced commit-before-challenge schedule
projection.py  the weight seam
relations.py   the relation zoo + tagged ragged batching
logup.py       LogUp with the ordering discipline
sumcheck.py    sum-of-products sumcheck + §7 mask compiler
semantics.py   real integer layer semantics -> proof trace
moe.py         sort / segment / permutation argument (not yet wired in)
layer.py       dense-layer proof + chain composition
full_layer.py  the complete layer proof and verifier
count_model.py count-based cost model, prover + verifier, no kappa
bench/         kernels.py (loop-body rates), rates.py, run_toy.py,
               validate_time.py, diagnose.py, runs.jsonl
tests/         66 tests
```

## Inherited caveat

The document's one measured input (3609 s semantic forward, memory high-water)
comes from `analysis/full-model-hidden-run-archive.md`, whose raw logs live on
`spark-c191` and could not be verified from this machine.
