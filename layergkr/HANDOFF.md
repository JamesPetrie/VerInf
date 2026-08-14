# Handoff — read this first

Cold-start instructions for continuing the Layer-GKR-LF work. Written to be read
by someone (or some session) with no memory of how any of it happened.

---

## 1. What this is

`layergkr/` implements the protocol proposed in
`analysis/VerInf_LayerGKR_4h_theorem_ru.md` — layer-local tensor-GKR with Ligero
reduced to a PCS for boundaries — plus a cost model for it. It is self-contained:
nothing outside `layergkr/` is modified except additions to `analysis/TODO.md`,
`analysis/bench/research_journal.md` (entries iter18–iter26) and the standing
document `analysis/layergkr-cost-model.md`.

The existing VerInf prover is untouched. Verify with `git status` before and after
any work here.

```
.venv/bin/python layergkr/tests/run_tests.py     # 90 tests, must be green
```

## 2. State: what is built and verified

Mechanisms, all with positive AND tamper tests:

| doc | mechanism | module |
|---|---|---|
| §4 L3–L5 | project-before-sumcheck seam, same-column equality | `projection.py` |
| §3.2, §6 | commit-before-challenge as an enforced schedule | `transcript.py` |
| §4 L3 | relation zoo batched into one tagged ragged sumcheck | `relations.py` |
| §6 | LogUp, `β → R_cmp → α` ordering | `logup.py` |
| §7 | affine mask compiler, wired into the prover | `sumcheck.py` |
| §5 | MoE hidden route: sort, delimiters, segmented scan | `moe.py`, `full_layer.py` |
| §8.1 | layer composition by exact root equality | `layer.py` |
| §9.2 | multi-row layout for `n_in > ELL` | `projection.py` |

Performance work, all gated on **bit-identical** output to the reference path:

* commitments are tensor-native (encode/hash/Merkle on device, only the `q` opened
  columns return to the host);
* the sumcheck runs on device, masked proofs included;
* the encoder is **NTT**, not a dense `ELL x N` product — see §5 below, it is the
  single most important thing in this file;
* weight enrolment accepts device tensors.

Largest configurations actually proved and verified: `d=768` (3 blocks) at
`ELL=512/N=2048`, and production Ligero geometry `ELL=8192/K=16384/N=65536`
bit-checked by sampled columns.

## 3. THE NEXT TASK

The previous next task — tensorise `semantics.py` — is **done**, gated on a
byte-identical proof, and written up in `analysis/layergkr-cost-model.md` §4.9.9
and journal iter27. The forward pass is now 10.8×–127× faster and reaches
`d=4096, d_ff=8192` in 0.24 s. It is also now 0.05–0.10% of its own pipeline.

**Tensorise the PROVER's Python-object choke points**, in this order — each is a
separate step with its own byte-identical gate, because they are independent:

1. `full_layer.Enrollment.enroll` keys weights by `tuple(tuple(row) for row in W)`.
   At `E=128` experts that key alone is ~3e10 Python ints. Key by content HASH of
   the device tensor instead; `PersistentWeights.enroll` already takes tensors.
2. `logup.LogUp` commits **one RS row per lookup QUERY** — a 2-tuple carried in an
   ELL-wide row. This is the same family of artefact as the dense Lagrange matrix
   (§5 below) and is now the largest single source of rows in a layer. It is also
   what made the CUDA grid cap bite. Packing many queries per row changes what
   LogUp commits to, so treat it as a protocol step, not a refactor.
3. `relations.prove_batch` pads every gate factor into a Python list. `_pad`
   already has a tensor branch; the rest of the path (and `sumcheck.prove_terms`,
   whose `T(v)` does `[x % P for x in v]`) does not.
4. `moe.source_records` builds one object per (token, coordinate) and the lane
   loop is `O(n_in² · S)`. That complexity, not the constant, is the problem.

Then, and only then, a production-width run. Do not quote a production number
before that: it is bounded by these, not by semantics.

**The gate to copy** (it is the one that caught nothing only because the work was
done carefully — do not read that as it being unnecessary):
`bench/validate_semantics.py` compares traces field by field, runs `check_trace`
on both, and requires the two to produce the same proof byte for byte.

## 4. Discipline that is not optional

These are not style preferences; each was learned by getting it wrong here.

* **Never swap a backend without a bit-exactness gate.** `rs._gpu_ok`,
  `rs._ntt_ok`, `sumcheck._sumcheck_gpu_ok` are the pattern: the fast path refuses
  to enable itself until it has reproduced the reference output. At geometries too
  large for a full reference, sample columns (`gpu.selftest_ntt_spot`).
* **Do not narrate performance. Measure it.** Use `profile.timeline` /
  `profile.stage`; they print an `unattributed` line by construction and refuse to
  be reasoned from above 10%.
* **Discard a warm-up** and report median/p95, never best-of.
* **No 400B number** until the model that produces it has been validated at the
  geometry it is extrapolating to. This has been the standing rule since iter18
  and it has been right every time.

## 5. Traps, and hypotheses already refuted

Do not re-derive these. Each cost real time.

**The encoder.** The dense `ELL x N` product was an artefact of this prototype,
not a protocol cost. Our message is the polynomial's VALUES at the K-th roots and
the codeword lives on the coset `gamma*<w_N>`, so a plain `rs_encode_rows` looks
unrelated — it is not a different convention, it is two missing steps:

```
values --iNTT_K--> coefficients --scale by gamma^i--> --pad to N--> NTT_N --> codeword
```

Bit-identical. The Lagrange matrix (537M cells, 13 min, 30 GB at production
geometry) is not needed at all.

**Where the time actually went, in order of discovery:**

| suspected | verdict |
|---|---|
| small commits / kernel launch overhead → the 78x | **wrong**, batching gives 1.5x and saturates by 256 rows/call |
| Fiat-Shamir → the 10.9% residual | **wrong**, residual unchanged |
| the profiler's own CUDA sync → the residual | real artefact (fixed), but **not** the residual |
| GC → 43% of runtime | **wrong**, confounded with warm-up; order-controlled value is ~9% |
| the residual itself | **first-call cost**; 0.0–0.1% after a warm-up |
| the NTT encode → the win | the encode is 1% of the path; **the win was not moving the data** |
| the tensor forward is 4.5 s at d=1024 | **wrong**, that was list→device marshalling; it is 0.07 s |

Seven confident explanations, six wrong. The stage timers caught each within
minutes. That is why they exist.

**Two hard limits, both measured, both easy to rediscover the slow way:**

* the batched NTT kernels use one block-row per message and a CUDA grid dimension
  is capped at **65535** — 65535 rows encode, 65536 raises `invalid configuration
  argument`. `gpu.encode_batch_ntt` chunks; `gpu.MAX_GRID_ROWS` is the constant.
* the tensor semantics carry the TRUE integer in int64, so every value must stay
  below **2^63**. Headroom is 32 bits at `d=128` and **6 bits at `d=4096`** — the
  toy's values grow by ~`n_in` per matmul because the rescale only divides by
  `scale`. `LayerTrace.headroom_bits` reports it on every run.

**The asymptotics, stated correctly** (an earlier claim of "linear in d" was
retracted — see `analysis/layergkr-cost-model.md` §4.9.1):

* `Θ(P)` in parameters — every weight is read and projected;
* `Θ(d²)` in width when `d_ff ∝ d`, because `C_proj = n_out·ELL·⌈n_in/ELL⌉`;
* `Θ(E)` in experts — binding a hidden route touches every expert's weights;
* `Θ(S²)` for dense attention;
* the Ligero "quadratic constraint" is the DEGREE of `z = x·y`, **not** a
  complexity — those batch and cost time linear in their count.

## 6. Open, and honestly unresolved

* The three-term fit `t = 7.21 s + 1587 ns·C + 0.469 ns·C·N` (R²=0.9998, 7 points)
  has a `C·N` coefficient 16x larger than a direct measurement of the encode path.
  **Where that 16x lives is unknown.** It predates the NTT and tensor-input fixes,
  so re-fit before reasoning about it.
* `ns/capacity` does not converge, and correctly so: cost needs two scaling terms
  plus a constant, so a single per-row price does not exist. `1587 ns·C` is
  everything linear in C, not "a price per row" — a true per-row term would be
  `b_row·(C/ELL)`.
* Prototype simplifications that are NOT the scheme: the small projected vectors
  and the LogUp reciprocals travel in the clear, bound by re-encoding rather than
  by a local LF proof over the small roots. Soundness intact, hiding of those
  vectors is not.
* Not implemented at all: enrolment ledger, streamed binary proof format,
  memory-phase liveness (doc §9.4, §10).

## 7. Where things are

```
layergkr/
  profile.py        stage timing with a mandatory `unattributed` line
  counters.py       op counters + KernelRates (per-backend, measured not fitted)
  gpu.py            NTT encoder, tensor commit path, all bit-exactness selftests
  rs.py             RS + column Merkle; picks CPU / dense-GPU / NTT per Config
  semantics.py      both paths: `forward` (reference) and `forward_tensor`
  full_layer.py     <- THE NEXT TASK lives here and in logup/relations/moe
  count_model.py    counts model, L1/L2 exact
  bench/            kernels, commit_sweep, row_capacity, long_run, production_run,
                    validate_semantics (the equality gate), semantics_ladder
  tests/            90 tests

analysis/layergkr-cost-model.md    the standing document; §7 is its change log
analysis/bench/research_journal.md iter18-iter26
```

## 8. One inherited caveat

The theorem's only measured input (3609 s semantic forward, memory high-water)
comes from `analysis/full-model-hidden-run-archive.md`, whose raw logs live on
`spark-c191` and could not be verified from this machine. Anything derived from
it inherits that.
