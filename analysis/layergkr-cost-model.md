# Layer-GKR-LF cost model

Living document. Every improvement to the model lands here, with the measurement
that motivated it, so nothing is rediscovered twice.

Scope: the model predicts the cost of proving and verifying one layer of
`analysis/VerInf_LayerGKR_4h_theorem_ru.md`. Code: `layergkr/count_model.py`
(counts), `layergkr/counters.py` (`KernelRates`), `layergkr/bench/` (measurement).

---

## 1. Shape of the model

```
        geometry            counts              seconds
    (S, d, d_ff, E)  ──►  loop-body     ──►   counts · rates
    (ELL, K, N, q)        iterations          (rates measured per machine)
          L1                  L2                    L3
```

Three levels, each validated separately, each falsifiable on its own:

| level | maps | validated by | current status |
|---|---|---|---|
| **L1** | geometry → relation counts | emitted trace | **exact** |
| **L2** | relations → loop-body iterations | op counters | **exact (0.00%)** |
| **L3** | iterations → seconds | stopwatch | **0.68×–1.28×**, see 4.6 |

**There is no calibration factor anywhere.** An earlier version of this model
produced seconds directly and multiplied by `κ ≤ 1.5`. That is not a safety
margin; it is the size of the modelling error, and it cannot be falsified — every
outcome is "within κ" for a large enough κ. Each time L3 disagreed with the
stopwatch, the fix was to find the machine characteristic that had been left out.
Section 4 is the log of those, and it is the part worth reading.

---

## 2. L1 — geometry to relations

For one layer with sequence `S`, width `d`, FFN width `d_ff`, experts `E`:

```
matmuls        = 5 + 3S                     Wq,Wk,Wv, attn·pv, Wo, then per token Wg,Wu,Wd
matmul cells   = 3·S·d² + S²·d + S·d² + S·(2·d·d_ff + d_ff·d)
gates          = 5 + 13S + matmuls
lookup queries = S (isqrt) + S² (exp) + S (recip) + S·d_ff (silu) + cells (range)
```

`gates = 5 + 13S + matmuls` decomposes as: 1 RMS square, 1 sum-of-squares
bracket, 2S RMS (hadamard + rescale), 4S RoPE (affine + rescale, for q and k),
1 causal-mask booleanity, 1 score bracket, 2S softmax, 1 sum bracket, S residual
1, 4S FFN (bracket + hadamard + rescale + residual), and one rescale gate per
matmul.

Verified exactly against the emitted trace for every configuration in
`layergkr/bench/runs.jsonl`.

**The MoE segmented path is now wired in.** The routed FFN is one node per
projection (gate, up, down) with the route SECRET, proved by the sort /
permutation-fingerprint / delimiter-segment argument of `moe.py` plus the §5.3
scalar identity, instead of one published matmul per token. Cost consequence: the
seam runs for **every** expert (that is what hides the route), so a MoE node costs
`E` projections rather than one — which is exactly the term §9.2 prices.

L1 for the FFN is therefore: `3` MoE nodes per layer, `E` weight tensors each,
`E·d·d_ff` projected cells; the per-token matmul count drops from `5 + 3S` to `5`.

---

## 3. L2 — relations to loop-body iterations

The unit of account is **one iteration of a loop that exists in the source**, not
an abstract "field multiplication". Section 4.2 explains why that distinction is
load-bearing rather than pedantic.

| unit | loop body | where |
|---|---|---|
| `enc_slot` | `acc += v * L[c]` — a **nonzero** slot | `rs.encode_row` |
| `enc_scan` | the same loop visiting a **zero** slot (skipped by `if v:`) | `rs.encode_row` |
| `comb_iter` | `out[j] = (out[j] + cw[j]*a) % P` | `rs.linear_combination`, `projection.project_message` |
| `fold_iter` | `(f[i] + x*(f[h+i] - f[i])) % P` | `sumcheck` round evaluation and folding |
| `red_op` | `(a * b) % P` | reductions, `eq_vector`, `batch_inv`, tuple compression |
| `xpose_iter` | one element of the column transpose | `rs.Commit.__init__` |
| `hash_calls` | pack + BLAKE3 of one column | `rs.Commit.__init__` |
| `hash_bytes` | bytes fed to BLAKE3 | as above |

Per-construct counts (all mirrored in `count_model.py`):

```
encode_row(cfg)          enc_scan = ELL·N,  enc_slot = nnz·N,  red_op = N
Commit(rows)             xpose_iter = rows·N,  hash_calls = N + (N-1),
                         hash_bytes = 8·rows·N + 64·(N-1)
linear_combination(m)    comb_iter = m·N
project_message(n_out)   comb_iter = n_out·ELL
sumcheck(size, terms)    per round, half = size/2^(r+1):
                           fold_iter = (deg+1)·half·Σ|factors|  + half·Σ|factors|
                           red_op    = (deg+1)·half·Σ|factors|
eq_vector(n)             red_op = 2·(2ⁿ − 1)
batch_inv(n)             red_op = 3n,  inv = 1        (Montgomery, ONE inversion)
```

`nnz` is the number of nonzero message slots — see 4.3, it is not `ELL`.

---

## 4. L3 — the machine model, and its history

Rates are measured by `layergkr/bench/kernels.py` and stored in `kernels.json`.
They are **measured, never fitted to a prove run**, so a modelling error cannot
hide inside them.

Measured on the dev box (CPython 3.12):

```
enc      100.5 ns   nonzero encode slot
scan      32.9 ns   zero encode slot (loop + truth test only)
comb     186.6 ns   combine iteration
fold     264.0 ns   sumcheck fold iteration
red      210.3 ns   isolated (a*b) % P
xpose     23.1 ns   one transposed element
hashcall  5450 ns   pack + BLAKE3 of one column
```

### 4.1 Reduced vs deferred multiplication → 2.48× to 2.12×

The first rate card measured `(a*b) % P`. The hot loops do not do that: they
accumulate `a*b` into a wide Python integer and reduce **once** at the end. Those
are different operations — 170 ns vs 63 ns. Charging the reduced cost for a
deferred multiply over-prices encoding roughly 2×.

### 4.2 "Field mul" and "field add" are not identifiable → the unit changed

With the split in place the model still drifted. A least-squares fit for the
per-primitive costs over the real workload returned a **negative** cost per
multiplication. Negative time is impossible, so the fit was not noisy — the
parameterisation was wrong: in this code multiplications and additions occur in a
fixed ratio, so they are collinear and cannot be separately identified from any
amount of data.

The conclusion is stronger than "use a better fit": **abstract primitives are the
wrong unit**. They do not correspond to anything the machine executes. A loop body
does. Everything from here on counts loop-body iterations.

*(This is the general lesson: if a fitted coefficient comes out physically
impossible, the model's variables are wrong, not its data.)*

### 4.3 The encoder skips zeros; the model did not → 2.12× to 0.95×

`rs.encode_row` tests `if v:` and does no arithmetic for a zero slot. Almost every
row it encodes is mostly padding — a LogUp multiplicity row is one value in an
ELL-wide message. A nonzero slot costs 100 ns, a zero slot 33 ns. Pricing all
`ELL·N` slots at the nonzero rate was the single largest error in the model.

Encoding is therefore `scan(ELL·N) + mac(nnz·N)`, two terms with two rates.

> **This has a protocol consequence, not just a modelling one.** At Maverick's
> geometry a 5120-wide contraction row in an `ELL = 8192` message is **37%
> padding**. If the production encoder is sparsity-aware, real encode cost is a
> third below what a dense `ELL·N` model predicts — and encode + opening is ~75%
> of the theorem's budget. Worth asking the author whether §9.2's `N_pad` counts
> row *capacity* or real work; if capacity, the budget is conservative in his
> favour. Related: the ELL-vs-row-width lever in `layergkr/README.md`.

### 4.4 Unmodelled work: transpose and hash-call overhead → 0.95× to 1.01×

Two things were counted as free:

* `rs.Commit` builds columns from rows — a pure list transpose, 23 ns per element,
  and there are `rows·N` of them;
* hashing a column is dominated by **call overhead** (5.4 µs per column), not by
  throughput. Pricing it in GB/s under-charges by orders of magnitude at these
  buffer sizes.

### 4.5 A different backend is a different machine

`layergkr/gpu.py` runs encoding through the production Goldilocks kernels
(`prover/cuda_primitives.gl_matmul`). Two consequences for the model, both of
which the first near-real run exposed:

* **Rates are per backend, counts are not.** The same counts priced with the
  CPython card over-predicted by 13x-35x, because a GPU slot costs 0.006 ns
  against CPython's 100 ns -- a factor of ~17,000. The count model did not change;
  only the card did.
* **The sparsity saving of 4.3 does not exist on the GPU.** `gl_matmul` is dense:
  it multiplies every slot, zero or not. So the GPU path carries its own unit
  (`enc_gpu`, dense `rows*ELL*N`) rather than the CPU loop's `scan` + `mac` split.
  The same protocol has different cost structure on different hardware, and the
  model has to say so rather than average over it.
* **Marshalling becomes the dominant term.** Once encoding is on the device, what
  is left is moving the result back: `C.cpu().tolist()` plus the Python list
  conversion, ~115-135 ns per element. It was unmodelled, and the model
  under-predicted 0.35x-0.82x until it was added. A production prover keeps the
  data on the device and would not pay it -- but this implementation does, so the
  model must count it.

The GPU backend refuses to enable itself for a `Config` until it has proved
bit-identical to the CPU path (`gpu.selftest`, gated in `rs._gpu_ok`, five tests
in `tests/test_gpu.py`). A faster backend that computes something else would
invalidate every proof and every measurement at once.

### 4.6 Current agreement

Two more terms were found by PROFILING a real prove rather than guessing, and
between them they were most of the remaining gap:

* **Column packing is per VALUE, not per hash call.** `protocol.pack_column` does
  one `int.to_bytes(8)` per value inside a generator, then a join. The profile put
  it at 6.3 s of a 19.8 s prove — the single largest cost in the whole prover —
  and the model charged it only per hash CALL. 145 ns per value.
* **The Lagrange matrix build is one-time setup, and was landing inside the
  measurement.** ~3 s of `pow()` per Config. It is now pre-warmed before the
  stopwatch starts, the way a production prover builds its tables once.

```
run         meas s   model s   ratio
real-128     79.46     54.23    0.68x
real-64       9.15      7.69    0.84x
large-2       4.82      4.37    0.91x
small-2       1.08      1.09    1.01x
large-1       2.75      2.81    1.02x
small-1       0.61      0.73    1.21x
toy-s         0.14      0.18    1.28x
```

The drift is no longer monotone: the model now scatters around 1 rather than
sliding with size, which is the signature of remaining per-call constants rather
than a missing scaling term. `real-128` at 0.68x is the worst case and the next
thing to profile.

Reproduce: `.venv/bin/python layergkr/bench/validate_time.py`.

### 4.7 The prover moved onto the device

Profiling said the prover was CPython-bound, and a GPU-utilisation trace
confirmed it: **38 of 40 samples read 0%** during a 76 s prove. So the card was
idle while the model was being blamed. Three things moved:

* **Commitments are tensor-native.** `rs.Commit` now keeps the codewords as a
  device tensor and uses the production kernels `hash_columns_streamed` +
  `merkle_build_blake3`. This deletes `pack_column` (the largest single cost, 6.3 s
  of 19.8 s) AND the device→host marshalling: only the `q` opened columns cross
  back, not the whole matrix. Per-value CPU packing at d=128 fell from millions to
  7,168.
* **The sumcheck runs on the device**, including MASKED proofs — the §7 mask
  touches only the scalar samples, never the vector work, so it does not block the
  device path.
* Both are gated on producing **bit-identical** output to the CPython path
  (`rs._gpu_ok`, `sumcheck._sumcheck_gpu_ok`, `tests/test_gpu.py`). A faster
  backend that computes something else would invalidate every proof at once.

Effect at the largest validated width:

```
                       prove    verify
d=128, before          76.0 s    4.5 s
d=128, after           16.8 s    2.8 s      4.5x
d=192, after           19.2 s    5.4 s      (a width that did not run before)
```

**Modelling consequence.** The unit table of §3 now has two columns, not one: the
same protocol step has a CPU unit and a device unit with different cost
structure (the CPU encoder skips zeros, `gl_matmul` does not). The model selects
per backend; `counters.KernelRates` carries both.

### 4.8 Multi-row layout

A contraction wider than `ELL` does not fit one RS row. Each output coordinate
now spans `n_blocks = ceil(n_in / ELL)` rows, laid out BLOCK-MAJOR
(`row = b * n_out + i`), so one block's values are contiguous at any opened
column and the seam's linear combination applies per block.

This is not optional at production geometry: Maverick's FFN contracts over 16384
against `ELL = 8192`, so it needs two blocks. It is exactly the `ceil(n_in/ELL)`
factor in the theorem's `N_pad` (§9.2), which this model already used — the
formula assumed a layout the implementation did not have.

Soundness carries over unchanged: the manifest aligns all output coordinates of a
block on the same message and padding positions, so the `(K/N)^q` argument holds
per block, and the verifier checks every block at every opened column.

Cost consequence for L2: `project_message` is `n_out * ELL * n_blocks`, the
projected commitment has `n_blocks` rows instead of one, and the opening carries
`n_blocks` values per column.

Note for anyone reading the proof objects: the FLAT projected vector (length
`n_in`) is what the contraction sumcheck runs on; the BLOCKS (full `ELL` wide,
padding tail included) are what the re-encode binding needs, because the
committed row is the projection of the secret padding too. Both are carried, and
tampering either one is rejected — by the binding and by the contraction
respectively.

### 4.9 Width ladder actually run

Everything below is a full layer proved AND verified to ACCEPT on the dev V100,
at the production code rate `K/N = 1/4`. `blocks` is `ceil(n_in/ELL)` for the
widest contraction in the layer, i.e. whether the multi-row layout of 4.8 is
engaged.

```
   d   d_ff   ELL     N   blocks    prove    verify
  32     64    64   256        1     4.6 s     0.2 s
 128    256   256  1024        1    16.8 s     2.8 s
 192    384   384  1536        1    19.2 s     5.4 s
 256    512   512  2048        1    30.1 s     7.7 s
 384    768   512  2048        2    31.2 s    11.5 s
 512   1024   512  2048        2    44.0 s    12.3 s     peak GPU 5.38 GB
 768   1536   512  2048        3    63.8 s    22.4 s     peak GPU 5.56 GB
```

**No wall was found**, and an earlier version of this section drew a WRONG
conclusion from that. It said "time grows linearly in d, not quadratically". That
is false as a general statement, and the model's own formula says so.

### 4.9.1 Correction: what is actually quadratic

The projection cost of one weight tensor is, from §3:

```
C_proj = n_out * ELL * ceil(n_in / ELL)
```

For a transformer matrix `n_in, n_out = Θ(d)`, so at fixed ELL and large enough d

```
C_proj = Θ(d * ELL * d/ELL) = Θ(d^2)
```

The multi-row layout does not make the work linear. It splits `d^2` weights into
blocks. The MoE line in §4.8 says the same thing explicitly: `E * d * d_ff`, which
at `d_ff = Θ(d)` is `Θ(E d^2)`. The document's own arithmetic contradicted its
prose, and the prose was wrong.

**Why the measurements looked linear.** The ladder is small and crosses block
boundaries in steps. While `n_in <= ELL` the block count is 1 and the cost really
does read as `n_out * ELL = Θ(d)` — but most of each row is padding, and that is a
transient regime, not an asymptote. Counting the row capacity the three FFN
projections actually process (`d_ff = 2d`, `ELL = 512`):

```
   d    FFN row capacity   vs d=384
 384          1,179,648       1.00x
 512          1,572,864       1.33x
 768          4,325,376       3.67x
```

Width doubled from 384 to 768; projection work grew **3.67x**, close to the 4x a
`d^2` term predicts. Wall-clock grew only 2.04x (31.2 -> 63.8 s). The two do not
match, and the conclusion is not "the protocol is linear" — it is that **the
projection term is not yet dominant at these sizes**. What the stopwatch is
measuring here is fixed per-call cost, under-occupied GPU, terms that do not grow
with `d`, and the step function of the block count. That is an observation about
this range, not an established asymptotic.

### 4.9.2 The quadratics that remain

| variable | cost | why |
|---|---|---|
| parameters `P` | `Θ(P)` | every weight must be read and projected |
| width `d` (with `d_ff ∝ d`) | `Θ(d^2)` | there are `Θ(d^2)` weights in the matrix |
| experts `E` | `Θ(E)` | binding the hidden route touches ALL experts' weights |
| context `S` | `Θ(S^2)` | dense attention and the S×S softmax |
| Ligero "quadratic constraint" | NOT `O(n^2)` | that is the DEGREE of the equation `z = x*y`, not a complexity; such constraints batch and cost time linear in their count |

So the protocol is **not** `O(P^2)` in parameters — it is roughly linear in `P` —
but because `P ~ d^2` it is quadratic in the model width, and dense attention stays
quadratic in context length. At a fixed `S = 1000` the attention term is a constant
share of the budget; it becomes the binding constraint if the context grows.

What the scheme actually removes is a different, artificial dimension:

```
   S*E*d*d_ff          ->     E*d*d_ff      +      S*d*d_ff
 (all experts,               (one weight          (only the active
  every token)                projection)          expert per token)
```

That is a real structural win. The `E*d*d_ff` term is still quadratic in width and
linear in the total expert parameter count, and no rearrangement removes it: an
online proof that applies a fresh random projection to all weights cannot be
sublinear in those weights without precomputed structure that can answer it.

**Lower bound to keep in mind:** `T >= Ω(P / R_projection)`. Any 400B claim has to
respect it.

### 4.9.3 What this ladder can and cannot support

It validates the implementation and the counters. It cannot price 400B, for two
independent reasons: it never varied `ELL`/`N` (the production geometry is ~512x
the per-row cost), and within it the projection term is not yet dominant.

The measurement that would fix this is a ladder over `ELL` and `N` at moderate
width, checking whether

```
        T(d)
  ---------------------------------
  n_out * ELL * ceil(n_in / ELL)
```

stabilises. If that ratio converges, it is an honest price per row of capacity and
extrapolation becomes defensible. Until it does, it is not.

### 4.9.4 First results from the capacity ladder

```
run           d   ELL      N       capacity  prove s   ns/cap   step
w-128       128   512   2048      1,658,880     11.9  7182.80
w-256       256   512   2048      3,297,280     12.3  3732.15   0.52x
w-512       512   512   2048      7,622,656     20.1  2640.96   0.71x
g-256       128   256   1024        829,440      6.9  8373.48
g-512       128   512   2048      1,658,880      8.4  5073.66   0.61x
g-1024      128  1024   4096      3,317,760     17.8  5376.16   1.06x
```

**Width axis: NOT converged.** ns-per-capacity keeps falling, 7183 -> 3732 ->
2641. The steps are shrinking (0.52x, 0.71x) but the price is still dropping, so
fixed costs are still a large share and the projection term is still not
dominant. This confirms the diagnosis of 4.9.1 rather than resolving it: the width
ladder cannot price anything.

**Geometry axis: the first flat step.** 8373 -> 5074 -> 5376, the last step 1.06x.
Between g-512 and g-1024 the capacity doubled and the clock went 8.4 -> 17.8 s
(2.12x), i.e. time tracked capacity almost exactly. That is what convergence would
look like.

**Three reasons it is not yet a price**, and they must be stated before anyone
uses the number:

1. One flat step is not convergence. Two points define a line.
2. `N` co-varied with `ELL` in this ladder, so the metric does not isolate either.
   The capacity formula counts message slots (`n_out * ELL * ceil(n_in/ELL)`) and
   says nothing about `N`, while the encode work is `capacity * N`. Whether the
   flat step reflects the ELL term or a cancellation between the two is untested.
3. Production is `ELL = 8192, N = 65536` -- 8x and 16x beyond `g-1024`.

Next measurement, in this order: extend the geometry ladder (ELL=2048/N=8192,
then 4096/16384), and vary `ELL` and `N` INDEPENDENTLY so the metric attributes
the cost to the right parameter. Only if ns-per-capacity is flat across several
steps AND the two parameters are separated does a per-row price exist.

### 4.9.5 The factorial experiment: there is no single per-row price

An hour-long run with `ELL` and `N` varied INDEPENDENTLY at fixed model width
(d=128). This is the measurement 4.9.4 said was needed.

```
run               d   ELL      N      capacity  setup s  prove s    ns/cap  ns/cap/N
E512-N4096      128   512   4096     1,658,880      3.2     13.8    8328.1   2033.23
E1024-N4096     128  1024   4096     3,317,760      6.4     18.1    5444.3   1329.16
E2048-N4096     128  2048   4096     6,635,520     12.4     30.5    4603.9   1124.01
E1024-N2048     128  1024   2048     3,317,760      3.1     15.4    4635.4   2263.39
E1024-N8192     128  1024   8192     3,317,760     12.5     24.9    7506.7    916.34
E2048-N8192     128  2048   8192     6,635,520     24.4     43.9    6619.5    808.04
E4096-N16384    128  4096  16384    13,271,040     99.8    130.2    9811.0    598.82
```

**Neither column is flat, and the reason is that the question was
underspecified.** The three N-axis rows have IDENTICAL capacity (3,317,760) and
times of 15.4 / 18.1 / 24.9 s — so capacity alone cannot be the unit. And
`ns/cap = a/cap + b + c*N` can never be flat while `N` varies. The prescribed test
was right and it correctly FAILED; what it told us is that the cost needs two
scaling terms, not one price.

**Three-term fit, R² = 0.9998 over 7 points spanning 8x in capacity and 8x in N:**

```
t  =  7.21 s                fixed
    + 1586.73 ns  * capacity            everything LINEAR IN C
    +    0.469 ns * capacity * N        everything linear in C*N
```

**Naming, corrected.** `1586.73 ns * C` is not "a price per row". It is the
coefficient of everything linear in capacity — a true per-ROW term would be
`b_row * (C / ELL)`, since a row holds ELL capacity slots. The fit does not
separate the two, and calling it a row price hid that.

The mix moves exactly as it should as the geometry grows toward production:

```
                 fixed    row   encode
E512-N4096         52%    19%      23%
E1024-N4096        40%    29%      35%
E2048-N8192        16%    24%      58%
E4096-N16384        6%    16%      78%
```

At the far point the fixed term is down to 6% and the `C*N` term is 78%. So the
structure IS emerging — the earlier ladders were not wrong to look flat, they were
simply in the regime where the constant dominated.

**This CONFIRMS the quadratic-in-width reading of 4.9.1, it does not soften it.**
With `C = n_out * ELL * ceil(n_in/ELL) = Θ(d^2)` for a transformer matrix, the
dominant term is `t_encode ~ c * C * N = Θ(d^2 * N)`. The experiment shows the
term that carries the `d^2` becoming dominant as configurations grow — the
opposite of the linearity this document once claimed. In parameters it stays
linear (`P = Θ(d^2)`, so `t = Θ(P*N)`): `O(d^2)`, not `O(P^2)`.

**Do NOT extrapolate this fit.** The fitted `C*N` coefficient is 0.469 ns per
slot-position against 0.006 ns/slot measured for `gl_matmul` in isolation — 78x.

**What that 78x is, is NOT established.** An earlier version of this section
asserted it was small per-commitment matmuls paying launch and host→device cost.
The regression does not show that. It gives one end-to-end number and does not
decompose it into kernel launches, H2D/D2H, synchronisation, occupancy, matrix
construction, allocation, hashing, or intermediate materialisation. "Batch the
encodes and it approaches 0.006" is a HYPOTHESIS. Batching may recover most of
it, or 2-5x with the remainder sitting in memory traffic and synchronisation.

**The experiment that settles it** holds `C` and `N` CONSTANT and varies only the
number of CUDA calls — rows per call:

| run | total work `C*N` | commitments | rows per call |
|---|---|---|---|
| A | same | many | few |
| B | same | some | some |
| C | same | few | many |

against the model

```
t = a + L*K_launch + b_transfer*B + c_kernel*C*N + t_hash
```

If `c_kernel` approaches 0.006 ns as the batch grows, the hypothesis holds. If it
settles at, say, 0.05 or 0.2 ns, THAT is the real production rate and it is what
any projection must use. Each stage — input prep, H2D, `gl_matmul`, sync,
commit/hash, D2H, Python orchestration — has to be timed separately with CUDA
events, and reported as median/p95 after warm-up rather than best-of-five, since
what is wanted is an upper bound.

Three fitted parameters on seven points is also thin. The fit is mechanistically
motivated (constant, linear in C, linear in C*N) rather than arbitrary, but it
wants more points once the commit sweep has named the terms.

**And this is still not a production run.** d=128, largest geometry 4096/16384.
Production `ELL=8192, N=65536` is 2x and 4x beyond that, with far more rows and
commitments, different memory pressure, and possible chunking boundaries not
exercised here.

### 4.9.6 Commit sweep: the batching hypothesis is REFUTED

`C` and `N` held fixed (ELL=1024, N=4096, 2048 rows = 8.59e9 slot-positions),
only the rows per CUDA call varied. Per-stage CUDA events, median and p95 after
warm-up.

```
rows/call  calls  wall ms   p95   prep    h2d  matmul   hash    d2h   ns/slot-pos  kernel
        8    256    378.1 380.3    0.1  225.3   100.3   19.9   10.7        0.0440  0.0117
       32     64    279.6 279.9    0.0  208.4    57.0    5.5    2.9        0.0325  0.0066
      256      8    256.0 257.5    0.0  201.4    51.9    1.4    0.4        0.0298  0.0060
     1024      2    252.6 255.2    0.0  201.3    50.0    1.0    0.1        0.0294  0.0058
     2048      1    254.3 259.5    0.0  203.4    49.7    0.9    0.1        0.0296  0.0058
```

**Batching buys 1.5x, not 78x, and it saturates by ~256 rows per call.** The
hypothesis in the previous revision of this section — that the 78x was small
per-commitment matmuls paying launch cost — is wrong. It was an assertion, it was
tested, and it failed.

What the stages actually say at saturation:

* **H2D is 80% of the encode+commit path** (201 ms of 253 ms). The messages are
  built as Python lists and shipped to the device every time. That, not launch
  overhead, is where this path's time goes.
* `gl_matmul` is 20%, at **0.0058 ns/slot-position — consistent with the 0.006
  measured in isolation.** The kernel is running at kernel speed.
* hashing and D2H are under 1% once the batch is large.

**And the number that matters: this whole path costs 0.0294 ns/slot-position,
while the full-prover fit attributed 0.469 ns to `C*N` — 16x more.** So most of
the fitted coefficient is NOT the encode/commit path at all. Either something else
in the prover scales like `C*N`, or the three-term fit is absorbing work of a
different shape into that term (7 points, 3 parameters, and no independent check
that the terms are separable).

**Where the 16x lives is unknown**, and the next step is to instrument the PROVER
per stage with the same CUDA events rather than propose another explanation. That
is twice now that a plausible story about performance was wrong; the profiler and
the stage timers have been right both times.

### 4.9.7 The residual was WARM-UP, and GC is 9% — two corrections in one section

The stage profiler prints an `unattributed` line by construction, and it read
10.9%, which under its own rule forbids reasoning from the table. Finding what it
was took three attempts, two of which produced confident wrong answers:

1. **Fiat-Shamir.** Wrapped `Transcript.absorb/coin` in a stage. Residual
   unchanged at 10.7%. Wrong.
2. **The profiler's own CUDA synchronisation.** `wall` was taken before
   `ev.synchronize()`, so the sync fell outside its own span. That WAS a real
   measurement artefact — it also serialised the device — and it is fixed (events
   are now resolved once at timeline exit). But the residual stayed at 10.9%.
   Wrong as the explanation.
3. **Garbage collection.** A paired run gave 20.47 s with GC on against 11.58 s
   with it off, and this section briefly claimed GC cost 43% of the runtime. That
   was wrong too: the second run was also WARMER. The comparison was confounded.

**Order-controlled measurement**, alternating after a discarded warm-up:

```
gc OFF   11.57  12.27  12.42  11.82     mean 12.02 s
gc ON    13.29  13.09  13.32  13.07     mean 13.19 s
```

So the cycle collector costs **~9%**, not 43%. Worth turning off for a batch proof
(`profile.no_gc()`), and a property of the prototype's Python objects rather than
of the protocol — but not a headline.

And the residual: **0.0–0.1% in BOTH modes** once a warm-up proof has run. The
10.9% was first-call cost — CUDA context setup, kernel JIT, the initial upload of
the Lagrange tensor — landing between stages on the first proof in a process. It
is not a missing stage; it is why measurements must discard a warm-up.

**Three plausible explanations, three refutations, one measurement that settled
it.** The instrumentation earned its place: each wrong story was caught in minutes
by a number rather than surviving into a projection. Any timing quoted from a
first proof in a process — which includes some earlier numbers here — carries that
warm-up.

### 4.9.8 The encoder was never the problem — the data movement was

Review pointed out that dense encoding kills the four-hour budget on its own:
`C_fresh * N * r_enc` with ~33.5e9 fresh message slots is 3.66 h even at the
isolated `gl_matmul` rate, and 286 h at the fitted end-to-end rate. Confirmed by
recomputation. Also confirmed: production geometry is 8x the last measured point,
not the 512x claimed here earlier (that was against the OLDEST point).

Two fixes, both of which were already available and neither of which is batching:

**1. NTT instead of a dense ELL x N product.** `prover/cuda_primitives` has
`ntt_forward/inverse_batched`. Our message is the polynomial's VALUES at the K-th
roots and our codeword lives on the coset `gamma*<w_N>`, so a plain
`rs_encode_rows` gives unrelated output -- which is what an earlier note here
mistook for an irreconcilable domain convention. With the two missing steps the
output is BIT-IDENTICAL:

```
values --iNTT_K--> coefficients --scale by gamma^i--> --pad to N--> NTT_N --> codeword
```

Verified bit-identical at four configurations, and at PRODUCTION geometry by
sampled columns in 9 s without building any matrix. The Lagrange matrix is gone:
537M cells, 13 minutes and 30 GB of host RAM, no longer needed.

**2. Do not ship the input through Python.** With the NTT encoder in place, the
stage timers said the encode itself was 1% of the path and the host->device copy
was 99%. Letting `Commit.from_messages` take an on-device tensor:

```
                              per 64 production rows      per row
Python lists                    103.3 ms (h2d 98.6)       1.614 ms
already on device                 0.3 ms (h2d  0.1)       0.005 ms     323x
```

The budget for one proof's fresh roots, end to end:

```
dense, isolated kernel rate        3.66 h
dense, fitted end-to-end rate    286 h
NTT, input via Python lists        1.56 h
NTT, input already on device        20 s      <- measured
```

**Lesson, and it is the same one three times over.** The dense product looked like
a protocol cost; it was an implementation choice. Then the NTT encode looked like
the win; the win was actually not moving the data. Each time the stage timers said
so within minutes and each time my first explanation was wrong. The remaining
consequence is structural rather than numerical: the forward pass still builds
Python objects, so anything it produces still pays the transfer. Tensorising
`semantics.py` removes that AND the reason a full Maverick layer cannot be
computed here.

### 4.9.9 The forward pass moved to the device — and the wall moved with it

`semantics.py` was the last thing building Python objects, and 4.9.8 ended by
naming it. It now has a second implementation, `forward_tensor`, gated on
equality rather than trusted: the trace is compared field by field, `check_trace`
(which shares no arithmetic with either path) accepts both, and the two traces
produce a **byte-identical proof**. `tests/test_semantics.py` pins all three;
`bench/validate_semantics.py` runs them across five shapes.

Both paths are fed the same weights through `LayerWeights`, drawn on the device
and converted down. Not the same seed — equal seeds only give equal weights while
both implementations consume the RNG stream identically, which is the assumption
under test.

Measured (`bench/semantics_ladder.py`, V100-SXM3-32GB, warm-up discarded):

```
     d   d_ff    S   E    python   tensor  speedup  peakGB  headroom
   128    256    8   4      0.22     0.02    10.8x    0.01     32 bits
   256    512   16   4      0.90     0.03    27.6x    0.03     24
   384    768   16   4      2.08     0.03    61.6x    0.07     22
   512   1024   32   4      7.52     0.06   127.2x    0.13     19
  1024   2048   32   8         —     0.07        —    0.88     16
  2048   4096   64   8         —     0.15        —    3.53     10
  4096   8192   64   8         —     0.24        —   13.90      6
```

**A correction to my own first reading of this ladder.** The first version drew
weights in Python and fed lists to `forward_tensor`, and reported 4.5 s at
d=1024 and 19.2 s at d=2048. Both were marshalling. With the weights already
resident the same layers take 0.07 s and 0.15 s — a 27× and 128× difference that
had nothing to do with the layer. This is 4.9.8's lesson for the fourth time: the
thing that looks like the cost is the data movement.

**What stops it now is memory, not time.** 13.9 GB at `d=4096, E=8`; `d=5120,
E=16` runs out on a 32 GB card. The resident term is the expert weights,
`3 * E * d * d_ff` int64 — which is the cost model's `Θ(E)` and `Θ(d²)` showing
up as an allocation instead of a duration. Values are bounded by
`table_size = 64` and are stored in 64 bits, so there is an easy 8× here
(uint8 storage, widened per expert on use) whenever memory becomes the binding
constraint.

**The 2^63 wall is real and it is close.** The tensor path carries the TRUE
integer in int64, not a field residue, because the Python path divides raw
accumulators by the scale and that is only the same operation while nothing has
wrapped. Headroom falls from 32 bits at `d=128` to **6 bits at `d=4096`**: the
toy's values grow by roughly `n_in` per matmul, since the rescale divides by
`scale` (64) while the accumulator grows by `n_in * scale`. The guard fires with
the node named and the arithmetic shown, and is exercised by a test — an
unexercised guard is not a guard:

```
scores: 73774343 * 73940156 * 4096 = 22343214818170912768 >= 2^63
```

This is a property of the toy semantics, not of the protocol. A faithful layer
keeps activations bounded because its weights are ~1/sqrt(d); here they are
uniform on `[0, 64)`. Fixing it means either drawing small weights or rescaling
by more than `scale` (which needs `table_bits >= scale_bits`, since the remainder
is range-checked against the table). Recorded, not fixed: it does not affect
relation counts, which is what the cost model consumes.

**A real bug, found on the way.** The batched NTT kernels launch one block-row
per message and a CUDA grid dimension is capped at 65535, so an encode past that
fails outright with `invalid configuration argument` — measured exactly: 65535
encodes, 65536 raises. LogUp commits one RS row per lookup QUERY, so the cap is
reached at about `d=256, S=8`, i.e. well inside the sizes this prototype is meant
to run; it is why the pipeline ladder below could not previously get past d=128.
`gpu.encode_batch_ntt` now chunks, and `tests/test_gpu.py` pins that a chunked
encode is bit-identical to an unchunked one.

That one RS row per query is itself the next dense-encoding artefact of the same
family as 4.9.8's Lagrange matrix: the message is a 2-tuple carried in an
ELL-wide row. It has not been touched, because packing many queries per row
changes what LogUp commits to and deserves its own step.

### 4.9.10 What this did NOT unlock

Stated plainly, because a fast forward pass invites the reading that a
production-width proof is now close. It is not. With the whole pipeline measured
at `ELL=1024, N=4096, q=16`:

```
     d    S   E   forward   to_python    prove   verify   forward share
   128    8   4      0.03        0.32    18.26    11.52          0.10%
   256    8   4      0.02        0.29    27.59    12.21          0.05%
```

The forward pass is now a rounding error in its own pipeline. Everything
downstream — `Enrollment.enroll`'s content key, the gate padding in
`relations.prove_batch`, `logup.LogUp`'s membership dict, `moe.source_records`
and its per-lane scan — still consumes Python objects, and `moe`'s lane loop is
`O(n_in² · S)` besides. Those are the next targets, and until they move, the
production-width number that matters is bounded by the prover, not by semantics.

### 4.10 Known remaining error

The residual under-prediction is largest on the **smallest** instances and shrinks
as they grow (0.80× → 0.99×). That is the signature of fixed per-call overhead
that is not modelled: object allocation, function-call cost, `dataclass`
construction. It matters least in the direction the model is used for
(extrapolation upward), so it is recorded rather than fixed. If it is ever
needed: model it as a constant per `Commit` and per sumcheck round.

---

## 5. Verifier

Modelled and validated on the same terms — `count_model.predict_verify`, exact
against the counters. The source document models no verification at all (its §9.4
GPU estimate is a conditional corollary explicitly outside the theorem).

Measured: **verification is a shrinking fraction of proving as the instance
grows**, 37% of prove time at S=2/d=4 down to 3% at S=8/d=32, because the verifier
rides the `q` opened columns and per-round interpolations, neither of which grows
with `d` the way encode and projection do.

---

## 6. Using the model on other hardware

The three levels separate cleanly: L1 and L2 are hardware-independent, L3 is one
rate card. To evaluate at 400B on a real card:

1. implement the seven loop bodies of §3 in the target kernel language;
2. measure them with the same method as `bench/kernels.py`;
3. build a `KernelRates` from the result;
4. evaluate L1 → L2 at Maverick geometry.

**Step 4 is blocked** until the MoE segmented path is wired into the prover (§2),
and step 1–2 need the GPU field kernels rather than the CPython loops. Until both
are done this model produces no 400B number, and any that appears elsewhere is not
from here.

---

## 7. Change log

| date | change | effect on L3 |
|---|---|---|
| 2026-08-05 | initial count model, `κ` removed; L1/L2 exact | 0.69×–2.48× |
| 2026-08-05 | split deferred from reduced multiplication (4.1) | 0.65×–2.12× |
| 2026-08-05 | unit changed to loop-body iterations (4.2) | — |
| 2026-08-05 | encode sparsity: `scan` + `mac` (4.3) | 0.47×–0.95× |
| 2026-08-05 | column transpose and hash-call overhead (4.4) | **0.80×–1.01×** |
| 2026-08-05 | GPU backend added; CPython card mispriced it | 13×–35× |
| 2026-08-05 | per-backend rates, dense GPU encode unit (4.5) | 0.35×–0.82× |
| 2026-08-05 | GPU→host transfer and marshalling term (4.5) | 0.47×–0.95× |
| 2026-08-05 | column packing priced per value; Lagrange build pre-warmed (4.6) | **0.68×–1.28×** |
| 2026-08-05 | MoE segmented path wired into the prover; L1/L2 exact again | — |
| 2026-08-05 | commitments and sumcheck moved onto the device (4.7) | prove 4.5x faster |
| 2026-08-05 | multi-row layout for n_in > ELL (4.8) | unblocks production widths |
| 2026-08-05 | **CORRECTION**: "linear in d" retracted; true asymptotics stated (4.9.1-3) | — |
| 2026-08-05 | `bench/row_capacity.py`: price per row of capacity, width and geometry swept separately | — |
| 2026-08-05 | NTT encoder + device-resident input (4.9.8) | fresh roots 3.66 h → 20 s |
| 2026-08-05 | tensor forward pass, gated on a byte-identical proof (4.9.9) | forward 10.8×–127× |
| 2026-08-05 | **BUG**: NTT encode failed past 65535 rows (CUDA grid cap); now chunked | unblocks d ≥ 256 |

Open items, in priority order:

1. **the prover, not the semantics** (4.9.10). The forward pass is now 0.05–0.10%
   of its own pipeline; `Enrollment.enroll`'s content key, `relations.prove_batch`
   padding, `logup.LogUp`'s membership dict and `moe.source_records` all still
   build Python objects, and the MoE lane loop is `O(n_in² · S)`;
2. LogUp commits one RS row per lookup QUERY — a 2-tuple carried in an ELL-wide
   row. Same family as the dense Lagrange matrix of 4.9.8, and now the largest
   single source of rows in a layer;
3. re-fit the three-term model: its `C·N` coefficient is 16× a direct measurement
   of the encode path, and the fit predates both the NTT encoder and the tensor
   input, so the discrepancy may no longer exist;
4. the ELL/N ladder — answered in 4.9.5, `ns/capacity` does NOT converge and
   correctly so; cost needs two scaling terms plus a constant;
5. fixed per-call overhead term (4.6);
6. local LF proof over the small roots, so `A`, `P` and the LogUp reciprocals stop
   travelling in the clear.
