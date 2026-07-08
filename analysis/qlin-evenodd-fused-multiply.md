# q_lin reorg: even/odd fused multiply (the consume-`r_i` side)

Status: **design note + analysis, not yet implemented or measured**
(2026-06-23). Companion to `prover-optimization-investigation.md` (the measured
breakdown + the banked inverse-NTT fuse) and to `qlin-family-object-reorg.md`
(which produces `r_eval`, the input here). This note covers the **multiply**
half of the q_lin fold — `q_lin += Σ_i r_i·f_i` — and a restructuring that makes
half the product NTT-free and the other half a half-size transform.

All magnitudes below the measured ones are **analysis, flagged as such**; the
de-risking plan (§7) is a prototype + benchmark to replace them with numbers.

## TL;DR

- `poly_mul` is the dominant single op (~74 s) and is **NTT-bound**, hence
  memory-bandwidth-bound (measured ~223 GB/s practical, flat ns/elem). The only
  lever on a bandwidth-bound NTT is **fewer / smaller** NTTs — the banked
  inverse fuse already does fewer (one global inverse), −25 s.
- **This pushes the same lever onto the forward transforms.** The ζ-domain
  (where the expander output `r_eval` and the encoded witness already live) is
  exactly the **even** subset of the `2K`-th roots. So:
  - the **even half** of every row's product is `r_eval ⊙ (witness values)` — a
    pure elementwise multiply, **no transform**;
  - the **odd half** is a size-`K` *coset* NTT, not size-`2K`.
- Forward-transform count per row drops from ~5 size-`K`-equivalents to ~3
  (**~40%**), with the even half free. Composes with the inverse fuse
  (orthogonal: forward vs inverse).
- **No NTT rewrite.** The coset-NTT machinery already exists (the LDE/codeword
  path twists by `γ^l` and transforms on a coset). The work is restructuring the
  multiply + bit-exact validation, not a new transform kernel.
- This is also the natural home for "fuse the expander into the multiply": the
  expander output `r_eval` flows straight into the eval-domain product (it *is*
  the even-root half), so no separate `r_i` polynomial is materialized.

## 1. Why the NTT is the floor — and why "fewer NTTs" is the only lever

`q_lin = Σ_i r_i·f_i` is a sum of polynomial products of degree-`< K` factors.
Multiplying isn't elementwise — coefficient `k` of the product depends on all
pairs summing to `k` — so the affordable method is NTT convolution: evaluate
both factors at `2K` points (forward NTT), multiply pointwise, accumulate, one
inverse NTT. Schoolbook is `O(K²)` per row (`K ~ 2¹⁴`, millions of rows —
hopeless); the NTT is `O(K log K)`.

The NTT is a butterfly network — `log K` stages, each combining elements across
the whole row with a barrier between stages — so it is a *global* op with a
different parallel structure than the per-slot expander, and it is
bandwidth-bound. The investigation measured this directly and concluded the only
way to cut its cost is to **do fewer NTTs**. The inverse fuse banked one such
cut. This note finds another on the forward side.

You cannot avoid producing the product polynomial: the verifier evaluates
`q_lin` at the ζ- and η-points and reconstructs it from the opened columns
(`core.py:2540-2572`), so the prover must output its `2K` coefficients. Removing
the NTT entirely means not forming a product polynomial at all — the
sumcheck/GKR protocol change, out of scope here.

## 2. The structural fact: ζ-domain = even `2K`-th roots

The message domain is `ζ_c = ω_K^c`. The product is evaluated on the `2K`-th
roots `ω_{2K}^j`. Since `ω_{2K}^2 = ω_K`:

- **even** `j = 2c`: `ω_{2K}^{2c} = ω_K^c = ζ_c` — the ζ-domain.
- **odd** `j = 2c+1`: `ω_{2K}^{2c+1} = γ·ζ_c` with `γ = ω_{2K}` — a *coset* of the
  K-th roots.

Now evaluate the two factors on the even half:

- `R_i[2c] = r_i(ζ_c) = r_eval[c]` (padded with 0 for `c ≥ ELL`, per the interp
  definition in `_interpolate_to_kdeg`, `core.py:896`). **In hand** — it's the
  expander output; no transform.
- `F_i[2c] = f_i(ζ_c) =` the encoded row's ζ-values = the padded `message +
  ZK-slack` array — i.e. exactly the **input** to the encode iNTT
  (`encode_messages`). **In hand**; no transform.

Therefore the even half of every row's product is free:

```
P_i[2c] = R_i[2c] · F_i[2c] = r_eval[c] · (message+slack)[c]      # elementwise, no NTT
```

(For `c ≥ ELL` the padded `r_eval` is 0, so those positions vanish.)

The odd half needs the factors at the coset `γ·ζ_c` — a size-`K` coset NTT each:

- `F_i[odd] = cosetNTT_K(f_i_coeffs)` — **one** size-`K` transform. `f_i_coeffs`
  is already available from the encode, so this is the only f-side transform.
- `R_i[odd] = cosetNTT_K(iNTT_K(r_eval))` — two size-`K` transforms (we hold
  `r_eval` as values, so interpolate first).

The "coset twist by `γ^l`" is the same operation the codeword LDE already uses
(`_coset_powers_2k`, `core.py:544`; coset evaluation at `η = γ·ω_N^j`,
`core.py:440-442`, `:600-608`).

## 3. Transform-count comparison (per row)

Counting in size-`K`-equivalent transforms (`NTT_2K ≈ 2 × NTT_K`):

| factor | today (`poly_mul_batched`) | even/odd |
|---|---|---|
| `r_i` side | `iNTT_K` + `NTT_2K` ≈ 3 | even free + `iNTT_K` + `cosetNTT_K` ≈ 2 |
| `f_i` side | `NTT_2K` ≈ 2 | even free + `cosetNTT_K` ≈ 1 |
| **forward total** | **≈ 5** | **≈ 3** |
| inverse | per-row `iNTT_2K` (≈ 2), or 1 global with the fuse | 1 global `iNTT_2K` (fuse) |

So ~**40% off the forward transforms**, plus the even-half product becomes a
free elementwise multiply. The inverse side is unchanged (still need all `2K`
output coefficients) and is already amortized by the banked fuse.

Because the NTT is bandwidth-bound with flat ns/elem, fewer-and-smaller forward
transforms translate ~proportionally to less HBM traffic and less wall-clock —
the same mechanism the inverse fuse relied on. The even half moves *zero* NTT
bytes; the odd half moves half the per-transform bytes of a size-`2K` pass.

## 4. The fused pipeline

Per row (or per chunk, batched), composing with the banked eval-domain fuse:

1. **build `r_eval`** from the constraint-family objects (see the family-object
   note) — per-slot parallel.
2. **even half (free):** `acc_even += r_eval ⊙ (message+slack)` — elementwise,
   no transform.
3. **odd half:** `F_odd = cosetNTT_K(f_i_coeffs)`; `R_odd =
   cosetNTT_K(iNTT_K(r_eval))`; `acc_odd += R_odd ⊙ F_odd`.
4. at `finalize()`: interleave `acc_even` (even positions) and `acc_odd` (odd
   positions) into the `2K` eval-domain accumulator and do **one global**
   `iNTT_2K` → the `q_lin` coefficients (then `+ u_lin` blinding,
   `core.py:581-583`).

Keep the NTT as the tuned library call (`ntt_*_batched`), invoked at size `K` on
the odd halves — do **not** write a monolithic gather+NTT kernel; that fights the
tuned transform for the secondary (spill) saving. Fuse the cheap parallel parts
(build, even-half product, accumulate) around the library NTT.

## 5. Why the verifier's direct-Lagrange trick does NOT port here

The verifier's `compute_r_at_eta` (`core.py:1418-1430`) evaluates `r_i` directly
from the expander entries — `coef · r_lin[cid] · L_slot(η_j)` — with "no rTA
materialization, no iNTT." That's a win for the verifier because it needs `r_i`
at only a *few* points (`T_QUERIES`). The prover needs `r_i` at **all `2K`**
points; a dense Lagrange evaluation is `O(nnz · 2K)` per row, far above the NTT's
`O(K log K)`. So the prover keeps the NTT and exploits the even/odd structure
instead — the even half is free precisely because it's the *message* domain the
prover already holds, not because of a dense evaluation.

## 6. Relationship to the other levers

- **Inverse-NTT fuse** (banked, −25 s): folds the per-row inverse into one
  global `iNTT_2K`. Orthogonal — it's the inverse side; this is the forward
  side. They stack: forward ~3 transforms/row + one global inverse.
- **Family-object reorg** (the build-`r_i` note): produces `r_eval` cheaply and
  densely. This note consumes `r_eval` directly as the even-root half — the
  expander output never becomes a separate `r_i` polynomial. The two are the
  build and consume halves of the same q_lin rewrite.
- **NTT floor:** the odd-half transforms remain — irreducible for forming a true
  degree-`2K` product. Below that is only the protocol change.

## 7. Rejected: reuse the committed codeword

Tempting idea: the codeword already holds `f_i` at `N_LIG` points, so why
re-transform `f_i` for the multiply? It doesn't help, for two reasons:

- **Sound mode has no codeword in memory at R3** — it was committed and flushed
  in R1/R2, and R3 regenerates the witness. Holding every row's codeword from R1
  to R3 is `m_total × N_LIG` — prohibitive.
- **The codeword is on the wrong domain.** It evaluates `f_i` on a γ-*coset*
  (`η_j = γ·ω_N^j`), deliberately disjoint from the ζ-domain so the verifier can
  open points outside the message domain. But the even/odd win comes precisely
  from the multiply domain *containing* ζ — that's what makes `r_eval` and the
  witness free there. The coset shares nothing free with `r_eval` (which lives at
  ζ), and at `N_LIG = 2·(2K)` it is also 2× oversized for a degree-`<2K` product.
  A `2K`-coset subset works out transform-neutral vs even/odd while adding a
  strided gather and coset bookkeeping. So: no gain.

## 8. Effort, risk, and de-risking

- **Effort:** ~1–2 weeks. The throughput pieces reuse existing primitives (coset
  twist + `ntt_*_batched` at size `K`, the fuse accumulator, `challenge_at`).
  The time sink is **bit-exact validation**, not kernel writing.
- **Main risk — correctness, not speed:** the even/odd `2K`-root index map, the
  `γ^l` coset twist, and the padding must line up exactly with the current
  `poly_mul_batched`. Gate bit-exact (the q_lin coefficients must match
  byte-for-byte; a mismatch is a REJECT, never a forged accept).
- **One assumption to check:** that size-`K` coset NTTs stay as bandwidth-
  efficient per element as size-`2K`. The measured "flat ns/elem across sizes"
  suggests yes, but confirm in the prototype.

**De-risk before committing:** implement the even/odd multiply for a single
matmul's rows, gate bit-exact against `poly_mul_batched`, and measure NTT bytes
moved + wall-clock vs the current path. That replaces the estimate below with a
number.

## 9. Magnitude estimate (analysis — replace with measurement)

If the ~40% forward-transform cut holds, that is roughly ~20 s off the measured
74 s `poly_mul` (forward transforms are ~2/3 of its NTTs). Stacked with the
banked inverse fuse (−25 s), `poly_mul` could move from ~74 s toward ~30 s.
Combined with the family-object reorg recovering the `compile` + `expand`
overhead, this is the lever that could push past the doc's measured ~1.4×
engineering ceiling — but it is a lever that ceiling did not include, and it is
**unproven**. Treat these numbers as hypotheses for the prototype to confirm or
kill.
