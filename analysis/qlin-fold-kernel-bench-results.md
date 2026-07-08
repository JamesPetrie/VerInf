# q_lin fold kernel-mapping benchmark (Freivalds LF1B)

Measured on the Spark (GB10, sm_121) via `pipeline/qlin_fold_bench.py`. The
bench isolates the Freivalds LF1B family (weight B-side, the dominant witness)
on a synthetic pure-Freivalds chunk, and compares kernel mappings that build
`chunk_rTA` directly (no sort) against the current
`expand → argsort → CSR → gl_spmv_challenged` reference. Every variant is
checked **bit-exact** (`torch.equal`) against the reference.

## Result (k=4096, n=4096, H=1, n_chunk=256, ELL=8192 → 2,097,152 slots)

| variant | ms | ns/slot | GB/s | vs reference | bit-exact |
|---|---|---|---|---|---|
| reference (expand+sort+spmv) | 18.06 | 8.61 | 0.9 | 1.0× | — |
| per-slot | 1.44 | 0.68 | 11.7 | 12.6× | ✓ |
| warpcid (shared cid hash) | 0.74 | 0.35 | 22.7 | ~24× | ✓ |
| **precompute (hash ≤k cids, then gather)** | **0.12** | **0.059** | **135.6** | **~147×** | ✓ |
| per-row (hash reuse, 1 thread/row) | 4.05 | 1.93 | 4.1 | 4.5× | ✓ |
| multirow ×2 | 7.57 | 3.61 | 2.2 | 2.4× | ✓ |
| multirow ×4 | 15.08 | 7.19 | 1.1 | 1.2× | ✓ |
| multirow ×8 | 30.09 | 14.35 | 0.6 | 0.6× | ✓ |

**warpcid** = one thread per slot (full parallelism + coalescing), but each warp
computes its shared `cid` challenge once and `__shfl`-broadcasts it, with a
per-lane fallback for the rare warp straddling a `cid` boundary.

**precompute** = hash the ≤`k` distinct cids of the chunk once
(`challenge_range`), then a pure `gather + mul + write` kernel reads
`challenge[i_k]` — no BLAKE3 in the hot path. **Timed as the combined cost**
(per-cid hashing + both kernel launches + gather), so the hashing is not hidden.

## Findings

1. **Per-slot wins decisively** — 12.6× over the current path, just by removing
   the sort and the torch expand and computing each slot's `cid` in closed form.
   This is the first hard number behind "kill the sort."
2. **Per-row is 2.8× slower than per-slot, and multi-packet-per-thread is
   monotonically worse** (more rows/thread → slower). The hash-reuse that per-row
   buys is swamped by the loss of parallelism: at `n_chunk=256`, per-row launches
   only 256 threads (multirow even fewer), nowhere near saturating the GB10,
   while per-slot launches one thread per slot (2.1 M). **Parallelism dominates
   hash-reuse here.** So "process several packets per thread" is counterproductive.
3. **Per-slot is compute-bound, not bandwidth-bound** — 11.7 GB/s is far below
   the GB10's ~223 GB/s practical bandwidth, so the BLAKE3 challenge hash is the
   limiter. And with `n=4096`, each `cid` is shared by 4096 consecutive slots, so
   per-slot re-hashes each `cid` ~4096× redundantly. Hash-reuse is therefore
   still on the table — but only if it keeps full per-slot parallelism and
   coalescing (one-thread-per-row does not, see (2)).
4. **The shared cid hash (warpcid) confirms it and wins: ~2× over per-slot**
   (0.36 vs 0.68 ns/slot), bit-exact. Hashing the shared `cid` once per warp
   removes 31/32 of the redundant hashing. **But 22 GB/s is still ≪ 223**, so it
   is *still* not bandwidth-bound: a `cid` spans 4096 slots = 128 warps, so the
   cross-warp redundancy (each cid still hashed ~128× across warps) remains.
   Next lever: **precompute the per-cid challenges once** for the chunk (≤ k
   distinct cids) and make the build kernel a pure gather+mul+write with *no*
   BLAKE3 in the hot path.
5. **precompute confirms it and is the floor: ~12× over per-slot, bandwidth-
   bound.** 0.059 ns/slot at 135.6 GB/s (≈270 GB/s effective counting the
   `neg_rho` read) — essentially the GB10's memory ceiling. Crucially this is the
   **combined** time: the per-cid hashing (only `k`=4096 hashes) and the second
   kernel launch are *inside* the measurement and still negligible, because
   #distinct-cids (≤k) ≪ #slots (2M). So once the sort/expand are gone, the
   irreducible build cost is just memory traffic.

   Caveat — this win is large precisely because the reuse is large (wide `n`:
   one cid spans 4096 slots). It is the right model for the **weight Freivalds**
   families that dominate the witness. For **narrow** families and especially
   **1:1 identity** families (#cids = #slots, no reuse), precompute degenerates
   to "hash every slot" and won't beat per-slot — the shape sweep should confirm
   where the crossover is. Plan: precompute for the high-reuse families,
   warpcid/per-slot for the rest, dispatched by the family's cid-span.

## Other families (same harness, `--family identity|stride`)

To check the dispatch rule, the two non-Freivalds regimes — **Identity**
(zero reuse: one cid per slot, the "activation copy" family) and **stride
fan-out** (anti-reuse: each slot sums `stride` distinct cids, e.g. softmax
`z = c2 − x`). Same chunk size (2,097,152 slots), all bit-exact.

**Identity** (`L2_IdentityScalar`):

| variant | ns/slot | GB/s | note |
|---|---|---|---|
| reference | 3.44 | 2.3 | |
| **per-slot** | **0.70** | **11.5** | winner — same as Freivalds per-slot (1 hash/slot) |
| precompute | 0.76 | 10.5 | *slightly worse* — #cids = #slots, so it hashes everything once anyway, plus a second pass |

**Stride fan-out** (`L2_StrideOneToManyScalar`, stride=16):

| variant | ns/slot | GB/s | note |
|---|---|---|---|
| reference | 47.99 | 0.2 | sort over 16× the nonzeros |
| **per-slot** | **9.89** | **0.8** | winner; ≈16× the 1-hash/slot cost (precompute N/A — same cids, no reuse) |

## Dispatch rule (data-backed)

| family | reuse (slots per cid) | best variant | ns/slot | bound |
|---|---|---|---|---|
| Freivalds (wide `n`) | high (`n`:1) | **precompute** | 0.059 | bandwidth |
| Identity | none (1:1) | **per-slot** | 0.70 | hash |
| Stride fan-out | anti (1:`stride`) | **per-slot** | ~0.70·stride | hash |

**precompute wins iff a cid is shared by many slots.** Zero/anti-reuse families
are irreducibly *hash-bound* (every slot needs its own challenge), so per-slot is
optimal and ~10–140× costlier per slot than the bandwidth-bound Freivalds. So the
real kernel dispatches by the family's cid-span: precompute for the wide weight
Freivalds (the witness bulk), per-slot for identity / fan-out / 1:1 families.
warpcid only helps the in-between (cid shared within a warp but not chunk-wide).

## Caveats / next

- Single shape and `n_chunk`. The per-row penalty is partly the small `n_chunk`
  (only 256 threads); sweep `n_chunk` and matrix shapes (incl. `transpose_b`,
  `H>1`) before generalizing — though per-slot's full parallelism makes it the
  robust base regardless.
- Next variant to test: **warp-collaborative cid-run hashing** (the one that
  could beat per-slot's 0.68 ns/slot by removing the ~4096× redundant hashing).
- Then extend beyond Freivalds to the other family kinds (identity, fan-out).
