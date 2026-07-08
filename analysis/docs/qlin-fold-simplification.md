# q_lin fold simplification — deduplicated challenge buffer + fused expand→write

**Status: SUPERSEDED (2026-07-04)** by `linear-fold-unification.md`, which
absorbed Piece 1 (the dedup challenge buffer → `ChalSource`) and implemented
Piece 2 as the descriptor-interpreter kernel (`k_interp_band` + four bespoke
irregular kernels) — see that doc's Phase-4 outcome. Kept for the measurements
(§1–3) and the "why not NTT-within-variables" analysis, which remain valid.

Original header: design for review (2026-06-28). Prover-side only; verifier
untouched; bit-exact; `LIGERO_QLIN_BANDCHK`-gated.

## Motivation

The linear-test fold (`q_lin`) *should* be NTT-bound. Right now it isn't: it is
dominated by the constraint-organizing ("expander") work, which is pure overhead,
not the transforms. All three causes live in the per-chunk path
(`_compute_q_lin_inner_chunk` → `_chunk_rTA_by_band` → `EXPANDERS` +
`_band_contribution`):

1. **Per-nonzero hashing.** `r_lin[cid] = challenge(s_comb, cid, "lin")` (BLAKE3)
   is derived once per *nonzero*. In a matmul, a B-matrix slot's cid is the output
   row `i_k`, shared across all `n` columns, so `challenge(i_k)` is recomputed `n`
   times — `O(nonzeros)` hashing where `O(constraints)` suffices. That is an
   `n`-fold (B) / `m`-fold (A) redundancy on the *dominant* matmul slots.
2. **Triple materialization.** Each band builds `(target, cid, coef)` tensors in
   HBM, then cats / argsorts / scatters them — roughly `24 bytes/nonzero` of
   intermediate write+read, plus the CSR build (`argsort`/`bincount`/`cumsum`).
3. **(residual) host overhead** — the Python per-chunk regroup (`qlin_group`) and
   the `_chunk_rTA_legacy` BANDCHK path.

On the GB10's LPDDR bandwidth (roughly an order of magnitude below an H100's HBM),
this overhead dominates the bandwidth-bound NTT. The goal of this change is to
remove (1) and (2) so the fold is left bandwidth-bound on the *inherent* data —
the `chunk_rTA` write and the NTT polys — which is what "NTT-bound" should mean.

## Scope and invariants

- **Prover-side only.** `verifier-rs` (`verify.rs::row_contrib`) is untouched — no
  cross-language churn, no re-port.
- **Keeps the current skeleton.** Streaming sweep, **per-row packets** (so
  cross-claim sharing is handled exactly as today — see *Why not "NTT within
  variables"*), `qlin_group`, the band classification (`_BAND_DISPATCH`:
  scatter / fan / spmv), and the iNTT / poly-mul / rowsum all stay.
- **Bit-exact.** Same field values, fewer ops. The challenge PRF is deterministic
  (computing `challenge(cid)` once equals computing it `n` times), and field add is
  commutative (reordering the accumulation cannot change a committed bit). Gated by
  `LIGERO_QLIN_BANDCHK` comparing the final `chunk_rTA` against the legacy path,
  and by GATE-E (prove → Rust ACCEPT).

## Piece 1 — deduplicated challenge buffer

Build the linear challenge vector once per chunk over the cids the chunk
references, then have every band *read* it instead of re-hashing.

```python
def q_lin_inner_chunk(chunk_lo, chunk_hi, row_polys, per_row_packets, s_comb, cfg):
    # (1) group packets by kind — UNCHANGED. per-row packets already accumulate
    #     every claim's contribution, so cross-claim sharing is handled here.
    by_kind = group_by_kind(per_row_packets, chunk_lo, chunk_hi)

    # (2) NEW: deduplicated challenge buffer — one BLAKE3 per DISTINCT cid.
    cid_lo, cid_hi = cid_span(by_kind)                      # near-contiguous: cids are assigned densely per claim
    R = challenge_range(s_comb, "lin", cid_lo, cid_hi)      # size O(#cids), reused across all nonzeros

    # (3) CHANGED: fused apply (Piece 2) — read R, write chunk_rTA, no triples.
    chunk_rTA = zeros((chunk_hi - chunk_lo) * cfg.ELL)
    for kind, band in by_kind.items():
        fused_apply[DISPATCH[kind]](kind, band, R, cid_lo, chunk_lo, chunk_rTA)

    # (4) NTT fold — UNCHANGED.
    rTA   = chunk_rTA.view(n_chunk, cfg.ELL)
    coeff = interp_to_kdeg(rTA, cfg)        # iNTT
    prods = poly_mul(coeff, row_polys)      # NTT
    return rowsum(prods)
```

`challenge_range` is a single batched-BLAKE3 kernel over a contiguous cid range;
cids are assigned densely per claim so `[cid_lo, cid_hi]` is essentially
contiguous and `R` is a plain dense tensor sized to the chunk's working set
(streaming-safe). PRF cost goes from `O(nonzeros)` to `O(distinct cids)`.

This piece alone is close to a signature change: pass `R` into
`gl_spmv_challenged` / the scatter and replace the inline `challenge_at(...)` with
a gather `R[cid - cid_lo]`. It carries most of the compute win at minimal risk.

## Piece 2 — fused expand→write (no triple materialization)

Replace each `EXPANDERS[kind]` (returns `(target, cid, coef)` tensors) +
`_band_contribution` (cats + scatters) with **three fused CUDA kernels keyed on
the existing dispatch class**, each computing its indices inline, reading `R`, and
accumulating into `chunk_rTA` — the triples never hit HBM:

```
fused_apply["scatter"](kind, band, R, cid_lo, chunk_lo, rTA):   # identity, freivalds A/B/C, rope-rot, …
    # one thread per slot:
    (target, cid, coef) = decode(kind, slot, band)             # closed-form index + small gather (rho/lam/cos…)
    rTA[target] = coef * R[cid - cid_lo]                        # 1:1 → direct write

fused_apply["fan"](...)    # softmax stride, transpose-fan: contiguous run → prefix-diff on R̄ (built once)
fused_apply["spmv"](...)   # causal-c2, embedding: ragged → segmented accumulate, reading R
```

The per-kind `decode` is exactly the index math the current expanders already
carry (`i_k = f//n` with the head / blinding-row / last-row-clip bookkeeping) —
relocated from torch into the kernel. (Note: there is no separate "outer-product"
kernel. The Freivalds case is `scatter` with `coef = -rho[j]` and `R[i_k]` served
from L2; the dedup buffer + cache give the reuse, not a 2D outer-product tiling.)

## Code accounting (add vs remove)

Net is roughly **neutral on total lines but fewer components**; it removes Python
and adds CUDA, and it conserves the irreducible per-kind index geometry (relocates
it rather than deleting it).

| | effect |
|---|---|
| **Remove (Python)** | `_chunk_rTA_by_band`, `_chunk_rTA_legacy`, `_band_contribution`, `_spmv_one_band`, `_band_key` glue; the triple materialization + CSR build; the 14 `_expand_*` functions in `packets.py` (their math moves into the kernels — `packets.py` roughly halves); eventually the BANDCHK legacy path |
| **Add (CUDA)** | `challenge_range` (Piece 1) + 3 fused family kernels (Piece 2) carrying the relocated decode |
| **Keep** | the 14 packet **dataclasses** (descriptors that parameterize the kernels); `qlin_group`; the streaming sweep; per-row packets; the iNTT/poly-mul/rowsum; the **entire verifier** |

Honest trade: Piece 2 swaps readable torch expanders for less-readable CUDA — a
maintenance cost — in exchange for the bandwidth/triple win. Piece 1 has no such
trade (pure simplification + the biggest single compute win).

## Why not "NTT within variables" (do NOT do this)

It is tempting to make chunks variable-aligned and run the NTT per variable, so the
index math becomes variable-relative (drop the `var_row_start` offset and the
partial-row/partial-triangle bookkeeping) and, intuitively, "one chunk = one
kind." **This does not work cleanly, because activations are shared across ops.**

The tape passes an op's output Variable directly as the next op's input (paper
§4.1: `norm → g → q`). So a matmul's input `A` *is* the previous op's output. The
matmul attaches its `LF2A` linear packets to `A`'s rows, while the producing op
attaches *its* packets to the same rows. Therefore a variable's rows carry linear
packets from **multiple claims** — `rᵀA` for a slot is inherently cross-claim
(this is exactly why the current design accumulates **per-row** packets).

Consequences:

1. **The regroup does not vanish.** A shared variable's rows still have several
   packet kinds (producer + consumer), so "one chunk = one op" is false.
2. **You still need the per-row cross-claim accumulation** — i.e. the thing
   `per_row_packets` already does; variable-aligned chunking does not remove it.
3. **It fragments the NTT batch.** The fixed-row chunk packs many small variables
   into one healthy iNTT/poly-mul batch; per-variable batches under-fill for small
   variables (the big matmuls are fine, but it is a regression for the tail).

So variable-aligned NTT buys only variable-relative indexing (modest) while adding
fragmentation and *not* removing the regroup or the cross-claim accumulation. Net:
not a simplification. **Keep the fixed-row-chunk NTT** — it is proven, has no
fragmentation, and handles cross-claim sharing via per-row packets for free. The
two wins above (dedup + fused) are independent of chunk shape and do not need it.

(The genuinely-clean per-variable form would have to *re-solve* cross-claim
sharing with a compile-time per-variable op-gather across claims. That is the
larger LinOp restructuring, deliberately out of scope here.)

## Migration (each step independently `BANDCHK`-gated + GATE-E)

1. **Piece 1 — dedup buffer.** Add `challenge_range`; make the scatter / spmv read
   `R[cid]` instead of hashing inline. Validate (`BANDCHK` final-`chunk_rTA`
   equality) and measure the hash-time drop. *Biggest win, smallest change.*
2. **Piece 2a — fuse the `scatter` family** (the matmul/identity bulk). Measure the
   triple-materialization traffic drop.
3. **Piece 2b — fuse `fan` and `spmv`** (softmax, causal, embedding).
4. **Optional adjunct — cache the `by_kind` / band layout across the 32 layers**
   (it is structural; only `R` and `rho/lam` rebind per layer). Cuts the residual
   `qlin_group` host overhead without any per-variable restructuring.
5. Once green at scale, delete `_chunk_rTA_legacy` and the BANDCHK path.

## Out of scope (deferred)

- The full per-variable / structured-operator (LinOp) rewrite — blocked on
  re-solving cross-claim sharing and on a verifier-side mirror; larger and riskier.
- Any verifier change (including applying the same dedup idea to `row_contrib`).
- The apply→iNTT fusion that avoids the `chunk_rTA` DRAM round-trip
  (shared-memory-bound; a follow-on once Pieces 1–2 land).
