# WC-LCRL-STC — status (the only authority on what is implemented)

Spec: [wc-lcrl-stc-spec.md](wc-lcrl-stc-spec.md). Branch: `wc-lcrl-stc`
(private repo, testing).

## Implemented and tested

- **S1 — coefficient-RS enrollment** (`prover/wc_bridge.py`): per-width
  packing into B-blocks, lam mask coefficients, coefficient-RS via zero-pad
  + batched forward NTT (probed: `rs_encode_rows` is evaluation-based and
  is the WRONG primitive for coefficient enrollment), column-major store,
  sha256 Merkle root, manifest digest. Zero-padded tails are bound.
- **S2 — late coefficient bridge**: shared per-width `rho` derived only
  from the post-R1 transcript state; `P_trace = W rho` and projected masks
  `pi` committed; `alpha`/`eta` derived only after that commitment;
  aggregated `c`, `v = c(eta)`; CPU python-int verifier (stand-in for the
  Rust twin) recomputes every coin and checks the bridge equation against
  Merkle-verified enrollment columns.
- **Tests** (`prover/tests/test_wc_bridge.py`, 8/8): domain/Horner
  cross-check; honest ACCEPT (+ independent `W rho` recomputation); exact
  `H_40 = C(16383,40)/C(32768,40) ≈ 8.859e-13`; negatives: tampered
  `P_trace`, internally consistent re-committed fake projection, tampered
  enrollment column, attacker-chosen `eta`, nonzero claim in the padded
  tail.
- **Bench** (`analysis/bench/wc_bench.py`): production geometry on a
  configurable block slice. V100, width 4096, one block (62.9M weights):
  enrollment 4.65 s (74 ns/param, one-time; dominated by the CPU sha256
  Merkle), bridge prove 0.07 s (1.13 ns/param), CPU verify 0.26 s, ACCEPT.

## NOT implemented (do not claim it)

- **§0.2/0.3 integration**: `P_trace` is not yet the semantic variable the
  terminal constraints consume — the bridge runs standalone, not inside
  the 5-round `prove_streaming` transcript. Wiring it in (and deleting the
  persistent-weight qIRS/qLin rows it replaces) is the actual speedup and
  the actual soundness surface.
- **§0.5 routed experts**: `compile_routed_projected` still proves
  `P = W rho` the old way; the bridge does not yet replace it.
- **§0.6 message cache**: the existing `LIGERO_WITNESS_CACHE`/`SPILL`
  caches compute_fn outputs only; canonical fresh message rows + pad
  metadata caching and the two-pass q/opening structure are not built.
- **§0.8 compiler chain invariant**: not enforced anywhere yet.
- **Rust twin** of `verify_bridge`; GPU (BLAKE3) enrollment Merkle;
  gain-block groups; enrollment-from-GGUF (current enrollment takes
  already-decoded tensors and does not verify them against a GGUF digest).

## Remote validation (vast, 2026-08-14)

Quadro RTX 8000 ($0.241/h, instance 47736613, 445 s wall ≈ $0.03,
destroyed after download). `wc_remote.sh`: gate_failures = 0
(wc_bridge 8/8 + fiat_shamir + routed_projected on the box). Production
geometry, growing slices:

| slice | weights | enroll ns/param | bridge prove ns/param |
|---|---|---|---|
| 4096 × 1 blk | 62.9M | 62.9 | 1.00 |
| 4096 × 4 blk | 251.7M | 50.6 | 0.41 |
| 4096+11008 × 2 blk | 464.0M | 49.3 | 0.38 |

Per-param cost still falling with slice size (fixed overhead amortizes),
so linear extrapolation from the largest slice is an upper bound.

## Modeled (not measured) production numbers

From the 464M-slice rates: 400B enrollment ≈ 19.7k s one-time (dominated
by the CPU sha256 Merkle — a GPU BLAKE3 accumulator cuts this hard),
bridge prove ≈ 150 s/proof. What it would replace in the current
admission model: persistent q_lin fold 3 625 s + persistent openings
1 812 s per proof. Both sides are models until the integrated pipeline
exists; treat the delta as a hypothesis, not a result.
