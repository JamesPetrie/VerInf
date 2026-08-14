# Metric ledger — best measured prove() times

The standing baselines the autoresearch loop must not regress and is trying to
beat. All on Tesla V100-SXM3-32GB, CFG ELL=512/K_DEG=1024/N_LIG=4096/T=16
unless noted, with the 4 core.py per-row-loop optimizations from the
formula-vs-reality session already applied (_iter_message_chunks, _band_key,
table_settlement_compile/_index_bands _PktRange, _build_row_map/_RowMap,
_gather_rows). Update when a change is APPLIED and ACCEPT-verified.

| config (d, d_ff, d_h, SEQ, layers) | m_total | best prove_s | witness_s share | source |
|---|---|---|---|---|
| toy: 16, 32, 8, 4, 1 | 329,579 | 3.1 | negligible | cache no-ops (witness tiny) |
| 512, 1024, 64, 256, 4 | 604,267 | **12.12** (was 17.84) | 21% (was 46%) | iter3 sweep, cache on |
| 512, 1536, 64, 384, 4 | 864,951 | **21.39** (was 36.17) | 17% (was 59%) | iter2 witness cache |
| 512, 2048, 64, 512, 4 | 1,210,627 | **24.44** (was 62.13) | 7% (was 65%) | iter8 gpu softmax |
| 512, 1536, 64, 768, 4 | 1,760,865 | 75.39 (cache; gpu-sm not yet remeasured) | 55% | iter3 sweep, cache on |
| 512, 2048, 64, 1024, 4 | 2,747,117 | **59.92** (was 218.28) | 6% (was 75%) | iter8 gpu softmax |

**iter2 (APPLIED): witness cache** — reuse softmax+silu compute_fn outputs across
the 4 Fiat-Shamir sweeps (prover/core.py, LIGERO_WITNESS_CACHE=1 default). On
d512,ff1536,seq384,L4: 36.17s → 21.39s (**42.7% faster**, 1.74×), proof
byte-identical + Rust ACCEPT.

**iter3 (SWEEP, measured OFF vs ON across the range)** — the cache win holds
34–48% at seq256/512/768, but **collapses to 10.8% at seq1024** because the
element cap `_WITNESS_CACHE_MAX_ELEMS=2e8` engages and gates most caching out
there (witness only −16% vs −72% elsewhere). Peak mem at seq1024 is only 5.7 GB.
CONFIRMED by cap A/B: lifting the cap to 5e9 gives seq1024 prove **214.89→98.10s
(−54.3%)**, witness 160.82→43.48s (−73%, matching the other configs), peak just
8.0 GB / 32. So the current seq1024 ledger row (191.60, cap-limited) is beatable
to ~98s once the cap is raised (next iter, ACCEPT-gated). Cache-ON is safe here:
both cached and recomputed paths were proven byte-identical in iter2, so the cap
only trades speed for memory, never soundness.

**iter4 (APPLIED): mem-gated witness cache** — replaced the fixed 2e8-element cap
with a budget = 25% of *free* GPU memory at cache-init (core.py, env
LIGERO_WITNESS_CACHE_MEM_FRACTION, default 0.25). Auto-scales to the card;
degrades to recompute only when memory is genuinely tight. On seq1024:
200.51→97.30s (**51.5% faster, 2.06×**), witness 146.57→42.94s (−71%), peak just
8.0 GB / 32. Byte-identical proof OFF vs ON (validate_witness_cache.py), Rust
verify_proof=ACCEPT end-to-end on the toy transformer (accept_toy_cache.py), and
71 tests green (test_reveal 2, test_persistent_weights 3, test_claims 21,
test_gather_rows 9, test_iter_message_chunks 15, test_pkt_range 13, test_row_map
8). Soundness argument: which ops get cached now depends on free memory, but
cached==recomputed byte-for-byte, so this changes ONLY timing. seq256/512/768
already fully cached under the old cap, so their rows are unchanged.

## Known decomposition (medium scale, LIGERO_PHASE_TIMING)

At d512,ff2048,seq1024,L4 (218s total): witness 75%, encode 5.7%, quad 6.2%,
fold_qlin 3.2%, compile 2.9%, merkle 1.2%, rest <2% each.

- **witness** is the dominant term and grows with scale. It is the forward-pass
  compute (rmsnorm, matmul, RoPE, softmax's `s1_at` causal-range binary
  search) run inside each of the 4 Fiat-Shamir sweeps. Prime target.
- The two-term formula `prove_s ≈ identity_floor + witness_recompute` matches
  actual to ~1.3-1.9× (gap shrinks with scale), so witness_recompute is
  measured, not assumed. See analysis/toy-transformer-prove-time-formula.md.

## Standing invariant

Any applied change MUST keep the standalone Rust `verify_proof` at ACCEPT on a
freshly dumped proof. A prover that is faster but changes/breaks the proof is
not an improvement — it is a regression, revert it.

**iter8 (APPLIED): GPU softmax witness** — ported _softmax_witness_vec (numpy/CPU)
to torch int64 on CUDA (compute_fns.py `_softmax_witness_gpu`, env LIGERO_GPU_SOFTMAX,
default on; drops the .cpu() round-trip, keeps x on device). BIT-EXACT to numpy
(byte-identical proof numpy vs gpu + Rust ACCEPT, validate_gpu_softmax.py; branch
coverage: test_claims non-saturate, test_rescale rescale ACCEPT, test_reveal Rust
verify). Isolated call 962ms->10ms (95x). Prove A/B (cache on both sides):
seq512 33.23->24.44 (26.4%, witness 10.35->1.74), seq1024 100.77->59.92 (40.5%,
witness 44.05->3.51, peak 8.0GB). Cumulative seq1024: 218.28 (pre-cache) -> 97.30
(cache) -> 59.92 = 3.64x. seq256/384/768 also benefit (softmax O(SEQ^2) share) but
were not re-measured this iter.
