# Prover autoresearch journal (append-only)

Newest entries at the bottom. One block per iteration, per the
prover-autoresearch skill §5. Read before picking a target.

---

## iter 2026-07-22 (seed) — target: infrastructure + baselines
Predicted (theory): n/a
Tried: Set up the autoresearch loop — skill (.claude/skills/prover-autoresearch),
metric ledger, this journal, and the tools it depends on (cost_calculator,
run_log/show_runs, coset_ntt_ab as the rigor template). Backfilled 36 prior
runs into prove_runs.jsonl.
Measured: baselines recorded in metric_ledger.md (toy 3.1s → medium 218s).
Outcome: prototype-only (infra)
ACCEPT: n/a (no real-prover change)
Why: This is the starting state, not an optimization. The four per-row-loop
fixes from the formula-vs-reality session are already applied and ACCEPT-
verified; witness_recompute (the 4× forward pass) is now the dominant term
(75% at medium scale) and is the standing frontier.
Next angle: Attack the 4× witness recompute. First measure whether the witness
forward pass is genuinely recomputed identically across all 4 Fiat-Shamir
rounds (grep the round loop in core.py), then decide between (a) caching the
deterministic compute_fn outputs across rounds if they fit in memory at test
scale, vs (b) the notebook's claim-streaming (witness_passes→1). Whichever,
the Rust verifier must stay ACCEPT.

---

## iter 2026-07-22 — target: 4x witness recompute (confirm + quantify)
Predicted (theory): notebook says the witness is regenerated once per Fiat-Shamir
sweep = 4x; at our scale it fits in memory (peak ~5GB) so the "can't store it"
reason for recompute doesn't apply, making it cacheable.
Tried: Measured, on d512,ff1536,seq384,L4: (A) one forward pass via
run_engine_pass = 5.19s; (B) the prove witness-phase bucket (sum over 4 sweeps,
LIGERO_PHASE_TIMING) = 21.90s.
Measured: ratio B/A = 4.22x — confirms ~4x recompute (the >4 is aux/compile
leakage into the bucket). Witness is 59% of the 37.3s prove here. Cache ceiling
= save 3 of 4 passes = 15.57s = ~42% of prove time.
Outcome: partial (lever confirmed + quantified; not yet implemented)
ACCEPT: n/a (no real-prover change; measurement only). Logged kind=witness_probe.
Why: The 4 sweeps live in prove_streaming (core.py ~2717-2733): R1 commit,
R2 aux-commit, R3 q-poly+p_0, R4 columns. Each calls _stream_sweep, which
recomputes outs = COMPUTE_FNS[type(claim)](claim, input_data) in the `witness`
phase. Those outs are DETERMINISTIC functions of committed inputs — byte-
identical every sweep — so reusing them across sweeps yields a byte-identical
witness -> byte-identical proof -> ACCEPT unchanged. Only two risks: (1) MEMORY
— the sweep deliberately frees per-op (live.pop) to keep peak O(one op); caching
ALL outs would break that bound and could OOM at larger scale; (2) must cache
ONLY the challenge-independent compute_fn outputs, NOT the aux/Freivalds
witnesses (those depend on ch0 and belong to the `aux` phase).
Next angle (iter 2): implement a MEMORY-GATED witness cache. Design sketch:
  - Cache the EXPENSIVE-to-recompute, SMALL-in-memory outputs first: softmax
    (s1_at, the CPU-bound O(SEQ^2) binary search — the single biggest witness
    cost) and silu. Skip caching the big lookup-table mults (memory-heavy,
    cheap to recompute) so peak memory stays bounded.
  - Plumb a per-claim cache dict into _stream_sweep keyed by claim identity;
    populate on sweep 1 (want_aux=False), reuse on sweeps 2-4. Gate on a
    memory budget / config flag so it no-ops when the witness wouldn't fit.
  - VALIDATE: new diff test (cached vs uncached outs bit-exact), full fast
    suite, and Rust verify_proof = ACCEPT on the SMALL toy config. Only apply
    if all green. A/B the medium-config prove_s; expect up to ~42% off if the
    softmax share dominates the recompute, less if the memory-heavy ops do.
  - Cross-check against cost_calculator: witness_recompute term should drop
    toward 1x its current value on the affected configs.

---

## iter 2026-07-22 — target: 4x witness recompute (IMPLEMENTED — memory-gated witness cache)
Predicted (theory): iter1 measured the 4x recompute; cache ceiling ~42% of prove
if the witness were reused across sweeps. Caching only softmax+silu (the
expensive, small-in-memory compute_fns) should capture most of it.
Tried: Added a memory-gated witness cache in prover/core.py: _stream_sweep now
takes witness_cache; prove_streaming creates one dict shared across the 4 sweeps.
On sweep 1 the deterministic softmax/silu compute_fn outputs are cloned into the
cache (gated by _WITNESS_CACHE_TYPES={SoftmaxClaim,SiluClaim} and a cumulative-
element cap _WITNESS_CACHE_MAX_ELEMS=2e8); sweeps 2-4 reuse a clone instead of
recomputing. Only challenge-independent compute_fn outputs cached; aux/Freivalds
untouched. Env gate LIGERO_WITNESS_CACHE=0 disables.
Measured: d512,ff1536,seq384,L4, median of 2 reps: prove 37.31s -> 21.39s =
42.7% faster (1.74x). Witness phase 22.06s -> 6.11s (72% of the witness term
eliminated). Logged kind=witness_cache_ab.
Outcome: APPLIED
ACCEPT: yes. Proof is BYTE-IDENTICAL cache off vs on (validate_witness_cache.py:
root_p1/root_p2/q_irs/q_lin/p_0/opened_p1/opened_p2 all equal on the same model),
and Rust verify_proof = ACCEPT on a cache-on toy proof. Regression green:
test_reveal 2, test_persistent_weights 3 (both use the streaming prover ->
exercise the cache), test_claims 21, + the 4 diff-test files (45).
Why it works: the 4 Fiat-Shamir sweeps recompute a DETERMINISTIC forward pass;
reusing the softmax/silu outputs (byte-identical every sweep) changes nothing in
the committed witness -> identical proof. softmax (s1_at binary search) + silu
dominate the witness term far more than the matmuls, so caching just those two
captured ~all of the ceiling. Memory stays bounded (their outputs are small;
the cap degrades to recompute at large scale).
Caveat / honest note: measured on ONE medium config, 2 reps. The saving scales
with softmax's share of the witness, which grows with SEQ (O(SEQ^2)); at small
SEQ or matmul-heavy shapes the win will be smaller. Should confirm across the
ledger's config range and check the memory cap actually engages at the largest.
Next angle (iter 3): (a) sweep the cache win across the ledger configs (toy ->
seq1024) to map where it helps and confirm no regression / no OOM at the top;
(b) formalize validate_witness_cache.py as prover/tests/test_witness_cache.py;
(c) THEN consider extending the cache to matmul/rmsnorm/rope outputs under a
tighter memory gate for the remaining witness recompute, OR attack the next
term (encode, ~11%). Cross-check the new prove_s against cost_calculator's
witness_recompute term (should drop toward 1x).

---

## iter 2026-07-22 — target: sweep witness cache across ledger range + OOM/cap check
Predicted (theory): iter2 cache removes ~72% of the witness term; witness is
46-75% of prove and grows with SEQ (softmax O(SEQ^2)), so the cache win should
GROW with SEQ across the ledger, up to ~50% at seq1024. Memory expected fine
(peak ~5-6GB, cap 2e8 elems ~1.6GB).
Tried: sweep_witness_cache.py — same-process OFF vs ON, fixed seed, on toy,
seq256, seq512, seq768, seq1024 (CUDA-synced prove wall-clock + witness phase +
torch peak alloc). Then confirm_cap_seq1024.py — seq1024 cache ON with the
element cap at default 2e8 vs lifted 5e9, to test the cap hypothesis.
Measured (OFF -> ON prove_s, % faster, witness OFF->ON):
  toy    3.71->3.58   3.4%   wit 0.05->0.04  (cache ~no-ops, witness negligible)
  seq256 18.39->12.12 34.1%  wit 8.83->2.55  peak 4.1GB
  seq512 64.27->33.58 47.8%  wit 41.78->11.09 peak 5.2GB
  seq768 129.93->75.39 42.0% wit 95.09->41.42 peak 5.5GB
  seq1024 214.89->191.60 10.8% wit 160.82->135.26 peak 5.7GB   <-- ANOMALY
  Cap A/B @seq1024 cache ON: cap=2e8 prove 190.07s wit 132.90s peak 5.70GB;
                             cap=5e9 prove  98.10s wit  43.48s peak 8.00GB.
Outcome: partial — sweep done + a strong, unexpected finding + root cause nailed.
No prover-code change applied this iter.
ACCEPT: n/a (measurement only; no real-prover change). Logged kind=
witness_cache_sweep (5 rows) + witness_cache_cap_confirm (2 rows).
Why measured != predicted: at seq1024 the win did NOT grow to ~50% as theory
said — it COLLAPSED to 10.8%. Cause: _WITNESS_CACHE_MAX_ELEMS=2e8 (core.py
~2320) is a CUMULATIVE per-sweep element budget; softmax+silu outputs across
4 layers at seq1024 exceed it partway through, so sweep-1 stops caching and the
later layers' witness is recomputed every sweep (witness only -16% vs -72%
elsewhere). Confirmed decisively by lifting the cap to 5e9: prove 214.89->98.10
(-54.3%), witness -73% matching seq512/768, peak only 8.0GB / 32GB. The 2e8 cap
(~1.6GB claimed) is far too conservative on a 32GB V100 and is the single thing
blocking a 2.2x speedup at the top scale. Also updated the ledger's four stale
"pre-cache" rows with real cache-ON numbers, and fixed the iter2 seq384 row's
swapped m_total/prove_s columns.
Next angle (iter 4): RAISE / re-gate the cap, then apply with the ACCEPT gate.
Design choice to make: (a) simplest — bump _WITNESS_CACHE_MAX_ELEMS default to
~2e9 (headroom: seq1024 peak was 8GB, ~24GB free); or (b) better — gate on
ACTUAL free GPU memory (torch.cuda.mem_get_info) with a safety margin instead of
a fixed element count, so it auto-scales and still degrades to recompute only
when memory is genuinely tight. Prefer (b) but (a) captures the win immediately.
VALIDATE before applying: cap only chooses cache-vs-recompute and both paths are
already byte-identical (iter2), so re-run validate_witness_cache.py (proof
byte-identical cap-low vs cap-high) + Rust verify_proof=ACCEPT on the SMALL toy
+ fast suite, then A/B seq1024 to confirm ~98s and log. Also worth: formalize
validate_witness_cache.py as prover/tests/test_witness_cache.py (still open from
iter2). Watch: cap raise must NOT push peak past a safe fraction of 32GB at the
largest ledger config — measure peak, don't assume.

---

## iter 2026-07-23 — target: apply the cap fix (mem-gated witness cache), ACCEPT-gated
Predicted (theory): iter3 measured that lifting the element cap at seq1024 gives
prove 214.89->98.10s (-54.3%), witness -73%, peak 8.0GB. A budget sized to free
GPU memory (instead of the fixed 2e8) should recover that full win at the top
scale while auto-degrading on smaller cards; smaller configs (seq256/512/768)
already fully cached under the old cap, so no change expected there.
Tried: prover/core.py — replaced _WITNESS_CACHE_MAX_ELEMS=2e8 (fixed) with a
byte budget = _WITNESS_CACHE_MEM_FRACTION (default 0.25) * free GPU mem at
cache-init (torch.cuda.mem_get_info), computed once in prove_streaming and stored
as witness_cache['_budget_bytes']; the per-op gate now compares cumulative cached
BYTES to the budget (legacy elem cap kept as optional hard ceiling, default off).
Env LIGERO_WITNESS_CACHE_MEM_FRACTION tunes it.
Measured: seq1024 same-process OFF vs ON (ab_memgated_seq1024.py, CUDA-synced):
OFF 200.51s -> ON 97.30s = 51.5% faster (2.06x); witness 146.57 -> 42.94s (-71%);
peak 8.00GB / 32. Matches the iter3 lifted-cap confirmation (98.10s) -> the new
default fully caches at seq1024. Logged kind=witness_cache_memgated_ab.
Outcome: APPLIED
ACCEPT: yes. (1) validate_witness_cache.py: proof BYTE-IDENTICAL cache off vs on
(root_p1/root_p2/q_irs/q_lin/p_0/opened_p1/opened_p2 all equal) under the new
code. (2) accept_toy_cache.py: Rust verify_proof=ACCEPT end-to-end on the toy
transformer (1 layer -> exercises the cached softmax+silu ops) with cache ON.
(3) Fast suite green: test_reveal 2 (uses rust_verify_tape -> real Rust TCB),
test_persistent_weights 3, test_claims 21, test_gather_rows 9,
test_iter_message_chunks 15, test_pkt_range 13, test_row_map 8 = 71 total.
Why it works / soundness: the cap only decides cache-vs-recompute, and both paths
produce byte-identical deterministic softmax/silu outputs (proven iter2), so
making the cached SET depend on free memory changes ONLY timing, never the proof.
The old 2e8 cap (~1.6GB) was ~3x too small for the seq1024 working set's cache
need (~2.3GB) and stopped caching mid-sweep-1; 25% of a 32GB card (~7-8GB budget)
covers it with headroom, and peak stayed 8.0GB. Ledger updated: seq1024
191.60->97.30. Net win of this + iter2: seq1024 218.28 (pre-cache) -> 97.30 = 2.24x.
Next angle (iter 5): the remaining witness at seq1024 is now 42.94s = 44% of
prove (down from 75%) but still the biggest bucket. Two levers: (a) extend the
cache to the OTHER deterministic compute_fns (matmul/rmsnorm/rope outputs) under
the same mem budget — they're memory-heavier but the budget now has room; A/B
whether caching them beats recomputing (matmul may be cheap enough that the clone
+ memory cost is a wash — MEASURE). (b) Attack the next term now that witness is
smaller: quad (14.1%) and encode (13.2%) are the next buckets at seq1024
(from the cap-confirm phase dump). Also still open from iter2: formalize
validate_witness_cache.py as prover/tests/test_witness_cache.py so the byte-
identical gate runs in the suite. Also worth: pick the mem-fraction more
carefully / measure peak at a config bigger than seq1024 to be sure 0.25 leaves
enough working-set headroom before recommending higher.

---

## iter 2026-07-23 — target: extend witness cache to matmul/rmsnorm/rope? (MEASURE first)
Predicted (theory): witness is still 44% of prove at seq1024 after caching
softmax+silu. iter4's "next angle (a)" asked whether caching the OTHER
deterministic compute_fns (matmul/rmsnorm/rope) captures more of it — with the
explicit caveat that matmul may be too cheap-to-recompute for caching to pay.
Tried: witness_type_probe.py — wrapped every COMPUTE_FN with a CUDA-synced
per-type timer + bytes/call, ran a cache-ON prove at d512,ff2048,seq512,L4, so
the remaining recompute is decomposed by op type (softmax/silu show ~1x call
count = already cached; others recompute 4x).
Measured (cache ON, seq512, witness compute total 10.05s):
  SoftmaxClaim  8.58s  4 calls (1x/layer)  168 MB/call   <- 85% of remaining witness
  SiluClaim     0.48s  4 calls             143 MB/call   (already cached)
  RoPEClaim     0.48s  32 calls (4x)         8 MB/call
  MatmulClaim   0.25s  148 calls (4x)       67 MB/call
  RmsNormClaim  0.24s  36 calls (4x)        11 MB/call
  Hadamard/Embed/Add  <0.03s each
Outcome: DEAD-END for lever (a) (cache extension). No code change.
ACCEPT: n/a (measurement only). Logged kind=witness_type_probe.
Why: the remaining witness is NOT spread across cacheable ops — it is the
UNAVOIDABLE ONE-TIME softmax forward pass. Softmax is already cached (4 calls =
once per layer, not 16), and that single computation costs 8.58s = 85% of the
remaining witness compute; caching cannot remove a cost you must pay at least
once. Extending the cache to matmul/rmsnorm/rope would save only ~3/4 of their
tiny totals = ~0.7s combined, while matmul alone would add 67MB/call x 148 calls
(GBs) of memory pressure -> confirms iter1's "matmul: memory-heavy, cheap to
recompute". So cache-extension is not worth it. The witness cache lever is now
EXHAUSTED (softmax+silu was the whole win; iter2+iter4 already captured it:
seq1024 218.28 -> 97.30 = 2.24x).
Next angle (iter 5): the remaining witness IS softmax's own compute. Root cause:
softmax_compute (compute_fns.py) offloads to _softmax_witness_vec which runs on
CPU numpy (x.cpu().numpy(); numpy exp tables T_A/T_B; O(SEQ^2) causal bracket /
s1_at binary search) and copies back. Two honest levers, pick ONE next iter:
  (A) Port _softmax_witness_vec's hot loop to GPU (or shrink the CPU<->GPU
      copies / vectorize the causal bracket). Soundness-critical: the witness
      (c2/z/y_A/y_B/s1/s2/r_lo/r_hi...) must be BIT-EXACT vs the numpy path ->
      strong diff test + Rust ACCEPT. This is the single biggest remaining
      prove-time lever at large SEQ (softmax is O(SEQ^2), grows fastest).
      Measure the CPU vs GPU split first (is it the numpy math or the .cpu()
      transfer that dominates the 2.15s/call?) before porting.
  (B) If (A) looks too invasive, switch to the next PHASE buckets at seq1024:
      quad 14.1% and encode 13.2% (from iter3 cap-confirm phase dump) are now
      co-equal with witness's non-softmax remainder.
Also still open: formalize validate_witness_cache.py as
prover/tests/test_witness_cache.py (the byte-identical gate should live in the
suite, since iter4 widened what the cache does).

---

## iter 2026-07-23 — target: profile softmax _softmax_witness_vec (CPU-math vs transfer) before porting
Predicted (theory): iter5 pinned the remaining witness on softmax's one-time
compute (~2.15s/call, 85% of remaining witness). Before a GPU port I must know
WHERE that goes — CPU numpy math or the host<->device transfer — and which
section, so the port targets the right work and the win is predictable.
Tried: softmax_internal_profile.py — captured the real kwargs of the largest
softmax call from a seq512 prove, then replayed a SECTION-TIMED copy of the body
and asserted it byte-identical to the original (so timings are trustworthy and
the copy is a validated reference for the port). Timed _orig and the copy
identically with warm-up (first version showed a spurious 2220ms cold number;
warm-up reconciled it).
Measured (B=4096, M=512, saturate+causal; median of 7, WARM):
  _orig full call        818.9 ms   (copy 804.9 ms, sections cover 99%)
  binary search          526.7 ms   65.4%  (18x s1_at, each a few (B,M)=2M numpy ops)
  output+saturate        205.7 ms   25.6%  (z_low/z_high decomp, mux, _to_field_np)
  ycell+s2                37.7 ms    4.7%
  input+mask+minmax       30.5 ms    3.8%
  In-prove call ~2150 ms (iter5: 8.58s/4) = ~2.6x the warm cost.
Outcome: measurement complete; port confirmed worthwhile. No code change.
ACCEPT: n/a (measurement only). Logged kind=softmax_internal_profile.
Why it matters: (1) it's ALL vectorized numpy (elementwise sub, table gather
T_A[z], boolean masks, axis-sum) — these map 1:1 onto GPU int64 torch (the math
is int64; only the final _to_field_np is mod-P, a conditional +P, also int64-able;
gl_inv for z_high is ALREADY on GPU). No inherently-serial CPU work. (2) The
in-prove call is ~2.6x the warm cost because the big exp tables T_A/T_B get
evicted from CPU cache between ops and re-paged each softmax; a GPU port keeps
them resident, so the real-prove win likely EXCEEDS the warm-profile share. (3)
Binary search (65%) + output/saturate (26%) = 91%, both trivially GPU-friendly.
Next angle (iter 6): PORT _softmax_witness_vec to GPU (torch int64 on cuda),
using the byte-exact copy in softmax_internal_profile.py as the reference. Plan:
  - Reimplement the body with torch cuda int64 tensors: x_signed, causal mask,
    max/min, the s1_at closure (z = c2[:,None]-x; in_range mask; T_A gather via
    index; masked int64 sum over dim=1), the log2(Z_max) search loop
    (torch.where updates), y_A/y_B, saturate decomp, and _to_field_np as an
    int64 conditional +P then .view(uint64). Keep the numpy path behind a flag.
  - VALIDATE (soundness-critical): a diff test asserting the GPU outputs are
    BIT-EXACT vs the numpy path on random inputs across saturate/causal/round_up/
    rescale variants (torch.equal on every out key) — this is the gate, softmax
    feeds the committed witness. Then full fast suite + Rust verify_proof=ACCEPT
    on the toy (softmax-bearing). A/B prove_s at seq512 AND seq1024.
  - PREDICT before measuring: if GPU makes the (B,M) ops ~free, softmax/call
    should drop from ~2.15s toward the non-gather overhead; at seq1024 softmax is
    ~85% of the 42.94s witness, so a large port win could take seq1024 well below
    97.30s. Write the predicted number down first, then measure.
  - Watch: torch CUDA lacks uint64 ARITH — stay in int64 until the final
    .view(torch.uint64); verify int64 wrap matches numpy uint64 wrap for the
    field reps (the _signed_floor_decomp docstring in tape.py confirms this holds
    on this build). If any op has no int64 CUDA kernel, fall back per-op, don't
    fake it. Also still open: prover/tests/test_witness_cache.py.

---

## iter 2026-07-23 — target: GPU port of _softmax_witness_vec (PROTOTYPE + bit-exact A/B)
Predicted (theory): iter6 profile said softmax witness is all vectorized numpy
(elementwise/gather/axis-sum), 91% in binary-search + saturate — ideal for GPU.
Expected a large speedup on the isolated call.
Tried: gpu_softmax_ab.py — full reimplementation of _softmax_witness_vec in torch
int64 on CUDA (isolated, NO prover code touched, per SKILL §3.3). Replicated the
Goldilocks field<->signed conversions exactly (P=2^64-2^32+1 > int64 max):
to_signed via order-preserving sign-bit-flip compare + int64-wrap (ui+FIELD_GAP);
to_field via where(s>=0, s, s-FIELD_GAP).view(uint64); table entries (<2^62) and
activations stay in int64; remainder/floor-div for the saturate z decomp.
Captured the REAL softmax args from a seq512 prove and diff-tested every out key.
Measured: BIT-EXACT vs numpy on the real captured inputs (all 15 out keys equal,
saturate+causal path). Timing (matched inputs): numpy warm 962.1ms vs GPU
10.1ms (cuda-event, median of 20, table H2D included) = 95.4x. Logged
kind=gpu_softmax_prototype_ab.
Outcome: prototype-only (validated). No prover change yet.
ACCEPT: n/a (isolated prototype; the in-prove ACCEPT gate is the next step).
Why it works: no inherently-serial CPU work — the 18-iteration binary search and
the (B,M)=2M-element gathers/sums run as a handful of GPU kernels; the field
conversions are exact in int64. The port also eliminates the ~2.6x cold-cache
table-paging penalty the numpy path pays in-prove (iter6), so the integrated win
should meet or exceed this isolated number.
PREDICT (write before measuring the integrated version): softmax in-prove was
~2.15s/call (8.58s over 4 calls at seq512); at ~10ms/call it becomes negligible.
  - seq512: witness compute ~10.05s -> ~1.5s; prove 33.58s -> ~25s (predict 24-27s).
  - seq1024: softmax ~85% of the 42.94s witness (~36s) -> ~0.1s; prove 97.30s ->
    ~61s (predict 55-65s). This would be the biggest single win of the whole loop.
Next angle (iter 7 — APPLY): wire gpu_softmax_witness into compute_fns.softmax_compute
behind a flag (env LIGERO_GPU_SOFTMAX, default on), keeping live[claim.x] on GPU
(drop the .cpu().numpy() round-trip), returning cuda tensors directly (skip the
_u64 wrap), and leaving inv_z_high (already GPU via gl_inv_batched) untouched.
VALIDATE (soundness gate): (1) extend the bit-exact diff test to the round_up=True,
non-saturate, and non-causal variants (this iter only covered the captured
saturate+causal path) — use real proves of configs that exercise them, or
construct valid inputs; (2) validate_witness_cache-style byte-identical PROOF
numpy-path vs gpu-path on the toy; (3) Rust verify_proof=ACCEPT on the toy;
(4) full fast suite; (5) A/B prove_s at seq512 AND seq1024, compare to the
predictions above. Apply only if all green; else revert and record why. Watch:
the rescale pre-path in softmax_compute (x decomposition) stays as-is — it feeds
x into the witness fn; ensure the GPU fn receives the same x tensor bit-for-bit.
Also confirm no other COMPUTE_FN besides softmax calls _softmax_witness_vec.

---

## iter 2026-07-23 — target: APPLY GPU softmax port (wire in + ACCEPT + prove A/B)
Predicted (theory, from iter7 prototype): softmax witness ~2.15s/call -> ~10ms;
seq512 33.58 -> ~24-27s; seq1024 97.30 -> ~55-65s.
Tried: wired _softmax_witness_gpu into compute_fns.softmax_compute behind
_GPU_SOFTMAX_ON (env LIGERO_GPU_SOFTMAX, default on). GPU path keeps x_for_bracket
on device (no .cpu().numpy()), builds T_A/T_B as int64 cuda tensors, returns cuda
uint64 tensors registered directly (_u64 = identity); numpy path preserved as the
fallback. inv_z_high (gl_inv_batched) unchanged.
Measured (same-process, cache on both sides, CUDA-synced):
  seq512  33.23 -> 24.44s  26.4% faster; witness 10.35 -> 1.74s; peak 5.18GB
  seq1024 100.77 -> 59.92s 40.5% faster; witness 44.05 -> 3.51s; peak 8.00GB
  Both land in the predicted range (seq1024 dead-center). Logged kind=gpu_softmax_ab.
Outcome: APPLIED
ACCEPT: yes. validate_gpu_softmax.py: proof BYTE-IDENTICAL numpy vs gpu on the toy
(root_p1/root_p2/q_irs/q_lin/p_0/opened_p1/opened_p2 all equal) + Rust
verify_proof=ACCEPT on the gpu-on proof. Branch coverage green: test_claims 21
(non-saturate/non-causal softmax), test_rescale softmax_rescale = ACCEPT (rescale
path), test_reveal 2 (rust_verify_tape). Isolated bit-exact + 95x from iter7.
Why it works: the whole witness is elementwise/gather/reduce; on GPU the 18-iter
binary search + (B,M)=2M gathers are a few kernels (~10ms) vs ~2.15s numpy, and
keeping data resident kills the cold-cache table-paging penalty. Field<->signed
conversions replicated exactly in int64 (P>int64 max) -> byte-identical. Witness at
seq1024 dropped 44.05->3.51s = no longer the dominant term. Ledger updated:
seq512 33.58->24.44, seq1024 97.30->59.92. Cumulative seq1024 218.28->59.92 = 3.64x.
Next angle (iter 9): witness is now SMALL (6% at seq1024). The prove-time frontier
has MOVED. From the seq1024 phase dump (iter3 cap-confirm, pre-gpu-softmax shares):
after removing ~40s of witness, the new top buckets are quad (~13.4s), encode
(~12.3s), fold_qlin (~7s), compile (~6.8s) — these are now co-dominant and are the
next targets. Re-run LIGERO_PHASE_TIMING at seq1024 WITH gpu-softmax+cache on to get
the fresh decomposition FIRST (the old percentages are stale), then attack the new
#1 bucket (likely quad or encode). Also: (a) re-measure seq256/384/768 with
gpu-softmax to refresh the ledger; (b) still open: fold the byte-identical gates
(validate_witness_cache.py, validate_gpu_softmax.py) into prover/tests as real
suite tests; (c) consider caching T_A/T_B on GPU across the 4 softmax ops/sweeps
(tiny, rebuilt each call now) — minor. Predict-before-measure on whichever bucket
is picked, using cost_calculator where it has a term.

---

## iter 2026-07-23 (this session, post-loop) — target: PRODUCTION frontier, not test frontier
Context correction: the witness cache (iter2-5) was a TEST-SCALE-ONLY win — it
works because the witness fits in memory, which is exactly false at 400B (7TB,
the reason the 4x recompute exists). Optimizing test prove_s led there. Fixed the
targeting: used cost_calculator at 400B geometry to find the PRODUCTION frontier.
Predicted (theory / cost model at 400B): witness_recompute 57% (claim-streaming
target — hard), ENCODE/NTT 32% (coset-NTT target), quad 10%, linear 0.4%. Note
the TEST frontier at seq1024 (quad 25%, encode 23%, witness 6%) is DIFFERENT —
quad is only 10% at production, so attacking it would be another test-scale trap.
Tried: wired coset-NTT into the real encode path (_coset_encode_codewords):
rho length-K coset NTTs (memory-efficient loop, strided write) gated to K>=2^16,
env LIGERO_COSET_NTT. Validated, then A/B'd it IN-SITU at K=2^16 on a small model.
Measured:
  - Soundness: validate_coset_ntt.py — proof BYTE-IDENTICAL single-N vs coset
    path (roots/q_irs/q_lin/p_0/opened all equal) + Rust verify_proof=ACCEPT.
  - In-situ A/B at K=2^16 (ab_coset_insitu.py): encode 0.88s -> 0.88s (NO win),
    prove 5.08 -> 4.82s (noise), peak 11.0 -> 12.4GB (+1.4GB). Logged
    kind=coset_insitu_ab.
Outcome: sound but in-situ benefit NOT demonstrated -> defaulted OPT-IN (off).
ACCEPT: yes (byte-identical + Rust ACCEPT). Correctness solid; speed unproven.
Why measured != predicted: the isolated NTT primitive is 1.19x at K=2^16
(coset_ntt_ab.py), but in a real prove at K=2^16 on a small model, encode isn't
NTT-bound (0.88s, dominated by non-NTT overhead), so the primitive win doesn't
surface. The microbenchmark OVERSTATED the in-prove benefit — the same
"isolated bench != in-situ" lesson as the whole formula-vs-reality arc. Coset's
production benefit (K>=2^18, 7TB witness where encode IS NTT-bound) is a
cost-model PREDICTION I could not measure (can't run 400B), so I did NOT claim
it as a win: default off, kept as a validated-correct opt-in lever, +memory
noted. HONEST result: this iteration targeted the right production term but did
not land a demonstrated win.
Standing production picture: the ONE demonstrated production-transferable win so
far is the loop's GPU softmax port (iter8) — it speeds each of the 4 witness
recomputes regardless of memory, so it dents the 57% witness term at any scale.
The witness cache does NOT transfer. Coset-NTT is unproven in-situ.
Next angle: the biggest real lever is claim-streaming (witness_passes 4->1) — cut
the 4x witness recompute WITHOUT storing the witness (works at 7TB). It is a
deep, soundness-critical restructuring of the 4-round protocol; do NOT rush it.
First STUDY whether it's soundly implementable (the round-N challenge depends on
the round-(N-1) global commitment, so per-claim "finish all rounds" may conflict
with Fiat-Shamir — understand the notebook's §9 precisely before touching code).
If too risky, the honest alternative is: accept that most remaining production
cost (witness) needs that deep change, and stop chasing test-scale buckets.

---
## iter9 — claim-streaming study + witness SPILL (store-once, re-read from host)

STUDY (claim-streaming, the 57%-lever): read paper §5.3/§5.4. The 4 rounds exist
so each commitment is fixed BEFORE the next challenge — this is load-bearing for
soundness ("prevents fitting a witness to a challenge already seen"; the whole
2^-16.6 bound rests on it, §5.4). "Complete each claim's rounds while its rows
are live" needs round-3/4 challenges DURING round-1 — impossible without letting
the prover see a challenge before committing. VERDICT: merging the 4 witness
passes is SOUNDNESS-BLOCKED, not a drop-in; the current code only merges because
round_seeds() pre-derives all challenges (its own docstring flags this as a TEST
shortcut) — a non-transferable trap like the witness cache. Paper lists it as §9
future work and hedges the benefit. Recorded; did NOT implement.

The SOUND remainder of the idea = "store witness once, RE-READ in rounds 2-4
instead of recompute" (doesn't merge rounds, doesn't touch challenge order).
Built it: host-memory spill (LIGERO_WITNESS_SPILL=1), pinned int64 view (byte-
identical, sidesteps uint64 CUDA/pin gaps). GATES PASS: validate_witness_spill.py
byte-identical, accept_toy_spill.py Rust ACCEPT.

A/B (ab_witness_spill.py, recompute vs gpu-cache vs host-spill):
  seq512  L4: recompute 21.50s | gpu-cache 21.13s (+1.7%) | host-spill 21.64s (-0.7%)
  seq1024 L2: recompute 28.77s | gpu-cache 27.92s (+2.9%) | host-spill 29.86s (-3.8%)
Spill DOES cut the witness TERM (-12% seq512, -21% seq1024 — scales with witness
share) BUT is net-NEGATIVE end-to-end: the PCIe write(sweep1)+re-reads add more
traffic than the recompute they save, and witness is only a SMALL share.

KEY HONEST CORRECTION: phase timing shows witness = ~6% of prove here (seq512),
NOT the 57% my cost_calculator claimed. Dominant phases are encode ~25% + quad
~21% + fold_qlin ~14%. The 57% was a cost_calculator over-weighting of
witness_recompute; real instrumented share (~6-10%) matches the paper's 10-29%.
So the entire "witness is the frontier" premise (and thus claim-streaming/spill's
value) was overstated. The real lever is ENCODE + QUAD, not witness passes.

RESULT: spill is sound + validated + kept as opt-in (default OFF, net-negative on
this hardware; only theoretical niche = witness > GPU AND recompute >> PCIe).
gpu-cache stays the only small consistent win. Redirect: target encode/quad.

---
## iter9-CORRECTION — I made the toy-scale error I warn about (user caught it)

The iter9 conclusion above ("witness only ~6%, real levers = encode+quad, spill
net-negative") was drawn from TOY configs (d512, L4, seq512) and is WRONG for
400B. Toy models have negligible model-compute while Ligero params (K_DEG=1024,
N_LIG=4096) are FIXED, so fixed-cost encode dominates and witness looks tiny.
That profile does NOT transfer to 400B.

PRODUCTION truth (cost_calculator --S 1093 --witness-mode notebook, floor scales
with W):
  identity floor (encode+quad+lin) = 12380s (3.44h)
  witness_recompute (4 passes)     = 16288s (4.53h)  = 56.8% of prove
  -> the cost_calculator's 57% witness was RIGHT; my toy 6% was the artifact.

SPILL crossover INVERTS at production (spill_costmodel_prod.py):
  effective RECOMPUTE throughput = 7.5TB / 3889s = 1.93 GB/s (a 400B forward pass)
  spill re-reads at storage BW; wins whenever eff-read-BW > 1.93 GB/s:
    NVMe single 3.5 -> +13.3% | NVMe 7 -> +28.3% | NVMe RAID/PCIe-cap -> +33.7%
    HDD 1.5 -> -26.5% (only loss)
The toy A/B showed net-negative ONLY because at toy scale recompute is trivial
(0.3s/sweep) so PCIe overhead dominates; at 400B recompute is ~1.08h/pass so even
disk re-read is far cheaper.

CORRECTED STATUS of spill: at production it is a PROMISING lever (+13-34% modeled),
NOT dead. Caveats (honest, unmeasurable here): (1) 7.5TB needs DISK backing, not
host RAM (84GB) -- the host-memory prototype validates SOUNDNESS (byte-identical
+ ACCEPT) and the mechanism, but the production backing store must be disk;
(2) full 57% needs spilling the FULL witness -- the softmax/silu-only prototype
captures just a fraction. So: mechanism proven sound; production benefit is a
cost-model PREDICTION (like coset-NTT), not a measured win. Do not oversell as
measured, do not dismiss as dead. The next real step is a disk-backed, full-
witness spill whose benefit is then confirmable only on production-scale hardware.

---
## iter10 — BabyBear / per-proof adaptive field (decision layer + run variants)

The prover field is HARDCODED to Goldilocks across 11 files: protocol.py roots
pow(7,(P-1)/K,P), CUDA kernels (`#define GL_P`), cuda_primitives.py, Rust
field.rs. A real BabyBear prove+ACCEPT is a multi-file Python+CUDA+Rust port
(new modulus/generator/two-adic roots — BabyBear is 2^27-adic vs 2^32 — and
31-bit packing). NOT done; not faked.

Built the DECISION layer (the brain + prerequisite): `field_policy.py` (sound
worst-case per-op accumulator bounds: weight-matmul tight s_a*s_b*R*||W||_1,
attention generic k*(sR)^2, P·V with softmax operand<=1) + `run_field_variants.py`
(policies goldilocks | babybear | adaptive-proof | adaptive-op; feasibility
sweep; first-order payoff; wired to prod_lens).

FINDING (400B, R=32, ||W||1=20) — BabyBear ceiling 30 bits:
  scale s   weight-matmul   attn score   attn P·V
  2^12         34.3 X          42 X         43 X
  2^8          26.3 .          34 X         35 X
  2^4          18.3 .          26 .         27 .
- At the current s=2^12 NOTHING but elementwise fits BabyBear -> pure BabyBear
  and adaptive-proof both fall all the way back to Goldilocks (0% payoff).
- Weight matmuls fit BabyBear only at s<=2^8 (lose 4 fractional bits); ALL ops
  fit only at s<=2^4 (4 fractional bits total — almost certainly too coarse to
  prove accurate inference).
- adaptive-op (mixed) at s=2^6: weight matmuls+silu+rms -> BabyBear, attention
  stays Goldilocks. BabyBear share only ~16% of witness -> +7.2% (first-order,
  NOT measured). Capped because the O(seq^2) attention-score witness dominates
  at seq=8192 and is exactly the BabyBear-hostile op.

VERDICT: BabyBear's real limiter is PRECISION, not field mechanics — you must
shrink s to fit, and whether inference stays accurate at s=2^6/2^8 is an untested
accuracy question (the true next experiment). Even granting it, the mixed payoff
is modest (~7%) because attention scores (act·act, O(seq^2)) can't leave
Goldilocks. Recorded as a decision tool + honest projection; execution blocked on
the field backend port.

---
## iter11 — PHONE-model validation campaign (local V100, remote_suite.py)

First real (non-toy) validation: Qwen2.5-0.5B (d896,L24) and Llama-3.2-1B
(d2048,L16) SHAPES with random weights (no download), seq256, real vocab. Ran
via analysis/bench/remote_suite.py --matrix phone --reps 2 on the dev V100.

Structural fix found (only surfaces on real d): embedding_lookup requires
ELL >= d. Default ELL=512 fails at d=896/2048. Fixed: ELL=next_pow2(d),
K_DEG=2*ELL, N_LIG=8*ELL (wired via AB_ELL; remote_suite computes it per config).

SOUNDNESS GATES: all 4 PASS at phone scale (byte-identical GPU softmax/silu/spill
+ Rust ACCEPT). Optimizations remain sound.

MEASURED (prove off->on, V100):
  GPU softmax (transferable win):
    Qwen-0.5B : 118.06 -> 97.43s  = 17.5% faster; witness 26.74->5.97s; peak 16.7GB
    Llama-1B  : 185.81 ->114.47s  = 38.4% faster; witness 80.23->10.08s; peak 21.6GB
    -> the win GROWS with model size (17.5%->38.4%); prod_lens +44.1%/+49.7%.
       This is the demonstrated transferable optimization, now confirmed at
       phone scale, not just toy.
  witness spill:
    Qwen-0.5B : recompute 98.80 / gpu-cache 96.94 (+1.9%) / host-spill 105.44 (-6.7%)
    Llama-1B  : OOM (CUDA OOM at d2048: prove ~30GB baseline, the gpu-cache 25%
                budget + host-spill pinning tip it over 32GB).
    -> spill net-negative at phone scale (as modeled; wins only at 400B where
       recompute >> storage-BW). OOM is a VRAM finding, not a spill-quality one.

VRAM (rental sizing, MEASURED): Qwen softmax 16.7GB, Llama softmax 21.6GB (fit
V100 32GB), Llama SPILL A/B OOM >32GB. -> a 48GB card (A6000) comfortably fits
Llama-1B incl. the memory-hungry cache/spill modes. Confirms the A6000-48GB
rental recommendation with real numbers.

Outcome: campaign tooling works end-to-end; GPU softmax validated as a real,
growing win at phone scale; spill confirmed net-negative below 400B; Llama-1B
needs >32GB for the spill A/B -> motivates the rented 48GB card. ACCEPT held.
