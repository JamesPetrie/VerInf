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

---
## iter12 — FIRST rented-GPU validation (A100 80GB, vast.ai, phone models)

Ran remote_suite.py --matrix phone --reps 1 on a rented A100 80GB PCIe ($0.563/h)
via InitBench/vast_run_verinf.py. Bootstrap was only 32s (fast box), not the 18min
estimate. Three real bugs found+fixed en route: (1) raw nvidia/cuda image has no
python -> vast can't start jupyter (Offline) -> use pytorch-devel image; (2)
remote_suite ROOT + all bench scripts hardcoded /home/riftuser/VerInf but the box
uses /workspace/VerInf -> exit2 -> fixed via Path(__file__).parents[2] + VERINF_ROOT
env. Cost incl. 2 failed attempts + debug ~ $0.5.

SOUNDNESS GATES on A100: all 4 PASS (byte-identical softmax/silu/spill + Rust ACCEPT).

MEASURED (A100, reps=1 — single-sample, noisier than V100 reps=2):
  GPU softmax (off->on prove):
    Qwen-0.5B : 89.0 -> 81.5s = 8.4% ; witness 12.0->4.7 ; peak 16.7GB ; lens +34.6%
    Llama-1B  : 105.2->96.5s = 8.3% ; witness 17.0->8.0 ; peak 24.2GB ; lens +30.1%
    -> KEY: the win is SMALLER on A100 (~8%) than V100 (17-38%). The CPU-numpy
       softmax is less of a bottleneck on the faster A100 box, so the GPU port
       saves less. The "transferable win" is itself hardware-dependent: bigger on
       slow GPUs, smaller on fast ones. (Still net-positive + sound everywhere.)
  witness spill (recompute / gpu-cache / host-spill):
    Qwen-0.5B : 81.75 / 81.24 (+0.6%) / 86.89 (-6.3%)
    Llama-1B  : 98.93 / 97.96 (+1.0%) / 102.76 (-3.9%)
    -> spill net-negative on A100 too (2nd GPU type) — consistent with the model
       that it only wins at 400B (recompute >> storage BW). Now confirmed on BOTH
       V100 and A100 below production. gpu-cache ~flat small win.

VRAM: Qwen 16.7GB, Llama 24.2GB — both fit A100 80GB easily. Llama-1B spill A/B,
which OOM'd on V100 32GB (iter11), ran fine on A100 -> confirms the >=48GB card
recommendation; the OOM was V100-specific, not a spill flaw.

Outcome: remote pipeline works end-to-end on rented hardware (gates + both A/Bs +
prod_lens + download + auto-destroy). Two honest findings: GPU softmax's win
shrinks on faster GPUs; spill is net-negative on a 2nd GPU type (as modeled).
Next: reps=3 for clean medians; a cheaper/slower 2nd card would extend the spill
hardware curve; disk-backed full-witness spill is still the only path to the
modeled +13-34% (needs prod scale).

## iter12b — theory-vs-run check (did the A100 runs match the cost model?)

Fed the phone shapes into cost_calculator (measured witness mode) and compared to
the A100/V100 measured recompute prove. Result: model predicted 10-18× TOO HIGH
(Qwen 1090s vs 99s meas; Llama 3310s vs 186s). Root cause = a BUG in the artifact's
shape-mode Q estimate: it counted quadratic constraints as matmul FLOPs (~m·n·k
≈ 1e11), but VerInf's Freivalds check emits only ~k per matmul (Q ≈ 3.5e5, ~10^6×
less). Fixed Q = L·(7d + d_ff + H·seq) in the artifact. After fix: pred 29s(Qwen)/
45s(Llama) — right order of magnitude, but the model still UNDERpredicts phone
scale by ~2-4× (un-modeled overhead: Python dispatch, kernel launches, fixed
setup; shrinks with scale). Verdict: soundness + structure agree; absolute cost
agrees only to order-of-magnitude at phone scale (that's why we validate
empirically). The 400B Llama-S mode uses lpd's real calibrated Q and is unaffected.

## iter12c — RETRACTION + verified root cause (I was wrong in 12b)

iter12b claimed the cost-model TERM STRUCTURE is wrong and "witness 57% is
suspect". That was an OVER-REACTION from a garbage input, not the model. Root
cause, verified by building the real Qwen tape and reading _layout:
  REAL m_total = 2,981,351 ; my shape estimator gave 848,384 -> 3.5x TOO LOW.
My estimator counted only primary activations+weights, missing the committed
gadget witness (aux Freivalds y/u/p, word decompositions for range checks, LogUp
inverses, softmax/silu tables). Both my Q "fixes" were also wrong: FLOP-based Q
was ~10^6x too high; Freivalds-only ~k was ~10^4x too low. The real ratio is
Q/W≈0.204, L/W≈0.190 (from Q_RUN/W_RUN, L_RUN/W_RUN of the 400B run).

Term-by-term vs measured A100 Qwen, with the REAL W (=3.05e9):
  streaming  model 28.3s  vs meas 29.8s  -> MATCHES (5%). Model is CORRECT.
  quad       model  6.4s  vs meas 18.8s  -> 3x under
  lin        model  0.4s  vs meas 16.2s  -> 40x under
  compile+aux+cols  un-modeled           -> 6.3s
The quad/lin gaps are SMALL-SCALE OVERHEAD (many tiny polynomial ops at K=2048,
each with a fixed kernel-launch/dispatch cost the asymptotic c(n)*size formula
omits). They vanish at 400B (K=16384) where the formulas were calibrated. So the
model is an ASYMPTOTIC model, correct at its 400B calibration, blind to per-op
overhead at small scale (already documented as the shrinking residual).

CORRECTED VERDICT: the 400B numbers/shares (witness 57%) are NOT undermined;
iter12b's alarm is retracted. Runs agree with theory in soundness + structure +
the dominant streaming term (once fed real W). The shape estimator (my addition)
was the bug; fixed with W×3.5 + Q=0.204W + L=0.190W and an honest small-scale
caveat. Lesson: verify the INPUTS (m_total from the real tape) before blaming the
model.

---

## iter13 — overhead-член: cost-модель стала «нормальной» на toy-масштабе

**Проблема (из iter12b карты доверия).** Асимптотический floor (c(n)·размер)
недооценивал замеренный floor (prove−witness) в 2–6× на toy/phone: ratio
pred/meas был 0.19 (d512 seq256) … 0.47 (d2048), сходясь к 1 только на 400B.
Причина — формула не содержит фиксированной цены на операцию/строку (диспетч
Python + запуск ядра), которая на маленькой модели доминирует.

**Вывод члена (из 39 сохранённых прогонов).** Посчитал gap = meas_floor −
pred_floor и OVH/m_total (нс/строку):

    d512 K1024:  15 700 … 22 700 ns/строку  (seq 256→1024, L2/L4)
    d896 K2048:  23 800
    d2048 K4096: 27 400

OVH/m_total ≈ КОНСТАНТА ~20 мкс/строку по всему d512→d2048 (НЕ ∝1/K — значит это
реальный per-row диспетч, а не ошибка калибровки c(n) ниже 2^12). Деление на
√(2^14/K) только УХУДШАЛО разброс — форму «√K» отверг.

**Форма члена (насыщающаяся).** Постоянный per-row overhead 20 мкс сломал бы 400B
(115M строк × 20мкс = 2300 s, +8%). Но на большом масштабе строки батчатся —
overhead/строку падает. Модель: `overhead = 22 мкс × min(m_total, 5·10⁶)`
(линейно до заполнения GPU ~5M строк, дальше плато). Физика: фиксированное число
«волн» на насыщенном GPU.

**Валидация (floor+overhead vs meas_floor).** ratio по всем прогонам:

    d512 seq256   0.98    d512 seq512  1.18    d512 seq1024  1.30
    d512 s256 L2  0.76    d896 (Qwen)  0.95    d2048 (Llama) 0.90

Все в ±30% (было 0.15–0.47). 400B: overhead = 22мкс×5M = 110 s = 0.4% (насыщен,
пренебрежимо; демонстрированный floor+witness 28 668 → prove ≈ 28 778 s). Член
НЕ байт-зависимый → BabyBear его не делит на 2.

**Куда внесено.**
- `cost_calculator.py`: константы `OVERHEAD_NS_PER_ROW=22000`,
  `OVERHEAD_SAT_ROWS=5e6`; в `predict()` `overhead_s` заменил старый крудовый
  `TOY_SCALE_FIXED_FLOOR_S=3.9` хак; отдельная строка в отчёте.
- Артефакт `cost_graph.html`: 5-й терм OVERHEAD везде — узел графа (M→OVH→TOT),
  строка в ручке (цвет --ovh плюм), 5-й член в мастер-формуле + eval, словарь,
  футер. Нарратив сшит: 4 асимптотических терма = физика, 5-й = реальность
  малого масштаба; overhead — единственный член, чья доля УБЫВАЕТ с ростом
  модели (доминирует floor на toy, 0.4% на 400B).

**Честная граница.** Член — это ФИТ (2 параметра: 22мкс, 5M строк) из данных,
как и witness measured-mode, не первопринципный вывод. Но он единственный и
физически мотивирован (per-launch диспетч, батч-насыщение), воспроизводит весь
диапазон toy d16 → phone d2048 в ±30% и не ломает 400B-якорь.

---

## iter14 — ρ=2 optimization validated end-to-end on vast.ai (A100)

**Question (from the 400B time-optimization):** the ~4.4h optimum rests on ρ=2,
T=17. §5.4 writes soundness as (1−1/ρ)^T and only ever uses ρ=4 — so does ρ=2
even produce a proof the Rust verifier ACCEPTs? Ran it.

**Setup.** New `optrun_rho.py` + `optrun_remote.sh`, `--optrun` flag in
vast_run_verinf.py. One phone model, proved under BASELINE (ρ=4, T=40, N_LIG=4·K)
vs OPTIMIZED (ρ=2, T=17, N_LIG=2·K), same seed/model, each Rust-verified.
Qwen-0.5B (d=896, ELL=1024, K=2048) and Llama-1B (d=2048, ELL=2048, K=4096).
A100 PCIe 80GB, ~27 min, ~$0.25. Instance auto-destroyed.

**Result — both ACCEPT, optimized faster:**

    Qwen-0.5B:  base ρ4/T40 85.3s (17.3GB)  →  opt ρ2/T17 78.5s (16.8GB)  −8.0%
    Llama-1B:   base ρ4/T40 99.4s (24.6GB)  →  opt ρ2/T17 89.2s (24.1GB)  −10.3%
    campaign_results.json = {"qwen_exit":0,"llama_exit":0}  (exit 0 ⟺ all 4 ACCEPT)

**Takeaways.**
1. **ρ=2, T=17 is mechanically legal** — prover builds it and the standalone Rust
   verifier ACCEPTs, both models. Removes the "does ρ=2 even run/verify" doubt.
2. **Optimized config measurably faster (−8…−10%)**, in the model-predicted
   direction (floor ↓ with ρ4→2, T40→17).

**Honest bounds.** ACCEPT ≠ proof of the soundness BOUND: one accepted proof shows
ρ=2 isn't mechanically broken, NOT that (1−1/ρ)^T is tight at rate 1/2 — still a
§5.4 theorem question. And −8…−10% is phone-scale; the 400B floor-win magnitude
differs (witness dominates differently). Direction + mechanism validated, not the
400B wall-clock.

**Bug (minor):** optrun_remote.sh cp looked for optrun_results.json in
analysis/bench/ but optrun_rho.py writes it to CWD (/workspace/VerInf), so the
per-model JSON didn't download — exit codes + phase log covered it. Fix the cp
path (or write absolute) before the next optrun.

---

## iter15 — optimization validated on the REAL Maverick MoE architecture (local V100, free)

Pivot: after the phone dense-transformer validation (iter14), ran the optimization
against the ACTUAL 400B claim types via `demo_maverick_moe.py` (real Maverick MoE
layer: router/RoutingClaim, top-1 mask, sigmoid lookup, per-expert gate/up/down
MatmulClaims, FreivaldsCombine x3, shared SwiGLU, AddClaim) — SYNTHETIC random
weights, NO gguf. New driver `analysis/bench/optrun_moe.py`. Ran on the LOCAL
Tesla V100-SXM3-32GB — no rental, $0. Real 400B Ligero geometry (ELL=8192,
K_DEG=16384). BASELINE rho=4/T=40 vs OPTIMIZED rho=2/T=17 + witness SPILL, each
Rust-verified (rust_verify_tape; verifier cargo-built locally on first call).

**Real Maverick MoE layer width (d=5120, d_ff=8192, E=8, seq=4), 44 claims:**

    BASELINE  rho4/T40        prove 23.0s  verify 670.8s  peak 13.75GB  ACCEPT
    OPTIMIZED rho2/T17 +SPILL  prove 19.5s  verify 314.3s  peak 12.45GB  ACCEPT
    prove -15% · verify -53% · both ACCEPT

Also smoke (E=4,d=256): both ACCEPT, prove ~5s.

**What this establishes.** rho=2/T=17 + witness spill produces a proof the
standalone Rust verifier ACCEPTs on the REAL Maverick MoE claim types at the real
per-layer width — not a dense toy. The two assumptions that were "maybe broken"
(rho=2 legality, spill on the real architecture) are now both empirically verified.
prove -15% (bigger than phone's -8-10%, because the real 400B Ligero geometry makes
the floor matter more); verify -53% from T=40->17.

**What it does NOT establish.** Not the full 48-layer / 128-expert / 7.5TB / S=1093
400B run: E=128 (~128GB field) won't fit a 32GB V100, and full-witness DISK spill
(the ~4h enabler) is still unwritten (host-memory spill only, and 7.5TB > any single
box RAM). The per-layer building block is proven; full-400B wall-clock still follows
from the cost model, not a measured run. The literal 4.4h needs the disk-spill build
+ big hardware; those remain the two real open items (engineering, not soundness).

---

## iter16 — DISK-backed full-witness spill BUILT and validated (byte-identical + ACCEPT)

The ~4h projection's missing piece: full-witness spill to DISK (7.5TB > any host
RAM). Built it in prover/core.py:
- `_disk_spill_open/store/load/close`: per-proof append-only file; witness rows
  written as their int64 bytes (identical bit pattern -> re-read == recompute),
  budget = filesystem free space. Env: LIGERO_WITNESS_SPILL_DISK=1 +
  LIGERO_WITNESS_SPILL_DIR=/path; file auto-deleted after prove.
- Coverage widened: disk mode spills the FULL witness (every compute claim),
  vs host-spill's softmax/silu-only.

Gate `validate_disk_spill.py` (local V100): prove one toy model recompute vs
disk-spill -> **byte-for-byte identical** (root_p1/p2, q_irs, q_lin, p_0,
opened_p1/p2 all identical) AND disk-spilled proof **Rust ACCEPT**. Byte-identity
is scale-free, so the mechanism is sound at any size.

Significance: this is the piece the registry/paper listed as unbuilt ("disk I/O
plumbing + widening coverage, not soundness"). Now built + validated. Combined
with disk-spill removing the witness-memory ceiling, larger runs (E=128 real
expert count) become feasible on a single 80GB card + big disk.

---

## iter17 — REAL card + real expert count: E=64 Maverick MoE + disk-spill + opt, ACCEPT

Ran the full optimized stack on rented A100 80GB (vast.ai, --optrun): Maverick MoE
layer E=64 (real-ish expert count; E=128 eager synthetic weights = ~129GB won't fit
80GB, E=64 = ~65GB fits), d=5120 real width, DISK-backed full-witness spill,
rho=2/T=17. Verify only the optimized run (cost control).

    OPTIMIZED rho=2/T=17 + DISK-spill : 212 claims, PROVE 101.9s, peak 65.07 GB
    spill=disk (full witness through disk file), e64_exit=0 (no REJECT; opt verify
    ran ~460s -> ACCEPT). Instance auto-destroyed. total 1184s, ~$0.18.

The explicit ACCEPT string scrolled past the poller's 120s tail and suite.log died
with the box, but exit 0 + the ~460s verify duration confirm ACCEPT (REJECT -> exit 1).

**Cumulative status of the 400B optimization (iter13-17):**
- rho=2/T=17 legal + measurably faster: validated phone (iter14) AND real Maverick
  MoE claims at real width (iter15) — Rust ACCEPT.
- disk-backed full-witness spill: BUILT (core.py) + byte-identical + ACCEPT gate
  (iter16), and ACCEPT on real MoE + real card at E=64 (iter17).
- So both pieces the "4.4h" rested on — rho=2 soundness-mechanism and disk-spill —
  are now built and verified on real hardware.

**Still not measured (honest):** the disk-spill TIME WIN is a 400B-scale property
(re-read beats recompute only when the forward pass is huge/slow: 7.5TB @ 1.9 GB/s).
At small seq recompute is fast, so spill is time-neutral/negative here — correctness
is proven, the speedup is not (and won't be at any rentable scale). Measuring the
actual 4.4h needs the real 400B run: the ~200GB GGUF checkpoint (no synthetic-weight
loader for the full model) + a box with >=7.5TB disk + ~$15/~10h. Everything up to
that — the optimization and the mechanism — is now real, not modeled.

---

## iter18 — Layer-GKR-LF: proposed replacement protocol prototyped + cost model audited

A protocol proposal arrived (`analysis/VerInf_LayerGKR_4h_theorem_ru.md`, 2026-08-04):
replace the flat Ligero trace with layer-local tensor-GKR, keeping Ligero only as a
PCS for four kinds of boundary (weights, layer I/O states, lookup boundaries, sort
records + mask tape). Claims T_prove <= 3.95 h at S=1000. Prototyped its mechanisms
and re-derived its cost model in a NEW self-contained package `layergkr/` — the
existing prover is not touched (`git diff` = one additive section in analysis/TODO.md).

**Built + gated (41 tests, `layergkr/tests/run_tests.py`):** the project-before-
sumcheck weight seam with same-column codeword equality (§4), the commit-before-
challenge ordering as an *enforced schedule* rather than prose (§3.2), the MoE
hidden stable sort + delimiters + segmented scan + permutation fingerprint (§5),
the affine ZK mask compiler and masked products (§7), and layer composition by
exact root equality (§8.1). Two things are MEASURED rather than quoted: a forged
projection codeword is shown to disagree in >= N-K positions (the (K/N)^q bound),
and the mask-tape -> transcript-polynomial map is inverted to exhibit the
uniformity §7.1 claims.

**Cost model audited.** `layergkr/cost_model.py` recomputes P and N_pad from
Maverick tensor shapes via the doc's own row-capacity rule and reproduces its §9.2
table exactly (P = 402,725,114,880; N_pad = 564,632,231,936). The arithmetic of
the theorem is sound; the disagreement is calibration.

**The finding.** The whole claim is T = 3609 + kappa*6995 + 86. The doc adopts
kappa <= 1.5 (break-even 1.530 — a 2% margin), justified against the *model's* own
upper edge 10/7.784 = 1.285. But kappa is "how far an implementation lands above
its own cost identity", and the one measured instance of that in this project is
our prover: 51,334.6 s vs a 28,059 s identity at S=1000 => **kappa_observed = 1.83**,
giving **4.57 h**, not 3.95 h. Still ~3.1x on the 14.26 h record; not a four-hour
theorem. Same class of error prod_lens.py exists to prevent — a model's self-
uncertainty is not the model-to-reality gap.

**Constructive:** 75% of the new proof-compute is weight projection + opening
encode, both driven by N_pad, and N_pad/P = 1.40 is almost entirely the 5120->8192
row rounding. ELL=5120 models to -1392 s (3.93 -> 3.54 h) — more than the entire
kappa margin. Question for the author, not a claim: ELL touches every row-layout
assumption in the scheme.

**Not built, stated plainly:** no tensor-GKR for the real layer semantics (SiLU/
softmax/RMSNorm/RoPE/range-rescale/booleanity as sumchecks — the doc budgets 305 s
total for all of it, the thinnest line in its table); no LogUp reciprocal argument;
the layer driver reveals the small projected vectors (binds them by re-encoding)
so it is not ZK as written; no enrollment, ledger, streamed proof format or memory
liveness. Nothing here is measured — there is no performance claim in the package.

**Inherited caveat:** the doc's one measured input (3609 s forward, memory
high-water) comes from full-model-hidden-run-archive.md, which is agent-authored
prose whose raw logs are on spark-c191 and could not be verified from this box.

---

## iter19 — Layer-GKR-LF implemented over real semantics; the cost model loses its kappa

Follow-up to iter18. Three complaints drove this: the prototype proved only a
dense `Y = XW` so nothing could be run; the model leaned on a calibration factor;
and verifier time was modelled nowhere (the doc's §9.4 is a conditional corollary
outside its theorem). All three are addressed in `layergkr/`; the existing prover
is still untouched.

**Implemented since iter18.** Real integer layer semantics (`semantics.py`):
RMSNorm via sum-of-squares → isqrt lookup, QKV matmuls, RoPE as affine mixing,
causal scores, exp lookup, reciprocal-normalised softmax, SiLU, SwiGLU hadamard,
both residuals, and a raw accumulator + deterministic range-checked rescale after
every multiply. The relation zoo (`relations.py`) batches hadamard / booleanity /
affine / rescale into one tagged ragged eq-weighted sumcheck; `sumcheck.py` was
generalised to sums of products to carry it. LogUp (`logup.py`) with the
`beta -> R_cmp -> alpha` order enforced by the staged API. The §7 masks are now
WIRED IN (they carry the gate batch). `full_layer.py` proves a whole layer and
verifies it; 66 tests, positive and tamper, all green.

**One real bug found by writing the tests, worth recording:** LogUp padding
cannot be zero-filled. The eq-weighted reciprocal constraint weights every
hypercube slot, so a slot with r=0 makes the sum fall short; padding must carry
real values (queries repeat a table entry and are counted in the multiplicity;
the table gets sentinel rows with multiplicity 0). Zero-padding IS safe for the
gate batch, because a zero relation stays a zero relation -- the two cases look
alike and are not.

**The model no longer has a kappa.** It predicts COUNTS (field muls with and
without deferred reduction, adds, inversions, hashed bytes, opened values), which
are a property of the protocol and geometry, not the machine; rates are measured
separately by `bench/rates.py` and never fitted. Seven configurations from
S=2/d=4 to S=12/d=32 were proved and verified with full instrumentation:
**predicted vs measured field-op counts agree to 0.00%**, per phase, for prover
AND verifier, and the geometry->relation-count level is exact too.

**What exactness bought.** With counts nailed, the residual uncertainty is
isolated in the rate card -- and the first run exposed a genuine error there: the
time model drifted 1.08x -> 4.14x with size because the card measured `(a*b) % P`
while the hot loops accumulate `a*b` and reduce once (170 ns vs 63 ns here).
Splitting the two primitives halved the drift to 0.69x -> 2.48x; the residual is
CPython operand-width growth, which does not transfer to a GPU projection. The
method is the result: a discrepancy became a found modelling bug, not a bigger
multiplier.

**Verifier cost, first time anywhere.** Predicted exactly and measured:
verification is a SHRINKING fraction of proving as the instance grows -- 37% of
prove time at S=2/d=4, 3% at S=8/d=32 -- because the verifier rides the q-column
openings and per-round interpolations, which do not grow with d the way encode
and projection do. Favourable for the scheme, and now a measurement.

**Deliberately NOT quoted: any 400B number.** `moe.py` implements and gates the
sort/segment argument, but `full_layer.py` still proves the routed FFN as
per-token matmuls, so extrapolating this code would price the very `S*E*K`
structure the scheme exists to remove. Wiring moe.py into full_layer.py is the
next step; until then a projection would be dishonest.

Still open: the local LF proof over the small roots (A, P and the LogUp
reciprocals travel in the clear -- soundness intact, hiding of those vectors is
not), the enrolment ledger, the streamed binary proof format, and memory-phase
liveness.

---

## iter20 — machine model instead of a coefficient; GPU backend; near-real widths

Three things, all driven by the same correction: a cost model that does not match
the stopwatch must be FIXED, not scaled.

**The kappa was still there, hiding.** iter19 reported "counts exact, kappa gone",
but the TIME model was off 0.69x-2.48x -- which is exactly where a kappa lives.
Fixed by modelling the machine instead of the abstraction. Four characteristics
were found and added, in order:

1. deferred vs reduced multiplication. The rate card measured `(a*b) % P`; the hot
   loops accumulate `a*b` and reduce once. 170 ns vs 63 ns. -> 2.12x
2. the UNIT was wrong. A least-squares fit for per-primitive costs over the real
   workload returned a NEGATIVE cost per multiply. Negative time is impossible, so
   the parameterisation was wrong, not the data: mul and add occur in a fixed
   ratio in this code, are collinear, and can never be separately identified. The
   unit became the LOOP BODY -- one iteration of a loop that exists in the source.
3. the encoder skips zeros, the model did not. `rs.encode_row` tests `if v:`, and
   almost every row it encodes is mostly padding (a LogUp multiplicity row is one
   value in an ELL-wide message). Nonzero slot 100 ns, zero slot 33 ns. Largest
   single error. -> 0.95x
4. unmodelled work: the column transpose in rs.Commit (23 ns/element) and the
   per-call cost of hashing a column (5.4 us -- call overhead, not GB/s). -> 1.01x

Now **0.80x-1.01x with no coefficient anywhere**, and the residual is largest on
the SMALLEST instances (fixed per-call overhead), i.e. it shrinks in the direction
the model is used for. `analysis/layergkr-cost-model.md` is the standing document;
its change log carries all of the above so none of it is rediscovered.

**Protocol consequence of (3), not just a modelling one.** At Maverick geometry a
5120-wide contraction row in an ELL=8192 message is 37% padding, and encode +
opening is ~75% of the theorem's budget. Worth asking the author whether §9.2's
N_pad counts row CAPACITY or real work; if capacity, his budget is conservative.

**GPU backend.** `layergkr/gpu.py` runs encode and linear-combination through the
PRODUCTION Goldilocks kernels (`prover/cuda_primitives.gl_matmul`), which both
lifts the size ceiling and makes a future rate card mean something for a real
prover. It refuses to enable itself for a Config until it has proved BIT-IDENTICAL
to the CPU path (5 tests). Same config went 53s -> 15s.

**Near-real widths now run.** S=16, d=128 (one Maverick attention-head group),
d_ff=256, E=8, ELL=256/K=512/N=1024 (production rate K/N = 1/4): prove 83.8 s,
verify 4.5 s, ACCEPT. That is ~2.65M matmul cells in one layer, against 336 cells
at the original toy size.

Unchanged and still blocking any 400B number: moe.py is not wired into
full_layer.py, so the routed FFN is still priced as per-token matmuls.

---

## iter21 — MoE path wired in; two more machine terms found by profiling

**MoE segmented path is now in the prover.** The routed FFN was previously one
published matmul per token -- cheaper, but it PUBLISHES the route: the trace said
which expert each token used. Now it is three nodes (gate/up/down) with the route
secret, proved by moe.py's sort + permutation fingerprint + delimiter segments
plus the §5.3 scalar identity. Cost consequence, which is the point: the seam runs
for EVERY expert, so a MoE node costs E projections rather than one. That is
exactly the term §9.2 prices, and it is now in the model. New tests cover it
(route not readable off the proof; tampered permutation product and tampered
segment sums both rejected). 74 tests green; counts exact again at 0.00%.

**Two more machine terms, found by PROFILING instead of guessing.** I had guessed
the missing time was sumcheck list allocation. A cProfile of a real prove said
otherwise:

  * `protocol.pack_column` -- one `int.to_bytes(8)` per value inside a generator,
    then a join -- was 6.3 s of a 19.8 s prove, the single largest cost in the
    whole prover. The model charged it per hash CALL. It is per VALUE, 145 ns.
  * the Lagrange matrix build (~3 s of pow() per Config) was landing inside the
    measured window. It is one-time setup; now pre-warmed before the stopwatch,
    the way a production prover builds its tables once.

Time model went 0.47x-0.95x -> **0.68x-1.28x**, and more importantly stopped
drifting monotonically with size: it now scatters around 1, which is the
signature of leftover per-call constants rather than a missing scaling term.
Still no coefficient anywhere. Worst case is real-128 at 0.68x -- next to profile.

Lesson worth keeping: my guess about where the missing time was (sumcheck list
allocation) was WRONG. The profiler found something I would not have proposed.
Profile before theorising about performance.

---

## iter22 — the prover moved onto the device; d=192 now runs

User asked to rent a box and run at realistic sizes. Before spending, two checks:
vast credit was fine ($23.49), but a GPU-utilisation trace during a 76 s prove read
**0% in 38 of 40 samples**. The card was idle -- the prover was CPython-bound, so a
rented H100 would have bought ~1.4x from a faster CPU core, not the ~30x needed to
reach real widths. Reported that instead of renting; user chose to fix the
bottleneck. Nothing was spent.

**What moved to the device.**

  * `rs.Commit` is tensor-native: codewords stay on the card, and the production
    kernels `hash_columns_streamed` + `merkle_build_blake3` do the column hashing
    and the tree. That deletes `pack_column` -- the single largest cost in the
    prover, 6.3 s of 19.8 s, one `int.to_bytes` per value -- and the device->host
    marshalling with it: only the q opened columns come back. Per-value CPU
    packing at d=128 fell from millions to 7,168.
  * the sumcheck runs on the device, INCLUDING masked proofs: the §7 mask touches
    only the scalar samples, never the vector work, so it does not block the
    device path.

Both are gated on producing bit-identical output to the CPython path, checked
before either is enabled (`rs._gpu_ok`, `sumcheck._sumcheck_gpu_ok`). The GPU
column digests and Merkle roots were verified equal to `protocol.merkle_leaf` +
`rs.build_tree` first -- if the packing layout differed, every root would.

**Result.**

    d=128   prove 76.0s -> 16.8s (4.5x),  verify 4.5s -> 2.8s
    d=192   prove 19.2s, verify 5.4s      -- a width that did not run before

74 tests green throughout.

**Modelling consequence, worth keeping:** the unit table now has two columns per
step, not one. The same protocol operation has a CPU unit and a device unit with
DIFFERENT cost structure -- the CPython encoder skips zero slots, `gl_matmul` does
not. The model selects per backend rather than averaging, which is the same
principle as the rest of iter20: model the machine, do not scale the answer.

---

## iter23 — multi-row layout: contractions wider than ELL

The ceiling after iter22 was not speed, it was structure: `PersistentWeights.enroll`
refused `n_in > ELL`. That is not a prototype limitation -- Maverick's FFN
contracts over 16384 against ELL=8192, so ANY implementation at production
geometry needs the layout, and it is exactly the `ceil(n_in/ELL)` factor the
theorem's N_pad formula already assumed. The formula assumed a layout the
implementation did not have.

**Implemented.** Each output coordinate now spans `n_blocks = ceil(n_in/ELL)` RS
rows, BLOCK-MAJOR (`row = b*n_out + i`), so at any opened column one block's
values are contiguous and the seam's linear combination applies per block. The
projected commitment gets one row per block; the opening carries one value per
block per column; the verifier checks every block.

Soundness is unchanged, and for a reason worth writing down: the manifest aligns
all output coordinates of a block on the same message and padding positions, so
the (K/N)^q argument holds per block independently. Nothing about the seam's
causal ordering changes.

**One subtlety that cost a debugging round.** Two different vectors come out of a
projection and they are not interchangeable:

  * the FLAT vector (length n_in) is what the contraction sumcheck runs on;
  * the BLOCKS (full ELL wide, secret-padding tail included) are what the
    re-encode binding needs -- the committed row is the projection of the padding
    too, so re-encoding a flattened vector reproduces a different codeword.

Both are now carried, and tampering either is rejected: the blocks by the binding,
the flat vector by the contraction. Two tests pin exactly that.

78 tests green. `d=384` (d_ff=768 against ELL=512, i.e. two blocks) now runs,
which it could not before at any speed.

**Bug in the same change, caught by running it (iter23 addendum).** The padding
length was still computed as `ELL - n_in`, which goes NEGATIVE once n_in > ELL
and silently produced empty padding rows -- enrolment then failed with a
misleading message. Only the LAST block has a tail, of length `(-n_in) % ELL`.
Fixed in both the callers and `PersistentWeights.enroll`, which now refuses
padding shorter than the tail instead of zero-filling it (zero padding is not
hiding, and silently degrading ZK is worse than an error). Regression test added.

**Width ladder, no wall (iter23 addendum 2).** d=512 (2 blocks) prove 44.0s /
verify 12.3s / peak GPU 5.38 GB; d=768 (3 blocks) prove 63.8s / verify 22.4s /
peak 5.56 GB. Both ACCEPT. Memory is NOT the constraint (5.6 GB of 32), and time
grows LINEARLY in d although matmul cells grow as d^2 -- because ELL and N are
held fixed, so widening adds blocks/rows while the per-row cost ELL*N stays put.
Conclusion: the untested axis is the LIGERO GEOMETRY, not the model width. At
production ELL=8192/N=65536 the per-row cost is ~512x this ladder's, and that is
exactly the term the ladder held constant -- so no 400B projection may be drawn
from it.

---

## iter24 — CORRECTION: the width ladder did not show linearity

An analyst review caught a wrong conclusion in iter23's addendum and in §4.9 of
the cost-model doc. I wrote "time grows LINEARLY in d, not quadratically". That is
false, and the model's own formula already said so.

`C_proj = n_out * ELL * ceil(n_in / ELL)`; for a transformer matrix
`n_in, n_out = Θ(d)`, so at fixed ELL it is `Θ(d * ELL * d/ELL) = Θ(d^2)`. The
multi-row layout SPLITS the d^2 weights into blocks; it does not remove them. The
MoE term in the same document, `E*d*d_ff`, is `Θ(E d^2)` at `d_ff ∝ d`. My prose
contradicted my own arithmetic and the arithmetic was right.

**Why the measurements looked linear.** Two reasons, both about the range rather
than the protocol. While `n_in <= ELL` the block count is 1 and the cost reads as
`n_out * ELL = Θ(d)` -- a transient regime in which most of each row is padding.
And the projection term simply is not dominant yet: recomputing the row capacity
the three FFN projections touch at ELL=512, d_ff=2d gives 1,179,648 (d=384) ->
4,325,376 (d=768), i.e. **3.67x for a 2x width, close to the 4x that d^2
predicts**, while wall-clock grew only 2.04x. When the work grows 3.7x and the
clock grows 2.0x, the clock is measuring something else -- fixed per-call cost,
under-occupied GPU, and the step function of the block count.

**Quadratics that genuinely remain:** Θ(P) in parameters (each weight is read and
projected), Θ(d^2) in width when d_ff ∝ d, Θ(E) in experts (binding a hidden route
touches every expert's weights), Θ(S^2) for dense attention. NOT quadratic: the
Ligero "quadratic constraint" -- that is the DEGREE of z = x*y, not a complexity;
those batch and cost time linear in their count. The protocol is roughly linear in
P, not O(P^2); it is quadratic in d only because P ~ d^2.

What the scheme removes is a different, artificial dimension:
`S*E*d*d_ff -> E*d*d_ff + S*d*d_ff`. Real structural win, but `E*d*d_ff` stays.
Lower bound to respect in any 400B claim: `T >= Ω(P / R_projection)` -- an online
proof applying a fresh random projection to all weights cannot be sublinear in
them.

**Fixed.** §4.9 of the cost-model doc now carries the correction, the true
asymptotics table, and an explicit statement that this ladder cannot price 400B --
for two independent reasons: it never varied ELL/N (production is ~512x the
per-row cost) and the projection term is not dominant within it.

**New instrument:** `bench/row_capacity.py` prices ONE ROW OF CAPACITY
(`seconds / sum n_out*ELL*ceil(n_in/ELL)`) and sweeps width and geometry
SEPARATELY -- the two axes the old ladder conflated. If ns-per-capacity converges,
that is an honest per-row price and extrapolation becomes defensible; if it keeps
falling, the run is still dominated by fixed costs. Neither outcome is a 400B
number by itself.

Lesson, and it is the same one as the kappa episode: I generalised an asymptotic
from a narrow range against a formula I had already written down. Check the
conclusion against the model's own algebra before writing it as a finding.

**Capacity ladder, first results (iter24 addendum).**

    run           d   ELL      N       capacity  prove s   ns/cap   step
    w-128       128   512   2048      1,658,880     11.9  7182.80
    w-256       256   512   2048      3,297,280     12.3  3732.15   0.52x
    w-512       512   512   2048      7,622,656     20.1  2640.96   0.71x
    g-256       128   256   1024        829,440      6.9  8373.48
    g-512       128   512   2048      1,658,880      8.4  5073.66   0.61x
    g-1024      128  1024   4096      3,317,760     17.8  5376.16   1.06x

Width axis NOT converged -- price still falling (7183 -> 3732 -> 2641), so fixed
costs still dominate and the width ladder cannot price anything. Confirms 4.9.1.

Geometry axis shows the FIRST flat step: 8373 -> 5074 -> 5376 (1.06x); between
g-512 and g-1024 capacity doubled and the clock went 8.4 -> 17.8 s (2.12x), i.e.
time tracked capacity. Not yet a price, for three reasons recorded in the doc:
one flat step is not convergence; N co-varied with ELL so the metric attributes
nothing (capacity counts message slots, encode work is capacity*N); and
production ELL=8192/N=65536 is 8x/16x further out. Next: extend the geometry
ladder and vary ELL and N INDEPENDENTLY.


---

## iter25 — factorial ELL/N experiment: no single per-row price, but a 3-term model

An hour, ELL and N varied INDEPENDENTLY at fixed width (d=128). Also batched the
Lagrange matrix build first (one eta^K per column, one Montgomery inversion per
column instead of ELL Fermat inversions): 52s -> 1.6s at ELL=512/N=2048, 32x,
bit-identical to protocol.lagrange, 4 new tests. Without it the far point would
have been an hour of pure setup.

**Result: ns-per-capacity does NOT converge, and the reason is that the question
was underspecified.** The three N-axis rows have IDENTICAL capacity and times of
15.4 / 18.1 / 24.9 s, so capacity alone is not the unit; and
ns/cap = a/cap + b + c*N cannot be flat while N varies. The prescribed test was
right and it correctly failed -- what it showed is that the cost needs TWO scaling
terms plus a constant.

    t = 7.21 s + 1586.73 ns * capacity + 0.469 ns * capacity * N     R^2 = 0.9998

over 7 points spanning 8x in capacity and 8x in N. The mix moves the right way:
fixed 52% -> 6% and encode 23% -> 78% from the nearest to the farthest point. The
earlier ladders were not wrong to look flat; they sat in the regime where the
constant dominated.

**The finding that blocks extrapolation:** the fitted encode coefficient is
0.469 ns/slot-position against 0.006 ns/slot measured for gl_matmul in isolation --
78x worse. Encoding is not running at kernel speed in situ: per-commitment calls
with modest row counts, each paying launch and host->device overhead. Projecting
from this fit would bake a 78x implementation inefficiency into its dominant term.

Next: batch encodes across commitments, re-fit, then consider production geometry.


---

## iter26 — commit sweep: my batching hypothesis was wrong

Analyst pushed back correctly on iter25: the regression gives one end-to-end
number and does NOT decompose into launch / H2D / sync / occupancy / allocation /
hashing, so "batch the encodes and it approaches 0.006 ns" was a hypothesis I had
written as a cause. Also correctly: 1586.73 ns*C is not "a price per row" -- it is
everything linear in C; a true per-row term would be b_row*(C/ELL). Both fixed in
the doc.

Ran the experiment they specified: C and N fixed, only the number of CUDA calls
varied, per-stage CUDA events, median/p95 after warm-up rather than best-of-five.

    rows/call    wall ms    h2d  matmul   hash   d2h   ns/slot-pos   kernel-only
            8      378.1  225.3   100.3   19.9  10.7        0.0440        0.0117
           32      279.6  208.4    57.0    5.5   2.9        0.0325        0.0066
          256      256.0  201.4    51.9    1.4   0.4        0.0298        0.0060
         2048      254.3  203.4    49.7    0.9   0.1        0.0296        0.0058

**Batching buys 1.5x and saturates by ~256 rows/call. Not 78x. Hypothesis dead.**

What the stages say instead: H2D is 80% of this path (Python lists shipped to the
device every time), gl_matmul is 20% and runs at 0.0058 ns/slot-position -- i.e.
AT the isolated-kernel rate, so the kernel was never the problem.

And the finding that matters: the whole encode+commit path costs 0.0294
ns/slot-position while the full-prover fit attributed 0.469 to C*N -- 16x more. So
most of that coefficient is not this path. Either something else in the prover
scales like C*N, or the 3-parameter fit on 7 points is absorbing differently
shaped work. Unknown, and the next step is to instrument the PROVER per stage with
the same events -- not to propose a third story.

Twice now a plausible performance narrative of mine has been wrong and the
measurement has been right. Stop narrating, keep instrumenting.

---

## iter27 — the forward pass moved to the device; the wall moved to the prover

The task named in HANDOFF.md §3: tensorise `semantics.py`, with a bit-exactness
step that was not to be skipped. Done, and the gate is stronger than the one the
handoff asked for.

**The gate.** Three levels: the traces compare equal field by field (every matmul
operand, every gate term in order, every lookup query); `check_trace` -- which
shares no arithmetic with either path -- accepts both; and the two traces produce
a BYTE-IDENTICAL proof. The third subsumes the first but says nothing about WHERE
a mismatch is, so all three run, in `tests/test_semantics.py` (6 tests) and
`bench/validate_semantics.py` (5 shapes). Suite: 83 -> 90 green.

Both paths are fed the same weights via `LayerWeights`, drawn on the device and
converted down. Deliberately not the same seed: equal seeds only give equal
weights while both implementations consume the RNG stream identically, which is
exactly the assumption a validation is supposed to test rather than rely on.

**Measured.** 10.8x at d=128 rising to 127x at d=512; d=4096/d_ff=8192 completes
in 0.24 s, which the Python path cannot do at all.

**I got the first reading of my own ladder wrong.** Version one drew weights in
Python and handed lists to the tensor path: 4.5 s at d=1024, 19.2 s at d=2048.
Both were marshalling, not arithmetic. With the weights already resident: 0.07 s
and 0.15 s. That is 4.9.8's lesson for the fourth time -- the thing that looks
like the cost is the data movement -- and the profile said so in one run.

**Two limits, both now measured rather than assumed.**

* Memory, not time, is what stops the forward pass: 13.9 GB at d=4096/E=8, OOM at
  d=5120/E=16 on a 32 GB card. The resident term is `3*E*d*d_ff` int64 of expert
  weights, i.e. the model's Theta(E) and Theta(d^2) appearing as an allocation.
* The int64 representation needs every TRUE value below 2^63 (the Python path
  divides raw accumulators by the scale, which is only the same operation while
  nothing has wrapped). Headroom: 32 bits at d=128 down to 6 bits at d=4096,
  because the toy's values grow by ~n_in per matmul. The guard fires with the
  node and the arithmetic shown, and a test makes it fire -- an unexercised guard
  is not a guard.

**A real bug found en route.** The batched NTT kernels use one block-row per
message and a CUDA grid dimension is capped at 65535, so an encode past that
fails outright: 65535 rows encode, 65536 raises `invalid configuration argument`.
LogUp commits one RS row per lookup QUERY, so this bites at about d=256, S=8 --
inside the range this prototype is meant to run, and the reason the pipeline
ladder had never got past d=128. Now chunked, with a bit-identical test.

**What this did NOT unlock, stated plainly.** Whole pipeline at ELL=1024/N=4096:

    d    S   E   forward   to_python    prove   verify   forward share
  128    8   4      0.03        0.32    18.39    11.40          0.10%
  256    8   4      0.02        0.29    27.59    12.21          0.05%

The forward pass is now a rounding error in its own pipeline. `Enrollment.enroll`
keys weights by content as a tuple-of-tuples; `relations.prove_batch` pads every
gate factor into a Python list; `logup.LogUp` builds a membership dict over every
query; `moe.source_records` makes one object per (token, coordinate) and the lane
loop is O(n_in^2 * S). Those are the prover, and they are next. A production-width
number is bounded by them, not by semantics -- so there still is not one.


## S5a admission probe — production-geometry kernel rates (2026-08-05)

**Lever:** none — this is a MEASUREMENT for the routed-projected admission gate
(`analysis/routed_projected_4h_model.py`), not an optimization A/B.

**Question:** at the target geometry (ELL=8192, K_DEG=16384, N_LIG=65536,
T_QUERIES=54), which admission stages does real hardware actually meet?

**Method:** `analysis/bench/admission_bench.py`, 30 runs/stage, bound =
max(observed max, mean+3sd). Per-slot rate x the row capacity the model prices.
`model_load` and the five semantic sweeps are NOT measured (they need real GGUF
shards; the runbook forbids random-weight substitutes), so the emitted report is
incomplete and the gate refuses it by design.

**Measured**

| stage | V100-SXM3-32GB | RTX A6000 (vast) | cap |
|---|---|---|---|
| fresh_commit_fold | 491.6 s | 428.4 s | 950 |
| quadratic | 821.7 s | 800.2 s | 765 |
| fresh_hash_coef | 32.7 s | 29.8 s | 140 |
| persistent_open | 94.1 s | 68.7 s | 1812 |
| fresh_open | 23.4 s | 17.1 s | 450 |
| proof_egress | 2114 s | 979.8 s | 879.6 |

Rates — V100: encode 4.92 ns/slot, hash 0.33, open 0.23, quad 18.28
ns/product, egress 44.9 MB/s. A6000: 4.28 / 0.30 / 0.17 / 17.78 / 97.0 MB/s.

**Result:** `quadratic` and `proof_egress` miss on BOTH cards (A6000: 1.05x and
1.11x over). The kernels are bandwidth-bound and the A6000 is not a step up
from a V100 SXM3 on memory bandwidth, so the card change bought ~13% on encode
and nothing decisive. The 4-hour envelope does not close on either machine.

**Side result:** the proof writer's chunk rendering moved from a Python
`",".join(str(v) ...)` to `json.dumps` (the C encoder). Output byte-identical
(checked on 2.5M values); single-run 47 -> 79 MB/s on the V100 box, but the
p99 bound only 42.0 -> 44.9 because I/O variance dominates on 4 vCPU. On the
A6000 box the same code bounds at 97 MB/s. Byte-level alternatives were slower
(b",".join of %d: 38.5 MB/s; numpy.savetxt: 14.9).

**Gates on the rented card:** all 12 suites passed (`gate_failures: 0`).

**Cost:** $0.07 (654 s at $0.404/h). Instance destroyed; API shows 0 instances.
