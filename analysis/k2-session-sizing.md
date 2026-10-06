# Kimi K2 sessions: what they need (2026-10-05)

Resource sizing for the two GPU sessions item 5 of
`analysis/topk-routing-design.md` §6 leads to: the two-layer real-GGUF gate,
and the full-model run, which this note budgets separately and only
provisionally. It ends with the gate's concrete requirement and the
measurements from it that would justify the full run.

Sources: the GGUF's own headers (§1, read by range requests, no download);
the profiler's `kimi_k2` builder and predictions on the session-4 B200
profile (#26); measured Maverick figures from session 6
(`analysis/b200-session-6-archive.md`); the cache code in `prover/core.py`.
Estimates scaled from Maverick are marked as such.

## 1. The GGUF: variant, shards, what each run touches

**Variant.** Unsloth's `UD-Q4_K_XL`, the variant every Maverick proof used:
`unsloth/Kimi-K2-Instruct-GGUF` at revision `23bf90d8`, 13 shards, 587.11 GB
(twelve of 47.7–49.5 GB and a last of 6.37 GB). Its 1,096 tensors are Q4_K
(497), F32 (365), Q8_0 (122), Q5_K (89) and Q6_K (23): the K-quants decode
through the prover's own kernel, Q8_0 and F32 through gguf-py, as Maverick's
mixed types do. The attention is the split form (`attn_k_b`, `attn_v_b`),
the path of #29's `mla_weights_from_gguf`.

**Bytes.** Routed experts are 579.22 GB, 98.7 percent: 9.51 GB per MoE layer
where gate, up and down are Q4_K (48 layers) and 10.22 GB where down is Q5_K
(12 layers, layer 1 among them). The rest is 7.9 GB: attention 3.83, shared
experts 1.54, LM head 0.96 (Q6_K), embedding 0.66, router 0.66 (F32), dense
FFN 0.22.

**The two-layer gate needs shard 1 only.** The embedding, the head, layer 0
(attention and the dense FFN) and layer 1 (attention, 384 experts, the
shared expert, the router) are all in shard 1 (48,935,303,072 bytes, sha256
`f636158537429349a24a2e343b7b07bfe9a033d450945cce949ad67b2ef65370`). The loader globs the shards present and memory-maps them
without reading (`loader._gguf_by_name`), so a pod holding shard 1 alone
builds the gate. Of shard 1 the gate touches 12.2 GB: embedding 0.66,
head 0.96, layer 0 0.28 (attention 0.06, FFN 0.22), layer 1 10.32 (experts
10.22, attention 0.06, shared 0.03, router 0.01). A mapped file costs only
the pages read, so page cache for the gate is about 12 GB, not 49 or 587.

**The full model touches everything:** the bridge's enrollment and its
per-proof pass decode every expert shard.

## 2. Host memory, against the cgroup

Three parts share the container's limit: page cache (the GGUF pages read),
anonymous memory (the process: torch and Python, host-side decode
temporaries, and the opened-column sink of T_QUERIES × rows × 8 bytes, which
is pre-sized in ordinary host memory and so already counted here), and
pinned cache tiers. Session 6, Maverick at S = 1,000 bridged: 32.7 GB
anonymous at its peak beside 206 GB of the 232 GB model in page cache, under
a 251 GB limit, with 29 s of memory-pressure stall. That is Maverick's
measurement, a reference for K2 and not a ceiling; the gate measures K2's.
Session 3 showed the cost of pinning beside a cgroup full of page cache:
each pinned block forced reclaim, and evicted shards came back from disk
every sweep. Both K2 runs therefore keep the caches' host tiers off
(`LIGERO_WEIGHT_CACHE_HOST_FRACTION=0`, `LIGERO_ROUTED_Y_HOST_FRACTION=0`)
unless the cgroup has room left after the page cache.

| | two-layer gate, S = 1,000 | full model, S = 1,000 |
|---|---|---|
| page cache | ≈ 12.2 GB | 587 GB to stay warm |
| anonymous, the column sink included | unmeasured for K2; Maverick's full-model 32.7 GB as a reference (its sink here: 0.5 GB, 54 × 1.23 M rows × 8 B) | ≈ 62 GB estimated (32.7 × 1.9, the rows ratio), of which the sink is 11–15 GB (54 × 25.8 M rows, × 1.37 for the model's row undercount at S = 1,000) |
| pinned tiers | 0 (caches on the GPU) | 0, or + 43 GB if the routed cache must go to the host (§3) |
| **cgroup needed** | **≈ 50 GB estimated; any pod's** | **≈ 700 GB warm, or the pod's usual ~250 GB disk-bound** |

Decode temporaries are on the GPU: the K-quant kernel decodes the head to
9.4 GB of field elements transiently.

## 3. Caches: budgets and placement

**Decoded-weight cache** (`LIGERO_WEIGHT_CACHE=1`): dense weights only,
packed as int32, 4 bytes per weight; GPU tier first, at 40 percent of the
HBM free at prove start. Two layers: 2.99 × 10⁹ dense weights, 12.0 GB, of
which the embedding and head are 9.4 GB. Full model: 1.17 × 10¹⁰, 46.9 GB
(Maverick's dense set is 65 GB).

**Routed-output cache** (`LIGERO_ROUTED_Y_CACHE=1`): the routed claims'
outputs, 8 bytes per slot; GPU first, at 25 percent of free HBM. Two layers
hold one MoE block: 0.72 GB at S = 1,000, 0.07 GB at S = 100. The full
model's 60 blocks: 43.25 GB at S = 1,000, 4.33 GB at S = 100.

**Witness cache** (on by default, `LIGERO_WITNESS_CACHE`): the softmax and
SiLU outputs, on the GPU at 25 percent of the memory free at its start.
Bounded above by those claims' witness slots at 8 bytes: two layers at
S = 1,000 want at most about 23 GB (softmax 15.4, SiLU 7.4); the full model
at most about 690 GB, so at full scale it always fills its budget and
recomputes the rest (timing only).

**Growing with the tape.** Two more GPU terms are not caches and do not
self-limit: the routed claims' projections P = W·ρ, kept for the whole proof
(E·K per routed claim at 8 bytes: 50 MB for the gate's one MoE layer, 3.0 GB
for 60), and the fold's challenge buffers, cached per span for the whole
fold. Both grow with the number of layers, so **the two-layer peak cannot by
itself establish that the full model fits**: the gate records the CUDA
allocation at each claim boundary in every sweep, and the growth per layer
extrapolates to 61.

| GPU | two-layer gate, S = 1,000 | full model, S = 1,000 |
|---|---|---|
| decoded-weight cache (40 % of free) | 12.0 GB | 46.9 GB wanted |
| routed-output cache (25 % of free) | 0.72 GB | 43.25 GB wanted |
| witness cache (25 % of free) | ≤ 23 GB | fills its budget |
| projections | 0.05 GB | 3.0 GB |
| challenge buffers | measured by the gate | extrapolated from the gate |
| working set (per claim, streaming) | measured by the S = 1,000 arm | the same, if no claim is wider than two layers' |

**Placement.** The gate fits everything on a B200. At full scale the three
caches' fractions, each of the memory free at prove start, sum to 90
percent and can all fill, leaving about 10 percent (18 GB) for the working
set and the growing terms. Maverick's S = 1,000 peak was 152.64 GiB with
about 69 GB of caches, a working set near 95 GB. So the full run sets the
fractions explicitly from the gate's measurements rather than taking the
defaults. Filling order: the weight cache first (session 6: every dense load
after R1 a hit), then the routed cache as far as it fits; the routed claims
that do not fit recompute, which under a disk-bound configuration means
re-reading their expert shards (§4).

## 4. I/O and runtime, warm and disk-bound

**Expert reads per proof**, bridge on: the bridge's pass decodes every
expert shard; R1 computes the routed outputs from the active shards (at
k = 8 and S = 1,000 nearly all 384 per layer); R2 walks every shard for the
projection P = W·ρ; from R3 on no shard is read (with the routed cache). So
about three passes over the expert bytes per proof, plus the enrollment's
one, **when every routed output stays cached**: two layers 10.2 GB per pass,
about 31 GB per proof; the full model 579 GB per pass, about 1.74 TB per
proof. A routed claim whose outputs do not fit the cache recomputes them in
R3, the fold and the opening too, reading its active shards in each: six
passes, not three. With a fraction f of the MoE layers' outputs cached, the
full model reads about 579 × (6 − 3f) GB per proof: 1.74 TB at f = 1,
3.47 TB at f = 0.

**Maverick, measured warm (session 6, B200):**
- **Before the prove:** the pull ran at 724 MB/s; streaming expert enrollment took 116.4 s and dense enrollment 138.7 s.
- **The bridge's pass:** 126.6 s at S = 100 (shard decode 36.7 s).
- **The S = 1,000 prove:** 1,780.2 s, with loaders 123 s given the weight cache.
- **After it:** the proof was 19.67 GB, written in 30.4 s; Rust verification took 5,385 s on 20 threads at 27.2 GB RSS.

**K2 full model, warm (estimates):** session 6's 1,780.2 s already
contains its S = 1,000 bridge pass, 116.3 s (`mavp-s1000-bridge.log`), so the
Ligero part is 1,663.9 s, scaled by the rows ratio, 1.87; the bridge pass
scales separately with the expert bytes, 2.6 times:

| step | estimate |
|---|---|
| prove | about 52 min for the Ligero part (1,663.9 × 1.87 s), plus the bridge's pass, about 5 min (116.3 × 2.6 s) |
| enrollment | about 5 min streaming, 2 min dense |
| proof | about 37–45 GB, written in about a minute |
| Rust verification | about 2.8 hours on 20 threads, at about 50 GB RSS |

**Disk-bound** adds the expert bytes over the disk's read rate: per proof,
with every routed output cached, about 15 minutes at 2 GB/s or 41 minutes at
0.7 GB/s; with none cached about 29 or 83 minutes; plus 5–14 minutes for the
enrollment. The container disk's read rate is unmeasured on these hosts;
the gate measures it.

**Two-layer gate (estimates):**
- **S = 1,000 prove:** about 3–5 minutes, from Maverick's measured time per witness slot, with one MoE layer's bridge pass about 5–10 s.
- **S = 100:** under a minute.
- **Proofs:** under 2 GB each.
- **Rust verification:** minutes.

## 5. The two-layer gate: requirement

**Prerequisite code**, CPU-validated before any pod:
- **The K2 loader:** #29's transforms; per-expert slices of the stacked Q4_K and Q5_K expert tensors; the F32 router and `exp_probs_b`; Q8_0 `attn_k_b` and `attn_v_b`; the Q6_K head.
- **The two-layer driver:** latent attention with YaRN (#27) and `HeadInterleaveClaim` (#28); the dense FFN; `topk_moe_ffn` (#25) with the router's lookups and the shared expert; the bridge for the routed experts; the LM head.
- **An integer reference of the two layers.**
- **Measurement hooks** for the ranges of §6.

**Resources:**

| | requirement |
|---|---|
| GPU | one B200 (180 GB HBM): the full run's card, so the working set and times extrapolate directly. An H200 fits the S = 100 arm, but its 141 GB could cap the S = 1,000 measurement |
| host | cgroup ≥ 128 GB (planned use ≤ 50 GB); threads capped to the quota |
| disk | container ≥ 150 GB: shard 1 (48.9 GB, sha256-checked), toolchain, proofs |
| caches | weight cache GPU tier (12.0 GB), routed cache GPU (0.72 GB), host tiers off |
| arms | S = 100; S = 1,000 from position 0; S = 1,000 at positions 130,072–131,071 for the long-position fidelity check. Each Rust-verified, with targeted negatives on the real weights (YaRN stripped, a wrong expert slice, an interleave tamper) |
| time and cost | about 2 hours, about $14 at $6.79/h |

## 6. The measurements that justify the full-model run

1. **Correctness.** The gate's proofs are accepted and its negatives rejected. The integer reference matches composition A's float64 reference (#29) within a stated tolerance at short and long positions, the router's top-k choices included.
2. **Ranges from the real weights:**
   - RMSNorm latents against both limits of the MLA note's §4.4 (the inverse-RMS window and the 2⁵⁴ energy cap) at d = 7,168, 1,536 and 512;
   - the softmax score range with σ folded in (Z_max);
   - the router-logit and bias ranges, which fix the selection table's domain and `K2_TOPK`'s word counts.

   Each must fall inside its configured window, or the change it forces is made before the full run.
3. **GPU memory.** The S = 1,000 arm's per-claim working set, and the per-layer growth of the projections and challenge buffers extrapolated to 61 layers, set the full run's cache fractions explicitly: working set + growth to 61 layers + weight cache + routed share + witness share + a 10 GB margin ≤ 180 GB.
4. **Time per layer.** The S = 1,000 arm's sweep time per layer and its bridge pass per MoE layer, extrapolated to 61 layers, give the full prove within its session budget.
5. **Disk and memory.** The container disk's sequential read rate (O_DIRECT, bypassing the page cache), K2's anonymous peak, and the cgroup's file, anonymous and pressure counters during the arm decide warm (cgroup ≈ 700 GB) against disk-bound, with §4's I/O time added at the routed-cache fraction the GPU allows.
6. **Proof and verification.** Proof size, Rust verification time and RSS at two layers, extrapolated by rows.

**The full run, provisionally:** one B200; container disk ≥ 1 TB (the 587 GB GGUF, a 37–45 GB proof, working space). The cgroup is either ≈ 700 GB warm or the usual ~250 GB disk-bound. The pull is about 14 minutes at Maverick's rate, the enrollment 7–25 minutes, the prove 1–1.5 hours, the verification about 3 hours: about 6 hours, roughly $40 on a B200 before any disk-bound overhead. The gate's measurements replace every estimate here before it is provisioned.
