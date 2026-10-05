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

Four parts share the container's limit: page cache (the GGUF pages read),
anonymous memory (the process: torch and Python, decode temporaries, the
opened-column sink of T_QUERIES × rows × 8 bytes, pre-sized on the host),
pinned cache tiers, and temporary buffers. Session 6, Maverick at S = 1,000
bridged: 32.7 GB anonymous at its peak beside 206 GB of the 232 GB model in
page cache, under a 251 GB limit, with 29 s of memory-pressure stall.
Session 3 showed the cost of pinning beside a cgroup full of page cache:
each pinned block forced reclaim, and evicted shards came back from disk
every sweep. Both K2 runs therefore keep the caches' host tiers off
(`LIGERO_WEIGHT_CACHE_HOST_FRACTION=0`, `LIGERO_ROUTED_Y_HOST_FRACTION=0`)
unless the cgroup has room left after the page cache.

| | two-layer gate, S = 1,000 | full model, S = 1,000 |
|---|---|---|
| page cache | ≈ 12.2 GB | 587 GB to stay warm |
| anonymous | ≤ 33 GB (Maverick's full peak as a ceiling) | ≈ 62 GB (33 × 1.9, the rows ratio; estimate) |
| opened-column sink | 0.5 GB (54 × 1.23 M rows × 8 B) | 11–15 GB (54 × 25.8 M rows, × 1.37 for the model's row undercount at S = 1,000) |
| pinned tiers | 0 (caches on the GPU) | 0, or + 43 GB if the routed cache must go to the host (§3) |
| decode temporaries | on the GPU: the head decodes to 9.4 GB of field elements transiently | same |
| **cgroup needed** | **≈ 50 GB; any pod's** | **≈ 700 GB warm, or the pod's usual ~250 GB disk-bound** |

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

**Placement.** The gate puts both on the GPU (12.7 GB). The full run's
caches total 90 GB at S = 1,000; on a B200's 180 GB they fit only beside a
working set of at most about 80 GB. Maverick's S = 1,000 peak was
152.64 GiB with about 69 GB of caches, a working set near 95 GB, so K2's
will likely not leave room for both. Filling order: the weight cache first
(session 6: every dense load after R1 a hit), then the routed cache on the
GPU as far as it fits; the rest recomputes, which under a disk-bound
configuration means re-reading those expert shards. The gate's S = 1,000 arm
measures the working set directly: the prover streams claim by claim, and
two layers already contain every claim type at full width.

## 4. I/O and runtime, warm and disk-bound

**Expert reads per proof**, bridge on: the bridge's pass decodes every
expert shard; R1 computes the routed outputs from the active shards (at
k = 8 and S = 1,000 nearly all 384 per layer); R2 walks every shard for the
projection P = W·ρ; from R3 on no shard is read (with the routed cache). So
about three passes over the expert bytes per proof, plus the enrollment's
one: two layers 10.2 GB per pass, about 31 GB per proof; the full model
579 GB per pass, about 1.74 TB per proof.

**Maverick, measured warm (session 6, B200):**
- **Before the prove:** the pull ran at 724 MB/s; streaming expert enrollment took 116.4 s and dense enrollment 138.7 s.
- **The bridge's pass:** 126.6 s at S = 100 (shard decode 36.7 s).
- **The S = 1,000 prove:** 1,780.2 s, with loaders 123 s given the weight cache.
- **After it:** the proof was 19.67 GB, written in 30.4 s; Rust verification took 5,385 s on 20 threads at 27.2 GB RSS.

**K2 full model, warm (estimates):** the Ligero part scales with the rows,
1.87 times Maverick's, and the expert work with the expert bytes, 2.6 times:

| step | estimate |
|---|---|
| prove | about 55 min, plus the bridge's pass (about 5–6 min) |
| enrollment | about 5 min streaming, 2 min dense |
| proof | about 37–45 GB, written in about a minute |
| Rust verification | about 2.8 hours on 20 threads, at about 50 GB RSS |

**Disk-bound** adds the expert bytes over the disk's read rate: per proof
about 15 minutes at 2 GB/s or 41 minutes at 0.7 GB/s, plus 5–14 minutes for
the enrollment. The container disk's read rate is unmeasured on these hosts;
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
3. **GPU working set.** The S = 1,000 arm's peak, less its caches, sets the full run's cache placement: the routed cache stays on the GPU only if working set + 46.9 GB + routed share + a 10 GB margin ≤ 180 GB.
4. **Time per layer.** The S = 1,000 arm's sweep time per layer and its bridge pass per MoE layer, extrapolated to 61 layers, give the full prove within its session budget.
5. **Disk and memory.** The container disk's sequential read rate (O_DIRECT, bypassing the page cache) and the cgroup's file, anonymous and pressure counters during the arm decide warm (cgroup ≈ 700 GB) against disk-bound (the I/O time of §4 added).
6. **Proof and verification.** Proof size, Rust verification time and RSS at two layers, extrapolated by rows.

**The full run, provisionally:** one B200; container disk ≥ 1 TB (the 587 GB GGUF, a 37–45 GB proof, working space). The cgroup is either ≈ 700 GB warm or the usual ~250 GB disk-bound. The pull is about 14 minutes at Maverick's rate, the enrollment 7–25 minutes, the prove 1–1.5 hours, the verification about 3 hours: about 6 hours, roughly $40 on a B200 before any disk-bound overhead. The gate's measurements replace every estimate here before it is provisioned.
