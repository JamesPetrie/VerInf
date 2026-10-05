# Multi-head latent attention for Kimi K2: design note (2026-10-08)

Kimi K2's attention is DeepSeek-V3's multi-head latent attention (MLA). This
note fixes how the prover composes it, the first piece of item 5 in
`analysis/topk-routing-design.md` §6 and the assumption the profiler's
`kimi_k2` builder carries until then (§4.1 there). A design, not an
implementation; code references are to the tree at `8d8859a` (main
`815634c` plus the item-4 profiler commits).

Sources: Kimi K2's `config.json` and `modeling_deepseek.py` at Hugging Face
revision `fd1984e2` of `moonshotai/Kimi-K2-Instruct`; llama.cpp at
`c2503049` (2026-10-05): `conversion/deepseek.py`, `src/models/deepseek2.cpp`,
`src/llama-model.cpp`, `src/llama-arch.cpp`.

## 1. Target semantics

From `DeepseekV3Attention.forward` (`modeling_deepseek.py:760-850`), with
K2's dimensions (d = 7,168, H = 64 heads, q rank 1,536, kv rank 512,
128 non-rotary and 64 rotary dimensions per query/key head, 128 per value
head):

    q        = W_qb · rmsnorm(W_qa · x)             (T, H, 192)
    q_nope, q_pe = split(q, [128, 64])              per head
    [ckv, k_pe]  = W_kva · x                        (T, 512 + 64): k_pe is ONE key for all heads
    kv       = W_kvb · rmsnorm(ckv)                 (T, H, 256)
    k_nope, v    = split(kv, [128, 128])            per head
    q_pe, k_pe   = rope(q_pe), rope(k_pe)           YaRN, below
    query[h] = [q_nope[h] | q_pe[h]]                (T, H, 192)
    key[h]   = [k_nope[h] | k_pe]                   k_pe broadcast to every head
    out      = W_o · concat_h( softmax(query[h] keyᵀ[h] · σ + mask) · v[h] )

**Softmax scale.** σ = 192^−½ · m², m = 0.1·ln(32)·mscale_all_dim + 1 =
1.346574 (`:690-696`), so σ = 0.130861 against the unscaled 0.072169.

**YaRN.** `rope_scaling` is `yarn` with factor 32, original context 4,096,
base 50,000, beta_fast = beta_slow = 1, mscale = mscale_all_dim = 1
(`:226-327`). The correction dimension for one rotation is 19.165, so the
ramp's bounds are 19 and 20 and the "ramp" is a step: rotary pairs 0–19
keep the base frequency 50,000^(−2i/64) and pairs 20–31 divide it by 32,
with no blended pair. The cos/sin factor m(32, 1)/m(32, 1) is exactly 1.
(The reference class defaults to beta_fast = 32, `:271-277`; K2's
degenerate ramp comes from its config, and the generator must follow the
formula, not assume a smooth band.) On K2's betas the step interpolates
exactly the pairs whose wavelength exceeds the original context, which is
the Llama-3 ramp with both frequency factors 1: pair 19's wavelength is
about 3,890, pair 20's about 5,422. The YaRN tests use that coincidence as an
independent cross-check of the tables.

**Pairing.** `apply_rotary_pos_emb` views each 64-wide rotary vector as
(32, 2) and transposes before `rotate_half` (`:364-370`): the stored order
rotates adjacent pairs (2i, 2i+1). llama.cpp agrees: `DEEPSEEK2` maps to
`LLAMA_ROPE_TYPE_NORM` (`llama-model.cpp:3083-3111`), adjacent pairs.

## 2. What the tree already has

- **Multi-head matmul** with any head width: `tape.matmul(..., heads=H,
  head_dim=…)` asserts only heads·head_dim = k (`tape.py:483-499`), so
  scores at head width 192 and AV at value width 128 are the existing claim.
- **RoPE** rotates (k, k + d_h/2), the half-split convention
  (`packets.py` `L2_RoPEXRot`), with tables built identically by
  `_rope_cos_sin` and the Rust `rope_cos_sin`, Llama-3 `rope_scaling`
  included and pinned by golden vectors in both test suites.
- **Public weight transforms in the loader**, recorded per committed
  variable by provenance: 1/√d_h folded into W_Q (`loader.py:149-151`),
  `_unpermute_rows`, which turns llama.cpp's interleaved Q/K rows into the
  half-split layout (`demo_maverick_block.py:57-64`), and the 5× K/V
  replication that proves GQA as plain multi-head attention.
- **RMSNorm** at any width, the windows derived from d·ε_int (below).
- **Causal softmax** over H·T rows.

What it lacks: YaRN tables, and any way to assemble a per-head vector from
two variables, one of them shared by all heads.

## 3. Composition options

Priced per layer at T = 1,000 with `profiler/claimcosts.py`. The S² term is
21 slots per score cell over 64 heads in every option, 1,344 T² per layer;
the options differ only in linear terms.

| | composition | witness, G slots | vs A | needs |
|---|---|---|---|---|
| **A** | split matmuls; half-split RoPE by de-interleaved weights; one pin per head assembling query and key, the key's rope part fanned out to the heads | 1.908 | — | YaRN tables; one new pin claim |
| B | as A, but the scores as a two-term product (nope·nopeᵀ + pe·peᵀ, one rescale), no pins | 1.884 | −1.3% | YaRN tables; a two-term matmul claim |
| D | one query matmul, RoPE over all 192 dimensions with identity entries on the nope part | 1.945 | +1.9% | YaRN tables; a pin for the key; a permuted head layout so half-split pairs stay inside the rotary part |
| A′ | as A, the shared key computed per head by replicating its weights 64× (the GQA pattern) | 1.957 | +2.5% | YaRN tables; the query pin; 1.76×10⁹ more enrolled weight slots |
| C | llama.cpp's absorbed form: queries projected into the 512-wide latent per head, one 576-wide shared key, AV in the latent, then the value decompression | 2.252 | +18% | YaRN tables; pins at width 576 |

C is how llama.cpp runs it (`deepseek2.cpp:317-350`): absorption shrinks
the decode-time cache, which a proof over the whole sequence has no use
for, and it costs the proof 18% more. A and B are within 1.3% of each
other.

**Recommendation: A.** It keeps the scores matmul, which carries the
dominant S² term, on the existing tested claim at head width 192, and its
one new object is a linear pin whose soundness is immediate: every slot of
the assembled vector is pinned to exactly one source slot. B is cheaper by
1.3% but changes the matmul's own argument (a two-term Freivalds check over
a multi-head layout); it is the natural later optimization, not the first
implementation.

## 4. Composition A in detail

### 4.1 Weights: GGUF tensors and public transforms

llama.cpp's converter splits each layer's `kv_b_proj` into `attn_k_b`,
transposed per head, and `attn_v_b` (`deepseek.py:433-449`); older GGUFs
keep the single `attn_kv_b`. Per layer (`deepseek2.cpp:95-125`;
names from `llama-arch.cpp:451-603`):

| GGUF tensor (ggml dims) | committed variable(s) | transform |
|---|---|---|
| `blk.N.attn_norm` {7168} | input-norm gain | — |
| `blk.N.attn_q_a` {7168, 1536} | W_qa (7168 × 1536) | transpose |
| `blk.N.attn_q_a_norm` {1536} | query-latent gain | — |
| `blk.N.attn_q_b` {1536, 64·192} | W_q_nope (1536 × 64·128), W_q_pe (1536 × 64·64) | split each head's 192 rows 128 / 64; σ folded into both; each head's 64 rotary rows de-interleaved by `_unpermute_rows` |
| `blk.N.attn_kv_a_mqa` {7168, 576} | W_ckv (7168 × 512), W_k_pe (7168 × 64) | split rows 512 / 64; the 64 rotary rows de-interleaved |
| `blk.N.attn_kv_a_norm` {512} | kv-latent gain | — |
| `blk.N.attn_k_b` {128, 512, 64} | W_k_nope (512 × 64·128) | head h's (512 × 128) slice placed at columns h·128 |
| `blk.N.attn_v_b` {512, 128, 64} | W_v (512 × 64·128) | head h's (128 × 512) slice transposed into columns h·128 |
| `blk.N.attn_output` {64·128, 7168} | W_o (64·128 × 7168) | transpose |

Every transform is a split, a permutation, a transpose, or the public scalar
σ, as in Maverick's loader, and provenance must name the source tensor and
the transform for each committed variable. Folding σ into W_qb is exact in
the real model (RoPE is linear); in the integer model it costs the query
precision Maverick already pays for its 1/√128. A GGUF with the legacy
`attn_kv_b` gives the same two matrices by splitting each head's 256 rows.
Because the query and key rotary parts are de-interleaved alike, their dot
product is unchanged, and the prover's half-split RoPE applies with no
pairing flag.

### 4.2 Claims per layer, in order

    rmsnorm(x, d=7168) · gain                                  existing
    a_q  = matmul(xn, W_qa)                     (T, 1536)
    rmsnorm(a_q, d=1536) · gain
    q_nope = matmul(a_qn, W_q_nope)             (T, 64·128)
    q_pe   = rope(matmul(a_qn, W_q_pe), heads=64, d_h=64, YaRN)
    ckv  = matmul(xn, W_ckv)                    (T, 512)
    k_pe = rope(matmul(xn, W_k_pe), heads=1, d_h=64, YaRN)
    rmsnorm(ckv, d=512) · gain
    k_nope = matmul(ckvn, W_k_nope)             (T, 64·128)
    v      = matmul(ckvn, W_v)                  (T, 64·128)
    query = head_interleave(q_nope, q_pe)                      NEW: per-head second part
    key   = head_interleave(k_nope, k_pe, shared=True)         NEW: second part fanned out
    scores = matmul(query, key, transpose_b, heads=64, head_dim=192)
    softmax(scores, causal, heads=64)
    av = matmul(softmax, v, heads=64, head_dim=T)
    resid + matmul(av, W_o); rmsnorm(resid1, d=7168) · gain

All but `head_interleave` and the YaRN tables are calls the tree makes today.

### 4.3 New code

1. **YaRN tables.** `RoPEConfig` gains the YaRN fields (factor, original
   context, beta_fast, beta_slow, mscale, mscale_all_dim; all defaults off,
   so existing tables stay byte-identical), `_rope_cos_sin` the YaRN
   frequencies and cos/sin factor, mirrored line-for-line in the Rust
   `rope_cos_sin` in the same f64 expression order, with golden vectors in
   both suites, as the Llama-3 scaling was, including long positions. The
   reference computes its frequencies and angles in float32; prover and
   verifier use f64. That changes nothing a verifier checks, but it is not
   negligible at long context: at position 131,071 the pinned reference's
   float32 formula moves 19 of the 64 rounded cos/sin entries at scale
   4,096, by up to 22 units (review of 2026-10-08), float32's error in
   position × frequency. The YaRN tests therefore compare the f64 tables with
   the reference formula at short and long positions under a stated
   tolerance, and the logit-level fidelity gate of §6 includes long
   positions with its own tolerance.
2. **`HeadInterleaveClaim`.** dst (T, H, w₁ + w₂) pinned slot for slot to
   a (T, H, w₁) and to b, either (T, H, w₂) per head or (T, w₂) fanned out to
   every head: one linear constraint per dst slot, no quads, cost (L, L, 0)
   like a concat. In the prover it is a packet with two modes whose index
   maps are mixed-radix strides, so it lowers to the fold's existing band
   descriptors (a case in `_lower_geometry`; `_band_key` is already generic;
   no new kernel); in the Rust verifier it
   is one new `Expander` variant, as the F01 repair's `CausalMaskedId` was.
   Uniqueness: each dst slot equals exactly one source slot. Negative tests:
   a tampered dst slot in each segment, and a fanned-out slot that differs
   from its source in one head only.
3. **The loader transforms of 4.1** and a K2 attention builder in the demo,
   with a provenance check on each transformed variable.

### 4.4 Ranges to confirm on the checkpoint

- **RMSNorm: two separate limits.** K2's ε is 10⁻⁶, so ε_int =
  round(10⁻⁶ · 2²⁴) = 17 against Maverick's 168.
  - *The inverse-RMS window* is set by the scale and ε through the smallest
    row sum, d·ε_int: at ε_int = 17 it is 22 bits at d = 7,168, 1,536 and
    512, the most the limb guard allows (2·22 + 18 = 62 ≤ 63). Larger
    latents only shrink the inverse RMS, so they cannot break it; it binds
    only if ε_int fell or the scale rose.
  - *The energy cap*: completeness needs Σx² + d·ε_int below 2⁵⁴, three
    18-bit limbs, which at scale 2¹² means a real RMS below about 387, 836
    and 1,448 at d = 7,168, 1,536 and 512 (458 at Maverick's 5,120, the
    paper's completeness cap). A larger ε_int does not raise this cap; a
    lower activation scale or a wider limb layout does, each a change to the
    bracket's parameters in both languages.
  - Raising ε_int is not a free knob either way: ε is part of the model, so
    a different ε_int changes the modeled normalization, a fidelity change.

  The measured latent magnitudes decide which, if any, applies, before the
  first proof.
- **Softmax table.** With σ folded into the query, the score range sets
  Z_max; measured from the numpy reference.
- **Latent magnitudes** feeding the two latent norms, against the norms'
  completeness windows (paper §3.6's upstream bounds).

## 5. Cost and the profiler

Composition A's 1.908 G slots per layer at T = 1,000 differ from the
`kimi_k2` builder's 1.912 G by its separate key broadcast (4.1 M slots, 0.2%),
which A folds into the key's pin; the builder adopts A's pins when the claim
exists. The model-wide S² coefficient, 81,984, is unchanged.

## 6. Gates

On the CPU: YaRN golden vectors in both languages, at long positions too,
and the f64 tables against the reference's float32 formula under a stated
tolerance; the new claim's cost row
against its compile, as `test_topk_claimcosts.py` does; the loader
transforms against reshapes of random tensors; a numpy reference of one MLA
layer against `modeling_deepseek.py` at small dimensions. On a GPU: a toy
MLA layer with honest Rust ACCEPT and the targeted REJECTs of 4.3; then
item 5's two-layer proof on the real GGUF. The logit-level fidelity check
against llama.cpp covers long positions (up to 131,071) with a stated
tolerance, since the rotary tables differ most there.

## 7. Open questions

- B (−1.3%) as a later optimization, once A is proven end to end.
- Whether llama.cpp's absorbed float computation and the integer model's
  non-absorbed one differ enough to matter for fidelity: the numpy reference
  compares both against llama.cpp's logits.
- Which RMSNorm limit of 4.4 the measured latents meet, if either, and the
  remedy that limit calls for.
