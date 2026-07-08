# Full-model proof v1 — agreed design (2026-06-11)

Target: 48-layer Maverick UD-Q4_K_XL, T=1000 (442 prompt + 500 greedy + pad),
T_QUERIES=4 fast mode, all 128 experts, real verification via verifier-rs.

## Input binding (decided with James)
- **Hidden prompt tokens (positions 0..441): one-hot path.** Commit indicator
  rows r_t (T_in × V); RoutingClaim constrains the derived mask one-hot
  (booleanity+cardinality hold regardless of r); x = m @ E via MatmulClaim.
  Token ids never appear as numbers. ~2-3e9 witness slots.
- **Public continuation (442..941): public EmbeddingLookupClaim** via the
  d=1024 sub-row trick (5120 = 5×1024, 1024 | 8192; token_ids expand 5t..5t+4).
- Wire-format binding (bytes → SHA-256/AES-256 in ZK) is a SEPARATE later
  milestone; interface point = committed token bytes via WordExtraction(8).
- PREREQUISITE before claiming hiding publicly: blinding audit — verify
  encode_messages fills the K_DEG=2·ELL slack with fresh randomness.

## Other settled points
- Attention temperature: inactive below position 8192 (llama-graph.cpp:
  scale = s·log(floor((pos+off)/8192)+1)+1 = 1 exactly) — no claims needed.
- UI claim at V=202048: gap width 44 = 26+18 (18 = tiebreak bonus bits),
  4×11-bit words, guard sound; suites pass.
- Run plan: builder → fast test (2 layers, E=8, both boxes, cross-arch
  witness diff) → DUAL LAUNCH: Spark full prove (~9-12 h) + H100
  witness-only (~1-2 h, free_intermediates) computing real UI + logit diff
  vs llama.cpp; kill Spark by pidfile if H100 finds divergence.
- Diagnostics mandatory: STREAM_DBG, per-claim-type timing (tape time_ops
  into sweep), launcher EXIT/memwatch, per-layer progress prints.
- Inference artifact: H100 ~/inference-1k/ (prompt.txt 442 tok; output.txt
  3 GB — llama-completion overran -n 500, TRIM to first 500 tokens; greedy
  prefix-stable so the prefix is canonical).
- 8 pre-existing silu/rmsnorm/softmax difftest divergences: resolve or
  document before the public "real verification" claim; also publish the
  weight-commitment root + byte→field contract for model identity.

## Result (2026-06-15): proven and verified end to end

The v1 proof was generated and **verified `ACCEPT`** by `verifier-rs` — the first
verifier-confirmed unexplained-information bound on a 400B-class model.

**What ran.** 48-layer Llama-4-Maverick UD-Q4_K_XL, all 128 experts, on the real
`~/inference-1k` tokens: 442 hidden prompt (one-hot path) + 405 public
continuation (EmbeddingLookup via the d=1024 sub-row trick) = **T = 847**. Fast
mode (`LIGERO_T_QUERIES=4`), config `ELL=8192, N_LIG=65536`. Both prove and verify
on one DGX Spark (GB10). Deviations from the plan above: **T=847, not 1000** — the
trimmed inference artifact had 405 continuation tokens, not 500; and the H100
dual-launch was dropped (the H100 is offline), so the float reference came from an
earlier H100 witness pass.

**The bound is delivered to the verifier** via an `AddClaim.public_rhs` pin
(`tape.reveal`): the prover runs an engine pass to compute `Sz`, pins it as the
public constant the verifier reads, re-zeros the LogUp multiplicity tables, then
proves. No new claim type, no Merkle open — the verifier confirms the committed
`Sz` equals the pinned public value.

- **Sz = 686386 → 241.8 bits total = 0.5969 bits/token** over the 405 continuation
  positions. Matches the H100 float reference (0.596 bits/token, 241 bits at
  σ=2.83) to four significant figures.
- Parameters: `s_c = 2^28` (kernel width, matched to the fine-scale LM head so the
  Gaussian predictor actually discriminates — the mismatch that made an earlier 7B
  run degenerate is absent here); `s_y = 2^28` (exp-table floor
  `log2(1 + V/s_y) ≈ 0.001` bits at V=202048 — negligible; 2^18 would have left
  ~0.8 bits); `s_b = 2^12`; `gap_max = 2^20`.

**Prove (Spark, GB10).** build 23 s → reveal engine pass 3302.9 s (55 min) → prove
sweep **28963.8 s = 8.04 h**, peak GPU **42.3 GB** → dump 248.9 s → **12.47 GB**
proof. `EXIT=0`.

**Verify (Spark, GB10).** all six checks pass —
`merkle, irs_col, lin_sum, lin_col, quad_zero, quad_col` → **`rust_verify:
ACCEPT`**. `verify_elapsed = 11,818,024 ms` (**197 min**, 20 rayon threads). The
12.47 GB proof parsed without OOM (typed `serde_json::from_reader`, not a Value
DOM). The `lin_col` phase (per-term BLAKE3 challenge hashing over
m_total = 99,370,592 constraint rows) dominates; the other five checks are minutes.

**Verifier resident memory: ~99.6 GB** (measured — process RSS == HWM, plateaued).
This is the `Constraints` materialization (`Vec<Vec<Expander>>`, one inner Vec per
witness row), the O(witness-rows) bottleneck described in
`verifier-streaming-architecture.md` — NOT the proof JSON. It is
query-count-independent, so it is the wall that blocks the sound `T=80` config (a
~250 GB proof piled on the same ~100 GB constraint floor): the lazy/streaming
verifier is the prerequisite for sound-grade full-model verification.

**What this result is and isn't.** It certifies, in fast mode (`T_QUERIES=4`,
**test-grade soundness — not** the sound four-round `T=80` protocol), that the
full 48-layer Maverick forward was computed and that the committed `Sz` equals the
revealed 0.5969 bits/token. The blinding/ZK audit, the sound-grade run, the
weight-commitment root + byte→field model-identity contract, and the 8 pre-existing
difftest divergences all remain open as listed above — this is not yet the public
hiding-and-soundness claim.
