# Full-model ALL-HIDDEN sound proof run — result archive (paper-critical)

Recorded 2026-07-10. **Purpose:** preserve every metric from the sound
full-model proof re-run on the hardened prover, with BOTH token streams
hidden, so it can be cited without re-running (~14 h prove + ~18 h verify).
This run supersedes the 2026-06-24 archive (`full-model-sound-run-archive.md`)
as the headline result: that run predated the RMSNorm bracket fix and the
fold shared-operand fix, and revealed its continuation tokens.

## Headline

A complete **sound** four-round Ligero proof of a **1000-token forward pass of
the full 48-layer Llama-4-Maverick (128 experts per MoE layer)**, with **all
1000 tokens hidden** (500-token prompt + 500-token continuation; the only
public quantity is the unexplained-information bound), at T=40 opened columns,
on a single DGX Spark. The independent Rust verifier **ACCEPTed** at T=30 of
the 40 checked (all six checks OK). The four-round commit-before-challenge
consistency check passed (blind root reproducible across rounds).

The scored output tokens are tied to the consumed input tokens by a shared
committed one-hot indicator (the input select and the unexplained-information
output select read the same variable rows), so one committed token stream
provably drives both the forward pass and the bound. Anchoring that committed
stream to an externally recorded transcript (the encrypted token-stream
commitment, paper Appendix E) remains future work.

## Configuration

| field | value |
|---|---|
| Model | Llama-4-Maverick, GGUF `UD-Q4_K_XL` (k-quant weights → field) |
| Layers | 48 (alternating dense / MoE; MoE layers E=128) |
| Vocab | 202048 |
| Sequence length T | 1000 = **500 prompt (hidden) + 500 continuation (hidden)** |
| Transcript | `tokens_1000.json`: greedy (temp 0) llama.cpp generation from the same GGUF, token ids captured by retokenization prefix-check |
| Claims | 11,624 |
| Witness | m_total = 109,267,016 rows (ELL=8192 slots/row ≈ 7.2 TB committed); 21,370,360 quad rows |
| Protocol | sound — four-round, commit-before-challenge |
| Query columns | T_QUERIES=40 opened (ε_IRS at T=30 checked ≈ 2^-12.5) |
| Code | VerInf @ f2bd3f3 (infproof port): RMSNorm integer-semantics bracket, FoldRunner shared-operand fix, persistent-weight layout B, all-hidden input + shared UI select, S2 verifier sentinel fix |
| Hardware | DGX Spark (GB10, 121 GiB unified, ARM64, CUDA 13.0) |

## Unexplained information (the public bound)

- **U = 0.8801 bits/token** over the 500 continuation positions
  (Sz = 1,249,299 pinned as the PUBLIC value; 440.0 bits total).
- The reveal engine pass produced a bit-identical Sz on the earlier
  public-continuation attempt of the same transcript — the all-hidden
  restructure is value-neutral; it changes what is committed, not what is
  computed.
- Higher than the June run's 0.394 bits/token: this transcript was generated
  by CPU llama.cpp (the June one by a different backend), and the integer
  model's greedy agreement with it is lower. Tightening levers if wanted:
  σ recalibration for this backend, or a transcript from the deployment
  backend the noise model was calibrated on.

## Runtimes

| phase | wall-clock |
|---|---|
| Reveal / witness engine pass (yields Sz before the prove) | 3,609 s (60.2 min) |
| **Prove (`prove returned`), four rounds** | **51,334.6 s = 14.26 h** |
| Proof dump (93.6 GB JSON) | 756.3 s (12.6 min) |
| **Verify (Rust, 20 threads, T=30 of 40 checked)** | **63,617.6 s = 17.67 h** |

Prove is 26% faster than the June run's 19.3 h (the descriptor-kernel /
fused-iNTT fold optimizations, landed after that run, at full scale).
Verify is longer than June's 14.0 h: this proof carries more constraint rows
(145,035,477 in lin_col vs ~115M — the all-hidden input indicators and their
routing claims).

## Memory

- Prove peak: **78.13 GB GPU**, **83.89 GB unified**.
- Verifier peak RSS: **75.7 GB** (79,389,540 KB maxresident; parse-dominated,
  query-count-independent as in the June run).

## Artifact

- Path on the Spark: `/tmp/mav_1000_hidden_t40.json`
- Size: **93,612,373,444 bytes (93.6 GB)**
- md5: `6b2de88a89809903b19f3501e6a9ca34`
- Verdict: `rust_verify: ACCEPT` — merkle, irs_col, lin_sum, lin_col,
  quad_zero, quad_col all `[OK]`, exit 0.

## Provenance notes

- The transcript prompt is a ~500-token essay; the continuation is the
  GGUF model's own greedy output (4-5 tok/s on 20 ARM cores via
  llama.cpp `llama-completion -no-cnv -ngl 0`). Prompt tokenizes to exactly
  500 ids (BOS-led); continuation ids are the first 500 of the generation,
  recovered by tokenizing prompt+generation and asserting the prompt ids as
  an exact prefix.
- An earlier attempt of this run (public continuation) was stopped ~7 h in
  by decision, to include the all-hidden input path in the headline; its
  reveal pass produced the identical Sz.
