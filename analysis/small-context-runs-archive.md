# Small-context Maverick runs (S=10, S=100): measurement archive

Prover-runtime measurements behind §9's "Llama-4-Maverick, small context"
paragraph and the measured points in `figures/fig_runtime_measured.png`.
Runs executed 2026-07-22/23 on spark-c191 (GB10), checkout `~/infproof-run1`
(commit f2bd3f3, the headline-run binary), launched via `tools/spark_run.sh`.

**Provenance.** The primary logs are `~/mav10h.log` and `~/mav100h.log` on
spark-c191 (with `.memwatch` files alongside). The box went offline on
2026-07-23 before the full S=100 round breakdown was pulled; everything below
was captured from the live logs during and immediately after the runs, over
ssh, and is verbatim except where marked *reconstructed*. The proofs were
dumped to spark-c191:`/tmp/mav_10_hidden_t40.json` (44.6 GB) and
`/tmp/mav_100_hidden_t40.json` — note `/tmp`, at risk if the box reboots.
Neither proof has been run through the verifier.

## Configuration (both runs)

Identical to the headline 1000-token run except the tokens file:
48 layers, E=128 experts all committed, V=202048, `LIGERO_T_QUERIES=40`,
all tokens hidden, four-round protocol, `--ui-abort-above 8.0` (loose guard,
did not trigger). Transcripts are greedy llama.cpp continuations
(`llama-completion --temp 0`, UD-Q4_K_XL GGUF - the same backend and
quantization that generated the headline transcript) of byte-exact token
prefixes of the same source document (`prompt_1000.txt`); built by
`~/inference-1k/make_small_transcripts.py` on the Spark.

- tokens_10.json: prompt = first 5 tokens `[200000, 64710, 24, 53403, 290]`,
  continuation = next 5 greedy `[17025, 17102, 24, 373, 262]` (which
  reproduce the source document's own continuation).
- tokens_100.json: prompt = first 50 tokens, continuation = 50 greedy.

## Headline numbers

| | S=10 (5+5) | S=100 (50+50) | S=1000 (headline, for reference) |
|---|---|---|---|
| prove returned | 19592.1 s = 5.44 h | 21020.3 s = 5.84 h | 51334.6 s = 14.26 h |
| peak GPU | 44.50 GB | 47.32 GB | 78.13 GB |
| stream-sound peak | 47.78 GB | 50.81 GB | 83.89 GB |
| U-hat (bits/token) | 0.5144 (5 positions) | 0.4269 (50 positions) | 0.880 (500 positions) |
| S_z | 7302 | 60596 | - |
| claims | 10634 | 10724 | 11666 |
| build time | 24.7 s | 25.1 s | - |
| reveal engine pass | 1297.1 s | 1609.5 s | - |
| proof dump | 451.9 s | 481.5 s | 756.3 s |
| exit | 0 | 0 | 0 |

`prove returned` excludes build, the reveal engine pass, and the proof dump.
"blind root reproducible across rounds: True" in both runs; W-block rows
49160720 in both (weights dominate the witness at small S).

## Per-round sweep times, S=10 (verbatim from mav10h.log)

| Round | sweep elapsed | identity budget (A.5 rates at S=10) |
|---|---|---|
| R1 (commit) | 80.1 min | ~32 min encode+hash + witness pass |
| R2 (aux commit) | 33.0 min | ~1 min aux + witness pass |
| R3 (test folds) | 118.3 min | ~24 min folds + witness pass |
| R4 (openings) | 86.8 min | ~28 min re-encode + witness pass |
| total | 318.2 min (+8.3 min inter-round) | |

For S=100 only R4 was captured verbatim: 92.0 min. R1+R2 together were about
93 min and R3 about 157 min (*reconstructed* from wall-clock: prove start
~01:02 UTC, R3 observed at 14% with 5.3 min elapsed at 02:40 UTC, R4's 92 min
ending at prove return).

## The finding: T_wit's weight-processing floor

R2 at S=10 is the diagnostic: its round work (challenge-dependent
auxiliaries) is negligible at this context, yet the sweep took 33 minutes.
What remains at O(W) scale in every sweep is witness regeneration, which
re-streams all 400B weights from the GGUF and converts kquant to field each
round: a cost constant in S, about 30 min per pass on the GB10, paid four
times (~2 h at any context).

The A.5 identity prices T_wit as compute-riding ("about an hour per pass at
S=1000", linear in S), which hides this constant at S=1000 and misprices it
at S=10 (4 x 36 s vs the measured ~4 x 30 min). Cross-check: modeling the
S=1000 pass as ~30 min weight-floor + ~30 min compute predicts a ~30 min
pass at S=10; R2 measured 33. With the floor added, the identity floor at
S=10 is ~3.5 h and the implementation gap is ~1.55x, consistent with 1.75x
at S=1000 - i.e. the gap is flat across two decades of S once the floor is
priced, rather than widening. Residual per-round excess is largest in R3
(2.1x), where A.5 already places the fold's remaining DRAM traffic.

Not yet confirmed (Spark offline): R2 sweep time for S=100, predicted
~33 min (S-independent) if the weight-floor explanation is right.

## Verbatim log tails

```
mav10h.log:
[maverick-full] build 24.7s, 10634 claims
[maverick-full] reveal engine pass 1297.1s: Sz=7302 -> 2.6 bits total (0.5144 bits/token over 5 positions)
[maverick-full] reveal: Sz=7302 pinned as PUBLIC bound; verifier reads 0.5144 bits/token from the claim
[sweep] op 10676/10676 (100%) elapsed=80.1m eta=0.0m cur=3.9GB
[sweep] op 10676/10676 (100%) elapsed=33.0m eta=0.0m cur=4.1GB
[sweep] op 10676/10676 (100%) elapsed=118.3m eta=0.0m cur=5.1GB
[sweep] op 10676/10676 (100%) elapsed=86.8m eta=0.0m cur=21.6GB
  [stream-sound] 4 rounds done; blind root reproducible across rounds: True; W-block rows 49160720; W-ref False; Wnew rows 0; peak 47.78 GB
[maverick-full] prove returned (19592.1s) peakGPU=44.50GB
[maverick-full] proof dumped to /tmp/mav_10_hidden_t40.json (451.9s)
EXIT=0

mav100h.log:
[maverick-full] layers=48 E=128 T=100 V=202048 T_QUERIES=40 witness_only=False
[maverick-full] input bound: all 100 tokens hidden (50 prompt + 50 continuation)
[maverick-full] UI claim over 50 continuation positions; 10724 claims total
[maverick-full] build 25.1s, 10724 claims
[maverick-full] reveal engine pass 1609.5s: Sz=60596 -> 21.3 bits total (0.4269 bits/token over 50 positions)
[maverick-full] reveal: Sz=60596 pinned as PUBLIC bound; verifier reads 0.4269 bits/token from the claim
[sweep] op 10766/10766 (100%) elapsed=92.0m eta=0.0m cur=24.0GB
  [stream-sound] 4 rounds done; blind root reproducible across rounds: True; W-block rows 49160720; W-ref False; Wnew rows 0; peak 50.81 GB
[maverick-full] prove returned (21020.3s) peakGPU=47.32GB
[maverick-full] proof dumped to /tmp/mav_100_hidden_t40.json (481.5s)
EXIT=0
```
