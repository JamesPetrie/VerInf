# B200 session 10 — archive (2026-10-05/06)

The Kimi K2 two-layer gate (analysis/k2-session-sizing.md §5–6): the toy
driver suite, the exact checks of the engine pass against the integer
reference on shard 1 of the real GGUF, the range review, fidelity against
the float reference, and the bridged real-weight proofs with their negatives.
One NVIDIA B200 (RunPod secure cloud, US-NC-2, 183 GB HBM, a 251 GB cgroup,
a 20.4-CPU quota against 192 visible cores), 250 GB container disk, host CUDA
13.0 (driver 580.126.20) under the cu128 image, 22:13 to about 00:07 UTC,
about 1.9 hours at $6.79/h. The thread pools were capped to the quota, the
host sampler ran throughout (`b200-session-10/logs/sampler.log`).

Repository state: branch `k2-gate-code`, shipped as a tarball at `ded16f2`
(stage 1) and `33feddd` (from stage 1b: the toy suites' ELL raised from 16
to 64, below). `verify_proof` was built on the pod at stage 1 (sha256
`1c700587…`, rustc 1.99.0); `verifier/` is identical in both revisions. The
CPU gate passed at each shipped revision. Weights: `unsloth/Kimi-K2-Instruct-GGUF`
at `23bf90d8`, shard 1 of UD-Q4_K_XL only, 48,935,303,072 bytes in 47 s,
sha256 `f6361585…` as pinned.

Every run here is a research run: synthetic token ids, the enrollment made
in process, the Rust check under the same run's anchors except where a
negative names the honest run's. **The verdict evidence for the negatives
is incomplete (§ Verdict evidence): the harness then counted a verifier with
no ACCEPT line as a rejection, and most negatives' archived output does not
show the verifier's own REJECT. Those negatives are to be rerun with the
corrected harness before they are cited.**

## What ran

1. Stage 1 (`stage-logs/stage1.log`): bootstrap, the shard-1 pull in the
   background, then test_claims 21/21, protocol-review negatives 10/10,
   topk_routing 21/21, head_interleave 5/5, rope_scaling 8/8. The toy driver
   suite failed 2 of 4: both proving tests stopped at the gain broadcast's
   compile, which needs d | ELL or a table of at most ELL entries, and the
   suite's ELL 16 is below the toy's d = 64. The target geometry (ELL 8192,
   gains of 7,168, 1,536 and 512) satisfies it. Fixed in `33feddd`.
2. Stage 1b (`stage-logs/stage1b.log`): the toy driver suite 4/4, then the
   Maverick binding probe (not gated).
3. Stage 2: the four exact checks.
4. Stage 3 (`stage-logs/stage3.log`): fidelity at S=100 from position 0 and
   at 130,072, the S=100 bridged proof with the four negatives, the S=1000
   bridged proof. Its bash was then stopped once the S=1000 prove had
   started (`kit/switch3b.sh`), and stage 3b (`stage-logs/stage3b.log`) made
   the long-position proof at S=100 instead of S=1000, then the S=1000
   fidelity run. The cut was for budget: the Rust check takes 11 minutes at
   S=1000.

## Results

**Toy driver suite** (`logs/test_k2_driver.log`, stage 1b's run): the engine
pass equals the integer reference at all 42 named intermediates at positions
0 and 130,072, every range inside its window; the honest proofs accepted,
bridged and not; the four negatives applied (interleave 24 applications,
yarn-witness 24, yarn-statement 4 RoPE claims, wrong-slice 6 decodes) and
reported rejected.

**Exact checks on shard 1** (`k2-check-*.json`): the integer reference first,
recording every range, then the engine pass compared name by name.

| arm | exact | ranges outside | reference | engine | Sz |
|---|---|---|---|---|---|
| S=100, from 0 | 42/42 | 0 of 139 | 177 s | 50.9 s | 3,137,119 |
| S=100, from 130,072 | 42/42 | 0 | — | 8.4 s | 3,137,267 |
| S=1000, from 0 | 42/42 | 0 | 264 s | 28.5 s | 30,688,889 |
| S=1000, from 130,072 | 42/42 | 0 | 331 s | 22.2 s | 30,691,020 |

**Range review** (§6.2, at S=1000): RMSNorm row energies at most 2^32 against
the 2^54 limb cap; the inverse RMS at most 1,293,786 against y_max 4,069,073
(layer 0's input norm on small embedding rows, ε about 10% of the
denominator; an honest y cannot exceed y_max); softmax |c2| at most 72,760
against 2^23 and spread at most 185,131 against Z_max·2^16; router logits at
most 19,341 (4.7 real units) against the sigmoid table's 2^18; the selection
bias spans 44,754 at S_sel = 2^16, a top-k width of 26 in two 13-bit words;
no rescaled output within 3 bits of 26; no field representative above 2^55.
Nothing forced a change.

**Fidelity** (`k2-fidelity-*.json`), integer against float64:

| arm | logits rel L2 | top-1 agreement | routing, end to end | routing, same input |
|---|---|---|---|---|
| S=100, from 0 | 0.0514 | 0.830 | 37/100 tokens | 3/100 |
| S=100, from 130,072 | 0.0523 | 0.840 | 40/100 | 1/100 |
| S=1000, from 0 | 0.0544 | 0.883 | 440/1000 | 11/1000 |

The error builds from the embedding on (S=100: x0 0.6%, after layer 0's
attention 2.3%, the MoE input 4.2%, the logits 5.1%; the routed output 18%
from expert flips). The median selection margin is 0.00075 real units, and
the same-input flips are ties at S_sel's resolution. Long positions are no
worse than short ones. **The exact checks establish agreement with the
integer computation; this fidelity is a separate blocker for scaling up**,
to be studied before a scale is chosen.

**Real-weight proofs** (`k2-prove-*.json`), bridged, GPU caches on, host
tiers off:

| proof | prove | peak GPU | claim-boundary peak | Rust (honest) | size |
|---|---|---|---|---|---|
| S=100, from 0, and the four negatives | 80.6 s | 50.4 GiB | 32.6 GiB | ACCEPT, 193 s | 837.7 MB |
| S=1000, from 0 | 255.9 s | 72.7 GiB | 52.7 GiB | ACCEPT, 682 s | 1,511.7 MB |
| S=100, from 130,072 | 64.1 s | 50.4 GiB | — | ACCEPT, 179 s | not saved |

The proofs are kept off the repository; `proofs-local.sha256` holds their
hashes (equal to the pod's, `session10-proofs.sha256`), the `.policy.json`
files their same-run anchors. At S=1000: sweeps 195 s (R1 39.9, R2 18.1, R3
9.5, fold 85.3, open 42.2), outside the sweeps 61 s; the bridge's per-proof
pass 5.5 s for the one MoE layer (Maverick's S=1000: 126.6 s over 24 MoE
layers); enrollment 11.3 s dense (365,514 W rows), 5.9 s bridge. CUDA
allocation at claim boundaries: layer 0 starts at about 48 GiB (the
embedding table and the one-hot inputs live), layer 1 at about 30 GiB, the
tail adds 0.1 GiB; the projections held reach 48 MiB, the sizing note's
50 MB. Host: anonymous memory peaked at 17.4 GB and the page cache at
53.2 GB; memory-pressure stall 0.02 s, IO 0.16 s; 9.6 s CPU-throttled.

**The S=1000 long-position proof is outstanding.** Positions 130,072–131,071
at S=1000 have their exact check (42/42, ranges inside) but no proof; it is
to be made at the chosen precision before a full-model run.

**Maverick binding probe** (`logs/maverick-binding-probe.log`): on a toy
tape built from demo_maverick_full's own pieces, with no public output so
the statement cannot change, each tampered tape proved against the honest
run's WeightCommitment and checked under its weight root and statement
digest: the honest control ACCEPT; `bc_ones = 2` ACCEPT with 256/256
output values changed; `g_out` raised about 0.1 in one channel ACCEPT with
255/256 changed. ACCEPT needed the verifier's own line, so these verdicts
stand. The enrolled-weight control (the router altered) was refused by the
prover's leaf check, so it has no Rust verdict. Maverick's gains and
`bc_ones` are not bound by its enrollment; the repair is a separate change.

## Verdict evidence

The harness of this session (`prover/tests/_rust_verify.py` before `f22d478`)
returned "rejected" whenever the verifier's output lacked `rust_verify:
ACCEPT`, so a verifier killed or crashed counted as a passing negative.
ACCEPT verdicts are unaffected. For each negative, what the archive holds:

| negative | archived evidence | status |
|---|---|---|
| S=100 real weights: wrong-slice | the verifier's output with `[XX ] wc bridge REJECT: bridge equation fails at eta[0]` and `rust_verify: REJECT` | explicit REJECT recovered; exit status not recorded |
| S=100 real weights: interleave, yarn-witness, yarn-statement | the last 1,500 characters of stdout + stderr, which are stderr's progress lines; the verdict line fell outside the tail | insufficient: rerun |
| yarn-statement's own-digest check | ACCEPT | stands |
| toy driver suite's six negatives (four bridged, two unbridged) | summary lines only | insufficient: rerun |
| every Rust negative in stage 1's suites (claims, protocol-review, topk_routing, head_interleave, rope_scaling) | pass/fail lines only | insufficient: rerun |
| topk_routing's threshold-guard refusal | the test checked the panic message | to be rerun under the explicit check (status 101 with the guard's message) |

The corrected harness (`f22d478`, `9abb475`) returns a verdict only when
`verify_proof` exits normally with exactly one explicit `rust_verify: ACCEPT`
or `REJECT` line, raises otherwise, and returns the verifier's check lines
apart from its output; the K2 driver records each verdict, its exit status
and its failed checks (`b913cb4`, `5d98c15`).

## Caveats

- Synthetic token ids; same-run anchors except for the negatives' honest
  ones; the enrollments were made in process.
- The fidelity figures compare against a float64 pass on the same dequantized
  weights (composition A attention); they say nothing about the GGUF's own
  quantization relative to the published model.
- The S=1000 long-position arm has no proof (above), and the negatives'
  verdict evidence is incomplete until the reruns.
- One host; the timings compare with session 6's only loosely (one MoE layer
  against 24).
