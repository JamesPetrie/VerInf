# Sampled local audit: protocol and cost model

## Executed protocol

The implementation is [`layergkr/sampled_audit.py`](../layergkr/sampled_audit.py).
It uses one global `C0` over uniquely named witness wires. A block refers to wire
IDs rather than recommitting private copies, so an upstream output and the next
block's input are the same commitment.

The interactive order per window is:

1. commit every witness wire referenced by the next 49 blocks;
2. fix the 49 block commitments (descriptor digest plus wire roots);
3. the persistent verifier secretly samples 5 block indices;
4. prove only those blocks: Freivalds for matmuls, a generic zero-relation
   sumcheck for Adam/rounding/routing/polynomial nonlinearities, and a random
   tuple fingerprint plus binary product trees for lookups;
5. fix each local transcript, then sample and open 61 distinct RS columns;
6. include every referenced wire root under the final global `C0` tree.

The real Maverick tape has 2,596 claims, used directly as 2,596 blocks: 52
complete windows and a 48-claim tail. The tail samples
`ceil(5*48/49) = 5`. Therefore the run proves 265 of 2,596 claims, or
**10.208%**. Every claim has sampling probability at least
`5/49 = 10.204%`; the tail probability is `5/48 = 10.417%`.

If the public block manifest is fixed, its dataflow is acyclic, boundary wires
are shared by commitment, and a false final result has a first false local
operation, one audit detects it with probability at least

`(5/49) * (1 - eps_local) > 10%`,

where `eps_local` combines the Freivalds, sumcheck, lookup-fingerprint and RS
binding errors. This is deliberately a covert/audit guarantee, not negligible
soundness. Independent repetitions amplify it to `1 - (44/49)^r` before the
negligible local error (about 53% at 7 audits and 90.6% at 22).

## Cost row

Run:

```bash
.venv/bin/python analysis/sampled_audit_cost_model.py
```

Default resident-model estimate:

| term | seconds | status |
|---|---:|---|
| one real Maverick forward | 289.1 | measured, saved Vast log |
| streaming `C0` commit | 113.4 | projected campaign hypothesis |
| 265/2,596 local strict proofs | 159.5 | projected from 1,562.498 s full equivalent |
| 61-column RS openings | 31.5 | projected campaign hypothesis |
| persistent verify/orchestration | 6.0 | projected campaign hypothesis |
| **total** | **599.5 (9.992 min)** | model load/download excluded |

The script exposes every rate as a CLI argument and emits JSON. There is no
calibration multiplier. The 599.5 s row remains the full-cryptographic target;
the real campaign below separately records which runtime terms are now measured.
A valid run reports both measured and projected values and must finish below the
requested 600 s timed-audit ceiling.

### Successful real RS-bound campaign, 2026-08-28

The production RS/Merkle adapter completed on the real 48-layer Maverick tape
and A100-SXM4-80GB with **ACCEPT** in **415.973 s (6.933 min)**. It committed
4,205,517 RS rows, opened 61 post-local-transcript columns (256,536,537 field
values), checked every Merkle path and selected-wire slice, sampled exactly
265/2,596 claims, and reported zero failures. The binding root was
`C0 = 921d83cc9202731b4624140b4c4b3d0ab2682424b2d843bdcb867e7ef11213f9`.

| measured RS-bound runtime term | seconds |
|---|---:|
| engine operations | 291.523 |
| total `C0` commit | 50.604 |
| of which RS encode/hash | 42.200 |
| selected exact local checks | 31.444 |
| 61-column RS rebuild/open | 37.922 |
| Merkle + selected-wire binding | 4.479 |
| **timed adapter total** | **415.973** |

Peak allocated GPU memory was 61.747 GiB. The same existing aria2 `x16/j5`
downloader fetched and exact-size-validated the five public GGUF shards in 512 s;
one-time enrollment committed 49,160,720 weight rows in 1,357.3 s. Both are
excluded from the timed adapter. The launcher downloaded the artifacts and
confirmed destruction of Vast instance `49040885`; evidence is under
`analysis/bench/remote_results/dc3f672fd559/`.

This validates the one-pass runtime plus real RS/Merkle commitment/opening path
comfortably below ten minutes. It still does **not** validate the complete
599.5 s cryptographic row: selected real claims were checked by exact
recomputation and the result explicitly reports
`cryptographic_local_proofs: false`. Freivalds/sumcheck/product-tree proof
objects remain validated by the separate portable 2,596-block smoke, not yet
bridged to the real Tape claim families.

### Earlier raw-commit runtime campaign, 2026-08-28

The repaired one-pass adapter completed on a real 48-layer Maverick tape and
A100-SXM4-80GB with **ACCEPT** in **406.775 s (6.780 min)**. It processed all
2,596 claims, sampled 265 post-commitment claims (10.208%), reported zero
failures, and produced
`C0 = 6979947a47cdead273a0efeac6b4fd920f89a421c8f00b45237ed27384181c3d`.
The measured split was:

| measured runtime-adapter term | seconds |
|---|---:|
| engine operations | 360.138 |
| striped/window-batched `C0` | 8.890 |
| 265 selected exact local checks | 37.748 |
| **timed adapter total** | **406.775** |

Peak allocated GPU memory was 61.747 GiB. The outer driver, including 48.7 s
of build/load grace, took 495 s. Model download (732 s through the existing
aria2 x16/j5 path) and one-time model enrollment (1,471.4 s) were excluded.
The full result and 1.5 MB claim/window trace are under
`analysis/bench/remote_results/5c40ddad035b/`.

This validates the one-pass/fold/C0/sampling runtime under the 600 s cap. It
does **not** validate the full 599.5 s cryptographic row: the real adapter used
exact selected-claim recomputation and reports zero RS/proof-verifier time. The
separate portable smoke validates Freivalds/sumcheck/lookup proof objects and
61 RS openings functionally, but not at the 400B witness scale.

### RS/Merkle preflight used for the real campaign

Before enabling RS in the paid run, the exact production streaming primitives
were measured locally on a V100 at `ELL=16322`, `K_DEG=16384`, `N_LIG=32768`.
This leaves 62 independent padding slots for 61 opened columns and keeps the
code rate close to 1/2. An 8,192-row probe took 0.181 s to encode/hash/commit and
0.116 s to rebuild and extract the 61 post-commitment columns. Linear projection
over the measured 508.5406 GiB witness is 92.3 s commit plus 59.3 s opening.
Streaming the opened-column BLAKE3 digests on GPU reduced the probe's Merkle
verification from 0.117 s to 0.0011 s.

These preflight projections were not substituted for the real result above.
The A100 campaign measured the effects omitted by the projection: per-variable
row padding, window boundaries, and selected-wire re-encoding. The harness still
enforces the 600 s timed-pass cap and records exact RS row/opening counts in every
window event.

### Failed real campaign, 2026-08-28

The first sampled campaign was manually stopped after a directly observed
timed-audit lower bound of **4,147.5 s (69.1 min)**; process elapsed time was at
least 4,226 s including the 78.5 s tape build. It did not validate the 599.5 s
row and must not be presented as a successful timing result.

Root cause was not model download or loading. The sampled driver forced
`LIGERO_NO_FOLD=1` so its window-local plaintext buffer could later recompute a
selected `FreivaldsCombineClaim`. That disabled Maverick's incremental MoE fold,
where 128 expert streams are absorbed and released as they are produced. The
same real geometry had previously measured 5.5 min with folding enabled. A
second bottleneck flattened every wire into one serial GPU hash column; a local
128 MiB probe measured only 0.027 GiB/s. A third harness bug used a post-exit
1,200 s assertion instead of terminating the workload, so the bad run continued
past its cap.

The repaired adapter keeps folding enabled and retains only fold-boundary host
wires until the combine's committed window is sampled. The campaign harness now
uses a 600 s timed-pass watchdog plus a 780 s outer GNU `timeout` and
writes per-claim/per-window JSONL telemetry. `C0` now stripes each wire across
up to 4,096 columns and batches equal-height matrices once per 49-claim window;
warm 128 MiB hashing probes measured 23.0--31.8 GiB/s. The reconstructed real
tape contains about 508.54 GiB of non-persistent witness, its largest window is
27.91 GiB, and its largest temporary hash batch is 16.56 GiB. Those are
diagnostics rather than a replacement for the real campaign measurement.
These changes are regression-tested locally, but the 599.5 s projection remains
**unvalidated** until a new real campaign completes below the cap.

## Real Maverick / Vast campaign

`analysis/bench/sampled_audit_vast.sh` first runs the executable 2,596-block
protocol smoke, then invokes `demo/demo_maverick_full.py` directly in
`--sampled-audit-out` mode. The timed region excludes download/model loading
and includes one engine pass, streaming C0 hashing, and the 265 selected exact
local checks. It requires an existing model enrollment and verifier-owned
secret:

```bash
TOKENS_JSON=/data/tokens.json \
WEIGHT_COMMITMENT=/data/maverick.wcommit \
EXPECTED_WEIGHT_ROOT=64_HEX_DIGITS \
PUBLIC_SZ=FIXED_SERVING_BOUND \
VERIFIER_SECRET_FILE=/data/verifier.secret \
analysis/bench/sampled_audit_vast.sh
```

The harness rejects a result unless it reports 2,596 claims, exactly 265
post-commitment selections, acceptance, with measured audit wall time below 600 s and total driver process wall time
below 780 s (the extra 180 s is build/load grace).

## Current confidentiality boundary

The executable prototype sends selected terminal wire messages to its verifier,
then binds them to `C0` with the post-proof RS columns. This gives real ordering,
commitment binding, sampling and local rejection tests, but reveals the sampled
10.2% witness. Replacing those terminal messages with the existing masked local
LF terminals is required before calling this path zero knowledge.
