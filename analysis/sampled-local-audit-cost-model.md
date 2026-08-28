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
calibration multiplier. Only the 289.1 s forward term is currently measured for
this exact real-model path; the other rows are admission targets until the Vast
campaign records them. A valid full run must report both measured and projected
values and must finish below 1,200 s (the requested 20-minute hard ceiling).

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
post-commitment selections, acceptance, and both measured audit wall time and
process wall time below 1,200 seconds.

## Current confidentiality boundary

The executable prototype sends selected terminal wire messages to its verifier,
then binds them to `C0` with the post-proof RS columns. This gives real ordering,
commitment binding, sampling and local rejection tests, but reveals the sampled
10.2% witness. Replacing those terminal messages with the existing masked local
LF terminals is required before calling this path zero knowledge.
