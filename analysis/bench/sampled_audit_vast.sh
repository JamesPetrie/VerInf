#!/usr/bin/env bash
# Resident-model Vast campaign for the sampled audit. Model download/loading is
# intentionally outside the timed region. The Maverick driver writes the audit
# result directly; this harness fails closed if it is absent or exceeds 20 min.
set -euo pipefail
cd "${VERINF_ROOT:-/workspace/VerInf}"
PY="uv run --project $PWD python3"
export LIGERO_T_QUERIES="${LIGERO_T_QUERIES:-54}"
OUT="${SAMPLED_AUDIT_OUT:-analysis/bench/remote_results/$(hostname)/sampled-audit}"
MODEL="${MODEL_DIR:-/workspace/gguf}/UD-Q4_K_XL/Llama-4-Maverick-17B-128E-Instruct-UD-Q4_K_XL-00001-of-00005.gguf"
TOKENS="${TOKENS_JSON:?set TOKENS_JSON to the fixed 1000-token statement}"
WCOMMIT="${WEIGHT_COMMITMENT:?set WEIGHT_COMMITMENT to the enrolled model commitment}"
ROOT="${EXPECTED_WEIGHT_ROOT:?set EXPECTED_WEIGHT_ROOT to its trusted 64-hex root}"
PUBLIC_SZ="${PUBLIC_SZ:?set PUBLIC_SZ to the fixed serving-statement bound}"
VERIFIER_SECRET="${VERIFIER_SECRET_FILE:?set VERIFIER_SECRET_FILE to persistent verifier entropy}"
mkdir -p "$OUT"
test -s "$MODEL" || { echo "resident model is missing: $MODEL"; exit 2; }
test -s "$TOKENS" || { echo "token statement is missing: $TOKENS"; exit 2; }
test -s "$WCOMMIT" || { echo "weight commitment is missing: $WCOMMIT"; exit 2; }
test -s "$VERIFIER_SECRET" || { echo "verifier secret is missing: $VERIFIER_SECRET"; exit 2; }

$PY analysis/sampled_audit_cost_model.py --json | tee "$OUT/cost-model.json"

# The executable protocol/cost gate is always run before spending the real pass.
$PY layergkr/bench/run_sampled_audit.py --blocks 2596 --window 49 --sample 5 \
  --columns 61 --out "$OUT/protocol-runs.jsonl" | tee "$OUT/protocol-smoke.json"

t0=$(date +%s)
$PY demo/demo_maverick_full.py --from-gguf "$MODEL" --tokens "$TOKENS" \
  --layers 48 --experts 128 --d 5120 --d-ff 8192 --vocab 202048 \
  --weight-commitment "$WCOMMIT" --expected-weight-root "$ROOT" \
  --public-sz "$PUBLIC_SZ" --verifier-secret-file "$VERIFIER_SECRET" \
  --sampled-audit-out "$OUT/stage_times.json" 2>&1 | tee "$OUT/full.log"
wall=$(( $(date +%s) - t0 ))
test -s "$OUT/stage_times.json" || { echo "missing stage_times.json"; exit 4; }
$PY - "$OUT/stage_times.json" "$wall" <<'PY'
import json, sys
p, wall = sys.argv[1], int(sys.argv[2])
r = json.load(open(p))
required = {"wall_s", "forward_s", "c0_commit_s", "local_proofs_s",
            "rs_open_s", "verify_s", "accepted", "claims", "selected",
            "fraction", "c0_root", "binding"}
missing = sorted(required - set(r))
assert not missing, f"missing stage fields: {missing}"
assert r["accepted"] is True, f"sampled verifier rejected: {r.get('failures')}"
assert r["claims"] == 2596, f"expected 2596 blocks, got {r['claims']}"
assert r["selected"] == 265, f"expected 265 sampled blocks, got {r['selected']}"
assert abs(r["fraction"] - 265 / 2596) < 1e-12
assert r["binding"] == "raw-blake3 exact-local runtime"
assert float(r["wall_s"]) <= 1200, \
    f"timed protocol {r['wall_s']:.1f}s exceeds 20-minute cap"
assert wall <= 1200, f"process wall {wall}s exceeds 20-minute cap"
print(f"SAMPLED-AUDIT ACCEPT: 265/2596, audit={r['wall_s']:.1f}s wall={wall}s")
PY
