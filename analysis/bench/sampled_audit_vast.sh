#!/usr/bin/env bash
# Resident-model Vast campaign for the sampled audit (600s timed-pass cap).
# Model download/loading is outside that timed region; the outer process gets
# 180s build grace. Missing, rejected, or over-cap results fail closed.
set -euo pipefail
cd "${VERINF_ROOT:-/workspace/VerInf}"
PY="uv run --project $PWD python3"
export LIGERO_T_QUERIES="${LIGERO_T_QUERIES:-54}"
OUT="${SAMPLED_AUDIT_OUT:-analysis/bench/remote_results/$(hostname)/sampled-audit}"
AUDIT_CAP_S="${SAMPLED_AUDIT_TIMEOUT_S:-600}"
BUILD_GRACE_S="${SAMPLED_AUDIT_BUILD_GRACE_S:-180}"
[[ "$AUDIT_CAP_S" =~ ^[1-9][0-9]*$ ]] || {
  echo "invalid SAMPLED_AUDIT_TIMEOUT_S=$AUDIT_CAP_S"; exit 2;
}
[[ "$BUILD_GRACE_S" =~ ^[0-9]+$ ]] || {
  echo "invalid SAMPLED_AUDIT_BUILD_GRACE_S=$BUILD_GRACE_S"; exit 2;
}
PROCESS_TIMEOUT_S=$((AUDIT_CAP_S + BUILD_GRACE_S))
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
set +e
timeout --signal=TERM --kill-after=30s "${PROCESS_TIMEOUT_S}s" \
  $PY demo/demo_maverick_full.py --from-gguf "$MODEL" --tokens "$TOKENS" \
  --layers 48 --experts 128 --d 5120 --d-ff 8192 --vocab 202048 \
  --weight-commitment "$WCOMMIT" --expected-weight-root "$ROOT" \
  --public-sz "$PUBLIC_SZ" --verifier-secret-file "$VERIFIER_SECRET" \
  --sampled-audit-out "$OUT/stage_times.json" \
  --sampled-audit-progress "$OUT/progress.jsonl" \
  --sampled-audit-rs-binding \
  --sampled-audit-rs-ell 16322 --sampled-audit-rs-k-deg 16384 \
  --sampled-audit-rs-n-lig 32768 \
  --sampled-audit-timeout-s "$AUDIT_CAP_S" \
  2>&1 | tee "$OUT/full.log"
run_rc=${PIPESTATUS[0]}
set -e
wall=$(( $(date +%s) - t0 ))
if (( run_rc == 124 || run_rc == 137 )); then
  echo "SAMPLED-AUDIT TIMEOUT: audit cap=${AUDIT_CAP_S}s process cap=${PROCESS_TIMEOUT_S}s (wall=${wall}s)"
  test -s "$OUT/progress.jsonl" && {
    echo "last progress event:"; tail -n 1 "$OUT/progress.jsonl";
  }
  exit 124
fi
(( run_rc == 0 )) || { echo "sampled audit process failed rc=$run_rc"; exit "$run_rc"; }
test -s "$OUT/stage_times.json" || { echo "missing stage_times.json"; exit 4; }
$PY - "$OUT/stage_times.json" "$wall" "$AUDIT_CAP_S" "$PROCESS_TIMEOUT_S" <<'PY'
import json, sys
p, wall, cap, process_cap = sys.argv[1], *map(int, sys.argv[2:])
r = json.load(open(p))
required = {"wall_s", "forward_s", "c0_commit_s", "selected_local_arguments_s",
            "rs_commit_s", "rs_open_s", "verify_s", "accepted", "claims", "selected",
            "fraction", "c0_root", "binding", "local_argument",
            "cryptographic_local_proofs", "rs_openings_materialized",
            "cryptographic_local_proof_coverage", "rs_geometry", "rs_rows",
            "rs_opened_values", "manifest_proof_family_counts",
            "manifest_materialized_local_proofs",
            "selected_proof_family_counts", "materialized_local_proof_counts",
            "materialized_local_proofs", "exact_fallbacks", "local_proof_bytes",
            "local_proof_digests", "local_receipts", "rs_column_samples"}
missing = sorted(required - set(r))
assert not missing, f"missing stage fields: {missing}"
assert r["accepted"] is True, f"sampled verifier rejected: {r.get('failures')}"
assert r["claims"] == 2596, f"expected 2596 blocks, got {r['claims']}"
assert r["selected"] == 265, f"expected 265 sampled blocks, got {r['selected']}"
assert abs(r["fraction"] - 265 / 2596) < 1e-12
assert r["binding"] == ("rs-window+striped-blake3 "
                         + r["local_argument"] + " runtime")
argument_families = set(r["local_argument"].split("+"))
assert argument_families == {"freivalds", "product-tree", "sumcheck"}
assert r["cryptographic_local_proofs"] is True
assert r["manifest_proof_family_counts"] == {
    "freivalds": 554, "product-tree": 755, "sumcheck": 1287}
assert r["manifest_materialized_local_proofs"] == r["claims"] == 2596
assert set(r["materialized_local_proof_counts"]) == argument_families
assert r["materialized_local_proofs"] == len(r["local_proof_digests"])
assert r["materialized_local_proofs"] == r["selected"] == 265
assert r["exact_fallbacks"] == 0
assert len(r["local_receipts"]) == r["selected"]
assert len(r["rs_column_samples"]) == 53
assert sum(s["local_receipts"] for s in r["rs_column_samples"]) == r["selected"]
assert all(len(s["columns"]) == len(set(s["columns"])) == 61
           for s in r["rs_column_samples"])
assert r["local_proof_bytes"] > 0
assert r["cryptographic_local_proof_coverage"] == 1
assert r["rs_openings_materialized"] is True
assert r["rs_geometry"] == {"ELL": 16322, "K_DEG": 16384, "N_LIG": 32768}
assert int(r["rs_rows"]) > 0
assert int(r["rs_opened_values"]) == int(r["rs_rows"]) * 61
assert float(r["rs_commit_s"]) > 0
assert float(r["rs_open_s"]) > 0
assert float(r["verify_s"]) > 0
assert float(r["wall_s"]) <= cap, \
    f"timed protocol {r['wall_s']:.1f}s exceeds {cap}s cap"
assert wall <= process_cap, f"process wall {wall}s exceeds {process_cap}s cap"
print(f"SAMPLED-AUDIT ACCEPT: 265/2596, audit={r['wall_s']:.1f}s wall={wall}s")
PY
