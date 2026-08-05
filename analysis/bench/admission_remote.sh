#!/usr/bin/env bash
# S5 remote batch: everything that must hold on the card the proof would run
# on, in ONE rental (the campaign skill's batching rule).
#   1. the structural + soundness gates — they are card-independent in theory,
#      which is exactly why they get re-run on new hardware;
#   2. the production-geometry kernel rates for the admission report.
# Results land in analysis/bench/remote_results/<host>/ so the runner's poller
# and downloader pick them up unchanged.
set -u
cd "${VERINF_ROOT:-/workspace/VerInf}"
PY="uv run --project $PWD python3"
HOST="$(hostname)"
OUT="analysis/bench/remote_results/$HOST"
mkdir -p "$OUT"

echo "=== gates ==="
GATE_FAILS=0
for t in test_claims test_fiat_shamir test_phase3_block test_routed_projected \
         test_rescale_claim test_moe_routed test_shard_streaming \
         test_admission_gate test_pipeline_integration test_routing_claim \
         test_persistent_weights_p3 test_persistent_weights_p5; do
  line=$( (cd prover && $PY tests/run_tests.py "$t" 2>&1 | tail -1) )
  echo "$t: $line"
  case "$line" in *" 0 failed"*) ;; *) GATE_FAILS=$((GATE_FAILS+1)) ;; esac
done
echo "gate failures: $GATE_FAILS"

echo "=== production-geometry kernel rates ==="
$PY analysis/bench/admission_bench.py --runs 30 --out "$OUT/admission_rates.json" || true
cat "$OUT/admission_rates.json" 2>/dev/null

# The poller waits for campaign_results.json; write it last so the runner only
# tears the box down once everything above is on disk.
$PY - "$OUT" "$GATE_FAILS" <<'PY'
import json, pathlib, sys
out, fails = pathlib.Path(sys.argv[1]), int(sys.argv[2])
rates = out / "admission_rates.json"
doc = {"kind": "admission_probe", "gate_failures": fails,
       "rates": json.loads(rates.read_text()) if rates.exists() else None}
(out / "campaign_results.json").write_text(json.dumps(doc, indent=2))
print("wrote", out / "campaign_results.json")
PY
echo DONE
