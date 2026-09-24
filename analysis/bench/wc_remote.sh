#!/usr/bin/env bash
# WC-LCRL-STC remote validation batch (one rental, campaign batching rule):
#   1. the wc_bridge suite (8 tests incl. negatives) — the soundness gate must
#      hold on the card the numbers will be quoted for;
#   2. wc_bench at production geometry over growing block slices, so the
#      per-param rates and their scaling are measured, not extrapolated from
#      one point;
#   3. the neighbouring gates the bridge will splice into (fiat_shamir,
#      routed_projected) as a canary that the branch is healthy on the box.
# Results land in analysis/bench/remote_results/<host>/ for the runner's
# poller/downloader.
set -u
cd "${VERINF_ROOT:-/workspace/VerInf}"
PY="uv run --project $PWD python3"
HOST="$(hostname)"
OUT="analysis/bench/remote_results/$HOST"
mkdir -p "$OUT"

echo "=== gates ==="
GATE_FAILS=0
for t in test_wc_bridge test_fiat_shamir test_routed_projected; do
  line=$( (cd prover && $PY tests/run_tests.py "$t" 2>&1 | tail -1) )
  echo "$t: $line"
  case "$line" in *" 0 failed"*) ;; *) GATE_FAILS=$((GATE_FAILS+1)) ;; esac
done
echo "gate failures: $GATE_FAILS"

echo "=== wc_bench: production geometry, growing slices ==="
for CFG in "4096 1" "4096 4" "4096,11008 2"; do
  set -- $CFG
  W="$1"; BLK="$2"
  echo "--- widths=$W blocks=$BLK ---"
  $PY analysis/bench/wc_bench.py --widths "$W" --blocks "$BLK" \
      --json "$OUT/wc_bench_${W//,/x}_b${BLK}.json" || true
done

$PY - "$OUT" "$GATE_FAILS" <<'PY'
import json, pathlib, sys
out, fails = pathlib.Path(sys.argv[1]), int(sys.argv[2])
runs = {p.name: json.loads(p.read_text())
        for p in out.glob("wc_bench_*.json")}
doc = {"kind": "wc_bridge_validation", "gate_failures": fails, "runs": runs}
(out / "campaign_results.json").write_text(json.dumps(doc, indent=2))
print("wrote", out / "campaign_results.json")
PY
echo DONE
