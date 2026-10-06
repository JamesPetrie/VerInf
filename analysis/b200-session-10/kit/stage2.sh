# Session 10, stage 2: shard 1 checked, then the exact checks on the real
# weights (integer reference first, recording every range, then the engine
# pass compared name by name). S=100 from 0 first: if it fails, stop.
set -u
source /workspace/session10.env
fail() { echo "STAGE2 FAILED at $* $(date -u +%H:%M:%S)"; exit 1; }
echo "STAGE2 START $(date -u +%H:%M:%S)"
until grep -q '^EXIT=' $VERINF_LOGS/pull.log 2>/dev/null; do sleep 20; done
grep -qx 'EXIT=0' $VERINF_LOGS/pull.log || fail "pull: $(tail -2 $VERINF_LOGS/pull.log | tr '\n' ' ')"
echo "shard 1: $(grep -E '^(pull|sha256)' $VERINF_LOGS/pull.log | tr '\n' ' ')"
run_check() {   # run_check <tag> <prompt_n> <cont_n> <offset>
  logrun check-$1 python3 demo/demo_k2.py --mode check --from-gguf "$VERINF_GGUF" \
    --prompt-n $2 --cont-n $3 --offset $4 --record $VERINF_REC/k2-check-$1.json > /dev/null 2>&1
  local rc=$?
  echo "check-$1 rc=$rc :: $(grep -E 'RESULT|EXACT|ranges:|STOPPED' $VERINF_LOGS/check-$1.log | sed 's/^\[demo_k2 [^]]*\] //' | tr '\n' ' ') $(date -u +%H:%M:%S)"
  return $rc
}
run_check s100 50 50 0 || fail "check-s100"
run_check s100-long 50 50 130072
run_check s1000 500 500 0
run_check s1000-long 500 500 130072
echo "STAGE2 DONE $(date -u +%H:%M:%S)"
