# Session 10, stage 1b: the toy driver suite at the corrected revision (gated),
# the Maverick binding probe (recorded, not gated), then stage 2 if the suite
# passed. Phase 0 and the other suites ran in stage 1; shard 1 is in place.
set -u
source /workspace/session10.env
fail() { echo "STAGE1B FAILED at $* $(date -u +%H:%M:%S)"; exit 1; }
echo "STAGE1B START $(cat /workspace/session10-rev) $(date -u +%H:%M:%S)"
echo "$(cat /workspace/session10-rev) shipped for stage 1b (toy suites at ELL 64); verifier/ is unchanged since ded16f2, the binary is stage 1's" >> $VERINF_REC/session10-revision.txt
logrun test_k2_driver python3 prover/tests/run_tests.py test_k2_driver > /dev/null 2>&1; rc=$?
echo "test_k2_driver rc=$rc :: $(grep -o '=== .* passed, .* failed[^=]*' $VERINF_LOGS/test_k2_driver.log | tail -1) $(date -u +%H:%M:%S)"
grep -E 'EXACT|negative|PROVE|rust verify|ranges:|^  (PASS|FAIL)' $VERINF_LOGS/test_k2_driver.log | sed 's/^\[demo_k2 [^]]*\] //; s/^/    /'
[ "$rc" -eq 0 ] || fail "test_k2_driver"
logrun maverick-binding-probe python3 prover/tests/run_tests.py test_maverick_binding_negatives > /dev/null 2>&1
echo "maverick-binding-probe rc=$? (ungated) :: $(grep -o '=== .* passed, .* failed[^=]*' $VERINF_LOGS/maverick-binding-probe.log | tail -1)"
grep -E '\[binding\]|^  (PASS|FAIL)' $VERINF_LOGS/maverick-binding-probe.log | sed 's/^/    /'
echo "STAGE1B DONE $(date -u +%H:%M:%S)"
bash /workspace/stage2.sh
