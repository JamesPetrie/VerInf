# Session 10, stage 1: phase 0 (pip, the shard-1 pull started in the
# background, rustup, the environment record, the verifier built from this
# revision, the CUDA primitives), the suites this tree touches, then the toy
# K2 driver suite (gated) and the Maverick binding probe (recorded, not
# gated: its last two tests are expected to fail if the gap is real).
set -u
source /workspace/session10.env
fail() { echo "STAGE1 FAILED at $* $(date -u +%H:%M:%S)"; exit 1; }
echo "STAGE1 START $(cat /workspace/session10-rev) $(date -u +%H:%M:%S)"
pip install --break-system-packages -q blake3 gguf safetensors ninja hf_transfer huggingface_hub pytest \
  > $VERINF_LOGS/pip-install.log 2>&1 || { tail -20 $VERINF_LOGS/pip-install.log; fail "pip install"; }
setsid nohup bash /workspace/pull10.sh > $VERINF_LOGS/pull.log 2>&1 < /dev/null &
echo "pull started $(date -u +%H:%M:%S)"
if ! command -v cargo >/dev/null; then
  curl -sSf https://sh.rustup.rs -o $VERINF_ROOT/rustup-init.sh || fail "rustup download"
  sh $VERINF_ROOT/rustup-init.sh -y --profile minimal > $VERINF_LOGS/rustup.log 2>&1 || fail "rustup"
fi
[ -f ~/.cargo/env ] && . ~/.cargo/env
command -v cargo >/dev/null || fail "cargo not on PATH"
nohup tools/host_sampler.sh $VERINF_LOGS/sampler.log > /dev/null 2>&1 &
logrun env-record tools/session2_env.sh > /dev/null && cp $VERINF_LOGS/env-record.log $VERINF_REC/session10-env.txt \
  && logrun cargo cargo build --release --manifest-path verifier/Cargo.toml --bin verify_proof > /dev/null \
  && sha256sum verifier/target/release/verify_proof > $VERINF_REC/session10-verifier.sha256 \
  && rustc --version > $VERINF_REC/session10-rustc.txt \
  && logrun cuda-primitives python3 prover/tests/run_tests.py test_cuda_primitives > /dev/null \
  && echo "$(cat /workspace/session10-rev) (tarball of the commit, no .git on the pod; verifier built from it)" > $VERINF_REC/session10-revision.txt \
  || fail "bootstrap"
pip freeze > $VERINF_REC/session10-pins.txt || fail "pip freeze"
echo "== bootstrap: $(tail -1 $VERINF_LOGS/cuda-primitives.log); verifier $(cut -c1-16 $VERINF_REC/session10-verifier.sha256); $(cat $VERINF_REC/session10-rustc.txt)"
echo "PHASE0 DONE $(date -u +%H:%M:%S)"
for t in test_claims test_protocol_review_negatives test_topk_routing test_head_interleave test_rope_scaling test_k2_driver; do
  logrun $t python3 prover/tests/run_tests.py $t > /dev/null 2>&1; rc=$?
  echo "$t rc=$rc :: $(grep -o '=== .* passed, .* failed[^=]*' $VERINF_LOGS/$t.log | tail -1) $(date -u +%H:%M:%S)"
  [ "$rc" -eq 0 ] || fail "$t"
done
grep -E 'demo_k2 .*(EXACT|negative|PROVE|rust verify|ranges)' $VERINF_LOGS/test_k2_driver.log | sed 's/^/    /'
logrun maverick-binding-probe python3 prover/tests/run_tests.py test_maverick_binding_negatives > /dev/null 2>&1
echo "maverick-binding-probe rc=$? (ungated) :: $(grep -o '=== .* passed, .* failed[^=]*' $VERINF_LOGS/maverick-binding-probe.log | tail -1)"
grep -E '^\s+\[binding\]|^  (PASS|FAIL)' $VERINF_LOGS/maverick-binding-probe.log | sed 's/^/    /'
echo "pull: $(tail -3 $VERINF_LOGS/pull.log | tr '\n' ' ')"
echo "STAGE1 DONE $(date -u +%H:%M:%S)"
