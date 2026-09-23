#!/bin/bash
# The CPU gate: every suite that runs without a card, and the Rust verifier's
# tests. Required before a GPU session (profiler/RUNBOOK-blackwell.md, before
# phase 0): a regression a laptop can catch must not cost pod time. A missing
# dependency, a missing suite, a failure or a skip in any listed suite fails
# the gate, loudly; nothing is silently left out.
#   PY=<python with torch, numpy, blake3, pytest, gguf> tools/cpu_gates.sh
# Suites that need CUDA are the GPU gates' (tools/session2_gates.sh and the
# runbook's early suites); a suite joins this list only if all of it runs on
# a CPU-only torch.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python3}
LOG=${CPU_GATE_LOG:-/tmp/verinf-cpu-gate.log}

SUITES=(
  prover/tests/test_aes_trace.py
  prover/tests/test_demo_bridge_policy.py
  prover/tests/test_gguf_loader.py
  prover/tests/test_instrument_bookkeeping.py
  prover/tests/test_layout_breakdown.py
  prover/tests/test_merkle_index.py
  prover/tests/test_row_map.py
  prover/tests/test_sampled_audit_prototype_gate.py
  prover/tests/test_sampled_local_transcript.py
  prover/tests/test_sha256_trace.py
  prover/tests/test_shard_plan.py
  prover/tests/test_shard_worker.py
  prover/tests/test_statement_stability.py
  prover/tests/test_token_recorder.py
  prover/tests/test_wc_bridge_cpu.py
  prover/tests/test_wc_identity.py
  prover/tests/test_weight_provenance.py
  layergkr/tests/test_count_model.py
  layergkr/tests/test_full_layer.py
  layergkr/tests/test_gpu.py
  layergkr/tests/test_layer.py
  layergkr/tests/test_logup.py
  layergkr/tests/test_moe.py
  layergkr/tests/test_projection.py
  layergkr/tests/test_relations_transcript.py
  layergkr/tests/test_rs.py
  layergkr/tests/test_sampled_audit.py
  layergkr/tests/test_sampled_audit_cost_model.py
  layergkr/tests/test_semantics.py
  layergkr/tests/test_sumcheck.py
  profiler/test_calibration_tools.py
  profiler/test_hbm_bench.py
  profiler/test_profiler.py
)

fail() { echo "CPU GATE FAILED: $*"; exit 1; }

echo "== dependencies ($PY)"
command -v "$PY" >/dev/null || fail "no interpreter '$PY' (set PY)"
missing=$("$PY" -c '
import importlib.util
need = ["torch", "numpy", "blake3", "pytest", "gguf"]
print(" ".join(m for m in need if importlib.util.find_spec(m) is None))') \
  || fail "the interpreter '$PY' does not run"
[ -z "$missing" ] || fail "missing Python modules: $missing (install them for $PY)"
command -v cargo >/dev/null || fail "cargo not found (the Rust verifier's tests)"
for s in "${SUITES[@]}"; do [ -f "$s" ] || fail "listed suite $s does not exist"; done

echo "== python: ${#SUITES[@]} suites (log: $LOG)"
CUDA_VISIBLE_DEVICES= "$PY" -m pytest -q -p no:cacheprovider -rs "${SUITES[@]}" \
  > "$LOG" 2>&1
rc=$?
tail -1 "$LOG"
[ "$rc" -eq 0 ] || { grep -E '^(FAILED|ERROR)' "$LOG"; fail "python suites (rc=$rc)"; }
if grep -qE '[0-9]+ skipped' "$LOG"; then
  grep -E '^SKIPPED' "$LOG"
  fail "a CPU suite skipped tests: the gate checks everything it lists"
fi

echo "== rust: the verifier's tests"
(cd verifier && RUSTC_WRAPPER= cargo test --release -q 2>&1) > "$LOG.rust" \
  || { tail -30 "$LOG.rust"; fail "cargo test"; }
grep -E '^test result' "$LOG.rust" | awk '{p += $4; f += $6} END {print p " passed, " f " failed"}'

echo "== CPU GATE PASSED"
