#!/bin/bash
# Phase G of profiler/RUNBOOK-blackwell.md: the gates before any timed
# prove. Every gate must pass, in order; the FIRST failure stops the script
# and is its exit status, and a pipeline cannot hide it. Each gate's output
# goes to $VERINF_LOGS/gate-<name>.log. Needs VERINF_GGUF, VERINF_GGUF_DIR,
# VERINF_LOGS (the runbook's bootstrap exports) and the built verifier.
set -euo pipefail
: "${VERINF_GGUF:?set by the bootstrap}" "${VERINF_GGUF_DIR:?}" "${VERINF_LOGS:?}"
cd "$(dirname "$0")/.."
mkdir -p "$VERINF_LOGS"
gate() {   # gate <name> <cmd...>: run, log, and stop on failure
  local name=$1; shift
  echo "== gate: $name"
  "$@" 2>&1 | tee "$VERINF_LOGS/gate-$name.log"
  local rc=${PIPESTATUS[0]}
  [ "$rc" -eq 0 ] || { echo "GATE FAILED: $name (rc=$rc)"; exit "$rc"; }
}
need() {   # need <name> <pattern>: the gate's log must contain the line
  grep -q -- "$2" "$VERINF_LOGS/gate-$1.log" \
    || { echo "GATE FAILED: $1 did not print '$2'"; exit 1; }
}
# The K-quant kernel against the real shards. Its check is main(), not
# test_* functions, so the file runs directly; it prints four [OK ] lines
# and the summary below (a run through run_tests.py finds nothing).
gate kquant python3 prover/tests/test_kquant_kernel.py "$VERINF_GGUF_DIR/UD-Q4_K_XL"
need kquant '=== kquant_kernel: 4/4 PASS ==='
gate shard-streaming python3 prover/tests/run_tests.py test_shard_streaming
need shard-streaming '=== 7 passed, 0 failed'
gate routed-projected python3 prover/tests/run_tests.py test_routed_projected
need routed-projected '=== 6 passed, 0 failed'
gate toy-ab env LIGERO_SWEEP_TIMING=1 python3 analysis/bench/ab_routed_cache.py
need toy-ab 'ab_routed_cache: counts as expected'
# A small REAL-GGUF proof through the research driver, independently
# verified: the leaf check is fail-closed in the prover, and the Rust
# verifier establishes the algebraic constraints.
gate driver-small python3 profiler/instrumented_prove.py --from-gguf "$VERINF_GGUF" \
    --t-queries 54 --prompt-n 2 --cont-n 2 --layers 2 --sweep-timing --verify
need driver-small 'rust verify_proof: ACCEPT'
need driver-small 'opened columns match committed leaves: True'
echo "== all gates passed"
