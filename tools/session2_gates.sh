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
# The decoded-weight cache: byte identity, Rust ACCEPT, one decode per
# dense weight per proof, shards untouched, zero budget falls back.
gate weight-cache python3 prover/tests/run_tests.py test_weight_cache
need weight-cache '=== 5 passed, 0 failed'
gate instrument-bookkeeping python3 prover/tests/run_tests.py test_instrument_bookkeeping
need instrument-bookkeeping '=== 3 passed, 0 failed'
# The collaborator's weight bridge, merged 2026-09-17: his suites (the
# standalone bridge, the flag inside the 5-round transcript, the tape link)
# and OUR negatives through the Rust binary (a tampered fold, the enrollment
# root wrong, missing or in the weight-root slot, q_w below the floor, the
# legacy seed path, a stripped wc section, and the production shape with a
# weight block beside the wc section).
gate wc-bridge python3 prover/tests/run_tests.py test_wc_bridge
need wc-bridge '=== 15 passed, 0 failed'
gate wc-streaming-flag python3 prover/tests/run_tests.py test_wc_streaming_flag
need wc-streaming-flag '=== 9 passed, 0 failed'
gate wc-tape-link python3 prover/tests/run_tests.py test_wc_tape_link
need wc-tape-link '=== 1 passed, 0 failed'
gate wc-bridge-rust python3 prover/tests/run_tests.py test_wc_bridge_rust_negatives
need wc-bridge-rust '=== 6 passed, 0 failed'
gate toy-ab env LIGERO_SWEEP_TIMING=1 python3 analysis/bench/ab_routed_cache.py
need toy-ab 'ab_routed_cache: counts as expected'
# A small REAL-GGUF proof through the research driver, independently
# verified: the leaf check is fail-closed in the prover, and the Rust
# verifier establishes the algebraic constraints.
gate driver-small python3 profiler/instrumented_prove.py --from-gguf "$VERINF_GGUF" \
    --t-queries 54 --prompt-n 2 --cont-n 2 --layers 2 --sweep-timing --verify
need driver-small 'rust verify_proof: ACCEPT'
need driver-small 'opened columns match committed leaves: True'
# The same two-layer proof with the decoded-weight cache and the group memo
# on, independently verified: the real loaders through the cache.
gate driver-small-wc python3 profiler/instrumented_prove.py --from-gguf "$VERINF_GGUF" \
    --t-queries 54 --prompt-n 2 --cont-n 2 --layers 2 --sweep-timing --weight-cache --verify
need driver-small-wc 'rust verify_proof: ACCEPT'
need driver-small-wc 'opened columns match committed leaves: True'
need driver-small-wc 'weights decoded once'
# The same two-layer proof under the collaborator's bridge: the expert
# shards leave the witness, the streaming enrollment authenticates them,
# the Rust verifier binds both anchors (weight root and enrollment root).
gate driver-small-bridge python3 profiler/instrumented_prove.py --from-gguf "$VERINF_GGUF" \
    --t-queries 54 --prompt-n 2 --cont-n 2 --layers 2 --sweep-timing --wc-bridge --verify
need driver-small-bridge 'rust verify_proof: ACCEPT'
need driver-small-bridge 'opened columns match committed leaves: True'
need driver-small-bridge 'wc enrollment (streaming coefficient-RS'
need driver-small-bridge '\[wc-bridge\] stages:'
need driver-small-bridge "the bridge's per-proof pass took"
echo "== all gates passed"
