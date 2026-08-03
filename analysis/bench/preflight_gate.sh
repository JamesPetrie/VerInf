#!/usr/bin/env bash
# Mandatory pre-flight gate. Source this at the TOP of every remote workload script,
# BEFORE any paid compute. It validates the machine against the advertised offer
# (PF_* env, set by the runner). On mismatch it writes PREFLIGHT_FAIL and exits the
# WHOLE workload script (so the runner tears the box down) — no experiment ever runs
# on a machine whose real specs don't match what we rented.
#   usage:  source analysis/bench/preflight_gate.sh   (from /workspace/VerInf)
PYRUN="${VERINF_PYRUN:-uv run --project /workspace/VerInf python3}"
echo "########## PRE-FLIGHT GATE ##########"
if $PYRUN analysis/bench/preflight.py; then
    echo "########## PRE-FLIGHT OK — running workload ##########"
else
    echo "PREFLIGHT_FAIL" | tee -a /workspace/PREFLIGHT_FAIL
    echo "########## PRE-FLIGHT FAILED — machine != offer, NOT running workload ##########"
    exit 4
fi
