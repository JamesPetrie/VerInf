#!/usr/bin/env bash
# REAL model + REAL card: Maverick MoE layer, E=64 (real-ish expert count),
# d=5120 (real width), DISK-backed full-witness spill, optimized rho=2/T=17.
# Confirms the full optimized stack ACCEPTs on rented hardware. Verify only the
# optimized run (cost control; ACCEPT already established at E=8 locally).
set -x
cd /workspace/VerInf
export VERINF_ROOT=/workspace/VerInf
RESDIR=/workspace/VerInf/analysis/bench/remote_results/optrun
mkdir -p "$RESDIR" /workspace/spill
df -h /workspace
PYRUN="uv run --project /workspace/VerInf python3"
export VERINF_PYRUN="$PYRUN"
# GATE: validate machine vs offer before spending on the optimized prove+verify.
export PF_DISK_DIR=/workspace/spill
source analysis/bench/preflight_gate.sh

echo "########## REAL-CARD Maverick MoE E=64 d=5120 + DISK-SPILL + opt rho=2/T=17 ##########"
MOE_E=64 MOE_SEQ=4 MOE_D=5120 MOE_DFF=8192 MOE_DISK=1 MOE_VERIFY=opt \
  LIGERO_WITNESS_SPILL_DIR=/workspace/spill \
  $PYRUN analysis/bench/optrun_moe.py
BIG=$?
cp -f optrun_results.json "$RESDIR/optrun_moe_e64.json" 2>/dev/null || true
printf '{"e64_exit": %d}\n' "$BIG" > "$RESDIR/campaign_results.json"
echo "########## OPTRUN ALL DONE (e64_exit=$BIG) ##########"
