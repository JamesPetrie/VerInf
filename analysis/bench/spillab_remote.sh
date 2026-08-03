#!/usr/bin/env bash
# Spill A/B on the rented fast-NVMe box: measures real disk BW + recompute throughput
# + times no-spill/host-spill/disk-spill (fadvise -> re-reads hit the NVMe, not cache).
set -x
cd /workspace/VerInf
export VERINF_ROOT=/workspace/VerInf
export VERINF_PYRUN="uv run --project /workspace/VerInf python3"
# GATE: validate machine vs offer (GPU/VRAM/RAM/CPU/disk) before spending on prove.
export PF_DISK_DIR=/workspace/spill PF_MIN_READ_GBPS="${AB_MIN_READ_GBPS:-2.5}"
source analysis/bench/preflight_gate.sh
RESDIR=/workspace/VerInf/analysis/bench/remote_results/optrun
mkdir -p "$RESDIR" /workspace/spill
echo "=== disk under /workspace ==="; df -h /workspace; mount | grep -E "workspace| / " | head
PYRUN="uv run --project /workspace/VerInf python3"
echo "########## SPILL A/B (fadvise, real NVMe) d=$AB_D seq=$AB_SEQ L=$AB_NL ##########"
LIGERO_WITNESS_SPILL_DIR=/workspace/spill AB_RESULT="$RESDIR/spill_ab_result.json" \
  $PYRUN analysis/bench/spill_ab.py 2>&1 | tee "$RESDIR/spillab_run.log"   # tee -> survives box destroy
PV=${PIPESTATUS[0]}
printf '{"spill_ab_exit": %d}\n' "$PV" > "$RESDIR/campaign_results.json"
echo "=== RESULT JSON ==="; cat "$RESDIR/spill_ab_result.json" 2>/dev/null
echo "########## SPILL A/B DONE (exit=$PV) ##########"
