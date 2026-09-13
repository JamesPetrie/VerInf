#!/bin/bash
# The environment record for a rented session (read-only): versions,
# device, memory limits, the filesystems under the chosen paths, and the
# revision. Prints; the runbook captures it into session2-env.txt. Needs
# VERINF_ROOT and the other VERINF_* exports from the bootstrap.
set -u
: "${VERINF_ROOT:?set by the bootstrap}"
cd "$(dirname "$0")/.."
echo "== $(date -u +%Y-%m-%dT%H:%M:%SZ) on $(hostname)"
echo "== paths"; env | grep '^VERINF_' | sort
echo "== python / torch"
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0), 'capability', torch.cuda.get_device_capability(0))" 2>&1 || echo "torch: unavailable"
echo "== nvcc"; nvcc --version 2>&1 | tail -1 || true
echo "== gpu"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>&1 || echo "nvidia-smi: unavailable"
echo "== host"; free -g; echo "cgroup memory.max: $(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo n/a)"; nproc
echo "== filesystems"
findmnt -T "$VERINF_ROOT" -o SOURCE,FSTYPE,TARGET,SIZE,AVAIL 2>&1 || df -h "$VERINF_ROOT"
findmnt -T / -o SOURCE,FSTYPE,TARGET,SIZE,AVAIL 2>&1 || true
[ -d /workspace ] && { findmnt -T /workspace -o SOURCE,FSTYPE,TARGET,SIZE,AVAIL 2>&1 || true; }
df -h "$VERINF_ROOT" / 2>&1 || true
echo "== revision"; git rev-parse HEAD; git status --short | head -20
echo "== packages"; pip list 2>/dev/null | grep -i -E '^(torch|numpy|blake3|gguf|safetensors|ninja|transformers|hf.transfer) ' || true
