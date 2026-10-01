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
# the container SEES the host's CPUs (nproc 288 on session 3's box) while the pod is allotted a quota:
# cpu.max is the quota, cpu.stat counts the throttling; a CPU-side decode that oversubscribes the quota is throttled
echo "cgroup cpu.max: $(cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo n/a)"; grep -E "^(nr_periods|nr_throttled|throttled_usec) " /sys/fs/cgroup/cpu.stat 2>/dev/null | tr "\n" " "; echo
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS:-unset} MKL_NUM_THREADS=${MKL_NUM_THREADS:-unset} torch threads: $(python3 -c "import torch; print(torch.get_num_threads())" 2>/dev/null)"
echo "== filesystems"
findmnt -T "$VERINF_ROOT" -o SOURCE,FSTYPE,TARGET,SIZE,AVAIL 2>&1 || df -h "$VERINF_ROOT"
findmnt -T / -o SOURCE,FSTYPE,TARGET,SIZE,AVAIL 2>&1 || true
[ -d /workspace ] && { findmnt -T /workspace -o SOURCE,FSTYPE,TARGET,SIZE,AVAIL 2>&1 || true; }
df -h "$VERINF_ROOT" / 2>&1 || true
echo "== revision"; git rev-parse HEAD; git status --short | head -20
echo "== packages"; pip list 2>/dev/null | grep -i -E '^(torch|numpy|blake3|gguf|safetensors|ninja|transformers|hf.transfer) ' || true
