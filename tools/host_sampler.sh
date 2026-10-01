#!/bin/bash
# tools/host_sampler.sh <out.log> [interval_s]: one block every interval (default
# 20 s) until killed — the host and cgroup state a timed arm ran under, so a
# slow phase can be placed against CPU throttling (cpu.stat nr_throttled and
# throttled_usec rising), memory reclaim (memory.stat file falling while anon
# or shmem rises; memory.pressure), and IO stalls (io.pressure). Session 3's
# on arm built in 303 s against 41.8 s and its R1 stores took 1.9 s each; the
# logs could not say which of these it was. Start it at bootstrap:
#   nohup tools/host_sampler.sh $VERINF_LOGS/sampler.log > /dev/null 2>&1 &
# and copy the log home with the arms.
OUT=${1:?usage: host_sampler.sh <out.log> [interval_s]}; INT=${2:-20}
CG=/sys/fs/cgroup
while :; do
  {
    echo "== $(date -u +%H:%M:%S)"
    free -m | sed -n 2p
    awk '/^(anon|file|shmem|file_mapped|unevictable) /{printf "%s %d MB; ", $1, $2/1048576} END{print ""}' $CG/memory.stat 2>/dev/null
    echo "memory.current $(( $(cat $CG/memory.current 2>/dev/null || echo 0) / 1048576 )) MB"
    grep -E "^(nr_periods|nr_throttled|throttled_usec) " $CG/cpu.stat 2>/dev/null | tr '\n' ' '; echo
    echo "cpu.pressure $(sed -n 1p $CG/cpu.pressure 2>/dev/null); memory.pressure $(sed -n 1p $CG/memory.pressure 2>/dev/null); io.pressure $(sed -n 1p $CG/io.pressure 2>/dev/null)"
    nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null
  } >> "$OUT"
  sleep "$INT"
done
