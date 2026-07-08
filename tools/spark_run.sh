#!/bin/bash
# Reliable detached launcher for long Spark runs.  Usage:
#   tools/spark_run.sh <name> <command...>
# Writes ~/<name>.log (truncated at launch — stale reads impossible),
# ~/<name>.pid (from $!, never from process-name matching), a memory
# trajectory in ~/<name>.memwatch, and ALWAYS a final "EXIT=<code>" log line,
# so silence can never masquerade as progress.
#
# KILL RULE: stop a run with `kill $(cat ~/<name>.pid)`.  NEVER `pkill -f` —
# the pattern matches the invoking ssh shell's own command line and kills the
# launcher instead of (or as well as) the run; this bug cost us three silent
# launch failures on 2026-06-11.
set -u
NAME=$1; shift
LOG=~/$NAME.log
: > "$LOG"
echo "LAUNCH $(date +%H:%M:%S) cmd: $*" >> "$LOG"
( "$@" >> "$LOG" 2>&1; echo "EXIT=$?" >> "$LOG" ) &
PID=$!
echo $PID > ~/$NAME.pid
( while kill -0 $PID 2>/dev/null; do
    read -r _ TOT USED FREE _ CACHE AVAIL <<< $(free -m | sed -n 2p)
    RSS=$(ps -o rss= -p $PID 2>/dev/null)
    echo "$(date +%H:%M:%S) rss_kb=${RSS:-gone} used_mb=$USED avail_mb=$AVAIL" >> ~/$NAME.memwatch
    sleep 15
  done
  echo "$(date +%H:%M:%S) pid $PID exited" >> ~/$NAME.memwatch ) &
echo "launched pid $PID (log ~/$NAME.log, pidfile ~/$NAME.pid)"
