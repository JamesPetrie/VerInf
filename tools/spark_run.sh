#!/bin/bash
# Reliable detached launcher for long runs.  Usage:
#   tools/spark_run.sh <name> <command...>
#
# Writes, under $HOME:
#   <name>.log       the job's stdout+stderr (truncated at launch) and ALWAYS a
#                    final "EXIT=<code>" line — "EXIT=killed ..." when the job
#                    was cancelled — so silence can never masquerade as progress
#   <name>.pid       the supervisor's PID, which is also the PROCESS GROUP id of
#                    the whole job (the supervisor runs in its own session)
#   <name>.job       the PID of the command itself
#   <name>.memwatch  RSS of the job's whole process group every 15 s
#
# The job is started in a new session (setsid): the SSH connection's death
# cannot HUP it, and its stdio never touches the launching terminal. The
# supervisor collects the job's exit status and writes the EXIT line even
# when the group is killed, because it traps the signal and waits for the
# job to be gone before it exits.
#
# KILL RULE:  kill -- -$(cat ~/<name>.pid)     (the whole process group)
# `kill $(cat ~/<name>.pid)` alone reaches only the supervisor and leaves
# the job running. NEVER `pkill -f`: the pattern matches the invoking ssh
# shell's own command line and kills the launcher (this cost three silent
# launch failures on 2026-06-11).
#
# A name whose process group is still alive is refused, so a re-launch can
# never overwrite a live run's log and pidfile.
set -u
NAME=$1; shift
LOG=~/$NAME.log; PIDF=~/$NAME.pid; JOBF=~/$NAME.job; MEMF=~/$NAME.memwatch
if [ -f "$PIDF" ] && kill -0 -- -"$(cat "$PIDF")" 2>/dev/null; then
  echo "refusing to launch: $NAME is still running (pgid $(cat "$PIDF")); kill -- -$(cat "$PIDF") first" >&2
  exit 1
fi
rm -f "$PIDF" "$JOBF"
: > "$LOG"; : > "$MEMF"
echo "LAUNCH $(date +%H:%M:%S) cmd: $*" >> "$LOG"
setsid bash -c '
  LOG=$1; PIDF=$2; JOBF=$3; MEMF=$4; shift 4
  echo $$ > "$PIDF"                        # $$ is the session/group leader
  KILLED=
  trap "KILLED=1" TERM INT HUP
  "$@" >> "$LOG" 2>&1 < /dev/null &
  JOB=$!
  echo $JOB > "$JOBF"
  ( while kill -0 "$JOB" 2>/dev/null; do
      read -r _ _ USED _ _ _ AVAIL <<< "$(free -m | sed -n 2p)"
      RSS=$(ps -o rss= -g "$$" 2>/dev/null | awk "{s+=\$1} END {print s+0}")
      echo "$(date +%H:%M:%S) group_rss_kb=$RSS used_mb=$USED avail_mb=$AVAIL" >> "$MEMF"
      sleep 15
    done
    echo "$(date +%H:%M:%S) job $JOB exited" >> "$MEMF" ) > /dev/null 2>&1 < /dev/null &
  wait "$JOB"; RC=$?
  if [ -n "$KILLED" ]; then
    # the signal interrupted wait; the job got the same signal (whole group)
    while kill -0 "$JOB" 2>/dev/null; do sleep 1; done
    wait "$JOB" 2>/dev/null; RC2=$?; [ "$RC2" -ne 127 ] && RC=$RC2
    echo "EXIT=killed (job rc=$RC) $(date +%H:%M:%S)" >> "$LOG"
  else
    echo "EXIT=$RC" >> "$LOG"
  fi
' _ "$LOG" "$PIDF" "$JOBF" "$MEMF" "$@" > /dev/null 2>&1 < /dev/null &
for _ in $(seq 1 50); do [ -s "$PIDF" ] && [ -s "$JOBF" ] && break; sleep 0.1; done
echo "launched pgid $(cat "$PIDF" 2>/dev/null) job $(cat "$JOBF" 2>/dev/null) (log ~/$NAME.log; kill -- -\$(cat ~/$NAME.pid))"
