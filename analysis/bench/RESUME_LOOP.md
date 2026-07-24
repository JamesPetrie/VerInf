# Resume the prover-autoresearch loop in tmux (survives disconnection)

The loop's state is all on disk (research_journal.md, metric_ledger.md,
prove_runs.jsonl, code). A cold restart resumes from exactly where it left off.
To make it survive a 12h disconnect, run claude inside tmux (which outlives the
VSCode-server terminal and SSH).

## One-time launch (run in a terminal on this host)

```bash
tmux new -s prover        # start a detachable session
# inside tmux:
claude                    # start Claude Code interactively
# then paste the RESUME PROMPT below as your first message
# once it's running and has armed a ScheduleWakeup, detach with:  Ctrl-b then d
```

Re-attach any time to watch progress:  `tmux attach -t prover`
Or just check state without attaching:
```bash
tail -40 /home/riftuser/VerInf/analysis/bench/research_journal.md
uv run --project /home/riftuser/VerInf python3 /home/riftuser/VerInf/analysis/bench/show_runs.py
```

## RESUME PROMPT (paste as the first message to the tmux'd claude)

Continue the prover-autoresearch loop (autonomous, user away). Read
/home/riftuser/VerInf/.claude/skills/prover-autoresearch/SKILL.md and FOLLOW IT.
Orient via analysis/bench/research_journal.md (newest entry is the plan),
metric_ledger.md, and analysis/toy-transformer-prove-time-formula.md. Do ONE
honest validated step toward the journal's "Next angle", record it per SKILL
§5, then ScheduleWakeup (~900s to continue, ~3600s if near a usage limit),
passing this same prompt back. HARD RULES: no fabricated numbers (every speedup
= a logged CUDA-event A/B); ACCEPT is the gate (any real-prover change must keep
Rust verify_proof=ACCEPT on a small toy proof + a bit-exact diff test + fast
suite green, else revert); keep validation fast; a measured dead-end is a valid
result; NO git/commits. Env: export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH";
run python via `uv run --project /home/riftuser/VerInf python3`.

## Note
If you relaunch in tmux, THIS current (non-tmux) session's armed wakeup is
superseded — a dead process can't fire wakeups, and if it somehow survives,
two loops touching the same files/GPU would collide, so only run one. Kill the
old session first if it's still attached.
