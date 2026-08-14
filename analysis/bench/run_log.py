"""Persistent, append-only log of prove() benchmark runs.

Every run of a toy/medium-scale benchmark (this session's formula-vs-reality
investigation, and any future one) should call `log_run(...)` instead of just
printing to stdout, so results survive past a single conversation and can be
compared later without re-running anything or holding them in chat context.

Storage: one JSON object per line in `prove_runs.jsonl`, next to this file.
Append-only -- never rewritten in place, so concurrent/interrupted runs can't
corrupt earlier entries. Read with `load_runs()`, browse with `show_runs.py`.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

LOG_PATH = Path(__file__).resolve().parent / "prove_runs.jsonl"


def _git_info() -> Dict[str, Optional[str]]:
    def _run(args):
        try:
            return subprocess.run(args, cwd=Path(__file__).resolve().parents[2],
                                   capture_output=True, text=True, timeout=5
                                   ).stdout.strip() or None
        except Exception:
            return None
    return {
        "git_branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "git_commit": _run(["git", "rev-parse", "--short", "HEAD"]),
    }


def log_run(kind: str, label: str, params: Dict[str, Any], measured: Dict[str, Any],
            predicted: Optional[Dict[str, Any]] = None, notes: str = "",
            log_path: Path = LOG_PATH) -> Dict[str, Any]:
    """Append one run record.

    kind: short category, e.g. "prove_sweep", "coset_ntt_ab", "calculator_check".
    label: human-readable run name, e.g. "d512,ff2048,seq1024,L4".
    params: everything needed to reproduce the run (model dims, LigeroConfig
        fields, hardware) -- flat dict of JSON-serializable values.
    measured: actual measured outputs (prove_s, witness_s, verify_s, ...).
    predicted: optional -- what a formula/calculator predicted for the same
        params, so measured-vs-predicted comparisons don't need re-derivation.
    notes: free text, e.g. why this run happened / what it's testing.
    """
    entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "kind": kind,
        "label": label,
        "params": params,
        "measured": measured,
        "predicted": predicted,
        "notes": notes,
        **_git_info(),
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def load_runs(log_path: Path = LOG_PATH, kind: Optional[str] = None) -> List[Dict[str, Any]]:
    """All logged runs, optionally filtered by `kind`. Empty list if the log
    doesn't exist yet (no error -- a fresh checkout has no history)."""
    if not log_path.exists():
        return []
    runs = []
    for line in log_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if kind is None or r.get("kind") == kind:
            runs.append(r)
    return runs
