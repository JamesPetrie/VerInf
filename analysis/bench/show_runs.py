#!/usr/bin/env python3
"""Browse the persistent prove-run log without holding it in chat context.

    python3 show_runs.py                      # table of every run
    python3 show_runs.py --kind coset_ntt_ab   # filter by kind
    python3 show_runs.py --last 10             # most recent N
    python3 show_runs.py --label "d512"        # substring match on label
    python3 show_runs.py --full <index>        # full JSON for one row (index from the table)
"""
from __future__ import annotations

import argparse
import json
import sys

from run_log import load_runs


def _fmt(v, width=None):
    if isinstance(v, float):
        s = f"{v:.3g}"
    else:
        s = str(v)
    if width:
        s = s[:width]
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default=None)
    ap.add_argument("--label", default=None, help="substring match")
    ap.add_argument("--last", type=int, default=None)
    ap.add_argument("--full", type=int, default=None, help="print full JSON for row N")
    args = ap.parse_args()

    runs = load_runs(kind=args.kind)
    if args.label:
        runs = [r for r in runs if args.label.lower() in r["label"].lower()]
    if args.last:
        runs = runs[-args.last:]

    if not runs:
        print("no matching runs logged yet", file=sys.stderr)
        return

    if args.full is not None:
        print(json.dumps(runs[args.full], indent=2))
        return

    # "size" and "headline" pull from whichever field this run's `kind` actually
    # populated (prove sweeps use m_total/prove_s; A/B primitive benches use
    # n_elements/speedup; etc.) -- --full always has the complete record.
    SIZE_KEYS = ["m_total", "n_elements"]
    HEADLINE_KEYS = ["prove_s", "speedup", "predicted_total_s"]

    def _first(d, keys):
        for k in keys:
            if k in d:
                return d[k]
        return ""

    cols = ["#", "ts", "kind", "label", "size", "headline", "notes"]
    widths = [3, 19, 16, 26, 10, 9, 30]
    print("  ".join(c.ljust(w) for c, w in zip(cols, widths)))
    print("  ".join("-" * w for w in widths))
    for i, r in enumerate(runs):
        m_total = _first(r["params"], SIZE_KEYS) or _first(r["measured"], SIZE_KEYS)
        prove_s = _first(r["measured"], HEADLINE_KEYS)
        row = [str(i), r["ts"][:19], r["kind"], r["label"],
               _fmt(m_total), _fmt(prove_s), r.get("notes", "")]
        print("  ".join(_fmt(v, w).ljust(w) for v, w in zip(row, widths)))
    print(f"\n{len(runs)} run(s). Use --full <#> for the complete record "
          f"(params/measured/predicted/git commit).")


if __name__ == "__main__":
    main()
