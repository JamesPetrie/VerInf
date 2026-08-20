"""Dependency-DAG export and parallelism summary.

Nodes are claims; an edge A->B exists when B consumes a variable A produced.
Persistent (weight) variables don't create edges — they're run inputs the
scheduler streams in, not dataflow dependencies. This graph plus the
streaming-sweep barriers IS the schedulable structure: four sweeps (R1,
R2, test polynomials, openings), five when the tape carries a phase-3
late-aux commitment (routed-projected claims always do) — see
partition.n_sweeps and the prove_streaming loop. Within one sweep,
anything not on a path is dispatchable in parallel (subject to memory), and
the width profile below is the direct evidence for how much parallelism a
multi-GPU dispatcher can actually harvest.

Level = longest path from any source (critical-path depth in claim counts);
width at a level = claims that could run concurrently once their inputs are
ready. Costs per node come from claimcosts, so width can be read in work
units, not just claim counts.
"""
from __future__ import annotations

import json
from collections import defaultdict

import claimcosts
from manifest import Manifest


def build(m: Manifest) -> dict:
    by_name = m.var_by_name()
    edges = set()
    for c in m.claims:
        for name in c.inputs:
            v = by_name.get(name)
            if v is None or v.persistent or v.producer is None:
                continue
            if v.producer != c.idx:
                edges.add((v.producer, c.idx))
    preds = defaultdict(list)
    succs = defaultdict(list)
    for a, b in edges:
        preds[b].append(a)
        succs[a].append(b)

    def claim_W(c):
        if c.w_slots is not None:
            return c.w_slots        # exact from the tape beats the formula
        return claimcosts.cost(c.type, c.params, w_hint=0.0)[0]

    level = {}
    crit = {}   # work of the heaviest dependency path ending at this claim
    width_claims = defaultdict(int)
    width_work = defaultdict(float)
    for c in m.claims:            # claims are in tape (topological) order
        level[c.idx] = 1 + max((level[p] for p in preds[c.idx]), default=-1)
        W = claim_W(c)
        crit[c.idx] = W + max((crit[p] for p in preds[c.idx]), default=0.0)
        width_claims[level[c.idx]] += 1
        width_work[level[c.idx]] += W

    n_levels = max(width_claims) + 1 if width_claims else 0
    total_work = sum(width_work.values())
    crit_work = max(crit.values(), default=0.0)   # true longest weighted path
    return dict(
        nodes=[dict(idx=c.idx, type=claimcosts.canonical(c.type),
                    label=c.label, layer=c.layer, level=level[c.idx])
               for c in m.claims],
        edges=sorted(edges),
        summary=dict(
            n_claims=len(m.claims), n_edges=len(edges),
            critical_path_len=n_levels,
            max_width_claims=max(width_claims.values(), default=0),
            avg_width_claims=(len(m.claims) / n_levels) if n_levels else 0,
            total_work_W=total_work,
            critical_path_work_W=crit_work,
            ideal_speedup_by_work=(total_work / crit_work) if crit_work else 0,
        ))


def summary_text(d: dict) -> str:
    s = d["summary"]
    wide = sorted(((n["level"], n) for n in d["nodes"]), key=lambda x: x[0])
    per_level = defaultdict(int)
    for lv, _ in wide:
        per_level[lv] += 1
    top = sorted(per_level.items(), key=lambda kv: -kv[1])[:3]
    L = [
        "-- dependency DAG --",
        f"claims: {s['n_claims']:,}   edges: {s['n_edges']:,}   "
        f"critical path: {s['critical_path_len']:,} levels",
        f"width: max {s['max_width_claims']:,} claims/level, "
        f"avg {s['avg_width_claims']:.1f}",
        f"work-weighted ideal speedup (infinite GPUs, zero comms): "
        f"{s['ideal_speedup_by_work']:.1f}x",
        f"widest levels (claims): " + ", ".join(f"L{lv}={n}" for lv, n in top),
    ]
    return "\n".join(L)


def save(d: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(d, f)
