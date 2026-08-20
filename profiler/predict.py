"""Dry-run prediction: manifest × machine profile → cost report.

Two time estimates bracket reality, per analysis/maverick-cost-model.md:
  floor     = A·W + B·cids + C·Q      — the NTT-bound estimate (post-reorg
              target; A/C ride memory bandwidth, B rides compute)
  aggregate = aggregate_ns_per_slot·W — calibrated on the current code
              (includes expand+sort and Python orchestration overhead)
The measured gap between them (~2.5×) is the current implementation overhead;
the validation loop (running instrumented proves and diffing against these
numbers) is what tightens both. Memory is reported term by term, with the
unattributed remainder shown explicitly — closing it via measurement is a
profiler goal, not something to paper over.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional

import claimcosts
from manifest import Manifest
from machine import MachineProfile

BYTES_PER_SLOT = 8            # Goldilocks element = u64

# Enrolled-weight per-proof passes. Under enrollment (core.WeightCommitment,
# the kept-trees/opening-ledger path on main) the weight block is never
# re-encoded per proof, but its linear fold and its opening remain per-proof
# work over the whole enrolled block. Ratios mirror the admissible rates in
# analysis/routed_projected_4h_model.py: qlin at the full A per-slot rate,
# opening at half — validation mode (README roadmap 2) refines both.
ENROLLED_QLIN_RATIO = 1.0
ENROLLED_OPEN_RATIO = 0.5
PROOF_JSON_BYTES_PER_VALUE = 21.4   # empirical: 93.6 GB / (40 * 109.27 M) — decimal-ASCII u64 + separators


@dataclass
class CostTotals:
    W: float = 0.0            # witness slots (weights + activations + aux)
    W_weights: float = 0.0    # persistent (streamed) share of W
    W_inputs: float = 0.0     # committed run inputs (producer-less, non-persistent)
    cids: float = 0.0
    Q: float = 0.0
    by_type: Dict[str, list] = field(default_factory=dict)  # type -> [W, cids, Q]
    n_claims: int = 0


def totals(m: Manifest) -> CostTotals:
    t = CostTotals()
    for c in m.claims:
        w_hint = c.w_slots if c.w_slots is not None else 0.0
        W, cids, Q = claimcosts.cost(c.type, c.params, w_hint=w_hint)
        if c.w_slots is not None:
            W = c.w_slots           # exact from the tape beats the formula
        name = claimcosts.canonical(c.type)
        acc = t.by_type.setdefault(name, [0.0, 0.0, 0.0])
        acc[0] += W; acc[1] += cids; acc[2] += Q
        t.W += W; t.cids += cids; t.Q += Q
        t.n_claims += 1
    for v in m.variables:
        if v.producer is not None:
            continue                # claim outputs: already counted above
        t.W += v.length             # committed rows, not claim-counted
        if v.persistent:
            t.W_weights += v.length # weights (streamed, own Merkle block)
        else:
            t.W_inputs += v.length  # run inputs: embeddings, one-hots, tables
    return t


def live_set_peak(m: Manifest) -> Optional[dict]:
    """Replay claims in tape order, freeing each non-persistent variable after
    its last consumer — the streaming prover's memory discipline. Committed
    run inputs with consumers are live from the start. Returns the peak
    resident activation bytes and where it happens."""
    if not m.variables or not m.claims:
        return None
    last_use: Dict[str, int] = {}
    by_name = m.var_by_name()
    for v in m.variables:
        if v.consumers:
            last_use[v.name] = max(v.consumers)
    live = 0
    peak, peak_idx = 0, 0
    expiring: Dict[int, list] = {}
    for v in m.variables:
        if v.producer is None and not v.persistent and v.consumers:
            live += v.length * BYTES_PER_SLOT
            expiring.setdefault(last_use[v.name], []).append(v.name)
    for c in m.claims:
        for name in c.outputs:
            v = by_name[name]
            if v.persistent:
                continue
            live += v.length * BYTES_PER_SLOT
            expiring.setdefault(last_use.get(name, c.idx), []).append(name)
        if live > peak:
            peak, peak_idx = live, c.idx
        for name in expiring.pop(c.idx, ()):
            live -= by_name[name].length * BYTES_PER_SLOT
    at = m.claims[peak_idx]
    return dict(peak_bytes=peak, at_claim=at.label or at.type, at_idx=peak_idx)


def _fmt_s(sec: float) -> str:
    if sec >= 3600:
        return f"{sec:,.0f} s ({sec / 3600:.2f} h)"
    if sec >= 60:
        return f"{sec:,.0f} s ({sec / 60:.1f} min)"
    return f"{sec:.1f} s"


def _gb(x: float) -> str:
    return f"{x / 1e9:,.1f} GB"


def report(m: Manifest, mp: MachineProfile, gpus: int = 1,
           bandwidth_ratio: float = None, compute_ratio: float = None,
           enrolled_weights: bool = False) -> str:
    t = totals(m)
    lig = m.run.get("ligero", {})
    ELL = lig.get("ELL", 8192)
    T_Q = lig.get("T_QUERIES", 40)
    seq = m.run.get("seq", "?")

    # Rows: per-variable row rounding when variables are present, else W/ELL.
    if m.variables:
        m_rows = sum(math.ceil(v.length / ELL) for v in m.variables)
        # phase-2/aux vars are only itemized by the extractor; cover the
        # formula-only remainder at W/ELL density. Core layout rounds every
        # variable up independently, so pooling understates slightly — the
        # report labels such totals approximate.
        itemized = sum(v.length for v in m.variables)
        aux_pooled = max(0.0, t.W - itemized)
        m_rows += aux_pooled / ELL
    else:
        aux_pooled = t.W
        m_rows = t.W / ELL
    m_rows = int(m_rows) + 3    # + blinding rows

    A = mp.get("prove_constants", "A_ns_per_slot")
    B = mp.get("prove_constants", "B_ns_per_cid")
    C = mp.get("prove_constants", "C_ns_per_product")
    agg = mp.get("prove_constants", "aggregate_ns_per_slot")

    L = []
    L.append(f"== VerInf dry-run prediction ==")
    L.append(f"model: {m.model.get('name', '?')}   seq: {seq}   "
             f"claims: {t.n_claims:,}   source: {m.source.get('kind', '?')}")
    L.append(f"machine: {mp.name}" + (f"   what-if: {gpus} GPUs" if gpus > 1 else ""))
    L.append("")
    L.append(f"-- workload totals --")
    share = f"weights {t.W_weights:.3e} = {100 * t.W_weights / t.W:.0f}%"
    if t.W_inputs:
        share += f", run inputs {t.W_inputs:.3e}"
    L.append(f"W     (witness slots)     : {t.W:.3e}   ({share})")
    L.append(f"#cids (linear constraints): {t.cids:.3e}")
    L.append(f"Q     (quad products)     : {t.Q:.3e}")
    rows_note = (" (approx — un-itemized aux slots row-packed at W/ELL; "
                 "core rounds each variable up)" if aux_pooled else "")
    L.append(f"rows  (m_total @ ELL={ELL}): {m_rows:,}{rows_note}")
    L.append("")
    top = sorted(t.by_type.items(), key=lambda kv: -kv[1][0])[:8]
    L.append(f"-- top claim types by W --")
    for name, (W, cids, Q) in top:
        L.append(f"  {name:20s} W {W:10.3e}  ({100 * W / max(t.W, 1):5.1f}%)   "
                 f"cids {cids:9.3e}   Q {Q:9.3e}")
    L.append("")

    L.append(f"-- prove time ({mp.name}) --")
    if None in (A, B, C):
        L.append("  floor: UNAVAILABLE — prove_constants not calibrated on this machine")
        floor = None
    else:
        bw_scale = (bandwidth_ratio or gpus)   # A/C ride aggregate bandwidth
        cp_scale = (compute_ratio or gpus)     # B rides compute
        W_fresh = (t.W - t.W_weights) if enrolled_weights else t.W
        tA = A * W_fresh * 1e-9 / bw_scale
        tB = B * t.cids * 1e-9 / cp_scale
        tC = C * t.Q * 1e-9 / bw_scale
        floor = tA + tB + tC
        tE = tOF = 0.0
        if enrolled_weights:
            tE = ((ENROLLED_QLIN_RATIO + ENROLLED_OPEN_RATIO)
                  * A * t.W_weights * 1e-9 / bw_scale)
            # Fresh rows are opened by the same final sweep — priced at the
            # same OPEN ratio (matches the fresh_open stage of
            # routed_projected_4h_model.py exactly: 0.5*A*fresh slots).
            tOF = ENROLLED_OPEN_RATIO * A * W_fresh * 1e-9 / bw_scale
            floor += tE + tOF
        L.append(f"  floor (NTT-bound, post-reorg target):  {_fmt_s(floor)}")
        wlab = "A*Wf   (encode+lin fold, fresh only" if enrolled_weights \
            else "A*W    (encode+lin fold"
        L.append(f"    {wlab}, BANDWIDTH-bound): {_fmt_s(tA)}")
        L.append(f"    B*cids (challenge hash,  COMPUTE-bound)  : {_fmt_s(tB)}")
        L.append(f"    C*Q    (quad fold,       BANDWIDTH-bound): {_fmt_s(tC)}")
        if enrolled_weights:
            L.append(f"    enrolled block qlin+open ({ENROLLED_QLIN_RATIO:g}+"
                     f"{ENROLLED_OPEN_RATIO:g})*A*Ww       : {_fmt_s(tE)}")
            L.append(f"    open (fresh rows) {ENROLLED_OPEN_RATIO:g}*A*Wf"
                     f"                 : {_fmt_s(tOF)}")
            L.append("    (enrolled weights: zero per-proof RS encode; "
                     "fold+opening passes priced per "
                     "routed_projected_4h_model.py ratios)")
            kd = lig.get("K_DEG", 16384)
            budget = max(0, (kd - ELL) // 2)
            L.append(f"    enrollment lifecycle (unpriced): one-time enroll "
                     f"of the weight block; refresh after {budget:,} "
                     f"distinct opened columns ((K_DEG-ELL)/2) — "
                     f">= {budget // max(T_Q, 1)} proofs at T={T_Q}")
        else:
            L.append("    (legacy floor excludes column-opening passes; "
                     "enrolled mode prices them)")
    if enrolled_weights:
        L.append("  aggregate: N/A under enrollment — calibrated on runs "
                 "that commit weights per proof")
    elif agg is None:
        L.append("  aggregate: UNAVAILABLE — aggregate_ns_per_slot not calibrated")
    else:
        taggr = agg * t.W * 1e-9 / (bandwidth_ratio or gpus)
        L.append(f"  aggregate (current-code calibration):  {_fmt_s(taggr)}")
        if floor:
            L.append(f"  bracket ratio (implementation overhead): {taggr / floor:.1f}x")
    if gpus > 1:
        L.append(f"  NOTE: {gpus}-GPU numbers are IDEAL scaling (A/C / bandwidth "
                 f"ratio, B / compute ratio); real efficiency needs the "
                 f"interconnect model + scheduler — unmodeled until topology known.")
    L.append("")

    L.append(f"-- memory ({mp.name}) --")
    opened = T_Q * m_rows * BYTES_PER_SLOT
    L.append(f"  opened-column payload (T={T_Q} x rows x 8B, GPU-resident): {_gb(opened)}")
    chunk_rows = 1024
    n_lig = lig.get("N_LIG", 65536)
    work = chunk_rows * (ELL + n_lig) * BYTES_PER_SLOT
    L.append(f"  encode working set (1024-row chunk, msg+codeword)        : {_gb(work)}")
    ls = live_set_peak(m)
    if ls:
        L.append(f"  peak activation live-set (streaming order)               : "
                 f"{_gb(ls['peak_bytes'])}  at claim {ls['at_idx']} ({ls['at_claim']})")
    mem_gb = mp.get("gpu", "mem_GB")
    known = opened + work + (ls["peak_bytes"] if ls else 0)
    L.append(f"  known terms total: {_gb(known)}"
             + (f"   (device mem: {mem_gb} GB)" if mem_gb else ""))
    L.append(f"  NOTE: measured peaks exceed known terms (fold accumulators, "
             f"in-flight weights, allocator slack) — the extraction+validate "
             f"loop on real runs is what closes this gap.")
    L.append("")

    L.append(f"-- proof & verify --")
    proof = T_Q * m_rows * PROOF_JSON_BYTES_PER_VALUE
    L.append(f"  proof size (JSON, T={T_Q}): {_gb(proof)}")
    bpr = mp.get("verify", "bytes_per_row")
    if bpr:
        L.append(f"  verifier peak RSS (~{bpr} B/row, wide error bar): {_gb(m_rows * bpr)}")
    dump = mp.get("io", "proof_dump_MBps")
    if dump:
        L.append(f"  proof dump time (~{dump} MB/s, Python-bound): {_fmt_s(proof / (dump * 1e6))}")
    return "\n".join(L)
