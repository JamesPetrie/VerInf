"""Partition evaluator: score claim->shard mappings before writing any
distribution code.

A parallelization strategy is a partition of manifest claims across N shards
(GPUs). For each candidate this module computes per-shard work (W, #cids, Q,
weight slots), load imbalance, and cross-shard traffic, then prices compute
with a machine profile and comms with a *swept* interconnect bandwidth —
topology is unknown, so every comms number is reported across the plausible
range. Conclusions that hold across the sweep are safe to act on now; ones
that flip are exactly the questions to settle when the cluster topology
lands.

Traffic model (deliberately explicit about its approximations):
- A variable produced on shard p and consumed on shards C costs
  len*8 bytes per shard in C \\ {p} (send once per remote shard, reuse there).
- Commutative combines (freivalds_combine) are reduction-aware: remote
  producers of partial-shaped inputs (length == the output's) send ONE
  partial buffer of the OUTPUT's size per remote shard, not their full
  inputs (partial sums, tree-reducible). Differently-shaped side inputs of
  those combines (the routing mask, ~1 MB) are dropped.
- LogUp multiplicities reduce the same way: every shard whose lookups hit
  a shared table mutates a local copy of the mult vector, and settlement
  needs their sum — each participating remote shard ships one T_LEN-sized
  partial to the settlement shard per sweep (extracted manifests only;
  synth does not model tables). Settlement z inputs reduce even harder:
  the global balance needs only Σz per shard, so each remote producing
  shard ships ONE field element per settlement, never the z vectors
  (which are output-sized and would dominate all other traffic ~1000x —
  the artifact this rule replaces).
- Witness activations regenerate every sweep, so activation+reduction
  traffic recurs x4 — x5 when the tape has a phase-3 commitment sweep
  (routed-projected claims; see n_sweeps) — unless the scheduler caches;
  the per-sweep number is the honest unit and the total is also shown.
- Weight-commit work (dependency-free) is split evenly across shards;
  weight *streaming* for witness compute is charged to each shard that
  consumes the weight (per sweep), and rides disk/host lanes, not the
  interconnect.
- NOT modeled yet: dependency-chain serialization within a shard schedule
  (needs the discrete-event scheduler sim — roadmap), Merkle column-stream
  chaining across row-shard boundaries (protocol-adjacent, costed
  separately), per-round barrier idle time.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Callable, Dict, List, Optional

import claimcosts
from manifest import Manifest
from machine import MachineProfile
from predict import (_fmt_s, _gb, BYTES_PER_SLOT,
                     ENROLLED_QLIN_RATIO, ENROLLED_OPEN_RATIO)

REDUCIBLE = {"freivalds_combine"}
DEFAULT_BANDWIDTHS_GBPS = (25.0, 100.0, 450.0, 900.0)   # IB .. NVLink-domain
SWEEPS = 4        # base sweeps: R1, R2, test polynomials, openings


def n_sweeps(m: Manifest) -> int:
    """Streaming-sweep count for this tape. prove_streaming runs R1, R2,
    a CONDITIONAL R3 commitment sweep (only when phase-3 late-aux
    variables exist), the test polynomials, and the openings — so 5 for
    routed-projected tapes (their f_y/f_u/f_p are phase 3), 4 otherwise.
    Detected from variable phases (extracted manifests) with a claim-type
    fallback (synth pools the routed aux instead of emitting vars)."""
    if any(v.phase >= 3 for v in m.variables):
        return SWEEPS + 1
    if any(claimcosts.canonical(c.type) == "routed_projected"
           for c in m.claims):
        return SWEEPS + 1
    return SWEEPS
# Expert id from a claim label, both conventions: synth labels expert claims
# "L3.moe.e12.gate"; extracted labels are output-variable names, which carry
# the demo weight names L{n}_Wg{e}/Wu{e}/Wd{e} via the "a@b" derivation
# (e.g. "x_r@L1_Wg0#123"). Shared-expert weights (Wgs/Wus/Wds) have no
# digit after the letter and never match. Rightmost hit wins, like layers.
_EXPERT_RES = (re.compile(r"\.e(\d+)\."),
               re.compile(r"_W[gud](\d+)(?=[#.@~]|$)"))


def _expert_of(label: str):
    best = None                    # rightmost by POSITION across conventions
    for rx in _EXPERT_RES:
        for mt in rx.finditer(label or ""):
            if best is None or mt.start() > best.start():
                best = mt
    return int(best.group(1)) if best else None


def _claim_W(c) -> float:
    if c.w_slots is not None:
        return c.w_slots            # exact from the tape beats the formula
    return claimcosts.cost(c.type, c.params, w_hint=0.0)[0]


def _contiguous_split(weights: List[float], n: int) -> List[int]:
    """Assign items (in order) to n contiguous groups with ~equal weight."""
    total = sum(weights) or 1.0
    out, cum, s = [], 0.0, 0
    for w in weights:
        # advance shard when its quota is filled (never past n-1)
        while s < n - 1 and cum >= total * (s + 1) / n:
            s += 1
        out.append(s)
        cum += w
    return out


def assign_rows(m: Manifest, n: int) -> List[int]:
    """Contiguous claim ranges balanced by W — models splitting the tape
    stream itself (commit sharding in tape order)."""
    return _contiguous_split([_claim_W(c) for c in m.claims], n)


def assign_layers(m: Manifest, n: int) -> List[int]:
    """Contiguous layer blocks balanced by per-layer W — the pipeline shape.
    Layer-less claims (embed, LM head) follow their neighborhood."""
    layer_w: Dict[int, float] = defaultdict(float)
    for c in m.claims:
        if c.layer is not None:
            layer_w[c.layer] += _claim_W(c)
    if not layer_w:
        raise ValueError(
            "manifest has no per-claim layer metadata — the layers/experts "
            "strategies would put everything on shard 0. Extracted manifests "
            "need labels the layer parser understands (extract._layer_of).")
    layers = sorted(layer_w)
    shard_of_layer = dict(zip(layers, _contiguous_split(
        [layer_w[l] for l in layers], n)))
    out, cur = [], 0
    for c in m.claims:
        if c.layer is not None:
            cur = shard_of_layer[c.layer]
        out.append(cur)
    return out


def _expert_of_claim(c, by_name) -> Optional[int]:
    """Expert identity of a claim, or None (backbone). Only per-expert
    matmuls have one, and it is derived from the claim's PERSISTENT
    WEIGHT INPUT's name — never the output label: matmul output names
    concatenate both operands, so an ordinary matmul whose activation
    ancestry passed through a routed claim inherits a _Wg0 substring
    (rp[..@L2_Wg0..]_rs@L3_W_Q), and routed claims themselves are named
    from their first shard. Weight-variable names are unambiguous:
    synth "L1.e3.W_gate", extracted "L1_Wg0"; attention (W_Q_L0) and
    shared-expert (L1_Wgs) weights never match."""
    if claimcosts.canonical(c.type) != "matmul":
        return None
    for name in c.inputs:
        v = by_name.get(name)
        if v is not None and v.persistent:
            e = _expert_of(v.name)
            if e is not None:
                return e
    return None


def assign_experts(m: Manifest, n: int) -> List[int]:
    """Expert matmuls spread expert->shard round-robin; everything else runs
    on the layer-pipeline backbone. The combine claims sit on the backbone
    and receive per-shard partials."""
    backbone = assign_layers(m, n)
    by_name = m.var_by_name()
    out = []
    for c, b in zip(m.claims, backbone):
        e = _expert_of_claim(c, by_name)
        out.append(e % n if e is not None else b)
    return out


STRATEGIES: Dict[str, Callable] = {
    "rows": assign_rows,
    "layers": assign_layers,
    "experts": assign_experts,
}


def evaluate(m: Manifest, assignment: List[int], n: int,
             mp: MachineProfile, *, weight_bytes_per_param: float = 1.0,
             skip_weight_commit: bool = False, sweeps: Optional[int] = None,
             enrolled_weights: bool = False) -> dict:
    _check_modes(enrolled_weights, skip_weight_commit)
    if sweeps is None:
        sweeps = n_sweeps(m)
    by_name = m.var_by_name()
    shard_W = [0.0] * n
    shard_cids = [0.0] * n
    shard_Q = [0.0] * n
    reducible_claims = []
    settle_claims = []
    for c, s in zip(m.claims, assignment):
        W, cids, Q = claimcosts.cost(c.type, c.params, w_hint=c.w_slots or 0.0)
        if c.w_slots is not None:
            W = c.w_slots
        shard_W[s] += W
        shard_cids[s] += cids
        shard_Q[s] += Q
        if claimcosts.canonical(c.type) in REDUCIBLE:
            reducible_claims.append((c, s))
        elif claimcosts.canonical(c.type) == "table_settle":
            settle_claims.append((c, s))
    reducible_idx = {c.idx for c, _ in reducible_claims}
    settle_idx = {c.idx for c, _ in settle_claims}

    # Weight streaming: each shard loads the weights its claims consume.
    shard_weight_slots = [0.0] * n
    seen = set()
    for v in m.variables:
        if not v.persistent:
            continue
        for ci in v.consumers:
            s = assignment[ci]
            if (v.name, s) not in seen:
                seen.add((v.name, s))
                shard_weight_slots[s] += v.length

    # Cross-shard activation traffic (reduction-aware).
    act_bytes = 0.0
    for v in m.variables:
        if v.persistent or v.producer is None:
            continue
        p = assignment[v.producer]
        remote = {assignment[ci] for ci in v.consumers
                  if ci not in reducible_idx
                  and ci not in settle_idx} - {p}
        act_bytes += v.length * BYTES_PER_SLOT * len(remote)
    red_bytes = 0.0
    for c, s in reducible_claims:
        # Logical combine output is T x F; extracted claims also list y/u/p
        # and other aux among outputs, so params are the reliable source.
        p = c.params
        if "T" in p and "F" in p:
            out_len = p["T"] * p["F"]
        else:
            out_len = next((by_name[o].length for o in c.outputs
                            if not by_name[o].persistent
                            and by_name[o].phase == 1), 0)
        # Only partial-shaped inputs (length == output's) reduce remotely;
        # differently-shaped side inputs (the routing mask) are dropped.
        producers = {assignment[by_name[i].producer] for i in c.inputs
                     if i in by_name and by_name[i].producer is not None
                     and by_name[i].length == out_len} - {s}
        red_bytes += len(producers) * out_len * BYTES_PER_SLOT
    # LogUp settlement reduction. mult (the producer-less non-persistent
    # input): every remote shard holding lookups that increment it sends one
    # summed partial of the full table length. z inputs (each produced by
    # its own lookup/rescale claim): the settlement's global balance needs
    # only Σz per shard, so each remote producing shard sends ONE summed
    # field element per settlement — never the z vectors themselves (they
    # are excluded from the activation loop above; shipping them whole was
    # the ~1000x traffic artifact found on the first extracted-manifest
    # scorecard, 2026-08-19).
    mult_bytes = 0.0
    for c, s in settle_claims:
        z_shards = set()
        for name in c.inputs:
            v = by_name.get(name)
            if v is None or v.persistent:
                continue
            if v.producer is None:
                senders = {assignment[ci] for ci in v.consumers} - {s}
                mult_bytes += len(senders) * v.length * BYTES_PER_SLOT
            else:
                z_shards.add(assignment[v.producer])
        mult_bytes += len(z_shards - {s}) * BYTES_PER_SLOT

    # Compute time per shard (whole-proof floor constants) + even split of
    # the dependency-free commit work: weights and producer-less run inputs
    # (embeddings, one-hots, table mult/w) — the slots predict.totals counts
    # outside claim W.
    A = mp.get("prove_constants", "A_ns_per_slot")
    B = mp.get("prove_constants", "B_ns_per_cid")
    C = mp.get("prove_constants", "C_ns_per_product")
    total_weight_slots = sum(v.length for v in m.variables if v.persistent)
    total_input_slots = sum(v.length for v in m.variables
                            if v.producer is None and not v.persistent)
    shard_t = None
    if None not in (A, B, C):
        # Enrolled weights: no per-proof encode; each shard instead pays
        # the qlin+open passes over the enrolled slots it owns (ratios
        # per predict.ENROLLED_*_RATIO / routed_projected_4h_model.py).
        wcommit = 0.0 if (skip_weight_commit or enrolled_weights) \
            else A * total_weight_slots / n
        icommit = A * total_input_slots / n
        enr = (ENROLLED_QLIN_RATIO + ENROLLED_OPEN_RATIO) * A \
            if enrolled_weights else 0.0
        # Enrolled mode also prices the fresh-row opening pass (Ed's
        # fresh_open stage: OPEN_RATIO * A over the shard's fresh slots);
        # the legacy floor never priced openings, stated in predict.
        fro = ENROLLED_OPEN_RATIO * A if enrolled_weights else 0.0
        # Fresh openings cover ALL fresh rows: claim witness AND the
        # producer-less input commitments (matching predict's
        # W_fresh = W - W_weights, which includes inputs).
        iopen = fro * total_input_slots / n
        shard_t = [(A * shard_W[s] + B * shard_cids[s] + C * shard_Q[s]
                    + wcommit + icommit + fro * shard_W[s] + iopen
                    + enr * shard_weight_slots[s]) * 1e-9
                   for s in range(n)]
    # Opened-column payload per shard: T_QUERIES x that shard's rows x 8B —
    # today's single-GPU 35 GB term, and how sharding shrinks it.
    lig = m.run.get("ligero", {})
    ELL, T_Q = lig.get("ELL", 8192), lig.get("T_QUERIES", 40)
    per_shard_committed = (total_weight_slots + total_input_slots) / n
    rows_max = max((shard_W[s] + per_shard_committed) / ELL for s in range(n))
    opened_max = T_Q * rows_max * BYTES_PER_SLOT

    return dict(
        n=n, sweeps=sweeps, opened_bytes_max=opened_max,
        shard_W=shard_W, shard_cids=shard_cids, shard_Q=shard_Q,
        shard_weight_slots=shard_weight_slots,
        weight_stream_bytes_max=max(shard_weight_slots) * weight_bytes_per_param * sweeps,
        act_bytes_per_sweep=act_bytes, red_bytes_per_sweep=red_bytes,
        mult_bytes_per_sweep=mult_bytes,
        traffic_per_sweep=act_bytes + red_bytes + mult_bytes,
        traffic_total=(act_bytes + red_bytes + mult_bytes) * sweeps,
        shard_t=shard_t,
        wall=max(shard_t) if shard_t else None,
        serial=sum(shard_t) if shard_t else None,
        imbalance=(max(shard_t) * n / sum(shard_t)) if shard_t and sum(shard_t) else None,
    )


def _comms_row(ev: dict, bandwidths) -> List[tuple]:
    out = []
    for bw in bandwidths:
        t_comms = ev["traffic_total"] / (bw * 1e9)
        frac = t_comms / ev["wall"] if ev["wall"] else None
        out.append((bw, t_comms, frac))
    return out


def _verdict(frac: Optional[float]) -> str:
    if frac is None:
        return "?"
    if frac < 0.01:
        return "negligible"
    if frac < 0.10:
        return "ok"
    return "BINDING"


def _check_modes(enrolled_weights: bool, skip_weight_commit: bool) -> None:
    """One validator for every public entry point: the two modes are
    contradictory (enrollment PRICES the reused commitment, skip DROPS
    all weight cost), and silently letting one win mislabels output."""
    if enrolled_weights and skip_weight_commit:
        raise ValueError(
            "enrolled_weights and skip_weight_commit are mutually "
            "exclusive: enrollment prices the reused commitment "
            "(qlin+open passes); skip_weight_commit drops all weight "
            "cost (diagnostic only)")


def _mode_suffix(m: Manifest, enrolled_weights: bool,
                 skip_weight_commit: bool) -> str:
    """Header marker for non-default cost modes, so copied output is
    self-describing: DIAGNOSTIC when weight cost was deliberately
    dropped; the enrollment assumption (refresh budget, matching
    predict's lifecycle line) when enrolled."""
    if skip_weight_commit:
        return " — DIAGNOSTIC: weight commitment omitted (not a protocol mode)"
    if enrolled_weights:
        lig = m.run.get("ligero", {})
        ell = lig.get("ELL", 8192)
        kd = lig.get("K_DEG", 16384)
        tq = lig.get("T_QUERIES", 40)
        budget = max(0, (kd - ell) // 2)
        return (f" — ENROLLED weights (qlin+open passes; one-time enroll "
                f"unpriced; refresh after {budget:,} distinct opened "
                f"columns >= {budget // max(tq, 1)} proofs at T={tq})")
    return ""


def _no_expert_labels(m: Manifest) -> bool:
    by_name = m.var_by_name()
    return all(_expert_of_claim(c, by_name) is None for c in m.claims)


def report(m: Manifest, strategy: str, n: int, mp: MachineProfile, *,
           bandwidths=DEFAULT_BANDWIDTHS_GBPS, weight_bytes_per_param=1.0,
           skip_weight_commit=False, enrolled_weights=False) -> str:
    _check_modes(enrolled_weights, skip_weight_commit)
    try:
        assignment = STRATEGIES[strategy](m, n)
    except ValueError as e:
        return f"== partition scorecard: {strategy} x{n} ==\nUNAVAILABLE: {e}"
    ev = evaluate(m, assignment, n, mp,
                  weight_bytes_per_param=weight_bytes_per_param,
                  skip_weight_commit=skip_weight_commit,
                  enrolled_weights=enrolled_weights)
    L = [f"== partition scorecard: {strategy} x{n} on {mp.name} "
         f"({m.model.get('name', '?')} S={m.run.get('seq', '?')})"
         + _mode_suffix(m, enrolled_weights, skip_weight_commit) + " =="]
    if strategy == "experts" and _no_expert_labels(m):
        L.append("NOTE: no expert labels ('.eN.' or '_Wg|u|dN') in this "
                 "manifest — assignment is identical to the layers backbone.")
    if ev["shard_t"]:
        L.append(f"wall (max shard, floor model): {_fmt_s(ev['wall'])}   "
                 f"serial: {_fmt_s(ev['serial'])}   "
                 f"speedup {ev['serial'] / ev['wall']:.2f}x of {n}   "
                 f"imbalance {ev['imbalance']:.2f}")
    for s in range(n):
        t = f"   t={_fmt_s(ev['shard_t'][s])}" if ev["shard_t"] else ""
        L.append(f"  shard {s}: W {ev['shard_W'][s]:.2e}  "
                 f"cids {ev['shard_cids'][s]:.2e}  Q {ev['shard_Q'][s]:.2e}  "
                 f"weights {ev['shard_weight_slots'][s]:.2e}{t}")
    tr = (f"traffic/sweep: activations {_gb(ev['act_bytes_per_sweep'])} + "
          f"reduction partials {_gb(ev['red_bytes_per_sweep'])}")
    if ev["mult_bytes_per_sweep"]:
        tr += f" + logup mult partials {_gb(ev['mult_bytes_per_sweep'])}"
    L.append(tr + f"   (x{ev['sweeps']} sweeps = {_gb(ev['traffic_total'])})")
    L.append(f"max shard weight stream (x{ev['sweeps']} sweeps, "
             f"{weight_bytes_per_param} B/param, disk/host lanes): "
             f"{_gb(ev['weight_stream_bytes_max'])}")
    L.append(f"max shard opened-column payload (GPU-resident): "
             f"{_gb(ev['opened_bytes_max'])}")
    L.append("interconnect sweep (topology unknown):")
    for bw, t_comms, frac in _comms_row(ev, bandwidths):
        vs_wall = (f"({100 * frac:.2f}% of wall)" if frac is not None
                   else "(wall unknown — no prove_constants)")
        L.append(f"  {bw:6.0f} GB/s: comms {_fmt_s(t_comms)}  "
                 f"{vs_wall}  -> {_verdict(frac)}")
    L.append("caveat: chain serialization within/between shards not modeled "
             "(scheduler sim is roadmap); floor-model compute times.")
    return "\n".join(L)


def compare(m: Manifest, n: int, mp: MachineProfile, *,
            bandwidths=DEFAULT_BANDWIDTHS_GBPS, weight_bytes_per_param=1.0,
            skip_weight_commit=False, enrolled_weights=False) -> str:
    _check_modes(enrolled_weights, skip_weight_commit)
    L = [f"== strategy comparison x{n} on {mp.name} "
         f"({m.model.get('name', '?')} S={m.run.get('seq', '?')}, "
         f"floor model, {n_sweeps(m)} sweeps)"
         + _mode_suffix(m, enrolled_weights, skip_weight_commit)
         + " ==", ""]
    header = (f"{'strategy':10s} {'wall':>12s} {'speedup':>8s} {'imbal':>6s} "
              f"{'traffic/sweep':>14s} {'wstream max':>12s} {'opened max':>11s}")
    header += "".join(f"{f'@{int(bw)}GB/s':>12s}" for bw in bandwidths)
    L.append(header)
    for name in STRATEGIES:
        try:
            assignment = STRATEGIES[name](m, n)
        except ValueError:
            L.append(f"{name:10s}  (skipped: manifest has no layer metadata)")
            continue
        ev = evaluate(m, assignment, n, mp,
                      weight_bytes_per_param=weight_bytes_per_param,
                      skip_weight_commit=skip_weight_commit,
                      enrolled_weights=enrolled_weights)
        if ev["shard_t"] is None:
            L.append(f"{name:10s}  (no prove_constants on this machine)")
            continue
        row = (f"{name:10s} {_fmt_s(ev['wall']).split(' (')[0]:>12s} "
               f"{ev['serial'] / ev['wall']:>7.2f}x {ev['imbalance']:>6.2f} "
               f"{_gb(ev['traffic_per_sweep']):>14s} "
               f"{_gb(ev['weight_stream_bytes_max']):>12s} "
               f"{_gb(ev['opened_bytes_max']):>10s}")
        for _, _, frac in _comms_row(ev, bandwidths):
            row += f"{_verdict(frac):>12s}"
        L.append(row)
    L.append("")
    L.append("wall = max-shard compute (floor constants); comms verdict = "
             "total cross-shard traffic vs wall at that bandwidth.")
    if _no_expert_labels(m):
        L.append("note: no expert labels ('.eN.' or '_Wg|u|dN') — the "
                 "experts row is identical to layers.")
    return "\n".join(L)
