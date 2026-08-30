"""Stage-aware weight-ownership model for the weight-split prover (M1).

Prices the milestone-1 architecture that `partition.evaluate` cannot: one
COORDINATOR device runs the whole fresh path (commits, witness, fresh
folds, fresh openings) and owns a run of the ENROLLED weight block; N-1
WORKER devices own the rest and touch it only in the two per-proof passes
the protocol leaves on enrolled rows — the test-polynomial fold (qlin,
1.0*A per slot) and the column opening (0.5*A per slot). Enrollment mode
only.

Barrier. s_col is derived from the test polynomials, which need every
device's fold partial merged first, so fold imbalance cannot cancel
opening imbalance:

    wall = commit + max_dev(fold_dev) + max_dev(open_dev)

Executable plans. Per stage a plan is a list of N contiguous runs of
WHOLE weight variables in layout order (the persistent-variable list is
the W-block layout walk), device 0 = coordinator first. Everything —
timing, packed bytes, HBM holds (the UNION of a device's fold and open
runs; interior workers' runs need not nest), opening payloads — derives
from the plans the prover would run. Slots are PHYSICAL: the prover
allocates ceil(length/ELL) rows per variable (core.Variable.n_rows), so a
non-aligned variable costs its padded rows in every pass and in the
opened columns.

Cuts are solved EXACTLY over variable boundaries, not on a fraction grid:
the coordinator's cut is found by bisection on the crossing of its
(increasing) time with the workers' (decreasing) min-max time, and the
workers' section is partitioned into N-1 contiguous runs by the standard
min-max greedy with a bisected bound — runs are UNEQUAL where that
balances time (and packed bytes under a cap). The two stages are
separable and solved independently unless the resident HBM cap binds on
their unions (or --static is given), in which case the cuts are TIED
across stages (a device then holds exactly its run) and chosen under the
cap for the true barrier-separated wall: EXACT at N=2 (one cut,
enumerated — 'tied-exact'/'static-exact'), HEURISTIC at N>=3 (three
proxy solves; 'tied-heuristic'/'static-heuristic' — an exhaustive search
can beat them, and the worst-case gap is uncharacterized). When nothing fits, the
reported plan is the least-infeasible one (minimum max hold), labelled
'least-infeasible(...)'. Explicit --x-* fractions use equal-slot worker
shares (an explicit scheduling heuristic).

Metrics. `kernel_floor_ratio` = N=1 compute-only floor / wall; the
`same_mode_speedup` = N=1 wall under the SAME storage model / wall — the
M1b measurement ("vs the N=1 run on the same code"). They coincide when
the N=1 wall equals the compute-only floor (resident, or streaming with
the I/O fully hidden); resident same-mode is n/a when N=1 does not fit.

Stage split. A*W_fresh is split between the commit sweeps (encode) and
the fold sweep (linear fold) at ENCODE_SHARE_OF_A, from the gb10-spark
provenance (encode ~4 + linear-fold poly_mul ~5 ns/slot); the B200's A
and C are bandwidth-derived from that same profile, so the split is a
modelling assumption and `report` prints its sensitivity. B*cids and
C*Q sit in the fold stage. This is the A/B/C kernel floor: it EXCLUDES
the semantic sweeps (witness regeneration), Python orchestration, worker
start-up, the duplicate compile and the opening hand-off, all of which
land on or through the coordinator.

Storage. A device either STREAMS its owned share from disk each pass or
holds it RESIDENT in HBM (packed bytes must fit under mem_GB minus a
workspace reserve — a path the current loaders do not provide; see the
design note). Under the resident model the N=1 run itself must fit one
device; when it does not, the same-mode speedup is reported as
unavailable (the M1b baseline is then the STREAMING N=1 run, a
cross-mode comparison the report does not silently make). Streaming has
two axes:
  disk_mode  'shared'     one volume, bandwidth is AGGREGATE across devices
             'per-device' each device has its own disk at the full rate
  io_overlap 'none'       a device resolves each variable before encoding
                          it (core._iter_message_chunks) — compute + I/O
             'perfect'    an idealised prefetch: max(compute, I/O)
Defaults are the one-node deployment: shared + none. Streaming figures
are a PACKED-BYTE SCHEDULING LOWER BOUND: each planned variable's packed
source is counted once; decode/transform cost and the group amplification
of today's attention and MoE closures (which decode a whole tensor group
per requested matrix) are unpriced — exact current-loader pricing needs
the source descriptor planned for M1c. Streaming without a disk
calibration, or resident without mem_GB, is UNAVAILABLE, not free.

Packed bytes/param come, in order, from a flat override, the variable's
extracted `packed_bytes` (must be finite and >= 0), its GGUF `quant`
type (must be a known type), else Q4_K.
"""
import math
from bisect import bisect_left
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from manifest import Manifest
from machine import MachineProfile
from predict import ENROLLED_OPEN_RATIO, ENROLLED_QLIN_RATIO, totals, _fmt_s, _gb

# Share of A*W_fresh that is commit-sweep encode (rest = fold-sweep linear
# fold). Source: profiler/machines/gb10-spark.json provenance for
# A_ns_per_slot ("encode ~4 + linear-fold poly_mul ~5 ns/slot").
ENCODE_SHARE_OF_A = 4.0 / 9.0

# Packed bytes per parameter by GGUF block type (block bytes / weights per
# block): Q4_K 144/256, Q5_K 176/256, Q6_K 210/256, Q8_0 34/32.
QUANT_BYTES_PER_PARAM: Dict[str, float] = {
    "Q4_K": 144 / 256, "Q5_K": 176 / 256, "Q6_K": 210 / 256,
    "Q8_0": 34 / 32, "F16": 2.0, "BF16": 2.0, "F32": 4.0,
}
BYTES_PER_SLOT = 8
DISK_MODES = ("shared", "per-device")
IO_OVERLAPS = ("none", "perfect")
_BISECT_ITERS = 60      # real-valued bisection on a time bound (seconds)

Run = Tuple[int, int]   # [var_lo, var_hi) into the persistent-variable list


@dataclass
class Stages:
    """Single-device (N=1) seconds per stage at the machine's A/B/C. The W
    terms are priced on PHYSICAL (row-padded) enrolled slots."""
    commit: float        # fresh encode, R1-R3
    fresh_fold: float    # fresh linear fold + B*cids + C*Q (test-poly sweep)
    fresh_open: float    # 0.5*A over fresh slots (openings sweep)
    w_fold: float        # 1.0*A over enrolled physical slots
    w_open: float        # 0.5*A over enrolled physical slots

    @property
    def floor(self) -> float:
        return (self.commit + self.fresh_fold + self.fresh_open
                + self.w_fold + self.w_open)


def packed_bytes_of(v, bytes_per_param: Optional[float]) -> float:
    """Packed source bytes of one persistent variable; raises ValueError on
    invalid provenance rather than silently sizing HBM wrong."""
    if bytes_per_param is not None:
        return v.length * bytes_per_param
    pb = getattr(v, "packed_bytes", None)
    if pb is not None:
        pb = float(pb)
        if not math.isfinite(pb) or pb < 0:
            raise ValueError(f"variable '{v.name}': packed_bytes must be finite "
                             f"and >= 0, got {pb!r}")
        return pb
    q = getattr(v, "quant", None)
    if q is None:
        return v.length * QUANT_BYTES_PER_PARAM["Q4_K"]
    if q not in QUANT_BYTES_PER_PARAM:
        raise ValueError(f"variable '{v.name}': unknown quant type {q!r} "
                         f"(known: {', '.join(sorted(QUANT_BYTES_PER_PARAM))})")
    return v.length * QUANT_BYTES_PER_PARAM[q]


class _Block:
    """The enrolled block: persistent variables in layout order with
    cumulative PHYSICAL-slot and packed-byte prefixes."""

    def __init__(self, m: Manifest, bytes_per_param: Optional[float], ELL: int):
        self.vars = [v for v in m.variables if v.persistent]
        if not self.vars:
            raise ValueError("manifest has no persistent (enrolled) weight variables")
        self.ELL = ELL
        self.cum_phys = [0.0]
        self.cum_logical = [0.0]
        self.cum_bytes = [0.0]
        for v in self.vars:
            rows = -(-int(v.length) // ELL)               # core.Variable.n_rows
            self.cum_phys.append(self.cum_phys[-1] + rows * ELL)
            self.cum_logical.append(self.cum_logical[-1] + v.length)
            self.cum_bytes.append(self.cum_bytes[-1] + packed_bytes_of(v, bytes_per_param))
        self.n = len(self.vars)
        self.total_phys = self.cum_phys[-1]
        self.total_logical = self.cum_logical[-1]
        self.total_bytes = self.cum_bytes[-1]
        self.aligned = self.total_phys == self.total_logical

    def phys(self, lo: int, hi: int) -> float:
        return self.cum_phys[hi] - self.cum_phys[lo]

    def bytes(self, lo: int, hi: int) -> float:
        return self.cum_bytes[hi] - self.cum_bytes[lo]

    def cut_at_fraction(self, frac: float, lo: int = 0) -> int:
        """Variable boundary nearest frac*total_phys, at or after lo."""
        target = frac * self.total_phys
        j = bisect_left(self.cum_phys, target, lo=lo)
        if lo < j < len(self.cum_phys) and \
                (target - self.cum_phys[j - 1]) < (self.cum_phys[j] - target):
            j -= 1
        return min(max(j, lo), self.n)

    def union_bytes(self, a: Run, b: Run) -> float:
        (alo, ahi), (blo, bhi) = a, b
        if ahi <= alo:
            return self.bytes(blo, bhi)
        if bhi <= blo:
            return self.bytes(alo, ahi)
        if ahi < blo or bhi < alo:                     # disjoint
            return self.bytes(alo, ahi) + self.bytes(blo, bhi)
        return self.bytes(min(alo, blo), max(ahi, bhi))


# ---------------------------------------------------------------- storage --

@dataclass
class _Storage:
    resident: bool
    disk_GBps: Optional[float]
    disk_mode: str
    io_overlap: str

    def dev_time(self, comp: float, byt: float) -> float:
        if self.resident:
            return comp
        io = byt / (self.disk_GBps * 1e9)
        return max(comp, io) if self.io_overlap == "perfect" else comp + io

    def stage(self, dev: Sequence[float], total_bytes: float) -> float:
        t = max(dev)
        if not self.resident and self.disk_mode == "shared":
            # one volume cannot deliver the stage's bytes faster than
            # aggregate bandwidth, whatever the split
            t = max(t, total_bytes / (self.disk_GBps * 1e9))
        return t


# ----------------------------------------------------------------- solver --

def _greedy_runs(cost: Callable[[int, int], float], lo: int, hi: int, k: int,
                 bound: float, cap: Optional[float],
                 bytes_fn: Callable[[int, int], float]) -> Optional[List[Run]]:
    """Split [lo, hi) into <= k contiguous runs with cost(run) <= bound and
    bytes(run) <= cap (if any), greedily extending each run as far as it
    goes (cost and bytes are monotone in the run). None if impossible."""
    runs = []
    start = lo
    while start < hi:
        if len(runs) == k:
            return None
        # farthest end with cost <= bound and bytes <= cap: bisect on end
        a, b = start + 1, hi
        if cost(start, a) > bound + 1e-12 or (cap is not None and bytes_fn(start, a) > cap):
            return None                                   # a single variable exceeds
        while a < b:
            mid = (a + b + 1) // 2
            if cost(start, mid) <= bound + 1e-12 and (cap is None or bytes_fn(start, mid) <= cap):
                a = mid
            else:
                b = mid - 1
        runs.append((start, a))
        start = a
    while len(runs) < k:
        runs.append((hi, hi))                             # idle worker
    return runs


def _minmax_partition(cost, lo, hi, k, cap, bytes_fn) -> Tuple[Optional[List[Run]], float]:
    """Min-max contiguous partition of [lo, hi) into k runs (bisection on
    the bound over the real line; exact up to _BISECT_ITERS)."""
    if k <= 0:
        return (None, math.inf) if hi > lo else ([], 0.0)
    if hi <= lo:
        return [(hi, hi)] * k, 0.0
    hi_b = cost(lo, hi)
    if _greedy_runs(cost, lo, hi, k, hi_b, cap, bytes_fn) is None:
        return None, math.inf                             # cap makes it impossible
    lo_b = 0.0
    for _ in range(_BISECT_ITERS):
        mid = 0.5 * (lo_b + hi_b)
        if _greedy_runs(cost, lo, hi, k, mid, cap, bytes_fn) is not None:
            hi_b = mid
        else:
            lo_b = mid
    runs = _greedy_runs(cost, lo, hi, k, hi_b, cap, bytes_fn)
    return runs, max(cost(a, b) for a, b in runs)


def _solve_stage(blk: _Block, n: int, coord_cost, worker_cost,
                 cap: Optional[float]) -> Tuple[Optional[List[Run]], float]:
    """Optimal plan for one objective: coordinator run [0, c0) and N-1
    worker runs over [c0, n). coord_cost(0, c0) is increasing in c0 and the
    workers' min-max is decreasing, so the optimum is at their crossing —
    found by bisection over the feasible c0 range."""
    if n == 1:
        return [(0, blk.n)], coord_cost(0, blk.n)
    c_lo, c_hi = 0, blk.n
    if cap is not None:
        # coordinator run must fit; the worker section must be partitionable
        while c_hi > 0 and blk.bytes(0, c_hi) > cap:
            c_hi -= 1
        lo_ok = 0
        # smallest c0 whose worker section fits into n-1 capped runs (monotone)
        a, b = 0, blk.n
        while a < b:
            mid = (a + b) // 2
            if _greedy_runs(lambda x, y: 0.0, mid, blk.n, n - 1, 0.0, cap, blk.bytes) is not None:
                b = mid
            else:
                a = mid + 1
        lo_ok = a
        c_lo = max(c_lo, lo_ok)
        if c_lo > c_hi:
            return None, math.inf

    def total(c0):
        runs, wmax = _minmax_partition(worker_cost, c0, blk.n, n - 1, cap, blk.bytes)
        if runs is None:
            return None, math.inf, math.inf
        return runs, coord_cost(0, c0), wmax

    # bisect for the smallest c0 with coord >= workers' min-max
    a, b = c_lo, c_hi
    while a < b:
        mid = (a + b) // 2
        _, cc, ww = total(mid)
        if cc >= ww:
            b = mid
        else:
            a = mid + 1
    best = None
    for c0 in (a - 1, a, a + 1):
        if c_lo <= c0 <= c_hi:
            runs, cc, ww = total(c0)
            if runs is None:
                continue
            t = max(cc, ww)
            if best is None or t < best[1]:
                best = ([(0, c0)] + runs, t)
    return best if best else (None, math.inf)


# --------------------------------------------------------------- evaluate --

def stages(m: Manifest, mp: MachineProfile,
           encode_share: float = ENCODE_SHARE_OF_A,
           bytes_per_param: Optional[float] = None) -> Optional[Stages]:
    A = mp.get("prove_constants", "A_ns_per_slot")
    B = mp.get("prove_constants", "B_ns_per_cid")
    C = mp.get("prove_constants", "C_ns_per_product")
    if None in (A, B, C):
        return None
    assert 0.0 <= encode_share <= 1.0
    t = totals(m)
    ELL = m.run.get("ligero", {}).get("ELL", 8192)
    blk = _Block(m, bytes_per_param, ELL)
    W_fresh = t.W - t.W_weights
    tA = A * W_fresh * 1e-9
    return Stages(
        commit=encode_share * tA,
        fresh_fold=(1.0 - encode_share) * tA + B * t.cids * 1e-9 + C * t.Q * 1e-9,
        fresh_open=ENROLLED_OPEN_RATIO * A * W_fresh * 1e-9,
        w_fold=ENROLLED_QLIN_RATIO * A * blk.total_phys * 1e-9,
        w_open=ENROLLED_OPEN_RATIO * A * blk.total_phys * 1e-9,
    )


@dataclass
class Interval:
    device: int
    var_lo: int
    var_hi: int
    first: str
    last: str
    slots: float           # physical
    packed_bytes: float


def _intervals(blk: _Block, runs: Sequence[Run]) -> List[Interval]:
    return [Interval(d, lo, hi,
                     blk.vars[lo].name if hi > lo else "",
                     blk.vars[hi - 1].name if hi > lo else "",
                     blk.phys(lo, hi), blk.bytes(lo, hi))
            for d, (lo, hi) in enumerate(runs)]


def _explicit_plan(blk: _Block, n: int, x: float) -> List[Run]:
    """Coordinator share x, workers equal PHYSICAL-slot shares (heuristic)."""
    if n == 1:
        return [(0, blk.n)]
    cuts = [0, blk.cut_at_fraction(x)]
    for d in range(1, n - 1):
        cuts.append(blk.cut_at_fraction(x + d * (1 - x) / (n - 1), cuts[-1]))
    cuts.append(blk.n)
    return [(cuts[d], cuts[d + 1]) for d in range(n)]


def evaluate(m: Manifest, mp: MachineProfile, n: int, *,
             x_fold: Optional[float] = None, x_open: Optional[float] = None,
             static: bool = False, bytes_per_param: Optional[float] = None,
             resident: bool = False, disk_GBps: Optional[float] = None,
             disk_mode: str = "shared", io_overlap: str = "none",
             workspace_GB: float = 10.0,
             encode_share: float = ENCODE_SHARE_OF_A) -> dict:
    """Stage-aware wall for N devices from EXECUTABLE plans (see module
    doc). Returns wall=None with a reason when the machine profile lacks
    what the selected storage model needs."""
    assert disk_mode in DISK_MODES and io_overlap in IO_OVERLAPS and n >= 1
    st = stages(m, mp, encode_share, bytes_per_param)
    if st is None:
        return dict(n=n, wall=None, floor=None,
                    reason="prove_constants not calibrated on this machine")
    disk = disk_GBps if disk_GBps is not None else mp.get("io", "disk_read_GBps")
    mem = mp.get("gpu", "mem_GB")
    if not resident and not disk:
        return dict(n=n, wall=None, floor=st.floor,
                    reason="streaming needs io.disk_read_GBps (or --disk-GBps); "
                           "use --resident for the in-HBM model")
    if resident and not mem:
        return dict(n=n, wall=None, floor=st.floor,
                    reason="resident model needs gpu.mem_GB in the machine profile")
    lig = m.run.get("ligero", {})
    ELL, T_Q = lig.get("ELL", 8192), lig.get("T_QUERIES", 40)
    blk = _Block(m, bytes_per_param, ELL)
    sto = _Storage(resident, disk, disk_mode, io_overlap)
    cap = (mem - workspace_GB) * 1e9 if resident else None
    A = mp.get("prove_constants", "A_ns_per_slot")
    rate = {"fold": ENROLLED_QLIN_RATIO * A * 1e-9, "open": ENROLLED_OPEN_RATIO * A * 1e-9}
    fresh = {"fold": st.fresh_fold, "open": st.fresh_open}

    def coord_cost(stage):
        return lambda lo, hi: sto.dev_time(fresh[stage] + rate[stage] * blk.phys(lo, hi),
                                           blk.bytes(lo, hi))

    def worker_cost(stage):
        return lambda lo, hi: sto.dev_time(rate[stage] * blk.phys(lo, hi), blk.bytes(lo, hi))

    def assess(pf: List[Run], po: List[Run], mode: str) -> dict:
        out = dict(plan_fold=pf, plan_open=po, plan_mode=mode)
        for stage, plan in (("fold", pf), ("open", po)):
            slots = [blk.phys(lo, hi) for lo, hi in plan]
            byt = [blk.bytes(lo, hi) for lo, hi in plan]
            comp = [fresh[stage] + slots[0] * rate[stage]] + [s * rate[stage] for s in slots[1:]]
            dev = [sto.dev_time(c, b) for c, b in zip(comp, byt)]
            out[stage + "_slots"] = slots
            out[stage + "_bytes"] = byt
            out[stage + "_compute"] = comp
            out[stage + "_dev"] = dev
            out[stage + "_io_total"] = (0.0 if resident else sum(byt) / (disk * 1e9))
            out[stage + "_t"] = sto.stage(dev, sum(byt))
        hold = [blk.union_bytes(a, b) for a, b in zip(pf, po)]
        fits = [h <= cap for h in hold] if cap is not None else [None] * n
        out.update(hold_bytes=hold, fits_hbm=fits,
                   feasible=(cap is None or all(fits)),
                   wall=st.commit + out["fold_t"] + out["open_t"],
                   x_fold=(blk.phys(*pf[0]) / blk.total_phys) if blk.total_phys else 1.0,
                   x_open=(blk.phys(*po[0]) / blk.total_phys) if blk.total_phys else 1.0)
        return out

    if n == 1:
        best = assess([(0, blk.n)], [(0, blk.n)], "single")
    elif x_fold is not None or x_open is not None:
        pf = _explicit_plan(blk, n, x_fold) if x_fold is not None else None
        po = _explicit_plan(blk, n, x_open) if x_open is not None else None
        if pf is None:
            pf, _ = _solve_stage(blk, n, coord_cost("fold"), worker_cost("fold"), None)
        if po is None:
            po, _ = _solve_stage(blk, n, coord_cost("open"), worker_cost("open"), None)
        best = assess(pf, po, "explicit")
    else:
        cands = []
        if not static:
            pf, _ = _solve_stage(blk, n, coord_cost("fold"), worker_cost("fold"), None)
            po, _ = _solve_stage(blk, n, coord_cost("open"), worker_cost("open"), None)
            cands.append(assess(pf, po, "independent"))
        if static or not cands[0]["feasible"]:
            # tied cuts: ONE plan for both stages, under the cap. The true
            # objective is the barrier-separated wall max(fold)+max(open):
            # at N=2 there is a single cut, so it is enumerated EXACTLY;
            # at N>=3 three proxy solves (summed per-device time, and each
            # stage alone) are used and the result is labelled heuristic.
            tag = "static" if static else "tied"
            if n == 2:
                best_t = None
                for c0 in range(blk.n + 1):
                    plan = [(0, c0), (c0, blk.n)]
                    if cap is not None and max(blk.bytes(0, c0), blk.bytes(c0, blk.n)) > cap:
                        continue
                    r = assess(plan, plan, f"{tag}-exact")
                    if best_t is None or r["wall"] < best_t["wall"]:
                        best_t = r
                if best_t is not None:
                    cands.append(best_t)
            else:
                def tied(stage_list):
                    cc = lambda lo, hi: sum(coord_cost(s)(lo, hi) for s in stage_list)
                    wc = lambda lo, hi: sum(worker_cost(s)(lo, hi) for s in stage_list)
                    plan, _ = _solve_stage(blk, n, cc, wc, cap)
                    return assess(plan, plan, f"{tag}-heuristic") if plan else None
                for sl in (("fold", "open"), ("fold",), ("open",)):
                    r = tied(sl)
                    if r is not None:
                        cands.append(r)
            if not any(c["feasible"] for c in cands) and cap is not None:
                # nothing fits: tied cuts minimising the max hold
                plan, _ = _solve_stage(blk, n, blk.bytes, blk.bytes, None)
                if plan:
                    cands.append(assess(plan, plan, "min-hold"))
        best = min(cands, key=lambda c: (not c["feasible"],
                                         max(c["hold_bytes"]) if not c["feasible"] else 0.0,
                                         c["wall"]))
        if not best["feasible"]:
            # chosen by minimum max-hold (then wall): say so, whichever
            # candidate family produced it
            best["plan_mode"] = f"least-infeasible({best['plan_mode']})"
    # same-storage N=1 wall for the M1b metric — only if that N=1 run is
    # itself executable (resident: the whole block must fit one device)
    single = best if n == 1 else assess([(0, blk.n)], [(0, blk.n)], "single")
    n1 = single["wall"] if single["feasible"] else None
    best.update(
        n=n, stages=st, floor=st.floor, n1_wall_same_mode=n1,
        n1_same_mode_reason=(None if n1 is not None else
                             "N=1 does not fit HBM under this model; measure the "
                             "N=1 baseline streaming (cross-mode)"),
        kernel_floor_ratio=st.floor / best["wall"] if best["wall"] else None,
        same_mode_speedup=(n1 / best["wall"]) if (n1 is not None and best["wall"]) else None,
        static=static, storage=sto, mem_GB=mem, workspace_GB=workspace_GB,
        packed_total=blk.total_bytes, aligned=blk.aligned,
        physical_slots=blk.total_phys, logical_slots=blk.total_logical,
        open_payload=[s / ELL * T_Q * BYTES_PER_SLOT for s in best["open_slots"]],
        intervals_fold=_intervals(blk, best["plan_fold"]),
        intervals_open=_intervals(blk, best["plan_open"]),
        encode_share=encode_share,
    )
    return best


# ----------------------------------------------------------------- report --

def report(m: Manifest, mp: MachineProfile, gpus: Sequence[int], **kw) -> str:
    L = []
    enc = kw.get("encode_share", ENCODE_SHARE_OF_A)
    st = stages(m, mp, enc, kw.get("bytes_per_param"))
    L.append(f"== weight-split (stage-aware, enrolled block, executable plans) — {mp.name} ==")
    if st is None:
        L.append("  UNAVAILABLE — prove_constants not calibrated on this machine")
        return "\n".join(L)
    resident = kw.get("resident", False)
    if resident:
        mode = "RESIDENT (packed weights in HBM; hold = union of a device's fold+open runs)"
    else:
        mode = (f"STREAM disk={kw.get('disk_mode', 'shared')} "
                f"overlap={kw.get('io_overlap', 'none')} — packed-byte scheduling "
                f"LOWER BOUND (decode/transform and loader group amplification unpriced)")
    bpp = kw.get("bytes_per_param")
    L.append(f"  storage: {mode}")
    L.append(f"  packed bytes/param: "
             f"{bpp if bpp is not None else 'per-variable (packed_bytes, quant, else Q4_K)'}")
    L.append(f"  N=1 kernel floor {_fmt_s(st.floor)} = commit {_fmt_s(st.commit)}"
             f" + fold {_fmt_s(st.fresh_fold + st.w_fold)} (fresh {_fmt_s(st.fresh_fold)}"
             f", W {_fmt_s(st.w_fold)}) + open {_fmt_s(st.fresh_open + st.w_open)}"
             f" (fresh {_fmt_s(st.fresh_open)}, W {_fmt_s(st.w_open)})"
             f"  [encode share of A = {enc:.3f}]")
    L.append("  EXCLUDES semantic sweeps, orchestration, worker start-up, "
             "duplicate compile, opening hand-off (all coordinator-side)")
    L.append("")
    L.append(f"  {'N':>2}  {'x_fold':>6} {'x_open':>6}  {'fold':>8} {'open':>8} "
             f"{'wall':>8} {'floor/w':>7} {'same-mode':>9}  {'max hold':>9}  {'HBM':>4}  "
             f"{'payload':>9}  {'I/O-bound':>9}  plan")
    rows = []
    for n in gpus:
        ev = evaluate(m, mp, n, **kw)
        if ev["wall"] is None:
            L.append(f"  {n:>2}  UNAVAILABLE — {ev['reason']}")
            continue
        rows.append(ev)
        hold = max(ev["hold_bytes"])
        fit_s = ("ok" if ev["feasible"] else "NO") if resident else "-"
        db = []
        if not resident:
            if ev["fold_t"] > max(ev["fold_compute"]) + 1e-9:
                db.append("fold")
            if ev["open_t"] > max(ev["open_compute"]) + 1e-9:
                db.append("open")
        pay = max(ev["open_payload"][1:]) if n > 1 else 0.0
        sm = (f"{ev['same_mode_speedup']:>8.2f}x" if ev["same_mode_speedup"] is not None
              else f"{'n/a':>9}")
        L.append(f"  {n:>2}  {ev['x_fold']:>6.3f} {ev['x_open']:>6.3f}  "
                 f"{ev['fold_t']:>8.1f} {ev['open_t']:>8.1f} "
                 f"{ev['wall']:>8.1f} {ev['kernel_floor_ratio']:>6.2f}x "
                 f"{sm}  "
                 f"{_gb(hold):>9}  {fit_s:>4}  {_gb(pay):>9}  "
                 f"{(','.join(db) or 'no'):>9}  {ev['plan_mode']}")
    L.append("")
    L.append("  floor/w = N=1 compute-only kernel floor / wall; same-mode = N=1 wall under "
             "this storage model / wall (the M1b measurement), n/a when that N=1 run is "
             "not executable (resident: block does not fit one device — measure the N=1 "
             "baseline streaming). payload = largest worker's opened-column bytes "
             "(physical rows x T_QUERIES x 8).")
    L.append("  plans: 'independent' = per-stage cuts solved exactly over variable "
             "boundaries, workers' runs unequal where that balances time"
             + (" and packed bytes; when the HBM cap binds on the fold/open unions the "
                "cuts are tied across stages: 'tied-exact' (N=2, enumerated) or "
                "'tied-heuristic' (N>=3, proxy objectives); 'least-infeasible(...)' = "
                "min max hold, nothing fits" if resident else "")
             + "; '--static' gives 'static-exact' (N=2) / 'static-heuristic' (N>=3); "
               "'explicit' = --x-* with equal-slot worker shares (heuristic).")
    if resident:
        L.append(f"  HBM cap: hold <= mem_GB {mp.get('gpu', 'mem_GB')} - workspace "
                 f"{kw.get('workspace_GB', 10.0)} GB.")
    if rows and not rows[0]["aligned"]:
        r = rows[0]
        L.append(f"  NOTE: enrolled variables are not ELL-aligned — physical slots "
                 f"{r['physical_slots']:.4g} vs logical {r['logical_slots']:.4g}; W terms, "
                 f"cuts and payloads use the padded rows the prover allocates.")
    if any(r["n"] == 2 for r in rows):
        sens = []
        for e in (0.0, enc, 1.0):
            ev = evaluate(m, mp, 2, **dict(kw, encode_share=e))
            sens.append(f"{e:.3f} -> {ev['kernel_floor_ratio']:.2f}x" if ev["wall"]
                        else f"{e:.3f} -> n/a")
        L.append("  encode-share sensitivity, N=2 floor/wall: " + "; ".join(sens))
    return "\n".join(L)


def intervals_text(ev: dict) -> str:
    L = [f"  ownership plans for N={ev['n']} ({ev['plan_mode']}; contiguous, whole "
         f"variables, layout order; slots are physical):"]
    for stage in ("fold", "open"):
        L.append(f"   {stage} stage (x={ev['x_' + stage]:.3f}):")
        for iv in ev["intervals_" + stage]:
            role = "coordinator" if iv.device == 0 else f"worker {iv.device}"
            L.append(f"    {role:>12}: vars [{iv.var_lo}, {iv.var_hi})  "
                     f"{iv.slots:.3e} slots  {_gb(iv.packed_bytes)} packed  "
                     f"{iv.first} .. {iv.last}")
    L.append("   HBM hold (union of the two runs per device): " +
             ", ".join(f"dev{d} {_gb(h)}" for d, h in enumerate(ev["hold_bytes"])))
    return "\n".join(L)
