"""Weight-split shard plans (multi-GPU milestone 1). Torch-free.

A plan says which device folds and opens which run of the ENROLLED weight
variables, per stage. Runs are contiguous, whole-variable intervals over
the W block in LAYOUT order (core._layout's `weight_vars`) — the only shape
the streaming primitives can execute exactly: `_iter_message_chunks` packs
its list as one contiguous sequence and a `ColumnSink` is one contiguous
range, so ownership must never compact across a gap. The two stages (the
test-polynomial fold and the column opening, separated by the s_col
transcript barrier) may cut differently.

Device 0 is the coordinator (the process running prove_streaming); it
folds/opens its own run inside its sweeps and merges the workers' partials.
Validation rejects gaps, overlaps, out-of-range or non-monotone runs, and
a device with two runs in one stage, BEFORE any weight row is touched.

The profiler's `weightsplit.evaluate(...)` returns `plan_fold`/`plan_open`
as lists of (lo, hi) indexed by device — `ShardPlan.from_pairs` takes
them directly.
"""
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

Run = Tuple[int, int, int]      # (device, var_lo, var_hi) — hi exclusive


def validate_runs(runs: Sequence[Run], n_vars: int, stage: str = "") -> List[Run]:
    """Check that `runs` tile [0, n_vars) exactly, in order, one run per
    device at most (empty runs allowed). Returns the runs sorted by lo."""
    tag = f"{stage} plan: " if stage else ""
    if n_vars < 0:
        raise ValueError(f"{tag}negative variable count")
    seen_dev = set()
    for r in runs:
        if not isinstance(r, (tuple, list)) or len(r) != 3:
            raise ValueError(f"{tag}run must be (device, lo, hi), got {r!r}")
        dev, lo, hi = r
        # indices must be true ints: a float cut (3.0) or a bool would pass
        # the range test and fail much later inside a slice
        for name, x in (("device", dev), ("lo", lo), ("hi", hi)):
            if not isinstance(x, int) or isinstance(x, bool):
                raise ValueError(f"{tag}{name} must be an int, got {x!r}")
        if dev < 0:
            raise ValueError(f"{tag}bad device {dev!r}")
        if not (0 <= lo <= hi <= n_vars):
            raise ValueError(f"{tag}run [{lo}, {hi}) outside [0, {n_vars}) "
                             f"or reversed (device {dev})")
        if dev in seen_dev:
            raise ValueError(f"{tag}device {dev} has two runs in one stage")
        seen_dev.add(dev)
    ordered = sorted(runs, key=lambda r: (r[1], r[2]))
    pos = 0
    for dev, lo, hi in ordered:
        if hi == lo:
            continue
        if lo > pos:
            raise ValueError(f"{tag}variables [{pos}, {lo}) are owned by nobody")
        if lo < pos:
            raise ValueError(f"{tag}variables [{lo}, {pos}) are owned twice "
                             f"(device {dev} overlaps the previous run)")
        pos = hi
    if pos != n_vars:
        raise ValueError(f"{tag}variables [{pos}, {n_vars}) are owned by nobody")
    return list(ordered)


@dataclass
class ShardPlan:
    fold: List[Run]
    open: List[Run]
    # device index -> torch device spec the worker should run on (None: the
    # current device; the in-process single-GPU gate uses None everywhere)
    devices: Dict[int, Optional[str]] = field(default_factory=dict)

    @staticmethod
    def from_pairs(fold_pairs: Sequence[Tuple[int, int]],
                   open_pairs: Optional[Sequence[Tuple[int, int]]] = None,
                   devices: Optional[Dict[int, Optional[str]]] = None) -> "ShardPlan":
        """Device d gets fold_pairs[d] / open_pairs[d] (the profiler's
        `plan_fold` / `plan_open`). open_pairs defaults to fold_pairs."""
        if open_pairs is None:
            open_pairs = fold_pairs
        fold = [(d, lo, hi) for d, (lo, hi) in enumerate(fold_pairs)]
        open_ = [(d, lo, hi) for d, (lo, hi) in enumerate(open_pairs)]
        return ShardPlan(fold=fold, open=open_, devices=dict(devices or {}))

    @staticmethod
    def two_way(cut_fold: int, cut_open: Optional[int] = None) -> "ShardPlan":
        """N=2 convenience: coordinator owns [0, cut), the worker the rest.
        The variable count is checked at validation time (n_vars is not
        known here), so `hi` is filled in by `validated`."""
        if cut_open is None:
            cut_open = cut_fold
        return ShardPlan(fold=[(0, 0, cut_fold), (1, cut_fold, -1)],
                         open=[(0, 0, cut_open), (1, cut_open, -1)])

    def validated(self, n_vars: int) -> "ShardPlan":
        def fill(runs):
            # Expand two_way's `hi` sentinel ONLY when it is the true int -1:
            # a float -1.0 (or any malformed run) is passed through untouched
            # so validate_runs raises its intended error, never a tuple-
            # unpacking or masked-type one.
            out = []
            for r in runs:
                if (isinstance(r, (tuple, list)) and len(r) == 3
                        and isinstance(r[2], int) and not isinstance(r[2], bool)
                        and r[2] == -1):
                    r = (r[0], r[1], n_vars)
                out.append(r)
            return out
        fold = validate_runs(fill(self.fold), n_vars, "fold")
        open_ = validate_runs(fill(self.open), n_vars, "open")
        return ShardPlan(fold=fold, open=open_, devices=dict(self.devices))

    def runs(self, stage: str) -> List[Run]:
        if stage == "fold":
            return self.fold
        if stage == "open":
            return self.open
        raise ValueError(f"unknown stage {stage!r}")

    def worker_runs(self, stage: str) -> List[Run]:
        """Non-empty runs of every device but the coordinator, in row order."""
        return [(d, lo, hi) for d, lo, hi in self.runs(stage) if d != 0 and hi > lo]

    def coordinator_run(self, stage: str) -> Tuple[int, int]:
        for d, lo, hi in self.runs(stage):
            if d == 0:
                return lo, hi
        return 0, 0

    def owned_ids(self, device: int, stage: str, weight_vars: Sequence) -> set:
        """{id(Variable)} this device owns in `stage` — the filter
        `_stream_sweep(w_owned=...)` applies to each claim's weight group."""
        for d, lo, hi in self.runs(stage):
            if d == device:
                return {id(v) for v in weight_vars[lo:hi]}
        return set()

    def device_of(self, device: int) -> Optional[str]:
        return self.devices.get(device)

    def n_devices(self) -> int:
        return 1 + max([d for d, _, _ in self.fold + self.open] or [0])


def as_plan(obj, n_vars: int) -> ShardPlan:
    """Accept a ShardPlan or a (fold_pairs, open_pairs) tuple; validate
    against the enrolled block's variable count."""
    if isinstance(obj, ShardPlan):
        return obj.validated(n_vars)
    if isinstance(obj, (tuple, list)) and len(obj) == 2:
        return ShardPlan.from_pairs(obj[0], obj[1]).validated(n_vars)
    raise TypeError("shard_plan must be a ShardPlan or (fold_pairs, open_pairs)")
