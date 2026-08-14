"""Per-stage timing built into the prover, with the residual made EXPLICIT.

Twice in this project a plausible story about where the time goes turned out to be
wrong, and both times the fix was measurement rather than a better story. The
second one is the reason this module exists: a fitted `C*N` coefficient of
0.469 ns was attributed to the encode path, and a direct measurement of that path
came out at 0.0294 ns -- 16x less. The gap had nowhere to show up, because nothing
was reporting what the prover actually spent its time on.

So the rule here is: **every stage is timed, and whatever is left over is printed
as a line called `unattributed`.** A breakdown that sums to less than the wall
clock is not a breakdown; showing the residual is what makes it one.

Two clocks, because they answer different questions:

  * `wall`   host time the stage occupied -- what the user waits for.
  * `device` CUDA time inside it, measured with events. A stage with a large wall
             and a small device time is waiting on Python, transfers or sync.
             Reported as a share of the wall, or `async` when the launch outlives
             the host call that issued it -- a percentage would be nonsense there.

Usage mirrors `counters.phase`:

    with profile.timeline("prove") as tl:
        with profile.stage("commit.states"):
            ...
    print(tl.report())

Off by default and free when off: `stage()` is a no-op unless a timeline is
active, so the instrumentation can live in the hot path.
"""
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator, List, Optional

import time

_STACK: List["Timeline"] = []


@dataclass
class Span:
    name: str
    depth: int
    wall_s: float = 0.0
    device_ms: float = 0.0
    n: int = 1
    events: list = field(default_factory=list)   # resolved once, at timeline exit


@dataclass
class Timeline:
    """Ordered spans with nesting depth, plus the residual at every level."""
    name: str = "total"
    spans: List[Span] = field(default_factory=list)
    _depth: int = 0
    _t0: float = 0.0
    wall_s: float = 0.0

    # ── recording ────────────────────────────────────────────────────────────
    def add(self, name: str, depth: int, wall_s: float, ev=None) -> None:
        for sp in self.spans:
            if sp.name == name and sp.depth == depth:
                sp.wall_s += wall_s
                sp.n += 1
                if ev:
                    sp.events.append(ev)
                return
        sp = Span(name, depth, wall_s, 0.0)
        if ev:
            sp.events.append(ev)
        self.spans.append(sp)

    def resolve(self) -> None:
        """Turn the recorded CUDA events into milliseconds, with ONE
        synchronisation for the whole timeline.

        Doing it per stage was a measurement artefact with teeth: `wall` was taken
        before the sync, so every stage's synchronisation cost fell OUTSIDE its own
        span and landed in `unattributed` -- the profiler was charging its own
        overhead to the residual, and with 700+ spans that was ~11% of the run.
        Resolving once at the end also removes the serialisation the per-stage sync
        imposed on the device."""
        for sp in self.spans:
            for a, b in sp.events:
                try:
                    b.synchronize()
                    sp.device_ms += a.elapsed_time(b)
                except Exception:
                    pass
            sp.events = []

    # ── reporting ────────────────────────────────────────────────────────────
    def report(self, min_share: float = 0.0) -> str:
        total = self.wall_s or sum(s.wall_s for s in self.spans if s.depth == 0)
        out = [f"{'stage':<34}{'calls':>6}{'wall s':>10}{'%':>7}"
               f"{'device ms':>11}{'on dev':>7}",
               "-" * 75]
        top = [s for s in self.spans if s.depth == 0]
        for sp in self.spans:
            share = sp.wall_s / total * 100 if total else 0.0
            if share < min_share:
                continue
            # A CUDA launch is asynchronous: the host returns immediately while
            # the device keeps working, so device time can exceed the host wall of
            # the stage that issued it. Printing that as a percentage produces
            # nonsense like 2841%, so async stages are labelled instead.
            ratio = sp.device_ms / 1e3 / sp.wall_s if sp.wall_s else 0.0
            dev = "async" if ratio > 1.05 else f"{ratio * 100:5.1f}%"
            out.append(f"{'  ' * sp.depth + sp.name:<34}{sp.n:>6}"
                       f"{sp.wall_s:>10.2f}{share:>6.1f}%"
                       f"{sp.device_ms:>11.1f}{dev:>7}")
        acc = sum(s.wall_s for s in top)
        resid = total - acc
        out.append("-" * 75)
        out.append(f"{'UNATTRIBUTED':<34}{'':>6}{resid:>10.2f}"
                   f"{(resid / total * 100 if total else 0):>6.1f}%")
        out.append(f"{'TOTAL':<34}{'':>6}{total:>10.2f}{100.0:>6.1f}%")
        if total and resid / total > 0.10:
            out.append("")
            out.append("!! more than 10% of the wall clock is unattributed. Do not")
            out.append("!! reason about where the time goes from this table -- add a")
            out.append("!! stage around the missing work first.")
        return "\n".join(out)


def active() -> Optional[Timeline]:
    return _STACK[-1] if _STACK else None


@contextmanager
def timeline(name: str = "total") -> Iterator[Timeline]:
    tl = Timeline(name)
    _STACK.append(tl)
    t0 = time.perf_counter()
    try:
        yield tl
    finally:
        tl.wall_s = time.perf_counter() - t0
        tl.resolve()
        _STACK.pop()


@contextmanager
def stage(name: str, device: bool = True) -> Iterator[None]:
    """Time a stage. `device=True` also brackets it with CUDA events, so a stage
    that is slow but not busy on the card is visible as such."""
    tl = active()
    if tl is None:
        yield
        return
    ev = None
    if device:
        ev = _events()
        if ev:
            ev[0].record()
    tl._depth += 1
    t0 = time.perf_counter()
    try:
        yield
    finally:
        if ev:
            ev[1].record()          # record only; resolved at timeline exit
        wall = time.perf_counter() - t0
        tl._depth -= 1
        tl.add(name, tl._depth, wall, ev)


@contextmanager
def no_gc() -> Iterator[None]:
    """Run a proof with CYCLIC garbage collection off.

    The prover allocates millions of short-lived Python lists with reference
    cycles for CPython's cycle collector to walk. Order-controlled measurement on
    a d=128 layer, alternating after a discarded warm-up:

        gc OFF   11.57 12.27 12.42 11.82    mean 12.02 s
        gc ON    13.29 13.09 13.32 13.07    mean 13.19 s

    **~9%.** An earlier note here claimed 43%; that came from a paired run whose
    second half was also warmer, and it was wrong. Worth switching off for a batch
    proof, not a headline. It is a property of the PROTOTYPE (Python objects), not
    of the protocol -- a prover holding its data in tensors gives the collector
    nothing to walk.

    Reference counting still frees everything as it goes; only the cycle detector
    is paused, so memory does not grow without bound for the duration of a proof.
    """
    import gc
    was = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was:
            gc.enable()


_TORCH = None


def _events():
    """(start, end) CUDA events, or None without CUDA. Cached import."""
    global _TORCH
    if _TORCH is None:
        try:
            import torch
            _TORCH = torch if torch.cuda.is_available() else False
        except Exception:
            _TORCH = False
    if not _TORCH:
        return None
    return (_TORCH.cuda.Event(enable_timing=True),
            _TORCH.cuda.Event(enable_timing=True))
