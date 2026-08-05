"""Primitive-operation counters — the unit of account that replaces kappa.

The previous version of the cost model priced a protocol in seconds and then
multiplied by kappa to cover whatever it had failed to count. That is a fudge
factor standing in for an incomplete model. This module removes the need for it
by separating two things that were conflated:

    COUNTS  (field muls, field adds, BLAKE3 bytes, opened values, proof bytes)
            -- a property of the PROTOCOL and the instance geometry. Hardware
               independent. Predictable exactly, and checkable exactly, because
               the implementation reports what it actually did.

    RATES   (ns per field mul, GB/s of hashing, ...) -- a property of the CARD,
            measured once by `bench/rates.py`, not fitted per run.

    time = counts . rates

A model that predicts counts can be VALIDATED to the op, not to within 50%.
`bench/run_toy.py` proves real instances and compares predicted counts against
measured ones; any gap is a missing term in the model, and gets added, rather
than being absorbed into a multiplier.

Usage:

    with Counter("round1") as c:
        ...
    c.report()          # {'mul': ..., 'add': ..., 'hash_bytes': ...}
"""
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional

# The active counter, if any. Deliberately a module-level stack: the counted
# field ops in field.py must not need a handle threaded through every call.
_STACK: List["Counter"] = []


@dataclass
class Counter:
    """Accumulates primitive counts for a named phase. Nesting is additive:
    a child's counts also land in every enclosing counter, so a phase total is
    the sum of its sub-phases by construction."""
    name: str = ""
    mul: int = 0            # field muls WITH reduction: (a*b) % P
    mul_defer: int = 0      # field muls with DEFERRED reduction: a*b accumulated
                            # into a wide sum, reduced once at the end. The hot
                            # loops (encode, linear combination, projection) work
                            # this way, and it is roughly half the cost of a
                            # reduced multiply -- so counting them as the same
                            # primitive is a modelling error, not a rounding one.
    add: int = 0            # field additions/subtractions
    # KERNEL ITERATIONS -- the unit that actually predicts time. "field mul" and
    # "field add" are not independently identifiable in this code (they occur in
    # a fixed ratio, so a fit over the real workload cannot separate them), and
    # worse, the abstract pair does not correspond to any loop the code runs.
    # These four DO: each is one iteration of a specific loop body, measured as
    # written by bench/kernels.py.
    enc_slot: int = 0       # acc += v * L[c]   -- NONZERO slots only
    enc_scan: int = 0       # slots visited by the encode loop, zero or not
    comb_iter: int = 0      # out[j] = (out[j] + cw[j]*a) % P
    fold_iter: int = 0      # (f[i] + x*(f[h+i] - f[i])) % P
    red_op: int = 0         # (a * b) % P
    xpose_iter: int = 0     # column-build list work in rs.Commit
    gpu_hash_value: int = 0 # values hashed into Merkle leaves ON THE DEVICE
                            # (pack + BLAKE3 fused in one kernel), replacing the
                            # per-value CPython packing that profiling showed was
                            # the single largest cost in the prover.
    pack_value: int = 0     # values packed to bytes for a Merkle leaf. Profiling
                            # a real prove showed this is the LARGEST single cost;
                            # it was charged per hash CALL, but it is per VALUE.
    gpu_elem: int = 0       # elements moved device->host and marshalled
    enc_gpu: int = 0        # slots encoded by the GPU matmul path. Kept SEPARATE
                            # from enc_slot/enc_scan because gl_matmul is DENSE:
                            # the CPU loop's zero-skipping saving does not exist
                            # there, and the per-slot cost differs by ~10^4. Same
                            # protocol, different machine, different card.
    inv: int = 0            # field inversions (each also charges its muls)
    hash_bytes: int = 0     # bytes fed to BLAKE3
    hash_calls: int = 0
    opened_values: int = 0  # field values opened to the verifier
    proof_bytes: int = 0
    seconds: float = 0.0    # wall time of this phase, so predicted-vs-measured
                            # can be compared in SECONDS per phase, not just in
                            # counts -- that is what localises a wrong rate.
    children: Dict[str, "Counter"] = field(default_factory=dict)
    _t0: float = 0.0

    # ── accounting ───────────────────────────────────────────────────────────
    def charge(self, mul: int = 0, mul_defer: int = 0, add: int = 0, inv: int = 0,
               hash_bytes: int = 0, hash_calls: int = 0,
               opened_values: int = 0, proof_bytes: int = 0,
               enc_slot: int = 0, enc_scan: int = 0, comb_iter: int = 0,
               fold_iter: int = 0, red_op: int = 0, xpose_iter: int = 0,
               enc_gpu: int = 0, gpu_elem: int = 0, pack_value: int = 0,
               gpu_hash_value: int = 0) -> None:
        self.mul += mul
        self.mul_defer += mul_defer
        self.add += add
        self.enc_slot += enc_slot
        self.enc_scan += enc_scan
        self.comb_iter += comb_iter
        self.fold_iter += fold_iter
        self.red_op += red_op
        self.xpose_iter += xpose_iter
        self.enc_gpu += enc_gpu
        self.gpu_elem += gpu_elem
        self.pack_value += pack_value
        self.gpu_hash_value += gpu_hash_value
        self.inv += inv
        self.hash_bytes += hash_bytes
        self.hash_calls += hash_calls
        self.opened_values += opened_values
        self.proof_bytes += proof_bytes

    def merge(self, other: "Counter") -> None:
        self.charge(other.mul, other.mul_defer, other.add, other.inv,
                    other.hash_bytes, other.hash_calls, other.opened_values,
                    other.proof_bytes, other.enc_slot, other.enc_scan,
                    other.comb_iter, other.fold_iter, other.red_op,
                    other.xpose_iter, other.enc_gpu,
                    other.gpu_elem, other.pack_value,
                    other.gpu_hash_value)   # seconds NOT summed:
        # a nested phase's time is already inside the parent's wall clock.

    def report(self) -> Dict[str, int]:
        return {"mul": self.mul, "mul_defer": self.mul_defer, "add": self.add,
                "inv": self.inv,
                "hash_bytes": self.hash_bytes, "hash_calls": self.hash_calls,
                "opened_values": self.opened_values, "proof_bytes": self.proof_bytes,
                "enc_slot": self.enc_slot, "enc_scan": self.enc_scan,
                "comb_iter": self.comb_iter,
                "fold_iter": self.fold_iter, "red_op": self.red_op,
                "xpose_iter": self.xpose_iter, "enc_gpu": self.enc_gpu,
                "gpu_elem": self.gpu_elem,
                "pack_value": self.pack_value,
                "gpu_hash_value": self.gpu_hash_value,
                "seconds": self.seconds}

    def flat(self, prefix: str = "") -> Dict[str, Dict[str, int]]:
        """This counter and every descendant, keyed by dotted path."""
        key = f"{prefix}{self.name}" if prefix else self.name
        out = {key: self.report()}
        for ch in self.children.values():
            out.update(ch.flat(f"{key}." if key else ""))
        return out

    # ── context management ───────────────────────────────────────────────────
    def __enter__(self) -> "Counter":
        self._t0 = time.perf_counter()
        _STACK.append(self)
        return self

    def __exit__(self, *exc) -> None:
        self.seconds = time.perf_counter() - self._t0
        _STACK.pop()
        if _STACK:
            _STACK[-1].merge(self)
            _STACK[-1].children[self.name] = self


def charge(**kw) -> None:
    """Charge the innermost active counter, if any. Cheap no-op when counting
    is off, so instrumentation can stay in the hot path."""
    if _STACK:
        _STACK[-1].charge(**kw)


def active() -> Optional[Counter]:
    return _STACK[-1] if _STACK else None


@contextmanager
def phase(name: str) -> Iterator[Counter]:
    """Named sub-phase of the enclosing counter."""
    c = Counter(name)
    with c:
        yield c


# ── the rate card: counts -> seconds ─────────────────────────────────────────
@dataclass(frozen=True)
class Rates:
    """Per-primitive cost on a specific machine. Measured by bench/rates.py --
    NOT fitted to a prove run, so it cannot silently absorb a modelling error.

    `mul_ns` is per field multiplication, `add_ns` per addition, `hash_GBps` the
    BLAKE3 throughput, `disk_MBps`/`net_MBps` the proof drain rates."""
    name: str
    mul_ns: float
    mul_defer_ns: float
    add_ns: float
    hash_GBps: float
    disk_MBps: float = 108.0
    net_MBps: float = 125.0

    def seconds(self, counts: Dict[str, int]) -> float:
        t = counts.get("mul", 0) * self.mul_ns * 1e-9
        t += counts.get("mul_defer", 0) * self.mul_defer_ns * 1e-9
        t += counts.get("add", 0) * self.add_ns * 1e-9
        t += counts.get("hash_bytes", 0) / (self.hash_GBps * 1e9)
        return t

    def drain_seconds(self, proof_bytes: int) -> float:
        """Streamed proof egress; disk and network read one tee in parallel, so
        the slower of the two drains bounds it (doc §9.2)."""
        return proof_bytes / (min(self.disk_MBps, self.net_MBps) * 1e6)


@dataclass(frozen=True)
class KernelRates:
    """Cost per LOOP ITERATION, measured by bench/kernels.py on this machine.

    This is the machine model that replaces both the kappa and the abstract
    mul/add card. Each field is the measured cost of one iteration of a loop the
    code actually contains -- so predicted seconds are a sum over real work, with
    no free parameter anywhere."""
    name: str
    enc_ns: float          # acc += v * L[c]  (a NONZERO slot)
    scan_ns: float         # a slot the encode loop visits and skips
    comb_ns: float         # out[j] = (out[j] + cw[j]*a) % P
    fold_ns: float         # (f[i] + x*(f[h+i] - f[i])) % P
    red_ns: float          # (a * b) % P
    gpu_enc_ns: float      # one slot through the GPU encode matmul (DENSE)
    gpu_elem_ns: float     # one element moved device->host and marshalled
    pack_ns: float         # one value packed into a Merkle leaf (CPU)
    gpu_hash_ns: float     # one value hashed on the device
    xpose_ns: float        # one element of the column transpose
    hash_call_ns: float    # pack + BLAKE3 of one column (call overhead)
    hash_GBps: float

    def seconds(self, c: Dict[str, int]) -> float:
        t = c.get("enc_slot", 0) * self.enc_ns * 1e-9
        t += max(c.get("enc_scan", 0) - c.get("enc_slot", 0), 0) * self.scan_ns * 1e-9
        t += c.get("comb_iter", 0) * self.comb_ns * 1e-9
        t += c.get("fold_iter", 0) * self.fold_ns * 1e-9
        t += (c.get("red_op", 0) + c.get("inv", 0) * 95) * self.red_ns * 1e-9
        t += c.get("enc_gpu", 0) * self.gpu_enc_ns * 1e-9
        t += c.get("gpu_elem", 0) * self.gpu_elem_ns * 1e-9
        t += c.get("pack_value", 0) * self.pack_ns * 1e-9
        t += c.get("gpu_hash_value", 0) * self.gpu_hash_ns * 1e-9
        t += c.get("xpose_iter", 0) * self.xpose_ns * 1e-9
        t += c.get("hash_calls", 0) * self.hash_call_ns * 1e-9
        t += c.get("hash_bytes", 0) / (self.hash_GBps * 1e9)
        return t
