"""Append-only transcript with an ENFORCED commit-before-challenge schedule.

The doc's §3.2 spends a page on ordering because every soundness hole it closes
is an ordering hole: LogUp's beta must land after the raw tuples are bound, alpha
after the compressed ones, the projected codeword R_P before the contraction
sumcheck's first input challenge, all local polynomials before the column coin.
Prose cannot enforce that. This module makes the order a data structure.

Two properties, both exercised by the tests:

  * Coins are a function of everything absorbed before them, in order. A prover
    who commits a root only after seeing a coin produces a transcript whose coins
    the verifier recomputes differently -> reject. This is what closes the causal
    counterexample of §4 (choose W-projection after the challenge).

  * A declared `Schedule` names the legal sequence of steps. Absorbing or
    drawing out of order raises `CausalityError` on the PROVER side too, so a
    mis-ordered implementation fails loudly at development time instead of
    silently producing an unsound-but-accepted proof.
"""
from typing import List, Optional, Sequence

import blake3

from prover.protocol import P

from .profile import stage


class CausalityError(RuntimeError):
    """Raised when a step is taken out of the declared schedule order."""


class Schedule:
    """The legal step order for one layer proof. Steps are (kind, label) with
    kind in {'absorb', 'coin'}."""

    def __init__(self, steps: Sequence[tuple]):
        self.steps = list(steps)
        self.pos = 0

    def check(self, kind: str, label: str) -> None:
        if self.pos >= len(self.steps):
            raise CausalityError(f"schedule exhausted; got extra {kind}:{label}")
        want_kind, want_label = self.steps[self.pos]
        if (kind, label) != (want_kind, want_label):
            raise CausalityError(
                f"step {self.pos}: expected {want_kind}:{want_label}, got {kind}:{label}")
        self.pos += 1

    def finished(self) -> bool:
        return self.pos == len(self.steps)


# The layer-local order of doc §3.2/§4. `contraction_*` is the project-before-
# sumcheck seam: R_P is absorbed BEFORE the input-dimension challenge exists.
LAYER_SCHEDULE = [
    ("absorb", "R_out"), ("absorb", "R_lk"), ("absorb", "R_sort"), ("absorb", "R_mask"),
    ("coin", "beta"),                 # local LogUp tuple compression
    ("absorb", "R_cmp"),              # compressed fingerprints / table values
    ("coin", "alpha"),                # reciprocal point, only now
    ("coin", "rho"),                  # output evaluation point of the matmul
    ("absorb", "R_P"),                # projected codeword — BEFORE any input coin
    ("coin", "contraction"),          # input-dimension sumcheck challenges
    ("absorb", "R_terminal"),         # terminal / mask-product commitments
    ("coin", "columns"),              # only after every polynomial is fixed
]


class Transcript:
    """Fiat-Shamir over an append-only log. `schedule` is optional; when given,
    every call is checked against it."""

    def __init__(self, domain: bytes = b"layergkr/v0",
                 schedule: Optional[Schedule] = None):
        self._h = blake3.blake3(domain)
        self.schedule = schedule
        self.log: List[tuple] = []

    def absorb(self, label: str, data: bytes) -> None:
        # Timed: Fiat-Shamir happens all over the prover and was invisible, which
        # is exactly the kind of work that ends up in the `unattributed` line.
        with stage("transcript", device=False):
            if self.schedule is not None:
                self.schedule.check("absorb", label)
            self._h.update(len(label).to_bytes(2, "little") + label.encode())
            self._h.update(len(data).to_bytes(8, "little") + data)
            self.log.append(("absorb", label, data))

    def absorb_root(self, label: str, root: bytes) -> None:
        self.absorb(label, root)

    def absorb_ints(self, label: str, values: Sequence[int]) -> None:
        self.absorb(label, b"".join(int(v).to_bytes(8, "little") for v in values))

    def coin(self, label: str, count: int = 1) -> List[int]:
        """Draw `count` field challenges. The digest at this point covers exactly
        what has been absorbed so far — nothing committed later can influence it,
        and nothing committed later can be back-dated into it."""
        with stage("transcript", device=False):
            if self.schedule is not None:
                self.schedule.check("coin", label)
            state = self._h.digest()
            self._h.update(b"\x00coin" + len(label).to_bytes(2, "little")
                           + label.encode())
            self.log.append(("coin", label, count))
            out = []
            for i in range(count):
                d = blake3.blake3(state + label.encode()
                                  + i.to_bytes(8, "little")).digest()
                out.append(int.from_bytes(d[:16], "little") % P)
            return out

    def coin_bytes(self, label: str) -> bytes:
        """A 32-byte coin, for seeding column selection."""
        if self.schedule is not None:
            self.schedule.check("coin", label)
        state = self._h.digest()
        self._h.update(b"\x00coin" + len(label).to_bytes(2, "little") + label.encode())
        self.log.append(("coin", label, 0))
        return blake3.blake3(state + label.encode() + b"cols").digest()

    def digest(self) -> bytes:
        return self._h.digest()


def replay(log: Sequence[tuple], domain: bytes = b"layergkr/v0",
           schedule: Optional[Schedule] = None) -> "Transcript":
    """Verifier side: re-run a claimed transcript log and return the transcript,
    so its coins can be compared against the ones the prover says it used. A
    prover that reordered anything produces different coins here."""
    t = Transcript(domain, schedule)
    for entry in log:
        if entry[0] == "absorb":
            t.absorb(entry[1], entry[2])
        else:
            _, label, count = entry
            t.coin_bytes(label) if count == 0 else t.coin(label, count)
    return t
