"""Multilinear sumcheck + the affine mask compiler of doc §7.

Two pieces.

`prove` / `verify` are an ordinary sumcheck for a PRODUCT of multilinear
polynomials given by their hypercube evaluations. Degree per round = number of
factors, which covers the degree-2 relations the doc leans on (Hadamard,
booleanity, raw-rescale brackets) and the degree-3 binary-gate terminals.

`MaskTape` / `mask_round` are §7.1. Every secret scalar travels as a public
masked value  x_hat = x + mu. For a degree-d round carrying mask mu the prover
draws d fresh tape fields u_1..u_d and sends g + h where

    h(X) = a_0 + sum_{k=1..d} u_k X^k,    a_0 = (mu - sum_k u_k) / 2.

Then h(0) + h(1) = mu, so the masked polynomial satisfies the masked claim
exactly as the bare one satisfies the bare claim, and after the challenge r the
carried mask becomes h(r). The doc's hiding argument is that
u -> free coefficients is triangular and full rank, hence the transmitted
polynomial is uniform in the affine space cut out by the previous masked claim
and nothing else. `solve_tape_for_target` inverts that map, and the test uses it
to show every polynomial in the affine space is hit by exactly one tape -- which
is the hiding claim, checked rather than quoted.

Soundness is unaffected: subtracting the authenticated masks turns an accepting
masked transcript into an accepting ordinary one, and `test_sumcheck.py` runs
both paths on the same instance to show they accept and reject together.
"""
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import blake3

from prover.protocol import P as FIELD_P, inv

from .counters import charge

INV2 = inv(2)


# ── multilinear helpers ──────────────────────────────────────────────────────
def mle_eval(values: Sequence[int], point: Sequence[int]) -> int:
    """Evaluate the multilinear extension of `values` (2^n hypercube evals, the
    variable order is most-significant-first) at `point`."""
    charge(mul=max(len(values) - 1, 0), add=2 * max(len(values) - 1, 0))
    # Production sampled-local proofs keep the authenticated wire vectors on
    # device.  Folding them through Python scalar iteration would synchronize
    # once per element, so use the bit-exact CUDA fold already shared by the
    # prover whenever a CUDA tensor reaches the verifier.
    if getattr(values, "is_cuda", False):
        import torch

        from . import gpu
        cur = values.detach().contiguous().view(-1).to(torch.uint64)
        for r in point:
            rt = torch.tensor([int(r) % FIELD_P], dtype=torch.uint64,
                              device=cur.device)
            cur = gpu.fold_t(cur, rt)
        return int(cur[0].item()) % FIELD_P
    cur = list(values)
    for r in point:
        half = len(cur) // 2
        cur = [(cur[i] + r * (cur[half + i] - cur[i])) % FIELD_P for i in range(half)]
    return cur[0] % FIELD_P


def _fold(values: List[int], r: int) -> List[int]:
    half = len(values) // 2
    return [(values[i] + r * (values[half + i] - values[i])) % FIELD_P
            for i in range(half)]


def _lagrange_interpolate(points: Sequence[Tuple[int, int]], x: int) -> int:
    """Evaluate the polynomial through `points` at x."""
    n = len(points)
    charge(mul=n * (2 * (n - 1) + 2) + n * 95, add=2 * n * (n - 1), inv=n)
    acc = 0
    for i, (xi, yi) in enumerate(points):
        num, den = 1, 1
        for k, (xk, _) in enumerate(points):
            if k == i:
                continue
            num = (num * (x - xk)) % FIELD_P
            den = (den * (xi - xk)) % FIELD_P
        acc = (acc + yi * num * inv(den % FIELD_P)) % FIELD_P
    return acc


# ── mask tape (§7.1) ─────────────────────────────────────────────────────────
@dataclass
class MaskTape:
    """Fresh field elements, committed before any challenge (the mask root)."""
    values: List[int]
    pos: int = 0

    def take(self, n: int) -> List[int]:
        if self.pos + n > len(self.values):
            raise RuntimeError("mask tape exhausted")
        out = self.values[self.pos:self.pos + n]
        self.pos += n
        return out


def mask_poly_coeffs(mu: int, u: Sequence[int]) -> List[int]:
    """h's ascending coefficients: [a_0, u_1, ..., u_d] with h(0)+h(1) = mu."""
    a0 = ((mu - sum(u)) % FIELD_P) * INV2 % FIELD_P
    return [a0] + [x % FIELD_P for x in u]


def poly_eval_coeffs(coeffs: Sequence[int], x: int) -> int:
    acc = 0
    for c in reversed(coeffs):
        acc = (acc * x + c) % FIELD_P
    return acc


def solve_tape_for_target(mu: int, target_h: Sequence[int]) -> Optional[List[int]]:
    """Invert the tape -> polynomial map. Given a desired h (ascending coeffs of
    degree d with h(0)+h(1) = mu), return the unique tape u that produces it, or
    None if h is not in the affine space. This is the full-rank/triangularity
    claim of §7.1, made executable."""
    if not target_h:
        return None
    if (poly_eval_coeffs(target_h, 0) + poly_eval_coeffs(target_h, 1)) % FIELD_P != mu % FIELD_P:
        return None
    u = [c % FIELD_P for c in target_h[1:]]
    return u if mask_poly_coeffs(mu, u) == [c % FIELD_P for c in target_h] else None


# ── sumcheck ─────────────────────────────────────────────────────────────────
@dataclass
class SumcheckProof:
    claim: int
    round_polys: List[List[Tuple[int, int]]] = field(default_factory=list)  # (x, y) samples
    challenges: List[int] = field(default_factory=list)
    final_point: List[int] = field(default_factory=list)
    masked: bool = False
    final_mask: int = 0
    n_terms: int = 1


Term = Tuple[int, Sequence[Sequence[int]]]     # (coefficient, factor list)


class RoundTranscript:
    """Fiat-Shamir coins for ONE sumcheck, a coin per round: round r's
    challenge is drawn from a running digest that has absorbed the binding
    context (the committed block, the relation, whatever the caller names)
    and every round polynomial up to and including round r's. A prover who
    knows a round's challenge before sending its polynomial can steer an
    incorrect claim onto the true terminal value; with this coin the
    polynomial is fixed first. Pass one to prove_terms/verify_terms in place
    of a coin function; the verifier builds its own from the same context.
    Drawing a coin before its round is absorbed raises."""

    DOMAIN = b"verinf/sumcheck-rounds/v1"

    def __init__(self, *context: bytes):
        h = blake3.blake3()
        for part in (self.DOMAIN, *context):
            h.update(len(part).to_bytes(8, "little") + bytes(part))
        self._state = h.digest()
        self._absorbed = 0

    def absorb(self, rnd: int, samples: Sequence[Tuple[int, int]]) -> None:
        if rnd != self._absorbed:
            raise ValueError(f"round {rnd} absorbed out of order "
                             f"(next is {self._absorbed})")
        h = blake3.blake3(self._state)
        h.update(rnd.to_bytes(8, "little") + len(samples).to_bytes(8, "little"))
        for x, y in samples:
            h.update((int(x) % FIELD_P).to_bytes(8, "little")
                     + (int(y) % FIELD_P).to_bytes(8, "little"))
        self._state = h.digest()
        self._absorbed += 1

    def __call__(self, rnd: int) -> int:
        if rnd != self._absorbed - 1:
            raise ValueError(f"round {rnd}'s coin drawn before its polynomial")
        d = blake3.blake3(self._state + b"coin").digest()
        return int.from_bytes(d[:16], "little") % FIELD_P


def draw_coin(coin, rnd: int, samples: Sequence[Tuple[int, int]]) -> int:
    """Round rnd's challenge, after its (transmitted) polynomial: a
    RoundTranscript absorbs the samples first; a plain function of the round
    index (the layergkr prototypes' pre-drawn coins) is called as before."""
    if isinstance(coin, RoundTranscript):
        coin.absorb(rnd, samples)
    return coin(rnd) % FIELD_P

GPU_MIN_SUMCHECK_WORK = 1 << 16     # size * total factors
_SC_GPU: dict = {}


def _sumcheck_gpu_ok() -> bool:
    """Enable the device path only once it has produced a proof identical to the
    CPython one. Checked once per process; the gate is the same discipline as
    rs._gpu_ok -- a faster sumcheck that computes something else is worthless."""
    if "ok" not in _SC_GPU:
        from . import gpu
        if not gpu.available():
            _SC_GPU["ok"] = False
        else:
            import random as _r
            rr = _r.Random(1)
            f = [[rr.randrange(FIELD_P) for _ in range(64)] for _ in range(2)]
            terms = [(1, [f[0], f[1]]), (FIELD_P - 1, [f[0]])]
            coins = [rr.randrange(FIELD_P) for _ in range(6)]
            cpu = prove_terms(terms, lambda i: coins[i])
            claim, polys, chals, _ = gpu.prove_terms_gpu(terms, lambda i: coins[i], FIELD_P)
            _SC_GPU["ok"] = (claim == cpu.claim and polys == cpu.round_polys
                             and chals == cpu.challenges)
    return _SC_GPU["ok"]


def prove_terms(terms: Sequence[Term], coin: Callable[[int], int],
                tape: Optional[MaskTape] = None, mu0: int = 0) -> SumcheckProof:
    """Sumcheck for  sum_x  sum_k coeff_k * prod_j terms[k].factors[j](x).

    A sum of products, not just one product: that is what the relation zoo needs
    (a Hadamard gate is a*b - c, a booleanity gate is b*b - b, a rescale bracket
    is raw - q*s - r), and it is the 'one tagged ragged sumcheck batches the
    layer's relations' of doc §4 L3. Round degree is the widest term.
    """
    if not terms:
        raise ValueError("no terms")
    size = len(terms[0][1][0])
    # Elementwise rounds over a big domain are the GPU's job. Masked proofs stay
    # on the CPU: the mask tape is drawn per round and the saving does not apply.
    work = size * sum(len(fs) for _, fs in terms)
    if work >= GPU_MIN_SUMCHECK_WORK and _sumcheck_gpu_ok():
        from . import gpu
        mk = (lambda d, mu: mask_poly_coeffs(mu, tape.take(d))) if tape is not None else None
        claim, polys, chals, mu = gpu.prove_terms_gpu(terms, coin, FIELD_P, mk, mu0)
        pf = SumcheckProof(claim=claim, round_polys=polys, challenges=chals,
                           final_point=list(chals), masked=tape is not None,
                           final_mask=mu)
        pf.n_terms = len(terms)
        return pf
    n = size.bit_length() - 1
    deg = max(len(f) for _, f in terms)
    cur = [(c % FIELD_P, [list(f) for f in fs]) for c, fs in terms]

    true_claim = 0
    for idx in range(size):
        acc = 0
        for c, fs in cur:
            prod = c
            for f in fs:
                prod = prod * f[idx] % FIELD_P
            acc += prod
        true_claim = (true_claim + acc) % FIELD_P

    mu = mu0 % FIELD_P
    proof = SumcheckProof(claim=(true_claim + mu) % FIELD_P, masked=tape is not None)
    h: List[int] = []
    for rnd in range(n):
        half = len(cur[0][1][0]) // 2
        samples = []
        for x in range(deg + 1):
            acc = 0
            for c, fs in cur:
                for i in range(half):
                    prod = c
                    for f in fs:
                        prod = prod * ((f[i] + x * (f[half + i] - f[i])) % FIELD_P) % FIELD_P
                    acc += prod
            samples.append((x, acc % FIELD_P))
        n_fac = sum(len(fs) for _, fs in cur)
        charge(mul=(deg + 1) * half * sum(len(fs) + 1 for _, fs in cur),
               add=(deg + 1) * half * n_fac,
               # per (evaluation point, term, position, factor): one fold body
               # and one reduced multiply into the running product
               fold_iter=(deg + 1) * half * n_fac,
               red_op=(deg + 1) * half * n_fac)
        if tape is not None:
            h = mask_poly_coeffs(mu, tape.take(deg))
            samples = [(x, (y + poly_eval_coeffs(h, x)) % FIELD_P) for x, y in samples]
        proof.round_polys.append(samples)
        r = draw_coin(coin, rnd, samples)
        proof.challenges.append(r)
        if tape is not None:
            mu = poly_eval_coeffs(h, r)
        charge(fold_iter=half * sum(len(fs) for _, fs in cur))
        cur = [(c, [_fold(f, r) for f in fs]) for c, fs in cur]
    proof.final_point = list(proof.challenges)
    proof.final_mask = mu
    proof.n_terms = len(terms)
    return proof


def prove(factors: Sequence[Sequence[int]], coin: Callable[[int], int],
          tape: Optional[MaskTape] = None, mu0: int = 0) -> SumcheckProof:
    """Single-product convenience wrapper over `prove_terms`."""
    return prove_terms([(1, list(factors))], coin, tape, mu0)


def _expected_rounds(size: int) -> int:
    """log2 of the hypercube size; the transcript MUST carry exactly this many
    rounds. Without the check a proof with fewer rounds folds fewer variables
    and the terminal is evaluated at a short point (mle_eval folds only over
    the point it is given), so a zero-round transcript is accepted whenever
    the sum of products at index 0 alone matches the claim."""
    if size <= 0 or size & (size - 1):
        raise ValueError(f"sumcheck domain size {size} is not a power of two")
    return size.bit_length() - 1


def _rounds_ok(proof: SumcheckProof, n: int):
    if len(proof.round_polys) != n or len(proof.challenges) != n \
            or len(proof.final_point) != n:
        return False, (f"expected {n} rounds, transcript has "
                       f"{len(proof.round_polys)} polys / {len(proof.challenges)} "
                       f"challenges / {len(proof.final_point)} point coordinates")
    return True, "ok"


def verify_terms(proof: SumcheckProof, terms: Sequence[Term],
                 coin: Callable[[int], int]) -> Tuple[bool, str]:
    """Check the round chain and the terminal evaluation of a sum-of-products
    sumcheck. For a masked proof the terminal check subtracts the carried mask --
    that subtraction is the 'authenticated masks' step of the ZK argument."""
    claim = proof.claim % FIELD_P
    sizes = {len(f) for _, fs in terms for f in fs}
    if len(sizes) != 1:
        return False, f"factors of unequal size {sorted(sizes)}"
    ok, why = _rounds_ok(proof, _expected_rounds(sizes.pop()))
    if not ok:
        return False, why
    deg = max(len(f) for _, f in terms)
    for rnd, samples in enumerate(proof.round_polys):
        if len(samples) != deg + 1:
            return False, f"round {rnd}: expected {deg + 1} samples"
        g0 = _lagrange_interpolate(samples, 0)
        g1 = _lagrange_interpolate(samples, 1)
        if (g0 + g1) % FIELD_P != claim:
            return False, f"round {rnd}: g(0)+g(1) != claim"
        r = draw_coin(coin, rnd, samples)
        if r != proof.challenges[rnd]:
            return False, f"round {rnd}: challenge mismatch"
        claim = _lagrange_interpolate(samples, r)
    # the terminal is evaluated at the round challenges and nowhere else: a
    # proof-chosen point could sit where the relation happens to hold
    if list(proof.final_point) != list(proof.challenges):
        return False, "terminal point is not the round challenges"
    terminal = 0
    for c, fs in terms:
        prod = c % FIELD_P
        for f in fs:
            prod = prod * mle_eval(f, proof.final_point) % FIELD_P
        terminal = (terminal + prod) % FIELD_P
    expected = (terminal + proof.final_mask) % FIELD_P if proof.masked else terminal
    if claim != expected:
        return False, "terminal evaluation mismatch"
    return True, "ok"


def verify(proof: SumcheckProof, factors: Sequence[Sequence[int]],
           coin: Callable[[int], int]) -> Tuple[bool, str]:
    """Single-product convenience wrapper over `verify_terms`."""
    claim = proof.claim % FIELD_P
    sizes = {len(f) for f in factors}
    if len(sizes) != 1:
        return False, f"factors of unequal size {sorted(sizes)}"
    ok, why = _rounds_ok(proof, _expected_rounds(sizes.pop()))
    if not ok:
        return False, why
    deg = len(factors)
    for rnd, samples in enumerate(proof.round_polys):
        if len(samples) != deg + 1:
            return False, f"round {rnd}: expected {deg + 1} samples"
        g0 = _lagrange_interpolate(samples, 0)
        g1 = _lagrange_interpolate(samples, 1)
        if (g0 + g1) % FIELD_P != claim:
            return False, f"round {rnd}: g(0)+g(1) != claim"
        r = draw_coin(coin, rnd, samples)
        if r != proof.challenges[rnd]:
            return False, f"round {rnd}: challenge mismatch"
        claim = _lagrange_interpolate(samples, r)
    # the terminal is evaluated at the round challenges and nowhere else: a
    # proof-chosen point could sit where the relation happens to hold
    if list(proof.final_point) != list(proof.challenges):
        return False, "terminal point is not the round challenges"
    terminal = 1
    for f in factors:
        terminal = terminal * mle_eval(f, proof.final_point) % FIELD_P
    expected = (terminal + proof.final_mask) % FIELD_P if proof.masked else terminal
    if claim != expected:
        return False, "terminal evaluation mismatch"
    return True, "ok"


# ── §7.2 masked products ─────────────────────────────────────────────────────
def masked_product_ok(x: int, y: int, a: int, b: int, c: int) -> bool:
    """z = xy carried as X = x+a, Y = y+b, Z = z+c must satisfy
    Z - c = XY - Xb - Ya + ab. q_lin proves the affine part, q_quad the mask
    products; two multiplication boundaries per cubic terminal, never four."""
    X, Y, Z = (x + a) % FIELD_P, (y + b) % FIELD_P, (x * y + c) % FIELD_P
    lhs = (Z - c) % FIELD_P
    rhs = (X * Y - X * b - Y * a + a * b) % FIELD_P
    return lhs == rhs
