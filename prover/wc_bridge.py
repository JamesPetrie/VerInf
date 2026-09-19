"""WC-LCRL-STC part 1+2: coefficient-RS weight enrollment and the late
coefficient bridge (analysis/wc-lcrl-stc-spec.md, §0.1–§0.4).

The enrollment replaces online authentication of persistent weights: every
fixed linear map is packed, per output width n, into blocks of B input
coordinates; block a / output coordinate j becomes the polynomial

    F_{n,a,j}(X) = sum_{b<B} W[aB+b, j] X^b + sum_{h<lam} z[a,j,h] X^{B+h}

with lam independent random mask coefficients on top (K_w = B + lam), RS-coded
on a domain of N_w points, stored column-major and bound by one Merkle root.

The bridge (post-R2 coins): one shared rho per output width, per-block alpha,
q_w distinct RS-domain points eta.  The prover commits the semantic projection
P_trace = W rho and the projected masks pi = z^T rho, aggregates
U = P_trace|pi over blocks into c = sum alpha_a U_a, and sends c plus
v_l = c(eta_l).  The verifier opens the enrollment columns at the same eta_l
and checks  v_l = sum_a alpha_a sum_j rho_j F_{a,j}(eta_l)  against the root.
A wrong P_trace survives with probability <= 1/p (alpha) plus
H_qw = C(K_w-1, q_w)/C(N_w, q_w)  (all q_w points hit a nonzero polynomial of
degree < K_w) — 8.86e-13 at the production geometry.

This module is deliberately self-contained (own transcript labels, CPU
verifier mirroring the future Rust twin); wiring it into the 5-round
prove_streaming transcript and the message cache is part 3.
"""
from __future__ import annotations

import hashlib
import math

import blake3
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

import protocol as pr
from cuda_primitives import (P, gl_axpy, gl_matvec, ntt_forward,
                             ntt_forward_batched, poly_eval)
import contextlib
import time

# Where the bridge's per-proof pass goes (session 4 measured 592 s outside
# the sweeps on a B200 with no counted loader call — the enrollment streams
# its shards directly). Seconds per stage, CUDA-synced, cleared by
# wc_times_reset() and printed by the prove's closing line.
WC_TIMES: Dict[str, float] = {}
WC_COUNTS: Dict[str, int] = {}


def wc_times_reset():
    WC_TIMES.clear(); WC_COUNTS.clear()


@contextlib.contextmanager
def _timed(key: str):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        WC_TIMES[key] = WC_TIMES.get(key, 0.0) + (time.time() - t0)
        WC_COUNTS[key] = WC_COUNTS.get(key, 0) + 1


def wc_times_line() -> str:
    """One line: stage seconds and counts, largest first."""
    if not WC_TIMES:
        return "[wc-bridge] no stage timings"
    parts = [f"{k} {v:.1f} s/{WC_COUNTS.get(k, 0)}" for k, v in
             sorted(WC_TIMES.items(), key=lambda kv: -kv[1])]
    return "[wc-bridge] stages: " + "; ".join(parts)


# Production geometry (spec §0.1).  Tests shrink these; every function takes
# the params object, nothing reads the module constants directly.
@dataclass(frozen=True)
class WcParams:
    B: int = 15360          # weight coefficients per block
    lam: int = 1024         # random mask coefficients per (block, j)
    N_w: int = 32768        # RS domain size
    q_w: int = 40           # opened domain points

    @property
    def K_w(self) -> int:   # polynomial length; must be a power of two for NTT
        return self.B + self.lam

    def __post_init__(self):
        k = self.B + self.lam
        assert k & (k - 1) == 0, "K_w = B + lam must be a power of two"
        assert self.N_w & (self.N_w - 1) == 0 and self.N_w > k, \
            "N_w must be a power of two above K_w"

    def soundness_bound(self) -> float:
        """H_qw: a nonzero polynomial of degree < K_w vanishes on all q_w of
        the N_w sampled distinct points."""
        return math.comb(self.K_w - 1, self.q_w) / math.comb(self.N_w, self.q_w)

    @staticmethod
    def min_qw_for_tau(tau: int) -> int:
        """Review §7: the bridge must never be weaker than the fresh Ligero
        part — at rate 1/2 each eta point buys ~1.0001 bits vs (3/4)^tau's
        0.415 bits per audited column, so q_w >= ceil(0.416 * tau)."""
        return math.ceil(0.416 * tau)


def _params_bytes(params: WcParams) -> bytes:
    """Geometry pinned into every transcript coin (review §5.5): a proof
    produced under different (B, lam, N_w, q_w) derives different coins and
    cannot be replayed against this verifier's parameters."""
    return b"wc-geom" + b"".join(
        v.to_bytes(8, "little")
        for v in (params.B, params.lam, params.N_w, params.q_w))


class LedgerExhausted(Exception):
    """Enrollment mask budget would be exceeded (review §4.1)."""


@dataclass
class EnrollmentLedger:
    """Prover-side union of all eta points ever opened against one
    enrollment root.  Up to lam DISTINCT points the Vandermonde mask system
    is underdetermined (perfect hiding); from point lam+1 every opening is a
    linear equation on the weight coefficients.  The margin rule refuses a
    proof attempt when fewer than q_w unspent points remain — a refresh
    (re-enrollment with fresh masks, deterministically re-verified against
    the same canonical model) resets the budget."""
    lam: int
    spent: set = field(default_factory=set)

    def remaining(self) -> int:
        return self.lam - len(self.spent)

    def precheck(self, q_w: int):
        if self.remaining() < q_w:
            raise LedgerExhausted(
                f"enrollment mask budget: {self.remaining()} points remain "
                f"< q_w={q_w}; refresh the enrollment before proving")

    def charge(self, eta_idx):
        new = self.spent | set(eta_idx)
        if len(new) > self.lam:
            raise LedgerExhausted(
                f"enrollment mask budget exceeded: {len(new)} distinct "
                f"points > lam={self.lam}")
        self.spent = new


def wc_block_masks(mask_seed: bytes, width: int, block: int,
                   params: WcParams) -> torch.Tensor:
    """(width, lam) mask coefficients for one block — per-(width, block)
    blake3-seeded PRG, shared by the resident and streaming builders."""
    g = torch.Generator(device="cpu")
    g.manual_seed(int.from_bytes(blake3.blake3(
        mask_seed + b"|" + width.to_bytes(8, "little")
        + block.to_bytes(8, "little")).digest()[:8], "little"))
    return (torch.randint(0, 1 << 62, (width, params.lam), generator=g,
                          dtype=torch.int64).to(torch.uint64)).cuda()


def _coeffs_to_codewords(coeffs: torch.Tensor, params: WcParams) -> torch.Tensor:
    """(m, K_w) COEFFICIENT rows -> (m, N_w) evaluations on the NTT domain.
    NOTE: rs_encode_rows is evaluation-based (Ligero messages) and is the
    wrong primitive here; coefficient-RS is zero-pad + forward NTT."""
    m = coeffs.size(0)
    padded = torch.zeros(m, params.N_w, dtype=torch.uint64, device="cuda")
    padded[:, :params.K_w] = coeffs
    ntt_forward_batched(padded)
    return padded


def _rs_domain(params: WcParams) -> torch.Tensor:
    """The NTT evaluation domain IN OUTPUT ORDER, read off by transforming
    the polynomial X: its evaluations are exactly the domain points.  Immune
    to whatever ordering (natural / bit-reversed) the kernel uses, because
    prover and verifier both work in codeword index space."""
    x = torch.zeros(params.N_w, dtype=torch.uint64, device="cuda")
    x[1] = 1
    ntt_forward(x)
    # protocol-pinned order: natural powers of omega = 7^((P-1)/N_w) — the
    # Rust twin recomputes the domain from this formula, so a kernel that
    # changes its output order must fail HERE, not fork the two verifiers.
    omega = pow(7, (P - 1) // params.N_w, P)
    assert int(x[1].item()) == omega and int(x[2].item()) == omega * omega % P, \
        "NTT kernel domain order changed — the Rust twin's formula would fork"
    return x


def _leaf_from_inner(inner32: bytes) -> bytes:
    """leaf = blake3("wc-leaf" || blake3(column bytes)): the inner hash is
    exactly what core's GPU column accumulator computes (streamed BLAKE3 ==
    one-shot), the outer wrap keeps the enrollment tree domain-separated.
    Both twins mirror this two-level definition."""
    return blake3.blake3(b"wc-leaf" + inner32).digest()


def _leaf(column_u64: torch.Tensor) -> bytes:
    inner = blake3.blake3(column_u64.cpu().numpy().tobytes()).digest()
    return _leaf_from_inner(inner)


def _tree(leaves: List[bytes]) -> List[List[bytes]]:
    levels = [leaves]
    while len(levels[-1]) > 1:
        lo = levels[-1]
        if len(lo) & 1:
            lo = lo + [lo[-1]]
        levels.append([blake3.blake3(lo[i] + lo[i + 1]).digest()
                       for i in range(0, len(lo), 2)])
    return levels


def _path(levels: List[List[bytes]], idx: int) -> List[Tuple[bytes, int]]:
    out = []
    for lvl in levels[:-1]:
        sib = idx ^ 1
        if sib >= len(lvl):
            sib = idx
        out.append((lvl[sib], idx & 1))
        idx >>= 1
    return out


def _verify_path(leaf: bytes, path: List[Tuple[bytes, int]], root: bytes) -> bool:
    h = leaf
    for sib, is_right in path:
        h = blake3.blake3((sib + h) if is_right else (h + sib)).digest()
    return h == root


@dataclass
class EnrolledGroup:
    width: int                     # n — output width shared by the group
    n_rows: int                    # concatenated input coords (pre-padding)
    n_blocks: int
    weights: torch.Tensor          # (rows_padded, n) uint64 cuda — decoded W
    masks: torch.Tensor            # (n_blocks, n, lam) uint64 cuda
    codewords: torch.Tensor        # (n_blocks * n, N_w) uint64 cuda


@dataclass
class Enrollment:
    params: WcParams
    groups: Dict[int, EnrolledGroup]      # width -> group, iterated sorted
    root: bytes
    levels: List[List[bytes]] = field(repr=False)
    manifest_digest: bytes = b""

    def poly_count(self) -> int:
        return sum(g.n_blocks * g.width for g in self.groups.values())


def build_enrollment(weight_groups: Dict[int, torch.Tensor],
                     mask_seed: bytes,
                     manifest: bytes,
                     params: WcParams) -> Enrollment:
    """weight_groups: width n -> (rows, n) uint64 CUDA tensor of decoded
    weights (input coords concatenated across all maps of that width, spec
    §0.1).  Rows are zero-padded to a multiple of B — the manifest records
    the true row count, so padding is not free weight material."""
    groups: Dict[int, EnrolledGroup] = {}
    for gi, n in enumerate(sorted(weight_groups)):
        W = weight_groups[n]
        assert W.dim() == 2 and W.size(1) == n and W.is_cuda
        rows = W.size(0)
        n_blocks = -(-rows // params.B)
        pad = n_blocks * params.B - rows
        if pad:
            W = torch.cat([W, torch.zeros(pad, n, dtype=torch.uint64,
                                          device="cuda")])
        # independent masks, PRG-seeded per (width, block) — the SAME
        # convention LazyEnrollment streams with, so both builds agree bit
        # for bit on the root
        masks = torch.stack([wc_block_masks(mask_seed, n, a, params)
                             for a in range(n_blocks)])
        # coefficient rows: one polynomial per (block a, output j)
        coeffs = torch.empty(n_blocks * n, params.K_w, dtype=torch.uint64,
                             device="cuda")
        for a in range(n_blocks):
            blk = W[a * params.B:(a + 1) * params.B]          # (B, n)
            coeffs[a * n:(a + 1) * n, :params.B] = blk.T      # (n, B)
            coeffs[a * n:(a + 1) * n, params.B:] = masks[a]   # (n, lam)
        codewords = _coeffs_to_codewords(coeffs, params)
        groups[n] = EnrolledGroup(n, rows, n_blocks, W, masks, codewords)

    all_cw = torch.cat([groups[n].codewords for n in sorted(groups)])
    leaves = [_leaf(all_cw[:, i]) for i in range(params.N_w)]
    levels = _tree(leaves)
    return Enrollment(params, groups, levels[-1][0], levels,
                      hashlib.sha256(manifest).digest())


# ---------------------------------------------------------------------------
# Bridge — prover side
# ---------------------------------------------------------------------------

@dataclass
class BridgeProof:
    rho: Dict[int, List[int]]              # width -> rho^(n)
    p_trace: Dict[int, torch.Tensor]       # width -> (rows_padded,) = W rho
    pi: Dict[int, torch.Tensor]            # width -> (n_blocks, lam)
    c: List[int]                           # aggregated coefficients, len K_w
    v: List[int]                           # c(eta_l), len q_w
    eta_idx: List[int]                     # opened domain indices
    opened: Dict[int, List[int]]           # domain idx -> full column values
    paths: Dict[int, List[Tuple[bytes, int]]]


def _commit_r2(p_trace: Dict[int, torch.Tensor],
               pi: Dict[int, torch.Tensor]) -> bytes:
    h = blake3.blake3(b"wc-r2")
    for n in sorted(p_trace):
        h.update(p_trace[n].cpu().numpy().tobytes())
        h.update(pi[n].cpu().numpy().tobytes())
    return h.digest()


def bridge_r2(enr: Enrollment, rho: Dict[int, List[int]]):
    """R2 phase with INJECTED coins: the host transcript supplies rho (one
    per output width, sampled after its real R1).  Returns the values the
    host must commit in its R2: P_trace = W rho per block and the projected
    masks pi = z^T rho.  This is the integration surface — the standalone
    prove_bridge wrapper derives rho itself for tests/benches."""
    p_trace, pi = {}, {}
    for n, g in enr.groups.items():
        rho_t = torch.tensor(rho[n], dtype=torch.uint64, device="cuda")
        p_trace[n] = gl_matvec(g.weights, rho_t)              # (rows_padded,)
        # pi[a,h] = sum_j masks[a,j,h] * rho_j
        pi[n] = torch.stack([
            gl_matvec(g.masks[a].T.contiguous(), rho_t)       # (lam,)
            for a in range(g.n_blocks)])
    return p_trace, pi


def prove_bridge(enr: Enrollment, s_r1: bytes,
                 ledger: Optional[EnrollmentLedger] = None) -> BridgeProof:
    """s_r1: transcript seed AFTER R1 (all semantic outputs fixed) — rho must
    not be derivable earlier (spec §0.2).  alpha/eta are derived only after
    the R2 commitment of P_trace and pi (spec §0.4).  With a ledger, the
    mask budget is prechecked before any work and charged with the drawn
    eta set (review §4.1)."""
    params = enr.params
    if ledger is not None:
        ledger.precheck(params.q_w)
    # --- coins after R1: one shared rho per output width --------------------
    s_rho = pr.fs_seed("wc/rho", s_r1, enr.root, enr.manifest_digest,
                       _params_bytes(params))
    rho: Dict[int, List[int]] = {}
    for gi, n in enumerate(sorted(enr.groups)):
        rho[n] = pr.op_vec(s_rho, gi, "rho", n)
    # --- R2: semantic projections and projected masks -----------------------
    p_trace, pi = bridge_r2(enr, rho)
    # --- coins after R2: alpha per block, q_w distinct eta ------------------
    s_late = pr.fs_seed("wc/late", s_rho, _commit_r2(p_trace, pi))
    return bridge_r3(enr, rho, p_trace, pi, s_late, ledger)


def bridge_r3(enr: Enrollment, rho, p_trace, pi, s_late: bytes,
              ledger: Optional[EnrollmentLedger] = None) -> BridgeProof:
    """Post-R2 phase with an INJECTED late seed: the host transcript derives
    s_late from its own state PLUS the R2 commitment of (P_trace, pi), then
    this computes alpha aggregation, the eta set, v = c(eta) and the
    enrollment openings.  Standalone prove_bridge wraps it."""
    params = enr.params
    c = torch.zeros(params.K_w, dtype=torch.uint64, device="cuda")
    bi = 0
    with _timed("aggregate c"):
        for n in sorted(enr.groups):
            g = enr.groups[n]
            for a in range(g.n_blocks):
                alpha = pr.challenge(s_late, bi, "alpha")
                u = torch.cat([p_trace[n][a * params.B:(a + 1) * params.B],
                               pi[n][a]])                          # (K_w,)
                gl_axpy(c, alpha, u)                               # c += alpha*u mod P
                bi += 1
    eta_idx = pr.random_columns_n(pr.fs_seed("wc/eta", s_late),
                                  params.q_w, params.N_w)
    if ledger is not None:
        ledger.charge(eta_idx)
    with _timed("eval v"):
        domain = _rs_domain(params)
        # uint64 CUDA tensors lack fancy indexing; gather via a bit-preserving
        # int64 view (values are raw 64-bit field words either way).
        idx = torch.tensor(eta_idx, dtype=torch.long, device="cuda")
        eta_pts = domain.view(torch.int64)[idx].view(torch.uint64)
        v = poly_eval(c, eta_pts).cpu().tolist()
    # --- openings ------------------------------------------------------------
    if isinstance(enr, LazyEnrollment):
        opened, paths = enr.open_columns(eta_idx)
    else:
        all_cw = torch.cat([enr.groups[n].codewords for n in sorted(enr.groups)])
        opened = {i: all_cw[:, i].cpu().tolist() for i in eta_idx}
        paths = {i: _path(enr.levels, i) for i in eta_idx}
    return BridgeProof(rho, p_trace, pi, c.cpu().tolist(), v,
                       eta_idx, opened, paths)


class ChainError(Exception):
    """Spec §0.8: a persistent block is not in exactly one
    GGUF -> F -> W rho -> P_trace -> terminal-constraint chain."""


class ChainRegistry:
    """Compile-time enforcement of the §0.8 invariant.  Terminal constraints
    register the P_trace slice they consume; the SAME tensor object the
    bridge committed must be passed — an unlinked copy (different storage)
    is the classic bridge bug and raises immediately.  finalize() fails
    closed if any block was consumed zero or more than one times."""

    def __init__(self, enr: Enrollment, proof: BridgeProof):
        self._B = enr.params.B
        self._canon = {n: proof.p_trace[n] for n in enr.groups}
        self._blocks = {n: enr.groups[n].n_blocks for n in enr.groups}
        self._consumed: Dict[Tuple[int, int], int] = {}

    def consume(self, width: int, block: int, tensor: torch.Tensor):
        canon = self._canon[width]
        if tensor.data_ptr() != canon.data_ptr():
            raise ChainError(
                f"terminal constraint for width {width} reads an UNLINKED "
                f"COPY of P_trace (different storage) — spec 0.3 forbids it")
        if not (0 <= block < self._blocks[width]):
            raise ChainError(f"width {width} has no block {block}")
        key = (width, block)
        self._consumed[key] = self._consumed.get(key, 0) + 1
        if self._consumed[key] > 1:
            raise ChainError(
                f"P_trace block {key} consumed twice — not exactly one chain")

    def finalize(self):
        missing = [(n, a) for n in self._blocks for a in range(self._blocks[n])
                   if (n, a) not in self._consumed]
        if missing:
            raise ChainError(
                f"persistent blocks outside any chain: {missing[:4]}"
                f"{'...' if len(missing) > 4 else ''} — compile must fail")


# ---------------------------------------------------------------------------
# Bridge — verifier side (CPU, python ints; mirrors the future Rust twin)
# ---------------------------------------------------------------------------

def hosted_s_late(s_bind: bytes, root: bytes, manifest_digest: bytes,
                  p_trace, pi, params: WcParams) -> bytes:
    """The late coin when the bridge lives INSIDE the 5-round transcript:
    derived from the host's s_bind (which exists only after the real R2
    root) plus the bridge's own R2 commitment of (P_trace, pi), the
    enrollment identity and the geometry.  Interim binding for pi until it
    becomes committed R2 rows proved by fresh qLin (spec §0.4 full form)."""
    return pr.fs_seed("wc/hosted-late", s_bind, root, manifest_digest,
                      _params_bytes(params), _commit_r2(p_trace, pi))


def verify_bridge_hosted(root: bytes, manifest_digest: bytes,
                         group_meta: Dict[int, Tuple[int, int]],
                         proof: BridgeProof, s_bind: bytes,
                         expected_rho: Dict[int, List[int]],
                         params: WcParams) -> Tuple[bool, str]:
    """Hosted-mode verify: rho is the HOST transcript's coin (the claim's
    routed_sample output, recomputed by the host verifier) — the bridge
    checks identity against it instead of deriving its own."""
    for n in sorted(group_meta):
        if proof.rho.get(n) != list(expected_rho[n]):
            return False, "rho mismatch vs host transcript"
    s_late = hosted_s_late(s_bind, root, manifest_digest,
                           proof.p_trace, proof.pi, params)
    return _verify_core(root, group_meta, proof, s_late, params)


def verify_bridge(root: bytes, manifest_digest: bytes,
                  group_meta: Dict[int, Tuple[int, int]],   # width->(blocks,n)
                  proof: BridgeProof, s_r1: bytes,
                  params: WcParams) -> Tuple[bool, str]:
    # recompute every coin — none is trusted from the proof (spec §0.4);
    # the geometry is part of every coin (review §5.5)
    s_rho = pr.fs_seed("wc/rho", s_r1, root, manifest_digest,
                       _params_bytes(params))
    for gi, n in enumerate(sorted(group_meta)):
        if proof.rho[n] != pr.op_vec(s_rho, gi, "rho", n):
            return False, "rho mismatch"
    s_late = pr.fs_seed("wc/late", s_rho, _commit_r2(proof.p_trace, proof.pi))
    return _verify_core(root, group_meta, proof, s_late, params)


def _verify_core(root: bytes, group_meta: Dict[int, Tuple[int, int]],
                 proof: BridgeProof, s_late: bytes,
                 params: WcParams) -> Tuple[bool, str]:
    eta_idx = pr.random_columns_n(pr.fs_seed("wc/eta", s_late),
                                  params.q_w, params.N_w)
    # H_qw assumes sampling WITHOUT replacement (review §5.3) — check the
    # CLAIMED set first so the reject reason is precise
    if len(set(proof.eta_idx)) != params.q_w:
        return False, "eta not distinct"
    if eta_idx != proof.eta_idx:
        return False, "eta mismatch"
    # c must aggregate exactly the committed P_trace|pi blocks
    c = [0] * params.K_w
    bi = 0
    for n in sorted(group_meta):
        n_blocks, _ = group_meta[n]
        pt = proof.p_trace[n].cpu().tolist()
        pim = proof.pi[n].cpu().tolist()
        for a in range(n_blocks):
            alpha = pr.challenge(s_late, bi, "alpha")
            for k in range(params.B):
                c[k] = (c[k] + alpha * pt[a * params.B + k]) % P
            for h in range(params.lam):
                c[params.B + h] = (c[params.B + h] + alpha * pim[a][h]) % P
            bi += 1
    if c != [x % P for x in proof.c]:
        return False, "c does not aggregate the committed P_trace/pi"
    # v = c(eta) and the enrollment side of the bridge
    domain = _rs_domain(params).cpu().tolist()
    for l, i in enumerate(eta_idx):
        col = proof.opened[i]
        if not _verify_path(_leaf(torch.tensor(col, dtype=torch.uint64)),
                            proof.paths[i], root):
            return False, f"merkle path fails at eta[{l}]"
        # v_l = c(eta_l)
        x, acc = domain[i], 0
        for k in reversed(range(params.K_w)):
            acc = (acc * x + c[k]) % P
        if acc != proof.v[l] % P:
            return False, f"v[{l}] != c(eta_{l})"
        # enrollment side: sum_a alpha_a sum_j rho_j F_{a,j}(eta_l)
        rhs, bi, off = 0, 0, 0
        for n in sorted(group_meta):
            n_blocks, width = group_meta[n]
            rho = proof.rho[n]
            for a in range(n_blocks):
                alpha = pr.challenge(s_late, bi, "alpha")
                s = 0
                for j in range(width):
                    s = (s + rho[j] * col[off + a * width + j]) % P
                rhs = (rhs + alpha * s) % P
                bi += 1
            off += n_blocks * width
        if rhs != proof.v[l] % P:
            return False, f"bridge equation fails at eta[{l}]"
    return True, "ACCEPT"


# ---------------------------------------------------------------------------
# Tape-level helpers (integration bricks 5+): canonical enrollment of every
# use_bridge claim's weights, and the canonical claim -> P_trace slice map.
# The map is NOT wire material: prover and verifier both recompute it from
# the claim set, so a proof cannot permute pins between claims.
# ---------------------------------------------------------------------------

def bridged_claim_map(claims):
    """[(claim_index, width, row_off_in_group, n_rows=E*K)] in claim order,
    offsets accumulated per width (groups are width-sorted downstream)."""
    from routed_projected import RoutedProjectedMatmulClaim
    out, off = [], {}
    for ci, c in enumerate(claims):
        if isinstance(c, RoutedProjectedMatmulClaim) and c.use_bridge:
            o = off.get(c.J, 0)
            out.append((ci, c.J, o, c.E * c.K))
            off[c.J] = o + c.E * c.K
    return out


def enroll_tape(tape, mask_seed: bytes, manifest: bytes,
                params: WcParams) -> Enrollment:
    """Enroll the expert weights of every use_bridge claim on the tape:
    per width, claims' experts concatenated in claim order (the same layout
    bridged_claim_map describes)."""
    cmap = bridged_claim_map(tape.claims)
    assert cmap, "no use_bridge claims on this tape"
    groups: Dict[int, list] = {}
    for ci, width, off, ek in cmap:
        c = tape.claims[ci]
        for wv in c.W:
            val = tape.inputs[wv]
            flat = val() if callable(val) else val
            groups.setdefault(width, []).append(
                flat.reshape(c.K, c.J).cuda())
    return build_enrollment(
        {n: torch.cat(rows) for n, rows in groups.items()},
        mask_seed, manifest, params)


class LazyEnrollment:
    """Production enrollment: the weights are NOT resident — the root/levels
    come from one streaming pass (GPU BLAKE3 column accumulator), the masks
    regenerate from their PRG seed, and the eta columns are re-extracted by
    a second streaming pass only after eta exists.  `unit_stream` is a
    zero-arg callable yielding (width, rows_tensor (r, width) uint64 cuda)
    in the SAME canonical order every time (bridged_claim_map order)."""

    def __init__(self, params: WcParams, mask_seed: bytes, manifest: bytes,
                 unit_stream, rows_per_width: Dict[int, int]):
        from core import _make_merkle_acc
        self.params = params
        self.mask_seed = mask_seed
        self.manifest_digest = blake3.blake3(manifest).digest() if len(
            manifest) != 32 else manifest
        self.unit_stream = unit_stream
        self.blocks_per_width = {n: -(-r // params.B)
                                 for n, r in rows_per_width.items()}
        self.rows_per_width = dict(rows_per_width)
        total_polys = sum(b * n for n, b in self.blocks_per_width.items())
        self.total_polys = total_polys
        acc = _make_merkle_acc(params.N_w, total_polys)
        for width, block, coeff_cw in self._stream_codewords():
            with _timed("merkle update"):
                acc.update(coeff_cw)
        with _timed("merkle finalize + tree"):
            inner = acc.finalize().cpu().numpy()
            leaves = [_leaf_from_inner(bytes(inner[i].tolist()))
                      for i in range(params.N_w)]
            self.levels = _tree(leaves)
            self.root = self.levels[-1][0]

    @property
    def groups(self):
        class _G:
            def __init__(self, n_blocks, n_rows):
                self.n_blocks, self.n_rows = n_blocks, n_rows
        return {n: _G(b, self.rows_per_width[n])
                for n, b in self.blocks_per_width.items()}

    def _masks(self, width, block):
        return wc_block_masks(self.mask_seed, width, block, self.params)

    def _timed_stream(self):
        """unit_stream with the shard decode timed (the loader's own work:
        GGUF slice, dequantize, field)."""
        it = self.unit_stream()
        while True:
            with _timed("shard decode"):
                try:
                    item = next(it)
                except StopIteration:
                    return
            yield item

    def _stream_codewords(self):
        """Yield (width, block, codewords (width, N_w)) in width-sorted,
        block order — the canonical poly order.  STREAMING: one unit_stream
        pass PER WIDTH (widths sorted), emitting each B-block immediately —
        resident state is one (B, width) buffer plus one codeword batch.
        (The first version collected every block on the GPU to reorder; at
        48 Maverick layers that is terabytes — the attempt-4 OOM.)"""
        params = self.params
        for width in sorted(self.blocks_per_width):
            buf = torch.zeros(params.B, width, dtype=torch.uint64,
                              device="cuda")
            fill, block = 0, 0

            def emit(rows_b, blk):
                with _timed("masks"):
                    masks = self._masks(width, blk)
                with _timed("pack"):
                    coeffs = torch.zeros(width, params.N_w, dtype=torch.uint64,
                                         device="cuda")
                    coeffs[:, :params.B] = (rows_b.view(torch.int64).T
                                            .contiguous().view(torch.uint64))
                    coeffs[:, params.B:params.K_w] = masks
                with _timed("ntt"):
                    ntt_forward_batched(coeffs)
                return coeffs

            for w, rows in self._timed_stream():
                if w != width:
                    continue
                r, off = rows.size(0), 0
                while r - off > 0:
                    take = min(params.B - fill, r - off)
                    buf[fill:fill + take] = rows[off:off + take]
                    fill += take
                    off += take
                    if fill == params.B:
                        yield width, block, emit(buf, block)
                        block += 1
                        fill = 0
                        buf.zero_()
                del rows
            if fill:
                yield width, block, emit(buf, block)
                block += 1
            assert block == self.blocks_per_width[width], (
                f"width {width}: streamed {block} blocks, expected "
                f"{self.blocks_per_width[width]}")
            del buf

    def pi(self, rho: Dict[int, List[int]]):
        out = {}
        for n, b in self.blocks_per_width.items():
            rho_t = torch.tensor(rho[n], dtype=torch.uint64, device="cuda")
            rows = []
            for a in range(b):
                with _timed("pi masks"):
                    m = self._masks(n, a).view(torch.int64).T.contiguous().view(torch.uint64)
                with _timed("pi matvec"):
                    rows.append(gl_matvec(m, rho_t))
            out[n] = torch.stack(rows)
        return out

    def open_columns(self, eta_idx: List[int]):
        """Second streaming pass: re-extract the eta columns and drift-check
        their inner digests against the committed leaves."""
        from core import _make_merkle_acc
        params = self.params
        idx = torch.tensor(eta_idx, dtype=torch.long, device="cuda")
        opened = {i: [] for i in eta_idx}
        chk = _make_merkle_acc(len(eta_idx), self.total_polys)
        for width, block, cw in self._stream_codewords():
            with _timed("columns gather"):
                cols = cw.view(torch.int64)[:, idx].view(torch.uint64)
                chk.update(cols)
            with _timed("columns to host"):
                cc = cols.cpu()
                for k, i in enumerate(eta_idx):
                    opened[i].append(cc[:, k])
        with _timed("columns drift check"):
            inner = chk.finalize().cpu().numpy()
            for k, i in enumerate(eta_idx):
                leaf = _leaf_from_inner(bytes(inner[k].tolist()))
                assert leaf == self.levels[0][i], (
                    f"enrollment drift at eta column {i}")
        with _timed("columns to python ints"):
            opened_lists = {i: torch.cat(opened[i]).tolist() for i in eta_idx}
        return (opened_lists, {i: _path(self.levels, i) for i in eta_idx})


def lazy_enroll_tape(tape, mask_seed: bytes, manifest: bytes,
                     params: WcParams) -> LazyEnrollment:
    """LazyEnrollment streaming straight from the tape's (external) weight
    inputs in canonical bridged_claim_map order."""
    cmap = bridged_claim_map(tape.claims)
    assert cmap, "no use_bridge claims on this tape"
    rows_per_width: Dict[int, int] = {}
    for ci, width, off, ek in cmap:
        rows_per_width[width] = rows_per_width.get(width, 0) + ek

    def stream():
        for ci, width, off, ek in cmap:
            c = tape.claims[ci]
            for wv in c.W:
                val = tape.inputs[wv]
                flat = val() if callable(val) else val
                yield width, flat.reshape(c.K, c.J).cuda()

    return LazyEnrollment(params, mask_seed, manifest, stream, rows_per_width)
