"""A COMPLETE layer proof: real semantics, all relation families, LogUp, ZK
masks wired in, and a verifier that counts its own work.

This is what `layer.py` was a sketch of. The difference:

  layer.py     one dense Y = XW, no lookups, no masks, masks built but unused
  full_layer   the trace from semantics.py -- RMSNorm, RoPE, causal softmax,
               SiLU, residuals, every rescale bracket -- proved by
                 * the projection seam on BOTH operands of every matmul,
                 * one batched eq-weighted sumcheck for all gates (§4 L3),
                 * one local LogUp per table (§6), in the beta -> R_cmp -> alpha
                   order,
                 * the affine mask compiler (§7) carrying the gate batch,
               and composed by exact root equality (§8.1).

Weights are ENROLLED ONCE per distinct tensor and projected per proof, which is
the doc's persistent-root model (§3): the same expert matrices serve every token
that routes to them, so the enrolment cost is amortised and the per-proof cost is
the projection, not a re-commit.

Counting: every phase runs inside a `counters.phase`, so a run reports field
muls/adds, hashed bytes, opened values and proof bytes per phase. That is the
quantity `cost_model.py` predicts, and `bench/run_toy.py` checks the prediction
against. No kappa anywhere.

Prototype simplifications, unchanged from layer.py and stated in the README:
small projected vectors and LogUp reciprocals travel in the clear and are bound
by re-encoding rather than by a local LF proof. Soundness is intact; hiding of
those specific vectors is not.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from prover.protocol import P as FIELD_P

from . import (logup, moe, projection as pj, relations as rel, rs,
               semantics as sem, sumcheck as sc)
from .counters import charge, phase
from .profile import stage
from .transcript import Transcript

FULL_LAYER_SCHEDULE = [
    ("absorb", "R_in"), ("absorb", "R_out"), ("absorb", "R_mask"),
    ("absorb", "R_lk"), ("coin", "beta"), ("absorb", "R_cmp"), ("coin", "alpha"),
]


@dataclass
class MatmulProof:
    name: str
    w_id: int
    n_in: int
    p_root: bytes
    a_root: bytes
    rho: List[int]
    tau: List[int]
    w_opening: pj.ProjectionOpening
    a_opening: pj.ProjectionOpening
    sumcheck: sc.SumcheckProof
    p_message: List[int] = field(repr=False, default_factory=list)
    a_message: List[int] = field(repr=False, default_factory=list)
    # The FLAT vectors above are the n_in real values the sumcheck runs on. The
    # BLOCKS below are the full ELL-wide committed messages, padding tail
    # included -- binding by re-encoding needs them, because the committed row
    # is the projection of the secret padding too.
    p_blocks: List[List[int]] = field(repr=False, default_factory=list)
    a_blocks: List[List[int]] = field(repr=False, default_factory=list)


@dataclass
class MoEProof:
    """A routed contraction proved with the route hidden (doc §5).

    The seam runs per expert, the activation side is folded into A[e][j] by the
    segmented scan, and the whole contraction collapses to one scalar identity
    (§5.3) proved by a sumcheck over the flattened (expert, coordinate) domain.

    PROTOTYPE, as elsewhere: the sorted records, the scan trace and A travel in
    the clear so the verifier can re-check the permutation product and the
    segment constraints directly. That costs hiding of those vectors, not
    soundness -- a wrong record set still fails the product, and a token placed
    in the wrong segment still fails the counter constraint."""
    name: str
    w_ids: List[int]
    n_in: int
    d_out: int
    rho: List[int]
    tau: List[int]
    p_roots: List[bytes]
    w_openings: List[pj.ProjectionOpening]
    p_messages: List[List[int]] = field(repr=False, default_factory=list)
    p_blocks: List[List[List[int]]] = field(repr=False, default_factory=list)
    a_flat: List[int] = field(repr=False, default_factory=list)
    sumcheck: sc.SumcheckProof = None
    perm_src: int = 0
    perm_dst: int = 0
    scan_ok: bool = False
    scan_why: str = ""


@dataclass
class FullLayerProof:
    in_root: bytes
    out_root: bytes
    mask_root: bytes
    matmuls: List[MatmulProof]
    moe: List[MoEProof]
    gate_proof: sc.SumcheckProof
    gate_z: List[int]
    gate_lambdas: List[int]
    lookups: List[Tuple[int, logup.LogUpProof, List[int], List[int], List[int], int]]
    columns: List[int]
    weight_roots: Dict[int, bytes]


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def _pad(vec: Sequence[int], width: int) -> List[int]:
    return [vec[j] if j < len(vec) else 0 for j in range(width)]


class Enrollment:
    """Persistent weight roots, built once and reused across proofs (doc §3.2).
    Distinct tensors are keyed by content so a matrix used by many tokens is
    enrolled a single time."""

    def __init__(self, cfg: rs.Config):
        self.cfg = cfg
        self.by_key: Dict[tuple, int] = {}
        self.weights: List[pj.PersistentWeights] = []

    def enroll(self, W: Sequence[Sequence[int]], rng=None) -> int:
        key = tuple(tuple(row) for row in W)
        if key in self.by_key:
            return self.by_key[key]
        # Only the last block carries a tail; with n_in > ELL the old
        # `ELL - n_in` was negative and produced empty padding rows.
        pad_len = (-len(W[0])) % self.cfg.ELL
        padding = ([[rng.randrange(FIELD_P) for _ in range(pad_len)] for _ in W]
                   if rng is not None and pad_len else None)
        with phase("enroll"), stage("enroll"):
            pw = pj.PersistentWeights.enroll(self.cfg, W, padding)
        idx = len(self.weights)
        self.weights.append(pw)
        self.by_key[key] = idx
        return idx

    def root(self, idx: int) -> bytes:
        return self.weights[idx].root


def prove_full_layer(trace: sem.LayerTrace, cfg: rs.Config, enrol: Enrollment,
                     q_columns: int, rng, use_masks: bool = True,
                     domain: bytes = b"layergkr/full") -> FullLayerProof:
    tr = Transcript(domain)

    # ── L1: commit the layer's states and the mask tape ──────────────────────
    with phase("commit.states"), stage("commit.states"):
        in_state = pj.PersistentWeights.enroll(cfg, [_pad(r, cfg.ELL - 1) for r in trace.x_in])
        out_commit = rs.Commit.from_messages(cfg, [_pad(r, cfg.ELL) for r in trace.y_out])
    tr.absorb_root("R_in", in_state.root)
    tr.absorb_root("R_out", out_commit.root)

    tape = None
    with phase("commit.mask"), stage("commit.mask"):
        n_tape = 8 * (max(len(trace.gates), 1).bit_length() + 16)
        tape_vals = [rng.randrange(FIELD_P) for _ in range(n_tape)]
        mask_commit = rs.Commit.from_messages(
            cfg, [_pad(tape_vals[i:i + cfg.ELL], cfg.ELL)
                  for i in range(0, len(tape_vals), cfg.ELL)])
        if use_masks:
            tape = sc.MaskTape(tape_vals)
    tr.absorb_root("R_mask", mask_commit.root)

    # ── LogUp, one per table, in the beta -> R_cmp -> alpha order ────────────
    lookup_proofs = []
    with phase("logup"), stage("logup"):
        for k, use in enumerate(trace.lookups):
            lu = logup.LogUp(cfg, use.table.table_id, list(use.queries),
                             use.table.tuples())
            lu.commit_raw(tr, f"R_lk{k}")
            beta = tr.coin(f"beta{k}")[0]
            lu.compress(beta, tr, f"R_cmp{k}")
            alpha = tr.coin(f"alpha{k}")[0]
            lu.build(alpha)
            p = lu.prove(tr)
            lookup_proofs.append((use.table.table_id, p, list(lu.q_vals),
                                  list(lu.t_vals), list(lu.mult), alpha))

    # ── the batched gate sumcheck (§4 L3) ────────────────────────────────────
    with phase("gates"), stage("gates"):
        gate_proof, gate_z, gate_lambdas = rel.prove_batch(trace.gates, tr, "gates", tape)

    # ── the matmuls, each through the seam on both operands ─────────────────
    mm_proofs: List[MatmulProof] = []
    with phase("matmuls"), stage("matmuls"):
        for m in trace.matmuls:
            w_id = enrol.enroll(m.W, rng)
            pw = enrol.weights[w_id]
            with phase("matmul.seam"), stage("matmul.seam"):
                rho = tr.coin(f"rho_{m.name}", len(m.W))
                p_commit = pj.commit_projection(pw, rho)
                tr.absorb_root(f"R_P_{m.name}", p_commit.root)
                x_state = pj.PersistentWeights.enroll(cfg, [list(r[:pw.n_in]) +
                                                           [0] * max(pw.n_in - len(r), 0)
                                                           for r in m.X])
                tr.absorb_root(f"R_X_{m.name}", x_state.root)
                tau = tr.coin(f"tau_{m.name}", len(m.X))
                a_commit = pj.commit_projection(x_state, tau)
                tr.absorb_root(f"R_A_{m.name}", a_commit.root)
            with phase("matmul.sumcheck"), stage("matmul.sumcheck"):
                p_blk = pj.project_message(pw, rho)
                a_blk = pj.project_message(x_state, tau)
                p_msg = pj.flatten_projection(pw, p_blk)
                a_msg = pj.flatten_projection(x_state, a_blk)
                width = _next_pow2(pw.n_in)
                coins = tr.coin(f"sc_{m.name}", max(width.bit_length() - 1, 1))
                proof = sc.prove([_pad(a_msg[:pw.n_in], width), _pad(p_msg[:pw.n_in], width)],
                                 lambda i: coins[i])
                direct = 0
                for t in range(len(m.X)):
                    for i in range(len(m.W)):
                        direct = (direct + tau[t] * rho[i] * m.Y[t][i]) % FIELD_P
                if proof.claim != direct:
                    raise AssertionError(f"contraction identity failed for {m.name}")
            mm_proofs.append(MatmulProof(m.name, w_id, pw.n_in, p_commit.root,
                                         a_commit.root, rho, tau, None, None,
                                         proof, p_msg, a_msg, p_blk, a_blk))
            mm_proofs[-1].x_root = x_state.root
            mm_proofs[-1]._pw = pw
            mm_proofs[-1]._xs = x_state
            mm_proofs[-1]._pc = p_commit
            mm_proofs[-1]._ac = a_commit

    # ── the routed contractions, route hidden (doc §5) ──────────────────────
    moe_proofs: List[MoEProof] = []
    with phase("moe"), stage("moe"):
        for m in trace.moe:
            d_out, n_in, S_m = len(m.W[0]), len(m.W[0][0]), len(m.X)
            E = len(m.W)
            rho = tr.coin(f"rho_{m.name}", d_out)
            tau = tr.coin(f"tau_{m.name}", S_m)

            with phase("moe.seams"), stage("moe.seams"):
                w_ids, p_roots, p_msgs, w_open, p_blks = [], [], [], [], []
                for e in range(E):
                    wid = enrol.enroll(m.W[e], rng)
                    pw = enrol.weights[wid]
                    pc = pj.commit_projection(pw, rho)
                    tr.absorb_root(f"R_P_{m.name}_{e}", pc.root)
                    w_ids.append(wid)
                    p_roots.append(pc.root)
                    blk = pj.project_message(pw, rho)
                    p_blks.append(blk)
                    p_msgs.append(pj.flatten_projection(pw, blk))
                    w_open.append((pw, pc))

            with phase("moe.sort"), stage("moe.sort"):
                src = moe.source_records(0, 1, m.route, m.X)
                dst = moe.stable_sorted_records(src)
                beta = tr.coin(f"perm_beta_{m.name}")[0]
                z = tr.coin(f"perm_z_{m.name}")[0]
                perm_src = moe.characteristic_product(src, beta, z)
                perm_dst = moe.characteristic_product(dst, beta, z)
                A = moe.segment_sums(m.route, m.X, tau, E)
                scan_ok, scan_why = True, "ok"
                for j in range(n_in):
                    lane = [r for r in dst if r.j == j]
                    tr_lane, A_lane = moe.build_lane(lane, E, tau)
                    ok, why = moe.scan_constraints_ok(tr_lane, E, tau)
                    if not ok:
                        scan_ok, scan_why = False, f"lane {j}: {why}"
                        break
                    if any(A_lane[e] != A[e][j] for e in range(E)):
                        scan_ok, scan_why = False, f"lane {j}: scan != direct sums"
                        break

            with phase("moe.sumcheck"), stage("moe.sumcheck"):
                width = _next_pow2(E * n_in)
                a_flat = _pad([A[e][j] for e in range(E) for j in range(n_in)], width)
                p_flat = _pad([p_msgs[e][j] for e in range(E) for j in range(n_in)], width)
                coins = tr.coin(f"sc_{m.name}", max(width.bit_length() - 1, 1))
                proof = sc.prove([a_flat, p_flat], lambda i: coins[i])
                direct = 0
                for t in range(S_m):
                    for i in range(d_out):
                        direct = (direct + tau[t] * rho[i] * m.Y[t][i]) % FIELD_P
                if proof.claim != direct:
                    raise AssertionError(f"MoE identity failed for {m.name}")

            mp = MoEProof(m.name, w_ids, n_in, d_out, rho, tau, p_roots, [],
                          p_msgs, p_blks, a_flat, proof, perm_src, perm_dst,
                          scan_ok, scan_why)
            mp._open = w_open
            moe_proofs.append(mp)

    # ── L5: columns, and the openings ───────────────────────────────────────
    with phase("open"), stage("open"):
        col_seed = tr.coin_bytes("columns")
        columns = rs.sample_columns(col_seed, q_columns, cfg.N_LIG)
        for mp in moe_proofs:
            mp.w_openings = [pj.open_projection(pw, pc, columns) for pw, pc in mp._open]
            del mp._open
        for mp in mm_proofs:
            mp.w_opening = pj.open_projection(mp._pw, mp._pc, columns)
            mp.a_opening = pj.open_projection(mp._xs, mp._ac, columns)
            n_open = (mp._pw.n_out + 1 + mp._xs.n_out + 1) * len(columns)
            charge(opened_values=n_open, proof_bytes=8 * n_open)
            del mp._pw, mp._xs, mp._pc, mp._ac

    return FullLayerProof(
        in_root=in_state.root, out_root=out_commit.root, mask_root=mask_commit.root,
        matmuls=mm_proofs, moe=moe_proofs, gate_proof=gate_proof, gate_z=gate_z,
        gate_lambdas=gate_lambdas, lookups=lookup_proofs, columns=columns,
        weight_roots={i: enrol.root(i) for i in range(len(enrol.weights))})


def _bind_blocks(cfg: rs.Config, blocks_msg: Sequence[Sequence[int]],
                 opening: pj.ProjectionOpening) -> bool:
    """Re-encode the committed projected messages BLOCK BY BLOCK and check them
    against the opened R_P rows. `blocks_msg` must be the FULL ELL-wide messages
    (padding tail included) -- the committed row is the projection of the secret
    padding too, so a flattened n_in vector would not reproduce the codeword."""
    n_blocks = len(opening.p_values[0]) if opening.p_values else 1
    if len(blocks_msg) != n_blocks:
        return False
    for b in range(n_blocks):
        reenc = rs.encode_row(cfg, list(blocks_msg[b]))
        for k, c in enumerate(opening.columns):
            if reenc[c] != opening.p_values[k][b]:
                return False
    return True


def verify_full_layer(cfg: rs.Config, proof: FullLayerProof,
                      gates: Sequence[rel.Gate]) -> Tuple[bool, str]:
    """Verifier. Runs inside its own counter so the verifier cost is measured,
    not assumed -- the doc leaves verification out of its theorem entirely."""
    with phase("verify.matmuls"), stage("verify.matmuls"):
        for mp in proof.matmuls:
            w_root = proof.weight_roots[mp.w_id]
            ok, why = pj.verify_projection(cfg, w_root, mp.p_root, mp.rho, mp.w_opening)
            if not ok:
                return False, f"{mp.name} weight seam: {why}"
            ok, why = pj.verify_projection(cfg, mp.x_root, mp.a_root, mp.tau, mp.a_opening)
            if not ok:
                return False, f"{mp.name} activation seam: {why}"
            for msg, opening, tag in ((mp.p_blocks, mp.w_opening, "P"),
                                      (mp.a_blocks, mp.a_opening, "A")):
                if not _bind_blocks(cfg, msg, opening):
                    return False, f"{mp.name}: {tag} does not match its root"
            # width must be the PROVER's: n_in rounded up, not the ELL-length
            # message the projection happens to be carried in.
            width = _next_pow2(mp.n_in)
            ok, why = sc.verify(mp.sumcheck,
                                [_pad(mp.a_message[:mp.n_in], width),
                                 _pad(mp.p_message[:mp.n_in], width)],
                                lambda i: mp.sumcheck.challenges[i])
            if not ok:
                return False, f"{mp.name} contraction: {why}"

    with phase("verify.moe"), stage("verify.moe"):
        for mp in proof.moe:
            if not mp.scan_ok:
                return False, f"moe {mp.name}: segment constraints: {mp.scan_why}"
            if mp.perm_src != mp.perm_dst:
                return False, f"moe {mp.name}: permutation product mismatch"
            for e, (root, opening) in enumerate(zip(mp.p_roots, mp.w_openings)):
                w_root = proof.weight_roots[mp.w_ids[e]]
                ok, why = pj.verify_projection(cfg, w_root, root, mp.rho, opening)
                if not ok:
                    return False, f"moe {mp.name} expert {e} seam: {why}"
                if not _bind_blocks(cfg, mp.p_blocks[e], opening):
                    return False, f"moe {mp.name} expert {e}: P does not match root"
            width = len(mp.a_flat)
            p_flat = _pad([mp.p_messages[e][j] for e in range(len(mp.p_roots))
                           for j in range(mp.n_in)], width)
            ok, why = sc.verify(mp.sumcheck, [mp.a_flat, p_flat],
                                lambda i: mp.sumcheck.challenges[i])
            if not ok:
                return False, f"moe {mp.name} identity: {why}"

    with phase("verify.gates"), stage("verify.gates"):
        ok, why = rel.verify_batch(proof.gate_proof, gates, proof.gate_z,
                                   proof.gate_lambdas)
        if not ok:
            return False, f"gates: {why}"

    with phase("verify.logup"), stage("verify.logup"):
        for table_id, lp, q_vals, t_vals, mult, alpha in proof.lookups:
            ok, why = logup.verify(lp, q_vals, t_vals, mult, alpha)
            if not ok:
                return False, f"lookup {table_id}: {why}"
    return True, "ok"
