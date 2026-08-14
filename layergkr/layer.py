"""One layer proof, and the composition lemma of doc §8.1.

A toy layer is Y = X W (output-major weights). Its proof strings together the
three pieces built in this package:

  1. commit the output hidden state -> R_out, which becomes the NEXT layer's
     R_in byte-for-byte. That equality IS the composition link: no separate
     unproven copy claim and no IVC splice (doc §8.1, §11 item 5).
  2. the projection seam (projection.py), used TWICE -- once on each operand:
       F_P = sum_i chi_rho(i) F_{W_i}   binds the weight side to R_W,
       F_A = sum_t tau_t  F_{X_t}       binds the activation side to R_in.
     Both are opened in the same columns as their persistent root, so the same
     (K/N)^q argument covers both.
  3. a sumcheck (sumcheck.py) for the contraction the seam collapsed:

        sum_{t,i} tau_t rho_i Y[t,i] == sum_j A[j] P[j],
        A[j] = sum_t tau_t X[t,j],   P[j] = sum_i rho_i W[j,i]

     the dense-matmul analogue of the MoE identity in moe.py. The S*d*d gate
     trace never appears; both sides of the sumcheck are length n_in.

PROTOTYPE SIMPLIFICATION, stated plainly: the small vectors A and P are sent in
the clear and bound to their roots by re-encoding. That binds them correctly but
REVEALS them, so this path is not zero-knowledge. The scheme replaces it with a
local LF proof over the small R_A / R_P (doc §4 L5) -- ordinary small-instance
Ligero, not implemented here. `sumcheck.MaskTape` is the mechanism that would
carry the terminal claims instead; it is built and gated in test_sumcheck.py but
is not wired into this driver.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from prover.protocol import P as FIELD_P

from . import projection as pj, rs, sumcheck as sc
from .transcript import Schedule, Transcript

# Ordering for a dense layer: both operand projections land before the sumcheck
# coins, and every polynomial is fixed before the column coin.
DENSE_LAYER_SCHEDULE = [
    ("absorb", "R_in"), ("absorb", "R_out"),
    ("coin", "rho"), ("absorb", "R_P"),
    ("coin", "tau"), ("absorb", "R_A"),
    ("coin", "sumcheck"),
    ("absorb", "R_terminal"),
    ("coin", "columns"),
]


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def _pad(vec: Sequence[int], width: int) -> List[int]:
    return [vec[j] if j < len(vec) else 0 for j in range(width)]


@dataclass
class LayerProof:
    in_root: bytes
    out_root: bytes
    p_root: bytes
    a_root: bytes
    rho: List[int]
    tau: List[int]
    w_opening: pj.ProjectionOpening
    a_opening: pj.ProjectionOpening
    sumcheck: sc.SumcheckProof
    columns: List[int]
    p_message: List[int] = field(repr=False, default_factory=list)
    a_message: List[int] = field(repr=False, default_factory=list)
    p_blocks: List[List[int]] = field(repr=False, default_factory=list)
    a_blocks: List[List[int]] = field(repr=False, default_factory=list)


@dataclass
class Layer:
    """Weights plus the input hidden state of one layer."""
    cfg: rs.Config
    weights: pj.PersistentWeights
    X: List[List[int]]                       # [S][n_in]
    Y: List[List[int]] = field(repr=False, default_factory=list)
    in_state: Optional[pj.PersistentWeights] = None      # R_in, as a projectable commit
    out_commit: Optional[rs.Commit] = None

    def compute(self) -> None:
        n_in, n_out = self.weights.n_in, self.weights.n_out
        W = self.weights.messages
        self.Y = [[sum(self.X[t][j] * W[i][j] for j in range(n_in)) % FIELD_P
                   for i in range(n_out)] for t in range(len(self.X))]
        # the input state is committed row-per-token, so tau-projection is the
        # same seam as the weight side
        self.in_state = pj.PersistentWeights.enroll(self.cfg, self.X)
        self.out_commit = rs.Commit.from_messages(self.cfg, self.Y)


def prove_layer(layer: Layer, transcript: Transcript, q_columns: int) -> LayerProof:
    cfg = layer.cfg
    if layer.out_commit is None:
        layer.compute()
    S, n_in, n_out = len(layer.X), layer.weights.n_in, layer.weights.n_out

    transcript.absorb_root("R_in", layer.in_state.root)
    transcript.absorb_root("R_out", layer.out_commit.root)

    rho = transcript.coin("rho", n_out)
    p_commit = pj.commit_projection(layer.weights, rho)
    transcript.absorb_root("R_P", p_commit.root)

    tau = transcript.coin("tau", S)
    a_commit = pj.commit_projection(layer.in_state, tau)
    transcript.absorb_root("R_A", a_commit.root)

    p_blk = pj.project_message(layer.weights, rho)
    a_blk = pj.project_message(layer.in_state, tau)
    p_msg = pj.flatten_projection(layer.weights, p_blk)
    a_msg = pj.flatten_projection(layer.in_state, a_blk)

    width = _next_pow2(n_in)
    coins = transcript.coin("sumcheck", max(width.bit_length() - 1, 1))
    proof = sc.prove([_pad(a_msg[:n_in], width), _pad(p_msg[:n_in], width)],
                     lambda i: coins[i])

    # the prover's own consistency check: the sumcheck claim must equal the
    # tau/rho-weighted output. If this ever fails the semantics are wrong, not
    # the proof system.
    direct = 0
    for t in range(S):
        for i in range(n_out):
            direct = (direct + tau[t] * rho[i] * layer.Y[t][i]) % FIELD_P
    if proof.claim != direct:
        raise AssertionError("contraction identity failed on the prover side")

    transcript.absorb_root("R_terminal", b"\x00" * 32)
    col_seed = transcript.coin_bytes("columns")
    columns = rs.sample_columns(col_seed, q_columns, cfg.N_LIG)

    return LayerProof(
        in_root=layer.in_state.root, out_root=layer.out_commit.root,
        p_root=p_commit.root, a_root=a_commit.root, rho=rho, tau=tau,
        w_opening=pj.open_projection(layer.weights, p_commit, columns),
        a_opening=pj.open_projection(layer.in_state, a_commit, columns),
        sumcheck=proof, columns=columns, p_message=p_msg, a_message=a_msg,
        p_blocks=p_blk, a_blocks=a_blk)


def _bind_small_vector(cfg: rs.Config, blocks_msg: Sequence[Sequence[int]],
                       opening: pj.ProjectionOpening, label: str) -> Tuple[bool, str]:
    """Re-encode the committed projected messages block by block and check them
    against the opened projection-root columns. Prototype stand-in for the local
    LF proof. Takes the FULL ELL-wide blocks: the committed row includes the
    projection of the secret padding, which a flattened vector would drop."""
    n_blocks = len(opening.p_values[0]) if opening.p_values else 1
    if len(blocks_msg) != n_blocks:
        return False, f"{label}: {len(blocks_msg)} blocks for {n_blocks} committed rows"
    for b in range(n_blocks):
        reenc = rs.encode_row(cfg, list(blocks_msg[b]))
        for k, c in enumerate(opening.columns):
            if reenc[c] != opening.p_values[k][b]:
                return False, f"{label} message does not match its root at column {c}"
    return True, "ok"


def verify_layer(cfg: rs.Config, w_root: bytes, proof: LayerProof,
                 n_in: int) -> Tuple[bool, str]:
    """Verifier for one layer against the enrolled weight root and the input root
    the proof claims."""
    ok, why = pj.verify_projection(cfg, w_root, proof.p_root, proof.rho, proof.w_opening)
    if not ok:
        return False, f"weight seam: {why}"
    ok, why = pj.verify_projection(cfg, proof.in_root, proof.a_root,
                                   proof.tau, proof.a_opening)
    if not ok:
        return False, f"activation seam: {why}"
    ok, why = _bind_small_vector(cfg, proof.p_blocks, proof.w_opening, "P")
    if not ok:
        return False, why
    ok, why = _bind_small_vector(cfg, proof.a_blocks, proof.a_opening, "A")
    if not ok:
        return False, why

    width = _next_pow2(n_in)
    ok, why = sc.verify(proof.sumcheck,
                        [_pad(proof.a_message[:n_in], width),
                         _pad(proof.p_message[:n_in], width)],
                        lambda i: proof.sumcheck.challenges[i])
    if not ok:
        return False, f"sumcheck: {why}"
    return True, "ok"


def verify_chain(cfg: rs.Config, w_roots: Sequence[bytes],
                 proofs: Sequence[LayerProof], n_ins: Sequence[int]
                 ) -> Tuple[bool, str]:
    """Composition (§8.1): every layer verifies, and R_in,l+1 == R_out,l exactly.
    Root equality is the only link -- there is no separate copy claim to forge."""
    for l, proof in enumerate(proofs):
        ok, why = verify_layer(cfg, w_roots[l], proof, n_ins[l])
        if not ok:
            return False, f"layer {l}: {why}"
        if l > 0 and proof.in_root != proofs[l - 1].out_root:
            return False, (f"layer {l}: input root != layer {l - 1} output root "
                           f"(splice)")
    return True, "ok"


def prove_chain(layers: Sequence[Layer], q_columns: int,
                domain: bytes = b"layergkr/chain") -> List[LayerProof]:
    """Prove a stack of layers, feeding each output state into the next input.
    Each layer gets its own schedule-checked transcript, which is what makes the
    challenges LAYER-LOCAL -- the property that lets a layer be freed once it is
    proven, and the reason one semantic sweep suffices."""
    proofs = []
    for l, layer in enumerate(layers):
        t = Transcript(domain + b"/%d" % l, Schedule(DENSE_LAYER_SCHEDULE))
        proofs.append(prove_layer(layer, t, q_columns))
    return proofs
