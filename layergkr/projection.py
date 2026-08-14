"""The project-before-sumcheck seam (doc §4 stages L3-L5) — the load-bearing
piece of Layer-GKR-LF, and the one that replaces the full-weight `q_lin`.

Setting. A weight tensor is enrolled OUTPUT-MAJOR: one RS row per output
coordinate i, whose message is that output's contracted-dimension weights
W[.,i] followed by secret enrollment padding. All rows share the same message
and padding positions (a manifest requirement), so the rows can be combined
column-wise.

Seam. Once the matmul's output point rho is fixed, the prover forms

    P[j]   = sum_i chi_rho(i) * W[j,i]          (small: one field per input coord)
    F_P    = sum_i chi_rho(i) * F_{W_i}         (a full codeword, committed as R_P)

and the contraction sumcheck then runs against the SMALL P instead of the 400B W.
Because RS is linear, F_P is exactly the codeword of P together with the same
combination of the secret padding halves -- `rs.linear_combination` computes it
and `tests/test_projection.py` checks the identity rather than assuming it.

Why it binds. The verifier opens the SAME q columns of the persistent F_W and of
the fresh R_P and checks

    F_P[c] == sum_i chi_rho(i) * F_{W_i}[c]     for every opened column c.

If a cheating prover commits some other codeword G != F_P, then G - F_P is a
nonzero RS codeword of degree < K, hence nonzero in at least N-K of the N
positions; it survives q independent columns with probability at most (K/N)^q.
At the doc's K/N = 1/4 and q = 54 that is 4^-54. `count_column_disagreements`
measures the ACTUAL number of disagreeing positions for a forged commitment, so
the test asserts the >= N-K guarantee empirically instead of quoting it.

Causality. R_P is absorbed into the transcript before the contraction coin and
before the column coin (`transcript.LAYER_SCHEDULE`). That is what closes the
counterexample the doc raises in §4: a prover who picks the projection after
seeing the challenge produces a transcript whose coins the verifier recomputes
differently.

NOT implemented here: the local LF proof that ties R_P's opened values to the
sumcheck's terminal claim about P. That is ordinary small-instance Ligero, which
the existing prover already does; this module covers the part that is new.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from prover.protocol import P as FIELD_P, mul

from . import rs
from .counters import charge
from .transcript import Transcript


@dataclass
class PersistentWeights:
    """R_W: the enrolled, reusable commitment to one weight tensor.

    MULTI-ROW LAYOUT. A contraction wider than ELL does not fit one RS row, so
    each output coordinate spans `n_blocks = ceil(n_in / ELL)` rows. This is not
    a prototype convenience -- it is required at production geometry (Maverick's
    FFN contracts over 16384 against ELL = 8192) and it is exactly the
    `ceil(n_in / ELL)` factor in the theorem's N_pad formula (§9.2).

    Rows are laid out BLOCK-MAJOR: row index = b * n_out + i. So at any opened
    column the values of one block are contiguous, and the seam's linear
    combination is applied per block. The manifest's alignment requirement (all
    output coordinates of a block share message and padding positions) is what
    makes that legal, and it is why the (K/N)^q argument carries over unchanged:
    it simply holds for each block."""
    cfg: rs.Config
    n_in: int
    n_out: int
    commit: rs.Commit
    messages: List[List[int]] = field(repr=False, default_factory=list)
    n_blocks: int = 1

    @classmethod
    def enroll(cls, cfg: rs.Config, W: Sequence[Sequence[int]],
               padding: Optional[Sequence[Sequence[int]]] = None) -> "PersistentWeights":
        """W[i][j] = weight of input j for output i (output-major rows).
        `padding` supplies the secret enrollment padding for the LAST block of
        each output coordinate -- only that block has a tail, of length
        `(-n_in) % ELL`. Anything longer is truncated; shorter is an error.
        Without padding the commitment is NOT hiding."""
        # Tensor fast path: weights that are already on the device are laid out
        # into block-major messages there, so enrolment never round-trips through
        # Python. Measured at production geometry: 1.614 ms/row via lists against
        # 0.005 ms/row from device memory -- 323x, and weights are the bulk of the
        # rows in a layer.
        if hasattr(W, "shape"):
            return cls._enroll_tensor(cfg, W, padding)
        n_out, n_in = len(W), len(W[0])
        blocks = -(-n_in // cfg.ELL)
        msgs = []
        for b in range(blocks):
            lo = b * cfg.ELL
            hi = min(lo + cfg.ELL, n_in)
            pad_len = cfg.ELL - (hi - lo)     # only the LAST block has a tail
            for i in range(n_out):
                if pad_len and padding is not None:
                    src = list(padding[i])
                    if len(src) < pad_len:
                        raise ValueError(
                            f"padding for output {i} is {len(src)} long but the "
                            f"block tail needs {pad_len}")
                    pad = src[:pad_len]
                else:
                    pad = [0] * pad_len
                msgs.append(list(W[i][lo:hi]) + pad)
        return cls(cfg, n_in, n_out, rs.Commit.from_messages(cfg, msgs), msgs,
                   blocks)

    @classmethod
    def _enroll_tensor(cls, cfg: rs.Config, W, padding=None) -> "PersistentWeights":
        """W is an (n_out, n_in) device tensor. Build the block-major message
        matrix on the device and commit it without leaving the card."""
        import torch
        n_out, n_in = int(W.shape[0]), int(W.shape[1])
        blocks = -(-n_in // cfg.ELL)
        msgs = torch.zeros((blocks * n_out, cfg.ELL), dtype=torch.uint64,
                           device=W.device)
        for b in range(blocks):
            lo = b * cfg.ELL
            hi = min(lo + cfg.ELL, n_in)
            msgs[b * n_out:(b + 1) * n_out, :hi - lo] = W[:, lo:hi]
            if padding is not None and hi - lo < cfg.ELL:
                pad = padding if hasattr(padding, "shape") else torch.tensor(
                    [list(r)[:cfg.ELL - (hi - lo)] for r in padding],
                    dtype=torch.uint64, device=W.device)
                msgs[b * n_out:(b + 1) * n_out, hi - lo:] = pad
        return cls(cfg, n_in, n_out, rs.Commit.from_messages(cfg, msgs), msgs, blocks)

    def block_slice(self, b: int) -> Tuple[int, int]:
        lo = b * self.cfg.ELL
        return lo, min(lo + self.cfg.ELL, self.n_in)

    @property
    def root(self) -> bytes:
        return self.commit.root


@dataclass
class ProjectionOpening:
    """What the prover sends for the seam at the opened columns."""
    columns: List[int]
    w_values: List[List[int]]                    # per column: one value per output row
    w_paths: List[rs.Path]
    p_values: List[List[int]]                    # per column: one value per BLOCK
    p_paths: List[rs.Path]


def project_message(pw: PersistentWeights, chi: Sequence[int]) -> List[List[int]]:
    """P^(b)[j] = sum_i chi[i] * W^(b)[i][j], one ELL-wide message per BLOCK.
    The tails are the combined secret padding, which the seam opens exactly like
    any other slot (doc §9.2 counts it -- it is not free)."""
    if len(chi) != pw.n_out:
        raise ValueError("chi length must equal n_out")
    charge(mul_defer=pw.n_out * pw.cfg.ELL * pw.n_blocks,
           add=pw.n_out * pw.cfg.ELL * pw.n_blocks,
           comb_iter=pw.n_out * pw.cfg.ELL * pw.n_blocks)
    if hasattr(pw.messages, "shape"):
        import torch
        from prover import cuda_primitives as cp
        chi_t = torch.tensor([list(chi)], dtype=torch.uint64, device=pw.messages.device)
        out_t = []
        for b in range(pw.n_blocks):
            blk = pw.messages[b * pw.n_out:(b + 1) * pw.n_out].contiguous()
            out_t.append([int(v) for v in cp.gl_matmul(chi_t, blk)[0].cpu().tolist()])
        return out_t
    out = []
    for b in range(pw.n_blocks):
        acc = [0] * pw.cfg.ELL
        base = b * pw.n_out
        for i, a in enumerate(chi):
            if a == 0:
                continue
            row = pw.messages[base + i]
            for j in range(pw.cfg.ELL):
                acc[j] = (acc[j] + row[j] * a) % FIELD_P
        out.append(acc)
    return out


def flatten_projection(pw: PersistentWeights, blocks: Sequence[Sequence[int]]
                       ) -> List[int]:
    """The n_in real projected values, with each block's padding tail dropped."""
    flat: List[int] = []
    for b, msg in enumerate(blocks):
        lo, hi = pw.block_slice(b)
        flat.extend(msg[: hi - lo])
    return flat


def project_codeword(pw: PersistentWeights, chi: Sequence[int]):
    """Single-block F_P = sum_i chi[i] * F_{W_i}, computed on the CODEWORDS.
    For a multi-block tensor use `commit_projection`, which does it per block. Equal to
    encode(project_message(...)) by linearity -- the tests check that.

    Returns a device tensor when the weights are committed on the GPU, so the
    projection never leaves the card."""
    if pw.n_blocks != 1:
        raise ValueError("multi-block tensor: use commit_projection")
    return pw.commit.combine(chi)


def commit_projection(pw: PersistentWeights, chi: Sequence[int]) -> rs.Commit:
    """R_P: one RS row per block of the projected codeword."""
    if pw.n_blocks == 1:
        cw = project_codeword(pw, chi)
        return rs.Commit(pw.cfg, mat=cw) if pw.commit.on_gpu else rs.Commit(pw.cfg, [cw])
    # Blocks are contiguous rows, so the per-block combination is the same
    # coefficient vector applied to each block's row range.
    chi_full = [0] * (pw.n_out * pw.n_blocks)
    rows = []
    for b in range(pw.n_blocks):
        v = list(chi_full)
        v[b * pw.n_out:(b + 1) * pw.n_out] = list(chi)
        rows.append(pw.commit.combine(v))
    if pw.commit.on_gpu:
        import torch
        return rs.Commit(pw.cfg, mat=torch.cat(rows, dim=0).contiguous())
    return rs.Commit(pw.cfg, rows)


def open_projection(pw: PersistentWeights, p_commit: rs.Commit,
                    columns: Sequence[int]) -> ProjectionOpening:
    wv, wp, pv, pp = [], [], [], []
    for c in columns:
        vals, path = pw.commit.open(c)
        wv.append(vals)
        wp.append(path)
        vals_p, path_p = p_commit.open(c)
        pv.append(list(vals_p))
        pp.append(path_p)
    return ProjectionOpening(list(columns), wv, wp, pv, pp)


def verify_projection(cfg: rs.Config, w_root: bytes, p_root: bytes,
                      chi: Sequence[int], opening: ProjectionOpening) -> Tuple[bool, str]:
    """Verifier side. Returns (ok, reason). Checks, per opened column:
       1. the W column opens against the persistent root,
       2. the P column opens against the fresh root,
       3. P's value equals the chi-combination of W's column values."""
    if not (len(opening.columns) == len(opening.w_values) == len(opening.p_values)):
        return False, "malformed opening"
    q = len(opening.columns)
    n_rows = len(chi)
    charge(mul=q * n_rows, add=q * n_rows)
    depth = max(cfg.N_LIG.bit_length() - 1, 1)
    charge(hash_bytes=q * (8 * n_rows + 8) + 2 * q * depth * 64,
           hash_calls=2 * q * (1 + depth))
    n_out = len(chi)
    for k, c in enumerate(opening.columns):
        wvals = opening.w_values[k]
        if len(wvals) % n_out:
            return False, f"column {c}: {len(wvals)} rows is not a multiple of {n_out}"
        blocks = len(wvals) // n_out
        pvals = opening.p_values[k]
        if len(pvals) != blocks:
            return False, f"column {c}: {len(pvals)} P rows for {blocks} blocks"
        if not rs.merkle_verify(rs.merkle_leaf(wvals), opening.w_paths[k], w_root):
            return False, f"column {c}: W merkle failed"
        if not rs.merkle_verify(rs.merkle_leaf(pvals), opening.p_paths[k], p_root):
            return False, f"column {c}: P merkle failed"
        for b in range(blocks):
            acc = 0
            for a, v in zip(chi, wvals[b * n_out:(b + 1) * n_out]):
                acc = (acc + mul(a, v)) % FIELD_P
            if acc != pvals[b]:
                return False, f"column {c} block {b}: projection equality failed"
    return True, "ok"


def count_column_disagreements(pw: PersistentWeights, chi: Sequence[int],
                               forged: rs.Codeword) -> int:
    """How many of the N codeword positions a forged R_P disagrees with the true
    projection in. The soundness argument says a forged CODEWORD must disagree in
    at least N - K positions; this measures it instead of asserting it."""
    truth = project_codeword(pw, chi)
    if hasattr(truth, "cpu"):
        truth = [int(v) for v in truth[0].cpu().tolist()]
    return sum(1 for a, b in zip(truth, forged) if a != b)


# ── the seam as it sits in a layer proof (ordering enforced) ──────────────────
def run_seam(cfg: rs.Config, pw: PersistentWeights, transcript: Transcript,
             q_columns: int) -> Tuple[List[int], rs.Commit, ProjectionOpening, List[int]]:
    """Honest prover flow for one matmul's weight seam, in the schedule order:
       coin rho -> build & absorb R_P -> coin contraction -> ... -> coin columns.
    Returns (chi, p_commit, opening, columns)."""
    rho = transcript.coin("rho", pw.n_out)
    p_commit = commit_projection(pw, rho)
    transcript.absorb_root("R_P", p_commit.root)
    transcript.coin("contraction", 4)          # stands in for the sumcheck challenges
    transcript.absorb_root("R_terminal", b"\x00" * 32)
    col_seed = transcript.coin_bytes("columns")
    columns = rs.sample_columns(col_seed, q_columns, cfg.N_LIG)
    return rho, p_commit, open_projection(pw, p_commit, columns), columns
