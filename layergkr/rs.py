"""RS encode + column Merkle commit, CPU, prototype scale.

Deliberately a THIN layer over the existing `prover/protocol.py`: same field,
same message->codeword map (`eval_zeta_form` over the zeta/eta domains), same
leaf hashing and the same path convention, so anything proven here is stated in
the encoding the production verifier already speaks. The only thing added is a
pure-Python tree builder (production builds it on GPU in cuda_primitives).

Nothing in here is new cryptography. It exists so `projection.py` and the tests
can commit and open at toy sizes without a GPU.
"""
from typing import List, Sequence, Tuple

import blake3

from prover.protocol import (  # noqa: F401  (re-exported on purpose)
    Config, P, add, inv, merkle_leaf, merkle_verify, mul, sub,
    challenge, eval_zeta_form,
)

from .counters import charge

Codeword = List[int]
Path = List[Tuple[bytes, int]]


def _b3(a: bytes, b: bytes) -> bytes:
    return blake3.blake3(a + b).digest()


_LAGRANGE_CACHE: dict = {}
_GPU_OK: dict = {}


_NTT_OK: dict = {}


def _gpu_ok(cfg: Config) -> bool:
    """Enable the GPU backend for this Config only after it has proved itself
    bit-identical to the CPU path. A fast backend that computes something else is
    worse than no backend, so the check is mandatory and cached per Config."""
    key = (cfg.ELL, cfg.K_DEG, cfg.N_LIG)
    if key not in _GPU_OK:
        from . import gpu
        _GPU_OK[key] = gpu.available() and gpu.selftest(cfg) is None
    return _GPU_OK[key]


def _ntt_ok(cfg: Config) -> bool:
    """Same gate for the NTT encoder. It is preferred when it passes, because it
    needs no Lagrange matrix at all -- at production geometry that matrix is 537M
    cells, 13 minutes and 30 GB to build, and the dense product it enables is what
    made the four-hour budget impossible."""
    key = (cfg.ELL, cfg.K_DEG, cfg.N_LIG)
    if key not in _NTT_OK:
        from . import gpu
        if not gpu.available():
            _NTT_OK[key] = False
        elif cfg.ELL * cfg.N_LIG <= (1 << 24):
            _NTT_OK[key] = gpu.selftest_ntt(cfg) is None          # whole codewords
        else:
            _NTT_OK[key] = gpu.selftest_ntt_spot(cfg) is None     # sampled columns
    return _NTT_OK[key]


def lagrange_matrix(cfg: Config) -> List[List[int]]:
    """L[j][c] = L_c(eta_j): the ELL x N_LIG map from message slots to codeword
    positions. Depends only on the Config, so it is built once and reused --
    without this, encoding dominates the prototype's wall-clock (every slot
    would redo two modular exponentiations per column).

    This is a pure restatement of protocol.lagrange, and `tests/test_rs.py`
    checks the cached path against `eval_zeta_form` element by element."""
    key = (cfg.ELL, cfg.K_DEG, cfg.N_LIG)
    if key not in _LAGRANGE_CACHE:
        _LAGRANGE_CACHE[key] = _build_lagrange(cfg)
    return _LAGRANGE_CACHE[key]


def _build_lagrange(cfg: Config) -> List[List[int]]:
    """L_c(eta_j) = zeta_c * (eta_j^K - 1) / (K * (eta_j - zeta_c)).

    The naive build calls protocol.lagrange per entry: two modular exponentiations
    and a Fermat inversion each, ~200 field ops per cell. At ELL=2048/N=8192 that
    is 16.8M cells and about a quarter of an hour of pure setup, which swamped the
    experiment it was setting up.

    Batched instead. Per column j: `eta^K` once, and the ELL denominators
    `K*(eta_j - zeta_c)` inverted TOGETHER by Montgomery's trick -- one inversion
    for the whole column instead of ELL of them. ~4 ops per cell.

    `tests/test_rs.py` checks the result cell by cell against protocol.lagrange:
    this is a faster route to the same matrix, not a different matrix."""
    from prover.protocol import lagrange as _ref  # noqa: F401  (used by the test)
    from .field import batch_inv

    K, ELL, N = cfg.K_DEG, cfg.ELL, cfg.N_LIG
    zetas = [cfg.zeta(c) for c in range(ELL)]
    Kmod = K % P
    out = []
    for j in range(N):
        eta = cfg.eta(j)
        num = (pow(eta, K, P) - 1) % P
        dens = [(Kmod * ((eta - z) % P)) % P for z in zetas]
        invs = batch_inv(dens)
        out.append([(z * num % P) * iv % P for z, iv in zip(zetas, invs)])
    return out


def encode_row(cfg: Config, message: Sequence[int]) -> Codeword:
    """Message (<= ELL field values, at the zeta points) -> full codeword of
    length N_LIG (values at the eta coset points)."""
    if len(message) > cfg.ELL:
        raise ValueError(f"message {len(message)} > ELL {cfg.ELL}")
    vals = list(message) + [0] * (cfg.ELL - len(message))
    L = lagrange_matrix(cfg)
    # The loop scans all ELL slots but only multiplies the NONZERO ones. Most
    # rows here are mostly padding, so those are very different numbers, and
    # charging the dense product over-prices encoding by several fold.
    nnz = sum(1 for v in vals if v)
    charge(mul_defer=nnz * cfg.N_LIG, add=max(nnz - 1, 0) * cfg.N_LIG,
           enc_slot=nnz * cfg.N_LIG, enc_scan=cfg.ELL * cfg.N_LIG,
           red_op=cfg.N_LIG)
    out = []
    for j in range(cfg.N_LIG):
        Lj = L[j]
        acc = 0
        for c, v in enumerate(vals):
            if v:
                acc += v * Lj[c]
        out.append(acc % P)
    return out


def build_tree(leaves: List[bytes]) -> List[List[bytes]]:
    """levels[0] = leaves, last level = [root]. Odd node is paired with itself,
    matching prover/core.merkle_path's out-of-range rule."""
    levels = [list(leaves)]
    while len(levels[-1]) > 1:
        cur = levels[-1]
        nxt = [_b3(cur[i], cur[i + 1] if i + 1 < len(cur) else cur[i])
               for i in range(0, len(cur), 2)]
        levels.append(nxt)
    return levels


def merkle_path(levels: List[List[bytes]], idx: int) -> Path:
    """Opening path for leaf `idx` — byte-identical convention to
    prover/core.merkle_path, so prover/protocol.merkle_verify checks it."""
    path: Path = []
    for level in levels[:-1]:
        sib = idx ^ 1
        if sib >= len(level):
            sib = idx
        path.append((level[sib], 1 if (idx & 1) == 0 else 0))
        idx //= 2
    return path


class Commit:
    """A column-Merkle commitment to a set of RS rows.

    Two backends, one meaning. On CPU the codewords are Python lists and the
    columns are packed and hashed in CPython. On GPU the codewords stay a device
    TENSOR, and the production kernels hash the columns and build the tree without
    ever bringing the matrix back -- only the q opened columns cross to the host.

    Profiling forced this: `protocol.pack_column` (one `int.to_bytes` per value)
    plus the device->host marshalling were together most of the prover's time.
    Both are gone on the GPU path. The two backends produce IDENTICAL roots, which
    tests/test_gpu.py checks -- if they did not, every proof would differ.
    """

    GPU_MIN_WORK = 1 << 20        # rows * ELL * N_LIG

    def __init__(self, cfg: Config, codewords: List[Codeword] = None, mat=None):
        self.cfg = cfg
        self.mat = mat
        if mat is not None:
            from . import gpu
            self._root, self._levels_dev = gpu.hash_and_tree(cfg, mat)
            self._levels = None
            self._codewords = None
            return
        self._codewords = codewords
        self.columns = [[cw[j] for cw in codewords] for j in range(cfg.N_LIG)]
        n_rows = len(codewords)
        charge(xpose_iter=n_rows * cfg.N_LIG)
        charge(hash_bytes=8 * n_rows * cfg.N_LIG + 64 * max(cfg.N_LIG - 1, 0),
               hash_calls=cfg.N_LIG + max(cfg.N_LIG - 1, 0),
               pack_value=n_rows * cfg.N_LIG)
        self._levels = build_tree([merkle_leaf(c) for c in self.columns])
        self._root = self._levels[-1][0]

    @classmethod
    def from_messages(cls, cfg: Config, messages) -> "Commit":
        """`messages` may be Python rows OR an already-on-device (rows x ELL)
        tensor. The tensor form exists because measurement said so: at production
        geometry the NTT encode itself is 1.1 ms per 64 rows while shipping the
        Python lists to the card is 86 ms -- 99% of the path. Data that is already
        on the device must not make that round trip."""
        if hasattr(messages, "shape"):
            from . import gpu
            if _ntt_ok(cfg):
                return cls(cfg, mat=gpu.encode_batch_ntt(cfg, messages))
        work = len(messages) * cfg.ELL * cfg.N_LIG
        if work >= cls.GPU_MIN_WORK:
            from . import gpu
            if _ntt_ok(cfg):
                return cls(cfg, mat=gpu.encode_batch_ntt(cfg, messages))
            if _gpu_ok(cfg):
                return cls(cfg, mat=gpu.encode_batch_t(cfg, messages))
        return cls(cfg, [encode_row(cfg, m) for m in messages])

    @property
    def on_gpu(self) -> bool:
        return self.mat is not None

    @property
    def levels(self) -> List[List[bytes]]:
        if self._levels is None:
            from . import gpu
            self._levels = gpu.levels_to_host(self._levels_dev)
        return self._levels

    @property
    def codewords(self) -> List[Codeword]:
        """Materialise the whole matrix on the host. Only tests and the CPU path
        need this; the prover deliberately never calls it on the GPU path."""
        if self._codewords is None:
            self._codewords = [[int(v) for v in row] for row in self.mat.cpu().tolist()]
        return self._codewords

    @property
    def root(self) -> bytes:
        return self._root

    @property
    def n_rows(self) -> int:
        return int(self.mat.shape[0]) if self.on_gpu else len(self._codewords)

    def col_values(self, col: int) -> List[int]:
        if self.on_gpu:
            from . import gpu
            return gpu.column(self.mat, col)
        return self.columns[col]

    def open(self, col: int) -> Tuple[List[int], Path]:
        return self.col_values(col), merkle_path(self.levels, col)

    def check_open(self, col: int, values: List[int], path: Path) -> bool:
        return merkle_verify(merkle_leaf(values), path, self.root)

    def combine(self, coeffs: Sequence[int]):
        """sum_i coeffs[i] * row_i, on whichever backend holds the data."""
        if self.on_gpu:
            from . import gpu
            return gpu.combine_t(self.cfg, self.mat, coeffs)
        return linear_combination(self.cfg, self._codewords, coeffs)


def linear_combination(cfg: Config, codewords: List[Codeword],
                       coeffs: Sequence[int]) -> Codeword:
    """sum_i coeffs[i] * codewords[i], pointwise. RS codes are linear, so this
    is the codeword of the same combination of the MESSAGES — including their
    secret padding halves. That identity is the whole basis of the projection
    seam in projection.py, and `tests/test_projection.py` checks it directly
    rather than assuming it."""
    if len(codewords) != len(coeffs):
        raise ValueError("coeffs/codewords length mismatch")
    charge(mul_defer=len(coeffs) * cfg.N_LIG, add=len(coeffs) * cfg.N_LIG,
           comb_iter=len(coeffs) * cfg.N_LIG)
    out = [0] * cfg.N_LIG
    for cw, a in zip(codewords, coeffs):
        if a == 0:
            continue
        for j in range(cfg.N_LIG):
            out[j] = (out[j] + cw[j] * a) % P
    return out


def sample_columns(seed: bytes, count: int, n_lig: int) -> List[int]:
    """Distinct column indices from a verifier coin. Distinctness matters: the
    opened-column ledger (doc §3.2) budgets ELL openings per persistent root,
    and repeats would waste that budget without adding soundness."""
    cols, i = [], 0
    seen = set()
    while len(cols) < count:
        c = challenge(seed, i, "cols") % n_lig
        i += 1
        if c not in seen:
            seen.add(c)
            cols.append(c)
    return cols
