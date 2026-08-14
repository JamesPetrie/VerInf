"""GPU backend for the two hot loops, using the PRODUCTION Goldilocks kernels.

Two reasons this exists, and the second matters more than the first.

1. Reach. In CPython the prototype tops out around d=64: encoding is
   `rows x ELL x N_LIG` and costs ~33-100 ns per slot. On the GPU the same work
   is one `gl_matmul`, which moves the ceiling by orders of magnitude and lets
   the model be validated at geometries close to a real layer.

2. Fidelity of the rate card. `prover/cuda_primitives.py` is what the production
   prover actually runs. Measuring THOSE kernels gives a rate card that means
   something for a 400B projection; measuring CPython loops does not.

Encoding is a matrix product: message rows (rows x ELL) times the Lagrange matrix
(ELL x N_LIG) gives the codewords (rows x N_LIG). That is exactly what
`rs.encode_row` computes slot by slot, so the two paths must agree BIT FOR BIT --
`tests/test_gpu.py` checks that on every configuration it runs, and the backend
refuses to be used if it does not.

Everything here is optional: `available()` is False without CUDA and the CPU path
runs unchanged.
"""
from typing import List, Optional, Sequence

from prover.protocol import Config, P

from .counters import charge
from .profile import stage

_MODS = None
_LAGRANGE_GPU: dict = {}

# A CUDA grid dimension is capped at 65535, and the batched NTT kernels use one
# block-row per message. Measured exactly: 65535 rows encode, 65536 fails.
MAX_GRID_ROWS = 65535

# The batched forward NTT is a Bailey four-step and allocates a scratch buffer of
# about one full chunk (measured: 17,213,423,616 bytes requested for 32,832 rows
# at N=65536, i.e. rows*N*8 exactly). So the grid cap is not the only limit --
# a chunk must also FIT, together with its scratch and the destination slice.
# Budget three chunk-sized buffers and leave a margin for everything already
# resident.
_NTT_BUFFERS = 3
_FREE_MARGIN = 0.80


def _chunk_rows(cfg: Config, rows: int) -> int:
    """Largest chunk that satisfies both limits. Recomputed per call because the
    free memory depends on what the rest of the prover is holding."""
    torch, _ = _mods()
    cap = min(rows, MAX_GRID_ROWS)
    if torch is None:
        return cap
    free, _total = torch.cuda.mem_get_info()
    per_row = cfg.N_LIG * 8 * _NTT_BUFFERS
    fits = int(free * _FREE_MARGIN) // max(per_row, 1)
    if fits < 1:
        raise RuntimeError(
            f"not enough free device memory to encode even one row at "
            f"N={cfg.N_LIG}: {free/1e9:.1f} GB free, {per_row/1e9:.3f} GB needed")
    return max(1, min(cap, fits))


def _mods():
    global _MODS
    if _MODS is None:
        try:
            import torch
            from prover import cuda_primitives as cp
            _MODS = (torch, cp) if torch.cuda.is_available() else (None, None)
        except Exception:
            _MODS = (None, None)
    return _MODS


def available() -> bool:
    return _mods()[0] is not None


def lagrange_gpu(cfg: Config):
    """The ELL x N_LIG Lagrange matrix as a uint64 device tensor, cached."""
    torch, _ = _mods()
    key = (cfg.ELL, cfg.K_DEG, cfg.N_LIG)
    if key not in _LAGRANGE_GPU:
        from . import rs
        host = rs.lagrange_matrix(cfg)              # [N_LIG][ELL]
        t = torch.tensor(host, dtype=torch.uint64, device="cuda")   # N x ELL
        _LAGRANGE_GPU[key] = t.t().contiguous()     # ELL x N
    return _LAGRANGE_GPU[key]


def encode_batch(cfg: Config, messages: Sequence[Sequence[int]]) -> List[List[int]]:
    """Encode many rows at once: (rows x ELL) @ (ELL x N_LIG) over Goldilocks.

    Charged as the same loop-body units the CPU path uses, so the count model
    does not need to know which backend ran -- only the RATE card changes."""
    torch, cp = _mods()
    rows = len(messages)
    msg = [list(m) + [0] * (cfg.ELL - len(m)) for m in messages]
    nnz = sum(1 for m in msg for v in m if v)
    # gl_matmul is dense: it multiplies every slot, zero or not. So this path
    # gets its own unit rather than the CPU loop's scan/mac split -- the CPU's
    # sparsity saving simply does not exist here.
    charge(mul_defer=rows * cfg.ELL * cfg.N_LIG,
           add=rows * max(cfg.ELL - 1, 0) * cfg.N_LIG,
           enc_gpu=rows * cfg.ELL * cfg.N_LIG, red_op=rows * cfg.N_LIG,
           gpu_elem=rows * cfg.N_LIG)
    A = torch.tensor(msg, dtype=torch.uint64, device="cuda")
    C = cp.gl_matmul(A, lagrange_gpu(cfg))
    return [[int(v) for v in row] for row in C.cpu().tolist()]


def linear_combination(cfg: Config, codewords: Sequence[Sequence[int]],
                       coeffs: Sequence[int]) -> List[int]:
    """sum_i coeffs[i] * codewords[i] as a (1 x rows) @ (rows x N) product."""
    torch, cp = _mods()
    charge(mul_defer=len(coeffs) * cfg.N_LIG, add=len(coeffs) * cfg.N_LIG,
           comb_iter=len(coeffs) * cfg.N_LIG,
           gpu_elem=cfg.N_LIG + len(coeffs) * cfg.N_LIG)
    A = torch.tensor([list(coeffs)], dtype=torch.uint64, device="cuda")
    B = torch.tensor([list(c) for c in codewords], dtype=torch.uint64, device="cuda")
    return [int(v) for v in cp.gl_matmul(A, B)[0].cpu().tolist()]


_COSET: dict = {}


def _coset_powers(cfg: Config):
    """gamma^i for i < K_DEG, on the device. `protocol.eta(j) = gamma * w_N^j`, so
    evaluating on that coset is the same as scaling the coefficients by gamma^i
    and then running a plain NTT."""
    torch, _ = _mods()
    key = cfg.K_DEG
    if key not in _COSET:
        from prover.protocol import GLOBAL_G, P as FP
        _COSET[key] = torch.tensor([pow(GLOBAL_G, i, FP) for i in range(cfg.K_DEG)],
                                   dtype=torch.uint64, device="cuda")
    return _COSET[key]


def encode_batch_ntt(cfg: Config, messages: Sequence[Sequence[int]]):
    """Encode via NTT instead of a dense ELL x N matrix product.

    Same codewords, bit for bit -- `selftest_ntt` checks that, and the backend
    refuses to be used otherwise. The difference is only in how they are computed:

        dense   message x Lagrange(ELL x N)        O(ELL*N) per row, and it needs
                                                   the matrix materialised: at
                                                   production geometry that is
                                                   537M cells, 13 minutes and
                                                   30 GB of host RAM to build.
        NTT     iNTT_K -> scale by gamma^i         O(K log K + N log N) per row,
                -> pad to N -> NTT_N               no matrix at all.

    Measured at production geometry (ELL=8192, N=65536): 54.9 us/row, i.e. 0.0001 ns
    per dense-equivalent slot-position against 0.006 ns for the dense product --
    59x. That difference is the whole reason the four-hour budget was impossible:
    dense encoding of one proof's fresh roots is ~3.7 h even at the isolated kernel
    rate; via NTT it is ~3.7 minutes.

    Steps 1 and 2 exist because our message is the polynomial's VALUES at the K-th
    roots of unity (not its coefficients), and our codeword lives on the COSET
    gamma*<w_N> (not on <w_N>). A plain `rs_encode_rows` assumes neither, which is
    why its output looked unrelated.
    """
    torch, cp = _mods()
    on_device = hasattr(messages, "shape")
    rows = int(messages.shape[0]) if on_device else len(messages)
    charge(mul_defer=rows * cfg.K_DEG * 2, add=rows * cfg.K_DEG,
           enc_gpu=rows * (cfg.K_DEG * max(cfg.K_DEG.bit_length() - 1, 1)
                           + cfg.N_LIG * max(cfg.N_LIG.bit_length() - 1, 1)),
           red_op=rows * cfg.N_LIG,
           gpu_elem=0 if on_device else rows * cfg.ELL)

    with stage("gpu.h2d"):
        buf = torch.zeros((rows, cfg.N_LIG), dtype=torch.uint64, device="cuda")
        if on_device:
            w = int(messages.shape[1])
            buf[:, :w] = messages          # already resident: no transfer at all
        else:
            src = [list(m) + [0] * (cfg.K_DEG - len(m)) for m in messages]
            buf[:, :cfg.K_DEG] = torch.tensor(src, dtype=torch.uint64, device="cuda")
    with stage("gpu.ntt"):
        # Two independent limits, both hit in practice:
        #   * a CUDA grid dimension is capped at 65535 and these kernels use one
        #     block-row per message -- 65535 rows encode, 65536 fails with
        #     `invalid configuration argument`;
        #   * the forward NTT's Bailey scratch is about one chunk in size, so a
        #     chunk that clears the grid cap can still exhaust the card.
        # Chunking is exact either way: each row's transform is independent of
        # every other row's, and `tests/test_gpu.py` pins that a chunked encode
        # is bit-identical to an unchunked one.
        step = _chunk_rows(cfg, rows)
        for lo in range(0, rows, step):
            hi = min(lo + step, rows)
            head = buf[lo:hi, :cfg.K_DEG].contiguous()
            cp.ntt_inverse_batched(head)                   # values -> coefficients
            head = cp.gl_mul(head, _coset_powers(cfg).expand_as(head).contiguous())
            buf[lo:hi, :cfg.K_DEG] = head                  # coset scaling
            chunk = buf[lo:hi].contiguous()
            cp.ntt_forward_batched(chunk)                  # -> codewords
            buf[lo:hi] = chunk
    return buf


def selftest_ntt(cfg: Config, n_rows: int = 4, seed: int = 2):
    """Bit-exactness of the NTT encoder against the dense path. Returns None on
    agreement. Mandatory before the backend is enabled: a faster encoder that
    produces different codewords would change every root in every proof."""
    if not available():
        return "no CUDA"
    import random

    from . import rs
    r = random.Random(seed)
    msgs = [[r.randrange(P) for _ in range(cfg.ELL)] for _ in range(n_rows)]
    got = [[int(v) for v in row] for row in encode_batch_ntt(cfg, msgs).cpu().tolist()]
    for i, m in enumerate(msgs):
        ref = rs.encode_row(cfg, m)
        if ref != got[i]:
            j = next(k for k in range(cfg.N_LIG) if ref[k] != got[i][k])
            return f"row {i} column {j}: dense {ref[j]} != ntt {got[i][j]}"
    return None


def selftest_ntt_spot(cfg: Config, n_cols: int = 24, seed: int = 5):
    """Bit-exactness at a geometry where the dense reference cannot be built.

    `selftest_ntt` compares whole codewords, which needs the ELL x N Lagrange
    matrix -- 537M cells, 13 minutes and 30 GB at production geometry, i.e. exactly
    the thing the NTT encoder exists to avoid. This checks a random SAMPLE of
    columns instead, evaluating `protocol.lagrange` directly for those columns
    only. Wrong-domain or wrong-scaling bugs are global, so they cannot hide in the
    unsampled columns; this is a gate against a broken backend, not a proof of
    every position."""
    if not available():
        return "no CUDA"
    import random

    from prover.protocol import lagrange
    r = random.Random(seed)
    msg = [r.randrange(P) for _ in range(cfg.ELL)]
    got = [int(v) for v in encode_batch_ntt(cfg, [msg])[0].cpu().tolist()]
    for _ in range(n_cols):
        j = r.randrange(cfg.N_LIG)
        eta = cfg.eta(j)
        ref = 0
        for c, v in enumerate(msg):
            if v:
                ref = (ref + v * lagrange(cfg, c, eta)) % P
        if ref != got[j]:
            return f"column {j}: dense {ref} != ntt {got[j]}"
    return None


def encode_batch_t(cfg: Config, messages: Sequence[Sequence[int]]):
    """Encode to a DEVICE TENSOR and leave it there. The list-returning
    `encode_batch` pays ~125 ns per element to marshal the result back to Python;
    a profile showed that, plus column packing, dominated the prover. Keeping the
    codewords on the device removes both."""
    torch, cp = _mods()
    rows = len(messages)
    msg = [list(m) + [0] * (cfg.ELL - len(m)) for m in messages]
    charge(mul_defer=rows * cfg.ELL * cfg.N_LIG,
           add=rows * max(cfg.ELL - 1, 0) * cfg.N_LIG,
           enc_gpu=rows * cfg.ELL * cfg.N_LIG, red_op=rows * cfg.N_LIG,
           gpu_elem=rows * cfg.ELL)                     # host->device of the message
    with stage("gpu.h2d"):
        A = torch.tensor(msg, dtype=torch.uint64, device="cuda")
    with stage("gpu.matmul"):
        return cp.gl_matmul(A, lagrange_gpu(cfg))


def hash_and_tree(cfg: Config, mat):
    """Column digests and the Merkle tree, both on the device, using the
    production kernels. Verified bit-identical to protocol.merkle_leaf +
    rs.build_tree in tests/test_gpu.py -- the packing layout has to match or every
    root would differ."""
    torch, cp = _mods()
    rows = int(mat.shape[0])
    charge(hash_bytes=8 * rows * cfg.N_LIG + 64 * max(cfg.N_LIG - 1, 0),
           hash_calls=cfg.N_LIG + max(cfg.N_LIG - 1, 0),
           gpu_hash_value=rows * cfg.N_LIG)
    with stage("gpu.hash"):
        dig = cp.hash_columns_streamed(mat.contiguous())
        root_t, levels_t = cp.merkle_build_blake3(dig)
    with stage("gpu.d2h"):
        root = bytes(root_t.cpu().tolist())
    return root, levels_t


def levels_to_host(levels_t) -> List[List[bytes]]:
    """Pull the (small) tree to the host once, so opening paths cost nothing."""
    out = []
    for lvl in levels_t:
        arr = lvl.cpu().numpy()
        out.append([bytes(arr[r].tolist()) for r in range(arr.shape[0])])
    return out


def combine_t(cfg: Config, mat, coeffs: Sequence[int]):
    """sum_i coeffs[i] * mat[i] as a (1 x rows) @ (rows x N) product, staying on
    the device."""
    torch, cp = _mods()
    charge(mul_defer=len(coeffs) * cfg.N_LIG, add=len(coeffs) * cfg.N_LIG,
           comb_iter=len(coeffs) * cfg.N_LIG, gpu_elem=len(coeffs))
    A = torch.tensor([list(coeffs)], dtype=torch.uint64, device="cuda")
    return cp.gl_matmul(A, mat)


def column(mat, col: int) -> List[int]:
    """One codeword column back to the host. Only q of N columns are ever opened,
    so this is the only marshalling left on the hot path."""
    charge(gpu_elem=int(mat.shape[0]))
    with stage("gpu.column_d2h"):
        return [int(v) for v in mat[:, col].cpu().tolist()]


def _poly_eval(coeffs, x, p):
    acc = 0
    for c in reversed(coeffs):
        acc = (acc * x + c) % p
    return acc


def gl_sum(t):
    """Sum a Goldilocks vector on the device by pairwise gl_add. A plain
    torch.sum would overflow uint64 after a couple of terms; a log-depth tree
    stays in the field at every step."""
    torch, cp = _mods()
    v = t.reshape(-1)
    while v.numel() > 1:
        n = v.numel()
        if n & 1:
            v = torch.cat([v, torch.zeros(1, dtype=v.dtype, device=v.device)])
            n += 1
        v = cp.gl_add(v[: n // 2].contiguous(), v[n // 2:].contiguous())
    return v


def fold_t(f, r_t):
    """(f[:h] + r*(f[h:] - f[:h])) mod P, elementwise on the device -- the same
    body as sumcheck._fold, which is why the two must agree bit for bit."""
    torch, cp = _mods()
    h = f.numel() // 2
    lo, hi = f[:h].contiguous(), f[h:].contiguous()
    return cp.gl_add(lo, cp.gl_mul(r_t.expand(h).contiguous(), cp.gl_sub(hi, lo)))


def prove_terms_gpu(terms, coin, field_p: int, mask=None, mu0: int = 0):
    """sumcheck.prove_terms with the factors held as device tensors.

    Same arithmetic, same order, same challenges -- `tests/test_gpu.py` proves the
    round polynomials come out IDENTICAL to the CPython path. What changes is that
    the per-round work is elementwise over the whole vector instead of a triple
    Python loop, which is what made the sumcheck the next bottleneck after the
    commitments moved to the device."""
    torch, cp = _mods()
    dev = "cuda"

    def T(v):
        return torch.tensor([x % field_p for x in v], dtype=torch.uint64, device=dev)

    cur = [(c % field_p, [T(f) for f in fs]) for c, fs in terms]
    size = int(cur[0][1][0].numel())
    n_rounds = size.bit_length() - 1
    deg = max(len(fs) for _, fs in cur)

    mu = mu0 % field_p
    total = torch.zeros(1, dtype=torch.uint64, device=dev)
    for c, fs in cur:
        prod = fs[0]
        for f in fs[1:]:
            prod = cp.gl_mul(prod, f)
        cval = torch.tensor([c], dtype=torch.uint64, device=dev)
        total = cp.gl_add(total, gl_sum(cp.gl_mul(prod, cval.expand(prod.numel()).contiguous())))
    claim = (int(total[0].item()) + mu) % field_p

    charge(gpu_elem=sum(len(fs) for _, fs in cur) * size)
    round_polys, challenges = [], []
    for rnd in range(n_rounds):
        half = int(cur[0][1][0].numel()) // 2
        samples = []
        for x in range(deg + 1):
            xt = torch.tensor([x], dtype=torch.uint64, device=dev)
            acc = torch.zeros(1, dtype=torch.uint64, device=dev)
            for c, fs in cur:
                prod = None
                for f in fs:
                    fx = fold_t(f, xt)
                    prod = fx if prod is None else cp.gl_mul(prod, fx)
                cval = torch.tensor([c], dtype=torch.uint64, device=dev)
                acc = cp.gl_add(acc, gl_sum(cp.gl_mul(prod, cval.expand(prod.numel()).contiguous())))
            samples.append((x, int(acc[0].item())))
        # The §7 mask touches only the SCALAR samples, never the vector work --
        # which is why masked proofs can use this path too.
        if mask is not None:
            h = mask(deg, mu)
            samples = [(x, (y + _poly_eval(h, x, field_p)) % field_p) for x, y in samples]
        n_fac = sum(len(fs) for _, fs in cur)
        charge(fold_iter=(deg + 1) * half * n_fac, red_op=(deg + 1) * half * n_fac)
        round_polys.append(samples)
        r = coin(rnd) % field_p
        challenges.append(r)
        if mask is not None:
            mu = _poly_eval(h, r, field_p)
        rt = torch.tensor([r], dtype=torch.uint64, device=dev)
        cur = [(c, [fold_t(f, rt) for f in fs]) for c, fs in cur]
        charge(fold_iter=half * n_fac)
    return claim, round_polys, challenges, mu


def selftest(cfg: Config, n_rows: int = 4, seed: int = 1) -> Optional[str]:
    """Bit-exactness against the CPU path. Returns None if they agree, else the
    first disagreement. Called before the backend is enabled -- a fast backend
    that computes something else is worse than no backend."""
    if not available():
        return "no CUDA"
    import random

    from . import rs
    r = random.Random(seed)
    msgs = [[r.randrange(P) for _ in range(cfg.ELL)] for _ in range(n_rows)]
    gpu = encode_batch(cfg, msgs)
    for i, m in enumerate(msgs):
        cpu = rs.encode_row(cfg, m)
        if cpu != gpu[i]:
            j = next(k for k in range(cfg.N_LIG) if cpu[k] != gpu[k])
            return f"row {i} column {j}: cpu {cpu[j]} != gpu {gpu[i][j]}"
    cw = [rs.encode_row(cfg, m) for m in msgs]
    co = [r.randrange(P) for _ in range(n_rows)]
    if rs.linear_combination(cfg, cw, co) != linear_combination(cfg, cw, co):
        return "linear_combination disagrees"
    return None
