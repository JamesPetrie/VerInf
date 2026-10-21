"""The field kernels an engine pass reaches, exact mod P in Python integers on
the CPU, so a toy tape's values can be computed without a card (inside
_cpu_tape.cuda_on_cpu). Slow: toys only. The proof's kernels (NTT, hashing,
interpolation) are not here, so a proof still needs a card.

    with ct.cuda_on_cpu(), cpu_field():
        live = tape.run_engine_pass(...)

The kernels are the compiled extension's (prover/kernels/*.cuh) by contract:
the same tapes on a card must give the same values, which the GPU suites
check (tests/test_k2_run_only.py)."""
import contextlib

import numpy as np
import torch

P = (1 << 64) - (1 << 32) + 1


def _ints(t):
    """A uint64 tensor's values as Python integers (an object array, its shape kept)."""
    return t.contiguous().numpy().astype(object)


def _u64(a):
    return torch.from_numpy(np.asarray(a, dtype=object).astype(np.uint64))


class Kernels:
    """The compiled extension's engine-pass entry points."""

    @staticmethod
    def gl_add(a, b):
        return _u64((_ints(a) + _ints(b)) % P)

    @staticmethod
    def gl_sub(a, b):
        return _u64((_ints(a) - _ints(b)) % P)

    @staticmethod
    def gl_neg(a):
        return _u64((-_ints(a)) % P)

    @staticmethod
    def gl_mul(a, b):
        return _u64((_ints(a) * _ints(b)) % P)

    @staticmethod
    def gl_pow(base, exp):
        return _u64(np.vectorize(lambda v: pow(int(v), int(exp), P), otypes=[object])(_ints(base)))

    @staticmethod
    def gl_inv_batched(a):
        return _u64(np.vectorize(lambda v: pow(int(v), P - 2, P), otypes=[object])(_ints(a)))

    @staticmethod
    def gl_axpy_inplace(y, alpha, x):
        y.copy_(_u64((_ints(y) + int(alpha) * _ints(x)) % P))

    @staticmethod
    def gl_matmul(A, B):
        return _u64(np.dot(_ints(A), _ints(B)) % P)

    @staticmethod
    def gl_matvec(M, v):
        return _u64(np.dot(_ints(M), _ints(v)) % P)

    @staticmethod
    def lookup_multiplicities_range_into(x, n, mult):
        keys = x.contiguous().view(-1).view(torch.int64).numpy()
        keys = keys[(keys >= 0) & (keys < n)]
        np.add.at(mult.view(torch.int64).numpy(), keys, 1)

    _index = {}

    @staticmethod
    def lookup_multiplicities_into(x, table, mult):
        key = (table.data_ptr(), table.numel())
        index = Kernels._index.get(key)
        if index is None:
            index = Kernels._index[key] = {int(v): j for j, v in enumerate(table.view(-1).tolist())}
        m = mult.view(torch.int64).numpy()
        for v in x.contiguous().view(-1).tolist():
            j = index.get(int(v))
            if j is not None:
                m[j] += 1

    @staticmethod
    def lookup_multiplicities(x, table):
        mult = torch.zeros(table.numel(), dtype=torch.int64).view(torch.uint64)
        Kernels.lookup_multiplicities_into(x, table, mult)
        return mult


@contextlib.contextmanager
def cpu_field():
    """cuda_primitives' compiled extension replaced by Kernels."""
    import cuda_primitives as cp
    saved = cp._ensure_compiled
    cp._ensure_compiled = lambda: Kernels
    try:
        yield
    finally:
        cp._ensure_compiled = saved
