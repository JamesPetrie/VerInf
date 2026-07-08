"""uint64 CUDA op-coverage shim.

Stock (x86) PyTorch ships CUDA kernels for only a subset of dtypes on several
data-movement ops; `uint64` is commonly missing, e.g.

    RuntimeError: "cuda_scatter_gather_base_kernel_func" not implemented for 'UInt64'

The Goldilocks field reps are carried as `uint64`, so `gather`/`index_select`/...
on them dispatch into that gap. Each of these ops is *pure data movement*: it
only copies/permutes 8-byte lanes and never interprets the sign, so rerunning it
on the `int64` view of the same storage (`.view(torch.int64)`) is bit-identical.

This module monkeypatches those ops to fall back to the int64 view **only** when
the native uint64 dispatch is actually missing (it tries the native path first
and matches the "...not implemented for 'UInt64'" message), so it is a no-op on
torch builds that already implement uint64 (e.g. the DGX Spark reference env).

Importing the module installs the patches. Import it once, early, before any
prover op runs (done at the top of `demo_llama7b.py`).

NOTE: only sign-agnostic ops are patched here. Sign-sensitive ops on field reps
(comparisons, sort, cumsum) are handled explicitly in the prover via deliberate
int64-view arithmetic and must NOT be routed through this shim.
"""
import torch

_U64 = torch.uint64
_I64 = torch.int64


def _is_u64_unimpl(exc) -> bool:
    s = str(exc)
    return "not implemented for 'UInt64'" in s or "not implemented for 'unsigned long'" in s


def _as_i64(x):
    return x.view(_I64) if torch.is_tensor(x) and x.dtype == _U64 else x


def _restore_u64(out, had_u64):
    if had_u64 and torch.is_tensor(out) and out.dtype == _I64:
        return out.view(_U64)
    return out


def _patch_method(name):
    orig = getattr(torch.Tensor, name)

    def wrapper(self, *args, **kwargs):
        try:
            return orig(self, *args, **kwargs)
        except (RuntimeError, NotImplementedError) as exc:
            if not (self.is_cuda and _is_u64_unimpl(exc)):
                raise
            had = self.dtype == _U64
            out = orig(_as_i64(self), *[_as_i64(a) for a in args],
                       **{k: _as_i64(v) for k, v in kwargs.items()})
            return _restore_u64(out, had)

    wrapper.__name__ = name
    setattr(torch.Tensor, name, wrapper)


def _patch_func(name):
    orig = getattr(torch, name)

    def wrapper(*args, **kwargs):
        try:
            return orig(*args, **kwargs)
        except (RuntimeError, NotImplementedError) as exc:
            if not _is_u64_unimpl(exc):
                raise
            had = any(torch.is_tensor(a) and a.dtype == _U64 for a in args)
            out = orig(*[_as_i64(a) for a in args],
                       **{k: _as_i64(v) for k, v in kwargs.items()})
            return _restore_u64(out, had)

    wrapper.__name__ = name
    setattr(torch, name, wrapper)


# Pure data-movement ops: copy/permute lanes only, sign never inspected.
_METHODS = ("gather", "index_select", "take", "masked_select", "repeat_interleave")
_FUNCS = ("gather", "index_select", "take", "masked_select", "repeat_interleave")

for _m in _METHODS:
    _patch_method(_m)
for _f in _FUNCS:
    _patch_func(_f)
