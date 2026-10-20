"""Structure-only tapes without a card. A lazy Tape records its claims,
variables and tables without computing a witness, so a driver's build runs
on the CPU once the device placements it makes are redirected
(`cuda_on_cpu`); `structure` then fingerprints what was built: every claim's
type and fields (variables by name, length and phase; tables by name, size
and a hash of their entries; configurations field by field), in build order.
It locks a driver's claim configuration and counts its witness. Values,
proofs and verdicts still need a card."""
import contextlib
import dataclasses
import functools
import hashlib
import json

import numpy as np
import torch

_FACTORIES = ("zeros", "ones", "empty", "full", "arange", "tensor", "as_tensor", "zeros_like",
              "ones_like", "empty_like", "full_like")


def _cpu(dev):
    if (isinstance(dev, str) and dev.startswith("cuda")) or (isinstance(dev, torch.device) and dev.type == "cuda"):
        return "cpu"
    return dev


@contextlib.contextmanager
def cuda_on_cpu():
    """torch's factories, Tensor.cuda and Tensor.to with "cuda" placed on the CPU."""
    saved = {n: getattr(torch, n) for n in _FACTORIES}
    saved_cuda, saved_to = torch.Tensor.cuda, torch.Tensor.to

    def remap(fn):
        @functools.wraps(fn)
        def f(*a, **kw):
            if "device" in kw:
                kw["device"] = _cpu(kw["device"])
            return fn(*a, **kw)
        return f

    def to(self, *a, **kw):
        a = tuple(_cpu(x) for x in a)
        if "device" in kw:
            kw["device"] = _cpu(kw["device"])
        return saved_to(self, *a, **kw)
    try:
        for n, fn in saved.items():
            setattr(torch, n, remap(fn))
        torch.Tensor.cuda = lambda self, *a, **kw: self
        torch.Tensor.to = to
        yield
    finally:
        for n, fn in saved.items():
            setattr(torch, n, fn)
        torch.Tensor.cuda, torch.Tensor.to = saved_cuda, saved_to


def _entries(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()[:16]


def _canon(v, tables):
    """`tables`: name -> canonical table, filled as tables are met."""
    from core import Table, Variable
    if isinstance(v, Variable):
        return ["var", v.name, v.length, v.phase, v.persistent]
    if isinstance(v, Table):
        t = ["table", v.name, int(v.T.numel()), _entries(v.T),
             None if v.T_Y is None else _entries(v.T_Y)]
        tables[v.name] = t
        return t
    if isinstance(v, torch.Tensor):
        return ["tensor", list(v.shape), str(v.dtype)]
    if isinstance(v, np.ndarray):
        return ["ndarray", list(v.shape), str(v.dtype)]
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        return [type(v).__name__, {f.name: _canon(getattr(v, f.name), tables) for f in dataclasses.fields(v)}]
    if isinstance(v, (list, tuple)):
        return [_canon(x, tables) for x in v]
    if isinstance(v, dict):
        return {str(k): _canon(x, tables) for k, x in v.items()}
    if isinstance(v, (np.integer, np.bool_)):
        return v.item()
    if v is None or isinstance(v, (int, float, str, bool)):
        return v
    return type(v).__name__


def structure(tape) -> dict:
    """The tape's claims in build order and the tables they use, canonical;
    with its digest (sha256 of the canonical JSON)."""
    tables = {}
    claims = [_canon(c, tables) for c in tape.claims]
    body = dict(claims=claims, tables=[tables[n] for n in sorted(tables)])
    return dict(body, digest=hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest())
