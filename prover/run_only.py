"""A run-only engine pass: a lazy tape's values computed once, without a
proof, keeping only the variables the caller names (the logits), with the two
checks a proof would otherwise make, summarized compactly.

- **Lookups.** The engine pass never refuses a key outside its table: a
  lookup clamps its index or counts only the keys it finds, and leaves the
  rejection to the proof's LogUp identities. Here every key is counted as it
  is looked up (at the multiplicity kernel, so every claim's lookups are
  covered), and the ones outside their table are reported per claim.
- **Field safety.** A product or a sum is computed modulo P, and its signed
  representative is the integer only while |v| ≤ (P − 1)/2. The extremes of a
  reduced value cannot show a wrap, so a claim's output is trusted only once a
  bound on it, from its inputs' magnitudes, is within that: K·max|a|·max|b| for
  a product, then max(|a|·|b|) in float64 for a matmul when that is not
  enough. An input's magnitude is exact once its own producer's bound held (a
  representative of a small enough integer is that integer), so the bounds
  are carried from the tape's inputs claim by claim (`ESTABLISHED_BY_BOUND`).
  The other claims (`ESTABLISHED_BY_CONSTRAINTS`) move values, look them up in
  tables or compute within windows their own range checks enforce; their
  outputs are trusted when every input is, and their lookups are counted.

It reuses the engine pass's machinery (tape.run_engine_pass with
free_intermediates), not a witness: each value is freed after its last use,
so nothing a proof needs survives the pass."""
import contextlib
import math
import time
from collections import defaultdict

import torch

P = (1 << 64) - (1 << 32) + 1
P_HALF = (P - 1) // 2
_GAP = (1 << 64) - P

# claims whose outputs are trusted once a bound from their inputs is within P_HALF
ESTABLISHED_BY_BOUND = ("MatmulClaim", "HadamardClaim", "RoPEClaim", "RoutedProjectedMatmulClaim",
                        "FreivaldsCombineClaim", "AddClaim", "WordExtractionClaim", "RescaleClaim")
# claims that move values, read tables or compute inside windows their own lookups check
ESTABLISHED_BY_CONSTRAINTS = ("ConcatClaim", "HeadInterleaveClaim", "EmbeddingLookupClaim", "SplitClaim",
                              "RoutingClaim", "TopkRoutingClaim", "TopkSlotsClaim", "GateBracketClaim",
                              "PairedTlookupClaim", "RangeWordClaim", "RmsNormClaim", "SiluClaim",
                              "SoftmaxClaim", "MaxClaim", "InfoFinalizeClaim", "LinCombClaim")


def lift(t):
    """Field representatives (uint64) as signed int64 in [−(P−1)/2, (P−1)/2]."""
    i = t.contiguous().view(-1).view(torch.int64)
    return torch.where((i < 0) | (i > P_HALF), i + _GAP, i)


def maxabs(t) -> int:
    return int(lift(t).abs().max()) if t.numel() else 0


def extremes(t):
    s = lift(t)
    return (int(s.min()), int(s.max())) if s.numel() else (0, 0)


def bits(v) -> float:
    """log2 of a magnitude, to two places (0 for 0)."""
    return round(math.log2(v), 2) if v else 0.0


def _tight_matmul_bound(A, B, c) -> int:
    """max over the output of Σ_k |a_ik|·|b_kj| in float64, padded for its
    rounding: an upper bound on |C| that the cheap one may overstate by K."""
    inner = c.head_dim if c.heads > 1 else c.k
    a = lift(A).abs().to(torch.float64)
    b = lift(B).abs().to(torch.float64)
    if c.heads == 1:
        a = a.view(c.m, c.k)
        b = b.view(c.n, c.k).t() if c.transpose_b else b.view(c.k, c.n)
        top = float((a @ b).max())
    else:
        H, K = c.heads, c.head_dim
        a3 = a.view(c.m, H, K)
        b3 = b.view(c.n, H, K) if c.transpose_b else b.view(K, H, c.n)
        top = 0.0
        for h in range(H):
            bh = b3[:, h, :].t() if c.transpose_b else b3[:, h, :]
            top = max(top, float((a3[:, h, :] @ bh).max()))
    return int(top * (1 + inner * 2.0 ** -50)) + 1


class RunOnly:
    """The observer of a run-only pass (tape.run_engine_pass(observer=...))
    and its kernel hooks (`hooks()`). `layer_of` maps id(claim) to a layer tag;
    `names` maps a Variable to a name whose extremes are recorded."""

    def __init__(self, tape, layer_of=None, names=None):
        self.tape = tape
        self.layer_of = layer_of or {}
        self.names = names or {}
        self.mag = {}              # Variable -> exact max |v| (int), or None: not established
        self.cur = (None, None)
        self.lookups = defaultdict(lambda: dict(keys=0, outside=0, min=None, max=None))
        self.outside = []
        self.bounds = []           # one row per bounded claim output
        self.unestablished = []
        self.trace = {}
        self.unknown_types = set()
        self._table_of = {}        # data_ptr of a multiplicity tensor -> table name
        for v, t in tape.inputs.items():
            if v.name.endswith("_mult") and isinstance(t, torch.Tensor):
                self._table_of[t.data_ptr()] = v.name[:-len("_mult")]

    # ---- hooks ----------------------------------------------------------------------

    def _count(self, mult, n_keys, outside, keys):
        i, claim = self.cur
        table = self._table_of.get(mult.data_ptr(), "?")
        tag = self.layer_of.get(id(claim)) if claim is not None else None
        row = self.lookups[f"{tag}|{table}"]
        row["keys"] += n_keys
        row["outside"] += outside
        if keys is not None and keys.numel():
            lo, hi = int(keys.min()), int(keys.max())
            row["min"] = lo if row["min"] is None else min(row["min"], lo)
            row["max"] = hi if row["max"] is None else max(row["max"], hi)
        if outside:
            self.outside.append(dict(claim=i, type=type(claim).__name__, layer=tag, table=table,
                                     outside=outside, of=n_keys))

    @contextlib.contextmanager
    def hooks(self):
        """The compiled extension's multiplicity kernels counted, and every
        expert shard's magnitude taken as the routed claims resolve it."""
        import cuda_primitives as cp
        import routed_projected as rp
        ro, ext_of, resolve = self, cp._ensure_compiled, rp._resolve

        class Counting:
            def __init__(self, ext):
                self._ext = ext

            def __getattr__(self, name):
                return getattr(self._ext, name)

            def lookup_multiplicities_range_into(self, x, n, mult):
                k = x.contiguous().view(-1).view(torch.int64)
                ro._count(mult, k.numel(), int(((k < 0) | (k >= n)).sum()), k)
                return self._ext.lookup_multiplicities_range_into(x, n, mult)

            def lookup_multiplicities_into(self, x, table, mult):
                m = mult.view(torch.int64)
                before = int(m.sum())
                out = self._ext.lookup_multiplicities_into(x, table, mult)
                n = x.numel()
                ro._count(mult, n, n - (int(m.sum()) - before), None)
                return out

        def resolving(live, var):
            t = resolve(live, var)
            if var not in ro.mag:
                ro.mag[var] = maxabs(t)
            return t
        cp._ensure_compiled = lambda: Counting(ext_of())
        rp._resolve = resolving
        try:
            yield self
        finally:
            cp._ensure_compiled, rp._resolve = ext_of, resolve

    # ---- the observer -----------------------------------------------------------------

    def before_claim(self, i, claim):
        self.cur = (i, claim)

    def _m(self, v, input_data, live):
        """An input's magnitude: carried, or read once from a tape input."""
        if v in self.mag:
            return self.mag[v]
        t = input_data.get(v)
        if t is None:
            t = live.get(v)            # a fold claim's input; never a loader resolved again here
        self.mag[v] = maxabs(t) if isinstance(t, torch.Tensor) else None
        return self.mag[v]

    def _bounded(self, claim, var, bound, outs, tight=None):
        """Trust `var` if `bound` (None: an input not established) is within
        P_HALF, after the tight bound if one is offered."""
        cheap = bound
        if bound is not None and bound > P_HALF and tight is not None:
            bound = tight()
        ok = bound is not None and bound <= P_HALF
        seen = maxabs(outs[var]) if ok else None
        row = dict(claim=self.cur[0], type=type(claim).__name__, layer=self.layer_of.get(id(claim)),
                   var=var.name, bound_bits=bits(bound) if bound is not None else None,
                   cheap_bits=bits(cheap) if cheap is not None else None,
                   observed_bits=bits(seen) if ok else None, ok=ok)
        self.bounds.append(row)
        if not ok:
            self.unestablished.append(row)
        self.mag[var] = seen
        return ok

    def __call__(self, i, claim, input_vars, input_data, outs, live):
        kind = type(claim).__name__
        m = lambda v: self._m(v, input_data, live)
        both = lambda a, b: None if a is None or b is None else a * b
        full = None
        if kind == "MatmulClaim":
            full = claim.C_full if claim.rescale_bits else claim.C
            inner = claim.head_dim if claim.heads > 1 else claim.k
            b = both(m(claim.A), m(claim.B))
            self._bounded(claim, full, None if b is None else inner * b, outs,
                          tight=lambda: _tight_matmul_bound(input_data[claim.A], input_data[claim.B], claim))
        elif kind == "HadamardClaim":
            full = claim.c_full if claim.rescale_bits else claim.c
            self._bounded(claim, full, both(m(claim.a), m(claim.b)), outs)
        elif kind == "RoPEClaim":
            from claims import _rope_cos_sin
            full = claim.x_rot_full if claim.rescale_bits else claim.x_rot
            c_l, s_l = _rope_cos_sin(claim.config)
            cs = max(max((min(v, P - v) for v in c_l), default=0) + max((min(v, P - v) for v in s_l), default=0), 0)
            self._bounded(claim, full, both(cs, m(claim.x)), outs)
        elif kind == "RoutedProjectedMatmulClaim":
            full = claim.Y
            ws = [self.mag.get(w) for w in claim.W if w in self.mag]
            wmax = None if any(w is None for w in ws) else max(ws, default=0)
            b = both(m(claim.X), wmax)
            self._bounded(claim, full, None if b is None else claim.K * b, outs)
        elif kind == "FreivaldsCombineClaim":
            if not claim.y_committed:
                full = claim.y
                xs = [m(v) for v in claim.xs]
                xmax = None if any(x is None for x in xs) else max(xs, default=0)
                b = both(m(claim.m), xmax)
                self._bounded(claim, full, None if b is None else claim.E * b, outs)
        elif kind == "AddClaim":
            if claim.b is not None:
                full = claim.c
                a, b = m(claim.a), m(claim.b)
                self._bounded(claim, full, None if a is None or b is None else a + b, outs)
        elif kind == "WordExtractionClaim" and claim.x in outs:       # the combine direction
            full = claim.x
            ws = [m(w) for w in claim.words]
            b = None if any(w is None for w in ws) else \
                sum(min(c, P - c) * w for c, w in zip(claim.coeffs, ws)) + min(claim.shift, P - claim.shift)
            self._bounded(claim, full, b, outs)
        elif kind == "RescaleClaim":
            full = claim.x
            src = m(claim.x_full)
            self._bounded(claim, full, None if src is None else (src >> claim.rescale_bits) + 1, outs)
        elif kind not in ESTABLISHED_BY_CONSTRAINTS and kind not in ESTABLISHED_BY_BOUND:
            self.unknown_types.add(kind)
        # every other output: trusted when every input is (and, for the bounded
        # claims, when the bounded output is)
        if kind == "RoutedProjectedMatmulClaim":   # the shards it never resolved play no part
            input_vars = [claim.X, claim.M] + [w for w in claim.W if w in self.mag]
        ins = [m(v) for v in input_vars]
        trusted = all(x is not None for x in ins) and (full is None or self.mag.get(full) is not None)
        for v, t in outs.items():
            if v is full:
                continue
            self.mag[v] = maxabs(t) if trusted else None
        for v, t in outs.items():
            name = self.names.get(v)
            if name is not None:
                self.trace[name] = extremes(t) if self.mag.get(v) is not None else None

    def record(self):
        by_layer = defaultdict(lambda: dict(bounded=0, max_bits=0.0, unestablished=0))
        for r in self.bounds:
            row = by_layer[str(r["layer"])]
            row["bounded"] += 1
            row["max_bits"] = max(row["max_bits"], r["observed_bits"] or 0.0)
            row["unestablished"] += not r["ok"]
        return dict(
            witness="not kept: the pass frees each value after its last use; a proof runs its own pass",
            field=dict(ok=not self.unestablished and not self.unknown_types, limit_bits=bits(P_HALF),
                       by_layer=dict(by_layer), unestablished=self.unestablished,
                       unknown_types=sorted(self.unknown_types), rows=self.bounds),
            lookups=dict(ok=not self.outside, outside=self.outside, tables=dict(self.lookups)),
            trace=self.trace)


def run(tape, keep, *, layer_of=None, names=None):
    """The run-only pass over a lazy tape. Returns ({Variable: its values as
    signed int64, on the device} for `keep`, the record)."""
    ro = RunOnly(tape, layer_of, names)
    t0 = time.time()
    with ro.hooks():
        live = tape.run_engine_pass(free_intermediates=True, keep=set(keep), observer=ro)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    rec = ro.record()
    rec["seconds"] = time.time() - t0
    rec["ok"] = rec["field"]["ok"] and rec["lookups"]["ok"]
    out = {v: lift(live[v]) for v in keep}
    del live
    return out, rec
