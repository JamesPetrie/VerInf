"""A toy transformer layer in EXACT integer semantics, emitting a proof trace.

Structurally faithful to what VerInf proves, at a size that runs in Python:
fixed-point values, a raw accumulator followed by a deterministic rescale with a
range-checked remainder, and every nonlinearity as a table lookup. The relation
TYPES and their proportions are the real ones, which is what the cost model
needs -- counts scale with (S, d, d_ff, E), so the model can be validated at toy
size and evaluated at 400B geometry.

Layer (pre-norm, one head group, top-1 MoE FFN):

    h      = x
    xn     = RMSNorm(x)                 sum of squares -> rescale -> isqrt lookup
    q,k,v  = xn Wq, xn Wk, xn Wv        matmuls (proved by the seam + sumcheck)
    q,k    = RoPE(q), RoPE(k)           affine mixing by public cos/sin
    s      = q k^T                      matmul, causal mask
    e      = exp[bracket(s)]            range decomposition + exp lookup
    p      = rescale(e * recip[sum e])  reciprocal lookup + rescale
    a      = p v                        matmul
    o      = a Wo                       matmul
    h      = h + o                      affine
    g,u    = RMSNorm(h) Wg, ... Wu      matmuls (per routed expert if E > 1)
    f      = silu[g] * u                lookup + hadamard
    y      = h + f Wd                   matmul + affine

Everything the layer does lands in one of five buckets, and `LayerTrace` holds
them: matmuls, gates (hadamard/affine/rescale/booleanity), lookups, the routing
records for MoE, and the committed input/output states.

`check_trace` re-derives every relation from the witness independently of how it
was produced, so a bug in the emitter shows up as a failed gate rather than a
proof of the wrong statement.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from prover.protocol import P as FIELD_P

from . import relations as rel
from .counters import phase
from .profile import stage

# ── tables ───────────────────────────────────────────────────────────────────
# Small unary tables over an explicit bounded domain. A query is brought into
# the domain by a RANGE DECOMPOSITION (v = hi*size + lo with lo range-checked),
# not by masking bits -- the decomposition is itself a proved relation, so the
# lookup domain is honest rather than assumed.


@dataclass(frozen=True)
class Table:
    table_id: int
    name: str
    rows: Tuple[Tuple[int, int], ...]        # (input, output) pairs

    @property
    def size(self) -> int:
        return len(self.rows)

    def lookup(self, x: int) -> int:
        return self.rows[x][1]

    def tuples(self) -> List[Tuple[int, int]]:
        return [tuple(r) for r in self.rows]


def build_tables(bits: int, scale: int) -> Dict[str, Table]:
    """Deterministic integer stand-ins with the right SHAPE (unary, bounded
    domain, one output per input). Their numeric content is not Llama's; the
    proof cost depends on the shape, not the values."""
    n = 1 << bits
    exp_rows = tuple((x, (x * x + 1) % scale) for x in range(n))
    silu_rows = tuple((x, (x * (x + 1) // 2) % scale) for x in range(n))
    isqrt_rows = tuple((x, (scale // (1 + int(x ** 0.5))) % scale) for x in range(n))
    recip_rows = tuple((x, (scale // (x + 1)) % scale) for x in range(n))
    range_rows = tuple((x, 0) for x in range(n))          # membership-only table
    return {
        "exp": Table(1, "exp", exp_rows),
        "silu": Table(2, "silu", silu_rows),
        "isqrt": Table(3, "isqrt", isqrt_rows),
        "recip": Table(4, "recip", recip_rows),
        "range": Table(5, "range", range_rows),
    }


@dataclass
class ToyConfig:
    S: int = 4                # sequence length
    d: int = 8                # model width
    d_ff: int = 16            # FFN width
    E: int = 1                # experts (1 = dense FFN, >1 = top-1 MoE)
    table_bits: int = 6
    scale_bits: int = 6       # fixed-point scale = 2^scale_bits

    @property
    def scale(self) -> int:
        return 1 << self.scale_bits

    @property
    def table_size(self) -> int:
        return 1 << self.table_bits


@dataclass
class Matmul:
    name: str
    X: List[List[int]]        # [rows][n_in]
    W: List[List[int]]        # [n_out][n_in]  (output-major, as enrolled)
    Y: List[List[int]]        # [rows][n_out]


@dataclass
class MoEMatmul:
    """A routed expert contraction with the route HIDDEN.

    The per-token form this replaces (one named matmul per token against that
    token's expert) is cheaper, but it publishes the route: the trace says which
    expert each token used. Doc §5 hides it -- the tokens are sorted by expert
    behind a permutation argument, segmented by delimiters, and the whole thing
    collapses to one scalar identity. What that buys is privacy of the routing;
    what it costs is the sort/segment machinery in moe.py."""
    name: str
    route: List[int]                       # SECRET: proved, never published
    X: List[List[int]]                     # [S][d_in]
    W: List[List[List[int]]]               # [E][d_in][d_out]
    Y: List[List[int]]                     # [S][d_out]


@dataclass
class LookupUse:
    table: Table
    queries: List[Tuple[int, int]]


@dataclass
class LayerTrace:
    cfg: ToyConfig
    x_in: List[List[int]]
    y_out: List[List[int]]
    matmuls: List[Matmul] = field(default_factory=list)
    moe: List[MoEMatmul] = field(default_factory=list)
    gates: List[rel.Gate] = field(default_factory=list)
    lookups: List[LookupUse] = field(default_factory=list)
    route: List[int] = field(default_factory=list)
    expert_weights: List[List[List[int]]] = field(default_factory=list)
    # How close the run came to the 2^63 wall the tensor path needs (0 = the
    # Python path, which has no wall because its ints are arbitrary precision).
    # Reported rather than assumed: see `_TBuilder._guard_prod`.
    peak_value: int = 0
    peak_bound: int = 0

    @property
    def headroom_bits(self) -> float:
        """log2(2^63 / peak_bound): spare bits before the tensor path would stop
        agreeing with the reference. Negative is impossible -- the run raises."""
        if not self.peak_bound:
            return float("inf")
        return 63 - self.peak_bound.bit_length()

    def counts(self) -> Dict[str, int]:
        return {
            "matmuls": len(self.matmuls),
            "moe_nodes": len(self.moe),
            "moe_cells": sum(len(m.W) * len(m.W[0]) * len(m.W[0][0]) for m in self.moe),
            "moe_route_len": sum(len(m.route) for m in self.moe),
            "matmul_cells": sum(len(m.X) * len(m.W) * len(m.W[0]) for m in self.matmuls),
            "gates": len(self.gates),
            "gate_slots": sum(g.size for g in self.gates),
            "lookup_uses": len(self.lookups),
            "lookup_queries": sum(len(l.queries) for l in self.lookups),
            "lookup_table_rows": sum(l.table.size for l in self.lookups),
        }


class _Builder:
    """Accumulates the trace while the layer is computed, so every emitted gate
    is produced by the same code path that produced the value."""

    def __init__(self, cfg: ToyConfig, tables: Dict[str, Table]):
        self.cfg = cfg
        self.tables = tables
        self.gates: List[rel.Gate] = []
        self.lookups: Dict[str, List[Tuple[int, int]]] = {k: [] for k in tables}
        self.matmuls: List[Matmul] = []
        self.moe: List[MoEMatmul] = []
        self._n = 0

    def _id(self, kind: str) -> str:
        self._n += 1
        return f"{kind}#{self._n}"

    # -- primitives ----------------------------------------------------------
    def rescale(self, raws: Sequence[int]) -> List[int]:
        """raw = scale*q + r with 0 <= r < scale, r range-checked by lookup.
        This is the deterministic rescale of VerInf's integer semantics; it is
        the reason a multiply is never just a multiply."""
        s = self.cfg.scale
        qs, rs = [], []
        for raw in raws:
            qs.append(raw // s)
            rs.append(raw % s)
        self.gates.append(rel.rescale(self._id("rescale"), list(raws), qs, rs, s))
        self._range_check(rs, bound=s)
        return qs

    def _range_check(self, vals: Sequence[int], bound: int) -> None:
        """Membership in [0, bound) via the range table. bound must not exceed
        the table domain -- the caller decomposes first if it does."""
        t = self.tables["range"]
        if bound > t.size:
            raise ValueError(f"range bound {bound} exceeds table {t.size}")
        self.lookups["range"].extend((v, 0) for v in vals)

    def bracket(self, vals: Sequence[int]) -> List[int]:
        """Bring a value into the table domain: v = hi*size + lo, lo in range.
        The decomposition is a proved relation, so the lookup's domain is not
        an assumption."""
        n = self.cfg.table_size
        his, los = [], []
        for v in vals:
            his.append(v // n)
            los.append(v % n)
        self.gates.append(rel.rescale(self._id("bracket"), list(vals), his, los, n))
        self._range_check(los, bound=n)
        return los

    def lookup(self, name: str, xs: Sequence[int]) -> List[int]:
        t = self.tables[name]
        ys = [t.lookup(x) for x in xs]
        self.lookups[name].extend(zip(xs, ys))
        return ys

    def mul_rescaled(self, a: Sequence[int], b: Sequence[int], tag: str) -> List[int]:
        raw = [(x * y) for x, y in zip(a, b)]
        self.gates.append(rel.hadamard(self._id(f"had_{tag}"),
                                       list(a), list(b), [r % FIELD_P for r in raw]))
        return self.rescale(raw)

    def add(self, a: Sequence[int], b: Sequence[int], tag: str) -> List[int]:
        y = [(x + z) for x, z in zip(a, b)]
        self.gates.append(rel.affine(self._id(f"add_{tag}"), [1, 1], [list(a), list(b)],
                                     [v % FIELD_P for v in y]))
        return y

    def matmul(self, name: str, X: List[List[int]], W: List[List[int]]) -> List[List[int]]:
        """Y[t][i] = rescale(sum_j X[t][j] W[i][j]). The contraction itself is
        proved by the projection seam + sumcheck (layer.py), not by a gate."""
        n_out = len(W)
        raws = [[sum(X[t][j] * W[i][j] for j in range(len(W[0]))) for i in range(n_out)]
                for t in range(len(X))]
        Yraw = [[v % FIELD_P for v in row] for row in raws]
        self.matmuls.append(Matmul(name, [list(r) for r in X], [list(r) for r in W], Yraw))
        flat = [v for row in raws for v in row]
        q = self.rescale(flat)
        return [q[i * n_out:(i + 1) * n_out] for i in range(len(X))]


    def moe_matmul(self, name: str, X: List[List[int]], W3, route: List[int]
                   ) -> List[List[int]]:
        """Y[t][i] = rescale(sum_j X[t][j] W[route[t]][j][i]) with the route kept
        secret. One node for all tokens; moe.py proves it."""
        # W3[e] is OUTPUT-MAJOR, like every other weight here: W3[e][i][j] is the
        # weight of input j for output i. Same orientation as enrolment expects.
        d_out, d_in = len(W3[0]), len(W3[0][0])
        raws = [[sum(X[t][j] * W3[route[t]][i][j] for j in range(d_in))
                 for i in range(d_out)] for t in range(len(X))]
        Yraw = [[v % FIELD_P for v in row] for row in raws]
        self.moe.append(MoEMatmul(name, list(route), [list(r) for r in X],
                                  [[list(c) for c in e] for e in W3], Yraw))
        flat = [v for row in raws for v in row]
        q = self.rescale(flat)
        return [q[i * d_out:(i + 1) * d_out] for i in range(len(X))]


def _rand_matrix(rng, rows: int, cols: int, bound: int) -> List[List[int]]:
    return [[rng.randrange(bound) for _ in range(cols)] for _ in range(rows)]


@dataclass
class LayerWeights:
    """Everything the layer reads that is not computed by it.

    Split out of `forward` for one reason: the tensor path and the Python path
    must be fed IDENTICAL numbers before their outputs can be compared bit for
    bit. Drawing them twice from a seeded RNG would work only as long as the two
    implementations consumed the stream in exactly the same order, which is
    precisely the kind of assumption that turns a validation into a formality.

    Field order is also the RNG consumption order of the original `forward`, so
    `draw()` reproduces every trace that existed before this was extracted."""
    x: List[List[int]]
    Wq: List[List[int]]
    Wk: List[List[int]]
    Wv: List[List[int]]
    Wo: List[List[int]]
    cos: List[int]
    sin: List[int]
    experts: List[Tuple[List[List[int]], List[List[int]], List[List[int]]]]
    route: List[int]

    @classmethod
    def draw(cls, cfg: ToyConfig, rng,
             x_in: Optional[List[List[int]]] = None) -> "LayerWeights":
        bound, S, d, d_ff, E = cfg.table_size, cfg.S, cfg.d, cfg.d_ff, cfg.E
        x = x_in or _rand_matrix(rng, S, d, bound)
        Wq = _rand_matrix(rng, d, d, bound)
        Wk = _rand_matrix(rng, d, d, bound)
        Wv = _rand_matrix(rng, d, d, bound)
        Wo = _rand_matrix(rng, d, d, bound)
        cos = [rng.randrange(bound) for _ in range(d)]
        sin = [rng.randrange(bound) for _ in range(d)]
        experts = [(_rand_matrix(rng, d_ff, d, bound),       # gate
                    _rand_matrix(rng, d_ff, d, bound),       # up
                    _rand_matrix(rng, d, d_ff, bound))       # down
                   for _ in range(max(E, 1))]
        route = [rng.randrange(max(E, 1)) for _ in range(S)]
        return cls(x, Wq, Wk, Wv, Wo, cos, sin, experts, route)

    @classmethod
    def draw_tensor(cls, cfg: ToyConfig, seed: int, device: str = "cuda"
                    ) -> "LayerWeights":
        """The same shapes, drawn on the device. Not the same NUMBERS as
        `draw` -- a different generator -- which is why the validator draws
        here and converts DOWN with `to_lists`, instead of seeding both.

        This is not a detail: after the forward pass moved to tensors, drawing
        the weights in Python became the largest single cost in the run
        (19.6 s against 4.5 s of forward at d=1024), so a benchmark that kept
        `draw` would have been measuring `random.randrange`."""
        import torch
        g = torch.Generator(device=device).manual_seed(seed)
        bound, S, d, d_ff, E = cfg.table_size, cfg.S, cfg.d, cfg.d_ff, cfg.E

        def R(*shape):
            return torch.randint(0, bound, shape, generator=g,
                                 dtype=torch.int64, device=device)

        experts = [(R(d_ff, d), R(d_ff, d), R(d, d_ff)) for _ in range(max(E, 1))]
        route = torch.randint(0, max(E, 1), (S,), generator=g,
                              dtype=torch.int64, device=device).cpu().tolist()
        return cls(R(S, d), R(d, d), R(d, d), R(d, d), R(d, d), R(d), R(d),
                   experts, route)

    def to_lists(self) -> "LayerWeights":
        """Materialise on the host, so the reference path can be fed exactly the
        numbers the tensor path was given."""
        def L(t):
            return t.cpu().tolist() if hasattr(t, "shape") else t

        return LayerWeights(
            L(self.x), L(self.Wq), L(self.Wk), L(self.Wv), L(self.Wo),
            L(self.cos), L(self.sin),
            [(L(a), L(b), L(c)) for a, b, c in self.experts], list(self.route))


def forward(cfg: ToyConfig, rng, x_in: Optional[List[List[int]]] = None,
            weights: Optional[LayerWeights] = None) -> LayerTrace:
    """Compute one layer and emit its trace."""
    tables = build_tables(cfg.table_bits, cfg.scale)
    b = _Builder(cfg, tables)
    S, d, d_ff, E = cfg.S, cfg.d, cfg.d_ff, cfg.E
    w = weights if weights is not None else LayerWeights.draw(cfg, rng, x_in)

    with phase("semantics.forward"), stage("semantics.forward"):
        x = w.x
        Wq, Wk, Wv, Wo = w.Wq, w.Wk, w.Wv, w.Wo

        # ---- RMSNorm ------------------------------------------------------
        ss = [sum(v * v for v in row) for row in x]
        b.gates.append(rel.hadamard("rms_sq", [v for row in x for v in row],
                                    [v for row in x for v in row],
                                    [(v * v) % FIELD_P for row in x for v in row]))
        ss_b = b.bracket([v % FIELD_P for v in ss])
        inv_rms = b.lookup("isqrt", ss_b)
        xn = []
        for t in range(S):
            xn.append(b.mul_rescaled(x[t], [inv_rms[t]] * d, "rms"))

        # ---- QKV ----------------------------------------------------------
        q = b.matmul("Wq", xn, Wq)
        k = b.matmul("Wk", xn, Wk)
        v = b.matmul("Wv", xn, Wv)

        # ---- RoPE: affine mixing by public constants ----------------------
        cos, sin = w.cos, w.sin
        for mat, tag in ((q, "q"), (k, "k")):
            for t in range(S):
                rot = [(mat[t][j] * cos[j] + mat[t][(j + 1) % d] * sin[j])
                       for j in range(d)]
                b.gates.append(rel.affine(f"rope_{tag}_{t}", [1, 1],
                                          [[(mat[t][j] * cos[j]) % FIELD_P for j in range(d)],
                                           [(mat[t][(j + 1) % d] * sin[j]) % FIELD_P
                                            for j in range(d)]],
                                          [r % FIELD_P for r in rot]))
                mat[t] = b.rescale(rot)

        # ---- attention scores, causal ------------------------------------
        scores = [[sum(q[t][j] * k[i][j] for j in range(d)) if i <= t else 0
                   for i in range(S)] for t in range(S)]
        mask = [1 if i <= t else 0 for t in range(S) for i in range(S)]
        b.gates.append(rel.booleanity("causal_mask", mask))
        flat_scores = [v % FIELD_P for row in scores for v in row]
        sb = b.bracket(flat_scores)
        e = b.lookup("exp", sb)
        e = [[e[t * S + i] for i in range(S)] for t in range(S)]

        # ---- softmax normalisation ---------------------------------------
        sums = [sum(row) % FIELD_P for row in e]
        sb2 = b.bracket(sums)
        recip = b.lookup("recip", sb2)
        p = []
        for t in range(S):
            p.append(b.mul_rescaled(e[t], [recip[t]] * S, "softmax"))

        # ---- attention output + projection + residual --------------------
        a = b.matmul("attn_pv", p, [[v[i][j] for i in range(S)] for j in range(d)])
        o = b.matmul("Wo", a, Wo)
        h = [b.add(x[t], o[t], "res1") for t in range(S)]

        # ---- FFN (dense or top-1 MoE) ------------------------------------
        experts, route = w.experts, w.route

        Wg3 = [e[0] for e in experts]
        Wu3 = [e[1] for e in experts]
        Wd3 = [e[2] for e in experts]
        g_all = b.moe_matmul("moe_gate", h, Wg3, route)
        u_all = b.moe_matmul("moe_up", h, Wu3, route)
        f_all = []
        for t in range(S):
            gb = b.bracket(g_all[t])
            sg = b.lookup("silu", gb)
            f_all.append(b.mul_rescaled(sg, u_all[t], "swiglu"))
        d_all = b.moe_matmul("moe_down", f_all, Wd3, route)
        y = [b.add(h[t], d_all[t], "res2") for t in range(S)]

    lookups = [LookupUse(tables[name], qs) for name, qs in b.lookups.items() if qs]
    return LayerTrace(cfg=cfg, x_in=x, y_out=[[v % FIELD_P for v in row] for row in y],
                      matmuls=b.matmuls, moe=b.moe, gates=b.gates, lookups=lookups,
                      route=route, expert_weights=[e[0] for e in experts])


# ── the same layer, on device tensors ────────────────────────────────────────
# WHY a second implementation exists, and what keeps it honest.
#
# The Python path above is the reference: it is obviously the semantics, and
# `check_trace` re-derives every relation from what it produced. It is also the
# reason a production-width layer cannot be computed here at all -- at
# d=5120, S=1000 the forward pass is ~2e11 multiply-accumulates in the
# interpreter, and every value it produces is a Python int that later pays a
# host->device transfer.
#
# The tensor path below emits the SAME trace: same node ids, same gate order,
# same values -- so the two produce byte-identical proofs, which is the gate
# `bench/validate_semantics.py` enforces before this path is used for anything.
#
# REPRESENTATION. Values are carried as int64 holding the TRUE (unreduced)
# integer, not a field residue. That is what makes the two paths comparable: the
# Python path divides raw accumulators by the scale (`raw // s`), which is only
# the same operation in the field while the true value has not wrapped. torch
# has no bitwise ops or comparisons on uint64, so int64 is also the only usable
# dtype; the uint64 view is taken solely at the boundary of the Goldilocks
# kernels, which is a reinterpretation, not a copy.
#
# The condition for the two paths to agree is therefore `true value < 2^63`, and
# it is CHECKED at every node rather than assumed -- see `_guard_*`. When it
# fails the run stops and names the node. It is a real limit of the toy
# semantics, not of the protocol: values grow by a factor of ~n_in per matmul
# because the rescale divides by `scale` (64) while the accumulator grows by
# `n_in * scale`. See `analysis/layergkr-cost-model.md` §8.
_TRUE_MAX = 1 << 63


class RangeOverflow(RuntimeError):
    """A true value would not fit below 2^63, so the tensor path and the Python
    path would stop agreeing. Raised instead of silently diverging."""


def _torch():
    import torch
    from prover import cuda_primitives as cp
    return torch, cp


class _TBuilder:
    """`_Builder` on device tensors. Method for method, id for id."""

    def __init__(self, cfg: ToyConfig, tables: Dict[str, Table], device: str):
        torch, _ = _torch()
        self.cfg = cfg
        # Largest true value and largest bound seen anywhere in the layer. A
        # guard that never fires proves nothing, so the run REPORTS its headroom
        # instead of silently succeeding: `peak_bound` against 2^63 is how close
        # the configuration came to the wall.
        self.peak_value = 0
        self.peak_bound = 0
        self.tables = tables
        self.device = device
        self.gates: List[rel.Gate] = []
        # per table, the pieces of the (n, 2) query log, concatenated at the end
        self.lookups: Dict[str, List] = {k: [] for k in tables}
        self.out: Dict[str, "object"] = {}
        for name, t in tables.items():
            if any(r[0] != i for i, r in enumerate(t.rows)):
                raise ValueError(f"table {name} is not indexed by its input")
            self.out[name] = torch.tensor([r[1] for r in t.rows],
                                          dtype=torch.int64, device=device)
        self.matmuls: List[Matmul] = []
        self.moe: List[MoEMatmul] = []
        self._n = 0

    def _id(self, kind: str) -> str:
        self._n += 1
        return f"{kind}#{self._n}"

    # -- the range discipline ------------------------------------------------
    def _hi(self, t) -> int:
        """Largest true value in a tensor, as a Python int. One device sync;
        there are a few hundred per layer against millions of elements, and the
        profile puts them at 0.025 s of an 0.081 s layer at d=1024."""
        v = int(t.max().item()) if t.numel() else 0
        self.peak_value = max(self.peak_value, v)
        return v

    def _guard(self, name: str, t) -> None:
        """A wrapped value shows up as a negative int64. Cheaper than comparing
        against 2^63, which torch cannot do on uint64 at all."""
        if t.numel() and int(t.min().item()) < 0:
            raise RangeOverflow(
                f"{name}: a value wrapped past 2^63. The toy semantics grow by "
                f"~n_in per matmul; reduce d/d_ff, or raise scale_bits (which "
                f"needs table_bits >= scale_bits).")

    def _guard_prod(self, name: str, a_hi: int, b_hi: int, terms: int = 1) -> None:
        bound = a_hi * b_hi * terms
        self.peak_bound = max(self.peak_bound, bound)
        if bound >= _TRUE_MAX:
            raise RangeOverflow(
                f"{name}: {a_hi} * {b_hi} * {terms} = {bound} >= 2^63, so the "
                f"true accumulator would not fit and the tensor path would stop "
                f"agreeing with the reference. This is the toy semantics' value "
                f"growth, not a protocol limit.")

    # -- primitives ----------------------------------------------------------
    def rescale(self, raws):
        """raw = scale*q + r. scale is a power of two, so the division and the
        remainder are a shift and a mask -- exact, and the same integers the
        Python path computes with `//` and `%`."""
        self._guard("rescale input", raws)
        s = self.cfg.scale
        qs = raws >> self.cfg.scale_bits
        rs_ = raws & (s - 1)
        self.gates.append(rel.rescale(self._id("rescale"), raws, qs, rs_, s))
        self._range_check(rs_, bound=s)
        return qs

    def _range_check(self, vals, bound: int) -> None:
        torch, _ = _torch()
        t = self.tables["range"]
        if bound > t.size:
            raise ValueError(f"range bound {bound} exceeds table {t.size}")
        self.lookups["range"].append(
            torch.stack([vals, torch.zeros_like(vals)], dim=1))

    def bracket(self, vals):
        self._guard("bracket input", vals)
        n = self.cfg.table_size
        his = vals >> self.cfg.table_bits
        los = vals & (n - 1)
        self.gates.append(rel.rescale(self._id("bracket"), vals, his, los, n))
        self._range_check(los, bound=n)
        return los

    def lookup(self, name: str, xs):
        torch, _ = _torch()
        ys = self.out[name][xs]
        self.lookups[name].append(torch.stack([xs, ys], dim=1))
        return ys

    def mul_rescaled(self, a, b, tag: str):
        self._guard_prod(f"had_{tag}", self._hi(a), self._hi(b))
        raw = a * b
        self.gates.append(rel.hadamard(self._id(f"had_{tag}"), a, b, raw))
        return self.rescale(raw)

    def add(self, a, b, tag: str):
        y = a + b
        self._guard(f"add_{tag}", y)
        self.gates.append(rel.affine(self._id(f"add_{tag}"), [1, 1], [a, b], y))
        return y

    def _contract(self, name: str, X, W):
        """raw[t][i] = sum_j X[t][j] * W[i][j], exactly. `gl_matmul` reduces mod
        P, which equals the true sum only while the true sum is below P; the
        guard above is the stricter 2^63 bound, so the equality is checked, not
        hoped for."""
        torch, cp = _torch()
        self._guard_prod(name, self._hi(X), self._hi(W), int(W.shape[1]))
        out = cp.gl_matmul(X.view(torch.uint64).contiguous(),
                           W.t().contiguous().view(torch.uint64)).view(torch.int64)
        self._guard(name, out)
        return out

    def matmul(self, name: str, X, W):
        n_out = int(W.shape[0])
        raws = self._contract(name, X, W)
        self.matmuls.append(Matmul(name, X, W, raws))
        q = self.rescale(raws.reshape(-1))
        return q.reshape(int(X.shape[0]), n_out)

    def moe_matmul(self, name: str, X, W3, route: List[int]):
        """One node for all tokens, with the route secret. The contraction runs
        once per EXPERT over that expert's token subset -- E kernel launches
        rather than S, and no (S, d_out, d_in) gather, which at production width
        would be the largest tensor in the layer by two orders of magnitude."""
        torch, _ = _torch()
        E, d_out = int(W3.shape[0]), int(W3.shape[1])
        S_m = int(X.shape[0])
        raws = torch.zeros((S_m, d_out), dtype=torch.int64, device=X.device)
        route_t = torch.tensor(route, dtype=torch.int64, device=X.device)
        for e in range(E):
            idx = (route_t == e).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            raws[idx] = self._contract(f"{name}[{e}]", X[idx].contiguous(), W3[e])
        self.moe.append(MoEMatmul(name, list(route), X, W3, raws))
        q = self.rescale(raws.reshape(-1))
        return q.reshape(S_m, d_out)


def forward_tensor(cfg: ToyConfig, rng, x_in: Optional[List[List[int]]] = None,
                   weights: Optional[LayerWeights] = None,
                   device: str = "cuda") -> LayerTrace:
    """The layer of `forward`, computed on device tensors, emitting the same
    trace. Every step below is deliberately in the same order as its Python
    counterpart -- gates are batched by index, so a reordering would change the
    proof even though it would not change the semantics."""
    torch, _ = _torch()
    tables = build_tables(cfg.table_bits, cfg.scale)
    b = _TBuilder(cfg, tables, device)
    S, d, E = cfg.S, cfg.d, cfg.E
    if weights is not None:
        w = weights
    elif x_in is None:
        w = LayerWeights.draw_tensor(cfg, rng.randrange(1 << 31) if rng else 0,
                                     device)
    else:
        w = LayerWeights.draw(cfg, rng, x_in)

    def T(m):
        if hasattr(m, "shape"):
            return m.to(device=device, dtype=torch.int64)
        return torch.tensor(m, dtype=torch.int64, device=device)

    with phase("semantics.forward"), stage("semantics.forward"):
        x = T(w.x)
        Wq, Wk, Wv, Wo = T(w.Wq), T(w.Wk), T(w.Wv), T(w.Wo)

        # ---- RMSNorm ------------------------------------------------------
        xf = x.reshape(-1)
        b._guard_prod("rms_sq", b._hi(x), b._hi(x), d)
        b.gates.append(rel.hadamard("rms_sq", xf, xf, xf * xf))
        ss = (x * x).sum(dim=1)
        b._guard("rms sum of squares", ss)
        ss_b = b.bracket(ss)
        inv_rms = b.lookup("isqrt", ss_b)
        xn = torch.stack([b.mul_rescaled(x[t], inv_rms[t:t + 1].expand(d), "rms")
                          for t in range(S)])

        # ---- QKV ----------------------------------------------------------
        q = b.matmul("Wq", xn, Wq)
        k = b.matmul("Wk", xn, Wk)
        v = b.matmul("Wv", xn, Wv)

        # ---- RoPE: affine mixing by public constants ----------------------
        cos, sin = T(w.cos), T(w.sin)
        rows = {"q": [q[t] for t in range(S)], "k": [k[t] for t in range(S)]}
        for tag in ("q", "k"):
            mat = rows[tag]
            for t in range(S):
                b._guard_prod(f"rope_{tag}", b._hi(mat[t]), max(b._hi(cos), b._hi(sin)), 2)
                lhs = mat[t] * cos
                rhs = torch.roll(mat[t], -1) * sin
                rot = lhs + rhs
                b.gates.append(rel.affine(f"rope_{tag}_{t}", [1, 1], [lhs, rhs], rot))
                mat[t] = b.rescale(rot)
        q = torch.stack(rows["q"])
        k = torch.stack(rows["k"])

        # ---- attention scores, causal ------------------------------------
        b._guard_prod("scores", b._hi(q), b._hi(k), d)
        scores = b._contract("scores", q, k)
        causal = torch.tril(torch.ones((S, S), dtype=torch.int64, device=device))
        scores = scores * causal
        b.gates.append(rel.booleanity("causal_mask", causal.reshape(-1)))
        sb = b.bracket(scores.reshape(-1))
        e = b.lookup("exp", sb).reshape(S, S)

        # ---- softmax normalisation ---------------------------------------
        sums = e.sum(dim=1)
        b._guard("softmax denominator", sums)
        sb2 = b.bracket(sums)
        recip = b.lookup("recip", sb2)
        p = torch.stack([b.mul_rescaled(e[t], recip[t:t + 1].expand(S), "softmax")
                         for t in range(S)])

        # ---- attention output + projection + residual --------------------
        a = b.matmul("attn_pv", p, v.t().contiguous())
        o = b.matmul("Wo", a, Wo)
        h = torch.stack([b.add(x[t], o[t], "res1") for t in range(S)])

        # ---- FFN (dense or top-1 MoE) ------------------------------------
        experts, route = w.experts, w.route
        Wg3 = torch.stack([T(ex[0]) for ex in experts])
        Wu3 = torch.stack([T(ex[1]) for ex in experts])
        Wd3 = torch.stack([T(ex[2]) for ex in experts])
        g_all = b.moe_matmul("moe_gate", h, Wg3, route)
        u_all = b.moe_matmul("moe_up", h, Wu3, route)
        f_rows = []
        for t in range(S):
            gb = b.bracket(g_all[t])
            sg = b.lookup("silu", gb)
            f_rows.append(b.mul_rescaled(sg, u_all[t], "swiglu"))
        f_all = torch.stack(f_rows)
        d_all = b.moe_matmul("moe_down", f_all, Wd3, route)
        y = torch.stack([b.add(h[t], d_all[t], "res2") for t in range(S)])

    lookups = [LookupUse(tables[name], torch.cat(parts))
               for name, parts in b.lookups.items() if parts]
    return LayerTrace(cfg=cfg, x_in=x, y_out=y, matmuls=b.matmuls, moe=b.moe,
                      gates=b.gates, lookups=lookups, route=route,
                      expert_weights=Wg3, peak_value=b.peak_value,
                      peak_bound=b.peak_bound)


def to_python(trace: LayerTrace) -> LayerTrace:
    """Materialise a tensor trace as the Python-object trace the reference path
    produces. Used by the validator and by consumers that have not been
    tensorised yet; it is the SLOW direction and exists to be compared against,
    not to be run at scale."""
    def L(t):
        return t.cpu().tolist() if hasattr(t, "shape") else t

    return LayerTrace(
        cfg=trace.cfg, x_in=L(trace.x_in), y_out=L(trace.y_out),
        matmuls=[Matmul(m.name, L(m.X), L(m.W), L(m.Y)) for m in trace.matmuls],
        moe=[MoEMatmul(m.name, list(m.route), L(m.X), L(m.W), L(m.Y))
             for m in trace.moe],
        gates=[rel.Gate(g.kind, g.node_id,
                        [(c, [L(f) for f in fs]) for c, fs in g.terms])
               for g in trace.gates],
        lookups=[LookupUse(u.table, [tuple(r) for r in L(u.queries)])
                 for u in trace.lookups],
        route=list(trace.route), expert_weights=L(trace.expert_weights),
        peak_value=trace.peak_value, peak_bound=trace.peak_bound)


def check_trace(trace: LayerTrace) -> Tuple[bool, str]:
    """Re-derive every relation from the witness. Independent of the emitter:
    a gate that does not vanish, a matmul whose Y is wrong, or a lookup whose
    output is not the table's, is reported here.

    Deliberately slow and deliberately Python: it is the arbiter, so it shares
    no arithmetic with either forward path. A tensor trace is materialised first
    rather than indexed elementwise -- 0-dim tensors are hashable by identity, so
    the table membership test below would silently pass on anything."""
    if hasattr(trace.x_in, "shape"):
        trace = to_python(trace)
    for g in trace.gates:
        size = g.size
        for idx in range(size):
            acc = 0
            for coeff, factors in g.terms:
                prod = coeff % FIELD_P
                for f in factors:
                    prod = prod * (f[idx] if idx < len(f) else 0) % FIELD_P
                acc = (acc + prod) % FIELD_P
            if acc != 0:
                return False, f"gate {g.node_id} ({g.kind}) does not vanish at {idx}"
    for m in trace.matmuls:
        for t in range(len(m.X)):
            for i in range(len(m.W)):
                want = sum(m.X[t][j] * m.W[i][j] for j in range(len(m.W[0]))) % FIELD_P
                if m.Y[t][i] != want:
                    return False, f"matmul {m.name} wrong at ({t},{i})"
    for m in trace.moe:
        d_out, d_in = len(m.W[0]), len(m.W[0][0])
        for t in range(len(m.X)):
            for i in range(d_out):
                want = sum(m.X[t][j] * m.W[m.route[t]][i][j]
                           for j in range(d_in)) % FIELD_P
                if m.Y[t][i] != want:
                    return False, f"moe {m.name} wrong at ({t},{i})"
    for lu in trace.lookups:
        rows = dict(lu.table.tuples())
        for x_, y_ in lu.queries:
            if rows.get(x_) != y_:
                return False, f"lookup {lu.table.name}: {x_} -> {y_} is not the table's"
    return True, "ok"
