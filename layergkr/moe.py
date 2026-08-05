"""MoE without the S*E*K selector tensor (doc §5).

The current prover materialises a private selector of size S*E*K per MoE layer.
Layer-GKR-LF replaces it with a hidden STABLE SORT plus a SEGMENTED SCAN, so the
expert contraction costs O(S*(2d+d_ff)) instead of O(S*E*(2d+d_ff)) -- the doc's
56.623B cells -> 442.368M segmented cells plus <492M permutation cells.

Three arguments, all implemented and gated here:

1. Permutation. The sorted records are committed before the challenges, then
   checked by the characteristic product
       prod_src (z - chi_beta(R)) == prod_dst (z - chi_beta(R'))
   with layer/kind/position/expert/coordinate all inside the fingerprint, so a
   forged multiset gives a nonzero difference polynomial (error <= n/p).

2. Stable order. Within a lane the key k = e*(S+1) + t must strictly increase.
   Strictness is what forbids duplicating or reordering a record; the lane is
   keyed by (kind, j) so the same (e,t) appearing under different coordinates is
   not a violation.

3. Segmentation. E+1 delimiters cut the sorted stream into per-expert segments.
   With d_q, a_q in {0,1}, d_q + a_q = 1:
       h_{q+1,j} = (1 - d_q) * (h_{q,j} + a_q * tau_{t'_q} * x'_{q,j})
   and a counter c pins which segment each row belongs to:
       c_0 = 0,  c_{q+1} = c_q + d_q,  c_final = E+1,
       d_q * (label_q - c_q) = 0,      a_q * (e'_q - (c_q - 1)) = 0.
   Consecutive delimiters emit a zero accumulator, so empty experts need no
   special case -- the doc's degenerate routes fall out of the same equations.

The payoff identity (§5.3): with A[e,j] = sum_{t: e_t = e} tau_t X[t,j] and
P[e,j] = sum_i rho_i W[e,j,i], the whole selected matmul collapses to one scalar

    sum_{t,i} tau_t rho_i Y[t,i]  ==  sum_{e,j} A[e,j] P[e,j].

`moe_identity_sides` computes both sides independently and the tests assert they
agree -- including for the all-one-expert and empty-expert routes.

This is a reference implementation in the field, not a prover: it builds the
witness and checks the relations directly. Turning each relation into a sumcheck
is the next step (see README, "what is not here").
"""
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from prover.protocol import P as FIELD_P

DELIMITER = -1          # sentinel expert id for a delimiter row


@dataclass(frozen=True)
class Record:
    """One (layer, kind, position, expert, coordinate, value) source record."""
    layer: int
    kind: int
    t: int
    e: int
    j: int
    x: int


def source_records(layer: int, kind: int, route: Sequence[int],
                   X: Sequence[Sequence[int]]) -> List[Record]:
    """Records in canonical token order: t ascending, then coordinate j."""
    return [Record(layer, kind, t, route[t], j, X[t][j] % FIELD_P)
            for t in range(len(route)) for j in range(len(X[0]))]


def stable_sorted_records(recs: Sequence[Record]) -> List[Record]:
    """Route order: by lane (kind, j), then expert, then position. Stability is
    the sort by t within an expert -- it is what makes the order canonical, so a
    prover cannot smuggle information into the permutation it chooses."""
    return sorted(recs, key=lambda r: (r.kind, r.j, r.e, r.t))


# ── 1. permutation ───────────────────────────────────────────────────────────
def fingerprint(r: Record, beta: int) -> int:
    """chi_beta(R): tuple compression. Every field of the record is inside, so
    two multisets that differ anywhere give different products."""
    acc = 0
    for v in (r.layer, r.kind, r.t, r.e, r.j, r.x):
        acc = (acc * beta + (v % FIELD_P)) % FIELD_P
    return acc


def characteristic_product(recs: Sequence[Record], beta: int, z: int) -> int:
    acc = 1
    for r in recs:
        acc = (acc * ((z - fingerprint(r, beta)) % FIELD_P)) % FIELD_P
    return acc


def permutation_ok(src: Sequence[Record], dst: Sequence[Record],
                   beta: int, z: int) -> bool:
    return (len(src) == len(dst)
            and characteristic_product(src, beta, z)
            == characteristic_product(dst, beta, z))


# ── 2. stable order ──────────────────────────────────────────────────────────
def strict_order_ok(dst: Sequence[Record], S: int) -> Tuple[bool, str]:
    """k = e*(S+1) + t must STRICTLY increase inside each (kind, j) lane."""
    last: Dict[Tuple[int, int], int] = {}
    for r in dst:
        lane = (r.kind, r.j)
        k = r.e * (S + 1) + r.t
        if lane in last and k <= last[lane]:
            return False, f"lane {lane}: key {k} not > {last[lane]}"
        last[lane] = k
    return True, "ok"


# ── 3. delimiters + segmented scan ───────────────────────────────────────────
@dataclass
class ScanRow:
    """One row of the segmented-scan trace, per lane."""
    is_delim: bool
    label: int          # delimiter label (delimiter rows only)
    e: int              # record's expert (token rows only)
    t: int
    x: int
    d: int              # delimiter flag
    a: int              # accumulate flag
    c_in: int           # counter before the row
    h_in: int           # accumulator before the row
    h_out: int          # accumulator after the row
    emitted: int        # value emitted at a delimiter (h_in), else 0


def build_lane(dst_lane: Sequence[Record], E: int, tau: Sequence[int]
               ) -> Tuple[List[ScanRow], List[int]]:
    """Interleave E+1 delimiters into one lane's sorted records and run the
    recurrence. Returns (trace, A_lane) where A_lane[e] is expert e's sum."""
    by_expert: Dict[int, List[Record]] = {}
    for r in dst_lane:
        by_expert.setdefault(r.e, []).append(r)

    trace: List[ScanRow] = []
    A = [0] * E
    h, c = 0, 0
    for label in range(E + 1):
        # delimiter: flush expert label-1 (label 0 just opens expert 0)
        trace.append(ScanRow(True, label, DELIMITER, -1, 0, 1, 0, c, h, 0, h))
        if label > 0:
            A[label - 1] = h
        h = 0
        c += 1
        if label < E:
            for r in by_expert.get(label, []):
                delta = (tau[r.t] * r.x) % FIELD_P
                h_new = (h + delta) % FIELD_P
                trace.append(ScanRow(False, -1, r.e, r.t, r.x, 0, 1, c, h, h_new, 0))
                h = h_new
    return trace, A


def scan_constraints_ok(trace: Sequence[ScanRow], E: int, tau: Sequence[int]
                        ) -> Tuple[bool, str]:
    """Check every relation of §5.2 on the trace, exactly as written."""
    c = 0
    h = 0
    for q, row in enumerate(trace):
        if row.d not in (0, 1) or row.a not in (0, 1):
            return False, f"row {q}: d/a not boolean"
        if row.d + row.a != 1:
            return False, f"row {q}: d + a != 1"
        if row.c_in != c:
            return False, f"row {q}: counter {row.c_in} != {c}"
        if row.h_in != h:
            return False, f"row {q}: accumulator {row.h_in} != {h}"
        # d_q * (label_q - c_q) = 0  -- delimiters are labelled in order
        if row.d and (row.label - row.c_in) % FIELD_P != 0:
            return False, f"row {q}: delimiter label {row.label} != counter {row.c_in}"
        # a_q * (e'_q - (c_q - 1)) = 0 -- a token sits in its own expert's segment
        if row.a and (row.e - (row.c_in - 1)) % FIELD_P != 0:
            return False, f"row {q}: token of expert {row.e} in segment {row.c_in - 1}"
        # h_{q+1} = (1 - d)(h + a * tau_t * x)
        delta = (row.a * tau[row.t] * row.x) % FIELD_P if row.a else 0
        want = 0 if row.d else (row.h_in + delta) % FIELD_P
        if row.h_out != want:
            return False, f"row {q}: recurrence {row.h_out} != {want}"
        if row.d and row.emitted != row.h_in:
            return False, f"row {q}: emitted {row.emitted} != accumulator {row.h_in}"
        c += row.d
        h = row.h_out
    if c != E + 1:
        return False, f"final counter {c} != E+1 = {E + 1}"
    return True, "ok"


# ── the payoff identity (§5.3) ───────────────────────────────────────────────
def segment_sums(route: Sequence[int], X: Sequence[Sequence[int]],
                 tau: Sequence[int], E: int) -> List[List[int]]:
    """A[e][j] = sum_{t: route[t] == e} tau_t * X[t][j], computed directly.
    The scan above is supposed to reproduce this; the tests compare them."""
    d = len(X[0])
    A = [[0] * d for _ in range(E)]
    for t, e in enumerate(route):
        for j in range(d):
            A[e][j] = (A[e][j] + tau[t] * X[t][j]) % FIELD_P
    return A


def project_experts(W: Sequence[Sequence[Sequence[int]]], rho: Sequence[int]
                    ) -> List[List[int]]:
    """P[e][j] = sum_i rho_i W[e][j][i] -- the same projection as projection.py,
    applied per expert."""
    E, d_in, d_out = len(W), len(W[0]), len(W[0][0])
    return [[sum(rho[i] * W[e][j][i] for i in range(d_out)) % FIELD_P
             for j in range(d_in)] for e in range(E)]


def selected_matmul(route: Sequence[int], X: Sequence[Sequence[int]],
                    W: Sequence[Sequence[Sequence[int]]]) -> List[List[int]]:
    """Y[t][i] = sum_j X[t][j] W[route[t]][j][i] -- top-1 routed expert matmul."""
    d_in, d_out = len(X[0]), len(W[0][0])
    return [[sum(X[t][j] * W[route[t]][j][i] for j in range(d_in)) % FIELD_P
             for i in range(d_out)] for t in range(len(route))]


def moe_identity_sides(route, X, W, tau, rho) -> Tuple[int, int]:
    """(lhs, rhs) of  sum_{t,i} tau_t rho_i Y[t,i] == sum_{e,j} A[e,j] P[e,j].
    Both sides are computed independently, so the test is a real check."""
    Y = selected_matmul(route, X, W)
    lhs = 0
    for t in range(len(route)):
        for i in range(len(Y[0])):
            lhs = (lhs + tau[t] * rho[i] * Y[t][i]) % FIELD_P
    A = segment_sums(route, X, tau, len(W))
    Pp = project_experts(W, rho)
    rhs = 0
    for e in range(len(W)):
        for j in range(len(X[0])):
            rhs = (rhs + A[e][j] * Pp[e][j]) % FIELD_P
    return lhs, rhs


def cell_counts(S: int, E: int, d: int, d_ff: int) -> Dict[str, int]:
    """The §5.3 accounting, at whatever scale you hand it. naive = S*E*(2d+d_ff)
    private selector cells; segmented = S*(2d+d_ff) plus the permutation records."""
    naive = S * E * (2 * d + d_ff)
    segmented = S * (2 * d + d_ff)
    return {"naive": naive, "segmented": segmented,
            "permutation_records": S * d, "ratio": naive // max(segmented, 1)}
