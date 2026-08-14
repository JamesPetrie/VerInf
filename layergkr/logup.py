"""LogUp lookups with the ordering discipline of doc §6.

The nonlinear layer semantics (SiLU, exp for softmax, isqrt for RMSNorm, the
rescale/range brackets) are table lookups in VerInf's exact integer semantics.
This module proves them.

The identity, for queries q_1..q_n against a table t_1..t_m with multiplicities:

    sum_i 1/(alpha - q_i)  ==  sum_k mult_k/(alpha - t_k)

with each side's reciprocals committed and then CHECKED, because a prover who
may choose r freely can make any sum come out. Each side carries a degree-3
sumcheck for its defining constraint, using the standard eq-weighting: for a
random z,

    sum_x eq(z,x) * r(x) * (alpha - v(x))  ==  sum_x eq(z,x)  ==  1

holds iff r(x)*(alpha - v(x)) = 1 at every hypercube point, i.e. r really is the
reciprocal vector. Both sumchecks run through `sumcheck.py`, so they are the same
machinery (and the same masking) as every other relation.

ORDERING IS THE POINT (§6). The challenge cannot be issued before the tuples are
bound, or the prover fits the lookup witness to the sampled reciprocal point.
The staged API enforces it:

    lu = LogUp(...)          # raw query/table tuples + multiplicities
    lu.commit_raw(tr)        # L1 roots
    beta  = tr.coin("beta")  # only now: tuple compression
    lu.compress(beta, tr)    # R_cmp: compressed fingerprints and table values
    alpha = tr.coin("alpha") # only now: the reciprocal point
    lu.build(alpha)          # reciprocals exist at last
    lu.prove(tr)

Calling them out of order raises, and with a scheduled transcript the coin is
refused outright. `tests/test_logup.py` gates both paths.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from prover.protocol import P as FIELD_P

from . import field as F, rs, sumcheck as sc
from .counters import phase
from .transcript import Transcript


# First element of a table-padding tuple. Distinct from every real table entry,
# so a query can never match one; its multiplicity stays 0.
_PAD_SENTINEL = FIELD_P - 12345


def compress_tuple(t: Sequence[int], beta: int) -> int:
    """chi_beta: Horner over the tuple. Table ids get public domain tags, so two
    different tables cannot collide by construction (doc §6)."""
    acc = 0
    for v in t:
        acc = (acc * beta + (v % FIELD_P)) % FIELD_P
    return acc


def eq_vector(z: Sequence[int]) -> List[int]:
    """eq(z, .) over the hypercube, most-significant variable first."""
    out = [1]
    for zi in z:
        out = [(v * (1 - zi)) % FIELD_P for v in out] + [(v * zi) % FIELD_P for v in out]
    from .counters import charge as _charge
    _charge(mul=2 * (len(out) - 1), add=len(out) - 1,
            red_op=2 * (len(out) - 1))
    return out


def _pad_pow2(vec: List[int], fill: int = 0) -> List[int]:
    n = 1
    while n < len(vec):
        n *= 2
    return vec + [fill] * (n - len(vec))


@dataclass
class LogUpProof:
    n_queries: int
    n_table: int
    sum_query: int
    sum_table: int
    query_constraint: sc.SumcheckProof
    table_constraint: sc.SumcheckProof
    query_sum_check: sc.SumcheckProof
    table_sum_check: sc.SumcheckProof
    z_query: List[int]
    z_table: List[int]
    # PROTOTYPE: the reciprocal vectors travel in the clear so the verifier can
    # check the constraint sumchecks against what the prover actually used. In
    # the scheme they are terminal claims on committed roots, discharged by the
    # LF proof. Revealing them here costs hiding, not soundness -- a wrong r
    # still fails the constraint.
    r_query: List[int] = field(default_factory=list, repr=False)
    r_table: List[int] = field(default_factory=list, repr=False)
    raw_root: bytes = b""
    cmp_root: bytes = b""


@dataclass
class LogUp:
    """One local lookup argument for a single layer (the doc uses one repetition
    per layer, which suffices for the T=40 profile: eps_lookup ~ 2^-28.5)."""
    cfg: rs.Config
    table_id: int
    queries: List[Tuple[int, ...]]
    table: List[Tuple[int, ...]]
    mult: List[int] = field(default_factory=list)
    _stage: str = "init"
    _beta: Optional[int] = None
    _alpha: Optional[int] = None
    q_vals: List[int] = field(default_factory=list, repr=False)
    t_vals: List[int] = field(default_factory=list, repr=False)
    r_q: List[int] = field(default_factory=list, repr=False)
    r_t: List[int] = field(default_factory=list, repr=False)
    raw_commit: Optional[rs.Commit] = None
    cmp_commit: Optional[rs.Commit] = None

    def __post_init__(self):
        if not self.table:
            raise ValueError("empty table")
        index = {t: k for k, t in enumerate(self.table)}
        for q in self.queries:
            if q not in index:
                raise KeyError(f"query {q} is not in table {self.table_id}")

        # Both sides are padded to a power of two BEFORE anything is committed,
        # and padded with values that satisfy the reciprocal constraint -- a slot
        # holding r = 0 would make sum_x eq(z,x)*r(x)*(alpha - v(x)) fall short of
        # sum_x eq(z,x), because eq-weighting does NOT drop padding.
        #   queries: repeat a real table entry, and count it in the multiplicity,
        #            so it is an ordinary lookup rather than a special case;
        #   table:   append sentinel tuples that no query can match, with
        #            multiplicity 0, so they contribute nothing to the identity
        #            while still having a well-defined reciprocal.
        width = len(self.table[0])
        self.queries = list(self.queries)
        self.table = list(self.table)
        while len(self.queries) & (len(self.queries) - 1) or not self.queries:
            self.queries.append(self.table[0])
        n_t = len(self.table)
        pad_t = 1
        while pad_t < n_t:
            pad_t *= 2
        for k in range(pad_t - n_t):
            self.table.append((_PAD_SENTINEL, k) + (0,) * (width - 2)
                              if width >= 2 else (_PAD_SENTINEL,))

        if not self.mult:
            index = {t: k for k, t in enumerate(self.table)}
            m = [0] * len(self.table)
            for q in self.queries:
                m[index[q]] += 1
            self.mult = m

    # ── stage 1: bind the raw tuples ─────────────────────────────────────────
    def commit_raw(self, tr: Transcript, label: str = "R_lk") -> bytes:
        if self._stage != "init":
            raise RuntimeError(f"commit_raw out of order (stage={self._stage})")
        with phase("logup.commit_raw"):
            rows = [_pad_pow2([v % FIELD_P for v in q])[:self.cfg.ELL]
                    for q in self.queries]
            rows += [[m % FIELD_P] for m in self.mult]
            rows = [r[:self.cfg.ELL] for r in rows]
            self.raw_commit = rs.Commit.from_messages(self.cfg, rows)
        tr.absorb_root(label, self.raw_commit.root)
        self._stage = "raw"
        return self.raw_commit.root

    # ── stage 2: compress, AFTER beta and BEFORE alpha ───────────────────────
    def compress(self, beta: int, tr: Transcript, label: str = "R_cmp") -> bytes:
        if self._stage != "raw":
            raise RuntimeError(f"compress out of order (stage={self._stage})")
        with phase("logup.compress"):
            tag = self.table_id
            self.q_vals = [compress_tuple((tag,) + q, beta) for q in self.queries]
            self.t_vals = [compress_tuple((tag,) + t, beta) for t in self.table]
            width = len(self.queries[0]) + 1 if self.queries else 1
            F.charge(mul=width * (len(self.queries) + len(self.table)),
                     add=width * (len(self.queries) + len(self.table)))
            rows = [_pad_pow2(self.q_vals)[i:i + self.cfg.ELL]
                    for i in range(0, len(_pad_pow2(self.q_vals)), self.cfg.ELL)]
            rows += [_pad_pow2(self.t_vals)[i:i + self.cfg.ELL]
                     for i in range(0, len(_pad_pow2(self.t_vals)), self.cfg.ELL)]
            self.cmp_commit = rs.Commit.from_messages(self.cfg, rows)
        self._beta = beta
        tr.absorb_root(label, self.cmp_commit.root)
        self._stage = "cmp"
        return self.cmp_commit.root

    # ── stage 3: reciprocals, only after alpha ───────────────────────────────
    def build(self, alpha: int) -> None:
        if self._stage != "cmp":
            raise RuntimeError(
                f"build out of order (stage={self._stage}) -- the reciprocal point "
                f"must come after the compressed tuples are committed")
        with phase("logup.reciprocals"):
            self._alpha = alpha
            self.r_q = F.batch_inv([(alpha - v) % FIELD_P for v in self.q_vals])
            self.r_t = F.batch_inv([(alpha - v) % FIELD_P for v in self.t_vals])
        self._stage = "built"

    # ── stage 4: prove ───────────────────────────────────────────────────────
    def prove(self, tr: Transcript) -> LogUpProof:
        if self._stage != "built":
            raise RuntimeError(f"prove out of order (stage={self._stage})")
        alpha = self._alpha
        with phase("logup.sumchecks"):
            rq = _pad_pow2(self.r_q)
            aq = _pad_pow2([(alpha - v) % FIELD_P for v in self.q_vals], fill=0)
            # padding slots carry r=0 and (alpha - v)=0, so their constraint term
            # is 0*0 = 0 and the eq-weighted sum drops the padding automatically;
            # the query-count is carried explicitly instead.
            rt = _pad_pow2([(m * r) % FIELD_P for m, r in zip(self.mult, self.r_t)])
            at = _pad_pow2([(alpha - v) % FIELD_P for v in self.t_vals], fill=0)
            rt_plain = _pad_pow2(self.r_t)

            nq = max(len(rq).bit_length() - 1, 1)
            nt = max(len(rt).bit_length() - 1, 1)
            zq = tr.coin("logup_zq", nq)
            zt = tr.coin("logup_zt", nt)
            eq_q, eq_t = eq_vector(zq), eq_vector(zt)

            coins_q = tr.coin("logup_cq", nq)
            coins_t = tr.coin("logup_ct", nt)
            qc = sc.prove([eq_q, rq, aq], lambda i: coins_q[i])
            tc = sc.prove([eq_t, rt_plain, at], lambda i: coins_t[i])

            coins_qs = tr.coin("logup_qs", nq)
            coins_ts = tr.coin("logup_ts", nt)
            qs = sc.prove([rq], lambda i: coins_qs[i])
            ts = sc.prove([rt], lambda i: coins_ts[i])

        return LogUpProof(
            n_queries=len(self.queries), n_table=len(self.table),
            sum_query=qs.claim, sum_table=ts.claim,
            query_constraint=qc, table_constraint=tc,
            query_sum_check=qs, table_sum_check=ts,
            z_query=zq, z_table=zt,
            r_query=list(self.r_q), r_table=list(self.r_t),
            raw_root=self.raw_commit.root if self.raw_commit else b"",
            cmp_root=self.cmp_commit.root if self.cmp_commit else b"")


def verify(proof: LogUpProof, q_vals: Sequence[int], t_vals: Sequence[int],
           mult: Sequence[int], alpha: int) -> Tuple[bool, str]:
    """Verifier side. In the full scheme the verifier does not hold q_vals/t_vals
    -- they are terminal claims on R_cmp discharged by the LF proof. Here they
    are passed in, which is the same prototype simplification as layer.py and is
    marked as such in the README."""
    # (alpha - v) comes from the compressed tuples the verifier holds a claim on;
    # r comes from the PROOF, so a prover who submits reciprocals that are not
    # reciprocals fails the constraint sumcheck below rather than being handed
    # correct ones by the verifier.
    aq = _pad_pow2([(alpha - v) % FIELD_P for v in q_vals], fill=0)
    at = _pad_pow2([(alpha - v) % FIELD_P for v in t_vals], fill=0)
    rq = _pad_pow2(list(proof.r_query))
    rt_plain = _pad_pow2(list(proof.r_table))
    rt_w = _pad_pow2([(m * r) % FIELD_P for m, r in zip(mult, proof.r_table)])

    eq_q, eq_t = eq_vector(proof.z_query), eq_vector(proof.z_table)

    ok, why = sc.verify(proof.query_constraint, [eq_q, rq, aq],
                        lambda i: proof.query_constraint.challenges[i])
    if not ok:
        return False, f"query reciprocal constraint: {why}"
    ok, why = sc.verify(proof.table_constraint, [eq_t, rt_plain, at],
                        lambda i: proof.table_constraint.challenges[i])
    if not ok:
        return False, f"table reciprocal constraint: {why}"
    ok, why = sc.verify(proof.query_sum_check, [rq],
                        lambda i: proof.query_sum_check.challenges[i])
    if not ok:
        return False, f"query sum: {why}"
    ok, why = sc.verify(proof.table_sum_check, [rt_w],
                        lambda i: proof.table_sum_check.challenges[i])
    if not ok:
        return False, f"table sum: {why}"

    # the eq-weighted constraints must each equal sum_x eq(z,x) = 1
    if proof.query_constraint.claim != sum(eq_q) % FIELD_P:
        return False, "query reciprocals are not reciprocals"
    if proof.table_constraint.claim != sum(eq_t) % FIELD_P:
        return False, "table reciprocals are not reciprocals"
    # and the two sides of the LogUp identity must agree
    if proof.sum_query != proof.sum_table:
        return False, "logup identity: query side != table side"
    return True, "ok"


def identity_sides(queries: Sequence[Tuple[int, ...]], table: Sequence[Tuple[int, ...]],
                   mult: Sequence[int], beta: int, alpha: int, table_id: int
                   ) -> Tuple[int, int]:
    """Both sides of sum 1/(alpha-q) == sum m/(alpha-t), computed independently."""
    qs = [compress_tuple((table_id,) + q, beta) for q in queries]
    ts = [compress_tuple((table_id,) + t, beta) for t in table]
    lhs = sum(pow((alpha - v) % FIELD_P, FIELD_P - 2, FIELD_P) for v in qs) % FIELD_P
    rhs = sum(m * pow((alpha - v) % FIELD_P, FIELD_P - 2, FIELD_P)
              for m, v in zip(mult, ts)) % FIELD_P
    return lhs, rhs
