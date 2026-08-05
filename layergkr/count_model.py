"""Count-based cost model — the replacement for kappa.

The previous model produced seconds and then multiplied by a calibration factor
`kappa <= 1.5` to cover the difference between the model and reality. A factor
like that is not a safety margin, it is the size of the modelling error, and it
cannot be validated: any outcome is "within kappa" for a large enough kappa.

This model predicts COUNTS instead:

    predict_layer(trace, cfg)  ->  {phase: {mul, add, inv, hash_bytes, ...}}

Counts are a property of the protocol and the instance geometry, not of the
machine, so they can be checked EXACTLY against an instrumented run. That is what
`bench/run_toy.py` does; a mismatch is a missing term to be added here, never a
multiplier to be raised. Seconds come afterwards, from a measured rate card:

    seconds = counts . rates          (counters.Rates, measured by bench/rates.py)

Two levels, validated separately:

  L1  geometry -> trace shape   (how many matmuls of what shape, how many gate
                                 slots, how many lookup queries)
  L2  trace shape -> primitive counts

L1 is checked against the emitted trace, L2 against the counters. Both report
relative error per phase, and the bench asserts they are exact where they should
be exact.

HONEST LIMIT. A model validated against this prototype predicts THIS prototype.
Extrapolating to 400B assumes the production implementation performs the same
operations per unit of geometry -- same encode, same seam, same sumcheck shapes.
That assumption is explicit and checkable, which is exactly what a kappa is not.
"""
from dataclasses import dataclass
from typing import Dict, List, Sequence

from . import semantics as sem
from .counters import Rates
from .field import INV_MULS
from .rs import Config

Counts = Dict[str, int]


def _zero() -> Counts:
    return {"mul": 0, "mul_defer": 0, "add": 0, "inv": 0, "hash_bytes": 0,
            "hash_calls": 0, "opened_values": 0, "proof_bytes": 0}


def _acc(dst: Counts, **kw) -> Counts:
    for k, v in kw.items():
        dst[k] = dst.get(k, 0) + v
    return dst


def _add(dst: Counts, src: Counts) -> Counts:
    for k, v in src.items():
        dst[k] = dst.get(k, 0) + v
    return dst


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


# ── primitive building blocks, mirroring the implementation exactly ──────────
def encode_row(cfg: Config) -> Counts:
    """rs.encode_row: one row through the cached Lagrange matrix."""
    return _acc(_zero(), mul_defer=cfg.ELL * cfg.N_LIG, add=(cfg.ELL - 1) * cfg.N_LIG)


def commit_hash(cfg: Config, n_rows: int) -> Counts:
    """rs.Commit: column leaves plus the internal tree nodes."""
    return _acc(_zero(),
                hash_bytes=8 * n_rows * cfg.N_LIG + 64 * max(cfg.N_LIG - 1, 0),
                hash_calls=cfg.N_LIG + max(cfg.N_LIG - 1, 0))


def commit_from_messages(cfg: Config, n_rows: int) -> Counts:
    c = _zero()
    for _ in range(n_rows):
        _add(c, encode_row(cfg))
    return _add(c, commit_hash(cfg, n_rows))


def sumcheck_counts(size: int, factor_counts: Sequence[int]) -> Counts:
    """sumcheck.prove_terms. Per round the prover evaluates every term at deg+1
    points over `half` positions; `half` halves each round, so the geometric sum
    over all rounds is (size - 1)."""
    if size < 2:
        return _zero()
    deg = max(factor_counts)
    n_rounds = size.bit_length() - 1
    total_half = 0
    half = size // 2
    for _ in range(n_rounds):
        total_half += half
        half //= 2
    mul = (deg + 1) * total_half * sum(f + 1 for f in factor_counts)
    add = (deg + 1) * total_half * sum(factor_counts)
    return _acc(_zero(), mul=mul, add=add)


def eq_vector_counts(n_vars: int) -> Counts:
    size = 1 << n_vars
    return _acc(_zero(), mul=2 * (size - 1), add=size - 1)


def batch_inv_counts(n: int) -> Counts:
    """field.batch_inv: Montgomery's trick -- 3n muls and ONE inversion."""
    if n == 0:
        return _zero()
    return _acc(_zero(), mul=3 * n + INV_MULS, inv=1)


# ── L2: trace shape -> counts, phase by phase ────────────────────────────────
def predict_layer(trace: sem.LayerTrace, cfg: Config, q_columns: int,
                  n_enrolled: int, masked: bool = True) -> Dict[str, Counts]:
    """Predict what `full_layer.prove_full_layer` will do on this trace.

    `n_enrolled` is the number of DISTINCT weight tensors (enrolment is amortised
    across tokens, so it is not len(trace.matmuls))."""
    out: Dict[str, Counts] = {}

    # commit.states: input state (S rows) + output state (S rows)
    S = len(trace.x_in)
    c = _zero()
    _add(c, commit_from_messages(cfg, S))
    _add(c, commit_from_messages(cfg, S))
    out["commit.states"] = c

    # commit.mask
    n_tape = 8 * (max(len(trace.gates), 1).bit_length() + 16)
    tape_rows = -(-n_tape // cfg.ELL)
    out["commit.mask"] = commit_from_messages(cfg, tape_rows)

    # logup, per table use
    c = _zero()
    for use in trace.lookups:
        nq = _next_pow2(len(use.queries))
        nt = _next_pow2(use.table.size)
        width = len(use.table.tuples()[0]) + 1
        _add(c, commit_from_messages(cfg, nq + nt))          # raw: queries + mults
        _acc(c, mul=width * (nq + nt), add=width * (nq + nt))  # compression
        _add(c, commit_from_messages(cfg, -(-nq // cfg.ELL) + -(-nt // cfg.ELL)))
        _add(c, batch_inv_counts(nq))
        _add(c, batch_inv_counts(nt))
        nqv, ntv = max(nq.bit_length() - 1, 1), max(nt.bit_length() - 1, 1)
        _add(c, eq_vector_counts(nqv))
        _add(c, eq_vector_counts(ntv))
        _add(c, sumcheck_counts(nq, [3]))     # eq * r * (alpha - v)
        _add(c, sumcheck_counts(nt, [3]))
        _add(c, sumcheck_counts(nq, [1]))     # sum of r
        _add(c, sumcheck_counts(nt, [1]))
    out["logup"] = c

    # gates: one batched eq-weighted sumcheck
    size = 1
    for g in trace.gates:
        size = max(size, g.size)
    size = _next_pow2(size)
    n_vars = max(size.bit_length() - 1, 1)
    c = eq_vector_counts(n_vars)
    _acc(c, mul=size)                      # relations.prove_batch charges eq once
    factor_counts = []
    for g in trace.gates:
        for _, factors in g.terms:
            factor_counts.append(len(factors) + 1)   # +1 for the eq factor
    _add(c, sumcheck_counts(size, factor_counts))
    out["gates"] = c

    # matmuls
    c = _zero()
    for _ in range(n_enrolled):
        pass                                # enrolment counted below, per tensor
    seen = set()
    for m in trace.matmuls:
        n_out, n_in, rows = len(m.W), len(m.W[0]), len(m.X)
        key = tuple(tuple(r) for r in m.W)
        if key not in seen:
            seen.add(key)
            _add(c, commit_from_messages(cfg, n_out))          # enroll W
        _acc(c, mul_defer=n_out * cfg.N_LIG, add=n_out * cfg.N_LIG)  # F_P
        _add(c, commit_hash(cfg, 1))
        _add(c, commit_from_messages(cfg, rows))               # enroll X state
        _acc(c, mul_defer=rows * cfg.N_LIG, add=rows * cfg.N_LIG)    # F_A
        _add(c, commit_hash(cfg, 1))
        _acc(c, mul_defer=n_out * cfg.ELL, add=n_out * cfg.ELL)  # project_message(W)
        _acc(c, mul_defer=rows * cfg.ELL, add=rows * cfg.ELL)    # project_message(X)
        _add(c, sumcheck_counts(_next_pow2(n_in), [2]))
    out["matmuls"] = c

    # MoE nodes: the routed contraction with the route hidden (doc §5). One
    # projection per EXPERT (all of them, which is what hides the route), then a
    # single sumcheck over the flattened (expert, coordinate) domain.
    c = _zero()
    for m in trace.moe:
        E = len(m.W)
        d_out, n_in = len(m.W[0]), len(m.W[0][0])
        for e in range(E):
            key = tuple(tuple(r) for r in m.W[e])
            if key not in seen:
                seen.add(key)
                _add(c, commit_from_messages(cfg, d_out))       # enroll expert e
            _acc(c, mul_defer=d_out * cfg.N_LIG, add=d_out * cfg.N_LIG,
                 comb_iter=d_out * cfg.N_LIG)                   # F_P
            _add(c, commit_hash(cfg, 1))
            _acc(c, mul_defer=d_out * cfg.ELL, add=d_out * cfg.ELL,
                 comb_iter=d_out * cfg.ELL)                     # project_message
        _add(c, sumcheck_counts(_next_pow2(E * n_in), [2]))
    out["moe"] = c

    # open
    opened = sum((len(m.W) + 1 + len(m.X) + 1) * q_columns for m in trace.matmuls)
    opened += sum(len(m.W) * (len(m.W[0]) + 1) * q_columns for m in trace.moe)
    out["open"] = _acc(_zero(), opened_values=opened, proof_bytes=8 * opened)

    total = _zero()
    for v in out.values():
        _add(total, v)
    out["TOTAL"] = total
    return out


def interp_counts(n_points: int) -> Counts:
    """sumcheck._lagrange_interpolate, one call. Mirrors its charge exactly."""
    n = n_points
    return _acc(_zero(), mul=n * (2 * (n - 1) + 2) + n * INV_MULS,
                add=2 * n * (n - 1), inv=n)


def mle_eval_counts(size: int) -> Counts:
    return _acc(_zero(), mul=max(size - 1, 0), add=2 * max(size - 1, 0))


def verify_sumcheck_counts(size: int, factor_counts: Sequence[int]) -> Counts:
    """sumcheck.verify_terms: per round three interpolations (at 0, at 1, at r),
    then one MLE evaluation per factor of every term."""
    c = _zero()
    if size < 2:
        return c
    deg = max(factor_counts)
    n_points = deg + 1
    n_rounds = size.bit_length() - 1
    for _ in range(n_rounds):
        for _ in range(3):
            _add(c, interp_counts(n_points))
    for f in factor_counts:
        for _ in range(f):
            _add(c, mle_eval_counts(size))
    return c


def predict_verify(trace: sem.LayerTrace, cfg: Config, q_columns: int) -> Dict[str, Counts]:
    """Predict the VERIFIER's counts, mirroring verify_full_layer call for call.

    The source document models NO verification at all -- its §9.4 GPU estimate is
    an explicitly conditional corollary outside the theorem -- so this half has no
    counterpart to compare against, only measurement."""
    out: Dict[str, Counts] = {}
    depth = max(cfg.N_LIG.bit_length() - 1, 1)

    c = _zero()
    for m in trace.matmuls:
        n_out, n_in, rows = len(m.W), len(m.W[0]), len(m.X)
        for n_rows in (n_out, rows):                       # two projection seams
            _acc(c, mul=q_columns * n_rows, add=q_columns * n_rows)
            _acc(c, hash_bytes=q_columns * (8 * n_rows + 8) + 2 * q_columns * depth * 64,
                 hash_calls=2 * q_columns * (1 + depth))
        _add(c, encode_row(cfg))                            # re-encode P
        _add(c, encode_row(cfg))                            # re-encode A
        _add(c, verify_sumcheck_counts(_next_pow2(n_in), [2]))
    out["verify.matmuls"] = c

    c = _zero()
    for m in trace.moe:
        E, d_out, n_in = len(m.W), len(m.W[0]), len(m.W[0][0])
        for _ in range(E):
            _acc(c, mul=q_columns * d_out, add=q_columns * d_out)
            _acc(c, hash_bytes=q_columns * (8 * d_out + 8) + 2 * q_columns * depth * 64,
                 hash_calls=2 * q_columns * (1 + depth))
            _add(c, encode_row(cfg))
        _add(c, verify_sumcheck_counts(_next_pow2(E * n_in), [2]))
    out["verify.moe"] = c

    size = 1
    for g in trace.gates:
        size = max(size, g.size)
    size = _next_pow2(size)
    n_vars = max(size.bit_length() - 1, 1)
    factor_counts = [len(f) + 1 for g in trace.gates for _, f in g.terms]
    c = eq_vector_counts(n_vars)
    _add(c, verify_sumcheck_counts(size, factor_counts))
    out["verify.gates"] = c

    c = _zero()
    for use in trace.lookups:
        nq = _next_pow2(len(use.queries))
        nt = _next_pow2(use.table.size)
        _add(c, eq_vector_counts(max(nq.bit_length() - 1, 1)))
        _add(c, eq_vector_counts(max(nt.bit_length() - 1, 1)))
        _add(c, verify_sumcheck_counts(nq, [3]))
        _add(c, verify_sumcheck_counts(nt, [3]))
        _add(c, verify_sumcheck_counts(nq, [1]))
        _add(c, verify_sumcheck_counts(nt, [1]))
    out["verify.logup"] = c

    total = _zero()
    for v in out.values():
        _add(total, v)
    out["TOTAL"] = total
    return out


# ── L1: geometry -> trace shape ──────────────────────────────────────────────
@dataclass
class TraceShape:
    matmuls: int
    matmul_cells: int
    gates: int
    gate_slots: int
    lookup_queries: int
    lookup_table_rows: int
    # The FFN is three hidden-route MoE nodes, not per-token matmuls. Leaving
    # them out of the shape is what let the stale L1 go unnoticed.
    moe_nodes: int = 0
    moe_cells: int = 0


def predict_trace_shape(cfg: sem.ToyConfig) -> TraceShape:
    """How many relations a layer of this geometry emits, from the structure of
    semantics.forward -- not by running it. Validated in bench/run_toy.py.

    CORRECTED 2026-08-05. The previous version modelled the PRE-MoE emitter: it
    counted `5 + 3*S` matmuls, one per token per expert projection, and derived
    the range-check count from matmul CELLS. The emitter has used three hidden-
    route MoE nodes since the §5 work, and range checks are one per rescaled
    OUTPUT, not per cell. The error was 8x on lookups, 3.4x on gate slots and
    2.4x on matmul cells -- in a level the doc called exact. It is exact again,
    on all five quantities, at every configuration in `tests/test_count_model`."""
    S, d, d_ff, E = cfg.S, cfg.d, cfg.d_ff, max(cfg.E, 1)
    n_tab = cfg.table_size

    # Five NAMED matmuls: Wq, Wk, Wv (X=(S,d), W=(d,d)), attn_pv (X=(S,S),
    # W=v^T=(d,S)), Wo (X=(S,d), W=(d,d)). The FFN is three MoE nodes, counted
    # separately -- they are not matmuls in the trace.
    matmuls = 5
    cells = 4 * S * d * d + S * S * d
    moe_cells = 3 * E * d * d_ff

    # Gates, in emission order. Constants first, then the per-token families.
    #   constant: rms_sq, ss bracket, Wq/Wk/Wv rescales, causal mask, score
    #             bracket, sums bracket, attn_pv, Wo, moe_gate, moe_up, moe_down
    #   per token: rms (had+rescale), RoPE q and k (affine+rescale each),
    #              softmax (had+rescale), residual 1, SwiGLU (bracket+had+
    #              rescale), residual 2
    gates = 13 + 13 * S

    # Slots, by gate family:
    #   hadamard  2*S*d + S*S + S*d_ff
    #   rescale   9*S*d + S*S + 3*S*d_ff
    #   bracket   2*S   + S*S + S*d_ff
    #   affine    4*S*d
    #   boolean   S*S
    slots = 15 * S * d + 4 * S * S + 5 * S * d_ff + 2 * S

    # Lookups = every range check (one per rescale remainder and per bracket
    # low half) plus the four value tables.
    range_checks = 9 * S * d + 2 * S * S + 4 * S * d_ff + 2 * S
    tables = 2 * S + S * S + S * d_ff          # isqrt, recip, exp, silu
    return TraceShape(matmuls=matmuls, matmul_cells=cells, gates=gates,
                      gate_slots=slots, lookup_queries=range_checks + tables,
                      lookup_table_rows=5 * n_tab, moe_nodes=3,
                      moe_cells=moe_cells)


def seconds(counts: Counts, rates: Rates) -> float:
    return rates.seconds(counts)


# ── L4: bytes ────────────────────────────────────────────────────────────────
# Everything above this line ends in SECONDS. That was the whole model, and it
# meant the model could not say the one thing that actually stopped two runs
# today: that they would not fit. A prediction of "4 hours" for a job that dies
# in the first minute on an allocation is not a conservative prediction, it is a
# wrong one.
#
# Memory is predicted from the same L1 geometry, in bytes, and validated the same
# way -- against measured peaks, with the error stated.

BYTES_PER_FIELD = 8

# Length-65536 NTTs dispatch to a Bailey 4-step in `prover/cuda_primitives.py`,
# which needs a per-row temp of 65536 uint64. It is allocated with a raw
# `cudaMalloc`, cached globally, keyed by row count, and only ever grows -- so it
# is invisible to `torch.cuda.max_memory_allocated()` and competes with torch's
# caching allocator for the same card.
BAILEY_N = 65536


def ntt_scratch_bytes(cfg: Config, rows: int) -> int:
    return rows * BAILEY_N * BYTES_PER_FIELD if cfg.N_LIG == BAILEY_N else 0


def encode_peak_bytes(cfg: Config, rows: int, chunk: int = None) -> int:
    """Transient device bytes to encode `rows` messages in chunks of `chunk`.

    Four live buffers, measured from `gpu.encode_batch_ntt`: the destination
    codeword matrix for ALL rows, plus per chunk a contiguous head copy, the
    expanded coset powers, and the forward NTT's own copy -- and, at N=65536,
    the Bailey scratch on top."""
    chunk = min(rows, chunk or rows)
    dest = rows * cfg.N_LIG * BYTES_PER_FIELD
    head = 2 * chunk * cfg.K_DEG * BYTES_PER_FIELD      # head + coset powers
    fwd = chunk * cfg.N_LIG * BYTES_PER_FIELD
    return dest + head + fwd + ntt_scratch_bytes(cfg, chunk)


def max_encode_rows(cfg: Config, free_bytes: int) -> int:
    """Largest chunk that fits in `free_bytes`, ignoring the destination (which
    is allocated once for all rows). This is the number `gpu._chunk_rows`
    computes at run time; having it here means a run can be REFUSED before it
    starts instead of dying inside a kernel."""
    per_row = 2 * cfg.K_DEG + cfg.N_LIG
    per_row += BAILEY_N if cfg.N_LIG == BAILEY_N else 0
    return max(0, int(free_bytes) // (per_row * BYTES_PER_FIELD))


def predict_forward_memory(toy: sem.ToyConfig) -> Dict[str, int]:
    """Peak device bytes of `semantics.forward_tensor`, by term.

    The expert weights dominate everything else by an order of magnitude, and
    they are counted TWICE: the per-expert tensors are built first and then
    `torch.stack`ed into (E, d_out, d_in), so both live at the moment of the
    stack. That doubling is not a detail -- it is most of the peak."""
    S, d, d_ff, E = toy.S, toy.d, toy.d_ff, max(toy.E, 1)
    B = BYTES_PER_FIELD
    experts = 3 * E * d * d_ff * B
    shape = predict_trace_shape(toy)
    return {
        "expert_weights": experts,
        "expert_stack_copy": experts,       # list and stack are both live
        "attention_weights": 4 * d * d * B,
        "activations": (2 * S * S + 8 * S * d + 2 * S * d_ff) * B,
        "gate_factors": shape.gate_slots * 2 * B,
        "lookup_log": shape.lookup_queries * 2 * B,
    }


def predict_prove_memory(trace: sem.LayerTrace, cfg: Config,
                         chunk: int = None) -> Dict[str, int]:
    """Peak device bytes of `full_layer.prove_full_layer`, by term.

    The term that decides whether a run starts at all is LogUp: it commits one
    RS row per lookup QUERY, so a 2-tuple query becomes a full N_LIG codeword.
    At production geometry that is 512 KB per query."""
    B = BYTES_PER_FIELD
    c = trace.counts()
    n_lookup_rows = c["lookup_queries"] + c["lookup_table_rows"]

    weight_rows = 0
    for m in trace.matmuls:
        n_out, n_in = len(m.W), len(m.W[0])
        weight_rows += n_out * -(-n_in // cfg.ELL)
    for m in trace.moe:
        E, d_out, n_in = len(m.W), len(m.W[0]), len(m.W[0][0])
        weight_rows += E * d_out * -(-n_in // cfg.ELL)

    terms = {
        # persistent, one codeword row per output coordinate per block
        "enrolled_weights": weight_rows * cfg.N_LIG * B,
        "enrolled_messages": weight_rows * cfg.ELL * B,
        # LogUp: ONE ROW PER QUERY. This is the artefact, and it is the peak.
        "logup_raw_commit": n_lookup_rows * cfg.N_LIG * B,
        "logup_compressed": (2 * -(-n_lookup_rows // cfg.ELL)) * cfg.N_LIG * B,
        # fresh projections, one row per block per matmul
        "projections": (len(trace.matmuls) + len(trace.moe)) * cfg.N_LIG * B,
        "states": 3 * len(trace.x_in) * cfg.N_LIG * B,
    }
    # the largest single encode is the LogUp one, and its transient rides on top
    terms["encode_transient"] = (
        encode_peak_bytes(cfg, n_lookup_rows, chunk)
        - n_lookup_rows * cfg.N_LIG * B)      # destination already counted above
    return terms


def will_it_fit(trace: sem.LayerTrace, cfg: Config, device_bytes: int,
                chunk: int = None) -> Dict[str, object]:
    """The question the model could not answer before. Returns the verdict, the
    predicted peak, and the term responsible -- naming the term matters, because
    'needs 41 GB' is not actionable and 'LogUp needs 41 GB' is."""
    terms = predict_prove_memory(trace, cfg, chunk)
    total = sum(terms.values())
    worst = max(terms.items(), key=lambda kv: kv[1])
    return {"fits": total <= device_bytes, "predicted_bytes": total,
            "device_bytes": device_bytes, "largest_term": worst[0],
            "largest_bytes": worst[1], "terms": terms}
