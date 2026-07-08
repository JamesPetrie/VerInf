"""
protocol.py — the byte-exact contract shared by the prover and the Rust verifier.

Holds ONLY the things the two sides must agree on bit-for-bit, and which the live
prover imports: the Goldilocks field, the evaluation domains + poly/Lagrange eval,
Merkle verification, the challenge PRF (`challenge`, `op_vec`, `round_seeds`,
`random_columns`), and the claim → JSON serializer (`claims_to_json`) that defines
the wire format the Rust verifier reads. No prover mechanics (encoding, NTT,
witness generation) live here.

Design: field elements are plain Python ints in [0, P) — exact and readable.

The constraint **compile** is the Rust verifier's job (`verifier-rs/handlers.rs`),
which is the single source of truth; end-to-end ACCEPT is the bit-exactness check.
The old standalone-Python-verifier compile that the Rust was ported from
(`REAL_COMPILE` / `compile_claims` / the `expand_*`/`_c_*`/`_emit_*` machinery) has
been retired to `deprecated/python_verifier_compile.py`.
"""

from dataclasses import dataclass
from typing import Callable, List, Tuple
import blake3

# ----------------------------------------------------------------------
# Field: Goldilocks  P = 2^64 - 2^32 + 1
# ----------------------------------------------------------------------
P = (1 << 64) - (1 << 32) + 1
GLOBAL_G = 7                       # primitive root of F_P (multiplicative order P-1)

def add(a, b): return (a + b) % P
def sub(a, b): return (a - b) % P
def mul(a, b): return (a * b) % P
def inv(a):    return pow(a % P, P - 2, P)     # Fermat inverse; a != 0

# ----------------------------------------------------------------------
# Parameters and evaluation domains (must match the prover's encoding)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    ELL: int          # constrained message slots per row
    K_DEG: int        # polynomial degree bound        (ELL  <= K_DEG)
    N_LIG: int        # codeword length / # of columns  (K_DEG <= N_LIG)
    T_QUERIES: int    # number of opened columns
    # Example production values: Config(ELL=8192, K_DEG=16384, N_LIG=65536, T_QUERIES=80)
    # For fast single-layer dev, drop T_QUERIES to ~4 (see notes on the W·T cost).

    @property
    def W_K(self): return pow(GLOBAL_G, (P - 1) // self.K_DEG, P)   # K-th root of unity
    @property
    def W_N(self): return pow(GLOBAL_G, (P - 1) // self.N_LIG, P)   # N-th root of unity

    def zeta(self, c: int) -> int:
        """Message point ζ_c = ω_K^c (c-th K-th root of unity), c in [0, ELL)."""
        return pow(self.W_K, c, P)

    def eta(self, j: int) -> int:
        """Codeword point η_j = γ·ω_N^j on a coset (γ = GLOBAL_G), disjoint from ζ."""
        return (GLOBAL_G * pow(self.W_N, j, P)) % P

# Row layout: rows 0..2 of the joint witness matrix are ZK blinding rows;
# witness rows start at row NUM_BLINDING_ROWS.
NUM_BLINDING_ROWS = 3
BLIND_IRS, BLIND_LIN, BLIND_QUAD = 0, 1, 2

# ----------------------------------------------------------------------
# Polynomial evaluation (the only "poly math" the verifier needs)
# ----------------------------------------------------------------------
def poly_eval(coeffs: List[int], x: int) -> int:
    """Evaluate a polynomial given by ascending coefficients at x (Horner)."""
    acc = 0
    for c in reversed(coeffs):
        acc = (acc * x + c) % P
    return acc

def lagrange(cfg: Config, c: int, eta: int) -> int:
    """L_c(η): the contribution of a value placed at message point ζ_c to a
    polynomial's value at codeword point η, for the size-K_DEG roots-of-unity
    domain (the unused K−ELL message slots are 0):
        L_c(η) = ζ_c · (η^K − 1) / (K · (η − ζ_c)).
    This closed form is what lets the verifier evaluate r^T A rows and the a/b
    vectors of quadratic constraints with no NTT.
    """
    K = cfg.K_DEG
    zc = cfg.zeta(c)
    return mul(mul(zc, sub(pow(eta, K, P), 1)), inv(mul(K % P, sub(eta, zc))))

def eval_zeta_form(cfg: Config, values: List[int], eta: int) -> int:
    """Evaluate a message-domain value vector at η:  Σ_c values[c] · L_c(η)."""
    acc = 0
    for c, v in enumerate(values):
        if v:
            acc = (acc + mul(v, lagrange(cfg, c, eta))) % P
    return acc

# ----------------------------------------------------------------------
# Merkle commitments over columns
# ----------------------------------------------------------------------
EMPTY_COMMIT_ROOT = b"\x00" * 32        # sentinel for a commit with no rows

def pack_column(col: List[int]) -> bytes:
    return b"".join(int(v).to_bytes(8, "little") for v in col)

def merkle_leaf(col: List[int]) -> bytes:
    return blake3.blake3(pack_column(col)).digest()

def merkle_verify(leaf: bytes, path: List[Tuple[bytes, int]], root: bytes) -> bool:
    """path: list of (sibling, side); side==0 → sibling is the left child."""
    h = leaf
    for sibling, side in path:
        h = blake3.blake3((sibling + h) if side == 0 else (h + sibling)).digest()
    return h == root

# ----------------------------------------------------------------------
# Challenge expansion (deterministic from a seed; both sides agree).
# Test driver feeds a random seed; production feeds a Fiat-Shamir hash of the
# transcript so far. The expansion logic is identical either way.
# ----------------------------------------------------------------------
def _seed_bytes(seed) -> bytes:
    """Normalize a seed to 32 bytes. Accepts bytes (interactive: the verifier's
    per-round coins / a transcript digest) or an int (test convenience) — one
    definition both prover and verifier share."""
    if isinstance(seed, (bytes, bytearray)):
        return bytes(seed).ljust(32, b"\x00")[:32]
    return int(seed).to_bytes(32, "little")

def challenge(seed, index: int, label: str = "") -> int:
    """The (label, index)-th field challenge for `seed` — a hash-based PRF, so
    it is O(1)-indexable: both sides derive the same value without storing the
    whole vector. `label` namespaces independent families ("irs"/"lin"/"quad")
    drawn from one seed. `seed` may be bytes (interactive coins / digest) or int."""
    h = blake3.blake3(_seed_bytes(seed) + label.encode()
                      + index.to_bytes(8, "little")).digest()
    return int.from_bytes(h[:16], "little") % P     # 128-bit reduce → bias ~2^-64

def random_columns_n(seed, count: int, range_max: int) -> List[int]:
    """`count` distinct indices in [0, range_max), sorted — the shared column
    sampler. blake3 rejection sampling so it is trivially portable (bit-exact in
    the Rust verifier): draw index k = challenge(seed, k, "col") mod range_max,
    skip duplicates, until `count` distinct. `seed` may be bytes or int.
    (Was numpy PCG64 — replaced so the whole verifier is blake3-only.)"""
    seen, out, k = set(), [], 0
    while len(out) < count:
        j = challenge(seed, k, "col") % range_max
        if j not in seen:
            seen.add(j); out.append(j)
        k += 1
    return sorted(out)

def random_columns(seed, cfg: Config) -> List[int]:
    """T_QUERIES distinct column indices in [0, N_LIG). Small set — fine to
    materialize; only the O(W)-sized per-constraint combiners are lazy."""
    return random_columns_n(seed, cfg.T_QUERIES, cfg.N_LIG)

def op_vec(s_op, claim_index: int, label: str, n: int) -> List[int]:
    """The op-challenge vector for one claim, derived by index from the round-1
    seed s_op:  v[i] = challenge(s_op, i, f"op{claim_index}:{label}").

    This is THE shared op-challenge primitive — prover and verifier both call it,
    so neither has to send the other any challenge values; both expand the same
    seed identically. The (claim_index, label) namespace keeps every op's vector
    independent: matmul claim 3's ρ is "op3:rho", its λ is "op3:lam", etc., so
    two matmuls (or a matmul and an rmsnorm) never collide. claim_index is the
    claim's position in the SETTLED list (ops then table settlements) — the one
    order both sides agree on."""
    return [challenge(s_op, i, f"op{claim_index}:{label}") for i in range(n)]

def round_seeds(base, n: int = 3) -> List[bytes]:
    """Derive n per-round verifier seeds from one base seed. Both prover and
    verifier must derive these identically, so it is shared/trusted here.

    This is a TEST / Fiat-Shamir convenience: in a live INTERACTIVE protocol
    each round's seed is the verifier's fresh coins, sent after that round's
    commitment (NOT derived from a base). For Fiat-Shamir, pass a transcript
    digest as `base` — the expansion is identical either way."""
    b = _seed_bytes(base)
    return [blake3.blake3(b + b"ligero-round" + bytes([r])).digest()
            for r in range(n)]

# ----------------------------------------------------------------------
# Claim introspection helpers (shared by _distinct_tables / claims_to_json).
# ----------------------------------------------------------------------
def _is_var(v):
    return hasattr(v, "row_start") and hasattr(v, "length")

def _obj_vars(cl):
    """vars() that works for slots dataclasses too (claims/packets gained
    slots=True for memory — 129M packet objects at 48-layer scale)."""
    if hasattr(cl, "__dict__"):
        return vars(cl)
    import dataclasses
    if dataclasses.is_dataclass(cl):
        return {f.name: getattr(cl, f.name) for f in dataclasses.fields(cl)}
    return {k: getattr(cl, k) for k in getattr(cl, "__slots__", ())}


def _distinct_tables(claim_list):
    """Tables referenced by the ops, first-seen order, deduped by identity.
    Mirrors core.py:_collect_tables so prover/verifier settle in the same order
    (cids are positional). A table is any public field exposing T + mult_var."""
    seen, out = set(), []
    for cl in claim_list:
        for v in _obj_vars(cl).values():
            if hasattr(v, "T") and hasattr(v, "mult_var") and id(v) not in seen:
                seen.add(id(v)); out.append(v)
    return out




# ======================================================================
# Claim serialization → JSON, for the standalone Rust verifier.
#
# Dumps the PUBLIC claim structure (op type + each Variable's [row_start, length]
# + table contents/layout + config scalars) — never the witness (tape.inputs).
# The Rust verifier reads this, deserializes into its own claim structs, and runs
# its OWN compile (handlers + expanders), so it trusts nothing from this dump
# beyond the public structure a real deployment would ship anyway.
#
# Generic by introspection (like _walk_vars): a Variable field → [row_start,
# length]; a Table field → {T, T_Y, mult_var, w_var, z_vars, alpha, beta}; a
# config → its scalar fields; a list[Variable] → list of [row_start, length];
# scalars pass through. So it tracks the handlers without hand-listing fields.
# ======================================================================
def _ser_var(v):
    return [v.row_start, v.length]

def _ser_table(t, _cache=None):
    # The table's key domain T is (verified, check_tables.py) always
    # range(T_LEN): the verifier reconstructs T[j]=j, so we transmit only the
    # length, not the array. This is the difference between a ~tens-of-MB and a
    # multi-GB proof when 2^24 range tables appear. Only non-range tables (none
    # currently) fall back to an explicit "T". T_Y (paired function values) stays
    # explicit but is small.
    #
    # One shared table (e.g. the 2^24 LogUp range table) is referenced by
    # hundreds of claims. Memoize on its stable id (mult_var.row_start, unique
    # per table) so we materialize + range-check the 16.7M-element T once, not
    # once per referencing claim — the latter turned the proof dump into ~18 min.
    key = t.mult_var.row_start
    if _cache is not None and key in _cache:
        return _cache[key]
    Tl = [int(x) for x in (t.T.tolist() if hasattr(t.T, "tolist") else t.T)]
    n = len(Tl)
    is_range = all(v == i for i, v in enumerate(Tl))
    out = {
        # Stable id for dedup across claims (JSON loses Python object identity).
        # mult_var.row_start is unique per table — the Rust verifier dedups
        # shared tables by it, mirroring Python's id()-based _distinct_tables.
        "id":     t.mult_var.row_start,
        "T_len":  n,
        "T_range": is_range,
        "T_Y":    (None if t.T_Y is None else
                   [int(x) for x in (t.T_Y.tolist() if hasattr(t.T_Y, "tolist") else t.T_Y)]),
        "mult_var": _ser_var(t.mult_var),
        "w_var":    _ser_var(t.w_var),
        "z_vars":   [_ser_var(z) for z in t.z_vars],
        "alpha": int(t.alpha),
        "beta":  int(t.beta),
    }
    if not is_range:                       # fallback: explicit domain (unused today)
        out["T"] = Tl
    if _cache is not None:
        _cache[key] = out
    return out

def _ser_value(v, _tbl_cache=None):
    """Serialize one claim field by its kind (None / Variable / Table / list /
    config / bool / str / scalar)."""
    if v is None:
        return {"none": True}
    if _is_var(v):
        return {"var": _ser_var(v)}
    if hasattr(v, "mult_var") and hasattr(v, "w_var"):          # Table
        return {"table": _ser_table(v, _tbl_cache)}
    if isinstance(v, (list, tuple)):
        return {"list": [_ser_value(it, _tbl_cache) for it in v]}
    import dataclasses as _dc
    if ((hasattr(v, "__dict__") or _dc.is_dataclass(v))
            and not isinstance(v, (int, float, bool))):
        # a config object (RmsNormConfig / SiluConfig / SoftmaxConfig / RoPEConfig)
        # — detected by __dict__ OR is_dataclass (slots=True configs lost
        # __dict__; falling through to int() crashed the proof dump):
        # dump its scalar fields, plus derived rescale_bits/output_rescale_bits/
        # s_x/magic which the handlers read as properties. Floats are kept as
        # floats (RoPEConfig.base — dropping it silently pinned the Rust
        # verifier to θ=10000 for every model).
        d = {k: (float(x) if isinstance(x, float) and not isinstance(x, bool)
                 else int(x))
             for k, x in _obj_vars(v).items() if isinstance(x, (int, float, bool))}
        for prop in ("rescale_bits", "output_rescale_bits", "s_x", "magic"):
            if hasattr(type(v), prop):
                try: d[prop] = int(getattr(v, prop))
                except Exception: pass
        return {"config": d}
    if isinstance(v, bool):
        return {"bool": v}
    if isinstance(v, str):
        return {"str": v}
    return {"scalar": int(v)}

def claims_to_json(claim_list, cfg: Config) -> dict:
    """The OPERATION claim list (tape.claims — NOT settled) as a JSON-able dict.
    The Rust verifier consumes this exactly as compile_claims does: it discovers
    + settles tables itself, indexing op challenges by op position then table
    position. Order matters; pass tape.claims in order. Tables carry a stable
    `id` (= mult_var.row_start) so the Rust side dedups shared tables the way
    Python's id()-based _distinct_tables does."""
    tbl_cache = {}          # share one serialized dict per table id (see _ser_table)
    out = []
    for cl in claim_list:
        fields = {k: _ser_value(v, tbl_cache) for k, v in _obj_vars(cl).items()}
        out.append({"op": type(cl).__name__, "fields": fields})
    # Explicit settle order (= _distinct_tables, by table id) so the Rust side
    # need not re-derive it from field-iteration order (which JSON does not
    # preserve). The settlement at settled-index n_ops+k settles table_order[k].
    table_order = [t.mult_var.row_start for t in _distinct_tables(claim_list)]
    return {"cfg": {"ELL": cfg.ELL, "K_DEG": cfg.K_DEG,
                    "N_LIG": cfg.N_LIG, "T_QUERIES": cfg.T_QUERIES},
            "claims": out,
            "table_order": table_order}
