"""Window-sampled local audit over one globally committed witness.

The protocol is deliberately interactive.  A verifier samples a window only
after every block commitment in that window is fixed, and samples RS columns
only after the selected block's local transcript is fixed.  The witness is a
set of uniquely named wires; blocks refer to those names, so adjacent blocks
cannot use unrelated copies of the same activation.

This module is the executable, prototype-scale protocol.  It reuses the
production Goldilocks field, RS encoder and Merkle convention from ``rs.py``
and the existing sumcheck implementation.  Selected wire messages currently
travel in the clear (as do terminal vectors in the Layer-GKR prototype); the
61-column binding and all local checks are real, but this module by itself is
not zero knowledge.
"""
from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Sequence, Tuple, Union

import blake3

from prover.protocol import P as FIELD_P
from prover.protocol import merkle_leaf, merkle_verify

from . import rs
from . import sumcheck as sc
from .logup import eq_vector


def _b3(domain: bytes, *parts: bytes) -> bytes:
    h = blake3.blake3(domain)
    for part in parts:
        h.update(len(part).to_bytes(8, "little"))
        h.update(part)
    return h.digest()


def _jsonable(value):
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, bytes):
        return {"__bytes__": value.hex()}
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda x: str(x[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _canonical(value) -> bytes:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":")).encode()


def _is_pow2(n: int) -> bool:
    return n > 0 and not (n & (n - 1))


def _field_stream(seed: bytes, label: bytes, count: int) -> List[int]:
    return [int.from_bytes(_b3(b"verinf/audit/field/v1", seed, label,
                               i.to_bytes(8, "little"))[:16], "little") % FIELD_P
            for i in range(count)]


def _tree(leaves: Sequence[bytes]) -> List[List[bytes]]:
    if not leaves:
        return [[_b3(b"verinf/audit/empty/v1")]]
    levels = [list(leaves)]
    while len(levels[-1]) > 1:
        cur = levels[-1]
        levels.append([_b3(b"verinf/audit/node/v1", cur[i],
                           cur[i + 1] if i + 1 < len(cur) else cur[i])
                       for i in range(0, len(cur), 2)])
    return levels


def _path(levels: Sequence[Sequence[bytes]], index: int) -> List[Tuple[bytes, int]]:
    out = []
    for level in levels[:-1]:
        sibling = index ^ 1
        if sibling >= len(level):
            sibling = index
        out.append((level[sibling], 1 if index & 1 == 0 else 0))
        index //= 2
    return out


def _path_root(leaf: bytes, path: Sequence[Tuple[bytes, int]]) -> bytes:
    cur = leaf
    for sibling, sibling_is_right in path:
        cur = (_b3(b"verinf/audit/node/v1", cur, sibling) if sibling_is_right
               else _b3(b"verinf/audit/node/v1", sibling, cur))
    return cur


@dataclass(frozen=True)
class AuditParams:
    total_blocks: int = 2596
    window_size: int = 49
    sample_per_window: int = 5
    rs_columns: int = 61

    def validate(self) -> None:
        if self.total_blocks <= 0 or self.window_size <= 0:
            raise ValueError("block and window counts must be positive")
        if not 0 < self.sample_per_window <= self.window_size:
            raise ValueError("sample_per_window must be in [1, window_size]")
        if self.rs_columns <= 0:
            raise ValueError("rs_columns must be positive")

    def sample_count(self, window_len: int) -> int:
        """Proportional tail rounding keeps its per-block probability >= 5/49."""
        if window_len <= 0 or window_len > self.window_size:
            raise ValueError("invalid window length")
        return min(window_len, math.ceil(self.sample_per_window * window_len /
                                         self.window_size))

    @property
    def selected_blocks(self) -> int:
        full, tail = divmod(self.total_blocks, self.window_size)
        return full * self.sample_per_window + (self.sample_count(tail) if tail else 0)

    @property
    def audit_fraction(self) -> float:
        return self.selected_blocks / self.total_blocks

    @property
    def min_detection_probability(self) -> float:
        full, tail = divmod(self.total_blocks, self.window_size)
        probs = [self.sample_per_window / self.window_size] if full else []
        if tail:
            probs.append(self.sample_count(tail) / tail)
        return min(probs)


@dataclass
class Wire:
    wire_id: str
    messages: List[List[int]]

    def __post_init__(self):
        if not self.wire_id:
            raise ValueError("empty wire id")
        if not self.messages:
            raise ValueError(f"wire {self.wire_id}: no messages")
        self.messages = [[int(x) % FIELD_P for x in row] for row in self.messages]
        if any(not row for row in self.messages):
            raise ValueError(f"wire {self.wire_id}: empty message row")

    def schema(self) -> dict:
        return {"wire_id": self.wire_id, "row_widths": [len(r) for r in self.messages]}


@dataclass
class MatmulBlock:
    index: int
    name: str
    x: Wire
    w: Wire
    y: Wire
    kind: str = field(init=False, default="matmul")

    def __post_init__(self):
        m, n_in, n_out = len(self.x.messages), len(self.x.messages[0]), len(self.w.messages)
        if any(len(r) != n_in for r in self.x.messages + self.w.messages):
            raise ValueError("matmul X/W row width mismatch")
        if len(self.y.messages) != m or any(len(r) != n_out for r in self.y.messages):
            raise ValueError("matmul Y shape mismatch")

    @property
    def wires(self) -> Dict[str, Wire]:
        return {"x": self.x, "w": self.w, "y": self.y}

    def descriptor(self) -> dict:
        return {"index": self.index, "name": self.name, "kind": self.kind,
                "wires": {k: v.schema() for k, v in self.wires.items()}}


@dataclass
class RelationBlock:
    """A zero relation proved by sumcheck, at EVERY coordinate.

    ``terms`` contains ``(coefficient, [wire-role, ...])``.  Each role names a
    one-row wire of the same power-of-two length.  This covers Adam updates,
    deterministic rounding/rescale, routing booleanity and simple nonlinear
    polynomial identities without making any of them a trusted prover opcode.
    The residual is weighted by eq(z, .) at a point z drawn after the wires
    are committed, so the sumcheck's zero claim is the residual's multilinear
    extension at z: errors at two coordinates no longer cancel.
    """
    index: int
    name: str
    family: str
    relation_wires: Dict[str, Wire]
    terms: List[Tuple[int, List[str]]]
    kind: str = field(init=False, default="sumcheck")

    def __post_init__(self):
        if not self.terms:
            raise ValueError("empty relation")
        widths = {len(w.messages[0]) for w in self.relation_wires.values()}
        if any(len(w.messages) != 1 for w in self.relation_wires.values()):
            raise ValueError("relation wires must have one row")
        if len(widths) != 1 or not _is_pow2(next(iter(widths))):
            raise ValueError("relation domain must be one shared power of two")
        for _, roles in self.terms:
            if not roles or any(r not in self.relation_wires for r in roles):
                raise ValueError("relation term references an unknown/empty role")

    @property
    def wires(self) -> Dict[str, Wire]:
        return self.relation_wires

    def descriptor(self) -> dict:
        return {"index": self.index, "name": self.name, "kind": self.kind,
                "family": self.family,
                "wires": {k: v.schema() for k, v in sorted(self.wires.items())},
                "terms": [(int(c) % FIELD_P, list(r)) for c, r in self.terms]}


@dataclass
class LookupBlock:
    """Tuple lookup checked by a random fingerprint and two product trees."""
    index: int
    name: str
    queries: Wire
    multiplicities: Wire
    table: List[Tuple[int, ...]]
    table_id: int = 0
    kind: str = field(init=False, default="lookup")

    def __post_init__(self):
        if not self.table:
            raise ValueError("empty lookup table")
        width = len(self.table[0])
        if width == 0 or any(len(t) != width for t in self.table):
            raise ValueError("ragged lookup table")
        if any(len(q) != width for q in self.queries.messages):
            raise ValueError("query/table tuple width mismatch")
        if (len(self.multiplicities.messages) != 1
                or len(self.multiplicities.messages[0]) != len(self.table)):
            raise ValueError("one multiplicity is required per table row")
        self.table = [tuple(int(x) % FIELD_P for x in t) for t in self.table]

    @property
    def wires(self) -> Dict[str, Wire]:
        return {"queries": self.queries, "multiplicities": self.multiplicities}

    def descriptor(self) -> dict:
        return {"index": self.index, "name": self.name, "kind": self.kind,
                "table_id": self.table_id, "table": self.table,
                "wires": {k: v.schema() for k, v in self.wires.items()}}


AuditBlock = Union[MatmulBlock, RelationBlock, LookupBlock]


@dataclass(frozen=True)
class AuditStatement:
    params: AuditParams
    public_io_digest: bytes
    descriptors: Tuple[dict, ...]

    @classmethod
    def from_blocks(cls, params: AuditParams, public_io_digest: bytes,
                    blocks: Sequence[AuditBlock]) -> "AuditStatement":
        params.validate()
        if len(blocks) != params.total_blocks:
            raise ValueError(f"expected {params.total_blocks} blocks, got {len(blocks)}")
        if [b.index for b in blocks] != list(range(len(blocks))):
            raise ValueError("block indices must be canonical 0..n-1")
        return cls(params, bytes(public_io_digest), tuple(b.descriptor() for b in blocks))

    @property
    def digest(self) -> bytes:
        return _b3(b"verinf/audit/statement/v1", _canonical(self.params),
                   self.public_io_digest, _canonical(self.descriptors))


@dataclass(frozen=True)
class WireRecord:
    wire_id: str
    root: bytes
    row_widths: Tuple[int, ...]


@dataclass
class WireOpening:
    wire_id: str
    messages: List[List[int]]
    columns: List[int]
    values: List[List[int]]
    paths: List[List[Tuple[bytes, int]]]
    c0_path: List[Tuple[bytes, int]] = field(default_factory=list)


@dataclass
class MatmulLocalProof:
    w_projection: List[int]
    y_projection: List[int]


@dataclass
class RelationLocalProof:
    sumcheck: sc.SumcheckProof


@dataclass
class LookupLocalProof:
    query_tree: List[List[int]]
    table_tree: List[List[int]]


LocalProof = Union[MatmulLocalProof, RelationLocalProof, LookupLocalProof]


@dataclass
class SelectedBlockProof:
    index: int
    challenge: bytes
    local: LocalProof
    openings: Dict[str, WireOpening]
    local_digest: bytes


@dataclass
class AuditProof:
    statement_digest: bytes
    c0_root: bytes
    wire_records: List[WireRecord]
    block_commitments: List[bytes]
    selected: List[SelectedBlockProof]


class VerifierSession:
    """Persistent verifier secret used for post-commit interactive coins."""

    def __init__(self, secret: bytes):
        if len(secret) < 16:
            raise ValueError("verifier secret must contain at least 128 bits")
        self._secret = bytes(secret)

    def _window_selection(self, statement_digest: bytes, window_index: int,
                          window_commitments: Sequence[bytes], params: AuditParams
                          ) -> List[int]:
        root = _tree(window_commitments)[-1][0]
        start = window_index * params.window_size
        k = params.sample_count(len(window_commitments))
        tagged = [(_b3(b"verinf/audit/select/v1", self._secret, statement_digest,
                       window_index.to_bytes(8, "little"), root,
                       j.to_bytes(4, "little")), start + j)
                  for j in range(len(window_commitments))]
        return sorted(index for _, index in sorted(tagged)[:k])

    def _challenge(self, statement_digest: bytes, window_root: bytes,
                   block_index: int) -> bytes:
        return _b3(b"verinf/audit/challenge/v1", self._secret, statement_digest,
                   window_root, block_index.to_bytes(8, "little"))

    def _columns(self, statement_digest: bytes, block_index: int,
                 local_digest: bytes, q: int, n_lig: int) -> List[int]:
        seed = _b3(b"verinf/audit/columns/v1", self._secret, statement_digest,
                   block_index.to_bytes(8, "little"), local_digest)
        return rs.sample_columns(seed, q, n_lig)


@dataclass
class _CommittedWire:
    record: WireRecord
    commit: rs.Commit
    messages: List[List[int]]


def _wire_leaf(record: WireRecord) -> bytes:
    return _b3(b"verinf/audit/wire/v1", record.wire_id.encode(), record.root,
               _canonical(record.row_widths))


def _block_commitment(index: int, descriptor: dict,
                      wire_roots: Mapping[str, bytes]) -> bytes:
    return _b3(b"verinf/audit/block/v1", index.to_bytes(8, "little"),
               _b3(b"verinf/audit/descriptor/v1", _canonical(descriptor)),
               _canonical({k: v.hex() for k, v in sorted(wire_roots.items())}))


def _product_tree(values: Sequence[int]) -> List[List[int]]:
    n = 1
    while n < len(values):
        n *= 2
    levels = [list(values) + [1] * (n - len(values))]
    while len(levels[-1]) > 1:
        cur = levels[-1]
        levels.append([(cur[i] * cur[i + 1]) % FIELD_P for i in range(0, len(cur), 2)])
    return levels


def _check_product_tree(tree: Sequence[Sequence[int]], leaves: Sequence[int]) -> bool:
    expected = _product_tree(leaves)
    return [[int(x) % FIELD_P for x in level] for level in tree] == expected


def _tuple_fp(table_id: int, values: Sequence[int], beta: int) -> int:
    acc = table_id % FIELD_P
    for value in values:
        acc = (acc * beta + int(value)) % FIELD_P
    return acc


def _relation_terms(terms, messages: Mapping[str, Sequence[Sequence[int]]],
                    challenge: bytes):
    """The relation's sumcheck terms, each weighted by eq(z, .) at z drawn
    from the block's post-commit challenge (the same z on both sides)."""
    width = len(messages[terms[0][1][0]][0])
    z = _field_stream(challenge, b"relation-eq", width.bit_length() - 1)
    eq = eq_vector(z)
    return [(int(c) % FIELD_P, [messages[role][0] for role in roles] + [eq])
            for c, roles in terms]


def _relation_transcript(challenge: bytes, index: int,
                         block_commitment: bytes) -> sc.RoundTranscript:
    """Per-round coins bound to the post-commit block challenge, the block and
    its commitment (which covers the relation's descriptor and wire roots);
    each round's coin follows that round's polynomial."""
    return sc.RoundTranscript(b"verinf/audit/relation/v1", challenge,
                              index.to_bytes(8, "little"), block_commitment)


def _prove_local(block: AuditBlock, challenge: bytes,
                 block_commitment: bytes) -> LocalProof:
    if isinstance(block, MatmulBlock):
        n_out = len(block.w.messages)
        rho = _field_stream(challenge, b"freivalds-rho", n_out)
        wp = [sum(block.w.messages[i][j] * rho[i] for i in range(n_out)) % FIELD_P
              for j in range(len(block.w.messages[0]))]
        yp = [sum(row[i] * rho[i] for i in range(n_out)) % FIELD_P
              for row in block.y.messages]
        return MatmulLocalProof(wp, yp)
    if isinstance(block, RelationBlock):
        terms = _relation_terms(block.terms,
                                {r: w.messages for r, w in block.wires.items()},
                                challenge)
        return RelationLocalProof(sc.prove_terms(
            terms, _relation_transcript(challenge, block.index, block_commitment)))
    beta, alpha = _field_stream(challenge, b"lookup-beta-alpha", 2)
    q = [_tuple_fp(block.table_id, row, beta) for row in block.queries.messages]
    mult = [int(x) for x in block.multiplicities.messages[0]]
    t = []
    for row, count in zip(block.table, mult):
        t.extend([_tuple_fp(block.table_id, row, beta)] * count)
    return LookupLocalProof(_product_tree([(alpha - x) % FIELD_P for x in q]),
                            _product_tree([(alpha - x) % FIELD_P for x in t]))


def _local_digest(index: int, challenge: bytes, local: LocalProof,
                  wires: Mapping[str, Wire]) -> bytes:
    payload = {role: {"id": wire.wire_id, "messages": wire.messages}
               for role, wire in sorted(wires.items())}
    return _b3(b"verinf/audit/local/v1", index.to_bytes(8, "little"), challenge,
               _canonical(local), _canonical(payload))


def prove(blocks: Sequence[AuditBlock], cfg: rs.Config, statement: AuditStatement,
          verifier: VerifierSession) -> AuditProof:
    """Commit one window, receive its secret sample, prove it, then release it.

    The Python object list is convenient for the prototype.  The commitment and
    challenge order is window-streaming: a production adapter may yield blocks
    from the real forward pass and retain only the current 49.
    """
    statement.params.validate()
    if cfg.N_LIG < statement.params.rs_columns:
        raise ValueError("N_LIG is smaller than the requested RS column count")
    if len(blocks) != statement.params.total_blocks:
        raise ValueError("block count does not match the statement")
    if tuple(b.descriptor() for b in blocks) != statement.descriptors:
        raise ValueError("blocks do not match the public manifest")

    registry: Dict[str, _CommittedWire] = {}
    block_coms: List[bytes] = []
    selected_pending: List[Tuple[SelectedBlockProof, Dict[str, rs.Commit]]] = []
    p = statement.params

    for window_index, start in enumerate(range(0, len(blocks), p.window_size)):
        window = list(blocks[start:start + p.window_size])
        window_coms = []
        commits_by_block: List[Dict[str, rs.Commit]] = []
        for block in window:
            role_commits = {}
            role_roots = {}
            for role, wire in block.wires.items():
                old = registry.get(wire.wire_id)
                if old is None:
                    commit = rs.Commit.from_messages(cfg, wire.messages)
                    record = WireRecord(wire.wire_id, commit.root,
                                        tuple(len(r) for r in wire.messages))
                    registry[wire.wire_id] = _CommittedWire(record, commit, wire.messages)
                else:
                    if old.messages != wire.messages:
                        raise ValueError(f"wire {wire.wire_id!r} was rebound")
                    commit, record = old.commit, old.record
                role_commits[role] = registry[wire.wire_id].commit
                role_roots[role] = record.root
            bc = _block_commitment(block.index, block.descriptor(), role_roots)
            block_coms.append(bc)
            window_coms.append(bc)
            commits_by_block.append(role_commits)

        selected = verifier._window_selection(statement.digest, window_index,
                                               window_coms, p)
        window_root = _tree(window_coms)[-1][0]
        for absolute_index in selected:
            local_index = absolute_index - start
            block = window[local_index]
            challenge = verifier._challenge(statement.digest, window_root, block.index)
            local = _prove_local(block, challenge, window_coms[local_index])
            ld = _local_digest(block.index, challenge, local, block.wires)
            columns = verifier._columns(statement.digest, block.index, ld,
                                        p.rs_columns, cfg.N_LIG)
            openings = {}
            for role, wire in block.wires.items():
                commit = commits_by_block[local_index][role]
                vals, paths = [], []
                for col in columns:
                    v, path = commit.open(col)
                    vals.append(v)
                    paths.append(path)
                openings[role] = WireOpening(wire.wire_id, wire.messages, columns,
                                             vals, paths)
            selected_pending.append((SelectedBlockProof(block.index, challenge, local,
                                                         openings, ld),
                                     commits_by_block[local_index]))

    # C0 order is a function of the public manifest, never Python dict order.
    wire_order = []
    for descriptor in statement.descriptors:
        for schema in descriptor["wires"].values():
            if schema["wire_id"] not in wire_order:
                wire_order.append(schema["wire_id"])
    records = [registry[wire_id].record for wire_id in wire_order]
    wire_levels = _tree([_wire_leaf(r) for r in records])
    wire_root = wire_levels[-1][0]
    c0 = _b3(b"verinf/audit/C0/v1", statement.digest, wire_root)
    positions = {r.wire_id: i for i, r in enumerate(records)}
    selected_proofs = []
    for sbp, _ in selected_pending:
        for opening in sbp.openings.values():
            opening.c0_path = _path(wire_levels, positions[opening.wire_id])
        selected_proofs.append(sbp)
    return AuditProof(statement.digest, c0, records, block_coms, selected_proofs)


def _verify_local(descriptor: dict, local: LocalProof,
                  openings: Mapping[str, WireOpening], challenge: bytes,
                  index: int, block_commitment: bytes) -> Tuple[bool, str]:
    messages = {role: op.messages for role, op in openings.items()}
    kind = descriptor["kind"]
    if kind == "matmul":
        if not isinstance(local, MatmulLocalProof):
            return False, "wrong local proof type for matmul"
        X, W, Y = messages["x"], messages["w"], messages["y"]
        rho = _field_stream(challenge, b"freivalds-rho", len(W))
        wp = [sum(W[i][j] * rho[i] for i in range(len(W))) % FIELD_P
              for j in range(len(W[0]))]
        yp = [sum(row[i] * rho[i] for i in range(len(W))) % FIELD_P for row in Y]
        if wp != local.w_projection or yp != local.y_projection:
            return False, "Freivalds projections do not match the committed wires"
        for t, row in enumerate(X):
            if sum(row[j] * wp[j] for j in range(len(wp))) % FIELD_P != yp[t]:
                return False, "Freivalds contraction failed"
        return True, "ok"
    if kind == "sumcheck":
        if not isinstance(local, RelationLocalProof):
            return False, "wrong local proof type for sumcheck"
        # the local argument is unmasked: a carried mask would be a free
        # term the prover could set to cancel the terminal value
        if local.sumcheck.masked:
            return False, "masked transcript in an unmasked local argument"
        terms = _relation_terms(descriptor["terms"], messages, challenge)
        if local.sumcheck.claim % FIELD_P != 0:
            return False, "sumcheck relation claim is not zero"
        ok, why = sc.verify_terms(local.sumcheck, terms,
                                  _relation_transcript(challenge, index,
                                                       block_commitment))
        return (ok, why if not ok else "ok")
    if not isinstance(local, LookupLocalProof):
        return False, "wrong local proof type for lookup"
    beta, alpha = _field_stream(challenge, b"lookup-beta-alpha", 2)
    queries = messages["queries"]
    mult = [int(x) for x in messages["multiplicities"][0]]
    if any(x < 0 for x in mult) or sum(mult) != len(queries):
        return False, "invalid lookup multiplicities"
    q = [_tuple_fp(int(descriptor["table_id"]), row, beta) for row in queries]
    t = []
    for row, count in zip(descriptor["table"], mult):
        t.extend([_tuple_fp(int(descriptor["table_id"]), row, beta)] * count)
    ql = [(alpha - x) % FIELD_P for x in q]
    tl = [(alpha - x) % FIELD_P for x in t]
    if not _check_product_tree(local.query_tree, ql):
        return False, "invalid query product tree"
    if not _check_product_tree(local.table_tree, tl):
        return False, "invalid table product tree"
    if local.query_tree[-1][0] != local.table_tree[-1][0]:
        return False, "lookup product roots differ"
    return True, "ok"


def verify(proof: AuditProof, cfg: rs.Config, statement: AuditStatement,
           verifier: VerifierSession) -> Tuple[bool, str]:
    p = statement.params
    if proof.statement_digest != statement.digest:
        return False, "statement digest mismatch"
    if len(proof.block_commitments) != p.total_blocks:
        return False, "wrong block commitment count"
    expected_wire_ids = []
    schemas = {}
    for desc in statement.descriptors:
        for schema in desc["wires"].values():
            wid = schema["wire_id"]
            schemas.setdefault(wid, tuple(schema["row_widths"]))
            if schemas[wid] != tuple(schema["row_widths"]):
                return False, f"wire {wid}: inconsistent public schema"
            if wid not in expected_wire_ids:
                expected_wire_ids.append(wid)
    if [r.wire_id for r in proof.wire_records] != expected_wire_ids:
        return False, "C0 wire set/order differs from the public manifest"
    if any(r.row_widths != schemas[r.wire_id] for r in proof.wire_records):
        return False, "C0 wire schema mismatch"

    wire_leaves = [_wire_leaf(r) for r in proof.wire_records]
    wire_root = _tree(wire_leaves)[-1][0]
    if proof.c0_root != _b3(b"verinf/audit/C0/v1", statement.digest, wire_root):
        return False, "C0 root mismatch"
    record_by_id = {r.wire_id: r for r in proof.wire_records}

    # Every sampling commitment is compiler-derived from the fixed descriptor
    # and C0 wire roots. Checking only selected blocks would leave the prover an
    # unnecessary knob in the commitments that determine its secret sample.
    for index, desc in enumerate(statement.descriptors):
        roots = {role: record_by_id[schema["wire_id"]].root
                 for role, schema in desc["wires"].items()}
        expected = _block_commitment(index, desc, roots)
        if proof.block_commitments[index] != expected:
            return False, f"block {index}: commitment/manifest mismatch"

    expected_selected = []
    window_roots = {}
    for wi, start in enumerate(range(0, p.total_blocks, p.window_size)):
        wc = proof.block_commitments[start:start + p.window_size]
        window_roots[wi] = _tree(wc)[-1][0]
        expected_selected.extend(verifier._window_selection(statement.digest, wi, wc, p))
    if [s.index for s in proof.selected] != expected_selected:
        return False, "selected block set/order is not the verifier sample"

    for selected in proof.selected:
        desc = statement.descriptors[selected.index]
        wi = selected.index // p.window_size
        challenge = verifier._challenge(statement.digest, window_roots[wi], selected.index)
        if challenge != selected.challenge:
            return False, f"block {selected.index}: challenge mismatch"
        if set(selected.openings) != set(desc["wires"]):
            return False, f"block {selected.index}: wire roles mismatch"
        wire_roots = {}
        for role, schema in desc["wires"].items():
            opening = selected.openings[role]
            if opening.wire_id != schema["wire_id"]:
                return False, f"block {selected.index}: wire id mismatch"
            record = record_by_id[opening.wire_id]
            if _path_root(_wire_leaf(record), opening.c0_path) != wire_root:
                return False, f"block {selected.index}: wire is not in C0"
            if len(opening.columns) != p.rs_columns or len(set(opening.columns)) != p.rs_columns:
                return False, f"block {selected.index}: wrong RS column count"
            wire_roots[role] = record.root
        bc = _block_commitment(selected.index, desc, wire_roots)
        if bc != proof.block_commitments[selected.index]:
            return False, f"block {selected.index}: commitment/manifest mismatch"

        ld = _b3(b"verinf/audit/local/v1", selected.index.to_bytes(8, "little"),
                 challenge, _canonical(selected.local),
                 _canonical({role: {"id": op.wire_id, "messages": op.messages}
                             for role, op in sorted(selected.openings.items())}))
        if ld != selected.local_digest:
            return False, f"block {selected.index}: local transcript digest mismatch"
        columns = verifier._columns(statement.digest, selected.index, ld,
                                    p.rs_columns, cfg.N_LIG)
        for role, opening in selected.openings.items():
            record = record_by_id[opening.wire_id]
            if opening.columns != columns:
                return False, f"block {selected.index}: RS columns were not post-proof coins"
            if not (len(opening.values) == len(opening.paths) == len(columns)):
                return False, f"block {selected.index}: malformed RS openings"
            if [len(r) for r in opening.messages] != list(record.row_widths):
                return False, f"block {selected.index}: message schema mismatch"
            reencoded = [rs.encode_row(cfg, row) for row in opening.messages]
            for k, col in enumerate(columns):
                values = opening.values[k]
                if not merkle_verify(merkle_leaf(values), opening.paths[k], record.root,
                                     col, cfg.N_LIG):
                    return False, f"block {selected.index}: Merkle opening failed"
                if values != [row[col] for row in reencoded]:
                    return False, f"block {selected.index}: local witness is not bound to C0"

        ok, why = _verify_local(desc, selected.local, selected.openings, challenge,
                                selected.index, bc)
        if not ok:
            return False, f"block {selected.index}: {why}"
    return True, "ok"


def detection_probability(params: AuditParams, repetitions: int = 1) -> float:
    """Lower bound for one fixed first-bad block over independent audits."""
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    return 1.0 - (1.0 - params.min_detection_probability) ** repetitions
