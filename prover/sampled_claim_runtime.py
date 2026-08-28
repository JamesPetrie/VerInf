"""One-pass sampled audit adapter for the real :class:`tape.Tape`.

One existing Tape claim is one audit block.  The adapter is synchronous with
`run_engine_pass`: it hashes each phase-1 wire while resident, buffers only the
current 49-claim window on host, asks the persistent verifier for its 5 claims,
checks those claims, and releases the window.  No semantic model replay occurs.

This adapter is the real-run bridge.  The portable proof objects, RS openings
and algebraic local arguments live in `layergkr.sampled_audit`; here selected
claims are checked exactly by their registered phase-1 computation plus explicit
handlers for constraint-only claims.  Consequently this runtime is suitable for
the Vast timing campaign, but its selected plaintext host buffer is not the
zero-knowledge wire format.
"""
from __future__ import annotations

import json
import pathlib
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence

import blake3
import compute_fns
import protocol
import torch
from claims import AddClaim, LinCombClaim, RangeWordClaim
from core import STREAMING_INPUT_CLAIMS, P, Variable
from cuda_primitives import gl_add, gl_mul, hash_columns_streamed

from layergkr.sampled_audit import AuditParams, VerifierSession


def _b3(domain: bytes, *parts: bytes) -> bytes:
    h = blake3.blake3(domain)
    for part in parts:
        h.update(len(part).to_bytes(8, "little"))
        h.update(part)
    return h.digest()


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _tree_root(leaves: Sequence[bytes]) -> bytes:
    if not leaves:
        return _b3(b"verinf/claim-audit/empty/v1")
    cur = list(leaves)
    while len(cur) > 1:
        cur = [_b3(b"verinf/claim-audit/node/v1", cur[i],
                   cur[i + 1] if i + 1 < len(cur) else cur[i])
               for i in range(0, len(cur), 2)]
    return cur[0]


def _claim_vars(claim) -> List[Variable]:
    out, seen = [], set()

    def walk(value):
        if protocol._is_var(value):
            if id(value) not in seen:
                seen.add(id(value))
                out.append(value)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)

    for value in protocol._obj_vars(claim).values():
        walk(value)
    return out


def _tensor_digest(tensor: torch.Tensor) -> bytes:
    flat = tensor.detach().contiguous().view(-1)
    if flat.dtype != torch.uint64:
        flat = flat.to(torch.uint64)
    if flat.is_cuda:
        digest = hash_columns_streamed(flat.view(-1, 1))[0]
        return bytes(digest.cpu().tolist())
    return blake3.blake3(flat.numpy().tobytes()).digest()


def _host_copy(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().contiguous().view(-1).cpu()


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(a.detach().contiguous().view(-1).cpu(),
                       b.detach().contiguous().view(-1).cpu())


@dataclass
class _Block:
    index: int
    claim: object
    variables: List[Variable]
    commitment: bytes


class ClaimWindowAudit:
    """Observer passed directly to `Tape.run_engine_pass`."""

    def __init__(self, tape, cfg, verifier_secret: bytes, public_io_digest: bytes,
                 model_root: bytes, *, expected_claims: int = 2596,
                 window_size: int = 49, sample_per_window: int = 5):
        self.tape = tape
        self.cfg = cfg
        self.params = AuditParams(len(tape.claims), window_size,
                                  sample_per_window, 61)
        if expected_claims and len(tape.claims) != expected_claims:
            raise ValueError(
                f"production manifest has {len(tape.claims)} claims, "
                f"expected {expected_claims}")
        manifest = protocol.claims_to_json(tape.claims, cfg)
        self.claim_descriptors = manifest["claims"]
        self.statement_digest = _b3(
            b"verinf/claim-audit/statement/v1", _canonical(manifest),
            bytes(public_io_digest), bytes(model_root))
        self.model_root = bytes(model_root)
        self.verifier = VerifierSession(verifier_secret)
        self.wire_digests: Dict[Variable, bytes] = {}
        self.wire_order: List[Variable] = []
        self.window_values: Dict[Variable, torch.Tensor] = {}
        self.window_blocks: List[_Block] = []
        self.block_commitments: List[bytes] = []
        self.selected: List[int] = []
        self.failures: List[str] = []
        self.commit_s = 0.0
        self.local_s = 0.0

    def _value(self, var, input_data, outs, live):
        value = outs.get(var, input_data.get(var, live.get(var)))
        return value() if callable(value) and not var.persistent else value

    def _wire_ref(self, var, input_data, outs, live) -> bytes:
        if var.persistent or var.external:
            return _b3(b"verinf/claim-audit/model-wire/v1", self.model_root,
                       var.name.encode(), int(var.length).to_bytes(8, "little"))
        value = self._value(var, input_data, outs, live)
        if value is None or callable(value):
            raise RuntimeError(f"{var.name}: phase-1 value is not resident")
        if var not in self.wire_digests:
            digest = _tensor_digest(value)
            self.wire_digests[var] = digest
            self.wire_order.append(var)
        if var not in self.window_values:
            self.window_values[var] = _host_copy(value)
        return self.wire_digests[var]

    def __call__(self, index, claim, input_vars, input_data, outs, live):
        del input_vars
        t0 = time.perf_counter()
        variables = [v for v in _claim_vars(claim) if getattr(v, "phase", 1) == 1]
        refs = {v.name: self._wire_ref(v, input_data, outs, live).hex()
                for v in variables}
        commitment = _b3(
            b"verinf/claim-audit/block/v1", index.to_bytes(8, "little"),
            _canonical(self.claim_descriptors[index]), _canonical(refs))
        self.block_commitments.append(commitment)
        self.window_blocks.append(_Block(index, claim, variables, commitment))
        self.commit_s += time.perf_counter() - t0
        if len(self.window_blocks) == self.params.window_size:
            self._flush_window()

    def _device_live(self, block: _Block):
        live = {}
        for var in block.variables:
            host = self.window_values.get(var)
            if host is not None:
                live[var] = host.to("cuda")
                continue
            value = self.tape.inputs.get(var)
            if callable(value) and type(block.claim) not in STREAMING_INPUT_CLAIMS:
                value = value()
            if value is not None:
                live[var] = value
        return live

    def _constraint_only(self, claim, live) -> tuple[bool, str]:
        if isinstance(claim, AddClaim) and claim.public_rhs is not None:
            want = torch.full_like(live[claim.a], int(claim.public_rhs) % P)
            return _same(live[claim.a], want), "public reveal pin"
        if isinstance(claim, LinCombClaim):
            acc = torch.zeros(claim.length, dtype=torch.uint64, device="cuda")
            for var, coef in zip(claim.xs, claim.coefs):
                co = torch.full_like(live[var], int(coef) % P)
                acc = gl_add(acc, gl_mul(live[var], co))
            rhs = claim.rhs
            if len(rhs) == 1:
                want = torch.full_like(acc, int(rhs[0]) % P)
            else:
                want = torch.tensor(rhs, dtype=torch.uint64, device="cuda")
            return _same(acc, want), "linear combination"
        if isinstance(claim, RangeWordClaim):
            x = live[claim.x].view(torch.int64)
            table = claim.table.T
            n = int(table.numel())
            # Production range tables are canonical 0..n-1. Non-range tables
            # take the general torch.isin path.
            probe = table[:min(n, 1024)].view(torch.int64)
            if torch.equal(probe, torch.arange(probe.numel(), device="cuda")):
                ok = bool(((x >= 0) & (x < n)).all().item())
            else:
                ok = bool(torch.isin(x, table.view(torch.int64)).all().item())
            return ok, "range lookup"
        return True, "no phase-1 outputs"

    def _check(self, block: _Block) -> tuple[bool, str]:
        claim = block.claim
        live = self._device_live(block)
        fn = compute_fns.COMPUTE_FNS.get(type(claim))
        if fn is None:
            return False, f"no local handler for {type(claim).__name__}"
        if type(claim) in STREAMING_INPUT_CLAIMS:
            recomputed = fn(claim, live, None)
        else:
            recomputed = fn(claim, live)
        if not recomputed:
            return self._constraint_only(claim, live)
        for var, expected in recomputed.items():
            actual = self.window_values.get(var)
            if actual is None:
                return False, f"{var.name}: output missing from C0 window"
            if not _same(expected, actual):
                return False, f"{var.name}: local operation mismatch"
        return True, "ok"

    def _flush_window(self):
        if not self.window_blocks:
            return
        window_index = self.window_blocks[0].index // self.params.window_size
        commitments = [b.commitment for b in self.window_blocks]
        selected = self.verifier._window_selection(
            self.statement_digest, window_index, commitments, self.params)
        by_index = {b.index: b for b in self.window_blocks}
        t0 = time.perf_counter()
        for index in selected:
            ok, why = self._check(by_index[index])
            self.selected.append(index)
            if not ok:
                self.failures.append(f"claim {index}: {why}")
        torch.cuda.empty_cache()
        self.local_s += time.perf_counter() - t0
        self.window_values.clear()
        self.window_blocks.clear()

    def finish(self) -> dict:
        self._flush_window()
        leaves = [_b3(b"verinf/claim-audit/wire/v1", var.name.encode(),
                      self.wire_digests[var]) for var in self.wire_order]
        witness_root = _tree_root(leaves)
        c0 = _b3(b"verinf/claim-audit/C0/v1", self.statement_digest,
                 self.model_root, witness_root)
        return {
            "kind": "sampled-claim-audit-v1",
            "claims": len(self.tape.claims),
            "selected": len(self.selected),
            "selected_indices": self.selected,
            "fraction": len(self.selected) / len(self.tape.claims),
            "window": self.params.window_size,
            "sample_per_window": self.params.sample_per_window,
            "rs_columns": self.params.rs_columns,
            "statement_digest": self.statement_digest.hex(),
            "model_root": self.model_root.hex(),
            "c0_root": c0.hex(),
            "commit_s": self.commit_s,
            "local_checks_s": self.local_s,
            "accepted": not self.failures,
            "failures": self.failures,
            "binding": "raw-blake3 exact-local runtime",
        }


def load_secret(path: str) -> bytes:
    data = pathlib.Path(path).read_bytes().strip()
    try:
        decoded = bytes.fromhex(data.decode())
        if len(decoded) >= 16:
            return decoded
    except (UnicodeDecodeError, ValueError):
        pass
    if len(data) < 16:
        raise ValueError("verifier secret file must contain >=16 raw bytes or hex")
    return data
