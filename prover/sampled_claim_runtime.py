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
import secrets
import sys
import time
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Sequence

import blake3
import compute_fns
import core as prover_core
import protocol
import sampled_local_proofs
import torch
from claims import (
    AddClaim,
    ConcatClaim,
    HadamardClaim,
    LinCombClaim,
    MatmulClaim,
    RangeWordClaim,
    WordExtractionClaim,
)
from core import STREAMING_INPUT_CLAIMS, P, Variable
from cuda_primitives import gl_add, gl_mul, hash_columns_streamed
from rescale_claim import RescaleClaim

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


def _sampling_tree_root(leaves: Sequence[bytes]) -> bytes:
    """The portable verifier's exact window-root convention.

    Selection and the subsequent local challenge must bind the same committed
    window.  C0 has a separate domain above, so keep this transcript root
    explicit instead of accidentally reusing it.
    """
    if not leaves:
        return _b3(b"verinf/audit/empty/v1")
    cur = list(leaves)
    while len(cur) > 1:
        cur = [_b3(b"verinf/audit/node/v1", cur[i],
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


def _tensor_digest_parts(tensor: torch.Tensor):
    flat = tensor.detach().contiguous().view(-1)
    if flat.dtype != torch.uint64:
        flat = flat.to(torch.uint64)
    n = int(flat.numel())
    # Keep enough lanes for GPU occupancy without emitting 128 KiB of leaves
    # for every small wire. Grow toward 4096 at about 32 KiB per column.
    columns = min(4096, max(min(1024, n), (n + 4095) // 4096))
    padded_n = ((n + columns - 1) // columns) * columns
    if padded_n != n:
        padded = torch.zeros(padded_n, dtype=torch.uint64, device=flat.device)
        padded[:n] = flat
        flat = padded
    matrix = flat.view(-1, columns)
    if matrix.is_cuda:
        payload = matrix
    else:
        # Slow portability path; production tensors are CUDA-resident.
        array = matrix.numpy()
        payload = b"".join(
            blake3.blake3(memoryview(array[:, j].copy()).cast("B")).digest()
            for j in range(columns))
    return n, columns, payload


def _finish_tensor_digest(n: int, columns: int, leaf_bytes) -> bytes:
    return _b3(b"verinf/claim-audit/striped-wire/v1",
               n.to_bytes(8, "little"), columns.to_bytes(8, "little"),
               bytes(leaf_bytes))


def _host_copy(tensor: torch.Tensor) -> torch.Tensor:
    flat = tensor.detach().contiguous().view(-1)
    if not flat.is_cuda:
        return flat
    host = torch.empty(flat.shape, dtype=flat.dtype, device="cpu",
                       pin_memory=True)
    host.copy_(flat, non_blocking=False)
    return host


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    aa = a.detach().contiguous().view(-1)
    bb = b.detach().contiguous().view(-1)
    if aa.device != bb.device:
        bb = bb.to(aa.device)
    return torch.equal(aa, bb)


@dataclass
class _Block:
    index: int
    claim: object
    variables: List[Variable]
    commitment: bytes


class _HashingColumnSink(prover_core.ColumnSink):
    """Land selected columns on host and hash them while still on the GPU."""

    def __init__(self, n_rows: int, columns: List[int], row_base: int = 0):
        super().__init__(n_rows, columns, row_base)
        self._digest_acc = prover_core._make_merkle_acc(len(columns), n_rows)

    def write(self, abs_row: int, chunk: torch.Tensor,
              columns: List[int]) -> None:
        self._digest_acc.update(chunk.contiguous())
        super().write(abs_row, chunk, columns)

    def finish_with_digests(self):
        opened = super().finish()
        raw = self._digest_acc.finalize().cpu().numpy()
        return opened, [bytes(row.tolist()) for row in raw]


class ClaimWindowAudit:
    """Observer passed directly to `Tape.run_engine_pass`."""

    def __init__(self, tape, cfg, verifier_secret: bytes, public_io_digest: bytes,
                 model_root: bytes, *, expected_claims: int = 2596,
                 window_size: int = 49, sample_per_window: int = 5,
                 progress_path: str | None = None, heartbeat_every: int = 25,
                 enable_rs_binding: bool = False, rs_columns: int = 61,
                 rs_ell: int | None = None, rs_k_deg: int | None = None,
                 rs_n_lig: int | None = None):
        self.tape = tape
        self.cfg = cfg
        self.params = AuditParams(len(tape.claims), window_size,
                                  sample_per_window, rs_columns)
        self.total_windows = ((len(tape.claims) + window_size - 1) //
                              window_size)
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
        # Fail closed at construction time: a newly introduced claim cannot
        # silently inherit the exact-recomputation fallback.
        self.manifest_proof_family_counts = (
            sampled_local_proofs.manifest_family_counts(tape.claims))
        self.wire_digests: Dict[Variable, bytes] = {}
        self.pending_digests: Dict[Variable, tuple] = {}
        self.wire_order: List[Variable] = []
        self.window_values: Dict[Variable, torch.Tensor] = {}
        # Foldable MoE expert streams are produced long before their combine
        # claim. Retain host copies only until that combine's window is sampled;
        # the engine remains free to fold and release the GPU tensors.
        self.fold_values: Dict[Variable, torch.Tensor] = {}
        self.fold_last_use: Dict[Variable, int] = {}
        for claim_index, (deferred_claim, _inputs, _side_effects) in enumerate(
                tape._deferred):
            fold_fn = compute_fns.FOLD_FNS.get(type(deferred_claim))
            if fold_fn is not None:
                for var in fold_fn["foldable"](deferred_claim):
                    if not var.persistent and not var.external:
                        self.fold_last_use[var] = claim_index
        self.fold_retained_bytes = 0
        self.fold_retained_peak_bytes = 0
        self.window_blocks: List[_Block] = []
        self.block_commitments: List[bytes] = []
        self.selected: List[int] = []
        self.failures: List[str] = []
        self.selected_proof_family_counts = Counter()
        self.materialized_local_proof_counts = Counter()
        self.exact_fallback_counts = Counter()
        self.local_proof_bytes = 0
        self.local_proof_digests: List[dict] = []
        self.local_receipts: List[dict] = []
        self.rs_column_samples: List[dict] = []
        self.commit_s = 0.0
        self.local_s = 0.0
        self.enable_rs_binding = bool(enable_rs_binding)
        self.rs_cfg = prover_core.LigeroConfig(
            ELL=cfg.ELL if rs_ell is None else int(rs_ell),
            K_DEG=cfg.K_DEG if rs_k_deg is None else int(rs_k_deg),
            N_LIG=cfg.N_LIG if rs_n_lig is None else int(rs_n_lig),
            T_QUERIES=rs_columns)
        if (self.enable_rs_binding
                and self.rs_cfg.N_LIG < self.params.rs_columns):
            raise ValueError("RS N_LIG is smaller than requested columns")
        if (self.enable_rs_binding
                and self.rs_cfg.K_DEG - self.rs_cfg.ELL <= rs_columns):
            raise ValueError("RS padding must exceed opened-column count")
        self.rs_seed_t = (prover_core._master_seed_to_cuda(secrets.token_bytes(32))
                          if self.enable_rs_binding else None)
        self.rs_row_cursor = 0
        self.rs_opened_values = 0
        self.rs_roots: List[bytes] = []
        self.rs_commit_s = 0.0
        self.rs_open_s = 0.0
        self.rs_verify_s = 0.0
        self.started_at = time.perf_counter()
        self.claim_started_at = self.started_at
        self.heartbeat_every = max(1, int(heartbeat_every))
        self.progress_path = pathlib.Path(progress_path) if progress_path else None
        self.progress_file = None
        if self.progress_path is not None:
            self.progress_path.parent.mkdir(parents=True, exist_ok=True)
            self.progress_file = self.progress_path.open("w", buffering=1)
        self._progress("audit_start", claims=len(tape.claims),
                       fold_wires=len(self.fold_last_use),
                       window_size=self.params.window_size,
                       sample_per_window=self.params.sample_per_window,
                       rs_binding=self.enable_rs_binding,
                       rs_geometry={"ELL": self.rs_cfg.ELL,
                                    "K_DEG": self.rs_cfg.K_DEG,
                                    "N_LIG": self.rs_cfg.N_LIG,
                                    "columns": self.params.rs_columns})

    def _gpu_memory(self) -> dict:
        if not torch.cuda.is_available():
            return {}
        return {
            "gpu_allocated_bytes": int(torch.cuda.memory_allocated()),
            "gpu_reserved_bytes": int(torch.cuda.memory_reserved()),
        }

    def _progress(self, event: str, **fields) -> None:
        payload = {
            "event": event,
            "elapsed_s": time.perf_counter() - self.started_at,
            **fields,
            **self._gpu_memory(),
        }
        if self.progress_file is not None:
            self.progress_file.write(json.dumps(payload, sort_keys=True) + "\n")
            self.progress_file.flush()

    def before_claim(self, index: int, claim) -> None:
        """Engine hook written before a claim, including one that times out."""
        self.claim_started_at = time.perf_counter()
        claim_type = type(claim).__name__
        self._progress("claim_start", claim=index, claim_type=claim_type)
        if index % self.heartbeat_every == 0:
            print(f"[sampled-audit-progress] claim_start {index + 1}/"
                  f"{len(self.tape.claims)} type={claim_type} "
                  f"elapsed={self.claim_started_at - self.started_at:.1f}s",
                  file=sys.stderr, flush=True)

    def _value(self, var, input_data, outs, live):
        value = outs.get(var, input_data.get(var, live.get(var)))
        return value() if callable(value) and not var.persistent else value

    def _wire_ref(self, var, input_data, outs, live) -> bytes:
        if var.persistent or var.external:
            return _b3(b"verinf/claim-audit/model-wire/v1", self.model_root,
                       var.name.encode(), int(var.length).to_bytes(8, "little"))
        if var in self.wire_digests or var in self.pending_digests:
            if var not in self.window_values:
                retained = self.fold_values.get(var)
                if retained is not None:
                    self.window_values[var] = retained
                else:
                    value = self._value(var, input_data, outs, live)
                    if value is None or callable(value):
                        raise RuntimeError(
                            f"{var.name}: phase-1 value is not resident")
                    self.window_values[var] = value.detach()
            return self.wire_digests.get(var, b"")
        value = self._value(var, input_data, outs, live)
        if value is None or callable(value):
            raise RuntimeError(f"{var.name}: phase-1 value is not resident")
        # Launch the striped GPU hash now, but batch the single D2H
        # synchronization for all wire leaves when this window is fixed.
        self.pending_digests[var] = _tensor_digest_parts(value)
        self.window_values[var] = value.detach()
        return b""

    def __call__(self, index, claim, input_vars, input_data, outs, live):
        del input_vars
        t0 = time.perf_counter()
        variables = [v for v in _claim_vars(claim) if getattr(v, "phase", 1) == 1]
        for var in variables:
            self._wire_ref(var, input_data, outs, live)
        for var in variables:
            if var in self.fold_last_use and var not in self.fold_values:
                retained = _host_copy(self.window_values[var])
                self.fold_values[var] = retained
                self.fold_retained_bytes += retained.numel() * retained.element_size()
                self.fold_retained_peak_bytes = max(
                    self.fold_retained_peak_bytes, self.fold_retained_bytes)
        self.window_blocks.append(_Block(index, claim, variables, b""))
        self.commit_s += time.perf_counter() - t0
        if len(self.window_blocks) == self.params.window_size:
            self._flush_window()
        done_at = time.perf_counter()
        self._progress(
            "claim_complete", claim=index, claim_type=type(claim).__name__,
            claim_wall_s=done_at - self.claim_started_at,
            pre_observer_s=max(0.0, t0 - self.claim_started_at),
            observer_s=done_at - t0, commit_total_s=self.commit_s,
            local_checks_total_s=self.local_s,
            rs_commit_total_s=self.rs_commit_s,
            rs_open_total_s=self.rs_open_s,
            rs_verify_total_s=self.rs_verify_s,
            fold_retained_bytes=self.fold_retained_bytes)

    def _finalize_pending_digests(self) -> None:
        if not self.pending_digests:
            return
        items = list(self.pending_digests.items())
        groups = {}
        leaf_by_var = {}
        for var, (_n, _columns, payload) in items:
            if isinstance(payload, torch.Tensor):
                groups.setdefault(int(payload.shape[0]), []).append((var, payload))
            else:
                leaf_by_var[var] = payload
        for entries in groups.values():
            combined = torch.cat([matrix for _var, matrix in entries], dim=1)
            combined_leaves = hash_columns_streamed(combined)
            offset = 0
            for var, matrix in entries:
                columns = int(matrix.shape[1])
                leaf_by_var[var] = combined_leaves[offset:offset + columns]
                offset += columns
        gpu_leaves = [leaf_by_var[var].reshape(-1) for var, _parts in items
                      if isinstance(leaf_by_var[var], torch.Tensor)]
        host_bytes = None
        if gpu_leaves:
            joined = torch.cat(gpu_leaves)
            host_bytes = memoryview(joined.cpu().numpy()).cast("B")
        offset = 0
        for var, (n, columns, _payload) in items:
            leaves = leaf_by_var[var]
            if isinstance(leaves, torch.Tensor):
                size = columns * 32
                leaf_bytes = host_bytes[offset:offset + size]
                offset += size
            else:
                leaf_bytes = leaves
            self.wire_digests[var] = _finish_tensor_digest(
                n, columns, leaf_bytes)
            self.wire_order.append(var)
        self.pending_digests.clear()

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

    def _exact_check(self, block: _Block, live) -> tuple[bool, str]:
        claim = block.claim
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

    def _check(self, block: _Block, challenge: bytes) -> tuple[bool, str]:
        claim = block.claim
        claim_name = type(claim).__name__
        family = sampled_local_proofs.proof_family(claim)
        self.selected_proof_family_counts[family] += 1
        live = self._device_live(block)

        if isinstance(claim, MatmulClaim):
            proof = sampled_local_proofs.prove_matmul(
                claim, live, claim_index=block.index, challenge=challenge)
            ok, why = sampled_local_proofs.verify_matmul(
                claim, live, proof, claim_index=block.index,
                challenge=challenge)
            self.materialized_local_proof_counts[family] += 1
            self.local_proof_bytes += proof.byte_size
            self.local_proof_digests.append({
                "claim": block.index,
                "claim_type": claim_name,
                "family": family,
                "bytes": proof.byte_size,
                "digest": proof.digest.hex(),
            })
            if not ok:
                return False, why
            # The Freivalds contraction binds the raw matmul.  A legacy fused
            # rescale also carries rounding/range relations which do not yet
            # have the sumcheck bridge, so check only that residual part via
            # the visibly counted fallback.
            if claim.rescale_bits > 0:
                self.exact_fallback_counts[claim_name] += 1
                return self._exact_check(block, live)
            return True, "ok"

        if isinstance(claim, (AddClaim, ConcatClaim, HadamardClaim,
                              LinCombClaim, RescaleClaim,
                              WordExtractionClaim)):
            proof = sampled_local_proofs.prove_sumcheck(
                claim, live, claim_index=block.index, challenge=challenge)
            ok, why = sampled_local_proofs.verify_sumcheck(
                claim, live, proof, claim_index=block.index,
                challenge=challenge)
            self.materialized_local_proof_counts[family] += 1
            self.local_proof_bytes += proof.byte_size
            self.local_proof_digests.append({
                "claim": block.index,
                "claim_type": claim_name,
                "family": family,
                "bytes": proof.byte_size,
                "digest": proof.digest.hex(),
            })
            if not ok:
                return False, why
            needs_residual_exact = (
                isinstance(claim, RescaleClaim)
                or (isinstance(claim, HadamardClaim)
                    and claim.rescale_bits > 0))
            if needs_residual_exact:
                self.exact_fallback_counts[claim_name] += 1
                return self._exact_check(block, live)
            return True, "ok"

        self.exact_fallback_counts[claim_name] += 1
        return self._exact_check(block, live)

    def _rs_window_commit(self):
        """Commit the current window with the production RS/Merkle encoder."""
        if not self.enable_rs_binding:
            return None, [], []
        variables = list(self.window_values)
        row_base = self.rs_row_cursor
        layout = []
        cursor = row_base
        for var in variables:
            rows = var.n_rows(self.rs_cfg.ELL)
            layout.append((var, cursor, rows))
            cursor += rows
        n_rows = cursor - row_base
        t0 = time.perf_counter()
        accumulator = prover_core._make_merkle_acc(self.rs_cfg.N_LIG, n_rows)
        prover_core._stream_phase(
            variables, self.window_values, self.rs_cfg,
            master_seed=self.rs_seed_t, abs_row_offset=row_base,
            merkle_acc=accumulator)
        artifact = prover_core._finalize_merkle_artifact(accumulator)
        elapsed = time.perf_counter() - t0
        self.rs_commit_s += elapsed
        self.rs_row_cursor = cursor
        self.rs_roots.append(artifact.root)
        return artifact, layout, variables

    def _rs_window_open_and_verify(self, artifact, layout, variables,
                                   selected, window_index):
        """Open post-local-proof RS columns and bind selected wire messages."""
        if not self.enable_rs_binding or not layout:
            return
        row_base = layout[0][1] if layout else self.rs_row_cursor
        n_rows = sum(rows for _var, _start, rows in layout)
        selected_set = set(selected)
        receipts = [receipt for receipt in self.local_receipts
                    if receipt["claim"] in selected_set]
        transcript = _b3(
            b"verinf/claim-audit/window-local/v1",
            window_index.to_bytes(8, "little"),
            b"".join(self.block_commitments[i] for i in selected),
            b"".join(bytes.fromhex(receipt["digest"])
                     for receipt in receipts))
        columns = self.verifier._columns(
            self.statement_digest, self.window_blocks[0].index, transcript,
            self.params.rs_columns, self.rs_cfg.N_LIG)
        self.rs_column_samples.append({
            "window": window_index,
            "local_receipts": len(receipts),
            "transcript_digest": transcript.hex(),
            "columns": columns,
        })

        t0 = time.perf_counter()
        sink = _HashingColumnSink(n_rows, columns, row_base)
        prover_core._stream_phase(
            variables, self.window_values, self.rs_cfg,
            master_seed=self.rs_seed_t, abs_row_offset=row_base,
            columns_at=columns, column_sink=sink)
        opened, opened_digests = sink.finish_with_digests()
        self.rs_opened_values += n_rows * len(columns)
        self.rs_open_s += time.perf_counter() - t0

        t0 = time.perf_counter()
        for k, column in enumerate(columns):
            path = prover_core.merkle_path(artifact.levels, column)
            if not prover_core.merkle_verify(
                    opened_digests[k], path, artifact.root):
                self.failures.append(
                    f"window {window_index}: RS Merkle opening failed at {column}")

        selected_vars = []
        seen = set()
        for block in self.window_blocks:
            if block.index not in selected_set:
                continue
            for var in block.variables:
                if (not var.persistent and not var.external
                        and var not in seen):
                    seen.add(var)
                    selected_vars.append(var)
        positions = {var: (start, rows) for var, start, rows in layout}
        for var in selected_vars:
            start, rows = positions[var]
            expected_sink = prover_core.ColumnSink(rows, columns, start)
            prover_core._stream_phase(
                [var], {var: self.window_values[var]}, self.rs_cfg,
                master_seed=self.rs_seed_t, abs_row_offset=start,
                columns_at=columns, column_sink=expected_sink)
            expected = expected_sink.finish()
            lo = start - row_base
            for column in columns:
                if not torch.equal(expected[column],
                                   opened[column][lo:lo + rows]):
                    self.failures.append(
                        f"window {window_index}: {var.name} is not RS-bound")
                    break
        self.rs_verify_s += time.perf_counter() - t0

    def _flush_window(self):
        if not self.window_blocks:
            return
        window_index = self.window_blocks[0].index // self.params.window_size
        commit_t0 = time.perf_counter()
        self._finalize_pending_digests()
        rs_artifact, rs_layout, rs_variables = self._rs_window_commit()
        rs_root = rs_artifact.root if rs_artifact is not None else b""
        commitments = []
        for block in self.window_blocks:
            refs = {}
            for var in block.variables:
                if var.persistent or var.external:
                    ref = _b3(b"verinf/claim-audit/model-wire/v1",
                              self.model_root, var.name.encode(),
                              int(var.length).to_bytes(8, "little"))
                else:
                    ref = self.wire_digests[var]
                refs[var.name] = ref.hex()
            block.commitment = _b3(
                b"verinf/claim-audit/block/v1",
                block.index.to_bytes(8, "little"),
                _canonical(self.claim_descriptors[block.index]),
                _canonical(refs), rs_root)
            commitments.append(block.commitment)
            self.block_commitments.append(block.commitment)
        self.commit_s += time.perf_counter() - commit_t0
        selected = self.verifier._window_selection(
            self.statement_digest, window_index, commitments, self.params)
        window_root = _sampling_tree_root(commitments)
        by_index = {b.index: b for b in self.window_blocks}
        t0 = time.perf_counter()
        for index in selected:
            challenge = self.verifier._challenge(
                self.statement_digest, window_root, index)
            ok, why = self._check(by_index[index], challenge)
            self.selected.append(index)
            proof_record = next(
                (record for record in reversed(self.local_proof_digests)
                 if record["claim"] == index), None)
            proof_digest = (bytes.fromhex(proof_record["digest"])
                            if proof_record is not None else b"")
            receipt = _b3(
                b"verinf/claim-audit/local-receipt/v1",
                index.to_bytes(8, "little"), challenge,
                by_index[index].commitment, proof_digest,
                sampled_local_proofs.proof_family(
                    by_index[index].claim).encode(),
                b"accepted" if ok else b"rejected", why.encode())
            self.local_receipts.append({
                "claim": index,
                "proof_digest": proof_record["digest"]
                if proof_record is not None else None,
                "digest": receipt.hex(),
            })
            if not ok:
                self.failures.append(f"claim {index}: {why}")
        window_check_s = time.perf_counter() - t0
        self.local_s += window_check_s
        self._rs_window_open_and_verify(
            rs_artifact, rs_layout, rs_variables, selected, window_index)
        torch.cuda.empty_cache()
        window_last = self.window_blocks[-1].index
        for var in [v for v, last in self.fold_last_use.items()
                    if last <= window_last and v in self.fold_values]:
            retained = self.fold_values.pop(var)
            self.fold_retained_bytes -= retained.numel() * retained.element_size()
        self._progress(
            "window_complete", window=window_index,
            first_claim=self.window_blocks[0].index,
            last_claim=window_last, selected=selected, check_s=window_check_s,
            commit_total_s=self.commit_s,
            local_checks_total_s=self.local_s,
            rs_commit_total_s=self.rs_commit_s,
            rs_open_total_s=self.rs_open_s,
            rs_verify_total_s=self.rs_verify_s,
            rs_rows_total=self.rs_row_cursor,
            rs_opened_values=self.rs_opened_values,
            materialized_local_proofs=sum(
                self.materialized_local_proof_counts.values()),
            exact_fallbacks=sum(self.exact_fallback_counts.values()),
            fold_retained_bytes=self.fold_retained_bytes,
            failures=len(self.failures))
        print(f"[sampled-audit-progress] window_complete {window_index + 1}/"
              f"{self.total_windows} claims={self.window_blocks[0].index + 1}-"
              f"{window_last + 1} selected={len(selected)} "
              f"check={window_check_s:.1f}s "
              f"elapsed={time.perf_counter() - self.started_at:.1f}s",
              file=sys.stderr, flush=True)
        self.window_values.clear()
        self.window_blocks.clear()

    def finish(self) -> dict:
        self._flush_window()
        leaves = [_b3(b"verinf/claim-audit/wire/v1", var.name.encode(),
                      self.wire_digests[var]) for var in self.wire_order]
        witness_root = _tree_root(leaves)
        if self.enable_rs_binding:
            c0 = _b3(b"verinf/claim-audit/C0/v1", self.statement_digest,
                     self.model_root, witness_root, _tree_root(self.rs_roots))
        else:
            c0 = _b3(b"verinf/claim-audit/C0/v1", self.statement_digest,
                     self.model_root, witness_root)
        materialized = sum(self.materialized_local_proof_counts.values())
        exact_fallbacks = sum(self.exact_fallback_counts.values())
        materialized_families = set(self.materialized_local_proof_counts)
        proof_label = "+".join(sorted(materialized_families))
        if materialized and exact_fallbacks:
            local_argument = f"{proof_label}+exact-recomputation"
        elif materialized:
            local_argument = proof_label
        else:
            local_argument = "exact-recomputation"
        binding_argument = (
            "exact-local" if local_argument == "exact-recomputation"
            else local_argument)
        result = {
            "kind": "sampled-claim-audit-v1",
            "claims": len(self.tape.claims),
            "selected": len(self.selected),
            "selected_indices": self.selected,
            "fraction": len(self.selected) / len(self.tape.claims),
            "window": self.params.window_size,
            "sample_per_window": self.params.sample_per_window,
            "rs_columns": self.params.rs_columns,
            "rs_geometry": {"ELL": self.rs_cfg.ELL,
                            "K_DEG": self.rs_cfg.K_DEG,
                            "N_LIG": self.rs_cfg.N_LIG},
            "statement_digest": self.statement_digest.hex(),
            "model_root": self.model_root.hex(),
            "c0_root": c0.hex(),
            "commit_s": self.commit_s,
            "local_checks_s": self.local_s,
            "rs_commit_s": self.rs_commit_s,
            "rs_open_s": self.rs_open_s,
            "rs_verify_s": self.rs_verify_s,
            "rs_rows": self.rs_row_cursor,
            "rs_opened_values": self.rs_opened_values,
            "fold_retained_peak_bytes": self.fold_retained_peak_bytes,
            "manifest_proof_family_counts": self.manifest_proof_family_counts,
            "selected_proof_family_counts": dict(sorted(
                self.selected_proof_family_counts.items())),
            "materialized_local_proof_counts": dict(sorted(
                self.materialized_local_proof_counts.items())),
            "materialized_local_proofs": materialized,
            "exact_fallback_counts": dict(sorted(
                self.exact_fallback_counts.items())),
            "exact_fallbacks": exact_fallbacks,
            "local_proof_bytes": self.local_proof_bytes,
            "local_proof_digests": self.local_proof_digests,
            "local_receipts": self.local_receipts,
            "rs_column_samples": self.rs_column_samples,
            "accepted": not self.failures,
            "failures": self.failures,
            "binding": (("rs-window+striped-blake3 " if self.enable_rs_binding
                         else "striped-blake3 ") + binding_argument + " runtime"),
            "local_argument": local_argument,
            "cryptographic_local_proofs": False,
            "cryptographic_local_proof_coverage": (
                materialized / len(self.selected) if self.selected else 0.0),
            "rs_openings_materialized": self.enable_rs_binding,
        }
        self._progress("audit_complete", accepted=result["accepted"],
                       selected=result["selected"], failures=len(self.failures))
        if self.progress_file is not None:
            self.progress_file.close()
            self.progress_file = None
        return result


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
