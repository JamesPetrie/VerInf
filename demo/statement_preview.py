# Copyright (c) 2026 Levi Neuwirth. Released under the MIT License (see LICENSE).
"""Write a tape's statement before proving it.

Two files: the canonical claim bytes a proof of this tape would carry, and a
JSON sidecar with the commitment-block order the prover would choose. A policy
can then be pinned from the statement before any proof exists, and the proof's
claims compared with it byte for byte afterwards.

This follows prove_streaming's own order (core.py): synthesize the table
settlements, run _layout (which assigns the row starts that table ids are
read from), then take pr.claims_canonical_bytes over tape.claims. The block
order is core's rule over the same layout.
"""
import fcntl
import hashlib
import json
import os
import tempfile
from contextlib import ExitStack
from pathlib import Path

import core
import protocol as pr


def block_order(layout, cfg):
    (_all, _p1, _p2, p3_vars, _m_p1, _m_p2, _m_total,
     _weight_vars, m_w_rows, _wnew_vars, m_wnew_rows) = layout
    p3_rows = sum(v.n_rows(cfg.ELL) for v in p3_vars)
    return (["blind"] + (["w"] if m_w_rows else []) + (["wnew"] if m_wnew_rows else [])
            + ["p1", "p2"] + (["p3"] if p3_rows else []))


def _publish_statement(path, raw, sidecar):
    side = Path(str(path) + ".json")
    dir_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        # Only a publisher holding this lock can have a live staging directory.
        fcntl.flock(dir_fd, fcntl.LOCK_EX)
        if os.path.lexists(path):
            raise FileExistsError(f"refusing to overwrite {str(path)!r}")
        if os.path.lexists(side):
            # Recover a sidecar linked by a killed writer only when its inode
            # still matches our staged file. Leave an unrelated sidecar alone.
            for stage in path.parent.glob(".verinf-preview-*"):
                staged_side = stage / "sidecar"
                if (not stage.is_symlink() and not side.is_symlink()
                        and not staged_side.is_symlink() and staged_side.is_file()
                        and side.samefile(staged_side)):
                    side.unlink()
                    staged_side.unlink()
                    (stage / "claims").unlink(missing_ok=True)
                    stage.rmdir()
                    break
            else:
                raise FileExistsError(f"refusing to overwrite {str(side)!r}")
        with tempfile.TemporaryDirectory(dir=path.parent, prefix=".verinf-preview-") as temp:
            stage = Path(temp)
            for name, data in (("claims", raw), ("sidecar", sidecar)):
                with (stage / name).open("xb") as f:
                    f.write(data)
                    f.flush()
                    os.fsync(f.fileno())
            stage_fd = os.open(stage, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(stage_fd)
            finally:
                os.close(stage_fd)
            os.fsync(dir_fd)
            with ExitStack() as rollback:
                os.link(stage / "sidecar", side)
                rollback.callback(side.unlink)
                # The claim path appears only after its sidecar is durable.
                os.fsync(dir_fd)
                os.link(stage / "claims", path)
                rollback.callback(path.unlink)
                os.fsync(dir_fd)
                rollback.pop_all()
    finally:
        os.close(dir_fd)


def write_statement(tape, cfg, path, *, ui_info=None, sz=None, weight_commitment=None):
    """Write `path` (claim bytes) and `path + '.json'` (sidecar). Refuses to
    overwrite either. A referenced enrollment must match the W layout. Its
    root is recorded in the sidecar."""
    layout = core._layout(core._with_synthesized_settlements(tape.claims), cfg)
    if weight_commitment is not None:
        weight_commitment.check_layout(cfg, layout[8])  # persistent W rows
    raw = pr.claims_canonical_bytes(tape.claims, cfg)
    blocks = block_order(layout, cfg)
    # The prover also tracks witness diagnostics. Publish only the public
    # parameters and the number of positions covered by the revealed sum.
    public_ui = {k: ui_info[k] for k in ("s_b", "s_c", "s_y", "gap_max", "n")
                 if ui_info and k in ui_info}
    meta = dict(claims_sha256=hashlib.sha256(raw).hexdigest(), claims_bytes=len(raw),
                blocks=blocks, statement_digest=pr.statement_digest(raw, blocks).hex(),
                n_claims=len(tape.claims), sz=sz, ui=public_ui,
                weight_root=weight_commitment.root.hex() if weight_commitment is not None else None)
    sidecar = (json.dumps(meta, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _publish_statement(Path(path), raw, sidecar)
    print(f"  statement: {len(raw):,} claim bytes, blocks {blocks} → {path}")
    return meta
