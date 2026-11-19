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
import hashlib
import json
import os

import core
import protocol as pr


def block_order(layout, cfg):
    (_all, _p1, _p2, p3_vars, _m_p1, _m_p2, _m_total,
     _weight_vars, m_w_rows, _wnew_vars, m_wnew_rows) = layout
    p3_rows = sum(v.n_rows(cfg.ELL) for v in p3_vars)
    return (["blind"] + (["w"] if m_w_rows else []) + (["wnew"] if m_wnew_rows else [])
            + ["p1", "p2"] + (["p3"] if p3_rows else []))


def write_statement(tape, cfg, path, *, ui_info=None, sz=None, weight_root=None):
    """Write `path` (claim bytes) and `path + '.json'` (sidecar). Refuses to
    overwrite either. `weight_root` (hex) records the enrolled W root the
    proof will reference; it is not part of the claim bytes."""
    side = path + ".json"
    for p in (path, side):
        if os.path.exists(p):
            raise FileExistsError(f"refusing to overwrite {p!r}")
    layout = core._layout(core._with_synthesized_settlements(tape.claims), cfg)
    raw = pr.claims_canonical_bytes(tape.claims, cfg)
    blocks = block_order(layout, cfg)
    with open(path, "xb") as f:
        f.write(raw)
    # The prover also tracks witness diagnostics. Publish only the public
    # parameters and the number of positions covered by the revealed sum.
    public_ui = {k: ui_info[k] for k in ("s_b", "s_c", "s_y", "gap_max", "n")
                 if ui_info and k in ui_info}
    meta = dict(claims_sha256=hashlib.sha256(raw).hexdigest(), claims_bytes=len(raw),
                blocks=blocks, statement_digest=pr.statement_digest(raw, blocks).hex(),
                n_claims=len(tape.claims), sz=sz, ui=public_ui,
                weight_root=weight_root)
    with open(side, "x") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"  statement: {len(raw):,} claim bytes, blocks {blocks} → {path}")
    return meta
