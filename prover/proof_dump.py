"""Streaming proof-JSON writer.

The dict-of-lists + json.dump path materializes every opened-column value as
a Python int (~2×10⁸ ints ≈ 6+ GB at full-model scale) before writing a byte.
This writer emits the identical JSON document incrementally — peak extra
memory is one CHUNK of ints — so dump cost is I/O, not RAM.
"""
import json

CHUNK = 1_000_000


def _w_u64_list(f, t):
    """One JSON array of u64s, written a chunk at a time.

    The chunk is rendered by json.dumps — the C encoder — instead of a Python
    ",".join(str(v) for v in ...). The output text is identical (separators are
    pinned to the compact form), but the measured throughput on the dev box is
    79 MB/s against 47 MB/s for the join, and the proof egress stage is priced
    at 95 GB: that difference is ~20 minutes of the 4-hour envelope. Byte-level
    alternatives (b",".join of %d-formatted ints, numpy.savetxt) were both
    slower and are not worth the loss of readability."""
    f.write("[")
    n = t.numel()
    for lo in range(0, n, CHUNK):
        vals = t[lo:lo + CHUNK].cpu().tolist()
        if lo:
            f.write(",")
        f.write(json.dumps(vals, separators=(",", ":"))[1:-1])
    f.write("]")


def proof_block_order(proof):
    """The commitment-block suffixes in row order (analysis/persistent-weights.md).
    Read from `proof.blocks`; default to the legacy two blocks. Single source of
    truth for which blocks a proof has and their order — every serializer and the
    Rust verifier agree on this, so a new block (e.g. a second weight tree in the
    P5 linking proof) needs no serializer edits."""
    return list(getattr(proof, "blocks", None) or ["p1", "p2"])


def estimated_bytes(proof, Q, claims_bytes_len=0):
    """A deliberately generous size estimate for the proof file.

    Values are u64 in decimal, at most 20 digits plus a separator; paths are
    hex node strings. Used to refuse a write that would run the filesystem out
    of space halfway through a multi-hour proof."""
    n_values = sum(t.numel() for b in proof_block_order(proof)
                   for t in getattr(proof, "opened_%s" % b).values())
    n_values += sum(getattr(proof, k).numel() for k in ("q_irs", "q_lin", "p_0"))
    n_paths = sum(len(steps) for b in proof_block_order(proof)
                  for steps in getattr(proof, "paths_%s" % b).values())
    return int(n_values * 21 + n_paths * 80 + claims_bytes_len + (1 << 20))


def dump_proof(path, claims_json, seeds, proof, Q, python_accept):
    """The single proof→JSON writer (streaming, so full-model proofs dump at I/O
    cost, not RAM). Block-driven off `proof.blocks`: each block b emits
    root_<b>/opened_<b>/paths_<b>. seeds: hex {s_op,s_comb,s_col}. Q: ordered
    columns. python_accept: True/False/None.

    A proof from the sequential Fiat-Shamir prover carries its own derived
    seeds, opened columns and canonical claim bytes; those WIN over the passed
    `seeds`/`Q`/`claims_json`, so no call site can pair a proof with a
    transcript it was not made under. The claim bytes are written verbatim
    because the statement digest is taken over exactly these bytes."""
    blocks = proof_block_order(proof)
    fs_seeds = getattr(proof, "seeds", None)
    if fs_seeds:
        seeds = {k: v.hex() for k, v in fs_seeds.items()}
    if getattr(proof, "Q_cols", None):
        Q = list(proof.Q_cols)
    claims_bytes = getattr(proof, "claims_bytes", None)
    # Reserve first, write atomically. A proof that dies out of disk space
    # after four hours leaves a truncated file that looks like a proof; the
    # rename makes the final path appear only once the whole document is on
    # disk and fsynced.
    import os
    import shutil
    need = estimated_bytes(proof, Q, len(claims_bytes or b""))
    target_dir = os.path.dirname(os.path.abspath(path)) or "."
    free = shutil.disk_usage(target_dir).free
    if free < need:
        raise OSError(
            f"refusing to write the proof: {target_dir} has {free/1e9:.1f} GB "
            f"free, the proof needs about {need/1e9:.1f} GB")
    part = path + ".part"
    with open(part, "w") as f:
        f.write('{"claims": ')
        if claims_bytes is not None:
            f.write(claims_bytes.decode())
        else:
            json.dump(claims_json, f)
        f.write(', "seeds": ')
        json.dump(seeds, f)
        if getattr(proof, "statement_digest", None) is not None:
            f.write(', "statement_digest": %s'
                    % json.dumps(proof.statement_digest.hex()))
        f.write(', "proof": {')
        f.write('"blocks": %s, ' % json.dumps(blocks))
        for b in blocks:
            f.write('"root_%s": %s, ' % (b, json.dumps(getattr(proof, "root_%s" % b).hex())))
        for key in ("q_irs", "q_lin", "p_0"):
            f.write('"%s": ' % key)
            _w_u64_list(f, getattr(proof, key))
            f.write(', ')
        for b in blocks:
            cols = getattr(proof, "opened_%s" % b)
            f.write('"opened_%s": {' % b)
            for k, j in enumerate(Q):
                if k:
                    f.write(",")
                f.write('"%d": ' % j)
                _w_u64_list(f, cols[j])
            f.write('}, ')
        pj = lambda paths: {str(j): [[sib.hex(), int(side)] for sib, side in paths[j]]
                             for j in Q}
        for i, b in enumerate(blocks):
            f.write('%s"paths_%s": ' % ("" if i == 0 else ", ", b))
            json.dump(pj(getattr(proof, "paths_%s" % b)), f)
        f.write('}, "python_accept": ')
        json.dump(python_accept, f)
        f.write('}')
        f.flush()
        os.fsync(f.fileno())
    os.replace(part, path)
