"""Streaming proof-JSON writer.

The dict-of-lists + json.dump path materializes every opened-column value as
a Python int (~2×10⁸ ints ≈ 6+ GB at full-model scale) before writing a byte.
This writer emits the identical JSON document incrementally — peak extra
memory is one CHUNK of ints — so dump cost is I/O, not RAM.
"""
import json
import base64
import os
import shutil

CHUNK = 1_000_000
# 999999 * sizeof(u64) is divisible by 3.  Therefore independently encoded
# non-final chunks concatenate into one valid base64 stream without interior
# '=' padding.
B64_CHUNK = 999_999


def _fsync_dir(path):
    """Persist a create/rename in the containing directory on POSIX."""
    dfd = os.open(os.path.dirname(os.path.abspath(path)) or ".", os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


def _w_u64_list(f, t):
    """One JSON array of u64s, written a chunk at a time.

    The chunk is rendered by json.dumps — the C encoder — instead of a Python
    ",".join(str(v) for v in ...). The output text is identical (separators are
    pinned to the compact form), but the measured throughput on the dev box is
    79 MB/s against 47 MB/s for the join, and the proof egress stage is priced
    at production scale: that difference is material to the 4-hour envelope.
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


def _w_u64_b64(f, t):
    """Versioned compact wire representation of a u64 vector.

    Fiat--Shamir hashes the canonical little-endian field bytes before the
    proof is serialized, so replacing decimal JSON integers by base64 of those
    SAME bytes is transport-only: the verifier reconstructs the identical
    Vec<u64>.  This cuts the production proof from roughly 21 bytes/value to
    10.67 bytes/value and moves formatting into CPython's C base64 loop.
    """
    f.write('"u64le:')
    n = t.numel()
    for lo in range(0, n, B64_CHUNK):
        # CPU tensors are native-endian; production is little-endian x86.  The
        # explicit dtype makes the wire contract unambiguous and remains a
        # zero-copy view on the production hosts.
        arr = t[lo:lo + B64_CHUNK].cpu().contiguous().numpy().astype("<u8", copy=False)
        f.write(base64.b64encode(memoryview(arr)).decode("ascii"))
    f.write('"')


def _write_u64(f, t, encoding):
    if encoding == "decimal":
        return _w_u64_list(f, t)
    if encoding == "u64le-base64":
        return _w_u64_b64(f, t)
    raise ValueError(f"unknown u64 proof encoding {encoding!r}")


def proof_block_order(proof):
    """The commitment-block suffixes in row order (analysis/persistent-weights.md).
    Read from `proof.blocks`; default to the legacy two blocks. Single source of
    truth for which blocks a proof has and their order — every serializer and the
    Rust verifier agree on this, so a new block (e.g. a second weight tree in the
    P5 linking proof) needs no serializer edits."""
    return list(getattr(proof, "blocks", None) or ["p1", "p2"])


def estimated_bytes(proof, Q, claims_bytes_len=0, *, u64_encoding="decimal"):
    """A deliberately generous size estimate for the proof file.

    Values are u64 in decimal, at most 20 digits plus a separator; paths are
    hex node strings. Used to refuse a write that would run the filesystem out
    of space halfway through a multi-hour proof."""
    n_values = sum(t.numel() for b in proof_block_order(proof)
                   for t in getattr(proof, "opened_%s" % b).values())
    n_values += sum(getattr(proof, k).numel() for k in ("q_irs", "q_lin", "p_0"))
    n_paths = sum(len(steps) for b in proof_block_order(proof)
                  for steps in getattr(proof, "paths_%s" % b).values())
    per_value = 21 if u64_encoding == "decimal" else 11
    return int(n_values * per_value + n_paths * 80 + claims_bytes_len + (1 << 20))


def reserve_output(path, reserve_bytes):
    """Reserve proof space BEFORE the multi-hour prover starts.

    Returns the `<path>.part` filename.  A stale reservation is refused rather
    than silently overwritten.  `posix_fallocate` is used when available so
    quota/ENOSPC is discovered now, not after proving; the writer later writes
    into this same inode and truncates it to the actual length before fsync.
    """
    part = path + ".part"
    target_dir = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(target_dir, exist_ok=True)
    if os.path.exists(path) or os.path.exists(part):
        raise FileExistsError(
            f"refusing to reserve proof output: {path!r} or its .part exists")
    free = shutil.disk_usage(target_dir).free
    if free < reserve_bytes:
        raise OSError(
            f"refusing to start the proof: {target_dir} has {free/1e9:.1f} GB "
            f"free, need {reserve_bytes/1e9:.1f} GB reserved for output")
    fd = os.open(part, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
    try:
        if not hasattr(os, "posix_fallocate"):
            raise OSError(
                "production proof reservation requires posix_fallocate; "
                "ftruncate would create a sparse file without reserving space")
        os.posix_fallocate(fd, 0, int(reserve_bytes))
        os.fsync(fd)
    except Exception:
        os.close(fd)
        try: os.unlink(part)
        except FileNotFoundError: pass
        raise
    os.close(fd)
    _fsync_dir(part)
    return part


def dump_proof(path, claims_json, seeds, proof, Q, python_accept, *,
               u64_encoding="decimal", reserved_part=None):
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
    need = estimated_bytes(proof, Q, len(claims_bytes or b""),
                           u64_encoding=u64_encoding)
    target_dir = os.path.dirname(os.path.abspath(path)) or "."
    part = path + ".part"
    if reserved_part is not None and os.path.abspath(reserved_part) != os.path.abspath(part):
        raise ValueError("reserved proof file does not match target .part")
    if reserved_part is not None:
        reserved = os.path.getsize(part)
        if reserved < need:
            raise OSError(
                f"reserved proof file is {reserved/1e9:.1f} GB, but this proof "
                f"needs about {need/1e9:.1f} GB")
    else:
        free = shutil.disk_usage(target_dir).free
        if free < need:
            raise OSError(
                f"refusing to write the proof: {target_dir} has {free/1e9:.1f} GB "
                f"free, the proof needs about {need/1e9:.1f} GB")
    mode = "r+" if reserved_part is not None else "w"
    with open(part, mode) as f:
        f.seek(0)
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
            _write_u64(f, getattr(proof, key), u64_encoding)
            f.write(', ')
        for b in blocks:
            cols = getattr(proof, "opened_%s" % b)
            f.write('"opened_%s": {' % b)
            for k, j in enumerate(Q):
                if k:
                    f.write(",")
                f.write('"%d": ' % j)
                _write_u64(f, cols[j], u64_encoding)
            f.write('}, ')
        pj = lambda paths: {str(j): [[sib.hex(), int(side)] for sib, side in paths[j]]
                             for j in Q}
        for i, b in enumerate(blocks):
            f.write('%s"paths_%s": ' % ("" if i == 0 else ", ", b))
            json.dump(pj(getattr(proof, "paths_%s" % b)), f)
        f.write('}, "python_accept": ')
        json.dump(python_accept, f)
        f.write('}')
        f.truncate()
        f.flush()
        os.fsync(f.fileno())
    os.replace(part, path)
    _fsync_dir(path)
