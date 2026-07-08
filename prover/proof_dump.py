"""Streaming proof-JSON writer.

The dict-of-lists + json.dump path materializes every opened-column value as
a Python int (~2×10⁸ ints ≈ 6+ GB at full-model scale) before writing a byte.
This writer emits the identical JSON document incrementally — peak extra
memory is one CHUNK of ints — so dump cost is I/O, not RAM.
"""
import json

CHUNK = 1_000_000


def _w_u64_list(f, t):
    f.write("[")
    n = t.numel()
    for lo in range(0, n, CHUNK):
        vals = t[lo:lo + CHUNK].cpu().tolist()
        if lo:
            f.write(",")
        f.write(",".join(str(int(v)) for v in vals))
    f.write("]")


def proof_block_order(proof):
    """The commitment-block suffixes in row order (analysis/persistent-weights.md).
    Read from `proof.blocks`; default to the legacy two blocks. Single source of
    truth for which blocks a proof has and their order — every serializer and the
    Rust verifier agree on this, so a new block (e.g. a second weight tree in the
    P5 linking proof) needs no serializer edits."""
    return list(getattr(proof, "blocks", None) or ["p1", "p2"])


def dump_proof(path, claims_json, seeds, proof, Q, python_accept):
    """The single proof→JSON writer (streaming, so full-model proofs dump at I/O
    cost, not RAM). Block-driven off `proof.blocks`: each block b emits
    root_<b>/opened_<b>/paths_<b>. seeds: hex {s_op,s_comb,s_col}. Q: ordered
    columns. python_accept: True/False/None."""
    blocks = proof_block_order(proof)
    with open(path, "w") as f:
        f.write('{"claims": ')
        json.dump(claims_json, f)
        f.write(', "seeds": ')
        json.dump(seeds, f)
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
