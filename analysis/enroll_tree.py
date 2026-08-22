#!/usr/bin/env python3
"""Build the model's enrolment tree from the per-layer weight commitments.

A subsampled proof only ever commits ONE layer's weights, so its root_w is a
per-layer value and there are n_layers of them. Rather than pin all of them,
pin ONE: a Merkle tree over the per-layer roots, in layer order. Policy then
carries a single MODEL_ROOT, and a proof of layer L is checked by

    1. root_w == leaf[L]            (verify_proof's existing enrolled-root check)
    2. path(leaf[L], L) == MODEL_ROOT   (checked here, verifier-side)

No hashing happens inside the circuit. root_w is already a binding commitment
to that layer's weights and it is already public in the proof, so the tree is
built over values the verifier can see. That is what makes this tractable --
proving a Merkle path in-circuit would need a hash gadget over the path and is
a different project entirely.

Usage:  enroll_tree.py <enroll-dir> [out.json]
"""
import glob, hashlib, json, os, struct, sys

MAGIC = b"VERINFWC"


def read_root(path):
    """Parse a WeightCommitment handle header: magic|u32 ver|u32 m_w|u32 n_lig|32B root."""
    with open(path, "rb") as fh:
        head = fh.read(len(MAGIC) + 12 + 32)
    if head[:len(MAGIC)] != MAGIC:
        raise ValueError("%s: not a WeightCommitment handle" % path)
    ver, m_w, n_lig = struct.unpack("<III", head[len(MAGIC):len(MAGIC) + 12])
    return m_w, n_lig, head[len(MAGIC) + 12:]


def node(a, b):
    return hashlib.sha256(b"\x01" + a + b).digest()


def build(leaves):
    """Binary SHA-256 tree; an odd node is paired with itself. Returns
    (root, levels) with levels[0] == leaves."""
    levels = [list(leaves)]
    cur = list(leaves)
    while len(cur) > 1:
        nxt = [node(cur[i], cur[i + 1] if i + 1 < len(cur) else cur[i])
               for i in range(0, len(cur), 2)]
        levels.append(nxt)
        cur = nxt
    return cur[0], levels


def path_for(levels, idx):
    """Sibling hashes from leaf `idx` up to the root."""
    out, i = [], idx
    for lvl in levels[:-1]:
        sib = i ^ 1
        out.append((lvl[sib] if sib < len(lvl) else lvl[i]).hex())
        i //= 2
    return out


def verify(leaf, idx, path, root):
    h, i = leaf, idx
    for sib_hex in path:
        sib = bytes.fromhex(sib_hex)
        h = node(h, sib) if i % 2 == 0 else node(sib, h)
        i //= 2
    return h == root


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else ".enroll"
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(d, "model_policy.json")
    files = sorted(glob.glob(os.path.join(d, "layer_*.wc")))
    if not files:
        sys.exit("no layer_*.wc handles in %s -- run proofs with ENROLL_DIR set" % d)
    leaves, meta = [], []
    for f in files:
        m_w, n_lig, root = read_root(f)
        leaves.append(root)
        meta.append({"layer": int(os.path.basename(f)[6:8]), "m_w": m_w,
                     "n_lig": n_lig, "root": root.hex()})
    if len({m["m_w"] for m in meta}) != 1:
        print("warning: layers disagree on m_w %s" % sorted({m["m_w"] for m in meta}))
    root, levels = build(leaves)
    policy = {"model_root": root.hex(), "n_layers": len(leaves),
              "leaves": [m["root"] for m in meta],
              "paths": {str(m["layer"]): path_for(levels, i)
                        for i, m in enumerate(meta)},
              "m_w": meta[0]["m_w"], "n_lig": meta[0]["n_lig"]}
    for i, m in enumerate(meta):
        assert verify(leaves[i], i, policy["paths"][str(m["layer"])], root), \
            "path check failed for layer %d" % m["layer"]
    with open(out, "w") as fh:
        json.dump(policy, fh, indent=1)
    print("enrolled %d layers, m_w=%d each" % (len(leaves), meta[0]["m_w"]))
    for m in meta:
        print("  L%-2d %s" % (m["layer"], m["root"][:32]))
    print("MODEL_ROOT = %s" % root.hex())
    print("all %d inclusion paths verify; policy -> %s" % (len(leaves), out))


if __name__ == "__main__":
    main()
