"""Produce tampered copies of /tmp/proof.json, one perturbed field each, so the
Rust verifier can be checked to REJECT every one. Each tamper targets a
different check (merkle / lin / quad)."""
import json, copy, sys

P = 0xFFFFFFFF00000001
src = json.load(open("/tmp/proof.json"))


def bump(x):
    return (int(x) + 1) % P


def first_key(d):
    return next(iter(d))


def make(kind):
    d = copy.deepcopy(src)
    pr = d["proof"]
    if kind == "root_p1":
        b = bytearray.fromhex(pr["root_p1"]); b[0] ^= 1; pr["root_p1"] = b.hex()
    elif kind == "opened_col":
        j = first_key(pr["opened_p1"]); pr["opened_p1"][j][0] = bump(pr["opened_p1"][j][0])
    elif kind == "path_sibling":
        j = first_key(pr["paths_p1"]); s = pr["paths_p1"][j][0]
        b = bytearray.fromhex(s[0]); b[0] ^= 1; s[0] = b.hex()
    elif kind == "q_lin":
        pr["q_lin"][0] = bump(pr["q_lin"][0])
    elif kind == "p_0":
        pr["p_0"][0] = bump(pr["p_0"][0])
    else:
        raise ValueError(kind)
    out = f"/tmp/proof_tamper_{kind}.json"
    json.dump(d, open(out, "w"))
    print(out)


for k in ["root_p1", "opened_col", "path_sibling", "q_lin", "p_0"]:
    make(k)
