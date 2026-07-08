"""Prover server (UNTRUSTED side) — a line-protocol wrapper around TapeProver,
so the Rust verifier can drive the staged protocol over a subprocess pipe.

Protocol (newline-delimited compact JSON, one message per line):
  stdin  request : {"stage": n, "s_op": <hex|null>, "s_comb": <hex|null>, "s_col": <hex|null>}
  stdout response: the stage's outputs in the same shapes as dump_proof.py
                   (roots hex; q_irs/q_lin/p_0 int lists; opened_* keyed by str(j);
                    paths_* as [[hex, side], ...]).
A one-time {"ready": true} line is emitted on stdout after CUDA init so the
driver can sync past startup. The witness stays entirely in THIS process — only
seeds (in) and the transcript (out) cross the boundary.

Usage:  python prover_server.py [matmul|chain|rmsnorm] [--tamper]
The --tamper flag flips one opened value (slot 0 of opened_p1) so the driver's
REJECT path can be exercised end-to-end.
"""
import sys, json
import torch          # noqa: F401 — CUDA prover
import core           # noqa: F401
import claims as C    # noqa: F401
import packets as PK  # noqa: F401
import protocol as pr
from tape_prover import TapeProver
from statements import STATEMENTS


def _path_json(path):
    return [[sib.hex(), int(side)] for sib, side in path]


def _opened_json(d):
    return {str(j): d[j] for j in d}     # d[j] is already an int list (TapeProver _ints)


def _paths_json(d):
    return {str(j): _path_json(d[j]) for j in d}


def respond(prover, req):
    stage = req["stage"]
    def seed(k):
        v = req.get(k)
        return bytes.fromhex(v) if v is not None else None
    s_op, s_comb, s_col = seed("s_op"), seed("s_comb"), seed("s_col")

    if stage == 1:
        return {"root_p1": prover(1).hex()}
    if stage == 2:
        return {"root_p2": prover(2, s_op).hex()}
    if stage == 3:
        q_irs, q_lin, p_0 = prover(3, s_op, s_comb)
        return {"q_irs": q_irs, "q_lin": q_lin, "p_0": p_0}
    if stage == 4:
        o1, o2, p1, p2 = prover(4, s_op, s_comb, s_col)
        return {"opened_p1": _opened_json(o1), "opened_p2": _opened_json(o2),
                "paths_p1": _paths_json(p1), "paths_p2": _paths_json(p2)}
    if stage == 0:   # fused: everything at once
        r1, r2, qi, ql, p0, o1, o2, p1, p2 = prover(0, s_op, s_comb, s_col)
        return {"root_p1": r1.hex(), "root_p2": r2.hex(),
                "q_irs": qi, "q_lin": ql, "p_0": p0,
                "opened_p1": _opened_json(o1), "opened_p2": _opened_json(o2),
                "paths_p1": _paths_json(p1), "paths_p2": _paths_json(p2)}
    raise ValueError(f"bad stage {stage}")


class TamperedProver(TapeProver):
    """Flips slot 0 of opened_p1 (commit 1 always has the blinding + first rows)."""
    def __call__(self, stage, *seeds):
        out = super().__call__(stage, *seeds)
        if stage in (4, 0):
            o1 = out[5] if stage == 0 else out[0]
            j = next(iter(o1)); o1[j][0] = (o1[j][0] + 1) % pr.P
        return out


def main():
    args = sys.argv[1:]
    tamper = "--tamper" in args
    rest = [a for a in args if not a.startswith("--")]
    name = rest[0] if rest else "matmul"
    claims, inputs, cfg = STATEMENTS[name]()
    prover = (TamperedProver if tamper else TapeProver)(claims, inputs, cfg)

    sys.stdout.write(json.dumps({"ready": True}) + "\n")
    sys.stdout.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        resp = respond(prover, json.loads(line))
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
