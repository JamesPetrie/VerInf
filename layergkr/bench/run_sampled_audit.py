"""Run the executable sampled-audit protocol on mixed local proof families."""
import argparse
import hashlib
import json
import pathlib
import random
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from layergkr import rs
from layergkr.sampled_audit import (
    AuditParams,
    AuditStatement,
    LookupBlock,
    MatmulBlock,
    RelationBlock,
    VerifierSession,
    Wire,
    prove,
    verify,
)
from prover.protocol import P


def blocks(n, seed):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        if i % 3 == 0:
            x = [[rng.randrange(32) for _ in range(4)] for _ in range(2)]
            w = [[rng.randrange(32) for _ in range(4)] for _ in range(3)]
            y = [[sum(a*b for a, b in zip(row, wr)) % P for wr in w] for row in x]
            out.append(MatmulBlock(i, f"matmul-{i}", Wire(f"x-{i}", x),
                                   Wire(f"w-{i}", w), Wire(f"y-{i}", y)))
        elif i % 3 == 1:
            a = [rng.randrange(32) for _ in range(8)]
            b = [rng.randrange(32) for _ in range(8)]
            c = [x*y % P for x, y in zip(a, b)]
            out.append(RelationBlock(
                i, f"rounding-{i}", "rounding",
                {"a": Wire(f"a-{i}", [a]), "b": Wire(f"b-{i}", [b]),
                 "c": Wire(f"c-{i}", [c])},
                [(1, ["a", "b"]), (P-1, ["c"])]))
        else:
            table = [(x,) for x in range(8)]
            q = [[rng.randrange(8)] for _ in range(8)]
            mult = [sum(row[0] == x for row in q) for x in range(8)]
            out.append(LookupBlock(i, f"lookup-{i}", Wire(f"q-{i}", q),
                                   Wire(f"mult-{i}", [mult]), table, 11))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, default=2596)
    ap.add_argument("--window", type=int, default=49)
    ap.add_argument("--sample", type=int, default=5)
    ap.add_argument("--columns", type=int, default=61)
    ap.add_argument("--seed", type=int, default=20260828)
    ap.add_argument("--out", default=None, help="optional JSONL result ledger")
    args = ap.parse_args()
    cfg = rs.Config(ELL=8, K_DEG=16, N_LIG=64, T_QUERIES=args.columns)
    bs = blocks(args.blocks, args.seed)
    params = AuditParams(args.blocks, args.window, args.sample, args.columns)
    statement = AuditStatement.from_blocks(
        params, hashlib.sha256(b"toy-public-io").digest(), bs)
    session = VerifierSession(
        hashlib.sha256(f"verifier/{args.seed}".encode()).digest())
    t0 = time.perf_counter()
    proof = prove(bs, cfg, statement, session)
    tp = time.perf_counter() - t0
    t0 = time.perf_counter()
    ok, why = verify(proof, cfg, statement, session)
    tv = time.perf_counter() - t0
    row = {"blocks": args.blocks, "selected": len(proof.selected),
           "fraction": len(proof.selected)/args.blocks, "columns": args.columns,
           "prove_s": tp, "verify_s": tv, "accepted": ok, "why": why,
           "c0": proof.c0_root.hex(), "ts": int(time.time())}
    if args.out:
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
    print(json.dumps(row, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
