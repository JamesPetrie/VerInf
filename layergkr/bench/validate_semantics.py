"""The gate for the tensor semantics path. Nothing else may use it until this
passes.

Two implementations of the same layer now exist: `semantics.forward` (Python
objects, the reference) and `semantics.forward_tensor` (device tensors). The
second exists because the first cannot compute a production-width layer at all.
That is exactly the situation in which a fast path that proves a slightly
different statement is the failure mode that matters, so this checks three
things, in increasing strength:

  1. the traces are equal FIELD BY FIELD -- every matmul operand, every gate
     term, every lookup query, in order;
  2. `check_trace`, the independent arbiter, accepts both;
  3. the two traces produce a BYTE-IDENTICAL proof, which is the standing accept
     gate for every prover change in this project.

(3) subsumes (1) but says nothing about where a mismatch is, so all three run.

Both paths are fed the SAME weights through `LayerWeights`, rather than the same
RNG seed: equal seeds only give equal weights while the two implementations
consume the stream identically, which is the kind of assumption this file is
supposed to be checking rather than relying on.

  .venv/bin/python layergkr/bench/validate_semantics.py
"""
import argparse
import pathlib
import random
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import blake3

from layergkr import full_layer as fl, rs, semantics as sem

CASES = [
    # S,  d, d_ff, E   -- dense FFN, MoE, a route with an empty expert, wider d
    (4, 8, 16, 1),
    (6, 16, 32, 3),
    (8, 16, 32, 5),
    (5, 32, 64, 2),
    (3, 64, 128, 4),
]


def _canon(h, obj) -> None:
    """Feed a value into a hash in a way that distinguishes shape from content:
    every container writes its kind and length first, so [[1],[2]] and [[1,2]]
    cannot collide."""
    if hasattr(obj, "shape"):                       # torch tensor
        h.update(b"T" + str(tuple(obj.shape)).encode())
        h.update(obj.reshape(-1).cpu().numpy().tobytes())
        return
    if isinstance(obj, bytes):
        h.update(b"b" + len(obj).to_bytes(8, "little") + obj)
    elif isinstance(obj, bool):
        h.update(b"?" + (b"1" if obj else b"0"))
    elif isinstance(obj, int):
        h.update(b"i" + (obj % (1 << 128)).to_bytes(16, "little"))
    elif isinstance(obj, str):
        h.update(b"s" + obj.encode())
    elif obj is None:
        h.update(b"n")
    elif isinstance(obj, (list, tuple)):
        h.update(b"[" + len(obj).to_bytes(8, "little"))
        for v in obj:
            _canon(h, v)
    elif isinstance(obj, dict):
        h.update(b"{" + len(obj).to_bytes(8, "little"))
        for k, v in obj.items():
            _canon(h, k)
            _canon(h, v)
    elif hasattr(obj, "__dict__"):
        h.update(b"o" + type(obj).__name__.encode())
        for k, v in sorted(vars(obj).items()):
            _canon(h, k)
            _canon(h, v)
    else:
        raise TypeError(f"cannot canonicalise {type(obj)}")


def digest(obj) -> str:
    h = blake3.blake3()
    _canon(h, obj)
    return h.hexdigest()[:16]


def compare_traces(a: sem.LayerTrace, b: sem.LayerTrace) -> str:
    """First difference, or '' if there is none. Reported by location, because
    'the digests differ' is not a debuggable statement."""
    if a.x_in != b.x_in:
        return "x_in"
    if a.y_out != b.y_out:
        return "y_out"
    if a.route != b.route:
        return "route"
    if len(a.matmuls) != len(b.matmuls):
        return f"matmul count {len(a.matmuls)} != {len(b.matmuls)}"
    for ma, mb in zip(a.matmuls, b.matmuls):
        if ma.name != mb.name:
            return f"matmul name {ma.name} != {mb.name}"
        for fld in ("X", "W", "Y"):
            if getattr(ma, fld) != getattr(mb, fld):
                return f"matmul {ma.name}.{fld}"
    if len(a.moe) != len(b.moe):
        return f"moe count {len(a.moe)} != {len(b.moe)}"
    for ma, mb in zip(a.moe, b.moe):
        if ma.name != mb.name:
            return f"moe name {ma.name} != {mb.name}"
        if ma.route != mb.route:
            return f"moe {ma.name}.route"
        for fld in ("X", "W", "Y"):
            if getattr(ma, fld) != getattr(mb, fld):
                return f"moe {ma.name}.{fld}"
    if len(a.gates) != len(b.gates):
        return f"gate count {len(a.gates)} != {len(b.gates)}"
    for ga, gb in zip(a.gates, b.gates):
        if (ga.kind, ga.node_id) != (gb.kind, gb.node_id):
            return f"gate id {ga.node_id} != {gb.node_id}"
        if ga.terms != gb.terms:
            return f"gate {ga.node_id} terms"
    if len(a.lookups) != len(b.lookups):
        return f"lookup count {len(a.lookups)} != {len(b.lookups)}"
    for ua, ub in zip(a.lookups, b.lookups):
        if ua.table.name != ub.table.name:
            return f"lookup table {ua.table.name} != {ub.table.name}"
        if list(ua.queries) != list(ub.queries):
            return f"lookup {ua.table.name} queries"
    return ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ell", type=int, default=64)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--q", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    cfg_rs = rs.Config(ELL=args.ell, K_DEG=2 * args.ell, N_LIG=args.n,
                       T_QUERIES=args.q)
    print(f"gate for the tensor semantics path   ELL={cfg_rs.ELL} N={cfg_rs.N_LIG}\n")
    print(f"{'case':<22}{'py':>9}{'tensor':>9}{'speedup':>9}  "
          f"{'trace':>7}{'check':>7}{'proof':>7}")

    failures = 0
    for S, d, d_ff, E in CASES:
        cfg = sem.ToyConfig(S=S, d=d, d_ff=d_ff, E=E, table_bits=6, scale_bits=6)
        # drawn on the device and converted DOWN, so the reference path is fed
        # exactly the numbers the tensor path got -- not a matching seed
        w_dev = sem.LayerWeights.draw_tensor(cfg, args.seed)
        w = w_dev.to_lists()

        t0 = time.perf_counter()
        ref = sem.forward(cfg, None, weights=w)
        t_py = time.perf_counter() - t0

        t0 = time.perf_counter()
        dev = sem.forward_tensor(cfg, None, weights=w_dev)
        t_t = time.perf_counter() - t0
        got = sem.to_python(dev)

        where = compare_traces(ref, got)
        ok_ref, why_ref = sem.check_trace(ref)
        ok_got, why_got = sem.check_trace(got)
        check = "ok" if (ok_ref and ok_got) else "FAIL"

        # the standing accept gate: same proof, byte for byte
        d_ref = digest(fl.prove_full_layer(ref, cfg_rs, fl.Enrollment(cfg_rs),
                                           args.q, random.Random(args.seed)))
        d_got = digest(fl.prove_full_layer(got, cfg_rs, fl.Enrollment(cfg_rs),
                                           args.q, random.Random(args.seed)))

        trace_ok = "ok" if where == "" else "FAIL"
        proof_ok = "ok" if d_ref == d_got else "FAIL"
        if where or check == "FAIL" or proof_ok == "FAIL":
            failures += 1
        print(f"S={S} d={d} dff={d_ff} E={E}".ljust(22)
              + f"{t_py:8.3f}s{t_t:8.3f}s{t_py/max(t_t,1e-9):8.1f}x  "
              + f"{trace_ok:>7}{check:>7}{proof_ok:>7}", flush=True)
        if where:
            print(f"    first difference: {where}")
        if not ok_ref:
            print(f"    reference trace inconsistent: {why_ref}")
        if not ok_got:
            print(f"    tensor trace inconsistent: {why_got}")
        if d_ref != d_got:
            print(f"    proof digests: {d_ref} != {d_got}")

    print()
    if failures:
        print(f"{failures} of {len(CASES)} cases FAILED -- the tensor path is not "
              f"equivalent and must not be used")
        return 1
    print(f"all {len(CASES)} cases identical: trace field by field, check_trace on "
          f"both, and byte-identical proofs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
