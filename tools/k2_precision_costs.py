"""Kimi K2's proof cost by precision profile, counted from the driver's own
tapes at real dimensions, without the weights or a card.

A lazy tape records its claims, variables and tables without computing a
witness, and the driver reads only tensor shapes while building one. So a
sparse stand-in for shard 1 (its real header, every key-value and tensor
info, extended with holes to the shard's full size: no disk) builds K2's
real-dimension tape on the CPU (prover/tests/_cpu_tape.py), and the
profiler's extractor (profiler/extract.py) counts it: witness slots, linear
constraints, quadratic products, the shared tables and their settlement,
the committed rows at ELL, the compact proof bytes and the live set's peak.

The two-layer tape is the gate. The full depth is the two-layer tape plus 59
times the third layer's increment, every MoE layer after the first being
the same shape; the shared tables are counted once. The routed experts are
bridge-held (outside the witness), as the full run's are. The stand-in's
data are zeros: the selection bias reads as 0, so the routing width is the
zero-bias one (25 bits, real 26: the same two 13-bit words).

    python tools/k2_precision_costs.py --head k2-shard1-head.bin --work DIR \\
        --record analysis/k2-precision-costs-2026-10-20.json

Runtime is not estimated here: the gate measures it."""
import argparse
import json
import math
import os
import pathlib
import subprocess
import sys
import time

_ROOT = pathlib.Path(__file__).resolve().parents[1]
SHARD1 = "Kimi-K2-Instruct-UD-Q4_K_XL-00001-of-00013.gguf"
SHARD1_SIZE = 48_935_303_072          # the pinned shard's size (session kits)
GEOMETRY = dict(LIGERO_ELL="8192", LIGERO_K_DEG="16384", LIGERO_N_LIG="65536", LIGERO_T_QUERIES="54")
PROFILES = ("baseline", "a14w14", "a16w16")


def stand_in(head, work, size=SHARD1_SIZE):
    """Shard 1's name and size, its header's bytes, holes after."""
    dst = pathlib.Path(work) / SHARD1
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(pathlib.Path(head).read_bytes())
    os.truncate(dst, size)
    return dst


def count(shard, precision, T, layers):
    """One tape's counts (call in a fresh process: names carry counters)."""
    import _cpu_tape as ct
    t0 = time.time()
    with ct.cuda_on_cpu():
        import demo_k2 as dk
        import predict
        from extract import extract_tape
        P = T // 2
        b = dk.build(str(shard), list(range(3, 3 + P)), list(range(1000, 1000 + T - P)), layers=layers,
                     kk=8, precision=precision, bridge=True)
        tables = {t[1]: t[2] for t in ct.structure(b.tape)["tables"]}
        man = extract_tape(b.tape, model=dict(name="kimi-k2", layers=layers, precision=precision), seq=T)
    tot = predict.totals(man)
    lig = man.run["ligero"]
    ELL, TQ = lig["ELL"], lig["T_QUERIES"]
    own = [v for v in man.variables if not getattr(v, "external", False)]
    rows = int(sum(math.ceil(v.length / ELL) for v in own)
               + max(0.0, tot.W - sum(v.length for v in own)) / ELL) + 3            # as predict.report
    settle = [c for c in man.claims if c.type == "TableSettlement"]
    peak = predict.live_set_peak(man)
    return dict(precision=precision, T=T, layers=layers, claims=tot.n_claims, W=tot.W, cids=tot.cids,
                Q=tot.Q, W_weights=tot.W_weights, W_inputs=tot.W_inputs, W_external=tot.W_external,
                rows=rows, ligero=lig, proof_bytes_compact=TQ * rows * predict.PROOF_COMPACT_BYTES_PER_VALUE,
                tables=tables, settlement=dict(tables=len(settle), W=sum(
                    predict.claimcosts.cost(c.type, c.params)[0] for c in settle)),
                by_type=dict(tot.by_type), live_peak_bytes=peak and peak["peak_bytes"],
                build_s=time.time() - t0)


KEYS = ("W", "cids", "Q", "rows", "proof_bytes_compact")


def full_depth(two, three, layers=61):
    """The two-layer counts plus (layers − 2) MoE-layer increments."""
    return {k: two[k] + (layers - 2) * (three[k] - two[k]) for k in KEYS}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--head", help="shard 1's header bytes (verinf-fixtures/k2-head/k2-shard1-head.bin)")
    ap.add_argument("--work", required=True)
    ap.add_argument("--record")
    ap.add_argument("--profiles", default=",".join(PROFILES))
    ap.add_argument("--tokens", default="100,1000")
    ap.add_argument("--one", nargs=4, metavar=("SHARD", "PRECISION", "T", "LAYERS"), help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.one:
        shard, precision, T, layers = a.one
        print(json.dumps(count(shard, precision, int(T), int(layers)), default=str))
        return 0
    shard = stand_in(a.head, a.work)
    env = dict(os.environ, **GEOMETRY, PYTHONPATH=os.pathsep.join(
        str(_ROOT / p) for p in ("prover", "prover/tests", "demo", "profiler")))
    runs = {}
    for T in map(int, a.tokens.split(",")):
        for layers in (2, 3):
            for p in a.profiles.split(","):
                out = subprocess.run([sys.executable, __file__, "--work", a.work, "--one", str(shard), p,
                                      str(T), str(layers)], env=env, capture_output=True, text=True, check=True)
                runs[f"{p}/T{T}/L{layers}"] = r = json.loads(out.stdout.strip().splitlines()[-1])
                print(f"{p:9s} T={T:<5d} L={layers}: W {r['W']:.4e}  rows {r['rows']:,}  "
                      f"tables {sum(r['tables'].values()):,}  ({r['build_s']:.0f} s)", flush=True)
    summary = {}
    for T in map(int, a.tokens.split(",")):
        for p in a.profiles.split(","):
            two, three = runs[f"{p}/T{T}/L2"], runs[f"{p}/T{T}/L3"]
            summary[f"{p}/T{T}"] = dict(gate={k: two[k] for k in KEYS}, full_depth=full_depth(two, three),
                                        per_moe_layer={k: three[k] - two[k] for k in KEYS},
                                        table_entries=sum(two["tables"].values()), tables=two["tables"])
    if a.record:
        pathlib.Path(a.record).write_text(json.dumps(dict(
            what="K2 proof cost by precision profile, from the driver's tapes at real dimensions "
                 "(tools/k2_precision_costs.py); routed experts bridge-held",
            geometry=GEOMETRY, summary=summary, runs=runs), indent=1, default=str) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
