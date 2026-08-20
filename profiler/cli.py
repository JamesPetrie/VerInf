"""VerInf dry-run profiler CLI. Pure Python — runs anywhere, no torch.

    python3 profiler/cli.py synth   --model maverick --seq 1000 -o man.json
    python3 profiler/cli.py predict man.json --machine gb10-spark
    python3 profiler/cli.py predict man.json --machine gb10-spark --gpus 8
    python3 profiler/cli.py dag     man.json [-o dag.json]
    python3 profiler/cli.py machines

Tape extraction (the exact, model-agnostic path) is profiler/extract.py and
runs where the prover runs.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from manifest import Manifest           # noqa: E402
from machine import MachineProfile, list_machines   # noqa: E402
import synth                            # noqa: E402
import predict                          # noqa: E402
import dag as dagmod                    # noqa: E402
import partition                        # noqa: E402


def _positive(cast):
    def parse(s):
        v = cast(s)
        try:
            # Every downstream accumulator (predict/dag/partition) is a
            # float; an int float() can't represent would OverflowError
            # there, so reject it at this boundary.
            f = float(v)
        except OverflowError:
            raise argparse.ArgumentTypeError(
                f"too large for the float cost model, got {s}")
        if not math.isfinite(f) or v <= 0:
            raise argparse.ArgumentTypeError(f"must be finite and > 0, got {s}")
        return v
    return parse


_posint, _posfloat = _positive(int), _positive(float)


def _load_manifest(p, path):
    try:
        return Manifest.load(path)
    except (OSError, ValueError) as e:      # includes JSONDecodeError
        p.error(f"manifest {path}: {e}")


def _bandwidth_list(s):
    try:
        return [_posfloat(x) for x in s.split(",")]
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e))


def main(argv=None):
    p = argparse.ArgumentParser(prog="profiler", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("synth", help="build a synthetic manifest")
    ps.add_argument("--model", choices=sorted(synth.BUILDERS), required=True)
    ps.add_argument("--seq", type=_posint, required=True)
    ps.add_argument("--layers", type=_posint, default=None,
                    help="llama7b only (default 32)")
    ps.add_argument("--t-queries", type=_posint, default=None)
    ps.add_argument("-o", "--out", required=True)

    pp = sub.add_parser("predict", help="predict costs for a manifest")
    pp.add_argument("manifest")
    pp.add_argument("--machine", default="gb10-spark")
    pp.add_argument("--gpus", type=_posint, default=1,
                    help="ideal-scaling what-if across N GPUs")
    pp.add_argument("--bandwidth-ratio", type=_posfloat, default=None,
                    help="aggregate memory-bandwidth ratio vs the profile "
                         "(overrides --gpus for the A/C terms)")
    pp.add_argument("--enrolled-weights", action="store_true",
                    help="price weights as ENROLLED (no per-proof encode; "
                         "qlin+open passes instead — the kept-trees path)")
    pp.add_argument("--compute-ratio", type=_posfloat, default=None,
                    help="compute ratio vs the profile (for the B term)")

    pd = sub.add_parser("dag", help="dependency DAG + parallelism summary")
    pd.add_argument("manifest")
    pd.add_argument("-o", "--out", default=None, help="write full DAG JSON here")

    pt = sub.add_parser("partition",
                        help="score sharding strategies across N GPUs")
    pt.add_argument("manifest")
    pt.add_argument("--shards", type=_posint, required=True)
    pt.add_argument("--strategy", choices=sorted(partition.STRATEGIES),
                    default=None,
                    help="one strategy in detail; omit to compare all")
    pt.add_argument("--machine", default="gb10-spark")
    pt.add_argument("--bandwidths", type=_bandwidth_list, default=None,
                    help="comma-separated GB/s sweep (default 25,100,450,900)")
    pt.add_argument("--weight-bytes-per-param", type=_posfloat, default=1.0,
                    help="on-disk bytes/param for weight streaming "
                         "(~0.7 GGUF Q4_K, 2.0 bf16 safetensors)")
    pt.add_argument("--enrolled-weights", action="store_true",
                    help="price weights as ENROLLED (per-shard qlin+open "
                         "over owned slots; no per-proof encode split)")
    pt.add_argument("--skip-weight-commit", action="store_true",
                    # diagnostic knob, NOT a protocol mode (see
                    # --enrolled-weights for the kept-trees model)
                    help="model a reused persistent weight commitment")

    sub.add_parser("machines", help="list machine profiles")

    a = p.parse_args(argv)
    if a.cmd == "synth":
        if a.layers is not None and a.model != "llama7b":
            p.error("--layers only applies to --model llama7b")
        kw = {}
        if a.layers is not None:
            kw["layers"] = a.layers
        if a.t_queries is not None:
            kw["t_queries"] = a.t_queries
        man = synth.BUILDERS[a.model](a.seq, **kw)
        man.save(a.out)
        print(f"wrote {a.out}: {len(man.claims):,} claims, "
              f"{len(man.variables):,} variables")
    elif a.cmd == "predict":
        man = _load_manifest(p, a.manifest)
        mp = MachineProfile.load(a.machine)
        print(predict.report(man, mp, gpus=a.gpus,
                             enrolled_weights=a.enrolled_weights,
                             bandwidth_ratio=a.bandwidth_ratio,
                             compute_ratio=a.compute_ratio))
    elif a.cmd == "dag":
        man = _load_manifest(p, a.manifest)
        d = dagmod.build(man)
        print(dagmod.summary_text(d))
        if a.out:
            dagmod.save(d, a.out)
            print(f"full DAG -> {a.out}")
    elif a.cmd == "partition":
        if a.enrolled_weights and a.skip_weight_commit:
            p.error("--enrolled-weights and --skip-weight-commit are "
                    "mutually exclusive: enrollment PRICES the reused "
                    "commitment (qlin+open passes); skip drops all weight "
                    "cost (diagnostic only)")
        man = _load_manifest(p, a.manifest)
        mp = MachineProfile.load(a.machine)
        kw = dict(weight_bytes_per_param=a.weight_bytes_per_param,
                  skip_weight_commit=a.skip_weight_commit,
                  enrolled_weights=a.enrolled_weights)
        if a.bandwidths:
            kw["bandwidths"] = a.bandwidths
        if a.strategy:
            print(partition.report(man, a.strategy, a.shards, mp, **kw))
        else:
            print(partition.compare(man, a.shards, mp, **kw))
    elif a.cmd == "machines":
        for name in list_machines():
            print(name)


if __name__ == "__main__":
    main()
