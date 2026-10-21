"""The run-only tape at K2's full depth, built on the CPU and measured apart
from any execution (review of 2026-10-21): the build's wall time, the part of
it spent allocating the shared tables, their entries and bytes, the claims
and variables, and the host memory, per precision profile, each in a fresh
process.

The tape reads shapes, never weights, while it is built, so the 13 shards'
real headers with holes after stand in for the model
(~/verinf-fixtures/k2-standin/PIN.txt). Each MoE layer's selection bias is
real there, so the routing widths are. Execution is measured on a card.

    python tools/k2_build_measure.py --gguf STANDIN/Kimi-K2-Instruct-UD-Q4_K_XL-00001-of-00013.gguf \\
        --record OUT.json [--T 924] [--precisions a16w16,a14w14,baseline]"""
import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

_ROOT = pathlib.Path(__file__).resolve().parents[1]
PATHS = ("prover", "prover/tests", "demo")


def _rss_gb():
    for line in pathlib.Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1e6
    return None


def measure(gguf, precision, T, layers):
    """One build (call in a fresh process: names carry counters)."""
    import resource
    import _cpu_tape as ct
    with ct.cuda_on_cpu():
        import demo_k2 as dk
        from tape import Tape
        tables = dict(seconds=0.0, calls=0, depth=0)

        def timed(fn):
            def run(*a, **kw):
                tables["depth"] += 1
                t0 = time.time()
                try:
                    return fn(*a, **kw)
                finally:
                    tables["depth"] -= 1
                    if tables["depth"] == 0:
                        tables["seconds"] += time.time() - t0
                        tables["calls"] += 1
            return run
        Tape.register_table = timed(Tape.register_table)
        Tape._range_table = timed(Tape._range_table)
        P = min(424, T // 2)
        rss0 = _rss_gb()
        t0 = time.time()
        b = dk.build(str(gguf), list(range(3, 3 + P)), list(range(1000, 1000 + T - P)), layers=layers, kk=8,
                     precision=precision)
        build_s = time.time() - t0
        mults = {v.name[:-len("_mult")]: t for v, t in b.tape.inputs.items() if v.name.endswith("_mult")}
        entries = sum(int(t.numel()) for t in mults.values())
        lazy = sum(1 for t in b.tape.inputs.values() if callable(t))
        return dict(precision=precision, layers=layers, T=T, n_prompt=P, build_s=build_s,
                    tables=dict(seconds=tables["seconds"], calls=tables["calls"], count=len(mults), entries=entries,
                                bytes_keys_and_multiplicities=16 * entries),
                    build_without_tables_s=build_s - tables["seconds"],
                    claims=len(b.tape.claims), deferred=len(b.tape._deferred), inputs=len(b.tape.inputs),
                    lazy_inputs=lazy, rss_before_gb=rss0, rss_after_gb=_rss_gb(),
                    rss_peak_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--record", required=True)
    ap.add_argument("--T", type=int, default=924, help="the longest frozen passage: 424 prompt + 500 continuation")
    ap.add_argument("--layers", type=int, default=61)
    ap.add_argument("--precisions", default="a16w16,a14w14,baseline")
    ap.add_argument("--one", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.one:
        print(json.dumps(measure(a.gguf, a.one, a.T, a.layers)))
        return 0
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(str(_ROOT / p) for p in PATHS),
               LIGERO_ELL="8192", LIGERO_K_DEG="16384", LIGERO_N_LIG="65536", CUDA_VISIBLE_DEVICES="")
    rec = dict(what=__doc__.split("\n\n")[0], gguf=pathlib.Path(a.gguf).name,
               stand_in="the 13 shards' real headers (8 MB each), holes after, each MoE layer's selection bias "
                        "real; unsloth/Kimi-K2-Instruct-GGUF @ 23bf90d8, UD-Q4_K_XL", runs={})
    for prec in a.precisions.split(","):
        p = subprocess.run([sys.executable, __file__, "--gguf", a.gguf, "--record", a.record, "--T", str(a.T),
                            "--layers", str(a.layers), "--one", prec], env=env, capture_output=True, text=True)
        if p.returncode != 0:
            raise SystemExit(f"{prec}: {p.stderr[-3000:]}")
        r = rec["runs"][prec] = json.loads(p.stdout.strip().splitlines()[-1])
        print(f"{prec}: build {r['build_s']:.0f} s (tables {r['tables']['seconds']:.0f} s, "
              f"{r['tables']['entries'] / 1e6:.0f} M entries), {r['claims']} claims, "
              f"{r['lazy_inputs']} lazy inputs, RSS {r['rss_after_gb']:.1f} GB (peak {r['rss_peak_gb']:.1f})", flush=True)
    pathlib.Path(a.record).write_text(json.dumps(rec, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
