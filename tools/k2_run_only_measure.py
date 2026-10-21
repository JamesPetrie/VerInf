"""The run-only pass and the float64 reference on K2's own weights at two
layers, timed for pricing the full-depth run (analysis/k2-fidelity-study-
design.md §7.1): per precision profile at the longest passage's length (424
prompt + 500 continuation tokens, real text), the build, the run, the card's
peak memory, the lookups' and field safety's checks and the widest value per
layer; then the torch p64's time on the same sequence. First, at the gate's
length, the pass's logits against the driver's own engine pass (check mode's
machinery): equal, or the pass is not what proving will run.

    python tools/k2_run_only_measure.py --gguf SHARD1.gguf --passages FROZEN.json --record OUT.json \\
        [--precisions a16w16,a14w14,baseline] [--T 924]"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parents[1]
for p in ("prover", "demo"):
    sys.path.insert(0, str(_ROOT / p))


def sequence(passages, T):
    """The longest frozen prompt, then the others' tokens as its continuation, to T."""
    ps = sorted(json.loads(pathlib.Path(passages).read_text())["passages"], key=lambda p: -len(p["ids"]))
    ids = list(ps[0]["ids"])
    n_prompt = len(ids)
    for p in ps[1:]:
        ids += p["ids"]
    return ids[:T], n_prompt


def main(argv=None):
    import torch
    import demo_k2 as dk
    import k2_float_torch as kt
    from k2_attention import K2_MLA
    from k2_loader import GgufWeights
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--passages", required=True)
    ap.add_argument("--record", required=True)
    ap.add_argument("--precisions", default="a16w16,a14w14,baseline")
    ap.add_argument("--T", type=int, default=924)
    a = ap.parse_args(argv)
    ids, P = sequence(a.passages, a.T)
    rec = dict(what=__doc__.split("\n\n")[0], T=len(ids), n_prompt=P, layers=2, runs={})

    # the pass is the driver's machinery: equal logits at the gate's length
    short, sp = ids[:100], 50
    logits, r = dk.run_only(a.gguf, short, sp, layers=2, precision="a16w16")
    b = dk.build(a.gguf, short[:sp], short[sp:], layers=2, precision="a16w16")
    got, _ = dk.engine_values(b)
    del b
    torch.cuda.empty_cache()
    S = dk.profile("a16w16").S_act
    rec["driver_equal"] = bool(np.array_equal(logits * S, got["logits"].reshape(logits.shape).astype(np.float64)))
    print("run-only logits equal the driver's engine pass at T=100:", rec["driver_equal"], flush=True)

    for prec in a.precisions.split(","):
        torch.cuda.reset_peak_memory_stats()
        logits, r = dk.run_only(a.gguf, ids, P, layers=2, precision=prec)
        run = r["run"]
        rec["runs"][prec] = dict(build_s=r["build_s"], run_s=r["run_s"], gpu_peak_gib=r.get("gpu_peak_gib"),
                                 tables=r["tables"]["entries"], ok=r["ok"], lookups_outside=run["lookups"]["outside"],
                                 field_unestablished=run["field"]["unestablished"], by_layer=run["field"]["by_layer"],
                                 U_bits=r["U_bits"])
        print(f"{prec}: build {r['build_s']:.1f} s, run {r['run_s']:.1f} s, peak {r.get('gpu_peak_gib', 0):.1f} GiB, "
              f"{'ok' if r['ok'] else 'NOT OK'}; widest {run['field']['by_layer']}", flush=True)
        del logits
        torch.cuda.empty_cache()

    W = GgufWeights(a.gguf, ints=False, cfg=K2_MLA)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    kt.forward(ids, kt.DeviceWeights(W, "cuda"), K2_MLA, layers=2, k=8, device="cuda", keep={"logits"})
    torch.cuda.synchronize()
    rec["p64"] = dict(seconds=time.time() - t0, gpu_peak_gib=torch.cuda.max_memory_allocated() / 2 ** 30)
    print(f"p64 (torch, cuda): {rec['p64']['seconds']:.1f} s, peak {rec['p64']['gpu_peak_gib']:.1f} GiB", flush=True)
    pathlib.Path(a.record).write_text(json.dumps(rec, indent=1, default=str))
    return 0 if rec["driver_equal"] and all(r["ok"] for r in rec["runs"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
