"""Multi-GPU execution qualified on its own (analysis/k2-fidelity-study-design.md
§7.1, review of 2026-10-21): the all-GPU deployment splits K2's layers across
cards, and a split run is not assumed to be the one-card run already measured.

On the two-layer truncation, each path (the batched replay, one token at a
time, and generation as the baseline runs it) runs on one card and split by
layer across two, with the scheduler's placement printed; the logits are
compared bit for bit, the placement must show both cards in use, and each
card's memory after the context is recorded.

    python tools/multigpu_qualify.py --runner R --gguf TWO_LAYER.gguf --passages FROZEN.json \\
        --work DIR [--engine JSON] [--cards 0,1]"""
import argparse
import json
import pathlib
import sys

import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "prover"))

import llama_runner as lr          # noqa: E402

PATHS = {"batch": dict(mode="replay", n_prompt=1, cont_chunk=512),
         "steps": dict(mode="replay", n_prompt=1, cont_chunk=1),
         "generate": dict(mode="generate", n_gen=64)}


def qualify(runner, gguf, ids, work, settings, cards):
    one = dict(settings, split_mode="none", main_gpu=0)
    split = dict(settings, split_mode="layer", tensor_split=",".join("1" for _ in cards))
    env_one = {"CUDA_VISIBLE_DEVICES": cards[0], "GGML_SCHED_DEBUG": "2"}
    env_split = {"CUDA_VISIBLE_DEVICES": ",".join(cards), "GGML_SCHED_DEBUG": "2"}
    out = {}
    for name, kw in PATHS.items():
        kw = dict(kw)
        if kw["mode"] == "generate":
            kw["n_prompt"] = len(ids)
        r1 = lr.run(runner, gguf, pathlib.Path(work) / f"{name}-one", ids=ids, env=env_one, settings=one, **kw)
        r2 = lr.run(runner, gguf, pathlib.Path(work) / f"{name}-split", ids=ids, env=env_split, settings=split, **kw)
        p1 = lr.placement((pathlib.Path(work) / f"{name}-one" / "run.log").read_text())
        p2 = lr.placement((pathlib.Path(work) / f"{name}-split" / "run.log").read_text())
        same = r1["logits"].shape == r2["logits"].shape and np.array_equal(r1["logits"], r2["logits"])
        diff = None if same or r1["logits"].shape != r2["logits"].shape else \
            float(np.abs(r1["logits"].astype(np.float64) - r2["logits"]).max())
        out[name] = dict(identical=same, max_abs_logit_difference=diff,
                         same_tokens=r1["tokens"] == r2["tokens"],
                         one_card=dict(nodes=p1["nodes_per_backend"], memory=r1["manifest"]["device_memory_after_context"]),
                         split=dict(nodes=p2["nodes_per_backend"], memory=r2["manifest"]["device_memory_after_context"],
                                    cards_used=sorted(b for b in p2["nodes_per_backend"] if b.startswith("CUDA"))))
        print(f"{name}: split {'IDENTICAL' if same else 'DIFFERS'} to one card"
              + (f" (max |Δ| {diff:.3g})" if diff is not None else "")
              + f"; split placement {p2['nodes_per_backend']}", flush=True)
    ok = all(r["split"]["cards_used"] == [f"CUDA{i}" for i in range(len(cards))] for r in out.values())
    return dict(cards=cards, settings=settings, paths=out, every_card_used=ok,
                identical=all(r["identical"] for r in out.values()))


def main(argv=None):
    import fidelity_study as fs
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runner", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--passages", required=True)
    ap.add_argument("--passage", default=None)
    ap.add_argument("--work", required=True)
    ap.add_argument("--engine", default="{}")
    ap.add_argument("--cards", default="0,1")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args(argv)
    rec = json.loads(pathlib.Path(a.passages).read_text())
    p = next(x for x in rec["passages"] if a.passage in (None, x["id"]))
    spec = fs.MODELS["kimi-k2"]
    settings = dict(n_ctx=fs.N_CTX, threads=a.threads, n_gpu_layers=99, **spec["adapter"]["engine"],
                    **json.loads(a.engine))
    out = qualify(a.runner, a.gguf, p["ids"], a.work, settings, a.cards.split(","))
    out["passage"] = p["id"]
    dst = pathlib.Path(a.work) / "multigpu-qualify.json"
    dst.write_text(json.dumps(out, indent=1, default=str))
    print("every card used:", out["every_card_used"], "| identical on every path:", out["identical"], "| record:", dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
