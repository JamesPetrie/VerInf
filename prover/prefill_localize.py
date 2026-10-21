"""Locating the prefill discrepancy (analysis/k2-fidelity-study-design.md §7.1,
review of 2026-10-21): on K2's weights at two layers, where llama.cpp's
batched path and its one-token path first part, intermediate by intermediate
in graph order, and which of the two is nearer p64 where an intermediate maps
onto ours. Each path also runs without captures, and its logits must be the
captured run's: a capture that changed the computation would localize the
wrong thing. CPU paths, from a CPU-only build (the GPU absent by
construction), are controls.

A layer comparison localizes; it cannot tell accumulated rounding from a
semantic error. The operation found first is then run alone, on identical
inputs, through the kernels of each path.

Paths, as session 12's E2 (n_prompt 1): "batch", positions 1 … T−1 in one
512-token batch; "steps", one token at a time. The intermediates are every
named node of each layer the scheduler prints (GGML_SCHED_DEBUG=2 on a short
run), captured by name and operation; one name and operation making several
tensors in a layer (the two query projections) are told apart by width.

    python prover/prefill_localize.py --runner R [--cpu-runner RC] --gguf TWO_LAYER.gguf \\
        --passages FROZEN.json --passage p1-research-review --work DIR [--engine JSON]"""
import argparse
import json
import pathlib
import re
import sys
import time

import numpy as np

import fidelity_adapters as fa
import llama_runner as lr

PATHS = {"batch": 512, "steps": 1}
# llama.cpp intermediates that map onto the p64 reference's (k2_float_torch with its attention trace)
P64 = {"attn_norm/MUL": "n1g", "kv_cmpr/MUL": "ckv", "kq_soft_max/SOFT_MAX": "sm", "kqv_out/CONT": "att",
       "ffn_inp/ADD": "r1", "ffn_norm/MUL": "n2g", "ffn_moe_probs/SIGMOID": "s", "ffn_moe_out/ADD": "y",
       "ffn_shexp/MUL_MAT": "sh", "l_out/ADD": "out"}
INTEGER_VALUED = ("ARGSORT", "TOP_K")
_NODE = re.compile(r"^node #\s*\d+ \(\s*([^)]*?)\):\s*(\S+)")


def log(*a):
    print(f"[localize {time.strftime('%H:%M:%S')}]", *a, flush=True)


def discover(runner, gguf, ids, work, settings, layers):
    """Every named node of each layer in graph order, as (name, op), from the
    scheduler's print of a short batched run."""
    out = pathlib.Path(work) / "discover"
    lr.run(runner, gguf, out, ids=ids[:8], mode="replay", n_prompt=1, cont_chunk=512, settings=settings,
           env={"GGML_SCHED_DEBUG": "2"})
    seen = []
    for line in (out / "run.log").read_text(errors="replace").splitlines():
        m = _NODE.match(line)
        if not m:
            continue
        op, name = m.group(1).strip(), m.group(2)
        layer = name.rsplit("-", 1)
        if len(layer) == 2 and layer[1].isdigit() and int(layer[1]) < layers and layer[0] != "norm":
            if (name, op) not in seen:
                seen.append((name, op))
    return seen


def _token_axis(arrays, n_tokens):
    """The axis (of ne3, ne2, ne1 as numpy's 0, 1, 2) whose size is each call's token count,
    from a call with more than one token; None when no call has."""
    for a, n in zip(arrays, n_tokens):
        if n > 1:
            hits = [ax for ax in (2, 1, 0) if a.shape[ax] == n]
            if len(hits) == 1:
                return hits[0]
    return None


def assemble(res, key, n_tokens, axes=None):
    """{(part, width): [one array per position]}: one capture's calls joined along the token
    axis, split into parts where one name and operation make several tensors per evaluation
    (the two query projections). The token axis is found on a batched path and passed
    (`axes`) to a one-token path, where every axis of a call may be 1. The attention's KV
    extent (kq, kq_soft_max: the row is the cells) is kept per position."""
    calls = lr.capture_calls(res, key)
    per_eval = len(calls) // len(n_tokens)
    if per_eval * len(n_tokens) != len(calls):
        raise ValueError(f"{key}: {len(calls)} calls over {len(n_tokens)} evaluations")
    groups = {}
    for j in range(per_eval):
        arrays = [calls[i * per_eval + j] for i in range(len(n_tokens))]
        ax = (axes or {}).get((j, arrays[0].shape[-1]))
        ax = _token_axis(arrays, n_tokens) if ax is None else ax
        if ax is None:
            return None                     # no token axis: the KV cells' extent (v_cont)
        rows = []
        for a in arrays:
            moved = np.moveaxis(a, ax, 0)                              # tokens first
            rows += [moved[t] for t in range(moved.shape[0])]
        groups[(j, arrays[0].shape[-1])] = rows
        groups.setdefault("axes", {})[(j, arrays[0].shape[-1])] = ax
    return groups


def compare_rows(a, b, causal=False):
    """Relative L2 difference and largest absolute difference over positions; for the
    attention's per-cell rows, only cells 0 … t of position t."""
    num = den = 0.0
    worst = 0.0
    for t, (x, y) in enumerate(zip(a, b)):
        if causal:
            x, y = x[..., :t + 1], y[..., :t + 1]
        else:
            w = min(x.shape[-1], y.shape[-1])
            x, y = x[..., :w], y[..., :w]
        d = np.asarray(x, np.float64) - np.asarray(y, np.float64)
        num += float((d ** 2).sum())
        den += float((np.asarray(y, np.float64) ** 2).sum())
        worst = max(worst, float(np.abs(d).max()) if d.size else 0.0)
    return dict(rel_l2=(num / den) ** 0.5 if den else 0.0, max_abs=worst)


def p64_rows(ref, key, il, T):
    """The p64 intermediate laid out as the capture's rows (one per position)."""
    v = ref[f"L{il}.{P64[key]}"]
    if P64[key] == "sm":                      # (H, T, T) -> per position (H, cells)
        return [v[:, t, :] for t in range(T)]
    return [v[t] for t in range(T)]


def run_path(runner, gguf, ids, work, name, chunk, settings, keys):
    plain = lr.run(runner, gguf, pathlib.Path(work) / f"{name}-plain", ids=ids, mode="replay", n_prompt=1,
                   cont_chunk=chunk, settings=settings)
    caught = lr.run(runner, gguf, pathlib.Path(work) / f"{name}-captured", ids=ids, mode="replay", n_prompt=1,
                    cont_chunk=chunk, settings=settings, capture=keys, captures_optional=True)
    T = len(ids)
    n_tokens = [1] + [min(chunk, T - 1 - i) for i in range(0, T - 1, chunk)]   # position 0, then the chunks
    return dict(caught=caught, n_tokens=n_tokens,
                capture_changes_logits=not np.array_equal(plain["logits"], caught["logits"]),
                skipped=caught["manifest"]["captures_skipped"], missing=caught["manifest"]["captures_missing"])


def localize(runner, gguf, ids, work, *, settings, layers=2, cpu_runner=None, spec=None, threshold=1e-4,
             ratio=1.25):
    work = pathlib.Path(work)
    nodes = discover(runner, gguf, ids, work, settings, layers)
    keys = [f"{n}/{op}" for n, op in nodes]
    log(len(keys), "intermediates over", layers, "layers")
    runs = {p: run_path(runner, gguf, ids, work, p, c, settings, keys) for p, c in PATHS.items()}
    if cpu_runner:
        cpu = dict(settings, n_gpu_layers=0)
        runs.update({f"cpu-{p}": run_path(cpu_runner, gguf, ids, work, f"cpu-{p}", c, cpu, keys)
                     for p, c in PATHS.items()})
    ref = None
    if spec is not None:
        import k2_float_torch as kt
        from k2_loader import GgufWeights
        W = GgufWeights(str(gguf), ints=False, cfg=spec["cfg"])
        want = {f"L{il}.{v}" for il in range(layers) for v in set(P64.values())}
        dev = "cuda" if _cuda() else "cpu"
        ref = kt.forward(ids, kt.DeviceWeights(W, dev), spec["cfg"], layers=layers, k=spec["k"], device=dev,
                         keep=want | {"logits"})
    rows, T = [], len(ids)
    for (name, op), key in zip(nodes, keys):
        il = int(name.rsplit("-", 1)[1])
        base = f"{name.rsplit('-', 1)[0]}/{op}"
        have = [p for p, r in runs.items() if key in r["caught"]["manifest"]["captures"]]
        if "batch" not in have or "steps" not in have:
            rows.append(dict(name=key, layer=il, op=op, captured=False))
            continue
        got = {"batch": assemble(runs["batch"]["caught"], key, runs["batch"]["n_tokens"])}
        if got["batch"] is None:
            rows.append(dict(name=key, layer=il, op=op, captured=True, compared=False,
                             reason="no token axis: the tensor spans the KV cells"))
            continue
        axes = got["batch"].pop("axes")
        for p in have:
            if p != "batch":
                got[p] = assemble(runs[p]["caught"], key, runs[p]["n_tokens"],
                                  axes=axes if PATHS.get(p.replace("cpu-", "")) == 1 else None)
                got[p].pop("axes", None)
        for g in got["batch"]:
            if g not in got["steps"]:
                continue
            causal = base in ("kq/MUL_MAT", "kq_soft_max/SOFT_MAX")
            row = dict(name=key, part=g[0], width=g[1], layer=il, op=op, captured=True)
            if op in INTEGER_VALUED:
                a, b = got["batch"][g], got["steps"][g]
                row["equal_fraction"] = float(np.mean([np.array_equal(x, y) for x, y in zip(a, b)]))
            else:
                row["batch_vs_steps"] = compare_rows(got["batch"][g], got["steps"][g], causal)
                for p in ("cpu-batch", "cpu-steps"):
                    if p in got and g in got[p]:
                        row[f"steps_vs_{p}"] = compare_rows(got["steps"][g], got[p][g], causal)
                        row[f"batch_vs_{p}"] = compare_rows(got["batch"][g], got[p][g], causal)
                if ref is not None and base in P64 and g[0] == 0:
                    r64 = p64_rows(ref, base, il, T)
                    row["p64"] = {p: compare_rows(got[p][g], r64, causal) for p in got if g in got[p]}
            rows.append(row)
    first_apart = next((r for r in rows if r.get("batch_vs_steps", {}).get("rel_l2", 0) > threshold), None)
    worse = [r for r in rows if "p64" in r and "batch" in r["p64"] and "steps" in r["p64"]
             and r["p64"]["batch"]["rel_l2"] > ratio * r["p64"]["steps"]["rel_l2"] and r["p64"]["batch"]["rel_l2"] > 1e-6]
    return dict(passage_tokens=T, layers=layers, settings=settings, paths={p: c for p, c in PATHS.items()},
                captures={p: dict(changed_logits=r["capture_changes_logits"], skipped=r["skipped"],
                                  missing=r["missing"]) for p, r in runs.items()},
                threshold=threshold, first_apart=first_apart, first_batch_worse_than_steps=worse[0] if worse else None,
                rows=rows)


def _cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def main(argv=None):
    import fidelity_study as fs
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runner", required=True)
    ap.add_argument("--cpu-runner", help="a CPU-only build's runner: the CPU paths as controls")
    ap.add_argument("--gguf", required=True, help="the two-layer truncation")
    ap.add_argument("--passages", required=True)
    ap.add_argument("--passage", default=None, help="the passage (default: the first)")
    ap.add_argument("--work", required=True)
    ap.add_argument("--engine", default="{}", help="JSON: engine settings beyond the adapter's")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--no-p64", action="store_true")
    a = ap.parse_args(argv)
    rec = json.loads(pathlib.Path(a.passages).read_text())
    p = next(x for x in rec["passages"] if a.passage in (None, x["id"]))
    spec = fs.MODELS["kimi-k2"]
    settings = dict(n_ctx=fs.N_CTX, threads=a.threads, n_gpu_layers=99, **spec["adapter"]["engine"],
                    **json.loads(a.engine))
    out = localize(a.runner, a.gguf, p["ids"], a.work, settings=settings, cpu_runner=a.cpu_runner,
                   spec=None if a.no_p64 else spec)
    out["passage"] = p["id"]
    dst = pathlib.Path(a.work) / "prefill-localize.json"
    dst.write_text(json.dumps(out, indent=1, default=str))
    for r in out["rows"]:
        if r.get("captured") is False:
            continue
        bs = r.get("batch_vs_steps", {})
        p64 = r.get("p64", {})
        log(f"{r['name']:34s} w={r['width']:<6} batch↔steps {bs.get('rel_l2', float('nan')):.2e}"
            + (f"  p64: batch {p64['batch']['rel_l2']:.2e} steps {p64['steps']['rel_l2']:.2e}" if "batch" in p64 else "")
            + (f"  equal {r['equal_fraction']:.3f}" if "equal_fraction" in r else ""))
    log("captures changed logits:", {k: v["changed_logits"] for k, v in out["captures"].items()})
    log("first apart:", out["first_apart"] and out["first_apart"]["name"],
        "| first where batch is worse against p64:", out["first_batch_worse_than_steps"]
        and out["first_batch_worse_than_steps"]["name"])
    log("record:", dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
