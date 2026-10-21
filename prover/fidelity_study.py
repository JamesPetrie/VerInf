"""The fidelity study's runs for one model of K2's family
(analysis/k2-fidelity-study-design.md §7): two layers, every arm, diagnostic;
and full depth, the engine's generation (the baseline) and the acceptance's
three arms. K2 runs it on the pods; Moonlight runs it locally as the
integration rehearsal, its integer arm the CPU integer model (the precision
study's baseline configuration, equal to the integer reference).

    python prover/fidelity_study.py two-layer --model moonlight --gguf M.gguf \\
        --passages FROZEN.json --runner R --render RENDER --work DIR
    python prover/fidelity_study.py full-depth --model moonlight --gguf M.gguf \\
        --passages FROZEN.json --runner R --render RENDER --work DIR

Two layers keep each integer configuration's ranges whole (every row's
signed extremes, bounds, claim and implementation; the violations per
implementation). --configs, --arms "" and --no-engine run the integer
configurations alone; --only runs some passages, and two-layer-merge
joins such records:

    python prover/fidelity_study.py two-layer ... --configs a16w16 --arms "" \\
        --no-engine --only p1-research-review --out p1.json
    python prover/fidelity_study.py two-layer-merge --work DIR --out all.json p1.json ...

Prompts: K2 uses the frozen token ids; another model renders the frozen
passages' user text with its own template (verinf_render). Records are JSON
in --work; nothing here decides anything the note has not frozen. Full
depth checkpoints each passage's p64 rows and each configuration's
per-position metrics in --work/full-depth-parts, so a stopped run started
again resumes at its last whole part."""
import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time

import numpy as np

import fidelity_adapters as fa
import fidelity_metrics as fm
import gguf_truncate as gt
import k2_emulation as em
import k2_float_reference as kf
import k2_int_reference as ki
import k2_precision_study as ps
import llama_runner as lr
from k2_attention import K2_MLA, MOONLIGHT_MLA
from k2_loader import GgufWeights, S_ACT, int_constants

MODELS = {"kimi-k2": dict(cfg=K2_MLA, adapter=fa.KIMI_K2, k=8),
          "moonlight": dict(cfg=MOONLIGHT_MLA, adapter=fa.MOONLIGHT, k=6)}
N_CTX = 1024                                   # pinned (§3.3)
CONTINUATION = 500
PARTS = "full-depth-parts"


def log(*a):
    print(f"[fidelity {time.strftime('%H:%M:%S')}]", *a, flush=True)


def prompts(model, passages_path, render, gguf, work):
    """{passage id: prompt ids}: K2's frozen ids, or another model's rendering
    of the same user text with its own template."""
    rec = json.loads(pathlib.Path(passages_path).read_text())
    if model == "kimi-k2":
        return {p["id"]: p["ids"] for p in rec["passages"]}
    src_path = pathlib.Path(passages_path).resolve().parent / pathlib.Path(rec["prompts_file"]).name
    raw = src_path.read_bytes()
    import hashlib
    if hashlib.sha256(raw).hexdigest() != rec["prompts_sha256"]:
        raise ValueError(f"{src_path} is not the prompts file the passages were frozen from")
    src = json.loads(raw)
    out = {}
    for p in src["passages"]:
        d = pathlib.Path(work) / "render" / p["id"]
        d.mkdir(parents=True, exist_ok=True)
        (d / "user.txt").write_text(p["messages"][0]["content"])
        subprocess.run([str(render), "--model", str(gguf), "--content", str(d / "user.txt"),
                        "--out", str(d)], check=True)
        out[p["id"]] = [int(x) for x in (d / "ids.txt").read_text().split()]
    return out


# ---- per arm ----------------------------------------------------------------------

def _gates_p64(out, il, W, k, E):
    s = out[f"L{il}.s"]
    top = out[f"L{il}.top"]
    w = np.take_along_axis(s, top, axis=1)
    return top, fm.gate_matrix(top, w / w.sum(axis=1, keepdims=True), E)


def _gates_int(out, il, E):
    m = np.asarray(out[f"L{il}.mask"])
    top = np.sort(np.argsort(-m, axis=1, kind="stable")[:, :int(m[0].sum())], axis=1)
    return top, None


def explain(ref, arm, W, layers, k, *, gates=None):
    """§4's explaining quantities for one arm against p64: per-layer residual
    error, and per MoE layer the support differences, the normalized gate
    distance (where the arm's gates are known) and p64's selection margins."""
    rows = {}
    for il in range(layers):
        if f"L{il}.out" in arm:
            e = fm.residual_error(ref[f"L{il}.out"], arm[f"L{il}.out"])
            rows[f"L{il}.residual"] = dict(mean=float(e.mean()), max=float(e.max()))
        if f"L{il}.s" not in ref or W.kind(il) != "moe":
            continue
        E = W.n_experts(il)
        top_ref, g_ref = _gates_p64(ref, il, W, k, E)
        top_arm = arm.get(f"L{il}.top")
        if top_arm is None:
            continue
        d = fm.support_differences(top_ref, top_arm)
        q = ref[f"L{il}.s"] + W.bias(il)[None, :]
        marg = fm.selection_margins(q, k)
        r = dict(tokens=len(d), tokens_differing=int((d > 0).sum()), experts_differing=int(d.sum()),
                 median_margin=float(np.median(marg)),
                 margins_of_differing=[float(marg[t]) for t in np.nonzero(d)[0][:32]])
        g_arm = (gates or {}).get(il)
        if g_arm is not None:
            gd = fm.gate_distance(g_ref, g_arm)
            sw = fm.swapped_weight(g_ref, g_arm)
            r.update(gate_distance_mean=float(gd.mean()), gate_distance_max=float(gd.max()),
                     swapped_dropped_mean=float(sw["dropped"].mean()))
        rows[f"L{il}.routing"] = r
    return rows


def summary(pm):
    return dict(mean_kl=float(pm["kl"].mean()), p99_kl=fm.p99(pm["kl"]), max_kl=float(pm["kl"].max()),
                mean_nll=float(pm["nll"].mean()))


# ---- two layers ---------------------------------------------------------------------

def violations_by_implementation(ranges) -> dict:
    """Each implementation's rows outside their constraint, whole: the
    quantity (keyed "name [implementation]" where the row is one
    implementation's own; its `implementation` None where it holds for
    both), its signed extremes, the inclusive bounds and the claim."""
    rows = ranges.as_dict()
    return {i: {key: rows[key] for key in sorted(ranges.violations(i))} for i in ps.IMPLEMENTATIONS}


def two_layer_feasibility(passages) -> dict:
    """Each integer configuration's ranges over every passage (the extremes
    of all; k2_precision_study.feasibility), its violations whole per
    implementation, and every row."""
    out = {}
    names = dict.fromkeys(key[4:] for p in passages.values() for key in p if key.startswith("int/"))
    for name in names:
        rg = ki.Ranges.from_dicts(*(p[f"int/{name}"]["ranges"] for p in passages.values()
                                    if f"int/{name}" in p))
        out[name] = dict(ps.feasibility(rg), violations=violations_by_implementation(rg), ranges=rg.as_dict())
    return out


def log_feasibility(feasibility):
    for name, f in feasibility.items():
        for impl, rows in f["violations"].items():
            log(f"int/{name} {impl}:", "every range inside its constraint" if not rows else
                f"{len(rows)} outside their constraint")
            for key, r in rows.items():
                log(f"    {key}: [{r['min']}, {r['max']}] against {r['inequality']} ({r['claim']})")


def truncated(gguf, runner, work):
    """The two-layer GGUF the engine runs (§5.2), in `work`, checked to load
    on the full model's branch."""
    trunc = pathlib.Path(work) / "two-layer.gguf"
    if not trunc.exists():
        rec = gt.truncate(gguf, trunc, 2)
        log("truncated:", rec["added_keys"] or "no keys added")
    two_fp = lr.run(runner, trunc, pathlib.Path(work) / "info-two", mode="info")["fingerprint"]
    try:
        full_fp = lr.run(runner, gguf, pathlib.Path(work) / "info-full", mode="info")["fingerprint"]
        branch = lr.branch_differences(full_fp, two_fp)
    except RuntimeError:                  # one shard of a split model: its metadata decides
        want = gt.deepseek2_expected_fingerprint(gt.metadata(gguf))
        branch = {k: (v, two_fp.get(k)) for k, v in want.items() if two_fp.get(k) != v}
        log("the full model does not load here; the branch checked against its metadata")
    if branch:
        raise RuntimeError(f"the truncation leaves the full model's branch: {branch}")
    return trunc


def two_layer(spec, gguf, ids_by_passage, runner, work, *, configs=tuple(ps.CONFIGS),
              arms=tuple(em.ARMS), threads=8, engine=None, with_engine=True):
    """Every arm at two layers on each passage's prompt, every position
    scored. `spec` is a MODELS entry (cfg, adapter, k). Each integer
    configuration's ranges are kept whole, per passage and over all of them
    (two_layer_feasibility). with_engine=False runs no llama.cpp: with no
    `arms`, the integer configurations alone."""
    cfg, k, adapter = spec["cfg"], spec["k"], spec["adapter"]
    work = pathlib.Path(work)
    work.mkdir(parents=True, exist_ok=True)
    if with_engine:
        trunc = truncated(gguf, runner, work)
    Wf = ps.CachedFloats(GgufWeights(str(gguf), ints=False, cfg=cfg))
    out = dict(model=adapter["name"], layers=2, passages={})
    for pid, ids in ids_by_passage.items():
        log(pid, len(ids), "tokens")
        T = len(ids)
        tgt = ids[1:]
        ref = kf.forward(ids, Wf, cfg, layers=2, k=k)
        p = {"p64": dict(n=T)}
        for name in configs:
            t0 = time.time()
            io, ranges = ps.forward(ids, Wf, cfg, ps.CONFIGS[name], layers=2, kk=k)
            A = 1 << ps.CONFIGS[name]["S_act"]
            lg = np.asarray(io["logits"], dtype=np.float64) / A
            io_f = {n: np.asarray(v, dtype=np.float64) / A for n, v in io.items()
                    if n.endswith(".out")}
            top = {f"L{il}.top": _gates_int(io, il, None)[0] for il in range(2) if f"L{il}.mask" in io}
            p[f"int/{name}"] = dict(**summary(fm.position_metrics(ref["logits"][:-1], lg[:-1], tgt)),
                                    explain=explain(ref, {**io_f, **top}, Wf, 2, k),
                                    violations=violations_by_implementation(ranges), ranges=ranges.as_dict(),
                                    seconds=time.time() - t0)
        for arm in arms:
            t0 = time.time()
            P = em.Policy(*em.ARMS[arm])
            eo = em.forward(ids, Wf, cfg, P, layers=2, k=k)
            gates = {il: _gates_p64(eo, il, Wf, k, Wf.n_experts(il))[1]
                     for il in range(2) if f"L{il}.s" in eo}
            p[f"emu/{arm}"] = dict(**summary(fm.position_metrics(ref["logits"][:-1], eo["logits"][:-1], tgt)),
                                   explain=explain(ref, eo, Wf, 2, k, gates=gates), seconds=time.time() - t0)
        if not with_engine:
            out["passages"][pid] = p
            log(pid, *(f"{key} mean KL {round(v['mean_kl'], 6)}" for key, v in p.items() if key != "p64"))
            continue
        t0 = time.time()
        r = lr.run(runner, trunc, work / "engine" / pid, ids=ids, mode="replay", n_prompt=1,
                   prompt_chunk=512, cont_chunk=512, capture=fa.captures(adapter, 2),
                   settings=dict(n_ctx=N_CTX, threads=threads, **(engine or {}), **adapter["engine"]))
        eng = fa.ours(adapter, r["captures"], Wf.n_experts(1))
        p["engine"] = dict(**summary(fm.position_metrics(ref["logits"][:-1], r["logits"], tgt)),
                           explain=explain(ref, eng, Wf, 2, k,
                                           gates={il: eng[f"L{il}.gates"] for il in range(2)
                                                  if f"L{il}.gates" in eng}),
                           manifest=r["manifest"]["settings"], seconds=time.time() - t0)
        out["passages"][pid] = p
        log(pid, "engine mean KL", round(p["engine"]["mean_kl"], 6),
            *(f"int {name} {round(p[f'int/{name}']['mean_kl'], 6)}" for name in configs))
    out["feasibility"] = two_layer_feasibility(out["passages"])
    log_feasibility(out["feasibility"])
    return out


def merge_two_layer(records) -> dict:
    """Two-layer records of disjoint passages (one process each) as one: the
    passages in the order given, the feasibility over all of them."""
    out = dict(model=records[0]["model"], layers=records[0]["layers"], passages={})
    for r in records:
        if (r["model"], r["layers"]) != (out["model"], out["layers"]):
            raise ValueError(f"{r['model']} at {r['layers']} layers is not {out['model']} at {out['layers']}")
        if out["passages"].keys() & r["passages"].keys():
            raise ValueError(f"passages in two records: {sorted(out['passages'].keys() & r['passages'].keys())}")
        out["passages"].update(r["passages"])
    out["feasibility"] = two_layer_feasibility(out["passages"])
    return out


# ---- the engine's kernel paths at two layers ---------------------------------------

# name: (n_gpu_layers, continuation chunk, environment). The batch is how the
# two-layer stage replays (positions 1.. as one 512-token chunk); the steps
# are how generation decodes, one token at a time.
ENGINE_PATHS = {"gpu-batch": (99, 512, {}), "gpu-steps": (99, 1, {}),
                "gpu-batch-no-tf32": (99, 512, {"NVIDIA_TF32_OVERRIDE": "0"}),
                "cpu-batch": (0, 512, {}), "cpu-steps": (0, 1, {})}


def engine_paths(spec, gguf, ids_by_passage, runner, work, *, paths=tuple(ENGINE_PATHS), threads=8,
                 engine=None, backend="numpy", device=None):
    """The two-layer engine arm along each kernel path of ENGINE_PATHS, each
    scored against p64 on the same passages; p64 once per passage, from
    `reference(backend, device)`. The settings are two_layer's engine arm's,
    `engine` included, with n_gpu_layers and the chunk set by the path."""
    cfg, k, adapter = spec["cfg"], spec["k"], spec["adapter"]
    work = pathlib.Path(work)
    work.mkdir(parents=True, exist_ok=True)
    trunc = truncated(gguf, runner, work)
    ident, p64_forward = reference(backend, device)
    W = GgufWeights(str(gguf), ints=False, cfg=cfg)
    out = dict(model=adapter["name"], layers=2, reference=ident, paths={n: ENGINE_PATHS[n] for n in paths},
               passages={})
    for pid, ids in ids_by_passage.items():
        t0 = time.time()
        ref = np.asarray(p64_forward(ids, W, cfg, 2, k))
        p = dict(p64=dict(n=len(ids), seconds=time.time() - t0))
        for name in paths:
            ngl, chunk, env = ENGINE_PATHS[name]
            t0 = time.time()
            settings = dict(n_ctx=N_CTX, threads=threads, **(engine or {}), **adapter["engine"])
            settings["n_gpu_layers"] = ngl
            r = lr.run(runner, trunc, work / "engine" / name / pid, ids=ids, mode="replay", n_prompt=1,
                       prompt_chunk=512, cont_chunk=chunk, settings=settings, env=env)
            p[name] = dict(**summary(fm.position_metrics(ref[:-1], r["logits"], ids[1:])),
                           settings=r["manifest"]["settings"], environment=r["manifest"]["environment"],
                           logits_sha256=hashlib.sha256(np.ascontiguousarray(r["logits"]).tobytes()).hexdigest(),
                           seconds=time.time() - t0)
        out["passages"][pid] = p
        log(pid, *(f"{n} {p[n]['mean_kl']:.3e}" for n in paths))
    out["pooled"] = {}
    for name in paths:
        rows = [out["passages"][pid][name] for pid in out["passages"]]
        n = [out["passages"][pid]["p64"]["n"] - 1 for pid in out["passages"]]
        out["pooled"][name] = dict(mean_kl=float(np.average([r["mean_kl"] for r in rows], weights=n)),
                                   max_kl=max(r["max_kl"] for r in rows))
    return out


# ---- full depth ---------------------------------------------------------------------

def generate(spec, gguf, ids_by_passage, runner, work, *, threads=8, n_gen=CONTINUATION, engine=None):
    """The baseline (§3.3): each prompt prefilled, then greedy one-token
    decoding, the choosing logits captured; the sequences frozen."""
    adapter = spec["adapter"]
    out = {}
    for pid, ids in ids_by_passage.items():
        t0 = time.time()
        r = lr.run(runner, gguf, pathlib.Path(work) / "generate" / pid, ids=ids, mode="generate",
                   n_prompt=len(ids), n_gen=n_gen, prompt_chunk=512,
                   settings=dict(n_ctx=N_CTX, threads=threads, **(engine or {}), **adapter["engine"]))
        out[pid] = dict(tokens=r["tokens"], n_prompt=len(ids), logits=r["logits"],
                        sha256=lr.ids_sha256(r["tokens"]), seconds=time.time() - t0)
        log(pid, "generated", len(r["tokens"]) - len(ids), "tokens in", round(time.time() - t0, 1), "s")
    return out


def _sources_sha256() -> str:
    """The prover tree's sources: a part is reused only by the code that wrote it."""
    h = hashlib.sha256()
    for f in sorted(pathlib.Path(__file__).resolve().parent.glob("*.py")):
        h.update(f.name.encode() + b"\0" + f.read_bytes())
    return h.hexdigest()


def _gguf_identity(gguf):
    """Each shard's name, size and modification time (hashing K2's would take an hour)."""
    return [[p.name, p.stat().st_size, p.stat().st_mtime_ns] for p in map(pathlib.Path, gt.shard_paths(gguf))]


def _part(path, key, compute):
    """`compute()`'s arrays, saved at `path` under `key`; or loaded, when an
    earlier run left them there under the same key. A part under another key
    is computed again and replaced. Returns (arrays, loaded)."""
    path = pathlib.Path(path)
    key = json.loads(json.dumps(key))
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            if json.loads(str(z["key"])) == key:
                return {n: z[n] for n in z.files if n != "key"}, True
        log(path, "was written under another key: computing it again")
    out = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, key=np.array(json.dumps(key)), **out)
    os.replace(tmp, path)                               # whole, or not there
    return out, False


def reference(backend="numpy", device=None):
    """The p64 forward a full-depth run uses, `fn(ids, W, cfg, layers, k)` ->
    the logits, and its identity for the record and the parts' key: the
    backend, the resolved device, the dtype and the library's version. NumPy
    runs on the CPU; torch (k2_float_torch, the bulk decoded on the device)
    on the CPU or a GPU."""
    if backend == "numpy":
        if device not in (None, "cpu"):
            raise ValueError(f"the numpy reference runs on the cpu, not {device!r}")
        return (dict(backend="numpy", device="cpu", dtype="float64", version=np.__version__),
                lambda ids, W, cfg, layers, k: kf.forward(ids, W, cfg, layers=layers, k=k)["logits"])
    if backend == "torch":
        import torch
        import k2_float_torch as kt
        dev = torch.device(device or "cpu")
        if dev.type == "cuda":
            dev = torch.device("cuda", torch.cuda.current_device() if dev.index is None else dev.index)
        name = torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu"

        def run(ids, W, cfg, layers, k):
            return kt.forward(ids, kt.DeviceWeights(W, dev), cfg, layers=layers, k=k, device=dev,
                              keep={"logits"})["logits"]
        return dict(backend="torch", device=str(dev), device_name=name, dtype="float64",
                    version=torch.__version__), run
    raise ValueError(f"unknown reference backend {backend!r}")


def _register_primary(parts, primary):
    """The work directory's primary configuration, recorded by its first run;
    a later run naming another is refused."""
    path = pathlib.Path(parts) / "primary.json"
    if path.exists():
        was = json.loads(path.read_text())["primary"]
        if was != primary:
            raise ValueError(f"{path.parent} was started with primary {was!r}; this run names {primary!r}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(dict(primary=primary)) + "\n")
    os.replace(tmp, path)


IMPLEMENTATIONS = ("numpy", "run-only")


def _driver_sha256() -> str:
    """The K2 driver's source: the run-only implementation's parts are keyed by it too."""
    return hashlib.sha256((pathlib.Path(__file__).resolve().parents[1] / "demo" / "demo_k2.py").read_bytes()).hexdigest()


def _run_only_arm(spec, gguf, n_layers, name):
    """A configuration's integer logits from the K2 driver's tape run once
    without a proof (demo_k2.run_only): the GPU's arithmetic, exact to the
    NumPy model's, its lookups and field safety checked as it runs."""
    if spec["adapter"]["name"] != fa.KIMI_K2["name"]:
        raise ValueError("the run-only implementation is the K2 driver's: K2's family members need their own")
    demo = str(pathlib.Path(__file__).resolve().parents[1] / "demo")
    if demo not in sys.path:
        sys.path.insert(0, demo)
    import demo_k2 as dk

    def run(ids, P):
        logits, rec = dk.run_only(str(gguf), list(ids), P, cfg=spec["cfg"], layers=n_layers, kk=spec["k"],
                                  precision=name)
        return logits, rec
    return run


def run_only_feasibility(recs) -> dict:
    """Proof feasibility from the run-only pass over every passage: every key
    inside its table and every claim's output within the field by its bound,
    on each passage; the widest established value per layer."""
    by_layer = {}
    for r in recs:
        for layer, row in r["run"]["field"]["by_layer"].items():
            b = by_layer.setdefault(layer, dict(max_bits=0.0, unestablished=0))
            b["max_bits"] = max(b["max_bits"], row["max_bits"])
            b["unestablished"] += row["unestablished"]
    return dict(implementation="run-only", ok=all(r["ok"] for r in recs),
                lookups_outside=[o for r in recs for o in r["run"]["lookups"]["outside"]],
                field_unestablished=[u for r in recs for u in r["run"]["field"]["unestablished"]],
                unknown_types=sorted({t for r in recs for t in r["run"]["field"]["unknown_types"]}),
                max_bits_by_layer=by_layer, U_bits=[r["U_bits"] for r in recs],
                seconds=dict(build=[r["build_s"] for r in recs], run=[r["run_s"] for r in recs]),
                witness=recs[0]["run"]["witness"] if recs else None)


def full_depth(spec, gguf, generated, work, *, primary, exploratory=(), layers=None, integer=None,
               backend="numpy", device=None, implementation="numpy"):
    """p64 and the integer model on each frozen sequence; per-position KL and
    NLL over the continuation for the integer arm and the engine baseline;
    §6's acceptance and the §6.5 calibration. `integer(ids)` returns the
    integer arm's real-valued logits (`primary` is then "integer"); by
    default the integer model runs at `primary` and at each `exploratory`
    configuration, all against one p64 and one baseline: the CPU model
    (`implementation` "numpy"), or the K2 driver's tape run once on the GPU
    without a proof ("run-only", demo_k2.run_only).

    p64 is computed for every passage before any integer arm runs, so the
    reference is complete whatever happens to the arms.

    The record's acceptance is the primary's, named before the run and kept
    by the work directory (a resume naming another primary is refused); the
    exploratory configurations have their own under "exploratory", and none
    replaces the primary's verdict. p64 is `reference(backend, device)`'s,
    recorded under "reference".

    Proof feasibility is recorded beside it and never enters it: for the
    CPU model, each configuration's ranges over every passage against the
    claims' constraints, per implementation (k2_precision_study.feasibility);
    for the run-only pass, its lookups and field safety (run_only_feasibility).

    Each passage's p64 rows (the continuation's) and each configuration's
    per-position metrics are parts in `work`/full-depth-parts, keyed by the
    model, the layers, the GGUF's shards, the prover's sources (and the
    driver's, for the run-only pass), the p64 reference's identity, the
    sequence, the configuration and its implementation, under one directory
    per reference (so switching never discards either's); a run finds the
    parts an earlier one finished and resumes after them. An `integer`
    callable's metrics are never saved, and it has no ranges."""
    cfg, k = spec["cfg"], spec["k"]
    if implementation not in IMPLEMENTATIONS:
        raise ValueError(f"implementation {implementation!r}: one of {IMPLEMENTATIONS}")
    W = GgufWeights(str(gguf), ints=False, cfg=cfg)
    md = gt.metadata(gguf)
    n_layers = layers or int(md[f"{md['general.architecture']}.block_count"])
    if primary in exploratory:
        raise ValueError(f"{primary!r} is the primary; it is not also exploratory")
    if integer is not None and (primary != "integer" or exploratory):
        raise ValueError("an integer callable is the one configuration, primary 'integer'")
    configs = (primary, *exploratory)
    if integer is not None:
        arms = {"integer": (lambda ids, P: (integer(ids), None), None)}
    elif implementation == "run-only":
        arms = {name: (_run_only_arm(spec, gguf, n_layers, name), dict(ps.CONFIGS[name], implementation="run-only",
                                                                        driver=_driver_sha256()))
                for name in configs}
    else:
        def make(name):
            def run(ids, P):
                io, ranges = ps.forward(ids, W, cfg, ps.CONFIGS[name], layers=n_layers, kk=k)
                return (np.asarray(io["logits"], dtype=np.float64) / (1 << ps.CONFIGS[name]["S_act"]),
                        ranges.as_dict())
            return run
        arms = {name: (make(name), ps.CONFIGS[name]) for name in configs}
    ident, p64_forward = reference(backend, device)
    _register_primary(pathlib.Path(work) / PARTS, primary)
    parts = pathlib.Path(work) / PARTS / f"{ident['backend']}-{ident['device'].replace(':', '')}"
    base = dict(model=spec["adapter"]["name"], layers=n_layers, k=k, gguf=_gguf_identity(gguf),
                sources=_sources_sha256(), reference=ident)
    rec = dict(model=spec["adapter"]["name"], layers=n_layers, reference=ident, implementation=implementation,
               passages={})

    # the reference, every passage, before any arm
    refs = {}
    for pid, g in generated.items():
        ids, P = g["tokens"], g["n_prompt"]
        rows = slice(P - 1, len(ids) - 1)                  # positions scoring the continuation

        def p64():
            t0 = time.time()
            ref = np.asarray(p64_forward(ids, W, cfg, n_layers, k))
            if ref.dtype != np.dtype(ident["dtype"]):
                raise TypeError(f"the {ident['backend']} reference gave {ref.dtype}, not {ident['dtype']}")
            return dict(ref=ref[rows], seconds=np.float64(time.time() - t0))
        key = dict(base, sequence=lr.ids_sha256(ids), n_prompt=P)
        part, loaded = _part(parts / pid / "p64.npz", key, p64)
        refs[pid] = (part, loaded, key)
        log(pid, "p64", "(resumed)" if loaded else f"{float(part['seconds']):.0f} s")

    per = {name: [] for name in arms}
    per_found = {name: [] for name in arms}
    per_eng, per_p64 = [], []
    for pid, g in generated.items():
        ids, P = g["tokens"], g["n_prompt"]
        rows = slice(P - 1, len(ids) - 1)
        tgt = ids[P:]
        part, loaded, key = refs[pid]
        ref = part["ref"]
        resumed = ["p64"] if loaded else []
        secs = dict(p64=float(part["seconds"]), generate=g.get("seconds"))
        pe = fm.position_metrics(ref, g["logits"], tgt)
        per_eng.append(pe)
        per_p64.append(fm.nll_bits(ref, tgt))
        prow = dict(scored=len(tgt), engine=summary(pe), p64_mean_nll=float(per_p64[-1].mean()),
                    seconds=secs, integer={}, resumed=resumed)
        for name, (fn, settings) in arms.items():
            def score():
                t0 = time.time()
                logits, found = fn(ids, P)
                pi = fm.position_metrics(ref, logits[rows], tgt)
                out = dict(kl=pi["kl"], nll=pi["nll"], seconds=np.float64(time.time() - t0))
                if found is not None:
                    out["ranges"] = np.array(json.dumps(found, default=str))   # the name earlier parts use
                return out
            if settings is None:
                part_i, loaded_i = score(), False
            else:
                part_i, loaded_i = _part(parts / pid / (f"{name}.npz" if implementation == "numpy"
                                                        else f"{name}.{implementation}.npz"),
                                         dict(key, config=name, settings=settings), score)
            pi = dict(kl=part_i["kl"], nll=part_i["nll"])
            secs[name] = float(part_i["seconds"])
            if loaded_i:
                resumed.append(name)
            per[name].append(pi)
            per_found[name].append(json.loads(str(part_i["ranges"])) if "ranges" in part_i else None)
            prow["integer"][name] = summary(pi)
            log(pid, name, *(["(resumed)"] if loaded_i else []), prow["integer"][name])
        log(pid, "engine", prow["engine"])
        rec["passages"][pid] = prow
    rec["primary"] = primary
    rec["acceptance"] = fm.acceptance(per[primary], per_eng)
    rec["exploratory"] = {name: fm.acceptance(per[name], per_eng) for name in exploratory}
    rec["nll_calibration"] = fm.nll_calibration(per_p64, [e["nll"] for e in per_eng])
    rec["feasibility"] = {}
    for name, found in per_found.items():
        if not found or any(f is None for f in found):
            continue
        if implementation == "run-only":
            f = rec["feasibility"][name] = run_only_feasibility(found)
            log(name, "run-only:", "every key inside its table, every bound within the field" if f["ok"] else
                f"{len(f['lookups_outside'])} lookups outside, {len(f['field_unestablished'])} not established",
                "widest", max((b["max_bits"] for b in f["max_bits_by_layer"].values()), default=0), "bits")
        else:
            rg = ki.Ranges.from_dicts(*found)
            rec["feasibility"][name] = dict(ps.feasibility(rg), ranges=rg.as_dict())
            f = rec["feasibility"][name]
            log(name, "feasible:", {i: v["ok"] for i, v in f["implementations"].items()},
                "output width", f["output_width_needed"], "RMSNorm carries", f["rmsnorm_carries"])
    return rec


# ---- the reusable artifacts: the engine's generation and the float64 reference ---------

ENGINE_ARTIFACT, REFERENCE_ARTIFACT = "engine-artifact", "reference-artifact"
SCORED = "rows n_prompt − 1 .. T − 2: row 0 is the prefill's prediction of the first continuation token"


def _sha256_file(path) -> str:
    return lr.file_sha256(path)


def _write_sums(d):
    d = pathlib.Path(d)
    lines = [f"{_sha256_file(f)}  {f.name}\n" for f in sorted(d.iterdir()) if f.name != "SHA256SUMS"]
    (d / "SHA256SUMS").write_text("".join(lines))


def _check_sums(d) -> None:
    d = pathlib.Path(d)
    listed = {}
    for line in (d / "SHA256SUMS").read_text().splitlines():
        h, name = line.split("  ", 1)
        listed[name] = h
    present = {f.name for f in d.iterdir() if f.name != "SHA256SUMS"}
    if present != set(listed):
        raise ValueError(f"{d}: files {sorted(present ^ set(listed))} are not as SHA256SUMS lists them")
    bad = [n for n, h in listed.items() if _sha256_file(d / n) != h]
    if bad:
        raise ValueError(f"{d}: {bad} differ from SHA256SUMS")


def tokenizer_sha256(gguf) -> str:
    """The GGUF's tokenizer: every tokenizer.* key-value (the vocabulary, its
    types and merges, the special ids, the chat template), hashed."""
    md = gt.metadata(gguf)
    tok = {k: v for k, v in md.items() if k.startswith("tokenizer.")}
    blob = json.dumps(tok, sort_keys=True, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))
    return hashlib.sha256(blob.encode()).hexdigest()


def write_engine_artifact(spec, gguf, generated, work, *, manifests, placements=None, model_sha256=None,
                          passages_sha256=None, attach=()):
    """F1's reusable record (§3.3, review of 2026-10-21), enough to score any
    arithmetic later without generating again: per passage the full token
    sequence, the prompt boundary, whether the continuation ended at EOS or
    at the cap, and the logits that chose each token as they were produced
    (SCORED); with the engine's complete configuration (the runner's
    manifest: settings, devices, their memory, the precision environment),
    the scheduler's placement where it was printed, and the model's and
    tokenizer's hashes. Refused when flash attention was left to llama.cpp
    ("auto"): the frozen configuration names it."""
    d = pathlib.Path(work) / ENGINE_ARTIFACT
    d.mkdir(parents=True, exist_ok=True)
    man = dict(kind="engine", model=spec["adapter"]["name"], n_ctx=N_CTX, continuation_cap=CONTINUATION,
               scored=SCORED, gguf=_gguf_identity(gguf), model_sha256=model_sha256,
               tokenizer_sha256=tokenizer_sha256(gguf), passages_sha256=passages_sha256, passages={})
    for pid, g in generated.items():
        m = manifests[pid]
        fa_setting = m["settings"]["flash_attn"]
        if fa_setting not in ("on", "off"):
            raise ValueError(f"{pid}: flash attention {fa_setting!r}; the frozen configuration names it on or off")
        tokens, P = list(g["tokens"]), int(g["n_prompt"])
        logits = np.ascontiguousarray(g["logits"], dtype=np.float32)
        if logits.shape[0] != len(tokens) - P:
            raise ValueError(f"{pid}: {logits.shape[0]} logit rows for {len(tokens) - P} continuation tokens")
        eos = bool(m["ended_by_eog"])
        np.savez(d / f"{pid}.npz", tokens=np.asarray(tokens, dtype=np.int32), n_prompt=np.int64(P), logits=logits)
        man["passages"][pid] = dict(n_prompt=P, n_tokens=len(tokens), n_scored=len(tokens) - P,
                                    ended_by="eos" if eos else "cap", last_token=int(tokens[-1]),
                                    tokens_sha256=lr.ids_sha256(tokens),
                                    logits_sha256=hashlib.sha256(logits.tobytes()).hexdigest(),
                                    runner=m, placement=(placements or {}).get(pid))
    for f in attach:
        f = pathlib.Path(f)
        (d / f.name).write_bytes(f.read_bytes())
        man.setdefault("attached", []).append(f.name)
    (d / "manifest.json").write_text(json.dumps(man, indent=1, default=str))
    _write_sums(d)
    log("engine artifact:", d)
    return d


def write_reference_artifact(spec, gguf, generated, work, *, layers=None, backend="numpy", device=None,
                             model_sha256=None):
    """F2's reusable record: p64's logits at the scored rows of every frozen
    sequence, from the full-depth parts (computed here when absent), with the
    reference's identity. It is complete before, and whatever, any integer
    arm does."""
    cfg, k = spec["cfg"], spec["k"]
    W = GgufWeights(str(gguf), ints=False, cfg=cfg)
    md = gt.metadata(gguf)
    n_layers = layers or int(md[f"{md['general.architecture']}.block_count"])
    ident, p64_forward = reference(backend, device)
    parts = pathlib.Path(work) / PARTS / f"{ident['backend']}-{ident['device'].replace(':', '')}"
    base = dict(model=spec["adapter"]["name"], layers=n_layers, k=k, gguf=_gguf_identity(gguf),
                sources=_sources_sha256(), reference=ident)
    d = pathlib.Path(work) / REFERENCE_ARTIFACT
    d.mkdir(parents=True, exist_ok=True)
    man = dict(kind="reference", model=spec["adapter"]["name"], layers=n_layers, reference=ident,
               sources=base["sources"], gguf=base["gguf"], model_sha256=model_sha256, scored=SCORED, passages={})
    for pid, g in generated.items():
        ids, P = g["tokens"], g["n_prompt"]

        def p64():
            t0 = time.time()
            ref = np.asarray(p64_forward(ids, W, cfg, n_layers, k))
            return dict(ref=ref[P - 1:len(ids) - 1], seconds=np.float64(time.time() - t0))
        part, loaded = _part(parts / pid / "p64.npz", dict(base, sequence=lr.ids_sha256(ids), n_prompt=P), p64)
        ref = np.ascontiguousarray(part["ref"], dtype=np.float64)
        np.savez(d / f"{pid}.npz", p64=ref)
        man["passages"][pid] = dict(n_prompt=P, n_tokens=len(ids), tokens_sha256=lr.ids_sha256(ids),
                                    p64_sha256=hashlib.sha256(ref.tobytes()).hexdigest(),
                                    seconds=float(part["seconds"]), resumed=loaded)
        log(pid, "p64", "(resumed)" if loaded else f"{float(part['seconds']):.0f} s")
    (d / "manifest.json").write_text(json.dumps(man, indent=1, default=str))
    _write_sums(d)
    log("reference artifact:", d)
    return d


def load_artifacts(engine_dir, reference_dir) -> dict:
    """Both artifacts, every file checked against its SHA256SUMS, the two
    checked to be of the same sequences: {passage: tokens, n_prompt, the
    engine's logits, p64's}, with both manifests."""
    _check_sums(engine_dir)
    _check_sums(reference_dir)
    em_ = json.loads((pathlib.Path(engine_dir) / "manifest.json").read_text())
    rm = json.loads((pathlib.Path(reference_dir) / "manifest.json").read_text())
    if list(em_["passages"]) != list(rm["passages"]):
        raise ValueError("the artifacts hold different passages")
    out = {}
    for pid, e in em_["passages"].items():
        r = rm["passages"][pid]
        if e["tokens_sha256"] != r["tokens_sha256"] or e["n_prompt"] != r["n_prompt"]:
            raise ValueError(f"{pid}: the reference is of another sequence")
        with np.load(pathlib.Path(engine_dir) / f"{pid}.npz") as z:
            tokens, P, logits = [int(t) for t in z["tokens"]], int(z["n_prompt"]), z["logits"]
        with np.load(pathlib.Path(reference_dir) / f"{pid}.npz") as z:
            p64 = z["p64"]
        if logits.shape != p64.shape:
            raise ValueError(f"{pid}: engine rows {logits.shape} against reference rows {p64.shape}")
        out[pid] = dict(tokens=tokens, n_prompt=P, logits=logits, p64=p64)
    return dict(passages=out, engine=em_, reference=rm)


def score(engine_dir, reference_dir, arm_logits, *, name="arm") -> dict:
    """§6 for a new arithmetic, from the artifacts alone: `arm_logits` maps
    each passage to its logits at the scored rows (SCORED), in real units.
    Nothing is generated or recomputed."""
    a = load_artifacts(engine_dir, reference_dir)
    per_arm, per_eng, per_p64, passages = [], [], [], {}
    for pid, x in a["passages"].items():
        arm = np.asarray(arm_logits[pid], dtype=np.float64)
        if arm.shape != x["p64"].shape:
            raise ValueError(f"{pid}: the arm gives {arm.shape} rows, the artifacts {x['p64'].shape}")
        tgt = x["tokens"][x["n_prompt"]:]
        pa = fm.position_metrics(x["p64"], arm, tgt)
        pe = fm.position_metrics(x["p64"], x["logits"], tgt)
        per_arm.append(pa)
        per_eng.append(pe)
        per_p64.append(fm.nll_bits(x["p64"], tgt))
        passages[pid] = dict(scored=len(tgt), arm=summary(pa), engine=summary(pe))
    return dict(arm=name, model=a["engine"]["model"], reference=a["reference"]["reference"],
                passages=passages, acceptance=fm.acceptance(per_arm, per_eng),
                nll_calibration=fm.nll_calibration(per_p64, [e["nll"] for e in per_eng]))


def sequences_from_engine_artifact(d) -> dict:
    """The frozen sequences and the baseline's logits, from an engine
    artifact checked against its SHA256SUMS."""
    _check_sums(d)
    man = json.loads((pathlib.Path(d) / "manifest.json").read_text())
    out = {}
    for pid in man["passages"]:
        with np.load(pathlib.Path(d) / f"{pid}.npz") as z:
            tokens = [int(t) for t in z["tokens"]]
            out[pid] = dict(tokens=tokens, n_prompt=int(z["n_prompt"]), logits=z["logits"],
                            sha256=lr.ids_sha256(tokens), seconds=None)
    return out


def load_generated(work) -> dict:
    """The sequences and the baseline's choosing logits a generate stage left
    in `work` (sequences.json and each passage's runner output), checked
    against the recorded hashes."""
    work = pathlib.Path(work)
    seqs = json.loads((work / "sequences.json").read_text())
    out = {}
    for pid, s in seqs.items():
        r = lr.load(work / "generate" / pid)
        if r["tokens"] != s["tokens"] or lr.ids_sha256(r["tokens"]) != s["sha256"]:
            raise ValueError(f"{pid}: the runner's output is not the frozen sequence")
        out[pid] = dict(tokens=s["tokens"], n_prompt=s["n_prompt"], logits=r["logits"],
                        sha256=s["sha256"], seconds=None)
    return out


def p64_check(spec, gguf, ids, *, device="cuda", layers=2):
    """The torch reference on `device` (DeviceWeights: bulk decoded there)
    against NumPy at `layers` layers on one prompt (§5.3): the largest
    relative difference per intermediate, routing equality, wall times and
    the device's peak memory."""
    import torch
    import k2_float_torch as kt
    cfg, k = spec["cfg"], spec["k"]
    W = GgufWeights(str(gguf), ints=False, cfg=cfg)
    t0 = time.time()
    ref = kf.forward(ids, W, cfg, layers=layers, k=k)
    t_np = time.time() - t0
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    got = kt.forward(ids, kt.DeviceWeights(W, device), cfg, layers=layers, k=k, device=device)
    t_t = time.time() - t0
    rel = {}
    for n in ref:
        a, b = np.asarray(ref[n]), np.asarray(got[n])
        if n.endswith(".top"):
            rel[n] = int((a != b).any(axis=1).sum())
        else:
            rel[n] = float(np.abs(a - b).max() / max(np.abs(a).max(), 1e-300))
    worst = max(v for n, v in rel.items() if not n.endswith(".top"))
    flips = sum(v for n, v in rel.items() if n.endswith(".top"))
    peak = torch.cuda.max_memory_allocated() if device.startswith("cuda") else None
    return dict(device=device, layers=layers, tokens=len(ids), max_relative=worst, routing_tokens_differing=flips,
                per_name=rel, seconds=dict(numpy=t_np, torch=t_t), peak_device_bytes=peak)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("stage", choices=("two-layer", "two-layer-merge", "engine-paths", "generate", "reference",
                                      "full-depth", "score", "p64-check"))
    ap.add_argument("records", nargs="*", help="two-layer-merge: the records to merge, in passage order")
    ap.add_argument("--model", choices=tuple(MODELS))
    ap.add_argument("--gguf")
    ap.add_argument("--passages")
    ap.add_argument("--runner")
    ap.add_argument("--render")
    ap.add_argument("--work", required=True)
    ap.add_argument("--configs", default=",".join(ps.CONFIGS),
                    help="two layers: the integer configurations, comma-separated")
    ap.add_argument("--arms", default=",".join(em.ARMS),
                    help="two layers: the emulations, comma-separated (empty: none)")
    ap.add_argument("--no-engine", action="store_true", help="two layers: no llama.cpp arm")
    ap.add_argument("--only", default="", help="two layers: these passages, comma-separated")
    ap.add_argument("--paths", default=",".join(ENGINE_PATHS),
                    help="engine-paths: the kernel paths, comma-separated (ENGINE_PATHS)")
    ap.add_argument("--out", default="two-layer.json", help="two layers: the record's name in --work")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--layers", type=int, default=None, help="full depth: fewer layers, for tests")
    ap.add_argument("--device", default=None,
                    help="the torch reference's device (p64-check: default cuda; full depth: default cpu)")
    ap.add_argument("--backend", choices=("numpy", "torch"), default="numpy",
                    help="full depth and engine-paths: the p64 reference (numpy runs on the cpu)")
    ap.add_argument("--primary", help="full depth: the configuration whose verdict is the record's, "
                    "named before the run (required)")
    ap.add_argument("--exploratory", default="",
                    help="full depth: further configurations, comma-separated, reported beside the primary")
    ap.add_argument("--reuse", action="store_true",
                    help="full depth: the sequences and baseline logits already in --work, not generated "
                    "again (finished parts in --work are always reused)")
    ap.add_argument("--engine", default="{}", help="JSON: further engine settings (n_gpu_layers, kv_type, "
                    "flash_attn, n_batch, n_ubatch, split_mode, main_gpu, tensor_split, override_tensor), "
                    "pinned by the kit")
    ap.add_argument("--implementation", choices=IMPLEMENTATIONS, default="numpy",
                    help="full depth: the integer model's implementation (run-only: the K2 driver on the GPU)")
    ap.add_argument("--model-sha256", help="generate, reference: a JSON file {shard name: sha256} the kit verified")
    ap.add_argument("--attach", default="", help="generate: further records for the engine artifact "
                    "(the GPU topology, the kit's pins), comma-separated")
    ap.add_argument("--engine-artifact", help="reference, score: the engine artifact's directory")
    ap.add_argument("--reference-artifact", help="score: the reference artifact's directory")
    ap.add_argument("--arm-logits", help="score: a directory of <passage>.npz, each with `logits` at the "
                    "scored rows, in real units")
    ap.add_argument("--name", default="arm", help="score: the arithmetic's name in the record")
    # intermixed: before 3.12's later patch releases (session 12's pod had 3.12.3),
    # parse_args leaves `records` empty when options separate it from the stage
    a = ap.parse_intermixed_args(argv)
    if a.stage == "full-depth" and not a.primary:
        ap.error("full-depth needs --primary: the configuration named before the run")
    work = pathlib.Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    if a.stage == "score":
        if not (a.engine_artifact and a.reference_artifact and a.arm_logits):
            ap.error("score needs --engine-artifact, --reference-artifact and --arm-logits")
        arm = {}
        for f in sorted(pathlib.Path(a.arm_logits).glob("*.npz")):
            with np.load(f) as z:
                arm[f.stem] = z["logits"]
        rec = score(a.engine_artifact, a.reference_artifact, arm, name=a.name)
        dst = work / f"score-{a.name}.json"
        dst.write_text(json.dumps(rec, indent=1, default=str))
        log(a.name, "acceptance:", rec["acceptance"]["overall"], "record:", dst)
        return 0
    if a.stage == "two-layer-merge":
        rec = merge_two_layer([json.loads(pathlib.Path(r).read_text()) for r in a.records])
        log_feasibility(rec["feasibility"])
        dst = work / a.out
        dst.write_text(json.dumps(rec, indent=1, default=str))
        log("record:", dst)
        return 0
    model_sha = json.loads(pathlib.Path(a.model_sha256).read_text()) if a.model_sha256 else None
    if a.stage == "reference":
        if a.model is None or a.gguf is None:
            ap.error("reference needs --model and --gguf")
        gen = (sequences_from_engine_artifact(a.engine_artifact) if a.engine_artifact else load_generated(work))
        write_reference_artifact(MODELS[a.model], a.gguf, gen, work, layers=a.layers, backend=a.backend,
                                 device=a.device, model_sha256=model_sha)
        return 0
    missing = [o for o in ("model", "gguf", "passages", "runner", "render") if getattr(a, o) is None]
    if missing:
        ap.error(f"{a.stage} needs " + ", ".join(f"--{o}" for o in missing))
    ids = prompts(a.model, a.passages, a.render, a.gguf, work)
    if a.only:
        ids = {pid: ids[pid] for pid in a.only.split(",")}
    if a.stage == "p64-check":
        first = next(iter(ids))
        rec = p64_check(MODELS[a.model], a.gguf, ids[first], device=a.device or "cuda", layers=a.layers or 2)
        rec["passage"] = first
        dst = work / "p64-check.json"
        dst.write_text(json.dumps(rec, indent=1))
        log(f"p64-check: max relative {rec['max_relative']:.2e}, routing tokens differing "
            f"{rec['routing_tokens_differing']}, numpy {rec['seconds']['numpy']:.0f} s, "
            f"torch {rec['seconds']['torch']:.0f} s, peak {rec['peak_device_bytes']}")
        return 0 if rec["max_relative"] < 1e-9 and rec["routing_tokens_differing"] == 0 else 1
    if a.stage == "engine-paths":
        rec = engine_paths(MODELS[a.model], a.gguf, ids, a.runner, work, paths=tuple(a.paths.split(",")),
                           threads=a.threads, engine=json.loads(a.engine), backend=a.backend, device=a.device)
        dst = work / "engine-paths.json"
    elif a.stage == "two-layer":
        rec = two_layer(MODELS[a.model], a.gguf, ids, a.runner, work, threads=a.threads,
                        engine=json.loads(a.engine), configs=tuple(c for c in a.configs.split(",") if c),
                        arms=tuple(c for c in a.arms.split(",") if c), with_engine=not a.no_engine)
        dst = work / a.out
    else:
        if a.reuse:
            gen = load_generated(work)
        else:
            gen = generate(MODELS[a.model], a.gguf, ids, a.runner, work, threads=a.threads,
                           engine=json.loads(a.engine))
            (work / "sequences.json").write_text(json.dumps(
                {pid: dict(tokens=g["tokens"], n_prompt=g["n_prompt"], sha256=g["sha256"])
                 for pid, g in gen.items()}, indent=1))
        if a.stage == "generate":
            out = {pid: pathlib.Path(work) / "generate" / pid for pid in gen}
            write_engine_artifact(MODELS[a.model], a.gguf, gen, work,
                                  manifests={pid: json.loads((o / "manifest.json").read_text()) for pid, o in out.items()},
                                  placements={pid: lr.placement((o / "run.log").read_text()) for pid, o in out.items()
                                              if (o / "run.log").exists()},
                                  model_sha256=model_sha, passages_sha256=_sha256_file(a.passages),
                                  attach=[f for f in a.attach.split(",") if f])
            return 0
        exploratory = tuple(c for c in a.exploratory.split(",") if c)
        rec = full_depth(MODELS[a.model], a.gguf, gen, work, primary=a.primary, exploratory=exploratory,
                         layers=a.layers, backend=a.backend, device=a.device, implementation=a.implementation)
        dst = work / (f"full-depth-primary-{a.primary}.json" if not exploratory
                      else f"full-depth-primary-{a.primary}-exploratory-{'-'.join(exploratory)}.json")
    dst.write_text(json.dumps(rec, indent=1, default=str))
    log("record:", dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
