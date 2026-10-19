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

Prompts: K2 uses the frozen token ids; another model renders the frozen
passages' user text with its own template (verinf_render). Records are JSON
in --work; nothing here decides anything the note has not frozen."""
import argparse
import json
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
import k2_precision_study as ps
import llama_runner as lr
from k2_attention import K2_MLA, MOONLIGHT_MLA
from k2_loader import GgufWeights, S_ACT, int_constants

MODELS = {"kimi-k2": dict(cfg=K2_MLA, adapter=fa.KIMI_K2, k=8),
          "moonlight": dict(cfg=MOONLIGHT_MLA, adapter=fa.MOONLIGHT, k=6)}
N_CTX = 1024                                   # pinned (§3.3)
CONTINUATION = 500


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

def two_layer(spec, gguf, ids_by_passage, runner, work, *, configs=tuple(ps.CONFIGS),
              arms=tuple(em.ARMS), threads=8, engine=None):
    """Every arm at two layers on each passage's prompt, every position
    scored. `spec` is a MODELS entry (cfg, adapter, k)."""
    cfg, k, adapter = spec["cfg"], spec["k"], spec["adapter"]
    work = pathlib.Path(work)
    work.mkdir(parents=True, exist_ok=True)
    trunc = work / "two-layer.gguf"
    if not trunc.exists():
        rec = gt.truncate(gguf, trunc, 2)
        log("truncated:", rec["added_keys"] or "no keys added")
    full_fp = lr.run(runner, gguf, work / "info-full", mode="info")["fingerprint"]
    two_fp = lr.run(runner, trunc, work / "info-two", mode="info")["fingerprint"]
    branch = lr.branch_differences(full_fp, two_fp)
    if branch:
        raise RuntimeError(f"the truncation leaves the full model's branch: {branch}")
    Wf = ps.CachedFloats(GgufWeights(str(gguf), ints=False, cfg=cfg))
    out = dict(model=adapter["name"], layers=2, passages={})
    for pid, ids in ids_by_passage.items():
        log(pid, len(ids), "tokens")
        T = len(ids)
        tgt = ids[1:]
        ref = kf.forward(ids, Wf, cfg, layers=2, k=k)
        p = {"p64": dict(n=T)}
        for name in configs:
            io, ranges = ps.forward(ids, Wf, cfg, ps.CONFIGS[name], layers=2, kk=k)
            A = 1 << ps.CONFIGS[name]["S_act"]
            lg = np.asarray(io["logits"], dtype=np.float64) / A
            io_f = {n: np.asarray(v, dtype=np.float64) / A for n, v in io.items()
                    if n.endswith(".out")}
            top = {f"L{il}.top": _gates_int(io, il, None)[0] for il in range(2) if f"L{il}.mask" in io}
            p[f"int/{name}"] = dict(**summary(fm.position_metrics(ref["logits"][:-1], lg[:-1], tgt)),
                                    explain=explain(ref, {**io_f, **top}, Wf, 2, k),
                                    violations=len(ranges.violations()))
        for arm in arms:
            P = em.Policy(*em.ARMS[arm])
            eo = em.forward(ids, Wf, cfg, P, layers=2, k=k)
            gates = {il: _gates_p64(eo, il, Wf, k, Wf.n_experts(il))[1]
                     for il in range(2) if f"L{il}.s" in eo}
            p[f"emu/{arm}"] = dict(**summary(fm.position_metrics(ref["logits"][:-1], eo["logits"][:-1], tgt)),
                                   explain=explain(ref, eo, Wf, 2, k, gates=gates))
        r = lr.run(runner, trunc, work / "engine" / pid, ids=ids, mode="replay", n_prompt=1,
                   prompt_chunk=512, cont_chunk=512, capture=fa.captures(adapter, 2),
                   settings=dict(n_ctx=N_CTX, threads=threads, **(engine or {}), **adapter["engine"]))
        eng = fa.ours(adapter, r["captures"], Wf.n_experts(1))
        p["engine"] = dict(**summary(fm.position_metrics(ref["logits"][:-1], r["logits"], tgt)),
                           explain=explain(ref, eng, Wf, 2, k,
                                           gates={il: eng[f"L{il}.gates"] for il in range(2)
                                                  if f"L{il}.gates" in eng}),
                           manifest=r["manifest"]["settings"])
        out["passages"][pid] = p
        log(pid, "engine mean KL", round(p["engine"]["mean_kl"], 6), "int baseline",
            round(p["int/baseline"]["mean_kl"], 6))
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


def full_depth(spec, gguf, generated, work, *, layers=None, integer=None):
    """p64 and the integer model on each frozen sequence; per-position KL and
    NLL over the continuation for the integer arm and the engine baseline;
    §6's acceptance and the §6.5 calibration. `integer(ids)` returns the
    integer arm's real-valued logits (default: the CPU integer model)."""
    cfg, k = spec["cfg"], spec["k"]
    W = GgufWeights(str(gguf), ints=False, cfg=cfg)
    n_layers = layers or int(gt.metadata(gguf)[f"{gt.metadata(gguf)['general.architecture']}.block_count"])
    if integer is None:
        def integer(ids):
            io, _ = ps.forward(ids, W, cfg, ps.CONFIGS["baseline"], layers=n_layers, kk=k)
            return np.asarray(io["logits"], dtype=np.float64) / S_ACT
    per = {"integer": [], "engine": [], "p64_nll": []}
    rec = dict(model=spec["adapter"]["name"], layers=n_layers, passages={})
    for pid, g in generated.items():
        ids, P = g["tokens"], g["n_prompt"]
        t0 = time.time()
        ref = kf.forward(ids, W, cfg, layers=n_layers, k=k)["logits"]
        t_ref = time.time() - t0
        t0 = time.time()
        lg_int = integer(ids)
        t_int = time.time() - t0
        rows = slice(P - 1, len(ids) - 1)                  # positions scoring the continuation
        tgt = ids[P:]
        pi = fm.position_metrics(ref[rows], lg_int[rows], tgt)
        pe = fm.position_metrics(ref[rows], g["logits"], tgt)
        per["integer"].append(pi)
        per["engine"].append(pe)
        per["p64_nll"].append(fm.nll_bits(ref[rows], tgt))
        rec["passages"][pid] = dict(scored=len(tgt), integer=summary(pi), engine=summary(pe),
                                    p64_mean_nll=float(per["p64_nll"][-1].mean()),
                                    seconds=dict(p64=t_ref, integer=t_int, generate=g["seconds"]))
        log(pid, "integer", rec["passages"][pid]["integer"], "engine", rec["passages"][pid]["engine"])
    rec["acceptance"] = fm.acceptance(per["integer"], per["engine"])
    rec["nll_calibration"] = fm.nll_calibration(per["p64_nll"], [e["nll"] for e in per["engine"]])
    return rec


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
    ap.add_argument("stage", choices=("two-layer", "full-depth", "p64-check"))
    ap.add_argument("--model", choices=tuple(MODELS), required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--passages", required=True)
    ap.add_argument("--runner", required=True)
    ap.add_argument("--render", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--layers", type=int, default=None, help="full depth: fewer layers, for tests")
    ap.add_argument("--device", default="cuda", help="p64-check: the torch reference's device")
    ap.add_argument("--engine", default="{}", help="JSON: further engine settings (n_gpu_layers, kv_type, "
                    "flash_attn, n_batch, n_ubatch, override_tensor), pinned by the kit")
    a = ap.parse_args(argv)
    work = pathlib.Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    ids = prompts(a.model, a.passages, a.render, a.gguf, work)
    if a.stage == "p64-check":
        first = next(iter(ids))
        rec = p64_check(MODELS[a.model], a.gguf, ids[first], device=a.device, layers=a.layers or 2)
        rec["passage"] = first
        dst = work / "p64-check.json"
        dst.write_text(json.dumps(rec, indent=1))
        log(f"p64-check: max relative {rec['max_relative']:.2e}, routing tokens differing "
            f"{rec['routing_tokens_differing']}, numpy {rec['seconds']['numpy']:.0f} s, "
            f"torch {rec['seconds']['torch']:.0f} s, peak {rec['peak_device_bytes']}")
        return 0 if rec["max_relative"] < 1e-9 and rec["routing_tokens_differing"] == 0 else 1
    if a.stage == "two-layer":
        rec = two_layer(MODELS[a.model], a.gguf, ids, a.runner, work, threads=a.threads,
                        engine=json.loads(a.engine))
        dst = work / "two-layer.json"
    else:
        gen = generate(MODELS[a.model], a.gguf, ids, a.runner, work, threads=a.threads,
                       engine=json.loads(a.engine))
        (work / "sequences.json").write_text(json.dumps(
            {pid: dict(tokens=g["tokens"], n_prompt=g["n_prompt"], sha256=g["sha256"])
             for pid, g in gen.items()}, indent=1))
        rec = full_depth(MODELS[a.model], a.gguf, gen, work, layers=a.layers)
        dst = work / "full-depth.json"
    dst.write_text(json.dumps(rec, indent=1, default=str))
    log("record:", dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
