"""Is the GPU engine's departure from float32 on the all-F32 toys internal
precision or a semantic mismatch? (Session 11, engine-cuda 8/10: layer
outputs 1.7e-7 off p64 at position 0 and 2-4e-3 after it, in K2's form and
Moonlight's, with and without YaRN; the CPU path within float32 rounding.)

Position 0 is decoded alone (n_prompt = 1) and positions 1-11 as one batch,
so the split is single-token against multi-token kernels. At the pinned
commit (f174553) multi-token float32 matmuls run in TF32 on Ampere and
later: up to 16 tokens in mmf, whose mma passes float32 bits as TF32
operands (10 mantissa bits; no switch turns it off), and beyond that in
cuBLAS, which llama.cpp sets to CUBLAS_TF32_TENSOR_OP_MATH
(NVIDIA_TF32_OVERRIDE=0 turns that off). Single tokens take the float32
vector kernels.

Each toy (K2's form and Moonlight's, all F32, an f32 KV cache, flash
attention off) is replayed under CONDITIONS: on the CPU; on the GPU with
the 11-token batch (session 11's case), in one-token steps, with cuBLAS's
TF32 off, with ggml's fusion off; and the same with a 39-token batch. Each
run reports its layer outputs' residual error against p64 at every
position and the logits' KL; a second run captures layer 0's intermediates,
compared with the CPU's at each position, so the first op to leave float32
is named.

The verdict follows a rule fixed before any run (verdict()):
- same function: the CPU runs and the GPU's one-token steps are within the
  float32 bound (test_llama_engine's: residual < 1e-5, KL < 1e-10) at every
  position. If not, the GPU computes something else: a semantic mismatch.
- internal precision: the same function, and every batched GPU run outside
  the bound departs first at a matmul's output (attention's or the FFN's,
  not a norm) by at most 2^-6 relative.
- otherwise unresolved, for review.

    PYTHONPATH=prover:prover/tests python tools/engine_precision.py \\
        --runner R --out DIR [--gpu-layers 99]"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np

import _k2_toy as toy
import fidelity_adapters as fa
import fidelity_metrics as fm
import k2_float_reference as kf
import k2_loader as kl
import llama_runner as lr

SHORT = [3, 17, 0, 41, 9, 9, 22, 30, 5, 12, 47, 1]                  # test_llama_engine.IDS
LONG = SHORT + [(7 * i + 3) % toy.TOY["V"] for i in range(28)]       # a 39-token batch: over mmf's 16
EXACT = dict(threads=2, kv_type="f32", flash_attn="off", n_ctx=256)
BOUND = dict(residual=1e-5, kl=1e-10)
MATMUL_SHARE = 2.0 ** -6
# layer 0 in graph order; a matmul produces kqv_out (attention's products),
# ffn_inp (its projections, plus the residual) and ffn_out
LAYER0 = ("attn_norm-0", "kqv_out-0", "ffn_inp-0", "ffn_norm-0", "ffn_out-0", "l_out-0")
MATMUL_OUTPUTS = ("kqv_out-0", "ffn_inp-0", "ffn_out-0")
FORMS = {"k2": (toy.TOY_MLA, fa.KIMI_K2), "moonlight": (toy.TOY_MOON_MLA, fa.MOONLIGHT)}
SAME_FUNCTION = ("cpu-12", "cpu-40", "gpu-12-steps", "gpu-40-steps")
TF32_OFF = {"NVIDIA_TF32_OVERRIDE": "0"}


def conditions(gpu_layers):
    """name: (n_gpu_layers, ids, continuation chunk, environment)."""
    out = {}
    for n, ids in (("12", SHORT), ("40", LONG)):
        out[f"cpu-{n}"] = (0, ids, 512, {})
        out[f"gpu-{n}"] = (gpu_layers, ids, 512, {})
        out[f"gpu-{n}-steps"] = (gpu_layers, ids, 1, {})
        out[f"gpu-{n}-no-tf32"] = (gpu_layers, ids, 512, TF32_OFF)
        out[f"gpu-{n}-no-fusion"] = (gpu_layers, ids, 512, {"GGML_CUDA_DISABLE_FUSION": "1"})
    return out


def _rel_rows(a, b):
    """Per position: max |a − b| over max |b|."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return (np.abs(a - b).max(axis=1) / np.maximum(np.abs(b).max(axis=1), 1e-300)).tolist()


def run_form(form, runner, out, gpu_layers):
    cfg, adapter = FORMS[form]
    path = out / f"{form}-f32.gguf"
    if not path.exists():
        toy.write(path, quantize_q8=False, mla=cfg)
    W = kl.GgufWeights(str(path), ints=False, cfg=cfg)
    refs = {len(ids): kf.forward(ids, W, cfg, layers=2, k=toy.TOY["k"]) for ids in (SHORT, LONG)}
    rec, locs = {}, {}
    for name, (ngl, ids, chunk, env) in conditions(gpu_layers).items():
        ref = refs[len(ids)]
        settings = dict(EXACT, n_gpu_layers=ngl, **adapter["engine"])
        kw = dict(ids=ids, mode="replay", n_prompt=1, prompt_chunk=512, cont_chunk=chunk,
                  settings=settings, env=env)
        row = dict(n_gpu_layers=ngl, tokens=len(ids), cont_chunk=chunk, env=env)
        t0 = time.time()
        try:
            r = lr.run(runner, path, out / form / name, capture=fa.captures(adapter, 2), **kw)
        except RuntimeError as e:
            rec[name] = dict(row, error=str(e)[-1500:])
            continue
        got = fa.ours(adapter, r["captures"], toy.TOY["E"])
        row.update(residual={f"L{il}": fm.residual_error(ref[f"L{il}.out"], got[f"L{il}.out"]).tolist()
                             for il in (0, 1)},
                   kl=fm.position_metrics(ref["logits"][:-1], r["logits"], ids[1:])["kl"].tolist(),
                   routing_equal=bool(np.array_equal(got["L1.top"], ref["L1.top"])),
                   environment=r["manifest"]["environment"], seconds=time.time() - t0)
        row["within_bound"] = (max(max(v) for v in row["residual"].values()) < BOUND["residual"]
                               and max(row["kl"]) < BOUND["kl"])
        try:
            loc = lr.run(runner, path, out / form / f"{name}-layer0", capture=LAYER0, **kw)
            locs[name] = loc["captures"]
            row["layer0_missing"] = [n for n in LAYER0 if n not in loc["captures"]]
        except RuntimeError as e:
            row["layer0_error"] = str(e)[-800:]
        rec[name] = row
    for name, row in rec.items():                    # layer 0 against the CPU's, position by position
        cpu = locs.get("cpu-40" if row["tokens"] == len(LONG) else "cpu-12")
        if name in locs and cpu is not None:
            row["layer0_vs_cpu"] = {n: _rel_rows(locs[name][n], cpu[n]) for n in LAYER0
                                    if n in locs[name] and n in cpu}
    return rec


def verdict(rec):
    """The rule in the module docstring, over one form's conditions."""
    same = all(rec.get(c, {}).get("within_bound") for c in SAME_FUNCTION)
    if not same:
        return dict(verdict="semantic mismatch", same_function=False,
                    outside=[c for c in SAME_FUNCTION if not rec.get(c, {}).get("within_bound")])
    departures, ok = {}, True
    for name, row in rec.items():
        if not name.startswith("gpu") or name in SAME_FUNCTION or row.get("within_bound"):
            continue
        lv = row.get("layer0_vs_cpu") or {}
        first = next((n for n in LAYER0 if n in lv and max(lv[n]) > BOUND["residual"]), None)
        size = max(lv[first]) if first else None
        departures[name] = dict(first=first, relative=size)
        ok = ok and first in MATMUL_OUTPUTS and size <= MATMUL_SHARE
    return dict(verdict="internal precision" if ok else "unresolved", same_function=True,
                departures=departures)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runner", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu-layers", type=int, default=99)
    ap.add_argument("--forms", default=",".join(FORMS))
    a = ap.parse_args(argv)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rec = dict(bound=BOUND, matmul_share=MATMUL_SHARE, forms={})
    for form in a.forms.split(","):
        conds = run_form(form, a.runner, out, a.gpu_layers)
        rec["forms"][form] = dict(conditions=conds, **verdict(conds))
        print(f"{form}: {rec['forms'][form]['verdict']}", flush=True)
        for name, row in conds.items():
            if "error" in row:
                print(f"  {name:22s} error: {row['error'][-200:]}")
                continue
            print(f"  {name:22s} L0 {max(row['residual']['L0']):.2e} L1 {max(row['residual']['L1']):.2e} "
                  f"KL {max(row['kl']):.1e} {'within' if row['within_bound'] else 'OUTSIDE'} "
                  f"position 0: {row['residual']['L0'][0]:.1e}")
    (out / "engine-precision.json").write_text(json.dumps(rec, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
