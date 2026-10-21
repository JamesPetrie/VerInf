"""The run-only pass (prover/run_only.py, demo_k2.run_only): the integer
model's logits from the driver's own tape, run once without a proof, every
lookup's keys counted and field safety bounded claim by claim.

Without a card each case runs in its own fresh process, with the field
kernels in Python (_cpu_field) and torch's device placements on the CPU
(_cpu_tape). With VERINF_RUN_ONLY_GPU=1 the same cases run on the card's own
kernels instead (the GPU session's check of them).
- On the toy at two and four layers, at the baseline, a14w14 and a16w16: the
  logits equal the NumPy model's (k2_precision_study.forward) and the existing
  driver's engine pass (check mode's), exactly; every traced name's extremes
  are the NumPy model's; every key is inside its table and every bound holds.
- The rescale-witness tamper (an output one higher, its low word 2^r lower) is
  a key outside the rescale table, counted at its claim, with the field still
  established.
- A product past the field is not established, nor anything computed from it;
  the tight bound establishes what K·max|a|·max|b| cannot.
- No witness survives the pass.
- fidelity_study's run-only implementation gives the NumPy model's metrics
  and acceptance, with its feasibility from the pass, in parts of its own."""
import json
import os
import pathlib
import subprocess
import sys

import pytest

_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

import _k2_toy as toy                  # noqa: E402

GPU = os.environ.get("VERINF_RUN_ONLY_GPU") == "1"
IDS, N_PROMPT = [3, 17, 0, 41, 9, 9], 3
CASES = {"baseline-2": ("baseline", 2), "a14w14-4": ("a14w14", 4), "a16w16-4": ("a16w16", 4)}

_PRELUDE = """
import contextlib, json, sys
import numpy as np
import _cpu_tape as ct, _cpu_field as cf, _k2_toy as toy
GPU = sys.argv[1] == "1"
shims = contextlib.ExitStack()
if not GPU:
    shims.enter_context(ct.cuda_on_cpu())
    shims.enter_context(cf.cpu_field())
"""

_K2 = _PRELUDE + """
import gc
path, precision, layers, tamper = sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5] == "1"
ids, n_prompt = %r, %r
with shims:
    import demo_k2 as dk, k2_precision_study as ps, k2_loader as kl
    kw = dict(cfg=toy.TOY_MLA, layers=layers, kk=toy.TOY["k"], precision=precision)
    logits, rec = dk.run_only(path, ids, n_prompt, **kw)
    S = 1 << ps.CONFIGS[precision]["S_act"]
    io, _ = ps.forward(ids, kl.GgufWeights(path, ints=False, cfg=toy.TOY_MLA), toy.TOY_MLA,
                       ps.CONFIGS[precision], layers=layers, kk=toy.TOY["k"], head_chunk=20)
    b = dk.build(path, ids[:n_prompt], ids[n_prompt:], **kw)
    got, _sz = dk.engine_values(b)
    del b
    gc.collect()
    run = rec["run"]
    trace = {n: v for n, v in run["trace"].items() if n in io}
    out = dict(numpy_equal=bool(np.array_equal(logits * S, np.asarray(io["logits"], dtype=np.float64))),
               driver_equal=bool(np.array_equal(logits * S, got["logits"].reshape(logits.shape).astype(np.float64))),
               ok=rec["ok"], lookups_ok=run["lookups"]["ok"], field_ok=run["field"]["ok"],
               unknown=run["field"]["unknown_types"], traced=len(trace),
               trace_differs=sorted(n for n, v in trace.items()
                                    if v is None or list(v) != [int(np.min(io[n])), int(np.max(io[n]))]),
               tables=rec["tables"]["entries"], claims=rec["claims"])
    if tamper:
        import compute_fns, rescale_claim
        honest = dict.__getitem__(compute_fns.COMPUTE_FNS, rescale_claim.RescaleClaim)
        hit = []

        def shifted(c, live):
            outs = honest(c, live)
            if not hit:
                hit.append(dict(x=c.x.name, rescale_bits=c.rescale_bits))
                outs[c.x] = dk._field_add_at(outs[c.x], 0, 1)
                outs[c.x_shifted] = dk._field_add_at(outs[c.x_shifted], 0, 1)
                outs[c.x_low] = dk._field_add_at(outs[c.x_low], 0, -(1 << c.rescale_bits))
            return outs
        compute_fns.COMPUTE_FNS[rescale_claim.RescaleClaim] = shifted
        _, trec = dk.run_only(path, ids, n_prompt, **kw)
        compute_fns.COMPUTE_FNS[rescale_claim.RescaleClaim] = honest
        out["tamper"] = dict(hit=hit, ok=trec["ok"], lookups=trec["run"]["lookups"]["outside"],
                             field_ok=trec["run"]["field"]["ok"])
print(json.dumps(out))
""" % (IDS, N_PROMPT)

_SYNTHETIC = _PRELUDE + """
with shims:
    import torch
    import core, run_only as ro
    from tape import Tape
    field = lambda vals: torch.tensor(vals, dtype=torch.int64, device="cuda").view(torch.uint64)
    cfg = core.LigeroConfig(ELL=8, K_DEG=8, N_LIG=32, T_QUERIES=4)
    out = {}
    # past the field: 4 · 2^40 · 2^30 = 2^72; and an add of it, untrusted
    t = Tape(cfg, lazy=True)
    a = t.commit("a", field([1 << 40] * 4), (1, 4))
    w = t.commit("w", field([1 << 30] * 4), (4, 1))
    c = t.matmul(a, w)
    d = c + c
    _, rec = ro.run(t, [d.var])
    out["past"] = rec["field"]
    out["past_kept"] = dict(product=c.var in t.inputs, kept=d.var in t.inputs)
    # the tight bound: K·max|a|·max|w| = 64 · 2^40 · 2^22 = 2^68, Σ|a||w| = 2^60 + 63 · 2^22
    t = Tape(cfg, lazy=True)
    a = t.commit("a", field([1 << 40] + [1] * 63), (1, 64))
    w = t.commit("w", field([1 << 20] + [1 << 22] * 63), (64, 1))
    c = t.matmul(a, w)
    e = c + c
    vals, rec = ro.run(t, [e.var])
    out["tight"] = rec["field"]
    out["tight_value"] = int(vals[e.var].reshape(-1)[0])
    out["tight_kept"] = dict(product=c.var in t.inputs, kept=e.var in t.inputs)
print(json.dumps(out, default=str))
"""


_STUDY = _PRELUDE + """
import pathlib
path, work = sys.argv[2], pathlib.Path(sys.argv[3])
with shims:
    import fidelity_adapters as fa, fidelity_study as fs
    spec = dict(cfg=toy.TOY_MLA, adapter=fa.KIMI_K2, k=toy.TOY["k"])
    rng = np.random.default_rng(0)
    gen = {}
    for pid, (n, P) in {"a": (9, 4), "b": (8, 3)}.items():
        tokens = [int(t) for t in rng.integers(0, toy.TOY["V"], n)]
        gen[pid] = dict(tokens=tokens, n_prompt=P, seconds=None,
                        logits=rng.normal(0, 2, (n - P, toy.TOY["V"])).astype(np.float32))
    a = fs.full_depth(spec, path, gen, work / "np", primary="baseline")
    b = fs.full_depth(spec, path, gen, work / "ro", primary="baseline", implementation="run-only")
    again = fs.full_depth(spec, path, gen, work / "ro", primary="baseline", implementation="run-only")
    out = dict(same_acceptance=a["acceptance"] == b["acceptance"],
               same_passages=all(a["passages"][p]["integer"] == b["passages"][p]["integer"] for p in gen),
               feasibility=b["feasibility"]["baseline"], implementation=b["implementation"],
               resumed=again["passages"]["a"]["resumed"],
               parts=sorted(f.name for f in (work / "ro" / fs.PARTS / "numpy-cpu" / "a").iterdir()))
print(json.dumps(out, default=str))
"""


def _env():
    return dict(os.environ, PYTHONPATH=os.pathsep.join(str(_ROOT / p) for p in ("prover", "prover/tests", "demo")),
                LIGERO_ELL="16")


def _run(script, *args):
    return subprocess.Popen([sys.executable, "-c", script, "1" if GPU else "0", *map(str, args)], env=_env(),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def _result(p, name):
    so, se = p.communicate()
    assert p.returncode == 0, (name, se[-3000:])
    return json.loads(so.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    """Every case in its own fresh process, two at a time (a16w16's tables are
    2^28 entries)."""
    if GPU:
        import torch
        if not torch.cuda.is_available():
            pytest.skip("VERINF_RUN_ONLY_GPU=1 without a card")
    d = tmp_path_factory.mktemp("runonly")
    paths = {n: d / f"toy{n}.gguf" for n in (2, 4)}
    for n, p in paths.items():
        toy.write(p, layers=n)
    jobs = [(name, (_K2, paths[layers], precision, layers, int(precision == "a16w16")))
            for name, (precision, layers) in CASES.items()] + [("synthetic", (_SYNTHETIC,)),
                                                                ("study", (_STUDY, paths[2], d / "study"))]
    out = {}
    for i in range(0, len(jobs), 2):
        procs = {name: _run(*args) for name, args in jobs[i:i + 2]}
        out.update({name: _result(p, name) for name, p in procs.items()})
    return out


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_logits_are_the_numpy_models_and_the_existing_drivers(results, case):
    r = results[case]
    assert r["numpy_equal"] and r["driver_equal"], r


@pytest.mark.parametrize("case", sorted(CASES))
def test_every_key_is_inside_its_table_and_every_bound_holds(results, case):
    r = results[case]
    assert r["lookups_ok"] and r["field_ok"] and r["ok"], r
    assert r["unknown"] == [], r["unknown"]


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_traced_extremes_are_the_numpy_models(results, case):
    r = results[case]
    assert r["traced"] >= 20 and r["trace_differs"] == [], r


def test_a_key_outside_its_table_is_counted_at_its_claim(results):
    t = results["a16w16-4"]["tamper"]
    assert len(t["hit"]) == 1 and not t["ok"], t
    assert t["field_ok"], t                                 # the values stay field-safe: only a range is broken
    assert [(o["type"], o["table"], o["outside"]) for o in t["lookups"]] == \
        [("RescaleClaim", f"rescale_w{t['hit'][0]['rescale_bits']}", 1)], t


def test_a_product_past_the_field_is_not_established_nor_what_it_feeds(results):
    f = results["synthetic"]["past"]
    assert not f["ok"]
    rows = {r["type"]: r for r in f["unestablished"]}
    assert rows["MatmulClaim"]["cheap_bits"] == 72.0 and rows["MatmulClaim"]["bound_bits"] == 72.0
    assert rows["AddClaim"]["bound_bits"] is None            # its input was never established


def test_the_tight_bound_establishes_what_the_cheap_one_cannot(results):
    s = results["synthetic"]
    assert s["tight"]["ok"], s["tight"]
    row = next(r for r in s["tight"]["rows"] if r["type"] == "MatmulClaim")
    assert row["cheap_bits"] == 68.0 and row["bound_bits"] < 61
    assert s["tight_value"] == 2 * ((1 << 60) + 63 * (1 << 22))


def test_no_witness_survives_the_pass(results):
    s = results["synthetic"]
    assert s["past_kept"] == dict(product=False, kept=True)
    assert s["tight_kept"] == dict(product=False, kept=True)


def test_the_studys_run_only_implementation_gives_the_numpy_models_metrics(results):
    r = results["study"]
    assert r["same_acceptance"] and r["same_passages"] and r["implementation"] == "run-only", r
    f = r["feasibility"]
    assert f["implementation"] == "run-only" and f["ok"] and not f["lookups_outside"] and not f["field_unestablished"]
    assert r["resumed"] == ["p64", "baseline"] and r["parts"] == ["baseline.run-only.npz", "p64.npz"]
