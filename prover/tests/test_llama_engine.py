"""The engine arm against the float64 reference on the toy K2
(analysis/k2-fidelity-study-design.md §3.3, §5), CPU with a llama.cpp build:
VERINF_LLAMA_RUNNER names tools/llama_runner's binary. Not in the CPU gate,
which has no llama.cpp; without the variable the suite is skipped.

- the toy loads on K2's branch (query-LoRA, one dense lead layer, sigmoid
  gating, YaRN);
- with K2's YaRN betas and nothing quantized (an all-F32 toy, an f32 KV
  cache) llama.cpp IS the float64 reference to float32 rounding: layer
  outputs, routing, gates and logits;
- with llama.cpp's own betas it is not, which is why the adapter sets them;
- generation is reproduced bit for bit by a replay with one-token steps;
- a truncation loads on the full model's branch;
- the runner keeps its device settings, their memory and the whole log, with
  the scheduler's placement;
- the prefill localization maps every intermediate onto p64, and capturing
  changes no logits; an operation runs alone on a run's own inputs."""
import json
import os
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import _k2_toy as toy                       # noqa: E402
import fidelity_adapters as fa              # noqa: E402
import fidelity_metrics as fm               # noqa: E402
import gguf_truncate as gt                  # noqa: E402
import k2_float_reference as kf             # noqa: E402
import k2_loader as kl                      # noqa: E402
import llama_runner as lr                   # noqa: E402

RUNNER = os.environ.get("VERINF_LLAMA_RUNNER")
# the session kit runs this suite again with the model offloaded to its GPU (e.g. 99)
GPU_LAYERS = int(os.environ.get("VERINF_LLAMA_GPU_LAYERS", "0"))
pytestmark = pytest.mark.skipif(not RUNNER, reason="VERINF_LLAMA_RUNNER not set (no llama.cpp)")
IDS = [3, 17, 0, 41, 9, 9, 22, 30, 5, 12, 47, 1]
CTX = 131072                     # pinned; the long-position checks need it all
EXACT = dict(threads=2, kv_type="f32", flash_attn="off", n_ctx=CTX, n_gpu_layers=GPU_LAYERS)


@pytest.fixture(scope="module")
def f32_toy(tmp_path_factory):
    p = tmp_path_factory.mktemp("engine") / "toy-f32.gguf"
    toy.write(p, quantize_q8=False)
    return p


def _reference(path, offset=0):
    W = kl.GgufWeights(str(path), ints=False, cfg=toy.TOY_MLA)
    return kf.forward(IDS, W, toy.TOY_MLA, layers=2, offset=offset, k=toy.TOY["k"]), W


def _replay(path, out, *, settings, offset=0, chunk=512):
    r = lr.run(RUNNER, path, out, ids=IDS, mode="replay", n_prompt=1, prompt_chunk=chunk,
               cont_chunk=chunk, pos_offset=offset, capture=fa.captures(fa.KIMI_K2, 2),
               settings=settings)
    return r, fa.ours(fa.KIMI_K2, r["captures"], toy.TOY["E"])


def test_the_toy_loads_on_k2s_branch(f32_toy, tmp_path):
    fp = lr.run(RUNNER, f32_toy, tmp_path / "info", mode="info")["fingerprint"]
    assert fp["arch"] == "deepseek2" and fp["n_lora_q"] == str(toy.TOY_MLA["q_rank"])
    assert fp["n_layer_dense_lead"] == "1" and fp["expert_gating_func"] == "sigmoid"
    assert fp["rope scaling"] == "yarn" and fp["n_expert_used"] == str(toy.TOY["k"])


def test_with_k2s_betas_the_engine_is_the_float64_reference(f32_toy, tmp_path):
    ref, W = _reference(f32_toy)
    r, got = _replay(f32_toy, tmp_path / "k2", settings=dict(EXACT, **fa.KIMI_K2["engine"]))
    for il in (0, 1):
        assert fm.residual_error(ref[f"L{il}.out"], got[f"L{il}.out"]).max() < 1e-5, il
    assert np.array_equal(got["L1.top"], ref["L1.top"])
    s = ref["L1.s"]
    w = np.take_along_axis(s, ref["L1.top"], axis=1)
    g_ref = fm.gate_matrix(ref["L1.top"], w / w.sum(axis=1, keepdims=True), toy.TOY["E"])
    assert np.allclose(got["L1.gates"].sum(axis=1), 1.0)
    assert fm.gate_distance(g_ref, got["L1.gates"]).max() < 1e-5
    assert np.allclose(got["L1.gates_scaled"], got["L1.gates"] * fa.KIMI_K2["routed_scale"])
    kl_t = fm.position_metrics(ref["logits"][:-1], r["logits"], IDS[1:])["kl"]
    assert kl_t.max() < 1e-10
    m = r["manifest"]
    assert m["settings"]["yarn_beta_fast"] == 1.0 and m["input_ids_sha256"] == lr.ids_sha256(IDS)


def test_llama_cpps_own_betas_are_not_k2s(f32_toy, tmp_path):
    ref, _ = _reference(f32_toy)
    _, got = _replay(f32_toy, tmp_path / "default", settings=EXACT)
    assert fm.residual_error(ref["L0.out"], got["L0.out"]).max() > 5e-3


def test_generation_is_reproduced_by_one_token_steps(tmp_path):
    p = tmp_path / "toy.gguf"
    toy.write(p)                                               # the Q8_0 toy
    settings = dict(threads=2, n_ctx=64, n_gpu_layers=GPU_LAYERS, **fa.KIMI_K2["engine"])
    gen = lr.run(RUNNER, p, tmp_path / "gen", ids=IDS[:5], mode="generate", n_prompt=5, n_gen=6,
                 prompt_chunk=512, settings=settings)
    seq = gen["tokens"]
    assert seq[:5] == IDS[:5] and len(seq) == 5 + len(gen["manifest"]["generated"])
    rep = lr.run(RUNNER, p, tmp_path / "rep", ids=seq, mode="replay", n_prompt=5,
                 prompt_chunk=512, cont_chunk=1, settings=settings)
    assert np.array_equal(gen["logits"], rep["logits"])
    assert np.array_equal(np.argmax(gen["logits"], axis=1), seq[5:])


def test_capture_does_not_change_the_computation_on_this_backend(tmp_path):
    """Capturing asks for outputs at every position; on this backend that
    leaves tokens and logits bit for bit unchanged. The kit repeats this on the
    target backend before capture is treated as overhead only (§5.1)."""
    p = tmp_path / "toy.gguf"
    toy.write(p)
    settings = dict(threads=2, n_ctx=64, n_gpu_layers=GPU_LAYERS, **fa.KIMI_K2["engine"])
    kw = dict(ids=IDS[:5], mode="generate", n_prompt=5, n_gen=6, prompt_chunk=512, settings=settings)
    plain = lr.run(RUNNER, p, tmp_path / "plain", **kw)
    caught = lr.run(RUNNER, p, tmp_path / "caught", capture=fa.captures(fa.KIMI_K2, 2), **kw)
    assert plain["tokens"] == caught["tokens"]
    assert np.array_equal(plain["logits"], caught["logits"])
    assert caught["captures"]["l_out-1"].shape[0] == len(caught["tokens"]) - 1


def test_the_context_size_is_required(tmp_path):
    p = tmp_path / "toy.gguf"
    toy.write(p)
    with pytest.raises(RuntimeError, match="--n-ctx is required"):
        lr.run(RUNNER, p, tmp_path / "x", ids=IDS, mode="replay", n_prompt=1, settings=dict(threads=2))
    with pytest.raises(RuntimeError, match="does not fit"):
        lr.run(RUNNER, p, tmp_path / "y", ids=IDS, mode="replay", n_prompt=1,
               settings=dict(threads=2, n_ctx=8))


def test_the_device_settings_their_memory_and_the_whole_log_are_kept(tmp_path):
    p = tmp_path / "toy.gguf"
    toy.write(p)
    r = lr.run(RUNNER, p, tmp_path / "x", ids=IDS, mode="replay", n_prompt=1,
               settings=dict(threads=2, n_ctx=64, split_mode="none", main_gpu=0, tensor_split="1"),
               env={"GGML_SCHED_DEBUG": "2"})
    s = r["manifest"]["settings"]
    assert (s["split_mode"], s["main_gpu"], s["tensor_split"]) == ("none", 0, [1])
    mem = r["manifest"]["device_memory_after_context"]
    assert [d["device"] for d in mem] and all(d["total"] >= d["free"] >= 0 for d in mem)
    log = (tmp_path / "x" / "run.log").read_text()
    assert "print_info:" in log and "node #" in log         # the load and the scheduler's placement
    pl = lr.placement(log)
    assert pl["printed"] and pl["evaluations"] >= 1 and set(pl["nodes_per_backend"]) == {"CPU"}, pl
    with pytest.raises(RuntimeError, match="split mode"):
        lr.run(RUNNER, p, tmp_path / "y", ids=IDS, mode="replay", n_prompt=1,
               settings=dict(threads=2, n_ctx=64, split_mode="rows"))


def test_a_truncation_loads_on_the_full_models_branch(f32_toy, tmp_path):
    gt.truncate(f32_toy, tmp_path / "one.gguf", 1)
    full = lr.run(RUNNER, f32_toy, tmp_path / "full", mode="info")["fingerprint"]
    one = lr.run(RUNNER, tmp_path / "one.gguf", tmp_path / "one", mode="info")["fingerprint"]
    assert one["n_layer"] == "1" and lr.branch_differences(full, one) == {}


def test_without_query_lora_or_yarn_the_engine_is_the_float64_reference(tmp_path):
    """Moonlight's variant (direct query, plain RoPE, scale 2.446), all F32."""
    cfg = toy.TOY_MOON_MLA
    p = tmp_path / "moon-f32.gguf"
    toy.write(p, quantize_q8=False, mla=cfg)
    fp = lr.run(RUNNER, p, tmp_path / "info", mode="info")["fingerprint"]
    assert fp["n_lora_q"] == "0" and fp["rope scaling"] != "yarn"
    W = kl.GgufWeights(str(p), ints=False, cfg=cfg)
    ref = kf.forward(IDS, W, cfg, layers=2, k=toy.TOY["k"])
    r = lr.run(RUNNER, p, tmp_path / "rep", ids=IDS, mode="replay", n_prompt=1, prompt_chunk=512,
               cont_chunk=512, capture=fa.captures(fa.MOONLIGHT, 2),
               settings=dict(EXACT, **fa.MOONLIGHT["engine"]))
    got = fa.ours(fa.MOONLIGHT, r["captures"], toy.TOY["E"])
    for il in (0, 1):
        assert fm.residual_error(ref[f"L{il}.out"], got[f"L{il}.out"]).max() < 1e-5, il
    assert np.array_equal(got["L1.top"], ref["L1.top"])
    assert fm.position_metrics(ref["logits"][:-1], r["logits"], IDS[1:])["kl"].max() < 1e-10


def test_the_study_runs_end_to_end_on_the_toy(tmp_path):
    """fidelity_study's stages on the toy: two layers with every kind of arm,
    then a generation (the baseline) and the full-depth acceptance record."""
    import fidelity_study as fs
    p = tmp_path / "toy.gguf"
    toy.write(p)
    spec = dict(cfg=toy.TOY_MLA, adapter=fa.KIMI_K2, k=toy.TOY["k"])
    rec = fs.two_layer(spec, p, {"a": IDS}, RUNNER, tmp_path / "two", configs=("baseline", "a16"),
                       arms=("bf16/R-fp32", "fp8/R-bf16"), threads=2)
    arms = rec["passages"]["a"]
    assert {"p64", "int/baseline", "int/a16", "emu/bf16/R-fp32", "emu/fp8/R-bf16", "engine"} == set(arms)
    for name, r in arms.items():
        if name != "p64":
            assert np.isfinite(r["mean_kl"]) and "L1.routing" in r["explain"], name
    gen = fs.generate(spec, p, {"a": IDS[:5], "b": IDS[3:9]}, RUNNER, tmp_path / "gen", threads=2, n_gen=6,
                      engine=dict(flash_attn="off"))
    full = fs.full_depth(spec, p, gen, tmp_path / "full", primary="a16", exploratory=("baseline",))
    assert full["layers"] == 2 and set(full["passages"]) == {"a", "b"}
    assert full["primary"] == "a16" and set(full["exploratory"]) == {"baseline"}
    assert full["acceptance"]["overall"] in ("accept", "fail", "inconclusive")
    assert set(full["acceptance"]["rows"]) == {"mean_kl", "p99_kl", "mean_nll", "max_kl"}
    assert "difference" in full["nll_calibration"]
    # the artifacts: the generation's and p64's, enough to score an arithmetic again without either
    runs = {pid: tmp_path / "gen" / "generate" / pid for pid in gen}
    eng = fs.write_engine_artifact(spec, p, gen, tmp_path / "gen",
                                   manifests={pid: lr.load(o)["manifest"] for pid, o in runs.items()},
                                   placements={pid: lr.placement((o / "run.log").read_text()) for pid, o in runs.items()})
    ref = fs.write_reference_artifact(spec, p, gen, tmp_path / "full")
    m = json.loads((eng / "manifest.json").read_text())
    for pid, g in gen.items():
        e = m["passages"][pid]
        assert e["runner"]["settings"]["flash_attn"] == "off" and e["n_scored"] == len(g["tokens"]) - g["n_prompt"]
        assert e["ended_by"] in ("eos", "cap") and e["placement"]["printed"] is False   # GGML_SCHED_DEBUG unset
    W = kl.GgufWeights(str(p), ints=False, cfg=toy.TOY_MLA)
    arm = {}
    for pid, g in gen.items():
        io, _ = fs.ps.forward(g["tokens"], W, toy.TOY_MLA, fs.ps.CONFIGS["a16"], layers=2, kk=toy.TOY["k"])
        arm[pid] = np.asarray(io["logits"], dtype=np.float64)[g["n_prompt"] - 1:-1] / (1 << fs.ps.CONFIGS["a16"]["S_act"])
    assert fs.score(eng, ref, arm, name="a16")["acceptance"] == full["acceptance"]


def test_the_expected_fingerprint_is_what_llama_cpp_prints(f32_toy, tmp_path):
    """Metadata's expectations against llama.cpp's own print_info, both toys."""
    for name, cfg in (("k2", toy.TOY_MLA), ("moon", toy.TOY_MOON_MLA)):
        p = tmp_path / f"{name}.gguf"
        toy.write(p, mla=cfg)
        fp = lr.run(RUNNER, p, tmp_path / name, mode="info")["fingerprint"]
        exp = gt.deepseek2_expected_fingerprint(gt.metadata(p))
        assert {k: fp.get(k) for k in exp if k != "rope scaling"} == \
            {k: v for k, v in exp.items() if k != "rope scaling"}, name
        if "rope scaling" in exp:
            assert fp["rope scaling"] == exp["rope scaling"]


def test_the_engine_paths_are_scored_on_the_same_passages(tmp_path):
    """fidelity_study.engine_paths: the two-layer engine arm as one batch and
    in one-token steps, each against p64, the environment it ran under
    recorded (cuBLAS's TF32 off for its path)."""
    import fidelity_study as fs
    p = tmp_path / "toy.gguf"
    toy.write(p)
    spec = dict(cfg=toy.TOY_MLA, adapter=fa.KIMI_K2, k=toy.TOY["k"])
    paths = ("gpu-batch", "gpu-steps", "gpu-batch-no-tf32")
    rec = fs.engine_paths(spec, p, {"a": IDS, "b": IDS[2:]}, RUNNER, tmp_path / "paths", paths=paths,
                          threads=2, engine=dict(n_gpu_layers=GPU_LAYERS))
    for pid in ("a", "b"):
        row = rec["passages"][pid]
        for name in paths:
            assert np.isfinite(row[name]["mean_kl"]) and row[name]["settings"]["cont_chunk"] in (1, 512)
        assert row["gpu-batch-no-tf32"]["environment"]["NVIDIA_TF32_OVERRIDE"] == "0"
        assert "NVIDIA_TF32_OVERRIDE" not in row["gpu-batch"]["environment"]
    assert set(rec["pooled"]) == set(paths)


def test_the_localization_maps_every_intermediate_and_captures_change_nothing(tmp_path):
    """prefill_localize on the all-F32 toy with an f32 KV cache, on the CPU: both
    paths' every mapped intermediate within float32 of p64 (the mapping is right),
    the paths within float32 of each other, and capturing leaves the logits alone."""
    import prefill_localize as pl
    p = tmp_path / "toy.gguf"
    toy.write(p, quantize_q8=False)
    spec = dict(cfg=toy.TOY_MLA, adapter=fa.KIMI_K2, k=toy.TOY["k"])
    ids = [3, 17, 0, 41, 9, 9, 2, 5, 11, 30, 7, 1]
    out = pl.localize(RUNNER, p, ids, tmp_path / "loc", spec=spec,
                      settings=dict(n_ctx=64, threads=2, flash_attn="off", kv_type="f32", **fa.KIMI_K2["engine"]))
    assert not any(c["changed_logits"] for c in out["captures"].values())
    mapped = [r for r in out["rows"] if "p64" in r]
    assert {r["name"].split("-")[0] for r in mapped} >= {"attn_norm", "kv_cmpr", "kq_soft_max", "kqv_out", "ffn_inp",
                                                        "ffn_norm", "ffn_moe_out", "ffn_shexp", "l_out"}
    assert all(v["rel_l2"] < 2e-6 for r in mapped for v in r["p64"].values()), mapped
    compared = [r for r in out["rows"] if "batch_vs_steps" in r]
    assert len(compared) > 40 and all(r["batch_vs_steps"]["rel_l2"] < 2e-6 for r in compared)
    assert out["first_apart"] is None


def test_an_operation_runs_alone_on_the_runs_own_inputs(tmp_path):
    """engine_optest on the toy's captured batched run, on the CPU backend: each
    operation's one-graph and per-token runs agree, and against float64 on the same
    inputs the F32 weights are within float32 and the f16 KV products within f16."""
    import engine_optest as eo
    import prefill_localize as pl
    optest = pathlib.Path(RUNNER).with_name("verinf_optest")
    if not optest.exists():
        pytest.skip("verinf_optest not built beside the runner")
    p = tmp_path / "toy.gguf"
    toy.write(p, quantize_q8=False)
    ids = [3, 17, 0, 41, 9, 9, 2, 5, 11, 30, 7, 1]
    pl.localize(RUNNER, p, ids, tmp_path / "loc", settings=dict(n_ctx=64, threads=2, flash_attn="off",
                                                                 **fa.KIMI_K2["engine"]))
    res = lr.load(tmp_path / "loc" / "batch-captured")
    bounds = {"kq-1": 1e-3, "kqv-1": 1e-3, "weight:blk.1.attn_q_a.weight:attn_norm-1/MUL": 1e-6,
              "experts:blk.1.ffn_up_exps.weight:ffn_norm-1/MUL:ffn_moe_argsort-1/ARGSORT": 1e-6}
    for op, bound in bounds.items():
        spec = eo.build(op, res, p)
        rec = eo.compare(optest, tmp_path / "op" / op.replace(":", "_").replace("/", "@"), ref=spec["ref"],
                         inputs=spec["inputs"], devices=("CPU",), mask=spec.get("mask"))
        assert all(v["rel_l2"] < bound for v in rec["against_float64"].values()), (op, rec)
        assert rec["between"]["CPU: all vs single"]["rel_l2"] < 1e-6, (op, rec)
