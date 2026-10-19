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
- a truncation loads on the full model's branch."""
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
pytestmark = pytest.mark.skipif(not RUNNER, reason="VERINF_LLAMA_RUNNER not set (no llama.cpp)")
IDS = [3, 17, 0, 41, 9, 9, 22, 30, 5, 12, 47, 1]
CTX = 131072                     # pinned; the long-position checks need it all
EXACT = dict(threads=2, kv_type="f32", flash_attn="off", n_ctx=CTX)


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
    settings = dict(threads=2, n_ctx=64, **fa.KIMI_K2["engine"])
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
    settings = dict(threads=2, n_ctx=64, **fa.KIMI_K2["engine"])
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


def test_a_truncation_loads_on_the_full_models_branch(f32_toy, tmp_path):
    gt.truncate(f32_toy, tmp_path / "one.gguf", 1)
    full = lr.run(RUNNER, f32_toy, tmp_path / "full", mode="info")["fingerprint"]
    one = lr.run(RUNNER, tmp_path / "one.gguf", tmp_path / "one", mode="info")["fingerprint"]
    assert one["n_layer"] == "1" and lr.branch_differences(full, one) == {}
