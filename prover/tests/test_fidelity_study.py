"""fidelity_study's full-depth parts, CPU, on the toy K2 with a stand-in
baseline (no llama.cpp): a run started again resumes after the last part an
earlier one finished, reproducing its record; a new configuration reuses the
p64 rows; a changed sequence is computed again. The stages with the engine
are test_llama_engine.py's."""
import json
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import _k2_toy as toy                       # noqa: E402
import fidelity_adapters as fa              # noqa: E402
import fidelity_study as fs                 # noqa: E402

SPEC = dict(cfg=toy.TOY_MLA, adapter=fa.KIMI_K2, k=toy.TOY["k"])


def _generated(seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for pid, (n, P) in {"a": (9, 4), "b": (8, 3)}.items():
        tokens = [int(t) for t in rng.integers(0, toy.TOY["V"], n)]
        out[pid] = dict(tokens=tokens, n_prompt=P, logits=rng.normal(0, 2, (n - P, toy.TOY["V"])),
                        seconds=None)
    return out


@pytest.fixture
def counted(monkeypatch):
    """Every p64 and integer forward, by S_act and passage length."""
    calls = []
    kf_forward, ps_forward = fs.kf.forward, fs.ps.forward

    def kf_counted(ids, *a, **kw):
        calls.append(("p64", len(ids)))
        return kf_forward(ids, *a, **kw)

    def ps_counted(ids, W, cfg, config, **kw):
        calls.append((f"S{config['S_act']}", len(ids)))
        return ps_forward(ids, W, cfg, config, **kw)
    monkeypatch.setattr(fs.kf, "forward", kf_counted)
    monkeypatch.setattr(fs.ps, "forward", ps_counted)
    return calls


@pytest.fixture(scope="module")
def gguf(tmp_path_factory):
    p = tmp_path_factory.mktemp("study") / "toy.gguf"
    toy.write(p)
    return p


def _without_resumed(rec):
    return {pid: {k: v for k, v in r.items() if k != "resumed"} for pid, r in rec["passages"].items()}


def test_a_run_started_again_resumes_and_reproduces_its_record(gguf, tmp_path, counted):
    gen = _generated()
    first = fs.full_depth(SPEC, gguf, gen, tmp_path, primary="baseline")
    assert sorted(counted) == [("S12", 8), ("S12", 9), ("p64", 8), ("p64", 9)]
    assert {p.name for p in (tmp_path / fs.PARTS / "numpy-cpu" / "a").iterdir()} == {"p64.npz", "baseline.npz"}
    counted.clear()
    again = fs.full_depth(SPEC, gguf, gen, tmp_path, primary="baseline")
    assert counted == []
    assert again["acceptance"] == first["acceptance"] and again["nll_calibration"] == first["nll_calibration"]
    assert again["feasibility"] == first["feasibility"]
    f = first["feasibility"]["baseline"]
    assert all(v["ok"] for v in f["implementations"].values()) and "L1.x_out" in f["ranges"]
    assert _without_resumed(again) == _without_resumed(first)
    assert again["passages"]["b"]["resumed"] == ["p64", "baseline"]


def test_a_new_configuration_reuses_the_p64_rows(gguf, tmp_path, counted):
    gen = _generated()
    first = fs.full_depth(SPEC, gguf, gen, tmp_path, primary="baseline")
    counted.clear()
    both = fs.full_depth(SPEC, gguf, gen, tmp_path, primary="baseline", exploratory=("a16",))
    assert sorted(counted) == [("S16", 8), ("S16", 9)]
    assert both["acceptance"] == first["acceptance"] and set(both["exploratory"]) == {"a16"}
    assert both["passages"]["a"]["resumed"] == ["p64", "baseline"]


def test_an_interrupted_run_keeps_its_whole_parts(gguf, tmp_path, counted, monkeypatch):
    gen = _generated()
    ps_forward = fs.ps.forward

    def stops_at_b(ids, *a, **kw):
        if len(ids) == 8:
            raise KeyboardInterrupt
        return ps_forward(ids, *a, **kw)
    monkeypatch.setattr(fs.ps, "forward", stops_at_b)
    with pytest.raises(KeyboardInterrupt):
        fs.full_depth(SPEC, gguf, gen, tmp_path, primary="baseline")
    assert {p.name for p in (tmp_path / fs.PARTS / "numpy-cpu" / "b").iterdir()} == {"p64.npz"}
    monkeypatch.setattr(fs.ps, "forward", ps_forward)
    counted.clear()
    rec = fs.full_depth(SPEC, gguf, gen, tmp_path, primary="baseline")
    assert counted == [("S12", 8)] and rec["passages"]["b"]["resumed"] == ["p64"]
    assert rec["acceptance"] == fs.full_depth(SPEC, gguf, gen, tmp_path / "fresh", primary="baseline")["acceptance"]


def test_a_changed_sequence_is_computed_again(gguf, tmp_path, counted):
    gen = _generated()
    fs.full_depth(SPEC, gguf, gen, tmp_path, primary="baseline")
    gen["b"]["tokens"][-1] = (gen["b"]["tokens"][-1] + 1) % toy.TOY["V"]
    counted.clear()
    rec = fs.full_depth(SPEC, gguf, gen, tmp_path, primary="baseline")
    assert sorted(counted) == [("S12", 8), ("p64", 8)]
    assert rec["passages"]["a"]["resumed"] == ["p64", "baseline"] and rec["passages"]["b"]["resumed"] == []


def test_a_callable_integer_arm_is_never_saved(gguf, tmp_path):
    gen = _generated()
    calls = []

    def integer(ids):
        calls.append(len(ids))
        return np.zeros((len(ids), toy.TOY["V"]))
    for _ in range(2):
        fs.full_depth(SPEC, gguf, gen, tmp_path, primary="integer", integer=integer)
    assert sorted(calls) == [8, 8, 9, 9]
    assert fs.full_depth(SPEC, gguf, gen, tmp_path, primary="integer", integer=integer)["feasibility"] == {}
    assert {p.name for p in (tmp_path / fs.PARTS / "numpy-cpu" / "a").iterdir()} == {"p64.npz"}


def test_the_primary_is_named_and_kept_by_the_work_directory(gguf, tmp_path):
    gen = _generated()
    rec = fs.full_depth(SPEC, gguf, gen, tmp_path, primary="a16", exploratory=("baseline",))
    assert rec["primary"] == "a16" and set(rec["exploratory"]) == {"baseline"}
    swapped = fs.full_depth(SPEC, gguf, gen, tmp_path, primary="a16", exploratory=())
    assert swapped["acceptance"] == rec["acceptance"]               # the order of the others is irrelevant
    with pytest.raises(ValueError, match="started with primary 'a16'"):
        fs.full_depth(SPEC, gguf, gen, tmp_path, primary="baseline", exploratory=("a16",))
    with pytest.raises(ValueError, match="not also exploratory"):
        fs.full_depth(SPEC, gguf, gen, tmp_path / "x", primary="a16", exploratory=("a16",))
    with pytest.raises(SystemExit):
        fs.main(["full-depth", "--model", "moonlight", "--gguf", "g", "--passages", "p", "--runner", "r",
                 "--render", "r", "--work", str(tmp_path / "cli")])


def test_the_reference_backend_is_recorded_and_keys_every_part(gguf, tmp_path, counted):
    gen = _generated()
    np_rec = fs.full_depth(SPEC, gguf, gen, tmp_path, primary="baseline")
    assert {k: np_rec["reference"][k] for k in ("backend", "device", "dtype")} == \
        dict(backend="numpy", device="cpu", dtype="float64")
    counted.clear()
    t_rec = fs.full_depth(SPEC, gguf, gen, tmp_path, primary="baseline", backend="torch", device="cpu")
    assert {k: t_rec["reference"][k] for k in ("backend", "device", "dtype")} == \
        dict(backend="torch", device="cpu", dtype="float64")
    # NumPy's p64 is not called, and the configuration is scored again against torch's
    assert sorted(counted) == [("S12", 8), ("S12", 9)] and t_rec["passages"]["a"]["resumed"] == []
    counted.clear()
    again = fs.full_depth(SPEC, gguf, gen, tmp_path, primary="baseline")      # NumPy's parts are kept
    assert counted == [] and again["acceptance"] == np_rec["acceptance"]
    for name in ("mean_kl", "mean_nll"):
        a, b = (r["acceptance"]["stats"]["integer"][name] for r in (np_rec, t_rec))
        assert abs(a - b) <= 1e-10 + 1e-9 * abs(a), (name, a, b)       # float64 both, summed apart
    with pytest.raises(ValueError, match="runs on the cpu"):
        fs.full_depth(SPEC, gguf, gen, tmp_path, primary="baseline", device="cuda")


def test_two_layers_keep_every_range_row_with_its_implementation(gguf, tmp_path):
    """The integer configurations alone (no llama.cpp): each passage's rows
    whole (signed extremes, bounds, claim, implementation), the violations
    per implementation, and the feasibility over every passage; per-passage
    runs merged give the one run's."""
    gen = _generated()
    ids = {pid: g["tokens"] for pid, g in gen.items()}
    kw = dict(configs=("baseline", "a14w14"), arms=(), with_engine=False)
    rec = fs.two_layer(SPEC, gguf, ids, None, tmp_path / "all", **kw)
    for pid in ids:
        row = rec["passages"][pid]["int/a14w14"]
        assert set(row["violations"]) == set(fs.ps.IMPLEMENTATIONS) and row["ranges"]
        for r in row["ranges"].values():
            assert {"min", "max", "lo", "hi", "ok", "inequality", "claim", "implementation"} <= set(r)
        own = {r["implementation"] for r in row["ranges"].values()}
        assert own == {None, *fs.ps.IMPLEMENTATIONS}            # common rows and each one's own
    f = rec["feasibility"]["a14w14"]
    assert f["ranges"] == fs.ki.Ranges.from_dicts(*(rec["passages"][p]["int/a14w14"]["ranges"]
                                                    for p in ids)).as_dict()
    assert set(f["implementations"]) == set(f["violations"]) == set(fs.ps.IMPLEMENTATIONS)
    parts = [fs.two_layer(SPEC, gguf, {pid: i}, None, tmp_path / pid, **kw) for pid, i in ids.items()]
    merged = fs.merge_two_layer(parts)
    assert merged["feasibility"] == rec["feasibility"] and list(merged["passages"]) == list(ids)
    with pytest.raises(ValueError, match="passages in two records"):
        fs.merge_two_layer([parts[0], parts[0]])
    files = []
    for part in parts:
        files.append(tmp_path / f"{next(iter(part['passages']))}.json")
        files[-1].write_text(json.dumps(part, default=str))
    assert fs.main(["two-layer-merge", "--work", str(tmp_path), "--out", "m.json", *map(str, files)]) == 0
    assert json.loads((tmp_path / "m.json").read_text())["feasibility"] == json.loads(
        json.dumps(rec["feasibility"], default=str))


def test_a_violation_is_reported_under_its_own_implementation():
    rg = fs.ki.Ranges()
    rg.bound("L0.sm.x", [-5, 40], -16, 15, "softmax input", implementation="input-rescale")
    rg.bound("L0.sm.x", [-5, 9], -16, 15, "softmax input", implementation="combined-rounding")
    rg.bound("L0.n1.out", [0, 3], 0, 7, "rmsnorm output")
    v = fs.violations_by_implementation(rg)
    assert v["combined-rounding"] == {}
    assert list(v["input-rescale"]) == ["L0.sm.x [input-rescale]"]
    r = v["input-rescale"]["L0.sm.x [input-rescale]"]
    assert (r["min"], r["max"], r["lo"], r["hi"], r["implementation"]) == (-5, 40, -16, 15, "input-rescale")
