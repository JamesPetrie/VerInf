"""The K2 driver's tape structure on the CPU (tests/_cpu_tape.py).

- The 2^12 baseline's claims, variables and tables are exactly those before
  the precision profiles existed: the digests were taken at f4c265d, the
  last revision without them, for the unbridged, bridged and long-position
  builds.
- a14w14 is combined rounding: no claim takes an input rescale; the scores,
  the gate projections and the router logits leave their matmuls at the
  table scale and the latents at s_norm; the residual stream reaches each
  RMSNorm through one standalone 2-bit rescale (two per layer and the final
  norm); RMSNorm runs at s_norm with its output at S_act; the tail takes
  the logits at 2^14 with its kernel unchanged; and every name the driver
  traces is one the precision study's model computes.
- a16w16 is the same design at its own scales, not a copy of a14w14's
  tape: the consumers' rescales are 20 bits, the residual reaches RMSNorm
  through a 4-bit rescale, the combine's low word is 2^28 entries and the
  tail's tables 2^24.

Each build runs in a fresh process (claim and table names carry
process-wide counters), all of them at once."""
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

BASELINE = {
    "plain": "49386aa1e14aa5dfe71f3d59b7018abbba83da4a5625dcdf798a939b8e259ee2",
    "bridge": "f47c6156f5f7025a0681660ed409983788e511e4db338fd198503c528f081f9b",
    "long": "53b4255e6c5e9efcdf3575d4bd6c40427136d42bde6959d2c8645b59ce8bf21c",
}
BUILDS = {"plain": {}, "bridge": dict(bridge=True), "long": dict(offset=130_072),
          "a14w14": dict(precision="a14w14"), "a16w16": dict(precision="a16w16")}
PROMPT, CONT = [3, 17, 0], [41, 9, 9]

_SCRIPT = """
import json, sys
import _cpu_tape as ct, _k2_toy as toy
with ct.cuda_on_cpu():
    import demo_k2 as dk
    kw = json.loads(sys.argv[2])
    b = dk.build(sys.argv[1], %r, %r, cfg=toy.TOY_MLA, layers=2, kk=toy.TOY["k"], **kw)
    s = ct.structure(b.tape)
    kinds = lambda name: [c for c in b.tape.claims if type(c).__name__ == name]
    made = {}                                   # an output's producer: (claim, its rescale bits)
    for c in kinds("MatmulClaim"):
        made[c.C.name] = ("matmul", c.rescale_bits)
    for c in kinds("RescaleClaim"):
        made[c.x.name] = ("rescale", c.rescale_bits)
    fed = lambda name: sorted({tuple(made.get(c.x.name, ("other", None))) for c in kinds(name)}, key=str)
    out = dict(digest=s["digest"], claims=len(s["claims"]), tables={t[1]: t[2] for t in s["tables"]},
               rescales=[c.rescale_bits for c in kinds("RescaleClaim")],
               matmul_bits=sorted({c.rescale_bits for c in kinds("MatmulClaim") if c.rescale_bits}),
               rmsnorms=sorted({(c.config.s_in, c.config.s, c.config.s_out) for c in kinds("RmsNormClaim")}),
               silu_in=sorted({c.config.s_in for c in kinds("SiluClaim")}),
               softmax=sorted({(c.config.s_x, c.config.rescale_bits) for c in kinds("SoftmaxClaim")}),
               fed=dict(softmax=fed("SoftmaxClaim"), silu=fed("SiluClaim"), rmsnorm=fed("RmsNormClaim")),
               trace=sorted(b.trace), info=b.info)
print(json.dumps(out, default=str))
""" % (PROMPT, CONT)


def _env():
    return dict(os.environ, PYTHONPATH=os.pathsep.join(str(_ROOT / p) for p in ("prover", "prover/tests", "demo")),
                LIGERO_ELL="8192", LIGERO_K_DEG="16384", LIGERO_N_LIG="65536")


@pytest.fixture(scope="module")
def gguf(tmp_path_factory):
    path = tmp_path_factory.mktemp("k2struct") / "toy.gguf"
    toy.write(path)
    return path


@pytest.fixture(scope="module")
def built(gguf):
    """Every build of BUILDS, each in its own fresh process, run concurrently."""
    procs = {n: subprocess.Popen([sys.executable, "-c", _SCRIPT, str(gguf), json.dumps(kw)], env=_env(),
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
             for n, kw in BUILDS.items()}
    out = {}
    for n, p in procs.items():
        so, se = p.communicate()
        assert p.returncode == 0, (n, se[-3000:])
        out[n] = json.loads(so.strip().splitlines()[-1])
    return out


@pytest.mark.parametrize("build", sorted(BASELINE))
def test_the_baseline_tape_is_unchanged(built, build):
    assert built[build]["digest"] == BASELINE[build], build
    assert built[build]["info"]["precision"]["S_act"] == 12


def test_a14w14_is_combined_rounding(built):
    a, base = built["a14w14"], built["plain"]
    A, N, Tb = 1 << 14, 1 << 12, 1 << 12
    # no claim takes an input rescale (s_in 0 or equal to s); RMSNorm at s_norm, its output at S_act
    assert [(s, s_out) for _, s, s_out in a["rmsnorms"]] == [(N, A)]
    assert all(s_in in (0, s) for s_in, s, _ in a["rmsnorms"])
    assert set(a["silu_in"]) <= {0, Tb} and a["softmax"] == [[Tb, 0]]
    # each table's input leaves its producer at the table's scale (16 = log2(A·A/Tb) = log2(A·Wt/Tb))
    assert a["fed"]["softmax"] == [["matmul", 16]]
    assert a["fed"]["silu"] == [["matmul", 16], ["rescale", 16]]          # dense/shared, routed
    # RMSNorm's inputs: the latents straight from their matmuls, the residual through a 2-bit rescale
    assert a["fed"]["rmsnorm"] == [["matmul", 16], ["rescale", 2]]
    assert a["rescales"].count(2) == 2 * 2 + 1 and base["rescales"].count(2) == 0
    assert a["matmul_bits"] == [12, 14, 16] and base["matmul_bits"] == [12]
    assert 26 in a["rescales"]                               # the combine, from S_w·A·Wt
    # the baseline's RMSNorms read the stream directly: the embedding select, the latents, the residual adds
    assert base["fed"]["rmsnorm"] == [["matmul", 0], ["matmul", 12], ["other", None]]


def test_a14w14s_tables(built):
    import k2_loader as kl
    a, base = built["a14w14"], built["plain"]
    tail = kl.tail_params(1 << 14)
    assert a["tables"]["ui_exp"] == tail["gap_max"] == 1 << 22 and base["tables"]["ui_exp"] == 1 << 20
    assert a["tables"]["ui_pow"] == base["tables"]["ui_pow"]           # the surprisal's resolution, s_b
    assert a["info"]["tail"] == tail and base["info"]["tail"]["s_b"] == tail["s_b"]
    assert a["tables"]["k2_sigmoid"] == base["tables"]["k2_sigmoid"]     # router logit at S_tab either way
    for w in (2, 10, 14, 16, 26):
        assert f"rescale_w{w}" in a["tables"], w
    assert a["tables"]["output_w26"] == base["tables"]["output_w26"]


def test_a14w14s_trace_is_the_precision_studys(built, gguf):
    import k2_loader as kl
    import k2_precision_study as ps
    out, ranges = ps.forward(PROMPT + CONT, kl.GgufWeights(str(gguf), ints=False, cfg=toy.TOY_MLA),
                             toy.TOY_MLA, ps.CONFIGS["a14w14"], kk=toy.TOY["k"], head_chunk=20)
    assert set(built["a14w14"]["trace"]) <= set(out)
    assert not ranges.violations("combined-rounding")


def test_a16w16_is_combined_rounding_at_its_own_scales(built):
    a = built["a16w16"]
    A, N, Tb = 1 << 16, 1 << 12, 1 << 12
    assert [(s, s_out) for _, s, s_out in a["rmsnorms"]] == [(N, A)]
    assert all(s_in in (0, s) for s_in, s, _ in a["rmsnorms"])
    assert set(a["silu_in"]) <= {0, Tb} and a["softmax"] == [[Tb, 0]]
    # 20 = log2(A·A/Tb) = log2(A·Wt/Tb) = log2(A·Wt/N)
    assert a["fed"]["softmax"] == [["matmul", 20]]
    assert a["fed"]["silu"] == [["matmul", 20], ["rescale", 20]]
    assert a["fed"]["rmsnorm"] == [["matmul", 20], ["rescale", 4]]       # the residual: 2^16 to 2^12
    assert a["rescales"].count(4) == 2 * 2 + 1 and 2 not in a["rescales"]
    assert a["matmul_bits"] == [12, 16, 20]
    assert 28 in a["rescales"]                               # the combine, from S_w·A·Wt
    assert a["claims"] == built["a14w14"]["claims"]          # the same sites as a14w14


def test_a16w16s_tables(built):
    import k2_loader as kl
    a, a14 = built["a16w16"], built["a14w14"]
    tail = kl.tail_params(1 << 16)
    assert tail == dict(s_c=1 << 36, s_y=1 << 28, s_b=1 << 12, gap_max=1 << 24)
    k = tail["s_c"] // tail["s_b"]
    assert ((tail["gap_max"] - 1) ** 2 + k - 1) // k == (1 << 24) - 1            # at its maximum, below 2^24
    assert a["info"]["tail"] == tail
    assert a["tables"]["ui_exp"] == a["tables"]["mx1_gap"] == tail["gap_max"]
    assert a["tables"]["ui1_rem"] == k == 1 << 24
    assert a["tables"]["ui_pow"] == a14["tables"]["ui_pow"]                     # s_b
    assert a["tables"]["rescale_w28"] == 1 << 28 and "rescale_w26" not in a["tables"]
    for w in (4, 16, 20, 28):
        assert f"rescale_w{w}" in a["tables"], w
    assert a["tables"]["k2_sigmoid"] == a14["tables"]["k2_sigmoid"]


def test_a16w16s_trace_is_the_precision_studys(built, gguf):
    import k2_loader as kl
    import k2_precision_study as ps
    out, ranges = ps.forward(PROMPT + CONT, kl.GgufWeights(str(gguf), ints=False, cfg=toy.TOY_MLA),
                             toy.TOY_MLA, ps.CONFIGS["a16w16"], kk=toy.TOY["k"], head_chunk=20)
    assert set(built["a16w16"]["trace"]) <= set(out)
    assert not ranges.violations("combined-rounding")
