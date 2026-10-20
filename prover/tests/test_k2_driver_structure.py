"""The K2 driver's tape structure on the CPU (tests/_cpu_tape.py): the 2^12
baseline's claims, variables and tables exactly as before the precision
profiles existed (the digests were taken at f4c265d, the last revision
without them), for the unbridged, bridged and long-position builds. Each
digest comes from a fresh process: claim and table names carry
process-wide counters."""
import json
import os
import pathlib
import subprocess
import sys

import pytest

_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_HERE))

import _k2_toy as toy                  # noqa: E402

BASELINE = {
    "plain": "49386aa1e14aa5dfe71f3d59b7018abbba83da4a5625dcdf798a939b8e259ee2",
    "bridge": "f47c6156f5f7025a0681660ed409983788e511e4db338fd198503c528f081f9b",
    "long": "53b4255e6c5e9efcdf3575d4bd6c40427136d42bde6959d2c8645b59ce8bf21c",
}
BUILDS = {"plain": {}, "bridge": dict(bridge=True), "long": dict(offset=130_072)}

_SCRIPT = """
import json, sys
import _cpu_tape as ct, _k2_toy as toy
with ct.cuda_on_cpu():
    import demo_k2 as dk
    kw = json.loads(sys.argv[2])
    b = dk.build(sys.argv[1], [3, 17, 0], [41, 9, 9], cfg=toy.TOY_MLA, layers=2, kk=toy.TOY["k"], **kw)
    s = ct.structure(b.tape)
print(json.dumps(dict(digest=s["digest"], claims=len(s["claims"]),
                      tables={t[1]: t[2] for t in s["tables"]},
                      kinds=sorted({c[0] for c in s["claims"]}))))
"""


def build_structure(gguf, **kw):
    """One fresh process's build of the toy K2 tape: its digest and counts."""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(str(_ROOT / p) for p in ("prover", "prover/tests", "demo")),
               LIGERO_ELL="8192", LIGERO_K_DEG="16384", LIGERO_N_LIG="65536")
    out = subprocess.run([sys.executable, "-c", _SCRIPT, str(gguf), json.dumps(kw)], env=env,
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def gguf(tmp_path_factory):
    path = tmp_path_factory.mktemp("k2struct") / "toy.gguf"
    toy.write(path)
    return path


@pytest.mark.parametrize("build", sorted(BUILDS))
def test_the_baseline_tape_is_unchanged(gguf, build):
    assert build_structure(gguf, **BUILDS[build])["digest"] == BASELINE[build], build
