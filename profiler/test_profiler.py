"""No-torch regression tests for the profiler.

    python3 profiler/test_profiler.py        # or: pytest profiler/test_profiler.py

Stubs prover `core` with shape-compatible fakes carrying the REAL field
spellings (nested configs, `length` vs `L`, shared tables, name collisions),
so extract_tape and every downstream consumer run on any box. The GPU
selftest (`extract.py --selftest`) complements this where torch exists.
The stub is installed only inside the `_fake_core()` context (any real
`core` module is restored on exit), so repo-wide pytest collection is safe.
"""
import argparse
import contextlib
import io
import json
import os
import sys
import tempfile
import types
from dataclasses import dataclass, field
from typing import List

# --- stub prover `core` (extract's deferred `from core import ...`) ---
_core = types.ModuleType("core")


@dataclass(eq=False)
class Variable:
    name: str
    length: int
    phase: int = 1
    persistent: bool = False


@dataclass(eq=False)
class Table:
    name: str
    mult_var: Variable
    w_var: Variable
    z_vars: List[Variable] = field(default_factory=list)


_core.Variable, _core.Table = Variable, Table
_ABSENT = object()


@contextlib.contextmanager
def _fake_core():
    """Scoped stub install: extract_tape's call-time `from core import ...`
    resolves to the fakes only inside this context; whatever was imported
    before (nothing, or the real prover core) is restored on exit."""
    prev = sys.modules.get("core", _ABSENT)
    sys.modules["core"] = _core
    try:
        yield
    finally:
        if prev is _ABSENT:
            sys.modules.pop("core", None)
        else:
            sys.modules["core"] = prev

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract import extract_tape            # noqa: E402
from machine import MachineProfile          # noqa: E402
from manifest import Manifest, ClaimRecord  # noqa: E402
import claimcosts                           # noqa: E402
import cli                                  # noqa: E402
import dag                                  # noqa: E402
import partition                            # noqa: E402
import predict                              # noqa: E402

# --- fake claims with the real prover field spellings ---


@dataclass
class MatmulClaim:
    a: Variable; b: Variable; c: Variable
    y: Variable; u: Variable; p: Variable
    m: int; k: int; n: int
    heads: int = 1
    rescale_bits: int = 0


@dataclass
class RmsNormConfig:
    # like the real config, the rescale flags are PROPERTIES derived from
    # the scale fields, not dataclass fields — extraction must surface them
    B: int; d: int
    s: int = 4096
    s_in: int = 0        # 0 -> no input rescale (the production mode)
    s_out: int = 4096    # 0 -> no output rescale

    @property
    def rescale_bits(self) -> int:
        return 0 if self.s_in in (0, self.s) else \
            (self.s_in // self.s).bit_length() - 1

    @property
    def output_rescale_bits(self) -> int:
        return 12 if self.s_out else 0


@dataclass
class RmsNormClaim:
    x: Variable; output: Variable
    config: RmsNormConfig = None


@dataclass
class SoftmaxConfig:
    B: int; M: int
    s_x: int = 4096
    s_in: int = 0        # 0 -> no input rescale
    saturate: bool = False   # production opts in, like the real default
    causal: bool = False
    heads: int = 1

    @property
    def rescale_bits(self) -> int:
        return 0 if self.s_in in (0, self.s_x) else \
            (self.s_in // self.s_x).bit_length() - 1


@dataclass
class SoftmaxClaim:
    x: Variable; output: Variable
    length: int = 0
    config: SoftmaxConfig = None


@dataclass
class AddClaim:
    a: Variable; b: Variable; c: Variable
    length: int = 0


@dataclass
class SiluConfig:
    r: int = 12
    s_in: int = 0        # 0 -> no rescale (the production mode)

    @property
    def rescale_bits(self) -> int:
        s_x = 1 << self.r
        return 0 if self.s_in in (0, s_x) else \
            (self.s_in // s_x).bit_length() - 1


@dataclass
class SiluClaim:
    x: Variable; output: Variable
    length: int = 0
    config: SiluConfig = None


@dataclass
class RoPEConfig:
    SEQ: int; d_h: int; s_x: int
    heads: int = 1


@dataclass
class RoPEClaim:
    x: Variable; x_rot: Variable
    rescale_bits: int = 0
    config: RoPEConfig = None


@dataclass
class EmbeddingLookupClaim:
    table: Variable; output: Variable
    d: int = 0
    token_ids: List[int] = field(default_factory=list)


@dataclass
class RangeWordClaim:
    x: Variable
    z: Variable
    table: Table = None
    length: int = 0


@dataclass
class WordExtractionClaim:
    x: Variable
    words: List[Variable]
    length: int = 0


@dataclass
class FreivaldsCombineClaim:
    y: Variable            # logical T*F output
    u: Variable            # phase-2 aux (pollutes naive out_len sums)
    T: int = 0
    E: int = 0
    F: int = 0
    y_committed: bool = True


@dataclass
class TableSettlement:
    table: Table


class FakeTape:
    def __init__(self, cfg, lazy=True):
        self.cfg, self.lazy = cfg, lazy
        self.claims, self._deferred = [], []

    def add(self, claim, inputs):
        self.claims.append(claim)
        if self.lazy:
            self._deferred.append((claim, tuple(inputs), None))


class Cfg:
    ELL, K_DEG, N_LIG, T_QUERIES = 64, 128, 256, 4


S, D = 8, 16
T_LEN = 1 << 12


def _build_manifest():
    t = FakeTape(Cfg())
    # layer-0 chain: rmsnorm -> matmul(weight) -> softmax -> silu -> rope
    x0 = Variable("x_input", S * D)
    w0 = Variable("W_Q_L0", D * D, persistent=True)
    n0 = Variable("x_input@rms_L0#1", S * D)
    t.add(RmsNormClaim(x=x0, output=n0, config=RmsNormConfig(B=S, d=D)), [x0])
    q0 = Variable("x_input@W_Q_L0#2", S * D)
    t.add(MatmulClaim(a=n0, b=w0, c=q0,
                      y=Variable("mm_y#3", D, phase=2),
                      u=Variable("mm_u#4", D, phase=2),
                      p=Variable("mm_p#5", D, phase=2),
                      m=S, k=D, n=D, rescale_bits=12), [n0, w0])
    sm = Variable("sm_L0#6", S * S)
    t.add(SoftmaxClaim(x=q0, output=sm, length=S * S,
                       config=SoftmaxConfig(B=S, M=S, causal=True,
                                            saturate=True)), [q0])
    # `side` is a _deferred-listed input of TWO claims without being a
    # dataclass field of either — the input=>consumer direction must hold
    # on the second sight too.
    side = Variable("side_stream", S)
    sl = Variable("silu_L0#7", S * S)
    t.add(SiluClaim(x=sm, output=sl, length=S * S,
                    config=SiluConfig()), [sm, side])
    rp = Variable("rope_L0#8", S * D)
    t.add(RoPEClaim(x=q0, x_rot=rp, rescale_bits=12,
                    config=RoPEConfig(SEQ=S, d_h=D, s_x=4096)), [q0, side])
    # name-COLLISION chain: legal names x, x~1, x again
    sm2 = Variable("sm_L0#6", S * D)
    t.add(AddClaim(a=rp, b=x0, c=sm2, length=S * D), [rp, x0])
    sm3a = Variable("sm_L0#6~1", S)           # literal ~1 already taken
    t.add(AddClaim(a=sm2, b=sm2, c=sm3a, length=S), [sm2])
    sm3b = Variable("sm_L0#6", 2 * S)         # third distinct object
    t.add(AddClaim(a=sm2, b=sm2, c=sm3b, length=2 * S), [sm2])
    # embed lookup + word extraction + TWO range lookups on one shared table
    tbl = Table("range_w12", Variable("range_w12_mult", T_LEN),
                Variable("range_w12_w", T_LEN, phase=2))
    el = Variable("emb_L1#9", 4 * D)
    t.add(EmbeddingLookupClaim(
        table=Variable("token_embd", 100 * D, persistent=True),
        output=el, d=D, token_ids=[1, 2, 3, 4]), [])
    wd = [Variable(f"w{i}_L1#1{i}", 4 * D, phase=2) for i in range(2)]
    t.add(WordExtractionClaim(x=el, words=wd, length=4 * D), [el])
    z1 = Variable("z_rw#10", S, phase=2)
    tbl.z_vars.append(z1)
    t.add(RangeWordClaim(x=el, z=z1, table=tbl, length=S), [el])
    z2 = Variable("z_rw#11", S, phase=2)
    tbl.z_vars.append(z2)
    t.add(RangeWordClaim(x=el, z=z2, table=tbl, length=S), [el])
    # freivalds combine whose extracted outputs include phase-2 aux
    e0 = Variable("e0_L1#20", 2 * 4)
    t.add(AddClaim(a=el, b=el, c=e0, length=8), [el])
    e1 = Variable("e1_L1#21", 2 * 4)
    t.add(AddClaim(a=el, b=el, c=e1, length=8), [el])
    t.add(FreivaldsCombineClaim(y=Variable("fc_y_L1#22", 2 * 4),
                                u=Variable("fc_u_L1#23", 2 * 2, phase=2),
                                T=2, E=2, F=4), [e0, e1])
    with _fake_core():
        return extract_tape(t, model=dict(name="fake"), seq=S)


def _assert_io_consistency(man):
    """Bidirectional contract: a claim lists exactly what it touches.
    consumer => listed input, input => registered consumer; same for
    producer/outputs."""
    by_name = {v.name: v for v in man.variables}
    for c in man.claims:
        for name in c.inputs:
            assert c.idx in by_name[name].consumers, ("input w/o consumer",
                                                      c.idx, name)
        for name in c.outputs:
            assert by_name[name].producer == c.idx, ("output w/o producer",
                                                     c.idx, name)
    for v in man.variables:
        for ci in v.consumers:
            assert v.name in man.claims[ci].inputs, ("consumer w/o input",
                                                     v.name, ci)
        if v.producer is not None:
            assert v.name in man.claims[v.producer].outputs, (
                "producer w/o output", v.name, v.producer)


def test_extractor():
    man = _build_manifest()
    by_name = {v.name: v for v in man.variables}
    _assert_io_consistency(man)
    # params: nested configs flattened, prover spellings normalized;
    # property-backed flags surfaced despite not being dataclass fields
    p0 = man.claims[0].params
    assert p0["B"] == S and p0["d"] == D
    assert p0["rescale_bits"] == 0 and p0["output_rescale_bits"] == 12
    assert p0["rescale"] is False        # input rescale OFF = production
    assert man.claims[1].params["rescale"] is True
    assert man.claims[2].params["causal"] is True and man.claims[2].params["B"] == S
    assert man.claims[2].params["saturate"] is True
    assert man.claims[3].params["rescale"] is False      # silu, via property
    emb = next(c for c in man.claims if c.type == "EmbeddingLookupClaim")
    assert emb.params["n_tokens"] == 4
    we = next(c for c in man.claims if c.type == "WordExtractionClaim")
    assert we.params["n_words"] == 2
    # labels + layers parsed from variable names
    assert man.claims[1].label == "x_input@W_Q_L0#2" and man.claims[1].layer == 0
    assert man.claims[0].layer == 0 and man.claims[5].layer == 0
    # weight is input, not output
    assert by_name["W_Q_L0"].producer is None and by_name["W_Q_L0"].persistent
    # ...even when a claim's _deferred input list omits it (the embed claim
    # is added with inputs=[]): persistent => committed run input
    emb_idx = next(c.idx for c in man.claims
                   if c.type == "EmbeddingLookupClaim")
    assert by_name["token_embd"].producer is None
    assert by_name["token_embd"].consumers == [emb_idx]
    # ...and the fallback keeps the manifest contract: consumed => listed
    assert man.claims[emb_idx].inputs == ["token_embd"]
    # non-field _deferred input consumed twice: both sights registered
    assert by_name["side_stream"].consumers == [3, 4]
    # matmul w_slots: output + y/u/p aux = S*D + 3*D
    assert man.claims[1].w_slots == S * D + 3 * D
    # collision chain: four distinct records, no dropped lengths
    sm_recs = [v for v in man.variables if v.name.startswith("sm_L0#6")]
    assert len(sm_recs) == 4 and len({v.name for v in sm_recs}) == 4
    assert sorted(v.length for v in sm_recs) == sorted([S * S, S * D, S, 2 * S])
    # settlement roles: mult is a shared run input touched by both lookups;
    # each z is consumed by the settlement ONLY; w is the settlement's output
    settle = man.claims[-1]
    assert settle.type == "TableSettlement" and settle.params["T_LEN"] == T_LEN
    rw = [c.idx for c in man.claims if c.type == "RangeWordClaim"]
    assert by_name["range_w12_mult"].producer is None
    assert by_name["range_w12_mult"].consumers == rw + [settle.idx]
    for ci in rw:   # lookups increment mult => it is an input of theirs
        assert "range_w12_mult" in man.claims[ci].inputs
    assert by_name["z_rw#10"].consumers == [settle.idx]
    assert by_name["z_rw#11"].consumers == [settle.idx]
    assert by_name["range_w12_w"].producer == settle.idx
    assert settle.outputs == ["range_w12_w"] and settle.w_slots == T_LEN
    assert set(settle.inputs) == {"range_w12_mult", "z_rw#10", "z_rw#11"}
    # global ordering invariant: nothing is consumed before it is produced
    for v in man.variables:
        if v.producer is not None:
            assert all(ci > v.producer for ci in v.consumers), (v.name, v)
    # eager tapes refused
    eager = FakeTape(Cfg(), lazy=False)
    eager.add(AddClaim(a=Variable("a", S), b=Variable("b", S),
                       c=Variable("c", S), length=S), [])
    try:
        with _fake_core():
            extract_tape(eager, model={}, seq=S)
        raise AssertionError("eager tape was accepted")
    except ValueError:
        pass


def test_explicit_settlement_reused():
    # A tape that carries its own TableSettlement must not get a second
    # synthesized one; the explicit record gets the full wiring instead.
    t = FakeTape(Cfg())
    tbl = Table("range_w8", Variable("range_w8_mult", 1 << 8),
                Variable("range_w8_w", 1 << 8, phase=2))
    x = Variable("x_in", S)
    z = Variable("z#1", S, phase=2)
    tbl.z_vars.append(z)
    t.add(RangeWordClaim(x=x, z=z, table=tbl, length=S), [x])
    t.add(TableSettlement(table=tbl), [])
    with _fake_core():
        man = extract_tape(t, model=dict(name="fake"), seq=S)
    _assert_io_consistency(man)
    settles = [c for c in man.claims if c.type == "TableSettlement"]
    assert len(settles) == 1 and settles[0].idx == 1
    s = settles[0]
    assert s.params["T_LEN"] == 1 << 8 and s.label == "settle.range_w8"
    assert set(s.inputs) == {"range_w8_mult", "z#1"}
    assert s.outputs == ["range_w8_w"] and s.w_slots == 1 << 8
    by_name = {v.name: v for v in man.variables}
    assert by_name["range_w8_w"].producer == s.idx
    assert by_name["range_w8_mult"].consumers == [0, s.idx]
    assert by_name["z#1"].consumers == [s.idx]


def test_core_isolation():
    # The stub must never leak into repo-wide pytest collection.
    before = sys.modules.get("core")
    with _fake_core():
        assert sys.modules["core"] is _core
    assert sys.modules.get("core") is before


def test_costs():
    # rmsnorm: wrap-free bracket per-row constants (paper A.1/B.4), NOT the
    # pre-fix 26B/7B/13B row that maverick-cost-model.md still carries
    assert claimcosts.cost("RmsNormClaim", {"B": S, "d": D}) == (
        7 * S * D + 82 * S, 17 * S + 2 * S * D, 3 * S * D + 42 * S)
    # prover spellings accepted; hadamard rescale-OFF has no linear packets
    assert claimcosts.cost("HadamardClaim",
                           {"length": 10, "rescale": False}) == (10, 0, 10)
    # word/range carry real generic costs
    assert claimcosts.cost("WordExtractionClaim",
                           {"length": 64, "n_words": 2}) == (128, 64, 0)
    assert claimcosts.cost("RangeWordClaim", {"length": S}) == (S, 0, S)
    # settlement: w output (T_LEN) + T_LEN+1 cids
    assert claimcosts.cost("TableSettlement",
                           {"T_LEN": T_LEN}) == (T_LEN, T_LEN + 1, 0.0)
    # routing_core + word + nw*range == the bundled synth `routing` row
    T_, E_, nw_ = 4, 8, 3
    TE_ = T_ * E_
    bundle = claimcosts.cost("routing", {"T": T_, "E": E_, "n_words": nw_})
    core = claimcosts.cost("RoutingClaim", {"T": T_, "E": E_, "L_bits": 3})
    word = claimcosts.cost("WordExtractionClaim",
                           {"length": TE_, "n_words": nw_})
    rng = claimcosts.cost("RangeWordClaim", {"length": TE_})
    assert tuple(c + w + nw_ * r for c, w, r in zip(core, word, rng)) == bundle
    # production rmsnorm = input rescale OFF + output rescale ON: the
    # explicit extracted spelling of that mode must be ACCEPTED
    assert claimcosts.cost("RmsNormClaim",
                           {"B": S, "d": D, "rescale": False,
                            "output_rescale_bits": 12}) == (
        7 * S * D + 82 * S, 17 * S + 2 * S * D, 3 * S * D + 42 * S)
    # non-production modes are REJECTED, not priced with the wrong formula
    for typ, params in [
        ("RoPEClaim", {"SEQ": 4, "heads": 1, "d_h": 4, "rescale": False}),
        ("SiluClaim", {"length": 8, "rescale": True}),
        ("RmsNormClaim", {"B": 2, "d": 4, "rescale": True}),   # input ⓡ
        ("RmsNormClaim", {"B": 2, "d": 4, "output_rescale_bits": 0}),
        ("SoftmaxClaim", {"B": 2, "M": 2, "rescale": True}),   # input ⓡ
        ("SoftmaxClaim", {"B": 2, "M": 2, "saturate": False}),
    ]:
        try:
            claimcosts.cost(typ, params)
            raise AssertionError(f"priced unsupported mode: {typ} {params}")
        except ValueError:
            pass


def test_mode_flags_extracted():
    # the guard chain end to end: unsupported modes carried only by
    # PROPERTY-backed config flags must be surfaced by extraction and
    # rejected by claimcosts (real-shaped configs, not hand-written dicts)
    t = FakeTape(Cfg())
    x = Variable("x0", S * D)
    t.add(RmsNormClaim(x=x, output=Variable("x0@rms#1", S * D),
                       config=RmsNormConfig(B=S, d=D, s_in=8192)), [x])
    t.add(SoftmaxClaim(x=x, output=Variable("x0@sm#2", S * S), length=S * S,
                       config=SoftmaxConfig(B=S, M=S, saturate=True,
                                            s_in=8192)), [x])
    t.add(SiluClaim(x=x, output=Variable("x0@silu#3", S), length=S,
                    config=SiluConfig(s_in=1 << 13)), [x])
    t.add(RmsNormClaim(x=x, output=Variable("x0@rms#4", S * D),
                       config=RmsNormConfig(B=S, d=D, s_out=0)), [x])
    with _fake_core():
        man = extract_tape(t, model=dict(name="fake"), seq=S)
    assert man.claims[0].params["rescale_bits"] == 1     # 8192/4096, property
    assert man.claims[0].params["rescale"] is True
    for c in man.claims:
        try:
            claimcosts.cost(c.type, c.params)
            raise AssertionError(f"priced unsupported mode: {c.type}")
        except ValueError:
            pass


def test_expert_labels():
    # synth convention and the extracted a@b-derived weight names both parse;
    # shared-expert and attention weights never do
    assert partition._expert_of("L3.moe.e12.gate") == 12
    assert partition._expert_of("x_r@L1_Wg0#123") == 0
    assert partition._expert_of("hidden@L7_Wd31#88") == 31
    assert partition._expert_of("x_r@L2_Wu5#40@more.derived") == 5
    assert partition._expert_of("n2g@L1_Wgs#4") is None      # shared expert
    assert partition._expert_of("x_input@W_Q_L0#2") is None  # attention
    assert partition._expert_of("") is None
    # mixed conventions: rightmost by POSITION wins, not by regex order
    assert partition._expert_of("x_r@L2_Wg2#3.more.e1.tail") == 1
    assert partition._expert_of("a.e3.b@L1_Wg7#2") == 7


def test_manifest_validation():
    mp = MachineProfile.load("gb10-spark")
    with tempfile.TemporaryDirectory() as td:
        empty = os.path.join(td, "empty.json")
        with open(empty, "w") as f:
            json.dump({"schema_version": 1, "claims": [], "variables": []}, f)
        future = os.path.join(td, "future.json")
        with open(future, "w") as f:
            json.dump({"schema_version": 99,
                       "claims": [{"idx": 0, "type": "add",
                                   "params": {"L": 4}}]}, f)
        # structurally malformed: list root, record missing a required
        # field, bool schema_version — all ValueError, never
        # AttributeError/KeyError/TypeError
        rootlist = os.path.join(td, "rootlist.json")
        with open(rootlist, "w") as f:
            json.dump([], f)
        badrec = os.path.join(td, "badrec.json")
        with open(badrec, "w") as f:
            json.dump({"schema_version": 1,
                       "claims": [{"type": "add"}]}, f)     # no idx
        boolver = os.path.join(td, "boolver.json")
        with open(boolver, "w") as f:
            json.dump({"schema_version": True,
                       "claims": [{"idx": 0, "type": "add",
                                   "params": {}}]}, f)
        for path in (empty, future, rootlist, badrec, boolver):
            try:
                Manifest.load(path)
                raise AssertionError(f"loaded {path}")
            except ValueError:
                pass
        # the CLI boundary turns these (and a missing file) into clean
        # exit-2 errors, not tracebacks
        for path in (empty, future, rootlist, badrec,
                     os.path.join(td, "nope.json")):
            try:
                with contextlib.redirect_stderr(io.StringIO()):
                    cli.main(["predict", path])
                raise AssertionError(f"predict accepted {path}")
            except SystemExit as e:
                assert e.code == 2, (path, e.code)


def test_consumers():
    man = _build_manifest()
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "fake_man.json")
        man.save(path)
        m2 = Manifest.load(path)
    mp = MachineProfile.load("gb10-spark")
    rep = predict.report(m2, mp)
    assert "workload totals" in rep
    # extracted manifests itemize every slot: rows are exact, no approx label
    assert "(approx" not in rep
    assert predict.live_set_peak(m2) is not None
    assert dag.build(m2)["summary"]["n_claims"] == len(man.claims)
    assert "layers x2" in partition.report(m2, "layers", 2, mp)
    # combine reduction traffic: e0 remote -> exactly one 8-slot partial (64 B)
    e0_idx = m2.var_by_name()["e0_L1#20"].producer
    assign = [1] * len(m2.claims)
    assign[e0_idx] = 0
    ev = partition.evaluate(m2, assign, 2, mp)
    assert ev["red_bytes_per_sweep"] == 64, ev["red_bytes_per_sweep"]
    # logup settlement reduction: the remote shard sends ONE summed
    # multiplicity partial of T_LEN slots (32,768 B) plus ONE summed z
    # field element (8 B) — z vectors themselves must NOT ship (they are
    # output-sized; shipping them whole was the ~1000x traffic artifact
    # on the first extracted-manifest scorecard, 2026-08-19)
    settle_idx = next(c.idx for c in m2.claims if c.type == "TableSettlement")
    assign2 = [0] * len(m2.claims)
    assign2[settle_idx] = 1
    ev2 = partition.evaluate(m2, assign2, 2, mp)
    assert ev2["mult_bytes_per_sweep"] == T_LEN * 8 + 8, \
        ev2["mult_bytes_per_sweep"]
    # with only the settlement remote, every other edge is co-located and
    # its z inputs are reduction-handled: zero activation traffic
    assert ev2["act_bytes_per_sweep"] == 0, ev2["act_bytes_per_sweep"]
    assert ev2["traffic_per_sweep"] >= T_LEN * 8
    # slot accounting: one-shard partition covers every slot totals() counts
    tot = predict.totals(m2)
    ev1 = partition.evaluate(m2, [0] * len(m2.claims), 1, mp)
    assert ev1["mult_bytes_per_sweep"] == 0    # co-located: nothing ships
    inputs = sum(v.length for v in m2.variables
                 if v.producer is None and not v.persistent)
    weights = sum(v.length for v in m2.variables if v.persistent)
    assert abs(ev1["shard_W"][0] + inputs + weights - tot.W) < 1e-6


def test_enrolled_weights():
    # Enrollment (kept-trees, main): weights leave the per-proof ENCODE but
    # pay qlin (1.0*A) + open (0.5*A) passes per proof. Net vs legacy
    # (weights inside A*W): +0.5*A*W_weights on the floor — enrollment buys
    # statefulness and zero re-encode, not per-proof floor time.
    man = _build_manifest()
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "fake_man.json")
        man.save(path)
        m2 = Manifest.load(path)
    mp = MachineProfile.load("gb10-spark")
    r = predict.report(m2, mp, enrolled_weights=True)
    assert "enrolled block qlin+open" in r
    assert "N/A under enrollment" in r          # aggregate doesn't transfer
    assert "fresh only" in r
    A = mp.get("prove_constants", "A_ns_per_slot")
    weights = sum(v.length for v in m2.variables if v.persistent)
    one = [0] * len(m2.claims)
    ev_leg = partition.evaluate(m2, one, 1, mp)
    ev_enr = partition.evaluate(m2, one, 1, mp, enrolled_weights=True)
    want = (predict.ENROLLED_QLIN_RATIO + predict.ENROLLED_OPEN_RATIO - 1.0) \
        * A * weights * 1e-9
    got = ev_enr["shard_t"][0] - ev_leg["shard_t"][0]
    assert abs(got - want) < 1e-12, (got, want)


def test_projected_protocol():
    # Formulas from the compile functions (prover/routed_projected.py,
    # prover/rescale_claim.py), regression-locked to the exact block
    # ledger of analysis/routed_projected_4h_model.py.
    import synth
    T, K, J, E = 7, 11, 13, 5
    W, cids, Q = claimcosts.cost("RoutedProjectedMatmulClaim",
                                 dict(T=T, K=K, J=J, E=E))
    assert W == T * J + E * K + 2 * T * K + T + 3 * E
    assert cids == E * K + 2 * T + 2 * E + 1
    assert Q == T * K + E
    assert claimcosts.cost("RescaleClaim", dict(length=100)) == \
        (500.0, 200.0, 200.0)

    m = synth.BUILDERS["maverick-projected"](1000)
    rp = [c for c in m.claims if c.type == "routed_projected"]
    rs = [c for c in m.claims if c.type == "rescale_claim"]
    assert len(rp) == 72 and len(rs) == 72       # 24 MoE layers x gate/up/down
    trip = [claimcosts.cost(c.type, c.params) for c in rp]
    # Ed's ledger, exact: L_ROUTE = R + 2*72*S + 72*(2E+1),
    # Q_ROUTE = N + 72*E with N = 24*S*(2d+f), R = 24*E*(2d+f)
    N, R = 442_368_000, 56_623_104
    assert sum(t[1] for t in trip) == 56_785_608          # L_ROUTE
    assert sum(t[2] for t in trip) == 442_377_216         # Q_ROUTE
    # raw W: Q+H (2N) + P (R) + yr (72*S) + f-vectors (72*3E) + Y (T*J sums)
    y_slots = 24 * 1000 * (8192 + 8192 + 5120)
    assert sum(t[0] for t in trip) == \
        2 * N + R + 72 * 1000 + 72 * 3 * 128 + y_slots
    # rescale side: 5/2/2 per selected element; the 2-cids/2-Q sums are
    # exactly the ledger's LQ_SELECTED_OLD
    rst = [claimcosts.cost(c.type, c.params) for c in rs]
    assert sum(c.params["length"] for c in rs) == y_slots == 516_096_000
    assert sum(t[1] for t in rst) == sum(t[2] for t in rst) == 1_032_192_000
    # the projected builder drops the three per-layer expert combines but
    # keeps s_rep: 24 freivalds_combine claims, not 96
    assert sum(1 for c in m.claims
               if c.type == "freivalds_combine") == 24
    # per-expert weight vars are the enrolled block: unchanged total
    mv = synth.BUILDERS["maverick"](1000)
    pw = lambda man: sum(v.length for v in man.variables if v.persistent)  # noqa: E731
    assert pw(m) == pw(mv) == 402_724_618_240


def test_cli_validation():
    bad = [
        ["predict", "x.json", "--gpus", "0"],
        ["predict", "x.json", "--gpus", str(10 ** 400)],
        ["predict", "x.json", "--bandwidth-ratio", "nan"],
        ["predict", "x.json", "--bandwidth-ratio", "inf"],
        ["synth", "--model", "maverick", "--seq", "8", "--layers", "2", "-o", "x"],
        ["partition", "x.json", "--shards", "0"],
        ["partition", "x.json", "--shards", "2", "--bandwidths", "100,-5"],
    ]
    for argv in bad:
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                cli.main(argv)
            raise AssertionError(f"accepted: {argv}")
        except SystemExit as e:
            assert e.code == 2, (argv, e.code)
    # ints past float range would OverflowError in every downstream float
    # accumulator — rejected at the boundary; float-range ints stay legal
    try:
        cli._posint(str(10 ** 400))
        raise AssertionError("accepted 10**400")
    except argparse.ArgumentTypeError:
        pass
    assert cli._posint(str(10 ** 300)) == 10 ** 300


def test_rows_approx_label():
    # a formula-only manifest (no itemized variables) row-packs aux slots
    # at W/ELL — the report must say so
    m = Manifest(model=dict(name="t"), run=dict(seq=1))
    m.claims.append(ClaimRecord(idx=0, type="add", params={"L": 100}))
    mp = MachineProfile.load("gb10-spark")
    assert "(approx" in predict.report(m, mp)


def main():
    test_extractor()
    test_explicit_settlement_reused()
    test_core_isolation()
    test_costs()
    test_mode_flags_extracted()
    test_expert_labels()
    test_manifest_validation()
    test_consumers()
    test_cli_validation()
    test_rows_approx_label()
    test_enrolled_weights()
    test_projected_protocol()
    print("profiler regression tests OK (no torch needed)")


if __name__ == "__main__":
    main()
