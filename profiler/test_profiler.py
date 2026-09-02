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


@dataclass
class RoutedProjectedMatmulClaim:
    # real spellings per prover/routed_projected.py: W is a LIST of
    # per-expert Variables and is NOT in the deferred input list (the
    # claim is in core.STREAMING_INPUT_CLAIMS; shards stream one at a
    # time); f_y/f_u/f_p are phase 3 (the conditional R3 sweep)
    X: Variable; Y: Variable; M: Variable
    W: List[Variable]
    Pj: Variable; Qm: Variable; Hd: Variable; yr: Variable
    f_y: Variable; f_u: Variable; f_p: Variable
    T: int; K: int; J: int; E: int


@dataclass
class RescaleClaim:
    x_full: Variable; x: Variable; x_low: Variable; x_shifted: Variable
    z_low: Variable; z_shifted: Variable
    range_rescale: Table; range_output: Table
    length: int; rescale_bits: int
    output_width: int = 24


class FakeTape:
    def __init__(self, cfg, lazy=True):
        self.cfg, self.lazy = cfg, lazy
        self.claims, self._deferred = [], []
        self.inputs = {}                 # Variable -> tensor | lazy loader

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
    # a lazy loader carrying source provenance (extract copies it onto the
    # persistent VariableRecord; see prover/loader.py gguf_provenance)
    def _w0_loader():
        raise AssertionError("extraction must not resolve weight loaders")
    _w0_loader.provenance = {"quant": "Q6_K",
                             "packed_bytes": D * D * 210 // 256}
    t.inputs[w0] = _w0_loader
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
    # lazy-loader source provenance is copied onto the persistent record;
    # a persistent var with no loader (token_embd: eager table) records
    # nothing and the profiler falls back to its quant table
    assert by_name["W_Q_L0"].quant == "Q6_K"
    assert by_name["W_Q_L0"].packed_bytes == float(D * D * 210 // 256)
    assert by_name["token_embd"].quant is None
    assert by_name["token_embd"].packed_bytes is None
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


def test_manifest_gz_roundtrip():
    # save gzips on a .gz suffix, matching load (an archive saved as
    # x.json.gz used to be plain JSON that load then failed to gunzip)
    man = _build_manifest()
    with tempfile.TemporaryDirectory() as td:
        gz = os.path.join(td, "m.json.gz")
        man.save(gz)
        import gzip
        with gzip.open(gz, "rt") as f:
            assert '"claims"' in f.read()          # gunzips: really gzipped
        m2 = Manifest.load(gz)
        assert len(m2.claims) == len(man.claims)
        plain = os.path.join(td, "m.json")
        man.save(plain)
        with open(plain) as f:
            assert f.read(1) == "{"


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
        # gzipped archives load transparently (analysis/blackwell-session-1)
        import gzip
        with open(path, "rb") as src, gzip.open(path + ".gz", "wb") as dst:
            dst.write(src.read())
        assert len(Manifest.load(path + ".gz").claims) == len(m2.claims)
    mp = MachineProfile.load("gb10-spark")
    rep = predict.report(m2, mp)
    assert "workload totals" in rep
    # production transport is u64le/base64 (11 B/value); legacy decimal
    # JSON (21.4 B/value) stays as the archive-validated reference
    assert "u64le/base64, production" in rep
    assert "legacy decimal JSON" in rep
    assert "A100 reference" in rep     # gb10 has no compact measurement
    # the reference must be the page-cache-excluded egress bound, never the
    # retracted 751 MB/s (analysis/routed-projected-status.md: "not a
    # measurement"; egress bench 200-315 MB/s, final run 245)
    assert 200 <= predict.PROOF_COMPACT_REF_MBPS <= 315
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
    # the z inputs cost nothing per settlement (their balance is a linear
    # constraint the owning shard folds); only the mult partial crosses,
    # per sweep — and the fold partials merge ONCE per proof
    assert ev2["mult_bytes_per_sweep"] == T_LEN * 8, ev2["mult_bytes_per_sweep"]
    k_deg = m2.run["ligero"].get("K_DEG", 16384)
    assert ev2["fold_merge_bytes"] == 1 * 3 * k_deg * 8
    # with only the settlement remote, every other edge is co-located and
    # its z inputs are reduction-handled: zero activation traffic
    assert ev2["act_bytes_per_sweep"] == 0, ev2["act_bytes_per_sweep"]
    assert ev2["traffic_per_sweep"] >= T_LEN * 8
    # slot accounting: one-shard partition covers every slot totals() counts
    tot = predict.totals(m2)
    ev1 = partition.evaluate(m2, [0] * len(m2.claims), 1, mp)
    assert ev1["mult_bytes_per_sweep"] == 0    # co-located: nothing ships
    assert ev1["fold_merge_bytes"] == 0
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
    assert "open (fresh rows)" in r
    assert "enrollment lifecycle" in r
    assert "N/A under enrollment" in r          # aggregate doesn't transfer
    assert "fresh only" in r
    # legacy mode states its opening omission instead
    assert "legacy floor excludes column-opening" in \
        predict.report(m2, mp)
    # copied partition output is self-describing in both non-default modes
    r_enr = partition.report(m2, "layers", 2, mp, enrolled_weights=True)
    assert "ENROLLED" in r_enr and "refresh after" in r_enr
    r_diag = partition.report(m2, "layers", 2, mp, skip_weight_commit=True)
    assert "DIAGNOSTIC" in r_diag
    c_enr = partition.compare(m2, 2, mp, enrolled_weights=True)
    assert "ENROLLED" in c_enr and "refresh after" in c_enr
    assert "DIAGNOSTIC" in partition.compare(m2, 2, mp,
                                             skip_weight_commit=True)
    # contradictory modes are rejected at the LIBRARY boundary, not just
    # the CLI — all three public entry points, one validator
    for fn in (lambda: partition.evaluate(m2, [0] * len(m2.claims), 1, mp,
                                          enrolled_weights=True,
                                          skip_weight_commit=True),
               lambda: partition.report(m2, "layers", 2, mp,
                                        enrolled_weights=True,
                                        skip_weight_commit=True),
               lambda: partition.compare(m2, 2, mp,
                                         enrolled_weights=True,
                                         skip_weight_commit=True)):
        try:
            fn()
            raise AssertionError("contradictory modes accepted")
        except ValueError as e:
            assert "mutually exclusive" in str(e)
    A = mp.get("prove_constants", "A_ns_per_slot")
    weights = sum(v.length for v in m2.variables if v.persistent)
    one = [0] * len(m2.claims)
    ev_leg = partition.evaluate(m2, one, 1, mp)
    ev_enr = partition.evaluate(m2, one, 1, mp, enrolled_weights=True)
    inputs = sum(v.length for v in m2.variables
                 if v.producer is None and not v.persistent)
    want = ((predict.ENROLLED_QLIN_RATIO + predict.ENROLLED_OPEN_RATIO - 1.0)
            * A * weights
            + predict.ENROLLED_OPEN_RATIO * A
            * (ev_leg["shard_W"][0] + inputs)) * 1e-9
    got = ev_enr["shard_t"][0] - ev_leg["shard_t"][0]
    assert abs(got - want) < 1e-12, (got, want)
    # predict's printed floor equals partition's N=1 wall in BOTH modes
    # (the two tools price the same slots with the same constants — the
    # agreement was asserted in commit messages, now locked)
    import re as _re
    def _floor_of(report_text):
        m_ = _re.search(r"floor \(NTT-bound, post-reorg target\):\s+([\d,.]+) s", report_text)
        assert m_, report_text
        return float(m_.group(1).replace(",", ""))
    for enr, ev in ((False, ev_leg), (True, ev_enr)):
        pf = _floor_of(predict.report(m2, mp, enrolled_weights=enr))
        # _fmt_s prints 1 decimal below 60 s and whole seconds above
        tol = 0.06 if ev["shard_t"][0] < 60 else 0.6
        assert abs(pf - ev["shard_t"][0]) <= tol, (enr, pf, ev["shard_t"][0])
    # no refresh budget at a toy geometry is said, not divided through
    tiny = Manifest(run=dict(seq=1, ligero=dict(ELL=64, K_DEG=128, T_QUERIES=80)),
                    claims=m2.claims, variables=m2.variables)
    assert "NO refresh budget" in partition._mode_suffix(tiny, True, False)
    # CLI: the two modes are mutually exclusive at the command line too
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "m.json")
        m2.save(path)
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                cli.main(["partition", path, "--shards", "2",
                          "--enrolled-weights", "--skip-weight-commit"])
            raise AssertionError("CLI accepted both modes")
        except SystemExit as e:
            assert e.code == 2


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
    # ...and against the ledger MODULE itself (independent of the numbers
    # typed above): W_ROUTE charges yr and the three f-vectors at full ELL
    # rows (one variable each); synth counts their logical lengths, so the
    # two agree once that padding is swapped for S and E per matmul
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_rp4h", os.path.join(os.path.dirname(__file__), "..", "analysis",
                              "routed_projected_4h_model.py"))
    rp4h = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rp4h)
    ledger_logical = (rp4h.W_ROUTE - 4 * rp4h.N_MATS * rp4h.ELL
                      + rp4h.N_MATS * rp4h.S + 3 * rp4h.N_MATS * rp4h.E)
    assert sum(t[0] for t in trip) - y_slots == ledger_logical
    assert sum(t[1] for t in trip) == rp4h.L_ROUTE
    assert sum(t[2] for t in trip) == rp4h.Q_ROUTE
    # LinComb: no own witness, one cid per slot (lincomb_compile), no quads
    assert claimcosts.cost("LinCombClaim", dict(length=40)) == (0.0, 40.0, 0.0)
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
    # routed tapes take FIVE streaming sweeps (conditional R3 commitment
    # for phase-3 late aux); classic tapes stay at four
    assert partition.n_sweeps(m) == 5
    assert partition.n_sweeps(mv) == 4
    mp = MachineProfile.load("gb10-spark")
    assert partition.evaluate(
        m, [0] * len(m.claims), 1, mp)["sweeps"] == 5
    # extracted routed claims inherit the FIRST weight shard's name
    # (rp[..@L1_Wg0..]) — expert parsing must NOT send them (or their
    # rescale/silu descendants) to expert 0; only per-expert matmuls parse
    from manifest import VariableRecord
    m3 = Manifest(
        run=dict(seq=4, ligero=dict(ELL=8192)),
        claims=[
            ClaimRecord(idx=0, type="RoutedProjectedMatmulClaim",
                        label="x_r@L0_Wg0#1", layer=0,
                        params=dict(T=4, K=8, J=8, E=4)),
            ClaimRecord(idx=1, type="RescaleClaim",
                        label="x_r@L0_Wg0#1_rs", layer=0,
                        params=dict(length=32)),
            ClaimRecord(idx=2, type="SiluClaim",
                        label="x_r@L0_Wg0#1_rs@silu#2", layer=0,
                        params=dict(L=32)),
            ClaimRecord(idx=3, type="MatmulClaim",
                        label="h@L1_Wd3#4", layer=1,
                        inputs=["h", "L1_Wd3"],
                        params=dict(m=4, k=8, n=8)),
            # reviewer repro: an ATTENTION matmul whose activation
            # ancestry passed through a routed claim — output label
            # carries _Wg0, but its weight input is W_Q: backbone
            ClaimRecord(idx=4, type="MatmulClaim",
                        label="rp[x@L2_Wg0..]_rs@L3_W_Q#5", layer=3,
                        inputs=["a", "W_Q_L3"],
                        params=dict(m=4, k=8, n=8)),
        ],
        variables=[VariableRecord(name="v", length=8),
                   VariableRecord(name="h", length=8),
                   VariableRecord(name="a", length=8),
                   VariableRecord(name="L1_Wd3", length=8, persistent=True),
                   VariableRecord(name="W_Q_L3", length=8, persistent=True)])
    a4 = partition.assign_experts(m3, 4)
    bb = partition.assign_layers(m3, 4)
    assert a4[:3] == bb[:3], (a4, bb)     # routed family: backbone
    assert a4[3] == 3                      # per-expert matmul: expert 3
    # DISCRIMINATING case: a routed family labelled _Wg1 on layer 0 — the
    # old label parser sent it to expert 1 -> shard 1, the type-aware rule
    # keeps it on the backbone (layer 0 -> shard 0); expert 0 above could
    # not tell the two apart (0 % 4 == backbone 0)
    m3b = Manifest(
        run=dict(seq=4, ligero=dict(ELL=8192)),
        claims=[ClaimRecord(idx=0, type="RoutedProjectedMatmulClaim",
                            label="x_r@L0_Wg1#1", layer=0,
                            params=dict(T=4, K=8, J=8, E=4)),
                ClaimRecord(idx=1, type="RescaleClaim",
                            label="x_r@L0_Wg1#1_rs", layer=0,
                            params=dict(length=32))],
        variables=[VariableRecord(name="v", length=8)])
    assert partition.assign_experts(m3b, 4) == partition.assign_layers(m3b, 4) == [0, 0]
    # routed-only manifest: NO expert labels under the type-aware rule (the
    # label-based version read _Wg1 and said False)
    assert partition._no_expert_labels(m3b)
    assert a4[4] == bb[4] == 3, (a4, bb)  # attention matmul: backbone (L3)
    # type-aware note: nothing here is expert-assignable except idx 3
    assert not partition._no_expert_labels(m3)


def test_projected_extraction():
    # The extraction walker has never met the projected claims on real
    # hardware; this locks the structural contract first: list-valued W
    # fields traverse, omitted-from-deferred persistent shards classify
    # as run inputs, phase-3 aux records as phase 3 (driving n_sweeps on
    # EXTRACTED manifests), and w_slots equals the compile-derived
    # formulas exactly.
    T_, K_, J_, E_ = 4, 6, 5, 3
    t = FakeTape(Cfg())
    x = Variable("xr_L1", T_ * K_)
    mask = Variable("rt1_m", T_ * E_)
    shards = [Variable(f"L1_Wg{e}", K_ * J_, persistent=True)
              for e in range(E_)]
    name = f"rp[{x.name}@{shards[0].name}..]"
    Y = Variable(name, T_ * J_)
    rp = RoutedProjectedMatmulClaim(
        X=x, Y=Y, M=mask, W=shards,
        Pj=Variable(name + ".P", E_ * K_, phase=2),
        Qm=Variable(name + ".Q", T_ * K_, phase=2),
        Hd=Variable(name + ".H", T_ * K_, phase=2),
        yr=Variable(name + ".yr", T_, phase=2),
        f_y=Variable(name + ".f_y", E_, phase=3),
        f_u=Variable(name + ".f_u", E_, phase=3),
        f_p=Variable(name + ".f_p", E_, phase=3),
        T=T_, K=K_, J=J_, E=E_)
    t.add(rp, [x, mask])          # W shards deliberately NOT deferred inputs
    # provenance-carrying lazy loaders on the shards (as maverick_lazy_expert
    # attaches); the walker must copy quant/packed_bytes onto the records
    # without calling the loaders
    for e, sh in enumerate(shards):
        def _ld(e=e):
            raise AssertionError(f"extraction resolved shard {e}")
        _ld.provenance = {"quant": "Q4_K", "packed_bytes": 144 * (K_ * J_ // 256 + e)}
        t.inputs[sh] = _ld
    L_ = T_ * J_
    zl = Variable(name + "_rs_zlow", L_, phase=2)
    zs = Variable(name + "_rs_zshift", L_, phase=2)
    tb_r = Table("rescale_w12", Variable("rescale_w12_mult", 1 << 4),
                 Variable("rescale_w12_w", 1 << 4), [zl])
    tb_o = Table("output_w24", Variable("output_w24_mult", 1 << 5),
                 Variable("output_w24_w", 1 << 5), [zs])
    rs = RescaleClaim(
        x_full=Y, x=Variable(name + "_rs", L_),
        x_low=Variable(name + "_rs_low", L_),
        x_shifted=Variable(name + "_rs_shift", L_),
        z_low=zl, z_shifted=zs,
        range_rescale=tb_r, range_output=tb_o,
        length=L_, rescale_bits=12, output_width=24)
    t.add(rs, [Y])
    with _fake_core():
        man = extract_tape(t, model=dict(name="projected-stub"), seq=T_)
    by = man.var_by_name()
    c_rp, c_rs = man.claims[0], man.claims[1]
    # params: scalars only — the W list must NOT leak into params
    assert c_rp.type == "RoutedProjectedMatmulClaim"
    assert c_rp.params == dict(T=T_, K=K_, J=J_, E=E_), c_rp.params
    # shards: extra inputs via the persistent rule, producer-less,
    # consumed by the routed claim
    for e in range(E_):
        nm = f"L1_Wg{e}"
        assert nm in c_rp.inputs
        v = by[nm]
        assert v.persistent and v.producer is None and 0 in v.consumers
        assert v.quant == "Q4_K" and v.packed_bytes == 144 * (K_ * J_ // 256 + e)
        assert v.w_new is False
    # outputs and exact W agree with the compile-derived formula
    assert c_rp.w_slots ==         claimcosts.cost("RoutedProjectedMatmulClaim", c_rp.params)[0]
    assert by[name + ".f_y"].phase == 3
    assert partition.n_sweeps(man) == 5      # extracted-phase detection
    # the routed claim itself rides the backbone despite its _Wg0 label
    assert partition._expert_of_claim(c_rp, by) is None
    # rescale: params, exact W = 5L, and the range tables settle
    assert c_rs.params["length"] == L_ and c_rs.params["rescale_bits"] == 12
    assert c_rs.w_slots == 5 * L_ ==         claimcosts.cost("RescaleClaim", c_rs.params)[0]
    settles = [c for c in man.claims if c.type == "TableSettlement"]
    assert len(settles) == 2
    z_consumers = set(by[name + "_rs_zlow"].consumers)
    assert any(c.idx in z_consumers for c in settles)


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


def _hetero_manifest(n_vars=60, ELL=8, T_Q=4, seed=7, fresh=0):
    # Many heterogeneous enrolled variables (mixed lengths, several
    # non-ELL-aligned, mixed quant types) with no claims: exercises the
    # interior-worker geometry, padding, and provenance paths that the
    # two-weight fake manifest cannot.
    import random
    from manifest import VariableRecord
    rng = random.Random(seed)
    m = Manifest()
    m.run = {"ligero": {"ELL": ELL, "T_QUERIES": T_Q}}
    quants = ["Q4_K", "Q6_K", "Q5_K", None]
    for i in range(n_vars):
        length = rng.choice([ELL, 2 * ELL, 3 * ELL, 5 * ELL, ELL + 1, 2 * ELL - 3, 9])
        m.variables.append(VariableRecord(name=f"w{i}", length=length, persistent=True,
                                          quant=rng.choice(quants)))
    m.variables.append(VariableRecord(name="x", length=ELL))    # a fresh input
    if fresh:   # substantial coordinator-only fresh work -> asymmetric optimum
        m.variables.append(VariableRecord(name="x_big", length=fresh))
    return m


def test_weightsplit():
    # Stage-aware weight-ownership model (the M1 coordinator/worker
    # architecture) from EXECUTABLE plans: wall = commit + max(fold) +
    # max(open) across the s_col barrier; N=1 reproduces predict's enrolled
    # floor exactly on an aligned manifest; plans are contiguous whole-
    # variable runs with cuts solved EXACTLY (matches brute force); slots
    # are physical (row-padded); holds are unions (interior workers do not
    # nest); shared-volume streaming is aggregate-bandwidth bound; two
    # metrics (kernel floor ratio, same-mode speedup); UNAVAILABLE when the
    # profile lacks disk (streaming) or mem_GB (resident); invalid
    # provenance fails; zero shares valid; HBM-constrained optimum found
    # even when the feasible band is narrower than any fraction grid.
    import weightsplit as ws
    import cli
    man = _build_manifest()
    for v in man.variables:
        if v.persistent:
            v.quant = "Q6_K"
    pv0 = [v for v in man.variables if v.persistent]
    pv0[0].packed_bytes = 1.0          # exact size beats the quant table
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "fake_man.json")
        man.save(path)
        m2 = Manifest.load(path)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cli.main(["weightsplit", path, "--machine", "gb10-spark",
                      "--gpus", "1", "2", "--resident", "--intervals", "2",
                      "--x-fold", "0", "--x-open", "0.5"])
        out = buf.getvalue()
        assert "fold stage (x=0.000)" in out and "open stage (x=" in out
        assert "encode-share sensitivity" in out and "same-mode" in out
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cli.main(["weightsplit", path, "--machine", "gb10-spark", "--gpus", "2"])
        assert "UNAVAILABLE" in buf.getvalue()      # no disk calibration
    pv = [v for v in m2.variables if v.persistent]
    assert pv[0].packed_bytes == 1.0 and all(v.quant == "Q6_K" for v in pv)
    assert ws.packed_bytes_of(pv[0], None) == 1.0
    assert ws.packed_bytes_of(pv[1], None) == pv[1].length * ws.QUANT_BYTES_PER_PARAM["Q6_K"]
    assert ws.packed_bytes_of(pv[1], 0.7) == pv[1].length * 0.7
    # invalid provenance fails loudly instead of sizing HBM wrong
    from manifest import VariableRecord
    for bad in (dict(packed_bytes=-1.0), dict(packed_bytes=float("nan")),
                dict(quant="Q4_0")):
        try:
            ws.packed_bytes_of(VariableRecord(name="b", length=8, persistent=True, **bad), None)
            raise AssertionError(f"accepted {bad}")
        except ValueError:
            pass
    mp = MachineProfile.load("gb10-spark")
    A = mp.get("prove_constants", "A_ns_per_slot")
    B = mp.get("prove_constants", "B_ns_per_cid")
    C = mp.get("prove_constants", "C_ns_per_product")
    t = predict.totals(m2)
    W_fresh = t.W - t.W_weights
    want = (A * W_fresh + B * t.cids + C * t.Q
            + (predict.ENROLLED_QLIN_RATIO + predict.ENROLLED_OPEN_RATIO)
            * A * t.W_weights
            + predict.ENROLLED_OPEN_RATIO * A * W_fresh) * 1e-9
    st = ws.stages(m2, mp)
    assert abs(st.floor - want) < 1e-12, (st.floor, want)      # aligned fixture
    ev1 = ws.evaluate(m2, mp, 1, resident=True)
    assert abs(ev1["wall"] - st.floor) < 1e-12 and ev1["kernel_floor_ratio"] == 1.0
    assert ev1["aligned"] and ev1["same_mode_speedup"] == 1.0
    evx = ws.evaluate(m2, mp, 2, x_fold=1.0, x_open=1.0, resident=True)
    assert abs(evx["wall"] - st.floor) < 1e-12 and evx["plan_mode"] == "explicit"
    # resident needs mem_GB
    nomem = MachineProfile({"name": "nomem", "prove_constants": mp.raw["prove_constants"]})
    assert ws.evaluate(m2, nomem, 2, resident=True)["wall"] is None
    assert ws.evaluate(m2, MachineProfile({"name": "x"}), 2)["wall"] is None
    assert "UNAVAILABLE" in ws.report(m2, MachineProfile({"name": "x"}), [1, 2])

    # --- heterogeneous many-variable fixture ---------------------------
    hm = _hetero_manifest()
    hpv = [v for v in hm.variables if v.persistent]
    ELL = 8
    ev = ws.evaluate(hm, mp, 2, resident=True)
    assert not ev["aligned"]
    phys = sum(-(-v.length // ELL) * ELL for v in hpv)
    assert ev["physical_slots"] == phys > ev["logical_slots"] == sum(v.length for v in hpv)
    # timing and payload use physical rows
    assert abs(sum(ev["fold_slots"]) - phys) < 1e-9
    assert abs(sum(ev["open_payload"]) - phys / ELL * 4 * 8) < 1e-9
    # the reviewer's two-weight example: lengths 1 and 9 at ELL=8 -> 3 rows
    tiny = Manifest(run={"ligero": {"ELL": 8, "T_QUERIES": 4}})
    tiny.variables = [VariableRecord(name="a", length=1, persistent=True),
                      VariableRecord(name="b", length=9, persistent=True)]
    e = ws.evaluate(tiny, mp, 1, resident=True)
    assert e["physical_slots"] == 24 and abs(sum(e["open_payload"]) - 96) < 1e-9
    # stage structure from the plans
    assert abs(ev["wall"] - (ws.stages(hm, mp).commit + ev["fold_t"] + ev["open_t"])) < 1e-12
    rate_f = predict.ENROLLED_QLIN_RATIO * A * 1e-9
    assert abs(ev["fold_compute"][1] - ev["fold_slots"][1] * rate_f) < 1e-12
    # exact cut optimum at N=2: brute force over every boundary per stage
    stg = ws.stages(hm, mp)
    blk = ws._Block(hm, None, ELL)
    def brute(stage_fresh, rate):
        best = None
        for c in range(blk.n + 1):
            tt = max(stage_fresh + rate * blk.phys(0, c), rate * blk.phys(c, blk.n))
            best = tt if best is None or tt < best else best
        return best
    rate_o = predict.ENROLLED_OPEN_RATIO * A * 1e-9
    assert abs(ev["fold_t"] - brute(stg.fresh_fold, rate_f)) < 1e-9
    assert abs(ev["open_t"] - brute(stg.fresh_open, rate_o)) < 1e-9
    assert ev["plan_mode"] == "independent"
    # per-stage optimum never loses to tied cuts (static)
    evs = ws.evaluate(hm, mp, 2, resident=True, static=True)
    assert evs["plan_fold"] == evs["plan_open"] and ev["wall"] <= evs["wall"] + 1e-9
    # N=3/4: plans contiguous/exhaustive per stage; holds = brute-force
    # variable-set unions; the solver beats or ties the equal-slot heuristic
    for n in (3, 4):
        e = ws.evaluate(hm, mp, n, x_fold=0.2, x_open=0.7, resident=True)
        for key in ("plan_fold", "plan_open"):
            plan = e[key]
            assert plan[0][0] == 0 and plan[-1][1] == len(hpv)
            assert all(a[1] == b[0] for a, b in zip(plan, plan[1:]))
        for d in range(n):
            (flo, fhi), (olo, ohi) = e["plan_fold"][d], e["plan_open"][d]
            union = set(range(flo, fhi)) | set(range(olo, ohi))
            assert abs(e["hold_bytes"][d]
                       - sum(ws.packed_bytes_of(hpv[i], None) for i in union)) < 1e-6
        solved = ws.evaluate(hm, mp, n, resident=True)
        assert solved["wall"] <= e["wall"] + 1e-9
        # unequal worker runs are allowed (physical slots differ)
        ws_ = solved["fold_slots"][1:]
        assert len(set(ws_)) > 1 or n == 2
    # shared volume: the stage cannot beat aggregate bandwidth whatever
    # the split; per-device disks can; 'none' overlap adds I/O to compute
    blk_bytes = sum(ws.packed_bytes_of(v, None) for v in hpv)
    slow = 1e-9                                  # GB/s -> I/O dominates
    sh = ws.evaluate(hm, mp, 2, disk_GBps=slow, disk_mode="shared", io_overlap="perfect")
    assert sh["fold_t"] >= blk_bytes / (slow * 1e9) - 1e-6
    pd = ws.evaluate(hm, mp, 2, disk_GBps=slow, disk_mode="per-device", io_overlap="perfect")
    assert pd["fold_t"] < sh["fold_t"]
    nn = ws.evaluate(hm, mp, 2, disk_GBps=slow, disk_mode="per-device", io_overlap="none",
                     x_fold=0.5, x_open=0.5)
    pp = ws.evaluate(hm, mp, 2, disk_GBps=slow, disk_mode="per-device", io_overlap="perfect",
                     x_fold=0.5, x_open=0.5)
    assert nn["fold_t"] > pp["fold_t"]
    # two metrics: same-mode speedup uses the N=1 wall under the same
    # storage; kernel-floor ratio uses the compute-only floor
    s2 = ws.evaluate(hm, mp, 2, disk_GBps=1e-6)
    s1 = ws.evaluate(hm, mp, 1, disk_GBps=1e-6)
    assert abs(s2["n1_wall_same_mode"] - s1["wall"]) < 1e-9
    assert abs(s2["same_mode_speedup"] - s1["wall"] / s2["wall"]) < 1e-12
    assert s2["kernel_floor_ratio"] < s2["same_mode_speedup"]
    nodisk = MachineProfile({"name": "nodisk", "gpu": {"mem_GB": 100},
                             "prove_constants": mp.raw["prove_constants"]})
    assert ws.evaluate(hm, nodisk, 2)["wall"] is None
    assert ws.evaluate(hm, nodisk, 2, disk_GBps=1.0)["wall"] is not None
    # HBM-constrained optimum: a cap just above the theoretical minimum
    # (half the block at N=2, up to one variable) admits only a narrow
    # band of cuts — far narrower than a 0.005 fraction step — and the
    # solver finds it; a cap below any two-run split is infeasible and
    # reported as the least-infeasible (min max-hold) plan
    # (the hetero fixture's fresh work is negligible, so its free optimum
    # already sits at the byte-balanced cut; the projected S=1000 tape on
    # the B200 profile has a genuinely asymmetric optimum — worker ~2/3 of
    # the block — and a one-shard-wide feasible band at the cap)
    import synth
    mproj = synth.BUILDERS["maverick-projected"](1000)
    mpb = MachineProfile.load("b200-runpod")
    pblk = ws._Block(mproj, None, mproj.run["ligero"]["ELL"])
    free = ws.evaluate(mproj, mpb, 2, resident=True, workspace_GB=0.0)
    bmin = min(max(pblk.bytes(0, c), pblk.bytes(c, pblk.n)) for c in range(pblk.n + 1))
    assert free["feasible"] and max(free["hold_bytes"]) > 1.2 * bmin
    cap_GB = bmin * (1 + 1e-9) / ws.MEM_GB_BYTES        # profile mem_GB is GiB
    assert (max(free["hold_bytes"]) - bmin) / pblk.total_bytes > 0.05     # far from the cap
    assert pblk.bytes(0, 1) / pblk.total_bytes < 0.005                    # band < a grid step
    tight = MachineProfile({"name": "tight", "gpu": {"mem_GB": cap_GB},
                            "prove_constants": mpb.raw["prove_constants"]})
    con = ws.evaluate(mproj, tight, 2, resident=True, workspace_GB=0.0)
    assert con["feasible"] and max(con["hold_bytes"]) <= cap_GB * ws.MEM_GB_BYTES
    assert con["wall"] >= free["wall"] - 1e-9 and con["plan_mode"] == "capped-exact"
    # resident same-mode needs an executable N=1: the whole block does not
    # fit one B200, so it is n/a (floor ratio still reported)
    assert free["same_mode_speedup"] is None and free["n1_wall_same_mode"] is None
    assert free["kernel_floor_ratio"] > 1.0 and "n/a" in ws.report(
        mproj, mpb, [2], resident=True, workspace_GB=0.0)
    # --static honours a binding cap and equals the exhaustive capped
    # single-cut optimum of the true wall (0.7 B/param: the speed-optimal
    # plan does not fit under the B200's 178 GiB - 10 GB)
    sta = ws.evaluate(mproj, mpb, 2, resident=True, static=True, bytes_per_param=0.7)
    assert sta["feasible"] and sta["plan_mode"] == "static-exact"
    cap7 = mpb.get("gpu", "mem_GB") * ws.MEM_GB_BYTES - 10.0 * 1e9
    assert sta["cap_bytes"] == cap7
    blk7 = ws._Block(mproj, 0.7, mproj.run["ligero"]["ELL"])
    st7 = ws.stages(mproj, mpb, bytes_per_param=0.7)
    Ab = mpb.get("prove_constants", "A_ns_per_slot") * 1e-9
    best = None
    for c in range(blk7.n + 1):
        if max(blk7.bytes(0, c), blk7.bytes(c, blk7.n)) > cap7:
            continue
        f = max(st7.fresh_fold + Ab * blk7.phys(0, c), Ab * blk7.phys(c, blk7.n))
        o = max(st7.fresh_open + 0.5 * Ab * blk7.phys(0, c), 0.5 * Ab * blk7.phys(c, blk7.n))
        w = st7.commit + f + o
        best = w if best is None or w < best else best
    assert abs(sta["wall"] - best) < 1e-9 and max(sta["hold_bytes"]) <= cap7
    # N>=3 tied/static plans are labelled heuristic
    assert ws.evaluate(hm, mp, 3, resident=True, static=True)["plan_mode"] == "static-heuristic"
    # capped N=2 is EXACT over UNTIED plans: on a hetero fixture with
    # substantial coordinator-only fresh work the free optimum is
    # asymmetric (the worker holds well over half); a cap between the
    # balanced split and that hold binds. Brute force every (c_fold,
    # c_open) pair under the union-hold cap: the model's wall equals the
    # brute-force optimum and beats or ties the best TIED plan.
    hf = _hetero_manifest(fresh=200 * ELL)
    blkf = ws._Block(hf, None, ELL)
    stg_h = ws.stages(hf, mp)
    rf, ro = predict.ENROLLED_QLIN_RATIO * A * 1e-9, predict.ENROLLED_OPEN_RATIO * A * 1e-9
    def _wall(cf, co):
        f = max(stg_h.fresh_fold + rf * blkf.phys(0, cf), rf * blkf.phys(cf, blkf.n))
        o = max(stg_h.fresh_open + ro * blkf.phys(0, co), ro * blkf.phys(co, blkf.n))
        return stg_h.commit + f + o
    def _fits(cf, co, cap):
        return (blkf.bytes(0, max(cf, co)) <= cap and blkf.bytes(min(cf, co), blkf.n) <= cap)
    free_f = ws.evaluate(hf, mp, 2, resident=True, workspace_GB=0.0)
    assert free_f["plan_mode"] == "independent"
    half = min(max(blkf.bytes(0, c), blkf.bytes(c, blkf.n)) for c in range(blkf.n + 1))
    assert max(free_f["hold_bytes"]) > 1.15 * half          # genuinely asymmetric
    cap_h = 0.5 * (half + max(free_f["hold_bytes"]))         # binds, box non-empty
    prof_h = MachineProfile({"name": "h", "gpu": {"mem_GB": cap_h / ws.MEM_GB_BYTES},
                             "prove_constants": mp.raw["prove_constants"]})
    capped = ws.evaluate(hf, prof_h, 2, resident=True, workspace_GB=0.0)
    brute = min(_wall(cf, co) for cf in range(blkf.n + 1) for co in range(blkf.n + 1)
                if _fits(cf, co, cap_h))
    tied_best = min(_wall(c, c) for c in range(blkf.n + 1) if _fits(c, c, cap_h))
    assert capped["feasible"] and capped["plan_mode"] == "capped-exact", capped["plan_mode"]
    assert abs(capped["wall"] - brute) < 1e-9, (capped["wall"], brute)
    assert capped["wall"] <= tied_best + 1e-9 and max(capped["hold_bytes"]) <= cap_h
    assert free_f["wall"] <= capped["wall"] + 1e-9
    # --static at the same cap is the tied optimum (and >= the untied one)
    sta_h = ws.evaluate(hf, prof_h, 2, resident=True, static=True, workspace_GB=0.0)
    assert abs(sta_h["wall"] - tied_best) < 1e-9 and sta_h["plan_mode"] == "static-exact"
    # N=3 under a binding cap (same fresh-heavy fixture: the free plan's
    # largest hold is well above the min-max 3-way split): the tied
    # heuristic returns a FEASIBLE plan whose holds respect the cap —
    # optimality is NOT claimed, the label says heuristic
    free3 = ws.evaluate(hf, mp, 3, resident=True, workspace_GB=0.0)
    minhold_plan, _ = ws._solve_stage(blkf, 3, blkf.bytes, blkf.bytes, None)
    minhold = max(blkf.bytes(lo, hi) for lo, hi in minhold_plan)
    assert max(free3["hold_bytes"]) > 1.1 * minhold
    cap3 = 0.5 * (minhold + max(free3["hold_bytes"]))
    prof3 = MachineProfile({"name": "h3", "gpu": {"mem_GB": cap3 / ws.MEM_GB_BYTES},
                            "prove_constants": mp.raw["prove_constants"]})
    e3 = ws.evaluate(hf, prof3, 3, resident=True, workspace_GB=0.0)
    assert e3["feasible"] and e3["plan_mode"] == "tied-heuristic", e3["plan_mode"]
    assert max(e3["hold_bytes"]) <= cap3 and e3["plan_fold"] == e3["plan_open"]
    assert e3["wall"] >= free3["wall"] - 1e-9
    # w_new (a linking proof's refreshed copy) is not the enrolled block:
    # excluded from the plan's variable list, as core._layout excludes it
    hm2 = Manifest(run=hm.run, claims=hm.claims,
                   variables=list(hm.variables) + [VariableRecord(
                       name="Wnew0", length=40, persistent=True, w_new=True)])
    assert ws._Block(hm2, None, ELL).n == ws._Block(hm, None, ELL).n
    # the W-fold-rate reading and the whole-proof line are in the report
    txt = ws.report(hm, mp, [1, 2], resident=True, semantic_s=1000.0)
    assert "W-fold-rate sensitivity" in txt and "whole-proof speedup" in txt
    e_sem = ws.evaluate(hm, mp, 2, resident=True, semantic_s=1000.0)
    assert abs(e_sem["whole_proof_speedup"]
               - (1000.0 + e_sem["n1_wall_same_mode"]) / (1000.0 + e_sem["wall"])) < 1e-12
    # when the N=1 run does not fit (B200 resident), the whole-proof line
    # falls back to the kernel floor as its reference and says so
    e_b = ws.evaluate(mproj, mpb, 2, resident=True, semantic_s=2024.0)
    assert e_b["n1_wall_same_mode"] is None and e_b["whole_proof_ref"] == "kernel-floor"
    assert abs(e_b["whole_proof_speedup"] - (2024.0 + e_b["floor"]) / (2024.0 + e_b["wall"])) < 1e-12
    assert "N=1 reference = kernel floor" in ws.report(mproj, mpb, [2], resident=True,
                                                       semantic_s=2024.0)
    e_wf = ws.evaluate(hm, mp, 2, resident=True, w_fold_ratio=0.5)
    assert e_wf["floor"] < ev["floor"] and e_wf["w_fold_ratio"] == 0.5
    # nothing fits: the least-infeasible (min max-hold) plan is reported
    hmin = min(max(blk.bytes(0, c), blk.bytes(c, blk.n)) for c in range(blk.n + 1))
    none = MachineProfile({"name": "none", "gpu": {"mem_GB": hmin * 0.99 / ws.MEM_GB_BYTES},
                           "prove_constants": mp.raw["prove_constants"]})
    inf = ws.evaluate(hm, none, 2, resident=True, workspace_GB=0.0)
    assert not inf["feasible"] and inf["plan_mode"].startswith("least-infeasible")
    assert abs(max(inf["hold_bytes"]) - hmin) < 1e-6
    assert "NO" in ws.report(hm, none, [2], resident=True, workspace_GB=0.0)
    # mem_GB is GiB (calibrate writes total_memory / 2**30; the b200 profile's
    # 178 is 191.1e9 bytes): the cap converts with 2**30, the workspace stays
    # decimal — a decimal reading (178e9 - 10e9) under-sized the cap 7%
    gib = MachineProfile({"name": "gib", "gpu": {"mem_GB": 178},
                          "prove_constants": mpb.raw["prove_constants"]})
    e178 = ws.evaluate(mproj, gib, 2, resident=True, workspace_GB=10.0)
    assert e178["cap_bytes"] == 178 * 2 ** 30 - 10e9 > (178 - 10) * 1e9
    assert "178 GiB (191.1 GB)" in ws.report(mproj, gib, [2], resident=True)
    # provenance fallback is reported, never silent: the synth manifest has
    # no quant/packed_bytes, so every enrolled byte is a Q4_K guess
    assert e178["default_sized_vars"] == len(pblk.vars) and e178["default_share"] == 1.0
    rep_txt = ws.report(mproj, mpb, [2], resident=True)
    assert "WARNING" in rep_txt and "100% of packed bytes" in rep_txt
    assert "WARNING" not in ws.report(mproj, mpb, [2], resident=True, bytes_per_param=0.7)
    assert "WARNING" not in ws.report(m2, mp, [1, 2], resident=True)  # quant on every var
    assert ws.evaluate(m2, mp, 2, resident=True)["default_share"] == 0.0

def main():
    test_extractor()
    test_explicit_settlement_reused()
    test_core_isolation()
    test_costs()
    test_mode_flags_extracted()
    test_expert_labels()
    test_manifest_gz_roundtrip()
    test_manifest_validation()
    test_consumers()
    test_cli_validation()
    test_rows_approx_label()
    test_enrolled_weights()
    test_projected_protocol()
    test_projected_extraction()
    test_weightsplit()
    print("profiler regression tests OK (no torch needed)")


if __name__ == "__main__":
    main()
