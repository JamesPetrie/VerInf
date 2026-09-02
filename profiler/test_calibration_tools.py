"""No-torch tests for crosscheck.py and calibrate.py (the parts that parse,
diff, and derive — everything that doesn't need a GPU).

    python3 profiler/test_calibration_tools.py   # or pytest

Kept separate from test_profiler.py so the PR-staged suite stays untouched;
fold in on the next suite revision if preferred.
"""
import contextlib
import copy
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import calibrate                                  # noqa: E402
import crosscheck                                 # noqa: E402
import synth                                      # noqa: E402
from manifest import Manifest, ClaimRecord, VariableRecord   # noqa: E402


def _quiet(fn, *args):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = fn(*args)
    return out, buf.getvalue()


# ------------------------------------------------- calibrate parsers

def test_parse_field_mul():
    out = ("device: NVIDIA B200  sm=10.0  SMs=148  totalGlobalMem=179.00 GB\n"
           "config: blocks=256 threads=256 K_LANES=8 iters=4096  warmup=3 runs=5\n"
           "  run 0: 1.234 ms  -> 2210.55 Gmul/s\n"
           "best: 2215.31 Gmul/s\n"
           "sink[0] = deadbeefdeadbeef (anti-DCE)\n")
    assert calibrate.parse_field_mul(out) == 2215.31
    assert calibrate.parse_field_mul("no match here") is None


def test_parse_ntt():
    out = ("device: NVIDIA B200  sm=10.0  SMs=148\n"
           "n=  1024 log2n=10  baseline  fwd+inv x 100  total=  1.000 ms  -> 5.00 us/NTT  1.02 Gbutterfly/s\n"
           "n= 65536 log2n=16  baseline  fwd+inv x 100  total= 80.000 ms  -> 40.00 us/NTT  13.11 Gbutterfly/s\n"
           "n= 65536 log2n=16  fused     fwd+inv x 100  total= 60.000 ms  -> 30.00 us/NTT  17.48 Gbutterfly/s\n"
           "n= 65536 log2n=16  bailey    fwd+inv x 100  total= 47.600 ms  -> 23.80 us/NTT  22.03 Gbutterfly/s\n")
    got = calibrate.parse_ntt(out)
    assert abs(got - 23.80 * 1000 / 65536) < 1e-9      # bailey wins
    # without bailey, fused wins
    got = calibrate.parse_ntt(out.replace("bailey", "ignored"))
    assert abs(got - 30.00 * 1000 / 65536) < 1e-9
    assert calibrate.parse_ntt("nothing") is None


def test_parse_blake3():
    out = ("device: NVIDIA B200  sm=10.0  SMs=148\n"
           "config: columns=65536 blocks=256 threads=256  B_BLOCK=8 (Goldilocks/block)\n"
           "m=    8  cols= 65536  bytes=  4.00 MB  ms=  0.100  -> 655.36 Mcols/s  0.66 Gcompress/s  41.94 GB/s absorbed\n"
           "m=  128  cols= 65536  bytes= 64.00 MB  ms=  0.500  -> 131.07 Mcols/s  13.10 Gcompress/s  838.86 GB/s absorbed\n")
    c, bulk = calibrate.parse_blake3(out)
    assert c == 13.10 and bulk == 838.86
    assert calibrate.parse_blake3("x") == (None, None)


def test_parse_matmul():
    out = ("device: NVIDIA B200 sm=10.0  SMs=148  total mem=179.0 GB\n"
           "matmul sweep: A(16384, 5120) @ B^T(n, 5120) = Y(16384, n)\n"
           "Arithmetic floor at 312 Gmul/s (measured peak from bench_field_mul);\n"
           "n=    1  ops=8.39e+07  time/run=    1.00 ms  throughput=  83.90 Gmul/s  (0.3x peak floor)\n"
           "n= 8192  ops=6.87e+11  time/run=  500.00 ms  throughput=1374.00 Gmul/s  (4.4x peak floor)\n")
    assert calibrate.parse_matmul(out) == 1374.00
    assert calibrate.parse_matmul("nope") is None


def test_parse_blake3_reg():
    out = ("device: NVIDIA B200  sm=10.0  SMs=148\n"
           "config: blocks=256 threads=256 iters=2048  warmup=3 runs=5  "
           "(register-resident chained compress)\n"
           "  run 0: 10.000 ms  -> 13.42 Gcompress/s\n"
           "best: 13.42 Gcompress/s (register-resident)\n"
           "sink[0] = deadbeef (anti-DCE)\n")
    assert calibrate.parse_blake3_reg(out) == 13.42
    # the column bench's line must NOT satisfy the register regex
    assert calibrate.parse_blake3_reg(
        "m=  128  cols= 65536  -> 131.07 Mcols/s  13.10 Gcompress/s") is None


def test_parse_session2_benches():
    out = ("device: NVIDIA B200  sm=10.0  SMs=148\n"
           "n= 65536 m=    1  fwd+inv x 5  total=   10.000 ms  ->  1000.000 us/NTT  15.2588 ns/elem\n"
           "n= 65536 m=  512  fwd+inv x 5  total=  100.000 ms  ->   19.531 us/NTT  0.2980 ns/elem\n"
           "n= 65536 m= 2048  fwd+inv x 5  total=  350.000 ms  ->   17.090 us/NTT  0.2608 ns/elem\n")
    assert calibrate.parse_ntt_batched(out) == 0.2608
    assert calibrate.parse_ntt_batched("nope") is None
    # chase ns/hop is the per-hop wall across all walkers (a throughput
    # figure — 1.84 ns is impossible as a latency), and the bench now says
    # so in the line the parser reads
    out = ("buffer: 8.0 GiB (1073741824 cells)\n"
           "  gather run 0: 500.0 ms -> 429.50 GB/s\n"
           "gather best: 430.11 GB/s (random 8B reads)\n"
           "chase best: 1.84 ns/hop (65536 parallel walkers; per-hop wall "
           "across all walkers — a throughput figure, not the single-access "
           "latency; rerun with --walkers 1 for latency)\n")
    assert calibrate.parse_hbm_random(out) == (430.11, 1.84)
    assert calibrate.parse_hbm_random("x") == (None, None)
    out = ("device: NVIDIA B200  sm=10.0\n"
           "sync: 8.91 us/launch (launch + device sync round trip)\n"
           "stream: 2.140 us/launch (back-to-back enqueue, one sync)\n")
    assert calibrate.parse_launch(out) == (8.91, 2.14)
    assert calibrate.parse_launch("x") == (None, None)


def test_dump_compact_smoke(tmp_path=None):
    import tempfile, pathlib as pl
    with tempfile.TemporaryDirectory() as td:
        rate = calibrate.bench_dump_compact_MBps(pl.Path(td), mb=8)
        assert rate > 0
        assert not list(pl.Path(td).iterdir())    # probe cleaned up


def test_bench_grid_fills_the_part():
    # the chained-ALU benches default to 256 blocks x 256 threads = 13.8
    # warps/SM on a 148-SM B200; calibrate sizes the grid to full occupancy
    # (2048 threads/SM) and keeps the bench default when the SM count is
    # unknown (nvidia-smi fallback)
    assert calibrate.bench_grid(148) == 148 * 8
    assert calibrate.bench_grid(48) == 384
    assert calibrate.bench_grid(None) is None and calibrate.bench_grid(0) is None


def test_synth_builder_chosen_from_tape():
    # a projected tape (RoutedProjectedMatmulClaim present) diffs against
    # maverick-projected; the legacy all-E fan against maverick — diffing
    # a projected tape against the legacy builder flags by construction
    class RoutedProjectedMatmulClaim: pass
    class MatmulClaim: pass
    class RescaleClaim: pass
    class T:
        def __init__(self, claims): self.claims = claims
    assert crosscheck.synth_builder_for(T([MatmulClaim(), MatmulClaim()])) == "maverick"
    assert crosscheck.synth_builder_for(
        T([MatmulClaim(), RoutedProjectedMatmulClaim(), RescaleClaim()])) == "maverick-projected"
    assert set(crosscheck.synth_builder_for(T([])) for _ in range(1)) == {"maverick"}
    assert "maverick-projected" in synth.BUILDERS and "maverick" in synth.BUILDERS


def test_derive_prove_constants():
    from machine import MachineProfile
    base = MachineProfile.load("gb10-spark")
    # 4x the bandwidth -> A/C shrink 4x. gb10 has no register-resident
    # compress baseline and the column-hash rate is bandwidth-limited
    # (review finding), so B derives DIRECT from this box's reg rate —
    # 1 compress/cid — keeping the floor computable (second-pass finding:
    # predict/partition need A, B, AND C together).
    consts, prov = calibrate.derive_prove_constants(base, 223.0 * 4, 13.4)
    assert abs(consts["A_ns_per_slot"] - 9.0 / 4) < 1e-6
    assert abs(consts["C_ns_per_product"] - 15.0 / 4) < 1e-6
    assert abs(consts["B_ns_per_cid"] - round(1.0 / 13.4, 4)) < 1e-9
    assert "DIRECT" in prov["B_ns_per_cid"]
    assert "13.4" in prov["B_ns_per_cid"]
    assert "aggregate_ns_per_slot" not in consts     # never derived
    # with a reg baseline on the base profile, the ratio path wins
    fake = MachineProfile(dict(name="fake-base",
                               gpu=dict(mem_bandwidth_GBps=223.0,
                                        blake3_reg_compress_Gps=2.5),
                               prove_constants=dict(A_ns_per_slot=9.0,
                                                    B_ns_per_cid=0.6,
                                                    C_ns_per_product=15.0)))
    consts, prov = calibrate.derive_prove_constants(fake, 223.0 * 4, 25.0)
    assert abs(consts["B_ns_per_cid"] - 0.6 / 10) < 1e-6
    assert "DERIVED" in prov["B_ns_per_cid"]
    # no reg measurement at all -> B null, provenance says the floor is blocked
    consts, prov = calibrate.derive_prove_constants(base, 223.0 * 4, None)
    assert "B_ns_per_cid" not in consts
    assert "left null" in prov["B_ns_per_cid"]
    # missing measurements derive nothing
    consts, _ = calibrate.derive_prove_constants(base, None, None)
    assert consts == {}


# ------------------------------------------------- crosscheck: layout

def test_parse_layout():
    text = ("noise before\n"
            "=== witness layout by claim type (m_total=1,234, W=10,108,928 elements) ===\n"
            "  MatmulClaim            rows=      1,000  elements=      8,000,000   79.1%\n"
            "  TableSettlement        rows=        234  elements=      1,600,000   15.8%\n")
    m_total, table = crosscheck.parse_layout(text)
    assert m_total == 1234
    assert table["MatmulClaim"] == (1000, 8_000_000)
    assert table["TableSettlement"] == (234, 1_600_000)
    assert crosscheck.parse_layout("no table") == (None, {})


def test_layout_from_manifest():
    ell = 8192
    # Mirrors extract's settlement shape: inputs = [mult] + z_vars (mult is
    # producer-less; each z's producer is its own lookup claim), outputs =
    # [w_tbl] (produced at settle time).
    man = Manifest(
        run=dict(seq=8, ligero=dict(ELL=ell)),
        claims=[
            ClaimRecord(idx=0, type="MatmulClaim",
                        inputs=["w", "x"], outputs=["y"]),
            # lookup touches mult BEFORE the settlement in tape order...
            ClaimRecord(idx=1, type="PairedTlookupClaim",
                        inputs=["y", "mult"], outputs=["z"]),
            ClaimRecord(idx=2, type="TableSettlement",
                        inputs=["mult", "z"], outputs=["w_tbl"]),
        ],
        variables=[
            VariableRecord(name="w", length=2 * ell, persistent=True),
            VariableRecord(name="x", length=ell),
            VariableRecord(name="y", length=3 * ell + 1, producer=0),
            VariableRecord(name="z", length=ell, producer=1),
            VariableRecord(name="mult", length=ell),
            VariableRecord(name="w_tbl", length=ell, producer=2),
        ])
    agg = crosscheck.layout_from_manifest(man)
    # ...but core books mult (reached via the Table on the settlement) under
    # TableSettlement, plus w_tbl as the settlement's own output...
    assert agg["TableSettlement"] == (2, 2 * ell)
    assert agg["MatmulClaim"] == (2 + 1 + 4, 2 * ell + ell + 3 * ell + 1)
    # ...while z — a direct field of its lookup claim, encountered there
    # first — must STAY with the lookup despite being a settlement input
    # (review finding: assigning all settlement inputs flags normal layouts)
    assert agg["PairedTlookupClaim"] == (1, ell)


def test_diff_layout_flags():
    ell = 8192
    man = Manifest(
        run=dict(seq=8, ligero=dict(ELL=ell)),
        claims=[ClaimRecord(idx=0, type="MatmulClaim",
                            inputs=["x"], outputs=["y"])],
        variables=[VariableRecord(name="x", length=ell),
                   VariableRecord(name="y", length=ell, producer=0)])
    ours = crosscheck.layout_from_manifest(man)
    # identical probe -> clean, and m_total above rows is fine (blinding)
    flags, _ = _quiet(crosscheck.diff_layout, 5, dict(ours), man)
    assert flags == []
    # manifest rows exceeding m_total is a flag
    flags, _ = _quiet(crosscheck.diff_layout, 1, dict(ours), man)
    assert any("EXCEED" in f for f in flags)
    # a diverging type is a flag
    bad = dict(ours)
    bad["MatmulClaim"] = (99, 99)
    flags, _ = _quiet(crosscheck.diff_layout, 5, bad, man)
    assert any("MatmulClaim" in f for f in flags)


# ------------------------------------------------- crosscheck: differ

def test_diff_report_clean_on_synth_vs_self():
    sy = synth.BUILDERS["llama7b"](8, layers=2)
    ex = synth.BUILDERS["llama7b"](8, layers=2)
    flags, out = _quiet(crosscheck.diff_report, sy, ex)
    assert flags == []
    assert "exact match" in out


def test_diff_report_known_extract_only_not_flagged():
    sy = synth.BUILDERS["llama7b"](8, layers=2)
    ex = copy.deepcopy(sy)
    ex.claims.append(ClaimRecord(idx=len(ex.claims), type="TableSettlement",
                                 params=dict(T_LEN=16), w_slots=32.0))
    flags, out = _quiet(crosscheck.diff_report, sy, ex)
    assert flags == []
    assert "extract-only" in out


def test_diff_report_flags_missing_and_weight_mismatch():
    sy = synth.BUILDERS["llama7b"](8, layers=2)
    ex = copy.deepcopy(sy)
    ex.claims = [c for c in ex.claims if crosscheck._canon_group(c.type) != "add"]
    for v in ex.variables:
        if v.persistent:
            v.length += 1
            break
    flags, _ = _quiet(crosscheck.diff_report, sy, ex)
    assert any("add" in f and "never emitted" in f for f in flags)
    assert any("persistent" in f for f in flags)


def test_diff_report_flags_cost_drift_at_equal_count():
    # Review repro: same claim count, same W (pinned via w_slots), but
    # cids/Q shifted — previously reported clean.
    sy = synth.BUILDERS["llama7b"](8, layers=2)
    ex = copy.deepcopy(sy)
    victim = next(c for c in ex.claims
                  if crosscheck._canon_group(c.type) == "hadamard")
    import claimcosts
    w, _, _ = claimcosts.cost(victim.type, victim.params)
    victim.w_slots = w                       # W unchanged...
    victim.params = {k: 2 * v if isinstance(v, int) else v
                     for k, v in victim.params.items()}   # ...cids/Q shift
    flags, _ = _quiet(crosscheck.diff_report, sy, ex)
    assert any("cost drift" in f for f in flags)


def test_diff_report_flags_extra_modeled_claims_strict():
    # Review repro: an extra claim of a MODELED type — previously labeled
    # "UI chain — expected" and skipped. Strict mode (no UI chain) flags.
    sy = synth.BUILDERS["llama7b"](8, layers=2)
    ex = copy.deepcopy(sy)
    dup = copy.deepcopy(next(c for c in ex.claims
                             if crosscheck._canon_group(c.type) == "add"))
    dup.idx = len(ex.claims)
    ex.claims.append(dup)
    flags, _ = _quiet(crosscheck.diff_report, sy, ex)
    assert any("no UI chain" in f for f in flags)


def _dup_claim(man, pred):
    dup = copy.deepcopy(next(c for c in man.claims if pred(c)))
    dup.idx = len(man.claims)
    man.claims.append(dup)


def test_diff_report_ui_expected_matmul_never_excused():
    # Review repro (second pass): a duplicated REAL expert matmul in
    # synthetic Maverick — +1 matmul, +0.011% W — must flag in UI mode.
    # The UI chain adds NO matmul (its select matmul maps to embed.select).
    sy = synth.BUILDERS["maverick"](4)
    ex = copy.deepcopy(sy)
    _dup_claim(ex, lambda c: "e0.gate" in (c.label or ""))
    flags, _ = _quiet(crosscheck.diff_report, sy, ex, 2)
    assert any("matmul" in f and "no UI chain" in f for f in flags)


def test_diff_report_ui_expected_per_type():
    C = 5
    sy = synth.BUILDERS["maverick"](4)
    # exactly the enumerated UI extras -> clean
    ex = copy.deepcopy(sy)
    _dup_claim(ex, lambda c: crosscheck._canon_group(c.type) == "hadamard")
    for i in range(C):
        ex.claims.append(ClaimRecord(idx=len(ex.claims),
                                     type="EmbeddingLookupClaim",
                                     params=dict(L=1), w_slots=1.0))
        ex.claims.append(ClaimRecord(idx=len(ex.claims), type="AddClaim",
                                     params=dict(length=1), w_slots=1.0))
    flags, out = _quiet(crosscheck.diff_report, sy, ex, C)
    assert flags == []
    assert "UI chain, expected" in out
    # one embed_lookup beyond the per-type expectation -> flag
    ex2 = copy.deepcopy(ex)
    ex2.claims.append(ClaimRecord(idx=len(ex2.claims),
                                  type="EmbeddingLookupClaim",
                                  params=dict(L=1), w_slots=1.0))
    flags, _ = _quiet(crosscheck.diff_report, sy, ex2, C)
    assert any("embed_lookup" in f and "at most" in f for f in flags)
    # a type the UI chain never emits -> flag even in UI mode
    ex3 = copy.deepcopy(sy)
    _dup_claim(ex3, lambda c: crosscheck._canon_group(c.type) == "silu")
    flags, _ = _quiet(crosscheck.diff_report, sy, ex3, C)
    assert any("silu" in f and "no UI chain" in f for f in flags)


def test_diff_report_routing_bundle_representation():
    # Third-pass P1 repro: a real tape emits every synth `routing` bundle
    # as core + word-extract + n_words range words (cost-sum identical by
    # the claimcosts bundle identity) — standard Maverick: 24 bundles ->
    # 120 records, +4 for the input route = routing[+aux] 24 -> 124. Must
    # NOT flag; the cap derives from synth's own bundles.
    T = 4
    sy = synth.BUILDERS["maverick"](T)
    ex = copy.deepcopy(sy)
    expanded = []
    for c in ex.claims:
        if crosscheck._canon_group(c.type) != "routing[+aux]":
            expanded.append(c)
            continue
        E, nw = c.params["E"], c.params["n_words"]
        expanded.append(ClaimRecord(idx=0, type="RoutingClaim",
                                    params=dict(T=T, E=E), label=c.label,
                                    layer=c.layer, inputs=c.inputs,
                                    outputs=c.outputs))
        expanded.append(ClaimRecord(idx=0, type="WordExtractionClaim",
                                    params=dict(length=T * E, n_words=nw)))
        expanded.extend(ClaimRecord(idx=0, type="RangeWordClaim",
                                    params=dict(length=T * E))
                        for _ in range(nw))
    # the input-route pieces (route_top1 over the token indicator, E=V)
    V = sy.model["vocab"]
    expanded.append(ClaimRecord(idx=0, type="RoutingClaim",
                                params=dict(T=T, E=V)))
    expanded.append(ClaimRecord(idx=0, type="WordExtractionClaim",
                                params=dict(length=T * V, n_words=2)))
    expanded.extend(ClaimRecord(idx=0, type="RangeWordClaim",
                                params=dict(length=T * V))
                    for _ in range(2))
    for i, c in enumerate(expanded):
        c.idx = i
    ex.claims = expanded
    flags, _ = _quiet(crosscheck.diff_report, sy, ex, 2)
    assert flags == []
    # pushing past the derived cap (input-route headroom is 6, used 4) flags
    for _ in range(3):
        ex.claims.append(ClaimRecord(idx=len(ex.claims),
                                     type="RangeWordClaim",
                                     params=dict(length=T), w_slots=4.0))
    flags, _ = _quiet(crosscheck.diff_report, sy, ex, 2)
    assert any("routing[+aux]" in f and "at most" in f for f in flags)


def test_routing_grouped_across_conventions():
    # synth bundles `routing`; a tape emits the pieces — same group either way
    assert crosscheck._canon_group("routing") == "routing[+aux]"
    assert crosscheck._canon_group("RoutingClaim") == "routing[+aux]"
    assert crosscheck._canon_group("WordExtractionClaim") == "routing[+aux]"
    assert crosscheck._canon_group("RangeWordClaim") == "routing[+aux]"
    assert crosscheck._canon_group("MatmulClaim") == "matmul"


def main():
    test_parse_field_mul()
    test_parse_ntt()
    test_parse_blake3()
    test_parse_matmul()
    test_parse_blake3_reg()
    test_parse_session2_benches()
    test_dump_compact_smoke()
    test_bench_grid_fills_the_part()
    test_synth_builder_chosen_from_tape()
    test_derive_prove_constants()
    test_parse_layout()
    test_layout_from_manifest()
    test_diff_layout_flags()
    test_diff_report_clean_on_synth_vs_self()
    test_diff_report_known_extract_only_not_flagged()
    test_diff_report_flags_missing_and_weight_mismatch()
    test_diff_report_flags_cost_drift_at_equal_count()
    test_diff_report_flags_extra_modeled_claims_strict()
    test_diff_report_ui_expected_matmul_never_excused()
    test_diff_report_ui_expected_per_type()
    test_diff_report_routing_bundle_representation()
    test_routing_grouped_across_conventions()
    print("calibration-tools tests OK (no torch needed)")


if __name__ == "__main__":
    main()
