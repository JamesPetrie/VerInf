"""Weight-split M1a gate: a proof whose ENROLLED weight block was folded and
opened across several "devices" is BYTE-IDENTICAL to the single-device proof
and Rust-verifies ACCEPT. Every role runs sequentially on cuda:0 — the
identity is what makes correctness testable without a second GPU.

Gates:
1. N=2 at every cut of the seven-variable block (0..7), fold and open cut
   at the same place — identical proof bytes; the cut inside a claim's
   weight group (rows non-chunk-aligned, multi-row variables) included.
2. N=3 with the fold and open stages cut DIFFERENTLY (interior worker whose
   two runs do not nest) — identical bytes; Rust ACCEPT on the sharded proof.
3. Fuse modes: LIGERO_FUSE_POLYMUL=0 (coeff-domain q_lin) and
   LIGERO_FUSE_CHECK=1 (both, asserted equal) — identical bytes.
4. Invalid plans (gap, overlap, device with two runs, wrong count) are
   rejected before any weight row is touched; a sink that receives a
   duplicated or missing piece fails its coverage check.
5. shard_plan without an enrollment is refused; a worker device other than
   the coordinator's is refused (M1b), never silently run on the wrong one.
6. CHUNK-GRID SHIFT: a
   block with a >1024-row variable and a >256-row variable, so a worker
   run spans several claim groups AND several encode/q_lin chunks — the
   coordinator emits per claim group (its chunk grid restarts at each
   group), the worker packs its run as one sequence, and the proof must
   still be byte-identical (padding keyed by absolute row, per-row NTTs,
   canonical field sums).
7. QUADS ON W ROWS: a
   MASTER_SEED enrollment with a matmul consuming an enrolled weight, so
   p_0's sparse re-encode touches W rows — the production shape; the
   fixture above is linear in W under a non-master seed, which forbids
   quads there.

Run on a GPU box:  python run_tests.py test_weight_split   (needs the Rust
verifier for the ACCEPT check). Gate history: 1-5 passed 4/4 on a rented A40
2026-08-30 (torch 2.8.0+cu128); all six passed 6/6 on a rented A40 2026-09-02
(torch 2.8.0+cu128, driver 580.178.04): gate 6 with 1,408 W rows across
groups and encode/q_lin chunks, gate 7 with Freivalds quads on the enrolled
rows under a MASTER_SEED enrollment, Rust ACCEPT on both sharded proofs.
"""
import os
import sys
import pathlib
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch
import core
import protocol as pr
from proof_dump import dump_proof
from shard_plan import ShardPlan
from tape import Tape
from _rust_verify import rust_verify_tape

CFG = core.LigeroConfig(ELL=8192, K_DEG=16384, N_LIG=65536, T_QUERIES=4)
ENROL_SEED = b"\x11" * 32
ZK_SEED = b"\x22" * 32
# seven persistent weights: mixed row counts (ELL=8192), several not
# ELL-aligned, so runs cut at non-chunk-aligned rows and pad per variable
LENS = [12000, 8192, 20000, 5, 16384, 9000, 3000]
N_W = len(LENS)


def _t(vals):
    return torch.tensor(vals, dtype=torch.int64, device="cuda").to(torch.uint64)


def _build():
    """Linear in W (concat = per-slot identity pins; no quads on weight
    rows, as the enrollment seed != MASTER_SEED requires). Claims group
    several weights of DIFFERENT lengths so that ownership cuts fall inside
    claims' weight groups (Tape.add would need equal shapes)."""
    tape = Tape(CFG, lazy=True)
    ws = [tape.commit(f"W{i}", _t([(v * (i + 3)) % 1000003 for v in range(n)]), (n,),
                      persistent=True) for i, n in enumerate(LENS)]
    a = tape.concat([ws[0], ws[1]], (LENS[0] + LENS[1],))          # group [W0, W1]
    b = tape.concat([ws[2], ws[3], ws[4]], (sum(LENS[2:5]),))      # group [W2, W3, W4]
    c = tape.concat([ws[5], ws[6]], (LENS[5] + LENS[6],))          # group [W5, W6]
    tape.concat([a, b, c], (sum(LENS),))                           # activations only
    return tape


def _enroll():
    return core.WeightCommitment.from_tape(_build(), CFG, master_seed=ENROL_SEED)


def _prove(wc, plan=None):
    tape = _build()
    proof = tape.prove(zk_seed=ZK_SEED, weight_commitment=wc, shard_plan=plan)
    return tape, proof


def _bytes(tape, proof):
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        dump_proof(path, pr.claims_to_json(tape.claims, CFG), None, proof, None, None)
        with open(path, "rb") as f:
            return f.read()
    finally:
        for p in (path, path + ".part"):
            if os.path.exists(p):
                os.unlink(p)


def _same(base, tape, proof, what):
    got = _bytes(tape, proof)
    assert got == base, f"{what}: sharded proof bytes differ from the single-device proof"
    assert len(got) > 1000


def test_n2_every_cut_is_byte_identical():
    wc = _enroll()
    t0, p0 = _prove(wc)
    base = _bytes(t0, p0)
    assert p0.root_w == wc.root and len(p0.opened_w) == CFG.T_QUERIES
    n_rows = sum(-(-n // CFG.ELL) for n in LENS)
    assert next(iter(p0.opened_w.values())).numel() == n_rows
    for cut in range(N_W + 1):
        _, p = _prove(wc, ShardPlan.two_way(cut))
        _same(base, t0, p, f"N=2 cut={cut}")
    print(f"    N=2: all {N_W + 1} cuts byte-identical ({n_rows} W rows, "
          f"{len(base)} proof bytes)")


def test_n3_different_stage_cuts_and_rust_accept():
    wc = _enroll()
    t0, p0 = _prove(wc)
    base = _bytes(t0, p0)
    # fold: coordinator [0,1), w1 [1,4), w2 [4,7); open: [0,2), [2,5), [5,7)
    # -> interior worker 1's runs [1,4) and [2,5) genuinely do NOT nest
    # (each contains rows the other lacks); worker 2's [4,7)/[5,7) nest
    plan = ShardPlan.from_pairs([(0, 1), (1, 4), (4, 7)], [(0, 2), (2, 5), (5, 7)])
    t, p = _prove(wc, plan)
    _same(base, t0, p, "N=3 mixed cuts")
    acc, msg = rust_verify_tape(t, p, seed=b"ws")
    assert acc, msg
    # coordinator idle in both stages (everything on the workers)
    plan = ShardPlan.from_pairs([(0, 0), (0, 3), (3, 7)], [(0, 0), (0, 5), (5, 7)])
    _, p = _prove(wc, plan)
    _same(base, t0, p, "N=3 coordinator idle")
    print("    N=3: differing fold/open cuts byte-identical; Rust ACCEPT")


def test_fuse_modes_are_byte_identical():
    wc = _enroll()
    plan = ShardPlan.two_way(3, cut_open=5)
    saved = {k: os.environ.get(k) for k in ("LIGERO_FUSE_POLYMUL", "LIGERO_FUSE_CHECK")}
    try:
        for env in ({"LIGERO_FUSE_POLYMUL": "0"}, {"LIGERO_FUSE_CHECK": "1"}):
            for k in saved:
                os.environ.pop(k, None)
            os.environ.update(env)
            t0, p0 = _prove(wc)
            base = _bytes(t0, p0)
            _, p = _prove(wc, plan)
            _same(base, t0, p, f"fuse mode {env}")
            print(f"    {env}: byte-identical")
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_invalid_plans_and_sink_coverage():
    wc = _enroll()
    bad = [
        ([(0, 2), (3, 7)], None, "owned by nobody"),
        ([(0, 4), (3, 7)], None, "owned twice"),
        ([(0, 4), (4, 8)], None, "outside"),
        ([(0, 4), (4, 7)], [(0, 4), (5, 7)], "open plan"),
    ]
    for fold, open_, needle in bad:
        try:
            _prove(wc, (fold, open_ if open_ is not None else fold))
            raise AssertionError(f"accepted invalid plan {fold}/{open_}")
        except ValueError as e:
            assert needle in str(e), (needle, str(e))
    # no enrollment -> refused (the sentinel message must NOT contain the
    # needle, or a removed guard would pass this check vacuously)
    try:
        tape = _build()
        tape.prove(zk_seed=ZK_SEED, shard_plan=ShardPlan.two_way(3))
        raise AssertionError("guard missing: plan accepted with no enrollment")
    except AssertionError as e:
        assert "needs weight_commitment" in str(e), str(e)
    # a worker on another device is M1b: refused loudly, not run on cuda:0
    try:
        _prove(wc, ShardPlan.from_pairs([(0, 3), (3, 7)], devices={1: "cuda:7"}))
        raise AssertionError("guard missing: foreign worker device accepted")
    except NotImplementedError as e:
        assert "M1b" in str(e)
    # sink coverage: a duplicated piece over-fills; a missing one under-fills;
    # and a count that happens to match still fails the range check
    Q = [1, 5]
    sink = core.ColumnSink(4, Q, row_base=10)
    piece = {j: torch.zeros(2, dtype=torch.uint64) for j in Q}
    sink.write_host(10, piece)
    sink.write_host(10, piece)                 # duplicate: count matches (4)
    try:
        sink.finish()
        raise AssertionError("duplicate piece accepted")
    except AssertionError as e:
        assert "coverage" in str(e)
    sink = core.ColumnSink(4, Q, row_base=10)
    sink.write_host(10, piece)
    try:
        sink.finish()
        raise AssertionError("missing piece accepted")
    except AssertionError as e:
        assert "filled 2 of 4" in str(e)
    sink = core.ColumnSink(4, Q, row_base=10)
    sink.write_host(12, piece)
    sink.write_host(10, piece)                 # out-of-order pieces are fine
    assert all(t.numel() == 4 for t in sink.finish().values())
    print("    invalid plans rejected before any row; sink coverage exact")


# ---- gates 6/7: chunk-grid shift and quads on W rows (GPU-untested) -----

BIG_LENS = [12000, 8192 * 1100 + 5, 300 * 8192, 9000, 8192 * 2 - 1, 3000]
# rows at ELL=8192: 2, 1101 (> _ENCODE_CHUNK_ROWS 1024), 300 (> inner 256),
# 2, 2, 1 — a worker run over [1,4) spans three groups and several chunks


def _build_big():
    tape = Tape(CFG, lazy=True)
    ws = [tape.commit(f"B{i}", _t([(v * (i + 5)) % 1000003 for v in range(n)]), (n,),
                      persistent=True) for i, n in enumerate(BIG_LENS)]
    a = tape.concat([ws[0], ws[1]], (BIG_LENS[0] + BIG_LENS[1],))
    b = tape.concat([ws[2], ws[3]], (BIG_LENS[2] + BIG_LENS[3],))
    c = tape.concat([ws[4], ws[5]], (BIG_LENS[4] + BIG_LENS[5],))
    tape.concat([a, b, c], (sum(BIG_LENS),))
    return tape


def test_chunk_grid_shift_is_byte_identical():
    wc = core.WeightCommitment.from_tape(_build_big(), CFG, master_seed=ENROL_SEED)
    t0 = _build_big()
    p0 = t0.prove(zk_seed=ZK_SEED, weight_commitment=wc)
    base = _bytes(t0, p0)
    n_rows = sum(-(-n // CFG.ELL) for n in BIG_LENS)
    assert n_rows > core._ENCODE_CHUNK_ROWS
    # worker run [1, 4) crosses groups [B0,B1] | [B2,B3] and the 1024-row
    # encode chunk inside B1; fold and open cut differently at N=3
    for plan, what in ((ShardPlan.two_way(1), "N=2 cut=1"),
                       (ShardPlan.two_way(2, cut_open=4), "N=2 fold=2/open=4"),
                       (ShardPlan.from_pairs([(0, 1), (1, 4), (4, 6)],
                                             [(0, 2), (2, 5), (5, 6)]), "N=3")):
        t = _build_big()
        p = t.prove(zk_seed=ZK_SEED, weight_commitment=wc, shard_plan=plan)
        _same(base, t0, p, what)
    print(f"    chunk-grid shift: {n_rows} W rows, runs across groups and chunks, "
          f"byte-identical")


def _build_quads():
    """MASTER_SEED enrollment allows quads on W rows: a matmul consuming an
    enrolled weight puts W rows into p_0's sparse re-encode, the
    production shape (every expert matrix is a matmul operand)."""
    tape = Tape(CFG, lazy=True)
    k, n = 64, 96
    w0 = tape.commit("Q0", _t([(v * 7) % 1000003 for v in range(k * n)]), (k, n),
                     persistent=True)
    w1 = tape.commit("Q1", _t([(v * 11) % 1000003 for v in range(k * n)]), (k, n),
                     persistent=True)
    w2 = tape.commit("Q2", _t([(v * 13) % 1000003 for v in range(9000)]), (9000,),
                     persistent=True)
    x = tape.commit("x", _t([(v * 3) % 1000 for v in range(4 * k)]), (4, k))
    # plain (no-rescale) matmuls: the Freivalds quads land on the W rows,
    # which is what this gate exercises; the rescaled form would only add
    # range tables (and scale 1 everywhere is a zero-bit rescale the
    # compute path does not produce aux for)
    y0 = tape.matmul(x, w0)
    y1 = tape.matmul(x, w1)
    tape.concat([y0, y1], (2 * 4 * n,))
    tape.concat([w2], (9000,))
    return tape


def test_quads_on_weight_rows_are_byte_identical():
    wc = core.WeightCommitment.from_tape(_build_quads(), CFG, master_seed=core.MASTER_SEED)
    t0 = _build_quads()
    p0 = t0.prove(zk_seed=ZK_SEED, weight_commitment=wc)
    base = _bytes(t0, p0)
    n_w = len(core.collect_weight_vars(t0))
    assert n_w == 3
    for cut in range(n_w + 1):
        t = _build_quads()
        p = t.prove(zk_seed=ZK_SEED, weight_commitment=wc, shard_plan=ShardPlan.two_way(cut))
        _same(base, t0, p, f"quads-on-W N=2 cut={cut}")
    t = _build_quads()
    p = t.prove(zk_seed=ZK_SEED, weight_commitment=wc,
                shard_plan=ShardPlan.from_pairs([(0, 1), (1, 2), (2, 3)], [(0, 0), (0, 2), (2, 3)]))
    _same(base, t0, p, "quads-on-W N=3")
    acc, msg = rust_verify_tape(t, p, seed=b"wq")
    assert acc, msg
    print("    quads on W rows (MASTER_SEED enrollment, matmul operands): byte-identical; Rust ACCEPT")

