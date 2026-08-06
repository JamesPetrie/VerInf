"""Production-geometry kernel rates for the admission report (S5).

Measures the EXECUTED loop bodies at the target Ligero geometry
(ELL=8192, K_DEG=16384, N_LIG=65536), and converts every
per-slot rate into the stage seconds the admission model caps.

What it does NOT do, on purpose:

  * it does not measure `model_load` or the five semantic sweeps — those need
    the real GGUF shards, and the runbook forbids substituting random weights
    for them. They are emitted as `null` and the gate then REFUSES the report,
    which is the correct outcome for a partial measurement.
  * it does not fit or extrapolate across geometries: the rate is measured at
    the geometry the proof runs at, and multiplied by the row capacity the
    model prices. No global multiplier anywhere.

Usage:
    python analysis/bench/admission_bench.py --runs 30 --out bench.json  # exploratory
"""
import argparse
import json
import math
import pathlib
import sys
import time

_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "prover"))

import torch                                            # noqa: E402
import core                                             # noqa: E402
import cuda_primitives as cp                            # noqa: E402
import packets                                          # noqa: E402
import admission                                        # noqa: E402

sys.path.insert(0, str(_ROOT / "analysis"))
import routed_projected_4h_model as model               # noqa: E402

CFG = core.LigeroConfig(**admission.TARGET)
ROWS = 512                                              # rows per timed batch


def _p99_upper(samples):
    """Distribution-free upper tolerance statistic: the observed maximum.

    With admission.MIN_RUNS=714, Bonferroni across every priced stage gives at
    least 99% confidence that all these maxima exceed their true p99 values.
    mean+3sd is not a distribution-free tail bound and is intentionally not
    used.  Smaller probes are exploratory only and the gate refuses them."""
    return max(samples)


def _time(fn, runs, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        out.append(time.perf_counter() - t0)
    return out


def measure(runs: int, egress_dir: str) -> dict:
    seed = core._master_seed_to_cuda(core.new_zk_seed())
    msgs = torch.randint(0, 1 << 62, (ROWS, CFG.ELL), dtype=torch.int64,
                         device="cuda").view(torch.uint64)
    slots = ROWS * CFG.ELL
    got, stage_samples = {}, {}

    # --- encode: pad + inverse NTT + coset LDE (the commit/fold body) ------
    hold = {}

    def do_encode():
        hold["cw"] = core.encode_messages(msgs, CFG, master_seed=seed)[1]

    enc = _time(do_encode, runs)
    got["encode_ns_per_slot"] = _p99_upper(enc) * 1e9 / slots
    # NOTE: fresh_commit_fold is NOT charged from this number. Its cap prices
    # the row/witness side of the round, which is encode AND the folds fed from
    # the same pass; charging the encode alone would understate it. It is
    # assigned from the full row-side body measured below.

    # --- blake3 column hashing --------------------------------------------
    cw = hold["cw"]
    hsh = _time(lambda: cp.hash_columns_streamed(cw), runs)
    got["hash_ns_per_slot"] = _p99_upper(hsh) * 1e9 / slots
    stage_samples["fresh_hash_coef"] = [
        x / slots * model.FRESH_ROW_CAPACITY for x in hsh]

    # --- opening: gather the queried columns and land them on the host ----
    cols = list(range(0, CFG.N_LIG, CFG.N_LIG // CFG.T_QUERIES))[:CFG.T_QUERIES]
    q_set = torch.tensor(cols, dtype=torch.long, device="cuda")

    def do_open():
        sink = core.ColumnSink(ROWS, cols, 0)
        sink.write(0, cw.index_select(1, q_set), cols)
        sink.finish()

    opn = _time(do_open, runs, warmup=1)
    got["open_ns_per_slot"] = _p99_upper(opn) * 1e9 / slots
    stage_samples["persistent_open"] = [
        x / slots * model.WEIGHT_ROW_CAPACITY for x in opn]
    stage_samples["fresh_open"] = [
        x / slots * model.FRESH_ROW_CAPACITY for x in opn]

    # --- quadratic: the real p_0 streaming body ---------------------------
    v1 = core.Variable("q_x", length=ROWS * CFG.ELL, phase=1)
    v2 = core.Variable("q_y", length=ROWS * CFG.ELL, phase=2)
    v1.row_start, v2.row_start = core.NUM_BLINDING_ROWS, core.NUM_BLINDING_ROWS + ROWS
    vals = torch.randint(0, 1 << 62, (ROWS * CFG.ELL,), dtype=torch.int64,
                         device="cuda").view(torch.uint64)
    inputs = {v1: vals, v2: vals}
    m_p1 = core.NUM_BLINDING_ROWS + ROWS
    fam = core.QuadFamily(name="bench", x_row=v1.row_start, y_row=v2.row_start,
                          z_row=v2.row_start, L=ROWS * CFG.ELL, ell=CFG.ELL,
                          a=core.P - 1, b=0)
    quads = fam.expand()
    r_quad = torch.randint(0, 1 << 62, (len(quads),), dtype=torch.int64,
                           device="cuda").view(torch.uint64)
    maps = (core._build_row_map([v1], CFG, core.NUM_BLINDING_ROWS),
            core._build_row_map([v2], CFG, m_p1))

    def do_quad():
        core.compute_p_0_streaming([v1], [v2], inputs, m_p1, r_quad, quads,
                                   CFG, seed, maps=maps)

    qd = _time(do_quad, runs, warmup=1)
    got["quad_ns_per_product"] = _p99_upper(qd) * 1e9 / (ROWS * CFG.ELL)
    stage_samples["quadratic"] = [
        x / (ROWS * CFG.ELL) * model.QUADRATIC_COUNT_CAP for x in qd]

    # --- the R3 q-fold sweep over the enrolled weight block ---------------
    # This is the stage the model prices as persistent_weight_qlin (cap
    # 3624.5 s, the largest kernel stage) and linear.  The body is the one the
    # prover executes: _stream_phase over weight variables with the two q
    # accumulators as its only sinks — no Merkle tree, no column extraction,
    # because a referenced enrollment supplies root and paths.
    #
    # The band index drives q_lin's work, so it is a real one: an identity
    # band per row, which is what a weight variable's linear constraints
    # lower to.  Measuring with an empty index would price an empty fold.
    qv = core.Variable("w_bench", length=ROWS * CFG.ELL, phase=1, persistent=True)
    qv.row_start = core.NUM_BLINDING_ROWS
    q_inputs = {qv: vals}
    ident = packets.L2_IdentityScalar(base=0, var_row_start=qv.row_start,
                                      L=ROWS * CFG.ELL, coef=1)
    per_row = [[] for _ in range(core.NUM_BLINDING_ROWS)] + [[ident]] * ROWS
    r_irs = torch.randint(0, 1 << 62, (ROWS,), dtype=torch.int64,
                          device="cuda").view(torch.uint64)
    seed_u8_lin = torch.tensor(list(core.new_zk_seed()), dtype=torch.uint8,
                               device="cuda")

    def do_qsweep():
        qi = core.QIrsAccumulator(r_irs, CFG)
        ql = core.QLinAccumulator(seed_u8_lin, per_row, CFG)
        core._stream_phase(
            [qv], q_inputs, CFG, master_seed=seed,
            abs_row_offset=core.NUM_BLINDING_ROWS,
            q_irs_acc=qi, q_lin_acc=ql)
        qi.finalize(); ql.finalize()

    qs = _time(do_qsweep, runs, warmup=1)
    got["qsweep_ns_per_slot"] = _p99_upper(qs) * 1e9 / slots
    # The SAME row-side body prices both per-slot stages; they differ only in
    # how many rows they run over. It contains the constraint-side fold too, so
    # against `linear` this is a deliberate over-charge, never an under-charge.
    stage_samples["persistent_weight_qlin"] = [
        x / slots * model.WEIGHT_ROW_CAPACITY for x in qs]
    stage_samples["fresh_commit_fold"] = [
        x / slots * model.FRESH_ROW_CAPACITY for x in qs]

    # --- the q_lin fold alone, per constraint id (REPORTED, NOT CHARGED) ---
    # Measured because it is the one number that says whether the fold is
    # affordable at all — but deliberately NOT mapped onto the model's
    # `linear` stage, because that would double-charge it.  The evidence that
    # the fold is already inside the two per-slot stages:
    #   fresh_commit_fold      cap 9.5 ns/slot, encode alone measures ~4.9
    #   persistent_weight_qlin cap 9.0 ns/slot, encode + BOTH folds measures ~8.0
    # i.e. both caps were sized for encode+fold, and the second one is met by a
    # body that contains the whole fold.  What the model's separate 25.6 s
    # `linear` line is meant to cover beyond that is not defined by the model,
    # so this bench reports the rate and leaves the stage unmeasured rather
    # than inventing a mapping that happens to pass or happens to fail.
    polys_pre = core.encode_messages(msgs, CFG, master_seed=seed)[0]

    def do_lin_fold():
        ql = core.QLinAccumulator(seed_u8_lin, per_row, CFG)
        ql.update(core.NUM_BLINDING_ROWS, polys_pre)
        ql.finalize()

    lin = _time(do_lin_fold, runs, warmup=1)
    # One identity band per row covers ELL constraint ids per row, so this
    # batch folds `slots` ids.
    got["lin_ns_per_cid"] = _p99_upper(lin) * 1e9 / slots
    stage_samples["linear"] = [
        x / slots * model.LINEAR_COUNT_CAP for x in lin]

    # --- proof egress: production u64le/base64 JSON transport -------------
    import proof_dump
    import tempfile, os
    payload = torch.randint(0, 1 << 62, (4_000_000,), dtype=torch.int64,
                            device="cpu").view(torch.uint64)
    fd, path = tempfile.mkstemp(suffix=".json", dir=egress_dir)
    os.close(fd)

    def do_write():
        with open(path, "w") as f:
            proof_dump._w_u64_b64(f, payload)

    try:
        wr = _time(do_write, runs, warmup=1)
        nbytes = os.path.getsize(path)
    finally:
        os.unlink(path)
    got["egress_bytes_per_s"] = nbytes / _p99_upper(wr)
    stage_samples["proof_egress"] = [
        x / nbytes * model.PROOF_BYTES_COMPACT for x in wr]
    return got, stage_samples


def stages_from_rates(r: dict) -> dict:
    """Per-slot rates x the row capacities the admission model prices."""
    ns = 1e-9
    return {
        "model_load": None,                       # needs the real GGUF
        "semantic_5_active_sweeps": None,         # needs the real GGUF
        "fresh_commit_fold": r["qsweep_ns_per_slot"] * ns * model.FRESH_ROW_CAPACITY,
        # `linear` is the CONSTRAINT side of the q_lin fold (the model's owner
        # settled this on 2026-08-06); the two per-slot stages price the
        # row/witness side.
        "linear": r["lin_ns_per_cid"] * ns * model.LINEAR_COUNT_CAP,
        "quadratic": r["quad_ns_per_product"] * ns * model.QUADRATIC_COUNT_CAP,
        "fresh_hash_coef": r["hash_ns_per_slot"] * ns * model.FRESH_ROW_CAPACITY,
        "persistent_weight_qlin": r["qsweep_ns_per_slot"] * ns * model.WEIGHT_ROW_CAPACITY,
        "persistent_open": r["open_ns_per_slot"] * ns * model.WEIGHT_ROW_CAPACITY,
        "fresh_open": r["open_ns_per_slot"] * ns * model.FRESH_ROW_CAPACITY,
        "proof_egress": model.PROOF_BYTES_COMPACT / r["egress_bytes_per_s"],
        "rtt": None,
        "tail": None,
        "orchestration_refresh": None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--out", default=None)
    ap.add_argument("--egress-dir", default=".",
                    help="the same filesystem that will hold the production proof")
    a = ap.parse_args()
    if a.runs < admission.MIN_RUNS:
        print(f"NOTE: {a.runs} runs is below the admission minimum "
              f"({admission.MIN_RUNS}); this cannot become a real report.")

    fp = admission.machine_fingerprint()
    print(f"machine: {fp['gpu_name']}  torch {fp['torch']}  cuda {fp['cuda']}")
    print(f"geometry: {admission.TARGET}\n")

    rates, stage_samples = measure(a.runs, a.egress_dir)
    stages = stages_from_rates(rates)

    print(f"{'stage':26s} {'measured':>12s} {'cap':>12s}  verdict")
    over = []
    for stage, cap in admission.STAGE_CAPS.items():
        got = stages.get(stage)
        if got is None:
            print(f"{stage:26s} {'NOT MEASURED':>12s} {cap:12.1f}  needs the real model / not built")
            continue
        ok = got <= cap
        if not ok:
            over.append((stage, got, cap))
        print(f"{stage:26s} {got:12.1f} {cap:12.1f}  {'ok' if ok else 'OVER CAP'}")

    print()
    for k, v in rates.items():
        print(f"  {k:26s} {v:,.3f}")
    if over:
        print("\nSTAGES OVER CAP on this machine:")
        for stage, got, cap in over:
            print(f"  {stage}: {got:.1f}s > {cap:.1f}s  ({got / cap:.2f}x)")
    else:
        print("\nEvery MEASURED stage is under its cap on this machine.")
    print("Unmeasured stages keep the report incomplete, so the gate will "
          "refuse it — by design.")

    if a.out:
        doc = {
            "machine": fp,
            "geometry": admission.TARGET,
            "runs": a.runs,
            "bound_kind": admission.BOUND_KIND,
            "weights": "none_kernel_only",
            "rates": rates,
            "stages": stages,
            "stage_samples": stage_samples,
            "egress_filesystem": admission.filesystem_fingerprint(a.egress_dir),
        }
        pathlib.Path(a.out).write_text(json.dumps(doc, indent=2))
        print(f"\nwrote {a.out} (NOT an admission report: stages are missing "
              f"and weights are not real GGUF)")


if __name__ == "__main__":
    main()
