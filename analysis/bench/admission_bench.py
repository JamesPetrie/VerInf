"""Production-geometry kernel rates for the admission report (S5).

Measures the EXECUTED loop bodies at the target Ligero geometry
(ELL=8192, K_DEG=16384, N_LIG=65536), 30+ runs each, and converts every
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
    python analysis/bench/admission_bench.py --runs 30 --out bench.json
"""
import argparse
import json
import math
import pathlib
import statistics
import sys
import time

_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "prover"))

import torch                                            # noqa: E402
import core                                             # noqa: E402
import cuda_primitives as cp                            # noqa: E402
import admission                                        # noqa: E402

sys.path.insert(0, str(_ROOT / "analysis"))
import routed_projected_4h_model as model               # noqa: E402

CFG = core.LigeroConfig(**admission.TARGET)
ROWS = 512                                              # rows per timed batch


def _p99_upper(samples):
    """A conservative >=99% upper bound from n samples.

    With n=30 the empirical 99th percentile is not resolvable, so this uses
    mean + 3*sd (one-sided, ~99.9% for anything near-normal) and never returns
    less than the observed maximum. Deliberately pessimistic: an admission
    bound that is too tight is the failure mode that costs four hours."""
    mx = max(samples)
    if len(samples) < 2:
        return mx
    m, sd = statistics.mean(samples), statistics.stdev(samples)
    return max(mx, m + 3.0 * sd)


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


def measure(runs: int) -> dict:
    seed = core._master_seed_to_cuda(core.new_zk_seed())
    msgs = torch.randint(0, 1 << 62, (ROWS, CFG.ELL), dtype=torch.int64,
                         device="cuda").view(torch.uint64)
    slots = ROWS * CFG.ELL
    got = {}

    # --- encode: pad + inverse NTT + coset LDE (the commit/fold body) ------
    hold = {}

    def do_encode():
        hold["cw"] = core.encode_messages(msgs, CFG, master_seed=seed)[1]

    enc = _time(do_encode, runs)
    got["encode_ns_per_slot"] = _p99_upper(enc) * 1e9 / slots

    # --- blake3 column hashing --------------------------------------------
    cw = hold["cw"]
    hsh = _time(lambda: cp.hash_columns_streamed(cw), runs)
    got["hash_ns_per_slot"] = _p99_upper(hsh) * 1e9 / slots

    # --- opening: gather the queried columns and land them on the host ----
    cols = list(range(0, CFG.N_LIG, CFG.N_LIG // CFG.T_QUERIES))[:CFG.T_QUERIES]
    q_set = torch.tensor(cols, dtype=torch.long, device="cuda")

    def do_open():
        sink = core.ColumnSink(ROWS, cols, 0)
        sink.write(0, cw.index_select(1, q_set), cols)
        sink.finish()

    opn = _time(do_open, runs, warmup=1)
    got["open_ns_per_slot"] = _p99_upper(opn) * 1e9 / slots

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

    qd = _time(do_quad, max(3, runs // 5), warmup=1)
    got["quad_ns_per_product"] = _p99_upper(qd) * 1e9 / (ROWS * CFG.ELL)

    # --- proof egress: the streaming decimal-JSON writer ------------------
    import proof_dump
    import tempfile, os
    payload = torch.randint(0, 1 << 62, (4_000_000,), dtype=torch.int64,
                            device="cpu").view(torch.uint64)
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)

    def do_write():
        with open(path, "w") as f:
            proof_dump._w_u64_list(f, payload)

    try:
        wr = _time(do_write, max(3, runs // 10), warmup=1)
        nbytes = os.path.getsize(path)
    finally:
        os.unlink(path)
    got["egress_bytes_per_s"] = nbytes / _p99_upper(wr)
    return got


def stages_from_rates(r: dict) -> dict:
    """Per-slot rates x the row capacities the admission model prices."""
    ns = 1e-9
    return {
        "model_load": None,                       # needs the real GGUF
        "semantic_5_active_sweeps": None,         # needs the real GGUF
        "fresh_commit_fold": r["encode_ns_per_slot"] * ns * model.FRESH_ROW_CAPACITY,
        "linear": None,                           # q_lin fold not measured yet
        "quadratic": r["quad_ns_per_product"] * ns * model.QUADRATIC_COUNT_CAP,
        "fresh_hash_coef": r["hash_ns_per_slot"] * ns * model.FRESH_ROW_CAPACITY,
        "persistent_weight_qlin": None,           # q_lin fold not measured yet
        "persistent_open": r["open_ns_per_slot"] * ns * model.WEIGHT_ROW_CAPACITY,
        "fresh_open": r["open_ns_per_slot"] * ns * model.FRESH_ROW_CAPACITY,
        "proof_egress": 95_000_000_000 / r["egress_bytes_per_s"],
        "rtt": None,
        "tail": None,
        "orchestration_refresh": None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.runs < admission.MIN_RUNS:
        print(f"NOTE: {a.runs} runs is below the admission minimum "
              f"({admission.MIN_RUNS}); this cannot become a real report.")

    fp = admission.machine_fingerprint()
    print(f"machine: {fp['gpu_name']}  torch {fp['torch']}  cuda {fp['cuda']}")
    print(f"geometry: {admission.TARGET}\n")

    rates = measure(a.runs)
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
            "bound_kind": "p99_upper",
            "weights": "none_kernel_only",
            "rates": rates,
            "stages": stages,
        }
        pathlib.Path(a.out).write_text(json.dumps(doc, indent=2))
        print(f"\nwrote {a.out} (NOT an admission report: stages are missing "
              f"and weights are not real GGUF)")


if __name__ == "__main__":
    main()
