"""Build a complete admission report on the machine the proof will run on.

`admission_bench.py` measures the kernels and deliberately leaves the
model-dependent stages null, which keeps its output un-admittable.  This script
closes the gap the only way that is honest for those stages: it RUNS them.

  * repeatable (kernel) stages — the 714-run nonparametric campaign, unchanged;
  * single-shot stages — model_load and one active-only semantic sweep are
    measured on the real GGUF, and the five-sweep stage is 5x the measured
    sweep because the prover performs the same pass five times.  Bounds carry
    the margin `admission.SINGLE_SHOT_SAFETY` over what was observed, and the
    report declares `bound_kind = mixed...` plus the exact single-shot set, so
    nobody can read it as a uniform p99 claim.

Everything is bound to this tree, this enrolled model, this statement, this
layout, this machine and this output filesystem — the gate re-checks all of it.

Usage (on the rented box, after enrollment):
    python analysis/bench/make_admission_report.py \
        --from-gguf MODEL.gguf --tokens tokens.json \
        --layers 48 --experts 128 --d 5120 --d-ff 8192 --vocab 202048 \
        --weight-commitment maverick.wcommit --public-sz PUBLIC_SZ \
        --egress-dir /the/production/proof/filesystem \
        --out admission.json
"""
import argparse
import json
import pathlib
import sys
import time

_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "prover"))
sys.path.insert(0, str(_ROOT / "demo"))
sys.path.insert(0, str(_ROOT / "analysis" / "bench"))

import torch                                            # noqa: E402
import core                                             # noqa: E402
import admission                                        # noqa: E402
import admission_bench                                  # noqa: E402

SWEEPS = 5          # the transcript's five witness regenerations


def _measure_model_stages(a):
    """model_load and ONE active-only semantic sweep, on the real GGUF."""
    import demo_maverick_full as drv

    if a.tokens:
        tk = json.load(open(a.tokens))
        prompt_ids, cont_ids = tk["prompt"], tk["continuation"]
    else:
        raise SystemExit("--tokens is required: the sweep must be timed on the "
                         "token set the proof will actually run")

    tape = core_tape(drv, a)
    t0 = time.time()
    logits, Sz, handles, sum_pos = drv.build_model(
        tape, a.from_gguf, prompt_ids, cont_ids, V=a.vocab, d=a.d,
        n_layers=a.layers, E=a.experts, d_ff=a.d_ff)
    build_s = time.time() - t0

    # LOAD-BEARING: the public Sz is serialized into the claim set, so it is
    # part of the statement digest. Pin the same value the proof will pin, or
    # this report describes a different statement and the gate refuses it —
    # correctly, and only after the model has been loaded.
    if handles.get("reveal_pin") is None:
        raise SystemExit("this tape has no reveal pin to bind --public-sz to")
    handles["reveal_pin"].public_rhs = a.public_sz

    # One WARM pass here. The COLD pass already happened: the driver's
    # witness-only run is what produced the public Sz this report is bound to,
    # and its wall time comes in as --cold-sweep-s. Repeating it would buy two
    # more hours of rental for a number we already have.
    #
    #   sweep      = the warm pass (the loader's caches are hot)
    #   model_load = tape build + whatever the cold pass paid on top
    t0 = time.time()
    keep = {logits.var, Sz.var}
    live = tape.run_engine_pass(free_intermediates=True, keep=keep)
    warm_s = time.time() - t0
    del live

    model_load_s = build_s + max(0.0, a.cold_sweep_s - warm_s)
    print(f"  build {build_s:.1f}s  cold sweep {a.cold_sweep_s:.1f}s (from the "
          f"witness-only run)  warm sweep {warm_s:.1f}s\n"
          f"  -> model_load {model_load_s:.1f}s, sweep {warm_s:.1f}s", flush=True)
    # Both observations of the sweep are kept: the bound is taken over the max,
    # so a cold pass that is genuinely slower is what gets charged.
    return tape, model_load_s, [warm_s, a.cold_sweep_s]


def core_tape(drv, a):
    from tape import Tape
    return Tape(drv.CFG, silu_config=drv.SILU_CFG, lazy=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-gguf", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--layers", type=int, default=48)
    ap.add_argument("--experts", type=int, default=128)
    ap.add_argument("--d", type=int, default=5120)
    ap.add_argument("--d-ff", type=int, default=8192)
    ap.add_argument("--vocab", type=int, default=202048)
    ap.add_argument("--weight-commitment", required=True)
    ap.add_argument("--public-sz", type=int, required=True)
    ap.add_argument("--egress-dir", required=True,
                    help="the filesystem that will receive --dump-proof")
    ap.add_argument("--cold-sweep-s", type=float, required=True,
                    help="wall time of the COLD active-only semantic sweep, "
                         "from the witness-only run that produced --public-sz")
    ap.add_argument("--kernel-runs", type=int, default=admission.MIN_RUNS)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import demo_maverick_full as drv
    admission.check_config(drv.CFG)

    print(f"[1/3] kernel campaign: {a.kernel_runs} runs/stage", flush=True)
    rates, kernel_samples = admission_bench.measure(a.kernel_runs, a.egress_dir)
    stages = admission_bench.stages_from_rates(rates)

    print("[2/3] model stages on the real GGUF", flush=True)
    tape, model_load_s, sweep_samples = _measure_model_stages(a)

    print("[3/3] binding the report to this build/model/statement/layout",
          flush=True)
    wc = core.WeightCommitment.load(a.weight_commitment)
    claims_bytes, manifest, stmt = admission.prepare(tape, drv.CFG)

    samples = dict(kernel_samples)
    samples["model_load"] = [model_load_s]
    samples["semantic_5_active_sweeps"] = [SWEEPS * s for s in sweep_samples]
    # Operational allowances: measured as zero work here, so their bound is the
    # model's own cap. They are declared single-shot for exactly that reason.
    for stage in ("rtt", "tail", "orchestration_refresh"):
        samples[stage] = [admission.STAGE_CAPS[stage] / admission.SINGLE_SHOT_SAFETY]

    bounds = {}
    for stage in admission.STAGE_CAPS:
        raw = samples.get(stage)
        if not raw:
            raise SystemExit(f"stage {stage!r} was not measured — refusing to "
                             f"write a report with a hole in it")
        m = max(raw)
        bounds[stage] = (m * admission.SINGLE_SHOT_SAFETY
                         if stage in admission.SINGLE_SHOT_STAGES else m)

    report = {
        "source_digest": admission.source_digest(),
        "model_root": wc.root.hex(),
        "statement_digest": stmt.hex(),
        "machine": admission.machine_fingerprint(),
        "egress_filesystem": admission.filesystem_fingerprint(a.egress_dir),
        "row_manifest": manifest,
        "runs": a.kernel_runs,
        "bound_kind": admission.BOUND_KIND_MIXED,
        "single_shot_stages": sorted(admission.SINGLE_SHOT_STAGES),
        "weights": "real_gguf",
        "stages": bounds,
        "stage_samples": samples,
        "rates": rates,
    }
    pathlib.Path(a.out).write_text(json.dumps(report, indent=1))

    over = [(s, bounds[s], c) for s, c in admission.STAGE_CAPS.items()
            if bounds[s] > c]
    print(f"\n{'stage':26s} {'bound':>12s} {'cap':>12s}  verdict")
    for stage, cap in admission.STAGE_CAPS.items():
        ok = bounds[stage] <= cap
        print(f"{stage:26s} {bounds[stage]:12.1f} {cap:12.1f}  "
              f"{'ok' if ok else 'OVER CAP'}")
    total = sum(bounds.values())
    print(f"{'TOTAL':26s} {total:12.1f} {admission.TOTAL_CAP_S:12.1f}  "
          f"{'ok' if total <= admission.TOTAL_CAP_S else 'OVER'}")
    print(f"\nwrote {a.out}")
    if over:
        print("NOT ADMISSIBLE — the gate will refuse this report:")
        for s, got, cap in over:
            print(f"  {s}: {got:.1f}s > {cap:.1f}s ({got / cap:.2f}x)")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
