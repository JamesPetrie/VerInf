"""VerInf optimization VALIDATION campaign — medium models, one GPU.

Runs the existing A/B scripts across a medium config matrix (via the AB_* env
overrides), plus the soundness gates (byte-identical + Rust ACCEPT), and collects
everything — with the GPU name attached — into remote_results/<host>/. Every A/B
already funnels its result through prod_lens, so no bare medium % escapes.

This is the payload the vast.ai runner executes on the rented box; it also runs
locally (a small-config shakedown on the dev V100) so it can be debugged before
spending money.

Honesty rules (mirrors the skills): a failed/ OOM run is RECORDED, never faked;
the GPU name is attached to every result; ACCEPT is a hard gate, not optional.

Usage:
  # local shakedown (small, fast):
  uv run --project /home/riftuser/VerInf python3 analysis/bench/remote_suite.py --smoke
  # medium campaign (what the rented box runs):
  uv run --project /home/riftuser/VerInf python3 analysis/bench/remote_suite.py \
      --matrix medium --reps 3
"""
from __future__ import annotations
import argparse, json, os, platform, socket, subprocess, sys, time
from pathlib import Path

ROOT = Path("/home/riftuser/VerInf")
BENCH = ROOT / "analysis" / "bench"
# how to invoke python in the VerInf env (uv on both dev box and rented box)
PYRUN = os.environ.get("VERINF_PYRUN", f"uv run --project {ROOT} python3").split()

MATRICES = {
    # (D, DFF, DH, SEQ, NL, VOCAB) — random weights at these SHAPES (no download)
    "smoke":  [(512, 1536, 64, 256, 2, 64)],
    # phone / on-device models (real architecture shapes, synthetic weights):
    "phone":  [(896, 4864, 64, 256, 24, 151936),   # Qwen2.5-0.5B
               (2048, 8192, 64, 256, 16, 128256)],  # Llama-3.2-1B
    "medium": [(1024, 3072, 64, 512, 4, 64),
               (2048, 6144, 128, 512, 4, 64),
               (2048, 6144, 128, 1024, 8, 64),
               (4096, 11008, 128, 512, 4, 64)],   # Llama-7B-ish width
}

# A/B levers: (script, human label). Each already calls prod_lens.report.
AB_LEVERS = [
    ("ab_gpu_softmax.py",  "GPU softmax (transferable witness-compute win)"),
    ("ab_witness_spill.py", "witness spill (host store-once/re-read)"),
]
# Soundness gates — correctness, run ONCE (not per medium config; ACCEPT is about
# the proof being unchanged, independent of model scale).
GATES = [
    ("validate_gpu_softmax.py",  "byte-identical: GPU vs numpy softmax"),
    ("validate_gpu_silu.py",     "byte-identical: GPU vs numpy silu"),
    ("validate_witness_spill.py", "byte-identical: spill vs recompute"),
    ("accept_toy_spill.py",      "Rust verify_proof = ACCEPT (spill on)"),
]


def gpu_name() -> str:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=15)
        if out.returncode == 0:
            return out.stdout.strip().splitlines()[0].strip()
    except Exception:
        pass
    return "unknown-gpu"


def run(script: str, env_extra: dict, timeout: int, log_path: Path) -> dict:
    """Run one bench script, tee output to log_path, return status + tail."""
    env = {**os.environ, **{k: str(v) for k, v in env_extra.items()}}
    t0 = time.time()
    try:
        p = subprocess.run(PYRUN + [str(BENCH / script)], cwd=str(ROOT), env=env,
                           capture_output=True, text=True, timeout=timeout)
        out = p.stdout + "\n" + p.stderr
        status = "ok" if p.returncode == 0 else f"exit{p.returncode}"
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "") + "\nTIMEOUT"
        status = "timeout"
    except Exception as e:  # OOM launcher errors etc. — record, never fake
        out = f"LAUNCH-ERROR: {e}"
        status = "error"
    log_path.write_text(out)
    dur = time.time() - t0
    # pull the lines that matter (prod_lens projection, ACCEPT/identical verdicts)
    keep = [ln for ln in out.splitlines()
            if any(k in ln for k in ("PRODUCTION PROJECTION", "ACCEPT", "REJECT",
                                     "IDENTICAL", "MISMATCH", "faster", "prove ",
                                     "OutOfMemory", "CUDA out of memory", "TIMEOUT"))]
    return {"script": script, "status": status, "dur_s": round(dur, 1),
            "highlights": keep[-12:]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrix", choices=list(MATRICES), default="medium")
    ap.add_argument("--smoke", action="store_true", help="alias for --matrix smoke")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--ab-timeout", type=int, default=3600, help="per-A/B timeout (s)")
    ap.add_argument("--skip-gates", action="store_true", help="skip soundness gates (NOT for a real campaign)")
    args = ap.parse_args()
    matrix = "smoke" if args.smoke else args.matrix

    host = socket.gethostname()
    gpu = gpu_name()
    outdir = BENCH / "remote_results" / host
    outdir.mkdir(parents=True, exist_ok=True)
    results = {"host": host, "gpu": gpu, "matrix": matrix, "reps": args.reps,
               "python": platform.python_version(), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "gates": [], "ab": []}
    print(f"=== VerInf validation campaign ===\n host={host} gpu={gpu} matrix={matrix} reps={args.reps}\n")

    # 1. soundness gates first — if these fail, the optimizations are unsound and
    #    the timing numbers below are worthless. Run once.
    if not args.skip_gates:
        print("--- soundness gates (byte-identical + Rust ACCEPT) ---")
        for script, label in GATES:
            r = run(script, {}, timeout=1200, log_path=outdir / f"gate_{script}.log")
            ok = r["status"] == "ok" and not any("MISMATCH" in h or "REJECT" in h for h in r["highlights"])
            r["label"] = label; r["pass"] = ok
            results["gates"].append(r)
            print(f"  [{'PASS' if ok else 'FAIL'}] {label}  ({r['status']}, {r['dur_s']}s)")
        if not all(g["pass"] for g in results["gates"]):
            print("\n!! a soundness gate FAILED — timing results are not trustworthy; "
                  "recording and continuing so the failure is on the record.")

    # 2. A/B levers across the matrix
    for (D, DFF, DH, SEQ, NL, VOCAB) in MATRICES[matrix]:
        ell = 512                       # Ligero ELL must be >= d (embedding lookup)
        while ell < D:
            ell *= 2
        env = dict(AB_D=D, AB_DFF=DFF, AB_DH=DH, AB_SEQ=SEQ, AB_NL=NL, AB_VOCAB=VOCAB,
                   AB_ELL=ell, AB_REPS=args.reps, LIGERO_PHASE_TIMING=1)
        tag = f"d{D}_ff{DFF}_seq{SEQ}_L{NL}_v{VOCAB}"
        print(f"\n--- config {tag} ---")
        for script, label in AB_LEVERS:
            r = run(script, env, timeout=args.ab_timeout, log_path=outdir / f"ab_{script}_{tag}.log")
            r["label"] = label; r["config"] = tag
            results["ab"].append(r)
            print(f"  [{r['status']:>7}] {label}  ({r['dur_s']}s)")
            for h in r["highlights"]:
                print(f"        {h}")

    # 3. write machine-readable + human summary
    (outdir / "campaign_results.json").write_text(json.dumps(results, indent=2))
    _write_summary(outdir / "SUMMARY.md", results)
    # carry the run log along if present
    jl = BENCH / "prove_runs.jsonl"
    if jl.exists():
        (outdir / "prove_runs.jsonl").write_bytes(jl.read_bytes())
    print(f"\n=== done. results in {outdir} ===")


def _write_summary(path: Path, r: dict):
    L = [f"# VerInf validation campaign — {r['gpu']}",
         f"host `{r['host']}` · matrix `{r['matrix']}` · reps {r['reps']} · {r['utc']}",
         "", "## Soundness gates (must all PASS or the timings are void)", ""]
    for g in r["gates"]:
        L.append(f"- {'✅' if g.get('pass') else '❌'} {g['label']} — {g['status']}")
    L += ["", "## A/B levers at medium scale", ""]
    for a in r["ab"]:
        L.append(f"### {a['label']} @ `{a['config']}` — {a['status']} ({a['dur_s']}s)")
        for h in a["highlights"]:
            L.append(f"    {h}")
        L.append("")
    L += ["> GPU-attached; every % above is paired with its prod_lens 400B projection.",
          "> A failed/OOM/timeout run is recorded as-is, never fabricated."]
    path.write_text("\n".join(L))


if __name__ == "__main__":
    main()
