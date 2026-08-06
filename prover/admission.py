"""Fail-closed admission gate.

A 400B proof is a ~4-hour, single-shot commitment of a rented machine. The
point of this module is that the run is REFUSED before it starts unless
somebody measured this build, on this machine, on this geometry, and every
stage came in under its cap.

What the gate refuses, and why each one matters:

  * a report from a different source tree — the measured code is not the code
    about to run;
  * a report for a different model root or statement — the numbers describe
    another proof;
  * a report from a different machine — the single most common way a good
    benchmark becomes a bad prediction;
  * a report whose row manifest differs from the layout just built — a smaller
    ELL/N geometry or a shorter context measures a different amount of work;
  * fewer than 714 runs per stage, or a bound that is not an upper confidence
    bound — an average is not an admission argument;
  * any stage over its cap in analysis/routed_projected_4h_model.py;
  * a report that admits it used random weights — the semantic sweep cap
    includes real GGUF decode and page migration.

Caps come from the model file itself, so there is exactly one place where the
4-hour envelope is written down.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import pathlib
import platform
import subprocess
from typing import Dict

_ANALYSIS = pathlib.Path(__file__).resolve().parents[1] / "analysis"

# The target Ligero geometry. A production proof runs at exactly this config:
# the admission model prices these row counts, and a smaller ELL/N or a lower
# query count is a different (cheaper, weaker) proof.
TARGET = dict(ELL=8192, K_DEG=16384, N_LIG=65536, T_QUERIES=54)

# Stage caps, in seconds, straight out of the admission model.
STAGE_CAPS = {
    "model_load": 400.0,
    "semantic_5_active_sweeps": 3609.0,
    "fresh_commit_fold": 950.0,
    "linear": 25.6,
    "quadratic": 765.0,
    "fresh_hash_coef": 140.0,
    "persistent_weight_qlin": 3624.522,
    "persistent_open": 1812.261,
    "fresh_open": 450.0,
    "proof_egress": 879.63,
    "rtt": 80.0,
    "tail": 20.0,
    "orchestration_refresh": 600.0,
}
TOTAL_CAP_S = 14_400.0
# Distribution-free simultaneous tolerance bound.  If each stage cap is the
# observed maximum of n independent runs, the probability that it lies below
# the true 99th percentile is 0.99**n.  Bonferroni across all priced stages
# requires len(STAGE_CAPS)*0.99**n <= 0.01, hence 714 (not 30).  A smaller
# exploratory probe may guide optimization but cannot authorize a 4-hour run.
MIN_RUNS = math.ceil(math.log(0.01 / len(STAGE_CAPS)) / math.log(0.99))
BOUND_KIND = "simultaneous_nonparametric_p99_max"
# Compact u64le/base64 proof is ~35--45 GB at the target manifest.  Reserve a
# deliberately larger inode before proving so ENOSPC cannot be discovered four
# hours later.  The writer truncates it to the actual length before rename.
PROOF_RESERVE_BYTES = 64_000_000_000


class AdmissionError(SystemExit):
    """Refusal to start. SystemExit so a driver aborts loudly."""


def check_config(cfg, *, allow_dev: bool = False) -> None:
    got = dict(ELL=cfg.ELL, K_DEG=cfg.K_DEG, N_LIG=cfg.N_LIG,
               T_QUERIES=cfg.T_QUERIES)
    if got == TARGET:
        return
    msg = (f"Ligero config {got} is not the target {TARGET}. "
           f"The admission envelope and the soundness budget are both stated "
           f"for the target geometry.")
    if not allow_dev:
        raise AdmissionError("refusing to run: " + msg)
    print(f"  [admission] DEV CONFIG: {msg}", flush=True)


def source_digest() -> str:
    """A digest of the prover + verifier source actually present.

    Tracked files only (git), so an untracked scratch file does not change it;
    if git is unavailable the tree is hashed directly."""
    root = pathlib.Path(__file__).resolve().parents[1]
    try:
        files = subprocess.run(
            ["git", "-C", str(root), "ls-files", "prover", "verifier", "demo",
             "analysis/bench", "analysis/routed_projected_4h_model.py"],
            capture_output=True, text=True, check=True).stdout.split()
    except Exception:
        suffixes = {".py", ".rs", ".cu", ".cuh", ".toml", ".lock", ".sh"}
        files = sorted(
            str(p.relative_to(root)) for p in root.rglob("*")
            if p.is_file() and p.suffix in suffixes and "deprecated" not in str(p))
    h = hashlib.sha256()
    for rel in sorted(files):
        f = root / rel
        if not f.is_file():
            continue
        h.update(rel.encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def machine_fingerprint() -> Dict[str, str]:
    import torch
    props = torch.cuda.get_device_properties(0)
    out = {
        "gpu_name": props.name,
        "gpu_total_mem": str(props.total_memory),
        "gpu_count": str(torch.cuda.device_count()),
        "gpu_capability": f"{props.major}.{props.minor}",
        "cuda": torch.version.cuda or "",
        "torch": torch.__version__,
        "python": platform.python_version(),
        "hostname": platform.node(),
    }
    try:
        smi = subprocess.run([
            "nvidia-smi", "--query-gpu=uuid,driver_version,pci.bus_id,power.limit",
            "--format=csv,noheader,nounits", "-i", "0"], capture_output=True,
            text=True, check=True).stdout.strip().split(", ")
        out.update(gpu_uuid=smi[0], driver=smi[1], gpu_pci=smi[2],
                   power_limit_w=smi[3])
    except Exception:
        out.update(gpu_uuid="", driver="", gpu_pci="", power_limit_w="")
    return out


def filesystem_fingerprint(path: str) -> Dict[str, str]:
    """Bind the egress measurement to the filesystem used for the proof."""
    p = pathlib.Path(path).resolve()
    target = p if p.exists() and p.is_dir() else p.parent
    while not target.exists() and target != target.parent:
        target = target.parent
    st = os.stat(target)
    out = {"st_dev": str(st.st_dev)}
    try:
        fields = subprocess.run(
            ["findmnt", "-n", "-o", "SOURCE,FSTYPE", "--target", str(target)],
            capture_output=True, text=True, check=True).stdout.strip().split(None, 1)
        out["source"] = fields[0] if fields else ""
        out["fstype"] = fields[1] if len(fields) > 1 else ""
    except Exception:
        out.update(source="", fstype="")
    return out


def row_manifest(tape, cfg) -> Dict[str, int]:
    """The layout the proof will actually commit — not a projection of it."""
    import core
    claims = core._with_synthesized_settlements(tape.claims)
    (_all, p1, p2, p3, m_p1, m_p2, m_total,
     _wv, m_w, _wn, m_wnew) = core._layout(claims, cfg)
    rows = lambda vs: sum(v.n_rows(cfg.ELL) for v in vs)
    return {
        "m_total": m_total, "m_w": m_w, "m_wnew": m_wnew,
        "n_p1": rows(p1), "n_p2": rows(p2), "n_p3": rows(p3),
        "n_claims": len(tape.claims),
        "ELL": cfg.ELL, "N_LIG": cfg.N_LIG, "T_QUERIES": cfg.T_QUERIES,
    }


def prepare(tape, cfg):
    """Fix the layout, then derive everything the statement depends on.

    ORDER IS LOAD-BEARING: a Variable's row_start is -1 until _layout assigns
    it, so serializing the claims before the layout produces a claim set the
    verifier cannot compile. Everything downstream (the canonical bytes, the
    digest the gate binds to, the digest the proof carries) has to come from
    the same, already-laid-out tape.

    Returns (claims_bytes, row_manifest, statement_digest).
    """
    import protocol as pr
    manifest = row_manifest(tape, cfg)             # runs _layout, assigns rows
    claims_bytes = pr.claims_canonical_bytes(tape.claims, cfg)
    return claims_bytes, manifest, statement_digest_for(tape, cfg, claims_bytes)


def statement_digest_for(tape, cfg, claims_bytes: bytes) -> bytes:
    """The digest the proof will carry — computed here so the gate can bind to
    it before a single round runs."""
    import core
    import protocol as pr
    claims = core._with_synthesized_settlements(tape.claims)
    (_a, _p1, _p2, p3, _m1, _m2, _mt, _wv, m_w, _wn, m_wnew) = core._layout(claims, cfg)
    blocks = (["blind"] + (["w"] if m_w else []) + (["wnew"] if m_wnew else [])
              + ["p1", "p2"]
              + (["p3"] if sum(v.n_rows(cfg.ELL) for v in p3) else []))
    return pr.statement_digest(claims_bytes, blocks)


def load_report(path: str) -> dict:
    with open(path) as f:
        report = json.load(f)
    for key in ("source_digest", "model_root", "statement_digest", "machine",
                "row_manifest", "runs", "bound_kind", "weights", "stages",
                "stage_samples", "egress_filesystem"):
        if key not in report:
            raise AdmissionError(
                f"admission report {path} has no '{key}'. It must bind the "
                f"measurement to the build, the model, the statement, the "
                f"machine and the geometry it was taken on.")
    return report


def check(report: dict, *, cfg, model_root: bytes, statement_digest: bytes,
          manifest: Dict[str, int], output_path: str = ".") -> None:
    fails = []

    def need(cond, msg):
        if not cond:
            fails.append(msg)

    need(report["source_digest"] == source_digest(),
         "source digest differs: the benchmark measured a different build")
    need(report["model_root"] == model_root.hex(),
         f"model root differs: report {report['model_root'][:16]}… vs enrolled "
         f"{model_root.hex()[:16]}…")
    need(report["statement_digest"] == statement_digest.hex(),
         "statement digest differs: the report describes another proof")

    here = machine_fingerprint()
    for k, v in here.items():
        need(report["machine"].get(k) == v,
             f"machine mismatch on {k}: report {report['machine'].get(k)!r} vs "
             f"this box {v!r}")

    report_fs = report["egress_filesystem"]
    if not isinstance(report_fs, dict):
        fails.append("egress_filesystem must be an object")
        report_fs = {}
    here_fs = filesystem_fingerprint(output_path)
    for k, v in here_fs.items():
        need(report_fs.get(k) == v,
             f"egress filesystem mismatch on {k}: report "
             f"{report_fs.get(k)!r} vs output {v!r}")

    for k, v in manifest.items():
        need(report["row_manifest"].get(k) == v,
             f"row manifest mismatch on {k}: report "
             f"{report['row_manifest'].get(k)} vs this layout {v}")

    try:
        n_runs = int(report["runs"])
    except (TypeError, ValueError):
        n_runs = -1
    need(n_runs >= MIN_RUNS,
         f"{report['runs']!r} runs per stage, need >= {MIN_RUNS}")
    need(report["bound_kind"] == BOUND_KIND,
         f"bound_kind is {report['bound_kind']!r}; admission needs simultaneous "
         f">=99% distribution-free p99 bounds ({BOUND_KIND!r}), not averages")
    need(report["weights"] == "real_gguf",
         f"weights are {report['weights']!r}; the semantic cap includes real "
         f"GGUF decode and page migration, so random weights cannot satisfy it")

    sample_map = report["stage_samples"]
    if not isinstance(sample_map, dict):
        fails.append("stage_samples must be an object of raw sample lists")
        sample_map = {}
    total = 0.0
    for stage, cap in STAGE_CAPS.items():
        if stage not in report["stages"]:
            fails.append(f"stage '{stage}' is not in the report")
            continue
        try:
            got = float(report["stages"][stage])
        except (TypeError, ValueError):
            fails.append(f"stage '{stage}' is not a numeric bound")
            continue
        need(math.isfinite(got) and got >= 0,
             f"stage '{stage}' must be finite and non-negative, got {got!r}")
        if not math.isfinite(got) or got < 0:
            continue
        samples = sample_map.get(stage)
        if not isinstance(samples, list):
            fails.append(f"stage '{stage}' has no raw sample list")
            continue
        if len(samples) < MIN_RUNS:
            fails.append(
                f"stage '{stage}' has {len(samples)} raw samples, need >= {MIN_RUNS}")
            continue
        try:
            raw = [float(x) for x in samples]
        except (TypeError, ValueError):
            fails.append(f"stage '{stage}' has a nonnumeric raw sample")
            continue
        if not all(math.isfinite(x) and x >= 0 for x in raw):
            fails.append(
                f"stage '{stage}' raw samples must be finite and non-negative")
            continue
        need(got >= max(raw),
             f"stage '{stage}' bound {got:.6f}s is below observed max "
             f"{max(raw):.6f}s")
        total += got
        need(got <= cap, f"stage '{stage}': {got:.3f}s over its cap {cap:.3f}s")
    extra = set(report["stages"]) - set(STAGE_CAPS)
    need(not extra, f"unpriced stages in the report: {sorted(extra)}")
    extra_samples = set(sample_map) - set(STAGE_CAPS)
    need(not extra_samples,
         f"unpriced stage samples in the report: {sorted(extra_samples)}")
    need(total <= TOTAL_CAP_S,
         f"stages sum to {total:.1f}s, over the {TOTAL_CAP_S:.0f}s envelope")

    if fails:
        raise AdmissionError(
            "ADMISSION REFUSED — the run would not be a controlled one:\n  - "
            + "\n  - ".join(fails))
    print(f"  [admission] all {len(STAGE_CAPS)} stages under cap; "
          f"total {total:.1f}s of {TOTAL_CAP_S:.0f}s", flush=True)
