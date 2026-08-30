"""One-run machine-profile calibration: profiler roadmap item 3.

Fills a `profiler/machines/<name>.json` (the _template-blackwell.json schema)
by measuring the box it runs on:

  CUDA microbenches — the four spark-bench sources compiled on the fly
  (sources untouched; headers resolve from prover/kernels, arch is DETECTED,
  not the Makefile's hardcoded sm_121 — wrong-arch builds run and print
  wrong numbers, per that Makefile's own warning), plus one profiler-authored
  bench; each isolated, one failure skips one bench:
      bench_field_mul          -> gpu.field_mul_Gps
      bench_ntt                -> gpu.ntt_ns_per_elem  (best variant, n=65536)
      bench_blake3_columns     -> gpu.blake3_compress_Gps, gpu.blake3_bulk_GBps
      bench_goldilocks_matmul  -> provenance note only (informational)
      bench_blake3_reg (ours)  -> gpu.blake3_reg_compress_Gps  (ALU-bound
                                  compress rate — B's proper scaling basis)
      bench_ntt_batched (ours) -> gpu.ntt_batched_ns_per_elem  (the
                                  prover-path NTT; single-transform is
                                  launch-bound on large-L2 parts)
      bench_hbm_random (ours)  -> gpu.hbm_random_GBps, gpu.hbm_chase_ns
                                  (challenge-protocol + profile numbers)
      bench_launch_latency (ours) -> gpu.launch_us_sync/_stream

  torch benches (skipped when torch is absent OR CPU-only):
      device-to-device copy -> gpu.mem_bandwidth_GBps  (counts read+write,
                               matching the gb10-spark provenance convention)
      pinned host-to-device -> io.h2d_GBps

  plain-Python benches (unique temp files, free-space preflight, removed on
  every path):
      write+evict+read      -> io.disk_read_GBps  (posix_fadvise DONTNEED
                               after fsync, so reads hit the device — still
                               not O_DIRECT-cold)
      JSON string-build+write proxy -> io.proof_dump_MBps  (PROXY — the real
                               dump is prover/proof_dump.py; recalibrate from
                               the first real dump on this hardware)

prove_constants: A/C ride memory bandwidth so they are RATIO-DERIVED from
gb10-spark by the measured bandwidth ratio — clearly labeled in provenance,
to be replaced by validation-mode recalibration (README roadmap 2). B is
ALU-bound (one challenge compress per cid): ratioed against the base box's
register-resident rate when that baseline exists, else DIRECT (1/measured
Gc-per-s — keeps the floor computable, since predict/partition require all
of A/B/C; leans floor-ward). It goes null only if bench_blake3_reg fails,
and the provenance then says the floor is blocked. aggregate is left null:
it bakes in code-level overheads that do not transfer. --no-derive leaves
all four null.

    python3 profiler/calibrate.py --name b200-node
    python3 profiler/calibrate.py --name gb200-nvl72-partition --disk-gb 32

Run on the target box. Needs nvcc + a CUDA torch for the full set; anything
missing is skipped with the field left null (predictions then say so).
Raw bench outputs land next to the profile for provenance.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_PROFILER = Path(__file__).resolve().parent
_REPO = _PROFILER.parent
sys.path.insert(0, str(_PROFILER))

from machine import MachineProfile, MACHINES_DIR    # noqa: E402

BENCH_DIR = _REPO / "prover" / "deprecated" / "spark-bench" / "bench"
PROF_BENCH_DIR = _PROFILER / "bench"      # profiler-authored benches
KERNELS_DIR = _REPO / "prover" / "kernels"

# ------------------------------------------------------------- parsers
# Pure functions against the benches' exact printf formats — unit-tested
# without CUDA in test_calibration_tools.py.

_FIELD_RE = re.compile(r"best:\s*([\d.]+)\s*Gmul/s")
_NTT_RE = re.compile(r"n=\s*(\d+)\s+log2n=\s*\d+\s+(\S+)\s+fwd\+inv x \d+"
                     r"\s+total=\s*[\d.]+\s*ms\s+->\s+([\d.]+)\s+us/NTT")
_B3_RE = re.compile(r"m=\s*(\d+)\s+cols=\s*[\d,]+.*->\s*([\d.]+)\s+Mcols/s"
                    r"\s+([\d.]+)\s+Gcompress/s\s+([\d.]+)\s+GB/s")
_MM_RE = re.compile(r"n=\s*(\d+)\s+ops=\S+\s+time/run=\s*[\d.]+\s*ms"
                    r"\s+throughput=\s*([\d.]+)\s+Gmul/s")
_REG_RE = re.compile(r"best:\s*([\d.]+)\s*Gcompress/s \(register-resident\)")
_NTTB_RE = re.compile(r"n=\s*(\d+)\s+m=\s*(\d+).*->\s*[\d.]+\s*us/NTT"
                      r"\s+([\d.]+)\s+ns/elem")
_GATHER_RE = re.compile(r"gather best:\s*([\d.]+)\s*GB/s")
_CHASE_RE = re.compile(r"chase best:\s*([\d.]+)\s*ns/hop")
_LSYNC_RE = re.compile(r"sync:\s*([\d.]+)\s*us/launch")
_LSTREAM_RE = re.compile(r"stream:\s*([\d.]+)\s*us/launch")


def parse_field_mul(out: str):
    m = _FIELD_RE.search(out)
    return float(m.group(1)) if m else None


def parse_matmul(out: str):
    """Best sustained Gmul/s across the matmul n sweep (peaks at large n)."""
    rows = _MM_RE.findall(out)
    return max(float(r[1]) for r in rows) if rows else None


def parse_blake3_reg(out: str):
    m = _REG_RE.search(out)
    return float(m.group(1)) if m else None


def parse_ntt_batched(out: str, n: int = 65536):
    """Best (min) ns/elem across the batch sweep at transform size n —
    the prover-path number; single-transform bench_ntt stays as the
    launch-overhead probe."""
    rows = [float(ns) for nn, m, ns in _NTTB_RE.findall(out)
            if int(nn) == n]
    return min(rows) if rows else None


def parse_hbm_random(out: str):
    """(random-gather GB/s, chase ns/hop)."""
    g = _GATHER_RE.search(out)
    c = _CHASE_RE.search(out)
    return (float(g.group(1)) if g else None,
            float(c.group(1)) if c else None)


def parse_launch(out: str):
    """(synced us/launch, streamed us/launch)."""
    a = _LSYNC_RE.search(out)
    b = _LSTREAM_RE.search(out)
    return (float(a.group(1)) if a else None,
            float(b.group(1)) if b else None)


def parse_ntt(out: str, n: int = 65536):
    """ns/elem from the best variant at transform size n (the prover's
    N_LIG). Variant preference mirrors the prover: bailey > fused > base."""
    rank = {"bailey": 0, "fused": 1, "baseline": 2}
    rows = [(rank.get(v, 9), float(us)) for nn, v, us in _NTT_RE.findall(out)
            if int(nn) == n]
    if not rows:
        return None
    us = min(rows)[1]
    return us * 1000.0 / n


def parse_blake3(out: str):
    """(best Gcompress/s, best GB/s absorbed) across the m sweep."""
    rows = _B3_RE.findall(out)
    if not rows:
        return None, None
    return (max(float(r[2]) for r in rows), max(float(r[3]) for r in rows))


# ------------------------------------------------------- gpu detection

def detect_gpu():
    """-> dict(name, count, mem_GB, arch) via torch, else nvidia-smi."""
    try:
        import torch
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            return dict(name=p.name, count=torch.cuda.device_count(),
                        mem_GB=round(p.total_memory / 2**30),
                        arch=f"sm_{p.major}{p.minor}")
    except Exception:
        pass
    smi = shutil.which("nvidia-smi")
    if smi:
        r = subprocess.run([smi, "--query-gpu=name,compute_cap,memory.total",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True)
        lines = [l for l in r.stdout.strip().splitlines() if l.strip()]
        if lines:
            name, cap, mem = [x.strip() for x in lines[0].split(",")]
            return dict(name=name, count=len(lines),
                        mem_GB=round(float(mem) / 1024),
                        arch="sm_" + cap.replace(".", ""))
    return None


# ------------------------------------------------------- cuda benches

def compile_bench(src: Path, build: Path, arch: str, nvcc: str):
    exe = build / src.stem
    cmd = [nvcc, f"-arch={arch}", "-std=c++17", "-O3",
           f"-I{KERNELS_DIR}", str(src), "-o", str(exe)]
    print(f"  nvcc {src.name} [{arch}]")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stderr[-3000:])
        raise RuntimeError(f"compile failed: {src.name}")
    return exe


def run_bench(exe: Path, args: list, raw_dir: Path, timeout: int = 1800):
    r = subprocess.run([str(exe)] + args, capture_output=True, text=True,
                       timeout=timeout)
    (raw_dir / f"{exe.name}.txt").write_text(r.stdout + r.stderr)
    if r.returncode != 0:
        sys.stderr.write(r.stdout[-1500:] + r.stderr[-1500:])
        raise RuntimeError(f"{exe.name} exited {r.returncode}")
    return r.stdout


# ------------------------------------------------------ torch benches

def _cuda_time(fn, warmup=3, runs=10):
    """Best-of wall time for fn() on the CUDA stream, seconds."""
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(runs):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) / 1e3)
    return best


def bench_membw_GBps(frac_of_mem: float = 0.05):
    """Device-to-device copy; counts read+write (2x bytes) — same convention
    as the gb10-spark provenance number (223 on a 273 GB/s-spec part; a
    single-sided count would exceed spec)."""
    import torch
    total = torch.cuda.get_device_properties(0).total_memory
    n = min(2 * 2**30, int(total * frac_of_mem))
    a = torch.empty(n, dtype=torch.uint8, device="cuda")
    b = torch.empty(n, dtype=torch.uint8, device="cuda")
    t = _cuda_time(lambda: b.copy_(a))
    del a, b
    torch.cuda.empty_cache()
    return 2 * n / t / 1e9


def bench_h2d_GBps():
    import torch
    n = 1 * 2**30
    h = torch.empty(n, dtype=torch.uint8, pin_memory=True)
    d = torch.empty(n, dtype=torch.uint8, device="cuda")
    t = _cuda_time(lambda: d.copy_(h, non_blocking=False))
    del h, d
    torch.cuda.empty_cache()
    return n / t / 1e9


# ----------------------------------------------------- python benches

def _preflight_space(tmpdir: Path, need_bytes: int, what: str):
    free = shutil.disk_usage(tmpdir).free
    if free < need_bytes * 1.1:
        raise RuntimeError(
            f"{what}: needs ~{need_bytes / 2**30:.1f} GiB free in {tmpdir}, "
            f"only {free / 2**30:.1f} GiB available")


def bench_disk_read_GBps(tmpdir: Path, size_gb: int):
    """Write size_gb to a unique temp file, EVICT it from page cache
    (fsync + posix_fadvise DONTNEED), then read it back in 64 MiB chunks.
    Eviction of clean pages makes the read hit the device on Linux; it is
    still not O_DIRECT-cold (readahead applies — which is also true of the
    prover's real streaming reads). The file is removed on every path."""
    _preflight_space(tmpdir, size_gb * 2**30, "disk probe")
    fd, name = tempfile.mkstemp(dir=tmpdir, prefix="calibrate-disk-",
                                suffix=".bin")
    try:
        chunk = os.urandom(64 * 2**20)
        with os.fdopen(fd, "wb") as fh:
            for _ in range(size_gb * 16):
                fh.write(chunk)
            fh.flush()
            os.fsync(fh.fileno())
            if hasattr(os, "posix_fadvise"):
                os.posix_fadvise(fh.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        t0 = time.perf_counter()
        n = 0
        with open(name, "rb") as fh:
            while True:
                buf = fh.read(64 * 2**20)
                if not buf:
                    break
                n += len(buf)
        dt = time.perf_counter() - t0
        return n / dt / 1e9
    finally:
        try:
            os.unlink(name)
        except OSError:
            pass


def bench_dump_compact_MBps(tmpdir: Path, mb: int = 2048):
    """The PRODUCTION dump path: u64le/base64 with minimal JSON framing
    (the transport of proof files on current main), written to tmpdir.
    Reuses one random chunk — encode cost is content-independent."""
    import base64
    _preflight_space(tmpdir, int(mb * 2**20 * 1.5), "compact dump probe")
    chunk_mb = min(64, mb)
    raw = os.urandom(chunk_mb * 2**20)
    fd, name = tempfile.mkstemp(dir=tmpdir, prefix="calibrate-dumpc-",
                                suffix=".b64")
    try:
        t0 = time.perf_counter()
        with os.fdopen(fd, "wb") as fh:
            fh.write(b"[\"")
            for _ in range(max(1, mb // chunk_mb)):
                fh.write(base64.b64encode(raw))
            fh.write(b"\"]")
            fh.flush()
            os.fsync(fh.fileno())
        dt = time.perf_counter() - t0
        return os.stat(name).st_size / dt / 1e6
    finally:
        try:
            os.unlink(name)
        except OSError:
            pass


def bench_dump_proxy_MBps(tmpdir: Path, n_vals: int = 20_000_000):
    """PROXY for the proof dump: JSON string-building over u64 arrays plus
    the write, the same shape of work proof_dump.py does (the gb10 number is
    'Python string-building bound'). Replace with a real-dump measurement
    when one exists on this hardware."""
    import random
    _preflight_space(tmpdir, n_vals * 21, "dump proxy")
    rng = random.Random(7)
    rows = [[rng.getrandbits(64) for _ in range(1000)]
            for _ in range(n_vals // 1000)]
    fd, name = tempfile.mkstemp(dir=tmpdir, prefix="calibrate-dump-",
                                suffix=".json")
    try:
        t0 = time.perf_counter()
        with os.fdopen(fd, "w") as fh:
            fh.write("[")
            for i, r in enumerate(rows):
                fh.write(("," if i else "") + json.dumps(r))
            fh.write("]")
        dt = time.perf_counter() - t0
        return os.stat(name).st_size / dt / 1e6
    finally:
        try:
            os.unlink(name)
        except OSError:
            pass


# ------------------------------------------------------------- deriv

def derive_prove_constants(base: MachineProfile, membw, blake3_reg):
    """A/C scale with memory bandwidth, B with compute (the bracket's
    scaling story). Returns (constants dict, provenance dict).

    B's basis must be the REGISTER-RESIDENT compress rate: the per-cid
    challenge hash is ALU-bound, and the column-hash bench's Gcompress rate
    is memory-bandwidth-limited at large m (spark-microbench-results.md),
    so it cannot carry the scaling. Preference order: ratio against the
    base box's own register-resident number when it exists; else DIRECT —
    one compress per cid at this box's measured rate (1/Gc-per-s ns), so
    the floor bracket stays computable (predict/partition need all of
    A/B/C). Expect DIRECT to lean floor-ward structurally: it prices the
    raw ALU compress alone, while a calibrated B carries whatever launch/
    orchestration margin the real per-cid path has (unquantified until the
    reg bench runs on a box with a calibrated B). Null only when the reg
    bench produced nothing."""
    consts, prov = {}, {}
    b_bw = base.get("gpu", "mem_bandwidth_GBps")
    b_reg = base.get("gpu", "blake3_reg_compress_Gps")
    if membw and b_bw:
        r = b_bw / membw
        for k in ("A_ns_per_slot", "C_ns_per_product"):
            consts[k] = round(base.get("prove_constants", k) * r, 3)
            prov[k] = (f"DERIVED from {base.name} x bandwidth ratio "
                       f"({b_bw}/{membw:.0f} GB/s) — replace via "
                       f"validation-mode recalibration")
    if blake3_reg and b_reg:
        r = b_reg / blake3_reg
        consts["B_ns_per_cid"] = round(
            base.get("prove_constants", "B_ns_per_cid") * r, 4)
        prov["B_ns_per_cid"] = (f"DERIVED from {base.name} x register-"
                                f"resident compress ratio ({b_reg}/"
                                f"{blake3_reg:.1f} Gc/s) — replace via "
                                f"validation-mode recalibration")
    elif blake3_reg:
        consts["B_ns_per_cid"] = round(1.0 / blake3_reg, 4)
        prov["B_ns_per_cid"] = (
            f"DIRECT: one challenge compress per cid at the measured "
            f"register-resident rate ({blake3_reg:.1f} Gc/s); no "
            f"cross-machine baseline yet — run bench_blake3_reg on "
            f"{base.name} to upgrade to a ratioed value. Leans floor-ward "
            f"structurally: raw ALU compress only, no launch/orchestration "
            f"margin — replace via validation-mode recalibration")
    else:
        prov["B_ns_per_cid"] = (
            "left null: B is ALU-bound (one challenge compress per cid) and "
            "the column-hash bench is memory-bandwidth-limited, so its rate "
            "cannot scale B; bench_blake3_reg produced no measurement on "
            "this box. NOTE: predict/partition need A, B, AND C for the "
            "floor — fix the reg bench or fill B by hand before step 4 of "
            "the runbook")
    prov["aggregate_ns_per_slot"] = ("left null: bakes in code-level "
                                     "overhead that does not transfer; "
                                     "needs a measured run on this box")
    return consts, prov


# ------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="fill a machine profile by "
                                             "benchmarking this box")
    ap.add_argument("--name", required=True,
                    help="profile name, e.g. b200-node (file goes to "
                         "profiler/machines/<name>.json)")
    ap.add_argument("--out", default=None, help="override output path")
    ap.add_argument("--nvcc", default=None)
    ap.add_argument("--arch", default=None,
                    help="override detected arch, e.g. sm_100")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--disk-gb", type=int, default=8)
    ap.add_argument("--dump-compact-mb", type=int, default=2048,
                    help="payload MB for the u64le/base64 compact-dump bench")
    ap.add_argument("--dump-vals", type=int, default=20_000_000,
                    help="values in the dump proxy (~20 bytes/value of JSON)")
    ap.add_argument("--tmpdir", default=".",
                    help="where probe files land — must be a REAL disk, not "
                         "tmpfs, or the disk/dump numbers measure RAM")
    ap.add_argument("--skip-cuda", action="store_true")
    ap.add_argument("--skip-disk", action="store_true")
    ap.add_argument("--no-derive", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)

    out_path = Path(a.out) if a.out else Path(MACHINES_DIR) / f"{a.name}.json"
    if out_path.exists() and not a.force:
        ap.error(f"{out_path} exists — pass --force to overwrite")
    raw_dir = out_path.parent / f"calibrate-raw-{a.name}"
    raw_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.date.today().isoformat()
    prov = {"_suite": f"profiler/calibrate.py run {today}; raw outputs in "
                      f"{raw_dir.name}/"}

    gpu_info = detect_gpu()
    if gpu_info:
        print(f"gpu: {gpu_info['name']} x{gpu_info['count']} "
              f"{gpu_info['mem_GB']} GB [{gpu_info['arch']}]")
    else:
        print("no GPU detected — CUDA and torch benches will be skipped")
    arch = a.arch or (gpu_info or {}).get("arch")

    gpu = dict(count=(gpu_info or {}).get("count"),
               mem_GB=(gpu_info or {}).get("mem_GB"),
               mem_bandwidth_GBps=None, field_mul_Gps=None,
               blake3_compress_Gps=None, blake3_bulk_GBps=None,
               blake3_reg_compress_Gps=None, ntt_ns_per_elem=None,
               ntt_batched_ns_per_elem=None, hbm_random_GBps=None,
               hbm_chase_ns=None, launch_us_sync=None,
               launch_us_stream=None)
    io = dict(disk_read_GBps=None, h2d_GBps=None, proof_dump_MBps=None,
              proof_dump_compact_MBps=None)

    nvcc = a.nvcc or shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    if not a.skip_cuda and arch and Path(nvcc).exists():
        build = raw_dir / "build"
        build.mkdir(exist_ok=True)
        print(f"[cuda benches] nvcc={nvcc}")
        specs = [   # each isolated: one failure skips one bench, not the rest
            ("bench_field_mul", BENCH_DIR, ["--runs", str(a.runs)]),
            ("bench_ntt", BENCH_DIR, []),
            ("bench_blake3_columns", BENCH_DIR, ["--runs", str(a.runs)]),
            ("bench_goldilocks_matmul", BENCH_DIR, []),
            ("bench_blake3_reg", PROF_BENCH_DIR, ["--runs", str(a.runs)]),
            ("bench_ntt_batched", PROF_BENCH_DIR, []),
            ("bench_hbm_random", PROF_BENCH_DIR, []),
            ("bench_launch_latency", PROF_BENCH_DIR, []),
        ]
        outs = {}
        for nm, src_dir, args_ in specs:
            try:
                exe = compile_bench(src_dir / f"{nm}.cu", build, arch, nvcc)
                outs[nm] = run_bench(exe, args_, raw_dir)
            except (RuntimeError, subprocess.TimeoutExpired) as e:
                print(f"  {nm} failed, continuing: {e}")
        if "bench_field_mul" in outs:
            gpu["field_mul_Gps"] = parse_field_mul(outs["bench_field_mul"])
            prov["field_mul_Gps"] = f"bench_field_mul best-of-{a.runs}, {today}"
        if "bench_ntt" in outs:
            gpu["ntt_ns_per_elem"] = parse_ntt(outs["bench_ntt"])
            prov["ntt_ns_per_elem"] = ("bench_ntt best variant at n=65536 "
                                       f"(us/NTT * 1000 / 65536), {today}")
        if "bench_blake3_columns" in outs:
            c, bulk = parse_blake3(outs["bench_blake3_columns"])
            gpu["blake3_compress_Gps"], gpu["blake3_bulk_GBps"] = c, bulk
            prov["blake3_compress_Gps"] = (
                f"bench_blake3_columns best over m sweep, {today} — "
                f"memory-bandwidth-limited at large m; NOT a basis for B")
            prov["blake3_bulk_GBps"] = \
                f"bench_blake3_columns best GB/s absorbed, {today}"
        if "bench_goldilocks_matmul" in outs:
            mm = parse_matmul(outs["bench_goldilocks_matmul"])
            if mm:
                prov["matmul_Gmul_s"] = (
                    f"bench_goldilocks_matmul best {mm:g} Gmul/s across the "
                    f"n sweep, {today} — informational (no profile field; "
                    f"its printed 'peak floor' line is GB10's 312, ignore)")
        if "bench_blake3_reg" in outs:
            gpu["blake3_reg_compress_Gps"] = \
                parse_blake3_reg(outs["bench_blake3_reg"])
            prov["blake3_reg_compress_Gps"] = (
                f"bench_blake3_reg register-resident chained compress, "
                f"{today} — the ALU-bound rate; B's proper scaling basis")
        if "bench_ntt_batched" in outs:
            gpu["ntt_batched_ns_per_elem"] = \
                parse_ntt_batched(outs["bench_ntt_batched"])
            prov["ntt_batched_ns_per_elem"] = (
                f"bench_ntt_batched best over batch sweep at n=65536, "
                f"{today} — the prover-path number; the single-transform "
                f"bench is launch-bound on large-L2 parts")
        if "bench_hbm_random" in outs:
            g, ch = parse_hbm_random(outs["bench_hbm_random"])
            gpu["hbm_random_GBps"], gpu["hbm_chase_ns"] = g, ch
            prov["hbm_random_GBps"] = (
                f"bench_hbm_random gather over an L2-exceeding buffer, "
                f"{today} — random 8B-read throughput")
            prov["hbm_chase_ns"] = (
                f"bench_hbm_random dependent pointer chase, {today} — "
                f"per-hop wall with parallel walkers")
        if "bench_launch_latency" in outs:
            ls, lt = parse_launch(outs["bench_launch_latency"])
            gpu["launch_us_sync"], gpu["launch_us_stream"] = ls, lt
            prov["launch_us_sync"] = (
                f"bench_launch_latency synced round trip, {today}")
            prov["launch_us_stream"] = (
                f"bench_launch_latency back-to-back enqueue, {today}")
    elif not a.skip_cuda:
        print(f"[cuda benches] skipped (nvcc at {nvcc}: "
              f"{Path(nvcc).exists()}, arch: {arch})")

    torch_ok = False
    if gpu_info:
        try:
            import torch
            torch_ok = torch.cuda.is_available()
            if not torch_ok:
                print("[torch benches] torch imports but CUDA is unavailable "
                      "(CPU-only build?) — skipped")
        except ImportError:
            print("[torch benches] torch not importable — skipped")
    if torch_ok:
        print("[torch benches]")
        for key, target, fn, note in [
                ("mem_bandwidth_GBps", gpu, bench_membw_GBps,
                 "torch d2d copy, read+write counted (2x bytes)"),
                ("h2d_GBps", io, bench_h2d_GBps,
                 "torch pinned-host->device copy")]:
            try:
                target[key] = round(fn(), 1)
                prov[key] = f"{note}, {today}"
                print(f"  {key} = {target[key]} GB/s")
            except Exception as e:            # isolate per measurement
                print(f"  {key} failed, continuing: {e}")

    tmp = Path(a.tmpdir)
    if not a.skip_disk:
        print(f"[disk] write + cache-evict + read over {a.disk_gb} GB")
        try:
            io["disk_read_GBps"] = round(
                bench_disk_read_GBps(tmp, a.disk_gb), 2)
            prov["disk_read_GBps"] = (
                f"python sequential read of a {a.disk_gb} GB probe after "
                f"fsync + posix_fadvise(DONTNEED) eviction, {today} — "
                f"device-served but not O_DIRECT-cold")
        except (RuntimeError, OSError) as e:
            print(f"  disk probe failed/skipped: {e}")
    print("[compact dump] u64le/base64 production transport")
    try:
        io["proof_dump_compact_MBps"] = round(
            bench_dump_compact_MBps(tmp, a.dump_compact_mb), 1)
        prov["proof_dump_compact_MBps"] = (
            f"bench_dump_compact_MBps: base64 of u64le chunks + fsync to "
            f"--tmpdir, {today} — the production transport; supersedes "
            f"the A100 reference in predict when present")
    except (RuntimeError, OSError) as e:
        print(f"  compact dump failed/skipped: {e}")
    print("[dump proxy]")
    try:
        io["proof_dump_MBps"] = round(
            bench_dump_proxy_MBps(tmp, a.dump_vals), 1)
        prov["proof_dump_MBps"] = ("PROXY: json string-build+write of u64 "
                                   f"arrays, {today} — replace with a real "
                                   "proof_dump measurement")
    except (RuntimeError, OSError) as e:
        print(f"  dump proxy failed/skipped: {e}")

    constants = dict(A_ns_per_slot=None, B_ns_per_cid=None,
                     C_ns_per_product=None, aggregate_ns_per_slot=None)
    if not a.no_derive:
        base = MachineProfile.load("gb10-spark")
        derived, dprov = derive_prove_constants(
            base, gpu["mem_bandwidth_GBps"], gpu["blake3_reg_compress_Gps"])
        constants.update(derived)
        prov.update(dprov)

    profile = {
        "name": a.name,
        "description": (f"{(gpu_info or {}).get('name', 'unknown GPU')} — "
                        f"calibrated by profiler/calibrate.py {today}"),
        "gpu": gpu,
        "prove_constants": constants,
        "verify": {"bytes_per_row": 700, "cores": os.cpu_count()},
        "io": io,
        "interconnect": None,
        "provenance": prov,
    }
    out_path.write_text(json.dumps(profile, indent=2) + "\n")
    print(f"\nwrote {out_path}")

    base = MachineProfile.load("gb10-spark")
    print(f"\nratios vs gb10-spark (the floor scales A/C by bandwidth, "
          f"B by compute):")
    for label, path in [("mem bandwidth", ("gpu", "mem_bandwidth_GBps")),
                        ("field mul", ("gpu", "field_mul_Gps")),
                        ("blake3 compress", ("gpu", "blake3_compress_Gps")),
                        ("blake3 reg", ("gpu", "blake3_reg_compress_Gps")),
                        ("blake3 bulk", ("gpu", "blake3_bulk_GBps"))]:
        new = profile["gpu"].get(path[1])
        old = base.get(*path)
        if new and old:
            print(f"  {label:16s} {new:>10.1f} / {old:<8g} = {new / old:5.2f}x")
    nt, ot = profile["gpu"].get("ntt_ns_per_elem"), \
        base.get("gpu", "ntt_ns_per_elem")
    if nt and ot:
        print(f"  {'ntt ns/elem':16s} {nt:>10.3f} / {ot:<8g} = "
              f"{ot / nt:5.2f}x faster")
    return 0


if __name__ == "__main__":
    sys.exit(main())
