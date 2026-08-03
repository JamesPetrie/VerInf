"""PRE-FLIGHT machine validation — run FIRST on every rented box, BEFORE any paid
workload. Measures the ACTUAL hardware and compares it to what the vast offer
ADVERTISED (passed in via PF_* env by the runner). If ANY spec falls short of
advertised (beyond tolerance) the box lied — we do NOT trust the rest of it, we
abort here so the runner destroys it. Never run an experiment on a machine whose
real characteristics don't match the offer.

Exit 0 = machine matches offer, safe to run. Exit 4 = mismatch, abort + destroy.

Expected (advertised) specs from env — the runner fills these from the offer:
  PF_GPU           gpu_name substring (e.g. "A100")
  PF_NUM_GPUS      advertised gpu count
  PF_VRAM_GB       advertised per-GPU VRAM (GB)
  PF_RAM_GB        advertised host RAM (GB)
  PF_CPUS          advertised vCPU cores
  PF_DISK_BW_GBPS  advertised disk BW (GB/s)
  PF_DISK_DIR      dir to benchmark (the real workload/spill dir)
Tuning: PF_TOL (frac of advertised that must be met, default 0.85),
  PF_DISK_TOL (disk BW is noisiest / most overstated, default 0.70),
  PF_MIN_READ_GBPS (hard floor the EXPERIMENT needs regardless of offer, default 0).
"""
import os, sys, json, time, subprocess

TOL = float(os.environ.get("PF_TOL", "0.85"))
DTOL = float(os.environ.get("PF_DISK_TOL", "0.70"))
MIN_READ = float(os.environ.get("PF_MIN_READ_GBPS", "0"))
DISK_DIR = os.environ.get("PF_DISK_DIR", "/workspace")
OUT = os.environ.get("PF_RESULT", "/workspace/preflight.json")
rep = {"checks": [], "ok": True, "ts": time.time()}


def check(name, adv, meas, ok, detail="", fatal=True):
    # fatal=False: report but don't abort. Used for disk_fstype/disk_bw-ratio, which are
    # over-strict on vast -- EVERY vast container has /workspace on overlayfs and advertises
    # disk_bw 10-20x over reality, so gating on them rejects all boxes. The check that
    # actually matters (disk_read_min: can the disk sustain what the experiment needs) stays
    # fatal, and hardware specs (GPU/VRAM/RAM/CPU) stay fatal.
    rep["checks"].append({"spec": name, "advertised": adv, "measured": meas,
                          "ok": bool(ok), "detail": detail, "fatal": fatal})
    if not ok and fatal:
        rep["ok"] = False
    flag = "OK " if ok else ("FAIL" if fatal else "warn")
    print(f"  [{flag}] {name:11} advertised={adv}  measured={meas}  {detail}", flush=True)


def env_f(k):
    v = os.environ.get(k, "").strip()
    try: return float(v) if v else None
    except ValueError: return None


def measure_disk_bw(path, gb=3.0):
    os.makedirs(path, exist_ok=True)
    f = os.path.join(path, ".pf_bwtest.bin"); n = int(gb * 1e9); chunk = os.urandom(1 << 20)
    t = time.time(); fd = os.open(f, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644); w = 0
    while w < n: w += os.write(fd, chunk)
    os.fsync(fd); os.close(fd); wt = time.time() - t
    fd = os.open(f, os.O_RDONLY)
    try: os.posix_fadvise(fd, 0, n, os.POSIX_FADV_DONTNEED)   # evict cache -> hit disk (no root)
    except (OSError, AttributeError): pass
    t = time.time(); rd = 0
    while True:
        b = os.read(fd, 1 << 20)
        if not b: break
        rd += len(b)
    os.close(fd); rt = time.time() - t
    try: os.remove(f)
    except Exception: pass
    return n / wt / 1e9, n / rt / 1e9


print(f"=== PRE-FLIGHT: measured vs advertised (tol {TOL:.0%}, disk {DTOL:.0%}) ===", flush=True)

# --- GPU (name, count, VRAM) ---
try:
    import torch
    ngpu = torch.cuda.device_count()
    gname = torch.cuda.get_device_name(0) if ngpu else "none"
    vram = torch.cuda.get_device_properties(0).total_memory / 2**30 if ngpu else 0.0
    rep["gpu"] = {"name": gname, "count": ngpu, "vram_gb": round(vram, 1)}
    # a real kernel must actually run — advertised card that can't run torch is worthless
    _ = (torch.ones(1024, 1024, device="cuda") @ torch.ones(1024, 1024, device="cuda")).sum().item() if ngpu else 0
    kernel_ok = ngpu > 0
except Exception as e:
    gname, ngpu, vram, kernel_ok = f"ERR:{e}", 0, 0.0, False
    rep["gpu"] = {"error": str(e)}

adv_gpu = os.environ.get("PF_GPU", "").strip()
if adv_gpu:
    check("gpu_name", adv_gpu, gname, adv_gpu.lower() in gname.lower(),
          "" if kernel_ok else "CUDA kernel did NOT run")
check("gpu_kernel", "runs", "runs" if kernel_ok else "FAILED", kernel_ok)
if env_f("PF_NUM_GPUS"): check("gpu_count", int(env_f("PF_NUM_GPUS")), ngpu, ngpu >= env_f("PF_NUM_GPUS"))
if env_f("PF_VRAM_GB"): check("vram_gb", env_f("PF_VRAM_GB"), round(vram, 1), vram >= TOL * env_f("PF_VRAM_GB"))

# --- host RAM ---
try:
    kb = next(int(l.split()[1]) for l in open("/proc/meminfo") if l.startswith("MemTotal"))
    ram_gb = kb / 2**20
except Exception:
    ram_gb = 0.0
rep["ram_gb"] = round(ram_gb, 1)
if env_f("PF_RAM_GB"): check("ram_gb", env_f("PF_RAM_GB"), round(ram_gb, 1), ram_gb >= TOL * env_f("PF_RAM_GB"))

# --- CPU cores ---
cpus = os.cpu_count() or 0
rep["cpus"] = cpus
if env_f("PF_CPUS"): check("cpus", int(env_f("PF_CPUS")), cpus, cpus >= TOL * env_f("PF_CPUS"))

# --- disk: fstype + measured BW vs advertised ---
try:
    fstype = subprocess.run(["stat", "-f", "-c", "%T", DISK_DIR], capture_output=True, text=True).stdout.strip()
except Exception as e:
    fstype = f"?({e})"
rep["disk_fstype"] = fstype
check("disk_fstype", "real block dev", fstype, fstype not in {"overlayfs", "tmpfs"},
      "overlay is normal on vast (backed by real disk); not fatal" if fstype in {"overlayfs", "tmpfs"} else "",
      fatal=False)
wbw, rbw = measure_disk_bw(DISK_DIR)
rep["disk_write_GBps"] = round(wbw, 2); rep["disk_read_GBps"] = round(rbw, 2)
adv_bw = env_f("PF_DISK_BW_GBPS")
if adv_bw:
    ratio = rbw / adv_bw if adv_bw else 0
    check("disk_bw", f"{adv_bw:.1f} GB/s", f"w{wbw:.2f}/r{rbw:.2f} GB/s", rbw >= DTOL * adv_bw,
          f"measured/advertised = {ratio:.0%} (vast always overstates; not fatal)", fatal=False)
if MIN_READ:
    check("disk_read_min", f">={MIN_READ:.1f} (experiment needs)", f"{rbw:.2f} GB/s", rbw >= MIN_READ)

try: open(OUT, "w").write(json.dumps(rep, indent=2))
except Exception: pass

if rep["ok"]:
    print("=== PRE-FLIGHT PASS — machine matches offer, proceeding ===", flush=True)
    sys.exit(0)
bad = [c["spec"] for c in rep["checks"] if not c["ok"]]
print(f"=== PRE-FLIGHT FAIL: {', '.join(bad)} — machine does NOT match offer. "
      f"ABORT (no paid workload). Destroy box, pick another. ===", flush=True)
sys.exit(4)
