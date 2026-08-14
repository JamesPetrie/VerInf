"""Instrumented A/B: does witness SPILL actually beat recompute when disk BW is good?

MEASURES (not guesses): the spill dir's real disk write/read BW, then proves the SAME
model 3 ways -- no-spill (recompute) / host-spill (pinned RAM) / disk-spill (file) --
and reports which is fastest + whether disk-spill still Rust-ACCEPTs. Softmax-heavy
shape (large seq) makes recompute expensive per spilled byte, the regime where spill
can win. Incremental JSON save after every step => survives drops/kills.

Env: AB_D/AB_SEQ/AB_NL/AB_DFF (shape), LIGERO_WITNESS_SPILL_DIR (where disk-spill goes,
point at a fast NVMe), AB_RESULT (result json path).
"""
import sys, os, time, gc, json
from pathlib import Path
R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R / "prover")); sys.path.insert(0, str(R / "prover/tests"))
sys.path.insert(0, str(R / "demo")); sys.path.insert(0, str(R / "analysis/bench"))
os.environ["LIGERO_PHASE_TIMING"] = "1"
import _uint64_compat  # noqa
import torch
import demo_toy_transformer as dt
import core as C
from core import LigeroConfig
from tape import Tape
from _rust_verify import rust_verify_tape

D   = int(os.environ.get("AB_D", "1024"))
SEQ = int(os.environ.get("AB_SEQ", "2048"))
NL  = int(os.environ.get("AB_NL", "4"))
DFF = int(os.environ.get("AB_DFF", str(3 * D)))
SPILLDIR = os.environ.get("LIGERO_WITNESS_SPILL_DIR", str(R / "analysis/bench/_spill_tmp"))
RESULT   = os.environ.get("AB_RESULT", str(R / "analysis/bench/spill_ab_result.json"))
_ELL = 1 << (D - 1).bit_length()          # ELL >= d (embedding_lookup needs d | ELL)
CFG  = LigeroConfig(ELL=_ELL, K_DEG=2 * _ELL, N_LIG=8 * _ELL, T_QUERIES=16)
SEED = b"spill-ab"
C._WITNESS_CACHE_ON = True

_t0 = time.time()
res = {"D": D, "SEQ": SEQ, "NL": NL, "DFF": DFF, "spilldir": SPILLDIR}
def lg(m): print(f"[spill-ab +{time.time()-_t0:7.1f}s] {m}", flush=True)
def save():
    try: Path(RESULT).write_text(json.dumps(res, indent=2))
    except Exception as e: lg(f"save err {e}")


def measure_disk_bw(path, gb=3.0):
    """Write then read a test file via raw fds; return (write_GBps, read_GBps).
    Read after fsync; O_DIRECT unavailable portably so caches may help reads a bit,
    but the WRITE number is honest and the ratio is what matters for the verdict."""
    os.makedirs(path, exist_ok=True)
    f = os.path.join(path, "bwtest.bin"); n = int(gb * 1e9); chunk = os.urandom(1 << 20)
    t = time.time(); fd = os.open(f, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644); w = 0
    while w < n:
        w += os.write(fd, chunk)
    os.fsync(fd); os.close(fd); wt = time.time() - t
    # evict from page cache with posix_fadvise (NO root needed) so read hits DISK
    fd = os.open(f, os.O_RDONLY); dropped = False
    try:
        os.posix_fadvise(fd, 0, n, os.POSIX_FADV_DONTNEED); dropped = True
    except (OSError, AttributeError):
        pass
    res["bw_cache_dropped"] = dropped
    t = time.time(); rd = 0
    while True:
        b = os.read(fd, 1 << 20)
        if not b: break
        rd += len(b)
    os.close(fd); rt = time.time() - t
    try: os.remove(f)
    except Exception: pass
    return n / wt / 1e9, n / rt / 1e9


def build():
    dt.d, dt.d_ff, dt.SEQ = D, DFF, SEQ
    tape = Tape(CFG, silu_config=dt.SILU_CFG, lazy=True)
    x = dt._rand_signed(SEQ * D, half=dt.HALF_X)
    resid = tape.commit("x_input", x, (SEQ, D))
    for L in range(NL):
        w = dt._commit_weights_random(tape, layer_idx=L)
        resid = dt._run_block(tape, resid, w, H=D // dt.d_h)
    vocab = 64
    fn = torch.full((D,), dt.S, dtype=torch.uint64, device="cuda")
    lm = dt._rand_signed(D * vocab, half=dt.HALF)
    fnw = tape.commit("final_norm_w", fn, (D,)); lmw = tape.commit("W_lm_head", lm, (D, vocab))
    dt._run_tail(tape, resid, fnw, lmw, vocab_size=vocab)
    return tape


def prove_timed(label, spill_on, disk_on):
    C._WITNESS_SPILL_ON = spill_on; C._WITNESS_SPILL_DISK = disk_on
    C._SPILL_FADVISE = disk_on          # disk mode: evict cache -> re-reads hit the real disk
    torch.manual_seed(1234); gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    tape = build()
    t = time.time(); proof = tape.prove(seed=SEED); pt = time.time() - t
    peak = torch.cuda.max_memory_allocated() / 2**30
    C._WITNESS_SPILL_ON = False; C._WITNESS_SPILL_DISK = False; C._SPILL_FADVISE = False
    lg(f"{label}: prove {pt:.1f}s  peak {peak:.2f} GB")
    return round(pt, 2), round(peak, 3), tape, proof


res["gpu"] = torch.cuda.get_device_name(0)
lg(f"GPU {res['gpu']} | d={D} seq={SEQ} L={NL} dff={DFF} | spilldir={SPILLDIR}"); save()

lg("1/4 measuring disk BW of the spill dir (write+read 3GB) ...")
# what filesystem is the spill dir REALLY on? overlay/tmpfs => not the NVMe we paid for
try:
    import subprocess
    fstype = subprocess.run(["stat", "-f", "-c", "%T", SPILLDIR], capture_output=True, text=True).stdout.strip()
    res["spill_fstype"] = fstype; lg(f"spill dir fstype: {fstype}")
except Exception as e:
    res["spill_fstype"] = f"?({e})"
wbw, rbw = measure_disk_bw(SPILLDIR)
res["disk_write_GBps"] = round(wbw, 2); res["disk_read_GBps"] = round(rbw, 2); save()
lg(f"disk BW: write {wbw:.2f} GB/s, read {rbw:.2f} GB/s")

# PRE-FLIGHT GATE: never pay for the full prove sweep if the disk is too slow to
# possibly show spill's regime. Abort+exit BEFORE proving so the runner destroys the
# box immediately. Threshold defaults to 2.5 GB/s (> 400B recompute tput 1.93 + margin).
MIN_READ = float(os.environ.get("AB_MIN_READ_GBPS", "2.5"))
BAD_FS = {"overlayfs", "tmpfs"}
res["min_read_required"] = MIN_READ
if rbw < MIN_READ or res.get("spill_fstype") in BAD_FS:
    reason = (f"read BW {rbw:.2f} < required {MIN_READ} GB/s" if rbw < MIN_READ
              else f"spill dir on {res.get('spill_fstype')} (not real disk)")
    res["aborted_low_bw"] = reason; save()
    lg(f"ABORT (no prove sweep, no wasted $$): {reason}. "
       f"Offer BW is overstated / spill dir is not on fast storage. Pick another box.")
    print(f"\n=== SPILL A/B ABORTED (pre-flight gate) ===\n  {reason}", flush=True)
    sys.exit(3)

lg("2/4 prove NO-SPILL (recompute baseline) ...")
res["noSpill_s"], res["noSpill_peak"], _, _ = prove_timed("no-spill", False, False); save()
lg("3/4 prove HOST-SPILL (pinned RAM re-read) ...")
res["hostSpill_s"], res["hostSpill_peak"], _, _ = prove_timed("host-spill", True, False); save()
lg("4/4 prove DISK-SPILL (file re-read) ...")
res["diskSpill_s"], res["diskSpill_peak"], tapeD, proofD = prove_timed("disk-spill", True, True); save()

# verdict
res["actual_fastest"] = min([("no-spill", res["noSpill_s"]), ("host-spill", res["hostSpill_s"]),
                             ("disk-spill", res["diskSpill_s"])], key=lambda x: x[1])[0]
res["disk_spill_vs_recompute_%"] = round(100 * (res["diskSpill_s"] - res["noSpill_s"]) / res["noSpill_s"], 1)
res["host_spill_vs_recompute_%"] = round(100 * (res["hostSpill_s"] - res["noSpill_s"]) / res["noSpill_s"], 1)
save()

lg("verifying disk-spill proof (Rust ACCEPT) ...")
acc, _ = rust_verify_tape(tapeD, proofD, seed=SEED); res["disk_spill_accept"] = bool(acc); save()
lg(f"disk-spill Rust verify: {'ACCEPT' if acc else 'REJECT'}")

print("\n=== SPILL A/B RESULT ===", flush=True)
print(f"  GPU {res['gpu']} | d={D} seq={SEQ} L={NL}")
print(f"  disk BW (spill dir): write {res['disk_write_GBps']} GB/s · read {res['disk_read_GBps']} GB/s")
print(f"  prove:  no-spill {res['noSpill_s']}s · host-spill {res['hostSpill_s']}s · disk-spill {res['diskSpill_s']}s")
print(f"  disk-spill vs recompute: {res['disk_spill_vs_recompute_%']:+}%   host-spill: {res['host_spill_vs_recompute_%']:+}%")
print(f"  FASTEST: {res['actual_fastest']}   disk-spill ACCEPT: {res['disk_spill_accept']}")
print(f"  (spill 'wins' iff faster than no-spill; needs disk read BW > witness recompute throughput)")
save()
sys.exit(0 if res.get("disk_spill_accept") else 1)
