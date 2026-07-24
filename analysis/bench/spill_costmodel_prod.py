"""PRODUCTION spill crossover (400B), correcting the toy-scale A/B. At 400B the
witness recompute is 57% of prove and its EFFECTIVE throughput is tiny (a full
forward pass producing 7.5TB), so re-reading from storage can beat recomputing —
the OPPOSITE of the toy regime where recompute is trivial and PCIe overhead wins.

Numbers from lpd (the demonstrated 400B run) + cost_calculator --S 1093:
  witness bytes W_B      = 7.50 TB
  one recompute pass     = T_WIT_S = 3888.8s (1.08h)
  4-pass witness term    = 16288s (56.8% of the 28668s prove)
  identity floor         = 12380s (encode+quad+lin)
Spill: compute once + write once, then re-read 3x instead of recompute 3x.
"""
import sys
from pathlib import Path
sys.path.insert(0, "analysis")
import ligero_param_derivation as lpd

S = 1093
W = lpd.W_of(S)
W_B = W * 8                      # witness bytes
T_pass = lpd.T_WIT_S            # one recompute pass (s)
E = lpd.E_FRACTION
floor_s = 12380.0              # from cost_calculator --S 1093
wit_term = 4 * T_pass / (1 - E)  # 4-pass witness term as costed
prove_s = floor_s + wit_term

rc_tput = W_B / T_pass / 1e9    # effective recompute throughput GB/s
print(f"witness = {W_B/1e12:.2f} TB   one recompute pass = {T_pass:.0f}s")
print(f"effective RECOMPUTE throughput = {rc_tput:.2f} GB/s  (a 400B forward pass)")
print(f"4-pass witness term = {wit_term:.0f}s ({100*wit_term/prove_s:.1f}% of {prove_s:.0f}s prove)\n")

print("spill = 1 compute + 1 write + 3 reads (vs 4 recomputes), by storage BW:")
print(f"{'store':>16} {'BW GB/s':>8} {'read/pass':>10} {'spill wit':>10} {'vs 4x rc':>9} {'prove':>8} {'faster':>8}")
for name, bw in [("NVMe single", 3.5), ("NVMe fast", 7.0), ("NVMe RAID", 20.0),
                 ("host PCIe(cap)", 11.0), ("HDD array", 1.5)]:
    t_io = W_B / (bw * 1e9)                       # one write or read
    spill_wit = T_pass + t_io + 3 * t_io          # compute once + write + 3 reads
    # effective: reads capped by min(storage, PCIe 11) since data still lands on GPU
    eff_bw = min(bw, 11.0)
    t_io_eff = W_B / (eff_bw * 1e9)
    spill_wit_eff = T_pass + t_io_eff + 3 * t_io_eff
    new_prove = floor_s + spill_wit_eff
    faster = 100 * (prove_s - new_prove) / prove_s
    win = "WIN" if new_prove < prove_s else "lose"
    print(f"{name:>16} {bw:8.1f} {t_io_eff:9.0f}s {spill_wit_eff:9.0f}s "
          f"{wit_term:8.0f}s {new_prove:7.0f}s {faster:+6.1f}% {win}")

print(f"\nKey: spill wins whenever effective read BW > recompute throughput "
      f"({rc_tput:.1f} GB/s).")
print("Caveats for the production version (NOT what the host prototype does):")
print("  - 7.5TB does NOT fit host RAM (84GB) -> must spill to DISK, not host.")
print("  - reads still land on GPU, so effective BW = min(disk, PCIe~11 GB/s).")
print("  - full 57% needs spilling the FULL witness; the softmax/silu-only")
print("    prototype captures just that fraction of the term.")
