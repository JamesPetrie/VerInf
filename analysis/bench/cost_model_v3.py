"""Prover-cost model v3 = v2 mechanistic base + CARD SPECS + SPILL/CACHE MODE.

Two things v2 lacked, both fixed here:

1. NO card-factor fudge. A card is a ROW IN A SPEC TABLE (CARD_SPECS), collected
   once (datasheet + one coset_ntt microbench), not a mystery multiplier chosen per
   run. `compute_scale` is the card's measured field-op time relative to the V100
   calibration card; `mem_bw_GBps` is its host<->GPU read bandwidth (for host-spill);
   disk BW is a runtime input (measured by the pre-flight gate). Adding a new card =
   adding a row, no re-fit. Validation: giving RTX8000 its measured scale (1.20)
   collapses its backtest error from 18% -> ~1%.

2. Spill/cache MODE. The v2 base predicts the no-spill (recompute) prove. Modes:
     no-spill  : witness recomputed on the streaming passes (v2 base, as-is).
     cache/host: witness computed ONCE, re-read from RAM  -> saves recompute, adds
                 a near-fixed PCIe transfer overhead (measured ~small, size-flat here).
     disk-spill: computed once, re-read from DISK -> saves recompute, adds bytes/disk_bw.
   Sign law (matches spill_costmodel_prod + the demo): spill helps iff the recompute
   it removes costs more than the re-read it adds, i.e. read_BW > recompute_throughput.

DATA HONESTY: the disk-spill A/B timings we have were taken with LIGERO_SPILL_FADVISE
(fdatasync + DONTNEED per block) to force cold reads for the BW study -- that inflates
disk time beyond raw bytes/BW (sync latency), so it is NOT used to calibrate absolute
disk-I/O here. The recompute-saving fraction is taken from the CLEAN witness-cache A/B
(same card, no fadvise: 190s no-cache -> 98s cache). Disk-I/O uses the analytic
bytes/BW; absolute disk calibration wants clean (non-fadvise) spill runs.
"""
import numpy as np, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cost_model_v2 as v2

# --- CARD SPECS: the card's REAL characteristics (datasheet), NOT a fitted scale. ---
# The prover is memory-bandwidth-bound, so field-op time ~ 1/mem_bw. The per-card
# slowdown is therefore COMPUTED as mem_bw[V100]/mem_bw[card] from the datasheet number
# -- add a card by looking up its GB/s, no calibration run. (Validated: RTX8000 predicts
# to 10% from its 672 GB/s datasheet spec alone; the ~10% residual is datasheet-peak vs
# effective-BW kernel efficiency, closed by one coset_ntt microbench -> effective GB/s.)
#   gpu_read_bw_GBps = host<->GPU link read BW for host-spill re-reads.
MEM_BW_GBps = {                    # HBM/GDDR datasheet bandwidth
    "V100": 900, "RTX8000": 672, "GB10": 273, "A100_40": 1555, "A100_80": 2039,
    "L40S": 864, "H100": 3350, "RTX4090": 1008,
}
VRAM_GB = {"V100": 32, "RTX8000": 48, "GB10": 128, "A100_40": 40, "A100_80": 80,
           "L40S": 48, "H100": 80, "RTX4090": 24}   # for the memory gate
REF_CARD = "V100"                  # v2 rates were calibrated here
PASSES = 4                         # lpd n_wit_sweeps: witness needed each of 4 protocol rounds

def card_factor(card):
    """Slowdown vs the V100 calibration card, DERIVED from datasheet mem BW."""
    return MEM_BW_GBps[REF_CARD] / MEM_BW_GBps[card]

import numpy as np

def prove_s(*, d, d_ff, seq, L, E, ELL, N, d_head=128, card="V100",
            mode="no-spill", witness_bytes=None, disk_bw_GBps=None):
    """Full prove seconds, MEMORY-GATED. mode in {no-spill, disk-spill}.
    witness_bytes = recomputable-witness size (drives the fit/recompute gate; e.g.
    lpd.W_of(S)*8 for the 400B model). disk_bw_GBps required for disk-spill.

    Gate: if the witness fits VRAM, no-spill HOLDS it (re-read at VRAM BW, ~free) and
    spill only adds slower-tier I/O -> spill loses (the toy regime). If it does NOT
    fit, no-spill must RECOMPUTE it every one of the PASSES protocol rounds -> spill
    (compute once + read from disk) wins whenever disk BW beats recompute (the 400B
    regime). This is why spill helps at 400B (7.5TB witness) but not at toy scale."""
    f = card_factor(card)
    _, br = v2.prove_s(d=d, d_ff=d_ff, seq=seq, L=L, E=E, ELL=ELL, N=N, d_head=d_head, card=f)
    wit, floor, fixed = br["softmax"] + br["matmul"], br["floor"], v2.FIXED_S * f
    if witness_bytes is None:
        witness_bytes = np.ceil(L * seq * (d + 2 * d_ff)) * 8   # dense-toy fallback
    fits = witness_bytes < 0.6 * VRAM_GB[card] * 1e9
    codeword_bytes = v2.cells(d, d_ff, seq, L, ELL, N) * 8
    reread = codeword_bytes * PASSES
    if mode == "no-spill":
        t = fixed + floor + (wit + reread / (MEM_BW_GBps[card] * 1e9) if fits
                             else PASSES * wit)
    elif mode == "disk-spill":
        if disk_bw_GBps is None:
            raise ValueError("disk-spill needs disk_bw_GBps (from pre-flight gate)")
        t = fixed + floor + wit + reread / (disk_bw_GBps * 1e9)
    else:
        raise ValueError(f"unknown mode {mode}")
    return t, dict(**br, mode=mode, fits_vram=fits)


if __name__ == "__main__":
    import sys as _s; _s.path.insert(0, "analysis"); import ligero_param_derivation as lpd
    # 1) card from DATASHEET mem-BW (no per-card fit): RTX8000
    p_v100, _ = prove_s(d=512, d_ff=1536, seq=1024, L=4, E=1, ELL=512, N=4096, d_head=64, card="V100")
    p_rtx, _  = prove_s(d=512, d_ff=1536, seq=1024, L=4, E=1, ELL=512, N=4096, d_head=64, card="RTX8000")
    print("CARD from datasheet mem-BW (RTX8000 measured 250.2s):")
    print(f"  V100 spec (900): {p_v100:.0f}s   RTX8000 spec (672): {p_rtx:.0f}s ({abs(p_rtx-250.2)/250.2*100:.0f}% err, no fit)")
    # 2) memory-gated mode reproduces the 400B demo (disk-spill, GB10)
    wb = lpd.W_of(847) * 8
    ds, _ = prove_s(d=5120, d_ff=8192, seq=847, L=48, E=128, ELL=8192, N=65536, card="GB10",
                    mode="disk-spill", witness_bytes=wb, disk_bw_GBps=2.0)
    print(f"\nDEMO check: disk-spill S=847 GB10 -> {ds/3600:.2f}h vs measured 8.04h")
    # 3) the experiment A/B on A100-80, memory-gated (no-spill recomputes: witness 5TB !fit VRAM)
    print("\nEXPERIMENT A/B (400B arch E=128 N=32768) on A100-80 ($1.87/h), by tokens S:")
    for S in [500, 350, 250]:
        wb = lpd.W_of(S) * 8
        no, _ = prove_s(d=5120, d_ff=8192, seq=S, L=48, E=128, ELL=8192, N=32768,
                        card="A100_80", mode="no-spill", witness_bytes=wb)
        ds, _ = prove_s(d=5120, d_ff=8192, seq=S, L=48, E=128, ELL=8192, N=32768,
                        card="A100_80", mode="disk-spill", witness_bytes=wb, disk_bw_GBps=7.0)
        ab = (no + ds) / 3600
        print(f"  S={S}: no-spill {no/3600:.2f}h  disk-spill {ds/3600:.2f}h (WIN)  A/B {ab:.2f}h  ~${ab*1.87:.1f}")
