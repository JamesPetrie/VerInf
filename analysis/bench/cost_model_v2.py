"""Mechanistic prover-cost model (v2) — replaces the extrapolated m_total curve-fit.

WHY v2: the old witness term (cost_calculator.WITNESS_RECOMPUTE_US_PER_ROW) was ONE
empirical slope fit on 5 toy runs (m_total <= 2.75M) and extrapolated x42 to 400B
(m_total 114M). It also hid the card in a single constant. Building this exposed the
real physics:

  * The witness is SOFTMAX-BOUND, not matmul-bound. Each causal-softmax range-proof
    op (binary search over the value range) costs ~1e5x a single matmul FLOP, so at
    BOTH toy and 400B scale the softmax term (~ L * seq^2 * heads) dominates the
    witness. (My earlier "MoE-matmul-bound" diagnosis was wrong.)
  * Because toy is softmax-dominated, it calibrates the softmax + floor rates well
    but leaves the matmul rate essentially unidentifiable (matmul is ~0.01% of toy
    time). The MoE matmul (x128 experts) only becomes visible at 400B, so its rate
    is pinned by the single demo point -- explicitly, as a subdominant correction.

Model: prove_s = fixed + softmax_ops/sm_rate + cells/floor_rate + matmul_flops/mm_rate
  softmax_ops   = sum_layers  seq^2 * (d / d_head)          [range-proof, own kernel]
  cells         = ceil(committed_elems / ELL) * N_LIG       [NTT/hash bandwidth floor]
  matmul_flops  = sum_layers 4*seq*d^2 + 2*seq^2*d + 3*seq*d*d_ff * E   [E=experts]

Calibration (this file, run as __main__):
  sm_rate, floor_rate, fixed : 18 prove_sweep runs on V100 (2.8-250s, x89 range),
                               weighted LS, R^2=0.999, median |err| 3.4%.
  mm_rate                    : pinned so the model reproduces the 400B demo (8.04h,
                               S=847, d=5120, E=128, on GB10). Subdominant (~2.1h/8h).

Card transfer: rates are per-card. lpd.SCALE re-anchors V100->GB10 (~2.47x). For any
other card, MEASURE its transform rate once (coset_ntt microbench) and rescale -- the
op COUNTS are exact from geometry, so one hardware number transfers the whole model.
NOT an invented multiplier: it's the model's single hardware anchor, measured.
"""
import numpy as np, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # analysis/
import ligero_param_derivation as lpd

# --- calibrated constants (see __main__ for the fit that produced them) ---
FIXED_S      = 3.13           # per-process floor (CUDA ctx + tape setup), from small runs
SM_S_PER_OP  = 5.671e-06      # softmax range-proof, s/op   (V100)
FLOOR_S_CELL = 1.096e-07      # NTT/hash bandwidth, s/codeword-cell (V100)
MM_S_PER_FLOP= None           # matmul, s/FLOP -- pinned from the demo at import (below)
DEMO = dict(d=5120, d_ff=8192, seq=847, L=48, E=128, ELL=8192, N=65536,
            d_head=128, hours=8.04, card=900.0 / 273.0)  # 400B GB10 anchor; card = mem-BW slowdown vs V100 (datasheet 900/273)


def softmax_ops(d, seq, L, d_head): return L * seq * seq * (d // d_head)
def cells(d, d_ff, seq, L, ELL, N): return np.ceil(L * seq * (d + 2 * d_ff) / ELL) * N
def matmul_flops(d, d_ff, seq, L, E):
    return L * (4 * seq * d * d + 2 * seq * seq * d + 3 * seq * d * d_ff * E)


def _pin_matmul():
    """Solve mm_rate so the model hits the demo's measured 8.04h."""
    d, ff, seq, L, E = DEMO["d"], DEMO["d_ff"], DEMO["seq"], DEMO["L"], DEMO["E"]
    card = DEMO["card"]
    sm = softmax_ops(d, seq, L, DEMO["d_head"]) * SM_S_PER_OP * card
    fl = cells(d, ff, seq, L, DEMO["ELL"], DEMO["N"]) * FLOOR_S_CELL * card
    mm_time = DEMO["hours"] * 3600 - sm - fl - FIXED_S * card
    return matmul_flops(d, ff, seq, L, E) / (mm_time / card)      # FLOP/s (V100-equiv)


MM_S_PER_FLOP = 1.0 / _pin_matmul()


def prove_s(*, d, d_ff, seq, L, E, ELL, N, d_head=128, card=lpd.SCALE):
    """Full (no-spill) prove time in seconds for the given geometry on `card`
    (card = ns/element ratio vs V100; lpd.SCALE for GB10, 1.0 for V100)."""
    sm = softmax_ops(d, seq, L, d_head) * SM_S_PER_OP * card
    fl = cells(d, d_ff, seq, L, ELL, N) * FLOOR_S_CELL * card
    mm = matmul_flops(d, d_ff, seq, L, E) * MM_S_PER_FLOP * card
    return FIXED_S * card + sm + fl + mm, dict(softmax=sm, floor=fl, matmul=mm)


if __name__ == "__main__":
    import json
    rows = [json.loads(l) for l in open(Path(__file__).parent / "prove_runs.jsonl") if l.strip()]
    D = []
    for r in rows:
        if r.get("kind") != "prove_sweep": continue
        p, m = r.get("params", {}), r.get("measured", {})
        t = m.get("prove_s") or m.get("prove")
        if t and p.get("d") and p.get("N_LIG") and p.get("SEQ") and p.get("num_layers"):
            D.append((p, t))
    # refit sm/floor/fixed on toy (matmul negligible here), verify the baked constants
    X = np.array([[1.0, softmax_ops(p["d"], p["SEQ"], p["num_layers"], 64),
                   cells(p["d"], p["d_ff"], p["SEQ"], p["num_layers"], p["ELL"], p["N_LIG"])] for p, _ in D])
    t = np.array([tt for _, tt in D])
    c, *_ = np.linalg.lstsq(X / t[:, None], np.ones(len(D)), rcond=None)
    pred = X @ c; rel = np.abs(pred - t) / t
    ss = 1 - np.sum((t - pred) ** 2) / np.sum((t - t.mean()) ** 2)
    print(f"toy fit: fixed={c[0]:.2f} sm={c[1]:.3e} floor={c[2]:.3e}  R^2={ss:.4f} med|err|={np.median(rel)*100:.1f}%")
    tp, br = prove_s(d=5120, d_ff=8192, seq=847, L=48, E=128, ELL=8192, N=65536)
    print(f"demo predict: {tp/3600:.2f}h (sm {br['softmax']/3600:.1f} fl {br['floor']/3600:.1f} mm {br['matmul']/3600:.1f}) vs 8.04h measured")
    print(f"matmul rate (demo-pinned) = {1/MM_S_PER_FLOP:.2e} FLOP/s")
