#!/usr/bin/env python3
"""Prover wall-clock vs context length: modeled floor, measured runs, cluster projection.

Rebuild of fig_runtime.png with the measured small-context points (S=10, S=100)
and the x-axis extended to 10^1 so the weights floor is visible.

Every curve constant is taken from paper.md:
- (W, L, Q)(S): the section 8 polynomials, verbatim.
- The wall-clock identity and its measured GB10 rates: Appendix A.5,
  T ~ 4 T_wit + (A_c + A_f + A_x) W + D W + E W + C Q + B L (+ T_aux).
- Cluster scaling: A.5, "A_c, A_f, A_x, C ride aggregate memory bandwidth and
  divide by a cluster's bandwidth ratio (about 2,580x for a 72-GPU NVL-class
  machine). D rides hash compute ... scales by roughly 170x."

Two terms the paper does not give in closed form, and how they are handled:
- E (constraint-coefficient work): stated as "4 to 5% of proving time" -- applied
  as a 4.5% share of the total rather than a per-slot rate.
- T_aux: unquantified in A.5; omitted.
- T_wit shape: "about an hour per pass at S=1000", and it "rides matmul compute
  and stays negligible at every scale" -- matmul FLOPs are linear in S at fixed
  parameters, so T_wit(S) = 3600 s x S/1000.

Checks against the paper's own statements (printed on each run): the GB10 curve
gives 8.2 h at S=1000 (section 8: "a floor of roughly 8 to 10 hours") and about
27 years at S=10^6 ("roughly 25 years"); the NVL72 curve gives about 5 days at
S=10^6, with the hash term 28% of it ("a quarter to a third of the cluster
floor").

Output: fig_runtime_measured.png (same directory).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Section 8 cost polynomials (Llama-4-Maverick) ---
def W(S): return 4.00e11 + 4.48e8 * S + 40320.0 * S**2   # witness slots
def L(S): return 1.19e8 + 1.50e8 * S + 12480.0 * S**2    # linear constraints
def Q(S): return 5.93e7 + 1.54e8 * S + 19200.0 * S**2    # quadratic products

# --- Appendix A.5 measured GB10 rates ---
A_C = 4.2e-9        # ns/slot, commit encode
A_F = 3.4e-9        # ns/slot, linear-fold transforms
A_X = 4.2e-9        # ns/slot, column-opening re-encode
D_HASH = 0.5e-9     # ns/slot, BLAKE3 column hashing ("about 0.5 ns per slot")
C_QUAD = 15e-9      # ns per quadratic product
B_LIN = 0.6e-9      # ns per linear constraint
E_SHARE = 0.045     # constraint-coefficient work, "4 to 5% of proving time"
T_WIT_1000 = 3600.0 # one witness pass at S=1000, "about an hour per pass"

def t_wit(S):
    return T_WIT_1000 * S / 1000.0

def gb10_floor(S):
    base = (4 * t_wit(S) + (A_C + A_F + A_X + D_HASH) * W(S)
            + C_QUAD * Q(S) + B_LIN * L(S))
    return base / (1 - E_SHARE)

# --- NVL72-class projection (A.5 scaling ratios) ---
BW_RATIO, HASH_RATIO = 2580.0, 170.0
def nvl72_floor(S):
    base = ((A_C + A_F + A_X) * W(S) + C_QUAD * Q(S) + B_LIN * L(S)
            + 4 * t_wit(S)) / BW_RATIO + D_HASH * W(S) / HASH_RATIO
    return base / (1 - E_SHARE)

# --- Measured runs (section 9) ---
HOUR = 3600.0
measured = [(10, 5.44 * HOUR), (100, 5.84 * HOUR), (1000, 14.3 * HOUR)]
pre_opt = (1093, 19.3 * HOUR)

# --- Figure ---
GB10, NVL = "#1a1a1a", "#b03a2e"
fig, ax = plt.subplots(figsize=(7.4, 5.2), dpi=160)

S = np.logspace(1, 6, 400)
ax.plot(S, gb10_floor(S), color=GB10, lw=2.2, label="DGX Spark (GB10), modeled", zorder=3)
ax.plot(S, nvl72_floor(S), color=NVL, lw=2.2, ls=(0, (6, 3)),
        label="NVL72-class cluster, projected", zorder=3)

ax.scatter([m[0] for m in measured], [m[1] for m in measured],
           s=62, color=GB10, edgecolor="white", linewidth=1.2, zorder=5,
           label="measured (Llama-4-Maverick)")
ax.scatter([pre_opt[0]], [pre_opt[1]], s=62, facecolor="white", edgecolor=GB10,
           linewidth=1.6, zorder=5, label="measured, pre-optimization")

# Direct labels, each anchored to its own point (14.3 h is the LOWER of the pair)
ax.annotate("5.4 h", measured[0], xytext=(0, 11), textcoords="offset points",
            ha="center", fontsize=9.5, color=GB10)
ax.annotate("5.8 h", measured[1], xytext=(0, 11), textcoords="offset points",
            ha="center", fontsize=9.5, color=GB10)
ax.annotate("14.3 h", measured[2], xytext=(10, -13), textcoords="offset points",
            ha="left", fontsize=9.5, color=GB10)
ax.annotate("19.3 h (pre-opt.)", pre_opt, xytext=(10, 7), textcoords="offset points",
            ha="left", fontsize=9.5, color=GB10)

ax.annotate("≈ 25 years, dense attention", (1e6, gb10_floor(1e6)),
            xytext=(-8, 10), textcoords="offset points", ha="right",
            fontsize=10, color=GB10)
ax.annotate("≈ 5 days", (1e6, nvl72_floor(1e6)), xytext=(-4, 12),
            textcoords="offset points", ha="right", fontsize=10, color=NVL)
ax.annotate("weights floor", (31.6, gb10_floor(31.6)), xytext=(0, -16),
            textcoords="offset points", ha="center", fontsize=9.5,
            color="#555555", style="italic")

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(6, 2.2e6)
ax.set_ylim(1.5e3, 2.5e9)
yticks = [HOUR, 24 * HOUR, 7 * 24 * HOUR, 30.44 * 24 * HOUR, 365.25 * 24 * HOUR,
          3652.5 * 24 * HOUR]
ax.set_yticks(yticks)
ax.set_yticklabels(["1 hour", "1 day", "1 week", "1 month", "1 year", "10 years"])
ax.set_xlabel("context length $S$ (tokens)", fontsize=12)
ax.set_ylabel("prover wall-clock (model floor)", fontsize=12)
ax.grid(True, which="major", color="#e3e3e3", lw=0.8, zorder=0)
ax.tick_params(labelsize=10.5)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.legend(loc="upper left", fontsize=10, frameon=False)

fig.tight_layout()
out = __file__.replace(".py", ".png")
fig.savefig(out, dpi=160, facecolor="white")

# Print the calibration checks
for s, label in [(1000, "S=1000"), (1e6, "S=1e6")]:
    print(f"{label}: GB10 {gb10_floor(s)/HOUR:.1f} h = {gb10_floor(s)/HOUR/24/365.25:.1f} y; "
          f"NVL72 {nvl72_floor(s)/HOUR/24:.1f} d")
print("hash share of NVL72 floor at 1e6:",
      round(D_HASH * W(1e6) / HASH_RATIO / (1 - E_SHARE) / nvl72_floor(1e6), 2))
print("wrote", out)
