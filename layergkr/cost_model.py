"""Layer-GKR-LF cost model — INDEPENDENT recomputation, not a copy of the theorem doc.

Source of the scheme: analysis/VerInf_LayerGKR_4h_theorem_ru.md (2026-08-04).

This module does three separate things, deliberately kept apart so a reader can
see which numbers are derived and which are assumed:

  1. `reference_model()` — the CURRENT flat-Ligero cost identity (paper Appendix
     A.5) at a given S. Used only as the calibration anchor. Self-checks against
     the doc's "known subtotal 7.784 h" at S=1000.

  2. `weight_geometry()` — P and N_pad recomputed FROM MAVERICK TENSOR SHAPES,
     via the doc's own row-capacity rule
         N_pad = ELL * sum_t count_t * n_out_t * ceil(n_in_t / ELL)
     This is the check that matters: the doc's §9.2 table is the largest single
     block of its budget (75% of the new proof-compute), so it is recomputed here
     from first principles rather than trusted.

  3. `layergkr_model()` — the new scheme's budget and the kappa-parameterised
     bound  T = T_forward + kappa * T_comp + T_rtt + T_tail.

KAPPA IS THE WHOLE BALL GAME. The doc adopts kappa <= 1.5 and derives 3.95 h,
with break-even at 1.530 — a 2% margin. But kappa is "how far an implementation
lands above its own cost identity", and the ONLY measured instance of that
quantity in this project is the current prover: 51,334.6 s measured against a
28,059 s identity at S=1000 => kappa_observed = 1.83. At that value the same
formula gives 4.58 h, not 3.95 h. Both are reported; neither is hidden.

Nothing here is a measurement of the new scheme. It is a model of a protocol
that does not exist yet, calibrated on one that does.
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple

# ── primitive costs (paper Appendix A.5, ns per unit) ─────────────────────────
A_C = 4.2      # commit encode        (per witness slot)
A_F = 3.4      # proof fold           (per witness slot)
A_X = 4.2      # opening re-encode    (per witness slot)
D_COEF = 0.5   # coefficient/aux      (per witness slot)
C_QUAD = 15.0  # per quadratic product
B_LIN = 0.6    # per linear-fold cell
E_STREAM = 3.0 # streamed factorized coefficient generation (per weight/padding slot)

T_FORWARD_S = 3609.0   # one exact semantic forward, MEASURED (hidden-run archive)

ELL = 8192             # RS message slots per row (the row-capacity quantum)


# ── 1. reference (current flat-Ligero) identity ───────────────────────────────
def _W(S: int) -> float:
    return 4.00e11 + 4.48e8 * S + 40320 * S * S

def _L(S: int) -> float:
    return 1.19e8 + 1.50e8 * S + 12480 * S * S

def _Q(S: int) -> float:
    return 5.93e7 + 1.54e8 * S + 19200 * S * S


def reference_model(S: int = 1000) -> Dict[str, float]:
    """The current scheme's cost identity. Terms in seconds."""
    W, L, Q = _W(S), _L(S), _Q(S)
    t = {
        "witness_4_sweeps": 4 * T_FORWARD_S,
        "encode_fold_open": (A_C + A_F + A_X) * W * 1e-9,
        "coefficients": D_COEF * W * 1e-9,
        "quadratic": C_QUAD * Q * 1e-9,
        "linear": B_LIN * L * 1e-9,
    }
    t["subtotal"] = sum(t.values())
    t["_W"], t["_L"], t["_Q"] = W, L, Q
    return t


# ── 2. Maverick weight geometry, recomputed from shapes ───────────────────────
@dataclass(frozen=True)
class Family:
    name: str
    count: int          # number of tensors in the family
    n_in: int           # contracted (input) dimension
    n_out: int          # output dimension = number of RS rows per tensor

    @property
    def P(self) -> int:
        """Projected-message fields: one P[j] per output coordinate, per tensor."""
        return self.count * self.n_out * self.n_in

    def N_pad(self, ell: int = ELL) -> int:
        """Row CAPACITY, not model entries. Each output coordinate occupies a whole
        number of ELL-wide RS rows; the slack is committed and opened like any
        other slot, so it is paid for. This is the doc's rule, recomputed."""
        rows_per_out = -(-self.n_in // ell)        # ceil
        return ell * self.count * self.n_out * rows_per_out


# Llama-4-Maverick: 48 blocks, 24 of them MoE with 128 experts (top-1), d=5120.
# Dense FFN d_ff=16384; expert/shared-expert d_ff=8192; vocab 202048.
# Tied embedding/LM-head is enrolled as TWO logical output-major views (§3.1), so
# both orientations are counted — a single layout cannot serve both projections.
MAVERICK: List[Family] = [
    Family("Attention QKVO",     192,   5120,   5120),   # 48 blocks x {q,k,v,o}
    Family("Dense gate/up",       48,   5120,  16384),   # 24 dense blocks x {gate,up}
    Family("Dense down",          24,  16384,   5120),
    Family("MoE expert gate/up", 6144,  5120,   8192),   # 24 x 128 x {gate,up}
    Family("MoE expert down",    3072,  8192,   5120),   # 24 x 128
    Family("MoE shared gate/up",  48,   5120,   8192),
    Family("MoE shared down",     24,   8192,   5120),
    Family("Router",              24,   5120,    128),
    Family("Embedding view",       1, 202048,   5120),
    Family("LM-head view",         1,   5120, 202048),
    Family("Gains",               97,   5120,      1),
]


def weight_geometry(families: List[Family] = None, ell: int = ELL
                    ) -> Tuple[int, int, List[Tuple[str, int, int]]]:
    """Return (P_total, N_pad_total, per-family rows)."""
    families = families or MAVERICK
    rows = [(f.name, f.P, f.N_pad(ell)) for f in families]
    return sum(r[1] for r in rows), sum(r[2] for r in rows), rows


# ── 3. the new scheme's budget ────────────────────────────────────────────────
# Every line the doc counts in §9.2 EXCEPT the two weight-geometry lines, which
# are recomputed above instead of copied. Values in seconds, pre-kappa.
OTHER_COMPUTE_S: Dict[str, float] = {
    "selected local GKR (C*Q_sel)":        632.6,
    "lookup boundary + multiplicity RS":   570.2,
    "stable-sort RS + segmented/perm":      20.5,
    "output/emitted roots":                  7.3,
    "embedding/token/UI edge superlayers":  50.0,
    "projected-root encode/IRS/LF reserve": 12.0,
    "other structured GKR":                105.0,
    "explicit fold reserve":               100.0,
    "general orchestration reserve":       100.0,
    "linear/mask products":                 20.0,
    "radix-sort traffic reserve":           29.0,
    "amortized weight refresh/link":        43.0,
}

T_RTT_S = 76.0     # 3500 GKR round trips + 2 extra LogUp challenge epochs per layer
T_TAIL_S = 10.0    # metadata tail after the streamed opening


def layergkr_model(kappa: float = 1.5, ell: int = ELL,
                   families: List[Family] = None) -> Dict[str, object]:
    """Online-prover bound for an ALREADY ENROLLED model (cold enrollment excluded,
    per §10). Returns the budget breakdown and the kappa-scaled total."""
    P, N_pad, rows = weight_geometry(families, ell)
    proj_codewords = E_STREAM * (P + N_pad) * 1e-9      # commit projected codewords
    open_encode = A_X * N_pad * 1e-9                    # persistent-weight opening
    counted = dict(OTHER_COMPUTE_S)
    counted["projected weight codewords E*(P+N_pad)"] = proj_codewords
    counted["persistent-weight opening encode A_x*N_pad"] = open_encode
    t_comp = sum(counted.values())
    total = T_FORWARD_S + kappa * t_comp + T_RTT_S + T_TAIL_S
    return {
        "P": P, "N_pad": N_pad, "rows": rows,
        "budget_s": counted, "T_comp_s": t_comp,
        "kappa": kappa, "total_s": total, "total_h": total / 3600.0,
        "kappa_breakeven_4h": (14400.0 - T_FORWARD_S - T_RTT_S - T_TAIL_S) / t_comp,
    }


# ── calibration: what kappa does the EXISTING prover actually exhibit? ────────
MEASURED_PROVE_S = 51334.6      # hidden-run archive, S=1000 (see caveat in README)


def observed_kappa(S: int = 1000) -> float:
    """The one empirical instance of 'implementation vs its own identity' we have."""
    return MEASURED_PROVE_S / reference_model(S)["subtotal"]


def _self_check() -> None:
    """Guard rails. These assert the model reproduces the numbers it claims to."""
    ref = reference_model(1000)
    assert abs(ref["subtotal"] / 3600 - 7.784) < 0.02, ref["subtotal"] / 3600
    P, N_pad, rows = weight_geometry()
    # The doc's §9.2 totals, recomputed here from shapes alone.
    assert P == 402_725_114_880, P
    assert N_pad == 564_632_231_936, N_pad
    per = dict((n, (p, q)) for n, p, q in rows)
    assert per["MoE expert gate/up"] == (257_698_037_760, 412_316_860_416)
    assert per["Embedding view"][1] == 1_048_576_000
    assert per["LM-head view"][1] == 1_655_177_216
    m = layergkr_model(kappa=1.5)
    # The doc states T_comp <= 6995 s and uses that rounded figure in its final
    # formula (-> 14187.5 s). Recomputing its own table line by line gives ~6963 s,
    # i.e. the doc carries a ~32 s pad. Check BOTH: that our sum sits just under
    # the stated cap, and that feeding the doc's rounded cap reproduces its number.
    assert 6900.0 < m["T_comp_s"] <= 6995.0, m["T_comp_s"]
    doc_total = T_FORWARD_S + 1.5 * 6995.0 + T_RTT_S + T_TAIL_S
    assert abs(doc_total - 14187.5) < 0.6, doc_total
    assert abs(m["kappa_breakeven_4h"] - 1.530) < 0.01, m["kappa_breakeven_4h"]


if __name__ == "__main__":
    _self_check()
    print("=== self-check PASSED: doc's reference model and §9.2 geometry reproduced ===\n")

    ref = reference_model(1000)
    print(f"CURRENT scheme, cost identity at S=1000")
    for k in ("witness_4_sweeps", "encode_fold_open", "coefficients", "quadratic", "linear"):
        print(f"  {k:<22} {ref[k]:9.0f} s  ({ref[k]/3600:5.3f} h)")
    print(f"  {'SUBTOTAL':<22} {ref['subtotal']:9.0f} s  ({ref['subtotal']/3600:5.3f} h)")
    print(f"  measured prove         {MEASURED_PROVE_S:9.0f} s  ({MEASURED_PROVE_S/3600:5.2f} h)")
    print(f"  -> kappa_observed      {observed_kappa():9.3f}\n")

    m = layergkr_model(kappa=1.5)
    print("NEW scheme (Layer-GKR-LF), counted proof compute, pre-kappa")
    for k, v in sorted(m["budget_s"].items(), key=lambda kv: -kv[1]):
        print(f"  {k:<45} {v:8.1f} s  ({100*v/m['T_comp_s']:4.1f}%)")
    print(f"  {'T_comp':<45} {m['T_comp_s']:8.1f} s")
    print(f"  one semantic forward (measured, not scaled)   {T_FORWARD_S:8.1f} s\n")

    print(f"{'kappa':>8} {'T_prove':>10} {'hours':>7}   note")
    for k, note in ((1.285, "upper edge of the paper's own 8-10h model range"),
                    (1.500, "adopted by the theorem -> its 3.95 h headline"),
                    (m["kappa_breakeven_4h"], "break-even for the 4-hour claim"),
                    (observed_kappa(), "OBSERVED for the existing prover")):
        t = T_FORWARD_S + k * m["T_comp_s"] + T_RTT_S + T_TAIL_S
        print(f"{k:8.3f} {t:9.0f}s {t/3600:6.2f} h   {note}")

    print("\n--- ELL sensitivity: the padding lever the doc leaves on the table ---")
    base = layergkr_model(kappa=1.5)
    for ell in (8192, 5120, 2560, 1024):
        mm = layergkr_model(kappa=1.5, ell=ell)
        print(f"  ELL={ell:5d}  N_pad={mm['N_pad']/1e9:7.1f}B  "
              f"T_prove={mm['total_h']:5.2f} h  "
              f"(delta {mm['total_s']-base['total_s']:+7.0f} s)")
    print("  NOTE: ELL is a live RS geometry parameter, not a free knob — changing it")
    print("  changes the code and every row-layout assumption. This table sizes the")
    print("  PRIZE, it does not claim the change is sound or free.")
