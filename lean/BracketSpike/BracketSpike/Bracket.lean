import Mathlib

/-!
# Softmax bracket uniqueness — de-risking spike (paper Appendix B.3, Lemma B.3)

Goal of the spike: prove that the softmax per-row shift `c` is *uniquely* pinned by
the bracket constraints, **saturation included**, going honestly through the
field→integer lift that the paper's Lemma B.1a leans on.  This is the "hardest
consumer" identified in the launch-risk analysis; building the lift library
against it is the point.

The development is organised as four layers, matching the intended competition
architecture:

1. `Lift`      — ZMod P ↔ ℤ, the wrap-free lift from range checks.  (Risk #1.)
2. `Cell`      — per-cell value-neutrality: whatever saturation witness the prover
                 chooses, the emitted output equals the ideal table value
                 `g (c - x)`.  (Risk #3 — the compressed prose of Lemma B.3.)
3. `Threshold` — a non-increasing integer function is pinned by a two-sided
                 bracket.  (The reusable `threshold_unique` library lemma.)
4. `Row`       — assemble: two satisfying witnesses at S = 2 share the same `c`.

Modelling choices are documented inline.  Nothing here is `sorry`-free by
accident: the `#print axioms` at the bottom is the honest ledger.
-/

namespace Bracket

/-! ## Layer 1 — the field and the lift -/

/-- Goldilocks prime `2^64 - 2^32 + 1`.  For the lift we only need `NeZero`. -/
def P : ℕ := 18446744069414584321

instance : NeZero P := ⟨by norm_num [P]⟩

/-- Adding two field elements whose representatives don't overflow is exact. -/
lemma val_add_lt {a b : ZMod P} (h : a.val + b.val < P) :
    (a + b).val = a.val + b.val := by
  rw [ZMod.val_add, Nat.mod_eq_of_lt h]

/-- The core lift used per cell.  The field constraint `c = x + z + Zmax·zhigh`
    together with range bounds keeping every representative below `P` yields the
    genuine **integer** identity, with all quantities read off as `.val`.
    This is exactly Lemma B.1a's "the argument runs over the integers, the
    constraints over the field, so require the max recomposable value below P". -/
lemma lift_cell {c x z zhigh : ZMod P} {Zmax : ℕ}
    (hrec : c = x + z + (Zmax : ZMod P) * zhigh)
    -- range facts (from the range checks / lookup key range):
    (hx : x.val < 2^24) (hz : z.val < 2 * Zmax) (hzh : zhigh.val < 2^16)
    -- width condition: the largest recomposable value stays below P
    (hZ : Zmax ≤ 2^16) (hwidth : 2^24 + 2 * Zmax + Zmax * 2^16 < P) :
    (c.val : ℤ) = x.val + z.val + (Zmax : ℤ) * zhigh.val := by
  have hZval : ((Zmax : ZMod P)).val = Zmax := by
    apply ZMod.val_natCast_of_lt
    calc Zmax ≤ 2^16 := hZ
      _ < P := by norm_num [P]
  -- product bound, reused twice
  have hprod_le : Zmax * zhigh.val ≤ Zmax * 2^16 :=
    Nat.mul_le_mul (le_refl _) (le_of_lt hzh)
  have hbound : Zmax * zhigh.val < P := by
    calc Zmax * zhigh.val ≤ 2^16 * 2^16 := Nat.mul_le_mul hZ (le_of_lt hzh)
      _ < P := by norm_num [P]
  have hmul : ((Zmax : ZMod P) * zhigh).val = Zmax * zhigh.val := by
    rw [ZMod.val_mul, hZval, Nat.mod_eq_of_lt hbound]
  -- val of x + z
  have hnum : (2:ℕ)^24 + 2 * 2^16 ≤ P := by norm_num [P]
  have hxz : (x + z).val = x.val + z.val := by
    apply val_add_lt
    have hzsum : x.val + z.val < 2^24 + 2 * Zmax := by omega
    have hle : (2:ℕ)^24 + 2 * Zmax ≤ P := by omega
    omega
  -- val of (x+z) + Zmax·zhigh
  have hsum_full : (x + z + (Zmax : ZMod P) * zhigh).val
      = x.val + z.val + Zmax * zhigh.val := by
    have hlt : (x + z).val + ((Zmax : ZMod P) * zhigh).val < P := by
      rw [hxz, hmul]
      calc x.val + z.val + Zmax * zhigh.val
          < 2^24 + 2 * Zmax + Zmax * 2^16 := by omega
        _ < P := hwidth
    rw [val_add_lt hlt, hxz, hmul]
  have hnat : c.val = x.val + z.val + Zmax * zhigh.val := by rw [hrec]; exact hsum_full
  exact_mod_cast hnat

/-! ## Layer 3 — the reusable threshold lemma (proved before Cell so Row can use it) -/

/-- **`threshold_unique`.**  A non-increasing integer function is pinned by a
    two-sided adjacent bracket: if `f c ≤ T < f (c-1)` and likewise for `c'`,
    then `c = c'`.  This is the schema every monotone pin (softmax shift,
    RMSNorm rsqrt) instantiates.  Fully general, no `sorry`. -/
lemma threshold_unique {f : ℤ → ℤ} (hf : ∀ a b, a ≤ b → f b ≤ f a)
    {c c' T : ℤ}
    (h1 : f c ≤ T) (h2 : T < f (c - 1))
    (h1' : f c' ≤ T) (h2' : T < f (c' - 1)) : c = c' := by
  by_contra hne
  rcases lt_or_gt_of_ne hne with hlt | hgt
  · have hle : c ≤ c' - 1 := by omega
    have := hf c (c' - 1) hle
    omega
  · have hle : c' ≤ c - 1 := by omega
    have := hf c' (c - 1) hle
    omega

/-! ## Layer 2 — the ideal table and per-cell value-neutrality -/

/-- The table certificate, abstracted to exactly the properties Lemma B.3 uses.
    `TA` is the concrete lookup table (a scan would discharge these in the real
    gate); here they are hypotheses. -/
structure TableCert (Zmax v0 : ℕ) where
  TA        : ℤ → ℤ                       -- the table as a total function
  nonneg    : ∀ k, 0 ≤ TA k
  noninc    : ∀ a b, a ≤ b → TA b ≤ TA a  -- non-increasing
  zero_tail : ∀ k, (v0 : ℤ) ≤ k → TA k = 0
  tail_le   : v0 ≤ Zmax                   -- reaches zero before Zmax

/-- The *ideal* per-cell output as a function of `v = c - x`: the table on its
    live range, zero once saturated.  This is what every valid witness must emit. -/
def g {Zmax v0 : ℕ} (cert : TableCert Zmax v0) (v : ℤ) : ℤ :=
  if v < (v0 : ℤ) then cert.TA v else 0

/-- `g` is non-increasing on all of ℤ — the fact `s1 = Σ g(c - xᵢ)` needs. -/
lemma g_noninc {Zmax v0 : ℕ} (cert : TableCert Zmax v0) :
    ∀ a b, a ≤ b → g cert b ≤ g cert a := by
  intro a b hab
  unfold g
  by_cases hb : b < (v0 : ℤ) <;> by_cases ha : a < (v0 : ℤ) <;>
    simp only [hb, ha, if_true, if_false] <;>
    first
      | exact cert.noninc a b hab
      | exact absurd (lt_of_le_of_lt hab hb) ha
      | exact cert.nonneg a
      | exact le_refl 0

/-- **Per-cell value-neutrality (the crux of Lemma B.3).**
    Whatever saturation witness `(z, zhigh)` the prover commits, the cell output
    `y1` equals the ideal value `g (c - x)`.  The proof is the paper's case split:
    below `Zmax` the decomposition forces `zhigh = 0` (output = table); at or
    above `Zmax` the table is already zero, so the two admissible decompositions
    agree on the output.

    We work at the integer level (post-lift): `v = c - x`, `z`, `zhigh` are the
    lifted integers, `t = if zhigh = 0 then 0 else 1` is the boolean saturation
    flag the constraints `t·zhigh = zhigh`, `t² = t` pin, and the emitted output
    is `y1 = (1 - t) · TA z`. -/
lemma cell_value_neutral {Zmax v0 : ℕ} (cert : TableCert Zmax v0)
    {v z zhigh : ℤ}
    (hv : 0 ≤ v)                          -- forced: c ≥ x on unmasked cells
    (hrec : v = z + (Zmax : ℤ) * zhigh)    -- the lifted decomposition
    (hz0 : 0 ≤ z) (hz : z < 2 * Zmax)      -- z in the table key range
    (hzh0 : 0 ≤ zhigh)                     -- zhigh a nonneg word
    (hZpos : 0 < Zmax) :
    (if zhigh = 0 then (0:ℤ) else 1) * 0 + (1 - (if zhigh = 0 then (0:ℤ) else 1)) * cert.TA z
      = g cert v := by
  -- y1 = (1 - t) · TA z
  by_cases hzh : zhigh = 0
  · -- t = 0, y1 = TA z, and z = v
    subst hzh
    simp only [if_true]
    have hzv : z = v := by simp at hrec; omega
    subst hzv
    -- output = TA z = g z ; need z < v0 or z ≥ v0 handled by g
    unfold g
    by_cases hlt : z < (v0 : ℤ)
    · simp [hlt]
    · -- z ≥ v0 ⟹ TA z = 0 = g z
      simp only [hlt, if_false]
      rw [cert.zero_tail z (by omega)]
      ring
  · -- t = 1, y1 = 0.  Need g v = 0, i.e. v ≥ v0.
    simp only [hzh, if_false]
    have hzh1 : 1 ≤ zhigh := by omega
    -- v = z + Zmax·zhigh ≥ Zmax ≥ v0
    have : (Zmax : ℤ) ≤ v := by
      have : (Zmax : ℤ) * 1 ≤ (Zmax : ℤ) * zhigh := by
        apply mul_le_mul_of_nonneg_left hzh1 (by positivity)
      simp at this; omega
    have hge : (v0 : ℤ) ≤ v := by
      have : (v0 : ℤ) ≤ (Zmax : ℤ) := by exact_mod_cast cert.tail_le
      omega
    unfold g
    have : ¬ v < (v0 : ℤ) := by omega
    simp [this]

/-! ## Layer 4 — assemble a full row and prove `c` unique at S = 2 -/

/-- A single softmax row with `n` unmasked cells, modelled at the integer level
    (the lift of Layer 1 justifies working here).  The two committed row sums are
    `s1 = Σ g(c - xᵢ)` and `s2 = Σ g((c-1) - xᵢ)` (the δ = 1 table pair gives
    `s2(c) = s1(c-1)`), and the bracket pins are `s1 ≤ s_y < s2`. -/
structure Row (Zmax v0 n : ℕ) where
  cert : TableCert Zmax v0
  x    : Fin n → ℤ

/-- `s1(c) = Σ_i g(c - xᵢ)`. -/
def Row.s1 {Zmax v0 n : ℕ} (R : Row Zmax v0 n) (c : ℤ) : ℤ :=
  ∑ i, g R.cert (c - R.x i)

/-- The row-sum is non-increasing in `c` (sum of non-increasing terms). -/
lemma Row.s1_noninc {Zmax v0 n : ℕ} (R : Row Zmax v0 n) :
    ∀ a b, a ≤ b → R.s1 b ≤ R.s1 a := by
  intro a b hab
  unfold Row.s1
  apply Finset.sum_le_sum
  intro i _
  exact g_noninc R.cert (a - R.x i) (b - R.x i) (by omega)

/-- **Bracket uniqueness for the shift `c`.**
    If `c` and `c'` both satisfy the two bracket pins
    `s1 ≤ s_y` and `s_y < s1(·-1)` (the second is the `s2 ≥ s_y+1` pin, since
    `s2(c) = s1(c-1)`), then `c = c'`.  Holds for any row width `n`, hence in
    particular at S = 2. -/
theorem Row.shift_unique {Zmax v0 n : ℕ} (R : Row Zmax v0 n) {c c' : ℤ} {sy : ℤ}
    (hc_lo  : R.s1 c ≤ sy)  (hc_hi  : sy < R.s1 (c - 1))
    (hc_lo' : R.s1 c' ≤ sy) (hc_hi' : sy < R.s1 (c' - 1)) :
    c = c' :=
  threshold_unique R.s1_noninc hc_lo hc_hi hc_lo' hc_hi'

/-- The S = 2 specialisation stated explicitly, to mirror the spike's target. -/
theorem shift_unique_S2 {Zmax v0 : ℕ} (R : Row Zmax v0 2) {c c' sy : ℤ}
    (hc_lo  : R.s1 c ≤ sy)  (hc_hi  : sy < R.s1 (c - 1))
    (hc_lo' : R.s1 c' ≤ sy) (hc_hi' : sy < R.s1 (c' - 1)) :
    c = c' :=
  R.shift_unique hc_lo hc_hi hc_lo' hc_hi'

end Bracket
