import Formal.MatrixGame.Transform
import Formal.Toeplitz.Basic

/-!
# The public leap window (STL, rung L2)

In STL's leap window only Baku, as Dropper, may drop at second 61. The Checker
stays capped at 60, so the drop at 61 always defeats the check. The window
stage is the square `60 × 60` DTH stage plus one row of 60 copies of the
failure payoff `f` (`stage_matrix` in `src/stl/solver/leap_oracle.py`). This
file proves:

* a drop at 61 fails against every legal check (`leap_drop_fails`);
* the window value is `max (value M) f` (`value_window`), with the row-mix
  formula, optimal strategies, and window certificates;
* the DTH window stage has value `max(v60, F)` (`value_windowStage`);
* the enclosure lift through `max (·) f` and the error of the stored value
  `max v_sq f` (`stored_error`, `stored_midpoint_error`, `two_stored_diff`).

The row player (the Dropper) maximizes `p ⬝ᵥ (M *ᵥ q)`. Rows and columns are
indexed from `0`, so row `60` of `window M f` is action `61`.
-/

open Finset Matrix

set_option linter.unusedSectionVars false

namespace Formal.Leap

open Formal.MatrixGame

/-! ## Rules: the drop at 61 always fails -/

/-- A check succeeds exactly when `check ≥ drop` (`docs/ACTION_TIMING.md`). -/
abbrev checkSucceeds (drop check : ℕ) : Prop := drop ≤ check

/-- The Checker's legal seconds: `1..60` in every stage, including the window. -/
def checkerActions : Finset ℕ := Finset.Icc 1 60

/-- The Dropper's legal seconds: `1..61` for Baku as Dropper in the leap window,
and `1..60` otherwise (`AGENTS.md`, frozen global rules). -/
def dropperActions (inWindow bakuDrops : Bool) : Finset ℕ :=
  if inWindow && bakuDrops then Finset.Icc 1 61 else Finset.Icc 1 60

/-- Only Baku, as Dropper, inside the window, may choose `61`. -/
theorem sixtyOne_mem_dropperActions_iff (w b : Bool) :
    61 ∈ dropperActions w b ↔ (w && b) = true := by
  unfold dropperActions
  cases w <;> cases b <;> simp

/-- Claims CANONICAL-RULE-2 and STL-RULE-3 (`docs/ACTION_TIMING.md:41-47`,
`docs/FORMULATION_LADDER.md:83-87`, `src/stl/docs/GAME_AND_SOLVER.md:218-220`):
the pair `(drop = 61, check)` fails for every legal check. -/
theorem leap_drop_fails {c : ℕ} (hc : c ∈ checkerActions) : ¬ checkSucceeds 61 c := by
  simp only [checkerActions, Finset.mem_Icc] at hc
  unfold checkSucceeds; omega

/-- One stage cell in 1-based seconds: success with inclusive lag
`check - drop + 1` pays `S`, a failed check pays `F`. -/
noncomputable def cell (S : ℕ → ℝ) (F : ℝ) (drop check : ℕ) : ℝ :=
  if checkSucceeds drop check then S (check - drop + 1) else F

/-- Claim STL-RULE-3 (`src/stl/docs/GAME_AND_SOLVER.md:218-220`,
`paper/stl.tex:160-162`): every cell of the drop-61 row is the failure payoff. -/
theorem cell_leap_row (S : ℕ → ℝ) (F : ℝ) {c : ℕ} (hc : c ∈ checkerActions) :
    cell S F 61 c = F := by
  simp [cell, leap_drop_fails hc]

/-! ## The window matrix -/

section Window

variable {m : ℕ} {n : Type*} [Fintype n]

/-- The window stage: the square rows of `M`, then one row (index `m`, that is,
action `m + 1`) with payoff `f` in every column. -/
def window (M : Matrix (Fin m) n ℝ) (f : ℝ) : Matrix (Fin (m + 1)) n ℝ :=
  Fin.snoc (α := fun _ => n → ℝ) (fun i => M i) (fun _ => f)

@[simp] theorem window_castSucc (M : Matrix (Fin m) n ℝ) (f : ℝ) (i : Fin m) (c : n) :
    window M f i.castSucc c = M i c := by
  simp [window]

@[simp] theorem window_last (M : Matrix (Fin m) n ℝ) (f : ℝ) (c : n) :
    window M f (Fin.last m) c = f := by
  simp [window]

theorem vecMul_window (M : Matrix (Fin m) n ℝ) (f : ℝ) (p : Fin (m + 1) → ℝ) (c : n) :
    (p ᵥ* window M f) c = ((fun i => p i.castSucc) ᵥ* M) c + p (Fin.last m) * f := by
  simp only [vecMul, dotProduct, Fin.sum_univ_castSucc, window_castSucc, window_last]

theorem mulVec_window_castSucc (M : Matrix (Fin m) n ℝ) (f : ℝ) (q : n → ℝ) (i : Fin m) :
    (window M f *ᵥ q) i.castSucc = (M *ᵥ q) i := by
  simp only [mulVec, dotProduct, window_castSucc]

theorem mulVec_window_last (M : Matrix (Fin m) n ℝ) (f : ℝ) {q : n → ℝ}
    (hq : q ∈ simplex n) : (window M f *ᵥ q) (Fin.last m) = f := by
  simp only [mulVec, dotProduct, window_last, ← Finset.mul_sum, hq.2, mul_one]

/-- The row mix with mass `a` on the leap row and `(1 - a) p'` on the square rows. -/
def mix (a : ℝ) (p' : Fin m → ℝ) : Fin (m + 1) → ℝ :=
  Fin.snoc (α := fun _ => ℝ) (fun i => (1 - a) * p' i) a

@[simp] theorem mix_castSucc (a : ℝ) (p' : Fin m → ℝ) (i : Fin m) :
    mix a p' i.castSucc = (1 - a) * p' i := by
  simp [mix]

@[simp] theorem mix_last (a : ℝ) (p' : Fin m → ℝ) : mix a p' (Fin.last m) = a := by
  simp [mix]

theorem mix_mem_simplex {a : ℝ} (ha0 : 0 ≤ a) (ha1 : a ≤ 1) {p' : Fin m → ℝ}
    (hp : p' ∈ simplex (Fin m)) : mix a p' ∈ simplex (Fin (m + 1)) := by
  refine ⟨fun i => ?_, ?_⟩
  · induction i using Fin.lastCases with
    | last => simpa using ha0
    | cast i => simpa using mul_nonneg (by linarith) (hp.1 i)
  · rw [Fin.sum_univ_castSucc]
    simp only [mix_castSucc, mix_last, ← Finset.mul_sum, hp.2]
    ring

variable [NeZero m] [Nonempty n]

/-- The Dropper's guarantee in the window splits into the leap row and the
square rows. -/
theorem lowerBound_window (M : Matrix (Fin m) n ℝ) (f : ℝ) (p : Fin (m + 1) → ℝ) :
    lowerBound (window M f) p = lowerBound M (fun i => p i.castSucc) + p (Fin.last m) * f := by
  apply le_antisymm
  · obtain ⟨j, hj⟩ := exists_lowerBound_eq M (fun i => p i.castSucc)
    rw [hj, ← vecMul_window]; exact lowerBound_le_col _ _ _
  · rw [le_lowerBound_iff]; intro j
    rw [vecMul_window]; linarith [lowerBound_le_col M (fun i => p i.castSucc) j]

/-- The Checker's guarantee in the window: the leap row adds `f` to the rows
the Checker must hold down. -/
theorem upperBound_window (M : Matrix (Fin m) n ℝ) (f : ℝ) {q : n → ℝ}
    (hq : q ∈ simplex n) : upperBound (window M f) q = max (upperBound M q) f := by
  apply le_antisymm
  · rw [upperBound_le_iff]; intro i
    induction i using Fin.lastCases with
    | last => rw [mulVec_window_last M f hq]; exact le_max_right _ _
    | cast i =>
        rw [mulVec_window_castSucc]; exact (row_le_upperBound M q i).trans (le_max_left _ _)
  · apply max_le
    · obtain ⟨i, hi⟩ := exists_upperBound_eq M q
      rw [hi, ← mulVec_window_castSucc M f]; exact row_le_upperBound _ _ _
    · exact (mulVec_window_last M f hq).symm.le.trans
        (row_le_upperBound (window M f) q (Fin.last m))

theorem lowerBound_scale (M : Matrix (Fin m) n ℝ) (p : Fin m → ℝ) {k : ℝ} (hk : 0 ≤ k) :
    lowerBound M (fun i => k * p i) = k * lowerBound M p := by
  have hv : ∀ j, ((fun i => k * p i) ᵥ* M) j = k * (p ᵥ* M) j := by
    intro j; simp only [vecMul, dotProduct, Finset.mul_sum, mul_assoc]
  apply le_antisymm
  · obtain ⟨j, hj⟩ := exists_lowerBound_eq M p
    rw [hj, ← hv]; exact lowerBound_le_col _ _ _
  · rw [le_lowerBound_iff]; intro j
    rw [hv]; exact mul_le_mul_of_nonneg_left (lowerBound_le_col M p j) hk

/-- Claim CRATES-LEAP-1 (`src/crates/docs/LEAP_CERTIFICATE.md:124-129`) and the
proof of STL-MG-1 (`paper/stl.tex:168-171`): a row mix with mass `a` on the
leap row has minimum column payoff `a f + (1 - a) min_c (p'ᵀ M)_c`. -/
theorem lowerBound_mix (M : Matrix (Fin m) n ℝ) (f : ℝ) {a : ℝ} (ha1 : a ≤ 1)
    (p' : Fin m → ℝ) :
    lowerBound (window M f) (mix a p') = a * f + (1 - a) * lowerBound M p' := by
  rw [lowerBound_window]
  simp only [mix_castSucc, mix_last]
  rw [lowerBound_scale M p' (by linarith)]
  ring

/-- The mix payoff is linear in `a`, so it never exceeds the better endpoint:
`a f + (1 - a) L ≤ max L f` for `a ∈ [0, 1]`. -/
theorem mix_payoff_le_max {a L f : ℝ} (ha0 : 0 ≤ a) (ha1 : a ≤ 1) :
    a * f + (1 - a) * L ≤ max L f := by
  have h1 : a * f ≤ a * max L f := mul_le_mul_of_nonneg_left (le_max_right _ _) ha0
  have h2 : (1 - a) * L ≤ (1 - a) * max L f :=
    mul_le_mul_of_nonneg_left (le_max_left _ _) (by linarith)
  linarith

/-- Claims CRATES-LEAP-1 and STL-MG-1 (`src/crates/docs/LEAP_CERTIFICATE.md:120-138`,
`paper/stl.tex:163-172`): the window value is `max(v60, f)`. -/
theorem value_window (M : Matrix (Fin m) n ℝ) (f : ℝ) :
    value (window M f) = max (value M) f := by
  obtain ⟨p, hp, q, hq, h⟩ := exists_optimal M
  obtain ⟨e1, e2⟩ := certificate_encloses_value M hp hq
  have hv : value M = lowerBound M p := le_antisymm (e2.trans_eq h) e1
  apply le_antisymm
  · rw [value_eq_upperValue]
    refine (upperValue_le_upperBound _ hq).trans ?_
    rw [upperBound_window M f hq, h, ← hv]
  · apply max_le
    · have hm := mix_mem_simplex (a := 0) le_rfl zero_le_one hp
      have := (certificate_encloses_value (window M f) hm hq).1
      rw [lowerBound_mix M f zero_le_one] at this
      linarith
    · have hm := mix_mem_simplex (a := 1) zero_le_one le_rfl hp
      have := (certificate_encloses_value (window M f) hm hq).1
      rw [lowerBound_mix M f le_rfl] at this
      linarith

/-- Claim STL-MG-1, lower side without the minimax theorem's value name:
`lowerValue (window M f) = max (lowerValue M) f`. -/
theorem lowerValue_window (M : Matrix (Fin m) n ℝ) (f : ℝ) :
    lowerValue (window M f) = max (lowerValue M) f :=
  value_window M f

/-- Claim STL-MG-1, upper side: `upperValue (window M f) = max (upperValue M) f`. -/
theorem upperValue_window (M : Matrix (Fin m) n ℝ) (f : ℝ) :
    upperValue (window M f) = max (upperValue M) f := by
  rw [← value_eq_upperValue, ← value_eq_upperValue]; exact value_window M f

/-- Claim STL-MG-1 (`paper/stl.tex:171`): an optimal Checker strategy of the
square stage stays optimal in the window. -/
theorem checker_optimal_window (M : Matrix (Fin m) n ℝ) (f : ℝ) {q : n → ℝ}
    (hq : q ∈ simplex n) (hopt : upperBound M q = value M) :
    upperBound (window M f) q = value (window M f) := by
  rw [upperBound_window M f hq, hopt, value_window]

/-- Claim STL-MG-1: an optimal square Dropper strategy `p` with the better
endpoint `a ∈ {0, 1}` is optimal in the window. -/
theorem dropper_optimal_window (M : Matrix (Fin m) n ℝ) (f : ℝ) {p : Fin m → ℝ}
    (hopt : lowerBound M p = value M) :
    lowerBound (window M f) (mix (if f ≤ value M then 0 else 1) p) = value (window M f) := by
  rw [value_window]
  split_ifs with h
  · rw [lowerBound_mix M f zero_le_one, hopt, max_eq_left h]; ring
  · rw [lowerBound_mix M f le_rfl, max_eq_right (le_of_not_ge h)]; ring

/-- Adding the leap row weakly helps the row player. -/
theorem value_le_value_window (M : Matrix (Fin m) n ℝ) (f : ℝ) :
    value M ≤ value (window M f) := by
  rw [value_window]; exact le_max_left _ _

/-- Claims CRATES-LEAP-2 and STL-MG-2 (`src/crates/docs/LEAP_CERTIFICATE.md:135-138`,
`paper/stl.tex:205-207`): a square certificate `(p, q)` lifts to the window
enclosure `[max(L, f), max(U, f)]`. -/
theorem window_certificate (M : Matrix (Fin m) n ℝ) (f : ℝ) {p : Fin m → ℝ} {q : n → ℝ}
    (hp : p ∈ simplex (Fin m)) (hq : q ∈ simplex n) :
    max (lowerBound M p) f ≤ value (window M f) ∧
      value (window M f) ≤ max (upperBound M q) f := by
  obtain ⟨e1, e2⟩ := certificate_encloses_value M hp hq
  rw [value_window]
  exact ⟨max_le_max e1 le_rfl, max_le_max e2 le_rfl⟩

/-- The lifted bounds are themselves window certificate bounds: some window row
mix guarantees `max(L, f)` and `q` holds the window to `max(U, f)`. -/
theorem window_certificate_attained (M : Matrix (Fin m) n ℝ) (f : ℝ) {p : Fin m → ℝ}
    {q : n → ℝ} (hp : p ∈ simplex (Fin m)) (hq : q ∈ simplex n) :
    (∃ p₁ ∈ simplex (Fin (m + 1)), lowerBound (window M f) p₁ = max (lowerBound M p) f) ∧
      upperBound (window M f) q = max (upperBound M q) f := by
  refine ⟨?_, upperBound_window M f hq⟩
  rcases le_total f (lowerBound M p) with h | h
  · refine ⟨mix 0 p, mix_mem_simplex le_rfl zero_le_one hp, ?_⟩
    rw [lowerBound_mix M f zero_le_one, max_eq_left h]; ring
  · refine ⟨mix 1 p, mix_mem_simplex zero_le_one le_rfl hp, ?_⟩
    rw [lowerBound_mix M f le_rfl, max_eq_right h]; ring

end Window

/-! ## The DTH window stage -/

/-- The DTH window stage: `DTH.stage` plus the constant failure row. -/
noncomputable def windowStage (W : DTH.State → ℝ) (x : DTH.State) :
    Matrix (Fin 61) (Fin 60) ℝ :=
  window (DTH.stage W x) (DTH.failPay W x)

/-- The DTH window value is `max(v60, F)`. -/
theorem value_windowStage (W : DTH.State → ℝ) (x : DTH.State) :
    value (windowStage W x) = max (value (DTH.stage W x)) (DTH.failPay W x) :=
  value_window _ _

/-! ## Persymmetry and the one-product certificate -/


/-! ## The enclosure lift and the stored window value -/

section Lift

/-- Claims CRATES-LEAP-2 and STL-MG-2 (`src/crates/docs/LEAP_CERTIFICATE.md:135-137`):
`x ↦ max x f` is monotone, so it maps `[MN, MX]` into `[max(MN, f), max(MX, f)]`. -/
theorem lift_mem {MN MX v f : ℝ} (h1 : MN ≤ v) (h2 : v ≤ MX) :
    max MN f ≤ max v f ∧ max v f ≤ max MX f :=
  ⟨max_le_max h1 le_rfl, max_le_max h2 le_rfl⟩

/-- Claims CRATES-LEAP-2 and STL-MG-2 (`src/crates/docs/LEAP_CERTIFICATE.md:137-138`):
`max (·) f` is 1-Lipschitz. -/
theorem lift_lipschitz (x y f : ℝ) : |max x f - max y f| ≤ |x - y| :=
  abs_max_sub_max_le_abs x y f

/-- Claim CRATES-LEAP-2: the lift never widens the enclosure. -/
theorem lift_gap_le {MN MX : ℝ} (f : ℝ) (h : MN ≤ MX) :
    max MX f - max MN f ≤ MX - MN := by
  have := lift_lipschitz MX MN f
  have h0 : 0 ≤ MX - MN := by linarith
  rw [abs_of_nonneg h0] at this
  exact (le_abs_self _).trans this

/-- Claim CRATES-LEAP-2 (`src/crates/stl_solver/src/leap.rs:559-567`): the kernel
records kind 2 when `f ≥ v_sq`; the stored value `max v_sq f` is then `f`,
the leap row's payoff. -/
theorem stored_eq_f_of_kind2 {vsq f : ℝ} (h : vsq ≤ f) : max vsq f = f := max_eq_right h

variable {m : ℕ} [NeZero m] {n : Type*} [Fintype n] [Nonempty n]

/-- Claim CRATES-LEAP-2 and its doubt (`src/crates/stl_solver/src/leap.rs:505-509,559-567`):
if the square value and the square estimate `v_sq` both lie in `[MN, MX]` and
the lifted gap is at most `ε`, the stored value `max v_sq f` lies in the lifted
interval and is within `ε` (not `ε / 2`) of the window value. -/
theorem stored_error (M : Matrix (Fin m) n ℝ) {f MN MX vsq ε : ℝ}
    (hv1 : MN ≤ value M) (hv2 : value M ≤ MX) (hs1 : MN ≤ vsq) (hs2 : vsq ≤ MX)
    (hgap : max MX f - max MN f ≤ ε) :
    (max MN f ≤ max vsq f ∧ max vsq f ≤ max MX f) ∧
      |max vsq f - value (window M f)| ≤ ε := by
  refine ⟨lift_mem hs1 hs2, ?_⟩
  rw [value_window]
  obtain ⟨a1, a2⟩ := lift_mem (f := f) hs1 hs2
  obtain ⟨b1, b2⟩ := lift_mem (f := f) hv1 hv2
  rw [abs_le]; constructor <;> linarith

/-- Doubt of CRATES-LEAP-2 and STL-MG-2, sharpened for the clipped path, where
`v_sq` is the unlifted midpoint `(MN + MX) / 2`: the stored value sits in the
lower half of the lifted interval, so it is within `[V - ε, V + ε / 2]` of the
window value `V`. -/
theorem stored_midpoint_error (M : Matrix (Fin m) n ℝ) {f MN MX ε : ℝ}
    (hv1 : MN ≤ value M) (hv2 : value M ≤ MX)
    (hgap : max MX f - max MN f ≤ ε) :
    value (window M f) - ε ≤ max ((MN + MX) / 2) f ∧
      max ((MN + MX) / 2) f ≤ value (window M f) + ε / 2 := by
  rw [value_window]
  have hle : MN ≤ MX := hv1.trans hv2
  obtain ⟨b1, b2⟩ := lift_mem (f := f) hv1 hv2
  have hhalf : max ((MN + MX) / 2) f ≤ max MN f + (max MX f - max MN f) / 2 := by
    simp only [max_def]; split_ifs <;> linarith
  have hlow : max MN f ≤ max ((MN + MX) / 2) f := max_le_max (by linarith) le_rfl
  constructor <;> linarith

/-- Two stored values of one stage, each in `[V - ε, V + ε / 2]`, differ by at
most `3ε / 2`. -/
theorem two_stored_diff {V a b ε : ℝ} (ha1 : V - ε ≤ a) (ha2 : a ≤ V + ε / 2)
    (hb1 : V - ε ≤ b) (hb2 : b ≤ V + ε / 2) : |a - b| ≤ 3 * ε / 2 := by
  rw [abs_le]; constructor <;> linarith

end Lift

end Formal.Leap
