import Formal.DTH.Rules

/-!
# Accumulated error of a certified sweep (paper §5)

The solver stores one number `V̂ x` per class. The number comes from a stage
matrix built from the stored child numbers `V̂`, not from the exact `V`.
Each stored number lies within `δ` of the value of its own stage matrix: a
certified saddle gap of at most `1e-6` with a stored midpoint gives
`δ = 5e-7` (`MatrixGame.abs_midpoint_sub_value_le`).

The paper claims `|V̂ - V| ≤ (1201 - Φ) · 5e-7`. This module proves it for any
`δ ≥ 0`: a child error `e` moves every stage payoff by at most `e`, because
`S = -V(child)` and `F = 1 - p - p V(child)` with `0 ≤ p ≤ 1`; the value is
1-Lipschitz in the entries (`MatrixGame.abs_value_sub_le`); and every child
lies on a strictly higher layer.
-/

open Finset Matrix

namespace Formal.DTH

/-- A child error of at most `e` moves every stage payoff by at most `e`. -/
theorem abs_stage_sub_le (W W' : State → ℝ) (x : State) {e : ℝ} (he : 0 ≤ e)
    (h : ∀ y, phi x < phi y → |W y - W' y| ≤ e) (d c : Fin 60) :
    |stage W x d c - stage W' x d c| ≤ e := by
  unfold stage succPay failPay
  split_ifs with h1 h2 h3
  · have := h _ (phi_lt_successChild x (by omega) h2)
    rw [show -W (successChild x _ h2) - -W' (successChild x _ h2) =
      -(W (successChild x _ h2) - W' (successChild x _ h2)) by ring, abs_neg]
    exact this
  · simpa using he
  · have hp0 := revival_nonneg x.c.s x.c.t
    have hp1 := revival_le_one x.c.s x.c.t
    have hc := h _ (phi_lt_failChild x h3)
    rw [show 1 - revival x.c.s x.c.t - revival x.c.s x.c.t * W (failChild x) -
        (1 - revival x.c.s x.c.t - revival x.c.s x.c.t * W' (failChild x)) =
        -(revival x.c.s x.c.t * (W (failChild x) - W' (failChild x))) by ring,
      abs_neg, abs_mul, abs_of_nonneg hp0]
    calc revival x.c.s x.c.t * |W (failChild x) - W' (failChild x)|
        ≤ 1 * e := mul_le_mul hp1 hc (abs_nonneg _) zero_le_one
      _ = e := one_mul e
  · simpa using he

/-- **Accumulated error bound** (paper §5): if every stored number lies within
`δ` of the value of the stage matrix built from the stored numbers, then
`|V̂ x - V x| ≤ (1201 - Φ(x)) δ`. -/
theorem abs_sub_V_le (Vhat : State → ℝ) {δ : ℝ} (hδ : 0 ≤ δ)
    (hstep : ∀ x, |Vhat x - MatrixGame.value (stage Vhat x)| ≤ δ) (x : State) :
    |Vhat x - V x| ≤ ((1201 - phi x : ℕ) : ℝ) * δ := by
  induction hm : 1201 - phi x using Nat.strong_induction_on generalizing x with
  | _ k ih =>
  have hphi := phi_le x
  -- every child sits on a higher layer, so its error is at most `(1200 - Φ(x)) δ`
  have hchild : ∀ y, phi x < phi y → |Vhat y - V y| ≤ ((1200 - phi x : ℕ) : ℝ) * δ := by
    intro y hy
    have hy' := phi_le y
    refine (ih (1201 - phi y) (by omega) y rfl).trans ?_
    have hle : 1201 - phi y ≤ 1200 - phi x := by omega
    exact mul_le_mul_of_nonneg_right (by exact_mod_cast hle) hδ
  have he : (0 : ℝ) ≤ ((1200 - phi x : ℕ) : ℝ) * δ := by positivity
  have hlip := MatrixGame.abs_value_sub_le (stage Vhat x) (stage V x)
    (abs_stage_sub_le Vhat V x he hchild)
  rw [← hm, V_bellman x]
  calc |Vhat x - MatrixGame.value (stage V x)|
      ≤ |Vhat x - MatrixGame.value (stage Vhat x)| +
          |MatrixGame.value (stage Vhat x) - MatrixGame.value (stage V x)| :=
        abs_sub_le _ _ _
    _ ≤ δ + ((1200 - phi x : ℕ) : ℝ) * δ := add_le_add (hstep x) hlip
    _ = ((1201 - phi x : ℕ) : ℝ) * δ := by
        rw [show 1201 - phi x = (1200 - phi x) + 1 by omega]
        push_cast; ring

/-- A certified midpoint lies within `5e-7` of its stage value. With that step
error the bound at the opening layer `Φ = 0` is `1201 · 5e-7 = 6.005e-4`, which
the paper reports as `V = 0.08985 ± 0.00061` and, for the win probability
`(1 + V) / 2`, `± 0.00031`. -/
theorem opening_error_bounds (Vhat : State → ℝ)
    (hstep : ∀ x, |Vhat x - MatrixGame.value (stage Vhat x)| ≤ 5e-7) (x : State)
    (hx : phi x = 0) :
    |Vhat x - V x| ≤ 0.00061 ∧ |(1 + Vhat x) / 2 - (1 + V x) / 2| ≤ 0.00031 := by
  have h := abs_sub_V_le Vhat (by norm_num) hstep x
  rw [hx] at h
  norm_num at h
  refine ⟨h.trans (by norm_num), ?_⟩
  rw [show (1 + Vhat x) / 2 - (1 + V x) / 2 = (Vhat x - V x) / 2 by ring, abs_div]
  norm_num
  linarith

end Formal.DTH
