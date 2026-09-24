import Formal.DTH.Rules

/-!
# The failure-fatal quotient (paper §2)

A profile is failure-fatal when its next failed check means death
(`¬ Survives s t`, equivalently `revival s t = 0`). The paper keeps the ST of
such a profile and replaces its TTD by a sentinel. This module proves the
paper's claim: two states whose Checker and Dropper profiles have the same
quotient class have the same value.

`src/dth_compact/main.py` implements the quotient in `profile`: a dead
profile maps to the sentinel id `N_ALIVE + s`, and every successor of a dead
profile is a dead sentinel (`build_table`).
-/

namespace Formal.DTH

/-- A quotient class: a revivable profile keeps `(s, t)`, a failure-fatal
profile keeps only `s`. -/
inductive QProfile
  | alive (s t : ℕ)
  | dead (s : ℕ)
  deriving DecidableEq

/-- The quotient map `φ` of paper §2. -/
def quot (p : Profile) : QProfile :=
  if Survives p.s p.t then .alive p.s p.t else .dead p.s

theorem quot_s {p p' : Profile} (h : quot p = quot p') : p.s = p'.s := by
  unfold quot at h
  split_ifs at h <;> simp_all

theorem survives_iff_of_quot {p p' : Profile} (h : quot p = quot p') :
    Survives p.s p.t ↔ Survives p'.s p'.t := by
  unfold quot at h
  split_ifs at h with h1 h2 h2 <;> simp_all

theorem t_eq_of_quot_of_survives {p p' : Profile} (h : quot p = quot p')
    (hs : Survives p.s p.t) : p.t = p'.t := by
  have hs' := (survives_iff_of_quot h).1 hs
  unfold quot at h
  simp only [hs, hs', ite_true, QProfile.alive.injEq] at h
  exact h.2

theorem rho_eq_of_quot {p p' : Profile} (h : quot p = quot p') : rho p = rho p' := by
  unfold rho
  by_cases hs : Survives p.s p.t
  · have hs' := (survives_iff_of_quot h).1 hs
    simp only [hs, hs', ite_true]
    exact t_eq_of_quot_of_survives h hs
  · have hs' : ¬ Survives p'.s p'.t := fun h' => hs ((survives_iff_of_quot h).2 h')
    simp [hs, hs']

/-- Quotient-equivalent states: both profiles lie in the same class. -/
def QEquiv (x y : State) : Prop := quot x.c = quot y.c ∧ quot x.d = quot y.d

theorem phi_eq_of_qequiv {x y : State} (h : QEquiv x y) : phi x = phi y := by
  unfold phi
  rw [quot_s h.1, quot_s h.2, rho_eq_of_quot h.1, rho_eq_of_quot h.2]

/-- A successful check maps equivalent states to equivalent states: a
failure-fatal Checker stays failure-fatal, and its TTD never matters. -/
theorem qequiv_successChild {x y : State} (h : QEquiv x y) (ℓ : ℕ) (hx : x.c.s + ℓ < 300)
    (hy : y.c.s + ℓ < 300) : QEquiv (successChild x ℓ hx) (successChild y ℓ hy) := by
  refine ⟨h.2, ?_⟩
  have hs := quot_s h.1
  show quot ⟨x.c.s + ℓ, x.c.t, hx⟩ = quot ⟨y.c.s + ℓ, y.c.t, hy⟩
  by_cases h1 : Survives x.c.s x.c.t
  · have ht := t_eq_of_quot_of_survives h.1 h1
    have heq : (⟨x.c.s + ℓ, x.c.t, hx⟩ : Profile) = ⟨y.c.s + ℓ, y.c.t, hy⟩ := by
      ext <;> simp [hs, ht]
    rw [heq]
  · have h1' : ¬ Survives y.c.s y.c.t := fun h' => h1 ((survives_iff_of_quot h.1).2 h')
    have h2 := not_survives_of_le (Nat.le_add_right x.c.s ℓ) h1
    have h2' := not_survives_of_le (Nat.le_add_right y.c.s ℓ) h1'
    unfold quot
    simp only [h2, ↓reduceIte]
    simp only [h2', ↓reduceIte, hs]

/-- A revived failed check maps equivalent states to equivalent states. -/
theorem qequiv_failChild {x y : State} (h : QEquiv x y) (hx : Survives x.c.s x.c.t) :
    QEquiv (failChild x) (failChild y) := by
  refine ⟨h.2, ?_⟩
  have ht := t_eq_of_quot_of_survives h.1 hx
  simp only [failChild, quot_s h.1, ht]

/-- **Quotient theorem** (paper §2): equivalent states have equal values. -/
theorem V_eq_of_qequiv (x y : State) (h : QEquiv x y) : V x = V y := by
  induction hm : 1201 - phi x using Nat.strong_induction_on generalizing x y with
  | _ k ih =>
  have hs := quot_s h.1
  have hsucc : ∀ ℓ, 1 ≤ ℓ → succPay V x ℓ = succPay V y ℓ := by
    intro ℓ hℓ
    unfold succPay
    by_cases h1 : x.c.s + ℓ < 300
    · have h2 : y.c.s + ℓ < 300 := by omega
      simp only [h1, h2, ↓reduceDIte]
      congr 1
      exact ih _ (by
          have := phi_lt_successChild x hℓ h1
          have := phi_le (successChild x ℓ h1)
          omega) _ _ (qequiv_successChild h ℓ h1 h2) rfl
    · have h2 : ¬ y.c.s + ℓ < 300 := by omega
      simp only [h1, h2, ↓reduceDIte]
  have hfail : failPay V x = failPay V y := by
    unfold failPay
    by_cases h1 : Survives x.c.s x.c.t
    · have h2 := (survives_iff_of_quot h.1).1 h1
      have ht := t_eq_of_quot_of_survives h.1 h1
      simp only [h1, h2, ↓reduceIte]
      rw [ih _ (by
          have := phi_lt_failChild x h1
          have := phi_le (failChild x)
          omega) _ _ (qequiv_failChild h h1) rfl, hs, ht]
    · have h2 : ¬ Survives y.c.s y.c.t := fun h' => h1 ((survives_iff_of_quot h.1).2 h')
      simp only [h1, h2, ↓reduceIte]
  rw [V_bellman x, V_bellman y]
  congr 1
  funext d c
  simp only [stage]
  split_ifs
  · exact hsucc _ (by omega)
  · exact hfail

end Formal.DTH
