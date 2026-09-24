import Formal.MatrixGame.Basic

/-!
# Pure Drop the Handkerchief: rules, potential, and backward induction

This module states the frozen pure-DTH rules of `AGENTS.md`,
`docs/REVIVAL_MODEL.md`, and `paper/dth_exact_solution.tex` §1–§4, and
mirrors `src/dth_compact/main.py` (`survives_injection`, `build_table`,
`class_values`, `full_matrix`).

* A profile is a player's squandered time `s ∈ 0..299` and total time dead
  `t ∈ ℕ`. ST at 300 or more is death, so a live profile carries `s < 300`.
* A state lists the Checker first: `x = (c, d)`.
* Actions are literal seconds `1..60`. We index them by `Fin 60`: index `i`
  is second `i + 1`. Row `d` is the Dropper's second, column `c` the
  Checker's. For `d ≤ c` the check succeeds with lag `c - d + 1`.
* `V x` is the current Dropper's win probability minus loss probability.
  The payoff matrix of a state reads `S ℓ = 1` when the Checker's ST reaches
  300, `S ℓ = -V(x_ℓ)` otherwise, and `F = 1` for a fatal failed check,
  `F = 1 - p - p V(x_f)` otherwise.

`V` is defined by well-founded recursion on the potential
`Φ = s_c + s_d + ρ_c + ρ_d`, which every live transition strictly increases
(`phi_lt_successChild`, `phi_lt_failChild`) and which never exceeds 1200
(`phi_le`).
-/

open Finset Matrix

namespace Formal.DTH

/-! ## Revival eligibility and probability -/

/-- A failed check with ST `s` and TTD `t` injects the dose `q = s + 60`.
The Checker can be revived exactly when `q < 300` and `t + q ≤ 300`
(`AGENTS.md`, frozen global rules; `survives_injection` in `main.py`). -/
def Survives (s t : ℕ) : Prop := s + 60 < 300 ∧ t + (s + 60) ≤ 300

instance (s t : ℕ) : Decidable (Survives s t) := inferInstanceAs (Decidable (_ ∧ _))

/-- The paper's form of the eligibility region: `s ≤ 239` and `s + t ≤ 240`. -/
theorem survives_iff (s t : ℕ) : Survives s t ↔ s ≤ 239 ∧ s + t ≤ 240 := by
  unfold Survives; omega

/-- Eligibility is lost when ST grows: a failure-fatal profile stays failure-fatal
after a successful check. -/
theorem not_survives_of_le {s s' t : ℕ} (hss' : s ≤ s') (h : ¬ Survives s t) :
    ¬ Survives s' t := by
  unfold Survives at *; omega

/-- The frozen revival surface of `docs/REVIVAL_MODEL.md`:
`P_rev(s, t) = 0.95 (1 - s/240) 0.75^(t/60)` on the eligible region, else `0`. -/
noncomputable def revival (s t : ℕ) : ℝ :=
  if Survives s t then 0.95 * (1 - (s : ℝ) / 240) * (0.75 : ℝ) ^ ((t : ℝ) / 60) else 0

theorem revival_pos_iff (s t : ℕ) : 0 < revival s t ↔ Survives s t := by
  unfold revival
  split_ifs with h
  · simp only [h, iff_true]
    have hs : (s : ℝ) ≤ 239 := by exact_mod_cast ((survives_iff s t).1 h).1
    have h1 : 0 < 1 - (s : ℝ) / 240 := by linarith
    have h2 : 0 < (0.75 : ℝ) ^ ((t : ℝ) / 60) := Real.rpow_pos_of_pos (by norm_num) _
    positivity
  · simp [h]

theorem revival_nonneg (s t : ℕ) : 0 ≤ revival s t := by
  unfold revival
  split_ifs with h
  · have hs : (s : ℝ) ≤ 239 := by exact_mod_cast ((survives_iff s t).1 h).1
    have h1 : 0 ≤ 1 - (s : ℝ) / 240 := by linarith
    have h2 : 0 ≤ (0.75 : ℝ) ^ ((t : ℝ) / 60) := Real.rpow_nonneg (by norm_num) _
    positivity
  · exact le_rfl

theorem revival_le (s t : ℕ) : revival s t ≤ 0.95 := by
  unfold revival
  split_ifs with h
  · have h1 : 1 - (s : ℝ) / 240 ≤ 1 := by
      have : (0 : ℝ) ≤ s / 240 := by positivity
      linarith
    have h1' : 0 ≤ 1 - (s : ℝ) / 240 := by
      have hs : (s : ℝ) ≤ 239 := by exact_mod_cast ((survives_iff s t).1 h).1
      linarith
    have h2 : (0.75 : ℝ) ^ ((t : ℝ) / 60) ≤ 1 :=
      Real.rpow_le_one (by norm_num) (by norm_num) (by positivity)
    have h2' : 0 ≤ (0.75 : ℝ) ^ ((t : ℝ) / 60) := Real.rpow_nonneg (by norm_num) _
    calc 0.95 * (1 - (s : ℝ) / 240) * (0.75 : ℝ) ^ ((t : ℝ) / 60)
        ≤ 0.95 * 1 * 1 := by gcongr
      _ = 0.95 := by norm_num
  · norm_num

theorem revival_le_one (s t : ℕ) : revival s t ≤ 1 := (revival_le s t).trans (by norm_num)

/-! ## Profiles, states, and transitions -/

/-- One player's profile: ST `s < 300` and TTD `t`. -/
@[ext]
structure Profile where
  s : ℕ
  t : ℕ
  hs : s < 300
deriving DecidableEq

/-- A state lists the Checker first. -/
@[ext]
structure State where
  c : Profile
  d : Profile
deriving DecidableEq

/-- A successful check with lag `ℓ` that leaves the Checker alive: the players
swap roles and the old Checker's ST grows by `ℓ`. -/
def successChild (x : State) (ℓ : ℕ) (h : x.c.s + ℓ < 300) : State :=
  ⟨x.d, ⟨x.c.s + ℓ, x.c.t, h⟩⟩

/-- A revived failed check: the players swap roles, the old Checker's ST resets
to `0`, and its TTD gains the dose `s + 60`. -/
def failChild (x : State) : State :=
  ⟨x.d, ⟨0, x.c.t + x.c.s + 60, by norm_num⟩⟩

/-! ## The potential -/

/-- `ρ(s, t) = t` while revival remains possible and `301` after. -/
def rho (p : Profile) : ℕ := if Survives p.s p.t then p.t else 301

/-- The progress score `Φ = s_c + s_d + ρ_c + ρ_d`. -/
def phi (x : State) : ℕ := x.c.s + x.d.s + rho x.c + rho x.d

theorem rho_le (p : Profile) : rho p ≤ 301 := by
  unfold rho; split_ifs with h
  · have := (survives_iff _ _).1 h; omega
  · exact le_rfl

theorem phi_le (x : State) : phi x ≤ 1200 := by
  have := rho_le x.c; have := rho_le x.d; have := x.c.hs; have := x.d.hs
  unfold phi; omega

/-- Role swaps preserve the potential. -/
theorem phi_swap (x : State) : phi ⟨x.d, x.c⟩ = phi x := by
  unfold phi; dsimp only; omega

/-- A successful check strictly increases `Φ` (paper §3). -/
theorem phi_lt_successChild (x : State) {ℓ : ℕ} (hℓ : 1 ≤ ℓ) (h : x.c.s + ℓ < 300) :
    phi x < phi (successChild x ℓ h) := by
  unfold phi successChild rho
  simp only
  by_cases h1 : Survives x.c.s x.c.t <;> by_cases h2 : Survives (x.c.s + ℓ) x.c.t <;>
    simp only [h1, h2, ite_true, ite_false]
  · omega
  · have := (survives_iff _ _).1 h1; omega
  · exact absurd h2 (not_survives_of_le (by omega) h1)
  · omega

/-- A revived failed check strictly increases `Φ`: by `60` while survival stays
possible, and by `301 - (s + t) ≥ 61` otherwise (paper §3). -/
theorem phi_lt_failChild (x : State) (hsurv : Survives x.c.s x.c.t) :
    phi x < phi (failChild x) := by
  have := (survives_iff _ _).1 hsurv
  unfold phi failChild rho
  simp only [hsurv, ite_true]
  split_ifs <;> omega

/-! ## The stage game -/

/-- The success payoff `S_ℓ` to the Dropper for lag `ℓ`, given child values `W`. -/
noncomputable def succPay (W : State → ℝ) (x : State) (ℓ : ℕ) : ℝ :=
  if h : x.c.s + ℓ < 300 then -W (successChild x ℓ h) else 1

/-- The failed-check payoff `F` to the Dropper, given child values `W`. -/
noncomputable def failPay (W : State → ℝ) (x : State) : ℝ :=
  if Survives x.c.s x.c.t then
    1 - revival x.c.s x.c.t - revival x.c.s x.c.t * W (failChild x)
  else 1

/-- The Toeplitz stage matrix `M[d, c] = S_{c-d+1}` for `c ≥ d` and `F` for
`c < d` (paper §4; `full_matrix` in `main.py`). Rows are the Dropper's
seconds and columns the Checker's, both indexed from `0`. -/
noncomputable def stage (W : State → ℝ) (x : State) : Matrix (Fin 60) (Fin 60) ℝ :=
  fun d c => if d ≤ c then succPay W x (c.val - d.val + 1) else failPay W x

/-! ## Backward induction -/

/-- The game value, by backward induction over the potential layers. The
recursive call is guarded by `phi x < phi y`; `V_bellman` removes the guard. -/
noncomputable def V (x : State) : ℝ :=
  MatrixGame.value (stage (fun y => if phi x < phi y then V y else 0) x)
termination_by 1201 - phi x
decreasing_by
  all_goals
    have := phi_le y
    omega

/-- Every child that the stage matrix reads has a larger potential. -/
theorem stage_congr {W W' : State → ℝ} (x : State)
    (h : ∀ y, phi x < phi y → W y = W' y) : stage W x = stage W' x := by
  funext d c
  unfold stage succPay failPay
  split_ifs with h1 h2 h3 <;> try rfl
  · rw [h _ (phi_lt_successChild x (by omega) h2)]
  · rw [h _ (phi_lt_failChild x h3)]

/-- The Bellman equation: `V x` is the value of the stage matrix built from `V`. -/
theorem V_bellman (x : State) : V x = MatrixGame.value (stage V x) := by
  rw [V]
  congr 1
  exact stage_congr x fun y hy => by simp [hy]

end Formal.DTH
