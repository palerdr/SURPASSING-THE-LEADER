import Formal.DTH.Rules
import Formal.MatrixGame.Transform
import Formal.Toeplitz.Equalizer

/-!
# The DTH stage matrix from its 61 transition classes

This module proves the structural facts about one DTH stage game that every
implementation relies on, and it connects each implementation's form of the
equalizer recurrence to `Formal.Toeplitz.weights`.

Conventions (shared with `Formal.DTH.Rules` and `Formal.Toeplitz.Basic`):

* Rows are the Dropper's seconds and columns the Checker's, indexed from `0`
  (`Fin 60`, index `i` is second `i + 1`). The row player (Dropper) maximizes
  `p ⬝ᵥ (M *ᵥ q)`.
* `S k` (`k = 0..59`) is the success payoff for physical lag `k + 1` and `F`
  is the failed-check payoff. The matrix is `M[d, c] = S (c - d)` for `d ≤ c`
  and `F` for `c < d` (`matrix_cell` in `src/dth_cpp/matrix_game.cpp`,
  `full_matrix` in `src/dth_compact/main.py`,
  `reconstruct_transition_class_matrix` in `src/dth/solver.py`).
* Values are stated from the current Dropper's seat. A live child has
  swapped roles, so a parent reads `-V(child)`; a Dropper win pays `+1`.
* `δ = S 0 - F` and `r = Toeplitz.weights s f` (the recurrence of
  `try_rung2`). The code forms of the recurrence (`b`-form in C, C++, Rust and
  `fast_kernel.py`; the 1-based doc form; the power-series form of
  `LEAP_CERTIFICATE.md`) are each proved equal to `r`.

Main results:

* `classMatrix`, `stage_eq_classMatrix`, `card_entries_le`,
  `classMatrixLin_injective`, `finrank_range_classMatrixLin`: 61 numbers
  determine the 3,600 cells, and exactly 61 degrees of freedom remain.
* `jointPayoff_eq_stage`: the per-cell branch expectation of `solver.py`
  (`transition` then `payoff`) equals the class reconstruction.
* `stage_mem_box`, `abs_V_le_one`, `certified_midpoint_mem_box`: the payoff
  box `[-1, 1]` and `F ∈ [1 - 2p, 1]`.
* `V_eq_lowerValue_upperValue`, `V_unique`, `value_checker_seat`: the Bellman
  value is the von Neumann value, backward induction defines it uniquely,
  and the Checker's seat sees `-V`.
* `rows_equal_iff`, `cols_equal_iff`: the equalization systems have the
  one-dimensional solution spaces spanned by `reverse r` and `r`.
* `mulVec_rev_weights`, `vecMul_weights`: `reverse r` and `r` equalize with
  the constant `f W + δ` for every `δ ≠ 0`, whatever the signs of `r`.
* `recB_codeB_eq_weights`, `weights_one_based`, `conv_a_weights`,
  `powerSeries_identity`, `upperToeplitz_mulVec_rev_weights`: each code form
  of the recurrence equals `r`.
-/

open Finset Matrix

set_option linter.unusedSectionVars false

namespace Formal.DTH.Stage

/-! ## The class matrix -/

/-- The C++ `matrix_cell`: `M[d, c] = S[c - d]` for `c ≥ d`, else `F`, with
zero-based `d, c`. `S k` is the success class for physical lag `k + 1`. -/
def classMatrix (S : Fin 60 → ℝ) (F : ℝ) : Matrix (Fin 60) (Fin 60) ℝ :=
  fun d c => if d ≤ c then S ⟨c.val - d.val, by omega⟩ else F

/-- Extend the 60 success classes to a sequence on `ℕ` (zero past index 59). -/
def extend (S : Fin 60 → ℝ) : ℕ → ℝ := fun k => if h : k < 60 then S ⟨k, h⟩ else 0

theorem classMatrix_eq_toeplitz (S : Fin 60 → ℝ) (F : ℝ) :
    classMatrix S F = Toeplitz.toeplitz 60 (extend S) F := by
  funext d c
  simp only [classMatrix, Toeplitz.toeplitz_apply, extend]
  split_ifs with h1 h2
  · rfl
  · omega
  · rfl

/-- The 60 success classes of a state: `S k = S_{k+1}` (`class_values` in
`main.py`, `continuation_class_values` in `solver.py`). -/
noncomputable def classS (W : State → ℝ) (x : State) : Fin 60 → ℝ :=
  fun k => succPay W x (k.val + 1)

/-- **Claims DTH-MAT-1, CANONICAL-MG-1, CPP-DOC-MAT-1, COMPACT-RULE-3.** The
61 transition classes rebuild the joint-action stage matrix entrywise
(`reconstruct_transition_class_matrix`, `src/dth/solver.py:428-446`;
`src/dth/docs/GAME_AND_SOLVER.md:102-106`). -/
theorem stage_eq_classMatrix (W : State → ℝ) (x : State) :
    stage W x = classMatrix (classS W x) (failPay W x) := by
  funext d c
  simp only [stage, classMatrix, classS]

/-- **Claims DTH-MAT-1, CANONICAL-MG-1.** The representative cells:
`(drop = 1, check = k + 1)` realizes success class `k`, and
`(drop = 2, check = 1)` realizes the failure class
(`continuation_class_values`, `src/dth/solver.py:361-389`). -/
theorem classes_realized (W : State → ℝ) (x : State) :
    (∀ k : Fin 60, classS W x k = stage W x 0 k) ∧ failPay W x = stage W x 1 0 := by
  refine ⟨fun k => ?_, ?_⟩
  · simp only [stage, classS, Fin.zero_le, ite_true, Fin.val_zero, Nat.sub_zero]
  · simp only [stage]; rfl

/-- **Claims CPP-DOC-MAT-1, PAPERS-OCAML-TOEPLITZ-1, COMPACT-MAT-1.** Every
cell strictly below the diagonal holds `F`, and every cell on or above it
holds the success class of its lag (`src/dth_cpp/BUILD.md:879-905`). Row `d`
is `d` copies of `F` followed by `S[0 .. 59 - d]`; column `c` is
`S[c], …, S[0]` followed by `F`. -/
theorem classMatrix_apply_lt (S : Fin 60 → ℝ) (F : ℝ) {d c : Fin 60} (h : c < d) :
    classMatrix S F d c = F := by
  simp only [classMatrix, not_le.mpr h, ite_false]

theorem classMatrix_apply_le (S : Fin 60 → ℝ) (F : ℝ) {d c : Fin 60} (h : d ≤ c) :
    classMatrix S F d c = S ⟨c.val - d.val, by omega⟩ := by
  simp only [classMatrix, h, ite_true]

/-- **Claims CPP-DOC-MAT-1, PAPERS-OCAML-TOEPLITZ-1, CANONICAL-MG-1,
COMPACT-MAT-1.** The class matrix is Toeplitz: it is constant along every
diagonal (`src/dth_ocaml/lib/solver/matrix_game.ml:81-84`). -/
theorem classMatrix_toeplitz (S : Fin 60 → ℝ) (F : ℝ) (d c : Fin 60)
    (hd : d.val + 1 < 60) (hc : c.val + 1 < 60) :
    classMatrix S F ⟨d.val + 1, hd⟩ ⟨c.val + 1, hc⟩ = classMatrix S F d c := by
  simp only [classMatrix, Fin.le_def]
  split_ifs with h1 h2 h2
  · congr 2; omega
  · omega
  · omega
  · rfl

/-- **Claim PAPERS-OCAML-TOEPLITZ-1.** A success cell depends only on the lag
`c - d + 1`: two cells with the same lag hold the same number. -/
theorem classMatrix_lag (S : Fin 60 → ℝ) (F : ℝ) {d c d' c' : Fin 60} (h : d ≤ c) (h' : d' ≤ c')
    (hlag : c.val - d.val = c'.val - d'.val) : classMatrix S F d c = classMatrix S F d' c' := by
  rw [classMatrix_apply_le _ _ h, classMatrix_apply_le _ _ h']
  congr 2

/-- **Claim COMPACT-MAT-1.** The class matrix is persymmetric:
`M[d, c] = M[59 - c, 59 - d]` (`src/dth_compact/architecture.md:69-72`). -/
theorem classMatrix_persymmetric (S : Fin 60 → ℝ) (F : ℝ) (d c : Fin 60) :
    classMatrix S F c.rev d.rev = classMatrix S F d c := by
  rw [classMatrix_eq_toeplitz]; exact Toeplitz.toeplitz_persymmetric _ _ d c

/-- **Claim COMPACT-MAT-1.** The matrix-vector formula of `_M_dot`
(`src/dth_compact/main.py:209-216`):
`(M q)[i] = F ∑_{j<i} q[j] + ∑_{k≥i} S[k-i] q[k]`. -/
theorem classMatrix_mulVec (S : Fin 60 → ℝ) (F : ℝ) (q : Fin 60 → ℝ) (i : Fin 60) :
    (classMatrix S F *ᵥ q) i =
      F * ∑ j ∈ univ.filter (· < i), q j +
        ∑ k ∈ univ.filter (i ≤ ·), extend S (k.val - i.val) * q k := by
  rw [classMatrix_eq_toeplitz]; exact Toeplitz.mulVec_toeplitz _ _ q i

/-- **Claim CPP-CODE-REC-1.** Subtracting the failure constant leaves an
upper-triangular Toeplitz matrix: `M - F J = toeplitz (S - F) 0`. -/
theorem classMatrix_sub_const (S : Fin 60 → ℝ) (F : ℝ) :
    classMatrix S F - Matrix.of (fun _ _ => F) =
      Toeplitz.toeplitz 60 (fun k => extend S k - F) 0 := by
  funext d c
  simp only [Matrix.sub_apply, Matrix.of_apply, classMatrix_eq_toeplitz, Toeplitz.toeplitz_apply]
  split_ifs <;> ring

set_option maxRecDepth 4000 in
/-- **Claims CPP-DOC-MAT-1, PAPERS-OCAML-TOEPLITZ-1, CANONICAL-MG-1,
COMPACT-MAT-1.** The 3,600 cells take at most 61 distinct values. -/
theorem card_entries_le (S : Fin 60 → ℝ) (F : ℝ) :
    (univ.image fun p : Fin 60 × Fin 60 => classMatrix S F p.1 p.2).card ≤ 61 := by
  have hsub : (univ.image fun p : Fin 60 × Fin 60 => classMatrix S F p.1 p.2) ⊆
      insert F (univ.image S) := by
    intro v hv
    simp only [mem_image, mem_univ, true_and] at hv
    obtain ⟨⟨d, c⟩, rfl⟩ := hv
    simp only [classMatrix]
    split_ifs
    · exact mem_insert_of_mem (mem_image_of_mem _ (mem_univ _))
    · exact mem_insert_self _ _
  calc _ ≤ (insert F (univ.image S)).card := card_le_card hsub
    _ ≤ (univ.image S).card + 1 := card_insert_le _ _
    _ ≤ (univ : Finset (Fin 60)).card + 1 := by gcongr; exact card_image_le
    _ = 61 := by simp

theorem card_cells : Fintype.card (Fin 60 × Fin 60) = 3600 := by simp

/-- The linear map `(S, F) ↦ M`. -/
def classMatrixLin : ((Fin 60 → ℝ) × ℝ) →ₗ[ℝ] Matrix (Fin 60) (Fin 60) ℝ where
  toFun x := classMatrix x.1 x.2
  map_add' x y := by
    funext d c; simp only [classMatrix, Prod.fst_add, Prod.snd_add, Pi.add_apply, Matrix.add_apply]
    split_ifs <;> rfl
  map_smul' a x := by
    funext d c
    simp only [classMatrix, Prod.smul_fst, Prod.smul_snd, Pi.smul_apply, smul_eq_mul,
      RingHom.id_apply, Matrix.smul_apply]
    split_ifs <;> rfl

/-- **Claim CANONICAL-MG-1.** The map `(S_1, …, S_60, F) ↦ M` is injective:
cell `(0, k)` reads `S k` and cell `(1, 0)` reads `F`
(`docs/FORMULATION_LADDER.md:42-44`). -/
theorem classMatrixLin_injective : Function.Injective classMatrixLin := by
  intro x y h
  change classMatrix x.1 x.2 = classMatrix y.1 y.2 at h
  have hS : ∀ k : Fin 60, x.1 k = y.1 k := fun k => by
    have := congrFun (congrFun h 0) k
    simpa [classMatrix] using this
  have hF : x.2 = y.2 := by
    have := congrFun (congrFun h 1) 0
    simpa [classMatrix] using this
  exact Prod.ext (funext hS) hF

/-- **Claim CANONICAL-MG-1.** The stage matrices form a 61-dimensional
subspace of the 3,600-dimensional matrix space: exactly `A + 1 = 61` degrees
of freedom (`paper/dth_exact_solution.tex:167-170`). -/
theorem finrank_range_classMatrixLin : Module.finrank ℝ (LinearMap.range classMatrixLin) = 61 := by
  rw [LinearMap.finrank_range_of_inj classMatrixLin_injective, Module.finrank_prod,
    Module.finrank_fin_fun, Module.finrank_self]

/-- Non-vacuity: with `S k = k` and `F = -1` the matrix attains 61 values. -/
example : (univ.image fun p : Fin 60 × Fin 60 =>
    classMatrix (fun k => (k.val : ℝ)) (-1) p.1 p.2).card = 61 := by
  apply le_antisymm (card_entries_le _ _)
  have hsub : (univ.image fun k : Fin 61 => if h : k.val < 60 then (k.val : ℝ) else -1) ⊆
      univ.image fun p : Fin 60 × Fin 60 => classMatrix (fun k => (k.val : ℝ)) (-1) p.1 p.2 := by
    intro v hv
    simp only [mem_image, mem_univ, true_and] at hv ⊢
    obtain ⟨k, rfl⟩ := hv
    by_cases hk : k.val < 60
    · refine ⟨(0, ⟨k.val, hk⟩), ?_⟩
      simp [classMatrix, hk]
    · refine ⟨(1, 0), ?_⟩
      simp [classMatrix, hk]
  refine le_trans ?_ (card_le_card hsub)
  rw [card_image_of_injective]
  · simp
  · intro a b hab
    simp only at hab
    have ha := a.isLt; have hb := b.isLt
    by_cases h1 : a.val < 60 <;> by_cases h2 : b.val < 60 <;> simp only [h1, h2, dite_true,
      dite_false] at hab
    · exact Fin.ext (by exact_mod_cast hab)
    · have : (0 : ℝ) ≤ a.val := by positivity
      linarith
    · have : (0 : ℝ) ≤ b.val := by positivity
      linarith
    · exact Fin.ext (by omega)

/-! ## One-step transitions and signs -/

/-- A branch outcome: the current Dropper wins, or play continues at a live
child whose Checker is the old Dropper. -/
inductive Outcome where
  | win
  | live (y : State)

/-- `transition` of `src/dth/solver.py:244-273`, per cell `(d, c)` of the
zero-based stage matrix. A success with lag `ℓ = c - d + 1` either overflows
(`s_c + ℓ ≥ 300`, Dropper wins) or continues at `(s_d, t_d, s_c + ℓ, t_c)`.
A failure with `p = P(s_c, t_c)` is a Dropper win when `p = 0`; otherwise
it continues at `(s_d, t_d, 0, t_c + s_c + 60)` with probability `p` and is a
Dropper win with probability `1 - p`. -/
noncomputable def transition (x : State) (d c : Fin 60) : List (ℝ × Outcome) :=
  if d ≤ c then
    if h : x.c.s + (c.val - d.val + 1) < 300 then [(1, .live (successChild x _ h))]
    else [(1, .win)]
  else if revival x.c.s x.c.t = 0 then [(1, .win)]
  else [(revival x.c.s x.c.t, .live (failChild x)), (1 - revival x.c.s x.c.t, .win)]

/-- The payoff of a branch to the current Dropper: `+1` for a win and
`-W(y)` for a live child, whose value `W y` is stated from the child's
Dropper, the opponent. -/
def branchValue (W : State → ℝ) : Outcome → ℝ
  | .win => 1
  | .live y => -W y

/-- The joint-action builder: the expected branch payoff of one cell
(`payoff`/`action_value` in `src/dth/solver.py`, `joint_payoff` in
`src/dth_ocaml/lib/solver/exact.ml:105-117`). -/
noncomputable def jointPayoff (W : State → ℝ) (x : State) (d c : Fin 60) : ℝ :=
  ((transition x d c).map fun b => b.1 * branchValue W b.2).sum

/-- Revival is possible exactly when the revival probability is nonzero, so
`solver.py`'s test `p == 0` and the eligibility test `Survives` agree. -/
theorem survives_iff_revival_ne_zero (s t : ℕ) : Survives s t ↔ revival s t ≠ 0 := by
  rw [← revival_pos_iff]
  exact ⟨fun h => h.ne', fun h => lt_of_le_of_ne (revival_nonneg s t) (Ne.symm h)⟩

/-- **Claims CANONICAL-MG-2, CPP-DOC-RULE-4, PAPERS-OCAML-SIGN-1.** The
failed-check payoff in the code's order, `F = p (-V(x_f)) + (1 - p)`, holds
for every state, fatal or not: a fatal profile has `p = 0` and pays `1`
(`src/dth_cpp/exact.cpp:318-331`, `src/dth_compact/main.py:184-188`). The
paper's `1 - p - p V(x_f)` and OCaml's `(1 - p) 1 + p cont` are the same real
number; the prescribed order matters only for floating-point parity. -/
theorem failPay_eq (W : State → ℝ) (x : State) :
    failPay W x = revival x.c.s x.c.t * (-W (failChild x)) + (1 - revival x.c.s x.c.t) ∧
      failPay W x = (1 - revival x.c.s x.c.t) * 1 + revival x.c.s x.c.t * (-W (failChild x)) := by
  unfold failPay
  split_ifs with h
  · constructor <;> ring
  · have : revival x.c.s x.c.t = 0 := by
      by_contra hne; exact h ((survives_iff_revival_ne_zero _ _).2 hne)
    rw [this]; constructor <;> ring

/-- **Claims CANONICAL-RULE-4, CANONICAL-MG-2, DTH-MAT-2.** The failure class
is `1` when the Checker's revival probability is zero, and
`1 - p - p V(x_f)` otherwise (`paper/dth_exact_solution.tex:161-166`). -/
theorem failPay_cases (W : State → ℝ) (x : State) :
    failPay W x = if revival x.c.s x.c.t = 0 then 1
      else 1 - revival x.c.s x.c.t - revival x.c.s x.c.t * W (failChild x) := by
  unfold failPay
  by_cases h : Survives x.c.s x.c.t
  · have := (survives_iff_revival_ne_zero _ _).1 h
    simp only [h, this, ite_true, ite_false]
  · have : revival x.c.s x.c.t = 0 := by
      by_contra hne; exact h ((survives_iff_revival_ne_zero _ _).2 hne)
    simp only [h, this, ite_true, ite_false]

/-- **Claims CANONICAL-RULE-4, CPP-DOC-RULE-4.** Roles swap on every live
edge: the child's Checker is the old Dropper, and the old Checker becomes the
child's Dropper with ST `s_c + ℓ` (success) or `0` and TTD `t_c + s_c + 60`
(revived failure). -/
theorem successChild_swap (x : State) (ℓ : ℕ) (h : x.c.s + ℓ < 300) :
    (successChild x ℓ h).c = x.d ∧ (successChild x ℓ h).d.s = x.c.s + ℓ ∧
      (successChild x ℓ h).d.t = x.c.t := ⟨rfl, rfl, rfl⟩

theorem failChild_swap (x : State) :
    (failChild x).c = x.d ∧ (failChild x).d.s = 0 ∧ (failChild x).d.t = x.c.t + x.c.s + 60 :=
  ⟨rfl, rfl, rfl⟩

/-- **Claim CANONICAL-RULE-4.** Every cell's branch list is a probability
distribution. -/
theorem transition_prob (x : State) (d c : Fin 60) :
    (∀ b ∈ transition x d c, 0 ≤ b.1) ∧ ((transition x d c).map Prod.fst).sum = 1 := by
  unfold transition
  split_ifs with h1 h2 h3
  · simp
  · simp
  · simp
  · have h0 := revival_nonneg x.c.s x.c.t
    have h1' := revival_le_one x.c.s x.c.t
    refine ⟨?_, by simp⟩
    intro b hb
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hb
    rcases hb with rfl | rfl
    · exact h0
    · simp only; linarith

/-- **Claim CANONICAL-RULE-4.** Every live child strictly raises the
potential, so the recursion is well founded. -/
theorem transition_live_phi (x : State) (d c : Fin 60) {pr : ℝ} {y : State}
    (hb : (pr, Outcome.live y) ∈ transition x d c) : phi x < phi y := by
  unfold transition at hb
  split_ifs at hb with h1 h2 h3
  · simp only [List.mem_singleton, Prod.mk.injEq, Outcome.live.injEq] at hb
    rw [hb.2]; exact phi_lt_successChild x (by omega) h2
  · simp at hb
  · simp at hb
  · simp only [List.mem_cons, Prod.mk.injEq, Outcome.live.injEq, List.not_mem_nil,
      or_false, reduceCtorEq, and_false] at hb
    rw [hb.2]
    exact phi_lt_failChild x ((survives_iff_revival_ne_zero _ _).2 h3)

/-- **Claims CANONICAL-RULE-4, DTH-MAT-1, DTH-MAT-2, PAPERS-OCAML-SIGN-1,
CPP-DOC-RULE-4.** The branch expectation of every cell, with `+1` for a
Dropper win and `-W(y)` for a live child, equals the stage matrix built from
the 61 classes (`src/dth/solver.py:244-273, 361-389, 530-545`;
`src/dth_ocaml/lib/solver/exact.ml:105-117`). -/
theorem jointPayoff_eq_stage (W : State → ℝ) (x : State) (d c : Fin 60) :
    jointPayoff W x d c = stage W x d c := by
  unfold jointPayoff transition stage succPay
  by_cases hdc : d ≤ c
  · simp only [hdc, ite_true]
    split_ifs with h
    · simp [branchValue]
    · simp [branchValue]
  · simp only [hdc, ite_false]
    rw [failPay_cases]
    split_ifs with h
    · simp [branchValue]
    · simp [branchValue]; ring

/-- The code's `WIN` sentinel: `main.py` stores `V[:, WIN] = -1`, so the
uniform gathers `s[k] = -V[pd, succ]` and `f = p (-V[pd, fail]) + (1 - p)`
give `+1` for an overflow or a fatal failure. `none` is `WIN`. -/
def withWin (W : State → ℝ) : Option State → ℝ
  | none => -1
  | some y => W y

/-- The success child id, `none` (`WIN`) on overflow. -/
def succChildOpt (x : State) (ℓ : ℕ) : Option State :=
  if h : x.c.s + ℓ < 300 then some (successChild x ℓ h) else none

/-- The failure child id, `none` (`WIN`) for a failure-fatal Checker. -/
def failChildOpt (x : State) : Option State :=
  if Survives x.c.s x.c.t then some (failChild x) else none

/-- **Claims COMPACT-RULE-3, CPP-CODE-BELL-1, CANONICAL-MG-2.** The sentinel
gather of `_solve_layer_kernel` (`src/dth_compact/main.py:389-392`) and the
`-1` child ids of `fast_kernel.c:29-43` reproduce the success and failure
classes. -/
theorem sentinel_gather (W : State → ℝ) (x : State) (ℓ : ℕ) :
    succPay W x ℓ = -withWin W (succChildOpt x ℓ) ∧
      failPay W x = revival x.c.s x.c.t * (-withWin W (failChildOpt x)) +
        (1 - revival x.c.s x.c.t) := by
  constructor
  · unfold succPay succChildOpt
    split_ifs <;> simp [withWin]
  · rw [(failPay_eq W x).1]
    unfold failChildOpt
    split_ifs with h
    · rfl
    · have : revival x.c.s x.c.t = 0 := by
        by_contra hne; exact h ((survives_iff_revival_ne_zero _ _).2 hne)
      simp [withWin, this]

/-! ## The payoff box `[-1, 1]` -/

/-- **Claims CPP-DOC-RULE-4, CPP-CODE-BELL-2, PAPERS-OCAML-SIGN-1.** With
child values in `[-1, 1]`, the failed-check class lies in `[1 - 2p, 1]`
(`src/dth_cpp/BUILD.md:909-934`). -/
theorem failPay_mem (W : State → ℝ) (x : State) (hW : ∀ y, |W y| ≤ 1) :
    1 - 2 * revival x.c.s x.c.t ≤ failPay W x ∧ failPay W x ≤ 1 := by
  rw [(failPay_eq W x).1]
  have h0 := revival_nonneg x.c.s x.c.t
  have hy := abs_le.mp (hW (failChild x))
  constructor <;> nlinarith [hy.1, hy.2]

/-- **Claims CPP-DOC-RULE-4, CPP-CODE-BELL-2, PAPERS-OCAML-SIGN-1.** With
child values in `[-1, 1]`, all 61 classes, and so all 3,600 cells, lie in
`[-1, 1]`. The `1e-9` slack of `assemble_transition_values`
(`src/dth_cpp/exact.cpp:335-343`) never fires in exact arithmetic. -/
theorem stage_mem_box (W : State → ℝ) (x : State) (hW : ∀ y, |W y| ≤ 1) (d c : Fin 60) :
    |stage W x d c| ≤ 1 := by
  unfold stage
  split_ifs
  · unfold succPay
    split_ifs
    · rw [abs_neg]; exact hW _
    · simp
  · have h := failPay_mem W x hW
    have h1 := revival_le_one x.c.s x.c.t
    rw [abs_le]; constructor <;> linarith [h.1, h.2]

/-- **Claims CPP-DOC-RULE-4, CPP-CODE-BELL-2, PAPERS-OCAML-SIGN-1.** The
value of a stage whose children lie in `[-1, 1]` lies in `[-1, 1]`. -/
theorem value_stage_mem_box (W : State → ℝ) (x : State) (hW : ∀ y, |W y| ≤ 1) :
    |MatrixGame.value (stage W x)| ≤ 1 := by
  have h := MatrixGame.value_mem_entry_range (stage W x) (a := -1) (b := 1)
    (fun i j => (abs_le.mp (stage_mem_box W x hW i j)).1)
    (fun i j => (abs_le.mp (stage_mem_box W x hW i j)).2)
  exact abs_le.mpr h

/-- **Claims CPP-CODE-BELL-2, PAPERS-OCAML-SIGN-1, CPP-DOC-RULE-4.** By
induction over the potential, every true class value lies in `[-1, 1]`. -/
theorem abs_V_le_one (x : State) : |V x| ≤ 1 := by
  suffices h : ∀ k, ∀ x : State, 1201 - phi x = k → |V x| ≤ 1 from h _ x rfl
  intro k
  induction k using Nat.strong_induction_on with
  | _ k ih =>
    intro x hx
    rw [V_bellman, stage_congr (W' := fun y => if phi x < phi y then V y else 0) x
      (fun y hy => by simp [hy])]
    apply value_stage_mem_box
    intro y
    by_cases hy : phi x < phi y
    · simp only [hy, ite_true]
      exact ih _ (by have := phi_le y; omega) y rfl
    · simp [hy]

/-- **Claims PAPERS-OCAML-SIGN-1, CPP-CODE-BELL-2.** For a matrix with entries
in `[-1, 1]`, both certificate bounds of any feasible pair, and so the stored
midpoint, lie in `[-1, 1]`. The range checks of `matrix_game.ml:164-169` and
`matrix_game.cpp:132-134` are redundant in exact arithmetic and guard only
floating point. -/
theorem certified_midpoint_mem_box {M : Matrix (Fin 60) (Fin 60) ℝ} (hM : ∀ i j, |M i j| ≤ 1)
    {p q : Fin 60 → ℝ} (hp : p ∈ MatrixGame.simplex (Fin 60)) (hq : q ∈ MatrixGame.simplex (Fin 60)) :
    |MatrixGame.lowerBound M p| ≤ 1 ∧ |MatrixGame.upperBound M q| ≤ 1 ∧
      |(MatrixGame.lowerBound M p + MatrixGame.upperBound M q) / 2| ≤ 1 := by
  have hcol : ∀ j, -1 ≤ (p ᵥ* M) j ∧ (p ᵥ* M) j ≤ 1 := fun j =>
    ⟨MatrixGame.le_dotProduct_of_mem_simplex hp fun i => (abs_le.mp (hM i j)).1,
      MatrixGame.dotProduct_le_of_mem_simplex hp fun i => (abs_le.mp (hM i j)).2⟩
  have hrow : ∀ i, -1 ≤ (M *ᵥ q) i ∧ (M *ᵥ q) i ≤ 1 := fun i => by
    rw [show (M *ᵥ q) i = q ⬝ᵥ M i from dotProduct_comm _ _]
    exact ⟨MatrixGame.le_dotProduct_of_mem_simplex hq fun j => (abs_le.mp (hM i j)).1,
      MatrixGame.dotProduct_le_of_mem_simplex hq fun j => (abs_le.mp (hM i j)).2⟩
  have hl1 : -1 ≤ MatrixGame.lowerBound M p :=
    (MatrixGame.le_lowerBound_iff M p _).2 fun j => (hcol j).1
  have hl2 : MatrixGame.lowerBound M p ≤ 1 :=
    (MatrixGame.lowerBound_le_col M p 0).trans (hcol 0).2
  have hu1 : MatrixGame.upperBound M q ≤ 1 :=
    (MatrixGame.upperBound_le_iff M q _).2 fun i => (hrow i).2
  have hu2 : -1 ≤ MatrixGame.upperBound M q :=
    (hrow 0).1.trans (MatrixGame.row_le_upperBound M q 0)
  refine ⟨abs_le.mpr ⟨hl1, hl2⟩, abs_le.mpr ⟨hu2, hu1⟩, abs_le.mpr ⟨by linarith, by linarith⟩⟩

/-- **Claim CPP-CODE-BELL-2, exact half.** Entries in `[-1, 1]` shift into
`[1, 3]`, the range that `try_linear_program` requires
(`src/dth_cpp/matrix_game.cpp:262-270`). -/
theorem lp_shift_mem {m : ℝ} (h : |m| ≤ 1) : 1 ≤ m + 2 ∧ m + 2 ≤ 3 := by
  have := abs_le.mp h; constructor <;> linarith [this.1, this.2]

/-- **Claim CPP-CODE-BELL-2, tolerance doubt.** The assembly box accepts
`|m| ≤ 1 + 1e-9`, and the LP rung demands `m + 2 ≤ 3` exactly. A value in
the slack band passes the first check and fails the second, so such a stage
aborts the build (fail closed); it never yields a wrong value. -/
theorem box_slack_lp_mismatch :
    ∃ m : ℝ, |m| ≤ 1 + 1e-9 ∧ ¬ (1 ≤ m + 2 ∧ m + 2 ≤ 3) :=
  ⟨1 + 1e-10, by rw [abs_of_pos (by norm_num)]; norm_num, by norm_num⟩

/-- The C++ assembly test (`test_transition_value_assembly_gate`,
`src/dth_cpp/tests.cpp`): `p = 0.25` and a stored child value `0.5` give
`F = 0.625`. -/
example : (0.25 : ℝ) * (-(0.5 : ℝ)) + (1 - 0.25) = 0.625 := by norm_num

/-! ## The Bellman value -/

/-- **Claims DTH-MAT-2, COMPACT-RULE-3, CPP-CODE-BELL-1.** `V x` is the von
Neumann value of its stage matrix: the Dropper's max-min over mixed rows
equals the Checker's min-max over mixed columns, and optimal mixtures attain
it (`src/dth/docs/GAME_AND_SOLVER.md:79-100`). -/
theorem V_eq_lowerValue_upperValue (x : State) :
    V x = MatrixGame.lowerValue (stage V x) ∧ V x = MatrixGame.upperValue (stage V x) ∧
      ∃ p ∈ MatrixGame.simplex (Fin 60), ∃ q ∈ MatrixGame.simplex (Fin 60),
        MatrixGame.lowerBound (stage V x) p = V x ∧ MatrixGame.upperBound (stage V x) q = V x := by
  rw [V_bellman]
  refine ⟨rfl, MatrixGame.value_eq_upperValue _, ?_⟩
  obtain ⟨p, hp, q, hq, h⟩ := MatrixGame.exists_optimal (stage V x)
  have e := MatrixGame.certificate_encloses_value (stage V x) hp hq
  refine ⟨p, hp, q, hq, ?_, ?_⟩
  · apply le_antisymm e.1; rw [← h]; exact e.2
  · apply le_antisymm _ e.2; rw [h]; exact e.1

/-- **Claim DTH-MAT-2.** Backward induction defines `V` uniquely: any function
that satisfies the Bellman equation at every state is `V`. -/
theorem V_unique (W : State → ℝ) (hW : ∀ x, W x = MatrixGame.value (stage W x)) (x : State) :
    W x = V x := by
  suffices h : ∀ k, ∀ x : State, 1201 - phi x = k → W x = V x from h _ x rfl
  intro k
  induction k using Nat.strong_induction_on with
  | _ k ih =>
    intro x hx
    rw [hW, V_bellman, stage_congr x fun y hy => ih _ (by have := phi_le y; omega) y rfl]

/-- **Claims DTH-MAT-2, CANONICAL-RULE-4, COMPACT-RULE-3.** The zero-sum sign
convention: seen from the Checker's seat (the Checker as row maximizer of
`-Mᵀ`) the stage is worth `-V x`. This is why a parent negates a child's
value: the child's Dropper is the parent's Checker. -/
theorem value_checker_seat (x : State) : MatrixGame.value (-(stage V x)ᵀ) = -V x := by
  rw [MatrixGame.value_neg_transpose, ← V_bellman]

/-! ## The equalizer recurrence, as each implementation writes it -/

section Recurrence

open Toeplitz

variable {n : ℕ} (s : ℕ → ℝ) (f : ℝ)

/-- Zero extension of a vector on `Fin n` to `ℕ`. -/
def extN (q : Fin n → ℝ) : ℕ → ℝ := fun k => if h : k < n then q ⟨k, h⟩ else 0

theorem extN_of_lt (q : Fin n → ℝ) {k : ℕ} (h : k < n) : extN q k = q ⟨k, h⟩ := by
  simp [extN, h]

/-- The compact recurrence with a range sum:
`r[k+1] = -(∑_{m ≤ k} dS[m] r[k-m]) / d0` (`try_rung2`,
`src/dth_compact/main.py:250-265`). -/
theorem weights_succ (k : ℕ) :
    weights s f (k + 1) =
      -(∑ m ∈ range (k + 1), (s (m + 1) - s m) * weights s f (k - m)) / (s 0 - f) := by
  rw [weights, ← Fin.sum_univ_eq_sum_range (fun m => (s (m + 1) - s m) * weights s f (k - m))]

theorem sum_filter_gt_reindex (g : ℕ → ℝ) (q : Fin n → ℝ) (i : Fin n) :
    ∑ k ∈ univ.filter (i < ·), g (k.val - i.val) * q k =
      ∑ m ∈ range (n - 1 - i.val), g (m + 1) * extN q (i.val + 1 + m) := by
  refine Finset.sum_bij' (fun k _ => k.val - i.val - 1)
    (fun m hm => ⟨i.val + 1 + m, by simp only [mem_range] at hm; omega⟩) ?_ ?_ ?_ ?_ ?_
  · intro k hk
    simp only [mem_filter, mem_univ, true_and] at hk
    simp only [mem_range]; have := Fin.lt_def.mp hk; omega
  · intro m hm
    simp only [mem_range] at hm
    simp only [mem_filter, mem_univ, true_and, Fin.lt_def]; omega
  · intro k hk
    simp only [mem_filter, mem_univ, true_and] at hk
    ext; simp only; have := Fin.lt_def.mp hk; omega
  · intro m hm; simp only [mem_range] at hm; dsimp only; omega
  · intro k hk
    simp only [mem_filter, mem_univ, true_and] at hk
    have hk' := Fin.lt_def.mp hk
    rw [extN_of_lt q (by omega)]
    congr 1
    · congr 1; omega
    · congr 1; ext; simp only; omega

theorem sum_filter_ge_reindex (g : ℕ → ℝ) (q : Fin n → ℝ) (i : Fin n) :
    ∑ k ∈ univ.filter (i ≤ ·), g (k.val - i.val) * q k =
      ∑ m ∈ range (n - i.val), g m * extN q (i.val + m) := by
  refine Finset.sum_bij' (fun k _ => k.val - i.val)
    (fun m hm => ⟨i.val + m, by simp only [mem_range] at hm; omega⟩) ?_ ?_ ?_ ?_ ?_
  · intro k hk
    simp only [mem_filter, mem_univ, true_and] at hk
    simp only [mem_range]; have := Fin.le_def.mp hk; omega
  · intro m hm
    simp only [mem_range] at hm
    simp only [mem_filter, mem_univ, true_and, Fin.le_def]; omega
  · intro k hk
    simp only [mem_filter, mem_univ, true_and] at hk
    ext; simp only; have := Fin.le_def.mp hk; omega
  · intro m hm; simp only [mem_range] at hm; dsimp only; omega
  · intro k hk
    simp only [mem_filter, mem_univ, true_and] at hk
    have hk' := Fin.le_def.mp hk
    rw [extN_of_lt q (by omega)]
    congr 1
    congr 1; ext; simp only; omega

/-- **Claims COMPACT-RUNG2-1, CANONICAL-MG-6, STL-REC-1.** Adjacent rows of
`M q` differ by `d0 q[i] + ∑_{m < 59 - i} dS[m] q[i + 1 + m]`, with
`d0 = s[0] - f` and `dS[m] = s[m+1] - s[m]`
(`src/dth_compact/architecture.md:169-186`). In the paper's 1-based form
this is `D b_d + ∑_{m=1}^{60-d} Δ_m b_{d+m}` with `d = i + 1`, `m ↦ m + 1`
(`paper/dth_exact_solution.tex:176-192`). -/
theorem rowDiff (q : Fin n → ℝ) (i : Fin n) (hi : i.val + 1 < n) :
    (toeplitz n s f *ᵥ q) i - (toeplitz n s f *ᵥ q) ⟨i.val + 1, hi⟩ =
      (s 0 - f) * q i +
        ∑ m ∈ range (n - 1 - i.val), (s (m + 1) - s m) * extN q (i.val + 1 + m) := by
  rw [mulVec_sub_succ, sum_filter_gt_reindex (fun a => s a - s (a - 1))]
  simp only [Nat.add_sub_cancel]

/-- The triangular system of the recurrence has the one-dimensional solution
space spanned by `r`: a sequence satisfies the row-difference equations up to
index `N` exactly when it is `u 0 • r` there. -/
theorem rec_iff (hd : s 0 - f ≠ 0) (u : ℕ → ℝ) (N : ℕ) :
    (∀ a, 1 ≤ a → a < N →
        (s 0 - f) * u a + ∑ m ∈ range a, (s (m + 1) - s m) * u (a - 1 - m) = 0) ↔
      ∀ a < N, u a = u 0 * weights s f a := by
  constructor
  · intro h a
    induction a using Nat.strong_induction_on with
    | _ a ih =>
      intro haN
      rcases Nat.eq_zero_or_pos a with rfl | ha
      · rw [weights_zero, mul_one]
      · have e := h a ha haN
        have hsum : ∑ m ∈ range a, (s (m + 1) - s m) * u (a - 1 - m) =
            u 0 * ∑ m ∈ range a, (s (m + 1) - s m) * weights s f (a - 1 - m) := by
          rw [Finset.mul_sum]
          refine sum_congr rfl fun m hm => ?_
          simp only [mem_range] at hm
          rw [ih (a - 1 - m) (by omega) (by omega)]; ring
        have hw := weights_rec s f hd ha
        rw [hsum] at e
        apply mul_left_cancel₀ hd
        linear_combination e - u 0 * hw
  · intro h a ha1 haN
    have hw := weights_rec s f hd ha1
    have hsum : ∑ m ∈ range a, (s (m + 1) - s m) * u (a - 1 - m) =
        u 0 * ∑ m ∈ range a, (s (m + 1) - s m) * weights s f (a - 1 - m) := by
      rw [Finset.mul_sum]
      refine sum_congr rfl fun m hm => ?_
      simp only [mem_range] at hm
      rw [h (a - 1 - m) (by omega)]; ring
    rw [hsum, h a haN]
    linear_combination u 0 * hw

/-- Adjacent equalities chain to the last entry. -/
theorem eq_last_of_adjacent {v : Fin n → ℝ} (hn : 0 < n)
    (h : ∀ i : Fin n, ∀ hi : i.val + 1 < n, v i = v ⟨i.val + 1, hi⟩) (i : Fin n) :
    v i = v ⟨n - 1, by omega⟩ := by
  suffices H : ∀ t, ∀ i : Fin n, i.val + t = n - 1 → v i = v ⟨n - 1, by omega⟩ from
    H (n - 1 - i.val) i (by have := i.isLt; omega)
  intro t
  induction t with
  | zero => intro i hi; congr 1; ext; simp only; omega
  | succ t ih =>
    intro i hi
    rw [h i (by omega), ih _ (by simp only; omega)]

/-- **Claims COMPACT-RUNG2-1, DTH-REC-1, CANONICAL-MG-6, CPP-DOC-REC-1.** The
Checker side: `M q` has equal adjacent rows exactly when
`q[j] = q[n-1] r[n-1-j]`, so the solution space is the line through
`reverse r`, and `r[0] = 1` fixes the scale
(`src/dth_compact/main.py:240-249`). -/
theorem rows_equal_iff (hd : s 0 - f ≠ 0) (hn : 0 < n) (q : Fin n → ℝ) :
    (∀ i : Fin n, ∀ hi : i.val + 1 < n,
        (toeplitz n s f *ᵥ q) i = (toeplitz n s f *ᵥ q) ⟨i.val + 1, hi⟩) ↔
      ∀ j : Fin n, q j = q ⟨n - 1, by omega⟩ * weights s f (n - 1 - j.val) := by
  set u : ℕ → ℝ := fun a => extN q (n - 1 - a) with hu
  have hu0 : u 0 = q ⟨n - 1, by omega⟩ := by
    simp only [hu, Nat.sub_zero]; exact extN_of_lt q (by omega)
  have hua : ∀ a (ha : a < n), u a = q ⟨n - 1 - a, by omega⟩ := fun a ha =>
    extN_of_lt q (by omega)
  have key : ∀ i : Fin n, ∀ hi : i.val + 1 < n,
      (toeplitz n s f *ᵥ q) i - (toeplitz n s f *ᵥ q) ⟨i.val + 1, hi⟩ =
        (s 0 - f) * u (n - 1 - i.val) +
          ∑ m ∈ range (n - 1 - i.val), (s (m + 1) - s m) * u (n - 1 - i.val - 1 - m) := by
    intro i hi
    rw [rowDiff]
    congr 1
    · rw [hua _ (by omega)]; congr 2; ext; simp only; omega
    · refine sum_congr rfl fun m hm => ?_
      simp only [mem_range] at hm
      simp only [hu]; congr 2; omega
  have hrec := rec_iff s f hd u n
  constructor
  · intro h
    have H : ∀ a < n, u a = u 0 * weights s f a := by
      refine hrec.1 fun a ha1 haN => ?_
      have hi : (n - 1 - a) + 1 < n := by omega
      have e := key ⟨n - 1 - a, by omega⟩ hi
      rw [h _ hi, sub_self] at e
      simp only at e
      rw [show n - 1 - (n - 1 - a) = a by omega] at e
      exact e.symm
    intro j
    have e := H (n - 1 - j.val) (by omega)
    rw [hua _ (by omega), hu0] at e
    rw [← e]; congr 1; ext; simp only; omega
  · intro h
    have H : ∀ a < n, u a = u 0 * weights s f a := by
      intro a ha
      rw [hua a ha, hu0, h ⟨n - 1 - a, by omega⟩]
      congr 2; simp only; omega
    have hr := hrec.2 H
    intro i hi
    have e := key i hi
    rw [hr _ (by omega) (by omega)] at e
    linarith

/-- **Claims DTH-REC-1, CPP-DOC-REC-1, CPP-CODE-REC-1, CANONICAL-MG-6.** The
Dropper side: `pᵀ M` has equal adjacent columns exactly when
`p[i] = p[0] r[i]` (`src/dth/docs/FAST_TABLEBASE.md:25-37`). With
`p[0] ≠ 0` this says `p` is proportional to `r`; with `p[0] = 0` it forces
`p = 0`, so fixing `r[0] = 1` loses no generality. -/
theorem cols_equal_iff (hd : s 0 - f ≠ 0) (hn : 0 < n) (p : Fin n → ℝ) :
    (∀ c : Fin n, ∀ hc : c.val + 1 < n,
        (p ᵥ* toeplitz n s f) ⟨c.val + 1, hc⟩ = (p ᵥ* toeplitz n s f) c) ↔
      ∀ i : Fin n, p i = p ⟨0, hn⟩ * weights s f i.val := by
  set q : Fin n → ℝ := fun i => p i.rev with hq
  have hp : p = fun i => q i.rev := by funext i; simp [hq]
  have hcol : ∀ j, (p ᵥ* toeplitz n s f) j = (toeplitz n s f *ᵥ q) j.rev := by
    intro j; rw [hp]; exact vecMul_rev_eq_mulVec s f q j
  have hswap : (∀ c : Fin n, ∀ hc : c.val + 1 < n,
        (p ᵥ* toeplitz n s f) ⟨c.val + 1, hc⟩ = (p ᵥ* toeplitz n s f) c) ↔
      ∀ i : Fin n, ∀ hi : i.val + 1 < n,
        (toeplitz n s f *ᵥ q) i = (toeplitz n s f *ᵥ q) ⟨i.val + 1, hi⟩ := by
    constructor
    · intro h i hi
      have e := h ⟨n - 2 - i.val, by omega⟩ (by simp only; omega)
      rw [hcol, hcol] at e
      convert e using 2 <;> ext <;> simp only [Fin.val_rev] <;> omega
    · intro h c hc
      rw [hcol, hcol]
      have e := h ⟨n - 2 - c.val, by omega⟩ (by simp only; omega)
      convert e using 2 <;> ext <;> simp only [Fin.val_rev] <;> omega
  rw [hswap, rows_equal_iff s f hd hn q]
  have h0 : q ⟨n - 1, by omega⟩ = p ⟨0, hn⟩ := by
    simp only [hq]; congr 1; ext; simp only [Fin.val_rev]; omega
  rw [h0]
  constructor
  · intro H i
    have e := H i.rev
    simp only [hq, Fin.rev_rev, Fin.val_rev] at e
    rw [e]; congr 2; omega
  · intro H j
    simp only [hq]
    rw [H j.rev]; simp only [Fin.val_rev]
    congr 2; omega

/-- **Claims CPP-DOC-REC-1, DTH-REC-1.** A Dropper equalizer with `p[0] = 0`
is zero. -/
theorem cols_equal_zero (hd : s 0 - f ≠ 0) (hn : 0 < n) (p : Fin n → ℝ)
    (h : ∀ c : Fin n, ∀ hc : c.val + 1 < n,
        (p ᵥ* toeplitz n s f) ⟨c.val + 1, hc⟩ = (p ᵥ* toeplitz n s f) c)
    (h0 : p ⟨0, hn⟩ = 0) : p = 0 := by
  funext i
  rw [(cols_equal_iff s f hd hn p).1 h i, h0, zero_mul]; rfl

/-- **Claims CPP-DOC-REC-1, STL-REC-1, CANONICAL-MG-6, COMPACT-RUNG2-1.** For
every `δ = s[0] - f ≠ 0`, whatever the signs of the weights, the unnormalized
Checker vector `q[j] = r[n-1-j]` makes every row of `M q` pay `f W + δ`,
`W = ∑ r` (`LEAP_CERTIFICATE.md`, last-row formula). -/
theorem mulVec_rev_weights (hd : s 0 - f ≠ 0) (hn : 0 < n) (i : Fin n) :
    (toeplitz n s f *ᵥ fun j => weights s f (n - 1 - j.val)) i =
      f * weightSum n s f + (s 0 - f) := by
  set q : Fin n → ℝ := fun j => weights s f (n - 1 - j.val) with hq
  have hadj := (rows_equal_iff s f hd hn q).2 fun j => by
    simp only [hq]; rw [show n - 1 - (n - 1) = 0 by omega, weights_zero, one_mul]
  rw [eq_last_of_adjacent hn hadj i, mulVec_toeplitz]
  set L : Fin n := ⟨n - 1, by omega⟩ with hL
  have hLval : L.val = n - 1 := rfl
  have hfilter : univ.filter (fun k : Fin n => L ≤ k) = {L} := by
    ext k; simp only [mem_filter, mem_univ, true_and, mem_singleton]
    constructor
    · intro h; apply Fin.ext; have h1 := Fin.le_def.mp h; have := k.isLt
      rw [hLval] at h1 ⊢; omega
    · rintro rfl; exact le_rfl
  have hlt : univ.filter (fun j : Fin n => j < L) = univ.erase L := by
    ext j; simp only [mem_filter, mem_univ, true_and, mem_erase, and_true]
    constructor
    · exact ne_of_lt
    · intro h; rw [Fin.lt_def]; have := j.isLt
      have h1 : j.val ≠ L.val := fun e => h (Fin.ext e)
      rw [hLval] at h1 ⊢; omega
  have htotal : ∑ j, q j = weightSum n s f := by
    rw [weightSum, ← Fin.sum_univ_eq_sum_range]
    exact Fintype.sum_equiv Fin.revPerm _ _ fun j => by
      simp only [hq, Fin.revPerm_apply, Fin.val_rev]; congr 1; omega
  have hqL : q L = 1 := by simp only [hq, hL, Nat.sub_self, weights_zero]
  rw [hfilter, hlt, Finset.sum_singleton, Nat.sub_self, Finset.sum_erase_eq_sub (mem_univ L),
    htotal, hqL]
  ring

/-- **Claims CPP-DOC-REC-1, CPP-CODE-REC-1, STL-REC-1.** The unnormalized
Dropper vector `p = r` makes every column of `pᵀ M` pay `f W + δ`. -/
theorem vecMul_weights (hd : s 0 - f ≠ 0) (hn : 0 < n) (j : Fin n) :
    ((fun i : Fin n => weights s f i.val) ᵥ* toeplitz n s f) j =
      f * weightSum n s f + (s 0 - f) := by
  have e : (fun i : Fin n => weights s f i.val) =
      fun i => (fun j : Fin n => weights s f (n - 1 - j.val)) i.rev := by
    funext i; simp only [Fin.val_rev]; congr 1; omega
  rw [e]
  exact (vecMul_rev_eq_mulVec s f _ j).trans (mulVec_rev_weights s f hd hn _)

/-- **Claims CPP-CODE-REC-1, CANONICAL-MG-6.** Adjacent columns of `pᵀ M`
differ by `δ p[c+1] + ∑_{d ≤ c} p[d] (s[c+1-d] - s[c-d])`. Setting the
difference to zero is the forward-substitution step of the recurrence with
`k = c + 1`, `j = d`. The paper's 1-based form
`D a_{c+1} + ∑_{m=1}^{c} Δ_m a_{c+1-m}` is the same sum reflected. -/
theorem colDiff (p : Fin n → ℝ) (c : Fin n) (hc : c.val + 1 < n) :
    (p ᵥ* toeplitz n s f) ⟨c.val + 1, hc⟩ - (p ᵥ* toeplitz n s f) c =
      (s 0 - f) * p ⟨c.val + 1, hc⟩ +
        ∑ d ∈ range (c.val + 1), (s (c.val + 1 - d) - s (c.val - d)) * extN p d := by
  set q : Fin n → ℝ := fun i => p i.rev with hq
  have hp : p = fun i => q i.rev := by funext i; simp [hq]
  have hcol : ∀ j, (p ᵥ* toeplitz n s f) j = (toeplitz n s f *ᵥ q) j.rev := by
    intro j; rw [hp]; exact vecMul_rev_eq_mulVec s f q j
  have hi : n - 2 - c.val + 1 < n := by omega
  have e1 : (⟨c.val + 1, hc⟩ : Fin n).rev = ⟨n - 2 - c.val, by omega⟩ := by
    ext; simp only [Fin.val_rev]; omega
  have e2 : c.rev = ⟨n - 2 - c.val + 1, hi⟩ := by ext; simp only [Fin.val_rev]; omega
  rw [hcol, hcol, e1, e2, rowDiff]
  dsimp only
  congr 1
  · simp only [hq]; congr 2; ext; simp only [Fin.val_rev]; omega
  · rw [show n - 1 - (n - 2 - c.val) = c.val + 1 by omega]
    rw [← Finset.sum_range_reflect (fun d => (s (c.val + 1 - d) - s (c.val - d)) * extN p d)
      (c.val + 1)]
    refine sum_congr rfl fun m hm => ?_
    simp only [mem_range] at hm
    rw [extN_of_lt q (by omega), extN_of_lt p (by omega)]
    simp only [hq]
    congr 1
    · rw [show c.val + 1 - (c.val + 1 - 1 - m) = m + 1 by omega,
        show c.val - (c.val + 1 - 1 - m) = m by omega]
    · congr 1; ext; simp only [Fin.val_rev]; omega

/-! ### The `b`-form of C, C++, Rust and `fast_kernel.py` -/

/-- The code coefficients `b[0] = 0`, `b[m] = (s[m-1] - s[m]) / δ`
(`try_recurrence`, `src/dth_cpp/matrix_game.cpp:376-392`; `Q` in
`recurrence.cpp` and `fast_kernel.c`; `b` in `leap_oracle.py`,
`leap_support.py:81-92`, `leap.rs:387-424`). -/
noncomputable def codeB (m : ℕ) : ℝ := if m = 0 then 0 else (s (m - 1) - s m) / (s 0 - f)

/-- The code recurrence `r[0] = 1`, `r[k] = ∑_{j<k} b[k-j] r[j]`. -/
noncomputable def recB (b : ℕ → ℝ) : ℕ → ℝ
  | 0 => 1
  | k + 1 => ∑ j : Fin (k + 1), b (k + 1 - j.val) * recB b j.val
termination_by k => k
decreasing_by omega

theorem recB_succ (b : ℕ → ℝ) (k : ℕ) :
    recB b (k + 1) = ∑ j ∈ range (k + 1), b (k + 1 - j) * recB b j := by
  rw [recB, ← Fin.sum_univ_eq_sum_range (fun j => b (k + 1 - j) * recB b j)]

/-- **Claims CPP-DOC-REC-1, CPP-CODE-REC-1, DTH-REC-1, STL-REC-1,
CANONICAL-MG-6.** The `b`-form recurrence of the C, C++, Rust and
`fast_kernel.py` kernels computes the same weights as the compact form
`weights`. The equality holds for every `δ`, with Lean's `x / 0 = 0` on both
sides; the kernels only accept `|δ| ≥ 1e-12`. `fast_kernel.py` stores
`b[i] = doc b[i+1]` and evaluates `np.dot(b[:k], r[k-1::-1])`, the same sum
(`src/dth/fast_kernel.py:26-41`). -/
theorem recB_codeB_eq_weights (k : ℕ) : recB (codeB s f) k = weights s f k := by
  induction k using Nat.strong_induction_on with
  | _ k ih =>
    rcases k with _ | k
    · rw [recB, weights_zero]
    · rw [recB_succ, weights_succ]
      rw [← Finset.sum_range_reflect (fun j => codeB s f (k + 1 - j) * recB (codeB s f) j) (k + 1)]
      rw [neg_div, Finset.sum_div, ← Finset.sum_neg_distrib]
      refine sum_congr rfl fun m hm => ?_
      simp only [mem_range] at hm
      rw [show k + 1 - 1 - m = k - m by omega, ih (k - m) (by omega),
        show k + 1 - (k - m) = m + 1 by omega]
      simp only [codeB, Nat.add_sub_cancel, Nat.add_one_ne_zero, ite_false]
      ring

/-- **Claims CANONICAL-MG-6, COMPACT-RUNG2-1.** The paper's 1-based form:
with `r₁[k] = r[k-1]`, `D = S_1 - F` and `Δ_m = S_{m+1} - S_m = s[m] - s[m-1]`,
`r₁[k] = -(∑_{m=1}^{k-1} Δ_m r₁[k-m]) / D` for `2 ≤ k`
(`paper/dth_exact_solution.tex:176-192`; `architecture.md:169-186`). The
doc's Checker mixture `q[c] = max(r₁[61-c], 0)` (1-based `c`) is
`max(r[59-c'], 0)` for `c' = c - 1`. -/
theorem weights_one_based (k : ℕ) (hk : 2 ≤ k) :
    weights s f (k - 1) =
      -(∑ m ∈ Ico 1 k, (s m - s (m - 1)) * weights s f (k - m - 1)) / (s 0 - f) := by
  obtain ⟨j, rfl⟩ : ∃ j, k = j + 2 := ⟨k - 2, by omega⟩
  rw [show j + 2 - 1 = j + 1 by omega, weights_succ, Finset.sum_Ico_eq_sum_range]
  congr 2
  rw [show j + 2 - 1 = j + 1 by omega]
  refine sum_congr rfl fun m hm => ?_
  simp only [mem_range] at hm
  rw [show 1 + m - 1 = m by omega, show j + 2 - (1 + m) - 1 = j - m by omega, add_comm 1 m]

/-! ### The power-series form of `LEAP_CERTIFICATE.md` -/

/-- `a[k] = (f - s[k]) / (f - s[0])`, so `a[0] = 1` (`LEAP_CERTIFICATE.md`,
recurrence crash basis). -/
noncomputable def aSeq (k : ℕ) : ℝ := (f - s k) / (f - s 0)

theorem aSeq_zero (hd : s 0 - f ≠ 0) : aSeq s f 0 = 1 := by
  have : f - s 0 ≠ 0 := fun h => hd (by linarith)
  simp [aSeq, this]

/-- **Claim STL-REC-1.** The kernel coefficients are the first differences of
`a`: `b[m] = a[m-1] - a[m]` for `m ≥ 1`. -/
theorem codeB_eq_aSeq (hd : s 0 - f ≠ 0) {m : ℕ} (hm : 1 ≤ m) :
    codeB s f m = aSeq s f (m - 1) - aSeq s f m := by
  have h' : f - s 0 ≠ 0 := fun h => hd (by linarith)
  simp only [codeB, aSeq, show m ≠ 0 by omega, ite_false]
  field_simp
  ring

/-- **Claim STL-REC-1.** The convolution identity `∑_{m ≤ k} a[m] r[k-m] = 1`
for every `k`: `a · r = 1 / (1 - x)` coefficientwise. -/
theorem conv_a_weights (hd : s 0 - f ≠ 0) (k : ℕ) :
    ∑ m ∈ range (k + 1), aSeq s f m * weights s f (k - m) = 1 := by
  have h' : f - s 0 ≠ 0 := fun h => hd (by linarith)
  induction k with
  | zero => simp [aSeq_zero s f hd, weights_zero]
  | succ k ih =>
    rw [Finset.sum_range_succ', aSeq_zero s f hd, one_mul, Nat.sub_zero, weights_succ]
    have hdiff : ∀ m, -(s (m + 1) - s m) / (s 0 - f) = aSeq s f m - aSeq s f (m + 1) := by
      intro m; simp only [aSeq]; field_simp; ring
    have hw : -(∑ m ∈ range (k + 1), (s (m + 1) - s m) * weights s f (k - m)) / (s 0 - f) =
        ∑ m ∈ range (k + 1), (aSeq s f m - aSeq s f (m + 1)) * weights s f (k - m) := by
      rw [neg_div, Finset.sum_div, ← Finset.sum_neg_distrib]
      refine sum_congr rfl fun m _ => ?_
      rw [← hdiff m]; ring
    rw [hw, ← ih, ← Finset.sum_add_distrib]
    refine sum_congr rfl fun m hm => ?_
    simp only [mem_range] at hm
    rw [show k + 1 - (m + 1) = k - m by omega]
    ring

/-- **Claim STL-REC-1.** The power-series identity `(1 - x) a(x) r(x) = 1`,
that is `r(x) = 1 / ((1 - x) a(x))` (`LEAP_CERTIFICATE.md:216-232`). The
identity holds for the full series, so every truncation holds too. -/
theorem powerSeries_identity (hd : s 0 - f ≠ 0) :
    (1 - PowerSeries.X) * (PowerSeries.mk (aSeq s f) * PowerSeries.mk (weights s f)) = 1 := by
  have hprod : PowerSeries.mk (aSeq s f) * PowerSeries.mk (weights s f) = PowerSeries.mk 1 := by
    ext k
    rw [PowerSeries.coeff_mul, PowerSeries.coeff_mk,
      Finset.Nat.sum_antidiagonal_eq_sum_range_succ
        (fun i j => PowerSeries.coeff i (PowerSeries.mk (aSeq s f)) *
          PowerSeries.coeff j (PowerSeries.mk (weights s f)))]
    simp only [PowerSeries.coeff_mk, Pi.one_apply]
    exact conv_a_weights s f hd k
  rw [hprod, mul_comm]
  exact PowerSeries.mk_one_mul_one_sub_eq_one ℝ

/-- **Claim STL-REC-1.** With the upper-triangular Toeplitz
`A[i, j] = a[j - i]` (`toeplitz n a 0`), `z[j] = r[n-1-j]` solves `A z = 1`,
so `r[k] = z[n-1-k]`. -/
theorem upperToeplitz_mulVec_rev_weights (hd : s 0 - f ≠ 0) (i : Fin n) :
    (toeplitz n (aSeq s f) 0 *ᵥ fun j => weights s f (n - 1 - j.val)) i = 1 := by
  rw [mulVec_toeplitz, zero_mul, zero_add, sum_filter_ge_reindex (aSeq s f)]
  rw [show n - i.val = (n - 1 - i.val) + 1 by omega, ← conv_a_weights s f hd (n - 1 - i.val)]
  refine sum_congr rfl fun m hm => ?_
  simp only [mem_range] at hm
  rw [extN_of_lt _ (by omega)]
  dsimp only
  congr 2; omega

/-- **Claim COMPACT-RUNG2-1, cost doubt.** The recurrence costs
`∑_{k=1}^{59} k = 1,770` multiply-adds. The matvec of `_M_dot` costs
`∑_{i=0}^{59} (60 - i) = 1,830`, not the 1,770 that the docstring states. -/
theorem cost_counts : ∑ k ∈ range 60, k = 1770 ∧ ∑ i ∈ range 60, (60 - i) = 1830 := by
  constructor <;> decide

end Recurrence

/-! ### The equalizer on the DTH stage -/

/-- **Claims CPP-DOC-REC-1, DTH-REC-1, STL-REC-1, CANONICAL-MG-6,
COMPACT-RUNG2-1.** When `δ = S_1 - F ≠ 0` and every weight is nonnegative,
the normalized pair `p = r / W`, `q = reverse r / W` is an exact equilibrium
of the DTH stage (gap `0`) and `V x = F + δ / W`, for either sign of `δ`. -/
theorem V_eq_equalizer (x : State) (hd : succPay V x (0 + 1) - failPay V x ≠ 0)
    (hr : ∀ k < 60, 0 ≤ Toeplitz.weights (fun k => succPay V x (k + 1)) (failPay V x) k) :
    V x = failPay V x + (succPay V x (0 + 1) - failPay V x) /
      Toeplitz.weightSum 60 (fun k => succPay V x (k + 1)) (failPay V x) := by
  rw [V_bellman, stage_eq_toeplitz]
  exact (Toeplitz.equalizer_value (n := 60) _ _ hd hr).1

/-- Non-vacuity of the nonnegativity hypothesis: constant success classes
`s = 1`, `F = 0` give `r = (1, 0, …, 0)`, `W = 1`, and value `1`. -/
example : MatrixGame.value (Toeplitz.toeplitz 60 (fun _ => 1) 0) = 1 := by
  have hw : ∀ k, Toeplitz.weights (fun _ => (1 : ℝ)) 0 (k + 1) = 0 := by
    intro k; rw [weights_succ]; simp
  have hr : ∀ k < 60, 0 ≤ Toeplitz.weights (fun _ => (1 : ℝ)) 0 k := by
    intro k _; rcases k with _ | k
    · rw [Toeplitz.weights_zero]; norm_num
    · rw [hw]
  have hW : Toeplitz.weightSum 60 (fun _ => (1 : ℝ)) 0 = 1 := by
    rw [Toeplitz.weightSum, Finset.sum_range_succ', Toeplitz.weights_zero]
    simp [hw]
  rw [(Toeplitz.equalizer_value (n := 60) _ _ (by norm_num) hr).1, hW]
  norm_num

end Formal.DTH.Stage
