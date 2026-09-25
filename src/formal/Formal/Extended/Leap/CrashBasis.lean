import Mathlib
import Formal.MatrixGame.Basic
import Formal.MatrixGame.Transform
import Formal.Toeplitz.Basic
import Formal.Toeplitz.Equalizer

/-!
# The recurrence crash basis of the packing LP

`src/crates/docs/LEAP_CERTIFICATE.md` ("Packing fallback", "Recurrence crash
basis") and `Solver::crash` in `src/crates/stl_solver/src/leap_packing.rs`
start the packing simplex from the basis in which every packing variable is
basic. This file proves the exact-arithmetic mathematics of that start.

Conventions. Indices are `0`-based, as in the Rust code. The stage matrix is
`Formal.Toeplitz.toeplitz n s f` (`M[d, c] = s[c - d]` for `d ≤ c`, else `f`;
the row player is the Dropper and maximizes). With `d = f - s[0]`, the packing
coefficients are `a[k] = (f - s[k]) / d` (`Solver::coefficients`), and
`A[i, j] = a[j - i]` for `j ≥ i`, so `M = f 11ᵀ - d A`. The packing LP is
`max 1ᵀ y` subject to `Aᵀ y + w = 1`, `y, w ≥ 0`; variables are indexed by
`Fin n ⊕ Fin n`, with `Sum.inl i` the packing variable `y[i]` (cost `1`) and
`Sum.inr j` the slack `w[j]` (cost `0`). The reduced cost of a variable is
`c_j - c_Bᵀ B⁻¹ G_j`, the value `Solver::objective` writes into row `N` of the
tableau; the problem maximizes, so optimality needs every reduced cost `≤ 0`
(`optimize`'s `dual` test).

Main results:

* `ltToeplitz_mul`, `ltToeplitz_mul_crash`, `inv_ltToeplitz`: lower-triangular
  Toeplitz matrices multiply as truncated power series, and the inverse of
  `Aᵀ` has first column `c`, `c[0] = 1`, `c[k] = -∑_{m=1}^{k} a[m] c[k-m]`
  (CRATES-CRASH-1).
* `crash_basicValues`, `crash_reducedCost_slack`: the basic values are the
  prefix sums of `c`, and the reduced cost of slack `j` is
  `-(c[0] + ... + c[n-1-j])` (CRATES-CRASH-2).
* `kernelWeights_eq_prefixSum`, `weights_eq_prefixSum`: the kernel weights
  `r` of `sweep_key_rs` are the same prefix sums (CRATES-CRASH-3).
* `crash_residue_infeasible`, `crash_primal_feasible_iff`,
  `crash_dual_feasible_iff`, `card_basic_packing_eq_card_nonbasic_slack`
  (STL-LP-2).
* `reducedCost_zeroPositive`, `objective_le_of_dual_feasible`,
  `abs_accepted_sub_le`, `abs_rounded_accepted_sub_le` (CRATES-CRASH-5).

All statements are over `ℝ` in exact arithmetic unless a theorem states an
explicit rounding model.
-/

open Finset Matrix PowerSeries

set_option linter.unusedSectionVars false

namespace Formal.Leap

/-! ## Reindexing helpers -/

/-- A sum over `Fin n` of a function supported in `[lo, hi]` is the sum over
`Icc lo hi`. -/
theorem sum_fin_eq_sum_Icc {n : ℕ} (F : ℕ → ℝ) {lo hi : ℕ} (hhi : hi < n)
    (hF : ∀ l, l < n → ¬ (lo ≤ l ∧ l ≤ hi) → F l = 0) :
    ∑ l : Fin n, F l.val = ∑ l ∈ Icc lo hi, F l := by
  rw [Fin.sum_univ_eq_sum_range F n]
  symm
  apply Finset.sum_subset
  · intro l hl
    simp only [mem_Icc] at hl
    simp only [mem_range]
    omega
  · intro l hl hl'
    simp only [mem_range] at hl
    simp only [mem_Icc] at hl'
    exact hF l hl hl'

theorem sum_Icc_shift (F : ℕ → ℝ) {lo hi : ℕ} (h : lo ≤ hi) :
    ∑ l ∈ Icc lo hi, F l = ∑ m ∈ range (hi - lo + 1), F (lo + m) := by
  refine Finset.sum_nbij' (fun l => l - lo) (fun m => lo + m) ?_ ?_ ?_ ?_ ?_ <;>
  · intros
    simp only [mem_Icc, mem_range] at *
    first | omega | (congr 1; omega)

theorem sum_Icc_reflect (F : ℕ → ℝ) {lo hi : ℕ} (h : lo ≤ hi) :
    ∑ l ∈ Icc lo hi, F l = ∑ m ∈ range (hi - lo + 1), F (hi - m) := by
  refine Finset.sum_nbij' (fun l => hi - l) (fun m => hi - m) ?_ ?_ ?_ ?_ ?_ <;>
  · intros
    simp only [mem_Icc, mem_range] at *
    first | omega | (congr 1; omega)

/-! ## Lower-triangular Toeplitz matrices and truncated power series -/

/-- The lower-triangular Toeplitz matrix with first column `a`:
`T[i, j] = a[i - j]` for `j ≤ i`, else `0`. This is `Solver::coefficient` for a
packing variable (`row ≥ variable ↦ a[row - variable]`). -/
def ltToeplitz (n : ℕ) (a : ℕ → ℝ) : Matrix (Fin n) (Fin n) ℝ :=
  fun i j => if j ≤ i then a (i.val - j.val) else 0

theorem ltToeplitz_apply (n : ℕ) (a : ℕ → ℝ) (i j : Fin n) :
    ltToeplitz n a i j = if j ≤ i then a (i.val - j.val) else 0 := rfl

theorem coeff_mk_mul_mk (a b : ℕ → ℝ) (k : ℕ) :
    coeff k (mk a * mk b) = ∑ m ∈ range (k + 1), a m * b (k - m) := by
  rw [coeff_mul, Finset.Nat.sum_antidiagonal_eq_sum_range_succ
    (fun i j => coeff i (mk a) * coeff j (mk b))]
  simp only [coeff_mk]

/-- **Lower-triangular Toeplitz matrices multiply as truncated power series**
(CRATES-CRASH-1; `LEAP_CERTIFICATE.md:220-221`): the product has first column
the first `n` coefficients of `(∑ a_k X^k)(∑ b_k X^k)`. -/
theorem ltToeplitz_mul (n : ℕ) (a b : ℕ → ℝ) :
    ltToeplitz n a * ltToeplitz n b = ltToeplitz n (fun k => coeff k (mk a * mk b)) := by
  ext i j
  simp only [mul_apply, ltToeplitz_apply, coeff_mk_mul_mk, Fin.le_iff_val_le_val]
  set F : ℕ → ℝ := fun l =>
    (if l ≤ i.val then a (i.val - l) else 0) * (if j.val ≤ l then b (l - j.val) else 0) with hFdef
  have hsum : ∑ l : Fin n, (if l.val ≤ i.val then a (i.val - l.val) else 0) *
      (if j.val ≤ l.val then b (l.val - j.val) else 0) = ∑ l : Fin n, F l.val := rfl
  rw [hsum]
  by_cases hji : j.val ≤ i.val
  · simp only [hji, ↓reduceIte]
    rw [sum_fin_eq_sum_Icc F (lo := j.val) i.isLt, sum_Icc_reflect F hji]
    · refine Finset.sum_congr rfl fun m hm => ?_
      simp only [mem_range] at hm
      simp only [hFdef, show i.val - m ≤ i.val by omega, show j.val ≤ i.val - m by omega,
        ↓reduceIte, show i.val - (i.val - m) = m by omega,
        show i.val - m - j.val = i.val - j.val - m by omega]
    · intro l _ hl
      simp only [hFdef]
      by_cases h1 : l ≤ i.val
      · simp only [h1, ↓reduceIte, show ¬ j.val ≤ l by omega, mul_zero]
      · simp only [h1, ↓reduceIte, zero_mul]
  · simp only [hji, ↓reduceIte]
    refine Finset.sum_eq_zero fun l _ => ?_
    simp only [hFdef]
    by_cases h1 : l.val ≤ i.val
    · simp only [h1, ↓reduceIte, show ¬ j.val ≤ l.val by omega, mul_zero]
    · simp only [h1, ↓reduceIte, zero_mul]

/-- The first column `c` of `(Aᵀ)⁻¹`, computed as in `Solver::crash`
(`leap_packing.rs:114-123`): `c[0] = 1` and `c[k] = -∑_{m=1}^{k} a[m] c[k-m]`.
Here the index `m : Fin (k + 1)` stands for the code's `m + 1`. -/
noncomputable def crashSeries (a : ℕ → ℝ) : ℕ → ℝ
  | 0 => 1
  | k + 1 => -∑ m : Fin (k + 1), a (m.val + 1) * crashSeries a (k - m.val)
termination_by k => k
decreasing_by omega

theorem crashSeries_zero (a : ℕ → ℝ) : crashSeries a 0 = 1 := by
  rw [crashSeries]

theorem crashSeries_succ (a : ℕ → ℝ) (k : ℕ) :
    crashSeries a (k + 1) = -∑ m ∈ range (k + 1), a (m + 1) * crashSeries a (k - m) := by
  rw [crashSeries, ← Fin.sum_univ_eq_sum_range (fun m => a (m + 1) * crashSeries a (k - m))]

/-- The recurrence for `c` makes `(∑ a_k X^k)(∑ c_k X^k) = 1` when `a[0] = 1`. -/
theorem mk_mul_mk_crashSeries (a : ℕ → ℝ) (ha : a 0 = 1) :
    mk a * mk (crashSeries a) = 1 := by
  ext k
  rw [coeff_mk_mul_mk, coeff_one]
  cases k with
  | zero => simp [ha, crashSeries_zero]
  | succ k =>
    rw [Finset.sum_range_succ', Nat.sub_zero, crashSeries_succ, ha]
    simp only [Nat.succ_ne_zero, ↓reduceIte, one_mul]
    rw [show ∑ m ∈ range (k + 1), a (m + 1) * crashSeries a (k + 1 - (m + 1)) =
        ∑ m ∈ range (k + 1), a (m + 1) * crashSeries a (k - m) from
      Finset.sum_congr rfl fun m _ => by rw [show k + 1 - (m + 1) = k - m by omega]]
    ring

/-- `c` is the coefficient sequence of the power-series inverse of `a`. -/
theorem mk_crashSeries_eq_inv (a : ℕ → ℝ) (ha : a 0 = 1) :
    mk (crashSeries a) = (mk a)⁻¹ := by
  rw [PowerSeries.eq_inv_iff_mul_eq_one (by rw [constantCoeff_mk, ha]; exact one_ne_zero),
    mul_comm]
  exact mk_mul_mk_crashSeries a ha

theorem ltToeplitz_coeff_one (n : ℕ) :
    ltToeplitz n (fun k => coeff k (1 : PowerSeries ℝ)) = 1 := by
  ext i j
  simp only [ltToeplitz_apply, coeff_one, one_apply, Fin.le_iff_val_le_val, Fin.ext_iff]
  by_cases h : j.val ≤ i.val
  · simp only [h, ↓reduceIte]
    by_cases h' : i.val = j.val
    · simp [h']
    · simp only [show i.val - j.val ≠ 0 by omega, ↓reduceIte, h']
  · simp only [h, ↓reduceIte, show i.val ≠ j.val by omega]

/-- **Inverse of `Aᵀ`** (CRATES-CRASH-1, STL-LP-2; `LEAP_CERTIFICATE.md:218-224`,
`leap_packing.rs:107-123`): for any first column `a` with `a[0] = 1`, the
lower-triangular Toeplitz matrix with first column `c = crashSeries a` is a
two-sided inverse of the one with first column `a`. -/
theorem ltToeplitz_mul_crash (n : ℕ) (a : ℕ → ℝ) (ha : a 0 = 1) :
    ltToeplitz n a * ltToeplitz n (crashSeries a) = 1 ∧
      ltToeplitz n (crashSeries a) * ltToeplitz n a = 1 := by
  refine ⟨?_, ?_⟩
  · rw [ltToeplitz_mul, mk_mul_mk_crashSeries a ha, ltToeplitz_coeff_one]
  · rw [ltToeplitz_mul, mul_comm, mk_mul_mk_crashSeries a ha, ltToeplitz_coeff_one]

/-- The matrix inverse of the lower-triangular Toeplitz matrix with first
column `a` (`a[0] = 1`) is the one with first column `crashSeries a`
(CRATES-CRASH-1; `LEAP_CERTIFICATE.md:218-224`). -/
theorem inv_ltToeplitz (n : ℕ) (a : ℕ → ℝ) (ha : a 0 = 1) :
    (ltToeplitz n a)⁻¹ = ltToeplitz n (crashSeries a) :=
  Matrix.inv_eq_right_inv (ltToeplitz_mul_crash n a ha).1

/-! ## The packing matrix of a stage -/

/-- The packing coefficients `a[k] = (f - s[k]) / (f - s[0])`
(`Solver::coefficients`, `leap_packing.rs`; `LEAP_CERTIFICATE.md:219`). -/
noncomputable def packCoeff (s : ℕ → ℝ) (f : ℝ) (k : ℕ) : ℝ := (f - s k) / (f - s 0)

theorem packCoeff_zero (s : ℕ → ℝ) (f : ℝ) (hd : f - s 0 ≠ 0) : packCoeff s f 0 = 1 :=
  div_self hd

/-- The upper-triangular Toeplitz matrix `A[i, j] = (f - s[j-i]) / d` for
`j ≥ i`, else `0` (`LEAP_CERTIFICATE.md`, "Packing fallback"). -/
noncomputable def packA (n : ℕ) (s : ℕ → ℝ) (f : ℝ) : Matrix (Fin n) (Fin n) ℝ :=
  fun i j => if i ≤ j then packCoeff s f (j.val - i.val) else 0

/-- `Aᵀ` is lower-triangular Toeplitz with first column `a`. -/
theorem packA_transpose (n : ℕ) (s : ℕ → ℝ) (f : ℝ) :
    (packA n s f)ᵀ = ltToeplitz n (packCoeff s f) := rfl

/-- The stage matrix is `M = f 11ᵀ - d A` with `d = f - s[0] ≠ 0`
(`LEAP_CERTIFICATE.md`, "Packing fallback"). -/
theorem toeplitz_eq_packA (n : ℕ) (s : ℕ → ℝ) (f : ℝ) (hd : f - s 0 ≠ 0) (i j : Fin n) :
    Toeplitz.toeplitz n s f i j = f - (f - s 0) * packA n s f i j := by
  simp only [Toeplitz.toeplitz_apply, packA, packCoeff]
  split_ifs
  · field_simp; ring
  · ring

/-! ## Prefix sums, basic values, and reduced costs -/

/-- The prefix sum `c[0] + ... + c[k]`. -/
def prefixSum (c : ℕ → ℝ) (k : ℕ) : ℝ := ∑ i ∈ range (k + 1), c i

/-- Row sums of a lower-triangular Toeplitz matrix are prefix sums of its first
column. -/
theorem ltToeplitz_mulVec_one (n : ℕ) (c : ℕ → ℝ) (i : Fin n) :
    (ltToeplitz n c *ᵥ fun _ => 1) i = prefixSum c i.val := by
  simp only [mulVec, dotProduct, mul_one, ltToeplitz_apply, Fin.le_iff_val_le_val]
  rw [sum_fin_eq_sum_Icc (fun l => if l ≤ i.val then c (i.val - l) else 0) (lo := 0) i.isLt,
    sum_Icc_reflect _ (Nat.zero_le _), prefixSum, Nat.sub_zero]
  · refine Finset.sum_congr rfl fun m hm => ?_
    simp only [mem_range] at hm
    simp only [show i.val - m ≤ i.val by omega, ↓reduceIte, show i.val - (i.val - m) = m by omega]
  · intro l _ hl
    simp only [show ¬ l ≤ i.val by omega, ↓reduceIte]

/-- Column sums of a lower-triangular Toeplitz matrix are prefix sums of its
first column: column `j` sums to `c[0] + ... + c[n-1-j]`. -/
theorem ltToeplitz_colSum (n : ℕ) (c : ℕ → ℝ) (j : Fin n) :
    ∑ i, ltToeplitz n c i j = prefixSum c (n - 1 - j.val) := by
  simp only [ltToeplitz_apply, Fin.le_iff_val_le_val]
  rw [sum_fin_eq_sum_Icc (fun l => if j.val ≤ l then c (l - j.val) else 0) (lo := j.val)
      (hi := n - 1) (by have := j.isLt; omega),
    sum_Icc_shift _ (by have := j.isLt; omega), prefixSum]
  · refine Finset.sum_congr rfl fun m _ => ?_
    simp only [show j.val ≤ j.val + m by omega, ↓reduceIte, show j.val + m - j.val = m by omega]
  · intro l hl hl'
    simp only [show ¬ j.val ≤ l by omega, ↓reduceIte]

/-! ### A basis of a standard-form LP -/

/-- The basis matrix `B = G[:, β]` of the basis `β : Fin m → ι`. -/
def basisMatrix {m : ℕ} {ι : Type*} (G : Matrix (Fin m) ι ℝ) (β : Fin m → ι) :
    Matrix (Fin m) (Fin m) ℝ :=
  G.submatrix id β

/-- The basic values `B⁻¹ b`. They do not depend on the cost vector. -/
noncomputable def basicValues {m : ℕ} {ι : Type*} (G : Matrix (Fin m) ι ℝ) (β : Fin m → ι)
    (b : Fin m → ℝ) : Fin m → ℝ :=
  (basisMatrix G β)⁻¹ *ᵥ b

/-- The reduced cost `c_j - c_Bᵀ B⁻¹ G_j` (`Solver::objective`,
`leap_packing.rs:94-106`: one minus the basic packing rows for a packing
column, zero minus them for a slack column). -/
noncomputable def reducedCost {m : ℕ} {ι : Type*} (G : Matrix (Fin m) ι ℝ) (β : Fin m → ι)
    (c : ι → ℝ) (j : ι) : ℝ :=
  c j - (c ∘ β) ⬝ᵥ ((basisMatrix G β)⁻¹ *ᵥ fun i => G i j)

/-- **Sign convention: maximization** (settles the STL-LP-2 doubt). If every
reduced cost is `≤ 0`, then every feasible point `x ≥ 0`, `G x = b` has
objective at most `c_Bᵀ B⁻¹ b`, the objective of the basic solution. So a
basis that is primal feasible and has all reduced costs `≤ 0` is optimal for
`max cᵀ x`; this is the exit test `primal && dual` of `optimize`
(`leap_packing.rs`). -/
theorem objective_le_of_dual_feasible {m : ℕ} {ι : Type*} [Fintype ι]
    (G : Matrix (Fin m) ι ℝ) (β : Fin m → ι) (c : ι → ℝ) (b : Fin m → ℝ)
    (hdual : ∀ j, reducedCost G β c j ≤ 0) {x : ι → ℝ} (hx : ∀ j, 0 ≤ x j)
    (hGx : G *ᵥ x = b) :
    c ⬝ᵥ x ≤ (c ∘ β) ⬝ᵥ basicValues G β b := by
  set π := (c ∘ β) ᵥ* (basisMatrix G β)⁻¹ with hπ
  have hrc : ∀ j, c j = reducedCost G β c j + (π ᵥ* G) j := by
    intro j
    simp only [reducedCost, hπ, dotProduct_mulVec]
    simp only [vecMul, dotProduct]
    ring
  have hsplit : c ⬝ᵥ x = (fun j => reducedCost G β c j) ⬝ᵥ x + π ⬝ᵥ b := by
    rw [← hGx, dotProduct_mulVec]
    simp only [dotProduct, ← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun j _ => ?_
    rw [hrc j]
    ring
  have hneg : (fun j => reducedCost G β c j) ⬝ᵥ x ≤ 0 :=
    Finset.sum_nonpos fun j _ => mul_nonpos_of_nonpos_of_nonneg (hdual j) (hx j)
  have hbasic : π ⬝ᵥ b = (c ∘ β) ⬝ᵥ basicValues G β b := by
    rw [hπ, basicValues, dotProduct_mulVec]
  linarith

/-- The perturbed costs of `solve_crash`: each nonbasic variable with a
positive reduced cost has that reduced cost subtracted from its cost; basic
costs are unchanged. -/
noncomputable def zeroPositive {m : ℕ} {ι : Type*} (G : Matrix (Fin m) ι ℝ) (β : Fin m → ι)
    (c : ι → ℝ) [DecidablePred (· ∈ Set.range β)] (j : ι) : ℝ :=
  if j ∈ Set.range β then c j else c j - max (reducedCost G β c j) 0

/-- **Zeroing positive reduced costs gives dual feasibility**
(CRATES-CRASH-5, STL-LP-2; `LEAP_CERTIFICATE.md:235-236`,
`leap_packing.rs:270-275`). The objective row that `solve_crash` obtains by
setting each positive entry to `0` is the reduced-cost row of the perturbed
costs `zeroPositive`: every nonbasic reduced cost becomes `min(rc, 0) ≤ 0`.
The basic values `B⁻¹ b` do not involve the costs, so primal feasibility
reached by the dual simplex survives the rebuild of the true objective row. -/
theorem reducedCost_zeroPositive {m : ℕ} {ι : Type*} (G : Matrix (Fin m) ι ℝ) (β : Fin m → ι)
    (c : ι → ℝ) [DecidablePred (· ∈ Set.range β)] {j : ι} (hj : j ∉ Set.range β) :
    reducedCost G β (zeroPositive G β c) j = min (reducedCost G β c j) 0 ∧
      reducedCost G β (zeroPositive G β c) j ≤ 0 := by
  have hβ : zeroPositive G β c ∘ β = c ∘ β := by
    funext i
    simp only [Function.comp_apply, zeroPositive, Set.mem_range_self, ↓reduceIte]
  have h : reducedCost G β (zeroPositive G β c) j = min (reducedCost G β c j) 0 := by
    have e : reducedCost G β (zeroPositive G β c) j =
        reducedCost G β c j - max (reducedCost G β c j) 0 := by
      simp only [reducedCost, hβ, zeroPositive, hj, ↓reduceIte]
      ring
    rw [e]
    rcases le_total (reducedCost G β c j) 0 with h0 | h0
    · rw [max_eq_right h0, min_eq_left h0, sub_zero]
    · rw [max_eq_left h0, min_eq_right h0, sub_self]
  exact ⟨h, h ▸ min_le_right _ _⟩

/-! ### The packing LP and its crash basis -/

/-- The packing constraint matrix `[Aᵀ | I]`: packing variables `Sum.inl i`,
slacks `Sum.inr j`. -/
noncomputable def packG (n : ℕ) (a : ℕ → ℝ) : Matrix (Fin n) (Fin n ⊕ Fin n) ℝ :=
  Matrix.fromCols (ltToeplitz n a) 1

/-- The packing costs: `1` per packing variable, `0` per slack. -/
def packCost (n : ℕ) : Fin n ⊕ Fin n → ℝ := Sum.elim (fun _ => 1) (fun _ => 0)

/-- The crash basis makes every packing variable basic: `basic[i] = i`
(`Solver::crash`). -/
def crashBasis (n : ℕ) : Fin n → Fin n ⊕ Fin n := Sum.inl

theorem basisMatrix_crash (n : ℕ) (a : ℕ → ℝ) :
    basisMatrix (packG n a) (crashBasis n) = ltToeplitz n a := by
  ext i j
  simp [basisMatrix, packG, crashBasis]

/-- **Crash basic values** (CRATES-CRASH-2, STL-LP-2; `LEAP_CERTIFICATE.md:227-228`,
`leap_packing.rs:107-140`): in the basis where every packing variable is basic,
`y = (Aᵀ)⁻¹ 1` and `y[i] = c[0] + ... + c[i]`. These are also the unique `y`
with `Aᵀ y = 1`. -/
theorem crash_basicValues (n : ℕ) (a : ℕ → ℝ) (ha : a 0 = 1) :
    basicValues (packG n a) (crashBasis n) (fun _ => 1) =
        (fun i => prefixSum (crashSeries a) i.val) ∧
      ∀ y : Fin n → ℝ, ltToeplitz n a *ᵥ y = (fun _ => 1) ↔
        y = fun i => prefixSum (crashSeries a) i.val := by
  have hv : basicValues (packG n a) (crashBasis n) (fun _ => 1) =
      fun i => prefixSum (crashSeries a) i.val := by
    funext i
    rw [basicValues, basisMatrix_crash, inv_ltToeplitz n a ha, ltToeplitz_mulVec_one]
  refine ⟨hv, fun y => ⟨fun h => ?_, fun h => ?_⟩⟩
  · funext i
    rw [← ltToeplitz_mulVec_one n (crashSeries a) i, ← h, mulVec_mulVec,
      (ltToeplitz_mul_crash n a ha).2, one_mulVec]
  · rw [h]
    funext i
    rw [show (fun i : Fin n => prefixSum (crashSeries a) i.val) =
        ltToeplitz n (crashSeries a) *ᵥ (fun _ => 1) from
      funext fun i => (ltToeplitz_mulVec_one n (crashSeries a) i).symm,
      mulVec_mulVec, (ltToeplitz_mul_crash n a ha).1, one_mulVec]

/-- **Crash reduced costs** (CRATES-CRASH-2, STL-LP-2; `LEAP_CERTIFICATE.md:228-229`,
`leap_packing.rs:107-113`): the reduced cost of slack `j` in the crash basis
is the negated sum of column `j` of `(Aᵀ)⁻¹`, which is
`-(c[0] + ... + c[n-1-j])`. -/
theorem crash_reducedCost_slack (n : ℕ) (a : ℕ → ℝ) (ha : a 0 = 1) (j : Fin n) :
    reducedCost (packG n a) (crashBasis n) (packCost n) (Sum.inr j) =
        -∑ i, (ltToeplitz n a)⁻¹ i j ∧
      reducedCost (packG n a) (crashBasis n) (packCost n) (Sum.inr j) =
        -prefixSum (crashSeries a) (n - 1 - j.val) := by
  have e : reducedCost (packG n a) (crashBasis n) (packCost n) (Sum.inr j) =
      -∑ i, (ltToeplitz n a)⁻¹ i j := by
    rw [reducedCost, basisMatrix_crash]
    have hc : packCost n ∘ crashBasis n = fun _ => 1 := rfl
    have hcol : (fun i => packG n a i (Sum.inr j)) = fun i => (1 : Matrix (Fin n) (Fin n) ℝ) i j := by
      funext i; simp [packG]
    rw [hc, hcol]
    simp only [packCost, Sum.elim_inr, zero_sub, dotProduct, one_mul, mulVec, one_apply,
      mul_ite, mul_one, mul_zero, Finset.sum_ite_eq', mem_univ, ↓reduceIte]
  refine ⟨e, ?_⟩
  rw [e, inv_ltToeplitz n a ha, ltToeplitz_colSum]

/-- At the code's size `N = 60` (`leap_packing.rs`), the crash basic values are
`y[i] = c[0] + ... + c[i]` and slack `j` has reduced cost
`-(c[0] + ... + c[59-j])` (CRATES-CRASH-2, STL-LP-2;
`LEAP_CERTIFICATE.md:227-229`). -/
theorem crash_sixty (a : ℕ → ℝ) (ha : a 0 = 1) (i j : Fin 60) :
    basicValues (packG 60 a) (crashBasis 60) (fun _ => 1) i = prefixSum (crashSeries a) i.val ∧
      reducedCost (packG 60 a) (crashBasis 60) (packCost 60) (Sum.inr j) =
        -prefixSum (crashSeries a) (59 - j.val) :=
  ⟨congrFun (crash_basicValues 60 a ha).1 i, (crash_reducedCost_slack 60 a ha j).2⟩

/-- In the crash basis, every basic packing variable has reduced cost `0`. -/
theorem crash_reducedCost_basic (n : ℕ) (a : ℕ → ℝ) (ha : a 0 = 1) (i : Fin n) :
    reducedCost (packG n a) (crashBasis n) (packCost n) (Sum.inl i) = 0 := by
  rw [reducedCost, basisMatrix_crash]
  have hc : packCost n ∘ crashBasis n = fun _ => 1 := rfl
  have hcol : (fun k => packG n a k (Sum.inl i)) = fun k => ltToeplitz n a k i := by
    funext k; simp [packG]
  rw [hc, hcol]
  have hcolv : (fun k => ltToeplitz n a k i) = ltToeplitz n a *ᵥ Pi.single i 1 := by
    funext k
    simp [mulVec, dotProduct, Pi.single_apply]
  rw [hcolv, mulVec_mulVec, inv_ltToeplitz n a ha, (ltToeplitz_mul_crash n a ha).2, one_mulVec]
  simp [packCost, dotProduct, Pi.single_apply]

/-! ## Kernel weights are the same prefix sums -/

/-- The kernel coefficients `b[m] = (s[m-1] - s[m]) / (s[0] - f)` of
`sweep_key_rs` (`leap.rs:392-397`, the scratch array `q[k]`). At `m = 0`
natural subtraction gives `b[0] = 0`; the kernel never reads `b[0]`. -/
noncomputable def kernelCoeff (s : ℕ → ℝ) (f : ℝ) (m : ℕ) : ℝ := (s (m - 1) - s m) / (s 0 - f)

/-- The kernel weights of `sweep_key_rs` (`leap.rs:399-409`): `r[0] = 1` and
`r[k] = ∑_{j<k} b[k-j] r[j]`. -/
noncomputable def kernelWeights (b : ℕ → ℝ) : ℕ → ℝ
  | 0 => 1
  | k + 1 => ∑ j : Fin (k + 1), b (k + 1 - j.val) * kernelWeights b j.val
termination_by k => k
decreasing_by exact j.isLt

theorem kernelWeights_zero (b : ℕ → ℝ) : kernelWeights b 0 = 1 := by
  rw [kernelWeights]

theorem kernelWeights_succ (b : ℕ → ℝ) (k : ℕ) :
    kernelWeights b (k + 1) = ∑ j ∈ range (k + 1), b (k + 1 - j) * kernelWeights b j := by
  rw [kernelWeights, ← Fin.sum_univ_eq_sum_range (fun j => b (k + 1 - j) * kernelWeights b j)]

/-- `b[m] = a[m-1] - a[m]` (CRATES-CRASH-3; `LEAP_CERTIFICATE.md:229-230`). It
holds for every `m` with `ℕ` subtraction, and needs no hypothesis on `d`
because `(-x) / (-y) = x / y` also holds at `y = 0` in Lean. -/
theorem kernelCoeff_eq_sub (s : ℕ → ℝ) (f : ℝ) (m : ℕ) :
    kernelCoeff s f m = packCoeff s f (m - 1) - packCoeff s f m := by
  rw [kernelCoeff, packCoeff, packCoeff, div_sub_div_same,
    show f - s (m - 1) - (f - s m) = -(s (m - 1) - s m) by ring,
    show f - s 0 = -(s 0 - f) by ring, neg_div_neg_eq]

theorem mk_kernelWeights (b : ℕ → ℝ) (hb0 : b 0 = 0) :
    mk (kernelWeights b) = 1 + mk b * mk (kernelWeights b) := by
  ext k
  rw [map_add, coeff_one, coeff_mk_mul_mk, coeff_mk]
  cases k with
  | zero => simp [kernelWeights_zero, hb0]
  | succ k =>
    rw [kernelWeights_succ, Finset.sum_range_succ' (fun m => b m * kernelWeights b (k + 1 - m)),
      hb0]
    simp only [Nat.succ_ne_zero, ↓reduceIte, zero_mul, add_zero, zero_add]
    rw [← Finset.sum_range_reflect (fun j => b (k + 1 - j) * kernelWeights b j) (k + 1)]
    refine Finset.sum_congr rfl fun m hm => ?_
    simp only [mem_range] at hm
    rw [show k + 1 - (k + 1 - 1 - m) = m + 1 by omega, show k + 1 - 1 - m = k + 1 - (m + 1) by omega]

/-- **Kernel weights are prefix sums of `c`** (generating-function form of
CRATES-CRASH-3): if `a[0] = 1` and `b[m] = a[m-1] - a[m]` for every `m` (so
`b[0] = 0`), then `r[k] = c[0] + ... + c[k]` with `c = crashSeries a`. The
proof is `1 - B(x) = (1 - x) A(x)`, so `(1 - x) R(x) = C(x)`. -/
theorem kernelWeights_eq_prefixSum (a b : ℕ → ℝ) (ha : a 0 = 1)
    (hb : ∀ m, b m = a (m - 1) - a m) (k : ℕ) :
    kernelWeights b k = prefixSum (crashSeries a) k := by
  have hb0 : b 0 = 0 := by rw [hb 0]; ring
  set R := mk (kernelWeights b) with hRdef
  set C := mk (crashSeries a)
  have h1 : (1 : PowerSeries ℝ) - PowerSeries.mk b = (1 - X) * PowerSeries.mk a := by
    ext k
    rw [map_sub, sub_mul, one_mul, map_sub, coeff_one, coeff_mk, coeff_mk]
    cases k with
    | zero => simp [hb0, ha]
    | succ k => rw [coeff_succ_X_mul, coeff_mk, hb]; simp
  have hR : R = 1 + mk b * R := mk_kernelWeights b hb0
  have h2 : (1 - X) * PowerSeries.mk a * R = 1 := by
    rw [← h1]; linear_combination hR
  have hAC : PowerSeries.mk a * C = 1 := mk_mul_mk_crashSeries a ha
  have h3 : (1 - X) * R = C := by
    linear_combination C * h2 - (1 - X) * R * hAC
  have hc0 : kernelWeights b 0 = crashSeries a 0 := by
    have := congrArg (coeff 0) h3
    rwa [sub_mul, one_mul, map_sub, coeff_zero_X_mul, sub_zero, hRdef, coeff_mk, coeff_mk] at this
  have hcs : ∀ k, kernelWeights b (k + 1) - kernelWeights b k = crashSeries a (k + 1) := by
    intro k
    have := congrArg (coeff (k + 1)) h3
    rwa [sub_mul, one_mul, map_sub, coeff_succ_X_mul, hRdef, coeff_mk, coeff_mk, coeff_mk] at this
  induction k with
  | zero => rw [hc0, prefixSum, Finset.sum_range_one]
  | succ k ih =>
    rw [prefixSum, Finset.sum_range_succ, ← prefixSum, ← ih]
    linarith [hcs k]

/-- The kernel weights of `sweep_key_rs` are the equalizer weights
`Formal.Toeplitz.weights` of `try_rung2` (`src/dth_compact/main.py`), written
with `b[m]` in place of `-dS[m-1] / d0`. -/
theorem kernelWeights_kernelCoeff (s : ℕ → ℝ) (f : ℝ) (k : ℕ) :
    kernelWeights (kernelCoeff s f) k = Toeplitz.weights s f k := by
  induction k using Nat.strong_induction_on with
  | _ k ih =>
    cases k with
    | zero => rw [kernelWeights_zero, Toeplitz.weights_zero]
    | succ k =>
      rw [kernelWeights_succ, Toeplitz.weights,
        Fin.sum_univ_eq_sum_range (fun m => (s (m + 1) - s m) * Toeplitz.weights s f (k - m))]
      rw [Finset.sum_congr rfl fun j hj => by
        rw [ih j (by simp only [mem_range] at hj; omega)]]
      rw [← Finset.sum_range_reflect
        (fun j => kernelCoeff s f (k + 1 - j) * Toeplitz.weights s f j) (k + 1),
        neg_div, Finset.sum_div, ← Finset.sum_neg_distrib]
      refine Finset.sum_congr rfl fun m hm => ?_
      simp only [mem_range] at hm
      rw [show k + 1 - (k + 1 - 1 - m) = m + 1 by omega, show k + 1 - 1 - m = k - m by omega,
        kernelCoeff, show m + 1 - 1 = m by omega]
      ring

/-- **Kernel weights equal crash basic values** (CRATES-CRASH-3, STL-LP-2;
`LEAP_CERTIFICATE.md:229-232`). For a stage with `d = f - s[0] ≠ 0`, the
kernel weights `r` of `sweep_key_rs` (`leap.rs:392-413`) and the equalizer
weights of `try_rung2` are the prefix sums of the crash series
`c = crashSeries a`, `a = packCoeff s f`; so the crash basic values are
`y[i] = r[i]`. This holds in exact arithmetic; `Solver::crash` and
`sweep_key_rs` round `c` and `r` along different paths, so their binary64
outputs can differ by rounding. The code requires `d > 1e-12`; the identity
needs only `d ≠ 0`. -/
theorem weights_eq_prefixSum (s : ℕ → ℝ) (f : ℝ) (hd : f - s 0 ≠ 0) (k : ℕ) :
    kernelWeights (kernelCoeff s f) k = prefixSum (crashSeries (packCoeff s f)) k ∧
      Toeplitz.weights s f k = prefixSum (crashSeries (packCoeff s f)) k := by
  have h := kernelWeights_eq_prefixSum (packCoeff s f) (kernelCoeff s f) (packCoeff_zero s f hd)
    (kernelCoeff_eq_sub s f) k
  exact ⟨h, (kernelWeights_kernelCoeff s f k).symm.trans h⟩

/-- The crash basic values are the kernel weights: `y[i] = r[i]`
(CRATES-CRASH-3; `LEAP_CERTIFICATE.md:229-232`). -/
theorem crash_basicValues_eq_kernelWeights (n : ℕ) (s : ℕ → ℝ) (f : ℝ) (hd : f - s 0 ≠ 0)
    (i : Fin n) :
    basicValues (packG n (packCoeff s f)) (crashBasis n) (fun _ => 1) i =
      kernelWeights (kernelCoeff s f) i.val := by
  rw [(crash_basicValues n _ (packCoeff_zero s f hd)).1, (weights_eq_prefixSum s f hd _).1]

/-! ## Feasibility of the crash basis -/

/-- The crash basis is primal feasible exactly when every kernel weight
`r[0..n-1]` is nonnegative (STL-LP-2; `LEAP_CERTIFICATE.md:227-235`). -/
theorem crash_primal_feasible_iff (n : ℕ) (s : ℕ → ℝ) (f : ℝ) (hd : f - s 0 ≠ 0) :
    (∀ i, 0 ≤ basicValues (packG n (packCoeff s f)) (crashBasis n) (fun _ => 1) i) ↔
      ∀ k < n, 0 ≤ kernelWeights (kernelCoeff s f) k := by
  simp only [crash_basicValues_eq_kernelWeights n s f hd]
  exact ⟨fun h k hk => h ⟨k, hk⟩, fun h i => h i.val i.isLt⟩

/-- The crash basis is dual feasible (every reduced cost `≤ 0`, the
maximization test of `optimize`) exactly when every kernel weight
`r[0..n-1]` is nonnegative (STL-LP-2; `LEAP_CERTIFICATE.md:227-235`,
`leap_packing.rs:166`). Slack `j` has reduced cost
`-r[n-1-j]`; basic packing variables have reduced cost `0`. -/
theorem crash_dual_feasible_iff (n : ℕ) (s : ℕ → ℝ) (f : ℝ) (hd : f - s 0 ≠ 0) :
    (∀ v, reducedCost (packG n (packCoeff s f)) (crashBasis n) (packCost n) v ≤ 0) ↔
      ∀ k < n, 0 ≤ kernelWeights (kernelCoeff s f) k := by
  have ha := packCoeff_zero s f hd
  have hslack : ∀ j : Fin n, reducedCost (packG n (packCoeff s f)) (crashBasis n) (packCost n)
      (Sum.inr j) = -kernelWeights (kernelCoeff s f) (n - 1 - j.val) := by
    intro j
    rw [(crash_reducedCost_slack n _ ha j).2, (weights_eq_prefixSum s f hd _).1]
  constructor
  · intro h k hk
    have := h (Sum.inr ⟨n - 1 - k, by omega⟩)
    rw [hslack] at this
    simp only at this
    rw [show n - 1 - (n - 1 - k) = k by omega] at this
    linarith
  · intro h v
    cases v with
    | inl i => rw [crash_reducedCost_basic n _ ha i]
    | inr j =>
      rw [hslack]
      have := h (n - 1 - j.val) (by omega)
      linarith

/-- **A residue stage makes the crash start neither primal nor dual feasible**
(STL-LP-2; `LEAP_CERTIFICATE.md:234-235`). If some kernel weight `r[k]`,
`k < n`, is negative, then the basic value `y[k] = r[k]` is negative and
slack `n-1-k` has the positive reduced cost `-r[k]`. -/
theorem crash_residue_infeasible (n : ℕ) (s : ℕ → ℝ) (f : ℝ) (hd : f - s 0 ≠ 0) {k : ℕ}
    (hk : k < n) (hr : kernelWeights (kernelCoeff s f) k < 0) :
    basicValues (packG n (packCoeff s f)) (crashBasis n) (fun _ => 1) ⟨k, hk⟩ < 0 ∧
      0 < reducedCost (packG n (packCoeff s f)) (crashBasis n) (packCost n)
        (Sum.inr ⟨n - 1 - k, by omega⟩) := by
  refine ⟨by rw [crash_basicValues_eq_kernelWeights n s f hd]; exact hr, ?_⟩
  rw [(crash_reducedCost_slack n _ (packCoeff_zero s f hd) _).2,
    ← (weights_eq_prefixSum s f hd _).1]
  simp only
  rw [show n - 1 - (n - 1 - k) = k by omega]
  linarith

/-- **Seed sizes agree** (STL-LP-2; `LEAP_CERTIFICATE.md:261-263`,
`Solver::basis_support`): a basis of the packing LP has `n` basic variables
among the `n` packing variables and `n` slacks, so the number of basic
packing variables (Dropper support) equals the number of nonbasic slacks
(Checker support). -/
theorem card_basic_packing_eq_card_nonbasic_slack (n : ℕ) (S : Finset (Fin n ⊕ Fin n))
    (hS : S.card = n) :
    (univ.filter fun i => Sum.inl i ∈ S).card = (univ.filter fun j => Sum.inr j ∉ S).card := by
  set L := univ.filter fun i : Fin n => Sum.inl i ∈ S
  set R := univ.filter fun j : Fin n => Sum.inr j ∈ S
  have hSLR : S = L.disjSum R := by
    ext x
    cases x <;> simp [L, R]
  have hcard : L.card + R.card = n := by rw [← card_disjSum, ← hSLR, hS]
  have hsplit := Finset.card_filter_add_card_filter_not
    (s := (univ : Finset (Fin n))) (fun j : Fin n => Sum.inr j ∈ S)
  rw [card_univ, Fintype.card_fin] at hsplit
  have hR : R.card + (univ.filter fun j : Fin n => Sum.inr j ∉ S).card = n := hsplit
  omega

/-! ## Acceptance: the start cannot move a stored value beyond the gate -/

section Acceptance

variable {m n : Type*} [Fintype m] [Fintype n] [Nonempty m] [Nonempty n]

open MatrixGame

/-- **Two starts store values within the gate** (CRATES-CRASH-5;
`LEAP_CERTIFICATE.md:240-242`, `leap_packing.rs` `certificate`). Exact
arithmetic: if two accepted certificates `(p₁, q₁)` and `(p₂, q₂)` of the same
stage matrix have gaps `U - L ≤ ε`, their stored midpoints differ by at most
`ε`, each lies within `ε / 2` of the value, and the window lift
`max(·, f)` keeps both bounds because it is 1-Lipschitz. The simplex path
that produced each certificate plays no role. -/
theorem abs_accepted_sub_le (M : Matrix m n ℝ) {p₁ p₂ : m → ℝ} {q₁ q₂ : n → ℝ}
    (hp₁ : p₁ ∈ simplex m) (hq₁ : q₁ ∈ simplex n) (hp₂ : p₂ ∈ simplex m)
    (hq₂ : q₂ ∈ simplex n) {ε : ℝ}
    (h₁ : upperBound M q₁ - lowerBound M p₁ ≤ ε) (h₂ : upperBound M q₂ - lowerBound M p₂ ≤ ε)
    (f : ℝ) :
    |(lowerBound M p₁ + upperBound M q₁) / 2 - (lowerBound M p₂ + upperBound M q₂) / 2| ≤ ε ∧
      |max ((lowerBound M p₁ + upperBound M q₁) / 2) f -
          max ((lowerBound M p₂ + upperBound M q₂) / 2) f| ≤ ε ∧
      |max ((lowerBound M p₁ + upperBound M q₁) / 2) f - max (value M) f| ≤ ε / 2 := by
  have e₁ := abs_midpoint_sub_value_le M hp₁ hq₁ h₁
  have e₂ := abs_midpoint_sub_value_le M hp₂ hq₂ h₂
  have hd : |(lowerBound M p₁ + upperBound M q₁) / 2 - (lowerBound M p₂ + upperBound M q₂) / 2|
      ≤ ε := by
    rw [abs_le] at e₁ e₂ ⊢
    constructor <;> linarith [e₁.1, e₁.2, e₂.1, e₂.2]
  exact ⟨hd, (abs_max_sub_max_le_abs _ _ _).trans hd, (abs_max_sub_max_le_abs _ _ _).trans e₁⟩

/-- **The gate under an explicit rounding model** (CRATES-CRASH-5 doubt).
Model: the computed bounds `L', U'` satisfy `|L' - L| ≤ η` and
`|U' - U| ≤ η` for the exact bounds `L = lowerBound M p`, `U = upperBound M q`
of normalized mixtures, the code accepts when `U' - L' ≤ g`, and the stored
value `w` satisfies `|w - (L' + U') / 2| ≤ η'` (rounding of the midpoint). No
IEEE bit-level claim is made; `η` and `η'` stand for whatever forward error
bounds a floating-point analysis supplies. Then `w` lies within
`g / 2 + η + η'` of the value, and the computed gap is at least `-2η`, so the
code's lower tolerance `-1e-12` only admits rounding. -/
theorem abs_rounded_accepted_sub_value_le (M : Matrix m n ℝ) {p : m → ℝ} {q : n → ℝ}
    (hp : p ∈ simplex m) (hq : q ∈ simplex n) {L' U' w η η' g : ℝ}
    (hL : |L' - lowerBound M p| ≤ η) (hU : |U' - upperBound M q| ≤ η) (hgap : U' - L' ≤ g)
    (hw : |w - (L' + U') / 2| ≤ η') :
    |w - value M| ≤ g / 2 + η + η' ∧ -(2 * η) ≤ U' - L' := by
  obtain ⟨h1, h2⟩ := certificate_encloses_value M hp hq
  rw [abs_le] at hL hU hw
  refine ⟨?_, by linarith [hL.1, hL.2, hU.1, hU.2]⟩
  rw [abs_le]
  constructor <;> linarith [hL.1, hL.2, hU.1, hU.2, hw.1, hw.2]

/-- Two starts under the rounding model of `abs_rounded_accepted_sub_value_le`
store values within `g + 2η + 2η'`, before and after the window lift. -/
theorem abs_rounded_accepted_sub_le (M : Matrix m n ℝ) {p₁ p₂ : m → ℝ} {q₁ q₂ : n → ℝ}
    (hp₁ : p₁ ∈ simplex m) (hq₁ : q₁ ∈ simplex n) (hp₂ : p₂ ∈ simplex m)
    (hq₂ : q₂ ∈ simplex n) {L₁ U₁ L₂ U₂ w₁ w₂ η η' g : ℝ}
    (hL₁ : |L₁ - lowerBound M p₁| ≤ η) (hU₁ : |U₁ - upperBound M q₁| ≤ η) (hg₁ : U₁ - L₁ ≤ g)
    (hw₁ : |w₁ - (L₁ + U₁) / 2| ≤ η')
    (hL₂ : |L₂ - lowerBound M p₂| ≤ η) (hU₂ : |U₂ - upperBound M q₂| ≤ η) (hg₂ : U₂ - L₂ ≤ g)
    (hw₂ : |w₂ - (L₂ + U₂) / 2| ≤ η') (f : ℝ) :
    |w₁ - w₂| ≤ g + 2 * η + 2 * η' ∧ |max w₁ f - max w₂ f| ≤ g + 2 * η + 2 * η' := by
  have e₁ := (abs_rounded_accepted_sub_value_le M hp₁ hq₁ hL₁ hU₁ hg₁ hw₁).1
  have e₂ := (abs_rounded_accepted_sub_value_le M hp₂ hq₂ hL₂ hU₂ hg₂ hw₂).1
  have hd : |w₁ - w₂| ≤ g + 2 * η + 2 * η' := by
    rw [abs_le] at e₁ e₂ ⊢
    constructor <;> linarith [e₁.1, e₁.2, e₂.1, e₂.2]
  exact ⟨hd, (abs_max_sub_max_le_abs _ _ _).trans hd⟩

end Acceptance

/-! ## Non-vacuity examples -/

/-- A stage with `f - s[0] = 1`: the hypothesis `a[0] = 1` holds. -/
example : packCoeff (fun _ => 0) 1 0 = 1 := packCoeff_zero _ _ (by norm_num)

/-- A kinked two-action stage: `s = (0, -1)`, `f = 1`. Its kernel weight
`r[1] = -1` is negative, so the crash start is primal and dual infeasible. -/
example :
    let s : ℕ → ℝ := fun k => if k = 0 then 0 else -1
    basicValues (packG 2 (packCoeff s 1)) (crashBasis 2) (fun _ => 1) ⟨1, by omega⟩ < 0 ∧
      0 < reducedCost (packG 2 (packCoeff s 1)) (crashBasis 2) (packCost 2) (Sum.inr ⟨0, by omega⟩) := by
  intro s
  have hd : (1 : ℝ) - s 0 ≠ 0 := by simp [s]
  have hr : kernelWeights (kernelCoeff s 1) 1 < 0 := by
    rw [kernelWeights_succ]
    simp [kernelWeights_zero, kernelCoeff, s]
  exact crash_residue_infeasible 2 s 1 hd (by omega) hr

/-- The acceptance hypotheses are satisfiable: an optimal pair of any game has
gap `0 ≤ 1e-6`. -/
example (M : Matrix (Fin 2) (Fin 2) ℝ) :
    ∃ p ∈ MatrixGame.simplex (Fin 2), ∃ q ∈ MatrixGame.simplex (Fin 2),
      MatrixGame.upperBound M q - MatrixGame.lowerBound M p ≤ (1e-6 : ℝ) := by
  obtain ⟨p, hp, q, hq, h⟩ := MatrixGame.exists_optimal M
  exact ⟨p, hp, q, hq, by rw [h]; norm_num⟩

end Formal.Leap
