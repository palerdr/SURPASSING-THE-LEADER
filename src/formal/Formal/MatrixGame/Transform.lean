import Formal.MatrixGame.Basic

/-!
# Value transformations

The solvers move stage matrices before they solve them: the C++ LP rung
shifts payoffs into `[1, 3]`, the STL scalar LP retries after subtracting a
common payoff constant, the packing fallback writes `M = f 11ᵀ - d A`, and a
role swap negates and transposes. These lemmas say how each move changes the
certificate bounds and the value.
-/

open Finset Matrix

set_option linter.unusedSectionVars false

namespace Formal.MatrixGame

variable {m n : Type*} [Fintype m] [Fintype n] [Nonempty m] [Nonempty n]

theorem vecMul_add_const {p : m → ℝ} (hp : p ∈ simplex m) (M : Matrix m n ℝ) (c : ℝ) (j : n) :
    (p ᵥ* (M + Matrix.of fun _ _ => c)) j = (p ᵥ* M) j + c := by
  simp only [vecMul, dotProduct, Matrix.add_apply, Matrix.of_apply, mul_add, Finset.sum_add_distrib,
    ← Finset.sum_mul, hp.2, one_mul]

theorem mulVec_add_const {q : n → ℝ} (hq : q ∈ simplex n) (M : Matrix m n ℝ) (c : ℝ) (i : m) :
    ((M + Matrix.of fun _ _ => c) *ᵥ q) i = (M *ᵥ q) i + c := by
  simp only [mulVec, dotProduct, Matrix.add_apply, Matrix.of_apply, add_mul, Finset.sum_add_distrib,
    ← Finset.mul_sum, hq.2, mul_one]

theorem lowerBound_add_const {p : m → ℝ} (hp : p ∈ simplex m) (M : Matrix m n ℝ) (c : ℝ) :
    lowerBound (M + Matrix.of fun _ _ => c) p = lowerBound M p + c := by
  apply le_antisymm
  · obtain ⟨j, hj⟩ := exists_lowerBound_eq M p
    rw [hj, ← vecMul_add_const hp]; exact lowerBound_le_col _ _ _
  · rw [le_lowerBound_iff]; intro j
    rw [vecMul_add_const hp]; linarith [lowerBound_le_col M p j]

theorem upperBound_add_const {q : n → ℝ} (hq : q ∈ simplex n) (M : Matrix m n ℝ) (c : ℝ) :
    upperBound (M + Matrix.of fun _ _ => c) q = upperBound M q + c := by
  apply le_antisymm
  · rw [upperBound_le_iff]; intro i
    rw [mulVec_add_const hq]; linarith [row_le_upperBound M q i]
  · obtain ⟨i, hi⟩ := exists_upperBound_eq M q
    rw [hi, ← mulVec_add_const hq]; exact row_le_upperBound _ _ _

/-- Adding a constant to every payoff adds it to the value. -/
theorem value_add_const (M : Matrix m n ℝ) (c : ℝ) :
    value (M + Matrix.of fun _ _ => c) = value M + c := by
  obtain ⟨p, hp, q, hq, h⟩ := exists_optimal M
  have e1 := certificate_encloses_value (M + Matrix.of fun _ _ => c) hp hq
  have e2 := certificate_encloses_value M hp hq
  rw [lowerBound_add_const hp, upperBound_add_const hq] at e1
  rw [h] at e1 e2
  linarith [e1.1, e1.2, e2.1, e2.2]

theorem vecMul_smul_apply (p : m → ℝ) (M : Matrix m n ℝ) (k : ℝ) (j : n) :
    (p ᵥ* (k • M)) j = k * (p ᵥ* M) j := by
  simp [Matrix.vecMul_smul, smul_eq_mul]

theorem smul_mulVec_apply (q : n → ℝ) (M : Matrix m n ℝ) (k : ℝ) (i : m) :
    ((k • M) *ᵥ q) i = k * (M *ᵥ q) i := by
  simp [Matrix.smul_mulVec, smul_eq_mul]

theorem lowerBound_smul {k : ℝ} (hk : 0 < k) (M : Matrix m n ℝ) (p : m → ℝ) :
    lowerBound (k • M) p = k * lowerBound M p := by
  apply le_antisymm
  · obtain ⟨j, hj⟩ := exists_lowerBound_eq M p
    rw [hj, ← vecMul_smul_apply]; exact lowerBound_le_col _ _ _
  · rw [le_lowerBound_iff]; intro j
    rw [vecMul_smul_apply]; exact mul_le_mul_of_nonneg_left (lowerBound_le_col M p j) hk.le

theorem upperBound_smul {k : ℝ} (hk : 0 < k) (M : Matrix m n ℝ) (q : n → ℝ) :
    upperBound (k • M) q = k * upperBound M q := by
  apply le_antisymm
  · rw [upperBound_le_iff]; intro i
    rw [smul_mulVec_apply]; exact mul_le_mul_of_nonneg_left (row_le_upperBound M q i) hk.le
  · obtain ⟨i, hi⟩ := exists_upperBound_eq M q
    rw [hi, ← smul_mulVec_apply]; exact row_le_upperBound _ _ _

/-- Scaling every payoff by `k > 0` scales the value. -/
theorem value_smul {k : ℝ} (hk : 0 < k) (M : Matrix m n ℝ) : value (k • M) = k * value M := by
  obtain ⟨p, hp, q, hq, h⟩ := exists_optimal M
  have e1 := certificate_encloses_value (k • M) hp hq
  have e2 := certificate_encloses_value M hp hq
  rw [lowerBound_smul hk, upperBound_smul hk, h] at e1
  rw [h] at e2
  have : value M = lowerBound M p := le_antisymm e2.2 e2.1
  rw [this]; exact le_antisymm e1.2 e1.1

/-- The role swap: the game `-Mᵀ`, in which the old column player maximizes. -/
theorem lowerBound_neg_transpose (M : Matrix m n ℝ) (q : n → ℝ) :
    lowerBound (-Mᵀ) q = -upperBound M q := by
  apply le_antisymm
  · obtain ⟨i, hi⟩ := exists_upperBound_eq M q
    rw [hi]
    have : (q ᵥ* (-Mᵀ)) i = -(M *ᵥ q) i := by
      simp [vecMul_neg, vecMul_transpose]
    rw [← this]; exact lowerBound_le_col _ _ _
  · rw [le_lowerBound_iff]; intro i
    have : (q ᵥ* (-Mᵀ)) i = -(M *ᵥ q) i := by simp [vecMul_neg, vecMul_transpose]
    rw [this, neg_le_neg_iff]; exact row_le_upperBound M q i

theorem upperBound_neg_transpose (M : Matrix m n ℝ) (p : m → ℝ) :
    upperBound (-Mᵀ) p = -lowerBound M p := by
  apply le_antisymm
  · rw [upperBound_le_iff]; intro j
    have : ((-Mᵀ) *ᵥ p) j = -(p ᵥ* M) j := by simp [neg_mulVec, mulVec_transpose]
    rw [this, neg_le_neg_iff]; exact lowerBound_le_col M p j
  · obtain ⟨j, hj⟩ := exists_lowerBound_eq M p
    rw [hj]
    have : ((-Mᵀ) *ᵥ p) j = -(p ᵥ* M) j := by simp [neg_mulVec, mulVec_transpose]
    rw [← this]; exact row_le_upperBound _ _ _

/-- Swapping roles negates the value: `value (-Mᵀ) = -value M`. -/
theorem value_neg_transpose (M : Matrix m n ℝ) : value (-Mᵀ) = -value M := by
  obtain ⟨p, hp, q, hq, h⟩ := exists_optimal M
  have e1 := certificate_encloses_value (-Mᵀ) hq hp
  have e2 := certificate_encloses_value M hp hq
  rw [lowerBound_neg_transpose, upperBound_neg_transpose] at e1
  rw [h] at e1 e2
  have : value M = lowerBound M p := le_antisymm e2.2 e2.1
  rw [this]; exact le_antisymm e1.2 e1.1

/-- The value is monotone in the payoffs. -/
theorem value_mono {M M' : Matrix m n ℝ} (h : ∀ i j, M i j ≤ M' i j) : value M ≤ value M' := by
  obtain ⟨p, hp, q, hq, hpq⟩ := exists_optimal M
  obtain ⟨p', hp', q', hq', hpq'⟩ := exists_optimal M'
  have hv : value M = lowerBound M p :=
    le_antisymm ((certificate_encloses_value M hp hq).2.trans_eq hpq) (certificate_encloses_value M hp hq).1
  calc value M = lowerBound M p := hv
    _ ≤ lowerBound M' p := by
        rw [le_lowerBound_iff]; intro j
        refine (lowerBound_le_col M p j).trans ?_
        simp only [vecMul, dotProduct]
        exact Finset.sum_le_sum fun i _ => mul_le_mul_of_nonneg_left (h i j) (hp.1 i)
    _ ≤ value M' := (certificate_encloses_value M' hp hq').1

/-- The value lies between the smallest and the largest payoff. -/
theorem value_mem_entry_range (M : Matrix m n ℝ) {a b : ℝ} (ha : ∀ i j, a ≤ M i j)
    (hb : ∀ i j, M i j ≤ b) : a ≤ value M ∧ value M ≤ b := by
  have h1 := value_mono (M := Matrix.of fun (_ : m) (_ : n) => a) (M' := M) (fun i j => ha i j)
  have h2 := value_mono (M := M) (M' := Matrix.of fun (_ : m) (_ : n) => b) (fun i j => hb i j)
  have hc : ∀ c : ℝ, value (Matrix.of fun (_ : m) (_ : n) => c) = c := by
    intro c
    have := value_add_const (0 : Matrix m n ℝ) c
    have h0 : value (0 : Matrix m n ℝ) = 0 := by
      have := value_smul (k := 2) (by norm_num) (0 : Matrix m n ℝ)
      simp only [smul_zero] at this
      linarith
    simpa [h0] using this
  rw [hc] at h1 h2
  exact ⟨h1, h2⟩

end Formal.MatrixGame
