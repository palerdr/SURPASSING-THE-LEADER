import Mathlib

/-!
# Finite zero-sum matrix games

This module fixes the matrix-game conventions that every solver certificate
uses. The row player maximizes and the column player minimizes the payoff
`p ⬝ᵥ (M *ᵥ q)`. A mixed strategy is a point of `simplex ι`.

For a row strategy `p`, `lowerBound M p` is the smallest column payoff
`min_j (pᵀ M)_j`; the row player guarantees it. For a column strategy `q`,
`upperBound M q` is the largest row payoff `max_i (M q)_i`; the column player
holds the payoff to it. Weak duality gives `lowerBound M p ≤ upperBound M q`.
The von Neumann minimax theorem (derived here from Mathlib's Sion theorem)
makes the lower and upper values equal; `value M` is that common number.

The solver certificate is `certificate_encloses_value`: any feasible pair
encloses the value, so a gap of at most `ε` puts the midpoint within `ε / 2`
of the value.
-/

open Finset Matrix Set

set_option linter.unusedSectionVars false

namespace Formal.MatrixGame

variable {m n : Type*} [Fintype m] [Fintype n]

/-- Expected payoff to the maximizing row player. -/
def payoff (M : Matrix m n ℝ) (p : m → ℝ) (q : n → ℝ) : ℝ := p ⬝ᵥ (M *ᵥ q)

theorem payoff_eq_vecMul (M : Matrix m n ℝ) (p : m → ℝ) (q : n → ℝ) :
    payoff M p q = (p ᵥ* M) ⬝ᵥ q := by
  simp [payoff, dotProduct_mulVec]

theorem payoff_eq_sum (M : Matrix m n ℝ) (p : m → ℝ) (q : n → ℝ) :
    payoff M p q = ∑ i, ∑ j, p i * M i j * q j := by
  simp only [payoff, dotProduct, mulVec, Finset.mul_sum]
  refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => ?_
  ring

/-- Mixed strategies over a finite action type: the standard simplex. -/
abbrev simplex (ι : Type*) [Fintype ι] : Set (ι → ℝ) :=
  {w | (∀ i, 0 ≤ w i) ∧ ∑ i, w i = 1}

set_option linter.deprecated false in
theorem simplex_eq_stdSimplex (ι : Type*) [Fintype ι] : simplex ι = stdSimplex ℝ ι := rfl

set_option linter.deprecated false in
theorem convex_simplex (ι : Type*) [Fintype ι] : Convex ℝ (simplex ι) := by
  rw [simplex_eq_stdSimplex]; exact convex_stdSimplex ℝ ι

set_option linter.deprecated false in
theorem isCompact_simplex (ι : Type*) [Fintype ι] : IsCompact (simplex ι) := by
  rw [simplex_eq_stdSimplex]; exact isCompact_stdSimplex ℝ ι

theorem single_mem_simplex {ι : Type*} [Fintype ι] [DecidableEq ι] (i : ι) :
    (Pi.single i 1 : ι → ℝ) ∈ simplex ι := by
  refine ⟨fun j => ?_, by simp⟩
  by_cases h : j = i
  · subst h; simp
  · simp [h]

/-- A convex combination of values each at least `c` is at least `c`. -/
theorem le_dotProduct_of_mem_simplex {ι : Type*} [Fintype ι] {w v : ι → ℝ}
    (hw : w ∈ simplex ι) {c : ℝ} (hv : ∀ i, c ≤ v i) : c ≤ w ⬝ᵥ v := by
  have h : ∑ i, w i * c ≤ ∑ i, w i * v i :=
    Finset.sum_le_sum fun i _ => mul_le_mul_of_nonneg_left (hv i) (hw.1 i)
  simpa [dotProduct, ← Finset.sum_mul, hw.2] using h

/-- A convex combination of values each at most `c` is at most `c`. -/
theorem dotProduct_le_of_mem_simplex {ι : Type*} [Fintype ι] {w v : ι → ℝ}
    (hw : w ∈ simplex ι) {c : ℝ} (hv : ∀ i, v i ≤ c) : w ⬝ᵥ v ≤ c := by
  have h : ∑ i, w i * v i ≤ ∑ i, w i * c :=
    Finset.sum_le_sum fun i _ => mul_le_mul_of_nonneg_left (hv i) (hw.1 i)
  simpa [dotProduct, ← Finset.sum_mul, hw.2] using h

variable [Nonempty m] [Nonempty n]

/-- The row player's guarantee with `p`: its smallest column payoff. -/
noncomputable def lowerBound (M : Matrix m n ℝ) (p : m → ℝ) : ℝ :=
  univ.inf' univ_nonempty fun j => (p ᵥ* M) j

/-- The column player's guarantee with `q`: its largest row payoff. -/
noncomputable def upperBound (M : Matrix m n ℝ) (q : n → ℝ) : ℝ :=
  univ.sup' univ_nonempty fun i => (M *ᵥ q) i

theorem lowerBound_le_col (M : Matrix m n ℝ) (p : m → ℝ) (j : n) :
    lowerBound M p ≤ (p ᵥ* M) j :=
  Finset.inf'_le _ (mem_univ j)

theorem row_le_upperBound (M : Matrix m n ℝ) (q : n → ℝ) (i : m) :
    (M *ᵥ q) i ≤ upperBound M q :=
  Finset.le_sup' (fun i => (M *ᵥ q) i) (mem_univ i)

theorem le_lowerBound_iff (M : Matrix m n ℝ) (p : m → ℝ) (c : ℝ) :
    c ≤ lowerBound M p ↔ ∀ j, c ≤ (p ᵥ* M) j := by
  simp [lowerBound, Finset.le_inf'_iff]

theorem upperBound_le_iff (M : Matrix m n ℝ) (q : n → ℝ) (c : ℝ) :
    upperBound M q ≤ c ↔ ∀ i, (M *ᵥ q) i ≤ c := by
  simp [upperBound, Finset.sup'_le_iff]

theorem exists_lowerBound_eq (M : Matrix m n ℝ) (p : m → ℝ) :
    ∃ j, lowerBound M p = (p ᵥ* M) j := by
  obtain ⟨j, -, hj⟩ := Finset.exists_mem_eq_inf' (univ_nonempty (α := n)) fun j => (p ᵥ* M) j
  exact ⟨j, hj⟩

theorem exists_upperBound_eq (M : Matrix m n ℝ) (q : n → ℝ) :
    ∃ i, upperBound M q = (M *ᵥ q) i := by
  obtain ⟨i, -, hi⟩ := Finset.exists_mem_eq_sup' (univ_nonempty (α := m)) fun i => (M *ᵥ q) i
  exact ⟨i, hi⟩

theorem lowerBound_le_payoff (M : Matrix m n ℝ) (p : m → ℝ) {q : n → ℝ}
    (hq : q ∈ simplex n) : lowerBound M p ≤ payoff M p q := by
  rw [payoff_eq_vecMul, dotProduct_comm]
  exact le_dotProduct_of_mem_simplex hq (lowerBound_le_col M p)

theorem payoff_le_upperBound (M : Matrix m n ℝ) {p : m → ℝ} (hp : p ∈ simplex m)
    (q : n → ℝ) : payoff M p q ≤ upperBound M q :=
  dotProduct_le_of_mem_simplex hp (row_le_upperBound M q)

/-- Weak duality for one pair of mixed strategies. -/
theorem lowerBound_le_upperBound (M : Matrix m n ℝ) {p : m → ℝ} {q : n → ℝ}
    (hp : p ∈ simplex m) (hq : q ∈ simplex n) :
    lowerBound M p ≤ upperBound M q :=
  (lowerBound_le_payoff M p hq).trans (payoff_le_upperBound M hp q)

theorem uniform_mem_simplex (ι : Type*) [Fintype ι] [Nonempty ι] :
    (fun _ : ι => (Fintype.card ι : ℝ)⁻¹) ∈ simplex ι := by
  refine ⟨fun _ => by positivity, ?_⟩
  simp [Finset.sum_const, Finset.card_univ]

/-- The lower value: the best guarantee of the row player. -/
noncomputable def lowerValue (M : Matrix m n ℝ) : ℝ :=
  ⨆ p : simplex m, lowerBound M p

/-- The upper value: the best guarantee of the column player. -/
noncomputable def upperValue (M : Matrix m n ℝ) : ℝ :=
  ⨅ q : simplex n, upperBound M q

private theorem simplex_nonempty (ι : Type*) [Fintype ι] [Nonempty ι] :
    Nonempty (simplex ι) :=
  ⟨⟨_, uniform_mem_simplex ι⟩⟩

private theorem bddAbove_lowerBound (M : Matrix m n ℝ) :
    BddAbove (range fun p : simplex m => lowerBound M p) := by
  refine ⟨upperBound M (fun _ => (Fintype.card n : ℝ)⁻¹), ?_⟩
  rintro _ ⟨p, rfl⟩
  exact lowerBound_le_upperBound M p.2 (uniform_mem_simplex n)

private theorem bddBelow_upperBound (M : Matrix m n ℝ) :
    BddBelow (range fun q : simplex n => upperBound M q) := by
  refine ⟨lowerBound M (fun _ => (Fintype.card m : ℝ)⁻¹), ?_⟩
  rintro _ ⟨q, rfl⟩
  exact lowerBound_le_upperBound M (uniform_mem_simplex m) q.2

theorem lowerBound_le_lowerValue (M : Matrix m n ℝ) {p : m → ℝ} (hp : p ∈ simplex m) :
    lowerBound M p ≤ lowerValue M :=
  le_ciSup (bddAbove_lowerBound M) ⟨p, hp⟩

theorem upperValue_le_upperBound (M : Matrix m n ℝ) {q : n → ℝ} (hq : q ∈ simplex n) :
    upperValue M ≤ upperBound M q :=
  ciInf_le (bddBelow_upperBound M) ⟨q, hq⟩

theorem lowerValue_le_upperBound (M : Matrix m n ℝ) {q : n → ℝ} (hq : q ∈ simplex n) :
    lowerValue M ≤ upperBound M q := by
  have := simplex_nonempty m
  exact ciSup_le fun p => lowerBound_le_upperBound M p.2 hq

theorem lowerBound_le_upperValue (M : Matrix m n ℝ) {p : m → ℝ} (hp : p ∈ simplex m) :
    lowerBound M p ≤ upperValue M := by
  have := simplex_nonempty n
  exact le_ciInf fun q => lowerBound_le_upperBound M hp q.2

theorem lowerValue_le_upperValue (M : Matrix m n ℝ) : lowerValue M ≤ upperValue M := by
  have := simplex_nonempty n
  exact le_ciInf fun q => lowerValue_le_upperBound M q.2

section Minimax

/-- The payoff is affine in the column strategy. -/
private theorem payoff_convex_comb_right (M : Matrix m n ℝ) (p : m → ℝ) (x y : n → ℝ)
    (a b : ℝ) : payoff M p (a • x + b • y) = a * payoff M p x + b * payoff M p y := by
  simp [payoff, mulVec_add, mulVec_smul, dotProduct_add, dotProduct_smul]

/-- The payoff is affine in the row strategy. -/
private theorem payoff_convex_comb_left (M : Matrix m n ℝ) (x y : m → ℝ) (q : n → ℝ)
    (a b : ℝ) : payoff M (a • x + b • y) q = a * payoff M x q + b * payoff M y q := by
  simp [payoff, add_dotProduct, smul_dotProduct]

private theorem continuous_payoff_right (M : Matrix m n ℝ) (p : m → ℝ) :
    Continuous fun q : n → ℝ => payoff M p q := by
  simp only [payoff_eq_sum]
  fun_prop

private theorem continuous_payoff_left (M : Matrix m n ℝ) (q : n → ℝ) :
    Continuous fun p : m → ℝ => payoff M p q := by
  simp only [payoff_eq_sum]
  fun_prop

/-- Mixed strategies that form a saddle point of the bilinear payoff. -/
theorem exists_saddle (M : Matrix m n ℝ) :
    ∃ q ∈ simplex n, ∃ p ∈ simplex m,
      ∀ q' ∈ simplex n, ∀ p' ∈ simplex m, payoff M p' q ≤ payoff M p q' := by
  have hX : (simplex n).Nonempty := ⟨_, uniform_mem_simplex n⟩
  have hY : (simplex m).Nonempty := ⟨_, uniform_mem_simplex m⟩
  obtain ⟨q, hq, p, hp, h⟩ :=
    Sion.exists_isSaddlePointOn' (f := fun (q : n → ℝ) (p : m → ℝ) => payoff M p q)
      (ne_X := hX) (cX := convex_simplex n) (kX := isCompact_simplex n)
      (hfy := fun p _ => (continuous_payoff_right M p).lowerSemicontinuous.lowerSemicontinuousOn _)
      (hfy' := fun p _ => ConvexOn.quasiconvexOn
        ⟨convex_simplex n, fun x _ y _ a b _ _ _ => by
          dsimp only; rw [payoff_convex_comb_right]; simp [smul_eq_mul]⟩)
      (ne_Y := hY) (cY := convex_simplex m) (kY := isCompact_simplex m)
      (hfx := fun q _ => (continuous_payoff_left M q).upperSemicontinuous.upperSemicontinuousOn _)
      (hfx' := fun q _ => ConcaveOn.quasiconcaveOn
        ⟨convex_simplex m, fun x _ y _ a b _ _ _ => by
          dsimp only; rw [payoff_convex_comb_left]; simp [smul_eq_mul]⟩)
  exact ⟨q, hq, p, hp, fun q' hq' p' hp' => h q' hq' p' hp'⟩

/-- Optimal strategies exist and certify each other: `upperBound q = lowerBound p`. -/
theorem exists_optimal (M : Matrix m n ℝ) :
    ∃ p ∈ simplex m, ∃ q ∈ simplex n, upperBound M q = lowerBound M p := by
  classical
  obtain ⟨q, hq, p, hp, h⟩ := exists_saddle M
  refine ⟨p, hp, q, hq, le_antisymm ?_ (lowerBound_le_upperBound M hp hq)⟩
  rw [upperBound_le_iff]
  intro i
  rw [le_lowerBound_iff]
  intro j
  have h1 : payoff M (Pi.single i 1) q = (M *ᵥ q) i := by simp [payoff, single_dotProduct]
  have h2 : payoff M p (Pi.single j 1) = (p ᵥ* M) j := by
    simp [payoff_eq_vecMul, dotProduct_single]
  have := h (Pi.single j 1) (single_mem_simplex j) (Pi.single i 1) (single_mem_simplex i)
  rwa [h1, h2] at this

/-- The von Neumann minimax theorem for finite matrix games. -/
theorem lowerValue_eq_upperValue (M : Matrix m n ℝ) : lowerValue M = upperValue M := by
  obtain ⟨p, hp, q, hq, h⟩ := exists_optimal M
  refine le_antisymm (lowerValue_le_upperValue M) ?_
  calc upperValue M ≤ upperBound M q := upperValue_le_upperBound M hq
    _ = lowerBound M p := h
    _ ≤ lowerValue M := lowerBound_le_lowerValue M hp

end Minimax

/-- The value of the matrix game. -/
noncomputable def value (M : Matrix m n ℝ) : ℝ := lowerValue M

theorem value_eq_upperValue (M : Matrix m n ℝ) : value M = upperValue M :=
  lowerValue_eq_upperValue M

/-- Every feasible strategy pair encloses the value. -/
theorem certificate_encloses_value (M : Matrix m n ℝ) {p : m → ℝ} {q : n → ℝ}
    (hp : p ∈ simplex m) (hq : q ∈ simplex n) :
    lowerBound M p ≤ value M ∧ value M ≤ upperBound M q :=
  ⟨lowerBound_le_lowerValue M hp, lowerValue_le_upperBound M hq⟩

/-- The solver acceptance rule: a saddle gap of at most `ε` puts the stored
midpoint within `ε / 2` of the value. -/
theorem abs_midpoint_sub_value_le (M : Matrix m n ℝ) {p : m → ℝ} {q : n → ℝ}
    (hp : p ∈ simplex m) (hq : q ∈ simplex n) {ε : ℝ}
    (hgap : upperBound M q - lowerBound M p ≤ ε) :
    |(lowerBound M p + upperBound M q) / 2 - value M| ≤ ε / 2 := by
  obtain ⟨h1, h2⟩ := certificate_encloses_value M hp hq
  rw [abs_le]
  constructor <;> linarith

/-- Any number inside a certified enclosure of width at most `ε` is within `ε`
of the value. -/
theorem abs_sub_value_le_of_mem (M : Matrix m n ℝ) {p : m → ℝ} {q : n → ℝ}
    (hp : p ∈ simplex m) (hq : q ∈ simplex n) {v ε : ℝ}
    (hv : v ∈ Icc (lowerBound M p) (upperBound M q))
    (hgap : upperBound M q - lowerBound M p ≤ ε) : |v - value M| ≤ ε := by
  obtain ⟨h1, h2⟩ := certificate_encloses_value M hp hq
  rw [abs_le]
  constructor <;> linarith [hv.1, hv.2]

/-- Equalizing strategies: if `p` makes every column pay `c` and `q` makes every
row pay `c'`, then `c = c'` and both equal the value. -/
theorem value_of_equalizers (M : Matrix m n ℝ) {p : m → ℝ} {q : n → ℝ}
    (hp : p ∈ simplex m) (hq : q ∈ simplex n) {c c' : ℝ}
    (hc : ∀ j, (p ᵥ* M) j = c) (hc' : ∀ i, (M *ᵥ q) i = c') :
    c = c' ∧ value M = c := by
  have hl : lowerBound M p = c :=
    le_antisymm ((lowerBound_le_col M p (Classical.arbitrary n)).trans_eq (hc _))
      ((le_lowerBound_iff M p c).2 fun j => (hc j).ge)
  have hu : upperBound M q = c' :=
    le_antisymm ((upperBound_le_iff M q c').2 fun i => (hc' i).le)
      ((hc' (Classical.arbitrary m)).symm.le.trans (row_le_upperBound M q _))
  obtain ⟨h1, h2⟩ := certificate_encloses_value M hp hq
  have hpq : payoff M p q = c := by
    rw [payoff_eq_vecMul, dotProduct_comm]
    exact le_antisymm (dotProduct_le_of_mem_simplex hq fun j => (hc j).le)
      (le_dotProduct_of_mem_simplex hq fun j => (hc j).ge)
  have hpq' : payoff M p q = c' := by
    exact le_antisymm (dotProduct_le_of_mem_simplex hp fun i => (hc' i).le)
      (le_dotProduct_of_mem_simplex hp fun i => (hc' i).ge)
  refine ⟨hpq.symm.trans hpq', le_antisymm ?_ ?_⟩
  · rw [← hpq, hpq', ← hu]; exact h2
  · rw [← hl]; exact h1

/-- The value is `1`-Lipschitz in the entrywise maximum norm. -/
theorem abs_value_sub_le (M M' : Matrix m n ℝ) {δ : ℝ} (h : ∀ i j, |M i j - M' i j| ≤ δ) :
    |value M - value M'| ≤ δ := by
  have key : ∀ A B : Matrix m n ℝ, (∀ i j, |A i j - B i j| ≤ δ) → value A - δ ≤ value B := by
    intro A B hAB
    obtain ⟨p, hp, q, hq, hopt⟩ := exists_optimal A
    obtain ⟨-, -, q', hq', -⟩ := exists_optimal B
    have hA : value A = lowerBound A p :=
      le_antisymm ((certificate_encloses_value A hp hq).2.trans_eq hopt)
        (certificate_encloses_value A hp hq).1
    -- the row player's optimal strategy in `A` loses at most `δ` in `B`
    have hcol : ∀ j, (p ᵥ* A) j - δ ≤ (p ᵥ* B) j := by
      intro j
      have hdiff : (p ᵥ* (A - B)) j ≤ δ := by
        show p ⬝ᵥ (fun i => A i j - B i j) ≤ δ
        exact dotProduct_le_of_mem_simplex hp fun i => (le_abs_self _).trans (hAB i j)
      rw [vecMul_sub] at hdiff
      simp only [Pi.sub_apply] at hdiff
      linarith
    calc value A - δ = lowerBound A p - δ := by rw [hA]
      _ ≤ lowerBound B p := by
          rw [le_lowerBound_iff]
          intro j
          exact (sub_le_sub_right (lowerBound_le_col A p j) δ).trans (hcol j)
      _ ≤ value B := (certificate_encloses_value B hp hq').1
  rw [abs_le]
  constructor
  · have := key M' M fun i j => by rw [abs_sub_comm]; exact h i j
    linarith
  · have := key M M' h
    linarith

/-- The pure-saddle test: if the best row minimum equals the smallest column
maximum, that common number is the value. -/
theorem value_of_pure_saddle (M : Matrix m n ℝ) (i₀ : m) (j₀ : n)
    (hrow : ∀ j, M i₀ j₀ ≤ M i₀ j) (hcol : ∀ i, M i j₀ ≤ M i₀ j₀) :
    value M = M i₀ j₀ := by
  classical
  have hp := single_mem_simplex i₀
  have hq := single_mem_simplex j₀
  obtain ⟨h1, h2⟩ := certificate_encloses_value M hp hq
  have hl : M i₀ j₀ ≤ lowerBound M (Pi.single i₀ 1) := by
    rw [le_lowerBound_iff]; intro j; simpa [single_vecMul] using hrow j
  have hu : upperBound M (Pi.single j₀ 1) ≤ M i₀ j₀ := by
    rw [upperBound_le_iff]; intro i; simpa [mulVec_single] using hcol i
  exact le_antisymm (h2.trans hu) (hl.trans h1)

end Formal.MatrixGame
