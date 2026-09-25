import Mathlib
import Formal.MatrixGame.Basic
import Formal.MatrixGame.Transform
import Formal.DTH.Rules
import Formal.Toeplitz.Basic

/-!
# Rung 1: the pure-saddle test

This module proves the pure-strategy shortcut that every solver runs before
the equalizer recurrence or the LP.

Conventions.

* Rows are Dropper seconds, columns are Checker seconds, both `0`-based. The
  row player maximizes, as in `Formal.MatrixGame`.
* `rowMin M d = min_c M[d, c]` and `colMax M c = max_d M[d, c]`.
  `pureMaximin M = max_d rowMin M d` and `pureMinimax M = min_c colMax M c`.
  These are defined over any linear order `α`. A linear order models finite
  IEEE doubles under `fmin`/`fmax`, `np.minimum`, and Rust `f64::min`: those
  operations select one of their arguments. The model identifies `+0` and
  `-0` and excludes NaN; the solvers reject nonfinite inputs first.
* `toeplitzL n s f` is `Formal.Toeplitz.toeplitz` over a linear order:
  `M[d, c] = s (c - d)` for `d ≤ c`, else `f`. Over `ℝ` the two are equal by
  `rfl`. A stage with `N = n + 1` actions uses `toeplitzL (n + 1)`; the
  solvers have `N = 60`, so `n = 59`.
* `prefixMin s j` and `prefixMax s j` are the running minimum and maximum of
  `s 0, …, s j`, as the C++ scan and the NumPy `accumulate` build them.

Main results.

* `rowMin_toeplitzL`, `colMax_toeplitzL`: the `O(60)` prefix-scan formulas
  (`row_min[d] = prefix_min[59-d]`, with `f` when `d > 0`;
  `col_max[c] = prefix_max[c]`, with `f` when `c + 1 < 60`).
* `pureMaximin_toeplitzL`, `pureMinimax_toeplitzL`: the closed forms
  `max(lo, min(f, s0))` and `min(hi, max(f, s0))`.
* `pureMaximin_le_value`, `value_le_pureMinimax`, `pure_midpoint_error`:
  soundness of accepting the midpoint when the pure gap is at most `ε`.
* `scanMax_spec`, `cpp_scan_drop`, `cpp_scan_check`: the strict-comparison
  scan returns the lowest maximizing row and minimizing column.
* `fdot_single`, `cpp_certify_onehot`: under a rounding model where rounding
  fixes representable numbers, the one-hot certificate recomputes the scan
  bounds exactly.
* `foldl_max_perm`, `foldl_min_perm`: min and max folds do not depend on order.
* `rustPureSaddle_isSome_iff`, `pythonPureSaddle_eq_rust`,
  `rustPureSaddle_error`, `nearPennies_counterexample`: the abstract solver's
  `1e-12` near-saddle search, its `2 τ` value error, and a near-saddle that
  it accepts although the matrix has no exact pure saddle.
* `window_pureMaximin`, `window_pureMinimax`, `leap_rust_eq_python`,
  `window_rung0_error`: the STL leap-window lift. The lifted bounds are the
  pure bounds of the 61-row window matrix, and the Python and Rust rung-0
  stores agree on the Toeplitz closed forms.
* `constant_stage_value`, `linspace_stage_value`,
  `rowMin_injective_counterexample`: the `dth_compact` tests and a
  counterexample to the "one of four numbers" step of `architecture.md`.

Floating point. Min and max are modelled over a linear order and are exact.
The gap subtraction and the midpoint use the abstract rounding model
`fl(x) = x (1 + δ) + η` in `rounded_pure_rung_error`. The `1e-12` closeness
tests and the window lift are stated over `ℝ` with exact subtraction.
-/

open Finset Matrix

set_option linter.unusedSectionVars false

namespace Formal.Toeplitz.PureSaddle

open Formal.MatrixGame

/-! ## Pure security levels of a finite matrix -/

section Generic

variable {α : Type*} [LinearOrder α] {m n : Type*} [Fintype m] [Fintype n]
  [Nonempty m] [Nonempty n]

/-- The smallest entry of row `i`: what pure row `i` guarantees. -/
def rowMin (M : Matrix m n α) (i : m) : α := univ.inf' univ_nonempty (M i)

/-- The largest entry of column `j`: what pure column `j` concedes. -/
def colMax (M : Matrix m n α) (j : n) : α := univ.sup' univ_nonempty fun i => M i j

/-- The pure maximin `max_i min_j M[i, j]`. -/
def pureMaximin (M : Matrix m n α) : α := univ.sup' univ_nonempty (rowMin M)

/-- The pure minimax `min_j max_i M[i, j]`. -/
def pureMinimax (M : Matrix m n α) : α := univ.inf' univ_nonempty (colMax M)

theorem rowMin_le (M : Matrix m n α) (i : m) (j : n) : rowMin M i ≤ M i j :=
  inf'_le _ (mem_univ j)

theorem le_colMax (M : Matrix m n α) (i : m) (j : n) : M i j ≤ colMax M j :=
  le_sup' (fun i => M i j) (mem_univ i)

theorem rowMin_le_pureMaximin (M : Matrix m n α) (i : m) : rowMin M i ≤ pureMaximin M :=
  le_sup' (rowMin M) (mem_univ i)

theorem pureMinimax_le_colMax (M : Matrix m n α) (j : n) : pureMinimax M ≤ colMax M j :=
  inf'_le (colMax M) (mem_univ j)

theorem rowMin_le_colMax (M : Matrix m n α) (i : m) (j : n) : rowMin M i ≤ colMax M j :=
  (rowMin_le M i j).trans (le_colMax M i j)

theorem exists_rowMin_eq (M : Matrix m n α) (i : m) : ∃ j, rowMin M i = M i j := by
  obtain ⟨j, -, hj⟩ := exists_mem_eq_inf' (univ_nonempty (α := n)) (M i)
  exact ⟨j, hj⟩

theorem exists_colMax_eq (M : Matrix m n α) (j : n) : ∃ i, colMax M j = M i j := by
  obtain ⟨i, -, hi⟩ := exists_mem_eq_sup' (univ_nonempty (α := m)) fun i => M i j
  exact ⟨i, hi⟩

theorem exists_pureMaximin_eq (M : Matrix m n α) : ∃ i, pureMaximin M = rowMin M i := by
  obtain ⟨i, -, hi⟩ := exists_mem_eq_sup' (univ_nonempty (α := m)) (rowMin M)
  exact ⟨i, hi⟩

theorem exists_pureMinimax_eq (M : Matrix m n α) : ∃ j, pureMinimax M = colMax M j := by
  obtain ⟨j, -, hj⟩ := exists_mem_eq_inf' (univ_nonempty (α := n)) (colMax M)
  exact ⟨j, hj⟩

/-- Pure weak duality: `max_i min_j M ≤ min_j max_i M`. -/
theorem pureMaximin_le_pureMinimax (M : Matrix m n α) : pureMaximin M ≤ pureMinimax M :=
  sup'_le _ _ fun i _ => le_inf' _ _ fun j _ => rowMin_le_colMax M i j

/-- Min and max select entries: the pure maximin is an entry of `M`. Over a
linear order that models floats, the computed bound carries no rounding. -/
theorem pureMaximin_mem_entries (M : Matrix m n α) : ∃ i j, pureMaximin M = M i j := by
  obtain ⟨i, hi⟩ := exists_pureMaximin_eq M
  obtain ⟨j, hj⟩ := exists_rowMin_eq M i
  exact ⟨i, j, hi.trans hj⟩

/-- Min and max select entries: the pure minimax is an entry of `M`. -/
theorem pureMinimax_mem_entries (M : Matrix m n α) : ∃ i j, pureMinimax M = M i j := by
  obtain ⟨j, hj⟩ := exists_pureMinimax_eq M
  obtain ⟨i, hi⟩ := exists_colMax_eq M j
  exact ⟨i, j, hj.trans hi⟩

/-- An exact pure saddle: the entry is the minimum of its row and the maximum
of its column. -/
def IsPureSaddle (M : Matrix m n α) (i : m) (j : n) : Prop :=
  M i j = rowMin M i ∧ M i j = colMax M j

/-- **ABSTRACT-MAT-2 (a)** (`src/abstract/matrix.py:128-160`,
`src/abstract/docs/MODEL.md:150-152`): an exact pure saddle exists if and only
if the pure maximin equals the pure minimax. -/
theorem exists_pureSaddle_iff (M : Matrix m n α) :
    (∃ i j, IsPureSaddle M i j) ↔ pureMaximin M = pureMinimax M := by
  constructor
  · rintro ⟨i, j, h1, h2⟩
    refine le_antisymm (pureMaximin_le_pureMinimax M) ?_
    calc pureMinimax M ≤ colMax M j := pureMinimax_le_colMax M j
      _ = rowMin M i := h2.symm.trans h1
      _ ≤ pureMaximin M := rowMin_le_pureMaximin M i
  · intro h
    obtain ⟨i, hi⟩ := exists_pureMaximin_eq M
    obtain ⟨j, hj⟩ := exists_pureMinimax_eq M
    have h1 := rowMin_le M i j
    have h2 := le_colMax M i j
    have h3 : rowMin M i = colMax M j := by rw [← hi, ← hj, h]
    exact ⟨i, j, le_antisymm (h3 ▸ h2) h1, le_antisymm h2 (h3 ▸ h1)⟩

end Generic

/-! ## Pure bounds enclose the mixed value -/

section Value

variable {m n : Type*} [Fintype m] [Fintype n] [Nonempty m] [Nonempty n]

/-- The row player's guarantee with pure row `i` is the row minimum. One-hot
dot products are exact sums of one entry. -/
theorem lowerBound_single [DecidableEq m] (M : Matrix m n ℝ) (i : m) :
    lowerBound M (Pi.single i 1) = rowMin M i := by
  unfold lowerBound rowMin
  congr 1
  funext j
  simp [single_vecMul]

/-- The column player's guarantee with pure column `j` is the column maximum. -/
theorem upperBound_single [DecidableEq n] (M : Matrix m n ℝ) (j : n) :
    upperBound M (Pi.single j 1) = colMax M j := by
  unfold upperBound colMax
  congr 1
  funext i
  simp [mulVec_single]

/-- **DTH-PURE-2, CPP-CODE-PURE-3**: the pure maximin is a lower bound on the
mixed value (`src/dth/docs/BUILD.md:100-106`). -/
theorem pureMaximin_le_value (M : Matrix m n ℝ) : pureMaximin M ≤ value M := by
  classical
  obtain ⟨i, hi⟩ := exists_pureMaximin_eq M
  rw [hi, ← lowerBound_single]
  exact (certificate_encloses_value M (single_mem_simplex i)
    (uniform_mem_simplex n)).1

/-- **DTH-PURE-2, CPP-CODE-PURE-3**: the pure minimax is an upper bound on the
mixed value. -/
theorem value_le_pureMinimax (M : Matrix m n ℝ) : value M ≤ pureMinimax M := by
  classical
  obtain ⟨j, hj⟩ := exists_pureMinimax_eq M
  rw [hj, ← upperBound_single]
  exact (certificate_encloses_value M (uniform_mem_simplex m)
    (single_mem_simplex j)).2

/-- **CRATES-PURE-1**: both the lower value (maximin over the simplex) and the
upper value (minimax over the simplex) lie in the pure enclosure. -/
theorem lowerValue_upperValue_mem (M : Matrix m n ℝ) :
    pureMaximin M ≤ lowerValue M ∧ lowerValue M ≤ pureMinimax M ∧
      pureMaximin M ≤ upperValue M ∧ upperValue M ≤ pureMinimax M := by
  have h1 := pureMaximin_le_value M
  have h2 := value_le_pureMinimax M
  have h3 := value_eq_upperValue M
  refine ⟨h1, h2, ?_, ?_⟩ <;> [rw [← h3]; rw [← h3]] <;> assumption

/-- **DTH-PURE-2, CPP-CODE-PURE-3, CRATES-PURE-1, CANONICAL-MG-5**
(`src/dth/complete_tablebase.py:1360-1368`, `src/dth/fast_kernel.c:84-85`,
`src/dth_cpp/recurrence.cpp:105-107`): when the pure gap is at most `ε`, the
stored midpoint is within `ε / 2` of the value. With `ε = 1e-6` the error is at
most `5e-7`. The accepted pair need not be an exact saddle; the rung stores a
`1e-6` bracket, not a pure equilibrium. -/
theorem pure_midpoint_error (M : Matrix m n ℝ) {ε : ℝ}
    (hgap : pureMinimax M - pureMaximin M ≤ ε) :
    |(pureMaximin M + pureMinimax M) / 2 - value M| ≤ ε / 2 := by
  have h1 := pureMaximin_le_value M
  have h2 := value_le_pureMinimax M
  rw [abs_le]
  constructor <;> linarith

end Value

/-! ## Running minima and maxima -/

section Prefix

variable {α : Type*} [LinearOrder α]

/-- The running minimum `min(s 0, …, s j)`: the C++ `prefix_min[j]` and
`np.minimum.accumulate`. -/
def prefixMin (s : ℕ → α) : ℕ → α
  | 0 => s 0
  | k + 1 => min (prefixMin s k) (s (k + 1))

/-- The running maximum `max(s 0, …, s j)`. -/
def prefixMax (s : ℕ → α) : ℕ → α
  | 0 => s 0
  | k + 1 => max (prefixMax s k) (s (k + 1))

theorem le_prefixMin_iff (s : ℕ → α) (x : α) (k : ℕ) :
    x ≤ prefixMin s k ↔ ∀ j ≤ k, x ≤ s j := by
  induction k with
  | zero =>
    constructor
    · intro h j hj; rw [Nat.le_zero.mp hj]; exact h
    · intro h; exact h 0 le_rfl
  | succ k ih =>
    rw [prefixMin, le_min_iff, ih]
    constructor
    · rintro ⟨h1, h2⟩ j hj
      rcases Nat.lt_or_eq_of_le hj with h | h
      · exact h1 j (by omega)
      · rw [h]; exact h2
    · intro h; exact ⟨fun j hj => h j (by omega), h _ le_rfl⟩

theorem prefixMax_le_iff (s : ℕ → α) (x : α) (k : ℕ) :
    prefixMax s k ≤ x ↔ ∀ j ≤ k, s j ≤ x := by
  induction k with
  | zero =>
    constructor
    · intro h j hj; rw [Nat.le_zero.mp hj]; exact h
    · intro h; exact h 0 le_rfl
  | succ k ih =>
    rw [prefixMax, max_le_iff, ih]
    constructor
    · rintro ⟨h1, h2⟩ j hj
      rcases Nat.lt_or_eq_of_le hj with h | h
      · exact h1 j (by omega)
      · rw [h]; exact h2
    · intro h; exact ⟨fun j hj => h j (by omega), h _ le_rfl⟩

theorem prefixMin_le (s : ℕ → α) {j k : ℕ} (h : j ≤ k) : prefixMin s k ≤ s j :=
  (le_prefixMin_iff s _ k).1 le_rfl j h

theorem le_prefixMax (s : ℕ → α) {j k : ℕ} (h : j ≤ k) : s j ≤ prefixMax s k :=
  (prefixMax_le_iff s _ k).1 le_rfl j h

theorem exists_prefixMin_eq (s : ℕ → α) (k : ℕ) : ∃ j ≤ k, prefixMin s k = s j := by
  induction k with
  | zero => exact ⟨0, le_rfl, rfl⟩
  | succ k ih =>
    obtain ⟨j, hj, hj'⟩ := ih
    rcases min_choice (prefixMin s k) (s (k + 1)) with h | h
    · exact ⟨j, by omega, by rw [prefixMin, h, hj']⟩
    · exact ⟨k + 1, le_rfl, by rw [prefixMin, h]⟩

theorem exists_prefixMax_eq (s : ℕ → α) (k : ℕ) : ∃ j ≤ k, prefixMax s k = s j := by
  induction k with
  | zero => exact ⟨0, le_rfl, rfl⟩
  | succ k ih =>
    obtain ⟨j, hj, hj'⟩ := ih
    rcases max_choice (prefixMax s k) (s (k + 1)) with h | h
    · exact ⟨j, by omega, by rw [prefixMax, h, hj']⟩
    · exact ⟨k + 1, le_rfl, by rw [prefixMax, h]⟩

/-- Longer prefixes have smaller minima. -/
theorem prefixMin_antitone (s : ℕ → α) {k k' : ℕ} (h : k ≤ k') :
    prefixMin s k' ≤ prefixMin s k :=
  (le_prefixMin_iff s _ k).2 fun _ hj => prefixMin_le s (hj.trans h)

/-- Longer prefixes have larger maxima. -/
theorem prefixMax_monotone (s : ℕ → α) {k k' : ℕ} (h : k ≤ k') :
    prefixMax s k ≤ prefixMax s k' :=
  (prefixMax_le_iff s _ k).2 fun _ hj => le_prefixMax s (hj.trans h)

/-- `prefixMin s n` is `lo = min_k s[k]` over the `n + 1` success payoffs. -/
theorem prefixMin_eq_inf' (s : ℕ → α) (n : ℕ) :
    prefixMin s n = univ.inf' univ_nonempty fun k : Fin (n + 1) => s k := by
  apply le_antisymm
  · exact le_inf' _ _ fun k _ => prefixMin_le s (by omega)
  · obtain ⟨j, hj, hj'⟩ := exists_prefixMin_eq s n
    rw [hj']
    exact inf'_le (fun k : Fin (n + 1) => s k) (mem_univ ⟨j, by omega⟩)

/-- `prefixMax s n` is `hi = max_k s[k]` over the `n + 1` success payoffs. -/
theorem prefixMax_eq_sup' (s : ℕ → α) (n : ℕ) :
    prefixMax s n = univ.sup' univ_nonempty fun k : Fin (n + 1) => s k := by
  apply le_antisymm
  · obtain ⟨j, hj, hj'⟩ := exists_prefixMax_eq s n
    rw [hj']
    exact le_sup' (fun k : Fin (n + 1) => s k) (mem_univ ⟨j, by omega⟩)
  · exact sup'_le _ _ fun k _ => le_prefixMax s (by omega)

end Prefix

/-! ## Row minima and column maxima of the Toeplitz stage -/

section ToeplitzBounds

variable {α : Type*} [LinearOrder α]

/-- The stage matrix over a linear order: `M[d, c] = s (c - d)` for `d ≤ c`,
else `f`. -/
def toeplitzL (N : ℕ) (s : ℕ → α) (f : α) : Matrix (Fin N) (Fin N) α :=
  fun d c => if d ≤ c then s (c.val - d.val) else f

theorem toeplitz_eq_toeplitzL (N : ℕ) (s : ℕ → ℝ) (f : ℝ) :
    toeplitz N s f = toeplitzL N s f := rfl

/-- The C++/NumPy row-minimum formula for a stage with `n + 1` actions:
`prefix_min[n - d]`, combined with `f` when `d > 0`. -/
def rowMinFormula (n : ℕ) (s : ℕ → α) (f : α) (d : ℕ) : α :=
  if d = 0 then prefixMin s (n - d) else min (prefixMin s (n - d)) f

/-- The C++/NumPy column-maximum formula: `prefix_max[c]`, combined with `f`
when `c + 1 < n + 1`. -/
def colMaxFormula (n : ℕ) (s : ℕ → α) (f : α) (c : ℕ) : α :=
  if c + 1 < n + 1 then max (prefixMax s c) f else prefixMax s c

variable {n : ℕ} (s : ℕ → α) (f : α)

theorem toeplitzL_of_le {d c : Fin (n + 1)} (h : d.val ≤ c.val) :
    toeplitzL (n + 1) s f d c = s (c.val - d.val) := by
  have h' : d ≤ c := Fin.le_def.mpr h
  simp only [toeplitzL, h', ↓reduceIte]

theorem toeplitzL_of_lt {d c : Fin (n + 1)} (h : c.val < d.val) :
    toeplitzL (n + 1) s f d c = f := by
  have h' : ¬ d ≤ c := fun h' => absurd (Fin.le_def.mp h') (not_le.mpr h)
  simp only [toeplitzL, h', ↓reduceIte]

/-- **CPP-DOC-PURE-1, CPP-CODE-PURE-1, DTH-PURE-1, CANONICAL-MG-5** (row part;
`src/dth_cpp/matrix_game.cpp:139-163`, `src/dth/complete_tablebase.py:386-401`):
row `d` of the stage reads `s[0..n-d]` and, when `d > 0`, `f`; its minimum is
`prefix_min[n - d]`, combined with `f` when `d > 0`. With `n = 59` this is
`prefix_min[59 - d]`. -/
theorem rowMin_toeplitzL (d : Fin (n + 1)) :
    rowMin (toeplitzL (n + 1) s f) d = rowMinFormula n s f d := by
  have hd := d.isLt
  have hle : rowMin (toeplitzL (n + 1) s f) d ≤ prefixMin s (n - d) := by
    obtain ⟨j, hj, hpj⟩ := exists_prefixMin_eq s (n - d)
    rw [hpj]
    have h := rowMin_le (toeplitzL (n + 1) s f) d ⟨d.val + j, by omega⟩
    rwa [toeplitzL_of_le s f (by simp), show d.val + j - d.val = j by omega] at h
  apply le_antisymm
  · unfold rowMinFormula
    split_ifs with h0
    · exact hle
    · refine le_min hle ?_
      have h := rowMin_le (toeplitzL (n + 1) s f) d ⟨0, by omega⟩
      rwa [toeplitzL_of_lt s f (show 0 < d.val by omega)] at h
  · refine le_inf' _ _ fun c _ => ?_
    unfold rowMinFormula
    rcases le_or_gt d.val c.val with h | h
    · rw [toeplitzL_of_le s f h]
      have h' : prefixMin s (n - d) ≤ s (c.val - d.val) := prefixMin_le s (by omega)
      split_ifs
      · exact h'
      · exact (min_le_left _ _).trans h'
    · rw [toeplitzL_of_lt s f h]
      split_ifs with h0
      · omega
      · exact min_le_right _ _

/-- **CPP-DOC-PURE-1, CPP-CODE-PURE-1, DTH-PURE-1, CANONICAL-MG-5** (column
part; `src/dth_cpp/matrix_game.cpp:166-175`): column `c` reads `s[0..c]` and,
when `c + 1 < n + 1`, `f`; its maximum is `prefix_max[c]`, combined with `f`
when `c + 1 < n + 1`. -/
theorem colMax_toeplitzL (c : Fin (n + 1)) :
    colMax (toeplitzL (n + 1) s f) c = colMaxFormula n s f c := by
  have hc := c.isLt
  have hge : prefixMax s c ≤ colMax (toeplitzL (n + 1) s f) c := by
    obtain ⟨j, hj, hpj⟩ := exists_prefixMax_eq s c
    rw [hpj]
    have h := le_colMax (toeplitzL (n + 1) s f) ⟨c.val - j, by omega⟩ c
    rwa [toeplitzL_of_le s f (by simp), show c.val - (c.val - j) = j by omega] at h
  apply le_antisymm
  · refine sup'_le _ _ fun d _ => ?_
    unfold colMaxFormula
    rcases le_or_gt d.val c.val with h | h
    · rw [toeplitzL_of_le s f h]
      have h' : s (c.val - d.val) ≤ prefixMax s c := le_prefixMax s (by omega)
      split_ifs
      · exact h'.trans (le_max_left _ _)
      · exact h'
    · rw [toeplitzL_of_lt s f h]
      split_ifs with h0
      · exact le_max_right _ _
      · omega
  · unfold colMaxFormula
    split_ifs with h0
    · refine max_le hge ?_
      have h := le_colMax (toeplitzL (n + 1) s f) ⟨n, by omega⟩ c
      rwa [toeplitzL_of_lt s f (show c.val < n by omega)] at h
    · exact hge

/-- Row `0` attains `lo = min_k s[k]`. -/
theorem rowMin_zero : rowMin (toeplitzL (n + 1) s f) 0 = prefixMin s n := by
  rw [rowMin_toeplitzL]; simp [rowMinFormula]

/-- Every row `d ≥ 1` has minimum at most `min(f, s0)`. -/
theorem rowMin_le_of_ne_zero {d : Fin (n + 1)} (hd : d.val ≠ 0) :
    rowMin (toeplitzL (n + 1) s f) d ≤ min f (s 0) := by
  rw [rowMin_toeplitzL, rowMinFormula]
  simp only [hd, ↓reduceIte]
  rw [min_comm f]
  exact min_le_min (prefixMin_le s (Nat.zero_le _)) le_rfl

/-- The last row (`d = n`, the doc's row 60) attains `min(f, s0)`. -/
theorem rowMin_last : rowMin (toeplitzL (n + 2) s f) (Fin.last (n + 1)) = min f (s 0) := by
  rw [rowMin_toeplitzL]; simp [rowMinFormula, prefixMin, min_comm]

/-- The last column (`c = n`, the doc's column 60) attains `hi = max_k s[k]`. -/
theorem colMax_last : colMax (toeplitzL (n + 1) s f) (Fin.last n) = prefixMax s n := by
  rw [colMax_toeplitzL]; simp [colMaxFormula]

/-- Every column `c < n` has maximum at least `max(f, s0)`. -/
theorem le_colMax_of_lt {c : Fin (n + 1)} (hc : c.val < n) :
    max f (s 0) ≤ colMax (toeplitzL (n + 1) s f) c := by
  have hc' : c.val + 1 < n + 1 := by omega
  rw [colMax_toeplitzL, colMaxFormula]
  simp only [hc', ↓reduceIte]
  rw [max_comm f]
  exact max_le_max (le_prefixMax s (Nat.zero_le _)) le_rfl

/-- Column `0` attains `max(f, s0)` when the stage has at least two actions. -/
theorem colMax_zero : colMax (toeplitzL (n + 2) s f) 0 = max f (s 0) := by
  rw [colMax_toeplitzL]; simp [colMaxFormula, prefixMax, max_comm]

/-- **COMPACT-RUNG1-1, CPP-DOC-PURE-3, CPP-CODE-PURE-2, DTH-PURE-1,
CRATES-PURE-1, CANONICAL-MG-5, STL-MG-3** (maximin; `src/dth_compact/main.py:226-237`,
`src/dth/fast_kernel.c:44`, `src/dth_cpp/recurrence.cpp:52`,
`src/crates/stl_solver/src/leap.rs:383`): the pure maximin of the stage is
`max(lo, min(f, s0))` with `lo = prefix_min[n] = min_k s[k]`. The identity
holds in every linear order, so the `O(60)` closed form and the 3,600-cell
brute force select the same element. -/
theorem pureMaximin_toeplitzL :
    pureMaximin (toeplitzL (n + 1) s f) = max (prefixMin s n) (min f (s 0)) := by
  apply le_antisymm
  · refine sup'_le _ _ fun d _ => ?_
    by_cases hd : d.val = 0
    · have : d = 0 := Fin.ext hd
      rw [this, rowMin_zero]; exact le_max_left _ _
    · exact (rowMin_le_of_ne_zero s f hd).trans (le_max_right _ _)
  · refine max_le ?_ ?_
    · rw [← rowMin_zero s f]; exact rowMin_le_pureMaximin _ _
    · rcases n with _ | n
      · have h := rowMin_le_pureMaximin (toeplitzL (0 + 1) s f) 0
        rw [rowMin_zero] at h
        exact (min_le_right _ _).trans h
      · rw [← rowMin_last s f]; exact rowMin_le_pureMaximin _ _

/-- **COMPACT-RUNG1-1, CPP-DOC-PURE-3, CPP-CODE-PURE-2, DTH-PURE-1,
CRATES-PURE-1, CANONICAL-MG-5, STL-MG-3** (minimax; `src/dth/fast_kernel.c:45`,
`src/dth_cpp/recurrence.cpp:53`, `src/crates/stl_solver/src/leap.rs:384`): the
pure minimax of the stage is `min(hi, max(f, s0))` with
`hi = prefix_max[n] = max_k s[k]`. -/
theorem pureMinimax_toeplitzL :
    pureMinimax (toeplitzL (n + 1) s f) = min (prefixMax s n) (max f (s 0)) := by
  apply le_antisymm
  · refine le_min ?_ ?_
    · rw [← colMax_last s f]; exact pureMinimax_le_colMax _ _
    · rcases n with _ | n
      · have h := pureMinimax_le_colMax (toeplitzL (0 + 1) s f) (Fin.last 0)
        rw [colMax_last] at h
        exact h.trans (le_max_right _ _)
      · rw [← colMax_zero s f]; exact pureMinimax_le_colMax _ _
  · refine le_inf' _ _ fun c _ => ?_
    by_cases hc : c.val < n
    · exact (min_le_right _ _).trans (le_colMax_of_lt s f hc)
    · have : c = Fin.last n := Fin.ext (by simp; omega)
      rw [this, colMax_last]; exact min_le_left _ _

/-- **COMPACT-RUNG1-1** (`src/dth_compact/main.py:231-237`, `try_rung1`): the
certifying Dropper row is `0` when `lo ≥ min(f, s0)` and the last row
otherwise; its row minimum is the pure maximin. -/
theorem rowMin_rung1_drop :
    rowMin (toeplitzL (n + 1) s f)
        (if min f (s 0) ≤ prefixMin s n then 0 else Fin.last n) =
      pureMaximin (toeplitzL (n + 1) s f) := by
  rw [pureMaximin_toeplitzL]
  split_ifs with h
  · rw [rowMin_zero, max_eq_left h]
  · rw [max_eq_right (le_of_not_ge h)]
    rcases n with _ | n
    · exact absurd ((min_le_right _ _).trans (le_of_eq (rfl : prefixMin s 0 = s 0).symm)) h
    · exact rowMin_last s f

/-- **COMPACT-RUNG1-1** (`try_rung1`): the certifying Checker column is the
last column when `hi ≤ max(f, s0)` and column `0` otherwise; its column
maximum is the pure minimax. -/
theorem colMax_rung1_check :
    colMax (toeplitzL (n + 1) s f)
        (if prefixMax s n ≤ max f (s 0) then Fin.last n else 0) =
      pureMinimax (toeplitzL (n + 1) s f) := by
  rw [pureMinimax_toeplitzL]
  split_ifs with h
  · rw [colMax_last, min_eq_left h]
  · rw [min_eq_right (le_of_not_ge h)]
    rcases n with _ | n
    · exact absurd ((le_of_eq (rfl : prefixMax s 0 = s 0)).trans (le_max_right _ _)) h
    · exact colMax_zero s f

end ToeplitzBounds

/-! ## The strict-comparison scan of `scan_pure_saddle` -/

section Scan

variable {α : Type*} [LinearOrder α]

/-- The C++ maximin loop (`src/dth_cpp/matrix_game.cpp:152-163`): keep the
running best `(value, index)` and replace it only when `candidate > best`.
The C++ loop starts from `-∞`; its first finite candidate always replaces
that, so the model starts from `(g 0, 0)`. -/
def scanMax (g : ℕ → α) : ℕ → α × ℕ
  | 0 => (g 0, 0)
  | k + 1 => if (scanMax g k).1 < g (k + 1) then (g (k + 1), k + 1) else scanMax g k

/-- The C++ minimax loop (`src/dth_cpp/matrix_game.cpp:165-175`): replace the
running best only when `candidate < best`. -/
def scanMin (g : ℕ → α) : ℕ → α × ℕ
  | 0 => (g 0, 0)
  | k + 1 => if g (k + 1) < (scanMin g k).1 then (g (k + 1), k + 1) else scanMin g k

/-- The strict scan returns the maximum of `g 0, …, g k` and the lowest index
that attains it. -/
theorem scanMax_spec (g : ℕ → α) (k : ℕ) :
    (scanMax g k).2 ≤ k ∧ (scanMax g k).1 = g (scanMax g k).2 ∧
      (∀ j ≤ k, g j ≤ (scanMax g k).1) ∧ ∀ j < (scanMax g k).2, g j < (scanMax g k).1 := by
  induction k with
  | zero =>
    change (0 : ℕ) ≤ 0 ∧ g 0 = g 0 ∧ (∀ j ≤ 0, g j ≤ g 0) ∧ ∀ j < 0, g j < g 0
    refine ⟨le_rfl, rfl, fun j hj => ?_, fun j hj => absurd hj (Nat.not_lt_zero j)⟩
    rw [Nat.le_zero.mp hj]
  | succ k ih =>
    obtain ⟨h1, h2, h3, h4⟩ := ih
    by_cases h : (scanMax g k).1 < g (k + 1)
    · rw [show scanMax g (k + 1) = (g (k + 1), k + 1) by simp only [scanMax, h, ↓reduceIte]]
      refine ⟨le_rfl, rfl, fun j hj => ?_, fun j hj => (h3 j (by omega)).trans_lt h⟩
      rcases Nat.lt_or_eq_of_le hj with hj | hj
      · exact (h3 j (by omega)).trans h.le
      · rw [hj]
    · rw [show scanMax g (k + 1) = scanMax g k by simp only [scanMax, h, ↓reduceIte]]
      refine ⟨by omega, h2, fun j hj => ?_, h4⟩
      rcases Nat.lt_or_eq_of_le hj with hj | hj
      · exact h3 j (by omega)
      · rw [hj]; exact le_of_not_gt h

/-- The strict scan returns the minimum of `g 0, …, g k` and the lowest index
that attains it. -/
theorem scanMin_spec (g : ℕ → α) (k : ℕ) :
    (scanMin g k).2 ≤ k ∧ (scanMin g k).1 = g (scanMin g k).2 ∧
      (∀ j ≤ k, (scanMin g k).1 ≤ g j) ∧ ∀ j < (scanMin g k).2, (scanMin g k).1 < g j := by
  induction k with
  | zero =>
    change (0 : ℕ) ≤ 0 ∧ g 0 = g 0 ∧ (∀ j ≤ 0, g 0 ≤ g j) ∧ ∀ j < 0, g 0 < g j
    refine ⟨le_rfl, rfl, fun j hj => ?_, fun j hj => absurd hj (Nat.not_lt_zero j)⟩
    rw [Nat.le_zero.mp hj]
  | succ k ih =>
    obtain ⟨h1, h2, h3, h4⟩ := ih
    by_cases h : g (k + 1) < (scanMin g k).1
    · rw [show scanMin g (k + 1) = (g (k + 1), k + 1) by simp only [scanMin, h, ↓reduceIte]]
      refine ⟨le_rfl, rfl, fun j hj => ?_, fun j hj => h.trans_le (h3 j (by omega))⟩
      rcases Nat.lt_or_eq_of_le hj with hj | hj
      · exact h.le.trans (h3 j (by omega))
      · rw [hj]
    · rw [show scanMin g (k + 1) = scanMin g k by simp only [scanMin, h, ↓reduceIte]]
      refine ⟨by omega, h2, fun j hj => ?_, h4⟩
      rcases Nat.lt_or_eq_of_le hj with hj | hj
      · exact h3 j (by omega)
      · rw [hj]; exact le_of_not_gt h

variable {n : ℕ} (s : ℕ → α) (f : α)

/-- **CPP-CODE-PURE-1, CPP-DOC-PURE-2** (`src/dth_cpp/matrix_game.cpp:152-163`):
the scan over `row_min[d]` returns the pure maximin, and `best_drop` is the
lowest row whose minimum attains it. -/
theorem cpp_scan_drop :
    let r := scanMax (rowMinFormula n s f) n
    ∃ h : r.2 < n + 1,
      r.1 = pureMaximin (toeplitzL (n + 1) s f) ∧
      rowMin (toeplitzL (n + 1) s f) ⟨r.2, h⟩ = pureMaximin (toeplitzL (n + 1) s f) ∧
      ∀ d : Fin (n + 1), d.val < r.2 →
        rowMin (toeplitzL (n + 1) s f) d < pureMaximin (toeplitzL (n + 1) s f) := by
  intro r
  obtain ⟨h1, h2, h3, h4⟩ : r.2 ≤ n ∧ r.1 = rowMinFormula n s f r.2 ∧
      (∀ j ≤ n, rowMinFormula n s f j ≤ r.1) ∧ ∀ j < r.2, rowMinFormula n s f j < r.1 :=
    scanMax_spec (rowMinFormula n s f) n
  have hb : r.2 < n + 1 := by omega
  have hr : rowMin (toeplitzL (n + 1) s f) ⟨r.2, hb⟩ = r.1 := by
    rw [rowMin_toeplitzL]; exact h2.symm
  have hval : r.1 = pureMaximin (toeplitzL (n + 1) s f) := by
    apply le_antisymm
    · rw [← hr]; exact rowMin_le_pureMaximin _ _
    · refine sup'_le _ _ fun d _ => ?_
      rw [rowMin_toeplitzL]; exact h3 d (by omega)
  refine ⟨hb, hval, hr.trans hval, fun d hd => ?_⟩
  rw [rowMin_toeplitzL, ← hval]; exact h4 d hd

/-- **CPP-CODE-PURE-1, CPP-DOC-PURE-2** (`src/dth_cpp/matrix_game.cpp:165-175`):
the scan over `col_max[c]` returns the pure minimax, and `best_check` is the
lowest column whose maximum attains it. -/
theorem cpp_scan_check :
    let r := scanMin (colMaxFormula n s f) n
    ∃ h : r.2 < n + 1,
      r.1 = pureMinimax (toeplitzL (n + 1) s f) ∧
      colMax (toeplitzL (n + 1) s f) ⟨r.2, h⟩ = pureMinimax (toeplitzL (n + 1) s f) ∧
      ∀ c : Fin (n + 1), c.val < r.2 →
        pureMinimax (toeplitzL (n + 1) s f) < colMax (toeplitzL (n + 1) s f) c := by
  intro r
  obtain ⟨h1, h2, h3, h4⟩ : r.2 ≤ n ∧ r.1 = colMaxFormula n s f r.2 ∧
      (∀ j ≤ n, r.1 ≤ colMaxFormula n s f j) ∧ ∀ j < r.2, r.1 < colMaxFormula n s f j :=
    scanMin_spec (colMaxFormula n s f) n
  have hb : r.2 < n + 1 := by omega
  have hr : colMax (toeplitzL (n + 1) s f) ⟨r.2, hb⟩ = r.1 := by
    rw [colMax_toeplitzL]; exact h2.symm
  have hval : r.1 = pureMinimax (toeplitzL (n + 1) s f) := by
    apply le_antisymm
    · refine le_inf' _ _ fun c _ => ?_
      rw [colMax_toeplitzL]; exact h3 c (by omega)
    · rw [← hr]; exact pureMinimax_le_colMax _ _
  refine ⟨hb, hval, hr.trans hval, fun c hc => ?_⟩
  rw [colMax_toeplitzL, ← hval]; exact h4 c hc

end Scan

/-! ## Floating one-hot certificates -/

section Float

/-- A left-to-right floating dot product, `acc ← fl(acc + fl(p_i * x_i))`,
the loop of `certify` (`src/dth_cpp/matrix_game.cpp:108-124`). Rounding is an
abstract map `fl : ℝ → ℝ`. -/
def fdot {ι : Type*} (fl : ℝ → ℝ) (l : List ι) (p x : ι → ℝ) : ℝ :=
  l.foldl (fun acc i => fl (acc + fl (p i * x i))) 0

variable {ι : Type*} [DecidableEq ι] (fl : ℝ → ℝ) (F : Set ℝ)

private theorem foldl_single_absent (hfl : ∀ y ∈ F, fl y = y) (h0 : (0 : ℝ) ∈ F)
    (x : ι → ℝ) (i : ι) :
    ∀ (l : List ι), i ∉ l → ∀ acc ∈ F,
      l.foldl (fun acc j => fl (acc + fl ((Pi.single i 1 : ι → ℝ) j * x j))) acc = acc := by
  intro l
  induction l with
  | nil => intro _ acc _; rfl
  | cons j l ih =>
    intro hi acc hacc
    have hj : j ≠ i := fun h => hi (h ▸ List.mem_cons_self)
    have hi' : i ∉ l := fun h => hi (List.mem_cons_of_mem j h)
    simp only [List.foldl_cons, Pi.single_eq_of_ne hj, zero_mul, hfl 0 h0, add_zero,
      hfl acc hacc]
    exact ih hi' acc hacc

/-- **CPP-CODE-PURE-3** (doubt on exactness of `0 * x`): if rounding fixes
every representable number (`fl y = y` for `y ∈ F`, as round-to-nearest does),
`0 ∈ F`, and every entry is representable, then the floating dot product of a
one-hot vector with `x` is `x i` exactly. The model works over `ℝ`, so it does
not distinguish `+0` from `-0`. -/
theorem fdot_single (hfl : ∀ y ∈ F, fl y = y) (h0 : (0 : ℝ) ∈ F) (x : ι → ℝ)
    (hx : ∀ j, x j ∈ F) (i : ι) (l : List ι) (hl : l.Nodup) (hi : i ∈ l) :
    fdot fl l (Pi.single i 1) x = x i := by
  unfold fdot
  induction l with
  | nil => exact absurd hi List.not_mem_nil
  | cons j l ih =>
    rw [List.nodup_cons] at hl
    by_cases hj : j = i
    · subst hj
      simp only [List.foldl_cons, Pi.single_eq_same, one_mul, hfl _ (hx j), zero_add]
      exact foldl_single_absent fl F hfl h0 x j l hl.1 (x j) (hx j)
    · have hi' : i ∈ l := by
        rcases List.mem_cons.mp hi with h | h
        · exact absurd h.symm hj
        · exact h
      simp only [List.foldl_cons, Pi.single_eq_of_ne hj, zero_mul, hfl 0 h0, add_zero]
      exact ih hl.2 hi'

/-- **CPP-CODE-PURE-3, CPP-DOC-PURE-2** (`src/dth_cpp/matrix_game.cpp:186-199`,
`certify` at lines 108-124): with one-hot policies on `(d, c)`, the computed
column payoffs of `certify` are the entries of row `d` and its computed row
payoffs are the entries of column `c`. Their minimum and maximum are therefore
`rowMin M d` and `colMax M c` exactly, the same numbers the scan compared, so
the certificate never rejects a pair the scan accepted. The model takes the
policy normalization of a one-hot vector as exact, which holds because
`0 + … + 1 + … + 0 = 1` and `x / 1 = x` are exact. -/
theorem cpp_certify_onehot {N : ℕ} [NeZero N] (hfl : ∀ y ∈ F, fl y = y)
    (h0 : (0 : ℝ) ∈ F) (M : Matrix (Fin N) (Fin N) ℝ) (hM : ∀ i j, M i j ∈ F)
    (d c : Fin N) :
    univ.inf' univ_nonempty
        (fun c' => fdot fl (List.finRange N) (Pi.single d 1) fun d' => M d' c') =
      rowMin M d ∧
    univ.sup' univ_nonempty
        (fun d' => fdot fl (List.finRange N) (Pi.single c 1) fun c' => M d' c') =
      colMax M c := by
  constructor
  · unfold rowMin; congr 1; funext c'
    exact fdot_single fl F hfl h0 _ (fun j => hM j c') d _ (List.nodup_finRange N)
      (List.mem_finRange d)
  · unfold colMax; congr 1; funext d'
    exact fdot_single fl F hfl h0 _ (fun j => hM d' j) c _ (List.nodup_finRange N)
      (List.mem_finRange c)

/-- **COMPACT-RUNG1-2** (`src/dth_compact/architecture.md:167`): under the
relative rounding model `fl(x) = x (1 + δ) + η` with `|δ| ≤ u < 1` for the gap
subtraction (`η = 0`, no underflow in a subtraction of two doubles) and
`|η| ≤ η₀` for the midpoint, a stage that passes the computed gate
`fl(mx - mn) ≤ ε` stores a value within `ε / (2 (1 - u)) + u |mid| + η₀` of
the exact matrix value. The bounds `mn`, `mx` themselves carry no rounding
(`pureMaximin_mem_entries`); the model rounds only the gap and the midpoint. -/
theorem rounded_pure_rung_error {m n : Type*} [Fintype m] [Fintype n] [Nonempty m]
    [Nonempty n] (M : Matrix m n ℝ) {ε u δ₁ δ₂ η η₀ : ℝ} (hu : u < 1)
    (hδ₁ : |δ₁| ≤ u) (hδ₂ : |δ₂| ≤ u) (hη : |η| ≤ η₀)
    (hgate : (pureMinimax M - pureMaximin M) * (1 + δ₁) ≤ ε) :
    |(pureMaximin M + pureMinimax M) / 2 * (1 + δ₂) + η - value M| ≤
      ε / (2 * (1 - u)) + u * |(pureMaximin M + pureMinimax M) / 2| + η₀ := by
  set y := pureMinimax M - pureMaximin M with hy
  have hy0 : 0 ≤ y := sub_nonneg.mpr (pureMaximin_le_pureMinimax M)
  have hδ₁' : -u ≤ δ₁ := (abs_le.mp hδ₁).1
  have h1u : 0 < 1 - u := by linarith
  have hyε : y ≤ ε / (1 - u) := by
    rw [le_div_iff₀ h1u]
    nlinarith
  have hmid := pure_midpoint_error M (le_of_eq (rfl : y = y) |>.trans (le_refl y))
  have hmid' : |(pureMaximin M + pureMinimax M) / 2 - value M| ≤ ε / (2 * (1 - u)) := by
    refine hmid.trans ?_
    rw [show ε / (2 * (1 - u)) = ε / (1 - u) / 2 by field_simp]
    linarith
  have hround : |(pureMaximin M + pureMinimax M) / 2 * (1 + δ₂) + η -
      (pureMaximin M + pureMinimax M) / 2| ≤ u * |(pureMaximin M + pureMinimax M) / 2| + η₀ := by
    have : (pureMaximin M + pureMinimax M) / 2 * (1 + δ₂) + η -
        (pureMaximin M + pureMinimax M) / 2 = (pureMaximin M + pureMinimax M) / 2 * δ₂ + η := by
      ring
    rw [this]
    refine (abs_add_le _ _).trans ?_
    rw [abs_mul]
    have := mul_le_mul_of_nonneg_left hδ₂ (abs_nonneg ((pureMaximin M + pureMinimax M) / 2))
    linarith
  calc _ = |((pureMaximin M + pureMinimax M) / 2 * (1 + δ₂) + η -
        (pureMaximin M + pureMinimax M) / 2) +
        ((pureMaximin M + pureMinimax M) / 2 - value M)| := by ring_nf
    _ ≤ _ := abs_add_le _ _
    _ ≤ _ := by linarith

end Float

/-! ## Order independence of min and max folds -/

section Order

variable {α : Type*} [LinearOrder α]

/-- **DTH-PURE-2, CANONICAL-MG-5** (`src/dth/complete_tablebase.py:398-399`,
"order-independent"): a left fold of `max` does not depend on the order of its
list. A NumPy reduction, a C loop, and a Rust `fold` over any permutation of
the same finite entries therefore return the same element. -/
theorem foldl_max_perm {l l' : List α} (h : l.Perm l') (a : α) :
    l.foldl max a = l'.foldl max a :=
  haveI : RightCommutative (max : α → α → α) :=
    ⟨fun a b c => by rw [max_assoc, max_comm b c, ← max_assoc]⟩
  h.foldl_eq a

/-- A left fold of `min` does not depend on the order of its list. -/
theorem foldl_min_perm {l l' : List α} (h : l.Perm l') (a : α) :
    l.foldl min a = l'.foldl min a :=
  haveI : RightCommutative (min : α → α → α) :=
    ⟨fun a b c => by rw [min_assoc, min_comm b c, ← min_assoc]⟩
  h.foldl_eq a

end Order

/-! ## The abstract solver's `1e-12` near-saddle search -/

section Abstract

variable {m n : ℕ} [NeZero m] [NeZero n]

/-- `|a - b| ≤ τ`: Rust `close` (`src/crates/abstract_solver/src/lib.rs:164-166`)
and NumPy `isclose(a, b, atol=τ, rtol=0)`. The model evaluates the
subtraction and the absolute value exactly. -/
def Close (τ a b : ℝ) : Prop := |a - b| ≤ τ

/-- The three tests of the inner loop at cell `(r, c)`: the row minimum is
close to `lower`, the column maximum is close to `upper`, and the cell is
close to `lower` (`src/crates/abstract_solver/src/lib.rs:179-190`,
`src/abstract/matrix.py:139-141`). -/
def NearCell (M : Matrix (Fin m) (Fin n) ℝ) (τ : ℝ) (rc : Fin m × Fin n) : Prop :=
  Close τ (rowMin M rc.1) (pureMaximin M) ∧ Close τ (colMax M rc.2) (pureMinimax M) ∧
    Close τ (M rc.1 rc.2) (pureMaximin M)

/-- All cells in row-major order: the order of both nested loops. -/
def cells (m n : ℕ) : List (Fin m × Fin n) :=
  (List.finRange m).flatMap fun r => (List.finRange n).map fun c => (r, c)

theorem mem_cells (rc : Fin m × Fin n) : rc ∈ cells m n := by
  simp [cells, List.mem_flatMap, List.mem_map, List.mem_finRange]

open Classical in
/-- Rust `pure_saddle` (`src/crates/abstract_solver/src/lib.rs:168-195`): reject
unless `lower` and `upper` are close, then return the first cell in row-major
order that passes the three tests. The Rust row-minimum and column-maximum
loops and the folds that form `lower` and `upper` compute `rowMin`, `colMax`,
`pureMaximin`, and `pureMinimax` (min and max select entries; `foldl_max_perm`). -/
noncomputable def rustPureSaddle (M : Matrix (Fin m) (Fin n) ℝ) (τ : ℝ) :
    Option (Fin m × Fin n) :=
  if Close τ (pureMaximin M) (pureMinimax M) then
    (cells m n).find? fun rc => decide (NearCell M τ rc)
  else none

/-- Python's `saddle_gap` for one-hot policies on `(r, c)`
(`src/abstract/matrix.py:112-125`): `expected = M[r, c]`,
`row_gain = max(0, colMax c - expected)`, and
`column_gain = max(0, expected - rowMin r)`. -/
noncomputable def pythonGap (M : Matrix (Fin m) (Fin n) ℝ) (r : Fin m) (c : Fin n) : ℝ :=
  max 0 (colMax M c - M r c) + max 0 (M r c - rowMin M r)

open Classical in
/-- Python's pure branch of `solve_matrix` (`src/abstract/matrix.py:134-156`).
`np.flatnonzero` lists indices in increasing order, so the double loop visits
the cells that pass the row and column filters in row-major order; a cell is
returned when it also passes the cell test and its gap is at most `g`
(`g = 2e-7`). -/
noncomputable def pythonPureSaddle (M : Matrix (Fin m) (Fin n) ℝ) (τ g : ℝ) :
    Option (Fin m × Fin n) :=
  if Close τ (pureMaximin M) (pureMinimax M) then
    (cells m n).find? fun rc => decide (NearCell M τ rc ∧ pythonGap M rc.1 rc.2 ≤ g)
  else none

/-- The Python gap of a one-hot pair is `colMax c - rowMin r`, the gap of the
certificate `(e_r, e_c)` in `Formal.MatrixGame`. -/
theorem pythonGap_eq (M : Matrix (Fin m) (Fin n) ℝ) (r : Fin m) (c : Fin n) :
    pythonGap M r c = colMax M c - rowMin M r ∧
      pythonGap M r c = upperBound M (Pi.single c 1) - lowerBound M (Pi.single r 1) := by
  have h1 := rowMin_le M r c
  have h2 := le_colMax M r c
  have h : pythonGap M r c = colMax M c - rowMin M r := by
    unfold pythonGap
    rw [max_eq_right (by linarith), max_eq_right (by linarith)]
    ring
  exact ⟨h, by rw [h, upperBound_single, lowerBound_single]⟩

/-- **ABSTRACT-MAT-2 (b), CRATES-ABS-5** (`src/abstract/matrix.py:138-141`,
`src/crates/abstract_solver/src/lib.rs:175-190`): when `lower` and `upper`
are close, the argmax row `i*` of the row minima and the argmin column `j*` of
the column maxima pass all three tests, with `rowMin i* = lower`,
`colMax j* = upper`, and `lower ≤ M[i*, j*] ≤ upper`. -/
theorem nearCell_of_close (M : Matrix (Fin m) (Fin n) ℝ) {τ : ℝ}
    (h : Close τ (pureMaximin M) (pureMinimax M)) :
    ∃ i j, rowMin M i = pureMaximin M ∧ colMax M j = pureMinimax M ∧
      pureMaximin M ≤ M i j ∧ M i j ≤ pureMinimax M ∧ NearCell M τ (i, j) := by
  obtain ⟨i, hi⟩ := exists_pureMaximin_eq M
  obtain ⟨j, hj⟩ := exists_pureMinimax_eq M
  have hτ : 0 ≤ τ := (abs_nonneg _).trans h
  have hle := pureMaximin_le_pureMinimax M
  have h1 : pureMaximin M ≤ M i j := hi ▸ rowMin_le M i j
  have h2 : M i j ≤ pureMinimax M := hj ▸ le_colMax M i j
  refine ⟨i, j, hi.symm, hj.symm, h1, h2, ?_, ?_, ?_⟩
  · show |rowMin M i - pureMaximin M| ≤ τ
    rw [← hi, sub_self, abs_zero]; exact hτ
  · show |colMax M j - pureMinimax M| ≤ τ
    rw [← hj, sub_self, abs_zero]; exact hτ
  · show |M i j - pureMaximin M| ≤ τ
    unfold Close at h
    rw [abs_sub_comm] at h
    rw [abs_of_nonneg (by linarith)]
    rw [abs_of_nonneg (by linarith)] at h
    linarith

/-- **ABSTRACT-MAT-2 (b), CRATES-ABS-5**: the Rust search returns a cell if and
only if `lower` and `upper` are close. It never returns `None` after the first
test passes. -/
theorem rustPureSaddle_isSome_iff (M : Matrix (Fin m) (Fin n) ℝ) (τ : ℝ) :
    (rustPureSaddle M τ).isSome ↔ Close τ (pureMaximin M) (pureMinimax M) := by
  unfold rustPureSaddle
  split_ifs with h
  · simp only [h, iff_true, List.find?_isSome, decide_eq_true_eq]
    obtain ⟨i, j, -, -, -, -, hc⟩ := nearCell_of_close M h
    exact ⟨(i, j), mem_cells _, hc⟩
  · simp [h]

/-- A cell that the Rust search returns passes the first test and all three
cell tests. -/
theorem rustPureSaddle_spec {M : Matrix (Fin m) (Fin n) ℝ} {τ : ℝ} {rc : Fin m × Fin n}
    (h : rustPureSaddle M τ = some rc) :
    Close τ (pureMaximin M) (pureMinimax M) ∧ NearCell M τ rc := by
  unfold rustPureSaddle at h
  split_ifs at h with hc
  · exact ⟨hc, by simpa using List.find?_some h⟩

/-- **ABSTRACT-MAT-2 (c)**: a cell that passes the three tests under a close
`lower`/`upper` pair has one-hot gap `colMax c - rowMin r` between
`upper - lower ≥ 0` and `3 τ`. With `τ = 1e-12` the gap is at most `3e-12`. -/
theorem pairGap_bounds {M : Matrix (Fin m) (Fin n) ℝ} {τ : ℝ} {rc : Fin m × Fin n}
    (h : Close τ (pureMaximin M) (pureMinimax M)) (hc : NearCell M τ rc) :
    0 ≤ pureMinimax M - pureMaximin M ∧
      pureMinimax M - pureMaximin M ≤ colMax M rc.2 - rowMin M rc.1 ∧
      colMax M rc.2 - rowMin M rc.1 ≤ 3 * τ := by
  obtain ⟨h1, h2, -⟩ := hc
  unfold Close at h h1 h2
  have hle := pureMaximin_le_pureMinimax M
  have hr := rowMin_le_pureMaximin M rc.1
  have hcm := pureMinimax_le_colMax M rc.2
  rw [abs_le] at h h1 h2
  refine ⟨by linarith, by linarith, by linarith⟩

/-- **ABSTRACT-MAT-2 (c)** (`src/abstract/matrix.py:150-151`,
`src/abstract/docs/PACKED_TABLEBASE_PARITY.md:81-86`): when `3 τ ≤ g`, the
Python `2e-7` re-check never rejects a candidate cell, so Python returns the
same first cell as Rust. Both scan rows, then columns, in increasing order.
With `τ = 1e-12` and `g = 2e-7` the hypothesis holds. In exact arithmetic the
pure-versus-LP routing of the two implementations is identical in both
directions. -/
theorem pythonPureSaddle_eq_rust (M : Matrix (Fin m) (Fin n) ℝ) {τ g : ℝ} (hg : 3 * τ ≤ g) :
    pythonPureSaddle M τ g = rustPureSaddle M τ := by
  unfold pythonPureSaddle rustPureSaddle
  split_ifs with h
  · congr 1
    funext rc
    apply decide_eq_decide.mpr
    constructor
    · exact fun h' => h'.1
    · intro hc
      refine ⟨hc, ?_⟩
      rw [(pythonGap_eq M rc.1 rc.2).1]
      exact (pairGap_bounds h hc).2.2.trans hg
  · rfl

/-- **ABSTRACT-MAT-2 (a, c)**: the Rust search returns `None`, and the row goes to
the LP, exactly when `upper - lower > τ`. For `τ ≥ 0` such a matrix has no
exact pure saddle. -/
theorem rustPureSaddle_eq_none_iff (M : Matrix (Fin m) (Fin n) ℝ) (τ : ℝ) :
    rustPureSaddle M τ = none ↔ τ < pureMinimax M - pureMaximin M := by
  rw [← Option.not_isSome_iff_eq_none, rustPureSaddle_isSome_iff]
  unfold Close
  rw [abs_sub_comm, abs_of_nonneg (sub_nonneg.mpr (pureMaximin_le_pureMinimax M)), not_le]

/-- The LP route implies no exact pure saddle. -/
theorem no_pureSaddle_of_rust_none (M : Matrix (Fin m) (Fin n) ℝ) {τ : ℝ} (hτ : 0 ≤ τ)
    (h : rustPureSaddle M τ = none) : ¬ ∃ i j, IsPureSaddle M i j := by
  rw [exists_pureSaddle_iff]
  rw [rustPureSaddle_eq_none_iff] at h
  intro he
  rw [he, sub_self] at h
  linarith

/-- **CRATES-ABS-5, ABSTRACT-MAT-2** (`src/crates/abstract_solver/src/lib.rs:168-194,404-417`):
the value the Rust search returns, `M[r, c]`, lies within `2 τ` of the matrix
value. With `τ = 1e-12` the error is at most `2e-12`. -/
theorem rustPureSaddle_error {M : Matrix (Fin m) (Fin n) ℝ} {τ : ℝ} {rc : Fin m × Fin n}
    (h : rustPureSaddle M τ = some rc) : |M rc.1 rc.2 - value M| ≤ 2 * τ := by
  obtain ⟨hc, -, -, h3⟩ := rustPureSaddle_spec h
  have hv1 := pureMaximin_le_value M
  have hv2 := value_le_pureMinimax M
  have hle := pureMaximin_le_pureMinimax M
  unfold Close at hc h3
  rw [abs_le] at hc h3 ⊢
  constructor <;> linarith

/-- The near-saddle counterexample `!![t, 0; 0, t]` with `t > 0`. -/
def nearPennies (t : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![t, 0; 0, t]

/-- **ABSTRACT-MAT-2** (reader doubt; `src/abstract/packed_tablebase.py:744`,
`src/abstract/docs/MODEL.md:150-152`): with `t = 1e-12` the Rust search accepts
`!![t, 0; 0, t]` as pure, although the matrix has no exact pure saddle, and
every returned cell has one-hot gap `colMax c - rowMin r > 0`. The builder
stores `saddle_gap = 0.0` for that row, which is not the certificate of the
stored one-hot pair. The doc's "genuinely mixed" reading of the non-pure
states is therefore one-directional: some mixed states take the pure path. -/
theorem nearPennies_counterexample :
    (rustPureSaddle (nearPennies 1e-12) 1e-12).isSome ∧
      (¬ ∃ i j, IsPureSaddle (nearPennies 1e-12) i j) ∧
      ∀ rc, rustPureSaddle (nearPennies 1e-12) 1e-12 = some rc →
        0 < colMax (nearPennies 1e-12) rc.2 - rowMin (nearPennies 1e-12) rc.1 := by
  set M := nearPennies (1e-12 : ℝ) with hM
  have ht : (0 : ℝ) < 1e-12 := by norm_num
  have hno : ¬ ∃ i j, IsPureSaddle M i j := by
    rintro ⟨i, j, h1, h2⟩
    fin_cases i <;> fin_cases j
    · have := rowMin_le M 0 1
      simp [hM, nearPennies] at h1 this; linarith
    · have := le_colMax M 1 1
      simp [hM, nearPennies] at h2 this; linarith
    · have := le_colMax M 0 0
      simp [hM, nearPennies] at h2 this; linarith
    · have := rowMin_le M 1 0
      simp [hM, nearPennies] at h1 this; linarith
  have hlo : 0 ≤ pureMaximin M :=
    (le_inf' _ _ fun j _ => by fin_cases j <;> norm_num [hM, nearPennies]).trans
      (rowMin_le_pureMaximin M 0)
  have hhi : pureMinimax M ≤ 1e-12 :=
    (pureMinimax_le_colMax M 0).trans
      (sup'_le _ _ fun i _ => by fin_cases i <;> norm_num [hM, nearPennies])
  have hclose : Close 1e-12 (pureMaximin M) (pureMinimax M) := by
    have := pureMaximin_le_pureMinimax M
    unfold Close; rw [abs_le]; constructor <;> linarith
  have hne : pureMaximin M ≠ pureMinimax M := fun h => hno ((exists_pureSaddle_iff M).2 h)
  refine ⟨(rustPureSaddle_isSome_iff M _).2 hclose, hno, fun rc hrc => ?_⟩
  obtain ⟨-, h2, -⟩ := pairGap_bounds hclose (rustPureSaddle_spec hrc).2
  have := lt_of_le_of_ne (pureMaximin_le_pureMinimax M) hne
  linarith

end Abstract

/-! ## The STL leap window -/

section Window

variable {α : Type*} [LinearOrder α] {β : Type*} [Fintype β] [Nonempty β] {N : ℕ} [NeZero N]

/-- The window stage (`src/stl/solver/leap_oracle.py:26-35`, `stage_matrix`):
rows `0, …, N - 1` are the square stage, and row `N` (the doc's row 61) is
Baku's leap action, with payoff `f` in every column. -/
def windowMatrix (M : Matrix (Fin N) β α) (f : α) : Matrix (Fin (N + 1)) β α :=
  fun i j => if h : i.val < N then M ⟨i.val, h⟩ j else f

theorem rowMin_window_castSucc (M : Matrix (Fin N) β α) (f : α) (i : Fin N) :
    rowMin (windowMatrix M f) i.castSucc = rowMin M i := by
  unfold rowMin windowMatrix
  simp only [Fin.val_castSucc, i.isLt, ↓reduceDIte]

theorem rowMin_window_last (M : Matrix (Fin N) β α) (f : α) :
    rowMin (windowMatrix M f) (Fin.last N) = f := by
  unfold rowMin windowMatrix
  simp only [Fin.val_last, lt_irrefl, ↓reduceDIte, inf'_const]

theorem colMax_window (M : Matrix (Fin N) β α) (f : α) (c : β) :
    colMax (windowMatrix M f) c = max (colMax M c) f := by
  apply le_antisymm
  · refine sup'_le _ _ fun i _ => ?_
    unfold windowMatrix
    split_ifs with h
    · exact (le_colMax M _ c).trans (le_max_left _ _)
    · exact le_max_right _ _
  · refine max_le ?_ ?_
    · obtain ⟨i, hi⟩ := exists_colMax_eq M c
      have h := le_colMax (windowMatrix M f) i.castSucc c
      simp only [windowMatrix, Fin.val_castSucc, i.isLt, ↓reduceDIte] at h
      exact hi ▸ h
    · have h := le_colMax (windowMatrix M f) (Fin.last N) c
      simp only [windowMatrix, Fin.val_last, lt_irrefl, ↓reduceDIte] at h
      exact h

/-- The window's pure maximin is `max(maximin, f)`: the lifted lower bound of
`leap.rs:436-441` and `leap_oracle.py:74-75` is the pure maximin of the
61-row window matrix. -/
theorem pureMaximin_window (M : Matrix (Fin N) β α) (f : α) :
    pureMaximin (windowMatrix M f) = max (pureMaximin M) f := by
  apply le_antisymm
  · refine sup'_le _ _ fun i _ => ?_
    by_cases h : i.val < N
    · have hi : i = (⟨i.val, h⟩ : Fin N).castSucc := Fin.ext rfl
      rw [hi, rowMin_window_castSucc]
      exact (rowMin_le_pureMaximin M _).trans (le_max_left _ _)
    · have hi : i = Fin.last N := Fin.ext (by have := i.isLt; simp; omega)
      rw [hi, rowMin_window_last]; exact le_max_right _ _
  · refine max_le ?_ ?_
    · obtain ⟨i, hi⟩ := exists_pureMaximin_eq M
      rw [hi, ← rowMin_window_castSucc M f]; exact rowMin_le_pureMaximin _ _
    · have h := rowMin_le_pureMaximin (windowMatrix M f) (Fin.last N)
      rwa [rowMin_window_last] at h

/-- The window's pure minimax is `max(minimax, f)`: the lifted upper bound. -/
theorem pureMinimax_window (M : Matrix (Fin N) β α) (f : α) :
    pureMinimax (windowMatrix M f) = max (pureMinimax M) f := by
  apply le_antisymm
  · obtain ⟨j, hj⟩ := exists_pureMinimax_eq M
    rw [hj, ← colMax_window]; exact pureMinimax_le_colMax _ _
  · refine le_inf' _ _ fun c _ => ?_
    rw [colMax_window]
    exact max_le_max (pureMinimax_le_colMax M c) le_rfl

variable {n : ℕ} (s : ℕ → α) (f : α)

/-- **STL-MG-3** (`src/stl/solver/leap_oracle.py:72-75`,
`src/crates/stl_solver/src/leap.rs:383-384,436-441`): in the leap window the
lifted lower bound `max(max(lo, min(f, s0)), f)` is the pure maximin of the
window stage. -/
theorem window_pureMaximin :
    pureMaximin (windowMatrix (toeplitzL (n + 1) s f) f) =
      max (max (prefixMin s n) (min f (s 0))) f := by
  rw [pureMaximin_window, pureMaximin_toeplitzL]

/-- **STL-MG-3**: the lifted upper bound `max(min(hi, max(f, s0)), f)` is the pure
minimax of the window stage. -/
theorem window_pureMinimax :
    pureMinimax (windowMatrix (toeplitzL (n + 1) s f) f) =
      max (min (prefixMax s n) (max f (s 0))) f := by
  rw [pureMinimax_window, pureMinimax_toeplitzL]

/-- When `s0 ≤ f`, both lifted bounds equal `f`: the window gate passes with
gap zero. -/
theorem window_bounds_of_s0_le (h : s 0 ≤ f) :
    max (max (prefixMin s n) (min f (s 0))) f = f ∧
      max (min (prefixMax s n) (max f (s 0))) f = f := by
  have hlo : prefixMin s n ≤ s 0 := prefixMin_le s (Nat.zero_le _)
  constructor
  · exact max_eq_right (max_le (hlo.trans h) (min_le_left _ _))
  · exact max_eq_right ((min_le_right _ _).trans (max_le le_rfl h))

/-- When `f < s0`, the lift changes neither bound. -/
theorem window_bounds_of_lt (h : f < s 0) :
    max (max (prefixMin s n) (min f (s 0))) f = max (prefixMin s n) (min f (s 0)) ∧
      max (min (prefixMax s n) (max f (s 0))) f = min (prefixMax s n) (max f (s 0)) := by
  have hhi : s 0 ≤ prefixMax s n := le_prefixMax s (Nat.zero_le _)
  constructor
  · exact max_eq_left ((min_eq_left h.le).symm.le.trans (le_max_right _ _))
  · exact max_eq_left (le_min (h.le.trans hhi) (le_max_left _ _))

end Window

section WindowReal

variable {n : ℕ} (s : ℕ → ℝ) (f : ℝ)

/-- The rung-0 closed-form bounds `mn = max(lo, min(f, s0))` and
`mx = min(hi, max(f, s0))`. -/
noncomputable def mn (n : ℕ) (s : ℕ → ℝ) (f : ℝ) : ℝ := max (prefixMin s n) (min f (s 0))

/-- See `mn`. -/
noncomputable def mx (n : ℕ) (s : ℕ → ℝ) (f : ℝ) : ℝ := min (prefixMax s n) (max f (s 0))

/-- Python's window rung-0 value (`leap_oracle.py:74-81`): the midpoint of the
lifted bounds. -/
noncomputable def pythonWindowValue (n : ℕ) (s : ℕ → ℝ) (f : ℝ) : ℝ :=
  (max (mn n s f) f + max (mx n s f) f) / 2

/-- Rust's window rung-0 value (`leap.rs:447-450,559-566`): the unlifted
midpoint `0.5 (mn + mx)`, then `max(v, f)`. -/
noncomputable def rustWindowValue (n : ℕ) (s : ℕ → ℝ) (f : ℝ) : ℝ :=
  max ((mn n s f + mx n s f) / 2) f

/-- **STL-MG-3, CRATES-PURE-1** (`leap.rs:447-450,559-566` against
`leap_oracle.py:74-81`): Python stores the midpoint of the lifted bounds and
Rust stores `max` of the unlifted midpoint and `f`. The two formulas differ for
general `mn < f < mx`, but on the Toeplitz closed forms they are equal:
`s0 ≤ f` makes both equal `f`, and `f < s0` makes `mn ≥ f`, so the lift is the
identity. The gate expression is the same in both. -/
theorem leap_rust_eq_python : rustWindowValue n s f = pythonWindowValue n s f := by
  unfold rustWindowValue pythonWindowValue
  rcases le_or_gt (s 0) f with h | h
  · obtain ⟨h1, h2⟩ := window_bounds_of_s0_le (n := n) s f h
    rw [mn, mx, h1, h2]
    have hlo : prefixMin s n ≤ s 0 := prefixMin_le s (Nat.zero_le _)
    have e1 : max (prefixMin s n) (min f (s 0)) ≤ f :=
      max_le (hlo.trans h) (min_le_left _ _)
    have e2 : min (prefixMax s n) (max f (s 0)) ≤ f :=
      (min_le_right _ _).trans (max_le le_rfl h)
    rw [max_eq_right (by linarith)]; ring
  · obtain ⟨h1, h2⟩ := window_bounds_of_lt (n := n) s f h
    rw [mn, mx, h1, h2]
    have e1 : f ≤ max (prefixMin s n) (min f (s 0)) :=
      (min_eq_left h.le).symm.le.trans (le_max_right _ _)
    have e2 : f ≤ min (prefixMax s n) (max f (s 0)) :=
      le_min (h.le.trans (le_prefixMax s (Nat.zero_le _))) (le_max_left _ _)
    rw [max_eq_left (by linarith)]

/-- **STL-MG-3** (`paper/stl.tex:178-181`, `src/crates/docs/LEAP_CERTIFICATE.md:112`):
the square stage's value lies in `[max(lo, min(f, s0)), min(hi, max(f, s0))]`. -/
theorem stage_value_mem :
    mn n s f ≤ value (toeplitz (n + 1) s f) ∧ value (toeplitz (n + 1) s f) ≤ mx n s f := by
  rw [toeplitz_eq_toeplitzL]
  constructor
  · rw [mn, ← pureMaximin_toeplitzL]; exact pureMaximin_le_value _
  · rw [mx, ← pureMinimax_toeplitzL]; exact value_le_pureMinimax _

/-- **STL-MG-3, CRATES-PURE-1** (`leap_oracle.py:72-81`, `leap.rs:436-451`): if
the lifted gap is at most `ε`, the stored rung-0 value (Python's, equal to
Rust's by `leap_rust_eq_python`) lies within `ε / 2` of the value of the
61-row window matrix. Without the window the same holds for the square stage
(`pure_midpoint_error`). -/
theorem window_rung0_error {ε : ℝ} (hgap : max (mx n s f) f - max (mn n s f) f ≤ ε) :
    |pythonWindowValue n s f - value (windowMatrix (toeplitz (n + 1) s f) f)| ≤ ε / 2 ∧
      |rustWindowValue n s f - value (windowMatrix (toeplitz (n + 1) s f) f)| ≤ ε / 2 := by
  have hl := window_pureMaximin (n := n) s f
  have hu := window_pureMinimax (n := n) s f
  rw [← toeplitz_eq_toeplitzL] at hl hu
  have key : |pythonWindowValue n s f - value (windowMatrix (toeplitz (n + 1) s f) f)| ≤ ε / 2 := by
    have := pure_midpoint_error (windowMatrix (toeplitz (n + 1) s f) f)
      (ε := ε) (by rw [hl, hu]; exact hgap)
    rw [hl, hu] at this
    exact this
  exact ⟨key, by rw [leap_rust_eq_python]; exact key⟩

/-- When `s0 ≤ f`, the window value equals `f` exactly, and both stores return
`f`. -/
theorem window_value_of_s0_le (h : s 0 ≤ f) :
    value (windowMatrix (toeplitz (n + 1) s f) f) = f ∧ pythonWindowValue n s f = f := by
  have hl := window_pureMaximin (n := n) s f
  have hu := window_pureMinimax (n := n) s f
  rw [← toeplitz_eq_toeplitzL] at hl hu
  obtain ⟨h1, h2⟩ := window_bounds_of_s0_le (n := n) s f h
  have a := pureMaximin_le_value (windowMatrix (toeplitz (n + 1) s f) f)
  have b := value_le_pureMinimax (windowMatrix (toeplitz (n + 1) s f) f)
  rw [hl, h1] at a; rw [hu, h2] at b
  refine ⟨le_antisymm b a, ?_⟩
  unfold pythonWindowValue; rw [mn, mx, h1, h2]; ring

end WindowReal

/-! ## Instances and the reader-doubt counterexample of `architecture.md` -/

section Examples

/-- Running minima of an antitone sequence are its last term. -/
theorem prefixMin_of_antitone {s : ℕ → ℝ} (hs : Antitone s) (k : ℕ) : prefixMin s k = s k := by
  induction k with
  | zero => rfl
  | succ k ih => rw [prefixMin, ih, min_eq_right (hs (Nat.le_succ k))]

/-- Running maxima of a monotone sequence are its last term. -/
theorem prefixMax_of_monotone {s : ℕ → ℝ} (hs : Monotone s) (k : ℕ) : prefixMax s k = s k := by
  induction k with
  | zero => rfl
  | succ k ih => rw [prefixMax, ih, max_eq_right (hs (Nat.le_succ k))]

/-- Running minima of a monotone sequence are its first term. -/
theorem prefixMin_of_monotone {s : ℕ → ℝ} (hs : Monotone s) (k : ℕ) : prefixMin s k = s 0 := by
  induction k with
  | zero => rfl
  | succ k ih => rw [prefixMin, ih, min_eq_left (hs (Nat.zero_le _))]

/-- **COMPACT-RUNG1-1** (`src/dth_compact/tests/test_rungs.py:81-96`): a constant
stage with every payoff `a` is a pure saddle with value `a`. -/
theorem constant_stage_value (a : ℝ) : value (toeplitz 60 (fun _ => a) a) = a := by
  have h := stage_value_mem (n := 59) (fun _ => a) a
  have hmin : prefixMin (fun _ : ℕ => a) 59 = a := prefixMin_of_antitone (fun _ _ _ => le_rfl) 59
  have hmax : prefixMax (fun _ : ℕ => a) 59 = a := prefixMax_of_monotone (fun _ _ _ => le_rfl) 59
  simp only [mn, mx, hmin, hmax, min_self, max_self] at h
  exact le_antisymm h.2 h.1

/-- The test sequence `linspace(-1, 1, 60)`. -/
noncomputable def linspace60 (k : ℕ) : ℝ := -1 + 2 * k / 59

theorem linspace60_monotone : Monotone linspace60 := by
  intro a b h
  unfold linspace60
  have : (a : ℝ) ≤ b := by exact_mod_cast h
  have : 2 * (a : ℝ) / 59 ≤ 2 * b / 59 := by
    apply div_le_div_of_nonneg_right _ (by norm_num); linarith
  linarith

/-- **COMPACT-RUNG1-1** (`test_rungs.py:81-96`): with `s = linspace(-1, 1, 60)` and
`f = -1`, the rung-1 bounds are both `-1`, and the stage value is `-1`. -/
theorem linspace_stage_value :
    mn 59 linspace60 (-1) = -1 ∧ mx 59 linspace60 (-1) = -1 ∧
      value (toeplitz 60 linspace60 (-1)) = -1 := by
  have h0 : linspace60 0 = -1 := by simp [linspace60]
  have h59 : linspace60 59 = 1 := by norm_num [linspace60]
  have hmin : prefixMin linspace60 59 = -1 := by
    rw [prefixMin_of_monotone linspace60_monotone, h0]
  have hmax : prefixMax linspace60 59 = 1 := by
    rw [prefixMax_of_monotone linspace60_monotone, h59]
  have e1 : mn 59 linspace60 (-1) = -1 := by
    simp only [mn, hmin, h0, min_self, max_self]
  have e2 : mx 59 linspace60 (-1) = -1 := by
    simp only [mx, hmax, h0, max_self]; norm_num
  have h := stage_value_mem (n := 59) linspace60 (-1)
  rw [e1, e2] at h
  exact ⟨e1, e2, le_antisymm h.2 h.1⟩

/-- **COMPACT-RUNG1-1** (reader doubt on `src/dth_compact/architecture.md:150-167`):
the doc's step "every row minimum and column maximum is one of four numbers"
is false. With `s[k] = -k/100` and `f = 1`, the 60 row minima are pairwise
distinct. The closed forms remain correct: only the maximum of the row minima
and the minimum of the column maxima are one of two numbers each. -/
theorem rowMin_injective_counterexample :
    Function.Injective (rowMin (toeplitzL 60 (fun k : ℕ => -(k : ℝ) / 100) (1 : ℝ))) := by
  set s : ℕ → ℝ := fun k => -(k : ℝ) / 100 with hs
  have anti : Antitone s := by
    intro a b h
    simp only [hs]
    have : (a : ℝ) ≤ b := by exact_mod_cast h
    apply div_le_div_of_nonneg_right _ (by norm_num); linarith
  have key : ∀ d : Fin 60, rowMin (toeplitzL 60 s 1) d = s (59 - d.val) := by
    intro d
    rw [rowMin_toeplitzL (n := 59), rowMinFormula, prefixMin_of_antitone anti]
    split_ifs
    · rfl
    · apply min_eq_left
      simp only [hs]
      have : (0 : ℝ) ≤ ((59 - d.val : ℕ) : ℝ) := Nat.cast_nonneg _
      linarith [show -(((59 - d.val : ℕ) : ℝ)) / 100 ≤ 0 by
        apply div_nonpos_of_nonpos_of_nonneg <;> linarith]
  intro a b hab
  rw [key, key] at hab
  simp only [hs] at hab
  have : ((59 - a.val : ℕ) : ℝ) = ((59 - b.val : ℕ) : ℝ) := by linarith
  have h2 : 59 - a.val = 59 - b.val := by exact_mod_cast this
  exact Fin.ext (by have := a.isLt; have := b.isLt; omega)

/-- **CPP-DOC-PURE-2, CPP-CODE-PURE-3** (non-vacuity): the DTH stage of any
state and any continuation `W` has pure bounds that enclose its value. -/
example (W : DTH.State → ℝ) (x : DTH.State) :
    pureMaximin (DTH.stage W x) ≤ value (DTH.stage W x) ∧
      value (DTH.stage W x) ≤ pureMinimax (DTH.stage W x) :=
  ⟨pureMaximin_le_value _, value_le_pureMinimax _⟩

/-- The all-zero stage has pure bounds `0` and `0`. -/
theorem zero_stage_bounds :
    pureMaximin (toeplitz 60 (fun _ => (0 : ℝ)) 0) = 0 ∧
      pureMinimax (toeplitz 60 (fun _ => (0 : ℝ)) 0) = 0 := by
  have h1 : prefixMin (fun _ : ℕ => (0 : ℝ)) 59 = 0 :=
    prefixMin_of_antitone (fun _ _ _ => le_rfl) 59
  have h2 : prefixMax (fun _ : ℕ => (0 : ℝ)) 59 = 0 :=
    prefixMax_of_monotone (fun _ _ _ => le_rfl) 59
  rw [toeplitz_eq_toeplitzL, pureMaximin_toeplitzL, pureMinimax_toeplitzL, h1, h2]
  simp

/-- Non-vacuity of `rounded_pure_rung_error`: its hypotheses hold for the
all-zero stage with `u = 2^-53`, `ε = 1e-6`, and zero rounding errors. -/
example : |(pureMaximin (toeplitz 60 (fun _ => (0 : ℝ)) 0) +
      pureMinimax (toeplitz 60 (fun _ => (0 : ℝ)) 0)) / 2 * (1 + 0) + 0 -
      value (toeplitz 60 (fun _ => (0 : ℝ)) 0)| ≤
    1e-6 / (2 * (1 - (2 : ℝ)⁻¹ ^ 53)) + (2 : ℝ)⁻¹ ^ 53 *
      |(pureMaximin (toeplitz 60 (fun _ => (0 : ℝ)) 0) +
        pureMinimax (toeplitz 60 (fun _ => (0 : ℝ)) 0)) / 2| + 0 :=
  rounded_pure_rung_error (δ₁ := 0) (δ₂ := 0) (η := 0) _ (by norm_num) (by simp)
    (by simp) (by simp) (by rw [zero_stage_bounds.1, zero_stage_bounds.2]; norm_num)

end Examples

end Formal.Toeplitz.PureSaddle
