import Formal.Toeplitz.Equalizer

/-!
# Monotone success payoffs and the recurrence residue

Conventions follow `Formal.Toeplitz.Equalizer`: the stage matrix is
`toeplitz n s f` with 0-based indices, `M[d,c] = s[c-d]` for `d ≤ c` and `f`
below the diagonal; the row player (Dropper) maximizes `p ⬝ᵥ (M *ᵥ q)`. The
weights are `weights s f`, with `r[0] = 1`; the kernel coefficients are
`b[m] = (s[m-1] - s[m]) / (s[0] - f)`. All results hold over `ℝ`. The
floating-point statements use an explicit abstract rounding model and never
assert IEEE bit-level behavior.

This file proves:

* `nonneg_equalizer_saddle`: nonnegative weights give an exact saddle point
  `(r / W, reverse(r) / W)` whose two equalized payoffs coincide with the value
  (COMPACT-RUNG2-3, CPP-CODE-REC-2, CANONICAL-MG-7);
* `weights_nonneg_of_monotone`: a nondecreasing `s` with `f > s[0]` gives
  nonnegative weights (CRATES-KINK-1, STL-REC-2);
* `value_eq_of_monotone`, `pureGap_eq_zero_of_monotone`: a nondecreasing `s`
  with `f ≤ s[0]` is a pure saddle at `(0,0)` (CRATES-KINK-2);
* `monotone_ladder`, `not_reachesResidue_of_monotone`,
  `kinkIndex_isSome_of_reachesResidue`: in exact arithmetic a residue stage
  needs a negative step (CRATES-KINK-2, STL-REC-2, CRATES-CRASH-4(b));
* `kinkIndex_eq_none_iff`, `kinkIndex_eq_some`: the specification of
  `kink_index` (CRATES-KINK-2);
* `computedWeights_nonneg`: monotone rounding without overflow preserves the
  sign of the weights (CRATES-KINK-1, STL-REC-2);
* `crash_basis`, `crash_optimal`: the recurrence crash basis of the packing
  LP is primal feasible iff dual feasible iff `r ≥ 0`, and then optimal with
  `z = W` (CRATES-CRASH-4(a));
* `overflow_counterexample`, `rounded_weights_overflow`: the stage
  `s = (0,1,...,1)`, `f = 2e-6` is nondecreasing, has nonnegative exact
  weights, fails the pure rung, and has weights beyond `2^1024`, so the
  binary64 kernel sends it to the residue with no negative step;
* `clip_counterexample`: a clipped candidate need not be an equilibrium.
-/

open Finset Matrix

namespace Formal.Toeplitz

/-! ## Sign of the recurrence weights -/

theorem weights_succ (s : ℕ → ℝ) (f : ℝ) (k : ℕ) :
    weights s f (k + 1) =
      -(∑ m : Fin (k + 1), (s (m.val + 1) - s m.val) * weights s f (k - m.val)) / (s 0 - f) := by
  rw [weights]

/-- **Nondecreasing `s` with `f > s[0]` gives nonnegative weights**
(CRATES-KINK-1, STL-REC-2; `LEAP_CERTIFICATE.md:246-248`,
`leap_support.py:28-32`, `leap.rs:928-931`), in exact arithmetic. Every
coefficient `b[m] = (s[m-1] - s[m]) / (s[0] - f)` is a nonpositive number
over a negative one, and the recurrence sums nonnegative products. -/
theorem weights_nonneg_of_monotone {n : ℕ} {s : ℕ → ℝ} {f : ℝ}
    (hs : ∀ m, m + 1 < n → s m ≤ s (m + 1)) (hf : s 0 < f) :
    ∀ k < n, 0 ≤ weights s f k := by
  intro k
  induction k using Nat.strong_induction_on with
  | _ k ih =>
    intro hk
    cases k with
    | zero => rw [weights_zero]; exact zero_le_one
    | succ k =>
      rw [weights_succ]
      apply div_nonneg_of_nonpos
      · rw [neg_nonpos]
        refine Finset.sum_nonneg fun m _ => mul_nonneg ?_ ?_
        · have := m.isLt
          have := hs m.val (by omega)
          linarith
        · have := m.isLt
          exact ih _ (by omega) (by omega)
      · linarith

/-- When `s` is constant from index `1` on, the weights are geometric:
`r[k] = ((s[0] - s[1]) / (s[0] - f))^k`. -/
theorem weights_geometric {s : ℕ → ℝ} (f : ℝ) (hs : ∀ m, 1 ≤ m → s (m + 1) = s m) (k : ℕ) :
    weights s f k = ((s 0 - s 1) / (s 0 - f)) ^ k := by
  induction k with
  | zero => rw [weights_zero, pow_zero]
  | succ k ih =>
    rw [weights_succ, Fin.sum_univ_succ]
    have hz : ∑ i : Fin k, (s ((Fin.succ i : Fin (k + 1)).val + 1) - s (Fin.succ i : Fin (k + 1)).val) *
        weights s f (k - (Fin.succ i : Fin (k + 1)).val) = 0 := by
      refine Finset.sum_eq_zero fun i _ => ?_
      simp only [Fin.val_succ]
      rw [hs (i.val + 1) (by omega), sub_self, zero_mul]
    rw [hz, add_zero]
    simp only [Fin.val_zero, Nat.sub_zero, zero_add, ih, pow_succ]
    ring

/-! ## Pure bounds and the exact-arithmetic ladder -/

section Ladder

variable (n : ℕ) [NeZero n]

/-- The saddle-gap gate `GAP = 1e-6` (`leap.rs:28`, `leap_oracle.py:8`). -/
noncomputable def gapTol : ℝ := 1 / 10 ^ 6

/-- The `d`-size guard `1e-12` of the recurrence rung (`leap.rs:453`, `leap_oracle.py:83`). -/
noncomputable def dTol : ℝ := 1 / 10 ^ 12

/-- `lo = min_k s[k]` over the `n` success payoffs. -/
noncomputable def succLo (s : ℕ → ℝ) : ℝ := univ.inf' univ_nonempty fun k : Fin n => s k.val

/-- `hi = max_k s[k]` over the `n` success payoffs. -/
noncomputable def succHi (s : ℕ → ℝ) : ℝ := univ.sup' univ_nonempty fun k : Fin n => s k.val

/-- `mn = max(lo, min(f, s[0]))` (`leap.rs:383`). -/
noncomputable def pureLower (s : ℕ → ℝ) (f : ℝ) : ℝ := max (succLo n s) (min f (s 0))

/-- `mx = min(hi, max(f, s[0]))` (`leap.rs:384`). -/
noncomputable def pureUpper (s : ℕ → ℝ) (f : ℝ) : ℝ := min (succHi n s) (max f (s 0))

/-- The pure-rung gap `upper - lower`, with both bounds lifted through
`max(·, f)` in a leap window (`leap.rs:441-447`, `leap_oracle.py:73-77`). -/
noncomputable def pureGap (s : ℕ → ℝ) (f : ℝ) (window : Bool) : ℝ :=
  if window then max (pureUpper n s f) f - max (pureLower n s f) f
  else pureUpper n s f - pureLower n s f

/-- The clipped weight sum `∑ max(r[k], 0)` (`leap.rs:430-439`). -/
noncomputable def clipSum (s : ℕ → ℝ) (f : ℝ) : ℝ := ∑ k ∈ range n, max (weights s f k) 0

/-- The clipped Checker candidate `q[j] = max(r[n-1-j], 0) / ∑ max(r, 0)`. -/
noncomputable def clipCol (s : ℕ → ℝ) (f : ℝ) : Fin n → ℝ :=
  fun j => max (weights s f (n - 1 - j.val)) 0 / clipSum n s f

/-- `min (M q)` for the clipped candidate: the Dropper bound of its mirror. -/
noncomputable def fullLower (s : ℕ → ℝ) (f : ℝ) : ℝ :=
  univ.inf' univ_nonempty fun i => (toeplitz n s f *ᵥ clipCol n s f) i

/-- `max (M q)` for the clipped candidate. -/
noncomputable def fullUpper (s : ℕ → ℝ) (f : ℝ) : ℝ :=
  univ.sup' univ_nonempty fun i => (toeplitz n s f *ᵥ clipCol n s f) i

/-- The full-check gap, lifted through `max(·, f)` in a window (`leap.rs:505-510`). -/
noncomputable def fullGap (s : ℕ → ℝ) (f : ℝ) (window : Bool) : ℝ :=
  if window then max (fullUpper n s f) f - max (fullLower n s f) f
  else fullUpper n s f - fullLower n s f

/-- `cum = ∑ q[j]` of the clipped candidate (`leap.rs:510`). -/
noncomputable def clipMass (s : ℕ → ℝ) (f : ℝ) : ℝ := ∑ j, clipCol n s f j

/-- The hypotheses of the weight-sum shortcut (`leap.rs:475`, `leap_oracle.py:99`):
all weights nonnegative, bounded payoffs, and `|d| ≥ GAP`. -/
def ShortcutOk (s : ℕ → ℝ) (f : ℝ) : Prop :=
  (∀ k < n, 0 ≤ weights s f k) ∧ -2 ≤ succLo n s ∧ succHi n s ≤ 2 ∧ |f| ≤ 2 ∧
    gapTol ≤ |s 0 - f|

/-- The kernel's route to the residue, evaluated in exact real arithmetic
(`sweep_key_rs`, `leap.rs:447-510`). A stage reaches the residue when the pure
rung fails and then either the recurrence guard fires (`|d| < 1e-12` or a
nonpositive clipped sum; over `ℝ` every sum is finite) or the shortcut does not
apply and the full check fails. Python's `solve_stage` omits the `cum` test;
its route is the same condition without that disjunct, so every statement
below that excludes this route excludes Python's route too. -/
def ReachesResidue (s : ℕ → ℝ) (f : ℝ) (window : Bool) : Prop :=
  gapTol < pureGap n s f window ∧
    (|s 0 - f| < dTol ∨ clipSum n s f ≤ 0 ∨
      (¬ ShortcutOk n s f ∧
        (dTol < |clipMass n s f - 1| ∨ gapTol < fullGap n s f window)))

variable {n}

theorem succLo_le (s : ℕ → ℝ) (k : Fin n) : succLo n s ≤ s k.val :=
  Finset.inf'_le _ (mem_univ k)

theorem le_succHi (s : ℕ → ℝ) (k : Fin n) : s k.val ≤ succHi n s :=
  Finset.le_sup' (fun k : Fin n => s k.val) (mem_univ k)

theorem succLo_le_zero (s : ℕ → ℝ) : succLo n s ≤ s 0 := by
  simpa using succLo_le s (0 : Fin n)

theorem zero_le_succHi (s : ℕ → ℝ) : s 0 ≤ succHi n s := by
  simpa using le_succHi s (0 : Fin n)

theorem pureLower_le (s : ℕ → ℝ) (f : ℝ) : pureLower n s f ≤ s 0 :=
  max_le (succLo_le_zero s) (min_le_right _ _)

theorem le_pureUpper (s : ℕ → ℝ) (f : ℝ) : s 0 ≤ pureUpper n s f :=
  le_min (zero_le_succHi s) (le_max_right _ _)

theorem pureUpper_sub_pureLower_le (s : ℕ → ℝ) (f : ℝ) :
    pureUpper n s f - pureLower n s f ≤ |s 0 - f| := by
  have h1 : pureUpper n s f ≤ max f (s 0) := min_le_right _ _
  have h2 : min f (s 0) ≤ pureLower n s f := le_max_right _ _
  have h3 : max f (s 0) - min f (s 0) = |f - s 0| := max_sub_min_eq_abs' f (s 0)
  rw [abs_sub_comm] at h3
  linarith

theorem pureGap_le_unlifted (s : ℕ → ℝ) (f : ℝ) (window : Bool) :
    pureGap n s f window ≤ pureUpper n s f - pureLower n s f := by
  have hle : pureLower n s f ≤ pureUpper n s f := (pureLower_le s f).trans (le_pureUpper s f)
  cases window with
  | false => simp [pureGap]
  | true =>
    simp only [pureGap, ↓reduceIte]
    have := abs_max_sub_max_le_abs (pureUpper n s f) (pureLower n s f) f
    rw [abs_of_nonneg (sub_nonneg.2 hle)] at this
    exact (le_abs_self _).trans this

/-- The pure-rung gap never exceeds `|s[0] - f|`, in or out of a window. -/
theorem pureGap_le_abs (s : ℕ → ℝ) (f : ℝ) (window : Bool) :
    pureGap n s f window ≤ |s 0 - f| :=
  (pureGap_le_unlifted s f window).trans (pureUpper_sub_pureLower_le s f)

end Ladder

/-! ## Nonnegative weights give an exact saddle -/

section Saddle

variable {n : ℕ} {s : ℕ → ℝ} {f : ℝ}

/-- The common payoff `w = f + (s[0] - f) / W` of the equalizing pair. -/
noncomputable def equalizerValue (n : ℕ) (s : ℕ → ℝ) (f : ℝ) : ℝ :=
  f + (s 0 - f) / weightSum n s f

/-- The Checker mixture is the reversal of the Dropper mixture `p = r / W`
(the C++ convention `q = reverse(p)` of `try_recurrence`). -/
theorem eqCol_eq_rev_eqRow (j : Fin n) : eqCol n s f j = eqRow n s f j.rev := by
  simp only [eqRow, Fin.rev_rev]

/-- **Nonnegative weights give an exact saddle point** (COMPACT-RUNG2-3,
CPP-CODE-REC-2, CANONICAL-MG-7; `architecture.md:73-77`, `BUILD.md:1-22`,
`dth_exact_solution.tex:182-192`; code `main.py:240-267`,
`matrix_game.cpp:376-392`, `fast_kernel.py:37-47`). With `d0 = s[0] - f ≠ 0`
and every `r[k] ≥ 0`: `W = ∑ r ≥ 1`; the Dropper mixture `p = r / W` and the
Checker mixture `q = reverse(r) / W = reverse(p)` lie in the simplex;
`M q = w·1` and `pᵀ M = w·1` for the same `w = f + (s[0] - f) / W`; hence
`pᵀ M q = w`, both certificate bounds equal `w`, the value is `w`, and
`(p, q)` is a saddle point (gap `0`). The equality `v = v'` of the two
equalized payoffs, which the paper leaves implicit, is the conjunct
`payoff ... = equalizerValue` together with `hrow` and `hcol`. -/
theorem nonneg_equalizer_saddle [NeZero n] (hd : s 0 - f ≠ 0)
    (hr : ∀ k < n, 0 ≤ weights s f k) :
    1 ≤ weightSum n s f ∧
      eqRow n s f ∈ MatrixGame.simplex (Fin n) ∧ eqCol n s f ∈ MatrixGame.simplex (Fin n) ∧
      (∀ i, (toeplitz n s f *ᵥ eqCol n s f) i = equalizerValue n s f) ∧
      (∀ j, (eqRow n s f ᵥ* toeplitz n s f) j = equalizerValue n s f) ∧
      MatrixGame.payoff (toeplitz n s f) (eqRow n s f) (eqCol n s f) = equalizerValue n s f ∧
      MatrixGame.lowerBound (toeplitz n s f) (eqRow n s f) = equalizerValue n s f ∧
      MatrixGame.upperBound (toeplitz n s f) (eqCol n s f) = equalizerValue n s f ∧
      MatrixGame.value (toeplitz n s f) = equalizerValue n s f ∧
      (∀ p ∈ MatrixGame.simplex (Fin n),
        MatrixGame.payoff (toeplitz n s f) p (eqCol n s f) ≤ equalizerValue n s f) ∧
      (∀ q ∈ MatrixGame.simplex (Fin n),
        equalizerValue n s f ≤ MatrixGame.payoff (toeplitz n s f) (eqRow n s f) q) := by
  have hn : 1 ≤ n := Nat.one_le_iff_ne_zero.mpr (NeZero.ne n)
  have hW := weightSum_ge_one s f hn hr
  have hq := eqCol_mem_simplex s f hn hr
  have hp : eqRow n s f ∈ MatrixGame.simplex (Fin n) := mem_simplex_rev hq
  have hrow : ∀ i, (toeplitz n s f *ᵥ eqCol n s f) i = equalizerValue n s f :=
    mulVec_eqCol s f hd hn (by linarith)
  have hcol : ∀ j, (eqRow n s f ᵥ* toeplitz n s f) j = equalizerValue n s f := by
    intro j
    rw [show eqRow n s f = fun i => eqCol n s f i.rev from rfl, vecMul_rev_eq_mulVec, hrow]
  obtain ⟨hv, hlo, hup⟩ := equalizer_value (n := n) s f hd hr
  have hpay : MatrixGame.payoff (toeplitz n s f) (eqRow n s f) (eqCol n s f) =
      equalizerValue n s f :=
    le_antisymm (MatrixGame.dotProduct_le_of_mem_simplex hp fun i => (hrow i).le)
      (MatrixGame.le_dotProduct_of_mem_simplex hp fun i => (hrow i).ge)
  refine ⟨hW, hp, hq, hrow, hcol, hpay, hlo, hup, hv, fun p hp' => ?_, fun q hq' => ?_⟩
  · exact MatrixGame.dotProduct_le_of_mem_simplex hp' fun i => (hrow i).le
  · rw [MatrixGame.payoff_eq_vecMul, dotProduct_comm]
    exact MatrixGame.le_dotProduct_of_mem_simplex hq' fun j => (hcol j).ge

/-- With nonnegative weights, clipping changes nothing. -/
theorem clip_eq_of_nonneg (hr : ∀ k < n, 0 ≤ weights s f k) :
    clipSum n s f = weightSum n s f ∧ clipCol n s f = eqCol n s f := by
  have hs : clipSum n s f = weightSum n s f :=
    Finset.sum_congr rfl fun k hk => max_eq_left (hr k (Finset.mem_range.1 hk))
  refine ⟨hs, funext fun j => ?_⟩
  simp only [clipCol, eqCol, hs]
  rw [max_eq_left (hr _ (by omega))]

/-- With nonnegative weights the full check sees a constant `M q`: its gap is
`0` and its mass is `1`, in or out of a window. -/
theorem fullGap_eq_zero_of_nonneg [NeZero n] (hd : s 0 - f ≠ 0)
    (hr : ∀ k < n, 0 ≤ weights s f k) (window : Bool) :
    fullGap n s f window = 0 ∧ clipMass n s f = 1 := by
  have hn : 1 ≤ n := Nat.one_le_iff_ne_zero.mpr (NeZero.ne n)
  obtain ⟨-, hcc⟩ := clip_eq_of_nonneg hr
  obtain ⟨-, -, hq, hrow, -⟩ := nonneg_equalizer_saddle hd hr
  have hfun : (fun i => (toeplitz n s f *ᵥ clipCol n s f) i) =
      fun _ => equalizerValue n s f := by
    funext i; rw [hcc, hrow]
  have hL : fullLower n s f = equalizerValue n s f := by
    simp only [fullLower, hfun, Finset.inf'_const]
  have hU : fullUpper n s f = equalizerValue n s f := by
    simp only [fullUpper, hfun, Finset.sup'_const]
  refine ⟨?_, ?_⟩
  · cases window <;> simp [fullGap, hL, hU]
  · simp only [clipMass, hcc]; exact hq.2

/-- **Nonnegative weights never reach the residue** (CRATES-CRASH-4(b),
exact-arithmetic version; `LEAP_CERTIFICATE.md:234`, `leap.rs:447-510`).
If every `r[k] ≥ 0`, the kernel route of `ReachesResidue` does not fire, in or
out of a leap window: a failed pure rung forces `|d| > 1e-6`, the clipped sum
is `W ≥ 1`, and the full check sees gap `0` and mass `1`. -/
theorem not_reachesResidue_of_nonneg [NeZero n] (hr : ∀ k < n, 0 ≤ weights s f k)
    (window : Bool) : ¬ ReachesResidue n s f window := by
  rintro ⟨hpure, hroute⟩
  have hn : 1 ≤ n := Nat.one_le_iff_ne_zero.mpr (NeZero.ne n)
  have hgap : gapTol < |s 0 - f| := hpure.trans_le (pureGap_le_abs s f window)
  have hd : s 0 - f ≠ 0 := by
    intro h; rw [h, abs_zero] at hgap; exact absurd hgap (by norm_num [gapTol])
  have hW := weightSum_ge_one s f hn hr
  obtain ⟨hcs, -⟩ := clip_eq_of_nonneg hr
  obtain ⟨hfg, hmass⟩ := fullGap_eq_zero_of_nonneg hd hr window
  rcases hroute with h | h | ⟨-, h | h⟩
  · have : dTol < gapTol := by norm_num [dTol, gapTol]
    linarith
  · linarith
  · rw [hmass, sub_self, abs_zero] at h; exact absurd h (by norm_num [dTol])
  · rw [hfg] at h; exact absurd h (by norm_num [gapTol])

end Saddle

/-! ## Nondecreasing success payoffs -/

section Monotone

variable {n : ℕ} {s : ℕ → ℝ} {f : ℝ}

theorem zero_le_of_monotone (hs : ∀ m, m + 1 < n → s m ≤ s (m + 1)) :
    ∀ k < n, s 0 ≤ s k := by
  intro k
  induction k with
  | zero => intro _; exact le_rfl
  | succ k ih => intro hk; exact (ih (by omega)).trans (hs k hk)

variable [NeZero n]

theorem succLo_of_monotone (hs : ∀ m, m + 1 < n → s m ≤ s (m + 1)) : succLo n s = s 0 :=
  le_antisymm (succLo_le_zero s)
    (Finset.le_inf' _ _ fun k _ => zero_le_of_monotone hs k.val k.isLt)

/-- **Pure saddle when `f ≤ s[0]`** (CRATES-KINK-2, implicit lemma): for a
nondecreasing `s` with `f ≤ s[0]`, both pure bounds equal `s[0]`, so the pure
rung's gap is `0`, in or out of a window (`leap.rs:383-384, 441-447`). -/
theorem pureGap_eq_zero_of_monotone (hs : ∀ m, m + 1 < n → s m ≤ s (m + 1)) (hf : f ≤ s 0)
    (window : Bool) : pureGap n s f window = 0 := by
  have hL : pureLower n s f = s 0 := by
    simp only [pureLower, succLo_of_monotone hs, min_eq_left hf, max_eq_left hf]
  have hU : pureUpper n s f = s 0 := by
    simp only [pureUpper, max_eq_right hf]; exact min_eq_right (zero_le_succHi s)
  cases window <;> simp [pureGap, hL, hU]

/-- **The value is `s[0]` when `s` is nondecreasing and `f ≤ s[0]`**
(CRATES-KINK-2, implicit lemma): `(0, 0)` is a pure saddle point. -/
theorem value_eq_of_monotone (hs : ∀ m, m + 1 < n → s m ≤ s (m + 1)) (hf : f ≤ s 0) :
    MatrixGame.value (toeplitz n s f) = s 0 := by
  have h := MatrixGame.value_of_pure_saddle (toeplitz n s f) (0 : Fin n) (0 : Fin n) ?_ ?_
  · simpa [toeplitz_apply] using h
  · intro j
    simp only [toeplitz_apply, Fin.zero_le, ↓reduceIte, Fin.val_zero, Nat.sub_zero, le_refl,
      Nat.sub_self]
    exact zero_le_of_monotone hs j.val j.isLt
  · intro i
    simp only [toeplitz_apply, Fin.zero_le, ↓reduceIte, Fin.val_zero, Nat.sub_self]
    split_ifs with h
    · simp
    · exact hf

theorem pureGap_le_of_monotone (hs : ∀ m, m + 1 < n → s m ≤ s (m + 1)) (hf : s 0 < f)
    (window : Bool) : pureGap n s f window ≤ f - s 0 := by
  have hL : pureLower n s f = s 0 := by
    simp only [pureLower, succLo_of_monotone hs, min_eq_right hf.le, max_self]
  have hU : pureUpper n s f ≤ f := (min_le_right _ _).trans (max_eq_left hf.le).le
  have := pureGap_le_unlifted (n := n) s f window
  linarith

/-- **The implemented ladder accepts every nondecreasing stage**
(STL-REC-2; `leap_support.py:28-32`, `leap.rs:441-475`), in exact
arithmetic. Either the pure gap is at most `1e-6` (always when `f ≤ s[0]`,
and when `0 < f - s[0] ≤ 1e-6` because the gap is at most `f - s[0]`), or the
weights are nonnegative, `|d| > 1e-6`, and `W ≥ 1`, which are the weight
conditions of the shortcut at `leap.rs:475`. -/
theorem monotone_ladder (hs : ∀ m, m + 1 < n → s m ≤ s (m + 1)) (window : Bool) :
    pureGap n s f window ≤ gapTol ∨
      ((∀ k < n, 0 ≤ weights s f k) ∧ gapTol < |s 0 - f| ∧ 1 ≤ weightSum n s f) := by
  have hn : 1 ≤ n := Nat.one_le_iff_ne_zero.mpr (NeZero.ne n)
  rcases le_or_gt f (s 0) with hf | hf
  · left; rw [pureGap_eq_zero_of_monotone hs hf]; norm_num [gapTol]
  · by_cases hsmall : f - s 0 ≤ gapTol
    · left; exact (pureGap_le_of_monotone hs hf window).trans hsmall
    · right
      have hr := weights_nonneg_of_monotone hs hf
      refine ⟨hr, ?_, weightSum_ge_one s f hn hr⟩
      rw [abs_sub_comm, abs_of_pos (by linarith)]
      linarith

/-- **A nondecreasing stage never reaches the residue** (CRATES-KINK-2,
STL-REC-2; `LEAP_CERTIFICATE.md:246-248`), in exact arithmetic, in or out of
a leap window. -/
theorem not_reachesResidue_of_monotone (hs : ∀ m, m + 1 < n → s m ≤ s (m + 1))
    (window : Bool) : ¬ ReachesResidue n s f window := by
  rcases le_or_gt f (s 0) with hf | hf
  · rintro ⟨hpure, -⟩
    rw [pureGap_eq_zero_of_monotone hs hf] at hpure
    exact absurd hpure (by norm_num [gapTol])
  · exact not_reachesResidue_of_nonneg (weights_nonneg_of_monotone hs hf) window

end Monotone

/-! ## The kink index -/

section KinkIndex

/-- The scan state of `kink_index` (`leap.rs:932-943`) after it has processed
the steps `k = 1..m`: the most negative step seen so far (starting at `0`)
and the index where it first occurred. A step replaces the state only when it
is strictly below the current best, so ties keep the first index. -/
noncomputable def kinkScan (s : ℕ → ℝ) : ℕ → ℝ × Option ℕ
  | 0 => (0, none)
  | m + 1 =>
    if s (m + 1) - s m < (kinkScan s m).1 then (s (m + 1) - s m, some (m + 1))
    else kinkScan s m

/-- `kink_index(s)` for `n` success payoffs (`n = 60` in the code): the loop
`for k in 1..A` of `leap.rs:932-943`. -/
noncomputable def kinkIndex (n : ℕ) (s : ℕ → ℝ) : Option ℕ := (kinkScan s (n - 1)).2

theorem kinkScan_inv (s : ℕ → ℝ) (m : ℕ) :
    (kinkScan s m).1 ≤ 0 ∧
      (∀ j, 1 ≤ j → j ≤ m → (kinkScan s m).1 ≤ s j - s (j - 1)) ∧
      ((kinkScan s m).2 = none → (kinkScan s m).1 = 0) ∧
      (∀ k, (kinkScan s m).2 = some k → 1 ≤ k ∧ k ≤ m ∧ s k - s (k - 1) = (kinkScan s m).1 ∧
        (kinkScan s m).1 < 0 ∧ ∀ j, 1 ≤ j → j < k → (kinkScan s m).1 < s j - s (j - 1)) := by
  induction m with
  | zero =>
    refine ⟨by simp [kinkScan], fun j h1 h2 => by omega, fun _ => by simp [kinkScan],
      fun k hk => by simp [kinkScan] at hk⟩
  | succ m ih =>
    obtain ⟨h1, h2, h3, h4⟩ := ih
    by_cases hlt : s (m + 1) - s m < (kinkScan s m).1
    · have e : kinkScan s (m + 1) = (s (m + 1) - s m, some (m + 1)) := by
        rw [kinkScan]; simp only [hlt, ↓reduceIte]
      rw [e]
      refine ⟨by dsimp only; linarith, ?_, fun h => by simp at h, ?_⟩
      · intro j hj1 hj2
        dsimp only
        rcases Nat.lt_or_ge j (m + 1) with hj | hj
        · have := h2 j hj1 (by omega); linarith
        · obtain rfl : j = m + 1 := by omega
          simp
      · intro k hk
        simp only [Option.some.injEq] at hk
        subst hk
        refine ⟨by omega, le_rfl, by simp, by dsimp only; linarith, ?_⟩
        intro j hj1 hj2
        dsimp only
        have := h2 j hj1 (by omega); linarith
    · have e : kinkScan s (m + 1) = kinkScan s m := by
        rw [kinkScan]; simp only [hlt, ↓reduceIte]
      rw [e]
      rw [not_lt] at hlt
      refine ⟨h1, ?_, h3, ?_⟩
      · intro j hj1 hj2
        rcases Nat.lt_or_ge j (m + 1) with hj | hj
        · exact h2 j hj1 (by omega)
        · obtain rfl : j = m + 1 := by omega
          simpa using hlt
      · intro k hk
        obtain ⟨a, b, c, d, e⟩ := h4 k hk
        exact ⟨a, by omega, c, d, e⟩

/-- **`kink_index` returns `None` exactly when no step is negative** (CRATES-KINK-2,
`leap.rs:928-943`, `leap_support.py:27-36`). -/
theorem kinkIndex_eq_none_iff (n : ℕ) (s : ℕ → ℝ) :
    kinkIndex n s = none ↔ ∀ j, 1 ≤ j → j < n → s (j - 1) ≤ s j := by
  have hI := kinkScan_inv s (n - 1)
  constructor
  · intro h j hj1 hj2
    have h0 := hI.2.2.1 h
    have := hI.2.1 j hj1 (by omega)
    linarith
  · intro h
    cases hk : kinkIndex n s with
    | none => rfl
    | some k =>
      obtain ⟨a, b, c, d, -⟩ := hI.2.2.2 k hk
      have := h k a (by omega)
      linarith

/-- **`kink_index` returns the first most negative step** (CRATES-KINK-2,
`leap.rs:928-943`). A returned index `k` lies in `1..n-1` (`1..59` for
`n = 60`, the range `kink_seed_rs` accepts at `leap.rs:1048-1063`), its step
`s[k] - s[k-1]` is negative and minimal, and every earlier step is strictly
larger. These properties determine `k`, so the NumPy version
(`np.argmin` over the steps, first index on a tie, then `None` unless the
minimum is negative, `leap_support.py:27-36`) returns the same index. -/
theorem kinkIndex_eq_some {n : ℕ} {s : ℕ → ℝ} {k : ℕ} (h : kinkIndex n s = some k) :
    1 ≤ k ∧ k < n ∧ s k < s (k - 1) ∧
      (∀ j, 1 ≤ j → j < n → s k - s (k - 1) ≤ s j - s (j - 1)) ∧
      (∀ j, 1 ≤ j → j < k → s k - s (k - 1) < s j - s (j - 1)) := by
  have hI := kinkScan_inv s (n - 1)
  obtain ⟨a, b, c, d, e⟩ := hI.2.2.2 k h
  refine ⟨a, by omega, by linarith, fun j hj1 hj2 => ?_, fun j hj1 hj2 => ?_⟩
  · rw [c]; exact hI.2.1 j hj1 (by omega)
  · rw [c]; exact e j hj1 hj2

/-- Concrete check: `s = (0, 0, -1)` has its kink at index `2`. -/
example : kinkIndex 3 (fun k => if k = 2 then (-1 : ℝ) else 0) = some 2 := by
  simp [kinkIndex, kinkScan]

/-- **A residue stage needs a negative step** (CRATES-KINK-2, STL-REC-2;
`LEAP_CERTIFICATE.md:246`, `leap.rs:928-931`, `leap_support.py:28-32`), in
exact real arithmetic: if the kernel route of `ReachesResidue` sends a stage
to the residue, in or out of a leap window, then `kink_index` returns an
index. The binary64 version is false; see `overflow_counterexample`. -/
theorem kinkIndex_isSome_of_reachesResidue {n : ℕ} [NeZero n] {s : ℕ → ℝ} {f : ℝ}
    {window : Bool} (h : ReachesResidue n s f window) : ∃ k, kinkIndex n s = some k := by
  cases hk : kinkIndex n s with
  | some k => exact ⟨k, rfl⟩
  | none =>
    have hmono : ∀ m, m + 1 < n → s m ≤ s (m + 1) := by
      intro m hm
      simpa using (kinkIndex_eq_none_iff n s).1 hk (m + 1) (by omega) hm
    exact absurd h (not_reachesResidue_of_monotone hmono window)

end KinkIndex

/-! ## Floating-point sign preservation -/

section Rounding

/-- The fused accumulation of `leap.rs:403-410`: for weight `k` the kernel
runs `acc ← fl(b[k-j] * r[j] + acc)` for `j = 0..k-1` from `acc = 0`.
`fmaFold R b r k j` is `acc` after `j` steps, with rounding function `R`. -/
def fmaFold (R : ℝ → ℝ) (b r : ℕ → ℝ) (k : ℕ) : ℕ → ℝ
  | 0 => 0
  | j + 1 => R (b (k - j) * r j + fmaFold R b r k j)

/-- A sequence `r̂` is a computed weight vector for kernel coefficients `b̂`
under rounding `R` when `r̂[0] = 1` and each `r̂[k]` is the fused fold. -/
def ComputedWeights (R : ℝ → ℝ) (b r : ℕ → ℝ) : Prop :=
  r 0 = 1 ∧ ∀ k, 1 ≤ k → r k = fmaFold R b r k k

/-- The computed kernel coefficient `b̂[m] = fl(fl(s[m-1] - s[m]) / d̂)`
(`leap.rs:392-397`), with `d̂ = fl(s[0] - f)`. -/
noncomputable def computedCoeff (R : ℝ → ℝ) (s : ℕ → ℝ) (f : ℝ) (m : ℕ) : ℝ :=
  R (R (s (m - 1) - s m) / R (s 0 - f))

theorem fmaFold_nonneg {R : ℝ → ℝ} (hR : Monotone R) (h0 : R 0 = 0) {b r : ℕ → ℝ} {k : ℕ}
    (hb : ∀ m, 0 ≤ b m) (hr : ∀ j < k, 0 ≤ r j) : ∀ j ≤ k, 0 ≤ fmaFold R b r k j := by
  intro j
  induction j with
  | zero => intro _; exact le_rfl
  | succ j ih =>
    intro hj
    rw [fmaFold, ← h0]
    exact hR (add_nonneg (mul_nonneg (hb _) (hr j (by omega))) (ih (by omega)))

/-- **Rounding preserves the sign of the weights** (CRATES-KINK-1, STL-REC-2;
`LEAP_CERTIFICATE.md:246-248`, kernel loop `leap.rs:392-413`). The rounding
model is an abstract function `R : ℝ → ℝ` that is monotone with `R 0 = 0`;
round-to-nearest in binary64 has both properties on its finite range. The
model has no overflow: `R` returns a real for every input. Under it, a
nondecreasing `s` with `R(s[0] - f) < 0` gives computed coefficients
`b̂[m] ≥ 0` and computed weights `r̂[k] ≥ 0`. In binary64 the hypothesis
"no overflow" can fail; `overflow_counterexample` shows a stage where the
exact weights leave the finite range. -/
theorem computedWeights_nonneg {R : ℝ → ℝ} (hR : Monotone R) (h0 : R 0 = 0)
    {n : ℕ} {s : ℕ → ℝ} {f : ℝ} (hs : ∀ m, m + 1 < n → s m ≤ s (m + 1))
    (hd : R (s 0 - f) < 0) {r : ℕ → ℝ}
    (hr : ComputedWeights R (fun m => if 1 ≤ m ∧ m < n then computedCoeff R s f m else 0) r) :
    (∀ m, 1 ≤ m → m < n → 0 ≤ computedCoeff R s f m) ∧ ∀ k, 0 ≤ r k := by
  have hb : ∀ m, 1 ≤ m → m < n → 0 ≤ computedCoeff R s f m := by
    intro m hm1 hm2
    have hnum : R (s (m - 1) - s m) ≤ 0 := by
      rw [← h0]; apply hR
      have := hs (m - 1) (by omega)
      rw [show m - 1 + 1 = m by omega] at this
      linarith
    rw [computedCoeff, ← h0]
    exact hR (div_nonneg_of_nonpos hnum hd.le)
  refine ⟨hb, ?_⟩
  intro k
  induction k using Nat.strong_induction_on with
  | _ k ih =>
    rcases Nat.eq_zero_or_pos k with rfl | hk
    · rw [hr.1]; exact zero_le_one
    · rw [hr.2 k hk]
      refine fmaFold_nonneg hR h0 (fun m => ?_) (fun j hj => ih j hj) k le_rfl
      split_ifs with h
      · exact hb m h.1 h.2
      · exact le_rfl

/-- Non-vacuity: with exact rounding and zero coefficients, `r̂ = (1, 0, 0, ...)`
is a computed weight vector. -/
example : ComputedWeights id (fun _ => 0) (fun k => if k = 0 then 1 else 0) := by
  refine ⟨rfl, fun k hk => ?_⟩
  have : ∀ j, fmaFold id (fun _ => (0 : ℝ)) (fun k => if k = 0 then 1 else 0) k j = 0 := by
    intro j; induction j with
    | zero => rfl
    | succ j ih => simp [fmaFold, ih]
  rw [this]; simp; omega

end Rounding

/-! ## The recurrence crash basis of the packing LP -/

section Crash

variable (s : ℕ → ℝ) (f : ℝ)

/-- The packing coefficients `a[k] = (f - s[k]) / d`, `d = f - s[0]`
(`Solver::coefficients`, `leap_packing.rs:293-300`). -/
noncomputable def packCoeff (k : ℕ) : ℝ := (f - s k) / (f - s 0)

/-- The upper-triangular Toeplitz packing matrix `A[i,j] = a[j-i]` for
`j ≥ i`, else `0` (`LEAP_CERTIFICATE.md:195-197`). -/
noncomputable def packMatrix (n : ℕ) : Matrix (Fin n) (Fin n) ℝ :=
  fun i j => if i ≤ j then packCoeff s f (j.val - i.val) else 0

/-- The crash series `c[0] = 1`, `c[k] = -∑_{m=1}^{k} a[m] c[k-m]`
(`Solver::crash`, `leap_packing.rs:113-121`). -/
noncomputable def crashSeries : ℕ → ℝ
  | 0 => 1
  | k + 1 => -∑ m : Fin (k + 1), packCoeff s f (m.val + 1) * crashSeries (k - m.val)
termination_by k => k
decreasing_by omega

/-- The basic value of packing variable `i` in the crash basis:
`y[i] = c[0] + ... + c[i]` (`prefix[i]`, `leap_packing.rs:122-135`). -/
noncomputable def crashPrimal (i : ℕ) : ℝ := ∑ l ∈ range (i + 1), crashSeries s f l

/-- The reduced cost of slack `j` in the crash basis,
`-(c[0] + ... + c[n-1-j])` (`leap_packing.rs:106-112`, `LEAP_CERTIFICATE.md:230-231`). -/
noncomputable def crashReduced (n j : ℕ) : ℝ := -crashPrimal s f (n - 1 - j)

variable {s f}

/-- `y` solves the unit lower-triangular Toeplitz system with first column `a`. -/
def LowerSolves (a y : ℕ → ℝ) : Prop := ∀ i, ∑ j ∈ range (i + 1), a (i - j) * y j = 1

theorem lowerSolves_unique {a y y' : ℕ → ℝ} (ha : a 0 = 1) (h : LowerSolves a y)
    (h' : LowerSolves a y') (i : ℕ) : y i = y' i := by
  induction i using Nat.strong_induction_on with
  | _ i ih =>
    have e := h i
    have e' := h' i
    rw [sum_range_succ, Nat.sub_self, ha, one_mul] at e e'
    have : ∑ j ∈ range i, a (i - j) * y j = ∑ j ∈ range i, a (i - j) * y' j :=
      sum_congr rfl fun j hj => by rw [ih j (mem_range.1 hj)]
    linarith

theorem packCoeff_zero (hd : f - s 0 ≠ 0) : packCoeff s f 0 = 1 := div_self hd

theorem weights_lowerSolves (hd : f - s 0 ≠ 0) : LowerSolves (packCoeff s f) (weights s f) := by
  intro i
  induction i with
  | zero => simp [packCoeff_zero hd, weights_zero]
  | succ i ih =>
    rw [sum_range_succ, Nat.sub_self, packCoeff_zero hd, one_mul]
    have hd' : s 0 - f ≠ 0 := fun h => hd (by linarith)
    have hrec := weights_rec s f hd' (a := i + 1) (by omega)
    have hrefl : ∑ m ∈ range (i + 1), (s (m + 1) - s m) * weights s f (i + 1 - 1 - m) =
        ∑ j ∈ range (i + 1), (s (i - j + 1) - s (i - j)) * weights s f j := by
      rw [← sum_range_reflect (fun j => (s (i - j + 1) - s (i - j)) * weights s f j) (i + 1)]
      refine sum_congr rfl fun m hm => ?_
      have hm := mem_range.1 hm
      have e1 : i - (i + 1 - 1 - m) = m := by omega
      simp only [e1]
    have key : ∑ j ∈ range (i + 1), packCoeff s f (i + 1 - j) * weights s f j =
        ∑ j ∈ range (i + 1), packCoeff s f (i - j) * weights s f j -
          (∑ j ∈ range (i + 1), (s (i - j + 1) - s (i - j)) * weights s f j) / (f - s 0) := by
      rw [sum_div, ← sum_sub_distrib]
      refine sum_congr rfl fun j hj => ?_
      have hj := mem_range.1 hj
      rw [show i + 1 - j = i - j + 1 by omega]
      unfold packCoeff
      field_simp
      ring
    rw [key, ih, ← hrefl]
    have : ∑ m ∈ range (i + 1), (s (m + 1) - s m) * weights s f (i + 1 - 1 - m) =
        -((s 0 - f) * weights s f (i + 1)) := by linarith
    rw [this]
    field_simp
    ring

theorem crash_conv (hd : f - s 0 ≠ 0) (k : ℕ) :
    ∑ l ∈ range (k + 2), packCoeff s f (k + 1 - l) * crashSeries s f l = 0 := by
  rw [sum_range_succ, Nat.sub_self, packCoeff_zero hd, one_mul, crashSeries,
    Fin.sum_univ_eq_sum_range (fun m => packCoeff s f (m + 1) * crashSeries s f (k - m)) (k + 1)]
  rw [← sum_range_reflect (fun m => packCoeff s f (m + 1) * crashSeries s f (k - m)) (k + 1)]
  rw [add_neg_eq_zero]
  refine sum_congr rfl fun l hl => ?_
  have hl := mem_range.1 hl
  have e1 : k + 1 - 1 - l + 1 = k + 1 - l := by omega
  have e2 : k - (k + 1 - 1 - l) = l := by omega
  simp only [e1, e2]

theorem crashPrimal_lowerSolves (hd : f - s 0 ≠ 0) :
    LowerSolves (packCoeff s f) (crashPrimal s f) := by
  intro i
  induction i with
  | zero => simp [crashPrimal, crashSeries, packCoeff_zero hd]
  | succ i ih =>
    have hc := crash_conv hd i
    rw [sum_range_succ'] at hc
    rw [sum_range_succ']
    have hP : ∀ j, crashPrimal s f (j + 1) = crashPrimal s f j + crashSeries s f (j + 1) := by
      intro j; simp only [crashPrimal]; rw [sum_range_succ]
    have hP0 : crashPrimal s f 0 = crashSeries s f 0 := by simp [crashPrimal]
    simp only [Nat.add_sub_add_right, hP, mul_add, sum_add_distrib, Nat.sub_zero, hP0] at hc ⊢
    linarith

/-- **The crash basic values are the recurrence weights** (CRATES-CRASH-4(a),
`LEAP_CERTIFICATE.md:231-235`): `c[0] + ... + c[k] = r[k]` for every `k`. -/
theorem crashPrimal_eq_weights (hd : f - s 0 ≠ 0) (k : ℕ) :
    crashPrimal s f k = weights s f k :=
  lowerSolves_unique (packCoeff_zero hd) (crashPrimal_lowerSolves hd) (weights_lowerSolves hd) k

theorem packMatrix_transpose_mulVec {n : ℕ} (y : ℕ → ℝ) (i : Fin n) :
    ((packMatrix s f n)ᵀ *ᵥ fun j : Fin n => y j) i =
      ∑ j ∈ range (i.val + 1), packCoeff s f (i.val - j) * y j := by
  have h := Fin.sum_univ_eq_sum_range
    (fun k => if k ≤ i.val then packCoeff s f (i.val - k) * y k else 0) n
  have hf : (range n).filter (· ≤ i.val) = range (i.val + 1) := by
    ext j; simp only [mem_filter, mem_range]; omega
  rw [← sum_filter (s := range n), hf] at h
  rw [← h]
  simp only [mulVec, dotProduct, transpose_apply, packMatrix, ite_mul, zero_mul]
  refine Fintype.sum_congr _ _ fun j => ?_
  by_cases h : j ≤ i
  · have h' : j.val ≤ i.val := h
    simp only [h, h', ↓reduceIte]
  · have h' : ¬ j.val ≤ i.val := h
    simp only [h, h', ↓reduceIte]

theorem packMatrix_rev (n : ℕ) (i j : Fin n) :
    packMatrix s f n j.rev i.rev = packMatrix s f n i j := by
  simp only [packMatrix, Fin.rev_le_rev, Fin.val_rev]
  split_ifs with h
  · congr 1; omega
  · rfl

/-- **The crash basis** (CRATES-CRASH-4(a); `LEAP_CERTIFICATE.md:219-235`,
`Solver::crash` at `leap_packing.rs:106-136`). With `d = f - s[0] ≠ 0`:
the basic values `y[i] = c[0] + ... + c[i]` equal the weights `r[i]` and
solve `Aᵀ y = 1`; the simplex multipliers `u[j] = r[n-1-j]` solve `A u = 1`;
the reduced cost of slack `j` is `-u[j] = -r[n-1-j]`. Hence the basis is
primal feasible (`y ≥ 0`) iff every `r[k] ≥ 0`, and dual feasible (every
slack reduced cost `≤ 0`) iff every `r[k] ≥ 0`. -/
theorem crash_basis (hd : f - s 0 ≠ 0) (n : ℕ) :
    (∀ i : Fin n, crashPrimal s f i = weights s f i) ∧
      (∀ j : Fin n, crashReduced s f n j = -weights s f (n - 1 - j)) ∧
      (packMatrix s f n)ᵀ *ᵥ (fun i : Fin n => crashPrimal s f i) = 1 ∧
      packMatrix s f n *ᵥ (fun j : Fin n => crashPrimal s f (n - 1 - j)) = 1 ∧
      ((∀ i : Fin n, 0 ≤ crashPrimal s f i) ↔ ∀ k < n, 0 ≤ weights s f k) ∧
      ((∀ j : Fin n, crashReduced s f n j ≤ 0) ↔ ∀ k < n, 0 ≤ weights s f k) := by
  have hy : (packMatrix s f n)ᵀ *ᵥ (fun i : Fin n => crashPrimal s f i) = 1 := by
    funext i
    rw [packMatrix_transpose_mulVec (crashPrimal s f) i, crashPrimal_lowerSolves hd]
    rfl
  refine ⟨fun i => crashPrimal_eq_weights hd i, fun j => ?_, hy, ?_, ?_, ?_⟩
  · simp only [crashReduced, crashPrimal_eq_weights hd]
  · funext i
    have h1 := congrFun hy i.rev
    simp only [mulVec, dotProduct, transpose_apply, Pi.one_apply] at h1 ⊢
    rw [← h1]
    refine Fintype.sum_equiv Fin.revPerm _ _ fun j => ?_
    simp only [Fin.revPerm_apply, Fin.val_rev]
    rw [packMatrix_rev n i j]
    congr 2
    omega
  · constructor
    · intro h k hk
      have := h ⟨k, hk⟩
      rwa [crashPrimal_eq_weights hd] at this
    · intro h i
      rw [crashPrimal_eq_weights hd]; exact h i i.isLt
  · constructor
    · intro h k hk
      have := h ⟨n - 1 - k, by omega⟩
      simp only [crashReduced, crashPrimal_eq_weights hd, neg_nonpos] at this
      rwa [show n - 1 - (n - 1 - k) = k by omega] at this
    · intro h j
      simp only [crashReduced, crashPrimal_eq_weights hd, neg_nonpos]
      exact h _ (by omega)

/-- The stage matrix splits as `M = f 11ᵀ - d A` (`LEAP_CERTIFICATE.md:195-197`). -/
theorem toeplitz_eq_pack (hd : f - s 0 ≠ 0) (n : ℕ) (i j : Fin n) :
    toeplitz n s f i j = f - (f - s 0) * packMatrix s f n i j := by
  simp only [toeplitz_apply, packMatrix, packCoeff]
  split_ifs
  · field_simp; ring
  · ring

/-- **The crash basis is optimal when the weights are nonnegative**
(CRATES-CRASH-4(a); `LEAP_CERTIFICATE.md:199-205, 231-235`). With `d ≠ 0`
and every `r[k] ≥ 0`: `y = r` is packing feasible, `x = reverse(r)` is
covering feasible, both objectives equal `W = ∑ r`, every packing-feasible
point has objective at most `W`, and every covering-feasible point has
objective at least `W`. So `z = W`, and the LP value `f - d / z` equals the
equalizer value `f + (s[0] - f) / W`. -/
theorem crash_optimal (hd : f - s 0 ≠ 0) {n : ℕ} (hr : ∀ k < n, 0 ≤ weights s f k) :
    let y : Fin n → ℝ := fun i => weights s f i
    let x : Fin n → ℝ := fun j => weights s f (n - 1 - j)
    (∀ i, 0 ≤ y i) ∧ (packMatrix s f n)ᵀ *ᵥ y = 1 ∧ ∑ i, y i = weightSum n s f ∧
      (∀ j, 0 ≤ x j) ∧ packMatrix s f n *ᵥ x = 1 ∧ ∑ j, x j = weightSum n s f ∧
      (∀ y' : Fin n → ℝ, (∀ i, 0 ≤ y' i) → (∀ j, ((packMatrix s f n)ᵀ *ᵥ y') j ≤ 1) →
        ∑ i, y' i ≤ weightSum n s f) ∧
      (∀ x' : Fin n → ℝ, (∀ j, 0 ≤ x' j) → (∀ i, 1 ≤ (packMatrix s f n *ᵥ x') i) →
        weightSum n s f ≤ ∑ j, x' j) ∧
      f - (f - s 0) / weightSum n s f = equalizerValue n s f := by
  intro y x
  obtain ⟨hP, -, hy, hx, -, -⟩ := crash_basis hd n
  have hy' : (packMatrix s f n)ᵀ *ᵥ y = 1 := by
    simpa only [y, ← crashPrimal_eq_weights hd] using hy
  have hx' : packMatrix s f n *ᵥ x = 1 := by
    simpa only [x, ← crashPrimal_eq_weights hd] using hx
  have hsy : ∑ i, y i = weightSum n s f := by
    simp only [y, weightSum]; exact Fin.sum_univ_eq_sum_range (weights s f) n
  have hsx : ∑ j, x j = weightSum n s f := by
    rw [← hsy]
    refine Fintype.sum_equiv Fin.revPerm _ _ fun j => ?_
    simp only [x, y, Fin.revPerm_apply, Fin.val_rev]
    congr 1; omega
  have duality : ∀ y' x' : Fin n → ℝ, (∀ i, 0 ≤ y' i) → (∀ j, ((packMatrix s f n)ᵀ *ᵥ y') j ≤ 1) →
      (∀ j, 0 ≤ x' j) → (∀ i, 1 ≤ (packMatrix s f n *ᵥ x') i) → ∑ i, y' i ≤ ∑ j, x' j := by
    intro y' x' hy0 hy1 hx0 hx1
    calc ∑ i, y' i ≤ ∑ i, y' i * (packMatrix s f n *ᵥ x') i :=
          sum_le_sum fun i _ => le_mul_of_one_le_right (hy0 i) (hx1 i)
      _ = ∑ j, ((packMatrix s f n)ᵀ *ᵥ y') j * x' j := by
          have := dotProduct_mulVec y' (packMatrix s f n) x'
          rw [mulVec_transpose]
          simpa only [dotProduct] using this
      _ ≤ ∑ j, x' j := sum_le_sum fun j _ => mul_le_of_le_one_left (hx0 j) (hy1 j)
  have hyn : ∀ i, 0 ≤ y i := fun i => hr i i.isLt
  have hxn : ∀ j, 0 ≤ x j := fun j => hr _ (by omega)
  refine ⟨hyn, hy', hsy, hxn, hx', hsx, fun y' h0 h1 => ?_, fun x' h0 h1 => ?_, ?_⟩
  · rw [← hsx]; exact duality y' x h0 h1 hxn (fun i => by rw [hx']; rfl)
  · rw [← hsy]; exact duality y x' hyn (fun j => by rw [hy']; rfl) h0 h1
  · simp only [equalizerValue]; ring

end Crash

/-! ## The binary64 overflow counterexample -/

section Overflow

/-- The success payoffs `s = (0, 1, 1, ..., 1)` of the counterexample. -/
def ovS (k : ℕ) : ℝ := if k = 0 then 0 else 1

/-- The failure payoff `f = 2e-6` of the counterexample. -/
noncomputable def ovF : ℝ := 2 / 10 ^ 6

theorem ovS_monotone : ∀ m, m + 1 < 60 → ovS m ≤ ovS (m + 1) := by
  intro m _; by_cases h : m = 0 <;> simp [ovS, h]

set_option exponentiation.threshold 1100 in
theorem two_pow_1024_lt : (2 : ℝ) ^ 1024 < 499500 ^ 55 := by norm_num

theorem ov_weights (k : ℕ) : weights ovS ovF k = 500000 ^ k := by
  rw [weights_geometric ovF (fun m hm => by unfold ovS; simp; omega)]
  norm_num [ovS, ovF]

set_option exponentiation.threshold 1100 in
/-- **The binary64 overflow counterexample** (settles the doubts of
CRATES-KINK-1, CRATES-KINK-2, CRATES-CRASH-4 and STL-REC-2). Take `n = 60`,
`s = (0, 1, ..., 1)` and `f = 2e-6`, outside a leap window. In exact
arithmetic:

* `s` is nondecreasing and `kink_index` returns `None`;
* the pure rung fails: its gap is `2e-6 > 1e-6`;
* `|d| = 2e-6` passes the `1e-12` guard, and the packing LP accepts `d > 1e-12`;
* the weights are `r[k] = 500000^k ≥ 0`, so the stage does not reach the
  residue (`not_reachesResidue_of_monotone`);
* `r[54] < 2^1024 < r[55]`, the crash series has `c[55] > 2^1024`, and the
  weight sum exceeds `2^1024`.

Every binary64 value at or above `2^1024` lies above the largest finite
double `(2^53 - 1) 2^971`, so the kernel cannot hold `r[55]` as a finite
number; `rounded_weights_overflow` proves this for any computed sequence with
per-step relative error at most `1e-3`. The kernel then sees a non-finite
weight sum (`leap.rs:453`) and sends the stage to the residue with no negative
step. The binary64 behavior itself (`r[55] = inf`, then `0 * inf = NaN` for
`r[57..59]`, so `x >= 0.0` fails) is a numerical observation, not a theorem
of this file. -/
theorem overflow_counterexample :
    (∀ m, m + 1 < 60 → ovS m ≤ ovS (m + 1)) ∧ kinkIndex 60 ovS = none ∧
      pureGap 60 ovS ovF false = 2 / 10 ^ 6 ∧ gapTol < pureGap 60 ovS ovF false ∧
      dTol ≤ |ovS 0 - ovF| ∧ dTol < ovF - ovS 0 ∧
      (∀ k, 0 ≤ weights ovS ovF k) ∧ ¬ ReachesResidue 60 ovS ovF false ∧
      weights ovS ovF 54 < 2 ^ 1024 ∧ (2 : ℝ) ^ 1024 < weights ovS ovF 55 ∧
      (2 : ℝ) ^ 1024 < crashSeries ovS ovF 55 ∧ (2 : ℝ) ^ 1024 < weightSum 60 ovS ovF ∧
      ((2 : ℝ) ^ 53 - 1) * 2 ^ 971 < 2 ^ 1024 := by
  have hmono := ovS_monotone
  have hgap : pureGap 60 ovS ovF false = 2 / 10 ^ 6 := by
    have hL : pureLower 60 ovS ovF = 0 := by
      have : succLo 60 ovS = 0 := by
        apply le_antisymm ((succLo_le_zero (n := 60) ovS).trans (by simp [ovS]))
        exact Finset.le_inf' _ _ fun k _ => by unfold ovS; split_ifs <;> norm_num
      simp only [pureLower, this]; norm_num [ovS, ovF]
    have hU : pureUpper 60 ovS ovF = 2 / 10 ^ 6 := by
      have : succHi 60 ovS = 1 := by
        apply le_antisymm (Finset.sup'_le _ _ fun k _ => by unfold ovS; split_ifs <;> norm_num)
        calc (1 : ℝ) = ovS ((1 : Fin 60) : ℕ) := by simp [ovS]
          _ ≤ succHi 60 ovS := le_succHi ovS 1
      simp only [pureUpper, this]; norm_num [ovS, ovF]
    simp [pureGap, hL, hU]
  have hd : ovF - ovS 0 ≠ 0 := by norm_num [ovS, ovF]
  have hc : crashSeries ovS ovF 55 = 500000 ^ 54 * 499999 := by
    have h1 := crashPrimal_eq_weights hd 55
    have h2 := crashPrimal_eq_weights hd 54
    simp only [crashPrimal, sum_range_succ (n := 55)] at h1
    simp only [crashPrimal] at h2
    rw [h2, ov_weights, ov_weights] at h1
    linarith
  refine ⟨hmono, (kinkIndex_eq_none_iff 60 ovS).2 fun j h1 h2 => ?_, hgap, ?_, ?_, ?_,
    fun k => by rw [ov_weights]; positivity, not_reachesResidue_of_monotone hmono false,
    ?_, ?_, ?_, ?_, ?_⟩
  · have := hmono (j - 1) (by omega); rwa [show j - 1 + 1 = j by omega] at this
  · rw [hgap]; norm_num [gapTol]
  · norm_num [dTol, ovS, ovF, abs_of_neg]
  · norm_num [dTol, ovS, ovF]
  · rw [ov_weights]; norm_num
  · rw [ov_weights]
    exact two_pow_1024_lt.trans (pow_lt_pow_left₀ (by norm_num) (by norm_num) (by norm_num))
  · rw [hc]; norm_num
  · calc (2 : ℝ) ^ 1024 < weights ovS ovF 55 := by
          rw [ov_weights]
          exact two_pow_1024_lt.trans (pow_lt_pow_left₀ (by norm_num) (by norm_num) (by norm_num))
      _ ≤ weightSum 60 ovS ovF := by
        unfold weightSum
        exact single_le_sum (fun k _ => by rw [ov_weights]; positivity)
          (mem_range.2 (by norm_num))
  · norm_num

/-- **Rounded weights overflow too** (settles the binary64 doubt of
CRATES-KINK-1 and CRATES-CRASH-4). Rounding model: each computed step of the
counterexample's recurrence is `r̂[k+1] = 500000 r̂[k] (1 + δ_k)` with
`|δ_k| ≤ 1e-3`. In the kernel the only nonzero coefficient is
`b̂[1] = fl(1 / fl(2e-6))`, so one step carries at most about `3 · 2^-53`
relative error, far inside the model. Then `r̂[55] > 2^1024`, above every
finite binary64 value. -/
theorem rounded_weights_overflow (r : ℕ → ℝ) (δ : ℕ → ℝ) (h0 : r 0 = 1)
    (hδ : ∀ k, |δ k| ≤ 1 / 1000) (hstep : ∀ k, r (k + 1) = 500000 * r k * (1 + δ k)) :
    (2 : ℝ) ^ 1024 < r 55 := by
  have hlow : ∀ k, 499500 ^ k ≤ r k := by
    intro k
    induction k with
    | zero => rw [h0]; norm_num
    | succ k ih =>
      rw [hstep, pow_succ]
      have := (abs_le.1 (hδ k)).1
      have hp : (0 : ℝ) ≤ 499500 ^ k := by positivity
      nlinarith
  exact lt_of_lt_of_le two_pow_1024_lt (hlow 55)

/-- The rounding model of `rounded_weights_overflow` is satisfiable: exact
arithmetic (`δ = 0`) gives the exact weights. -/
example : (2 : ℝ) ^ 1024 < weights ovS ovF 55 :=
  rounded_weights_overflow (weights ovS ovF) (fun _ => 0) (weights_zero _ _)
    (fun _ => by norm_num) (fun k => by rw [ov_weights, ov_weights, pow_succ]; ring)

end Overflow

/-! ## Clipping can break the equilibrium -/

section Clip

/-- **A clipped candidate need not be an equilibrium** (settles the doubts of
COMPACT-RUNG2-3 and CANONICAL-MG-7). For `n = 2`, `s = (0, 1)` and `f = -1`
the weights are `r = (1, -1)`. Clipping gives the Checker mixture
`q = (0, 1)` with `M q = (1, 0)`, so the full-check gap is `1 > 1e-6`, while
the value is `0` (a pure saddle at `(0, 0)`). -/
theorem clip_counterexample :
    weights ovS (-1) 1 = -1 ∧
      (toeplitz 2 ovS (-1) *ᵥ clipCol 2 ovS (-1)) 0 = 1 ∧
      (toeplitz 2 ovS (-1) *ᵥ clipCol 2 ovS (-1)) 1 = 0 ∧
      gapTol < fullGap 2 ovS (-1) false ∧
      MatrixGame.value (toeplitz 2 ovS (-1)) = 0 := by
  have hw1 : weights ovS (-1) 1 = -1 := by
    rw [weights_succ]; simp [ovS, weights_zero]
  have hcs : clipSum 2 ovS (-1) = 1 := by
    simp [clipSum, sum_range_succ, weights_zero, hw1]
  have hq0 : clipCol 2 ovS (-1) 0 = 0 := by simp [clipCol, hcs, hw1]
  have hq1 : clipCol 2 ovS (-1) 1 = 1 := by simp [clipCol, hcs, weights_zero]
  have hm0 : (toeplitz 2 ovS (-1) *ᵥ clipCol 2 ovS (-1)) 0 = 1 := by
    simp [mulVec, dotProduct, Fin.sum_univ_two, hq1, toeplitz_apply, ovS]
  have hm1 : (toeplitz 2 ovS (-1) *ᵥ clipCol 2 ovS (-1)) 1 = 0 := by
    simp [mulVec, dotProduct, Fin.sum_univ_two, hq0, toeplitz_apply, ovS]
  refine ⟨hw1, hm0, hm1, ?_, ?_⟩
  · have hU : (toeplitz 2 ovS (-1) *ᵥ clipCol 2 ovS (-1)) 0 ≤ fullUpper 2 ovS (-1) :=
      Finset.le_sup' (fun i => (toeplitz 2 ovS (-1) *ᵥ clipCol 2 ovS (-1)) i) (mem_univ 0)
    have hL : fullLower 2 ovS (-1) ≤ (toeplitz 2 ovS (-1) *ᵥ clipCol 2 ovS (-1)) 1 :=
      Finset.inf'_le (fun i => (toeplitz 2 ovS (-1) *ᵥ clipCol 2 ovS (-1)) i) (mem_univ 1)
    simp only [fullGap, Bool.false_eq_true, ↓reduceIte, gapTol]
    rw [hm0] at hU; rw [hm1] at hL
    norm_num; linarith
  · have := value_eq_of_monotone (n := 2) (s := ovS) (f := -1)
      (fun m hm => by by_cases h : m = 0 <;> simp [ovS, h]) (by simp [ovS])
    simpa [ovS] using this

end Clip

end Formal.Toeplitz
