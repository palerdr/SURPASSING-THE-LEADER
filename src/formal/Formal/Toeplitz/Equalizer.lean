import Formal.Toeplitz.Basic

/-!
# The Toeplitz equalizer recurrence (paper §4, rung 2)

`try_rung2` in `src/dth_compact/main.py` computes, with `d0 = s[0] - f` and
`dS[m] = s[m+1] - s[m]`,

```text
r[0] = 1,   r[k] = -(∑_{m<k} dS[m] r[k-1-m]) / d0   (1 ≤ k < n),
```

then the Checker mixture `q = reverse(r) / ∑ r` and the Dropper mixture
`p = reverse(q)`. The paper states the same recurrence with 1-based indices.

This module proves:

* `mulVec_sub_succ_eqCol`: the recurrence makes adjacent rows of `M q` equal;
* `mulVec_eqCol`: every row of `M q` pays `f + (s[0] - f) / W`, `W = ∑ r`;
* `equalizer_value`: when every weight is nonnegative, `q` and `p` are
  optimal and the value is `f + (s[0] - f) / W`;
* `mirror_certificate`: for any Checker mixture `q` and its mirror `p`, the
  minimum and maximum of `M q` are the two certificate bounds, so a clipped
  mixture still yields a sound enclosure (`try_rung2`'s docstring).
-/

open Finset Matrix

namespace Formal.Toeplitz

/-- The recurrence weights `r[k]` of `try_rung2`. -/
noncomputable def weights (s : ℕ → ℝ) (f : ℝ) : ℕ → ℝ
  | 0 => 1
  | k + 1 => -(∑ m : Fin (k + 1), (s (m.val + 1) - s m.val) * weights s f (k - m.val)) / (s 0 - f)
termination_by k => k
decreasing_by omega

theorem weights_zero (s : ℕ → ℝ) (f : ℝ) : weights s f 0 = 1 := by
  rw [weights]

/-- The recurrence in the row-difference form of the paper's proof:
`d0 r[a] + ∑_{m<a} dS[m] r[a-1-m] = 0` for `a ≥ 1`. -/
theorem weights_rec (s : ℕ → ℝ) (f : ℝ) (hd : s 0 - f ≠ 0) {a : ℕ} (ha : 1 ≤ a) :
    (s 0 - f) * weights s f a +
      ∑ m ∈ range a, (s (m + 1) - s m) * weights s f (a - 1 - m) = 0 := by
  obtain ⟨k, rfl⟩ : ∃ k, a = k + 1 := ⟨a - 1, by omega⟩
  rw [weights, mul_div_cancel₀ _ hd, Nat.add_sub_cancel]
  rw [← Fin.sum_univ_eq_sum_range (fun m => (s (m + 1) - s m) * weights s f (k - m))]
  ring

/-- The weight sum `W = ∑_{k<n} r[k]`. -/
noncomputable def weightSum (n : ℕ) (s : ℕ → ℝ) (f : ℝ) : ℝ := ∑ k ∈ range n, weights s f k

/-- The Checker mixture `q[j] = r[n-1-j] / W`. -/
noncomputable def eqCol (n : ℕ) (s : ℕ → ℝ) (f : ℝ) : Fin n → ℝ :=
  fun j => weights s f (n - 1 - j.val) / weightSum n s f

/-- The mirrored Dropper mixture `p[i] = q[n-1-i] = r[i] / W`. -/
noncomputable def eqRow (n : ℕ) (s : ℕ → ℝ) (f : ℝ) : Fin n → ℝ :=
  fun i => eqCol n s f i.rev

theorem eqRow_apply (n : ℕ) (s : ℕ → ℝ) (f : ℝ) (i : Fin n) :
    eqRow n s f i = weights s f i.val / weightSum n s f := by
  simp only [eqRow, eqCol, Fin.val_rev]
  congr 2
  omega

/-- The recurrence equalizes adjacent rows of `M q`. -/
theorem mulVec_sub_succ_eqCol {n : ℕ} (s : ℕ → ℝ) (f : ℝ) (hd : s 0 - f ≠ 0) (i : Fin n)
    (hi : i.val + 1 < n) :
    (toeplitz n s f *ᵥ eqCol n s f) i - (toeplitz n s f *ᵥ eqCol n s f) ⟨i.val + 1, hi⟩ = 0 := by
  rw [mulVec_sub_succ]
  set a := n - 1 - i.val with ha
  have ha1 : 1 ≤ a := by omega
  have key := weights_rec s f hd ha1
  -- reindex the strict upper sum by `m = k - i - 1`
  have hsum : ∑ k ∈ univ.filter (i < ·), (s (k.val - i.val) - s (k.val - i.val - 1)) * eqCol n s f k =
      (∑ m ∈ range a, (s (m + 1) - s m) * weights s f (a - 1 - m)) / weightSum n s f := by
    rw [Finset.sum_div]
    refine Finset.sum_bij' (fun k _ => k.val - i.val - 1)
      (fun m hm => ⟨i.val + 1 + m, by simp only [mem_range] at hm; omega⟩) ?_ ?_ ?_ ?_ ?_
    · intro k hk
      simp only [mem_filter, mem_univ, true_and] at hk
      simp only [mem_range]; have := Fin.lt_def.mp hk; omega
    · intro m hm
      simp only [mem_filter, mem_univ, true_and, Fin.lt_def]; omega
    · intro k hk
      simp only [mem_filter, mem_univ, true_and] at hk
      ext; simp only; have := Fin.lt_def.mp hk; omega
    · intro m hm; simp only [mem_range] at hm; dsimp only; omega
    · intro k hk
      simp only [mem_filter, mem_univ, true_and] at hk
      have hk' := Fin.lt_def.mp hk
      simp only [eqCol, mul_div_assoc]
      rw [show k.val - i.val = k.val - i.val - 1 + 1 by omega,
        show k.val - i.val - 1 + 1 - 1 = k.val - i.val - 1 by omega,
        show n - 1 - k.val = a - 1 - (k.val - i.val - 1) by omega]
  rw [hsum]
  simp only [eqCol]
  rw [show n - 1 - i.val = a from rfl, mul_div_assoc', ← add_div, key, zero_div]

/-- Every row of `M q` pays `f + (s[0] - f) / W`. -/
theorem mulVec_eqCol {n : ℕ} (s : ℕ → ℝ) (f : ℝ) (hd : s 0 - f ≠ 0) (hn : 1 ≤ n)
    (hW : weightSum n s f ≠ 0) (i : Fin n) :
    (toeplitz n s f *ᵥ eqCol n s f) i = f + (s 0 - f) / weightSum n s f := by
  -- downward induction from the last row
  suffices h : ∀ t, ∀ i : Fin n, i.val + t = n - 1 →
      (toeplitz n s f *ᵥ eqCol n s f) i = f + (s 0 - f) / weightSum n s f by
    exact h (n - 1 - i.val) i (by omega)
  intro t
  induction t with
  | zero =>
    intro i hi
    rw [mulVec_toeplitz]
    have hfilter : univ.filter (fun k : Fin n => i ≤ k) = {i} := by
      ext k; simp only [mem_filter, mem_univ, true_and, mem_singleton]
      constructor
      · intro h; ext; have := Fin.le_def.mp h; have := k.isLt; omega
      · rintro rfl; exact le_rfl
    have hlt : univ.filter (fun j : Fin n => j < i) = univ.erase i := by
      ext j; simp only [mem_filter, mem_univ, true_and, mem_erase, and_true]
      constructor
      · exact ne_of_lt
      · intro h; rw [Fin.lt_def]; have := j.isLt; have : j.val ≠ i.val := fun e => h (Fin.ext e)
        omega
    have htotal : ∑ j, eqCol n s f j = 1 := by
      simp only [eqCol, ← Finset.sum_div]
      rw [div_eq_one_iff_eq hW, weightSum]
      rw [← Fin.sum_univ_eq_sum_range]
      exact Fintype.sum_equiv Fin.revPerm _ _ fun j => by
        simp only [Fin.revPerm_apply, Fin.val_rev]; congr 1; omega
    have hi' : eqCol n s f i = 1 / weightSum n s f := by
      simp only [eqCol]; rw [show n - 1 - i.val = 0 by omega, weights_zero]
    rw [hfilter, hlt, Finset.sum_singleton, Nat.sub_self, Finset.sum_erase_eq_sub (mem_univ i),
      htotal, hi']
    field_simp
    ring
  | succ t ih =>
    intro i hi
    have hi1 : i.val + 1 < n := by omega
    have := mulVec_sub_succ_eqCol s f hd i hi1
    rw [ih ⟨i.val + 1, hi1⟩ (by simp only; omega)] at this
    linarith

theorem weightSum_ge_one {n : ℕ} (s : ℕ → ℝ) (f : ℝ) (hn : 1 ≤ n)
    (hr : ∀ k < n, 0 ≤ weights s f k) : 1 ≤ weightSum n s f := by
  unfold weightSum
  obtain ⟨m, rfl⟩ : ∃ m, n = m + 1 := ⟨n - 1, by omega⟩
  rw [Finset.sum_range_succ', weights_zero]
  have : 0 ≤ ∑ k ∈ range m, weights s f (k + 1) :=
    Finset.sum_nonneg fun k hk => hr _ (by simp only [mem_range] at hk; omega)
  linarith

theorem eqCol_mem_simplex {n : ℕ} (s : ℕ → ℝ) (f : ℝ) (hn : 1 ≤ n)
    (hr : ∀ k < n, 0 ≤ weights s f k) : eqCol n s f ∈ MatrixGame.simplex (Fin n) := by
  have hW := weightSum_ge_one s f hn hr
  refine ⟨fun j => div_nonneg (hr _ (by omega)) (by linarith), ?_⟩
  simp only [eqCol, ← Finset.sum_div]
  rw [div_eq_one_iff_eq (by linarith), weightSum, ← Fin.sum_univ_eq_sum_range]
  exact Fintype.sum_equiv Fin.revPerm _ _ fun j => by
        simp only [Fin.revPerm_apply, Fin.val_rev]; congr 1; omega

theorem mem_simplex_rev {n : ℕ} {q : Fin n → ℝ} (hq : q ∈ MatrixGame.simplex (Fin n)) :
    (fun i => q i.rev) ∈ MatrixGame.simplex (Fin n) :=
  ⟨fun i => hq.1 _, by rw [← hq.2]; exact Fintype.sum_equiv Fin.revPerm _ _ fun _ => rfl⟩

/-- **Mirror certificate** (rung 2): for any Checker mixture `q` and its mirror
`p = reverse q`, `lowerBound p = min (M q)` and `upperBound q = max (M q)`, so
`[min (M q), max (M q)]` encloses the value, clipped weights or not. -/
theorem mirror_certificate {n : ℕ} [NeZero n] (s : ℕ → ℝ) (f : ℝ) {q : Fin n → ℝ}
    (hq : q ∈ MatrixGame.simplex (Fin n)) :
    MatrixGame.lowerBound (toeplitz n s f) (fun i => q i.rev) =
        univ.inf' univ_nonempty (fun i => (toeplitz n s f *ᵥ q) i) ∧
      MatrixGame.lowerBound (toeplitz n s f) (fun i => q i.rev) ≤ MatrixGame.value (toeplitz n s f) ∧
      MatrixGame.value (toeplitz n s f) ≤ MatrixGame.upperBound (toeplitz n s f) q := by
  have hmin : MatrixGame.lowerBound (toeplitz n s f) (fun i => q i.rev) =
      univ.inf' univ_nonempty (fun i => (toeplitz n s f *ᵥ q) i) := by
    unfold MatrixGame.lowerBound
    simp only [vecMul_rev_eq_mulVec]
    apply le_antisymm
    · exact Finset.le_inf' _ _ fun i _ =>
        (Finset.inf'_le (fun j : Fin n => (toeplitz n s f *ᵥ q) j.rev) (mem_univ i.rev)).trans_eq
          (by simp only [Fin.rev_rev])
    · exact Finset.le_inf' _ _ fun j _ => Finset.inf'_le _ (mem_univ _)
  exact ⟨hmin, (MatrixGame.certificate_encloses_value _ (mem_simplex_rev hq) hq).1,
    (MatrixGame.certificate_encloses_value _ (mem_simplex_rev hq) hq).2⟩

/-- **Equalizer theorem** (paper §4): when every recurrence weight is
nonnegative, the mixtures `q = reverse(r) / W` and `p = r / W` are optimal and
the value is `f + (s[0] - f) / W`. -/
theorem equalizer_value {n : ℕ} [NeZero n] (s : ℕ → ℝ) (f : ℝ) (hd : s 0 - f ≠ 0)
    (hr : ∀ k < n, 0 ≤ weights s f k) :
    MatrixGame.value (toeplitz n s f) = f + (s 0 - f) / weightSum n s f ∧
      MatrixGame.lowerBound (toeplitz n s f) (eqRow n s f) = f + (s 0 - f) / weightSum n s f ∧
      MatrixGame.upperBound (toeplitz n s f) (eqCol n s f) = f + (s 0 - f) / weightSum n s f := by
  have hn : 1 ≤ n := Nat.one_le_iff_ne_zero.mpr (NeZero.ne n)
  have hW := weightSum_ge_one s f hn hr
  have hq := eqCol_mem_simplex s f hn hr
  have hp : eqRow n s f ∈ MatrixGame.simplex (Fin n) := mem_simplex_rev hq
  have hrow := mulVec_eqCol s f hd hn (by linarith) (n := n)
  have hcol : ∀ j, (eqRow n s f ᵥ* toeplitz n s f) j = f + (s 0 - f) / weightSum n s f := by
    intro j
    rw [show eqRow n s f = fun i => eqCol n s f i.rev from rfl, vecMul_rev_eq_mulVec, hrow]
  obtain ⟨-, hv⟩ := MatrixGame.value_of_equalizers _ hp hq hcol hrow
  refine ⟨hv, ?_, ?_⟩
  · apply le_antisymm
    · exact (MatrixGame.lowerBound_le_col _ _ ⟨0, by omega⟩).trans_eq (hcol _)
    · exact (MatrixGame.le_lowerBound_iff _ _ _).2 fun j => (hcol j).ge
  · apply le_antisymm
    · exact (MatrixGame.upperBound_le_iff _ _ _).2 fun i => (hrow i).le
    · exact (hrow ⟨0, by omega⟩).symm.le.trans (MatrixGame.row_le_upperBound _ _ _)

end Formal.Toeplitz
