import Formal.DTH.Rules

/-!
# The Toeplitz stage matrix

Every DTH and STL stage game has the shape `M[d, c] = s[c - d]` for `c ≥ d`
and `M[d, c] = f` for `c < d`, with rows and columns indexed from `0`
(`full_matrix` in `src/dth_compact/main.py`; `docs/FOUNDATIONS.md`;
`src/crates/docs/LEAP_CERTIFICATE.md`). Index `k` of `s` is the success
payoff for lag `k + 1`; `f` is the failed-check payoff.

This module proves the structural facts that the solver shortcuts rely on:

* `toeplitz_persymmetric`: `M[n-1-c, n-1-d] = M[d, c]`;
* `vecMul_rev_eq_mulVec`: for the mirrored row strategy `p = reverse q`,
  `(pᵀ M)_j = (M q)_{n-1-j}` whatever `q` is;
* `mulVec_toeplitz`: `(M q)_i = f ∑_{j<i} q_j + ∑_{k≥i} s[k-i] q_k`;
* `mulVec_sub_succ`: adjacent rows differ by
  `(s[0] - f) q_i + ∑_{k>i} (s[k-i] - s[k-i-1]) q_k`;
* `DTH.stage_eq_toeplitz`: the DTH stage matrix is this matrix.
-/

open Finset Matrix

namespace Formal.Toeplitz

variable {n : ℕ}

/-- The stage matrix `M[d, c] = s[c - d]` for `d ≤ c`, else `f`. -/
def toeplitz (n : ℕ) (s : ℕ → ℝ) (f : ℝ) : Matrix (Fin n) (Fin n) ℝ :=
  fun d c => if d ≤ c then s (c.val - d.val) else f

theorem toeplitz_apply (s : ℕ → ℝ) (f : ℝ) (d c : Fin n) :
    toeplitz n s f d c = if d ≤ c then s (c.val - d.val) else f := rfl

/-- The stage matrix is persymmetric: it is symmetric about its anti-diagonal. -/
theorem toeplitz_persymmetric (s : ℕ → ℝ) (f : ℝ) (d c : Fin n) :
    toeplitz n s f c.rev d.rev = toeplitz n s f d c := by
  simp only [toeplitz_apply, Fin.rev_le_rev, Fin.val_rev]
  split_ifs with h
  · congr 1; omega
  · rfl

/-- The mirror identity behind the rung-2 certificate: for `p = reverse q`,
`(pᵀ M)_j = (M q)_{n-1-j}` for every `q`, clipped or not. -/
theorem vecMul_rev_eq_mulVec (s : ℕ → ℝ) (f : ℝ) (q : Fin n → ℝ) (j : Fin n) :
    ((fun i => q i.rev) ᵥ* toeplitz n s f) j = (toeplitz n s f *ᵥ q) j.rev := by
  simp only [vecMul, mulVec, dotProduct]
  refine Fintype.sum_equiv Fin.revPerm _ _ fun i => ?_
  rw [Fin.revPerm_apply, toeplitz_persymmetric, mul_comm]

/-- Row `i` of `M q`. -/
theorem mulVec_toeplitz (s : ℕ → ℝ) (f : ℝ) (q : Fin n → ℝ) (i : Fin n) :
    (toeplitz n s f *ᵥ q) i =
      f * ∑ j ∈ univ.filter (· < i), q j + ∑ k ∈ univ.filter (i ≤ ·), s (k.val - i.val) * q k := by
  simp only [mulVec, dotProduct, toeplitz_apply, Finset.mul_sum]
  rw [← Finset.sum_filter_add_sum_filter_not univ (i ≤ ·)]
  rw [add_comm]
  congr 1
  · refine Finset.sum_congr (by ext j; simp) fun j hj => ?_
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hj
    simp [not_le.mpr hj]
  · refine Finset.sum_congr rfl fun k hk => ?_
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hk
    simp [hk]

/-- Adjacent rows of `M q` differ by the diagonal step `s[0] - f` and the
success steps `s[m] - s[m-1]` above the diagonal (paper §4, proof of the
recurrence; `LEAP_CERTIFICATE.md`, recurrence residual). -/
theorem mulVec_sub_succ (s : ℕ → ℝ) (f : ℝ) (q : Fin n → ℝ) (i : Fin n) (hi : i.val + 1 < n) :
    (toeplitz n s f *ᵥ q) i - (toeplitz n s f *ᵥ q) ⟨i.val + 1, hi⟩ =
      (s 0 - f) * q i +
        ∑ k ∈ univ.filter (i < ·), (s (k.val - i.val) - s (k.val - i.val - 1)) * q k := by
  simp only [mulVec, dotProduct, toeplitz_apply]
  rw [← Finset.sum_sub_distrib]
  rw [← Finset.add_sum_erase _ _ (mem_univ i)]
  have hsplit : ∑ x ∈ univ.erase i,
        ((if i ≤ x then s (x.val - i.val) else f) * q x -
          (if (⟨i.val + 1, hi⟩ : Fin n) ≤ x then s (x.val - (i.val + 1)) else f) * q x) =
      ∑ k ∈ univ.filter (i < ·), (s (k.val - i.val) - s (k.val - i.val - 1)) * q k := by
    rw [← Finset.sum_filter_add_sum_filter_not (univ.erase i) (i < ·)]
    have hzero : ∑ x ∈ (univ.erase i).filter (fun x => ¬ i < x),
        ((if i ≤ x then s (x.val - i.val) else f) * q x -
          (if (⟨i.val + 1, hi⟩ : Fin n) ≤ x then s (x.val - (i.val + 1)) else f) * q x) = 0 := by
      refine Finset.sum_eq_zero fun x hx => ?_
      simp only [Finset.mem_filter, Finset.mem_erase, Finset.mem_univ, and_true, not_lt] at hx
      have h1 : ¬ i ≤ x := fun h => hx.1 (le_antisymm hx.2 h)
      have h2 : ¬ (⟨i.val + 1, hi⟩ : Fin n) ≤ x := by
        rw [Fin.le_def]; simp only; have := Fin.le_def.mp hx.2; omega
      simp [h1, h2]
    rw [hzero, add_zero]
    refine Finset.sum_congr (by ext k; simp; intro h; exact ne_of_gt h) fun k hk => ?_
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hk
    have h1 : i ≤ k := le_of_lt hk
    have h2 : (⟨i.val + 1, hi⟩ : Fin n) ≤ k := by
      rw [Fin.le_def]; simp only; exact Fin.lt_def.mp hk
    simp only [h1, h2, ite_true]
    rw [show k.val - (i.val + 1) = k.val - i.val - 1 by omega]
    ring
  rw [hsplit]
  have h1 : ¬ (⟨i.val + 1, hi⟩ : Fin n) ≤ i := by rw [Fin.le_def]; simp
  simp only [le_refl, ite_true, h1, ite_false, Nat.sub_self]
  ring

end Formal.Toeplitz

namespace Formal.DTH

/-- The DTH stage matrix is the Toeplitz stage matrix with `s[k] = S_{k+1}`
and `f = F`. -/
theorem stage_eq_toeplitz (W : State → ℝ) (x : State) :
    stage W x = Toeplitz.toeplitz 60 (fun k => succPay W x (k + 1)) (failPay W x) := by
  funext d c
  simp only [stage, Toeplitz.toeplitz_apply]

end Formal.DTH
