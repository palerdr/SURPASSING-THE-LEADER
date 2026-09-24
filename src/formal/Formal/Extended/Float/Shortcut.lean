import Mathlib
import Formal.Toeplitz.Equalizer

/-!
# The nonnegative-weight shortcut certificate

This file proves the certificate of `src/crates/docs/LEAP_CERTIFICATE.md:1-118`
for the rung-1 shortcut of `leap.rs:475-479` and `leap_oracle.py:99-101`: when
the computed recurrence weights are finite and nonnegative, the kernel stores
`v = fl(F + fl(d̂ / Ŵ))` with radius `CERT_RADIUS = 1e-10` and never forms `M q`.

## Rounding model

We state every floating-point claim over `ℝ`. One rounded binary64 operation
with exact result `x` returns `y` with `Rounds x y`, meaning
`y = x (1 + δ) + e`, `|δ| ≤ u = 2^-53`, `|e| ≤ η = 2^-1075`. This is the
standard model for round-to-nearest with gradual underflow and no overflow.
We do not model IEEE bit patterns. A fused multiply-add is one rounding; a
separate multiply and add is two. `Run` records every computed quantity and
`Run.Valid` records the rounding of each one; the gathered payoffs `s`, `f` are
exact inputs, as the doc states.

## Conventions

Actions are zero-based. `toeplitz 60 s f` is the stage matrix
`M[i, j] = s[j - i]` for `i ≤ j`, else `f`; rows are the Dropper, who
maximizes. `q_j = r_{59-j} / W` and `p_j = r_j / W` use the exact sum
`W = ∑ r_j` of the computed weights.

## Results

* `dot_error`: a rounded dot product of length `n` has relative error
  `(1+u)^(n+1) - 1 ≤ γ_{n+1}` and absolute error `2n(1+u)^n η`.
* `residual_decomposition` (CRATES-CERT-1) and `Run.Valid.residual_le`
  (CRATES-CERT-2): each row residual divided by `W` is at most
  `4γ_118(1+γ_2) + 4u + 4γ_2 + 500η < 6e-14`.
* `Run.Valid.spread` (CRATES-CERT-3): `max (M q) - min (M q) ≤ 59 ρ < 3.54e-12`.
* `Run.Valid.last_row_close` (CRATES-CERT-4): `|v - (M q)_59| < 4e-14`.
* `Run.Valid.enclosure` (CRATES-CERT-5, STL-FP-1): both saddle bounds lie within
  `4e-12` of `v`, and `[v - 1e-10, v + 1e-10]` encloses them and the value.
* `equalizer_of_recurrence`, `value_of_recurrence` (CRATES-REC-2) and
  `adjacent_row_difference` (CRATES-REC-1) in exact arithmetic.
* `stage_cells_eq_toeplitz`, `stage_payoffs_le_one` (CRATES-TOEP-1).
* `ExtRun.all_finite` (CRATES-CERT-6), `foldl_min_mem`, `foldl_min_max_perm`
  (CPP-DOC-FP-1, CANONICAL-FP-1), `pivot_factor_le_one` (DTH-SUP-2) and
  `assembly_error` for the environment claims.
-/

open Finset Matrix

namespace Formal.Float

/-! ## Constants of the binary64 model -/

/-- Unit roundoff of binary64 round-to-nearest, `u = 2^-53`. -/
noncomputable def u : ℝ := (1 / 2) ^ 53

/-- Absolute error bound of one underflowing binary64 operation, `η = 2^-1075`. -/
noncomputable def eta : ℝ := (1 / 2) ^ 1075

/-- Higham's constant `γ_n = n u / (1 - n u)`. -/
noncomputable def gam (n : ℕ) : ℝ := n * u / (1 - n * u)

theorem u_pos : 0 < u := by unfold u; positivity
theorem eta_pos : 0 < eta := by unfold eta; positivity
theorem u_le : u ≤ 1 / 10 ^ 15 := by unfold u; norm_num
theorem eta_le : eta ≤ 1 / 10 ^ 30 := by
  unfold eta
  rw [div_pow, one_pow, div_le_div_iff₀ (by positivity) (by positivity), one_mul, one_mul]
  calc (10:ℝ) ^ 30 ≤ 2 ^ 100 := by norm_num
    _ ≤ 2 ^ 1075 := pow_le_pow_right₀ (by norm_num) (by norm_num)

/-- The standard model of one rounded binary64 operation with exact result `x`
and computed result `y`: `y = x (1 + δ) + e`, `|δ| ≤ u`, `|e| ≤ η`. -/
def Rounds (x y : ℝ) : Prop := ∃ δ e : ℝ, |δ| ≤ u ∧ |e| ≤ eta ∧ y = x * (1 + δ) + e

theorem Rounds.refl (x : ℝ) : Rounds x x :=
  ⟨0, 0, by simp [u_pos.le], by simp [eta_pos.le], by ring⟩

theorem Rounds.abs_sub_le {x y : ℝ} (h : Rounds x y) : |y - x| ≤ u * |x| + eta := by
  obtain ⟨δ, e, hδ, he, rfl⟩ := h
  calc |x * (1 + δ) + e - x| = |x * δ + e| := by ring_nf
    _ ≤ |x * δ| + |e| := abs_add_le _ _
    _ = |x| * |δ| + |e| := by rw [abs_mul]
    _ ≤ |x| * u + eta := by gcongr
    _ = u * |x| + eta := by ring

theorem Rounds.abs_le {x y : ℝ} (h : Rounds x y) : |y| ≤ (1 + u) * |x| + eta := by
  have := h.abs_sub_le
  have h2 : |y| ≤ |y - x| + |x| := by
    calc |y| = |(y - x) + x| := by ring_nf
      _ ≤ |y - x| + |x| := abs_add_le _ _
  linarith

/-- Higham, Lemma 3.1 in the form used here: `(1 + u)^n - 1 ≤ γ_n` when `n u < 1`. -/
theorem pow_sub_one_le_gam (n : ℕ) (h : (n : ℝ) * u < 1) : (1 + u) ^ n - 1 ≤ gam n := by
  have hu := u_pos
  have hb : 1 + n * (-u) ≤ (1 + -u) ^ n := one_add_mul_le_pow (by linarith [u_le]) n
  have hprod : (1 + u) ^ n * (1 + -u) ^ n ≤ 1 := by
    rw [← mul_pow]
    apply pow_le_one₀ (by nlinarith [u_le]) (by nlinarith)
  have hpos : 0 < 1 - n * u := by linarith
  have key : (1 + u) ^ n * (1 - n * u) ≤ 1 := by
    have h0 : 0 ≤ (1 + u) ^ n := by positivity
    calc (1 + u) ^ n * (1 - n * u) ≤ (1 + u) ^ n * (1 + -u) ^ n := by
          apply mul_le_mul_of_nonneg_left _ h0; linarith
      _ ≤ 1 := hprod
  unfold gam
  rw [le_div_iff₀ hpos]
  nlinarith


/-- Error of a rounded running sum `acc (t+1) = fl(p t + acc t)`, `acc 0 = 0`
(Higham's summation analysis with absolute underflow terms). -/
theorem accum_error (p acc : ℕ → ℝ) (n : ℕ) (h0 : acc 0 = 0)
    (hstep : ∀ t < n, Rounds (p t + acc t) (acc (t + 1))) :
    |acc n - ∑ t ∈ range n, p t| ≤
      ((1 + u) ^ n - 1) * ∑ t ∈ range n, |p t| + n * (1 + u) ^ n * eta := by
  have hu := u_pos
  have he := eta_pos
  induction n with
  | zero => simp [h0]
  | succ n ih =>
    have ih := ih fun t ht => hstep t (by omega)
    obtain ⟨δ, e, hδ, hee, hacc⟩ := hstep n (by omega)
    set S := ∑ t ∈ range n, p t
    set A := ∑ t ∈ range n, |p t|
    set a := (1 + u) ^ n with ha_def
    have ha : 1 ≤ a := one_le_pow₀ (by linarith)
    have hA : |S| ≤ A := abs_sum_le_sum_abs _ _
    have hA0 : 0 ≤ A := Finset.sum_nonneg fun _ _ => abs_nonneg _
    rw [Finset.sum_range_succ, Finset.sum_range_succ, pow_succ, hacc]
    have hid : (p n + acc n) * (1 + δ) + e - (S + p n) =
        (acc n - S) * (1 + δ) + (S + p n) * δ + e := by ring
    rw [hid]
    have h1 : |acc n - S| * |1 + δ| ≤ ((a - 1) * A + n * a * eta) * (1 + u) := by
      apply mul_le_mul ih _ (abs_nonneg _)
        (add_nonneg (mul_nonneg (by linarith) hA0)
          (mul_nonneg (mul_nonneg (Nat.cast_nonneg n) (by linarith)) he.le))
      calc |1 + δ| ≤ |1| + |δ| := abs_add_le _ _
        _ ≤ 1 + u := by simp; linarith
    have h2 : |S + p n| * |δ| ≤ (A + |p n|) * u :=
      mul_le_mul ((abs_add_le _ _).trans (by linarith)) hδ (abs_nonneg _)
        (by linarith [abs_nonneg (p n)])
    calc |(acc n - S) * (1 + δ) + (S + p n) * δ + e|
        ≤ |(acc n - S) * (1 + δ)| + |(S + p n) * δ| + |e| := abs_add_three _ _ _
      _ = |acc n - S| * |1 + δ| + |S + p n| * |δ| + |e| := by rw [abs_mul, abs_mul]
      _ ≤ ((a - 1) * A + n * a * eta) * (1 + u) + (A + |p n|) * u + eta := by linarith
      _ ≤ (a * (1 + u) - 1) * (A + |p n|) + ((n : ℕ) + 1 : ℕ) * (a * (1 + u)) * eta := by
        push_cast
        have hpn := abs_nonneg (p n)
        nlinarith [mul_nonneg (mul_nonneg hpn hu.le) (sub_nonneg.mpr ha),
          mul_nonneg (mul_nonneg he.le (sub_nonneg.mpr ha)) (by linarith : (0:ℝ) ≤ 1 + u),
          mul_nonneg he.le hu.le]

/-- Error of a rounded dot product `acc (t+1) = fl(p' t + acc t)`, where
`p' t` is the product `p t` as rounded (separate multiply) or exact (fused
multiply-add, `p' = p`). The relative constant is `(1+u)^(n+1) - 1 ≤ γ_{n+1}`,
sharper than the `γ_{2n}` of `LEAP_CERTIFICATE.md:39-41`. -/
theorem dot_error (p p' acc : ℕ → ℝ) (n : ℕ) (h0 : acc 0 = 0)
    (hp : ∀ t < n, Rounds (p t) (p' t))
    (hstep : ∀ t < n, Rounds (p' t + acc t) (acc (t + 1))) :
    |acc n - ∑ t ∈ range n, p t| ≤
      ((1 + u) ^ (n + 1) - 1) * ∑ t ∈ range n, |p t| + 2 * n * (1 + u) ^ n * eta := by
  have hu := u_pos
  have he := eta_pos
  have h1 := accum_error p' acc n h0 hstep
  set a := (1 + u) ^ n with ha_def
  have ha : 1 ≤ a := one_le_pow₀ (by linarith)
  set A := ∑ t ∈ range n, |p t|
  have hA0 : 0 ≤ A := Finset.sum_nonneg fun _ _ => abs_nonneg _
  have h2 : ∑ t ∈ range n, |p' t| ≤ (1 + u) * A + n * eta := by
    calc ∑ t ∈ range n, |p' t| ≤ ∑ t ∈ range n, ((1 + u) * |p t| + eta) :=
          Finset.sum_le_sum fun t ht => (hp t (by simpa using ht)).abs_le
      _ = (1 + u) * A + n * eta := by
          rw [Finset.sum_add_distrib, ← Finset.mul_sum, Finset.sum_const, Finset.card_range,
            nsmul_eq_mul]
  have h3 : |∑ t ∈ range n, p' t - ∑ t ∈ range n, p t| ≤ u * A + n * eta := by
    rw [← Finset.sum_sub_distrib]
    calc |∑ t ∈ range n, (p' t - p t)| ≤ ∑ t ∈ range n, |p' t - p t| := abs_sum_le_sum_abs _ _
      _ ≤ ∑ t ∈ range n, (u * |p t| + eta) :=
          Finset.sum_le_sum fun t ht => (hp t (by simpa using ht)).abs_sub_le
      _ = u * A + n * eta := by
          rw [Finset.sum_add_distrib, ← Finset.mul_sum, Finset.sum_const, Finset.card_range,
            nsmul_eq_mul]
  have htri : |acc n - ∑ t ∈ range n, p t| ≤
      |acc n - ∑ t ∈ range n, p' t| + |∑ t ∈ range n, p' t - ∑ t ∈ range n, p t| := by
    calc _ = |(acc n - ∑ t ∈ range n, p' t) + (∑ t ∈ range n, p' t - ∑ t ∈ range n, p t)| := by
          ring_nf
      _ ≤ _ := abs_add_le _ _
  have hG : 0 ≤ a - 1 := by linarith
  have h4 : (a - 1) * ∑ t ∈ range n, |p' t| ≤ (a - 1) * ((1 + u) * A + n * eta) :=
    mul_le_mul_of_nonneg_left h2 hG
  rw [pow_succ]
  have hn : (0:ℝ) ≤ n := Nat.cast_nonneg n
  nlinarith


/-! ## Exact identities for arbitrary weights -/

open Formal.Toeplitz

/-- The row-difference residual `d r_k + ∑_{j<k} Δ_{k-j} r_j` of
`LEAP_CERTIFICATE.md:54`, with `d = s 0 - f` and `Δ_m = s m - s (m-1)`. -/
def residual (s : ℕ → ℝ) (f : ℝ) (r : ℕ → ℝ) (k : ℕ) : ℝ :=
  (s 0 - f) * r k + ∑ j ∈ range k, (s (k - j) - s (k - j - 1)) * r j

/-- **Residual decomposition** (claim CRATES-CERT-1, `LEAP_CERTIFICATE.md:50-59`).
For any reals `d`, `d̂`, `Δ_m`, `b̂_m`, `r_j`:
`d r_k + ∑_{j<k} Δ_{k-j} r_j = d̂ (r_k - ∑_{j<k} b̂_{k-j} r_j) + (d - d̂) r_k
  + ∑_{j<k} (d̂ b̂_{k-j} + Δ_{k-j}) r_j`. It is a ring identity; the computed
weights enter as exact inputs. -/
theorem residual_decomposition (d dh : ℝ) (Δ bh r : ℕ → ℝ) (k : ℕ) :
    d * r k + ∑ j ∈ range k, Δ (k - j) * r j =
      dh * (r k - ∑ j ∈ range k, bh (k - j) * r j) + (d - dh) * r k +
        ∑ j ∈ range k, (dh * bh (k - j) + Δ (k - j)) * r j := by
  simp only [mul_sub, Finset.mul_sum, add_mul, Finset.sum_add_distrib]
  have : ∑ j ∈ range k, dh * (bh (k - j) * r j) = ∑ j ∈ range k, dh * bh (k - j) * r j :=
    Finset.sum_congr rfl fun _ _ => by ring
  rw [this]; ring

/-- The exact weight sum `W = ∑_{j<n} r_j`. -/
def wsum (n : ℕ) (r : ℕ → ℝ) : ℝ := ∑ j ∈ range n, r j

/-- The Checker mixture `q_j = r_{n-1-j} / W` built from arbitrary weights. -/
noncomputable def mixQ (n : ℕ) (r : ℕ → ℝ) : Fin n → ℝ := fun j => r (n - 1 - j.val) / wsum n r

/-- The Dropper mixture `p_j = q_{n-1-j} = r_j / W`. -/
noncomputable def mixP (n : ℕ) (r : ℕ → ℝ) : Fin n → ℝ := fun i => mixQ n r i.rev

theorem mixP_apply (n : ℕ) (r : ℕ → ℝ) (i : Fin n) : mixP n r i = r i.val / wsum n r := by
  simp only [mixP, mixQ, Fin.val_rev]; congr 2; omega

theorem sum_mixQ {n : ℕ} (r : ℕ → ℝ) (hW : wsum n r ≠ 0) : ∑ j, mixQ n r j = 1 := by
  simp only [mixQ, ← Finset.sum_div]
  rw [div_eq_one_iff_eq hW, wsum, ← Fin.sum_univ_eq_sum_range]
  exact Fintype.sum_equiv Fin.revPerm _ _ fun j => by
    simp only [Fin.revPerm_apply, Fin.val_rev]; congr 1; omega

theorem mixQ_mem_simplex {n : ℕ} (r : ℕ → ℝ) (hr : ∀ j < n, 0 ≤ r j) (hW : 0 < wsum n r) :
    mixQ n r ∈ MatrixGame.simplex (Fin n) :=
  ⟨fun j => div_nonneg (hr _ (by omega)) hW.le, sum_mixQ r hW.ne'⟩

/-- **Adjacent row difference for arbitrary weights** (claims CRATES-REC-1 and
CRATES-CERT-3, `LEAP_CERTIFICATE.md:79-81`): with `q_j = r_{n-1-j} / W`,
`(M q)_i - (M q)_{i+1} = residual (n-1-i) / W`. At `n = 60` and `i = 59 - k`
this is the residual of row `k`. -/
theorem mulVec_sub_succ_mixQ {n : ℕ} (s : ℕ → ℝ) (f : ℝ) (r : ℕ → ℝ) (i : Fin n)
    (hi : i.val + 1 < n) :
    (toeplitz n s f *ᵥ mixQ n r) i - (toeplitz n s f *ᵥ mixQ n r) ⟨i.val + 1, hi⟩ =
      residual s f r (n - 1 - i.val) / wsum n r := by
  rw [mulVec_sub_succ]
  set a := n - 1 - i.val with ha
  have hsum : ∑ k ∈ univ.filter (i < ·), (s (k.val - i.val) - s (k.val - i.val - 1)) * mixQ n r k =
      (∑ j ∈ range a, (s (a - j) - s (a - j - 1)) * r j) / wsum n r := by
    rw [Finset.sum_div]
    refine Finset.sum_bij' (fun k _ => n - 1 - k.val)
      (fun j hj => ⟨n - 1 - j, by simp only [mem_range] at hj; omega⟩) ?_ ?_ ?_ ?_ ?_
    · intro k hk
      simp only [mem_filter, mem_univ, true_and] at hk
      simp only [mem_range]; have := Fin.lt_def.mp hk; omega
    · intro j hj
      simp only [mem_range] at hj
      simp only [mem_filter, mem_univ, true_and, Fin.lt_def]; omega
    · intro k hk
      ext; simp only; omega
    · intro j hj; simp only [mem_range] at hj; dsimp only; omega
    · intro k hk
      simp only [mem_filter, mem_univ, true_and] at hk
      have hk' := Fin.lt_def.mp hk
      simp only [mixQ, mul_div_assoc]
      rw [show a - (n - 1 - k.val) = k.val - i.val by omega]
  rw [hsum, residual, add_div]
  simp only [mixQ]
  rw [mul_div_assoc]

/-- **Last-row payoff** (`LEAP_CERTIFICATE.md:92-98`): if `r_0 = 1` and
`W ≠ 0`, then `(M q)_{n-1} = f + (s_0 - f) / W`, because `q_{n-1} = r_0 / W = 1 / W`. -/
theorem mulVec_last_mixQ {n : ℕ} (s : ℕ → ℝ) (f : ℝ) (r : ℕ → ℝ) (hn : 1 ≤ n)
    (hr0 : r 0 = 1) (hW : wsum n r ≠ 0) :
    (toeplitz n s f *ᵥ mixQ n r) ⟨n - 1, by omega⟩ = f + (s 0 - f) / wsum n r := by
  set i : Fin n := ⟨n - 1, by omega⟩ with hi_def
  rw [mulVec_toeplitz]
  have hfilter : univ.filter (fun k : Fin n => i ≤ k) = {i} := by
    ext k; simp only [mem_filter, mem_univ, true_and, mem_singleton]
    constructor
    · intro h; apply Fin.ext; have h1 : n - 1 ≤ k.val := h; have := k.isLt
      simp only [hi_def]; omega
    · rintro rfl; exact le_rfl
  have hlt : univ.filter (fun j : Fin n => j < i) = univ.erase i := by
    ext j; simp only [mem_filter, mem_univ, true_and, mem_erase, and_true]
    constructor
    · exact ne_of_lt
    · intro h
      have h1 := j.isLt
      have h2 : j.val ≠ n - 1 := fun e => h (Fin.ext e)
      show j.val < n - 1
      omega
  have hi' : mixQ n r i = 1 / wsum n r := by
    simp only [mixQ, hi_def]; rw [show n - 1 - (n - 1) = 0 by omega, hr0]
  rw [hfilter, hlt, Finset.sum_singleton, Nat.sub_self, Finset.sum_erase_eq_sub (mem_univ i),
    sum_mixQ r hW, hi']
  field_simp
  ring

/-- Telescoping: if adjacent entries of `x` differ by at most `ρ`, entries `a ≤ b`
differ by at most `(b - a) ρ`. -/
theorem abs_sub_le_of_adjacent {n : ℕ} (x : Fin n → ℝ) {ρ : ℝ}
    (h : ∀ (i : Fin n) (hi : i.val + 1 < n), |x i - x ⟨i.val + 1, hi⟩| ≤ ρ)
    (a b : Fin n) (hab : a ≤ b) : |x a - x b| ≤ ((b.val - a.val : ℕ) : ℝ) * ρ := by
  obtain ⟨t, ht⟩ : ∃ t, b.val = a.val + t := ⟨b.val - a.val, by have := Fin.le_def.mp hab; omega⟩
  induction t generalizing b with
  | zero =>
    have : a = b := Fin.ext (by omega)
    subst this; simp
  | succ t ih =>
    have hb1 : a.val + t < n := by have := b.isLt; omega
    set b' : Fin n := ⟨a.val + t, hb1⟩
    have h1 := ih b' (by rw [Fin.le_def]; simp [b']) rfl
    have h2 := h b' (by simp only [b']; have := b.isLt; omega)
    have hbb : (⟨b'.val + 1, by simp only [b']; have := b.isLt; omega⟩ : Fin n) = b :=
      Fin.ext (by simp only [b']; omega)
    rw [hbb] at h2
    have hc : ((b.val - a.val : ℕ) : ℝ) = ((b'.val - a.val : ℕ) : ℝ) + 1 := by
      simp only [b']; rw [ht]
      rw [show a.val + (t + 1) - a.val = t + 1 by omega, show a.val + t - a.val = t by omega]
      push_cast; ring
    calc |x a - x b| ≤ |x a - x b'| + |x b' - x b| := abs_sub_le _ _ _
      _ ≤ ((b'.val - a.val : ℕ) : ℝ) * ρ + ρ := add_le_add h1 h2
      _ = ((b.val - a.val : ℕ) : ℝ) * ρ := by rw [hc]; ring

/-- Spread of a vector whose adjacent entries differ by at most `ρ ≥ 0`. -/
theorem sub_le_of_adjacent {n : ℕ} (x : Fin n → ℝ) {ρ : ℝ} (hρ : 0 ≤ ρ)
    (h : ∀ (i : Fin n) (hi : i.val + 1 < n), |x i - x ⟨i.val + 1, hi⟩| ≤ ρ)
    (a b : Fin n) : x a - x b ≤ ((n - 1 : ℕ) : ℝ) * ρ := by
  rcases le_total a b with hab | hab
  · have := abs_sub_le_of_adjacent x h a b hab
    have hle : ((b.val - a.val : ℕ) : ℝ) ≤ ((n - 1 : ℕ) : ℝ) := by
      exact_mod_cast (by have := b.isLt; omega : b.val - a.val ≤ n - 1)
    calc x a - x b ≤ |x a - x b| := le_abs_self _
      _ ≤ _ := this
      _ ≤ _ := mul_le_mul_of_nonneg_right hle hρ
  · have := abs_sub_le_of_adjacent x h b a hab
    have hle : ((a.val - b.val : ℕ) : ℝ) ≤ ((n - 1 : ℕ) : ℝ) := by
      exact_mod_cast (by have := a.isLt; omega : a.val - b.val ≤ n - 1)
    calc x a - x b ≤ |x b - x a| := by rw [abs_sub_comm]; exact le_abs_self _
      _ ≤ _ := this
      _ ≤ _ := mul_le_mul_of_nonneg_right hle hρ


/-! ## Numeric constants -/

theorem u_le' : u ≤ 12 / 10 ^ 17 := by unfold u; norm_num

theorem gam_nonneg_of (n : ℕ) (h : (n : ℝ) * u < 1) : 0 ≤ gam n := by
  unfold gam; apply div_nonneg (mul_nonneg (Nat.cast_nonneg n) u_pos.le); linarith

theorem gam2_le : gam 2 ≤ 3 / 10 ^ 16 := by unfold gam u; norm_num
theorem gam59_le : gam 59 ≤ 1 / 10 ^ 3 := by unfold gam u; norm_num
theorem gam60_le : gam 60 ≤ 67 / 10 ^ 16 := by unfold gam u; norm_num
theorem gam118_le : gam 118 ≤ 1 / 10 ^ 13 := by unfold gam u; norm_num

theorem nu_lt_one {n : ℕ} (hn : n ≤ 118) : (n : ℝ) * u < 1 := by
  have : (n : ℝ) ≤ 118 := by exact_mod_cast hn
  nlinarith [u_le, u_pos]

/-- The per-row residual constant of `LEAP_CERTIFICATE.md:66`:
`4 γ_118 (1 + γ_2) + 4u + 4γ_2 + 500η`. -/
noncomputable def rho : ℝ := 4 * gam 118 * (1 + gam 2) + 4 * u + 4 * gam 2 + 500 * eta

/-- `ρ < 6e-14` (`LEAP_CERTIFICATE.md:67`); exact rational evaluation. -/
theorem rho_lt : rho < 6 / 10 ^ 14 := by
  have h := eta_le
  have : 4 * gam 118 * (1 + gam 2) + 4 * u + 4 * gam 2 + 500 * (1 / 10 ^ 30) < 6 / 10 ^ 14 := by
    unfold gam u; norm_num
  unfold rho; linarith

theorem rho_nonneg : 0 ≤ rho := by
  have h1 := gam_nonneg_of 118 (nu_lt_one le_rfl)
  have h2 := gam_nonneg_of 2 (nu_lt_one (by norm_num))
  have h3 := u_pos.le
  have h4 := eta_pos.le
  have h5 : 0 ≤ 4 * gam 118 * (1 + gam 2) := mul_nonneg (by linarith) (by linarith)
  unfold rho; linarith

/-- `59 · 6e-14 = 3.54e-12` (`LEAP_CERTIFICATE.md:85`). -/
theorem spread_const : (59 : ℝ) * (6 / 10 ^ 14) = 354 / 10 ^ 14 := by norm_num

/-! ## The computed run of the shortcut branch -/

/-- Every computed quantity of the shortcut branch, as a real number
(`leap_oracle.py:92-100`, `leap.rs:380-426,479`). `acc k t` is the recurrence
accumulator for weight `k` after `t` terms, `prod k t` the product
`b (k-t) * r t` as it enters the addition, and `wacc t` the weight-sum
accumulator after adding `r 59, …, r (60 - t)`. -/
structure Run where
  /-- Success payoffs `s[0..59]` (index `k` is lag `k + 1`). -/
  s : ℕ → ℝ
  /-- Failure payoff. -/
  f : ℝ
  /-- `fl(s[0] - f)`. -/
  dh : ℝ
  /-- `fl(s[m-1] - s[m])`. -/
  nd : ℕ → ℝ
  /-- `fl(nd m / dh)`. -/
  b : ℕ → ℝ
  /-- Product term entering accumulator `k` at step `t`. -/
  prod : ℕ → ℕ → ℝ
  /-- Accumulator for `r k` after `t` terms. -/
  acc : ℕ → ℕ → ℝ
  /-- Computed weights. -/
  r : ℕ → ℝ
  /-- Weight-sum accumulator. -/
  wacc : ℕ → ℝ
  /-- `fl(dh / Ŵ)`. -/
  y : ℝ
  /-- The stored value `fl(f + y)`. -/
  v : ℝ

/-- The hypotheses of the shortcut branch under the abstract rounding model.
Each rounded operation satisfies `Rounds`; the finite results mean no operation
overflowed (see `nonfinite_propagates`). With a fused multiply-add set
`prod k t = b (k-t) * r t` (`Rounds.refl`); with separate multiply and add,
`prod k t` is the rounded product. The code's guards `|dh| ≥ 1e-6`
(`leap.rs:475`, `leap_oracle.py:99`) and `Ŵ > 0` are not needed: `dh ≠ 0`
suffices, and `Ŵ > 0` follows. -/
structure Run.Valid (R : Run) : Prop where
  s_bound : ∀ k < 60, |R.s k| ≤ 2
  f_bound : |R.f| ≤ 2
  dh_round : Rounds (R.s 0 - R.f) R.dh
  dh_ne : R.dh ≠ 0
  nd_round : ∀ m, 1 ≤ m → m < 60 → Rounds (R.s (m - 1) - R.s m) (R.nd m)
  b_round : ∀ m, 1 ≤ m → m < 60 → Rounds (R.nd m / R.dh) (R.b m)
  prod_round : ∀ k t, 1 ≤ k → k < 60 → t < k → Rounds (R.b (k - t) * R.r t) (R.prod k t)
  acc_zero : ∀ k, R.acc k 0 = 0
  acc_step : ∀ k t, 1 ≤ k → k < 60 → t < k →
    Rounds (R.prod k t + R.acc k t) (R.acc k (t + 1))
  r_zero : R.r 0 = 1
  r_eq : ∀ k, 1 ≤ k → k < 60 → R.r k = R.acc k k
  r_nonneg : ∀ j < 60, 0 ≤ R.r j
  wacc_zero : R.wacc 0 = 0
  wacc_step : ∀ t < 60, Rounds (R.wacc t + R.r (59 - t)) (R.wacc (t + 1))
  y_round : Rounds (R.dh / R.wacc 60) R.y
  v_round : Rounds (R.f + R.y) R.v

namespace Run

variable {R : Run}

theorem Valid.one_le_W (hR : R.Valid) : 1 ≤ wsum 60 R.r := by
  have := Finset.single_le_sum (f := R.r) (fun j hj => hR.r_nonneg j (by simpa using hj))
    (show 0 ∈ range 60 by simp)
  rw [hR.r_zero] at this; exact this

theorem Valid.r_le_W (hR : R.Valid) {k : ℕ} (hk : k < 60) : R.r k ≤ wsum 60 R.r :=
  Finset.single_le_sum (f := R.r) (fun j hj => hR.r_nonneg j (by simpa using hj))
    (show k ∈ range 60 by simpa using hk)

theorem Valid.sum_r_le_W (hR : R.Valid) {k : ℕ} (hk : k < 60) :
    ∑ j ∈ range k, R.r j ≤ wsum 60 R.r :=
  Finset.sum_le_sum_of_subset_of_nonneg (Finset.range_mono hk.le)
    fun j hj _ => hR.r_nonneg j (by simpa using hj)

theorem Valid.abs_d_le (hR : R.Valid) : |R.s 0 - R.f| ≤ 4 := by
  have h1 := hR.s_bound 0 (by norm_num); have h2 := hR.f_bound
  calc |R.s 0 - R.f| ≤ |R.s 0| + |R.f| := abs_sub _ _
    _ ≤ 4 := by linarith

theorem Valid.abs_Δ_le (hR : R.Valid) {m : ℕ} (hm : m < 60) : |R.s m - R.s (m - 1)| ≤ 4 := by
  have h1 := hR.s_bound m hm; have h2 := hR.s_bound (m - 1) (by omega)
  calc |R.s m - R.s (m - 1)| ≤ |R.s m| + |R.s (m - 1)| := abs_sub _ _
    _ ≤ 4 := by linarith

theorem Valid.abs_dh_le (hR : R.Valid) : |R.dh| ≤ 4 * (1 + u) + eta := by
  have := hR.dh_round.abs_le
  have := hR.abs_d_le
  have hu := u_pos
  nlinarith

/-- Coefficient-formation error: `|d̂ b̂_m + Δ_m| ≤ 4γ_2 + (1 + u + D) η` with
`D = 4(1+u) + η` bounding `|d̂|`. -/
theorem Valid.coef_bound (hR : R.Valid) {m : ℕ} (hm1 : 1 ≤ m) (hm : m < 60) :
    |R.dh * R.b m + (R.s m - R.s (m - 1))| ≤
      4 * gam 2 + (1 + u + (4 * (1 + u) + eta)) * eta := by
  have hΔ := hR.abs_Δ_le hm
  have hD := hR.abs_dh_le
  have hu := u_pos
  have he := eta_pos
  have hg : (1 + u) ^ 2 - 1 ≤ gam 2 := pow_sub_one_le_gam 2 (nu_lt_one (by norm_num))
  obtain ⟨δ', e', hδ', he', hnd⟩ := hR.nd_round m hm1 hm
  obtain ⟨δ, e, hδ, hee, hb⟩ := hR.b_round m hm1 hm
  set Δ := R.s m - R.s (m - 1)
  have hid : R.dh * R.b m + Δ = -Δ * (δ' + δ + δ' * δ) + e' * (1 + δ) + R.dh * e := by
    rw [hb, hnd]
    have : R.s (m - 1) - R.s m = -Δ := by simp only [Δ]; ring
    rw [this]
    field_simp [hR.dh_ne]
    ring
  rw [hid]
  have h1 : |δ' + δ + δ' * δ| ≤ 2 * u + u ^ 2 := by
    calc |δ' + δ + δ' * δ| ≤ |δ'| + |δ| + |δ' * δ| := abs_add_three _ _ _
      _ = |δ'| + |δ| + |δ'| * |δ| := by rw [abs_mul]
      _ ≤ u + u + u * u := by gcongr
      _ = 2 * u + u ^ 2 := by ring
  have h2 : |1 + δ| ≤ 1 + u := (abs_add_le _ _).trans (by simp; linarith)
  calc |-Δ * (δ' + δ + δ' * δ) + e' * (1 + δ) + R.dh * e|
      ≤ |-Δ * (δ' + δ + δ' * δ)| + |e' * (1 + δ)| + |R.dh * e| := abs_add_three _ _ _
    _ = |Δ| * |δ' + δ + δ' * δ| + |e'| * |1 + δ| + |R.dh| * |e| := by
        rw [abs_mul, abs_mul, abs_mul, abs_neg]
    _ ≤ 4 * (2 * u + u ^ 2) + eta * (1 + u) + (4 * (1 + u) + eta) * eta := by
        have hA := mul_le_mul hΔ h1 (abs_nonneg _) (by norm_num)
        have hB := mul_le_mul he' h2 (abs_nonneg _) he.le
        have hC := mul_le_mul hD hee (abs_nonneg _) (by linarith)
        linarith
    _ ≤ 4 * gam 2 + (1 + u + (4 * (1 + u) + eta)) * eta := by
        have : 2 * u + u ^ 2 ≤ gam 2 := by linarith [hg]
        nlinarith


/-- The dot-product error of one computed weight (`leap.rs:404-413`,
`leap_oracle.py:87-90`): `|r_k - ∑_{j<k} b̂_{k-j} r_j| ≤ ((1+u)^(k+1) - 1) ∑ |b̂_{k-j}| r_j
+ 2 k (1+u)^k η`, for fused or separate multiply-add. -/
theorem Valid.dot (hR : R.Valid) {k : ℕ} (hk1 : 1 ≤ k) (hk : k < 60) :
    |R.r k - ∑ j ∈ range k, R.b (k - j) * R.r j| ≤
      ((1 + u) ^ (k + 1) - 1) * ∑ j ∈ range k, |R.b (k - j)| * R.r j +
        2 * k * (1 + u) ^ k * eta := by
  have h := dot_error (fun t => R.b (k - t) * R.r t) (R.prod k) (R.acc k) k (hR.acc_zero k)
    (fun t ht => hR.prod_round k t hk1 hk ht) (fun t ht => hR.acc_step k t hk1 hk ht)
  rw [hR.r_eq k hk1 hk]
  have hs : ∑ t ∈ range k, |R.b (k - t) * R.r t| = ∑ j ∈ range k, |R.b (k - j)| * R.r j :=
    Finset.sum_congr rfl fun t ht => by
      rw [abs_mul, abs_of_nonneg (hR.r_nonneg t (by simp at ht; omega))]
  rw [← hs]; exact h

/-- The numeric step of the residual bound: the collected constant is at most
`ρ = 4γ_118(1+γ_2) + 4u + 4γ_2 + 500η`. The `η` coefficient is about 480. -/
theorem rho_const :
    gam 60 * (4 + (4 * gam 2 + (1 + u + (4 * (1 + u) + eta)) * eta)) + (4 * u + eta) +
      (4 * gam 2 + (1 + u + (4 * (1 + u) + eta)) * eta) +
      (4 * (1 + u) + eta) * (118 * (1 + gam 59) * eta) ≤ rho := by
  have h60 : gam 60 ≤ gam 118 := by unfold gam u; norm_num
  have hg60 : gam 60 ≤ 1 / 10 ^ 3 := by unfold gam u; norm_num
  have hg59 := gam59_le
  have hg2' := gam_nonneg_of 2 (nu_lt_one (by norm_num))
  have hg59' := gam_nonneg_of 59 (nu_lt_one (by norm_num))
  have hg60' := gam_nonneg_of 60 (nu_lt_one (by norm_num))
  have hu := u_le; have hu0 := u_pos; have he := eta_le; have he0 := eta_pos
  set K := gam 60 * (1 + u + (4 * (1 + u) + eta)) + 1 + (1 + u + (4 * (1 + u) + eta)) +
    (4 * (1 + u) + eta) * (118 * (1 + gam 59)) with hK
  have hDg : (4 * (1 + u) + eta) * (1 + gam 59) ≤ (4 + 1 / 10 ^ 12) * (1 + 1 / 10 ^ 3) :=
    mul_le_mul (by linarith) (by linarith) (by linarith) (by norm_num)
  have hgK : gam 60 * (1 + u + (4 * (1 + u) + eta)) ≤ 1 / 10 ^ 3 * 6 :=
    mul_le_mul hg60 (by linarith) (by linarith) (by norm_num)
  have hK500 : K ≤ 500 := by rw [hK]; nlinarith
  have hKe : eta * K ≤ eta * 500 := mul_le_mul_of_nonneg_left hK500 he0.le
  have hgg : gam 60 * gam 2 ≤ gam 118 * gam 2 := mul_le_mul_of_nonneg_right h60 hg2'
  have hid : gam 60 * (4 + (4 * gam 2 + (1 + u + (4 * (1 + u) + eta)) * eta)) + (4 * u + eta) +
      (4 * gam 2 + (1 + u + (4 * (1 + u) + eta)) * eta) +
      (4 * (1 + u) + eta) * (118 * (1 + gam 59) * eta) =
      4 * gam 60 + 4 * (gam 60 * gam 2) + 4 * u + 4 * gam 2 + eta * K := by rw [hK]; ring
  rw [hid]; unfold rho; nlinarith

/-- **Per-row residual bound** (claims CRATES-CERT-2 and STL-FP-1,
`LEAP_CERTIFICATE.md:61-75`). Under the rounding model `Run.Valid`, for every
`k ∈ 1..59`, `|d r_k + ∑_{j<k} Δ_{k-j} r_j| ≤ ρ W` with
`ρ = 4γ_118(1+γ_2) + 4u + 4γ_2 + 500η`. The proof bounds the three terms of
`residual_decomposition`: the dot-product error (with `γ_{k+1} ≤ γ_60`, not
the doc's `γ_{2k} ≤ γ_118`), the rounding of `d`, and the coefficient error
`Valid.coef_bound`; every subset sum of the nonnegative weights is at most `W`,
and `W ≥ 1` absorbs the underflow terms (about `480η`). -/
theorem Valid.residual_le (hR : R.Valid) {k : ℕ} (hk1 : 1 ≤ k) (hk : k < 60) :
    |residual R.s R.f R.r k| ≤ rho * wsum 60 R.r := by
  set W := wsum 60 R.r with hWdef
  have hW1 : 1 ≤ W := hR.one_le_W
  have hu := u_pos; have he := eta_pos
  have hg2 := gam_nonneg_of 2 (nu_lt_one (by norm_num))
  have hg59 := gam_nonneg_of 59 (nu_lt_one (by norm_num))
  have hg60 := gam_nonneg_of 60 (nu_lt_one (by norm_num))
  set D := 4 * (1 + u) + eta with hD
  set c2 := 4 * gam 2 + (1 + u + D) * eta with hc2
  have hD0 : 0 ≤ D := by rw [hD]; linarith
  have hc20 : 0 ≤ c2 := by
    rw [hc2]; have := mul_nonneg (by linarith : 0 ≤ 1 + u + D) he.le; linarith
  have hres : residual R.s R.f R.r k =
      R.dh * (R.r k - ∑ j ∈ range k, R.b (k - j) * R.r j) + (R.s 0 - R.f - R.dh) * R.r k +
      ∑ j ∈ range k, (R.dh * R.b (k - j) + (R.s (k - j) - R.s (k - j - 1))) * R.r j :=
    residual_decomposition (R.s 0 - R.f) R.dh (fun m => R.s m - R.s (m - 1)) R.b R.r k
  have hrk : 0 ≤ R.r k := hR.r_nonneg k hk
  have hrW : R.r k ≤ W := hR.r_le_W hk
  have hsumW : ∑ j ∈ range k, R.r j ≤ W := hR.sum_r_le_W hk
  have hrj : ∀ j ∈ range k, 0 ≤ R.r j := fun j hj => hR.r_nonneg j (by simp at hj; omega)
  have hcoef : ∀ j ∈ range k,
      |R.dh * R.b (k - j) + (R.s (k - j) - R.s (k - j - 1))| ≤ c2 := fun j hj => by
    have hj' : j < k := by simpa using hj
    exact hR.coef_bound (m := k - j) (by omega) (by omega)
  -- the coefficient term
  have hT3 : |∑ j ∈ range k, (R.dh * R.b (k - j) + (R.s (k - j) - R.s (k - j - 1))) * R.r j| ≤
      c2 * W := by
    calc _ ≤ ∑ j ∈ range k, |(R.dh * R.b (k - j) + (R.s (k - j) - R.s (k - j - 1))) * R.r j| :=
          abs_sum_le_sum_abs _ _
      _ ≤ ∑ j ∈ range k, c2 * R.r j := Finset.sum_le_sum fun j hj => by
          rw [abs_mul, abs_of_nonneg (hrj j hj)]
          exact mul_le_mul_of_nonneg_right (hcoef j hj) (hrj j hj)
      _ = c2 * ∑ j ∈ range k, R.r j := by rw [Finset.mul_sum]
      _ ≤ c2 * W := mul_le_mul_of_nonneg_left hsumW hc20
  -- the rounding of d
  have hdd : |R.s 0 - R.f - R.dh| ≤ 4 * u + eta := by
    have h1 := hR.dh_round.abs_sub_le
    rw [abs_sub_comm] at h1
    have h2 := mul_le_mul_of_nonneg_left hR.abs_d_le hu.le
    linarith
  have hT2 : |(R.s 0 - R.f - R.dh) * R.r k| ≤ (4 * u + eta) * W := by
    rw [abs_mul, abs_of_nonneg hrk]
    exact mul_le_mul hdd hrW hrk (by linarith)
  -- the dot-product term
  have hS0 : 0 ≤ ∑ j ∈ range k, |R.b (k - j)| * R.r j :=
    Finset.sum_nonneg fun j hj => mul_nonneg (abs_nonneg _) (hrj j hj)
  have hbsum : |R.dh| * ∑ j ∈ range k, |R.b (k - j)| * R.r j ≤ (4 + c2) * W := by
    rw [Finset.mul_sum]
    calc ∑ j ∈ range k, |R.dh| * (|R.b (k - j)| * R.r j) ≤ ∑ j ∈ range k, (4 + c2) * R.r j :=
          Finset.sum_le_sum fun j hj => by
            have hj' : j < k := by simpa using hj
            have hΔ := hR.abs_Δ_le (m := k - j) (by omega)
            have hc := hcoef j hj
            have hb : |R.dh * R.b (k - j)| ≤ 4 + c2 := by
              calc |R.dh * R.b (k - j)| = |(R.dh * R.b (k - j) + (R.s (k - j) - R.s (k - j - 1))) -
                    (R.s (k - j) - R.s (k - j - 1))| := by congr 1; ring
                _ ≤ |R.dh * R.b (k - j) + (R.s (k - j) - R.s (k - j - 1))| +
                    |R.s (k - j) - R.s (k - j - 1)| := abs_sub _ _
                _ ≤ c2 + 4 := add_le_add hc hΔ
                _ = 4 + c2 := by ring
            rw [← mul_assoc, ← abs_mul]
            exact mul_le_mul_of_nonneg_right hb (hrj j hj)
      _ = (4 + c2) * ∑ j ∈ range k, R.r j := by rw [Finset.mul_sum]
      _ ≤ (4 + c2) * W := mul_le_mul_of_nonneg_left hsumW (by linarith)
  have hG : (1 + u) ^ (k + 1) - 1 ≤ gam 60 := by
    have h1 := pow_le_pow_right₀ (by linarith : (1:ℝ) ≤ 1 + u) (show k + 1 ≤ 60 by omega)
    have h2 := pow_sub_one_le_gam 60 (nu_lt_one (by norm_num))
    linarith
  have hG0 : 0 ≤ (1 + u) ^ (k + 1) - 1 := by
    have := one_le_pow₀ (by linarith : (1:ℝ) ≤ 1 + u) (n := k + 1); linarith
  have hpk : (1 + u) ^ k ≤ 1 + gam 59 := by
    have h1 := pow_le_pow_right₀ (by linarith : (1:ℝ) ≤ 1 + u) (show k ≤ 59 by omega)
    have h2 := pow_sub_one_le_gam 59 (nu_lt_one (by norm_num))
    linarith
  have hpk0 : 0 ≤ (1 + u) ^ k := (zero_le_one.trans (one_le_pow₀ (by linarith)))
  have hA : 2 * (k : ℝ) * (1 + u) ^ k * eta ≤ 118 * (1 + gam 59) * eta := by
    have hk' : (k : ℝ) ≤ 59 := by exact_mod_cast (by omega : k ≤ 59)
    have := mul_le_mul (by linarith : 2 * (k : ℝ) ≤ 118) hpk hpk0 (by norm_num)
    exact mul_le_mul_of_nonneg_right this he.le
  have hA0 : 0 ≤ 2 * (k : ℝ) * (1 + u) ^ k * eta :=
    mul_nonneg (mul_nonneg (by positivity) hpk0) he.le
  have hT1 : |R.dh * (R.r k - ∑ j ∈ range k, R.b (k - j) * R.r j)| ≤
      gam 60 * ((4 + c2) * W) + D * (118 * (1 + gam 59) * eta) := by
    rw [abs_mul]
    have hdot := hR.dot hk1 hk
    have hdh : |R.dh| ≤ D := hR.abs_dh_le
    calc |R.dh| * |R.r k - ∑ j ∈ range k, R.b (k - j) * R.r j|
        ≤ |R.dh| * (((1 + u) ^ (k + 1) - 1) * ∑ j ∈ range k, |R.b (k - j)| * R.r j +
            2 * k * (1 + u) ^ k * eta) := mul_le_mul_of_nonneg_left hdot (abs_nonneg _)
      _ = ((1 + u) ^ (k + 1) - 1) * (|R.dh| * ∑ j ∈ range k, |R.b (k - j)| * R.r j) +
            |R.dh| * (2 * k * (1 + u) ^ k * eta) := by ring
      _ ≤ gam 60 * ((4 + c2) * W) + D * (118 * (1 + gam 59) * eta) :=
          add_le_add (mul_le_mul hG hbsum (mul_nonneg (abs_nonneg _) hS0) hg60)
            (mul_le_mul hdh hA hA0 hD0)
  have hK0 : 0 ≤ D * (118 * (1 + gam 59) * eta) :=
    mul_nonneg hD0 (mul_nonneg (by linarith) he.le)
  have hKW : D * (118 * (1 + gam 59) * eta) ≤ D * (118 * (1 + gam 59) * eta) * W := by
    have := mul_le_mul_of_nonneg_left hW1 hK0; linarith
  have hconst := rho_const
  have hW0 : 0 ≤ W := by linarith
  have hfin : (gam 60 * (4 + c2) + (4 * u + eta) + c2 + D * (118 * (1 + gam 59) * eta)) * W ≤
      rho * W := mul_le_mul_of_nonneg_right hconst hW0
  rw [hres]
  calc _ ≤ _ := abs_add_three _ _ _
    _ ≤ gam 60 * ((4 + c2) * W) + D * (118 * (1 + gam 59) * eta) + (4 * u + eta) * W + c2 * W :=
        add_le_add (add_le_add hT1 hT2) hT3
    _ ≤ rho * W := by nlinarith

theorem Valid.W_pos (hR : R.Valid) : 0 < wsum 60 R.r := lt_of_lt_of_le one_pos hR.one_le_W

/-- **Adjacent rows of `M q`** (claim CRATES-CERT-3, `LEAP_CERTIFICATE.md:79-81`):
under `Run.Valid`, adjacent entries of `M q` differ by at most `ρ`, where
`M = toeplitz 60 s f` and `q_j = r_{59-j} / W` with the exact sum `W`. -/
theorem Valid.adjacent_le (hR : R.Valid) (i : Fin 60) (hi : i.val + 1 < 60) :
    |(toeplitz 60 R.s R.f *ᵥ mixQ 60 R.r) i -
        (toeplitz 60 R.s R.f *ᵥ mixQ 60 R.r) ⟨i.val + 1, hi⟩| ≤ rho := by
  rw [mulVec_sub_succ_mixQ, abs_div, abs_of_pos hR.W_pos, div_le_iff₀ hR.W_pos]
  exact hR.residual_le (by omega) (by omega)

/-- **Spread of `M q`** (claims CRATES-CERT-3 and STL-FP-1, `LEAP_CERTIFICATE.md:82-86`):
for all rows `a, b`, `(M q)_a - (M q)_b ≤ 59 ρ < 59 · 6e-14 = 3.54e-12`, by
telescoping the 59 adjacent differences. -/
theorem Valid.spread (hR : R.Valid) (a b : Fin 60) :
    (toeplitz 60 R.s R.f *ᵥ mixQ 60 R.r) a - (toeplitz 60 R.s R.f *ᵥ mixQ 60 R.r) b ≤ 59 * rho ∧
      59 * rho < 354 / 10 ^ 14 := by
  refine ⟨?_, by have := rho_lt; linarith⟩
  have := sub_le_of_adjacent _ rho_nonneg (fun i hi => hR.adjacent_le i hi) a b
  norm_num at this ⊢; linarith

/-- The Rust weight sum `Ŵ` (`leap.rs:417-426`, `sum += r[59-k]` for
`k = 0..59`) satisfies `|Ŵ - W| ≤ ((1+u)^60 - 1) W + 60 (1+u)^60 η`. -/
theorem Valid.wacc_error (hR : R.Valid) :
    |R.wacc 60 - wsum 60 R.r| ≤ ((1 + u) ^ 60 - 1) * wsum 60 R.r + 60 * (1 + u) ^ 60 * eta := by
  have h := accum_error (fun t => R.r (59 - t)) R.wacc 60 hR.wacc_zero
    (fun t ht => by rw [add_comm]; exact hR.wacc_step t ht)
  have hsum : ∑ t ∈ range 60, R.r (59 - t) = wsum 60 R.r := by
    rw [wsum, ← Finset.sum_range_reflect]
    exact Finset.sum_congr rfl fun j hj => by simp only [mem_range] at hj; congr 1; omega
  have habs : ∑ t ∈ range 60, |R.r (59 - t)| = wsum 60 R.r := by
    rw [← hsum]
    exact Finset.sum_congr rfl fun t ht => abs_of_nonneg (hR.r_nonneg _ (by omega))
  rw [hsum, habs] at h
  push_cast at h
  exact h

/-- The stored-value error of `v = fl(F + fl(d̂ / Ŵ))` against `F + (S_0 - F) / W`,
for any computed sum `Ŵ` with relative error `ε ≤ 7e-15`
(`LEAP_CERTIFICATE.md:98-108`). The Rust sequential sum gives
`ε = γ_60 + 60(1+γ_60)η` (`Valid.last_row_close`). Python's builtin `sum`
(`leap_oracle.py:94`) compensates its rounding from Python 3.12 on; this lemma
applies to it once its relative error is at most `7e-15`, which we do not
formalize. -/
theorem value_close {s0 f dh W Wh y v ε : ℝ} (hs0 : |s0| ≤ 2) (hf : |f| ≤ 2) (hW : 1 ≤ W)
    (hε : ε ≤ 7 / 10 ^ 15) (hWh : |Wh - W| ≤ ε * W)
    (hdh : Rounds (s0 - f) dh) (hy : Rounds (dh / Wh) y) (hv : Rounds (f + y) v) :
    |v - (f + (s0 - f) / W)| < 4 / 10 ^ 14 := by
  have hu := u_le'; have hu0 := u_pos; have he := eta_le; have he0 := eta_pos
  set d := s0 - f with hd_def
  have hd : |d| ≤ 4 := (abs_sub _ _).trans (by linarith)
  have hW0 : 0 < W := by linarith
  have hWh1 : W * (1 - ε) ≤ Wh := by
    have := neg_abs_le (Wh - W); nlinarith
  have hWhpos : 0 < Wh := by nlinarith
  have hdd : |dh - d| ≤ 4 * u + eta := by
    have h1 := hdh.abs_sub_le; have h2 := mul_le_mul_of_nonneg_left hd hu0.le; linarith
  have hE1 : |dh / Wh - d / W| ≤ 3 / 10 ^ 14 := by
    have hid : dh / Wh - d / W = ((dh - d) * W - d * (Wh - W)) / (Wh * W) := by
      field_simp; ring
    have hden := mul_pos hWhpos hW0
    rw [hid, abs_div, abs_of_pos hden, div_le_iff₀ hden]
    have hnum : |(dh - d) * W - d * (Wh - W)| ≤ (4 * u + eta) * W + 4 * (ε * W) := by
      calc _ ≤ |(dh - d) * W| + |d * (Wh - W)| := abs_sub _ _
        _ = |dh - d| * W + |d| * |Wh - W| := by rw [abs_mul, abs_mul, abs_of_pos hW0]
        _ ≤ (4 * u + eta) * W + 4 * (ε * W) :=
            add_le_add (mul_le_mul_of_nonneg_right hdd hW0.le)
              (mul_le_mul hd hWh (abs_nonneg _) (by norm_num))
    have h1 : (1 - ε) * W ≤ Wh * W := by
      have h3 : W * (1 - ε) * W ≤ Wh * W := mul_le_mul_of_nonneg_right hWh1 hW0.le
      have h4 : 0 ≤ (1 - ε) * W := mul_nonneg (by linarith) hW0.le
      have h5 := mul_le_mul_of_nonneg_left hW h4
      nlinarith
    have h2 : 4 * u + eta + 4 * ε ≤ 3 / 10 ^ 14 * (1 - ε) := by linarith
    calc |(dh - d) * W - d * (Wh - W)| ≤ (4 * u + eta + 4 * ε) * W := by linarith
      _ ≤ (3 / 10 ^ 14 * (1 - ε)) * W := mul_le_mul_of_nonneg_right h2 hW0.le
      _ = 3 / 10 ^ 14 * ((1 - ε) * W) := by ring
      _ ≤ 3 / 10 ^ 14 * (Wh * W) := mul_le_mul_of_nonneg_left h1 (by norm_num)
  have hdW : |d / W| ≤ 4 := by
    rw [abs_div, abs_of_pos hW0, div_le_iff₀ hW0]; nlinarith
  have hq : |dh / Wh| ≤ 4 + 3 / 10 ^ 14 := by
    have := abs_sub_abs_le_abs_sub (dh / Wh) (d / W); linarith
  have hE2 : |y - d / W| ≤ 31 / 10 ^ 15 := by
    have h1 := hy.abs_sub_le
    have h2 := mul_le_mul_of_nonneg_left hq hu0.le
    have h3 : |y - d / W| ≤ |y - dh / Wh| + |dh / Wh - d / W| := abs_sub_le _ _ _
    linarith
  have hconv : |f + d / W| ≤ 2 := by
    have hid : f + d / W = f * (1 - 1 / W) + s0 * (1 / W) := by rw [hd_def]; field_simp; ring
    have ha : 0 ≤ 1 / W := by positivity
    have hb : 1 / W ≤ 1 := by rw [div_le_one hW0]; exact hW
    rw [hid]
    calc |f * (1 - 1 / W) + s0 * (1 / W)| ≤ |f * (1 - 1 / W)| + |s0 * (1 / W)| := abs_add_le _ _
      _ = |f| * (1 - 1 / W) + |s0| * (1 / W) := by
          rw [abs_mul, abs_mul, abs_of_nonneg (by linarith : 0 ≤ 1 - 1 / W), abs_of_nonneg ha]
      _ ≤ 2 * (1 - 1 / W) + 2 * (1 / W) :=
          add_le_add (mul_le_mul_of_nonneg_right hf (by linarith))
            (mul_le_mul_of_nonneg_right hs0 ha)
      _ = 2 := by ring
  have hfy : |f + y| ≤ 2 + 31 / 10 ^ 15 := by
    have : |f + y| ≤ |f + d / W| + |y - d / W| := by
      calc |f + y| = |(f + d / W) + (y - d / W)| := by ring_nf
        _ ≤ _ := abs_add_le _ _
    linarith
  have hv1 := hv.abs_sub_le
  have hv2 := mul_le_mul_of_nonneg_left hfy hu0.le
  have htri : |v - (f + d / W)| ≤ |v - (f + y)| + |y - d / W| := by
    calc |v - (f + d / W)| = |(v - (f + y)) + (y - d / W)| := by ring_nf
      _ ≤ _ := abs_add_le _ _
  linarith

/-- **Last-row value** (claims CRATES-CERT-4 and STL-FP-1, `LEAP_CERTIFICATE.md:92-108`):
`(M q)_59 = F + (S_0 - F) / W` exactly, and the stored value
`v = fl(F + fl(d̂ / Ŵ))` satisfies `|v - (M q)_59| < 4e-14`. The sum's relative
constant is `γ_60`, as in the doc. -/
theorem Valid.last_row_close (hR : R.Valid) :
    (toeplitz 60 R.s R.f *ᵥ mixQ 60 R.r) ⟨59, by norm_num⟩ = R.f + (R.s 0 - R.f) / wsum 60 R.r ∧
      |R.v - (toeplitz 60 R.s R.f *ᵥ mixQ 60 R.r) ⟨59, by norm_num⟩| < 4 / 10 ^ 14 := by
  have hlast : (toeplitz 60 R.s R.f *ᵥ mixQ 60 R.r) ⟨59, by norm_num⟩ =
      R.f + (R.s 0 - R.f) / wsum 60 R.r :=
    mulVec_last_mixQ R.s R.f R.r (by norm_num) hR.r_zero hR.W_pos.ne'
  refine ⟨hlast, ?_⟩
  rw [hlast]
  have hu := u_pos; have he := eta_pos; have he' := eta_le
  have hW1 := hR.one_le_W
  have hg60 := gam_nonneg_of 60 (nu_lt_one (by norm_num))
  have hg60' := gam60_le
  have hpow : (1 + u) ^ 60 ≤ 1 + gam 60 := by
    have := pow_sub_one_le_gam 60 (nu_lt_one (by norm_num)); linarith
  have hpow0 : 0 ≤ (1 + u) ^ 60 := zero_le_one.trans (one_le_pow₀ (by linarith))
  have herr := hR.wacc_error
  set ε := gam 60 + 60 * (1 + gam 60) * eta
  have hε : ε ≤ 7 / 10 ^ 15 := by
    have : 60 * (1 + gam 60) * eta ≤ 60 * 2 * (1 / 10 ^ 30) :=
      mul_le_mul (by linarith) he' he.le (by norm_num)
    linarith
  have hWh : |R.wacc 60 - wsum 60 R.r| ≤ ε * wsum 60 R.r := by
    have h1 : ((1 + u) ^ 60 - 1) * wsum 60 R.r ≤ gam 60 * wsum 60 R.r :=
      mul_le_mul_of_nonneg_right (by linarith) (by linarith)
    have h2 : 60 * (1 + u) ^ 60 * eta ≤ 60 * (1 + gam 60) * eta :=
      mul_le_mul_of_nonneg_right (by linarith) he.le
    have h3 : 60 * (1 + gam 60) * eta ≤ 60 * (1 + gam 60) * eta * wsum 60 R.r := by
      have h0 : 0 ≤ 60 * (1 + gam 60) * eta :=
        mul_nonneg (mul_nonneg (by norm_num) (by linarith)) he.le
      have := mul_le_mul_of_nonneg_left hW1 h0; linarith
    have : ε * wsum 60 R.r = gam 60 * wsum 60 R.r + 60 * (1 + gam 60) * eta * wsum 60 R.r := by
      ring
    linarith
  exact value_close (hR.s_bound 0 (by norm_num)) hR.f_bound hW1 hε hWh hR.dh_round
    hR.y_round hR.v_round

/-- `CERT_RADIUS = 1e-10` (`leap_oracle.py:9`). -/
noncomputable def certRadius : ℝ := 1 / 10 ^ 10

/-- **Shortcut enclosure** (claims CRATES-CERT-5 and STL-FP-1,
`LEAP_CERTIFICATE.md:16-18,79-110`; code `leap.rs:475-479`,
`leap_oracle.py:99-101`). Under `Run.Valid`, with `M = toeplitz 60 s f`,
`p_j = r_j / W` and `q_j = r_{59-j} / W` (exact `W`):
* both mixtures are feasible;
* every row payoff `(M q)_i` and every column payoff `(pᵀ M)_j` lies within
  `3.58e-12 = 3.54e-12 + 4e-14` of the stored `v`;
* hence `v - 4e-12 ≤ lowerBound M p ≤ value M ≤ upperBound M q ≤ v + 4e-12`;
* the stored interval `[v - 1e-10, v + 1e-10]` contains both saddle bounds and
  the value, and its width `2e-10` is within the `1e-6` gate.

The doc says the bound is used in `kernel.c`; the shortcut exists only in
`leap.rs` and `leap_oracle.py` (`src/dth/fast_kernel.c` always forms `M q`). -/
theorem Valid.enclosure (hR : R.Valid) :
    mixP 60 R.r ∈ MatrixGame.simplex (Fin 60) ∧ mixQ 60 R.r ∈ MatrixGame.simplex (Fin 60) ∧
    (∀ i, |(toeplitz 60 R.s R.f *ᵥ mixQ 60 R.r) i - R.v| < 358 / 10 ^ 14) ∧
    (∀ j, |(mixP 60 R.r ᵥ* toeplitz 60 R.s R.f) j - R.v| < 358 / 10 ^ 14) ∧
    R.v - 4 / 10 ^ 12 ≤ MatrixGame.lowerBound (toeplitz 60 R.s R.f) (mixP 60 R.r) ∧
    MatrixGame.lowerBound (toeplitz 60 R.s R.f) (mixP 60 R.r) ≤
      MatrixGame.value (toeplitz 60 R.s R.f) ∧
    MatrixGame.value (toeplitz 60 R.s R.f) ≤
      MatrixGame.upperBound (toeplitz 60 R.s R.f) (mixQ 60 R.r) ∧
    MatrixGame.upperBound (toeplitz 60 R.s R.f) (mixQ 60 R.r) ≤ R.v + 4 / 10 ^ 12 ∧
    |MatrixGame.value (toeplitz 60 R.s R.f) - R.v| < 358 / 10 ^ 14 ∧
    MatrixGame.value (toeplitz 60 R.s R.f) ∈ Set.Icc (R.v - certRadius) (R.v + certRadius) ∧
    MatrixGame.lowerBound (toeplitz 60 R.s R.f) (mixP 60 R.r) ∈
      Set.Icc (R.v - certRadius) (R.v + certRadius) ∧
    MatrixGame.upperBound (toeplitz 60 R.s R.f) (mixQ 60 R.r) ∈
      Set.Icc (R.v - certRadius) (R.v + certRadius) ∧
    2 * certRadius ≤ 1 / 10 ^ 6 := by
  set M := toeplitz 60 R.s R.f
  have hq : mixQ 60 R.r ∈ MatrixGame.simplex (Fin 60) :=
    mixQ_mem_simplex R.r (fun j hj => hR.r_nonneg j hj) hR.W_pos
  have hp : mixP 60 R.r ∈ MatrixGame.simplex (Fin 60) := mem_simplex_rev hq
  have hlast := hR.last_row_close
  have hrow : ∀ i, |(M *ᵥ mixQ 60 R.r) i - R.v| < 358 / 10 ^ 14 := fun i => by
    have h1 := (hR.spread i ⟨59, by norm_num⟩)
    have h2 := (hR.spread ⟨59, by norm_num⟩ i).1
    have h3 := hlast.2
    rw [abs_sub_comm] at h3
    rw [abs_lt] at h3 ⊢
    constructor <;> linarith [h1.1, h1.2]
  have hcol : ∀ j, |(mixP 60 R.r ᵥ* M) j - R.v| < 358 / 10 ^ 14 := fun j => by
    have : (mixP 60 R.r ᵥ* M) j = (M *ᵥ mixQ 60 R.r) j.rev :=
      vecMul_rev_eq_mulVec R.s R.f (mixQ 60 R.r) j
    rw [this]; exact hrow _
  have hlo : R.v - 4 / 10 ^ 12 ≤ MatrixGame.lowerBound M (mixP 60 R.r) :=
    (MatrixGame.le_lowerBound_iff M _ _).2 fun j => by
      have := (abs_lt.1 (hcol j)).1; linarith
  have hhi : MatrixGame.upperBound M (mixQ 60 R.r) ≤ R.v + 4 / 10 ^ 12 :=
    (MatrixGame.upperBound_le_iff M _ _).2 fun i => by
      have := (abs_lt.1 (hrow i)).2; linarith
  obtain ⟨hL, hU⟩ := MatrixGame.certificate_encloses_value M hp hq
  have hlo' : R.v - 358 / 10 ^ 14 < MatrixGame.lowerBound M (mixP 60 R.r) := by
    obtain ⟨j, hj⟩ := MatrixGame.exists_lowerBound_eq M (mixP 60 R.r)
    rw [hj]; have := (abs_lt.1 (hcol j)).1; linarith
  have hhi' : MatrixGame.upperBound M (mixQ 60 R.r) < R.v + 358 / 10 ^ 14 := by
    obtain ⟨i, hi⟩ := MatrixGame.exists_upperBound_eq M (mixQ 60 R.r)
    rw [hi]; have := (abs_lt.1 (hrow i)).2; linarith
  have hval : |MatrixGame.value M - R.v| < 358 / 10 ^ 14 := by
    rw [abs_lt]; constructor <;> linarith
  have hc : 4 / 10 ^ 12 ≤ certRadius := by unfold certRadius; norm_num
  refine ⟨hp, hq, hrow, hcol, hlo, hL, hU, hhi, hval, ⟨by linarith, by linarith⟩,
    ⟨by linarith, by linarith⟩, ⟨by linarith, by linarith⟩, by unfold certRadius; norm_num⟩

end Run

/-! ## Non-vacuity: an exact run -/

/-- A run with `s ≡ 1`, `f = 0`: `d = 1`, `b ≡ 0`, `r = (1, 0, …, 0)`, `W = 1`,
`v = 1`, every operation exact. -/
noncomputable def exampleRun : Run where
  s := fun _ => 1
  f := 0
  dh := 1
  nd := fun _ => 0
  b := fun _ => 0
  prod := fun _ _ => 0
  acc := fun _ _ => 0
  r := fun j => if j = 0 then 1 else 0
  wacc := fun t => if t = 60 then 1 else 0
  y := 1
  v := 1

theorem exampleRun_valid : exampleRun.Valid where
  s_bound k _ := by simp [exampleRun]
  f_bound := by simp [exampleRun]
  dh_round := by simpa [exampleRun] using Rounds.refl 1
  dh_ne := by simp [exampleRun]
  nd_round m _ _ := by simpa [exampleRun] using Rounds.refl 0
  b_round m _ _ := by simpa [exampleRun] using Rounds.refl 0
  prod_round k t _ _ _ := by simpa [exampleRun] using Rounds.refl 0
  acc_zero k := rfl
  acc_step k t _ _ _ := by simpa [exampleRun] using Rounds.refl 0
  r_zero := by simp [exampleRun]
  r_eq k hk _ := by simp [exampleRun]; omega
  r_nonneg j _ := by simp only [exampleRun]; split_ifs <;> norm_num
  wacc_zero := by simp [exampleRun]
  wacc_step t ht := by
    simp only [exampleRun]
    by_cases h : t = 59
    · subst h; simpa using Rounds.refl 1
    · have h1 : t ≠ 60 := by omega
      have h2 : t + 1 ≠ 60 := by omega
      have h3 : 59 - t ≠ 0 := by omega
      simpa [h1, h2, h3] using Rounds.refl 0
  y_round := by simpa [exampleRun] using Rounds.refl 1
  v_round := by simpa [exampleRun] using Rounds.refl 1

example : |MatrixGame.value (toeplitz 60 exampleRun.s exampleRun.f) - exampleRun.v| <
    358 / 10 ^ 14 :=
  exampleRun_valid.enclosure.2.2.2.2.2.2.2.2.1

/-! ## Exact arithmetic: the row difference and the equalizer recurrence -/

/-- **Adjacent row difference** (claim CRATES-REC-1, `LEAP_CERTIFICATE.md:50-59,81`,
code `leap.rs:392-413`): for every vector `q` and row `i ∈ 0..58`,
`(M q)_i - (M q)_{i+1} = d q_i + ∑_{j>i} Δ_{j-i} q_j` with `d = s_0 - f`
and `Δ_m = s_m - s_{m-1}`. It specializes `Toeplitz.mulVec_sub_succ` to 60 actions. -/
theorem adjacent_row_difference (s : ℕ → ℝ) (f : ℝ) (q : Fin 60 → ℝ) (i : Fin 60)
    (hi : i.val + 1 < 60) :
    (toeplitz 60 s f *ᵥ q) i - (toeplitz 60 s f *ᵥ q) ⟨i.val + 1, hi⟩ =
      (s 0 - f) * q i +
        ∑ j ∈ univ.filter (i < ·), (s (j.val - i.val) - s (j.val - i.val - 1)) * q j :=
  mulVec_sub_succ s f q i hi

/-- **Equalizer recurrence** (claim CRATES-REC-2, `LEAP_CERTIFICATE.md:22-36,79-98`,
code `leap.rs:392-413`, `leap_oracle.py:85-90`). If `d = s_0 - f ≠ 0`, `r_0 = 1`,
`r_k = ∑_{j<k} b_{k-j} r_j` with `b_m = (s_{m-1} - s_m) / d` for `k ∈ 1..59`,
and `W = ∑_{j<60} r_j ≠ 0`, then `q_j = r_{59-j} / W` makes every row pay
`f + d / W`, and `p_j = r_j / W` makes every column pay `f + d / W`. -/
theorem equalizer_of_recurrence (s : ℕ → ℝ) (f : ℝ) (r : ℕ → ℝ) (hd : s 0 - f ≠ 0)
    (hr0 : r 0 = 1)
    (hrec : ∀ k, 1 ≤ k → k < 60 →
      r k = ∑ j ∈ range k, (s (k - j - 1) - s (k - j)) / (s 0 - f) * r j)
    (hW : wsum 60 r ≠ 0) :
    (∀ i, (toeplitz 60 s f *ᵥ mixQ 60 r) i = f + (s 0 - f) / wsum 60 r) ∧
      (∀ j, (mixP 60 r ᵥ* toeplitz 60 s f) j = f + (s 0 - f) / wsum 60 r) := by
  have hres : ∀ k, 1 ≤ k → k < 60 → residual s f r k = 0 := fun k hk1 hk => by
    have h := residual_decomposition (s 0 - f) (s 0 - f) (fun m => s m - s (m - 1))
      (fun m => (s (m - 1) - s m) / (s 0 - f)) r k
    have h1 : ∀ j ∈ range k, ((s 0 - f) * ((s (k - j - 1) - s (k - j)) / (s 0 - f)) +
        (s (k - j) - s (k - j - 1))) * r j = 0 := fun j _ => by
      have : (s 0 - f) * ((s (k - j - 1) - s (k - j)) / (s 0 - f)) = s (k - j - 1) - s (k - j) := by
        field_simp
      rw [this]; ring
    show (s 0 - f) * r k + ∑ j ∈ range k, (s (k - j) - s (k - j - 1)) * r j = 0
    rw [h, Finset.sum_eq_zero h1, ← hrec k hk1 hk]; ring
  have hadj : ∀ (i : Fin 60) (hi : i.val + 1 < 60),
      |(toeplitz 60 s f *ᵥ mixQ 60 r) i - (toeplitz 60 s f *ᵥ mixQ 60 r) ⟨i.val + 1, hi⟩| ≤ 0 :=
    fun i hi => by
      rw [mulVec_sub_succ_mixQ, hres _ (by omega) (by omega), zero_div, abs_zero]
  have hlast := mulVec_last_mixQ s f r (by norm_num : 1 ≤ 60) hr0 hW
  have hrow : ∀ i, (toeplitz 60 s f *ᵥ mixQ 60 r) i = f + (s 0 - f) / wsum 60 r := fun i => by
    have hi := i.isLt
    have h := abs_sub_le_of_adjacent _ hadj i ⟨59, by norm_num⟩ (by rw [Fin.le_def]; simp; omega)
    rw [mul_zero, abs_nonpos_iff, sub_eq_zero] at h
    rw [h]; exact hlast
  refine ⟨hrow, fun j => ?_⟩
  rw [show mixP 60 r = fun i => mixQ 60 r i.rev from rfl, vecMul_rev_eq_mulVec]
  exact hrow _

/-- **Equalizer value** (claim CRATES-REC-2, `LEAP_CERTIFICATE.md:92-98`): if the
recurrence weights are also nonnegative, `(p, q)` is a saddle point and the
stage value is `f + (s_0 - f) / W`. -/
theorem value_of_recurrence (s : ℕ → ℝ) (f : ℝ) (r : ℕ → ℝ) (hd : s 0 - f ≠ 0)
    (hr0 : r 0 = 1)
    (hrec : ∀ k, 1 ≤ k → k < 60 →
      r k = ∑ j ∈ range k, (s (k - j - 1) - s (k - j)) / (s 0 - f) * r j)
    (hnn : ∀ j < 60, 0 ≤ r j) :
    MatrixGame.value (toeplitz 60 s f) = f + (s 0 - f) / wsum 60 r := by
  have hW1 : 1 ≤ wsum 60 r := by
    have := Finset.single_le_sum (f := r) (fun j hj => hnn j (by simpa using hj))
      (show 0 ∈ range 60 by simp)
    rw [hr0] at this; exact this
  have hWpos : 0 < wsum 60 r := by linarith
  obtain ⟨hrow, hcol⟩ := equalizer_of_recurrence s f r hd hr0 hrec hWpos.ne'
  have hq := mixQ_mem_simplex r hnn hWpos
  have hp : mixP 60 r ∈ MatrixGame.simplex (Fin 60) := mem_simplex_rev hq
  exact (MatrixGame.value_of_equalizers _ hp hq hcol hrow).2

/-- The exact run satisfies the recurrence hypotheses (non-vacuity). -/
example : MatrixGame.value (toeplitz 60 (fun _ => (1:ℝ)) 0) =
    0 + ((fun _ => (1:ℝ)) 0 - 0) / wsum 60 (fun j => if j = 0 then (1:ℝ) else 0) :=
  value_of_recurrence _ _ _ (by norm_num) (by simp)
    (fun k hk _ => by simp; omega) (fun j _ => by split_ifs <;> norm_num)

/-! ## Stage-matrix conventions of the kernels -/

/-- The `leap.rs` cell (zero-based `i` Dropper, `j` Checker):
`M[i,j] = s[j-i]` if `j ≥ i`, else `f` (`leap.rs:1-12`, `leap_oracle.py:26-30`). -/
def leapCell (s : ℕ → ℝ) (f : ℝ) (i j : ℕ) : ℝ := if i ≤ j then s (j - i) else f

/-- The `abstract_solver` cell with one-based actions `d, c`: failure if `c < d`,
else `success[c - d]` (`abstract_solver/src/lib.rs:387-402`). -/
def abstractCell (succ : ℕ → ℝ) (fail : ℝ) (d c : ℕ) : ℝ := if c < d then fail else succ (c - d)

/-- The legacy `payoff.rs` cell with one-based `drop, check`: `success_values[st]`
with `st = check - drop + 1` when `check ≥ drop` (`payoff.rs:20-42`). Index `0`
of `success_values` is unused. -/
def payoffRsCell (sv : ℕ → ℝ) (fail : ℝ) (drop check : ℕ) : ℝ :=
  if drop ≤ check then sv (check - drop + 1) else fail

/-- **Stage-matrix conventions** (claim CRATES-TOEP-1): the three kernel cells are
the Toeplitz stage matrix `toeplitz n s f`. The zero-based `leap.rs` and the
one-based `abstract_solver` index the success vector by `k = ST - 1`; the legacy
`payoff.rs` indexes it by `ST = check - drop + 1`, so its vector is `s` shifted
by one. -/
theorem stage_cells_eq_toeplitz {n : ℕ} (s : ℕ → ℝ) (f : ℝ) (sv : ℕ → ℝ) (d c : Fin n) :
    leapCell s f d.val c.val = toeplitz n s f d c ∧
      abstractCell s f (d.val + 1) (c.val + 1) = toeplitz n s f d c ∧
      payoffRsCell sv f (d.val + 1) (c.val + 1) = toeplitz n (fun k => sv (k + 1)) f d c := by
  by_cases h : d ≤ c
  · have hv : d.val ≤ c.val := h
    have h' : ¬ c.val + 1 < d.val + 1 := by omega
    have h'' : d.val + 1 ≤ c.val + 1 := by omega
    simp only [leapCell, abstractCell, payoffRsCell, toeplitz_apply, h, hv, h', h'',
      ↓reduceIte, true_and]
    refine ⟨by congr 1; omega, by congr 1; omega⟩
  · have hv : ¬ d.val ≤ c.val := h
    have h' : c.val + 1 < d.val + 1 := by omega
    have h'' : ¬ d.val + 1 ≤ c.val + 1 := by omega
    simp only [leapCell, abstractCell, payoffRsCell, toeplitz_apply, h, hv, h', h'',
      ↓reduceIte, and_self]

/-- **Payoff range** (claim CRATES-TOEP-1, `leap.rs:357-367`): with child values in
`[-1, 1]` and revival probability `ρ ∈ [0, 1]`, the success payoffs `s_k = -child_k`
and the failure payoff `f = ρ (-child_f) + (1 - ρ)` lie in `[-1, 1]`, so every
stage entry does. -/
theorem stage_payoffs_le_one {n : ℕ} (child : ℕ → ℝ) (childF rev : ℝ)
    (hs : ∀ k, |child k| ≤ 1) (hf : |childF| ≤ 1) (hr0 : 0 ≤ rev) (hr1 : rev ≤ 1)
    (d c : Fin n) :
    |toeplitz n (fun k => -child k) (rev * -childF + (1 - rev)) d c| ≤ 1 := by
  rw [toeplitz_apply]
  split_ifs
  · rw [abs_neg]; exact hs _
  · rw [abs_le] at hf ⊢
    constructor <;> nlinarith [hf.1, hf.2]

/-! ## The floating-point environment -/

/-- **Nonfinite values propagate** (claim CRATES-CERT-6(a),
`LEAP_CERTIFICATE.md:72-75,113-114`). Model binary64 values as a type `α` with a
predicate `fin`. If each step maps a nonfinite value to a nonfinite value (true of
IEEE multiply, add and fused multiply-add for an infinite or NaN operand), a
nonfinite value at step `t₀` stays nonfinite. Round-to-nearest turns overflow into
an infinity; that part is an environment hypothesis, not a theorem. -/
theorem nonfinite_propagates {α : Type*} (fin : α → Prop) (x : ℕ → α)
    (hstep : ∀ t, ¬ fin (x t) → ¬ fin (x (t + 1))) {t₀ t : ℕ} (h : t₀ ≤ t)
    (h0 : ¬ fin (x t₀)) : ¬ fin (x t) := by
  induction t, h using Nat.le_induction with
  | base => exact h0
  | succ t _ ih => exact hstep t ih

/-- The contrapositive: a finite last value forces every earlier value finite. -/
theorem fin_of_fin_last {α : Type*} (fin : α → Prop) (x : ℕ → α)
    (hstep : ∀ t, ¬ fin (x t) → ¬ fin (x (t + 1))) {n : ℕ} (hn : fin (x n)) :
    ∀ t ≤ n, fin (x t) := fun t ht => by
  by_contra h; exact nonfinite_propagates fin x hstep ht h hn

/-- The shortcut's intermediate values over an abstract binary64 type (`leap.rs:404-426`):
`acc k (t+1) = fma(b (k-t), r t, acc k t)`, `r k = acc k k`,
`wacc (t+1) = wacc t + clip (r (59-t))`, where `clip` maps a nonfinite weight
to NaN. -/
structure ExtRun (α : Type*) where
  fin : α → Prop
  fma : α → α → α → α
  add : α → α → α
  clip : α → α
  b : ℕ → α
  acc : ℕ → ℕ → α
  r : ℕ → α
  wacc : ℕ → α
  fma_absorb : ∀ a x y, ¬ fin a ∨ ¬ fin x ∨ ¬ fin y → ¬ fin (fma a x y)
  add_absorb : ∀ x y, ¬ fin x ∨ ¬ fin y → ¬ fin (add x y)
  clip_absorb : ∀ x, ¬ fin x → ¬ fin (clip x)
  acc_step : ∀ k t, acc k (t + 1) = fma (b (k - t)) (r t) (acc k t)
  r_eq : ∀ k, 1 ≤ k → k < 60 → r k = acc k k
  wacc_step : ∀ t, wacc (t + 1) = add (wacc t) (clip (r (59 - t)))

/-- **Overflow exclusion** (claim CRATES-CERT-6(a)): if the weight sum is finite,
every weight, every recurrence accumulator and every partial sum is finite. So a
finite `Ŵ` (the gate `sum.is_finite()` of `leap.rs:453`) rules out every
overflow, and each operation obeys the finite rounding model `Rounds`. -/
theorem ExtRun.all_finite {α : Type*} (E : ExtRun α) (h : E.fin (E.wacc 60)) :
    (∀ t ≤ 60, E.fin (E.wacc t)) ∧ (∀ j < 60, E.fin (E.r j)) ∧
      (∀ k t, 1 ≤ k → k < 60 → t ≤ k → E.fin (E.acc k t)) := by
  have hw : ∀ t ≤ 60, E.fin (E.wacc t) :=
    fin_of_fin_last E.fin E.wacc (fun t ht => by
      rw [E.wacc_step]; exact E.add_absorb _ _ (Or.inl ht)) h
  have hr : ∀ j < 60, E.fin (E.r j) := fun j hj => by
    by_contra hc
    have h1 := hw (59 - j + 1) (by omega)
    rw [E.wacc_step, show 59 - (59 - j) = j by omega] at h1
    exact E.add_absorb _ _ (Or.inr (E.clip_absorb _ hc)) h1
  refine ⟨hw, hr, fun k t hk1 hk htk => ?_⟩
  have hk' : E.fin (E.acc k k) := E.r_eq k hk1 hk ▸ hr k hk
  exact fin_of_fin_last E.fin (E.acc k) (fun t ht => by
    rw [E.acc_step]; exact E.fma_absorb _ _ _ (Or.inr (Or.inr ht))) hk' t htk

/-- **Min and max do not round** (claims CPP-DOC-FP-1 and CANONICAL-FP-1(a),
`complete_tablebase.py:386-412`, `BUILD.md:101-104`): over any linear order, a
left fold of `min` returns one of its inputs. The model has no NaN and treats
`-0` and `+0` as one value; IEEE `min(-0, +0)` may return either zero, so bit
equality across backends also needs a common tie rule for signed zeros. -/
theorem foldl_min_mem {α : Type*} [LinearOrder α] (a : α) (l : List α) :
    l.foldl min a ∈ a :: l := by
  induction l generalizing a with
  | nil => simp
  | cons x l ih =>
    rcases min_choice a x with h | h
    · have := ih (min a x); rw [h] at this
      simp only [List.foldl_cons, h, List.mem_cons] at this ⊢
      rcases this with h1 | h1
      · exact Or.inl h1
      · exact Or.inr (Or.inr h1)
    · have := ih (min a x); rw [h] at this
      simp only [List.foldl_cons, h, List.mem_cons] at this ⊢
      rcases this with h1 | h1
      · exact Or.inr (Or.inl h1)
      · exact Or.inr (Or.inr h1)

/-- The `max` version of `foldl_min_mem`. -/
theorem foldl_max_mem {α : Type*} [LinearOrder α] (a : α) (l : List α) :
    l.foldl max a ∈ a :: l :=
  foldl_min_mem (α := αᵒᵈ) a l

/-- **Min and max are order-independent** (claims CPP-DOC-FP-1 and
CANONICAL-FP-1(a)): permuting the inputs of a `min` or `max` fold does not
change its result. -/
theorem foldl_min_max_perm {α : Type*} [LinearOrder α] {l l' : List α} (h : l.Perm l')
    (a : α) : l.foldl min a = l'.foldl min a ∧ l.foldl max a = l'.foldl max a :=
  ⟨h.foldl_eq' (fun x _ y _ z => min_right_comm z x y) a,
    h.foldl_eq' (fun x _ y _ z => max_right_comm z x y) a⟩

/-- **Partial-pivoting factor bound** (claim DTH-SUP-2, `DTH_COMPLETE_PARITY.md:94-102`,
`complete_tablebase.py:429-459`). Let row `p` hold a maximum magnitude of column
`col` over rows `col..n-1` (the first maximum, after the swap) with
`|a p| ≥ 1e-12`. Then every elimination factor `a r / a p` has magnitude at most
one, and so does its rounding under any monotone rounding `fl` that fixes `±1`
(round-to-nearest is monotone and `±1` are representable). The additive model
`Rounds` alone does not give this bound. -/
theorem pivot_factor_le_one {n col p : ℕ} (a : ℕ → ℝ)
    (hmax : ∀ r, col ≤ r → r < n → |a r| ≤ |a p|) (hpiv : 1 / 10 ^ 12 ≤ |a p|)
    {fl : ℝ → ℝ} (hmono : Monotone fl) (h1 : fl 1 = 1) (hm1 : fl (-1) = -1)
    (r : ℕ) (hr1 : col ≤ r) (hr2 : r < n) :
    |a r / a p| ≤ 1 ∧ |fl (a r / a p)| ≤ 1 := by
  have hp : 0 < |a p| := lt_of_lt_of_le (by norm_num) hpiv
  have hq : |a r / a p| ≤ 1 := by
    rw [abs_div, div_le_one hp]; exact hmax r hr1 hr2
  refine ⟨hq, ?_⟩
  rw [abs_le] at hq ⊢
  exact ⟨hm1 ▸ hmono hq.1, h1 ▸ hmono hq.2⟩

/-- Class assembly `F = ρ (-v) + (1 - ρ)` with one rounded multiply, one rounded
subtraction and one rounded addition (`DTH_COMPLETE_PARITY.md:68-73`) stays within
`4u < 2e-14` of the exact value when `|v| ≤ 1` and `ρ ∈ [0, 1]`. This bounds one
assembly from identical inputs; it does not derive the cross-platform fallback
bound `2e-14` of claim DTH-FP-1. -/
theorem assembly_error {rev v m c F : ℝ} (hr0 : 0 ≤ rev) (hr1 : rev ≤ 1) (hv : |v| ≤ 1)
    (hm : Rounds (rev * -v) m) (hc : Rounds (1 - rev) c) (hF : Rounds (m + c) F) :
    |F - (rev * -v + (1 - rev))| ≤ 4 * u ∧ 4 * u < 2 / 10 ^ 14 := by
  have hu := u_le'; have hu0 := u_pos; have he := eta_le; have he0 := eta_pos
  have hrv : |rev * -v| ≤ rev := by
    rw [abs_mul, abs_neg, abs_of_nonneg hr0]
    calc rev * |v| ≤ rev * 1 := mul_le_mul_of_nonneg_left hv hr0
      _ = rev := mul_one _
  have h1 := hm.abs_sub_le
  have h2 := hc.abs_sub_le
  rw [abs_of_nonneg (by linarith : 0 ≤ 1 - rev)] at h2
  have h3 := hF.abs_sub_le
  have hmr : u * |rev * -v| ≤ u * rev := mul_le_mul_of_nonneg_left hrv hu0.le
  have hsum : |m + c - (rev * -v + (1 - rev))| ≤ u + 2 * eta := by
    calc |m + c - (rev * -v + (1 - rev))| = |(m - rev * -v) + (c - (1 - rev))| := by ring_nf
      _ ≤ |m - rev * -v| + |c - (1 - rev)| := abs_add_le _ _
      _ ≤ u + 2 * eta := by nlinarith
  have hF0 : |rev * -v + (1 - rev)| ≤ 1 := by
    have := abs_add_le (rev * -v) (1 - rev)
    rw [abs_of_nonneg (by linarith : 0 ≤ 1 - rev)] at this; linarith
  have hmc : |m + c| ≤ 1 + u + 2 * eta := by
    have := abs_sub_abs_le_abs_sub (m + c) (rev * -v + (1 - rev)); linarith
  have h4 : u * |m + c| ≤ u * (1 + u + 2 * eta) := mul_le_mul_of_nonneg_left hmc hu0.le
  refine ⟨?_, by linarith⟩
  calc |F - (rev * -v + (1 - rev))| = |(F - (m + c)) + (m + c - (rev * -v + (1 - rev)))| := by
        ring_nf
    _ ≤ |F - (m + c)| + |m + c - (rev * -v + (1 - rev))| := abs_add_le _ _
    _ ≤ 4 * u := by
        have hul : 1 / 10 ^ 17 ≤ u := by unfold u; norm_num
        have h5 : u * u ≤ u * (12 / 10 ^ 17) := mul_le_mul_of_nonneg_left hu hu0.le
        have h6 : u * eta ≤ u * (1 / 10 ^ 30) := mul_le_mul_of_nonneg_left he hu0.le
        linarith

end Formal.Float
