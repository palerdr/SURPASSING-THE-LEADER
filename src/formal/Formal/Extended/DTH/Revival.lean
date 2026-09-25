import Formal.DTH.Rules
import Formal.MatrixGame.Transform

/-!
# The frozen revival surface

This module proves the properties of the repository-wide revival surface
(`docs/REVIVAL_MODEL.md`) that the solvers and their documents rely on.

Conventions.

* `Formal.DTH.revival s t` (from `Formal/DTH/Rules.lean`) is the seconds surface
  `0.95 (1 - s/240) 0.75^(t/60)` on the eligible region and `0` elsewhere. `s`
  is the ST in the vial before the failed check, `t` the accrued TTD, both
  natural numbers of seconds. The dose is `q = s + 60`.
* `Survives s t` is the dose form `s + 60 < 300 ∧ t + (s + 60) ≤ 300`, the form
  of `src/dth/solver.py:63-65`.
* `bucketRevival C F τ q` mirrors `AbstractRuleset.revival_probability`
  (`src/abstract/rules.py:178-189`) and the Rust `revival_probability`
  (`src/crates/abstract_solver/src/lib.rs:153-162`): it takes the prior TTD
  `τ` and the dose `q` in bucket units, recovers the load `q - F`, and clamps
  to `[0, 1]`.
* `refereeProb` mirrors `Referee.compute_survival_probability`
  (`src/stl/engine/game.py:171-200`).
* All probabilities are real numbers. `0.75^(t/60)` is `Real.rpow`. Where a
  claim concerns binary64 results, the module states an explicit abstract
  rounding model and proves the claim inside that model. No theorem here
  asserts IEEE bit-level behaviour.
* "Rounds to `e` at 4 decimals" means `e - 1/20000 ≤ x < e + 1/20000`
  (round half up), `RoundsTo4`.

Main results: eligibility equivalences, positivity iff eligibility, the range
and strict bounds, monotonicity, the zero set of the dose factor, the true
minimum `P(239, 1)`, the table and ledger values, the half-life
`60 ln 2 / ln (4/3) ∈ (144.55, 144.6)` (the docs formerly stated `144.3`, now corrected to `144.6`), the
`0.748` rounding error, the death-count bound `deaths ≤ t/60` (the docs
formerly reversed it), the referee floor threshold `ln 0.4 / ln 0.88 ∈ [7.165, 7.168)`, the
frontier values, bucket invariance, independence of the game graph from the
numeric surface, and `V ∈ [-1, 1]`.
-/

open Real

set_option exponentiation.threshold 3000

namespace Formal.DTH

namespace Revival

/-! ## Real powers of rational bases -/

/-- `(a ^ e) ^ n = a ^ k` when `e n = k`. -/
theorem rpow_pow_eq {a e : ℝ} (ha : 0 ≤ a) {k n : ℕ} (he : e * n = k) :
    (a ^ e) ^ n = a ^ k := by
  rw [← Real.rpow_natCast, ← Real.rpow_mul ha, he, Real.rpow_natCast]

theorem le_rpow_of_pow_le {a e lo : ℝ} (ha : 0 ≤ a) {k n : ℕ} (hn : n ≠ 0)
    (he : e * n = k) (hlo : 0 ≤ lo) (h : lo ^ n ≤ a ^ k) : lo ≤ a ^ e := by
  rw [← pow_le_pow_iff_left₀ hlo (Real.rpow_nonneg ha e) hn, rpow_pow_eq ha he]
  exact h

theorem rpow_le_of_le_pow {a e hi : ℝ} (ha : 0 ≤ a) {k n : ℕ} (hn : n ≠ 0)
    (he : e * n = k) (hhi : 0 ≤ hi) (h : a ^ k ≤ hi ^ n) : a ^ e ≤ hi := by
  rw [← pow_le_pow_iff_left₀ (Real.rpow_nonneg ha e) hhi hn, rpow_pow_eq ha he]
  exact h

theorem lt_rpow_of_pow_lt {a e lo : ℝ} (ha : 0 ≤ a) {k n : ℕ} (hn : n ≠ 0)
    (he : e * n = k) (hlo : 0 ≤ lo) (h : lo ^ n < a ^ k) : lo < a ^ e := by
  rw [← pow_lt_pow_iff_left₀ hlo (Real.rpow_nonneg ha e) hn, rpow_pow_eq ha he]
  exact h

theorem rpow_lt_of_lt_pow {a e hi : ℝ} (ha : 0 ≤ a) {k n : ℕ} (hn : n ≠ 0)
    (he : e * n = k) (hhi : 0 ≤ hi) (h : a ^ k < hi ^ n) : a ^ e < hi := by
  rw [← pow_lt_pow_iff_left₀ (Real.rpow_nonneg ha e) hhi hn, rpow_pow_eq ha he]
  exact h

theorem rpow75_nat (n : ℕ) : (0.75 : ℝ) ^ (n : ℝ) = 0.75 ^ n := Real.rpow_natCast _ _

theorem rpow75_pos (e : ℝ) : 0 < (0.75 : ℝ) ^ e := Real.rpow_pos_of_pos (by norm_num) e

theorem rpow75_le_one {e : ℝ} (he : 0 ≤ e) : (0.75 : ℝ) ^ e ≤ 1 :=
  Real.rpow_le_one (by norm_num) (by norm_num) he

/-- `0.75^e` is antitone in the exponent. -/
theorem rpow75_anti {e e' : ℝ} (h : e ≤ e') : (0.75 : ℝ) ^ e' ≤ 0.75 ^ e :=
  Real.rpow_le_rpow_of_exponent_ge (by norm_num) (by norm_num) h

/-- `0.75^e` is strictly antitone in the exponent. -/
theorem rpow75_strictAnti {e e' : ℝ} (h : e < e') : (0.75 : ℝ) ^ e' < 0.75 ^ e :=
  Real.rpow_lt_rpow_of_exponent_gt (by norm_num) (by norm_num) h

/-! ## Eligibility -/

/-- **Eligibility equivalence** (claims CPP-DOC-RULE-1, DTH-RULE-2,
CANONICAL-RULE-3). The dose form `q < 300 ∧ t + q ≤ 300` with `q = s + 60`
(`src/dth/solver.py:63-65`, `docs/REVIVAL_MODEL.md:22-24`) is equivalent to the
integer form `s ≤ 239 ∧ s + t ≤ 240` that `src/dth_cpp/exact.cpp:45-50`
implements (`src/dth_cpp/BUILD.md:344-353`). The equivalence holds for every
natural `s`, `t`, not only on the domain `s ≤ 299`, `t ≤ 300`. -/
theorem survives_iff_integer_form (s t : ℕ) :
    (s + 60 < 300 ∧ t + (s + 60) ≤ 300) ↔ (s ≤ 239 ∧ s + t ≤ 240) := by
  omega

/-- The STL form `s + 60 < 300 ∧ s + t + 60 ≤ 300` of
`src/stl/solver/leap_profiles.py:29-30` is the same predicate (claim STL-RULE-5). -/
theorem stl_eligible_iff (s t : ℕ) : (s + 60 < 300 ∧ s + t + 60 ≤ 300) ↔ Survives s t := by
  unfold Survives; omega

/-- Boundary instances of claim CPP-DOC-RULE-1 (`src/dth_cpp/BUILD.md:390-395`):
`(0,0)`, `(239,0)`, `(0,240)`, `(180,60)` are eligible and `(240,0)`, `(239,2)`,
`(0,241)` are fatal. -/
theorem survives_boundary :
    Survives 0 0 ∧ Survives 239 0 ∧ Survives 0 240 ∧ Survives 180 60 ∧
      ¬ Survives 240 0 ∧ ¬ Survives 239 2 ∧ ¬ Survives 0 241 := by
  decide

/-- Every ST of `240` or more is fatal, whatever the TTD (claim CPP-DOC-RULE-1). -/
theorem not_survives_of_ge {s : ℕ} (t : ℕ) (hs : 240 ≤ s) : ¬ Survives s t := by
  unfold Survives; omega

/-- Equality `t + q = 300` stays eligible when `q < 300` (claims CPP-DOC-RULE-1,
DTH-RULE-2, ABSTRACT-REV-1). The lethal dose `q = 300` at `t = 0` is fatal
although `t + q = 300`. -/
theorem survives_of_total_eq {s t : ℕ} (hq : s + 60 < 300) (h : t + (s + 60) = 300) :
    Survives s t := ⟨hq, h.le⟩

/-- Eligibility bounds the TTD: `t ≤ 240`, that is, `4` death-minutes
(claims ABSTRACT-REV-4 (d), CANONICAL-REV-4, DTH-RULE-4). -/
theorem ttd_le_of_survives {s t : ℕ} (h : Survives s t) : t ≤ 240 := by
  unfold Survives at h; omega

/-- At `t = 238`, revival requires `s ≤ 2` (claim CANONICAL-REV-10,
`docs/REVIVAL_MODEL.md:171-174`). -/
theorem survives_238_iff (s : ℕ) : Survives s 238 ↔ s ≤ 2 := by
  unfold Survives; omega

/-- The largest survivable ST is `239` (claims DTH-RULE-3 (b), ABSTRACT-REV-4 (a)). -/
theorem s_le_of_survives {s t : ℕ} (h : Survives s t) : s ≤ 239 := by
  unfold Survives at h; omega

/-- On the reachable TTD domain `{0} ∪ [60, ∞)`, ST `239` forces `t = 0`
(claim DTH-RULE-3 (b)). -/
theorem ttd_eq_zero_of_239 {t : ℕ} (hdom : t = 0 ∨ 60 ≤ t) (h : Survives 239 t) : t = 0 := by
  unfold Survives at h; omega

/-! ## Closed forms and the basic range -/

theorem revival_of_survives {s t : ℕ} (h : Survives s t) :
    revival s t = 0.95 * (1 - (s : ℝ) / 240) * (0.75 : ℝ) ^ ((t : ℝ) / 60) := by
  unfold revival; simp only [h, ↓reduceIte]

theorem revival_of_not_survives {s t : ℕ} (h : ¬ Survives s t) : revival s t = 0 := by
  unfold revival; simp only [h, ↓reduceIte]

/-- The surface on the grid `t = 60 k`, where `0.75^(t/60) = 0.75^k` is rational. -/
theorem revival_grid (s k : ℕ) :
    revival s (60 * k) =
      if Survives s (60 * k) then 0.95 * (1 - (s : ℝ) / 240) * (0.75 : ℝ) ^ k else 0 := by
  unfold revival
  have : ((60 * k : ℕ) : ℝ) / 60 = (k : ℝ) := by push_cast; ring
  rw [this, Real.rpow_natCast]

theorem revival_ne_zero_iff (s t : ℕ) : revival s t ≠ 0 ↔ Survives s t := by
  rw [← revival_pos_iff]
  exact ⟨fun h => lt_of_le_of_ne (revival_nonneg s t) (Ne.symm h), fun h => ne_of_gt h⟩

/-- The dose factor is at least `1/240` on an eligible profile, with equality
exactly at `s = 239` (claim CANONICAL-REV-1 (d)). -/
theorem doseFactor_ge {s t : ℕ} (h : Survives s t) : 1 / 240 ≤ 1 - (s : ℝ) / 240 := by
  have hs : (s : ℝ) ≤ 239 := by exact_mod_cast s_le_of_survives h
  linarith [show (s : ℝ) / 240 ≤ 239 / 240 by linarith]

theorem doseFactor_eq_min_iff (s : ℕ) : 1 - (s : ℝ) / 240 = 1 / 240 ↔ s = 239 := by
  constructor
  · intro h
    have : (s : ℝ) = 239 := by linarith
    exact_mod_cast this
  · rintro rfl; norm_num

/-- **Dose factor and its zero set** (claims CANONICAL-REV-1 (c), (e) and
DTH-RULE-3 (a), `docs/REVIVAL_MODEL.md:37-44,63-69`). The dose factor
`1 - s/240` equals `(300 - q)/240` with `q = s + 60`; it vanishes exactly at
`s = 240`, that is at the lethal dose `q = 300`; `240 = 300 - 60`; and it never
vanishes on an eligible profile. -/
theorem doseFactor_facts (s t : ℕ) :
    1 - (s : ℝ) / 240 = (300 - ((s + 60 : ℕ) : ℝ)) / 240 ∧
      (1 - (s : ℝ) / 240 = 0 ↔ s = 240) ∧ (s = 240 ↔ s + 60 = 300) ∧
      (300 - 60 : ℕ) = 240 ∧ (Survives s t → 1 - (s : ℝ) / 240 ≠ 0) := by
  refine ⟨by push_cast; ring, ⟨fun h => ?_, fun h => by subst h; norm_num⟩, by omega, rfl,
    fun h => ?_⟩
  · have : (s : ℝ) = 240 := by linarith
    exact_mod_cast this
  · have := doseFactor_ge h
    intro h0; rw [h0] at this; norm_num at this

/-- **Positivity iff eligibility, and the range** (claims CANONICAL-REV-1 (a),
(b), STL-RULE-5, DTH-RULE-3 (d), `docs/REVIVAL_MODEL.md:176-184`):
`P > 0 ↔ eligible`, `0 ≤ P ≤ 0.95 < 1`, and `P(0,0) = 0.95` is the maximum. -/
theorem revival_range (s t : ℕ) :
    (0 < revival s t ↔ Survives s t) ∧ 0 ≤ revival s t ∧ revival s t ≤ 0.95 ∧
      revival s t < 1 ∧ revival 0 0 = 0.95 := by
  refine ⟨revival_pos_iff s t, revival_nonneg s t, revival_le s t,
    (revival_le s t).trans_lt (by norm_num), ?_⟩
  rw [revival_of_survives (by decide)]; simp

theorem revival_lt_one (s t : ℕ) : revival s t < 1 := (revival_range s t).2.2.2.1

theorem revival_239_0 : revival 239 0 = 0.95 / 240 := by
  rw [revival_of_survives (by decide)]; norm_num

theorem revival_239_1 : revival 239 1 = 0.95 / 240 * (0.75 : ℝ) ^ ((1 : ℝ) / 60) := by
  rw [revival_of_survives (by decide)]; norm_num

theorem revival_0_240 : revival 0 240 = 0.95 * 0.75 ^ 4 := by
  rw [show (240 : ℕ) = 60 * 4 from rfl, revival_grid]; simp only [show Survives 0 (60 * 4) by decide,
    ↓reduceIte]; norm_num

/-- `0.95/240` rounds to the documented `0.003958` (claims ABSTRACT-REV-4 (a),
CANONICAL-REV-1 (d), DTH-RULE-3 (b)). -/
theorem min_dose_value : (0.0039583 : ℝ) < 0.95 / 240 ∧ (0.95 / 240 : ℝ) < 0.0039584 := by
  norm_num

/-- **Monotonicity in `s`** (claim CANONICAL-REV-2, `docs/REVIVAL_MODEL.md:182-183`):
on the whole grid, not only the eligible region. -/
theorem revival_anti_s {s s' : ℕ} (t : ℕ) (hs : s ≤ s') : revival s' t ≤ revival s t := by
  by_cases h' : Survives s' t
  · have h : Survives s t := by unfold Survives at *; omega
    rw [revival_of_survives h', revival_of_survives h]
    have : (s : ℝ) ≤ s' := by exact_mod_cast hs
    have hr := (rpow75_pos ((t : ℝ) / 60)).le
    gcongr
  · rw [revival_of_not_survives h']; exact revival_nonneg s t

/-- **Monotonicity in `t`** (claim CANONICAL-REV-2). -/
theorem revival_anti_t (s : ℕ) {t t' : ℕ} (ht : t ≤ t') : revival s t' ≤ revival s t := by
  by_cases h' : Survives s t'
  · have h : Survives s t := by unfold Survives at *; omega
    rw [revival_of_survives h', revival_of_survives h]
    have hd : 0 ≤ 1 - (s : ℝ) / 240 := le_trans (by norm_num) (doseFactor_ge h)
    have : (t : ℝ) / 60 ≤ (t' : ℝ) / 60 := by
      have : (t : ℝ) ≤ t' := by exact_mod_cast ht
      linarith
    have := rpow75_anti this
    gcongr
  · rw [revival_of_not_survives h']; exact revival_nonneg s t

/-- **Strict decrease in `s` on the eligible region** (claims CPP-CODE-RULE-2,
PAPERS-OCAML-REV-1 (c)). -/
theorem revival_strictAnti_s {s s' t : ℕ} (hs : s < s') (h' : Survives s' t) :
    revival s' t < revival s t := by
  have h : Survives s t := by unfold Survives at *; omega
  rw [revival_of_survives h', revival_of_survives h]
  have : (s : ℝ) < s' := by exact_mod_cast hs
  have hr := rpow75_pos ((t : ℝ) / 60)
  nlinarith [mul_pos hr (sub_pos.2 this)]

/-- **Strict decrease in `t` on the eligible region** (claims CPP-CODE-RULE-2,
PAPERS-OCAML-REV-1 (c)). -/
theorem revival_strictAnti_t {s t t' : ℕ} (ht : t < t') (h' : Survives s t') :
    revival s t' < revival s t := by
  have h : Survives s t := by unfold Survives at *; omega
  rw [revival_of_survives h', revival_of_survives h]
  have hd : 0 < 1 - (s : ℝ) / 240 := lt_of_lt_of_le (by norm_num) (doseFactor_ge h)
  have : (t : ℝ) / 60 < (t' : ℝ) / 60 := by
    have : (t : ℝ) < t' := by exact_mod_cast ht
    linarith
  have := rpow75_strictAnti this
  have h95 : (0 : ℝ) < 0.95 * (1 - (s : ℝ) / 240) := by positivity
  exact mul_lt_mul_of_pos_left this h95

/-! ## Lower bounds and the true minimum -/

/-- `0.75^(t/60) ≥ 0.75^4` for every eligible TTD. -/
theorem ttdFactor_ge {s t : ℕ} (h : Survives s t) : (0.75 : ℝ) ^ 4 ≤ (0.75 : ℝ) ^ ((t : ℝ) / 60) := by
  have ht : (t : ℝ) ≤ 240 := by exact_mod_cast ttd_le_of_survives h
  rw [← rpow75_nat 4]
  exact rpow75_anti (by push_cast; linarith)

/-- **Uniform lower bound** (claim CPP-CODE-RULE-2, `src/dth_cpp/exact.cpp:52-65`):
every eligible profile has `0.95/240 · 0.75^4 ≤ P ≤ 0.95`. -/
theorem revival_ge_uniform {s t : ℕ} (h : Survives s t) :
    0.95 / 240 * 0.75 ^ 4 ≤ revival s t ∧ revival s t ≤ 0.95 := by
  refine ⟨?_, revival_le s t⟩
  rw [revival_of_survives h]
  have h1 := doseFactor_ge h
  have h2 := ttdFactor_ge h
  calc (0.95 : ℝ) / 240 * 0.75 ^ 4 = 0.95 * (1 / 240) * 0.75 ^ 4 := by ring
    _ ≤ 0.95 * (1 - (s : ℝ) / 240) * (0.75 : ℝ) ^ ((t : ℝ) / 60) := by gcongr

/-- Off `s = 239`, every eligible profile has `P ≥ 0.95/240`. -/
theorem revival_ge_of_le_238 {s t : ℕ} (h : Survives s t) (hs : s ≤ 238) :
    0.95 / 240 ≤ revival s t := by
  rw [revival_of_survives h]
  have hS := (survives_iff s t).1 h
  rcases le_total t 60 with ht | ht
  · have hd : 2 / 240 ≤ 1 - (s : ℝ) / 240 := by
      have : (s : ℝ) ≤ 238 := by exact_mod_cast hs
      linarith [show (s : ℝ) / 240 ≤ 238 / 240 by linarith]
    have hr : (0.75 : ℝ) ≤ (0.75 : ℝ) ^ ((t : ℝ) / 60) := by
      have ht' : (t : ℝ) ≤ 60 := by exact_mod_cast ht
      have := rpow75_anti (e := (t : ℝ) / 60) (e' := 1) (by
        rw [div_le_one (by norm_num)]; exact ht')
      rwa [Real.rpow_one] at this
    calc (0.95 : ℝ) / 240 ≤ 0.95 * (2 / 240) * 0.75 := by norm_num
      _ ≤ 0.95 * (1 - (s : ℝ) / 240) * (0.75 : ℝ) ^ ((t : ℝ) / 60) := by gcongr
  · have hd : 60 / 240 ≤ 1 - (s : ℝ) / 240 := by
      have : (s : ℝ) ≤ 180 := by exact_mod_cast (by omega : s ≤ 180)
      linarith [show (s : ℝ) / 240 ≤ 180 / 240 by linarith]
    have hr := ttdFactor_ge h
    calc (0.95 : ℝ) / 240 ≤ 0.95 * (60 / 240) * 0.75 ^ 4 := by norm_num
      _ ≤ 0.95 * (1 - (s : ℝ) / 240) * (0.75 : ℝ) ^ ((t : ℝ) / 60) := by gcongr

/-- **The true minimum over all eligible integer profiles is `P(239, 1)`**
(settles the doubts of claims CPP-CODE-RULE-2 and PAPERS-OCAML-REV-1). -/
theorem revival_min {s t : ℕ} (h : Survives s t) : revival 239 1 ≤ revival s t := by
  have h239 : revival 239 1 ≤ 0.95 / 240 := by
    rw [revival_239_1]
    have := rpow75_le_one (e := (1 : ℝ) / 60) (by norm_num)
    have : (0 : ℝ) ≤ 0.95 / 240 := by norm_num
    nlinarith
  rcases Nat.lt_or_ge s 239 with hs | hs
  · exact h239.trans (revival_ge_of_le_238 h (by omega))
  · have hs' : s = 239 := le_antisymm (s_le_of_survives h) hs
    subst hs'
    have ht : t ≤ 1 := by unfold Survives at h; omega
    exact revival_anti_t 239 ht

/-- `P(239, 1)` is eligible, lies strictly below `0.95/240`, and rounds to the
documented `0.003939` (claims PAPERS-OCAML-REV-1 (b), CANONICAL-REV-9). It
refutes the literal claim that `0.95/240` bounds `P` below on every eligible
integer profile. -/
theorem revival_239_1_facts :
    Survives 239 1 ∧ revival 239 1 < 0.95 / 240 ∧
      0.0039393 < revival 239 1 ∧ revival 239 1 < 0.0039395 := by
  have lo : (0.995216 : ℝ) ≤ (0.75 : ℝ) ^ ((1 : ℝ) / 60) :=
    le_rpow_of_pow_le (k := 1) (n := 60) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
      (by norm_num)
  have hi : (0.75 : ℝ) ^ ((1 : ℝ) / 60) ≤ 0.995217 :=
    rpow_le_of_le_pow (k := 1) (n := 60) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
      (by norm_num)
  have hlt : (0.75 : ℝ) ^ ((1 : ℝ) / 60) < 1 :=
    Real.rpow_lt_one (by norm_num) (by norm_num) (by norm_num)
  refine ⟨by decide, ?_, ?_, ?_⟩ <;> rw [revival_239_1]
  · nlinarith
  · nlinarith
  · nlinarith

/-- **Lower bound on the reachable TTD domain** (claims PAPERS-OCAML-REV-1 (b),
DTH-RULE-3 (b)): when `t = 0` or `t ≥ 60`, which covers every TTD the game
reaches (`ttd_domain`), an eligible profile has `P ≥ 0.95/240 = P(239, 0)`. -/
theorem revival_ge_on_domain {s t : ℕ} (h : Survives s t) (hdom : t = 0 ∨ 60 ≤ t) :
    revival 239 0 ≤ revival s t := by
  rw [revival_239_0]
  rcases Nat.lt_or_ge s 239 with hs | hs
  · exact revival_ge_of_le_238 h (by omega)
  · have hs' : s = 239 := le_antisymm (s_le_of_survives h) hs
    subst hs'
    rw [ttd_eq_zero_of_239 hdom h, revival_239_0]

/-- **The literal lower bound `0.95/240` fails on integer profiles** (claim
PAPERS-OCAML-REV-1 (b), `src/dth_ocaml/RULES.md:18-37`): the eligible profile
`(239, 1)` lies below it. The bound holds on the reachable TTD domain
(`revival_ge_on_domain`). -/
theorem min_bound_literal_false :
    ¬ ∀ s t : ℕ, Survives s t → 0.95 / 240 ≤ revival s t := by
  intro h
  have := h 239 1 (by decide)
  linarith [revival_239_1_facts.2.1]

/-- **OCaml boundary tests** (claim PAPERS-OCAML-REV-1 (d),
`src/dth_ocaml/test/exact_test.ml:41-47`): `P(0,0) = 0.95`, `P(240,0) = 0`,
`P(1,239) > 0` (there `t + q = 300`), `P(1,240) = 0`, and `P(0,60) < P(0,0)`. -/
theorem ocaml_boundary :
    revival 0 0 = 0.95 ∧ revival 240 0 = 0 ∧ 0 < revival 1 239 ∧ revival 1 240 = 0 ∧
      revival 0 60 < revival 0 0 :=
  ⟨(revival_range 0 0).2.2.2.2, revival_of_not_survives (by decide),
    (revival_pos_iff 1 239).2 (by decide), revival_of_not_survives (by decide),
    revival_strictAnti_t (by norm_num) (by decide)⟩

/-- **Bucket-form lower bound in seconds** (claim ABSTRACT-REV-2 at `B = 1`):
`0.95 · 0.75^5 / 240 ≤ P` on the eligible region. -/
theorem revival_ge_bucket_seconds {s t : ℕ} (h : Survives s t) :
    0.95 * 0.75 ^ 5 / 240 ≤ revival s t :=
  le_trans (by norm_num) (revival_ge_uniform h).1

/-! ## The table of `docs/REVIVAL_MODEL.md` -/

/-- Round-half-up to four decimals: `x` prints as `e`. -/
def RoundsTo4 (x e : ℝ) : Prop := e - 1 / 20000 ≤ x ∧ x < e + 1 / 20000

/-- **Revival table** (claim CANONICAL-REV-8, `docs/REVIVAL_MODEL.md:112-133`):
every printed entry of the table rounds (half up, four decimals) to the true
value, and every `--` cell is an ineligible profile with `P = 0`. -/
theorem revival_table :
    RoundsTo4 (revival 0 (60 * 0)) 0.9500 ∧
    RoundsTo4 (revival 0 (60 * 1)) 0.7125 ∧
    RoundsTo4 (revival 0 (60 * 2)) 0.5344 ∧
    RoundsTo4 (revival 0 (60 * 3)) 0.4008 ∧
    RoundsTo4 (revival 0 (60 * 4)) 0.3006 ∧
    RoundsTo4 (revival 20 (60 * 0)) 0.8708 ∧
    RoundsTo4 (revival 20 (60 * 1)) 0.6531 ∧
    RoundsTo4 (revival 20 (60 * 2)) 0.4898 ∧
    RoundsTo4 (revival 20 (60 * 3)) 0.3674 ∧
    revival 20 (60 * 4) = 0 ∧
    RoundsTo4 (revival 40 (60 * 0)) 0.7917 ∧
    RoundsTo4 (revival 40 (60 * 1)) 0.5938 ∧
    RoundsTo4 (revival 40 (60 * 2)) 0.4453 ∧
    RoundsTo4 (revival 40 (60 * 3)) 0.3340 ∧
    revival 40 (60 * 4) = 0 ∧
    RoundsTo4 (revival 60 (60 * 0)) 0.7125 ∧
    RoundsTo4 (revival 60 (60 * 1)) 0.5344 ∧
    RoundsTo4 (revival 60 (60 * 2)) 0.4008 ∧
    RoundsTo4 (revival 60 (60 * 3)) 0.3006 ∧
    revival 60 (60 * 4) = 0 ∧
    RoundsTo4 (revival 80 (60 * 0)) 0.6333 ∧
    RoundsTo4 (revival 80 (60 * 1)) 0.4750 ∧
    RoundsTo4 (revival 80 (60 * 2)) 0.3563 ∧
    revival 80 (60 * 3) = 0 ∧
    revival 80 (60 * 4) = 0 ∧
    RoundsTo4 (revival 100 (60 * 0)) 0.5542 ∧
    RoundsTo4 (revival 100 (60 * 1)) 0.4156 ∧
    RoundsTo4 (revival 100 (60 * 2)) 0.3117 ∧
    revival 100 (60 * 3) = 0 ∧
    revival 100 (60 * 4) = 0 ∧
    RoundsTo4 (revival 120 (60 * 0)) 0.4750 ∧
    RoundsTo4 (revival 120 (60 * 1)) 0.3563 ∧
    RoundsTo4 (revival 120 (60 * 2)) 0.2672 ∧
    revival 120 (60 * 3) = 0 ∧
    revival 120 (60 * 4) = 0 ∧
    RoundsTo4 (revival 140 (60 * 0)) 0.3958 ∧
    RoundsTo4 (revival 140 (60 * 1)) 0.2969 ∧
    revival 140 (60 * 2) = 0 ∧
    revival 140 (60 * 3) = 0 ∧
    revival 140 (60 * 4) = 0 ∧
    RoundsTo4 (revival 160 (60 * 0)) 0.3167 ∧
    RoundsTo4 (revival 160 (60 * 1)) 0.2375 ∧
    revival 160 (60 * 2) = 0 ∧
    revival 160 (60 * 3) = 0 ∧
    revival 160 (60 * 4) = 0 ∧
    RoundsTo4 (revival 180 (60 * 0)) 0.2375 ∧
    RoundsTo4 (revival 180 (60 * 1)) 0.1781 ∧
    revival 180 (60 * 2) = 0 ∧
    revival 180 (60 * 3) = 0 ∧
    revival 180 (60 * 4) = 0 ∧
    RoundsTo4 (revival 200 (60 * 0)) 0.1583 ∧
    revival 200 (60 * 1) = 0 ∧
    revival 200 (60 * 2) = 0 ∧
    revival 200 (60 * 3) = 0 ∧
    revival 200 (60 * 4) = 0 ∧
    RoundsTo4 (revival 220 (60 * 0)) 0.0792 ∧
    revival 220 (60 * 1) = 0 ∧
    revival 220 (60 * 2) = 0 ∧
    revival 220 (60 * 3) = 0 ∧
    revival 220 (60 * 4) = 0 ∧
    RoundsTo4 (revival 239 (60 * 0)) 0.0040 ∧
    revival 239 (60 * 1) = 0 ∧
    revival 239 (60 * 2) = 0 ∧
    revival 239 (60 * 3) = 0 ∧
    revival 239 (60 * 4) = 0 := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  all_goals (rw [revival_grid]; norm_num [Survives, RoundsTo4])


/-- **Table endpoints** (claim CANONICAL-REV-8, `docs/REVIVAL_MODEL.md:132-133`):
`P(0,0) = 0.95`, `P(239,0) = 0.95/240`, `P(0,240) = 0.95 · 0.75^4 = 0.3005859375`. -/
theorem table_endpoints :
    revival 0 0 = 0.95 ∧ revival 239 0 = 0.95 / 240 ∧ revival 0 240 = 0.3005859375 := by
  refine ⟨(revival_range 0 0).2.2.2.2, revival_239_0, ?_⟩
  rw [revival_0_240]; norm_num

/-- `--` appears exactly where `s + t > 240`: for `s ≤ 239` the cell is zero iff
`s + t > 240` (claim CANONICAL-REV-8). -/
theorem table_dash_iff {s t : ℕ} (hs : s ≤ 239) : revival s t = 0 ↔ 240 < s + t := by
  rw [← not_iff_not, ← ne_eq, revival_ne_zero_iff, survives_iff]; omega

/-! ## The frontier `s + t = 240` -/

/-- On `s + t = 240` the total load is `t + q = 300` (claim CANONICAL-REV-9). -/
theorem frontier_total {s t : ℕ} (h : s + t = 240) : t + (s + 60) = 300 := by omega

/-- **Frontier table** (claim CANONICAL-REV-9, `docs/REVIVAL_MODEL.md:135-151`):
`P(180,60) = 0.178125`, `P(120,120) = 0.2671875`, `P(60,180) = P(0,240) =
0.95 · 0.75^4`, and `P(0,240)/P(239,1) ∈ (76.3, 76.31)`, the documented `76x`. -/
theorem frontier_values :
    revival 180 60 = 0.178125 ∧ revival 120 120 = 0.2671875 ∧
      revival 60 180 = 0.3005859375 ∧ revival 0 240 = 0.3005859375 ∧
      76.3 < revival 0 240 / revival 239 1 ∧ revival 0 240 / revival 239 1 < 76.31 := by
  have g : ∀ s k : ℕ, Survives s (60 * k) →
      revival s (60 * k) = 0.95 * (1 - (s : ℝ) / 240) * (0.75 : ℝ) ^ k := by
    intro s k h; rw [revival_grid]; simp only [h, ↓reduceIte]
  have e1 : revival 180 60 = 0.178125 := by
    rw [show (60 : ℕ) = 60 * 1 from rfl, g _ _ (by decide)]; norm_num
  have e2 : revival 120 120 = 0.2671875 := by
    rw [show (120 : ℕ) = 60 * 2 from rfl, g _ _ (by decide)]; norm_num
  have e3 : revival 60 180 = 0.3005859375 := by
    rw [show (180 : ℕ) = 60 * 3 from rfl, g _ _ (by decide)]; norm_num
  have e4 : revival 0 240 = 0.3005859375 := table_endpoints.2.2
  obtain ⟨-, -, lo, hi⟩ := revival_239_1_facts
  have hp : 0 < revival 239 1 := by linarith
  refine ⟨e1, e2, e3, e4, ?_, ?_⟩
  · rw [lt_div_iff₀ hp, e4]; nlinarith
  · rw [div_lt_iff₀ hp, e4]; nlinarith

/-- `y · 0.75^y ≥ 0.75` for `y ∈ [1, 4]`. -/
theorem mul_rpow75_ge {y : ℝ} (h1 : 1 ≤ y) (h4 : y ≤ 4) : 0.75 ≤ y * (0.75 : ℝ) ^ y := by
  set z := y - 1 with hz
  have hy : y = 1 + z := by ring
  have hz0 : 0 ≤ z := by linarith
  have hsplit : (0.75 : ℝ) ^ y = 0.75 * (0.75 : ℝ) ^ z := by
    rw [hy, Real.rpow_add (by norm_num), Real.rpow_one]
  rw [hsplit]
  have key : 1 ≤ (1 + z) * (0.75 : ℝ) ^ z := by
    have ha := rpow75_pos z
    rcases le_total z 1 with hz1 | hz1
    · have hb : ((1 : ℝ) + 1 / 3) ^ z ≤ 1 + z * (1 / 3) :=
        rpow_one_add_le_one_add_mul_self (by norm_num) hz0 hz1
      have hab : (0.75 : ℝ) ^ z * ((1 : ℝ) + 1 / 3) ^ z = 1 := by
        rw [← Real.mul_rpow (by norm_num) (by norm_num)]; norm_num
      nlinarith [mul_nonneg ha.le hz0]
    · rcases le_total z 2 with hz2 | hz2
      · have : (0.75 : ℝ) ^ 2 ≤ (0.75 : ℝ) ^ z := by
          rw [← rpow75_nat 2]; exact rpow75_anti (by push_cast; linarith)
        nlinarith
      · have : (0.75 : ℝ) ^ 3 ≤ (0.75 : ℝ) ^ z := by
          rw [← rpow75_nat 3]; exact rpow75_anti (by push_cast; linarith)
        nlinarith
  rw [hy]; nlinarith

/-- **Reachable frontier** (settles the doubt of claim CANONICAL-REV-9). For
frontier profiles with `t ∈ [60, 240]` (TTD `1` is unreachable) the minimum is
`P(180, 60) = 0.178125`, and the frontier point `(32, 208)` exceeds `0.3037`,
above `P(0, 240)`. The reachable span is therefore at least
`0.3037/0.178125 > 1.70`, not the `1.69` of the doubt, and far below `76`. -/
theorem frontier_reachable (t : ℕ) (h60 : 60 ≤ t) (h240 : t ≤ 240) :
    Survives (240 - t) t ∧ 0.178125 ≤ revival (240 - t) t := by
  have hS : Survives (240 - t) t := by unfold Survives; omega
  refine ⟨hS, ?_⟩
  rw [revival_of_survives hS]
  have hc : ((240 - t : ℕ) : ℝ) = 240 - t := by
    rw [Nat.cast_sub h240]; norm_num
  rw [hc]
  have ht60 : (60 : ℝ) ≤ t := by exact_mod_cast h60
  have ht240 : (t : ℝ) ≤ 240 := by exact_mod_cast h240
  have key := mul_rpow75_ge (y := (t : ℝ) / 60) (by rw [le_div_iff₀ (by norm_num)]; linarith)
    (by rw [div_le_iff₀ (by norm_num)]; linarith)
  have : 1 - (240 - (t : ℝ)) / 240 = (t : ℝ) / 60 / 4 := by ring
  rw [this]
  nlinarith

theorem frontier_point_32_208 :
    Survives 32 208 ∧ 0.3037 < revival 32 208 ∧ revival 0 240 < revival 32 208 ∧
      1.70 < revival 32 208 / revival 180 60 := by
  have lo : (0.368874 : ℝ) ≤ (0.75 : ℝ) ^ (((208 : ℕ) : ℝ) / 60) :=
    le_rpow_of_pow_le (k := 52) (n := 15) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
      (by norm_num)
  have e : revival 32 208 = 0.95 * (1 - (32 : ℝ) / 240) * (0.75 : ℝ) ^ (((208 : ℕ) : ℝ) / 60) := by
    rw [revival_of_survives (by decide)]; norm_num
  have p1 : 0.3037 < revival 32 208 := by rw [e]; nlinarith
  refine ⟨by decide, p1, by rw [table_endpoints.2.2]; linarith, ?_⟩
  rw [frontier_values.1, lt_div_iff₀ (by norm_num)]; linarith

/-- **The full frontier span exceeds `77x`** (claim CANONICAL-REV-9,
`docs/REVIVAL_MODEL.md:135-151`). The frontier maximum is interior, not at
`P(60,180) = P(0,240)`: the frontier point `(32, 208)` beats both, so the ratio
of frontier maximum to frontier minimum `P(239,1)` is above `77`, while the
listed rows give `76.3`. -/
theorem frontier_span : 77 < revival 32 208 / revival 239 1 := by
  obtain ⟨-, lo32, -, -⟩ := frontier_point_32_208
  obtain ⟨-, -, lo, hi⟩ := revival_239_1_facts
  have hp : 0 < revival 239 1 := by linarith
  rw [lt_div_iff₀ hp]; nlinarith

/-! ## Ledger validation (`docs/REVIVAL_MODEL.md:153-174`) -/

/-- **Ledger values** (claim CANONICAL-REV-10). Each recorded revival is
eligible; `P(0,0) = 0.95`, `P(24,0) = 0.855`, `P(33,60) = 0.61453125`,
`P(94,84)` rounds to `0.3863`, `P(0,238)` rounds to `0.3035`; the sequence
strictly decreases; and the joint likelihood lies in `(0.05852, 0.05853)`,
which prints as `0.0585`. -/
theorem ledger :
    (Survives 0 0 ∧ Survives 24 0 ∧ Survives 33 60 ∧ Survives 94 84 ∧ Survives 0 238) ∧
      revival 0 0 = 0.95 ∧ revival 24 0 = 0.855 ∧ revival 33 60 = 0.61453125 ∧
      RoundsTo4 (revival 94 84) 0.3863 ∧ RoundsTo4 (revival 0 238) 0.3035 ∧
      (revival 24 0 < revival 0 0 ∧ revival 33 60 < revival 24 0 ∧
        revival 94 84 < revival 33 60 ∧ revival 0 238 < revival 94 84) ∧
      0.05852 < revival 0 0 * revival 24 0 * revival 33 60 * revival 94 84 * revival 0 238 ∧
      revival 0 0 * revival 24 0 * revival 33 60 * revival 94 84 * revival 0 238 < 0.05853 := by
  have e0 : revival 0 0 = 0.95 := (revival_range 0 0).2.2.2.2
  have e1 : revival 24 0 = 0.855 := by rw [revival_of_survives (by decide)]; norm_num
  have e2 : revival 33 60 = 0.61453125 := by
    rw [show (60 : ℕ) = 60 * 1 from rfl, revival_grid]
    simp only [show Survives 33 (60 * 1) by decide, ↓reduceIte]; norm_num
  have a_lo : (0.668475 : ℝ) ≤ (0.75 : ℝ) ^ (((84 : ℕ) : ℝ) / 60) :=
    le_rpow_of_pow_le (k := 7) (n := 5) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
      (by norm_num)
  have a_hi : (0.75 : ℝ) ^ (((84 : ℕ) : ℝ) / 60) ≤ 0.668476 :=
    rpow_le_of_le_pow (k := 7) (n := 5) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
      (by norm_num)
  have b_lo : (0.319454 : ℝ) ≤ (0.75 : ℝ) ^ (((238 : ℕ) : ℝ) / 60) :=
    le_rpow_of_pow_le (k := 119) (n := 30) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
      (by norm_num)
  have b_hi : (0.75 : ℝ) ^ (((238 : ℕ) : ℝ) / 60) ≤ 0.319455 :=
    rpow_le_of_le_pow (k := 119) (n := 30) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
      (by norm_num)
  have e3 : revival 94 84 = 0.95 * (1 - (94 : ℝ) / 240) * (0.75 : ℝ) ^ (((84 : ℕ) : ℝ) / 60) := by
    rw [revival_of_survives (by decide)]; norm_num
  have e4 : revival 0 238 = 0.95 * (0.75 : ℝ) ^ (((238 : ℕ) : ℝ) / 60) := by
    rw [revival_of_survives (by decide)]; norm_num
  set a := (0.75 : ℝ) ^ (((84 : ℕ) : ℝ) / 60)
  set b := (0.75 : ℝ) ^ (((238 : ℕ) : ℝ) / 60)
  have ha : 0.38632 < revival 94 84 ∧ revival 94 84 < 0.38634 := by
    rw [e3]; constructor <;> nlinarith
  have hb : 0.30348 < revival 0 238 ∧ revival 0 238 < 0.30349 := by
    rw [e4]; constructor <;> nlinarith
  refine ⟨by decide, e0, e1, e2, ⟨by linarith, by linarith⟩, ⟨by linarith, by linarith⟩,
    ⟨by rw [e0, e1]; norm_num, by rw [e1, e2]; norm_num, by rw [e2]; linarith, by linarith⟩, ?_, ?_⟩
  · rw [e0, e1, e2]
    have := mul_lt_mul'' ha.1 hb.1 (by norm_num) (by norm_num)
    nlinarith
  · rw [e0, e1, e2]
    have := mul_lt_mul'' ha.2 hb.2 (by linarith) (by linarith)
    nlinarith

/-! ## Half-life (`docs/REVIVAL_MODEL.md:39-40`) -/

/-- The half-life of the TTD factor, `h = 60 ln 2 / ln (4/3)`. -/
noncomputable def halfLife : ℝ := 60 * Real.log 2 / Real.log (4 / 3)

theorem rpow_halfLife : (0.75 : ℝ) ^ (halfLife / 60) = 1 / 2 := by
  have hl : 0 < Real.log (4 / 3) := Real.log_pos (by norm_num)
  have h75 : Real.log 0.75 = -Real.log (4 / 3) := by
    rw [show (0.75 : ℝ) = (4 / 3)⁻¹ by norm_num, Real.log_inv]
  rw [Real.rpow_def_of_pos (by norm_num), h75, halfLife]
  have : -Real.log (4 / 3) * (60 * Real.log 2 / Real.log (4 / 3) / 60) = -Real.log 2 := by
    field_simp
  rw [this, Real.exp_neg, Real.exp_log (by norm_num)]; norm_num

/-- `h` is the unique solution of `0.75^(h/60) = 1/2`. -/
theorem rpow_eq_half_iff (h : ℝ) : (0.75 : ℝ) ^ (h / 60) = 1 / 2 ↔ h = halfLife := by
  constructor
  · intro H
    rcases lt_trichotomy (h / 60) (halfLife / 60) with hl | he | hg
    · have := rpow75_strictAnti hl; rw [H, rpow_halfLife] at this; exact absurd this (lt_irrefl _)
    · linarith
    · have := rpow75_strictAnti hg; rw [H, rpow_halfLife] at this; exact absurd this (lt_irrefl _)
  · rintro rfl; exact rpow_halfLife

/-- **The half-life lies in `(144.55, 144.6)`** (claims CANONICAL-REV-5,
DTH-RULE-3 (c), ABSTRACT-REV-4 (b)). -/
theorem halfLife_bounds : 144.55 < halfLife ∧ halfLife < 144.6 := by
  constructor
  · by_contra hc
    push Not at hc
    have h1 : (1 / 2 : ℝ) < (0.75 : ℝ) ^ ((144.55 : ℝ) / 60) :=
      lt_rpow_of_pow_lt (k := 2891) (n := 1200) (by norm_num) (by norm_num) (by norm_num)
        (by norm_num) (by norm_num)
    have h2 := rpow75_anti (show (144.55 : ℝ) / 60 ≥ halfLife / 60 by linarith)
    rw [rpow_halfLife] at h2; linarith
  · by_contra hc
    push Not at hc
    have h1 : (0.75 : ℝ) ^ ((144.6 : ℝ) / 60) < 1 / 2 :=
      rpow_lt_of_lt_pow (k := 241) (n := 100) (by norm_num) (by norm_num) (by norm_num)
        (by norm_num) (by norm_num)
    have h2 := rpow75_anti (show (144.6 : ℝ) / 60 ≤ halfLife / 60 by linarith)
    rw [rpow_halfLife] at h2; linarith

/-- **The former half-life `144.3 s` was wrong** (claims CANONICAL-REV-5,
DTH-RULE-3 (c), ABSTRACT-REV-4 (b); `docs/REVIVAL_MODEL.md:40`,
`src/dth/docs/GAME_AND_SOLVER.md:56-57`, `src/abstract/docs/MODEL.md:13-17,62`):
`0.75^(144.3/60) > 0.5006`, so `144.3` is not the half-life; the bucket
half-lives are `h/10 ∈ (14.455, 14.46)` and `h/5 ∈ (28.91, 28.92)`, not the
former `14.43` and `28.86`. The value `h ≈ 144.565` rounds to `144.6`, which
the docs now state. -/
theorem halfLife_doc_wrong :
    0.5006 < (0.75 : ℝ) ^ ((144.3 : ℝ) / 60) ∧ halfLife ≠ 144.3 ∧
      (14.455 < halfLife / 10 ∧ halfLife / 10 < 14.46) ∧ halfLife / 10 ≠ 14.43 ∧
      (28.91 < halfLife / 5 ∧ halfLife / 5 < 28.92) ∧ halfLife / 5 ≠ 28.86 := by
  obtain ⟨lo, hi⟩ := halfLife_bounds
  refine ⟨lt_rpow_of_pow_lt (k := 481) (n := 200) (by norm_num) (by norm_num) (by norm_num)
      (by norm_num) (by norm_num), by intro h; linarith, ⟨by linarith, by linarith⟩,
    by intro h; linarith, ⟨by linarith, by linarith⟩, by intro h; linarith⟩

/-- In bucket units with `F = 60/B` the half-life is `h/B`: for `B = 10`
(`F = 6`) the solution of `0.75^(τ/6) = 1/2` is `τ = h/10`, and for `B = 5`
(`F = 12`) it is `τ = h/5` (claim ABSTRACT-REV-4 (b)). -/
theorem bucket_halfLife (τ : ℝ) :
    ((0.75 : ℝ) ^ (τ / 6) = 1 / 2 ↔ τ = halfLife / 10) ∧
      ((0.75 : ℝ) ^ (τ / 12) = 1 / 2 ↔ τ = halfLife / 5) := by
  constructor
  · rw [show τ / 6 = (10 * τ) / 60 by ring, rpow_eq_half_iff]; constructor <;> intro h <;> linarith
  · rw [show τ / 12 = (5 * τ) / 60 by ring, rpow_eq_half_iff]; constructor <;> intro h <;> linarith

/-! ## The `0.748` rounding (`docs/REVIVAL_MODEL.md:88-91`) -/

/-- `0.85 · 0.88 = 0.748` (claims CANONICAL-REV-6, ABSTRACT-REV-4 (c)). -/
theorem decay_product : (0.85 : ℝ) * 0.88 = 0.748 := by norm_num

/-- **Rounding error of `0.748 → 0.75`** (claim CANONICAL-REV-6). For
`t ∈ [0, 240]` the ratio `0.75^(t/60) / 0.748^(t/60) = (0.75/0.748)^(t/60)`
is increasing in `t`, so the error relative to the `0.748` surface is at most
`(0.75/0.748)^4 - 1 ∈ (0.010738, 0.010739)`, attained at `t = 240`. That is
`1.074%`, which prints as `1.07%` but exceeds `1.07%` literally. The error
relative to the `0.75` surface, `1 - (0.748/0.75)^(t/60)`, is at most
`1 - (0.748/0.75)^4 ∈ (0.010624, 0.010625)`, below `1.07%` literally. -/
theorem rounding_error (t : ℝ) (h0 : 0 ≤ t) (h240 : t ≤ 240) :
    (0.75 : ℝ) ^ (t / 60) / (0.748 : ℝ) ^ (t / 60) = (0.75 / 0.748 : ℝ) ^ (t / 60) ∧
      0 ≤ (0.75 / 0.748 : ℝ) ^ (t / 60) - 1 ∧
      (0.75 / 0.748 : ℝ) ^ (t / 60) - 1 ≤ (0.75 / 0.748 : ℝ) ^ 4 - 1 ∧
      0 ≤ 1 - (0.748 / 0.75 : ℝ) ^ (t / 60) ∧
      1 - (0.748 / 0.75 : ℝ) ^ (t / 60) ≤ 1 - (0.748 / 0.75 : ℝ) ^ 4 := by
  have ht : t / 60 ≤ (4 : ℕ) := by push_cast; linarith
  have ht0 : 0 ≤ t / 60 := by positivity
  refine ⟨(Real.div_rpow (by norm_num) (by norm_num) _).symm, ?_, ?_, ?_, ?_⟩
  · have := Real.one_le_rpow (x := (0.75 / 0.748 : ℝ)) (by norm_num) ht0; linarith
  · have := Real.rpow_le_rpow_of_exponent_le (x := (0.75 / 0.748 : ℝ)) (by norm_num) ht
    rw [Real.rpow_natCast] at this; linarith
  · have := Real.rpow_le_one (x := (0.748 / 0.75 : ℝ)) (by norm_num) (by norm_num) ht0; linarith
  · have := Real.rpow_le_rpow_of_exponent_ge (x := (0.748 / 0.75 : ℝ)) (by norm_num) (by norm_num) ht
    rw [Real.rpow_natCast] at this; linarith

theorem rounding_error_values :
    (0.010738 : ℝ) < (0.75 / 0.748) ^ 4 - 1 ∧ (0.75 / 0.748 : ℝ) ^ 4 - 1 < 0.010739 ∧
      (0.010624 : ℝ) < 1 - (0.748 / 0.75) ^ 4 ∧ 1 - (0.748 / 0.75 : ℝ) ^ 4 < 0.010625 ∧
      (0.0107 : ℝ) < (0.75 / 0.748) ^ 4 - 1 := by
  norm_num

/-! ## The referee floor (`docs/REVIVAL_MODEL.md:93-99`) -/

/-- The death-minute count at which `max(0.40, 0.88^x)` starts to bind:
`x* = ln 0.4 / ln 0.88`. -/
noncomputable def floorThreshold : ℝ := Real.log 0.4 / Real.log 0.88

/-- The floor binds exactly beyond `x*`: `0.88^y < 0.4 ↔ x* < y`. -/
theorem floor_binds_iff (y : ℝ) : (0.88 : ℝ) ^ y < 0.4 ↔ floorThreshold < y := by
  have hn : Real.log 0.88 < 0 := Real.log_neg (by norm_num) (by norm_num)
  rw [Real.rpow_def_of_pos (by norm_num),
    show (0.4 : ℝ) = Real.exp (Real.log 0.4) from (Real.exp_log (by norm_num)).symm,
    Real.exp_lt_exp, floorThreshold, div_lt_iff_of_neg hn]
  constructor <;> intro h <;> linarith [mul_comm (Real.log 0.88) y]

/-- **`x* ∈ [7.165, 7.168)`**, the documented `7.17` (claims CANONICAL-REV-4,
DTH-RULE-4, ABSTRACT-REV-4 (d)), and the margin above the eligibility cap of
`4` death-minutes is at least `3.165`, the documented `3.17`. -/
theorem floorThreshold_bounds :
    7.165 ≤ floorThreshold ∧ floorThreshold < 7.168 ∧ 3.165 ≤ floorThreshold - 4 := by
  have hlo : ¬ floorThreshold < 7.165 := by
    rw [← floor_binds_iff]; push Not
    exact (lt_rpow_of_pow_lt (k := 1433) (n := 200) (by norm_num) (by norm_num) (by norm_num)
      (by norm_num) (by norm_num)).le
  have hhi : floorThreshold < 7.168 := by
    rw [← floor_binds_iff]
    exact rpow_lt_of_lt_pow (k := 896) (n := 125) (by norm_num) (by norm_num) (by norm_num)
      (by norm_num) (by norm_num)
  push Not at hlo
  exact ⟨hlo, hhi, by linarith⟩

/-- **The referee floor never binds on the eligible region** (claims CANONICAL-REV-4,
DTH-RULE-4, ABSTRACT-REV-4 (d); `src/dth/docs/GAME_AND_SOLVER.md:57-58`). For
`t ∈ [0, 240]`, `0.88^(t/60) ≥ 0.88^4 = 0.59969536 > 0.4`, so
`max(0.40, 0.88^(t/60)) = 0.88^(t/60)`. The bound needs only `t ≤ 240`. -/
theorem floor_never_binds {t : ℝ} (h240 : t ≤ 240) :
    (0.88 : ℝ) ^ 4 ≤ (0.88 : ℝ) ^ (t / 60) ∧ (0.4 : ℝ) < 0.88 ^ 4 ∧
      max 0.4 ((0.88 : ℝ) ^ (t / 60)) = (0.88 : ℝ) ^ (t / 60) := by
  have h1 : (0.88 : ℝ) ^ 4 ≤ (0.88 : ℝ) ^ (t / 60) := by
    rw [← Real.rpow_natCast]
    exact Real.rpow_le_rpow_of_exponent_ge (by norm_num) (by norm_num) (by push_cast; linarith)
  have h2 : (0.4 : ℝ) < 0.88 ^ 4 := by norm_num
  exact ⟨h1, h2, max_eq_right (by linarith)⟩

theorem floor_never_binds_of_survives {s t : ℕ} (h : Survives s t) :
    max 0.4 ((0.88 : ℝ) ^ ((t : ℝ) / 60)) = (0.88 : ℝ) ^ ((t : ℝ) / 60) :=
  (floor_never_binds (by exact_mod_cast ttd_le_of_survives h)).2.2

/-! ## TTD and the death count (`docs/REVIVAL_MODEL.md:101-110`) -/

/-- The TTD accrued by a sequence of survived deaths with vial STs `sts`: each
death adds its dose `s + 60` (`src/dth/solver.py:271`, `failChild`). -/
def deathTTD (sts : List ℕ) : ℕ := (sts.map (· + 60)).sum

/-- A revived failed check adds the dose `s + 60` to the old Checker's TTD. -/
theorem failChild_ttd (x : State) : (failChild x).d.t = x.c.t + (x.c.s + 60) := by
  simp only [failChild]; omega

theorem sixty_mul_length_le (sts : List ℕ) : 60 * sts.length ≤ deathTTD sts := by
  induction sts with
  | nil => simp [deathTTD]
  | cons a l ih =>
    unfold deathTTD at *
    simp only [List.map_cons, List.sum_cons, List.length_cons]
    omega

/-- **`t/60` bounds the death count from above** (claim CANONICAL-REV-7):
`#deaths ≤ t/60`. -/
theorem deaths_le_ttd_div (sts : List ℕ) : (sts.length : ℝ) ≤ (deathTTD sts : ℝ) / 60 := by
  rw [le_div_iff₀ (by norm_num)]
  have := sixty_mul_length_le sts
  have : ((60 * sts.length : ℕ) : ℝ) ≤ (deathTTD sts : ℝ) := by exact_mod_cast this
  push_cast at this; linarith

/-- **The former direction was wrong** (claim CANONICAL-REV-7): `t/60` is not
a lower bound on deaths, as `docs/REVIVAL_MODEL.md` once said; it is an upper bound. One death at `s = 180` gives `t = 240` and
`t/60 = 4 > 1`. -/
theorem ttd_div_not_lower_bound :
    ¬ ∀ sts : List ℕ, (deathTTD sts : ℝ) / 60 ≤ sts.length := by
  intro h
  have := h [180]
  simp [deathTTD] at this
  norm_num at this

/-- Every reachable TTD is `0` or at least `60`: the domain `{0} ∪ [60, ∞)`. -/
theorem ttd_domain (sts : List ℕ) : deathTTD sts = 0 ∨ 60 ≤ deathTTD sts := by
  cases sts with
  | nil => left; rfl
  | cons a l => right; have := sixty_mul_length_le (a :: l); simp at this; omega

/-! ## The superseded cubic (`docs/REVIVAL_MODEL.md:72-74`) -/

/-- **The cubic `1 - (q/300)^3` is flat below `90`** (claim CANONICAL-REV-11):
for `0 ≤ q ≤ 90` it is at least `0.973`, hence above `0.97`. -/
theorem cubic_flat {q : ℝ} (h0 : 0 ≤ q) (h90 : q ≤ 90) :
    0.973 ≤ 1 - (q / 300) ^ 3 ∧ 0.97 < 1 - (q / 300) ^ 3 := by
  have h1 : q / 300 ≤ 90 / 300 := by linarith
  have h2 : (q / 300) ^ 3 ≤ (90 / 300) ^ 3 := pow_le_pow_left₀ (by positivity) h1 3
  norm_num at h2
  constructor <;> norm_num <;> linarith

/-! ## Bucket form (`src/abstract/rules.py:178-189`) -/

/-- `AbstractRuleset.revival_probability(prior_ttd = τ, dose_units = q)` with load
cap `C` and penalty `F`: `0` if `q ≥ C` or `τ + q > C`, else the clamp to `[0,1]`
of `0.95 (1 - (q - F)/(C - F)) 0.75^(τ/F)`. The Rust
`revival_probability` (`src/crates/abstract_solver/src/lib.rs:153-162`) has
the same structure. -/
noncomputable def bucketRevival (C F τ q : ℕ) : ℝ :=
  if C ≤ q ∨ C < τ + q then 0
  else max 0 (min 1 (0.95 * (1 - ((q : ℝ) - F) / ((C : ℝ) - F)) * (0.75 : ℝ) ^ ((τ : ℝ) / F)))

/-- Bucket eligibility: `q < C ∧ τ + q ≤ C`. -/
def BucketEligible (C τ q : ℕ) : Prop := q < C ∧ τ + q ≤ C

instance (C τ q : ℕ) : Decidable (BucketEligible C τ q) := inferInstanceAs (Decidable (_ ∧ _))

/-- On an eligible bucket profile with load `l` (dose `l + F`) the clamp does not
bind and `P = 0.95 (1 - l/(C - F)) 0.75^(τ/F)`, with `0 ≤ P ≤ 0.95`. -/
theorem bucketRevival_eq {C F τ l : ℕ} (h : BucketEligible C τ (l + F)) :
    bucketRevival C F τ (l + F) = 0.95 * (1 - (l : ℝ) / ((C : ℝ) - F)) * (0.75 : ℝ) ^ ((τ : ℝ) / F) ∧
      0 < 1 - (l : ℝ) / ((C : ℝ) - F) ∧
      0.95 * (1 - (l : ℝ) / ((C : ℝ) - F)) * (0.75 : ℝ) ^ ((τ : ℝ) / F) ≤ 0.95 := by
  obtain ⟨h1, h2⟩ := h
  have hlt : (l : ℝ) + F < C := by exact_mod_cast h1
  have hD : 0 < (C : ℝ) - F := by have : (0 : ℝ) ≤ l := Nat.cast_nonneg l; linarith
  have hd : 0 < 1 - (l : ℝ) / ((C : ℝ) - F) := by
    rw [sub_pos, div_lt_one hD]; linarith
  have hd1 : 1 - (l : ℝ) / ((C : ℝ) - F) ≤ 1 := by
    have : 0 ≤ (l : ℝ) / ((C : ℝ) - F) := div_nonneg (Nat.cast_nonneg l) hD.le
    linarith
  have hr0 := rpow75_pos ((τ : ℝ) / F)
  have hr1 := rpow75_le_one (e := (τ : ℝ) / F) (by positivity)
  have hle : 0.95 * (1 - (l : ℝ) / ((C : ℝ) - F)) * (0.75 : ℝ) ^ ((τ : ℝ) / F) ≤ 0.95 := by
    calc 0.95 * (1 - (l : ℝ) / ((C : ℝ) - F)) * (0.75 : ℝ) ^ ((τ : ℝ) / F) ≤ 0.95 * 1 * 1 := by
          gcongr
      _ = 0.95 := by norm_num
  refine ⟨?_, hd, hle⟩
  unfold bucketRevival
  have hn : ¬ (C ≤ l + F ∨ C < τ + (l + F)) := by omega
  have hq : ((l + F : ℕ) : ℝ) - F = l := by push_cast; ring
  simp only [hn, ↓reduceIte, hq]
  rw [min_eq_right (by linarith), max_eq_right (by positivity)]

theorem bucketRevival_of_not {C F τ q : ℕ} (h : ¬ BucketEligible C τ q) : bucketRevival C F τ q = 0 := by
  unfold bucketRevival BucketEligible at *
  have : C ≤ q ∨ C < τ + q := by omega
  simp only [this, ↓reduceIte]

/-- **Positivity iff eligibility, range, and bounds in bucket units** (claims
ABSTRACT-REV-1, ABSTRACT-REV-2; `src/abstract/docs/MODEL.md:42-63,169-180`).
For `0 < F < C` and load `l`: `P > 0 ↔ (l + F < C ∧ τ + l + F ≤ C)`; `P < 1`
always; and on eligible profiles `0.95 · 0.75^(C/F) / (C - F) ≤ P ≤ 0.95`. So
the clamp never binds and every failed check has a death branch. -/
theorem bucketRevival_bounds {C F τ l : ℕ} (hF : 0 < F) :
    (0 < bucketRevival C F τ (l + F) ↔ BucketEligible C τ (l + F)) ∧
      bucketRevival C F τ (l + F) < 1 ∧
      (BucketEligible C τ (l + F) →
        0.95 * (0.75 : ℝ) ^ ((C : ℝ) / F) / ((C : ℝ) - F) ≤ bucketRevival C F τ (l + F) ∧
          bucketRevival C F τ (l + F) ≤ 0.95) := by
  by_cases h : BucketEligible C τ (l + F)
  · obtain ⟨e, hd, hle⟩ := bucketRevival_eq h
    have hr := rpow75_pos ((τ : ℝ) / F)
    have hpos : 0 < bucketRevival C F τ (l + F) := by rw [e]; positivity
    refine ⟨⟨fun _ => h, fun _ => hpos⟩, by rw [e]; linarith, fun _ => ⟨?_, by rw [e]; exact hle⟩⟩
    obtain ⟨h1, h2⟩ := h
    have hlt : (l : ℝ) + 1 + F ≤ C := by exact_mod_cast (by omega : l + 1 + F ≤ C)
    have hD : 0 < (C : ℝ) - F := by have : (0 : ℝ) ≤ l := Nat.cast_nonneg l; linarith
    have hdl : 1 / ((C : ℝ) - F) ≤ 1 - (l : ℝ) / ((C : ℝ) - F) := by
      rw [div_le_iff₀ hD]
      have : (1 - (l : ℝ) / ((C : ℝ) - F)) * ((C : ℝ) - F) = (C : ℝ) - F - l := by
        field_simp
      rw [this]; linarith
    have hτ : (τ : ℝ) / F ≤ (C : ℝ) / F := by
      have : (τ : ℝ) ≤ C := by exact_mod_cast (by omega : τ ≤ C)
      exact div_le_div_of_nonneg_right this (Nat.cast_nonneg F)
    have hrr := rpow75_anti hτ
    have hC0 := (rpow75_pos ((C : ℝ) / F)).le
    rw [e]
    calc 0.95 * (0.75 : ℝ) ^ ((C : ℝ) / F) / ((C : ℝ) - F)
        = 0.95 * (1 / ((C : ℝ) - F)) * (0.75 : ℝ) ^ ((C : ℝ) / F) := by ring
      _ ≤ 0.95 * (1 - (l : ℝ) / ((C : ℝ) - F)) * (0.75 : ℝ) ^ ((τ : ℝ) / F) := by gcongr
  · rw [bucketRevival_of_not h]
    exact ⟨⟨fun h' => absurd h' (lt_irrefl 0), fun h' => absurd h' h⟩, by norm_num,
      fun h' => absurd h' h⟩

/-- **Bucket eligibility equals physical eligibility** (claims CANONICAL-RULE-3,
CANONICAL-REV-3, ABSTRACT-REV-1): with `B C = 300`, `B F = 60`, load `l`, and
TTD `τ`, `(l + F < C ∧ τ + l + F ≤ C) ↔ Survives (B l) (B τ)`. -/
theorem bucketEligible_iff {B C F : ℕ} (hB : 0 < B) (hC : B * C = 300) (hF : B * F = 60)
    (l τ : ℕ) : BucketEligible C τ (l + F) ↔ Survives (B * l) (B * τ) := by
  unfold BucketEligible Survives
  have e1 : l + F < C ↔ B * l + 60 < 300 := by
    rw [← Nat.mul_lt_mul_left hB, mul_add, hF, hC]
  have e2 : τ + (l + F) ≤ C ↔ B * τ + (B * l + 60) ≤ 300 := by
    rw [← Nat.mul_le_mul_left_iff hB, mul_add, mul_add, hF, hC]
  rw [e1, e2]

/-- **Exact bucket invariance over the reals** (claims ABSTRACT-REV-1,
ABSTRACT-REV-3, CANONICAL-REV-3; `docs/REVIVAL_MODEL.md:46-57`,
`src/abstract/docs/MODEL.md:73-84`). With `B C = 300` and `B F = 60`,
`bucketRevival C F τ (l + F) = revival (B l) (B τ)` for every load `l` and
TTD `τ`, eligible or not. -/
theorem bucket_eq_seconds {B C F : ℕ} (hB : 0 < B) (hC : B * C = 300) (hF : B * F = 60)
    (l τ : ℕ) : bucketRevival C F τ (l + F) = revival (B * l) (B * τ) := by
  by_cases h : BucketEligible C τ (l + F)
  · rw [(bucketRevival_eq h).1, revival_of_survives ((bucketEligible_iff hB hC hF l τ).1 h)]
    have hB' : (B : ℝ) ≠ 0 := by positivity
    have hCr : (B : ℝ) * C = 300 := by exact_mod_cast hC
    have hFr : (B : ℝ) * F = 60 := by exact_mod_cast hF
    have hCF : (C : ℝ) - F = 240 / B := by field_simp; linarith
    have hF' : (F : ℝ) = 60 / B := by field_simp; linarith
    have a1 : (l : ℝ) / ((C : ℝ) - F) = ((B * l : ℕ) : ℝ) / 240 := by
      rw [hCF]; push_cast; field_simp
    have a2 : (τ : ℝ) / F = ((B * τ : ℕ) : ℝ) / 60 := by
      rw [hF']; push_cast; field_simp
    rw [a1, a2]
  · rw [bucketRevival_of_not h,
      revival_of_not_survives (fun h' => h ((bucketEligible_iff hB hC hF l τ).2 h'))]

/-- The production rulesets: `bucket6` (`B = 10`, `C = 30`, `F = 6`, `C - F = 24`)
and `bucket12` (`B = 5`, `C = 60`, `F = 12`, `C - F = 48`) agree with the seconds
surface at `B = 1` on the shared grid (claims ABSTRACT-REV-3, CANONICAL-REV-3). -/
theorem bucket_invariance (l τ : ℕ) :
    bucketRevival 30 6 τ (l + 6) = revival (10 * l) (10 * τ) ∧
      bucketRevival 60 12 (2 * τ) (2 * l + 12) = revival (10 * l) (10 * τ) ∧
      bucketRevival 300 60 (10 * τ) (10 * l + 60) = revival (10 * l) (10 * τ) ∧
      (30 - 6 = 24 ∧ 60 - 12 = 48) := by
  refine ⟨bucket_eq_seconds (B := 10) (by norm_num) rfl rfl l τ, ?_, ?_, ⟨rfl, rfl⟩⟩
  · rw [bucket_eq_seconds (B := 5) (by norm_num) rfl rfl]; ring_nf
  · rw [bucket_eq_seconds (B := 1) (by norm_num) rfl rfl]; ring_nf

/-- At zero TTD and dose `F` (load `0`) the bucket surface gives `0.95`
(claim ABSTRACT-REV-1). -/
theorem bucketRevival_zero {C F : ℕ} (hFC : F < C) : bucketRevival C F 0 (0 + F) = 0.95 := by
  rw [(bucketRevival_eq (by unfold BucketEligible; omega)).1]; simp

/-- In `bucket6` the largest survivable load is `23`, with `P = 0.95/24`
(claim ABSTRACT-REV-4 (a) at `B = 10`), and eligibility caps the TTD at
`C - F` (claim ABSTRACT-REV-4 (d)). -/
theorem bucket6_extremes :
    (∀ l τ : ℕ, BucketEligible 30 τ (l + 6) → l ≤ 23 ∧ τ ≤ 30 - 6) ∧
      bucketRevival 30 6 0 (23 + 6) = 0.95 / 24 := by
  refine ⟨fun l τ h => by unfold BucketEligible at h; omega, ?_⟩
  rw [(bucketRevival_eq (by unfold BucketEligible; omega)).1]; norm_num

/-- `Referee.compute_survival_probability(player with ttd = t, death_duration = q)`
(`src/stl/engine/game.py:171-200`) with `CYLINDER_MAX = TOTAL_TTD_MAX = 300`,
`FAILED_CHECK_PENALTY = 60`. -/
noncomputable def refereeProb (t q : ℕ) : ℝ :=
  if 300 ≤ q ∨ 300 < t + q then 0
  else max 0 (min 1 (0.95 * (1 - max 0 ((q : ℝ) - 60) / (300 - 60)) * (0.75 : ℝ) ^ ((t : ℝ) / 60)))

/-- The STL referee computes the frozen surface (claim STL-RULE-5): for every
`s`, `t`, `refereeProb t (s + 60) = revival s t`, so it is positive exactly on
eligible profiles. -/
theorem refereeProb_eq (s t : ℕ) : refereeProb t (s + 60) = revival s t := by
  have hb : refereeProb t (s + 60) = bucketRevival 300 60 t (s + 60) := by
    unfold refereeProb bucketRevival
    have : max 0 (((s + 60 : ℕ) : ℝ) - 60) = ((s + 60 : ℕ) : ℝ) - 60 := by
      apply max_eq_right; push_cast; linarith [(Nat.cast_nonneg s : (0 : ℝ) ≤ s)]
    rw [this]; norm_num
  rw [hb, bucket_eq_seconds (B := 1) (by norm_num) rfl rfl s t]; simp

/-! ## Failed-check branches and the game graph -/

/-- The branches a failed check emits (`src/abstract/rules.py:227-258`,
`src/dth/solver.py:262-273`): `true` (revived, live child) when `p > 0`, and
`false` (death) when `p < 1`. -/
noncomputable def failBranches (p : ℝ) : Finset Bool :=
  (if 0 < p then {true} else ∅) ∪ (if p < 1 then {false} else ∅)

/-- **The branch set depends only on eligibility** (claims ABSTRACT-REV-2 (ii),
(iii)): with `F > 0`, the failed-check branches of the bucket surface are
`{revive, die}` on eligible profiles and `{die}` elsewhere. The predicate test
of `packed_live_successors` and the float test `p > 0` of
`expand_joint_action` therefore select the same live successors. -/
theorem failBranches_bucket {C F τ l : ℕ} (hF : 0 < F) :
    failBranches (bucketRevival C F τ (l + F)) =
      if BucketEligible C τ (l + F) then {true, false} else {false} := by
  obtain ⟨hpos, hlt, -⟩ := bucketRevival_bounds (C := C) (τ := τ) (l := l) hF
  unfold failBranches
  by_cases h : BucketEligible C τ (l + F)
  · simp only [hpos.2 h, hlt, ↓reduceIte, h]; rfl
  · have : ¬ 0 < bucketRevival C F τ (l + F) := fun h' => h (hpos.1 h')
    simp only [this, hlt, h, ↓reduceIte]; rfl

/-- The seconds-surface version of `failBranches_bucket`. -/
theorem failBranches_revival (s t : ℕ) :
    failBranches (revival s t) = if Survives s t then {true, false} else {false} := by
  unfold failBranches
  by_cases h : Survives s t
  · simp only [(revival_pos_iff s t).2 h, revival_lt_one, ↓reduceIte, h]; rfl
  · have : ¬ 0 < revival s t := fun h' => h ((revival_pos_iff s t).1 h')
    simp only [this, revival_lt_one, h, ↓reduceIte]; rfl

/-- One live transition of pure DTH for a revival surface `P`, as
`src/dth/solver.py:262-273` builds it: a successful check with lag
`ℓ ∈ 1..60` that leaves the Checker below `300` moves to `successChild`; a
failed check moves to `failChild` exactly when `P(s_c, t_c) ≠ 0` (the code
tests `p == 0.0`). -/
def Step (P : ℕ → ℕ → ℝ) (x y : State) : Prop :=
  (∃ ℓ, ∃ h : x.c.s + ℓ < 300, 1 ≤ ℓ ∧ ℓ ≤ 60 ∧ y = successChild x ℓ h) ∨
    (P x.c.s x.c.t ≠ 0 ∧ y = failChild x)

/-- **The game graph depends on the surface only through its zero set** (claims
CANONICAL-REV-12, ABSTRACT-REV-2 (iii); `docs/REVIVAL_MODEL.md:178-181`). Two
surfaces with the same zero set give the same transition relation, hence the
same reachable set from every start, hence the same state counts. -/
theorem step_eq_of_zero_set {P P' : ℕ → ℕ → ℝ} (h : ∀ s t, P s t = 0 ↔ P' s t = 0) :
    Step P = Step P' ∧
      ∀ x₀, {y | Relation.ReflTransGen (Step P) x₀ y} =
        {y | Relation.ReflTransGen (Step P') x₀ y} := by
  have e : Step P = Step P' := by
    funext x y; simp only [Step, ne_eq, h]
  exact ⟨e, fun x₀ => by rw [e]⟩

/-- The frozen surface's graph is the eligibility graph: `P ≠ 0 ↔ Survives`. Any
surface whose zero set is the ineligible set has the same graph (claims
CANONICAL-REV-12, STL-RULE-5). -/
theorem step_revival (x y : State) :
    Step revival x y ↔
      (∃ ℓ, ∃ h : x.c.s + ℓ < 300, 1 ≤ ℓ ∧ ℓ ≤ 60 ∧ y = successChild x ℓ h) ∨
        (Survives x.c.s x.c.t ∧ y = failChild x) := by
  simp only [Step, revival_ne_zero_iff]

theorem step_eq_revival {P : ℕ → ℕ → ℝ} (h : ∀ s t, P s t ≠ 0 ↔ Survives s t) :
    Step P = Step revival :=
  (step_eq_of_zero_set fun s t => by
    rw [← not_iff_not, ← ne_eq, ← ne_eq, h, revival_ne_zero_iff]).1

/-- Non-vacuity: a different surface with the same support. -/
example : Step (fun s t => if Survives s t then 1 / 2 else 0) = Step revival :=
  step_eq_revival fun s t => by by_cases h : Survives s t <;> simp [h]

/-! ## Floating-point models -/

/-- **Positivity and the open unit interval under rounding** (claims
CPP-CODE-RULE-2, ABSTRACT-REV-2, CANONICAL-REV-12). Abstract rounding model: the
computed value is `p̂ = p (1 + δ) + η` with `|δ| ≤ 2^-40` (room for several
binary64 roundings and a `pow` error of a few ulps) and `|η| ≤ 2^-1074`
(underflow). Then every eligible profile gives `0 < p̂ < 1`. This mirrors the
guard `0 < p < 1` of `src/dth_cpp/exact.cpp:61-63`. -/
theorem computed_mem_open_unit {s t : ℕ} (h : Survives s t) {pHat δ η : ℝ}
    (hround : pHat = revival s t * (1 + δ) + η) (hδ : |δ| ≤ 1 / 2 ^ 40)
    (hη : |η| ≤ 1 / 2 ^ 1074) : 0 < pHat ∧ pHat < 1 := by
  obtain ⟨hlo, hhi⟩ := revival_ge_uniform h
  obtain ⟨hδ1, hδ2⟩ := abs_le.1 hδ
  obtain ⟨hη1, hη2⟩ := abs_le.1 hη
  have hp0 : 0 ≤ revival s t := revival_nonneg s t
  subst hround
  constructor
  · have : (0.95 : ℝ) / 240 * 0.75 ^ 4 * (1 - 1 / 2 ^ 40) - 1 / 2 ^ 1074 > 0 := by norm_num
    nlinarith
  · have : (0.95 : ℝ) * (1 + 1 / 2 ^ 40) + 1 / 2 ^ 1074 < 1 := by norm_num
    nlinarith

/-- The code returns the literal `0.0` on ineligible profiles and a rounded
value elsewhere. Under the rounding model of `computed_mem_open_unit`, the
float test `p ≠ 0` agrees with `Survives`, so the float graph equals the
eligibility graph (claim CANONICAL-REV-12). -/
theorem computed_ne_zero_iff (fl : ℕ → ℕ → ℝ)
    (hfl : ∀ s t, Survives s t → ∃ δ η : ℝ, fl s t = revival s t * (1 + δ) + η ∧
      |δ| ≤ 1 / 2 ^ 40 ∧ |η| ≤ 1 / 2 ^ 1074) :
    Step (fun s t => if Survives s t then fl s t else 0) = Step revival := by
  apply step_eq_revival
  intro s t
  by_cases h : Survives s t
  · obtain ⟨δ, η, e, hδ, hη⟩ := hfl s t h
    simp only [h, ↓reduceIte, iff_true]
    exact ne_of_gt (computed_mem_open_unit h e hδ hη).1
  · simp [h]

/-- An abstract binary64 evaluation model: `fl` rounds a real result, `pow` is
the library power on binary64 operands, and `c95`, `c75` are the stored values
of the literals `0.95` and `0.75`. All are arbitrary deterministic functions.
Integer operands below `2^53` convert to binary64 exactly, so the model feeds
the exact integers to the first division. -/
structure FloatModel where
  fl : ℝ → ℝ
  pow : ℝ → ℝ → ℝ
  c95 : ℝ
  c75 : ℝ

/-- The evaluation order of `src/abstract/rules.py:183-189` and
`src/crates/abstract_solver/src/lib.rs:157-161`:
`clamp(fl(fl(0.95 · fl(1 - fl(l / span))) · pow(0.75, fl(τ / F))))`. -/
noncomputable def FloatModel.evalRevival (M : FloatModel) (l span τ F : ℕ) : ℝ :=
  max 0 (min 1 (M.fl (M.fl (M.c95 * M.fl (1 - M.fl ((l : ℝ) / span))) *
    M.pow M.c75 (M.fl ((τ : ℝ) / F)))))

/-- **Bit identity across bucket widths, inside the model** (claims
ABSTRACT-REV-3, CANONICAL-REV-3). The computed result depends on its integer
inputs only through the real quotients `l / span` and `τ / F`. The bucket
quotients coincide with the seconds quotients (`bucket_invariance`), so one
implementation with a fixed `fl` and `pow` returns identical binary64 results
at `B = 10`, `B = 5`, and `B = 1`. The model says nothing across two
implementations with different `pow` functions (Python `**` against Rust
`powf`), and a form that rounds `1/span` before multiplying by `l` would feed
different reals to `fl`. -/
theorem evalRevival_bucket_invariant (M : FloatModel) (l τ : ℕ) :
    M.evalRevival l 24 τ 6 = M.evalRevival (2 * l) 48 (2 * τ) 12 ∧
      M.evalRevival l 24 τ 6 = M.evalRevival (10 * l) 240 (10 * τ) 60 := by
  have q1 : ((l : ℝ) / (24 : ℕ)) = ((2 * l : ℕ) : ℝ) / (48 : ℕ) := by push_cast; ring
  have q2 : ((τ : ℝ) / (6 : ℕ)) = ((2 * τ : ℕ) : ℝ) / (12 : ℕ) := by push_cast; ring
  have q3 : ((l : ℝ) / (24 : ℕ)) = ((10 * l : ℕ) : ℝ) / (240 : ℕ) := by push_cast; ring
  have q4 : ((τ : ℝ) / (6 : ℕ)) = ((10 * τ : ℕ) : ℝ) / (60 : ℕ) := by push_cast; ring
  unfold FloatModel.evalRevival
  exact ⟨by rw [q1, q2], by rw [q3, q4]⟩

/-- The alternative form `(span - l)/span` also feeds equal reals to one
correctly rounded division across bucket widths, so it is invariant in the
same model (settles the doubt of claim CANONICAL-REV-3). -/
theorem alt_form_quotients (l : ℕ) (hl : l ≤ 24) :
    (((24 - l : ℕ) : ℝ) / (24 : ℕ)) = ((240 - 10 * l : ℕ) : ℝ) / (240 : ℕ) := by
  rw [Nat.cast_sub hl, Nat.cast_sub (by omega)]; push_cast; ring

/-! ## Values lie in `[-1, 1]` -/

/-- Every stage entry lies in `[-1, 1]` when every child value does. -/
theorem stage_entry_mem (W : State → ℝ) (x : State)
    (hW : ∀ y, phi x < phi y → -1 ≤ W y ∧ W y ≤ 1) (d c : Fin 60) :
    -1 ≤ stage W x d c ∧ stage W x d c ≤ 1 := by
  unfold stage succPay failPay
  split_ifs with h1 h2 h3
  · have := hW _ (phi_lt_successChild x (by omega) h2)
    constructor <;> linarith [this.1, this.2]
  · norm_num
  · have hp0 := revival_nonneg x.c.s x.c.t
    have hp1 := revival_le_one x.c.s x.c.t
    have hc := hW _ (phi_lt_failChild x h3)
    constructor <;> nlinarith [hc.1, hc.2]
  · norm_num

/-- **Values lie in `[-1, 1]`** (claim DTH-RULE-6, `src/dth/docs/BUILD.md:26-28`):
by induction on the potential, since every success entry is `1` or `-V(child)`,
every failed entry is `1` or `1 - p - p V(child)` with `p ∈ [0, 1]`, and a
matrix game value lies in the entry range. -/
theorem V_mem_Icc (x : State) : -1 ≤ V x ∧ V x ≤ 1 := by
  induction hm : 1201 - phi x using Nat.strong_induction_on generalizing x with
  | _ k ih =>
  have hchild : ∀ y, phi x < phi y → -1 ≤ V y ∧ V y ≤ 1 := by
    intro y hy
    have := phi_le y
    exact ih (1201 - phi y) (by omega) y rfl
  rw [V_bellman x]
  exact MatrixGame.value_mem_entry_range _ (fun d c => (stage_entry_mem V x hchild d c).1)
    (fun d c => (stage_entry_mem V x hchild d c).2)

end Revival

end Formal.DTH
