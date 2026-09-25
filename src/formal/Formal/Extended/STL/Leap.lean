import Mathlib
import Formal.DTH.Quotient
import Formal.Toeplitz.Basic
import Formal.MatrixGame.Transform

/-!
# The STL public leap game (rung L2)

This module formalizes the public leap game of `paper/stl.tex` §1–§3 and
`src/stl/docs/GAME_AND_SOLVER.md`, and mirrors the builder code in
`src/stl/solver/canonical.py`, `leap_profiles.py`, `leap_build.py`,
`leap_oracle.py`, `leap_audit.py`, and `leap_lp.py`.

Conventions.

* A clock `τ : ℕ` counts physical seconds after 8:00:00 and includes the
  inserted leap second, so `τ = 3600` is 8:59:60 and `τ = 3601` is 9:00:00.
* Actions are literal seconds from `1`. Matrix rows and columns are indexed
  from `0`: row `d` is the Dropper's second `d + 1`, column `c` the Checker's
  second `c + 1`. The Dropper is the maximizing row player.
* A state lists the Checker first (`Formal.DTH.State`). Half `1` has Hal as
  Dropper, half `2` has Baku as Dropper.
* Values are the current Dropper's win probability minus loss probability.
* A profile class is `Formal.DTH.QProfile`; the builder's integer class id is
  `classIndex`, with the terminal-win sentinel `17011`.

Main results.

* Rules: the leap window and turn duration, the legal action sets, and the
  equivalence of the builder's `is_window` with `leap_drop_available`.
* Clocks: the child clock matches the engine step, `snap` is the least
  wall-clock minute boundary after its argument, every child clock is larger
  than its parent and at most `540` seconds later while it stays at or below
  `3600`, and the clock label map is injective.
* Keys: `childKey` returns the key whose clock is the child clock, keys stay
  in their ranges, and the sweep order sees every child before its parent.
* Classes: the `17011` class ids, the block order, the success and failure
  class maps, the `183` reset Dropper profiles with the inverse map `idx0`,
  and the success-vector shift.
* Values: the leap-aware value `VL` by backward induction over the DTH
  potential, its Bellman equation, the quotient theorem at a fixed half and
  clock, the exact DTH boundary after `3600`, and the utility range.
* Engine: the Checker update of `resolve_half_round` gives the DTH success and
  failure children.
* Builder transforms: the OR-shift lemma behind `_success_bits`; forward
  reachability, where the bitmap that `forward_reachability` holds for each key
  is the encoded set of reachable quotient pairs (`pending_eq_states`); packed
  rank and select; revival-cache exactness; the retry rewrite of the resume
  check as a program transformation; and a replay of the canonical trace at the
  level of half-round outcomes.
* Bitmaps are modeled as sets of set `(row, column)` bits. Byte packing enters
  only in the rank and select section.

No claim here depends on floating point: every real-valued statement is over
`ℝ`, and the solver's floating-point gates enter only as real hypotheses.
-/

open Finset Matrix

set_option linter.unusedSectionVars false

namespace Formal.STL

open Formal.DTH Formal.MatrixGame

/-! ## Leap window and turn duration (STL-RULE-1) -/

/-- The leap window: a half-round starting at `τ` spans the inserted second
exactly when `3540 ≤ τ ≤ 3600` (`is_leap_window`, `canonical.py:48-51`; the
closed interval of `game.py:28-34`). -/
def IsLeapWindow (τ : ℕ) : Prop := 3540 ≤ τ ∧ τ ≤ 3600

instance (τ : ℕ) : Decidable (IsLeapWindow τ) := inferInstanceAs (Decidable (_ ∧ _))

/-- The turn duration `T(τ)` (`turn_duration`, `canonical.py:54-57`). -/
def turnDuration (τ : ℕ) : ℕ := if IsLeapWindow τ then 61 else 60

/-- STL-RULE-1 (`paper/stl.tex:99`, `canonical.py:48-57`): `T(τ) = 61` exactly
on the closed window `[3540, 3600]` and `60` elsewhere. `T` is a function of
the clock alone, so it does not depend on the half or on who drops. -/
theorem turnDuration_eq_61_iff (τ : ℕ) : turnDuration τ = 61 ↔ 3540 ≤ τ ∧ τ ≤ 3600 := by
  by_cases h : 3540 ≤ τ ∧ τ ≤ 3600 <;> simp [turnDuration, IsLeapWindow, h]

theorem turnDuration_eq_60_iff (τ : ℕ) : turnDuration τ = 60 ↔ τ < 3540 ∨ 3600 < τ := by
  unfold turnDuration
  by_cases h : IsLeapWindow τ
  · simp only [h, ↓reduceIte]; unfold IsLeapWindow at h; constructor <;> intro <;> omega
  · simp only [h, ↓reduceIte, true_iff]; unfold IsLeapWindow at h; omega

theorem turnDuration_le (τ : ℕ) : 60 ≤ turnDuration τ ∧ turnDuration τ ≤ 61 := by
  unfold turnDuration; split_ifs <;> omega

/-- The window ends: `3539` and `3601` are ordinary, `3540` and `3600` are leap
turns. The half-round `H1_59` at `τ = 3540` also lasts 61 s although Hal drops. -/
example : turnDuration 3539 = 60 ∧ turnDuration 3540 = 61 ∧ turnDuration 3600 = 61 ∧
    turnDuration 3601 = 60 := by decide

/-! ## Legal actions (STL-RULE-2) -/

/-- The two canonical identities. -/
inductive Identity
  | hal
  | baku
  deriving DecidableEq

/-- The two roles of a half-round. -/
inductive Role
  | dropper
  | checker
  deriving DecidableEq

/-- `legal_max_second` of `src/stl/engine/actions.py:35-45`. -/
def legalMaxSecond (a : Identity) (r : Role) (T : ℕ) : ℕ :=
  if T < 61 then min T 60 else if r = .dropper ∧ a = .baku then 61 else 60

/-- The legal literal seconds `{1, …, legal_max_second}` (`legal_seconds`). -/
def legalSeconds (a : Identity) (r : Role) (T : ℕ) : Finset ℕ :=
  Finset.Icc 1 (legalMaxSecond a r T)

/-- Action `0` is never legal. -/
theorem zero_not_mem_legalSeconds (a : Identity) (r : Role) (T : ℕ) :
    0 ∉ legalSeconds a r T := by
  simp [legalSeconds]

/-- The Checker's legal set is `{1, …, 60}` at every clock. -/
theorem legalSeconds_checker (a : Identity) (τ : ℕ) :
    legalSeconds a .checker (turnDuration τ) = Finset.Icc 1 60 := by
  unfold legalSeconds legalMaxSecond turnDuration
  split_ifs <;> simp_all

/-- Second `61` is legal exactly for Baku as Dropper in a turn of 61 seconds. -/
theorem sixtyOne_mem_legalSeconds_iff (a : Identity) (r : Role) (T : ℕ) :
    61 ∈ legalSeconds a r T ↔ 61 ≤ T ∧ r = .dropper ∧ a = .baku := by
  unfold legalSeconds legalMaxSecond
  split_ifs with h1 h2 <;> simp only [Finset.mem_Icc] <;> first | omega | (simp_all)

/-- Every legal second lies in `1..61`, and in `1..60` unless it is Baku's
leap drop. -/
theorem mem_legalSeconds_le (a : Identity) (r : Role) (τ : ℕ) {x : ℕ}
    (hx : x ∈ legalSeconds a r (turnDuration τ)) :
    1 ≤ x ∧ (x ≤ 60 ∨ (x = 61 ∧ IsLeapWindow τ ∧ r = .dropper ∧ a = .baku)) := by
  unfold legalSeconds legalMaxSecond turnDuration at hx
  simp only [Finset.mem_Icc] at hx
  split_ifs at hx with h1 h2 h3 <;> first
    | (refine ⟨hx.1, Or.inl ?_⟩; omega)
    | (by_cases h61 : x ≤ 60
       · exact ⟨hx.1, Or.inl h61⟩
       · exact ⟨hx.1, Or.inr ⟨by omega, h1, h3.1, h3.2⟩⟩)

/-- Hal drops in half `1`, Baku in half `2`. -/
def dropperOf (h : ℕ) : Identity := if h = 1 then .hal else .baku

/-- `leap_drop_available` (`canonical.py:72-75`): half `2` in the window. -/
def LeapDropAvailable (h τ : ℕ) : Prop := h = 2 ∧ IsLeapWindow τ

instance (h τ : ℕ) : Decidable (LeapDropAvailable h τ) := inferInstanceAs (Decidable (_ ∧ _))

/-- STL-RULE-2 (`AGENTS.md`, `src/stl/docs/GAME_AND_SOLVER.md:218-220`, `actions.py:35-52`,
`canonical.py:72-75`): the current Dropper may choose second `61` exactly when
`leap_drop_available` holds. -/
theorem sixtyOne_legal_iff_leapDropAvailable (h τ : ℕ) (hh : h = 1 ∨ h = 2) :
    61 ∈ legalSeconds (dropperOf h) .dropper (turnDuration τ) ↔ LeapDropAvailable h τ := by
  rw [sixtyOne_mem_legalSeconds_iff]
  unfold LeapDropAvailable dropperOf turnDuration
  rcases hh with rfl | rfl <;> split_ifs with hw <;> simp_all

/-! ## Next-minute snap (STL-CLOCK-2) -/

/-- `snap` of `leap_profiles.py:15-19` (= `snap_clock_to_next_minute`,
`game.py:318-345`). -/
def snap (c : ℕ) : ℕ :=
  if c < 3600 then (if (c / 60 + 1) * 60 = 3600 then 3601 else (c / 60 + 1) * 60)
  else if c = 3600 then 3601 else 3601 + ((c - 3601) / 60 + 1) * 60

/-- Wall-clock minute boundaries: multiples of 60 before the leap second, and
`3601 + 60 k` after it. -/
def IsBoundary (b : ℕ) : Prop := (b % 60 = 0 ∧ b < 3600) ∨ (3601 ≤ b ∧ (b - 3601) % 60 = 0)

/-- STL-CLOCK-2 (`paper/stl.tex:105-106`, `game.py:318-345`): `snap c` is the
least wall-clock minute boundary strictly after `c`. -/
theorem snap_isLeast (c : ℕ) : IsLeast {b | IsBoundary b ∧ c < b} (snap c) := by
  refine ⟨?_, ?_⟩
  · unfold snap IsBoundary
    show _ ∧ _
    split_ifs <;> omega
  · rintro b ⟨hb, hcb⟩
    unfold snap; unfold IsBoundary at hb
    split_ifs <;> omega

theorem lt_snap (c : ℕ) : c < snap c := (snap_isLeast c).1.2

theorem snap_ne_3600 (c : ℕ) : snap c ≠ 3600 := by
  unfold snap; split_ifs <;> omega

/-- `snap c ≤ c + 60`, except when the pre-leap boundary would be `3600`;
then `snap c = 3601 ≤ c + 61`. -/
theorem snap_le (c : ℕ) :
    snap c ≤ c + 60 ∨ (3540 ≤ c ∧ c ≤ 3599 ∧ snap c = 3601) := by
  unfold snap; split_ifs <;> omega

/-- A snapped clock at or below `3600` is a multiple of 60 and at most `3540`. -/
theorem snap_mod_of_le {c : ℕ} (h : snap c ≤ 3600) : snap c % 60 = 0 ∧ snap c ≤ 3540 := by
  unfold snap at *; split_ifs at * <;> omega

/-- The engine docstring examples. -/
example : snap 1740 = 1800 ∧ snap 1753 = 1800 ∧ snap 3810 = 3841 ∧ snap 3540 = 3601 ∧
    snap 3600 = 3601 := by decide

/-! ## Child clock (STL-CLOCK-1) -/

/-- The child clock of `leap_profiles.py:22-26`: `e = τ + T(τ) + (q + 120)·[q > 0]`,
then `e + 60` after half `1` and `snap e` after half `2`. The code raises on a
half outside `{1, 2}`; here any half other than `1` is treated as half `2`, and
the theorems assume `h = 1 ∨ h = 2`. -/
def childClock (h τ q : ℕ) : ℕ :=
  if h = 1 then τ + turnDuration τ + (if q = 0 then 0 else q + 120) + 60
  else snap (τ + turnDuration τ + (if q = 0 then 0 else q + 120))

/-- The engine's clock step (`game.py:478-512`): advance `T`, then
`death_duration + DEATH_PROCEDURE_OVERHEAD (120)` on a death, then
`WITHIN_ROUND_OVERHEAD (60)` after half `1` or a snap after half `2`. -/
def engineClock (h τ : ℕ) (death : Option ℕ) : ℕ :=
  let a := τ + turnDuration τ + (match death with | none => 0 | some dd => dd + 120)
  if h = 1 then a + 60 else snap a

/-- STL-CLOCK-1 (`paper/stl.tex:100-108`, `leap_profiles.py:22-26`,
`game.py:478-512`): the builder's child clock equals the engine's clock step,
with `q = 0` after a success and `q = dd > 0` after a revived death. -/
theorem childClock_eq_engineClock (h τ : ℕ) :
    childClock h τ 0 = engineClock h τ none ∧
    ∀ dd, 0 < dd → childClock h τ dd = engineClock h τ (some dd) := by
  refine ⟨?_, fun dd hdd => ?_⟩
  · simp [childClock, engineClock]
  · have : dd ≠ 0 := by omega
    simp [childClock, engineClock, this]

/-- The clocks of `test_clock_matches_engine` at the leap boundary. -/
example : childClock 1 3480 0 = 3600 ∧ childClock 2 3480 0 = 3601 ∧
    childClock 1 3540 0 = 3661 ∧ childClock 2 3540 60 = 3841 ∧ childClock 2 3600 0 = 3721 ∧
    childClock 1 720 60 = 1020 := by decide

/-! ## Clock monotonicity and the cold-table bound (STL-CLOCK-3) -/

/-- A valid dose: `0` after a success, `s + 60 ∈ [60, 299]` after a revival. -/
def ValidDose (q : ℕ) : Prop := q = 0 ∨ (60 ≤ q ∧ q ≤ 299)

/-- STL-CLOCK-3, first part: the child clock is at least `τ + 120` after half
`1` and at least `τ + 61` after half `2`. -/
theorem childClock_ge (h τ q : ℕ) :
    (h = 1 → τ + 120 ≤ childClock h τ q) ∧ (h = 2 → τ + 61 ≤ childClock h τ q) := by
  have hT := turnDuration_le τ
  have hs := lt_snap (τ + turnDuration τ + (if q = 0 then 0 else q + 120))
  refine ⟨fun h1 => ?_, fun h2 => ?_⟩
  · simp only [childClock, h1, ↓reduceIte]; omega
  · have : h ≠ 1 := by omega
    simp only [childClock, this, ↓reduceIte]; omega

theorem lt_childClock (h τ q : ℕ) (hh : h = 1 ∨ h = 2) : τ < childClock h τ q := by
  have := childClock_ge h τ q
  rcases hh with rfl | rfl
  · have := this.1 rfl; omega
  · have := this.2 rfl; omega

/-- STL-CLOCK-3 (`leap_build.py:247-255,489-508`): a child clock at or below
`3600` is at most `540` seconds after its parent when the dose is at most
`299`. The reader doubt that `q ≤ 299` needs live revival is settled by
`dose_le_299`. -/
theorem childClock_le_add_540 (h τ q : ℕ) (hq : q ≤ 299)
    (hle : childClock h τ q ≤ 3600) : childClock h τ q ≤ τ + 540 := by
  have hT := turnDuration_le τ
  unfold childClock at *
  split_ifs at * with h1 h2
  · omega
  · omega
  · rcases snap_le (τ + turnDuration τ + 0) with h3 | h3 <;> omega
  · rcases snap_le (τ + turnDuration τ + (q + 120)) with h3 | h3 <;> omega

/-- The revival dose `q = s + 60` of a revivable Checker is at most `299`: live
revival requires `s + 60 < 300`. -/
theorem dose_le_299 {s t : ℕ} (h : Survives s t) : ValidDose (s + 60) := by
  unfold Survives at h; right; omega

/-! ## Builder keys (STL-KEY-1, STL-RULE-2, STL-ORDER-1) -/

/-- A builder key (`leap_build.py:26-56`): `('H1', m)` at clock `60 m`,
`('H2', m)` at clock `60 m + 120`, and `('REV', c)` at clock `c`. -/
inductive Key
  | h1 (m : ℕ)
  | h2 (m : ℕ)
  | rev (c : ℕ)
  deriving DecidableEq

namespace Key

/-- The half of a key: `H1` is half `1`, `H2` and `REV` are half `2`. -/
def half : Key → ℕ
  | .h1 _ => 1
  | .h2 _ => 2
  | .rev _ => 2

/-- `key_clock` (`leap_build.py:26-28`). -/
def clock : Key → ℕ
  | .h1 m => 60 * m
  | .h2 m => 60 * m + 120
  | .rev c => c

/-- `key_columns` (`leap_build.py:35-36`). -/
def columns : Key → ℕ
  | .rev _ => 183
  | _ => 17011

/-- The ranges of `all_keys` (`leap_build.py:39-42`). -/
def Valid : Key → Prop
  | .h1 m => 12 ≤ m ∧ m ≤ 59
  | .h2 m => 12 ≤ m ∧ m ≤ 58
  | .rev c => 1020 ≤ c ∧ c ≤ 3600

/-- `is_window` (`leap_build.py:55-56`): not an `H1` key, clock in `[3540, 3600]`. -/
def IsWindow : Key → Prop
  | .h1 _ => False
  | k => 3540 ≤ k.clock ∧ k.clock ≤ 3600

instance (k : Key) : Decidable k.IsWindow := by
  cases k <;> unfold IsWindow <;> infer_instance

end Key

/-- STL-RULE-2 (`leap_build.py:55-56`, `canonical.py:72-75`): the builder's
`is_window(key)` is `leap_drop_available` at the key's half and clock. `H1`
keys are never windows. -/
theorem Key.isWindow_iff (k : Key) : k.IsWindow ↔ LeapDropAvailable k.half k.clock := by
  cases k <;> simp [Key.IsWindow, LeapDropAvailable, IsLeapWindow, Key.half, Key.clock]

/-- `child_key` (`leap_build.py:45-52`): the key of the child clock, or `none`
(the DTH table) when the child clock exceeds `3600`. -/
def childKey : Key → ℕ → Option Key
  | .h1 m, q =>
    if 3600 < childClock 1 (60 * m) q then none
    else if q = 0 then some (.h2 m) else some (.rev (childClock 1 (60 * m) q))
  | .h2 m, q =>
    if 3600 < childClock 2 (60 * m + 120) q then none
    else some (.h1 (childClock 2 (60 * m + 120) q / 60))
  | .rev c, q =>
    if 3600 < childClock 2 c q then none else some (.h1 (childClock 2 c q / 60))

/-- STL-KEY-1 (b): `child_key` is `none` exactly when the child clock exceeds
`3600`. -/
theorem childKey_eq_none_iff (k : Key) (q : ℕ) :
    childKey k q = none ↔ 3600 < childClock k.half k.clock q := by
  cases k <;> simp only [childKey, Key.half, Key.clock] <;> split_ifs <;> simp_all

/-- STL-KEY-1 (a), (b), (c) (`paper/stl.tex:94-96`, `leap_build.py:26-56`):
for a valid key and a valid dose, `child_key` returns a valid key of the
other half whose clock is the child clock. In particular every half-1 clock at
or below `3600` is a multiple of 60 and the least `REV` clock is `1020`. -/
theorem childKey_spec {k k' : Key} {q : ℕ} (hk : k.Valid) (hq : ValidDose q)
    (h : childKey k q = some k') :
    k'.clock = childClock k.half k.clock q ∧ k'.half = 3 - k.half ∧ k'.Valid := by
  cases k with
  | h1 m =>
    simp only [Key.Valid] at hk
    simp only [childKey] at h
    split_ifs at h with h1 h2
    · cases h
      subst h2
      have hw : ¬ IsLeapWindow (60 * m) := by
        intro hw
        apply h1
        simp only [childClock, turnDuration, hw, ↓reduceIte]
        unfold IsLeapWindow at hw; omega
      have hw' : ¬ (3540 ≤ 60 * m ∧ 60 * m ≤ 3600) := hw
      refine ⟨?_, rfl, ?_⟩
      · simp only [Key.clock, Key.half, childClock, turnDuration, hw, ↓reduceIte]
      · simp only [Key.Valid]; omega
    · cases h
      have hT := turnDuration_le (60 * m)
      refine ⟨rfl, rfl, ?_⟩
      simp only [Key.Valid]
      simp only [childClock, h2, ↓reduceIte] at h1 ⊢
      rcases hq with hq | hq <;> omega
  | h2 m =>
    simp only [Key.Valid] at hk
    simp only [childKey] at h
    split_ifs at h with h1
    cases h
    have h1' : snap (60 * m + 120 + turnDuration (60 * m + 120) +
        (if q = 0 then 0 else q + 120)) ≤ 3600 := by
      simp only [childClock, show (2 : ℕ) ≠ 1 by omega, ↓reduceIte] at h1; omega
    have hm := snap_mod_of_le h1'
    have hgt := lt_childClock 2 (60 * m + 120) q (by omega)
    simp only [childClock, show (2 : ℕ) ≠ 1 by omega, ↓reduceIte] at hgt ⊢
    refine ⟨?_, rfl, ?_⟩
    · simp only [Key.clock, Key.half, show (2 : ℕ) ≠ 1 by omega, ↓reduceIte]; omega
    · simp only [Key.Valid]; omega
  | rev c =>
    simp only [Key.Valid] at hk
    simp only [childKey] at h
    split_ifs at h with h1
    cases h
    have h1' : snap (c + turnDuration c + (if q = 0 then 0 else q + 120)) ≤ 3600 := by
      simp only [childClock, show (2 : ℕ) ≠ 1 by omega, ↓reduceIte] at h1; omega
    have hm := snap_mod_of_le h1'
    have hgt := lt_childClock 2 c q (by omega)
    simp only [childClock, show (2 : ℕ) ≠ 1 by omega, ↓reduceIte] at hgt ⊢
    refine ⟨?_, rfl, ?_⟩
    · simp only [Key.clock, Key.half, show (2 : ℕ) ≠ 1 by omega, ↓reduceIte]; omega
    · simp only [Key.Valid]; omega

theorem Key.half_eq (k : Key) : k.half = 1 ∨ k.half = 2 := by
  cases k <;> simp [Key.half]

/-- STL-ORDER-1: every child key has a strictly larger clock. -/
theorem clock_lt_of_childKey {k k' : Key} {q : ℕ} (hk : k.Valid) (hq : ValidDose q)
    (h : childKey k q = some k') : k.clock < k'.clock := by
  rw [(childKey_spec hk hq h).1]
  exact lt_childClock _ _ _ k.half_eq

/-- STL-ORDER-1 (`leap_build.py:559-573`, `paper/stl.tex:110-111`): in a list of
keys sorted by non-increasing clock (the order of `run_full`), every child key
appears before its parent, so its table is complete when the parent is swept.
Keys of equal clock are never parent and child. -/
theorem child_before_parent (L : List Key)
    (hL : L.Pairwise (fun a b => b.clock ≤ a.clock)) {i j : ℕ} (hi : i < L.length)
    (hj : j < L.length) {q : ℕ} (hk : L[i].Valid) (hq : ValidDose q)
    (h : childKey L[i] q = some L[j]) : j < i := by
  have hlt := clock_lt_of_childKey hk hq h
  by_contra hji
  rcases Nat.lt_or_eq_of_le (Nat.le_of_not_lt hji) with hij | hij
  · have := List.pairwise_iff_getElem.1 hL i j hi hj hij
    omega
  · subst hij; omega

/-- STL-CLOCK-3, consequence (`TableStore.pack_cold`, `commit_full_checkpoint`):
once the sweep has reached clock `clk`, every later parent has clock at most
`clk`, so it reads only child tables with clock at most `clk + 540`. A table
whose clock exceeds `clk + 540` is never read again. -/
theorem childKey_clock_le_add_540 {clk : ℕ} {p k' : Key} {q : ℕ} (hp : p.Valid)
    (hpc : p.clock ≤ clk) (hq : ValidDose q) (h : childKey p q = some k') :
    k'.clock ≤ clk + 540 := by
  obtain ⟨hc, -, hv⟩ := childKey_spec hp hq h
  have hle : childClock p.half p.clock q ≤ 3600 := by
    rw [← hc]; cases k' <;> simp only [Key.Valid, Key.clock] at hv ⊢ <;> omega
  have hq' : q ≤ 299 := by rcases hq with hq | hq <;> omega
  have := childClock_le_add_540 p.half p.clock q hq' hle
  omega

/-- STL-KEY-1 doubt: an `H2` key and a `REV` key can share a half and a clock.
`H1_12` with a revived dose `60` reaches `REV_1020`; `H1_15` with a success
reaches `H2_15`, whose clock is also `1020`. -/
theorem shared_half_clock :
    childKey (.h1 12) 60 = some (.rev 1020) ∧ childKey (.h1 15) 0 = some (.h2 15) ∧
    (Key.rev 1020).clock = (Key.h2 15).clock ∧ (Key.rev 1020).half = (Key.h2 15).half := by
  decide

/-! ## Wall-clock labels (STL-CLOCK-5) -/

/-- `format_game_clock` (`game.py:346-372`) as `(hour, minute, second)`. -/
def clockLabel (gc : ℕ) : ℕ × ℕ × ℕ :=
  if gc ≤ 3599 then (8 + gc / 3600, gc % 3600 / 60, gc % 60)
  else if gc = 3600 then (8, 59, 60)
  else (8 + (gc - 1) / 3600, (gc - 1) % 3600 / 60, (gc - 1) % 60)

/-- STL-CLOCK-5 (`src/stl/docs/GAME_AND_SOLVER.md:117-119`, `game.py:346-372`): the clock
label map is injective. -/
theorem clockLabel_injective : Function.Injective clockLabel := by
  intro a b h
  unfold clockLabel at h
  split_ifs at h <;> simp only [Prod.mk.injEq] at h <;> omega

/-- STL-CLOCK-5: after the leap second the minute boundaries `3601 + 60 k` are
the labels `9:00 + k` with zero seconds. -/
theorem clockLabel_after_leap (k : ℕ) :
    clockLabel (3601 + 60 * k) = (8 + (3600 + 60 * k) / 3600, (60 * k) % 3600 / 60, 0) := by
  unfold clockLabel
  simp only [show ¬ 3601 + 60 * k ≤ 3599 by omega, show 3601 + 60 * k ≠ 3600 by omega,
    ↓reduceIte, Prod.mk.injEq]
  omega

example : clockLabel 720 = (8, 12, 0) ∧ clockLabel 3540 = (8, 59, 0) ∧
    clockLabel 3600 = (8, 59, 60) ∧ clockLabel 3601 = (9, 0, 0) := by decide

/-! ## Leap Second Route classes (STL-LSR-1) -/

/-- `lsr_variation` (`canonical.py:60-63`). Python's `%` with a positive
modulus is `Int.emod`. -/
def lsrVariation (m : ℤ) : ℤ := 1 + (m - 12) % 4

/-- `is_active_lsr` (`canonical.py:66-69`). -/
def IsActiveLsr (m : ℤ) : Prop := lsrVariation m = 2

theorem lsrVariation_mem (m : ℤ) : 1 ≤ lsrVariation m ∧ lsrVariation m ≤ 4 := by
  unfold lsrVariation; omega

/-- The route classes are the residues of `m` modulo `4`: `V1 = {12, 16, …}`,
`V2 = {13, 17, …}`, `V3 = {14, 18, …}`, `V4 = {15, 19, …}`. -/
theorem lsrVariation_eq (m : ℕ) : lsrVariation m = 1 + ((m : ℤ) % 4) := by
  unfold lsrVariation; omega

/-- STL-LSR-1 (`src/stl/docs/GAME_AND_SOLVER.md:237-261`): a no-death round from `H1` at
minute `m ≤ 55` reaches `H2` at `60 m + 120` and returns to `H1` at minute
`m + 4`, so it preserves the route class. -/
theorem noDeath_round (m : ℕ) (hm : m ≤ 55) :
    childClock 1 (60 * m) 0 = 60 * m + 120 ∧ childClock 2 (60 * m + 120) 0 = 60 * (m + 4) ∧
    lsrVariation (m + 4 : ℕ) = lsrVariation m := by
  have h1 : turnDuration (60 * m) = 60 := (turnDuration_eq_60_iff _).2 (by omega)
  have h2 : turnDuration (60 * m + 120) = 60 := (turnDuration_eq_60_iff _).2 (by omega)
  refine ⟨?_, ?_, ?_⟩
  · simp only [childClock, h1, ↓reduceIte]
  · simp only [childClock, h2, show (2 : ℕ) ≠ 1 by omega, ↓reduceIte, snap]
    split_ifs <;> omega
  · unfold lsrVariation; push_cast; omega

/-- STL-LSR-1: from `H1` at minute `m` (12 ≤ m ≤ 59), a success reaches a
half-2 state where Baku may drop at second 61 exactly when `m = 57` or
`m = 58`. -/
theorem leap_after_success_iff (m : ℕ) (h1 : 12 ≤ m) (h2 : m ≤ 59) :
    LeapDropAvailable 2 (childClock 1 (60 * m) 0) ↔ m = 57 ∨ m = 58 := by
  interval_cases m <;> decide

/-- STL-LSR-1 doubt, settled against the code: the closed window makes route
`V3` active too. A round starting at 8:58 (`m = 58`, `V3`) places Baku's
half-2 turn at `τ = 3600`, where `is_window` holds and second 61 is legal, but
`is_active_lsr 58` is false. The claim "`is_active_lsr(m) ⇔ V(m) = 2` marks
the active route" is therefore false for the builder's closed window. -/
theorem lsr_V3_active :
    lsrVariation 58 = 3 ∧ ¬ IsActiveLsr 58 ∧ childClock 1 (60 * 58) 0 = 3600 ∧
    (Key.h2 58).IsWindow ∧ LeapDropAvailable 2 3600 := by
  unfold IsActiveLsr lsrVariation; decide

/-- The documented active case: a round at 8:57 (`V2`) places Baku's drop at
`τ = 3540`, inside the window. -/
theorem lsr_V2_active :
    lsrVariation 57 = 2 ∧ childClock 1 (60 * 57) 0 = 3540 ∧ LeapDropAvailable 2 3540 := by
  unfold lsrVariation; decide

/-! ## Half-round resolution (STL-RULE-4) -/

/-- The engine's revival probability `compute_survival_probability`
(`game.py:185-200`) for a death of duration `dd` with prior TTD `ttd`, over
`ℝ`. -/
noncomputable def engineSurvival (ttd dd : ℕ) : ℝ :=
  if 300 ≤ dd ∨ 300 < ttd + dd then 0
  else max 0 (min 1 (0.95 * (1 - max 0 ((dd : ℝ) - 60) / 240) * (0.75 : ℝ) ^ ((ttd : ℝ) / 60)))

/-- STL-RULE-4 (`src/stl/docs/GAME_AND_SOLVER.md:208-231`, `game.py:439-470`): a failed
check injects `min (s + 60) 300`, and the engine's revival probability for
that dose is the frozen surface `revival s t`. -/
theorem engineSurvival_fail (s t : ℕ) :
    engineSurvival t (min (s + 60) 300) = revival s t := by
  unfold engineSurvival
  by_cases hsv : Survives s t
  · have hsv' := hsv
    unfold Survives at hsv'
    have hmin : min (s + 60) 300 = s + 60 := by omega
    have hc : ¬ (300 ≤ s + 60 ∨ 300 < t + (s + 60)) := by omega
    rw [hmin]; simp only [hc, ↓reduceIte]
    have hr : revival s t = 0.95 * (1 - (s : ℝ) / 240) * (0.75 : ℝ) ^ ((t : ℝ) / 60) := by
      unfold revival; simp only [hsv, ↓reduceIte]
    have hmax : max 0 (((s + 60 : ℕ) : ℝ) - 60) = s := by
      push_cast; rw [max_eq_right (by ring_nf; positivity)]; ring
    rw [hmax, ← hr, min_eq_right (revival_le_one s t), max_eq_right (revival_nonneg s t)]
  · have hr : revival s t = 0 := by unfold revival; simp only [hsv, ↓reduceIte]
    have hc : 300 ≤ min (s + 60) 300 ∨ 300 < t + min (s + 60) 300 := by
      unfold Survives at hsv; omega
    rw [hr]; simp only [hc, ↓reduceIte]

/-- STL-RULE-4: a successful check that brings the vial to `300` or more is a
death with dose `300` and revival probability `0`, so the Dropper wins. -/
theorem engineSurvival_overflow (s ℓ t : ℕ) (h : 300 ≤ s + ℓ) :
    engineSurvival t (min (s + ℓ) 300) = 0 := by
  have hc : 300 ≤ min (s + ℓ) 300 ∨ 300 < t + min (s + ℓ) 300 := Or.inl (by omega)
  unfold engineSurvival; simp only [hc, ↓reduceIte]

/-- STL-RULE-4: legal seconds `d, c ∈ 1..60` with `c ≥ d` give a lag
`c - d + 1 ∈ 1..60`, and Baku's leap drop `61` defeats every legal check. -/
theorem lag_mem {d c : ℕ} (hd : 1 ≤ d) (hc : c ≤ 60) (h : d ≤ c) :
    1 ≤ c - d + 1 ∧ c - d + 1 ≤ 60 := by omega

theorem leapDrop_fails (a : Identity) {c : ℕ} (τ : ℕ)
    (hc : c ∈ legalSeconds a .checker (turnDuration τ)) : c < 61 := by
  rw [legalSeconds_checker] at hc; simp only [Finset.mem_Icc] at hc; omega

/-- STL-RULE-4: after a revived failed check the roles swap, the reviving
player's vial resets to `0`, and his TTD gains the dose `q = s + 60`. After a
live success the roles swap and the Checker's vial grows by the lag. -/
theorem children_profiles (x : State) (ℓ : ℕ) (h : x.c.s + ℓ < 300) :
    (failChild x).c = x.d ∧ (failChild x).d.s = 0 ∧ (failChild x).d.t = x.c.t + (x.c.s + 60) ∧
    (successChild x ℓ h).c = x.d ∧ (successChild x ℓ h).d.s = x.c.s + ℓ ∧
    (successChild x ℓ h).d.t = x.c.t := by
  refine ⟨rfl, rfl, ?_, rfl, rfl, rfl⟩
  simp only [failChild]; omega


/-- The engine's Checker update in `resolve_half_round` (`game.py:439-470`),
with `add_to_cylinder`, `on_death`, and `on_revival` (`game.py:109-139`). From
cylinder `cyl`, TTD `ttd`, drop `d`, check `c`, and the revival outcome
`survived`, it returns the Checker's new cylinder, TTD, and alive flag. A
success adds `ST = c - d + 1`; a cylinder of `300` or more is a death of
duration `min(cyl + ST, 300)`. A failed check adds `60` and is a death of
duration `min(cyl + 60, 300)`. A death adds its duration to the TTD, and a
revival resets the cylinder to `0`. -/
def engineChecker (cyl ttd d c : ℕ) (survived : Bool) : ℕ × ℕ × Bool :=
  if d ≤ c then
    if 300 ≤ cyl + (c - d + 1) then
      if survived then (0, ttd + min (cyl + (c - d + 1)) 300, true)
      else (cyl + (c - d + 1), ttd + min (cyl + (c - d + 1)) 300, false)
    else (cyl + (c - d + 1), ttd, true)
  else if survived then (0, ttd + min (cyl + 60) 300, true)
  else (cyl + 60, ttd + min (cyl + 60) 300, false)

/-- STL-RULE-4 (`game.py:439-470`): a live success in the engine gives the
Checker the profile of `successChild` with lag `c - d + 1`, whatever the
revival flag says, because no death occurs. -/
theorem engineChecker_success (x : State) {d c : ℕ} (b : Bool) (hdc : d ≤ c)
    (h : x.c.s + (c - d + 1) < 300) :
    engineChecker x.c.s x.c.t d c b =
      ((successChild x (c - d + 1) h).d.s, (successChild x (c - d + 1) h).d.t, true) := by
  simp only [engineChecker, hdc, ↓reduceIte, show ¬ 300 ≤ x.c.s + (c - d + 1) by omega,
    successChild]

/-- STL-RULE-4 (`game.py:449-451,461-462`): a revived failed check of a
revivable Checker gives him the profile of `failChild`: cylinder `0` and TTD
`t + (s + 60)`. -/
theorem engineChecker_fail (x : State) {d c : ℕ} (hdc : c < d) (hs : Survives x.c.s x.c.t) :
    engineChecker x.c.s x.c.t d c true =
      ((failChild x).d.s, (failChild x).d.t, true) := by
  have hs' := hs
  unfold Survives at hs'
  simp only [engineChecker, show ¬ d ≤ c by omega, ↓reduceIte, failChild, Prod.mk.injEq,
    true_and, and_true]
  omega

/-- STL-RULE-4 (`game.py:452-456`): the engine refuses a forced revival of
probability `0`. A failed check has positive revival probability exactly when
the Checker is revivable, and an overflow never has. -/
theorem engineSurvival_pos_iff (s t : ℕ) :
    0 < engineSurvival t (min (s + 60) 300) ↔ Survives s t := by
  rw [engineSurvival_fail]; exact revival_pos_iff s t

theorem engineSurvival_overflow_not_pos (s ℓ t : ℕ) (h : 300 ≤ s + ℓ) :
    ¬ 0 < engineSurvival t (min (s + ℓ) 300) := by
  rw [engineSurvival_overflow s ℓ t h]; exact lt_irrefl 0

example : engineChecker 10 0 5 7 false = (13, 0, true) ∧
    engineChecker 10 0 7 5 true = (0, 70, true) := by decide

/-! ## Profile classes and their ids (STL-QUOT-1) -/

/-- The first class id of the block of TTD `t` (`profiles`,
`leap_profiles.py:45-57`): block `0` holds ids `0..239`, and block `t ≥ 60`
starts at `240 + ∑_{u=60}^{t-1} (241 - u) = 240 + (t - 60)(423 - t)/2`. -/
def blockStart (t : ℕ) : ℕ := if t = 0 then 0 else 240 + (t - 60) * (423 - t) / 2

/-- The length of the block of TTD `t`: the number of eligible `s`. -/
def blockLen (t : ℕ) : ℕ := if t = 0 then 240 else 241 - t

/-- The TTD values with a nonempty block: `0` and `60..240`. -/
def ValidT (t : ℕ) : Prop := t = 0 ∨ (60 ≤ t ∧ t ≤ 240)

instance (t : ℕ) : Decidable (ValidT t) := inferInstanceAs (Decidable (_ ∨ _))

/-- The builder's class id: `id(s, t) = blockStart t + s` for a revivable
class and `N_ALIVE + s = 16711 + s` for a failure-fatal marker. -/
def classIndex : QProfile → ℕ
  | .alive s t => blockStart t + s
  | .dead s => 16711 + s

/-- The terminal-win sentinel column `WIN = N = 17011`. -/
def winIndex : ℕ := 17011

/-- The classes the builder enumerates: revivable `(s, t)` with
`t ∈ {0} ∪ [60, 300]`, and the 300 fatal markers. -/
def ValidClass : QProfile → Prop
  | .alive s t => (t = 0 ∨ 60 ≤ t) ∧ Survives s t
  | .dead s => s < 300

/-- A profile whose TTD is `0` or at least `60`, as every reachable TTD is. -/
def WellFormedTTD (p : Profile) : Prop := p.t = 0 ∨ 60 ≤ p.t

theorem quot_valid (p : Profile) (hp : WellFormedTTD p) : ValidClass (quot p) := by
  unfold quot
  split_ifs with h
  · exact ⟨hp, h⟩
  · exact p.hs

theorem survives_iff_blockLen (s t : ℕ) (ht : t = 0 ∨ 60 ≤ t) :
    Survives s t ↔ ValidT t ∧ s < blockLen t := by
  unfold Survives ValidT blockLen
  split_ifs <;> omega

theorem blockStart_add_len_le_of_lt :
    ∀ t < 241, ∀ t' < 241, ValidT t → ValidT t' → t < t' →
      blockStart t + blockLen t ≤ blockStart t' := by
  decide +kernel

theorem blockStart_add_len_le :
    ∀ t < 241, ValidT t → blockStart t + blockLen t ≤ 16711 := by
  decide +kernel

theorem blockStart_241 : blockStart 241 = 16711 := by decide

theorem ValidT.lt {t : ℕ} (h : ValidT t) : t < 241 := by unfold ValidT at h; omega

theorem validClass_alive_iff (s t : ℕ) :
    ValidClass (.alive s t) ↔ ValidT t ∧ s < blockLen t := by
  constructor
  · rintro ⟨ht, hs⟩; exact (survives_iff_blockLen s t ht).1 hs
  · rintro ⟨ht, hs⟩
    have ht' : t = 0 ∨ 60 ≤ t := by unfold ValidT at ht; omega
    exact ⟨ht', (survives_iff_blockLen s t ht').2 ⟨ht, hs⟩⟩

/-- STL-QUOT-1: every class id is below `N = 17011`, and revivable ids are
below `N_ALIVE = 16711`. -/
theorem classIndex_lt {c : QProfile} (hc : ValidClass c) :
    classIndex c < 17011 ∧ (∀ s t, c = .alive s t → classIndex c < 16711) := by
  cases c with
  | alive s t =>
    obtain ⟨ht, hs⟩ := (validClass_alive_iff s t).1 hc
    have := blockStart_add_len_le t ht.lt ht
    refine ⟨by simp only [classIndex]; omega, fun _ _ _ => by simp only [classIndex]; omega⟩
  | dead s =>
    simp only [ValidClass] at hc
    refine ⟨by simp only [classIndex]; omega, fun _ _ h => by cases h⟩

/-- STL-QUOT-1 (`leap_profiles.py:45-57`): alive ids are ordered by TTD block
first and by ST inside a block. -/
theorem classIndex_lt_iff {s t s' t' : ℕ} (h : ValidClass (.alive s t))
    (h' : ValidClass (.alive s' t')) :
    classIndex (.alive s t) < classIndex (.alive s' t') ↔ t < t' ∨ (t = t' ∧ s < s') := by
  obtain ⟨ht, hs⟩ := (validClass_alive_iff s t).1 h
  obtain ⟨ht', hs'⟩ := (validClass_alive_iff s' t').1 h'
  simp only [classIndex]
  rcases lt_trichotomy t t' with htt | rfl | htt
  · have := blockStart_add_len_le_of_lt t ht.lt t' ht'.lt ht ht' htt; omega
  · omega
  · have := blockStart_add_len_le_of_lt t' ht'.lt t ht.lt ht' ht htt; omega

/-- STL-QUOT-1: the class id is injective on the enumerated classes. -/
theorem classIndex_injective {c c' : QProfile} (hc : ValidClass c) (hc' : ValidClass c')
    (h : classIndex c = classIndex c') : c = c' := by
  cases c with
  | alive s t =>
    cases c' with
    | alive s' t' =>
      have h1 := (classIndex_lt_iff hc hc').not.1 (by omega)
      have h2 := (classIndex_lt_iff hc' hc).not.1 (by omega)
      have : t = t' ∧ s = s' := by omega
      rw [this.1, this.2]
    | dead s' =>
      have := (classIndex_lt hc).2 s t rfl
      simp only [classIndex] at h this; omega
  | dead s =>
    cases c' with
    | alive s' t' =>
      have := (classIndex_lt hc').2 s' t' rfl
      simp only [classIndex] at h this; omega
    | dead s' => simp only [classIndex] at h; rw [show s = s' by omega]

/-- An enumerated revivable pair `(t, s)`, as in `profiles()`: `t ∈ {0} ∪ [60, 300]`,
`s < 300`, and `eligible(s, t)`. -/
def AlivePair (p : ℕ × ℕ) : Prop := (p.1 = 0 ∨ 60 ≤ p.1) ∧ Survives p.2 p.1

instance : DecidablePred AlivePair := fun _ => inferInstanceAs (Decidable (_ ∧ _))

/-- The candidate pairs `t < 301`, `s < 300`. It is irreducible so that the
elaborator never evaluates it; the kernel still does in `alivePairs_card`. -/
@[irreducible] def pairDomain : Finset (ℕ × ℕ) := Finset.range 301 ×ˢ Finset.range 300

/-- The enumerated revivable `(t, s)` pairs. -/
def alivePairs : Finset (ℕ × ℕ) := pairDomain.filter AlivePair

/-- STL-QUOT-1 (`paper/stl.tex:91-96`): there are `16711 = 240 + ∑_{t=60}^{240}
(241 - t)` revivable classes. -/
theorem alivePairs_card : alivePairs.card = 16711 := by
  unfold alivePairs pairDomain; decide +kernel

theorem aliveCount_formula : 240 + ∑ t ∈ Finset.Icc 60 240, (241 - t) = 16711 := by
  decide +kernel

/-- STL-QUOT-1: the revivable ids fill `0..16710` exactly, so with the 300
fatal markers the classes are `0..17010` and `WIN = 17011`. -/
theorem validClass_of_mem_alivePairs {p : ℕ × ℕ} (hp : p ∈ alivePairs) :
    ValidClass (.alive p.2 p.1) := by
  exact (Finset.mem_filter.1 hp).2

theorem alivePairs_image :
    alivePairs.image (fun p => classIndex (.alive p.2 p.1)) = Finset.range 16711 := by
  have hvalid : ∀ p, p ∈ alivePairs → ValidClass (.alive p.2 p.1) :=
    fun _ hp => validClass_of_mem_alivePairs hp
  have hinj : Set.InjOn (fun p : ℕ × ℕ => classIndex (.alive p.2 p.1)) alivePairs := by
    intro p hp p' hp' h
    beta_reduce at h
    have h' : classIndex (.alive p.2 p.1) = classIndex (.alive p'.2 p'.1) := h
    have := classIndex_injective (hvalid p hp) (hvalid p' hp') h'
    simp only [QProfile.alive.injEq] at this
    exact Prod.ext this.2 this.1
  refine Finset.eq_of_subset_of_card_le ?_ ?_
  · intro i hi
    obtain ⟨p, hp, rfl⟩ := Finset.mem_image.1 hi
    have := (classIndex_lt (hvalid p hp)).2 p.2 p.1 rfl
    exact Finset.mem_range.2 this
  · rw [Finset.card_image_of_injOn hinj, alivePairs_card, Finset.card_range]

/-- STL-QUOT-1: the nonempty TTD blocks are `t = 0` and `t = 60..240`: 182
blocks. -/
theorem block_count :
    ((Finset.range 301).filter (fun t => (t = 0 ∨ 60 ≤ t) ∧ t ≤ 240)).card = 182 := by
  decide +kernel

/-- A TTD block is nonempty exactly when `t ≤ 240`. -/
theorem block_nonempty_iff (t : ℕ) : (∃ s < 300, Survives s t) ↔ t ≤ 240 := by
  constructor
  · rintro ⟨s, -, hs⟩; unfold Survives at hs; omega
  · intro h; exact ⟨0, by norm_num, by unfold Survives; omega⟩

/-! ## Success and failure maps on classes (STL-QUOT-2) -/

/-- `succ[pc, ℓ - 1]` (`leap_profiles.py:63-65`) on classes; `none` is `WIN`.
A fatal marker has TTD `301`, which no eligible id carries, so its successor is
the fatal marker of the grown ST. -/
def succClass : QProfile → ℕ → Option QProfile
  | .alive s t, ℓ =>
    if 300 ≤ s + ℓ then none
    else if Survives (s + ℓ) t then some (.alive (s + ℓ) t) else some (.dead (s + ℓ))
  | .dead s, ℓ => if 300 ≤ s + ℓ then none else some (.dead (s + ℓ))

/-- `fail[pc]` (`leap_profiles.py:66-67`) on classes; `none` is `WIN`. -/
def failClass : QProfile → Option QProfile
  | .alive s t => some (if Survives 0 (s + t + 60) then .alive 0 (s + t + 60) else .dead 0)
  | .dead _ => none

/-- `rev[pc]` (`leap_profiles.py:68`) on classes. -/
noncomputable def revClass : QProfile → ℝ
  | .alive s t => revival s t
  | .dead _ => 0

/-- STL-QUOT-2: the success map commutes with the quotient: the class of the
grown profile is `succ` of the class, and `WIN` exactly on overflow. -/
theorem succClass_quot (p : Profile) (ℓ : ℕ) :
    succClass (quot p) ℓ = if h : p.s + ℓ < 300 then some (quot ⟨p.s + ℓ, p.t, h⟩) else none := by
  unfold quot
  by_cases hs : Survives p.s p.t
  · simp only [hs, ↓reduceIte, succClass]
    by_cases h : p.s + ℓ < 300
    · simp only [show ¬ 300 ≤ p.s + ℓ by omega, ↓reduceIte, h, ↓reduceDIte]
      split_ifs <;> rfl
    · simp only [show 300 ≤ p.s + ℓ by omega, ↓reduceIte, h, ↓reduceDIte]
  · simp only [hs, ↓reduceIte, succClass]
    by_cases h : p.s + ℓ < 300
    · have h2 := not_survives_of_le (Nat.le_add_right p.s ℓ) hs
      simp only [show ¬ 300 ≤ p.s + ℓ by omega, ↓reduceIte, h, ↓reduceDIte, h2]
    · simp only [show 300 ≤ p.s + ℓ by omega, ↓reduceIte, h, ↓reduceDIte]

/-- STL-QUOT-2: the failure map commutes with the quotient on revivable
profiles, and a fatal class fails to `WIN`. -/
theorem failClass_quot (p : Profile) :
    failClass (quot p) = if h : Survives p.s p.t then
      some (quot ⟨0, p.t + p.s + 60, by norm_num⟩) else none := by
  unfold quot
  by_cases hs : Survives p.s p.t
  · simp only [hs, ↓reduceIte, ↓reduceDIte, failClass, Option.some.injEq]
    rw [show p.s + p.t + 60 = p.t + p.s + 60 by omega]
  · simp only [hs, ↓reduceIte, ↓reduceDIte, failClass]

/-- STL-QUOT-2: a revivable Checker fails to a revivable reset class exactly
when `s + t ≤ 180`. -/
theorem failClass_alive_iff (s t : ℕ) : Survives 0 (s + t + 60) ↔ s + t ≤ 180 := by
  unfold Survives; omega

/-- STL-QUOT-2: `rev[pc]` is the revival probability of every profile in the
class; it is `0` on fatal markers. -/
theorem revClass_quot (p : Profile) : revClass (quot p) = revival p.s p.t := by
  unfold quot
  split_ifs with h
  · rfl
  · simp only [revClass, revival, h, ↓reduceIte]

/-- STL-QUOT-2 / STL-REACH-1: the id of a success class. Inside a block of TTD
`t` the grown ST `s + ℓ` stays in the same block exactly when
`s + ℓ < blockLen t`; otherwise it is the fatal marker `16711 + s + ℓ`. -/
theorem classIndex_succClass_alive {s t ℓ : ℕ} (hv : ValidClass (.alive s t))
    (h : s + ℓ < 300) :
    (succClass (.alive s t) ℓ).map classIndex =
      some (if s + ℓ < blockLen t then blockStart t + (s + ℓ) else 16711 + (s + ℓ)) := by
  have ht : t = 0 ∨ 60 ≤ t := hv.1
  have hb := survives_iff_blockLen (s + ℓ) t ht
  have hvt := ((validClass_alive_iff s t).1 hv).1
  simp only [succClass, show ¬ 300 ≤ s + ℓ by omega, ↓reduceIte]
  by_cases hsv : Survives (s + ℓ) t
  · simp only [hsv, ↓reduceIte, Option.map_some, classIndex, show s + ℓ < blockLen t from
      (hb.1 hsv).2]
  · have : ¬ s + ℓ < blockLen t := fun h' => hsv (hb.2 ⟨hvt, h'⟩)
    simp only [hsv, ↓reduceIte, Option.map_some, classIndex, this]

theorem classIndex_succClass_dead {s ℓ : ℕ} (h : s + ℓ < 300) :
    (succClass (.dead s) ℓ).map classIndex = some (16711 + (s + ℓ)) := by
  simp only [succClass, show ¬ 300 ≤ s + ℓ by omega, ↓reduceIte, Option.map_some, classIndex]

/-! ## The 183 reset Dropper profiles (STL-QUOT-3) -/

/-- The reset classes `s0` of `leap_profiles.py:69`: `(0, 0)`, `(0, t)` for
`t = 60..240`, and the fatal marker with ST `0`. -/
def resetClass (j : ℕ) : QProfile :=
  if j = 0 then .alive 0 0 else if j ≤ 181 then .alive 0 (j + 59) else .dead 0

/-- `s0[j]`, the full class id of reset column `j`. -/
def s0 (j : ℕ) : ℕ := classIndex (resetClass j)

theorem resetClass_valid {j : ℕ} (hj : j < 183) : ValidClass (resetClass j) := by
  unfold resetClass
  split_ifs with h1 h2
  · exact ⟨Or.inl rfl, by unfold Survives; omega⟩
  · exact ⟨Or.inr (by omega), by unfold Survives; omega⟩
  · show 0 < 300; omega

theorem resetClass_injective {i j : ℕ} (hi : i < 183) (hj : j < 183)
    (h : resetClass i = resetClass j) : i = j := by
  unfold resetClass at h
  split_ifs at h <;> (simp_all; try omega)

/-- STL-QUOT-3: the 183 reset columns have distinct full ids. -/
theorem s0_injective {i j : ℕ} (hi : i < 183) (hj : j < 183) (h : s0 i = s0 j) : i = j :=
  resetClass_injective hi hj (classIndex_injective (resetClass_valid hi) (resetClass_valid hj) h)

theorem s0_card : ((Finset.range 183).image s0).card = 183 := by
  rw [Finset.card_image_of_injOn, Finset.card_range]
  intro i hi j hj h
  exact s0_injective (Finset.mem_range.1 hi) (Finset.mem_range.1 hj) h

/-- STL-QUOT-3 (`paper/stl.tex:95-96`): the reviving player's class after a
revived failed check is a reset class `j ∈ 1..182`. -/
theorem failChild_reset (x : State) :
    ∃ j, 1 ≤ j ∧ j < 183 ∧ quot (failChild x).d = resetClass j := by
  have hu : 60 ≤ x.c.t + x.c.s + 60 := by omega
  by_cases hs : x.c.t + x.c.s + 60 ≤ 240
  · refine ⟨x.c.t + x.c.s + 1, by omega, by omega, ?_⟩
    have hq : quot (failChild x).d = if Survives 0 (x.c.t + x.c.s + 60) then
        QProfile.alive 0 (x.c.t + x.c.s + 60) else QProfile.dead 0 := rfl
    have hsv : Survives 0 (x.c.t + x.c.s + 60) := by unfold Survives; omega
    rw [hq]
    simp only [hsv, ↓reduceIte, resetClass, show x.c.t + x.c.s + 1 ≠ 0 by omega,
      show x.c.t + x.c.s + 1 ≤ 181 by omega, show x.c.t + x.c.s + 1 + 59 = x.c.t + x.c.s + 60 by omega]
  · refine ⟨182, by omega, by omega, ?_⟩
    have hq : quot (failChild x).d = if Survives 0 (x.c.t + x.c.s + 60) then
        QProfile.alive 0 (x.c.t + x.c.s + 60) else QProfile.dead 0 := rfl
    have hsv : ¬ Survives 0 (x.c.t + x.c.s + 60) := by unfold Survives; omega
    rw [hq]
    simp only [hsv, ↓reduceIte, resetClass, show (182 : ℕ) ≠ 0 by omega,
      show ¬ (182 : ℕ) ≤ 181 by omega]

/-- STL-QUOT-3 doubt: the reset column `(0, 0)` (`j = 0`) is never the class of
a revived player, because his TTD is at least `60`. Only 182 of the 183
columns can hold `REV` states. -/
theorem failChild_ne_reset_zero (x : State) : quot (failChild x).d ≠ resetClass 0 := by
  obtain ⟨j, hj1, hj2, h⟩ := failChild_reset x
  rw [h]
  intro h'
  have := resetClass_injective hj2 (by omega) h'
  omega

/-- `idx0` (`leap_profiles.py:71`): the reset column of a full class id,
`183` for `WIN`, and `-1` elsewhere. -/
def idx0 (i : ℕ) : ℤ :=
  if i = winIndex then 183
  else ((List.range 183).find? (fun j => s0 j = i)).elim (-1) (fun j => (j : ℤ))

/-- STL-QUOT-3: `idx0` inverts `s0`. -/
theorem idx0_s0 : ∀ j < 183, idx0 (s0 j) = j := by
  decide +kernel

theorem idx0_win : idx0 winIndex = 183 := by simp [idx0]

/-- STL-QUOT-3: a nonnegative `idx0` value below `183` names the reset column
of its argument. -/
theorem eq_s0_of_idx0 {i : ℕ} {j : ℤ} (h : idx0 i = j) (hj0 : 0 ≤ j) (hj : j < 183) :
    i = s0 j.toNat := by
  unfold idx0 at h
  split_ifs at h with hw
  · omega
  · cases hf : (List.range 183).find? (fun j => s0 j = i) with
    | none => rw [hf] at h; simp at h; omega
    | some k =>
      rw [hf] at h
      simp only [Option.elim_some] at h
      have := List.find?_some hf
      simp only [decide_eq_true_eq] at this
      rw [← h]; simp [this]

/-- STL-QUOT-3: an id outside the reset columns and not `WIN` has `idx0 = -1`. -/
theorem idx0_eq_neg_one {i : ℕ} (hw : i ≠ winIndex) (h : ∀ j < 183, s0 j ≠ i) : idx0 i = -1 := by
  unfold idx0
  simp only [hw, ↓reduceIte]
  rw [List.find?_eq_none.2]
  · rfl
  · intro j hj
    simpa using h j (List.mem_range.1 hj)


/-- STL-QUOT-3 at the class level (`leap_profiles.py:66-67`): `fail[pc]` of a
revivable class is a reset class `j ∈ 1..182`. -/
theorem failClass_reset (s t : ℕ) :
    ∃ j, 1 ≤ j ∧ j < 183 ∧ failClass (.alive s t) = some (resetClass j) := by
  by_cases h : s + t + 60 ≤ 240
  · refine ⟨s + t + 1, by omega, by omega, ?_⟩
    have hsv : Survives 0 (s + t + 60) := by unfold Survives; omega
    simp only [failClass, hsv, ↓reduceIte, resetClass, show s + t + 1 ≠ 0 by omega,
      show s + t + 1 ≤ 181 by omega, show s + t + 1 + 59 = s + t + 60 by omega]
  · refine ⟨182, by omega, by omega, ?_⟩
    have hsv : ¬ Survives 0 (s + t + 60) := by unfold Survives; omega
    simp only [failClass, hsv, ↓reduceIte, resetClass, show (182 : ℕ) ≠ 0 by omega,
      show ¬ (182 : ℕ) ≤ 181 by omega]

/-- The success map keeps classes inside the enumeration. -/
theorem succClass_valid {a b : QProfile} {ℓ : ℕ} (ha : ValidClass a)
    (h : succClass a ℓ = some b) : ValidClass b := by
  cases a with
  | alive s t =>
    simp only [succClass] at h
    by_cases h1 : 300 ≤ s + ℓ
    · simp [h1] at h
    · by_cases h2 : Survives (s + ℓ) t
      · simp only [h1, h2, ↓reduceIte, Option.some.injEq] at h
        subst h; exact ⟨ha.1, h2⟩
      · simp only [h1, h2, ↓reduceIte, Option.some.injEq] at h
        subst h; show s + ℓ < 300; omega
  | dead s =>
    simp only [succClass] at h
    by_cases h1 : 300 ≤ s + ℓ
    · simp [h1] at h
    · simp only [h1, ↓reduceIte, Option.some.injEq] at h
      subst h; show s + ℓ < 300; omega

theorem resetClass_idx0 {j : ℕ} (hj : j < 183) :
    (idx0 (classIndex (resetClass j))).toNat = j ∧ s0 j = classIndex (resetClass j) := by
  refine ⟨?_, rfl⟩
  have := idx0_s0 j hj
  unfold s0 at this
  rw [this]; simp

/-! ## Success-vector shift (STL-QUOT-4) -/

/-- STL-QUOT-4 (`LEAP_CERTIFICATE.md:251-253`, `leap_profiles.py:63-65`): one
more second of Checker ST shifts the success classes by one lag:
`succ[id(s+1, t), k] = succ[id(s, t), k + 1]`. -/
theorem succClass_shift (p p' : Profile) (hs : p'.s = p.s + 1) (ht : p'.t = p.t) (ℓ : ℕ) :
    succClass (quot p') ℓ = succClass (quot p) (ℓ + 1) := by
  rw [succClass_quot, succClass_quot]
  by_cases h : p'.s + ℓ < 300
  · have h' : p.s + (ℓ + 1) < 300 := by omega
    simp only [h, h', ↓reduceDIte, Option.some.injEq]
    congr 1
    ext <;> (simp only [hs, ht]; try omega)
  · have h' : ¬ p.s + (ℓ + 1) < 300 := by omega
    simp only [h, h', ↓reduceDIte]

/-- STL-QUOT-4 at the level of states: raising the Checker's ST by one second
turns the success child of lag `ℓ + 1` into the success child of lag `ℓ`, so
the success payoff vectors satisfy `S'(k) = S(k + 1)`. -/
theorem successChild_shift (x : State) (hs : x.c.s + 1 < 300) (ℓ : ℕ)
    (h : x.c.s + 1 + ℓ < 300) (h' : x.c.s + (ℓ + 1) < 300) :
    successChild ⟨⟨x.c.s + 1, x.c.t, hs⟩, x.d⟩ ℓ h = successChild x (ℓ + 1) h' := by
  simp only [successChild, State.mk.injEq, true_and]
  ext <;> (simp only; try omega)

/-- The kink of `kink_index` (`leap_support.py:27-36`): the first index `k ∈ 1..59`
of the most negative step `S k - S (k - 1)`, required to be negative. -/
def IsKink (S : ℕ → ℝ) (k : ℕ) : Prop :=
  1 ≤ k ∧ k ≤ 59 ∧ S k - S (k - 1) < 0 ∧
  (∀ j, 1 ≤ j → j ≤ 59 → S k - S (k - 1) ≤ S j - S (j - 1)) ∧
  (∀ j, 1 ≤ j → j < k → S k - S (k - 1) < S j - S (j - 1))

/-- STL-QUOT-4, evident meaning: after the shift `S'(k) = S(k + 1)` the kink
moves from `k` to `k - 1`, provided the old kink is not the removed first step
(`k ≥ 2`) and the new last step is no smaller than the old minimum. -/
theorem isKink_shift {S S' : ℕ → ℝ} {k : ℕ} (hk : IsKink S k) (hk2 : 2 ≤ k)
    (hshift : ∀ i ≤ 58, S' i = S (i + 1))
    (hlast : S k - S (k - 1) ≤ S' 59 - S' 58) : IsKink S' (k - 1) := by
  obtain ⟨h1, h2, h3, h4, h5⟩ := hk
  have e1 : S' (k - 1) = S k := by rw [hshift _ (by omega)]; congr 1; omega
  have e2 : S' (k - 1 - 1) = S (k - 1) := by rw [hshift _ (by omega)]; congr 1; omega
  refine ⟨by omega, by omega, by rw [e1, e2]; exact h3, fun j hj1 hj2 => ?_, fun j hj1 hj2 => ?_⟩
  · rw [e1, e2]
    rcases Nat.lt_or_ge j 59 with hj | hj
    · rw [hshift j (by omega), hshift (j - 1) (by omega),
        show j - 1 + 1 = j + 1 - 1 by omega]
      exact h4 (j + 1) (by omega) (by omega)
    · have : j = 59 := by omega
      subst this; exact hlast
  · rw [e1, e2, hshift j (by omega), hshift (j - 1) (by omega),
      show j - 1 + 1 = j + 1 - 1 by omega]
    exact h5 (j + 1) (by omega) (by omega)

/-- STL-QUOT-4 doubt: the literal "the kink moves by `-1`" fails when the kink
is the first step. `S = (1, 0, 0, …)` has its kink at `1`; after the shift the
vector is constant and has no kink. -/
theorem kink_shift_counterexample :
    IsKink (fun i => if i = 0 then 1 else 0) 1 ∧ ∀ k, ¬ IsKink (fun _ => (0 : ℝ)) k := by
  refine ⟨⟨le_rfl, by norm_num, by norm_num, fun j hj _ => ?_, fun j hj1 hj2 => by omega⟩,
    fun k hk => by simp [IsKink] at hk⟩
  by_cases hj1 : j = 1
  · subst hj1; exact le_rfl
  · simp [show j ≠ 0 by omega, show j - 1 ≠ 0 by omega]

/-- STL-KEY-1 (d) (`paper/stl.tex:94-96,152`): a Dropper reached by a success
(an `H2` key) has ST at least `1`, and a revived Dropper (a `REV` key) has ST
`0`. Their classes differ, so an `H2` key and a `REV` key at the same clock
never hold the same quotient state, and the per-key counts do not double count. -/
theorem h2_rev_disjoint (x y : State) {ℓ : ℕ} (hℓ : 1 ≤ ℓ) (h : x.c.s + ℓ < 300) :
    1 ≤ (successChild x ℓ h).d.s ∧ (failChild y).d.s = 0 ∧
    quot (successChild x ℓ h).d ≠ quot (failChild y).d := by
  refine ⟨by simp only [successChild]; omega, rfl, fun hq => ?_⟩
  have := quot_s hq
  simp only [successChild, failChild] at this
  omega

/-! ## Stage matrices (STL-STAGE-2) -/

/-- The square stage `M[d, c] = s[c - d]` for `c ≥ d`, else `f`. -/
def stage60 (s : ℕ → ℝ) (f : ℝ) : Matrix (Fin 60) (Fin 60) ℝ := Toeplitz.toeplitz 60 s f

/-- The window stage as `explicit_matrix` builds it (`leap_audit.py:143-149`):
`M[d, c] = s[c - d]` when `c ≥ d`, else `f`, with 61 rows. Row `60` is Baku's
drop at second 61, which no check reaches. -/
def stage61 (s : ℕ → ℝ) (f : ℝ) : Matrix (Fin 61) (Fin 60) ℝ :=
  fun d c => if d.val ≤ c.val then s (c.val - d.val) else f

/-- The window stage as `stage_matrix` builds it (`leap_oracle.py:26-30`): fill
with `f`, then set `matrix[row, row:] = s[:60 - row]` for `row < 60`. -/
def oracleStage61 (s : ℕ → ℝ) (f : ℝ) : Matrix (Fin 61) (Fin 60) ℝ :=
  fun d c => if d.val < 60 then (if d.val ≤ c.val then s (c.val - d.val) else f) else f

/-- STL-STAGE-2 (`leap_oracle.py:26-30`, `leap_audit.py:143-149`): the two
constructions of the window stage agree. -/
theorem oracleStage61_eq (s : ℕ → ℝ) (f : ℝ) : oracleStage61 s f = stage61 s f := by
  funext d c
  unfold oracleStage61 stage61
  have := c.isLt
  split_ifs <;> first | rfl | omega

/-- STL-STAGE-2: the first 60 rows of the window stage are the square stage,
and the extra row is 60 copies of `f`. -/
theorem stage61_castSucc (s : ℕ → ℝ) (f : ℝ) (d c : Fin 60) :
    stage61 s f d.castSucc c = stage60 s f d c := by
  unfold stage61 stage60
  rw [Toeplitz.toeplitz_apply]
  by_cases hdc : d ≤ c
  · have h' : (d : ℕ) ≤ c := hdc
    simp only [Fin.val_castSucc, h', hdc, ↓reduceIte]
  · have h' : ¬ (d : ℕ) ≤ c := hdc
    simp only [Fin.val_castSucc, h', hdc, ↓reduceIte]

theorem stage61_last (s : ℕ → ℝ) (f : ℝ) (c : Fin 60) : stage61 s f (Fin.last 60) c = f := by
  have := c.isLt
  simp only [stage61, Fin.val_last, show ¬ 60 ≤ c.val by omega, ↓reduceIte]

/-- STL-STAGE-2 (`paper/stl.tex:155-162`): all success cells of equal lag hold
the same payoff, so the stage needs only the 60 success payoffs and `f`. -/
theorem stage60_lag (s : ℕ → ℝ) (f : ℝ) {d c d' c' : Fin 60} (h : d ≤ c) (h' : d' ≤ c')
    (hl : c.val - d.val = c'.val - d'.val) : stage60 s f d c = stage60 s f d' c' := by
  simp only [stage60, Toeplitz.toeplitz_apply, h, h', ↓reduceIte, hl]

/-! ## The leap-aware value (STL-STAGE-1, STL-ORDER-1) -/

/-- The half after a half-round: roles swap. -/
def otherHalf (h : ℕ) : ℕ := if h = 1 then 2 else 1

theorem otherHalf_mem (h : ℕ) : otherHalf h = 1 ∨ otherHalf h = 2 := by
  unfold otherHalf; split_ifs <;> simp

/-- The success payoff `S_ℓ` to the current Dropper (`evaluate_states`,
`leap_build.py:162-175`): `+1` on overflow (the `WIN` column stores `-1`), and
otherwise minus the child value at the success clock. -/
noncomputable def succPayL (W : State → ℕ → ℕ → ℝ) (x : State) (h τ ℓ : ℕ) : ℝ :=
  if hs : x.c.s + ℓ < 300 then -W (successChild x ℓ hs) (otherHalf h) (childClock h τ 0) else 1

/-- The failure payoff `F = rev (-V_fail) + (1 - rev)` in the DTH form
`1 - p - p V_fail`, with the failure child at clock `childClock h τ (s + 60)`. -/
noncomputable def failPayL (W : State → ℕ → ℕ → ℝ) (x : State) (h τ : ℕ) : ℝ :=
  if Survives x.c.s x.c.t then
    1 - revival x.c.s x.c.t -
      revival x.c.s x.c.t * W (failChild x) (otherHalf h) (childClock h τ (x.c.s + 60))
  else 1

/-- The stage value: the `61 × 60` window stage when Baku may drop at 61,
the square stage otherwise (`solve_stage(s, f, is_window(key))`). -/
noncomputable def stageL (W : State → ℕ → ℕ → ℝ) (x : State) (h τ : ℕ) : ℝ :=
  if LeapDropAvailable h τ then
    value (stage61 (fun k => succPayL W x h τ (k + 1)) (failPayL W x h τ))
  else value (stage60 (fun k => succPayL W x h τ (k + 1)) (failPayL W x h τ))

/-- The L2 value of `(x, h, τ)` from the current Dropper's view, by backward
induction over the DTH potential `Φ`, which every live transition raises
(STL-ORDER-1: the L2 game graph is a finite DAG). -/
noncomputable def VL (x : State) (h τ : ℕ) : ℝ :=
  stageL (fun y h' τ' => if phi x < phi y then VL y h' τ' else 0) x h τ
termination_by 1201 - phi x
decreasing_by
  all_goals
    have := phi_le y
    omega

theorem succPayL_congr {W W' : State → ℕ → ℕ → ℝ} (x : State) (h τ ℓ : ℕ) (hℓ : 1 ≤ ℓ)
    (hW : ∀ y h' τ', phi x < phi y → W y h' τ' = W' y h' τ') :
    succPayL W x h τ ℓ = succPayL W' x h τ ℓ := by
  unfold succPayL
  split_ifs with hs
  · rw [hW _ _ _ (phi_lt_successChild x hℓ hs)]
  · rfl

theorem failPayL_congr {W W' : State → ℕ → ℕ → ℝ} (x : State) (h τ : ℕ)
    (hW : ∀ y h' τ', phi x < phi y → W y h' τ' = W' y h' τ') :
    failPayL W x h τ = failPayL W' x h τ := by
  unfold failPayL
  split_ifs with hs
  · rw [hW _ _ _ (phi_lt_failChild x hs)]
  · rfl

theorem stageL_congr {W W' : State → ℕ → ℕ → ℝ} (x : State) (h τ : ℕ)
    (hW : ∀ y h' τ', phi x < phi y → W y h' τ' = W' y h' τ') :
    stageL W x h τ = stageL W' x h τ := by
  have hs : (fun k => succPayL W x h τ (k + 1)) = (fun k => succPayL W' x h τ (k + 1)) :=
    funext fun k => succPayL_congr x h τ (k + 1) (by omega) hW
  unfold stageL
  rw [hs, failPayL_congr x h τ hW]

/-- STL-STAGE-1 (`paper/stl.tex:155-162`, `leap_build.py:162-175`): the
Bellman equation. `VL x h τ` is the value of its stage, whose success entries
are `-VL` of the success child at clock `childClock h τ 0` and whose failure
entry is `1 - p - p VL` of the revived child at `childClock h τ (s + 60)`. -/
theorem VL_bellman (x : State) (h τ : ℕ) : VL x h τ = stageL VL x h τ := by
  rw [VL]
  exact stageL_congr x h τ fun y h' τ' hy => by simp [hy]

/-- STL-STAGE-1: the builder's failure payoff `rev · (-T_fail) + (1 - rev)`, used
when `rev > 0`, equals `failPayL`; a failure-fatal Checker gives `F = 1`. -/
theorem failPayL_eq_builder (W : State → ℕ → ℕ → ℝ) (x : State) (h τ : ℕ) :
    failPayL W x h τ =
      if 0 < revival x.c.s x.c.t then
        revival x.c.s x.c.t *
            -W (failChild x) (otherHalf h) (childClock h τ (x.c.s + 60)) +
          (1 - revival x.c.s x.c.t)
      else 1 := by
  unfold failPayL
  by_cases hs : Survives x.c.s x.c.t
  · simp only [hs, (revival_pos_iff _ _).2 hs, ↓reduceIte]; ring
  · have : ¬ 0 < revival x.c.s x.c.t := fun h' => hs ((revival_pos_iff _ _).1 h')
    simp only [hs, this, ↓reduceIte]

/-- STL-STAGE-1: the table row read by a success is the old Dropper's class, and
the column is `succ[pc, ℓ - 1]` of the old Checker's class. -/
theorem successChild_index (x : State) (ℓ : ℕ) (hs : x.c.s + ℓ < 300) :
    (successChild x ℓ hs).c = x.d ∧ succClass (quot x.c) ℓ = some (quot (successChild x ℓ hs).d) := by
  refine ⟨rfl, ?_⟩
  rw [succClass_quot]
  simp only [hs, ↓reduceDIte]
  rfl

/-- STL-STAGE-1: the failure column is `fail[pc]` of the old Checker's class. -/
theorem failChild_index (x : State) (hs : Survives x.c.s x.c.t) :
    (failChild x).c = x.d ∧ failClass (quot x.c) = some (quot (failChild x).d) := by
  refine ⟨rfl, ?_⟩
  rw [failClass_quot]
  simp only [hs, ↓reduceDIte]
  rfl


/-- STL-STAGE-1 (`leap_build.py:314-330`, `leap_audit.py:152-163`): the
builder's `REV` failure lookup `fail_table[pd, idx0[fail[pc]]]` reads the fail
child. For a revivable Checker, `fail[pc]` is the class of the fail child's
Dropper, `idx0` of its id is a reset column `j ∈ 1..182`, and reset column `j`
is that class id. -/
theorem failLookup_idx0 (x : State) (hs : Survives x.c.s x.c.t) :
    ∃ j : ℕ, 1 ≤ j ∧ j < 183 ∧ failClass (quot x.c) = some (quot (failChild x).d) ∧
      idx0 (classIndex (quot (failChild x).d)) = j ∧
      s0 j = classIndex (quot (failChild x).d) := by
  obtain ⟨j, h1, h2, h⟩ := failChild_reset x
  refine ⟨j, h1, h2, (failChild_index x hs).2, ?_, ?_⟩
  · rw [h]; exact idx0_s0 j h2
  · rw [h]; rfl

/-! ## The quotient at a fixed half and clock (STL-QUOT-1) -/

/-- STL-QUOT-1 (`paper/stl.tex:91-96`): at a fixed half and clock, two states
whose Checker and Dropper profiles lie in the same classes have the same L2
value. A success uses only the grown ST, a failure of a fatal Checker ends the
game, the child clock depends only on `(h, τ, s + 60)`, and a fatal profile
never becomes revivable again. -/
theorem VL_eq_of_qequiv (x y : State) (hq : QEquiv x y) (h τ : ℕ) : VL x h τ = VL y h τ := by
  induction hm : 1201 - phi x using Nat.strong_induction_on generalizing x y h τ with
  | _ k ih =>
  have hs := quot_s hq.1
  have hsucc : ∀ ℓ, 1 ≤ ℓ → succPayL VL x h τ ℓ = succPayL VL y h τ ℓ := by
    intro ℓ hℓ
    unfold succPayL
    by_cases h1 : x.c.s + ℓ < 300
    · have h2 : y.c.s + ℓ < 300 := by omega
      simp only [h1, h2, ↓reduceDIte]
      congr 1
      exact ih _ (by
          have := phi_lt_successChild x hℓ h1
          have := phi_le (successChild x ℓ h1)
          omega) _ _ (qequiv_successChild hq ℓ h1 h2) _ _ rfl
    · have h2 : ¬ y.c.s + ℓ < 300 := by omega
      simp only [h1, h2, ↓reduceDIte]
  have hfail : failPayL VL x h τ = failPayL VL y h τ := by
    unfold failPayL
    by_cases h1 : Survives x.c.s x.c.t
    · have h2 := (survives_iff_of_quot hq.1).1 h1
      have ht := t_eq_of_quot_of_survives hq.1 h1
      simp only [h1, h2, ↓reduceIte]
      rw [ih _ (by
          have := phi_lt_failChild x h1
          have := phi_le (failChild x)
          omega) _ _ (qequiv_failChild hq h1) _ _ rfl, hs, ht]
    · have h2 : ¬ Survives y.c.s y.c.t := fun h' => h1 ((survives_iff_of_quot hq.1).2 h')
      simp only [h1, h2, ↓reduceIte]
  have hS : (fun k => succPayL VL x h τ (k + 1)) = (fun k => succPayL VL y h τ (k + 1)) :=
    funext fun k => hsucc (k + 1) (by omega)
  rw [VL_bellman x, VL_bellman y]
  unfold stageL
  rw [hS, hfail]

/-! ## The exact DTH boundary after 3600 (STL-CLOCK-4) -/

/-- STL-CLOCK-4 (`src/stl/docs/GAME_AND_SOLVER.md:330-348`, `paper/stl.tex:110-112`,
`leap_build.py:45-52,282-304`): after 8:59:60 every descendant clock exceeds
`3600`, so no stage is a window and the L2 value is the pure DTH value of the
role-relative state. The builder reads the DTH table whenever `child_key`
returns `None`. -/
theorem VL_eq_V (x : State) (h τ : ℕ) (hh : h = 1 ∨ h = 2) (hτ : 3600 < τ) :
    VL x h τ = DTH.V x := by
  induction hm : 1201 - phi x using Nat.strong_induction_on generalizing x h τ with
  | _ k ih =>
  have hsucc : ∀ ℓ, 1 ≤ ℓ → succPayL VL x h τ ℓ = succPay DTH.V x ℓ := by
    intro ℓ hℓ
    unfold succPayL succPay
    by_cases h1 : x.c.s + ℓ < 300
    · simp only [h1, ↓reduceDIte]
      congr 1
      exact ih _ (by
          have := phi_lt_successChild x hℓ h1
          have := phi_le (successChild x ℓ h1)
          omega) _ _ _ (otherHalf_mem h)
          (by have := lt_childClock h τ 0 hh; omega) rfl
    · simp only [h1, ↓reduceDIte]
  have hfail : failPayL VL x h τ = failPay DTH.V x := by
    unfold failPayL failPay
    by_cases h1 : Survives x.c.s x.c.t
    · simp only [h1, ↓reduceIte]
      rw [ih _ (by
          have := phi_lt_failChild x h1
          have := phi_le (failChild x)
          omega) _ _ _ (otherHalf_mem h)
          (by have := lt_childClock h τ (x.c.s + 60) hh; omega) rfl]
    · simp only [h1, ↓reduceIte]
  have hnw : ¬ LeapDropAvailable h τ := by
    unfold LeapDropAvailable IsLeapWindow; omega
  have hS : (fun k => succPayL VL x h τ (k + 1)) = (fun k => succPay DTH.V x (k + 1)) :=
    funext fun k => hsucc (k + 1) (by omega)
  rw [VL_bellman, DTH.V_bellman x, stage_eq_toeplitz]
  unfold stageL
  simp only [hnw, ↓reduceIte]
  rw [hS, hfail]
  rfl

/-! ## Utility range (STL-VAL-1) -/

theorem failPayL_mem {W : State → ℕ → ℕ → ℝ} (x : State) (h τ : ℕ)
    (hW : ∀ y h' τ', -1 ≤ W y h' τ' ∧ W y h' τ' ≤ 1) :
    -1 ≤ failPayL W x h τ ∧ failPayL W x h τ ≤ 1 := by
  unfold failPayL
  split_ifs
  · have hp0 := revival_nonneg x.c.s x.c.t
    have hp1 := revival_le_one x.c.s x.c.t
    obtain ⟨hw0, hw1⟩ := hW (failChild x) (otherHalf h) (childClock h τ (x.c.s + 60))
    constructor <;> nlinarith
  · norm_num

theorem succPayL_mem {W : State → ℕ → ℕ → ℝ} (x : State) (h τ ℓ : ℕ)
    (hW : ∀ y h' τ', -1 ≤ W y h' τ' ∧ W y h' τ' ≤ 1) :
    -1 ≤ succPayL W x h τ ℓ ∧ succPayL W x h τ ℓ ≤ 1 := by
  unfold succPayL
  split_ifs with hs
  · obtain ⟨h1, h2⟩ := hW (successChild x ℓ hs) (otherHalf h) (childClock h τ 0)
    constructor <;> linarith
  · norm_num

/-- STL-VAL-1 (`paper/stl.tex:155-162`, `leap_audit.py:90-91`): every L2 value
lies in `[-1, 1]`; stage payoffs are `-V_child`, `+1`, or convex combinations
of `-V_child` and `+1`. -/
theorem VL_mem (x : State) (h τ : ℕ) : -1 ≤ VL x h τ ∧ VL x h τ ≤ 1 := by
  induction hm : 1201 - phi x using Nat.strong_induction_on generalizing x h τ with
  | _ k ih =>
  have hW : ∀ y h' τ', -1 ≤ (fun y h' τ' => if phi x < phi y then VL y h' τ' else 0) y h' τ' ∧
      (fun y h' τ' => if phi x < phi y then VL y h' τ' else 0) y h' τ' ≤ 1 := by
    intro y h' τ'
    simp only
    split_ifs with hy
    · exact ih _ (by have := phi_le y; omega) y h' τ' rfl
    · norm_num
  rw [VL]
  have hs := fun k => succPayL_mem (W := fun y h' τ' => if phi x < phi y then VL y h' τ' else 0)
    x h τ (k + 1) hW
  have hf := failPayL_mem (W := fun y h' τ' => if phi x < phi y then VL y h' τ' else 0) x h τ hW
  unfold stageL
  split_ifs
  · refine value_mem_entry_range _ (fun d c => ?_) (fun d c => ?_) <;> simp only [stage61] <;>
      split_ifs <;> (first | exact (hs _).1 | exact (hs _).2 | exact hf.1 | exact hf.2)
  · refine value_mem_entry_range _ (fun d c => ?_) (fun d c => ?_) <;>
      simp only [stage60, Toeplitz.toeplitz_apply] <;>
      split_ifs <;> (first | exact (hs _).1 | exact (hs _).2 | exact hf.1 | exact hf.2)

/-- STL-VAL-1: the Dropper's win probability `(1 + V) / 2` lies in `[0, 1]`. -/
theorem winProb_mem (x : State) (h τ : ℕ) :
    0 ≤ (1 + VL x h τ) / 2 ∧ (1 + VL x h τ) / 2 ≤ 1 := by
  obtain ⟨h1, h2⟩ := VL_mem x h τ
  constructor <;> linarith

/-- STL-VAL-1 arithmetic (`paper/stl.tex:55,238-253`): with `V = 0.03717`,
Hal wins `51.86%`; the DTH root `0.08985007281413855` gives `54.49%`; Baku's
gain `50 (V_DTH - V_STL)` rounds to `2.63` points; and the routing counts sum
to the reachable total. The root value itself is an artifact value. -/
theorem root_arithmetic :
    (1 + (0.03717 : ℝ)) / 2 = 0.518585 ∧
    |(1 + (0.08985007281413855 : ℝ)) / 2 - 0.5449| < 0.00005 ∧
    |50 * ((0.08985007281413855 : ℝ) - 0.03717) - 2.63| < 0.005 ∧
    (8843634465 : ℕ) + 609698652 = 9453333117 := by
  refine ⟨by norm_num, ?_, ?_, by norm_num⟩
  · rw [abs_lt]; constructor <;> norm_num
  · rw [abs_lt]; constructor <;> norm_num

/-! ## Public Markov state, projection, and opening node
(STL-MARKOV-1, STL-PROJ-1, STL-OPEN-1) -/

/-- `WorldState` of `canonical.py:19-27`, with each player's `(load, ttd)` as a
DTH profile. -/
structure WorldState where
  baku : Profile
  hal : Profile
  half : ℕ
  clock : ℕ
  halLeapMemory : Bool

/-- `GameState` of `canonical.py:39-45`: the world state and the revealed
public history `(drop, check, survived)`. -/
structure GameState where
  world : WorldState
  history : List (ℕ × ℕ × Option Bool)

/-- The role-relative projection of `src/stl/docs/GAME_AND_SOLVER.md:310-328`: half `1` is
`(baku, hal)` (Baku checks, Hal drops), half `2` is `(hal, baku)`. -/
def project (w : WorldState) : State := if w.half = 1 then ⟨w.baku, w.hal⟩ else ⟨w.hal, w.baku⟩

/-- Hal's utility: the Dropper-view value keeps its sign in half `1` and is
negated in half `2`. -/
noncomputable def halUtility (g : GameState) : ℝ :=
  if g.world.half = 1 then VL (project g.world) 1 g.world.clock
  else -VL (project g.world) 2 g.world.clock

/-- STL-MARKOV-1 (`src/stl/docs/GAME_AND_SOLVER.md:13-16,191-206,381-393`): the L2 value
depends only on `(baku_load, baku_ttd, hal_load, hal_ttd, half, clock)`; Hal's
leap memory and the public history do not enter it. This holds by construction:
the rules (`turnDuration`, `legalSeconds`, `childClock`, `revival`) and `VL`
take only physical arguments, so the proof unfolds the definition. -/
theorem halUtility_physical (g g' : GameState) (hb : g.world.baku = g'.world.baku)
    (hh : g.world.hal = g'.world.hal) (hhalf : g.world.half = g'.world.half)
    (hc : g.world.clock = g'.world.clock) : halUtility g = halUtility g' := by
  simp only [halUtility, project, hb, hh, hhalf, hc]

/-- STL-PROJ-1 (`src/stl/docs/GAME_AND_SOLVER.md:310-328`): half `1` projects Baku to the
Checker slot and Hal to the Dropper slot, half `2` the reverse; after the leap
second Hal's utility is `+V_DTH` in half `1` and `-V_DTH` in half `2`. -/
theorem halUtility_after_leap (g : GameState) (hh : g.world.half = 1 ∨ g.world.half = 2)
    (hc : 3600 < g.world.clock) :
    halUtility g = if g.world.half = 1 then DTH.V ⟨g.world.baku, g.world.hal⟩
      else -DTH.V ⟨g.world.hal, g.world.baku⟩ := by
  unfold halUtility project
  rcases hh with h1 | h2
  · simp only [h1, ↓reduceIte]; exact VL_eq_V _ 1 _ (Or.inl rfl) hc
  · simp only [h2, show (2 : ℕ) ≠ 1 by omega, ↓reduceIte]
    rw [VL_eq_V _ 2 _ (Or.inr rfl) hc]

theorem dropperOf_projection : dropperOf 1 = .hal ∧ dropperOf 2 = .baku := by decide

/-- STL-MARKOV-1 doubt: each stage is a simultaneous-move matrix game, not a
perfect-information move. This square stage has no pure saddle point: every
Dropper row is answered by a column worth `-1`, and every Checker column by a
row worth `+1`. -/
theorem stage_no_pure_saddle :
    (∀ d : Fin 60, ∃ c, stage60 (fun k => if k = 0 then -1 else 1) 1 d c = -1) ∧
    (∀ c : Fin 60, ∃ d, stage60 (fun k => if k = 0 then -1 else 1) 1 d c = 1) := by
  refine ⟨fun d => ⟨d, by simp [stage60, Toeplitz.toeplitz_apply]⟩, fun c => ?_⟩
  by_cases hc : c.val = 0
  · refine ⟨⟨1, by norm_num⟩, ?_⟩
    have : ¬ (⟨1, by norm_num⟩ : Fin 60) ≤ c := by rw [Fin.le_def]; simp; omega
    simp only [stage60, Toeplitz.toeplitz_apply, this, ↓reduceIte]
  · refine ⟨⟨0, by norm_num⟩, ?_⟩
    have : (⟨0, by norm_num⟩ : Fin 60) ≤ c := by rw [Fin.le_def]; simp
    simp only [stage60, Toeplitz.toeplitz_apply, this, ↓reduceIte]
    simp [hc]

/-- The canonical opening (`root_node`, `canonical.py:78-89`): 8:12:00, half
`1`, all loads and TTDs `0`, no leap memory. -/
def rootWorld : WorldState := ⟨⟨0, 0, by norm_num⟩, ⟨0, 0, by norm_num⟩, 1, 720, false⟩

/-- STL-OPEN-1 (`src/stl/docs/GAME_AND_SOLVER.md:80-99`, `paper/stl.tex:63-64,87-89`,
`leap_build.py:116`): the opening is cell `[0, 0]` of key `H1_12`, Hal drops,
and the opening stage is not a window. -/
theorem root_node :
    project rootWorld = ⟨⟨0, 0, by norm_num⟩, ⟨0, 0, by norm_num⟩⟩ ∧
    classIndex (quot ⟨0, 0, by norm_num⟩) = 0 ∧ (Key.h1 12).clock = rootWorld.clock ∧
    (Key.h1 12).Valid ∧ ¬ (Key.h1 12).IsWindow ∧ ¬ LeapDropAvailable rootWorld.half rootWorld.clock ∧
    dropperOf rootWorld.half = .hal ∧ turnDuration rootWorld.clock = 60 := by
  refine ⟨rfl, ?_, rfl, by simp [Key.Valid], by simp [Key.IsWindow], by decide, by decide,
    by decide⟩
  have : Survives 0 0 := by unfold Survives; norm_num
  simp [quot, this, classIndex, blockStart]

/-! ## The success-image transform `_success_bits` (STL-REACH-1) -/

/-- One in-place step `expanded[d:] |= expanded[:-d]` on an array of length `n`
(numpy evaluates an overlapping in-place ufunc as if on a copy). -/
def orShift (n d : ℕ) (A : Set ℕ) : Set ℕ := A ∪ {x | d ≤ x ∧ x < n ∧ x - d ∈ A}

/-- The six shifts `1, 2, 4, 8, 16, 28` of `leap_build.py:71-72`. -/
def orShifts (n : ℕ) (A : Set ℕ) : Set ℕ := [1, 2, 4, 8, 16, 28].foldl (fun B d => orShift n d B) A

/-- The set of indices below `n` reached from `A` by an offset `k ≤ w`. -/
def shiftWindow (n w : ℕ) (A : Set ℕ) : Set ℕ := {x | x < n ∧ ∃ k ≤ w, k ≤ x ∧ x - k ∈ A}

theorem orShift_window {n w d : ℕ} {A : Set ℕ} (hd : d ≤ w + 1) :
    orShift n d (shiftWindow n w A) = shiftWindow n (w + d) A := by
  ext x
  simp only [orShift, shiftWindow, Set.mem_union, Set.mem_ofPred_eq]
  constructor
  · rintro (⟨hx, k, hk, hkx, hA⟩ | ⟨hdx, hx, -, k, hk, hkx, hA⟩)
    · exact ⟨hx, k, by omega, hkx, hA⟩
    · exact ⟨hx, k + d, by omega, by omega, by rwa [show x - (k + d) = x - d - k by omega]⟩
  · rintro ⟨hx, k, hk, hkx, hA⟩
    by_cases hkw : k ≤ w
    · exact Or.inl ⟨hx, k, hkw, hkx, hA⟩
    · refine Or.inr ⟨by omega, hx, by omega, k - d, by omega, by omega, ?_⟩
      rwa [show x - d - (k - d) = x - k by omega]

theorem shiftWindow_zero {n : ℕ} {A : Set ℕ} (hA : ∀ a ∈ A, a < n) : shiftWindow n 0 A = A := by
  ext x
  simp only [shiftWindow, Set.mem_ofPred_eq]
  constructor
  · rintro ⟨-, k, hk, -, hA'⟩; rwa [show x - k = x by omega] at hA'
  · intro hx; exact ⟨hA x hx, 0, le_rfl, by omega, by simpa using hx⟩

/-- STL-REACH-1 key sublemma: the six OR-shifts by `1, 2, 4, 8, 16, 28` add
every offset `0..59`, because the subset sums of `{1, 2, 4, 8, 16, 28}` are
exactly `{0, …, 59}`. -/
theorem orShifts_eq {n : ℕ} {A : Set ℕ} (hA : ∀ a ∈ A, a < n) :
    orShifts n A = shiftWindow n 59 A := by
  rw [← shiftWindow_zero hA]
  simp only [orShifts, List.foldl]
  rw [orShift_window (by norm_num), orShift_window (by norm_num), orShift_window (by norm_num),
    orShift_window (by norm_num), orShift_window (by norm_num), orShift_window (by norm_num)]
  rw [shiftWindow_zero hA]

/-- The seeded array of one block (`leap_build.py:69-70`): row `r` of the block
is written at index `r + 1` for `r < min(length, 299)`. -/
def seed (L : ℕ) (R : Set ℕ) : Set ℕ := {i | 1 ≤ i ∧ i - 1 ∈ R ∧ i - 1 < min L 299}

/-- The block's `expanded` array after the shifts; its length is
`min(300, length + 60)`. -/
def expandedSet (L : ℕ) (R : Set ℕ) : Set ℕ := orShifts (min 300 (L + 60)) (seed L R)

/-- STL-REACH-1: index `i` of `expanded` is set exactly when some set row `r`
of the block reaches ST `i = r + ℓ < 300` by a lag `ℓ ∈ 1..60`. -/
theorem mem_expandedSet {L : ℕ} {R : Set ℕ} (hL : L ≤ 300) (hR : ∀ r ∈ R, r < L) (i : ℕ) :
    i ∈ expandedSet L R ↔ ∃ r ∈ R, ∃ ℓ, 1 ≤ ℓ ∧ ℓ ≤ 60 ∧ r + ℓ < 300 ∧ i = r + ℓ := by
  have hseed : ∀ a ∈ seed L R, a < min 300 (L + 60) := by
    rintro a ⟨h1, -, h3⟩; omega
  unfold expandedSet
  rw [orShifts_eq hseed]
  simp only [shiftWindow, seed, Set.mem_ofPred_eq]
  constructor
  · rintro ⟨hi, k, hk, hki, h1, hr, h3⟩
    exact ⟨i - k - 1, hr, k + 1, by omega, by omega, by omega, by omega⟩
  · rintro ⟨r, hr, ℓ, h1, h2, h3, rfl⟩
    have := hR r hr
    refine ⟨by omega, ℓ - 1, by omega, by omega, by omega, ?_, by omega⟩
    rwa [show r + ℓ - (ℓ - 1) - 1 = r by omega]

/-- Where `_success_bits` places index `i` of a block of TTD `t`
(`leap_build.py:75-77`): inside the block if `i < length`, else at the fatal
marker `N_ALIVE + i`. -/
def placeIndex (t i : ℕ) : ℕ := if i < blockLen t then blockStart t + i else 16711 + i

/-- STL-REACH-1 (`leap_build.py:64-78`): for a revivable block of TTD `t` and a
set `R` of its rows, `_success_bits` sets exactly the class ids
`succ[id(r, t), ℓ - 1]` for `r ∈ R` and `ℓ ∈ 1..60` that are not `WIN`. -/
theorem success_bits_block {t : ℕ} (ht : ValidT t) {R : Set ℕ} (hR : ∀ r ∈ R, r < blockLen t)
    (id : ℕ) :
    (∃ i ∈ expandedSet (blockLen t) R, id = placeIndex t i) ↔
      ∃ r ∈ R, ∃ ℓ, 1 ≤ ℓ ∧ ℓ ≤ 60 ∧ (succClass (.alive r t) ℓ).map classIndex = some id := by
  have hL : blockLen t ≤ 300 := by unfold blockLen; split_ifs <;> omega
  constructor
  · rintro ⟨i, hi, rfl⟩
    obtain ⟨r, hr, ℓ, h1, h2, h3, rfl⟩ := (mem_expandedSet hL hR i).1 hi
    refine ⟨r, hr, ℓ, h1, h2, ?_⟩
    rw [classIndex_succClass_alive ((validClass_alive_iff r t).2 ⟨ht, hR r hr⟩) h3]
    rfl
  · rintro ⟨r, hr, ℓ, h1, h2, hid⟩
    by_cases h3 : r + ℓ < 300
    · rw [classIndex_succClass_alive ((validClass_alive_iff r t).2 ⟨ht, hR r hr⟩) h3] at hid
      refine ⟨r + ℓ, (mem_expandedSet hL hR _).2 ⟨r, hr, ℓ, h1, h2, h3, rfl⟩, ?_⟩
      simp only [Option.some.injEq] at hid
      rw [← hid]; rfl
    · simp [succClass, show 300 ≤ r + ℓ by omega] at hid

/-- STL-REACH-1, fatal block (`start = N_ALIVE`, length `300`): row `s` maps to
the fatal marker `N_ALIVE + s + ℓ` for `s + ℓ < 300`. -/
theorem success_bits_fatal {R : Set ℕ} (hR : ∀ r ∈ R, r < 300) (id : ℕ) :
    (∃ i ∈ expandedSet 300 R, id = 16711 + i) ↔
      ∃ r ∈ R, ∃ ℓ, 1 ≤ ℓ ∧ ℓ ≤ 60 ∧ (succClass (.dead r) ℓ).map classIndex = some id := by
  constructor
  · rintro ⟨i, hi, rfl⟩
    obtain ⟨r, hr, ℓ, h1, h2, h3, rfl⟩ := (mem_expandedSet le_rfl hR i).1 hi
    exact ⟨r, hr, ℓ, h1, h2, classIndex_succClass_dead h3⟩
  · rintro ⟨r, hr, ℓ, h1, h2, hid⟩
    by_cases h3 : r + ℓ < 300
    · rw [classIndex_succClass_dead h3] at hid
      simp only [Option.some.injEq] at hid
      exact ⟨r + ℓ, (mem_expandedSet le_rfl hR _).2 ⟨r, hr, ℓ, h1, h2, h3, rfl⟩, hid.symm⟩
    · simp [succClass, show 300 ≤ r + ℓ by omega] at hid

/-! ## Forward reachability order (STL-REACH-2) -/

section Worklist

variable {α : Type*} (clk : α → ℕ) (step : α → α → Prop) (init : α)

/-- The set a clock-ordered worklist holds for state `a` when it pops `a`'s key
(`forward_reachability`, `leap_build.py:102-159`): the initial state, or the
image of a processed state of strictly smaller clock. -/
def Processed (a : α) : Prop :=
  a = init ∨ ∃ b, ∃ _ : clk b < clk a, Processed b ∧ step b a
termination_by clk a

theorem processed_iff_unfold (a : α) :
    Processed clk step init a ↔ a = init ∨ ∃ b, ∃ _ : clk b < clk a, Processed clk step init b ∧ step b a := by
  rw [Processed]

/-- STL-REACH-2, abstract worklist lemma (`leap_build.py:102-159`): when every
transition strictly raises the clock, the clock-ordered closure is the
reachable set. `creach_iff` instantiates it with the builder's class-level step
`CStep` and the key clock. -/
theorem processed_iff_reachable (hstep : ∀ a b, step a b → clk a < clk b) (a : α) :
    Processed clk step init a ↔ Relation.ReflTransGen step init a := by
  constructor
  · intro ha
    induction hm : clk a using Nat.strong_induction_on generalizing a with
    | _ k ih =>
    rcases (processed_iff_unfold clk step init a).1 ha with rfl | ⟨b, hb, hpb, hs⟩
    · exact Relation.ReflTransGen.refl
    · exact (ih _ (hm ▸ hb) b hpb rfl).tail hs
  · intro ha
    induction ha with
    | refl => exact (processed_iff_unfold clk step init init).2 (Or.inl rfl)
    | tail _ hbc ih =>
      exact (processed_iff_unfold clk step init _).2 (Or.inr ⟨_, hstep _ _ hbc, ih, hbc⟩)

end Worklist


/-! ## Forward reachability on keys and classes (STL-REACH-2)

`forward_reachability` (`leap_build.py:102-159`) keeps one bitmap per key. Row
`r` is the Checker's class id; column `c` is the Dropper's class id, or, for a
`REV` key, the reset column `idx0` of the Dropper's class. This section models
a bitmap as the set of its set `(row, column)` bits, mirrors the per-key
update, and proves that the bitmap of every key is the encoded set of
reachable quotient pairs. -/

/-- `key[0] == 'REV'`. -/
def Key.isRev : Key → Bool
  | .rev _ => true
  | _ => false

/-- The support of one L2 half-round from `(k, x)` (`src/stl/docs/GAME_AND_SOLVER.md:208-231`):
a live success of lag `ℓ ∈ 1..60` goes to the success key, and a revived failed
check of a revivable Checker goes to the failure key of dose `s + 60`. A child
whose key is `none` (the DTH table) or whose clock exceeds `limit` is not
tracked, as in `forward_reachability`. -/
inductive VStep (limit : ℕ) : Key × State → Key × State → Prop
  | succ {k k' : Key} {x : State} (ℓ : ℕ) (hs : x.c.s + ℓ < 300) (hk : k.Valid) (h1 : 1 ≤ ℓ)
      (h60 : ℓ ≤ 60) (hc : childKey k 0 = some k') (hl : k'.clock ≤ limit) :
      VStep limit (k, x) (k', successChild x ℓ hs)
  | rev {k k' : Key} {x : State} (hk : k.Valid) (hsv : Survives x.c.s x.c.t)
      (hc : childKey k (x.c.s + 60) = some k') (hl : k'.clock ≤ limit) :
      VStep limit (k, x) (k', failChild x)

/-- Every lag `ℓ ∈ 1..60` comes from legal seconds (`d = 1`, `c = ℓ`), and a
failed check comes from `d = 2`, `c = 1`; a revival has positive probability
exactly for a revivable Checker. So `VStep` is the support of the L2
transition. -/
theorem vstep_support :
    (∀ ℓ, 1 ≤ ℓ → ℓ ≤ 60 → ∃ d c, 1 ≤ d ∧ d ≤ c ∧ c ≤ 60 ∧ c - d + 1 = ℓ) ∧
    (∃ d c : ℕ, 1 ≤ c ∧ c < d ∧ d ≤ 60) ∧ (∀ s t, 0 < revival s t ↔ Survives s t) :=
  ⟨fun ℓ h1 h2 => ⟨1, ℓ, le_rfl, h1, h2, by omega⟩, ⟨2, 1, le_rfl, by norm_num, by norm_num⟩,
    revival_pos_iff⟩

/-- STL-REACH-2 / STL-STAGE-1: a `VStep` child is a state that the Bellman
equation `VL_bellman` reads, at the other half and at the child clock of its
dose. -/
theorem vstep_bellman_child {limit : ℕ} {k k' : Key} {x y : State}
    (h : VStep limit (k, x) (k', y)) :
    k'.half = otherHalf k.half ∧
      ((∃ ℓ, ∃ hs : x.c.s + ℓ < 300, 1 ≤ ℓ ∧ ℓ ≤ 60 ∧ y = successChild x ℓ hs ∧
          k'.clock = childClock k.half k.clock 0) ∨
        (Survives x.c.s x.c.t ∧ y = failChild x ∧
          k'.clock = childClock k.half k.clock (x.c.s + 60))) := by
  have hoh : ∀ k : Key, 3 - k.half = otherHalf k.half := fun k => by
    rcases k.half_eq with h | h <;> simp [h, otherHalf]
  cases h with
  | succ ℓ hs hk h1 h60 hc hl =>
    obtain ⟨hcl, hh, -⟩ := childKey_spec hk (Or.inl rfl) hc
    exact ⟨hh.trans (hoh k), Or.inl ⟨ℓ, hs, h1, h60, rfl, hcl⟩⟩
  | rev hk hsv hc hl =>
    obtain ⟨hcl, hh, -⟩ := childKey_spec hk (dose_le_299 hsv) hc
    exact ⟨hh.trans (hoh k), Or.inr ⟨hsv, rfl, hcl⟩⟩

/-- The class-level step on `(key, checker class, dropper class)`
(`leap_build.py:127-154`): a success moves the old Dropper to the Checker slot
and makes `succ[pc, ℓ - 1]` the new Dropper (a `WIN` successor is dropped); a
revival of a revivable Checker class makes `fail[pc]` the new Dropper at the
key of dose `st[pc] + 60`. -/
inductive CStep (limit : ℕ) : Key × QProfile × QProfile → Key × QProfile × QProfile → Prop
  | succ {k k' : Key} {a b b' : QProfile} (ℓ : ℕ) (hk : k.Valid) (h1 : 1 ≤ ℓ) (h60 : ℓ ≤ 60)
      (hsc : succClass a ℓ = some b') (hc : childKey k 0 = some k') (hl : k'.clock ≤ limit) :
      CStep limit (k, a, b) (k', b, b')
  | rev {k k' : Key} {s t : ℕ} {b f : QProfile} (hk : k.Valid) (hsv : Survives s t)
      (hf : failClass (.alive s t) = some f) (hc : childKey k (s + 60) = some k')
      (hl : k'.clock ≤ limit) :
      CStep limit (k, .alive s t, b) (k', b, f)

/-- The projection of a keyed state to its keyed class pair. -/
def classOf (kx : Key × State) : Key × QProfile × QProfile := (kx.1, quot kx.2.c, quot kx.2.d)

theorem vstep_classOf {limit : ℕ} {a a' : Key × State} (h : VStep limit a a') :
    CStep limit (classOf a) (classOf a') := by
  cases h with
  | succ ℓ hs hk h1 h60 hc hl =>
    exact CStep.succ ℓ hk h1 h60 (successChild_index _ ℓ hs).2 hc hl
  | @rev k k' x hk hsv hc hl =>
    have hq : quot x.c = .alive x.c.s x.c.t := by simp only [quot, hsv, ↓reduceIte]
    have hf := (failChild_index x hsv).2
    rw [hq] at hf
    show CStep limit (k, quot x.c, quot x.d) (k', quot x.d, quot (failChild x).d)
    rw [hq]
    exact CStep.rev hk hsv hf hc hl

theorem cstep_lift {limit : ℕ} {z z' : Key × QProfile × QProfile} (h : CStep limit z z') :
    ∀ kx : Key × State, classOf kx = z → ∃ kx', VStep limit kx kx' ∧ classOf kx' = z' := by
  cases h with
  | @succ k k' a b b' ℓ hk h1 h60 hsc hc hl =>
    rintro ⟨k0, x⟩ he
    simp only [classOf, Prod.mk.injEq] at he
    obtain ⟨rfl, rfl, rfl⟩ := he
    rw [succClass_quot] at hsc
    by_cases hs : x.c.s + ℓ < 300
    · simp only [hs, ↓reduceDIte, Option.some.injEq] at hsc
      refine ⟨(k', successChild x ℓ hs), VStep.succ ℓ hs hk h1 h60 hc hl, ?_⟩
      simp only [classOf, successChild, ← hsc]
    · simp [hs] at hsc
  | @rev k k' s t b f hk hsv hf hc hl =>
    rintro ⟨k0, x⟩ he
    simp only [classOf, Prod.mk.injEq] at he
    obtain ⟨rfl, hqc, rfl⟩ := he
    have hsx : Survives x.c.s x.c.t := by
      by_contra h'
      simp only [quot, h', ↓reduceIte] at hqc
      cases hqc
    have hst : x.c.s = s ∧ x.c.t = t := by
      simp only [quot, hsx, ↓reduceIte, QProfile.alive.injEq] at hqc; exact hqc
    obtain ⟨rfl, rfl⟩ := hst
    have hf' := (failChild_index x hsx).2
    rw [hqc, hf, Option.some.injEq] at hf'
    refine ⟨(k', failChild x), VStep.rev hk hsx hc hl, ?_⟩
    simp only [classOf, hf']
    rfl

/-- A simulation lemma: a step relation on `A` that projects onto a step
relation on `B`, with every `B`-step from a projected point lifted, has the
projected reachable set. -/
theorem reach_map_iff {A B : Type*} (π : A → B) {sA : A → A → Prop} {sB : B → B → Prop}
    (h1 : ∀ a a', sA a a' → sB (π a) (π a'))
    (h2 : ∀ a b', sB (π a) b' → ∃ a', sA a a' ∧ π a' = b') (a0 : A) (b : B) :
    Relation.ReflTransGen sB (π a0) b ↔ ∃ a, Relation.ReflTransGen sA a0 a ∧ π a = b := by
  constructor
  · intro h
    induction h with
    | refl => exact ⟨a0, .refl, rfl⟩
    | tail _ hs ih =>
      obtain ⟨a, ha, rfl⟩ := ih
      obtain ⟨a', ha', rfl⟩ := h2 a _ hs
      exact ⟨a', ha.tail ha', rfl⟩
  · rintro ⟨a, ha, rfl⟩
    induction ha with
    | refl => exact .refl
    | tail _ hs ih => exact ih.tail (h1 _ _ hs)

/-- The opening state (`root_node`): both profiles `(0, 0)`. -/
def rootState : State := ⟨⟨0, 0, by norm_num⟩, ⟨0, 0, by norm_num⟩⟩

/-- The seed of `forward_reachability` (`leap_build.py:116`): key `H1_12`,
Checker class `0 = (0, 0)`, Dropper class `0 = (0, 0)`. -/
def cRoot : Key × QProfile × QProfile := (.h1 12, .alive 0 0, .alive 0 0)

theorem classOf_root : classOf (.h1 12, rootState) = cRoot := by
  have : Survives 0 0 := by unfold Survives; norm_num
  simp [classOf, rootState, cRoot, quot, this]

/-- STL-REACH-2, quotient part (`paper/stl.tex:94-96`): a class pair is
reachable at a key under `CStep` exactly when some L2 state that is reachable
at that key under `VStep` has these classes. -/
theorem creach_iff_vreach (limit : ℕ) (z : Key × QProfile × QProfile) :
    Relation.ReflTransGen (CStep limit) cRoot z ↔
      ∃ kx, Relation.ReflTransGen (VStep limit) (.h1 12, rootState) kx ∧ classOf kx = z := by
  rw [← classOf_root]
  exact reach_map_iff classOf (fun _ _ h => vstep_classOf h)
    (fun a b' h => cstep_lift h a rfl) _ z

theorem childKey_zero_isRev {k k' : Key} (h : childKey k 0 = some k') : k'.isRev = false := by
  cases k <;> simp only [childKey] at h <;> split_ifs at h <;> (cases h; rfl)

/-- Every `CStep` raises the key clock (STL-CLOCK-3). -/
theorem cstep_clock {limit : ℕ} {z z' : Key × QProfile × QProfile} (h : CStep limit z z') :
    z.1.clock < z'.1.clock := by
  cases h with
  | succ ℓ hk h1 h60 hsc hc hl => exact clock_lt_of_childKey hk (Or.inl rfl) hc
  | rev hk hsv hf hc hl => exact clock_lt_of_childKey hk (dose_le_299 hsv) hc

/-- The invariant of a reachable keyed class pair: the key is valid, both
classes are enumerated, and a `REV` key's Dropper is a reset class. -/
def ReachInv (k : Key) (a b : QProfile) : Prop :=
  k.Valid ∧ ValidClass a ∧ ValidClass b ∧ (k.isRev = true → ∃ j < 183, b = resetClass j)

theorem reach_inv {limit : ℕ} {z : Key × QProfile × QProfile}
    (h : Relation.ReflTransGen (CStep limit) cRoot z) : ReachInv z.1 z.2.1 z.2.2 := by
  induction h with
  | refl =>
    refine ⟨⟨le_rfl, by norm_num⟩, ⟨Or.inl rfl, by unfold Survives; norm_num⟩,
      ⟨Or.inl rfl, by unfold Survives; norm_num⟩, fun h => by simp [cRoot, Key.isRev] at h⟩
  | tail _ hs ih =>
    cases hs with
    | succ ℓ hk h1 h60 hsc hc hl =>
      obtain ⟨-, ha, hb, -⟩ := ih
      refine ⟨(childKey_spec hk (Or.inl rfl) hc).2.2, hb, succClass_valid ha hsc, fun h => ?_⟩
      rw [childKey_zero_isRev hc] at h; cases h
    | rev hk hsv hf hc hl =>
      obtain ⟨-, -, hb, -⟩ := ih
      obtain ⟨j, -, hj, hj'⟩ := failClass_reset _ _
      rw [hf, Option.some.injEq] at hj'
      subst hj'
      exact ⟨(childKey_spec hk (dose_le_299 hsv) hc).2.2, hb, resetClass_valid hj,
        fun _ => ⟨j, hj, rfl⟩⟩

/-- STL-REACH-2, worklist part: a keyed class pair is reachable exactly when
it is the seed or the image of a reachable pair at a key of strictly smaller
clock. This instantiates `processed_iff_reachable` with the key clock. -/
theorem creach_iff (limit : ℕ) (z : Key × QProfile × QProfile) :
    Relation.ReflTransGen (CStep limit) cRoot z ↔
      z = cRoot ∨ ∃ y, y.1.clock < z.1.clock ∧ Relation.ReflTransGen (CStep limit) cRoot y ∧
        CStep limit y z := by
  have hst : ∀ a b, CStep limit a b → a.1.clock < b.1.clock := fun _ _ h => cstep_clock h
  rw [← processed_iff_reachable (fun y : Key × QProfile × QProfile => y.1.clock) _ cRoot hst,
    processed_iff_unfold]
  simp only [processed_iff_reachable (fun y : Key × QProfile × QProfile => y.1.clock) _ cRoot hst]
  constructor
  · rintro (h | ⟨y, hy, h1, h2⟩)
    · exact Or.inl h
    · exact Or.inr ⟨y, hy, h1, h2⟩
  · rintro (h | ⟨y, hy, h1, h2⟩)
    · exact Or.inl h
    · exact Or.inr ⟨y, hy, h1, h2⟩

/-! ### The bitmap update -/

/-- The bitmap column of a Dropper class: the full class id, or for a `REV` key
the reset column `idx0` (`key_columns`, `leap_build.py:35-36`). -/
def colIndex : Key → QProfile → ℕ
  | .rev _, b => (idx0 (classIndex b)).toNat
  | _, b => classIndex b

/-- The bit of a class pair at a key. -/
def enc (k : Key) (z : QProfile × QProfile) : ℕ × ℕ := (classIndex z.1, colIndex k z.2)

/-- `rows` (`leap_build.py:131`): a `REV` parent writes row `j` of its
transposed array to full row `s0[j]`; other parents write row `j` to row `j`. -/
def rowMap : Key → ℕ → ℕ
  | .rev _, j => s0 j
  | _, j => j

/-- `_success_bits` (`leap_build.py:64-78`) on the set of set bits. Column by
column, each nonempty TTD block `t` expands its rows `R` to `expandedSet` and
places index `i` at `placeIndex t i`; the fatal block (`N_ALIVE..N`) places
index `i` at `N_ALIVE + i`. -/
def successBits (B : Set (ℕ × ℕ)) : Set (ℕ × ℕ) :=
  {p | (∃ t, ValidT t ∧ ∃ i ∈ expandedSet (blockLen t)
          {r | r < blockLen t ∧ (blockStart t + r, p.2) ∈ B}, p.1 = placeIndex t i) ∨
       (∃ i ∈ expandedSet 300 {r | r < 300 ∧ (16711 + r, p.2) ∈ B}, p.1 = 16711 + i)}

/-- The success merge `target(success)[rows] |= _transpose(_success_bits(bits))`
(`leap_build.py:132-134`): bit `(id, col)` becomes bit `(rowMap col, id)`. -/
def successUpdate (p : Key) (B : Set (ℕ × ℕ)) : Set (ℕ × ℕ) :=
  {q | ∃ id col, (id, col) ∈ successBits B ∧ q = (rowMap p col, id)}

/-- `fail[pc]` for the eligible row `id(s, t)` (`leap_profiles.py:66-67`):
`ids.get((0, s + t + 60), N_ALIVE)`. -/
def failId (s t : ℕ) : ℕ :=
  if Survives 0 (s + t + 60) then classIndex (.alive 0 (s + t + 60)) else 16711

/-- The revival merge into `dest` (`leap_build.py:136-154`). For each set row
`id(s, t) < N_ALIVE` whose dose `s + 60` leads to `dest`, row bit `col` is
merged into row `j = idx0[fail[pc]]`, transposed, and written to
`(rows[col], j)` for a `REV` destination or to the dense column
`(rows[col], s0[j])` otherwise. -/
def revivalUpdate (p dest : Key) (B : Set (ℕ × ℕ)) : Set (ℕ × ℕ) :=
  {q | ∃ s t col, ValidClass (.alive s t) ∧ (classIndex (.alive s t), col) ∈ B ∧
      childKey p (s + 60) = some dest ∧ ∃ j : ℕ, idx0 (failId s t) = j ∧
      q = (rowMap p col, if dest.isRev then j else s0 j)}

/-- Everything parent `p` ORs into key `k`. -/
def mergeUpdate (p k : Key) (B : Set (ℕ × ℕ)) : Set (ℕ × ℕ) :=
  {q | childKey p 0 = some k ∧ q ∈ successUpdate p B} ∪ revivalUpdate p k B

/-- The bitmap `pending[k]` when the loop pops `k` (`leap_build.py:116-128`):
the seed bit `(0, 0)` of `H1_12`, and the merges of every valid parent popped
earlier. The loop pops keys by increasing clock and every child key has a
larger clock (`clock_lt_of_childKey`), so the parents that write to `k` are
exactly the valid keys of smaller clock; the recursion states that bound. A
key or child beyond `limit` is not processed. -/
def pending (limit : ℕ) (k : Key) : Set (ℕ × ℕ) :=
  (if k = .h1 12 then {(0, 0)} else ∅) ∪
    {q | ∃ p : Key, ∃ _ : p.clock < k.clock, p.Valid ∧ k.clock ≤ limit ∧
      q ∈ mergeUpdate p k (pending limit p)}
termination_by k.clock

theorem pending_unfold (limit : ℕ) (k : Key) :
    pending limit k = (if k = .h1 12 then {(0, 0)} else ∅) ∪
      {q | ∃ p : Key, ∃ _ : p.clock < k.clock, p.Valid ∧ k.clock ≤ limit ∧
        q ∈ mergeUpdate p k (pending limit p)} := by
  rw [pending]

theorem rowMap_colIndex {k : Key} {b : QProfile} (h : k.isRev = true → ∃ j < 183, b = resetClass j) :
    rowMap k (colIndex k b) = classIndex b := by
  cases k with
  | rev c =>
    obtain ⟨j, hj, rfl⟩ := h rfl
    simp only [rowMap, colIndex, (resetClass_idx0 hj).1]; rfl
  | h1 _ => rfl
  | h2 _ => rfl

theorem colIndex_of_not_rev {k : Key} (h : k.isRev = false) (b : QProfile) :
    colIndex k b = classIndex b := by
  cases k <;> simp_all [colIndex, Key.isRev]

theorem colIndex_reset (k : Key) {j : ℕ} (hj : j < 183) :
    colIndex k (resetClass j) = if k.isRev then j else s0 j := by
  cases k <;> simp [colIndex, Key.isRev, (resetClass_idx0 hj).1, s0]

/-- A valid class is either an in-block alive class or a fatal marker, and
its id decodes it. -/
theorem eq_of_classIndex_alive {a : QProfile} {s t : ℕ} (ha : ValidClass a)
    (hv : ValidClass (.alive s t)) (h : classIndex a = classIndex (.alive s t)) :
    a = .alive s t := classIndex_injective ha hv h

/-- The success transform on an encoded set: bit `(id, col)` is set exactly
when some pair `(a, b)` of the set has column `col` and a success class of `a`
with id `id`. -/
theorem mem_successBits {C : Set (QProfile × QProfile)} (g : QProfile → ℕ)
    (hC : ∀ z ∈ C, ValidClass z.1) (id col : ℕ) :
    (id, col) ∈ successBits ((fun z => (classIndex z.1, g z.2)) '' C) ↔
      ∃ z ∈ C, g z.2 = col ∧ ∃ ℓ, 1 ≤ ℓ ∧ ℓ ≤ 60 ∧ (succClass z.1 ℓ).map classIndex = some id := by
  constructor
  · rintro (⟨t, ht, i, hi, hid⟩ | ⟨i, hi, hid⟩)
    · have hR : ∀ r ∈ {r | r < blockLen t ∧
          (blockStart t + r, col) ∈ (fun z => (classIndex z.1, g z.2)) '' C}, r < blockLen t :=
        fun r hr => hr.1
      obtain ⟨r, ⟨hrl, ⟨z, hz, hze⟩⟩, ℓ, h1, h2, hm⟩ :=
        (success_bits_block ht hR id).1 ⟨i, hi, hid⟩
      simp only [Prod.mk.injEq] at hze
      have hv : ValidClass (.alive r t) := (validClass_alive_iff r t).2 ⟨ht, hrl⟩
      have hz1 := eq_of_classIndex_alive (hC z hz) hv hze.1
      exact ⟨z, hz, hze.2, ℓ, h1, h2, by rw [hz1]; exact hm⟩
    · have hR : ∀ r ∈ {r | r < 300 ∧
          (16711 + r, col) ∈ (fun z => (classIndex z.1, g z.2)) '' C}, r < 300 :=
        fun r hr => hr.1
      obtain ⟨r, ⟨hrl, ⟨z, hz, hze⟩⟩, ℓ, h1, h2, hm⟩ :=
        (success_bits_fatal hR id).1 ⟨i, hi, hid⟩
      simp only [Prod.mk.injEq] at hze
      have hv : ValidClass (.dead r) := hrl
      have hz1 := classIndex_injective (hC z hz) hv hze.1
      exact ⟨z, hz, hze.2, ℓ, h1, h2, by rw [hz1]; exact hm⟩
  · rintro ⟨⟨a, b⟩, hz, rfl, ℓ, h1, h2, hm⟩
    have ha := hC _ hz
    cases a with
    | alive s t =>
      obtain ⟨ht, hs⟩ := (validClass_alive_iff s t).1 ha
      have hR : ∀ r ∈ {r | r < blockLen t ∧
          (blockStart t + r, g b) ∈ (fun z => (classIndex z.1, g z.2)) '' C}, r < blockLen t :=
        fun r hr => hr.1
      obtain ⟨i, hi, hid⟩ := (success_bits_block ht hR id).2
        ⟨s, ⟨hs, ⟨(.alive s t, b), hz, rfl⟩⟩, ℓ, h1, h2, hm⟩
      exact Or.inl ⟨t, ht, i, hi, hid⟩
    | dead s =>
      have hs : s < 300 := ha
      have hR : ∀ r ∈ {r | r < 300 ∧
          (16711 + r, g b) ∈ (fun z => (classIndex z.1, g z.2)) '' C}, r < 300 :=
        fun r hr => hr.1
      obtain ⟨i, hi, hid⟩ := (success_bits_fatal hR id).2
        ⟨s, ⟨hs, ⟨(.dead s, b), hz, rfl⟩⟩, ℓ, h1, h2, hm⟩
      exact Or.inr ⟨i, hi, hid⟩

/-- STL-REACH-2: the success merge of an encoded set is the encoded set of
success images, placed at a success key (never a `REV` key). -/
theorem successUpdate_enc {p k : Key} {C : Set (QProfile × QProfile)}
    (hC : ∀ z ∈ C, ReachInv p z.1 z.2) (hk : k.isRev = false) :
    successUpdate p (enc p '' C) =
      enc k '' {z | ∃ a, (a, z.1) ∈ C ∧ ∃ ℓ, 1 ≤ ℓ ∧ ℓ ≤ 60 ∧ succClass a ℓ = some z.2} := by
  have hC1 : ∀ z ∈ C, ValidClass z.1 := fun z hz => (hC z hz).2.1
  ext ⟨r, c⟩
  simp only [successUpdate, Set.mem_ofPred_eq, Set.mem_image]
  constructor
  · rintro ⟨id, col, hmem, he⟩
    obtain ⟨⟨a, b⟩, hz, rfl, ℓ, h1, h2, hm⟩ := (mem_successBits (colIndex p) hC1 id col).1 hmem
    obtain ⟨b', hb', rfl⟩ := Option.map_eq_some_iff.1 hm
    refine ⟨(b, b'), ⟨a, hz, ℓ, h1, h2, hb'⟩, ?_⟩
    rw [he, rowMap_colIndex (hC _ hz).2.2.2]
    simp only [enc, colIndex_of_not_rev hk]
  · rintro ⟨⟨b, b'⟩, ⟨a, hz, ℓ, h1, h2, hb'⟩, he⟩
    refine ⟨classIndex b', colIndex p b, (mem_successBits (colIndex p) hC1 _ _).2
      ⟨(a, b), hz, rfl, ℓ, h1, h2, by rw [hb']; rfl⟩, ?_⟩
    rw [← he, rowMap_colIndex (hC _ hz).2.2.2]
    simp only [enc, colIndex_of_not_rev hk]

/-- STL-REACH-2: the revival merge of an encoded set into `dest` is the encoded
set of revival images: the old Dropper becomes the Checker, and `fail[pc]`,
a reset class, becomes the Dropper. -/
theorem revivalUpdate_enc {p dest : Key} {C : Set (QProfile × QProfile)}
    (hC : ∀ z ∈ C, ReachInv p z.1 z.2) :
    revivalUpdate p dest (enc p '' C) =
      enc dest '' {z | ∃ s t, (QProfile.alive s t, z.1) ∈ C ∧ Survives s t ∧
        failClass (.alive s t) = some z.2 ∧ childKey p (s + 60) = some dest} := by
  ext ⟨r, c⟩
  simp only [revivalUpdate, Set.mem_ofPred_eq, Set.mem_image]
  constructor
  · rintro ⟨s, t, col, hv, ⟨⟨a, b⟩, hz, he⟩, hcd, j, hj, hq⟩
    simp only [enc, Prod.mk.injEq] at he
    have ha := eq_of_classIndex_alive (hC _ hz).2.1 hv he.1
    subst ha
    obtain ⟨j', -, hj', hf⟩ := failClass_reset s t
    have hfid : failId s t = s0 j' := by
      simp only [failClass, Option.some.injEq] at hf
      simp only [failId, s0, ← hf]; split_ifs <;> rfl
    have hjj : j = j' := by
      rw [hfid, idx0_s0 j' hj'] at hj; exact_mod_cast hj.symm
    subst hjj
    refine ⟨(b, resetClass j), ⟨s, t, hz, hv.2, hf, hcd⟩, ?_⟩
    rw [hq, ← he.2, rowMap_colIndex (hC _ hz).2.2.2]
    simp only [enc, colIndex_reset dest hj']
  · rintro ⟨⟨b, f⟩, ⟨s, t, hz, hsv, hf, hcd⟩, he⟩
    obtain ⟨j, -, hj, hf'⟩ := failClass_reset s t
    rw [hf, Option.some.injEq] at hf'
    subst hf'
    have hfid : failId s t = s0 j := by
      obtain ⟨j2, -, hj2, hf2⟩ := failClass_reset s t
      rw [hf, Option.some.injEq] at hf2
      have := resetClass_injective hj hj2 hf2
      subst this
      simp only [failClass, Option.some.injEq] at hf
      simp only [failId, s0, ← hf]; split_ifs <;> rfl
    refine ⟨s, t, colIndex p b, (hC _ hz).2.1, ⟨(.alive s t, b), hz, rfl⟩, hcd, j,
      by rw [hfid, idx0_s0 j hj], ?_⟩
    rw [← he, rowMap_colIndex (hC _ hz).2.2.2]
    simp only [enc, colIndex_reset dest hj]

/-- The class-level step, spelled out by case. -/
theorem cstep_iff {limit : ℕ} {p k : Key} {a b b' : QProfile} :
    CStep limit (p, a, b) (k, b, b') ↔ p.Valid ∧ k.clock ≤ limit ∧
      ((childKey p 0 = some k ∧ ∃ ℓ, 1 ≤ ℓ ∧ ℓ ≤ 60 ∧ succClass a ℓ = some b') ∨
        ∃ s t, a = .alive s t ∧ Survives s t ∧ failClass (.alive s t) = some b' ∧
          childKey p (s + 60) = some k) := by
  constructor
  · intro h
    cases h with
    | succ ℓ hk h1 h60 hsc hc hl => exact ⟨hk, hl, Or.inl ⟨hc, ℓ, h1, h60, hsc⟩⟩
    | rev hk hsv hf hc hl => exact ⟨hk, hl, Or.inr ⟨_, _, rfl, hsv, hf, hc⟩⟩
  · rintro ⟨hp, hl, ⟨hc, ℓ, h1, h60, hsc⟩ | ⟨s, t, rfl, hsv, hf, hc⟩⟩
    · exact CStep.succ ℓ hp h1 h60 hsc hc hl
    · exact CStep.rev hp hsv hf hc hl

theorem cstep_mid {limit : ℕ} {y z : Key × QProfile × QProfile} (h : CStep limit y z) :
    z.2.1 = y.2.2 := by
  cases h <;> rfl

/-- STL-REACH-2: everything a valid parent merges into `k` is the encoded set
of class pairs one `CStep` away from the parent's set. -/
theorem mergeUpdate_enc {limit : ℕ} {p k : Key} (hp : p.Valid) (hl : k.clock ≤ limit)
    {C : Set (QProfile × QProfile)} (hC : ∀ z ∈ C, ReachInv p z.1 z.2) :
    mergeUpdate p k (enc p '' C) =
      enc k '' {z | ∃ a, (a, z.1) ∈ C ∧ CStep limit (p, a, z.1) (k, z.1, z.2)} := by
  ext q
  simp only [mergeUpdate, Set.mem_union, Set.mem_ofPred_eq]
  constructor
  · rintro (⟨hc, hq⟩ | hq)
    · rw [successUpdate_enc hC (childKey_zero_isRev hc)] at hq
      obtain ⟨z, ⟨a, hz, ℓ, h1, h2, hs⟩, rfl⟩ := hq
      exact ⟨z, ⟨a, hz, cstep_iff.2 ⟨hp, hl, Or.inl ⟨hc, ℓ, h1, h2, hs⟩⟩⟩, rfl⟩
    · rw [revivalUpdate_enc hC] at hq
      obtain ⟨z, ⟨s, t, hz, hsv, hf, hcd⟩, rfl⟩ := hq
      exact ⟨z, ⟨_, hz, cstep_iff.2 ⟨hp, hl, Or.inr ⟨s, t, rfl, hsv, hf, hcd⟩⟩⟩, rfl⟩
  · rintro ⟨z, ⟨a, hz, hs⟩, rfl⟩
    obtain ⟨-, -, ⟨hc, ℓ, h1, h2, hsc⟩ | ⟨s, t, rfl, hsv, hf, hcd⟩⟩ := cstep_iff.1 hs
    · refine Or.inl ⟨hc, ?_⟩
      rw [successUpdate_enc hC (childKey_zero_isRev hc)]
      exact ⟨z, ⟨a, hz, ℓ, h1, h2, hsc⟩, rfl⟩
    · refine Or.inr ?_
      rw [revivalUpdate_enc hC]
      exact ⟨z, ⟨s, t, hz, hsv, hf, hcd⟩, rfl⟩

/-- STL-REACH-2 (`leap_build.py:102-159`, `paper/stl.tex:94-96,152`): for every
valid key at or below `limit`, the bitmap that `forward_reachability` holds
when it pops the key is the encoded set of class pairs reachable at that key
from the seed `(H1_12, (0,0), (0,0))`. -/
theorem pending_eq (limit : ℕ) (k : Key) (hk : k.Valid) (hkl : k.clock ≤ limit) :
    pending limit k = enc k '' {z | Relation.ReflTransGen (CStep limit) cRoot (k, z.1, z.2)} := by
  induction hn : k.clock using Nat.strong_induction_on generalizing k with
  | _ n ih =>
  subst hn
  have hIH : ∀ p : Key, p.clock < k.clock → p.Valid →
      pending limit p = enc p '' {z | Relation.ReflTransGen (CStep limit) cRoot (p, z.1, z.2)} :=
    fun p hp hpv => ih _ hp p hpv (by omega) rfl
  ext q
  rw [pending_unfold]
  simp only [Set.mem_union, Set.mem_ofPred_eq, Set.mem_image]
  constructor
  · rintro (hq | ⟨p, hp, hpv, -, hq⟩)
    · split_ifs at hq with hk12
      · subst hk12
        rw [Set.mem_singleton_iff] at hq
        subst hq
        refine ⟨(.alive 0 0, .alive 0 0), .refl, ?_⟩
        have : Survives 0 0 := by unfold Survives; norm_num
        simp [enc, colIndex, classIndex, blockStart]
      · simp at hq
    · rw [hIH p hp hpv, mergeUpdate_enc (limit := limit)
        (C := {z | Relation.ReflTransGen (CStep limit) cRoot (p, z.1, z.2)}) hpv hkl
        (fun z hz => reach_inv hz)] at hq
      obtain ⟨z, ⟨a, hz, hs⟩, rfl⟩ := hq
      exact ⟨z, hz.tail hs, rfl⟩
  · rintro ⟨z, hz, rfl⟩
    rcases (creach_iff limit _).1 hz with h0 | ⟨⟨p, a, b⟩, hy, hr, hs⟩
    · simp only [cRoot, Prod.mk.injEq] at h0
      obtain ⟨rfl, h1, h2⟩ := h0
      left
      simp only [↓reduceIte, Set.mem_singleton_iff, enc, h1, h2, colIndex, classIndex, blockStart]
    · have hb : z.1 = b := cstep_mid hs
      obtain ⟨z1, z2⟩ := z
      simp only at hb hs hy
      subst hb
      have hpv : p.Valid := (cstep_iff.1 hs).1
      right
      refine ⟨p, hy, hpv, hkl, ?_⟩
      rw [hIH p hy hpv, mergeUpdate_enc (limit := limit)
        (C := {z | Relation.ReflTransGen (CStep limit) cRoot (p, z.1, z.2)}) hpv hkl
        (fun z hz => reach_inv hz)]
      exact ⟨(z1, z2), ⟨a, hr, hs⟩, rfl⟩

theorem enc_injOn (k : Key) {C : Set (QProfile × QProfile)}
    (hC : ∀ z ∈ C, ReachInv k z.1 z.2) : Set.InjOn (enc k) C := by
  rintro ⟨a, b⟩ hz ⟨a', b'⟩ hz' h
  simp only [enc, Prod.mk.injEq] at h
  have ha : a = a' := classIndex_injective (hC _ hz).2.1 (hC _ hz').2.1 h.1
  have hb : b = b' := by
    have e1 := rowMap_colIndex (hC _ hz).2.2.2
    have e2 := rowMap_colIndex (hC _ hz').2.2.2
    rw [h.2] at e1
    exact classIndex_injective (hC _ hz).2.2.1 (hC _ hz').2.2.1 (e1.symm.trans e2)
  rw [ha, hb]

/-- STL-REACH-2: `counts[key]`, the popcount of the key's bitmap, is the number
of reachable class pairs at the key, each counted once. -/
theorem counts_eq (limit : ℕ) (k : Key) (hk : k.Valid) (hkl : k.clock ≤ limit) :
    (pending limit k).ncard =
      {z : QProfile × QProfile | Relation.ReflTransGen (CStep limit) cRoot (k, z.1, z.2)}.ncard := by
  rw [pending_eq limit k hk hkl]
  exact (enc_injOn k fun z hz => reach_inv hz).ncard_image

/-- STL-REACH-2, end to end: the bitmap of a valid key at or below `limit`
encodes exactly the quotient pairs `(quot x.c, quot x.d)` of the L2 states `x`
reachable at that key from the opening under `VStep`, and `counts[key]` is
their number. -/
theorem pending_eq_states (limit : ℕ) (k : Key) (hk : k.Valid) (hkl : k.clock ≤ limit) :
    pending limit k = enc k '' {z | ∃ x : State,
      Relation.ReflTransGen (VStep limit) (.h1 12, rootState) (k, x) ∧
        (quot x.c, quot x.d) = z} ∧
    (pending limit k).ncard = {z : QProfile × QProfile | ∃ x : State,
      Relation.ReflTransGen (VStep limit) (.h1 12, rootState) (k, x) ∧
        (quot x.c, quot x.d) = z}.ncard := by
  have hset : {z : QProfile × QProfile | Relation.ReflTransGen (CStep limit) cRoot (k, z.1, z.2)} =
      {z | ∃ x : State, Relation.ReflTransGen (VStep limit) (.h1 12, rootState) (k, x) ∧
        (quot x.c, quot x.d) = z} := by
    ext ⟨a, b⟩
    simp only [Set.mem_ofPred_eq, creach_iff_vreach, Prod.mk.injEq]
    constructor
    · rintro ⟨⟨k', x⟩, hr, he⟩
      simp only [classOf, Prod.mk.injEq] at he
      obtain ⟨rfl, h1, h2⟩ := he
      exact ⟨x, hr, h1, h2⟩
    · rintro ⟨x, hr, h1, h2⟩
      exact ⟨(k, x), hr, by simp [classOf, h1, h2]⟩
  rw [← hset]
  exact ⟨pending_eq limit k hk hkl, counts_eq limit k hk hkl⟩

/-- STL-RESUME-1, membership gate (`leap_resume.py:63-86`): a prefix table
whose finite cells equal the reachability bitmap holds a value exactly at the
reachable class pairs of its key. -/
theorem prefix_membership (limit : ℕ) (k : Key) (hk : k.Valid) (hkl : k.clock ≤ limit)
    (finiteCells : Set (ℕ × ℕ)) (hgate : finiteCells = pending limit k) (z : QProfile × QProfile)
    (hz : ReachInv k z.1 z.2) :
    enc k z ∈ finiteCells ↔ Relation.ReflTransGen (CStep limit) cRoot (k, z.1, z.2) := by
  rw [hgate, pending_eq limit k hk hkl]
  constructor
  · rintro ⟨z', hz', he⟩
    have := enc_injOn k (C := {z, z'}) (by
      rintro w (rfl | rfl)
      · exact hz
      · exact reach_inv hz') (by simp) (by simp) he.symm
    rw [this]; exact hz'
  · intro h; exact ⟨z, h, rfl⟩

example : childKey (.h1 12) 0 = some (.h2 12) ∧ (Key.h2 12).clock ≤ 3600 := by decide

/-- Non-vacuity: the hypotheses of `pending_eq` hold at the seed key, whose
bitmap holds the seed bit. -/
example : (Key.h1 12).Valid ∧ (Key.h1 12).clock ≤ 3600 ∧
    ((0, 0) : ℕ × ℕ) ∈ pending 3600 (.h1 12) := by
  refine ⟨⟨le_rfl, by norm_num⟩, by simp [Key.clock], ?_⟩
  rw [pending_unfold]; simp

/-! ## Packed tables: rank and select (STL-PACK-1) -/

/-- The set bits below `n`, in increasing order (row-major, `WIN` excluded). -/
def setBits (b : ℕ → Bool) (n : ℕ) : List ℕ := (List.range n).filter b

/-- The number of set bits below `i`. -/
def rank (b : ℕ → Bool) (i : ℕ) : ℕ := ((List.range i).filter b).length

/-- `values` of `TableStore.pack`: the dense values at set bits in increasing
bit order. -/
def packedValues (b : ℕ → Bool) (v : ℕ → ℝ) (n : ℕ) : List ℝ := (setBits b n).map v

theorem rank_add (b : ℕ → Bool) (m k : ℕ) :
    rank b (m + k) = rank b m + ((List.range k).filter (fun j => b (m + j))).length := by
  unfold rank
  rw [List.range_add, List.filter_append, List.length_append, List.filter_map, List.length_map]
  rfl

theorem setBits_rank {b : ℕ → Bool} {i n : ℕ} (hi : i < n) (hb : b i = true) :
    (setBits b n)[rank b i]? = some i := by
  obtain ⟨k, rfl⟩ : ∃ k, n = i + (k + 1) := ⟨n - i - 1, by omega⟩
  unfold setBits rank
  rw [List.range_add, List.filter_append, List.range_succ_eq_map]
  simp [hb]

/-- STL-PACK-1 (`leap_audit.py:33-58`): the packed value at the rank of a set
bit is the dense value of that bit. -/
theorem packed_get {b : ℕ → Bool} {v : ℕ → ℝ} {i n : ℕ} (hi : i < n) (hb : b i = true) :
    (packedValues b v n)[rank b i]? = some (v i) := by
  simp [packedValues, List.getElem?_map, setBits_rank hi hb]

/-- The rank strictly increases across a set bit. -/
theorem rank_lt {b : ℕ → Bool} {i j : ℕ} (hij : i < j) (hb : b i = true) : rank b i < rank b j := by
  obtain ⟨k, rfl⟩ : ∃ k, j = i + k := ⟨j - i, by omega⟩
  rw [rank_add]
  have : 0 ∈ (List.range k).filter (fun j => b (i + j)) :=
    List.mem_filter.2 ⟨List.mem_range.2 (by omega), by simpa using hb⟩
  have := List.length_pos_of_mem this
  omega

/-- STL-PACK-1: `sample` inverts the rank: the `t`-th set bit has rank `t`. -/
theorem rank_select {b : ℕ → Bool} {n t : ℕ} (ht : t < (setBits b n).length) :
    rank b (setBits b n)[t] = t := by
  have hmem := List.getElem_mem ht
  simp only [setBits, List.mem_filter, List.mem_range] at hmem
  have h := setBits_rank hmem.1 hmem.2
  rw [List.getElem?_eq_some_iff] at h
  obtain ⟨hlt, heq⟩ := h
  exact ((List.nodup_range.filter b).getElem_inj_iff).1 heq

/-- The number of set bits in byte `B` (bits `8 B .. 8 B + 7`, little endian). -/
def byteCount (b : ℕ → Bool) (B : ℕ) : ℕ := ((List.range 8).filter (fun j => b (8 * B + j))).length

theorem rank_eight_mul (b : ℕ → Bool) (B : ℕ) : rank b (8 * B) = ∑ B' ∈ Finset.range B, byteCount b B' := by
  induction B with
  | zero => simp [rank]
  | succ B ih => rw [show 8 * (B + 1) = 8 * B + 8 by ring, rank_add, ih, Finset.sum_range_succ]; rfl

theorem sum_range_blocks (f : ℕ → ℕ) (m q r : ℕ) :
    ∑ B ∈ Finset.range (m * q + r), f B =
      ∑ blk ∈ Finset.range q, ∑ j ∈ Finset.range m, f (m * blk + j) +
        ∑ j ∈ Finset.range r, f (m * q + j) := by
  rw [Finset.sum_range_add]
  congr 1
  induction q with
  | zero => simp
  | succ q ih => rw [show m * (q + 1) = m * q + m by ring, Finset.sum_range_add, ih,
      Finset.sum_range_succ]

/-- STL-PACK-1 (`PackedReader.get`, `leap_audit.py:33-58`): the rank of bit `i`
is the prefix over whole 4096-byte blocks, plus the in-block byte prefix, plus
the set bits of byte `i / 8` below bit `i % 8`. -/
theorem rank_eq_reader (b : ℕ → Bool) (i : ℕ) :
    rank b i = ∑ blk ∈ Finset.range (i / 8 / 4096), ∑ j ∈ Finset.range 4096,
        byteCount b (4096 * blk + j) +
      ∑ j ∈ Finset.range (i / 8 % 4096), byteCount b (4096 * (i / 8 / 4096) + j) +
      ((List.range (i % 8)).filter (fun j => b (8 * (i / 8) + j))).length := by
  conv_lhs => rw [← Nat.div_add_mod i 8, rank_add, rank_eight_mul]
  rw [← sum_range_blocks, Nat.div_add_mod]

/-- The numpy mask `byte & ((1 << bit) - 1)` keeps exactly the bits below `bit`. -/
theorem testBit_lowMask (x k j : ℕ) :
    Nat.testBit (x &&& (2 ^ k - 1)) j = (x.testBit j && decide (j < k)) := by
  rw [Nat.testBit_and, Nat.testBit_two_pow_sub_one]

/-- STL-PACK-1 (`TableStore.pack`, `leap_build.py:229-235`): a 64-row chunk
starts on a byte boundary, so bit `j` of the chunk is bit `j % 8` of byte
`lo cols / 8 + j / 8`, and chunked packing equals global packing. -/
theorem chunk_aligned (k cols j : ℕ) :
    (64 * k * cols + j) / 8 = 64 * k * cols / 8 + j / 8 ∧ (64 * k * cols + j) % 8 = j % 8 := by
  rw [show 64 * k * cols = 8 * (8 * (k * cols)) by ring]
  omega

/-- STL-PACK-1: the row-major flat index `r · cols + c` with `c < cols`
determines `(r, c)`. -/
theorem flat_index_inj {cols r c r' c' : ℕ} (hc : c < cols) (hc' : c' < cols)
    (h : r * cols + c = r' * cols + c') : r = r' ∧ c = c' := by
  have h1 : (r * cols + c) / cols = r := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ (by omega), Nat.div_eq_of_lt hc, zero_add]
  have h2 : (r' * cols + c') / cols = r' := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ (by omega), Nat.div_eq_of_lt hc', zero_add]
  have hr : r = r' := by rw [← h1, ← h2, h]
  subst hr
  exact ⟨rfl, by omega⟩

/-- `PackedReader.get`: `-1` for the implicit `WIN` column, else the packed
value at the rank of the flat index. -/
def readerGet (b : ℕ → Bool) (v : ℕ → ℝ) (n cols r c : ℕ) : Option ℝ :=
  if c = cols then some (-1) else (packedValues b v n)[rank b (r * cols + c)]?

/-- STL-PACK-1: the reader returns the dense table value of every reachable
cell and `-1` in the `WIN` column. -/
theorem readerGet_spec {b : ℕ → Bool} {v : ℕ → ℝ} {n cols r c : ℕ} :
    readerGet b v n cols r cols = some (-1) ∧
    (c < cols → r * cols + c < n → b (r * cols + c) = true →
      readerGet b v n cols r c = some (v (r * cols + c))) := by
  refine ⟨by simp [readerGet], fun hc hn hb => ?_⟩
  unfold readerGet
  simp only [show c ≠ cols by omega, ↓reduceIte]
  exact packed_get hn hb

/-! ## Revival-cache exactness (STL-CACHE-1) -/

/-- The stage inputs of `_solve_chunk` (`leap_lp.py:116-129`): success payoffs
`-T_succ[pd, succ[pc, k]]` and `F = rev · (-T_fail[pd, col(pc)]) + (1 - rev)`,
where the tables and the failure-column map depend only on the child keys. -/
def stageInputs (tables : Option Key → ℕ → ℕ → ℝ) (failCol : Option Key → ℕ → ℕ)
    (succ : ℕ → ℕ → ℕ) (rev : ℕ → ℝ) (succKey failKey : Option Key) (pc pd : ℕ) :
    (ℕ → ℝ) × ℝ :=
  (fun k => -tables succKey pd (succ pc k),
    rev pc * -tables failKey pd (failCol failKey pc) + (1 - rev pc))

/-- STL-CACHE-1 (`RevivalCache`, `leap_lp.py:69-103,172-179`): a cache hit
returns a value certified for an identical stage. The cache groups by success
key and window, tags each `(pc, idx0[pd])` cell with an injective token of the
failure key, and caches only reset columns `0 ≤ idx0[pd] < 183`; `idx0`
identifies `pd` on those columns (`eq_s0_of_idx0`). -/
theorem cache_hit_exact (tables : Option Key → ℕ → ℕ → ℝ) (failCol : Option Key → ℕ → ℕ)
    (succ : ℕ → ℕ → ℕ) (rev : ℕ → ℝ) (tok : Option Key → ℕ) (htok : Function.Injective tok)
    {succKey failKey failKey' : Option Key} {pc pd pd' : ℕ} {j : ℤ}
    (hj0 : 0 ≤ j) (hj : j < 183) (hpd : idx0 pd = j) (hpd' : idx0 pd' = j)
    (htoken : tok failKey = tok failKey') :
    stageInputs tables failCol succ rev succKey failKey pc pd =
      stageInputs tables failCol succ rev succKey failKey' pc pd' := by
  rw [htok htoken, eq_s0_of_idx0 hpd hj0 hj, eq_s0_of_idx0 hpd' hj0 hj]

/-! ## Sampled Bellman audit (STL-AUDIT-1) -/

/-- STL-AUDIT-1 (`leap_audit.py:166-204`): a stored value within `1e-6` of the
midpoint of a certificate whose saddle gap is at most `1e-6` lies within
`1.5e-6` of the stage value. -/
theorem audit_gate {m n : Type*} [Fintype m] [Fintype n] [Nonempty m] [Nonempty n]
    (M : Matrix m n ℝ) {p : m → ℝ} {q : n → ℝ} (hp : p ∈ simplex m) (hq : q ∈ simplex n)
    {stored : ℝ} (hgap : upperBound M q - lowerBound M p ≤ 1e-6)
    (hres : |stored - (lowerBound M p + upperBound M q) / 2| ≤ 1e-6) :
    |stored - value M| ≤ 1.5e-6 := by
  have h := abs_midpoint_sub_value_le M hp hq hgap
  calc |stored - value M|
      ≤ |stored - (lowerBound M p + upperBound M q) / 2| +
        |(lowerBound M p + upperBound M q) / 2 - value M| := abs_sub_le _ _ _
    _ ≤ 1e-6 + 1e-6 / 2 := add_le_add hres h
    _ = 1.5e-6 := by norm_num

/-- STL-AUDIT-1 doubt: with 2,243 keys, 200 samples for `H2_57`, and
`min(count, 2)` for each other key, the total `4,621` holds exactly when 63
keys hold a single state; the maximum is `4,684`. -/
theorem audit_sample_count (n1 : ℕ) (h : n1 ≤ 2242) :
    200 + (2 * (2242 - n1) + n1) = 4621 ↔ n1 = 63 := by omega

/-! ## Restricted source changes (STL-RESUME-1) -/

/-- A solver after the accepted change: the old function, followed by a retry
only where the old function raised. -/
def withRetry {α ε β : Type*} (old retry : α → Except ε β) (a : α) : Except ε β :=
  match old a with
  | .ok v => .ok v
  | .error _ => retry a

/-- STL-RESUME-1 (`leap_resume.py:15-38`, `leap_recover.py:40-66`): every input
on which the old solver returned a value gets the same value from the new
solver, so an accepted prefix stays valid. -/
theorem withRetry_of_ok {α ε β : Type*} (old retry : α → Except ε β) {a : α} {v : β}
    (h : old a = .ok v) : withRetry old retry a = .ok v := by
  simp [withRetry, h]


/-- A top-level statement of the solver body, over an environment `σ` and a
return type `β`. Every statement may read the `linprog` method string. `step`
is an ordinary statement, `ret` a return, `raiseIf c` the statement
`if c: raise RuntimeError(...)`, and `retryIf c` the rewritten statement
`if c: (if _method == 'highs': return retry(...)); raise RuntimeError(...)`. -/
inductive Stmt (σ β : Type*)
  | step (f : String → σ → σ)
  | ret (f : String → σ → β)
  | raiseIf (c : String → σ → Bool)
  | retryIf (c : String → σ → Bool)

/-- The outcome of a run: a return, a raise with the environment at the raise,
or falling off the end. -/
inductive Run (σ β : Type*)
  | ret (v : β)
  | raise (s : σ)
  | done

/-- Execution with method `m`; `retry` is the call
`solve_lp(<arguments>, _method='highs-ipm')` evaluated in the environment at
the raise. -/
def exec {σ β : Type*} (retry : σ → β) (m : String) : List (Stmt σ β) → σ → Run σ β
  | [], _ => .done
  | .step f :: P, s => exec retry m P (f m s)
  | .ret f :: _, s => .ret (f m s)
  | .raiseIf c :: P, s => if c m s then .raise s else exec retry m P s
  | .retryIf c :: P, s =>
    if c m s then (if m = "highs" then .ret (retry s) else .raise s) else exec retry m P s

/-- The rewrite that `verify_retry_change` (`leap_resume.py:15-38`) demands: each
top-level `if ...: raise RuntimeError` gains the retry. The literal
`method='highs'` becomes `method=_method` with default `'highs'`, so a call
with the default runs every statement with the same method string. -/
def addRetry {σ β : Type*} : Stmt σ β → Stmt σ β
  | .raiseIf c => .retryIf c
  | s => s

/-- The body before the change has no retry statement. -/
def NoRetry {σ β : Type*} : List (Stmt σ β) → Prop
  | [] => True
  | .retryIf _ :: _ => False
  | _ :: P => NoRetry P

/-- STL-RESUME-1 (`leap_resume.py:15-38`, `README.md` resume paragraph): the
rewritten body, called with the default method `'highs'`, returns what the old
body returns, falls off the end where the old body does, and calls the retry
exactly where the old body raised. -/
theorem exec_addRetry {σ β : Type*} (retry : σ → β) :
    ∀ (P : List (Stmt σ β)), NoRetry P → ∀ s : σ,
      exec retry "highs" (P.map addRetry) s =
        match exec retry "highs" P s with
        | .raise s' => .ret (retry s')
        | r => r := by
  intro P
  induction P with
  | nil => intro _ s; rfl
  | cons st P ih =>
    intro hP s
    cases st with
    | step f => exact ih hP _
    | ret f => rfl
    | raiseIf c =>
      simp only [List.map_cons, addRetry, exec, ↓reduceIte]
      by_cases hc : c "highs" s = true
      · simp [hc]
      · simp only [hc, Bool.false_eq_true, ↓reduceIte]; exact ih hP s
    | retryIf c => exact hP.elim

/-- STL-RESUME-1: every input on which the old body returned a value gets the
same value from the rewritten body, so a certified prefix built by the old
source keeps its values. -/
theorem exec_addRetry_of_ret {σ β : Type*} (retry : σ → β) {P : List (Stmt σ β)}
    (hP : NoRetry P) {s : σ} {v : β} (h : exec retry "highs" P s = .ret v) :
    exec retry "highs" (P.map addRetry) s = .ret v := by
  rw [exec_addRetry retry P hP s, h]

/-! ## The canonical trace (STL-TRACE-1) -/

/-- A half-round outcome of the canonical trace: a success with its lag, or a
failed check followed by a revival. -/
inductive Outcome
  | success (lag : ℕ)
  | revive
  deriving DecidableEq

/-- A trace state: `(ST, TTD)` for Baku and Hal, the half, and the clock. -/
structure TraceState where
  baku : ℕ × ℕ
  hal : ℕ × ℕ
  half : ℕ
  clock : ℕ
  deriving DecidableEq

/-- One half-round of the trace under the frozen rules: Baku checks in half `1`,
Hal in half `2`. A revival requires eligibility. -/
def traceStep (g : TraceState) : Outcome → Option TraceState
  | .success ℓ =>
    let c := if g.half = 1 then g.baku else g.hal
    if 1 ≤ ℓ ∧ ℓ ≤ 60 ∧ c.1 + ℓ < 300 then
      let c' := (c.1 + ℓ, c.2)
      some { baku := if g.half = 1 then c' else g.baku, hal := if g.half = 1 then g.hal else c',
             half := otherHalf g.half, clock := childClock g.half g.clock 0 }
    else none
  | .revive =>
    let c := if g.half = 1 then g.baku else g.hal
    if Survives c.1 c.2 then
      let c' := (0, c.2 + (c.1 + 60))
      some { baku := if g.half = 1 then c' else g.baku, hal := if g.half = 1 then g.hal else c',
             half := otherHalf g.half, clock := childClock g.half g.clock (c.1 + 60) }
    else none

/-- The states at the start of each half-round, and the final state. -/
def replay : TraceState → List Outcome → Option (List TraceState)
  | g, [] => some [g]
  | g, o :: os => (traceStep g o).bind fun g' => (replay g' os).map (g :: ·)

/-- The eighteen canonical half-round outcomes (`docs/game-sources/
IN_DEPTH_SUMMARY.md`, Round 1 Turn 1 to Round 9 Turn 2). -/
def canonicalTrace : List Outcome :=
  [.revive, .success 24, .success 25, .revive, .success 4, .success 36, .success 3, .success 34,
   .success 1, .success 15, .revive, .success 8, .success 60, .success 1, .success 55, .revive,
   .success 60, .revive]

/-- The opening trace state at 8:12. -/
def traceRoot : TraceState := ⟨(0, 0), (0, 0), 1, 720⟩

/-- The expected states: the summary's `Accumulation/Near-death` headers (with
the Round 5 Turn 2 erratum of `EVIDENCE.md` corrected to `1M0S`, and the
missing Round 2 Turn 1 header filled in), and the exact clocks. -/
def canonicalStates : List TraceState :=
  [⟨(0, 0), (0, 0), 1, 720⟩, ⟨(0, 60), (0, 0), 2, 1020⟩, ⟨(0, 60), (24, 0), 1, 1140⟩,
   ⟨(25, 60), (24, 0), 2, 1260⟩, ⟨(25, 60), (0, 84), 1, 1560⟩, ⟨(29, 60), (0, 84), 2, 1680⟩,
   ⟨(29, 60), (36, 84), 1, 1800⟩, ⟨(32, 60), (36, 84), 2, 1920⟩, ⟨(32, 60), (70, 84), 1, 2040⟩,
   ⟨(33, 60), (70, 84), 2, 2160⟩, ⟨(33, 60), (85, 84), 1, 2280⟩, ⟨(0, 153), (85, 84), 2, 2613⟩,
   ⟨(0, 153), (93, 84), 1, 2700⟩, ⟨(60, 153), (93, 84), 2, 2820⟩, ⟨(60, 153), (94, 84), 1, 2940⟩,
   ⟨(115, 153), (94, 84), 2, 3060⟩, ⟨(115, 153), (0, 238), 1, 3420⟩,
   ⟨(175, 153), (0, 238), 2, 3540⟩, ⟨(175, 153), (0, 298), 1, 3841⟩]

/-- STL-TRACE-1 (`src/stl/docs/GAME_AND_SOLVER.md:257-261,276-289,377-379,458-460`): the
canonical trace replays under the frozen rules and the exact clock. It has
eighteen half-rounds and five revivals; its round starts are minutes
`12, 19, 26, 30, 34, 38, 45, 49, 57` with route classes
`V1, V4, V3, V3, V3, V3, V2, V2, V2`; Round 9 Turn 2 starts at `3540`, where Baku
may drop at second 61; and Hal's final TTD is `298`, two seconds below the
cap. The memory labels are cognitive annotations, not physics, and are not
modeled. -/
theorem canonical_trace_replay :
    replay traceRoot canonicalTrace = some canonicalStates ∧ canonicalTrace.length = 18 ∧
    canonicalTrace.count .revive = 5 ∧
    ((canonicalStates.filter (fun g => g.half = 1)).map (fun g => g.clock / 60)).take 9 =
      [12, 19, 26, 30, 34, 38, 45, 49, 57] ∧
    ([12, 19, 26, 30, 34, 38, 45, 49, 57] : List ℤ).map lsrVariation = [1, 4, 3, 3, 3, 3, 2, 2, 2] ∧
    LeapDropAvailable 2 3540 ∧ 300 - (238 + 60) = 2 := by
  refine ⟨by decide +kernel, by decide, by decide, by decide, by decide, by decide, by decide⟩

end Formal.STL
