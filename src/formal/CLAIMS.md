# Claims ledger

This file maps the core DTH and STL claims, and the extended library, to the
Lean declarations that prove them. `lake -d src/formal build` checks every declaration and fails if
one depends on `sorry` or on an axiom other than `propext`,
`Classical.choice`, and `Quot.sound`.

## Pure DTH (`paper/dth_exact_solution.tex`)

| Claim | Source | Lean |
| --- | --- | --- |
| Revival eligibility `q < 300 ∧ t + q ≤ 300` equals `s ≤ 239 ∧ s + t ≤ 240` | §1, `AGENTS.md` | `DTH.survives_iff` |
| `P_rev > 0` exactly on the eligible region, and `0 ≤ P_rev ≤ 0.95` | §1, `docs/REVIVAL_MODEL.md` | `DTH.revival_pos_iff`, `DTH.revival_nonneg`, `DTH.revival_le` |
| A failure-fatal profile stays failure-fatal after a success | §2 | `DTH.not_survives_of_le` |
| Quotient: equivalent states have equal values | §2 | `DTH.V_eq_of_qequiv` |
| `Φ ≤ 1200`, and every live transition strictly increases `Φ` | §3, `src/dth/docs/EXACTNESS_PROOF.md` | `DTH.phi_le`, `DTH.phi_lt_successChild`, `DTH.phi_lt_failChild` |
| Backward induction defines `V`, and `V` satisfies the Bellman equation | §3–§4 | `DTH.V`, `DTH.V_bellman` |
| The stage matrix is Toeplitz: `M[d,c] = S_{c-d+1}` for `c ≥ d`, else `F` | §4 | `DTH.stage_eq_toeplitz` |
| Minimax: the matrix game has a value and optimal strategies | §4 | `MatrixGame.lowerValue_eq_upperValue`, `MatrixGame.exists_optimal` |
| Pure-saddle test gives the value | §4 | `MatrixGame.value_of_pure_saddle` |
| Adjacent rows differ by `D b_d + ∑ Δ_m b_{d+m}` | §4, proof of the recurrence | `Toeplitz.mulVec_sub_succ` |
| The recurrence equalizes the rows; with nonnegative weights both mixtures are optimal and the value is `F + (S_1 - F)/W` | §4 | `Toeplitz.mulVec_eqCol`, `Toeplitz.equalizer_value` |
| Persymmetry: for `p = reverse q`, `(pᵀM)_j = (Mq)_{59-j}`, so a clipped mixture still gives a sound enclosure | §4, `try_rung2` | `Toeplitz.toeplitz_persymmetric`, `Toeplitz.mirror_certificate` |
| `L ≤ V ≤ U` for any mixed pair; gap `≤ 1e-6` puts the midpoint within `5e-7` | §5 | `MatrixGame.certificate_encloses_value`, `MatrixGame.abs_midpoint_sub_value_le` |
| The value is 1-Lipschitz in the payoffs | §5 | `MatrixGame.abs_value_sub_le` |
| `|V̂ - V| ≤ (1201 - Φ) · 5e-7`, and at the opening `V ± 0.00061`, `P ± 0.00031` | §5–§6 | `DTH.abs_sub_V_le`, `DTH.opening_error_bounds` |

## STL leap window (`src/crates/docs/LEAP_CERTIFICATE.md`, `src/stl/docs/GAME_AND_SOLVER.md`)

| Claim | Lean |
| --- | --- |
| A drop at 61 fails against every legal check | `Leap.leap_drop_fails` |
| The window game (square stage plus a constant row `f`) has value `max(v60, f)` | `Leap.value_window`, `Leap.value_windowStage` |
| The extra row never lowers the value | `Leap.value_le_value_window` |
| A certified enclosure `[MN, MX]` lifts to `[max(MN, f), max(MX, f)]`, and the lift is 1-Lipschitz | `Leap.lift_mem`, `Leap.lift_lipschitz`, `Leap.lift_gap_le` |
| The stored value `max(v_sq, f)` is within `ε` of the window value | `Leap.stored_error`, `Leap.stored_midpoint_error` |
| Two stored window values of one stage can differ by up to `3ε/2`, not `ε` | `Leap.two_stored_diff` |

## Extended library (`FormalExtended.lean`)

`Formal/Extended/` holds deeper solver and game results. The default build
compiles and audits them, but nothing in the core imports them.

| Module | What it proves | Key declarations |
| --- | --- | --- |
| `DTH/Revival.lean` | The revival surface: range, monotonicity, zero set, minimum on the reachable domain, table values, bucket invariance, the half-life `60 ln 2 / ln(4/3) ∈ (144.55, 144.6)`, and `deaths ≤ t/60` | `halfLife_bounds`, `bucket_halfLife`, `deaths_le_ttd_div`, `revival_min` |
| `DTH/Stage.lean` | 61 transition classes fix the 60×60 matrix and give exactly 61 degrees of freedom; payoffs and `V` lie in `[-1, 1]`; the Bellman solution is unique | `stage_eq_classMatrix`, `finrank_range_classMatrixLin`, `abs_V_le_one`, `V_unique` |
| `Toeplitz/PureSaddle.lean` | Rung 1: the O(60) closed forms `max(lo, min(f, s0))` and `min(hi, max(f, s0))`, their soundness, and the lifted window form | `pureMaximin_toeplitzL`, `pureMinimax_toeplitzL`, `window_value_of_s0_le` |
| `Toeplitz/Kink.lean` | Nondecreasing success payoffs with `f > s0` give nonnegative weights; with `f ≤ s0` a pure saddle exists; exact-arithmetic residue needs a negative step | `weights_nonneg_of_monotone`, `monotone_ladder` |
| `Leap/CrashBasis.lean` | The packing-LP crash basis: Toeplitz inverse as a power series, kernel weights as prefix sums, crash infeasibility on residue stages | `inv_ltToeplitz`, `weights_eq_prefixSum`, `crash_basicValues_eq_kernelWeights`, `crash_residue_infeasible` |
| `Float/Shortcut.lean` | The nonnegative-weight shortcut of `LEAP_CERTIFICATE.md` under an abstract binary64 rounding model: residual `< 6e-14`, spread `< 3.54e-12`, last-row value within `4e-14`, enclosure within `4e-12` | `Run.Valid.residual_le`, `Run.Valid.spread`, `Run.Valid.last_row_close`, `Run.Valid.enclosure` |
| `STL/Leap.lean` | The STL leap game: 61-second window turns, action 61 legal only for Baku as Dropper, child clocks and next-minute snap, leap-route variations, the leap value equals the DTH value after 3600, the 16,711 revivable classes, and the class maps under the quotient | `turnDuration_eq_61_iff`, `sixtyOne_mem_legalSeconds_iff`, `childClock_eq_engineClock`, `VL_eq_V`, `alivePairs_card`, `succClass_quot`, `failClass_quot` |

### Findings from the extended library

- **Fixed in this change:** `docs/REVIVAL_MODEL.md`, `src/dth/docs/GAME_AND_SOLVER.md`,
  and `src/abstract/docs/MODEL.md` gave the half-life as `144.3` s (`14.43` and
  `28.86` bucket units). The value is `144.6` s (`14.46` and `28.91` units);
  `halfLife_doc_wrong` refutes the old values.
- **Fixed in this change:** `docs/REVIVAL_MODEL.md` called `t/60` a lower bound
  on deaths. It is an upper bound; `ttd_div_not_lower_bound` gives the
  counterexample (one death at `s = 180`).
- The rung-1 doc step "every row minimum is one of four numbers"
  (`src/dth_compact/architecture.md`) is false; the 60 row minima can be
  distinct (`rowMin_injective_counterexample`). The closed forms stay correct.
- The Rust abstract pure path accepts near-saddles at `1e-12` and stores a gap
  of `0.0` (`nearPennies_counterexample`); the value error is at most `2e-12`.
- In binary64 the monotone-payoff guarantee fails by overflow: `s = (0, 1, …, 1)`,
  `f = 2e-6` has exact weights `500000^k > 2^1024` at `k = 55`
  (`overflow_counterexample`). The stage then reaches the residue without a
  negative step, so kink seeding returns no seed. Soundness is unaffected.
- A clipped recurrence mixture can have gap `1`, so clipping alone does not
  certify (`clip_counterexample`); the full check is required.
- The C++ assembly box accepts `|m| ≤ 1 + 1e-9`, but the LP rung requires the
  shifted matrix in `[1, 3]` exactly; the build then aborts, fail closed
  (`box_slack_lp_mismatch`).
- The kink moves by `-1` under one more second of Checker ST only when the old
  kink index is at least 2 (`kink_shift_counterexample`).

### Open items in the extended library

- The STL forward-reachability lemma is proved for an abstract step relation;
  its instantiation to the builder's keys is not proved.
- The reachable-state totals and figure percentages are artifact measurements,
  not theorems.
- Termination of the crash-start simplex within 512 pivots is not proved;
  soundness rests on the certificate.

## Assumptions

Theorems are about real-number mathematics. Each definition mirrors the cited
code. Only `Formal/Extended/Float/` and a few named extended lemmas treat
rounding, and they use an explicit abstract rounding model. The value `V` is defined by the Bellman recursion of paper §4.
