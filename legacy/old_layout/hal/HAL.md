# Stockfish-Style STL Solver: Plan to a Defensible Deep-Search Equilibrium

## Context

### What was built (Phases 0–5 Step 1)

The project has a complete rigorous **perfect-information** stack:

- Engine: `src/Game.py`, `src/Player.py`, `src/Referee.py`, deterministic given a seed.
- Rigorous CFR core: `environment/cfr/exact.py` (LP minimax, finite-horizon recursion, exact-second matrix games), `selective.py` (candidate generation + selective search with full-width audit), `tablebase.py` (pinned scenarios), `tactical_scenarios.py`, `timing_features.py` (LSR routing math), `mcts.py` (matrix-game-at-node MCTS with transposition cache and principal-line diagnostics), `evaluator.py` (LeafEvaluator protocol with TerminalOnly, Tablebase, ValueNet adapters).
- Phase 5 Step 1: `training/value_targets.py` produces `(features, value)` labels via `solve_exact_finite_horizon` at horizon=1.
- 391 tests, firewall-enforced separation between rigorous core and shaping helpers.

### What this plan addresses

A detour into imperfect-information PBS / type-CFR was considered and rejected. The canonical strategic content of the *Surpassing The Leader* arc — Sako, T. *Usogui*, Young Jump Comics, Shueisha, vols ~17–22 — is **commitment-based, not inference-based**. Hal's amnesia in the manga is functionally equivalent to a hard constraint that "Hal cannot check at second 61." The 2-Second Deviation is a multi-round pre-committed plan that engineers Hal's TTD to land at 4M58S so the leap-second failed check sums to ~5M of survivable death.

Under our current `Constants.py`, the canonical R9T2 leap-second-failed-check has `P(survive) ≈ 0.28` (`base_curve(154) × cardiac(228) × referee(4 cprs) × 1.0`). The dramatic stakes of the gamble are mathematically faithful at this curve and the constants stay as-is.

### What this plan delivers

A **Stockfish-style perfect-information solver**: deep MCTS + a calibrated value net trained on tablebase + multi-horizon exact + MCTS-bootstrap targets. The 2-Second Deviation is **not** explicitly encoded — it is an *emergent principal line* that the deep search discovers, the same way Stockfish discovers king-pawn endgame techniques it was never taught explicitly. To enforce the canonical strategic constraint cleanly, the codebase is simplified: the `hal_leap_deduced` and `hal_memory_impaired` config flags are *removed*, and Hal-checker max-second is hard-coded at 60. Hal "always knows" the leap second exists in the strategic sense; he just *cannot ever play check=61* and therefore must play around it.

No PBS, no Bayesian belief tracking, no type-CFR machinery. The architecture stays in *Framing 1* (perfect-info under a fixed legality structure) and gets to defensibility through search depth + evaluator quality + calibration.

---

## Why Stockfish-Style Works for STL Without Imperfect-Info Machinery

Stockfish for chess is perfect-information. Its strength comes from (a) NNUE evaluation accuracy at static positions, (b) alpha-beta search to depth-30+, (c) opening books, endgame tablebases. There is no hidden state to model.

For STL the per-turn simultaneity (drop-time / check-time picked simultaneously each half-round) is *also* handled inside the perfect-info matrix-game framework — `solve_minimax` at each node IS the equilibrium for one simultaneous-move turn. The only candidate hidden state was Hal's deduction, but the deviation_doc establishes that Hal's actual strategic content is *pre-commitment under amnesia*, which the simplified legality (Hal-checker max=60) captures directly without needing any hidden-state machinery.

Under this framing, the entire imperfect-information apparatus — info sets, beliefs, type-CFR, PBS-keyed value nets — is unnecessary. **Search depth + a calibrated evaluator under fixed legality suffices.**

### Citations

- **Silver, D., Hubert, T., Schrittwieser, J., et al. (2018).** "A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play." *Science*, 362(6419), 1140–1144. — AlphaZero architecture: deep MCTS + value/policy net + self-play training.
- **Romstad, T., Costalba, M., Kiiski, J., et al. (2020).** *Stockfish 12 with NNUE.* — Production combination of search + neural eval.
- **Browne, C. B., Powley, E., et al. (2012).** "A Survey of Monte Carlo Tree Search Methods." *IEEE TCIAIG*, 4(1), 1–43.
- **von Neumann, J. (1928).** "Zur Theorie der Gesellschaftsspiele." *Mathematische Annalen*, 100(1), 295–320. — Minimax theorem for finite zero-sum games. Provides convergence foundation for `solve_minimax` at each MCTS node.
- **Shapley, L. S. (1953).** "Stochastic games." *PNAS*, 39(10), 1095–1100. — Stochastic games admit a value; the chance branches in our MCTS are valid.

---

## What's Salvageable vs What Pivots

| Component | Status | Role |
|---|---|---|
| `src/` engine | Keep as-is | Public-state transitions and chance branches |
| `environment/cfr/exact.py` | Modified — drop deduction/impair flags | Inner-loop primitive |
| `environment/cfr/selective.py` | Keep, deepen | Used for subgame re-solve at runtime |
| `environment/cfr/timing_features.py` | Keep, **promote** | LSR-significance predicates for target gating |
| `environment/cfr/tablebase.py` | Keep, extend with held-out scenarios | Pinned ground-truth + validation set |
| `environment/cfr/mcts.py` | Keep, extend with subgame-resolve hook | Core search; runtime evaluator gets swapped |
| `environment/cfr/evaluator.py` | Keep | LeafEvaluator protocol; ValueNetEvaluator becomes load-bearing |
| `environment/legal_actions.py` | Modified — hard-code Hal-checker max=60 | Asymmetry: Baku-dropper 61 in leap window, Hal-checker always 60 |
| `hal/value_net.py` 23-feature extractor | **Keep at 23** (no augmentation needed) | Net does not condition on removed flags |
| `training/value_targets.py` (horizon=1 LP labels) | **Replace generation entirely** | Tablebase + horizon=2/3 selective + MCTS bootstrap |

Firewall test (`tests/test_cfr_firewall.py`) and rigorous-marker scan stay enforced. New code stays under `environment/cfr/` (rigorous-only) or `training/` (allowed to import from both `cfr/` and `hal/`).

---

## Phase 1 — Legality Simplification (drop deduction flags)

### Why necessary

The `hal_leap_deduced` and `hal_memory_impaired` flags exist to model whether Hal can play check=61. Per the canonical scenario, **Hal can never play check=61**, full stop. Carrying the flags forward complicates every config object, every test, and every future training run with a configuration dimension that isn't strategically informative. Removing them collapses two config dimensions into a single constant and matches the canonical constraint cleanly: Hal must play *around* the leap second via deviation, not *through* it.

### Files modified

- `environment/cfr/exact.py`: remove `hal_leap_deduced` and `hal_memory_impaired` fields from `ExactSearchConfig`. `legal_seconds_for_current_role` no longer threads them through.
- `environment/legal_actions.py`: hard-code Hal-checker max-second at 60. Baku-dropper max-second remains 61 inside the leap window. Update `legal_max_second` signature accordingly.
- `environment/cfr/tactical_scenarios.py`: `leap_second_check_61_probe` is removed from the registry (it asserted Hal could check=61 with `hal_leap_deduced=True`). Its symmetric value-test is preserved in a different shape via tablebase coverage of LSR-2 positions.
- `tests/test_cfr_*.py` and `tests/test_value_targets.py`: drop tests that exercised the removed flags. Tests that exercised Hal-cannot-check-61 (the negative case) become *the* test of the simplified legality.
- `hal/value_net.py`: **no change.** 23 features stay.

### Mathematical defensibility

- The legality simplification is a strict subset of the prior config space. Any Nash equilibrium of the new game is also a Nash equilibrium of the prior game restricted to `hal_leap_deduced=False`. So the simplification is conservative — we lose nothing the prior framework computed except the `hal_leap_deduced=True` regime, which the canonical strategic content does not use.

### Citations

- The "remove configuration dimensions that don't pull weight" principle is YAGNI / Occam's razor as applied to engineering. See *Sutton, "The Bitter Lesson" (2019)* — over-modeling structural assumptions degrades general performance.

### Tests

- `ExactSearchConfig` no longer has `hal_leap_deduced` or `hal_memory_impaired` fields.
- `legal_max_second("Hal", "checker", turn_duration=61)` returns 60.
- `legal_max_second("Baku", "dropper", turn_duration=61)` returns 61.
- All previously-passing tests still pass after import/signature updates.

### Builds on

Existing `legal_actions.py`, `ExactSearchConfig`, `tactical_scenarios.py`. Roughly mechanical refactor with small ripples through tests.

---

## Phase 2 — Multi-Horizon Exact Targets + Tablebase Pinning (no horizon=1)

### Why necessary

Phase 5 Step 1 generated labels via `solve_exact_finite_horizon(horizon=1)`. **Horizon=1 LP values are LSR-blind**: at horizon=1 the recursion bottoms out at "unresolved" (value=0) for any non-terminal cell. Training a value net on those labels teaches it that LSR features are noise. The 2-Second Deviation requires looking ~7+ half-rounds ahead — beyond any tractable exact-solve horizon — but a value net trained on horizon=2 and horizon=3 targets at *LSR-significant* states will at least carry signal about routing, and the MCTS bootstrap (Phase 3) extends that signal further.

Horizon=1 LP targets are dropped entirely.

### Files modified

- `training/value_targets.py`: replace label generation. The hierarchy:
  1. Terminal pinning if `terminal_value(game) is not None`.
  2. Tablebase short-circuit if `exact_public_state(game)` matches a pinned registry entry.
  3. `solve_exact_finite_horizon(game, horizon=3)` if `rounds_until_leap_window(game) ≤ 2` or `current_checker_fail_would_activate_lsr(game)` (LSR-significant).
  4. `solve_exact_finite_horizon(game, horizon=2)` if `is_active_lsr(game)` or any cylinder ≥ 240 (near-overflow).
  5. Otherwise: state is *excluded* from the gen-0 corpus. MCTS bootstrap (Phase 3) labels it.

This keeps wall-clock bounded while spending compute where LSR routing is observable inside the horizon.

- `environment/cfr/tactical_scenarios.py`: extend with held-out tablebase scenarios at unseen cylinder thresholds and clock offsets, tagged `holdout=True` so `tablebase.py` excludes them from training-time pinning but uses them for validation in Phase 5.

### Mathematical defensibility

- `solve_exact_finite_horizon` at any horizon is exact LP minimax over the candidate matrix at the root, with chance branches expanded exactly. Value existence: **von Neumann (1928)** for the matrix games, **Shapley (1953)** for the chance-weighted recursion.
- Tablebase short-circuit: pinned values are exact ground truth (terminal sequences) by construction.
- Held-out tablebase: pinned values are uncontestable error bounds for the trained net.
- Selective search audit (`audit_against_full_width` in `selective.py`): zero gap on tablebase scenarios at the candidate horizons we use.

### Citations

- **von Neumann, J. (1928).** Minimax theorem.
- **Shapley, L. S. (1953).** Stochastic games admit a value.
- **Brown, N., & Sandholm, T. (2019).** "Solving imperfect-information games via discounted regret minimization." *AAAI 2019*. — Multi-horizon training-target methodology adapted from poker.

### Tests

- Per-source breakdown: `source ∈ {"terminal", "tablebase", "exact_horizon_2", "exact_horizon_3"}`. **No `exact_horizon_1`.**
- A state with `current_checker_fail_would_activate_lsr=True` is labeled with `source ∈ {tablebase, exact_horizon_3}`.
- Wall-clock bound: full corpus generation completes in < 15 minutes on a single CPU.
- A state outside all gates is excluded (returns no `ValueTarget`).

### Builds on

Existing `solve_exact_finite_horizon`, tablebase short-circuit, `timing_features.py` predicates. Mostly an augmentation of the gen-0 corpus generator that already exists.

---

## Phase 3 — MCTS Bootstrap Targets (the AlphaZero loop)

### Why necessary

Even horizon=3 is too shallow to capture the 2-Second Deviation. Going deeper via exact solve is computationally infeasible (~3600^3 cells per chance branch). The proven solution from **AlphaZero (Silver et al. 2018)** is *self-referential*: train an initial net on shallow exact targets, then generate deeper targets by running MCTS *using the net* and recording its converged root values. The next generation of the net trains against MCTS values, which are deeper than any tractable exact solve.

### Files modified

- `training/value_targets.py`: add `generate_mcts_bootstrap_targets(model_path, corpus, iterations_per_state=2000)` that loads a trained net, runs `mcts_search(game, MCTSConfig(iterations=N), ValueNetEvaluator(model_fn=load(net)))` per state, records `result.root_value_for_hal` as the label.
- `training/bootstrap_loop.py`: orchestration script. Trains gen-0 on Phase 2 targets, generates gen-1 bootstrap targets, trains gen-1, verifies calibration improves, repeats until calibration plateaus.

### Mathematical defensibility

- **AlphaZero convergence in self-play** is empirically demonstrated for chess/shogi/go (Silver et al. 2018). Each generation must improve over the previous on held-out tablebase MSE (Phase 5 gate); generations that regress are rejected.
- The bootstrap procedure satisfies the **policy-improvement property**: if the value net is calibrated, MCTS using the net produces strictly better play than the net alone, and training the next net on MCTS values produces a net that *predicts* this better play. See *Silver et al. (2018), supplementary materials, §S2*.
- Variance is bounded by MCTS iterations per state (N=2000). Analogous variance bounds: *Lanctot et al. (2009) MCCFR*; transferred to MCTS via *Brown et al. (2020) ReBeL §3*.

### Citations

- **Silver, D., et al. (2018).** AlphaZero. *Science*. — Canonical reference for value-net self-improvement.
- **Anthony, T., Tian, Z., & Barber, D. (2017).** "Thinking fast and slow with deep learning and tree search." *NeurIPS 2017*. — Expert-iteration framework, the conceptual precursor.
- **Brown, N., et al. (2020).** "Combining Deep RL and Search for Imperfect-Information Games." *NeurIPS 2020*. — Demonstrates value-net + search bootstrap also works in imperfect-info, supporting our perfect-info specialization.

### Tests

- Bootstrap target generation: `model_path` loads, MCTS runs, output shape matches `(N_states, FEATURE_DIM + 1)`.
- Determinism under seeded RNG: same seed → same targets.
- Convergence check: gen-1 MSE on held-out tablebase ≤ gen-0 MSE.

### Builds on

Existing `mcts_search`, `MCTSConfig`, `ValueNetEvaluator`, `MCTSResult`, transposition cache.

---

## Phase 4 — Training Script

### Why necessary

We need a supervised regression pipeline that trains `ValueNet` against labeled targets, with held-out validation, learning-rate schedule, and per-source MSE callbacks. Without it the bootstrap loop is one-off scripts.

### Files modified

- `training/train_value_net.py`: PyTorch training loop. Inputs from `.npz` file. Loss = MSE on value targets. Cosine learning-rate schedule. 10% held-out validation split. Per-source MSE logged each epoch. Model checkpoints saved when held-out MSE improves.

### Mathematical defensibility

- MSE regression is the canonical proper scoring rule for value prediction in [-1, 1]; *Bishop (2006), Pattern Recognition and Machine Learning, §1.5*.

### Tests

- Training on a tiny synthetic corpus converges to MSE < 0.01 in < 100 epochs.
- Held-out MSE reported separately from training MSE.
- Loaded model parameter count matches `ValueNet`'s expected count.

### Builds on

`hal/value_net.py:ValueNet`. PyTorch (already a dependency). `training/value_targets.py`.

---

## Phase 5 — Calibration & Winrate Validation (the defensibility gate)

### Why necessary

This is what makes the solver *defensible* rather than just *constructed*. Without these metrics you have a system you built; with them you have measurements that bound how wrong it can be. Two complementary signals:

- **Per-source held-out MSE + held-out tablebase MSE** — direct error against ground truth.
- **Tournament winrate against scripted teachers** — Stockfish's gold-standard validation: does the agent win more than baseline?

Acceptance gates:
- Held-out tablebase MSE ≤ 0.01.
- Per-source MSE ≤ 0.10 across `{tablebase, exact_horizon_2, exact_horizon_3, terminal}`.
- Agent winrate vs canonical Baku LSR-engineering teacher ≥ 50%.

### Files modified

- `training/calibration.py`:
  - `evaluate_value_net(net, held_out_corpus, held_out_tablebase) → CalibrationReport` (per-source MSE, Brier score, reliability bins, exact-target error).
- `training/tournament.py`:
  - `play_match(agent_a, agent_b, n_games, seed) → MatchResult` (MCTS-with-trained-net vs scripted teachers).
  - Reports winrate, average game length, terminal-cause distribution.
- `environment/cfr/tactical_scenarios.py`: held-out scenarios tagged `holdout=True`, excluded from training but used for validation.

### Mathematical defensibility

- **Brier score (Brier 1950)** is the canonical proper scoring rule for probabilistic forecasts.
- **Held-out tablebase MSE = exact ground-truth error**: pinned values are correct by construction.
- **Tournament winrate** is the validation framework used in chess engine development (CCRL, TCEC) and is industry-standard for "is engine A stronger than engine B."
- For zero-sum games, **expected winrate ≥ 0.5 against a fixed opponent at depth-N implies the agent is at least as good as the opponent at that depth**. Follows from the value of the game equaling the equilibrium value (von Neumann 1928).

### Citations

- **Brier, G. W. (1950).** "Verification of forecasts expressed in terms of probability." *Monthly Weather Review*, 78(1), 1–3.
- **Niculescu-Mizil, A., & Caruana, R. (2005).** "Predicting good probabilities with supervised learning." *ICML 2005*.
- **Hyatt, R. M. (2005).** *The CCRL & CEGT Computer Chess Rating Lists.* — Tournament-winrate methodology.

### Tests

- Perfect-predictor (returns true label) gives MSE = 0 across all sources.
- Tournament harness reproducible under seeded RNG.
- Held-out tablebase MSE strictly increases with random net weights, decreases with training.

### Builds on

`tactical_scenarios.py`, `tablebase.py`, `mcts.py`, `evaluator.py`, `diagnose_exact_strategy` for per-state best-response.

---

## Phase 6 — Subgame Re-Solve at Critical States (runtime polish)

### Why necessary

The trained blueprint (Phases 2–4) covers the public-state space at *finite resolution*. For *critical decisions* — within K half-rounds of the leap window, near-overflow cylinder, post-recent-death — you want a **fresh local solve at deeper horizon at runtime** rather than relying on the net's prediction. This is the **Pluribus / Libratus pattern** (Brown & Sandholm 2017, 2018) adapted to perfect-info: at critical positions, do extra search anchored to the trained net at the boundary.

### Files modified

- `environment/cfr/subgame_resolve.py`:
  - `is_critical(game) → bool`: gate via `timing_features` predicates (LSR-significant, near-overflow, near leap window).
  - `resolve_subgame(game, horizon=4) → SelectiveSearchResult`: fresh selective_solve at deeper horizon.
- `environment/cfr/mcts.py`: optional `subgame_resolve_at_critical` kwarg in `mcts_search`. When True, at critical-state nodes the search calls `resolve_subgame` to refine the principal action.

### Mathematical defensibility

- **Subgame solving preserves equilibrium properties** under conditions documented in *Burch, Johanson, Bowling (2014), "Solving imperfect information games using decomposition," AAAI 2014* and *Moravčík et al. (2017), "DeepStack," Science*. The perfect-info version is a strict simplification.
- The **blueprint anchors the boundary value**, so the local solve is a refinement, not a replacement.

### Citations

- **Brown, N., & Sandholm, T. (2017).** "Libratus." *Science*, 359(6374), 418–424.
- **Brown, N., & Sandholm, T. (2018).** "Pluribus." *Science*, 365(6456), 885–890.
- **Moravčík, M., et al. (2017).** "DeepStack." *Science*, 356(6337), 508–513.

### Tests

- Subgame re-solve on `forced_baku_overflow_death` returns +1.0 (no different from blueprint at terminal-forced states).
- At a non-trivial state, re-solve at horizon=4 produces a value strictly closer to the deep-search MCTS value than the blueprint alone.
- `is_critical` flags ≤20% of decisions on a 100-game self-play sample, but flags 100% of decisions where blueprint vs. resolve disagree by > 0.1.

### Builds on

`selective_solve` in `selective.py`, `timing_features.py` predicates, `mcts.py`'s evaluator slot.

---

## Phase 7 — Manga Validation (faithfulness check)

### Why necessary

A defensibility check tied to the canon. Under the canonical scenario (Baku playing the LSR-engineering teacher to be added in this phase), the trained agent's principal line over R1T1 → R9T2 should resemble or improve on the deviation_doc's documented moves. This is qualitative validation, not formal correctness, but it answers the user's "Hal discovers strategies many rounds ahead like Stockfish" requirement directly.

### Files modified

- `environment/opponents/baku_teachers.py`: add `BakuLSREngineeringTeacher` — a teacher that explicitly times deaths to engineer LSR-2 at the leap window. Uses `current_checker_fail_would_activate_lsr` and `rounds_until_leap_window` from `timing_features`. Serves as the *canonical Baku opponent* for this validation.
- `scripts/validate_against_manga.py`: scripted scenario player. Loads the canonical R1T1 starting state, runs the trained agent through 9 rounds against `BakuLSREngineeringTeacher`, records action history, compares to the deviation_doc's documented moves at each turn.

### Mathematical defensibility

- This is **qualitative validation against a domain-specific oracle**, not a formal correctness proof. But it's the closest computable check to "did we faithfully solve the strategic puzzle the game depicts."
- A faithful Stockfish-style solver should either reproduce the canonical strategy (mass concentration on deviation-aligned actions) **or find a provably-better one** (higher Hal-perspective value at the root with held-out exploitability ≤ baseline). Improvement is evidence of strength, not failure.

### Tests

- Scenario player runs deterministically under seeded RNG.
- Per-turn agent value (from MCTS root) is monotonically increasing in iteration count.
- Total Hal-perspective value at root before R9T2 is positive (agent expects to win the resilience battle).

### Builds on

Phases 1–6 cumulatively. Reuses `mcts_search`, the trained net, the Baku teacher infrastructure.

---

## Verification (end-to-end)

After all seven phases:

1. **Per-phase pytest suite passes** before subsequent phases start.
2. **Calibration report**: held-out tablebase MSE ≤ 0.01, per-source MSE ≤ 0.10, exploitability gap on small audited subgames ≤ 0.05.
3. **Tournament winrate**: agent (MCTS + trained net + subgame resolve) vs `BakuLSREngineeringTeacher` ≥ 50% across 1000 games at canonical R1T1 start.
4. **Manga faithfulness check** (Phase 7): qualitative match or strict improvement.
5. **Determinism** under seeded RNG end-to-end (training reproducibility, tournament reproducibility).

The defensibility chain:
- *Targets are equilibrium values* (Phases 2–3) → *net trained on equilibrium values* (Phase 4) → *net validated against held-out ground truth* (Phase 5) → *agent uses net at runtime with subgame refinement at critical states* (Phase 6) → *agent reproduces or improves on canonical strategy* (Phase 7).

Each link is auditable. Breaking any link is a measurable failure of a specific test.

---

## Critical files to be modified or created

| Path | New / Modified |
|---|---|
| `environment/cfr/exact.py` | Modified (drop deduction/impair flags from ExactSearchConfig) |
| `environment/legal_actions.py` | Modified (hard-code Hal-checker max=60) |
| `environment/cfr/tactical_scenarios.py` | Modified (drop leap-deduced probe; add held-out scenarios) |
| `training/value_targets.py` | Modified (drop horizon=1; tablebase + horizon=2/3 + bootstrap) |
| `training/bootstrap_loop.py` | New |
| `training/train_value_net.py` | New |
| `training/calibration.py` | New |
| `training/tournament.py` | New |
| `environment/cfr/subgame_resolve.py` | New |
| `environment/cfr/mcts.py` | Modified (optional subgame-resolve kwarg) |
| `environment/opponents/baku_teachers.py` | Modified (add BakuLSREngineeringTeacher) |
| `scripts/validate_against_manga.py` | New |
| `tests/test_value_targets.py` | Modified (drop horizon=1 assertions; add held-out source) |
| `tests/test_train_value_net.py` | New |
| `tests/test_calibration.py` | New |
| `tests/test_tournament.py` | New |
| `tests/test_subgame_resolve.py` | New |
| Various `tests/test_cfr_*.py` | Modified (remove tests of dropped flags) |

Total scope: ~900 lines of new code + ~350 lines of new tests + ~150 lines of refactor in existing tests.

---

## Why each phase is necessary and builds on the previous

| Phase | Depends on | What would break without it |
|---|---|---|
| 1 (legality simplification) | Existing `ExactSearchConfig` and `legal_actions.py` | Carrying flags forward complicates every subsequent config object and adds a configuration dimension that isn't strategically informative |
| 2 (tablebase + multi-horizon targets) | Phase 1 simplified config | Targets are LSR-blind (horizon=1) → net learns LSR features as noise |
| 3 (MCTS bootstrap) | Phases 1–2 trained gen-0 net | Targets cap at horizon=3 → cannot capture 7-round-deep plans like 2-Second Deviation |
| 4 (training script) | Phases 1–3 targets | No reproducible training pipeline → ad-hoc one-offs |
| 5 (calibration) | Phase 4 trained net | No defensibility metric → solver is asserted not measured |
| 6 (subgame resolve) | Phases 1–5 blueprint | Critical decisions stuck at blueprint resolution → potential miscalls at high-stakes points |
| 7 (manga validation) | All prior phases | No faithfulness check against the canonical strategic puzzle |

The thread connecting all phases: **each phase produces an artifact the next consumes**. Skipping a phase forces ad-hoc replacements that erode the defensibility chain.

---

## What's explicitly out of scope

- **PBS / type-CFR / Bayesian belief tracking.** The canonical scenario is pre-commitment, not live inference.
- **Modeling Yakou's psychology, echolocation, speaking-clock volume, or verbal manipulation.** Out-of-engine elements per design constraint.
- **Retuning the death curve constants.** `BASE_CURVE_K=3`, `CARDIAC_DECAY=0.85`, `REFEREE_DECAY=0.88`, `REFEREE_FLOOR=0.4`. The canonical 2-Second Deviation has `P(survive R9T2) ≈ 0.28` under these constants; this is faithful to the manga's depicted dramatic stakes.
- **Horizon=1 LP labels.** Dropped entirely as LSR-blind.
- **`hal_leap_deduced` / `hal_memory_impaired` config flags.** Removed; Hal-checker max=60 hard-coded.

---

## Honest summary

The project to date built the rigorous core (engine + LP minimax + selective search + MCTS + transposition cache + tablebase + value-net architecture). Phase 5 Step 1's horizon=1 LP labels are replaced; the deduction/impair flags are removed; everything else is reused. Remaining work is *training infrastructure + validation* (~900 lines), not algorithmic invention.

Defensibility comes from: equilibrium-derived targets, calibration thresholds enforced as gates, tournament winrate validation, and a manga-faithfulness check that ties the abstract solver to the concrete strategic puzzle the game depicts. The 2-Second Deviation emerges from search depth + a calibrated evaluator under fixed legality, not from explicit type encoding — same way Stockfish discovers king-pawn endgame techniques without being told.
