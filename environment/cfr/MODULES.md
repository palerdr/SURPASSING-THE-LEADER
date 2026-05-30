# `environment/cfr/` — Rigorous Solver Core

This namespace promises:

- **Exact-second action space.** No bucketing, no discretization, no representative-value substitution. Drop and check times are integers in [1, 60] (or [1, 61] in the leap-second window). The structural firewall test (`tests/test_cfr_firewall.py`) AST-walks every file here and fails the build if any identifier matches `bucket`, `discretize`, `quantize`, `AbstractState`, `representative_`, `_BUCKET_SIZE`, or `NUM_BUCKETS`.
- **Terminal-only utility.** Leaf values come from `terminal_value` (+1 / −1 / 0 from Hal's perspective), never from shaped reward, route bonuses, or value-net frontier evaluation. The import firewall blocks `environment.{reward, route_math, route_stages, awareness, observation}`.
- **Engine-derived chance branches.** Survival probabilities, death durations, and clock advancement come from the actual `src/` engine. No parallel/truncated reimplementation lives here.

> **Naming note.** Despite the `cfr/` directory name, this namespace does **not** run information-set counterfactual-regret minimization. It is finite-horizon stochastic zero-sum **minimax** (a row-player LP solved at each simultaneous move) plus a **matrix-game MCTS**. The name is historical; `half_round.py` is the only file that does iterative regret matching, and it is a legacy baseline. Treat "CFR" here as "the exact/search core", not as Zinkevich-style CFR.

Bucketed legacy code lives under `environment/abstractions/`. The `hal/` directory is a separate, intentionally-bucketed playable baseline; it does not share state with this namespace.

## Module index

(Reflects the merged file layout actually on disk — earlier split files like `exact_solver.py` / `minimax.py` / `candidates.py` were consolidated into `exact.py` and `selective.py`.)

| File | Purpose | Notes |
|---|---|---|
| `exact.py` | Canonical rigorous solver. `ExactPublicState`, `solve_minimax(payoff)` (zero-sum LP via `scipy.optimize.linprog`), `ExactGameSnapshot`, `enumerate_joint_actions`, `expand_joint_action`, `evaluate_joint_action`, `solve_exact_finite_horizon`, `UtilityBreakdown`. | Recurses through engine-derived chance branches; bounded LRU cache keyed by (public state, horizon, perspective). No frontier heuristic. The maximizing player is placed on the LP rows per role (see the role-solve comment in `solve_exact_finite_horizon`). |
| `selective.py` | `generate_candidates(...)`, `selective_solve(...)`, `audit_against_full_width(...)`. | Mirrors `solve_exact_finite_horizon` over a reduced candidate-second set (critical seconds + diagonal + overflow / safe-check boundaries); the audit reports the value gap vs full width. |
| `mcts.py` | Matrix-game-at-node MCTS: PUCT-augmented Q, per-node minimax selection (`solve_minimax` on the exploration-augmented matrix), chance sampling via the engine, transposition table, optional critical-root subgame re-solve. | Backup keeps Hal-perspective values everywhere (no sign flips). |
| `evaluator.py` | `LeafEvaluator` protocol + `TerminalOnlyEvaluator`, `TablebaseEvaluator`, `ValueNetEvaluator`. | Thin leaf-value glue layer; the natural reading entry point. |
| `tablebase.py` | Lazy factory registry. `solve_target(name)`, `verify_pinned_value(name)`, tag filters. | Pinned scenarios are exact supervised anchors for the value net / calibration. No model dependency. |
| `tactical_scenarios.py` | Named `TacticalScenario` fixtures (forced overflows, leap-window probes, safe-budget pairs, CPR pairs, …). | Pinned scenarios carry `expected_value`; relational pairs carry only `tags`. |
| `timing_features.py` | Engine-derived LSR / safe-budget / projected-gap arithmetic (`lsr_variation_from_clock`, `is_leap_window`, `safe_strategy_budget`, `rounds_until_leap_window`, …). | Pure arithmetic; no category labels, no shaping. |
| `subgame_resolve.py` | `is_critical(...)` heuristic + `resolve_subgame(...)` wrapper around `selective_solve`. | Used by `mcts.py` to re-solve critical roots more exactly. |
| `diagnostics.py` | `diagnose_exact_strategy(game, result)` → exploitability + best-response value + Nash gap. | Catches non-zero-sum leaks empirically; off the hot path. |
| `half_round.py` | **Pre-Phase-1 regret-matching baseline** (single 1..60 matrix solved by iterative regret matching). Also exports `survival_probability`, imported by `hal/value_net.py` and `abstractions/tree.py`. | Legacy comparison fixture; new rigorous code uses `exact.py`, not `solve_half_round`. |

## Where things go that don't belong here

- **Bucketed / abstracted CFR**: `environment/abstractions/` (`tree.py`, `game_state.py`, `full_round.py`).
- **Bucketed playable baseline (CanonicalHal)**: `hal/`.
- **Reward shaping, stage labels, observation features**: `environment/{reward.py, route_stages.py, observation.py, awareness.py}` — explicitly forbidden by the firewall test from being imported here.
- **Route-math category labels** (`classify_round_gap_seconds`, etc.): `environment/route_math.py`. The pure subset migrated here as `timing_features.py`; the labels stayed behind.
