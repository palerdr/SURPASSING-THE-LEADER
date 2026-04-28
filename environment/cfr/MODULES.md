# `environment/cfr/` — Rigorous CFR Core

This namespace promises:

- **Exact-second action space.** No bucketing, no discretization, no representative-value substitution. Drop and check times are integers in [1, 60] (or [1, 61] in the leap-second window). The structural firewall test (`tests/test_cfr_firewall.py`) AST-walks every file here and fails the build if any identifier matches `bucket`, `discretize`, `quantize`, `AbstractState`, `representative_`, `_BUCKET_SIZE`, or `NUM_BUCKETS`.
- **Terminal-only utility.** Leaf values come from `utility.terminal_value` (+1 / −1 / 0 from Hal's perspective), never from shaped reward, route bonuses, or value-net frontier evaluation. The import firewall blocks `environment.{reward, route_math, route_stages, awareness, observation}`.
- **Engine-derived chance branches.** Survival probabilities, death durations, and clock advancement come from the actual `src/` engine. No parallel/truncated reimplementation lives here.

Bucketed legacy code lives under `environment/abstractions/`. The `hal/` directory is a separate, intentionally-bucketed playable baseline; it does not share state with this namespace.

## Module index

| File | Purpose | Notes |
|---|---|---|
| `exact_solver.py` | Canonical rigorous solver. `solve_exact_finite_horizon(game, h, config) → ExactSolveResult`; `evaluate_joint_action(...)`; `exact_immediate_checker_payoff_matrix(...)`. | Recurses through chance branches; no frontier heuristic. |
| `exact_state.py` | `ExactPublicState` — full-precision state key. | For transposition caches and abstract-vs-exact audits. |
| `exact_transition.py` | `enumerate_joint_actions`, `expand_joint_action`, `ExactGameSnapshot`, `ExactSearchConfig`. | Snapshot/restore lets the recursion mutate and roll back the engine in place. |
| `minimax.py` | `solve_minimax(payoff)` — zero-sum LP via `scipy.optimize.linprog`. | One row-player solve; column-player obtained by transpose. |
| `utility.py` | `terminal_value`, `terminal_breakdown`, `UtilityBreakdown`. | Terminal-only contract. Module docstring forbids shaping leaks. |
| `diagnostics.py` | `diagnose_exact_strategy(game, result)` → exploitability + best-response value + Nash gap. | Catches non-zero-sum leaks empirically. |
| `candidates.py` | `generate_candidates(game, config) → CandidateActions`; pure function, no recursion. | Critical seconds + diagonal + overflow / safe-check boundaries. |
| `selective_search.py` | `selective_solve(...)`, `audit_against_full_width(...)`. | Mirrors `solve_exact_finite_horizon` over candidate cells; audit reports value gap. |
| `tactical_scenarios.py` | Named `TacticalScenario` fixtures (forced overflows, leap-second probe, safe-budget pair, CPR pair). | Pinned scenarios carry `expected_value`; relational pairs carry only `tags`. |
| `tablebase.py` | Lazy factory registry. `solve_target(name)`, `verify_pinned_value(name)`, tag filters. | No model dependency; targets are supervised labels for downstream MCCFR / MCTS / value net. |
| `timing_features.py` | Engine-derived LSR / safe-budget / projected-gap arithmetic (`lsr_variation_from_clock`, `safe_strategy_budget`, etc.). | Shared by rigorous CFR and the legacy `route_math` shim. No category labels live here. |
| `half_round.py` | **Pre-Phase-1 regret-matching baseline.** Exact 1..60 matrix solved by iterative regret matching. | Kept as a comparison fixture. New rigorous code should call `exact_solver`, not this. |

## Where things go that don't belong here

- **Bucketed / abstracted CFR**: `environment/abstractions/` (`tree.py`, `game_state.py`, `full_round.py`).
- **Bucketed playable baseline (CanonicalHal)**: `hal/`.
- **Reward shaping, stage labels, observation features**: `environment/{reward.py, route_stages.py, observation.py, awareness.py}` — explicitly forbidden by the firewall test from being imported here.
- **Route-math category labels** (`classify_round_gap_seconds`, etc.): `environment/route_math.py`. The pure subset migrated here as `timing_features.py`; the labels stayed behind.
