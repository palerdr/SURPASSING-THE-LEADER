"""Search, leaf evaluation, selective solve, subgame resolve, and MCTS."""


# ============================================================================
# Leaf evaluators
# ============================================================================

"""Leaf evaluators for MCTS.

When MCTS reaches a leaf, it needs a Hal-perspective value in [-1, 1].
This module defines the LeafEvaluator protocol plus concrete evaluators:

    - TerminalOnlyEvaluator: returns terminal_value or 0.0 for unresolved.
    - TablebaseEvaluator: short-circuits on positions matching pinned tablebase
        entries; falls back to a wrapped evaluator otherwise.
    - ValueNetEvaluator: wraps a trained value net behind the protocol.
"""

from typing import Callable, Literal, Protocol

import numpy as np

from stl.engine.actions import legal_max_second

from stl.solver.exact import ExactPublicState, exact_public_state, terminal_value
from stl.engine.game import Game


LeafEvaluation = tuple[float, np.ndarray, np.ndarray]


def _uniform_over_legal(max_second: int) -> np.ndarray:
    dist = np.zeros(61, dtype=np.float64)
    if max_second <= 0:
        return dist
    dist[:max_second] = 1.0 / max_second
    return dist


def uniform_policy_for_current_roles(game: Game) -> tuple[np.ndarray, np.ndarray]:
    """Return length-61 uniform distributions over legal root seconds."""
    if game.game_over:
        return np.zeros(61, dtype=np.float64), np.zeros(61, dtype=np.float64)

    dropper, checker = game.get_roles_for_half(game.current_half)
    turn_duration = game.get_turn_duration()
    drop_max = legal_max_second(dropper.name, "dropper", turn_duration)
    check_max = legal_max_second(checker.name, "checker", turn_duration)
    return _uniform_over_legal(drop_max), _uniform_over_legal(check_max)


def normalize_policy_vector(policy: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    arr = np.asarray(policy, dtype=np.float64).reshape(-1)
    if arr.shape[0] != 61:
        raise ValueError(f"policy vector must have length 61, got {arr.shape[0]}")
    arr = np.maximum(arr, 0.0)
    total = float(arr.sum())
    if total > 1e-12:
        arr = arr / total
    return arr


def normalize_leaf_evaluation(value, game: Game) -> LeafEvaluation:
    """Coerce scalar legacy outputs or explicit triples into a leaf triple."""
    if isinstance(value, tuple):
        if len(value) != 3:
            raise ValueError("leaf evaluator tuples must be (value, dropper_policy, checker_policy)")
        scalar, dropper_policy, checker_policy = value
        return (
            float(scalar),
            normalize_policy_vector(dropper_policy),
            normalize_policy_vector(checker_policy),
        )

    dropper_policy, checker_policy = uniform_policy_for_current_roles(game)
    return float(value), dropper_policy, checker_policy


class LeafEvaluator(Protocol):
    def __call__(self, game: Game) -> LeafEvaluation: ...


class TerminalOnlyEvaluator:
    """Returns terminal_value(game) for terminal positions, 0.0 otherwise."""

    def __init__(self, perspective_name: str = "Hal") -> None:
        self.perspective = perspective_name

    def __call__(self, game: Game) -> LeafEvaluation:
        tval = terminal_value(game, perspective_name=self.perspective)
        if tval is not None:
            value = tval
        else:
            value = 0.0
        dropper_policy, checker_policy = uniform_policy_for_current_roles(game)
        return float(value), dropper_policy, checker_policy


class TablebaseEvaluator:
    """Wraps another evaluator; short-circuits on tablebase hits."""

    def __init__(self, fallback: LeafEvaluator) -> None:
        self.fallback = fallback
        # Evaluates against the table of E[public state]. Imported lazily to avoid a module cycle.
        from stl.solver.tablebase import REGISTRY

        self._table: dict[ExactPublicState, float] = {}
        for factory in REGISTRY.values():
            scenario = factory()
            if scenario.expected_value is not None and not scenario.holdout:
                key = exact_public_state(scenario.game)
                self._table[key] = scenario.expected_value

    def __call__(self, game: Game) -> LeafEvaluation:
        key = exact_public_state(game)
        if key in self._table:
            dropper_policy, checker_policy = uniform_policy_for_current_roles(game)
            return float(self._table[key]), dropper_policy, checker_policy
        return normalize_leaf_evaluation(self.fallback(game), game)


class ValueNetEvaluator:
    """Wraps the trained value net behind a clean interface."""

    def __init__(self, model_fn: Callable[[Game], float]) -> None:
        self.model = model_fn

    def __call__(self, game: Game) -> LeafEvaluation:
        return normalize_leaf_evaluation(self.model(game), game)


# ============================================================================
# Selective exact search
# ============================================================================

"""Selective exact-second search: candidate generation + minimax recursion.

Single module covering:

    - Candidate exact-second generation (CRITICAL_SECONDS, CandidateActions,
        overflow_st_threshold, safe_st_budget, generate_candidates).
    - Selective minimax search over candidate cells (selective_solve)
        plus a full-width audit (audit_against_full_width).

Mirrors solve_exact_finite_horizon but expands only the strategically
meaningful cells per state. The audit reports the value gap so any
candidate-set insufficiency is observable, not silent.
"""


from dataclasses import dataclass

import numpy as np

from stl.engine.game import CYLINDER_MAX
from stl.engine.game import Game

from stl.solver.exact import (
    CFRPlusConfig,
    ExactGameSnapshot,
    ExactJointAction,
    ExactSearchConfig,
    UtilityBreakdown,
    legal_seconds_for_current_role,
    solve_cfr_plus,
    solve_cfr_plus_rust,
    solve_exact_finite_horizon,
    solve_minimax,
    terminal_value,
)


# ── Candidate generation ──────────────────────────────────────────────────


CRITICAL_SECONDS: tuple[int, ...] = (1, 2, 58, 59, 60, 61)
PLAYABLE_GRID_SECONDS: tuple[int, ...] = (5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55)


@dataclass(frozen=True)
class CandidateActions:
    drop_seconds: tuple[int, ...]
    check_seconds: tuple[int, ...]

    @property
    def joint_count(self) -> int:
        return len(self.drop_seconds) * len(self.check_seconds)


def overflow_st_threshold(checker_cylinder: float) -> int:
    """Smallest ST that drives the cylinder to CYLINDER_MAX or above."""
    return max(1, int(CYLINDER_MAX) - int(checker_cylinder))


def safe_st_budget(checker_cylinder: float) -> int:
    """Largest ST that leaves the cylinder strictly below CYLINDER_MAX."""
    return max(0, int(CYLINDER_MAX) - 1 - int(checker_cylinder))


def _legal_filter(seconds: set[int], legal: range) -> set[int]:
    return {s for s in seconds if s in legal}


def generate_candidates(
    game: Game,
    config: ExactSearchConfig | None = None,
    *,
    include_playable_grid: bool = False,
) -> CandidateActions:
    """Return candidate exact seconds for the dropper and checker."""
    config = config or ExactSearchConfig()
    dropper, checker = game.get_roles_for_half(game.current_half)
    drop_legal = legal_seconds_for_current_role(game, dropper.name, "dropper", config)
    check_legal = legal_seconds_for_current_role(game, checker.name, "checker", config)

    baseline_seconds = set(CRITICAL_SECONDS)
    if include_playable_grid:
        baseline_seconds |= set(PLAYABLE_GRID_SECONDS)
    drop_seconds: set[int] = _legal_filter(baseline_seconds, drop_legal)
    check_seconds: set[int] = _legal_filter(baseline_seconds, check_legal)

    overflow_st = overflow_st_threshold(checker.cylinder)
    safe_st = safe_st_budget(checker.cylinder)

    for d in tuple(drop_seconds):
        for c in (
            d - 1, d, d + 1,
            d + safe_st, d + safe_st + 1,
            d + overflow_st - 1, d + overflow_st,
        ):
            if c in check_legal:
                check_seconds.add(c)

    for c in tuple(check_seconds):
        for d in (
            c - 1, c, c + 1,
            c - safe_st - 1, c - safe_st,
            c - overflow_st, c - overflow_st + 1,
        ):
            if d in drop_legal:
                drop_seconds.add(d)

    return CandidateActions(
        drop_seconds=tuple(sorted(drop_seconds)),
        check_seconds=tuple(sorted(check_seconds)),
    )


# ── Selective minimax search ──────────────────────────────────────────────


@dataclass(frozen=True)
class SelectiveSearchResult:
    value_for_hal: float
    breakdown: UtilityBreakdown
    unresolved_probability: float
    half_round_horizon: int
    drop_seconds: tuple[int, ...]
    check_seconds: tuple[int, ...]
    dropper_strategy: np.ndarray
    checker_strategy: np.ndarray
    payoff_for_hal: np.ndarray | None
    candidate_count: int


@dataclass(frozen=True)
class SelectiveAuditResult:
    selective: SelectiveSearchResult
    full_width_value: float
    value_gap: float
    candidate_joint_count: int
    full_width_joint_count: int


def _terminal_breakdown(value: float | None) -> UtilityBreakdown:
    if value is None:
        return UtilityBreakdown(0.0, 0.0, 0.0, 1.0)
    if value > 0.0:
        return UtilityBreakdown(value, 1.0, 0.0, 0.0)
    if value < 0.0:
        return UtilityBreakdown(value, 0.0, 1.0, 0.0)
    return UtilityBreakdown(value, 0.0, 0.0, 0.0)


def _frontier_breakdown(game: Game, evaluator: LeafEvaluator | None) -> UtilityBreakdown:
    if evaluator is None:
        return _terminal_breakdown(None)
    value, _, _ = normalize_leaf_evaluation(evaluator(game), game)
    return UtilityBreakdown(value=float(value), hal_win_probability=0.0, baku_win_probability=0.0, unresolved_probability=0.0)


def _weighted_breakdown(parts: list[tuple[float, UtilityBreakdown]]) -> UtilityBreakdown:
    value = sum(weight * part.value for weight, part in parts)
    hal = sum(weight * part.hal_win_probability for weight, part in parts)
    baku = sum(weight * part.baku_win_probability for weight, part in parts)
    unresolved = sum(weight * part.unresolved_probability for weight, part in parts)
    return UtilityBreakdown(value, hal, baku, unresolved)


MatrixSolver = Literal["lp", "cfr_plus", "rust_cfr_plus"]


def _solve_role_strategies(
    hal_payoff: np.ndarray,
    *,
    hal_is_dropper: bool,
    solver: MatrixSolver = "lp",
    cfr_plus_config: CFRPlusConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    if solver == "lp":
        solve = solve_minimax
    elif solver == "cfr_plus":
        solve = lambda matrix: solve_cfr_plus(matrix, cfr_plus_config)
    elif solver == "rust_cfr_plus":
        solve = lambda matrix: solve_cfr_plus_rust(matrix, cfr_plus_config)
    else:
        raise ValueError(
            f"unknown matrix solver {solver!r}; expected 'lp', 'cfr_plus', or 'rust_cfr_plus'"
        )

    if hal_is_dropper:
        dropper_strategy, value_for_hal = solve(hal_payoff)
        checker_strategy, _ = solve((-hal_payoff).T)
    else:
        checker_strategy, value_for_hal = solve(hal_payoff.T)
        dropper_strategy, _ = solve(-hal_payoff)

    if solver in ("cfr_plus", "rust_cfr_plus"):
        value_for_hal = float(dropper_strategy @ hal_payoff @ checker_strategy)

    return dropper_strategy, checker_strategy, float(value_for_hal)


def _evaluate_joint_action_selective(
    game: Game,
    action: ExactJointAction,
    half_round_horizon: int,
    config: ExactSearchConfig,
    evaluator: LeafEvaluator | None = None,
    solver: MatrixSolver = "lp",
    cfr_plus_config: CFRPlusConfig | None = None,
) -> UtilityBreakdown:
    if game.game_over:
        return _terminal_breakdown(terminal_value(game, perspective_name=config.perspective_name))
    if half_round_horizon <= 0:
        return _frontier_breakdown(game, evaluator)

    snap = ExactGameSnapshot(game)
    probe = game.resolve_half_round(action.drop_time, action.check_time, survived_outcome=None)
    death_occurred = probe.survived is not None
    survival_probability = probe.survival_probability
    snap.restore(game)

    if not death_occurred:
        game.resolve_half_round(action.drop_time, action.check_time, survived_outcome=None)
        value = terminal_value(game, perspective_name=config.perspective_name)
        if value is not None or half_round_horizon == 1:
            part = _terminal_breakdown(value) if value is not None else _frontier_breakdown(game, evaluator)
        else:
            part = selective_solve(
                game,
                half_round_horizon - 1,
                config,
                evaluator=evaluator,
                solver=solver,
                cfr_plus_config=cfr_plus_config,
            ).breakdown
        snap.restore(game)
        return part

    assert survival_probability is not None
    parts: list[tuple[float, UtilityBreakdown]] = []
    for survived, probability in ((True, survival_probability), (False, 1.0 - survival_probability)):
        if probability <= 0.0:
            continue
        game.resolve_half_round(action.drop_time, action.check_time, survived_outcome=survived)
        value = terminal_value(game, perspective_name=config.perspective_name)
        if value is not None or half_round_horizon == 1:
            part = _terminal_breakdown(value) if value is not None else _frontier_breakdown(game, evaluator)
        else:
            part = selective_solve(
                game,
                half_round_horizon - 1,
                config,
                evaluator=evaluator,
                solver=solver,
                cfr_plus_config=cfr_plus_config,
            ).breakdown
        parts.append((probability, part))
        snap.restore(game)
    return _weighted_breakdown(parts)


def selective_solve(
    game: Game,
    half_round_horizon: int,
    config: ExactSearchConfig | None = None,
    *,
    candidates: CandidateActions | None = None,
    evaluator: LeafEvaluator | None = None,
    solver: MatrixSolver = "lp",
    cfr_plus_config: CFRPlusConfig | None = None,
) -> SelectiveSearchResult:
    """Selective candidate-only minimax over exact-second matrix games."""
    config = config or ExactSearchConfig()
    terminal = terminal_value(game, perspective_name=config.perspective_name)
    if terminal is not None or half_round_horizon <= 0:
        breakdown = _terminal_breakdown(terminal) if terminal is not None else _frontier_breakdown(game, evaluator)
        return SelectiveSearchResult(
            value_for_hal=breakdown.value,
            breakdown=breakdown,
            unresolved_probability=breakdown.unresolved_probability,
            half_round_horizon=half_round_horizon,
            drop_seconds=(),
            check_seconds=(),
            dropper_strategy=np.zeros(0),
            checker_strategy=np.zeros(0),
            payoff_for_hal=None,
            candidate_count=0,
        )

    if candidates is None:
        candidates = generate_candidates(game, config)
    drop_actions = candidates.drop_seconds
    check_actions = candidates.check_seconds
    d_index = {s: i for i, s in enumerate(drop_actions)}
    c_index = {s: i for i, s in enumerate(check_actions)}

    dropper, _checker = game.get_roles_for_half(game.current_half)
    hal_is_dropper = dropper.name.lower() == config.perspective_name.lower()

    hal_payoff = np.zeros((len(drop_actions), len(check_actions)), dtype=np.float64)
    breakdowns: dict[tuple[int, int], UtilityBreakdown] = {}

    for d in drop_actions:
        for c in check_actions:
            action = ExactJointAction(d, c)
            breakdown = _evaluate_joint_action_selective(
                game,
                action,
                half_round_horizon,
                config,
                evaluator=evaluator,
                solver=solver,
                cfr_plus_config=cfr_plus_config,
            )
            i = d_index[d]
            j = c_index[c]
            hal_payoff[i, j] = breakdown.value
            breakdowns[(i, j)] = breakdown

    # See exact.py: the maximizing player must sit on solver rows. Hal=dropper
    # maximizes hal_payoff; Hal=checker maximizes hal_payoff.T; Baku takes the
    # negated matrix for whichever role he plays.
    dropper_strategy, checker_strategy, value_for_hal = _solve_role_strategies(
        hal_payoff,
        hal_is_dropper=hal_is_dropper,
        solver=solver,
        cfr_plus_config=cfr_plus_config,
    )

    parts: list[tuple[float, UtilityBreakdown]] = []
    for i, dp in enumerate(dropper_strategy):
        if dp <= 0.0:
            continue
        for j, cp in enumerate(checker_strategy):
            weight = float(dp * cp)
            if weight > 0.0:
                parts.append((weight, breakdowns[(i, j)]))

    breakdown = _weighted_breakdown(parts)
    value = float(value_for_hal)
    return SelectiveSearchResult(
        value_for_hal=value,
        breakdown=breakdown,
        unresolved_probability=breakdown.unresolved_probability,
        half_round_horizon=half_round_horizon,
        drop_seconds=drop_actions,
        check_seconds=check_actions,
        dropper_strategy=dropper_strategy,
        checker_strategy=checker_strategy,
        payoff_for_hal=hal_payoff,
        candidate_count=len(drop_actions) * len(check_actions),
    )


def audit_against_full_width(
    game: Game,
    half_round_horizon: int,
    config: ExactSearchConfig | None = None,
) -> SelectiveAuditResult:
    """Run selective and full-width on the same state; report the value gap."""
    config = config or ExactSearchConfig()
    selective = selective_solve(game, half_round_horizon, config)
    full = solve_exact_finite_horizon(game, half_round_horizon, config)
    return SelectiveAuditResult(
        selective=selective,
        full_width_value=full.value_for_hal,
        value_gap=abs(selective.value_for_hal - full.value_for_hal),
        candidate_joint_count=selective.candidate_count,
        full_width_joint_count=len(full.drop_actions) * len(full.check_actions),
    )


# ============================================================================
# Subgame resolve
# ============================================================================

"""Subgame re-solve at critical states (Phase 6, Stockfish-style runtime polish).

The blueprint (trained value net + MCTS) covers the public-state space at
finite resolution. For *critical decisions* — within K half-rounds of the
leap window, near-overflow cylinder, about to flip the Active LSR parity
— we want a fresh local solve at deeper horizon at runtime rather than
relying on the net's prediction. This is the **Pluribus / Libratus pattern**
(Brown & Sandholm 2017, 2018) adapted to perfect-info: at critical positions,
do extra search anchored to the blueprint at the boundary.

Pure rigorous-core module: no shaping imports, no bucketing, no curriculum
labels. Reads timing predicates from ``timing_features`` and runs a fresh
``selective_solve`` at the requested horizon.
"""


from stl.engine.game import Game

from stl.solver.exact import ExactSearchConfig
from stl.solver.exact import (
    current_checker_fail_would_activate_lsr,
    is_active_lsr,
    rounds_until_leap_window,
)


__all__ = ["is_critical", "resolve_subgame"]


_NEAR_OVERFLOW_CYLINDER = 240.0
_LEAP_WINDOW_PROXIMITY = 2


def is_critical(game: Game) -> bool:
    """Heuristic gate for when subgame re-solving is worthwhile.

    Flags states near the leap window, near overflow, or about to flip
    LSR parity. The blueprint is most likely to misjudge these decisions
    because their consequences only become observable many half-rounds
    after the action.
    """
    if game.game_over:
        return False
    return any(
        (
            rounds_until_leap_window(game) <= _LEAP_WINDOW_PROXIMITY,
            current_checker_fail_would_activate_lsr(game),
            is_active_lsr(game) and game.current_half == 1,
            game.player1.cylinder >= _NEAR_OVERFLOW_CYLINDER,
            game.player2.cylinder >= _NEAR_OVERFLOW_CYLINDER,
        )
    )


def resolve_subgame(
    game: Game,
    horizon: int = 1,
    config: ExactSearchConfig | None = None,
    evaluator: LeafEvaluator | None = None,
    *,
    solver: MatrixSolver = "cfr_plus",
    cfr_plus_config: CFRPlusConfig | None = None,
) -> SelectiveSearchResult:
    """Fresh bounded selective_solve for a critical state.

    DeepStack-style use is ``horizon=1`` with a learned/tablebase evaluator at
    the frontier. That bounds critical resolves to the current half-round while
    preserving engine-derived chance and exact-second candidate actions.
    """
    config = config or ExactSearchConfig()
    return selective_solve(
        game,
        horizon,
        config,
        evaluator=evaluator,
        solver=solver,
        cfr_plus_config=cfr_plus_config,
    )


# ============================================================================
# Exact strategy diagnostics
# ============================================================================

"""Exploitability and best-response diagnostics for exact CFR results."""


from dataclasses import dataclass

import numpy as np

from stl.engine.game import Game

from stl.solver.exact import ExactSolveResult


@dataclass(frozen=True)
class ExactStrategyDiagnostics:
    """One-state best-response audit for a simultaneous exact matrix game."""

    expected_value: float
    dropper_best_response_value: float
    checker_best_response_value: float
    dropper_exploitability: float
    checker_exploitability: float
    nash_gap: float
    dropper_best_action: int | None
    checker_best_action: int | None


def _require_payoff(result: ExactSolveResult) -> np.ndarray:
    if result.payoff_for_hal is None:
        raise ValueError("ExactSolveResult has no payoff matrix; terminal states cannot be diagnosed")
    return result.payoff_for_hal


def diagnose_exact_strategy(
    game: Game,
    result: ExactSolveResult,
    *,
    perspective_name: str = "Hal",
) -> ExactStrategyDiagnostics:
    """Compute one-step exploitability from the exact Hal-perspective payoff matrix.

    The payoff matrix rows are dropper actions and columns are checker actions.
    If Hal is the dropper, rows maximize the payoff and columns minimize it.
    If Hal is the checker, rows minimize the payoff and columns maximize it.
    """
    payoff = _require_payoff(result)
    if payoff.shape != (len(result.dropper_strategy), len(result.checker_strategy)):
        raise ValueError("Strategy lengths do not match payoff matrix shape")

    dropper, _checker = game.get_roles_for_half(game.current_half)
    hal_is_dropper = dropper.name.lower() == perspective_name.lower()

    dropper_action_values = payoff @ result.checker_strategy
    checker_action_values = result.dropper_strategy @ payoff
    expected = float(result.dropper_strategy @ payoff @ result.checker_strategy)

    if hal_is_dropper:
        drop_idx = int(np.argmax(dropper_action_values))
        check_idx = int(np.argmin(checker_action_values))
        dropper_br = float(dropper_action_values[drop_idx])
        checker_br = float(checker_action_values[check_idx])
        dropper_exploitability = max(0.0, dropper_br - expected)
        checker_exploitability = max(0.0, expected - checker_br)
    else:
        drop_idx = int(np.argmin(dropper_action_values))
        check_idx = int(np.argmax(checker_action_values))
        dropper_br = float(dropper_action_values[drop_idx])
        checker_br = float(checker_action_values[check_idx])
        dropper_exploitability = max(0.0, expected - dropper_br)
        checker_exploitability = max(0.0, checker_br - expected)

    return ExactStrategyDiagnostics(
        expected_value=expected,
        dropper_best_response_value=dropper_br,
        checker_best_response_value=checker_br,
        dropper_exploitability=dropper_exploitability,
        checker_exploitability=checker_exploitability,
        nash_gap=dropper_exploitability + checker_exploitability,
        dropper_best_action=result.drop_actions[drop_idx] if result.drop_actions else None,
        checker_best_action=result.check_actions[check_idx] if result.check_actions else None,
    )


# ============================================================================
# Matrix-game MCTS
# ============================================================================

"""MCTS over exact-second matrix-game-at-node search for STL.

Each node stores: candidate seconds, a Q-matrix and visit counts, an
ExactGameSnapshot for restoring the engine, and a prior. Selection
solves the matrix game at the node on an exploration-augmented Q;
expansion samples chance branches via the engine; backup propagates
leaf values without sign flips (Hal-perspective everywhere).
"""

from dataclasses import dataclass

import numpy as np

from stl.engine.game import Game

from stl.solver.exact import ExactGameSnapshot, ExactJointAction, ExactSearchConfig
from stl.solver.exact import solve_minimax
from stl.solver.exact import terminal_value
from stl.solver.exact import ExactPublicState, exact_public_state


@dataclass
class MCTSNode:
    drop_seconds: tuple[int, ...]
    check_seconds: tuple[int, ...]
    game_snapshot: ExactGameSnapshot
    Q: np.ndarray
    N_cell: np.ndarray
    N_node: int
    is_expanded: bool
    prior: np.ndarray
    terminal_value: float | None
    children: dict[tuple[int, int, bool | None], "MCTSNode"]
    hal_is_dropper: bool


@dataclass
class MCTSConfig:
    iterations: int
    exploration_c: float
    evaluator: None
    use_tablebase: bool
    include_playable_grid: bool = False


@dataclass
class MCTSResult:
    root_strategy_dropper: np.ndarray
    root_strategy_checker: np.ndarray
    root_value_for_hal: float
    root_visits: int
    principal_line: list[ExactJointAction]
    cells_used: int
    # The seconds each strategy index refers to. When a critical-root
    # subgame resolve replaces the MCTS output these reflect the resolve's
    # action sets, so (seconds, strategy) always pair up.
    root_drop_seconds: tuple[int, ...] = ()
    root_check_seconds: tuple[int, ...] = ()
    # Average of the per-iteration root selection strategies — the object
    # SM-MCTS theory proves convergent (Lisy et al. 2013), and smoother
    # than a single LP vertex over mean-Q at small budgets where flat
    # matrices make the final LP degenerate (and arbitrarily PURE — an
    # exploitable tell). Empty when no root selections occurred.
    root_strategy_dropper_avg: np.ndarray = None  # type: ignore[assignment]
    root_strategy_checker_avg: np.ndarray = None  # type: ignore[assignment]


def _project_policy_to_candidates(policy: np.ndarray, candidates: tuple[int, ...]) -> np.ndarray:
    if not candidates:
        return np.zeros(0, dtype=np.float64)
    projected = np.array([policy[second - 1] for second in candidates], dtype=np.float64)
    projected = np.maximum(projected, 0.0)
    total = float(projected.sum())
    if total <= 1e-12:
        return np.full(len(candidates), 1.0 / len(candidates), dtype=np.float64)
    return projected / total


def _prior_from_evaluator(
    game: Game,
    drop_seconds: tuple[int, ...],
    check_seconds: tuple[int, ...],
    evaluator: LeafEvaluator | None,
) -> np.ndarray:
    prior, _ = _prior_and_value_from_evaluator(game, drop_seconds, check_seconds, evaluator)
    return prior


def _prior_and_value_from_evaluator(
    game: Game,
    drop_seconds: tuple[int, ...],
    check_seconds: tuple[int, ...],
    evaluator: LeafEvaluator | None,
) -> tuple[np.ndarray, float]:
    """Single evaluator call yielding both the joint prior and the leaf value.

    The value seeds Q for never-visited cells so they enter the selection
    and root LPs at the node's estimated value rather than a synthetic 0.0
    (which sits at the midpoint of the value scale and mispriced
    off-equilibrium cells permanently).
    """
    D = len(drop_seconds)
    C = len(check_seconds)
    if D == 0 or C == 0:
        return np.zeros((0, 0), dtype=np.float64), 0.0
    if evaluator is None:
        return np.full((D, C), 1.0 / (D * C), dtype=np.float64), 0.0

    value, dropper_policy, checker_policy = normalize_leaf_evaluation(evaluator(game), game)
    drop_prior = _project_policy_to_candidates(dropper_policy, drop_seconds)
    check_prior = _project_policy_to_candidates(checker_policy, check_seconds)
    return np.outer(drop_prior, check_prior).astype(np.float64), float(value)


def make_node(
    game: Game,
    config: ExactSearchConfig | None = None,
    evaluator: LeafEvaluator | None = None,
    *,
    include_playable_grid: bool = False,
) -> MCTSNode:
    """Build an MCTSNode at the current game state.

    Snapshots the engine state, generates candidates, allocates zero
    Q/N matrices. Sets terminal_value if the game is already over.
    """
    config = config or ExactSearchConfig()
    tval = terminal_value(game=game, perspective_name=config.perspective_name)

    if game.game_over:
        return MCTSNode(
            drop_seconds=(),
            check_seconds=(),
            game_snapshot=ExactGameSnapshot(game=game),
            Q=np.zeros((0, 0), dtype=np.float64),
            N_cell=np.zeros((0, 0), dtype=np.int64),
            N_node=0,
            is_expanded=False,
            prior=np.zeros((0, 0), dtype=np.float64),
            terminal_value=tval,
            children={},
            hal_is_dropper=False,
        )

    dropper, _ = game.get_roles_for_half(game.current_half)
    hdrop = dropper.name.lower() == config.perspective_name.lower()
    cands = generate_candidates(game, config, include_playable_grid=include_playable_grid)
    D = len(cands.drop_seconds)
    C = len(cands.check_seconds)

    prior, leaf_value = _prior_and_value_from_evaluator(
        game, cands.drop_seconds, cands.check_seconds, evaluator
    )

    return MCTSNode(
        drop_seconds=cands.drop_seconds,
        check_seconds=cands.check_seconds,
        game_snapshot=ExactGameSnapshot(game=game),
        Q=np.full((D, C), leaf_value, dtype=np.float64),
        N_cell=np.zeros((D, C), dtype=np.int64),
        N_node=0,
        is_expanded=False,
        prior=prior,
        terminal_value=tval,
        children={},
        hal_is_dropper=hdrop,
    )


def _exploration_augmented_matrix(node: MCTSNode, exploration_c: float) -> np.ndarray:
    """Return Q + c * P * sqrt(N_node) / (1 + N_cell). Shape (D, C)."""
    U = exploration_c * node.prior * np.sqrt(node.N_node) / (1 + node.N_cell)
    return node.Q + U


def _selection_strategies(
    node: MCTSNode,
    exploration_c: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-player equilibrium strategies of the optimism-augmented node
    matrices (each player optimistic in its own direction)."""
    U = exploration_c * node.prior * np.sqrt(node.N_node) / (1 + node.N_cell)
    Q_max_optimistic = node.Q + U
    Q_min_optimistic = -(node.Q - U)

    if node.hal_is_dropper:
        dropper_strategy, _ = solve_minimax(Q_max_optimistic)
        checker_strategy, _ = solve_minimax(Q_min_optimistic.T)
    else:
        dropper_strategy, _ = solve_minimax(Q_min_optimistic)
        checker_strategy, _ = solve_minimax(Q_max_optimistic.T)
    return dropper_strategy, checker_strategy


def _select_joint_action(
    node: MCTSNode,
    exploration_c: float,
    rng: np.random.Generator,
) -> tuple[int, int]:
    """Sample (d_idx, c_idx) from the matrix-game equilibrium of the
    exploration-augmented Q at this node.

    Each player solves the saddle point of the matrix that is optimistic in
    its OWN direction (O'Donoghue, Lattimore & Osband, UAI 2021): the
    Hal-perspective maximizer sees Q + U; the minimizer's payoff is -Q, so
    its optimistic matrix is -Q + U = -(Q - U). Giving both players the
    same Q + U makes the bonus repel the minimizer from under-visited
    cells, breaking the guaranteed-exploration property.
    """
    D = len(node.drop_seconds)
    C = len(node.check_seconds)
    dropper_strategy, checker_strategy = _selection_strategies(node, exploration_c)

    d_idx = rng.choice(D, p=dropper_strategy)
    c_idx = rng.choice(C, p=checker_strategy)
    return int(d_idx), int(c_idx)


def _step_into_child(
    node: MCTSNode,
    game: Game,
    d_idx: int,
    c_idx: int,
    rng: np.random.Generator,
    config: ExactSearchConfig,
    transposition: dict[ExactPublicState, MCTSNode] | None = None,
    evaluator: LeafEvaluator | None = None,
) -> tuple["MCTSNode", bool | None]:
    """Apply the joint action at (d_idx, c_idx) to the engine, sample the
    chance outcome if a death is possible, and return the child node along
    with the survival flag (True/False/None). Caches the child in
    node.children so subsequent iterations reuse it.
    """
    d_time = node.drop_seconds[d_idx]
    c_time = node.check_seconds[c_idx]

    snap = ExactGameSnapshot(game)
    probe = game.resolve_half_round(d_time, c_time, survived_outcome=None)
    death_possible = probe.survived is not None
    survival_probability = probe.survival_probability
    snap.restore(game)

    survived_outcome: bool | None = None
    if death_possible:
        survived_outcome = bool(rng.random() < survival_probability)

    key = (d_time, c_time, survived_outcome)
    game.resolve_half_round(d_time, c_time, survived_outcome)

    if key not in node.children:
        state_key = exact_public_state(game)
        cached = transposition is not None and state_key in transposition
        if cached:
            node.children[key] = transposition[state_key]
        else:
            child = make_node(game, config, evaluator=evaluator, include_playable_grid=False)
            node.children[key] = child
            if transposition is not None:
                transposition[state_key] = child

    return node.children[key], survived_outcome


def _backup(path: list[tuple[MCTSNode, int, int]], value: float) -> None:
    """Apply value to every (node, d_idx, c_idx) in path.

    Increments N_node, N_cell[d_idx, c_idx], and updates Q[d_idx, c_idx]
    with a running-mean step. Mutates the nodes in place.
    """
    for (node, d_idx, c_idx) in path:
        node.N_node += 1
        node.N_cell[d_idx, c_idx] += 1
        node.Q[d_idx, c_idx] += (value - node.Q[d_idx, c_idx]) / node.N_cell[d_idx, c_idx]


def mcts_search(
    game: Game,
    config: MCTSConfig,
    evaluator: LeafEvaluator,
    rng: np.random.Generator,
    exact_config: ExactSearchConfig | None = None,
    subgame_resolve_at_critical: bool = False,
    subgame_resolve_horizon: int = 1,
    subgame_resolve_solver: MatrixSolver = "cfr_plus",
    subgame_resolve_cfr_plus_config: CFRPlusConfig | None = None,
) -> MCTSResult:
    """Run MCTS for config.iterations iterations from the current game state.

    Returns the root's equilibrium strategies, the Hal-perspective value,
    and visit-count diagnostics.

    When ``subgame_resolve_at_critical`` is True and ``is_critical`` flags the
    root state, the MCTS run is supplemented by a bounded root re-solve whose
    default horizon is the current half-round. The boundary is anchored by the
    supplied evaluator, matching the DeepStack pattern without unbounded
    terminal search. Default ``False`` preserves existing behavior.
    """
    exact_config = exact_config or ExactSearchConfig()
    root = make_node(
        game,
        exact_config,
        evaluator=evaluator,
        include_playable_grid=config.include_playable_grid,
    )
    c = config.exploration_c

    transposition : dict[ExactPublicState, MCTSNode] = {}
    transposition[exact_public_state(game)] = root

    root_drop_sum = np.zeros(len(root.drop_seconds), dtype=np.float64)
    root_check_sum = np.zeros(len(root.check_seconds), dtype=np.float64)
    root_strat_count = 0

    for _ in range(config.iterations):
        root.game_snapshot.restore(game=game)
        node = root
        path: list[tuple[MCTSNode, int, int]] = []
        while True:
            if node.terminal_value is not None:
                leaf_value = node.terminal_value
                break
            if not node.is_expanded:
                leaf_value, _, _ = normalize_leaf_evaluation(evaluator(game), game)
                node.is_expanded = True
                break
            if node is root:
                drop_strat, check_strat = _selection_strategies(node, c)
                root_drop_sum += drop_strat
                root_check_sum += check_strat
                root_strat_count += 1
                d_idx = int(rng.choice(len(node.drop_seconds), p=drop_strat))
                c_idx = int(rng.choice(len(node.check_seconds), p=check_strat))
            else:
                d_idx, c_idx = _select_joint_action(node, c, rng)
            path.append((node, d_idx, c_idx))
            node, _ = _step_into_child(
                node,
                game,
                d_idx,
                c_idx,
                rng,
                exact_config,
                transposition=transposition,
                evaluator=evaluator,
            )
        _backup(path, leaf_value)

    Q = root.Q
    if root.hal_is_dropper:
        dropper_strat, _ = solve_minimax(Q)
        checker_strat, _ = solve_minimax((-Q).T)
    else:
        dropper_strat, _ = solve_minimax(-Q)
        checker_strat, _ = solve_minimax(Q.T)

    value_for_hal = float(dropper_strat @ Q @ checker_strat)

    principal_line = _principal_line(root)
    root_drop_seconds = root.drop_seconds
    root_check_seconds = root.check_seconds
    if root_strat_count > 0:
        dropper_avg = root_drop_sum / root_strat_count
        checker_avg = root_check_sum / root_strat_count
    else:
        dropper_avg = np.asarray(dropper_strat, dtype=np.float64).copy()
        checker_avg = np.asarray(checker_strat, dtype=np.float64).copy()

    if subgame_resolve_at_critical:
        root.game_snapshot.restore(game=game)
        if is_critical(game):
            resolve_result = resolve_subgame(
                game,
                horizon=subgame_resolve_horizon,
                config=exact_config,
                evaluator=evaluator,
                solver=subgame_resolve_solver,
                cfr_plus_config=subgame_resolve_cfr_plus_config,
            )
            dropper_strat = resolve_result.dropper_strategy
            checker_strat = resolve_result.checker_strategy
            value_for_hal = float(resolve_result.value_for_hal)
            root_drop_seconds = tuple(resolve_result.drop_seconds)
            root_check_seconds = tuple(resolve_result.check_seconds)
            dropper_avg = np.asarray(dropper_strat, dtype=np.float64).copy()
            checker_avg = np.asarray(checker_strat, dtype=np.float64).copy()
            if (
                len(resolve_result.drop_seconds) > 0
                and len(resolve_result.check_seconds) > 0
                and dropper_strat.size > 0
                and checker_strat.size > 0
            ):
                d_idx = int(np.argmax(dropper_strat))
                c_idx = int(np.argmax(checker_strat))
                principal_line = [
                    ExactJointAction(
                        resolve_result.drop_seconds[d_idx],
                        resolve_result.check_seconds[c_idx],
                    )
                ]
            else:
                principal_line = []
        root.game_snapshot.restore(game=game)

    root.game_snapshot.restore(game=game)

    return MCTSResult(
        root_strategy_dropper=dropper_strat,
        root_strategy_checker=checker_strat,
        root_value_for_hal=value_for_hal,
        root_visits=root.N_node,
        principal_line=principal_line,
        cells_used=int(root.N_cell.sum()),
        root_drop_seconds=root_drop_seconds,
        root_check_seconds=root_check_seconds,
        root_strategy_dropper_avg=dropper_avg,
        root_strategy_checker_avg=checker_avg,
    )


def _principal_line(root: MCTSNode) -> list[ExactJointAction]:
    """most-visited path in tree"""
    line = []
    node = root
    while True:

        if node.terminal_value is not None or node.N_node == 0 or node.N_cell.max() == 0:
            break
        C = len(node.check_seconds)
        flat_idx = int(np.argmax(node.N_cell))
        d_idx, c_idx = divmod(flat_idx, C)
        d_time, c_time = node.drop_seconds[d_idx], node.check_seconds[c_idx]
        line.append(ExactJointAction(d_time, c_time))

        candidate_keys = [k for k in node.children if k[0] == d_time and k[1] == c_time]
        if not candidate_keys:
            break
        next_key = max(candidate_keys, key = lambda k: node.children[k].N_node)

        node = node.children[next_key]

    return line
