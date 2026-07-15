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

from stl.engine.actions import ACTION_SIZE, legal_max_second

from stl.solver.exact import ExactPublicState, exact_public_state, terminal_value
from stl.engine.game import Game


LeafEvaluation = tuple[float, np.ndarray, np.ndarray]


def _uniform_over_legal(max_second: int) -> np.ndarray:
    dist = np.zeros(ACTION_SIZE, dtype=np.float64)
    if max_second < 1:
        return dist
    dist[1 : max_second + 1] = 1.0 / max_second
    return dist


def uniform_policy_for_current_roles(game: Game) -> tuple[np.ndarray, np.ndarray]:
    """Return length-62 uniform distributions over legal root seconds."""
    if game.game_over:
        return np.zeros(ACTION_SIZE, dtype=np.float64), np.zeros(ACTION_SIZE, dtype=np.float64)

    dropper, checker = game.get_roles_for_half(game.current_half)
    turn_duration = game.get_turn_duration()
    drop_max = legal_max_second(dropper.name, "dropper", turn_duration)
    check_max = legal_max_second(checker.name, "checker", turn_duration)
    return _uniform_over_legal(drop_max), _uniform_over_legal(check_max)


def normalize_policy_vector(policy: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    arr = np.asarray(policy, dtype=np.float64).reshape(-1)
    if arr.shape[0] != ACTION_SIZE:
        raise ValueError(f"policy vector must have length {ACTION_SIZE}, got {arr.shape[0]}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("policy vector must contain only finite values")
    if np.any(arr < 0.0):
        raise ValueError("policy vector must be nonnegative")
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
    def __call__(
        self, game: Game, *, value_horizon: int | None = None
    ) -> LeafEvaluation: ...


def evaluate_leaf(
    evaluator: LeafEvaluator | None,
    game: Game,
    *,
    value_horizon: int | None = None,
) -> LeafEvaluation:
    """Evaluate a leaf while preserving scalar legacy test evaluators.

    Horizon-aware evaluators opt in with ``supports_horizon_context``. This
    avoids catching arbitrary ``TypeError`` exceptions from evaluator bodies
    while the old P3 terminal fixtures remain source compatible.
    """

    if evaluator is None:
        dropper_policy, checker_policy = uniform_policy_for_current_roles(game)
        return 0.0, dropper_policy, checker_policy
    if bool(getattr(evaluator, "supports_horizon_context", False)):
        result = evaluator(game, value_horizon=value_horizon)
    else:
        result = evaluator(game)  # type: ignore[call-arg]
    return normalize_leaf_evaluation(result, game)


class TerminalOnlyEvaluator:
    """Returns terminal_value(game) for terminal positions, 0.0 otherwise."""

    def __init__(self, perspective_name: str = "Hal") -> None:
        self.perspective = perspective_name

    supports_horizon_context = True

    def __call__(
        self, game: Game, *, value_horizon: int | None = None
    ) -> LeafEvaluation:
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

    supports_horizon_context = True

    def __call__(
        self, game: Game, *, value_horizon: int | None = None
    ) -> LeafEvaluation:
        key = exact_public_state(game)
        if key in self._table:
            dropper_policy, checker_policy = uniform_policy_for_current_roles(game)
            return float(self._table[key]), dropper_policy, checker_policy
        return evaluate_leaf(
            self.fallback, game, value_horizon=value_horizon
        )


class ValueNetEvaluator:
    """Wraps the trained value net behind a clean interface."""

    supports_horizon_context = True

    def __init__(
        self,
        model_fn: Callable[..., float],
        *,
        default_value_horizon: int | None = None,
    ) -> None:
        self.model = model_fn
        self.default_value_horizon = default_value_horizon

    def __call__(
        self, game: Game, *, value_horizon: int | None = None
    ) -> LeafEvaluation:
        horizon = (
            self.default_value_horizon
            if value_horizon is None
            else value_horizon
        )
        if horizon is None:
            raise ValueError("ValueNetEvaluator requires an explicit value horizon")
        return normalize_leaf_evaluation(
            self.model(game, horizon=horizon), game
        )


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
FULL_WIDTH_SAFE_ST_BAND = 30


@dataclass(frozen=True)
class CandidateActions:
    drop_seconds: tuple[int, ...]
    check_seconds: tuple[int, ...]

    @property
    def joint_count(self) -> int:
        return len(self.drop_seconds) * len(self.check_seconds)


def overflow_st_threshold(checker_cylinder: float) -> int:
    """Smallest ST that drives the cylinder to CYLINDER_MAX or above."""
    return max(0, int(CYLINDER_MAX) - int(checker_cylinder))


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

    # The near-overflow marginal band has a long cyclic best-response chain:
    # each newly added boundary induces another +/-1 response.  The frozen P1
    # omitted-action audit found profitable deviations when this band was
    # truncated, so use literal full width instead of pretending the candidate
    # subset is reliable.  At safe_st=0 every successful check is already an
    # overflow and the compact terminal candidate set remains sufficient.
    if 0 < safe_st <= FULL_WIDTH_SAFE_ST_BAND:
        return CandidateActions(
            drop_seconds=tuple(drop_legal),
            check_seconds=tuple(check_legal),
        )

    def add_checker_responses() -> None:
        for d in tuple(drop_seconds):
            for c in (
                d - 1, d, d + 1,
                d + safe_st, d + safe_st + 1,
                d + overflow_st - 1, d + overflow_st,
            ):
                if c in check_legal:
                    check_seconds.add(c)

    def add_dropper_responses() -> None:
        for c in tuple(check_seconds):
            for d in (
                c - 1, c, c + 1,
                c - safe_st - 1, c - safe_st,
                c - overflow_st, c - overflow_st + 1,
            ):
                if d in drop_legal:
                    drop_seconds.add(d)

    # Two checker passes bracket one dropper pass.  The final checker pass is
    # necessary because the middle pass can introduce a boundary such as
    # drop=9 whose matching check=9 was not in the original seed set.  Taking
    # this to an unrestricted fixed point would let the +/-1 relation flood
    # the entire 60x60 grid and defeat selective search.
    add_checker_responses()
    add_dropper_responses()
    add_checker_responses()

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
    dropper_omitted_action_gain: float
    checker_omitted_action_gain: float
    max_omitted_action_gain: float


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
    value, _, _ = evaluate_leaf(evaluator, game)
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
    """Compare candidate search with the full-width one-step best responses.

    Value agreement alone can hide an exploitable candidate policy.  The
    omitted-action gains embed both candidate marginals in the full literal-
    second matrix and ask how much either role could gain by deviating to an
    action that candidate generation left out.
    """
    config = config or ExactSearchConfig()
    selective = selective_solve(game, half_round_horizon, config)
    full = solve_exact_finite_horizon(game, half_round_horizon, config)
    if full.payoff_for_hal is None:
        raise ValueError("full-width audit requires a non-terminal payoff matrix")

    full_drop_index = {second: i for i, second in enumerate(full.drop_actions)}
    full_check_index = {second: i for i, second in enumerate(full.check_actions)}
    drop_policy = np.zeros(len(full.drop_actions), dtype=np.float64)
    check_policy = np.zeros(len(full.check_actions), dtype=np.float64)
    for second, probability in zip(selective.drop_seconds, selective.dropper_strategy):
        drop_policy[full_drop_index[second]] = float(probability)
    for second, probability in zip(selective.check_seconds, selective.checker_strategy):
        check_policy[full_check_index[second]] = float(probability)

    payoff = np.asarray(full.payoff_for_hal, dtype=np.float64)
    expected = float(drop_policy @ payoff @ check_policy)
    drop_values = payoff @ check_policy
    check_values = drop_policy @ payoff
    omitted_drop = [
        i for i, second in enumerate(full.drop_actions)
        if second not in set(selective.drop_seconds)
    ]
    omitted_check = [
        i for i, second in enumerate(full.check_actions)
        if second not in set(selective.check_seconds)
    ]

    dropper, _checker = game.get_roles_for_half(game.current_half)
    hal_is_dropper = dropper.name.lower() == config.perspective_name.lower()
    if omitted_drop:
        omitted_values = drop_values[omitted_drop]
        drop_gain = (
            float(np.max(omitted_values)) - expected
            if hal_is_dropper
            else expected - float(np.min(omitted_values))
        )
    else:
        drop_gain = 0.0
    if omitted_check:
        omitted_values = check_values[omitted_check]
        check_gain = (
            expected - float(np.min(omitted_values))
            if hal_is_dropper
            else float(np.max(omitted_values)) - expected
        )
    else:
        check_gain = 0.0
    drop_gain = max(0.0, drop_gain)
    check_gain = max(0.0, check_gain)

    return SelectiveAuditResult(
        selective=selective,
        full_width_value=full.value_for_hal,
        value_gap=abs(selective.value_for_hal - full.value_for_hal),
        candidate_joint_count=selective.candidate_count,
        full_width_joint_count=len(full.drop_actions) * len(full.check_actions),
        dropper_omitted_action_gain=drop_gain,
        checker_omitted_action_gain=check_gain,
        max_omitted_action_gain=max(drop_gain, check_gain),
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
    dropper_prior: np.ndarray
    checker_prior: np.ndarray
    prior: np.ndarray
    terminal_value: float | None
    children: dict[tuple[int, int, bool | None], "MCTSNode"]
    hal_is_dropper: bool
    value_horizon: int | None


MCTSActionMode = Literal["candidate", "candidate_playable", "full_width"]


@dataclass(frozen=True)
class MCTSConfig:
    iterations: int
    exploration_c: float
    action_mode: MCTSActionMode = "candidate"
    root_noise_epsilon: float = 0.0
    root_dirichlet_alpha_scale: float = 10.0
    max_depth: int | None = None
    root_value_horizon: int | None = None

    def __post_init__(self) -> None:
        if self.iterations <= 0:
            raise ValueError("MCTS iterations must be positive")
        if not np.isfinite(self.exploration_c) or self.exploration_c < 0.0:
            raise ValueError("MCTS exploration_c must be finite and nonnegative")
        if self.action_mode not in ("candidate", "candidate_playable", "full_width"):
            raise ValueError(f"unknown MCTS action_mode {self.action_mode!r}")
        if self.root_value_horizon is not None and self.root_value_horizon <= 0:
            raise ValueError("root_value_horizon must be positive when provided")
        if (
            self.root_value_horizon is not None
            and self.max_depth is not None
            and self.root_value_horizon < self.max_depth
        ):
            raise ValueError(
                "root_value_horizon must cover every configured MCTS depth"
            )
        if not 0.0 <= self.root_noise_epsilon <= 1.0:
            raise ValueError("root_noise_epsilon must be in [0, 1]")
        if (
            not np.isfinite(self.root_dirichlet_alpha_scale)
            or self.root_dirichlet_alpha_scale <= 0.0
        ):
            raise ValueError("root_dirichlet_alpha_scale must be finite and positive")
        if self.max_depth is not None and self.max_depth <= 0:
            raise ValueError("max_depth must be positive when provided")


@dataclass
class MCTSResult:
    improved_dropper_policy: np.ndarray
    improved_checker_policy: np.ndarray
    root_value_for_hal: float
    mean_q_dropper_policy: np.ndarray
    mean_q_checker_policy: np.ndarray
    mean_q_value_for_hal: float
    root_visits: int
    principal_line: list[ExactJointAction]
    cells_used: int
    root_unique_cells_visited: int
    action_mode: str
    # The seconds each strategy index refers to. When a critical-root
    # subgame resolve replaces the MCTS output these reflect the resolve's
    # action sets, so (seconds, strategy) always pair up.
    root_drop_seconds: tuple[int, ...] = ()
    root_check_seconds: tuple[int, ...] = ()
    # Linearly weighted average of per-iteration empirical mean-Q equilibria.
    # This is the canonical improvement target; optimistic policies are used
    # only to choose exploratory samples.

    @property
    def root_strategy_dropper(self) -> np.ndarray:
        """Deprecated alias for the final mean-Q diagnostic policy."""

        return self.mean_q_dropper_policy

    @property
    def root_strategy_checker(self) -> np.ndarray:
        """Deprecated alias for the final mean-Q diagnostic policy."""

        return self.mean_q_checker_policy

    @property
    def root_strategy_dropper_avg(self) -> np.ndarray:
        """Deprecated alias for the canonical improved dropper policy."""

        return self.improved_dropper_policy

    @property
    def root_strategy_checker_avg(self) -> np.ndarray:
        """Deprecated alias for the canonical improved checker policy."""

        return self.improved_checker_policy


def _project_policy_to_candidates(policy: np.ndarray, candidates: tuple[int, ...]) -> np.ndarray:
    if not candidates:
        return np.zeros(0, dtype=np.float64)
    projected = np.array([policy[second] for second in candidates], dtype=np.float64)
    projected = np.maximum(projected, 0.0)
    total = float(projected.sum())
    if total <= 1e-12:
        return np.full(len(candidates), 1.0 / len(candidates), dtype=np.float64)
    return projected / total


def _validate_legal_role_policy(
    policy: np.ndarray,
    legal_seconds: tuple[int, ...],
    *,
    role: str,
) -> None:
    legal = np.zeros(ACTION_SIZE, dtype=bool)
    legal[list(legal_seconds)] = True
    if not np.all(np.isfinite(policy)):
        raise ValueError(f"{role} prior contains non-finite values")
    if np.any(policy < 0.0):
        raise ValueError(f"{role} prior contains negative values")
    if float(policy[~legal].sum()) > 1e-10:
        raise ValueError(f"{role} prior assigns mass to illegal seconds")
    if not np.isclose(float(policy.sum()), 1.0, atol=1e-8):
        raise ValueError(f"{role} prior must sum to one")


def _validate_distribution(policy: np.ndarray, size: int, *, name: str) -> np.ndarray:
    result = np.asarray(policy, dtype=np.float64).reshape(-1)
    if result.shape != (size,):
        raise ValueError(f"{name} shape {result.shape} does not match {(size,)}")
    if not np.all(np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError(f"{name} must be finite and nonnegative")
    total = float(result.sum())
    if total <= 1e-12:
        raise ValueError(f"{name} has zero mass")
    result = result / total
    if not np.isclose(float(result.sum()), 1.0, atol=1e-8):
        raise ValueError(f"{name} failed normalization")
    return result


def _prior_from_evaluator(
    game: Game,
    drop_seconds: tuple[int, ...],
    check_seconds: tuple[int, ...],
    evaluator: LeafEvaluator | None,
    value_horizon: int | None = None,
) -> np.ndarray:
    _drop, _check, prior, _value = _prior_and_value_from_evaluator(
        game, drop_seconds, check_seconds, evaluator, value_horizon=value_horizon
    )
    return prior


def _prior_and_value_from_evaluator(
    game: Game,
    drop_seconds: tuple[int, ...],
    check_seconds: tuple[int, ...],
    evaluator: LeafEvaluator | None,
    *,
    value_horizon: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Single evaluator call yielding both the joint prior and the leaf value.

    The value seeds Q for never-visited cells so they enter the selection
    and root LPs at the node's estimated value rather than a synthetic 0.0
    (which sits at the midpoint of the value scale and mispriced
    off-equilibrium cells permanently).
    """
    D = len(drop_seconds)
    C = len(check_seconds)
    if D == 0 or C == 0:
        return (
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros((0, 0), dtype=np.float64),
            0.0,
        )
    if evaluator is None:
        drop_prior = np.full(D, 1.0 / D, dtype=np.float64)
        check_prior = np.full(C, 1.0 / C, dtype=np.float64)
        return drop_prior, check_prior, np.outer(drop_prior, check_prior), 0.0

    value, dropper_policy, checker_policy = evaluate_leaf(
        evaluator, game, value_horizon=value_horizon
    )
    dropper, checker = game.get_roles_for_half(game.current_half)
    exact_config = ExactSearchConfig()
    legal_drop = legal_seconds_for_current_role(game, dropper.name, "dropper", exact_config)
    legal_check = legal_seconds_for_current_role(game, checker.name, "checker", exact_config)
    _validate_legal_role_policy(dropper_policy, legal_drop, role="dropper")
    _validate_legal_role_policy(checker_policy, legal_check, role="checker")
    drop_prior = _project_policy_to_candidates(dropper_policy, drop_seconds)
    check_prior = _project_policy_to_candidates(checker_policy, check_seconds)
    drop_prior = _validate_distribution(drop_prior, D, name="dropper prior")
    check_prior = _validate_distribution(check_prior, C, name="checker prior")
    return drop_prior, check_prior, np.outer(drop_prior, check_prior), float(value)


def _seconds_for_action_mode(
    game: Game,
    exact_config: ExactSearchConfig,
    action_mode: MCTSActionMode,
) -> CandidateActions:
    if action_mode == "full_width":
        dropper, checker = game.get_roles_for_half(game.current_half)
        return CandidateActions(
            drop_seconds=legal_seconds_for_current_role(
                game, dropper.name, "dropper", exact_config
            ),
            check_seconds=legal_seconds_for_current_role(
                game, checker.name, "checker", exact_config
            ),
        )
    return generate_candidates(
        game,
        exact_config,
        include_playable_grid=action_mode == "candidate_playable",
    )


def make_node(
    game: Game,
    config: ExactSearchConfig | None = None,
    evaluator: LeafEvaluator | None = None,
    *,
    action_mode: MCTSActionMode = "candidate",
    value_horizon: int | None = None,
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
            dropper_prior=np.zeros(0, dtype=np.float64),
            checker_prior=np.zeros(0, dtype=np.float64),
            prior=np.zeros((0, 0), dtype=np.float64),
            terminal_value=tval,
            children={},
            hal_is_dropper=False,
            value_horizon=value_horizon,
        )

    dropper, _ = game.get_roles_for_half(game.current_half)
    hdrop = dropper.name.lower() == config.perspective_name.lower()
    cands = _seconds_for_action_mode(game, config, action_mode)
    D = len(cands.drop_seconds)
    C = len(cands.check_seconds)

    dropper_prior, checker_prior, prior, leaf_value = _prior_and_value_from_evaluator(
        game,
        cands.drop_seconds,
        cands.check_seconds,
        evaluator,
        value_horizon=value_horizon,
    )

    return MCTSNode(
        drop_seconds=cands.drop_seconds,
        check_seconds=cands.check_seconds,
        game_snapshot=ExactGameSnapshot(game=game),
        Q=np.full((D, C), leaf_value, dtype=np.float64),
        N_cell=np.zeros((D, C), dtype=np.int64),
        N_node=0,
        is_expanded=False,
        dropper_prior=dropper_prior,
        checker_prior=checker_prior,
        prior=prior,
        terminal_value=tval,
        children={},
        hal_is_dropper=hdrop,
        value_horizon=value_horizon,
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


def _mean_q_strategies(node: MCTSNode) -> tuple[np.ndarray, np.ndarray]:
    """Current simultaneous equilibrium of empirical mean-Q.

    This is accumulated for the policy-improvement target.  Optimistic
    strategies remain responsible for selecting samples, but their exploration
    bonuses are not mislabeled as policy probability.
    """

    if node.hal_is_dropper:
        dropper_strategy, _ = solve_minimax(node.Q)
        checker_strategy, _ = solve_minimax((-node.Q).T)
    else:
        dropper_strategy, _ = solve_minimax(-node.Q)
        checker_strategy, _ = solve_minimax(node.Q.T)
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
    transposition: dict[tuple[ExactPublicState, int | None], MCTSNode] | None = None,
    evaluator: LeafEvaluator | None = None,
    action_mode: MCTSActionMode = "candidate",
    value_horizon: int | None = None,
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
        state_key = (exact_public_state(game), value_horizon)
        cached = transposition is not None and state_key in transposition
        if cached:
            node.children[key] = transposition[state_key]
        else:
            child = make_node(
                game,
                config,
                evaluator=evaluator,
                action_mode=action_mode,
                value_horizon=value_horizon,
            )
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
    root_noise_rng: np.random.Generator | None = None,
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
        action_mode=config.action_mode,
        value_horizon=config.root_value_horizon,
    )
    c = config.exploration_c

    if config.root_noise_epsilon > 0.0:
        if root_noise_rng is None:
            raise ValueError("root_noise_rng is required when root noise is enabled")
        if root.dropper_prior.size and root.checker_prior.size:
            drop_noise = root_noise_rng.dirichlet(
                np.full(
                    root.dropper_prior.size,
                    config.root_dirichlet_alpha_scale / root.dropper_prior.size,
                )
            )
            check_noise = root_noise_rng.dirichlet(
                np.full(
                    root.checker_prior.size,
                    config.root_dirichlet_alpha_scale / root.checker_prior.size,
                )
            )
            epsilon = config.root_noise_epsilon
            root.dropper_prior = _validate_distribution(
                (1.0 - epsilon) * root.dropper_prior + epsilon * drop_noise,
                root.dropper_prior.size,
                name="noisy root dropper prior",
            )
            root.checker_prior = _validate_distribution(
                (1.0 - epsilon) * root.checker_prior + epsilon * check_noise,
                root.checker_prior.size,
                name="noisy root checker prior",
            )
            root.prior = np.outer(root.dropper_prior, root.checker_prior)

    transposition: dict[tuple[ExactPublicState, int | None], MCTSNode] = {}
    transposition[(exact_public_state(game), config.root_value_horizon)] = root

    root_drop_sum = np.zeros(len(root.drop_seconds), dtype=np.float64)
    root_check_sum = np.zeros(len(root.check_seconds), dtype=np.float64)
    root_strat_count = 0
    root_strat_weight = 0.0

    for _ in range(config.iterations):
        root.game_snapshot.restore(game=game)
        node = root
        depth = 0
        path: list[tuple[MCTSNode, int, int]] = []
        while True:
            if node.terminal_value is not None:
                leaf_value = node.terminal_value
                break
            if not node.is_expanded:
                leaf_value, _, _ = evaluate_leaf(
                    evaluator, game, value_horizon=node.value_horizon
                )
                node.is_expanded = True
                break
            if config.max_depth is not None and depth >= config.max_depth:
                leaf_value, _, _ = evaluate_leaf(
                    evaluator, game, value_horizon=node.value_horizon
                )
                break
            if node is root:
                improved_drop, improved_check = _mean_q_strategies(node)
                weight = float(root_strat_count + 1)
                root_drop_sum += weight * improved_drop
                root_check_sum += weight * improved_check
                root_strat_weight += weight
                root_strat_count += 1
                selection_drop, selection_check = _selection_strategies(node, c)
                d_idx = int(rng.choice(len(node.drop_seconds), p=selection_drop))
                c_idx = int(rng.choice(len(node.check_seconds), p=selection_check))
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
                action_mode=config.action_mode,
                value_horizon=(
                    None
                    if node.value_horizon is None
                    else node.value_horizon - 1
                ),
            )
            depth += 1
        _backup(path, leaf_value)

    Q = root.Q
    if root.hal_is_dropper:
        dropper_strat, _ = solve_minimax(Q)
        checker_strat, _ = solve_minimax((-Q).T)
    else:
        dropper_strat, _ = solve_minimax(-Q)
        checker_strat, _ = solve_minimax(Q.T)

    mean_q_value_for_hal = float(dropper_strat @ Q @ checker_strat)

    principal_line = _principal_line(root)
    root_drop_seconds = root.drop_seconds
    root_check_seconds = root.check_seconds
    if root_strat_count > 0:
        dropper_avg = _validate_distribution(
            root_drop_sum / root_strat_weight,
            len(root.drop_seconds),
            name="improved dropper policy",
        )
        checker_avg = _validate_distribution(
            root_check_sum / root_strat_weight,
            len(root.check_seconds),
            name="improved checker policy",
        )
    else:
        dropper_avg = np.asarray(dropper_strat, dtype=np.float64).copy()
        checker_avg = np.asarray(checker_strat, dtype=np.float64).copy()
    value_for_hal = float(dropper_avg @ Q @ checker_avg)

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
            mean_q_value_for_hal = value_for_hal
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
        improved_dropper_policy=dropper_avg,
        improved_checker_policy=checker_avg,
        root_value_for_hal=value_for_hal,
        mean_q_dropper_policy=np.asarray(dropper_strat, dtype=np.float64),
        mean_q_checker_policy=np.asarray(checker_strat, dtype=np.float64),
        mean_q_value_for_hal=mean_q_value_for_hal,
        root_visits=root.N_node,
        principal_line=principal_line,
        cells_used=int(root.N_cell.sum()),
        root_unique_cells_visited=int(np.count_nonzero(root.N_cell)),
        action_mode=config.action_mode,
        root_drop_seconds=root_drop_seconds,
        root_check_seconds=root_check_seconds,
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
