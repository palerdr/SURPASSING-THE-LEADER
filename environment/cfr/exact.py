"""Rigorous exact-second CFR core.

Single module covering the rigorous foundation:

    - Terminal-only utility (UtilityBreakdown, terminal_value).
    - Exact public state (ExactPublicState).
    - Zero-sum LP minimax (solve_minimax).
    - Engine-aware action enumeration and transition expansion
        (ExactSearchConfig, enumerate_joint_actions, expand_joint_action,
        ExactGameSnapshot).
    - Exact-second matrix-game solver and finite-horizon evaluator
        (exact_immediate_checker_payoff_matrix, evaluate_joint_action,
        solve_exact_finite_horizon).

No reward shaping, no bucketing, no value-net frontier. Frontier states
beyond the configured horizon are reported as unresolved mass.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

from src.Constants import CYLINDER_MAX, FAILED_CHECK_PENALTY, TURN_DURATION_NORMAL
from src.Game import Game, HalfRoundRecord

from environment.legal_actions import legal_max_second


# ── Terminal utility ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class UtilityBreakdown:
    """Terminal-only utility summary from Hal's perspective."""

    value: float
    hal_win_probability: float
    baku_win_probability: float
    unresolved_probability: float


def terminal_value(game: Game, perspective_name: str = "Hal") -> float | None:
    """Return terminal utility, or None when the match is unresolved."""
    if not game.game_over:
        return None
    if game.winner is None:
        return 0.0
    return 1.0 if game.winner.name.lower() == perspective_name.lower() else -1.0


def terminal_breakdown(game: Game, perspective_name: str = "Hal") -> UtilityBreakdown:
    """Return a terminal-only breakdown for a single deterministic state."""
    value = terminal_value(game, perspective_name=perspective_name)
    if value is None:
        return UtilityBreakdown(
            value=0.0,
            hal_win_probability=0.0,
            baku_win_probability=0.0,
            unresolved_probability=1.0,
        )
    if value > 0:
        return UtilityBreakdown(value=value, hal_win_probability=1.0, baku_win_probability=0.0, unresolved_probability=0.0)
    if value < 0:
        return UtilityBreakdown(value=value, hal_win_probability=0.0, baku_win_probability=1.0, unresolved_probability=0.0)
    return UtilityBreakdown(value=value, hal_win_probability=0.0, baku_win_probability=0.0, unresolved_probability=0.0)


# ── Exact public state ────────────────────────────────────────────────────


@dataclass(frozen=True)
class ExactPublicState:
    p1_name: str
    p1_physicality: float
    p1_cylinder: float
    p1_ttd: float
    p1_deaths: int
    p1_alive: bool
    p2_name: str
    p2_physicality: float
    p2_cylinder: float
    p2_ttd: float
    p2_deaths: int
    p2_alive: bool
    referee_cprs: int
    game_clock: float
    current_half: int
    round_num: int
    first_dropper_name: str
    game_over: bool
    winner_name: str | None
    loser_name: str | None


def exact_public_state(game: Game) -> ExactPublicState:
    first_dropper_name = game.first_dropper.name if game.first_dropper is not None else ""
    return ExactPublicState(
        p1_name=game.player1.name,
        p1_physicality=game.player1.physicality,
        p1_cylinder=game.player1.cylinder,
        p1_ttd=game.player1.ttd,
        p1_deaths=game.player1.deaths,
        p1_alive=game.player1.alive,
        p2_name=game.player2.name,
        p2_physicality=game.player2.physicality,
        p2_cylinder=game.player2.cylinder,
        p2_ttd=game.player2.ttd,
        p2_deaths=game.player2.deaths,
        p2_alive=game.player2.alive,
        referee_cprs=game.referee.cprs_performed,
        game_clock=game.game_clock,
        current_half=game.current_half,
        round_num=game.round_num,
        first_dropper_name=first_dropper_name,
        game_over=game.game_over,
        winner_name=game.winner.name if game.winner is not None else None,
        loser_name=game.loser.name if game.loser is not None else None,
    )


# ── Zero-sum LP minimax ───────────────────────────────────────────────────


def solve_minimax(payoff: np.ndarray) -> tuple[np.ndarray, float]:
    """Solve the row player's maximin strategy for a zero-sum payoff matrix."""
    m, n = payoff.shape
    if m == 1:
        return np.array([1.0]), float(payoff[0].min())
    if n == 1:
        best = int(np.argmax(payoff[:, 0]))
        strategy = np.zeros(m)
        strategy[best] = 1.0
        return strategy, float(payoff[best, 0])

    c = np.zeros(m + 1)
    c[m] = -1.0

    a_ub = np.zeros((n, m + 1))
    a_ub[:, :m] = -payoff.T
    a_ub[:, m] = 1.0
    b_ub = np.zeros(n)

    a_eq = np.zeros((1, m + 1))
    a_eq[0, :m] = 1.0
    b_eq = np.array([1.0])

    result = linprog(
        c,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=[(0, None)] * m + [(None, None)],
        method="highs",
    )

    if not result.success or result.x is None:
        uniform = np.ones(m) / m
        return uniform, float(np.min(uniform @ payoff))

    strategy = np.maximum(result.x[:m], 0.0)
    total = strategy.sum()
    if total > 1e-9:
        strategy /= total
    else:
        strategy = np.ones(m) / m
    return strategy, float(result.x[m])


# ── Action enumeration, snapshot/restore, transition expansion ────────────


@dataclass(frozen=True)
class ExactSearchConfig:
    """Knowledge/legality switches for exact action expansion."""

    hal_leap_deduced: bool = False
    hal_memory_impaired: bool = False
    perspective_name: str = "Hal"


@dataclass(frozen=True)
class ExactJointAction:
    drop_time: int
    check_time: int


@dataclass(frozen=True)
class ExactTransition:
    action: ExactJointAction
    probability: float
    state: ExactPublicState
    terminal_value: float | None
    record: HalfRoundRecord


class ExactGameSnapshot:
    """Copy/restore mutable Game state for exact tree traversal."""

    __slots__ = (
        "p1_cylinder", "p1_ttd", "p1_deaths", "p1_alive", "p1_dh_len",
        "p2_cylinder", "p2_ttd", "p2_deaths", "p2_alive", "p2_dh_len",
        "cprs", "clock", "current_half", "round_num", "game_over",
        "winner", "loser", "hist_len",
    )

    def __init__(self, game: Game):
        self.p1_cylinder = game.player1.cylinder
        self.p1_ttd = game.player1.ttd
        self.p1_deaths = game.player1.deaths
        self.p1_alive = game.player1.alive
        self.p1_dh_len = len(game.player1.death_history)
        self.p2_cylinder = game.player2.cylinder
        self.p2_ttd = game.player2.ttd
        self.p2_deaths = game.player2.deaths
        self.p2_alive = game.player2.alive
        self.p2_dh_len = len(game.player2.death_history)
        self.cprs = game.referee.cprs_performed
        self.clock = game.game_clock
        self.current_half = game.current_half
        self.round_num = game.round_num
        self.game_over = game.game_over
        self.winner = game.winner
        self.loser = game.loser
        self.hist_len = len(game.history)

    def restore(self, game: Game) -> None:
        game.player1.cylinder = self.p1_cylinder
        game.player1.ttd = self.p1_ttd
        game.player1.deaths = self.p1_deaths
        game.player1.alive = self.p1_alive
        del game.player1.death_history[self.p1_dh_len:]
        game.player2.cylinder = self.p2_cylinder
        game.player2.ttd = self.p2_ttd
        game.player2.deaths = self.p2_deaths
        game.player2.alive = self.p2_alive
        del game.player2.death_history[self.p2_dh_len:]
        game.referee.cprs_performed = self.cprs
        game.game_clock = self.clock
        game.current_half = self.current_half
        game.round_num = self.round_num
        game.game_over = self.game_over
        game.winner = self.winner
        game.loser = self.loser
        del game.history[self.hist_len:]


def legal_seconds_for_current_role(game: Game, actor_name: str, role: str, config: ExactSearchConfig) -> range:
    turn_duration = game.get_turn_duration()
    max_second = legal_max_second(
        actor_name,
        role,
        turn_duration,
        hal_leap_deduced=config.hal_leap_deduced,
        hal_memory_impaired=config.hal_memory_impaired,
    )
    if role == "checker":
        max_second = min(max_second, max(turn_duration, TURN_DURATION_NORMAL))
    return range(1, max_second + 1)


def enumerate_joint_actions(game: Game, config: ExactSearchConfig | None = None) -> list[ExactJointAction]:
    config = config or ExactSearchConfig()
    dropper, checker = game.get_roles_for_half(game.current_half)
    drop_seconds = legal_seconds_for_current_role(game, dropper.name, "dropper", config)
    check_seconds = legal_seconds_for_current_role(game, checker.name, "checker", config)
    return [ExactJointAction(d, c) for d in drop_seconds for c in check_seconds]


def expand_joint_action(
    game: Game,
    action: ExactJointAction,
    config: ExactSearchConfig | None = None,
) -> tuple[ExactTransition, ...]:
    """Expand a joint action into deterministic/no-death or survival chance branches.

    The input game is restored to its original state before this function returns.
    """
    config = config or ExactSearchConfig()
    snap = ExactGameSnapshot(game)
    probe = game.resolve_half_round(action.drop_time, action.check_time, survived_outcome=None)
    death_occurred = probe.survived is not None
    survival_probability = probe.survival_probability
    snap.restore(game)

    if not death_occurred:
        record = game.resolve_half_round(action.drop_time, action.check_time, survived_outcome=None)
        transition = ExactTransition(
            action=action,
            probability=1.0,
            state=exact_public_state(game),
            terminal_value=terminal_value(game, perspective_name=config.perspective_name),
            record=record,
        )
        snap.restore(game)
        return (transition,)

    assert survival_probability is not None
    branches: list[ExactTransition] = []
    for survived, probability in ((True, survival_probability), (False, 1.0 - survival_probability)):
        if probability <= 0.0:
            continue
        record = game.resolve_half_round(action.drop_time, action.check_time, survived_outcome=survived)
        branches.append(
            ExactTransition(
                action=action,
                probability=probability,
                state=exact_public_state(game),
                terminal_value=terminal_value(game, perspective_name=config.perspective_name),
                record=record,
            )
        )
        snap.restore(game)
    return tuple(branches)


# ── Exact-second matrix-game solver and finite-horizon evaluator ──────────


@dataclass(frozen=True)
class ExactMatrixGame:
    drop_actions: tuple[int, ...]
    check_actions: tuple[int, ...]
    payoff: np.ndarray


@dataclass(frozen=True)
class ExactSolveResult:
    dropper_strategy: np.ndarray
    checker_strategy: np.ndarray
    value_for_hal: float
    breakdown: UtilityBreakdown
    unresolved_probability: float
    half_round_horizon: int
    drop_actions: tuple[int, ...] = ()
    check_actions: tuple[int, ...] = ()
    payoff_for_hal: np.ndarray | None = None


def exact_immediate_checker_payoff_matrix(game: Game, config: ExactSearchConfig | None = None) -> ExactMatrixGame:
    """Build exact checker-perspective immediate payoff matrix for the current half-round."""
    config = config or ExactSearchConfig()
    dropper, checker = game.get_roles_for_half(game.current_half)
    actions = enumerate_joint_actions(game, config)
    drop_actions = tuple(sorted({a.drop_time for a in actions}))
    check_actions = tuple(sorted({a.check_time for a in actions}))
    d_index = {second: i for i, second in enumerate(drop_actions)}
    c_index = {second: i for i, second in enumerate(check_actions)}
    payoff = np.zeros((len(drop_actions), len(check_actions)), dtype=np.float64)

    for action in actions:
        if action.check_time >= action.drop_time:
            st = max(1, action.check_time - action.drop_time)
            payoff[d_index[action.drop_time], c_index[action.check_time]] = (
                -CYLINDER_MAX if checker.cylinder + st >= CYLINDER_MAX else -st
            )
        else:
            payoff[d_index[action.drop_time], c_index[action.check_time]] = -min(
                checker.cylinder + FAILED_CHECK_PENALTY,
                CYLINDER_MAX,
            )
    return ExactMatrixGame(drop_actions=drop_actions, check_actions=check_actions, payoff=payoff)


def _terminal_breakdown(value: float | None) -> UtilityBreakdown:
    if value is None:
        return UtilityBreakdown(0.0, 0.0, 0.0, 1.0)
    if value > 0.0:
        return UtilityBreakdown(value, 1.0, 0.0, 0.0)
    if value < 0.0:
        return UtilityBreakdown(value, 0.0, 1.0, 0.0)
    return UtilityBreakdown(value, 0.0, 0.0, 0.0)


def _weighted_breakdown(parts: list[tuple[float, UtilityBreakdown]]) -> UtilityBreakdown:
    value = sum(weight * part.value for weight, part in parts)
    hal = sum(weight * part.hal_win_probability for weight, part in parts)
    baku = sum(weight * part.baku_win_probability for weight, part in parts)
    unresolved = sum(weight * part.unresolved_probability for weight, part in parts)
    return UtilityBreakdown(value, hal, baku, unresolved)


def evaluate_joint_action(
    game: Game,
    action: ExactJointAction,
    half_round_horizon: int,
    config: ExactSearchConfig | None = None,
) -> UtilityBreakdown:
    config = config or ExactSearchConfig()
    if game.game_over:
        return _terminal_breakdown(terminal_value(game, perspective_name=config.perspective_name))
    if half_round_horizon <= 0:
        return _terminal_breakdown(None)

    snap = ExactGameSnapshot(game)
    probe = game.resolve_half_round(action.drop_time, action.check_time, survived_outcome=None)
    death_occurred = probe.survived is not None
    survival_probability = probe.survival_probability
    snap.restore(game)

    parts: list[tuple[float, UtilityBreakdown]] = []

    if not death_occurred:
        game.resolve_half_round(action.drop_time, action.check_time, survived_outcome=None)
        value = terminal_value(game, perspective_name=config.perspective_name)
        if value is not None or half_round_horizon == 1:
            part = _terminal_breakdown(value)
        else:
            part = solve_exact_finite_horizon(game, half_round_horizon - 1, config).breakdown
        snap.restore(game)
        return part

    assert survival_probability is not None
    for survived, probability in ((True, survival_probability), (False, 1.0 - survival_probability)):
        if probability <= 0.0:
            continue
        game.resolve_half_round(action.drop_time, action.check_time, survived_outcome=survived)
        value = terminal_value(game, perspective_name=config.perspective_name)
        if value is not None or half_round_horizon == 1:
            part = _terminal_breakdown(value)
        else:
            part = solve_exact_finite_horizon(game, half_round_horizon - 1, config).breakdown
        parts.append((probability, part))
        snap.restore(game)

    return _weighted_breakdown(parts)


def solve_exact_finite_horizon(
    game: Game,
    half_round_horizon: int,
    config: ExactSearchConfig | None = None,
) -> ExactSolveResult:
    """Solve exact-second zero-sum matrix games recursively to a finite horizon.

    Horizon cutoff is reported as unresolved probability; no heuristic frontier
    value is used.
    """
    config = config or ExactSearchConfig()
    terminal = terminal_value(game, perspective_name=config.perspective_name)
    if terminal is not None or half_round_horizon <= 0:
        breakdown = _terminal_breakdown(terminal)
        return ExactSolveResult(
            dropper_strategy=np.zeros(0),
            checker_strategy=np.zeros(0),
            value_for_hal=breakdown.value,
            breakdown=breakdown,
            unresolved_probability=breakdown.unresolved_probability,
            half_round_horizon=half_round_horizon,
            payoff_for_hal=None,
        )

    dropper, _checker = game.get_roles_for_half(game.current_half)
    hal_is_dropper = dropper.name.lower() == config.perspective_name.lower()
    actions = enumerate_joint_actions(game, config)
    drop_actions = tuple(sorted({a.drop_time for a in actions}))
    check_actions = tuple(sorted({a.check_time for a in actions}))
    d_index = {second: i for i, second in enumerate(drop_actions)}
    c_index = {second: i for i, second in enumerate(check_actions)}
    hal_payoff = np.zeros((len(drop_actions), len(check_actions)), dtype=np.float64)
    breakdowns: dict[tuple[int, int], UtilityBreakdown] = {}

    for action in actions:
        breakdown = evaluate_joint_action(game, action, half_round_horizon, config)
        i = d_index[action.drop_time]
        j = c_index[action.check_time]
        hal_payoff[i, j] = breakdown.value
        breakdowns[(i, j)] = breakdown

    row_payoff = hal_payoff if hal_is_dropper else -hal_payoff.T
    row_strategy, row_value = solve_minimax(row_payoff)
    if hal_is_dropper:
        dropper_strategy = row_strategy
        checker_strategy, _ = solve_minimax((-hal_payoff).T)
    else:
        checker_strategy = row_strategy
        dropper_strategy, _ = solve_minimax(hal_payoff)

    parts: list[tuple[float, UtilityBreakdown]] = []
    for i, dp in enumerate(dropper_strategy):
        if dp <= 0.0:
            continue
        for j, cp in enumerate(checker_strategy):
            weight = float(dp * cp)
            if weight > 0.0:
                parts.append((weight, breakdowns[(i, j)]))

    breakdown = _weighted_breakdown(parts)
    value = float(row_value if hal_is_dropper else -row_value)
    return ExactSolveResult(
        dropper_strategy=dropper_strategy,
        checker_strategy=checker_strategy,
        value_for_hal=value,
        breakdown=breakdown,
        unresolved_probability=breakdown.unresolved_probability,
        half_round_horizon=half_round_horizon,
        drop_actions=drop_actions,
        check_actions=check_actions,
        payoff_for_hal=hal_payoff,
    )
