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


from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

from stl.engine.game import CYLINDER_MAX, FAILED_CHECK_PENALTY, TURN_DURATION_NORMAL
from stl.engine.game import Game, HalfRoundRecord

from stl.engine.actions import legal_max_second


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


# ── Zero-sum matrix solvers ───────────────────────────────────────────────


@dataclass(frozen=True)
class CFRPlusConfig:
    """Configuration for the bounded local CFR+ matrix solver."""

    iterations: int = 2000
    average_delay: int = 100
    linear_weighting: bool = True


def _regret_plus_strategy(cumulative_regret: np.ndarray) -> np.ndarray:
    positive = np.maximum(cumulative_regret, 0.0)
    total = float(np.sum(positive))
    if total > 1e-12:
        return positive / total
    return np.ones(len(cumulative_regret), dtype=np.float64) / len(cumulative_regret)


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


def solve_cfr_plus(
    payoff: np.ndarray,
    config: CFRPlusConfig | None = None,
) -> tuple[np.ndarray, float]:
    """Approximate the row player's maximin strategy with CFR+.

    This is for bounded local resolves where a fast iterative matrix solve is
    useful. The certified exact tower continues to call ``solve_minimax``.
    """
    matrix = np.asarray(payoff, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"payoff must be a 2D matrix, got shape {matrix.shape}")
    m, n = matrix.shape
    if m == 0 or n == 0:
        raise ValueError("payoff matrix must have at least one row and one column")
    if m == 1 or n == 1:
        return solve_minimax(matrix)

    config = config or CFRPlusConfig()
    iterations = int(config.iterations)
    if iterations <= 0:
        raise ValueError(f"CFR+ iterations must be positive, got {iterations}")
    average_delay = max(0, int(config.average_delay))

    row_regret = np.zeros(m, dtype=np.float64)
    col_regret = np.zeros(n, dtype=np.float64)
    row_strategy_sum = np.zeros(m, dtype=np.float64)

    for t in range(1, iterations + 1):
        row_strategy = _regret_plus_strategy(row_regret)
        col_strategy = _regret_plus_strategy(col_regret)

        row_action_values = matrix @ col_strategy
        row_value = float(row_strategy @ row_action_values)
        row_regret = np.maximum(row_regret + row_action_values - row_value, 0.0)

        # Alternating update: the column player responds to the row player's
        # freshly updated regret-matching strategy.
        row_strategy = _regret_plus_strategy(row_regret)
        col_action_values = -(row_strategy @ matrix)
        col_value = float(col_strategy @ col_action_values)
        col_regret = np.maximum(col_regret + col_action_values - col_value, 0.0)

        if t > average_delay:
            weight = float(t) if config.linear_weighting else 1.0
            row_strategy_sum += weight * row_strategy

    total = float(row_strategy_sum.sum())
    if total > 1e-12:
        strategy = row_strategy_sum / total
    else:
        strategy = _regret_plus_strategy(row_regret)

    value = float(np.min(strategy @ matrix))
    return strategy, value


def solve_cfr_plus_rust(
    payoff: np.ndarray,
    config: CFRPlusConfig | None = None,
) -> tuple[np.ndarray, float]:
    """Approximate the row player's maximin strategy with the Rust CFR+ kernel.

    This is an opt-in compatibility wrapper. It preserves the Python solver's
    input checks and single-row/column exact fallback, then delegates only the
    dense iterative matrix loop to ``stl_solver_rs``.
    """
    matrix = np.asarray(payoff, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"payoff must be a 2D matrix, got shape {matrix.shape}")
    m, n = matrix.shape
    if m == 0 or n == 0:
        raise ValueError("payoff matrix must have at least one row and one column")
    if m == 1 or n == 1:
        return solve_minimax(matrix)

    config = config or CFRPlusConfig()
    iterations = int(config.iterations)
    if iterations <= 0:
        raise ValueError(f"CFR+ iterations must be positive, got {iterations}")
    average_delay = max(0, int(config.average_delay))

    try:
        import stl_solver_rs
    except ImportError as exc:
        raise ImportError(
            "stl_solver_rs is not installed; run "
            "`uv run --project ..\\.. maturin develop --release` from "
            "`crates/stl_solver` to enable the Rust CFR+ kernel"
        ) from exc

    strategy, value = stl_solver_rs.solve_cfr_plus_rs(
        np.ascontiguousarray(matrix, dtype=np.float64),
        iterations=iterations,
        average_delay=average_delay,
        linear_weighting=bool(config.linear_weighting),
    )
    return np.asarray(strategy, dtype=np.float64), float(value)


# ── Action enumeration, snapshot/restore, transition expansion ────────────


@dataclass(frozen=True)
class ExactSearchConfig:
    """Perspective configuration for exact action expansion.

    The historical hal_leap_deduced / hal_memory_impaired flags are gone —
    Hal can never play check=61 in this codebase, period. The strategic
    constraint is hard-coded in legal_actions.py, not configured here.
    """

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
    del config  # ExactSearchConfig no longer carries legality switches.
    turn_duration = game.get_turn_duration()
    max_second = legal_max_second(actor_name, role, turn_duration)
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


# ── Memoization for solve_exact_finite_horizon ────────────────────────────
#
# Per-process cache keyed on (ExactPublicState, horizon, perspective_name).
# Each multiprocessing worker has its own module state, so the cache is local
# to one process — that's the right scope: the recursion within a single
# corpus state often revisits the same (state, horizon) via distinct paths
# (e.g. (drop=10,check=15) and (drop=20,check=25) can both land on the same
# post-action state). Cross-state cache hits across workers would require
# shared memory and the IPC overhead would eat the speedup.
#
# The cache is invalidated trivially: each worker process runs to completion
# on its slice and exits, freeing the cache. Tests can call
# ``clear_solve_cache()`` between runs to assert determinism.
#
# BOUNDED LRU: the cache is capped at ``_SOLVE_CACHE_MAXSIZE`` entries with
# least-recently-used eviction. An unbounded cache exhausts RAM during
# wide-grid corpus generation: each worker independently caches the deep
# horizon-3 substate space (shared across starting cylinders), and N workers
# hold N duplicate copies. Eviction is correctness-safe — the cache is pure
# memoization, so an evicted (state, horizon) is simply recomputed identically
# on next visit. The cap trades a little recompute for bounded memory.
# Override via STL_SOLVE_CACHE_MAXSIZE for tuning (default 30000 ≈ ~1 GiB/worker).

import os
from collections import OrderedDict

_SOLVE_CACHE_MAXSIZE: int = int(os.environ.get("STL_SOLVE_CACHE_MAXSIZE", "30000"))
_SOLVE_CACHE: "OrderedDict[tuple, ExactSolveResult]" = OrderedDict()


def clear_solve_cache() -> None:
    """Clear the per-process memoization cache. Tests use this to isolate runs."""
    _SOLVE_CACHE.clear()


def solve_cache_size() -> int:
    """Diagnostic: number of cached (state, horizon, perspective) entries."""
    return len(_SOLVE_CACHE)


def solve_cache_maxsize() -> int:
    """Diagnostic: the LRU eviction ceiling for the cache."""
    return _SOLVE_CACHE_MAXSIZE


def solve_exact_finite_horizon(
    game: Game,
    half_round_horizon: int,
    config: ExactSearchConfig | None = None,
) -> ExactSolveResult:
    """Solve exact-second zero-sum matrix games recursively to a finite horizon.

    Horizon cutoff is reported as unresolved probability; no heuristic frontier
    value is used.

    Memoized: results are cached on ``(exact_public_state(game), horizon,
    perspective_name)``. The cache is per-process and grows only for the
    duration of one worker's slice of corpus generation. On a near-leap-window
    state at horizon=3, intra-recursion revisits saturate the cache quickly
    and shrink wall-clock by orders of magnitude.
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

    cache_key = (exact_public_state(game), half_round_horizon, config.perspective_name)
    cached = _SOLVE_CACHE.get(cache_key)
    if cached is not None:
        _SOLVE_CACHE.move_to_end(cache_key)  # mark most-recently-used
        return cached

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

    # Place the maximizing player on the LP rows. ``hal_payoff`` is Hal-perspective
    # with rows=drop, cols=check. The dropper maximizes Hal value; the checker, as
    # the opponent, maximizes -Hal. When Hal is the checker we solve Hal over
    # ``hal_payoff.T`` (Hal still maximizing) and give Baku the dropper LP over
    # ``-hal_payoff``. Feeding the un-negated matrix to the off-side role (the prior
    # bug) returned the wrong player's best response AND a wrong value_for_hal
    # whenever Hal was the checker.
    if hal_is_dropper:
        dropper_strategy, value_for_hal = solve_minimax(hal_payoff)
        checker_strategy, _ = solve_minimax((-hal_payoff).T)
    else:
        checker_strategy, value_for_hal = solve_minimax(hal_payoff.T)
        dropper_strategy, _ = solve_minimax(-hal_payoff)

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
    result = ExactSolveResult(
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
    _SOLVE_CACHE[cache_key] = result
    _SOLVE_CACHE.move_to_end(cache_key)
    if len(_SOLVE_CACHE) > _SOLVE_CACHE_MAXSIZE:
        _SOLVE_CACHE.popitem(last=False)  # evict least-recently-used
    return result


# ============================================================================
# Half-round matrix helpers
# ============================================================================

"""Single half-round regret-matching CFR baseline.

This module is the original (pre-Phase-1) regret-matching solver over the
exact 1..60 second matrix. It is *not bucketed* — the action space is
exact — but the canonical rigorous solver in this namespace is
``exact_solver.solve_exact_finite_horizon`` (LP minimax + chance-branch
recursion). Keep this file as a comparison baseline for matrix-equivalence
tests and as the source of pure helpers like ``survival_probability``;
new rigorous code should not call ``solve_half_round`` from here.
"""


import numpy as np

from stl.engine.game import (
    FAILED_CHECK_PENALTY,
    CYLINDER_MAX,
    BASE_CURVE_K,
    CARDIAC_DECAY,
    REFEREE_DECAY,
    REFEREE_FLOOR,
)


def regret_match(cumulative_regret: np.ndarray) -> np.ndarray:
    """Convert cumulative regret vector into a strategy (probability distribution)."""
    positive = np.maximum(0, cumulative_regret)
    total = np.sum(positive)
    if total > 0:
        return positive / total
    return np.ones(len(cumulative_regret)) / len(cumulative_regret)


def survival_probability(
    death_duration: float,
    player_ttd: float,
    cprs_performed: int,
    physicality: float,
) -> float:
    """Compute survival probability using the engine's exact formula."""
    if death_duration >= CYLINDER_MAX:
        return 0.0
    base = max(0.0, 1.0 - (death_duration / CYLINDER_MAX) ** BASE_CURVE_K)
    cardiac = CARDIAC_DECAY ** (player_ttd / 60.0)
    referee = max(REFEREE_FLOOR, REFEREE_DECAY ** cprs_performed)
    return base * cardiac * referee * physicality


def compute_payoff_matrix(
    checker_cylinder: float,
    turn_duration: int = 60,
) -> np.ndarray:
    """Build the Checker's immediate payoff matrix (no continuation values)."""
    n = turn_duration
    payoff = np.zeros((n, n), dtype=np.float64)

    for d in range(n):
        drop_time = d + 1
        for c in range(n):
            check_time = c + 1

            if check_time >= drop_time:
                st = max(1, check_time - drop_time)
                if checker_cylinder + st >= CYLINDER_MAX:
                    payoff[d][c] = -CYLINDER_MAX
                else:
                    payoff[d][c] = -st
            else:
                injection = min(checker_cylinder + FAILED_CHECK_PENALTY, CYLINDER_MAX)
                payoff[d][c] = -injection

    return payoff


def build_augmented_payoff_matrix(
    st_to_cont_val: dict[int, float],
    fail_cont_val: float,
    fail_surv_prob: float,
    turn_duration: int = 60,
    lose_value: float = -1.0,
) -> np.ndarray:
    """Build augmented payoff matrix from precomputed continuation values.

    Args:
        st_to_cont_val: Maps ST (1..59) → checker continuation value for that outcome.
            For overflow STs, the value already includes survival probability weighting.
        fail_cont_val: Checker continuation value for failed check (survived).
        fail_surv_prob: Survival probability for failed check.
    """
    n = turn_duration
    payoff = np.zeros((n, n), dtype=np.float64)

    fail_payoff = fail_surv_prob * fail_cont_val + (1 - fail_surv_prob) * lose_value

    for d in range(n):
        drop_time = d + 1
        for c in range(n):
            check_time = c + 1

            if check_time >= drop_time:
                st = max(1, check_time - drop_time)
                payoff[d][c] = st_to_cont_val.get(st, 0.0)
            else:
                payoff[d][c] = fail_payoff

    return payoff


def solve_half_round(
    checker_cylinder: float,
    turn_duration: int = 60,
    iterations: int = 10_000,
    payoff_matrix: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Solve a single half-round for Nash equilibrium strategies."""
    n = turn_duration
    if payoff_matrix is not None:
        payoff = payoff_matrix
    else:
        payoff = compute_payoff_matrix(checker_cylinder, turn_duration)

    dropper_regret = np.zeros(n)
    checker_regret = np.zeros(n)
    dropper_strategy_sum = np.zeros(n)
    checker_strategy_sum = np.zeros(n)

    for _ in range(iterations):
        dropper_strat = regret_match(dropper_regret)
        checker_strat = regret_match(checker_regret)

        dropper_strategy_sum += dropper_strat
        checker_strategy_sum += checker_strat

        dropper_action_values = -payoff @ checker_strat
        checker_action_values = payoff.T @ dropper_strat

        dropper_EV = dropper_strat @ dropper_action_values
        checker_EV = checker_strat @ checker_action_values

        dropper_regret += dropper_action_values - dropper_EV
        checker_regret += checker_action_values - checker_EV

    avg_dropper = dropper_strategy_sum / dropper_strategy_sum.sum()
    avg_checker = checker_strategy_sum / checker_strategy_sum.sum()
    game_value = avg_dropper @ payoff @ avg_checker

    return (avg_dropper, avg_checker, game_value)


# ============================================================================
# Timing and LSR predicates
# ============================================================================

"""Engine-derived timing and route arithmetic for rigorous CFR.

Pure functions over the engine state: clock geometry, LSR variation,
projected gaps, and survival-budget math. No reward shaping, no stage
labels, no curriculum milestones, no "good for Hal" categories. Anything
that asserts progress without proving it through terminal utility lives
in ``environment/route_math.py`` or further out.
"""


from dataclasses import dataclass

from stl.engine.game import (
    CYLINDER_MAX,
    FAILED_CHECK_PENALTY,
    LS_WINDOW_END,
    LS_WINDOW_START,
    TURN_DURATION_NORMAL,
)


__all__ = [
    "PlayerBudget",
    "current_checker_fail_would_activate_lsr",
    "current_dropper_checker",
    "current_lsr_variation",
    "get_named_players",
    "is_active_lsr",
    "is_leap_window",
    "lsr_variation_from_clock",
    "player_budget",
    "player_named",
    "projected_failed_check_death_duration",
    "projected_post_fail_ttd",
    "projected_round_gap_for_death_duration",
    "projected_round_gap_for_two_deaths",
    "projected_variation_after_current_checker_fail",
    "projected_variation_after_gap",
    "role_for_player",
    "rounds_until_leap_window",
    "safe_strategy_budget",
    "strict_next_minute",
]


def strict_next_minute(seconds: float) -> int:
    whole_seconds = int(seconds)
    return ((whole_seconds // 60) + 1) * 60


def lsr_variation_from_clock(game_clock: float) -> int:
    minutes_since_start = int(game_clock) // 60
    return (minutes_since_start % 4) + 1


def current_lsr_variation(game) -> int:
    return lsr_variation_from_clock(game.game_clock)


def is_active_lsr(game) -> bool:
    return current_lsr_variation(game) == 2


def is_leap_window(game_clock: float) -> bool:
    # Inclusive of LS_WINDOW_END to match the authoritative engine rule
    # (Game.get_turn_duration treats game_clock == LS_WINDOW_END as a leap turn).
    return LS_WINDOW_START <= game_clock <= LS_WINDOW_END


def rounds_until_leap_window(game) -> int:
    remaining = max(0.0, LS_WINDOW_START - game.game_clock)
    return int(remaining // (TURN_DURATION_NORMAL * 4))


def get_named_players(game):
    hal = game.player1 if game.player1.name.lower() == "hal" else game.player2
    baku = game.player1 if game.player1.name.lower() == "baku" else game.player2
    return hal, baku


def player_named(game, name: str):
    lowered = name.lower()
    hal, baku = get_named_players(game)
    if lowered == "hal":
        return hal
    if lowered == "baku":
        return baku
    raise ValueError(f"Unknown player name: {name}")


def current_dropper_checker(game):
    return game.get_roles_for_half(game.current_half)


def role_for_player(game, player) -> str:
    dropper, checker = current_dropper_checker(game)
    if player is dropper:
        return "dropper"
    if player is checker:
        return "checker"
    raise ValueError("player is not part of this game")


def projected_failed_check_death_duration(player) -> float:
    return player.cylinder + FAILED_CHECK_PENALTY


def projected_post_fail_ttd(player) -> float:
    return player.ttd + projected_failed_check_death_duration(player)


def safe_strategy_budget(player) -> int:
    return max(0, int((CYLINDER_MAX - 1 - player.cylinder) // TURN_DURATION_NORMAL))


def projected_round_gap_for_death_duration(death_duration: float) -> int:
    return strict_next_minute(death_duration + 300.0)


def projected_round_gap_for_two_deaths(first_death_duration: float, second_death_duration: float) -> int:
    return strict_next_minute(first_death_duration + second_death_duration + 420.0)


def projected_variation_after_gap(round_start_clock: float, gap_seconds: int) -> int:
    return lsr_variation_from_clock(round_start_clock + gap_seconds)


def projected_variation_after_current_checker_fail(game) -> int:
    round_start_clock = game.game_clock if game.current_half == 1 else game.game_clock - 120.0
    checker = current_dropper_checker(game)[1]
    gap_seconds = projected_round_gap_for_death_duration(projected_failed_check_death_duration(checker))
    return projected_variation_after_gap(round_start_clock, gap_seconds)


def current_checker_fail_would_activate_lsr(game) -> bool:
    return projected_variation_after_current_checker_fail(game) == 2


@dataclass(frozen=True)
class PlayerBudget:
    cylinder: float
    ttd: float
    deaths: int
    safe_budget: int
    fail_death_duration: float
    fail_post_ttd: float


def player_budget(player) -> PlayerBudget:
    fail_death_duration = projected_failed_check_death_duration(player)
    return PlayerBudget(
        cylinder=player.cylinder,
        ttd=player.ttd,
        deaths=player.deaths,
        safe_budget=safe_strategy_budget(player),
        fail_death_duration=fail_death_duration,
        fail_post_ttd=player.ttd + fail_death_duration,
    )


# ============================================================================
# Exact strategy diagnostics
# ============================================================================

"""Exploitability and best-response diagnostics for exact CFR results."""


from dataclasses import dataclass

import numpy as np

from stl.engine.game import Game


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
