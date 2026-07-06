"""Value-net training-target generation (Phase 2).

Sweeps a corpus of game states and labels each with an exact
Hal-perspective equilibrium value. Sources, in priority order:

    1. Terminal positions: pinned by ``terminal_value`` (+1/-1/0).
    2. Tablebase scenarios: pinned by the registry's ``expected_value``.
    3. LSR-significant non-terminal positions: solved at horizon=3.
       Significance gate: ``rounds_until_leap_window(game) <= 2`` or
       ``current_checker_fail_would_activate_lsr(game)``.
    4. Other LSR-pressure positions: solved at horizon=2.
       Pressure gate: ``is_active_lsr(game)`` or any player cylinder
       >= 240 (near-overflow).
    5. Otherwise: state is *excluded* from the gen-0 corpus.
       Phase-3 MCTS bootstrap will cover those.

No horizon=1 LP labels — they are LSR-blind by construction (the
recursion bottoms out as unresolved on every non-terminal cell). No
reward shaping, no rollouts. Every emitted label is either a true
terminal value, a pinned tablebase value, or a finite-horizon exact
LP minimax value at horizon >= 2. Output is a list of ValueTarget
records or an .npz file with arrays X (N, FEATURE_DIM), y (N,),
sources (N,), horizons (N,).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from stl.solver.exact import (
    ExactPublicState,
    ExactSearchConfig,
    exact_public_state,
    legal_seconds_for_current_role,
    solve_exact_finite_horizon,
    terminal_value,
)
from stl.solver.tablebase import REGISTRY
from stl.solver.exact import (
    current_checker_fail_would_activate_lsr,
    is_active_lsr,
    rounds_until_leap_window,
)
from stl.learning.model import FEATURE_DIM, extract_features
from stl.engine.game import PHYSICALITY_BAKU, PHYSICALITY_HAL
from stl.engine.game import Game
from stl.engine.game import Player
from stl.engine.game import Referee


# ── Source labels ─────────────────────────────────────────────────────────

SOURCE_TERMINAL = "terminal"
SOURCE_TABLEBASE = "tablebase"
# Phase F-2: interior-valued pins are scored as a SEPARATE held-out class so the
# gate can't average-mask a badly-calibrated interior behind the 19 easy ±1
# boundary pins. They still TRAIN under SOURCE_TABLEBASE (they are tablebase
# pins) — the split is held-out-only, so the Phase-G training corpus is byte-for-
# byte unchanged. See ``_generate_tablebase_targets(split_interior=...)``.
SOURCE_TABLEBASE_INTERIOR = "tablebase_interior"
SOURCE_EXACT_HORIZON_2 = "exact_horizon_2"
SOURCE_EXACT_HORIZON_3 = "exact_horizon_3"
SOURCE_MCTS_BOOTSTRAP = "mcts_bootstrap"
# Phase H reanalysis: states the horizon-2/3 LP left unresolved (the rejected
# pool) are re-solved at a deeper horizon. Those that now resolve exactly are
# SOURCE_EXACT_HORIZON_4; those still unresolved even at the deeper exact horizon
# fall back to a high-iter MCTS estimate (SOURCE_REANALYSIS_MCTS). Both are
# ADDITIVE — Phase H never overwrites the Phase-G corpus.
SOURCE_EXACT_HORIZON_4 = "exact_horizon_4"
SOURCE_REANALYSIS_MCTS = "reanalysis_mcts"

VALID_SOURCES = (
    SOURCE_TERMINAL,
    SOURCE_TABLEBASE,
    SOURCE_TABLEBASE_INTERIOR,
    SOURCE_EXACT_HORIZON_2,
    SOURCE_EXACT_HORIZON_3,
    SOURCE_EXACT_HORIZON_4,
    SOURCE_REANALYSIS_MCTS,
    SOURCE_MCTS_BOOTSTRAP,
)

# Pins carrying this tag are the Phase F-2 interior-valued anchors.
INTERIOR_PIN_TAG = "interior_value"


# ── Gating thresholds ─────────────────────────────────────────────────────

NEAR_LEAP_ROUNDS_THRESHOLD = 2  # rounds_until_leap_window <= this => horizon=3
NEAR_OVERFLOW_CYLINDER_THRESHOLD = 240.0  # any cylinder >= this => horizon=2
REJECTED_UNRESOLVED_THRESHOLD = 0.5


# ── Target record ─────────────────────────────────────────────────────────


def _zero_policy() -> np.ndarray:
    return np.zeros(61, dtype=np.float32)


@dataclass(frozen=True)
class ValueTarget:
    features: np.ndarray
    value: float
    source: str
    horizon: int
    dropper_dist: np.ndarray = field(default_factory=_zero_policy)
    checker_dist: np.ndarray = field(default_factory=_zero_policy)
    dropper_legal_mask: np.ndarray = field(default_factory=_zero_policy)
    checker_legal_mask: np.ndarray = field(default_factory=_zero_policy)
    unresolved_probability: float = 0.0


@dataclass(frozen=True)
class LabelResult:
    value: float
    source: str
    horizon: int
    unresolved_probability: float
    dropper_dist: np.ndarray
    checker_dist: np.ndarray
    dropper_legal_mask: np.ndarray
    checker_legal_mask: np.ndarray

    def __iter__(self):
        # Backward-compatible unpacking: value, source, horizon = label_state(...)
        yield self.value
        yield self.source
        yield self.horizon


# ── Helpers ───────────────────────────────────────────────────────────────


def _build_pinned_table() -> dict[ExactPublicState, float]:
    """Materialize the tablebase registry's pinned-value lookup."""
    table: dict[ExactPublicState, float] = {}
    for factory in REGISTRY.values():
        scenario = factory()
        if scenario.expected_value is not None and not scenario.holdout:
            table[exact_public_state(scenario.game)] = scenario.expected_value
    return table


def _build_game(
    *,
    baku_cylinder: float = 0.0,
    hal_cylinder: float = 0.0,
    clock: float = 720.0,
    current_half: int = 1,
    baku_deaths: int = 0,
    hal_deaths: int = 0,
    referee_cprs: int = 0,
) -> Game:
    hal = Player(name="Hal", physicality=PHYSICALITY_HAL)
    baku = Player(name="Baku", physicality=PHYSICALITY_BAKU)
    hal.cylinder = hal_cylinder
    baku.cylinder = baku_cylinder
    hal.deaths = hal_deaths
    baku.deaths = baku_deaths
    referee = Referee()
    referee.cprs_performed = referee_cprs
    game = Game(player1=hal, player2=baku, referee=referee, first_dropper=hal)
    game.seed(0)
    game.game_clock = clock
    game.current_half = current_half
    return game


def _build_terminal_game(
    *,
    winner_name: str,
    baku_cylinder: float,
    hal_cylinder: float,
    clock: float,
    current_half: int,
) -> Game:
    """Construct a game already in the terminal post-permadeath state.

    Held-out terminals are produced directly rather than through play_half_round
    so we control which side won and what features the surrounding state holds.
    The game_over / winner / loser / alive flags are set to match how
    Game._resolve_death_sequence ends a real game.
    """
    game = _build_game(
        baku_cylinder=baku_cylinder,
        hal_cylinder=hal_cylinder,
        clock=clock,
        current_half=current_half,
    )
    game.game_over = True
    if winner_name.lower() == "hal":
        game.winner = game.player1
        game.loser = game.player2
        game.player2.alive = False
    else:
        game.winner = game.player2
        game.loser = game.player1
        game.player1.alive = False
    return game


def _empty_policy_pair() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    zero = np.zeros(61, dtype=np.float32)
    return zero.copy(), zero.copy(), zero.copy(), zero.copy()


def _legal_policy_vectors(
    game: Game,
    config: ExactSearchConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if game.game_over:
        return _empty_policy_pair()

    dropper, checker = game.get_roles_for_half(game.current_half)
    drop_seconds = tuple(legal_seconds_for_current_role(game, dropper.name, "dropper", config))
    check_seconds = tuple(legal_seconds_for_current_role(game, checker.name, "checker", config))
    dropper_dist = np.zeros(61, dtype=np.float32)
    checker_dist = np.zeros(61, dtype=np.float32)
    dropper_mask = np.zeros(61, dtype=np.float32)
    checker_mask = np.zeros(61, dtype=np.float32)
    for second in drop_seconds:
        dropper_mask[second - 1] = 1.0
        dropper_dist[second - 1] = 1.0 / max(1, len(drop_seconds))
    for second in check_seconds:
        checker_mask[second - 1] = 1.0
        checker_dist[second - 1] = 1.0 / max(1, len(check_seconds))
    return dropper_dist, checker_dist, dropper_mask, checker_mask


def _strategy_vectors(
    *,
    drop_seconds: tuple[int, ...],
    check_seconds: tuple[int, ...],
    dropper_strategy: np.ndarray,
    checker_strategy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    dropper_dist = np.zeros(61, dtype=np.float32)
    checker_dist = np.zeros(61, dtype=np.float32)
    for second, probability in zip(drop_seconds, dropper_strategy):
        dropper_dist[second - 1] = float(probability)
    for second, probability in zip(check_seconds, checker_strategy):
        checker_dist[second - 1] = float(probability)
    return dropper_dist, checker_dist


def _target_from_label(game: Game, label: LabelResult) -> ValueTarget:
    return ValueTarget(
        features=extract_features(game),
        value=label.value,
        source=label.source,
        horizon=label.horizon,
        dropper_dist=label.dropper_dist.astype(np.float32),
        checker_dist=label.checker_dist.astype(np.float32),
        dropper_legal_mask=label.dropper_legal_mask.astype(np.float32),
        checker_legal_mask=label.checker_legal_mask.astype(np.float32),
        unresolved_probability=float(label.unresolved_probability),
    )


def _save_rejected_pool(states: list[ExactPublicState], path: str | Path) -> None:
    snapshots = np.array(
        [json.dumps(asdict(state), sort_keys=True) for state in states],
        dtype=np.str_,
    )
    np.savez(path, state_snapshots=snapshots)


def load_rejected_pool(path: str | Path) -> list[dict]:
    data = np.load(path, allow_pickle=False)
    return [json.loads(str(item)) for item in data["state_snapshots"]]


def game_from_public_state(state: ExactPublicState | dict) -> Game:
    """Reconstruct a Game from an ExactPublicState (or its JSON-dict form).

    Inverse of ``exact_public_state`` / ``_save_rejected_pool``: rebuilds the
    mutable Game so a rejected (high-``unresolved_probability``) pool entry can be
    re-solved at a deeper horizon for Phase H reanalysis. Round-trips exactly —
    ``exact_public_state(game_from_public_state(s)) == s`` — for every field the
    public state carries (cylinder, ttd, deaths, alive, physicality, referee
    CPRs, clock, half, round, first_dropper, and the terminal flags), so the
    deeper solve sees the identical position the shallow solve rejected.
    """
    s = state if isinstance(state, dict) else asdict(state)

    p1 = Player(name=s["p1_name"], physicality=s["p1_physicality"])
    p1.cylinder = s["p1_cylinder"]
    p1.ttd = s["p1_ttd"]
    p1.deaths = s["p1_deaths"]
    p1.alive = s["p1_alive"]

    p2 = Player(name=s["p2_name"], physicality=s["p2_physicality"])
    p2.cylinder = s["p2_cylinder"]
    p2.ttd = s["p2_ttd"]
    p2.deaths = s["p2_deaths"]
    p2.alive = s["p2_alive"]

    referee = Referee()
    referee.cprs_performed = s["referee_cprs"]

    by_name = {p1.name: p1, p2.name: p2}
    first_dropper = by_name.get(s["first_dropper_name"]) if s["first_dropper_name"] else None

    game = Game(player1=p1, player2=p2, referee=referee, first_dropper=first_dropper)
    game.seed(0)
    game.game_clock = s["game_clock"]
    game.current_half = s["current_half"]
    game.round_num = s["round_num"]
    game.game_over = s["game_over"]
    game.winner = by_name.get(s["winner_name"]) if s["winner_name"] else None
    game.loser = by_name.get(s["loser_name"]) if s["loser_name"] else None
    return game


def _is_lsr_significant(game: Game) -> bool:
    """Horizon=3 gate: state's outcome is LSR-routing observable inside 3 half-rounds."""
    if rounds_until_leap_window(game) <= NEAR_LEAP_ROUNDS_THRESHOLD:
        return True
    if current_checker_fail_would_activate_lsr(game):
        return True
    return False


def _is_lsr_pressure(game: Game) -> bool:
    """Horizon=2 gate: state is in active LSR variation or near-overflow on any side."""
    if is_active_lsr(game):
        return True
    hal_cyl = float(game.player1.cylinder if game.player1.name.lower() == "hal" else game.player2.cylinder)
    baku_cyl = float(game.player1.cylinder if game.player1.name.lower() == "baku" else game.player2.cylinder)
    if hal_cyl >= NEAR_OVERFLOW_CYLINDER_THRESHOLD:
        return True
    if baku_cyl >= NEAR_OVERFLOW_CYLINDER_THRESHOLD:
        return True
    return False


# ── Label hierarchy ───────────────────────────────────────────────────────


def label_state(
    game: Game,
    config: ExactSearchConfig | None = None,
    pinned_table: dict[ExactPublicState, float] | None = None,
) -> LabelResult | None:
    """Return (Hal-perspective value, source, horizon) or None if excluded.

    Hierarchy:
      1. terminal_value(game) is not None        -> ("terminal", horizon=0)
      2. exact_public_state(game) in pinned_table -> ("tablebase", horizon=0)
      3. LSR-significant (rounds_until_leap_window <= 2
         OR current_checker_fail_would_activate_lsr)
                                                  -> ("exact_horizon_3", horizon=3)
      4. LSR-pressure (is_active_lsr OR any cylinder >= 240)
                                                  -> ("exact_horizon_2", horizon=2)
      5. Otherwise                                -> None (state excluded).
    """
    config = config or ExactSearchConfig()
    pinned_table = pinned_table if pinned_table is not None else _build_pinned_table()

    tval = terminal_value(game, perspective_name=config.perspective_name)
    if tval is not None:
        drop_dist, check_dist, drop_mask, check_mask = _legal_policy_vectors(game, config)
        return LabelResult(
            value=tval,
            source=SOURCE_TERMINAL,
            horizon=0,
            unresolved_probability=0.0,
            dropper_dist=drop_dist,
            checker_dist=check_dist,
            dropper_legal_mask=drop_mask,
            checker_legal_mask=check_mask,
        )

    state = exact_public_state(game)
    if state in pinned_table:
        drop_dist, check_dist, drop_mask, check_mask = _legal_policy_vectors(game, config)
        return LabelResult(
            value=pinned_table[state],
            source=SOURCE_TABLEBASE,
            horizon=0,
            unresolved_probability=0.0,
            dropper_dist=drop_dist,
            checker_dist=check_dist,
            dropper_legal_mask=drop_mask,
            checker_legal_mask=check_mask,
        )

    if _is_lsr_significant(game):
        result = solve_exact_finite_horizon(game, 3, config)
        drop_dist, check_dist = _strategy_vectors(
            drop_seconds=result.drop_actions,
            check_seconds=result.check_actions,
            dropper_strategy=result.dropper_strategy,
            checker_strategy=result.checker_strategy,
        )
        _, _, drop_mask, check_mask = _legal_policy_vectors(game, config)
        return LabelResult(
            value=result.value_for_hal,
            source=SOURCE_EXACT_HORIZON_3,
            horizon=3,
            unresolved_probability=result.unresolved_probability,
            dropper_dist=drop_dist,
            checker_dist=check_dist,
            dropper_legal_mask=drop_mask,
            checker_legal_mask=check_mask,
        )

    if _is_lsr_pressure(game):
        result = solve_exact_finite_horizon(game, 2, config)
        drop_dist, check_dist = _strategy_vectors(
            drop_seconds=result.drop_actions,
            check_seconds=result.check_actions,
            dropper_strategy=result.dropper_strategy,
            checker_strategy=result.checker_strategy,
        )
        _, _, drop_mask, check_mask = _legal_policy_vectors(game, config)
        return LabelResult(
            value=result.value_for_hal,
            source=SOURCE_EXACT_HORIZON_2,
            horizon=2,
            unresolved_probability=result.unresolved_probability,
            dropper_dist=drop_dist,
            checker_dist=check_dist,
            dropper_legal_mask=drop_mask,
            checker_legal_mask=check_mask,
        )

    return None


# ── Corpus generation ─────────────────────────────────────────────────────


# Worker-process state for multiprocessing.Pool. Set once per worker by
# ``_worker_init``; avoids re-pickling the pinned table on every call.
_WORKER_CONFIG: ExactSearchConfig | None = None
_WORKER_PINNED: dict | None = None
_WORKER_THRESHOLD: float = REJECTED_UNRESOLVED_THRESHOLD


def _worker_init(
    config: ExactSearchConfig | None,
    pinned_table: dict,
    threshold: float,
) -> None:
    global _WORKER_CONFIG, _WORKER_PINNED, _WORKER_THRESHOLD
    _WORKER_CONFIG = config
    _WORKER_PINNED = pinned_table
    _WORKER_THRESHOLD = threshold


def _death_grid_pairs(
    *,
    deaths_grid: tuple[int, ...] | None,
    baku_deaths_grid: tuple[int, ...] | None,
    hal_deaths_grid: tuple[int, ...] | None,
) -> tuple[tuple[int, int], ...]:
    """Return explicit ``(baku_deaths, hal_deaths)`` grid pairs.

    ``deaths_grid`` is retained as a legacy shorthand for symmetric states.
    New corpus generation should prefer the explicit axes so asymmetric
    death-pressure states are not silently omitted.
    """
    if deaths_grid is not None:
        if baku_deaths_grid is not None or hal_deaths_grid is not None:
            raise ValueError(
                "Use either deaths_grid or baku_deaths_grid/hal_deaths_grid, not both"
            )
        return tuple((deaths, deaths) for deaths in deaths_grid)

    baku_axis = tuple((0, 1) if baku_deaths_grid is None else baku_deaths_grid)
    hal_axis = tuple((0, 1) if hal_deaths_grid is None else hal_deaths_grid)
    return tuple((baku_deaths, hal_deaths) for baku_deaths in baku_axis for hal_deaths in hal_axis)


def _label_one_state(
    state_tuple: tuple[float, float, float, int, int, int, int],
) -> tuple[ValueTarget | None, ExactPublicState | None]:
    """Per-state worker. Returns ``(target, rejected_state)`` with at most one
    non-None, or ``(None, None)`` for skipped states. Module-level so it
    pickles cleanly for ``multiprocessing.Pool``."""
    baku_cyl, hal_cyl, clock, half, baku_deaths, hal_deaths, cprs = state_tuple
    game = _build_game(
        baku_cylinder=baku_cyl,
        hal_cylinder=hal_cyl,
        clock=clock,
        current_half=half,
        baku_deaths=baku_deaths,
        hal_deaths=hal_deaths,
        referee_cprs=cprs,
    )
    label = label_state(game, _WORKER_CONFIG, _WORKER_PINNED)
    if label is None:
        return None, None
    if label.unresolved_probability > _WORKER_THRESHOLD:
        return None, exact_public_state(game)
    return _target_from_label(game, label), None


def generate_targets(
    *,
    baku_cylinder_grid: tuple[float, ...] = (0.0, 60.0, 120.0, 180.0, 240.0, 290.0, 299.0),
    hal_cylinder_grid: tuple[float, ...] = (0.0, 120.0, 240.0),
    clock_grid: tuple[float, ...] = (720.0, 2000.0, 3540.0),
    half_grid: tuple[int, ...] = (1, 2),
    deaths_grid: tuple[int, ...] | None = None,
    baku_deaths_grid: tuple[int, ...] | None = None,
    hal_deaths_grid: tuple[int, ...] | None = None,
    cpr_grid: tuple[int, ...] = (0, 5),
    config: ExactSearchConfig | None = None,
    rejected_pool_path: str | Path | None = None,
    unresolved_threshold: float = REJECTED_UNRESOLVED_THRESHOLD,
    workers: int = 1,
) -> list[ValueTarget]:
    """Sweep the cartesian product of axis grids; label each state.

    States that fall outside all label-source gates are skipped: the
    Phase-3 MCTS bootstrap is responsible for those. This keeps the
    gen-0 corpus concentrated on positions where exact LP minimax at
    horizon 2 or 3 carries genuine LSR signal.

    Default grid is ~1,000 candidate states; the emitted corpus is
    smaller (only those passing one of the gates).

    With ``workers > 1`` the per-state labeling runs in a
    ``multiprocessing.Pool``. Output ordering matches the single-thread
    path bit-for-bit because ``Pool.map`` preserves input order, so the
    serialized .npz is identical regardless of worker count.
    """
    pinned_table = _build_pinned_table()
    rejected_states: list[ExactPublicState] = []

    death_pairs = _death_grid_pairs(
        deaths_grid=deaths_grid,
        baku_deaths_grid=baku_deaths_grid,
        hal_deaths_grid=hal_deaths_grid,
    )

    state_tuples: list[tuple[float, float, float, int, int, int, int]] = [
        (baku_cyl, hal_cyl, clock, half, baku_deaths, hal_deaths, cprs)
        for baku_cyl in baku_cylinder_grid
        for hal_cyl in hal_cylinder_grid
        for clock in clock_grid
        for half in half_grid
        for baku_deaths, hal_deaths in death_pairs
        for cprs in cpr_grid
    ]

    if workers > 1:
        from multiprocessing import Pool
        with Pool(
            processes=workers,
            initializer=_worker_init,
            initargs=(config, pinned_table, unresolved_threshold),
        ) as pool:
            results = pool.map(_label_one_state, state_tuples)
    else:
        _worker_init(config, pinned_table, unresolved_threshold)
        results = [_label_one_state(st) for st in state_tuples]

    targets: list[ValueTarget] = []
    for target, rejected in results:
        if rejected is not None:
            rejected_states.append(rejected)
        elif target is not None:
            targets.append(target)

    if rejected_pool_path is not None:
        _save_rejected_pool(rejected_states, rejected_pool_path)
    return targets


# ── Held-out reference corpus ─────────────────────────────────────────────

# Grids deliberately disjoint from generate_targets() defaults on every axis
# where disjointness is possible (half and deaths only have two valid values).
# The held-out corpus is the fixed ruler against which calibration gates
# measure each generation, so it MUST not overlap the training set in any
# axis where overlap is avoidable.
HOLDOUT_BAKU_CYL_GRID: tuple[float, ...] = (30.0, 150.0, 270.0, 294.0)
HOLDOUT_HAL_CYL_GRID: tuple[float, ...] = (60.0, 180.0)
HOLDOUT_CLOCK_GRID: tuple[float, ...] = (1200.0, 2400.0, 3450.0)
HOLDOUT_HALF_GRID: tuple[int, ...] = (1, 2)
HOLDOUT_BAKU_DEATHS_GRID: tuple[int, ...] = (0, 1)
HOLDOUT_HAL_DEATHS_GRID: tuple[int, ...] = (0, 1)
HOLDOUT_CPR_GRID: tuple[int, ...] = (1, 4)

# Programmatically constructed terminal states. Each tuple is
# (winner_name, baku_cylinder, hal_cylinder, clock, current_half).
# These cover both winners across the clock range, with varied cylinders
# on the surviving and the deceased side. Feature vectors are diverse;
# values are exactly +1 (Hal win) or -1 (Baku win).
HOLDOUT_TERMINAL_CONFIGS: tuple[tuple[str, float, float, float, int], ...] = (
    ("Hal", 200.0, 0.0, 1500.0, 1),
    ("Hal", 240.0, 30.0, 1800.0, 2),
    ("Hal", 290.0, 60.0, 2200.0, 1),
    ("Hal", 299.0, 120.0, 2700.0, 2),
    ("Hal", 270.0, 180.0, 3100.0, 1),
    ("Hal", 285.0, 240.0, 3450.0, 2),
    ("Hal", 250.0, 60.0, 1300.0, 1),
    ("Hal", 240.0, 30.0, 2000.0, 2),
    ("Hal", 270.0, 90.0, 2500.0, 1),
    ("Hal", 295.0, 150.0, 2900.0, 2),
    ("Baku", 0.0, 200.0, 1500.0, 1),
    ("Baku", 30.0, 240.0, 1800.0, 2),
    ("Baku", 60.0, 290.0, 2200.0, 1),
    ("Baku", 120.0, 299.0, 2700.0, 2),
    ("Baku", 180.0, 270.0, 3100.0, 1),
    ("Baku", 240.0, 285.0, 3450.0, 2),
    ("Baku", 60.0, 250.0, 1300.0, 1),
    ("Baku", 30.0, 240.0, 2000.0, 2),
    ("Baku", 90.0, 270.0, 2500.0, 1),
    ("Baku", 150.0, 295.0, 2900.0, 2),
    # Edge cases: terminal states inside / after the leap-second window
    ("Hal", 240.0, 0.0, 3540.0, 2),
    ("Hal", 0.0, 240.0, 3590.0, 1),
    ("Baku", 240.0, 0.0, 3590.0, 2),
    ("Baku", 0.0, 240.0, 3540.0, 1),
)


# Terminal-state configurations for the TRAINING corpus. Disjoint from
# HOLDOUT_TERMINAL_CONFIGS on cylinder + clock axes so the held-out
# ruler genuinely measures generalization, not memorization.
TRAINING_TERMINAL_CONFIGS: tuple[tuple[str, float, float, float, int], ...] = (
    ("Hal", 220.0, 0.0, 1000.0, 1),
    ("Hal", 250.0, 100.0, 1700.0, 2),
    ("Hal", 280.0, 160.0, 2300.0, 1),
    ("Hal", 100.0, 220.0, 2800.0, 2),
    ("Hal", 160.0, 250.0, 3200.0, 1),
    ("Hal", 220.0, 280.0, 3500.0, 2),
    ("Hal", 250.0, 100.0, 1000.0, 2),
    ("Hal", 220.0, 160.0, 1700.0, 1),
    ("Hal", 280.0, 100.0, 2300.0, 2),
    ("Hal", 160.0, 220.0, 2800.0, 1),
    ("Hal", 100.0, 250.0, 3200.0, 2),
    ("Hal", 280.0, 220.0, 3500.0, 1),
    ("Baku", 0.0, 220.0, 1000.0, 1),
    ("Baku", 100.0, 250.0, 1700.0, 2),
    ("Baku", 160.0, 280.0, 2300.0, 1),
    ("Baku", 220.0, 100.0, 2800.0, 2),
    ("Baku", 250.0, 160.0, 3200.0, 1),
    ("Baku", 280.0, 220.0, 3500.0, 2),
    ("Baku", 100.0, 250.0, 1000.0, 2),
    ("Baku", 160.0, 220.0, 1700.0, 1),
    ("Baku", 100.0, 280.0, 2300.0, 2),
    ("Baku", 220.0, 160.0, 2800.0, 1),
    ("Baku", 250.0, 100.0, 3200.0, 2),
    ("Baku", 280.0, 220.0, 3500.0, 1),
)


def _generate_terminal_targets(
    config: ExactSearchConfig | None,
    terminal_configs: tuple[tuple[str, float, float, float, int], ...] = HOLDOUT_TERMINAL_CONFIGS,
) -> list[ValueTarget]:
    """Materialize terminal states from a configuration list. Serial — each
    state is cheap (no LP solve). Used by both held-out and training corpora
    with their respective disjoint config lists.
    """
    config = config or ExactSearchConfig()
    targets: list[ValueTarget] = []
    for winner, baku_cyl, hal_cyl, clock, half in terminal_configs:
        game = _build_terminal_game(
            winner_name=winner,
            baku_cylinder=baku_cyl,
            hal_cylinder=hal_cyl,
            clock=clock,
            current_half=half,
        )
        tval = terminal_value(game, perspective_name=config.perspective_name)
        if tval is None:
            # Defensive — _build_terminal_game sets game_over=True, so this
            # branch should be unreachable. Skip rather than emit a bad label.
            continue
        drop_dist, check_dist, drop_mask, check_mask = _legal_policy_vectors(game, config)
        targets.append(
            ValueTarget(
                features=extract_features(game),
                value=tval,
                source=SOURCE_TERMINAL,
                horizon=0,
                dropper_dist=drop_dist,
                checker_dist=check_dist,
                dropper_legal_mask=drop_mask,
                checker_legal_mask=check_mask,
                unresolved_probability=0.0,
            )
        )
    return targets


def _generate_tablebase_targets(
    config: ExactSearchConfig | None,
    *,
    split_interior: bool = False,
) -> list[ValueTarget]:
    """Materialize tablebase-pinned scenarios for the held-out corpus.

    Walks the tablebase REGISTRY and emits every scenario with a non-None
    ``expected_value``. Each becomes a SOURCE_TABLEBASE target with the
    pinned ground-truth value — no LP solve required.

    The trained net's MSE on these is the cleanest possible regression
    indicator: pinned values are exact by construction, so any drift is
    a learning regression (catastrophic forgetting), not approximation
    error. The calibration gate's tablebase_mse_threshold (default 0.01)
    fires on this.

    Args:
        split_interior: When True (held-out path only), Phase F-2 interior
            pins (tagged ``INTERIOR_PIN_TAG``) are labeled
            ``SOURCE_TABLEBASE_INTERIOR`` instead of ``SOURCE_TABLEBASE`` so the
            calibration gate scores them as a distinct class. The 19 all-±1
            boundary pins otherwise average-mask a miscalibrated interior. The
            training path leaves this False, so the training corpus is unchanged.
    """
    config = config or ExactSearchConfig()
    targets: list[ValueTarget] = []
    for factory in REGISTRY.values():
        scenario = factory()
        if scenario.expected_value is None:
            continue
        game = scenario.game
        drop_dist, check_dist, drop_mask, check_mask = _legal_policy_vectors(game, config)
        is_interior = split_interior and INTERIOR_PIN_TAG in scenario.tags
        targets.append(
            ValueTarget(
                features=extract_features(game),
                value=float(scenario.expected_value),
                source=SOURCE_TABLEBASE_INTERIOR if is_interior else SOURCE_TABLEBASE,
                horizon=0,
                dropper_dist=drop_dist,
                checker_dist=check_dist,
                dropper_legal_mask=drop_mask,
                checker_legal_mask=check_mask,
                unresolved_probability=0.0,
            )
        )
    return targets


def generate_holdout_targets(
    *,
    config: ExactSearchConfig | None = None,
    rejected_pool_path: str | Path | None = None,
    unresolved_threshold: float = REJECTED_UNRESOLVED_THRESHOLD,
    workers: int = 1,
) -> list[ValueTarget]:
    """Build the held-out reference corpus used by Phase 4 calibration gates.

    Three concatenated parts:
      1. Exact-LP sweep over a non-overlapping shifted grid (terminal,
         tablebase, exact_horizon_2, exact_horizon_3 sources depending on
         each state's gate).
      2. Programmatically constructed terminal states (SOURCE_TERMINAL),
         guaranteeing the terminal class is always present.
      3. Implicitly, any holdout tablebase scenarios that fall under the
         exact-LP path (the training pinned_table excludes holdout=True
         scenarios, so they pass through to the LP solver here and are
         labeled with high-confidence horizon-3 values).

    Output is the same .npz schema as generate_targets, so the calibration
    gate can consume it via load_targets_as_records().
    """
    lp_targets = generate_targets(
        baku_cylinder_grid=HOLDOUT_BAKU_CYL_GRID,
        hal_cylinder_grid=HOLDOUT_HAL_CYL_GRID,
        clock_grid=HOLDOUT_CLOCK_GRID,
        half_grid=HOLDOUT_HALF_GRID,
        baku_deaths_grid=HOLDOUT_BAKU_DEATHS_GRID,
        hal_deaths_grid=HOLDOUT_HAL_DEATHS_GRID,
        cpr_grid=HOLDOUT_CPR_GRID,
        config=config,
        rejected_pool_path=rejected_pool_path,
        unresolved_threshold=unresolved_threshold,
        workers=workers,
    )
    terminal_targets = _generate_terminal_targets(config, HOLDOUT_TERMINAL_CONFIGS)
    tablebase_targets = _generate_tablebase_targets(config, split_interior=True)
    return lp_targets + terminal_targets + tablebase_targets


# ── Phase 3: MCTS bootstrap targets ───────────────────────────────────────


def generate_mcts_bootstrap_targets(
    predict_fn,
    *,
    baku_cylinder_grid: tuple[float, ...] = (0.0, 60.0, 120.0, 180.0, 240.0, 290.0),
    hal_cylinder_grid: tuple[float, ...] = (0.0, 120.0, 240.0),
    clock_grid: tuple[float, ...] = (720.0, 2000.0, 3540.0),
    half_grid: tuple[int, ...] = (1, 2),
    deaths_grid: tuple[int, ...] | None = None,
    baku_deaths_grid: tuple[int, ...] | None = None,
    hal_deaths_grid: tuple[int, ...] | None = None,
    cpr_grid: tuple[int, ...] = (0, 5),
    iterations_per_state: int = 2000,
    exploration_c: float = 1.0,
    seed: int = 0,
    config: ExactSearchConfig | None = None,
    include_anchor_classes: bool = True,
    subgame_resolve_at_critical: bool = False,
    subgame_resolve_horizon: int = 1,
    subgame_resolve_cfr_iters: int = 2000,
    bootstrap_critical_only: bool = False,
    bootstrap_max_states: int | None = None,
    split_interior: bool = False,
) -> list[ValueTarget]:
    """Bootstrap labels: run MCTS using ``predict_fn`` at the leaves and record
    the converged root value as the training target.

    AlphaZero's self-improvement loop. The ``predict_fn`` is a callable
    ``(game) -> (value, dropper_dist, checker_dist)`` produced by wrapping
    a trained ValueNet (e.g. via ``ValueNetEvaluator``). For each
    non-terminal, non-tablebase state in the corpus, we run ``mcts_search`` with that predictor as the
    leaf evaluator and record ``result.root_value_for_hal`` as the label.
    Source = ``"mcts_bootstrap"``; horizon is reported as
    ``iterations_per_state`` (a search-budget proxy, not an exact horizon).

    When ``include_anchor_classes=True`` (default), the resulting corpus
    is augmented with programmatically constructed terminal states
    (TRAINING_TERMINAL_CONFIGS) and tablebase REGISTRY entries so the
    training set spans all four exact source classes — preventing the
    "MCTS-only training drift" regression mode the calibration gate
    catches.

    Imports are deferred to keep this module's import graph tight: MCTS
    machinery is only needed when the bootstrap is actually invoked.
    """
    from stl.solver.exact import CFRPlusConfig
    from stl.solver.search import ValueNetEvaluator
    from stl.solver.search import MCTSConfig, mcts_search
    from stl.solver.search import is_critical

    config = config or ExactSearchConfig()
    pinned_table = _build_pinned_table()
    targets: list[ValueTarget] = []
    rng_root = np.random.default_rng(seed)
    evaluator = ValueNetEvaluator(model_fn=predict_fn)
    death_pairs = _death_grid_pairs(
        deaths_grid=deaths_grid,
        baku_deaths_grid=baku_deaths_grid,
        hal_deaths_grid=hal_deaths_grid,
    )
    bootstrap_count = 0

    for baku_cyl in baku_cylinder_grid:
        for hal_cyl in hal_cylinder_grid:
            for clock in clock_grid:
                for half in half_grid:
                    for baku_deaths, hal_deaths in death_pairs:
                        for cprs in cpr_grid:
                            game = _build_game(
                                baku_cylinder=baku_cyl,
                                hal_cylinder=hal_cyl,
                                clock=clock,
                                current_half=half,
                                baku_deaths=baku_deaths,
                                hal_deaths=hal_deaths,
                                referee_cprs=cprs,
                            )

                            tval = terminal_value(game, perspective_name=config.perspective_name)
                            if tval is not None:
                                drop_dist, check_dist, drop_mask, check_mask = _legal_policy_vectors(game, config)
                                targets.append(
                                    ValueTarget(
                                        features=extract_features(game),
                                        value=tval,
                                        source=SOURCE_TERMINAL,
                                        horizon=0,
                                        dropper_dist=drop_dist,
                                        checker_dist=check_dist,
                                        dropper_legal_mask=drop_mask,
                                        checker_legal_mask=check_mask,
                                        unresolved_probability=0.0,
                                    )
                                )
                                continue

                            state = exact_public_state(game)
                            if state in pinned_table:
                                drop_dist, check_dist, drop_mask, check_mask = _legal_policy_vectors(game, config)
                                targets.append(
                                    ValueTarget(
                                        features=extract_features(game),
                                        value=pinned_table[state],
                                        source=SOURCE_TABLEBASE,
                                        horizon=0,
                                        dropper_dist=drop_dist,
                                        checker_dist=check_dist,
                                        dropper_legal_mask=drop_mask,
                                        checker_legal_mask=check_mask,
                                        unresolved_probability=0.0,
                                    )
                                )
                                continue

                            if bootstrap_critical_only and not is_critical(game):
                                continue
                            if (
                                bootstrap_max_states is not None
                                and bootstrap_count >= bootstrap_max_states
                            ):
                                continue

                            state_seed = int(rng_root.integers(0, 1 << 31))
                            mcts_rng = np.random.default_rng(state_seed)
                            mcts_config = MCTSConfig(
                                iterations=iterations_per_state,
                                exploration_c=exploration_c,
                                evaluator=None,
                                use_tablebase=False,
                            )
                            result = mcts_search(
                                game=game,
                                config=mcts_config,
                                evaluator=evaluator,
                                rng=mcts_rng,
                                exact_config=config,
                                subgame_resolve_at_critical=subgame_resolve_at_critical,
                                subgame_resolve_horizon=subgame_resolve_horizon,
                                subgame_resolve_cfr_plus_config=CFRPlusConfig(
                                    iterations=subgame_resolve_cfr_iters,
                                ),
                            )
                            drop_dist, check_dist = _strategy_vectors(
                                drop_seconds=result.root_drop_seconds,
                                check_seconds=result.root_check_seconds,
                                dropper_strategy=result.root_strategy_dropper,
                                checker_strategy=result.root_strategy_checker,
                            )
                            _, _, drop_mask, check_mask = _legal_policy_vectors(game, config)
                            targets.append(
                                ValueTarget(
                                    features=extract_features(game),
                                    value=float(result.root_value_for_hal),
                                    source=SOURCE_MCTS_BOOTSTRAP,
                                    horizon=iterations_per_state,
                                    dropper_dist=drop_dist,
                                    checker_dist=check_dist,
                                    dropper_legal_mask=drop_mask,
                                    checker_legal_mask=check_mask,
                                    unresolved_probability=0.0,
                                )
                            )
                            bootstrap_count += 1

    if include_anchor_classes:
        targets.extend(_generate_terminal_targets(config, TRAINING_TERMINAL_CONFIGS))
        targets.extend(_generate_tablebase_targets(config, split_interior=split_interior))

    return targets


# ── I/O ───────────────────────────────────────────────────────────────────


def save_targets(targets: list[ValueTarget], path: str | Path) -> None:
    """Write targets to .npz with arrays X, y, sources, horizons."""
    if targets:
        X = np.stack([t.features for t in targets]).astype(np.float32)
        dropper_dists = np.stack([t.dropper_dist for t in targets]).astype(np.float32)
        checker_dists = np.stack([t.checker_dist for t in targets]).astype(np.float32)
        dropper_masks = np.stack([t.dropper_legal_mask for t in targets]).astype(np.float32)
        checker_masks = np.stack([t.checker_legal_mask for t in targets]).astype(np.float32)
    else:
        X = np.zeros((0, FEATURE_DIM), dtype=np.float32)
        dropper_dists = np.zeros((0, 61), dtype=np.float32)
        checker_dists = np.zeros((0, 61), dtype=np.float32)
        dropper_masks = np.zeros((0, 61), dtype=np.float32)
        checker_masks = np.zeros((0, 61), dtype=np.float32)
    y = np.array([t.value for t in targets], dtype=np.float32)
    sources = np.array([t.source for t in targets])
    horizons = np.array([t.horizon for t in targets], dtype=np.int32)
    unresolved = np.array([t.unresolved_probability for t in targets], dtype=np.float32)
    np.savez(
        path,
        X=X,
        y=y,
        sources=sources,
        horizons=horizons,
        dropper_dists=dropper_dists,
        checker_dists=checker_dists,
        dropper_legal_masks=dropper_masks,
        checker_legal_masks=checker_masks,
        unresolved_probabilities=unresolved,
    )


def load_targets(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load (X, y) arrays from a saved .npz file."""
    data = np.load(path, allow_pickle=True)
    return data["X"], data["y"]


def load_targets_as_records(path: str | Path) -> list[ValueTarget]:
    """Load saved .npz back into ``list[ValueTarget]`` records.

    Reconstructs the dataclass form including policy distributions, legal
    masks, and unresolved_probability when present (older corpora without
    those fields default to zero arrays / 0.0).
    """
    data = np.load(path, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.float32)
    sources = np.array(data["sources"]).astype(str)
    horizons = data["horizons"].astype(np.int32) if "horizons" in data else np.zeros(len(X), dtype=np.int32)
    n = len(X)
    dropper_dists = data["dropper_dists"].astype(np.float32) if "dropper_dists" in data else np.zeros((n, 61), dtype=np.float32)
    checker_dists = data["checker_dists"].astype(np.float32) if "checker_dists" in data else np.zeros((n, 61), dtype=np.float32)
    dropper_masks = data["dropper_legal_masks"].astype(np.float32) if "dropper_legal_masks" in data else np.zeros((n, 61), dtype=np.float32)
    checker_masks = data["checker_legal_masks"].astype(np.float32) if "checker_legal_masks" in data else np.zeros((n, 61), dtype=np.float32)
    unresolved = data["unresolved_probabilities"].astype(np.float32) if "unresolved_probabilities" in data else np.zeros(n, dtype=np.float32)
    records = [
        ValueTarget(
            features=X[i],
            value=float(y[i]),
            source=str(sources[i]),
            horizon=int(horizons[i]),
            dropper_dist=dropper_dists[i],
            checker_dist=checker_dists[i],
            dropper_legal_mask=dropper_masks[i],
            checker_legal_mask=checker_masks[i],
            unresolved_probability=float(unresolved[i]),
        )
        for i in range(n)
    ]
    return records


def source_breakdown(targets: list[ValueTarget]) -> dict[str, int]:
    """Count targets by label source for diagnostics."""
    counts: dict[str, int] = {}
    for t in targets:
        counts[t.source] = counts.get(t.source, 0) + 1
    return counts


# ── CLI ───────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=str, default="value_targets.npz")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker processes for per-state labeling.",
    )
    parser.add_argument(
        "--holdout",
        action="store_true",
        help="Build the held-out reference corpus (shifted grid + terminal "
        "states) rather than the training corpus. Used as the fixed ruler "
        "for Phase 4 calibration gates.",
    )
    args = parser.parse_args()

    t0 = time.time()
    rejected_path = f"{args.out}.rejected.npz"
    if args.holdout:
        targets = generate_holdout_targets(
            rejected_pool_path=rejected_path,
            workers=args.workers,
        )
    else:
        targets = generate_targets(
            rejected_pool_path=rejected_path,
            workers=args.workers,
        )
    elapsed = time.time() - t0
    save_targets(targets, args.out)
    print(f"Wrote {len(targets)} targets to {args.out} in {elapsed:.1f}s")
    print(f"Wrote high-unresolved candidate pool to {rejected_path}")
    print(f"Source breakdown: {source_breakdown(targets)}")
