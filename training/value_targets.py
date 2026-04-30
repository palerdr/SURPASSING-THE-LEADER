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

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from environment.cfr.exact import (
    ExactPublicState,
    ExactSearchConfig,
    exact_public_state,
    solve_exact_finite_horizon,
    terminal_value,
)
from environment.cfr.tablebase import REGISTRY
from environment.cfr.timing_features import (
    current_checker_fail_would_activate_lsr,
    is_active_lsr,
    rounds_until_leap_window,
)
from hal.value_net import FEATURE_DIM, extract_features
from src.Constants import PHYSICALITY_BAKU, PHYSICALITY_HAL
from src.Game import Game
from src.Player import Player
from src.Referee import Referee


# ── Source labels ─────────────────────────────────────────────────────────

SOURCE_TERMINAL = "terminal"
SOURCE_TABLEBASE = "tablebase"
SOURCE_EXACT_HORIZON_2 = "exact_horizon_2"
SOURCE_EXACT_HORIZON_3 = "exact_horizon_3"
SOURCE_MCTS_BOOTSTRAP = "mcts_bootstrap"

VALID_SOURCES = (
    SOURCE_TERMINAL,
    SOURCE_TABLEBASE,
    SOURCE_EXACT_HORIZON_2,
    SOURCE_EXACT_HORIZON_3,
    SOURCE_MCTS_BOOTSTRAP,
)


# ── Gating thresholds ─────────────────────────────────────────────────────

NEAR_LEAP_ROUNDS_THRESHOLD = 2  # rounds_until_leap_window <= this => horizon=3
NEAR_OVERFLOW_CYLINDER_THRESHOLD = 240.0  # any cylinder >= this => horizon=2


# ── Target record ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ValueTarget:
    features: np.ndarray
    value: float
    source: str
    horizon: int


# ── Helpers ───────────────────────────────────────────────────────────────


def _build_pinned_table() -> dict[ExactPublicState, float]:
    """Materialize the tablebase registry's pinned-value lookup."""
    table: dict[ExactPublicState, float] = {}
    for factory in REGISTRY.values():
        scenario = factory()
        if scenario.expected_value is not None:
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
) -> tuple[float, str, int] | None:
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
        return tval, SOURCE_TERMINAL, 0

    state = exact_public_state(game)
    if state in pinned_table:
        return pinned_table[state], SOURCE_TABLEBASE, 0

    if _is_lsr_significant(game):
        result = solve_exact_finite_horizon(game, 3, config)
        return result.value_for_hal, SOURCE_EXACT_HORIZON_3, 3

    if _is_lsr_pressure(game):
        result = solve_exact_finite_horizon(game, 2, config)
        return result.value_for_hal, SOURCE_EXACT_HORIZON_2, 2

    return None


# ── Corpus generation ─────────────────────────────────────────────────────


def generate_targets(
    *,
    baku_cylinder_grid: tuple[float, ...] = (0.0, 60.0, 120.0, 180.0, 240.0, 290.0, 299.0),
    hal_cylinder_grid: tuple[float, ...] = (0.0, 120.0, 240.0),
    clock_grid: tuple[float, ...] = (720.0, 2000.0, 3540.0),
    half_grid: tuple[int, ...] = (1, 2),
    deaths_grid: tuple[int, ...] = (0, 1),
    cpr_grid: tuple[int, ...] = (0, 5),
    config: ExactSearchConfig | None = None,
) -> list[ValueTarget]:
    """Sweep the cartesian product of axis grids; label each state.

    States that fall outside all label-source gates are skipped: the
    Phase-3 MCTS bootstrap is responsible for those. This keeps the
    gen-0 corpus concentrated on positions where exact LP minimax at
    horizon 2 or 3 carries genuine LSR signal.

    Default grid is ~500 candidate states; the emitted corpus is
    smaller (only those passing one of the gates).
    """
    pinned_table = _build_pinned_table()
    targets: list[ValueTarget] = []

    for baku_cyl in baku_cylinder_grid:
        for hal_cyl in hal_cylinder_grid:
            for clock in clock_grid:
                for half in half_grid:
                    for deaths in deaths_grid:
                        for cprs in cpr_grid:
                            game = _build_game(
                                baku_cylinder=baku_cyl,
                                hal_cylinder=hal_cyl,
                                clock=clock,
                                current_half=half,
                                baku_deaths=deaths,
                                hal_deaths=deaths,
                                referee_cprs=cprs,
                            )
                            label = label_state(game, config, pinned_table)
                            if label is None:
                                continue
                            value, source, horizon = label
                            features = extract_features(game)
                            targets.append(
                                ValueTarget(
                                    features=features,
                                    value=value,
                                    source=source,
                                    horizon=horizon,
                                )
                            )
    return targets


# ── Phase 3: MCTS bootstrap targets ───────────────────────────────────────


def generate_mcts_bootstrap_targets(
    predict_fn,
    *,
    baku_cylinder_grid: tuple[float, ...] = (0.0, 60.0, 120.0, 180.0, 240.0, 290.0),
    hal_cylinder_grid: tuple[float, ...] = (0.0, 120.0, 240.0),
    clock_grid: tuple[float, ...] = (720.0, 2000.0, 3540.0),
    half_grid: tuple[int, ...] = (1, 2),
    deaths_grid: tuple[int, ...] = (0, 1),
    cpr_grid: tuple[int, ...] = (0, 5),
    iterations_per_state: int = 2000,
    exploration_c: float = 1.0,
    seed: int = 0,
    config: ExactSearchConfig | None = None,
) -> list[ValueTarget]:
    """Bootstrap labels: run MCTS using ``predict_fn`` at the leaves and record
    the converged root value as the training target.

    AlphaZero's self-improvement loop. The ``predict_fn`` is a callable
    ``(game) -> float`` produced by wrapping a trained ValueNet
    (e.g. via ``ValueNetEvaluator``). For each non-terminal, non-tablebase
    state in the corpus, we run ``mcts_search`` with that predictor as the
    leaf evaluator and record ``result.root_value_for_hal`` as the label.
    Source = ``"mcts_bootstrap"``; horizon is reported as
    ``iterations_per_state`` (a search-budget proxy, not an exact horizon).

    Imports are deferred to keep this module's import graph tight: MCTS
    machinery is only needed when the bootstrap is actually invoked.
    """
    from environment.cfr.evaluator import ValueNetEvaluator
    from environment.cfr.mcts import MCTSConfig, mcts_search

    config = config or ExactSearchConfig()
    pinned_table = _build_pinned_table()
    targets: list[ValueTarget] = []
    rng_root = np.random.default_rng(seed)
    evaluator = ValueNetEvaluator(model_fn=predict_fn)

    for baku_cyl in baku_cylinder_grid:
        for hal_cyl in hal_cylinder_grid:
            for clock in clock_grid:
                for half in half_grid:
                    for deaths in deaths_grid:
                        for cprs in cpr_grid:
                            game = _build_game(
                                baku_cylinder=baku_cyl,
                                hal_cylinder=hal_cyl,
                                clock=clock,
                                current_half=half,
                                baku_deaths=deaths,
                                hal_deaths=deaths,
                                referee_cprs=cprs,
                            )

                            tval = terminal_value(game, perspective_name=config.perspective_name)
                            if tval is not None:
                                targets.append(
                                    ValueTarget(
                                        features=extract_features(game),
                                        value=tval,
                                        source=SOURCE_TERMINAL,
                                        horizon=0,
                                    )
                                )
                                continue

                            state = exact_public_state(game)
                            if state in pinned_table:
                                targets.append(
                                    ValueTarget(
                                        features=extract_features(game),
                                        value=pinned_table[state],
                                        source=SOURCE_TABLEBASE,
                                        horizon=0,
                                    )
                                )
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
                            )
                            targets.append(
                                ValueTarget(
                                    features=extract_features(game),
                                    value=float(result.root_value_for_hal),
                                    source=SOURCE_MCTS_BOOTSTRAP,
                                    horizon=iterations_per_state,
                                )
                            )
    return targets


# ── I/O ───────────────────────────────────────────────────────────────────


def save_targets(targets: list[ValueTarget], path: str | Path) -> None:
    """Write targets to .npz with arrays X, y, sources, horizons."""
    X = np.stack([t.features for t in targets]).astype(np.float32)
    y = np.array([t.value for t in targets], dtype=np.float32)
    sources = np.array([t.source for t in targets])
    horizons = np.array([t.horizon for t in targets], dtype=np.int32)
    np.savez(path, X=X, y=y, sources=sources, horizons=horizons)


def load_targets(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load (X, y) arrays from a saved .npz file."""
    data = np.load(path, allow_pickle=True)
    return data["X"], data["y"]


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
    args = parser.parse_args()

    t0 = time.time()
    targets = generate_targets()
    elapsed = time.time() - t0
    save_targets(targets, args.out)
    print(f"Wrote {len(targets)} targets to {args.out} in {elapsed:.1f}s")
    print(f"Source breakdown: {source_breakdown(targets)}")
