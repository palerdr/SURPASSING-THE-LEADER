"""Value-net training-target generation.

Sweeps a corpus of game states and labels each with an exact
Hal-perspective equilibrium value. Sources, in priority order:

    1. Terminal positions: pinned by `terminal_value` (+1/-1/0).
    2. Tablebase scenarios: pinned by the registry's expected_value.
    3. Otherwise: solve_exact_finite_horizon at the requested horizon.

No reward shaping, no rollouts. Every label is a genuine
equilibrium value (or a finite-horizon lower bound on it). Output is
a list of ValueTarget records or an .npz file with arrays X (N,
FEATURE_DIM), y (N,), sources (N,), horizons (N,).
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
from hal.value_net import FEATURE_DIM, extract_features
from src.Constants import PHYSICALITY_BAKU, PHYSICALITY_HAL
from src.Game import Game
from src.Player import Player
from src.Referee import Referee


@dataclass(frozen=True)
class ValueTarget:
    features: np.ndarray
    value: float
    source: str
    horizon: int


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


def label_state(
    game: Game,
    horizon: int,
    pinned_table: dict[ExactPublicState, float] | None = None,
    config: ExactSearchConfig | None = None,
) -> tuple[float, str]:
    """Return (Hal-perspective value, source) for a game state."""
    config = config or ExactSearchConfig()
    pinned_table = pinned_table if pinned_table is not None else _build_pinned_table()

    tval = terminal_value(game, perspective_name=config.perspective_name)
    if tval is not None:
        return tval, "terminal"

    state = exact_public_state(game)
    if state in pinned_table:
        return pinned_table[state], "tablebase"

    result = solve_exact_finite_horizon(game, horizon, config)
    return result.value_for_hal, "exact"


def generate_targets(
    *,
    baku_cylinder_grid: tuple[float, ...] = (0.0, 60.0, 120.0, 180.0, 240.0, 290.0, 299.0),
    hal_cylinder_grid: tuple[float, ...] = (0.0, 120.0, 240.0),
    clock_grid: tuple[float, ...] = (720.0, 2000.0, 3540.0),
    half_grid: tuple[int, ...] = (1, 2),
    deaths_grid: tuple[int, ...] = (0, 1),
    cpr_grid: tuple[int, ...] = (0, 5),
    horizon: int = 1,
    config: ExactSearchConfig | None = None,
) -> list[ValueTarget]:
    """Sweep the cartesian product of axis grids; label each state.

    Default grid is ~500 states; override any axis to tighten or widen
    the corpus.
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
                            value, source = label_state(game, horizon, pinned_table, config)
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


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=str, default="value_targets.npz")
    parser.add_argument("--horizon", type=int, default=1)
    args = parser.parse_args()

    t0 = time.time()
    targets = generate_targets(horizon=args.horizon)
    elapsed = time.time() - t0
    save_targets(targets, args.out)
    print(f"Wrote {len(targets)} targets to {args.out} in {elapsed:.1f}s")
    print(f"Source breakdown: {source_breakdown(targets)}")
