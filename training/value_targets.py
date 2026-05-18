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

from environment.cfr.exact import (
    ExactPublicState,
    ExactSearchConfig,
    exact_public_state,
    legal_seconds_for_current_role,
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


def _label_one_state(
    state_tuple: tuple[float, float, float, int, int, int],
) -> tuple[ValueTarget | None, ExactPublicState | None]:
    """Per-state worker. Returns ``(target, rejected_state)`` with at most one
    non-None, or ``(None, None)`` for skipped states. Module-level so it
    pickles cleanly for ``multiprocessing.Pool``."""
    baku_cyl, hal_cyl, clock, half, deaths, cprs = state_tuple
    game = _build_game(
        baku_cylinder=baku_cyl,
        hal_cylinder=hal_cyl,
        clock=clock,
        current_half=half,
        baku_deaths=deaths,
        hal_deaths=deaths,
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
    deaths_grid: tuple[int, ...] = (0, 1),
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

    Default grid is ~500 candidate states; the emitted corpus is
    smaller (only those passing one of the gates).

    With ``workers > 1`` the per-state labeling runs in a
    ``multiprocessing.Pool``. Output ordering matches the single-thread
    path bit-for-bit because ``Pool.map`` preserves input order, so the
    serialized .npz is identical regardless of worker count.
    """
    pinned_table = _build_pinned_table()
    rejected_states: list[ExactPublicState] = []

    state_tuples: list[tuple[float, float, float, int, int, int]] = [
        (baku_cyl, hal_cyl, clock, half, deaths, cprs)
        for baku_cyl in baku_cylinder_grid
        for hal_cyl in hal_cylinder_grid
        for clock in clock_grid
        for half in half_grid
        for deaths in deaths_grid
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
    ``(game) -> (value, dropper_dist, checker_dist)`` produced by wrapping
    a trained ValueNet (e.g. via ``ValueNetEvaluator``). For each
    non-terminal, non-tablebase state in the corpus, we run ``mcts_search`` with that predictor as the
    leaf evaluator and record ``result.root_value_for_hal`` as the label.
    Source = ``"mcts_bootstrap"``; horizon is reported as
    ``iterations_per_state`` (a search-budget proxy, not an exact horizon).

    Imports are deferred to keep this module's import graph tight: MCTS
    machinery is only needed when the bootstrap is actually invoked.
    """
    from environment.cfr.evaluator import ValueNetEvaluator
    from environment.cfr.mcts import MCTSConfig, make_node, mcts_search

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

                            state_seed = int(rng_root.integers(0, 1 << 31))
                            mcts_rng = np.random.default_rng(state_seed)
                            mcts_config = MCTSConfig(
                                iterations=iterations_per_state,
                                exploration_c=exploration_c,
                                evaluator=None,
                                use_tablebase=False,
                            )
                            root_node = make_node(game, config, evaluator=evaluator)
                            result = mcts_search(
                                game=game,
                                config=mcts_config,
                                evaluator=evaluator,
                                rng=mcts_rng,
                                exact_config=config,
                            )
                            drop_dist, check_dist = _strategy_vectors(
                                drop_seconds=root_node.drop_seconds,
                                check_seconds=root_node.check_seconds,
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
    args = parser.parse_args()

    t0 = time.time()
    rejected_path = f"{args.out}.rejected.npz"
    targets = generate_targets(rejected_pool_path=rejected_path, workers=args.workers)
    elapsed = time.time() - t0
    save_targets(targets, args.out)
    print(f"Wrote {len(targets)} targets to {args.out} in {elapsed:.1f}s")
    print(f"Wrote high-unresolved candidate pool to {rejected_path}")
    print(f"Source breakdown: {source_breakdown(targets)}")
