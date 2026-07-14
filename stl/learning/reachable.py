"""Deterministic, engine-derived Generation-Zero candidate trajectories.

Every snapshot in this module starts at the canonical opening and is reached
only through :meth:`Game.resolve_half_round`.  Explicit chance outcomes are
allowed solely when the engine assigns them non-zero probability.  The action
trace is retained so the replay boundary can independently reproduce the row.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable

import numpy as np

from stl.engine.game import (
    CYLINDER_MAX,
    FAILED_CHECK_PENALTY,
    PHYSICALITY_BAKU,
    PHYSICALITY_HAL,
    Game,
    Player,
    Referee,
)
from stl.learning.model import extract_features
from stl.learning.replay import exact_state_hash
from stl.solver.exact import ExactPublicState, exact_public_state


@dataclass(frozen=True)
class TrajectoryStep:
    drop_time: int
    check_time: int
    survived_outcome: bool | None = None

    def as_tuple(self) -> tuple[int, int, bool | None]:
        return (self.drop_time, self.check_time, self.survived_outcome)


@dataclass(frozen=True)
class TrajectoryRecipe:
    episode_id: str
    split_family: str
    steps: tuple[TrajectoryStep, ...]
    capture_after_steps: tuple[int, ...]


@dataclass(frozen=True)
class ReachableSnapshot:
    exact_state: ExactPublicState
    episode_id: str
    half_round_index: int
    trajectory_actions: tuple[tuple[int, int, bool | None], ...]


@dataclass(frozen=True)
class ReachableSplit:
    train: tuple[ReachableSnapshot, ...]
    ruler: tuple[ReachableSnapshot, ...]


def canonical_game() -> Game:
    hal = Player(name="Hal", physicality=PHYSICALITY_HAL)
    baku = Player(name="Baku", physicality=PHYSICALITY_BAKU)
    game = Game(player1=hal, player2=baku, referee=Referee(), first_dropper=hal)
    game.seed(0)
    return game


def validate_physical_state(game: Game, *, engine_snapshot: bool) -> None:
    deaths = game.player1.deaths + game.player2.deaths
    if game.referee.cprs_performed != deaths:
        raise ValueError("referee CPR count must equal completed death attempts")
    for player in (game.player1, game.player2):
        if player.deaths < 0:
            raise ValueError("player death count must be non-negative")
        if not 60.0 * player.deaths <= player.ttd <= CYLINDER_MAX * player.deaths:
            raise ValueError("player TTD is incompatible with the death count")
        if player.cylinder < 0.0:
            raise ValueError("player cylinder must be non-negative")
        if engine_snapshot and not game.game_over and player.cylinder >= CYLINDER_MAX:
            raise ValueError("nonterminal engine snapshots cannot retain overflow")
    if game.game_over:
        if game.winner is None or game.loser is None or game.winner is game.loser:
            raise ValueError("terminal state requires distinct winner and loser")
        if game.loser.alive or not game.winner.alive:
            raise ValueError("terminal alive flags do not match winner and loser")
        if game.loser.deaths <= 0 or game.loser.ttd <= 0.0:
            raise ValueError("terminal loser must contain the fatal death mechanics")
    elif game.winner is not None or game.loser is not None:
        raise ValueError("nonterminal state may not name a winner or loser")
    elif not game.player1.alive or not game.player2.alive:
        raise ValueError("nonterminal state requires both players alive")


def _apply_step(game: Game, step: TrajectoryStep, *, index: int) -> None:
    if game.game_over:
        raise ValueError(f"trajectory continues after terminal action {index}")
    _dropper, checker = game.get_roles_for_half(game.current_half)
    success = step.check_time >= step.drop_time
    addition = (
        float(step.check_time - step.drop_time)
        if success
        else float(FAILED_CHECK_PENALTY)
    )
    death_occurs = checker.cylinder + addition >= CYLINDER_MAX if success else True
    if death_occurs:
        if step.survived_outcome is None:
            raise ValueError(f"death action {index} requires an explicit chance outcome")
        duration = min(checker.cylinder + addition, CYLINDER_MAX)
        probability = game.referee.compute_survival_probability(checker, duration)
        if step.survived_outcome and probability <= 0.0:
            raise ValueError(f"action {index} forces a zero-probability survival")
        if not step.survived_outcome and probability >= 1.0:
            raise ValueError(f"action {index} forces a zero-probability death")
    elif step.survived_outcome is not None:
        raise ValueError(f"non-death action {index} may not carry a chance outcome")
    game.resolve_half_round(
        step.drop_time,
        step.check_time,
        survived_outcome=step.survived_outcome,
    )


def execute_recipe(recipe: TrajectoryRecipe) -> tuple[ReachableSnapshot, ...]:
    if not recipe.episode_id:
        raise ValueError("trajectory recipe requires an episode_id")
    captures = set(recipe.capture_after_steps)
    if not captures or min(captures) <= 0 or max(captures) > len(recipe.steps):
        raise ValueError("capture indices must name completed trajectory steps")
    game = canonical_game()
    actions: list[tuple[int, int, bool | None]] = []
    snapshots: list[ReachableSnapshot] = []
    for index, step in enumerate(recipe.steps, start=1):
        _apply_step(game, step, index=index)
        actions.append(step.as_tuple())
        if index in captures:
            validate_physical_state(game, engine_snapshot=True)
            snapshots.append(
                ReachableSnapshot(
                    exact_state=exact_public_state(game),
                    episode_id=recipe.episode_id,
                    half_round_index=index,
                    trajectory_actions=tuple(actions),
                )
            )
    if len(snapshots) != len(captures):
        raise AssertionError("trajectory capture count mismatch")
    return tuple(snapshots)


def _success(gain: int) -> TrajectoryStep:
    if not 0 < gain < 60:
        raise ValueError("successful accumulation gain must be in [1, 59]")
    return TrajectoryStep(60 - gain, 60)


def _exact_pair(gain: int) -> tuple[TrajectoryRecipe, TrajectoryRecipe]:
    minor = _success(1)
    major = _success(gain)
    hal_failure = TrajectoryStep(60, 1, True)
    baku_failure = TrajectoryStep(60, 1, True)

    hal_pressure_steps = tuple([major, minor] * 5 + [minor, hal_failure] * 2)
    baku_pressure_steps = tuple([minor, major] * 5 + [baku_failure, minor] * 2)
    family = f"exact-pressure-g{gain}"
    return (
        TrajectoryRecipe(
            episode_id=f"gen0-{family}-hal-failures",
            split_family=family,
            steps=hal_pressure_steps,
            capture_after_steps=(12, 14),
        ),
        TrajectoryRecipe(
            episode_id=f"gen0-{family}-baku-failures",
            split_family=family,
            steps=baku_pressure_steps,
            capture_after_steps=(12, 14),
        ),
    )


def _terminal_pair() -> tuple[TrajectoryRecipe, TrajectoryRecipe]:
    # Four 59-second gains produce 236 seconds.  A final four-second gain puts
    # the future checker at 240; the failed check then injects exactly 300 and
    # has zero survival probability under the engine.
    success = _success(59)
    adjustment = _success(4)
    minor = _success(1)
    fatal = TrajectoryStep(60, 1, False)
    family = "terminal-certain-300"
    hal_win_steps = tuple([success, success] * 4 + [adjustment, minor, fatal])
    baku_win_steps = tuple(
        [success, success] * 4 + [minor, adjustment, minor, fatal]
    )
    return (
        TrajectoryRecipe(
            episode_id=f"gen0-{family}-hal-win",
            split_family=family,
            steps=hal_win_steps,
            capture_after_steps=(len(hal_win_steps),),
        ),
        TrajectoryRecipe(
            episode_id=f"gen0-{family}-baku-win",
            split_family=family,
            steps=baku_win_steps,
            capture_after_steps=(len(baku_win_steps),),
        ),
    )


def generation_recipes(*, smoke: bool) -> tuple[TrajectoryRecipe, ...]:
    gains = (59,) if smoke else (48, 50, 55, 59)
    recipes: list[TrajectoryRecipe] = []
    for gain in gains:
        recipes.extend(_exact_pair(gain))
    # 60 reaches 300 in five checks, so the following failed check is a certain
    # terminal death.  Equal-time actions are deliberately avoided.
    recipes.extend(_terminal_pair())
    return tuple(recipes)


def _rank_key(seed: int, episode_id: str) -> str:
    return hashlib.sha256(f"{seed}:{episode_id}".encode("utf-8")).hexdigest()


def _assert_disjoint(
    train: Iterable[ReachableSnapshot], ruler: Iterable[ReachableSnapshot]
) -> None:
    train = tuple(train)
    ruler = tuple(ruler)
    train_states = {exact_state_hash(item.exact_state) for item in train}
    ruler_states = {exact_state_hash(item.exact_state) for item in ruler}
    if train_states & ruler_states:
        raise ValueError("reachable train/ruler exact-state hashes overlap")
    train_features = {
        hashlib.sha256(extract_features_from_state(item.exact_state).tobytes()).hexdigest()
        for item in train
    }
    ruler_features = {
        hashlib.sha256(extract_features_from_state(item.exact_state).tobytes()).hexdigest()
        for item in ruler
    }
    if train_features & ruler_features:
        raise ValueError("reachable train/ruler feature hashes overlap")
    train_episodes = {item.episode_id for item in train}
    ruler_episodes = {item.episode_id for item in ruler}
    if train_episodes & ruler_episodes:
        raise ValueError("reachable train/ruler episode IDs overlap")


def extract_features_from_state(state: ExactPublicState) -> np.ndarray:
    # Imported lazily to keep the candidate builder independent of target logic.
    from stl.learning.replay import reconstruct_game

    return extract_features(reconstruct_game(state))


def split_reachable_candidates(*, smoke: bool, split_seed: int) -> ReachableSplit:
    """Assign whole paired episodes before any target labeling occurs."""

    by_family: dict[str, list[TrajectoryRecipe]] = {}
    for recipe in generation_recipes(smoke=smoke):
        by_family.setdefault(recipe.split_family, []).append(recipe)

    train: list[ReachableSnapshot] = []
    ruler: list[ReachableSnapshot] = []
    for family, recipes in sorted(by_family.items()):
        if len(recipes) != 2:
            raise ValueError(f"split family {family!r} must contain exactly two episodes")
        ordered = sorted(recipes, key=lambda item: _rank_key(split_seed, item.episode_id))
        ruler.extend(execute_recipe(ordered[0]))
        train.extend(execute_recipe(ordered[1]))
    _assert_disjoint(train, ruler)
    return ReachableSplit(tuple(train), tuple(ruler))


__all__ = [
    "ReachableSnapshot",
    "ReachableSplit",
    "TrajectoryRecipe",
    "TrajectoryStep",
    "canonical_game",
    "execute_recipe",
    "generation_recipes",
    "split_reachable_candidates",
    "validate_physical_state",
]
