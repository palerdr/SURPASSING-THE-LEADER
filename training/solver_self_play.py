"""Solver-backed self-play data for the existing value/policy trainer.

This is the replacement path for new RL-style data. It does not use the
legacy ``hal.self_play`` CanonicalHal loop. Labels come from ``SolverAgent``
search and are saved as normal ``ValueTarget`` rows with source
``self_play_mcts`` so the current training pipeline can consume them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from environment.cfr.exact import ExactSearchConfig
from environment.legal_actions import validate_action
from environment.opponents.base import Opponent
from environment.opponents.factory import create_scripted_opponent
from hal.agent import SolverAgent
from hal.value_net import FEATURE_DIM, extract_features
from src.Constants import OPENING_START_CLOCK, PHYSICALITY_BAKU, PHYSICALITY_HAL
from src.Game import Game
from src.Player import Player
from src.Referee import Referee
from training.value_targets import (
    SOURCE_SELF_PLAY_MCTS,
    ValueTarget,
    _legal_policy_vectors,
    _strategy_vectors,
)


@dataclass(frozen=True)
class SelfPlayRecord:
    """One state label plus the metadata needed to audit self-play origin."""

    target: ValueTarget
    root_value: float
    final_outcome: float
    seed: int
    half_round_index: int
    hal_checkpoint_id: str
    opponent_checkpoint_id: str | None
    opponent_source: str


def _new_game(seed: int) -> Game:
    hal = Player(name="Hal", physicality=PHYSICALITY_HAL)
    baku = Player(name="Baku", physicality=PHYSICALITY_BAKU)
    game = Game(player1=hal, player2=baku, referee=Referee(), first_dropper=hal)
    game.seed(seed)
    game.game_clock = OPENING_START_CLOCK
    game.current_half = 1
    return game


def _target_from_search(
    game: Game,
    agent: SolverAgent,
    config: ExactSearchConfig,
) -> ValueTarget:
    result = agent.search(game)
    drop_dist, check_dist = _strategy_vectors(
        drop_seconds=result.root_drop_seconds,
        check_seconds=result.root_check_seconds,
        dropper_strategy=result.root_strategy_dropper_avg,
        checker_strategy=result.root_strategy_checker_avg,
    )
    _, _, drop_mask, check_mask = _legal_policy_vectors(game, config)
    return ValueTarget(
        features=extract_features(game),
        value=float(result.root_value_for_hal),
        source=SOURCE_SELF_PLAY_MCTS,
        horizon=int(agent.iterations),
        dropper_dist=drop_dist,
        checker_dist=check_dist,
        dropper_legal_mask=drop_mask,
        checker_legal_mask=check_mask,
        unresolved_probability=0.0,
    )


def _checked_action(
    controller,
    game: Game,
    *,
    actor: str,
    role: str,
    turn_duration: int,
) -> int:
    second = int(controller.choose_action(game, role, turn_duration))
    validate_action(second, actor=actor, role=role, turn_duration=turn_duration)
    return second


def _final_outcome(game: Game) -> float:
    if not game.game_over or game.winner is None:
        return 0.0
    return 1.0 if game.winner.name.lower() == "hal" else -1.0


def play_solver_self_play_game(
    *,
    hal_agent: SolverAgent,
    baku_controller: SolverAgent | Opponent,
    seed: int,
    hal_checkpoint_id: str,
    opponent_source: str,
    opponent_checkpoint_id: str | None = None,
    max_half_rounds: int = 200,
    config: ExactSearchConfig | None = None,
) -> list[SelfPlayRecord]:
    """Play one game and label every visited public state with solver MCTS.

    ``hal_agent`` is the search teacher for the emitted labels. The Baku side
    may be another ``SolverAgent`` checkpoint or a scripted opponent; the
    metadata records which source produced the opposing actions.
    """
    config = config or ExactSearchConfig()
    game = _new_game(seed)
    hal_agent.reset()
    baku_controller.reset()

    rows: list[tuple[ValueTarget, float, int]] = []
    half_round = 0
    while not game.game_over and half_round < max_half_rounds:
        target = _target_from_search(game, hal_agent, config)
        rows.append((target, float(target.value), half_round))

        dropper, checker = game.get_roles_for_half(game.current_half)
        turn_duration = game.get_turn_duration()
        if dropper.name.lower() == "hal":
            drop_time = _checked_action(
                hal_agent,
                game,
                actor="hal",
                role="dropper",
                turn_duration=turn_duration,
            )
            check_time = _checked_action(
                baku_controller,
                game,
                actor="baku",
                role="checker",
                turn_duration=turn_duration,
            )
        else:
            drop_time = _checked_action(
                baku_controller,
                game,
                actor="baku",
                role="dropper",
                turn_duration=turn_duration,
            )
            check_time = _checked_action(
                hal_agent,
                game,
                actor="hal",
                role="checker",
                turn_duration=turn_duration,
            )

        game.play_half_round(drop_time, check_time)
        half_round += 1

    outcome = _final_outcome(game)
    return [
        SelfPlayRecord(
            target=target,
            root_value=root_value,
            final_outcome=outcome,
            seed=int(seed),
            half_round_index=int(index),
            hal_checkpoint_id=hal_checkpoint_id,
            opponent_checkpoint_id=opponent_checkpoint_id,
            opponent_source=opponent_source,
        )
        for target, root_value, index in rows
    ]


def generate_solver_self_play_records(
    *,
    hal_checkpoint: str | Path,
    games: int,
    iterations: int,
    seed: int = 0,
    opponent_checkpoint: str | Path | None = None,
    opponent_name: str = "safe",
    max_half_rounds: int = 200,
) -> list[SelfPlayRecord]:
    """Generate solver self-play records from checkpoint-vs-checkpoint or scripted games."""
    hal_id = str(hal_checkpoint)
    hal_agent = SolverAgent(
        hal_checkpoint,
        player_name="Hal",
        iterations=iterations,
        seed=seed,
    )
    if opponent_checkpoint is not None:
        opponent_id = str(opponent_checkpoint)
        opponent_source = "checkpoint"
        baku_controller: SolverAgent | Opponent = SolverAgent(
            opponent_checkpoint,
            player_name="Baku",
            iterations=iterations,
            seed=seed + 10_000,
        )
    else:
        opponent_id = None
        opponent_source = f"scripted:{opponent_name}"
        baku_controller = create_scripted_opponent(opponent_name, seed=seed)

    records: list[SelfPlayRecord] = []
    for game_idx in range(games):
        game_seed = seed + game_idx
        if opponent_checkpoint is None:
            baku_controller = create_scripted_opponent(opponent_name, seed=game_seed)
        records.extend(
            play_solver_self_play_game(
                hal_agent=hal_agent,
                baku_controller=baku_controller,
                seed=game_seed,
                hal_checkpoint_id=hal_id,
                opponent_checkpoint_id=opponent_id,
                opponent_source=opponent_source,
                max_half_rounds=max_half_rounds,
            )
        )
    return records


def targets_from_self_play(records: list[SelfPlayRecord]) -> list[ValueTarget]:
    return [record.target for record in records]


def save_self_play_records(records: list[SelfPlayRecord], path: str | Path) -> None:
    """Save self-play rows as a trainer-compatible NPZ with audit metadata."""
    targets = targets_from_self_play(records)
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

    np.savez(
        path,
        X=X,
        y=np.array([t.value for t in targets], dtype=np.float32),
        sources=np.array([t.source for t in targets]),
        horizons=np.array([t.horizon for t in targets], dtype=np.int32),
        dropper_dists=dropper_dists,
        checker_dists=checker_dists,
        dropper_legal_masks=dropper_masks,
        checker_legal_masks=checker_masks,
        unresolved_probabilities=np.array(
            [t.unresolved_probability for t in targets],
            dtype=np.float32,
        ),
        root_values=np.array([r.root_value for r in records], dtype=np.float32),
        final_outcomes=np.array([r.final_outcome for r in records], dtype=np.float32),
        seeds=np.array([r.seed for r in records], dtype=np.int64),
        half_round_indices=np.array([r.half_round_index for r in records], dtype=np.int32),
        hal_checkpoint_ids=np.array([r.hal_checkpoint_id for r in records]),
        opponent_checkpoint_ids=np.array(
            [r.opponent_checkpoint_id or "" for r in records]
        ),
        opponent_sources=np.array([r.opponent_source for r in records]),
    )
