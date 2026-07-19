from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
import pytest

from stl.engine.game import Game, PHYSICALITY_BAKU, PHYSICALITY_HAL, Player, Referee
from stl.learning.contracts import (
    EVAL_MAX_HALF_ROUNDS,
    RL_CONFIG_SCHEMA_VERSION,
    TRAIN_MAX_HALF_ROUNDS,
    canonical_config_json,
    config_digest,
    current_role_assignment,
    episode_outcome,
    assess_horizon_sensitivity,
    validate_rl_config,
)


def _game() -> Game:
    hal = Player(name="Hal", physicality=PHYSICALITY_HAL)
    baku = Player(name="Baku", physicality=PHYSICALITY_BAKU)
    game = Game(hal, baku, Referee(), first_dropper=hal)
    game.seed(7)
    return game


def test_safe_diagonal_is_wrapper_truncated_without_engine_termination():
    game = _game()
    cap = 6
    for half_round in range(1, cap + 1):
        game.play_half_round(1, 1)
        outcome = episode_outcome(
            game,
            half_rounds=half_round,
            max_half_rounds=cap,
        )
        if half_round < cap:
            assert outcome is None

    assert not game.game_over
    assert game.winner is None
    assert outcome is not None
    assert outcome.value_for_hal == 0.0
    assert outcome.truncated
    assert not outcome.terminal


def test_engine_terminal_outcome_takes_precedence_over_cap():
    game = _game()
    game.game_over = True
    game.winner = game.player1
    game.loser = game.player2

    outcome = episode_outcome(game, half_rounds=64, max_half_rounds=64)

    assert outcome is not None
    assert outcome.value_for_hal == 1.0
    assert outcome.terminal
    assert not outcome.truncated

    game.winner, game.loser = game.player2, game.player1
    outcome = episode_outcome(game, half_rounds=1, max_half_rounds=64)
    assert outcome is not None
    assert outcome.value_for_hal == -1.0


def test_role_mapping_keeps_hal_as_maximizer_on_both_halves():
    game = _game()
    first = current_role_assignment(game)
    assert first.hal_role == "dropper"
    assert first.baku_role == "checker"
    assert first.maximizing_actor == "Hal"

    game.current_half = 2
    second = current_role_assignment(game)
    assert second.hal_role == "checker"
    assert second.baku_role == "dropper"
    assert second.maximizing_actor == "Hal"


def test_baseline_hydra_config_is_versioned_resolved_and_hash_stable():
    config_dir = str(Path(__file__).resolve().parents[2] / "config")
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        first = compose(config_name="config")
        second = compose(config_name="config")

    assert first.rl.schema_version == RL_CONFIG_SCHEMA_VERSION
    assert first.rl.episode.train_max_half_rounds == TRAIN_MAX_HALF_ROUNDS
    assert first.rl.episode.eval_max_half_rounds == EVAL_MAX_HALF_ROUNDS
    assert first.rl.mcts.matrix_solver == "lp"
    assert first.rl.model.action_size == 62
    assert config_digest(first.rl) == config_digest(second.rl)
    assert canonical_config_json(first.rl) == canonical_config_json(second.rl)
    assert "stl_solver_rs" not in canonical_config_json(first.rl)
    assert validate_rl_config(first.rl)["schema_version"] == RL_CONFIG_SCHEMA_VERSION


def test_config_digest_is_mapping_order_independent_and_value_sensitive():
    assert config_digest({"a": 1, "b": 2}) == config_digest({"b": 2, "a": 1})
    assert config_digest({"a": 1}) != config_digest({"a": 2})


def test_horizon_sensitivity_reports_threshold_crossings():
    stable = assess_horizon_sensitivity(
        score_at_64=0.51,
        score_at_128=0.52,
        shared_state_policy_tv=0.08,
    )
    assert not stable.outcome_relevant
    assert stable.paired_score_delta == pytest.approx(0.01)

    relevant = assess_horizon_sensitivity(
        score_at_64=0.50,
        score_at_128=0.53,
        shared_state_policy_tv=0.05,
    )
    assert relevant.outcome_relevant
