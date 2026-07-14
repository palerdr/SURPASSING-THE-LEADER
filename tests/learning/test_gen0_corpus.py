from __future__ import annotations

import hashlib
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import stl.learning.train as train_module
from stl.engine.actions import legal_mask
from stl.commands.gen0_corpus import _dedupe_by_certificate, _materialize_chunk
from stl.learning.model import extract_features
from stl.learning.reachable import (
    TrajectoryStep,
    execute_recipe,
    generation_recipes,
    split_reachable_candidates,
)
from stl.learning.replay import (
    ShardRole,
    StateOrigin,
    TargetKind,
    TrainingRecordV3,
    exact_state_hash,
    reconstruct_game,
)
from stl.solver.exact import exact_public_state


def test_gen0_candidates_are_replayable_and_split_by_whole_episode():
    split = split_reachable_candidates(smoke=True, split_seed=20260714)

    assert len(split.train) == 3
    assert len(split.ruler) == 3
    train_states = {exact_state_hash(row.exact_state) for row in split.train}
    ruler_states = {exact_state_hash(row.exact_state) for row in split.ruler}
    assert train_states.isdisjoint(ruler_states)
    assert {row.episode_id for row in split.train}.isdisjoint(
        {row.episode_id for row in split.ruler}
    )
    for snapshot in (*split.train, *split.ruler):
        game = reconstruct_game(snapshot.exact_state)
        assert game.referee.cprs_performed == game.player1.deaths + game.player2.deaths
        for player in (game.player1, game.player2):
            assert 60 * player.deaths <= player.ttd <= 300 * player.deaths
        if game.game_over:
            assert game.loser is not None and not game.loser.alive
            assert game.loser.deaths > 0
            assert game.loser.ttd > 0


def test_zero_probability_forced_survival_is_rejected():
    recipe = next(
        item for item in generation_recipes(smoke=True) if "hal-win" in item.episode_id
    )
    impossible = replace(
        recipe,
        steps=(*recipe.steps[:-1], TrajectoryStep(60, 1, True)),
    )
    with pytest.raises(ValueError, match="zero-probability survival"):
        execute_recipe(impossible)


def _fixture_record() -> TrainingRecordV3:
    game = reconstruct_game(split_reachable_candidates(smoke=True, split_seed=7).train[0].exact_state)
    dropper, checker = game.get_roles_for_half(game.current_half)
    drop_mask = legal_mask(dropper.name, "dropper", game.get_turn_duration())
    check_mask = legal_mask(checker.name, "checker", game.get_turn_duration())
    zero = np.zeros(62, dtype=np.float32)
    return TrainingRecordV3(
        features=extract_features(game),
        exact_state=exact_public_state(game),
        value=0.0,
        target_kind=TargetKind.TABLEBASE_VALUE,
        source="resume_fixture",
        dropper_dist=zero,
        checker_dist=zero,
        dropper_legal_mask=drop_mask,
        checker_legal_mask=check_mask,
        state_origin=StateOrigin.TACTICAL_TABLEBASE,
        source_artifact="test_gen0_corpus.py::fixture",
        source_artifact_digest=hashlib.sha256(b"fixture").hexdigest(),
        episode_id="fixture",
    )


def test_committed_chunk_is_reused_and_plan_mismatch_refuses(tmp_path):
    path = tmp_path / "train.fixture.000000.npz"
    calls = 0

    def factory():
        nonlocal calls
        calls += 1
        return [_fixture_record()]

    provenance = {"schema": "test", "plan_digest": "a" * 64}
    first = _materialize_chunk(
        path, role=ShardRole.REPLAY, provenance=provenance, factory=factory
    )
    second = _materialize_chunk(
        path, role=ShardRole.REPLAY, provenance=provenance, factory=factory
    )
    assert calls == 1
    assert exact_state_hash(first[0].exact_state) == exact_state_hash(second[0].exact_state)

    with pytest.raises(Exception, match="plan mismatch"):
        _materialize_chunk(
            path,
            role=ShardRole.REPLAY,
            provenance={"schema": "test", "plan_digest": "b" * 64},
            factory=factory,
        )


def test_certificate_precedence_is_explicit_and_conflicts_fail():
    tactical = _fixture_record()
    approximate = replace(
        tactical,
        source="exact_horizon_2",
        target_kind=TargetKind.EXACT_VALUE,
        state_origin=StateOrigin.ENGINE_TRAJECTORY,
    )
    selected, decisions = _dedupe_by_certificate([approximate, tactical])
    assert len(selected) == 1 and selected[0] is tactical
    assert decisions == {"resume_fixture_over_exact_horizon_2": 1}

    with pytest.raises(RuntimeError, match="conflicting labels"):
        _dedupe_by_certificate([tactical, replace(approximate, value=0.5)])


def test_train_module_exposes_hydra_dispatchable_main(monkeypatch, tmp_path):
    captured = {}

    def fake_train(targets, out, config):
        captured.update(targets=targets, out=out, config=config)
        return SimpleNamespace(
            best_val_mse=0.0,
            best_epoch=0,
            checkpoint_path=str(tmp_path / "best.pt"),
            best_per_source_mse={},
        )

    monkeypatch.setattr(train_module, "train", fake_train)
    monkeypatch.setattr(
        "sys.argv",
        [
            "stl.learning.train",
            "--targets",
            "train-v3.npz",
            "--out",
            "checkpoint-v3",
            "--epochs",
            "1",
            "--batch-size",
            "8",
            "--no-allow-legacy-targets",
        ],
    )

    assert train_module.main() == 0
    assert captured["targets"] == "train-v3.npz"
    assert captured["out"] == "checkpoint-v3"
    assert captured["config"].epochs == 1
    assert captured["config"].batch_size == 8
    assert captured["config"].allow_legacy_targets is False
