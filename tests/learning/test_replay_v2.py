from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from stl.engine.actions import ACTION_SIZE, legal_mask
from stl.engine.game import Game, PHYSICALITY_BAKU, PHYSICALITY_HAL, Player, Referee
from stl.learning.model import FEATURE_DIM, extract_features
from stl.learning.replay import (
    DEFAULT_FEATURE_SCHEMA,
    REPLAY_MANIFEST_SCHEMA_V2,
    ReplayValidationError,
    ShardRole,
    TargetKind,
    TrainingRecordV2,
    audit_feature_collisions,
    canonical_exact_state_json,
    exact_state_from_json,
    exact_state_hash,
    grouped_split_indices,
    load_replay_manifest,
    load_replay_shard,
    manifest_path_for,
    reconstruct_game,
    save_replay_shard,
)
from stl.solver.exact import exact_public_state
from stl.solver.tablebase import pinned_scenarios


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _game(index: int) -> Game:
    hal = Player(name="Hal", physicality=PHYSICALITY_HAL)
    baku = Player(name="Baku", physicality=PHYSICALITY_BAKU)
    game = Game(player1=hal, player2=baku, referee=Referee(), first_dropper=hal)
    game.seed(index)
    game.game_clock = 720.0 + index
    game.current_half = 1 + index % 2
    game.round_num = index // 2
    hal.cylinder = float(index % 200)
    baku.cylinder = float((index * 3) % 200)
    hal.ttd = float(index % 60)
    baku.ttd = float((index * 2) % 60)
    return game


def _policy_and_mask(game: Game, role: str) -> tuple[np.ndarray, np.ndarray]:
    dropper, checker = game.get_roles_for_half(game.current_half)
    actor = dropper if role == "dropper" else checker
    mask = legal_mask(actor.name, role, game.get_turn_duration())
    policy = mask.astype(np.float32)
    if policy.sum() > 0.0:
        policy /= policy.sum()
    return policy, mask


def _record(
    index: int,
    *,
    target_kind: TargetKind = TargetKind.EXACT_VALUE,
    episode_id: str | None = None,
) -> TrainingRecordV2:
    game = _game(index)
    dropper_dist, dropper_mask = _policy_and_mask(game, "dropper")
    checker_dist, checker_mask = _policy_and_mask(game, "checker")
    return TrainingRecordV2(
        features=extract_features(game),
        exact_state=exact_public_state(game),
        value=((index % 11) - 5) / 5.0,
        target_kind=target_kind,
        source=f"fixture_{target_kind.value}",
        dropper_dist=dropper_dist,
        checker_dist=checker_dist,
        dropper_legal_mask=dropper_mask,
        checker_legal_mask=checker_mask,
        episode_id=episode_id if episode_id is not None else f"episode-{index // 5}",
        half_round_index=index % 5,
        truncated=index % 13 == 0,
        parent_checkpoint_digest=_digest("parent"),
        search_config_digest=_digest(f"search-{index % 3}"),
        rng_seeds={"chance": index, "mcts": 10_000 + index},
    )


def test_mixed_source_shard_round_trips_100_reconstructable_records(tmp_path: Path):
    kinds = tuple(TargetKind)
    records = [
        _record(index, target_kind=kinds[index % len(kinds)]) for index in range(100)
    ]
    path = tmp_path / "mixed-v2.npz"

    saved_manifest = save_replay_shard(records, path)
    loaded_manifest = load_replay_manifest(path)
    loaded = load_replay_shard(
        path,
        expected_parent_checkpoint_digest=_digest("parent"),
    )

    assert saved_manifest == loaded_manifest
    assert loaded_manifest["schema"] == REPLAY_MANIFEST_SCHEMA_V2
    assert loaded_manifest["record_count"] == 100
    assert loaded_manifest["feature_schema"] == DEFAULT_FEATURE_SCHEMA
    assert loaded_manifest["feature_dim"] == FEATURE_DIM
    assert loaded_manifest["action_size"] == ACTION_SIZE
    assert len(loaded) == len(records)

    for original, copy in zip(records, loaded):
        assert copy.exact_state == original.exact_state
        assert (
            exact_public_state(reconstruct_game(copy.exact_state)) == copy.exact_state
        )
        assert copy.target_kind is original.target_kind
        assert copy.source == original.source
        assert copy.value == pytest.approx(original.value, abs=1e-7)
        assert copy.episode_id == original.episode_id
        assert copy.half_round_index == original.half_round_index
        assert copy.truncated is original.truncated
        assert copy.parent_checkpoint_digest == original.parent_checkpoint_digest
        assert copy.search_config_digest == original.search_config_digest
        assert copy.rng_seeds == original.rng_seeds
        np.testing.assert_array_equal(copy.features, original.features)
        np.testing.assert_array_equal(copy.dropper_dist, original.dropper_dist)
        np.testing.assert_array_equal(copy.checker_dist, original.checker_dist)
        np.testing.assert_array_equal(
            copy.dropper_legal_mask, original.dropper_legal_mask
        )
        np.testing.assert_array_equal(
            copy.checker_legal_mask, original.checker_legal_mask
        )

    with np.load(path, allow_pickle=False) as payload:
        assert payload.files
        assert all(not payload[name].dtype.hasobject for name in payload.files)


def test_payload_corruption_fails_before_arrays_are_loaded(tmp_path: Path):
    path = tmp_path / "corrupt.npz"
    save_replay_shard([_record(1)], path)
    payload = bytearray(path.read_bytes())
    payload[len(payload) // 2] ^= 0x01
    path.write_bytes(payload)

    with pytest.raises(ReplayValidationError, match="SHA-256 mismatch"):
        load_replay_shard(path)


def test_manifest_with_old_61_action_size_is_rejected(tmp_path: Path):
    path = tmp_path / "old-actions.npz"
    save_replay_shard([_record(2)], path)
    manifest_path = manifest_path_for(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["action_size"] = 61
    manifest_path.write_bytes(
        (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
    )

    with pytest.raises(ReplayValidationError, match="action size mismatch: 61"):
        load_replay_shard(path)


def test_length_61_record_is_rejected_before_publication(tmp_path: Path):
    valid = _record(3)
    stale = replace(
        valid,
        dropper_dist=valid.dropper_dist[:61],
        dropper_legal_mask=valid.dropper_legal_mask[:61],
    )
    path = tmp_path / "stale.npz"

    with pytest.raises(ReplayValidationError, match="action size 62"):
        save_replay_shard([stale], path)
    assert not path.exists()
    assert not manifest_path_for(path).exists()


def test_exact_state_serialization_hash_and_reconstruction_are_canonical():
    state = _record(7).exact_state
    payload = canonical_exact_state_json(state)

    assert exact_state_from_json(payload) == state
    assert exact_state_hash(exact_state_from_json(payload)) == exact_state_hash(state)
    assert exact_public_state(reconstruct_game(state)) == state
    assert payload == json.dumps(
        json.loads(payload), sort_keys=True, separators=(",", ":")
    )

    noncanonical = json.dumps(json.loads(payload), indent=2)
    with pytest.raises(ReplayValidationError, match="not canonical"):
        exact_state_from_json(noncanonical)


def test_grouped_split_uses_transitive_state_or_episode_components():
    first = _record(10, episode_id="episode-a")
    second = _record(11, episode_id="episode-a")
    # Connect episode-b transitively to episode-a through second's exact state.
    bridge = replace(second, episode_id="episode-b", half_round_index=1)
    fourth = _record(12, episode_id="episode-c")
    fifth = _record(13, episode_id="episode-d")
    records = [first, second, bridge, fourth, fifth]

    train, validation = grouped_split_indices(
        records, validation_fraction=0.4, seed=123
    )
    train_set = set(train.tolist())
    validation_set = set(validation.tolist())

    assert train_set.isdisjoint(validation_set)
    assert train_set | validation_set == set(range(len(records)))
    assert {0, 1, 2}.issubset(train_set) or {0, 1, 2}.issubset(validation_set)

    train_states = {exact_state_hash(records[index].exact_state) for index in train}
    validation_states = {
        exact_state_hash(records[index].exact_state) for index in validation
    }
    assert train_states.isdisjoint(validation_states)
    train_episodes = {
        records[index].episode_id for index in train if records[index].episode_id
    }
    validation_episodes = {
        records[index].episode_id for index in validation if records[index].episode_id
    }
    assert train_episodes.isdisjoint(validation_episodes)


def test_external_ruler_loads_for_evaluation_but_not_training(tmp_path: Path):
    path = tmp_path / "external-ruler.npz"
    save_replay_shard([_record(20)], path, shard_role=ShardRole.EXTERNAL_RULER)

    assert len(load_replay_shard(path)) == 1
    with pytest.raises(ReplayValidationError, match="forbidden in training"):
        load_replay_shard(path, for_training=True)


def test_collision_audit_reports_only_distinct_states_with_divergent_values():
    first = replace(_record(30), value=1.0)
    second = replace(_record(31), features=first.features, value=-1.0)
    same_value = replace(_record(32), features=first.features, value=1.0)

    divergent = audit_feature_collisions([first, second, same_value])
    assert len(divergent) == 1
    assert divergent[0].record_indices == (0, 1, 2)
    assert len(divergent[0].exact_state_hashes) == 3
    assert divergent[0].value_span == pytest.approx(2.0)

    equal_a = replace(first, value=0.25)
    equal_b = replace(second, value=0.25)
    assert audit_feature_collisions([equal_a, equal_b]) == []
    assert len(audit_feature_collisions([equal_a, equal_b], divergent_only=False)) == 1


def test_wrong_parent_digest_fails_before_training(tmp_path: Path):
    path = tmp_path / "parent.npz"
    save_replay_shard([_record(40)], path)

    with pytest.raises(ReplayValidationError, match="parent digest"):
        load_replay_shard(
            path,
            for_training=True,
            expected_parent_checkpoint_digest=_digest("different-parent"),
        )


def test_v2_has_no_divergent_feature_collisions_on_pinned_tablebase():
    records = []
    for index, scenario in enumerate(pinned_scenarios()):
        game = scenario.game
        dropper_dist, dropper_mask = _policy_and_mask(game, "dropper")
        checker_dist, checker_mask = _policy_and_mask(game, "checker")
        records.append(
            TrainingRecordV2(
                features=extract_features(game),
                exact_state=exact_public_state(game),
                value=float(scenario.expected_value),
                target_kind=TargetKind.TABLEBASE_VALUE,
                source="pinned_tablebase",
                dropper_dist=dropper_dist,
                checker_dist=checker_dist,
                dropper_legal_mask=dropper_mask,
                checker_legal_mask=checker_mask,
                episode_id=f"pin-{index}",
            )
        )

    assert records
    assert audit_feature_collisions(records) == []
