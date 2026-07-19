import numpy as np

from toy.artifacts import write_npz_artifact
from toy.network import feature_dim
from toy.rules import Bucket12Fixed50Rules
from toy.targets import TARGET_SCHEMA
from toy.train import ToyTrainConfig, load_toy_checkpoint, train_exact_targets


def test_tiny_exact_target_training_reduces_losses(tmp_path):
    rules = Bucket12Fixed50Rules(max_half_rounds=2)
    physical = [(hal, baku, phase) for hal, baku, phase in ((0, 0, 0), (10, 5, 1), (20, 15, 0), (30, 20, 1), (40, 30, 0), (50, 45, 1))]
    states = np.asarray([state for state in physical for _horizon in (1, 2)], dtype=np.int16)
    horizons = np.asarray([horizon for _state in physical for horizon in (1, 2)], dtype=np.int16)
    values = np.asarray([(hal - baku) / 120.0 for hal, baku, _phase in states], dtype=np.float32)
    policies = np.zeros((len(states), rules.action_size), dtype=np.float32)
    for index, (hal, _baku, _phase) in enumerate(states):
        policies[index, (int(hal) % rules.action_size)] = 1.0
    arrays = {
        "states": states,
        "horizon": horizons,
        "state_id": np.asarray([f"state-{index}" for index in range(len(states))], dtype="U64"),
        "physical_state_id": np.asarray([str(tuple(state)) for state in states], dtype="U128"),
        "value": values,
        "drop_policy": policies,
        "check_policy": policies.copy(),
    }
    target_npz = tmp_path / "targets.npz"
    target_manifest = tmp_path / "targets.json"
    write_npz_artifact(
        arrays,
        target_npz,
        target_manifest,
        metadata={"ruleset_id": rules.ruleset_id},
        schema_version=TARGET_SCHEMA,
    )
    result = train_exact_targets(
        target_npz,
        target_manifest,
        rules,
        tmp_path / "checkpoint",
        config=ToyTrainConfig(epochs=6, batch_size=4, hidden_dim=16, val_fraction=0.33),
    )
    assert result.best_epoch >= 0
    assert result.history[-1]["train_value_mse"] < result.history[0]["train_value_mse"]
    assert result.history[-1]["train_policy_loss"] < result.history[0]["train_policy_loss"]
    model = load_toy_checkpoint(result.checkpoint_path, rules)
    assert model.feature_dim == feature_dim(rules)
