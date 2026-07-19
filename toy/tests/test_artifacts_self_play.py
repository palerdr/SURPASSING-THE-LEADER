import numpy as np

from toy.artifacts import load_npz_artifact, write_npz_artifact
from toy.network import ToyPolicyValueNet, feature_dim
from toy.rules import Bucket12Fixed50Rules
from toy.self_play import ToySelfPlayConfig, generate_self_play, write_self_play


def test_npz_artifact_round_trip_and_digest_validation(tmp_path):
    arrays = {"x": np.arange(6, dtype=np.int16), "y": np.eye(2, dtype=np.float32)}
    npz_path = tmp_path / "artifact.npz"
    manifest_path = tmp_path / "artifact.json"
    write_npz_artifact(
        arrays,
        npz_path,
        manifest_path,
        metadata={"purpose": "test"},
        schema_version="toy.test.v1",
    )
    loaded, manifest = load_npz_artifact(npz_path, manifest_path, expected_schema_version="toy.test.v1")
    assert np.array_equal(loaded["x"], arrays["x"])
    assert manifest["arrays"]["y"]["shape"] == [2, 2]


def test_self_play_labels_and_replay_round_trip(tmp_path):
    rules = Bucket12Fixed50Rules(max_half_rounds=2)
    model = ToyPolicyValueNet(feature_dim(rules), rules.action_size)
    config = ToySelfPlayConfig(games=4, seed=9, mcts_iterations=4)
    first = generate_self_play(rules, model, config=config)
    second = generate_self_play(rules, model, config=config)
    assert first.metadata["trajectory_sha256"] == second.metadata["trajectory_sha256"]
    arrays = first.arrays
    assert arrays["states"].shape[1] == 3
    assert np.array_equal(arrays["hal_load"], arrays["states"][:, 0])
    assert np.array_equal(arrays["baku_load"], arrays["states"][:, 1])
    assert np.array_equal(arrays["role_phase"], arrays["states"][:, 2])
    assert np.allclose(arrays["drop_policy"].sum(axis=1), 1.0)
    assert np.allclose(arrays["check_policy"].sum(axis=1), 1.0)
    assert np.all((arrays["drop_action"] >= 1) & (arrays["drop_action"] <= 12))
    assert np.all((arrays["check_action"] >= 1) & (arrays["check_action"] <= 12))
    assert np.all(np.isin(arrays["outcome"], (-1.0, 0.0, 1.0)))
    assert np.all(arrays["truncated"] | np.isin(arrays["outcome"], (-1.0, 1.0)))
    npz_path, manifest_path, manifest = write_self_play(first, tmp_path / "replay")
    loaded, _ = __import__("toy.artifacts", fromlist=["load_npz_artifact"]).load_npz_artifact(
        npz_path, manifest_path, expected_schema_version="toy.self_play_replay.v2"
    )
    assert loaded["states"].shape == arrays["states"].shape
    assert manifest["metadata"]["trajectory_sha256"] == first.metadata["trajectory_sha256"]
