import numpy as np

from dth.mcts import Evaluation
from dth.self_play import SelfPlayConfig, generate_self_play, validate_replay


class _UniformEvaluator:
    def __call__(self, state, horizon) -> Evaluation:
        del state, horizon
        policy = np.full(60, 1.0 / 60)
        return Evaluation(0.0, policy, policy)


def test_self_play_is_reproducible_and_well_formed(tmp_path) -> None:
    config = SelfPlayConfig(
        checkpoint=str(tmp_path / "missing.pt"),
        output=str(tmp_path / "replay"),
        games=2,
        max_half_rounds=2,
        simulations=2,
        root_warmup_cells=0,
        max_depth=1,
    )
    first = generate_self_play(config, evaluator=_UniformEvaluator())
    second = generate_self_play(config, evaluator=_UniformEvaluator())

    assert first.metadata["trajectory_sha256"] == second.metadata["trajectory_sha256"]
    arrays = first.arrays
    assert arrays["states"].shape[1] == 4
    assert arrays["features"].shape[1] == 5
    assert np.allclose(arrays["drop_policy"].sum(axis=1), 1.0)
    assert np.allclose(arrays["check_policy"].sum(axis=1), 1.0)
    assert np.all((arrays["drop_action"] >= 1) & (arrays["drop_action"] <= 60))
    assert np.all((arrays["check_action"] >= 1) & (arrays["check_action"] <= 60))
    assert np.all(np.isin(arrays["outcome"], (-1.0, 0.0, 1.0)))
    assert np.all(arrays["truncated"] | np.isin(arrays["outcome"], (-1.0, 1.0)))
    assert validate_replay(arrays)["unreconstructable_states"] == 0


def test_self_play_supports_strategic_boundary_starts(tmp_path) -> None:
    config = SelfPlayConfig(
        checkpoint=str(tmp_path / "missing.pt"),
        output=str(tmp_path / "replay"),
        games=1,
        max_half_rounds=1,
        simulations=0,
        root_warmup_cells=0,
        max_depth=1,
        starts=[{"state": [240, 0, 0, 0], "horizon": 1}],
    )

    result = generate_self_play(config, evaluator=_UniformEvaluator())

    np.testing.assert_array_equal(result.arrays["states"][0], [240, 0, 0, 0])
    np.testing.assert_array_equal(
        result.arrays["initial_state"][0], [240, 0, 0, 0]
    )
    assert validate_replay(result.arrays)["episodes"] == 1


def test_h5_boundary_replay_is_deterministic_for_both_orientations(tmp_path) -> None:
    config = SelfPlayConfig(
        checkpoint=str(tmp_path / "missing.pt"),
        output=str(tmp_path / "replay"),
        games=2,
        max_half_rounds=5,
        simulations=0,
        root_warmup_cells=0,
        max_depth=1,
        starts=[
            {"state": [239, 0, 0, 240], "horizon": 5},
            {"state": [0, 240, 239, 0], "horizon": 5},
        ],
    )

    first = generate_self_play(config, evaluator=_UniformEvaluator())
    second = generate_self_play(config, evaluator=_UniformEvaluator())

    assert first.metadata["trajectory_sha256"] == second.metadata["trajectory_sha256"]
    assert validate_replay(first.arrays)["episodes"] == 2
    assert set(first.arrays["horizon"].tolist()).issubset({1, 2, 3, 4, 5})
