import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.dth_env import DTHEnv
from environment.route_stages import current_route_stage_flags
from training.behavior_clone import collect_trace_samples, scenario_options_for_trace
from training.bridge_traces import BRIDGE_TRACE_SETS, load_trace_file, save_trace_file
from training.train_ppo import build_parser, make_opponent


def test_train_parser_exposes_bridge_supervision_args():
    args = build_parser().parse_args([])

    assert args.bridge_trace_set is None
    assert args.bridge_trace_file is None
    assert args.teacher_demo_file == []
    assert args.bc_epochs == 0
    assert args.bc_batch_size == 32
    assert args.bc_learning_rate == 1e-4


def test_collect_trace_samples_returns_legal_demonstrations():
    samples, counts = collect_trace_samples(
        BRIDGE_TRACE_SETS["seed_exact_bridge"],
        opponent_factory=make_opponent,
    )

    assert len(samples) == 18
    assert sum(counts) == 18
    assert all(sample.action_mask[sample.action_index] for sample in samples)


def test_collect_trace_samples_supports_opening_seed_trace():
    samples, counts = collect_trace_samples(
        BRIDGE_TRACE_SETS["seed_opening_round7"],
        opponent_factory=make_opponent,
    )

    assert len(samples) == 12
    assert sum(counts) == 12
    assert all(sample.action_mask[sample.action_index] for sample in samples)


def test_full_bridge_seed_set_covers_opening_and_late_segments():
    samples, counts = collect_trace_samples(
        BRIDGE_TRACE_SETS["seed_full_exact_bridge"],
        opponent_factory=make_opponent,
    )

    assert len(samples) == 30
    assert sum(counts) == 30


def test_opening_trace_uses_no_scenario_override():
    spec = BRIDGE_TRACE_SETS["seed_opening_round7"][0]

    assert scenario_options_for_trace(spec) is None


def test_opening_seed_trace_reaches_round7_pressure():
    spec = BRIDGE_TRACE_SETS["seed_opening_round7"][0]
    assert spec.opponent_name is not None
    opponent = make_opponent(spec.opponent_name)
    assert opponent is not None
    env = DTHEnv(opponent=opponent, agent_role=spec.agent_role, seed=spec.seed)

    env.reset(options=scenario_options_for_trace(spec))
    for second in spec.actions:
        env.step(second - 1)

    assert env.game is not None
    assert env.agent is not None
    assert env.agent.alive == True
    assert env.game.game_over == False
    assert current_route_stage_flags(env.game)["round7_pressure"] == True


def test_trace_file_round_trip(tmp_path):
    path = tmp_path / "trace_file.json"
    specs = BRIDGE_TRACE_SETS["seed_full_exact_bridge"]

    save_trace_file(str(path), specs)
    loaded = load_trace_file(str(path))

    assert loaded == specs
