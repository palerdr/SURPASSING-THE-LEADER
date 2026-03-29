import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.opponents.baku_teachers import BakuLeapExecutorTeacher
from environment.opponents.factory import create_scripted_opponent
from training.teacher_demos import (
    load_teacher_demo_file,
    reached_stage_name,
    rollout_teacher_episode,
    save_teacher_demo_file,
)


def test_reached_stage_name_prioritizes_leap_turn():
    assert reached_stage_name({"leap_window": True, "leap_turn": True}) == "leap_turn"


def test_rollout_teacher_episode_collects_legal_samples():
    teacher = BakuLeapExecutorTeacher()
    opponent = create_scripted_opponent("bridge_pressure")

    samples, reached_stage, _won = rollout_teacher_episode(
        teacher=teacher,
        teacher_name="baku_leap",
        opponent=opponent,
        opponent_name="bridge_pressure",
        opponent_model_path=None,
        agent_role="baku",
        seed=42,
        game_index=0,
        scenario_name="round9_leap_deduced",
        max_steps=1,
    )

    assert len(samples) == 1
    assert samples[0].action_mask[samples[0].action_index] is True
    assert reached_stage == "leap_turn"


def test_teacher_demo_file_round_trip(tmp_path):
    teacher = BakuLeapExecutorTeacher()
    opponent = create_scripted_opponent("bridge_pressure")
    path = tmp_path / "teacher_demos.jsonl"

    samples, _reached_stage, _won = rollout_teacher_episode(
        teacher=teacher,
        teacher_name="baku_leap",
        opponent=opponent,
        opponent_name="bridge_pressure",
        opponent_model_path=None,
        agent_role="baku",
        seed=42,
        game_index=0,
        scenario_name="round9_leap_deduced",
        max_steps=1,
    )

    save_teacher_demo_file(str(path), samples)
    loaded = load_teacher_demo_file(str(path))

    assert loaded == samples
