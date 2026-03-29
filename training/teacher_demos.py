from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from environment.dth_env import DTHEnv
from environment.route_stages import current_route_stage_flags
from training.curriculum import get_scenario


@dataclass(frozen=True)
class TeacherDemoSample:
    observation: list[float]
    action_mask: list[bool]
    action_index: int
    agent_role: str
    teacher_name: str
    opponent_name: str | None
    opponent_model_path: str | None
    scenario_name: str
    seed: int
    game_index: int
    turn_index: int
    reached_stage: str | None
    won: bool


def scenario_options_for_name(name: str) -> dict | None:
    if name == "opening":
        return None
    return {"scenario": get_scenario(name)}


def reached_stage_name(flags: dict[str, bool]) -> str | None:
    for stage_name in ("leap_turn", "leap_window", "round9_pre_leap", "round8_bridge", "round7_pressure"):
        if flags.get(stage_name, False):
            return stage_name
    return None


def merge_reached_stage(current_stage: str | None, flags: dict[str, bool]) -> str | None:
    order = {
        None: -1,
        "round7_pressure": 0,
        "round8_bridge": 1,
        "round9_pre_leap": 2,
        "leap_window": 3,
        "leap_turn": 4,
    }
    candidate = reached_stage_name(flags)
    if order[candidate] > order[current_stage]:
        return candidate
    return current_stage


def rollout_teacher_episode(
    *,
    teacher,
    teacher_name: str,
    opponent,
    opponent_name: str | None,
    opponent_model_path: str | None,
    agent_role: str,
    seed: int,
    game_index: int,
    scenario_name: str,
    max_steps: int,
):
    env = DTHEnv(opponent=opponent, agent_role=agent_role, seed=seed)
    obs, _ = env.reset(options=scenario_options_for_name(scenario_name))
    assert env.game is not None
    assert env.agent is not None

    samples: list[TeacherDemoSample] = []
    reached_stage = reached_stage_name(current_route_stage_flags(env.game))
    won = False

    for turn_index in range(max_steps):
        mask = env.action_masks()
        turn_duration = env.game.get_turn_duration()
        dropper, checker = env.game.get_roles_for_half(env.game.current_half)
        role = "dropper" if env.agent is dropper else "checker"
        second = teacher.choose_action(env.game, role, turn_duration)
        action_index = second - 1
        if action_index < 0 or action_index >= len(mask) or not mask[action_index]:
            raise ValueError(
                f"Teacher {teacher_name} produced illegal action second={second} role={role} "
                f"scenario={scenario_name} seed={seed}"
            )

        samples.append(
            TeacherDemoSample(
                observation=np.asarray(obs, dtype=np.float32).tolist(),
                action_mask=np.asarray(mask, dtype=bool).tolist(),
                action_index=action_index,
                agent_role=agent_role,
                teacher_name=teacher_name,
                opponent_name=opponent_name,
                opponent_model_path=opponent_model_path,
                scenario_name=scenario_name,
                seed=seed,
                game_index=game_index,
                turn_index=turn_index,
                reached_stage=reached_stage,
                won=won,
            )
        )
        obs, reward, terminated, truncated, _info = env.step(action_index)
        flags = current_route_stage_flags(env.game)
        reached_stage = merge_reached_stage(reached_stage, flags)
        won = bool(terminated and env.game.winner is env.agent and reward > 0)
        if samples:
            samples[-1] = TeacherDemoSample(
                **{
                    **asdict(samples[-1]),
                    "reached_stage": reached_stage,
                    "won": won,
                }
            )
        if terminated or truncated:
            break

    return samples, reached_stage, won


def save_teacher_demo_file(path: str, samples: list[TeacherDemoSample]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(asdict(sample), sort_keys=True) + "\n")


def load_teacher_demo_file(path: str) -> list[TeacherDemoSample]:
    loaded: list[TeacherDemoSample] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            loaded.append(TeacherDemoSample(**row))
    return loaded
