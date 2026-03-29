from __future__ import annotations

from copy import deepcopy

import numpy as np

from environment.route_stages import (
    ROUND7_PRESSURE_STAGE,
    ROUND8_BRIDGE_STAGE,
    ROUND9_PRE_LEAP_STAGE,
)

ROUND7_PRESSURE = {
    "name": "round7_pressure",
    "game_clock": ROUND7_PRESSURE_STAGE.game_clock,
    "round_num": ROUND7_PRESSURE_STAGE.round_num,
    "current_half": ROUND7_PRESSURE_STAGE.current_half,
    "first_dropper": "hal",
    "hal": {"cylinder": 93.0, "ttd": 84.0, "deaths": 1, "alive": True},
    "baku": {"cylinder": 0.0, "ttd": 153.0, "deaths": 2, "alive": True},
    "referee_cprs": 3,
    "awareness": "deduced",
}

ROUND8_BRIDGE = {
    "name": "round8_bridge",
    "game_clock": ROUND8_BRIDGE_STAGE.game_clock,
    "round_num": ROUND8_BRIDGE_STAGE.round_num,
    "current_half": ROUND8_BRIDGE_STAGE.current_half,
    "first_dropper": "hal",
    "hal": {"cylinder": 0.0, "ttd": 178.0, "deaths": 2, "alive": True},
    "baku": {"cylinder": 58.0, "ttd": 153.0, "deaths": 2, "alive": True},
    "referee_cprs": 4,
    "awareness": "deduced",
}

ROUND9_PRE_LEAP = {
    "name": "round9_pre_leap",
    "game_clock": ROUND9_PRE_LEAP_STAGE.game_clock,
    "round_num": ROUND9_PRE_LEAP_STAGE.round_num,
    "current_half": ROUND9_PRE_LEAP_STAGE.current_half,
    "first_dropper": "hal",
    "hal": {"cylinder": 0.0, "ttd": 238.0, "deaths": 2, "alive": True},
    "baku": {"cylinder": 115.0, "ttd": 153.0, "deaths": 2, "alive": True},
    "referee_cprs": 4,
    "awareness": "deduced",
}

ROUND9_LEAP_DEDUCED = {
    "name": "round9_leap_deduced",
    "game_clock": 3540.0,
    "round_num": 8,
    "current_half": 2,
    "first_dropper": "hal",
    "hal": {"cylinder": 0.0, "ttd": 238.0, "deaths": 2, "alive": True},
    "baku": {"cylinder": 175.0, "ttd": 153.0, "deaths": 2, "alive": True},
    "referee_cprs": 4,
    "awareness": "deduced",
}

ROUND9_LEAP_IMPAIRED = {
    "name": "round9_leap_impaired",
    "game_clock": 3540.0,
    "round_num": 8,
    "current_half": 2,
    "first_dropper": "hal",
    "hal": {"cylinder": 0.0, "ttd": 238.0, "deaths": 2, "alive": True},
    "baku": {"cylinder": 175.0, "ttd": 153.0, "deaths": 2, "alive": True},
    "referee_cprs": 4,
    "awareness": "memory_impaired",
}


SCENARIOS = {
    "round7_pressure": ROUND7_PRESSURE,
    "round8_bridge": ROUND8_BRIDGE,
    "round9_pre_leap": ROUND9_PRE_LEAP,
    "round9_leap_deduced": ROUND9_LEAP_DEDUCED,
    "round9_leap_impaired": ROUND9_LEAP_IMPAIRED,
}


CURRICULA = {
    "none": [(1.0, None)],
    "late": [
        (0.20, None),
        (0.30, "round7_pressure"),
        (0.50, "round9_pre_leap"),
    ],
    "critical": [
        (0.10, None),
        (0.15, "round7_pressure"),
        (0.30, "round9_pre_leap"),
        (0.25, "round9_leap_deduced"),
        (0.20, "round9_leap_impaired"),
    ],
    "bridge": [
        (0.20, None),
        (0.20, "round7_pressure"),
        (0.30, "round8_bridge"),
        (0.20, "round9_pre_leap"),
        (0.10, "round9_leap_deduced"),
    ],
    "opening_to_round7": [
        (1.00, None),
    ],
    "round7_to_round8": [
        (1.00, "round7_pressure"),
    ],
    "round8_to_round9": [
        (1.00, "round8_bridge"),
    ],
    "mixed": [
        (0.20, None),
        (0.20, "round7_pressure"),
        (0.25, "round9_pre_leap"),
        (0.20, "round9_leap_deduced"),
        (0.15, "round9_leap_impaired"),
    ],
}


def get_scenario(name: str) -> dict:
    return deepcopy(SCENARIOS[name])


def make_curriculum_sampler(name: str, seed: int | None = None):
    if name not in CURRICULA:
        raise ValueError(f"Unknown curriculum: {name}")

    entries = CURRICULA[name]
    weights = np.array([weight for weight, _ in entries], dtype=np.float64)
    weights = weights / weights.sum()
    names = [scenario_name for _, scenario_name in entries]

    fallback_rng = np.random.default_rng(seed)

    def sample(rng: np.random.Generator | None = None) -> dict | None:
        active_rng = rng if rng is not None else fallback_rng
        index = int(active_rng.choice(len(names), p=weights))
        scenario_name = names[index]
        if scenario_name is None:
            return None
        return get_scenario(scenario_name)

    return sample
