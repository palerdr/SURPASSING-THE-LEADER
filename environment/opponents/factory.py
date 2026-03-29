"""Central opponent registry for bots, teachers, models, and leagues.

Training, evaluation, and demo harvest all go through this module so new
opponents only need to be registered once.
"""

from __future__ import annotations

from pathlib import Path

from .baku_teachers import (
    BakuActiveLsrPreserverTeacher,
    BakuAntiEcholocationTeacher,
    BakuLeapExecutorTeacher,
    BakuResilienceFallbackTeacher,
    BakuRouteBuilderTeacher,
    BakuTeacher,
)
from .hal_teachers import (
    HalDeathTradeTeacher,
    HalDeviationTeacher,
    HalEcholocationTeacher,
    HalLeapInferenceTeacher,
    HalMemoryLossTeacher,
    HalPressureTeacher,
    HalResilienceTeacher,
    HalTeacher,
)
from .league import LeagueEntry, WeightedOpponentLeague
from .model_opponent import ModelOpponent
from .random_bot import RandomBot
from .safe_bot import BridgePressureBot, LeapAwareSafeBot, SafeBot


SCRIPTED_OPPONENTS = {
    "none": None,
    "random": RandomBot,
    "safe": SafeBot,
    "leap_safe": LeapAwareSafeBot,
    "bridge_pressure": BridgePressureBot,
    "baku_teacher": BakuTeacher,
    "baku_route": BakuRouteBuilderTeacher,
    "baku_preserve": BakuActiveLsrPreserverTeacher,
    "baku_leap": BakuLeapExecutorTeacher,
    "baku_counter": BakuAntiEcholocationTeacher,
    "baku_resilience": BakuResilienceFallbackTeacher,
    "hal_teacher": HalTeacher,
    "hal_infer": HalLeapInferenceTeacher,
    "hal_death_trade": HalDeathTradeTeacher,
    "hal_pressure": HalPressureTeacher,
    "hal_deviation": HalDeviationTeacher,
    "hal_echo": HalEcholocationTeacher,
    "hal_memory": HalMemoryLossTeacher,
    "hal_resilience": HalResilienceTeacher,
}


SCRIPTED_LEAGUES = {
    "opening_league": (
        ("bridge_pressure", 1.0),
        ("hal_death_trade", 1.0),
        ("hal_pressure", 1.0),
    ),
}


def scripted_opponent_names() -> list[str]:
    return sorted([*SCRIPTED_OPPONENTS.keys(), *SCRIPTED_LEAGUES.keys()])


def create_scripted_opponent(name: str, seed: int | None = None):
    if name in SCRIPTED_LEAGUES:
        entries = [
            LeagueEntry(label=label, weight=weight, opponent=create_scripted_opponent(label, seed=seed))
            for label, weight in SCRIPTED_LEAGUES[name]
        ]
        return WeightedOpponentLeague(entries, seed=seed)
    if name not in SCRIPTED_OPPONENTS:
        raise ValueError(f"Unknown opponent: {name}")
    constructor = SCRIPTED_OPPONENTS[name]
    if constructor is None:
        return None
    return constructor()


def opponent_role_for_agent(agent_role: str) -> str:
    return "baku" if agent_role == "hal" else "hal"


def create_model_opponent(model_path: str, *, agent_role: str):
    return ModelOpponent(model_path, role=opponent_role_for_agent(agent_role))


def parse_weighted_model_spec(spec: str) -> tuple[str, float]:
    model_path = spec
    weight = 1.0
    if ":" in spec:
        maybe_path, maybe_weight = spec.rsplit(":", 1)
        try:
            weight = float(maybe_weight)
            model_path = maybe_path
        except ValueError:
            model_path = spec

    if weight <= 0:
        raise ValueError(f"Opponent model weight must be > 0, got {weight} for {spec}")
    return model_path, weight


def build_opponent_league(
    *,
    agent_role: str,
    opponent_name: str,
    opponent_weight: float,
    opponent_model_specs: list[str],
    seed: int | None,
):
    scripted_opponent = create_scripted_opponent(opponent_name, seed=seed)
    entries: list[LeagueEntry] = []

    if scripted_opponent is not None and opponent_weight > 0:
        entries.append(LeagueEntry(label=opponent_name, weight=opponent_weight, opponent=scripted_opponent))

    for spec in opponent_model_specs:
        model_path, weight = parse_weighted_model_spec(spec)
        entries.append(
            LeagueEntry(
                label=Path(model_path).name,
                weight=weight,
                opponent=create_model_opponent(model_path, agent_role=agent_role),
            )
        )

    if not entries:
        raise ValueError("No opponent entries configured. Provide --opponent or --opponent-model.")
    if len(entries) == 1:
        return entries[0].opponent
    return WeightedOpponentLeague(entries, seed=seed)


def opponent_config_label(opponent_name: str, opponent_model_specs: list[str], opponent_weight: float) -> str:
    if opponent_model_specs:
        if opponent_name != "none" and opponent_weight > 0:
            return "league"
        if len(opponent_model_specs) > 1:
            return "league"
        model_path, _ = parse_weighted_model_spec(opponent_model_specs[0])
        return Path(model_path).stem
    return opponent_name
