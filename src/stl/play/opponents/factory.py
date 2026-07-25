"""Central opponent registry for bots, teachers, models, and leagues.

Training, evaluation, and demo harvest all go through this module so new
opponents only need to be registered once.
"""

from __future__ import annotations

import inspect

from .baku_teachers import (
    BakuActiveLsrPreserverTeacher,
    BakuAntiEcholocationTeacher,
    BakuLeapExecutorTeacher,
    BakuLSREngineeringTeacher,
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
from .pattern_reader import PatternReaderBaku
from .random_bot import RandomBot
from .safe_bot import BridgePressureBot, LeapAwareSafeBot, SafeBot

from stl.play.canonical_hal import CanonicalHal


def _hal_solver():
    """Lazy import: the solver agent pulls in the cfr/search stack + torch."""
    from stl.play.agent import HalSolverAgent

    return HalSolverAgent()


def _baku_solver():
    from stl.play.agent import BakuSolverAgent

    return BakuSolverAgent()


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
    "baku_lsr_engineering": BakuLSREngineeringTeacher,
    "baku_counter": BakuAntiEcholocationTeacher,
    "baku_resilience": BakuResilienceFallbackTeacher,
    "pattern_reader": PatternReaderBaku,
    "hal_teacher": HalTeacher,
    "hal_infer": HalLeapInferenceTeacher,
    "hal_death_trade": HalDeathTradeTeacher,
    "hal_pressure": HalPressureTeacher,
    "hal_deviation": HalDeviationTeacher,
    "hal_echo": HalEcholocationTeacher,
    "hal_memory": HalMemoryLossTeacher,
    "hal_resilience": HalResilienceTeacher,
    "hal_canonical": CanonicalHal,
    "hal_solver": _hal_solver,
    "baku_solver": _baku_solver,
}


SCRIPTED_LEAGUES = {
    "opening_league": (
        ("bridge_pressure", 1.0),
        ("hal_death_trade", 1.0),
        ("hal_pressure", 1.0),
    ),
}


def _construct_with_optional_seed(constructor, seed: int | None):
    try:
        signature = inspect.signature(constructor)
    except (TypeError, ValueError):
        return constructor()
    if "seed" in signature.parameters:
        return constructor(seed=seed)
    return constructor()


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
    return _construct_with_optional_seed(constructor, seed)
