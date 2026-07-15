"""Awareness gating for leap-second information and actions.

The environment owns what each side is allowed to know, while the engine
stays fully objective. This keeps Baku/Hal knowledge modes configurable.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from stl.engine.game import Game, HalfRoundRecord
from stl.engine.game import TURN_DURATION_LEAP


class LeapAwareness(str, Enum):
    UNAWARE = "unaware"
    DEDUCED = "deduced"
    MEMORY_IMPAIRED = "memory_impaired"


@dataclass(frozen=True)
class AwarenessConfig:
    hal_mode: str = "first_death_evidence"


def initial_awareness_for_role(role: str) -> LeapAwareness:
    if role.lower() == "baku":
        return LeapAwareness.DEDUCED
    return LeapAwareness.UNAWARE


def exposes_leap_features(awareness: LeapAwareness) -> bool:
    return awareness == LeapAwareness.DEDUCED


def build_action_mask(
    *,
    role: str,
    is_leap_turn: bool,
    awareness: LeapAwareness,
    actor: str = "baku",
) -> np.ndarray:
    from stl.engine.actions import legal_mask
    from stl.engine.game import TURN_DURATION_NORMAL

    del awareness  # Leap knowledge affects features; structural legality is actor-aware.
    turn_duration = TURN_DURATION_LEAP if is_leap_turn else TURN_DURATION_NORMAL
    return legal_mask(actor, role, turn_duration)


def is_structural_evidence_event(game: Game, record: HalfRoundRecord) -> bool | None:
    return record.death_duration > 0


def update_awareness(
    awareness: LeapAwareness,
    *,
    controlled_role_name: str,
    config: AwarenessConfig,
    game: Game,
    record: HalfRoundRecord,
) -> LeapAwareness:
    if controlled_role_name.lower() == "baku":
        return LeapAwareness.DEDUCED

    if awareness == LeapAwareness.MEMORY_IMPAIRED:
        return awareness

    if config.hal_mode == "oracle":
        return LeapAwareness.DEDUCED

    if config.hal_mode == "blind":
        return LeapAwareness.UNAWARE

    if config.hal_mode != "first_death_evidence":
        raise ValueError(f"Unknown hal awareness mode: {config.hal_mode}")

    inferred = is_structural_evidence_event(game, record)
    if inferred is None:
        inferred = record.death_duration > 0

    if inferred:
        return LeapAwareness.DEDUCED

    return awareness
