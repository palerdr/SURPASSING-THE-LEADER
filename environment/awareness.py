"""Awareness gating for leap-second information and actions.

The environment owns what each side is allowed to know, while the engine
stays fully objective. This keeps Baku/Hal knowledge modes configurable.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from src.Game import Game, HalfRoundRecord
from src.Constants import TURN_DURATION_LEAP, TURN_DURATION_NORMAL


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


def checker_can_use_extra_second(awareness: LeapAwareness, *, actor: str = "baku") -> bool:
    if actor.lower() != "hal":
        return False
    return awareness == LeapAwareness.DEDUCED


def build_action_mask(
    *,
    role: str,
    is_leap_turn: bool,
    awareness: LeapAwareness,
    actor: str = "baku",
) -> np.ndarray:
    from environment.legal_actions import legal_max_second

    mask = np.zeros(61, dtype=bool)
    mask[:TURN_DURATION_NORMAL] = True

    if not is_leap_turn:
        return mask

    max_sec = legal_max_second(actor, role, TURN_DURATION_LEAP)
    if max_sec >= TURN_DURATION_LEAP:
        mask[TURN_DURATION_NORMAL] = True  # action index 60 -> second 61

    return mask


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
