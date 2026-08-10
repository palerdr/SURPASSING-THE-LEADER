"""Deterministic public-history opponent league for pure-DTH Arena training.

The providers in this module deliberately know less than the canonical game
engine.  They consume :class:`~arena.contracts.CanonicalDecision` numeric
fields and revealed :class:`~arena.contracts.PublicHalfRound` records, never
``CanonicalDecision.native_state``.  Every family returns a distribution over
literal seconds 1..60 and is therefore unsuitable for an STL leap-action
decision.

Random seeds choose an opponent's fixed parameters.  They do not introduce
hidden live randomness: for a fixed seed and public history,
``true_distribution`` is deterministic and side-effect free.  Canonical Arena
sampling remains outside the provider.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterator, Mapping

import numpy as np

from arena.contracts import CanonicalDecision, PublicGameOutcome, PublicHalfRound

ACTION_COUNT = 60
ACTIONS = tuple(range(1, ACTION_COUNT + 1))
_POSITIONS = np.arange(1, ACTION_COUNT + 1, dtype=np.float64)

FIXED = "fixed"
DETERMINISTIC = "deterministic"
EARLY = "early"
LATE = "late"
NARROW = "narrow"
MULTIMODAL = "multimodal"
STATE_THRESHOLD = "state_threshold"
PERIODIC = "periodic"
WIN_STAY_LOSE_SHIFT = "win_stay_lose_shift"
COPY_RECENT = "copy_recent"
COUNTER_RECENT = "counter_recent"
SWITCH = "switch"
RETREAT_AFTER_DETECTED_EXPLOITATION = "retreat_after_detected_exploitation"
BAIT_THEN_REVERSE = "bait_then_reverse"

SUPPORTED_FAMILIES = (
    FIXED,
    DETERMINISTIC,
    EARLY,
    LATE,
    NARROW,
    MULTIMODAL,
    STATE_THRESHOLD,
    PERIODIC,
    WIN_STAY_LOSE_SHIFT,
    COPY_RECENT,
    COUNTER_RECENT,
    SWITCH,
    RETREAT_AFTER_DETECTED_EXPLOITATION,
    BAIT_THEN_REVERSE,
)

_ALIASES = MappingProxyType(
    {
        "state_conditioned_threshold": STATE_THRESHOLD,
        "state_conditioned_thresholds": STATE_THRESHOLD,
        "win_stay": WIN_STAY_LOSE_SHIFT,
        "copy": COPY_RECENT,
        "counter": COUNTER_RECENT,
        "retreat": RETREAT_AFTER_DETECTED_EXPLOITATION,
        "bait_reverse": BAIT_THEN_REVERSE,
    }
)


@dataclass(frozen=True, slots=True)
class OpponentManifestEntry:
    """One immutable family and its predeclared parameter seeds."""

    family: str
    seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.family not in SUPPORTED_FAMILIES:
            raise ValueError(f"unsupported opponent family {self.family!r}")
        if not self.seeds or any(isinstance(seed, bool) for seed in self.seeds):
            raise ValueError("manifest seeds must be a nonempty tuple of integers")
        if any(not isinstance(seed, int) or seed < 0 for seed in self.seeds):
            raise ValueError("manifest seeds must be nonnegative integers")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("manifest seeds must be unique within a family")


@dataclass(frozen=True, slots=True)
class OpponentFamilyManifest:
    """Immutable opponent-family split used by training and evaluation."""

    split: str
    entries: tuple[OpponentManifestEntry, ...]
    schema_version: str = "arena-pure-dth-opponent-family-manifest-v1"

    def __post_init__(self) -> None:
        if self.split not in {"train", "validation", "test", "audit"}:
            raise ValueError(
                "opponent manifest split must be train/validation/test/audit"
            )
        families = tuple(entry.family for entry in self.entries)
        if not families or len(set(families)) != len(families):
            raise ValueError("manifest families must be nonempty and unique")

    @property
    def families(self) -> tuple[str, ...]:
        return tuple(entry.family for entry in self.entries)


TRAIN_FAMILY_MANIFEST = OpponentFamilyManifest(
    split="train",
    entries=(
        OpponentManifestEntry(FIXED, (1101, 1102, 1103, 1104)),
        OpponentManifestEntry(DETERMINISTIC, (1201, 1202, 1203, 1204)),
        OpponentManifestEntry(EARLY, (1301, 1302, 1303, 1304)),
        OpponentManifestEntry(LATE, (1401, 1402, 1403, 1404)),
        OpponentManifestEntry(NARROW, (1501, 1502, 1503, 1504)),
        OpponentManifestEntry(MULTIMODAL, (1601, 1602, 1603, 1604)),
        OpponentManifestEntry(STATE_THRESHOLD, (1701, 1702, 1703, 1704)),
        OpponentManifestEntry(PERIODIC, (1801, 1802, 1803, 1804)),
        OpponentManifestEntry(WIN_STAY_LOSE_SHIFT, (1901, 1902, 1903, 1904)),
        OpponentManifestEntry(COPY_RECENT, (2001, 2002, 2003, 2004)),
    ),
)

VALIDATION_FAMILY_MANIFEST = OpponentFamilyManifest(
    split="validation",
    entries=(
        OpponentManifestEntry(COUNTER_RECENT, (21_001, 21_002, 21_003, 21_004)),
        OpponentManifestEntry(SWITCH, (22_001, 22_002, 22_003, 22_004)),
    ),
)

TEST_FAMILY_MANIFEST = OpponentFamilyManifest(
    split="test",
    entries=(
        OpponentManifestEntry(
            RETREAT_AFTER_DETECTED_EXPLOITATION,
            (31_001, 31_002, 31_003, 31_004),
        ),
        OpponentManifestEntry(
            BAIT_THEN_REVERSE,
            (32_001, 32_002, 32_003, 32_004),
        ),
    ),
)

# Registered before the corrected v1 retrain.  Open this parameter-seed holdout
# only for the final four-condition memory/adapter audit; do not tune on it.
AUDIT_FAMILY_MANIFEST = OpponentFamilyManifest(
    split="audit",
    entries=(
        OpponentManifestEntry(
            RETREAT_AFTER_DETECTED_EXPLOITATION,
            tuple(range(41_001, 41_009)),
        ),
        OpponentManifestEntry(
            BAIT_THEN_REVERSE,
            tuple(range(42_001, 42_009)),
        ),
    ),
)

FAMILY_MANIFESTS: Mapping[str, OpponentFamilyManifest] = MappingProxyType(
    {
        "train": TRAIN_FAMILY_MANIFEST,
        "validation": VALIDATION_FAMILY_MANIFEST,
        "test": TEST_FAMILY_MANIFEST,
        "audit": AUDIT_FAMILY_MANIFEST,
    }
)


def _normalize(raw: np.ndarray) -> np.ndarray:
    values = np.asarray(raw, dtype=np.float64)
    if (
        values.shape != (ACTION_COUNT,)
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or float(values.sum()) <= 0.0
    ):
        raise ValueError("opponent distribution must be finite nonnegative length 60")
    result = values / float(values.sum())
    result.setflags(write=False)
    return result


def _point(action: int) -> np.ndarray:
    if not 1 <= action <= ACTION_COUNT:
        raise ValueError("pure-DTH action must lie in 1..60")
    result = np.zeros(ACTION_COUNT, dtype=np.float64)
    result[action - 1] = 1.0
    return result


def _gaussian(center: float, width: float) -> np.ndarray:
    return _normalize(np.exp(-0.5 * ((_POSITIONS - center) / width) ** 2) + 1e-8)


def _early(scale: float) -> np.ndarray:
    return _normalize(np.exp(-(_POSITIONS - 1.0) / scale) + 1e-8)


def _late(scale: float) -> np.ndarray:
    return _normalize(np.exp(-(ACTION_COUNT - _POSITIONS) / scale) + 1e-8)


def _canonical_family(raw: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("opponent family must be a nonempty string")
    normalized = raw.strip().lower().replace("-", "_")
    normalized = _ALIASES.get(normalized, normalized)
    if normalized not in SUPPORTED_FAMILIES:
        choices = ", ".join(SUPPORTED_FAMILIES)
        raise ValueError(f"unknown opponent family {raw!r}; choose one of: {choices}")
    return normalized


def _validate_decision(decision: CanonicalDecision) -> None:
    if decision.role not in {"dropper", "checker"}:
        raise ValueError("pure-DTH opponent role must be dropper or checker")
    if decision.legal_seconds != ACTIONS or decision.turn_duration != ACTION_COUNT:
        raise ValueError("pure-DTH opponent league supports literal actions 1..60 only")
    coordinates = (
        decision.checker_cylinder_seconds,
        decision.checker_ttd_seconds,
        decision.dropper_cylinder_seconds,
        decision.dropper_ttd_seconds,
    )
    if not all(np.isfinite(value) and value >= 0.0 for value in coordinates):
        raise ValueError(
            "public pure-DTH state coordinates must be finite and nonnegative"
        )


class ReactiveDTHOpponent:
    """One seeded public-history policy from the pure-DTH opponent league."""

    def __init__(self, family: str, *, seed: int) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("opponent seed must be a nonnegative integer")
        self.family = _canonical_family(family)
        self.seed = int(seed)
        rng = np.random.default_rng(self.seed)

        self._fixed = _normalize(rng.gamma(1.5, 1.0, ACTION_COUNT) + 0.05)
        self._deterministic_action = int(rng.integers(1, ACTION_COUNT + 1))
        self._early = _early(float(rng.uniform(4.0, 11.0)))
        self._late = _late(float(rng.uniform(4.0, 11.0)))
        self._narrow_center = int(rng.integers(5, 57))
        self._narrow = _gaussian(
            self._narrow_center,
            float(rng.uniform(1.2, 3.0)),
        )
        multimodal_centers = (
            int(rng.integers(5, 16)),
            int(rng.integers(24, 37)),
            int(rng.integers(45, 57)),
        )
        multimodal = np.full(ACTION_COUNT, 1e-5, dtype=np.float64)
        for center in multimodal_centers:
            width = float(rng.uniform(1.0, 2.2))
            multimodal += np.exp(-0.5 * ((_POSITIONS - center) / width) ** 2)
        self._multimodal = _normalize(multimodal)

        self._cylinder_threshold = float(rng.integers(105, 226))
        self._ttd_threshold = float(rng.integers(75, 196))
        self._period = int(rng.integers(2, 6))
        self._current_action = int(rng.integers(1, ACTION_COUNT + 1))
        self._lose_shift = int(rng.integers(7, 30))
        self._recent_opponent_action = int(rng.integers(1, ACTION_COUNT + 1))
        self._switch_game = int(rng.integers(1, 4))
        self._switch_decision = int(rng.integers(7, 15))
        self._retreat_trigger = int(rng.integers(2, 4))
        self._retreat_duration = int(rng.integers(4, 9))
        self._retreat_center = int(rng.integers(8, 53))
        self._retreat_bait = _gaussian(
            self._retreat_center,
            float(rng.uniform(1.0, 2.0)),
        )
        self._bait_decisions = int(rng.integers(3, 8))
        self._bait_center = int(rng.integers(7, 55))
        self._bait = _gaussian(self._bait_center, float(rng.uniform(1.0, 2.0)))
        self._reverse = _gaussian(
            ACTION_COUNT + 1 - self._bait_center,
            float(rng.uniform(1.0, 2.0)),
        )
        self._uniform = _normalize(np.ones(ACTION_COUNT, dtype=np.float64))

        self._self_name: str | None = None
        self._pending_role: str | None = None
        self._awaiting_reveal = False
        self._game_started = False
        self._game_index = 0
        self._decision_count = 0
        self._observation_count = 0
        self._loss_streak = 0
        self._retreat_until = 0
        self._recent_local_wins: deque[bool] = deque(maxlen=8)
        self._last_outcome: PublicGameOutcome | None = None

    @property
    def game_index(self) -> int:
        return self._game_index

    @property
    def decision_count(self) -> int:
        return self._decision_count

    @property
    def observation_count(self) -> int:
        return self._observation_count

    def reset_game(self) -> None:
        """Start a game while retaining session-persistent reactive memory."""

        if self._awaiting_reveal:
            raise RuntimeError("opponent game reset with an unrevealed action")
        if self._game_started:
            self._game_index += 1
        else:
            self._game_started = True
        # Repeated-opponent training may alternate the learner's seat between
        # games.  This provider then controls the other public seat name while
        # retaining the same opponent strategy and public-history memory.
        self._self_name = None
        self._pending_role = None

    def reset_session(self) -> None:
        """Restore this seeded opponent to its independent-session initial state."""

        if self._awaiting_reveal:
            raise RuntimeError("opponent session reset with an unrevealed action")
        family, seed = self.family, self.seed
        self.__init__(family, seed=seed)

    def true_distribution(self, decision: CanonicalDecision) -> np.ndarray:
        """Return the deterministic current policy without advancing its state."""

        _validate_decision(decision)
        family = self.family
        if family == FIXED:
            distribution = self._fixed
        elif family == DETERMINISTIC:
            distribution = _point(self._deterministic_action)
        elif family == EARLY:
            distribution = self._early
        elif family == LATE:
            distribution = self._late
        elif family == NARROW:
            distribution = self._narrow
        elif family == MULTIMODAL:
            distribution = self._multimodal
        elif family == STATE_THRESHOLD:
            actor_cylinder, actor_ttd = self._actor_load(decision)
            high_load = (
                actor_cylinder >= self._cylinder_threshold
                or actor_ttd >= self._ttd_threshold
            )
            distribution = self._late if high_load else self._early
        elif family == PERIODIC:
            offset = 0 if decision.role == "dropper" else 1
            mode = (self._decision_count // self._period + offset) % 3
            distribution = (self._early, self._late, self._narrow)[mode]
        elif family == WIN_STAY_LOSE_SHIFT:
            distribution = _point(self._current_action)
        elif family == COPY_RECENT:
            distribution = _point(self._recent_opponent_action)
        elif family == COUNTER_RECENT:
            distribution = _point(ACTION_COUNT + 1 - self._recent_opponent_action)
        elif family == SWITCH:
            switched = (
                self._game_index >= self._switch_game
                or self._decision_count >= self._switch_decision
            )
            distribution = self._late if switched else self._early
        elif family == RETREAT_AFTER_DETECTED_EXPLOITATION:
            distribution = (
                self._uniform
                if self._decision_count < self._retreat_until
                else self._retreat_bait
            )
        elif family == BAIT_THEN_REVERSE:
            distribution = (
                self._bait
                if self._decision_count < self._bait_decisions
                else self._reverse
            )
        else:  # pragma: no cover - construction validates the closed family set.
            raise RuntimeError(f"unimplemented opponent family {family!r}")
        return np.asarray(distribution, dtype=np.float64).copy()

    def policy(self, decision: CanonicalDecision) -> Mapping[int, float]:
        """Return a canonical policy and await its simultaneous public reveal."""

        if self._awaiting_reveal:
            raise RuntimeError("opponent was asked to act twice before a reveal")
        if self._self_name is None:
            self._self_name = decision.actor_name
        elif decision.actor_name.casefold() != self._self_name.casefold():
            raise RuntimeError(
                "one reactive opponent provider cannot control two seats"
            )
        distribution = self.true_distribution(decision)
        self._pending_role = decision.role
        self._awaiting_reveal = True
        self._decision_count += 1
        return {
            action: float(distribution[action - 1])
            for action in ACTIONS
            if distribution[action - 1] > 0.0
        }

    def observe(self, record: PublicHalfRound) -> None:
        """Update reactive memory from one simultaneous public reveal."""

        if not self._awaiting_reveal or self._self_name is None:
            raise RuntimeError("opponent received a reveal without a pending action")
        if self._self_name.casefold() == record.dropper_name.casefold():
            role = "dropper"
            own_action = int(record.drop_time)
            opponent_action = int(record.check_time)
            local_win = record.check_time < record.drop_time
        elif self._self_name.casefold() == record.checker_name.casefold():
            role = "checker"
            own_action = int(record.check_time)
            opponent_action = int(record.drop_time)
            local_win = record.check_time >= record.drop_time
        else:
            raise RuntimeError("public reveal does not contain this opponent's seat")
        if role != self._pending_role:
            raise RuntimeError("public reveal role disagrees with the pending decision")
        if (
            not 1 <= own_action <= ACTION_COUNT
            or not 1 <= opponent_action <= ACTION_COUNT
        ):
            raise ValueError("pure-DTH opponent reveal actions must lie in 1..60")

        self._recent_opponent_action = opponent_action
        self._recent_local_wins.append(bool(local_win))
        if self.family == WIN_STAY_LOSE_SHIFT:
            self._current_action = (
                own_action
                if local_win
                else ((own_action - 1 + self._lose_shift) % ACTION_COUNT) + 1
            )
        if self.family == RETREAT_AFTER_DETECTED_EXPLOITATION:
            self._loss_streak = 0 if local_win else self._loss_streak + 1
            if self._loss_streak >= self._retreat_trigger:
                self._retreat_until = self._decision_count + self._retreat_duration
                self._loss_streak = 0

        self._observation_count += 1
        self._pending_role = None
        self._awaiting_reveal = False

    def end_game(self, outcome: PublicGameOutcome) -> None:
        """Record a public game boundary without erasing opponent memory."""

        if self._awaiting_reveal:
            raise RuntimeError("opponent game ended with an unrevealed action")
        self._last_outcome = outcome

    @staticmethod
    def _actor_load(decision: CanonicalDecision) -> tuple[float, float]:
        if decision.role == "dropper":
            return (
                float(decision.dropper_cylinder_seconds),
                float(decision.dropper_ttd_seconds),
            )
        return (
            float(decision.checker_cylinder_seconds),
            float(decision.checker_ttd_seconds),
        )


def make_opponent(family: str, *, seed: int) -> ReactiveDTHOpponent:
    """Construct a deterministic seeded pure-DTH opponent family member."""

    return ReactiveDTHOpponent(family, seed=seed)


def iter_manifest_opponents(
    manifest: OpponentFamilyManifest,
) -> Iterator[ReactiveDTHOpponent]:
    """Instantiate every predeclared family/seed pair in manifest order."""

    for entry in manifest.entries:
        for seed in entry.seeds:
            yield make_opponent(entry.family, seed=seed)


__all__ = [
    "ACTION_COUNT",
    "ACTIONS",
    "SUPPORTED_FAMILIES",
    "OpponentManifestEntry",
    "OpponentFamilyManifest",
    "TRAIN_FAMILY_MANIFEST",
    "VALIDATION_FAMILY_MANIFEST",
    "TEST_FAMILY_MANIFEST",
    "FAMILY_MANIFESTS",
    "ReactiveDTHOpponent",
    "make_opponent",
    "iter_manifest_opponents",
]
