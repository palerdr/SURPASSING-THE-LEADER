"""Injectable memory-necessity curriculum for Aggro Hal.

Each curriculum case is a matched pair of legal pure-DTH sessions.  The two
arms have the same mechanics and the same current target tensors, but earlier
public opponent actions identify one of two latent modes.  Those modes require
different unique best responses in both seats.  A learner therefore cannot
solve the target from the current observation alone.

This module owns data construction only.  It does not import evaluation
internals or choose a training loss.  :class:`MemoryCurriculumBatch` exposes
the exact five tensors accepted by :class:`~arena.policies.aggro_hal.AggroHalNetwork`
plus target-only supervision.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
import torch
from torch import Tensor

from arena.contracts import CanonicalDecision, PublicGameOutcome, PublicHalfRound
from arena.policies.aggro_env import AggroDecision, AggroSessionEnv
from arena.policies.aggro_hal import (
    ACTION_COUNT,
    OBSERVATION_DIM,
    OBSERVATION_FEATURES,
    OBSERVATION_SCHEMA,
)
from dth.agent import CertifiedStageGame

MEMORY_CURRICULUM_SCHEMA = "arena-aggro-hal-memory-curriculum-v1"
MEMORY_CURRICULUM_SPLIT_SCHEMA = "arena-aggro-hal-memory-curriculum-split-v1"
MEMORY_CURRICULUM_TOKEN_SCHEMA = "arena-aggro-hal-memory-curriculum-token-v1"
MEMORY_CURRICULUM_BATCH_SCHEMA = "arena-aggro-hal-memory-curriculum-batch-v1"
MEMORY_CURRICULUM_BINDING_SCHEMA = "arena-aggro-hal-memory-curriculum-binding-v1"
MEMORY_CURRICULUM_GENERATOR_CONTRACT_SCHEMA = (
    "arena-aggro-hal-memory-curriculum-generator-contract-v1"
)

Mode = Literal["a", "b"]
Role = Literal["dropper", "checker"]
SplitName = Literal["train", "validation"]

_ROLES: tuple[Role, ...] = ("dropper", "checker")
_MODES: tuple[Mode, ...] = ("a", "b")


class StageGameProvider(Protocol):
    """The sole exact-solver capability needed by the curriculum."""

    def stage_game(self, state: tuple[int, int, int, int]) -> CertifiedStageGame: ...


def _readonly_array(values: object, *, dtype: np.dtype) -> np.ndarray:
    result = np.ascontiguousarray(values, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _validate_actions(values: Sequence[int], *, label: str) -> tuple[int, ...]:
    raw = tuple(values)
    if not raw:
        raise ValueError(f"{label} must be nonempty")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or not 1 <= int(value) <= ACTION_COUNT
        for value in raw
    ):
        raise ValueError(f"{label} must contain literal actions in 1..60")
    return tuple(int(value) for value in raw)


def _distribution(values: object, *, label: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if (
        result.shape != (ACTION_COUNT,)
        or not np.all(np.isfinite(result))
        or np.any(result < 0.0)
        or not np.isclose(float(result.sum()), 1.0, atol=1e-10)
    ):
        raise ValueError(f"{label} must be a finite length-60 probability distribution")
    return _readonly_array(result, dtype=np.dtype(np.float32))


def mode_target_distributions() -> dict[Mode, np.ndarray]:
    """Return the frozen checkpoint-independent target distributions.

    These targets were selected from the exact zero-load DTH stage game.  Every
    constructed case independently rechecks that they have distinct unique
    best responses in both roles; a changed solver artifact therefore fails
    closed instead of silently weakening the curriculum.
    """

    mode_a = np.zeros(ACTION_COUNT, dtype=np.float64)
    mode_b = np.zeros(ACTION_COUNT, dtype=np.float64)
    mode_a[7], mode_a[59] = 0.20, 0.80
    mode_b[20], mode_b[58] = 0.90, 0.10
    return {
        "a": _distribution(mode_a, label="mode a target"),
        "b": _distribution(mode_b, label="mode b target"),
    }


@dataclass(frozen=True, slots=True)
class MemoryCurriculumSplit:
    """Immutable seed and public-cue namespace for one experimental split."""

    name: SplitName
    example_seeds: tuple[int, ...]
    cue_template_a: tuple[int, ...]
    cue_template_b: tuple[int, ...]
    cover_actions: tuple[int, ...]
    cover_games: tuple[int, ...]
    start_clocks: tuple[int, ...]
    session_seed_offset: int
    schema_version: str = MEMORY_CURRICULUM_SPLIT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != MEMORY_CURRICULUM_SPLIT_SCHEMA:
            raise ValueError("unsupported memory curriculum split schema")
        if self.name not in {"train", "validation"}:
            raise ValueError("curriculum split must be train or validation")
        seeds = tuple(int(seed) for seed in self.example_seeds)
        if (
            not seeds
            or len(set(seeds)) != len(seeds)
            or any(
                isinstance(seed, bool)
                or not isinstance(seed, (int, np.integer))
                or int(seed) < 0
                for seed in self.example_seeds
            )
        ):
            raise ValueError("example seeds must be unique nonnegative integers")
        cue_a = _validate_actions(self.cue_template_a, label="mode a cue template")
        cue_b = _validate_actions(self.cue_template_b, label="mode b cue template")
        if len(cue_a) != len(cue_b) or len(cue_a) < 2:
            raise ValueError(
                "mode cue templates must have equal length of at least two"
            )
        if set(cue_a) & set(cue_b):
            raise ValueError("mode cue supports must be disjoint and identifiable")
        covers = _validate_actions(self.cover_actions, label="cover actions")
        cover_games = tuple(int(value) for value in self.cover_games)
        if not cover_games or any(
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or int(value) < 2
            for value in self.cover_games
        ):
            raise ValueError("cover_games must contain integers of at least two")
        clocks = tuple(int(value) for value in self.start_clocks)
        if not clocks or any(
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or int(value) < 0
            for value in self.start_clocks
        ):
            raise ValueError("start clocks must be nonnegative integers")
        if (
            isinstance(self.session_seed_offset, bool)
            or not isinstance(self.session_seed_offset, (int, np.integer))
            or int(self.session_seed_offset) < 0
        ):
            raise ValueError("session seed offset must be a nonnegative integer")
        object.__setattr__(self, "example_seeds", seeds)
        object.__setattr__(self, "cue_template_a", cue_a)
        object.__setattr__(self, "cue_template_b", cue_b)
        object.__setattr__(self, "cover_actions", covers)
        object.__setattr__(self, "cover_games", cover_games)
        object.__setattr__(self, "start_clocks", clocks)

    @property
    def parameter_support(self) -> dict[str, frozenset[int]]:
        """Return all public generator parameters whose split overlap is forbidden."""

        return {
            "cue_actions": frozenset(self.cue_template_a + self.cue_template_b),
            "cover_actions": frozenset(self.cover_actions),
            "cover_games": frozenset(self.cover_games),
            "start_clocks": frozenset(self.start_clocks),
        }


TRAIN_MEMORY_CURRICULUM = MemoryCurriculumSplit(
    name="train",
    example_seeds=tuple(range(64)),
    cue_template_a=tuple(range(42, 61, 2)),
    cue_template_b=tuple(range(2, 21, 2)),
    cover_actions=(28, 30),
    cover_games=(2, 4, 6),
    start_clocks=(600, 720),
    session_seed_offset=100_000_000,
)

VALIDATION_MEMORY_CURRICULUM = MemoryCurriculumSplit(
    name="validation",
    example_seeds=tuple(range(10_000, 10_016)),
    cue_template_a=tuple(range(41, 60, 2)),
    cue_template_b=tuple(range(3, 22, 2)),
    cover_actions=(27, 29),
    cover_games=(8,),
    start_clocks=(840, 960),
    session_seed_offset=200_000_000,
)


def assert_split_disjointness(
    train: MemoryCurriculumSplit = TRAIN_MEMORY_CURRICULUM,
    validation: MemoryCurriculumSplit = VALIDATION_MEMORY_CURRICULUM,
) -> None:
    """Fail unless seeds and every exposed generator parameter are disjoint."""

    if train.name != "train" or validation.name != "validation":
        raise ValueError("split disjointness requires train and validation specs")
    if set(train.example_seeds) & set(validation.example_seeds):
        raise ValueError("train and validation curriculum seeds overlap")
    for label in train.parameter_support:
        overlap = train.parameter_support[label] & validation.parameter_support[label]
        if overlap:
            raise ValueError(f"train and validation {label} overlap: {sorted(overlap)}")
    train_sessions = {
        train.session_seed_offset + seed * 16 + role_index
        for seed in train.example_seeds
        for role_index in range(len(_ROLES))
    }
    validation_sessions = {
        validation.session_seed_offset + seed * 16 + role_index
        for seed in validation.example_seeds
        for role_index in range(len(_ROLES))
    }
    if train_sessions & validation_sessions:
        raise ValueError("train and validation session seed namespaces overlap")


assert_split_disjointness()


def memory_curriculum_split(name: SplitName) -> MemoryCurriculumSplit:
    if name == "train":
        return TRAIN_MEMORY_CURRICULUM
    if name == "validation":
        return VALIDATION_MEMORY_CURRICULUM
    raise ValueError("curriculum split must be train or validation")


def memory_curriculum_generator_contract() -> dict[str, object]:
    """Freeze the procedural mechanics that turn split values into histories."""

    return {
        "schema_version": MEMORY_CURRICULUM_GENERATOR_CONTRACT_SCHEMA,
        "parameter_rng": {
            "engine": "numpy.default_rng-pcg64",
            "seed": "session_seed_offset + example_seed",
            "draw_order": [
                "shared_cue_permutation",
                "cover_action_index",
                "cover_games_index",
                "start_clock_index",
            ],
            "same_permutation_applied_to_both_mode_templates": True,
        },
        "session_seed": (
            "session_seed_offset + example_seed * 16 + int(role == checker)"
        ),
        "target_game_index": "len(cue_actions) + cover_games",
        "environment": {
            "games_per_session": "target_game_index + 1",
            "max_half_rounds": 1,
            "start_clocks": "singleton selected start_clock",
            "learner_starts_in_hal_seat": (
                "(role == dropper) if target_game_index is even else (role != dropper)"
            ),
        },
        "scripted_learner_action": {
            "dropper": 1,
            "checker": 60,
            "purpose": "equal check-success outcomes and mechanics across modes",
        },
        "opponent_schedule": [
            "one deterministic cue action per cue game",
            "one deterministic shared cover action per cover game",
            "mode target distribution in the final game",
        ],
        "truth_boundary": "target distribution becomes available only after action",
        "network_supervision": {
            "prefix_and_cover_tokens_update_gru": True,
            "objective_weight": "one on final target token and zero elsewhere",
        },
    }


def memory_curriculum_generator_contract_sha256() -> str:
    canonical = json.dumps(
        memory_curriculum_generator_contract(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def memory_curriculum_config_payload(
    split: SplitName | MemoryCurriculumSplit,
) -> dict[str, object]:
    """Return the canonical JSON-safe config a collector can checkpoint-bind.

    The trainer-facing collector may wrap this payload in its own binding
    schema.  Keeping the curriculum hash independent avoids an import cycle
    with ``train_aggro_hal``.
    """

    spec = memory_curriculum_split(split) if isinstance(split, str) else split
    targets = mode_target_distributions()
    payload: dict[str, object] = {
        "schema_version": MEMORY_CURRICULUM_BINDING_SCHEMA,
        "curriculum_schema": MEMORY_CURRICULUM_SCHEMA,
        "split_schema": MEMORY_CURRICULUM_SPLIT_SCHEMA,
        "observation_schema": OBSERVATION_SCHEMA,
        "observation_features_sha256": hashlib.sha256(
            json.dumps(
                list(OBSERVATION_FEATURES),
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "action_count": ACTION_COUNT,
        "generator_contract": memory_curriculum_generator_contract(),
        "generator_contract_sha256": (memory_curriculum_generator_contract_sha256()),
        "mode_targets": {
            mode: {
                str(action): float(distribution[action - 1])
                for action in range(1, ACTION_COUNT + 1)
                if distribution[action - 1] > 0.0
            }
            for mode, distribution in targets.items()
        },
        "split": {
            "name": spec.name,
            "example_seeds": list(spec.example_seeds),
            "cue_template_a": list(spec.cue_template_a),
            "cue_template_b": list(spec.cue_template_b),
            "cover_actions": list(spec.cover_actions),
            "cover_games": list(spec.cover_games),
            "start_clocks": list(spec.start_clocks),
            "session_seed_offset": spec.session_seed_offset,
        },
        "loss_contract": {
            "prefix_consumes_recurrence": True,
            "supervision": "target_only",
            "target_input_identical_across_modes": True,
            "required_roles": list(_ROLES),
        },
    }
    return payload


def memory_curriculum_config_sha256(
    split: SplitName | MemoryCurriculumSplit,
) -> str:
    """Return a stable digest for resume/checkpoint compatibility gates."""

    canonical = json.dumps(
        memory_curriculum_config_payload(split),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


@dataclass(frozen=True, slots=True)
class MemoryCurriculumParameters:
    """Concrete deterministic parameters selected for one paired case."""

    split: SplitName
    example_seed: int
    cue_actions_a: tuple[int, ...]
    cue_actions_b: tuple[int, ...]
    cover_action: int
    cover_games: int
    start_clock: int
    session_seed: int

    def __post_init__(self) -> None:
        if self.split not in {"train", "validation"}:
            raise ValueError("parameter split must be train or validation")
        if (
            isinstance(self.example_seed, bool)
            or not isinstance(self.example_seed, (int, np.integer))
            or int(self.example_seed) < 0
        ):
            raise ValueError("example seed must be nonnegative")
        cue_a = _validate_actions(self.cue_actions_a, label="mode a cue actions")
        cue_b = _validate_actions(self.cue_actions_b, label="mode b cue actions")
        if len(cue_a) != len(cue_b) or set(cue_a) & set(cue_b):
            raise ValueError("concrete mode cues must be equal-length and disjoint")
        if (
            isinstance(self.cover_action, bool)
            or not isinstance(self.cover_action, (int, np.integer))
            or not 1 <= int(self.cover_action) <= ACTION_COUNT
            or isinstance(self.cover_games, bool)
            or not isinstance(self.cover_games, (int, np.integer))
            or int(self.cover_games) < 2
        ):
            raise ValueError("cover schedule is outside the curriculum contract")
        if (
            isinstance(self.start_clock, bool)
            or not isinstance(self.start_clock, (int, np.integer))
            or int(self.start_clock) < 0
            or isinstance(self.session_seed, bool)
            or not isinstance(self.session_seed, (int, np.integer))
            or int(self.session_seed) < 0
        ):
            raise ValueError("clock and session seed must be nonnegative")
        object.__setattr__(self, "example_seed", int(self.example_seed))
        object.__setattr__(self, "cue_actions_a", cue_a)
        object.__setattr__(self, "cue_actions_b", cue_b)
        object.__setattr__(self, "cover_action", int(self.cover_action))
        object.__setattr__(self, "cover_games", int(self.cover_games))
        object.__setattr__(self, "start_clock", int(self.start_clock))
        object.__setattr__(self, "session_seed", int(self.session_seed))


def select_curriculum_parameters(
    split: SplitName | MemoryCurriculumSplit,
    *,
    example_seed: int,
    role: Role,
) -> MemoryCurriculumParameters:
    """Select one deterministic paired schedule from a frozen split."""

    spec = memory_curriculum_split(split) if isinstance(split, str) else split
    if role not in _ROLES:
        raise ValueError("role must be dropper or checker")
    if example_seed not in spec.example_seeds:
        raise ValueError(f"seed {example_seed} is not registered in split {spec.name}")
    rng = np.random.default_rng(spec.session_seed_offset + int(example_seed))
    permutation = rng.permutation(len(spec.cue_template_a))
    cue_a = tuple(np.asarray(spec.cue_template_a, dtype=np.int64)[permutation].tolist())
    cue_b = tuple(np.asarray(spec.cue_template_b, dtype=np.int64)[permutation].tolist())
    cover_action = int(spec.cover_actions[int(rng.integers(len(spec.cover_actions)))])
    cover_games = int(spec.cover_games[int(rng.integers(len(spec.cover_games)))])
    start_clock = int(spec.start_clocks[int(rng.integers(len(spec.start_clocks)))])
    session_seed = (
        spec.session_seed_offset + int(example_seed) * 16 + int(role == "checker")
    )
    return MemoryCurriculumParameters(
        split=spec.name,
        example_seed=int(example_seed),
        cue_actions_a=cue_a,
        cue_actions_b=cue_b,
        cover_action=cover_action,
        cover_games=cover_games,
        start_clock=start_clock,
        session_seed=session_seed,
    )


@dataclass(frozen=True, slots=True)
class MemoryCurriculumToken:
    """One immutable set of tensors consumed by ``AggroHalNetwork``."""

    features: np.ndarray
    stage_matrix: np.ndarray
    exact_policy: np.ndarray
    role_is_dropper: bool
    legal_mask: np.ndarray
    schema_version: str = MEMORY_CURRICULUM_TOKEN_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != MEMORY_CURRICULUM_TOKEN_SCHEMA:
            raise ValueError("unsupported memory curriculum token schema")
        features = _readonly_array(self.features, dtype=np.dtype(np.float32))
        matrix = _readonly_array(self.stage_matrix, dtype=np.dtype(np.float32))
        exact = _readonly_array(self.exact_policy, dtype=np.dtype(np.float32))
        legal = _readonly_array(self.legal_mask, dtype=np.dtype(np.bool_))
        if features.shape != (OBSERVATION_DIM,) or not np.all(np.isfinite(features)):
            raise ValueError("token features must be finite observation_dim float32")
        if matrix.shape != (ACTION_COUNT, ACTION_COUNT) or not np.all(
            np.isfinite(matrix)
        ):
            raise ValueError("token stage matrix must be finite 60x60")
        if (
            exact.shape != (ACTION_COUNT,)
            or not np.all(np.isfinite(exact))
            or np.any(exact < 0.0)
            or not np.isclose(float(exact.sum()), 1.0, atol=1e-5)
        ):
            raise ValueError("token exact policy must be a length-60 distribution")
        if legal.shape != (ACTION_COUNT,) or not np.any(legal):
            raise ValueError("token legal mask must admit at least one of 60 actions")
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "stage_matrix", matrix)
        object.__setattr__(self, "exact_policy", exact)
        object.__setattr__(self, "legal_mask", legal)

    @classmethod
    def from_decision(cls, decision: AggroDecision) -> MemoryCurriculumToken:
        legal = np.asarray(
            [
                action in decision.canonical_decision.legal_seconds
                for action in range(1, ACTION_COUNT + 1)
            ],
            dtype=np.bool_,
        )
        return cls(
            features=decision.observation,
            stage_matrix=decision.stage_matrix,
            exact_policy=decision.exact_policy,
            role_is_dropper=decision.role == "dropper",
            legal_mask=legal,
        )

    def bitwise_equal(self, other: MemoryCurriculumToken) -> bool:
        if not isinstance(other, MemoryCurriculumToken):
            return False

        def same(left: np.ndarray, right: np.ndarray) -> bool:
            return (
                left.dtype == right.dtype
                and left.shape == right.shape
                and left.tobytes() == right.tobytes()
            )

        return (
            self.schema_version == other.schema_version
            and self.role_is_dropper == other.role_is_dropper
            and same(self.features, other.features)
            and same(self.stage_matrix, other.stage_matrix)
            and same(self.exact_policy, other.exact_policy)
            and same(self.legal_mask, other.legal_mask)
        )

    def sha256(self) -> str:
        digest = hashlib.sha256(self.schema_version.encode("ascii"))
        for label, array in (
            ("features", self.features),
            ("stage_matrix", self.stage_matrix),
            ("exact_policy", self.exact_policy),
            ("legal_mask", self.legal_mask),
        ):
            digest.update(label.encode("ascii"))
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(repr(array.shape).encode("ascii"))
            digest.update(array.tobytes())
        digest.update(bytes((int(self.role_is_dropper),)))
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class MemoryCurriculumArm:
    """One latent-mode sequence, including its shared current target token."""

    mode: Mode
    cue_actions: tuple[int, ...]
    tokens: tuple[MemoryCurriculumToken, ...]
    opponent_targets: np.ndarray

    def __post_init__(self) -> None:
        if self.mode not in _MODES:
            raise ValueError("curriculum arm mode must be a or b")
        cues = _validate_actions(self.cue_actions, label=f"mode {self.mode} cues")
        tokens = tuple(self.tokens)
        targets = _readonly_array(self.opponent_targets, dtype=np.dtype(np.float32))
        if not tokens:
            raise ValueError("curriculum arm must contain recurrent tokens")
        if targets.shape != (len(tokens), ACTION_COUNT):
            raise ValueError("opponent targets must have shape [time, 60]")
        if (
            not np.all(np.isfinite(targets))
            or np.any(targets < 0.0)
            or not np.allclose(targets.sum(axis=-1), 1.0, atol=1e-5)
        ):
            raise ValueError("every curriculum opponent target must be a distribution")
        object.__setattr__(self, "cue_actions", cues)
        object.__setattr__(self, "tokens", tokens)
        object.__setattr__(self, "opponent_targets", targets)

    @property
    def target(self) -> MemoryCurriculumToken:
        return self.tokens[-1]

    @property
    def target_truth(self) -> np.ndarray:
        return self.opponent_targets[-1]


@dataclass(frozen=True, slots=True)
class ExactBestResponseAudit:
    """Unique one-based best responses and runner-up gaps for both modes/seats."""

    dropper_actions: tuple[int, int]
    checker_actions: tuple[int, int]
    dropper_gaps: tuple[float, float]
    checker_gaps: tuple[float, float]

    def __post_init__(self) -> None:
        for label, actions in (
            ("dropper", self.dropper_actions),
            ("checker", self.checker_actions),
        ):
            if len(actions) != 2 or any(
                not 1 <= action <= ACTION_COUNT for action in actions
            ):
                raise ValueError(f"{label} audit actions must lie in 1..60")
            if actions[0] == actions[1]:
                raise ValueError(f"latent modes do not conflict for {label}")
        for label, gaps in (
            ("dropper", self.dropper_gaps),
            ("checker", self.checker_gaps),
        ):
            if len(gaps) != 2 or any(
                not np.isfinite(gap) or gap <= 0.0 for gap in gaps
            ):
                raise ValueError(f"{label} best response must be unique in both modes")

    def action(self, role: Role, mode: Mode) -> int:
        if role not in _ROLES or mode not in _MODES:
            raise ValueError("best response lookup requires a valid role and mode")
        actions = self.dropper_actions if role == "dropper" else self.checker_actions
        return actions[0 if mode == "a" else 1]


def _unique_best_response(values: np.ndarray, *, label: str) -> tuple[int, float]:
    if values.shape != (ACTION_COUNT,) or not np.all(np.isfinite(values)):
        raise RuntimeError(f"{label} action values are malformed")
    order = np.argsort(values, kind="stable")
    best = int(order[-1])
    gap = float(values[best] - values[int(order[-2])])
    tolerance = 1e-8 * max(1.0, float(np.max(np.abs(values))))
    if gap <= tolerance:
        raise RuntimeError(f"{label} has no unique exact best response")
    return best + 1, gap


def audit_exact_best_responses(
    stage_matrix: np.ndarray,
    mode_targets: Mapping[Mode, np.ndarray] | None = None,
) -> ExactBestResponseAudit:
    """Prove that the two latent truths demand conflicting actions in both roles."""

    matrix = np.asarray(stage_matrix, dtype=np.float64)
    if matrix.shape != (ACTION_COUNT, ACTION_COUNT) or not np.all(np.isfinite(matrix)):
        raise ValueError("curriculum audit requires a finite 60x60 stage matrix")
    targets = mode_target_distributions() if mode_targets is None else mode_targets
    truth_a = np.asarray(targets["a"], dtype=np.float64)
    truth_b = np.asarray(targets["b"], dtype=np.float64)
    drop_a = _unique_best_response(matrix @ truth_a, label="dropper mode a")
    drop_b = _unique_best_response(matrix @ truth_b, label="dropper mode b")
    check_a = _unique_best_response(-matrix.T @ truth_a, label="checker mode a")
    check_b = _unique_best_response(-matrix.T @ truth_b, label="checker mode b")
    return ExactBestResponseAudit(
        dropper_actions=(drop_a[0], drop_b[0]),
        checker_actions=(check_a[0], check_b[0]),
        dropper_gaps=(drop_a[1], drop_b[1]),
        checker_gaps=(check_a[1], check_b[1]),
    )


class _ScheduledModeOpponent:
    """Private deterministic provider used to produce legal public histories."""

    def __init__(
        self,
        *,
        cue_actions: Sequence[int],
        cover_action: int,
        cover_games: int,
        target_truth: np.ndarray,
    ) -> None:
        self.cue_actions = _validate_actions(cue_actions, label="scheduled cues")
        self.cover_action = int(cover_action)
        self.cover_games = int(cover_games)
        self.target_truth = _distribution(target_truth, label="scheduled target truth")
        self._game_index = -1

    @property
    def target_game_index(self) -> int:
        return len(self.cue_actions) + self.cover_games

    def reset_session(self) -> None:
        self._game_index = -1

    def reset_game(self) -> None:
        self._game_index += 1

    def _current_distribution(self) -> np.ndarray:
        if self._game_index < 0:
            raise RuntimeError("scheduled opponent acted before reset_game")
        if self._game_index < len(self.cue_actions):
            action = self.cue_actions[self._game_index]
            result = np.zeros(ACTION_COUNT, dtype=np.float64)
            result[action - 1] = 1.0
            return result
        if self._game_index < self.target_game_index:
            result = np.zeros(ACTION_COUNT, dtype=np.float64)
            result[self.cover_action - 1] = 1.0
            return result
        if self._game_index == self.target_game_index:
            return np.asarray(self.target_truth, dtype=np.float64).copy()
        raise RuntimeError("scheduled opponent advanced beyond target game")

    def true_distribution(self, decision: CanonicalDecision) -> np.ndarray:
        del decision
        return self._current_distribution()

    def policy(self, decision: CanonicalDecision) -> Mapping[int, float]:
        del decision
        distribution = self._current_distribution()
        return {
            action: float(distribution[action - 1])
            for action in range(1, ACTION_COUNT + 1)
            if distribution[action - 1] > 0.0
        }

    def observe(self, record: PublicHalfRound) -> None:
        del record

    def end_game(self, outcome: PublicGameOutcome) -> None:
        del outcome


def _scripted_learner_action(decision: AggroDecision) -> int:
    # Every cue and cover action resolves as check_success.  Thus aligned arms
    # differ in public opponent-action one-hots, never in outcome or load state.
    return 1 if decision.role == "dropper" else ACTION_COUNT


def _collect_arm(
    exact_agent: StageGameProvider,
    *,
    mode: Mode,
    role: Role,
    cue_actions: tuple[int, ...],
    parameters: MemoryCurriculumParameters,
    target_truth: np.ndarray,
) -> MemoryCurriculumArm:
    target_index = len(cue_actions) + parameters.cover_games
    learner_is_hal = role == "dropper"
    starts_hal = learner_is_hal if target_index % 2 == 0 else not learner_is_hal
    opponent = _ScheduledModeOpponent(
        cue_actions=cue_actions,
        cover_action=parameters.cover_action,
        cover_games=parameters.cover_games,
        target_truth=target_truth,
    )
    env = AggroSessionEnv(
        opponent,
        exact_agent,  # type: ignore[arg-type]
        games_per_session=target_index + 1,
        seed=parameters.session_seed,
        start_clocks=(parameters.start_clock,),
        max_half_rounds=1,
        learner_starts_in_hal_seat=starts_hal,
    )
    tokens: list[MemoryCurriculumToken] = []
    targets: list[np.ndarray] = []
    decision = env.reset(seed=parameters.session_seed)
    try:
        while True:
            tokens.append(MemoryCurriculumToken.from_decision(decision))
            step = env.step(_scripted_learner_action(decision))
            truth = step.record.opponent_true_distribution
            if truth is None:
                raise RuntimeError("scheduled opponent omitted its training truth")
            targets.append(np.asarray(truth, dtype=np.float32))
            if step.session_done:
                if decision.game_index != target_index or decision.role != role:
                    raise RuntimeError(
                        "curriculum session ended outside its target role/game"
                    )
                break
            if step.next_decision is None:
                raise RuntimeError("curriculum session ended before its target")
            decision = step.next_decision
    finally:
        env.close()
    return MemoryCurriculumArm(
        mode=mode,
        cue_actions=cue_actions,
        tokens=tuple(tokens),
        opponent_targets=np.stack(targets),
    )


def _assert_input_isolation(
    mode_a: MemoryCurriculumArm,
    mode_b: MemoryCurriculumArm,
    *,
    cover_games: int,
) -> None:
    if len(mode_a.tokens) != len(mode_b.tokens):
        raise RuntimeError("mode arms have different recurrent lengths")
    action_slots = np.asarray(
        [
            name.startswith("previous_drop_action_")
            or name.startswith("previous_check_action_")
            for name in OBSERVATION_FEATURES
        ],
        dtype=np.bool_,
    )
    saw_public_cue = False
    for left, right in zip(mode_a.tokens, mode_b.tokens, strict=True):
        if (
            left.role_is_dropper != right.role_is_dropper
            or not np.array_equal(left.stage_matrix, right.stage_matrix)
            or not np.array_equal(left.exact_policy, right.exact_policy)
            or not np.array_equal(left.legal_mask, right.legal_mask)
        ):
            raise RuntimeError("mode arms differ in mechanical network inputs")
        differences = left.features != right.features
        if np.any(differences & ~action_slots):
            raise RuntimeError(
                "latent mode leaked outside earlier revealed-action slots"
            )
        saw_public_cue = saw_public_cue or bool(np.any(differences))
    if not saw_public_cue:
        raise RuntimeError("mode arms contain no identifiable public cue")
    if not all(
        left.bitwise_equal(right)
        for left, right in zip(
            mode_a.tokens[-cover_games:], mode_b.tokens[-cover_games:], strict=True
        )
    ):
        raise RuntimeError(
            "common cover suffix failed to erase current input differences"
        )


@dataclass(frozen=True, slots=True)
class MemoryCurriculumCase:
    """A causal matched pair with an identical target and conflicting labels."""

    split: SplitName
    example_seed: int
    role: Role
    parameters: MemoryCurriculumParameters
    mode_a: MemoryCurriculumArm
    mode_b: MemoryCurriculumArm
    best_responses: ExactBestResponseAudit
    target_sha256: str
    schema_version: str = MEMORY_CURRICULUM_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != MEMORY_CURRICULUM_SCHEMA:
            raise ValueError("unsupported memory curriculum case schema")
        if self.split not in {"train", "validation"} or self.role not in _ROLES:
            raise ValueError("curriculum case has an invalid split or role")
        if (
            self.parameters.split != self.split
            or self.parameters.example_seed != self.example_seed
        ):
            raise ValueError("curriculum case metadata disagrees with its parameters")
        if self.mode_a.mode != "a" or self.mode_b.mode != "b":
            raise ValueError("curriculum arms are not in canonical a/b order")
        if (
            self.mode_a.cue_actions != self.parameters.cue_actions_a
            or self.mode_b.cue_actions != self.parameters.cue_actions_b
        ):
            raise ValueError("curriculum arms disagree with their paired cue schedule")
        expected_length = len(self.mode_a.cue_actions) + self.parameters.cover_games + 1
        if (
            len(self.mode_a.tokens) != expected_length
            or len(self.mode_b.tokens) != expected_length
        ):
            raise ValueError("curriculum arm length disagrees with its cover schedule")
        _assert_input_isolation(
            self.mode_a, self.mode_b, cover_games=self.parameters.cover_games
        )
        if not self.mode_a.target.bitwise_equal(self.mode_b.target):
            raise ValueError(
                "current target tensors must be byte-identical across modes"
            )
        if self.mode_a.target.sha256() != self.target_sha256:
            raise ValueError("curriculum target digest does not match its tensors")
        expected_targets = mode_target_distributions()
        if not np.array_equal(
            self.mode_a.target_truth, expected_targets["a"]
        ) or not np.array_equal(self.mode_b.target_truth, expected_targets["b"]):
            raise ValueError(
                "curriculum target truths do not match the frozen mode contract"
            )
        expected_audit = audit_exact_best_responses(
            self.mode_a.target.stage_matrix, expected_targets
        )
        if self.best_responses != expected_audit:
            raise ValueError(
                "curriculum best-response audit does not match target matrix"
            )
        target_role = "dropper" if self.target.role_is_dropper else "checker"
        if target_role != self.role:
            raise ValueError("curriculum target role disagrees with case role")
        for mode in _MODES:
            self.best_responses.action(self.role, mode)

    @property
    def target(self) -> MemoryCurriculumToken:
        return self.mode_a.target

    @property
    def target_index(self) -> int:
        return len(self.mode_a.tokens) - 1

    def to_batch(self, *, device: torch.device | str = "cpu") -> MemoryCurriculumBatch:
        return memory_curriculum_batch(self, device=device)


def build_memory_curriculum_case(
    exact_agent: StageGameProvider,
    *,
    split: SplitName | MemoryCurriculumSplit,
    example_seed: int,
    role: Role,
) -> MemoryCurriculumCase:
    """Build one deterministic legal paired-history curriculum case."""

    spec = memory_curriculum_split(split) if isinstance(split, str) else split
    parameters = select_curriculum_parameters(
        spec, example_seed=example_seed, role=role
    )
    targets = mode_target_distributions()
    mode_a = _collect_arm(
        exact_agent,
        mode="a",
        role=role,
        cue_actions=parameters.cue_actions_a,
        parameters=parameters,
        target_truth=targets["a"],
    )
    mode_b = _collect_arm(
        exact_agent,
        mode="b",
        role=role,
        cue_actions=parameters.cue_actions_b,
        parameters=parameters,
        target_truth=targets["b"],
    )
    _assert_input_isolation(mode_a, mode_b, cover_games=parameters.cover_games)
    if not mode_a.target.bitwise_equal(mode_b.target):
        raise RuntimeError("latent modes produced different current target tensors")
    audit = audit_exact_best_responses(mode_a.target.stage_matrix, targets)
    return MemoryCurriculumCase(
        split=spec.name,
        example_seed=int(example_seed),
        role=role,
        parameters=parameters,
        mode_a=mode_a,
        mode_b=mode_b,
        best_responses=audit,
        target_sha256=mode_a.target.sha256(),
    )


def build_memory_curriculum_role_pair(
    exact_agent: StageGameProvider,
    *,
    split: SplitName | MemoryCurriculumSplit,
    example_seed: int,
) -> tuple[MemoryCurriculumCase, MemoryCurriculumCase]:
    """Build the predeclared Dropper and Checker cases for one identity seed."""

    return tuple(
        build_memory_curriculum_case(
            exact_agent, split=split, example_seed=example_seed, role=role
        )
        for role in _ROLES
    )  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class MemoryCurriculumBatch:
    """Two-arm tensor batch directly consumable by ``AggroHalNetwork``."""

    features: Tensor
    stage_matrices: Tensor
    exact_policies: Tensor
    role_is_dropper: Tensor
    legal_masks: Tensor
    opponent_targets: Tensor
    target_mask: Tensor
    best_response_actions: Tensor
    mode_indices: Tensor
    schema_version: str = MEMORY_CURRICULUM_BATCH_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != MEMORY_CURRICULUM_BATCH_SCHEMA:
            raise ValueError("unsupported memory curriculum batch schema")
        if self.features.ndim != 3 or self.features.shape[-1] != OBSERVATION_DIM:
            raise ValueError(
                "batch features must have shape [2, time, observation_dim]"
            )
        batch, time, _ = self.features.shape
        if batch != 2 or time <= 0:
            raise ValueError("a memory curriculum batch must contain two nonempty arms")
        expected = {
            "stage_matrices": (batch, time, ACTION_COUNT, ACTION_COUNT),
            "exact_policies": (batch, time, ACTION_COUNT),
            "role_is_dropper": (batch, time),
            "legal_masks": (batch, time, ACTION_COUNT),
            "opponent_targets": (batch, time, ACTION_COUNT),
            "target_mask": (batch, time),
            "best_response_actions": (batch,),
            "mode_indices": (batch,),
        }
        for label, shape in expected.items():
            if tuple(getattr(self, label).shape) != shape:
                raise ValueError(f"batch {label} must have shape {shape}")
        tensors = (
            self.features,
            self.stage_matrices,
            self.exact_policies,
            self.role_is_dropper,
            self.legal_masks,
            self.opponent_targets,
            self.target_mask,
            self.best_response_actions,
            self.mode_indices,
        )
        if any(tensor.device != self.features.device for tensor in tensors):
            raise ValueError("all curriculum batch tensors must share one device")
        expected_dtypes = {
            "features": torch.float32,
            "stage_matrices": torch.float32,
            "exact_policies": torch.float32,
            "role_is_dropper": torch.bool,
            "legal_masks": torch.bool,
            "opponent_targets": torch.float32,
            "target_mask": torch.bool,
            "best_response_actions": torch.long,
            "mode_indices": torch.long,
        }
        for label, dtype in expected_dtypes.items():
            if getattr(self, label).dtype != dtype:
                raise ValueError(f"batch {label} must use dtype {dtype}")
        for label in (
            "features",
            "stage_matrices",
            "exact_policies",
            "opponent_targets",
        ):
            if not torch.all(torch.isfinite(getattr(self, label))):
                raise ValueError(f"batch {label} must be finite")
        if not torch.all(self.legal_masks.any(dim=-1)):
            raise ValueError("every curriculum token must admit a legal action")
        if not torch.allclose(
            self.exact_policies.sum(dim=-1),
            torch.ones((batch, time), device=self.features.device),
            atol=1e-5,
        ) or not torch.allclose(
            self.opponent_targets.sum(dim=-1),
            torch.ones((batch, time), device=self.features.device),
            atol=1e-5,
        ):
            raise ValueError("batch policy and truth rows must sum to one")
        expected_target_mask = torch.zeros_like(self.target_mask)
        expected_target_mask[:, -1] = True
        if not torch.equal(self.target_mask, expected_target_mask):
            raise ValueError("each curriculum arm must mark only its final target step")
        if not torch.equal(self.mode_indices.cpu(), torch.tensor([0, 1])):
            raise ValueError("curriculum batch mode order must be a then b")
        if torch.any(self.best_response_actions < 0) or torch.any(
            self.best_response_actions >= ACTION_COUNT
        ):
            raise ValueError("best response targets must be zero-based action indices")

    def network_inputs(self) -> dict[str, Tensor]:
        """Return keyword arguments accepted verbatim by ``AggroHalNetwork``."""

        return {
            "features": self.features,
            "stage_matrices": self.stage_matrices,
            "exact_policies": self.exact_policies,
            "role_is_dropper": self.role_is_dropper,
            "legal_masks": self.legal_masks,
        }


def memory_curriculum_batch(
    case: MemoryCurriculumCase,
    *,
    device: torch.device | str = "cpu",
) -> MemoryCurriculumBatch:
    """Convert one paired case into a target-masked two-sequence tensor batch."""

    arms = (case.mode_a, case.mode_b)
    features = np.stack([[token.features for token in arm.tokens] for arm in arms])
    matrices = np.stack([[token.stage_matrix for token in arm.tokens] for arm in arms])
    exact = np.stack([[token.exact_policy for token in arm.tokens] for arm in arms])
    roles = np.asarray(
        [[token.role_is_dropper for token in arm.tokens] for arm in arms],
        dtype=np.bool_,
    )
    legal = np.stack([[token.legal_mask for token in arm.tokens] for arm in arms])
    opponent_targets = np.stack([arm.opponent_targets for arm in arms])
    target_mask = np.zeros((2, len(case.mode_a.tokens)), dtype=np.bool_)
    target_mask[:, -1] = True
    best_actions = np.asarray(
        [
            case.best_responses.action(case.role, "a") - 1,
            case.best_responses.action(case.role, "b") - 1,
        ],
        dtype=np.int64,
    )

    def tensor(values: np.ndarray, dtype: torch.dtype) -> Tensor:
        return torch.as_tensor(values, dtype=dtype, device=device)

    return MemoryCurriculumBatch(
        features=tensor(features, torch.float32),
        stage_matrices=tensor(matrices, torch.float32),
        exact_policies=tensor(exact, torch.float32),
        role_is_dropper=tensor(roles, torch.bool),
        legal_masks=tensor(legal, torch.bool),
        opponent_targets=tensor(opponent_targets, torch.float32),
        target_mask=tensor(target_mask, torch.bool),
        best_response_actions=tensor(best_actions, torch.long),
        mode_indices=torch.tensor([0, 1], dtype=torch.long, device=device),
    )
