"""CPU-safe training for the direct recurrent Aggro Hal policy.

The trainer has two deliberately small phases:

1. A simulator warm start uses truth-independent exploratory play. After each
   action, privileged targets teach the GRU the opponent distribution and
   maximize expected payoff in the exact continuation-adjusted DTH matrix.
2. Recurrent PPO fine-tunes complete repeated-game sessions using only public
   observations at policy time. Prediction and tactical expected-payoff losses
   remain auxiliary objectives so sparse terminal rewards do not erase useful
   one-step structure.

Generated checkpoints and reports belong under the gitignored ``outputs/``
tree.  The default device is always CPU; CUDA is used only when a caller
explicitly requests it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
import yaml
from torch import Tensor

from arena.contracts import CanonicalPolicyProvider
from arena.policies.aggro_env import (
    ACTION_COUNT,
    AggroSessionEnv,
)
from arena.policies.aggro_hal import (
    OBSERVATION_DIM,
    AggroHalConfig,
    AggroHalNetwork,
    dth_compatibility,
    load_checkpoint,
    save_checkpoint,
)
from arena.policies.aggro_memory_curriculum import (
    MemoryCurriculumArm,
    MemoryCurriculumSplit,
    build_memory_curriculum_role_pair,
    memory_curriculum_config_payload,
    memory_curriculum_config_sha256,
    memory_curriculum_split,
)
from arena.policies.opponent_league import (
    TRAIN_FAMILY_MANIFEST,
    OpponentFamilyManifest,
    make_opponent,
)
from dth.agent import CompleteDTHAgent

TRAINING_CONFIG_SCHEMA = "arena-aggro-hal-training-config-v2"
TRAINING_REPORT_SCHEMA = "arena-aggro-hal-training-report-v2"
TRAINING_OBJECTIVE_SCHEMA = "arena-aggro-hal-tactical-objective-v2"
SESSION_COLLECTOR_BINDING_SCHEMA = "arena-aggro-hal-session-collector-binding-v1"
SESSION_COLLECTOR_CONFIG_SCHEMA = "arena-aggro-hal-session-collector-config-v1"
ADAPTIVE_EXPERIMENT_SCHEMA = "arena-aggro-hal-adaptive-memory-experiment-v1"
ADAPTIVE_EXPERIMENT_BINDING_SCHEMA = (
    "arena-aggro-hal-adaptive-memory-experiment-binding-v1"
)
ADAPTIVE_GOAL_SCHEMA = "arena-aggro-hal-adaptive-exploitation-goal-v1"
DEFAULT_SESSION_COLLECTOR_IDENTITY = "arena.opponent-league.train-v1"
MEMORY_SESSION_COLLECTOR_IDENTITY = "arena.memory-necessity.train-v1"

WARMSTART_EXACT_BEHAVIOR_WEIGHT = 0.75
_RESUME_MUTABLE_TRAINER_FIELDS = frozenset(
    {"warmstart_updates", "ppo_updates", "device", "cpu_threads"}
)


@dataclass(frozen=True, slots=True)
class AggroTrainerConfig:
    """Validated optimizer, rollout, and objective settings."""

    seed: int = 20260808
    device: str = "cpu"
    cpu_threads: int = 6
    dth_artifact: str = "src/dth/artifacts/complete_full_v1"
    games_per_session: int = 8
    max_half_rounds: int = 24
    start_clocks: tuple[int, ...] = (720,)
    sessions_per_update: int = 8
    warmstart_updates: int = 40
    ppo_updates: int = 20
    ppo_epochs: int = 3
    learning_rate: float = 8e-4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    ppo_clip: float = 0.20
    value_coef: float = 0.25
    entropy_coef: float = 0.003
    prediction_coef: float = 1.0
    ppo_prediction_coef: float = 0.25
    tactical_coef: float = 1.0
    direct_tactical_coef: float = 0.5
    ppo_tactical_coef: float = 0.15
    max_grad_norm: float = 1.0
    snapshot_interval: int = 10

    def __post_init__(self) -> None:
        if self.seed < 0 or self.cpu_threads <= 0:
            raise ValueError("seed must be nonnegative and cpu_threads positive")
        try:
            device = torch.device(self.device)
        except (RuntimeError, ValueError) as error:
            raise ValueError(f"invalid torch device {self.device!r}") from error
        if device.type not in {"cpu", "cuda"}:
            raise ValueError(
                "Aggro Hal training supports only explicit cpu or cuda devices"
            )
        positive_ints = {
            "games_per_session": self.games_per_session,
            "max_half_rounds": self.max_half_rounds,
            "sessions_per_update": self.sessions_per_update,
            "ppo_epochs": self.ppo_epochs,
        }
        if any(
            isinstance(value, bool) or value <= 0 for value in positive_ints.values()
        ):
            raise ValueError(f"positive integer settings required: {positive_ints}")
        if self.warmstart_updates < 0 or self.ppo_updates < 0:
            raise ValueError("training update counts must be nonnegative")
        if self.warmstart_updates + self.ppo_updates <= 0:
            raise ValueError("at least one warm-start or PPO update is required")
        if not self.start_clocks or any(clock < 0 for clock in self.start_clocks):
            raise ValueError("start_clocks must contain nonnegative clocks")
        unit_interval = {
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "ppo_clip": self.ppo_clip,
        }
        if not 0.0 < self.gamma <= 1.0 or not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError(f"invalid discount settings: {unit_interval}")
        if not 0.0 < self.ppo_clip < 1.0:
            raise ValueError("ppo_clip must lie in (0, 1)")
        nonnegative = (
            self.value_coef,
            self.entropy_coef,
            self.prediction_coef,
            self.ppo_prediction_coef,
            self.tactical_coef,
            self.direct_tactical_coef,
            self.ppo_tactical_coef,
        )
        if any(not np.isfinite(value) or value < 0.0 for value in nonnegative):
            raise ValueError("loss coefficients must be finite and nonnegative")
        if self.learning_rate <= 0.0 or self.max_grad_norm <= 0.0:
            raise ValueError("learning_rate and max_grad_norm must be positive")
        if self.snapshot_interval < 0:
            raise ValueError("snapshot_interval must be nonnegative")


@dataclass(slots=True)
class TrainingSequence:
    """One complete repeated-opponent session."""

    features: np.ndarray
    stage_matrices: np.ndarray
    exact_policies: np.ndarray
    roles_are_dropper: np.ndarray
    legal_masks: np.ndarray
    opponent_targets: np.ndarray
    opponent_actions: np.ndarray
    learner_actions: np.ndarray
    rewards: np.ndarray
    returns: np.ndarray
    old_log_probabilities: np.ndarray | None = None
    old_values: np.ndarray | None = None
    advantages: np.ndarray | None = None
    objective_weights: np.ndarray | None = None

    @property
    def length(self) -> int:
        return int(self.features.shape[0])


@dataclass(frozen=True, slots=True)
class PaddedBatch:
    features: Tensor
    stage_matrices: Tensor
    exact_policies: Tensor
    roles_are_dropper: Tensor
    legal_masks: Tensor
    opponent_targets: Tensor
    opponent_actions: Tensor
    learner_actions: Tensor
    returns: Tensor
    valid_steps: Tensor
    objective_weights: Tensor
    old_log_probabilities: Tensor | None
    advantages: Tensor | None


class TrainingSessionCollector(Protocol):
    """Bounded, checkpoint-identifiable source of complete training sessions."""

    def checkpoint_binding(self) -> Mapping[str, object]:
        """Return a stable JSON-compatible identity and configuration."""

        ...

    def collect_update(
        self,
        *,
        update_index: int,
        exact_agent: CompleteDTHAgent,
        model: AggroHalNetwork,
        trainer: AggroTrainerConfig,
        on_policy: bool,
        device: torch.device,
    ) -> Sequence[TrainingSequence]:
        """Return exactly ``trainer.sessions_per_update`` complete sessions."""

        ...


OpponentFactory = Callable[[str, int], CanonicalPolicyProvider]


def _json_write(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha256(value: object) -> str:
    canonical = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _mapping(raw: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(raw, dict):
        raise ValueError(f"{label} must be a mapping")
    return raw


def _canonical_session_collector_binding(
    raw: object,
    *,
    label: str = "session collector binding",
) -> dict[str, object]:
    """Validate and copy the checkpoint-visible collector contract."""

    if not isinstance(raw, Mapping):
        raise ValueError(f"{label} must be a mapping")
    binding = dict(raw)
    required_keys = {"schema_version", "identity", "config"}
    if set(binding) != required_keys:
        raise ValueError(f"{label} fields must be exactly {sorted(required_keys)!r}")
    if binding.get("schema_version") != SESSION_COLLECTOR_BINDING_SCHEMA:
        raise ValueError(f"{label} has an unsupported schema")
    identity = binding.get("identity")
    if not isinstance(identity, str) or not identity.strip():
        raise ValueError(f"{label} identity must be a nonempty string")
    raw_config = binding.get("config")
    if not isinstance(raw_config, Mapping):
        raise ValueError(f"{label} config must be a mapping")
    config = dict(raw_config)
    try:
        normalized_config = json.loads(
            json.dumps(config, allow_nan=False, sort_keys=True, separators=(",", ":"))
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} config must be finite JSON data") from error
    if not isinstance(normalized_config, dict):
        raise ValueError(f"{label} config must be a JSON object")
    return {
        "schema_version": SESSION_COLLECTOR_BINDING_SCHEMA,
        "identity": identity,
        "config": normalized_config,
    }


def _default_session_collector_binding() -> dict[str, object]:
    return _canonical_session_collector_binding(
        {
            "schema_version": SESSION_COLLECTOR_BINDING_SCHEMA,
            "identity": DEFAULT_SESSION_COLLECTOR_IDENTITY,
            "config": {
                "factory": "arena.policies.opponent_league.make_opponent",
                "manifest": {
                    "schema_version": TRAIN_FAMILY_MANIFEST.schema_version,
                    "split": TRAIN_FAMILY_MANIFEST.split,
                    "entries": [
                        {"family": entry.family, "seeds": list(entry.seeds)}
                        for entry in TRAIN_FAMILY_MANIFEST.entries
                    ],
                },
            },
        }
    )


def _validate_resume_session_collector(
    stored: object,
    current: Mapping[str, object],
) -> None:
    """Bind resume to the same session source and curriculum configuration."""

    current_binding = _canonical_session_collector_binding(current)
    stored_binding = (
        _default_session_collector_binding()
        if stored is None
        else _canonical_session_collector_binding(
            stored, label="checkpoint session collector binding"
        )
    )
    if stored_binding != current_binding:
        raise ValueError(
            "resume session collector is incompatible: "
            f"checkpoint={stored_binding!r}, current={current_binding!r}"
        )


def _model_config(raw: Mapping[str, object]) -> AggroHalConfig:
    return AggroHalConfig(
        hidden_size=int(raw.get("hidden_size", 128)),
        gru_layers=int(raw.get("gru_layers", 2)),
        head_hidden_size=int(raw.get("head_hidden_size", 96)),
        gru_dropout=float(raw.get("gru_dropout", 0.0)),
        tactical_logit_scale=float(raw.get("tactical_logit_scale", 12.0)),
    )


def _trainer_config(raw: Mapping[str, object]) -> AggroTrainerConfig:
    return AggroTrainerConfig(
        seed=int(raw.get("seed", 20260808)),
        device=str(raw.get("device", "cpu")),
        cpu_threads=int(raw.get("cpu_threads", 6)),
        dth_artifact=str(raw.get("dth_artifact", "src/dth/artifacts/complete_full_v1")),
        games_per_session=int(raw.get("games_per_session", 8)),
        max_half_rounds=int(raw.get("max_half_rounds", 24)),
        start_clocks=tuple(int(value) for value in raw.get("start_clocks", [720])),
        sessions_per_update=int(raw.get("sessions_per_update", 8)),
        warmstart_updates=int(raw.get("warmstart_updates", 40)),
        ppo_updates=int(raw.get("ppo_updates", 20)),
        ppo_epochs=int(raw.get("ppo_epochs", 3)),
        learning_rate=float(raw.get("learning_rate", 8e-4)),
        gamma=float(raw.get("gamma", 0.995)),
        gae_lambda=float(raw.get("gae_lambda", 0.95)),
        ppo_clip=float(raw.get("ppo_clip", 0.20)),
        value_coef=float(raw.get("value_coef", 0.25)),
        entropy_coef=float(raw.get("entropy_coef", 0.003)),
        prediction_coef=float(raw.get("prediction_coef", 1.0)),
        ppo_prediction_coef=float(raw.get("ppo_prediction_coef", 0.25)),
        tactical_coef=float(raw.get("tactical_coef", 1.0)),
        direct_tactical_coef=float(raw.get("direct_tactical_coef", 0.5)),
        ppo_tactical_coef=float(raw.get("ppo_tactical_coef", 0.15)),
        max_grad_norm=float(raw.get("max_grad_norm", 1.0)),
        snapshot_interval=int(raw.get("snapshot_interval", 10)),
    )


def _load_training_root(path: str | Path) -> Mapping[str, object]:
    source = Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    root = _mapping(payload, label="training config")
    if root.get("schema_version") != TRAINING_CONFIG_SCHEMA:
        raise ValueError("unsupported Aggro Hal training config schema")
    return root


def load_training_config(path: str | Path) -> tuple[AggroHalConfig, AggroTrainerConfig]:
    root = _load_training_root(path)
    return (
        _model_config(_mapping(root.get("model"), label="model config")),
        _trainer_config(_mapping(root.get("training"), label="training config")),
    )


def _configured_experiment_binding(
    config_path: str | Path,
) -> dict[str, object] | None:
    """Validate and bind an optional adaptive experiment to its frozen goal."""

    root = _load_training_root(config_path)
    raw_experiment = root.get("experiment")
    if raw_experiment is None:
        return None
    experiment = _mapping(raw_experiment, label="adaptive experiment")
    required = {
        "schema_version",
        "goal_manifest",
        "goal_manifest_canonical_json_sha256",
        "tactical_baseline",
        "initial_checkpoint",
        "generated_output",
        "phase",
        "ppo_locked",
    }
    if set(experiment) != required:
        raise ValueError(
            f"adaptive experiment fields must be exactly {sorted(required)!r}"
        )
    if experiment.get("schema_version") != ADAPTIVE_EXPERIMENT_SCHEMA:
        raise ValueError("unsupported adaptive experiment schema")

    raw_goal_manifest = experiment.get("goal_manifest")
    raw_goal_sha256 = experiment.get("goal_manifest_canonical_json_sha256")
    raw_checkpoint = experiment.get("initial_checkpoint")
    raw_baseline = experiment.get("tactical_baseline")
    raw_output = experiment.get("generated_output")
    if not isinstance(raw_goal_manifest, str) or not raw_goal_manifest:
        raise ValueError("experiment goal_manifest must be a nonempty path")
    if (
        not isinstance(raw_goal_sha256, str)
        or len(raw_goal_sha256) != 64
        or any(character not in "0123456789abcdef" for character in raw_goal_sha256)
    ):
        raise ValueError("experiment goal manifest SHA-256 is malformed")
    if not isinstance(raw_checkpoint, str) or not raw_checkpoint:
        raise ValueError("experiment initial_checkpoint must be a nonempty path")
    if not isinstance(raw_baseline, str) or not raw_baseline:
        raise ValueError(
            "a configured initial_checkpoint requires a tactical_baseline manifest"
        )
    if not isinstance(raw_output, str) or not raw_output:
        raise ValueError("experiment generated_output must be a nonempty path")
    if (
        experiment.get("phase") != "warmstart-only"
        or experiment.get("ppo_locked") is not True
    ):
        raise ValueError(
            "adaptive memory experiment must remain warmstart-only with PPO locked"
        )
    for label, value in (
        ("experiment goal_manifest", raw_goal_manifest),
        ("experiment initial_checkpoint", raw_checkpoint),
        ("experiment tactical_baseline", raw_baseline),
        ("experiment generated_output", raw_output),
    ):
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"{label} must be a repository-relative nonescaping path")

    goal_payload = json.loads(Path(raw_goal_manifest).read_text(encoding="utf-8"))
    goal = _mapping(goal_payload, label="adaptive goal manifest")
    if goal.get("schema_version") != ADAPTIVE_GOAL_SCHEMA:
        raise ValueError("configured adaptive goal manifest has an unsupported schema")
    actual_goal_sha256 = _canonical_json_sha256(goal)
    if actual_goal_sha256 != raw_goal_sha256:
        raise ValueError("configured adaptive goal manifest bytes have changed")

    baseline_payload = json.loads(Path(raw_baseline).read_text(encoding="utf-8"))
    baseline = _mapping(baseline_payload, label="tactical baseline manifest")
    if (
        baseline.get("schema_version")
        != ("arena-aggro-hal-tactical-baseline-freeze-v1")
        or baseline.get("immutable") is not True
    ):
        raise ValueError("configured tactical baseline is not an immutable v1 freeze")
    artifact = _mapping(baseline.get("artifact"), label="tactical baseline artifact")
    artifact_path = artifact.get("path")
    artifact_sha256 = artifact.get("sha256")
    if artifact_path != raw_checkpoint:
        raise ValueError(
            "experiment initial_checkpoint disagrees with the tactical baseline "
            "artifact"
        )
    if (
        not isinstance(artifact_sha256, str)
        or len(artifact_sha256) != 64
        or any(character not in "0123456789abcdef" for character in artifact_sha256)
    ):
        raise ValueError("tactical baseline checkpoint SHA-256 is malformed")
    return {
        "schema_version": ADAPTIVE_EXPERIMENT_BINDING_SCHEMA,
        "goal_manifest": {
            "path": raw_goal_manifest,
            "canonical_json_sha256": actual_goal_sha256,
        },
        "tactical_baseline": raw_baseline,
        "initial_checkpoint": {
            "path": raw_checkpoint,
            "sha256": artifact_sha256,
        },
        "generated_output": raw_output,
        "phase": "warmstart-only",
        "ppo_locked": True,
    }


def _configured_initial_checkpoint_binding(
    config_path: str | Path,
) -> dict[str, str] | None:
    """Resolve the tactical initialization bound by an adaptive experiment."""

    experiment = _configured_experiment_binding(config_path)
    if experiment is None:
        return None
    initial = _mapping(
        experiment.get("initial_checkpoint"),
        label="adaptive experiment initial checkpoint binding",
    )
    return {"path": str(initial["path"]), "sha256": str(initial["sha256"])}


def _validate_resume_trainer_config(
    stored: Mapping[str, object],
    current: AggroTrainerConfig,
) -> None:
    """Reject resumed runs whose data distribution or objective has changed."""

    current_values = asdict(current)
    mismatches: list[str] = []
    for field, current_value in current_values.items():
        if field in _RESUME_MUTABLE_TRAINER_FIELDS:
            continue
        if field not in stored:
            mismatches.append(f"{field} is missing")
            continue
        stored_value = stored[field]
        if field == "start_clocks":
            stored_value = (
                tuple(stored_value)
                if isinstance(stored_value, (list, tuple))
                else stored_value
            )
        if stored_value != current_value:
            mismatches.append(
                f"{field}: checkpoint={stored_value!r}, current={current_value!r}"
            )
    if mismatches:
        raise ValueError(
            "resume trainer configuration is incompatible: " + "; ".join(mismatches)
        )


def discounted_returns(rewards: np.ndarray, *, gamma: float) -> np.ndarray:
    result = np.zeros_like(np.asarray(rewards, dtype=np.float32))
    running = 0.0
    for index in range(len(result) - 1, -1, -1):
        running = float(rewards[index]) + gamma * running
        result[index] = running
    return result


def generalized_advantages(
    rewards: np.ndarray,
    values: np.ndarray,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute GAE across one whole repeated-game session."""

    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    if rewards.shape != values.shape or rewards.ndim != 1:
        raise ValueError("rewards and values must be equal one-dimensional arrays")
    advantages = np.zeros_like(rewards)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        next_value = 0.0 if index + 1 == len(values) else float(values[index + 1])
        delta = float(rewards[index]) + gamma * next_value - float(values[index])
        running = delta + gamma * gae_lambda * running
        advantages[index] = running
    return advantages, advantages + values


def _manifest_variants(manifest: OpponentFamilyManifest) -> tuple[tuple[str, int], ...]:
    return tuple(
        (entry.family, seed) for entry in manifest.entries for seed in entry.seeds
    )


def _default_opponent_factory(family: str, seed: int) -> CanonicalPolicyProvider:
    return make_opponent(family, seed=seed)


def _sequence_from_lists(
    *,
    features: list[np.ndarray],
    matrices: list[np.ndarray],
    exact: list[np.ndarray],
    roles: list[bool],
    legal: list[np.ndarray],
    opponent_targets: list[np.ndarray],
    opponent_actions: list[int],
    learner_actions: list[int],
    rewards: list[float],
    gamma: float,
    old_log_probabilities: list[float] | None = None,
    old_values: list[float] | None = None,
    gae_lambda: float | None = None,
) -> TrainingSequence:
    rewards_array = np.asarray(rewards, dtype=np.float32)
    values_array = (
        None if old_values is None else np.asarray(old_values, dtype=np.float32)
    )
    if values_array is not None:
        if gae_lambda is None:
            raise ValueError("GAE lambda is required with rollout values")
        advantages, returns = generalized_advantages(
            rewards_array, values_array, gamma=gamma, gae_lambda=gae_lambda
        )
    else:
        advantages = None
        returns = discounted_returns(rewards_array, gamma=gamma)
    return TrainingSequence(
        features=np.stack(features).astype(np.float32),
        stage_matrices=np.stack(matrices).astype(np.float32),
        exact_policies=np.stack(exact).astype(np.float32),
        roles_are_dropper=np.asarray(roles, dtype=np.bool_),
        legal_masks=np.stack(legal).astype(np.bool_),
        opponent_targets=np.stack(opponent_targets).astype(np.float32),
        opponent_actions=np.asarray(opponent_actions, dtype=np.int64),
        learner_actions=np.asarray(learner_actions, dtype=np.int64),
        rewards=rewards_array,
        returns=returns,
        old_log_probabilities=(
            None
            if old_log_probabilities is None
            else np.asarray(old_log_probabilities, dtype=np.float32)
        ),
        old_values=values_array,
        advantages=advantages,
    )


def collect_teacher_session(
    *,
    exact_agent: CompleteDTHAgent,
    trainer: AggroTrainerConfig,
    family: str,
    opponent_seed: int,
    session_seed: int,
    learner_starts_in_hal_seat: bool,
    opponent_factory: OpponentFactory = _default_opponent_factory,
) -> TrainingSequence:
    """Collect one privileged analytic-teacher session."""

    opponent = opponent_factory(family, opponent_seed)
    env = AggroSessionEnv(
        opponent,
        exact_agent,
        games_per_session=trainer.games_per_session,
        seed=session_seed,
        start_clocks=trainer.start_clocks,
        max_half_rounds=trainer.max_half_rounds,
        learner_starts_in_hal_seat=learner_starts_in_hal_seat,
    )
    rng = np.random.default_rng(session_seed + 17)
    decision = env.reset()
    features: list[np.ndarray] = []
    matrices: list[np.ndarray] = []
    exact: list[np.ndarray] = []
    roles: list[bool] = []
    legal: list[np.ndarray] = []
    opponent_targets: list[np.ndarray] = []
    opponent_actions: list[int] = []
    learner_actions: list[int] = []
    rewards: list[float] = []
    try:
        while True:
            legal_mask = np.asarray(
                [
                    second in decision.canonical_decision.legal_seconds
                    for second in range(1, ACTION_COUNT + 1)
                ],
                dtype=np.bool_,
            )
            uniform_legal = legal_mask.astype(np.float64) / float(legal_mask.sum())
            behavior_policy = (
                WARMSTART_EXACT_BEHAVIOR_WEIGHT
                * np.asarray(decision.exact_policy, dtype=np.float64)
                + (1.0 - WARMSTART_EXACT_BEHAVIOR_WEIGHT) * uniform_legal
            )
            behavior_policy *= legal_mask
            behavior_policy /= float(behavior_policy.sum())
            action = int(rng.choice(np.arange(1, ACTION_COUNT + 1), p=behavior_policy))
            step = env.step(action)
            truth = step.record.opponent_true_distribution
            if truth is None:
                raise RuntimeError(
                    "training opponent did not expose its simulator truth"
                )
            features.append(np.asarray(decision.observation, dtype=np.float32))
            matrices.append(np.asarray(decision.stage_matrix))
            exact.append(np.asarray(decision.exact_policy))
            roles.append(decision.role == "dropper")
            legal.append(legal_mask)
            opponent_targets.append(np.asarray(truth))
            opponent_actions.append(step.record.opponent_action - 1)
            learner_actions.append(action - 1)
            rewards.append(step.record.terminal_game_reward)
            if step.session_done:
                break
            if step.next_decision is None:
                raise RuntimeError("active teacher session has no next decision")
            decision = step.next_decision
    finally:
        env.close()
    return _sequence_from_lists(
        features=features,
        matrices=matrices,
        exact=exact,
        roles=roles,
        legal=legal,
        opponent_targets=opponent_targets,
        opponent_actions=opponent_actions,
        learner_actions=learner_actions,
        rewards=rewards,
        gamma=trainer.gamma,
    )


def collect_policy_session(
    *,
    model: AggroHalNetwork,
    exact_agent: CompleteDTHAgent,
    trainer: AggroTrainerConfig,
    family: str,
    opponent_seed: int,
    session_seed: int,
    learner_starts_in_hal_seat: bool,
    device: torch.device,
    opponent_factory: OpponentFactory = _default_opponent_factory,
) -> TrainingSequence:
    """Collect one on-policy recurrent rollout without retaining an autograd graph."""

    opponent = opponent_factory(family, opponent_seed)
    env = AggroSessionEnv(
        opponent,
        exact_agent,
        games_per_session=trainer.games_per_session,
        seed=session_seed,
        start_clocks=trainer.start_clocks,
        max_half_rounds=trainer.max_half_rounds,
        learner_starts_in_hal_seat=learner_starts_in_hal_seat,
    )
    rng = np.random.default_rng(session_seed + 29)
    decision = env.reset()
    hidden: Tensor | None = None
    features: list[np.ndarray] = []
    matrices: list[np.ndarray] = []
    exact: list[np.ndarray] = []
    roles: list[bool] = []
    legal: list[np.ndarray] = []
    opponent_targets: list[np.ndarray] = []
    opponent_actions: list[int] = []
    learner_actions: list[int] = []
    rewards: list[float] = []
    old_log_probabilities: list[float] = []
    old_values: list[float] = []
    model.eval()
    try:
        while True:
            encoded = np.array(decision.observation, dtype=np.float32, copy=True)
            legal_mask = np.asarray(
                [
                    second in decision.canonical_decision.legal_seconds
                    for second in range(1, 61)
                ],
                dtype=np.bool_,
            )
            with torch.inference_mode():
                output = model(
                    torch.as_tensor(encoded, dtype=torch.float32, device=device).view(
                        1, 1, -1
                    ),
                    torch.tensor(
                        decision.stage_matrix, dtype=torch.float32, device=device
                    ).view(1, 1, ACTION_COUNT, ACTION_COUNT),
                    torch.tensor(
                        decision.exact_policy, dtype=torch.float32, device=device
                    ).view(1, 1, ACTION_COUNT),
                    torch.tensor(
                        [[decision.role == "dropper"]], dtype=torch.bool, device=device
                    ),
                    torch.as_tensor(legal_mask, dtype=torch.bool, device=device).view(
                        1, 1, ACTION_COUNT
                    ),
                    hidden,
                )
            hidden = output.hidden_state.detach()
            policy = output.policy[0, 0].detach().cpu().numpy().astype(np.float64)
            action_index = int(rng.choice(ACTION_COUNT, p=policy / policy.sum()))
            features.append(encoded)
            matrices.append(np.asarray(decision.stage_matrix))
            exact.append(np.asarray(decision.exact_policy))
            roles.append(decision.role == "dropper")
            legal.append(legal_mask)
            learner_actions.append(action_index)
            old_log_probabilities.append(
                math.log(max(float(policy[action_index]), 1e-12))
            )
            old_values.append(float(output.value[0, 0].item()))
            step = env.step(action_index + 1)
            truth = step.record.opponent_true_distribution
            if truth is None:
                raise RuntimeError(
                    "training opponent did not expose its simulator truth"
                )
            opponent_targets.append(np.asarray(truth))
            opponent_actions.append(step.record.opponent_action - 1)
            rewards.append(step.record.terminal_game_reward)
            if step.session_done:
                break
            if step.next_decision is None:
                raise RuntimeError("active on-policy session has no next decision")
            decision = step.next_decision
    finally:
        env.close()
    return _sequence_from_lists(
        features=features,
        matrices=matrices,
        exact=exact,
        roles=roles,
        legal=legal,
        opponent_targets=opponent_targets,
        opponent_actions=opponent_actions,
        learner_actions=learner_actions,
        rewards=rewards,
        gamma=trainer.gamma,
        old_log_probabilities=old_log_probabilities,
        old_values=old_values,
        gae_lambda=trainer.gae_lambda,
    )


def pad_sequences(
    sequences: Sequence[TrainingSequence], *, device: torch.device
) -> PaddedBatch:
    if not sequences:
        raise ValueError("at least one training sequence is required")
    batch = len(sequences)
    time_steps = max(sequence.length for sequence in sequences)
    features = np.zeros((batch, time_steps, OBSERVATION_DIM), dtype=np.float32)
    matrices = np.zeros(
        (batch, time_steps, ACTION_COUNT, ACTION_COUNT), dtype=np.float32
    )
    exact = np.full(
        (batch, time_steps, ACTION_COUNT), 1.0 / ACTION_COUNT, dtype=np.float32
    )
    roles = np.zeros((batch, time_steps), dtype=np.bool_)
    legal = np.ones((batch, time_steps, ACTION_COUNT), dtype=np.bool_)
    opponent_targets = np.full(
        (batch, time_steps, ACTION_COUNT), 1.0 / ACTION_COUNT, dtype=np.float32
    )
    opponent_actions = np.zeros((batch, time_steps), dtype=np.int64)
    learner_actions = np.zeros((batch, time_steps), dtype=np.int64)
    returns = np.zeros((batch, time_steps), dtype=np.float32)
    valid = np.zeros((batch, time_steps), dtype=np.bool_)
    objective_weights = np.zeros((batch, time_steps), dtype=np.float32)
    has_ppo = all(sequence.advantages is not None for sequence in sequences)
    if any((sequence.advantages is not None) != has_ppo for sequence in sequences):
        raise ValueError("cannot mix teacher and PPO sequences in one batch")
    old_log_probabilities = (
        np.zeros((batch, time_steps), dtype=np.float32) if has_ppo else None
    )
    advantages = np.zeros((batch, time_steps), dtype=np.float32) if has_ppo else None
    for item, sequence in enumerate(sequences):
        length = sequence.length
        features[item, :length] = sequence.features
        matrices[item, :length] = sequence.stage_matrices
        exact[item, :length] = sequence.exact_policies
        roles[item, :length] = sequence.roles_are_dropper
        legal[item, :length] = sequence.legal_masks
        opponent_targets[item, :length] = sequence.opponent_targets
        opponent_actions[item, :length] = sequence.opponent_actions
        learner_actions[item, :length] = sequence.learner_actions
        returns[item, :length] = sequence.returns
        valid[item, :length] = True
        sequence_weights = (
            np.ones(length, dtype=np.float32)
            if sequence.objective_weights is None
            else np.asarray(sequence.objective_weights, dtype=np.float32)
        )
        if (
            sequence_weights.shape != (length,)
            or not np.all(np.isfinite(sequence_weights))
            or np.any(sequence_weights < 0.0)
            or float(sequence_weights.sum()) <= 0.0
        ):
            raise ValueError(
                "sequence objective_weights must be a finite nonnegative "
                "length-sized vector with positive total weight"
            )
        objective_weights[item, :length] = sequence_weights
        if has_ppo:
            assert old_log_probabilities is not None and advantages is not None
            assert sequence.old_log_probabilities is not None
            assert sequence.advantages is not None
            old_log_probabilities[item, :length] = sequence.old_log_probabilities
            advantages[item, :length] = sequence.advantages

    def tensor(values: np.ndarray, dtype: torch.dtype) -> Tensor:
        return torch.as_tensor(values, dtype=dtype, device=device)

    return PaddedBatch(
        features=tensor(features, torch.float32),
        stage_matrices=tensor(matrices, torch.float32),
        exact_policies=tensor(exact, torch.float32),
        roles_are_dropper=tensor(roles, torch.bool),
        legal_masks=tensor(legal, torch.bool),
        opponent_targets=tensor(opponent_targets, torch.float32),
        opponent_actions=tensor(opponent_actions, torch.long),
        learner_actions=tensor(learner_actions, torch.long),
        returns=tensor(returns, torch.float32),
        valid_steps=tensor(valid, torch.bool),
        objective_weights=tensor(objective_weights, torch.float32),
        old_log_probabilities=(
            None
            if old_log_probabilities is None
            else tensor(old_log_probabilities, torch.float32)
        ),
        advantages=None if advantages is None else tensor(advantages, torch.float32),
    )


def _weighted_mean(values: Tensor, weights: Tensor) -> Tensor:
    cast_weights = weights.to(values.dtype)
    denominator = cast_weights.sum().clamp_min(torch.finfo(values.dtype).eps)
    return (values * cast_weights).sum() / denominator


def _auxiliary_losses(
    output,
    batch: PaddedBatch,
) -> dict[str, Tensor]:
    tiny = torch.finfo(output.policy.dtype).tiny
    prediction = -(
        batch.opponent_targets * output.opponent_policy.clamp_min(tiny).log()
    ).sum(-1)
    target_column = batch.opponent_targets.unsqueeze(-1)
    dropper_values = torch.matmul(batch.stage_matrices, target_column).squeeze(-1)
    checker_values = -torch.matmul(
        batch.stage_matrices.transpose(-1, -2), target_column
    ).squeeze(-1)
    true_action_values = torch.where(
        batch.roles_are_dropper.unsqueeze(-1),
        dropper_values,
        checker_values,
    )
    tactical = -(output.policy * true_action_values).sum(-1)
    direct_tactical = -(output.direct_policy * true_action_values).sum(-1)
    value = (output.value - batch.returns).square()
    opponent_nll = (
        -output.opponent_policy.gather(-1, batch.opponent_actions.unsqueeze(-1))
        .squeeze(-1)
        .clamp_min(tiny)
        .log()
    )
    return {
        "prediction": _weighted_mean(prediction, batch.objective_weights),
        "tactical": _weighted_mean(tactical, batch.objective_weights),
        "direct_tactical": _weighted_mean(direct_tactical, batch.objective_weights),
        "value": _weighted_mean(value, batch.objective_weights),
        "opponent_nll": _weighted_mean(opponent_nll, batch.objective_weights),
        "direct_weight": _weighted_mean(output.direct_weight, batch.objective_weights),
    }


def warmstart_step(
    model: AggroHalNetwork,
    optimizer: torch.optim.Optimizer,
    batch: PaddedBatch,
    trainer: AggroTrainerConfig,
) -> dict[str, float]:
    model.train()
    output = model(
        batch.features,
        batch.stage_matrices,
        batch.exact_policies,
        batch.roles_are_dropper,
        batch.legal_masks,
    )
    losses = _auxiliary_losses(output, batch)
    total = (
        trainer.prediction_coef * losses["prediction"]
        + trainer.tactical_coef * losses["tactical"]
        + trainer.direct_tactical_coef * losses["direct_tactical"]
        + trainer.value_coef * losses["value"]
    )
    optimizer.zero_grad(set_to_none=True)
    total.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), trainer.max_grad_norm
    )
    optimizer.step()
    return {
        "total_loss": float(total.detach().cpu()),
        **{name: float(value.detach().cpu()) for name, value in losses.items()},
        "gradient_norm": float(torch.as_tensor(gradient_norm).detach().cpu()),
    }


def ppo_step(
    model: AggroHalNetwork,
    optimizer: torch.optim.Optimizer,
    batch: PaddedBatch,
    trainer: AggroTrainerConfig,
) -> dict[str, float]:
    if batch.old_log_probabilities is None or batch.advantages is None:
        raise ValueError("PPO update requires old probabilities and advantages")
    model.train()
    output = model(
        batch.features,
        batch.stage_matrices,
        batch.exact_policies,
        batch.roles_are_dropper,
        batch.legal_masks,
    )
    tiny = torch.finfo(output.policy.dtype).tiny
    new_log_probability = (
        output.policy.gather(-1, batch.learner_actions.unsqueeze(-1))
        .squeeze(-1)
        .clamp_min(tiny)
        .log()
    )
    advantage_mean = _weighted_mean(batch.advantages, batch.objective_weights)
    advantage_variance = _weighted_mean(
        (batch.advantages - advantage_mean).square(), batch.objective_weights
    )
    normalized_advantages = (batch.advantages - advantage_mean) / (
        advantage_variance.sqrt().clamp_min(1e-6)
    )
    ratio = torch.exp(new_log_probability - batch.old_log_probabilities)
    unclipped = ratio * normalized_advantages
    clipped = (
        torch.clamp(ratio, 1.0 - trainer.ppo_clip, 1.0 + trainer.ppo_clip)
        * normalized_advantages
    )
    policy_loss = -_weighted_mean(
        torch.minimum(unclipped, clipped), batch.objective_weights
    )
    entropy = -(
        output.policy.clamp_min(tiny) * output.policy.clamp_min(tiny).log()
    ).sum(-1)
    entropy_mean = _weighted_mean(entropy, batch.objective_weights)
    losses = _auxiliary_losses(output, batch)
    total = (
        policy_loss
        + trainer.value_coef * losses["value"]
        - trainer.entropy_coef * entropy_mean
        + trainer.ppo_prediction_coef * losses["prediction"]
        + trainer.ppo_tactical_coef * losses["tactical"]
    )
    optimizer.zero_grad(set_to_none=True)
    total.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), trainer.max_grad_norm
    )
    optimizer.step()
    clipped_fraction = _weighted_mean(
        (torch.abs(ratio - 1.0) > trainer.ppo_clip).to(torch.float32),
        batch.objective_weights,
    )
    return {
        "total_loss": float(total.detach().cpu()),
        "policy_loss": float(policy_loss.detach().cpu()),
        "entropy": float(entropy_mean.detach().cpu()),
        "clipped_fraction": float(clipped_fraction.detach().cpu()),
        **{name: float(value.detach().cpu()) for name, value in losses.items()},
        "gradient_norm": float(torch.as_tensor(gradient_norm).detach().cpu()),
    }


def _mean_metrics(items: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not items:
        return {}
    keys = tuple(items[0])
    return {key: float(np.mean([item[key] for item in items])) for key in keys}


@dataclass(frozen=True, slots=True)
class LeagueSessionCollector:
    """Default manifest schedule with an explicitly bound opponent factory."""

    manifest: OpponentFamilyManifest
    opponent_factory: OpponentFactory
    binding: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "binding",
            _canonical_session_collector_binding(self.binding),
        )

    def checkpoint_binding(self) -> Mapping[str, object]:
        return _canonical_session_collector_binding(self.binding)

    def collect_update(
        self,
        *,
        update_index: int,
        exact_agent: CompleteDTHAgent,
        model: AggroHalNetwork,
        trainer: AggroTrainerConfig,
        on_policy: bool,
        device: torch.device,
    ) -> Sequence[TrainingSequence]:
        variants = _manifest_variants(self.manifest)
        sequences: list[TrainingSequence] = []
        for offset in range(trainer.sessions_per_update):
            variant_index = update_index * trainer.sessions_per_update + offset
            family, base_seed = variants[variant_index % len(variants)]
            session_seed = (
                trainer.seed + (1_000_000 if on_policy else 0) + variant_index * 101
            )
            kwargs = {
                "exact_agent": exact_agent,
                "trainer": trainer,
                "family": family,
                "opponent_seed": base_seed + update_index * 10_007,
                "session_seed": session_seed,
                "learner_starts_in_hal_seat": variant_index % 2 == 0,
                "opponent_factory": self.opponent_factory,
            }
            if on_policy:
                sequence = collect_policy_session(
                    model=model,
                    device=device,
                    **kwargs,
                )
            else:
                sequence = collect_teacher_session(**kwargs)
            sequences.append(sequence)
        return sequences


def _memory_arm_training_sequence(
    arm: MemoryCurriculumArm,
    *,
    best_response_action: int,
) -> TrainingSequence:
    """Convert one legal paired-history arm into target-only supervision."""

    tokens = arm.tokens
    length = len(tokens)
    opponent_targets = np.asarray(arm.opponent_targets, dtype=np.float32)
    opponent_actions = np.argmax(opponent_targets, axis=-1).astype(np.int64)
    learner_actions = np.asarray(
        [int(np.argmax(token.exact_policy)) for token in tokens], dtype=np.int64
    )
    learner_actions[-1] = int(best_response_action) - 1
    objective_weights = np.zeros(length, dtype=np.float32)
    objective_weights[-1] = 1.0
    zeros = np.zeros(length, dtype=np.float32)
    return TrainingSequence(
        features=np.stack([token.features for token in tokens]).astype(np.float32),
        stage_matrices=np.stack([token.stage_matrix for token in tokens]).astype(
            np.float32
        ),
        exact_policies=np.stack([token.exact_policy for token in tokens]).astype(
            np.float32
        ),
        roles_are_dropper=np.asarray(
            [token.role_is_dropper for token in tokens], dtype=np.bool_
        ),
        legal_masks=np.stack([token.legal_mask for token in tokens]).astype(np.bool_),
        opponent_targets=opponent_targets,
        opponent_actions=opponent_actions,
        learner_actions=learner_actions,
        rewards=zeros.copy(),
        returns=zeros.copy(),
        objective_weights=objective_weights,
    )


@dataclass(frozen=True, slots=True)
class MemoryCurriculumSessionCollector:
    """Balanced warm-start-only collector for the causal memory curriculum."""

    split: MemoryCurriculumSplit

    def __post_init__(self) -> None:
        if self.split.name != "train":
            raise ValueError(
                "the training collector requires the train curriculum split"
            )

    def checkpoint_binding(self) -> Mapping[str, object]:
        return _canonical_session_collector_binding(
            {
                "schema_version": SESSION_COLLECTOR_BINDING_SCHEMA,
                "identity": MEMORY_SESSION_COLLECTOR_IDENTITY,
                "config": {
                    "curriculum": memory_curriculum_config_payload(self.split),
                    "curriculum_sha256": memory_curriculum_config_sha256(self.split),
                    "schedule": {
                        "balance_unit": "role-by-mode",
                        "sequences_per_identity": 4,
                        "ppo_allowed": False,
                    },
                },
            }
        )

    def collect_update(
        self,
        *,
        update_index: int,
        exact_agent: CompleteDTHAgent,
        model: AggroHalNetwork,
        trainer: AggroTrainerConfig,
        on_policy: bool,
        device: torch.device,
    ) -> Sequence[TrainingSequence]:
        del model, device
        if on_policy:
            raise ValueError("memory-necessity curriculum is warm-start-only")
        if trainer.sessions_per_update % 4:
            raise ValueError(
                "memory curriculum sessions_per_update must be divisible by four"
            )
        identities_per_update = trainer.sessions_per_update // 4
        seeds = self.split.example_seeds
        sequences: list[TrainingSequence] = []
        for identity_offset in range(identities_per_update):
            identity_index = update_index * identities_per_update + identity_offset
            example_seed = seeds[identity_index % len(seeds)]
            for case in build_memory_curriculum_role_pair(
                exact_agent,
                split=self.split,
                example_seed=example_seed,
            ):
                for mode, arm in (("a", case.mode_a), ("b", case.mode_b)):
                    sequences.append(
                        _memory_arm_training_sequence(
                            arm,
                            best_response_action=case.best_responses.action(
                                case.role, mode
                            ),
                        )
                    )
        return sequences


def default_training_session_collector() -> LeagueSessionCollector:
    """Construct the legacy train-manifest collector used when none is injected."""

    return LeagueSessionCollector(
        manifest=TRAIN_FAMILY_MANIFEST,
        opponent_factory=_default_opponent_factory,
        binding=_default_session_collector_binding(),
    )


def configured_training_session_collector(
    config_path: str | Path,
) -> TrainingSessionCollector:
    """Build the collector selected by a tracked training configuration."""

    root = _load_training_root(config_path)
    raw = root.get("session_collector")
    if raw is None:
        return default_training_session_collector()
    config = _mapping(raw, label="session collector config")
    required = {"schema_version", "type", "split"}
    if set(config) != required:
        raise ValueError(
            f"session collector config fields must be exactly {sorted(required)!r}"
        )
    if config.get("schema_version") != SESSION_COLLECTOR_CONFIG_SCHEMA:
        raise ValueError("unsupported session collector config schema")
    if config.get("type") != "memory-necessity":
        raise ValueError("unsupported session collector type")
    split_name = config.get("split")
    if split_name != "train":
        raise ValueError("memory-necessity training must use the train split")
    return MemoryCurriculumSessionCollector(memory_curriculum_split("train"))


def _collect_update_sessions(
    *,
    update_index: int,
    exact_agent: CompleteDTHAgent,
    model: AggroHalNetwork,
    trainer: AggroTrainerConfig,
    session_collector: TrainingSessionCollector,
    on_policy: bool,
    device: torch.device,
) -> list[TrainingSequence]:
    sequences = list(
        session_collector.collect_update(
            update_index=update_index,
            exact_agent=exact_agent,
            model=model,
            trainer=trainer,
            on_policy=on_policy,
            device=device,
        )
    )
    if len(sequences) != trainer.sessions_per_update:
        raise ValueError(
            "session collector returned "
            f"{len(sequences)} sessions; expected {trainer.sessions_per_update}"
        )
    max_decisions = trainer.games_per_session * trainer.max_half_rounds
    for index, sequence in enumerate(sequences):
        if not isinstance(sequence, TrainingSequence):
            raise TypeError(f"session collector item {index} is not a TrainingSequence")
        if sequence.length <= 0 or sequence.length > max_decisions:
            raise ValueError(
                f"session collector item {index} has {sequence.length} decisions; "
                f"expected 1..{max_decisions}"
            )
        if on_policy and (
            sequence.old_log_probabilities is None
            or sequence.old_values is None
            or sequence.advantages is None
        ):
            raise ValueError(
                f"on-policy session collector item {index} lacks PPO rollout fields"
            )
    return sequences


def train(
    config_path: str | Path,
    output_dir: str | Path,
    *,
    resume: str | Path | None = None,
    initial_checkpoint: str | Path | None = None,
    device_override: str | None = None,
    warmstart_updates_override: int | None = None,
    ppo_updates_override: int | None = None,
    exact_agent_override: CompleteDTHAgent | None = None,
    session_collector: TrainingSessionCollector | None = None,
) -> dict[str, object]:
    """Train, checkpoint, and report one Aggro Hal run."""

    if resume is not None and initial_checkpoint is not None:
        raise ValueError("resume and initial_checkpoint are mutually exclusive")
    experiment_binding = _configured_experiment_binding(config_path)
    configured_initial = _configured_initial_checkpoint_binding(config_path)
    if initial_checkpoint is not None and configured_initial is not None:
        supplied_sha256 = _sha256_file(initial_checkpoint)
        if supplied_sha256 != configured_initial["sha256"]:
            raise ValueError(
                "initial_checkpoint does not match the configured tactical baseline"
            )
    effective_initial_checkpoint = (
        initial_checkpoint
        if initial_checkpoint is not None
        else (
            None
            if resume is not None or configured_initial is None
            else configured_initial["path"]
        )
    )
    model_config, loaded_trainer = load_training_config(config_path)
    raw_trainer = asdict(loaded_trainer)
    if device_override is not None:
        raw_trainer["device"] = device_override
    if warmstart_updates_override is not None:
        raw_trainer["warmstart_updates"] = int(warmstart_updates_override)
    if ppo_updates_override is not None:
        raw_trainer["ppo_updates"] = int(ppo_updates_override)
    raw_trainer["start_clocks"] = tuple(raw_trainer["start_clocks"])
    trainer = AggroTrainerConfig(**raw_trainer)
    device = torch.device(trainer.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was explicitly requested but is unavailable")
    if device.type == "cpu":
        torch.set_num_threads(trainer.cpu_threads)
    torch.manual_seed(trainer.seed)
    np.random.seed(trainer.seed)
    random.seed(trainer.seed)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    exact_agent = exact_agent_override or CompleteDTHAgent(Path(trainer.dth_artifact))
    compatibility = dth_compatibility(exact_agent)
    collector = (
        configured_training_session_collector(config_path)
        if session_collector is None
        else session_collector
    )
    collector_binding = _canonical_session_collector_binding(
        collector.checkpoint_binding()
    )
    initial_checkpoint_binding: dict[str, object] | None = None
    if effective_initial_checkpoint is None:
        model = AggroHalNetwork(model_config).to(device)
    else:
        initial_source = Path(effective_initial_checkpoint)
        initial_sha256 = _sha256_file(initial_source)
        if (
            configured_initial is not None
            and initial_sha256 != configured_initial["sha256"]
        ):
            raise ValueError(
                "configured tactical baseline checkpoint bytes have changed"
            )
        model, _ = load_checkpoint(
            initial_source,
            expected_config=model_config,
            dth_ruleset=compatibility,
            device=device,
        )
        initial_checkpoint_binding = {
            "path": str(initial_source),
            "sha256": initial_sha256,
        }
    optimizer = torch.optim.AdamW(model.parameters(), lr=trainer.learning_rate)
    completed_warmstart = 0
    completed_ppo = 0
    history: list[dict[str, object]] = []
    if resume is not None:
        model, payload = load_checkpoint(
            resume,
            expected_config=model_config,
            dth_ruleset=compatibility,
            device=device,
        )
        state = _mapping(
            payload.get("training_state"), label="checkpoint training state"
        )
        if state.get("trainer_schema") != TRAINING_CONFIG_SCHEMA:
            raise ValueError("resume checkpoint has an incompatible trainer schema")
        if state.get("objective_schema") != TRAINING_OBJECTIVE_SCHEMA:
            raise ValueError("resume checkpoint has an incompatible training objective")
        stored_trainer = _mapping(
            state.get("trainer_config"), label="checkpoint trainer config"
        )
        _validate_resume_trainer_config(stored_trainer, trainer)
        _validate_resume_session_collector(
            state.get("session_collector"), collector_binding
        )
        if state.get("experiment") != experiment_binding:
            raise ValueError(
                "resume checkpoint is bound to a different adaptive experiment"
            )
        raw_optimizer = payload.get("optimizer_state_dict")
        if raw_optimizer is None:
            raise ValueError("resume checkpoint has no optimizer state")
        optimizer = torch.optim.AdamW(model.parameters(), lr=trainer.learning_rate)
        optimizer.load_state_dict(raw_optimizer)
        if any(
            not math.isclose(
                float(group["lr"]),
                trainer.learning_rate,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            for group in optimizer.param_groups
        ):
            raise ValueError(
                "resume optimizer learning rate does not match current config"
            )
        completed_warmstart = int(state.get("completed_warmstart_updates", 0))
        completed_ppo = int(state.get("completed_ppo_updates", 0))
        if (
            completed_warmstart > trainer.warmstart_updates
            or completed_ppo > trainer.ppo_updates
        ):
            raise ValueError("resume target updates cannot be below completed updates")
        raw_history = state.get("history", [])
        if not isinstance(raw_history, list):
            raise ValueError("checkpoint training history is malformed")
        history = list(raw_history)
        raw_initial_checkpoint = state.get("initial_checkpoint")
        if raw_initial_checkpoint is not None:
            initial_checkpoint_binding = dict(
                _mapping(
                    raw_initial_checkpoint,
                    label="checkpoint initial checkpoint binding",
                )
            )
        if configured_initial is not None and (
            initial_checkpoint_binding is None
            or initial_checkpoint_binding.get("sha256") != configured_initial["sha256"]
        ):
            raise ValueError(
                "resume checkpoint does not originate from the configured "
                "tactical baseline"
            )

    started = time.perf_counter()

    def persist() -> Path:
        return save_checkpoint(
            output / "checkpoint.pt",
            model=model,
            config=model_config,
            dth_ruleset=compatibility,
            optimizer=optimizer,
            training_state={
                "trainer_schema": TRAINING_CONFIG_SCHEMA,
                "objective_schema": TRAINING_OBJECTIVE_SCHEMA,
                "trainer_config": asdict(trainer),
                "session_collector": collector_binding,
                "experiment": experiment_binding,
                "initial_checkpoint": initial_checkpoint_binding,
                "completed_warmstart_updates": completed_warmstart,
                "completed_ppo_updates": completed_ppo,
                "history": history,
            },
        )

    for update in range(completed_warmstart, trainer.warmstart_updates):
        sequences = _collect_update_sessions(
            update_index=update,
            exact_agent=exact_agent,
            model=model,
            trainer=trainer,
            session_collector=collector,
            on_policy=False,
            device=device,
        )
        metrics = warmstart_step(
            model, optimizer, pad_sequences(sequences, device=device), trainer
        )
        completed_warmstart = update + 1
        history.append(
            {
                "phase": "warmstart",
                "update": completed_warmstart,
                "decisions": sum(sequence.length for sequence in sequences),
                "behavior_session_return": float(
                    np.mean([sequence.rewards.sum() for sequence in sequences])
                ),
                **metrics,
            }
        )
        if (
            trainer.snapshot_interval
            and completed_warmstart % trainer.snapshot_interval == 0
        ):
            persist()

    for update in range(completed_ppo, trainer.ppo_updates):
        sequences = _collect_update_sessions(
            update_index=update,
            exact_agent=exact_agent,
            model=model,
            trainer=trainer,
            session_collector=collector,
            on_policy=True,
            device=device,
        )
        batch = pad_sequences(sequences, device=device)
        epoch_metrics = [
            ppo_step(model, optimizer, batch, trainer)
            for _ in range(trainer.ppo_epochs)
        ]
        completed_ppo = update + 1
        history.append(
            {
                "phase": "ppo",
                "update": completed_ppo,
                "decisions": sum(sequence.length for sequence in sequences),
                "policy_session_return": float(
                    np.mean([sequence.rewards.sum() for sequence in sequences])
                ),
                **_mean_metrics(epoch_metrics),
            }
        )
        if trainer.snapshot_interval and completed_ppo % trainer.snapshot_interval == 0:
            persist()

    checkpoint = persist()
    elapsed = time.perf_counter() - started
    report: dict[str, object] = {
        "schema_version": TRAINING_REPORT_SCHEMA,
        "config": str(config_path),
        "checkpoint": str(checkpoint),
        "device": str(device),
        "gpu_was_opt_in": device.type == "cuda",
        "model_config": asdict(model_config),
        "trainer_config": asdict(trainer),
        "session_collector": collector_binding,
        "experiment": experiment_binding,
        "initial_checkpoint": initial_checkpoint_binding,
        "completed_warmstart_updates": completed_warmstart,
        "completed_ppo_updates": completed_ppo,
        "training_families": (
            list(collector.manifest.families)
            if isinstance(collector, LeagueSessionCollector)
            else []
        ),
        "wall_seconds": elapsed,
        "history": history,
        "latest": history[-1] if history else None,
    }
    _json_write(output / "training-report.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m arena.policies.train_aggro_hal")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--config", required=True)
    train_parser.add_argument("--output-dir", required=True)
    recovery = train_parser.add_mutually_exclusive_group()
    recovery.add_argument("--resume")
    recovery.add_argument("--initial-checkpoint")
    train_parser.add_argument(
        "--device", default=None, help="explicit torch device; config defaults to cpu"
    )
    train_parser.add_argument("--warmstart-updates", type=int)
    train_parser.add_argument("--ppo-updates", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "train":
        report = train(
            args.config,
            args.output_dir,
            resume=args.resume,
            initial_checkpoint=args.initial_checkpoint,
            device_override=args.device,
            warmstart_updates_override=args.warmstart_updates,
            ppo_updates_override=args.ppo_updates,
        )
        print(json.dumps(report["latest"], indent=2, sort_keys=True))
        return 0
    raise RuntimeError(f"unknown command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
