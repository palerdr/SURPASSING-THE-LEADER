"""Versioned reinforcement-learning contracts around the canonical engine.

The engine decides only real game termination.  Training and evaluation use a
finite episode wrapper so a safe diagonal line cannot run forever; a capped
non-terminal position is a tagged draw, never an engine terminal.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
from typing import Any, Mapping

from omegaconf import DictConfig, ListConfig, OmegaConf

from stl.engine.game import Game
from stl.solver.exact import terminal_value


RL_CONFIG_SCHEMA_VERSION = "stl.rl-config.v1"
TRAIN_MAX_HALF_ROUNDS = 64
EVAL_MAX_HALF_ROUNDS = 128


@dataclass(frozen=True)
class EpisodeOutcome:
    """Resolved learning objective for one engine trajectory."""

    value_for_hal: float
    terminal: bool
    truncated: bool
    half_rounds: int
    winner_name: str | None


@dataclass(frozen=True)
class RoleAssignment:
    """Current simultaneous roles under the fixed Hal-value perspective."""

    dropper_name: str
    checker_name: str
    hal_role: str
    baku_role: str
    maximizing_actor: str = "Hal"
    minimizing_actor: str = "Baku"


@dataclass(frozen=True)
class HorizonSensitivityReport:
    """Paired comparison of the two frozen episode caps."""

    score_at_64: float
    score_at_128: float
    paired_score_delta: float
    shared_state_policy_tv: float
    max_score_delta: float = 0.02
    max_policy_tv: float = 0.10

    @property
    def outcome_relevant(self) -> bool:
        return (
            self.paired_score_delta > self.max_score_delta
            or self.shared_state_policy_tv > self.max_policy_tv
        )


def current_role_assignment(game: Game) -> RoleAssignment:
    """Return actor/role orientation without changing the value perspective."""

    dropper, checker = game.get_roles_for_half(game.current_half)
    by_name = {dropper.name.lower(): "dropper", checker.name.lower(): "checker"}
    if "hal" not in by_name or "baku" not in by_name:
        raise ValueError("RL role mapping requires canonical Hal and Baku players")
    return RoleAssignment(
        dropper_name=dropper.name,
        checker_name=checker.name,
        hal_role=by_name["hal"],
        baku_role=by_name["baku"],
    )


def episode_outcome(
    game: Game,
    *,
    half_rounds: int,
    max_half_rounds: int,
) -> EpisodeOutcome | None:
    """Return a terminal/capped outcome, or ``None`` while play continues.

    True engine termination takes precedence when it occurs on the cap.  The
    wrapper deliberately does not mutate ``game_over``, ``winner``, or
    ``loser`` when it adjudicates a capped non-terminal as a training draw.
    """

    if max_half_rounds <= 0:
        raise ValueError("max_half_rounds must be positive")
    if half_rounds < 0:
        raise ValueError("half_rounds must be nonnegative")

    value = terminal_value(game, perspective_name="Hal")
    if value is not None:
        return EpisodeOutcome(
            value_for_hal=float(value),
            terminal=True,
            truncated=False,
            half_rounds=int(half_rounds),
            winner_name=game.winner.name if game.winner is not None else None,
        )
    if half_rounds >= max_half_rounds:
        return EpisodeOutcome(
            value_for_hal=0.0,
            terminal=False,
            truncated=True,
            half_rounds=int(half_rounds),
            winner_name=None,
        )
    return None


def _plain_config(value: Any) -> Any:
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Mapping):
        return {str(key): _plain_config(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_config(item) for item in value]
    return value


def canonical_config_json(config: Any) -> str:
    """Serialize resolved configuration deterministically for manifests."""

    return json.dumps(
        _plain_config(config),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def config_digest(config: Any) -> str:
    """Return the SHA-256 digest of the canonical resolved configuration."""

    payload = canonical_config_json(config).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def assess_horizon_sensitivity(
    *,
    score_at_64: float,
    score_at_128: float,
    shared_state_policy_tv: float,
    max_score_delta: float = 0.02,
    max_policy_tv: float = 0.10,
) -> HorizonSensitivityReport:
    """Create the mandatory 64-vs-128 promotion sensitivity record."""

    values = (score_at_64, score_at_128, shared_state_policy_tv)
    if not all(isinstance(value, (int, float)) for value in values):
        raise TypeError("horizon sensitivity metrics must be numeric")
    if not 0.0 <= score_at_64 <= 1.0 or not 0.0 <= score_at_128 <= 1.0:
        raise ValueError("paired scores must be in [0, 1]")
    if not 0.0 <= shared_state_policy_tv <= 1.0:
        raise ValueError("policy total variation must be in [0, 1]")
    return HorizonSensitivityReport(
        score_at_64=float(score_at_64),
        score_at_128=float(score_at_128),
        paired_score_delta=abs(float(score_at_128) - float(score_at_64)),
        shared_state_policy_tv=float(shared_state_policy_tv),
        max_score_delta=float(max_score_delta),
        max_policy_tv=float(max_policy_tv),
    )


def validate_rl_config(config: Any) -> dict[str, Any]:
    """Validate the resolved baseline contract before command dispatch."""

    plain = _plain_config(config)
    if not isinstance(plain, dict):
        raise TypeError("rl config must be a mapping")
    if plain.get("schema_version") != RL_CONFIG_SCHEMA_VERSION:
        raise ValueError("unsupported rl config schema_version")
    episode = plain.get("episode", {})
    if episode.get("train_max_half_rounds") != TRAIN_MAX_HALF_ROUNDS:
        raise ValueError("train_max_half_rounds does not match the frozen P0 contract")
    if episode.get("eval_max_half_rounds") != EVAL_MAX_HALF_ROUNDS:
        raise ValueError("eval_max_half_rounds does not match the frozen P0 contract")
    model = plain.get("model", {})
    if model.get("feature_schema") != "stl.features.v2":
        raise ValueError("baseline must use feature schema V2")
    if model.get("action_size") != 62:
        raise ValueError("baseline action size must be 62")
    replay = plain.get("replay", {})
    if replay.get("schema") != "stl.training-record.v3":
        raise ValueError("baseline must use reconstructable training-record V3")
    mcts = plain.get("mcts", {})
    if mcts.get("matrix_solver") not in {"lp", "cfr_plus"}:
        raise ValueError("default matrix solver must be Python LP or CFR+")
    if mcts.get("action_mode") not in {
        "candidate",
        "candidate_playable",
        "full_width",
    }:
        raise ValueError("unsupported MCTS action mode")
    if "stl_solver_rs" in canonical_config_json(plain):
        raise ValueError("P0-P8 baseline may not depend on stl_solver_rs")
    return plain
