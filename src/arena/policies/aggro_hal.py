"""Direct recurrent pure-DTH policy for aggressively exploiting repeated opponents.

The network chooses among the 60 literal pure-DTH seconds.  Exact DTH remains
the source of continuation-adjusted stage matrices and equilibrium policies,
but it is an input rather than a safety restriction: a learned gate may put all
mass on the opponent-directed policy.

One recurrent token is consumed per Hal decision.  A revealed half-round is
queued by :meth:`AggroHalPolicyProvider.observe` and cannot affect the network
until a later call to :meth:`AggroHalPolicyProvider.policy`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from arena.contracts import (
    CanonicalDecision,
    CanonicalPolicyProvider,
    PublicGameOutcome,
    PublicHalfRound,
)
from arena.dth_adapter import project_to_dth_state
from dth.agent import CertifiedStageGame, CompleteDTHAgent

ACTION_COUNT = 60
FAST_ADAPTATION_DECAY = 0.70
FAST_ADAPTATION_PRIOR_STRENGTH = 0.25
FAST_ADAPTATION_MAX_WEIGHT = 0.85
CHECKPOINT_SCHEMA = "arena-aggro-hal-checkpoint-v1"
OBSERVATION_SCHEMA = "arena-aggro-hal-public-sequence-v1"
DTH_COMPATIBILITY_SCHEMA = "arena-aggro-hal-dth-compatibility-v1"

_OUTCOMES = (
    "check_success",
    "check_fail_survived",
    "check_fail_died",
    "overflow_survived",
    "overflow_died",
    "unknown",
)
_DTH_METADATA_KEYS = (
    "solver_schema_hash",
    "table_digest",
    "class_encoding",
    "profile_count",
    "class_count",
    "max_support",
    "saddle_gap_tolerance",
    "canonical_table",
    "ladder",
    "code_config_digest",
)


@dataclass(frozen=True, slots=True)
class AggroHalConfig:
    """Network shape and tactical bridge configuration.

    The first version deliberately fixes the game interface and recurrent
    depth.  Widths and the strength of the exact-matrix tactical signal remain
    configurable and are all checkpoint compatibility gates.
    """

    action_count: int = ACTION_COUNT
    hidden_size: int = 256
    gru_layers: int = 2
    head_hidden_size: int = 128
    gru_dropout: float = 0.0
    tactical_logit_scale: float = 5.0

    def __post_init__(self) -> None:
        if self.action_count != ACTION_COUNT:
            raise ValueError("Aggro Hal v1 requires exactly 60 pure-DTH actions")
        if self.gru_layers != 2:
            raise ValueError("Aggro Hal v1 requires a two-layer GRU")
        if self.hidden_size <= 0 or self.head_hidden_size <= 0:
            raise ValueError("network widths must be positive")
        if float(self.gru_dropout) != 0.0:
            raise ValueError(
                "Aggro Hal v1 requires gru_dropout=0 so recurrent PPO ratios are stable"
            )
        if (
            not np.isfinite(self.tactical_logit_scale)
            or self.tactical_logit_scale < 0.0
        ):
            raise ValueError("tactical_logit_scale must be finite and nonnegative")


def observation_feature_names() -> tuple[str, ...]:
    """Return the immutable order of one public recurrent input token."""

    names = [
        "current_role_is_dropper",
        "current_role_is_checker",
        "current_turn_duration_over_61",
        "current_checker_cylinder_over_300",
        "current_checker_ttd_over_300",
        "current_dropper_cylinder_over_300",
        "current_dropper_ttd_over_300",
        "current_hal_stage_value",
        "current_saddle_gap",
        "current_new_game",
    ]
    names.extend(
        f"current_legal_action_{action}" for action in range(1, ACTION_COUNT + 1)
    )
    names.extend(
        f"current_exact_action_probability_{action}"
        for action in range(1, ACTION_COUNT + 1)
    )
    names.extend(
        (
            "previous_reveal_present",
            "previous_game_index_squashed",
            "previous_half_round_index_squashed",
            "previous_game_clock_over_3601",
            "previous_round_index_squashed",
            "previous_half_index_over_2",
            "previous_turn_duration_over_61",
            "previous_self_cylinder_over_300",
            "previous_self_ttd_over_300",
            "previous_opponent_cylinder_over_300",
            "previous_opponent_ttd_over_300",
            "previous_dropper_is_self",
            "previous_checker_is_self",
            "previous_drop_action_supported",
            "previous_check_action_supported",
        )
    )
    names.extend(
        f"previous_drop_action_{action}" for action in range(1, ACTION_COUNT + 1)
    )
    names.extend(
        f"previous_check_action_{action}" for action in range(1, ACTION_COUNT + 1)
    )
    names.extend(f"previous_outcome_{outcome}" for outcome in _OUTCOMES)
    names.extend(
        (
            "previous_game_over",
            "previous_winner_is_self",
            "previous_winner_is_opponent",
            "previous_winner_is_none",
        )
    )
    return tuple(names)


OBSERVATION_FEATURES = observation_feature_names()
OBSERVATION_DIM = len(OBSERVATION_FEATURES)


def _squash_nonnegative(value: int | float) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("public counters must be finite and nonnegative")
    return value / (1.0 + value)


def _validate_stage(stage: CertifiedStageGame) -> None:
    if stage.matrix.shape != (ACTION_COUNT, ACTION_COUNT):
        raise ValueError("certified DTH stage matrix must be 60x60")
    if not np.all(np.isfinite(stage.matrix)):
        raise ValueError("certified DTH stage matrix must be finite")
    for label, policy in (
        ("dropper", stage.drop_policy),
        ("checker", stage.check_policy),
    ):
        values = np.asarray(policy, dtype=np.float64)
        if (
            values.shape != (ACTION_COUNT,)
            or not np.all(np.isfinite(values))
            or np.any(values < 0.0)
            or abs(float(values.sum()) - 1.0) > 1e-8
        ):
            raise ValueError(f"certified DTH {label} policy is malformed")
    if not np.isfinite(stage.value) or not np.isfinite(stage.saddle_gap):
        raise ValueError("certified DTH value and saddle gap must be finite")


def _player_loads(record: PublicHalfRound, self_name: str) -> tuple[float, ...]:
    players = record.pre_decision_state.players
    self_players = [p for p in players if p.name.casefold() == self_name.casefold()]
    opponents = [p for p in players if p.name.casefold() != self_name.casefold()]
    if len(self_players) != 1 or len(opponents) != 1:
        raise ValueError("public reveal must contain exactly self and one opponent")
    own = self_players[0]
    opponent = opponents[0]
    return (
        float(own.cylinder_seconds) / 300.0,
        float(own.ttd_seconds) / 300.0,
        float(opponent.cylinder_seconds) / 300.0,
        float(opponent.ttd_seconds) / 300.0,
    )


def encode_public_observation(
    decision: CanonicalDecision,
    stage: CertifiedStageGame,
    previous_reveal: PublicHalfRound | None,
    *,
    previous_self_name: str | None = None,
    new_game: bool,
) -> np.ndarray:
    """Encode only information public before the current Hal action exists."""

    if decision.role not in {"dropper", "checker"}:
        raise ValueError("decision role must be 'dropper' or 'checker'")
    _validate_stage(stage)
    exact = stage.drop_policy if decision.role == "dropper" else stage.check_policy
    legal = np.zeros(ACTION_COUNT, dtype=np.float32)
    for second in decision.legal_seconds:
        if 1 <= int(second) <= ACTION_COUNT:
            legal[int(second) - 1] = 1.0
    if not np.any(legal):
        raise ValueError("decision has no legal pure-DTH action in 1..60")

    values: list[float] = [
        float(decision.role == "dropper"),
        float(decision.role == "checker"),
        float(decision.turn_duration) / 61.0,
        float(decision.checker_cylinder_seconds) / 300.0,
        float(decision.checker_ttd_seconds) / 300.0,
        float(decision.dropper_cylinder_seconds) / 300.0,
        float(decision.dropper_ttd_seconds) / 300.0,
        float(stage.value if decision.role == "dropper" else -stage.value),
        float(stage.saddle_gap),
        float(bool(new_game)),
    ]
    values.extend(float(x) for x in legal)
    values.extend(float(x) for x in exact)

    if previous_reveal is None:
        if previous_self_name is not None:
            raise ValueError("previous_self_name requires a previous public reveal")
        values.extend([0.0] * (OBSERVATION_DIM - len(values)))
    else:
        if previous_self_name is None:
            raise ValueError("a previous public reveal requires its observed self name")
        state = previous_reveal.pre_decision_state
        drop_supported = 1 <= previous_reveal.drop_time <= ACTION_COUNT
        check_supported = 1 <= previous_reveal.check_time <= ACTION_COUNT
        values.extend(
            (
                1.0,
                _squash_nonnegative(previous_reveal.game_index),
                _squash_nonnegative(previous_reveal.half_round_index),
                float(state.game_clock_seconds) / 3601.0,
                _squash_nonnegative(state.round_index),
                float(state.half_index) / 2.0,
                float(state.turn_duration) / 61.0,
                *_player_loads(previous_reveal, previous_self_name),
                float(
                    previous_reveal.dropper_name.casefold()
                    == previous_self_name.casefold()
                ),
                float(
                    previous_reveal.checker_name.casefold()
                    == previous_self_name.casefold()
                ),
                float(drop_supported),
                float(check_supported),
            )
        )
        drop_one_hot = np.zeros(ACTION_COUNT, dtype=np.float32)
        check_one_hot = np.zeros(ACTION_COUNT, dtype=np.float32)
        if drop_supported:
            drop_one_hot[previous_reveal.drop_time - 1] = 1.0
        if check_supported:
            check_one_hot[previous_reveal.check_time - 1] = 1.0
        values.extend(float(x) for x in drop_one_hot)
        values.extend(float(x) for x in check_one_hot)
        outcome = (
            previous_reveal.outcome
            if previous_reveal.outcome in _OUTCOMES[:-1]
            else "unknown"
        )
        values.extend(float(outcome == label) for label in _OUTCOMES)
        winner = previous_reveal.winner_name
        values.extend(
            (
                float(previous_reveal.game_over),
                float(
                    winner is not None
                    and winner.casefold() == previous_self_name.casefold()
                ),
                float(
                    winner is not None
                    and winner.casefold() != previous_self_name.casefold()
                ),
                float(winner is None),
            )
        )

    encoded = np.asarray(values, dtype=np.float32)
    if encoded.shape != (OBSERVATION_DIM,) or not np.all(np.isfinite(encoded)):
        raise RuntimeError("Aggro Hal public observation encoding is malformed")
    return encoded


@dataclass(frozen=True, slots=True)
class AggroHalNetworkOutput:
    """All heads and the analytic bridge for training and diagnostics."""

    policy: Tensor
    opponent_policy: Tensor
    analytic_action_values: Tensor
    residual_logits: Tensor
    direct_logits: Tensor
    direct_policy: Tensor
    direct_weight: Tensor
    value: Tensor
    hidden_state: Tensor


class AggroHalNetwork(nn.Module):
    """Two-layer recurrent actor with an exact-matrix best-response bridge."""

    def __init__(self, config: AggroHalConfig = AggroHalConfig()) -> None:
        super().__init__()
        self.config = config
        self.gru = nn.GRU(
            input_size=OBSERVATION_DIM,
            hidden_size=config.hidden_size,
            num_layers=config.gru_layers,
            dropout=config.gru_dropout,
            batch_first=True,
        )

        def head(output_size: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(config.hidden_size, config.head_hidden_size),
                nn.Tanh(),
                nn.Linear(config.head_hidden_size, output_size),
            )

        self.opponent_head = head(ACTION_COUNT)
        self.residual_head = head(ACTION_COUNT)
        self.mixture_gate = head(1)
        self.value_head = head(1)

    def initial_hidden(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        """Return the canonical zero state for a fresh opponent session."""

        parameter = next(self.parameters())
        return torch.zeros(
            self.config.gru_layers,
            int(batch_size),
            self.config.hidden_size,
            device=parameter.device if device is None else device,
            dtype=parameter.dtype if dtype is None else dtype,
        )

    @staticmethod
    def _validate_inputs(
        features: Tensor,
        stage_matrices: Tensor,
        exact_policies: Tensor,
        role_is_dropper: Tensor,
        legal_masks: Tensor,
    ) -> tuple[int, int]:
        if features.ndim != 3 or features.shape[-1] != OBSERVATION_DIM:
            raise ValueError("features must have shape [batch, time, observation_dim]")
        batch, time, _ = features.shape
        if stage_matrices.shape != (batch, time, ACTION_COUNT, ACTION_COUNT):
            raise ValueError("stage_matrices must have shape [batch, time, 60, 60]")
        if exact_policies.shape != (batch, time, ACTION_COUNT):
            raise ValueError("exact_policies must have shape [batch, time, 60]")
        if role_is_dropper.shape != (batch, time):
            raise ValueError("role_is_dropper must have shape [batch, time]")
        if legal_masks.shape != (batch, time, ACTION_COUNT):
            raise ValueError("legal_masks must have shape [batch, time, 60]")
        if not torch.all(legal_masks.to(dtype=torch.bool).any(dim=-1)):
            raise ValueError("each decision must have at least one legal action")
        return batch, time

    def forward(
        self,
        features: Tensor,
        stage_matrices: Tensor,
        exact_policies: Tensor,
        role_is_dropper: Tensor,
        legal_masks: Tensor,
        hidden_state: Tensor | None = None,
    ) -> AggroHalNetworkOutput:
        """Evaluate a batch of decision sequences.

        ``CertifiedStageGame.matrix`` is the Dropper payoff matrix.  Therefore
        a Dropper facing predicted Checker distribution ``q`` uses ``M @ q``;
        a Checker facing predicted Dropper distribution ``q`` uses
        ``-(M.T @ q)``.
        """

        batch, _ = self._validate_inputs(
            features,
            stage_matrices,
            exact_policies,
            role_is_dropper,
            legal_masks,
        )
        if hidden_state is None:
            hidden_state = self.initial_hidden(
                batch, device=features.device, dtype=features.dtype
            )
        recurrent, next_hidden = self.gru(features, hidden_state)
        opponent_logits = self.opponent_head(recurrent)
        opponent_policy = torch.softmax(opponent_logits, dim=-1)

        q_column = opponent_policy.unsqueeze(-1)
        dropper_values = torch.matmul(stage_matrices, q_column).squeeze(-1)
        checker_values = -torch.matmul(
            stage_matrices.transpose(-1, -2), q_column
        ).squeeze(-1)
        analytic = torch.where(
            role_is_dropper.to(dtype=torch.bool).unsqueeze(-1),
            dropper_values,
            checker_values,
        )
        residual = self.residual_head(recurrent)
        direct_logits = residual + self.config.tactical_logit_scale * analytic

        legal = legal_masks.to(dtype=torch.bool)
        masked_direct_logits = direct_logits.masked_fill(~legal, -torch.inf)
        direct_policy = torch.softmax(masked_direct_logits, dim=-1)

        exact = torch.clamp(exact_policies, min=0.0) * legal.to(exact_policies.dtype)
        exact_mass = exact.sum(dim=-1, keepdim=True)
        uniform_legal = legal.to(exact.dtype) / legal.sum(dim=-1, keepdim=True)
        normalized_exact = torch.where(
            exact_mass > 0.0,
            exact / exact_mass.clamp_min(torch.finfo(exact.dtype).tiny),
            uniform_legal,
        )

        direct_weight = torch.sigmoid(self.mixture_gate(recurrent))
        policy = (
            1.0 - direct_weight
        ) * normalized_exact + direct_weight * direct_policy
        policy = policy * legal.to(policy.dtype)
        policy = policy / policy.sum(dim=-1, keepdim=True)
        value = self.value_head(recurrent).squeeze(-1)
        return AggroHalNetworkOutput(
            policy=policy,
            opponent_policy=opponent_policy,
            analytic_action_values=analytic,
            residual_logits=residual,
            direct_logits=direct_logits,
            direct_policy=direct_policy,
            direct_weight=direct_weight.squeeze(-1),
            value=value,
            hidden_state=next_hidden,
        )


def dth_compatibility(agent: CompleteDTHAgent) -> dict[str, object]:
    """Bind a checkpoint to the exact completed DTH artifact it was trained on."""

    metadata = agent.tablebase.metadata
    missing = [key for key in _DTH_METADATA_KEYS if key not in metadata]
    if missing:
        raise ValueError(
            f"DTH artifact metadata is missing compatibility keys {missing}"
        )
    if metadata["canonical_table"] is not True:
        raise ValueError("Aggro Hal requires a canonical complete DTH artifact")
    return {
        "schema_version": DTH_COMPATIBILITY_SCHEMA,
        **{key: metadata[key] for key in _DTH_METADATA_KEYS},
    }


def _stable_hash(payload: Mapping[str, object] | Sequence[object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _checkpoint_gates(
    config: AggroHalConfig,
    compatibility: Mapping[str, object],
) -> dict[str, object]:
    config_payload = asdict(config)
    architecture = {
        "input_size": OBSERVATION_DIM,
        "hidden_size": config.hidden_size,
        "gru_layers": config.gru_layers,
        "head_hidden_size": config.head_hidden_size,
        "action_count": ACTION_COUNT,
    }
    return {
        "observation_schema": OBSERVATION_SCHEMA,
        "observation_features": list(OBSERVATION_FEATURES),
        "observation_dim": OBSERVATION_DIM,
        "action_count": ACTION_COUNT,
        "config": config_payload,
        "config_hash": _stable_hash(config_payload),
        "network_architecture": architecture,
        "dth_ruleset_compatibility": dict(compatibility),
    }


def save_checkpoint(
    path: str | Path,
    *,
    model: AggroHalNetwork,
    config: AggroHalConfig,
    dth_ruleset: Mapping[str, object],
    optimizer: torch.optim.Optimizer | None = None,
    training_state: Mapping[str, object] | None = None,
) -> Path:
    """Atomically save a model with strict shape, schema, and DTH gates."""

    if model.config != config:
        raise ValueError("model configuration does not match checkpoint configuration")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "schema_version": CHECKPOINT_SCHEMA,
        **_checkpoint_gates(config, dth_ruleset),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
        if optimizer is not None
        else None,
        "training_state": dict(training_state or {}),
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)
    return destination


def load_checkpoint(
    path: str | Path,
    *,
    expected_config: AggroHalConfig | None = None,
    dth_ruleset: Mapping[str, object],
    device: str | torch.device = "cpu",
) -> tuple[AggroHalNetwork, dict[str, object]]:
    """Load a self-describing strict checkpoint; CPU is the fail-safe default."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Aggro Hal checkpoint is required at {source}")
    payload = torch.load(source, map_location=torch.device(device), weights_only=False)
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != CHECKPOINT_SCHEMA
    ):
        raise ValueError("unsupported Aggro Hal checkpoint schema")
    raw_config = payload.get("config")
    if not isinstance(raw_config, dict):
        raise ValueError("Aggro Hal checkpoint is missing its model configuration")
    try:
        stored_config = AggroHalConfig(**raw_config)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Aggro Hal checkpoint model configuration is malformed"
        ) from error
    if expected_config is not None and stored_config != expected_config:
        raise ValueError("Aggro Hal checkpoint config is incompatible")
    resolved_config = stored_config if expected_config is None else expected_config
    for key, expected in _checkpoint_gates(resolved_config, dth_ruleset).items():
        if payload.get(key) != expected:
            raise ValueError(f"Aggro Hal checkpoint {key} is incompatible")
    model = AggroHalNetwork(resolved_config)
    try:
        model.load_state_dict(payload["model_state_dict"], strict=True)
    except (KeyError, RuntimeError) as error:
        raise ValueError(
            "Aggro Hal checkpoint model shapes are incompatible"
        ) from error
    model.to(torch.device(device))
    model.eval()
    return model, payload


@dataclass(frozen=True, slots=True)
class AggroHalDecision:
    """Read-only diagnostics from the most recent live decision."""

    role: str
    policy: tuple[float, ...]
    opponent_policy: tuple[float, ...]
    analytic_action_values: tuple[float, ...]
    direct_weight: float
    fast_adaptation_weight: float
    value: float


class AggroHalPolicyProvider(CanonicalPolicyProvider):
    """Live recurrent provider whose memory persists across repeated games."""

    def __init__(
        self,
        artifact_dir: str | Path,
        model: AggroHalNetwork,
        config: AggroHalConfig,
        *,
        agent: CompleteDTHAgent | None = None,
        device: str | torch.device = "cpu",
        fast_adaptation: bool = False,
    ) -> None:
        if model.config != config:
            raise ValueError("Aggro Hal model and provider configurations differ")
        self.artifact_dir = Path(artifact_dir)
        self.config = config
        self.agent = agent or CompleteDTHAgent(self.artifact_dir)
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.fast_adaptation = bool(fast_adaptation)
        self.decisions: list[AggroHalDecision] = []
        self._hidden_state: Tensor | None = None
        self._pending_reveal: PublicHalfRound | None = None
        self._pending_reveal_self_name: str | None = None
        self._seen_reveals: set[tuple[int, int, int]] = set()
        self._game_epoch = -1
        self._current_actor_name: str | None = None
        self._new_game = True
        self._game_started = False
        self._last_outcome: PublicGameOutcome | None = None
        self._fast_evidence = {
            "dropper": np.zeros(ACTION_COUNT, dtype=np.float64),
            "checker": np.zeros(ACTION_COUNT, dtype=np.float64),
        }

    @classmethod
    def from_checkpoint(
        cls,
        *,
        artifact_dir: str | Path,
        checkpoint: str | Path,
        config: AggroHalConfig | None = None,
        device: str | torch.device = "cpu",
        fast_adaptation: bool = False,
    ) -> AggroHalPolicyProvider:
        """Construct the live provider only after artifact compatibility passes."""

        agent = CompleteDTHAgent(artifact_dir)
        model, _ = load_checkpoint(
            checkpoint,
            expected_config=config,
            dth_ruleset=dth_compatibility(agent),
            device=device,
        )
        return cls(
            artifact_dir,
            model,
            model.config,
            agent=agent,
            device=device,
            fast_adaptation=fast_adaptation,
        )

    @property
    def last_decision(self) -> AggroHalDecision | None:
        return self.decisions[-1] if self.decisions else None

    @property
    def has_session_memory(self) -> bool:
        return self._hidden_state is not None

    def close(self) -> None:
        """Match the provider lifecycle; tablebase memory maps need no close."""

    def reset_session(self) -> None:
        """Forget the opponent and begin an independent repeated-game session."""

        self._hidden_state = None
        self._pending_reveal = None
        self._pending_reveal_self_name = None
        self._seen_reveals.clear()
        self._game_epoch = -1
        self._current_actor_name = None
        self._new_game = True
        self._game_started = False
        self._last_outcome = None
        self.decisions.clear()
        for evidence in self._fast_evidence.values():
            evidence.fill(0.0)

    def reset_game(self) -> None:
        """Mark a game boundary without erasing recurrent opponent memory."""

        self._game_epoch += 1
        self._new_game = True
        self._game_started = True
        self._last_outcome = None
        self._current_actor_name = None

    def observe(self, record: PublicHalfRound) -> None:
        """Queue one public reveal; it is consumed only by the next decision."""

        if self._current_actor_name is None:
            raise RuntimeError("Aggro Hal observed a half-round before acting")
        if record.game_index < 0 or record.half_round_index < 0:
            raise ValueError("public reveal indices must be nonnegative")
        names = {record.dropper_name.casefold(), record.checker_name.casefold()}
        if self._current_actor_name.casefold() not in names:
            raise ValueError("public reveal does not include the Aggro Hal player")
        key = (self._game_epoch, record.game_index, record.half_round_index)
        if key in self._seen_reveals:
            raise RuntimeError("public reveal was delivered more than once")
        if self._pending_reveal is not None:
            raise RuntimeError("previous public reveal has not been consumed")
        self._seen_reveals.add(key)
        self._pending_reveal = record
        self._pending_reveal_self_name = self._current_actor_name
        if self.fast_adaptation:
            if record.dropper_name.casefold() == self._current_actor_name.casefold():
                opponent_role = "checker"
                opponent_action = int(record.check_time)
            else:
                opponent_role = "dropper"
                opponent_action = int(record.drop_time)
            if not 1 <= opponent_action <= ACTION_COUNT:
                raise ValueError(
                    "fast pure-DTH adaptation requires opponent actions 1..60"
                )
            evidence = self._fast_evidence[opponent_role]
            evidence *= FAST_ADAPTATION_DECAY
            evidence[opponent_action - 1] += 1.0

    def end_game(self, outcome: PublicGameOutcome) -> None:
        """Record a public terminal result while retaining recurrent memory."""

        if outcome.game_index < 0 or outcome.half_rounds < 0:
            raise ValueError("public game outcome indices must be nonnegative")
        self._last_outcome = outcome

    def policy(self, decision: CanonicalDecision) -> Mapping[int, float]:
        if decision.turn_duration != ACTION_COUNT or tuple(
            decision.legal_seconds
        ) != tuple(range(1, ACTION_COUNT + 1)):
            raise ValueError(
                "Aggro Hal supports pure DTH only: turn duration and legal actions "
                "must be exactly literal seconds 1..60"
            )
        if self._current_actor_name is None:
            self._current_actor_name = decision.actor_name
        elif self._current_actor_name.casefold() != decision.actor_name.casefold():
            raise RuntimeError(
                "Aggro Hal provider cannot switch player identity mid-game"
            )

        stage = self.agent.stage_game(project_to_dth_state(decision))
        features = encode_public_observation(
            decision,
            stage,
            self._pending_reveal,
            previous_self_name=self._pending_reveal_self_name,
            new_game=self._new_game,
        )
        exact = stage.drop_policy if decision.role == "dropper" else stage.check_policy
        legal = np.asarray(
            [second in decision.legal_seconds for second in range(1, ACTION_COUNT + 1)],
            dtype=np.bool_,
        )
        with torch.inference_mode():
            output = self.model(
                torch.as_tensor(features, dtype=torch.float32, device=self.device).view(
                    1, 1, -1
                ),
                torch.tensor(
                    stage.matrix, dtype=torch.float32, device=self.device
                ).view(1, 1, ACTION_COUNT, ACTION_COUNT),
                torch.tensor(exact, dtype=torch.float32, device=self.device).view(
                    1, 1, ACTION_COUNT
                ),
                torch.tensor(
                    [[decision.role == "dropper"]], dtype=torch.bool, device=self.device
                ),
                torch.as_tensor(legal, dtype=torch.bool, device=self.device).view(
                    1, 1, ACTION_COUNT
                ),
                self._hidden_state,
            )
        self._hidden_state = output.hidden_state.detach()
        self._pending_reveal = None
        self._pending_reveal_self_name = None
        self._new_game = False

        policy = output.policy[0, 0].detach().cpu().numpy().astype(np.float64)
        opponent = (
            output.opponent_policy[0, 0].detach().cpu().numpy().astype(np.float64)
        )
        analytic = (
            output.analytic_action_values[0, 0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        fast_weight = 0.0
        if self.fast_adaptation:
            opponent_role = "checker" if decision.role == "dropper" else "dropper"
            evidence = self._fast_evidence[opponent_role]
            evidence_mass = float(evidence.sum())
            if evidence_mass > 0.0:
                empirical = (
                    evidence + FAST_ADAPTATION_PRIOR_STRENGTH / ACTION_COUNT
                ) / (evidence_mass + FAST_ADAPTATION_PRIOR_STRENGTH)
                concentration = max(
                    0.0,
                    (float(empirical.max()) - 1.0 / ACTION_COUNT)
                    / (1.0 - 1.0 / ACTION_COUNT),
                )
                evidence_fraction = evidence_mass / (
                    evidence_mass + FAST_ADAPTATION_PRIOR_STRENGTH
                )
                fast_weight = (
                    FAST_ADAPTATION_MAX_WEIGHT * evidence_fraction * concentration
                )
                if decision.role == "dropper":
                    empirical_values = np.asarray(stage.matrix) @ empirical
                else:
                    empirical_values = -np.asarray(stage.matrix).T @ empirical
                empirical_logits = self.config.tactical_logit_scale * empirical_values
                empirical_logits = np.where(legal, empirical_logits, -np.inf)
                empirical_logits -= float(np.max(empirical_logits[legal]))
                empirical_policy = np.zeros(ACTION_COUNT, dtype=np.float64)
                empirical_policy[legal] = np.exp(
                    np.clip(empirical_logits[legal], -80.0, 0.0)
                )
                empirical_policy /= float(empirical_policy.sum())
                policy = (1.0 - fast_weight) * policy + fast_weight * empirical_policy
                opponent = (1.0 - fast_weight) * opponent + fast_weight * empirical
                analytic = (
                    1.0 - fast_weight
                ) * analytic + fast_weight * empirical_values
                policy /= float(policy.sum())
        if (
            not np.all(np.isfinite(policy))
            or np.any(policy < 0.0)
            or abs(float(policy.sum()) - 1.0) > 1e-6
            or np.any(policy[~legal] > 0.0)
        ):
            raise RuntimeError("Aggro Hal network returned a malformed legal policy")
        diagnostic = AggroHalDecision(
            role=decision.role,
            policy=tuple(float(x) for x in policy),
            opponent_policy=tuple(float(x) for x in opponent),
            analytic_action_values=tuple(float(x) for x in analytic),
            direct_weight=float(output.direct_weight[0, 0].item()),
            fast_adaptation_weight=fast_weight,
            value=float(output.value[0, 0].item()),
        )
        self.decisions.append(diagnostic)
        return {
            action: float(policy[action - 1])
            for action in range(1, ACTION_COUNT + 1)
            if legal[action - 1] and policy[action - 1] > 0.0
        }


def make_live_provider(
    *,
    artifact_dir: str | Path,
    checkpoint: str | Path,
    config: AggroHalConfig | None = None,
    device: str | torch.device = "cpu",
    fast_adaptation: bool = False,
) -> AggroHalPolicyProvider:
    """Load the strict checkpoint and return the canonical live provider."""

    return AggroHalPolicyProvider.from_checkpoint(
        artifact_dir=artifact_dir,
        checkpoint=checkpoint,
        config=config,
        device=device,
        fast_adaptation=fast_adaptation,
    )
