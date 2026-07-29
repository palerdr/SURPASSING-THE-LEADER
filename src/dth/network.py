"""Small policy/value network for canonical DTH."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn


FEATURE_SCHEMA = (
    "checker_st/300",
    "checker_ttd/300",
    "dropper_st/300",
    "dropper_ttd/300",
    "remaining_horizon/horizon_scale",
)

FEATURE_LIFTS = ("identity", "boundary_v1", "boundary_v2")
CONTINUATION_RESIDUAL_MODES = ("matrix_head", "action_mlp")

# 60 successful-lag classes plus one failed-check class: the complete stage
# game has exactly these degrees of freedom (docs/GAME_AND_SOLVER.md).
TRANSITION_CLASS_COUNT = 61


@dataclass(frozen=True)
class DTHNetworkConfig:
    hidden_width: int = 64
    hidden_layers: int = 2
    action_count: int = 60
    horizon_scale: float = 3.0
    feature_lift: str = "identity"
    continuation_residual: bool = False
    continuation_residual_mode: str = "matrix_head"
    transition_class_head: bool = False
    play_value_head: bool = False

    def to_dict(self) -> dict[str, int | float | str | bool]:
        return asdict(self)


def encode_features(
    states: Tensor,
    horizons: Tensor,
    *,
    horizon_scale: float,
) -> Tensor:
    """Normalize raw exact-target coordinates into five network features."""

    if states.ndim != 2 or states.shape[1] != 4:
        raise ValueError(f"states must have shape (N, 4), got {tuple(states.shape)}")
    if horizons.ndim not in (1, 2):
        raise ValueError("horizons must have shape (N,) or (N, 1)")
    horizons = horizons.reshape(-1, 1)
    if horizons.shape[0] != states.shape[0]:
        raise ValueError("states and horizons must contain the same number of rows")
    if horizon_scale <= 0:
        raise ValueError("horizon_scale must be positive")

    dtype = torch.get_default_dtype()
    state_features = states.to(dtype=dtype) / 300.0
    horizon_feature = horizons.to(dtype=dtype) / float(horizon_scale)
    return torch.cat((state_features, horizon_feature), dim=1)


class DTHPolicyValueNet(nn.Module):
    """Shared MLP with current-role value, Dropper, and Checker heads."""

    def __init__(self, config: DTHNetworkConfig | None = None) -> None:
        super().__init__()
        self.config = config or DTHNetworkConfig()
        if self.config.hidden_width <= 0 or self.config.hidden_layers <= 0:
            raise ValueError("hidden width and layer count must be positive")
        if self.config.action_count <= 0:
            raise ValueError("action count must be positive")
        if self.config.feature_lift not in FEATURE_LIFTS:
            raise ValueError(f"unknown feature lift {self.config.feature_lift!r}")
        if self.config.continuation_residual_mode not in CONTINUATION_RESIDUAL_MODES:
            raise ValueError(
                "unknown continuation residual mode "
                f"{self.config.continuation_residual_mode!r}"
            )
        if self.config.transition_class_head and self.config.continuation_residual:
            raise ValueError(
                "transition class head and continuation residual both own the "
                "root matrix; enable at most one"
            )

        layers: list[nn.Module] = []
        input_width = self._lifted_width
        for _ in range(self.config.hidden_layers):
            layers.extend(
                (
                    nn.Linear(input_width, self.config.hidden_width),
                    nn.ReLU(),
                )
            )
            input_width = self.config.hidden_width

        self.trunk = nn.Sequential(*layers)
        self.value_head = nn.Linear(input_width, 1)
        self.play_value_head = (
            nn.Linear(input_width, 1) if self.config.play_value_head else None
        )
        self.drop_head = nn.Linear(input_width, self.config.action_count)
        self.check_head = nn.Linear(input_width, self.config.action_count)
        self.transition_class_head = (
            nn.Linear(input_width, TRANSITION_CLASS_COUNT)
            if self.config.transition_class_head
            else None
        )
        self.continuation_residual_head = (
            nn.Linear(input_width, self.config.action_count**2)
            if (
                self.config.continuation_residual
                and self.config.continuation_residual_mode == "matrix_head"
            )
            else None
        )
        self.continuation_action_hidden = (
            nn.Linear(input_width + 2, self.config.hidden_width)
            if (
                self.config.continuation_residual
                and self.config.continuation_residual_mode == "action_mlp"
            )
            else None
        )
        self.continuation_action_out = (
            nn.Linear(self.config.hidden_width, 1)
            if self.continuation_action_hidden is not None
            else None
        )
        if self.continuation_residual_head is not None:
            # Preserve the scalar Bellman model exactly at initialization.  The
            # residual only acquires a correction when exact matrix targets
            # justify one.
            nn.init.zeros_(self.continuation_residual_head.weight)
            nn.init.zeros_(self.continuation_residual_head.bias)
        if self.continuation_action_out is not None:
            # The action MLP is a genuine Q(s, d, c) model, but its final layer
            # starts at zero so migration preserves the scalar Bellman matrix.
            nn.init.zeros_(self.continuation_action_out.weight)
            nn.init.zeros_(self.continuation_action_out.bias)

    @property
    def _lifted_width(self) -> int:
        return len(FEATURE_SCHEMA) + (
            4 if self.config.feature_lift in {"boundary_v1", "boundary_v2"} else 0
        )

    @staticmethod
    def _boundary_features(features: Tensor) -> Tensor:
        """Return the four deterministic boundary coordinates for five inputs."""

        checker_st = features[:, 0] * 300.0
        checker_ttd = features[:, 1] * 300.0
        dropper_st = features[:, 2] * 300.0
        dropper_ttd = features[:, 3] * 300.0
        return torch.stack(
            (
                ((checker_st - 240.0) / 60.0).clamp_min(0.0),
                ((checker_st + checker_ttd - 240.0) / 300.0).clamp_min(0.0),
                ((dropper_st - 240.0) / 60.0).clamp_min(0.0),
                ((dropper_st + dropper_ttd - 240.0) / 300.0).clamp_min(0.0),
            ),
            dim=1,
        )

    @staticmethod
    def _boundary_v2_features(features: Tensor) -> Tensor:
        """Return exact indicators for the inclusive dose and strict TTD boundaries."""

        checker_st = features[:, 0] * 300.0
        checker_ttd = features[:, 1] * 300.0
        dropper_st = features[:, 2] * 300.0
        dropper_ttd = features[:, 3] * 300.0
        return torch.stack(
            (
                (checker_st >= 240.0).to(dtype=features.dtype),
                (checker_st + checker_ttd > 240.0).to(dtype=features.dtype),
                (dropper_st >= 240.0).to(dtype=features.dtype),
                (dropper_st + dropper_ttd > 240.0).to(dtype=features.dtype),
            ),
            dim=1,
        )

    def apply_feature_lift(self, features: Tensor) -> Tensor:
        """Apply the configured deterministic lift to the five external inputs."""

        if features.ndim != 2 or features.shape[1] != len(FEATURE_SCHEMA):
            raise ValueError(
                f"features must have shape (N, {len(FEATURE_SCHEMA)}), "
                f"got {tuple(features.shape)}"
            )
        if self.config.feature_lift == "identity":
            return features
        if self.config.feature_lift == "boundary_v1":
            return torch.cat((features, self._boundary_features(features)), dim=1)
        if self.config.feature_lift == "boundary_v2":
            return torch.cat((features, self._boundary_v2_features(features)), dim=1)
        raise ValueError(f"unknown feature lift {self.config.feature_lift!r}")

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        hidden = self.trunk(self.apply_feature_lift(features))
        value = torch.tanh(self.value_head(hidden)).squeeze(-1)
        return value, self.drop_head(hidden), self.check_head(hidden)

    def play_values(self, features: Tensor) -> Tensor:
        """Predict complete-game play values, distinct from finite-horizon values.

        Resolve-labeled rows carry depth-amplified complete-game estimates at
        the resolve's query horizon.  Those must not supervise ``value_head``,
        whose meaning is the exact value at the row's literal horizon, so play
        estimates get their own scalar head over the shared trunk.
        """

        if self.play_value_head is None:
            raise ValueError("play value head is disabled for this model")
        hidden = self.trunk(self.apply_feature_lift(features))
        return torch.tanh(self.play_value_head(hidden)).squeeze(-1)

    def transition_class_values(self, features: Tensor) -> Tensor:
        """Predict the 61 bounded continuation-class values in one pass.

        Indices ``0..59`` are the successful lags ``check - drop + 1`` in
        order; index ``60`` is the action-independent failed-check class.
        ``reconstruct_transition_class_matrix`` expands these to the literal
        stage matrix, so network error enters the matrix through exactly 61
        numbers and the whitepaper's matrix-accuracy saddle-gap bound applies
        to the quantity a class loss optimizes.
        """

        if self.transition_class_head is None:
            raise ValueError("transition class head is disabled for this model")
        hidden = self.trunk(self.apply_feature_lift(features))
        return torch.tanh(self.transition_class_head(hidden))

    def continuation_residual_matrix(self, features: Tensor) -> Tensor:
        """Return an action-conditioned correction to a Bellman root matrix."""

        if not self.config.continuation_residual:
            raise ValueError("continuation residual is disabled for this model")
        hidden = self.trunk(self.apply_feature_lift(features))
        if self.continuation_residual_head is not None:
            return self.continuation_residual_head(hidden).reshape(
                -1, self.config.action_count, self.config.action_count
            )
        if self.continuation_action_hidden is None or self.continuation_action_out is None:
            raise RuntimeError("action-conditioned continuation residual is incomplete")
        action_values = torch.arange(
            1,
            self.config.action_count + 1,
            dtype=features.dtype,
            device=features.device,
        ) / float(self.config.action_count)
        drop_actions, check_actions = torch.meshgrid(
            action_values,
            action_values,
            indexing="ij",
        )
        action_features = torch.stack(
            (drop_actions.reshape(-1), check_actions.reshape(-1)), dim=1
        )
        pair_count = int(action_features.shape[0])
        repeated_hidden = hidden.unsqueeze(1).expand(-1, pair_count, -1)
        repeated_actions = action_features.unsqueeze(0).expand(
            hidden.shape[0], -1, -1
        )
        inputs = torch.cat((repeated_hidden, repeated_actions), dim=-1)
        residual = self.continuation_action_out(
            torch.relu(self.continuation_action_hidden(inputs))
        ).squeeze(-1)
        return residual.reshape(
            -1, self.config.action_count, self.config.action_count
        )

    def encode(self, states: Tensor, horizons: Tensor) -> Tensor:
        return encode_features(
            states,
            horizons,
            horizon_scale=self.config.horizon_scale,
        )
