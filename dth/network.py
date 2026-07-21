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

FEATURE_LIFTS = ("identity", "boundary_v1")


@dataclass(frozen=True)
class DTHNetworkConfig:
    hidden_width: int = 64
    hidden_layers: int = 2
    action_count: int = 60
    horizon_scale: float = 3.0
    feature_lift: str = "identity"

    def to_dict(self) -> dict[str, int | float | str]:
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
        self.drop_head = nn.Linear(input_width, self.config.action_count)
        self.check_head = nn.Linear(input_width, self.config.action_count)

    @property
    def _lifted_width(self) -> int:
        return len(FEATURE_SCHEMA) + (4 if self.config.feature_lift == "boundary_v1" else 0)

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
        raise ValueError(f"unknown feature lift {self.config.feature_lift!r}")

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        hidden = self.trunk(self.apply_feature_lift(features))
        value = torch.tanh(self.value_head(hidden)).squeeze(-1)
        return value, self.drop_head(hidden), self.check_head(hidden)

    def encode(self, states: Tensor, horizons: Tensor) -> Tensor:
        return encode_features(
            states,
            horizons,
            horizon_scale=self.config.horizon_scale,
        )
