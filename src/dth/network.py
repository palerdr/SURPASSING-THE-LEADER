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


@dataclass(frozen=True)
class DTHNetworkConfig:
    hidden_width: int = 64
    hidden_layers: int = 2
    action_count: int = 60
    horizon_scale: float = 3.0

    def to_dict(self) -> dict[str, int | float]:
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

        layers: list[nn.Module] = []
        input_width = len(FEATURE_SCHEMA)
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

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if features.ndim != 2 or features.shape[1] != len(FEATURE_SCHEMA):
            raise ValueError(
                f"features must have shape (N, {len(FEATURE_SCHEMA)}), "
                f"got {tuple(features.shape)}"
            )
        hidden = self.trunk(features)
        value = torch.tanh(self.value_head(hidden)).squeeze(-1)
        return value, self.drop_head(hidden), self.check_head(hidden)

    def encode(self, states: Tensor, horizons: Tensor) -> Tensor:
        return encode_features(
            states,
            horizons,
            horizon_scale=self.config.horizon_scale,
        )
