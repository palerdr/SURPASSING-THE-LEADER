"""Tiny standalone value/policy network and evaluator adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch import nn

from stl.toy.rules import ToyRuleset
from stl.toy.state import ToyState


class ToyPolicyValueNet(nn.Module):
    def __init__(self, feature_dim: int, action_size: int, *, hidden_dim: int = 64):
        super().__init__()
        if feature_dim <= 0 or action_size <= 0 or hidden_dim <= 0:
            raise ValueError("network dimensions must be positive")
        self.feature_dim = int(feature_dim)
        self.action_size = int(action_size)
        self.hidden_dim = int(hidden_dim)
        self.trunk = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Tanh())
        self.drop_head = nn.Linear(hidden_dim, action_size)
        self.check_head = nn.Linear(hidden_dim, action_size)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"expected {self.feature_dim} features, got {features.shape[-1]}"
            )
        hidden = self.trunk(features)
        return self.value_head(hidden), self.drop_head(hidden), self.check_head(hidden)


def feature_dim(rules: ToyRuleset) -> int:
    return int(rules.encode_state(rules.initial_state(), rules.max_half_rounds).shape[0])


def _masked_softmax(logits: torch.Tensor, actions: tuple[int, ...], action_size: int) -> np.ndarray:
    values = logits.detach().cpu().numpy().astype(np.float64, copy=True)
    mask = np.zeros(action_size, dtype=bool)
    mask[np.asarray(actions, dtype=np.int64) - 1] = True
    values[~mask] = -np.inf
    finite = values[mask]
    shifted = np.exp(finite - np.max(finite))
    shifted /= shifted.sum()
    result = np.zeros(action_size, dtype=np.float64)
    result[mask] = shifted
    return result


def make_network_evaluator(
    model: ToyPolicyValueNet,
    rules: ToyRuleset,
    *,
    device: str = "cpu",
) -> Callable:
    model = model.to(device)
    model.eval()

    def evaluate(
        state: ToyState,
        remaining_horizon: int,
        active_rules: ToyRuleset,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        if active_rules.ruleset_id != rules.ruleset_id:
            raise ValueError("network evaluator ruleset does not match search rules")
        features = torch.as_tensor(
            rules.encode_state(state, remaining_horizon),
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        with torch.no_grad():
            value, drop_logits, check_logits = model(features)
        drop_actions = rules.legal_drop_actions(state)
        check_actions = rules.legal_check_actions(state)
        return (
            float(value[0, 0].item()),
            _masked_softmax(drop_logits[0], drop_actions, rules.action_size),
            _masked_softmax(check_logits[0], check_actions, rules.action_size),
        )

    return evaluate


@dataclass(frozen=True, slots=True)
class ToyModelConfig:
    feature_dim: int
    action_size: int
    hidden_dim: int = 64

    def to_dict(self) -> dict[str, int]:
        return {
            "feature_dim": self.feature_dim,
            "action_size": self.action_size,
            "hidden_dim": self.hidden_dim,
        }
