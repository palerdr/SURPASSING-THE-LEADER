"""Public-history neural research components; no live deployment integration."""
from __future__ import annotations

from collections import deque

import numpy as np
import torch
from torch import nn

from arena.policies.translated_hal import TranslatedHalOpponentModel

WINDOW = 16
TOKEN_DIM = 16
EXPERTS = 24
GATE_DIM = TOKEN_DIM + 3 * EXPERTS


class SequencePredictor(nn.Module):
    """Predict the next opponent action from a bounded public-history window."""

    def __init__(self, kind="transformer", width=48):
        super().__init__()
        if kind not in ("transformer", "gru"):
            raise ValueError("unknown sequence architecture")
        self.kind = kind
        self.embed = nn.Linear(TOKEN_DIM, width)
        self.position = nn.Parameter(torch.zeros(WINDOW, width))
        if kind == "transformer":
            self.layers = nn.ModuleList([nn.TransformerEncoderLayer(width, 4, 2 * width,
                dropout=0, batch_first=True, norm_first=True) for _ in range(2)])
        else:
            self.gru = nn.GRU(width, width, num_layers=2, batch_first=True)
        self.head = nn.Linear(width, 60)

    def forward(self, window):
        x = torch.tanh(self.embed(window))
        if self.kind == "transformer":
            x = x + self.position[:x.shape[1]]
            mask = torch.ones(x.shape[1], x.shape[1], dtype=torch.bool, device=x.device).triu(1)
            for layer in self.layers:
                x = layer(x, src_mask=mask)
        else:
            x, _ = self.gru(x)
        return self.head(x[:, -1])


class ExpertSelector(nn.Module):
    """Learn a bounded correction to the frozen model's expert log weights."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(GATE_DIM, 64), nn.Tanh(), nn.Linear(64, EXPERTS))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features, experts):
        log_weights = features[:, TOKEN_DIM:TOKEN_DIM + EXPERTS] * 10
        weights = torch.softmax(log_weights + 4 * torch.tanh(self.net(features)), dim=-1)
        return torch.einsum("be,bea->ba", weights, experts)


class SessionActor(nn.Module):
    """Learn an action correction with a GRU and exact-matrix tactical input."""

    def __init__(self):
        super().__init__()
        self.gru = nn.GRUCell(TOKEN_DIM + 120, 64)
        self.actor = nn.Linear(64, 60)
        self.value = nn.Linear(64, 1)
        nn.init.zeros_(self.actor.weight)
        nn.init.zeros_(self.actor.bias)

    def forward(self, token, forecast, values, hidden=None):
        hidden = self.gru(torch.cat((token, forecast, values), dim=-1), hidden)
        logits = 12 * values + 4 * torch.tanh(self.actor(hidden))
        return logits, self.value(hidden).squeeze(-1), hidden


class PublicMemory:
    """Build inputs before a reveal and update independent opponent memory after it."""

    def __init__(self, config):
        self.base = TranslatedHalOpponentModel(config)
        self.tokens = deque(maxlen=WINDOW - 1)
        self.losses = {role: np.zeros(EXPERTS) for role in ("dropper", "checker")}
        self.last = None
        self.pending = None
        self.neural_log_odds = {role: -np.log(EXPERTS) for role in self.losses}

    def prepare(self, decision):
        if self.pending is not None:
            raise RuntimeError("a public reveal is pending")
        canonical = decision.canonical_decision
        if canonical.turn_duration != 60 or canonical.legal_seconds != tuple(range(1, 61)):
            raise ValueError("neural pilots require pure DTH actions 1..60")
        role = decision.opponent_role
        forecast = self.base.predict(role, state_regime=self.base.state_regime(canonical),
            game_index=decision.game_index, game_decision_index=decision.half_round_index)
        context = forecast.context
        token = np.zeros(TOKEN_DIM, dtype=np.float32)
        token[:6] = [role == "checker", *[x / 5 for x in context.state_regime], decision.half_round_index == 0]
        if self.last is not None:
            own, opponent, last_role = self.last
            favorable = opponent >= own if last_role == "checker" else opponent > own
            token[6:12] = [own / 60, opponent / 60, (opponent - own) / 59, favorable, last_role == "checker", 1]
        token[12] = np.log1p(self.base.total_observations) / 5
        if context.previous_self_action is not None:
            own, opponent = context.previous_self_action, context.previous_opponent_action
            token[13:] = [own / 60, opponent / 60, (opponent - own) / 59]
        window = np.zeros((WINDOW, TOKEN_DIM), dtype=np.float32)
        history = [*self.tokens, token]
        window[-len(history):] = history
        entropy = -(forecast.expert_policies * np.log(np.maximum(forecast.expert_policies, 1e-9))).sum(axis=1) / np.log(60)
        gate = np.concatenate((token, np.log(np.maximum(forecast.expert_weights, 1e-9)) / 10,
            self.losses[role] / 10, entropy)).astype(np.float32)
        self.pending = (forecast, token, None)
        return forecast, window, gate

    def predict(self, network, kind, forecast, window, gate, *, ablation=None, rng=None):
        if ablation is not None:
            window = window.copy()
            if ablation == "reset":
                window[:-1] = 0
            elif ablation == "shuffle":
                window[:-1] = window[:-1][rng.permutation(WINDOW - 1)]
            else:
                raise ValueError("unknown memory ablation")
        with torch.no_grad():
            if kind == "selector":
                raw = network(torch.tensor(gate[None]), torch.tensor(forecast.expert_policies[None], dtype=torch.float32)).numpy()[0]
                result, weight = raw, 1.
            else:
                raw = torch.softmax(network(torch.tensor(window[None])), -1).numpy()[0]
                role = forecast.context.opponent_role
                weight = .002 + .996 / (1 + np.exp(-np.clip(self.neural_log_odds[role], -40, 40)))
                result = (1 - weight) * forecast.policy + weight * raw
            result = np.maximum(np.asarray(result, dtype=float), 1e-12)
            result /= result.sum()
        self.pending = (forecast, self.pending[1], raw if kind != "selector" else None)
        return result, weight

    def observe(self, opponent_action, self_action):
        if self.pending is None:
            raise RuntimeError("no prediction awaits a reveal")
        forecast, token, neural = self.pending
        self.base.observe(forecast, opponent_action=opponent_action, self_action=self_action)
        role, index = forecast.context.opponent_role, opponent_action - 1
        self.losses[role] = .8 * self.losses[role] - .2 * np.log(np.maximum(forecast.expert_policies[:, index], 1e-9))
        if neural is not None:
            self.neural_log_odds[role] = .97 * self.neural_log_odds[role] + 1.25 * (
                np.log(max(float(neural[index]), 1e-9)) - np.log(max(float(forecast.policy[index]), 1e-9)))
        self.tokens.append(token.copy())
        self.last = (self_action, opponent_action, role)
        self.pending = None


def tensor(value):
    return torch.tensor(np.asarray(value), dtype=torch.float32).unsqueeze(0)


def make_network(kind):
    if kind in ("transformer", "gru"):
        return SequencePredictor(kind)
    if kind == "selector":
        return ExpertSelector()
    if kind in ("adversary", "probe", "myopic", "initial"):
        return SessionActor()
    raise ValueError(f"unknown network {kind}")
