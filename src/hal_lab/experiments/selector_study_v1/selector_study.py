"""Selector architectures and retrained feature ablations for pure-DTH research."""
from __future__ import annotations

import torch
from torch import nn

from hal_lab.experiments.neural_pilots_v1.neural_pilots import EXPERTS, GATE_DIM, TOKEN_DIM


class Selector(nn.Module):
    """Adjust frozen expert weights through an explicit, masked input contract."""

    def __init__(self, architecture="small", ablation="none"):
        super().__init__()
        if architecture not in ("small", "wide", "attention", "static"):
            raise ValueError("unknown selector architecture")
        if ablation not in ("none", "no_errors", "no_context", "no_prior", "no_shapes"):
            raise ValueError("unknown selector ablation")
        if ablation == "no_shapes" and architecture != "attention":
            raise ValueError("forecast-shape ablation requires attention")
        self.architecture, self.ablation = architecture, ablation
        if architecture == "small":
            self.body = nn.Sequential(nn.Linear(GATE_DIM, 64), nn.Tanh())
            self.head = nn.Linear(64, EXPERTS)
        elif architecture == "wide":
            self.input = nn.Linear(GATE_DIM, 256)
            self.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 256), nn.GELU()) for _ in range(2)])
            self.head = nn.Linear(256, EXPERTS)
        elif architecture == "attention":
            self.input = nn.Linear(TOKEN_DIM + 3 + 60, 48)
            self.identities = nn.Parameter(torch.randn(EXPERTS, 48) * .02)
            self.norm1, self.norm2 = nn.LayerNorm(48), nn.LayerNorm(48)
            self.attention = nn.MultiheadAttention(48, 4, dropout=0, batch_first=True)
            self.feedforward = nn.Sequential(nn.Linear(48, 96), nn.GELU(), nn.Linear(96, 48))
            self.head = nn.Linear(48, 1)
        else:
            self.bias = nn.Parameter(torch.zeros(EXPERTS))
        if architecture != "static":
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def weights(self, features, experts):
        features = features.clone()
        if self.ablation == "no_context":
            features[:, :TOKEN_DIM] = 0
        if self.ablation == "no_errors":
            features[:, TOKEN_DIM + EXPERTS:TOKEN_DIM + 2 * EXPERTS] = 0
        if self.ablation == "no_prior":
            features[:, TOKEN_DIM:TOKEN_DIM + EXPERTS] = 0
        inherited = features[:, TOKEN_DIM:TOKEN_DIM + EXPERTS] * 10
        if self.architecture == "small":
            correction = self.head(self.body(features))
        elif self.architecture == "wide":
            x = torch.nn.functional.gelu(self.input(features))
            for block in self.blocks:
                x = x + block(x)
            correction = self.head(x)
        elif self.architecture == "attention":
            summary = features[:, TOKEN_DIM:].reshape(-1, 3, EXPERTS).transpose(1, 2)
            context = features[:, None, :TOKEN_DIM].expand(-1, EXPERTS, -1)
            shapes = torch.zeros_like(experts) if self.ablation == "no_shapes" else experts
            x = self.input(torch.cat((context, summary, shapes), dim=-1)) + self.identities
            normalized = self.norm1(x)
            x = x + self.attention(normalized, normalized, normalized, need_weights=False)[0]
            x = x + self.feedforward(self.norm2(x))
            correction = self.head(x).squeeze(-1)
        else:
            correction = self.bias.expand(features.shape[0], -1)
        return torch.softmax(inherited + 4 * torch.tanh(correction), dim=-1)

    def forward(self, features, experts):
        return torch.einsum("be,bea->ba", self.weights(features, experts), experts)


class SelectorEnsemble(nn.Module):
    """Average all registered training seeds without selecting the best seed."""

    def __init__(self, members):
        super().__init__()
        if not members:
            raise ValueError("ensemble needs members")
        self.members = nn.ModuleList(members)

    def weights(self, features, experts):
        return torch.stack([m.weights(features, experts) for m in self.members]).mean(0)

    def forward(self, features, experts):
        return torch.einsum("be,bea->ba", self.weights(features, experts), experts)
