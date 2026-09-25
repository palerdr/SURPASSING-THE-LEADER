"""Baseline Hal providers and the fitted human opponent emulators."""

from __future__ import annotations

import numpy as np

from arena.dth_adapter import project_to_dth_state
from arena.policies.bayesian_hal import ROLES
from hal_lab.harness.human_data import human_observation


class ExactProvider:
    def __init__(self, agent):
        self.agent = agent

    def policy(self, decision):
        move = self.agent.decide(project_to_dth_state(decision))
        policy = move.drop_policy if decision.role == "dropper" else move.check_policy
        return {i + 1: float(p) for i, p in enumerate(policy) if p > 0}


class UniformProvider:
    def policy(self, decision):
        return {i: 1 / 60 for i in range(1, 61)}


class ResetMemoryProvider:
    def __init__(self, provider):
        self.provider = provider

    def policy(self, decision):
        self.provider.opponent_model.reset()
        return self.provider.policy(decision)

    def __getattr__(self, name):
        return getattr(self.provider, name)


class FittedHumanOpponent:
    """Independent smoothed categorical/response emulator fit to early games."""

    def __init__(self, games, *, reactive, pseudocount=0.5):
        self.reactive = reactive
        self.counts = {r: np.full(60, pseudocount) for r in ROLES}
        self.responses = {r: {} for r in ROLES}
        previous = {r: None for r in ROLES}
        for game in games:
            for move in game["public_history"]:
                obs = human_observation(move)
                if obs is None:
                    previous = {r: None for r in ROLES}
                    continue
                _, role, action, hal_action = obs
                self.counts[role][action - 1] += 1
                if previous[role] is not None:
                    self.responses[role].setdefault(previous[role], np.zeros(60))[action - 1] += 1
                previous[role] = hal_action
        self.previous = {r: None for r in ROLES}

    def policy(self, decision):
        self.name = decision.actor_name
        role = decision.role
        prior = self.counts[role] / self.counts[role].sum()
        counts = self.responses[role].get(self.previous[role]) if self.reactive else None
        policy = prior if counts is None else (counts + 5 * prior) / (counts.sum() + 5)
        return {i + 1: float(p) for i, p in enumerate(policy)}

    def observe(self, record):
        if record.dropper_name == self.name:
            self.previous["dropper"] = record.check_time
        else:
            self.previous["checker"] = record.drop_time
