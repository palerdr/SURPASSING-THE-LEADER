"""Research-only transfer of reward-conditioned repetition priors to Hal."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from arena.policies.translated_hal import TranslatedHalConfig, TranslatedHalOpponentModel


def transfer_prior(binary_prior, action_count=60):
    """Preserve repetition odds relative to uniform chance and total strength."""
    prior = np.asarray(binary_prior, dtype=float)
    if prior.shape != (2, 2) or not np.isfinite(prior).all() or (prior <= 0).any():
        raise ValueError("prior must contain positive finite repeat/switch counts for two outcomes")
    if isinstance(action_count, bool) or not isinstance(action_count, int) or action_count < 2:
        raise ValueError("action_count must be an integer of at least two")
    strength = prior.sum(axis=1)
    mean = prior[:, 0] / (prior[:, 0] + (action_count - 1) * prior[:, 1])
    return np.column_stack((mean, 1 - mean)) * strength[:, None]


@dataclass(frozen=True, slots=True)
class RewardPriorConfig(TranslatedHalConfig):
    @property
    def expert_names(self):
        return super(RewardPriorConfig, self).expert_names + ("reward_repeat",)


class RewardPriorHal(TranslatedHalOpponentModel):
    """Add one scored expert; retain the frozen translated expert updates."""

    schema = "arena-reward-prior-research-v1"

    def __init__(self, config, binary_prior):
        self.repeat_prior = transfer_prior(binary_prior)
        self.repeat_prior.setflags(write=False)
        super().__init__(config)
        self._reset_rewards()

    def _reset_rewards(self):
        self._repeat_counts = {role: np.zeros((2, 2)) for role in self._roles}
        self._outcomes = {role: None for role in self._roles}

    def reset(self):
        super().reset()
        self._reset_rewards()

    def break_sequence(self):
        super().break_sequence()
        self._outcomes = {role: None for role in self._roles}

    def _expert_policies(self, state, context):
        policies = super()._expert_policies(state, context)
        outcome = self._outcomes[context.opponent_role]
        previous = context.previous_opponent_action
        prediction = policies[1].copy()
        if previous is not None and outcome is not None:
            counts = self.repeat_prior[outcome] + self._repeat_counts[context.opponent_role][outcome]
            repeat = counts[0] / counts.sum()
            prediction[previous - 1] = 0
            prediction *= (1 - repeat) / prediction.sum()
            prediction[previous - 1] = repeat
        return np.vstack((policies, prediction))

    def observe(self, forecast, *, opponent_action, self_action):
        # The parent validates the reveal and token before we mutate reward memory.
        super().observe(forecast, opponent_action=opponent_action, self_action=self_action)
        role = forecast.context.opponent_role
        outcome = self._outcomes[role]
        previous = forecast.context.previous_opponent_action
        if previous is not None and outcome is not None:
            self._repeat_counts[role][outcome, int(opponent_action != previous)] += 1
        favorable = opponent_action >= self_action if role == "checker" else opponent_action > self_action
        self._outcomes[role] = int(favorable)
