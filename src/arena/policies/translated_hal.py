"""Old Hal with role-specific forecasts of translated public responses."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np

from arena.policies.perfect_hal import (
    PerfectHalConfig, PerfectHalContext, PerfectHalOpponentModel,
)


@dataclass(frozen=True, slots=True)
class TranslatedHalConfig(PerfectHalConfig):
    offset_retention: float = 0.9
    translated: bool = True

    def __post_init__(self):
        super(TranslatedHalConfig, self).__post_init__()
        if not np.isfinite(self.offset_retention) or not 0 < self.offset_retention <= 1:
            raise ValueError("offset_retention must be in (0, 1]")
        if not isinstance(self.translated, bool):
            raise ValueError("translated must be boolean")
        if self.expert_weight_floor * len(self.expert_names) >= 1:
            raise ValueError("expert floor leaves no probability mass")

    @property
    def expert_names(self):
        base = super(TranslatedHalConfig, self).expert_names
        return base + (tuple(f"offset_{anchor}_{direction}_{rate}"
            for anchor in ("role", "last") for direction in ("copy", "mirror")
            for rate in ("global", "recent")) if self.translated else ())


@dataclass(frozen=True, slots=True)
class TranslatedContext(PerfectHalContext):
    last_self_action: int | None = None


class TranslatedHalOpponentModel(PerfectHalOpponentModel):
    """Score Old's experts and offset experts before each public reveal."""

    schema = "arena-translated-hal-v1"

    def __init__(self, config=TranslatedHalConfig()):
        super().__init__(config)
        base = {key: value for key, value in asdict(config).items()
                if key in PerfectHalConfig.__dataclass_fields__}
        self._base = PerfectHalOpponentModel(PerfectHalConfig(**base))
        self._offsets = {role: np.zeros((2, 2, 2, 119)) for role in self._roles}
        self._last_self = None

    def reset(self):
        super().reset()
        self._offsets = {role: np.zeros((2, 2, 2, 119)) for role in self._roles}
        self._last_self = None

    def break_sequence(self):
        """Keep learned counts; clear references after an excluded leap reveal."""
        self._last_self = None
        for state in self._roles.values():
            state.previous_self_action = None
            state.previous_opponent_action = None

    def _expert_policies(self, state, context):
        base = self._base._expert_policies(state, context)
        if not self.config.translated:
            return base
        policies = list(base)
        prior = base[1]
        offsets = self._offsets[context.opponent_role]
        for anchor_index, anchor in enumerate((context.previous_self_action, self._last_self)):
            for mirrored in range(2):
                for rate in range(2):
                    if anchor is None:
                        policies.append(prior)
                        continue
                    reference = 61 - anchor if mirrored else anchor
                    actions = np.clip(reference + np.arange(-59, 60), 1, 60) - 1
                    counts = np.bincount(actions, weights=offsets[anchor_index, mirrored, rate], minlength=60)
                    policies.append(self._predictive(counts, prior,
                        strength=self.config.conditional_prior_strength))
        return np.stack(policies)

    def predict(self, *args, **kwargs):
        forecast = super().predict(*args, **kwargs)
        context = TranslatedContext(**asdict(forecast.context), last_self_action=self._last_self)
        return replace(forecast, context=context)

    def observe(self, forecast, *, opponent_action, self_action):
        super().observe(forecast, opponent_action=opponent_action, self_action=self_action)
        if self.config.translated:
            offsets = self._offsets[forecast.context.opponent_role]
            anchors = (forecast.context.previous_self_action, forecast.context.last_self_action)
            for anchor_index, anchor in enumerate(anchors):
                if anchor is None:
                    continue
                for mirrored in range(2):
                    reference = 61 - anchor if mirrored else anchor
                    offset = opponent_action - reference + 59
                    offsets[anchor_index, mirrored, 1] *= self.config.offset_retention
                    offsets[anchor_index, mirrored, :, offset] += 1
        self._last_self = int(self_action)
