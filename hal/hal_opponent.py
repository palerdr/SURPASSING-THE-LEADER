from __future__ import annotations

from random import Random

from src.Game import Game
from environment.opponents.base import Opponent
from environment.legal_actions import clamp_action

from .state import HalState, MemoryMode, update_belief, update_memory
from .action_model import get_legal_buckets, resolve_bucket
from .search import search, adaptive_depth

DEFAULT_SEARCH_DEPTH = 3


class CanonicalHal(Opponent):

    def __init__(self, seed: int | None = None, depth: int = DEFAULT_SEARCH_DEPTH, use_adaptive: bool = True):
        self._state = HalState()
        self._rng = Random(seed)
        self._base_depth = depth
        self._use_adaptive = use_adaptive

    def _hal_leap_flags(self) -> tuple[bool, bool]:
        hal_leap_deduced = self._state.leap_deduced
        hal_memory_impaired = self._state.memory == MemoryMode.AMNESIA
        return hal_leap_deduced, hal_memory_impaired

    def choose_action(self, game: Game, role: str, turn_duration: int) -> int:
        last_record = game.history[-1] if game.history else None

        new_belief = self._state.belief
        if last_record is not None:
            new_belief = update_belief(new_belief, last_record)

        death_occurred = last_record is not None and last_record.survived is not None
        new_memory = update_memory(self._state.memory, game, death_occurred)

        leap_deduced = self._state.leap_deduced
        if not leap_deduced and death_occurred:
            leap_deduced = True

        self._state = HalState(
            memory=new_memory,
            belief=new_belief,
            leap_deduced=leap_deduced,
        )

        depth = adaptive_depth(self._base_depth, game) if self._use_adaptive else self._base_depth
        strategy, _ = search(
            game, depth, self._state.belief, self._state.memory, leap_deduced,
        )

        hal_leap_deduced, hal_memory_impaired = self._hal_leap_flags()

        if strategy is None:
            return clamp_action(
                turn_duration, actor="hal", role=role, turn_duration=turn_duration,
                hal_leap_deduced=hal_leap_deduced, hal_memory_impaired=hal_memory_impaired,
            )

        buckets = get_legal_buckets(
            "hal", role, turn_duration,
            hal_leap_deduced=hal_leap_deduced,
            hal_memory_impaired=hal_memory_impaired,
        )

        weights = strategy.tolist()
        if len(weights) != len(buckets):
            weights = weights[:len(buckets)]
            total = sum(weights)
            if total > 0:
                weights = [w / total for w in weights]
            else:
                weights = [1.0 / len(buckets)] * len(buckets)

        chosen_idx = self._rng.choices(range(len(buckets)), weights=weights, k=1)[0]
        chosen_bucket = buckets[chosen_idx]
        second = resolve_bucket(chosen_bucket, self._rng)

        return clamp_action(
            second, actor="hal", role=role, turn_duration=turn_duration,
            hal_leap_deduced=hal_leap_deduced, hal_memory_impaired=hal_memory_impaired,
        )

    def reset(self) -> None:
        self._state = HalState()
