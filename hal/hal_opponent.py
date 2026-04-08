from __future__ import annotations

from random import Random

from src.Game import Game
from environment.opponents.base import Opponent
from environment.opponents.teacher_helpers import clamp_second

from .types import BeliefState, HalState, MemoryMode
from .buckets import get_buckets, resolve_bucket
from .search import search
from .belief import update_belief
from .memory import update_memory

DEFAULT_SEARCH_DEPTH = 3


class CanonicalHal(Opponent):

    def __init__(self, seed: int | None = None, depth: int = DEFAULT_SEARCH_DEPTH):
        self._state = HalState()
        self._rng = Random(seed)
        self._depth = depth

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

        strategy, _ = search(game, self._depth, self._state.belief, self._state.memory)

        if strategy is None:
            return clamp_second(turn_duration, role=role, turn_duration=turn_duration)

        hal_is_dropper = role == "dropper"
        effective_td = turn_duration
        if self._state.memory == MemoryMode.AMNESIA and turn_duration == 61:
            effective_td = 60

        hal_knows = self._state.memory != MemoryMode.AMNESIA and self._state.leap_deduced
        buckets = get_buckets(effective_td, knows_leap=hal_knows)

        chosen_idx = self._rng.choices(range(len(buckets)), weights=strategy.tolist(), k=1)[0]
        chosen_bucket = buckets[chosen_idx]
        second = resolve_bucket(chosen_bucket, self._rng)

        return clamp_second(second, role=role, turn_duration=turn_duration)

    def reset(self) -> None:
        self._state = HalState()
