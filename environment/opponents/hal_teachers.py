"""Scripted Hal teachers used for bootstrap, demos, and regression checks.

Each class emphasizes one strategic motif from the docs while sharing the
same route-aware snapshot interface and timing helpers.
"""

from __future__ import annotations

from src.Game import Game

from environment.strategy_features import build_strategy_snapshot

from .base import Opponent
from .hal_policy_helpers import choose_hal_checker_second, choose_hal_dropper_second
from .teacher_helpers import clamp_second, death_probe_check, instant_check, instant_drop, safe_check, target_st_drop


class HalTeacher(Opponent):
    def choose_action(self, game: Game, role: str, turn_duration: int) -> int:
        snapshot = build_strategy_snapshot(game)
        if role == "checker":
            return clamp_second(choose_hal_checker_second(snapshot, turn_duration), role=role, turn_duration=turn_duration)
        return clamp_second(choose_hal_dropper_second(snapshot, turn_duration), role=role, turn_duration=turn_duration)


class HalLeapInferenceTeacher(HalTeacher):
    def choose_action(self, game: Game, role: str, turn_duration: int) -> int:
        snapshot = build_strategy_snapshot(game)
        if role == "checker" and snapshot.round_num == 0 and snapshot.current_half == 2:
            return 24
        return super().choose_action(game, role, turn_duration)


class HalDeathTradeTeacher(HalTeacher):
    def choose_action(self, game: Game, role: str, turn_duration: int) -> int:
        snapshot = build_strategy_snapshot(game)
        if role == "checker" and not snapshot.active_lsr and snapshot.lsr_variation in (3, 4):
            if 20.0 <= snapshot.hal_budget.cylinder <= 40.0 or snapshot.round_num == 1:
                return death_probe_check(turn_duration)
        return super().choose_action(game, role, turn_duration)


class HalPressureTeacher(HalTeacher):
    def choose_action(self, game: Game, role: str, turn_duration: int) -> int:
        snapshot = build_strategy_snapshot(game)
        if role == "dropper" and snapshot.active_lsr:
            if snapshot.route_flags["round7_pressure"]:
                return instant_drop(turn_duration)
            if snapshot.route_flags["round8_bridge"]:
                return target_st_drop(55, turn_duration)
        return super().choose_action(game, role, turn_duration)


class HalDeviationTeacher(HalTeacher):
    def choose_action(self, game: Game, role: str, turn_duration: int) -> int:
        snapshot = build_strategy_snapshot(game)
        if role == "checker":
            if snapshot.round_num == 3 and snapshot.current_half == 2:
                return 34
            if snapshot.round_num == 4 and snapshot.current_half == 2:
                return 15
            if snapshot.round_num == 5 and snapshot.current_half == 2:
                return 8
            if snapshot.round_num == 6 and snapshot.current_half == 2:
                return instant_check(turn_duration)
        return super().choose_action(game, role, turn_duration)


class HalEcholocationTeacher(HalTeacher):
    def choose_action(self, game: Game, role: str, turn_duration: int) -> int:
        snapshot = build_strategy_snapshot(game)
        if role == "checker" and snapshot.round_num in (3, 4, 5):
            mapping = {3: 34, 4: 15, 5: 8}
            return mapping[snapshot.round_num]
        return super().choose_action(game, role, turn_duration)


class HalMemoryLossTeacher(HalTeacher):
    def choose_action(self, game: Game, role: str, turn_duration: int) -> int:
        snapshot = build_strategy_snapshot(game)
        if role == "checker" and snapshot.round_num == 7 and snapshot.current_half == 2:
            return instant_check(turn_duration)
        if role == "checker" and snapshot.is_leap_turn:
            return 60
        return super().choose_action(game, role, turn_duration)


class HalResilienceTeacher(HalTeacher):
    def choose_action(self, game: Game, role: str, turn_duration: int) -> int:
        snapshot = build_strategy_snapshot(game)
        if role == "dropper" and snapshot.active_lsr and snapshot.route_flags["round7_pressure"]:
            return instant_drop(turn_duration)
        if role == "checker" and snapshot.hal_budget.fail_post_ttd < 299.0:
            return safe_check(turn_duration)
        return super().choose_action(game, role, turn_duration)
