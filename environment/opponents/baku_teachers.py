"""Scripted Baku teachers used for bootstrap, demos, and evaluation.

These are not meant to be final solved agents. They are structured
teachers that populate the right route-building state distribution before
PPO and self-play take over.
"""

from __future__ import annotations

from src.Game import Game

from environment.strategy_features import build_strategy_snapshot

from .base import Opponent
from .baku_policy_helpers import choose_baku_checker_second, choose_baku_dropper_second
from .teacher_helpers import clamp_second, instant_check, instant_drop, late_drop, safe_check, target_st_drop


class BakuTeacher(Opponent):
    def choose_action(self, game: Game, role: str, turn_duration: int) -> int:
        snapshot = build_strategy_snapshot(game)
        if role == "checker":
            return clamp_second(
                choose_baku_checker_second(snapshot, turn_duration),
                role=role, turn_duration=turn_duration, actor="baku",
            )
        return clamp_second(
            choose_baku_dropper_second(snapshot, turn_duration),
            role=role, turn_duration=turn_duration, actor="baku",
        )


class BakuRouteBuilderTeacher(BakuTeacher):
    def choose_action(self, game: Game, role: str, turn_duration: int) -> int:
        snapshot = build_strategy_snapshot(game)
        if role == "checker":
            if not snapshot.active_lsr and snapshot.lsr_variation == 3 and snapshot.baku_budget.fail_post_ttd < 300.0:
                return instant_check(turn_duration)
            if snapshot.round_num == 0:
                return 30
        else:
            if snapshot.round_num == 2 and snapshot.current_half == 2:
                return target_st_drop(36, turn_duration, actor="baku")
        return super().choose_action(game, role, turn_duration)


class BakuActiveLsrPreserverTeacher(BakuTeacher):
    def choose_action(self, game: Game, role: str, turn_duration: int) -> int:
        snapshot = build_strategy_snapshot(game)
        if snapshot.active_lsr:
            if role == "checker":
                return safe_check(turn_duration)
            if snapshot.route_flags["round8_bridge"] or snapshot.route_flags["round9_pre_leap"]:
                return instant_drop(turn_duration, actor="baku")
        return super().choose_action(game, role, turn_duration)


class BakuLeapExecutorTeacher(BakuTeacher):
    def choose_action(self, game: Game, role: str, turn_duration: int) -> int:
        snapshot = build_strategy_snapshot(game)
        if snapshot.is_leap_turn:
            if role == "dropper":
                return turn_duration
            return 60
        if snapshot.route_flags["round9_pre_leap"]:
            if role == "checker":
                return safe_check(turn_duration)
            return late_drop(turn_duration, second=60, actor="baku")
        return super().choose_action(game, role, turn_duration)


class BakuAntiEcholocationTeacher(BakuTeacher):
    def choose_action(self, game: Game, role: str, turn_duration: int) -> int:
        snapshot = build_strategy_snapshot(game)
        if role == "dropper" and snapshot.round_num == 6 and snapshot.current_half == 2:
            return clamp_second(10, role=role, turn_duration=turn_duration, actor="baku")
        if role == "dropper" and snapshot.round_num == 7 and snapshot.current_half == 2:
            return late_drop(turn_duration, second=60, actor="baku")
        return super().choose_action(game, role, turn_duration)


class BakuResilienceFallbackTeacher(BakuTeacher):
    def choose_action(self, game: Game, role: str, turn_duration: int) -> int:
        snapshot = build_strategy_snapshot(game)
        if role == "checker" and snapshot.baku_budget.fail_post_ttd >= 300.0:
            return safe_check(turn_duration)
        if role == "dropper" and snapshot.hal_budget.fail_post_ttd < 298.0 and snapshot.active_lsr:
            return instant_drop(turn_duration, actor="baku")
        return super().choose_action(game, role, turn_duration)
