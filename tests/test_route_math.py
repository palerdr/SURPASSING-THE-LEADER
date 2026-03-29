import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.dth_env import DTHEnv
from environment.opponents.random_bot import RandomBot
from environment.route_math import (
    classify_round_gap_seconds,
    current_lsr_variation,
    is_active_lsr,
    projected_round_gap_for_death_duration,
    projected_round_gap_for_two_deaths,
    safe_strategy_budget,
    strict_next_minute,
)
from environment.strategy_features import build_strategy_snapshot
from training.curriculum import get_scenario


def make_env(**kwargs) -> DTHEnv:
    return DTHEnv(opponent=RandomBot(), agent_role="hal", seed=123, **kwargs)


def test_strict_next_minute_uses_strict_ceiling():
    assert strict_next_minute(180.0) == 240
    assert strict_next_minute(181.0) == 240
    assert strict_next_minute(239.0) == 240


def test_projected_round_gap_for_death_duration_matches_doc_classes():
    assert projected_round_gap_for_death_duration(60.0) == 420
    assert projected_round_gap_for_death_duration(84.0) == 420
    assert projected_round_gap_for_death_duration(176.0) == 480


def test_projected_round_gap_for_two_deaths_matches_ten_minute_case():
    assert projected_round_gap_for_two_deaths(60.0, 60.0) == 600


def test_round_gap_classification_matches_bridge_language():
    assert classify_round_gap_seconds(240) == "clean_4m"
    assert classify_round_gap_seconds(420) == "death_7m"
    assert classify_round_gap_seconds(480) == "heavy_8m"
    assert classify_round_gap_seconds(600) == "double_or_larger"


def test_lsr_variation_and_active_lsr_match_round7_scenario():
    env = make_env()
    env.reset(options={"scenario": get_scenario("round7_pressure")})

    assert current_lsr_variation(env.game) == 2
    assert is_active_lsr(env.game) is True


def test_strategy_snapshot_exposes_budget_features():
    env = make_env()
    env.reset(options={"scenario": get_scenario("round7_pressure")})

    snapshot = build_strategy_snapshot(env.game)

    assert snapshot.active_lsr is True
    assert snapshot.baku_budget.safe_budget == safe_strategy_budget(env.game.player2)
    assert snapshot.hal_budget.fail_post_ttd > snapshot.hal_budget.ttd
