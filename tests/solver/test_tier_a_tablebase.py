import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.getcwd())

from stl.engine.game import PHYSICALITY_BAKU, PHYSICALITY_HAL
from stl.engine.game import Game
from stl.engine.game import Player
from stl.engine.game import Referee
from stl.learning.strength import best_response_interval, uniform_policy
from stl.solver.tablebase import TierAEvaluator, TierALookup, frontier_interval_fn


TIER_A_DIR = os.path.join(
    os.getcwd(),
    "checkpoints",
    "tablebase",
    "tier_a",
)


def make_game(
    *,
    clock=3661.0,
    half=1,
    hal_cyl=0.0,
    baku_cyl=0.0,
    hal_deaths=0,
    baku_deaths=0,
    hal_ttd=0.0,
    baku_ttd=0.0,
) -> Game:
    hal = Player(name="Hal", physicality=PHYSICALITY_HAL)
    baku = Player(name="Baku", physicality=PHYSICALITY_BAKU)
    game = Game(player1=hal, player2=baku, referee=Referee(), first_dropper=hal)
    game.seed(0)
    game.game_clock = float(clock)
    game.current_half = int(half)
    hal.cylinder = float(hal_cyl)
    baku.cylinder = float(baku_cyl)
    hal.deaths = int(hal_deaths)
    baku.deaths = int(baku_deaths)
    hal.ttd = float(hal_ttd)
    baku.ttd = float(baku_ttd)
    game.referee.cprs_performed = int(hal_deaths + baku_deaths)
    return game


def require_tier_a() -> TierALookup:
    if not os.path.exists(os.path.join(TIER_A_DIR, "manifest.json")):
        pytest.skip("Tier A artifact manifest absent")
    return TierALookup(TIER_A_DIR)


def test_tier_a_manifest_verifies():
    lookup = require_tier_a()
    manifest = lookup.verify_manifest()
    assert len(manifest) == 481
    assert "d0.npz" in manifest
    assert "d1_hal_120.npz" in manifest
    assert "d1_baku_120.npz" in manifest


def test_tier_a_lookup_hits_postleap_d0_and_rejects_preleap():
    lookup = require_tier_a()

    hit = lookup.lookup(make_game(hal_cyl=240, baku_cyl=240))
    assert hit.interval is not None
    assert hit.interval.source == "d0.npz"
    assert hit.interval.width == pytest.approx(0.0)

    miss = lookup.lookup(make_game(clock=3300.0, hal_cyl=240, baku_cyl=240))
    assert miss.interval is None
    assert miss.miss_reason == "not_post_leap"


def test_tier_a_lookup_hits_one_death_epochs():
    lookup = require_tier_a()

    hal = lookup.lookup(make_game(half=2, hal_cyl=230, baku_cyl=230, hal_deaths=1, hal_ttd=120))
    baku = lookup.lookup(make_game(half=1, hal_cyl=230, baku_cyl=230, baku_deaths=1, baku_ttd=120))

    assert hal.interval is not None
    assert hal.interval.source == "d1_hal_120.npz"
    assert baku.interval is not None
    assert baku.interval.source == "d1_baku_120.npz"


def test_tier_a_evaluator_uses_low_width_and_falls_back_on_wide_interval():
    lookup = require_tier_a()
    calls = {"count": 0}
    drop = np.zeros(61, dtype=np.float64)
    check = np.zeros(61, dtype=np.float64)
    drop[0] = 1.0
    check[1] = 1.0

    def fallback(game):
        calls["count"] += 1
        return 0.25, drop, check

    evaluator = TierAEvaluator(fallback, lookup=lookup, max_width=0.05)

    exact_value, exact_drop, exact_check = evaluator(make_game(hal_cyl=240, baku_cyl=240))
    assert exact_value != pytest.approx(0.25)
    assert exact_drop[0] == pytest.approx(1.0)
    assert exact_check[1] == pytest.approx(1.0)
    assert calls["count"] == 1

    wide_value, _, _ = evaluator(make_game())
    assert wide_value == pytest.approx(0.25)
    assert calls["count"] == 2
    assert evaluator.misses["wide_interval"] == 1


def test_tier_a_evaluator_falls_back_when_artifacts_are_absent(tmp_path):
    lookup = TierALookup(tmp_path)
    calls = {"count": 0}

    def fallback(game):
        calls["count"] += 1
        return -0.125

    evaluator = TierAEvaluator(fallback, lookup=lookup, max_width=0.05)
    value, _, _ = evaluator(make_game(hal_cyl=240, baku_cyl=240))
    assert value == pytest.approx(-0.125)
    assert calls["count"] == 1
    assert evaluator.misses["artifact_missing"] == 1


def test_tier_a_miss_matches_structured_fallback_exactly():
    lookup = require_tier_a()
    drop = np.zeros(61, dtype=np.float64)
    check = np.zeros(61, dtype=np.float64)
    drop[2] = 0.25
    drop[59] = 0.75
    check[0] = 0.4
    check[10] = 0.6

    def fallback(game):
        return 0.125, drop, check

    game = make_game(clock=3300.0, hal_cyl=240, baku_cyl=240)
    expected = fallback(game)
    actual = TierAEvaluator(fallback, lookup=lookup, max_width=0.01)(game)

    assert actual[0] == pytest.approx(expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
    np.testing.assert_array_equal(actual[2], expected[2])


def test_best_response_frontier_can_use_tier_a_interval():
    lookup = require_tier_a()
    game = make_game(hal_cyl=240, baku_cyl=240)

    plain = best_response_interval(game, uniform_policy(), depth=0)
    tier = best_response_interval(
        game,
        uniform_policy(),
        depth=0,
        frontier_fn=frontier_interval_fn(lookup),
    )

    assert plain.width == pytest.approx(2.0)
    assert tier.width == pytest.approx(0.0)
    assert tier.tablebase_frontier_hits == 1
