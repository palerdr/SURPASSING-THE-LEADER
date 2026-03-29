"""Tests for death, revival, and survival probability mechanics."""

import pytest
from src.Player import Player
from src.Referee import Referee
from src.Game import Game, HalfRoundResult
from src.Constants import CYLINDER_MAX, PHYSICALITY_HAL, PHYSICALITY_BAKU


class TestSurvivalProbability:
    def test_first_death_60s_high_survival(self, referee, hal):
        """First death at 60s: base_curve is generous, no cardiac/fatigue penalty."""
        prob = referee.compute_survival_probability(hal, death_duration=60)
        # base_curve(60) = 1 - (60/300)^3 = 1 - 0.008 = 0.992
        # cardiac(0) = 0.85^0 = 1.0
        # referee(0) = max(0.4, 0.88^0) = 1.0
        # physicality = 1.0
        # P = 0.992
        assert prob == pytest.approx(0.992, abs=0.01)

    def test_death_at_300_is_always_fatal(self, referee, hal):
        """Cylinder at 300 = guaranteed death regardless of other factors."""
        prob = referee.compute_survival_probability(hal, death_duration=300)
        assert prob == 0.0

    def test_death_above_300_is_fatal(self, referee, hal):
        prob = referee.compute_survival_probability(hal, death_duration=350)
        assert prob == 0.0

    def test_cardiac_degradation(self, referee, hal):
        """Prior deaths weaken the heart, reducing survival probability."""
        prob_fresh = referee.compute_survival_probability(hal, death_duration=60)

        # Simulate a prior death
        hal.on_death(death_duration=60)
        hal.on_revival()
        prob_damaged = referee.compute_survival_probability(hal, death_duration=60)

        assert prob_damaged < prob_fresh

    def test_referee_fatigue(self):
        """Each CPR degrades the referee's effectiveness."""
        ref = Referee()
        p = Player(name="Test", physicality=1.0)

        prob_fresh = ref.compute_survival_probability(p, death_duration=60)
        ref.cprs_performed = 3
        prob_tired = ref.compute_survival_probability(p, death_duration=60)

        assert prob_tired < prob_fresh

    def test_baku_lower_physicality(self, referee):
        """Baku has strictly lower survival than Hal for the same death."""
        hal = Player(name="Hal", physicality=PHYSICALITY_HAL)
        baku = Player(name="Baku", physicality=PHYSICALITY_BAKU)

        p_hal = referee.compute_survival_probability(hal, death_duration=60)
        p_baku = referee.compute_survival_probability(baku, death_duration=60)

        assert p_baku < p_hal
        assert p_baku == pytest.approx(p_hal * PHYSICALITY_BAKU, abs=0.001)


class TestDeathSequence:
    def test_failed_check_triggers_death(self, game):
        rec = game.play_half_round(drop_time=50, check_time=10)
        assert rec.death_duration > 0
        assert rec.survived is not None

    def test_survived_death_resets_cylinder(self, game):
        """If player survives, cylinder resets to 0."""
        game.seed(1)  # seed that produces survival for 60s death
        # Force a failed check
        rec = game.play_half_round(drop_time=50, check_time=10)
        if rec.survived:
            assert game.player2.cylinder == 0  # Baku is checker, cylinder reset

    def test_permanent_death_ends_game(self):
        """If revival fails, game is over."""
        # Create a scenario with very low survival probability
        hal = Player(name="Hal", physicality=1.0)
        baku = Player(name="Baku", physicality=0.1)  # very weak
        ref = Referee()
        ref.cprs_performed = 10  # exhausted referee
        # Give baku huge prior damage
        baku.ttd = 600
        baku.cylinder = 200

        game = Game(player1=hal, player2=baku, referee=ref, first_dropper=hal)
        game.game_clock = 720.0

        # Failed check: cylinder = 200 + 60 = 260. Death at 260s with terrible odds.
        rec = game.play_half_round(drop_time=50, check_time=10)
        # With such bad stats, death is nearly certain
        if not rec.survived:
            assert game.game_over
            assert game.winner.name == "Hal"
            assert game.loser.name == "Baku"


class TestCylinderOverflow:
    def test_overflow_from_st_accumulation(self):
        """Cylinder hits 300 from accumulated ST, triggers injection."""
        hal = Player(name="Hal", physicality=1.0)
        baku = Player(name="Baku", physicality=1.0)
        ref = Referee()
        game = Game(player1=hal, player2=baku, referee=ref, first_dropper=hal)
        game.seed(99)
        game.game_clock = 720.0

        # Fill Baku's cylinder to 250
        baku.cylinder = 250

        # Now a check with 55 ST: 250 + 55 = 305 >= 300 -> overflow
        rec = game.play_half_round(drop_time=5, check_time=60)
        assert rec.result in (HalfRoundResult.CYLINDER_OVERFLOW_SURVIVED,
                               HalfRoundResult.CYLINDER_OVERFLOW_DIED)
        assert rec.death_duration == 305

    def test_cylinder_at_exactly_300_triggers(self):
        """Cylinder reaching exactly 300 triggers injection."""
        hal = Player(name="Hal", physicality=1.0)
        baku = Player(name="Baku", physicality=1.0)
        ref = Referee()
        game = Game(player1=hal, player2=baku, referee=ref, first_dropper=hal)
        game.seed(99)
        game.game_clock = 720.0

        baku.cylinder = 260

        # ST of 40: 260 + 40 = 300 -> overflow
        rec = game.play_half_round(drop_time=10, check_time=50)
        assert rec.result in (HalfRoundResult.CYLINDER_OVERFLOW_SURVIVED,
                               HalfRoundResult.CYLINDER_OVERFLOW_DIED)


class TestSafeStrategy:
    def test_safe_strategies_remaining_fresh(self, baku):
        """Fresh player has 4 safe checks: 4*60 = 240 < 300. The 5th hits 300 = injection."""
        assert baku.safe_strategies_remaining == 4

    def test_safe_strategies_remaining_with_cylinder(self, baku):
        baku.cylinder = 120
        assert baku.safe_strategies_remaining == 2  # 120+60+60=240 < 300, third would hit 300

    def test_safe_strategies_remaining_at_240(self, baku):
        baku.cylinder = 240
        assert baku.safe_strategies_remaining == 0  # next check hits 300 = injection

    def test_safe_strategies_remaining_at_300(self, baku):
        baku.cylinder = 300
        assert baku.safe_strategies_remaining == 0
