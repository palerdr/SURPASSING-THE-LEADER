"""Tests for plan tickets 9+10: SPRT gate, frozen ladder, pattern reader.

No SolverAgent here (checkpoint + torch + ~0.5s/move would blow the
runtime budget); everything runs on stubs and scripted opponents so the
whole file stays well under a minute.
"""

from __future__ import annotations

import math
import os
import random
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.getcwd())

from stl.engine.actions import validate_action
from stl.play.opponents.base import Opponent
from stl.play.opponents.factory import create_scripted_opponent
from stl.play.opponents.pattern_reader import PatternReaderBaku
from stl.engine.game import (
    PHYSICALITY_BAKU,
    PHYSICALITY_HAL,
    OPENING_START_CLOCK,
    TURN_DURATION_LEAP,
    TURN_DURATION_NORMAL,
)
from stl.engine.game import Game, HalfRoundRecord, HalfRoundResult
from stl.engine.game import Player
from stl.engine.game import Referee
from stl.learning.strength.match_gate import (
    gate_report,
    reset_per_game,
    run_ladder,
    wilson_interval,
)
from stl.learning.strength.sprt import SPRTState, sprt_llr, sprt_verdict
from stl.learning.tournament import MatchResult, play_match


# ── SPRT ──────────────────────────────────────────────────────────────


def test_sprt_win_heavy_sequence_accepts():
    # 80/10/10 over 100 games is decisive at elo1=20.
    assert sprt_verdict(80, 10, 10, elo0=0.0, elo1=20.0) == "accept"


def test_sprt_all_wins_long_sequence_accepts():
    # Degenerate (zero-variance) record, but long enough to terminate.
    assert sprt_verdict(50, 0, 0) == "accept"


def test_sprt_loss_heavy_sequence_rejects():
    assert sprt_verdict(10, 10, 80, elo0=0.0, elo1=20.0) == "reject"


def test_sprt_all_losses_long_sequence_rejects():
    assert sprt_verdict(0, 0, 50) == "reject"


def test_sprt_balanced_small_n_continues():
    assert sprt_verdict(3, 0, 3) == "continue"
    assert sprt_verdict(2, 2, 2) == "continue"


def test_sprt_degenerate_tiny_n_continues():
    # 1-2 identical results must never terminate the test.
    assert sprt_verdict(2, 0, 0) == "continue"
    assert sprt_verdict(0, 0, 2) == "continue"
    assert sprt_verdict(0, 0, 0) == "continue"


def test_sprt_llr_monotone_in_wins():
    llrs = [sprt_llr(w, 10, 20) for w in (5, 15, 30, 60)]
    assert llrs == sorted(llrs)
    assert llrs[0] < 0 < llrs[-1]


def test_sprt_alpha_beta_bound_monotonicity():
    # W=60 D=20 L=20 has LLR ~ 3.34 at elo1=20: inside the (0.05, 0.05)
    # accept bound (2.944) but short of the (0.01, 0.01) bound (4.595).
    # Tighter error rates must demand MORE evidence, never less.
    record = (60, 20, 20)
    assert sprt_verdict(*record, alpha=0.05, beta=0.05) == "accept"
    assert sprt_verdict(*record, alpha=0.01, beta=0.01) == "continue"
    # And anything accepted at strict bounds stays accepted at loose ones.
    strong = (200, 50, 50)
    assert sprt_verdict(*strong, alpha=0.01, beta=0.01) == "accept"
    assert sprt_verdict(*strong, alpha=0.20, beta=0.20) == "accept"


def test_sprt_state_accumulates_and_matches_function():
    state = SPRTState()
    for _ in range(12):
        state.record("win")
    for _ in range(3):
        state.record("draw")
    for _ in range(5):
        state.record("loss")
    assert (state.wins, state.draws, state.losses) == (12, 3, 5)
    assert state.n == 20
    assert state.verdict() == sprt_verdict(12, 3, 5)
    state.record_match(8, 0, 2)
    assert state.n == 30


def test_sprt_state_rejects_bogus_result():
    with pytest.raises(ValueError):
        SPRTState().record("victory")


def test_sprt_verdict_validates_params():
    with pytest.raises(ValueError):
        sprt_verdict(1, 1, 1, alpha=0.0)
    with pytest.raises(ValueError):
        sprt_verdict(1, 1, 1, elo0=20.0, elo1=10.0)
    with pytest.raises(ValueError):
        sprt_verdict(-1, 0, 0)


# ── Pattern reader ────────────────────────────────────────────────────


def _fresh_game(seed: int = 0) -> Game:
    hal = Player(name="Hal", physicality=PHYSICALITY_HAL)
    baku = Player(name="Baku", physicality=PHYSICALITY_BAKU)
    game = Game(player1=hal, player2=baku, referee=Referee(), first_dropper=hal)
    game.seed(seed)
    game.game_clock = OPENING_START_CLOCK
    game.current_half = 1
    return game


def _record(
    dropper: str,
    checker: str,
    drop_time: int,
    check_time: int,
) -> HalfRoundRecord:
    return HalfRoundRecord(
        round_num=1,
        half=1,
        dropper=dropper,
        checker=checker,
        drop_time=drop_time,
        check_time=check_time,
        turn_duration=TURN_DURATION_NORMAL,
        result=HalfRoundResult.CHECK_SUCCESS,
        st_gained=1.0,
        death_duration=0.0,
        survived=None,
        game_clock_at_start=720.0,
        survival_probability=None,
    )


def test_pattern_reader_exploits_repeated_check_in_real_game():
    """Against a Hal that always checks 30, after 3 half-rounds of evidence
    the reader drops at 31 (guaranteed failed check if Hal repeats)."""
    game = _fresh_game()
    reader = PatternReaderBaku(target_name="Hal")

    # Drive 3 full rounds: Hal drops 1 / checks 30; Baku plays the reader.
    for _ in range(3):
        td = game.get_turn_duration()
        # Half 1: Hal drops, Baku checks.
        game.play_half_round(1, reader.choose_action(game, "checker", td))
        td = game.get_turn_duration()
        # Half 2: Baku drops, Hal checks 30.
        game.play_half_round(reader.choose_action(game, "dropper", td), 30)

    assert not game.game_over
    hal_checks = [r.check_time for r in game.history if r.checker == "Hal"]
    assert hal_checks == [30, 30, 30]

    td = game.get_turn_duration()
    assert reader.choose_action(game, "dropper", td) == 31
    # And the checker side reads Hal's repeated drop at 1.
    assert reader.choose_action(game, "checker", td) == 1


def test_pattern_reader_falls_back_without_modal_majority():
    """Uniform-ish target: no modal second reaches 50%, so the reader
    keeps its scripted fallbacks (drop 30 / check 60) without crashing."""
    rng = random.Random(7)
    seconds = rng.sample(range(5, 55), 8)  # 8 distinct values: max freq 1/8
    history = [_record("Baku", "Hal", 1, s) for s in seconds]
    history += [_record("Hal", "Baku", s, 60) for s in seconds]
    game = SimpleNamespace(history=history)

    reader = PatternReaderBaku(target_name="Hal")
    assert reader.choose_action(game, "dropper", TURN_DURATION_NORMAL) == 30
    assert reader.choose_action(game, "checker", TURN_DURATION_NORMAL) == 60


def test_pattern_reader_needs_three_samples():
    history = [_record("Baku", "Hal", 1, 30)] * 2  # only 2 observations
    game = SimpleNamespace(history=history)
    reader = PatternReaderBaku(target_name="Hal")
    assert reader.choose_action(game, "dropper", TURN_DURATION_NORMAL) == 30


def test_pattern_reader_legal_at_leap_turn():
    reader = PatternReaderBaku(target_name="Hal")

    # Hal repeatedly checks 60: modal+1 = 61, legal for a Baku DROPPER
    # at a leap turn — the kill shot.
    game = SimpleNamespace(history=[_record("Baku", "Hal", 1, 60)] * 3)
    drop = reader.choose_action(game, "dropper", TURN_DURATION_LEAP)
    assert drop == 61
    validate_action(drop, actor="baku", role="dropper", turn_duration=TURN_DURATION_LEAP)

    # At a normal turn the same read clamps back to 60.
    drop_normal = reader.choose_action(game, "dropper", TURN_DURATION_NORMAL)
    assert drop_normal == 60
    validate_action(
        drop_normal, actor="baku", role="dropper", turn_duration=TURN_DURATION_NORMAL
    )

    # As checker the reader can never exceed 60, even reading a 61 drop.
    game = SimpleNamespace(history=[_record("Hal", "Baku", 61, 60)] * 3)
    check = reader.choose_action(game, "checker", TURN_DURATION_LEAP)
    assert check == 60
    validate_action(check, actor="baku", role="checker", turn_duration=TURN_DURATION_LEAP)

    # Fallbacks are legal at the leap turn too.
    empty = SimpleNamespace(history=[])
    for role in ("dropper", "checker"):
        sec = reader.choose_action(empty, role, TURN_DURATION_LEAP)
        validate_action(sec, actor="baku", role=role, turn_duration=TURN_DURATION_LEAP)


def test_pattern_reader_registered_in_factory():
    opponent = create_scripted_opponent("pattern_reader")
    assert isinstance(opponent, PatternReaderBaku)
    assert opponent.target_name == "Hal"
    opponent.reset()  # explicit no-op must exist and not raise


# ── Ladder / gate report ──────────────────────────────────────────────


def _hal_stub(game: Game, role: str, turn_duration: int) -> int:
    """Fast deterministic legal Hal: drop 1, check 60."""
    del game, turn_duration
    return 60 if role == "checker" else 1


def test_run_ladder_smoke_and_report_shape():
    results = run_ladder(_hal_stub, ["random"], n_games=3, seed=0)
    assert set(results) == {"random"}
    match = results["random"]
    assert isinstance(match, MatchResult)
    assert match.games_played == 3
    assert match.hal_wins + match.baku_wins + match.draws == 3
    assert match.avg_game_length_half_rounds > 0

    report = gate_report(results, elo0=0.0, elo1=50.0)
    assert report["elo0"] == 0.0 and report["elo1"] == 50.0
    stats = report["opponents"]["random"]
    expected_keys = {
        "games", "wins", "draws", "losses", "win_rate", "score_rate",
        "wilson_lo", "wilson_hi", "llr", "sprt",
        "avg_game_length_half_rounds", "cause_of_termination",
    }
    assert expected_keys <= set(stats)
    assert stats["games"] == 3
    assert 0.0 <= stats["wilson_lo"] <= stats["win_rate"] <= stats["wilson_hi"] <= 1.0
    assert stats["sprt"] in {"accept", "reject", "continue"}
    overall = report["overall"]
    assert overall["games"] == 3
    assert overall["wins"] + overall["draws"] + overall["losses"] == 3


def test_run_ladder_random_rung_is_reproducible_for_same_seed():
    first = run_ladder(_hal_stub, ["random"], n_games=5, seed=123)["random"]
    second = run_ladder(_hal_stub, ["random"], n_games=5, seed=123)["random"]

    assert first == second


def test_run_ladder_accepts_explicit_learning_half_round_cap():
    results = run_ladder(
        _hal_stub,
        ["random"],
        n_games=1,
        seed=0,
        max_half_rounds=50,
    )
    assert results["random"].games_played == 1


def test_reset_per_game_fires_once_per_game():
    class CountingOpponent(Opponent):
        def __init__(self) -> None:
            self.resets = 0

        def choose_action(self, game: Game, role: str, turn_duration: int) -> int:
            del game, turn_duration
            return 60 if role == "checker" else 1

        def reset(self) -> None:
            self.resets += 1

    opponent = CountingOpponent()
    result = play_match(
        hal_choose_action=_hal_stub,
        baku_choose_action=reset_per_game(opponent),
        n_games=4,
        seed=0,
    )
    assert result.games_played == 4
    assert opponent.resets == 4  # exactly one fresh-game call per game


def test_wilson_interval_basics():
    assert wilson_interval(0, 0) == (0.0, 1.0)

    lo, hi = wilson_interval(5, 10)
    assert lo < 0.5 < hi

    lo_all, hi_all = wilson_interval(10, 10)
    assert hi_all == 1.0 and lo_all > 0.6

    lo_none, hi_none = wilson_interval(0, 10)
    assert lo_none == 0.0 and hi_none < 0.4

    # More evidence at the same proportion narrows the interval.
    lo_small, hi_small = wilson_interval(7, 10)
    lo_big, hi_big = wilson_interval(70, 100)
    assert (hi_big - lo_big) < (hi_small - lo_small)

    assert not math.isnan(lo) and not math.isnan(hi)
    with pytest.raises(ValueError):
        wilson_interval(11, 10)
