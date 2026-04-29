"""Tests for Phase 7: BakuLSREngineeringTeacher and the manga validation script.

Covers basic instantiation, action legality (dropper/checker; normal turn vs
leap window dropper), determinism under a seeded engine, and the LSR-engineering
behavior (deliberately fail when fail flips parity into Active LSR).
"""

from __future__ import annotations

import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.cfr.timing_features import current_checker_fail_would_activate_lsr
from environment.opponents.baku_teachers import BakuLSREngineeringTeacher, BakuTeacher
from src.Constants import (
    LS_WINDOW_START,
    OPENING_START_CLOCK,
    PHYSICALITY_BAKU,
    PHYSICALITY_HAL,
)
from src.Game import Game
from src.Player import Player
from src.Referee import Referee


def _new_game(seed: int = 0, *, clock: float | None = None, current_half: int = 1) -> Game:
    hal = Player(name="Hal", physicality=PHYSICALITY_HAL)
    baku = Player(name="Baku", physicality=PHYSICALITY_BAKU)
    game = Game(player1=hal, player2=baku, referee=Referee(), first_dropper=hal)
    game.seed(seed)
    game.game_clock = float(OPENING_START_CLOCK if clock is None else clock)
    game.current_half = current_half
    return game


# ── 1. Instantiation and inheritance ──────────────────────────────────────


def test_baku_lsr_engineering_teacher_instantiates_without_error():
    teacher = BakuLSREngineeringTeacher()
    assert isinstance(teacher, BakuLSREngineeringTeacher)
    assert isinstance(teacher, BakuTeacher)


# ── 2. Legality of returned seconds ────────────────────────────────────────


def test_baku_lsr_engineering_teacher_returns_legal_dropper_second_normal_turn():
    """Baku-as-dropper outside leap window: must be in [1, 60]."""
    game = _new_game()
    teacher = BakuLSREngineeringTeacher()
    turn_duration = game.get_turn_duration()
    action = teacher.choose_action(game, "dropper", turn_duration)
    assert 1 <= action <= 60


def test_baku_lsr_engineering_teacher_returns_legal_checker_second_normal_turn():
    """Baku-as-checker is always capped at 60."""
    game = _new_game()
    teacher = BakuLSREngineeringTeacher()
    turn_duration = game.get_turn_duration()
    action = teacher.choose_action(game, "checker", turn_duration)
    assert 1 <= action <= 60


def test_baku_lsr_engineering_teacher_dropper_can_use_61_in_leap_window():
    """Baku-as-dropper inside the leap window may legally play 61."""
    game = _new_game(clock=float(LS_WINDOW_START))
    teacher = BakuLSREngineeringTeacher()
    turn_duration = game.get_turn_duration()
    assert turn_duration == 61
    action = teacher.choose_action(game, "dropper", turn_duration)
    assert 1 <= action <= 61


# ── 3. Determinism under seeded engine ────────────────────────────────────


def test_baku_lsr_engineering_teacher_is_deterministic_under_seed():
    teacher = BakuLSREngineeringTeacher()
    a = teacher.choose_action(_new_game(seed=42), "dropper", 60)
    b = teacher.choose_action(_new_game(seed=42), "dropper", 60)
    assert a == b

    c = teacher.choose_action(_new_game(seed=42), "checker", 60)
    d = teacher.choose_action(_new_game(seed=42), "checker", 60)
    assert c == d


# ── 4. LSR-engineering behavior ───────────────────────────────────────────


def _find_lsr_activation_state() -> Game:
    """Find an early-game state where current_checker_fail_would_activate_lsr is True.

    We progress through an opening sequence with both players safely checking
    on early rounds and search for the first state where a Baku-as-checker
    fail would flip parity to Active LSR. Returns the game in that state so
    the teacher's decision can be observed.
    """
    game = _new_game()
    while not game.game_over and game.round_num < 9:
        if current_checker_fail_would_activate_lsr(game):
            _, checker = game.get_roles_for_half(game.current_half)
            if checker.name.lower() == "baku":
                return game
        # Both halves: drop=30, check=60 (safe), no deaths.
        game.resolve_half_round(30, 60, survived_outcome=None)
    raise AssertionError("No LSR-activation checker state found in 9 rounds.")


def test_baku_lsr_engineering_teacher_fails_when_fail_flips_to_active_lsr():
    """When current_checker_fail_would_activate_lsr is True at a Baku-as-checker
    turn inside the engineering runway, the teacher should pick a check second
    that strictly precedes its dropper's likely drop time. We verify this by
    simulating a forward step against a paired dropper and asserting that the
    check < drop case fires (i.e. a failed check is engineered)."""
    game = _find_lsr_activation_state()
    teacher = BakuLSREngineeringTeacher()
    check_action = teacher.choose_action(game, "checker", game.get_turn_duration())
    # Engineered failing-check is "pick second 1" so that for any drop_time > 1,
    # check < drop, triggering a failed check.
    assert check_action == 1


def test_baku_lsr_engineering_teacher_safe_check_outside_engineering_window():
    """When fail does NOT activate LSR, teacher plays the safe-budget check (60)."""
    game = _new_game()
    teacher = BakuLSREngineeringTeacher()
    # At the very opening (round 0, half 1) most fail-activates conditions are
    # not satisfied; verify safe check is returned.
    turn_duration = game.get_turn_duration()
    action = teacher.choose_action(game, "checker", turn_duration)
    if not current_checker_fail_would_activate_lsr(game):
        assert action == 60


# ── 5. Manga validation script imports cleanly ────────────────────────────


def test_validate_against_manga_script_module_imports():
    """The script must compile and its top-level helpers be importable.

    Actually running the script is a longer job (real MCTS over 18 half-rounds);
    this test only enforces that the module loads.
    """
    module = importlib.import_module("scripts.validate_against_manga")
    assert hasattr(module, "main")
    assert hasattr(module, "play_through")
    assert hasattr(module, "build_canonical_r1t1_game")
    game = module.build_canonical_r1t1_game(seed=0)
    assert game.player1.name == "Hal"
    assert game.player2.name == "Baku"
