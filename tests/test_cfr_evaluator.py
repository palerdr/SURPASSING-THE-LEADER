"""Tests for MCTS Steps 2 and 4b: leaf evaluator interfaces and concrete evaluators."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.cfr.evaluator import (
    LeafEvaluator,
    TablebaseEvaluator,
    TerminalOnlyEvaluator,
    ValueNetEvaluator,
)
from environment.cfr.tablebase import get_scenario
from src.Constants import PHYSICALITY_BAKU, PHYSICALITY_HAL
from src.Game import Game
from src.Player import Player
from src.Referee import Referee


def _make_fresh_game() -> Game:
    hal = Player(name="Hal", physicality=PHYSICALITY_HAL)
    baku = Player(name="Baku", physicality=PHYSICALITY_BAKU)
    game = Game(player1=hal, player2=baku, referee=Referee(), first_dropper=hal)
    game.seed(0)
    return game


def _make_terminal_game(*, hal_wins: bool) -> Game:
    game = _make_fresh_game()
    game.game_over = True
    if hal_wins:
        game.winner = game.player1
        game.loser = game.player2
    else:
        game.winner = game.player2
        game.loser = game.player1
    return game


def test_terminal_only_evaluator_returns_zero_on_non_terminal_game():
    evaluator = TerminalOnlyEvaluator()
    game = _make_fresh_game()
    assert evaluator(game) == 0.0


def test_terminal_only_evaluator_returns_plus_one_on_hal_win_terminal():
    evaluator = TerminalOnlyEvaluator()
    game = _make_terminal_game(hal_wins=True)
    assert evaluator(game) == 1.0


def test_terminal_only_evaluator_returns_minus_one_on_baku_win_terminal():
    evaluator = TerminalOnlyEvaluator()
    game = _make_terminal_game(hal_wins=False)
    assert evaluator(game) == -1.0


def test_terminal_only_evaluator_perspective_name_flips_sign():
    # Same terminal Hal-win game, but the evaluator is built from Baku's
    # perspective. From Baku's POV, Hal winning is a loss → return -1.0.
    hal_evaluator = TerminalOnlyEvaluator(perspective_name="Hal")
    baku_evaluator = TerminalOnlyEvaluator(perspective_name="Baku")
    game = _make_terminal_game(hal_wins=True)
    assert hal_evaluator(game) == 1.0
    assert baku_evaluator(game) == -1.0


def test_terminal_only_evaluator_default_perspective_is_hal():
    evaluator = TerminalOnlyEvaluator()
    game = _make_terminal_game(hal_wins=True)
    # No perspective_name passed → default is "Hal" → Hal-win returns +1.
    assert evaluator(game) == 1.0


def test_terminal_only_evaluator_satisfies_leaf_evaluator_protocol_structurally():
    # Protocol conformance is structural in Python's typing system: any object
    # whose __call__ has the right signature satisfies LeafEvaluator. We verify
    # the contract by exercising a function that takes a LeafEvaluator-typed
    # argument with our concrete class.
    def call_with_protocol(evaluator: LeafEvaluator, game: Game) -> float:
        return evaluator(game)

    evaluator = TerminalOnlyEvaluator()
    game = _make_terminal_game(hal_wins=True)
    assert call_with_protocol(evaluator, game) == 1.0


def test_terminal_only_evaluator_returns_float_type():
    # The protocol promises a float. Callers (MCTS backup) rely on this for
    # numpy arithmetic, so make sure the return is actually a float, not None
    # leaking through, not int.
    evaluator = TerminalOnlyEvaluator()
    assert isinstance(evaluator(_make_fresh_game()), float)
    assert isinstance(evaluator(_make_terminal_game(hal_wins=True)), float)
    assert isinstance(evaluator(_make_terminal_game(hal_wins=False)), float)


# ── TablebaseEvaluator (Slice 4b) ─────────────────────────────────────────


class _RecordingEvaluator:
    """Test double: a LeafEvaluator that records how many times it was called."""

    def __init__(self, fixed_value: float = 0.42) -> None:
        self.calls = 0
        self.fixed_value = fixed_value

    def __call__(self, game: Game) -> float:
        self.calls += 1
        return self.fixed_value


def test_tablebase_evaluator_construction_loads_pinned_registry_entries():
    # Two pinned scenarios in the registry today: forced_baku_overflow_death
    # (+1.0) and forced_hal_overflow_death (-1.0). The table must contain both.
    evaluator = TablebaseEvaluator(fallback=TerminalOnlyEvaluator())
    assert len(evaluator._table) == 2
    assert 1.0 in evaluator._table.values()
    assert -1.0 in evaluator._table.values()


def test_tablebase_evaluator_hit_returns_pinned_value_without_calling_fallback():
    recorder = _RecordingEvaluator()
    evaluator = TablebaseEvaluator(fallback=recorder)

    scenario = get_scenario("forced_baku_overflow_death")
    value = evaluator(scenario.game)

    assert value == 1.0
    assert recorder.calls == 0  # short-circuit; fallback never invoked


def test_tablebase_evaluator_hit_works_for_negative_pinned_value():
    recorder = _RecordingEvaluator()
    evaluator = TablebaseEvaluator(fallback=recorder)

    scenario = get_scenario("forced_hal_overflow_death")
    value = evaluator(scenario.game)

    assert value == -1.0
    assert recorder.calls == 0


def test_tablebase_evaluator_miss_delegates_to_fallback():
    # A fresh game (cyl=0/0) is not in the pinned table; fallback should run.
    recorder = _RecordingEvaluator(fixed_value=0.42)
    evaluator = TablebaseEvaluator(fallback=recorder)

    value = evaluator(_make_fresh_game())

    assert value == 0.42
    assert recorder.calls == 1


def test_tablebase_evaluator_composes_with_terminal_only_evaluator():
    # End-to-end: tablebase hit returns pinned, miss falls through to
    # TerminalOnlyEvaluator which returns 0.0 for non-terminal positions.
    evaluator = TablebaseEvaluator(fallback=TerminalOnlyEvaluator())

    pinned = get_scenario("forced_baku_overflow_death")
    fresh = _make_fresh_game()

    assert evaluator(pinned.game) == 1.0
    assert evaluator(fresh) == 0.0


def test_tablebase_evaluator_satisfies_leaf_evaluator_protocol_structurally():
    def call_with_protocol(evaluator: LeafEvaluator, game: Game) -> float:
        return evaluator(game)

    evaluator = TablebaseEvaluator(fallback=TerminalOnlyEvaluator())
    scenario = get_scenario("forced_baku_overflow_death")
    assert call_with_protocol(evaluator, scenario.game) == 1.0


# ── ValueNetEvaluator (Slice 4b Step 2) ───────────────────────────────────


def test_value_net_evaluator_returns_model_fn_value():
    evaluator = ValueNetEvaluator(model_fn=lambda game: 0.42)
    assert evaluator(_make_fresh_game()) == 0.42


def test_value_net_evaluator_passes_game_to_model_fn():
    # The model_fn must receive the live game object, not be called with no
    # arguments or an abstracted state. Verify by reading game state inside.
    def model_fn(game: Game) -> float:
        return float(game.player2.cylinder)

    g = _make_fresh_game()
    g.player2.cylinder = 173.0
    evaluator = ValueNetEvaluator(model_fn=model_fn)
    assert evaluator(g) == 173.0


def test_value_net_evaluator_satisfies_leaf_evaluator_protocol_structurally():
    def call_with_protocol(evaluator: LeafEvaluator, game: Game) -> float:
        return evaluator(game)

    evaluator = ValueNetEvaluator(model_fn=lambda game: -0.3)
    assert call_with_protocol(evaluator, _make_fresh_game()) == -0.3


def test_value_net_evaluator_composes_as_tablebase_fallback():
    # The intended Slice 4b composition: tablebase short-circuits on hits;
    # value net handles every other position. Pinned state -> tablebase value;
    # fresh game -> value net's output.
    fallback = ValueNetEvaluator(model_fn=lambda game: 0.25)
    evaluator = TablebaseEvaluator(fallback=fallback)

    pinned = get_scenario("forced_baku_overflow_death")
    assert evaluator(pinned.game) == 1.0           # tablebase hit
    assert evaluator(_make_fresh_game()) == 0.25   # delegates to value net


def test_tablebase_evaluator_relational_scenarios_are_not_in_table():
    # Scenarios with expected_value=None (CPR pair, leap probe, safe-budget
    # pair) must not appear in the pinned table — they're paired tests, not
    # values we'd want to short-circuit on.
    evaluator = TablebaseEvaluator(fallback=TerminalOnlyEvaluator())
    relational_names = (
        "safe_budget_pressure_at_cylinder_241",
        "safe_budget_pressure_at_cylinder_240",
        "cpr_degradation_fresh_referee",
        "cpr_degradation_fatigued_referee",
    )
    from environment.cfr.exact import exact_public_state

    for name in relational_names:
        scenario = get_scenario(name)
        key = exact_public_state(scenario.game)
        assert key not in evaluator._table, name
