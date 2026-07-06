"""SolverAgent adapter tests (plan Track A / tickets 4, 6, 7).

What these pin down:
- loud failure when no checkpoint exists (no silent fallback),
- every emitted action is legal, including leap-window states (Hal never 61),
- actions are SAMPLED from the root mixture (distribution match), never argmax,
- the policy is deterministic per state (Markov purity — required by the
  best-response exploitability probe) while action draws stay stochastic,
- seeded determinism end-to-end,
- factory and play_cli construct the solver path.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.getcwd())

from stl.solver.evaluator import TerminalOnlyEvaluator
from stl.engine.actions import legal_max_second
from stl.play.opponents.factory import SCRIPTED_OPPONENTS, create_scripted_opponent
from stl.play.agent import DEFAULT_CHECKPOINT, BakuSolverAgent, HalSolverAgent, SolverAgent
from stl.engine.game import LS_WINDOW_START, PHYSICALITY_BAKU, PHYSICALITY_HAL
from stl.engine.game import Game
from stl.engine.game import Player
from stl.engine.game import Referee

HAS_CHECKPOINT = os.path.exists(DEFAULT_CHECKPOINT)
needs_checkpoint = pytest.mark.skipif(not HAS_CHECKPOINT, reason="headline checkpoint not pulled")
HAS_TIER_A = os.path.exists(
    os.path.join(
        os.getcwd(),
        "checkpoints",
        "tablebase",
        "tier_a",
        "manifest.json",
    )
)
needs_tier_a = pytest.mark.skipif(not HAS_TIER_A, reason="Tier A artifacts absent")


def make_game(*, clock=720.0, half=1, hal_cyl=0.0, baku_cyl=0.0, seed=7) -> Game:
    hal = Player(name="Hal", physicality=PHYSICALITY_HAL)
    baku = Player(name="Baku", physicality=PHYSICALITY_BAKU)
    game = Game(player1=hal, player2=baku, referee=Referee(), first_dropper=hal)
    game.seed(seed)
    game.game_clock = clock
    game.current_half = half
    hal.cylinder = hal_cyl
    baku.cylinder = baku_cyl
    return game


def terminal_agent(player_name="Hal", *, iterations=8, seed=0, **kw) -> SolverAgent:
    """Cheap agent for structural tests: no torch, no checkpoint needed."""
    return SolverAgent(
        "does-not-matter",
        player_name=player_name,
        iterations=iterations,
        seed=seed,
        evaluator=TerminalOnlyEvaluator(),
        **kw,
    )


# ── Loud loading ──────────────────────────────────────────────────────────


def test_missing_checkpoint_raises_not_falls_back(tmp_path):
    with pytest.raises(FileNotFoundError, match="no silent fallback"):
        SolverAgent(tmp_path / "nope.pt", player_name="Hal")


@needs_checkpoint
def test_default_checkpoint_constructs_real_agent():
    agent = HalSolverAgent()
    assert agent.player_name == "Hal"


@needs_tier_a
def test_solver_agent_can_wrap_tier_a_runtime_evaluator():
    from stl.solver.tier_a import TierAEvaluator

    agent = terminal_agent("Hal", use_tier_a=True, tier_a_width=0.05)
    assert isinstance(agent.evaluator, TierAEvaluator)
    game = make_game(clock=3661.0, half=1, hal_cyl=240.0, baku_cyl=240.0)
    value, _, _ = agent.evaluator(game)
    assert value != 0.0


def test_tier_a_agent_with_missing_artifacts_matches_base_policy(tmp_path, monkeypatch):
    from stl.solver.epoch_tablebase import tier_a

    monkeypatch.setattr(tier_a, "DEFAULT_TIER_A_DIR", tmp_path)
    game = make_game(clock=720.0, half=1)
    base = terminal_agent("Hal", iterations=8, seed=11)
    wrapped = terminal_agent("Hal", iterations=8, seed=11, use_tier_a=True)

    base_seconds, base_probs = base.policy(game, "dropper")
    wrapped_seconds, wrapped_probs = wrapped.policy(game, "dropper")

    assert wrapped_seconds == base_seconds
    np.testing.assert_array_equal(wrapped_probs, base_probs)


# ── Legality ──────────────────────────────────────────────────────────────


def test_actions_always_legal_across_states_including_leap_window():
    agent = terminal_agent("Hal")
    states = [
        make_game(clock=720.0, half=1),                      # opening, Hal drops
        make_game(clock=720.0, half=2),                      # Hal checks
        make_game(clock=3300.0, half=1, hal_cyl=120, baku_cyl=120),
        make_game(clock=float(LS_WINDOW_START), half=1),     # leap turn, Hal drops
        make_game(clock=float(LS_WINDOW_START), half=2),     # leap turn, Hal checks
        make_game(clock=3601.0, half=1),                     # post-leap
        make_game(clock=720.0, half=1, baku_cyl=290.0),      # near-overflow pressure
    ]
    for game in states:
        dropper, checker = game.get_roles_for_half(game.current_half)
        role = "dropper" if dropper.name == "Hal" else "checker"
        turn_duration = game.get_turn_duration()
        for _ in range(3):
            action = agent.choose_action(game, role, turn_duration)
            max_sec = legal_max_second("Hal", role, turn_duration)
            assert 1 <= action <= max_sec, (
                f"illegal action {action} (max {max_sec}) at clock={game.game_clock} "
                f"half={game.current_half} role={role}"
            )


def test_hal_never_emits_61_in_leap_turn_as_checker():
    agent = terminal_agent("Hal", iterations=16)
    game = make_game(clock=float(LS_WINDOW_START), half=2)  # Baku drops, Hal checks
    turn_duration = game.get_turn_duration()
    assert turn_duration == 61
    for _ in range(25):
        assert agent.choose_action(game, "checker", turn_duration) <= 60


def test_wrong_seat_raises():
    agent = terminal_agent("Hal")
    game = make_game(half=2)  # half 2: Baku drops
    with pytest.raises(ValueError, match="holds that role"):
        agent.choose_action(game, "dropper", game.get_turn_duration())


# ── Mixed play: sampling, not argmax ─────────────────────────────────────


def test_actions_are_sampled_from_the_root_mixture():
    agent = terminal_agent("Hal", iterations=32, seed=3)
    game = make_game(clock=720.0, half=1)
    seconds, probs = agent.policy(game, "dropper")

    n = 400
    counts = {s: 0 for s in seconds}
    for _ in range(n):
        counts[agent.choose_action(game, "dropper", game.get_turn_duration())] += 1

    empirical = np.array([counts[s] / n for s in seconds])
    tvd = 0.5 * float(np.abs(empirical - probs).sum())
    assert tvd < 0.15, f"empirical action law far from root mixture (TVD={tvd:.3f})"

    if (probs > 1e-9).sum() > 1:  # mixed equilibrium => argmax play is detectable
        assert max(counts.values()) < n, "agent is argmaxing a mixed root strategy"


def test_policy_is_pure_function_of_state():
    """Markov purity: two queries at the same public state agree exactly,
    even across separate agent instances with the same base seed."""
    game_a = make_game(clock=720.0, half=1)
    game_b = make_game(clock=720.0, half=1)

    agent_a = terminal_agent("Hal", iterations=24, seed=11)
    agent_b = terminal_agent("Hal", iterations=24, seed=11)

    sec_a, probs_a = agent_a.policy(game_a, "dropper")
    sec_b, probs_b = agent_b.policy(game_b, "dropper")

    assert sec_a == sec_b
    np.testing.assert_array_equal(probs_a, probs_b)


def test_action_stream_is_deterministic_under_same_seed():
    game = make_game(clock=720.0, half=1)
    a = terminal_agent("Hal", iterations=16, seed=5)
    b = terminal_agent("Hal", iterations=16, seed=5)
    actions_a = [a.choose_action(game, "dropper", 60) for _ in range(10)]
    actions_b = [b.choose_action(game, "dropper", 60) for _ in range(10)]
    assert actions_a == actions_b


# ── Wiring ────────────────────────────────────────────────────────────────


def test_factory_registers_solver_agents_and_lsr_teacher():
    assert "hal_solver" in SCRIPTED_OPPONENTS
    assert "baku_solver" in SCRIPTED_OPPONENTS
    teacher = create_scripted_opponent("baku_lsr_engineering")
    assert type(teacher).__name__ == "BakuLSREngineeringTeacher"


@needs_checkpoint
def test_factory_constructs_real_solver_agent():
    agent = create_scripted_opponent("hal_solver")
    assert isinstance(agent, HalSolverAgent)


@needs_checkpoint
def test_play_cli_solver_path_builds_solver_agent():
    from stl.play import cli as play_cli

    hal_ai = play_cli.load_hal_ai(depth=2, checkpoint=None, agent="solver", iterations=8)
    assert isinstance(hal_ai, SolverAgent)


@needs_checkpoint
def test_play_cli_solver_path_can_enable_tier_a():
    from stl.solver.tier_a import TierAEvaluator

    from stl.play import cli as play_cli

    hal_ai = play_cli.load_hal_ai(
        depth=2,
        checkpoint=None,
        agent="solver",
        iterations=8,
        use_tier_a=True,
    )
    assert isinstance(hal_ai, SolverAgent)
    assert isinstance(hal_ai.evaluator, TierAEvaluator)


def test_play_cli_solver_path_fails_loudly_without_checkpoint(tmp_path):
    from stl.play import cli as play_cli

    with pytest.raises(FileNotFoundError):
        play_cli.load_hal_ai(depth=2, checkpoint=str(tmp_path / "missing.pt"), agent="solver")


# ── Baku seat ─────────────────────────────────────────────────────────────


def test_baku_seat_agent_acts_legally_including_leap_drop():
    agent = terminal_agent("Baku", iterations=16)
    game = make_game(clock=float(LS_WINDOW_START), half=2)  # Baku drops in leap turn
    turn_duration = game.get_turn_duration()
    for _ in range(10):
        action = agent.choose_action(game, "dropper", turn_duration)
        assert 1 <= action <= 61
