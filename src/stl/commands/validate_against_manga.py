"""Run the solver agent through canonical R1T1 -> R9T2 against BakuLSREngineeringTeacher.

Records action history. Reports per-turn Hal action, MCTS root value, principal
line. Compares qualitatively to the canonical deviation_doc moves.

Uses the trained checkpoint (hal.agent.SolverAgent) when present; pass
--terminal-only to fall back to the TerminalOnlyEvaluator placeholder.
Hal's actions are SAMPLED from the root equilibrium mixture — argmaxing a
mixed equilibrium is maximally exploitable in this game and was bug B4.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stl.solver.search import TerminalOnlyEvaluator
from stl.engine.actions import validate_action
from stl.play.opponents.baku_teachers import BakuLSREngineeringTeacher
from stl.play.agent import DEFAULT_CHECKPOINT, SolverAgent
from stl.engine.game import OPENING_START_CLOCK, PHYSICALITY_BAKU, PHYSICALITY_HAL
from stl.engine.game import Game
from stl.engine.game import Player
from stl.engine.game import Referee


MAX_HALF_ROUNDS = 18
MCTS_ITERATIONS = 200


@dataclass(frozen=True)
class TurnLog:
    half_round_index: int
    round_num: int
    current_half: int
    game_clock: float
    hal_role: str
    hal_action: int
    baku_action: int
    hal_value_at_root: float
    principal_line: tuple[tuple[int, int], ...]


def build_canonical_r1t1_game(seed: int = 0) -> Game:
    hal = Player(name="Hal", physicality=PHYSICALITY_HAL)
    baku = Player(name="Baku", physicality=PHYSICALITY_BAKU)
    game = Game(player1=hal, player2=baku, referee=Referee(), first_dropper=hal)
    game.seed(seed)
    game.game_clock = float(OPENING_START_CLOCK)
    game.current_half = 1
    return game


def _hal_role(game: Game) -> str:
    dropper, _ = game.get_roles_for_half(game.current_half)
    return "dropper" if dropper.name.lower() == "hal" else "checker"


def build_agent(*, seed: int, iterations: int, terminal_only: bool) -> SolverAgent:
    if terminal_only:
        return SolverAgent(
            DEFAULT_CHECKPOINT,
            player_name="Hal",
            iterations=iterations,
            seed=seed,
            resolve_at_critical=True,
            evaluator=TerminalOnlyEvaluator(),
        )
    return SolverAgent(
        DEFAULT_CHECKPOINT,
        player_name="Hal",
        iterations=iterations,
        seed=seed,
        resolve_at_critical=True,
    )


def play_through(
    *,
    seed: int = 0,
    iterations: int = MCTS_ITERATIONS,
    max_half_rounds: int = MAX_HALF_ROUNDS,
    terminal_only: bool = False,
) -> list[TurnLog]:
    """Run sampled-equilibrium Hal vs BakuLSREngineeringTeacher half-rounds."""
    game = build_canonical_r1t1_game(seed=seed)
    teacher = BakuLSREngineeringTeacher()
    agent = build_agent(seed=seed, iterations=iterations, terminal_only=terminal_only)

    logs: list[TurnLog] = []
    for half_index in range(max_half_rounds):
        if game.game_over:
            break

        hal_role = _hal_role(game)
        turn_duration = game.get_turn_duration()

        result = agent.search(game)
        hal_action = agent.choose_action(game, hal_role, turn_duration)

        if hal_role == "dropper":
            baku_action = teacher.choose_action(game, "checker", turn_duration)
            validate_action(hal_action, actor="hal", role="dropper", turn_duration=turn_duration)
            validate_action(baku_action, actor="baku", role="checker", turn_duration=turn_duration)
            drop_time, check_time = hal_action, baku_action
        else:
            baku_action = teacher.choose_action(game, "dropper", turn_duration)
            validate_action(baku_action, actor="baku", role="dropper", turn_duration=turn_duration)
            validate_action(hal_action, actor="hal", role="checker", turn_duration=turn_duration)
            drop_time, check_time = baku_action, hal_action

        logs.append(
            TurnLog(
                half_round_index=half_index,
                round_num=game.round_num,
                current_half=game.current_half,
                game_clock=float(game.game_clock),
                hal_role=hal_role,
                hal_action=int(hal_action),
                baku_action=int(baku_action),
                hal_value_at_root=float(result.root_value_for_hal),
                principal_line=tuple(
                    (act.drop_time, act.check_time) for act in result.principal_line
                ),
            )
        )

        try:
            game.resolve_half_round(drop_time, check_time, survived_outcome=None)
        except Exception:  # game_over signaled in-engine, stop reporting.
            break

    return logs


def format_log_line(log: TurnLog) -> str:
    return (
        f"hr={log.half_round_index:02d} "
        f"R{log.round_num}H{log.current_half} "
        f"clock={log.game_clock:.0f} "
        f"hal_role={log.hal_role:>7} "
        f"hal={log.hal_action:>2} baku={log.baku_action:>2} "
        f"V_hal={log.hal_value_at_root:+.3f} "
        f"PL={log.principal_line[:3]}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=MCTS_ITERATIONS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--terminal-only", action="store_true",
                        help="Use the TerminalOnlyEvaluator placeholder instead of the trained net")
    args = parser.parse_args()

    evaluator_name = "TerminalOnlyEvaluator" if args.terminal_only else "trained ValueNet (tablebase-probed)"
    print("Running canonical R1T1 -> R9T2 against BakuLSREngineeringTeacher.")
    print(f"MCTS iterations per half-round: {args.iterations}")
    print(f"Evaluator: {evaluator_name}; actions SAMPLED from root mixture.")
    print()

    logs = play_through(seed=args.seed, iterations=args.iterations, terminal_only=args.terminal_only)
    for log in logs:
        print(format_log_line(log))

    print()
    print(f"Total half-rounds played: {len(logs)}")
    if logs:
        print(f"Final Hal-perspective root value: {logs[-1].hal_value_at_root:+.3f}")


if __name__ == "__main__":
    main()
