"""Run the agent through canonical R1T1 -> R9T2 against BakuLSREngineeringTeacher.

Records action history. Reports per-turn Hal action, MCTS root value, principal
line. Compares qualitatively to the canonical deviation_doc moves.

For now (no trained net yet), uses TerminalOnlyEvaluator. The validation is
that the script runs to completion without errors and produces a structured
output. Real validation comes after training on calibrated targets so the
trained ValueNetEvaluator can replace TerminalOnlyEvaluator and the principal
line carries strategic content beyond terminal-forced positions.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.cfr.evaluator import TerminalOnlyEvaluator
from environment.cfr.exact import ExactSearchConfig
from environment.cfr.mcts import MCTSConfig, MCTSResult, make_node, mcts_search
from environment.opponents.baku_teachers import BakuLSREngineeringTeacher
from src.Constants import OPENING_START_CLOCK, PHYSICALITY_BAKU, PHYSICALITY_HAL
from src.Game import Game
from src.Player import Player
from src.Referee import Referee


MAX_HALF_ROUNDS = 18
MCTS_ITERATIONS = 200
EXPLORATION_C = 1.0


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


def _resolve_root_value_for_hal(result: MCTSResult, hal_role: str) -> float:
    """MCTSResult already encodes Hal-perspective value at the root."""
    del hal_role  # Hal-perspective is baked into mcts_search.
    return float(result.root_value_for_hal)


def _principal_line_pairs(result: MCTSResult) -> tuple[tuple[int, int], ...]:
    return tuple((act.drop_time, act.check_time) for act in result.principal_line)


def play_through(
    *,
    seed: int = 0,
    iterations: int = MCTS_ITERATIONS,
    max_half_rounds: int = MAX_HALF_ROUNDS,
) -> list[TurnLog]:
    """Run alternating MCTS-Hal vs BakuLSREngineeringTeacher half-rounds."""
    game = build_canonical_r1t1_game(seed=seed)
    teacher = BakuLSREngineeringTeacher()
    evaluator = TerminalOnlyEvaluator()
    rng = np.random.default_rng(seed)
    exact_config = ExactSearchConfig()
    mcts_config = MCTSConfig(
        iterations=iterations,
        exploration_c=EXPLORATION_C,
        evaluator=None,
        use_tablebase=False,
    )

    logs: list[TurnLog] = []
    for half_index in range(max_half_rounds):
        if game.game_over:
            break

        hal_role = _hal_role(game)
        result = mcts_search(
            game=game,
            config=mcts_config,
            evaluator=evaluator,
            rng=rng,
            exact_config=exact_config,
            subgame_resolve_at_critical=True,
        )

        D = len(result.root_strategy_dropper)
        C = len(result.root_strategy_checker)
        d_choice = (
            int(np.argmax(result.root_strategy_dropper)) if D > 0 else 1
        )
        c_choice = (
            int(np.argmax(result.root_strategy_checker)) if C > 0 else 1
        )

        node = make_node(game, exact_config)
        node.game_snapshot.restore(game)
        drop_seconds = node.drop_seconds
        check_seconds = node.check_seconds
        if not drop_seconds or not check_seconds:
            break

        turn_duration = game.get_turn_duration()
        if hal_role == "dropper":
            hal_action = drop_seconds[d_choice] if d_choice < len(drop_seconds) else drop_seconds[0]
            baku_action = teacher.choose_action(game, "checker", turn_duration)
            drop_time, check_time = hal_action, baku_action
        else:
            hal_action = check_seconds[c_choice] if c_choice < len(check_seconds) else check_seconds[0]
            baku_action = teacher.choose_action(game, "dropper", turn_duration)
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
                hal_value_at_root=_resolve_root_value_for_hal(result, hal_role),
                principal_line=_principal_line_pairs(result),
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
    print("Running canonical R1T1 -> R9T2 against BakuLSREngineeringTeacher.")
    print(f"MCTS iterations per half-round: {MCTS_ITERATIONS}")
    print(f"Evaluator: TerminalOnlyEvaluator (placeholder; replace with ValueNetEvaluator after training).")
    print()

    logs = play_through()
    for log in logs:
        print(format_log_line(log))

    print()
    print(f"Total half-rounds played: {len(logs)}")
    if logs:
        print(f"Final Hal-perspective root value: {logs[-1].hal_value_at_root:+.3f}")


if __name__ == "__main__":
    main()
