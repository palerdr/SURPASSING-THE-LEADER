#!/usr/bin/env python3
"""Play against CFR-Hal in the terminal.

Precomputes the Nash strategy table, then plays a full game where:
  - Hal samples actions from the CFR equilibrium strategy
  - Baku is the human player

Run from STL/:
    python scripts/play_vs_cfr.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.Player import Player
from src.Referee import Referee
from src.Game import Game, HalfRoundResult, HalfRoundRecord
from src.Constants import (
    PHYSICALITY_HAL, PHYSICALITY_BAKU, CYLINDER_MAX,
    TURN_DURATION_NORMAL, DEATH_PROCEDURE_OVERHEAD,
    OPENING_START_CLOCK,
)
from cfr.tree import solve_game, StrategyTable
from cfr.game_state import make_abstract_state, CYL_BUCKET_SIZE, CLOCK_BUCKET_SIZE


# ── Display helpers (reused from play_cli.py) ────────────────────────────

DIVIDER = "-" * 60

def clear():
    os.system("cls" if os.name == "nt" else "clear")

def cylinder_bar(cylinder: float) -> str:
    pct = min(cylinder / CYLINDER_MAX, 1.0)
    filled = int(pct * 20)
    return "#" * filled + "." * (20 - filled)

def print_banner():
    clear()
    print()
    print("  ============================================")
    print("   D R O P   T H E   H A N D K E R C H I E F")
    print("   Surpassing the Leader  --  vs CFR Hal")
    print("  ============================================")
    print()

def print_status(game: Game):
    print(f"  {'':40s} Clock: {game.format_game_clock()}")
    print(f"  {DIVIDER}")
    for p in [game.player1, game.player2]:
        bar = cylinder_bar(p.cylinder)
        safe = p.safe_strategies_remaining
        print(f"  {p.name:<12} [{bar}] {p.cylinder:>5.0f}/{CYLINDER_MAX}s"
              f"   Deaths: {p.deaths}   TTD: {p.ttd:.0f}s"
              f"   Safe: {safe}")
    print(f"  Referee CPRs: {game.referee.cprs_performed}")
    if game.is_leap_second_turn():
        print(f"  >>> LEAP SECOND WINDOW <<<")
    print(f"  {DIVIDER}")
    print()

def print_half_result(rec: HalfRoundRecord):
    print()
    print(f"  D ({rec.dropper}) dropped at second {rec.drop_time}")
    print(f"  C ({rec.checker}) checked at second {rec.check_time}")
    print()

    if rec.result == HalfRoundResult.CHECK_SUCCESS:
        print(f"  FOUND IT -- the handkerchief was on the ground.")
        print(f"  Squandered Time: {rec.st_gained:.0f}s added to {rec.checker}'s cylinder.")

    elif rec.result in (HalfRoundResult.CHECK_FAIL_SURVIVED,
                        HalfRoundResult.CHECK_FAIL_DIED):
        print(f"  MISSED -- {rec.checker} turned around too early!")
        print(f"  +60s penalty. Entire cylinder injected.")
        print()
        print_death(rec)

    elif rec.result in (HalfRoundResult.CYLINDER_OVERFLOW_SURVIVED,
                        HalfRoundResult.CYLINDER_OVERFLOW_DIED):
        print(f"  FOUND IT -- but {rec.checker}'s cylinder overflowed!")
        print(f"  ST pushed cylinder to {rec.death_duration:.0f}s (>= {CYLINDER_MAX}). Injection.")
        print()
        print_death(rec)
    print()

def print_death(rec: HalfRoundRecord):
    W = 45
    def box(text: str):
        print(f"  | {text:<{W}}|")
    print(f"  +{'-' * (W + 1)}+")
    print(f"  | {'DEATH SEQUENCE':^{W}}|")
    print(f"  +{'-' * (W + 1)}+")
    box(f"{rec.checker}'s heart stops.")
    box(f"Dead for {rec.death_duration:.0f} seconds.")
    box(f"Survival probability: {rec.survival_probability:.1%}")
    box("")
    time.sleep(0.5)
    box("...revival drug administered...")
    time.sleep(0.3)
    box(f"...CPR & recovery ({DEATH_PROCEDURE_OVERHEAD}s)...")
    time.sleep(0.3)
    if rec.survived:
        box(f">>> {rec.checker} SURVIVES. Cylinder reset. <<<")
    else:
        box(f">>> {rec.checker} IS DEAD. Revival failed. <<<")
    print(f"  +{'-' * (W + 1)}+")

def print_history(game: Game):
    print(f"  {'Half':<10} {'Dropper':>8}  drop  {'Checker':>8}  chk   Result")
    print(f"  {DIVIDER}")
    for rec in game.history:
        label = f"R{rec.round_num + 1}.{rec.half}"
        outcome = rec.result.value.replace("_", " ")
        print(f"  {label:<10} {rec.dropper:>8}  @{rec.drop_time:<4}"
              f"  {rec.checker:>8}  @{rec.check_time:<4}  {outcome}")


# ── Human input ──────────────────────────────────────────────────────────

def get_human_action(name: str, role: str, max_t: int) -> int:
    """Get action from human player. Hidden input for dropper."""
    import getpass
    prompt = f"  {name} ({role}) -- choose second [1-{max_t}]: "
    while True:
        try:
            if role == "Dropper":
                raw = getpass.getpass(prompt)
            else:
                raw = input(prompt)
            val = int(raw)
            if 1 <= val <= max_t:
                return val
            print(f"    Must be 1-{max_t}.")
        except ValueError:
            print("    Enter a number.")
        except (EOFError, KeyboardInterrupt):
            print("\n  Game aborted.")
            sys.exit(0)


# ── CFR action sampling ─────────────────────────────────────────────────

def cfr_action(
    strategy_table: StrategyTable,
    game: Game,
    hal: Player,
    baku: Player,
    is_dropper: bool,
    rng: np.random.Generator,
) -> int:
    """Sample an action from the CFR strategy table for Hal.

    Looks up the current game state in the strategy table and samples
    from the Nash equilibrium distribution.
    """
    # Build abstract state from Hal's perspective (Hal = P1 = my)
    state = make_abstract_state(
        round_num=game.round_num,
        half=game.current_half,
        my_cylinder=hal.cylinder,
        opp_cylinder=baku.cylinder,
        my_deaths=hal.deaths,
        opp_deaths=baku.deaths,
        game_clock=game.game_clock,
    )

    if state in strategy_table:
        dropper_strat, checker_strat, _ = strategy_table[state]
        if is_dropper:
            strat = dropper_strat
        else:
            strat = checker_strat
        # Sample from the distribution
        action_idx = rng.choice(len(strat), p=strat)
        return int(action_idx) + 1  # convert 0-indexed to 1-indexed second
    else:
        # State not in table (beyond solved rounds) — fall back to safe play
        if is_dropper:
            return 1  # instant drop
        else:
            return TURN_DURATION_NORMAL  # safe check at 60


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    print_banner()
    print("  Precomputing CFR strategy table...")
    print("  (This takes ~30 seconds for 10 rounds)")
    print()

    strategy_table = solve_game(max_rounds=10, iterations_per_state=5_000)
    print(f"  Solved {len(strategy_table)} game states.")
    print()

    hal = Player(name="Hal", physicality=PHYSICALITY_HAL)
    baku = Player(name="Baku", physicality=PHYSICALITY_BAKU)
    ref = Referee()
    game = Game(player1=hal, player2=baku, referee=ref, first_dropper=hal)
    game.seed(42)
    game.game_clock = OPENING_START_CLOCK

    rng = np.random.default_rng(42)

    input("  Press Enter to begin...")

    while not game.game_over:
        clear()
        round_display = game.round_num + 1
        print()
        print(f"  ==========  ROUND {round_display}  ==========")
        print()
        print_status(game)

        for half_num in (1, 2):
            if game.game_over:
                break

            dropper, checker = game.get_roles_for_half(game.current_half)
            turn_dur = game.get_turn_duration()

            print(f"  -- Half {half_num} --  (Turn: {turn_dur}s)")
            print(f"  Dropper: {dropper.name}    Checker: {checker.name}")
            print()

            hal_is_dropper = (dropper is hal)

            if hal_is_dropper:
                # Hal drops (CFR), Baku checks (human)
                drop_t = cfr_action(strategy_table, game, hal, baku, True, rng)
                check_t = get_human_action("Baku", "Checker", TURN_DURATION_NORMAL)
            else:
                # Baku drops (human), Hal checks (CFR)
                drop_t = get_human_action("Baku", "Dropper", turn_dur)
                check_t = cfr_action(strategy_table, game, hal, baku, False, rng)

            record = game.play_half_round(drop_t, check_t)
            print_half_result(record)

            if not game.game_over and half_num == 1:
                input("  Press Enter for Half 2...")
                print()

        if not game.game_over:
            input("  Press Enter for next round...")

    # Game over
    print()
    print(f"  ========================================")
    print(f"  GAME OVER")
    print(f"  Winner: {game.winner.name}")
    print(f"  Loser:  {game.loser.name}")
    print(f"  Clock:  {game.format_game_clock()}")
    print(f"  Rounds: {game.round_num + 1}")
    print(f"  ========================================")
    print()
    print_history(game)
    print()


if __name__ == "__main__":
    main()
