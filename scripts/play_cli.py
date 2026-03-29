#!/usr/bin/env python3
"""
Drop The Handkerchief -- CLI

Two-player terminal game. Both players share a terminal.
Dropper input is hidden so the checker can't see it.

Run from project root:
    python scripts/play_cli.py
"""

import sys
import os
import getpass
import time

# Add project root to path so `src` resolves as a package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.Player import Player
from src.Referee import Referee
from src.Game import Game, HalfRoundResult, HalfRoundRecord
from src.Constants import (
    PHYSICALITY_HAL, PHYSICALITY_BAKU, CYLINDER_MAX,
    TURN_DURATION_NORMAL, DEATH_PROCEDURE_OVERHEAD,
)

# -- Display helpers -------------------------------------------------

DIVIDER = "-" * 60

def clear():
    os.system("cls" if os.name == "nt" else "clear")

def pause(msg="Press Enter to continue..."):
    input(f"\n  {msg}")

def cylinder_bar(cylinder: float) -> str:
    pct = min(cylinder / CYLINDER_MAX, 1.0)
    filled = int(pct * 20)
    return "#" * filled + "." * (20 - filled)

def print_banner():
    clear()
    print()
    print("  ============================================")
    print("   D R O P   T H E   H A N D K E R C H I E F")
    print("   Surpassing the Leader  --  Death Game")
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


# -- Input -----------------------------------------------------------

def get_drop_time(name: str, max_t: int) -> int:
    """Hidden input for dropper."""
    while True:
        try:
            raw = getpass.getpass(f"  {name} (Dropper) -- choose second [1-{max_t}]: ")
            val = int(raw)
            if 1 <= val <= max_t:
                return val
            print(f"    Must be 1-{max_t}.")
        except ValueError:
            print("    Enter a number.")
        except (EOFError, KeyboardInterrupt):
            print("\n  Game aborted.")
            sys.exit(0)

def get_check_time(name: str) -> int:
    """Visible input for checker."""
    while True:
        try:
            raw = input(f"  {name} (Checker) -- choose second [1-{TURN_DURATION_NORMAL}]: ")
            val = int(raw)
            if 1 <= val <= TURN_DURATION_NORMAL:
                return val
            print(f"    Must be 1-{TURN_DURATION_NORMAL}.")
        except ValueError:
            print("    Enter a number.")
        except (EOFError, KeyboardInterrupt):
            print("\n  Game aborted.")
            sys.exit(0)


# -- Result display --------------------------------------------------

def print_half_result(rec: HalfRoundRecord):
    print()
    print(f"  D ({rec.dropper}) dropped at second {rec.drop_time}")
    print(f"  C ({rec.checker}) checked at second {rec.check_time}")
    print()

    if rec.result == HalfRoundResult.CHECK_SUCCESS:
        if rec.st_gained == 0:
            print(f"  FOUND IT -- {rec.checker} checked at the exact drop moment.")
            print(f"  No squandered time.")
        else:
            print(f"  FOUND IT -- the handkerchief was on the ground.")
            print(f"  Squandered Time: {rec.st_gained:.0f}s added to {rec.checker}'s cylinder.")

    elif rec.result in (HalfRoundResult.CHECK_FAIL_SURVIVED,
                        HalfRoundResult.CHECK_FAIL_DIED):
        print(f"  MISSED -- {rec.checker} turned around too early!")
        print(f"  The handkerchief wasn't on the ground yet.")
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
    W = 45  # inner width of the box

    def box(text: str):
        print(f"  | {text:<{W}}|")

    print(f"  +{'-' * (W + 1)}+")
    print(f"  | {'DEATH SEQUENCE':^{W}}|")
    print(f"  +{'-' * (W + 1)}+")
    box(f"{rec.checker}'s heart stops.")
    box(f"Dead for {rec.death_duration:.0f} seconds.")
    box(f"Survival probability: {rec.survival_probability:.1%}")
    box("")

    time.sleep(0.8)

    box("...revival drug administered...")
    time.sleep(0.4)
    box(f"...CPR & recovery ({DEATH_PROCEDURE_OVERHEAD}s)...")
    time.sleep(0.4)

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


# -- Main game loop --------------------------------------------------

def main():
    print_banner()

    p1_name = input("  Player 1 name [Hal]: ").strip() or "Hal"
    p2_name = input("  Player 2 name [Baku]: ").strip() or "Baku"

    p1 = Player(name=p1_name, physicality=PHYSICALITY_HAL)
    p2 = Player(name=p2_name, physicality=PHYSICALITY_BAKU)
    ref = Referee()
    game = Game(player1=p1, player2=p2, referee=ref)

    # Pre-game ceremony: 12 minutes before R1 (matches manga R1 at 8:12)
    game.game_clock = 720.0

    print()
    print(f"  Coin flip... {game.first_dropper.name} drops first!")
    pause("Press Enter to begin the game...")

    while not game.game_over:
        # -- Round header --
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

            drop_t = get_drop_time(dropper.name, turn_dur)
            check_t = get_check_time(checker.name)

            record = game.play_half_round(drop_t, check_t)
            print_half_result(record)

            if not game.game_over and half_num == 1:
                pause("Press Enter for Half 2...")
                print()

        if not game.game_over:
            pause("Press Enter for next round...")

    # -- Game over --
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
