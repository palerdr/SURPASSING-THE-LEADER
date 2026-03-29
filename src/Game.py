"""
Drop The Handkerchief — Game

The Game is the state machine that orchestrates everything.
It owns two Players, one Referee, and the game clock.

A Game consists of Rounds. Each Round has two HalfRounds.
Each HalfRound has a Dropper and a Checker.

Flow per HalfRound:
    1. Determine turn_duration (60 normally, 61 if leap second window).
    2. D chooses drop_time ∈ [1, turn_duration].
    3. C chooses check_time ∈ [1, turn_duration].  (but C doesn't know about 61)
    4. Resolve: successful check or failed check.
    5. On success: add ST to C's cylinder. Check for overflow.
    6. On failure: add penalty to C's cylinder. Inject. Death sequence.
    7. Advance game clock.
    8. Check for game-over conditions.

The Game does NOT decide what actions to take — it receives them.
Action selection is the responsibility of the caller (CLI, AI, Gym env).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .Player import Player
from .Referee import Referee

from .Constants import LS_WINDOW_START, LS_WINDOW_END, OPENING_START_CLOCK, WITHIN_ROUND_OVERHEAD
from .Constants import TURN_DURATION_LEAP, TURN_DURATION_NORMAL, CYLINDER_MAX, FAILED_CHECK_PENALTY, DEATH_PROCEDURE_OVERHEAD


class HalfRoundResult(Enum):
    """Outcome of a single half-round."""
    CHECK_SUCCESS = "check_success"                         # C found the handkerchief
    CHECK_FAIL_SURVIVED = "check_fail_survived"             # C failed, died, but was revived
    CHECK_FAIL_DIED = "check_fail_died"                     # C failed, died permanently → game over
    CYLINDER_OVERFLOW_SURVIVED = "overflow_survived"        # C's cylinder hit 300 from ST accumulation, survived
    CYLINDER_OVERFLOW_DIED = "overflow_died"                # C's cylinder hit 300 from ST accumulation, died


@dataclass
class HalfRoundRecord:
    """Immutable record of what happened in one half-round. For history/analysis."""
    round_num: int
    half: int                        # 1 or 2
    dropper: str                     # player name
    checker: str                     # player name
    drop_time: int                   # second D dropped
    check_time: int                  # second C checked
    turn_duration: int               # 60 or 61
    result: HalfRoundResult
    st_gained: float                 # squandered time C accumulated (0 if failed check)
    death_duration: float            # how long C was dead (0 if no death)
    survived: Optional[bool]         # None if no death, True/False if death occurred
    game_clock_at_start: float       # absolute game clock when this half-round began
    survival_probability: Optional[float]  # the computed P before the roll (None if no death)


@dataclass
class Game:
    player1: Player
    player2: Player
    referee: Referee
    first_dropper: Optional[Player] = None

    # ── Game clock (seconds since 8:00:00 AM) ──
    # Canonical opening is R1 at 8:12:00 AM.
    game_clock: float = OPENING_START_CLOCK

    # ── Round tracking ──
    current_half: int = 1
    round_num: int = 0
    game_over: bool = False
    winner: Optional[Player] = None
    loser: Optional[Player] = None

    # ── History ──
    history: list[HalfRoundRecord] = field(default_factory=list)

    # ── RNG (seed for deterministic replays) ──
    rng: random.Random = field(default_factory=random.Random)

    def __post_init__(self):
        if self.first_dropper is None:
            self.first_dropper = self.player1

    def seed(self, seed: int) -> None:
        """Seed the RNG for deterministic games."""
        self.rng.seed(seed)

    # CLOCK

    def get_turn_duration(self) -> int:
        """
        Returns 60 normally, 61 if the current game_clock falls in the
        leap second window [LS_WINDOW_START, LS_WINDOW_END).

        This is called at the START of a half-round to determine how long
        the turn lasts. The D player gets the full turn_duration to drop.
        The C player's action space is still [1, 60] — they don't know
        about the extra second unless they've figured it out.
        """
        if LS_WINDOW_START <= self.game_clock <= LS_WINDOW_END:
            return TURN_DURATION_LEAP
        return TURN_DURATION_NORMAL

    def advance_clock(self, elapsed: float) -> None:
        """Advance game clock by the given number of elapsed seconds."""
        self.game_clock += elapsed

    def snap_clock_to_next_minute(self) -> None:
        """
        Round the game clock UP to the next wall-clock minute boundary.

        Always advances to the next minute, even if already on a boundary.

        Pre-leap (gc < 3600): wall-clock minutes are at multiples of 60.
            Example: 1740 (8:29:00) → 1800 (8:30:00)
                     1753 (8:29:13) → 1800 (8:30:00)

        At/near leap second (gc 3540–3600): the 8:59 minute has 61 game-clock
        seconds (the leap second), so the next minute (9:00:00) is gc 3601.

        Post-leap (gc > 3600): the leap second inserted an extra game-clock
        second, so wall-clock minutes are at 3601 + n*60, not multiples of 60.
            Example: 3810 → 3841 (9:04:00), not 3840 (which would be 9:03:59)
        """
        gc = int(self.game_clock)
        if gc < 3600:
            snapped = ((gc // 60) + 1) * 60
            # If snap lands on 3600 (the leap second itself, 8:59:60),
            # push to 3601 (9:00:00) — 3600 is not a minute boundary.
            if snapped == 3600:
                snapped = 3601
            self.game_clock = snapped
        elif gc <= 3600:
            # At the leap second: next wall-clock minute is 9:00:00
            self.game_clock = 3601
        else:
            # Post-leap: wall-clock minutes at 3601, 3661, 3721, 3781, ...
            elapsed = gc - 3601
            self.game_clock = 3601 + ((elapsed // 60) + 1) * 60

    def format_game_clock(self) -> str:
        """
        Format the current game clock as a human-readable time string.
        e.g., game_clock=720 → "8:12 AM"
        MUST consider that the leap second is literally inserted into reality for calculations after 8:59:60
        Useful for CLI display and matching against manga timestamps.
        """
        gc = int(self.game_clock)
    
        if gc <= 3599:
            # Pre-leap: normal conversion
            total_seconds = gc
            hours = 8 + total_seconds // 3600
            remainder = total_seconds % 3600
            minutes = remainder // 60
            seconds = remainder % 60
        elif gc == 3600:
            # THE leap second
            hours, minutes, seconds = 8, 59, 60
        else:
            # Post-leap: subtract 1 because one elapsed second was the inserted LS
            total_seconds = gc - 1
            hours = 8 + total_seconds // 3600
            remainder = total_seconds % 3600
            minutes = remainder // 60
            seconds = remainder % 60
        
        return f"{hours}:{minutes:02d}:{seconds:02d} AM"

    # ROLE ASSIGNMENT

    def get_roles_for_half(self, half: int) -> tuple[Player, Player]:
        """
        Returns (dropper, checker) for the given half of the current round.

        Half 1: whoever is D this round drops, the other checks.
        Half 2: roles swap.

        Who starts as D in round 1 is determined before the game
        (in the manga: rock-paper-scissors). After that, the starting D
        alternates each round? Or stays the same?

        They play rock paper scissors to decide so it's a coin flip 
        """
        if self.first_dropper == self.player1:
            if half == 1:
                D = self.player1
                C = self.player2
            else:
                D = self.player2
                C = self.player1
        else:
            if half == 1:
                D = self.player2
                C = self.player1
            else:
                D = self.player1
                C = self.player2
        
        return D, C

    
    # CORE GAME LOOP


    def play_half_round(self, drop_time: int, check_time: int) -> HalfRoundRecord:
        """
        Execute one half-round with the given actions.

        This is the main entry point. The caller provides drop_time and check_time.
        The Game resolves the outcome, updates all state, and returns a record.

        Args:
            drop_time: Second at which D drops the handkerchief. Must be in [1, turn_duration].
            check_time: Second at which C checks. Must be in [1, 60].
                        (C's max is always 60 — they don't know about leap second.)

        Returns:
            HalfRoundRecord describing everything that happened.

        Raises:
            ValueError: If actions are out of valid range.
            GameOverError: If the game is already over.

        Flow:
            1. Validate inputs.
            2. Determine turn_duration (check leap second).
            3. Resolve check: success or failure.
            4. If success:
                a. Compute ST = check_time - drop_time.
                b. Add ST to checker's cylinder.
                c. If cylinder >= CYLINDER_MAX: overflow → death sequence.
            5. If failure:
                a. Add FAILED_CHECK_PENALTY to checker's cylinder.
                b. Inject entire cylinder → death sequence.
            6. If death sequence:
                a. Compute death_duration = cylinder contents at injection.
                b. Compute survival probability.
                c. Roll for survival.
                d. If survived: call on_death then on_revival on the player.
                e. If died: call on_death then on_permanent_death. Set game_over.
            7. Advance game clock appropriately.
            8. Record and return the HalfRoundRecord.
        """
        if self.game_over:
            raise GameOverError("Game is Already Over")

        # Snapshot clock BEFORE any advancement for the record
        clock_at_start = self.game_clock

        dropper, checker = self.get_roles_for_half(self.current_half)
        turn_duration = self.get_turn_duration()

        self.validate_drop_time(drop_time, turn_duration)
        self.validate_check_time(check_time, turn_duration)

        # Successful check: handkerchief is on the ground when C looks.
        # check_time >= drop_time means C sees it (even mid-drop).
        # Yakou rules ST cannot be 0 — minimum is 1 second.
        success = check_time >= drop_time
        death_occurred = False
        death_duration = 0.0
        survived = None
        survival_probability = None
        ST = 0.0

        if success:
            ST = max(1, check_time - drop_time)
            overflow = checker.add_to_cylinder(ST)
            if overflow:
                death_occurred = True
                death_duration = checker.cylinder
        else:
            checker.add_to_cylinder(FAILED_CHECK_PENALTY)
            death_occurred = True
            death_duration = checker.cylinder

        if death_occurred:
            survival_probability = self.referee.compute_survival_probability(
                checker, death_duration=death_duration
            )
            survived = self.referee.attempt_revival(
                checker, death_duration=death_duration, rng=self.rng
            )

            checker.on_death(death_duration=death_duration)
            if survived:
                checker.on_revival()
            else:
                checker.on_permanent_death()
                self.game_over = True
                self.winner = dropper
                self.loser = checker

        # Turn itself
        self.advance_clock(turn_duration)
        # Death procedure: player is dead for death_duration, then injection/CPR/recovery overhead
        if death_occurred:
            self.advance_clock(death_duration + DEATH_PROCEDURE_OVERHEAD)

        # Determine result enum
        if not death_occurred:
            result = HalfRoundResult.CHECK_SUCCESS
        elif success and survived:
            result = HalfRoundResult.CYLINDER_OVERFLOW_SURVIVED
        elif success and not survived:
            result = HalfRoundResult.CYLINDER_OVERFLOW_DIED
        elif not success and survived:
            result = HalfRoundResult.CHECK_FAIL_SURVIVED
        else:
            result = HalfRoundResult.CHECK_FAIL_DIED

        # Build record
        record = HalfRoundRecord(
            round_num=self.round_num,
            half=self.current_half,
            dropper=dropper.name,
            checker=checker.name,
            drop_time=drop_time,
            check_time=check_time,
            turn_duration=turn_duration,
            result=result,
            st_gained=ST,
            death_duration=death_duration if death_occurred else 0.0,
            survived=survived,
            game_clock_at_start=clock_at_start,
            survival_probability=survival_probability,
        )
        self.history.append(record)

        # ── Advance half/round counters + inter-half/inter-round clock ──
        if not self.game_over:
            if self.current_half == 1:
                # Mid-round: add within-round overhead (settling, role swap)
                # No snapping — the round is still in progress.
                self.advance_clock(WITHIN_ROUND_OVERHEAD)
                self.current_half = 2
            else:
                # Round complete: snap clock to the next whole-minute boundary.
                # This models the between-round gap observed in the manga where
                # every round starts on an exact minute mark.
                self.snap_clock_to_next_minute()
                self.current_half = 1
                self.round_num += 1

        return record

    def play_round(self, half1_drop: int, half1_check: int,
                   half2_drop: int, half2_check: int) -> list[HalfRoundRecord]:
        """
        Convenience method: play both halves of a round.

        Returns list of 1 or 2 HalfRoundRecords (may be 1 if game ends in half 1).
        """
        records: list[HalfRoundRecord] = []

        record1 = self.play_half_round(half1_drop, half1_check)
        records.append(record1)

        if not self.game_over:
            record2 = self.play_half_round(half2_drop, half2_check)
            records.append(record2)

        return records


    # VALIDATION


    def validate_drop_time(self, drop_time: int, turn_duration: int) -> None:
        """
        Validate that drop_time is in [1, turn_duration].
        Raise ValueError with descriptive message if not.

        Note: drop_time can be 61 during a leap second turn.
        This is the WHOLE POINT of the leap second — D gets
        an extra second that C doesn't know about.
        """
        if not (1 <= drop_time <= turn_duration):
            raise ValueError(
                f"drop_time must be in [1, {turn_duration}], got {drop_time}"
            )

    def validate_check_time(self, check_time: int, turn_duration: int) -> None:
        """
        Validate that check_time is in [1, turn_duration].

        The engine allows check up to turn_duration (61 during leap turns).
        Knowledge-based restrictions (checker doesn't know about LS) are
        enforced by the environment's action mask, not here.
        """
        if not (1 <= check_time <= turn_duration):
            raise ValueError(
                f"check_time must be in [1, {turn_duration}], got {check_time}"
            )

    # GAME STATE QUERIES

    def is_leap_second_turn(self) -> bool:
        """Is the current game clock in the leap second window?"""
        return LS_WINDOW_START <= self.game_clock <= LS_WINDOW_END

    def get_state_summary(self) -> dict:
        """
        Return a dict summarizing the current game state.
        Useful for CLI display, logging, and later for Gym observations.
        """
        def player_summary(p: Player) -> dict:
            return {
                "name": p.name,
                "cylinder": p.cylinder,
                "ttd": p.ttd,
                "deaths": p.deaths,
                "alive": p.alive,
                "safe_strategies_remaining": p.safe_strategies_remaining,
                "physicality": p.physicality,
            }

        return {
            "round_num": self.round_num,
            "current_half": self.current_half,
            "game_clock": self.game_clock,
            "game_clock_display": self.format_game_clock(),
            "is_leap_second_turn": self.is_leap_second_turn(),
            "player1": player_summary(self.player1),
            "player2": player_summary(self.player2),
            "referee_cprs": self.referee.cprs_performed,
            "game_over": self.game_over,
            "winner": self.winner.name if self.winner else None,
            "loser": self.loser.name if self.loser else None,
        }


class GameOverError(Exception):
    """Raised when trying to play after the game has ended."""
    pass
