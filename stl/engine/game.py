"""Engine truth for Surpassing The Leader / Drop the Handkerchief.

This module intentionally collapses the old Constants, Player, Referee, and
Game modules into one import surface. Solver and learning code should depend
on this module rather than reimplementing game rules.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

"""
Drop The Handkerchief — Game Constants
All timing values are in seconds unless otherwise noted.
The game clock is absolute: second 0 = 8:00:00 AM.
"""
# ──────────────────────────────────────────────
# CLOCK
# ──────────────────────────────────────────────
GAME_START_HOUR = 8                 # 8:00 AM
SECONDS_PER_MINUTE = 60
MINUTES_PER_HOUR = 60
OPENING_START_CLOCK = 12 * 60       # Canonical R1 start: 8:12:00 AM
# Leap second: inserted at exactly 8:59:60 AM.
# In game clock seconds: 8:59:00 AM = 59 * 60 = 3540s after game start.
# A half-round whose start falls in the CLOSED interval [3540, 3600] spans the
# leap second — a half-round starting at exactly 3600 (8:59:60) still spans
# the inserted second.
LS_WINDOW_START = 59 * 60         # 3540 — start of the 8:59 minute
LS_WINDOW_END = 60 * 60               #8:59:60 AM (inclusive)
# ──────────────────────────────────────────────
# TURN
# ──────────────────────────────────────────────
TURN_DURATION_NORMAL = 60           # seconds per half-round (normal)
TURN_DURATION_LEAP = 61             # seconds per half-round (during LS window)
FAILED_CHECK_PENALTY = 60           # 1 minute NDD added on failed check
# ──────────────────────────────────────────────
# CYLINDER / NDD
# ──────────────────────────────────────────────
CYLINDER_MAX = 300                  # 5 minutes — at or above this, instant injection
DEATH_PROCEDURE_OVERHEAD = 120      # injection + waiting + CPR + recovery (~2 min)
# ──────────────────────────────────────────────
# CLOCK ADVANCEMENT
#
# Within a round the game clock advances continuously:
#   turn_duration (per half) + death_duration + DEATH_PROCEDURE_OVERHEAD (if death)
#   + WITHIN_ROUND_OVERHEAD
#
# After a round completes, the clock SNAPS FORWARD to the next whole-minute boundary.
# This snap absorbs imprecision and matches the manga, where every round starts on
# an exact minute (R1: 8:12, R2: 8:19, R3: 8:26, R4: 8:30, etc.).
#
# No-death round: 60 + 60 + 60 (overhead) = 180s. Snap → 4-minute gap between rounds.
# Death round (60s death): 60 + 60 + 120 + 60 + 60 = 360s. Snap → 7-minute gap.
# Manga: "1 MINUTE OF NEAR-DEATH WOULD CONSUME 5 MINUTES OF GAME TIME"
# ──────────────────────────────────────────────
WITHIN_ROUND_OVERHEAD = 60          # total procedural time within a round (settling,
                                    # injection procedure, role swap). Applied between halves.
# ──────────────────────────────────────────────
# SURVIVAL PROBABILITY
#
# P(survival) = base_curve(t) * cardiac(ttd_prior) * referee(n) * physicality
#
# base_curve(t)       = max(0, 1 - (t / CYLINDER_MAX) ^ BASE_CURVE_K)
# cardiac(ttd_prior)  = CARDIAC_DECAY ^ (ttd_prior / 60)
# referee(n)          = max(REFEREE_FLOOR, REFEREE_DECAY ^ n)
# physicality         = per-player constant
# ──────────────────────────────────────────────
# Base curve: oxygen deprivation danger for THIS death
BASE_CURVE_K = 3
# Cardiac degradation: accumulated heart damage from prior deaths
# Each minute of prior cumulative death weakens the heart by this factor
CARDIAC_DECAY = 0.85
# Referee fatigue: CPR quality degrades with each revival performed
REFEREE_DECAY = 0.88
REFEREE_FLOOR = 0.4                 # minimum effectiveness (even exhausted)
# Player physicality presets
PHYSICALITY_HAL = 1.0
PHYSICALITY_BAKU = 0.94

from dataclasses import dataclass, field

@dataclass
class Player:
    name: str
    physicality: float = 1.0

    # ── Mutable game state ──
    cylinder: float = 0.0
    ttd: float = 0.0
    deaths: int = 0
    alive: bool = True

    # ── Per-death history (for analysis / debugging) ──
    # Each entry is the duration of one death episode in seconds.
    death_history: list[float] = field(default_factory=list)

    @property
    def safe_strategies_remaining(self) -> int:
        """
        How many consecutive safe-strategy checks can this player do
        without the cylinder hitting CYLINDER_MAX?

        Safe if: cylinder + 60 < CYLINDER_MAX (strictly less — 300 triggers injection).

        Returns the number of times the player can absorb a worst-case safe check.
        """
        return max(0, int((CYLINDER_MAX - 1 - self.cylinder) // TURN_DURATION_NORMAL))
    

    def add_to_cylinder(self, amount: float) -> bool:
        """
        Add NDD to the cylinder.

        Returns True if cylinder >= CYLINDER_MAX after addition,
        meaning immediate injection is triggered.
        """
        self.cylinder += amount
        return self.cylinder >= CYLINDER_MAX

    def on_death(self, death_duration: float) -> None:
        """
        Called when this player dies.
        - Increment deaths count
        - Add death_duration to ttd
        - Append to death_history
        - Set alive = False
        """
        self.deaths += 1
        self.ttd += death_duration
        self.death_history.append(death_duration)
        self.alive = False

    def on_revival(self) -> None:
        """
        Called when revival succeeds.
        - Reset cylinder to 0
        - Set alive = True
        """
        self.cylinder = 0
        self.alive = True

    def on_permanent_death(self) -> None:
        """
        Called when revival fails. Player is dead for good.
        - alive stays False
        - This is a terminal state for the player.
        """
        self.alive = False

"""
Drop The Handkerchief — Referee

The Referee (Yakou) manages the death/revival process.

The referee has ONE job in the engine: determine whether a player survives
a death episode. This involves:

1. Computing the survival probability from four independent factors.
2. Rolling against that probability.
3. Tracking CPR fatigue (how many revivals have been performed total).

The referee is a GLOBAL resource — fatigue accumulates regardless of
which player died. If Baku dies twice and Hal dies once, Yakou has
performed 3 CPRs and is exhausted for whoever needs revival next.

Survival Probability:
    P = base_curve(this_death_duration)
      x cardiac_modifier(player.ttd before this death)
      x referee_modifier(self.cprs_performed)
      x player.physicality

Where:
    base_curve(t)           = max(0, 1 - (t / 300)^k)
    cardiac_modifier(ttd)   = α^(ttd / 60)
    referee_modifier(n)     = max(β_min, β^n)
"""

import random
from dataclasses import dataclass


@dataclass
class Referee:
    cprs_performed: int = 0

    def compute_survival_probability(self, player: Player, death_duration: float) -> float:
        """
        Compute the probability that `player` survives a death of `death_duration` seconds.

        Args:
            player: The player who is dying. Use player.ttd (BEFORE this death
                    is added) for cardiac modifier, and player.physicality for baseline.
            death_duration: How long the player will be dead THIS time (seconds).
                            This is the cylinder contents at time of injection.

        Returns:
            Float in [0.0, 1.0] — probability of successful revival.

        Implementation notes:
            - base_curve uses death_duration (THIS episode's danger)
            - cardiac uses player.ttd (accumulated PRIOR damage to the heart)
            - referee uses self.cprs_performed (global fatigue)
            - physicality is player.physicality (constant per player)
            - Multiply all four. Clamp to [0.0, 1.0].
            - At death_duration >= 300: always return 0.0 regardless of other factors.
        """
        def death_curve(t):
            return max(0, 1- (t/300)**BASE_CURVE_K)
        def cardiac_modifier(ttd):
            return CARDIAC_DECAY**(ttd/60)
        def referee_modifier(n):
            return max(REFEREE_FLOOR, REFEREE_DECAY**n)
        death_pr = death_curve(death_duration) * cardiac_modifier(player.ttd) * referee_modifier(self.cprs_performed) * player.physicality

        return 0.0 if death_duration >= 300 else death_pr

    def attempt_revival(self, player: Player, death_duration: float, rng: random.Random | None = None) -> bool:
        """
        Attempt to revive a player. Rolls against survival probability.

        Args:
            player: The dying player. NOTE: player.ttd should reflect state
                    BEFORE this death (on_death hasn't been called yet, or if it has,
                    you need to subtract this death_duration back out for the cardiac calc).
            death_duration: Duration of this death episode.
            rng: Optional seeded RNG for deterministic testing. If None, use random.random().

        Returns:
            True if revival succeeds, False if player dies permanently.

        Side effects:
            - Increments self.cprs_performed (always, regardless of outcome).
            - On the Player side, the caller (Game) should handle on_death/on_revival/on_permanent_death.

        Design note:
            The referee doesn't mutate the player — it only reads player state and
            returns a bool. The Game object orchestrates the state transitions.
        """
        if rng is None:
            rng = random.Random()
            
        prob = self.compute_survival_probability(player, death_duration)
        roll = rng.random()
        survived = prob > roll
        self.cprs_performed += 1
        return survived

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
        leap second window [LS_WINDOW_START, LS_WINDOW_END] — the CLOSED
        interval [3540, 3600]: a half-round starting at exactly 3600
        (8:59:60) still spans the inserted second.
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
        Settled: the starting D does NOT alternate across rounds —
        ``first_dropper`` is fixed for the whole match, decided by
        rock-paper-scissors before the game (canon doc: First D = Leader
        every round R1-R9).
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
        """Execute one half-round, rolling RNG for any required survival check.
        This is the main entry point for normal gameplay. For deterministic
        forward simulation (e.g., search), use resolve_half_round directly with
        an explicit survived_outcome.
        """
        return self.resolve_half_round(drop_time, check_time, survived_outcome=None)
    def resolve_half_round(
        self,
        drop_time: int,
        check_time: int,
        survived_outcome: Optional[bool] = None,
    ) -> HalfRoundRecord:
        """Apply a half-round, optionally with an externally-supplied survival outcome.
        This is the single source of truth for half-round resolution. Both
        play_half_round (which passes survived_outcome=None to roll the RNG) and
        Hal's search forward simulator (which branches on survived_outcome=True
        and survived_outcome=False) call this method.
        Args:
            drop_time: Second at which D drops. Must be in [1, turn_duration].
            check_time: Second at which C checks. Must be in [1, turn_duration].
            survived_outcome: If a death occurs, this is the survival outcome to
                use. None means "roll the RNG" (live play). True/False means
                "force this outcome" (search forward simulation).
        Returns:
            HalfRoundRecord describing everything that happened.
        """
        if self.game_over:
            raise GameOverError("Game is Already Over")
        clock_at_start = self.game_clock
        dropper, checker = self.get_roles_for_half(self.current_half)
        turn_duration = self.get_turn_duration()
        self.validate_drop_time(drop_time, turn_duration)
        self.validate_check_time(check_time, turn_duration)
        success = check_time >= drop_time
        death_occurred = False
        death_duration = 0.0
        survived: Optional[bool] = None
        survival_probability: Optional[float] = None
        ST = 0.0
        if success:
            ST = max(1, check_time - drop_time)
            overflow = checker.add_to_cylinder(ST)
            if overflow:
                death_occurred = True
                death_duration = min(checker.cylinder, CYLINDER_MAX)
        else:
            checker.add_to_cylinder(FAILED_CHECK_PENALTY)
            death_occurred = True
            death_duration = min(checker.cylinder, CYLINDER_MAX)
        if death_occurred:
            survival_probability = self.referee.compute_survival_probability(
                checker, death_duration=death_duration
            )
            if survived_outcome is None:
                survived = self.referee.attempt_revival(
                    checker, death_duration=death_duration, rng=self.rng
                )
            else:
                survived = survived_outcome
                self.referee.cprs_performed += 1
            checker.on_death(death_duration=death_duration)
            if survived:
                checker.on_revival()
            else:
                checker.on_permanent_death()
                self.game_over = True
                self.winner = dropper
                self.loser = checker
        self.advance_clock(turn_duration)
        if death_occurred:
            self.advance_clock(death_duration + DEATH_PROCEDURE_OVERHEAD)
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
        if not self.game_over:
            if self.current_half == 1:
                self.advance_clock(WITHIN_ROUND_OVERHEAD)
                self.current_half = 2
            else:
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
