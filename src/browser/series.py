"""One repeated-opponent series, the unit that ``terminal play --games N``
plays: one Hal retained across games, game ``N`` seeded with the base seed
plus ``N``, and every finished game appended to a public transcript in the
CLI's ``arena-public-play-session-v1`` shape.

``app.py`` sequences HTTP requests onto a ``Series`` under its lock. This
module holds no route, no lock, and no serializer; it builds games and keeps
the record of the finished ones.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from arena.contracts import reset_provider_game
from arena.session import (
    CANONICAL_HAL_NAME,
    CANONICAL_HUMAN_NAME,
    Phase,
    PlaySession,
    validate_human_display_name,
)
from arena.transcript import write_play_transcript
from arena.variants import PureDTHGame
from stl.engine.game import (
    OPENING_START_CLOCK,
    PHYSICALITY_BAKU,
    PHYSICALITY_HAL,
    Game,
    Player,
    Referee,
)

TRANSCRIPT_SCHEMA = "arena-public-play-session-v1"


@dataclass
class SessionConfig:
    """Per-game settings. ``pure_dth`` is fixed for the whole server."""

    human_name: str = CANONICAL_HUMAN_NAME
    seed: int | None = None
    start_clock: int = OPENING_START_CLOCK
    max_half_rounds: int | None = None
    pure_dth: bool = False

    def __post_init__(self) -> None:
        self.human_name = validate_human_display_name(self.human_name)
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, int)
        ):
            raise TypeError("seed must be an integer or None")
        if isinstance(self.start_clock, bool) or not isinstance(self.start_clock, int):
            raise TypeError("start_clock must be an integer")
        if self.start_clock < 0:
            raise ValueError("start_clock must be nonnegative")
        if self.max_half_rounds is not None:
            if isinstance(self.max_half_rounds, bool) or not isinstance(
                self.max_half_rounds, int
            ):
                raise TypeError("max_half_rounds must be an integer or None")
            if self.max_half_rounds <= 0:
                raise ValueError("max_half_rounds must be positive")
        if not isinstance(self.pure_dth, bool):
            raise TypeError("pure_dth must be a boolean")


@dataclass
class SeriesConfig:
    """Session-wide presentation and record-keeping, as ``terminal play`` takes it."""

    hal_agent: str = "dth"
    public_hal_label: str | None = None
    conceal_hal_details: bool = False
    transcript_path: Path | None = None

    @property
    def hal_label(self) -> str:
        return self.public_hal_label or self.hal_agent


def new_session(
    hal_agent: object,
    config: SessionConfig,
    *,
    game_index: int = 0,
    game_seed: int | None = None,
    sequence_start: int = 0,
    reset_provider: bool = True,
) -> PlaySession:
    """Build one game with a fresh canonical referee around the retained Hal."""

    hal = Player(name=CANONICAL_HAL_NAME, physicality=PHYSICALITY_HAL)
    human = Player(name=CANONICAL_HUMAN_NAME, physicality=PHYSICALITY_BAKU)
    game_type: type[Game] = Game
    if config.pure_dth:
        game_type = PureDTHGame
    game = game_type(
        player1=hal,
        player2=human,
        referee=Referee(),
        rng=random.Random(game_seed),
    )
    game.game_clock = config.start_clock
    if reset_provider:
        reset_provider_game(getattr(hal_agent, "provider", None))
    return PlaySession(
        game=game,
        hal_agent=hal_agent,
        hal=hal,
        human=human,
        human_display_name=config.human_name,
        game_index=game_index,
        game_seed=game_seed,
        start_clock=config.start_clock,
        max_half_rounds=config.max_half_rounds,
        sequence_start=sequence_start,
        pure_dth=config.pure_dth,
    )


def game_seed(base_seed: int | None, game_index: int) -> int | None:
    """The CLI's per-game seed rule: base seed plus game index."""

    return None if base_seed is None else base_seed + game_index


class Series:
    """The live game, the finished games, and the tally of one series."""

    def __init__(
        self,
        hal_agent: object,
        config: SessionConfig | None = None,
        series: SeriesConfig | None = None,
        *,
        sequence_start: int = 0,
    ) -> None:
        self.hal_agent = hal_agent
        self.config = config or SessionConfig()
        self.series = series or SeriesConfig()
        self.series_seed = self.config.seed
        self.game_index = 0
        self.recorded = False
        self.session = new_session(
            hal_agent,
            self.config,
            game_seed=game_seed(self.config.seed, 0),
            sequence_start=sequence_start,
        )
        self.transcript: dict[str, object] = {
            "schema_version": TRANSCRIPT_SCHEMA,
            "hal_agent": self.series.hal_agent,
            "public_hal_label": self.series.public_hal_label,
            "human_name": self.config.human_name,
            "base_seed": self.config.seed,
            "start_clock": self.config.start_clock,
            "start_clock_sequence": None,
            # A browser series is open-ended; the CLI records its --games count here.
            "requested_games": None,
            "pure_dth": self.config.pure_dth,
            "games": [],
        }
        self.tally = {"human_wins": 0, "hal_wins": 0, "no_winner": 0, "stopped": 0}

    @property
    def provider(self) -> object | None:
        return getattr(self.hal_agent, "provider", None)

    def hal_details(self) -> dict[str, object]:
        details: dict[str, object] = {}
        match_summary = getattr(self.provider, "match_summary", None)
        if callable(match_summary):
            details["hal_summary"] = match_summary()
        diagnostics = getattr(self.provider, "experiment_diagnostics", None)
        if callable(diagnostics):
            details["hal_diagnostics"] = diagnostics()
        return details

    def record_finished_game(self) -> None:
        """Append the terminal live game to the series once, then persist the file."""

        if self.recorded:
            return
        session = self.session
        record = session.finish()
        games = self.transcript["games"]
        assert isinstance(games, list)
        games.append(record)
        if session.stopped:
            self.tally["stopped"] += 1
        elif session.game.winner is None:
            self.tally["no_winner"] += 1
        elif session.game.winner is session.human:
            self.tally["human_wins"] += 1
        else:
            self.tally["hal_wins"] += 1
        self.recorded = True
        if self.series.transcript_path is not None:
            write_play_transcript(
                self.series.transcript_path, {**self.transcript, **self.hal_details()}
            )

    def next_game(
        self,
        *,
        human_name: str | None = None,
        start_clock: int | None = None,
        max_half_rounds: int | None = None,
        seed: int | None = None,
    ) -> PlaySession:
        """Replace the live game. The caller has checked that it may be replaced.

        A finished game is recorded first and advances the game index; the
        provider's game memory resets only then, so a game abandoned on the
        rules screen keeps Hal where it was.
        """

        previous = self.session
        finished = previous.phase is Phase.GAME_OVER
        if finished:
            self.record_finished_game()
            self.game_index += 1
        config = SessionConfig(
            human_name=(
                human_name if human_name is not None else previous.human_display_name
            ),
            seed=self.series_seed,
            start_clock=start_clock if start_clock is not None else self.config.start_clock,
            max_half_rounds=(
                max_half_rounds
                if max_half_rounds is not None
                else self.config.max_half_rounds
            ),
            pure_dth=self.config.pure_dth,
        )
        self.session = new_session(
            self.hal_agent,
            config,
            game_index=self.game_index,
            game_seed=seed if seed is not None else game_seed(self.series_seed, self.game_index),
            sequence_start=previous.sequence + 1,
            reset_provider=finished,
        )
        self.recorded = False
        self.transcript["human_name"] = config.human_name
        return self.session

    def restart(self) -> PlaySession:
        """Abandon the live game and clear the series, keeping the sequence."""

        sequence = self.session.sequence + 1
        self.series_seed = game_seed(self.config.seed, sequence)
        self.session = new_session(
            self.hal_agent, self.config, game_seed=self.series_seed, sequence_start=sequence
        )
        self.game_index = 0
        self.recorded = False
        self.transcript["games"] = []
        self.transcript["human_name"] = self.config.human_name
        self.transcript["base_seed"] = self.series_seed
        self.tally.update({key: 0 for key in self.tally})
        return self.session

    def public_transcript(self) -> dict[str, object]:
        """The public series so far: finished games plus the live game's reveals.

        Everything here has already been revealed on an outcome screen. The
        live game's history contains only resolved half-rounds, so a request
        made while the human is deciding cannot see an unrevealed second.
        """

        session = self.session
        payload: dict[str, object] = {
            **self.transcript,
            "tally": dict(self.tally),
            "current_game": {
                "game_index": session.game_index,
                "seed": session.game_seed,
                "start_clock": session.start_clock,
                "phase": session.phase.value,
                "half_rounds": session.half_rounds,
                "public_history": list(session.public_history),
            },
        }
        if not self.series.conceal_hal_details:
            payload.update(self.hal_details())
        return payload
