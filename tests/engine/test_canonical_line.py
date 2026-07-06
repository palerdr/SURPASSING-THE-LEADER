"""Engine-faithfulness regression: replay the canonical Usogui match line.

Replays the canonical match from the manga — as transcribed in the strategy
doc "SURPASSING THE LEADER - HAL DOC" — through src/Game.py action by action,
asserting game clock, both cylinders, ttd, and referee CPR count at every
half-round boundary. This locks the engine against the canon and the doc
transcription against the engine: if either the engine's clock/cylinder/death
arithmetic or the doc's state tables drift, this file fails.

Replay semantics: every death in the canonical line was survived in the manga,
so death half-rounds are resolved with resolve_half_round(..., survived_outcome=True)
(forced outcome). The forced path still increments referee.cprs_performed, so
referee fatigue accumulates exactly as in live play.

KNOWN CANON-VS-ENGINE DISCREPANCIES (deliberate, documented):

1. Insta-drop 1-second drift. The engine floors actions at second 1
   (drop_time in [1, turn_duration]) and scores ST = max(1, check - drop);
   the tie rule "ST cannot be 0" is confirmed canon. The doc's narrative,
   however, treats an instant drop as second ~0, so when Hal insta-drops and
   Baku checks at second N the doc credits Baku ST = N while the engine
   produces ST = N - 1. This makes engine STs 1s lower than the doc narrative
   for every insta-drop half-round (R7H1: doc 60 vs engine 59; R8H1: doc 55
   vs engine 54; R9H1: doc 60 vs engine 59). The drift accumulates in Baku's
   cylinder: doc round-start values R8 B-1m(60) / R9 B-1m55s(115) correspond
   to engine values 59 / 113, and Baku's cylinder after R9H1 is 172 (doc-
   narrative arithmetic would give 175). The assertions below pin the ENGINE
   values.

2. R9H2 survival probability. hal/HAL.md derives ~0.28 for Hal's R9H2 death;
   that derivation is stale. With the engine's actual state at the R9H2
   injection — death_duration=60, hal.ttd=238 (prior damage, before this
   death), cprs_performed=4, physicality=1.0 — the referee computes
       P = (1 - (60/300)**3) * 0.85**(238/60) * max(0.4, 0.88**4) * 1.0
         = 0.992 * 0.85**(238/60) * 0.88**4
         ~= 0.312
   test_r9h2_survival_probability pins this corrected value.

Clock pins: rows whose start times the doc states explicitly (8:12, 8:17,
8:19, 8:21, 8:26, 8:28, 8:30, 8:32, 8:34, 8:36, 8:38, 8:45, 8:49, 8:57, 8:59)
are canon pins; rows marked "engine" (R6H2 @2613, R7H2 @2820, R8H2 @3060) are
mid-round clocks the doc does not state — they are engine-consistency locks
computed from the engine's clock rules (turn 60s, death +duration+120s, +60s
between halves, snap-to-next-minute after half 2). Death durations are pinned
by the doc (60, 84, 93, 154, 60), so all clock gaps are forced.
"""

from typing import NamedTuple, Optional

import pytest

from stl.engine.game import Player
from stl.engine.game import Referee
from stl.engine.game import Game, HalfRoundResult
from stl.engine.game import PHYSICALITY_HAL, PHYSICALITY_BAKU
from stl.engine.actions import legal_max_second


class Row(NamedTuple):
    label: str
    clock_at_start: float        # asserted before the half-round is played
    dropper: str
    drop: int
    check: int
    survived: Optional[bool]     # None = no death; True = forced revival (replay)
    st: float                    # expected st_gained on the record
    death_duration: float        # expected death_duration on the record (0 if none)
    hal_cyl: float               # ── post-half state assertions ──
    baku_cyl: float
    hal_ttd: float
    baku_ttd: float
    cprs: int


# The canonical line. One Row per half-round, in play order.
# Post-state columns are the FULL state after the half-round resolves
# (cylinder reset on revival already applied).
CANONICAL_LINE = [
    #    label    clock    dropper  drop check survived   st death | hal_cyl baku_cyl hal_ttd baku_ttd cprs
    Row("R1H1",  720.0,  "Hal",   60,  30,  True,       0,  60,     0,      0,      0,     60,     1),  # 8:12 pin — Baku fails, dies 60s
    Row("R1H2", 1020.0,  "Baku",   1,  25,  None,      24,   0,    24,      0,      0,     60,     1),  # 8:17 pin
    Row("R2H1", 1140.0,  "Hal",   35,  60,  None,      25,   0,    24,     25,      0,     60,     1),  # 8:19 pin
    Row("R2H2", 1260.0,  "Baku",  60,   5,  True,       0,  84,     0,     25,     84,     60,     2),  # 8:21 pin — Hal fails, dies 84s (24+60)
    Row("R3H1", 1560.0,  "Hal",   56,  60,  None,       4,   0,     0,     29,     84,     60,     2),  # 8:26 pin
    Row("R3H2", 1680.0,  "Baku",  24,  60,  None,      36,   0,    36,     29,     84,     60,     2),  # 8:28 pin
    Row("R4H1", 1800.0,  "Hal",    7,  10,  None,       3,   0,    36,     32,     84,     60,     2),  # 8:30 pin
    Row("R4H2", 1920.0,  "Baku",  26,  60,  None,      34,   0,    70,     32,     84,     60,     2),  # 8:32 pin
    Row("R5H1", 2040.0,  "Hal",    1,   1,  None,       1,   0,    70,     33,     84,     60,     2),  # 8:34 pin — tie, ST floors to 1
    Row("R5H2", 2160.0,  "Baku",   1,  16,  None,      15,   0,    85,     33,     84,     60,     2),  # 8:36 pin — doc: "waits 15s", ST=15
    Row("R6H1", 2280.0,  "Hal",   60,   1,  True,       0,  93,    85,      0,     84,    153,     3),  # 8:38 pin — Baku fails, dies 93s (33+60)
    Row("R6H2", 2613.0,  "Baku",   2,  10,  None,       8,   0,    93,      0,     84,    153,     3),  # engine: 2280+60+(93+120)+60
    Row("R7H1", 2700.0,  "Hal",    1,  60,  None,      59,   0,    93,     59,     84,    153,     3),  # 8:45 pin — insta-drop: doc ST 60, engine 59
    Row("R7H2", 2820.0,  "Baku",   1,   2,  None,       1,   0,    94,     59,     84,    153,     3),  # engine: 2700+60+60
    Row("R8H1", 2940.0,  "Hal",    1,  55,  None,      54,   0,    94,    113,     84,    153,     3),  # 8:49 pin — insta-drop: doc ST 55, engine 54
    Row("R8H2", 3060.0,  "Baku",  60,   1,  True,       0, 154,     0,    113,    238,    153,     4),  # engine: 2940+60+60 — Hal dies 154s (94+60)
    Row("R9H1", 3420.0,  "Hal",    1,  60,  None,      59,   0,     0,    172,    238,    153,     4),  # 8:57 pin — insta-drop: doc ST 60, engine 59
    Row("R9H2", 3540.0,  "Baku",  61,  60,  True,       0,  60,     0,    172,    298,    153,     5),  # 8:59 pin — THE LEAP TURN: Baku drops 61,
    #                                                                                                     Hal capped at 60 → fail, dies 60s (0+60).
    #                                                                                                     hal.ttd = 298 (4m58s): the 2-Second
    #                                                                                                     Deviation payoff vs a 300s overflow death.
]


def _make_game() -> Game:
    hal = Player(name="Hal", physicality=PHYSICALITY_HAL)
    baku = Player(name="Baku", physicality=PHYSICALITY_BAKU)
    game = Game(player1=hal, player2=baku, referee=Referee(), first_dropper=hal)
    game.game_clock = 720.0
    game.seed(0)
    return game


def _replay(game: Game, rows) -> None:
    for row in rows:
        game.resolve_half_round(row.drop, row.check, survived_outcome=row.survived)


def test_canonical_line_replay():
    """Replay all 18 canonical half-rounds, asserting clock + full state each row."""
    game = _make_game()
    hal, baku, ref = game.player1, game.player2, game.referee

    for row in CANONICAL_LINE:
        # Clock at the START of the half-round (doc pin or engine consistency lock).
        assert game.game_clock == pytest.approx(row.clock_at_start), (
            f"{row.label}: clock at start — expected {row.clock_at_start}, "
            f"engine has {game.game_clock} ({game.format_game_clock()})"
        )

        record = game.resolve_half_round(row.drop, row.check, survived_outcome=row.survived)

        # Record-level locks.
        assert record.dropper == row.dropper, f"{row.label}: dropper"
        assert record.game_clock_at_start == pytest.approx(row.clock_at_start), f"{row.label}: record clock"
        assert record.st_gained == pytest.approx(row.st), f"{row.label}: st_gained"
        assert record.death_duration == pytest.approx(row.death_duration), f"{row.label}: death_duration"
        if row.survived is None:
            assert record.result == HalfRoundResult.CHECK_SUCCESS, f"{row.label}: result"
            assert record.survived is None, f"{row.label}: survived"
        else:
            assert record.result == HalfRoundResult.CHECK_FAIL_SURVIVED, f"{row.label}: result"
            assert record.survived is True, f"{row.label}: survived"

        # Full post-half state lock.
        assert hal.cylinder == pytest.approx(row.hal_cyl), f"{row.label}: hal.cylinder"
        assert baku.cylinder == pytest.approx(row.baku_cyl), f"{row.label}: baku.cylinder"
        assert hal.ttd == pytest.approx(row.hal_ttd), f"{row.label}: hal.ttd"
        assert baku.ttd == pytest.approx(row.baku_ttd), f"{row.label}: baku.ttd"
        assert ref.cprs_performed == row.cprs, f"{row.label}: cprs_performed"
        assert not game.game_over, f"{row.label}: game should continue"

    # Whole-line sanity: 5 deaths total, both players alive, 9 rounds played.
    assert hal.deaths == 3 and baku.deaths == 2
    assert hal.alive and baku.alive
    assert len(game.history) == 18


def test_leap_turn_action_space():
    """R9H2 is the leap turn: duration 61, Baku may drop 61, Hal capped at 60."""
    game = _make_game()
    _replay(game, CANONICAL_LINE[:17])  # everything before R9H2

    # The leap window [3540, 3600] is a CLOSED interval and R9H2 starts exactly on it.
    assert game.game_clock == pytest.approx(3540.0)
    assert game.is_leap_second_turn()
    assert game.get_turn_duration() == 61

    # Actor-aware legality: only Baku-as-dropper sees the 61st second.
    assert legal_max_second("Baku", "dropper", 61) == 61
    assert legal_max_second("Hal", "checker", 61) == 60
    assert legal_max_second("Hal", "dropper", 61) == 60
    assert legal_max_second("Baku", "checker", 61) == 60

    # The engine accepts Baku's drop at 61; Hal's check at 60 < 61 fails.
    record = game.resolve_half_round(61, 60, survived_outcome=True)
    assert record.turn_duration == 61
    assert record.result == HalfRoundResult.CHECK_FAIL_SURVIVED
    assert record.death_duration == pytest.approx(60.0)


def test_r9h2_survival_probability():
    """Pin the corrected R9H2 survival probability (~0.312, not HAL.md's stale 0.28).

    At injection: death_duration=60 (cylinder 0+60), hal.ttd=238 prior damage,
    referee has performed 4 CPRs, physicality 1.0:
        P = (1 - (60/300)**3) * 0.85**(238/60) * max(0.4, 0.88**4) * 1.0
    """
    game = _make_game()
    _replay(game, CANONICAL_LINE)

    record = game.history[-1]
    assert record.death_duration == pytest.approx(60.0)
    assert record.survived is True

    expected = 0.992 * 0.85 ** (238 / 60) * max(0.4, 0.88 ** 4) * 1.0
    assert record.survival_probability == pytest.approx(expected, abs=1e-3)
    assert record.survival_probability == pytest.approx(0.312, abs=1e-3)
