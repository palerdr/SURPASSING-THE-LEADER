"""Scenario records and scenario application for seeded training states.

Curricula, tests, and teacher evaluation all use this module to create
clean reproducible starting points such as `round7_pressure`.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.Game import Game
from src.Player import Player

from .awareness import LeapAwareness


@dataclass(frozen=True)
class PlayerScenario:
    cylinder: float = 0.0
    ttd: float = 0.0
    deaths: int = 0
    alive: bool = True


@dataclass(frozen=True)
class EpisodeScenario:
    game_clock: float = 0.0
    round_num: int = 0
    current_half: int = 1
    first_dropper_name: str | None = None
    hal: PlayerScenario = PlayerScenario()
    baku: PlayerScenario = PlayerScenario()
    referee_cprs: int = 0
    awareness: LeapAwareness | None = None
    name: str = "custom"


BUILT_IN_SCENARIO_EXPECTATIONS = {
    "round7_pressure": {
        "first_dropper_name": "hal",
        "current_half": 1,
        "is_leap_turn": False,
        "awareness": LeapAwareness.DEDUCED,
    },
    "round8_bridge": {
        "first_dropper_name": "hal",
        "current_half": 1,
        "is_leap_turn": False,
        "awareness": LeapAwareness.DEDUCED,
    },
    "round9_pre_leap": {
        "first_dropper_name": "hal",
        "current_half": 1,
        "is_leap_turn": False,
        "awareness": LeapAwareness.DEDUCED,
    },
    "round9_leap_deduced": {
        "first_dropper_name": "hal",
        "current_half": 2,
        "is_leap_turn": True,
        "awareness": LeapAwareness.DEDUCED,
    },
    "round9_leap_impaired": {
        "first_dropper_name": "hal",
        "current_half": 2,
        "is_leap_turn": True,
        "awareness": LeapAwareness.MEMORY_IMPAIRED,
    },
}


def player_scenario_from_dict(data: dict | None) -> PlayerScenario:
    if data is None:
        return PlayerScenario()

    return PlayerScenario(
        cylinder=float(data.get("cylinder", 0.0)),
        ttd=float(data.get("ttd", 0.0)),
        deaths=int(data.get("deaths", 0)),
        alive=bool(data.get("alive", True)),
    )


def scenario_from_options(options: dict | None) -> EpisodeScenario | None:
    if not options:
        return None

    raw = options.get("scenario", options)
    if raw is None:
        return None

    awareness_raw = raw.get("awareness")
    awareness = None if awareness_raw is None else LeapAwareness(awareness_raw)

    return EpisodeScenario(
        game_clock=float(raw.get("game_clock", 0.0)),
        round_num=int(raw.get("round_num", 0)),
        current_half=int(raw.get("current_half", 1)),
        first_dropper_name=raw.get("first_dropper") or raw.get("first_dropper_name"),
        hal=player_scenario_from_dict(raw.get("hal")),
        baku=player_scenario_from_dict(raw.get("baku")),
        referee_cprs=int(raw.get("referee_cprs", 0)),
        awareness=awareness,
        name=str(raw.get("name", "custom")),
    )


def apply_player_scenario(player: Player, scenario: PlayerScenario) -> None:
    player.cylinder = scenario.cylinder
    player.ttd = scenario.ttd
    player.deaths = scenario.deaths
    player.alive = scenario.alive


def apply_scenario(game: Game, hal: Player, baku: Player, scenario: EpisodeScenario) -> None:
    game.game_clock = scenario.game_clock
    game.round_num = scenario.round_num
    game.current_half = scenario.current_half
    game.referee.cprs_performed = scenario.referee_cprs

    if scenario.first_dropper_name is not None:
        name = scenario.first_dropper_name.lower()
        if name == "hal":
            game.first_dropper = hal
        elif name == "baku":
            game.first_dropper = baku
        else:
            raise ValueError(f"Unknown first_dropper_name: {scenario.first_dropper_name}")

    apply_player_scenario(hal, scenario.hal)
    apply_player_scenario(baku, scenario.baku)


def validate_scenario_reachability(game: Game, scenario: EpisodeScenario) -> None:
    if scenario.current_half not in (1, 2):
        raise ValueError(f"current_half must be 1 or 2, got {scenario.current_half}")
    if scenario.round_num < 0:
        raise ValueError(f"round_num must be >= 0, got {scenario.round_num}")
    if scenario.game_clock < 0:
        raise ValueError(f"game_clock must be >= 0, got {scenario.game_clock}")
    if scenario.referee_cprs < 0:
        raise ValueError(f"referee_cprs must be >= 0, got {scenario.referee_cprs}")

    for label, player in (("hal", scenario.hal), ("baku", scenario.baku)):
        if player.cylinder < 0:
            raise ValueError(f"{label}.cylinder must be >= 0, got {player.cylinder}")
        if player.ttd < 0:
            raise ValueError(f"{label}.ttd must be >= 0, got {player.ttd}")
        if player.deaths < 0:
            raise ValueError(f"{label}.deaths must be >= 0, got {player.deaths}")


def validate_named_scenario_semantics(game: Game, scenario: EpisodeScenario) -> None:
    expected = BUILT_IN_SCENARIO_EXPECTATIONS.get(scenario.name)
    if expected is None:
        return

    if scenario.first_dropper_name != expected["first_dropper_name"]:
        raise ValueError(
            f"{scenario.name} requires first_dropper_name="
            f"{expected['first_dropper_name']}, got {scenario.first_dropper_name}"
        )

    if scenario.current_half != expected["current_half"]:
        raise ValueError(
            f"{scenario.name} requires current_half={expected['current_half']}, "
            f"got {scenario.current_half}"
        )

    if game.is_leap_second_turn() != expected["is_leap_turn"]:
        state = "leap turn" if expected["is_leap_turn"] else "non-leap turn"
        raise ValueError(f"{scenario.name} must begin on a {state}")

    if scenario.awareness != expected["awareness"]:
        raise ValueError(
            f"{scenario.name} requires awareness={expected['awareness'].value}, "
            f"got {None if scenario.awareness is None else scenario.awareness.value}"
        )
