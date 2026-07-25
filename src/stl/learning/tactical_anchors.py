"""Scalable exact tablebase anchors with independent engine-derived proofs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable

from stl.engine.game import (
    FAILED_CHECK_PENALTY,
    PHYSICALITY_BAKU,
    PHYSICALITY_HAL,
    Game,
    Player,
    Referee,
)
from stl.learning.replay import exact_state_hash
from stl.solver.exact import exact_public_state
from stl.solver.tablebase import REGISTRY


INTERIOR_PIN_TAG = "interior_value"


@dataclass(frozen=True)
class TacticalAnchor:
    name: str
    game: Game
    value_for_hal: float
    stratum: str
    derivation: str
    family: str = "legacy"
    history_profile: str = "legacy"
    parameters: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class TacticalAnchorQuota:
    boundary: int
    interior: int


@dataclass(frozen=True)
class TacticalAnchorSplit:
    train: tuple[TacticalAnchor, ...]
    development: tuple[TacticalAnchor, ...]
    ruler: tuple[TacticalAnchor, ...]


BOUNDARY_CLOCKS = (0, 721, 2160, 3480, 3540, 3599)
BOUNDARY_OTHER_CYLINDERS = (0, 59, 60, 119, 120, 179, 180, 239, 240, 299, 300)
BOUNDARY_HISTORIES = (
    ("fresh", 0, 0, 0, 0),
    ("hal_shallow", 1, 60, 0, 0),
    ("hal_deep", 1, 300, 0, 0),
    ("baku_shallow", 0, 0, 1, 60),
    ("baku_deep", 0, 0, 1, 300),
    ("both_shallow", 1, 60, 1, 60),
    ("hal_min_baku_max_fatigue", 5, 300, 5, 1500),
    ("hal_max_baku_min_fatigue", 5, 1500, 5, 300),
)
INTERIOR_HAL_CYLINDERS = (*range(0, 240, 10), 239)
INTERIOR_HISTORIES = (
    ("fresh", 0, 0, 0, 0),
    ("hal_1_min", 1, 60, 0, 0),
    ("hal_1_mid", 1, 180, 0, 0),
    ("hal_1_max", 1, 300, 0, 0),
    ("baku_1_min", 0, 0, 1, 60),
    ("baku_1_mid", 0, 0, 1, 180),
    ("baku_1_max", 0, 0, 1, 300),
    ("split_hal_min", 1, 60, 1, 300),
    ("split_hal_max", 1, 300, 1, 60),
    ("hal_5_min", 5, 300, 0, 0),
    ("baku_5_max", 0, 0, 5, 1500),
    ("split_10_hal_min", 5, 300, 5, 1500),
    ("split_10_hal_max", 5, 1500, 5, 300),
)


def _base_game(*, clock: float, current_half: int) -> Game:
    hal = Player(name="Hal", physicality=PHYSICALITY_HAL)
    baku = Player(name="Baku", physicality=PHYSICALITY_BAKU)
    game = Game(player1=hal, player2=baku, referee=Referee(), first_dropper=hal)
    game.seed(0)
    game.game_clock = clock
    game.current_half = current_half
    return game


def _apply_history(
    game: Game,
    history: tuple[str, int, int, int, int],
) -> str:
    name, hal_deaths, hal_ttd, baku_deaths, baku_ttd = history
    game.player1.deaths = hal_deaths
    game.player1.ttd = float(hal_ttd)
    game.player2.deaths = baku_deaths
    game.player2.ttd = float(baku_ttd)
    game.referee.cprs_performed = hal_deaths + baku_deaths
    return name


def _parameters(**values: object) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((key, str(value)) for key, value in values.items()))


def _v4_boundary_anchor(
    *,
    current_half: int,
    other_cylinder: int,
    clock: int,
    history: tuple[str, int, int, int, int],
) -> TacticalAnchor:
    game = _base_game(clock=float(clock), current_half=current_half)
    history_name = _apply_history(game, history)
    checker = game.player2 if current_half == 1 else game.player1
    other = game.player1 if current_half == 1 else game.player2
    checker.cylinder = 300.0
    other.cylinder = float(other_cylinder)
    checker_name = checker.name.lower()
    family = "boundary_double_overflow" if other_cylinder == 300 else "boundary_single_overflow"
    name = (
        f"v4_boundary_h{current_half}_other{other_cylinder}_clock{clock}_"
        f"{history_name}"
    )
    return TacticalAnchor(
        name=name,
        game=game,
        value_for_hal=1.0 if checker_name == "baku" else -1.0,
        stratum="boundary",
        derivation=(
            "current checker cylinder=300; every legal joint action injects a "
            "fatal 300-second dose into the checker"
        ),
        family=family,
        history_profile=history_name,
        parameters=_parameters(
            current_half=current_half,
            checker=checker_name,
            other_cylinder=other_cylinder,
            clock=clock,
        ),
    )


def _v4_interior_anchor(
    *,
    hal_cylinder: int,
    history: tuple[str, int, int, int, int],
) -> TacticalAnchor:
    game = _base_game(clock=3540.0, current_half=2)
    history_name = _apply_history(game, history)
    game.player1.cylinder = float(hal_cylinder)
    game.player2.cylinder = 300.0
    death_duration = float(hal_cylinder + FAILED_CHECK_PENALTY)
    probability = game.referee.compute_survival_probability(
        game.player1,
        death_duration=death_duration,
    )
    return TacticalAnchor(
        name=f"v4_interior_hal{hal_cylinder}_{history_name}",
        game=game,
        value_for_hal=2.0 * probability - 1.0,
        stratum="interior",
        derivation=(
            "Baku drop=61 forces Hal failure; fatal branch=-1, revived branch=+1; "
            "value=2*engine_survival_probability-1"
        ),
        family="interior_forced_hal_fail",
        history_profile=history_name,
        parameters=_parameters(
            hal_cylinder=hal_cylinder,
            death_duration=death_duration,
            clock=3540,
        ),
    )


def _boundary_anchor(index: int) -> TacticalAnchor:
    # The current checker starts at the five-minute cylinder ceiling. Every
    # successful or failed check therefore injects/deaths for 300 seconds,
    # whose engine survival probability is zero. The sign is fixed solely by
    # which player is checking.
    current_half = 1 + index % 2
    game = _base_game(
        clock=float(721 + 2 * index + 120 * (index % 29)),
        current_half=current_half,
    )
    checker = game.player2 if current_half == 1 else game.player1
    other = game.player1 if current_half == 1 else game.player2
    checker.cylinder = 300.0
    other.cylinder = float((index * 53) % 240)
    hal_wins = checker.name.lower() == "baku"
    return TacticalAnchor(
        name=f"generated_forced_boundary_{index:06d}",
        game=game,
        value_for_hal=1.0 if hal_wins else -1.0,
        stratum="boundary",
        derivation=(
            "current checker cylinder=300; every legal cell has death_duration=300; "
            "engine survival probability is zero"
        ),
    )


def _interior_anchor(index: int) -> TacticalAnchor:
    # At the leap window Baku can drop on true second 61 while Hal can check
    # only through 60. Hal therefore fails in every matrix cell. If Hal dies,
    # utility is -1. If revived, Baku checks next with cylinder=300 and dies
    # certainly, utility +1. Thus v = p(+1) + (1-p)(-1) = 2p-1.
    hal_cylinder = float(1 + index % 238)
    cprs = int((index // 17) % 11)
    game = _base_game(clock=3540.0, current_half=2)
    game.player1.cylinder = hal_cylinder
    game.player2.cylinder = 300.0
    game.player2.deaths = cprs
    game.player2.ttd = float(60 * cprs)
    game.referee.cprs_performed = cprs
    probability = game.referee.compute_survival_probability(
        game.player1,
        death_duration=hal_cylinder + FAILED_CHECK_PENALTY,
    )
    return TacticalAnchor(
        name=f"generated_forced_interior_{index:06d}",
        game=game,
        value_for_hal=2.0 * probability - 1.0,
        stratum="interior",
        derivation=(
            "Baku drop=61 forces Hal failure; revive branch is +1 and fatal branch "
            "is -1; value=2*engine_survival_probability-1"
        ),
    )


def _canonical_anchors() -> list[TacticalAnchor]:
    anchors: list[TacticalAnchor] = []
    for name, factory in sorted(REGISTRY.items()):
        scenario = factory()
        if scenario.expected_value is None:
            continue
        stratum = "interior" if INTERIOR_PIN_TAG in scenario.tags else "boundary"
        anchors.append(
            TacticalAnchor(
                name=name,
                game=scenario.game,
                value_for_hal=float(scenario.expected_value),
                stratum=stratum,
                derivation=f"canonical stl.solver.tablebase fixture {name}",
                family=(
                    "canonical_interior"
                    if stratum == "interior"
                    else (
                        "boundary_double_overflow"
                        if name.startswith("both_overflow_")
                        else "canonical_boundary"
                    )
                ),
                history_profile="canonical",
                parameters=_parameters(canonical_name=name),
            )
        )
    return anchors


def _rank(seed: int, anchor: TacticalAnchor) -> str:
    state_hash = exact_state_hash(exact_public_state(anchor.game))
    return hashlib.sha256(
        f"{seed}:tactical-anchor:{anchor.stratum}:{state_hash}".encode("ascii")
    ).hexdigest()


def build_tactical_anchor_split(
    *,
    train_quota: TacticalAnchorQuota,
    development_quota: TacticalAnchorQuota,
    ruler_quota: TacticalAnchorQuota,
    split_seed: int,
) -> TacticalAnchorSplit:
    """Allocate unique exact anchors; canonical registry pins stay on ruler."""

    quotas = {
        "train": train_quota,
        "development": development_quota,
        "ruler": ruler_quota,
    }
    for quota in quotas.values():
        if quota.boundary < 0 or quota.interior < 0:
            raise ValueError("tactical anchor quotas must be non-negative")

    canonical = _canonical_anchors()
    outputs: dict[str, list[TacticalAnchor]] = {
        "train": [],
        "development": [],
        "ruler": list(canonical),
    }
    for stratum in ("boundary", "interior"):
        canonical_count = sum(item.stratum == stratum for item in canonical)
        if canonical_count > getattr(ruler_quota, stratum):
            raise ValueError(
                f"ruler {stratum} quota cannot hold {canonical_count} canonical pins"
            )
        needed = {
            role: getattr(quota, stratum) - (canonical_count if role == "ruler" else 0)
            for role, quota in quotas.items()
        }
        blocked = {
            exact_state_hash(exact_public_state(item.game)) for item in canonical
        }
        candidates: list[TacticalAnchor] = []
        index = 0
        total_needed = sum(needed.values())
        while len(candidates) < total_needed:
            anchor = (
                _boundary_anchor(index)
                if stratum == "boundary"
                else _interior_anchor(index)
            )
            index += 1
            state_hash = exact_state_hash(exact_public_state(anchor.game))
            if state_hash in blocked:
                continue
            blocked.add(state_hash)
            candidates.append(anchor)
        candidates.sort(key=lambda item: _rank(split_seed, item))
        offset = 0
        for role in ("train", "development", "ruler"):
            count = needed[role]
            outputs[role].extend(candidates[offset : offset + count])
            offset += count

    return TacticalAnchorSplit(
        train=tuple(outputs["train"]),
        development=tuple(outputs["development"]),
        ruler=tuple(outputs["ruler"]),
    )


def build_v4_tactical_anchor_split(
    *, blocked_ruler_state_hashes: set[str] | None = None
) -> TacticalAnchorSplit:
    """Build the fixed, causally varied V4 tablebase benchmark.

    Generated states use a deterministic combinatorial split. Canonical pins
    have already been inspected through the consumed V3 ruler, so they are
    development regressions rather than hidden holdout evidence.
    """

    train: list[TacticalAnchor] = []
    development: list[TacticalAnchor] = list(_canonical_anchors())
    ruler: list[TacticalAnchor] = []
    canonical_hashes = {
        exact_state_hash(exact_public_state(anchor.game)) for anchor in development
    }
    blocked_ruler_state_hashes = (
        set()
        if blocked_ruler_state_hashes is None
        else set(blocked_ruler_state_hashes)
    )
    generated_hashes: set[str] = set()

    def add(role: str, anchor: TacticalAnchor) -> bool:
        state_hash = exact_state_hash(exact_public_state(anchor.game))
        if role == "ruler" and state_hash in blocked_ruler_state_hashes:
            return False
        if state_hash in canonical_hashes:
            return False
        if state_hash in generated_hashes:
            return False
        generated_hashes.add(state_hash)
        {"train": train, "development": development, "ruler": ruler}[role].append(
            anchor
        )
        return True

    for half_index, current_half in enumerate((1, 2)):
        for cylinder_index, other_cylinder in enumerate(BOUNDARY_OTHER_CYLINDERS):
            for history_index, history in enumerate(BOUNDARY_HISTORIES):
                for clock_index, clock in enumerate(BOUNDARY_CLOCKS):
                    residue = (
                        clock_index + half_index + cylinder_index + history_index
                    ) % 6
                    role = (
                        "ruler"
                        if residue == 0
                        else "development" if residue == 1 else "train"
                    )
                    anchor = _v4_boundary_anchor(
                            current_half=current_half,
                            other_cylinder=other_cylinder,
                            clock=clock,
                            history=history,
                        )
                    replacement_offset = 0
                    while not add(role, anchor):
                        replacement_offset += 1
                        replacement_clock = (clock + replacement_offset) % 3600
                        anchor = _v4_boundary_anchor(
                            current_half=current_half,
                            other_cylinder=other_cylinder,
                            clock=replacement_clock,
                            history=history,
                        )

    for history_index, history in enumerate(INTERIOR_HISTORIES):
        for cylinder_index, hal_cylinder in enumerate(INTERIOR_HAL_CYLINDERS):
            residue = (cylinder_index + history_index) % 5
            role = (
                "ruler" if residue == 0 else "development" if residue == 1 else "train"
            )
            anchor = _v4_interior_anchor(
                    hal_cylinder=hal_cylinder,
                    history=history,
                )
            replacement_offset = 0
            while not add(role, anchor):
                replacement_offset += 1
                replacement_cylinder = (hal_cylinder + replacement_offset) % 240
                anchor = _v4_interior_anchor(
                    hal_cylinder=replacement_cylinder,
                    history=history,
                )

    expected = {
        "train": (704, 195),
        "development_generated": (176, 65),
        "ruler": (176, 65),
    }
    observed = {
        "train": (
            sum(anchor.stratum == "boundary" for anchor in train),
            sum(anchor.stratum == "interior" for anchor in train),
        ),
        "development_generated": (
            sum(
                anchor.stratum == "boundary" and anchor.history_profile != "canonical"
                for anchor in development
            ),
            sum(
                anchor.stratum == "interior" and anchor.history_profile != "canonical"
                for anchor in development
            ),
        ),
        "ruler": (
            sum(anchor.stratum == "boundary" for anchor in ruler),
            sum(anchor.stratum == "interior" for anchor in ruler),
        ),
    }
    if observed != expected:
        raise AssertionError(f"unexpected V4 tactical counts: {observed}")
    return TacticalAnchorSplit(tuple(train), tuple(development), tuple(ruler))


def tactical_taxonomy(anchors: Iterable[TacticalAnchor]) -> dict[str, dict[str, object]]:
    """Return a stable state-hash keyed evaluation taxonomy."""

    return {
        exact_state_hash(exact_public_state(anchor.game)): {
            "name": anchor.name,
            "stratum": anchor.stratum,
            "family": anchor.family,
            "history_profile": anchor.history_profile,
            "parameters": dict(anchor.parameters),
            "derivation": anchor.derivation,
        }
        for anchor in anchors
    }


__all__ = [
    "TacticalAnchor",
    "TacticalAnchorQuota",
    "TacticalAnchorSplit",
    "BOUNDARY_CLOCKS",
    "BOUNDARY_HISTORIES",
    "BOUNDARY_OTHER_CYLINDERS",
    "INTERIOR_HAL_CYLINDERS",
    "INTERIOR_HISTORIES",
    "build_tactical_anchor_split",
    "build_v4_tactical_anchor_split",
    "tactical_taxonomy",
]
