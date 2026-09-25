"""Tests for ``python -m hal_lab match`` and the paired-seat series."""

from __future__ import annotations

import pytest

from arena.contracts import CanonicalDecision
from hal_lab import cli
from hal_lab.cli import build_parser, command_match


def test_aggro_hal_match_requires_an_explicit_checkpoint_and_defaults_to_cpu() -> None:
    args = cli.build_parser().parse_args(
        [
            "match",
            "--candidate",
            "aggro-hal",
            "--opponent",
            "dth",
            "--pure-dth",
            "--output",
            "unused.json",
        ]
    )
    assert args.aggro_hal_device == "cpu"
    with pytest.raises(ValueError, match="aggro-hal-checkpoint"):
        cli._make_provider("aggro-hal", args)


def test_aggro_hal_match_requires_the_pure_dth_surface() -> None:
    args = cli.build_parser().parse_args(
        [
            "match",
            "--candidate",
            "aggro-hal",
            "--opponent",
            "dth",
            "--output",
            "unused.json",
        ]
    )
    with pytest.raises(ValueError, match="--pure-dth"):
        cli.command_match(args)


def test_match_series_pairs_seats_and_reports_sprt(tmp_path) -> None:
    from hal_lab.harness.series import run_paired_series

    class _Fixed:
        def __init__(self, second: int) -> None:
            self.second = second

        def policy(self, decision: CanonicalDecision) -> dict[int, float]:
            return {self.second: 1.0}

    report = run_paired_series(
        "late",
        "early",
        make_candidate=lambda: _Fixed(60),
        make_opponent=lambda: _Fixed(1),
        base_seeds=4,
        seed_start=41,
        start_clock=720,
        max_half_rounds=120,
        stop_early=False,
    )

    assert report["schema_version"] == "arena-match-report-v1"
    assert report["pure_dth"] is False
    assert report["seed_start"] == 41
    assert len(report["games"]) == 8
    assert [game["seed"] for game in report["games"]] == [
        41,
        41,
        42,
        42,
        43,
        43,
        44,
        44,
    ]
    first_seats = [game["first_seat_agent"] for game in report["games"]]
    assert first_seats.count("late") == 4 and first_seats.count("early") == 4
    sprt = report["sprt"]
    assert sprt["decisive_games"] == sprt["wins"] + sprt["losses"]
    assert sprt["decision"] in {"accept-h1", "accept-h0", "continue"}


def test_match_cli_exposes_perfect_hal_only_with_explicit_pure_dth() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "match",
            "--candidate",
            "perfect-hal",
            "--opponent",
            "dth",
            "--output",
            "unused.json",
        ]
    )
    with pytest.raises(ValueError, match="pure-DTH"):
        command_match(args)

    pure = parser.parse_args(
        [
            "match",
            "--candidate",
            "perfect-hal",
            "--opponent",
            "dth",
            "--pure-dth",
            "--output",
            "unused.json",
        ]
    )
    assert pure.candidate == "perfect-hal"
    assert pure.pure_dth is True
    assert pure.perfect_hal_response_temperature == 0.0


def test_match_cli_exposes_pm_hal_only_with_explicit_pure_dth() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "match",
            "--candidate",
            "pm-hal",
            "--opponent",
            "dth",
            "--output",
            "unused.json",
        ]
    )
    with pytest.raises(ValueError, match="pure-DTH"):
        command_match(args)

    pure = parser.parse_args(
        [
            "match",
            "--candidate",
            "pm-hal",
            "--opponent",
            "dth",
            "--pure-dth",
            "--output",
            "unused.json",
        ]
    )
    assert pure.candidate == "pm-hal"
    assert pure.pure_dth is True
    assert pure.pm_hal_aggro_checkpoint is None
    assert pure.pm_hal_config.endswith("pm_hal_controller_v3.json")
    assert pure.pm_hal_game_epsilon_budget is None
