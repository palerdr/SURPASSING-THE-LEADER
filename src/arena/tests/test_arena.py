from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from abstract.rules import Bucket12Unified80Rules
from arena import abstract_adapter, cli
from arena.abstract_adapter import AbstractTablebasePolicyProvider, project_to_abstract_state
from arena.agent import PolicyDrivenAgent, decision_from_game, normalize_legal_policy
from arena.contracts import CanonicalDecision
from stl.engine.game import Game, Player, Referee


def _decision(*, legal: tuple[int, ...] = (1, 2, 3)) -> CanonicalDecision:
    return CanonicalDecision(
        role="dropper",
        actor_name="Hal",
        turn_duration=60,
        legal_seconds=legal,
        checker_cylinder_seconds=19,
        checker_ttd_seconds=120,
        dropper_cylinder_seconds=29,
        dropper_ttd_seconds=70,
        native_state=object(),
    )


def test_projection_floors_seconds_to_role_relative_ten_second_buckets() -> None:
    state = project_to_abstract_state(_decision())
    assert (state.checker_load, state.checker_ttd, state.dropper_load, state.dropper_ttd) == (1, 12, 2, 7)


def test_projection_floors_seconds_to_role_relative_five_second_buckets() -> None:
    state = project_to_abstract_state(_decision(), Bucket12Unified80Rules())
    assert (state.checker_load, state.checker_ttd, state.dropper_load, state.dropper_ttd) == (
        3,
        24,
        5,
        14,
    )


def test_five_second_provider_uses_packed_lookup_and_lifts_actions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    class _Packed:
        manifest = {"metadata": {"ruleset_id": "bucket12_frozen95"}}

        def __init__(self, artifact_dir) -> None:
            assert artifact_dir == tmp_path

        def lookup(self, state) -> dict[str, np.ndarray]:
            assert (state.checker_load, state.checker_ttd) == (3, 24)
            drop = np.zeros(12, dtype=np.float32)
            check = np.zeros(12, dtype=np.float32)
            drop[[0, 11]] = (0.25, 0.75)
            check[5] = 1.0
            return {"drop_policy": drop, "check_policy": check}

    monkeypatch.setattr(abstract_adapter, "PackedTablebase", _Packed)
    provider = AbstractTablebasePolicyProvider(tmp_path, bucket_seconds=5)
    assert provider.policy(_decision()) == pytest.approx({5: 0.25, 60: 0.75})


def test_bucket_flag_selects_bucket_specific_default_artifact() -> None:
    parser = cli.build_parser()
    five = parser.parse_args(["play", "--hal-agent", "abstract", "--buckets", "5"])
    ten = parser.parse_args(["play", "--hal-agent", "abstract", "--buckets", "10"])
    assert cli._abstract_artifact(five) == (
        Path("src/abstract/outputs/bucket12_frozen95"),
        "bucket12_frozen95",
    )
    assert cli._abstract_artifact(ten) == (
        Path("src/abstract/outputs/bucket6_frozen95"),
        "bucket6_frozen95",
    )


def test_policy_normalization_discards_illegal_zero_mass_entries() -> None:
    actions, probabilities = normalize_legal_policy({0: 4.0, 1: 1.0, 2: 3.0, 61: 2.0}, (1, 2))
    assert actions.tolist() == [1, 2]
    assert probabilities.tolist() == pytest.approx([0.25, 0.75])


@dataclass
class _Provider:
    def policy(self, decision: CanonicalDecision) -> dict[int, float]:
        return {0: 100.0, 1: 1.0, 60: 5.0}


def test_algorithm_agnostic_agent_samples_only_a_legal_engine_action() -> None:
    game = Game(Player("Hal"), Player("Baku"), Referee())
    agent = PolicyDrivenAgent(_Provider(), player_name="Hal", seed=4)
    action = agent.choose_action(game, "dropper", game.get_turn_duration())
    assert 1 <= action <= 60
    assert action != 0


def test_decision_uses_engine_role_relative_state() -> None:
    game = Game(Player("Hal", cylinder=20, ttd=30), Player("Baku", cylinder=40, ttd=50), Referee())
    game.first_dropper = game.player1
    decision = decision_from_game(game, role="dropper", turn_duration=60)
    assert decision.actor_name == "Hal"
    assert decision.dropper_cylinder_seconds == 20
    assert decision.checker_cylinder_seconds == 40


def test_cli_exits_cleanly_on_control_c(monkeypatch: pytest.MonkeyPatch) -> None:
    def interrupt(_args: object) -> int:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "command_play", interrupt)
    assert cli.main(["play"]) == 130


def test_dth_projection_is_the_identity_with_defensive_clamps() -> None:
    from arena.dth_adapter import project_to_dth_state

    assert project_to_dth_state(_decision()) == (19, 120, 29, 70)
    wide = CanonicalDecision(
        role="checker",
        actor_name="Hal",
        turn_duration=60,
        legal_seconds=tuple(range(1, 61)),
        checker_cylinder_seconds=305.0,
        checker_ttd_seconds=301.0,
        dropper_cylinder_seconds=-2.0,
        dropper_ttd_seconds=299.6,
        native_state=object(),
    )
    assert project_to_dth_state(wide) == (299, 300, 0, 299)


def test_dth_provider_serves_certified_policies_without_artifacts() -> None:
    from arena.dth_adapter import DTHResolvePolicyProvider
    from dth.agent import ResolveBudget

    provider = DTHResolvePolicyProvider(
        budget=ResolveBudget(deadline_seconds=10.0, finite_fallback_horizon=1)
    )
    try:
        decision = _decision()
        policy = provider.policy(decision)
        assert policy
        assert all(1 <= second <= 60 for second in policy)
        assert all(weight > 0.0 for weight in policy.values())
        assert provider.decisions[0].provenance == "finite-horizon-exact"
        summary = provider.match_summary()
        assert "certified 1/1" in summary
        assert "latency p95" in summary
    finally:
        provider.close()


def test_dth_cli_dispatch_builds_a_policy_driven_agent(tmp_path) -> None:
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "play",
            "--hal-agent",
            "dth",
            "--dth-tablebase",
            str(tmp_path / "missing-but-default.sqlite"),
        ]
    )
    with pytest.raises(FileNotFoundError):
        cli._make_hal(args)

    defaults = parser.parse_args(["play", "--hal-agent", "dth"])
    defaults.dth_checkpoint = cli.DEFAULT_DTH_CHECKPOINT
    defaults.dth_tablebase = cli.DEFAULT_DTH_TABLEBASE
    if not Path(cli.DEFAULT_DTH_TABLEBASE).is_file():
        agent = cli._make_hal(defaults)
        assert isinstance(agent, PolicyDrivenAgent)
    else:
        agent = cli._make_hal(defaults)
        assert isinstance(agent, PolicyDrivenAgent)
        agent.provider.close()


def test_match_series_pairs_seats_and_reports_sprt(tmp_path) -> None:
    from arena.match import run_paired_series

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
        start_clock=720,
        max_half_rounds=120,
        stop_early=False,
    )

    assert report["schema_version"] == "arena-match-report-v1"
    assert len(report["games"]) == 8
    first_seats = [game["first_seat_agent"] for game in report["games"]]
    assert first_seats.count("late") == 4 and first_seats.count("early") == 4
    sprt = report["sprt"]
    assert sprt["decisive_games"] == sprt["wins"] + sprt["losses"]
    assert sprt["decision"] in {"accept-h1", "accept-h0", "continue"}


def test_sprt_thresholds_are_predeclared_and_reachable() -> None:
    from arena.match import sprt_verdict

    assert sprt_verdict(0, 0)["decision"] == "continue"
    assert sprt_verdict(30, 2)["decision"] == "accept-h1"
    assert sprt_verdict(2, 30)["decision"] == "accept-h0"


def test_abstract_adapter_falls_back_to_uniform_outside_the_closure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    class _Packed:
        manifest = {"metadata": {"ruleset_id": "bucket12_frozen95"}}

        def __init__(self, artifact_dir) -> None:
            del artifact_dir

        def lookup(self, state):
            raise LookupError("outside the root's reachable closure")

    monkeypatch.setattr(abstract_adapter, "PackedTablebase", _Packed)
    provider = AbstractTablebasePolicyProvider(tmp_path, bucket_seconds=5)
    policy = provider.policy(_decision())
    assert set(policy) == {5 * (index + 1) for index in range(12)}
    assert all(weight == 1.0 for weight in policy.values())
