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
        manifest = {"metadata": {"ruleset_id": "bucket12_unified80"}}

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
        Path("abstract/outputs/bucket12_unified80"),
        "bucket12_unified80",
    )
    assert cli._abstract_artifact(ten) == (
        Path("abstract/outputs/bucket6_unified80"),
        "bucket6_unified80",
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
