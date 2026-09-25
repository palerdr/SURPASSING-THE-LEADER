from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from abstract.rules import Bucket12Frozen95Rules
from arena import abstract_adapter
from arena.abstract_adapter import (
    AbstractTablebasePolicyProvider,
    project_to_abstract_state,
)
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
    assert (
        state.checker_load,
        state.checker_ttd,
        state.dropper_load,
        state.dropper_ttd,
    ) == (1, 12, 2, 7)


def test_projection_floors_seconds_to_role_relative_five_second_buckets() -> None:
    state = project_to_abstract_state(_decision(), Bucket12Frozen95Rules())
    assert (
        state.checker_load,
        state.checker_ttd,
        state.dropper_load,
        state.dropper_ttd,
    ) == (
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


def test_match_lifecycle_delivers_each_public_reveal_exactly_once() -> None:
    from arena.match import play_match_game

    class _LifecycleProvider:
        def __init__(self) -> None:
            self.resets = 0
            self.records = []
            self.outcomes = []

        def reset_game(self) -> None:
            self.resets += 1

        def policy(self, decision):
            del decision
            return {2: 1.0}

        def observe(self, record) -> None:
            self.records.append(record)

        def end_game(self, outcome) -> None:
            self.outcomes.append(outcome)

    first = _LifecycleProvider()
    second = _LifecycleProvider()
    play_match_game(
        first,
        second,
        seed=11,
        start_clock=720,
        max_half_rounds=2,
        game_index=3,
    )

    assert first.resets == second.resets == 1
    assert len(first.records) == len(second.records) == 2
    assert [record.half_round_index for record in first.records] == [0, 1]
    assert first.records == second.records
    assert len(first.outcomes) == len(second.outcomes) == 1
    assert first.outcomes[0].game_index == 3


def test_policy_normalization_discards_illegal_zero_mass_entries() -> None:
    actions, probabilities = normalize_legal_policy(
        {0: 4.0, 1: 1.0, 2: 3.0, 61: 2.0}, (1, 2)
    )
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
    game = Game(
        Player("Hal", cylinder=20, ttd=30),
        Player("Baku", cylinder=40, ttd=50),
        Referee(),
    )
    game.first_dropper = game.player1
    decision = decision_from_game(game, role="dropper", turn_duration=60)
    assert decision.actor_name == "Hal"
    assert decision.dropper_cylinder_seconds == 20
    assert decision.checker_cylinder_seconds == 40


def test_stl_and_dth_revival_surfaces_match_over_the_full_domain() -> None:
    from dth.solver import revival_model

    referee = Referee(cprs_performed=17)
    player = Player(name="Either", physicality=0.01)
    for st_in_vial in range(300):
        for ttd in range(301):
            player.ttd = ttd
            actual = referee.compute_survival_probability(
                player, death_duration=st_in_vial + 60
            )
            assert actual == revival_model(st_in_vial, ttd)


def test_dth_projection_is_the_exact_literal_second_identity() -> None:
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
    with pytest.raises(ValueError, match="literal-second"):
        project_to_dth_state(wide)


def test_dth_provider_serves_only_complete_exact_policies(
    monkeypatch, tmp_path
) -> None:
    import arena.dth_adapter as adapter
    from dth.agent import MoveDecision

    class _Agent:
        def __init__(self, artifact_dir) -> None:
            assert artifact_dir == tmp_path

        def decide(self, state):
            return MoveDecision(
                state=state,
                value=0.1,
                drop_policy=(1.0,) + (0.0,) * 59,
                check_policy=(0.0,) * 59 + (1.0,),
                saddle_gap=1e-7,
                elapsed_seconds=0.001,
            )

    monkeypatch.setattr(adapter, "CompleteDTHAgent", _Agent)
    provider = adapter.DTHCompletePolicyProvider(tmp_path)
    assert provider.policy(_decision()) == {1: 1.0}
    assert "1 exact moves" in provider.match_summary()


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
