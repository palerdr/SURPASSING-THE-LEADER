from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from abstract.rules import Bucket12Frozen95Rules
from arena import abstract_adapter, cli
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


def test_dth_complete_tablebase_is_default_without_legacy_alias() -> None:
    parser = cli.build_parser()
    defaults = parser.parse_args(["play"])
    canonical = parser.parse_args(
        ["play", "--hal-agent", "dth", "--dth-complete-tablebase", "complete-tablebase"]
    )

    assert defaults.hal_agent == "dth"
    assert defaults.dth_complete_tablebase == cli.DEFAULT_DTH_COMPLETE_TABLEBASE
    assert canonical.dth_complete_tablebase == "complete-tablebase"
    with pytest.raises(SystemExit):
        parser.parse_args(["play", "--dth-backup", "legacy-tablebase"])


def test_play_rules_gate_uses_public_identity_and_waits_for_enter(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class _TTY:
        @staticmethod
        def isatty() -> bool:
            return True

    prompts: list[str] = []
    monkeypatch.setattr(cli.sys, "stdin", _TTY())
    monkeypatch.setattr(
        "builtins.input",
        lambda prompt="": prompts.append(prompt) or "",
    )
    args = cli.build_parser().parse_args(
        ["play", "--public-hal-label", "concealed opponent"]
    )

    cli._show_rules(args)

    output = capsys.readouterr().out
    assert "GAME RULES" in output
    assert "Opponent: Hal (concealed opponent)" in output
    assert "ST means Squandered Time" in output
    assert "TTD means Total Time Dead" in output
    assert "choose 61" not in output
    assert prompts == ["\nPress Enter to begin: "]


def test_play_rules_do_not_consume_piped_actions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Pipe:
        @staticmethod
        def isatty() -> bool:
            return False

    monkeypatch.setattr(cli.sys, "stdin", _Pipe())
    monkeypatch.setattr(
        "builtins.input",
        lambda prompt="": pytest.fail(f"unexpected input prompt: {prompt}"),
    )
    args = cli.build_parser().parse_args(["play"])

    cli._show_rules(args)


def test_play_rules_can_be_skipped_for_automation(
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = cli.build_parser().parse_args(["play", "--skip-rules"])

    cli._show_rules(args)

    assert capsys.readouterr().out == ""


def test_adaptive_dth_cli_exposes_predeclared_safety_controls() -> None:
    args = cli.build_parser().parse_args(
        [
            "play",
            "--hal-agent",
            "adaptive-dth",
            "--adaptive-prior-strength",
            "2",
            "--adaptive-decay",
            "0.8",
            "--adaptive-epsilon-grid",
            "0",
            "0.01",
            "--adaptive-match-epsilon-budget",
            "0.03",
            "--adaptive-confidence",
            "0.975",
            "--adaptive-posterior-samples",
            "256",
        ]
    )

    assert args.hal_agent == "adaptive-dth"
    assert args.adaptive_prior_strength == 2.0
    assert args.adaptive_decay == 0.8
    assert args.adaptive_epsilon_grid == [0.0, 0.01]
    assert args.adaptive_match_epsilon_budget == 0.03
    assert args.adaptive_confidence == 0.975
    assert args.adaptive_posterior_samples == 256


def test_adaptive_dth_cli_loads_a_role_population_prior(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import arena.adaptive_dth as adaptive

    class _Agent:
        def __init__(self, artifact_dir) -> None:
            assert artifact_dir == tmp_path / "tablebase"

    tablebase = tmp_path / "tablebase"
    tablebase.mkdir()
    (tablebase / "tablebase.json").write_text("{}", encoding="utf-8")
    prior = tmp_path / "prior.json"
    mean = [1.0 / 60.0] * 60
    prior.write_text(
        json.dumps(
            {
                "schema_version": "adaptive-dth-role-prior-v1",
                "dropper": {"mean": mean, "strength": 3.0},
                "checker": {"mean": mean, "strength": 5.0},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(adaptive, "CompleteDTHAgent", _Agent)
    args = cli.build_parser().parse_args(
        [
            "play",
            "--hal-agent",
            "adaptive-dth",
            "--dth-complete-tablebase",
            str(tablebase),
            "--adaptive-prior-json",
            str(prior),
            "--adaptive-decay",
            "0.7",
        ]
    )

    provider = cli._make_adaptive_dth_provider(args)
    assert provider.opponent.drop_prior.strength == 3.0
    assert provider.opponent.check_prior.strength == 5.0
    assert provider.opponent.decay == 0.7

    prior.write_text(
        json.dumps(
            {
                "schema_version": "adaptive-dth-role-mixture-prior-v1",
                "weights": [0.4, 0.6],
                "components": [
                    {
                        "dropper": {"mean": mean, "strength": 1.0},
                        "checker": {"mean": mean, "strength": 1.0},
                    },
                    {
                        "dropper": {"mean": mean, "strength": 2.0},
                        "checker": {"mean": mean, "strength": 2.0},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    mixture_provider = cli._make_adaptive_dth_provider(args)
    assert isinstance(mixture_provider.opponent, adaptive.RoleMixtureOpponent)
    assert mixture_provider.opponent.posterior_weights == pytest.approx((0.4, 0.6))


def test_play_session_retains_one_hal_and_writes_public_transcript(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class _SummaryProvider:
        def match_summary(self) -> str:
            return "one repeated opponent"

    class _Hal:
        provider = _SummaryProvider()

        def choose_action(self, game, role, turn_duration):
            del game, role
            return min(60, turn_duration)

    hal = _Hal()
    monkeypatch.setattr(cli, "_make_hal", lambda args: hal)
    monkeypatch.setattr(
        cli,
        "_human_action",
        lambda *, actor, role, legal: legal[-1],
    )
    transcript = tmp_path / "session.json"
    args = cli.build_parser().parse_args(
        [
            "play",
            "--games",
            "2",
            "--seed",
            "41",
            "--public-hal-label",
            "concealed",
            "--conceal-hal-details",
            "--start-clock-sequence",
            "720",
            "3420",
            "--max-half-rounds",
            "2",
            "--transcript",
            str(transcript),
        ]
    )

    assert cli.command_play(args) == 0
    assert "one repeated opponent" not in capsys.readouterr().out
    payload = json.loads(transcript.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "arena-public-play-session-v1"
    assert payload["hal_summary"] == "one repeated opponent"
    assert payload["public_hal_label"] == "concealed"
    assert [game["seed"] for game in payload["games"]] == [41, 42]
    assert [game["start_clock"] for game in payload["games"]] == [720, 3420]
    assert all(len(game["public_history"]) == 2 for game in payload["games"])
    first = payload["games"][0]["public_history"][0]
    assert first["public_state_before"]["clock_display"] == "8:12:00 AM"
    assert first["dropper"] == "Hal"
    assert first["checker"] == "Baku"
    assert first["drop_second"] == 60
    assert first["check_second"] == 60


def test_play_session_requires_one_start_clock_per_game() -> None:
    args = cli.build_parser().parse_args(
        ["play", "--games", "2", "--start-clock-sequence", "720"]
    )
    with pytest.raises(ValueError, match="one value per game"):
        cli.command_play(args)


def test_exploit_hal_cli_requires_an_explicit_checkpoint() -> None:
    args = cli.build_parser().parse_args(
        ["play", "--hal-agent", "exploit-hal", "--skip-rules"]
    )
    with pytest.raises(ValueError, match="exploit-hal-checkpoint"):
        cli._make_provider("exploit-hal", args)


def test_exploit_hal_cli_defaults_to_the_supported_v2_protocol() -> None:
    args = cli.build_parser().parse_args(
        ["play", "--hal-agent", "exploit-hal", "--skip-rules"]
    )
    assert args.exploit_hal_config == "src/arena/config/exploit_hal_v2.yaml"


def test_aggro_hal_is_not_exposed_on_canonical_stl_play() -> None:
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["play", "--hal-agent", "aggro-hal", "--skip-rules"])


def test_retired_stl_mcts_is_not_advertised_and_fails_closed() -> None:
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["play", "--hal-agent", "stl-mcts", "--skip-rules"])
    args = parser.parse_args(["play", "--skip-rules"])
    with pytest.raises(ValueError, match="retired"):
        cli._make_provider("stl-mcts", args)


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


def test_cli_exits_cleanly_on_control_c(monkeypatch: pytest.MonkeyPatch) -> None:
    def interrupt(_args: object) -> int:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "command_play", interrupt)
    assert cli.main(["play"]) == 130


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


def test_dth_cli_dispatch_builds_a_policy_driven_agent(tmp_path) -> None:
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "play",
            "--dth-complete-tablebase",
            str(tmp_path / "missing"),
        ]
    )
    with pytest.raises(FileNotFoundError):
        cli._make_hal(args)


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
