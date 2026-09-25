"""Tests for ``python -m terminal play`` and its parity with the play session."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from arena.session import Phase
from arena.testing import RecordingHal as _RecordingHal, make_session as _session
from terminal import cli
from terminal.cli import build_parser, command_play


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
    import arena.policies.adaptive as adaptive

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


def test_cli_exits_cleanly_on_control_c(monkeypatch: pytest.MonkeyPatch) -> None:
    def interrupt(_args: object) -> int:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "command_play", interrupt)
    assert cli.main(["play"]) == 130


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


def test_session_reproduces_the_cli_public_history(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The refactor guard: driving the session by hand matches the CLI."""

    def _fresh_hal(*_args, **_kwargs):
        return _RecordingHal()

    monkeypatch.setattr(cli, "_make_hal", _fresh_hal)
    monkeypatch.setattr(cli, "_human_action", lambda *, actor, role, legal: legal[-1])
    transcript = tmp_path / "session.json"
    args = cli.build_parser().parse_args(
        [
            "play",
            "--seed",
            "41",
            "--max-half-rounds",
            "4",
            "--transcript",
            str(transcript),
        ]
    )
    assert cli.command_play(args) == 0
    from_cli = json.loads(transcript.read_text(encoding="utf-8"))["games"][0]

    session = _session(max_half_rounds=4)
    session.begin()
    while session.phase is Phase.AWAITING_ACTION:
        session.submit(session.legal_actions()[-1])
        session.acknowledge()

    assert session.finish()["public_history"] == from_cli["public_history"]


def test_human_play_requires_pm_hal_pure_dth_surface() -> None:
    args = build_parser().parse_args(["play", "--hal-agent", "pm-hal", "--skip-rules"])
    with pytest.raises(ValueError, match="pure-DTH"):
        command_play(args)

    pure = build_parser().parse_args(
        ["play", "--hal-agent", "pm-hal", "--pure-dth", "--skip-rules"]
    )
    assert pure.pure_dth is True
