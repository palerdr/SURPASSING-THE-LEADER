from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from arena.contracts import (
    CanonicalDecision,
    PublicDecisionState,
    PublicGameOutcome,
    PublicHalfRound,
    PublicPlayerState,
    end_provider_game,
    observe_provider,
    reset_provider_game,
)
from arena.policies import evaluate_aggro_hal as evaluation
from arena.policies.aggro_hal import (
    AggroHalConfig,
    AggroHalNetwork,
    AggroHalPolicyProvider,
)
from arena.policies.opponent_league import (
    ACTIONS,
    DETERMINISTIC,
    OpponentFamilyManifest,
    OpponentManifestEntry,
)
from dth.agent import CertifiedStageGame

CPU = torch.device("cpu")


class _ExactAgent:
    def __init__(self) -> None:
        self.stage_calls = 0
        self.decide_calls = 0

    @staticmethod
    def _stage(state) -> CertifiedStageGame:
        actions = np.arange(1, 61, dtype=np.float64)
        matrix = (actions[:, None] - actions[None, :]) / 60.0
        uniform = np.full(60, 1.0 / 60.0, dtype=np.float64)
        return CertifiedStageGame(
            state=tuple(state),
            value=0.0,
            matrix=matrix,
            drop_policy=uniform.copy(),
            check_policy=uniform.copy(),
            saddle_gap=0.0,
        )

    def stage_game(self, state) -> CertifiedStageGame:
        self.stage_calls += 1
        return self._stage(state)

    def decide(self, state):
        self.decide_calls += 1
        stage = self._stage(state)
        return SimpleNamespace(
            drop_policy=tuple(stage.drop_policy),
            check_policy=tuple(stage.check_policy),
        )


def _decision(role: str, actor_name: str) -> CanonicalDecision:
    return CanonicalDecision(
        role=role,
        actor_name=actor_name,
        turn_duration=60,
        legal_seconds=ACTIONS,
        checker_cylinder_seconds=0.0,
        checker_ttd_seconds=60.0,
        dropper_cylinder_seconds=0.0,
        dropper_ttd_seconds=60.0,
        native_state=object(),
    )


def _argmax_policy(provider, decision: CanonicalDecision) -> int:
    raw = provider.policy(decision)
    return max(raw, key=raw.__getitem__)


def _fake_match(memory_before_game: list[bool]):
    def play(
        provider_one,
        provider_two,
        *,
        seed: int,
        start_clock: int,
        max_half_rounds: int,
        game_index: int = 0,
        pure_dth: bool = False,
    ) -> tuple[str | None, int]:
        del seed, start_clock, max_half_rounds
        assert pure_dth
        reset_provider_game(provider_one)
        reset_provider_game(provider_two)
        candidate = (
            provider_two
            if isinstance(provider_one, evaluation._TruthTrackingOpponent)
            else provider_one
        )
        if isinstance(candidate, evaluation._MeasuredAggroProvider):
            memory_before_game.append(candidate.provider.has_session_memory)

        drop = _argmax_policy(provider_one, _decision("dropper", "Hal"))
        check = _argmax_policy(provider_two, _decision("checker", "Baku"))
        candidate_first = not isinstance(
            provider_one, evaluation._TruthTrackingOpponent
        )
        winner = "Hal" if candidate_first else "Baku"
        record = PublicHalfRound(
            game_index=game_index,
            half_round_index=0,
            pre_decision_state=PublicDecisionState(
                game_clock_seconds=720.0,
                round_index=0,
                half_index=1,
                turn_duration=60,
                players=(
                    PublicPlayerState("Hal", 0.0, 60.0),
                    PublicPlayerState("Baku", 0.0, 60.0),
                ),
            ),
            dropper_name="Hal",
            checker_name="Baku",
            drop_time=drop,
            check_time=check,
            outcome="check_success",
            game_over=True,
            winner_name=winner,
        )
        observe_provider(provider_one, record)
        observe_provider(provider_two, record)
        outcome = PublicGameOutcome(
            game_index=game_index,
            winner_name=winner,
            half_rounds=1,
        )
        end_provider_game(provider_one, outcome)
        end_provider_game(provider_two, outcome)
        return winner, 1

    return play


def test_prediction_summary_reports_nll_calibration_regret_and_gate() -> None:
    samples = [
        evaluation._PredictionSample(
            role="dropper",
            realized_nll=0.2,
            expected_nll=0.3,
            brier_score=0.1,
            top_confidence=0.8,
            top_correct=True,
            oracle_regret=0.1,
            direct_weight=0.2,
        ),
        evaluation._PredictionSample(
            role="checker",
            realized_nll=1.0,
            expected_nll=0.9,
            brier_score=0.7,
            top_confidence=0.6,
            top_correct=False,
            oracle_regret=0.3,
            direct_weight=0.8,
        ),
    ]

    summary = evaluation.summarize_prediction_metrics(samples)

    assert summary["decisions"] == 2
    assert summary["opponent_realized_nll"] == pytest.approx(0.6)
    assert summary["opponent_expected_nll"] == pytest.approx(0.6)
    assert summary["opponent_brier_score"] == pytest.approx(0.4)
    assert summary["opponent_top_label_ece_10_bin"] == pytest.approx(0.4)
    assert summary["mean_one_step_oracle_regret"] == pytest.approx(0.2)
    assert summary["direct_gate"]["mean"] == pytest.approx(0.5)
    assert summary["direct_gate"]["fraction_at_least_half"] == pytest.approx(0.5)
    assert summary["by_role"]["dropper"]["decisions"] == 1
    assert summary["by_role"]["checker"]["decisions"] == 1


def test_evaluation_pairs_seats_and_preserves_only_within_session_memory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    torch.manual_seed(17)
    config = AggroHalConfig(hidden_size=8, head_hidden_size=6)
    model = AggroHalNetwork(config).to(CPU).eval()
    exact = _ExactAgent()
    provider = AggroHalPolicyProvider(
        tmp_path,
        model,
        config,
        agent=exact,
        device=CPU,
    )
    manifest = OpponentFamilyManifest(
        split="validation",
        entries=(OpponentManifestEntry(DETERMINISTIC, (7,)),),
    )
    memory_before_game: list[bool] = []
    monkeypatch.setattr(
        evaluation,
        "play_match_game",
        _fake_match(memory_before_game),
    )

    report = evaluation.evaluate_aggro_hal(
        split="validation",
        provider=provider,
        manifest=manifest,
        sessions_per_opponent=2,
        seat_pairs_per_session=1,
        max_half_rounds=1,
        success_criteria=evaluation.SuccessCriteria(
            min_decisive_games=1,
            min_opponent_seed_units=1,
            min_decisive_win_rate=0.0,
            min_each_seat_decisive_win_rate=0.0,
            max_stop_rate=1.0,
            min_decisive_win_rate_uplift_vs_exact=-1.0,
        ),
        exact_agent=exact,
        device="cpu",
    )

    assert report["device"] == "cpu"
    assert report["summaries"]["aggro"]["games"] == 4
    assert report["summaries"]["exact"]["games"] == 4
    assert report["summaries"]["aggro"]["decisive_win_rate"] == 1.0
    assert report["summaries"]["aggro"]["stopped"] == 0
    assert report["summaries"]["aggro"]["by_seat"]["Hal"]["games"] == 2
    assert report["summaries"]["aggro"]["by_seat"]["Baku"]["games"] == 2
    assert report["prediction_metrics"]["decisions"] == 4
    assert report["prediction_metrics"]["opponent_realized_nll"] is not None
    assert report["prediction_metrics"]["mean_one_step_oracle_regret"] is not None
    assert report["success_gate"]["passed"] is True

    # One fresh hidden state per session, then the same memory across the seat swap.
    assert memory_before_game == [False, True, False, True]
    for session in report["sessions"]["aggro"]:
        first, second = session["games"]
        assert first["seed"] == second["seed"]
        assert {first["candidate_seat"], second["candidate_seat"]} == {"Hal", "Baku"}
    assert [
        session["games"][0]["candidate_seat"] for session in report["sessions"]["aggro"]
    ] == ["Baku", "Hal"]

    memory_before_game.clear()
    ablation = evaluation.evaluate_aggro_hal(
        split="validation",
        provider=provider,
        manifest=manifest,
        sessions_per_opponent=1,
        seat_pairs_per_session=1,
        max_half_rounds=1,
        reset_recurrent_each_game=True,
        success_criteria=evaluation.SuccessCriteria(
            min_decisive_games=1,
            min_opponent_seed_units=1,
            min_decisive_win_rate=0.0,
            min_each_seat_decisive_win_rate=0.0,
            max_stop_rate=1.0,
            min_decisive_win_rate_uplift_vs_exact=-1.0,
        ),
        exact_agent=exact,
        device="cpu",
    )
    assert ablation["reset_recurrent_each_game"] is True
    assert memory_before_game == [False, False]

    history_free = evaluation.evaluate_aggro_hal(
        split="validation",
        provider=provider,
        manifest=manifest,
        sessions_per_opponent=1,
        seat_pairs_per_session=1,
        max_half_rounds=1,
        erase_history_each_decision=True,
        success_criteria=evaluation.SuccessCriteria(
            min_decisive_games=1,
            min_opponent_seed_units=1,
            min_decisive_win_rate=0.0,
            min_each_seat_decisive_win_rate=0.0,
            max_stop_rate=1.0,
            min_decisive_win_rate_uplift_vs_exact=-1.0,
        ),
        exact_agent=exact,
        device="cpu",
    )
    assert history_free["erase_history_each_decision"] is True

    comparison = report["common_scenario_all_game_score"]
    assert comparison["opponent_seed_units"] == 1
    assert comparison["common_scenarios"] == 4
    assert "paired_decisive_win_rate_delta" not in report
    assert [
        game["game_index"]
        for session in report["sessions"]["aggro"]
        for game in session["games"]
    ] == [0, 1, 0, 1]

    destination = evaluation.write_evaluation_report(report, tmp_path / "report.json")
    assert (
        json.loads(destination.read_text(encoding="utf-8"))["success_gate"]["passed"]
        is True
    )


def test_evaluation_rejects_non_cpu_device_before_loading_any_artifact() -> None:
    config = AggroHalConfig(hidden_size=4, head_hidden_size=4)
    model = AggroHalNetwork(config).to(CPU)

    with pytest.raises(ValueError, match="CPU-only"):
        evaluation.evaluate_aggro_hal(
            network=model,
            split="validation",
            device="cuda",
        )
