from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from arena.policies.evaluate_pm_hal import (
    EVALUATION_SCHEMA,
    PMEvaluationConfig,
    PMEvaluationEntry,
    PMPredictionSample,
    PMPromotionGate,
    _change_metrics,
    _policy_metrics,
    evaluate_pm_hal,
    load_evaluation_config,
    write_evaluation_report,
)
from arena.policies.pm_hal import PMHalConfig, PMHalDecision
from dth.agent import CertifiedStageGame


class _StageAgent:
    def stage_game(self, state) -> CertifiedStageGame:
        normalized = tuple(int(value) for value in state)
        uniform = np.full(60, 1.0 / 60.0, dtype=np.float64)
        return CertifiedStageGame(
            state=normalized,
            value=1.0 / 60.0,
            matrix=np.eye(60, dtype=np.float64),
            drop_policy=uniform.copy(),
            check_policy=uniform.copy(),
            saddle_gap=0.0,
        )


def _protocol() -> PMEvaluationConfig:
    return PMEvaluationConfig(
        status="git-registered-before-execution",
        benchmark_id="pm-test",
        entries=(PMEvaluationEntry("deterministic", (91,)),),
        sessions_per_opponent=1,
        seat_pairs_per_session=1,
        start_clocks=(720,),
        max_half_rounds=2,
        evaluation_seed=1000,
        bootstrap_seed=2000,
        bootstrap_samples=100,
        promotion_gate=PMPromotionGate(
            minimum_opponent_identities=1,
            maximum_risk_violations=0,
            minimum_clustered_lower_bound_vs_exact=-1.0,
            minimum_clustered_lower_bound_vs_strongest_component=-1.0,
        ),
        notes=("test",),
    )


def test_tracked_evaluation_config_is_hash_bound_and_has_56_identities() -> None:
    config = load_evaluation_config()
    assert config.schema_version == "arena-pm-hal-evaluation-config-v2"
    assert config.status == "git-registered-before-execution"
    assert config.opponent_identities == 56
    assert len(config.families) == 14
    assert config.benchmark_id == "pm-hal-broad-synthetic-confirmation-v3"
    assert config.entries[0].seeds[0] == 71001
    assert config.expected_pm_config_sha256 is not None
    assert config.expected_aggro_checkpoint_sha256 is not None
    assert config.expected_dth_table_digest is not None


def test_v3_confirmation_freeze_binds_tracked_protocol_bytes() -> None:
    root = Path(__file__).resolve().parents[3]
    freeze = json.loads(
        (root / "src/arena/config/pm_hal_confirmation_v3.json").read_text(
            encoding="utf-8"
        )
    )

    assert freeze["schema_version"] == "arena-pm-hal-confirmation-freeze-v1"
    assert freeze["immutable"] is True
    assert freeze["evidence"]["promotion_gate"]["passed"] is False
    assert freeze["protocol_commit"] == "b76147fc3718492353b24131ce756c558c4c43c8"
    for binding in ("evaluation_config", "controller_config"):
        evidence = freeze["bindings"][binding]
        payload = (root / evidence["path"]).read_bytes()
        assert len(payload) == evidence["bytes"]
        assert hashlib.sha256(payload).hexdigest() == evidence["sha256"]
    assert freeze["bindings"]["evaluation_report"]["generated_artifact"] is True
    assert freeze["immutability"]["v3_must_not_be_rerun_or_rebound"] is True


def test_smoke_evaluation_reports_prediction_risk_and_paired_metrics(
    tmp_path: Path,
) -> None:
    report = evaluate_pm_hal(
        evaluation_config=_protocol(),
        artifact_dir=tmp_path,
        pm_config=PMHalConfig(posterior_samples=16),
        exact_agent=_StageAgent(),
    )
    assert report["schema_version"] == EVALUATION_SCHEMA
    assert report["claim_scope"].startswith("development")
    assert report["git_provenance"] is None
    assert report["human_validation"] is False
    assert report["aggro_component_enabled"] is False
    assert report["prediction_metrics"]["decisions"] > 0
    assert set(report["prediction_metrics"]["component_prediction_metrics"]) == {
        "uniform",
        "equilibrium",
        "adaptive",
        "perfect",
        "change_point",
        "outcome",
    }
    clustered = report["prediction_metrics"]["opponent_identity_clustered"]
    assert clustered["expected_nll"]["opponent_identity_units"] == 1
    assert len(clustered["expected_nll"]["cluster_bootstrap_95"]) == 2
    assert report["policy_metrics"]["per_game_budget_violations"] == 0
    assert report["policy_metrics"]["mode_cap_violations"] == 0
    assert report["policy_metrics"]["total_independent_risk_audit_violations"] == 0
    assert report["policy_metrics"]["independent_matrix_recomputation"] is True
    assert (
        report["paired_all_game_score_comparisons"]["exact"]["opponent_identity_units"]
        == 1
    )
    assert set(report["summaries"]) == {"pm", "perfect", "adaptive", "exact"}
    json.dumps(report)

    destination = write_evaluation_report(report, tmp_path / "report.json")
    written = json.loads(destination.read_text(encoding="utf-8"))
    assert written["schema_version"] == EVALUATION_SCHEMA


def _sample(
    *,
    index: int,
    truth_shift: bool = False,
    change_detected: bool = False,
    actual_loss: float = 0.0,
    budget_charge: float = 0.0,
) -> PMPredictionSample:
    return PMPredictionSample(
        family="deterministic",
        opponent_seed=1,
        replicate=0,
        game_index=0,
        session_decision_index=index,
        role="dropper",
        opponent_role="checker",
        realized_nll=0.0,
        expected_nll=0.0,
        realized_brier=0.0,
        expected_brier=0.0,
        component_expected_nll=(),
        component_expected_brier=(),
        oracle_regret=0.0,
        top_confidence=1.0,
        top_correct=True,
        forecast_confidence=1.0,
        forecast_disagreement=0.0,
        mode="shield",
        actual_worst_case_loss=actual_loss,
        budget_charge=budget_charge,
        truth_shift=truth_shift,
        change_detected=change_detected,
    )


def test_change_metric_requires_a_new_alarm_onset_after_a_shift() -> None:
    metrics = _change_metrics(
        [
            _sample(index=0, change_detected=True),
            _sample(index=1, truth_shift=True, change_detected=True),
        ],
        detector_threshold=0.35,
    )
    assert metrics["truth_shifts"] == 1
    assert metrics["detected_shifts"] == 0
    assert metrics["undetected_shifts"] == 1
    assert metrics["false_alarm_onsets"] == 1


def test_policy_metrics_recompute_loss_instead_of_trusting_diagnostics() -> None:
    point = np.zeros(60, dtype=np.float64)
    point[0] = 1.0
    uniform = tuple(float(1.0 / 60.0) for _ in range(60))
    decision = PMHalDecision(
        state=(0, 0, 0, 0),
        game_index=0,
        role="dropper",
        opponent_role="checker",
        mode="shield",
        policy=tuple(float(value) for value in point),
        opponent_policy=uniform,
        component_weights=(),
        component_policies=(),
        evidence_count=0,
        effective_evidence=0.0,
        forecast_entropy=0.0,
        forecast_disagreement=0.0,
        change_probability=0.0,
        expected_run_length=0.0,
        prequential_skill=1.0,
        confidence=0.0,
        selected_candidate="exact",
        selected_source="exact",
        selected_actual_worst_case_loss=0.0,
        selected_budget_charge=0.0,
        game_epsilon_spent=0.0,
        game_epsilon_remaining=12.0,
        expected_improvement=0.0,
        improvement_support=0.0,
        aggro_enabled=False,
        aggro_direct_weight=None,
        candidates=(),
    )
    metrics = _policy_metrics(
        [_sample(index=0)],
        [decision],
        config=PMHalConfig(posterior_samples=16),
        agent=_StageAgent(),
    )
    assert metrics["diagnostic_loss_mismatches"] == 1
    assert metrics["risk_charge_understates_actual_loss_violations"] == 1
    assert metrics["mode_cap_violations"] == 1
    assert metrics["total_independent_risk_audit_violations"] > 0


def test_benchmark_results_are_invariant_to_manifest_order(tmp_path: Path) -> None:
    gate = PMPromotionGate(
        minimum_opponent_identities=2,
        maximum_risk_violations=0,
        minimum_clustered_lower_bound_vs_exact=-1.0,
        minimum_clustered_lower_bound_vs_strongest_component=-1.0,
    )
    entries = (
        PMEvaluationEntry("deterministic", (91,)),
        PMEvaluationEntry("periodic", (92,)),
    )

    def run(ordered_entries: tuple[PMEvaluationEntry, ...]) -> dict[str, object]:
        protocol = PMEvaluationConfig(
            status="git-registered-before-execution",
            benchmark_id="order-test",
            entries=ordered_entries,
            sessions_per_opponent=1,
            seat_pairs_per_session=1,
            start_clocks=(720,),
            max_half_rounds=2,
            evaluation_seed=3000,
            bootstrap_seed=4000,
            bootstrap_samples=100,
            promotion_gate=gate,
            notes=("test",),
        )
        return evaluate_pm_hal(
            evaluation_config=protocol,
            artifact_dir=tmp_path,
            pm_config=PMHalConfig(posterior_samples=16),
            exact_agent=_StageAgent(),
        )

    forward = run(entries)
    reverse = run(tuple(reversed(entries)))
    assert forward["policy_metrics"] == reverse["policy_metrics"]
    assert (
        forward["summaries"]["pm"]["all_game_score"]
        == reverse["summaries"]["pm"]["all_game_score"]
    )
