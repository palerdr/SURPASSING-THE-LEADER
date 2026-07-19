"""Tests for the monotonicity gate in scripts/run_gen_iteration.py (ticket B6).

The gate used to be print-only: after the calibration gate passed, the
comparison against --prev-gen-holdout-mse printed 'improvement'/'regression'
and returned 0 either way, so a regressing generation shipped green. These
tests pin the enforcement contract:

  - ``monotonicity_verdict(new, prev)`` is the acceptance rule (strictly
    below = improvement; equal or worse = regression),
  - ``--enforce-monotonicity`` defaults to True (regression => exit 1),
  - ``--no-enforce-monotonicity`` restores report-only behavior.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.getcwd())

from stl.commands.train_generation import (
    build_parser,
    canary_gate_decision,
    monotonicity_verdict,
    write_gate_report,
)

REQUIRED_ARGS = [
    "--in-checkpoint", "ckpt.pt",
    "--out-dir", "out/",
    "--out-targets", "targets.npz",
    "--held-out-targets", "holdout.npz",
]


class TestMonotonicityVerdict:
    def test_strict_improvement_is_accepted(self):
        assert monotonicity_verdict(0.05, 0.06) is True

    def test_regression_is_rejected(self):
        assert monotonicity_verdict(0.07, 0.06) is False

    def test_equal_mse_is_a_regression(self):
        """new == prev must NOT count as improvement: the charter requires the
        new gen to be STRICTLY below the prior gen (exit non-zero on >=)."""
        assert monotonicity_verdict(0.06, 0.06) is False

    def test_regression_exit_code_is_nonzero_under_default_enforcement(self):
        """The exact exit-code rule main() applies after the calibration gate
        passes: regression + enforcement => non-zero exit."""
        args = build_parser().parse_args(
            REQUIRED_ARGS + ["--prev-gen-holdout-mse", "0.06"]
        )
        new_mse = 0.07  # worse than prev 0.06
        improved = monotonicity_verdict(new_mse, args.prev_gen_holdout_mse)
        exit_code = 1 if (not improved and args.enforce_monotonicity) else 0
        assert exit_code == 1

    def test_improvement_exit_code_is_zero_under_default_enforcement(self):
        args = build_parser().parse_args(
            REQUIRED_ARGS + ["--prev-gen-holdout-mse", "0.06"]
        )
        improved = monotonicity_verdict(0.05, args.prev_gen_holdout_mse)
        exit_code = 1 if (not improved and args.enforce_monotonicity) else 0
        assert exit_code == 0


class TestEnforceMonotonicityFlagPlumbing:
    def test_help_text_is_windows_console_encodable(self):
        """argparse --help must not crash under the default Windows cp1252
        console encoding."""
        build_parser().format_help().encode("cp1252")

    def test_default_is_enforced(self):
        args = build_parser().parse_args(REQUIRED_ARGS)
        assert args.enforce_monotonicity is True

    def test_explicit_enable(self):
        args = build_parser().parse_args(REQUIRED_ARGS + ["--enforce-monotonicity"])
        assert args.enforce_monotonicity is True

    def test_opt_out_flag_disables_enforcement(self):
        args = build_parser().parse_args(REQUIRED_ARGS + ["--no-enforce-monotonicity"])
        assert args.enforce_monotonicity is False

    def test_opt_out_makes_regression_report_only(self):
        args = build_parser().parse_args(
            REQUIRED_ARGS
            + ["--prev-gen-holdout-mse", "0.06", "--no-enforce-monotonicity"]
        )
        improved = monotonicity_verdict(0.07, args.prev_gen_holdout_mse)
        exit_code = 1 if (not improved and args.enforce_monotonicity) else 0
        assert exit_code == 0


class TestCanaryGateVerdict:
    def test_no_canary_report_has_no_decision(self):
        assert canary_gate_decision(None) is None

    def test_non_regressing_canary_passes(self):
        decision = canary_gate_decision(
            {"aggregate": {"delta": {"wins_delta": 0, "score_rate_delta": 0.0}}},
            min_wins_delta=0,
        )

        assert decision["status"] == "passed"
        assert decision["wins_delta"] == 0

    def test_regressing_canary_fails(self):
        decision = canary_gate_decision(
            {"aggregate": {"delta": {"wins_delta": -1, "score_rate_delta": -0.1}}},
            min_wins_delta=0,
        )

        assert decision["status"] == "failed"
        assert "wins_delta -1" in decision["reason"]

    def test_malformed_canary_fails_closed(self):
        decision = canary_gate_decision({"aggregate": {}}, min_wins_delta=0)

        assert decision["status"] == "failed"
        assert "missing aggregate.delta.wins_delta" in decision["reason"]


class TestTierAExtraTargetPlumbing:
    def test_extra_targets_are_repeatable_and_weighted(self):
        args = build_parser().parse_args(
            REQUIRED_ARGS
            + [
                "--extra-targets", "tier_a_1.npz",
                "--extra-targets", "tier_a_2.npz",
                "--tier-a-weight", "3.5",
                "--tier-a-policy-weight", "0.25",
                "--tier-a-replicate", "4",
            ]
        )
        assert args.extra_targets == ["tier_a_1.npz", "tier_a_2.npz"]
        assert args.tier_a_weight == 3.5
        assert args.tier_a_policy_weight == 0.25
        assert args.tier_a_replicate == 4

    def test_generation_training_can_warm_start_from_checkpoint(self):
        args = build_parser().parse_args(
            REQUIRED_ARGS
            + ["--init-checkpoint", "current/best.pt", "--learning-rate", "3e-6"]
        )

        assert args.init_checkpoint == "current/best.pt"
        assert args.learning_rate == 3e-6

    def test_generation_training_has_policy_distillation_plumbing(self):
        args = build_parser().parse_args(
            REQUIRED_ARGS
            + [
                "--reference-checkpoint", "current/best.pt",
                "--value-distill-weight", "10",
                "--policy-distill-weight", "2.5",
                "--policy-source-weight", "self_play_mcts:0.1",
                "--source-weight", "self_play_mcts:0.05",
                "--selection-metric", "policy_nll",
                "--early-stopping-patience", "7",
            ]
        )

        assert args.reference_checkpoint == "current/best.pt"
        assert args.value_distill_weight == 10.0
        assert args.policy_distill_weight == 2.5
        assert args.policy_source_weight == ["self_play_mcts:0.1"]
        assert args.source_weight == ["self_play_mcts:0.05"]
        assert args.selection_metric == "policy_nll"
        assert args.early_stopping_patience == 7

    def test_bootstrap_can_be_bounded_to_critical_states(self):
        args = build_parser().parse_args(
            REQUIRED_ARGS
            + [
                "--subgame-resolve-at-critical",
                "--subgame-resolve-horizon", "1",
                "--subgame-resolve-cfr-iters", "50",
                "--bootstrap-critical-only",
                "--bootstrap-max-states", "12",
            ]
        )

        assert args.subgame_resolve_at_critical is True
        assert args.subgame_resolve_horizon == 1
        assert args.subgame_resolve_cfr_iters == 50
        assert args.bootstrap_critical_only is True
        assert args.bootstrap_max_states == 12

    def test_canary_gate_threshold_defaults_to_non_regression(self):
        args = build_parser().parse_args(REQUIRED_ARGS)

        assert args.canary_min_wins_delta == 0

    def test_canary_gate_threshold_can_be_raised(self):
        args = build_parser().parse_args(
            REQUIRED_ARGS + ["--canary-min-wins-delta", "2"]
        )

        assert args.canary_min_wins_delta == 2

    def test_subgame_resolve_horizon_defaults_to_bounded_h3(self):
        args = build_parser().parse_args(REQUIRED_ARGS)

        assert args.subgame_resolve_horizon == 3


def test_run_gen_iteration_help_works_on_windows():
    result = subprocess.run(
        [sys.executable, "-m", "stl.commands.train_generation", "--help"],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--bootstrap-max-states" in result.stdout


def test_write_gate_report_records_pass_and_fail(tmp_path):
    args = SimpleNamespace(
        out_dir=str(tmp_path),
        out_targets="targets.npz",
        extra_targets=None,
        canary_report=None,
        selection_metric="value_mse",
    )
    train_result = SimpleNamespace(
        checkpoint_path="out/best.pt",
        best_val_mse=0.1,
        best_epoch=2,
        best_selection_score=0.1,
        best_per_source_mse={"terminal": 0.01},
    )
    report = SimpleNamespace(
        overall_mse=0.05,
        mse_per_source={"terminal": 0.01},
        mean_unresolved_probability_per_source={"terminal": 0.0},
    )
    bootstrap_report = {"runtime_seconds": 1.2, "rows_generated": 2}
    training_corpus_report = {"rows": 3}

    pass_path = write_gate_report(
        args=args,
        train_result=train_result,
        report=report,
        gate_passed=True,
        gate_error=None,
        beats_prev_gen=True,
        bootstrap_report=bootstrap_report,
        training_corpus_report=training_corpus_report,
    )
    passed = json.loads(pass_path.read_text(encoding="utf-8"))
    assert passed["gate_passed"] is True
    assert passed["bootstrap"]["rows_generated"] == 2
    assert passed["train_selection_metric"] == "value_mse"
    assert passed["train_best_selection_score"] == 0.1

    fail_path = write_gate_report(
        args=args,
        train_result=train_result,
        report=report,
        gate_passed=False,
        gate_error="failed",
        beats_prev_gen=False,
        bootstrap_report=bootstrap_report,
        training_corpus_report=training_corpus_report,
    )
    failed = json.loads(fail_path.read_text(encoding="utf-8"))
    assert failed["gate_passed"] is False
    assert failed["gate_error"] == "failed"


def test_write_gate_report_embeds_canary_gate_decision(tmp_path):
    canary_path = tmp_path / "pattern_canary.json"
    canary_path.write_text(
        json.dumps({"aggregate": {"delta": {"wins_delta": -2}}}),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        out_dir=str(tmp_path),
        out_targets="targets.npz",
        extra_targets=None,
        canary_report=str(canary_path),
        canary_min_wins_delta=0,
        selection_metric="value_mse",
    )
    train_result = SimpleNamespace(
        checkpoint_path="out/best.pt",
        best_val_mse=0.1,
        best_epoch=2,
        best_selection_score=0.1,
        best_per_source_mse={"terminal": 0.01},
    )
    report = SimpleNamespace(
        overall_mse=0.05,
        mse_per_source={"terminal": 0.01},
        mean_unresolved_probability_per_source={"terminal": 0.0},
    )

    path = write_gate_report(
        args=args,
        train_result=train_result,
        report=report,
        gate_passed=False,
        gate_error="canary wins_delta -2 < required 0",
        beats_prev_gen=True,
        bootstrap_report={"rows_generated": 2},
        training_corpus_report={"rows": 3},
    )

    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["canary_gate"]["status"] == "failed"
    assert written["canary_gate"]["wins_delta"] == -2
