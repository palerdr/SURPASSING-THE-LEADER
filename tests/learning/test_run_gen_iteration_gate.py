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

import os
import sys

sys.path.insert(0, os.getcwd())

from stl.commands.train_generation import build_parser, monotonicity_verdict

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

    def test_bootstrap_can_be_bounded_to_critical_states(self):
        args = build_parser().parse_args(
            REQUIRED_ARGS
            + ["--bootstrap-critical-only", "--bootstrap-max-states", "12"]
        )

        assert args.bootstrap_critical_only is True
        assert args.bootstrap_max_states == 12
