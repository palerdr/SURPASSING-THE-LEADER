import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train_saved_corpus_gate import (
    build_parser,
    monotonicity_verdict,
    parse_source_weights,
    parse_thresholds,
)


def test_parse_thresholds_accepts_source_value_pairs():
    assert parse_thresholds(["exact_horizon_2:0.05", "tablebase_interior:0.04"]) == {
        "exact_horizon_2": 0.05,
        "tablebase_interior": 0.04,
    }


def test_parse_thresholds_rejects_malformed_item():
    with pytest.raises(SystemExit):
        parse_thresholds(["exact_horizon_2"])


def test_parse_source_weights_returns_tuple_pairs():
    assert parse_source_weights(["tier_a_d1_hal:0.02"]) == (("tier_a_d1_hal", 0.02),)


def test_monotonicity_requires_strict_improvement():
    assert monotonicity_verdict(0.04, 0.05)
    assert not monotonicity_verdict(0.05, 0.05)


def test_saved_corpus_parser_exposes_policy_distillation():
    args = build_parser().parse_args(
        [
            "--targets", "targets.npz",
            "--out-dir", "out",
            "--held-out-targets", "holdout.npz",
            "--reference-checkpoint", "current.pt",
            "--policy-distill-weight", "3.0",
        ]
    )

    assert args.reference_checkpoint == "current.pt"
    assert args.policy_distill_weight == 3.0


def test_saved_corpus_parser_exposes_target_dedupe():
    args = build_parser().parse_args(
        [
            "--targets", "targets.npz",
            "--out-dir", "out",
            "--held-out-targets", "holdout.npz",
            "--dedupe-targets",
        ]
    )

    assert args.dedupe_targets is True


def test_saved_corpus_parser_exposes_selection_metric():
    args = build_parser().parse_args(
        [
            "--targets", "targets.npz",
            "--out-dir", "out",
            "--held-out-targets", "holdout.npz",
            "--selection-metric", "policy_nll",
        ]
    )

    assert args.selection_metric == "policy_nll"


def test_saved_corpus_parser_exposes_dedupe_sources():
    args = build_parser().parse_args(
        [
            "--targets", "targets.npz",
            "--out-dir", "out",
            "--held-out-targets", "holdout.npz",
            "--dedupe-targets",
            "--dedupe-source", "policy_guard",
        ]
    )

    assert args.dedupe_source == ["policy_guard"]
