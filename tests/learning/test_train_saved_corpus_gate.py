import os
import sys

import pytest

sys.path.insert(0, os.getcwd())

from stl.commands.train_saved_corpus import (
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
