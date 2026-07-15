from __future__ import annotations

import json

import pytest

from stl.commands.gen0_eval import _gate
from stl.learning.holdout import claim_holdout_use, complete_holdout_use


def _policy() -> dict:
    aggregate = {"count": 0, "mean": None, "median": None, "maximum": None}
    return {
        "saddle_gap": {"count": 1, "mean": 0.01, "median": 0.01, "maximum": 0.01},
        "maximum_illegal_mass": 0.0,
        "label_value_recompute_error": {**aggregate, "maximum": 0.0},
        "cutoff_recompute_error": {**aggregate, "maximum": 0.0},
        "strict_unique_dropper_tv": aggregate,
        "strict_unique_checker_tv": aggregate,
    }


def _sources() -> dict[str, dict[str, float | int]]:
    return {
        source: {
            "count": 1,
            "mse": 0.0,
            "max_abs_error": 0.0,
            "max_cutoff_probability": 0.5 if source.startswith("exact_") else 0.0,
        }
        for source in (
            "terminal",
            "tablebase",
            "tablebase_interior",
            "exact_horizon_2",
            "exact_horizon_3",
        )
    }


def test_taxonomy_family_gate_prevents_easy_rows_from_masking_double_overflow():
    taxonomy = {
        "family:boundary_single_overflow": {
            "count": 100,
            "mse": 0.0,
            "max_abs_error": 0.0,
        },
        "family:boundary_double_overflow": {
            "count": 1,
            "mse": 0.04,
            "max_abs_error": 0.2,
        },
    }
    gate = _gate(
        value_sources=_sources(),
        policy=_policy(),
        reload_max_abs_difference=0.0,
        tablebase_mse_max=0.01,
        tablebase_interior_mse_max=0.05,
        exact_saddle_gap_max=0.05,
        unique_policy_tv_median_max=0.15,
        exact_cutoff_max=0.5,
        taxonomy_metrics=taxonomy,
        boundary_max_abs_error=0.1,
    )
    assert gate["passed"] is False
    assert "taxonomy_mse:family:boundary_double_overflow" in gate["failures"]
    assert "taxonomy_max_abs_error:family:boundary_double_overflow" in gate["failures"]


def test_holdout_ledger_binds_one_checkpoint_and_allows_identical_resume(tmp_path):
    ledger = tmp_path / "holdout.use.json"
    config = {"tablebase_mse_max": 0.01}
    first = claim_holdout_use(
        ledger,
        seal_digest="seal-a",
        checkpoint_sha256="checkpoint-a",
        evaluation_config=config,
    )
    resumed = claim_holdout_use(
        ledger,
        seal_digest="seal-a",
        checkpoint_sha256="checkpoint-a",
        evaluation_config=config,
    )
    assert first == resumed
    with pytest.raises(ValueError, match="another candidate"):
        claim_holdout_use(
            ledger,
            seal_digest="seal-a",
            checkpoint_sha256="checkpoint-b",
            evaluation_config=config,
        )

    report = tmp_path / "report.json"
    report.write_text(json.dumps({"passed": True}), encoding="utf-8")
    complete_holdout_use(ledger, report_path=report, passed=True)
    assert json.loads(ledger.read_text(encoding="utf-8"))["status"] == "completed"
