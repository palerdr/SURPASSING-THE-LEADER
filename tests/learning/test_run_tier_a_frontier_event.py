import os
import sys
from argparse import Namespace

sys.path.insert(0, os.getcwd())

from stl.commands.tier_a_frontier import build_next_run, summarize


def _row(width: float, tier_width: float, frontier_hits: int = 10, tablebase_hits: int = 8):
    return {
        "baseline": {"width": width},
        "tier_a_frontier": {
            "width": tier_width,
            "frontier_hits": frontier_hits,
            "tablebase_frontier_hits": tablebase_hits,
        },
        "width_reduction": width - tier_width,
        "width_reduction_fraction": (width - tier_width) / width,
        "tablebase_frontier_coverage": tablebase_hits / frontier_hits if frontier_hits else 0.0,
    }


def _args():
    return Namespace(
        checkpoint="checkpoints/gen_ceiling_tbw15_wd1e-4/best.pt",
        next_target_count=50_000,
        next_target_max_width=0.01,
        next_iterations=300,
        next_epochs=120,
        next_tier_a_weight=0.1,
        next_tier_a_policy_weight=0.25,
        prev_gen_holdout_mse=0.0568296855,
        min_reduction_fraction=0.25,
        min_coverage=0.5,
        seed=0,
    )


def test_summarize_reports_frontier_coverage_and_reduction():
    summary = summarize([_row(2.0, 0.5), _row(1.0, 0.25, tablebase_hits=6)])

    assert summary["cases"] == 2
    assert summary["supported_cases"] == 2
    assert summary["median_width_reduction"] == 1.125
    assert summary["median_width_reduction_fraction"] == 0.75
    assert summary["median_tablebase_frontier_coverage"] == 0.7
    assert summary["total_tablebase_frontier_hits"] == 14


def test_next_run_recommends_aux_generation_when_signal_is_strong():
    summary = summarize([_row(2.0, 0.25), _row(1.0, 0.4)])
    next_run = build_next_run(_args(), summary)

    assert next_run["status"] == "ready_for_aux_generation"
    assert "run_tier_a_targets.py" in next_run["target_generation_command"]
    assert "--limit 50000" in next_run["target_generation_command"]
    assert "--tier-a-weight 0.1" in next_run["training_command"]
    assert "--no-enforce-monotonicity" in next_run["training_command"]


def test_next_run_blocks_long_training_when_frontier_signal_is_weak():
    summary = summarize([_row(2.0, 1.8, tablebase_hits=1), _row(1.0, 0.95, tablebase_hits=1)])
    next_run = build_next_run(_args(), summary)

    assert next_run["status"] == "needs_more_frontier_signal"
    assert "Do not spend a long training run yet" in next_run["recommendation"]
