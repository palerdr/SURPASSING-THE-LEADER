import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.getcwd())

from stl.commands.next_run import build_next_run


def _args(**overrides):
    defaults = dict(
        base_targets="checkpoints/base_targets.npz",
        held_out_targets="checkpoints/holdout.npz",
        next_target_count=150_000,
        next_target_max_width=0.01,
        next_seed=3,
        epochs=5,
        learning_rate=3e-6,
        tier_a_weight=5e-4,
        critical_bootstrap_max_states=24,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_next_run_excludes_d1_after_live_strength_rejection():
    accepted = {
        "status": "accepted",
        "checkpoint": "checkpoints/current/best.pt",
        "held_out_overall_mse": 0.0559,
    }
    rejected = [
        {
            "checkpoint": "checkpoints/gen_tier_a_d1hal/best.pt",
            "reasons": ["ladder wins_delta -11 < required 1"],
            "ladder_wins_delta": -11,
            "exploitability_verdicts": {"candidate_certified_worse": 1},
        }
    ]

    next_run = build_next_run(
        _args(),
        accepted,
        rejected,
        {"wins_delta": 0},
    )

    assert next_run["status"] == "ready_for_d0_tier_a_scaleup"
    assert "--death-filter d0" in next_run["target_generation_command"]
    assert "--no-include-d1" in next_run["target_generation_command"]
    assert "--tier-a-policy-weight 0.0" in next_run["training_command"]
    assert any("d1-Hal follow-ups" in reason for reason in next_run["rationale"])


def test_next_run_keeps_runtime_diagnostic_when_runtime_ladder_is_neutral():
    accepted = {
        "status": "accepted",
        "checkpoint": "checkpoints/current/best.pt",
        "held_out_overall_mse": 0.0559,
    }

    next_run = build_next_run(
        _args(),
        accepted,
        [],
        {"wins_delta": 0},
    )

    assert next_run["status"] == "ready_for_tier_a_scaleup"
    assert any("runtime leaf replacement is neutral" in reason for reason in next_run["rationale"])


def test_next_run_switches_to_mcts_refresh_after_d0_append_rejection():
    accepted = {
        "status": "accepted",
        "checkpoint": "checkpoints/current/best.pt",
        "held_out_overall_mse": 0.0559,
    }
    rejected = [
        {
            "checkpoint": "checkpoints/gen_tier_a_aux_plus_d0_150k/best.pt",
            "reasons": ["ladder wins_delta -10 < required 1"],
            "ladder_wins_delta": -10,
            "exploitability_verdicts": {},
        }
    ]

    next_run = build_next_run(_args(), accepted, rejected, {"wins_delta": 0})

    assert next_run["status"] == "ready_for_mcts_bootstrap_refresh"
    assert next_run["target_generation_command"] is None
    assert "scripts/run_gen_iteration.py" in next_run["training_command"]
    assert "--init-checkpoint checkpoints/current/best.pt" in next_run["training_command"]
    assert "--iterations 300" in next_run["training_command"]
    assert "--learning-rate 3e-06" in next_run["training_command"]
    assert any(
        "compare_checkpoints_policy_drift.py" in command and "--iterations 200" in command
        for command in next_run["post_training_commands"]
    )


def test_next_run_requires_pattern_reader_hardening_after_mcts_rejection():
    accepted = {
        "status": "accepted",
        "checkpoint": "checkpoints/current/best.pt",
        "held_out_overall_mse": 0.0559,
    }
    rejected = [
        {
            "checkpoint": "checkpoints/gen_current_mcts300_tiera50k/best.pt",
            "reasons": ["ladder wins_delta -10 < required 1"],
            "ladder_wins_delta": -10,
            "opponent_wins_delta": {"pattern_reader": -12},
            "exploitability_verdicts": {"candidate_certified_worse": 3},
        }
    ]

    next_run = build_next_run(_args(), accepted, rejected, {"wins_delta": 0})

    assert next_run["status"] == "needs_pattern_reader_hardening_generation"
    assert "--subgame-resolve-at-critical" in next_run["training_command"]
    assert "--subgame-resolve-horizon 1" in next_run["training_command"]
    assert "--subgame-resolve-cfr-iters 2000" in next_run["training_command"]
    assert "--bootstrap-critical-only" in next_run["training_command"]
    assert "--bootstrap-max-states 24" in next_run["training_command"]
    assert "--learning-rate 3e-06" in next_run["training_command"]
    assert any("--opponents pattern_reader" in command for command in next_run["post_training_commands"])
    assert any(
        "compare_checkpoints_policy_drift.py" in command and "--iterations 200" in command
        for command in next_run["post_training_commands"]
    )


def test_next_run_blocks_without_accepted_promotion_report():
    next_run = build_next_run(_args(), {"status": "rejected"}, [], None)

    assert next_run["status"] == "blocked_until_promoted_checkpoint"
    assert next_run["target_generation_command"] is None
