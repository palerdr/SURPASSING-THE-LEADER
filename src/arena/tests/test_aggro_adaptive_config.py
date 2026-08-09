from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

from arena.policies.evaluate_aggro_hal_adaptive import PROMOTION_THRESHOLDS
from arena.policies.train_aggro_hal import (
    MEMORY_SESSION_COLLECTOR_IDENTITY,
    MemoryCurriculumSessionCollector,
    _configured_experiment_binding,
    _configured_initial_checkpoint_binding,
    configured_training_session_collector,
)

ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = ROOT / "src" / "arena" / "config"
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _json(name: str) -> dict[str, object]:
    return json.loads((CONFIG_DIR / name).read_text(encoding="utf-8"))


def test_corrected_v1_is_frozen_as_tactical_not_adaptive() -> None:
    freeze = _json("aggro_hal_tactical_baseline_v1.json")

    assert freeze["schema_version"] == "arena-aggro-hal-tactical-baseline-freeze-v1"
    assert freeze["immutable"] is True
    assert freeze["artifact"]["sha256"] == (
        "925cb9b63e88911e711e911ab5faa3918c27bf13c2731de1233c4ffe762f49d4"
    )
    assert (
        freeze["evidence"]["memory_ablations"]["long_horizon_adaptation_supported"]
        is False
    )
    assert freeze["immutability"]["generated_artifacts_remain_gitignored"] is True
    assert SHA256.fullmatch(
        freeze["bindings"]["training_config"]["canonical_yaml_sha256"]
    )
    for binding in freeze["bindings"].values():
        assert SHA256.fullmatch(binding["sha256"])


def test_adaptive_goal_selects_target_only_memory_training_and_strict_gate() -> None:
    path = CONFIG_DIR / "aggro_hal_adaptive_memory_v1.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert config["training"]["ppo_updates"] == 0
    experiment = config["experiment"]
    assert experiment["schema_version"] == (
        "arena-aggro-hal-adaptive-memory-experiment-v1"
    )
    assert experiment["ppo_locked"] is True
    assert config["session_collector"] == {
        "schema_version": "arena-aggro-hal-session-collector-config-v1",
        "type": "memory-necessity",
        "split": "train",
    }
    gate = config["promotion_gate"]
    assert gate["held_out_cover_games"] == PROMOTION_THRESHOLDS.minimum_cover_games
    assert gate["per_role_mode_no_pooled_rescue"] == {
        "normalized_payoff_gain_lower_95_strictly_above": (
            PROMOTION_THRESHOLDS.normalized_payoff_gain
        ),
        "forecast_nll_gain_lower_95_strictly_above_nats": (
            PROMOTION_THRESHOLDS.nll_gain_nats
        ),
    }
    assert gate["expansion"]["pass"] == (
        "require_tactical_regression_before_capped_30_ppo_updates"
    )

    collector = configured_training_session_collector(path)
    assert isinstance(collector, MemoryCurriculumSessionCollector)
    assert (
        collector.checkpoint_binding()["identity"] == MEMORY_SESSION_COLLECTOR_IDENTITY
    )
    assert _configured_initial_checkpoint_binding(path) == {
        "path": "outputs/aggro-hal-v1/corrected-v1/checkpoint.pt",
        "sha256": ("925cb9b63e88911e711e911ab5faa3918c27bf13c2731de1233c4ffe762f49d4"),
    }
    binding = _configured_experiment_binding(path)
    assert binding is not None
    assert binding["schema_version"] == (
        "arena-aggro-hal-adaptive-memory-experiment-binding-v1"
    )
    assert binding["goal_manifest"] == {
        "path": "src/arena/config/aggro_hal_adaptive_exploitation_goal_v1.json",
        "canonical_json_sha256": (
            "44d09c9cd4e1bc765a5a095ab2f7fdf62ce9f89061f51a4a18b2c59cc80b66ec"
        ),
    }

    goal = _json("aggro_hal_adaptive_exploitation_goal_v1.json")
    assert goal["per_cell_gate"]["pooled_rescue_allowed"] is False
    assert goal["split_policy"]["sealed_audit_reservation"] == {
        "registered_example_seeds": "20000..20031",
        "session_seed_offset": 300000000,
        "cover_delays": [8],
        "status": "reserved_not_implemented_not_opened",
        "rule": (
            "Implement and hash-bind the audit generator before opening it; "
            "validation results may not change its seeds, targets, controls, or "
            "thresholds."
        ),
    }
