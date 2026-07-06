import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.run_pattern_reader_guard_targets as guard_cli
from environment.cfr.mcts import MCTSResult
from training.pattern_reader_guard import (
    generate_pattern_reader_guard_records,
    pattern_reader_active_for_state,
)
from training.tournament import _default_starting_game
from training.value_targets import SOURCE_POLICY_GUARD


class _FixedProbeAgent:
    def choose_action(self, game, role, turn_duration):
        return 4 if role == "dropper" else min(60, turn_duration)


class _LabelAgent:
    iterations = 8
    policy_ensemble_size = 1
    policy_uniform_mix = 0.0

    def search(self, game):
        return MCTSResult(
            root_strategy_dropper=np.array([1.0]),
            root_strategy_checker=np.array([1.0]),
            root_value_for_hal=0.125,
            root_visits=1,
            principal_line=[],
            cells_used=1,
            root_drop_seconds=(1,),
            root_check_seconds=(1,),
            root_strategy_dropper_avg=np.array([1.0]),
            root_strategy_checker_avg=np.array([1.0]),
        )

    def policy(self, game, role):
        return (1,), np.array([1.0])


def test_pattern_reader_active_for_repeated_hal_drop():
    game = _default_starting_game(0)
    for _ in range(3):
        game.resolve_half_round(4, 60, survived_outcome=True)
        game.resolve_half_round(30, 60, survived_outcome=True)

    assert pattern_reader_active_for_state(game)


def test_generate_pattern_reader_guard_records_labels_trace_states():
    records, summary = generate_pattern_reader_guard_records(
        probe_agent=_FixedProbeAgent(),
        label_agent=_LabelAgent(),
        label_checkpoint="champion.pt",
        seeds=[0],
        games_per_seed=1,
        max_states=2,
        active_only=False,
        min_history=0,
    )

    assert summary.records == len(records) == 2
    assert summary.games == 1
    assert all(record.target.source == SOURCE_POLICY_GUARD for record in records)
    assert records[0].metadata.scenario.startswith("pattern_reader_s0_g0_h")
    assert records[0].target.value == 0.125


def test_pattern_reader_guard_cli_passes_resolve_settings_to_label_agent(
    monkeypatch,
    tmp_path,
):
    constructed = []

    class FakeAgent:
        def __init__(self, checkpoint, **kwargs):
            constructed.append({"checkpoint": checkpoint, **kwargs})

    def fake_generate_pattern_reader_guard_records(**_kwargs):
        return [], SimpleNamespace(records=0, games=0, hal_wins=0, baku_wins=0, draws=0)

    monkeypatch.setattr(guard_cli, "SolverAgent", FakeAgent)
    monkeypatch.setattr(
        guard_cli,
        "generate_pattern_reader_guard_records",
        fake_generate_pattern_reader_guard_records,
    )
    monkeypatch.setattr(guard_cli, "save_policy_guard_records", lambda _records, _out: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_pattern_reader_guard_targets.py",
            "--probe-checkpoint",
            "probe.pt",
            "--label-checkpoint",
            "label.pt",
            "--resolve-at-critical",
            "--resolve-horizon",
            "3",
            "--out",
            str(tmp_path / "guard.npz"),
        ],
    )

    assert guard_cli.main() == 0
    assert constructed[0]["checkpoint"] == "probe.pt"
    assert constructed[1]["checkpoint"] == "label.pt"
    assert constructed[0]["resolve_at_critical"] is True
    assert constructed[0]["resolve_horizon"] == 3
    assert constructed[1]["resolve_at_critical"] is True
    assert constructed[1]["resolve_horizon"] == 3
