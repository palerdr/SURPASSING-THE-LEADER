import os
import sys

import pytest

sys.path.insert(0, os.getcwd())

from stl.solver.search import TerminalOnlyEvaluator
from stl.solver.search import MCTSConfig
from stl.solver.tablebase import get_scenario
from stl.learning.audit import (
    AuditGateError,
    AuditPackRecord,
    audit_gate,
    load_audit_pack,
    run_audit_pack,
)


def _config() -> MCTSConfig:
    return MCTSConfig(iterations=5, exploration_c=1.0)


def test_audit_pack_record_carries_all_fields(tmp_path):
    scenario = get_scenario("forced_baku_overflow_death")
    records = run_audit_pack(
        [scenario],
        _config(),
        TerminalOnlyEvaluator(),
        [0],
        output_path=tmp_path / "audit.jsonl",
    )

    assert len(records) == 1
    record = records[0]
    assert isinstance(record, AuditPackRecord)
    assert record.scenario_name == scenario.name
    assert record.tablebase_value == pytest.approx(1.0)
    assert record.legal_dropper_count > 0
    assert record.legal_checker_count > 0


def test_audit_pack_run_is_deterministic_under_seeded_rng(tmp_path):
    scenario = get_scenario("forced_baku_overflow_death")
    a = run_audit_pack([scenario], _config(), TerminalOnlyEvaluator(), [7], output_path=tmp_path / "a.jsonl")
    b = run_audit_pack([scenario], _config(), TerminalOnlyEvaluator(), [7], output_path=tmp_path / "b.jsonl")

    assert a == b


def test_audit_gate_raises_on_pinned_value_drift_above_threshold():
    record = AuditPackRecord(
        scenario_name="broken",
        category="forced_terminal",
        exact_value=1.0,
        tablebase_value=1.0,
        mcts_value=0.0,
        mcts_strategy_dropper=[],
        mcts_strategy_checker=[],
        mcts_strategy_entropy_dropper=0.0,
        mcts_strategy_entropy_checker=0.0,
        principal_line=[],
        root_visits=0,
        cells_used=0,
        mean_unresolved_probability=0.0,
        legal_dropper_count=0,
        legal_checker_count=0,
        seed=0,
    )

    with pytest.raises(AuditGateError):
        audit_gate([record], max_drift=0.05)


def test_audit_gate_passes_when_mcts_matches_pinned_value(tmp_path):
    scenario = get_scenario("forced_hal_overflow_death")
    records = run_audit_pack(
        [scenario],
        _config(),
        TerminalOnlyEvaluator(),
        [0],
        output_path=tmp_path / "audit.jsonl",
    )

    audit_gate(records, max_drift=0.05)


def test_audit_pack_jsonl_round_trip(tmp_path):
    scenario = get_scenario("forced_baku_overflow_death")
    path = tmp_path / "audit.jsonl"
    records = run_audit_pack([scenario], _config(), TerminalOnlyEvaluator(), [3], output_path=path)

    assert load_audit_pack(path) == records


def test_holdout_scenarios_present_in_manual_audit_pack(tmp_path):
    scenario = get_scenario("baku_dropper_leap_window_alignment")
    records = run_audit_pack([scenario], _config(), TerminalOnlyEvaluator(), [0], output_path=tmp_path / "audit.jsonl")

    assert records[0].tablebase_value is None
    assert records[0].category == "forced_leap"


def test_audit_pack_strategy_entropy_in_unit_interval_after_normalization(tmp_path):
    scenario = get_scenario("forced_baku_overflow_death")
    records = run_audit_pack([scenario], _config(), TerminalOnlyEvaluator(), [0], output_path=tmp_path / "audit.jsonl")
    record = records[0]

    assert 0.0 <= record.mcts_strategy_entropy_dropper <= 1.0
    assert 0.0 <= record.mcts_strategy_entropy_checker <= 1.0
