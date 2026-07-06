"""Fixed tactical audit pack for pre-training generation gates."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from environment.cfr.evaluator import LeafEvaluator
from environment.cfr.exact import solve_exact_finite_horizon
from environment.cfr.mcts import MCTSConfig, make_node, mcts_search
from environment.cfr.tactical_scenarios import TacticalScenario


class AuditGateError(RuntimeError):
    """Raised when an audit-pack pinned value drifts beyond threshold."""


@dataclass(frozen=True)
class AuditPackRecord:
    scenario_name: str
    category: str
    exact_value: float | None
    tablebase_value: float | None
    mcts_value: float
    mcts_strategy_dropper: list[float]
    mcts_strategy_checker: list[float]
    mcts_strategy_entropy_dropper: float
    mcts_strategy_entropy_checker: float
    principal_line: list[tuple[int, int]]
    root_visits: int
    cells_used: int
    mean_unresolved_probability: float
    legal_dropper_count: int
    legal_checker_count: int
    seed: int


def _primary_category(scenario: TacticalScenario) -> str:
    if not scenario.tags:
        return "uncategorized"
    for tag in scenario.tags:
        if tag != "holdout":
            return tag
    return scenario.tags[0]


def _normalized_entropy(strategy: np.ndarray) -> float:
    probs = np.asarray(strategy, dtype=np.float64)
    probs = probs[probs > 0.0]
    if probs.size <= 1:
        return 0.0
    entropy = float(-(probs * np.log(probs)).sum())
    return float(entropy / math.log(probs.size))


def _record_for(
    scenario: TacticalScenario,
    *,
    mcts_config: MCTSConfig,
    evaluator: LeafEvaluator,
    seed: int,
) -> AuditPackRecord:
    exact_value: float | None = None
    unresolved_probability = 0.0
    if scenario.expected_value is not None:
        exact_game = deepcopy(scenario.game)
        exact_result = solve_exact_finite_horizon(
            exact_game,
            scenario.half_round_horizon,
            scenario.config,
        )
        exact_value = float(exact_result.value_for_hal)
        unresolved_probability = float(exact_result.unresolved_probability)

    game = deepcopy(scenario.game)
    node = make_node(game, scenario.config, evaluator=evaluator)
    rng = np.random.default_rng(seed)
    result = mcts_search(
        game,
        mcts_config,
        evaluator,
        rng,
        scenario.config,
        subgame_resolve_at_critical=scenario.expected_value is not None,
    )

    return AuditPackRecord(
        scenario_name=scenario.name,
        category=_primary_category(scenario),
        exact_value=exact_value,
        tablebase_value=(
            float(scenario.expected_value) if scenario.expected_value is not None else None
        ),
        mcts_value=float(result.root_value_for_hal),
        mcts_strategy_dropper=[float(x) for x in result.root_strategy_dropper_avg],
        mcts_strategy_checker=[float(x) for x in result.root_strategy_checker_avg],
        mcts_strategy_entropy_dropper=_normalized_entropy(result.root_strategy_dropper_avg),
        mcts_strategy_entropy_checker=_normalized_entropy(result.root_strategy_checker_avg),
        principal_line=[(int(a.drop_time), int(a.check_time)) for a in result.principal_line],
        root_visits=int(result.root_visits),
        cells_used=int(result.cells_used),
        mean_unresolved_probability=unresolved_probability,
        legal_dropper_count=len(node.drop_seconds),
        legal_checker_count=len(node.check_seconds),
        seed=int(seed),
    )


def run_audit_pack(
    scenarios: Iterable[TacticalScenario],
    mcts_config: MCTSConfig,
    evaluator: LeafEvaluator,
    seeds: Iterable[int],
    *,
    output_path: str | Path,
) -> list[AuditPackRecord]:
    records: list[AuditPackRecord] = []
    for scenario in scenarios:
        for seed in seeds:
            records.append(
                _record_for(
                    scenario,
                    mcts_config=mcts_config,
                    evaluator=evaluator,
                    seed=int(seed),
                )
            )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        for record in records:
            f.write(json.dumps(asdict(record), sort_keys=True) + "\n")
    return records


def load_audit_pack(path: str | Path) -> list[AuditPackRecord]:
    records: list[AuditPackRecord] = []
    with Path(path).open() as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            data["principal_line"] = [tuple(pair) for pair in data["principal_line"]]
            records.append(AuditPackRecord(**data))
    return records


def audit_gate(records: Iterable[AuditPackRecord], *, max_drift: float = 0.05) -> None:
    failures: list[str] = []
    for record in records:
        pinned = record.tablebase_value
        if pinned is None:
            continue
        drift = abs(record.mcts_value - pinned)
        if drift > max_drift:
            failures.append(
                f"{record.scenario_name} seed={record.seed} drift={drift:.6f} "
                f"mcts={record.mcts_value:.6f} pinned={pinned:.6f}"
            )
    if failures:
        raise AuditGateError("; ".join(failures))
