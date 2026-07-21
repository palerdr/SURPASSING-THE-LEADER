"""Machine-readable promotion gate for one frozen DTH readiness ladder."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from dth.cfr import solve_matrix_cfr_plus
from dth.mcts import ExactTargetStore, payoff_from_exact_targets
from dth.network import DTHNetworkConfig, DTHPolicyValueNet
from dth.self_play import validate_replay
from dth.solver import NTState, solve_matrix
from dth.torch_cfr import solve_matrix_cfr_plus_torch
from dth.train import approximate_payoff_from_network


def _key(row: dict[str, Any]) -> tuple[NTState, int]:
    return tuple(int(value) for value in row["state"]), int(row["horizon"])


def _root_means(
    report: dict[str, Any], budget: int
) -> dict[tuple[NTState, int], dict[str, float]]:
    grouped: dict[tuple[NTState, int], list[dict[str, Any]]] = defaultdict(list)
    for row in report["records"]:
        if int(row["budget"]) == budget and row["evaluator"] == "network":
            grouped[_key(row)].append(row)
    return {
        key: {
            "saddle_gap": float(np.mean([row["saddle_gap"] for row in rows])),
            "value_error": float(np.mean([row["value_error"] for row in rows])),
            "gap_seed_std": float(np.std([row["saddle_gap"] for row in rows])),
            "value_seed_std": float(np.std([row["mcts_value"] for row in rows])),
            "seeds": len({int(row["seed"]) for row in rows}),
        }
        for key, rows in grouped.items()
    }


def compare_ladders(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    budgets = sorted({int(row["budget"]) for row in baseline["records"]})
    candidate_budgets = sorted({int(row["budget"]) for row in candidate["records"]})
    if budgets != candidate_budgets:
        raise ValueError("baseline and candidate budgets differ")
    roots = []
    for row in baseline["records"]:
        key = _key(row)
        if key not in roots:
            roots.append(key)
    if len(roots) < 4:
        raise ValueError("readiness ladder needs three anchors and evaluation roots")

    by_budget: dict[str, Any] = {}
    for budget in budgets:
        base = _root_means(baseline, budget)
        cand = _root_means(candidate, budget)
        if base.keys() != cand.keys() or set(roots) != base.keys():
            raise ValueError("baseline and candidate roots differ")
        anchor_rows = roots[:3]
        eval_rows = roots[3:]
        base_anchor_worst = max(base[key]["saddle_gap"] for key in anchor_rows)
        cand_anchor_worst = max(cand[key]["saddle_gap"] for key in anchor_rows)
        base_eval_gaps = [base[key]["saddle_gap"] for key in eval_rows]
        cand_eval_gaps = [cand[key]["saddle_gap"] for key in eval_rows]
        by_budget[str(budget)] = {
            "anchors": [
                {
                    "state": list(key[0]),
                    "horizon": key[1],
                    "baseline": base[key],
                    "candidate": cand[key],
                    "gap_delta": cand[key]["saddle_gap"] - base[key]["saddle_gap"],
                    "value_error_delta": cand[key]["value_error"]
                    - base[key]["value_error"],
                }
                for key in anchor_rows
            ],
            "evaluation": [
                {
                    "state": list(key[0]),
                    "horizon": key[1],
                    "baseline": base[key],
                    "candidate": cand[key],
                }
                for key in eval_rows
            ],
            "anchor_worst_gap": {
                "baseline": base_anchor_worst,
                "candidate": cand_anchor_worst,
                "improvement_fraction": (base_anchor_worst - cand_anchor_worst)
                / base_anchor_worst,
            },
            "evaluation_median_gap": {
                "baseline": float(np.median(base_eval_gaps)),
                "candidate": float(np.median(cand_eval_gaps)),
                "improvement_fraction": float(
                    (np.median(base_eval_gaps) - np.median(cand_eval_gaps))
                    / np.median(base_eval_gaps)
                ),
            },
            "evaluation_max_gap": {
                "baseline": max(base_eval_gaps),
                "candidate": max(cand_eval_gaps),
            },
        }

    primary = by_budget[str(max(budgets))]
    anchors = primary["anchors"]
    seed_std = max(
        max(row["candidate"]["gap_seed_std"], row["candidate"]["value_seed_std"])
        for group in (primary["anchors"], primary["evaluation"])
        for row in group
    )
    gates = {
        "anchor_worst_gap_improvement": primary["anchor_worst_gap"][
            "improvement_fraction"
        ]
        >= 0.10,
        "anchor_gap_regression": max(row["gap_delta"] for row in anchors) <= 0.01,
        "anchor_value_error_regression": max(
            row["value_error_delta"] for row in anchors
        )
        <= 0.01,
        "evaluation_median_gap_improvement": primary["evaluation_median_gap"][
            "improvement_fraction"
        ]
        >= 0.10,
        "evaluation_max_gap_no_regression": primary["evaluation_max_gap"]["candidate"]
        <= primary["evaluation_max_gap"]["baseline"] + 1e-12,
        "seed_stable": seed_std <= 1e-6
        and all(
            row["candidate"]["seeds"] == 3
            for group in (primary["anchors"], primary["evaluation"])
            for row in group
        ),
    }
    return {
        "budgets": budgets,
        "primary_budget": max(budgets),
        "by_budget": by_budget,
        "gates": gates,
    }


def _exact_gap(matrix: np.ndarray, drop: np.ndarray, check: np.ndarray) -> float:
    return max(0.0, float(np.max(matrix @ check) - np.min(matrix.T @ drop)))


def audit_solvers(
    checkpoint: str | Path,
    exact_targets: str | Path,
    roots: Iterable[tuple[NTState, int]],
) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = DTHPolicyValueNet(DTHNetworkConfig(**payload["model_config"]))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    targets = ExactTargetStore.load(exact_targets)
    records = []
    for state, horizon in roots:
        approximate = approximate_payoff_from_network(
            model, state, horizon, device=torch.device("cpu")
        )
        exact = payoff_from_exact_targets(state, horizon, targets)
        numpy_solution = solve_matrix_cfr_plus(approximate, iterations=64)
        torch_solution = solve_matrix_cfr_plus_torch(
            torch.as_tensor(approximate, dtype=torch.float64), iterations=64
        )
        _, lp_drop, lp_check = solve_matrix(approximate)
        torch_drop = torch_solution.drop_policy.numpy()
        torch_check = torch_solution.check_policy.numpy()
        records.append(
            {
                "state": list(state),
                "horizon": horizon,
                "torch_numpy_policy_max_abs": max(
                    float(np.max(np.abs(torch_drop - numpy_solution.drop_policy))),
                    float(np.max(np.abs(torch_check - numpy_solution.check_policy))),
                ),
                "induced_exact_saddle_gap": {
                    "torch_cfr_plus_64": _exact_gap(exact, torch_drop, torch_check),
                    "numpy_cfr_plus_64": _exact_gap(
                        exact, numpy_solution.drop_policy, numpy_solution.check_policy
                    ),
                    "lp": _exact_gap(exact, lp_drop, lp_check),
                },
            }
        )
    parity = max(row["torch_numpy_policy_max_abs"] for row in records)
    return {
        "iterations": 64,
        "roots": records,
        "torch_numpy_policy_max_abs": parity,
        "passed": parity <= 1e-10,
    }


def audit_replay(first: str | Path, second: str | Path) -> dict[str, Any]:
    manifests = [
        json.loads(Path(path).read_text(encoding="utf-8")) for path in (first, second)
    ]
    validations = []
    for path in (first, second):
        with np.load(Path(path).with_suffix(".npz"), allow_pickle=False) as artifact:
            validations.append(
                validate_replay({name: artifact[name] for name in artifact.files})
            )
    deterministic = (
        manifests[0]["trajectory_sha256"] == manifests[1]["trajectory_sha256"]
    )
    return {
        "deterministic": deterministic,
        "trajectory_sha256": manifests[0]["trajectory_sha256"],
        "validations": validations,
        "passed": deterministic and validations[0] == validations[1],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--exact-targets", required=True)
    parser.add_argument("--replay-a", required=True)
    parser.add_argument("--replay-b", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    candidate = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    comparison = compare_ladders(baseline, candidate)
    roots = [_key(row) for row in baseline["records"]]
    roots = list(dict.fromkeys(roots))
    solvers = audit_solvers(args.checkpoint, args.exact_targets, roots)
    replay = audit_replay(args.replay_a, args.replay_b)
    gates = {
        **comparison["gates"],
        "solver_compatible": solvers["passed"],
        "replay_valid_deterministic": replay["passed"],
    }
    report = {
        "schema_version": "dth-self-play-readiness-v1",
        "baseline": args.baseline,
        "candidate": args.candidate,
        "checkpoint": args.checkpoint,
        "comparison": comparison,
        "solver_audit": solvers,
        "replay_audit": replay,
        "promotion_gates": gates,
        "recommendation": "promote" if all(gates.values()) else "no-promote",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
