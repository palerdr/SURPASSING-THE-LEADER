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


def depth_gate(
    reports: list[dict[str, Any]], *, cvar_alpha: float = 0.5
) -> dict[str, Any]:
    """Compare one checkpoint's resolve ladders across increasing max_depth.

    This is the G1 search-effectiveness gate: at a fixed evaluator, deeper
    full-width resolve must strictly improve the upper-tail (CVaR) exact
    saddle gap over the root pack without worsening the maximum.  The first
    measured ladder showed the raw median is zero-inflated — several roots
    sit at exactly zero gap at depth one, and deeper resolve spreads
    sub-0.03 gaps onto them while every materially wrong root improves — so
    the tail mean, this repository's existing selection vocabulary, is the
    gated aggregate and the median stays report-only.  Reports whose records
    do not carry ``lp_fallbacks`` are invalid, because an uncounted silent
    LP fallback could fake a healthy ladder.
    """

    if len(reports) < 2:
        raise ValueError("depth gate needs at least two ladder reports")
    checkpoints = {str(report["checkpoint"]) for report in reports}
    if len(checkpoints) != 1:
        raise ValueError("depth gate reports must share one checkpoint")

    depths: list[int] = []
    roots_reference: list[tuple[NTState, int]] | None = None
    per_depth: list[dict[str, Any]] = []
    for report in reports:
        raw_depth = report["config"]["mcts"].get("max_depth")
        if raw_depth is None:
            raise ValueError("depth gate reports must declare a finite max_depth")
        depth = int(raw_depth)
        network_records = [
            row for row in report["records"] if row["evaluator"] == "network"
        ]
        if not network_records:
            raise ValueError("depth gate reports need network-evaluator records")
        for row in network_records:
            if "lp_fallbacks" not in row:
                raise ValueError(
                    "ladder records lack lp_fallbacks; uncounted fallbacks are invalid"
                )
        roots = list(dict.fromkeys(_key(row) for row in network_records))
        if roots_reference is None:
            roots_reference = roots
        elif roots != roots_reference:
            raise ValueError("depth gate reports must share one root pack")
        budget = max(int(row["budget"]) for row in network_records)
        means = _root_means(report, budget)
        gaps = [means[key]["saddle_gap"] for key in roots]
        value_errors = [means[key]["value_error"] for key in roots]
        tail = max(1, int(np.ceil(cvar_alpha * len(gaps))))
        depths.append(depth)
        per_depth.append(
            {
                "max_depth": depth,
                "budget": budget,
                "cvar_gap": float(np.mean(np.sort(gaps)[-tail:])),
                "median_gap": float(np.median(gaps)),
                "max_gap": float(np.max(gaps)),
                "median_value_error": float(np.median(value_errors)),
                "max_value_error": float(np.max(value_errors)),
                "total_lp_fallbacks": int(
                    np.sum([row["lp_fallbacks"] for row in network_records])
                ),
                "roots": [
                    {
                        "state": list(key[0]),
                        "horizon": key[1],
                        "saddle_gap": means[key]["saddle_gap"],
                        "value_error": means[key]["value_error"],
                    }
                    for key in roots
                ],
            }
        )

    if depths != sorted(depths) or len(set(depths)) != len(depths):
        raise ValueError("depth gate reports must use strictly increasing depths")

    cvar_decreasing = all(
        per_depth[index + 1]["cvar_gap"] < per_depth[index]["cvar_gap"]
        for index in range(len(per_depth) - 1)
    )
    max_no_increase = all(
        per_depth[index + 1]["max_gap"] <= per_depth[index]["max_gap"] + 1e-12
        for index in range(len(per_depth) - 1)
    )
    gates = {
        "cvar_gap_strictly_decreasing": cvar_decreasing,
        "max_gap_no_increase": max_no_increase,
    }
    return {
        "schema_version": "dth-depth-gate-v1",
        "checkpoint": checkpoints.pop(),
        "depths": depths,
        "cvar_alpha": cvar_alpha,
        "per_depth": per_depth,
        "gates": gates,
        "recommendation": (
            "depth-effective" if all(gates.values()) else "depth-ineffective"
        ),
    }


def orientation_gate(
    report: dict[str, Any],
    *,
    max_ratio: float = 1.5,
    consistent_floor: float = 0.01,
) -> dict[str, Any]:
    """Compare learned gap quality across role-orientation mirror pairs.

    No algebraic mirror-value identity holds (verified against 43,189 exact
    pairs), so this is a generalization-balance audit, not a symmetry check:
    mirrored roots of the same difficulty class should not be learned to
    wildly different quality.  Pairs where both gaps are below
    ``consistent_floor`` are consistent regardless of ratio.
    """

    network_records = [
        row for row in report["records"] if row["evaluator"] == "network"
    ]
    if not network_records:
        raise ValueError("orientation gate needs network-evaluator records")
    for row in network_records:
        if "lp_fallbacks" not in row:
            raise ValueError(
                "ladder records lack lp_fallbacks; uncounted fallbacks are invalid"
            )
    budget = max(int(row["budget"]) for row in network_records)
    means = _root_means(report, budget)

    def mirror(state: NTState) -> NTState:
        return (state[2], state[3], state[0], state[1])

    pairs = []
    seen: set[tuple[NTState, int]] = set()
    for state, horizon in means:
        mirrored = (mirror(state), horizon)
        if state == mirror(state) or (state, horizon) in seen:
            continue
        if mirrored in means:
            seen.add((state, horizon))
            seen.add(mirrored)
            first = means[(state, horizon)]["saddle_gap"]
            second = means[mirrored]["saddle_gap"]
            low, high = sorted((first, second))
            consistent = high <= consistent_floor or (
                high <= max_ratio * max(low, 1e-9)
            )
            pairs.append(
                {
                    "state": list(state),
                    "mirror": list(mirror(state)),
                    "horizon": horizon,
                    "saddle_gaps": [first, second],
                    "ratio": (high / max(low, 1e-9)),
                    "consistent": consistent,
                }
            )
    if not pairs:
        raise ValueError("orientation gate found no mirror pairs in the root pack")
    gates = {"orientation_pairs_consistent": all(row["consistent"] for row in pairs)}
    return {
        "schema_version": "dth-orientation-gate-v1",
        "checkpoint": str(report["checkpoint"]),
        "budget": budget,
        "max_ratio": max_ratio,
        "consistent_floor": consistent_floor,
        "pairs": pairs,
        "gates": gates,
        "recommendation": (
            "orientation-consistent"
            if all(gates.values())
            else "orientation-inconsistent"
        ),
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


def run_depth_gate(report_paths: Iterable[str | Path], output: str | Path) -> None:
    reports = [
        json.loads(Path(path).read_text(encoding="utf-8")) for path in report_paths
    ]
    verdict = depth_gate(reports)
    verdict["reports"] = [str(path) for path in report_paths]
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    promotion = commands.add_parser("promotion")
    promotion.add_argument("--baseline", required=True)
    promotion.add_argument("--candidate", required=True)
    promotion.add_argument("--checkpoint", required=True)
    promotion.add_argument("--exact-targets", required=True)
    promotion.add_argument("--replay-a", required=True)
    promotion.add_argument("--replay-b", required=True)
    promotion.add_argument("--output", required=True)
    depth = commands.add_parser("depth-gate")
    depth.add_argument("--reports", nargs="+", required=True)
    depth.add_argument("--output", required=True)
    orientation = commands.add_parser("orientation-gate")
    orientation.add_argument("--report", required=True)
    orientation.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "depth-gate":
        run_depth_gate(args.reports, args.output)
        return
    if args.command == "orientation-gate":
        verdict = orientation_gate(
            json.loads(Path(args.report).read_text(encoding="utf-8"))
        )
        verdict["report"] = args.report
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return
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
