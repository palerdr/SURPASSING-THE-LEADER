#!/usr/bin/env python3
"""Compare root mixed strategies between two SolverAgent checkpoints.

This is a cheap diagnostic before a full ladder: it answers where a candidate
changed the deployed search policy, by role and scenario. It does not replace
the promotion gate; it explains why a candidate may later pass or fail it.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hal.agent import DEFAULT_CHECKPOINT, SolverAgent
from scripts.compare_checkpoints_ladder import _json_safe
from scripts.run_exploitability import SCENARIOS
from scripts.run_tier_a_frontier_event import _split_csv


DEFAULT_SCENARIOS = "opening,postleap_230,postleap_d1_hal_120_230"


def _policy_vector(seconds: tuple[int, ...], probs: np.ndarray) -> np.ndarray:
    out = np.zeros(61, dtype=np.float64)
    for second, probability in zip(seconds, probs):
        out[int(second) - 1] = float(probability)
    total = float(out.sum())
    if total > 0.0:
        out /= total
    return out


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    mask = p > 0.0
    if not bool(mask.any()):
        return 0.0
    q_safe = np.clip(q[mask], 1e-12, 1.0)
    return float(np.sum(p[mask] * np.log(p[mask] / q_safe)))


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    return 0.5 * _kl_divergence(p, m) + 0.5 * _kl_divergence(q, m)


def _top_seconds(policy: np.ndarray, limit: int = 5) -> list[dict]:
    order = np.argsort(policy)[::-1]
    rows = []
    for idx in order[:limit]:
        probability = float(policy[idx])
        if probability <= 0.0:
            continue
        rows.append({"second": int(idx) + 1, "probability": probability})
    return rows


def policy_delta(champion: np.ndarray, candidate: np.ndarray) -> dict:
    diff = candidate - champion
    return {
        "tv": float(0.5 * np.abs(diff).sum()),
        "l1": float(np.abs(diff).sum()),
        "l_inf": float(np.abs(diff).max()) if diff.size else 0.0,
        "js": _js_divergence(champion, candidate),
        "champion_entropy": _entropy(champion),
        "candidate_entropy": _entropy(candidate),
    }


def _entropy(policy: np.ndarray) -> float:
    mask = policy > 0.0
    if not bool(mask.any()):
        return 0.0
    return float(-np.sum(policy[mask] * np.log(policy[mask])))


def run_case(args, *, scenario: str, seed: int, role: str) -> dict:
    game = SCENARIOS[scenario]()
    champion_agent = SolverAgent(
        args.champion_checkpoint,
        player_name=args.player,
        iterations=args.iterations,
        seed=seed,
        policy_ensemble_size=args.policy_ensemble_size,
        policy_uniform_mix=args.policy_uniform_mix,
        resolve_at_critical=args.resolve_at_critical,
        resolve_horizon=args.resolve_horizon,
    )
    candidate_agent = SolverAgent(
        args.candidate_checkpoint,
        player_name=args.player,
        iterations=args.iterations,
        seed=seed,
        policy_ensemble_size=args.policy_ensemble_size,
        policy_uniform_mix=args.policy_uniform_mix,
        resolve_at_critical=args.resolve_at_critical,
        resolve_horizon=args.resolve_horizon,
    )
    champion_seconds, champion_probs = champion_agent.policy(game, role)
    candidate_seconds, candidate_probs = candidate_agent.policy(game, role)
    champion_policy = _policy_vector(champion_seconds, champion_probs)
    candidate_policy = _policy_vector(candidate_seconds, candidate_probs)
    return {
        "seed": seed,
        "scenario": scenario,
        "role": role,
        "delta": policy_delta(champion_policy, candidate_policy),
        "champion_top": _top_seconds(champion_policy),
        "candidate_top": _top_seconds(candidate_policy),
    }


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {
            "cases": 0,
            "max_tv": 0.0,
            "mean_tv": 0.0,
            "max_js": 0.0,
            "mean_js": 0.0,
            "max_entropy_drop": 0.0,
            "mean_entropy_delta": 0.0,
            "min_champion_entropy": 0.0,
            "min_candidate_entropy": 0.0,
        }
    tvs = [float(row["delta"]["tv"]) for row in rows]
    jss = [float(row["delta"]["js"]) for row in rows]
    entropy_deltas = [
        float(row["delta"].get("candidate_entropy", 0.0))
        - float(row["delta"].get("champion_entropy", 0.0))
        for row in rows
    ]
    champion_entropies = [
        float(row["delta"].get("champion_entropy", 0.0))
        for row in rows
    ]
    candidate_entropies = [
        float(row["delta"].get("candidate_entropy", 0.0))
        for row in rows
    ]
    worst = max(rows, key=lambda row: row["delta"]["tv"])
    return {
        "cases": len(rows),
        "max_tv": float(max(tvs)),
        "mean_tv": float(sum(tvs) / len(tvs)),
        "max_js": float(max(jss)),
        "mean_js": float(sum(jss) / len(jss)),
        "max_entropy_drop": float(max(0.0, max(-delta for delta in entropy_deltas))),
        "mean_entropy_delta": float(sum(entropy_deltas) / len(entropy_deltas)),
        "min_champion_entropy": float(min(champion_entropies)),
        "min_candidate_entropy": float(min(candidate_entropies)),
        "worst_case": {
            "seed": worst["seed"],
            "scenario": worst["scenario"],
            "role": worst["role"],
            "tv": worst["delta"]["tv"],
            "js": worst["delta"]["js"],
        },
    }


def drift_gate_decision(
    summary: dict,
    *,
    max_tv: float | None = None,
    max_entropy_drop: float | None = None,
    min_candidate_entropy: float | None = None,
) -> dict | None:
    """Return an optional pass/fail verdict for policy-drift diagnostics."""
    if max_tv is None and max_entropy_drop is None and min_candidate_entropy is None:
        return None
    thresholds = {
        "max_tv": max_tv,
        "max_entropy_drop": max_entropy_drop,
        "min_candidate_entropy": min_candidate_entropy,
    }
    reasons: list[str] = []
    if max_tv is not None and float(summary.get("max_tv", 0.0)) > max_tv:
        reasons.append(
            f"max_tv {float(summary.get('max_tv', 0.0)):.6g} > {max_tv:.6g}"
        )
    if (
        max_entropy_drop is not None
        and float(summary.get("max_entropy_drop", 0.0)) > max_entropy_drop
    ):
        reasons.append(
            "max_entropy_drop "
            f"{float(summary.get('max_entropy_drop', 0.0)):.6g} > "
            f"{max_entropy_drop:.6g}"
        )
    if (
        min_candidate_entropy is not None
        and float(summary.get("min_candidate_entropy", 0.0)) < min_candidate_entropy
    ):
        reasons.append(
            "min_candidate_entropy "
            f"{float(summary.get('min_candidate_entropy', 0.0)):.6g} < "
            f"{min_candidate_entropy:.6g}"
        )
    return {
        "status": "failed" if reasons else "passed",
        "thresholds": thresholds,
        "reasons": reasons,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--champion-checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument("--player", default="Hal", choices=("Hal", "Baku"))
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument(
        "--policy-ensemble-size",
        type=int,
        default=1,
        help="Average this many independent state-seeded MCTS root policies per policy query.",
    )
    parser.add_argument(
        "--policy-uniform-mix",
        type=float,
        default=0.0,
        help="Blend each deployed root policy with this much uniform mass over its support.",
    )
    parser.add_argument(
        "--resolve-at-critical",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use SolverAgent critical-state subgame resolve during deployed search.",
    )
    parser.add_argument(
        "--resolve-horizon",
        type=int,
        default=3,
        help="Selective-solve horizon for --resolve-at-critical.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated seeds. Overrides --seed and aggregates all runs.",
    )
    parser.add_argument("--scenarios", default=DEFAULT_SCENARIOS)
    parser.add_argument("--roles", default="dropper,checker")
    parser.add_argument(
        "--out",
        default=str(Path("checkpoints") / "checkpoint_policy_drift" / "report.json"),
    )
    parser.add_argument(
        "--max-tv",
        type=float,
        default=None,
        help="Optional gate: fail when summary.max_tv exceeds this threshold.",
    )
    parser.add_argument(
        "--max-entropy-drop",
        type=float,
        default=None,
        help="Optional gate: fail when candidate entropy drops more than this.",
    )
    parser.add_argument(
        "--min-candidate-entropy",
        type=float,
        default=None,
        help="Optional gate: fail when any candidate policy entropy is below this.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    scenarios = _split_csv(args.scenarios)
    roles = _split_csv(args.roles)
    unknown = [name for name in scenarios if name not in SCENARIOS]
    if unknown:
        raise SystemExit(f"unknown scenarios: {', '.join(unknown)}")
    invalid_roles = [role for role in roles if role not in ("dropper", "checker")]
    if invalid_roles:
        raise SystemExit(f"invalid roles: {', '.join(invalid_roles)}")
    seeds = (
        [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
        if args.seeds
        else [args.seed]
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"Comparing champion={args.champion_checkpoint} candidate={args.candidate_checkpoint} "
        f"iterations={args.iterations} resolve_at_critical={args.resolve_at_critical} "
        f"resolve_horizon={args.resolve_horizon} seeds={seeds}"
    )
    print(f"Scenarios: {', '.join(scenarios)}  roles={', '.join(roles)}")
    start = time.time()
    rows = []
    for seed in seeds:
        for scenario in scenarios:
            for role in roles:
                print(f"  seed={seed} scenario={scenario} role={role}", flush=True)
                row = run_case(args, scenario=scenario, seed=seed, role=role)
                rows.append(row)
                print(f"    tv={row['delta']['tv']:.4f} js={row['delta']['js']:.4f}", flush=True)
    summary = summarize(rows)
    gate = drift_gate_decision(
        summary,
        max_tv=args.max_tv,
        max_entropy_drop=args.max_entropy_drop,
        min_candidate_entropy=args.min_candidate_entropy,
    )
    report = {
        "config": vars(args),
        "seeds": seeds,
        "scenarios": scenarios,
        "roles": roles,
        "rows": rows,
        "summary": summary,
        "gate": gate,
        "elapsed_seconds": round(time.time() - start, 2),
    }
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(_json_safe(report), fh, indent=2, allow_nan=False)
    print(f"Summary: {summary}")
    if gate is not None:
        print(f"Gate: {gate['status'].upper()}")
        for reason in gate["reasons"]:
            print(f"  {reason}")
    print(f"Report: {out_path}")
    return 1 if gate is not None and gate["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
