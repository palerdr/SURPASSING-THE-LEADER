"""Probe value-vs-policy search sensitivity at audited trace states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from hal.agent import SolverAgent
from training.trace_bootstrap import replay_trace_state
from training.trace_policy_probe import (
    CheckpointProbeSpec,
    _games_by_trajectory,
    _paired_divergence_by_hint,
    _policy_report,
    _public_state_summary,
    _roles_for_hint,
    _trace_action_probabilities,
)


PredictFn = Callable[[Any], tuple[float, np.ndarray, np.ndarray]]
AgentFactory = Callable[..., Any]


@dataclass(frozen=True)
class HybridSearchSpec:
    label: str
    value_source: CheckpointProbeSpec
    policy_source: CheckpointProbeSpec


def _default_predict_fns(
    baseline: CheckpointProbeSpec,
    candidate: CheckpointProbeSpec,
    device: str,
) -> dict[str, PredictFn]:
    from training.train_value_net import load_checkpoint, make_predict_fn

    return {
        baseline.label: make_predict_fn(load_checkpoint(baseline.checkpoint, device=device), device),
        candidate.label: make_predict_fn(
            load_checkpoint(candidate.checkpoint, device=device), device
        ),
    }


def _hybrid_specs(
    baseline: CheckpointProbeSpec,
    candidate: CheckpointProbeSpec,
) -> list[HybridSearchSpec]:
    return [
        HybridSearchSpec("baseline_full", baseline, baseline),
        HybridSearchSpec("candidate_full", candidate, candidate),
        HybridSearchSpec("candidate_value_baseline_policy", candidate, baseline),
        HybridSearchSpec("baseline_value_candidate_policy", baseline, candidate),
    ]


def _hybrid_evaluator(value_predict: PredictFn, policy_predict: PredictFn):
    def evaluate(game):
        value, _, _ = value_predict(game)
        _, dropper_policy, checker_policy = policy_predict(game)
        return (
            float(value),
            np.asarray(dropper_policy, dtype=np.float64),
            np.asarray(checker_policy, dtype=np.float64),
        )

    return evaluate


def _new_summary_maps(labels: list[str]) -> tuple[dict[str, float], dict[str, float]]:
    return (
        {label: 0.0 for label in labels if label != labels[0]},
        {label: 0.0 for label in labels if label != labels[0]},
    )


def probe_trace_search_sensitivity(
    *,
    trace_report: dict[str, Any],
    audit_report: dict[str, Any],
    baseline: CheckpointProbeSpec,
    candidate: CheckpointProbeSpec,
    iterations: int,
    seed: int = 0,
    device: str = "cpu",
    policy_ensemble_size: int = 1,
    policy_uniform_mix: float = 0.0,
    search_prior_uniform_mix: float = 0.0,
    resolve_at_critical: bool = False,
    resolve_horizon: int = 3,
    use_tablebase: bool = True,
    roles: tuple[str, ...] = ("trace",),
    top_k: int = 5,
    predict_fns: dict[str, PredictFn] | None = None,
    agent_factory: AgentFactory = SolverAgent,
) -> dict[str, Any]:
    """Run full and cross-wired value/policy search probes on target hints.

    The four default variants are:

    - baseline value + baseline policy,
    - candidate value + candidate policy,
    - candidate value + baseline policy,
    - baseline value + candidate policy.

    Comparing the two hybrids against the full baseline shows whether a
    regression is primarily value-head/search amplification or policy-prior drift.
    By default each hybrid evaluator is wrapped in the same pinned tablebase
    layer used by ``SolverAgent`` when it loads a checkpoint.
    """
    invalid_roles = [role for role in roles if role not in ("trace", "dropper", "checker")]
    if invalid_roles:
        raise ValueError(f"invalid roles: {invalid_roles}")

    predictors = predict_fns or _default_predict_fns(baseline, candidate, device)
    missing = [
        spec.label
        for spec in (baseline, candidate)
        if spec.label not in predictors
    ]
    if missing:
        raise ValueError(f"missing predict_fns for labels: {missing}")

    variants = _hybrid_specs(baseline, candidate)
    variant_labels = [variant.label for variant in variants]
    max_value_delta_by_variant, max_policy_tv_by_variant = _new_summary_maps(
        variant_labels
    )
    games = _games_by_trajectory(trace_report)
    divergences = _paired_divergence_by_hint(audit_report)

    agents = []
    for variant in variants:
        evaluator = _hybrid_evaluator(
            predictors[variant.value_source.label],
            predictors[variant.policy_source.label],
        )
        if use_tablebase:
            from environment.cfr.evaluator import TablebaseEvaluator

            evaluator = TablebaseEvaluator(evaluator)
        agents.append(
            (
                variant,
                agent_factory(
                    variant.value_source.checkpoint,
                    player_name="Hal",
                    iterations=iterations,
                    seed=seed,
                    evaluator=evaluator,
                    policy_ensemble_size=policy_ensemble_size,
                    policy_uniform_mix=policy_uniform_mix,
                    search_prior_uniform_mix=search_prior_uniform_mix,
                    resolve_at_critical=resolve_at_critical,
                    resolve_horizon=resolve_horizon,
                ),
            )
        )

    rows = []
    for hint in audit_report.get("target_hints", []):
        trajectory = str(hint.get("trajectory", "candidate"))
        key = (int(hint["seed"]), int(hint["game_index"]))
        row = games.get(trajectory, {}).get(key)
        if row is None:
            continue
        history_index = int(hint["history_index"])
        game = replay_trace_state(row, history_index)
        probe_roles = _roles_for_hint(hint, roles)
        divergence = divergences.get((key[0], key[1], history_index))
        variant_reports = []
        vectors_by_label: dict[str, dict[str, np.ndarray]] = {}

        for variant, agent in agents:
            search_result = agent.search(game)
            policies: dict[str, dict[str, Any]] = {}
            vectors_by_label[variant.label] = {}
            for role in probe_roles:
                policy = _policy_report(agent, game, role, top_k)
                vectors_by_label[variant.label][role] = policy.pop("vector")
                policy["trace_actions"] = _trace_action_probabilities(
                    vectors_by_label[variant.label][role],
                    divergence,
                    role,
                )
                policies[role] = policy
            variant_reports.append(
                {
                    "label": variant.label,
                    "value_source": variant.value_source.label,
                    "policy_source": variant.policy_source.label,
                    "root_value_for_hal": float(search_result.root_value_for_hal),
                    "root_visits": int(search_result.root_visits),
                    "cells_used": int(search_result.cells_used),
                    "policies": policies,
                }
            )

        baseline_report = variant_reports[0]
        baseline_label = str(baseline_report["label"])
        deltas = []
        for report in variant_reports[1:]:
            label = str(report["label"])
            value_delta = float(report["root_value_for_hal"]) - float(
                baseline_report["root_value_for_hal"]
            )
            max_value_delta_by_variant[label] = max(
                max_value_delta_by_variant[label],
                abs(value_delta),
            )
            policy_deltas = {}
            for role in probe_roles:
                baseline_vector = vectors_by_label[baseline_label][role]
                vector = vectors_by_label[label][role]
                tv = float(0.5 * np.abs(vector - baseline_vector).sum())
                max_policy_tv_by_variant[label] = max(
                    max_policy_tv_by_variant[label],
                    tv,
                )
                policy_deltas[role] = {
                    "tv": tv,
                    "entropy_delta": float(
                        report["policies"][role]["entropy"]
                        - baseline_report["policies"][role]["entropy"]
                    ),
                }
            deltas.append(
                {
                    "label": label,
                    "value_delta": value_delta,
                    "policies": policy_deltas,
                }
            )

        rows.append(
            {
                "hint": hint,
                "state": _public_state_summary(game),
                "trace_divergence": divergence,
                "roles": list(probe_roles),
                "variants": variant_reports,
                "deltas_vs_baseline": deltas,
            }
        )

    return {
        "config": {
            "iterations": int(iterations),
            "seed": int(seed),
            "device": device,
            "policy_ensemble_size": int(policy_ensemble_size),
            "policy_uniform_mix": float(policy_uniform_mix),
            "search_prior_uniform_mix": float(search_prior_uniform_mix),
            "resolve_at_critical": bool(resolve_at_critical),
            "resolve_horizon": int(resolve_horizon),
            "use_tablebase": bool(use_tablebase),
            "roles": list(roles),
            "top_k": int(top_k),
        },
        "sources": {
            "baseline": baseline.__dict__,
            "candidate": candidate.__dict__,
        },
        "variants": [
            {
                "label": variant.label,
                "value_source": variant.value_source.label,
                "policy_source": variant.policy_source.label,
            }
            for variant in variants
        ],
        "summary": {
            "states": len(rows),
            "max_abs_value_delta_by_variant": max_value_delta_by_variant,
            "max_policy_tv_by_variant": max_policy_tv_by_variant,
        },
        "rows": rows,
    }
