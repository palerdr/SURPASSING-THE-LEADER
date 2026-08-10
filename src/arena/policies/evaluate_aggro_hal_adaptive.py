"""Promotion gate for Aggro Hal's learned recurrent exploitation.

The evaluator uses the held-out memory-necessity curriculum.  For each role,
latent mode, and example seed it feeds one byte-identical target observation
with four memory interventions:

``correct``
    Hidden state produced by that mode's legal public history.
``swapped``
    Hidden state produced by the other latent mode's history.
``zero_target``
    A canonical zero state supplied directly at the target token.
``history_free``
    The complete history is evaluated token-by-token while resetting the
    hidden state before every token.  Its target output must agree with
    ``zero_target``; it is retained as a reset-path consistency check, not an
    independent capacity-matched model.

Promotion is deliberately conjunctive.  Every role-by-mode cell must clear
the predeclared lower confidence bounds against swapped and erased memory (and
the equivalent history-free control).  A pooled average can never rescue a
failed cell.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from torch import Tensor

from arena.policies.aggro_hal import (
    AggroHalNetwork,
    AggroHalNetworkOutput,
    dth_compatibility,
    load_checkpoint,
)
from arena.policies.aggro_memory_curriculum import (
    MemoryCurriculumCase,
    MemoryCurriculumSplit,
    MemoryCurriculumToken,
    Role,
    SplitName,
    build_memory_curriculum_case,
    memory_curriculum_config_payload,
    memory_curriculum_config_sha256,
    memory_curriculum_split,
)
from dth.agent import CompleteDTHAgent

ADAPTIVE_EVALUATION_SCHEMA = "arena-aggro-hal-adaptive-evaluation-v1"
ADAPTIVE_PROTOCOL_SCHEMA = "arena-aggro-hal-adaptive-protocol-v1"
DEFAULT_ARTIFACT_DIR = Path("src/dth/artifacts/complete_full_v1")
DEFAULT_BOOTSTRAP_REPLICATES = 5_000
DEFAULT_BOOTSTRAP_SEED = 20_260_809
HISTORY_FREE_EQUIVALENCE_ATOL = 1e-7

_ROLES: tuple[Role, ...] = ("dropper", "checker")
_MODES = ("a", "b")
_BASELINES = ("swapped", "zero_target", "history_free")


class StageGameProvider(Protocol):
    """The exact-solver capability required by the curriculum generator."""

    def stage_game(self, state: tuple[int, int, int, int]): ...


@dataclass(frozen=True, slots=True)
class AdaptivePromotionThresholds:
    """Checkpoint-independent thresholds frozen before model evaluation."""

    normalized_payoff_gain: float = 0.02
    nll_gain_nats: float = 0.01
    minimum_cover_games: int = 8

    def __post_init__(self) -> None:
        if self.normalized_payoff_gain <= 0.0 or self.nll_gain_nats <= 0.0:
            raise ValueError("adaptive promotion effect thresholds must be positive")
        if self.minimum_cover_games < 8:
            raise ValueError(
                "adaptive promotion requires a cover delay of at least eight"
            )


PROMOTION_THRESHOLDS = AdaptivePromotionThresholds()


@dataclass(frozen=True, slots=True)
class _NetworkSnapshot:
    policy: np.ndarray
    opponent_policy: np.ndarray
    direct_weight: float


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def adaptive_memory_protocol(
    split: SplitName | MemoryCurriculumSplit = "validation",
    *,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, object]:
    """Return the checkpoint-independent adaptive promotion commitment."""

    spec = memory_curriculum_split(split) if isinstance(split, str) else split
    if bootstrap_replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    return {
        "schema_version": ADAPTIVE_PROTOCOL_SCHEMA,
        "split": spec.name,
        "curriculum_config_sha256": memory_curriculum_config_sha256(spec),
        "curriculum_config": memory_curriculum_config_payload(spec),
        "example_seeds": list(spec.example_seeds),
        "roles": list(_ROLES),
        "modes": list(_MODES),
        "interventions": {
            "correct": "target token with same-mode recurrent prefix state",
            "swapped": "same target token with other-mode recurrent prefix state",
            "zero_target": "same target token with canonical zero hidden state",
            "history_free": (
                "reset-path consistency check: full arm evaluated token-by-token "
                "with hidden reset before every token; target must equal zero_target"
            ),
        },
        "bootstrap": {
            "unit": "example_seed",
            "replicates": int(bootstrap_replicates),
            "seed": int(bootstrap_seed),
            "interval": "two-sided percentile 95%",
        },
        "promotion": {
            "eligible_split": "validation",
            "thresholds": asdict(PROMOTION_THRESHOLDS),
            "strict_inequality": True,
            "required_comparisons": [
                "correct_vs_swapped",
                "correct_vs_zero_target",
                "correct_vs_history_free",
            ],
            "required_cells": [f"{role}/{mode}" for role in _ROLES for mode in _MODES],
            "complete_registered_split_required": True,
            "target_byte_identity_required": True,
            "history_free_zero_target_equivalence_required": True,
            "history_free_is_independent_capacity_control": False,
            "pooled_rescue_allowed": False,
        },
    }


def _model_device_dtype(model: AggroHalNetwork) -> tuple[torch.device, torch.dtype]:
    parameter = next(model.parameters())
    return parameter.device, parameter.dtype


def _forward_tokens(
    model: AggroHalNetwork,
    tokens: Sequence[MemoryCurriculumToken],
    hidden_state: Tensor | None = None,
) -> AggroHalNetworkOutput:
    if not tokens:
        raise ValueError("adaptive evaluation requires at least one network token")
    device, dtype = _model_device_dtype(model)

    def floating(values: np.ndarray) -> Tensor:
        return torch.as_tensor(values, dtype=dtype, device=device).unsqueeze(0)

    features = floating(np.stack([token.features for token in tokens]))
    matrices = floating(np.stack([token.stage_matrix for token in tokens]))
    exact = floating(np.stack([token.exact_policy for token in tokens]))
    roles = torch.as_tensor(
        [[token.role_is_dropper for token in tokens]],
        dtype=torch.bool,
        device=device,
    )
    legal = torch.as_tensor(
        np.stack([token.legal_mask for token in tokens])[None, ...],
        dtype=torch.bool,
        device=device,
    )
    return model(features, matrices, exact, roles, legal, hidden_state)


def _snapshot(output: AggroHalNetworkOutput) -> _NetworkSnapshot:
    return _NetworkSnapshot(
        policy=output.policy[0, -1].detach().cpu().numpy().astype(np.float64),
        opponent_policy=(
            output.opponent_policy[0, -1].detach().cpu().numpy().astype(np.float64)
        ),
        direct_weight=float(output.direct_weight[0, -1].detach().cpu()),
    )


def _history_free_output(
    model: AggroHalNetwork,
    tokens: Sequence[MemoryCurriculumToken],
) -> AggroHalNetworkOutput:
    """Evaluate an arm while preventing all recurrent carry between tokens."""

    output: AggroHalNetworkOutput | None = None
    device, dtype = _model_device_dtype(model)
    for token in tokens:
        zero = model.initial_hidden(1, device=device, dtype=dtype)
        output = _forward_tokens(model, (token,), zero)
    if output is None:  # guarded by the curriculum, retained for type narrowing
        raise ValueError("history-free evaluation requires a nonempty arm")
    return output


def _score(
    snapshot: _NetworkSnapshot,
    *,
    truth: np.ndarray,
    action_values: np.ndarray,
    best_response_action: int,
) -> dict[str, float | int]:
    tiny = 1e-12
    expected_payoff = float(np.dot(snapshot.policy, action_values))
    nll = float(-np.sum(truth * np.log(np.clip(snapshot.opponent_policy, tiny, 1.0))))
    return {
        "expected_payoff": expected_payoff,
        "opponent_nll_nats": nll,
        "best_response_action": int(best_response_action),
        "best_response_mass": float(snapshot.policy[best_response_action - 1]),
        "top_action": int(np.argmax(snapshot.policy)) + 1,
        "direct_weight": snapshot.direct_weight,
    }


def _snapshot_max_abs_difference(
    left: _NetworkSnapshot, right: _NetworkSnapshot
) -> float:
    return max(
        float(np.max(np.abs(left.policy - right.policy))),
        float(np.max(np.abs(left.opponent_policy - right.opponent_policy))),
        abs(left.direct_weight - right.direct_weight),
    )


def evaluate_adaptive_memory_case(
    model: AggroHalNetwork,
    case: MemoryCurriculumCase,
) -> list[dict[str, object]]:
    """Score one paired case and return one row for each latent mode."""

    target_integrity = (
        case.mode_a.target.bitwise_equal(case.mode_b.target)
        and case.mode_a.target.sha256() == case.target_sha256
        and case.mode_b.target.sha256() == case.target_sha256
    )
    if not target_integrity:
        raise RuntimeError("adaptive curriculum target identity check failed")
    if case.target_index <= 0:
        raise RuntimeError("adaptive curriculum case has no recurrent prefix")

    model.eval()
    with torch.inference_mode():
        prefix_a = _forward_tokens(model, case.mode_a.tokens[:-1])
        prefix_b = _forward_tokens(model, case.mode_b.tokens[:-1])
        hidden_a = prefix_a.hidden_state.detach().clone()
        hidden_b = prefix_b.hidden_state.detach().clone()
        device, dtype = _model_device_dtype(model)
        zero = model.initial_hidden(1, device=device, dtype=dtype)

        correct_a = _snapshot(_forward_tokens(model, (case.target,), hidden_a))
        correct_b = _snapshot(_forward_tokens(model, (case.target,), hidden_b))
        swapped_a = _snapshot(_forward_tokens(model, (case.target,), hidden_b))
        swapped_b = _snapshot(_forward_tokens(model, (case.target,), hidden_a))
        zero_target = _snapshot(_forward_tokens(model, (case.target,), zero))
        history_free_a = _snapshot(_history_free_output(model, case.mode_a.tokens))
        history_free_b = _snapshot(_history_free_output(model, case.mode_b.tokens))

    history_free_differences = {
        "a": _snapshot_max_abs_difference(history_free_a, zero_target),
        "b": _snapshot_max_abs_difference(history_free_b, zero_target),
    }
    history_free_matches = all(
        difference <= HISTORY_FREE_EQUIVALENCE_ATOL
        for difference in history_free_differences.values()
    )

    matrix = np.asarray(case.target.stage_matrix, dtype=np.float64)
    oriented = matrix if case.role == "dropper" else -matrix.T
    truths = {
        "a": np.asarray(case.mode_a.target_truth, dtype=np.float64),
        "b": np.asarray(case.mode_b.target_truth, dtype=np.float64),
    }
    correct = {"a": correct_a, "b": correct_b}
    swapped = {"a": swapped_a, "b": swapped_b}
    history_free = {"a": history_free_a, "b": history_free_b}

    rows: list[dict[str, object]] = []
    for mode_index, mode in enumerate(_MODES):
        other_mode = _MODES[1 - mode_index]
        truth = truths[mode]
        action_values = oriented @ truth
        best_action = case.best_responses.action(case.role, mode)
        wrong_mode_action = case.best_responses.action(case.role, other_mode)
        if int(np.argmax(action_values)) + 1 != best_action:
            raise RuntimeError(
                "curriculum best-response audit disagrees with target matrix"
            )
        reference = float(
            action_values[best_action - 1] - action_values[wrong_mode_action - 1]
        )
        if not np.isfinite(reference) or reference <= 0.0:
            raise RuntimeError("adaptive payoff normalization has no crossover penalty")

        condition_snapshots = {
            "correct": correct[mode],
            "swapped": swapped[mode],
            "zero_target": zero_target,
            "history_free": history_free[mode],
        }
        scores = {
            condition: _score(
                snapshot,
                truth=truth,
                action_values=action_values,
                best_response_action=best_action,
            )
            for condition, snapshot in condition_snapshots.items()
        }
        correct_score = scores["correct"]
        contrasts: dict[str, dict[str, float]] = {}
        for baseline in _BASELINES:
            baseline_score = scores[baseline]
            contrasts[f"correct_vs_{baseline}"] = {
                "normalized_payoff_gain": (
                    float(correct_score["expected_payoff"])
                    - float(baseline_score["expected_payoff"])
                )
                / reference,
                "nll_gain_nats": (
                    float(baseline_score["opponent_nll_nats"])
                    - float(correct_score["opponent_nll_nats"])
                ),
            }

        rows.append(
            {
                "split": case.split,
                "example_seed": case.example_seed,
                "role": case.role,
                "mode": mode,
                "cover_games": case.parameters.cover_games,
                "prefix_tokens": case.target_index,
                "target_sha256": case.target_sha256,
                "target_integrity_passed": target_integrity,
                "history_free_matches_zero_target": history_free_matches,
                "history_free_zero_target_max_abs_difference": (
                    history_free_differences[mode]
                ),
                "reference_wrong_mode_penalty": reference,
                "scores": scores,
                "contrasts": contrasts,
            }
        )
    return rows


def cluster_bootstrap_interval(
    values_by_seed: Mapping[int, float],
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    """Bootstrap independent example-seed clusters deterministically."""

    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    ordered = np.asarray(
        [float(values_by_seed[key]) for key in sorted(values_by_seed)],
        dtype=np.float64,
    )
    if ordered.size == 0 or not np.all(np.isfinite(ordered)):
        raise ValueError("bootstrap values must be finite and nonempty")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, ordered.size, size=(replicates, ordered.size))
    means = ordered[indices].mean(axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    return float(low), float(high)


def _metric_summary(
    rows: Sequence[Mapping[str, object]],
    *,
    comparison: str,
    metric: str,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    values: dict[int, list[float]] = {}
    for row in rows:
        contrasts = row.get("contrasts")
        if not isinstance(contrasts, Mapping):
            raise TypeError("adaptive evaluation row has malformed contrasts")
        comparison_values = contrasts.get(comparison)
        if not isinstance(comparison_values, Mapping):
            raise TypeError(f"adaptive evaluation row is missing {comparison}")
        value = float(comparison_values[metric])
        values.setdefault(int(row["example_seed"]), []).append(value)
    clustered = {
        example_seed: float(np.mean(seed_values))
        for example_seed, seed_values in values.items()
    }
    interval = cluster_bootstrap_interval(
        clustered,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    return {
        "mean": float(np.mean(list(clustered.values()))),
        "bootstrap_95": list(interval),
        "example_seed_units": len(clustered),
    }


def summarize_adaptive_memory_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    expected_example_seeds: Sequence[int],
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, object]:
    """Build role-by-mode summaries without calculating a pooled rescue metric."""

    if not rows:
        raise ValueError("cannot summarize an empty adaptive evaluation")
    expected = tuple(sorted(int(seed) for seed in expected_example_seeds))
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("expected example seeds must be nonempty and unique")
    if bootstrap_replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")

    by_role_mode: dict[str, dict[str, object]] = {}
    for role_index, role in enumerate(_ROLES):
        role_result: dict[str, object] = {}
        for mode_index, mode in enumerate(_MODES):
            cell_rows = [
                row
                for row in rows
                if row.get("role") == role and row.get("mode") == mode
            ]
            if not cell_rows:
                raise ValueError(f"adaptive evaluation is missing cell {role}/{mode}")
            observed = tuple(sorted({int(row["example_seed"]) for row in cell_rows}))
            exact_one_row_per_seed = len(cell_rows) == len(observed)
            comparisons: dict[str, object] = {}
            for comparison_index, baseline in enumerate(_BASELINES):
                comparison = f"correct_vs_{baseline}"
                comparison_result: dict[str, object] = {}
                for metric_index, metric in enumerate(
                    ("normalized_payoff_gain", "nll_gain_nats")
                ):
                    metric_seed = (
                        int(bootstrap_seed)
                        + role_index * 100_000
                        + mode_index * 10_000
                        + comparison_index * 1_000
                        + metric_index * 101
                    )
                    comparison_result[metric] = _metric_summary(
                        cell_rows,
                        comparison=comparison,
                        metric=metric,
                        bootstrap_replicates=bootstrap_replicates,
                        bootstrap_seed=metric_seed,
                    )
                comparisons[comparison] = comparison_result

            role_result[mode] = {
                "row_count": len(cell_rows),
                "example_seeds": list(observed),
                "seed_set_complete": observed == expected and exact_one_row_per_seed,
                "target_integrity_passed": all(
                    bool(row.get("target_integrity_passed")) for row in cell_rows
                ),
                "history_free_matches_zero_target": all(
                    bool(row.get("history_free_matches_zero_target"))
                    for row in cell_rows
                ),
                "minimum_cover_games": min(
                    int(row["cover_games"]) for row in cell_rows
                ),
                "maximum_cover_games": max(
                    int(row["cover_games"]) for row in cell_rows
                ),
                "comparisons": comparisons,
            }
        by_role_mode[role] = role_result

    cells = [by_role_mode[role][mode] for role in _ROLES for mode in _MODES]
    return {
        "bootstrap_unit": "example_seed",
        "expected_example_seeds": list(expected),
        "pooled_metrics_computed": False,
        "all_cells_complete": all(bool(cell["seed_set_complete"]) for cell in cells),
        "target_integrity_passed": all(
            bool(cell["target_integrity_passed"]) for cell in cells
        ),
        "history_free_matches_zero_target": all(
            bool(cell["history_free_matches_zero_target"]) for cell in cells
        ),
        "by_role_mode": by_role_mode,
    }


def _lower_bound(cell: Mapping[str, object], comparison: str, metric: str) -> float:
    comparisons = cell.get("comparisons")
    if not isinstance(comparisons, Mapping):
        raise TypeError("adaptive summary cell has malformed comparisons")
    comparison_result = comparisons.get(comparison)
    if not isinstance(comparison_result, Mapping):
        raise TypeError(f"adaptive summary cell is missing {comparison}")
    metric_result = comparison_result.get(metric)
    if not isinstance(metric_result, Mapping):
        raise TypeError(f"adaptive summary comparison is missing {metric}")
    interval = metric_result.get("bootstrap_95")
    if not isinstance(interval, Sequence) or len(interval) != 2:
        raise TypeError("adaptive summary metric has no two-sided interval")
    return float(interval[0])


def adaptive_promotion_decision(
    summary: Mapping[str, object],
    *,
    split: str,
) -> dict[str, object]:
    """Apply the frozen every-cell gate; pooled performance is never consulted."""

    by_role_mode = summary.get("by_role_mode")
    if not isinstance(by_role_mode, Mapping):
        raise TypeError("adaptive summary is missing role-by-mode cells")

    failed: list[str] = []
    cell_results: dict[str, object] = {}
    for role in _ROLES:
        role_cells = by_role_mode.get(role)
        if not isinstance(role_cells, Mapping):
            raise TypeError(f"adaptive summary is missing role {role}")
        for mode in _MODES:
            cell = role_cells.get(mode)
            if not isinstance(cell, Mapping):
                raise TypeError(f"adaptive summary is missing cell {role}/{mode}")
            label = f"{role}/{mode}"
            checks: dict[str, bool] = {
                "complete_registered_seed_set": bool(cell.get("seed_set_complete")),
                "target_integrity": bool(cell.get("target_integrity_passed")),
                "history_free_zero_target_equivalence": bool(
                    cell.get("history_free_matches_zero_target")
                ),
                "minimum_cover_delay": (
                    int(cell.get("minimum_cover_games", -1))
                    >= PROMOTION_THRESHOLDS.minimum_cover_games
                ),
            }
            lower_bounds: dict[str, dict[str, float]] = {}
            for baseline in _BASELINES:
                comparison = f"correct_vs_{baseline}"
                payoff = _lower_bound(cell, comparison, "normalized_payoff_gain")
                nll = _lower_bound(cell, comparison, "nll_gain_nats")
                lower_bounds[comparison] = {
                    "normalized_payoff_gain": payoff,
                    "nll_gain_nats": nll,
                }
                checks[f"{comparison}_payoff"] = (
                    payoff > PROMOTION_THRESHOLDS.normalized_payoff_gain
                )
                checks[f"{comparison}_nll"] = nll > PROMOTION_THRESHOLDS.nll_gain_nats
            for check, passed in checks.items():
                if not passed:
                    failed.append(f"{label}:{check}")
            cell_results[label] = {
                "passed": all(checks.values()),
                "checks": checks,
                "lower_bounds": lower_bounds,
            }

    split_passed = split == "validation"
    if not split_passed:
        failed.append("split:validation_required")
    passed = split_passed and not failed
    return {
        "passed": passed,
        "decision": "unlock_ppo" if passed else "hold_warmstart",
        "eligible_split": "validation",
        "evaluated_split": split,
        "thresholds": asdict(PROMOTION_THRESHOLDS),
        "strict_inequality": True,
        "pooled_rescue_allowed": False,
        "cells": cell_results,
        "failed_checks": failed,
    }


def evaluate_adaptive_memory_model(
    model: AggroHalNetwork,
    exact_agent: StageGameProvider,
    *,
    split: SplitName | MemoryCurriculumSplit = "validation",
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, object]:
    """Evaluate a model over one complete registered curriculum split."""

    spec = memory_curriculum_split(split) if isinstance(split, str) else split
    protocol = adaptive_memory_protocol(
        spec,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    rows: list[dict[str, object]] = []
    for example_seed in spec.example_seeds:
        for role in _ROLES:
            case = build_memory_curriculum_case(
                exact_agent,
                split=spec,
                example_seed=example_seed,
                role=role,
            )
            rows.extend(evaluate_adaptive_memory_case(model, case))
    summary = summarize_adaptive_memory_rows(
        rows,
        expected_example_seeds=spec.example_seeds,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    return {
        "schema_version": ADAPTIVE_EVALUATION_SCHEMA,
        "protocol": protocol,
        "protocol_sha256": hashlib.sha256(_canonical_json(protocol)).hexdigest(),
        "rows": rows,
        "summary": summary,
        "promotion": adaptive_promotion_decision(summary, split=spec.name),
    }


def evaluate_adaptive_memory_checkpoint(
    checkpoint: str | Path,
    *,
    split: SplitName = "validation",
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
    device: str | torch.device = "cpu",
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    cpu_threads: int = 6,
) -> dict[str, object]:
    """Load a strict checkpoint and run the complete adaptive promotion gate."""

    resolved_device = torch.device(device)
    if resolved_device.type == "cpu":
        if cpu_threads <= 0:
            raise ValueError("CPU thread count must be positive")
        torch.set_num_threads(int(cpu_threads))
    exact_agent = CompleteDTHAgent(Path(artifact_dir))
    ruleset = dth_compatibility(exact_agent)
    model, payload = load_checkpoint(
        checkpoint,
        dth_ruleset=ruleset,
        device=resolved_device,
    )
    report = evaluate_adaptive_memory_model(
        model,
        exact_agent,
        split=split,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    report["checkpoint"] = {
        "path": str(Path(checkpoint)),
        "sha256": _sha256_file(checkpoint),
        "schema_version": payload.get("schema_version"),
        "config": payload.get("config"),
    }
    report["dth_compatibility"] = ruleset
    report["device"] = str(resolved_device)
    return report


def _write_json(path: str | Path, payload: object) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol-output", type=Path)
    parser.add_argument(
        "--split", choices=("train", "validation"), default="validation"
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES
    )
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--cpu-threads", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    protocol = adaptive_memory_protocol(
        args.split,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    protocol_path = args.protocol_output or args.output.with_name(
        args.output.stem + "-protocol.json"
    )
    # Commit the model-independent gate before checkpoint weights are opened.
    _write_json(protocol_path, protocol)
    report = evaluate_adaptive_memory_checkpoint(
        args.checkpoint,
        split=args.split,
        artifact_dir=args.artifact_dir,
        device=args.device,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
        cpu_threads=args.cpu_threads,
    )
    _write_json(args.output, report)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "protocol_output": str(protocol_path),
                "checkpoint_sha256": report["checkpoint"]["sha256"],
                "promotion": report["promotion"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
