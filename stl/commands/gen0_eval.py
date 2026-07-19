"""Evaluate one strict Gen-0 checkpoint on the immutable V3 external ruler."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Callable, Sequence

import numpy as np

from stl.learning.certificates import (
    ExactPolicyCertificate,
    certificates_by_state,
    load_certificate_shard,
)
from stl.learning.replay import (
    ShardRole,
    TargetKind,
    TrainingRecordV3,
    exact_state_hash,
    load_replay_manifest,
    load_replay_shard,
    reconstruct_game,
)
from stl.learning.holdout import (
    V4_HOLDOUT_GATES,
    claim_holdout_use,
    complete_holdout_use,
    load_holdout_seal,
    load_taxonomy,
    sha256_file,
)
from stl.learning.train import (
    load_checkpoint,
    load_checkpoint_bundle,
    make_predict_fn,
)
from stl.solver.exact import (
    ExactSearchConfig,
    ExactSolveResult,
    UtilityBreakdown,
    solve_exact_finite_horizon,
)
from stl.solver.mcts_conformance import unique_equilibrium_policy_tv
from stl.solver.search import diagnose_exact_strategy


REPORT_SCHEMA = "stl.gen0-external-ruler-report.v1"
# Replay values and cutoff probabilities are serialized as float32.  Exact
# re-solves use float64, so this gate permits only the expected storage roundoff.
FLOAT32_RECOMPUTE_TOLERANCE = 1e-6
REQUIRED_RULER_SOURCES = (
    "terminal",
    "tablebase",
    "tablebase_interior",
    "exact_horizon_2",
    "exact_horizon_3",
)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _feature_hash(record: TrainingRecordV3) -> str:
    return hashlib.sha256(np.ascontiguousarray(record.features).tobytes()).hexdigest()


def _assert_isolated(
    train: Sequence[TrainingRecordV3], ruler: Sequence[TrainingRecordV3]
) -> dict[str, int]:
    overlaps = {
        "exact_states": len(
            {exact_state_hash(record.exact_state) for record in train}
            & {exact_state_hash(record.exact_state) for record in ruler}
        ),
        "episodes": len(
            {record.episode_id for record in train if record.episode_id}
            & {record.episode_id for record in ruler if record.episode_id}
        ),
        "features": len(
            {_feature_hash(record) for record in train}
            & {_feature_hash(record) for record in ruler}
        ),
    }
    if any(overlaps.values()):
        raise ValueError(f"Gen-0 train/external-ruler overlap: {overlaps}")
    return overlaps


def _source_value_metrics(
    records: Sequence[TrainingRecordV3], predictions: np.ndarray
) -> tuple[dict[str, dict[str, float | int]], float, float, float]:
    labels = np.asarray([record.value for record in records], dtype=np.float64)
    errors = np.asarray(predictions, dtype=np.float64) - labels
    squared = errors**2
    metrics: dict[str, dict[str, float | int]] = {}
    sources = np.asarray([record.source for record in records])
    for source in sorted(set(sources.tolist())):
        mask = sources == source
        source_errors = errors[mask]
        source_squared = squared[mask]
        cutoffs = np.asarray(
            [
                record.cutoff_probability
                for record, selected in zip(records, mask)
                if selected
            ],
            dtype=np.float64,
        )
        metrics[source] = {
            "count": int(mask.sum()),
            "mse": float(source_squared.mean()),
            "mae": float(np.abs(source_errors).mean()),
            "max_abs_error": float(np.abs(source_errors).max()),
            "mean_cutoff_probability": float(cutoffs.mean()),
            "max_cutoff_probability": float(cutoffs.max()),
        }

    nonzero = labels != 0.0
    if nonzero.any():
        probabilities = 0.5 * (np.clip(predictions[nonzero], -1.0, 1.0) + 1.0)
        outcomes = (labels[nonzero] > 0.0).astype(np.float64)
        brier = float(np.mean((probabilities - outcomes) ** 2))
    else:
        brier = 0.0
    return metrics, float(squared.mean()), float(np.sqrt(squared.mean())), brier


def _taxonomy_value_metrics(
    records: Sequence[TrainingRecordV3],
    predictions: np.ndarray,
    taxonomy: dict[str, dict[str, object]],
) -> dict[str, dict[str, float | int]]:
    groups: dict[str, list[float]] = {}
    tablebase_hashes: set[str] = set()
    for record, prediction in zip(records, predictions):
        if record.source not in {"tablebase", "tablebase_interior"}:
            continue
        state_hash = exact_state_hash(record.exact_state)
        tablebase_hashes.add(state_hash)
        entry = taxonomy.get(state_hash)
        if entry is None:
            raise ValueError(f"tablebase taxonomy misses state {state_hash}")
        error = float(prediction) - float(record.value)
        family = str(entry["family"])
        groups.setdefault(f"family:{family}", []).append(error)
        history = str(entry["history_profile"])
        if record.source == "tablebase_interior":
            groups.setdefault(f"interior_history:{history}", []).append(error)
        elif family.startswith("boundary_"):
            parameters = entry.get("parameters", {})
            if isinstance(parameters, dict) and "checker" in parameters:
                groups.setdefault(
                    f"boundary_checker:{parameters['checker']}", []
                ).append(error)
    extra = set(taxonomy) - tablebase_hashes
    if extra:
        raise ValueError("tablebase taxonomy contains states absent from replay shard")
    return {
        name: {
            "count": len(errors),
            "mse": float(np.mean(np.square(errors))),
            "max_abs_error": float(np.max(np.abs(errors))),
        }
        for name, errors in sorted(groups.items())
    }


def _normalized_policy(full_policy: np.ndarray, actions: Sequence[int]) -> np.ndarray:
    policy = np.asarray([full_policy[action] for action in actions], dtype=np.float64)
    if not np.all(np.isfinite(policy)) or np.any(policy < 0.0):
        raise ValueError("predicted policy is non-finite or negative")
    total = float(policy.sum())
    if total <= 0.0:
        raise ValueError("predicted policy has no mass on exact legal actions")
    return policy / total


def _exact_from_certificate(
    certificate: ExactPolicyCertificate,
) -> ExactSolveResult:
    resolved = 1.0 - certificate.unresolved_probability
    hal_win = 0.5 * (resolved + certificate.value_for_hal)
    baku_win = 0.5 * (resolved - certificate.value_for_hal)
    return ExactSolveResult(
        dropper_strategy=certificate.dropper_strategy,
        checker_strategy=certificate.checker_strategy,
        value_for_hal=certificate.value_for_hal,
        breakdown=UtilityBreakdown(
            value=certificate.value_for_hal,
            hal_win_probability=hal_win,
            baku_win_probability=baku_win,
            unresolved_probability=certificate.unresolved_probability,
        ),
        unresolved_probability=certificate.unresolved_probability,
        half_round_horizon=certificate.horizon,
        drop_actions=certificate.drop_actions,
        check_actions=certificate.check_actions,
        payoff_for_hal=certificate.payoff_for_hal,
    )


def _policy_report(
    records: Sequence[TrainingRecordV3],
    predict_game: Callable,
    *,
    exact_solver: Callable[[object, int, ExactSearchConfig], ExactSolveResult] = (
        solve_exact_finite_horizon
    ),
    certificates: dict[str, ExactPolicyCertificate] | None = None,
    certificate_recompute_count: int = 0,
    uniqueness_tolerance: float,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    illegal_masses: list[float] = []
    gaps: list[float] = []
    target_drop_tvs: list[float] = []
    target_check_tvs: list[float] = []
    unique_drop_tvs: list[float] = []
    unique_check_tvs: list[float] = []
    label_recompute_errors: list[float] = []
    cutoff_recompute_errors: list[float] = []
    certificate_matrix_errors: list[float] = []
    active_records = [
        record
        for record in records
        if float(record.dropper_dist.sum(dtype=np.float64)) > 0.0
        or float(record.checker_dist.sum(dtype=np.float64)) > 0.0
    ]

    for index, record in enumerate(active_records, start=1):
        if record.target_kind is not TargetKind.EXACT_VALUE:
            raise ValueError(
                f"active ruler policy is not exact_value: {record.episode_id}"
            )
        game = reconstruct_game(record.exact_state)
        _value, dropper_full, checker_full = predict_game(
            game, horizon=record.value_horizon_half_rounds
        )
        illegal_mass = float(
            dropper_full[~record.dropper_legal_mask].sum(dtype=np.float64)
            + checker_full[~record.checker_legal_mask].sum(dtype=np.float64)
        )
        illegal_masses.append(illegal_mass)

        state_hash = exact_state_hash(record.exact_state)
        certificate = None if certificates is None else certificates.get(state_hash)
        if certificate is None:
            print(
                f"[gen0-eval] exact policy {index}/{len(active_records)}: "
                f"re-solving episode={record.episode_id}, "
                f"horizon={record.value_horizon_half_rounds}",
                flush=True,
            )
            exact = exact_solver(
                game,
                record.value_horizon_half_rounds,
                ExactSearchConfig(),
            )
        else:
            if certificate.search_config_digest != record.search_config_digest:
                raise ValueError("policy certificate search config digest mismatch")
            if certificate.horizon != record.value_horizon_half_rounds:
                raise ValueError("policy certificate horizon mismatch")
            exact = _exact_from_certificate(certificate)
            if index <= certificate_recompute_count:
                recomputed = exact_solver(
                    game,
                    record.value_horizon_half_rounds,
                    ExactSearchConfig(),
                )
                if recomputed.payoff_for_hal is None:
                    raise ValueError("certificate spot check produced no payoff matrix")
                if recomputed.drop_actions != exact.drop_actions or (
                    recomputed.check_actions != exact.check_actions
                ):
                    raise ValueError("certificate spot-check action schema mismatch")
                matrix_error = float(
                    np.max(
                        np.abs(recomputed.payoff_for_hal - exact.payoff_for_hal),
                        initial=0.0,
                    )
                )
                certificate_matrix_errors.append(matrix_error)
                if matrix_error > FLOAT32_RECOMPUTE_TOLERANCE:
                    raise ValueError("certificate spot-check payoff matrix mismatch")
        if exact.payoff_for_hal is None:
            raise ValueError("active exact policy row produced no payoff matrix")
        dropper = _normalized_policy(dropper_full, exact.drop_actions)
        checker = _normalized_policy(checker_full, exact.check_actions)
        predicted = replace(
            exact,
            dropper_strategy=dropper,
            checker_strategy=checker,
        )
        diagnostics = diagnose_exact_strategy(game, predicted)
        gaps.append(diagnostics.nash_gap)
        label_recompute_errors.append(abs(record.value - exact.value_for_hal))
        cutoff_recompute_errors.append(
            abs(record.cutoff_probability - exact.unresolved_probability)
        )

        drop_target_tv = 0.5 * float(
            np.abs(dropper_full - record.dropper_dist).sum(dtype=np.float64)
        )
        check_target_tv = 0.5 * float(
            np.abs(checker_full - record.checker_dist).sum(dtype=np.float64)
        )
        target_drop_tvs.append(drop_target_tv)
        target_check_tvs.append(check_target_tv)

        dropper_role, _checker_role = game.get_roles_for_half(game.current_half)
        hal_is_dropper = dropper_role.name.lower() == "hal"
        unique_tv = unique_equilibrium_policy_tv(
            exact.payoff_for_hal,
            dropper,
            checker,
            hal_is_dropper=hal_is_dropper,
            tolerance=uniqueness_tolerance,
        )
        if unique_tv is not None:
            unique_drop_tvs.append(unique_tv[0])
            unique_check_tvs.append(unique_tv[1])

        rows.append(
            {
                "episode_id": record.episode_id,
                "half_round_index": record.half_round_index,
                "source": record.source,
                "horizon": record.value_horizon_half_rounds,
                "stored_value": record.value,
                "recomputed_value": exact.value_for_hal,
                "stored_cutoff_probability": record.cutoff_probability,
                "recomputed_cutoff_probability": exact.unresolved_probability,
                "label_value_recompute_error": abs(record.value - exact.value_for_hal),
                "cutoff_recompute_error": abs(
                    record.cutoff_probability - exact.unresolved_probability
                ),
                "predicted_policy_value": diagnostics.expected_value,
                "dropper_exploitability": diagnostics.dropper_exploitability,
                "checker_exploitability": diagnostics.checker_exploitability,
                "saddle_gap": diagnostics.nash_gap,
                "target_dropper_tv_report_only": drop_target_tv,
                "target_checker_tv_report_only": check_target_tv,
                "strict_unique_pure_equilibrium": unique_tv is not None,
                "unique_dropper_tv": None if unique_tv is None else unique_tv[0],
                "unique_checker_tv": None if unique_tv is None else unique_tv[1],
                "illegal_policy_mass": illegal_mass,
            }
        )

    def aggregate(values: Sequence[float]) -> dict[str, float | int | None]:
        if not values:
            return {"count": 0, "mean": None, "median": None, "maximum": None}
        array = np.asarray(values, dtype=np.float64)
        return {
            "count": len(values),
            "mean": float(array.mean()),
            "median": float(np.median(array)),
            "maximum": float(array.max()),
        }

    return {
        "active_rows": len(active_records),
        "inactive_rows": len(records) - len(active_records),
        "maximum_illegal_mass": max(illegal_masses, default=0.0),
        "saddle_gap": aggregate(gaps),
        "target_dropper_tv_report_only": aggregate(target_drop_tvs),
        "target_checker_tv_report_only": aggregate(target_check_tvs),
        "strict_unique_dropper_tv": aggregate(unique_drop_tvs),
        "strict_unique_checker_tv": aggregate(unique_check_tvs),
        "label_value_recompute_error": aggregate(label_recompute_errors),
        "cutoff_recompute_error": aggregate(cutoff_recompute_errors),
        "certificate_spot_checks": {
            "count": len(certificate_matrix_errors),
            "maximum_matrix_error": max(certificate_matrix_errors, default=0.0),
        },
        "rows": rows,
    }


def _gate(
    *,
    value_sources: dict[str, dict[str, float | int]],
    policy: dict[str, object],
    reload_max_abs_difference: float,
    tablebase_mse_max: float,
    tablebase_interior_mse_max: float,
    exact_saddle_gap_max: float,
    unique_policy_tv_median_max: float,
    exact_cutoff_max: float,
    taxonomy_metrics: dict[str, dict[str, float | int]] | None = None,
    boundary_max_abs_error: float | None = None,
    interior_max_abs_error: float | None = None,
    exact_horizon_mse_max: float | None = None,
    exact_max_abs_error: float | None = None,
    terminal_mse_max: float | None = None,
    terminal_max_abs_error: float | None = None,
) -> dict[str, object]:
    checks: dict[str, dict[str, object]] = {}

    def maximum_check(name: str, observed: float | None, limit: float) -> None:
        checks[name] = {
            "observed": observed,
            "maximum": limit,
            "passed": observed is not None
            and math.isfinite(observed)
            and observed <= limit,
        }

    for source in REQUIRED_RULER_SOURCES:
        checks[f"required_source:{source}"] = {
            "observed": int(value_sources.get(source, {}).get("count", 0)),
            "minimum": 1,
            "passed": int(value_sources.get(source, {}).get("count", 0)) >= 1,
        }
    maximum_check(
        "tablebase_mse",
        None
        if "tablebase" not in value_sources
        else float(value_sources["tablebase"]["mse"]),
        tablebase_mse_max,
    )
    maximum_check(
        "tablebase_interior_mse",
        None
        if "tablebase_interior" not in value_sources
        else float(value_sources["tablebase_interior"]["mse"]),
        tablebase_interior_mse_max,
    )
    if boundary_max_abs_error is not None:
        maximum_check(
            "tablebase_max_abs_error",
            None
            if "tablebase" not in value_sources
            else float(value_sources["tablebase"]["max_abs_error"]),
            boundary_max_abs_error,
        )
    if interior_max_abs_error is not None:
        maximum_check(
            "tablebase_interior_max_abs_error",
            None
            if "tablebase_interior" not in value_sources
            else float(value_sources["tablebase_interior"]["max_abs_error"]),
            interior_max_abs_error,
        )
    if exact_horizon_mse_max is not None:
        for source in ("exact_horizon_2", "exact_horizon_3"):
            maximum_check(
                f"{source}_mse",
                None if source not in value_sources else float(value_sources[source]["mse"]),
                exact_horizon_mse_max,
            )
    if exact_max_abs_error is not None:
        for source in ("exact_horizon_2", "exact_horizon_3"):
            maximum_check(
                f"{source}_max_abs_error",
                None
                if source not in value_sources
                else float(value_sources[source]["max_abs_error"]),
                exact_max_abs_error,
            )
    if terminal_mse_max is not None:
        maximum_check(
            "terminal_mse",
            None
            if "terminal" not in value_sources
            else float(value_sources["terminal"]["mse"]),
            terminal_mse_max,
        )
    if terminal_max_abs_error is not None:
        maximum_check(
            "terminal_max_abs_error",
            None
            if "terminal" not in value_sources
            else float(value_sources["terminal"]["max_abs_error"]),
            terminal_max_abs_error,
        )
    if taxonomy_metrics is not None:
        for name, metrics in taxonomy_metrics.items():
            if name.startswith("interior_history:"):
                mse_limit = tablebase_interior_mse_max
                max_limit = interior_max_abs_error
            elif name in {
                "family:boundary_single_overflow",
                "family:boundary_double_overflow",
            } or name.startswith("boundary_checker:"):
                mse_limit = tablebase_mse_max
                max_limit = boundary_max_abs_error
            else:
                continue
            maximum_check(
                f"taxonomy_mse:{name}", float(metrics["mse"]), mse_limit
            )
            if max_limit is not None:
                maximum_check(
                    f"taxonomy_max_abs_error:{name}",
                    float(metrics["max_abs_error"]),
                    max_limit,
                )
    maximum_check(
        "exact_policy_saddle_gap",
        policy["saddle_gap"]["maximum"],
        exact_saddle_gap_max,
    )
    maximum_check(
        "illegal_policy_mass",
        float(policy["maximum_illegal_mass"]),
        0.0,
    )
    maximum_check(
        "checkpoint_reload_prediction_difference",
        reload_max_abs_difference,
        1e-7,
    )
    maximum_check(
        "exact_label_recompute_error",
        policy["label_value_recompute_error"]["maximum"],
        FLOAT32_RECOMPUTE_TOLERANCE,
    )
    maximum_check(
        "exact_cutoff_recompute_error",
        policy["cutoff_recompute_error"]["maximum"],
        FLOAT32_RECOMPUTE_TOLERANCE,
    )
    exact_cutoffs = [
        float(metrics["max_cutoff_probability"])
        for source, metrics in value_sources.items()
        if source.startswith("exact_horizon_")
    ]
    maximum_check(
        "exact_cutoff_probability",
        max(exact_cutoffs) if exact_cutoffs else None,
        exact_cutoff_max,
    )

    unique_medians = [
        value
        for value in (
            policy["strict_unique_dropper_tv"]["median"],
            policy["strict_unique_checker_tv"]["median"],
        )
        if value is not None
    ]
    if unique_medians:
        maximum_check(
            "strict_unique_policy_median_tv",
            max(float(value) for value in unique_medians),
            unique_policy_tv_median_max,
        )
    else:
        checks["strict_unique_policy_median_tv"] = {
            "observed": None,
            "maximum": unique_policy_tv_median_max,
            "passed": True,
            "not_applicable": "no strict unique pure equilibrium in the ruler",
        }

    failures = [name for name, check in checks.items() if not check["passed"]]
    return {
        "passed": not failures,
        "failures": failures,
        "checks": checks,
    }


def evaluate_checkpoint(
    *,
    checkpoint_path: str | Path,
    train_path: str | Path,
    ruler_path: str | Path,
    certificate_path: str | Path | None = None,
    device: str = "cpu",
    tablebase_mse_max: float = 0.01,
    tablebase_interior_mse_max: float = 0.05,
    exact_saddle_gap_max: float = 0.05,
    unique_policy_tv_median_max: float = 0.15,
    exact_cutoff_max: float = 0.5,
    uniqueness_tolerance: float = 1e-9,
    certificate_recompute_count: int = 8,
    taxonomy_path: str | Path | None = None,
    boundary_max_abs_error: float | None = None,
    interior_max_abs_error: float | None = None,
    exact_horizon_mse_max: float | None = None,
    exact_max_abs_error: float | None = None,
    terminal_mse_max: float | None = None,
    terminal_max_abs_error: float | None = None,
) -> dict[str, object]:
    train_manifest = load_replay_manifest(train_path)
    ruler_manifest = load_replay_manifest(ruler_path)
    if ShardRole(str(train_manifest["shard_role"])) is not ShardRole.REPLAY:
        raise ValueError("Gen-0 training shard is not marked replay")
    if ShardRole(str(ruler_manifest["shard_role"])) is not ShardRole.EXTERNAL_RULER:
        raise ValueError("Gen-0 ruler is not marked external_ruler")

    train = load_replay_shard(train_path, for_training=True)
    ruler = load_replay_shard(ruler_path)
    if certificate_path is None:
        candidate = Path(ruler_path).with_name(
            f"{Path(ruler_path).stem}.certificates.npz"
        )
        certificate_path = candidate if candidate.exists() else None
    certificate_map = None
    if certificate_path is not None:
        certificate_map = certificates_by_state(
            load_certificate_shard(
                certificate_path,
                expected_role=ShardRole.EXTERNAL_RULER,
            )
        )
        active_hashes = {
            exact_state_hash(record.exact_state)
            for record in ruler
            if float(record.dropper_dist.sum(dtype=np.float64)) > 0.0
            or float(record.checker_dist.sum(dtype=np.float64)) > 0.0
        }
        if set(certificate_map) != active_hashes:
            raise ValueError(
                "external ruler certificates do not exactly cover active policy rows"
            )
    overlaps = _assert_isolated(train, ruler)
    bundle = load_checkpoint_bundle(checkpoint_path, device=device)
    corpus_digests = tuple(bundle["provenance"].get("corpus_digests", ()))
    train_digest = str(train_manifest["data_sha256"])
    ruler_digest = str(ruler_manifest["data_sha256"])
    if train_digest not in corpus_digests:
        raise ValueError("checkpoint provenance does not name the Gen-0 training shard")
    if ruler_digest in corpus_digests:
        raise ValueError(
            "external ruler digest appears in checkpoint training provenance"
        )

    model = load_checkpoint(checkpoint_path, device=device)
    predict = make_predict_fn(model, device=device)
    predictions = np.asarray(
        [
            predict(
                reconstruct_game(record.exact_state),
                horizon=record.value_horizon_half_rounds,
            )[0]
            for record in ruler
        ],
        dtype=np.float64,
    )
    reloaded = load_checkpoint(checkpoint_path, device=device)
    predict_reloaded = make_predict_fn(reloaded, device=device)
    repeated_predictions = np.asarray(
        [
            predict_reloaded(
                reconstruct_game(record.exact_state),
                horizon=record.value_horizon_half_rounds,
            )[0]
            for record in ruler
        ],
        dtype=np.float64,
    )
    reload_max_abs_difference = float(
        np.max(np.abs(predictions - repeated_predictions), initial=0.0)
    )

    source_metrics, overall_mse, overall_rmse, brier = _source_value_metrics(
        ruler, predictions
    )
    taxonomy_metrics = None
    if taxonomy_path is not None:
        taxonomy_metrics = _taxonomy_value_metrics(
            ruler,
            predictions,
            load_taxonomy(taxonomy_path, replay_path=ruler_path),
        )
    policy = _policy_report(
        ruler,
        predict,
        certificates=certificate_map,
        certificate_recompute_count=certificate_recompute_count,
        uniqueness_tolerance=uniqueness_tolerance,
    )
    gate = _gate(
        value_sources=source_metrics,
        policy=policy,
        reload_max_abs_difference=reload_max_abs_difference,
        tablebase_mse_max=tablebase_mse_max,
        tablebase_interior_mse_max=tablebase_interior_mse_max,
        exact_saddle_gap_max=exact_saddle_gap_max,
        unique_policy_tv_median_max=unique_policy_tv_median_max,
        exact_cutoff_max=exact_cutoff_max,
        taxonomy_metrics=taxonomy_metrics,
        boundary_max_abs_error=boundary_max_abs_error,
        interior_max_abs_error=interior_max_abs_error,
        exact_horizon_mse_max=exact_horizon_mse_max,
        exact_max_abs_error=exact_max_abs_error,
        terminal_mse_max=terminal_mse_max,
        terminal_max_abs_error=terminal_max_abs_error,
    )
    labels = np.asarray([record.value for record in ruler], dtype=np.float64)
    return {
        "schema": REPORT_SCHEMA,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": _sha256_file(checkpoint_path),
            "epoch": bundle["training"].get("epoch"),
            "seed": bundle["provenance"].get("seed"),
            "resolved_config_digest": bundle["provenance"].get(
                "resolved_config_digest"
            ),
            "corpus_digests": list(corpus_digests),
        },
        "training_shard": {
            "path": str(train_path),
            "records": len(train),
            "sha256": train_digest,
            "source_counts": dict(Counter(record.source for record in train)),
        },
        "external_ruler": {
            "path": str(ruler_path),
            "records": len(ruler),
            "sha256": ruler_digest,
            "source_counts": dict(Counter(record.source for record in ruler)),
            "overlaps_with_training": overlaps,
            "certificate_path": (
                None if certificate_path is None else str(certificate_path)
            ),
            "certificate_sha256": (
                None if certificate_path is None else _sha256_file(certificate_path)
            ),
            "taxonomy_path": None if taxonomy_path is None else str(taxonomy_path),
            "taxonomy_sha256": (
                None if taxonomy_path is None else sha256_file(taxonomy_path)
            ),
        },
        "value": {
            "overall_mse": overall_mse,
            "overall_rmse": overall_rmse,
            "brier_score": brier,
            "sources": source_metrics,
            "taxonomy": taxonomy_metrics,
            "predictions": [
                {
                    "episode_id": record.episode_id,
                    "source": record.source,
                    "target": float(label),
                    "prediction": float(prediction),
                    "absolute_error": abs(float(prediction - label)),
                }
                for record, label, prediction in zip(ruler, labels, predictions)
            ],
        },
        "policy": policy,
        "checkpoint_reload_max_abs_difference": reload_max_abs_difference,
        "gate": gate,
    }


def _atomic_write_json(report: dict[str, object], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train", default="stl/outputs/regen2rl/gen0_train_v3.npz")
    parser.add_argument(
        "--ruler", default="stl/outputs/regen2rl/gen0_external_ruler_v3.npz"
    )
    parser.add_argument("--certificates", default=None)
    parser.add_argument("--taxonomy", default=None)
    parser.add_argument("--seal", default=None)
    parser.add_argument("--ledger", default=None)
    parser.add_argument(
        "--out",
        default="stl/outputs/regen2rl/gen0_external_ruler_report_v3.json",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tablebase-mse-max", type=float, default=0.01)
    parser.add_argument("--tablebase-interior-mse-max", type=float, default=0.05)
    parser.add_argument("--exact-saddle-gap-max", type=float, default=0.05)
    parser.add_argument("--unique-policy-tv-median-max", type=float, default=0.15)
    parser.add_argument("--exact-cutoff-max", type=float, default=0.5)
    parser.add_argument("--boundary-max-abs-error", type=float, default=None)
    parser.add_argument("--interior-max-abs-error", type=float, default=None)
    parser.add_argument("--exact-horizon-mse-max", type=float, default=None)
    parser.add_argument("--exact-max-abs-error", type=float, default=None)
    parser.add_argument("--terminal-mse-max", type=float, default=None)
    parser.add_argument("--terminal-max-abs-error", type=float, default=None)
    parser.add_argument("--uniqueness-tolerance", type=float, default=1e-9)
    parser.add_argument("--certificate-recompute-count", type=int, default=8)
    parser.add_argument(
        "--fail-on-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    print(f"[gen0-eval] checkpoint: {args.checkpoint}", flush=True)
    print(f"[gen0-eval] external ruler: {args.ruler}", flush=True)
    seal = None
    if args.seal is not None:
        if args.certificates is None or args.taxonomy is None or args.ledger is None:
            raise ValueError(
                "sealed evaluation requires --certificates, --taxonomy, and --ledger"
            )
        seal = load_holdout_seal(
            args.seal,
            holdout_path=args.ruler,
            certificate_path=args.certificates,
            taxonomy_path=args.taxonomy,
        )
        evaluation_config = {
            key: getattr(args, key)
            for key in (
                "tablebase_mse_max",
                "tablebase_interior_mse_max",
                "exact_saddle_gap_max",
                "unique_policy_tv_median_max",
                "exact_cutoff_max",
                "boundary_max_abs_error",
                "interior_max_abs_error",
                "exact_horizon_mse_max",
                "exact_max_abs_error",
                "terminal_mse_max",
                "terminal_max_abs_error",
                "uniqueness_tolerance",
                "certificate_recompute_count",
            )
        }
        if evaluation_config != seal["gates"] | {
            "uniqueness_tolerance": args.uniqueness_tolerance,
            "certificate_recompute_count": args.certificate_recompute_count,
        }:
            raise ValueError("evaluation thresholds differ from the sealed gates")
        claim_holdout_use(
            args.ledger,
            seal_digest=str(seal["seal_digest"]),
            checkpoint_sha256=sha256_file(args.checkpoint),
            evaluation_config=evaluation_config,
        )
    report = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        train_path=args.train,
        ruler_path=args.ruler,
        certificate_path=args.certificates,
        device=args.device,
        tablebase_mse_max=args.tablebase_mse_max,
        tablebase_interior_mse_max=args.tablebase_interior_mse_max,
        exact_saddle_gap_max=args.exact_saddle_gap_max,
        unique_policy_tv_median_max=args.unique_policy_tv_median_max,
        exact_cutoff_max=args.exact_cutoff_max,
        uniqueness_tolerance=args.uniqueness_tolerance,
        certificate_recompute_count=args.certificate_recompute_count,
        taxonomy_path=args.taxonomy,
        boundary_max_abs_error=args.boundary_max_abs_error,
        interior_max_abs_error=args.interior_max_abs_error,
        exact_horizon_mse_max=args.exact_horizon_mse_max,
        exact_max_abs_error=args.exact_max_abs_error,
        terminal_mse_max=args.terminal_mse_max,
        terminal_max_abs_error=args.terminal_max_abs_error,
    )
    _atomic_write_json(report, args.out)
    if seal is not None:
        complete_holdout_use(
            args.ledger,
            report_path=args.out,
            passed=bool(report["gate"]["passed"]),
        )
    print(
        f"[gen0-eval] value MSE={report['value']['overall_mse']:.6f}; "
        f"max saddle gap={report['policy']['saddle_gap']['maximum']:.6f}",
        flush=True,
    )
    print(
        f"[gen0-eval] gate={'PASS' if report['gate']['passed'] else 'FAIL'}; "
        f"failures={report['gate']['failures']}",
        flush=True,
    )
    print(f"[gen0-eval] report: {args.out}", flush=True)
    return 0 if report["gate"]["passed"] or not args.fail_on_gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
