"""Generate coordinated, reachable V3 Generation-Zero train/ruler shards.

The command owns both splits in one immutable run plan.  Whole legal-engine
episodes are assigned before exact labeling, tactical pins are split once, and
Tier A is training-only.  Committed component chunks are strict replay shards
and are reused after interruption; a changed plan or corrupt chunk fails closed.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict, replace
import hashlib
import os
from pathlib import Path
import platform
import subprocess
from time import perf_counter
from types import SimpleNamespace
from typing import Callable, Sequence

import numpy as np
import scipy

from stl.commands import tier_a_targets
from stl.learning.certificates import (
    ExactPolicyCertificate,
    certificate_from_result,
    certificate_manifest_path,
    load_certificate_shard,
    save_certificate_shard,
)
from stl.learning.contracts import canonical_config_json, config_digest
from stl.learning.model import extract_features
from stl.learning.holdout import (
    taxonomy_path_for,
    write_holdout_seal,
    write_taxonomy,
)
from stl.learning.reachable import (
    ReachableQuota,
    ReachableSnapshot,
    build_scaled_reachable_split,
    split_reachable_candidates,
    validate_physical_state,
)
from stl.learning.replay import (
    ReplayValidationError,
    ShardRole,
    StateOrigin,
    TargetKind,
    TrainingRecordV3,
    audit_feature_collisions,
    exact_state_hash,
    load_replay_manifest,
    load_replay_shard,
    manifest_path_for,
    reconstruct_game,
    save_replay_shard,
)
from stl.learning.targets import (
    INTERIOR_PIN_TAG,
    REJECTED_UNRESOLVED_THRESHOLD,
    SOURCE_TABLEBASE,
    SOURCE_TABLEBASE_INTERIOR,
    ValueTarget,
    _inactive_policy_vectors,
    _target_from_label,
    label_state,
    to_training_record_v3,
)
from stl.learning.tactical_anchors import (
    TacticalAnchorQuota,
    build_tactical_anchor_split,
    build_v4_tactical_anchor_split,
    tactical_taxonomy,
)
from stl.solver.exact import ExactSearchConfig, exact_public_state
from stl.solver.tablebase import REGISTRY


GENERATION_PLAN_SCHEMA = "stl.gen0-generation-plan.v1"
GENERATION_PROVENANCE_SCHEMA = "stl.gen0-generation-provenance.v1"
DEFAULT_SPLIT_SEED = 20260714
SCALED_REACHABLE_QUOTAS = {
    "scaled_pilot": {
        "train": ReachableQuota(16, 8, 4),
        "development": ReachableQuota(8, 4, 2),
        "ruler": ReachableQuota(8, 4, 2),
    },
    "scaled_full": {
        "train": ReachableQuota(2_048, 1_024, 128),
        "development": ReachableQuota(384, 192, 32),
        "ruler": ReachableQuota(384, 192, 32),
    },
    "scaled_v4": {
        "train": ReachableQuota(0, 0, 0),
        "development": ReachableQuota(0, 0, 0),
        "ruler": ReachableQuota(384, 192, 32),
    },
}


def _log(message: str) -> None:
    """Print one immediately visible generation progress line."""

    print(f"[gen0] {message}", flush=True)


def _active_policy_rows(records: Sequence[TrainingRecordV3]) -> int:
    return sum(
        1
        for record in records
        if float(record.dropper_dist.sum(dtype=np.float64)) > 0.0
        or float(record.checker_dist.sum(dtype=np.float64)) > 0.0
    )


def _source_summary(records: Sequence[TrainingRecordV3]) -> str:
    counts = Counter(record.source for record in records)
    return ", ".join(f"{source}={count}" for source, count in sorted(counts.items()))


def _tier_a_row_count(args: argparse.Namespace) -> int:
    if args.tier_a_rows is not None:
        return int(args.tier_a_rows)
    if args.smoke:
        return 2
    if args.profile == "scaled_pilot":
        return 16
    if args.profile in {"scaled_full", "scaled_v4"}:
        return 384
    return 32


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_tree_digest(repo_root: Path) -> str:
    paths = [
        *repo_root.glob("stl/**/*.py"),
        *repo_root.glob("stl/config/**/*.yaml"),
        repo_root / "pyproject.toml",
        repo_root / "uv.lock",
    ]
    digest = hashlib.sha256()
    for path in sorted({item for item in paths if item.is_file()}):
        relative = path.relative_to(repo_root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _git_revision(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _search_config_payload(config: ExactSearchConfig) -> dict[str, object]:
    return {
        "exact_search_config": asdict(config),
        "label_gate": {
            "unresolved_probability_max": REJECTED_UNRESOLVED_THRESHOLD,
            "terminal_utility_perspective": "Hal",
            "policy": "full-width LP equilibrium marginals",
        },
    }


def _resolved_plan(args: argparse.Namespace, repo_root: Path) -> dict[str, object]:
    exact_config = ExactSearchConfig()
    tier_a_manifest = Path(args.tier_a_dir) / "manifest.json"
    if not tier_a_manifest.exists():
        raise FileNotFoundError(f"Tier A manifest not found: {tier_a_manifest}")
    return {
        "schema": GENERATION_PLAN_SCHEMA,
        "profile": str(args.profile),
        "smoke": bool(args.smoke),
        "workers": int(args.workers),
        "split_seed": int(args.split_seed),
        "tier_a": {
            "root": Path(args.tier_a_dir).as_posix(),
            "manifest_sha256": _sha256_file(tier_a_manifest),
            "max_width": float(args.tier_a_max_width),
            "train_rows": _tier_a_row_count(args),
            "external_ruler_rows": 0,
            "policy_labels": "inactive",
        },
        "exact_labeling": _search_config_payload(exact_config),
        "split_rule": (
            "whole paired engine episodes before labeling; tactical hash strata; "
            "global state/episode/feature disjointness"
        ),
        "certificate_precedence": [
            "engine_terminal",
            "tactical_exact_pin",
            "engine_exact_lp",
            "tier_a_interval_midpoint",
        ],
        "reachable_quotas": (
            None
            if args.profile == "baseline"
            else {
                role: asdict(quota)
                for role, quota in SCALED_REACHABLE_QUOTAS[args.profile].items()
            }
        ),
        "blocked_artifacts": {
            str(Path(path)): _sha256_file(Path(path))
            for path in (getattr(args, "blocked_artifact", None) or ())
        },
        "reused_exact_artifacts": {
            "train": (
                None
                if getattr(args, "reuse_train_artifact", None) is None
                else {
                    "path": str(Path(args.reuse_train_artifact)),
                    "sha256": _sha256_file(Path(args.reuse_train_artifact)),
                }
            ),
            "development": (
                None
                if getattr(args, "reuse_dev_artifact", None) is None
                else {
                    "path": str(Path(args.reuse_dev_artifact)),
                    "sha256": _sha256_file(Path(args.reuse_dev_artifact)),
                }
            ),
        },
        "source_tree_sha256": _source_tree_digest(repo_root),
        "git_head": _git_revision(repo_root),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
        },
    }


def _canonical_bytes(value: object) -> bytes:
    return (canonical_config_json(value) + "\n").encode("utf-8")


@contextmanager
def _exclusive_lock(work_dir: Path):
    lock_path = work_dir / "generation.lock"
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"another generation owns lock {lock_path}") from exc
    try:
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        os.close(descriptor)
        yield
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass
        lock_path.unlink(missing_ok=True)


def _write_or_validate_plan(work_dir: Path, plan: dict[str, object]) -> str:
    plan_path = work_dir / "plan.json"
    payload = _canonical_bytes(plan)
    digest = hashlib.sha256(payload).hexdigest()
    if plan_path.exists():
        if plan_path.read_bytes() != payload:
            raise RuntimeError(
                "generation plan differs from the existing work directory; use a new --work-dir"
            )
        _log(f"verified unchanged run plan: {plan_path}")
        return digest
    temporary = plan_path.with_name(f".{plan_path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, plan_path)
    _log(f"wrote immutable run plan: {plan_path}")
    return digest


def _record_from_snapshot(
    snapshot: ReachableSnapshot,
    *,
    search_config: ExactSearchConfig,
    search_digest: str,
) -> tuple[TrainingRecordV3, ExactPolicyCertificate | None]:
    game = reconstruct_game(snapshot.exact_state)
    label = label_state(game, search_config, pinned_table={})
    if label is None:
        raise RuntimeError(
            f"reachable candidate {exact_state_hash(snapshot.exact_state)} has no exact label gate"
        )
    if label.unresolved_probability > REJECTED_UNRESOLVED_THRESHOLD:
        raise RuntimeError(
            "reachable candidate exceeds the declared unresolved-probability ceiling"
        )
    target = replace(
        _target_from_label(game, label),
        state_origin=StateOrigin.ENGINE_TRAJECTORY.value,
        trajectory_actions=snapshot.trajectory_actions,
        episode_id=snapshot.episode_id,
        half_round_index=snapshot.half_round_index,
    )
    record = to_training_record_v3(
        target,
        search_config_digest=search_digest,
        rng_seeds={"trajectory": 0},
    )
    certificate = None
    if label.exact_result is not None:
        certificate = certificate_from_result(
            state_hash=exact_state_hash(snapshot.exact_state),
            search_config_digest=search_digest,
            result=label.exact_result,
        )
    return record, certificate


def _trajectory_records(
    snapshots: Sequence[ReachableSnapshot],
    *,
    search_config: ExactSearchConfig,
    search_digest: str,
    workers: int = 1,
) -> tuple[list[TrainingRecordV3], list[ExactPolicyCertificate]]:
    def log_result(
        index: int,
        snapshot: ReachableSnapshot,
        record: TrainingRecordV3,
        elapsed: float,
    ) -> None:
        policy_status = "active" if _active_policy_rows([record]) else "inactive"
        _log(
            f"exact label {index}/{len(snapshots)} complete in {elapsed:.1f}s: "
            f"value={record.value:.6f}, source={record.source}, "
            f"cutoff={record.cutoff_probability:.6f}, policy={policy_status}, "
            f"episode={snapshot.episode_id}"
        )

    if workers <= 1:
        records: list[TrainingRecordV3] = []
        certificates: list[ExactPolicyCertificate] = []
        for index, snapshot in enumerate(snapshots, start=1):
            started_at = perf_counter()
            _log(
                f"exact label {index}/{len(snapshots)}: "
                f"episode={snapshot.episode_id}, half_round={snapshot.half_round_index}"
            )
            record, certificate = _record_from_snapshot(
                snapshot,
                search_config=search_config,
                search_digest=search_digest,
            )
            log_result(index, snapshot, record, perf_counter() - started_at)
            records.append(record)
            if certificate is not None:
                certificates.append(certificate)
        return records, certificates

    ordered: list[
        tuple[TrainingRecordV3, ExactPolicyCertificate | None, float] | None
    ] = [None] * len(snapshots)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for index, snapshot in enumerate(snapshots):
            _log(
                f"queued exact label {index + 1}/{len(snapshots)}: "
                f"episode={snapshot.episode_id}, half_round={snapshot.half_round_index}"
            )
            future = executor.submit(
                _record_from_snapshot_timed,
                snapshot,
                search_config,
                search_digest,
            )
            futures[future] = index
        for future in as_completed(futures):
            index = futures[future]
            record, certificate, elapsed = future.result()
            ordered[index] = (record, certificate, elapsed)
            log_result(index + 1, snapshots[index], record, elapsed)
    if any(item is None for item in ordered):
        raise AssertionError("parallel exact labeling lost a result")
    completed = [item for item in ordered if item is not None]
    return (
        [item[0] for item in completed],
        [item[1] for item in completed if item[1] is not None],
    )


def _record_from_snapshot_timed(
    snapshot: ReachableSnapshot,
    search_config: ExactSearchConfig,
    search_digest: str,
) -> tuple[TrainingRecordV3, ExactPolicyCertificate | None, float]:
    """Pickle-safe process worker for one expensive exact root solve."""

    started_at = perf_counter()
    record, certificate = _record_from_snapshot(
        snapshot,
        search_config=search_config,
        search_digest=search_digest,
    )
    return record, certificate, perf_counter() - started_at


def _rank(seed: int, namespace: str, key: str) -> str:
    return hashlib.sha256(f"{seed}:{namespace}:{key}".encode("utf-8")).hexdigest()


def _tactical_targets(
    *,
    args: argparse.Namespace,
    split_seed: int,
    search_config: ExactSearchConfig,
    blocked_ruler_state_hashes: set[str] | None = None,
) -> tuple[
    list[ValueTarget],
    list[ValueTarget],
    list[ValueTarget],
    dict[str, dict[str, dict[str, object]]],
]:
    if args.profile != "baseline":
        if args.profile == "scaled_v4":
            split = build_v4_tactical_anchor_split(
                blocked_ruler_state_hashes=blocked_ruler_state_hashes
            )
        else:
            quotas = (
            {
                "train": TacticalAnchorQuota(16, 8),
                "development": TacticalAnchorQuota(8, 4),
                "ruler": TacticalAnchorQuota(24, 8),
            }
            if args.profile == "scaled_pilot"
            else {
                "train": TacticalAnchorQuota(192, 96),
                "development": TacticalAnchorQuota(48, 24),
                "ruler": TacticalAnchorQuota(48, 24),
            }
            )
            split = build_tactical_anchor_split(
                train_quota=quotas["train"],
                development_quota=quotas["development"],
                ruler_quota=quotas["ruler"],
                split_seed=split_seed,
            )
        source_path = Path(__file__).parents[1] / "learning" / "tactical_anchors.py"
        source_digest = _sha256_file(source_path)

        def convert(anchors) -> list[ValueTarget]:
            targets: list[ValueTarget] = []
            for anchor in anchors:
                validate_physical_state(anchor.game, engine_snapshot=False)
                drop, check, drop_mask, check_mask = _inactive_policy_vectors(
                    anchor.game, search_config
                )
                targets.append(
                    ValueTarget(
                        features=extract_features(anchor.game),
                        value=anchor.value_for_hal,
                        source=(
                            SOURCE_TABLEBASE_INTERIOR
                            if anchor.stratum == "interior"
                            else SOURCE_TABLEBASE
                        ),
                        horizon=0,
                        dropper_dist=drop,
                        checker_dist=check,
                        dropper_legal_mask=drop_mask,
                        checker_legal_mask=check_mask,
                        exact_state=exact_public_state(anchor.game),
                        target_kind="tablebase_value",
                        state_origin=StateOrigin.TACTICAL_TABLEBASE.value,
                        source_artifact=(
                            f"stl/learning/tactical_anchors.py::{anchor.name}"
                        ),
                        source_artifact_digest=source_digest,
                        episode_id=f"tactical-{anchor.name}",
                    )
                )
            return targets

        converted = (
            convert(split.train),
            convert(split.development),
            convert(split.ruler),
        )
        _log(
            "scaled tactical anchors ready: "
            f"train={len(converted[0])}, development={len(converted[1])}, "
            f"external_test={len(converted[2])}"
        )
        taxonomy = {
            "train": tactical_taxonomy(split.train),
            "development": tactical_taxonomy(split.development),
            "ruler": tactical_taxonomy(split.ruler),
        }
        return (*converted, taxonomy)

    _log("loading and physically validating pinned tactical value states")
    source_path = Path(__file__).parents[1] / "solver" / "tablebase.py"
    source_digest = _sha256_file(source_path)
    strata: dict[str, list[tuple[str, object]]] = {"boundary": [], "interior": []}
    for name, factory in sorted(REGISTRY.items()):
        scenario = factory()
        if scenario.expected_value is None:
            continue
        validate_physical_state(scenario.game, engine_snapshot=False)
        stratum = "interior" if INTERIOR_PIN_TAG in scenario.tags else "boundary"
        strata[stratum].append((name, scenario))

    train: list[ValueTarget] = []
    ruler: list[ValueTarget] = []
    for stratum, entries in strata.items():
        ordered = sorted(
            entries,
            key=lambda item: _rank(
                split_seed,
                f"tactical-{stratum}",
                exact_state_hash(exact_public_state(item[1].game)),
            ),
        )
        ruler_count = max(1, round(len(ordered) * 0.2)) if len(ordered) > 1 else 0
        for index, (name, scenario) in enumerate(ordered):
            game = scenario.game
            drop, check, drop_mask, check_mask = _inactive_policy_vectors(
                game, search_config
            )
            source = (
                SOURCE_TABLEBASE_INTERIOR if stratum == "interior" else SOURCE_TABLEBASE
            )
            target = ValueTarget(
                features=extract_features(game),
                value=float(scenario.expected_value),
                source=source,
                horizon=0,
                dropper_dist=drop,
                checker_dist=check,
                dropper_legal_mask=drop_mask,
                checker_legal_mask=check_mask,
                exact_state=exact_public_state(game),
                target_kind="tablebase_value",
                state_origin=StateOrigin.TACTICAL_TABLEBASE.value,
                source_artifact=f"stl/solver/tablebase.py::{name}",
                source_artifact_digest=source_digest,
                episode_id=f"tactical-{name}",
            )
            (ruler if index < ruler_count else train).append(target)
    _log(f"tactical split ready: train={len(train)}, external_test={len(ruler)}")
    return train, [], ruler, {"train": {}, "development": {}, "ruler": {}}


def _tier_a_records(
    args: argparse.Namespace,
    *,
    blocked_hashes: set[str],
    search_digest: str,
) -> list[TrainingRecordV3]:
    requested = _tier_a_row_count(args)
    if requested == 0:
        _log("Tier A selection disabled: requested 0 rows")
        return []
    _log(
        f"selecting {requested} certified Tier A value rows "
        f"with interval width <= {float(args.tier_a_max_width):.6f}"
    )
    tier_args = SimpleNamespace(
        tier_a_dir=args.tier_a_dir,
        verify_manifest=True,
        runtime_width=0.0,
        policy_horizon=0,
        max_width=float(args.tier_a_max_width),
        cylinder_points=4 if args.smoke else 24,
        near_boundary_points=4 if args.smoke else 30,
        ttd_points=2 if args.smoke else 10,
        ttd_min=None,
        ttd_max=None,
        death_filter="all",
        include_d1=True,
        limit=max(requested * 4, requested + 8),
        seed=int(args.split_seed),
        source="tier_a",
    )
    targets, _stats = tier_a_targets.generate_targets(tier_args)
    _log(f"Tier A generator offered {len(targets)} candidate rows")
    accepted: list[TrainingRecordV3] = []
    seen = set(blocked_hashes)
    for target in targets:
        state_hash = exact_state_hash(target.exact_state)
        if state_hash in seen:
            continue
        seen.add(state_hash)
        accepted.append(
            to_training_record_v3(
                target,
                search_config_digest=search_digest,
                rng_seeds={"tier_a_sampling": int(args.split_seed)},
            )
        )
        if len(accepted) == requested:
            break
    if len(accepted) != requested:
        raise RuntimeError(
            f"Tier A supplied {len(accepted)} disjoint rows; {requested} were requested"
        )
    _log(f"Tier A selection complete: accepted={len(accepted)}, policy=inactive")
    return accepted


def _materialize_chunk(
    path: Path,
    *,
    role: ShardRole,
    provenance: dict[str, object],
    factory: Callable[[], Sequence[TrainingRecordV3]],
) -> list[TrainingRecordV3]:
    manifest_path = manifest_path_for(path)
    if manifest_path.exists():
        _log(f"validating resumable component: {path.name}")
        if not path.exists():
            raise ReplayValidationError(f"committed chunk payload is missing: {path}")
        records = load_replay_shard(path, for_training=role is ShardRole.REPLAY)
        manifest = load_replay_manifest(path)
        if manifest.get("generation_provenance") != provenance:
            raise ReplayValidationError(f"committed chunk plan mismatch: {path}")
        _log(f"resumed verified component: {path.name} ({len(records)} rows)")
        return records
    if path.exists():
        # A payload without the manifest commit marker is an interrupted write.
        path.unlink()
        _log(f"removed uncommitted partial component: {path.name}")
    _log(f"building component: {path.name}")
    records = list(factory())
    save_replay_shard(
        records,
        path,
        shard_role=role,
        generation_provenance=provenance,
    )
    _log(f"committed component: {path.name} ({len(records)} rows)")
    return records


def _materialize_trajectory_chunk(
    path: Path,
    *,
    role: ShardRole,
    provenance: dict[str, object],
    factory: Callable[[], tuple[list[TrainingRecordV3], list[ExactPolicyCertificate]]],
) -> tuple[list[TrainingRecordV3], list[ExactPolicyCertificate]]:
    """Commit one replay/certificate pair with a final bundle marker."""

    certificate_path = path.with_name(f"{path.stem}.certificates.npz")
    marker_path = path.with_name(f"{path.name}.bundle.json")
    artifacts = (
        path,
        manifest_path_for(path),
        certificate_path,
        certificate_manifest_path(certificate_path),
    )
    if marker_path.exists():
        _log(f"validating resumable trajectory bundle: {path.name}")
        marker = marker_path.read_bytes()
        expected_marker = _canonical_bytes(
            {
                "replay_manifest_sha256": _sha256_file(manifest_path_for(path)),
                "certificate_manifest_sha256": _sha256_file(
                    certificate_manifest_path(certificate_path)
                ),
            }
        )
        if marker != expected_marker:
            raise ReplayValidationError(f"trajectory bundle marker mismatch: {path}")
        records = load_replay_shard(path, expected_role=role)
        certificates = load_certificate_shard(certificate_path, expected_role=role)
        if load_replay_manifest(path).get("generation_provenance") != provenance:
            raise ReplayValidationError(f"committed chunk plan mismatch: {path}")
        _log(
            f"resumed trajectory bundle: {path.name} "
            f"({len(records)} rows, {len(certificates)} exact matrices)"
        )
        return records, certificates
    if any(artifact.exists() for artifact in artifacts):
        for artifact in artifacts:
            artifact.unlink(missing_ok=True)
        _log(f"removed incomplete trajectory bundle: {path.name}")

    _log(f"building trajectory bundle: {path.name}")
    records, certificates = factory()
    record_hashes = {
        exact_state_hash(record.exact_state)
        for record in records
        if record.value_horizon_half_rounds > 0
    }
    certificate_hashes = {certificate.state_hash for certificate in certificates}
    if record_hashes != certificate_hashes:
        raise RuntimeError(
            "exact replay rows and policy certificates do not cover the same states"
        )
    try:
        save_replay_shard(
            records,
            path,
            shard_role=role,
            generation_provenance=provenance,
        )
        save_certificate_shard(
            certificates,
            certificate_path,
            shard_role=role,
            generation_provenance=provenance,
        )
        marker_path.write_bytes(
            _canonical_bytes(
                {
                    "replay_manifest_sha256": _sha256_file(manifest_path_for(path)),
                    "certificate_manifest_sha256": _sha256_file(
                        certificate_manifest_path(certificate_path)
                    ),
                }
            )
        )
    except Exception:
        marker_path.unlink(missing_ok=True)
        for artifact in artifacts:
            artifact.unlink(missing_ok=True)
        raise
    _log(
        f"committed trajectory bundle: {path.name} "
        f"({len(records)} rows, {len(certificates)} exact matrices)"
    )
    return records, certificates


def _feature_hash(record: TrainingRecordV3) -> str:
    return hashlib.sha256(np.ascontiguousarray(record.features).tobytes()).hexdigest()


def _blocked_inventory(
    paths: Sequence[str | Path],
) -> tuple[set[str], set[str], set[str], dict[str, str]]:
    state_hashes: set[str] = set()
    episode_ids: set[str] = set()
    feature_hashes: set[str] = set()
    digests: dict[str, str] = {}
    for raw_path in paths:
        path = Path(raw_path)
        records = load_replay_shard(path)
        state_hashes.update(exact_state_hash(record.exact_state) for record in records)
        episode_ids.update(record.episode_id for record in records if record.episode_id)
        feature_hashes.update(_feature_hash(record) for record in records)
        digests[str(path)] = _sha256_file(path)
    return state_hashes, episode_ids, feature_hashes, digests


def _reused_exact_inventory(
    replay_path: str | Path,
    *,
    expected_role: ShardRole,
) -> tuple[list[TrainingRecordV3], list[ExactPolicyCertificate]]:
    records = [
        record
        for record in load_replay_shard(replay_path, expected_role=expected_role)
        if record.source in {"exact_horizon_2", "exact_horizon_3", "terminal"}
    ]
    certificate_path = _certificate_output_path(replay_path)
    certificates = load_certificate_shard(
        certificate_path,
        expected_role=expected_role,
    )
    expected_counts = (
        {"exact_horizon_2": 2_048, "exact_horizon_3": 1_024, "terminal": 128}
        if expected_role is ShardRole.REPLAY
        else {"exact_horizon_2": 384, "exact_horizon_3": 192, "terminal": 32}
    )
    observed = Counter(record.source for record in records)
    if any(observed[source] != count for source, count in expected_counts.items()):
        raise ValueError(
            f"reused exact inventory has wrong counts: {dict(observed)}; "
            f"expected={expected_counts}"
        )
    active_hashes = {
        exact_state_hash(record.exact_state)
        for record in records
        if record.target_kind is TargetKind.EXACT_VALUE
    }
    if {certificate.state_hash for certificate in certificates} != active_hashes:
        raise ValueError("reused exact certificates do not cover active rows exactly")
    return records, list(certificates)


def _certificate_rank(record: TrainingRecordV3) -> int:
    if record.target_kind is TargetKind.TERMINAL_OUTCOME:
        return 0
    if record.state_origin is StateOrigin.TACTICAL_TABLEBASE:
        return 1
    if record.target_kind is TargetKind.EXACT_VALUE:
        return 2
    if record.state_origin is StateOrigin.TIER_A:
        return 3
    return 4


def _dedupe_by_certificate(
    records: Sequence[TrainingRecordV3],
) -> tuple[list[TrainingRecordV3], dict[str, int]]:
    """Resolve repeated exact states by declared certificate precedence."""

    selected: dict[str, TrainingRecordV3] = {}
    order: list[str] = []
    decisions: Counter[str] = Counter()
    for record in records:
        state_hash = exact_state_hash(record.exact_state)
        previous = selected.get(state_hash)
        if previous is None:
            selected[state_hash] = record
            order.append(state_hash)
            continue
        if abs(previous.value - record.value) > 1e-6:
            raise RuntimeError(
                "conflicting labels offered for exact state "
                f"{state_hash}: {previous.source}={previous.value}, "
                f"{record.source}={record.value}"
            )
        if _certificate_rank(record) < _certificate_rank(previous):
            selected[state_hash] = record
            decisions[f"{record.source}_over_{previous.source}"] += 1
        else:
            decisions[f"{previous.source}_over_{record.source}"] += 1
    return [selected[state_hash] for state_hash in order], dict(decisions)


def _assert_pair_isolated(
    train: Sequence[TrainingRecordV3], ruler: Sequence[TrainingRecordV3]
) -> None:
    dimensions = {
        "exact-state": (
            {exact_state_hash(record.exact_state) for record in train},
            {exact_state_hash(record.exact_state) for record in ruler},
        ),
        "episode": (
            {record.episode_id for record in train},
            {record.episode_id for record in ruler},
        ),
        "feature": (
            {_feature_hash(record) for record in train},
            {_feature_hash(record) for record in ruler},
        ),
    }
    for name, (train_values, ruler_values) in dimensions.items():
        overlap = train_values & ruler_values
        if overlap:
            raise RuntimeError(f"Gen-0 train/ruler {name} overlap: {sorted(overlap)}")
    if audit_feature_collisions(train) or audit_feature_collisions(ruler):
        raise RuntimeError(
            "Gen-0 contains divergent targets for a collapsed feature vector"
        )


def _assert_roles_isolated(
    roles: dict[str, Sequence[TrainingRecordV3]],
) -> None:
    names = tuple(roles)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            try:
                _assert_pair_isolated(roles[left], roles[right])
            except RuntimeError as exc:
                raise RuntimeError(
                    f"Gen-0 {left}/{right} isolation failed: {exc}"
                ) from exc


def _certificate_output_path(replay_path: str | Path) -> Path:
    path = Path(replay_path)
    return path.with_name(f"{path.stem}.certificates.npz")


def _publish_or_resume_certificates(
    path: Path,
    certificates: Sequence[ExactPolicyCertificate],
    *,
    role: ShardRole,
    provenance: dict[str, object],
) -> dict[str, object]:
    manifest_path = certificate_manifest_path(path)
    if path.exists() and manifest_path.exists():
        loaded = load_certificate_shard(path, expected_role=role)
        if [item.state_hash for item in loaded] != [
            item.state_hash for item in certificates
        ]:
            raise ReplayValidationError(
                f"published certificate shard row order mismatch: {path}"
            )
        _log(f"verified previously published {role.value} certificates: {path}")
        import json

        return json.loads(manifest_path.read_text(encoding="utf-8"))
    if path.exists() or manifest_path.exists():
        raise ReplayValidationError(
            f"partial final certificate shard requires manual audit: {path}"
        )
    manifest = save_certificate_shard(
        certificates,
        path,
        shard_role=role,
        generation_provenance=provenance,
    )
    _log(
        f"published {role.value} exact-policy certificates: "
        f"{len(certificates)} matrices -> {path}"
    )
    return manifest


def _publish_or_resume_final(
    path: Path,
    records: Sequence[TrainingRecordV3],
    *,
    role: ShardRole,
    provenance: dict[str, object],
) -> dict[str, object]:
    manifest_path = manifest_path_for(path)
    if path.exists() and manifest_path.exists():
        _log(f"validating previously published {role.value} output: {path}")
        loaded = load_replay_shard(path, for_training=role is ShardRole.REPLAY)
        manifest = load_replay_manifest(path)
        if manifest.get("generation_provenance") != provenance:
            raise ReplayValidationError(f"published final shard plan mismatch: {path}")
        if [exact_state_hash(row.exact_state) for row in loaded] != [
            exact_state_hash(row.exact_state) for row in records
        ]:
            raise ReplayValidationError(
                f"published final shard row order mismatch: {path}"
            )
        _log(f"verified previously published {role.value} output: {path}")
        return manifest
    if path.exists() or manifest_path.exists():
        raise ReplayValidationError(
            f"partial final shard requires manual audit: {path}"
        )
    _log(f"publishing {role.value} output: {path}")
    manifest = save_replay_shard(
        records,
        path,
        shard_role=role,
        generation_provenance=provenance,
    )
    _log(
        f"published {role.value} output: {len(records)} rows, "
        f"sha256={str(manifest['data_sha256'])[:12]}..."
    )
    return manifest


def generate_gen0_pair(args: argparse.Namespace) -> dict[str, object]:
    started_at = perf_counter()
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    if args.chunk_size <= 0:
        raise ValueError("chunk-size must be positive")
    if args.smoke and args.profile != "baseline":
        raise ValueError("--smoke applies only to the baseline profile")
    if _tier_a_row_count(args) < 0:
        raise ValueError("tier-a-rows must be non-negative")
    repo_root = Path(__file__).resolve().parents[2]
    work_dir = Path(args.work_dir)
    _log(
        f"starting Generation Zero ({'smoke' if args.smoke else args.profile}): "
        f"split_seed={int(args.split_seed)}, workers={int(args.workers)}"
    )
    _log(f"training output: {Path(args.train_out)}")
    _log(f"development output: {Path(args.dev_out)}")
    _log(f"external test output: {Path(args.ruler_out)}")
    _log(f"resumable work directory: {work_dir}")
    _log(
        f"Tier A source: {Path(args.tier_a_dir)} "
        f"(requested rows={_tier_a_row_count(args)})"
    )
    with _exclusive_lock(work_dir):
        _log(f"acquired exclusive generation lock: {work_dir / 'generation.lock'}")
        plan = _resolved_plan(args, repo_root)
        plan_digest = _write_or_validate_plan(work_dir, plan)
        _log(
            f"plan digest={plan_digest[:12]}..., "
            f"source tree={str(plan['source_tree_sha256'])[:12]}..., "
            f"git={str(plan['git_head'])[:12] or 'unknown'}"
        )
        search_config = ExactSearchConfig()
        search_digest = config_digest(_search_config_payload(search_config))
        base_provenance = {
            "schema": GENERATION_PROVENANCE_SCHEMA,
            "plan_digest": plan_digest,
            "source_tree_sha256": plan["source_tree_sha256"],
            "split_seed": int(args.split_seed),
            "tier_a_manifest_sha256": plan["tier_a"]["manifest_sha256"],
        }
        blocked_paths = tuple(getattr(args, "blocked_artifact", None) or ())
        if args.profile == "scaled_v4" and not blocked_paths:
            raise ValueError(
                "scaled_v4 requires --blocked-artifact for the consumed V3 "
                "training, development, and ruler shards"
            )
        (
            blocked_state_hashes,
            blocked_episode_ids,
            blocked_feature_hashes,
            blocked_digests,
        ) = _blocked_inventory(blocked_paths)
        if blocked_paths:
            _log(
                "loaded blocked prior inventory: "
                f"states={len(blocked_state_hashes)}, "
                f"episodes={len(blocked_episode_ids)}, "
                f"features={len(blocked_feature_hashes)}"
            )

        if args.profile == "baseline":
            baseline_reachable = split_reachable_candidates(
                smoke=bool(args.smoke), split_seed=int(args.split_seed)
            )
            reachable_roles = {
                "train": baseline_reachable.train,
                "development": (),
                "ruler": baseline_reachable.ruler,
            }
        else:
            quotas = SCALED_REACHABLE_QUOTAS[args.profile]
            scaled_reachable = build_scaled_reachable_split(
                train_quota=quotas["train"],
                development_quota=quotas["development"],
                ruler_quota=quotas["ruler"],
                split_seed=int(args.split_seed),
                blocked_state_hashes=(
                    blocked_state_hashes if args.profile == "scaled_v4" else None
                ),
                blocked_episode_ids=(
                    blocked_episode_ids if args.profile == "scaled_v4" else None
                ),
                blocked_feature_hashes=(
                    blocked_feature_hashes if args.profile == "scaled_v4" else None
                ),
            )
            reachable_roles = {
                "train": scaled_reachable.train,
                "development": scaled_reachable.development,
                "ruler": scaled_reachable.ruler,
            }
        _log(
            "legal engine positions ready: "
            + ", ".join(f"{role}={len(rows)}" for role, rows in reachable_roles.items())
        )
        (
            tactical_train,
            tactical_development,
            tactical_ruler,
            tactical_taxonomies,
        ) = _tactical_targets(
            args=args,
            split_seed=int(args.split_seed),
            search_config=search_config,
            blocked_ruler_state_hashes=(
                blocked_state_hashes if args.profile == "scaled_v4" else None
            ),
        )
        tactical_roles = {
            "train": tactical_train,
            "development": tactical_development,
            "ruler": tactical_ruler,
        }
        role_enums = {
            "train": ShardRole.REPLAY,
            "development": ShardRole.DEVELOPMENT,
            "ruler": ShardRole.EXTERNAL_RULER,
        }
        role_records: dict[str, list[TrainingRecordV3]] = {
            role: [] for role in role_enums
        }
        role_certificates: dict[str, list[ExactPolicyCertificate]] = {
            role: [] for role in role_enums
        }
        if args.profile == "scaled_v4":
            reuse_paths = {
                "train": getattr(args, "reuse_train_artifact", None),
                "development": getattr(args, "reuse_dev_artifact", None),
            }
            if any(path is None for path in reuse_paths.values()):
                raise ValueError(
                    "scaled_v4 requires --reuse-train-artifact and "
                    "--reuse-dev-artifact"
                )
            for role, path in reuse_paths.items():
                records, certificates = _reused_exact_inventory(
                    path,
                    expected_role=role_enums[role],
                )
                role_records[role].extend(records)
                role_certificates[role].extend(certificates)
                _log(
                    f"reused certified {role} exact/terminal inventory: "
                    f"rows={len(records)}, certificates={len(certificates)}"
                )
        for role, snapshots in reachable_roles.items():
            for chunk_index, offset in enumerate(
                range(0, len(snapshots), int(args.chunk_size))
            ):
                chunk = snapshots[offset : offset + int(args.chunk_size)]
                name = f"{role}.trajectory.{chunk_index:06d}"
                provenance = {**base_provenance, "component": name}
                records, certificates = _materialize_trajectory_chunk(
                    work_dir / f"{name}.npz",
                    role=role_enums[role],
                    provenance=provenance,
                    factory=lambda chunk=chunk: _trajectory_records(
                        chunk,
                        search_config=search_config,
                        search_digest=search_digest,
                        workers=int(args.workers),
                    ),
                )
                role_records[role].extend(records)
                role_certificates[role].extend(certificates)

        for role, targets in tactical_roles.items():
            name = f"{role}.tactical"
            provenance = {**base_provenance, "component": name}
            role_records[role].extend(
                _materialize_chunk(
                    work_dir / f"{name}.000000.npz",
                    role=role_enums[role],
                    provenance=provenance,
                    factory=lambda targets=targets: [
                        to_training_record_v3(
                            target, search_config_digest=search_digest
                        )
                        for target in targets
                    ],
                )
            )

        deduped_roles: dict[str, list[TrainingRecordV3]] = {}
        dedupe_decisions: dict[str, dict[str, int]] = {}
        for role, records in role_records.items():
            deduped_roles[role], dedupe_decisions[role] = _dedupe_by_certificate(
                records
            )
        pre_tier_train = deduped_roles["train"]
        development = deduped_roles["development"]
        ruler = deduped_roles["ruler"]
        blocked_hashes = {
            exact_state_hash(record.exact_state)
            for record in [*pre_tier_train, *development, *ruler]
        }
        tier_provenance = {**base_provenance, "component": "train.tier_a"}
        tier_records = _materialize_chunk(
            work_dir / "train.tier_a.000000.npz",
            role=ShardRole.REPLAY,
            provenance=tier_provenance,
            factory=lambda: _tier_a_records(
                args,
                blocked_hashes=blocked_hashes,
                search_digest=search_digest,
            ),
        )
        train, final_train_dedupe = _dedupe_by_certificate(
            [*pre_tier_train, *tier_records]
        )
        _log(
            "checking final separation and feature consistency: "
            f"train={len(train)}, development={len(development)}, "
            f"external_test={len(ruler)}"
        )
        final_roles = {
            "train": train,
            "development": development,
            "ruler": ruler,
        }
        _assert_roles_isolated(final_roles)
        _log("final separation passed: no state, episode, or feature overlap")

        chunk_paths = sorted(work_dir.glob("*.000000.npz.manifest.json"))
        parent_chunks = [
            {
                "manifest": path.relative_to(work_dir).as_posix(),
                "sha256": _sha256_file(path),
            }
            for path in chunk_paths
        ]
        common_final = {
            **base_provenance,
            "parent_chunks": parent_chunks,
            "source_counts": {
                role: dict(Counter(record.source for record in records))
                for role, records in final_roles.items()
            },
            "certificate_dedupe": {
                **dedupe_decisions,
                "train_final": final_train_dedupe,
            },
        }
        output_paths = {
            "train": Path(args.train_out),
            "development": Path(args.dev_out),
            "ruler": Path(args.ruler_out),
        }
        manifests: dict[str, dict[str, object]] = {}
        certificate_manifests: dict[str, dict[str, object]] = {}
        taxonomy_paths: dict[str, Path] = {}
        for role, records in final_roles.items():
            final_provenance = {**common_final, "component": f"final.{role}"}
            manifests[role] = _publish_or_resume_final(
                output_paths[role],
                records,
                role=role_enums[role],
                provenance=final_provenance,
            )
            by_hash = {
                certificate.state_hash: certificate
                for certificate in role_certificates[role]
            }
            final_certificates = [
                by_hash[exact_state_hash(record.exact_state)]
                for record in records
                if record.value_horizon_half_rounds > 0
            ]
            certificate_manifests[role] = _publish_or_resume_certificates(
                _certificate_output_path(output_paths[role]),
                final_certificates,
                role=role_enums[role],
                provenance={
                    **final_provenance,
                    "artifact": "exact-policy-certificates",
                },
            )
            taxonomy_paths[role] = write_taxonomy(
                output_paths[role], tactical_taxonomies[role]
            )
            _log(
                f"{role} rows ready: values={len(records)}, "
                f"active_policies={_active_policy_rows(records)}, "
                f"inactive_policies={len(records) - _active_policy_rows(records)}"
            )
            _log(f"{role} sources: {_source_summary(records)}")
        seal_payload = None
        if args.profile == "scaled_v4":
            seal_payload = write_holdout_seal(
                getattr(
                    args,
                    "holdout_seal_out",
                    "stl/outputs/regen2rl/gen0_v4_holdout.seal.json",
                ),
                holdout_path=output_paths["ruler"],
                certificate_path=_certificate_output_path(output_paths["ruler"]),
                taxonomy_path=taxonomy_paths["ruler"],
                generation_plan_digest=plan_digest,
                blocked_artifacts=blocked_digests,
            )
            _log(
                "sealed holdout: "
                f"{getattr(args, 'holdout_seal_out', 'stl/outputs/regen2rl/gen0_v4_holdout.seal.json')} "
                f"digest={str(seal_payload['seal_digest'])[:12]}..."
            )
        _log(f"Generation Zero complete in {perf_counter() - started_at:.1f}s")
        return {
            "plan_digest": plan_digest,
            "holdout_seal": seal_payload,
            **{
                role: {
                    "path": str(output_paths[role]),
                    "certificate_path": str(
                        _certificate_output_path(output_paths[role])
                    ),
                    "taxonomy_path": str(taxonomy_paths[role]),
                    "records": len(records),
                    "sources": dict(Counter(record.source for record in records)),
                    "manifest": manifests[role],
                    "certificate_manifest": certificate_manifests[role],
                }
                for role, records in final_roles.items()
            },
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-out", default="stl/outputs/regen2rl/gen0_train_v3.npz")
    parser.add_argument("--dev-out", default="stl/outputs/regen2rl/gen0_development_v3.npz")
    parser.add_argument(
        "--ruler-out", default="stl/outputs/regen2rl/gen0_external_ruler_v3.npz"
    )
    parser.add_argument("--work-dir", default="stl/outputs/regen2rl/gen0_work_v3")
    parser.add_argument("--tier-a-dir", default="stl/checkpoints/tablebase/tier_a")
    parser.add_argument("--tier-a-max-width", type=float, default=0.05)
    parser.add_argument(
        "--tier-a-rows",
        type=int,
        default=None,
        help="Training-only Tier A rows (default: 2 for --smoke, otherwise 32).",
    )
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument(
        "--blocked-artifact",
        action="append",
        default=None,
        help="Prior replay shard excluded from a fresh V4 holdout.",
    )
    parser.add_argument("--reuse-train-artifact", default=None)
    parser.add_argument("--reuse-dev-artifact", default=None)
    parser.add_argument(
        "--holdout-seal-out",
        default="stl/outputs/regen2rl/gen0_v4_holdout.seal.json",
    )
    parser.add_argument(
        "--profile",
        choices=("baseline", "scaled_pilot", "scaled_full", "scaled_v4"),
        default="baseline",
    )
    parser.add_argument(
        "--smoke",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use one paired exact trajectory and two Tier-A rows.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = generate_gen0_pair(args)
    print(
        f"Gen-0 V3 train: {summary['train']['records']} records -> "
        f"{summary['train']['path']}"
    )
    print(
        f"Development set: {summary['development']['records']} records -> "
        f"{summary['development']['path']}"
    )
    print(
        f"External ruler: {summary['ruler']['records']} records -> "
        f"{summary['ruler']['path']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
