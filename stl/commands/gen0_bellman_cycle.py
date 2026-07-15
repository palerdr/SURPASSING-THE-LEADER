"""Generate the horizon-aware Bellman closure for the next Gen-0 cycle."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np

from stl.learning.bellman import (
    BellmanBundle,
    BellmanGateThresholds,
    BellmanRootSpec,
    bellman_closure_hashes,
    build_bellman_bundle,
    bundle_training_records,
    candidate_action_representability,
    load_bellman_bundle,
    merge_bellman_bundles,
    save_bellman_bundle,
    select_causal_roots,
)
from stl.learning.certificates import (
    ExactPolicyCertificate,
    load_certificate_shard,
    save_certificate_shard,
)
from stl.learning.holdout import sha256_file, write_bellman_holdout_seal
from stl.learning.replay import (
    ShardRole,
    TrainingRecordV3,
    exact_state_hash,
    load_replay_shard,
    save_replay_shard,
)
from stl.solver.mcts_conformance import frozen_horizon_one_scenarios
from stl.solver.exact import exact_public_state


ROLE_COUNTS = {
    ShardRole.REPLAY: 96,
    ShardRole.DEVELOPMENT: 32,
    ShardRole.EXTERNAL_RULER: 32,
}


@dataclass(frozen=True)
class RootCacheEntry:
    bellman_path: Path
    replay_path: Path
    certificate_path: Path
    source_commit: str
    source_role: ShardRole
    binding_sha256: str


def _artifact_paths(prefix: str | Path, role: ShardRole) -> dict[str, Path]:
    base = Path(prefix)
    stem = f"{base.name}_{role.value}"
    return {
        "replay": base.with_name(stem + ".npz"),
        "certificates": base.with_name(stem + ".certificates.npz"),
        "bellman": base.with_name(stem + ".bellman.npz"),
    }


def _chunk_paths(
    work_dir: str | Path, role: ShardRole, index: int
) -> dict[str, Path]:
    stem = Path(work_dir) / f"{role.value}.root.{index:06d}"
    return {
        "replay": stem.with_name(stem.name + ".npz"),
        "certificates": stem.with_name(stem.name + ".certificates.npz"),
        "bellman": stem.with_name(stem.name + ".bellman.npz"),
        "commit": stem.with_name(stem.name + ".commit.json"),
    }


def _manifest_path(path: Path) -> Path:
    return path.with_name(path.name + ".manifest.json")


def _artifact_state(paths: dict[str, Path]) -> str:
    required = [item for path in paths.values() for item in (path, _manifest_path(path))]
    present = [path.exists() for path in required]
    if all(present):
        return "complete"
    if any(present):
        raise ValueError(
            "partial Bellman artifact set; preserve it for diagnosis and remove "
            f"only after review: {paths}"
        )
    return "missing"


def _chunk_state(paths: dict[str, Path]) -> str:
    payload = {name: path for name, path in paths.items() if name != "commit"}
    required = [
        item for path in payload.values() for item in (path, _manifest_path(path))
    ]
    commit = paths["commit"]
    if commit.exists():
        if not all(path.exists() for path in required):
            raise ValueError("committed Bellman root chunk is incomplete")
        descriptor = json.loads(commit.read_text(encoding="utf-8"))
        expected = {
            str(path.name): sha256_file(path) for path in required
        }
        if descriptor.get("files") != expected:
            raise ValueError("Bellman root chunk commit digest mismatch")
        return "complete"
    if any(path.exists() for path in required):
        raise ValueError(
            "uncommitted partial Bellman root chunk found; preserve it for "
            f"diagnosis before cleanup: {paths}"
        )
    return "missing"


def _commit_chunk(paths: dict[str, Path]) -> None:
    payload = {name: path for name, path in paths.items() if name != "commit"}
    required = [
        item for path in payload.values() for item in (path, _manifest_path(path))
    ]
    descriptor = {
        "schema": "stl.gen0-bellman-root-commit.v1",
        "files": {str(path.name): sha256_file(path) for path in required},
    }
    destination = paths["commit"]
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(descriptor, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def _load_root_cache(work_dirs: list[str]) -> dict[str, RootCacheEntry]:
    """Strictly load committed singleton roots for reuse under a new split."""

    cache: dict[str, RootCacheEntry] = {}
    pattern = re.compile(
        r"^(replay|development|external_ruler)\.root\.(\d{6})\.commit\.json$"
    )
    for raw_dir in work_dirs:
        work_dir = Path(raw_dir)
        if not work_dir.exists():
            raise ValueError(f"Bellman reuse work directory does not exist: {work_dir}")
        plan_path = work_dir / "plan.json"
        if not plan_path.exists():
            raise ValueError(f"Bellman reuse plan does not exist: {plan_path}")
        plan_bytes = plan_path.read_bytes()
        try:
            plan = json.loads(plan_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid Bellman reuse plan: {plan_path}") from exc
        roots_by_role = plan.get("roots")
        if not isinstance(roots_by_role, dict):
            raise ValueError(f"Bellman reuse plan has no root inventory: {plan_path}")
        plan_sha256 = hashlib.sha256(plan_bytes).hexdigest()
        for commit_path in sorted(work_dir.glob("*.commit.json")):
            match = pattern.match(commit_path.name)
            if match is None:
                continue
            source_role = ShardRole(match.group(1))
            index = int(match.group(2))
            paths = _chunk_paths(work_dir, source_role, index)
            if _chunk_state(paths) != "complete":
                raise ValueError(f"Bellman cache chunk is not committed: {commit_path}")
            role_roots = roots_by_role.get(source_role.value)
            if not isinstance(role_roots, list) or index >= len(role_roots):
                raise ValueError(
                    f"Bellman reuse chunk is absent from its plan: {commit_path}"
                )
            root_hash = str(role_roots[index])
            commit_sha256 = sha256_file(commit_path)
            binding_sha256 = hashlib.sha256(
                (
                    f"{plan_sha256}:{source_role.value}:{index}:"
                    f"{root_hash}:{commit_sha256}"
                ).encode("ascii")
            ).hexdigest()
            entry = RootCacheEntry(
                bellman_path=paths["bellman"],
                replay_path=paths["replay"],
                certificate_path=paths["certificates"],
                source_commit=str(commit_path),
                source_role=source_role,
                binding_sha256=binding_sha256,
            )
            if root_hash in cache:
                raise ValueError(f"duplicate exact root in Bellman caches: {root_hash}")
            cache[root_hash] = entry
    return cache


@contextmanager
def _exclusive_generation_lock(work_dir: Path):
    work_dir.mkdir(parents=True, exist_ok=True)
    lock_path = work_dir / "generation.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise ValueError(f"Bellman generation lock already exists: {lock_path}") from exc
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as stream:
            stream.write(str(os.getpid()))
        yield
    finally:
        lock_path.unlink(missing_ok=True)


def _write_or_validate_plan(
    work_dir: Path,
    *,
    split_seed: int,
    blocked_digests: dict[str, str],
    role_specs: dict[ShardRole, tuple[BellmanRootSpec, ...]],
    role_closures: dict[ShardRole, set[str]],
    reusable_root_chunks: dict[str, str],
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    source_paths = [
        *repo_root.glob("stl/**/*.py"),
        *repo_root.glob("configs/**/*.yaml"),
        repo_root / "pyproject.toml",
        repo_root / "uv.lock",
    ]
    source_digest = hashlib.sha256()
    for source_path in sorted(
        {path for path in source_paths if path.is_file()}
    ):
        relative = source_path.relative_to(repo_root).as_posix()
        source_digest.update(relative.encode("utf-8"))
        source_digest.update(b"\0")
        source_digest.update(source_path.read_bytes())
        source_digest.update(b"\0")
    plan = {
        "schema": "stl.gen0-bellman-resume-plan.v2",
        "split_seed": int(split_seed),
        "source_tree_sha256": source_digest.hexdigest(),
        "blocked_artifacts": blocked_digests,
        "reusable_root_chunks": reusable_root_chunks,
        "roots": {
            role.value: [
                exact_state_hash(exact_public_state(spec.game)) for spec in specs
            ]
            for role, specs in role_specs.items()
        },
        "closure_reservations": {
            role.value: {
                "state_count": len(role_closures[role]),
                "sha256": hashlib.sha256(
                    json.dumps(
                        sorted(role_closures[role]), separators=(",", ":")
                    ).encode("ascii")
                ).hexdigest(),
            }
            for role in role_specs
        },
    }
    encoded = json.dumps(plan, sort_keys=True, separators=(",", ":")) + "\n"
    path = work_dir / "plan.json"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise ValueError("Bellman resume plan differs from the committed plan")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, path)


def _blocked_hashes(paths: list[str]) -> tuple[set[str], dict[str, str]]:
    hashes: set[str] = set()
    digests: dict[str, str] = {}
    for raw_path in paths:
        path = Path(raw_path)
        records = load_replay_shard(path, for_training=False)
        hashes.update(exact_state_hash(record.exact_state) for record in records)
        digests[str(path)] = sha256_file(path)
    return hashes, digests


def _fixture_specs() -> tuple[BellmanRootSpec, ...]:
    return tuple(
        BellmanRootSpec(
            name=scenario.name,
            game=scenario.game,
            strata=(("fixture", scenario.name),),
        )
        for scenario in frozen_horizon_one_scenarios()
    )


def _mcts_subset_hashes(bundle, count: int = 8) -> list[str]:
    remaining = sorted(bundle.roots, key=lambda row: row.state_hash)
    selected = []
    covered: set[tuple[str, str]] = set()
    names = {"checker", "checker_band", "clock_class", "history_severity"}
    while remaining and len(selected) < count:
        best = max(
            remaining,
            key=lambda row: (
                len(
                    {
                        pair
                        for pair in row.strata
                        if pair[0] in names and pair not in covered
                    }
                ),
                -int(row.state_hash, 16),
            ),
        )
        selected.append(best)
        remaining.remove(best)
        covered.update(pair for pair in best.strata if pair[0] in names)
    if len(selected) != count:
        raise ValueError("sealed MCTS subset could not reach its fixed size")
    return [row.state_hash for row in selected]


def _records_equivalent(left: TrainingRecordV3, right: TrainingRecordV3) -> bool:
    return all(
        (
            left.exact_state == right.exact_state,
            left.value_horizon_half_rounds == right.value_horizon_half_rounds,
            left.target_kind == right.target_kind,
            left.source == right.source,
            left.search_config_digest == right.search_config_digest,
            np.isclose(left.value, right.value, atol=1.0e-8, rtol=0.0),
            np.isclose(
                left.cutoff_probability,
                right.cutoff_probability,
                atol=1.0e-8,
                rtol=0.0,
            ),
            np.array_equal(left.features, right.features),
            np.allclose(left.dropper_dist, right.dropper_dist, atol=1.0e-8),
            np.allclose(left.checker_dist, right.checker_dist, atol=1.0e-8),
            np.array_equal(left.dropper_legal_mask, right.dropper_legal_mask),
            np.array_equal(left.checker_legal_mask, right.checker_legal_mask),
        )
    )


def _certificates_equivalent(
    left: ExactPolicyCertificate, right: ExactPolicyCertificate
) -> bool:
    return all(
        (
            left.state_hash == right.state_hash,
            left.search_config_digest == right.search_config_digest,
            left.horizon == right.horizon,
            left.drop_actions == right.drop_actions,
            left.check_actions == right.check_actions,
            np.isclose(left.value_for_hal, right.value_for_hal, atol=1.0e-8),
            np.isclose(
                left.unresolved_probability,
                right.unresolved_probability,
                atol=1.0e-8,
            ),
            np.allclose(left.payoff_for_hal, right.payoff_for_hal, atol=1.0e-8),
            np.allclose(left.dropper_strategy, right.dropper_strategy, atol=1.0e-8),
            np.allclose(left.checker_strategy, right.checker_strategy, atol=1.0e-8),
        )
    )


def _merge_role_records(
    chunks: list[list[TrainingRecordV3]],
    *,
    role: ShardRole,
    plan_digest: str,
) -> list[TrainingRecordV3]:
    merged: dict[tuple[str, int], TrainingRecordV3] = {}
    for rows in chunks:
        for row in rows:
            normalized = replace(
                row,
                source_artifact=f"gen0-bellman-{role.value}",
                source_artifact_digest=plan_digest,
            )
            key = (
                exact_state_hash(normalized.exact_state),
                normalized.value_horizon_half_rounds,
            )
            existing = merged.get(key)
            if existing is not None:
                if not _records_equivalent(existing, normalized):
                    raise ValueError("duplicate Bellman training row disagrees")
                continue
            merged[key] = normalized
    return list(merged.values())


def _merge_role_certificates(
    chunks: list[list[ExactPolicyCertificate]],
) -> list[ExactPolicyCertificate]:
    merged: dict[str, ExactPolicyCertificate] = {}
    for rows in chunks:
        for row in rows:
            existing = merged.get(row.state_hash)
            if existing is not None:
                if not _certificates_equivalent(existing, row):
                    raise ValueError("duplicate Bellman certificate disagrees")
                continue
            merged[row.state_hash] = row
    return list(merged.values())


def _build_or_resume_role(
    specs: tuple[BellmanRootSpec, ...],
    *,
    role: ShardRole,
    work_dir: Path,
    root_cache: dict[str, RootCacheEntry] | None = None,
) -> tuple[BellmanBundle, list[TrainingRecordV3], list[ExactPolicyCertificate]]:
    bundles = []
    record_chunks: list[list[TrainingRecordV3]] = []
    certificate_chunks: list[list[ExactPolicyCertificate]] = []
    for index, spec in enumerate(specs):
        paths = _chunk_paths(work_dir, role, index)
        state = _chunk_state(paths)
        expected_hash = exact_state_hash(exact_public_state(spec.game))
        if state == "complete":
            bundle = load_bellman_bundle(paths["bellman"])
            rows = load_replay_shard(
                paths["replay"], for_training=False, expected_role=role
            )
            certificates = load_certificate_shard(
                paths["certificates"], expected_role=role
            )
            print(
                f"[bellman] resumed {role.value} root {index + 1}/{len(specs)}: "
                f"{spec.name}",
                flush=True,
            )
        else:
            cached = (root_cache or {}).get(expected_hash)
            if cached is None:
                print(
                    f"[bellman] solving {role.value} root "
                    f"{index + 1}/{len(specs)}: {spec.name}",
                    flush=True,
                )
                bundle = build_bellman_bundle((spec,))
                rows, certificates = bundle_training_records(
                    bundle,
                    source_artifact=(
                        f"gen0-bellman-{role.value}-root-{index:06d}"
                    ),
                )
                generation_mode = "solved"
                cache_provenance: dict[str, object] = {}
            else:
                bundle = load_bellman_bundle(cached.bellman_path)
                rows = load_replay_shard(
                    cached.replay_path,
                    for_training=False,
                    expected_role=cached.source_role,
                )
                certificates = load_certificate_shard(
                    cached.certificate_path, expected_role=cached.source_role
                )
                if (
                    len(bundle.roots) != 1
                    or bundle.roots[0].state_hash != expected_hash
                ):
                    raise ValueError(
                        "cached Bellman root does not match the new run plan"
                    )
                cached_states = {bundle.roots[0].state_hash} | {
                    successor.state_hash for successor in bundle.successors
                }
                if {
                    exact_state_hash(row.exact_state) for row in rows
                } != cached_states:
                    raise ValueError(
                        "cached Bellman replay does not match its exact closure"
                    )
                generation_mode = "imported_committed_root"
                cache_provenance = {
                    "source_commit": cached.source_commit,
                    "source_binding_sha256": cached.binding_sha256,
                    "source_role": cached.source_role.value,
                }
                print(
                    f"[bellman] importing {role.value} root "
                    f"{index + 1}/{len(specs)} from committed cache: {spec.name}",
                    flush=True,
                )
            provenance = {
                "schema": "stl.gen0-bellman-root-generation.v1",
                "role": role.value,
                "root_index": index,
                "root_hash": expected_hash,
                "plan_digest": bundle.plan_digest,
                "generation_mode": generation_mode,
                **cache_provenance,
            }
            save_bellman_bundle(bundle, paths["bellman"])
            save_replay_shard(
                rows,
                paths["replay"],
                shard_role=role,
                generation_provenance=provenance,
            )
            save_certificate_shard(
                certificates,
                paths["certificates"],
                shard_role=role,
                generation_provenance=provenance,
            )
            _commit_chunk(paths)
            # Continue from the committed bytes, not higher-precision in-memory
            # arrays.  This makes uninterrupted and resumed runs feed identical
            # float32 replay rows into final publication.
            bundle = load_bellman_bundle(paths["bellman"])
            rows = load_replay_shard(
                paths["replay"], for_training=False, expected_role=role
            )
            certificates = load_certificate_shard(
                paths["certificates"], expected_role=role
            )
            print(
                f"[bellman] committed {generation_mode} {role.value} root "
                f"{index + 1}/{len(specs)}",
                flush=True,
            )
        if len(bundle.roots) != 1 or bundle.roots[0].state_hash != expected_hash:
            raise ValueError("resumed Bellman root does not match the run plan")
        expected_states = {bundle.roots[0].state_hash} | {
            successor.state_hash for successor in bundle.successors
        }
        observed_states = {exact_state_hash(row.exact_state) for row in rows}
        if observed_states != expected_states:
            raise ValueError("Bellman root replay does not match its closure")
        bundles.append(bundle)
        record_chunks.append(list(rows))
        certificate_chunks.append(list(certificates))

    merged_bundle = merge_bellman_bundles(bundles)
    root_hashes = {root.state_hash for root in merged_bundle.roots}
    successor_hashes = {
        successor.state_hash for successor in merged_bundle.successors
    }
    if root_hashes & successor_hashes:
        raise ValueError(
            "a Bellman state is both a selected root and successor; choose a "
            "different fixed split"
        )
    return (
        merged_bundle,
        _merge_role_records(
            record_chunks, role=role, plan_digest=merged_bundle.plan_digest
        ),
        _merge_role_certificates(certificate_chunks),
    )


def _bundles_equivalent(left: BellmanBundle, right: BellmanBundle) -> bool:
    if (
        left.plan_digest != right.plan_digest
        or left.search_config_digest != right.search_config_digest
        or len(left.roots) != len(right.roots)
        or len(left.successors) != len(right.successors)
        or len(left.branches) != len(right.branches)
    ):
        return False
    for left_root, right_root in zip(left.roots, right.roots):
        if (
            left_root.state_hash != right_root.state_hash
            or left_root.drop_actions != right_root.drop_actions
            or left_root.check_actions != right_root.check_actions
            or not np.isclose(left_root.value_h2, right_root.value_h2, atol=1.0e-8)
            or not np.isclose(left_root.value_h3, right_root.value_h3, atol=1.0e-8)
            or not np.allclose(left_root.q3_for_hal, right_root.q3_for_hal, atol=1.0e-8)
        ):
            return False
    for left_row, right_row in zip(left.successors, right.successors):
        if (
            left_row.state_hash != right_row.state_hash
            or not np.isclose(left_row.value_h2, right_row.value_h2, atol=1.0e-8)
            or not np.isclose(
                left_row.unresolved_h2, right_row.unresolved_h2, atol=1.0e-8
            )
        ):
            return False
    return left.branches == right.branches


def _publish_or_validate_role(
    *,
    paths: dict[str, Path],
    bundle: BellmanBundle,
    rows: list[TrainingRecordV3],
    certificates: list[ExactPolicyCertificate],
    role: ShardRole,
    provenance: dict[str, object],
) -> str:
    state = _artifact_state(paths)
    if state == "missing":
        save_bellman_bundle(bundle, paths["bellman"])
        save_replay_shard(
            rows,
            paths["replay"],
            shard_role=role,
            generation_provenance=provenance,
        )
        save_certificate_shard(
            certificates,
            paths["certificates"],
            shard_role=role,
            generation_provenance=provenance,
        )
        return "published"

    stored_bundle = load_bellman_bundle(paths["bellman"])
    stored_rows = load_replay_shard(
        paths["replay"], for_training=False, expected_role=role
    )
    stored_certificates = load_certificate_shard(
        paths["certificates"], expected_role=role
    )
    if not _bundles_equivalent(stored_bundle, bundle):
        raise ValueError(f"published {role.value} Bellman bundle disagrees")
    if len(stored_rows) != len(rows) or any(
        not _records_equivalent(left, right)
        for left, right in zip(stored_rows, rows)
    ):
        raise ValueError(f"published {role.value} replay shard disagrees")
    if len(stored_certificates) != len(certificates) or any(
        not _certificates_equivalent(left, right)
        for left, right in zip(stored_certificates, certificates)
    ):
        raise ValueError(f"published {role.value} certificates disagree")
    return "resumed"


def _generate_cycle_locked(
    args: argparse.Namespace, work_dir: Path
) -> dict[str, object]:
    blocked, blocked_digests = _blocked_hashes(list(args.blocked_artifact or ()))
    root_cache = _load_root_cache(list(args.reuse_work_dir or ()))
    closure_cache: dict[str, frozenset[str]] = {}

    def closure_of(specs: tuple[BellmanRootSpec, ...]) -> set[str]:
        closure: set[str] = set()
        for spec in specs:
            root_hash = exact_state_hash(exact_public_state(spec.game))
            footprint = closure_cache.get(root_hash)
            if footprint is None:
                footprint = bellman_closure_hashes(spec)
                closure_cache[root_hash] = footprint
            closure.update(footprint)
        return closure

    def progress_for(role: ShardRole):
        def report(computed: int, available: int) -> None:
            if computed == 1 or computed % 256 == 0:
                print(
                    f"[bellman] preflight {role.value}: computed "
                    f"{computed} uncached root closures "
                    f"({available} candidates before closure filtering)",
                    flush=True,
                )

        return report

    counts = dict(ROLE_COUNTS)
    if args.smoke:
        counts = {role: 1 for role in ROLE_COUNTS}
    role_specs: dict[ShardRole, tuple[BellmanRootSpec, ...]] = {}
    role_closures: dict[ShardRole, set[str]] = {}
    fixtures = () if args.smoke else _fixture_specs()
    fixture_hashes = {
        exact_state_hash(exact_public_state(spec.game)) for spec in fixtures
    }
    fixture_closure = closure_of(fixtures)
    allocated_roots = set(blocked) | fixture_hashes

    # Reserve public development fixtures before concealing the holdout. Then
    # allocate each split against every exact state in all prior split closures.
    external_specs = select_causal_roots(
        count=counts[ShardRole.EXTERNAL_RULER],
        salt=f"{args.split_seed}:sealed-holdout",
        blocked_state_hashes=allocated_roots | fixture_closure,
        blocked_closure_hashes=set(blocked) | fixture_closure,
        closure_cache=closure_cache,
        progress=progress_for(ShardRole.EXTERNAL_RULER),
        require_full_coverage=not args.smoke,
    )
    role_specs[ShardRole.EXTERNAL_RULER] = external_specs
    role_closures[ShardRole.EXTERNAL_RULER] = closure_of(external_specs)
    allocated_roots.update(
        exact_state_hash(exact_public_state(spec.game)) for spec in external_specs
    )

    development_specs = select_causal_roots(
        count=counts[ShardRole.DEVELOPMENT],
        salt=f"{args.split_seed}:development",
        blocked_state_hashes=allocated_roots | fixture_closure,
        blocked_closure_hashes=(
            set(blocked)
            | role_closures[ShardRole.EXTERNAL_RULER]
            | fixture_hashes
        ),
        closure_cache=closure_cache,
        progress=progress_for(ShardRole.DEVELOPMENT),
        require_full_coverage=not args.smoke,
    )
    development_specs = (*development_specs, *fixtures)
    role_specs[ShardRole.DEVELOPMENT] = development_specs
    role_closures[ShardRole.DEVELOPMENT] = closure_of(development_specs)
    allocated_roots.update(
        exact_state_hash(exact_public_state(spec.game))
        for spec in development_specs
    )

    replay_specs = select_causal_roots(
        count=counts[ShardRole.REPLAY],
        salt=f"{args.split_seed}:training",
        blocked_state_hashes=allocated_roots,
        blocked_closure_hashes=(
            set(blocked)
            | role_closures[ShardRole.EXTERNAL_RULER]
            | role_closures[ShardRole.DEVELOPMENT]
        ),
        closure_cache=closure_cache,
        progress=progress_for(ShardRole.REPLAY),
        require_full_coverage=not args.smoke,
    )
    role_specs[ShardRole.REPLAY] = replay_specs
    role_closures[ShardRole.REPLAY] = closure_of(replay_specs)

    roles = tuple(ROLE_COUNTS)
    for index, left in enumerate(roles):
        for right in roles[index + 1 :]:
            overlap = role_closures[left] & role_closures[right]
            if overlap:
                raise ValueError(
                    f"preflight Bellman {left.value}/{right.value} closure "
                    f"overlap: {len(overlap)} states"
                )

    selected_hashes = {
        exact_state_hash(exact_public_state(spec.game))
        for specs in role_specs.values()
        for spec in specs
    }
    reusable_root_chunks = {
        root_hash: entry.binding_sha256
        for root_hash, entry in sorted(root_cache.items())
        if root_hash in selected_hashes
    }
    print(
        "[bellman] closure preflight passed: "
        + ", ".join(
            f"{role.value}={len(role_closures[role])} states"
            for role in roles
        )
        + f"; reusable roots={len(reusable_root_chunks)}/{len(selected_hashes)}",
        flush=True,
    )

    _write_or_validate_plan(
        work_dir,
        split_seed=args.split_seed,
        blocked_digests=blocked_digests,
        role_specs=role_specs,
        role_closures=role_closures,
        reusable_root_chunks=reusable_root_chunks,
    )
    if args.preflight_only:
        return {
            "schema": "stl.gen0-bellman-cycle-preflight.v1",
            "outputs": {},
            "closure_state_counts": {
                role.value: len(role_closures[role]) for role in roles
            },
            "planned_root_count": len(selected_hashes),
            "reusable_root_count": len(reusable_root_chunks),
        }
    role_evidence = {
        role: _build_or_resume_role(
            specs,
            role=role,
            work_dir=work_dir,
            root_cache=root_cache,
        )
        for role, specs in role_specs.items()
    }
    bundles = {role: evidence[0] for role, evidence in role_evidence.items()}
    record_sets = {
        role: (evidence[1], evidence[2])
        for role, evidence in role_evidence.items()
    }
    identities: dict[ShardRole, set[str]] = {
        role: {exact_state_hash(record.exact_state) for record in rows}
        for role, (rows, _certificates) in record_sets.items()
    }
    episode_ids: dict[ShardRole, set[str]] = {
        role: {record.episode_id for record in rows}
        for role, (rows, _certificates) in record_sets.items()
    }
    feature_rows: dict[ShardRole, set[bytes]] = {
        role: {record.features.tobytes() for record in rows}
        for role, (rows, _certificates) in record_sets.items()
    }
    for index, left in enumerate(roles):
        for right in roles[index + 1 :]:
            overlap = identities[left] & identities[right]
            if overlap:
                raise ValueError(
                    f"Bellman {left.value}/{right.value} successor overlap: "
                    f"{len(overlap)} states"
                )
            episode_overlap = episode_ids[left] & episode_ids[right]
            if episode_overlap:
                raise ValueError(
                    f"Bellman {left.value}/{right.value} episode overlap: "
                    f"{len(episode_overlap)} episodes"
                )
            feature_overlap = feature_rows[left] & feature_rows[right]
            if feature_overlap:
                raise ValueError(
                    f"Bellman {left.value}/{right.value} feature overlap: "
                    f"{len(feature_overlap)} rows"
                )
    representability_by_role = {
        role: (
            None
            if role is ShardRole.EXTERNAL_RULER
            else candidate_action_representability(bundles[role])
        )
        for role in roles
    }
    candidate_gate_passed = all(
        report is None or bool(report["passed"])
        for report in representability_by_role.values()
    )
    outputs: dict[str, object] = {}
    for role in roles:
        paths = _artifact_paths(args.out_prefix, role)
        rows, certificates = record_sets[role]
        provenance = {
            "schema": "stl.gen0-bellman-generation.v1",
            "split_seed": int(args.split_seed),
            "plan_digest": bundles[role].plan_digest,
            "blocked_artifacts": blocked_digests,
        }
        publication = _publish_or_validate_role(
            paths=paths,
            bundle=bundles[role],
            rows=rows,
            certificates=certificates,
            role=role,
            provenance=provenance,
        )
        representability = representability_by_role[role]
        outputs[role.value] = {
            "paths": {name: str(path) for name, path in paths.items()},
            "root_count": len(bundles[role].roots),
            "successor_count": len(bundles[role].successors),
            "branch_count": len(bundles[role].branches),
            "record_count": len(rows),
            "certificate_count": len(certificates),
            "representability": representability,
            "publication": publication,
        }

    seal = None
    if not args.smoke and candidate_gate_passed:
        required = (
            args.calibration_holdout,
            args.calibration_certificates,
            args.calibration_taxonomy,
            args.holdout_seal_out,
        )
        if any(value is None for value in required):
            raise ValueError(
                "full Bellman generation requires fresh calibration holdout, "
                "certificates, taxonomy, and --holdout-seal-out"
            )
        holdout_paths = _artifact_paths(
            args.out_prefix, ShardRole.EXTERNAL_RULER
        )
        holdout_bundle = bundles[ShardRole.EXTERNAL_RULER]
        seal = write_bellman_holdout_seal(
            args.holdout_seal_out,
            holdout_path=holdout_paths["replay"],
            certificate_path=holdout_paths["certificates"],
            bellman_path=holdout_paths["bellman"],
            calibration_holdout_path=args.calibration_holdout,
            calibration_certificate_path=args.calibration_certificates,
            calibration_taxonomy_path=args.calibration_taxonomy,
            generation_plan_digest=holdout_bundle.plan_digest,
            blocked_artifacts=blocked_digests,
            bellman_gates=asdict(BellmanGateThresholds()),
            mcts_root_hashes=_mcts_subset_hashes(holdout_bundle),
        )
    report = {
        "schema": "stl.gen0-bellman-cycle-report.v1",
        "outputs": outputs,
        "candidate_gate_passed": candidate_gate_passed,
        "holdout_seal": seal,
    }
    report_path = Path(str(args.out_prefix) + ".report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not args.smoke and not candidate_gate_passed:
        raise ValueError(
            "candidate-action representability failed; exact Bellman artifacts "
            f"and diagnostic report were preserved at {args.out_prefix}"
        )
    return report


def generate_cycle(args: argparse.Namespace) -> dict[str, object]:
    work_dir = Path(
        args.work_dir
        if args.work_dir is not None
        else str(args.out_prefix) + "_work"
    )
    with _exclusive_generation_lock(work_dir):
        return _generate_cycle_locked(args, work_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-prefix", default="outputs/regen2rl/gen0_v5_bellman"
    )
    parser.add_argument(
        "--work-dir",
        default=None,
        help="Per-root committed work directory; defaults to OUT_PREFIX_work.",
    )
    parser.add_argument(
        "--reuse-work-dir",
        action="append",
        default=None,
        help=(
            "Import strictly committed singleton roots from an earlier work "
            "directory when their exact root hashes occur in the new plan."
        ),
    )
    parser.add_argument("--split-seed", type=int, default=20260715)
    parser.add_argument("--blocked-artifact", action="append", default=None)
    parser.add_argument("--calibration-holdout", default=None)
    parser.add_argument("--calibration-certificates", default=None)
    parser.add_argument("--calibration-taxonomy", default=None)
    parser.add_argument("--holdout-seal-out", default=None)
    parser.add_argument(
        "--preflight-only", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--smoke", action=argparse.BooleanOptionalAction, default=False
    )
    return parser


def main() -> int:
    report = generate_cycle(build_parser().parse_args())
    for role, row in report["outputs"].items():
        print(
            f"[bellman] {role}: roots={row['root_count']} "
            f"successors={row['successor_count']} records={row['record_count']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
