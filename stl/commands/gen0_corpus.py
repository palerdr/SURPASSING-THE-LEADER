"""Generate coordinated, reachable V3 Generation-Zero train/ruler shards.

The command owns both splits in one immutable run plan.  Whole legal-engine
episodes are assigned before exact labeling, tactical pins are split once, and
Tier A is training-only.  Committed component chunks are strict replay shards
and are reused after interruption; a changed plan or corrupt chunk fails closed.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, replace
import hashlib
import os
from pathlib import Path
import platform
import subprocess
from types import SimpleNamespace
from typing import Callable, Sequence

import numpy as np
import scipy

from stl.commands import tier_a_targets
from stl.learning.contracts import canonical_config_json, config_digest
from stl.learning.model import extract_features
from stl.learning.reachable import (
    ReachableSnapshot,
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
from stl.solver.exact import ExactSearchConfig, exact_public_state
from stl.solver.tablebase import REGISTRY


GENERATION_PLAN_SCHEMA = "stl.gen0-generation-plan.v1"
GENERATION_PROVENANCE_SCHEMA = "stl.gen0-generation-provenance.v1"
DEFAULT_SPLIT_SEED = 20260714


def _tier_a_row_count(args: argparse.Namespace) -> int:
    return (2 if args.smoke else 32) if args.tier_a_rows is None else int(args.tier_a_rows)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_tree_digest(repo_root: Path) -> str:
    paths = [
        *repo_root.glob("stl/**/*.py"),
        *repo_root.glob("configs/**/*.yaml"),
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
        return digest
    temporary = plan_path.with_name(f".{plan_path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, plan_path)
    return digest


def _record_from_snapshot(
    snapshot: ReachableSnapshot,
    *,
    search_config: ExactSearchConfig,
    search_digest: str,
) -> TrainingRecordV3:
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
    return to_training_record_v3(
        target,
        search_config_digest=search_digest,
        rng_seeds={"trajectory": 0},
    )


def _trajectory_records(
    snapshots: Sequence[ReachableSnapshot],
    *,
    search_config: ExactSearchConfig,
    search_digest: str,
) -> list[TrainingRecordV3]:
    # The corrected candidate set is deliberately small.  Keeping exact LP
    # solves serial here avoids nested process pools and makes chunk restart
    # boundaries unambiguous; --workers is reserved for later scale-out.
    return [
        _record_from_snapshot(
            snapshot,
            search_config=search_config,
            search_digest=search_digest,
        )
        for snapshot in snapshots
    ]


def _rank(seed: int, namespace: str, key: str) -> str:
    return hashlib.sha256(f"{seed}:{namespace}:{key}".encode("utf-8")).hexdigest()


def _tactical_targets(
    *, split_seed: int, search_config: ExactSearchConfig
) -> tuple[list[ValueTarget], list[ValueTarget]]:
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
    return train, ruler


def _tier_a_records(
    args: argparse.Namespace,
    *,
    blocked_hashes: set[str],
    search_digest: str,
) -> list[TrainingRecordV3]:
    requested = _tier_a_row_count(args)
    if requested == 0:
        return []
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
        if not path.exists():
            raise ReplayValidationError(f"committed chunk payload is missing: {path}")
        records = load_replay_shard(path, for_training=role is ShardRole.REPLAY)
        manifest = load_replay_manifest(path)
        if manifest.get("generation_provenance") != provenance:
            raise ReplayValidationError(f"committed chunk plan mismatch: {path}")
        print(f"resume: {path.name} ({len(records)} records)")
        return records
    if path.exists():
        # A payload without the manifest commit marker is an interrupted write.
        path.unlink()
    records = list(factory())
    save_replay_shard(
        records,
        path,
        shard_role=role,
        generation_provenance=provenance,
    )
    print(f"wrote: {path.name} ({len(records)} records)")
    return records


def _feature_hash(record: TrainingRecordV3) -> str:
    return hashlib.sha256(np.ascontiguousarray(record.features).tobytes()).hexdigest()


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
        raise RuntimeError("Gen-0 contains divergent targets for a collapsed feature vector")


def _publish_or_resume_final(
    path: Path,
    records: Sequence[TrainingRecordV3],
    *,
    role: ShardRole,
    provenance: dict[str, object],
) -> dict[str, object]:
    manifest_path = manifest_path_for(path)
    if path.exists() and manifest_path.exists():
        loaded = load_replay_shard(path, for_training=role is ShardRole.REPLAY)
        manifest = load_replay_manifest(path)
        if manifest.get("generation_provenance") != provenance:
            raise ReplayValidationError(f"published final shard plan mismatch: {path}")
        if [exact_state_hash(row.exact_state) for row in loaded] != [
            exact_state_hash(row.exact_state) for row in records
        ]:
            raise ReplayValidationError(f"published final shard row order mismatch: {path}")
        return manifest
    if path.exists() or manifest_path.exists():
        raise ReplayValidationError(f"partial final shard requires manual audit: {path}")
    return save_replay_shard(
        records,
        path,
        shard_role=role,
        generation_provenance=provenance,
    )


def generate_gen0_pair(args: argparse.Namespace) -> dict[str, object]:
    if args.workers != 1:
        raise ValueError(
            "the resumable V3 baseline currently requires --workers 1"
        )
    if _tier_a_row_count(args) < 0:
        raise ValueError("tier-a-rows must be non-negative")
    repo_root = Path(__file__).resolve().parents[2]
    work_dir = Path(args.work_dir)
    with _exclusive_lock(work_dir):
        plan = _resolved_plan(args, repo_root)
        plan_digest = _write_or_validate_plan(work_dir, plan)
        search_config = ExactSearchConfig()
        search_digest = config_digest(_search_config_payload(search_config))
        base_provenance = {
            "schema": GENERATION_PROVENANCE_SCHEMA,
            "plan_digest": plan_digest,
            "source_tree_sha256": plan["source_tree_sha256"],
            "split_seed": int(args.split_seed),
            "tier_a_manifest_sha256": plan["tier_a"]["manifest_sha256"],
        }

        reachable = split_reachable_candidates(
            smoke=bool(args.smoke), split_seed=int(args.split_seed)
        )
        tactical_train, tactical_ruler = _tactical_targets(
            split_seed=int(args.split_seed), search_config=search_config
        )

        components: list[tuple[str, ShardRole, Callable[[], Sequence[TrainingRecordV3]]]] = [
            (
                "train.trajectory",
                ShardRole.REPLAY,
                lambda: _trajectory_records(
                    reachable.train,
                    search_config=search_config,
                    search_digest=search_digest,
                ),
            ),
            (
                "ruler.trajectory",
                ShardRole.EXTERNAL_RULER,
                lambda: _trajectory_records(
                    reachable.ruler,
                    search_config=search_config,
                    search_digest=search_digest,
                ),
            ),
            (
                "train.tactical",
                ShardRole.REPLAY,
                lambda: [
                    to_training_record_v3(target, search_config_digest=search_digest)
                    for target in tactical_train
                ],
            ),
            (
                "ruler.tactical",
                ShardRole.EXTERNAL_RULER,
                lambda: [
                    to_training_record_v3(target, search_config_digest=search_digest)
                    for target in tactical_ruler
                ],
            ),
        ]
        chunk_records: dict[str, list[TrainingRecordV3]] = {}
        for name, role, factory in components:
            provenance = {**base_provenance, "component": name}
            chunk_records[name] = _materialize_chunk(
                work_dir / f"{name}.000000.npz",
                role=role,
                provenance=provenance,
                factory=factory,
            )

        pre_tier_train, train_dedupe = _dedupe_by_certificate([
            *chunk_records["train.trajectory"],
            *chunk_records["train.tactical"],
        ])
        ruler, ruler_dedupe = _dedupe_by_certificate([
            *chunk_records["ruler.trajectory"],
            *chunk_records["ruler.tactical"],
        ])
        blocked_hashes = {
            exact_state_hash(record.exact_state) for record in [*pre_tier_train, *ruler]
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
        _assert_pair_isolated(train, ruler)

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
                "train": dict(Counter(record.source for record in train)),
                "ruler": dict(Counter(record.source for record in ruler)),
            },
            "certificate_dedupe": {
                "train_before_tier_a": train_dedupe,
                "ruler": ruler_dedupe,
                "train_final": final_train_dedupe,
            },
        }
        train_manifest = _publish_or_resume_final(
            Path(args.train_out),
            train,
            role=ShardRole.REPLAY,
            provenance={**common_final, "component": "final.train"},
        )
        ruler_manifest = _publish_or_resume_final(
            Path(args.ruler_out),
            ruler,
            role=ShardRole.EXTERNAL_RULER,
            provenance={**common_final, "component": "final.ruler"},
        )
        return {
            "plan_digest": plan_digest,
            "train": {
                "path": str(args.train_out),
                "records": len(train),
                "sources": dict(Counter(record.source for record in train)),
                "manifest": train_manifest,
            },
            "ruler": {
                "path": str(args.ruler_out),
                "records": len(ruler),
                "sources": dict(Counter(record.source for record in ruler)),
                "manifest": ruler_manifest,
            },
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-out", default="outputs/regen2rl/gen0_train_v3.npz")
    parser.add_argument(
        "--ruler-out", default="outputs/regen2rl/gen0_external_ruler_v3.npz"
    )
    parser.add_argument("--work-dir", default="outputs/regen2rl/gen0_work_v3")
    parser.add_argument("--tier-a-dir", default="checkpoints/tablebase/tier_a")
    parser.add_argument("--tier-a-max-width", type=float, default=0.05)
    parser.add_argument(
        "--tier-a-rows",
        type=int,
        default=None,
        help="Training-only Tier A rows (default: 2 for --smoke, otherwise 32).",
    )
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--workers", type=int, default=1)
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
        f"External ruler: {summary['ruler']['records']} records -> "
        f"{summary['ruler']['path']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
