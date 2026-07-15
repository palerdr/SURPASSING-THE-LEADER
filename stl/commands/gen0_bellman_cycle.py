"""Generate the horizon-aware Bellman closure for the next Gen-0 cycle."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from stl.learning.bellman import (
    BellmanGateThresholds,
    BellmanRootSpec,
    build_bellman_bundle,
    bundle_training_records,
    candidate_action_representability,
    save_bellman_bundle,
    select_causal_roots,
)
from stl.learning.certificates import save_certificate_shard
from stl.learning.holdout import sha256_file, write_bellman_holdout_seal
from stl.learning.replay import (
    ShardRole,
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


def _artifact_paths(prefix: str | Path, role: ShardRole) -> dict[str, Path]:
    base = Path(prefix)
    stem = f"{base.name}_{role.value}"
    return {
        "replay": base.with_name(stem + ".npz"),
        "certificates": base.with_name(stem + ".certificates.npz"),
        "bellman": base.with_name(stem + ".bellman.npz"),
    }


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


def generate_cycle(args: argparse.Namespace) -> dict[str, object]:
    blocked, blocked_digests = _blocked_hashes(list(args.blocked_artifact or ()))
    counts = dict(ROLE_COUNTS)
    if args.smoke:
        counts = {role: 1 for role in ROLE_COUNTS}
    role_specs: dict[ShardRole, tuple[BellmanRootSpec, ...]] = {}
    allocated = set(blocked)
    # The concealed split is allocated first and never selected in response to
    # model results. Development and training are then blocked against it.
    for role, salt in (
        (ShardRole.EXTERNAL_RULER, "sealed-holdout"),
        (ShardRole.DEVELOPMENT, "development"),
        (ShardRole.REPLAY, "training"),
    ):
        specs = select_causal_roots(
            count=counts[role],
            salt=f"{args.split_seed}:{salt}",
            blocked_state_hashes=allocated,
            require_full_coverage=not args.smoke,
        )
        if role is ShardRole.DEVELOPMENT and not args.smoke:
            specs = (*specs, *_fixture_specs())
        role_specs[role] = tuple(specs)
        allocated.update(
            exact_state_hash(exact_public_state(spec.game))
            for spec in specs
        )

    bundles = {
        role: build_bellman_bundle(specs)
        for role, specs in role_specs.items()
    }
    record_sets = {
        role: bundle_training_records(
            bundle,
            source_artifact=f"gen0-bellman-{role.value}",
        )
        for role, bundle in bundles.items()
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
    roles = tuple(ROLE_COUNTS)
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
    for role, representability in representability_by_role.items():
        if (
            not args.smoke
            and representability is not None
            and not representability["passed"]
        ):
            raise ValueError(
                f"{role.value} candidate-action representability failed"
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
        save_bellman_bundle(bundles[role], paths["bellman"])
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
        representability = representability_by_role[role]
        outputs[role.value] = {
            "paths": {name: str(path) for name, path in paths.items()},
            "root_count": len(bundles[role].roots),
            "successor_count": len(bundles[role].successors),
            "branch_count": len(bundles[role].branches),
            "record_count": len(rows),
            "certificate_count": len(certificates),
            "representability": representability,
        }

    seal = None
    if not args.smoke:
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
        "holdout_seal": seal,
    }
    report_path = Path(str(args.out_prefix) + ".report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-prefix", default="outputs/regen2rl/gen0_v5_bellman"
    )
    parser.add_argument("--split-seed", type=int, default=20260715)
    parser.add_argument("--blocked-artifact", action="append", default=None)
    parser.add_argument("--calibration-holdout", default=None)
    parser.add_argument("--calibration-certificates", default=None)
    parser.add_argument("--calibration-taxonomy", default=None)
    parser.add_argument("--holdout-seal-out", default=None)
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
