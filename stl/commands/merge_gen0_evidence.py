"""Merge disclosed Gen-0 evidence into one isolated replay/development shard."""

from __future__ import annotations

import argparse
from pathlib import Path

from stl.learning.certificates import (
    load_certificate_shard,
    save_certificate_shard,
)
from stl.learning.holdout import load_taxonomy, write_taxonomy
from stl.learning.replay import (
    ShardRole,
    exact_state_hash,
    load_replay_manifest,
    load_replay_shard,
    save_replay_shard,
)


def merge_evidence(args: argparse.Namespace) -> dict[str, object]:
    role = ShardRole(args.role)
    records = []
    input_digests = {}
    for raw_path in args.input:
        path = Path(raw_path)
        manifest = load_replay_manifest(path)
        input_digests[str(path)] = str(manifest["data_sha256"])
        records.extend(load_replay_shard(path, for_training=False))
    state_hashes = [
        (exact_state_hash(record.exact_state), record.value_horizon_half_rounds)
        for record in records
    ]
    episode_ids = [record.episode_id for record in records if record.episode_id]
    feature_hashes = [
        (record.features.tobytes(), record.value_horizon_half_rounds)
        for record in records
    ]
    for name, values in (
        ("exact states", state_hashes),
        ("episode ids", episode_ids),
        ("feature vectors", feature_hashes),
    ):
        if len(values) != len(set(values)):
            raise ValueError(f"merged evidence contains duplicate {name}")

    certificates = []
    for path in args.certificate or ():
        certificates.extend(load_certificate_shard(path))
    certificate_keys = [
        (row.state_hash, row.horizon) for row in certificates
    ]
    if len(certificate_keys) != len(set(certificate_keys)):
        raise ValueError("merged evidence contains duplicate policy certificates")
    active_hashes = {
        exact_state_hash(record.exact_state)
        for record in records
        if record.dropper_dist.sum() > 0.0 or record.checker_dist.sum() > 0.0
    }
    if {row.state_hash for row in certificates} != active_hashes:
        raise ValueError(
            "merged certificates do not exactly cover active policy rows"
        )

    provenance = {
        "schema": "stl.merged-gen0-evidence.v1",
        "inputs": input_digests,
    }
    save_replay_shard(
        records,
        args.out,
        shard_role=role,
        generation_provenance=provenance,
    )
    save_certificate_shard(
        certificates,
        args.out_certificates,
        shard_role=role,
        generation_provenance=provenance,
    )
    taxonomy = {}
    for path in args.taxonomy or ():
        entries = load_taxonomy(path)
        overlap = set(taxonomy) & set(entries)
        if overlap:
            raise ValueError("merged taxonomy contains duplicate state hashes")
        taxonomy.update(entries)
    taxonomy_path = write_taxonomy(args.out, taxonomy)
    return {
        "records": len(records),
        "certificates": len(certificates),
        "taxonomy_entries": len(taxonomy),
        "taxonomy_path": str(taxonomy_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--certificate", action="append", default=None)
    parser.add_argument("--taxonomy", action="append", default=None)
    parser.add_argument("--role", choices=[role.value for role in ShardRole], required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--out-certificates", required=True)
    return parser


def main() -> int:
    report = merge_evidence(build_parser().parse_args())
    print(
        f"[merge-gen0] records={report['records']} "
        f"certificates={report['certificates']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
