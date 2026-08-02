"""Hydra-configured exact-target generation for pure DTH."""

from __future__ import annotations

from collections import defaultdict
import hashlib
from itertools import product
import json
from pathlib import Path
import time
from typing import Iterable, Mapping, Sequence

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from dth.solver import (
    CHECKER_ACTIONS,
    DROPPER_ACTIONS,
    VALUE_SEMANTICS_PLAY,
    NTState,
    Solution,
    TARGET_SCHEMA,
    clear_solver_cache,
    horizon_one_cache_info,
    resolve_value_semantics,
    solve,
    transition,
)


def _normalize_state(raw: Sequence[int]) -> NTState:
    if len(raw) != 4:
        raise ValueError(f"root state must have four coordinates, got {raw!r}")

    state = tuple(int(value) for value in raw)
    checker_st, checker_ttd, dropper_st, dropper_ttd = state
    if not (0 <= checker_st < 300 and 0 <= dropper_st < 300):
        raise ValueError(f"root ST coordinates must be in 0..299, got {state!r}")
    if not (0 <= checker_ttd <= 300 and 0 <= dropper_ttd <= 300):
        raise ValueError(f"root TTD coordinates must be in 0..300, got {state!r}")
    return state


def mirror_state(raw: Sequence[int]) -> NTState:
    """Swap Checker and Dropper coordinates for a role-orientation pair."""

    checker_st, checker_ttd, dropper_st, dropper_ttd = _normalize_state(raw)
    return dropper_st, dropper_ttd, checker_st, checker_ttd


def live_successors(state: NTState) -> set[NTState]:
    """Return every distinct live successor using transition-equivalent cells."""

    children: set[NTState] = set()

    # Drop 1 paired with checks 1..60 realizes every successful ST increment.
    for check in CHECKER_ACTIONS:
        for _, child in transition(state, 1, check):
            if isinstance(child, tuple):
                children.add(child)

    # Every failed cell has the same chance branches; (drop=2, check=1) suffices.
    for _, child in transition(state, 2, 1):
        if isinstance(child, tuple):
            children.add(child)

    return children


def failure_margin_class(state_pair: Sequence[int]) -> str:
    """Classify one player's next failed-check boundary."""

    st, ttd = (int(value) for value in state_pair)
    dose = st + 60
    if dose >= 300:
        return "dose_fatal"
    total = ttd + dose
    if total > 300:
        return "ttd_fatal"
    if total == 300:
        return "exact_300"
    margin = 300 - total
    if margin <= 5:
        return "near_1_5"
    if margin <= 60:
        return "pressure_6_60"
    return "safe"


def strategic_stratum(state: NTState) -> str:
    checker = failure_margin_class(state[:2])
    dropper = failure_margin_class(state[2:])
    return f"checker={checker}|dropper={dropper}"


def sample_strategic_roots(
    *,
    count: int,
    st_values: Sequence[int],
    ttd_values: Sequence[int],
    forced_roots: Iterable[Sequence[int]],
    seed: int,
) -> tuple[NTState, ...]:
    """Deterministically round-robin sample the joint boundary strata."""

    if count <= 0:
        raise ValueError(f"strategic root count must be positive, got {count}")

    forced = tuple(dict.fromkeys(_normalize_state(state) for state in forced_roots))
    if len(forced) > count:
        raise ValueError(
            f"{len(forced)} forced roots do not fit in requested count {count}"
        )

    st_support = tuple(dict.fromkeys(int(value) for value in st_values))
    ttd_support = tuple(dict.fromkeys(int(value) for value in ttd_values))
    if not st_support or not ttd_support:
        raise ValueError("strategic ST and TTD supports must be non-empty")

    groups: dict[str, list[NTState]] = defaultdict(list)
    forced_set = set(forced)
    for raw in product(st_support, ttd_support, st_support, ttd_support):
        state = _normalize_state(raw)
        if state not in forced_set:
            groups[strategic_stratum(state)].append(state)

    rng = np.random.default_rng(seed)
    keys = sorted(groups)
    rng.shuffle(keys)
    for key in keys:
        rng.shuffle(groups[key])

    selected = list(forced)
    cursors = {key: 0 for key in keys}
    while len(selected) < count:
        made_progress = False
        for key in keys:
            cursor = cursors[key]
            if cursor >= len(groups[key]):
                continue
            selected.append(groups[key][cursor])
            cursors[key] = cursor + 1
            made_progress = True
            if len(selected) == count:
                break
        if not made_progress:
            raise ValueError("strategic support contains fewer states than requested")

    return tuple(selected)


def reachable_layers(
    root_states: Iterable[Sequence[int]],
    horizon: int,
) -> list[set[NTState]]:
    """Enumerate the live state layers needed for positive-horizon targets."""

    if not 1 <= horizon <= 255:
        raise ValueError(f"horizon must be in 1..255, got {horizon}")

    roots = {_normalize_state(state) for state in root_states}
    if not roots:
        raise ValueError("at least one root state is required")

    layers = [roots]
    for _ in range(horizon - 1):
        next_layer: set[NTState] = set()
        for state in layers[-1]:
            next_layer.update(live_successors(state))
        layers.append(next_layer)
    return layers


def _boundary_tablebase_roots(
    pairs: Sequence[Mapping[str, object]],
) -> tuple[tuple[NTState, int, str], ...]:
    """Expand declared boundary roots into both mechanical role orientations."""

    if not pairs:
        raise ValueError("boundary tablebase requires at least one paired root")

    roots: list[tuple[NTState, int, str]] = []
    for pair_index, pair in enumerate(pairs):
        if "state" not in pair or "horizons" not in pair:
            raise ValueError(
                f"boundary tablebase pair {pair_index} needs state and horizons"
            )
        primary = _normalize_state(pair["state"])
        mirrored = mirror_state(primary)
        if primary == mirrored:
            raise ValueError(
                f"boundary tablebase pair {pair_index} is self-mirrored: {primary!r}"
            )
        horizons = tuple(int(value) for value in pair["horizons"])
        if not horizons:
            raise ValueError(f"boundary tablebase pair {pair_index} has no horizons")
        for horizon in horizons:
            if not 1 <= horizon <= 255:
                raise ValueError(
                    f"boundary tablebase horizon must be in 1..255, got {horizon}"
                )
            roots.append((primary, horizon, "primary"))
            roots.append((mirrored, horizon, "mirror"))

    identities = [(state, horizon) for state, horizon, _ in roots]
    if len(set(identities)) != len(identities):
        raise ValueError("boundary tablebase roots contain duplicate state/horizon rows")
    return tuple(roots)


def boundary_tablebase_identities(
    pairs: Sequence[Mapping[str, object]],
) -> tuple[tuple[tuple[NTState, int, str], ...], set[tuple[NTState, int]]]:
    """Return paired roots and their complete positive-horizon live closure."""

    roots = _boundary_tablebase_roots(pairs)
    identities: set[tuple[NTState, int]] = set()
    for state, horizon, _ in roots:
        for depth, layer in enumerate(reachable_layers((state,), horizon)):
            remaining = horizon - depth
            identities.update((child, remaining) for child in layer)
    return roots, identities


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generate_boundary_tablebase(
    *,
    output: str | Path,
    report_output: str | Path,
    pairs: Sequence[Mapping[str, object]],
    progress_every: int = 1_000,
    dataset_version: str = "boundary_tablebase_v1",
) -> Path:
    """Materialize a reusable exact closure for paired boundary roots.

    This is a tablebase artifact, not a sampled corpus: every live state needed
    to evaluate each configured root at its configured horizon is emitted with
    exact LP value and policy targets.  The paired root declaration ensures both
    role orientations are included without assuming that they are symmetric.
    """

    if progress_every < 0:
        raise ValueError("progress_every must be non-negative")

    roots, identities = boundary_tablebase_identities(pairs)
    clear_solver_cache()
    started = time.monotonic()
    solutions: dict[tuple[NTState, int], Solution] = {}
    solve_order = sorted(identities, key=lambda item: (-item[1], item[0]))
    total = len(solve_order)
    for index, (state, horizon) in enumerate(solve_order, start=1):
        solutions[(state, horizon)] = solve(state, horizon)
        if progress_every and index % progress_every == 0:
            print(f"Solved {index}/{total} tablebase states", flush=True)

    rows = [
        (state, horizon, solutions[(state, horizon)])
        for state, horizon in sorted(identities, key=lambda item: (item[1], item[0]))
    ]
    coverage_horizons = sorted({horizon for _, horizon in identities})
    coverage_counts = [
        sum(1 for _, horizon in identities if horizon == remaining)
        for remaining in coverage_horizons
    ]
    cache_info = solve.cache_info()
    horizon_one_info = horizon_one_cache_info()

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        states=np.asarray([row[0] for row in rows], dtype=np.int16),
        horizons=np.asarray([row[1] for row in rows], dtype=np.uint8),
        values=np.asarray([row[2].value for row in rows], dtype=np.float32),
        drop_policies=np.asarray(
            [row[2].drop_policy for row in rows], dtype=np.float32
        ),
        check_policies=np.asarray(
            [row[2].check_policy for row in rows], dtype=np.float32
        ),
        saddle_gaps=np.asarray(
            [row[2].saddle_gap for row in rows], dtype=np.float32
        ),
        drop_actions=np.asarray(DROPPER_ACTIONS, dtype=np.int16),
        check_actions=np.asarray(CHECKER_ACTIONS, dtype=np.int16),
        dataset_version=np.asarray(dataset_version),
        emission=np.asarray("boundary_tablebase_closure"),
        root_states=np.asarray([state for state, _, _ in roots], dtype=np.int16),
        root_horizons=np.asarray([horizon for _, horizon, _ in roots], dtype=np.uint8),
        root_orientations=np.asarray(
            [orientation for _, _, orientation in roots], dtype=np.str_
        ),
        coverage_horizons=np.asarray(coverage_horizons, dtype=np.uint8),
        coverage_counts=np.asarray(coverage_counts, dtype=np.int64),
        solver_cache_hits=np.asarray(cache_info.hits, dtype=np.int64),
        solver_cache_misses=np.asarray(cache_info.misses, dtype=np.int64),
        horizon_one_equivalence_hits=np.asarray(horizon_one_info.hits, dtype=np.int64),
        horizon_one_equivalence_misses=np.asarray(
            horizon_one_info.misses, dtype=np.int64
        ),
        schema_version=np.asarray(TARGET_SCHEMA),
    )

    report_path = Path(report_output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": "dth-boundary-tablebase-report-v1",
        "dataset_version": dataset_version,
        "emission": "boundary_tablebase_closure",
        "artifact": {
            "path": str(destination),
            "bytes": destination.stat().st_size,
            "sha256": _sha256(destination),
            "target_schema": TARGET_SCHEMA,
            "rows": len(rows),
        },
        "roots": {
            "rows": len(roots),
            "primary_rows": sum(orientation == "primary" for _, _, orientation in roots),
            "mirror_rows": sum(orientation == "mirror" for _, _, orientation in roots),
        },
        "coverage_by_remaining_horizon": {
            str(horizon): count
            for horizon, count in zip(coverage_horizons, coverage_counts, strict=True)
        },
        "solver_cache": {
            "state_horizon": {
                "hits": cache_info.hits,
                "misses": cache_info.misses,
                "currsize": cache_info.currsize,
            },
            "horizon_one_equivalence": {
                "hits": horizon_one_info.hits,
                "misses": horizon_one_info.misses,
                "currsize": horizon_one_info.currsize,
            },
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {len(rows)} exact tablebase targets to {destination} "
        f"and coverage report to {report_path}",
        flush=True,
    )
    return destination


def generate_exact_targets(
    *,
    output: str | Path,
    horizon: int,
    root_states: Iterable[Sequence[int]],
    progress_every: int = 250,
    base_datasets: Iterable[str | Path] = (),
    dataset_version: str | None = None,
) -> Path:
    """Solve and write deterministic reachable exact targets."""

    if progress_every < 0:
        raise ValueError("progress_every must be non-negative")

    layers = reachable_layers(root_states, horizon)
    rows: list[tuple[NTState, int, Solution]] = []
    clear_solver_cache()

    # Solve deepest states first so shallower solves reuse the memoized values.
    for depth in range(horizon - 1, -1, -1):
        remaining = horizon - depth
        layer = sorted(layers[depth])
        print(
            f"Solving {len(layer)} states with remaining horizon {remaining}",
            flush=True,
        )
        for index, state in enumerate(layer, start=1):
            rows.append((state, remaining, solve(state, remaining)))
            if progress_every and index % progress_every == 0:
                print(f"  solved {index}/{len(layer)}", flush=True)

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        states=np.asarray([row[0] for row in rows], dtype=np.int16),
        horizons=np.asarray([row[1] for row in rows], dtype=np.uint8),
        values=np.asarray([row[2].value for row in rows], dtype=np.float32),
        drop_policies=np.asarray(
            [row[2].drop_policy for row in rows], dtype=np.float32
        ),
        check_policies=np.asarray(
            [row[2].check_policy for row in rows], dtype=np.float32
        ),
        saddle_gaps=np.asarray(
            [row[2].saddle_gap for row in rows], dtype=np.float32
        ),
        drop_actions=np.asarray(DROPPER_ACTIONS, dtype=np.int16),
        check_actions=np.asarray(CHECKER_ACTIONS, dtype=np.int16),
        dataset_version=np.asarray(dataset_version or destination.stem),
        schema_version=np.asarray(TARGET_SCHEMA),
    )
    bases = tuple(Path(path) for path in base_datasets)
    if bases:
        merge_exact_target_artifacts(
            (*bases, destination),
            destination,
            dataset_version=dataset_version or destination.stem,
        )
    print(f"Wrote {len(rows)} targets to {destination}", flush=True)
    return destination


def merge_exact_target_artifacts(
    inputs: Iterable[str | Path],
    output: str | Path,
    *,
    dataset_version: str,
) -> Path:
    """Merge exact artifacts by state/horizon identity, keeping later rows.

    Each row keeps its source's value semantics, so listing play-valued
    resolve artifacts before exact closures lets exact rows win collisions
    while the merged artifact still names every surviving play row.
    """

    records: dict[
        tuple[int, NTState], tuple[float, np.ndarray, np.ndarray, float, int]
    ] = {}
    for source in inputs:
        with np.load(Path(source), allow_pickle=False) as artifact:
            if str(np.asarray(artifact["schema_version"]).item()) != TARGET_SCHEMA:
                raise ValueError(f"target schema mismatch in {source}")
            states = artifact["states"]
            horizons = artifact["horizons"]
            values = artifact["values"]
            drop_policies = artifact["drop_policies"]
            check_policies = artifact["check_policies"]
            saddle_gaps = artifact["saddle_gaps"]
            emission = (
                str(np.asarray(artifact["emission"]).item())
                if "emission" in artifact.files
                else ""
            )
            semantics = resolve_value_semantics(
                emission,
                artifact["value_semantics"]
                if "value_semantics" in artifact.files
                else None,
                len(states),
            )
            for index in range(len(states)):
                state = tuple(int(value) for value in states[index])
                key = (int(horizons[index]), state)
                records[key] = (
                    float(values[index]),
                    drop_policies[index].astype(np.float32, copy=True),
                    check_policies[index].astype(np.float32, copy=True),
                    float(saddle_gaps[index]),
                    int(semantics[index]),
                )

    ordered = sorted(records.items(), key=lambda item: (item[0][0], item[0][1]))
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        states=np.asarray([key[1] for key, _ in ordered], dtype=np.int16),
        horizons=np.asarray([key[0] for key, _ in ordered], dtype=np.uint8),
        values=np.asarray([value[0] for _, value in ordered], dtype=np.float32),
        drop_policies=np.asarray([value[1] for _, value in ordered], dtype=np.float32),
        check_policies=np.asarray([value[2] for _, value in ordered], dtype=np.float32),
        saddle_gaps=np.asarray([value[3] for _, value in ordered], dtype=np.float32),
        value_semantics=np.asarray([value[4] for _, value in ordered], dtype=np.uint8),
        drop_actions=np.asarray(DROPPER_ACTIONS, dtype=np.int16),
        check_actions=np.asarray(CHECKER_ACTIONS, dtype=np.int16),
        dataset_version=np.asarray(dataset_version),
        emission=np.asarray("merged_reachable"),
        schema_version=np.asarray(TARGET_SCHEMA),
    )
    print(f"Merged {len(ordered)} targets to {destination}", flush=True)
    return destination


def generate_strategic_targets(
    *,
    output: str | Path,
    target_sets: Sequence[Mapping[str, int]],
    st_values: Sequence[int],
    ttd_values: Sequence[int],
    forced_roots: Iterable[Sequence[int]],
    seed: int,
    progress_every: int = 100,
    dataset_version: str = "strategic_exact_v1",
) -> Path:
    """Solve a horizon-balanced, stratified set of root targets only."""

    if progress_every < 0:
        raise ValueError("progress_every must be non-negative")

    requested: list[tuple[NTState, int]] = []
    for target_set in target_sets:
        horizon = int(target_set["horizon"])
        count = int(target_set["count"])
        if not 1 <= horizon <= 255:
            raise ValueError(f"horizon must be in 1..255, got {horizon}")
        roots = sample_strategic_roots(
            count=count,
            st_values=st_values,
            ttd_values=ttd_values,
            forced_roots=forced_roots,
            seed=seed + 1009 * horizon,
        )
        requested.extend((state, horizon) for state in roots)

    identities = tuple(dict.fromkeys(requested))
    if len(identities) != len(requested):
        raise ValueError("strategic target sets contain duplicate state/horizon rows")

    clear_solver_cache()
    solved: list[tuple[NTState, int, Solution]] = []
    solve_order = sorted(identities, key=lambda item: (-item[1], item[0]))
    total = len(solve_order)
    for index, (state, horizon) in enumerate(solve_order, start=1):
        solved.append((state, horizon, solve(state, horizon)))
        if progress_every and index % progress_every == 0:
            print(f"Solved {index}/{total} strategic roots", flush=True)

    rows = sorted(solved, key=lambda row: (row[1], row[0]))
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        states=np.asarray([row[0] for row in rows], dtype=np.int16),
        horizons=np.asarray([row[1] for row in rows], dtype=np.uint8),
        values=np.asarray([row[2].value for row in rows], dtype=np.float32),
        drop_policies=np.asarray(
            [row[2].drop_policy for row in rows], dtype=np.float32
        ),
        check_policies=np.asarray(
            [row[2].check_policy for row in rows], dtype=np.float32
        ),
        saddle_gaps=np.asarray(
            [row[2].saddle_gap for row in rows], dtype=np.float32
        ),
        drop_actions=np.asarray(DROPPER_ACTIONS, dtype=np.int16),
        check_actions=np.asarray(CHECKER_ACTIONS, dtype=np.int16),
        sampling_strata=np.asarray(
            [strategic_stratum(row[0]) for row in rows], dtype=np.str_
        ),
        dataset_version=np.asarray(dataset_version),
        emission=np.asarray("roots_only"),
        seed=np.asarray(seed, dtype=np.int64),
        schema_version=np.asarray(TARGET_SCHEMA),
    )
    print(f"Wrote {len(rows)} strategic targets to {destination}", flush=True)
    return destination


def generate_paired_orientation_targets(
    *,
    train_output: str | Path,
    holdout_output: str | Path,
    pairs: Sequence[Mapping[str, object]],
    train_orientation: str,
    progress_every: int = 100,
    dataset_version: str = "paired_orientation_v1",
) -> tuple[Path, Path]:
    """Write exact root-only targets for one orientation and its held-out mirror.

    Each pair declares the primary orientation through ``state`` and one or more
    horizons.  The opposite role orientation is derived mechanically, which
    prevents a hand-written holdout from drifting away from its training pair.
    """

    if train_orientation not in {"primary", "mirror"}:
        raise ValueError("train_orientation must be 'primary' or 'mirror'")
    if progress_every < 0:
        raise ValueError("progress_every must be non-negative")
    if not pairs:
        raise ValueError("paired orientation targets require at least one pair")

    primary_rows: list[tuple[NTState, int]] = []
    mirror_rows: list[tuple[NTState, int]] = []
    for pair_index, pair in enumerate(pairs):
        if "state" not in pair or "horizons" not in pair:
            raise ValueError(
                f"paired orientation pair {pair_index} needs state and horizons"
            )
        primary = _normalize_state(pair["state"])
        mirrored = mirror_state(primary)
        if primary == mirrored:
            raise ValueError(
                f"paired orientation pair {pair_index} is self-mirrored: {primary!r}"
            )
        horizons = tuple(int(value) for value in pair["horizons"])
        if not horizons:
            raise ValueError(f"paired orientation pair {pair_index} has no horizons")
        for horizon in horizons:
            if not 1 <= horizon <= 255:
                raise ValueError(
                    f"paired orientation horizon must be in 1..255, got {horizon}"
                )
            primary_rows.append((primary, horizon))
            mirror_rows.append((mirrored, horizon))

    primary_identities = set(primary_rows)
    mirror_identities = set(mirror_rows)
    if len(primary_identities) != len(primary_rows):
        raise ValueError("paired orientation primary rows contain duplicates")
    if len(mirror_identities) != len(mirror_rows):
        raise ValueError("paired orientation mirror rows contain duplicates")
    if primary_identities.intersection(mirror_identities):
        raise ValueError("paired orientation train and holdout identities overlap")

    train_identities, holdout_identities = (
        (primary_identities, mirror_identities)
        if train_orientation == "primary"
        else (mirror_identities, primary_identities)
    )
    clear_solver_cache()
    solve_order = sorted(
        train_identities.union(holdout_identities),
        key=lambda item: (-item[1], item[0]),
    )
    solutions: dict[tuple[NTState, int], Solution] = {}
    total = len(solve_order)
    for index, (state, horizon) in enumerate(solve_order, start=1):
        solutions[(state, horizon)] = solve(state, horizon)
        if progress_every and index % progress_every == 0:
            print(f"Solved {index}/{total} paired orientation roots", flush=True)

    def write_artifact(
        destination: str | Path,
        identities: set[tuple[NTState, int]],
        *,
        role: str,
    ) -> Path:
        ordered = sorted(identities, key=lambda item: (item[1], item[0]))
        rows = [(state, horizon, solutions[(state, horizon)]) for state, horizon in ordered]
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            states=np.asarray([row[0] for row in rows], dtype=np.int16),
            horizons=np.asarray([row[1] for row in rows], dtype=np.uint8),
            values=np.asarray([row[2].value for row in rows], dtype=np.float32),
            drop_policies=np.asarray(
                [row[2].drop_policy for row in rows], dtype=np.float32
            ),
            check_policies=np.asarray(
                [row[2].check_policy for row in rows], dtype=np.float32
            ),
            saddle_gaps=np.asarray(
                [row[2].saddle_gap for row in rows], dtype=np.float32
            ),
            drop_actions=np.asarray(DROPPER_ACTIONS, dtype=np.int16),
            check_actions=np.asarray(CHECKER_ACTIONS, dtype=np.int16),
            dataset_version=np.asarray(f"{dataset_version}_{role}"),
            emission=np.asarray("paired_mirror_roots"),
            paired_role=np.asarray(role),
            schema_version=np.asarray(TARGET_SCHEMA),
        )
        return path

    train_path = write_artifact(train_output, train_identities, role="train")
    holdout_path = write_artifact(
        holdout_output,
        holdout_identities,
        role="heldout_mirror",
    )
    print(
        f"Wrote {len(train_identities)} training and {len(holdout_identities)} "
        f"held-out paired targets",
        flush=True,
    )
    return train_path, holdout_path


def generate_resolve_labeled_targets(
    *,
    output: str | Path,
    report_output: str | Path,
    games: int,
    max_half_rounds: int,
    seed: int,
    label_depth: int,
    label_deadline_seconds: float,
    max_resolves: int,
    leaf_horizon: int,
    dataset_version: str = "resolve_labeled_v1",
    starts: Sequence[Sequence[int]] | None = None,
    agent=None,
    checkpoint: str | Path | None = None,
    tablebase: str | Path | None = None,
) -> Path:
    """Expert-iteration coverage: label self-play states with the resolve.

    Self-play trajectories run under the bounded-resolve agent itself; every
    unlabeled state met on a trajectory triggers one depth-limited resolve
    whose interior solutions all become training rows.  Rows carry the
    agent's query horizon, because their values are depth-amplified play
    estimates, not finite-horizon certificates; every row is marked
    ``value_semantics=1`` so training routes it to the play head, and merges
    must list this artifact before exact closures so exact rows win
    collisions.
    """

    if games <= 0 or max_half_rounds <= 0:
        raise ValueError("games and max_half_rounds must be positive")
    if label_depth <= 0 or label_deadline_seconds <= 0.0:
        raise ValueError("label depth and deadline must be positive")
    if max_resolves <= 0:
        raise ValueError("max_resolves must be positive")

    if agent is None:
        from dth.research_agent import (
            BoundedResolveAgent,
            NetworkLeafModel,
            ResolveBudget,
        )

        if checkpoint is None:
            raise ValueError("resolve labelling needs a checkpoint or an agent")
        agent = BoundedResolveAgent(
            complete_path=tablebase,
            network=NetworkLeafModel(checkpoint),
            budget=ResolveBudget(
                deadline_seconds=label_deadline_seconds,
                max_depth=label_depth,
                leaf_horizon=leaf_horizon,
            ),
        )

    started = time.monotonic()
    rng = np.random.default_rng(seed)
    labeled: dict[NTState, tuple[float, np.ndarray, np.ndarray, float]] = {}
    resolves = 0
    trajectory_states = 0

    def sample_policy(policy: np.ndarray) -> int:
        weights = np.clip(np.asarray(policy, dtype=np.float64), 0.0, None)
        weights /= weights.sum()
        return int(rng.choice(60, p=weights)) + 1

    start_states = (
        tuple(_normalize_state(start) for start in starts)
        if starts
        else ((0, 0, 0, 0),)
    )
    with agent:
        for game_index in range(games):
            state: NTState = start_states[game_index % len(start_states)]
            for _ in range(max_half_rounds):
                trajectory_states += 1
                if state not in labeled and resolves < max_resolves:
                    harvest = agent.resolve_labels(
                        state,
                        depth=label_depth,
                        deadline_seconds=label_deadline_seconds,
                    )
                    resolves += 1
                    for child, row in harvest.items():
                        labeled.setdefault(child, row)
                row = labeled.get(state)
                if row is not None:
                    drop = sample_policy(row[1])
                    check = sample_policy(row[2])
                else:
                    decision = agent.decide(state)
                    drop = sample_policy(np.asarray(decision.drop_policy))
                    check = sample_policy(np.asarray(decision.check_policy))
                branches = transition(state, drop, check)
                probabilities = np.asarray(
                    [probability for probability, _ in branches]
                )
                child = branches[
                    int(rng.choice(len(branches), p=probabilities))
                ][1]
                if not isinstance(child, tuple):
                    break
                state = child

    ordered = sorted(labeled.items(), key=lambda item: item[0])
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        states=np.asarray([state for state, _ in ordered], dtype=np.int16),
        horizons=np.full(len(ordered), leaf_horizon, dtype=np.uint8),
        values=np.asarray([row[0] for _, row in ordered], dtype=np.float32),
        drop_policies=np.asarray([row[1] for _, row in ordered], dtype=np.float32),
        check_policies=np.asarray(
            [row[2] for _, row in ordered], dtype=np.float32
        ),
        saddle_gaps=np.asarray([row[3] for _, row in ordered], dtype=np.float32),
        value_semantics=np.full(len(ordered), VALUE_SEMANTICS_PLAY, dtype=np.uint8),
        drop_actions=np.asarray(DROPPER_ACTIONS, dtype=np.int16),
        check_actions=np.asarray(CHECKER_ACTIONS, dtype=np.int16),
        dataset_version=np.asarray(dataset_version),
        emission=np.asarray("resolve_labeled"),
        label_depth=np.asarray(label_depth, dtype=np.int64),
        leaf_horizon=np.asarray(leaf_horizon, dtype=np.int64),
        seed=np.asarray(seed, dtype=np.int64),
        schema_version=np.asarray(TARGET_SCHEMA),
    )
    report_path = Path(report_output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": "dth-resolve-labeled-report-v1",
        "dataset_version": dataset_version,
        "emission": "resolve_labeled",
        "artifact": {
            "path": str(destination),
            "bytes": destination.stat().st_size,
            "sha256": _sha256(destination),
            "rows": len(ordered),
        },
        "games": games,
        "trajectory_states": trajectory_states,
        "resolves": resolves,
        "max_resolves": max_resolves,
        "label_depth": label_depth,
        "leaf_horizon": leaf_horizon,
        "elapsed_seconds": time.monotonic() - started,
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {len(ordered)} resolve-labeled targets from {resolves} resolves "
        f"to {destination}",
        flush=True,
    )
    return destination


@hydra.main(version_base="1.3", config_path="config", config_name="dataset")
def main(config: DictConfig) -> None:
    values = OmegaConf.to_container(config, resolve=True)
    if not isinstance(values, dict):
        raise TypeError("dataset config must resolve to a mapping")
    emission = str(values.get("emission", "reachable_layers"))
    if emission == "reachable_layers":
        generate_exact_targets(
            output=str(values["output"]),
            horizon=int(values["horizon"]),
            root_states=values["root_states"],
            progress_every=int(values["progress_every"]),
            base_datasets=values.get("base_datasets", ()),
            dataset_version=str(
                values.get("dataset_version", Path(str(values["output"])).stem)
            ),
        )
        return
    if emission == "roots_only":
        sampler = values["sampler"]
        if not isinstance(sampler, dict):
            raise TypeError("sampler config must resolve to a mapping")
        generate_strategic_targets(
            output=str(values["output"]),
            target_sets=values["target_sets"],
            st_values=sampler["st_values"],
            ttd_values=sampler["ttd_values"],
            forced_roots=sampler["forced_roots"],
            seed=int(values["seed"]),
            progress_every=int(values["progress_every"]),
            dataset_version=str(values["dataset_version"]),
        )
        return
    if emission == "paired_mirror_roots":
        pairs = values["pairs"]
        if not isinstance(pairs, list):
            raise TypeError("paired orientation pairs must resolve to a list")
        generate_paired_orientation_targets(
            train_output=str(values["train_output"]),
            holdout_output=str(values["holdout_output"]),
            pairs=pairs,
            train_orientation=str(values["train_orientation"]),
            progress_every=int(values["progress_every"]),
            dataset_version=str(values["dataset_version"]),
        )
        return
    if emission == "boundary_tablebase_closure":
        pairs = values["pairs"]
        if not isinstance(pairs, list):
            raise TypeError("boundary tablebase pairs must resolve to a list")
        generate_boundary_tablebase(
            output=str(values["output"]),
            report_output=str(values["report_output"]),
            pairs=pairs,
            progress_every=int(values["progress_every"]),
            dataset_version=str(values["dataset_version"]),
        )
        return
    if emission == "merge":
        inputs = values["inputs"]
        if not isinstance(inputs, list) or not inputs:
            raise TypeError("merge emission requires a non-empty inputs list")
        merge_exact_target_artifacts(
            [str(path) for path in inputs],
            str(values["output"]),
            dataset_version=str(values["dataset_version"]),
        )
        return
    if emission == "resolve_labeled":
        generate_resolve_labeled_targets(
            output=str(values["output"]),
            report_output=str(values["report_output"]),
            games=int(values["games"]),
            max_half_rounds=int(values["max_half_rounds"]),
            seed=int(values["seed"]),
            label_depth=int(values["label_depth"]),
            label_deadline_seconds=float(values["label_deadline_seconds"]),
            max_resolves=int(values["max_resolves"]),
            leaf_horizon=int(values["leaf_horizon"]),
            dataset_version=str(values["dataset_version"]),
            starts=values.get("starts"),
            checkpoint=values.get("checkpoint"),
            tablebase=values.get("tablebase"),
        )
        return
    raise ValueError(f"unsupported emission mode {emission!r}")


if __name__ == "__main__":
    main()
