"""Hydra-configured exact-target generation for pure DTH."""

from __future__ import annotations

from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from dth.solver import (
    CHECKER_ACTIONS,
    DROPPER_ACTIONS,
    NTState,
    Solution,
    solve,
    transition,
)


TARGET_SCHEMA = "dth-v1-ttd-strict-overflow"


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
    solve.cache_clear()

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
    """Merge exact artifacts by state/horizon identity, keeping later rows."""

    records: dict[tuple[int, NTState], tuple[float, np.ndarray, np.ndarray, float]] = {}
    for source in inputs:
        with np.load(Path(source), allow_pickle=False) as artifact:
            if str(np.asarray(artifact["schema_version"]).item()) != TARGET_SCHEMA:
                raise ValueError(f"target schema mismatch in {source}")
            states = artifact["states"]
            horizons = artifact["horizons"]
            for index in range(len(states)):
                state = tuple(int(value) for value in states[index])
                key = (int(horizons[index]), state)
                records[key] = (
                    float(artifact["values"][index]),
                    artifact["drop_policies"][index].astype(np.float32, copy=True),
                    artifact["check_policies"][index].astype(np.float32, copy=True),
                    float(artifact["saddle_gaps"][index]),
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

    solve.cache_clear()
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
    raise ValueError(f"unsupported emission mode {emission!r}")


if __name__ == "__main__":
    main()
