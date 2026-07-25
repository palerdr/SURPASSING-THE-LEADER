"""Exact Bellman-closure artifacts for the bounded Generation-Zero bridge."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Callable, Mapping, Sequence

import numpy as np

from stl.engine.actions import ACTION_SIZE
from stl.engine.game import (
    PHYSICALITY_BAKU,
    PHYSICALITY_HAL,
    Game,
    Player,
    Referee,
)
from stl.learning.certificates import (
    ExactPolicyCertificate,
    certificate_from_result,
)
from stl.learning.contracts import config_digest
from stl.learning.model import extract_features
from stl.learning.replay import (
    StateOrigin,
    TargetKind,
    TrainingRecordV3,
    exact_state_hash,
    reconstruct_game,
)
from stl.learning.tactical_anchors import BOUNDARY_CLOCKS, BOUNDARY_HISTORIES
from stl.solver.exact import (
    ExactPublicState,
    ExactSearchConfig,
    enumerate_joint_actions,
    exact_public_state,
    expand_joint_action,
    solve_exact_finite_horizon,
    solve_minimax,
    terminal_value,
)
from stl.solver.search import generate_candidates


BELLMAN_SCHEMA = "stl.bellman-closure.v1"
BELLMAN_MANIFEST_SCHEMA = "stl.bellman-closure-manifest.v1"
BELLMAN_PLAN_SCHEMA = "stl.bellman-causal-grid.v1"
BELLMAN_IDENTITY_TOLERANCE = 1.0e-8
BELLMAN_RECHECK_TOLERANCE = 1.0e-6

CHECKER_CYLINDERS = (180, 239, 240, 241, 242, 269, 270, 299)
OTHER_CYLINDERS = (0, 120, 240, 299)


@dataclass(frozen=True)
class BellmanRootSpec:
    name: str
    game: Game
    strata: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class BellmanSuccessor:
    state_hash: str
    state: ExactPublicState
    value_h2: float
    unresolved_h2: float
    result_h2: object | None


@dataclass(frozen=True)
class BellmanBranch:
    root_index: int
    drop_second: int
    check_second: int
    probability: float
    successor_index: int


@dataclass(frozen=True)
class BellmanRoot:
    name: str
    state_hash: str
    state: ExactPublicState
    strata: tuple[tuple[str, str], ...]
    value_h2: float
    value_h3: float
    unresolved_h2: float
    unresolved_h3: float
    drop_actions: tuple[int, ...]
    check_actions: tuple[int, ...]
    q3_for_hal: np.ndarray
    result_h2: object | None
    result_h3: object | None


@dataclass(frozen=True)
class BellmanBundle:
    roots: tuple[BellmanRoot, ...]
    successors: tuple[BellmanSuccessor, ...]
    branches: tuple[BellmanBranch, ...]
    search_config_digest: str
    plan_digest: str


@dataclass(frozen=True)
class BellmanGateThresholds:
    successor_h2_mse: float = 0.01
    root_h3_mse: float = 0.005
    backed_root_mse: float = 0.005
    maximum_root_error: float = 0.10
    maximum_bellman_residual: float = 0.05
    maximum_saddle_gap: float = 0.04
    horizon_difference_mse: float = 0.005


def _base_game(*, clock: int, current_half: int) -> Game:
    game = Game(
        player1=Player(name="Hal", physicality=PHYSICALITY_HAL),
        player2=Player(name="Baku", physicality=PHYSICALITY_BAKU),
        referee=Referee(),
        first_dropper=None,
    )
    game.first_dropper = game.player1
    game.seed(0)
    game.game_clock = float(clock)
    game.current_half = int(current_half)
    return game


def _causal_root(
    *,
    current_half: int,
    checker_cylinder: int,
    other_cylinder: int,
    clock: int,
    history: tuple[str, int, int, int, int],
) -> BellmanRootSpec:
    game = _base_game(clock=clock, current_half=current_half)
    history_name, hal_deaths, hal_ttd, baku_deaths, baku_ttd = history
    game.player1.deaths = hal_deaths
    game.player1.ttd = float(hal_ttd)
    game.player2.deaths = baku_deaths
    game.player2.ttd = float(baku_ttd)
    game.referee.cprs_performed = hal_deaths + baku_deaths
    checker = game.player2 if current_half == 1 else game.player1
    other = game.player1 if current_half == 1 else game.player2
    checker.cylinder = float(checker_cylinder)
    other.cylinder = float(other_cylinder)
    name = (
        f"bellman_h{current_half}_checker{checker_cylinder}_other{other_cylinder}_"
        f"clock{clock}_{history_name}"
    )
    severity = (
        "fresh"
        if hal_deaths + baku_deaths == 0
        else "deep"
        if max(hal_ttd, baku_ttd) >= 300
        else "shallow"
    )
    checker_band = (
        "pressure"
        if checker_cylinder < 239
        else "fixture"
        if checker_cylinder <= 242
        else "full_width"
        if checker_cylinder <= 270
        else "fatal_edge"
    )
    return BellmanRootSpec(
        name=name,
        game=game,
        strata=tuple(
            sorted(
                {
                    "checker": checker.name.lower(),
                    "checker_cylinder": str(checker_cylinder),
                    "checker_band": checker_band,
                    "checker_band_by_role": f"{checker.name.lower()}:{checker_band}",
                    "other_cylinder": str(other_cylinder),
                    "clock": str(clock),
                    "clock_class": "leap" if 3540 <= clock <= 3599 else "normal",
                    "history": history_name,
                    "history_severity": severity,
                    "clock_history": (
                        f"{'leap' if 3540 <= clock <= 3599 else 'normal'}:{severity}"
                    ),
                }.items()
            )
        ),
    )


def causal_root_pool() -> tuple[BellmanRootSpec, ...]:
    """Enumerate the fixed, engine-valid causal grid before split selection."""

    return tuple(
        _causal_root(
            current_half=half,
            checker_cylinder=checker,
            other_cylinder=other,
            clock=clock,
            history=history,
        )
        for half in (1, 2)
        for checker in CHECKER_CYLINDERS
        for other in OTHER_CYLINDERS
        for clock in BOUNDARY_CLOCKS
        for history in BOUNDARY_HISTORIES
    )


def _rank(spec: BellmanRootSpec, *, salt: str) -> str:
    return hashlib.sha256(
        f"{BELLMAN_PLAN_SCHEMA}:{salt}:{exact_state_hash(exact_public_state(spec.game))}".encode(
            "ascii"
        )
    ).hexdigest()


def bellman_closure_hashes(
    spec: BellmanRootSpec,
    *,
    config: ExactSearchConfig | None = None,
) -> frozenset[str]:
    """Return the exact root plus every state reachable after one joint action.

    This is deliberately label-free: it calls the engine transition expansion
    but performs no horizon-2 or horizon-3 solve.  Split allocation can
    therefore reserve a root's complete Bellman footprint before generating
    any values or exposing any model-selection evidence.
    """

    config = config or ExactSearchConfig()
    hashes = {exact_state_hash(exact_public_state(spec.game))}
    # ExactPublicState intentionally excludes literal action history. Under the
    # engine's single transition function, successful cells affect physical
    # state only through check_time - drop_time + 1, while every failed cell adds
    # the same fixed penalty. These keys are therefore an exact transition
    # quotient, not an action abstraction: every distinct successor class is
    # still expanded through the engine, including all survival branches.
    representatives = {}
    for action in enumerate_joint_actions(spec.game, config):
        key = (
            ("success", action.check_time - action.drop_time)
            if action.check_time >= action.drop_time
            else ("failure", 0)
        )
        representatives.setdefault(key, action)
    for action in representatives.values():
        hashes.update(
            exact_state_hash(outcome.state)
            for outcome in expand_joint_action(spec.game, action, config)
        )
    return frozenset(hashes)


def select_causal_roots(
    *,
    count: int,
    salt: str,
    blocked_state_hashes: set[str] | None = None,
    blocked_closure_hashes: set[str] | None = None,
    closure_cache: dict[str, frozenset[str]] | None = None,
    progress: Callable[[int, int], None] | None = None,
    require_full_coverage: bool = True,
) -> tuple[BellmanRootSpec, ...]:
    """Select deterministic roots whose full one-step closures are unblocked."""

    if count <= 0:
        raise ValueError("Bellman root count must be positive")
    blocked_roots = set(blocked_state_hashes or ())
    closure_aware = blocked_closure_hashes is not None
    blocked_closure = set(blocked_closure_hashes or ())
    available = [
        item
        for item in causal_root_pool()
        if exact_state_hash(exact_public_state(item.game)) not in blocked_roots
    ]
    cache = closure_cache if closure_cache is not None else {}
    newly_computed = 0

    def footprint(item: BellmanRootSpec) -> frozenset[str]:
        nonlocal newly_computed
        root_hash = exact_state_hash(exact_public_state(item.game))
        stored = cache.get(root_hash)
        if stored is None:
            stored = bellman_closure_hashes(item)
            cache[root_hash] = stored
            newly_computed += 1
            if progress is not None:
                progress(newly_computed, len(available))
        return stored

    candidates = (
        [item for item in available if footprint(item).isdisjoint(blocked_closure)]
        if closure_aware
        else list(available)
    )
    if len(candidates) < count:
        raise ValueError(
            "not enough causal roots with closures disjoint from reserved states"
        )
    candidates.sort(key=lambda item: _rank(item, salt=salt))
    selected: list[BellmanRootSpec] = []
    remaining = list(candidates)
    factor_names = (
        "checker",
        "checker_cylinder",
        "other_cylinder",
        "clock",
        "history",
        "checker_band",
        "checker_band_by_role",
        "clock_class",
        "history_severity",
        "clock_history",
    )
    required = {
        (name, value)
        for candidate in available
        for name, value in candidate.strata
        if name in factor_names
    }
    covered: set[tuple[str, str]] = set()
    selected_root_hashes: set[str] = set()
    selected_successor_hashes: set[str] = set()
    while len(selected) < count:
        compatible = list(remaining)
        if closure_aware:
            compatible = []
            for item in remaining:
                root_hash = exact_state_hash(exact_public_state(item.game))
                item_footprint = footprint(item)
                if root_hash in selected_successor_hashes:
                    continue
                if item_footprint.isdisjoint(selected_root_hashes):
                    compatible.append(item)
        if not compatible:
            raise ValueError(
                "root quota cannot be filled without selecting a root that is "
                "another selected root's successor"
            )
        best = max(
            compatible,
            key=lambda item: (
                len(set(item.strata) & (required - covered)),
                -int(_rank(item, salt=salt), 16),
            ),
        )
        selected.append(best)
        remaining.remove(best)
        if closure_aware:
            best_hash = exact_state_hash(exact_public_state(best.game))
            selected_root_hashes.add(best_hash)
            selected_successor_hashes.update(footprint(best) - {best_hash})
        covered.update(best.strata)
    if require_full_coverage and required - covered:
        raise ValueError(
            f"root quota {count} cannot cover strata {sorted(required - covered)}"
        )
    return tuple(selected)


def _config_digest(config: ExactSearchConfig) -> str:
    return config_digest(asdict(config))


def _plan_digest(root_hashes: Sequence[str], search_digest: str) -> str:
    return hashlib.sha256(
        json.dumps(
            {
                "schema": BELLMAN_PLAN_SCHEMA,
                "roots": list(root_hashes),
                "search_config_digest": search_digest,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()


def build_bellman_bundle(
    specs: Sequence[BellmanRootSpec],
    *,
    config: ExactSearchConfig | None = None,
    maximum_unresolved: float = 1.0,
) -> BellmanBundle:
    """Build exact V3 roots and the complete unique V2 successor closure.

    ``unresolved_probability`` is retained as label metadata, not treated as
    solver error: under terminal-only finite-horizon minimax, a position that
    safely reaches the configured frontier can legitimately have unresolved
    mass one and exact truncated value zero.
    """

    config = config or ExactSearchConfig()
    if not 0.0 <= maximum_unresolved <= 1.0:
        raise ValueError("maximum_unresolved must be in [0, 1]")
    search_digest = _config_digest(config)
    root_hashes = [exact_state_hash(exact_public_state(item.game)) for item in specs]
    if len(root_hashes) != len(set(root_hashes)):
        raise ValueError("Bellman root specs contain duplicate exact states")
    plan_digest = _plan_digest(root_hashes, search_digest)
    successor_rows: list[BellmanSuccessor] = []
    successor_index: dict[str, int] = {}
    branches: list[BellmanBranch] = []
    roots: list[BellmanRoot] = []

    for root_index, spec in enumerate(specs):
        state = exact_public_state(spec.game)
        result_h2 = solve_exact_finite_horizon(spec.game, 2, config)
        result_h3 = solve_exact_finite_horizon(spec.game, 3, config)
        if result_h3.payoff_for_hal is None:
            raise ValueError(f"Bellman root {spec.name!r} is terminal")
        d_index = {second: i for i, second in enumerate(result_h3.drop_actions)}
        c_index = {second: i for i, second in enumerate(result_h3.check_actions)}
        q3 = np.zeros_like(result_h3.payoff_for_hal, dtype=np.float64)
        cell_probability = np.zeros_like(q3)
        for action in enumerate_joint_actions(spec.game, config):
            for outcome in expand_joint_action(spec.game, action, config):
                successor_hash = exact_state_hash(outcome.state)
                index = successor_index.get(successor_hash)
                if index is None:
                    successor_game = reconstruct_game(outcome.state)
                    successor_result = solve_exact_finite_horizon(
                        successor_game, 2, config
                    )
                    index = len(successor_rows)
                    successor_index[successor_hash] = index
                    successor_rows.append(
                        BellmanSuccessor(
                            state_hash=successor_hash,
                            state=outcome.state,
                            value_h2=float(successor_result.value_for_hal),
                            unresolved_h2=float(
                                successor_result.unresolved_probability
                            ),
                            result_h2=successor_result,
                        )
                    )
                branch = BellmanBranch(
                    root_index=root_index,
                    drop_second=action.drop_time,
                    check_second=action.check_time,
                    probability=float(outcome.probability),
                    successor_index=index,
                )
                branches.append(branch)
                i = d_index[action.drop_time]
                j = c_index[action.check_time]
                q3[i, j] += branch.probability * successor_rows[index].value_h2
                cell_probability[i, j] += branch.probability
        if not np.allclose(cell_probability, 1.0, atol=1.0e-12):
            raise ValueError(f"Bellman root {spec.name!r} has incomplete chance cells")
        if not np.allclose(
            q3,
            result_h3.payoff_for_hal,
            atol=BELLMAN_IDENTITY_TOLERANCE,
            rtol=0.0,
        ):
            raise ValueError(f"Bellman root {spec.name!r} payoff identity failed")
        dropper, _checker = spec.game.get_roles_for_half(spec.game.current_half)
        matrix = q3 if dropper.name.lower() == "hal" else -q3
        _strategy, backed_value = solve_minimax(matrix)
        backed_for_hal = backed_value if dropper.name.lower() == "hal" else -backed_value
        if abs(backed_for_hal - result_h3.value_for_hal) > BELLMAN_IDENTITY_TOLERANCE:
            raise ValueError(f"Bellman root {spec.name!r} value identity failed")
        roots.append(
            BellmanRoot(
                name=spec.name,
                state_hash=exact_state_hash(state),
                state=state,
                strata=spec.strata,
                value_h2=float(result_h2.value_for_hal),
                value_h3=float(result_h3.value_for_hal),
                unresolved_h2=float(result_h2.unresolved_probability),
                unresolved_h3=float(result_h3.unresolved_probability),
                drop_actions=result_h3.drop_actions,
                check_actions=result_h3.check_actions,
                q3_for_hal=q3,
                result_h2=result_h2,
                result_h3=result_h3,
            )
        )

    unresolved = [root.unresolved_h3 for root in roots] + [
        row.unresolved_h2 for row in successor_rows
    ]
    if unresolved and max(unresolved) > maximum_unresolved + 1.0e-12:
        raise ValueError(
            f"Bellman unresolved probability {max(unresolved):.6f} exceeds "
            f"{maximum_unresolved:.6f}"
        )
    return BellmanBundle(
        roots=tuple(roots),
        successors=tuple(successor_rows),
        branches=tuple(branches),
        search_config_digest=search_digest,
        plan_digest=plan_digest,
    )


def merge_bellman_bundles(
    bundles: Sequence[BellmanBundle],
) -> BellmanBundle:
    """Merge ordered, independently committed root bundles losslessly.

    Successors are deduplicated by exact-state hash and branch indices are
    remapped.  The resulting plan digest is identical to a single uninterrupted
    ``build_bellman_bundle`` call over the same ordered roots.
    """

    if not bundles:
        raise ValueError("at least one Bellman bundle is required")
    search_digest = bundles[0].search_config_digest
    roots: list[BellmanRoot] = []
    successors: list[BellmanSuccessor] = []
    successor_indices: dict[str, int] = {}
    branches: list[BellmanBranch] = []
    seen_roots: set[str] = set()

    for bundle in bundles:
        if bundle.search_config_digest != search_digest:
            raise ValueError("Bellman bundles use different search configurations")
        root_offset = len(roots)
        for root in bundle.roots:
            if root.state_hash in seen_roots:
                raise ValueError(f"duplicate Bellman root {root.state_hash}")
            seen_roots.add(root.state_hash)
            roots.append(root)

        local_to_merged: dict[int, int] = {}
        for local_index, successor in enumerate(bundle.successors):
            merged_index = successor_indices.get(successor.state_hash)
            if merged_index is None:
                merged_index = len(successors)
                successor_indices[successor.state_hash] = merged_index
                successors.append(successor)
            else:
                existing = successors[merged_index]
                if existing.state != successor.state:
                    raise ValueError("Bellman successor hash collision")
                if not np.isclose(
                    existing.value_h2,
                    successor.value_h2,
                    atol=BELLMAN_IDENTITY_TOLERANCE,
                    rtol=0.0,
                ):
                    raise ValueError("Bellman successor value mismatch")
                if not np.isclose(
                    existing.unresolved_h2,
                    successor.unresolved_h2,
                    atol=BELLMAN_IDENTITY_TOLERANCE,
                    rtol=0.0,
                ):
                    raise ValueError("Bellman successor cutoff mismatch")
            local_to_merged[local_index] = merged_index

        for branch in bundle.branches:
            if not 0 <= branch.root_index < len(bundle.roots):
                raise ValueError("Bellman branch has invalid local root index")
            if branch.successor_index not in local_to_merged:
                raise ValueError("Bellman branch has invalid local successor index")
            branches.append(
                BellmanBranch(
                    root_index=root_offset + branch.root_index,
                    drop_second=branch.drop_second,
                    check_second=branch.check_second,
                    probability=branch.probability,
                    successor_index=local_to_merged[branch.successor_index],
                )
            )

    merged = BellmanBundle(
        roots=tuple(roots),
        successors=tuple(successors),
        branches=tuple(branches),
        search_config_digest=search_digest,
        plan_digest=_plan_digest(
            [root.state_hash for root in roots], search_digest
        ),
    )
    _recheck_branch_identity(merged)
    return merged


def _policy_vectors(game: Game, result) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    drop = np.zeros(ACTION_SIZE, dtype=np.float32)
    check = np.zeros(ACTION_SIZE, dtype=np.float32)
    drop_mask = np.zeros(ACTION_SIZE, dtype=np.bool_)
    check_mask = np.zeros(ACTION_SIZE, dtype=np.bool_)
    if result.payoff_for_hal is not None:
        for second, probability in zip(result.drop_actions, result.dropper_strategy):
            drop[second] = float(probability)
            drop_mask[second] = True
        for second, probability in zip(result.check_actions, result.checker_strategy):
            check[second] = float(probability)
            check_mask[second] = True
    return drop, check, drop_mask, check_mask


def bundle_training_records(
    bundle: BellmanBundle,
    *,
    source_artifact: str,
) -> tuple[list[TrainingRecordV3], list[ExactPolicyCertificate]]:
    """Convert a Bellman bundle into V3 replay rows and policy certificates."""

    rows: list[TrainingRecordV3] = []
    certificates: list[ExactPolicyCertificate] = []

    def append_row(
        *,
        state: ExactPublicState,
        state_hash: str,
        value: float,
        result,
        horizon: int,
        source: str,
        active_policy: bool = True,
    ) -> None:
        if result is None:
            raise ValueError(
                "training records require an in-memory generated Bellman bundle"
            )
        game = reconstruct_game(state)
        drop, check, drop_mask, check_mask = _policy_vectors(game, result)
        if not active_policy:
            drop.fill(0.0)
            check.fill(0.0)
        is_terminal = terminal_value(game) is not None
        rows.append(
            TrainingRecordV3(
                features=extract_features(game),
                exact_state=state,
                value=value,
                target_kind=(
                    TargetKind.TERMINAL_OUTCOME
                    if is_terminal
                    else TargetKind.EXACT_VALUE
                ),
                source=source,
                dropper_dist=drop,
                checker_dist=check,
                dropper_legal_mask=drop_mask,
                checker_legal_mask=check_mask,
                value_horizon_half_rounds=0 if is_terminal else horizon,
                cutoff_probability=0.0 if is_terminal else result.unresolved_probability,
                state_origin=StateOrigin.TACTICAL_TABLEBASE,
                source_artifact=source_artifact,
                source_artifact_digest=bundle.plan_digest,
                episode_id=f"bellman:{state_hash}:{horizon}",
                search_config_digest=bundle.search_config_digest,
            )
        )
        if active_policy and result.payoff_for_hal is not None:
            certificates.append(
                certificate_from_result(
                    state_hash=state_hash,
                    search_config_digest=bundle.search_config_digest,
                    result=result,
                )
            )

    for root in bundle.roots:
        append_row(
            state=root.state,
            state_hash=root.state_hash,
            value=root.value_h2,
            result=root.result_h2,
            horizon=2,
            source="bellman_root_h2",
            active_policy=False,
        )
        append_row(
            state=root.state,
            state_hash=root.state_hash,
            value=root.value_h3,
            result=root.result_h3,
            horizon=3,
            source="bellman_root_h3",
        )
    root_hashes = {root.state_hash for root in bundle.roots}
    for successor in bundle.successors:
        if successor.state_hash in root_hashes:
            raise ValueError(
                "a Bellman state is both a root and successor; choose another split "
                "to keep one certificate horizon per physical state"
            )
        append_row(
            state=successor.state,
            state_hash=successor.state_hash,
            value=successor.value_h2,
            result=successor.result_h2,
            horizon=2,
            source="bellman_successor_h2",
        )
    return rows, certificates


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def bellman_manifest_path(path: str | Path) -> Path:
    source = Path(path)
    return source.with_name(source.name + ".manifest.json")


def save_bellman_bundle(bundle: BellmanBundle, path: str | Path) -> dict[str, object]:
    """Atomically publish a pickle-free Bellman graph and commit manifest."""

    destination = Path(path)
    manifest_path = bellman_manifest_path(destination)
    if destination.exists() or manifest_path.exists():
        raise ValueError(f"refusing to overwrite Bellman bundle {destination}")
    max_drop = ACTION_SIZE - 1
    max_check = ACTION_SIZE - 2
    q3 = np.full((len(bundle.roots), max_drop, max_check), np.nan, np.float64)
    drop_actions = np.full((len(bundle.roots), max_drop), -1, np.int16)
    check_actions = np.full((len(bundle.roots), max_check), -1, np.int16)
    for index, root in enumerate(bundle.roots):
        d_count, c_count = root.q3_for_hal.shape
        q3[index, :d_count, :c_count] = root.q3_for_hal
        drop_actions[index, :d_count] = root.drop_actions
        check_actions[index, :c_count] = root.check_actions
    arrays = {
        "root_names": np.asarray([row.name for row in bundle.roots], dtype=np.str_),
        "root_hashes": np.asarray([row.state_hash for row in bundle.roots], dtype="<U64"),
        "root_states_json": np.asarray(
            [json.dumps(asdict(row.state), sort_keys=True) for row in bundle.roots],
            dtype=np.str_,
        ),
        "root_strata_json": np.asarray(
            [json.dumps(dict(row.strata), sort_keys=True) for row in bundle.roots],
            dtype=np.str_,
        ),
        "root_values_h2": np.asarray([row.value_h2 for row in bundle.roots]),
        "root_values_h3": np.asarray([row.value_h3 for row in bundle.roots]),
        "root_unresolved_h2": np.asarray([row.unresolved_h2 for row in bundle.roots]),
        "root_unresolved_h3": np.asarray([row.unresolved_h3 for row in bundle.roots]),
        "root_drop_counts": np.asarray([len(row.drop_actions) for row in bundle.roots], dtype=np.int16),
        "root_check_counts": np.asarray([len(row.check_actions) for row in bundle.roots], dtype=np.int16),
        "root_drop_actions": drop_actions,
        "root_check_actions": check_actions,
        "root_q3_for_hal": q3,
        "successor_hashes": np.asarray([row.state_hash for row in bundle.successors], dtype="<U64"),
        "successor_states_json": np.asarray(
            [json.dumps(asdict(row.state), sort_keys=True) for row in bundle.successors],
            dtype=np.str_,
        ),
        "successor_values_h2": np.asarray([row.value_h2 for row in bundle.successors]),
        "successor_unresolved_h2": np.asarray([row.unresolved_h2 for row in bundle.successors]),
        "branch_root_indices": np.asarray([row.root_index for row in bundle.branches], dtype=np.int32),
        "branch_drop_seconds": np.asarray([row.drop_second for row in bundle.branches], dtype=np.int16),
        "branch_check_seconds": np.asarray([row.check_second for row in bundle.branches], dtype=np.int16),
        "branch_probabilities": np.asarray([row.probability for row in bundle.branches]),
        "branch_successor_indices": np.asarray([row.successor_index for row in bundle.branches], dtype=np.int32),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".npz", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    manifest_tmp = manifest_path.with_name(f".{manifest_path.name}.{os.getpid()}.tmp")
    try:
        np.savez_compressed(temporary, **arrays)
        manifest = {
            "schema": BELLMAN_MANIFEST_SCHEMA,
            "bundle_schema": BELLMAN_SCHEMA,
            "data_file": destination.name,
            "data_sha256": _sha256_file(temporary),
            "root_count": len(bundle.roots),
            "successor_count": len(bundle.successors),
            "branch_count": len(bundle.branches),
            "search_config_digest": bundle.search_config_digest,
            "plan_digest": bundle.plan_digest,
        }
        manifest_tmp.write_bytes(_canonical_json(manifest))
        os.replace(temporary, destination)
        os.replace(manifest_tmp, manifest_path)
        return manifest
    finally:
        temporary.unlink(missing_ok=True)
        manifest_tmp.unlink(missing_ok=True)


def _state_from_json(value: str) -> ExactPublicState:
    return ExactPublicState(**json.loads(value))


def load_bellman_bundle(path: str | Path) -> BellmanBundle:
    """Strict-load a Bellman artifact and recheck all stored identities."""

    source = Path(path)
    manifest = json.loads(bellman_manifest_path(source).read_text(encoding="utf-8"))
    if manifest.get("schema") != BELLMAN_MANIFEST_SCHEMA:
        raise ValueError("unsupported Bellman manifest schema")
    if manifest.get("data_sha256") != _sha256_file(source):
        raise ValueError("Bellman payload digest mismatch")
    with np.load(source, allow_pickle=False) as payload:
        data = {name: np.array(payload[name], copy=True) for name in payload.files}
    successors: list[BellmanSuccessor] = []
    for index, state_hash in enumerate(data["successor_hashes"]):
        state = _state_from_json(str(data["successor_states_json"][index]))
        if exact_state_hash(state) != str(state_hash):
            raise ValueError("Bellman successor exact-state hash mismatch")
        value = float(data["successor_values_h2"][index])
        successors.append(
            BellmanSuccessor(
                state_hash=str(state_hash),
                state=state,
                value_h2=value,
                unresolved_h2=float(data["successor_unresolved_h2"][index]),
                result_h2=None,
            )
        )
    branches = tuple(
        BellmanBranch(
            root_index=int(root),
            drop_second=int(drop),
            check_second=int(check),
            probability=float(probability),
            successor_index=int(successor),
        )
        for root, drop, check, probability, successor in zip(
            data["branch_root_indices"],
            data["branch_drop_seconds"],
            data["branch_check_seconds"],
            data["branch_probabilities"],
            data["branch_successor_indices"],
        )
    )
    roots: list[BellmanRoot] = []
    for index, state_hash in enumerate(data["root_hashes"]):
        state = _state_from_json(str(data["root_states_json"][index]))
        if exact_state_hash(state) != str(state_hash):
            raise ValueError("Bellman root exact-state hash mismatch")
        d_count = int(data["root_drop_counts"][index])
        c_count = int(data["root_check_counts"][index])
        q3 = data["root_q3_for_hal"][index, :d_count, :c_count]
        roots.append(
            BellmanRoot(
                name=str(data["root_names"][index]),
                state_hash=str(state_hash),
                state=state,
                strata=tuple(sorted(json.loads(str(data["root_strata_json"][index])).items())),
                value_h2=float(data["root_values_h2"][index]),
                value_h3=float(data["root_values_h3"][index]),
                unresolved_h2=float(data["root_unresolved_h2"][index]),
                unresolved_h3=float(data["root_unresolved_h3"][index]),
                drop_actions=tuple(int(value) for value in data["root_drop_actions"][index, :d_count]),
                check_actions=tuple(int(value) for value in data["root_check_actions"][index, :c_count]),
                q3_for_hal=q3,
                result_h2=None,
                result_h3=None,
            )
        )
    bundle = BellmanBundle(
        roots=tuple(roots),
        successors=tuple(successors),
        branches=branches,
        search_config_digest=str(manifest["search_config_digest"]),
        plan_digest=str(manifest["plan_digest"]),
    )
    _recheck_branch_identity(bundle)
    return bundle


def _recheck_branch_identity(bundle: BellmanBundle) -> None:
    for root_index, root in enumerate(bundle.roots):
        d_index = {second: i for i, second in enumerate(root.drop_actions)}
        c_index = {second: i for i, second in enumerate(root.check_actions)}
        observed = np.zeros_like(root.q3_for_hal)
        probabilities = np.zeros_like(root.q3_for_hal)
        for branch in bundle.branches:
            if branch.root_index != root_index:
                continue
            i = d_index[branch.drop_second]
            j = c_index[branch.check_second]
            observed[i, j] += branch.probability * bundle.successors[
                branch.successor_index
            ].value_h2
            probabilities[i, j] += branch.probability
        if not np.allclose(probabilities, 1.0, atol=1.0e-12):
            raise ValueError("Bellman branch probabilities do not cover every cell")
        if not np.allclose(
            observed, root.q3_for_hal, atol=BELLMAN_IDENTITY_TOLERANCE, rtol=0.0
        ):
            raise ValueError("Bellman branch backup does not reproduce root matrix")


def spot_recheck_bellman_bundle(
    bundle: BellmanBundle,
    *,
    root_count: int = 8,
    successor_count: int = 8,
    config: ExactSearchConfig | None = None,
) -> dict[str, object]:
    """Deterministically re-solve a bounded subset of a stored bundle."""

    config = config or ExactSearchConfig()
    selected_roots = sorted(bundle.roots, key=lambda row: row.state_hash)[:root_count]
    selected_successors = sorted(
        bundle.successors, key=lambda row: row.state_hash
    )[:successor_count]
    maximum_matrix_error = 0.0
    maximum_value_error = 0.0
    for root in selected_roots:
        result = solve_exact_finite_horizon(reconstruct_game(root.state), 3, config)
        if result.payoff_for_hal is None:
            raise ValueError("stored Bellman root became terminal during recheck")
        maximum_matrix_error = max(
            maximum_matrix_error,
            float(np.max(np.abs(result.payoff_for_hal - root.q3_for_hal))),
        )
        maximum_value_error = max(
            maximum_value_error, abs(result.value_for_hal - root.value_h3)
        )
    for successor in selected_successors:
        result = solve_exact_finite_horizon(
            reconstruct_game(successor.state), 2, config
        )
        maximum_value_error = max(
            maximum_value_error, abs(result.value_for_hal - successor.value_h2)
        )
    passed = (
        maximum_matrix_error <= BELLMAN_RECHECK_TOLERANCE
        and maximum_value_error <= BELLMAN_RECHECK_TOLERANCE
    )
    return {
        "schema": "stl.bellman-spot-recheck.v1",
        "root_count": len(selected_roots),
        "successor_count": len(selected_successors),
        "maximum_matrix_error": maximum_matrix_error,
        "maximum_value_error": maximum_value_error,
        "tolerance": BELLMAN_RECHECK_TOLERANCE,
        "passed": passed,
    }


def _saddle_gap(
    matrix: np.ndarray,
    dropper_policy: np.ndarray,
    checker_policy: np.ndarray,
    *,
    hal_is_dropper: bool,
) -> float:
    expected = float(dropper_policy @ matrix @ checker_policy)
    drop_values = matrix @ checker_policy
    check_values = dropper_policy @ matrix
    if hal_is_dropper:
        return max(0.0, float(drop_values.max()) - expected) + max(
            0.0, expected - float(check_values.min())
        )
    return max(0.0, expected - float(drop_values.min())) + max(
        0.0, float(check_values.max()) - expected
    )


def evaluate_bellman_gate(
    bundle: BellmanBundle,
    predict: Callable[..., tuple[float, np.ndarray, np.ndarray]],
    *,
    thresholds: BellmanGateThresholds | None = None,
) -> dict[str, object]:
    """Evaluate direct, backed-up, contrast, and saddle-gap evidence."""

    limits = thresholds or BellmanGateThresholds()
    successor_predictions = np.asarray(
        [
            terminal_value(reconstruct_game(row.state))
            if terminal_value(reconstruct_game(row.state)) is not None
            else predict(reconstruct_game(row.state), horizon=2)[0]
            for row in bundle.successors
        ],
        dtype=np.float64,
    )
    successor_exact = np.asarray([row.value_h2 for row in bundle.successors])
    direct_h2: list[float] = []
    direct_h3: list[float] = []
    backed_h3: list[float] = []
    saddle_gaps: list[float] = []
    for root_index, root in enumerate(bundle.roots):
        game = reconstruct_game(root.state)
        p2 = predict(game, horizon=2)
        p3 = predict(game, horizon=3)
        direct_h2.append(float(p2[0]))
        direct_h3.append(float(p3[0]))
        d_index = {second: i for i, second in enumerate(root.drop_actions)}
        c_index = {second: i for i, second in enumerate(root.check_actions)}
        qhat = np.zeros_like(root.q3_for_hal)
        for branch in bundle.branches:
            if branch.root_index == root_index:
                qhat[d_index[branch.drop_second], c_index[branch.check_second]] += (
                    branch.probability * successor_predictions[branch.successor_index]
                )
        dropper, _checker = game.get_roles_for_half(game.current_half)
        hal_is_dropper = dropper.name.lower() == "hal"
        matrix = qhat if hal_is_dropper else -qhat
        _strategy, value = solve_minimax(matrix)
        backed_h3.append(float(value if hal_is_dropper else -value))
        drop_policy = np.asarray([p3[1][second] for second in root.drop_actions])
        check_policy = np.asarray([p3[2][second] for second in root.check_actions])
        drop_policy /= drop_policy.sum()
        check_policy /= check_policy.sum()
        saddle_gaps.append(
            _saddle_gap(
                root.q3_for_hal,
                drop_policy,
                check_policy,
                hal_is_dropper=hal_is_dropper,
            )
        )
    direct_h2_arr = np.asarray(direct_h2)
    direct_h3_arr = np.asarray(direct_h3)
    backed_arr = np.asarray(backed_h3)
    exact_h2 = np.asarray([row.value_h2 for row in bundle.roots])
    exact_h3 = np.asarray([row.value_h3 for row in bundle.roots])
    successor_mse = float(np.mean((successor_predictions - successor_exact) ** 2))
    root_mse = float(np.mean((direct_h3_arr - exact_h3) ** 2))
    backed_mse = float(np.mean((backed_arr - exact_h3) ** 2))
    max_root_error = float(
        max(
            np.max(np.abs(direct_h3_arr - exact_h3)),
            np.max(np.abs(backed_arr - exact_h3)),
        )
    )
    residual = np.abs(direct_h3_arr - backed_arr)
    contrast_exact = exact_h3 - exact_h2
    contrast_predicted = direct_h3_arr - direct_h2_arr
    contrast_mse = float(np.mean((contrast_predicted - contrast_exact) ** 2))
    material = np.abs(contrast_exact) >= 0.10
    sign_ok = bool(
        np.all(np.sign(contrast_predicted[material]) == np.sign(contrast_exact[material]))
    )
    metrics = {
        "successor_h2_mse": successor_mse,
        "root_h3_mse": root_mse,
        "backed_root_mse": backed_mse,
        "maximum_root_error": max_root_error,
        "maximum_bellman_residual": float(np.max(residual)),
        "maximum_saddle_gap": float(max(saddle_gaps, default=0.0)),
        "horizon_difference_mse": contrast_mse,
        "material_horizon_signs_correct": sign_ok,
    }
    failures = [
        name
        for name in (
            "successor_h2_mse",
            "root_h3_mse",
            "backed_root_mse",
            "maximum_root_error",
            "maximum_bellman_residual",
            "maximum_saddle_gap",
            "horizon_difference_mse",
        )
        if float(metrics[name]) > float(getattr(limits, name))
    ]
    if not sign_ok:
        failures.append("material_horizon_signs_correct")
    return {
        "schema": "stl.bellman-gate-report.v1",
        "metrics": metrics,
        "thresholds": asdict(limits),
        "failures": failures,
        "passed": not failures,
    }


def candidate_action_representability(bundle: BellmanBundle) -> dict[str, object]:
    """Audit candidate action subsets against every full Bellman root matrix."""

    rows = []
    for root in bundle.roots:
        game = reconstruct_game(root.state)
        candidates = generate_candidates(game)
        d_index = {second: i for i, second in enumerate(root.drop_actions)}
        c_index = {second: i for i, second in enumerate(root.check_actions)}
        restricted = root.q3_for_hal[
            np.ix_(
                [d_index[second] for second in candidates.drop_seconds],
                [c_index[second] for second in candidates.check_seconds],
            )
        ]
        dropper, _checker = game.get_roles_for_half(game.current_half)
        hal_is_dropper = dropper.name.lower() == "hal"
        oriented = restricted if hal_is_dropper else -restricted
        drop_strategy, value = solve_minimax(oriented)
        check_strategy, _ = solve_minimax((-oriented).T)
        restricted_value = float(value if hal_is_dropper else -value)
        drop_full = np.zeros(len(root.drop_actions))
        check_full = np.zeros(len(root.check_actions))
        for second, probability in zip(candidates.drop_seconds, drop_strategy):
            drop_full[d_index[second]] = probability
        for second, probability in zip(candidates.check_seconds, check_strategy):
            check_full[c_index[second]] = probability
        gap = _saddle_gap(
            root.q3_for_hal,
            drop_full,
            check_full,
            hal_is_dropper=hal_is_dropper,
        )
        rows.append(
            {
                "root": root.name,
                "value_error": abs(restricted_value - root.value_h3),
                "full_width_saddle_gap": gap,
                "candidate_cells": candidates.joint_count,
                "full_cells": int(root.q3_for_hal.size),
            }
        )
    failures = [
        row["root"]
        for row in rows
        if row["value_error"] > 0.02 or row["full_width_saddle_gap"] > 0.05
    ]
    return {
        "schema": "stl.candidate-action-representability.v1",
        "roots": rows,
        "failures": failures,
        "passed": not failures,
    }


__all__ = [
    "BELLMAN_IDENTITY_TOLERANCE",
    "BELLMAN_MANIFEST_SCHEMA",
    "BELLMAN_SCHEMA",
    "BellmanBundle",
    "BellmanGateThresholds",
    "BellmanRootSpec",
    "build_bellman_bundle",
    "bundle_training_records",
    "candidate_action_representability",
    "causal_root_pool",
    "evaluate_bellman_gate",
    "load_bellman_bundle",
    "merge_bellman_bundles",
    "save_bellman_bundle",
    "select_causal_roots",
    "spot_recheck_bellman_bundle",
]
