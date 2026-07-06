"""Policy guard targets for drift-sensitive SolverAgent states.

These rows are champion-search labels meant to be mixed into saved-corpus
training when a calibration-improving candidate becomes too readable. They use
the same ``ValueTarget`` schema as the rest of the trainer, with source
``policy_guard`` and root average MCTS policies from ``SolverAgent.search()``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from environment.cfr.exact import ExactGameSnapshot, ExactSearchConfig
from hal.agent import SolverAgent
from hal.value_net import extract_features
from src.Game import Game
from training.value_targets import (
    SOURCE_POLICY_GUARD,
    ValueTarget,
    _legal_policy_vectors,
    _strategy_vectors,
    save_targets,
)


@dataclass(frozen=True)
class PolicyGuardMetadata:
    scenario: str
    seed: int
    checkpoint: str
    iterations: int
    policy_ensemble_size: int = 1
    policy_uniform_mix: float = 0.0
    resolve_at_critical: bool = False
    resolve_horizon: int = 3
    depth: int = 0
    drop_time: int | None = None
    check_time: int | None = None
    survival_outcome: str = "root"
    branch_probability: float = 1.0


@dataclass(frozen=True)
class PolicyGuardRecord:
    target: ValueTarget
    metadata: PolicyGuardMetadata


def policy_guard_target_for_game(
    *,
    game: Game,
    agent: SolverAgent,
    scenario: str,
    seed: int,
    checkpoint: str,
    source: str = SOURCE_POLICY_GUARD,
    config: ExactSearchConfig | None = None,
    depth: int = 0,
    drop_time: int | None = None,
    check_time: int | None = None,
    survival_outcome: str = "root",
    branch_probability: float = 1.0,
) -> PolicyGuardRecord:
    """Label one game state with the champion's deployed search object."""
    config = config or ExactSearchConfig()
    result = agent.search(game)
    drop_seconds, drop_probs = agent.policy(game, "dropper")
    check_seconds, check_probs = agent.policy(game, "checker")
    drop_dist, check_dist = _strategy_vectors(
        drop_seconds=drop_seconds,
        check_seconds=check_seconds,
        dropper_strategy=drop_probs,
        checker_strategy=check_probs,
    )
    _, _, drop_mask, check_mask = _legal_policy_vectors(game, config)
    return PolicyGuardRecord(
        target=ValueTarget(
            features=extract_features(game),
            value=float(result.root_value_for_hal),
            source=source,
            horizon=int(agent.iterations),
            dropper_dist=drop_dist,
            checker_dist=check_dist,
            dropper_legal_mask=drop_mask,
            checker_legal_mask=check_mask,
            unresolved_probability=0.0,
        ),
        metadata=PolicyGuardMetadata(
            scenario=scenario,
            seed=int(seed),
            checkpoint=str(checkpoint),
            iterations=int(agent.iterations),
            policy_ensemble_size=int(getattr(agent, "policy_ensemble_size", 1)),
            policy_uniform_mix=float(getattr(agent, "policy_uniform_mix", 0.0)),
            resolve_at_critical=bool(getattr(agent, "resolve_at_critical", False)),
            resolve_horizon=int(getattr(agent, "resolve_horizon", 3)),
            depth=int(depth),
            drop_time=drop_time,
            check_time=check_time,
            survival_outcome=str(survival_outcome),
            branch_probability=float(branch_probability),
        ),
    )


def _top_seconds(
    seconds: tuple[int, ...],
    probabilities: np.ndarray,
    top_k: int,
) -> tuple[int, ...]:
    if top_k <= 0 or not seconds:
        return ()
    probs = np.asarray(probabilities, dtype=np.float64)
    order = sorted(
        range(len(seconds)),
        key=lambda idx: (-float(probs[idx]) if idx < probs.size else 0.0, seconds[idx]),
    )
    return tuple(int(seconds[idx]) for idx in order[:top_k])


def _survival_label(survived: bool | None) -> str:
    if survived is None:
        return "none"
    return "survived" if survived else "died"


def child_policy_guard_targets_for_game(
    *,
    game: Game,
    agent: SolverAgent,
    scenario: str,
    seed: int,
    checkpoint: str,
    source: str = SOURCE_POLICY_GUARD,
    top_k: int = 2,
    config: ExactSearchConfig | None = None,
) -> list[PolicyGuardRecord]:
    """Label non-terminal one-step children under top root average actions."""
    config = config or ExactSearchConfig()
    if top_k <= 0 or game.game_over:
        return []

    root_drop_seconds, root_drop_probs = agent.policy(game, "dropper")
    root_check_seconds, root_check_probs = agent.policy(game, "checker")
    root_snapshot = ExactGameSnapshot(game)
    records: list[PolicyGuardRecord] = []

    drop_top = _top_seconds(tuple(root_drop_seconds), np.asarray(root_drop_probs), top_k)
    check_top = _top_seconds(tuple(root_check_seconds), np.asarray(root_check_probs), top_k)
    for drop_time, check_time in (
        (drop_second, check_second)
        for drop_second in drop_top
        for check_second in check_top
    ):
        # Probe with a forced survival outcome so branch discovery does not
        # consume the live-game RNG. The state is restored immediately.
        probe = game.resolve_half_round(drop_time, check_time, survived_outcome=True)
        death_occurred = probe.survived is not None
        survival_probability = probe.survival_probability
        root_snapshot.restore(game)

        if death_occurred:
            assert survival_probability is not None
            branches: tuple[tuple[bool | None, float], ...] = (
                (True, float(survival_probability)),
                (False, float(1.0 - survival_probability)),
            )
        else:
            branches = ((None, 1.0),)

        for survived, probability in branches:
            if probability <= 0.0:
                continue
            game.resolve_half_round(drop_time, check_time, survived_outcome=survived)
            if not game.game_over:
                records.append(
                    policy_guard_target_for_game(
                        game=game,
                        agent=agent,
                        scenario=scenario,
                        seed=seed,
                        checkpoint=checkpoint,
                        source=source,
                        config=config,
                        depth=1,
                        drop_time=drop_time,
                        check_time=check_time,
                        survival_outcome=_survival_label(survived),
                        branch_probability=probability,
                    )
                )
            root_snapshot.restore(game)

    return records


def policy_guard_records_for_game(
    *,
    game: Game,
    agent: SolverAgent,
    scenario: str,
    seed: int,
    checkpoint: str,
    source: str = SOURCE_POLICY_GUARD,
    config: ExactSearchConfig | None = None,
    include_children: bool = False,
    child_top_k: int = 2,
) -> list[PolicyGuardRecord]:
    """Label a root state and, optionally, top one-step child states."""
    config = config or ExactSearchConfig()
    records = [
        policy_guard_target_for_game(
            game=game,
            agent=agent,
            scenario=scenario,
            seed=seed,
            checkpoint=checkpoint,
            source=source,
            config=config,
        )
    ]
    if include_children:
        records.extend(
            child_policy_guard_targets_for_game(
                game=game,
                agent=agent,
                scenario=scenario,
                seed=seed,
                checkpoint=checkpoint,
                source=source,
                top_k=child_top_k,
                config=config,
            )
        )
    return records


def generate_policy_guard_records(
    *,
    checkpoint: str | Path,
    scenario_factories: dict[str, Callable[[], Game]],
    seeds: list[int],
    iterations: int,
    player: str = "Hal",
    include_children: bool = False,
    child_top_k: int = 2,
    policy_ensemble_size: int = 1,
    policy_uniform_mix: float = 0.0,
    resolve_at_critical: bool = False,
    resolve_horizon: int = 3,
) -> list[PolicyGuardRecord]:
    """Generate champion policy guard labels for named scenarios and seeds."""
    records: list[PolicyGuardRecord] = []
    checkpoint_id = str(checkpoint)
    for seed in seeds:
        agent = SolverAgent(
            checkpoint,
            player_name=player,
            iterations=iterations,
            seed=seed,
            policy_ensemble_size=policy_ensemble_size,
            policy_uniform_mix=policy_uniform_mix,
            resolve_at_critical=resolve_at_critical,
            resolve_horizon=resolve_horizon,
        )
        for name, factory in scenario_factories.items():
            game = factory()
            records.extend(
                policy_guard_records_for_game(
                    game=game,
                    agent=agent,
                    scenario=name,
                    seed=seed,
                    checkpoint=checkpoint_id,
                    include_children=include_children,
                    child_top_k=child_top_k,
                )
            )
    return records


def save_policy_guard_records(records: list[PolicyGuardRecord], path: str | Path) -> None:
    """Save targets plus sidecar metadata arrays for auditability."""
    targets = [record.target for record in records]
    save_targets(targets, path)
    if not records:
        return

    # Append metadata arrays to the existing target NPZ.
    with np.load(path, allow_pickle=True) as data:
        existing = {key: data[key] for key in data.files}
    existing.update(
        {
            "policy_guard_scenarios": np.array(
                [record.metadata.scenario for record in records]
            ),
            "policy_guard_seeds": np.array(
                [record.metadata.seed for record in records],
                dtype=np.int64,
            ),
            "policy_guard_checkpoints": np.array(
                [record.metadata.checkpoint for record in records]
            ),
            "policy_guard_iterations": np.array(
                [record.metadata.iterations for record in records],
                dtype=np.int32,
            ),
            "policy_guard_policy_ensemble_sizes": np.array(
                [record.metadata.policy_ensemble_size for record in records],
                dtype=np.int32,
            ),
            "policy_guard_policy_uniform_mixes": np.array(
                [record.metadata.policy_uniform_mix for record in records],
                dtype=np.float32,
            ),
            "policy_guard_resolve_at_critical": np.array(
                [record.metadata.resolve_at_critical for record in records],
                dtype=np.bool_,
            ),
            "policy_guard_resolve_horizons": np.array(
                [record.metadata.resolve_horizon for record in records],
                dtype=np.int32,
            ),
            "policy_guard_depths": np.array(
                [record.metadata.depth for record in records],
                dtype=np.int32,
            ),
            "policy_guard_drop_times": np.array(
                [
                    -1 if record.metadata.drop_time is None else record.metadata.drop_time
                    for record in records
                ],
                dtype=np.int16,
            ),
            "policy_guard_check_times": np.array(
                [
                    -1 if record.metadata.check_time is None else record.metadata.check_time
                    for record in records
                ],
                dtype=np.int16,
            ),
            "policy_guard_survival_outcomes": np.array(
                [record.metadata.survival_outcome for record in records]
            ),
            "policy_guard_branch_probabilities": np.array(
                [record.metadata.branch_probability for record in records],
                dtype=np.float32,
            ),
        }
    )
    np.savez(path, **existing)
