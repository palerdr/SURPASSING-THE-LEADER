"""Deterministic MCTS self-play and replay artifacts for DTH."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from dth.mcts import (
    AnchoredLeafEvaluator,
    Evaluator,
    ExactTargetStore,
    MCTSConfig,
    NetworkEvaluator,
    mcts_search,
)
from dth.solver import CHECKER_ACTIONS, DROPPER_ACTIONS, NTState, TState, reward, transition


REPLAY_SCHEMA = "dth-self-play-v1"


@dataclass(frozen=True)
class SelfPlayConfig:
    checkpoint: str
    output: str
    games: int = 4
    seed: int = 4
    max_half_rounds: int = 3
    device: str = "cpu"
    action_temperature: float = 1.0
    simulations: int = 64
    confidence_constant: float = 1.5
    exploration_scale: float = 1.0
    prior_uniform_mix: float = 0.05
    policy_update_interval: int = 16
    root_warmup_cells: int = 3600
    internal_warmup_cells: int = 0
    internal_warmup_horizons: list[int] | None = None
    max_depth: int | None = 1
    root_noise_epsilon: float = 0.25
    root_dirichlet_alpha_scale: float = 10.0
    starts: list[dict[str, object]] | None = None
    exact_targets: str | None = None
    anchor_horizons: list[int] | None = None

    def __post_init__(self) -> None:
        if self.games <= 0 or self.max_half_rounds <= 0:
            raise ValueError("games and max_half_rounds must be positive")
        if self.action_temperature <= 0.0:
            raise ValueError("action_temperature must be positive")
        for start in self.starts or ():
            state = tuple(int(value) for value in start["state"])
            horizon = int(start["horizon"])
            if len(state) != 4 or horizon <= 0:
                raise ValueError("each self-play start needs a four-value state and horizon")
            if not all(0 <= value <= 300 for value in state):
                raise ValueError("self-play start coordinates must be in 0..300")
        if self.anchor_horizons is not None and any(
            int(horizon) <= 0 for horizon in self.anchor_horizons
        ):
            raise ValueError("anchor horizons must be positive")
        if self.internal_warmup_horizons is not None and any(
            int(horizon) <= 0 for horizon in self.internal_warmup_horizons
        ):
            raise ValueError("internal warmup horizons must be positive")


@dataclass(frozen=True)
class SelfPlayResult:
    arrays: dict[str, np.ndarray]
    metadata: dict[str, object]


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _trajectory_digest(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _tempered(policy: np.ndarray, temperature: float) -> np.ndarray:
    values = np.asarray(policy, dtype=np.float64)
    if values.shape != (len(DROPPER_ACTIONS),):
        raise ValueError("policy must contain all 60 DTH actions")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("policy must be finite and nonnegative")
    if temperature != 1.0:
        values = np.power(values, 1.0 / temperature)
    total = float(values.sum())
    if total <= 0.0:
        raise ValueError("policy has no probability mass")
    return values / total


def _episode_seeds(seed: int) -> tuple[int, int, int, int, int]:
    rng = np.random.default_rng(seed)
    values = rng.integers(0, 2**63 - 1, size=5, dtype=np.int64)
    return tuple(int(value) for value in values)


def validate_replay(arrays: dict[str, np.ndarray]) -> dict[str, int]:
    """Reconstruct every episode and reject illegal or mislabeled rows."""

    if not np.allclose(arrays["drop_policy"].sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("invalid Dropper policy distribution")
    if not np.allclose(arrays["check_policy"].sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("invalid Checker policy distribution")
    if not np.all(np.isfinite(arrays["drop_policy"])) or not np.all(
        np.isfinite(arrays["check_policy"])
    ):
        raise ValueError("non-finite replay policy")
    if np.any(arrays["drop_policy"] < 0.0) or np.any(arrays["check_policy"] < 0.0):
        raise ValueError("negative replay policy mass")
    if not np.all((arrays["drop_action"] >= 1) & (arrays["drop_action"] <= 60)):
        raise ValueError("illegal Dropper action")
    if not np.all((arrays["check_action"] >= 1) & (arrays["check_action"] <= 60)):
        raise ValueError("illegal Checker action")

    episodes = np.unique(arrays["episode_id"])
    terminal_episodes = 0
    truncated_episodes = 0
    for episode_id in episodes:
        indices = np.flatnonzero(arrays["episode_id"] == episode_id)
        indices = indices[np.argsort(arrays["step"][indices])]
        if "initial_state" in arrays:
            initial_states = arrays["initial_state"][indices]
            if not np.all(initial_states == initial_states[0]):
                raise ValueError("episode has inconsistent initial states")
            state = tuple(int(value) for value in initial_states[0])
        else:
            state = (0, 0, 0, 0)
        chance_rng = np.random.default_rng(int(arrays["chance_seed"][indices[0]]))
        terminal_position: int | None = None
        terminal_value = 0.0
        for position, index in enumerate(indices):
            recorded_state = tuple(int(value) for value in arrays["states"][index])
            if recorded_state != state:
                raise ValueError(
                    f"unreconstructable state in episode {episode_id} step {position}"
                )
            branches = transition(
                state,
                int(arrays["drop_action"][index]),
                int(arrays["check_action"][index]),
            )
            probabilities = np.asarray([probability for probability, _ in branches])
            child = branches[int(chance_rng.choice(len(branches), p=probabilities))][1]
            if isinstance(child, TState):
                if position != len(indices) - 1:
                    raise ValueError("terminal transition is followed by replay rows")
                terminal_position = position
                terminal_value = float(reward(child))
                break
            state = child

        if terminal_position is None:
            truncated_episodes += 1
            if not np.all(arrays["truncated"][indices]):
                raise ValueError("unterminated episode lacks truncation labels")
            if not np.allclose(arrays["outcome"][indices], 0.0):
                raise ValueError("truncated episode has nonzero outcomes")
        else:
            terminal_episodes += 1
            if np.any(arrays["truncated"][indices]):
                raise ValueError("terminal episode is marked truncated")
            expected = np.asarray(
                [
                    terminal_value
                    if (terminal_position - position) % 2 == 0
                    else -terminal_value
                    for position in range(len(indices))
                ]
            )
            if not np.array_equal(arrays["outcome"][indices], expected):
                raise ValueError("terminal outcomes have incorrect role parity")

    return {
        "episodes": int(len(episodes)),
        "rows": int(len(arrays["states"])),
        "terminal_episodes": terminal_episodes,
        "truncated_episodes": truncated_episodes,
        "illegal_actions": 0,
        "invalid_distributions": 0,
        "unreconstructable_states": 0,
    }


def generate_self_play(
    config: SelfPlayConfig,
    *,
    evaluator: Evaluator | None = None,
) -> SelfPlayResult:
    checkpoint_path = Path(config.checkpoint)
    if evaluator is None:
        evaluator = NetworkEvaluator(checkpoint_path, device=config.device)
    exact_targets = (
        ExactTargetStore.load(config.exact_targets)
        if config.exact_targets is not None
        else None
    )
    anchor_horizons = (
        frozenset(int(value) for value in config.anchor_horizons)
        if config.anchor_horizons is not None
        else None
    )
    checkpoint_digest = (
        _sha256_file(checkpoint_path) if checkpoint_path.exists() else "none"
    )
    mcts_config = MCTSConfig(
        simulations=config.simulations,
        confidence_constant=config.confidence_constant,
        exploration_scale=config.exploration_scale,
        prior_uniform_mix=config.prior_uniform_mix,
        policy_update_interval=config.policy_update_interval,
        root_warmup_cells=config.root_warmup_cells,
        internal_warmup_cells=config.internal_warmup_cells,
        internal_warmup_horizons=(
            frozenset(config.internal_warmup_horizons)
            if config.internal_warmup_horizons is not None
            else None
        ),
        max_depth=config.max_depth,
        root_noise_epsilon=config.root_noise_epsilon,
        root_dirichlet_alpha_scale=config.root_dirichlet_alpha_scale,
    )

    rows: list[dict[str, object]] = []
    for episode_id in range(config.games):
        episode_seed = config.seed + episode_id
        search_seed, search_chance_seed, root_seed, action_seed, chance_seed = (
            _episode_seeds(episode_seed)
        )
        search_rng = np.random.default_rng(search_seed)
        search_chance_rng = np.random.default_rng(search_chance_seed)
        root_rng = np.random.default_rng(root_seed)
        action_rng = np.random.default_rng(action_seed)
        chance_rng = np.random.default_rng(chance_seed)
        if config.starts:
            start = config.starts[episode_id % len(config.starts)]
            state = tuple(int(value) for value in start["state"])
            episode_horizon = int(start["horizon"])
        else:
            state = (0, 0, 0, 0)
            episode_horizon = config.max_half_rounds
        initial_state = state
        episode_rows: list[dict[str, object]] = []
        terminal_step: int | None = None
        terminal_value = 0.0
        truncated = False

        for step in range(episode_horizon):
            horizon = episode_horizon - step
            search_evaluator = evaluator
            if exact_targets is not None:
                search_evaluator = AnchoredLeafEvaluator(
                    evaluator,
                    exact_targets,
                    state,
                    horizon,
                    anchor_horizons,
                )
            result = mcts_search(
                state,
                horizon,
                search_evaluator,
                mcts_config,
                search_rng,
                search_chance_rng,
                root_rng,
            )
            drop_policy = _tempered(result.drop_policy, config.action_temperature)
            check_policy = _tempered(result.check_policy, config.action_temperature)
            drop = int(action_rng.choice(DROPPER_ACTIONS, p=drop_policy))
            check = int(action_rng.choice(CHECKER_ACTIONS, p=check_policy))
            features = np.asarray(
                (*[value / 300.0 for value in state], horizon / 3.0),
                dtype=np.float32,
            )
            episode_rows.append(
                {
                    "state": state,
                    "initial_state": initial_state,
                    "features": features,
                    "horizon": horizon,
                    "drop_policy": drop_policy.astype(np.float32),
                    "check_policy": check_policy.astype(np.float32),
                    "drop_action": drop,
                    "check_action": check,
                    "episode_id": episode_id,
                    "step": step,
                    "episode_seed": episode_seed,
                    "search_seed": search_seed,
                    "search_chance_seed": search_chance_seed,
                    "root_noise_seed": root_seed,
                    "action_seed": action_seed,
                    "chance_seed": chance_seed,
                    "search_value": result.value,
                    "unique_cells": result.unique_cells,
                }
            )

            branches = transition(state, drop, check)
            probabilities = np.asarray([probability for probability, _ in branches])
            branch_index = int(chance_rng.choice(len(branches), p=probabilities))
            child = branches[branch_index][1]
            if isinstance(child, TState):
                terminal_step = step
                terminal_value = float(reward(child))
                break
            state = child
        else:
            truncated = True

        for row in episode_rows:
            if terminal_step is None:
                outcome = 0.0
            else:
                distance = terminal_step - int(row["step"])
                outcome = terminal_value if distance % 2 == 0 else -terminal_value
            row["outcome"] = outcome
            row["truncated"] = truncated
            rows.append(row)

    arrays = {
        "states": np.asarray([row["state"] for row in rows], dtype=np.int16),
        "initial_state": np.asarray(
            [row["initial_state"] for row in rows], dtype=np.int16
        ),
        "features": np.asarray([row["features"] for row in rows], dtype=np.float32),
        "horizon": np.asarray([row["horizon"] for row in rows], dtype=np.uint8),
        "drop_policy": np.asarray([row["drop_policy"] for row in rows], dtype=np.float32),
        "check_policy": np.asarray([row["check_policy"] for row in rows], dtype=np.float32),
        "drop_action": np.asarray([row["drop_action"] for row in rows], dtype=np.int16),
        "check_action": np.asarray([row["check_action"] for row in rows], dtype=np.int16),
        "outcome": np.asarray([row["outcome"] for row in rows], dtype=np.float32),
        "truncated": np.asarray([row["truncated"] for row in rows], dtype=np.bool_),
        "episode_id": np.asarray([row["episode_id"] for row in rows], dtype=np.int32),
        "step": np.asarray([row["step"] for row in rows], dtype=np.uint8),
        "episode_seed": np.asarray([row["episode_seed"] for row in rows], dtype=np.int64),
        "search_seed": np.asarray([row["search_seed"] for row in rows], dtype=np.int64),
        "search_chance_seed": np.asarray([row["search_chance_seed"] for row in rows], dtype=np.int64),
        "root_noise_seed": np.asarray([row["root_noise_seed"] for row in rows], dtype=np.int64),
        "action_seed": np.asarray([row["action_seed"] for row in rows], dtype=np.int64),
        "chance_seed": np.asarray([row["chance_seed"] for row in rows], dtype=np.int64),
        "search_value": np.asarray(
            [row["search_value"] for row in rows], dtype=np.float32
        ),
        "unique_cells": np.asarray(
            [row["unique_cells"] for row in rows], dtype=np.uint16
        ),
    }
    validation = validate_replay(arrays)
    metadata: dict[str, object] = {
        "schema_version": REPLAY_SCHEMA,
        "games": config.games,
        "rows": len(rows),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_digest,
        "config": asdict(config),
        "trajectory_sha256": _trajectory_digest(arrays),
        "state_schema": ["checker_st", "checker_ttd", "dropper_st", "dropper_ttd"],
        "action_schema": list(DROPPER_ACTIONS),
        "outcome_perspective": "current dropper; sign alternates after every live role swap",
        "validation": validation,
    }
    return SelfPlayResult(arrays, metadata)


def write_self_play(result: SelfPlayResult, output: str | Path) -> tuple[Path, Path]:
    base = Path(output)
    npz_path = base.with_suffix(".npz")
    json_path = base.with_suffix(".json")
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **result.arrays)
    manifest = {
        **result.metadata,
        "npz_sha256": _sha256_file(npz_path),
        "arrays": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in sorted(result.arrays.items())
        },
    }
    json_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return npz_path, json_path


@hydra.main(version_base="1.3", config_path="config", config_name="self_play")
def main(config: DictConfig) -> None:
    values = OmegaConf.to_container(config, resolve=True)
    if not isinstance(values, dict):
        raise TypeError("self-play config must resolve to a mapping")
    settings = SelfPlayConfig(**values)
    result = generate_self_play(settings)
    npz_path, json_path = write_self_play(result, settings.output)
    print(f"Wrote {len(result.arrays['horizon'])} rows to {npz_path}", flush=True)
    print(f"Wrote replay manifest to {json_path}", flush=True)


if __name__ == "__main__":
    main()
