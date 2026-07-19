"""MCTS self-play and deterministic replay artifacts for ToySTL."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np

from toy.artifacts import digest_files, digest_json, sha256_file, write_npz_artifact
from toy.mcts import ToyMCTSConfig, mcts_search
from toy.network import ToyPolicyValueNet, make_network_evaluator
from toy.rules import ToyRuleset


REPLAY_SCHEMA = "toy.self_play_replay.v2"


@dataclass(frozen=True, slots=True)
class ToySelfPlayConfig:
    games: int = 4
    seed: int = 4
    action_temperature: float = 1.0
    root_noise_epsilon: float = 0.25
    root_dirichlet_alpha_scale: float = 10.0
    mcts_iterations: int = 256
    exploration_c: float = 1.0

    def __post_init__(self) -> None:
        if self.games <= 0 or self.mcts_iterations <= 0:
            raise ValueError("games and mcts_iterations must be positive")
        if self.action_temperature <= 0.0:
            raise ValueError("action_temperature must be positive")
        if not 0.0 <= self.root_noise_epsilon <= 1.0:
            raise ValueError("root_noise_epsilon must be in [0, 1]")
        if self.root_dirichlet_alpha_scale <= 0.0:
            raise ValueError("root Dirichlet alpha scale must be positive")


@dataclass(slots=True)
class ToySelfPlayResult:
    arrays: dict[str, np.ndarray]
    metadata: dict


def _seed_triplet(seed: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(seed)
    values = rng.integers(0, 2**63 - 1, size=3, dtype=np.int64)
    return tuple(int(value) for value in values)


def _full_policy(
    actions: tuple[int, ...],
    local_policy: np.ndarray,
    action_size: int,
    *,
    temperature: float,
) -> np.ndarray:
    values = np.asarray(local_policy, dtype=np.float64)
    if values.shape != (len(actions),):
        raise ValueError("MCTS policy shape does not match the root action set")
    if np.any(values < 0.0) or not np.all(np.isfinite(values)):
        raise ValueError("MCTS policy must be finite and nonnegative")
    if temperature != 1.0:
        values = np.power(values, 1.0 / temperature)
    total = float(values.sum())
    if total <= 1e-12:
        raise ValueError("MCTS policy has no probability mass")
    values = values / total
    full = np.zeros(action_size, dtype=np.float32)
    full[np.asarray(actions, dtype=np.int64) - 1] = values.astype(np.float32)
    return full


def _sample_action(
    actions: tuple[int, ...],
    policy: np.ndarray,
    rng: np.random.Generator,
) -> int:
    probabilities = np.asarray(policy, dtype=np.float64)
    total = float(probabilities.sum())
    if probabilities.shape != (len(actions),) or total <= 0.0:
        raise ValueError("invalid local action distribution")
    probabilities = probabilities / total
    return int(actions[int(rng.choice(len(actions), p=probabilities))])


def _tempered_local(policy: np.ndarray, temperature: float) -> np.ndarray:
    values = np.asarray(policy, dtype=np.float64)
    if temperature != 1.0:
        values = np.power(values, 1.0 / temperature)
    total = float(values.sum())
    if total <= 1e-12 or not np.all(np.isfinite(values)):
        raise ValueError("invalid temperature-adjusted policy")
    return values / total


def _trajectory_digest(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def play_episode(
    rules: ToyRuleset,
    model: ToyPolicyValueNet,
    *,
    episode_id: int,
    seed: int,
    config: ToySelfPlayConfig,
    checkpoint_digest: str | None = None,
    ruleset_digest: str | None = None,
) -> dict[str, np.ndarray | int | bool]:
    """Play one episode and return row-wise replay arrays."""

    root_seed, action_seed, chance_seed = _seed_triplet(seed)
    root_noise_rng = np.random.default_rng(root_seed)
    action_rng = np.random.default_rng(action_seed)
    chance_rng = np.random.default_rng(chance_seed)
    evaluator = make_network_evaluator(model, rules)
    mcts_config = ToyMCTSConfig(
        rules=rules,
        iterations=config.mcts_iterations,
        exploration_c=config.exploration_c,
        max_depth=rules.max_half_rounds,
        root_noise_epsilon=config.root_noise_epsilon,
        root_dirichlet_alpha_scale=config.root_dirichlet_alpha_scale,
    )

    state = rules.initial_state()
    rows: list[dict[str, object]] = []
    terminal_outcome: float | None = None
    truncated = False

    for step in range(rules.max_half_rounds):
        remaining_horizon = rules.max_half_rounds - step
        result = mcts_search(
            state,
            remaining_horizon,
            evaluator,
            mcts_config,
            action_rng,
            chance_rng,
            root_noise_rng,
        )
        drop_policy = _full_policy(
            result.root_drop_actions,
            result.improved_dropper_policy,
            rules.action_size,
            temperature=config.action_temperature,
        )
        check_policy = _full_policy(
            result.root_check_actions,
            result.improved_checker_policy,
            rules.action_size,
            temperature=config.action_temperature,
        )
        drop = _sample_action(
            result.root_drop_actions,
            _tempered_local(result.improved_dropper_policy, config.action_temperature),
            action_rng,
        )
        check = _sample_action(
            result.root_check_actions,
            _tempered_local(result.improved_checker_policy, config.action_temperature),
            action_rng,
        )
        rows.append(
            {
                "state": rules.state_fields(state),
                "features": rules.encode_state(state, remaining_horizon),
                "horizon": remaining_horizon,
                "drop_policy": drop_policy,
                "check_policy": check_policy,
                "drop_action": drop,
                "check_action": check,
                "episode_id": episode_id,
                "step": step,
                "root_noise_seed": root_seed,
                "action_seed": action_seed,
                "chance_seed": chance_seed,
                "episode_seed": seed,
                "checkpoint_digest": checkpoint_digest or "none",
                "ruleset_digest": ruleset_digest or "none",
            }
        )

        branches = rules.expand_joint_action(state, drop, check)
        probabilities = np.asarray([branch.probability for branch in branches], dtype=np.float64)
        branch = branches[int(chance_rng.choice(len(branches), p=probabilities / probabilities.sum()))]
        if branch.terminal_value is not None:
            terminal_outcome = float(branch.terminal_value)
            break
        assert branch.state is not None
        state = branch.state
    else:
        terminal_outcome = 0.0
        truncated = True

    if terminal_outcome is None:
        raise RuntimeError("self-play episode ended without an outcome")
    output: dict[str, np.ndarray | int | bool] = {
        "states": np.asarray([row["state"] for row in rows], dtype=np.int16),
        "features": np.asarray([row["features"] for row in rows], dtype=np.float32),
        "horizon": np.asarray([row["horizon"] for row in rows], dtype=np.int16),
        "drop_policy": np.asarray([row["drop_policy"] for row in rows], dtype=np.float32),
        "check_policy": np.asarray([row["check_policy"] for row in rows], dtype=np.float32),
        "drop_action": np.asarray([row["drop_action"] for row in rows], dtype=np.int16),
        "check_action": np.asarray([row["check_action"] for row in rows], dtype=np.int16),
        "outcome": np.full(len(rows), terminal_outcome, dtype=np.float32),
        "truncated": np.full(len(rows), truncated, dtype=np.bool_),
        "episode_id": np.asarray([row["episode_id"] for row in rows], dtype=np.int32),
        "step": np.asarray([row["step"] for row in rows], dtype=np.int16),
        "root_noise_seed": np.asarray([row["root_noise_seed"] for row in rows], dtype=np.int64),
        "action_seed": np.asarray([row["action_seed"] for row in rows], dtype=np.int64),
        "chance_seed": np.asarray([row["chance_seed"] for row in rows], dtype=np.int64),
        "episode_seed": np.asarray([row["episode_seed"] for row in rows], dtype=np.int64),
        "checkpoint_digest": np.asarray([row["checkpoint_digest"] for row in rows], dtype="U64"),
        "ruleset_digest": np.asarray([row["ruleset_digest"] for row in rows], dtype="U64"),
    }
    return {**output, "terminal_outcome": terminal_outcome, "episode_truncated": truncated}


def generate_self_play(
    rules: ToyRuleset,
    model: ToyPolicyValueNet,
    *,
    config: ToySelfPlayConfig | None = None,
    checkpoint_path: str | Path | None = None,
) -> ToySelfPlayResult:
    config = config or ToySelfPlayConfig()
    checkpoint_digest = None if checkpoint_path is None else sha256_file(checkpoint_path)
    ruleset_digest = digest_json(
        {
            "ruleset_id": rules.ruleset_id,
            "schema_version": rules.schema_version,
            "state_fields": rules.state_field_names,
            "actions": rules.action_values,
            "action_size": rules.action_size,
        }
    )
    episodes = [
        play_episode(
            rules,
            model,
            episode_id=episode_id,
            seed=config.seed + episode_id,
            config=config,
            checkpoint_digest=checkpoint_digest,
            ruleset_digest=ruleset_digest,
        )
        for episode_id in range(config.games)
    ]
    array_names = (
        "states",
        "features",
        "horizon",
        "drop_policy",
        "check_policy",
        "drop_action",
        "check_action",
        "outcome",
        "truncated",
        "episode_id",
        "step",
        "root_noise_seed",
        "action_seed",
        "chance_seed",
        "episode_seed",
        "checkpoint_digest",
        "ruleset_digest",
    )
    arrays = {
        name: np.concatenate([np.asarray(episode[name]) for episode in episodes], axis=0)
        for name in array_names
    }
    for field_index, field_name in enumerate(rules.state_field_names):
        arrays[field_name] = arrays["states"][:, field_index].copy()
    trajectory_sha256 = _trajectory_digest(arrays)
    source_root = Path(__file__).resolve().parent
    source_config_digest = digest_files(
        [source_root / name for name in ("state.py", "rules.py", "matrix.py", "mcts.py", "network.py", "self_play.py")],
        config=asdict(config),
    )
    metadata = {
        "ruleset_id": rules.ruleset_id,
        "state_schema": rules.schema_version,
        "state_field_names": list(rules.state_field_names),
        "action_values": list(rules.action_values),
        "action_size": rules.action_size,
        "max_half_rounds": rules.max_half_rounds,
        "games": config.games,
        "base_seed": config.seed,
        "self_play_config": asdict(config),
        "checkpoint_sha256": checkpoint_digest,
        "ruleset_digest": ruleset_digest,
        "trajectory_sha256": trajectory_sha256,
        "source_config_digest": source_config_digest,
        "rows": int(len(arrays["horizon"])),
    }
    return ToySelfPlayResult(arrays=arrays, metadata=metadata)


def write_self_play(
    result: ToySelfPlayResult,
    output_dir: str | Path,
) -> tuple[Path, Path, dict]:
    output_dir = Path(output_dir)
    npz_path = output_dir / "replay.npz"
    manifest_path = output_dir / "replay.json"
    manifest = write_npz_artifact(
        result.arrays,
        npz_path,
        manifest_path,
        metadata=result.metadata,
        schema_version=REPLAY_SCHEMA,
    )
    return npz_path, manifest_path, manifest
