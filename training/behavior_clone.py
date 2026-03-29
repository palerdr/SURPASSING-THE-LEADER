from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import torch

from environment.dth_env import DTHEnv
from environment.opponents.model_opponent import ModelOpponent
from environment.route_stages import current_route_stage_flags
from training.bridge_traces import BRIDGE_TRACE_SETS, BridgeTraceSpec, load_trace_file
from training.curriculum import get_scenario
from training.teacher_demos import load_teacher_demo_file


# Highest stage first so we can break early in priority order.
_STAGE_PRIORITY = ("round9_pre_leap", "round8_bridge", "round7_pressure")

QUALITY_WEIGHT_SCHEMES: dict[str, dict[str, float]] = {
    "stage_linear": {
        "none": 0.5,
        "round7_pressure": 1.0,
        "round8_bridge": 3.0,
        "round9_pre_leap": 5.0,
    },
    "stage_exp": {
        "none": 0.25,
        "round7_pressure": 1.0,
        "round8_bridge": 4.0,
        "round9_pre_leap": 16.0,
    },
    "stage_extreme": {
        "none": 0.1,
        "round7_pressure": 1.0,
        "round8_bridge": 20.0,
        "round9_pre_leap": 100.0,
    },
}


@dataclass(frozen=True)
class TraceSample:
    observation: np.ndarray
    action_mask: np.ndarray
    action_index: int


def scenario_options_for_trace(spec: BridgeTraceSpec) -> dict | None:
    if spec.scenario_name == "opening":
        return None
    return {"scenario": get_scenario(spec.scenario_name)}


def collect_trace_samples(
    trace_specs: tuple[BridgeTraceSpec, ...],
    opponent_factory,
) -> tuple[list[TraceSample], list[int]]:
    """Replay traces and collect (obs, mask, action) samples.

    Returns (samples, per_trace_counts) where per_trace_counts[i] is the
    number of samples produced by trace_specs[i].
    """
    samples: list[TraceSample] = []
    per_trace_counts: list[int] = []

    for spec in trace_specs:
        trace_start = len(samples)

        if spec.opponent_model_path is not None:
            opponent_role = "baku" if spec.agent_role == "hal" else "hal"
            opponent = ModelOpponent(spec.opponent_model_path, role=opponent_role)
        else:
            opponent = opponent_factory(spec.opponent_name)

        env = DTHEnv(
            opponent=opponent,
            agent_role=spec.agent_role,
            seed=spec.seed,
        )
        obs, _ = env.reset(options=scenario_options_for_trace(spec))

        for second in spec.actions:
            mask = env.action_masks()
            action_index = second - 1
            if action_index < 0 or action_index >= len(mask) or not mask[action_index]:
                raise ValueError(
                    f"Illegal trace action second={second} for trace={spec.name} "
                    f"scenario={spec.scenario_name}"
                )

            samples.append(
                TraceSample(
                    observation=np.asarray(obs, dtype=np.float32),
                    action_mask=np.asarray(mask, dtype=bool),
                    action_index=action_index,
                )
            )

            obs, _reward, terminated, truncated, _info = env.step(action_index)
            if terminated or truncated:
                break

        per_trace_counts.append(len(samples) - trace_start)

    return samples, per_trace_counts


def behavior_clone_policy(
    model,
    samples: list[TraceSample],
    *,
    epochs: int,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    entropy_coeff: float = 0.0,
    sample_weights: np.ndarray | None = None,
) -> None:
    if not samples or epochs <= 0:
        return

    policy = model.policy
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

    obs_array = np.stack([sample.observation for sample in samples], axis=0)
    mask_array = np.stack([sample.action_mask for sample in samples], axis=0)
    action_array = np.asarray([sample.action_index for sample in samples], dtype=np.int64)
    if sample_weights is not None:
        weight_array = np.asarray(sample_weights, dtype=np.float32)
    else:
        weight_array = None

    for _epoch in range(epochs):
        indices = np.random.permutation(len(samples))
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start:start + batch_size]
            obs_tensor, _ = policy.obs_to_tensor(obs_array[batch_indices])
            action_tensor = torch.as_tensor(action_array[batch_indices], device=policy.device)
            mask_tensor = torch.as_tensor(mask_array[batch_indices], device=policy.device)
            distribution = policy.get_distribution(obs_tensor, action_masks=mask_tensor)

            log_probs = distribution.log_prob(action_tensor)
            if weight_array is not None:
                w = torch.as_tensor(weight_array[batch_indices], device=policy.device)
                ce_loss = -(w * log_probs).mean()
            else:
                ce_loss = -log_probs.mean()

            loss = ce_loss
            if entropy_coeff > 0:
                loss = loss - entropy_coeff * distribution.entropy().mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def compute_family_weights(
    trace_specs: tuple[BridgeTraceSpec, ...],
    per_trace_counts: list[int],
    family_prefix_len: int = 8,
) -> np.ndarray:
    """Compute per-sample weights so each opening family contributes equally."""
    families = [tuple(spec.actions[:family_prefix_len]) for spec in trace_specs]
    family_counts = Counter(families)
    num_families = len(family_counts)

    sample_weights: list[float] = []
    for fam, count in zip(families, per_trace_counts):
        w = num_families / (family_counts[fam] * len(trace_specs))
        sample_weights.extend([w] * count)

    arr = np.array(sample_weights, dtype=np.float32)
    arr *= len(arr) / arr.sum()
    return arr


def assess_corpus_quality(
    trace_specs: tuple[BridgeTraceSpec, ...],
    opponent_factory,
) -> list[dict]:
    """Replay each trace and return quality metadata (highest stage, deaths)."""
    qualities: list[dict] = []
    for spec in trace_specs:
        if spec.opponent_model_path is not None:
            opponent_role = "baku" if spec.agent_role == "hal" else "hal"
            opponent = ModelOpponent(spec.opponent_model_path, role=opponent_role)
        else:
            opponent = opponent_factory(spec.opponent_name)

        env = DTHEnv(opponent=opponent, agent_role=spec.agent_role, seed=spec.seed)
        _obs, _ = env.reset(options=scenario_options_for_trace(spec))

        reached = {s: False for s in _STAGE_PRIORITY}
        flags = current_route_stage_flags(env.game)
        for s in reached:
            reached[s] = reached[s] or flags.get(s, False)

        for second in spec.actions:
            action_index = second - 1
            _obs, _reward, terminated, truncated, _info = env.step(action_index)
            flags = current_route_stage_flags(env.game)
            for s in reached:
                reached[s] = reached[s] or flags.get(s, False)
            if terminated or truncated:
                break

        highest = "none"
        for stage in _STAGE_PRIORITY:
            if reached[stage]:
                highest = stage
                break

        agent_player = env.game.player2 if spec.agent_role == "baku" else env.game.player1
        qualities.append({
            "name": spec.name,
            "highest_stage": highest,
            "trace_length": len(spec.actions),
            "agent_deaths": agent_player.deaths,
            "agent_ttd": agent_player.ttd,
        })
    return qualities


def compute_quality_weights(
    per_trace_counts: list[int],
    trace_qualities: list[dict],
    scheme: str = "stage_linear",
) -> np.ndarray:
    """Compute per-sample weights based on trace quality (stage reached)."""
    weight_map = QUALITY_WEIGHT_SCHEMES[scheme]
    sample_weights: list[float] = []
    for quality, count in zip(trace_qualities, per_trace_counts):
        w = weight_map.get(quality["highest_stage"], weight_map["none"])
        sample_weights.extend([w] * count)
    arr = np.array(sample_weights, dtype=np.float32)
    arr *= len(arr) / arr.sum()
    return arr


def load_teacher_demo_samples(paths: list[str]) -> list[TraceSample]:
    samples: list[TraceSample] = []
    for path in paths:
        for row in load_teacher_demo_file(path):
            samples.append(
                TraceSample(
                    observation=np.asarray(row.observation, dtype=np.float32),
                    action_mask=np.asarray(row.action_mask, dtype=bool),
                    action_index=int(row.action_index),
                )
            )
    return samples


def run_behavior_cloning(
    model,
    *,
    trace_set_name: str | None,
    trace_file: str | None,
    teacher_demo_files: list[str],
    opponent_factory,
    epochs: int,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    entropy_coeff: float = 0.0,
    family_balanced: bool = False,
    quality_scheme: str | None = None,
) -> int:
    trace_specs: list[BridgeTraceSpec] = []
    if trace_set_name is not None:
        if trace_set_name not in BRIDGE_TRACE_SETS:
            raise ValueError(f"Unknown bridge trace set: {trace_set_name}")
        trace_specs.extend(BRIDGE_TRACE_SETS[trace_set_name])
    if trace_file is not None:
        trace_specs.extend(load_trace_file(trace_file))
    demo_samples = load_teacher_demo_samples(teacher_demo_files)
    if not trace_specs and not demo_samples:
        raise ValueError("Behavior cloning requires a trace set, trace file, or teacher demo file")

    trace_samples, per_trace_counts = collect_trace_samples(
        tuple(trace_specs), opponent_factory,
    )
    samples = trace_samples + demo_samples

    sample_weights = None
    if (family_balanced or quality_scheme) and trace_specs:
        trace_weight = np.ones(sum(per_trace_counts), dtype=np.float32)

        if quality_scheme is not None:
            trace_qualities = assess_corpus_quality(
                tuple(trace_specs), opponent_factory,
            )
            quality_w = compute_quality_weights(
                per_trace_counts, trace_qualities, quality_scheme,
            )
            trace_weight *= quality_w

        if family_balanced:
            family_w = compute_family_weights(
                tuple(trace_specs), per_trace_counts,
            )
            trace_weight *= family_w

        trace_weight *= len(trace_weight) / trace_weight.sum()
        demo_weight = np.ones(len(demo_samples), dtype=np.float32)
        sample_weights = np.concatenate([trace_weight, demo_weight])

    behavior_clone_policy(
        model,
        samples,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        entropy_coeff=entropy_coeff,
        sample_weights=sample_weights,
    )
    return len(samples)
