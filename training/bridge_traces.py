from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class BridgeTraceSpec:
    name: str
    agent_role: str
    opponent_name: str | None
    opponent_model_path: str | None
    scenario_name: str
    actions: tuple[int, ...]
    seed: int = 42


def validate_trace_spec(spec: BridgeTraceSpec) -> None:
    if not spec.opponent_name and not spec.opponent_model_path:
        raise ValueError(f"Trace {spec.name} must define opponent_name or opponent_model_path")
    if not spec.scenario_name:
        raise ValueError(f"Trace {spec.name} must define scenario_name")
    if spec.seed < 0:
        raise ValueError(f"Trace {spec.name} must use seed >= 0")
    if not spec.actions:
        raise ValueError(f"Trace {spec.name} must contain at least one action")


def trace_spec_from_dict(data: dict) -> BridgeTraceSpec:
    spec = BridgeTraceSpec(
        name=str(data["name"]),
        agent_role=str(data["agent_role"]),
        opponent_name=data.get("opponent_name"),
        opponent_model_path=data.get("opponent_model_path"),
        scenario_name=str(data["scenario_name"]),
        seed=int(data.get("seed", 42)),
        actions=tuple(int(second) for second in data["actions"]),
    )
    validate_trace_spec(spec)
    return spec


def trace_spec_to_dict(spec: BridgeTraceSpec) -> dict:
    validate_trace_spec(spec)
    return {
        "name": spec.name,
        "agent_role": spec.agent_role,
        "opponent_name": spec.opponent_name,
        "opponent_model_path": spec.opponent_model_path,
        "scenario_name": spec.scenario_name,
        "seed": spec.seed,
        "actions": list(spec.actions),
    }


def load_trace_file(path: str) -> tuple[BridgeTraceSpec, ...]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)

    if not isinstance(raw, list):
        raise ValueError(f"Trace file must contain a JSON list: {path}")

    return tuple(trace_spec_from_dict(item) for item in raw)


def save_trace_file(path: str, specs: tuple[BridgeTraceSpec, ...]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = [trace_spec_to_dict(spec) for spec in specs]
    with open(destination, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, sort_keys=True)
        handle.write("\n")


BRIDGE_TRACE_SETS: dict[str, tuple[BridgeTraceSpec, ...]] = {
    "seed_opening_round7": (
        BridgeTraceSpec(
            name="opening_to_round7_seed",
            agent_role="baku",
            opponent_name="bridge_pressure",
            opponent_model_path=None,
            scenario_name="opening",
            seed=1990,
            actions=(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3),
        ),
    ),
    "seed_exact_bridge": (
        BridgeTraceSpec(
            name="round7_pressure_seed",
            agent_role="baku",
            opponent_name="bridge_pressure",
            opponent_model_path=None,
            scenario_name="round7_pressure",
            seed=42,
            actions=(25, 1, 25, 1, 25, 1, 8, 61),
        ),
        BridgeTraceSpec(
            name="round8_bridge_seed",
            agent_role="baku",
            opponent_name="bridge_pressure",
            opponent_model_path=None,
            scenario_name="round8_bridge",
            seed=42,
            actions=(6, 1, 6, 1, 61, 1, 6, 1),
        ),
        BridgeTraceSpec(
            name="round9_pre_leap_seed",
            agent_role="baku",
            opponent_name="leap_safe",
            opponent_model_path=None,
            scenario_name="round9_pre_leap",
            seed=42,
            actions=(8, 61),
        ),
    ),
    "seed_full_exact_bridge": (
        BridgeTraceSpec(
            name="opening_to_round7_seed",
            agent_role="baku",
            opponent_name="bridge_pressure",
            opponent_model_path=None,
            scenario_name="opening",
            seed=1990,
            actions=(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3),
        ),
        BridgeTraceSpec(
            name="round7_pressure_seed",
            agent_role="baku",
            opponent_name="bridge_pressure",
            opponent_model_path=None,
            scenario_name="round7_pressure",
            seed=42,
            actions=(25, 1, 25, 1, 25, 1, 8, 61),
        ),
        BridgeTraceSpec(
            name="round8_bridge_seed",
            agent_role="baku",
            opponent_name="bridge_pressure",
            opponent_model_path=None,
            scenario_name="round8_bridge",
            seed=42,
            actions=(6, 1, 6, 1, 61, 1, 6, 1),
        ),
        BridgeTraceSpec(
            name="round9_pre_leap_seed",
            agent_role="baku",
            opponent_name="leap_safe",
            opponent_model_path=None,
            scenario_name="round9_pre_leap",
            seed=42,
            actions=(8, 61),
        ),
    ),
    "seed_opening_heavy_exact_bridge": (
        BridgeTraceSpec(
            name="opening_to_round7_seed_1",
            agent_role="baku",
            opponent_name="bridge_pressure",
            opponent_model_path=None,
            scenario_name="opening",
            seed=1990,
            actions=(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3),
        ),
        BridgeTraceSpec(
            name="opening_to_round7_seed_2",
            agent_role="baku",
            opponent_name="bridge_pressure",
            opponent_model_path=None,
            scenario_name="opening",
            seed=1990,
            actions=(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3),
        ),
        BridgeTraceSpec(
            name="opening_to_round7_seed_3",
            agent_role="baku",
            opponent_name="bridge_pressure",
            opponent_model_path=None,
            scenario_name="opening",
            seed=1990,
            actions=(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3),
        ),
        BridgeTraceSpec(
            name="opening_to_round7_seed_4",
            agent_role="baku",
            opponent_name="bridge_pressure",
            opponent_model_path=None,
            scenario_name="opening",
            seed=1990,
            actions=(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3),
        ),
        BridgeTraceSpec(
            name="opening_to_round7_seed_5",
            agent_role="baku",
            opponent_name="bridge_pressure",
            opponent_model_path=None,
            scenario_name="opening",
            seed=1990,
            actions=(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3),
        ),
        BridgeTraceSpec(
            name="opening_to_round7_seed_6",
            agent_role="baku",
            opponent_name="bridge_pressure",
            opponent_model_path=None,
            scenario_name="opening",
            seed=1990,
            actions=(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3),
        ),
        BridgeTraceSpec(
            name="round7_pressure_seed",
            agent_role="baku",
            opponent_name="bridge_pressure",
            opponent_model_path=None,
            scenario_name="round7_pressure",
            seed=42,
            actions=(25, 1, 25, 1, 25, 1, 8, 61),
        ),
        BridgeTraceSpec(
            name="round8_bridge_seed",
            agent_role="baku",
            opponent_name="bridge_pressure",
            opponent_model_path=None,
            scenario_name="round8_bridge",
            seed=42,
            actions=(6, 1, 6, 1, 61, 1, 6, 1),
        ),
        BridgeTraceSpec(
            name="round9_pre_leap_seed",
            agent_role="baku",
            opponent_name="leap_safe",
            opponent_model_path=None,
            scenario_name="round9_pre_leap",
            seed=42,
            actions=(8, 61),
        ),
    ),
}
