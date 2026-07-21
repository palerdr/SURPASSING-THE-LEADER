"""Deterministically widen a trained DTH MLP without changing its function."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor

from dth.network import DTHNetworkConfig, DTHPolicyValueNet


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def widen_state_dict(
    state_dict: Mapping[str, Tensor],
    source_config: DTHNetworkConfig,
    *,
    target_width: int,
) -> tuple[dict[str, Tensor], DTHNetworkConfig]:
    """Embed a narrower ReLU MLP using trainable asymmetric duplicates.

    Each hidden neuron is copied deterministically. Its outgoing weight is
    divided among copies using unequal shares that sum to one. The network's
    function is preserved, while unequal shares give duplicate incoming
    weights different gradients on the first optimization step.
    """

    if target_width <= source_config.hidden_width:
        raise ValueError("target width must be greater than source width")
    target_config = DTHNetworkConfig(
        hidden_width=target_width,
        hidden_layers=source_config.hidden_layers,
        action_count=source_config.action_count,
        horizon_scale=source_config.horizon_scale,
    )

    source_model = DTHPolicyValueNet(source_config)
    source_model.load_state_dict(state_dict, strict=True)
    source_state = source_model.state_dict()
    target_model = DTHPolicyValueNet(target_config)
    target_state = target_model.state_dict()
    if source_state.keys() != target_state.keys():
        raise ValueError("source and target network parameters differ")

    source_width = source_config.hidden_width
    copy_from = torch.arange(target_width, dtype=torch.long) % source_width
    copy_counts = torch.bincount(copy_from, minlength=source_width)
    seen = torch.zeros(source_width, dtype=torch.long)
    shares = torch.empty(target_width, dtype=source_state["trunk.0.weight"].dtype)
    for target_index, source_index_tensor in enumerate(copy_from):
        source_index = int(source_index_tensor)
        seen[source_index] += 1
        count = int(copy_counts[source_index])
        shares[target_index] = float(seen[source_index]) / (
            count * (count + 1) / 2
        )

    widened: dict[str, Tensor] = {}
    for layer_index in range(source_config.hidden_layers):
        parameter_index = 2 * layer_index
        weight_name = f"trunk.{parameter_index}.weight"
        bias_name = f"trunk.{parameter_index}.bias"
        source_weight = source_state[weight_name]
        if layer_index == 0:
            widened[weight_name] = source_weight.index_select(0, copy_from).clone()
        else:
            widened[weight_name] = (
                source_weight.index_select(0, copy_from)
                .index_select(1, copy_from)
                .mul(shares.unsqueeze(0))
            )
        widened[bias_name] = (
            source_state[bias_name].index_select(0, copy_from).clone()
        )

    for head_name in ("value_head", "drop_head", "check_head"):
        weight_name = f"{head_name}.weight"
        bias_name = f"{head_name}.bias"
        widened[weight_name] = (
            source_state[weight_name]
            .index_select(1, copy_from)
            .mul(shares.unsqueeze(0))
        )
        widened[bias_name] = source_state[bias_name].clone()

    for name, target_tensor in target_state.items():
        if name not in widened or widened[name].shape != target_tensor.shape:
            raise ValueError(f"widening did not produce the expected tensor for {name}")

    target_model.load_state_dict(widened, strict=True)
    return widened, target_config


def widen_checkpoint(
    source: str | Path,
    destination: str | Path,
    *,
    target_width: int,
) -> dict[str, Any]:
    """Write a widened checkpoint and retain the source training metadata."""

    source_path = Path(source)
    destination_path = Path(destination)
    payload = torch.load(source_path, map_location="cpu", weights_only=False)
    if "model_config" not in payload or "state_dict" not in payload:
        raise ValueError("checkpoint must contain model_config and state_dict")
    source_config = DTHNetworkConfig(**dict(payload["model_config"]))
    state_dict, target_config = widen_state_dict(
        payload["state_dict"], source_config, target_width=target_width
    )

    widened_payload = dict(payload)
    widened_payload["state_dict"] = state_dict
    widened_payload["model_config"] = target_config.to_dict()
    widened_payload["widening"] = {
        "method": "asymmetric_neuron_duplication_v1",
        "outgoing_share_rule": "copy_rank_over_triangular_copy_count",
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": _sha256(source_path),
        "source_hidden_width": source_config.hidden_width,
        "target_hidden_width": target_width,
    }
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(widened_payload, destination_path)
    return widened_payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--target-width", required=True, type=int)
    args = parser.parse_args()
    widen_checkpoint(
        args.source,
        args.destination,
        target_width=args.target_width,
    )
    print(f"Wrote widened checkpoint to {args.destination}", flush=True)


if __name__ == "__main__":
    main()
