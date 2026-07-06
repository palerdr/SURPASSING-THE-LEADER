#!/usr/bin/env python3
"""Interpolate two ValueNet checkpoints into a trust-region candidate."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.train_value_net import load_checkpoint


def interpolate_checkpoints(base: str | Path, target: str | Path, alpha: float, out: str | Path) -> None:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    base_model = load_checkpoint(str(base))
    target_model = load_checkpoint(str(target))
    base_state = base_model.state_dict()
    target_state = target_model.state_dict()
    if base_state.keys() != target_state.keys():
        missing = sorted(set(base_state) ^ set(target_state))
        raise ValueError(f"checkpoint keys differ: {missing[:8]}")

    mixed = {}
    for key, base_tensor in base_state.items():
        target_tensor = target_state[key]
        if base_tensor.shape != target_tensor.shape:
            raise ValueError(
                f"checkpoint tensor shape differs for {key}: "
                f"{tuple(base_tensor.shape)} != {tuple(target_tensor.shape)}"
            )
        if torch.is_floating_point(base_tensor):
            mixed[key] = (1.0 - alpha) * base_tensor + alpha * target_tensor
        else:
            mixed[key] = target_tensor.clone()

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mixed, out_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--out", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    interpolate_checkpoints(args.base, args.target, args.alpha, args.out)
    print(f"Wrote interpolated checkpoint alpha={args.alpha:g} to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
